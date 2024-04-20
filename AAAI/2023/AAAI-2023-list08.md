## [1400] Fully Computer-Assisted Proofs in Extremal Combinatorics

        **Authors**: *Olaf Parczyk, Sebastian Pokutta, Christoph Spiegel, Tibor Szabó*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i10.26470](https://doi.org/10.1609/aaai.v37i10.26470)

        **Abstract**:

        We present a fully computer-assisted proof system for solving a particular family of problems in Extremal Combinatorics. Existing techniques using Flag Algebras have proven powerful in the past, but have so far lacked a computational counterpart to derive matching constructive bounds. We demonstrate that common search heuristics are capable of finding constructions far beyond the reach of human intuition. Additionally, the most obvious downside of such heuristics, namely a missing guarantee of global optimality, can often be fully eliminated in this case through lower bounds and stability results coming from the Flag Algebra approach.

To illustrate the potential of this approach, we study two related and well-known problems in Extremal Graph Theory that go back to questions of Erdős from the 60s.
Most notably, we present the first major improvement in the upper bound of the Ramsey multiplicity of K_4 in 25 years, precisely determine the first off-diagonal Ramsey multiplicity number, and settle the minimum number of independent sets of size four in graphs with clique number strictly less than five.

        ----

        ## [1401] Electrophysiological Brain Source Imaging via Combinatorial Search with Provable Optimality

        **Authors**: *Guihong Wan, Meng Jiao, Xinglong Ju, Yu Zhang, Haim Schweitzer, Feng Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i10.26471](https://doi.org/10.1609/aaai.v37i10.26471)

        **Abstract**:

        Electrophysiological Source Imaging (ESI) refers to reconstructing the underlying brain source activation from non-invasive Electroencephalography (EEG) and Magnetoencephalography (MEG) measurements on the scalp. Estimating the source locations and their extents is a fundamental tool in clinical and neuroscience applications. However, the estimation is challenging because of the ill-posedness and high coherence in the leadfield matrix as well as the noise in the EEG/MEG data. In this work, we proposed a combinatorial search framework to address the ESI problem with a provable optimality guarantee. Specifically, by exploiting the graph neighborhood information in the brain source space, we converted the ESI problem into a graph search problem and designed a combinatorial search algorithm under the framework of A* to solve it. The proposed algorithm is guaranteed to give an optimal solution to the ESI problem. Experimental results on both synthetic data and real epilepsy EEG data demonstrated that the proposed algorithm could faithfully reconstruct the source activation in the brain.

        ----

        ## [1402] Improved Algorithm for Regret Ratio Minimization in Multi-Objective Submodular Maximization

        **Authors**: *Yanhao Wang, Jiping Zheng, Fanxu Meng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i10.26472](https://doi.org/10.1609/aaai.v37i10.26472)

        **Abstract**:

        Submodular maximization has attracted extensive attention due to its numerous applications in machine learning and artificial intelligence. Many real-world problems require maximizing multiple submodular objective functions at the same time. In such cases, a common approach is to select a representative subset of Pareto optimal solutions with different trade-offs among multiple objectives. To this end, in this paper, we investigate the regret ratio minimization (RRM) problem in multi-objective submodular maximization, which aims to find at most k solutions to best approximate all Pareto optimal solutions w.r.t. any linear combination of objective functions. We propose a novel HS-RRM algorithm by transforming RRM into HittingSet problems based on the notions of ε-kernel and δ-net, where any α-approximation algorithm for single-objective submodular maximization is used as an oracle. We improve upon the previous best-known bound on the maximum regret ratio (MRR) of the output of HS-RRM and show that the new bound is nearly asymptotically optimal for any fixed number d of objective functions. Experiments on real-world and synthetic data confirm that HS-RRM achieves lower MRRs than existing algorithms.

        ----

        ## [1403] Efficient Gradient Approximation Method for Constrained Bilevel Optimization

        **Authors**: *Siyuan Xu, Minghui Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i10.26473](https://doi.org/10.1609/aaai.v37i10.26473)

        **Abstract**:

        Bilevel optimization has been developed for many machine learning tasks with large-scale and high-dimensional data. This paper considers a constrained bilevel optimization problem, where the lower-level optimization problem is convex with equality and inequality constraints and the upper-level optimization problem is non-convex. The overall objective function is non-convex and non-differentiable. To solve the problem, we develop a gradient-based approach, called gradient approximation method, which determines the descent direction by computing several representative gradients of the objective function inside a neighborhood of the current estimate. We show that the algorithm asymptotically converges to the set of Clarke stationary points, and demonstrate the efficacy of the algorithm by the experiments on hyperparameter optimization and meta-learning.

        ----

        ## [1404] A Generalized Scalarization Method for Evolutionary Multi-Objective Optimization

        **Authors**: *Ruihao Zheng, Zhenkun Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i10.26474](https://doi.org/10.1609/aaai.v37i10.26474)

        **Abstract**:

        The decomposition-based multi-objective evolutionary algorithm (MOEA/D) transforms a multi-objective optimization problem (MOP) into a set of single-objective subproblems for collaborative optimization. Mismatches between subproblems and solutions can lead to severe performance degradation of MOEA/D. Most existing mismatch coping strategies only work when the L∞ scalarization is used. A mismatch coping strategy that can use any Lp scalarization, even when facing MOPs with non-convex Pareto fronts, is of great significance for MOEA/D. This paper uses the global replacement (GR) as the backbone. We analyze how GR can no longer avoid mismatches when L∞ is replaced by another Lp with p ∈ [1, ∞), and find that the Lp-based (1 ≤ p < ∞) subproblems having inconsistently large preference regions. When p is set to a small value, some middle subproblems have very small preference regions so that their direction vectors cannot pass through their corresponding preference regions. Therefore, we propose a generalized Lp (GLp) scalarization to ensure that the subproblem’s direction vector passes through its preference region. Our theoretical analysis shows that GR can always avoid mismatches when using the GLp scalarization for any p ≥ 1. The experimental studies on various MOPs conform to the theoretical analysis.

        ----

        ## [1405] Generalized Category Discovery with Decoupled Prototypical Network

        **Authors**: *Wenbin An, Feng Tian, Qinghua Zheng, Wei Ding, Qianying Wang, Ping Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26475](https://doi.org/10.1609/aaai.v37i11.26475)

        **Abstract**:

        Generalized Category Discovery (GCD) aims to recognize both known and novel categories from a set of unlabeled data, based on another dataset labeled with only known categories. Without considering differences between known and novel categories, current methods learn about them in a coupled manner, which can hurt model's generalization and discriminative ability. Furthermore, the coupled training approach prevents these models transferring category-specific knowledge explicitly from labeled data to unlabeled data, which can lose high-level semantic information and impair model performance. To mitigate above limitations, we present a novel model called Decoupled Prototypical Network (DPN). By formulating a bipartite matching problem for category prototypes, DPN can not only decouple known and novel categories to achieve different training targets effectively, but also align known categories in labeled and unlabeled data to transfer category-specific knowledge explicitly and capture high-level semantics. Furthermore, DPN can learn more discriminative features for both known and novel categories through our proposed Semantic-aware Prototypical Learning (SPL). Besides capturing meaningful semantic information, SPL can also alleviate the noise of hard pseudo labels through semantic-weighted soft assignment. Extensive experiments show that DPN outperforms state-of-the-art models by a large margin on all evaluation metrics across multiple benchmark datasets. Code and data are available at https://github.com/Lackel/DPN.

        ----

        ## [1406] Structured Case-Based Reasoning for Inference-Time Adaptation of Text-to-SQL Parsers

        **Authors**: *Abhijeet Awasthi, Soumen Chakrabarti, Sunita Sarawagi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26476](https://doi.org/10.1609/aaai.v37i11.26476)

        **Abstract**:

        Inference-time adaptation methods for semantic parsing are useful for leveraging examples from newly-observed domains without repeated fine-tuning. Existing approaches typically bias the decoder by simply concatenating input-output example pairs (cases) from the new domain at the encoder’s input in a Seq-to-Seq model. Such methods cannot adequately leverage the structure of logical forms in the case examples. We propose StructCBR, a structured case-based reasoning approach, which leverages subtree-level similarity between logical forms of cases and candidate outputs, resulting in better decoder decisions. For the task of adapting Text-to-SQL models to unseen schemas, we show that exploiting case examples in a structured manner via StructCBR offers consistent performance improvements over prior inference-time adaptation methods across five different databases. To the best of our knowledge, we are the first to attempt inference-time adaptation of Text-to-SQL models, and harness trainable structured similarity between subqueries.

        ----

        ## [1407] SegFormer: A Topic Segmentation Model with Controllable Range of Attention

        **Authors**: *Haitao Bai, Pinghui Wang, Ruofei Zhang, Zhou Su*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26477](https://doi.org/10.1609/aaai.v37i11.26477)

        **Abstract**:

        Topic segmentation aims to reveal the latent structure of a document and divide it into multiple parts. However, current neural solutions are limited in the context modeling of sentences and feature representation of candidate boundaries. This causes the model to suffer from inefficient sentence context encoding and noise information interference. In this paper, we design a new text segmentation model SegFormer with unidirectional attention blocks to better model sentence representations. To alleviate the problem of noise information interference, SegFormer uses a novel additional context aggregator and a topic classification loss to guide the model to aggregate the information within the appropriate range.  In addition, SegFormer applies an iterative prediction algorithm to search for optimal boundaries progressively. We evaluate SegFormer's generalization ability, multilingual ability, and application ability on multiple challenging real-world datasets. Experiments show that our model significantly improves the performance by 7.5% on the benchmark WIKI-SECTION compared to several strong baselines. The application of SegFormer to a real-world dataset to separate normal and advertisement segments in product marketing essays also achieves superior performance in the evaluation with other cutting-edge models.

        ----

        ## [1408] Rich Event Modeling for Script Event Prediction

        **Authors**: *Long Bai, Saiping Guan, Zixuan Li, Jiafeng Guo, Xiaolong Jin, Xueqi Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26478](https://doi.org/10.1609/aaai.v37i11.26478)

        **Abstract**:

        Script is a kind of structured knowledge extracted from texts, which contains a sequence of events. Based on such knowledge, script event prediction aims to predict the subsequent event. To do so, two aspects should be considered for events, namely, event description (i.e., what the events should contain) and event encoding (i.e., how they should be encoded). Most existing methods describe an event by a verb together with a few core arguments (i.e., subject, object, and indirect object), which are not precise enough. In addition, existing event encoders are limited to a fixed number of arguments, which are not flexible enough to deal with extra information. Thus, in this paper, we propose the Rich Event Prediction (REP) framework for script event prediction. Fundamentally, it is based on the proposed rich event description, which enriches the existing ones with three kinds of important information, namely, the senses of verbs, extra semantic roles, and types of participants. REP contains an event extractor to extract such information from texts. Based on the extracted rich information, a predictor then selects the most probable subsequent event. The core component of the predictor is a transformer-based event encoder that integrates the above information flexibly. Experimental results on the widely used Gigaword Corpus show the effectiveness of the proposed framework.

        ----

        ## [1409] Avocodo: Generative Adversarial Network for Artifact-Free Vocoder

        **Authors**: *Taejun Bak, Junmo Lee, Hanbin Bae, Jinhyeok Yang, Jae-Sung Bae, Young-Sun Joo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26479](https://doi.org/10.1609/aaai.v37i11.26479)

        **Abstract**:

        Neural vocoders based on the generative adversarial neural network (GAN) have been widely used due to their fast inference speed and lightweight networks while generating high-quality speech waveforms. Since the perceptually important speech components are primarily concentrated in the low-frequency bands, most GAN-based vocoders perform multi-scale analysis that evaluates downsampled speech waveforms. This multi-scale analysis helps the generator improve speech intelligibility. However, in preliminary experiments, we discovered that the multi-scale analysis which focuses on the low-frequency bands causes unintended artifacts, e.g., aliasing and imaging artifacts, which degrade the synthesized speech waveform quality. Therefore, in this paper, we investigate the relationship between these artifacts and GAN-based vocoders and propose a GAN-based vocoder, called Avocodo, that allows the synthesis of high-fidelity speech with reduced artifacts. We introduce two kinds of discriminators to evaluate speech waveforms in various perspectives: a collaborative multi-band discriminator and a sub-band discriminator. We also utilize a pseudo quadrature mirror filter bank to obtain downsampled multi-band speech waveforms while avoiding aliasing. According to experimental results, Avocodo outperforms baseline GAN-based vocoders, both objectively and subjectively, while reproducing speech with fewer artifacts.

        ----

        ## [1410] End-to-End Deep Reinforcement Learning for Conversation Disentanglement

        **Authors**: *Karan Bhukar, Harshit Kumar, Dinesh Raghu, Ajay Gupta*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26480](https://doi.org/10.1609/aaai.v37i11.26480)

        **Abstract**:

        Collaborative Communication platforms (e.g., Slack) support multi-party conversations which contain a large number of messages on shared channels. Multiple conversations intermingle within these messages. The task of conversation disentanglement is to cluster these intermingled messages into conversations. Existing approaches are trained using loss functions that optimize only local decisions, i.e. predicting reply-to links for each message and thereby creating clusters of conversations. In this work, we propose an end-to-end reinforcement learning (RL) approach that directly optimizes a global metric. We observe that using existing global metrics such as variation of information and adjusted rand index as a reward for the RL agent deteriorates its performance. This behaviour is because these metrics completely ignore the reply-to links between messages (local decisions) during reward computation. Therefore, we propose a novel thread-level reward function that captures the global metric without ignoring the local decisions. Through experiments on the Ubuntu IRC dataset, we demonstrate that the proposed RL model improves the performance on both link-level and conversation-level metrics.

        ----

        ## [1411] Self-Supervised Logic Induction for Explainable Fuzzy Temporal Commonsense Reasoning

        **Authors**: *Bibo Cai, Xiao Ding, Zhouhao Sun, Bing Qin, Ting Liu, Baojun Wang, Lifeng Shang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26481](https://doi.org/10.1609/aaai.v37i11.26481)

        **Abstract**:

        Understanding temporal commonsense concepts, such as times of occurrence and durations is crucial for event-centric language understanding. Reasoning about such temporal concepts in a complex context requires reasoning over both the stated context and the world knowledge that underlines it. A recent study shows massive pre-trained LM still struggle with such temporal reasoning under complex contexts (e.g., dialog) because they only implicitly encode the relevant contexts and fail to explicitly uncover the underlying logical compositions for complex inference, thus may not be robust enough. In this work, we propose to augment LMs with the temporal logic induction ability, which frames the temporal reasoning by defining three modular components: temporal dependency inducer and temporal concept defuzzifier and logic validator. The former two components disentangle the explicit/implicit dependency between temporal concepts across context (before, after, ...) and the specific meaning of fuzzy temporal concepts, respectively, while the validator combines the intermediate reasoning clues for robust contextual reasoning about the temporal concepts. Extensive experimental results on TIMEDIAL, a challenging dataset for temporal reasoning over dialog, show that our method, Logic Induction Enhanced Contextualized TEmporal Reasoning (LECTER), can yield great improvements over the traditional language model for temporal reasoning.

        ----

        ## [1412] Zero-Shot Cross-Lingual Event Argument Extraction with Language-Oriented Prefix-Tuning

        **Authors**: *Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26482](https://doi.org/10.1609/aaai.v37i11.26482)

        **Abstract**:

        Event argument extraction (EAE) aims to identify the arguments of a given event, and classify the roles that those arguments play. Due to high data demands of training EAE models, zero-shot cross-lingual EAE has attracted increasing attention, as it greatly reduces human annotation effort. Some prior works indicate that generation-based methods have achieved promising performance for monolingual EAE. However, when applying existing generation-based methods to zero-shot cross-lingual EAE, we find two critical challenges, including Language Discrepancy and Template Construction. In this paper, we propose a novel method termed as Language-oriented Prefix-tuning Network (LAPIN) to address the above challenges. Specifically, we devise a Language-oriented Prefix Generator module to handle the discrepancies between source and target languages. Moreover, we leverage a Language-agnostic Template Constructor module to design templates that can be adapted to any language. Extensive experiments demonstrate that our proposed method achieves the best performance, outperforming the previous state-of-the-art model by 4.8% and 2.3% of the average F1-score on two multilingual EAE datasets.

        ----

        ## [1413] RPA: Reasoning Path Augmentation in Iterative Retrieving for Multi-Hop QA

        **Authors**: *Ziyi Cao, Bingquan Liu, Shaobo Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26483](https://doi.org/10.1609/aaai.v37i11.26483)

        **Abstract**:

        Multi-hop questions are associated with a series of justifications, and one needs to obtain the answers by following the reasoning path (RP) that orders the justifications adequately. So reasoning path retrieval becomes a critical preliminary stage for multi-hop Question Answering (QA). Within the RP, two fundamental challenges emerge for better performance: (i) what the order of the justifications in the RP should be, and (ii) what if the wrong justification has been in the path. In this paper, we propose Reasoning Path Augmentation (RPA), which uses reasoning path reordering and augmentation to handle the above two challenges, respectively. Reasoning path reordering restructures the reasoning by targeting the easier justification first but difficult one later, in which the difficulty is determined by the overlap between query and justifications since the higher overlap means more lexical relevance and easier searchable. Reasoning path augmentation automatically generates artificial RPs, in which the distracted justifications are inserted to aid the model recover from the wrong justification. We build RPA with a naive pre-trained model and evaluate RPA on the QASC and MultiRC datasets. The evaluation results demonstrate that RPA outperforms previously published reasoning path retrieval methods, showing the effectiveness of the proposed methods. Moreover, we present detailed experiments on how the orders of justifications and the percent of augmented paths affect the question- answering performance, revealing the importance of polishing RPs and the necessity of augmentation.

        ----

        ## [1414] Leveraging Modality-Specific Representations for Audio-Visual Speech Recognition via Reinforcement Learning

        **Authors**: *Chen Chen, Yuchen Hu, Qiang Zhang, Heqing Zou, Beier Zhu, Eng Siong Chng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26484](https://doi.org/10.1609/aaai.v37i11.26484)

        **Abstract**:

        Audio-visual speech recognition (AVSR) has gained remarkable success for ameliorating the noise-robustness of speech recognition. Mainstream methods focus on fusing audio and visual inputs to obtain modality-invariant representations. However, such representations are prone to over-reliance on audio modality as it is much easier to recognize than video modality in clean conditions. As a result, the AVSR model underestimates the importance of visual stream in face of noise corruption. To this end, we leverage visual modality-specific representations to provide stable complementary information for the AVSR task. Specifically, we propose a reinforcement learning (RL) based framework called MSRL, where the agent dynamically harmonizes modality-invariant and modality-specific representations in the auto-regressive decoding process. We customize a reward function directly related to task-specific metrics (i.e., word error rate), which encourages the MSRL to effectively explore the optimal integration strategy. Experimental results on the LRS3 dataset show that the proposed method achieves state-of-the-art in both clean and various noisy conditions. Furthermore, we demonstrate the better generality of MSRL system than other baselines when test set contains unseen noises.

        ----

        ## [1415] Converge to the Truth: Factual Error Correction via Iterative Constrained Editing

        **Authors**: *Jiangjie Chen, Rui Xu, Wenxuan Zeng, Changzhi Sun, Lei Li, Yanghua Xiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26485](https://doi.org/10.1609/aaai.v37i11.26485)

        **Abstract**:

        Given a possibly false claim sentence, how can we automatically correct it with minimal editing? Existing methods either require a large number of pairs of false and corrected claims for supervised training or do not handle well errors spanning over multiple tokens within an utterance. In this paper, we propose VENCE, a novel method for factual error correction (FEC) with minimal edits. VENCE formulates the FEC problem as iterative sampling editing actions with respect to a target density function. We carefully design the target function with predicted truthfulness scores from an offline trained fact verification model. VENCE samples the most probable editing positions based on back-calculated gradients of the truthfulness score concerning input tokens and the editing actions using a distantly-supervised language model (T5). Experiments on a public dataset show that VENCE improves the well-adopted SARI metric by 5.3 (or a relative improvement of 11.8%) over the previous best distantly-supervised methods.

        ----

        ## [1416] Adversarial Word Dilution as Text Data Augmentation in Low-Resource Regime

        **Authors**: *Junfan Chen, Richong Zhang, Zheyan Luo, Chunming Hu, Yongyi Mao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26486](https://doi.org/10.1609/aaai.v37i11.26486)

        **Abstract**:

        Data augmentation is widely used in text classification, especially in the low-resource regime where a few examples for each class are available during training. Despite the success, generating data augmentations as hard positive examples that may increase their effectiveness is under-explored. This paper proposes an Adversarial Word Dilution (AWD) method that can generate hard positive examples as text data augmentations to train the low-resource text classification model efficiently. Our idea of augmenting the text data is to dilute the embedding of strong positive words by weighted mixing with unknown-word embedding, making the augmented inputs hard to be recognized as positive by the classification model. We adversarially learn the dilution weights through a constrained min-max optimization process with the guidance of the labels. Empirical studies on three benchmark datasets show that AWD can generate more effective data augmentations and outperform the state-of-the-art text data augmentation methods. The additional analysis demonstrates that the data augmentations generated by AWD are interpretable and can flexibly extend to new examples without further training.

        ----

        ## [1417] CP-Rec: Contextual Prompting for Conversational Recommender Systems

        **Authors**: *Keyu Chen, Shiliang Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26487](https://doi.org/10.1609/aaai.v37i11.26487)

        **Abstract**:

        The conversational recommender system (CRS) aims to provide high-quality recommendations through interactive dialogues. However, previous CRS models have no effective mechanisms for task planning and topic elaboration, and thus they hardly maintain coherence in multi-task recommendation dialogues. Inspired by recent advances in prompt-based learning, we propose a novel contextual prompting framework for dialogue management, which optimizes prompts based on context, topics, and user profiles. Specifically, we develop a topic controller to sequentially plan the subtasks, and a prompt search module to construct context-aware prompts. We further adopt external knowledge to enrich user profiles and make knowledge-aware recommendations. Incorporating these techniques, we propose a conversational recommender system with contextual prompting, namely CP-Rec. Experimental results demonstrate that it achieves state-of-the-art recommendation accuracy and generates more coherent and informative conversations.

        ----

        ## [1418] A Vector Quantized Approach for Text to Speech Synthesis on Real-World Spontaneous Speech

        **Authors**: *Li-Wei Chen, Shinji Watanabe, Alexander Rudnicky*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26488](https://doi.org/10.1609/aaai.v37i11.26488)

        **Abstract**:

        Recent Text-to-Speech (TTS) systems trained on reading or acted corpora have achieved near human-level naturalness. The diversity of human speech, however, often goes beyond the coverage of these corpora. We believe the ability to handle such diversity is crucial for AI systems to achieve human-level communication. Our work explores the use of more abundant real-world data for building speech synthesizers. We train TTS systems using real-world speech from YouTube and podcasts. We observe the mismatch between training and inference alignments in mel-spectrogram based autoregressive models, leading to unintelligible synthesis, and demonstrate that learned discrete codes within multiple code groups effectively resolves this issue. We introduce our MQTTS system whose architecture is designed for multiple code generation and monotonic alignment, along with the use of a clean silence prompt to improve synthesis quality. We conduct ablation analyses to identify the efficacy of our methods. We show that MQTTS outperforms existing TTS systems in several objective and subjective measures.

        ----

        ## [1419] Learning to Memorize Entailment and Discourse Relations for Persona-Consistent Dialogues

        **Authors**: *Ruijun Chen, Jin Wang, Liang-Chih Yu, Xuejie Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26489](https://doi.org/10.1609/aaai.v37i11.26489)

        **Abstract**:

        Maintaining engagement and consistency is particularly important in dialogue systems. Existing works have improved the performance of dialogue systems by intentionally learning interlocutor personas with sophisticated network structures. One issue with this approach is that it requires more personal corpora with annotations. Additionally, these models typically perform the next utterance prediction to generate a response but neglect the discourse coherence in the entire conversation. To address these issues, this study proposes a method of learning to memorize entailment and discourse relations for persona-consistent dialogue tasks. Entailment text pairs in natural language inference dataset were applied to learn latent entailment relations as external memories by premise-to-hypothesis generation task. Furthermore, an internal memory with a similar architecture was applied to the discourse information in the dialogue. Placing orthogonality restrictions on these two memory spaces ensures that the latent entailment relations remain dialogue-independent. Both memories collaborate to obtain entailment and discourse representation for the generation, allowing a deeper understanding of both consistency and coherence. Experiments on two large public datasets, PersonaChat and DSTC7-AVSD, demonstrated the effectiveness of the proposed method. Both automatic and human evaluations indicate that the proposed model outperforms several strong baselines in terms of both persona consistency and response coherence. Our source code is availabled at https://github.com/Chenrj233/LMEDR.

        ----

        ## [1420] Preference-Controlled Multi-Objective Reinforcement Learning for Conditional Text Generation

        **Authors**: *Wenqing Chen, Jidong Tian, Caoyun Fan, Yitian Li, Hao He, Yaohui Jin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26490](https://doi.org/10.1609/aaai.v37i11.26490)

        **Abstract**:

        Conditional text generation is to generate text sequences conditioning on linguistic or non-linguistic data. The main line of existing work proposed deterministic models to improve the fidelity of the generated text but often ignored the diversity. Another line relied on conditional variational auto-encoders (CVAEs), which increased the diversity over their deterministic backbones. However, CVAEs regard diversity as an implicit objective and may not be optimal. In this paper, we raise two questions:  i) Can diversity be further improved with an explicit objective? ii) Since fidelity and diversity are two conflicting objectives, how can we obtain different multi-objective optimal solutions according to user preferences? To answer question i), we propose a multi-objective reinforcement learning (MORL) method which explicitly takes CIDEr and Self-CIDEr scores as the fidelity-oriented and diversity-oriented rewards respectively. To answer question ii), we propose a preference-controlled MORL method, which can obtain infinite multi-objective optimal solutions by tuning the preference variable. We conduct extensive experiments on paraphrasing and image captioning tasks, which show that in the fidelity-diversity trade-off space, our model outperforms both deterministic and CVAE-based baselines.

        ----

        ## [1421] Learning towards Selective Data Augmentation for Dialogue Generation

        **Authors**: *Xiuying Chen, Mingzhe Li, Jiayi Zhang, Xiaoqiang Xia, Chen Wei, Jianwei Cui, Xin Gao, Xiangliang Zhang, Rui Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26491](https://doi.org/10.1609/aaai.v37i11.26491)

        **Abstract**:

        As it is cumbersome and expensive to acquire a huge amount of data for training neural dialog models, data augmentation is proposed to effectively utilize existing training samples.
However, current data augmentation techniques on the dialog generation task mostly augment all cases in the training dataset without considering the intrinsic attributes between different cases.
We argue that not all cases are beneficial for augmentation task, and the cases suitable for augmentation should obey the following two attributes: 
(1) low-quality (the dialog model cannot generate a high-quality response for the case),
(2) representative (the case should represent the property of the whole dataset).
Herein, we explore this idea by proposing a Selective Data Augmentation framework (SDA) for the response generation task.
SDA employs a dual adversarial network to select the lowest quality and most representative data points for augmentation in one stage. 
Extensive experiments conducted on two publicly available datasets, i.e., DailyDialog and OpenSubtitles, show that our framework can improve the response generation performance with respect to various metrics

        ----

        ## [1422] Learn from Yesterday: A Semi-supervised Continual Learning Method for Supervision-Limited Text-to-SQL Task Streams

        **Authors**: *Yongrui Chen, Xinnan Guo, Tongtong Wu, Guilin Qi, Yang Li, Yang Dong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26492](https://doi.org/10.1609/aaai.v37i11.26492)

        **Abstract**:

        Conventional text-to-SQL studies are limited to a single task with a fixed-size training and test set. When confronted with a stream of tasks common in real-world applications, existing methods struggle with the problems of insufficient supervised data and high retraining costs. The former tends to cause overfitting on unseen databases for the new task, while the latter makes a full review of instances from past tasks impractical for the model, resulting in forgetting of learned SQL structures and database schemas. To address the problems, this paper proposes integrating semi-supervised learning (SSL) and continual learning (CL) in a stream of text-to-SQL tasks and offers two promising solutions in turn. The first solution Vanilla is to perform self-training, augmenting the supervised training data with predicted pseudo-labeled instances of the current task, while replacing the full volume retraining with episodic memory replay to balance the training efficiency with the performance of previous tasks. The improved solution SFNet takes advantage of the intrinsic connection between CL and SSL. It uses in-memory past information to help current SSL, while adding high-quality pseudo instances in memory to improve future replay.  The experiments on two datasets shows that SFNet outperforms the widely-used SSL-only and CL-only baselines on multiple metrics.

        ----

        ## [1423] A Scope Sensitive and Result Attentive Model for Multi-Intent Spoken Language Understanding

        **Authors**: *Lizhi Cheng, Wenmian Yang, Weijia Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26493](https://doi.org/10.1609/aaai.v37i11.26493)

        **Abstract**:

        Multi-Intent Spoken Language Understanding (SLU), a novel and more complex scenario of SLU, is attracting increasing attention. Unlike traditional SLU, each intent in this scenario has its specific scope. Semantic information outside the scope even hinders the prediction, which tremendously increases the difficulty of intent detection. More seriously, guiding slot filling with these inaccurate intent labels suffers error propagation problems, resulting in unsatisfied overall performance. To solve these challenges, in this paper, we propose a novel Scope-Sensitive Result Attention Network (SSRAN) based on Transformer, which contains a Scope Recognizer (SR) and a Result Attention Network (RAN). SR assignments scope information to each token, reducing the distraction of out-of-scope tokens. RAN effectively utilizes the bidirectional interaction between SF and ID results, mitigating the error propagation problem. Experiments on two public datasets indicate that our model significantly improves SLU performance (5.4% and 2.1% on Overall accuracy) over the state-of-the-art baseline.

        ----

        ## [1424] Unsupervised Explanation Generation via Correct Instantiations

        **Authors**: *Sijie Cheng, Zhiyong Wu, Jiangjie Chen, Zhixing Li, Yang Liu, Lingpeng Kong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26494](https://doi.org/10.1609/aaai.v37i11.26494)

        **Abstract**:

        While large pre-trained language models (PLM) have shown their great skills at solving discriminative tasks, a significant gap remains when compared with humans for explanation-related tasks. Among them, explaining the reason why a statement is wrong (e.g., against commonsense) is incredibly challenging. 
The major difficulty is finding the conflict point, where the statement contradicts our real world. This paper proposes Neon, a two-phrase, unsupervised explanation generation framework. Neon first generates corrected instantiations of the statement (phase I), then uses them to prompt large PLMs to find the conflict point and complete the explanation (phase II). We conduct extensive experiments on two standard explanation benchmarks, i.e., ComVE and e-SNLI. According to both automatic and human evaluations, Neon outperforms baselines, even for those with human-annotated instantiations. In addition to explaining a negative prediction, we further demonstrate that Neon remains effective when generalizing to different scenarios. The resources of Neon are available at: https://github.com/Shark-NLP/Neon.

        ----

        ## [1425] Prompt-Augmented Linear Probing: Scaling beyond the Limit of Few-Shot In-Context Learners

        **Authors**: *Hyunsoo Cho, Hyuhng Joon Kim, Junyeob Kim, Sang-Woo Lee, Sang-goo Lee, Kang Min Yoo, Taeuk Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26495](https://doi.org/10.1609/aaai.v37i11.26495)

        **Abstract**:

        Through in-context learning (ICL), large-scale language models are effective few-shot learners without additional model fine-tuning.  However, the ICL performance does not scale well with the number of available training sample as it is limited by the inherent input length constraint of the underlying language model. Meanwhile, many studies have revealed that language models are also powerful feature extractors, allowing them to be utilized in a black-box manner and enabling the linear probing paradigm, where lightweight discriminators are trained on top of the pre-extracted input representations. This paper proposes prompt-augmented linear probing (PALP), a hybrid of linear probing and ICL, which leverages the best of both worlds. PALP inherits the scalability of linear probing and the capability of enforcing language models to derive more meaningful representations via tailoring input into a more conceivable form. Throughout in-depth investigations on various datasets, we verified that PALP significantly closes the gap between ICL in the data-hungry scenario and fine-tuning in the data-abundant scenario with little training overhead, potentially making PALP a strong alternative in a black-box scenario.

        ----

        ## [1426] Neural Dynamic Focused Topic Model

        **Authors**: *Kostadin Cvejoski, Ramsés J. Sánchez, César Ojeda*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26496](https://doi.org/10.1609/aaai.v37i11.26496)

        **Abstract**:

        Topic models and all their variants analyse text by learning meaningful representations through word co-occurrences. As pointed out by previous work, such models implicitly assume that the probability of a topic to be active and its proportion within each document are positively correlated. This correlation can be strongly detrimental in the case of documents created over time, simply because recent documents are likely better described by new and hence rare topics. In this work we leverage recent advances in neural variational inference and present an alternative neural approach to the dynamic Focused Topic Model. Indeed, we develop a neural model for topic evolution which exploits sequences of Bernoulli random variables in order to track the appearances of topics, thereby decoupling their activities from their proportions. We evaluate our model on three different datasets (the UN general debates, the collection of NeurIPS papers, and the ACL Anthology dataset) and show that it (i) outperforms state-of-the-art topic models in generalization tasks and (ii) performs comparably to them on prediction tasks, while employing roughly the same number of parameters, and converging about two times faster.

        ----

        ## [1427] Improving Simultaneous Machine Translation with Monolingual Data

        **Authors**: *Hexuan Deng, Liang Ding, Xuebo Liu, Meishan Zhang, Dacheng Tao, Min Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26497](https://doi.org/10.1609/aaai.v37i11.26497)

        **Abstract**:

        Simultaneous machine translation (SiMT) is usually done via sequence-level knowledge distillation (Seq-KD) from a full-sentence neural machine translation (NMT) model. However, there is still a significant performance gap between NMT and SiMT. In this work, we propose to leverage monolingual data to improve SiMT, which trains a SiMT student on the combination of bilingual data and external monolingual data distilled by Seq-KD. Preliminary experiments on En-Zh and En-Ja news domain corpora demonstrate that monolingual data can significantly improve translation quality (e.g., +3.15 BLEU on En-Zh). Inspired by the behavior of human simultaneous interpreters, we propose a novel monolingual sampling strategy for SiMT, considering both chunk length and monotonicity. Experimental results show that our sampling strategy consistently outperforms the random sampling strategy (and other conventional typical NMT monolingual sampling strategies) by avoiding the key problem of SiMT -- hallucination, and has better scalability. We achieve +0.72 BLEU improvements on average against random sampling on En-Zh and En-Ja. Data and codes can be found at https://github.com/hexuandeng/Mono4SiMT.

        ----

        ## [1428] Domain-Adapted Dependency Parsing for Cross-Domain Named Entity Recognition

        **Authors**: *Chenxiao Dou, Xianghui Sun, Yaoshu Wang, Yunjie Ji, Baochang Ma, Xiangang Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26498](https://doi.org/10.1609/aaai.v37i11.26498)

        **Abstract**:

        In recent years, many researchers have leveraged structural information from dependency trees to improve Named Entity Recognition (NER). Most of their methods take dependency-tree labels as input features for NER model training. However, such dependency information is not inherently provided in most NER corpora, making the methods with low usability in practice. To effectively exploit the potential of word-dependency knowledge, motivated by the success of Multi-Task Learning on cross-domain NER, we investigate a novel NER learning method incorporating cross-domain Dependency Parsing (DP) as its auxiliary learning task. Then, considering the high consistency of word-dependency relations across domains, we present an unsupervised domain-adapted method to transfer word-dependency knowledge from high-resource domains to low-resource ones. With the help of cross-domain DP to bridge different domains, both useful cross-domain and cross-task knowledge can be learned by our model to considerably benefit cross-domain NER. To make better use of the cross-task knowledge between NER and DP, we unify both tasks in a shared network architecture for joint learning, using Maximum Mean Discrepancy(MMD). Finally, through extensive experiments, we show our proposed method can not only effectively take advantage of word-dependency knowledge, but also significantly outperform other Multi-Task Learning methods on cross-domain NER. Our code is open-source and available at https://github.com/xianghuisun/DADP.

        ----

        ## [1429] MultiSpider: Towards Benchmarking Multilingual Text-to-SQL Semantic Parsing

        **Authors**: *Longxu Dou, Yan Gao, Mingyang Pan, Dingzirui Wang, Wanxiang Che, Dechen Zhan, Jian-Guang Lou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26499](https://doi.org/10.1609/aaai.v37i11.26499)

        **Abstract**:

        Text-to-SQL semantic parsing is an important NLP task, which facilitates the interaction between users and the database. Much recent progress in text-to-SQL has been driven by large-scale datasets, but most of them are centered on English. In this work, we present MultiSpider, the largest multilingual text-to-SQL semantic parsing dataset which covers seven languages (English, German, French, Spanish, Japanese, Chinese, and Vietnamese). Upon MultiSpider we further identify the lexical and structural challenges of text-to-SQL  (caused by specific language properties and dialect sayings) and their intensity across different languages. Experimental results under various settings (zero-shot, monolingual and multilingual) reveal a 6.1% absolute drop in accuracy in non-English languages. Qualitative and quantitative analyses are conducted to understand the reason for the performance drop of each language. Besides the dataset, we also propose a simple schema augmentation framework SAVe (Schema-Augmentation-with-Verification), which significantly boosts the overall performance by about 1.8% and closes the 29.5% performance gap across languages.

        ----

        ## [1430] Learning to Select from Multiple Options

        **Authors**: *Jiangshu Du, Wenpeng Yin, Congying Xia, Philip S. Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26500](https://doi.org/10.1609/aaai.v37i11.26500)

        **Abstract**:

        Many NLP tasks can be regarded as a selection problem from a set of options, such as classification tasks, multi-choice question answering, etc. Textual entailment (TE) has been shown as the state-of-the-art (SOTA) approach to dealing with those selection problems. TE treats input texts as premises (P), options as hypotheses (H), then handles the selection problem by modeling (P, H) pairwise. Two limitations: first, the pairwise modeling is unaware of other options, which is less intuitive since humans often determine the best options by comparing competing candidates; second, the inference process of pairwise TE is time-consuming, especially when the option space is large. To deal with the two issues, this work first proposes a contextualized TE model (Context-TE) by appending other k options as the context of the current (P, H) modeling. Context-TE is able to learn more reliable decision for the H since it considers various context. Second, we speed up Context-TE by coming up with Parallel-TE, which learns the decisions of multiple options simultaneously. Parallel-TE significantly improves the inference speed while keeping comparable performance with Context-TE. Our methods are evaluated on three tasks (ultra-fine entity typing, intent detection and multi-choice QA) that are typical selection problems with different sizes of options. Experiments show our models set new SOTA performance; particularly, Parallel-TE is faster than the pairwise TE by k times in inference.

        ----

        ## [1431] Real or Fake Text?: Investigating Human Ability to Detect Boundaries between Human-Written and Machine-Generated Text

        **Authors**: *Liam Dugan, Daphne Ippolito, Arun Kirubarajan, Sherry Shi, Chris Callison-Burch*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26501](https://doi.org/10.1609/aaai.v37i11.26501)

        **Abstract**:

        As text generated by large language models proliferates, it becomes vital to understand how humans engage with such text, and whether or not they are able to detect when the text they are reading did not originate with a human writer. Prior work on human detection of generated text focuses on the case where an entire passage is either human-written or machine-generated. In this paper, we study a more realistic setting where text begins as human-written and transitions to being generated by state-of-the-art neural language models. We show that, while annotators often struggle at this task, there is substantial variance in annotator skill and that given proper incentives, annotators can improve at this task over time. Furthermore, we conduct a detailed comparison study and analyze how a variety of variables (model size, decoding strategy, fine-tuning, prompt genre, etc.) affect human detection performance. Finally, we collect error annotations from our participants and use them to show that certain textual genres influence models to make different types of errors and that certain sentence-level features correlate highly with annotator selection. We release the RoFT dataset: a collection of over 21,000 human annotations paired with error classifications to encourage future work in human detection and evaluation of generated text.

        ----

        ## [1432] Diffuser: Efficient Transformers with Multi-Hop Attention Diffusion for Long Sequences

        **Authors**: *Aosong Feng, Irene Li, Yuang Jiang, Rex Ying*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26502](https://doi.org/10.1609/aaai.v37i11.26502)

        **Abstract**:

        Efficient Transformers have been developed for long sequence modeling, due to their subquadratic memory and time complexity. Sparse Transformer is a popular approach to improving the efficiency of Transformers by restricting self-attention to locations specified by the predefined sparse patterns. However, leveraging sparsity may sacrifice expressiveness compared to full-attention, when important token correlations are multiple hops away. To combine advantages of both the efficiency of sparse transformer and the expressiveness of full-attention Transformer, we propose Diffuser, a new state-of-the-art efficient Transformer. Diffuser incorporates all token interactions within one attention layer while maintaining low computation and memory costs. The key idea is to expand the receptive field of sparse attention using Attention Diffusion, which computes multi-hop token correlations based on all paths between corresponding disconnected tokens, besides attention among neighboring tokens. Theoretically, we show the expressiveness of Diffuser as a universal sequence approximator for sequence-to-sequence modeling, and investigate its ability to approximate full-attention by analyzing the graph expander property from the spectral perspective. Experimentally, we investigate the effectiveness of Diffuser with extensive evaluations, including language modeling, image modeling, and Long Range Arena (LRA).  Evaluation results show that Diffuser achieves improvements by an average of 0.94% on text classification tasks and 2.30% on LRA, with 1.67x memory savings compared to state-of-the-art benchmarks, which demonstrates superior performance of Diffuser in both expressiveness and efficiency aspects.

        ----

        ## [1433] Cogito Ergo Summ: Abstractive Summarization of Biomedical Papers via Semantic Parsing Graphs and Consistency Rewards

        **Authors**: *Giacomo Frisoni, Paolo Italiani, Stefano Salvatori, Gianluca Moro*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26503](https://doi.org/10.1609/aaai.v37i11.26503)

        **Abstract**:

        The automatic synthesis of biomedical publications catalyzes a profound research interest elicited by literature congestion. Current sequence-to-sequence models mainly rely on the lexical surface and seldom consider the deep semantic interconnections between the entities mentioned in the source document. Such superficiality translates into fabricated, poorly informative, redundant, and near-extractive summaries that severely restrict their real-world application in biomedicine, where the specialized jargon and the convoluted facts further emphasize task complexity. To fill this gap, we argue that the summarizer should acquire semantic interpretation over input, exploiting structured and unambiguous representations to capture and conserve the most relevant parts of the text content. This paper presents CogitoErgoSumm, the first framework for biomedical abstractive summarization equipping large pre-trained language models with rich semantic graphs. Precisely, we infuse graphs from two complementary semantic parsing techniques with different goals and granularities—Event Extraction and Abstract Meaning Representation, also designing a reward signal to maximize information content preservation through reinforcement learning. Extensive quantitative and qualitative evaluations on the CDSR dataset show that our solution achieves competitive performance according to multiple metrics, despite using 2.5x fewer parameters. Results and ablation studies indicate that our joint text-graph model generates more enlightening, readable, and consistent summaries. Code available at: https://github.com/disi-unibo-nlp/cogito-ergo-summ.

        ----

        ## [1434] MIGA: A Unified Multi-Task Generation Framework for Conversational Text-to-SQL

        **Authors**: *Yingwen Fu, Wenjie Ou, Zhou Yu, Yue Lin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26504](https://doi.org/10.1609/aaai.v37i11.26504)

        **Abstract**:

        Conversational text-to-SQL is designed to translate multi-turn natural language questions into their corresponding SQL queries. Most advanced conversational text-to-SQL methods are incompatible with generative pre-trained language models (PLMs), such as T5. In this paper, we present a two-stage unified MultI-task Generation frAmework (MIGA) that leverages PLMs’ ability to tackle conversational text-to-SQL. In the pre-training stage, MIGA first decomposes the main task into several related sub-tasks and then unifies them into the same sequence-to-sequence (Seq2Seq) paradigm with task-specific natural language prompts to boost the main task from multi-task training. Later in the fine-tuning stage, we propose four SQL perturbations to alleviate the error propagation problem. MIGA tends to achieve state-of-the-art performance on two benchmarks (SparC and CoSQL). We also provide extensive analyses and discussions to shed light on some new perspectives for conversational text-to-SQL.

        ----

        ## [1435] On the Effectiveness of Parameter-Efficient Fine-Tuning

        **Authors**: *Zihao Fu, Haoran Yang, Anthony Man-Cho So, Wai Lam, Lidong Bing, Nigel Collier*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26505](https://doi.org/10.1609/aaai.v37i11.26505)

        **Abstract**:

        Fine-tuning pre-trained models has been ubiquitously proven to be effective in a wide range of NLP tasks. However, fine-tuning the whole model is parameter inefficient as it always yields an entirely new model for each task. Currently, many research works propose to only fine-tune a small portion of the parameters while keeping most of the parameters shared across different tasks. These methods achieve surprisingly good performance and are shown to be more stable than their corresponding fully fine-tuned counterparts. However, such kind of methods is still not well understood. Some natural questions arise: How does the parameter sparsity lead to promising performance? Why is the model more stable than the fully fine-tuned models? How to choose the tunable parameters? In this paper, we first categorize the existing methods into random approaches, rule-based approaches, and projection-based approaches based on how they choose which parameters to tune. Then, we show that all of the methods are actually sparse fine-tuned models and conduct a novel theoretical analysis of them. We indicate that the sparsity is actually imposing a regularization on the original model by controlling the upper bound of the stability. Such stability leads to better generalization capability which has been empirically observed in a lot of recent research works. Despite the effectiveness of sparsity grounded by our theory, it still remains an open problem of how to choose the tunable parameters. Currently, the random and rule-based methods do not utilize task-specific data information while the projection-based approaches suffer from the projection discontinuity problem. To better choose the tunable parameters, we propose a novel Second-order Approximation Method (SAM) which approximates the original problem with an analytically solvable optimization function. The tunable parameters are determined by directly optimizing the approximation function. We conduct extensive experiments on several tasks. The experimental results show that our proposed SAM model outperforms many strong baseline models and it also verifies our theoretical analysis. The source code of this paper can be obtained from https://github.com/fuzihaofzh/AnalyzeParameterEff\/icientFinetune .

        ----

        ## [1436] SumREN: Summarizing Reported Speech about Events in News

        **Authors**: *Revanth Gangi Reddy, Heba Elfardy, Hou Pong Chan, Kevin Small, Heng Ji*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26506](https://doi.org/10.1609/aaai.v37i11.26506)

        **Abstract**:

        A primary objective of news articles is to establish the factual record for an event, frequently achieved by conveying both the details of the specified event (i.e., the 5 Ws; Who, What, Where, When and Why regarding the event) and how people reacted to it (i.e., reported statements). However, existing work on news summarization almost exclusively focuses on the event details. In this work, we propose the novel task of summarizing the reactions of different speakers, as expressed by their reported statements, to a given event. To this end, we create a new multi-document summarization benchmark, SumREN, comprising 745 summaries of reported statements from various public figures obtained from 633 news articles discussing 132 events. We propose an automatic silver-training data generation approach for our task, which helps smaller models like BART achieve GPT-3 level performance on this task. Finally, we introduce a pipeline-based framework for summarizing reported speech, which we empirically show to generate summaries that are more abstractive and factual than baseline query-focused summarization approaches.

        ----

        ## [1437] ProKD: An Unsupervised Prototypical Knowledge Distillation Network for Zero-Resource Cross-Lingual Named Entity Recognition

        **Authors**: *Ling Ge, Chunming Hu, Guanghui Ma, Hong Zhang, Jihong Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26507](https://doi.org/10.1609/aaai.v37i11.26507)

        **Abstract**:

        For named entity recognition (NER) in zero-resource languages, utilizing knowledge distillation methods to transfer language-independent knowledge from the rich-resource source languages to zero-resource languages is an effective means. Typically, these approaches adopt a teacher-student architecture, where the teacher network is trained in the source language, and the student network seeks to learn knowledge from the teacher network and is expected to perform well in the target language. Despite the impressive performance achieved by these methods, we argue that they have two limitations. Firstly, the teacher network fails to effectively learn language-independent knowledge shared across languages due to the differences in the feature distribution between the source and target languages. Secondly,  the student network acquires all of its knowledge from the teacher network and ignores the learning of target language-specific knowledge.
Undesirably, these limitations would hinder the model's performance in the target language. This paper proposes an unsupervised prototype knowledge distillation network (ProKD) to address these issues. Specifically, ProKD presents a contrastive learning-based prototype alignment method to achieve class feature alignment by adjusting the prototypes' distance from the source and target languages, boosting the teacher network's capacity to acquire language-independent knowledge. In addition, ProKD introduces a prototype self-training method to learn the intrinsic structure of the language by retraining the student network on the target data using samples' distance information from prototypes, thereby enhancing the student network's ability to acquire language-specific knowledge. Extensive experiments on three benchmark cross-lingual NER datasets demonstrate the effectiveness of our approach.

        ----

        ## [1438] Denoising Pre-training for Machine Translation Quality Estimation with Curriculum Learning

        **Authors**: *Xiang Geng, Yu Zhang, Jiahuan Li, Shujian Huang, Hao Yang, Shimin Tao, Yimeng Chen, Ning Xie, Jiajun Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26508](https://doi.org/10.1609/aaai.v37i11.26508)

        **Abstract**:

        Quality estimation (QE) aims to assess the quality of machine translations when reference translations are unavailable. QE plays a crucial role in many real-world applications of machine translation. Because labeled QE data are usually limited in scale, recent research, such as DirectQE, pre-trains QE models with pseudo QE data and obtains remarkable performance. However, there tends to be inevitable noise in the pseudo data, hindering models from learning QE accurately. Our study shows that the noise mainly comes from the differences between pseudo and real translation outputs. To handle this problem, we propose CLQE, a denoising pre-training framework for QE based on curriculum learning. More specifically, we propose to measure the degree of noise in the pseudo QE data with some metrics based on statistical or distributional features. With the guidance of these metrics, CLQE gradually pre-trains the QE model using data from cleaner to noisier. Experiments on various benchmarks reveal that CLQE outperforms DirectQE and other strong baselines. We also show that with our framework, pre-training converges faster than directly using the pseudo data. We make our CLQE code available (https://github.com/NJUNLP/njuqe).

        ----

        ## [1439] Generating Coherent Narratives by Learning Dynamic and Discrete Entity States with a Contrastive Framework

        **Authors**: *Jian Guan, Zhenyu Yang, Rongsheng Zhang, Zhipeng Hu, Minlie Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26509](https://doi.org/10.1609/aaai.v37i11.26509)

        **Abstract**:

        Despite advances in generating fluent texts, existing pretraining models tend to attach incoherent event sequences to involved entities when generating narratives such as stories and news. We conjecture that such issues result from representing entities as static embeddings of superficial words, while neglecting to model their ever-changing states, i.e., the information they carry, as the text unfolds. Therefore, we extend the Transformer model to dynamically conduct entity state updates and sentence realization for narrative generation. We propose a contrastive framework to learn the state representations in a discrete space, and insert additional attention layers into the decoder to better exploit these states. Experiments on two narrative datasets show that our model can generate more coherent and diverse narratives than strong baselines with the guidance of meaningful entity states.

        ----

        ## [1440] Learning to Imagine: Distillation-Based Interactive Context Exploitation for Dialogue State Tracking

        **Authors**: *Jinyu Guo, Kai Shuang, Kaihang Zhang, Yixuan Liu, Jijie Li, Zihan Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26510](https://doi.org/10.1609/aaai.v37i11.26510)

        **Abstract**:

        In dialogue state tracking (DST), the exploitation of dialogue history is a crucial research direction, and the existing DST models can be divided into two categories: full-history models and partial-history models. Since the “select first, use later” mechanism explicitly filters the distracting information being passed to the downstream state prediction, the partial-history models have recently achieved a performance advantage over the full-history models. However, besides the redundant information, some critical dialogue context information was inevitably filtered out by the partial-history models simultaneously. To reconcile the contextual consideration with avoiding the introduction of redundant information, we propose DICE-DST, a model-agnostic module widely applicable to the partial-history DST models, which aims to strengthen the ability of context exploitation for the encoder of each DST model. Specifically, we first construct a teacher encoder and devise two contextual reasoning tasks to train it to acquire extensive dialogue contextual knowledge. Then we transfer the contextual knowledge from the teacher encoder to the student encoder via a novel turn-level attention-alignment distillation. Experimental results show that our approach extensively improves the performance of partial-history DST models and thereby achieves new state-of-the-art performance on multiple mainstream datasets while keeping high efficiency.

        ----

        ## [1441] RenewNAT: Renewing Potential Translation for Non-autoregressive Transformer

        **Authors**: *Pei Guo, Yisheng Xiao, Juntao Li, Min Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26511](https://doi.org/10.1609/aaai.v37i11.26511)

        **Abstract**:

        Non-autoregressive neural machine translation (NAT) models are proposed to accelerate the inference process while maintaining relatively high performance. However, existing NAT models are difficult to achieve the desired efficiency-quality trade-off. For one thing, fully NAT models with efficient inference perform inferior to their autoregressive counterparts. For another, iterative NAT models can, though, achieve comparable performance while diminishing the advantage of speed. In this paper, we propose RenewNAT, a flexible framework with high efficiency and effectiveness, to incorporate the merits of fully and iterative NAT models. RenewNAT first generates the potential translation results and then renews them in a single pass. It can achieve significant performance improvements at the same expense as traditional NAT models (without introducing additional model parameters and decoding latency). Experimental results on various translation benchmarks (e.g., 4 WMT) show that our framework consistently improves the performance of strong fully NAT methods (e.g., GLAT and DSLP) without additional speed overhead.

        ----

        ## [1442] Instance Smoothed Contrastive Learning for Unsupervised Sentence Embedding

        **Authors**: *Hongliang He, Junlei Zhang, Zhenzhong Lan, Yue Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26512](https://doi.org/10.1609/aaai.v37i11.26512)

        **Abstract**:

        Contrastive learning-based methods, such as unsup-SimCSE, have achieved state-of-the-art (SOTA) performances in learning unsupervised sentence embeddings. However, in previous studies, each embedding used for contrastive learning only
derived from one sentence instance, and we call these embeddings instance-level embeddings. In other words, each embedding is regarded as a unique class of its own, which may hurt the generalization performance. In this study, we propose IS-CSE (instance smoothing contrastive sentence embedding) to smooth the boundaries of embeddings in the feature space. Specifically, we retrieve embeddings from a dynamic memory buffer according to the semantic similarity to get a positive embedding group. Then embeddings in the group are aggregated by a self-attention operation to produce a smoothed instance embedding for further analysis. We evaluate our method on standard semantic text similarity (STS) tasks and achieve an average of 78.30%, 79.47%, 77.73%, and 79.42% Spearman’s correlation on the base of BERT-base, BERT-large, RoBERTa-base, and RoBERTa-large respectively, a 2.05%, 1.06%, 1.16% and 0.52% improvement compared to unsup-SimCSE.

        ----

        ## [1443] Competition or Cooperation? Exploring Unlabeled Data via Challenging Minimax Game for Semi-supervised Relation Extraction

        **Authors**: *Yu Hong, Jiahang Li, Jianchuan Feng, Chenghua Huang, Zhixu Li, Jianfeng Qu, Yanghua Xiao, Wei Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26513](https://doi.org/10.1609/aaai.v37i11.26513)

        **Abstract**:

        Semi-Supervised Relation Extraction aims at learning well-performed RE models with limited labeled and large-scale unlabeled data. Existing methods mainly suffer from semantic drift and insufficient supervision, which severely limit the performance. To address these problems, recent work tends to design dual modules to work cooperatively for mutual enhancement. However, the consensus of two modules greatly restricts the model from exploring diverse relation expressions in unlabeled set, which hinders the performance as well as model generalization. To tackle this problem, in this paper, we propose a novel competition-based method AdvSRE. We set up a challenging minimax game on unlabeled data between two modules, Generator and Discriminator, and assign them with conflicting objectives. During the competition game, one module may find any possible chance to beat the other, which develops two modules' abilities until relation expressions cannot be further explored. To exploit label information, Discriminator is further asked to predict specific relation for each sentence. Experiment results on two benchmarks show new state-of-the-art performance over baselines, demonstrating the effectiveness of proposed AdvSRE.

        ----

        ## [1444] Feature Normalization and Cartography-Based Demonstrations for Prompt-Based Fine-Tuning on Emotion-Related Tasks

        **Authors**: *Mahshid Hosseini, Cornelia Caragea*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26514](https://doi.org/10.1609/aaai.v37i11.26514)

        **Abstract**:

        To train a model in a traditional supervised learning classification system for natural language processing (NLP) tasks, it is essential to have labeled data, which is not present in large amounts for many tasks. Prompt-based learning methods attempt to combat the supervised learning need for labeled data by directly adapting pre-trained language models and modeling the probability of text itself. In this paper, we propose a novel data-agnostic strategy for prompt-based fine-tuning that leverages feature moments (a.k.a., mean and standard deviation) as a data augmentation technique and employs training dynamics (i.e., confidence and variability) to allow more informative samples to be concatenated for generating demonstrations as input context. Our approach is a strong method for few-shot learning that forces the language model to pay special attention to the feature moments and allows more informative samples to be concatenated for generating demonstrations as input context by selecting high confidence and low variance samples. To demonstrate its effectiveness given limited training data, we conduct extensive experiments in different few-shot settings on three empathy and emotion classification datasets (from various domains).  We further evaluate our method's robustness by introducing noise to our few-shot input data and labels and show that exchanging moments between samples and incorporating cartography-based demonstrations are beneficial when the available data is limited and noisy.

        ----

        ## [1445] A Simple Yet Effective Subsequence-Enhanced Approach for Cross-Domain NER

        **Authors**: *Jinpeng Hu, Dandan Guo, Yang Liu, Zhuo Li, Zhihong Chen, Xiang Wan, Tsung-Hui Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26515](https://doi.org/10.1609/aaai.v37i11.26515)

        **Abstract**:

        Cross-domain named entity recognition (NER), aiming to address the limitation of labeled resources in the target domain, is a challenging yet important task. Most existing studies alleviate the data discrepancy across different domains at the coarse level via combing NER with language modelings or introducing domain-adaptive pre-training (DAPT). Notably, source and target domains tend to share more fine-grained local information within denser subsequences than global information within the whole sequence, such that subsequence features are easier to transfer, which has not been explored well. Besides, compared to token-level representation, subsequence-level information can help the model distinguish different meanings of the same word in different domains. In this paper, we propose to incorporate subsequence-level features for promoting the cross-domain NER. In detail, we first utilize a pre-trained encoder to extract the global information. Then, we re-express each sentence as a group of subsequences and propose a novel bidirectional memory recurrent unit (BMRU) to capture features from the subsequences. Finally, an adaptive coupling unit (ACU) is proposed to combine global information and subsequence features for predicting entity labels. Experimental results on several benchmark datasets illustrate the effectiveness of our model, which achieves considerable improvements.

        ----

        ## [1446] A Question-Answering Approach to Key Value Pair Extraction from Form-Like Document Images

        **Authors**: *Kai Hu, Zhuoyuan Wu, Zhuoyao Zhong, Weihong Lin, Lei Sun, Qiang Huo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26516](https://doi.org/10.1609/aaai.v37i11.26516)

        **Abstract**:

        In this paper, we present a new question-answering (QA) based key-value pair extraction approach, called KVPFormer, to robustly extracting key-value relationships between entities from form-like document images. Specifically, KVPFormer first identifies key entities from all entities in an image with a Transformer encoder, then takes these key entities as questions and feeds them into a Transformer decoder to predict their corresponding answers (i.e., value entities) in parallel. To achieve higher answer prediction accuracy, we propose a coarse-to-fine answer prediction approach further, which first extracts multiple answer candidates for each identified question in the coarse stage and then selects the most likely one among these candidates in the fine stage. In this way, the learning difficulty of answer prediction can be effectively reduced so that the prediction accuracy can be improved. Moreover, we introduce a spatial compatibility attention bias into the self-attention/cross-attention mechanism for KVPFormer to better model the spatial interactions between entities. With these new techniques, our proposed KVPFormer achieves state-of-the-art results on FUNSD and XFUND datasets, outperforming the previous best-performing method by 7.2% and 13.2% in F1 score, respectively.

        ----

        ## [1447] SEAT: Stable and Explainable Attention

        **Authors**: *Lijie Hu, Yixin Liu, Ninghao Liu, Mengdi Huai, Lichao Sun, Di Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26517](https://doi.org/10.1609/aaai.v37i11.26517)

        **Abstract**:

        Attention mechanism has become a standard fixture in many state-of-the-art natural language processing (NLP) models, not only due to its outstanding performance, but also because it provides plausible innate explanations for neural architectures. However, recent studies show that attention is unstable against randomness and perturbations during training or testing, such as random seeds and slight perturbation of embeddings, which impedes it from being a faithful explanation tool. Thus, a natural question is whether we can find an alternative to vanilla attention, which is more stable and could keep the key characteristics of the explanation. In this paper, we provide a rigorous definition of such an attention method named SEAT (Stable and Explainable ATtention). Specifically, SEAT has the following three properties: (1) Its prediction distribution is close to the prediction of the vanilla attention; (2) Its top-k indices largely overlap with those of the vanilla attention; (3) It is robust w.r.t perturbations, i.e., any slight perturbation on SEAT will not change the attention and prediction distribution too much, which implicitly indicates that it is stable to randomness and perturbations. Furthermore, we propose an optimization method for obtaining SEAT, which could be considered as revising the vanilla attention. Finally, through intensive experiments on various datasets, we compare our SEAT with other baseline methods using RNN, BiLSTM and BERT architectures, with different evaluation metrics on model interpretation, stability and accuracy. Results show that, besides preserving the original explainability and model performance, SEAT is more stable against input perturbations and training randomness, which indicates it is a more faithful explanation.

        ----

        ## [1448] Personalized Dialogue Generation with Persona-Adaptive Attention

        **Authors**: *Qiushi Huang, Yu Zhang, Tom Ko, Xubo Liu, Bo Wu, Wenwu Wang, H. Lilian Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26518](https://doi.org/10.1609/aaai.v37i11.26518)

        **Abstract**:

        Persona-based dialogue systems aim to generate consistent responses based on historical context and predefined persona. Unlike conventional dialogue generation, the persona-based dialogue needs to consider both dialogue context and persona, posing a challenge for coherent training. Specifically, this requires a delicate weight balance between context and persona. To achieve that, in this paper, we propose an effective framework with Persona-Adaptive Attention (PAA), which adaptively integrates the weights from the persona and context information via our designed attention. In addition, a dynamic masking mechanism is applied to the PAA to not only drop redundant information in context and persona but also serve as a regularization mechanism to avoid overfitting. Experimental results demonstrate the superiority of the proposed PAA framework compared to the strong baselines in both automatic and human evaluation. Moreover, the proposed PAA approach can perform equivalently well in a low-resource regime compared to models trained in a full-data setting, which achieve a similar result with only 20% to 30% of data compared to the larger models trained in the full-data setting. To fully exploit the effectiveness of our design, we designed several variants for handling the weighted information in different ways, showing the necessity and sufficiency of our weighting and masking designs.

        ----

        ## [1449] Question Decomposition Tree for Answering Complex Questions over Knowledge Bases

        **Authors**: *Xiang Huang, Sitao Cheng, Yiheng Shu, Yuheng Bao, Yuzhong Qu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26519](https://doi.org/10.1609/aaai.v37i11.26519)

        **Abstract**:

        Knowledge base question answering (KBQA) has attracted a lot of interest in recent years, especially for complex questions which require multiple facts to answer. Question decomposition is a promising way to answer complex questions. Existing decomposition methods split the question into sub-questions according to a single compositionality type, which is not sufficient for questions involving multiple compositionality types. In this paper, we propose Question Decomposition Tree (QDT) to represent the structure of complex questions. Inspired by recent advances in natural language generation (NLG), we present a two-staged method called Clue-Decipher to generate QDT. It can leverage the strong ability of NLG model and simultaneously preserve the original questions. To verify that QDT can enhance KBQA task, we design a decomposition-based KBQA system called QDTQA. Extensive experiments show that QDTQA outperforms previous state-of-the-art methods on ComplexWebQuestions dataset. Besides, our decomposition method improves an existing KBQA system by 12% and sets a new state-of-the-art on LC-QuAD 1.0.

        ----

        ## [1450] Hierarchical Text Classification as Sub-hierarchy Sequence Generation

        **Authors**: *Sanghun Im, Gibaeg Kim, Heung-Seon Oh, Seongung Jo, Donghwan Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26520](https://doi.org/10.1609/aaai.v37i11.26520)

        **Abstract**:

        Hierarchical text classification (HTC) is essential for various real applications. However, HTC models are challenging to develop because they often require processing a large volume of documents and labels with hierarchical taxonomy. Recent HTC models based on deep learning have attempted to incorporate hierarchy information into a model structure. Consequently, these models are challenging to implement when the model parameters increase for a large-scale hierarchy because the model structure depends on the hierarchy size. To solve this problem, we formulate HTC as a sub-hierarchy sequence generation to incorporate hierarchy information into a target label sequence instead of the model structure. Subsequently, we propose the Hierarchy DECoder (HiDEC), which decodes a text sequence into a sub-hierarchy sequence using recursive hierarchy decoding, classifying all parents at the same level into children at once. In addition, HiDEC is trained to use hierarchical path information from a root to each leaf in a sub-hierarchy composed of the labels of a target document via an attention mechanism and hierarchy-aware masking. HiDEC achieved state-of-the-art performance with significantly fewer model parameters than existing models on benchmark datasets, such as RCV1-v2, NYT, and EURLEX57K.

        ----

        ## [1451] IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian Languages

        **Authors**: *Tahir Javed, Kaushal Santosh Bhogale, Abhigyan Raman, Pratyush Kumar, Anoop Kunchukuttan, Mitesh M. Khapra*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26521](https://doi.org/10.1609/aaai.v37i11.26521)

        **Abstract**:

        A cornerstone in AI research has been the creation and adoption of standardized training and test datasets to earmark the progress of state-of-the-art models. A particularly successful example is the GLUE dataset for training and evaluating Natural Language Understanding (NLU) models for English. The large body of research around self-supervised BERT-based language models revolved around performance improvements on NLU tasks in GLUE. To evaluate language models in other languages, several language-specific GLUE datasets were created. The area of speech language understanding (SLU) has followed a similar trajectory. The success of large self-supervised models such as wav2vec2 enable creation of speech models with relatively easy to access unlabelled data. These models can then be evaluated on SLU tasks, such as the SUPERB benchmark. In this work, we extend this to Indic languages by releasing the IndicSUPERB benchmark. Specifically, we make the following three contributions. (i) We collect Kathbath containing 1,684 hours of labelled speech data across 12 Indian languages from 1,218 contributors located in 203 districts in India. (ii) Using Kathbath, we create benchmarks across 6 speech tasks: Automatic Speech Recognition, Speaker Verification, Speaker Identification (mono/multi), Language Identification, Query By Example, and Keyword Spotting for 12 languages. (iii) On the released benchmarks, we train and evaluate different self-supervised models alongside the a commonly used baseline FBANK. We show that language-specific fine-tuned models are more accurate than baseline on most of the tasks, including a large gap of 76% for Language Identification task. However, for speaker identification, self-supervised models trained on large datasets demonstrate an advantage. We hope IndicSUPERB contributes to the progress of developing speech language understanding models for Indian languages.

        ----

        ## [1452] SheetPT: Spreadsheet Pre-training Based on Hierarchical Attention Network

        **Authors**: *Ran Jia, Qiyu Li, Zihan Xu, Xiaoyuan Jin, Lun Du, Haoyu Dong, Xiao Lv, Shi Han, Dongmei Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26522](https://doi.org/10.1609/aaai.v37i11.26522)

        **Abstract**:

        Spreadsheets are an important and unique type of business document for data storage, analysis and presentation. The distinction between spreadsheets and most other types of digital documents lies in that spreadsheets provide users with high flexibility of data organization on the grid. Existing related techniques mainly focus on the tabular data and are incompetent in understanding the entire sheet. On the one hand, spreadsheets have no explicit separation across tabular data and other information, leaving a gap for the deployment of such techniques. On the other hand, pervasive data dependence and semantic relations across the sheet require comprehensive modeling of all the information rather than only the tables. In this paper, we propose SheetPT, the first pre-training technique on spreadsheets to enable effective representation learning under this scenario. For computational effectiveness and efficiency, we propose the coherent chunk, an intermediate semantic unit of sheet structure; and we accordingly devise a hierarchical attention-based architecture to capture contextual information across different structural granularities. Three pre-training objectives are also designed to ensure sufficient training against millions of spreadsheets. Two representative downstream tasks, formula prediction and sheet structure recognition are utilized to evaluate its capability and the prominent results reveal its superiority over existing state-of-the-art methods.

        ----

        ## [1453] SeDepTTS: Enhancing the Naturalness via Semantic Dependency and Local Convolution for Text-to-Speech Synthesis

        **Authors**: *Chenglong Jiang, Ying Gao, Wing W. Y. Ng, Jiyong Zhou, Jinghui Zhong, Hongzhong Zhen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26523](https://doi.org/10.1609/aaai.v37i11.26523)

        **Abstract**:

        Self-attention-based networks have obtained impressive performance in parallel training and global context modeling. However, it is weak in local dependency capturing, especially for data with strong local correlations such as utterances. Therefore, we will mine linguistic information of the original text based on a semantic dependency and the semantic relationship between nodes is regarded as prior knowledge to revise the distribution of self-attention. On the other hand, given the strong correlation between input characters, we introduce a one-dimensional (1-D) convolution neural network (CNN) producing query(Q) and value(V) in the self-attention mechanism for a better fusion of local contextual information. Then, we migrate this variant of the self-attention networks to speech synthesis tasks and propose a non-autoregressive (NAR) neural Text-to-Speech (TTS): SeDepTTS. Experimental results show that our model yields good performance in speech synthesis. Specifically, the proposed method yields significant improvement for the processing of pause, stress, and intonation in speech.

        ----

        ## [1454] Prototypical Fine-Tuning: Towards Robust Performance under Varying Data Sizes

        **Authors**: *Yiqiao Jin, Xiting Wang, Yaru Hao, Yizhou Sun, Xing Xie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26524](https://doi.org/10.1609/aaai.v37i11.26524)

        **Abstract**:

        In this paper, we move towards combining large parametric models with non-parametric prototypical networks. We propose prototypical fine-tuning, a novel prototypical framework for fine-tuning pretrained language models (LM), which automatically learns a bias to improve predictive performance for varying data sizes, especially low-resource settings. Our prototypical fine-tuning approach can automatically adjust the model capacity according to the number of data points and the model's inherent attributes. Moreover, we propose four principles for effective prototype fine-tuning towards the optimal solution. Experimental results across various datasets show that our work achieves significant performance improvements under various low-resource settings, as well as comparable and usually better performances in high-resource scenarios.

        ----

        ## [1455] Cross-Modal Distillation for Speaker Recognition

        **Authors**: *Yufeng Jin, Guosheng Hu, Haonan Chen, Duoqian Miao, Liang Hu, Cairong Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26525](https://doi.org/10.1609/aaai.v37i11.26525)

        **Abstract**:

        Speaker recognition achieved great progress recently, however, it is not easy or efficient to further improve its performance via traditional solutions: collecting more data and designing new neural networks. Aiming at the fundamental challenge of speech data, i.e. low information density, multimodal learning can mitigate this challenge by introducing richer and more discriminative information as input for identity recognition. Specifically, since the face image is more discriminative than the speech for identity recognition, we conduct multimodal learning by introducing a face recognition model (teacher) to transfer discriminative knowledge to a speaker recognition model (student) during training. However, this knowledge transfer via distillation is not trivial because the big domain gap between face and speech can easily lead to overfitting. In this work, we introduce a multimodal learning framework, VGSR (Vision-Guided Speaker Recognition). Specifically, we propose a MKD (Margin-based Knowledge Distillation) strategy for cross-modality distillation by introducing a loose constrain to align the teacher and student, greatly reducing overfitting. Our MKD strategy can easily adapt to various existing knowledge distillation methods. In addition, we propose a QAW (Quality-based Adaptive Weights) module to weight input samples via quantified data quality, leading to a robust model training. Experimental results on the VoxCeleb1 and CN-Celeb datasets show our proposed strategies can effectively improve the accuracy of speaker recognition by a margin of 10% ∼ 15%, and our methods are very robust to different noises.

        ----

        ## [1456] Explaining (Sarcastic) Utterances to Enhance Affect Understanding in Multimodal Dialogues

        **Authors**: *Shivani Kumar, Ishani Mondal, Md. Shad Akhtar, Tanmoy Chakraborty*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26526](https://doi.org/10.1609/aaai.v37i11.26526)

        **Abstract**:

        Conversations emerge as the primary media for exchanging ideas and conceptions. From the listener’s perspective, identifying various affective qualities, such as sarcasm, humour, and emotions, is paramount for comprehending the true connotation of the emitted utterance. However, one of the major hurdles faced in learning these affect dimensions is the presence of figurative language, viz. irony, metaphor, or sarcasm. We hypothesize that any detection system constituting the exhaustive and explicit presentation of the emitted utterance would improve the overall comprehension of the dialogue. To this end, we explore the task of Sarcasm Explanation in Dialogues, which aims to unfold the hidden irony behind sarcastic utterances. We propose MOSES, a deep neural network which takes a multimodal (sarcastic) dialogue instance as an input and generates a natural language sentence as its explanation. Subsequently, we leverage the generated explanation for various natural language understanding tasks in a conversational dialogue setup, such as sarcasm detection, humour identification, and emotion recognition. Our evaluation shows that MOSES outperforms the state-of-the-art system for SED by an average of ∼2% on different evaluation metrics, such as ROUGE, BLEU, and METEOR. Further, we observe that leveraging the generated explanation advances three downstream tasks for affect classification – an average improvement of ~14% F1-score in the sarcasm detection task and ∼2% in the humour identification and emotion recognition task. We also perform extensive analyses to assess the quality of the results.

        ----

        ## [1457] COCA: COllaborative CAusal Regularization for Audio-Visual Question Answering

        **Authors**: *Mingrui Lao, Nan Pu, Yu Liu, Kai He, Erwin M. Bakker, Michael S. Lew*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26527](https://doi.org/10.1609/aaai.v37i11.26527)

        **Abstract**:

        Audio-Visual Question Answering (AVQA) is a sophisticated QA task, which aims at answering textual questions over given video-audio pairs with comprehensive multimodal reasoning. Through detailed causal-graph analyses and careful inspections of their learning processes, we reveal that AVQA models are not only prone to over-exploit prevalent language bias, but also suffer from additional joint-modal biases caused by the shortcut relations between textual-auditory/visual co-occurrences and dominated answers. In this paper, we propose a COllabrative CAusal (COCA) Regularization to remedy this more challenging issue of data biases.
Specifically, a novel Bias-centered Causal Regularization (BCR) is proposed to alleviate specific shortcut biases by intervening bias-irrelevant causal effects, and further introspect the predictions of AVQA models in counterfactual and factual scenarios. Based on the fact that the dominated bias impairing model robustness for different samples tends to be different, we introduce a Multi-shortcut Collaborative Debiasing (MCD) to measure how each sample suffers from different biases, and dynamically adjust their debiasing concentration to different shortcut correlations. Extensive experiments demonstrate the effectiveness as well as backbone-agnostic ability of our COCA strategy, and it achieves state-of-the-art performance on the large-scale MUSIC-AVQA dataset.

        ----

        ## [1458] Script, Language, and Labels: Overcoming Three Discrepancies for Low-Resource Language Specialization

        **Authors**: *Jaeseong Lee, Dohyeon Lee, Seung-won Hwang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26528](https://doi.org/10.1609/aaai.v37i11.26528)

        **Abstract**:

        Although multilingual pretrained models (mPLMs) enabled support of various natural language processing in diverse languages, its limited coverage of 100+ languages lets 6500+ languages remain ‘unseen’. One common approach for an unseen language is specializing the model for it as target, by performing additional masked language modeling (MLM) with the target language corpus. However, we argue that, due to the discrepancy from multilingual MLM pretraining, a naive specialization as such can be suboptimal. Specifically, we pose three discrepancies to overcome. Script and linguistic discrepancy of the target language from the related seen languages, hinder a positive transfer, for which we propose to maximize representation similarity, unlike existing approaches maximizing overlaps. In addition, label space for MLM prediction can vary across languages, for which we propose to reinitialize top layers for a more effective adaptation. Experiments over four different language families and three tasks shows that our method improves the task performance of unseen languages with statistical significance, while previous approach fails to.

        ----

        ## [1459] LIQUID: A Framework for List Question Answering Dataset Generation

        **Authors**: *Seongyun Lee, Hyunjae Kim, Jaewoo Kang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26529](https://doi.org/10.1609/aaai.v37i11.26529)

        **Abstract**:

        Question answering (QA) models often rely on large-scale training datasets, which necessitates the development of a data generation framework to reduce the cost of manual annotations. Although several recent studies have aimed to generate synthetic questions with single-span answers, no study has been conducted on the creation of list questions with multiple, non-contiguous spans as answers. To address this gap, we propose LIQUID, an automated framework for generating list QA datasets from unlabeled corpora. We first convert a passage from Wikipedia or PubMed into a summary and extract named entities from the summarized text as candidate answers. This allows us to select answers that are semantically correlated in context and is, therefore, suitable for constructing list questions. We then create questions using an off-the-shelf question generator with the extracted entities and original passage. Finally, iterative filtering and answer expansion are performed to ensure the accuracy and completeness of the answers. Using our synthetic data, we significantly improve the performance of the previous best list QA models by exact-match F1 scores of 5.0 on MultiSpanQA, 1.9 on Quoref, and 2.8 averaged across three BioASQ benchmarks.

        ----

        ## [1460] UniSyn: An End-to-End Unified Model for Text-to-Speech and Singing Voice Synthesis

        **Authors**: *Yi Lei, Shan Yang, Xinsheng Wang, Qicong Xie, Jixun Yao, Lei Xie, Dan Su*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26530](https://doi.org/10.1609/aaai.v37i11.26530)

        **Abstract**:

        Text-to-speech (TTS) and singing voice synthesis (SVS) aim at generating high-quality speaking and singing voice according to textual input and music scores, respectively. Unifying TTS and SVS into a single system is crucial to the applications requiring both of them. Existing methods usually suffer from some limitations, which rely on either both singing and speaking data from the same person or cascaded models of multiple tasks. To address these problems, a simplified elegant framework for TTS and SVS, named UniSyn, is proposed in this paper. It is an end-to-end unified model that can make a voice speak and sing with only singing or speaking data from this person. To be specific, a multi-conditional variational autoencoder (MC-VAE), which constructs two independent latent sub-spaces with the speaker- and style-related (i.e. speak or sing) conditions for flexible control, is proposed in UniSyn. Moreover, supervised guided-VAE and timbre perturbation with the Wasserstein distance constraint are leveraged to further disentangle the speaker timbre and style. Experiments conducted on two speakers and two singers demonstrate that UniSyn can generate natural speaking and singing voice without corresponding training data. The proposed approach outperforms the state-of-the-art end-to-end voice generation work, which proves the effectiveness and advantages of UniSyn.

        ----

        ## [1461] SoftCorrect: Error Correction with Soft Detection for Automatic Speech Recognition

        **Authors**: *Yichong Leng, Xu Tan, Wenjie Liu, Kaitao Song, Rui Wang, Xiang-Yang Li, Tao Qin, Edward Lin, Tie-Yan Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26531](https://doi.org/10.1609/aaai.v37i11.26531)

        **Abstract**:

        Error correction in automatic speech recognition (ASR) aims to correct those incorrect words in sentences generated by ASR models. Since recent ASR models usually have low word error rate (WER), to avoid affecting originally correct tokens, error correction models should only modify incorrect words, and therefore detecting incorrect words is important for error correction. Previous works on error correction either implicitly detect error words through target-source attention or CTC (connectionist temporal classification) loss, or explicitly locate specific deletion/substitution/insertion errors. However, implicit error detection does not provide clear signal about which tokens are incorrect and explicit error detection suffers from low detection accuracy. In this paper, we propose SoftCorrect with a soft error detection mechanism to avoid the limitations of both explicit and implicit error detection. Specifically, we first detect whether a token is correct or not through a probability produced by a dedicatedly designed language model, and then design a constrained CTC loss that only duplicates the detected incorrect tokens to let the decoder focus on the correction of error tokens. Compared with implicit error detection with CTC loss, SoftCorrect provides explicit signal about which words are incorrect and thus does not need to duplicate every token but only incorrect tokens; compared with explicit error detection, SoftCorrect does not detect specific deletion/substitution/insertion errors but just leaves it to CTC loss. Experiments on AISHELL-1 and Aidatatang datasets show that SoftCorrect achieves 26.1% and 9.4% CER reduction respectively, outperforming previous works by a large margin, while still enjoying fast speed of parallel generation.

        ----

        ## [1462] Sequence Generation with Label Augmentation for Relation Extraction

        **Authors**: *Bo Li, Dingyao Yu, Wei Ye, Jinglei Zhang, Shikun Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26532](https://doi.org/10.1609/aaai.v37i11.26532)

        **Abstract**:

        Sequence generation demonstrates promising performance in recent information extraction efforts, by incorporating large-scale pre-trained Seq2Seq models. This paper investigates the merits of employing sequence generation in relation extraction, finding that with relation names or synonyms as generation targets, their textual semantics and the correlation (in terms of word sequence pattern) among them affect model performance. We then propose Relation Extraction with Label Augmentation (RELA), a Seq2Seq model with automatic label augmentation for RE. By saying label augmentation, we mean prod semantically synonyms for each relation name as the generation target. Besides, we present an in-depth analysis of the Seq2Seq model's behavior when dealing with RE. Experimental results show that RELA achieves competitive results compared with previous methods on four RE datasets.

        ----

        ## [1463] Reviewing Labels: Label Graph Network with Top-k Prediction Set for Relation Extraction

        **Authors**: *Bo Li, Wei Ye, Jinglei Zhang, Shikun Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26533](https://doi.org/10.1609/aaai.v37i11.26533)

        **Abstract**:

        The typical way for relation extraction is fine-tuning large pre-trained language models on task-specific datasets, then selecting the label with the highest probability of the output
distribution as the final prediction. However, the usage of the Top-k prediction set for a given sample is commonly overlooked. In this paper, we first reveal that the Top-k prediction
set of a given sample contains useful information for predicting the correct label. To effectively utilizes the Top-k prediction set, we propose Label Graph Network with Top-k Prediction Set, termed as KLG. Specifically, for a given sample, we build a label graph to review candidate labels in the Top-k prediction set and learn the connections between them. We also design a dynamic k selection mechanism to learn more powerful and discriminative relation representation. Our experiments show that KLG achieves the best performances on three relation extraction datasets. Moreover, we observe thatKLG is more effective in dealing with long-tailed classes.

        ----

        ## [1464] Online Noisy Continual Relation Learning

        **Authors**: *Guozheng Li, Peng Wang, Qiqing Luo, Yanhe Liu, Wenjun Ke*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26534](https://doi.org/10.1609/aaai.v37i11.26534)

        **Abstract**:

        Recent work for continual relation learning has achieved remarkable progress. However, most existing methods only focus on tackling catastrophic forgetting to improve performance in the existing setup, while continually learning relations in the real-world must overcome many other challenges. One is that the data possibly comes in an online streaming fashion with data distributions gradually changing and without distinct task boundaries. Another is that noisy labels are inevitable in real-world, as relation samples may be contaminated by label inconsistencies or labeled with distant supervision. In this work, therefore, we propose a novel continual relation learning framework that simultaneously addresses both online and noisy relation learning challenges. Our framework contains three key modules: (i) a sample separated online purifying module that divides the online data stream into clean and noisy samples, (ii) a self-supervised online learning module that circumvents inferior training signals caused by noisy data, and (iii) a semi-supervised offline finetuning module that ensures the participation of both clean and noisy samples. Experimental results on FewRel, TACRED and NYT-H with real-world noise demonstrate that our framework greatly outperforms the combinations of the state-of-the-art online continual learning and noisy label learning methods.

        ----

        ## [1465] RESDSQL: Decoupling Schema Linking and Skeleton Parsing for Text-to-SQL

        **Authors**: *Haoyang Li, Jing Zhang, Cuiping Li, Hong Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26535](https://doi.org/10.1609/aaai.v37i11.26535)

        **Abstract**:

        One of the recent best attempts at Text-to-SQL is the pre-trained language model. Due to the structural property of the SQL queries, the seq2seq model takes the responsibility of parsing both the schema items (i.e., tables and columns) and the skeleton (i.e., SQL keywords). Such coupled targets increase the difficulty of parsing the correct SQL queries especially when they involve many schema items and logic operators. This paper proposes a ranking-enhanced encoding and skeleton-aware decoding framework to decouple the schema linking and the skeleton parsing. Specifically, for a seq2seq encoder-decode model, its encoder is injected by the most relevant schema items instead of the whole unordered ones, which could alleviate the schema linking effort during SQL parsing, and its decoder first generates the skeleton and then the actual SQL query, which could implicitly constrain the SQL parsing. We evaluate our proposed framework on Spider and its three robustness variants: Spider-DK, Spider-Syn, and Spider-Realistic. The experimental results show that our framework delivers promising performance and robustness. Our code is available at https://github.com/RUCKBReasoning/RESDSQL.

        ----

        ## [1466] Graphix-T5: Mixing Pre-trained Transformers with Graph-Aware Layers for Text-to-SQL Parsing

        **Authors**: *Jinyang Li, Binyuan Hui, Reynold Cheng, Bowen Qin, Chenhao Ma, Nan Huo, Fei Huang, Wenyu Du, Luo Si, Yongbin Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26536](https://doi.org/10.1609/aaai.v37i11.26536)

        **Abstract**:

        The task of text-to-SQL parsing, which aims at converting natural language questions into executable SQL queries, has garnered increasing attention in recent years. One of the major challenges in text-to-SQL parsing is domain generalization, i.e., how to generalize well to unseen databases. Recently, the pre-trained text-to-text transformer model, namely T5, though not specialized for text-to-SQL parsing, has achieved state-of-the-art performance on standard benchmarks targeting domain generalization. In this work, we explore ways to further augment the pre-trained T5 model with specialized components for text-to-SQL parsing. Such components are expected to introduce structural inductive bias into text-to-SQL parsers thus improving the model’s capacity on (potentially multi-hop) reasoning, which is critical for generating structure-rich SQLs. To this end, we propose a new architecture GRAPHIX-T5, a mixed model with the standard pre-trained transformer model augmented by specially-designed graph-aware layers. Extensive experiments and analysis demonstrate the effectiveness of GRAPHIX-T5 across four text-to-SQL benchmarks: SPIDER, SYN, REALISTIC and DK. GRAPHIX-T5 surpasses all other T5-based parsers with a significant margin, achieving new state-of-the-art performance. Notably, GRAPHIX-T5-large reaches performance superior to the original T5-large by 5.7% on exact match (EM) accuracy and 6.6% on execution accuracy (EX). This even outperforms the T5-3B by 1.2% on EM and 1.5% on EX

        ----

        ## [1467] Compressed Heterogeneous Graph for Abstractive Multi-Document Summarization

        **Authors**: *Miao Li, Jianzhong Qi, Jey Han Lau*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26537](https://doi.org/10.1609/aaai.v37i11.26537)

        **Abstract**:

        Multi-document summarization (MDS) aims to generate a summary for a number of related documents. We propose HGSum — an MDS model that extends an encoder-decoder architecture to incorporate a heterogeneous graph to represent different semantic units (e.g., words and sentences) of the documents. This contrasts with existing MDS models which do not consider different edge types of graphs and as such do not capture the diversity of relationships in the documents. To preserve only key information and relationships of the documents in the heterogeneous graph, HGSum uses graph pooling to compress the input graph. And to guide HGSum to learn the compression, we introduce an additional objective that maximizes the similarity between the compressed graph and the graph constructed from the ground-truth summary during training. HGSum is trained end-to-end with the graph similarity and standard cross-entropy objectives. Experimental results over Multi-News, WCEP-100, and Arxiv show that HGSum outperforms state-of-the-art MDS models. The code for our model and experiments is available at: https://github.com/oaimli/HGSum.

        ----

        ## [1468] TrOCR: Transformer-Based Optical Character Recognition with Pre-trained Models

        **Authors**: *Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei A. F. Florêncio, Cha Zhang, Zhoujun Li, Furu Wei*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26538](https://doi.org/10.1609/aaai.v37i11.26538)

        **Abstract**:

        Text recognition is a long-standing research problem for document digitalization. Existing approaches are usually built based on CNN for image understanding and RNN for char-level text generation. In addition, another language model is usually needed to improve the overall accuracy as a post-processing step. In this paper, we propose an end-to-end text recognition approach with pre-trained image Transformer and text Transformer models, namely TrOCR, which leverages the Transformer architecture for both image understanding and wordpiece-level text generation. The TrOCR model is simple but effective, and can be pre-trained with large-scale synthetic data and fine-tuned with human-labeled datasets. Experiments show that the TrOCR model outperforms the current state-of-the-art models on the printed, handwritten and scene text recognition tasks. The TrOCR models and code are publicly available at https://aka.ms/trocr.

        ----

        ## [1469] Mitigating Negative Style Transfer in Hybrid Dialogue System

        **Authors**: *Shimin Li, Qinyuan Cheng, Linyang Li, Xipeng Qiu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26539](https://doi.org/10.1609/aaai.v37i11.26539)

        **Abstract**:

        As the functionality of dialogue systems evolves, hybrid dialogue systems that accomplish user-specific goals and participate in open-topic chitchat with users are attracting growing attention. Existing research learns both tasks concurrently utilizing a multi-task fusion technique but ignores the negative transfer phenomenon induced by the unique textual style differences. Therefore, contrastive learning based on the latent variable model is used to decouple the various textual genres in the latent space. We devise supervised and self-supervised positive and negative sample constructions for diverse datasets. In addition, to capitalize on the style information contained in the decoupled latent variables, we employ a style prefix that incorporates latent variables further to control the generation of responses with varying styles. We performed extensive experiments on three dialogue datasets, including a hybrid dialogue dataset and two task-oriented dialogue datasets. The experimental results demonstrate that our method can mitigate the negative style transfer issue and achieves state-of-the-art performance on multiple dialogue datasets.

        ----

        ## [1470] Low Resource Quantitative Information Extraction via Structure Searching and Prefix-Based Text Generation

        **Authors**: *Tongliang Li, Zixiang Wang, Zhoujun Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26540](https://doi.org/10.1609/aaai.v37i11.26540)

        **Abstract**:

        Quantitative information plays an important part in the financial and data analysis areas. Prior work relied on pattern-matching methods and complex hand-crafted rules to extract quantitative information due to the lack of labeled data. Such methods can be unstable and difficult to scale to the open domain. In this paper, we study quantitative information extraction in the low-resource setting. We propose a search-based approach by searching from the syntactic structures to acquire basic training data. The search process is simple yet effective. Then, a prefix-based text-to-text generation method is employed to extract the quantitative information. The prefix design can fully leverage pre-trained language models for text generation to serve the information extraction purpose. Experimental results show that our approaches achieves high performance with a limited amount of labeled data. The extraction result could further boost the performance of other tasks such as quantitative reasoning.

        ----

        ## [1471] SKIER: A Symbolic Knowledge Integrated Model for Conversational Emotion Recognition

        **Authors**: *Wei Li, Luyao Zhu, Rui Mao, Erik Cambria*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26541](https://doi.org/10.1609/aaai.v37i11.26541)

        **Abstract**:

        Emotion recognition in conversation (ERC) has received increasing attention from the research community. However, the ERC task is challenging, largely due to the complex and unstructured properties of multi-party conversations. Besides, the majority of daily dialogues take place in a specific context or circumstance, which requires rich external knowledge to understand the background of a certain dialogue. In this paper, we address these challenges by explicitly modeling the discourse relations between utterances and incorporating symbolic knowledge into multi-party conversations. We first introduce a dialogue parsing algorithm into ERC and further improve the algorithm through a transfer learning method. Moreover, we leverage different symbolic knowledge graph relations to learn knowledge-enhanced features for the ERC task. Extensive experiments on three benchmarks demonstrate that both dialogue structure graphs and symbolic knowledge are beneficial to the model performance on the task. Additionally, experimental results indicate that the proposed model surpasses baseline models on several indices.

        ----

        ## [1472] PGSS: Pitch-Guided Speech Separation

        **Authors**: *Xiang Li, Yiwen Wang, Yifan Sun, Xihong Wu, Jing Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26542](https://doi.org/10.1609/aaai.v37i11.26542)

        **Abstract**:

        Monaural speech separation aims to separate concurrent speakers from a single-microphone mixture recording. Inspired by the effect of pitch priming in auditory scene analysis (ASA) mechanisms, a novel pitch-guided speech separation framework is proposed in this work. The prominent advantage of this framework is that both the permutation problem and the unknown speaker number problem existing in general models can be avoided by using pitch contours as the primary means to guide the target speaker. In addition, adversarial training is applied, instead of a traditional time-frequency mask, to improve the perceptual quality of separated speech. Specifically, the proposed framework can be divided into two phases: pitch extraction and speech separation. The former aims to extract pitch contour candidates for each speaker from the mixture, modeling the bottom-up process in ASA mechanisms. Any pitch contour can be selected as the condition in the second phase to separate the corresponding speaker, where a conditional generative adversarial network (CGAN) is applied. The second phase models the effect of pitch priming in ASA. Experiments on the WSJ0-2mix corpus reveal that the proposed approaches can achieve higher pitch extraction accuracy and better separation performance, compared to the baseline models, and have the potential to be applied to SOTA architectures.

        ----

        ## [1473] DyRRen: A Dynamic Retriever-Reranker-Generator Model for Numerical Reasoning over Tabular and Textual Data

        **Authors**: *Xiao Li, Yin Zhu, Sichen Liu, Jiangzhou Ju, Yuzhong Qu, Gong Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26543](https://doi.org/10.1609/aaai.v37i11.26543)

        **Abstract**:

        Numerical reasoning over hybrid data containing tables and long texts has recently received research attention from the AI community. To generate an executable reasoning program consisting of math and table operations to answer a question, state-of-the-art methods use a retriever-generator pipeline. However, their retrieval results are static, while different generation steps may rely on different sentences. To attend to the retrieved information that is relevant to each generation step, in this paper, we propose DyRRen, an extended retriever-reranker-generator framework where each generation step is enhanced by a dynamic reranking of retrieved sentences. It outperforms existing baselines on the FinQA dataset.

        ----

        ## [1474] Heterogeneous-Branch Collaborative Learning for Dialogue Generation

        **Authors**: *Yiwei Li, Shaoxiong Feng, Bin Sun, Kan Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26544](https://doi.org/10.1609/aaai.v37i11.26544)

        **Abstract**:

        With the development of deep learning, advanced dialogue generation methods usually require a greater amount of computational resources. One promising approach to obtaining a high-performance and lightweight model is knowledge distillation, which relies heavily on the pre-trained powerful teacher. Collaborative learning, also known as online knowledge distillation, is an effective way to conduct one-stage group distillation in the absence of a well-trained large teacher model. However, previous work has a severe branch homogeneity problem due to the same training objective and the independent identical training sets.  To alleviate this problem, we consider the dialogue attributes in the training of network branches. Each branch learns the attribute-related features based on the selected subset. Furthermore, we propose a dual group-based knowledge distillation method, consisting of positive distillation and negative distillation, to further diversify the features of different branches in a steadily and interpretable way. The proposed approach significantly improves branch heterogeneity and outperforms state-of-the-art collaborative learning methods on two widely used open-domain dialogue datasets.

        ----

        ## [1475] Learning to Know Myself: A Coarse-to-Fine Persona-Aware Training Framework for Personalized Dialogue Generation

        **Authors**: *Yunpeng Li, Yue Hu, Yajing Sun, Luxi Xing, Ping Guo, Yuqiang Xie, Wei Peng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26545](https://doi.org/10.1609/aaai.v37i11.26545)

        **Abstract**:

        A critical challenge for open-domain dialogue agents is to generate persona-relevant and consistent responses. Due to the nature of persona sparsity in conversation scenarios, previous persona-based dialogue agents trained with Maximum Likelihood Estimation tend to overlook the given personas and generate responses irrelevant or inconsistent with personas. To address this problem, we propose a two-stage coarse-to-fine persona-aware training framework to improve the persona consistency of a dialogue agent progressively. Specifically, our framework first trains the dialogue agent to answer the constructed persona-aware questions, making it highly sensitive to the personas to generate persona-relevant responses. Then the dialogue agent is further trained with a contrastive learning paradigm by explicitly perceiving the difference between the consistent and the generated inconsistent responses, forcing it to pay more attention to the key persona information to generate consistent responses. By applying our proposed training framework to several representative baseline models, experimental results show significant boosts on both automatic and human evaluation metrics, especially the consistency of generated responses.

        ----

        ## [1476] WIERT: Web Information Extraction via Render Tree

        **Authors**: *Zimeng Li, Bo Shao, Linjun Shou, Ming Gong, Gen Li, Daxin Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26546](https://doi.org/10.1609/aaai.v37i11.26546)

        **Abstract**:

        Web information extraction (WIE) is a fundamental problem in web document understanding, with a significant impact on various applications. Visual information plays a crucial role in WIE tasks as the nodes containing relevant information are often visually distinct, such as being in a larger font size or having a brighter color, from the other nodes. However, rendering visual information of a web page can be computationally expensive. Previous works have mainly focused on the Document Object Model (DOM) tree, which lacks visual information. To efficiently exploit visual information, we propose leveraging the render tree, which combines the DOM tree and Cascading Style Sheets Object Model (CSSOM) tree, and contains not only content and layout information but also rich visual information at a little additional acquisition cost compared to the DOM tree. In this paper, we present WIERT, a method that effectively utilizes the render tree of a web page based on a pretrained language model. We evaluate WIERT on the Klarna product page dataset, a manually labeled dataset of renderable e-commerce web pages, demonstrating its effectiveness and robustness.

        ----

        ## [1477] STAGE: Span Tagging and Greedy Inference Scheme for Aspect Sentiment Triplet Extraction

        **Authors**: *Shuo Liang, Wei Wei, Xian-Ling Mao, Yuanyuan Fu, Rui Fang, Dangyang Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26547](https://doi.org/10.1609/aaai.v37i11.26547)

        **Abstract**:

        Aspect Sentiment Triplet Extraction (ASTE) has become an emerging task in sentiment analysis research, aiming to extract triplets of the aspect term, its corresponding opinion term, and its associated sentiment polarity from a given sentence. Recently, many neural networks based models with different tagging schemes have been proposed, but almost all of them have their limitations: heavily relying on 1) prior assumption that each word is only associated with a single role (e.g., aspect term, or opinion term, etc. ) and 2) word-level interactions and treating each opinion/aspect as a set of independent words. Hence, they perform poorly on the complex ASTE task, such as a word associated with multiple roles or an aspect/opinion term with multiple words. Hence, we propose a novel approach, Span TAgging and Greedy infErence (STAGE), to extract sentiment triplets in span-level, where each span may consist of multiple words and play different roles simultaneously. To this end, this paper formulates the ASTE task as a multi-class span classification problem. Specifically, STAGE generates more accurate aspect sentiment triplet extractions via exploring span-level information and constraints, which consists of two components, namely, span tagging scheme and greedy inference strategy. The former tag all possible candidate spans based on a newly-defined tagging set. The latter retrieves the aspect/opinion term with the maximum length from the candidate sentiment snippet to output sentiment triplets. Furthermore, we propose a simple but effective model based on the STAGE, which outperforms the state-of-the-arts by a large margin on four widely-used datasets. Moreover, our STAGE can be easily generalized to other pair/triplet extraction tasks, which also demonstrates the superiority of the proposed scheme STAGE.

        ----

        ## [1478] Generalizing Math Word Problem Solvers via Solution Diversification

        **Authors**: *Zhenwen Liang, Jipeng Zhang, Lei Wang, Yan Wang, Jie Shao, Xiangliang Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26548](https://doi.org/10.1609/aaai.v37i11.26548)

        **Abstract**:

        Current math word problem (MWP) solvers are usually Seq2Seq models  trained by the (one-problem; one-solution) pairs, each of which is made of a problem description and a solution showing reasoning flow to get the correct answer. However, one MWP problem naturally has multiple solution equations. The training of an MWP solver with (one-problem; one-solution) pairs excludes other correct solutions, and thus limits the generalizability of the MWP solver. One feasible solution to this limitation is to augment multiple solutions to a given problem. However, it is difficult to collect diverse and accurate augment solutions through human efforts. In this paper, we design a new training framework for an MWP solver by introducing a solution buffer and a solution discriminator. The buffer includes solutions generated by an MWP solver to encourage  the training data diversity.  The discriminator controls the quality of buffered solutions to participate in training. Our framework is flexibly applicable to a wide setting of fully, semi-weakly and weakly supervised training for all Seq2Seq MWP solvers. We conduct extensive experiments on a benchmark dataset Math23k and a new dataset named Weak12k, and show that our framework improves the performance of various MWP solvers under different settings by generating correct and diverse solutions.

        ----

        ## [1479] On Grounded Planning for Embodied Tasks with Language Models

        **Authors**: *Bill Yuchen Lin, Chengsong Huang, Qian Liu, Wenda Gu, Sam Sommerer, Xiang Ren*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26549](https://doi.org/10.1609/aaai.v37i11.26549)

        **Abstract**:

        Language models (LMs) have demonstrated their capability in possessing commonsense knowledge of the physical world, a crucial aspect of performing tasks in everyday life. However, it remains unclear whether they have the capacity to generate grounded, executable plans for embodied tasks. This is a challenging task as LMs lack the ability to perceive the environment through vision and feedback from the physical environment. In this paper, we address this important research question and present the first investigation into the topic. Our novel problem formulation, named G-PlanET, inputs a high-level goal and a data table about objects in a specific environment, and then outputs a step-by-step actionable plan for a robotic agent to follow. To facilitate the study, we establish an evaluation protocol and design a dedicated metric, KAS, to assess the quality of the plans. Our experiments demonstrate that the use of tables for encoding the environment and an iterative decoding strategy can significantly enhance the LMs' ability in grounded planning. Our analysis also reveals interesting and non-trivial findings.

        ----

        ## [1480] DeAR: A Deep-Learning-Based Audio Re-recording Resilient Watermarking

        **Authors**: *Chang Liu, Jie Zhang, Han Fang, Zehua Ma, Weiming Zhang, Nenghai Yu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26550](https://doi.org/10.1609/aaai.v37i11.26550)

        **Abstract**:

        Audio watermarking is widely used for leaking source tracing. The robustness of the watermark determines the traceability of the algorithm. With the development of digital technology, audio re-recording (AR) has become an efficient and covert means to steal secrets. AR process could drastically destroy the watermark signal while preserving the original information. This puts forward a new requirement for audio watermarking at this stage, that is, to be robust to AR distortions. Unfortunately, none of the existing algorithms can effectively resist AR attacks due to the complexity of the AR process. To address this limitation, this paper proposes DeAR, a deep-learning-based audio re-recording resistant watermarking. Inspired by DNN-based image watermarking, we pioneer a deep learning framework for audio carriers, based on which the watermark signal can be effectively embedded and extracted. Meanwhile, in order to resist the AR attack, we delicately analyze the distortions that occurred in the AR process and design the corresponding distortion layer to cooperate with the proposed watermarking framework. Extensive experiments show that the proposed algorithm can resist not only common electronic channel distortions but also AR distortions. Under the premise of high-quality embedding (SNR=25.86dB), in the case of a common re-recording distance (20cm), the algorithm can effectively achieve an average bit recovery accuracy of 98.55%.

        ----

        ## [1481] Detecting and Grounding Important Characters in Visual Stories

        **Authors**: *Danyang Liu, Frank Keller*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26551](https://doi.org/10.1609/aaai.v37i11.26551)

        **Abstract**:

        Characters are essential to the plot of any story. Establishing the characters before writing a story can improve the clarity of the plot and the overall flow of the narrative. However, previous work on visual storytelling tends to focus on detecting objects in images and discovering relationships between them. In this approach, characters are not distinguished from other objects when they are fed into the generation pipeline. The result is a coherent sequence of events rather than a character-centric story. In order to address this limitation, we introduce the VIST-Character dataset, which provides rich character-centric annotations, including visual and textual co-reference chains and importance ratings for characters. Based on this dataset, we propose two new tasks: important character detection and character grounding in visual stories. For both tasks, we develop simple, unsupervised models based on distributional similarity and pre-trained vision-and-language models. Our new dataset, together with these models, can serve as the foundation for subsequent work on analysing and generating stories from a character-centric perspective.

        ----

        ## [1482] Boosting Few-Shot Text Classification via Distribution Estimation

        **Authors**: *Han Liu, Feng Zhang, Xiaotong Zhang, Siyang Zhao, Fenglong Ma, Xiao-Ming Wu, Hongyang Chen, Hong Yu, Xianchao Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26552](https://doi.org/10.1609/aaai.v37i11.26552)

        **Abstract**:

        Distribution estimation has been demonstrated as one of the most effective approaches in dealing with few-shot image classification, as the low-level patterns and underlying representations can be easily transferred across different tasks in computer vision domain. However, directly applying this approach to few-shot text classification is challenging, since leveraging the statistics of known classes with sufficient samples to calibrate the distributions of novel classes may cause negative effects due to serious category difference in text domain. To alleviate this issue, we propose two simple yet effective strategies to estimate the distributions of the novel classes by utilizing unlabeled query samples, thus avoiding the potential negative transfer issue. Specifically, we first assume a class or sample follows the Gaussian distribution, and use the original support set and the nearest few query samples to estimate the corresponding mean and covariance. Then, we augment the labeled samples by sampling from the estimated distribution, which can provide sufficient supervision for training the classification model. Extensive experiments on eight few-shot text classification datasets show that the proposed method outperforms state-of-the-art baselines significantly.

        ----

        ## [1483] SSPAttack: A Simple and Sweet Paradigm for Black-Box Hard-Label Textual Adversarial Attack

        **Authors**: *Han Liu, Zhi Xu, Xiaotong Zhang, Xiaoming Xu, Feng Zhang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26553](https://doi.org/10.1609/aaai.v37i11.26553)

        **Abstract**:

        Hard-label textual adversarial attack is a challenging task, as only the predicted label information is available, and the text space is discrete and non-differentiable. Relevant research work is still in fancy and just a handful of methods are proposed. However, existing methods suffer from either the high complexity of genetic algorithms or inaccurate gradient estimation, thus are arduous to obtain adversarial examples with high semantic similarity and low perturbation rate under the tight-budget scenario. In this paper, we propose a simple and sweet paradigm for hard-label textual adversarial attack, named SSPAttack. Specifically, SSPAttack first utilizes initialization to generate an adversarial example, and removes unnecessary replacement words to reduce the number of changed words. Then it determines the replacement order and searches for an anchor synonym, thus avoiding going through all the synonyms. Finally, it pushes substitution words towards original words until an appropriate adversarial example is obtained. The core idea of SSPAttack is just swapping words whose mechanism is simple. Experimental results on eight benchmark datasets and two real-world APIs have shown that the performance of SSPAttack is sweet in terms of similarity, perturbation rate and query efficiency.

        ----

        ## [1484] LADA-Trans-NER: Adaptive Efficient Transformer for Chinese Named Entity Recognition Using Lexicon-Attention and Data-Augmentation

        **Authors**: *Jiguo Liu, Chao Liu, Nan Li, Shihao Gao, Mingqi Liu, Dali Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26554](https://doi.org/10.1609/aaai.v37i11.26554)

        **Abstract**:

        Recently, word enhancement has become very popular for Chinese Named Entity Recognition (NER), reducing segmentation errors and increasing the semantic and boundary information of Chinese words. However, these methods tend to ignore the semantic relationship before and after the sentence after integrating lexical information. Therefore, the regularity of word length information has not been fully explored in various word-character fusion methods. In this work, we propose a Lexicon-Attention and Data-Augmentation (LADA) method for Chinese NER. We discuss the challenges of using existing methods in incorporating word information for NER and show how our proposed methods could be leveraged to overcome those challenges. LADA is based on a Transformer Encoder that utilizes lexicon to construct a directed graph and fuses word information through updating the optimal edge of the graph. Specially, we introduce the advanced data augmentation method to obtain the optimal representation for the NER task. Experimental results show that the augmentation done using LADA can considerably boost the performance of our NER system and achieve significantly better results than previous state-of-the-art methods and variant models in the literature on four publicly available NER datasets, namely Resume, MSRA, Weibo, and OntoNotes v4. We also observe better generalization and application to a real-world setting from LADA on multi-source complex entities.

        ----

        ## [1485] Selective Knowledge Distillation for Non-Autoregressive Neural Machine Translation

        **Authors**: *Min Liu, Yu Bao, Chengqi Zhao, Shujian Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26555](https://doi.org/10.1609/aaai.v37i11.26555)

        **Abstract**:

        Benefiting from the sequence-level knowledge distillation, the Non-Autoregressive Transformer (NAT) achieves great success in neural machine translation tasks. 
However, existing knowledge distillation has side effects, such as propagating errors from the teacher to NAT students, which may limit further improvements of NAT models and are rarely discussed in existing research.
In this paper, we introduce selective knowledge distillation by introducing an NAT evaluator to select NAT-friendly targets that are of high quality and easy to learn.
In addition, we introduce a simple yet effective progressive distillation method to boost NAT performance. 
Experiment results on multiple WMT language directions and several representative NAT models show that our approach can realize a flexible trade-off between the quality and complexity of training data for NAT models, achieving strong performances.
Further analysis shows that distilling only 5% of the raw translations can help an NAT outperform its counterpart trained on raw data by about 2.4 BLEU.

        ----

        ## [1486] A Disentangled-Attention Based Framework with Persona-Aware Prompt Learning for Dialogue Generation

        **Authors**: *Pingsheng Liu, Zhengjie Huang, Xiechi Zhang, Linlin Wang, Gerard de Melo, Xin Lin, Liang Pang, Liang He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26556](https://doi.org/10.1609/aaai.v37i11.26556)

        **Abstract**:

        Endowing dialogue agents with personas is the key to delivering more human-like conversations. However, existing persona-grounded dialogue systems still lack informative details of human conversations and tend to reply with inconsistent and generic responses. One of the main underlying causes is that pre-defined persona sentences are generally short and merely superficial descriptions of personal attributes, making appropriate persona selection and understanding non-trivial. Another challenge is that it is crucial to consider the context and the conversation flow to dynamically determine when to invoke different types of persona signals. To address these problems, we propose a disentangled-attention based pre-training architecture, which incorporates persona-aware prompt learning to bridge the connection between the selected persona and response generation. Our model first exploits the conversation flow to select context-relevant personas, and subsequently enriches the superficial persona descriptions with extra personality traits through persona-aware prompting. Finally, the decoder leverages a disentangled-attention mechanism to flexibly control the reliance on personas and dialogue contexts, and incorporates A*-like keyword-based heuristic estimates for controllable generation. Extensive experiments show that our approach can outperform strong baselines and deliver more consistent and engaging responses on the PERSONA-CHAT dataset.

        ----

        ## [1487] Towards Credible Human Evaluation of Open-Domain Dialog Systems Using Interactive Setup

        **Authors**: *Sijia Liu, Patrick Lange, Behnam Hedayatnia, Alexandros Papangelis, Di Jin, Andrew Wirth, Yang Liu, Dilek Hakkani-Tur*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26557](https://doi.org/10.1609/aaai.v37i11.26557)

        **Abstract**:

        Evaluating open-domain conversation models has been an open challenge due to the open-ended nature of conversations. In addition to static evaluations, recent work has started to explore a variety of per-turn and per-dialog interactive evaluation mechanisms and provide advice on the best setup. In this work, we adopt the interactive evaluation framework and further apply to multiple models with a focus on per-turn evaluation techniques. Apart from the widely used setting where participants select the best response among different candidates at each turn, one more novel per-turn evaluation setting is adopted, where participants can select all appropriate responses with different fallback strategies to continue the conversation when no response is selected. We evaluate these settings based on sensitivity and consistency using four GPT2-based models that differ in model sizes or fine-tuning data. To better generalize to any model groups with no prior assumptions on their rankings and control evaluation costs for all setups, we also propose a methodology to estimate the required sample size given a minimum performance gap of interest before running most experiments. Our comprehensive human evaluation results shed light on how to conduct credible human evaluations of open domain dialog systems using the interactive setup, and suggest additional future directions.

        ----

        ## [1488] Unsupervised Paraphrasing under Syntax Knowledge

        **Authors**: *Tianyuan Liu, Yuqing Sun, Jiaqi Wu, Xi Xu, Yuchen Han, Cheng Li, Bin Gong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26558](https://doi.org/10.1609/aaai.v37i11.26558)

        **Abstract**:

        The soundness of syntax is an important issue for the paraphrase generation task. 
Most methods control the syntax of paraphrases by embedding the syntax and semantics in the generation process, which cannot guarantee the syntactical correctness of the results. 
Different from them, in this paper we investigate the structural patterns of word usages termed as the word composable knowledge and integrate it into the paraphrase generation to control the syntax in an explicit way.
This syntax knowledge is pretrained on a large corpus with the dependency relationships and formed as the probabilistic functions on the word-level syntactical soundness.
For the sentence-level correctness, we design a hierarchical syntax structure loss to quantitatively verify the syntactical soundness of the paraphrase against the given dependency template. 
Thus, the generation process can select the appropriate words with consideration on both semantics and syntax. 
The proposed method is evaluated on a few paraphrase datasets.
The experimental results show that the quality of paraphrases by our proposed method outperforms the compared methods, especially in terms of syntax correctness.

        ----

        ## [1489] Adjective Scale Probe: Can Language Models Encode Formal Semantics Information?

        **Authors**: *Wei Liu, Ming Xiang, Nai Ding*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26559](https://doi.org/10.1609/aaai.v37i11.26559)

        **Abstract**:

        It is an open question what semantic representations transformer-based language models can encode and whether they have access to more abstract aspects of semantic meaning. Here, we propose a diagnostic dataset to investigate how well language models understand the degree semantics of adjectives. In the dataset, referred as the Adjective Scale Probe (ASP), we semi-automatically generate 8 tests of Natural Language Inference (NLI) questions to test 8 key capabilities of adjective interpretation. We apply the ASP dataset to evaluate the performance of 3 language models, i.e., BERT, DeBERTa, and T0. It is found that language models perform below the majority baseline for most tests of the ASP, even when the models have been fine-tuned to achieve high performance on the large-scale MNLI dataset. But after we fine-tune the pre-trained models on a subset of the ASP, DeBERTa can achieve high performance on the untrained adjectives and untrained tests, suggesting that DeBERTa may have captured degree semantic information of adjectives through pre-training but it needs specific training data to learn how to apply such information to the current tasks. In sum, the ASP provides an easy-to-use method to test fine-grained formal semantic properties of adjectives, and reveals language models' abilities to access formal semantic information.

        ----

        ## [1490] Effective Open Intent Classification with K-center Contrastive Learning and Adjustable Decision Boundary

        **Authors**: *Xiaokang Liu, Jianquan Li, Jingjing Mu, Min Yang, Ruifeng Xu, Benyou Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26560](https://doi.org/10.1609/aaai.v37i11.26560)

        **Abstract**:

        Open intent classification, which aims to correctly classify the known intents into their corresponding classes while identifying the new unknown (open) intents, is an essential but challenging task in dialogue systems. In this paper, we introduce novel K-center contrastive learning and adjustable decision boundary learning (CLAB) to improve the effectiveness of open intent classification. First, we pre-train a feature encoder on the labeled training instances, which transfers knowledge from known intents to unknown intents. Specifically, we devise a K-center contrastive learning algorithm to learn discriminative and balanced intent features, improving the generalization of the model for recognizing open intents. Second, we devise an adjustable decision boundary learning method with expanding and shrinking (ADBES) to determine the suitable decision conditions. Concretely, we learn a decision boundary for each known intent class, which consists of a decision center and the radius of the decision boundary. We then expand the radius of the decision boundary to accommodate more in-class instances if the out-of-class instances are far from the decision boundary; otherwise, we shrink the radius of the decision boundary. Extensive experiments on three benchmark datasets clearly demonstrate the effectiveness of our method for open intent classification.For reproducibility, we submit the code at: https://github.com/lxk00/CLAP

        ----

        ## [1491] Learning Compositional Tasks from Language Instructions

        **Authors**: *Lajanugen Logeswaran, Wilka Carvalho, Honglak Lee*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26561](https://doi.org/10.1609/aaai.v37i11.26561)

        **Abstract**:

        The ability to combine learned knowledge and skills to solve novel tasks is a key aspect of generalization in humans that allows us to understand and perform tasks described by novel language utterances. While progress has been made in supervised learning settings, no work has yet studied compositional generalization of a reinforcement learning agent following natural language instructions in an embodied environment. We develop a set of tasks in a photo-realistic simulated kitchen environment that allow us to study the degree to which a behavioral policy captures the systematicity in language by studying its zero-shot generalization performance on held out natural language instructions. We show that our agent which leverages a novel additive action-value decomposition in tandem with attention based subgoal prediction is able to exploit composition in text instructions to generalize to unseen tasks.

        ----

        ## [1492] SPRING: Situated Conversation Agent Pretrained with Multimodal Questions from Incremental Layout Graph

        **Authors**: *Yuxing Long, Binyuan Hui, Fulong Ye, Yanyang Li, Zhuoxin Han, Caixia Yuan, Yongbin Li, Xiaojie Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26562](https://doi.org/10.1609/aaai.v37i11.26562)

        **Abstract**:

        Existing multimodal conversation agents have shown impressive abilities to locate absolute positions or retrieve attributes in simple scenarios, but they fail to perform well when complex relative positions and information alignments are involved, which poses a bottleneck in response quality. In this paper, we propose a Situated Conversation Agent Pretrained with Multimodal Questions from Incremental Layout Graph (SPRING) with abilities of reasoning multi-hops spatial relations and connecting them with visual attributes in crowded situated scenarios. Specifically, we design two types of Multimodal Question Answering (MQA) tasks to pretrain the agent. All QA pairs utilized during pretraining are generated from novel Increment Layout Graphs (ILG). QA pair difficulty labels automatically annotated by ILG are used to promote MQA-based Curriculum Learning. Experimental results verify the SPRING's effectiveness, showing that it significantly outperforms state-of-the-art approaches on both SIMMC 1.0 and SIMMC 2.0 datasets. We release our code and data at https://github.com/LYX0501/SPRING.

        ----

        ## [1493] Universal Information Extraction as Unified Semantic Matching

        **Authors**: *Jie Lou, Yaojie Lu, Dai Dai, Wei Jia, Hongyu Lin, Xianpei Han, Le Sun, Hua Wu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26563](https://doi.org/10.1609/aaai.v37i11.26563)

        **Abstract**:

        The challenge of information extraction (IE) lies in the diversity of label schemas and the heterogeneity of structures.
Traditional methods require task-specific model design and rely heavily on expensive supervision, making them difficult to generalize to new schemas.
In this paper, we decouple IE into two basic abilities, structuring and conceptualizing, which are shared by different tasks and schemas.
Based on this paradigm, we propose to universally model various IE tasks with Unified Semantic Matching (USM) framework, which introduces three unified token linking operations to model the abilities of structuring and conceptualizing.
In this way, USM can jointly encode schema and input text, uniformly extract substructures in parallel, and controllably decode target structures on demand.
Empirical evaluation on 4 IE tasks shows that the proposed method achieves state-of-the-art performance under the supervised experiments and shows strong generalization ability in zero/few-shot transfer settings.

        ----

        ## [1494] PUnifiedNER: A Prompting-Based Unified NER System for Diverse Datasets

        **Authors**: *Jinghui Lu, Rui Zhao, Brian Mac Namee, Fei Tan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26564](https://doi.org/10.1609/aaai.v37i11.26564)

        **Abstract**:

        Much of named entity recognition (NER) research focuses on developing dataset-specific models based on data from the domain of interest, and a limited set of related entity types. This is frustrating as each new dataset requires a new model to be trained and stored. In this work, we present a ``versatile'' model---the Prompting-based Unified NER system (PUnifiedNER)---that works with data from different domains and can recognise up to 37 entity types simultaneously, and theoretically it could be as many as possible. By using prompt learning, PUnifiedNER is a novel approach that is able to jointly train across multiple corpora, implementing intelligent on-demand entity recognition. Experimental results show that PUnifiedNER leads to significant prediction benefits compared to dataset-specific models with impressively reduced model deployment costs. Furthermore, the performance of PUnifiedNER can achieve competitive or even better performance than state-of-the-art domain-specific methods for some datasets. We also perform comprehensive pilot and ablation studies to support in-depth analysis of each component in PUnifiedNER.

        ----

        ## [1495] KICE: A Knowledge Consolidation and Expansion Framework for Relation Extraction

        **Authors**: *Yilin Lu, Xiaoqiang Wang, Haofeng Yang, Siliang Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26565](https://doi.org/10.1609/aaai.v37i11.26565)

        **Abstract**:

        Machine Learning is often challenged by insufficient labeled data. Previous methods employing implicit commonsense knowledge of pre-trained language models (PLMs) or pattern-based symbolic knowledge have achieved great success in mitigating manual annotation efforts. In this paper, we focus on the collaboration among different knowledge sources and present KICE, a Knowledge-evolving framework by Iterative Consolidation and Expansion with the guidance of PLMs and rule-based patterns. Specifically, starting with limited labeled data as seeds, KICE first builds a Rule Generator by prompt-tuning to stimulate the rich knowledge distributed in PLMs, generate seed rules, and initialize the rules set. Afterwards, based on the rule-labeled data, the task model is trained in a self-training pipeline where the knowledge in rules set is consolidated with self-learned high-confidence rules. Finally, for the low-confidence rules, KICE solicits human-enlightened understanding and expands the knowledge coverage for better task model training. Our framework is verified on relation extraction (RE) task, and the experiments on TACRED show that the model performance (F1) grows from 33.24% to 79.84% with the enrichment of knowledge, outperforming all the baselines including other knowledgeable methods.

        ----

        ## [1496] Zero-Shot Slot Filling with Slot-Prefix Prompting and Attention Relationship Descriptor

        **Authors**: *Qiaoyang Luo, Lingqiao Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26566](https://doi.org/10.1609/aaai.v37i11.26566)

        **Abstract**:

        This paper addresses zero-shot slot filling, which tries to build a system that can generalize to unseen slot types without any training data. The key to zero-shot slot-filling is to match the tokens from the utterance with the semantic definition of the slot without training data in the target domain. This paper tackles this problem by devising a scheme to fully leverage pre-trained language models (PLMs). To this end, we propose a new prompting scheme that utilizes both learnable tokens and slot names to guide the model to focus on the relevant text spans for a given slot. Furthermore, we use attention values between tokens to form a feature descriptor for each token, which is motivated by the fact that the attention value in a PLM naturally characterizes various relationships, e.g., syntactic or semantic, between tokens. By further consolidating those features with an additional transformer-based aggregation module, we create a simple-but-effective zero-shot slot filling system that can achieve significantly better performance than the previous methods, as demonstrated by our experimental studies.

        ----

        ## [1497] Feature-Level Debiased Natural Language Understanding

        **Authors**: *Yougang Lyu, Piji Li, Yechang Yang, Maarten de Rijke, Pengjie Ren, Yukun Zhao, Dawei Yin, Zhaochun Ren*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26567](https://doi.org/10.1609/aaai.v37i11.26567)

        **Abstract**:

        Natural language understanding (NLU) models often rely on dataset biases rather than intended task-relevant features to achieve high performance on specific datasets. As a result, these models perform poorly on datasets outside the training distribution. Some recent studies address this issue by reducing the weights of biased samples during the training process. However, these methods still encode biased latent features in representations and neglect the dynamic nature of bias, which hinders model prediction. We propose an NLU debiasing method, named debiasing contrastive learning (DCT), to simultaneously alleviate the above problems based on contrastive learning. We devise a debiasing, positive sampling strategy to mitigate biased latent features by selecting the least similar biased positive samples. We also propose a dynamic negative sampling strategy to capture the dynamic influence of biases by employing a bias-only model to dynamically select the most similar biased negative samples. We conduct experiments on three NLU benchmark datasets. Experimental results show that DCT outperforms state-of-the-art baselines on out-of-distribution datasets while maintaining in-distribution performance. We also verify that DCT can reduce biased latent features from the model's representation.

        ----

        ## [1498] Graph Component Contrastive Learning for Concept Relatedness Estimation

        **Authors**: *Yueen Ma, Zixing Song, Xuming Hu, Jingjing Li, Yifei Zhang, Irwin King*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26568](https://doi.org/10.1609/aaai.v37i11.26568)

        **Abstract**:

        Concept relatedness estimation (CRE) aims to determine whether two given concepts are related. Existing methods only consider the pairwise relationship between concepts, while overlooking the higher-order relationship that could be encoded in a concept-level graph structure. We discover that this underlying graph satisfies a set of intrinsic properties of CRE, including reflexivity, commutativity, and transitivity. In this paper, we formalize the CRE properties and introduce a graph structure named ConcreteGraph. To address the data scarcity issue in CRE, we introduce a novel data augmentation approach to sample new concept pairs from the graph. As it is intractable for data augmentation to fully capture the structural information of the ConcreteGraph due to a large amount of potential concept pairs, we further introduce a novel Graph Component Contrastive Learning framework to implicitly learn the complete structure of the ConcreteGraph. Empirical results on three datasets show significant improvement over the state-of-the-art model. Detailed ablation studies demonstrate that our proposed approach can effectively capture the high-order relationship among concepts.

        ----

        ## [1499] HybridPrompt: Bridging Language Models and Human Priors in Prompt Tuning for Visual Question Answering

        **Authors**: *Zhiyuan Ma, Zhihuan Yu, Jianjun Li, Guohui Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26569](https://doi.org/10.1609/aaai.v37i11.26569)

        **Abstract**:

        Visual Question Answering (VQA) aims to answer the natural language question about a given image by understanding multimodal content. However, the answer quality of most existing visual-language pre-training (VLP) methods is still limited, mainly due to: (1) Incompatibility. Upstream pre-training tasks are generally incompatible with downstream question answering tasks, which makes the knowledge from the language model not well transferable to downstream tasks, and greatly limits their performance in few-shot scenarios; (2) Under-fitting. They generally do not integrate human priors to compensate for universal knowledge from language models, so as to fit the challenging VQA problem and generate reliable answers. To address these issues, we propose HybridPrompt, a cloze- and verify-style hybrid prompt framework with bridging language models and human priors in prompt tuning for VQA. Specifically, we first modify the input questions into the cloze-style prompts to narrow the gap between upstream pre-training tasks and downstream VQA task, which ensures that the universal knowledge in the language model can be better transferred to subsequent human prior-guided prompt tuning. Then, we imitate the cognitive process of human brain to introduce topic and sample related priors to construct a dynamic learnable prompt template for human prior-guided prompt learning. Finally, we add fixed-length learnable free-parameters to further enhance the generalizability and scalability of prompt learning in the VQA model. Experimental results verify the effectiveness of HybridPrompt, showing that it achieves competitive performance against previous methods on widely-used VQAv2 dataset and obtains new state-of-the-art results. Our code is released at: https://github.com/zhizhi111/hybrid.

        ----

        ## [1500] Inferential Knowledge-Enhanced Integrated Reasoning for Video Question Answering

        **Authors**: *Jianguo Mao, Wenbin Jiang, Hong Liu, Xiangdong Wang, Yajuan Lyu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26570](https://doi.org/10.1609/aaai.v37i11.26570)

        **Abstract**:

        Recently, video question answering has attracted growing attention. It involves answering a question based on a fine-grained understanding of video multi-modal information. Most existing methods have successfully explored the deep understanding of visual modality. We argue that a deep understanding of linguistic modality is also essential for answer reasoning, especially for videos that contain character dialogues. To this end, we propose an Inferential Knowledge-Enhanced Integrated Reasoning method. Our method consists of two main components: 1) an Inferential Knowledge Reasoner to generate inferential knowledge for linguistic modality inputs that reveals deeper semantics, including the implicit causes, effects, mental states, etc. 2) an Integrated Reasoning Mechanism to enhance video content understanding and answer reasoning by leveraging the generated inferential knowledge. Experimental results show that our method achieves significant improvement on two mainstream datasets. The ablation study further demonstrates the effectiveness of each component of our approach.

        ----

        ## [1501] AUC Maximization for Low-Resource Named Entity Recognition

        **Authors**: *Ngoc Dang Nguyen, Wei Tan, Lan Du, Wray L. Buntine, Richard Beare, Changyou Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26571](https://doi.org/10.1609/aaai.v37i11.26571)

        **Abstract**:

        Current work in named entity recognition (NER) uses either cross entropy (CE) or conditional random fields (CRF) as the objective/loss functions to optimize the underlying NER model. Both of these traditional objective functions for the NER problem generally produce adequate performance when the data distribution is balanced and there are sufficient annotated training examples. But since NER is inherently an imbalanced tagging problem, the model performance under the low-resource settings could suffer using these standard objective functions. Based on  recent advances in area under the ROC curve (AUC) maximization, we propose to optimize the NER model by maximizing the AUC score. We give evidence that by simply combining two binary-classifiers that maximize the AUC score, significant performance improvement over traditional loss functions is achieved under low-resource NER settings. We also conduct extensive experiments to demonstrate the advantages of our method under the low-resource and highly-imbalanced data distribution settings. To the best of our knowledge, this is the first work that brings AUC maximization to the NER setting. Furthermore, we show that our method is agnostic to different types of NER embeddings, models and domains. The code of this work is available at https://github.com/dngu0061/NER-AUC-2T.

        ----

        ## [1502] Unveiling the Black Box of PLMs with Semantic Anchors: Towards Interpretable Neural Semantic Parsing

        **Authors**: *Lunyiu Nie, Jiuding Sun, Yanlin Wang, Lun Du, Shi Han, Dongmei Zhang, Lei Hou, Juanzi Li, Jidong Zhai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26572](https://doi.org/10.1609/aaai.v37i11.26572)

        **Abstract**:

        The recent prevalence of pretrained language models (PLMs) has dramatically shifted the paradigm of semantic parsing, where the mapping from natural language utterances to structured logical forms is now formulated as a Seq2Seq task. Despite the promising performance, previous PLM-based approaches often suffer from hallucination problems due to their negligence of the structural information contained in the sentence, which essentially constitutes the key semantics of the logical forms. Furthermore, most works treat PLM as a black box in which the generation process of the target logical form is hidden beneath the decoder modules, which greatly hinders the model's intrinsic interpretability. To address these two issues, we propose to incorporate the current PLMs with a hierarchical decoder network. By taking the first-principle structures as the semantic anchors, we propose two novel intermediate supervision tasks, namely Semantic Anchor Extraction and Semantic Anchor Alignment, for training the hierarchical decoders and probing the model intermediate representations in a self-adaptive manner alongside the fine-tuning process. We conduct intensive experiments on several semantic parsing benchmarks and demonstrate that our approach can consistently outperform the baselines. More importantly, by analyzing the intermediate representations of the hierarchical decoders, our approach also makes a huge step toward the interpretability of PLMs in the domain of semantic parsing.

        ----

        ## [1503] Towards a Holistic Understanding of Mathematical Questions with Contrastive Pre-training

        **Authors**: *Yuting Ning, Zhenya Huang, Xin Lin, Enhong Chen, Shiwei Tong, Zheng Gong, Shijin Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26573](https://doi.org/10.1609/aaai.v37i11.26573)

        **Abstract**:

        Understanding mathematical questions effectively is a crucial task, which can benefit many applications, such as difficulty estimation. Researchers have drawn much attention to designing pre-training models for question representations due to the scarcity of human annotations (e.g., labeling difficulty). However, unlike general free-format texts (e.g., user comments), mathematical questions are generally designed with explicit purposes and mathematical logic, and usually consist of more complex content, such as formulas, and related mathematical knowledge (e.g., Function). Therefore, the problem of holistically representing mathematical questions remains underexplored. To this end, in this paper, we propose a novel contrastive pre-training approach for mathematical question representations, namely QuesCo, which attempts to bring questions with more similar purposes closer. Specifically, we first design two-level question augmentations, including content-level and structure-level, which generate literally diverse question pairs with similar purposes. Then, to fully exploit hierarchical information of knowledge concepts, we propose a knowledge hierarchy-aware rank strategy (KHAR), which ranks the similarities between questions in a fine-grained manner. Next, we adopt a ranking contrastive learning task to optimize our model based on the augmented and ranked questions. We conduct extensive experiments on two real-world mathematical datasets. The experimental results demonstrate the effectiveness of our model.

        ----

        ## [1504] Improving the Cross-Lingual Generalisation in Visual Question Answering

        **Authors**: *Farhad Nooralahzadeh, Rico Sennrich*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26574](https://doi.org/10.1609/aaai.v37i11.26574)

        **Abstract**:

        While several benefits were realized for multilingual vision-language pretrained models, recent benchmarks across various tasks and languages showed poor cross-lingual generalisation when multilingually pre-trained vision-language models are applied to non-English data, with a large gap between (supervised) English performance and (zero-shot) cross-lingual transfer. In this work, we explore the poor performance of these models on a zero-shot cross-lingual visual question answering (VQA) task, where models are fine-tuned on English visual-question data and evaluated on 7 typologically diverse languages. We improve cross-lingual transfer with three strategies: (1) we introduce a linguistic prior objective to augment the cross-entropy loss with a similarity-based loss to guide the model during training, (2) we learn a task-specific subnetwork that improves cross-lingual generalisation and reduces variance without model modification, (3) we augment training examples using synthetic code-mixing to promote alignment of embeddings between source and target languages. Our experiments on xGQA using the pretrained multilingual multimodal transformers UC2 and M3P demonstrates the consistent effectiveness of the proposed fine-tuning strategy for 7 languages, outperforming existing transfer methods with sparse models.

        ----

        ## [1505] RWEN-TTS: Relation-Aware Word Encoding Network for Natural Text-to-Speech Synthesis

        **Authors**: *Shinhyeok Oh, HyeongRae Noh, Yoonseok Hong, Insoo Oh*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26575](https://doi.org/10.1609/aaai.v37i11.26575)

        **Abstract**:

        With the advent of deep learning, a huge number of text-to-speech (TTS) models which produce human-like speech have emerged. Recently, by introducing syntactic and semantic information w.r.t the input text, various approaches have been proposed to enrich the naturalness and expressiveness of TTS models. Although these strategies showed impressive results, they still have some limitations in utilizing language information. First, most approaches only use graph networks to utilize syntactic and semantic information without considering linguistic features. Second, most previous works do not explicitly consider adjacent words when encoding syntactic and semantic information, even though it is obvious that adjacent words are usually meaningful when encoding the current word. To address these issues, we propose Relation-aware Word Encoding Network (RWEN), which effectively allows syntactic and semantic information based on two modules (i.e., Semantic-level Relation Encoding and Adjacent Word Relation Encoding). Experimental results show substantial improvements compared to previous works.

        ----

        ## [1506] Hierarchical Event Grounding

        **Authors**: *Jiefu Ou, Adithya Pratapa, Rishubh Gupta, Teruko Mitamura*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26576](https://doi.org/10.1609/aaai.v37i11.26576)

        **Abstract**:

        Event grounding aims at linking mention references in text corpora to events from a knowledge base (KB). Previous work on this task focused primarily on linking to a single KB event, thereby overlooking the hierarchical aspects of events. Events in documents are typically described at various levels of spatio-temporal granularity. These hierarchical relations are utilized in downstream tasks of narrative understanding and schema construction. In this work, we present an extension to the event grounding task that requires tackling hierarchical event structures from the KB. Our proposed task involves linking a mention reference to a set of event labels from a subevent hierarchy in the KB. We propose a retrieval methodology that leverages event hierarchy through an auxiliary hierarchical loss. On an automatically created multilingual dataset from Wikipedia and Wikidata, our experiments demonstrate the effectiveness of the hierarchical loss against retrieve and re-rank baselines. Furthermore, we demonstrate the systems' ability to aid hierarchical discovery among unseen events. Code is available at https://github.com/JefferyO/Hierarchical-Event-Grounding

        ----

        ## [1507] RINK: Reader-Inherited Evidence Reranker for Table-and-Text Open Domain Question Answering

        **Authors**: *Eunhwan Park, Sung-Min Lee, Daeryong Seo, Seonhoon Kim, Inho Kang, Seung-Hoon Na*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26577](https://doi.org/10.1609/aaai.v37i11.26577)

        **Abstract**:

        Most approaches used in open-domain question answering on hybrid data that comprises both tabular-and-textual contents are based on a Retrieval-Reader pipeline in which the retrieval module finds relevant  “heterogenous” evidence for a given question and the reader module generates an answer from the retrieved evidence. In this paper, we present a Retriever-Reranker-Reader framework by newly proposing a Reader-INherited evidence reranKer (RINK) where a reranker module is designed by finetuning the reader’s neural architecture based on a simple prompting method. Our underlying assumption of reusing the reader’s module for the reranker is that the reader’s ability to generating an answer from evidence contains the knowledge required for the reranking, because the reranker needs to “read” in-depth a question and evidences more carefully and elaborately than a baseline retriever. Furthermore, we present a simple and effective pretraining method by extensively deploying the commonly used data augmentation methods of cell corruption and cell reordering based on the pretraining tasks - tabular-and-textual entailment and cross-modal masked language modeling. Experimental results on OTT-QA, a large-scale table-and-text open-domain question answering dataset, show that the proposed RINK armed with our pretraining procedure makes improvements over the baseline reranking method and leads to state-of-the-art performance.

        ----

        ## [1508] Relation-Aware Language-Graph Transformer for Question Answering

        **Authors**: *Jinyoung Park, Hyeong Kyu Choi, Juyeon Ko, Hyeon-Jin Park, Ji-Hoon Kim, Jisu Jeong, Kyung-Min Kim, Hyunwoo J. Kim*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26578](https://doi.org/10.1609/aaai.v37i11.26578)

        **Abstract**:

        Question Answering (QA) is a task that entails reasoning over natural language contexts, and many relevant works augment language models (LMs) with graph neural networks (GNNs) to encode the Knowledge Graph (KG) information. However, most existing GNN-based modules for QA do not take advantage of rich relational information of KGs and depend on limited information interaction between the LM and the KG. To address these issues, we propose Question Answering Transformer (QAT), which is designed to jointly reason over language and graphs with respect to entity relations in a unified manner. Specifically, QAT constructs Meta-Path tokens, which learn relation-centric embeddings based on diverse structural and semantic relations. Then, our Relation-Aware Self-Attention module comprehensively integrates different modalities via the Cross-Modal Relative Position Bias, which guides information exchange between relevant entities of different modalities. We validate the effectiveness of QAT on commonsense question answering datasets like CommonsenseQA and OpenBookQA, and on a medical question answering dataset, MedQA-USMLE. On all the datasets, our method achieves state-of-the-art performance. Our code is available at http://github.com/mlvlab/QAT.

        ----

        ## [1509] Multi-Mask Label Mapping for Prompt-Based Learning

        **Authors**: *Jirui Qi, Richong Zhang, Jaein Kim, Junfan Chen, Wenyi Qin, Yongyi Mao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26579](https://doi.org/10.1609/aaai.v37i11.26579)

        **Abstract**:

        Prompt-based Learning has shown significant success in few-shot classification. The mainstream approach is to concatenate a template for the input text to transform the classification task into a cloze-type task where label mapping plays an important role in finding the ground-truth labels.  While current label mapping methods only use the contexts in one single input, it could be crucial if wrong information is contained in the text. Specifically, it is proved in recent work that even the large language models like BERT/RoBERTa make classification decisions heavily dependent on a specific keyword regardless of the task or the context. Such a word is referred to as a lexical cue and if a misleading lexical cue is included in the instance it will lead the model to make a wrong prediction. We propose a multi-mask prompt-based approach with Multi-Mask Label Mapping (MMLM) to reduce the impact of misleading lexical cues by allowing the model to exploit multiple lexical cues. To satisfy the conditions of few-shot learning, an instance augmentation approach for the cloze-type model is proposed and the misleading cues are gradually excluded through training. We demonstrate the effectiveness of MMLM by both theoretical analysis and empirical studies, and show that MMLM outperforms other existing label mapping approaches.

        ----

        ## [1510] SSMI: Semantic Similarity and Mutual Information Maximization Based Enhancement for Chinese NER

        **Authors**: *Pengnian Qi, Biao Qin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26580](https://doi.org/10.1609/aaai.v37i11.26580)

        **Abstract**:

        The Chinese NER task consists of two steps, first determining entity boundaries and then labeling them. Some previous work incorporating related words from pre-trained vocabulary into character-based models has been demonstrated to be effective. However, the number of words that characters can match in the vocabulary is large, and their meanings vary widely. It is unreasonable to concatenate all the matched words into the character's representation without making semantic distinctions. This is because words with different semantics also have distinct vectors by the distributed representation. Moreover, mutual information maximization (MIM) provides a unified way to characterize the correction between different granularity of embeddings, we find it can be used to enhance the features in our task. Consequently, this paper introduces a novel Chinese NER model named SSMI based on semantic similarity and MIM. We first match all the potential word boundaries of the input characters from the pre-trained vocabulary and employ BERT to segment the input sentence to get the segmentation containing these characters. After computing their cosine similarity, we obtain the word boundary with the highest similarity and the word group with similarity score larger than a specific threshold. Then, we concatenate the most relevant word boundaries with character vectors. We further calculate the mutual information maximization of group, character and sentence, respectively. Finally, we feed the result from the above steps to our novel network. The results on four Chinese public NER datasets show that our SSMI achieves state-of-the-art performance.

        ----

        ## [1511] Towards Complex Scenarios: Building End-to-End Task-Oriented Dialogue System across Multiple Knowledge Bases

        **Authors**: *Libo Qin, Zhouyang Li, Qiying Yu, Lehan Wang, Wanxiang Che*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26581](https://doi.org/10.1609/aaai.v37i11.26581)

        **Abstract**:

        With the success of the sequence-to-sequence model, end-to-end task-oriented dialogue systems (EToDs) have obtained remarkable progress. However, most existing EToDs are limited to single KB settings where dialogues can be supported by a single KB, which is still far from satisfying the requirements of some complex applications (multi-KBs setting). In this work, we first empirically show that the existing single-KB EToDs fail to work on multi-KB settings that require models to reason across various KBs. To solve this issue, we take the first step to consider the multi-KBs scenario in EToDs and introduce a KB-over-KB Heterogeneous Graph Attention Network (KoK-HAN) to facilitate model to reason over multiple KBs. The core module is a triple-connection graph interaction layer that can model different granularity levels of interaction information across different KBs (i.e., intra-KB connection, inter-KB connection and dialogue-KB connection). Experimental results confirm the superiority of our model for multiple KBs reasoning.

        ----

        ## [1512] BERT-ERC: Fine-Tuning BERT Is Enough for Emotion Recognition in Conversation

        **Authors**: *Xiangyu Qin, Zhiyu Wu, Tingting Zhang, Yanran Li, Jian Luan, Bin Wang, Li Wang, Jinshi Cui*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26582](https://doi.org/10.1609/aaai.v37i11.26582)

        **Abstract**:

        Previous works on emotion recognition in conversation (ERC) follow a two-step paradigm, which can be summarized as first producing context-independent features via fine-tuning pretrained language models (PLMs) and then analyzing contextual information and dialogue structure information among the extracted features. However, we discover that this paradigm has several limitations. Accordingly, we propose a novel paradigm, i.e., exploring contextual information and dialogue structure information in the fine-tuning step, and adapting the PLM to the ERC task in terms of input text, classification structure, and training strategy. Furthermore, we develop our model BERT-ERC according to the proposed paradigm, which improves ERC performance in three aspects, namely suggestive text, fine-grained classification module, and two-stage training. Compared to existing methods, BERT-ERC achieves substantial improvement on four datasets, indicating its effectiveness and generalization capability. Besides, we also set up the limited resources scenario and the online prediction scenario to approximate real-world scenarios. Extensive experiments demonstrate that the proposed paradigm significantly outperforms the previous one and can be adapted to various scenes.

        ----

        ## [1513] Distantly-Supervised Named Entity Recognition with Adaptive Teacher Learning and Fine-Grained Student Ensemble

        **Authors**: *Xiaoye Qu, Jun Zeng, Daizong Liu, Zhefeng Wang, Baoxing Huai, Pan Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26583](https://doi.org/10.1609/aaai.v37i11.26583)

        **Abstract**:

        Distantly-Supervised Named Entity Recognition (DS-NER) effectively alleviates the data scarcity problem in NER by automatically generating training samples. Unfortunately, the distant supervision may induce noisy labels, thus undermining the robustness of the learned models and restricting the practical application. 
To relieve this problem, recent works adopt self-training teacher-student frameworks to gradually refine the training labels and improve the generalization ability of NER models. However, we argue that the performance of the current self-training frameworks for DS-NER is severely underestimated by their plain designs, including both inadequate student learning and coarse-grained teacher updating. Therefore, in this paper, we make the first attempt to alleviate these issues by proposing:  
(1) adaptive teacher learning comprised of joint training of two teacher-student networks and considering both consistent and inconsistent predictions between two teachers, thus promoting comprehensive student learning. 
(2) fine-grained student ensemble that updates each fragment of the teacher model with a temporal moving average of the corresponding fragment of the student, which enhances consistent predictions on each model fragment against noise. 
To verify the effectiveness of our proposed method, we conduct experiments on four DS-NER datasets. The experimental results demonstrate that our method significantly surpasses previous SOTA methods. The code is available at https://github.com/zenhjunpro/ATSEN.

        ----

        ## [1514] Unsupervised Cross-Domain Rumor Detection with Contrastive Learning and Cross-Attention

        **Authors**: *Hongyan Ran, Caiyan Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26584](https://doi.org/10.1609/aaai.v37i11.26584)

        **Abstract**:

        Massive rumors usually appear along with breaking news or trending topics, seriously hindering the truth. Existing rumor detection methods are mostly focused on the same domain, thus have poor performance in cross-domain scenarios due to domain shift. In this work, we propose an end-to-end instance-wise and prototype-wise contrastive learning model with cross-attention mechanism for cross-domain rumor detection. The model not only performs cross-domain
feature alignment, but also enforces target samples to align with the corresponding prototypes of a given source domain. Since target labels in a target domain are unavailable, we use a clustering-based approach with carefully initialized centers
by a batch of source domain samples to produce pseudo labels. Moreover, we use a cross-attention mechanism on a pair of source data and target data with the same labels to learn domain-invariant representations. Because the samples in a
domain pair tend to express similar semantic patterns especially on the people’s attitudes (e.g., supporting or denying) towards the same category of rumors, the discrepancy between a pair of source domain and target domain will be decreased. We conduct experiments on four groups of cross-domain datasets and show that our proposed model achieves state-of-the-art performance.

        ----

        ## [1515] Prompting Neural Machine Translation with Translation Memories

        **Authors**: *Abudurexiti Reheman, Tao Zhou, Yingfeng Luo, Di Yang, Tong Xiao, Jingbo Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26585](https://doi.org/10.1609/aaai.v37i11.26585)

        **Abstract**:

        Improving machine translation (MT) systems with translation memories (TMs) is of great interest to practitioners in the MT community. However, previous approaches require either a significant update of the model architecture and/or additional training efforts to make the models well-behaved when TMs are taken as additional input. In this paper, we present a simple but effective method to introduce TMs into neural machine translation (NMT) systems. Specifically, we treat TMs as prompts to the NMT model at test time, but leave the training process unchanged. The result is a slight update of an existing NMT system, which can be implemented in a few hours by anyone who is familiar with NMT. Experimental results on several datasets demonstrate that our system significantly outperforms strong baselines.

        ----

        ## [1516] Improving Interpretability via Explicit Word Interaction Graph Layer

        **Authors**: *Arshdeep Sekhon, Hanjie Chen, Aman Shrivastava, Zhe Wang, Yangfeng Ji, Yanjun Qi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26586](https://doi.org/10.1609/aaai.v37i11.26586)

        **Abstract**:

        Recent NLP literature has seen growing interest in improving model interpretability. Along this direction, we propose a trainable neural network layer that learns a global interaction graph between words and then selects more informative words using the learned word interactions. Our layer, we call WIGRAPH, can plug into any neural network-based NLP text classifiers right after its word embedding layer. Across multiple SOTA NLP models and various NLP datasets, we demonstrate that adding the WIGRAPH layer substantially improves NLP models' interpretability and enhances models' prediction performance at the same time.

        ----

        ## [1517] Rephrasing the Reference for Non-autoregressive Machine Translation

        **Authors**: *Chenze Shao, Jinchao Zhang, Jie Zhou, Yang Feng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26587](https://doi.org/10.1609/aaai.v37i11.26587)

        **Abstract**:

        Non-autoregressive neural machine translation (NAT) models suffer from the multi-modality problem that there may exist multiple possible translations of a source sentence, so the reference sentence may be inappropriate for the training when the NAT output is closer to other translations. In response to this problem, we introduce a rephraser to provide a better training target for NAT by rephrasing the reference sentence according to the NAT output. As we train NAT based on the rephraser output rather than the reference sentence, the rephraser output should fit well with the NAT output and not deviate too far from the reference, which can be quantified as reward functions and optimized by reinforcement learning. Experiments on major WMT benchmarks and NAT baselines show that our approach consistently improves the translation quality of NAT. Specifically, our best variant achieves comparable performance to the autoregressive Transformer, while being 14.7 times more efficient in inference.

        ----

        ## [1518] Drop Clause: Enhancing Performance, Robustness and Pattern Recognition Capabilities of the Tsetlin Machine

        **Authors**: *Jivitesh Sharma, Rohan Kumar Yadav, Ole-Christoffer Granmo, Lei Jiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26588](https://doi.org/10.1609/aaai.v37i11.26588)

        **Abstract**:

        Logic-based machine learning has the crucial advantage of transparency. However, despite significant recent progress, further research is needed to close the accuracy gap between logic-based architectures and deep neural network ones. This paper introduces a novel variant of the Tsetlin machine (TM) that randomly drops clauses, the logical learning element of TMs. In effect, TM with Drop Clause ignores a random selection of the clauses in each epoch, selected according to a predefined probability. In this way, the TM learning phase becomes more diverse. To explore the effects that Drop Clause has on accuracy, training time and robustness, we conduct extensive experiments on nine benchmark datasets in natural language processing (IMDb, R8, R52, MR, and TREC) and image classification (MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100). Our proposed model outperforms baseline machine learning algorithms by a wide margin and achieves competitive performance compared with recent deep learning models, such as BERT-Large and AlexNet-DFA. In brief, we observe up to +10% increase in accuracy and 2x to 4x faster learning than for the standard TM. We visualize the patterns learnt by Drop Clause TM in the form of heatmaps and show evidence of the ability of drop clause to learn more unique and discriminative patterns. We finally evaluate how Drop Clause affects learning robustness by introducing corruptions and alterations in the image/language test data, which exposes increased learning robustness.

        ----

        ## [1519] CoP: Factual Inconsistency Detection by Controlling the Preference

        **Authors**: *Shuaijie She, Xiang Geng, Shujian Huang, Jiajun Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26589](https://doi.org/10.1609/aaai.v37i11.26589)

        **Abstract**:

        Abstractive summarization is the process of generating a summary given a document as input. Although significant progress has been made, the factual inconsistency between the document and the generated summary still limits its practical applications. Previous work found that the probabilities assigned by the generation model reflect its preferences for the generated summary, including the preference for factual consistency, and the preference for the language or knowledge prior as well. To separate the preference for factual consistency, we propose an unsupervised framework named CoP by controlling the preference of the generation model with the help of prompt. More specifically, the framework performs an extra inference step in which a text prompt is introduced as an additional input. In this way, another preference is described by the generation probability of this extra inference process. The difference between the above two preferences, i.e. the difference between the probabilities, could be used as measurements for detecting factual inconsistencies. Interestingly, we found that with the properly designed prompt, our framework could evaluate specific preferences and serve as measurements for fine-grained categories of inconsistency, such as entity-related inconsistency, coreference-related inconsistency, etc. Moreover, our framework could also be extended to the supervised setting to learn better prompt from the labeled data as well. Experiments show that our framework achieves new SOTA results on three factual inconsisency detection tasks.

        ----

        ## [1520] Which Shortcut Solution Do Question Answering Models Prefer to Learn?

        **Authors**: *Kazutoshi Shinoda, Saku Sugawara, Akiko Aizawa*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26590](https://doi.org/10.1609/aaai.v37i11.26590)

        **Abstract**:

        Question answering (QA) models for reading comprehension tend to exploit spurious correlations in training sets and thus learn shortcut solutions rather than the solutions intended by QA datasets.
QA models that have learned shortcut solutions can achieve human-level performance in shortcut examples where shortcuts are valid, but these same behaviors degrade generalization potential on anti-shortcut examples where shortcuts are invalid.
Various methods have been proposed to mitigate this problem, but they do not fully take the characteristics of shortcuts themselves into account.
We assume that the learnability of shortcuts, i.e., how easy it is to learn a shortcut, is useful to mitigate the problem.
Thus, we first examine the learnability of the representative shortcuts on extractive and multiple-choice QA datasets.
Behavioral tests using biased training sets reveal that shortcuts that exploit answer positions and word-label correlations are preferentially learned for extractive and multiple-choice QA, respectively.
We find that the more learnable a shortcut is, the flatter and deeper the loss landscape is around the shortcut solution in the parameter space.
We also find that the availability of the preferred shortcuts tends to make the task easier to perform from an information-theoretic viewpoint.
Lastly, we experimentally show that the learnability of shortcuts can be utilized to construct an effective QA training set; the more learnable a shortcut is, the smaller the proportion of anti-shortcut examples required to achieve comparable performance on shortcut and anti-shortcut examples.
We claim that the learnability of shortcuts should be considered when designing mitigation methods.

        ----

        ## [1521] Exploring Faithful Rationale for Multi-Hop Fact Verification via Salience-Aware Graph Learning

        **Authors**: *Jiasheng Si, Yingjie Zhu, Deyu Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26591](https://doi.org/10.1609/aaai.v37i11.26591)

        **Abstract**:

        The opaqueness of the multi-hop fact verification model imposes imperative requirements for explainability. One feasible way is to extract rationales, a subset of inputs, where the performance of prediction drops dramatically when being removed. Though being explainable, most rationale extraction methods for multi-hop fact verification explore the semantic information within each piece of evidence individually, while ignoring the topological information interaction among different pieces of evidence. Intuitively, a faithful rationale bears complementary information being able to extract other rationales through the multi-hop reasoning process. To tackle such disadvantages, we cast explainable multi-hop fact verification as subgraph extraction, which can be solved based on graph convolutional network (GCN) with salience-aware graph learning. In specific, GCN is utilized to incorporate the topological interaction information among multiple pieces of evidence for learning evidence representation. Meanwhile, to alleviate the influence of noisy evidence, the salience-aware graph perturbation is induced into the message passing of GCN. Moreover, the multi-task model with three diagnostic properties of rationale is elaborately designed to improve the quality of an explanation without any explicit annotations. Experimental results on the FEVEROUS benchmark show significant gains over previous state-of-the-art methods for both rationale extraction and fact verification.

        ----

        ## [1522] A Speaker Turn-Aware Multi-Task Adversarial Network for Joint User Satisfaction Estimation and Sentiment Analysis

        **Authors**: *Kaisong Song, Yangyang Kang, Jiawei Liu, Xurui Li, Changlong Sun, Xiaozhong Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26592](https://doi.org/10.1609/aaai.v37i11.26592)

        **Abstract**:

        User Satisfaction Estimation is an important task and increasingly being applied in goal-oriented dialogue systems to estimate whether the user is satisfied with the service. It is observed that whether the user’s needs are met often triggers various sentiments, which can be pertinent to the successful estimation of user satisfaction, and vice versa. Thus, User Satisfaction Estimation (USE) and Sentiment Analysis (SA) should be treated as a joint, collaborative effort, considering the strong connections between the sentiment states of speakers and the user satisfaction. Existing joint learning frameworks mainly unify the two highly pertinent tasks over cascade or shared-bottom implementations, however they fail to distinguish task-specific and common features, which will produce sub-optimal utterance representations for downstream tasks. In this paper, we propose a novel Speaker Turn-Aware Multi-Task Adversarial Network (STMAN) for dialogue-level USE and utterance-level SA. Specifically, we first introduce a multi-task adversarial strategy which trains a task discriminator to make utterance representation more task-specific, and then utilize a speaker-turn aware multi-task interaction strategy to extract the common features which are complementary to each task. Extensive experiments conducted on two real-world service dialogue datasets show that our model outperforms several state-of-the-art methods.

        ----

        ## [1523] A Latent-Variable Model for Intrinsic Probing

        **Authors**: *Karolina Stanczak, Lucas Torroba Hennigen, Adina Williams, Ryan Cotterell, Isabelle Augenstein*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26593](https://doi.org/10.1609/aaai.v37i11.26593)

        **Abstract**:

        The success of pre-trained contextualized representations has prompted researchers to analyze them for the presence of linguistic information. 
Indeed, it is natural to assume that these pre-trained representations do encode some level of linguistic knowledge as they have brought about large empirical improvements on a wide variety of NLP tasks, which suggests they are learning true linguistic generalization.
In this work, we focus on intrinsic probing, an analysis technique where the goal is not only to identify whether a representation encodes a linguistic attribute but also to pinpoint where this attribute is encoded.
We propose a novel latent-variable formulation for constructing intrinsic probes and derive a tractable variational approximation to the log-likelihood.
Our results show that our model is versatile and yields tighter mutual information estimates than two intrinsic probes previously proposed in the literature.
Finally, we find empirical evidence that pre-trained representations 
develop a cross-lingually entangled notion of morphosyntax.

        ----

        ## [1524] Towards Diverse, Relevant and Coherent Open-Domain Dialogue Generation via Hybrid Latent Variables

        **Authors**: *Bin Sun, Yitong Li, Fei Mi, Weichao Wang, Yiwei Li, Kan Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26594](https://doi.org/10.1609/aaai.v37i11.26594)

        **Abstract**:

        Conditional variational models, using either continuous or discrete latent variables, are powerful for open-domain dialogue response generation. However, previous works show that continuous latent variables tend to reduce the coherence of generated responses. In this paper, we also found that discrete latent variables have difficulty capturing more diverse expressions. To tackle these problems, we combine the merits of both continuous and discrete latent variables and propose a Hybrid Latent Variable (HLV) method. Specifically, HLV constrains the global semantics of responses through discrete latent variables and enriches responses with continuous latent variables. Thus, we diversify the generated responses while maintaining relevance and coherence. In addition, we propose Conditional Hybrid Variational Transformer (CHVT) to construct and to utilize HLV with transformers for dialogue generation. Through fine-grained symbolic-level semantic information and additive Gaussian mixing, we construct the distribution of continuous variables, prompting the generation of diverse expressions. Meanwhile, to maintain the relevance and coherence, the discrete latent variable is optimized by self-separation training. Experimental results on two dialogue generation datasets (DailyDialog and Opensubtitles) show that CHVT is superior to traditional transformer-based variational mechanism w.r.t. diversity, relevance and coherence metrics. Moreover, we also demonstrate the benefit of applying HLV to fine-tuning two pre-trained dialogue models (PLATO and BART-base).

        ----

        ## [1525] ConvNTM: Conversational Neural Topic Model

        **Authors**: *Hongda Sun, Quan Tu, Jinpeng Li, Rui Yan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26595](https://doi.org/10.1609/aaai.v37i11.26595)

        **Abstract**:

        Topic models have been thoroughly investigated for multiple years due to their great potential in analyzing and understanding texts. Recently, researchers combine the study of topic models with deep learning techniques, known as Neural Topic Models (NTMs). However, existing NTMs are mainly tested based on general document modeling without considering different textual analysis scenarios. We assume that there are different characteristics to model topics in different textual analysis tasks. In this paper, we propose a Conversational Neural Topic Model (ConvNTM) designed in particular for the conversational scenario. Unlike the general document topic modeling, a conversation session lasts for multiple turns: each short-text utterance complies with a single topic distribution and these topic distributions are dependent across turns. Moreover, there are roles in conversations, a.k.a., speakers and addressees. Topic distributions are partially determined by such roles in conversations. We take these factors into account to model topics in conversations via the multi-turn and multi-role formulation. We also leverage the word co-occurrence relationship as a new training objective to further improve topic quality. Comprehensive experimental results based on the benchmark datasets demonstrate that our proposed ConvNTM achieves the best performance both in topic modeling and in typical downstream tasks within conversational research (i.e., dialogue act classification and dialogue response generation).

        ----

        ## [1526] Contrastive Learning Reduces Hallucination in Conversations

        **Authors**: *Weiwei Sun, Zhengliang Shi, Shen Gao, Pengjie Ren, Maarten de Rijke, Zhaochun Ren*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26596](https://doi.org/10.1609/aaai.v37i11.26596)

        **Abstract**:

        Pre-trained language models (LMs) store knowledge in their parameters and can generate informative responses when used in conversational systems. However, LMs suffer from the problem of “hallucination:” they may generate plausible-looking statements that are irrelevant or factually incorrect. To address this problem, we propose a contrastive learning scheme, named MixCL. A novel mixed contrastive objective is proposed to explicitly optimize the implicit knowledge elicitation process of LMs, and thus reduce their hallucination in conversations. We also examine negative sampling strategies of retrieved hard negatives and model-generated negatives. We conduct experiments on Wizard-of-Wikipedia, a public, open-domain knowledge-grounded dialogue benchmark, and assess the effectiveness of MixCL. MixCL effectively reduces the hallucination of LMs in conversations and achieves the highest performance among LM-based dialogue agents in terms of relevancy and factuality. We show that MixCL achieves comparable performance to state-of-the-art KB-based approaches while enjoying notable advantages in terms of efficiency and scalability.

        ----

        ## [1527] Revisiting Denoising Diffusion Probabilistic Models for Speech Enhancement: Condition Collapse, Efficiency and Refinement

        **Authors**: *Wenxin Tai, Fan Zhou, Goce Trajcevski, Ting Zhong*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26597](https://doi.org/10.1609/aaai.v37i11.26597)

        **Abstract**:

        Recent literature has shown that denoising diffusion probabilistic models (DDPMs) can be used to synthesize high-fidelity samples with a competitive (or sometimes better) quality than previous state-of-the-art approaches. However, few attempts have been made to apply DDPM for the speech enhancement task. The reported performance of the existing works is relatively poor and significantly inferior to other generative methods. In this work, we first reveal the difficulties in applying existing diffusion models to the field of speech enhancement. Then we introduce DR-DiffuSE, a simple and effective framework for speech enhancement using conditional diffusion models. We present three strategies (two in diffusion training and one in reverse sampling) to tackle the condition collapse and guarantee the sufficient use of condition information. For efficiency, we introduce the fast sampling technique to reduce the sampling process into several steps and exploit a refinement network to calibrate the defective speech. Our proposed method achieves the state-of-the-art performance to the GAN-based model and shows a significant improvement over existing DDPM-based algorithms.

        ----

        ## [1528] SlideVQA: A Dataset for Document Visual Question Answering on Multiple Images

        **Authors**: *Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, Kuniko Saito*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26598](https://doi.org/10.1609/aaai.v37i11.26598)

        **Abstract**:

        Visual question answering on document images that contain textual, visual, and layout information, called document VQA, has received much attention recently. Although many datasets have been proposed for developing document VQA systems, most of the existing datasets focus on understanding the content relationships within a single image and not across multiple images. In this study, we propose a new multi-image document VQA dataset, SlideVQA, containing 2.6k+ slide decks composed of 52k+ slide images and 14.5k questions about a slide deck. SlideVQA requires complex reasoning, including single-hop, multi-hop, and numerical reasoning, and also provides annotated arithmetic expressions of numerical answers for enhancing the ability of numerical reasoning. Moreover, we developed a new end-to-end document VQA model that treats evidence selection and question answering as a unified sequence-to-sequence format. Experiments on SlideVQA show that our model outperformed existing state-of-the-art QA models, but that it still has a large gap behind human performance. We believe that our dataset will facilitate research on document VQA.

        ----

        ## [1529] Reducing Sentiment Bias in Pre-trained Sentiment Classification via Adaptive Gumbel Attack

        **Authors**: *Jiachen Tian, Shizhan Chen, Xiaowang Zhang, Xin Wang, Zhiyong Feng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26599](https://doi.org/10.1609/aaai.v37i11.26599)

        **Abstract**:

        Pre-trained language models (PLMs) have recently enabled rapid progress on sentiment classification under the pre-train and fine-tune paradigm, where the fine-tuning phase aims to transfer the factual knowledge learned by PLMs to sentiment classification. However, current fine-tuning methods ignore the risk that PLMs cause the problem of sentiment bias, that is, PLMs tend to inject positive or negative sentiment from the contextual information of certain entities (or aspects) into their word embeddings, leading them to establish spurious correlations with labels. In this paper, we propose an adaptive Gumbel-attacked classifier that immunes sentiment bias from an adversarial-attack perspective. Due to the complexity and diversity of sentiment bias, we construct multiple Gumbel-attack expert networks to generate various noises from mixed Gumbel distribution constrained by mutual information minimization, and design an adaptive training framework to synthesize complex noise by confidence-guided controlling the number of expert networks. Finally, we capture these noises that effectively simulate sentiment bias based on the feedback of the classifier, and then propose a multi-channel parameter updating algorithm to strengthen the classifier to recognize these noises by fusing the parameters between the classifier and each expert network. Experimental results illustrate that our method significantly reduced sentiment bias and improved the performance of sentiment classification.

        ----

        ## [1530] Latent Constraints on Unsupervised Text-Graph Alignment with Information Asymmetry

        **Authors**: *Jidong Tian, Wenqing Chen, Yitian Li, Caoyun Fan, Hao He, Yaohui Jin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26600](https://doi.org/10.1609/aaai.v37i11.26600)

        **Abstract**:

        Unsupervised text-graph alignment (UTGA) is a fundamental task that bidirectionally generates texts and graphs without parallel data. Most available models of UTGA suffer from information asymmetry, a common phenomenon that texts and graphs include additional information invisible to each other. On the one hand, these models fail to supplement asymmetric information effectively due to the lack of ground truths. On the other hand, it is challenging to indicate asymmetric information with explicit indicators because it cannot be decoupled from the data directly. To address the challenge posed by information asymmetry, we propose the assumption that asymmetric information is encoded in unobservable latent variables and only affects the one-way generation processes. These latent variables corresponding to asymmetric information should obey prior distributions recovered approximately from original data. Therefore, we first propose a taxonomy of the latent variable that classifies the latent variable into transferrable (TV) and non-transferable (NTV) variables and further distinguish NTV as the dependent variable (DV) and the independent variable (IV). Next, we propose three latent VAE-based regularizations on TV, DV, and IV to constrain their distributions to well-designed prior distributions to introduce asymmetric information into models and enhance the preservation of shared contents. Finally, we impose the three proposed constraints on a cycle-consistent learning framework, back-translation (BT), named ConstrainedBT. Experimental results on three UTGA tasks demonstrate the effectiveness of ConstrainedBT on the information-asymmetric challenge.

        ----

        ## [1531] M-sense: Modeling Narrative Structure in Short Personal Narratives Using Protagonist's Mental Representations

        **Authors**: *Prashanth Vijayaraghavan, Deb Roy*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26601](https://doi.org/10.1609/aaai.v37i11.26601)

        **Abstract**:

        Narrative is a ubiquitous component of human communication. Understanding its structure plays a critical role in a wide variety of applications, ranging from simple comparative analyses to enhanced narrative retrieval, comprehension, or reasoning capabilities. Prior research in narratology has highlighted the importance of studying the links between cognitive and linguistic aspects of narratives for effective comprehension. This interdependence is related to the textual semantics and mental language in narratives, referring to characters' motivations, feelings or emotions, and beliefs. However, this interdependence is hardly explored for modeling narratives. In this work, we propose the task of automatically detecting prominent elements of the narrative structure by analyzing the role of characters' inferred mental state along with linguistic information at the syntactic and semantic levels. We introduce a STORIES dataset of short personal narratives containing manual annotations of key elements of narrative structure, specifically climax and resolution. To this end, we implement a computational model that leverages the protagonist's mental state information obtained from a pre-trained model trained on social commonsense knowledge and integrates their representations with contextual semantic embed-dings using a multi-feature fusion approach. Evaluating against prior zero-shot and supervised baselines, we find that our model is able to achieve significant improvements in the task of identifying climax and resolution.

        ----

        ## [1532] Taming Continuous Posteriors for Latent Variational Dialogue Policies

        **Authors**: *Marin Vlastelica, Patrick Ernst, Gyuri Szarvas*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26602](https://doi.org/10.1609/aaai.v37i11.26602)

        **Abstract**:

        Utilizing amortized variational inference for latent-action reinforcement learning (RL) has been shown to be an effective approach in Task-oriented Dialogue (ToD) systems for optimizing dialogue success.Until now, categorical posteriors have been argued to be one of the main drivers of performance. In this work we revisit Gaussian variational posteriors for latent-action RL and show that they can yield even better performance than categoricals. We achieve this by introducing an improved variational inference objective for learning continuous representations without auxiliary learning objectives, which streamlines the training procedure. Moreover, we propose ways to regularize the latent dialogue policy, which helps to retain good response coherence. Using continuous latent representations our model achieves state of the art dialogue success rate on the MultiWOZ benchmark, and also compares well to categorical latent methods in response coherence.

        ----

        ## [1533] Uncertainty-Aware Self-Training for Low-Resource Neural Sequence Labeling

        **Authors**: *Jianing Wang, Chengyu Wang, Jun Huang, Ming Gao, Aoying Zhou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26603](https://doi.org/10.1609/aaai.v37i11.26603)

        **Abstract**:

        Neural sequence labeling (NSL) aims at assigning labels for input language tokens, which covers a broad range of applications, such as named entity recognition (NER) and slot filling, etc. However, the satisfying results achieved by traditional supervised-based approaches heavily depend on the large amounts of human annotation data, which may not be feasible in real-world scenarios due to data privacy and computation efficiency issues. This paper presents SeqUST, a novel uncertain-aware self-training framework for NSL to address the labeled data scarcity issue and to effectively utilize unlabeled data. Specifically, we incorporate Monte Carlo (MC) dropout in Bayesian neural network (BNN) to perform uncertainty estimation at the token level and then select reliable language tokens from unlabeled data based on the model confidence and certainty. A well-designed masked sequence labeling task with a noise-robust loss supports robust training, which aims to suppress the problem of noisy pseudo labels. In addition, we develop a Gaussian-based consistency regularization technique to further improve the model robustness on Gaussian-distributed perturbed representations. This effectively alleviates the over-fitting dilemma originating from pseudo-labeled augmented data. Extensive experiments over six benchmarks demonstrate that our SeqUST framework effectively improves the performance of self-training, and consistently outperforms strong baselines by a large margin in low-resource scenarios.

        ----

        ## [1534] Disentangled CVAEs with Contrastive Learning for Explainable Recommendation

        **Authors**: *Linlin Wang, Zefeng Cai, Gerard de Melo, Zhu Cao, Liang He*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26604](https://doi.org/10.1609/aaai.v37i11.26604)

        **Abstract**:

        Modern recommender systems are increasingly expected to provide informative explanations that enable users to understand the reason for particular recommendations. However, previous methods struggle to interpret the input IDs of user--item pairs in real-world datasets, failing to extract adequate characteristics for controllable generation. To address this issue, we propose disentangled conditional variational autoencoders (CVAEs) for explainable recommendation, which leverage disentangled latent preference factors and guide the explanation generation with the refined condition of CVAEs via a self-regularization contrastive learning loss. Extensive experiments demonstrate that our method generates high-quality explanations and achieves new state-of-the-art results in diverse domains.

        ----

        ## [1535] fmLRE: A Low-Resource Relation Extraction Model Based on Feature Mapping Similarity Calculation

        **Authors**: *Peng Wang, Tong Shao, Ke Ji, Guozheng Li, Wenjun Ke*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26605](https://doi.org/10.1609/aaai.v37i11.26605)

        **Abstract**:

        Low-resource relation extraction (LRE) aims to extract relations from limited labeled corpora.  Existing work takes advantages of self-training or distant supervision to expand the limited labeled data in the data-driven approaches, while the selection bias of pseudo labels may cause the error accumulation in subsequent relation classification. To address this issue, this paper proposes fmLRE, an iterative feedback method based on feature mapping similarity calculation to improve the accuracy of pseudo labels. First, it calculates the similarities between pseudo-label and real-label data of the same category in a feature mapping space based on semantic features of labeled dataset after feature projection. Then, it fine-tunes initial model according to the iterative process of reinforcement learning. Finally, the similarity is used as a threshold for screening high-precision pseudo-labels and the basis for setting different rewards, which also acts as a penalty term for the loss function of relation classifier. Experimental results demonstrate that fmLRE achieves the state-of-the-art performance compared with strong baselines on two public datasets.

        ----

        ## [1536] Towards Reliable Neural Machine Translation with Consistency-Aware Meta-Learning

        **Authors**: *Rongxiang Weng, Qiang Wang, Wensen Cheng, Changfeng Zhu, Min Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26606](https://doi.org/10.1609/aaai.v37i11.26606)

        **Abstract**:

        Neural machine translation (NMT) has achieved remarkable success in producing high-quality translations. However, current NMT systems suffer from a lack of reliability, as their outputs that are often affected by lexical or syntactic changes in inputs, resulting in large variations in quality. This limitation hinders the practicality and trustworthiness of NMT. A contributing factor to this problem is that NMT models trained with the one-to-one paradigm struggle to handle the source diversity phenomenon, where inputs with the same meaning can be expressed differently. In this work, we treat this problem as a bilevel optimization problem and present a consistency-aware meta-learning (CAML) framework derived from the model-agnostic meta-learning (MAML) algorithm to address it. Specifically, the NMT model with CAML (named CoNMT) first learns a consistent meta representation of semantically equivalent sentences in the outer loop. Subsequently, a mapping from the meta representation to the output sentence is learned in the inner loop, allowing the NMT model to translate semantically equivalent sentences to the same target sentence. We conduct experiments on the NIST Chinese to English task, three WMT translation tasks, and the TED M2O task. The results demonstrate that CoNMT effectively improves overall translation quality and reliably handles diverse inputs.

        ----

        ## [1537] Zero-Shot Face-Based Voice Conversion: Bottleneck-Free Speech Disentanglement in the Real-World Scenario

        **Authors**: *Shao-En Weng, Hong-Han Shuai, Wen-Huang Cheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26607](https://doi.org/10.1609/aaai.v37i11.26607)

        **Abstract**:

        Often a face has a voice. Appearance sometimes has a strong relationship with one's voice. In this work, we study how a face can be converted to a voice, which is a face-based voice conversion. Since there is no clean dataset that contains face and speech, voice conversion faces difficult learning and low-quality problems caused by background noise or echo. Too much redundant information for face-to-voice also causes synthesis of a general style of speech. Furthermore, previous work tried to disentangle speech with bottleneck adjustment. However, it is hard to decide on the size of the bottleneck. Therefore, we propose a bottleneck-free strategy for speech disentanglement. To avoid synthesizing the general style of speech, we utilize framewise facial embedding. It applied adversarial learning with a multi-scale discriminator for the model to achieve better quality. In addition, the self-attention module is added to focus on content-related features for in-the-wild data. Quantitative experiments show that our method outperforms previous work.

        ----

        ## [1538] Adversarial Self-Attention for Language Understanding

        **Authors**: *Hongqiu Wu, Ruixue Ding, Hai Zhao, Pengjun Xie, Fei Huang, Min Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26608](https://doi.org/10.1609/aaai.v37i11.26608)

        **Abstract**:

        Deep neural models (e.g. Transformer) naturally learn spurious features, which create a ``shortcut'' between the labels and inputs, thus impairing the generalization and robustness. This paper advances self-attention mechanism to its robust variant for Transformer-based pre-trained language models (e.g. BERT). We propose Adversarial Self-Attention mechanism (ASA), which adversarially biases the attentions to effectively suppress the model reliance on features (e.g. specific keywords) and encourage its exploration of broader semantics. We conduct comprehensive evaluation across a wide range of tasks for both pre-training and fine-tuning stages. For pre-training, ASA unfolds remarkable performance gain compared to naive training for longer steps. For fine-tuning, ASA-empowered models outweigh naive models by a large margin considering both generalization and robustness.

        ----

        ## [1539] See How You Read? Multi-Reading Habits Fusion Reasoning for Multi-Modal Fake News Detection

        **Authors**: *Lianwei Wu, Pusheng Liu, Yanning Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26609](https://doi.org/10.1609/aaai.v37i11.26609)

        **Abstract**:

        The existing approaches based on different neural networks automatically capture and fuse the multimodal semantics of news, which have achieved great success for fake news detection. However, they still suffer from the limitations of both shallow fusion of multimodal features and less attention to the inconsistency between different modalities. To overcome them, we propose multi-reading habits fusion reasoning networks (MRHFR) for multi-modal fake news detection. In MRHFR, inspired by people's different reading habits for multimodal news, we summarize three basic cognitive reading habits and put forward cognition-aware fusion layer to learn the dependencies between multimodal features of news, so as to deepen their semantic-level integration. To explore the inconsistency of different modalities of news, we develop coherence constraint reasoning layer from two perspectives, which first measures the semantic consistency between the comments and different modal features of the news, and then probes the semantic deviation caused by unimodal features to the multimodal news content through constraint strategy. Experiments on two public datasets not only demonstrate that MRHFR not only achieves the excellent performance but also provides a new paradigm for capturing inconsistencies between multi-modal news.

        ----

        ## [1540] Identify Event Causality with Knowledge and Analogy

        **Authors**: *Sifan Wu, Ruihui Zhao, Yefeng Zheng, Jian Pei, Bang Liu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26610](https://doi.org/10.1609/aaai.v37i11.26610)

        **Abstract**:

        Event causality identification (ECI) aims to identify the
causal relationship between events, which plays a crucial role
in deep text understanding. Due to the diversity of real-world
causality events and difficulty in obtaining sufficient training
data, existing ECI approaches have poor generalizability and
struggle to identify the relation between seldom seen events.
In this paper, we propose to utilize both external knowledge
and internal analogy to improve ECI. On the one hand, we
utilize a commonsense knowledge graph called ConceptNet
to enrich the description of an event sample and reveal the
commonalities or associations between different events. On
the other hand, we retrieve similar events as analogy exam-
ples and glean useful experiences from such analogous neigh-
bors to better identify the relationship between a new event
pair. By better understanding different events through exter-
nal knowledge and making an analogy with similar events, we
can alleviate the data sparsity issue and improve model gener-
alizability. Extensive evaluations on two benchmark datasets
show that our model outperforms other baseline methods by
around 18% on the F1-value on average

        ----

        ## [1541] Continual Graph Convolutional Network for Text Classification

        **Authors**: *Tiandeng Wu, Qijiong Liu, Yi Cao, Yao Huang, Xiao-Ming Wu, Jiandong Ding*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26611](https://doi.org/10.1609/aaai.v37i11.26611)

        **Abstract**:

        Graph convolutional network (GCN) has been successfully applied to capture global non-consecutive and long-distance semantic information for text classification. However, while GCN-based methods have shown promising results in offline evaluations, they commonly follow a seen-token-seen-document paradigm by constructing a fixed document-token graph and cannot make inferences on new documents. It is a challenge to deploy them in online systems to infer steaming text data. In this work, we present a continual GCN model (ContGCN) to generalize inferences from observed documents to unobserved documents. Concretely, we propose a new all-token-any-document paradigm to dynamically update the document-token graph in every batch during both the training and testing phases of an online system. Moreover, we design an occurrence memory module and a self-supervised contrastive learning objective to update ContGCN in a label-free manner. A 3-month A/B test on Huawei public opinion analysis system shows ContGCN achieves 8.86% performance gain compared with state-of-the-art methods. Offline experiments on five public datasets also show ContGCN can improve inference quality. The source code will be released at https://github.com/Jyonn/ContGCN.

        ----

        ## [1542] InfoCTM: A Mutual Information Maximization Perspective of Cross-Lingual Topic Modeling

        **Authors**: *Xiaobao Wu, Xinshuai Dong, Thong Nguyen, Chaoqun Liu, Liangming Pan, Anh Tuan Luu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26612](https://doi.org/10.1609/aaai.v37i11.26612)

        **Abstract**:

        Cross-lingual topic models have been prevalent for cross-lingual text analysis by revealing aligned latent topics. However, most existing methods suffer from producing repetitive topics that hinder further analysis and performance decline caused by low-coverage dictionaries. In this paper, we propose the Cross-lingual Topic Modeling with Mutual Information (InfoCTM). Instead of the direct alignment in previous work, we propose a topic alignment with mutual information method. This works as a regularization to properly align topics and prevent degenerate topic representations of words, which mitigates the repetitive topic issue. To address the low-coverage dictionary issue, we further propose a cross-lingual vocabulary linking method that finds more linked cross-lingual words for topic alignment beyond the translations of a given dictionary. Extensive experiments on English, Chinese, and Japanese datasets demonstrate that our method outperforms state-of-the-art baselines, producing more coherent, diverse, and well-aligned topics and showing better transferability for cross-lingual classification tasks.

        ----

        ## [1543] VideoDubber: Machine Translation with Speech-Aware Length Control for Video Dubbing

        **Authors**: *Yihan Wu, Junliang Guo, Xu Tan, Chen Zhang, Bohan Li, Ruihua Song, Lei He, Sheng Zhao, Arul Menezes, Jiang Bian*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26613](https://doi.org/10.1609/aaai.v37i11.26613)

        **Abstract**:

        Video dubbing aims to translate the original speech in a film or television program into the speech in a target language, which can be achieved with a cascaded system consisting of speech recognition, machine translation and speech synthesis. To ensure the translated speech to be well aligned with the corresponding video, the length/duration of the translated speech should be as close as possible to that of the original speech, which requires strict length control. Previous works usually control the number of words or characters generated by the machine translation model to be similar to the source sentence, without considering the isochronicity of speech as the speech duration of words/characters in different languages varies. In this paper, we propose VideoDubber, a machine translation system tailored for the task of video dubbing, which directly considers the speech duration of each token in translation, to match the length of source and target speech. Specifically, we control the speech length of generated sentence by guiding the prediction of each word with the duration information, including the speech duration of itself as well as how much duration is left for the remaining words. We design experiments on four language directions (German -> English, Spanish -> English, Chinese <-> English), and the results show that VideoDubber achieves better length control ability on the generated speech than baseline methods. To make up the lack of real-world datasets, we also construct a real-world test set collected from films to provide comprehensive evaluations on the video dubbing task.

        ----

        ## [1544] Don't Be So Sure! Boosting ASR Decoding via Confidence Relaxation

        **Authors**: *Tomer Wullach, Shlomo E. Chazan*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26614](https://doi.org/10.1609/aaai.v37i11.26614)

        **Abstract**:

        Automatic Speech Recognition (ASR) systems frequently use a search-based decoding strategy aiming to find the best attainable transcript by considering multiple candidates. One prominent speech recognition decoding heuristic is beam search, which seeks the transcript with the greatest likelihood computed using the predicted distribution. While showing substantial performance gains in various tasks, beam search loses some of its effectiveness when the predicted probabilities are highly confident, i.e., the predicted distribution is massed for a single or very few classes. We show that recently proposed Self-Supervised Learning (SSL)-based ASR models tend to yield exceptionally confident predictions that may hamper beam search from truly considering a diverse set of candidates. We perform a layer analysis to reveal and visualize how predictions evolve, and propose a decoding procedure that improves the performance of fine-tuned ASR models. Our proposed approach does not require further training beyond the original fine-tuning, nor additional model parameters. In fact, we find that our proposed method requires significantly less inference computation than current approaches. We propose aggregating the top M layers, potentially leveraging useful information encoded in intermediate layers, and relaxing model confidence. We demonstrate the effectiveness of our approach by conducting an empirical study on varying amounts of labeled resources and different model sizes, showing consistent improvements in particular when applied to low-resource scenarios.

        ----

        ## [1545] AMOM: Adaptive Masking over Masking for Conditional Masked Language Model

        **Authors**: *Yisheng Xiao, Ruiyang Xu, Lijun Wu, Juntao Li, Tao Qin, Tie-Yan Liu, Min Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26615](https://doi.org/10.1609/aaai.v37i11.26615)

        **Abstract**:

        Transformer-based autoregressive (AR) methods have achieved appealing performance for varied sequence-to-sequence generation tasks, e.g., neural machine translation, summarization, and code generation, but suffer from low inference efficiency. To speed up the inference stage, many non-autoregressive (NAR) strategies have been proposed in the past few years. Among them, the conditional masked language model (CMLM) is one of the most versatile frameworks, as it can support many different sequence generation scenarios and achieve very competitive performance on these tasks. In this paper, we further introduce a simple yet effective adaptive masking over masking strategy to enhance the refinement capability of the decoder and make the encoder optimization easier. Experiments on 3 different tasks (neural machine translation, summarization, and code generation) with 15 datasets in total confirm that our proposed simple method achieves significant performance improvement over the strong CMLM model. Surprisingly, our proposed model yields state-of-the-art performance on neural machine translation (34.62 BLEU on WMT16 EN to RO, 34.82 BLEU on WMT16 RO to EN, and 34.84 BLEU on IWSLT De to En) and even better performance than the AR Transformer on 7 benchmark datasets with at least 2.2x speedup. Our code is available at GitHub.

        ----

        ## [1546] Global Mixup: Eliminating Ambiguity with Clustering

        **Authors**: *Xiangjin Xie, Yangning Li, Wang Chen, Kai Ouyang, Zuotong Xie, Haitao Zheng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26616](https://doi.org/10.1609/aaai.v37i11.26616)

        **Abstract**:

        Data augmentation with Mixup has been proven an effective method to regularize the current deep neural networks. Mixup generates virtual samples and corresponding labels simultaneously by linear interpolation. However, the one-stage generation paradigm and the use of linear interpolation have two defects: (1) The label of the generated sample is simply combined from the labels of the original sample pairs without reasonable judgment, resulting in ambiguous labels. (2) Linear combination significantly restricts the sampling space for generating samples. To address these issues, we propose a novel and effective augmentation method, Global Mixup, based on global clustering relationships.
Specifically, we transform the previous one-stage augmentation process into two-stage by decoupling the process of generating virtual samples from the labeling. And for the labels of the generated samples, relabeling is performed based on clustering by calculating the global relationships of the generated samples.
Furthermore, we are no longer restricted to linear relationships, which allows us to generate more reliable virtual samples in a larger sampling space.
Extensive experiments for CNN, LSTM, and BERT on five tasks show that Global Mixup outperforms previous baselines. Further experiments also demonstrate the advantage of Global Mixup in low-resource scenarios.

        ----

        ## [1547] MoEC: Mixture of Expert Clusters

        **Authors**: *Yuan Xie, Shaohan Huang, Tianyu Chen, Furu Wei*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26617](https://doi.org/10.1609/aaai.v37i11.26617)

        **Abstract**:

        Sparsely Mixture of Experts (MoE) has received great interest due to its promising scaling capability with affordable computational overhead. MoE models convert dense layers into sparse experts, and utilize a gated routing network to make experts conditionally activated. However, as the number of experts grows, MoE with outrageous parameters suffers from overfitting and sparse data allocation. Such problems are especially severe on tasks with limited data, thus hindering the progress towards improving performance by scaling up. We verify that there exists a performance upper bound of scaling up sparse MoE. In this work, we propose Mixture of Expert Clusters — a general approach to enable expert layers to learn more diverse and appropriate knowledge by imposing variance-based constraints on the routing stage. Given this, we could further propose a cluster-level expert dropout strategy specifically designed for the expert cluster structure. Our experiments reveal that MoEC could improve performance on machine translation and natural language understanding tasks. MoEC plays a positive role in mitigating overfitting and sparse data allocation problems, thus fully releasing the potential of large-scale sparse models.

        ----

        ## [1548] Factual and Informative Review Generation for Explainable Recommendation

        **Authors**: *Zhouhang Xie, Sameer Singh, Julian J. McAuley, Bodhisattwa Prasad Majumder*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26618](https://doi.org/10.1609/aaai.v37i11.26618)

        **Abstract**:

        Recent models can generate fluent and grammatical synthetic reviews while accurately predicting user ratings. The generated reviews, expressing users' estimated opinions towards related products, are often viewed as natural language ‘rationales’ for the jointly predicted rating. However, previous studies found that existing models often generate repetitive, universally applicable, and generic explanations, resulting in uninformative rationales. Further, our analysis shows that previous models' generated content often contain factual hallucinations. These issues call for novel solutions that could generate both informative and factually grounded explanations. Inspired by recent success in using retrieved content in addition to parametric knowledge for generation, we propose to augment the generator with a personalized retriever, where the retriever's output serves as external knowledge for enhancing the generator. Experiments on Yelp, TripAdvisor, and Amazon Movie Reviews dataset show our model could generate explanations that more reliably entail existing reviews, are more diverse, and are rated more informative by human evaluators.

        ----

        ## [1549] Dialogue Rewriting via Skeleton-Guided Generation

        **Authors**: *Chunlei Xin, Hongyu Lin, Shan Wu, Xianpei Han, Bo Chen, Wen Dai, Shuai Chen, Bin Wang, Le Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26619](https://doi.org/10.1609/aaai.v37i11.26619)

        **Abstract**:

        Dialogue rewriting aims to transform multi-turn, context-dependent dialogues into well-formed, context-independent text for most NLP systems. Previous dialogue rewriting benchmarks and systems assume a fluent and informative utterance to rewrite. Unfortunately, dialogue utterances from real-world systems are frequently noisy and with various kinds of errors that can make them almost uninformative. In this paper, we first present Real-world Dialogue Rewriting Corpus (RealDia), a new benchmark to evaluate how well current dialogue rewriting systems can deal with real-world noisy and uninformative dialogue utterances. RealDia contains annotated multi-turn dialogues from real scenes with ASR errors, spelling errors, redundancies and other noises that are ignored by previous dialogue rewriting benchmarks. We show that previous dialogue rewriting approaches are neither effective nor data-efficient to resolve RealDia. Then this paper presents Skeleton-Guided Rewriter (SGR), which can resolve the task of dialogue rewriting via a skeleton-guided generation paradigm. Experiments show that RealDia is a much more challenging benchmark for real-world dialogue rewriting, and SGR can effectively resolve the task and outperform previous approaches by a large margin.

        ----

        ## [1550] Dialogue State Distillation Network with Inter-slot Contrastive Learning for Dialogue State Tracking

        **Authors**: *Jing Xu, Dandan Song, Chong Liu, Siu Cheung Hui, Fei Li, Qiang Ju, Xiaonan He, Jian Xie*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26620](https://doi.org/10.1609/aaai.v37i11.26620)

        **Abstract**:

        In task-oriented dialogue systems, Dialogue State Tracking (DST) aims to extract users' intentions from the dialogue history. Currently, most existing approaches suffer from error propagation and are unable to dynamically select relevant information when utilizing previous dialogue states. Moreover, the relations between the updates of different slots provide vital clues for DST. However, the existing approaches rely only on predefined graphs to indirectly capture the relations. In this paper, we propose a Dialogue State Distillation Network (DSDN) to utilize relevant information of previous dialogue states and migrate the gap of utilization between training and testing. Thus, it can dynamically exploit previous dialogue states and avoid introducing error propagation simultaneously. Further, we propose an inter-slot contrastive learning loss to effectively capture the slot co-update relations from dialogue context. Experiments are conducted on the widely used MultiWOZ 2.0 and MultiWOZ 2.1 datasets. The experimental results show that our proposed model achieves the state-of-the-art performance for DST.

        ----

        ## [1551] Balanced Meta Learning and Diverse Sampling for Lifelong Task-Oriented Dialogue Systems

        **Authors**: *Qiancheng Xu, Min Yang, Ruifeng Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26621](https://doi.org/10.1609/aaai.v37i11.26621)

        **Abstract**:

        In real-world scenarios, it is crucial to build a lifelong taskoriented dialogue system (TDS) that continually adapts to new knowledge without forgetting previously acquired experiences. Existing approaches mainly focus on mitigating the catastrophic forgetting in lifelong TDS. However, the transfer ability to generalize the accumulated old knowledge to new tasks is underexplored. In this paper, we propose a two-stage lifelong task-oriented dialogue generation method to mitigate catastrophic forgetting and encourage knowledge transfer simultaneously, inspired by the learning process. In the first stage, we learn task-specific masks which adaptively preserve the knowledge of each visited task so as to mitigate catastrophic forgetting. In this stage, we are expected to learn the task-specific knowledge which is tailored for each task. In the second stage, we bring the knowledge from the encountered tasks together and understand thoroughly. To this end, we devise a balanced meta learning strategy for both forward and backward knowledge transfer in the lifelong learning process. In particular, we perform meta-update with a meta-test set sampled from the current training data for forward knowledge transfer. In addition, we employ an uncertainty-based sampling strategy to select and store representative dialogue samples into episodic memory and perform meta-update with a meta-test set sampled from the memory for backward knowledge transfer. With extensive experiments on 29 tasks, we show that MetaLTDS outperforms the strong baselines in terms of both effectiveness and efficiency. For reproducibility, we submit our code at: https: //github.com/travis-xu/MetaLTDS.

        ----

        ## [1552] Selector-Enhancer: Learning Dynamic Selection of Local and Non-local Attention Operation for Speech Enhancement

        **Authors**: *Xinmeng Xu, Weiping Tu, Yuhong Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26622](https://doi.org/10.1609/aaai.v37i11.26622)

        **Abstract**:

        Attention mechanisms, such as local and non-local attention, play a fundamental role in recent deep learning based speech enhancement (SE) systems. However, a natural speech contains many fast-changing and relatively briefly acoustic events, therefore, capturing the most informative speech features by indiscriminately using local and non-local attention is challenged. We observe that the noise type and speech feature vary within a sequence of speech and the local and non-local can respectively process different types of corrupted speech regions. To leverage this, we propose Selector-Enhancer, a dual-attention based convolution neural network (CNN) with a feature-filter that can dynamically select regions from low-resolution speech features and feed them to local or non-local attention operations. In particular, the proposed feature-filter is trained by using reinforcement learning (RL) with a developed difficulty-regulated reward that related to network performance, model complexity and “the difficulty of the SE task”. The results show that our method achieves comparable or superior performance to existing approaches. In particular, Selector-Enhancer is effective for real-world denoising, where the number and types of noise are varies on a single noisy mixture.

        ----

        ## [1553] A Graph Fusion Approach for Cross-Lingual Machine Reading Comprehension

        **Authors**: *Zenan Xu, Linjun Shou, Jian Pei, Ming Gong, Qinliang Su, Xiaojun Quan, Daxin Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26623](https://doi.org/10.1609/aaai.v37i11.26623)

        **Abstract**:

        Although great progress has been made for Machine Reading Comprehension (MRC) in English, scaling out to a large number of languages remains a huge challenge due to the lack of large amounts of annotated training data in non-English languages. To address this challenge, some recent efforts of cross-lingual MRC employ machine translation to transfer knowledge from English to other languages, through either explicit alignment or implicit attention. For effective knowledge transition, it is beneficial to leverage both semantic and syntactic information. However, the existing methods fail to explicitly incorporate syntax information in model learning. Consequently, the models are not robust to errors in alignment and noises in attention. In this work, we propose a novel approach, which jointly models the cross-lingual alignment information and the mono-lingual syntax information using a graph. We develop a series of algorithms, including graph construction, learning, and pre-training. The experiments on two benchmark datasets for cross-lingual MRC show that our approach outperforms all strong baselines, which verifies the effectiveness of syntax information for cross-lingual MRC.

        ----

        ## [1554] Improving Biomedical Entity Linking with Cross-Entity Interaction

        **Authors**: *Zhenran Xu, Yulin Chen, Baotian Hu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26624](https://doi.org/10.1609/aaai.v37i11.26624)

        **Abstract**:

        Biomedical entity linking (EL) is the task of linking mentions in a biomedical document to corresponding entities in a knowledge base (KB). The challenge in biomedical EL lies in leveraging mention context to select the most appropriate entity among possible candidates. Although some EL models achieve competitive results by retrieving candidate entities and then exploiting context to re-rank them, these re-ranking models concatenate mention context with one candidate at a time. They lack fine-grained interaction among candidates, and potentially cannot handle ambiguous mentions when facing candidates both with high lexical similarity. We cope with this issue using a re-ranking model based on prompt tuning, which represents mention context and all candidates at once, letting candidates in comparison attend to each other. We also propose a KB-enhanced self-supervised pretraining strategy. Instead of large-scale pretraining on biomedical EL data in previous work, we use masked language modeling with synonyms from KB. Our method achieves state-of-the-art results on 3 biomedical EL datasets: NCBI disease, BC5CDR and COMETA, showing the effectiveness of cross-entity interaction and KB-enhanced pretraining strategy. Code is available at https://github.com/HITsz-TMG/Prompt-BioEL.

        ----

        ## [1555] Nested Named Entity Recognition as Building Local Hypergraphs

        **Authors**: *Yukun Yan, Bingling Cai, Sen Song*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26625](https://doi.org/10.1609/aaai.v37i11.26625)

        **Abstract**:

        Named entity recognition is a fundamental task in natural language processing. Based on the sequence labeling paradigm for flat named entity recognition, multiple methods have been developed to handle the nested structures. However, they either require fixed recognition order or introduce complex hypergraphs. To tackle this problem, we propose a novel model named Local Hypergraph Builder Network (LHBN) that builds multiple simpler local hypergraphs to capture named entities instead of a single complex full-size hypergraph. The proposed model has three main properties: (1) The named entities that share boundaries are captured in the same local hypergraph. (2) The boundary information is enhanced by building local hypergraphs. (3) The hypergraphs can be built bidirectionally to take advantage of the identification direction preference of different named entities. Experiments illustrate that our model outperforms previous state-of-the-art methods on four widely used nested named entity recognition datasets: ACE04, ACE05, GENIA, and KBP17. The code is available at https://github.com/yanyk13/local-hypergraph-building-network.git.

        ----

        ## [1556] A Domain-Transfer Meta Task Design Paradigm for Few-Shot Slot Tagging

        **Authors**: *Fengyi Yang, Xi Zhou, Yating Yang, Bo Ma, Rui Dong, Abibulla Atawulla*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26626](https://doi.org/10.1609/aaai.v37i11.26626)

        **Abstract**:

        Few-shot slot tagging is an important task in dialogue systems and attracts much attention of researchers. Most previous few-shot slot tagging methods utilize meta-learning procedure for training and strive to construct a large number of different meta tasks to simulate the testing situation of insufficient data. However, there is a widespread phenomenon of overlap slot between two domains in slot tagging. Traditional meta tasks ignore this special phenomenon and cannot simulate such realistic few-shot slot tagging scenarios. It violates the basic principle of meta-learning which the meta task is consistent with the real testing task, leading to historical information forgetting problem. In this paper, we introduce a novel domain-transfer meta task design paradigm to tackle this problem. We distribute a basic domain to each target domain based on the coincidence degree of slot labels between these two domains. Unlike classic meta tasks which only rely on small samples of target domain, our meta tasks aim to correctly infer the class of target domain query samples based on both abundant data in basic domain and scarce data in target domain. To accomplish our meta task, we propose a Task Adaptation Network to effectively transfer the historical information from the basic domain to the target domain. We carry out sufficient experiments on the benchmark slot tagging dataset SNIPS and the name entity recognition dataset NER. Results demonstrate that our proposed model outperforms previous methods and achieves the state-of-the-art performance.

        ----

        ## [1557] Orders Are Unwanted: Dynamic Deep Graph Convolutional Network for Personality Detection

        **Authors**: *Tao Yang, Jinghao Deng, Xiaojun Quan, Qifan Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26627](https://doi.org/10.1609/aaai.v37i11.26627)

        **Abstract**:

        Predicting personality traits based on online posts has emerged as an important task in many fields such as social network analysis. One of the challenges of this task is assembling information from various posts into an overall profile for each user. While many previous solutions simply concatenate the posts into a long text and then encode the text by sequential or hierarchical models, they introduce unwarranted orders for the posts, which may mislead the models. In this paper, we propose a dynamic deep graph convolutional network (D-DGCN) to overcome the above limitation. Specifically, we design a learn-to-connect approach that adopts a dynamic multi-hop structure instead of a deterministic structure, and combine it with the DGCN module to automatically learn the connections between posts. The modules of post encoder, learn-to-connect, and DGCN are jointly trained in an end-to-end manner. Experimental results on the Kaggle and Pandora datasets show the superior performance of D-DGCN to state-of-the-art baselines. Our code is available at https://github.com/djz233/D-DGCN.

        ----

        ## [1558] What Does Your Face Sound Like? 3D Face Shape towards Voice

        **Authors**: *Zhihan Yang, Zhiyong Wu, Ying Shan, Jia Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26628](https://doi.org/10.1609/aaai.v37i11.26628)

        **Abstract**:

        Face-based speech synthesis provides a practical solution to generate voices from human faces. However, directly using 2D face images leads to the problems of uninterpretability and entanglement. In this paper, to address the issues, we introduce 3D face shape which (1) has an anatomical relationship between voice characteristics, partaking in the "bone conduction" of human timbre production, and (2) is naturally independent of irrelevant factors by excluding the blending process. We devise a three-stage framework to generate speech from 3D face shapes. Fully considering timbre production in anatomical and acquired terms, our framework incorporates three additional relevant attributes including face texture, facial features, and demographics. Experiments and subjective tests demonstrate our method can generate utterances matching faces well, with good audio quality and voice diversity. We also explore and visualize how the voice changes with the face. Case studies show that our method upgrades the face-voice inference to personalized custom-made voice creating, revealing a promising prospect in virtual human and dubbing applications.

        ----

        ## [1559] FiTs: Fine-Grained Two-Stage Training for Knowledge-Aware Question Answering

        **Authors**: *Qichen Ye, Bowen Cao, Nuo Chen, Weiyuan Xu, Yuexian Zou*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26629](https://doi.org/10.1609/aaai.v37i11.26629)

        **Abstract**:

        Knowledge-aware question answering (KAQA) requires the model to answer questions over a knowledge base, which is essential for both open-domain QA and domain-specific QA, especially when language models alone cannot provide all the knowledge needed.
Despite the promising result of recent KAQA systems which tend to integrate linguistic knowledge from pre-trained language models (PLM) and factual knowledge from knowledge graphs (KG) to answer complex questions, a bottleneck exists in effectively fusing the representations from PLMs and KGs because of (i) the semantic and distributional gaps between them, and (ii) the difficulties in joint reasoning over the provided knowledge from both modalities.
To address the above two problems, we propose a Fine-grained Two-stage training framework (FiTs) to boost the KAQA system performance: The first stage aims at aligning representations from the PLM and the KG, thus bridging the modality gaps between them, named knowledge adaptive post-training. The second stage, called knowledge-aware fine-tuning, aims to improve the model's joint reasoning ability based on the aligned representations.
In detail,  we fine-tune the post-trained model via two auxiliary self-supervised tasks in addition to the QA supervision.
Extensive experiments demonstrate that our approach achieves state-of-the-art performance on three benchmarks in the commonsense reasoning (i.e., CommonsenseQA, OpenbookQA) and medical question answering (i.e., MedQA-USMILE) domains.

        ----

        ## [1560] On the Calibration and Uncertainty with Pólya-Gamma Augmentation for Dialog Retrieval Models

        **Authors**: *Tong Ye, Shijing Si, Jianzong Wang, Ning Cheng, Zhitao Li, Jing Xiao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26630](https://doi.org/10.1609/aaai.v37i11.26630)

        **Abstract**:

        Deep neural retrieval models have amply demonstrated their power but estimating the reliability of their predictions remains challenging. Most dialog response retrieval models output a single score for a response on how relevant it is to a given question. However, the bad calibration of deep neural network results in various uncertainty for the single score such that the unreliable predictions always misinform user decisions. To investigate these issues, we present an efficient calibration and uncertainty estimation framework PG-DRR for dialog response retrieval models which adds a Gaussian Process layer to a deterministic deep neural network and recovers conjugacy for tractable posterior inference by Pólya-Gamma augmentation. Finally, PG-DRR achieves the lowest empirical calibration error (ECE) in the in-domain datasets and the distributional shift task while keeping R10@1 and MAP performance.

        ----

        ## [1561] Preserve Context Information for Extract-Generate Long-Input Summarization Framework

        **Authors**: *Ruifeng Yuan, Zili Wang, Ziqiang Cao, Wenjie Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26631](https://doi.org/10.1609/aaai.v37i11.26631)

        **Abstract**:

        The Extract-generate framework has been a classic approach for text summarization. As pretrained language models struggling with long-input summarization for their high memory cost, extract-generate framework regains researchers' interests. However, the cost of its effectiveness in dealing with long-input summarization is the loss of context information. In this paper, we present a context-aware extract-generate framework (CAEG) for long-input text summarization. It focuses on preserving both local and global context information in an extract-generate framework with little cost, and can be applied to most of existing extract-generate summarization models. CAEG generates a set of context-related text spans called context prompts for each text snippet and use them to transfer the context information from the extractor and generator. To find such context prompts, we propose to capture the context information based on the interpretation of the extractor, where the text spans having the highest contribution to the extraction decision are considered as containing the richest context information. We evaluate our approach on both long-document and long-dialogue summarization datasets: arXiv and QMSum. The experiment results show that CAEG achieves the-state-of-art result on QMSum and outperforms other extract-generate based models in arXiv.

        ----

        ## [1562] Transferable Post-hoc Calibration on Pretrained Transformers in Noisy Text Classification

        **Authors**: *Jun Zhang, Wen Yao, Xiaoqian Chen, Ling Feng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26632](https://doi.org/10.1609/aaai.v37i11.26632)

        **Abstract**:

        Recent work has demonstrated that pretrained transformers are overconfident in text classification tasks, which can be calibrated by the famous post-hoc calibration method temperature scaling (TS). Character or word spelling mistakes are frequently encountered in real applications and greatly threaten transformer model safety. Research on calibration under noisy settings is rare, and we focus on this direction. Based on a toy experiment, we discover that TS performs poorly when the datasets are perturbed by slight noise, such as swapping the characters, which results in distribution shift. We further utilize two metrics, predictive uncertainty and maximum mean discrepancy (MMD), to measure the distribution shift between clean and noisy datasets, based on which we propose a simple yet effective transferable TS method for calibrating models dynamically. To evaluate the performance of the proposed methods under noisy settings, we construct a benchmark consisting of four noise types and five shift intensities based on the QNLI, AG-News, and Emotion tasks. Experimental results on the noisy benchmark show that (1) the metrics are effective in measuring distribution shift and (2) transferable TS can significantly decrease the expected calibration error (ECE) compared with the competitive baseline ensemble TS by approximately 46.09%.

        ----

        ## [1563] Quantum-Inspired Representation for Long-Tail Senses of Word Sense Disambiguation

        **Authors**: *Junwei Zhang, Ruifang He, Fengyu Guo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26633](https://doi.org/10.1609/aaai.v37i11.26633)

        **Abstract**:

        Data imbalance, also known as the long-tail distribution of data, is an important challenge for data-driven models. In the Word Sense Disambiguation (WSD) task, the long-tail phenomenon of word sense distribution is more common, making it difficult to effectively represent and identify Long-Tail Senses (LTSs). Therefore exploring representation methods that do not rely heavily on the training sample size is an important way to combat LTSs. Considering that many new states, namely superposition states, can be constructed from several known states in quantum mechanics, superposition states provide the possibility to obtain more accurate representations from inferior representations learned from a small sample size. Inspired by quantum superposition states, a representation method in Hilbert space is proposed to reduce the dependence on large sample sizes and thus combat LTSs. We theoretically prove the correctness of the method, and verify its effectiveness under the standard WSD evaluation framework and obtain state-of-the-art performance. Furthermore, we also test on the constructed LTS and the latest cross-lingual datasets, and achieve promising results.

        ----

        ## [1564] MPMQA: Multimodal Question Answering on Product Manuals

        **Authors**: *Liang Zhang, Anwen Hu, Jing Zhang, Shuo Hu, Qin Jin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26634](https://doi.org/10.1609/aaai.v37i11.26634)

        **Abstract**:

        Visual contents, such as illustrations and images, play a big role in product manual understanding. Existing Product Manual Question Answering (PMQA) datasets tend to ignore visual contents and only retain textual parts. 
In this work, to emphasize the importance of multimodal contents, we propose a Multimodal Product Manual Question Answering (MPMQA) task. For each question, MPMQA requires the model not only to process multimodal contents but also to provide multimodal answers. To support MPMQA, a large-scale dataset PM209 is constructed with human annotations, which contains 209 product manuals from 27 well-known consumer electronic brands. Human annotations include 6 types of semantic regions for manual contents and 22,021 pairs of question and answer. Especially, each answer consists of a textual sentence and related visual regions from manuals. Taking into account the length of product manuals and the fact that a question is always related to a small number of pages, MPMQA can be naturally split into two subtasks: retrieving most related pages and then generating multimodal answers. We further propose a unified model that can perform these two subtasks all together and achieve comparable performance with multiple task-specific models. The PM209 dataset is available at https://github.com/AIM3-RUC/MPMQA.

        ----

        ## [1565] Exploring Self-Distillation Based Relational Reasoning Training for Document-Level Relation Extraction

        **Authors**: *Liang Zhang, Jinsong Su, Zijun Min, Zhongjian Miao, Qingguo Hu, Biao Fu, Xiaodong Shi, Yidong Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26635](https://doi.org/10.1609/aaai.v37i11.26635)

        **Abstract**:

        Document-level relation extraction (RE) aims to extract relational triples from a document. One of its primary challenges is to predict implicit relations between entities, which are not explicitly expressed in the document but can usually be extracted through relational reasoning. Previous methods mainly implicitly model relational reasoning through the interaction among entities or entity pairs. However, they suffer from two deficiencies: 1) they often consider only one reasoning pattern, of which coverage on relational triples is limited; 2) they do not explicitly model the process of relational reasoning. In this paper, to deal with the first problem, we propose a document-level RE model with a reasoning module that contains a core unit, the reasoning multi-head self-attention unit. This unit is a variant of the conventional multi-head self-attention and utilizes four attention heads to model four common reasoning patterns, respectively, which can cover more relational triples than previous methods. Then, to address the second issue, we propose a self-distillation training framework, which contains two branches sharing parameters. In the first branch, we first randomly mask some entity pair feature vectors in the document, and then train our reasoning module to infer their relations  by exploiting the feature information of other related entity pairs. By doing so, we can explicitly model the process of relational reasoning. However, because the additional masking operation is not used during testing, it causes an input gap between training and testing scenarios, which would hurt the model performance. To reduce this gap, we perform conventional supervised training without masking operation in the second branch and utilize Kullback-Leibler divergence loss to minimize the difference between the predictions of the two branches. Finally, we conduct comprehensive experiments on three benchmark datasets, of which experimental results demonstrate that our model consistently outperforms all competitive baselines. Our source code is available at https://github.com/DeepLearnXMU/DocRE-SD

        ----

        ## [1566] Multi-Action Dialog Policy Learning from Logged User Feedback

        **Authors**: *Shuo Zhang, Junzhou Zhao, Pinghui Wang, Tianxiang Wang, Zi Liang, Jing Tao, Yi Huang, Junlan Feng*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26636](https://doi.org/10.1609/aaai.v37i11.26636)

        **Abstract**:

        Multi-action dialog policy (MADP), which generates multiple atomic dialog actions per turn, has been widely applied in task-oriented dialog systems to provide expressive and efficient system responses. Existing MADP models usually imitate action combinations from the labeled multi-action dialog samples. Due to data limitations, they generalize poorly toward unseen dialog flows. While reinforcement learning-based methods are proposed to incorporate the service ratings from real users and user simulators as external supervision signals, they suffer from sparse and less credible dialog-level rewards. To cope with this problem, we explore to improve MADPL with explicit and implicit turn-level user feedback received for historical predictions (i.e., logged user feedback) that are cost-efficient to collect and faithful to real-world scenarios. The task is challenging since the logged user feedback provides only partial label feedback limited to the particular historical dialog actions predicted by the agent. To fully exploit such feedback information, we propose BanditMatch, which addresses the task from a feedback-enhanced semi-supervised learning perspective with a hybrid learning objective of SSL and bandit learning. BanditMatch integrates pseudo-labeling methods to better explore the action space through constructing full label feedback. Extensive experiments show that our BanditMatch improves MADPL over the state-of-the-art methods by generating more concise and informative responses. The source code and the appendix of this paper can be obtained from https://github.com/ShuoZhangXJTU/BanditMatch.

        ----

        ## [1567] Improving End-to-End Speech Translation by Leveraging Auxiliary Speech and Text Data

        **Authors**: *Yuhao Zhang, Chen Xu, Bojie Hu, Chunliang Zhang, Tong Xiao, Jingbo Zhu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26637](https://doi.org/10.1609/aaai.v37i11.26637)

        **Abstract**:

        We present a method for introducing a text encoder into pre-trained end-to-end speech translation systems. It enhances the ability of adapting one modality (i.e., source-language speech) to another (i.e., source-language text). Thus, the speech translation model can learn from both unlabeled and labeled data, especially when the source-language text data is abundant. Beyond this, we present a denoising method to build a robust text encoder that can deal with both normal and noisy text data. Our system sets new state-of-the-arts on the MuST-C En-De, En-Fr, and LibriSpeech En-Fr tasks.

        ----

        ## [1568] A Neural Span-Based Continual Named Entity Recognition Model

        **Authors**: *Yunan Zhang, Qingcai Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26638](https://doi.org/10.1609/aaai.v37i11.26638)

        **Abstract**:

        Named Entity Recognition (NER) models capable of Continual Learning (CL) are realistically valuable in areas where entity types continuously increase (e.g., personal assistants). Meanwhile the learning paradigm of NER advances to new patterns such as the span-based methods. However, its potential to CL has not been fully explored. In this paper, we propose SpanKL, a simple yet effective Span-based model with Knowledge distillation (KD) to preserve memories and multi-Label prediction to prevent conflicts in CL-NER. Unlike prior sequence labeling approaches, the inherently independent modeling in span and entity level with the designed coherent optimization on SpanKL promotes its learning at each incremental step and mitigates the forgetting. Experiments on synthetic CL datasets derived from OntoNotes and Few-NERD show that SpanKL significantly outperforms previous SoTA in many aspects, and obtains the smallest gap from CL to the upper bound revealing its high practiced value. The code is available at https://github.com/Qznan/SpanKL.

        ----

        ## [1569] Language Model Pre-training on True Negatives

        **Authors**: *Zhuosheng Zhang, Hai Zhao, Masao Utiyama, Eiichiro Sumita*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26639](https://doi.org/10.1609/aaai.v37i11.26639)

        **Abstract**:

        Discriminative pre-trained language models (PrLMs) learn to predict original texts from intentionally corrupted ones. Taking the former text as positive and the latter as negative samples, the PrLM can be trained effectively for contextualized representation. However, the training of such a type of PrLMs highly relies on the quality of the automatically constructed samples. Existing PrLMs simply treat all corrupted texts as equal negative without any examination, which actually lets the resulting model inevitably suffer from the false negative issue where training is carried out on pseudo-negative data and leads to less efficiency and less robustness in the resulting PrLMs. In this work, on the basis of defining the false negative issue in discriminative PrLMs that has been ignored for a long time, we design enhanced pre-training methods to counteract false negative predictions and encourage pre-training language models on true negatives by correcting the harmful gradient updates subject to false negative predictions. Experimental results on GLUE and SQuAD benchmarks show that our counter-false-negative pre-training methods indeed bring about better performance together with stronger robustness.

        ----

        ## [1570] MCL: Multi-Granularity Contrastive Learning Framework for Chinese NER

        **Authors**: *Shan Zhao, Chengyu Wang, Minghao Hu, Tianwei Yan, Meng Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26640](https://doi.org/10.1609/aaai.v37i11.26640)

        **Abstract**:

        Recently, researchers have applied the word-character lattice framework to integrated word information, which has become very popular for  Chinese named entity recognition (NER). However, prior approaches  fuse word  information by different variants of encoders such as Lattice LSTM or Flat-Lattice Transformer, but are still not data-efficient indeed to fully  grasp the depth interaction of cross-granularity and important word information from the lexicon. In this paper, we go beyond the typical lattice structure and propose a novel  Multi-Granularity Contrastive Learning  framework (MCL), that aims to optimize the inter-granularity  distribution distance and emphasize the critical matched words in the lexicon. By carefully combining cross-granularity contrastive learning and bi-granularity contrastive learning, the network can explicitly leverage lexicon information  on the  initial lattice structure, and further provide more dense interactions of across-granularity, thus significantly improving model performance. Experiments on four Chinese NER datasets show that MCL obtains state-of-the-art results while considering model efficiency. The source code of the proposed method is publicly available at https://github.com/zs50910/MCL

        ----

        ## [1571] Knowledge-Bridged Causal Interaction Network for Causal Emotion Entailment

        **Authors**: *Weixiang Zhao, Yanyan Zhao, Zhuojun Li, Bing Qin*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26641](https://doi.org/10.1609/aaai.v37i11.26641)

        **Abstract**:

        Causal Emotion Entailment aims to identify causal utterances that are responsible for the target utterance with a non-neutral emotion in conversations. Previous works are limited in thorough understanding of the conversational context and accurate reasoning of the emotion cause. To this end, we propose Knowledge-Bridged Causal Interaction Network (KBCIN) with commonsense knowledge (CSK) leveraged as three bridges. Specifically, we construct a conversational graph for each conversation and leverage the event-centered CSK as the semantics-level bridge (S-bridge) to capture the deep inter-utterance dependencies in the conversational context via the CSK-Enhanced Graph Attention module. Moreover, social-interaction CSK serves as emotion-level bridge (E-bridge) and action-level bridge (A-bridge) to connect candidate utterances with the target one, which provides explicit causal clues for the Emotional Interaction module and Actional Interaction module to reason the target emotion. Experimental results show that our model achieves better performance over most baseline models. Our source code is publicly available at https://github.com/circle-hit/KBCIN.

        ----

        ## [1572] Query Your Model with Definitions in FrameNet: An Effective Method for Frame Semantic Role Labeling

        **Authors**: *Ce Zheng, Yiming Wang, Baobao Chang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26642](https://doi.org/10.1609/aaai.v37i11.26642)

        **Abstract**:

        Frame Semantic Role Labeling (FSRL) identifies arguments and labels them with frame semantic roles defined in FrameNet. Previous researches tend to divide FSRL into argument identification and role classification. Such methods usually model role classification as naive multi-class classification and treat arguments individually, which neglects label semantics and interactions between arguments and thus hindering performance and generalization of models. In this paper, we propose a query-based framework named ArGument Extractor with Definitions in FrameNet (AGED) to mitigate these problems. Definitions of frames and frame elements (FEs) in FrameNet can be used to query arguments in text. Encoding text-definition pairs can guide models in learning label semantics and strengthening argument interactions. Experiments show that AGED outperforms previous state-of-the-art by up to 1.3 F1-score in two FrameNet datasets and the generalization power of AGED in zero-shot and fewshot scenarios. Our code and technical appendix is available at https://github.com/PKUnlp-icler/AGED.

        ----

        ## [1573] Event Process Typing via Hierarchical Optimal Transport

        **Authors**: *Bo Zhou, Yubo Chen, Kang Liu, Jun Zhao*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26643](https://doi.org/10.1609/aaai.v37i11.26643)

        **Abstract**:

        Understanding intention behind event processes in texts is important to many applications. One challenging task in this line is event process typing, which aims to tag the process with one action label and one object label describing the overall action of the process and object the process likely affects respectively. To tackle this task, existing methods mainly rely on the matching of the event process level and label level representation, which ignores two important characteristics: Process Hierarchy and Label Hierarchy. In this paper, we propose a Hierarchical Optimal Transport (HOT) method to address the above problem. Specifically, we first explicitly extract the process hierarchy and label hierarchy. Then the HOT optimally matches the two types of hierarchy. Experimental results show that our model outperforms the baseline models, illustrating the effectiveness of our model.

        ----

        ## [1574] Improving Distantly Supervised Relation Extraction by Natural Language Inference

        **Authors**: *Kang Zhou, Qiao Qiao, Yuepei Li, Qi Li*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26644](https://doi.org/10.1609/aaai.v37i11.26644)

        **Abstract**:

        To reduce human annotations for relation extraction (RE) tasks, distantly supervised approaches have been proposed, while struggling with low performance. In this work, we propose a novel DSRE-NLI framework, which considers both distant supervision from existing knowledge bases and indirect supervision from pretrained language models for other tasks. DSRE-NLI energizes an off-the-shelf natural language inference (NLI) engine with a semi-automatic relation verbalization (SARV) mechanism to provide indirect supervision and further consolidates the distant annotations to benefit multi-classification RE models. The NLI-based indirect supervision acquires only one relation verbalization template from humans as a semantically general template for each relationship, and then the template set is enriched by high-quality textual patterns automatically mined from the distantly annotated corpus. With two simple and effective data consolidation strategies, the quality of training data is substantially improved. Extensive experiments demonstrate that the proposed framework significantly improves the SOTA performance (up to 7.73% of F1) on distantly supervised RE benchmark datasets. Our code is available at https://github.com/kangISU/DSRE-NLI.

        ----

        ## [1575] A Generative Approach for Script Event Prediction via Contrastive Fine-Tuning

        **Authors**: *Fangqi Zhu, Jun Gao, Changlong Yu, Wei Wang, Chen Xu, Xin Mu, Min Yang, Ruifeng Xu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26645](https://doi.org/10.1609/aaai.v37i11.26645)

        **Abstract**:

        Script event prediction aims to predict the subsequent event given the context. This requires the capability to infer the correlations between events. Recent works have attempted to improve event correlation reasoning by using pretrained language models and incorporating external knowledge (e.g., discourse relations). Though promising results have been achieved, some challenges still remain. First, the pretrained language models adopted by current works ignore event-level knowledge, resulting in an inability to capture the correlations between events well. Second, modeling correlations between events with discourse relations is limited because it can only capture explicit correlations between events with discourse markers, and cannot capture many implicit correlations. To this end, we propose a novel generative approach for this task, in which a pretrained language model is fine-tuned with an event-centric pretraining objective and predicts the next event within a generative paradigm. Specifically, we first introduce a novel event-level blank infilling strategy as the learning objective to inject event-level knowledge into the pretrained language model, and then design a likelihood-based contrastive loss for fine-tuning the generative model. Instead of using an additional prediction layer, we perform prediction by using sequence likelihoods generated by the generative model. Our approach models correlations between events in a soft way without any external knowledge. The likelihood-based prediction eliminates the need to use additional networks to make predictions and is somewhat interpretable since it scores each word in the event. Experimental results on the multi-choice narrative cloze (MCNC) task demonstrate that our approach achieves better results than other state-of-the-art baselines. Our code will be available at https://github.com/zhufq00/mcnc.

        ----

        ## [1576] KPT: Keyword-Guided Pre-training for Grounded Dialog Generation

        **Authors**: *Qi Zhu, Fei Mi, Zheng Zhang, Yasheng Wang, Yitong Li, Xin Jiang, Qun Liu, Xiaoyan Zhu, Minlie Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26646](https://doi.org/10.1609/aaai.v37i11.26646)

        **Abstract**:

        Incorporating external knowledge into the response generation process is essential to building more helpful and reliable dialog agents. However, collecting knowledge-grounded conversations is often costly, calling for a better pre-trained model for grounded dialog generation that generalizes well w.r.t. different types of knowledge. In this work, we propose KPT (Keyword-guided Pre-Training), a novel self-supervised pre-training method for grounded dialog generation without relying on extra knowledge annotation. Specifically, we use a pre-trained language model to extract the most uncertain tokens in the dialog as keywords. With these keywords, we construct two kinds of knowledge and pre-train a knowledge-grounded response generation model, aiming at handling two different scenarios: (1) the knowledge should be faithfully grounded; (2) it can be selectively used. For the former, the grounding knowledge consists of keywords extracted from the response. For the latter, the grounding knowledge is additionally augmented with keywords extracted from other utterances in the same dialog. Since the knowledge is extracted from the dialog itself, KPT can be easily performed on a large volume and variety of dialogue data.  We considered three data sources (open-domain, task-oriented, conversational QA) with a total of 2.5M dialogues. We conduct extensive experiments on various few-shot knowledge-grounded generation tasks, including grounding on dialog acts, knowledge graphs, persona descriptions, and Wikipedia passages. Our comprehensive experiments and analyses demonstrate that KPT consistently outperforms state-of-the-art methods on these tasks with diverse grounding knowledge.

        ----

        ## [1577] An Ensemble Distillation Framework for Sentence Embeddings with Multilingual Round-Trip Translation

        **Authors**: *Tianyu Zong, Likun Zhang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i11.26647](https://doi.org/10.1609/aaai.v37i11.26647)

        **Abstract**:

        In this work, we propose a novel unsupervised contrastive learning framework to improve state-of-the-art sentence embeddings. First, we train a set of contrastive submodels which take multilingual round-trip translation(RTT) as data augmentation. The RTT naturally changes the length of the same sentence and replaces Synonyms simultaneously. Then we incorporate them into a single model through knowledge distillation. Specifically, it takes an input sentence and predicts the ensemble output of all submodels via a contrastive objective. Thus we preserve nearly the same semantic expressiveness as the ensemble model without increasing the test cost. We evaluate our framework on standard semantic textual similarity (STS) tasks. Experimental results show the advantage of our framework that we achieve an average of 79.27% Spearman's correlation, a 3.02% improvement compared to the previous best results using BERT-base.

        ----

        ## [1578] COSMOS: Catching Out-of-Context Image Misuse Using Self-Supervised Learning

        **Authors**: *Shivangi Aneja, Chris Bregler, Matthias Nießner*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26648](https://doi.org/10.1609/aaai.v37i12.26648)

        **Abstract**:

        Despite the recent attention to DeepFakes, one of the most prevalent ways to mislead audiences on social media is the use of unaltered images in a new but false context. We propose a new method that automatically highlights out-of-context image and text pairs, for assisting fact-checkers. Our key insight is to leverage the grounding of images with text to distinguish out-of-context scenarios that cannot be disambiguated with language alone. We propose a self-supervised training strategy where we only need a set of captioned images. At train time, our method learns to selectively align individual objects in an image with textual claims, without explicit supervision. At test time, we check if both captions correspond to the same object(s) in the image but are semantically different, which allows us to make fairly accurate out-of-context predictions. Our method achieves 85% out-of-context detection accuracy. To facilitate benchmarking of this task, we create a large-scale dataset of 200K images with 450K textual captions from a variety of news websites, blogs, and social media posts

        ----

        ## [1579] Med-EASi: Finely Annotated Dataset and Models for Controllable Simplification of Medical Texts

        **Authors**: *Chandrayee Basu, Rosni Vasu, Michihiro Yasunaga, Qian Yang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26649](https://doi.org/10.1609/aaai.v37i12.26649)

        **Abstract**:

        Automatic medical text simplification can assist providers with patient-friendly communication and make medical texts more accessible, thereby improving health literacy. But curating a quality corpus for this task requires the supervision of medical experts. In this work, we present Med-EASi (Medical dataset for Elaborative and Abstractive Simplification), a uniquely crowdsourced and finely annotated dataset for supervised simplification of short medical texts. Its expert-layman-AI collaborative annotations facilitate controllability over text simplification by marking four kinds of textual transformations: elaboration, replacement, deletion, and insertion. To learn medical text simplification, we fine-tune T5-large with four different styles of input-output combinations, leading to two control-free and two controllable versions of the model. We add two types of controllability into text simplification, by using a multi-angle training approach: position-aware, which uses in-place annotated inputs and outputs, and position-agnostic, where the model only knows the contents to be edited, but not their positions. Our results show that our fine-grained annotations improve learning compared to the unannotated baseline. Furthermore, our position-aware control enhances the model's ability to generate better simplification than the position-agnostic version. The data and code are available at https://github.com/Chandrayee/CTRL-SIMP.

        ----

        ## [1580] On the Challenges of Using Reinforcement Learning in Precision Drug Dosing: Delay and Prolongedness of Action Effects

        **Authors**: *Sumana Basu, Marc-André Legault, Adriana Romero-Soriano, Doina Precup*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26650](https://doi.org/10.1609/aaai.v37i12.26650)

        **Abstract**:

        Drug dosing is an important application of AI, which can be formulated as a Reinforcement Learning (RL) problem. In this paper, we identify two major challenges of using RL for drug dosing: delayed and prolonged effects of administering medications, which break the Markov assumption of the RL framework. We focus on prolongedness and define PAE-POMDP (Prolonged Action Effect-Partially Observable Markov Decision Process), a subclass of POMDPs in which the Markov assumption does not hold specifically due to prolonged effects of actions. Motivated by the pharmacology literature, we propose a simple and effective approach to converting drug dosing PAE-POMDPs into MDPs, enabling the use of the existing RL algorithms to solve such problems. We validate the proposed approach on a toy task, and a challenging glucose control task, for which we devise a clinically-inspired reward function. Our results demonstrate that: (1) the proposed method to restore the Markov assumption leads to significant improvements over a vanilla baseline; (2) the approach is competitive with recurrent policies which may inherently capture the prolonged affect of actions; (3) it is remarkably more time and memory efficient than the recurrent baseline and hence more suitable for real-time dosing control systems; and (4) it exhibits favourable qualitative behavior in our policy analysis.

        ----

        ## [1581] On the Cost of Demographic Parity in Influence Maximization

        **Authors**: *Ruben Becker, Gianlorenzo D'Angelo, Sajjad Ghobadi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26651](https://doi.org/10.1609/aaai.v37i12.26651)

        **Abstract**:

        Modeling and shaping how information spreads through a network is a major research topic in network analysis. While initially the focus has been mostly on efficiency, recently fairness criteria have been taken into account in this setting.
Most work has focused on the maximin criteria however, and thus still different groups can receive very different shares of information. In this work we propose to consider fairness as a notion to be guaranteed by an algorithm rather than as a criterion to be maximized. To this end, we propose three optimization problems that aim at maximizing the overall spread while enforcing strict levels of demographic parity fairness via constraints (either ex-post or ex-ante). The level of fairness hence becomes a user choice rather than a property to be observed upon output. We study this setting from various perspectives.
First, we prove that the cost of introducing demographic parity can be high in terms of both overall spread and computational complexity, i.e., the price of fairness may be unbounded for all three problems and optimal solutions are hard to compute, in some case even approximately or when fairness constraints may be violated. 
For one of our problems, we still design an algorithm with both constant approximation factor and fairness violation.
We also give two heuristics that allow the user to choose the tolerated fairness violation. By means of an extensive experimental study, we show that our algorithms perform well in practice, that is, they achieve the best demographic parity fairness values. For certain instances we additionally even obtain an overall spread comparable to the most efficient algorithms that come without any fairness guarantee, indicating that the empirical price of fairness may actually be small when using our algorithms.

        ----

        ## [1582] Improving Fairness in Information Exposure by Adding Links

        **Authors**: *Ruben Becker, Gianlorenzo D'Angelo, Sajjad Ghobadi*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26652](https://doi.org/10.1609/aaai.v37i12.26652)

        **Abstract**:

        Fairness in influence maximization has been a very active research topic recently. Most works in this context study the question of how to find seeding strategies (deterministic or probabilistic) such that nodes or communities in the network get their fair share of coverage. Different fairness criteria have been used in this context. All these works assume that the entity that is spreading the information has an inherent interest in spreading the information fairly, otherwise why would they want to use the developed fair algorithms? This assumption may however be flawed in reality -- the spreading entity may be purely efficiency-oriented. In this paper we propose to study two optimization problems with the goal to modify the network structure by adding links in such a way that efficiency-oriented information spreading becomes automatically fair. We study the proposed optimization problems both from a theoretical and experimental perspective, that is, we give several hardness and hardness of approximation results, provide efficient algorithms for some special cases, and more importantly provide heuristics for solving one of the problems in practice. In our experimental study we then first compare the proposed heuristics against each other and establish the most successful one. In a second experiment, we then show that our approach can be very successful in practice. That is, we show that already after adding a few edges to the networks the greedy algorithm that purely maximizes spread surpasses all fairness-tailored algorithms in terms of ex-post fairness. Maybe surprisingly, we even show that our approach achieves ex-post fairness values that are comparable or even better than the ex-ante fairness values of the currently most efficient algorithms that optimize ex-ante fairness.

        ----

        ## [1583] A Fair Incentive Scheme for Community Health Workers

        **Authors**: *Avinandan Bose, Tracey Li, Arunesh Sinha, Tien Mai*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26653](https://doi.org/10.1609/aaai.v37i12.26653)

        **Abstract**:

        Community health workers (CHWs) play a crucial role in
the last mile delivery of essential health services to underserved
populations in low-income countries. Many nongovernmental
organizations (NGOs) provide training and
support to enable CHWs to deliver health services to their
communities, with no charge to the recipients of the services.
This includes monetary compensation for the work that
CHWs perform, which is broken down into a series of well defined
tasks. In this work, we partner with a NGO D-Tree
International to design a fair monetary compensation scheme
for tasks performed by CHWs in the semi-autonomous region
of Zanzibar in Tanzania, Africa. In consultation with
stakeholders, we interpret fairness as the equal opportunity
to earn, which means that each CHW has the opportunity to
earn roughly the same total payment over a given T month
period, if the CHW reacts to the incentive scheme almost rationally.
We model this problem as a reward design problem
for a Markov Decision Process (MDP) formulation for the
CHWs’ earning. There is a need for the mechanism to be
simple so that it is understood by the CHWs, thus, we explore
linear and piecewise linear rewards in the CHWs’ measured
units of work. We solve this design problem via a novel
policy-reward gradient result. Our experiments using two real
world parameters from the ground provide evidence of reasonable
incentive output by our scheme.

        ----

        ## [1584] Rehabilitating Homeless: Dataset and Key Insights

        **Authors**: *Anna Bykova, Nikolay Filippov, Ivan P. Yamshchikov*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26654](https://doi.org/10.1609/aaai.v37i12.26654)

        **Abstract**:

        This paper presents a large anonymized dataset of homelessness alongside insights into the data-driven rehabilitation of homeless people. The dataset was gathered by a large non-profit organization working on rehabilitating the homeless for twenty years. This is the first dataset that we know of that contains rich information on thousands of homeless individuals seeking rehabilitation. We show how data analysis can help to make the rehabilitation of homeless people more effective and successful. Thus, we hope this paper alerts the data science community to the problem of homelessness.

        ----

        ## [1585] Counterfactuals for the Future

        **Authors**: *Lucius E. J. Bynum, Joshua R. Loftus, Julia Stoyanovich*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26655](https://doi.org/10.1609/aaai.v37i12.26655)

        **Abstract**:

        Counterfactuals are often described as 'retrospective,' focusing on hypothetical alternatives to a realized past. This description relates to an often implicit assumption about the structure and stability of exogenous variables in the system being modeled --- an assumption that is reasonable in many settings where counterfactuals are used. In this work, we consider cases where we might reasonably make a different assumption about exogenous variables; namely, that the exogenous noise terms of each unit do exhibit some unit-specific structure and/or stability. This leads us to a different use of counterfactuals --- a forward-looking rather than retrospective counterfactual. We introduce "counterfactual treatment choice," a type of treatment choice problem that motivates using forward-looking counterfactuals. We then explore how mismatches between interventional versus forward-looking counterfactual approaches to treatment choice, consistent with different assumptions about exogenous noise, can lead to counterintuitive results.

        ----

        ## [1586] Towards Learning to Discover Money Laundering Sub-network in Massive Transaction Network

        **Authors**: *Ziwei Chai, Yang Yang, Jiawang Dan, Sheng Tian, Changhua Meng, Weiqiang Wang, Yifei Sun*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26656](https://doi.org/10.1609/aaai.v37i12.26656)

        **Abstract**:

        Anti-money laundering (AML) systems play a critical role in safeguarding global economy. As money laundering is considered as one of the top group crimes, there is a crucial need to discover money laundering sub-network behind a particular money laundering transaction for a robust AML system. However, existing rule-based methods for money laundering sub-network discovery is heavily based on domain knowledge and may lag behind the modus operandi of launderers. Therefore, in this work, we first address the money laundering sub-network discovery problem with a neural network based approach, and propose an AML framework AMAP equipped with an adaptive sub-network proposer. In particular, we design an adaptive sub-network proposer guided by a supervised contrastive loss to discriminate money laundering transactions from massive benign transactions. We conduct extensive experiments on real-word datasets in AliPay of Ant Group. The result demonstrates the effectiveness of our AMAP in both money laundering transaction detection and money laundering sub-network discovering. The learned framework which yields money laundering sub-network from massive transaction network leads to a more comprehensive risk coverage and a deeper insight to money laundering strategies.

        ----

        ## [1587] Estimating Geographic Spillover Effects of COVID-19 Policies from Large-Scale Mobility Networks

        **Authors**: *Serina Chang, Damir Vrabac, Jure Leskovec, Johan Ugander*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26657](https://doi.org/10.1609/aaai.v37i12.26657)

        **Abstract**:

        Many policies in the US are determined locally, e.g., at the county-level. Local policy regimes provide flexibility between regions, but may become less effective in the presence of geographic spillovers, where populations circumvent local restrictions by traveling to less restricted regions nearby. Due to the endogenous nature of policymaking, there have been few opportunities to reliably estimate causal spillover effects or evaluate their impact on local policies. In this work, we identify a novel setting and develop a suitable methodology that allow us to make unconfounded estimates of spillover effects of local policies. Focusing on California’s Blueprint for a Safer Economy, we leverage how county-level mobility restrictions were deterministically set by public COVID-19 severity statistics, enabling a regression discontinuity design framework to estimate spillovers between counties. We estimate these effects using a mobility network with billions of timestamped edges and find significant spillover movement, with larger effects in retail, eating places, and gyms. Contrasting local and global policy regimes, our spillover estimates suggest that county-level restrictions are only 54% as effective as statewide restrictions at reducing mobility. However, an intermediate strategy of macro-county restrictions---where we optimize county partitions by solving a minimum k-cut problem on a graph weighted by our spillover estimates---can recover over 90% of statewide mobility reductions, while maintaining substantial flexibility between counties.

        ----

        ## [1588] Seq2Seq Surrogates of Epidemic Models to Facilitate Bayesian Inference

        **Authors**: *Giovanni Charles, Timothy M. Wolock, Peter Winskill, Azra C. Ghani, Samir Bhatt, Seth R. Flaxman*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26658](https://doi.org/10.1609/aaai.v37i12.26658)

        **Abstract**:

        Epidemic models are powerful tools in understanding infectious disease. However, as they increase in size and complexity, they can quickly become computationally intractable. Recent progress in modelling methodology has shown that surrogate models can be used to emulate complex epidemic models with a high-dimensional parameter space. We show that deep sequence-to-sequence (seq2seq) models can serve as accurate surrogates for complex epidemic models with sequence based model parameters, effectively replicating seasonal and long-term transmission dynamics. Once trained, our surrogate can predict scenarios a several thousand times faster than the original model, making them ideal for policy exploration. We demonstrate that replacing a traditional epidemic model with a learned simulator facilitates robust Bayesian inference.

        ----

        ## [1589] Leveraging Old Knowledge to Continually Learn New Classes in Medical Images

        **Authors**: *Evelyn Chee, Mong-Li Lee, Wynne Hsu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26659](https://doi.org/10.1609/aaai.v37i12.26659)

        **Abstract**:

        Class-incremental continual learning is a core step towards developing artificial intelligence systems that can continuously adapt to changes in the environment by learning new concepts without forgetting those previously learned. This is especially needed in the medical domain where continually learning from new incoming data is required to classify an expanded set of diseases. In this work, we focus on how old knowledge can be leveraged to learn new classes without catastrophic forgetting. We propose a framework that comprises of two main components: (1) a dynamic architecture with expanding representations to preserve previously learned features and accommodate new features; and (2) a training procedure alternating between two objectives to balance the learning of new features while maintaining the model’s performance on old classes. Experiment results on multiple medical datasets show that our solution is able to achieve superior performance over state-of-the-art baselines in terms of class accuracy and forgetting.

        ----

        ## [1590] SARAS-Net: Scale and Relation Aware Siamese Network for Change Detection

        **Authors**: *Chao-Peng Chen, Jun-Wei Hsieh, Ping-Yang Chen, Yi-Kuan Hsieh, Bor-Shiun Wang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26660](https://doi.org/10.1609/aaai.v37i12.26660)

        **Abstract**:

        Change detection (CD) aims to find the difference between two images at different times and output a change map to represent whether the region has changed or not. To achieve a better result in generating the change map, many State-of-The-Art (SoTA) methods design a deep learning model that has a powerful discriminative ability. However, these methods still get lower performance because they ignore spatial information and scaling changes between objects, giving rise to blurry boundaries. In addition to these, they also neglect the interactive information of two different images. To alleviate these problems, we propose our network, the Scale and Relation-Aware Siamese Network (SARAS-Net) to deal with this issue. In this paper, three modules are proposed that include relation-aware, scale-aware, and cross-transformer to tackle the problem of scene change detection more effectively.  To verify our model, we tested three public datasets, including LEVIR-CD, WHU-CD, and DSFIN, and obtained SoTA accuracy. Our code is available at https://github.com/f64051041/SARAS-Net.

        ----

        ## [1591] Improving Interpretability of Deep Sequential Knowledge Tracing Models with Question-centric Cognitive Representations

        **Authors**: *Jiahao Chen, Zitao Liu, Shuyan Huang, Qiongqiong Liu, Weiqi Luo*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26661](https://doi.org/10.1609/aaai.v37i12.26661)

        **Abstract**:

        Knowledge tracing (KT) is a crucial technique to predict students’ future performance by observing their historical learning processes. Due to the powerful representation ability of deep neural networks, remarkable progress has been made by using deep learning techniques to solve the KT problem. The majority of existing approaches rely on the homogeneous question assumption that questions have equivalent contributions if they share the same set of knowledge components. Unfortunately, this assumption is inaccurate in real-world educational scenarios. Furthermore, it is very challenging to interpret the prediction results from the existing deep learning based KT models. Therefore, in this paper, we present QIKT, a question-centric interpretable KT model to address the above challenges. The proposed QIKT approach explicitly models students’ knowledge state variations at a ﬁne-grained level with question-sensitive cognitive representations that are jointly learned from a question-centric knowledge acquisition module and a question-centric problem solving module. Meanwhile, the QIKT utilizes an item response theory based prediction layer to generate interpretable prediction results. The proposed QIKT model is evaluated on three public real-world educational datasets. The results demonstrate that our approach is superior on the KT prediction task, and it outperforms a wide range of deep learning based KT models in terms of prediction accuracy with better model interpretability. To encourage reproducible results, we have provided all the datasets and code at https://pykt.org/.

        ----

        ## [1592] Critical Firms Prediction for Stemming Contagion Risk in Networked-Loans through Graph-Based Deep Reinforcement Learning

        **Authors**: *Dawei Cheng, Zhibin Niu, Jianfu Zhang, Yiyi Zhang, Changjun Jiang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26662](https://doi.org/10.1609/aaai.v37i12.26662)

        **Abstract**:

        The networked-loan is major financing support for Micro, Small and Medium-sized Enterprises (MSMEs) in some developing countries. But external shocks may weaken the financial networks' robustness; an accidental default may spread across the network and collapse the whole network. Thus, predicting the critical firms in networked-loans to stem contagion risk and prevent potential systemic financial crises is of crucial significance to the long-term health of inclusive finance and sustainable economic development. Existing approaches in the banking industry dismiss the contagion risk across loan networks and need extensive knowledge with sophisticated financial expertise. Regarding the issues, we propose a novel approach to predict critical firms for stemming contagion risk in the bank industry with deep reinforcement learning integrated with high-order graph message-passing networks. We demonstrate that our approach outperforms the state-of-the-art baselines significantly on the dataset from a large commercial bank. Moreover, we also conducted empirical studies on the real-world loan dataset for risk mitigation. The proposed approach enables financial regulators and risk managers to better track and understands contagion and systemic risk in networked-loans. The superior performance also represents a paradigm shift in addressing the modern challenges in financing support of MSMEs and sustainable economic development.

        ----

        ## [1593] GAN-Based Domain Inference Attack

        **Authors**: *Yuechun Gu, Keke Chen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26663](https://doi.org/10.1609/aaai.v37i12.26663)

        **Abstract**:

        Model-based attacks can infer training data information from deep neural network models. These attacks heavily depend on the attacker's knowledge of the application domain, e.g., using it to determine the auxiliary data for model-inversion attacks. However, attackers may not know what the model is used for in practice. We propose a generative adversarial network (GAN) based method to explore likely or similar domains of a target model -- the model domain inference (MDI) attack. For a given target (classification) model, we assume that the attacker knows nothing but the input and output formats and can use the model to derive the prediction for any input in the desired form. Our basic idea is to use the target model to affect a GAN training process for a candidate domain's dataset that is easy to obtain. We find that the target model may distort the training procedure less if the domain is more similar to the target domain. We then measure the distortion level with the distance between GAN-generated datasets, which can be used to rank candidate domains for the target model. Our experiments show that the auxiliary dataset from an MDI top-ranked domain can effectively boost the result of model-inversion attacks.

        ----

        ## [1594] Physics Guided Neural Networks for Time-Aware Fairness: An Application in Crop Yield Prediction

        **Authors**: *Erhu He, Yiqun Xie, Licheng Liu, Weiye Chen, Zhenong Jin, Xiaowei Jia*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26664](https://doi.org/10.1609/aaai.v37i12.26664)

        **Abstract**:

        This paper proposes a physics-guided neural network model to predict crop yield and maintain the fairness over space. Failures to preserve the spatial fairness in predicted maps of crop yields can result in biased policies and intervention strategies in the distribution of assistance or subsidies in supporting individuals at risk. Existing methods for fairness enforcement are not designed for capturing the complex physical processes that underlie the crop growing process, and thus are unable to produce good predictions over large regions under different weather conditions and soil properties. More importantly, the fairness is often degraded when existing methods are applied to different years due to the change of weather conditions and farming practices. To address these issues, we propose a physics-guided neural network model, which leverages the physical knowledge from existing physics-based models to guide the extraction of representative physical information and discover the temporal data shift across years. In particular, we use a reweighting strategy to discover the relationship between training years and testing years using the physics-aware representation. Then the physics-guided neural network will be refined via a bi-level optimization process based on the reweighted fairness objective. The proposed method has been evaluated using real county-level crop yield data and simulated data produced by a physics-based model. The results demonstrate that this method can significantly improve the predictive performance and preserve the spatial fairness when generalized to different years.

        ----

        ## [1595] "Nothing Abnormal": Disambiguating Medical Reports via Contrastive Knowledge Infusion

        **Authors**: *Zexue He, An Yan, Amilcare Gentili, Julian J. McAuley, Chun-Nan Hsu*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26665](https://doi.org/10.1609/aaai.v37i12.26665)

        **Abstract**:

        Sharing medical reports is essential for patient-centered care. A recent line of work has focused on automatically generating reports with NLP methods. However, different audiences have different purposes when writing/reading medical reports – for example, healthcare professionals care more about pathology, whereas patients are more concerned with the diagnosis ("Is there any abnormality?"). The expectation gap results in a common situation where patients find their medical reports to be ambiguous and therefore unsure about the next steps. In this work, we explore the audience expectation gap in healthcare and summarize common ambiguities that lead patients to be confused about their diagnosis into three categories: medical jargon, contradictory findings, and misleading grammatical errors. Based on our analysis, we define a disambiguation rewriting task to regenerate an input to be unambiguous while preserving information about the original content. We further propose a rewriting algorithm based on contrastive pretraining and perturbation-based rewriting. In addition, we create two datasets, OpenI-Annotated based on chest reports and VA-Annotated based on general medical reports, with available binary labels for ambiguity and abnormality presence annotated by radiology specialists. Experimental results on these datasets show that our proposed algorithm effectively rewrites input sentences in a less ambiguous way with high content fidelity. Our code and annotated data will be released to facilitate future research.

        ----

        ## [1596] MTDiag: An Effective Multi-Task Framework for Automatic Diagnosis

        **Authors**: *Zhenyu Hou, Yukuo Cen, Ziding Liu, Dongxue Wu, Baoyan Wang, Xuanhe Li, Lei Hong, Jie Tang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26666](https://doi.org/10.1609/aaai.v37i12.26666)

        **Abstract**:

        Automatic diagnosis systems aim to probe for symptoms (i.e., symptom checking) and diagnose disease through multi-turn conversations with patients. Most previous works formulate it as a sequential decision process and use reinforcement learning (RL) to decide whether to inquire about symptoms or make a diagnosis. However, these RL-based methods heavily rely on the elaborate reward function and usually suffer from an unstable training process and low data efficiency. In this work, we propose an effective multi-task framework for automatic diagnosis called MTDiag. We first reformulate symptom checking as a multi-label classification task by direct supervision. Each medical dialogue is equivalently converted into multiple samples for classification, which can also help alleviate the data scarcity problem. Furthermore, we design a multi-task learning strategy to guide the symptom checking procedure with disease information and further utilize contrastive learning to better distinguish symptoms between diseases. Extensive experimental results show that our method achieves state-of-the-art performance on four public datasets with 1.7%~3.1% improvement in disease diagnosis, demonstrating the superiority of the proposed method. Additionally, our model is now deployed in an online medical consultant system as an assistant tool for real-life doctors.

        ----

        ## [1597] Walkability Optimization: Formulations, Algorithms, and a Case Study of Toronto

        **Authors**: *Weimin Huang, Elias B. Khalil*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26667](https://doi.org/10.1609/aaai.v37i12.26667)

        **Abstract**:

        The concept of walkable urban development has gained increased
attention due to its public health, economic, and environmental
sustainability benefits. Unfortunately, land zoning
and historic under-investment have resulted in spatial inequality
in walkability and social inequality among residents.
We tackle the problem of Walkability Optimization through
the lens of combinatorial optimization. The task is to select
locations in which additional amenities (e.g., grocery stores,
schools, restaurants) can be allocated to improve resident access
via walking while taking into account existing amenities
and providing multiple options (e.g., for restaurants).
To this end, we derive Mixed-Integer Linear Programming
(MILP) and Constraint Programming (CP) models. Moreover,
we show that the problem’s objective function is submodular
in special cases, which motivates an efficient greedy
heuristic. We conduct a case study on 31 underserved neighborhoods
in the City of Toronto, Canada. MILP finds the
best solutions in most scenarios but does not scale well with
network size. The greedy algorithm scales well and finds
high-quality solutions. Our empirical evaluation shows that
neighbourhoods with low walkability have a great potential
for transformation into pedestrian-friendly neighbourhoods
by strategically placing new amenities. Allocating 3 additional
grocery stores, schools, and restaurants can improve the
“WalkScore” by more than 50 points (on a scale of 100) for 4
neighbourhoods and reduce the walking distances to amenities
for 75% of all residential locations to 10 minutes for all
amenity types. Our code and paper appendix are available at
https://github.com/khalil-research/walkability.

        ----

        ## [1598] Low Emission Building Control with Zero-Shot Reinforcement Learning

        **Authors**: *Scott R. Jeen, Alessandro Abate, Jonathan M. Cullen*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26668](https://doi.org/10.1609/aaai.v37i12.26668)

        **Abstract**:

        Heating and cooling systems in buildings account for 31% of global energy use, much of which are regulated by Rule Based Controllers (RBCs) that neither maximise energy efficiency nor minimise emissions by interacting optimally with the grid. Control via Reinforcement Learning (RL) has been shown to significantly improve building energy efficiency, but existing solutions require access to building-specific simulators or data that cannot be expected for every building in the world. In response, we show it is possible to obtain emission-reducing policies without such knowledge a priori–a paradigm we call zero-shot building control. We combine ideas from system identification and model-based RL to create PEARL (Probabilistic Emission-Abating Reinforcement Learning) and show that a short period of active exploration is all that is required to build a performant model. In experiments across three varied building energy simulations, we show PEARL outperforms an existing RBC once, and popular RL baselines in all cases, reducing building emissions by as much as 31% whilst maintaining thermal comfort. Our source code is available online via: https://enjeeneer.io/projects/pearl/.

        ----

        ## [1599] Spatio-Temporal Graph Neural Point Process for Traffic Congestion Event Prediction

        **Authors**: *Guangyin Jin, Lingbo Liu, Fuxian Li, Jincai Huang*

        **Conference**: *aaai 2023*

        **URL**: [https://doi.org/10.1609/aaai.v37i12.26669](https://doi.org/10.1609/aaai.v37i12.26669)

        **Abstract**:

        Traffic congestion event prediction is an important yet challenging task in intelligent transportation systems. Many existing works about traffic prediction integrate various temporal encoders and graph convolution networks (GCNs), called spatio-temporal graph-based neural networks, which focus on predicting dense variables such as flow, speed and demand in time snapshots, but they can hardly forecast the traffic congestion events that are sparsely distributed on the continuous time axis. In recent years, neural point process (NPP) has emerged as an appropriate framework for event prediction in continuous time scenarios. However, most conventional works about NPP cannot model the complex spatio-temporal dependencies and congestion evolution patterns. To address these limitations, we propose a spatio-temporal graph neural point process framework, named STGNPP for traffic congestion event prediction. Specifically, we first design the spatio-temporal graph learning module to fully capture the long-range spatio-temporal dependencies from the historical traffic state data along with the road network. The extracted spatio-temporal hidden representation and congestion event information are then fed into a continuous gated recurrent unit to model the congestion evolution patterns. In particular, to fully exploit the periodic information, we also improve the intensity function calculation of the point process with a periodic gated mechanism. Finally, our model simultaneously predicts the occurrence time and duration of the next congestion. Extensive experiments on two real-world datasets demonstrate that our method achieves superior performance in comparison to existing state-of-the-art approaches.

        ----

        

[Go to the previous page](AAAI-2023-list07.md)

[Go to the next page](AAAI-2023-list09.md)

[Go to the catalog section](README.md)