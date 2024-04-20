## [2000] AdaCCD: Adaptive Semantic Contrasts Discovery Based Cross Lingual Adaptation for Code Clone Detection

**Authors**: *Yangkai Du, Tengfei Ma, Lingfei Wu, Xuhong Zhang, Shouling Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29749](https://doi.org/10.1609/aaai.v38i16.29749)

**Abstract**:

Code Clone Detection, which aims to retrieve functionally similar programs from large code bases, has been attracting increasing attention. Modern software often involves a diverse range of programming languages. However, current code clone detection methods are generally limited to only a few popular programming languages due to insufficient annotated data as well as their own model design constraints. To address these issues, we present AdaCCD, a novel cross-lingual adaptation method that can detect cloned codes in a new language without annotations in that language. AdaCCD leverages language-agnostic code representations from pre-trained programming language models and propose an Adaptively Refined Contrastive Learning framework to transfer knowledge from resource-rich languages to resource-poor languages. We evaluate the cross-lingual adaptation results of AdaCCD by constructing a multilingual code clone detection benchmark consisting of 5 programming languages. AdaCCD achieves significant improvements over other baselines, and achieve comparable performance to supervised fine-tuning.

----

## [2001] Frugal LMs Trained to Invoke Symbolic Solvers Achieve Parameter-Efficient Arithmetic Reasoning

**Authors**: *Subhabrata Dutta, Ishan Pandey, Joykirat Singh, Sunny Manchanda, Soumen Chakrabarti, Tanmoy Chakraborty*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29750](https://doi.org/10.1609/aaai.v38i16.29750)

**Abstract**:

Large Language Models (LLM) exhibit zero-shot mathematical reasoning capacity as a behavior emergent with scale, commonly manifesting as chain-of-thoughts (CoT) reasoning. However, multiple empirical findings suggest that this prowess is exclusive to LLMs that have exorbitant sizes (beyond 50 billion parameters). Meanwhile, educational neuroscientists suggest that symbolic algebraic manipulation be introduced around the same time as arithmetic word problems so as to modularize language-to-formulation, symbolic manipulation of the formulation, and endgame arithmetic.
In this paper, we start with the hypothesis that much smaller LMs, which are weak at multi-step reasoning, can achieve reasonable arithmetic reasoning if arithmetic word problems are posed as a formalize-then-solve task.
In our architecture, which we call SyReLM, the LM serves the role of a translator to map natural language arithmetic questions into a formal language (FL) description. A symbolic solver then evaluates the FL expression to obtain the answer.
A small frozen LM, equipped with an efficient low-rank adapter, is capable of generating FL expressions that incorporate natural language descriptions of the arithmetic problem (e.g., variable names and their purposes, formal expressions combining variables, etc.).
We adopt policy-gradient reinforcement learning to train the adapted LM, informed by the non-differentiable symbolic solver. This marks a sharp departure from the recent development in tool-augmented LLMs, in which the external tools (e.g., calculator, Web search, etc.) are essentially detached from the learning phase of the LM. SyReLM shows massive improvements (e.g., +30.65 absolute point improvement in accuracy on the SVAMP dataset using GPT-J 6B model)  over base LMs, while keeping our testbed easy to diagnose and interpret, and within the reach of most researchers.

----

## [2002] Can Large Language Models Serve as Rational Players in Game Theory? A Systematic Analysis

**Authors**: *Caoyun Fan, Jindou Chen, Yaohui Jin, Hao He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29751](https://doi.org/10.1609/aaai.v38i16.29751)

**Abstract**:

Game theory, as an analytical tool, is frequently utilized to analyze human behavior in social science research. With the high alignment between the behavior of Large Language Models (LLMs) and humans, a promising research direction is to employ LLMs as substitutes for humans in game experiments, enabling social science research. However, despite numerous empirical researches on the combination of LLMs and game theory, the capability boundaries of LLMs in game theory remain unclear. In this research, we endeavor to systematically analyze LLMs in the context of game theory. Specifically, rationality, as the fundamental principle of game theory, serves as the metric for evaluating players' behavior --- building a clear desire, refining belief about uncertainty, and taking optimal actions. Accordingly, we select three classical games (dictator game, Rock-Paper-Scissors, and ring-network game) to analyze to what extent LLMs can achieve rationality in these three aspects. The experimental results indicate that even the current state-of-the-art LLM (GPT-4) exhibits substantial disparities compared to humans in game theory. For instance, LLMs struggle to build desires based on uncommon preferences, fail to refine belief from many simple patterns, and may overlook or modify refined belief when taking actions. Therefore, we consider that introducing LLMs into game experiments in the field of social science should be approached with greater caution.

----

## [2003] Enhancing Low-Resource Relation Representations through Multi-View Decoupling

**Authors**: *Chenghao Fan, Wei Wei, Xiaoye Qu, Zhenyi Lu, Wenfeng Xie, Yu Cheng, Dangyang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29752](https://doi.org/10.1609/aaai.v38i16.29752)

**Abstract**:

Recently, prompt-tuning with pre-trained language models (PLMs) has demonstrated the significantly enhancing ability of relation extraction (RE) tasks. 
However, in low-resource scenarios, where the available training data is scarce, previous prompt-based methods may still perform poorly for prompt-based representation learning due to a superficial understanding of the relation. 
To this end, we highlight the importance of learning high-quality relation representation in low-resource scenarios for RE, and propose a novel prompt-based relation representation method, named MVRE (Multi-View Relation Extraction), to better leverage the capacity of PLMs to improve the performance of RE within the low-resource prompt-tuning paradigm. Specifically, MVRE decouples each relation into different perspectives to encompass multi-view relation representations for maximizing the likelihood during relation inference.
Furthermore, we also design a Global-Local loss and a Dynamic-Initialization method for better alignment of the multi-view relation-representing virtual words, containing the semantics of relation labels during the optimization learning process and initialization. Extensive experiments on
three benchmark datasets show that our method can achieve
state-of-the-art in low-resource settings.

----

## [2004] Quantum-Inspired Neural Network with Runge-Kutta Method

**Authors**: *Zipeng Fan, Jing Zhang, Peng Zhang, Qianxi Lin, Hui Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29753](https://doi.org/10.1609/aaai.v38i16.29753)

**Abstract**:

In recent years, researchers have developed novel Quantum-Inspired Neural Network (QINN) frameworks for the Natural Language Processing (NLP) tasks, inspired by the theoretical investigations of quantum cognition. However, we have found that the training efficiency of QINNs is significantly lower than that of classical networks. We analyze the unitary transformation modules of existing QINNs based on the time displacement symmetry of quantum mechanics and discover that they are resembling a mathematical form similar to the first-order Euler method. The high truncation error associated with Euler method affects the training efficiency of QINNs. In order to enhance the training efficiency of QINNs, we generalize QINNs' unitary transformation modules to the Quantum-like high-order Runge-Kutta methods (QRKs). Moreover, we present the results of experiments on conversation emotion recognition and text classification tasks to validate the effectiveness of the proposed approach.

----

## [2005] Large Language Models Are Neurosymbolic Reasoners

**Authors**: *Meng Fang, Shilong Deng, Yudi Zhang, Zijing Shi, Ling Chen, Mykola Pechenizkiy, Jun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29754](https://doi.org/10.1609/aaai.v38i16.29754)

**Abstract**:

A wide range of real-world applications is characterized by their symbolic nature, necessitating a strong capability for symbolic reasoning. This paper investigates the potential application of Large Language Models (LLMs) as symbolic reasoners. We focus on text-based games, significant benchmarks for agents with natural language capabilities, particularly in symbolic tasks like math, map reading, sorting, and applying common sense in text-based worlds. To facilitate these agents, we propose an LLM agent designed to tackle symbolic challenges and achieve in-game objectives. We begin by initializing the LLM agent and informing it of its role. The agent then receives observations and a set of valid actions from the text-based games, along with a specific symbolic module. With these inputs, the LLM agent chooses an action and interacts with the game environments. Our experimental results demonstrate that our method significantly enhances the capability of LLMs as automated agents for symbolic reasoning, and our LLM agent is effective in text-based games involving symbolic tasks, achieving an average performance of 88% across all tasks.

----

## [2006] Combining Multiple Supervision for Robust Zero-Shot Dense Retrieval

**Authors**: *Yan Fang, Qingyao Ai, Jingtao Zhan, Yiqun Liu, Xiaolong Wu, Zhao Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29755](https://doi.org/10.1609/aaai.v38i16.29755)

**Abstract**:

Recently, dense retrieval (DR) models, which represent queries and documents with fixed-width vectors and retrieve relevant ones via nearest neighbor search, have drawn increasing attention from the IR community. 
However, previous studies have shown that the effectiveness of DR critically relies on sufficient training signals, which leads to severe performance degradation when applied in out-of-domain scenarios, where large-scale training data are usually unavailable.
To solve this problem, existing studies adopt a data-augmentation-plus-joint-training paradigm to construct weak/pseudo supervisions on the target domain and combine them with the large-scale human annotated data on the source domain to train the DR models. However, they don't explicitly distinguish the data and the supervision signals in the training process and simply assume that the DR models are mighty enough to capture and memorize different domain knowledge and relevance matching patterns without guidance, which, as shown in this paper, is not true.
Based on this observation, we propose a Robust Multi-Supervision Combining strategy (RMSC) that
decouples the domain and supervision signals by explicitly telling the DR models how the domain data and supervision signals are combined in the training data with specially designed soft tokens. 
With the extra soft tokens to store the domain-specific and supervision-specific knowledge, RMSC allows the DR models 
to conduct retrieval based on human-like relevance matching patterns and target-specific language distribution on the target domain without human annotations.
Extensive experiments on zero-shot DR benchmarks show that RMSC significantly improves the ranking performance on the target domain compared to strong DR baselines and domain adaptation methods, while being stable during training and can be combined with query generation or second-stage pre-training.

----

## [2007] Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy

**Authors**: *Yu Fu, Deyi Xiong, Yue Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29756](https://doi.org/10.1609/aaai.v38i16.29756)

**Abstract**:

To mitigate potential risks associated with language models (LMs), recent AI detection research proposes incorporating watermarks into machine-generated text through random vocabulary restrictions and utilizing this information for detection. In this paper, we show that watermarking algorithms designed for LMs cannot be seamlessly applied to conditional text generation (CTG) tasks without a notable decline in downstream task performance. To address this issue, we introduce a simple yet effective semantic-aware watermarking algorithm that considers the characteristics of conditional text generation with the input context. Compared to the baseline watermarks, our proposed watermark yields significant improvements in both automatic and human evaluations across various text generation models, including BART and Flan-T5, for CTG tasks such as summarization and data-to-text generation. Meanwhile, it maintains detection ability with higher z-scores but lower AUC scores, suggesting the presence of a detection paradox that poses additional challenges for watermarking CTG.

----

## [2008] BAND: Biomedical Alert News Dataset

**Authors**: *Zihao Fu, Meiru Zhang, Zaiqiao Meng, Yannan Shen, David L. Buckeridge, Nigel Collier*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29757](https://doi.org/10.1609/aaai.v38i16.29757)

**Abstract**:

Infectious disease outbreaks continue to pose a significant threat to human health and well-being. To improve disease surveillance and understanding of disease spread, several surveillance systems have been developed to monitor daily news alerts and social media. However, existing systems lack thorough epidemiological analysis in relation to corresponding alerts or news, largely due to the scarcity of well-annotated reports data. To address this gap, we introduce the Biomedical Alert News Dataset (BAND), which includes 1,508 samples from existing reported news articles, open emails, and alerts, as well as 30 epidemiology-related questions. These questions necessitate the model's expert reasoning abilities, thereby offering valuable insights into the outbreak of the disease. The BAND dataset brings new challenges to the NLP world, requiring better inference capability of the content and the ability to infer important information. We provide several benchmark tasks, including Named Entity Recognition (NER), Question Answering (QA), and Event Extraction (EE), to demonstrate existing models' capabilities and limitations in handling epidemiology-specific tasks. It is worth noting that some models may lack the human-like inference capability required to fully utilize the corpus. To the best of our knowledge, the BAND corpus is the largest corpus of well-annotated biomedical outbreak alert news with elaborately designed questions, making it a valuable resource for epidemiologists and NLP researchers alike.

----

## [2009] Winnie: Task-Oriented Dialog System with Structure-Aware Contrastive Learning and Enhanced Policy Planning

**Authors**: *Kaizhi Gao, Tianyu Wang, Zhongjing Ma, Suli Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29758](https://doi.org/10.1609/aaai.v38i16.29758)

**Abstract**:

Pre-trained encoder-decoder models are widely applied in Task-Oriented Dialog (TOD) systems on the session level, mainly focusing on modeling the dialog semantic information. Dialogs imply structural information indicating the interaction among user utterances, belief states, database search results, system acts and responses, which is also crucial for TOD systems. In addition, for the system acts,  additional pre-training and datasets are considered to improve their accuracies, undoubtedly introducing a burden. Therefore, a novel end-to-end TOD system named Winnie is proposed in this paper to improve the TOD performance. First, to make full use of the intrinsic structural information, supervised contrastive learning is adopted to narrow the gap in the representation space between text representations of the same category and enlarge the overall continuous representation margin between text representations of different categories in dialog context. Then, a system act classification task is introduced for policy optimization during fine-tuning. Empirical results show that Winnie substantially improves the performance of the TOD system. By introducing the supervised contrastive and system act classification losses, Winnie achieves state-of-the-art results on benchmark datasets, including MultiWOZ2.2, In-Car, and Camrest676. Their end-to-end combined scores are improved by 3.2, 1.9, and 1.1 points, respectively.

----

## [2010] Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum

**Authors**: *Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma, Zhaochun Ren*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29759](https://doi.org/10.1609/aaai.v38i16.29759)

**Abstract**:

Augmenting large language models (LLMs) with external tools has emerged as a promising approach to extending the capability of LLMs. Although there are some works that employ open-source LLMs for the tool-learning task, most of them are trained in a controlled environment in which LLMs only learn to execute the human-provided tools. However, selecting proper tools from the large toolset is also a crucial ability for the tool-learning model to be applied in real-world applications. Existing methods usually directly employ self-instruction methods to train the model, which ignores differences in tool complexity. In this paper, we propose the Confucius a novel tool-learning framework to train LLM to use complicated tools in real-world scenarios, which contains two main phases: (1) We first propose a multi-stage learning method to teach the LLM to use various tools from an easy-to-difficult curriculum; (2) thenceforth, we propose the Iterative Self-instruct from Introspective Feedback (ISIF) to dynamically construct the dataset to improve the ability to use the complicated tool. Extensive experiments conducted on both controlled and real-world settings demonstrate the superiority of our tool-learning framework in the real-world application scenario compared to both tuning-free (e.g., ChatGPT, Claude) and tuning-based baselines (e.g., GPT4Tools).

----

## [2011] Customizing Language Model Responses with Contrastive In-Context Learning

**Authors**: *Xiang Gao, Kamalika Das*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29760](https://doi.org/10.1609/aaai.v38i16.29760)

**Abstract**:

Large language models (LLMs) are becoming increasingly important for machine learning applications. However, it can be challenging to align LLMs with our intent, particularly when we want to generate content that is preferable over others or when we want the LLM to respond in a certain style or tone that is hard to describe. To address this challenge, we propose an approach that uses contrastive examples to better describe our intent. This involves providing positive examples that illustrate the true intent, along with negative examples that show what characteristics we want LLMs to avoid. The negative examples can be retrieved from labeled data, written by a human, or generated by the LLM itself.
Before generating an answer, we ask the model to analyze the examples to teach itself what to avoid. This reasoning step provides the model with the appropriate articulation of the user's need and guides it towards generting a better answer. We tested our approach on both synthesized and real-world datasets, including StackExchange and Reddit, and found that it significantly improves performance compared to standard few-shot prompting.

----

## [2012] DA-Net: A Disentangled and Adaptive Network for Multi-Source Cross-Lingual Transfer Learning

**Authors**: *Ling Ge, Chunming Hu, Guanghui Ma, Jihong Liu, Hong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29761](https://doi.org/10.1609/aaai.v38i16.29761)

**Abstract**:

Multi-Source cross-lingual transfer learning deals with the transfer of task knowledge from multiple labelled source languages to an unlabeled target language under the language shift. Existing methods typically focus on weighting the predictions produced by language-specific classifiers of different sources that follow a shared encoder. However, all source languages share the same encoder, which is updated by all these languages. The extracted representations inevitably contain different source languages' information, which may disturb the learning of the language-specific classifiers. Additionally,  due to the language gap,  language-specific classifiers trained with source labels are unable to make accurate predictions for the target language. Both facts impair the model's performance. To address these challenges, we propose a Disentangled and Adaptive Network ~(DA-Net). Firstly, we devise a feedback-guided collaborative disentanglement method that seeks to purify input representations of classifiers, thereby mitigating mutual interference from multiple sources. Secondly, we propose a class-aware parallel adaptation method that aligns class-level distributions for each source-target language pair, thereby alleviating the language pairs' language gap. Experimental results on three different tasks involving 38  languages validate the effectiveness of our approach.

----

## [2013] Discrepancy and Uncertainty Aware Denoising Knowledge Distillation for Zero-Shot Cross-Lingual Named Entity Recognition

**Authors**: *Ling Ge, Chunming Hu, Guanghui Ma, Jihong Liu, Hong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29762](https://doi.org/10.1609/aaai.v38i16.29762)

**Abstract**:

The knowledge distillation-based approaches have recently yielded state-of-the-art (SOTA) results for cross-lingual NER tasks in zero-shot scenarios. 
These approaches typically employ a teacher network trained with the labelled source (rich-resource) language to infer pseudo-soft labels for the unlabelled  target (zero-shot) language, and force a student network to approximate these pseudo labels to achieve knowledge transfer.
However, previous works have rarely discussed the issue of pseudo-label noise caused by the source-target language gap, which can mislead the training of the student network and result in negative knowledge transfer. 
This paper proposes an discrepancy and uncertainty aware Denoising Knowledge Distillation model (DenKD) to tackle this issue. 
Specifically, DenKD uses a discrepancy-aware denoising representation learning method to optimize the class representations of the target language produced by the teacher network, thus enhancing the quality of pseudo labels and reducing noisy predictions. Further, DenKD employs an uncertainty-aware denoising method to quantify the pseudo-label noise and adjust the focus of the student network on different samples during knowledge distillation, thereby mitigating the noise's adverse effects. We conduct extensive experiments on 28  languages including 4 languages not covered by the pre-trained models, and the results demonstrate the effectiveness of our   DenKD.

----

## [2014] Who Knows the Answer? Finding the Best Model and Prompt for Each Query Using Confidence-Based Search

**Authors**: *Walter Gerych, Yara Rizk, Vatche Isahagian, Vinod Muthusamy, Evelyn Duesterwald, Praveen Venkateswaran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29763](https://doi.org/10.1609/aaai.v38i16.29763)

**Abstract**:

There are increasingly many large language models (LLMs) available to the public. While these LLMs have exhibited impressive abilities on a variety of task, any individual LLM in particular may do well on some tasks and worse on others. Additionally, the performance of these models is heavily dependent on the choice of prompt template used. For instance, they exhibit sensitivity to the few shot examples chosen or brittleness to the wording of instructions. Moreover, a prompt template that makes a model perform well for one input may not be the optimal template for another input. This necessitates an approach for adaptively selecting LLM and prompt template pairs for each input. Recent work has shown that the accuracy of LLM's responses is correlated with the LLM's confidence in the response. Thus, a natural choice for selecting which model and prompt template to use is to select the pair that is most confident in its response. However, existing confidence metrics are expensive to calculate - necessitating multiple calls to each LLm and prompt pair. We thus propose an approach to predict the confidence of each pair using an auxiliary regression model that is inexpensive to run. Using this auxiliary model, we select the LLM and prompt template with the highest predicted confidence for a given input. Results on a range of benchmark datasets show that our confidence-based instance-level prompt search method consistently improves the performance of LLMs.

----

## [2015] A General Search-Based Framework for Generating Textual Counterfactual Explanations

**Authors**: *Daniel Gilo, Shaul Markovitch*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29764](https://doi.org/10.1609/aaai.v38i16.29764)

**Abstract**:

One of the prominent methods for explaining the decision of a machine-learning classifier is by a counterfactual example.
Most current algorithms for generating such examples in the textual domain are based on generative language models.  Generative models, however, are trained to minimize a specific loss function in order to fulfill certain requirements for the generated texts.  Any change in the requirements may necessitate costly retraining, thus potentially limiting their applicability.
In this paper, we present a general search-based framework for generating counterfactual explanations in the textual domain.  
Our framework is model-agnostic, domain-agnostic, anytime, and does not require retraining in order to adapt to changes in the user requirements. 
We model the task as a search problem in a space where the initial state is the classified text, and the goal state is a text in a given target class.  Our framework includes domain-independent modification operators, but can also exploit domain-specific knowledge through specialized operators. The search algorithm attempts to find a text from the target class with minimal user-specified distance from the original classified object.

----

## [2016] What Makes Quantization for Large Language Model Hard? An Empirical Study from the Lens of Perturbation

**Authors**: *Zhuocheng Gong, Jiahao Liu, Jingang Wang, Xunliang Cai, Dongyan Zhao, Rui Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29765](https://doi.org/10.1609/aaai.v38i16.29765)

**Abstract**:

Quantization has emerged as a promising technique for improving the memory and computational efficiency of large language models (LLMs). Though the trade-off between performance and efficiency is well-known, there is still much to be learned about the relationship between quantization and LLM performance. To shed light on this relationship, we propose a new perspective on quantization, viewing it as perturbations added to the weights and activations of LLMs. We call this approach ``the lens of perturbation". Using this lens, we conduct experiments with various artificial perturbations to explore their impact on LLM performance. Our findings reveal several connections between the properties of perturbations and LLM performance, providing insights into the failure cases of uniform quantization and suggesting potential solutions to improve the robustness of LLM quantization.
To demonstrate the significance of our findings, we implement a simple non-uniform quantization approach based on our insights. Our experiments show that this approach achieves minimal performance degradation on both 4-bit weight quantization and 8-bit quantization for weights and activations. These results validate the correctness of our approach and highlight its potential to improve the efficiency of LLMs without sacrificing performance.

----

## [2017] CoPL: Contextual Prompt Learning for Vision-Language Understanding

**Authors**: *Koustava Goswami, Srikrishna Karanam, Prateksha Udhayanan, K. J. Joseph, Balaji Vasan Srinivasan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29766](https://doi.org/10.1609/aaai.v38i16.29766)

**Abstract**:

Recent advances in multimodal learning has resulted in powerful vision-language models, whose representations are generalizable across a variety of downstream tasks. Recently, their generalization ability has been further extended by incorporating trainable prompts, borrowed from the natural language processing literature. While such prompt learning techniques have shown impressive results, we identify that these prompts are trained based on global image features which limits itself in two aspects: First, by using global features, these prompts could be focusing less on the discriminative foreground image, resulting in poor generalization to various out-of-distribution test cases. Second, existing work weights all prompts equally whereas intuitively, prompts should be reweighed according to the semantics of the image. We address these as part of our proposed Contextual Prompt Learning (CoPL) framework, capable of aligning the prompts to
the localized features of the image. Our key innovations over earlier works include using local image features as part of the prompt learning process, and more crucially, learning to weight these prompts based on local features that are appropriate for the task at hand. This gives us dynamic prompts that are both aligned to local image features as well as aware of local contextual relationships. Our extensive set of experiments on a variety of standard and few-shot datasets show that our method produces substantially improved performance when compared to the current state of the art methods. We also demonstrate both few-shot and out-of-distribution performance to establish the utility of learning dynamic prompts that are aligned to local image features.

----

## [2018] Xiezhi: An Ever-Updating Benchmark for Holistic Domain Knowledge Evaluation

**Authors**: *Zhouhong Gu, Xiaoxuan Zhu, Haoning Ye, Lin Zhang, Jianchen Wang, Yixin Zhu, Sihang Jiang, Zhuozhi Xiong, Zihan Li, Weijie Wu, Qianyu He, Rui Xu, Wenhao Huang, Jingping Liu, Zili Wang, Shusen Wang, Weiguo Zheng, Hongwei Feng, Yanghua Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29767](https://doi.org/10.1609/aaai.v38i16.29767)

**Abstract**:

New Natural Langauge Process~(NLP) benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present Xiezhi, the most comprehensive evaluation suite designed to assess holistic domain knowledge.Xiezhi comprises multiple-choice questions across 516 diverse disciplines ranging from 13 different subjects with 249,587 questions and accompanied by Xiezhi-Specialty with 14,041 questions and Xiezhi-Interdiscipline with 10,746 questions. We conduct evaluation of the 47 cutting-edge LLMs on Xiezhi. Results indicate that LLMs exceed average performance of humans in science, engineering, agronomy, medicine, and art, but fall short in economics, jurisprudence, pedagogy, literature, history, and management. All the evaluation code and data are open sourced in https://github.com/MikeGu721/XiezhiBenchmark

----

## [2019] DINGO: Towards Diverse and Fine-Grained Instruction-Following Evaluation

**Authors**: *Zihui Gu, Xingwu Sun, Fengzong Lian, Zhanhui Kang, Chengzhong Xu, Ju Fan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29768](https://doi.org/10.1609/aaai.v38i16.29768)

**Abstract**:

Instruction-following is particularly crucial for large language models (LLMs) to support diverse user requests. While existing work has made progress in aligning LLMs with human preferences, evaluating their capabilities on instruction-following remains a challenge due to complexity and diversity of real-world user instructions. While existing evaluation methods focus on general skills, they suffer from two main shortcomings, i.e., lack of fine-grained task-level evaluation and reliance on singular instruction expression. To address these problems, this paper introduces DINGO, a fine-grained and diverse instruction-following evaluation dataset that has two main advantages: (1) DINGO is based on a manual annotated, fine-grained and multi-level category tree with 130 nodes derived from real-world user requests; (2) DINGO includes diverse instructions, generated by both GPT-4 and human experts. Through extensive experiments, we demonstrate that DINGO can not only provide more challenging and comprehensive evaluation for LLMs, but also provide task-level fine-grained directions to further improve LLMs.

----

## [2020] MM-TTS: Multi-Modal Prompt Based Style Transfer for Expressive Text-to-Speech Synthesis

**Authors**: *Wenhao Guan, Yishuang Li, Tao Li, Hukai Huang, Feng Wang, Jiayan Lin, Lingyan Huang, Lin Li, Qingyang Hong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29769](https://doi.org/10.1609/aaai.v38i16.29769)

**Abstract**:

The style transfer task in Text-to-Speech (TTS) refers to the process of transferring style information into text content to generate corresponding speech with a specific style. However, most existing style transfer approaches are either based on fixed emotional labels or reference speech clips, which cannot achieve flexible style transfer.  Recently, some methods have adopted text descriptions to guide style transfer. In this paper, we propose a more flexible multi-modal and style controllable TTS framework named MM-TTS. It can utilize any modality as the prompt in unified multi-modal prompt space, including reference speech, emotional facial images, and text descriptions, to control the style of the generated speech in a system. The challenges of modeling such a multi-modal style controllable TTS mainly lie in two aspects: 1) aligning the multi-modal information into a unified style space to enable the input of arbitrary modality as the style prompt in a single system, and 2) efficiently transferring the unified style representation into the given text content, thereby empowering the ability to generate prompt style-related voice. To address these problems, we propose an aligned multi-modal prompt encoder that embeds different modalities into a unified style space, supporting style transfer for different modalities. Additionally, we present a new adaptive style transfer method named Style Adaptive Convolutions (SAConv) to achieve a better style representation. Furthermore, we design a Rectified Flow based Refiner to solve the problem of over-smoothing Mel-spectrogram and generate audio of higher fidelity. Since there is no public dataset for multi-modal TTS, we construct a dataset named MEAD-TTS, which is related to the field of expressive talking head. Our experiments on the MEAD-TTS dataset and out-of-domain datasets demonstrate that MM-TTS can achieve satisfactory results based on multi-modal prompts. The audio samples and constructed dataset are available at https://multimodal-tts.github.io.

----

## [2021] Mitigating Large Language Model Hallucinations via Autonomous Knowledge Graph-Based Retrofitting

**Authors**: *Xinyan Guan, Yanjiang Liu, Hongyu Lin, Yaojie Lu, Ben He, Xianpei Han, Le Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29770](https://doi.org/10.1609/aaai.v38i16.29770)

**Abstract**:

Incorporating factual knowledge in knowledge graph is regarded as a promising approach for mitigating the hallucination of large language models (LLMs). Existing methods usually only use the user's input to query the knowledge graph, thus failing to address the factual hallucination generated by LLMs during its reasoning process. To address this problem, this paper proposes Knowledge Graph-based Retrofitting (KGR), a new framework that incorporates LLMs with KGs to mitigate factual hallucination during the reasoning process by retrofitting the initial draft responses of LLMs based on the factual knowledge stored in KGs. Specifically, KGR leverages LLMs to extract, select, validate, and retrofit factual statements within the model-generated responses, which enables an autonomous knowledge verifying and refining procedure without any additional manual efforts. Experiments show that KGR can significantly improve the performance of LLMs on factual QA benchmarks especially when involving complex reasoning processes, which demonstrates the necessity and effectiveness of KGR in mitigating hallucination and enhancing the reliability of LLMs.

----

## [2022] Detecting and Preventing Hallucinations in Large Vision Language Models

**Authors**: *Anisha Gunjal, Jihan Yin, Erhan Bas*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29771](https://doi.org/10.1609/aaai.v38i16.29771)

**Abstract**:

Instruction tuned Large Vision Language Models (LVLMs) have significantly advanced in generalizing across a diverse set of multi-modal tasks, especially for Visual Question Answering (VQA). However, generating detailed responses that are visually grounded is still a challenging task for these models. We find that even the current state-of-the-art LVLMs (InstructBLIP) still contain a staggering 30 percent of the hallucinatory text in the form of non-existent objects, unfaithful descriptions, and inaccurate relationships. To address this, we introduce M-HalDetect, a Multimodal Hallucination Detection Dataset that can be used to train and benchmark models for hallucination detection and prevention. M-HalDetect consists of 16k fine-grained annotations on VQA examples, making it the first comprehensive multi-modal hallucination detection dataset for detailed image descriptions. Unlike previous work that only consider object hallucination, we additionally annotate both entity descriptions and relationships that are unfaithful. To demonstrate the potential of this dataset for hallucination prevention, we optimize InstructBLIP through our novel Fine-grained Direct Preference Optimization (FDPO). We also train fine-grained multi-modal reward models from InstructBLIP and evaluate their effectiveness with best-of-n rejection sampling (RS). We perform human evaluation on both FDPO and rejection sampling, and find that they reduce hallucination rates in InstructBLIP by 41% and 55% respectively. We also find that our reward model generalizes to other multi-modal models, reducing hallucinations in LLaVA and mPLUG-OWL by 15% and 57% respectively, and has strong correlation with human evaluated accuracy scores. The dataset is available at https://github.com/hendryx-scale/mhal-detect.

----

## [2023] MolTailor: Tailoring Chemical Molecular Representation to Specific Tasks via Text Prompts

**Authors**: *Haoqiang Guo, Sendong Zhao, Haochun Wang, Yanrui Du, Bing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29772](https://doi.org/10.1609/aaai.v38i16.29772)

**Abstract**:

Deep learning is now widely used in drug discovery, providing significant acceleration and cost reduction. As the most fundamental building block, molecular representation is essential for predicting molecular properties to enable various downstream applications. Most existing methods attempt to incorporate more information to learn better representations. However, not all features are equally important for a specific task. Ignoring this would potentially compromise the training efficiency and predictive accuracy. To address this issue, we propose a novel approach, which treats language models as an agent and molecular pretraining models as a knowledge base. The agent accentuates task-relevant features in the molecular representation by understanding the natural language description of the task, just as a tailor customizes clothes for clients. Thus, we call this approach MolTailor. Evaluations demonstrate MolTailor's superior performance over baselines, validating the efficacy of enhancing relevance for molecular representation learning. This illustrates the potential of language model guided optimization to better exploit and unleash the capabilities of existing powerful molecular representation methods. Our code and appendix are available at https://github.com/SCIR-HI/MolTailor.

----

## [2024] Audio Generation with Multiple Conditional Diffusion Model

**Authors**: *Zhifang Guo, Jianguo Mao, Rui Tao, Long Yan, Kazushige Ouchi, Hong Liu, Xiangdong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29773](https://doi.org/10.1609/aaai.v38i16.29773)

**Abstract**:

Text-based audio generation models have limitations as they cannot encompass all the information in audio, leading to restricted controllability when relying solely on text. To address this issue, we propose a novel model that enhances the controllability of existing pre-trained text-to-audio models by incorporating additional conditions including content (timestamp) and style (pitch contour and energy contour) as supplements to the text. This approach achieves fine-grained control over the temporal order, pitch, and energy of generated audio. To preserve the diversity of generation, we employ a trainable control condition encoder that is enhanced by a large language model and a trainable Fusion-Net to encode and fuse the additional conditions while keeping the weights of the pre-trained text-to-audio model frozen. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing datasets into a new dataset comprising the audio and corresponding conditions and use a series of evaluation metrics to evaluate the controllability performance. Experimental results demonstrate that our model successfully achieves fine-grained control to accomplish controllable audio generation.

----

## [2025] Small Language Model Can Self-Correct

**Authors**: *Haixia Han, Jiaqing Liang, Jie Shi, Qianyu He, Yanghua Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29774](https://doi.org/10.1609/aaai.v38i16.29774)

**Abstract**:

Generative Language Models (LMs) such as ChatGPT have exhibited remarkable performance across various downstream tasks. Nevertheless, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. Previous studies have devised sophisticated pipelines and prompts to induce large LMs to exhibit the capability for self-correction. However, large LMs are explicitly prompted to verify and modify their answers separately rather than completing all steps spontaneously like humans. Moreover, these complex prompts are extremely challenging for small LMs to follow. In this paper, we introduce the Intrinsic Self-Correction (ISC) in generative language models, aiming to correct the initial output of LMs in a self-triggered manner, even for those small LMs with 6 billion parameters. Specifically, we devise a pipeline for constructing self-correction data and propose Partial Answer Masking (PAM), aiming to endow the model with the capability for intrinsic self-correction through fine-tuning. We conduct experiments using LMs with parameters sizes ranging from 6 billion to 13 billion in two tasks, including commonsense reasoning and factual knowledge reasoning. Our experiments demonstrate that the outputs generated using ISC outperform those generated without self-correction. We believe that the output quality of even small LMs can be further improved by empowering them with the ability to intrinsic self-correct.

----

## [2026] Decoupling Representation and Knowledge for Few-Shot Intent Classification and Slot Filling

**Authors**: *Jie Han, Yixiong Zou, Haozhao Wang, Jun Wang, Wei Liu, Yao Wu, Tao Zhang, Ruixuan Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29775](https://doi.org/10.1609/aaai.v38i16.29775)

**Abstract**:

Few-shot intent classification and slot filling are important but challenging tasks due to the scarcity of finely labeled data. Therefore, current works first train a model on source domains with sufficiently labeled data, and then transfer the model to target domains where only rarely labeled data is available. However, experience transferring as a whole usually suffers from gaps that exist among source domains and target domains. For instance, transferring domain-specific-knowledge-related experience is difficult. To tackle this problem, we propose a new method that explicitly decouples the transferring of general-semantic-representation-related experience and the domain-specific-knowledge-related experience. Specifically, for domain-specific-knowledge-related experience, we design two modules to capture intent-slot relation and slot-slot relation respectively. Extensive experiments on Snips and FewJoint datasets show that our method achieves state-of-the-art performance. The method improves the joint accuracy metric from 27.72% to 42.20% in the 1-shot setting, and from 46.54% to 60.79% in the 5-shot setting.

----

## [2027] Multi-Modal Latent Space Learning for Chain-of-Thought Reasoning in Language Models

**Authors**: *Liqi He, Zuchao Li, Xiantao Cai, Ping Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29776](https://doi.org/10.1609/aaai.v38i16.29776)

**Abstract**:

Chain-of-thought (CoT) reasoning has exhibited impressive performance in language models for solving complex tasks and answering questions. However, many real-world questions require multi-modal information, such as text and images. Previous research on multi-modal CoT has primarily focused on extracting fixed image features from off-the-shelf vision models and then fusing them with text using attention mechanisms. This approach has limitations because these vision models were not designed for complex reasoning tasks and do not align well with language thoughts. To overcome this limitation, we introduce a novel approach for multi-modal CoT reasoning that utilizes latent space learning via diffusion processes to generate effective image features that align with language thoughts. Our method fuses image features and text representations at a deep level and improves the complex reasoning ability of multi-modal CoT. We demonstrate the efficacy of our proposed method on multi-modal ScienceQA and machine translation benchmarks, achieving state-of-the-art performance on ScienceQA. Overall, our approach offers a more robust and effective solution for multi-modal reasoning in language models, enhancing their ability to tackle complex real-world problems.

----

## [2028] Can Large Language Models Understand Real-World Complex Instructions?

**Authors**: *Qianyu He, Jie Zeng, Wenhao Huang, Lina Chen, Jin Xiao, Qianxi He, Xunzhe Zhou, Jiaqing Liang, Yanghua Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29777](https://doi.org/10.1609/aaai.v38i16.29777)

**Abstract**:

Large language models (LLMs) can understand human instructions, showing their potential for pragmatic applications beyond traditional NLP tasks. However, they still struggle with complex instructions, which can be either complex task descriptions that require multiple tasks and constraints, or complex input that contains long context, noise, heterogeneous information and multi-turn format. Due to these features, LLMs often ignore semantic constraints from task descriptions, generate incorrect formats, violate length or sample count constraints, and be unfaithful to the input text. Existing benchmarks are insufficient to assess LLMs’ ability to understand complex instructions, as they are close-ended and simple. To bridge this gap, we propose CELLO, a benchmark for evaluating LLMs' ability to follow complex instructions systematically. We design eight features for complex instructions and construct a comprehensive evaluation dataset from real-world scenarios. We also establish four criteria and develop corresponding metrics, as current ones are inadequate, biased or too strict and coarse-grained. We compare the performance of representative Chinese-oriented and English-oriented models in following complex instructions through extensive experiments. Resources of CELLO are publicly available at https://github.com/Abbey4799/CELLO.

----

## [2029] Improving Factual Error Correction by Learning to Inject Factual Errors

**Authors**: *Xingwei He, Qianru Zhang, A-Long Jin, Jun Ma, Yuan Yuan, Siu Ming Yiu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29778](https://doi.org/10.1609/aaai.v38i16.29778)

**Abstract**:

Factual error correction (FEC) aims to revise factual errors in false claims with minimal editing, making them faithful to the provided evidence. This task is crucial for alleviating the hallucination problem encountered by large language models. Given the lack of paired data (i.e., false claims and their corresponding correct claims), existing methods typically adopt the ‘mask-then-correct’ paradigm. This paradigm relies solely on unpaired false claims and correct claims, thus being referred to as distantly supervised methods. These methods require a masker to explicitly identify factual errors within false claims before revising with a corrector. However, the absence of paired data to train the masker makes accurately pinpointing factual errors within claims challenging. To mitigate this, we propose to improve FEC by Learning to Inject Factual Errors (LIFE), a three-step distantly supervised method: ‘mask-corrupt-correct’. Specifically, we first train a corruptor using the ‘mask-then-corrupt’ procedure, allowing it to deliberately introduce factual errors into correct text. The corruptor is then applied to correct claims, generating a substantial amount of paired data. After that, we filter out low-quality data, and use the remaining data to train a corrector. Notably, our corrector does not require a masker, thus circumventing the bottleneck associated with explicit factual error identification. Our experiments on a public dataset verify the effectiveness of LIFE in two key aspects: Firstly, it outperforms the previous best-performing distantly supervised method by a notable margin of 10.59 points in SARI Final (19.3% improvement). Secondly, even compared to ChatGPT prompted with in-context examples, LIFE achieves a superiority of 7.16 points in SARI Final.

----

## [2030] Text2Analysis: A Benchmark of Table Question Answering with Advanced Data Analysis and Unclear Queries

**Authors**: *Xinyi He, Mengyu Zhou, Xinrun Xu, Xiaojun Ma, Rui Ding, Lun Du, Yan Gao, Ran Jia, Xu Chen, Shi Han, Zejian Yuan, Dongmei Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29779](https://doi.org/10.1609/aaai.v38i16.29779)

**Abstract**:

Tabular data analysis is crucial in various fields, and large language models show promise in this area. However, current research mostly focuses on rudimentary tasks like Text2SQL and TableQA, neglecting advanced analysis like forecasting and chart generation. To address this gap, we developed the Text2Analysis benchmark, incorporating advanced analysis tasks that go beyond the SQL-compatible operations and require more in-depth analysis. We also develop five innovative and effective annotation methods, harnessing the capabilities of large language models to enhance data quality and quantity. Additionally, we include unclear queries that resemble real-world user questions to test how well models can understand and tackle such challenges. Finally, we collect 2249 query-result pairs with 347 tables. We evaluate five state-of-the-art models using three different metrics and the results show that our benchmark presents introduces considerable challenge  in the field of tabular data analysis, paving the way for more advanced research opportunities.

----

## [2031] ParaGuide: Guided Diffusion Paraphrasers for Plug-and-Play Textual Style Transfer

**Authors**: *Zachary Horvitz, Ajay Patel, Chris Callison-Burch, Zhou Yu, Kathleen R. McKeown*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29780](https://doi.org/10.1609/aaai.v38i16.29780)

**Abstract**:

Textual style transfer is the task of transforming stylistic properties of text while preserving meaning. Target "styles" can be defined in numerous ways, ranging from single attributes (e.g. formality) to authorship (e.g. Shakespeare). Previous unsupervised style-transfer approaches generally rely on significant amounts of labeled data for only a fixed set of styles or require large language models. In contrast, we introduce a novel diffusion-based framework for general-purpose style transfer that can be flexibly adapted to arbitrary target styles at inference time. Our parameter-efficient approach, ParaGuide, leverages paraphrase-conditioned diffusion models alongside gradient-based guidance from both off-the-shelf classifiers and strong existing style embedders to transform the style of text while preserving semantic information. We validate the method on the Enron Email Corpus, with both human and automatic evaluations, and find that it outperforms strong baselines on formality, sentiment, and even authorship style transfer.

----

## [2032] ShareBERT: Embeddings Are Capable of Learning Hidden Layers

**Authors**: *Jia-Cheng Hu, Roberto Cavicchioli, Giulia Berardinelli, Alessandro Capotondi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29781](https://doi.org/10.1609/aaai.v38i16.29781)

**Abstract**:

The deployment of Pre-trained Language Models in memory-limited devices is hindered by their massive number of parameters, which motivated the interest in developing smaller architectures.
Established works in the model compression literature showcased that small models often present a noticeable performance degradation and need to be paired with transfer learning methods, such as Knowledge Distillation. 
In this work, we propose a parameter-sharing method that consists of sharing parameters between embeddings and the hidden layers, enabling the design of near-zero parameter encoders. To demonstrate its effectiveness, we present an architecture design called ShareBERT, which can preserve up to 95.5%
of BERT Base performances, using only 5M parameters (21.9× fewer parameters) without the help of Knowledge Distillation. We demonstrate empirically that our proposal does not negatively affect the model learning capabilities and that it is even beneficial for representation learning. Code will be available at https://github.com/jchenghu/sharebert.

----

## [2033] LLM vs Small Model? Large Language Model Based Text Augmentation Enhanced Personality Detection Model

**Authors**: *Linmei Hu, Hongyu He, Duokang Wang, Ziwang Zhao, Yingxia Shao, Liqiang Nie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29782](https://doi.org/10.1609/aaai.v38i16.29782)

**Abstract**:

Personality detection aims to detect one's personality traits underlying in social media posts. One challenge of this task is the scarcity of ground-truth personality traits which are collected from self-report questionnaires. Most existing methods learn post features directly by fine-tuning the pre-trained language models under the supervision of limited personality labels. This leads to inferior quality of post features and consequently affects the performance. In addition, they treat personality traits as one-hot classification labels, overlooking the semantic information within them. In this paper, we propose a large language model (LLM) based text augmentation enhanced personality detection model, which distills the LLM's knowledge to enhance the small model for personality detection, even when the LLM fails in this task. Specifically, we enable LLM to generate post analyses (augmentations) from the aspects of semantic, sentiment, and linguistic, which are critical for personality detection. By using contrastive learning to pull them together in the embedding space, the post encoder can better capture the psycho-linguistic information within the post representations, thus improving personality detection. Furthermore, we utilize the LLM to enrich the information of personality labels for enhancing the detection performance. Experimental results on the benchmark datasets demonstrate that our model outperforms the state-of-the-art methods on personality detection.

----

## [2034] Learning Robust Rationales for Model Explainability: A Guidance-Based Approach

**Authors**: *Shuaibo Hu, Kui Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29783](https://doi.org/10.1609/aaai.v38i16.29783)

**Abstract**:

Selective rationalization can be regarded as a straightforward self-explaining approach for enhancing model explainability in natural language processing tasks. It aims to provide explanations that are more accessible and understandable to non-technical users by first selecting subsets of input texts as rationales and then predicting based on chosen subsets. However, existing methods that follow this select-then-predict framework may suffer from the rationalization degeneration problem, resulting in sub-optimal or unsatisfactory rationales that do not align with human judgments. This problem may further lead to rationalization failure, resulting in meaningless rationales that ultimately undermine people's trust in the rationalization model. To address these challenges, we propose a Guidance-based Rationalization method (G-RAT) that effectively improves robustness against failure situations and the quality of rationales by using a guidance module to regularize selections and distributions. Experimental results on two synthetic settings prove that our method is robust to the rationalization degeneration and failure problems, while the results on two real datasets show its effectiveness in providing rationales in line with human judgments. The source code is available at https://github.com/shuaibo919/g-rat.

----

## [2035] Separate the Wheat from the Chaff: Model Deficiency Unlearning via Parameter-Efficient Module Operation

**Authors**: *Xinshuo Hu, Dongfang Li, Baotian Hu, Zihao Zheng, Zhenyu Liu, Min Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29784](https://doi.org/10.1609/aaai.v38i16.29784)

**Abstract**:

Large language models (LLMs) have been widely used in various applications but are known to suffer from issues related to untruthfulness and toxicity. While parameter-efficient modules (PEMs) have demonstrated their effectiveness in equipping models with new skills, leveraging PEMs for deficiency unlearning remains underexplored. In this work, we propose a PEMs operation approach, namely Extraction-before-Subtraction (Ext-Sub), to enhance the truthfulness and detoxification of LLMs through the integration of ``expert'' PEM and ``anti-expert'' PEM. Remarkably, even anti-expert PEM possess valuable capabilities due to their proficiency in generating fabricated content, which necessitates language modeling and logical narrative competence. Rather than merely negating the parameters, our approach involves extracting and eliminating solely the deficiency capability within anti-expert PEM while preserving the general capabilities. To evaluate the effectiveness of our approach in terms of truthfulness and detoxification, we conduct extensive experiments on LLMs, encompassing additional abilities such as language modelling and mathematical reasoning. Our empirical results demonstrate that our approach effectively improves truthfulness and detoxification, while largely preserving the fundamental abilities of LLMs.

----

## [2036] Three Heads Are Better than One: Improving Cross-Domain NER with Progressive Decomposed Network

**Authors**: *Xuming Hu, Zhaochen Hong, Yong Jiang, Zhichao Lin, Xiaobin Wang, Pengjun Xie, Philip S. Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29785](https://doi.org/10.1609/aaai.v38i16.29785)

**Abstract**:

Cross-domain named entity recognition (NER) tasks encourage NER models to transfer knowledge from data-rich source domains to sparsely labeled target domains. Previous works adopt the paradigms of pre-training on the source domain followed by fine-tuning on the target domain. However, these works ignore that general labeled NER source domain data can be easily retrieved in the real world, and soliciting more source domains could bring more benefits. Unfortunately, previous paradigms cannot efficiently transfer knowledge from multiple source domains. In this work, to transfer multiple source domains' knowledge, we decouple the NER task into the pipeline tasks of mention detection and entity typing, where the mention detection unifies the training object across domains, thus providing the entity typing with higher-quality entity mentions. Additionally, we request multiple general source domain models to suggest the potential named entities for sentences in the target domain explicitly, and transfer their knowledge to the target domain models through the knowledge progressive networks implicitly. Furthermore, we propose two methods to analyze in which source domain knowledge transfer occurs, thus helping us judge which source domain brings the greatest benefit. In our experiment, we develop a Chinese cross-domain NER dataset. Our model improved the F1 score by an average of 12.50% across 8 Chinese and English datasets compared to models without source domain data.

----

## [2037] Uncovering and Mitigating the Hidden Chasm: A Study on the Text-Text Domain Gap in Euphemism Identification

**Authors**: *Yuxue Hu, Junsong Li, Mingmin Wu, Zhongqiang Huang, Gang Chen, Ying Sha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29786](https://doi.org/10.1609/aaai.v38i16.29786)

**Abstract**:

Euphemisms are commonly used on social media and darknet marketplaces to evade platform regulations by masking their true meanings with innocent ones. For instance, “weed” is used instead of “marijuana” for illicit transactions. Thus, euphemism identification, i.e., mapping a given euphemism (“weed”) to its specific target word (“marijuana”), is essential for improving content moderation and combating underground markets. Existing methods employ self-supervised schemes to automatically construct labeled training datasets for euphemism identification. However, they overlook the text-text domain gap caused by the discrepancy between the constructed training data and the test data, leading to performance deterioration. In this paper, we present the text-text domain gap and explain how it forms in terms of the data distribution and the cone effect. Moreover, to bridge this gap, we introduce a feature alignment network (FA-Net), which can both align the in-domain and cross-domain features, thus mitigating the domain gap from training data to test data and improving the performance of the base models for euphemism identification. We apply this FA-Net to the base models, obtaining markedly better results, and creating a state-of-the-art model which beats the large language models.

----

## [2038] PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation

**Authors**: *Zhiyuan Hu, Chumin Liu, Yue Feng, Anh Tuan Luu, Bryan Hooi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29787](https://doi.org/10.1609/aaai.v38i16.29787)

**Abstract**:

Controllable text generation is a challenging and meaningful field in natural language generation (NLG). Especially, poetry generation is a typical one with well-defined and strict conditions for text generation which is an ideal playground for the assessment of current methodologies. While prior works succeeded in controlling either semantic or metrical aspects of poetry generation, simultaneously addressing both remains a challenge. In this paper, we pioneer the use of the Diffusion model for generating sonnets and Chinese SongCi poetry to tackle such challenges. In terms of semantics, our PoetryDiffusion model, built upon the Diffusion model, generates entire sentences or poetry by comprehensively considering the entirety of sentence information. This approach enhances semantic expression, distinguishing it from autoregressive and large language models (LLMs). For metrical control, its constraint control module which can be trained individually enables us to flexibly incorporate a novel metrical controller to manipulate and evaluate metrics (format and rhythm).
The denoising process in PoetryDiffusion allows for the gradual enhancement of semantics and flexible integration of the metrical controller which can calculate and impose penalties on states that stray significantly from the target control distribution. Experimental results on two datasets demonstrate that our model outperforms existing models in terms of automatic evaluation of semantic, metrical, and overall performance as well as human evaluation. Codes are released to https://github.com/ChorlingLau/PoetryDiffusion.

----

## [2039] Towards Equipping Transformer with the Ability of Systematic Compositionality

**Authors**: *Chen Huang, Peixin Qin, Wenqiang Lei, Jiancheng Lv*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29788](https://doi.org/10.1609/aaai.v38i16.29788)

**Abstract**:

One of the key factors in language productivity and human cognition is the ability of Systematic Compositionality, which refers to understanding composed, unseen examples of seen primitives. However, recent evidence reveals that the Transformers have difficulty in generalizing the composed context based on the seen primitives. To this end, we take the first step to propose a compositionality-aware Transformer called CAT and two novel pre-training tasks to facilitate the systematic compositionality. We tentatively provide a successful implementation of a multi-layer CAT on the basis of the especially popular BERT. The experimental results demonstrate that CAT outperforms baselines on compositionality-aware tasks with minimal impact on effectiveness on standardized language understanding tasks.

----

## [2040] Cross-Modal and Uni-Modal Soft-Label Alignment for Image-Text Retrieval

**Authors**: *Hailang Huang, Zhijie Nie, Ziqiao Wang, Ziyu Shang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29789](https://doi.org/10.1609/aaai.v38i16.29789)

**Abstract**:

Current image-text retrieval methods have demonstrated impressive performance in recent years. However, they still face two problems: the inter-modal matching missing problem and the intra-modal semantic loss problem. These problems can significantly affect the accuracy of image-text retrieval. To address these challenges, we propose a novel method called Cross-modal and Uni-modal Soft-label Alignment (CUSA). Our method leverages the power of uni-modal pre-trained models to provide soft-label supervision signals for the image-text retrieval model. Additionally, we introduce two alignment techniques, Cross-modal Soft-label Alignment (CSA) and Uni-modal Soft-label Alignment (USA), to overcome false negatives and enhance similarity recognition between uni-modal samples. Our method is designed to be plug-and-play, meaning it can be easily applied to existing image-text retrieval models without changing their original architectures. Extensive experiments on various image-text retrieval models and datasets, we demonstrate that our method can consistently improve the performance of image-text retrieval and achieve new state-of-the-art results. Furthermore, our method can also boost the uni-modal retrieval performance of image-text retrieval models, enabling it to achieve universal retrieval. The code and supplementary files can be found at https://github.com/lerogo/aaai24_itr_cusa.

----

## [2041] Response Enhanced Semi-supervised Dialogue Query Generation

**Authors**: *Jianheng Huang, Ante Wang, Linfeng Gao, Linfeng Song, Jinsong Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29790](https://doi.org/10.1609/aaai.v38i16.29790)

**Abstract**:

Leveraging vast and continually updated knowledge from the Internet has been considered an important ability for a dialogue system. Therefore, the dialogue query generation task is proposed for generating search queries from dialogue histories, which will be submitted to a search engine for retrieving relevant websites on the Internet. In this regard, previous efforts were devoted to collecting conversations with annotated queries and training a query producer (QP) via standard supervised learning. However, these studies still face the challenges of data scarcity and domain adaptation.
To address these issues, in this paper, we propose a semi-supervised learning framework -- SemiDQG, to improve model performance with unlabeled conversations. Based on the observation that the search query is typically related to the topic of dialogue response, we train a response-augmented query producer (RA) to provide rich and effective training signals for QP.
We first apply a similarity-based query selection strategy to select high-quality RA-generated pseudo queries, which are used to construct pseudo instances for training QP and RA.
Then, we adopt the REINFORCE algorithm to further enhance QP, with RA-provided rewards as fine-grained training signals. Experimental results and in-depth analysis of three benchmarks show the effectiveness of our framework in cross-domain and low-resource scenarios. Particularly, SemiDQG significantly surpasses ChatGPT and competitive baselines. Our code is available at \url{https://github.com/DeepLearnXMU/SemiDQG}.

----

## [2042] PMRC: Prompt-Based Machine Reading Comprehension for Few-Shot Named Entity Recognition

**Authors**: *Jin Huang, Danfeng Yan, Yuanqiang Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29791](https://doi.org/10.1609/aaai.v38i16.29791)

**Abstract**:

The prompt-based method has been proven effective in improving the performance of pre-trained language models (PLMs) on sentence-level few-shot tasks. However, when applying prompting to token-level tasks such as Named Entity Recognition (NER), specific templates need to be designed, and all possible segments of the input text need to be enumerated. These methods have high computational complexity in both training and inference processes, making them difficult to apply in real-world scenarios. To address these issues, we redefine the NER task as a Machine Reading Comprehension (MRC) task and incorporate prompting into the MRC framework. Specifically, we sequentially insert boundary markers for various entity types into the templates and use these markers as anchors during the inference process to differentiate entity types. In contrast to the traditional multi-turn question-answering extraction in the MRC framework, our method can extract all spans of entity types in one round. Furthermore, we propose word-based template and example-based template that enhance the MRC framework's perception of entity start and end positions while significantly reducing the manual effort required for template design. It is worth noting that in cross-domain scenarios, PMRC does not require redesigning the model architecture and can continue training by simply replacing the templates to recognize entity types in the target domain. Experimental results demonstrate that our approach outperforms state-of-the-art models in low-resource settings, achieving an average performance improvement of +5.2% in settings where access to source domain data is limited. Particularly, on the ATIS dataset with a large number of entity types and 10-shot setting, PMRC achieves a performance improvement of +15.7%. Moreover, our method achieves a decoding speed 40.56 times faster than the template-based cloze-style approach.

----

## [2043] Revisiting Document-Level Relation Extraction with Context-Guided Link Prediction

**Authors**: *Monika Jain, Raghava Mutharaju, Ramakanth Kavuluru, Kuldeep Singh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29792](https://doi.org/10.1609/aaai.v38i16.29792)

**Abstract**:

Document-level relation extraction (DocRE) poses the challenge of identifying relationships between entities within a document. Existing approaches rely on logical reasoning or contextual cues from entities. This paper reframes document-level RE as link prediction over a Knowledge Graph (KG) with distinct benefits: 1) Our approach amalgamates entity context and document-derived logical reasoning, enhancing link prediction quality. 2) Predicted links between entities offer interpretability, elucidating employed reasoning. We evaluate our approach on benchmark datasets - DocRED, ReDocRED, and DWIE. The results indicate that our proposed method outperforms the state-of-the-art models and suggests that incorporating context-based Knowledge Graph link prediction techniques can enhance the performance of document-level relation extraction models.

----

## [2044] Enhancing Zero-Shot Multi-Speaker TTS with Negated Speaker Representations

**Authors**: *Yejin Jeon, Yunsu Kim, Gary Geunbae Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29793](https://doi.org/10.1609/aaai.v38i16.29793)

**Abstract**:

Zero-shot multi-speaker TTS aims to synthesize speech with the voice of a chosen target speaker without any fine-tuning. Prevailing methods, however, encounter limitations at adapting to new speakers of out-of-domain settings, primarily due to inadequate speaker disentanglement and content leakage. To overcome these constraints, we propose an innovative negation feature learning paradigm that models decoupled speaker attributes as deviations from the complete audio representation by utilizing the subtraction operation. By eliminating superfluous content information from the speaker representation, our negation scheme not only mitigates content leakage, thereby enhancing synthesis robustness, but also improves speaker fidelity. In addition, to facilitate the learning of diverse speaker attributes, we leverage multi-stream Transformers, which retain multiple hypotheses and instigate a training paradigm akin to ensemble learning. To unify these hypotheses and realize the final speaker representation, we employ attention pooling. Finally, in light of the imperative to generate target text utterances in the desired voice, we adopt adaptive layer normalizations to effectively fuse the previously generated speaker representation with the target text representations, as opposed to mere concatenation of the text and audio modalities. Extensive experiments and validations substantiate the efficacy of our proposed approach in preserving and harnessing speaker-specific attributes vis-à-vis alternative baseline models.

----

## [2045] Chain-of-Thought Improves Text Generation with Citations in Large Language Models

**Authors**: *Bin Ji, Huijun Liu, Mingzhe Du, See-Kiong Ng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29794](https://doi.org/10.1609/aaai.v38i16.29794)

**Abstract**:

Previous studies disclose that Large Language Models (LLMs) suffer from hallucinations when generating texts, bringing a novel and challenging research topic to the public, which centers on enabling LLMs to generate texts with citations. Existing work exposes two limitations when using LLMs to generate answers to questions with provided documents: unsatisfactory answer correctness and poor citation quality. To tackle the above issues, we investigate using Chain-of-Thought (CoT) to elicit LLMs’ ability to synthesize correct answers from multiple documents, as well as properly cite these documents. Moreover, we propose a Citation Insurance Mechanism, which enables LLMs to detect and cite those missing citations. We conduct experiments on the ALCE benchmark with six open-source LLMs. Experimental results demonstrate that: (1) the CoT prompting strategy significantly improves the quality of text generation with citations; (2) the Citation Insurance Mechanism delivers impressive gains in citation quality at a low cost; (3) our best approach performs comparably as previous best ChatGPT-based baselines. Extensive analyses further validate the effectiveness of the proposed approach.

----

## [2046] Debiasing Multimodal Sarcasm Detection with Contrastive Learning

**Authors**: *Mengzhao Jia, Can Xie, Liqiang Jing*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29795](https://doi.org/10.1609/aaai.v38i16.29795)

**Abstract**:

Despite commendable achievements made by existing work, prevailing multimodal sarcasm detection studies rely more on textual content over visual information. It unavoidably induces spurious correlations between textual words and labels, thereby significantly hindering the models' generalization capability. To address this problem, we define the task of out-of-distribution (OOD) multimodal sarcasm detection, which aims to evaluate models' generalizability when the word distribution is different in training and testing settings. Moreover, we propose a novel debiasing multimodal sarcasm detection framework with contrastive learning, which aims to mitigate the harmful effect of biased textual factors for robust OOD generalization. In particular, we first design counterfactual data augmentation to construct the positive samples with dissimilar word biases and negative samples with similar word biases. Subsequently, we devise an adapted debiasing contrastive learning mechanism to empower the model to learn robust task-relevant features and alleviate the adverse effect of biased words. Extensive experiments show the superiority of the proposed framework.

----

## [2047] ZO-AdaMU Optimizer: Adapting Perturbation by the Momentum and Uncertainty in Zeroth-Order Optimization

**Authors**: *Shuoran Jiang, Qingcai Chen, Youcheng Pan, Yang Xiang, Yukang Lin, Xiangping Wu, Chuanyi Liu, Xiaobao Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29796](https://doi.org/10.1609/aaai.v38i16.29796)

**Abstract**:

Lowering the memory requirement in full-parameter training on large models has become a hot research area. MeZO fine-tunes the large language models (LLMs) by just forward passes in a zeroth-order SGD optimizer (ZO-SGD), demonstrating excellent performance with the same GPU memory usage as inference. However, the simulated perturbation stochastic approximation for gradient estimate in MeZO leads to severe oscillations and incurs a substantial time overhead. Moreover, without momentum regularization, MeZO shows severe over-fitting problems. Lastly, the perturbation-irrelevant momentum on ZO-SGD does not improve the convergence rate. This study proposes ZO-AdaMU to resolve the above problems by adapting the simulated perturbation with momentum in its stochastic approximation. Unlike existing adaptive momentum methods, we relocate momentum on simulated perturbation in stochastic gradient approximation. Our convergence analysis and experiments prove this is a better way to improve convergence stability and rate in ZO-SGD. Extensive experiments demonstrate that ZO-AdaMU yields better generalization for LLMs fine-tuning across various NLP tasks than MeZO and its momentum variants.

----

## [2048] Unsupervised Extractive Summarization with Learnable Length Control Strategies

**Authors**: *Renlong Jie, Xiaojun Meng, Xin Jiang, Qun Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29797](https://doi.org/10.1609/aaai.v38i16.29797)

**Abstract**:

Unsupervised extractive summarization is an important technique in information extraction and retrieval. Compared with supervised method, it does not require high-quality human-labelled summaries for training and thus can be easily applied for documents with different types, domains or languages. Most of existing unsupervised methods including TextRank and PACSUM rely on graph-based ranking on sentence centrality. However, this scorer can not be directly applied in end-to-end training, and the positional-related prior assumption is often needed for achieving good summaries. In addition, less attention is paid to length-controllable extractor, where users can decide to summarize texts under particular length constraint. This paper introduces an unsupervised extractive summarization model based on a siamese network, for which we develop a trainable bidirectional prediction objective between the selected summary and the original document. Different from the centrality-based ranking methods, our extractive scorer can be trained in an end-to-end manner, with no other requirement of positional assumption. In addition, we introduce a differentiable length control module by approximating 0-1 knapsack solver for end-to-end length-controllable extracting. Experiments show that our unsupervised method largely outperforms the centrality-based baseline using a same sentence encoder. In terms of length control ability, via our trainable knapsack module, the performance consistently outperforms the strong baseline without utilizing end-to-end training. Human evaluation further evidences that our method performs the best among baselines in terms of relevance and consistency.

----

## [2049] BOK-VQA: Bilingual outside Knowledge-Based Visual Question Answering via Graph Representation Pretraining

**Authors**: *MinJun Kim, Seungwoo Song, Youhan Lee, Haneol Jang, Kyungtae Lim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29798](https://doi.org/10.1609/aaai.v38i16.29798)

**Abstract**:

The current research direction in generative models, such as the recently developed GPT4, aims to find relevant knowledge information for multimodal and multilingual inputs to provide answers. Under these research circumstances, the demand for multilingual evaluation of visual question answering (VQA) tasks, a representative task of multimodal systems, has increased. Accordingly, we propose a bilingual outside-knowledge VQA (BOK-VQA) dataset in this study that can be extended to multilingualism. The proposed data include 17K images, 17K question-answer pairs for both Korean and English and 280K instances of knowledge information related to question-answer content. We also present a framework that can effectively inject knowledge information into a VQA system by pretraining the knowledge information of BOK-VQA data in the form of graph embeddings. Finally, through in-depth analysis, we demonstrated the actual effect of the knowledge information contained in the constructed training data on VQA.

----

## [2050] Improving Knowledge Extraction from LLMs for Task Learning through Agent Analysis

**Authors**: *James R. Kirk, Robert E. Wray, Peter Lindes, John E. Laird*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29799](https://doi.org/10.1609/aaai.v38i16.29799)

**Abstract**:

Large language models (LLMs) offer significant promise as a knowledge source for task learning. Prompt engineering has been shown to be effective for eliciting knowledge from an LLM, but alone it is insufficient for acquiring relevant, situationally grounded knowledge for an embodied agent learning novel tasks. We describe a cognitive-agent approach, STARS, that extends and complements prompt engineering, mitigating its limitations and thus enabling an agent to acquire new task knowledge matched to its native language capabilities, embodiment, environment, and user preferences. The STARS approach is to increase the response space of LLMs and deploy general strategies, embedded within the autonomous agent, to evaluate, repair, and select among candidate responses produced by the LLM. We describe the approach and experiments that show how an agent, by retrieving and evaluating a breadth of responses from the LLM, can achieve 77-94% task completion in one-shot learning without user oversight. The approach achieves 100% task completion when human oversight (such as an indication of preference) is provided. Further, the type of oversight largely shifts from explicit, natural language instruction to simple confirmation/discomfirmation of high-quality responses that have been vetted by the agent before presentation to a user.

----

## [2051] On Unsupervised Domain Adaptation: Pseudo Label Guided Mixup for Adversarial Prompt Tuning

**Authors**: *Fanshuang Kong, Richong Zhang, Ziqiao Wang, Yongyi Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29800](https://doi.org/10.1609/aaai.v38i16.29800)

**Abstract**:

To date, a backbone of methods for unsupervised domain adaptation (UDA) involves learning label-discriminative features via a label classifier and domain-invariant features through a domain discriminator in an adversarial scheme. However, these methods lack explicit control for aligning the source data and target data within the same label class, degrading the classifier's performance in the target domain. In this paper, we propose PL-Mix, a pseudo label guided Mixup method based on adversarial prompt tuning. Specifically, our PL-Mix facilitates class-dependent alignment and can alleviate the impact of noisy pseudo-labels. We then theoretically justify that PL-Mix can improve the generalization for UDA. Extensive experiments of the comparison with existing models also demonstrate the effectiveness of PL-Mix.

----

## [2052] A Hierarchical Network for Multimodal Document-Level Relation Extraction

**Authors**: *Lingxing Kong, Jiuliang Wang, Zheng Ma, Qifeng Zhou, Jianbing Zhang, Liang He, Jiajun Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29801](https://doi.org/10.1609/aaai.v38i16.29801)

**Abstract**:

Document-level relation extraction aims to extract entity relations that span across multiple sentences. This task faces two critical issues: long dependency and mention selection. Prior works address the above problems from the textual perspective, however, it is hard to handle these problems solely based on text information. In this paper, we leverage video information to provide additional evidence for understanding long dependencies and offer a wider perspective for identifying relevant mentions, thus giving rise to a new task named Multimodal Document-level Relation Extraction (MDocRE). To tackle this new task, we construct a human-annotated dataset including documents and relevant videos, which, to the best of our knowledge, is the first document-level relation extraction dataset equipped with video clips. We also propose a hierarchical framework to learn interactions between different dependency levels and a textual-guided transformer architecture that incorporates both textual and video modalities. In addition, we utilize a mention gate module to address the mention-selection problem in both modalities. Experiments on our proposed dataset show that 1) incorporating video information greatly improves model performance; 2) our hierarchical framework has state-of-the-art results compared with both unimodal and multimodal baselines; 3) through collaborating with video information, our model better solves the long-dependency and mention-selection problems.

----

## [2053] Large Language Models Are Clinical Reasoners: Reasoning-Aware Diagnosis Framework with Prompt-Generated Rationales

**Authors**: *Taeyoon Kwon, Kai Tzu-iunn Ong, Dongjin Kang, Seungjun Moon, Jeong Ryong Lee, Dosik Hwang, Beomseok Sohn, Yongsik Sim, Dongha Lee, Jinyoung Yeo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29802](https://doi.org/10.1609/aaai.v38i16.29802)

**Abstract**:

Machine reasoning has made great progress in recent years owing to large language models (LLMs). In the clinical domain, however, most NLP-driven projects mainly focus on clinical classification or reading comprehension, and under-explore clinical reasoning for disease diagnosis due to the expensive rationale annotation with clinicians. In this work, we present a "reasoning-aware" diagnosis framework that rationalizes the diagnostic process via prompt-based learning in a time- and labor-efficient manner, and learns to reason over the prompt-generated rationales. Specifically, we address the clinical reasoning for disease diagnosis, where the LLM generates diagnostic rationales providing its insight on presented patient data and the reasoning path towards the diagnosis, namely Clinical Chain-of-Thought (Clinical CoT). We empirically demonstrate LLMs/LMs' ability of clinical reasoning via extensive experiments and analyses on both rationale generation and disease diagnosis in various settings. We further propose a novel set of criteria for evaluating machine-generated rationales' potential for real-world clinical settings, facilitating and benefiting future research in this area.

----

## [2054] Frequency Spectrum Is More Effective for Multimodal Representation and Fusion: A Multimodal Spectrum Rumor Detector

**Authors**: *An Lao, Qi Zhang, Chongyang Shi, Longbing Cao, Kun Yi, Liang Hu, Duoqian Miao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29803](https://doi.org/10.1609/aaai.v38i16.29803)

**Abstract**:

Multimodal content, such as mixing text with images, presents significant challenges to rumor detection in social media. Existing multimodal rumor detection has focused on mixing tokens among spatial and sequential locations for unimodal representation or fusing clues of rumor veracity across modalities. However, they suffer from less discriminative unimodal representation and are vulnerable to intricate location dependencies in the time-consuming fusion of spatial and sequential tokens. This work makes the first attempt at multimodal rumor detection in the frequency domain, which efficiently transforms spatial features into the frequency spectrum and obtains highly discriminative spectrum features for multimodal representation and fusion. A novel Frequency Spectrum Representation and fUsion network (FSRU) with dual contrastive learning reveals the frequency spectrum is more effective for multimodal representation and fusion, extracting the informative components for rumor detection. FSRU involves three novel mechanisms: utilizing the Fourier transform to convert features in the spatial domain to the frequency domain, the unimodal spectrum compression, and the cross-modal spectrum co-selection module in the frequency domain. Substantial experiments show that FSRU achieves satisfactory multimodal rumor detection performance.

----

## [2055] LAMPAT: Low-Rank Adaption for Multilingual Paraphrasing Using Adversarial Training

**Authors**: *Khoi M. Le, Trinh Pham, Tho Quan, Anh Tuan Luu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29804](https://doi.org/10.1609/aaai.v38i16.29804)

**Abstract**:

Paraphrases are texts that convey the same meaning while using different words or sentence structures. It can be used as an automatic data augmentation tool for many Natural Language Processing tasks, especially when dealing with low-resource languages, where data shortage is a significant problem. To generate a paraphrase in multilingual settings, previous studies have leveraged the knowledge from the machine translation field, i.e., forming a paraphrase through zero-shot machine translation in the same language. Despite good performance on human evaluation, those methods still require parallel translation datasets, thus making them inapplicable to languages that do not have parallel corpora. To mitigate that problem, we proposed the first unsupervised multilingual paraphrasing model, LAMPAT (Low-rank Adaptation for Multilingual Paraphrasing using Adversarial Training), by which monolingual dataset is sufficient enough to generate a human-like and diverse sentence. Throughout the experiments, we found out that our method not only works well for English but can generalize on unseen languages as well. Data and code are available at https://github.com/phkhanhtrinh23/LAMPAT.

----

## [2056] Continual Relation Extraction via Sequential Multi-Task Learning

**Authors**: *Thanh-Thien Le, Manh Nguyen, Tung Thanh Nguyen, Ngo Van Linh, Thien Huu Nguyen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29805](https://doi.org/10.1609/aaai.v38i16.29805)

**Abstract**:

To build continual relation extraction (CRE) models, those can adapt to an ever-growing ontology of relations, is a cornerstone information extraction task that serves in various dynamic real-world domains. To mitigate catastrophic forgetting in CRE, existing state-of-the-art approaches have effectively utilized rehearsal techniques from continual learning and achieved remarkable success. However, managing multiple objectives associated with memory-based rehearsal remains underexplored, often relying on simple summation and overlooking complex trade-offs. In this paper, we propose Continual Relation Extraction via Sequential Multi-task Learning (CREST), a novel CRE approach built upon a tailored Multi-task Learning framework for continual learning. CREST takes into consideration the disparity in the magnitudes of gradient signals of different objectives, thereby effectively handling the inherent difference between multi-task learning and continual learning. Through extensive experiments on multiple datasets, CREST demonstrates significant improvements in CRE performance as well as superiority over other state-of-the-art Multi-task Learning frameworks, offering a promising solution to the challenges of continual learning in this domain.

----

## [2057] Labels Need Prompts Too: Mask Matching for Natural Language Understanding Tasks

**Authors**: *Bo Li, Wei Ye, Quansen Wang, Wen Zhao, Shikun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29806](https://doi.org/10.1609/aaai.v38i16.29806)

**Abstract**:

Textual label names (descriptions) are typically semantically rich in many natural language understanding (NLU) tasks. In this paper, we incorporate the prompting methodology, which is widely used to enrich model input, into the label side for the first time. Specifically, we propose a Mask Matching method, which equips an input with a prompt and its label with another, and then makes predictions by matching their mask representations. We evaluate our method extensively on 8 NLU tasks with 14 datasets. The experimental results show that Mask Matching significantly outperforms its counterparts of fine-tuning and conventional prompt-tuning, setting up state-of-the-art performances in several datasets. Mask Matching is particularly good at handling NLU tasks with large label counts and informative label names. As pioneering efforts that investigate the label-side prompt, we also discuss open issues for future study.

----

## [2058] Harnessing Holistic Discourse Features and Triadic Interaction for Sentiment Quadruple Extraction in Dialogues

**Authors**: *Bobo Li, Hao Fei, Lizi Liao, Yu Zhao, Fangfang Su, Fei Li, Donghong Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29807](https://doi.org/10.1609/aaai.v38i16.29807)

**Abstract**:

Dialogue Aspect-based Sentiment Quadruple (DiaASQ) is a newly-emergent task aiming to extract the sentiment quadruple (i.e., targets, aspects, opinions, and sentiments) from conversations. While showing promising performance, the prior DiaASQ approach unfortunately falls prey to the key crux of DiaASQ, including insufficient modeling of discourse features, and lacking quadruple extraction, which hinders further task improvement. To this end, we introduce a novel framework that not only capitalizes on comprehensive discourse feature modeling, but also captures the intrinsic interaction for optimal quadruple extraction. On the one hand, drawing upon multiple discourse features, our approach constructs a token-level heterogeneous graph and enhances token interactions through a heterogeneous attention network. We further propose a novel triadic scorer, strengthening weak token relations within a quadruple, thereby enhancing the cohesion of the quadruple extraction.  Experimental results on the DiaASQ benchmark showcase that our model significantly outperforms existing baselines across both English and Chinese datasets. Our code is available at https://bit.ly/3v27pqA.

----

## [2059] Task Contamination: Language Models May Not Be Few-Shot Anymore

**Authors**: *Changmao Li, Jeffrey Flanigan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29808](https://doi.org/10.1609/aaai.v38i16.29808)

**Abstract**:

Large language models (LLMs) offer impressive performance in various zero-shot and few-shot tasks. However, their success in zero-shot or few-shot settings may be affected by task contamination, a potential limitation that has not been thoroughly examined. This paper investigates how zero-shot and few-shot performance of LLMs has changed chronologically over datasets released over time, and over  LLMs released over time. Utilizing GPT-3 series models and several other recent open-sourced LLMs, and controlling for dataset difficulty, we find that datasets released prior to the LLM training data creation date perform surprisingly better than datasets released post the LLM training data creation date. This strongly indicates that, for many LLMs, there exists task contamination on zero-shot and few-shot evaluation for datasets prior to the LLMs' training data creation date. Additionally, we utilize training data inspection, training data extraction, and a membership inference attack, which reveal further evidence of task contamination. Importantly, we find that for tasks with no possibility of task contamination, LLMs rarely demonstrate statistically significant improvements over simple majority baselines, in both zero and few-shot settings.

----

## [2060] Dialogue for Prompting: A Policy-Gradient-Based Discrete Prompt Generation for Few-Shot Learning

**Authors**: *Chengzhengxu Li, Xiaoming Liu, Yichen Wang, Duyi Li, Yu Lan, Chao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29809](https://doi.org/10.1609/aaai.v38i16.29809)

**Abstract**:

Prompt-based pre-trained language models (PLMs) paradigm has succeeded substantially in few-shot natural language processing (NLP) tasks. However, prior discrete prompt optimization methods require expert knowledge to design the base prompt set and identify high-quality prompts, which is costly, inefficient, and subjective. Meanwhile, existing continuous prompt optimization methods improve the performance by learning the ideal prompts through the gradient information of PLMs, whose high computational cost, and low readability and generalizability are often concerning. To address the research gap, we propose a Dialogue-comprised Policy-gradient-based Discrete Prompt Optimization (DP_2O) method. We first design a multi-round dialogue alignment strategy for readability prompt set generation based on GPT-4. Furthermore, we propose an efficient prompt screening metric to identify high-quality prompts with linear complexity. Finally, we construct a reinforcement learning (RL) framework based on policy gradients to match the prompts to inputs optimally. By training a policy network with only 0.62M parameters on the tasks in the few-shot setting, DP_2O outperforms the state-of-the-art (SOTA) method by 1.52% in accuracy on average on four open-source datasets. Moreover, subsequent experiments also demonstrate that DP_2O has good universality, robustness and generalization ability.

----

## [2061] DeepSpeed Data Efficiency: Improving Deep Learning Model Quality and Training Efficiency via Efficient Data Sampling and Routing

**Authors**: *Conglong Li, Zhewei Yao, Xiaoxia Wu, Minjia Zhang, Connor Holmes, Cheng Li, Yuxiong He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i16.29810](https://doi.org/10.1609/aaai.v38i16.29810)

**Abstract**:

Recent advances on deep learning models come at the price of formidable training cost. The increasing model size is one of the root causes, but another less-emphasized fact is that data scale is actually increasing at a similar speed as model scale, and the training cost is proportional to both of them. Compared to the rapidly evolving model architecture, how to efficiently use the training data (especially for the expensive foundation model pretraining) is both less explored and difficult to realize due to the lack of a convenient framework that focus on data efficiency capabilities. To this end, we present DeepSpeed Data Efficiency, a framework that makes better use of data, increases training efficiency, and improves model quality. Specifically, we propose and combine two data efficiency techniques: efficient data sampling via a general curriculum learning library, and efficient data routing via a novel random layerwise token dropping technique. For GPT-3 1.3B language model pretraining, our work achieves 12.5x less data/time/cost ($3.7K if rent on Azure), while still maintaining 95% of model quality compared to baseline with full data and cost ($46.3K). For GPT-3 1.3B and BERT-large pretraining, our work can also achieve the same model quality with up to 2x less data/time/cost, or achieve better model quality under same data/time/cost. DeepSpeed Data Efficiency is easy to use and tune, enabling us to easily apply it and verify its benefit on additional tasks including GPT-3 MoE model pretraining and small-scale GPT-2/ViT finetuning.

----

## [2062] Advancing Spatial Reasoning in Large Language Models: An In-Depth Evaluation and Enhancement Using the StepGame Benchmark

**Authors**: *Fangjun Li, David C. Hogg, Anthony G. Cohn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29811](https://doi.org/10.1609/aaai.v38i17.29811)

**Abstract**:

Artificial intelligence (AI) has made remarkable progress across various domains, with large language models like ChatGPT gaining substantial attention for their human-like text-generation capabilities. Despite these achievements, improving spatial reasoning remains a significant challenge for these models. Benchmarks like StepGame evaluate AI spatial reasoning, where ChatGPT has shown unsatisfactory performance. However, the presence of template errors in the benchmark has an impact on the evaluation results. Thus there is potential for ChatGPT to perform better if these template errors are addressed, leading to more accurate assessments of its spatial reasoning capabilities. In this study, we refine the StepGame benchmark, providing a more accurate dataset for model evaluation. We analyze GPT’s spatial reasoning performance on the rectified benchmark, identifying proficiency in mapping natural language text to spatial relations but limitations in multi-hop reasoning. We provide a flawless solution to the benchmark by combining template-to-relation mapping with logic-based reasoning. This combination demonstrates proficiency in performing qualitative reasoning on StepGame without encountering any errors. We then address the limitations of GPT models in spatial reasoning. To improve spatial reasoning, we deploy Chain-of-Thought and Tree-of-thoughts prompting strategies, offering insights into GPT’s cognitive process. Our investigation not only sheds light on model deficiencies but also proposes enhancements, contributing to the advancement of AI with more robust spatial reasoning capabilities.

----

## [2063] Exploiting Auxiliary Caption for Video Grounding

**Authors**: *Hongxiang Li, Meng Cao, Xuxin Cheng, Yaowei Li, Zhihong Zhu, Yuexian Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29812](https://doi.org/10.1609/aaai.v38i17.29812)

**Abstract**:

Video grounding aims to locate a moment of interest matching the given query sentence from an untrimmed video. Previous works ignore the sparsity dilemma in video annotations, which fails to provide the context information between potential events and query sentences in the dataset. In this paper, we contend that exploiting easily available captions which describe general actions, i.e., auxiliary captions defined in our paper, will significantly boost the performance. To this end, we propose an Auxiliary Caption Network (ACNet) for video grounding. Specifically, we first introduce dense video captioning to generate dense captions and then obtain auxiliary captions by Non-Auxiliary Caption Suppression (NACS). To capture the potential information in auxiliary captions, we propose Caption Guided Attention (CGA) project the semantic relations between auxiliary captions and query sentences into temporal space and fuse them into visual representations. Considering the gap between auxiliary captions and ground truth, we propose Asymmetric Cross-modal Contrastive Learning  (ACCL) for constructing more negative pairs to maximize cross-modal mutual information. Extensive experiments on three public datasets (i.e., ActivityNet Captions, TACoS and ActivityNet-CG) demonstrate that our method significantly outperforms state-of-the-art methods.

----

## [2064] VLN-Video: Utilizing Driving Videos for Outdoor Vision-and-Language Navigation

**Authors**: *Jialu Li, Aishwarya Padmakumar, Gaurav S. Sukhatme, Mohit Bansal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29813](https://doi.org/10.1609/aaai.v38i17.29813)

**Abstract**:

Outdoor Vision-and-Language Navigation (VLN) requires an agent to navigate through realistic 3D outdoor environments based on natural language instructions. The performance of existing VLN methods is limited by insufficient diversity in navigation environments and limited training data. To address these issues, we propose VLN-Video, which utilizes the diverse outdoor environments present in driving videos in multiple cities in the U.S. augmented with automatically generated navigation instructions and actions to improve outdoor VLN performance. VLN-Video combines the best of intuitive classical approaches and modern deep learning techniques, using template infilling to generate grounded non-repetitive navigation instructions, combined with an image rotation similarity based navigation action predictor to obtain VLN style data from driving videos for pretraining deep learning VLN models. We pre-train the model on the Touchdown dataset and our video-augmented dataset created from driving videos with three proxy tasks: Masked Language Modeling, Instruction and Trajectory Matching, and Next Action Prediction, so as to learn temporally-aware and visually-aligned instruction representations. The learned instruction representation is adapted to the state-of-the-art navigation agent when fine-tuning on the Touchdown dataset. Empirical results demonstrate that VLN-Video significantly outperforms previous state-of-the-art models by 2.1% in task completion rate, achieving a new state-of-the-art on the Touchdown dataset.

----

## [2065] Enhancing Multi-Label Classification via Dynamic Label-Order Learning

**Authors**: *Jiangnan Li, Yice Zhang, Shiwei Chen, Ruifeng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29814](https://doi.org/10.1609/aaai.v38i17.29814)

**Abstract**:

Generative methods tackle Multi-Label Classification (MLC) by autoregressively generating label sequences. These methods excel at modeling label correlations and have achieved outstanding performance. However, a key challenge is determining the order of labels, as empirical findings indicate the significant impact of different orders on model learning and inference. Previous works adopt static label-ordering methods, assigning a unified label order for all samples based on label frequencies or co-occurrences. Nonetheless, such static methods neglect the unique semantics of each sample. More critically, these methods can cause the model to rigidly memorize training order, resulting in missing labels during inference. In light of these limitations, this paper proposes a dynamic label-order learning approach that adaptively learns a label order for each sample. Specifically, our approach adopts a difficulty-prioritized principle and iteratively constructs the label sequence based on the sample s semantics. To reduce the additional cost incurred by label-order learning, we use the same SEQ2SEQ model for label-order learning and MLC learning and introduce a unified loss function for joint optimization. Extensive experiments on public datasets reveal that our approach greatly outperforms previous methods. We will release our code at https: //github.com/KagamiBaka/DLOL.

----

## [2066] Norm Tweaking: High-Performance Low-Bit Quantization of Large Language Models

**Authors**: *Liang Li, Qingyuan Li, Bo Zhang, Xiangxiang Chu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29815](https://doi.org/10.1609/aaai.v38i17.29815)

**Abstract**:

As the size of large language models (LLMs) continues to grow, model compression without sacrificing accuracy has become a crucial challenge for deployment.  While some quantization methods, such as GPTQ, have made progress in achieving acceptable 4-bit weight-only quantization, attempts at lower-bit quantization often result in severe performance degradation. In this paper, we introduce a technique called norm tweaking, which can be used as a plugin in current PTQ methods to achieve high precision while being cost-efficient. Our approach is inspired by the observation that rectifying the quantized activation distribution to match its float counterpart can readily restore accuracy for LLMs.  To achieve this, we carefully design a tweaking strategy that includes calibration data generation and channel-wise distance constraint to update the weights of normalization layers for better generalization. We conduct extensive experiments on various datasets using several open-sourced LLMs. Our method demonstrates significant improvements in both weight-only quantization and joint quantization of weights and activations, surpassing existing PTQ methods. On GLM-130B and OPT-66B, our method even achieves the same level of accuracy at 2-bit quantization as their float ones. Our simple and effective approach makes it more practical for real-world applications.

----

## [2067] Object Attribute Matters in Visual Question Answering

**Authors**: *Peize Li, Qingyi Si, Peng Fu, Zheng Lin, Yan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29816](https://doi.org/10.1609/aaai.v38i17.29816)

**Abstract**:

Visual question answering is a multimodal task that requires the joint comprehension of visual and textual information. However, integrating visual and textual semantics solely through attention layers is insufficient to comprehensively understand and align information from both modalities.  Intuitively, object attributes can naturally serve as a bridge to unify them,  which has been overlooked in previous research. In this paper, we propose a novel VQA approach from the perspective of utilizing object attribute, aiming to achieve better object-level visual-language alignment and multimodal scene understanding. Specifically, we design an attribute fusion module and a contrastive knowledge distillation module. The attribute fusion module constructs a multimodal graph neural network to fuse attributes and visual features through message passing. The enhanced object-level visual features contribute to solving fine-grained problem like counting-question. The better object-level visual-language alignment aids in understanding multimodal scenes, thereby improving the model's robustness. Furthermore, to augment scene understanding and the out-of-distribution performance, the contrastive knowledge distillation module introduces a series of implicit knowledge. We distill knowledge into attributes through contrastive loss, which further strengthens the representation learning of attribute features and facilitates visual-linguistic alignment. Intensive experiments on six datasets, COCO-QA, VQAv2, VQA-CPv2, VQA-CPv1, VQAvs and TDIUC, show the superiority of the proposed method.

----

## [2068] Translate Meanings, Not Just Words: IdiomKB's Role in Optimizing Idiomatic Translation with Language Models

**Authors**: *Shuang Li, Jiangjie Chen, Siyu Yuan, Xinyi Wu, Hao Yang, Shimin Tao, Yanghua Xiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29817](https://doi.org/10.1609/aaai.v38i17.29817)

**Abstract**:

To translate well, machine translation (MT) systems and general-purposed language models (LMs) need a deep understanding of both source and target languages and cultures. Therefore, idioms, with their non-compositional nature, pose particular challenges for Transformer-based systems, as literal translations often miss the intended meaning. Traditional methods, which replace idioms using existing knowledge bases (KBs), often lack scale and context-awareness. Addressing these challenges, our approach prioritizes context-awareness and scalability, allowing for offline storage of idioms in a manageable KB size. This ensures efficient serving with smaller models and provides a more comprehensive understanding of idiomatic expressions. We introduce a multilingual idiom KB (IdiomKB) developed using large LMs to address this. This KB facilitates better translation by smaller models, such as BLOOMZ (7.1B), Alpaca (7B), and InstructGPT (6.7B), by retrieving idioms' figurative meanings. We present a novel, GPT-4-powered metric for human-aligned evaluation, demonstrating that IdiomKB considerably boosts model performance. Human evaluations further validate our KB's quality.

----

## [2069] PMET: Precise Model Editing in a Transformer

**Authors**: *Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, Jie Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29818](https://doi.org/10.1609/aaai.v38i17.29818)

**Abstract**:

Model editing techniques modify a minor proportion of knowledge in Large Language Models (LLMs) at a relatively low cost, which have demonstrated notable success. Existing methods assume Transformer Layer (TL) hidden states are values of key-value memories of the Feed-Forward Network (FFN). They usually optimize the TL hidden states to memorize target knowledge and use it to update the weights of the FFN in LLMs. However, the information flow of TL hidden states comes from three parts: Multi-Head Self-Attention (MHSA), FFN, and residual connections. Existing methods neglect the fact that the TL hidden states contains information not specifically required for FFN. Consequently, the performance of model editing decreases. To achieve more precise model editing, we analyze hidden states of MHSA and FFN, finding that MHSA encodes certain general knowledge extraction patterns. This implies that MHSA weights do not require updating when new knowledge is introduced. Based on above findings, we introduce PMET, which simultaneously optimizes Transformer Component (TC, namely MHSA and FFN) hidden states, while only using the optimized TC hidden states of FFN to precisely update FFN weights. Our experiments demonstrate that PMET exhibits state-of-the-art performance on both the \textsc{counterfact} and zsRE datasets. Our ablation experiments substantiate the effectiveness of our enhancements, further reinforcing the finding that the MHSA encodes certain general knowledge extraction patterns and indicating its storage of a small amount of factual knowledge. Our code is available at \url{https://github.com/xpq-tech/PMET}.

----

## [2070] Dialogues Are Not Just Text: Modeling Cognition for Dialogue Coherence Evaluation

**Authors**: *Xue Li, Jia Su, Yang Yang, Zipeng Gao, Xinyu Duan, Yi Guan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29819](https://doi.org/10.1609/aaai.v38i17.29819)

**Abstract**:

The generation of logically coherent dialogues by humans relies on underlying cognitive abilities. Based on this, we redefine the dialogue coherence evaluation process, combining cognitive judgment with the basic text to achieve a more human-like evaluation. We propose a novel dialogue evaluation framework based on Dialogue Cognition Graph (DCGEval) to implement the fusion by in-depth interaction between cognition modeling and text modeling. The proposed Abstract Meaning Representation (AMR) based graph structure called DCG aims to uniformly model four dialogue cognitive abilities. Specifically, core-semantic cognition is modeled by converting the utterance into an AMR graph, which can extract essential semantic information without redundancy. The temporal and role cognition are modeled by establishing logical relationships among the different AMR graphs. Finally, the commonsense knowledge from ConceptNet is fused to express commonsense cognition. Experiments demonstrate the necessity of modeling human cognition for
dialogue evaluation, and our DCGEval presents stronger correlations with human judgments compared to other state-of-the-art evaluation metrics.

----

## [2071] EcomGPT: Instruction-Tuning Large Language Models with Chain-of-Task Tasks for E-commerce

**Authors**: *Yangning Li, Shirong Ma, Xiaobin Wang, Shen Huang, Chengyue Jiang, Haitao Zheng, Pengjun Xie, Fei Huang, Yong Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29820](https://doi.org/10.1609/aaai.v38i17.29820)

**Abstract**:

Recently, instruction-following Large Language Models (LLMs) , represented by ChatGPT, have exhibited exceptional performance in general Natural Language Processing (NLP) tasks. However, the unique characteristics of E-commerce data pose significant challenges to general LLMs. An LLM tailored specifically for E-commerce scenarios, possessing robust cross-dataset/task generalization capabilities, is a pressing necessity. To solve this issue, in this work, we proposed the first E-commerce instruction dataset EcomInstruct, with a total of 2.5 million instruction data. EcomInstruct scales up the data size and task diversity by constructing atomic tasks with E-commerce basic data types, such as product information, user reviews. Atomic tasks are defined as intermediate tasks implicitly involved in solving a final task, which we also call Chain-of-Task tasks. We developed EcomGPT
with different parameter scales by training the backbone model BLOOMZ with the EcomInstruct. Benefiting from the fundamental semantic understanding capabilities acquired from the Chain-of-Task tasks, EcomGPT exhibits excellent zero-shot generalization capabilities. Extensive experiments and human evaluations demonstrate that EcomGPT outperforms ChatGPT in term of cross-dataset/task generalization on E-commerce tasks. The EcomGPT will be public at https://github.com/Alibaba-NLP/EcomGPT.

----

## [2072] Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data

**Authors**: *Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Bin Sun, Xinglin Wang, Heda Wang, Kan Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29821](https://doi.org/10.1609/aaai.v38i17.29821)

**Abstract**:

Large Language Models (LLMs) have performed well on various reasoning tasks, but their inaccessibility and numerous parameters hinder wide application in practice. One promising way is distilling the reasoning ability from LLMs to small models by the generated chain-of-thought reasoning paths. In some cases, however, LLMs may produce incorrect reasoning chains, especially when facing complex mathematical problems. Previous studies only transfer knowledge from positive samples and drop the synthesized data with wrong answers. In this work, we illustrate the merit of negative data and propose a model specialization framework to distill LLMs with negative samples besides positive ones. The framework consists of three progressive steps, covering from training to inference stages, to absorb knowledge from negative data. We conduct extensive experiments across arithmetic reasoning tasks to demonstrate the role of negative data in distillation from LLM.

----

## [2073] LatestEval: Addressing Data Contamination in Language Model Evaluation through Dynamic and Time-Sensitive Test Construction

**Authors**: *Yucheng Li, Frank Guerin, Chenghua Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29822](https://doi.org/10.1609/aaai.v38i17.29822)

**Abstract**:

Data contamination in evaluation is getting increasingly prevalent with the emergence of language models pre-trained on super large, automatically crawled corpora. This problem leads to significant challenges in the accurate assessment of model capabilities and generalisations. In this paper, we propose LatestEval, an automatic method that leverages the most recent texts to create uncontaminated reading comprehension evaluations. LatestEval avoids data contamination by only using texts published within a recent time window, ensuring no overlap with the training corpora of pre-trained language models. We develop the LatestEval automated pipeline to 1) gather the latest texts; 2) identify key information, and 3) construct questions targeting the information while removing the existing answers from the context. This encourages models to infer the answers themselves based on the remaining context, rather than just copy-paste. Our experiments demonstrate that language models exhibit negligible memorisation behaviours on LatestEval as opposed to previous benchmarks, suggesting a significantly reduced risk of data contamination and leading to a more robust evaluation. Data and code are publicly available at: https://github.com/liyucheng09/LatestEval.

----

## [2074] FlexKBQA: A Flexible LLM-Powered Framework for Few-Shot Knowledge Base Question Answering

**Authors**: *Zhenyu Li, Sunqi Fan, Yu Gu, Xiuxing Li, Zhichao Duan, Bowen Dong, Ning Liu, Jianyong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29823](https://doi.org/10.1609/aaai.v38i17.29823)

**Abstract**:

Knowledge base question answering (KBQA) is a critical yet challenging task due to the vast number of entities within knowledge bases and the diversity of natural language questions posed by users. Unfortunately, the performance of most KBQA models tends to decline significantly in real-world scenarios where high-quality annotated data is insufficient. To mitigate the burden associated with manual annotation, we introduce FlexKBQA by utilizing Large Language Models (LLMs) as program translators for addressing the challenges inherent in the few-shot KBQA task. Specifically, FlexKBQA leverages automated algorithms to sample diverse programs, such as SPARQL queries, from the knowledge base, which are subsequently converted into natural language questions via LLMs. This synthetic dataset facilitates training a specialized lightweight model for the KB. Additionally, to reduce the barriers of distribution shift between synthetic data and real user questions, FlexKBQA introduces an executionguided self-training method to iterative leverage unlabeled user questions. Furthermore, we explore harnessing the inherent reasoning capability of LLMs to enhance the entire framework. Consequently, FlexKBQA delivers substantial flexibility, encompassing data annotation, deployment, and being domain agnostic. Through extensive experiments on GrailQA, WebQSP, and KQA Pro, we observe that under the few-shot even the more challenging zero-shot scenarios, FlexKBQA achieves impressive results with a few annotations, surpassing all previous baselines and even approaching the performance of supervised models, achieving a remarkable 93% performance relative to the fully-supervised models. We posit that FlexKBQA represents a significant advancement towards exploring better integration of large and lightweight models. Code is available at https://github.com/leezythu/FlexKBQA.

----

## [2075] Machine-Created Universal Language for Cross-Lingual Transfer

**Authors**: *Yaobo Liang, Quanzhi Zhu, Junhe Zhao, Nan Duan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29824](https://doi.org/10.1609/aaai.v38i17.29824)

**Abstract**:

There are two primary approaches to addressing cross-lingual transfer: multilingual pre-training, which implicitly aligns the hidden representations of various languages, and translate-test, which explicitly translates different languages into an intermediate language, such as English. Translate-test offers better interpretability compared to multilingual pre-training. However, it has lower performance than multilingual pre-training and struggles with word-level tasks due to translation altering word order. As a result, we propose a new Machine-created Universal Language (MUL) as an alternative intermediate language. MUL comprises a set of discrete symbols forming a universal vocabulary and a natural language to MUL translator for converting multiple natural languages to MUL. MUL unifies shared concepts from various languages into a single universal word, enhancing cross-language transfer. Additionally, MUL retains language-specific words and word order, allowing the model to be easily applied to word-level tasks. Our experiments demonstrate that translating into MUL yields improved performance compared to multilingual pre-training, and our analysis indicates that MUL possesses strong interpretability. The code is at: https://github.com/microsoft/Unicoder/tree/master/MCUL.

----

## [2076] CFEVER: A Chinese Fact Extraction and VERification Dataset

**Authors**: *Ying-Jia Lin, Chun-Yi Lin, Chia-Jen Yeh, Yi-Ting Li, Yun-Yu Hu, Chih-Hao Hsu, Mei-Feng Lee, Hung-Yu Kao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29825](https://doi.org/10.1609/aaai.v38i17.29825)

**Abstract**:

We present CFEVER, a Chinese dataset designed for Fact Extraction and VERification. CFEVER comprises 30,012 manually created claims based on content in Chinese Wikipedia. Each claim in CFEVER is labeled as “Supports”, “Refutes”, or “Not Enough Info” to depict its degree of factualness. Similar to the FEVER dataset, claims in the “Supports” and “Refutes” categories are also annotated with corresponding evidence sentences sourced from single or multiple pages in Chinese Wikipedia. Our labeled dataset holds a Fleiss’ kappa value of 0.7934 for five-way inter-annotator agreement. In addition, through the experiments with the state-of-the-art approaches developed on the FEVER dataset and a simple baseline for CFEVER, we demonstrate that our dataset is a new rigorous benchmark for factual extraction and verification, which can be further used for developing automated systems to alleviate human fact-checking efforts. CFEVER is available at https://ikmlab.github.io/CFEVER.

----

## [2077] Bootstrapping Large Language Models for Radiology Report Generation

**Authors**: *Chang Liu, Yuanhe Tian, Weidong Chen, Yan Song, Yongdong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29826](https://doi.org/10.1609/aaai.v38i17.29826)

**Abstract**:

Radiology report generation (RRG) aims to automatically generate a free-text description from a specific clinical radiograph, e.g., chest X-Ray images. Existing approaches tend to perform RRG with specific models trained on the public yet limited data from scratch, where they often lead to inferior performance owing to the problem of inefficient capabilities in both aligning visual and textual features and generating informative reports accordingly. Currently, large language models (LLMs) offered a promising solution to text generation with their power in learning from big data, especially for cross-modal scenarios such as RRG. However, most existing LLMs are pre-trained on general data, and suffer from the same problem of conventional approaches caused by knowledge gap between general and medical domain if they are applied to RRG. Therefore in this paper, we propose an approach to bootstrapping LLMs for RRG with a in-domain instance induction and a coarse-to-fine decoding process. Specifically, the in-domain instance induction process learns to align the LLM to radiology reports from general texts through contrastive learning. The coarse-to-fine decoding performs a text elevating process for those reports from the ranker, further enhanced with visual features and refinement prompts. Experimental results on two prevailing RRG datasets, namely, IU X-Ray and MIMIC-CXR, demonstrate the superiority of our approach to previous state-of-the-art solutions. Further analyses illustrate that, for the LLM, the induction process enables it to better align with the medical domain and the coarse-to-fine generation allows it to conduct more precise text generation.

----

## [2078] Liberating Seen Classes: Boosting Few-Shot and Zero-Shot Text Classification via Anchor Generation and Classification Reframing

**Authors**: *Han Liu, Siyang Zhao, Xiaotong Zhang, Feng Zhang, Wei Wang, Fenglong Ma, Hongyang Chen, Hong Yu, Xianchao Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29827](https://doi.org/10.1609/aaai.v38i17.29827)

**Abstract**:

Few-shot and zero-shot text classification aim to recognize samples from novel classes with limited labeled samples or no labeled samples at all. While prevailing methods have shown promising performance via transferring knowledge from seen classes to unseen classes, they are still limited by (1) Inherent dissimilarities among classes make the transformation of features learned from seen classes to unseen classes both difficult and inefficient. (2) Rare labeled novel samples usually cannot provide enough supervision signals to enable the model to adjust from the source distribution to the target distribution, especially for complicated scenarios. To alleviate the above issues, we propose a simple and effective strategy for few-shot and zero-shot text classification. We aim to liberate the model from the confines of seen classes, thereby enabling it to predict unseen categories without the necessity of training on seen classes. Specifically, for mining more related unseen category knowledge, we utilize a large pre-trained language model to generate pseudo novel samples, and select the most representative ones as category anchors. After that, we convert the multi-class classification task into a binary classification task and use the similarities of query-anchor pairs for prediction to fully leverage the limited supervision signals. Extensive experiments on six widely used public datasets show that our proposed method can outperform other strong baselines significantly in few-shot and zero-shot tasks, even without using any seen class samples.

----

## [2079] Beyond Entities: A Large-Scale Multi-Modal Knowledge Graph with Triplet Fact Grounding

**Authors**: *Jingping Liu, Mingchuan Zhang, Weichen Li, Chao Wang, Shuang Li, Haiyun Jiang, Sihang Jiang, Yanghua Xiao, Yunwen Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29828](https://doi.org/10.1609/aaai.v38i17.29828)

**Abstract**:

Much effort has been devoted to building multi-modal knowledge graphs by visualizing entities on images, but ignoring the multi-modal information of the relation between entities. Hence, in this paper, we aim to construct a new large-scale multi-modal knowledge graph with triplet facts grounded on images that reflect not only entities but also their relations. To achieve this purpose, we propose a novel pipeline method, including triplet fact filtering, image retrieving, entity-based image filtering, relation-based image filtering, and image clustering. In this way, a multi-modal knowledge graph named ImgFact is constructed, which contains 247,732 triplet facts and 3,730,805 images. In experiments, the manual and automatic evaluations prove the reliable quality of our ImgFact. We further use the obtained images to enhance model performance on two tasks. In particular, the model optimized by our ImgFact achieves an impressive 8.38% and 9.87% improvement over the solutions enhanced by an existing multi-modal knowledge graph and VisualChatGPT on F1 of relation classification. We release ImgFact and its instructions at https://github.com/kleinercubs/ImgFact.

----

## [2080] Chinese Spelling Correction as Rephrasing Language Model

**Authors**: *Linfeng Liu, Hongqiu Wu, Hai Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29829](https://doi.org/10.1609/aaai.v38i17.29829)

**Abstract**:

This paper studies Chinese Spelling Correction (CSC), which aims to detect and correct potential spelling errors in a given sentence. Current state-of-the-art methods regard CSC as a sequence tagging task and fine-tune BERT-based models on sentence pairs. However, we note a critical flaw in the process of tagging one character to another, that the correction is excessively conditioned on the error. This is opposite from human mindset, where individuals rephrase the complete sentence based on its semantics, rather than solely on the error patterns memorized before. Such a counter-intuitive learning process results in the bottleneck of generalizability and transferability of machine spelling correction. To address this, we propose Rephrasing Language Modeling (ReLM), where the model is trained to rephrase the entire sentence by infilling additional slots, instead of character-to-character tagging. This novel training paradigm achieves the new state-of-theart results across fine-tuned and zero-shot CSC benchmarks, outperforming previous counterparts by a large margin. Our method also learns transferable language representation when CSC is jointly trained with other tasks.

----

## [2081] TA&AT: Enhancing Task-Oriented Dialog with Turn-Level Auxiliary Tasks and Action-Tree Based Scheduled Sampling

**Authors**: *Longxiang Liu, Xiuxing Li, Yang Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29830](https://doi.org/10.1609/aaai.v38i17.29830)

**Abstract**:

Task-oriented dialog systems have witnessed substantial progress due to conversational pre-training techniques. Yet, two significant challenges persist. First, most systems primarily utilize the latest turn's state label for the generator. This practice overlooks the comprehensive value of state labels in boosting the model's understanding for future generations. Second, an overreliance on generated policy often leads to error accumulation, resulting in suboptimal responses when adhering to incorrect actions. To combat these challenges, we propose turn-level multi-task objectives for the encoder. With the guidance of essential information from labeled intermediate states, we establish a more robust representation for both understanding and generation. For the decoder, we introduce an action tree-based scheduled sampling technique. Specifically, we model the hierarchical policy as trees and utilize the similarity between trees to sample negative policy based on scheduled sampling, hoping the model to generate invariant responses under perturbations. This method simulates potential pitfalls by sampling similar negative policy, bridging the gap between task-oriented dialog training and inference. Among methods without continual pre-training, our approach achieved state-of-the-art (SOTA) performance on the MultiWOZ dataset series and was also competitive with pre-trained SOTA methods.

----

## [2082] Hierarchical Aligned Multimodal Learning for NER on Tweet Posts

**Authors**: *Peipei Liu, Hong Li, Yimo Ren, Jie Liu, Shuaizong Si, Hongsong Zhu, Limin Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29831](https://doi.org/10.1609/aaai.v38i17.29831)

**Abstract**:

Mining structured knowledge from tweets using named entity recognition (NER) can be beneficial for many downstream applications such as recommendation and intention under standing. With tweet posts tending to be multimodal, multimodal named entity recognition (MNER) has attracted more attention. In this paper, we propose a novel approach, which can dynamically align the image and text sequence and achieve the multi-level cross-modal learning to augment textual word representation for MNER improvement. To be specific, our framework can be split into three main stages: the first stage focuses on intra-modality representation learning to derive the implicit global and local knowledge of each modality, the second evaluates the relevance between the text and its accompanying image and integrates different grained visual information based on the relevance, the third enforces semantic refinement via iterative cross-modal interactions and co-attention. We conduct experiments on two open datasets, and the results and detailed analysis demonstrate the advantage of our model.

----

## [2083] Adaptive Prompt Routing for Arbitrary Text Style Transfer with Pre-trained Language Models

**Authors**: *Qingyi Liu, Jinghui Qin, Wenxuan Ye, Hao Mou, Yuxuan He, Keze Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29832](https://doi.org/10.1609/aaai.v38i17.29832)

**Abstract**:

Recently, arbitrary text style transfer (TST) has made significant progress with the paradigm of prompt learning. In this paradigm, researchers often design or search for a fixed prompt for any input. However, existing evidence shows that large language models (LLMs) are prompt-sensitive and it is sub-optimal to apply the same prompt to any input for downstream TST tasks. Besides, the prompts obtained by searching are often unreadable and unexplainable to humans. To address these issues, we propose an Adaptive Prompt Routing (APR) framework to adaptively route prompts from a human-readable prompt set for various input texts and given styles. Specifically, we first construct a candidate prompt set of diverse and human-readable prompts for the target style. This set consists of several seed prompts and their variants paraphrased by an LLM. Subsequently, we train a prompt routing model to select the optimal prompts efficiently according to inputs. The adaptively selected prompt can guide the LLMs to perform a precise style transfer for each input sentence while maintaining readability for humans. Extensive experiments on 4 public TST benchmarks over 3 popular LLMs (with parameter sizes ranging from 1.5B to 175B) demonstrate that our APR achieves superior style transfer performances, compared to the state-of-the-art prompt-based and fine-tuning methods. The source code is available at https://github.com/DwyaneLQY/APR

----

## [2084] Emotion Rendering for Conversational Speech Synthesis with Heterogeneous Graph-Based Context Modeling

**Authors**: *Rui Liu, Yifan Hu, Yi Ren, Xiang Yin, Haizhou Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29833](https://doi.org/10.1609/aaai.v38i17.29833)

**Abstract**:

Conversational Speech Synthesis (CSS) aims to accurately express an utterance with the appropriate prosody and emotional inflection within a conversational setting. While recognising the significance of CSS task, the prior studies have not thoroughly investigated the emotional expressiveness problems due to the scarcity of emotional conversational datasets and the difficulty of stateful emotion modeling. In this paper, we propose a novel emotional CSS model, termed ECSS, that includes two main components: 1) to enhance emotion understanding, we introduce a heterogeneous graph-based emotional context modeling mechanism, which takes the multi-source dialogue history as input to model the dialogue context and learn the emotion cues from the context; 2) to achieve emotion rendering, we employ a contrastive learning-based emotion renderer module to infer the accurate emotion style for the target utterance. To address the issue of data scarcity, we meticulously create emotional labels in terms of category and intensity, and annotate additional emotional information on the existing conversational dataset (DailyTalk). Both objective and subjective evaluations suggest that our model outperforms the baseline models in understanding and rendering emotions. These evaluations also underscore the importance of comprehensive emotional annotations.  Code and audio samples can be found at: https://github.com/walker-hyf/ECSS.

----

## [2085] Robust Evaluation Measures for Evaluating Social Biases in Masked Language Models

**Authors**: *Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29834](https://doi.org/10.1609/aaai.v38i17.29834)

**Abstract**:

Many evaluation measures are used to evaluate social biases in masked language models (MLMs). However, we find that these previously proposed evaluation measures are lacking robustness in scenarios with limited datasets. This is because these measures are obtained by comparing the pseudo-log-likelihood (PLL) scores of the stereotypical and anti-stereotypical samples using an indicator function. The disadvantage is the limited mining of the PLL score sets without capturing its distributional information. In this paper, we represent a PLL score set as a Gaussian distribution and use Kullback-Leibler (KL) divergence and Jensen–Shannon (JS) divergence to construct evaluation measures for the distributions of stereotypical and anti-stereotypical PLL scores. Experimental results on the publicly available datasets StereoSet (SS) and CrowS-Pairs (CP) show that our proposed measures are significantly more robust and interpretable than those proposed previously.

----

## [2086] Improved Graph Contrastive Learning for Short Text Classification

**Authors**: *Yonghao Liu, Lan Huang, Fausto Giunchiglia, Xiaoyue Feng, Renchu Guan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29835](https://doi.org/10.1609/aaai.v38i17.29835)

**Abstract**:

Text classification occupies an important role in natural language processing and has many applications in real life. Short text classification, as one of its subtopics, has attracted increasing interest from researchers since it is more challenging due to its semantic sparsity and insufficient labeled data. Recent studies attempt to combine graph learning and contrastive learning to alleviate the above problems in short text classification. Despite their fruitful success, there are still several inherent limitations. First, the generation of augmented views may disrupt the semantic structure within the text and introduce negative effects due to noise permutation. Second, they ignore the clustering-friendly features in unlabeled data and fail to further utilize the prior information in few valuable labeled data. To this end, we propose a novel model that utilizes improved Graph contrastIve learning for short text classiFicaTion (GIFT). Specifically, we construct a heterogeneous graph containing several component graphs by mining from an internal corpus and introducing an external knowledge graph. Then, we use singular value decomposition to generate augmented views for graph contrastive learning. Moreover, we employ constrained kmeans on labeled texts to learn clustering-friendly features, which facilitate cluster-oriented contrastive learning and assist in obtaining better category boundaries. Extensive experimental results show that GIFT significantly outperforms previous state-of-the-art methods. Our code can be found in
https://github.com/KEAML-JLU/GIFT.

----

## [2087] QuerySum: A Multi-Document Query-Focused Summarization Dataset Augmented with Similar Query Clusters

**Authors**: *Yushan Liu, Zili Wang, Ruifeng Yuan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29836](https://doi.org/10.1609/aaai.v38i17.29836)

**Abstract**:

Query-focused summarization (QFS) aims to summarize the source document(s) with regard to a specific aspect of information given in a query. It plays an important role in presenting users with a concise answer summary from a set of query-relevant documents retrieved by the information retrieval system. Nonetheless, the QFS research has long been hampered by the lack of adequate datasets in terms of both quality and quantity. In this paper, we introduce a large-scale multi-document query-focused summarization dataset, called QuerySum, which contains 27,041 data samples covering diverse topics and its quality is guaranteed through human verification. Unlike some previous QFS datasets constructed directly from the question answering datasets, 74% queries in our dataset are the challenging non-factoid What-, Why-, and How- questions. More importantly, we also provide a set of similar queries together with the corresponding summaries pairs for each query as the retrieved context, presenting a new feature of QuerySum. We aim to encourage research efforts in query intention understanding in the context of QFS. Leveraging QuerySum's depth, we propose a model for query-aware multi-document summarization and set a new QFS benchmark.

----

## [2088] Generative Multi-Modal Knowledge Retrieval with Large Language Models

**Authors**: *Xinwei Long, Jiali Zeng, Fandong Meng, Zhiyuan Ma, Kaiyan Zhang, Bowen Zhou, Jie Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29837](https://doi.org/10.1609/aaai.v38i17.29837)

**Abstract**:

Knowledge retrieval with multi-modal queries plays a crucial role in supporting knowledge-intensive multi-modal applications. However, existing methods face challenges in terms of their effectiveness and training efficiency, especially when it comes to training and integrating multiple retrievers to handle multi-modal queries. In this paper, we propose an innovative end-to-end generative framework for multi-modal knowledge retrieval. Our framework takes advantage of the fact that large language models (LLMs) can effectively serve as virtual knowledge bases, even when trained with limited data. We retrieve knowledge via a two-step process: 1) generating knowledge clues related to the queries, and 2) obtaining the relevant document by searching databases using the knowledge clue. In particular, we first introduce an object-aware prefix-tuning technique to guide multi-grained visual learning. Then, we align multi-grained visual features into the textual feature space of the LLM, employing the LLM to capture cross-modal interactions. Subsequently, we construct instruction data with a unified format for model training. Finally, we propose the knowledge-guided generation strategy to impose prior constraints in the decoding steps, thereby promoting the generation of distinctive knowledge clues. Through experiments conducted on three benchmarks, we demonstrate significant improvements ranging from 3.0% to 14.6% across all evaluation metrics when compared to strong baselines.

----

## [2089] Synergistic Anchored Contrastive Pre-training for Few-Shot Relation Extraction

**Authors**: *Da Luo, Yanglei Gan, Rui Hou, Run Lin, Qiao Liu, Yuxiang Cai, Wannian Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29838](https://doi.org/10.1609/aaai.v38i17.29838)

**Abstract**:

Few-shot Relation Extraction (FSRE) aims to extract relational facts from a sparse set of labeled corpora. Recent studies have shown promising results in FSRE by employing Pre-trained Language Models (PLMs) within the framework of supervised contrastive learning, which considers both instances and label facts. However, how to effectively harness massive instance-label pairs to encompass the learned representation with semantic richness in this learning paradigm is not fully explored. To address this gap, we introduce a novel synergistic anchored contrastive pre-training framework. This framework is motivated by the insight that the diverse viewpoints conveyed through instance-label pairs capture incomplete yet complementary intrinsic textual semantics. Specifically, our framework involves a symmetrical contrastive objective that encompasses both sentence-anchored and label-anchored contrastive losses. By combining these two losses, the model establishes a robust and uniform representation space. This space effectively captures the reciprocal alignment of feature distributions among instances and relational facts, simultaneously enhancing the maximization of mutual information across diverse perspectives within the same relation. Experimental results demonstrate that our framework achieves significant performance enhancements compared to baseline models in downstream FSRE tasks. Furthermore, our approach exhibits superior adaptability to handle the challenges of domain shift and zero-shot relation extraction. Our code is available online at https://github.com/AONE-NLP/FSRE-SaCon.

----

## [2090] STAR: Boosting Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models

**Authors**: *Mingyu Derek Ma, Xiaoxuan Wang, Po-Nien Kung, P. Jeffrey Brantingham, Nanyun Peng, Wei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29839](https://doi.org/10.1609/aaai.v38i17.29839)

**Abstract**:

Information extraction tasks such as event extraction require an in-depth understanding of the output structure and sub-task dependencies. They heavily rely on task-specific training data in the form of (passage, target structure) pairs to obtain reasonable performance. However, obtaining such data through human annotation is costly, leading to a pressing need for low-resource information extraction approaches that require minimal human labeling for real-world applications. Fine-tuning supervised models with synthesized training data would be a generalizable method, but the existing data generation methods either still rely on large-scale ground-truth data or cannot be applied to complicated IE tasks due to their poor performance. To address these challenges, we propose STAR, a data generation method that leverages Large Language Models (LLMs) to synthesize data instances given limited seed demonstrations, thereby boosting low-resource information extraction performance. Our approach involves generating target structures (Y) followed by generating passages (X), all accomplished with the aid of LLMs. We design fine-grained step-by-step instructions to obtain the initial data instances. We further reduce errors and improve data quality through self-reflection error identification and self-refinement with iterative revision. Our experiments show that the data generated by STAR significantly improve the performance of low-resource event extraction and relation extraction tasks, even surpassing the effectiveness of human-curated data. Human assessment of the data quality shows STAR-generated data exhibit higher passage quality and better align with the task definitions compared with the human-curated data.

----

## [2091] Mastering Context-to-Label Representation Transformation for Event Causality Identification with Diffusion Models

**Authors**: *Hieu Man, Franck Dernoncourt, Thien Huu Nguyen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29840](https://doi.org/10.1609/aaai.v38i17.29840)

**Abstract**:

To understand event structures of documents, event causality identification (ECI) emerges as a crucial task, aiming to discern causal relationships among event mentions. The latest approach for ECI has introduced advanced deep learning models where transformer-based encoding models, complemented by enriching components, are typically leveraged to learn effective event context representations for causality prediction. As such, an important step for ECI models is to transform the event context representations into causal label representations to perform logits score computation for training and inference purposes. Within this framework, event context representations might encapsulate numerous complicated and noisy structures due to the potential long context between the input events while causal label representations are intended to capture pure information about the causal relations to facilitate score estimation. Nonetheless, a notable drawback of existing ECI models stems from their reliance on simple feed-forward networks to handle the complex context-to-label representation transformation process, which might require drastic changes in the representations to hinder the learning process. To overcome this issue, our work introduces a novel method for ECI where, instead abrupt transformations, event context representations are gradually updated to achieve effective label representations. This process will be done incrementally to allow filtering of irrelevant structures at varying levels of granularity for causal relations. To realize this, we present a diffusion model to learn gradual representation transition processes between context and causal labels. It operates through a forward pass for causal label representation noising and a reverse pass for reconstructing label representations from random noise.  Our experiments on different datasets across multiple languages demonstrate the advantages of the diffusion model with state-of-the-art performance for ECI.

----

## [2092] Span Graph Transformer for Document-Level Named Entity Recognition

**Authors**: *Hongli Mao, Xian-Ling Mao, Hanlin Tang, Yuming Shang, Heyan Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29841](https://doi.org/10.1609/aaai.v38i17.29841)

**Abstract**:

Named Entity Recognition (NER), which aims to identify the span and category of entities within  text, is a fundamental task in natural language processing.  Recent NER approaches have featured  pre-trained transformer-based models (e.g., BERT) as a crucial encoding component to achieve state-of-the-art performance. However, due to the length limit for input text, these models typically consider text at the sentence-level and cannot capture the long-range contextual dependency within a document. To address this issue, we propose a novel Span Graph Transformer (SGT) method for document-level NER, which constructs long-range contextual dependencies at both the token and span levels. Specifically,  we first retrieve relevant contextual sentences in the document for each target sentence, and jointly encode them by BERT to capture token-level dependencies.  Then,  our proposed model extracts candidate spans from each sentence and integrates these spans into a document-level span graph, where nested spans within sentences and identical spans across sentences are connected. By leveraging the power of Graph Transformer and well-designed position encoding, our span graph can fully exploit span-level dependencies within the document. Extensive experiments on both resource-rich nested and flat NER datasets, as well as low-resource distantly supervised NER datasets, demonstrate that proposed SGT model achieves better performance than previous state-of-the-art models.

----

## [2093] Underspecification in Language Modeling Tasks: A Causality-Informed Study of Gendered Pronoun Resolution

**Authors**: *Emily McMilin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29842](https://doi.org/10.1609/aaai.v38i17.29842)

**Abstract**:

Modern language modeling tasks are often underspecified: for a given token prediction, many words may satisfy the user’s intent of producing natural language at inference time, however only one word will minimize the task’s loss function at training time. We introduce a simple causal mechanism to describe the role underspecification plays in the generation of spurious correlations. Despite its simplicity, our causal model directly informs the development of two lightweight black-box evaluation methods, that we apply to gendered pronoun resolution tasks on a wide range of LLMs to 1) aid in the detection of inference-time task underspecification by exploiting 2) previously unreported gender vs. time and gender vs. location spurious correlations on LLMs with a range of A) sizes: from BERT-base to GPT-3.5, B) pre-training objectives: from masked & autoregressive language modeling to a mixture of these objectives, and C) training stages: from pre-training only to reinforcement learning from human feedback (RLHF). Code and open-source demos available at https://github.com/2dot71mily/uspec.

----

## [2094] MCL-NER: Cross-Lingual Named Entity Recognition via Multi-View Contrastive Learning

**Authors**: *Ying Mo, Jian Yang, Jiahao Liu, Qifan Wang, Ruoyu Chen, Jingang Wang, Zhoujun Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29843](https://doi.org/10.1609/aaai.v38i17.29843)

**Abstract**:

Cross-lingual named entity recognition (CrossNER) faces challenges stemming from uneven performance due to the scarcity of multilingual corpora, especially for non-English
data. While prior efforts mainly focus on data-driven transfer methods, a significant aspect that has not been fully explored is aligning both semantic and token-level representations across diverse languages. In this paper, we propose Multi-view Contrastive Learning for Cross-lingual Named
Entity Recognition (MCL-NER). Specifically, we reframe the CrossNER task into a problem of recognizing relationships between pairs of tokens. This approach taps into the
inherent contextual nuances of token-to-token connections within entities, allowing us to align representations across
different languages. A multi-view contrastive learning framework is introduced to encompass semantic contrasts between
source, codeswitched, and target sentences, as well as contrasts among token-to-token relations. By enforcing agreement within both semantic and relational spaces, we minimize the gap between source sentences and their counterparts of both codeswitched and target sentences. This alignment
extends to the relationships between diverse tokens, enhancing the projection of entities across languages. We further
augment CrossNER by combining self-training with labeled source data and unlabeled target data. Our experiments on
the XTREME benchmark, spanning 40 languages, demonstrate the superiority of MCL-NER over prior data-driven
and model-based approaches. It achieves a substantial increase of nearly +2.0 F1 scores across a broad spectrum and
establishes itself as the new state-of-the-art performer.

----

## [2095] KAM-CoT: Knowledge Augmented Multimodal Chain-of-Thoughts Reasoning

**Authors**: *Debjyoti Mondal, Suraj Modi, Subhadarshi Panda, Rituraj Singh, Godawari Sudhakar Rao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29844](https://doi.org/10.1609/aaai.v38i17.29844)

**Abstract**:

Large Language Models (LLMs) have demonstrated impressive performance in natural language processing tasks by leveraging chain of thought (CoT) that enables step-by-step thinking. Extending LLMs with multimodal capabilities is the recent interest, but incurs computational cost and requires substantial hardware resources. To address these challenges, we propose KAM-CoT a framework that integrates CoT reasoning, Knowledge Graphs (KGs), and multiple modalities for a comprehensive understanding of multimodal tasks. KAM-CoT adopts a two-stage training process with KG grounding to generate effective rationales and answers. By incorporating external knowledge from KGs during reasoning, the model gains a deeper contextual understanding reducing hallucinations and enhancing the quality of answers. This knowledge-augmented CoT reasoning empowers the model to handle questions requiring external context, providing more informed answers. Experimental findings show KAM-CoT outperforms the state-of-the-art methods. On the ScienceQA dataset, we achieve an average accuracy of 93.87%, surpassing GPT-3.5 (75.17%) by 18% and GPT-4 (83.99%) by 10%. Remarkably, KAM-CoT achieves these results with only 280M trainable parameters at a time, demonstrating its cost-efficiency and effectiveness.

----

## [2096] Accelerating the Global Aggregation of Local Explanations

**Authors**: *Alon Mor, Yonatan Belinkov, Benny Kimelfeld*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29845](https://doi.org/10.1609/aaai.v38i17.29845)

**Abstract**:

Local explanation methods highlight the input tokens that have a considerable impact on the outcome of classifying the document at hand. For example, the Anchor algorithm applies a statistical analysis of the sensitivity of the classifier to changes in the token. Aggregating local explanations over a dataset provides a global explanation of the model.
Such aggregation aims to detect words with the most impact, giving valuable insights about the model, like what it has learned in training and which adversarial examples expose its weaknesses.
However, standard aggregation methods bear a high computational cost:
a naive implementation applies a costly algorithm to each token of each document, and hence, it is infeasible for a simple user running in the scope of a short analysis session.  

We devise techniques for accelerating the global aggregation of the Anchor algorithm. Specifically, our goal is to compute a set of top-k words with the highest global impact according to different aggregation functions. Some of our techniques are lossless and some are lossy.
We show that for a very mild loss of quality, we are able to accelerate the computation by up to 30 times, reducing the computation from hours to minutes. We also devise and study a probabilistic model that accounts for noise in the Anchor algorithm and diminishes the bias toward words that are frequent yet low in impact.

----

## [2097] Self-Supervised Disentangled Representation Learning for Robust Target Speech Extraction

**Authors**: *Zhaoxi Mu, Xinyu Yang, Sining Sun, Qing Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29846](https://doi.org/10.1609/aaai.v38i17.29846)

**Abstract**:

Speech signals are inherently complex as they encompass both global acoustic characteristics and local semantic information. However, in the task of target speech extraction, certain elements of global and local semantic information in the reference speech, which are irrelevant to speaker identity, can lead to speaker confusion within the speech extraction network. To overcome this challenge, we propose a self-supervised disentangled representation learning method. Our approach tackles this issue through a two-phase process, utilizing a reference speech encoding network and a global information disentanglement network to gradually disentangle the speaker identity information from other irrelevant factors. We exclusively employ the disentangled speaker identity information to guide the speech extraction network. Moreover, we introduce the adaptive modulation Transformer to ensure that the acoustic representation of the mixed signal remains undisturbed by the speaker embeddings. This component incorporates speaker embeddings as conditional information, facilitating natural and efficient guidance for the speech extraction network. Experimental results substantiate the effectiveness of our meticulously crafted approach, showcasing a substantial reduction in the likelihood of speaker confusion.

----

## [2098] READ-PVLA: Recurrent Adapter with Partial Video-Language Alignment for Parameter-Efficient Transfer Learning in Low-Resource Video-Language Modeling

**Authors**: *Thong Nguyen, Xiaobao Wu, Xinshuai Dong, Khoi M. Le, Zhiyuan Hu, Cong-Duy Nguyen, See-Kiong Ng, Anh Tuan Luu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29847](https://doi.org/10.1609/aaai.v38i17.29847)

**Abstract**:

Fully fine-tuning pretrained large-scale transformer models has become a popular paradigm for video-language modeling tasks, such as temporal language grounding and video-language summarization. With a growing number of tasks and limited training data, such full fine-tuning approach leads to costly model storage and unstable training. To overcome these shortcomings, we introduce lightweight adapters to the pre-trained model and only update them at fine-tuning time. However, existing adapters fail to capture intrinsic temporal relations among video frames or textual words. Moreover, they neglect the preservation of critical task-related information that flows from the raw video-language input into the adapter’s low-dimensional space. To address these issues, we first propose a novel REcurrent ADapter (READ) that employs recurrent computation to enable temporal modeling capability. Second, we propose Partial Video-Language Alignment (PVLA) objective via the use of partial optimal transport to maintain task-related information flowing into our READ modules. We validate our READ-PVLA framework through extensive experiments where READ-PVLA significantly outperforms all existing fine-tuning strategies on multiple low-resource temporal language grounding and video-language summarization benchmarks.

----

## [2099] Code-Style In-Context Learning for Knowledge-Based Question Answering

**Authors**: *Zhijie Nie, Richong Zhang, Zhongyuan Wang, Xudong Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29848](https://doi.org/10.1609/aaai.v38i17.29848)

**Abstract**:

Current methods for Knowledge-Based Question Answering (KBQA) usually rely on complex training techniques and model frameworks, leading to many limitations in practical applications. Recently, the emergence of In-Context Learning (ICL) capabilities in Large Language Models (LLMs) provides a simple and training-free semantic parsing paradigm for KBQA: Given a small number of questions and their labeled logical forms as demo examples, LLMs can understand the task intent and generate the logic form for a new question. However, current powerful LLMs have little exposure to logic forms during pre-training, resulting in a high format error rate. To solve this problem, we propose a code-style in-context learning method for KBQA, which converts the generation process of unfamiliar logical form into the more familiar code generation process for LLMs. Experimental results on three mainstream datasets show that our method dramatically mitigated the formatting error problem in generating logic forms while realizing a new SOTA on WebQSP, GrailQA, and GraphQ under the few-shot setting. The code and supplementary files are released at https://github.com/Arthurizijar/KB-Coder.

----

## [2100] Aspect-Based Sentiment Analysis with Explicit Sentiment Augmentations

**Authors**: *Jihong Ouyang, Zhiyao Yang, Silong Liang, Bing Wang, Yimeng Wang, Ximing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29849](https://doi.org/10.1609/aaai.v38i17.29849)

**Abstract**:

Aspect-based sentiment analysis (ABSA), a fine-grained sentiment classification task, has received much attention recently. Many works investigate sentiment information through opinion words, such as "good'' and "bad''. However, implicit sentiment data widely exists in the ABSA dataset, whose sentiment polarity is hard to determine due to the lack of distinct opinion words. To deal with implicit sentiment, this paper proposes an ABSA method that integrates explicit sentiment augmentations (ABSA-ESA) to add more sentiment clues. We propose an ABSA-specific explicit sentiment generation method to create such augmentations. Specifically, we post-train T5 by rule-based data and employ three strategies to constrain the sentiment polarity and aspect term of the generated augmentations. We employ Syntax Distance Weighting and Unlikelihood Contrastive Regularization in the training procedure to guide the model to generate the explicit opinion words with the same polarity as the input sentence. Meanwhile, we utilize the Constrained Beam Search to ensure the augmentations are aspect-related. We test ABSA-ESA on two ABSA benchmarks. The results show that ABSA-ESA outperforms the SOTA baselines on implicit and explicit sentiment accuracy.

----

## [2101] Fact-Driven Logical Reasoning for Machine Reading Comprehension

**Authors**: *Siru Ouyang, Zhuosheng Zhang, Hai Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29850](https://doi.org/10.1609/aaai.v38i17.29850)

**Abstract**:

Recent years have witnessed an increasing interest in training machines with reasoning ability, which deeply relies on accurately and clearly presented clue forms. The clues are usually modeled as entity-aware knowledge in existing studies. However, those entity-aware clues are primarily focused on commonsense, making them insufficient for tasks that require knowledge of temporary facts or events, particularly in logical reasoning for reading comprehension. To address this challenge, we are motivated to cover both commonsense and temporary knowledge clues hierarchically. Specifically, we propose a general formalism of knowledge units by extracting backbone constituents of the sentence, such as the subject-verb-object formed ``facts''. We then construct a supergraph on top of the fact units, allowing for the benefit of sentence-level (relations among fact groups) and entity-level interactions (concepts or actions inside a fact). Experimental results on logical reasoning benchmarks and dialogue modeling datasets show that our approach improves the baselines substantially, and it is general across backbone models. Code is available at https://github.com/ozyyshr/FocalReasoner.

----

## [2102] Preparing Lessons for Progressive Training on Language Models

**Authors**: *Yu Pan, Ye Yuan, Yichun Yin, Jiaxin Shi, Zenglin Xu, Ming Zhang, Lifeng Shang, Xin Jiang, Qun Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29851](https://doi.org/10.1609/aaai.v38i17.29851)

**Abstract**:

The rapid progress of Transformers in artificial intelligence has come at the cost of increased resource consumption and greenhouse gas emissions due to growing model sizes. Prior work suggests using pretrained small models to improve training efficiency, but this approach may not be suitable for new model structures. On the other hand, training from scratch can be slow, and progressively stacking layers often fails to achieve significant acceleration. To address these challenges, we propose a novel method called Apollo, which prepares lessons for expanding operations by learning high-layer functionality during training of low layers. Our approach involves low-value-prioritized sampling (LVPS) to train different depths and weight sharing to facilitate efficient expansion. We also introduce an interpolation method for stable model depth extension. Experiments demonstrate that Apollo achieves state-of-the-art acceleration ratios, even rivaling methods using pretrained models, making it a universal and efficient solution for training deep models while reducing time, financial, and environmental costs.

----

## [2103] A Novel Energy Based Model Mechanism for Multi-Modal Aspect-Based Sentiment Analysis

**Authors**: *Tianshuo Peng, Zuchao Li, Ping Wang, Lefei Zhang, Hai Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29852](https://doi.org/10.1609/aaai.v38i17.29852)

**Abstract**:

Multi-modal aspect-based sentiment analysis (MABSA) has recently attracted increasing attention. The span-based extraction methods, such as FSUIE, demonstrate strong performance in sentiment analysis due to their joint modeling of input sequences and target labels. However, previous methods still have certain limitations: (i) They ignore the difference in the focus of visual information between different analysis targets (aspect or sentiment). (ii) Combining features from uni-modal encoders directly may not be sufficient to eliminate the modal gap and can cause difficulties in capturing the image-text pairwise relevance. (iii) Existing span-based methods for MABSA ignore the pairwise relevance of target span boundaries. To tackle these limitations, we propose a novel framework called DQPSA. Specifically, our model contains a Prompt as Dual Query (PDQ) module that uses the prompt as both a visual query and a language query to extract prompt-aware visual information and strengthen the pairwise relevance between visual information and the analysis target. Additionally, we introduce an Energy-based Pairwise Expert (EPE) module that models the boundaries pairing of the analysis target from the perspective of an Energy-based Model. This expert predicts aspect or sentiment span based on pairwise stability. Experiments on three widely used benchmarks demonstrate that DQPSA outperforms previous approaches and achieves a new state-of-the-art performance. The code will be released at https://github.com/pengts/DQPSA.

----

## [2104] A Joint Framework with Heterogeneous-Relation-Aware Graph and Multi-Channel Label Enhancing Strategy for Event Causality Extraction

**Authors**: *Ruili Pu, Yang Li, Jun Zhao, Suge Wang, Deyu Li, Jian Liao, Jianxing Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29853](https://doi.org/10.1609/aaai.v38i17.29853)

**Abstract**:

Event Causality Extraction (ECE) aims to extract the cause-effect event pairs with their structured event information from plain texts. As far as we know, the existing ECE methods mainly focus on the correlation between arguments, without explicitly modeling the causal relationship between events, and usually design two independent frameworks to extract cause events and effect events, respectively, which cannot effectively capture the dependency between the subtasks. Therefore, we propose a joint multi-label extraction framework for ECE to alleviate the above limitations. In particular, 1) we design a heterogeneous-relation-aware graph module to learn the potential relationships between events and arguments, in which we construct the heterogeneous graph by taking the predefined event types and all the words in the sentence as nodes, and modeling three relationships of "event-event", "event-argument" and "argument-argument" as edges. 2) We also design a multi-channel label enhancing module to better learn the distributed representation of each label in the multi-label extraction framework, and further enhance the interaction between the subtasks by considering the preliminary results of cause-effect type identification and event argument extraction. The experimental results on the benchmark dataset ECE-CCKS show that our approach outperforms previous state-of-the-art methods, and that our model also performs well on the complex samples with multiple cause-effect event pairs.

----

## [2105] MULTISCRIPT: Multimodal Script Learning for Supporting Open Domain Everyday Tasks

**Authors**: *Jingyuan Qi, Minqian Liu, Ying Shen, Zhiyang Xu, Lifu Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29854](https://doi.org/10.1609/aaai.v38i17.29854)

**Abstract**:

Automatically generating scripts (i.e. sequences of key steps described in text) from video demonstrations and reasoning about the subsequent steps are crucial to the modern AI virtual assistants to guide humans to complete everyday tasks, especially unfamiliar ones. However, current methods for generative script learning rely heavily on well-structured preceding steps described in text and/or images or are limited to a certain domain, resulting in a disparity with real-world user scenarios. To address these limitations, we present a new benchmark challenge – MULTISCRIPT, with two new tasks on task-oriented multimodal script learning: (1) multimodal script generation, and (2) subsequent step prediction. For both tasks, the input consists of a target task name and a video illustrating what has been done to complete the target task, and the expected output is (1) a sequence of structured step descriptions in text based on the demonstration video, and (2) a single text description for the subsequent step, respectively. Built from WikiHow, MULTISCRIPT covers multimodal scripts in videos and text descriptions for over 6,655 human everyday tasks across 19 diverse domains. To establish baseline performance on MULTISCRIPT, we propose two knowledge-guided multimodal generative frameworks that incorporate the task-related knowledge prompted from large language models such as Vicuna. Experimental results show that our proposed approaches significantly improve over the competitive baselines.

----

## [2106] Exploring Transformer Extrapolation

**Authors**: *Zhen Qin, Yiran Zhong, Hui Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29855](https://doi.org/10.1609/aaai.v38i17.29855)

**Abstract**:

Length extrapolation has attracted considerable attention recently since it allows transformers to be tested on longer sequences than those used in training. Previous research has shown that this property can be attained by using carefully designed Relative Positional Encodings (RPEs). While these methods perform well on a variety of corpora, the conditions for length extrapolation have yet to be investigated. This paper attempts to determine what types of RPEs allow for length extrapolation through a thorough mathematical and empirical analysis. We discover that a transformer is certain to possess this property as long as the series that corresponds to the RPE's exponential converges. Two practices are derived from the conditions and examined in language modeling tasks on a variety of corpora. As a bonus from the conditions, we derive a new Theoretical Receptive Field (TRF) to measure the receptive field of RPEs without taking any training steps. Extensive experiments are conducted on the Wikitext-103, Books, Github, and WikiBook datasets to demonstrate the viability of our discovered conditions. We also compare TRF to Empirical Receptive Field (ERF) across different models, showing consistently matched trends on these datasets. Code is released at: https://github.com/OpenNLPLab/Rpe.

----

## [2107] Using Artificial Populations to Study Psychological Phenomena in Neural Models

**Authors**: *Jesse Roberts, Kyle Moore, Drew Wilenzick, Douglas H. Fisher*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29856](https://doi.org/10.1609/aaai.v38i17.29856)

**Abstract**:

The recent proliferation of research into transformer based natural language processing has led to a number of studies which attempt to detect the presence of human-like cognitive behavior in the models. We contend that, as is true of human psychology, the investigation of cognitive behavior in language models must be conducted in an appropriate population of an appropriate size for the results to be meaningful. We leverage work in uncertainty estimation in a novel approach to efficiently construct experimental populations. The resultant tool, PopulationLM, has been made open source. We provide theoretical grounding in the uncertainty estimation literature and motivation from current cognitive work regarding language models. We discuss the methodological lessons from other scientific communities and attempt to demonstrate their application to two artificial population studies. Through population based experimentation we find that language models exhibit behavior consistent with typicality effects among categories highly represented in training. However, we find that language models don't tend to exhibit structural priming effects. Generally, our results show that single models tend to over estimate the presence of cognitive behaviors in neural models.

----

## [2108] Better than Random: Reliable NLG Human Evaluation with Constrained Active Sampling

**Authors**: *Jie Ruan, Xiao Pu, Mingqi Gao, Xiaojun Wan, Yuesheng Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29857](https://doi.org/10.1609/aaai.v38i17.29857)

**Abstract**:

Human evaluation is viewed as a reliable evaluation method for NLG which is expensive and time-consuming. To save labor and costs, researchers usually perform human evaluation on a small subset of data sampled from the whole dataset in practice. However, different selection subsets will lead to different rankings of the systems. To give a more correct inter-system ranking and make the gold standard human evaluation more reliable, we propose a Constrained Active Sampling Framework (CASF) for reliable human judgment. CASF operates through a Learner, a Systematic Sampler and a Constrained Controller to select representative samples for getting a more correct inter-system ranking. Experiment results on 137 real NLG evaluation setups with 44 human evaluation metrics across 16 datasets and 5 NLG tasks demonstrate CASF receives 93.18\% top-ranked system recognition accuracy and ranks first or ranks second on 90.91\% of the human metrics with 0.83 overall inter-system ranking Kendall correlation. Code and data are publicly available online.

----

## [2109] VELMA: Verbalization Embodiment of LLM Agents for Vision and Language Navigation in Street View

**Authors**: *Raphael Schumann, Wanrong Zhu, Weixi Feng, Tsu-Jui Fu, Stefan Riezler, William Yang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29858](https://doi.org/10.1609/aaai.v38i17.29858)

**Abstract**:

Incremental decision making in real-world environments is one of the most challenging tasks in embodied artificial intelligence. One particularly demanding scenario is Vision and Language Navigation (VLN) which requires visual and natural language understanding as well as spatial and temporal reasoning capabilities. The embodied agent needs to ground its understanding of navigation instructions in observations of a real-world environment like Street View. Despite the impressive results of LLMs in other research areas, it is an ongoing problem of how to best connect them with an interactive visual environment. In this work, we propose VELMA, an embodied LLM agent that uses a verbalization of the trajectory and of visual environment observations as contextual prompt for the next action. Visual information is verbalized by a pipeline that extracts landmarks from the human written navigation instructions and uses CLIP to determine their visibility in the current panorama view. We show that VELMA is able to successfully follow navigation instructions in Street View with only two in-context examples. We further finetune the LLM agent on a few thousand examples and achieve around 25% relative improvement in task completion over the previous state-of-the-art for two datasets.

----

## [2110] OntoFact: Unveiling Fantastic Fact-Skeleton of LLMs via Ontology-Driven Reinforcement Learning

**Authors**: *Ziyu Shang, Wenjun Ke, Nana Xiu, Peng Wang, Jiajun Liu, Yanhui Li, Zhizhao Luo, Ke Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29859](https://doi.org/10.1609/aaai.v38i17.29859)

**Abstract**:

Large language models (LLMs) have demonstrated impressive proficiency in information retrieval, while they are prone to generating incorrect responses that conflict with reality, a phenomenon known as intrinsic hallucination. The critical challenge lies in the unclear and unreliable fact distribution within LLMs trained on vast amounts of data. The prevalent approach frames the factual detection task as a question-answering paradigm, where the LLMs are asked about factual knowledge and examined for correctness. However, existing studies primarily focused on deriving test cases only from several specific domains, such as movies and sports, limiting the comprehensive observation of missing knowledge and the analysis of unexpected hallucinations. To address this issue, we propose OntoFact, an adaptive framework for detecting unknown facts of LLMs, devoted to mining the ontology-level skeleton of the missing knowledge. Specifically, we argue that LLMs could expose the ontology-based similarity among missing facts and introduce five representative knowledge graphs (KGs) as benchmarks. We further devise a sophisticated ontology-driven reinforcement learning (ORL) mechanism to produce error-prone test cases with specific entities and relations automatically. The ORL mechanism rewards the KGs for navigating toward a feasible direction for unveiling factual errors. Moreover, empirical efforts demonstrate that dominant LLMs are biased towards answering Yes rather than No, regardless of whether this knowledge is included. To mitigate the overconfidence of LLMs, we leverage a hallucination-free detection (HFD) strategy to tackle unfair comparisons between baselines, thereby boosting the result robustness. Experimental results on 5 datasets, using 32 representative LLMs, reveal a general lack of fact in current LLMs. Notably, ChatGPT exhibits fact error rates of 51.6% on DBpedia and 64.7% on YAGO, respectively. Additionally, the ORL mechanism demonstrates promising error prediction scores, with F1 scores ranging from 70% to 90% across most LLMs. Compared to the exhaustive testing, ORL achieves an average recall of 80% while reducing evaluation time by 35.29% to 63.12%.

----

## [2111] Agile-Quant: Activation-Guided Quantization for Faster Inference of LLMs on the Edge

**Authors**: *Xuan Shen, Peiyan Dong, Lei Lu, Zhenglun Kong, Zhengang Li, Ming Lin, Chao Wu, Yanzhi Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29860](https://doi.org/10.1609/aaai.v38i17.29860)

**Abstract**:

Large Language Models (LLMs) stand out for their impressive performance in intricate language modeling tasks. However, their demanding computational and memory needs pose obstacles for broad use on edge devices. Quantization is then introduced to boost LLMs' on-device efficiency. Recent works show that 8-bit or lower weight quantization is feasible with minimal impact on end-to-end task performance, while the activation is still not quantized. On the other hand, mainstream commodity edge devices still struggle to execute these sub-8-bit quantized networks effectively. In this paper, we propose Agile-Quant, an Activation-Guided quantization framework for faster Inference of popular Large Language Models (LLMs) on the Edge. Considering the hardware profiling and activation analysis, we first introduce a basic activation quantization strategy to balance the trade-off of task performance and real inference speed. Then we leverage the activation-aware token pruning technique to reduce the outliers and the adverse impact on attentivity. Ultimately, we utilize the SIMD-based 4-bit multiplier and our efficient TRIP matrix multiplication to implement the accelerator for LLMs on the edge. We apply our framework on different scales of LLMs including LLaMA, OPT, and BLOOM with 4-bit or 8-bit for the activation and 4-bit for the weight quantization. Experiments show that Agile-Quant achieves simultaneous quantization of model weights and activations while maintaining task performance comparable to existing weight-only quantization methods. Moreover, in the 8- and 4-bit scenario, Agile-Quant achieves an on-device speedup of up to 2.55x compared to its FP16 counterparts across multiple edge devices, marking a pioneering advancement in this domain.

----

## [2112] CORECODE: A Common Sense Annotated Dialogue Dataset with Benchmark Tasks for Chinese Large Language Models

**Authors**: *Dan Shi, Chaobin You, Jiantao Huang, Taihao Li, Deyi Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29861](https://doi.org/10.1609/aaai.v38i17.29861)

**Abstract**:

As an indispensable ingredient of intelligence, commonsense reasoning is crucial for large language models (LLMs) in real-world scenarios. In this paper, we propose CORECODE, a dataset that contains abundant commonsense knowledge manually annotated on dyadic dialogues, to evaluate the commonsense reasoning and commonsense conflict detection capabilities of Chinese LLMs. We categorize commonsense knowledge in everyday conversations into three dimensions: entity, event, and social interaction. For easy and consistent annotation, we standardize the form of commonsense knowledge annotation in open-domain dialogues as "domain: slot = value". A total of 9 domains and 37 slots are defined to capture diverse commonsense knowledge. With these pre-defined domains and slots, we collect 76,787 commonsense knowledge annotations from 19,700 dialogues through crowdsourcing. To evaluate and enhance the commonsense reasoning capability for LLMs on the curated dataset, we establish a series of dialogue-level reasoning and detection tasks, including commonsense knowledge filling, commonsense knowledge generation, commonsense conflict phrase detection, domain identification, slot identification, and event causal inference. A wide variety of existing open-source Chinese LLMs are evaluated with these tasks on our dataset. Experimental results demonstrate that these models are not competent to predict CORECODE's plentiful reasoning content, and even ChatGPT could only achieve 0.275 and 0.084 accuracy on the domain identification and slot identification tasks under the zero-shot setting. We release the data and codes of CORECODE at https://github.com/danshi777/CORECODE to promote commonsense reasoning evaluation and study of LLMs in the context of daily conversations.

----

## [2113] A Unified Knowledge Transfer Network for Generalized Category Discovery

**Authors**: *Wenkai Shi, Wenbin An, Feng Tian, Yan Chen, Yaqiang Wu, Qianying Wang, Ping Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29862](https://doi.org/10.1609/aaai.v38i17.29862)

**Abstract**:

Generalized Category Discovery (GCD) aims to recognize both known and novel categories in an unlabeled dataset by leveraging another labeled dataset with only known categories. Without considering knowledge transfer from known to novel categories, current methods usually perform poorly on novel categories due to the lack of corresponding supervision. To mitigate this issue, we propose a unified Knowledge Transfer Network (KTN), which solves two obstacles to knowledge transfer in GCD. First, the mixture of known and novel categories in unlabeled data makes it difficult to identify transfer candidates (i.e., samples with novel categories). For this, we propose an entropy-based method that leverages knowledge in the pre-trained classifier to differentiate known and novel categories without requiring extra data or parameters. Second, the lack of prior knowledge of novel categories presents challenges in quantifying semantic relationships between categories to decide the transfer weights. For this, we model different categories with prototypes and treat their similarities as transfer weights to measure the semantic similarities between categories. On the basis of two treatments, we transfer knowledge from known to novel categories by conducting pre-adjustment of logits and post-adjustment of labels for transfer candidates based on the transfer weights between different categories. With the weighted adjustment, KTN can generate more accurate pseudo-labels for unlabeled data, which helps to learn more discriminative features and boost model performance on novel categories. Extensive experiments show that our method outperforms state-of-the-art models on all evaluation metrics across multiple benchmark datasets. Furthermore, different from previous clustering-based methods that can only work offline with abundant data, KTN can be deployed online conveniently with faster inference speed. Code and data are available at https://github.com/yibai-shi/KTN.

----

## [2114] RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting

**Authors**: *Lei Shu, Liangchen Luo, Jayakumar Hoskere, Yun Zhu, Yinxiao Liu, Simon Tong, Jindong Chen, Lei Meng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29863](https://doi.org/10.1609/aaai.v38i17.29863)

**Abstract**:

Large Language Models (LLMs) have demonstrated impressive  capabilities in creative tasks such as storytelling and E-mail generation. However, as LLMs are primarily trained on final text results rather than intermediate revisions, it might be challenging for them to perform text rewriting tasks. Most studies in the rewriting tasks focus on a particular transformation type within the boundaries of single sentences. In this work, we develop new strategies for instruction tuning and reinforcement learning to better align LLMs for cross-sentence rewriting tasks using diverse wording and structures expressed through natural languages including 1) generating rewriting instruction data from Wiki edits and public corpus through instruction generation and chain-of-thought prompting; 2) collecting comparison data for reward model training through a new ranking function. To facilitate this research, we introduce OpenRewriteEval, a novel benchmark covers a wide variety of rewriting types expressed through natural language instructions. Our results show significant improvements over a variety of baselines.

----

## [2115] Well, Now We Know! Unveiling Sarcasm: Initiating and Exploring Multimodal Conversations with Reasoning

**Authors**: *Gopendra Vikram Singh, Mauajama Firdaus, Dushyant Singh Chauhan, Asif Ekbal, Pushpak Bhattacharyya*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29864](https://doi.org/10.1609/aaai.v38i17.29864)

**Abstract**:

Sarcasm is a widespread linguistic phenomenon that poses a considerable challenge to explain due to its subjective nature, absence of contextual cues, and rooted personal
perspectives. Even though the identification of sarcasm has been extensively studied in dialogue analysis, merely detecting sarcasm falls short of enabling conversational systems to genuinely comprehend the underlying meaning of a conversation and generate fitting responses. It is imperative to not only detect sarcasm but also pinpoint its origination and the rationale behind the sarcastic expressions to capture its authentic essence. In this paper, we delve into the discourse structure of conversations infused with sarcasm and introduce a novel task - Sarcasm Initiation and Reasoning in Conversations (SIRC). Embedded in a multimodal environment and
involving a combination of both English and code-mixed interactions, the objective of the task is to discern the trigger or starting point of sarcasm. Additionally, the task involves producing a natural language explanation that rationalizes the satirical dialogues. To this end, we introduce Sarcasm Initiation and Reasoning Dataset (SIRD) to facilitate our task and provide sarcasm initiation annotations and reasoning. We develop a comprehensive model named Sarcasm Initiation and Reasoning Generation (SIRG), which is designed to encompass textual, audio, and visual representations. To achieve this, we introduce a unique shared fusion method that employs cross-attention mechanisms to seamlessly integrate these diverse modalities. Our experimental outcomes, conducted on the SIRC dataset, demonstrate that our proposed framework establishes a new benchmark for both sarcasm initiation and its reasoning generation in the context of multimodal conversations. The code and dataset can be accessed from https://www.iitp.ac.in/∼ai-nlp-ml resources.html#sarcasm-explain and https://github.com/GussailRaat/SIRG-Sarcasm-Initiation-and-Reasoning-Generation.

----

## [2116] Preference Ranking Optimization for Human Alignment

**Authors**: *Feifan Song, Bowen Yu, Minghao Li, Haiyang Yu, Fei Huang, Yongbin Li, Houfeng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29865](https://doi.org/10.1609/aaai.v38i17.29865)

**Abstract**:

Large language models (LLMs) often contain misleading content, emphasizing the need to align them with human values to ensure secure AI systems. Reinforcement learning from human feedback (RLHF) has been employed to achieve this alignment. However, it encompasses two main drawbacks: (1) RLHF exhibits complexity, instability, and sensitivity to hyperparameters in contrast to SFT. (2) Despite massive trial-and-error, multiple sampling is reduced to pair-wise contrast, thus lacking contrasts from a macro perspective. In this paper, we propose Preference Ranking Optimization (PRO) as an efficient SFT algorithm to directly fine-tune LLMs for human alignment. PRO extends the pair-wise contrast to accommodate preference rankings of any length. By iteratively contrasting candidates, PRO instructs the LLM to prioritize the best response while progressively ranking the rest responses. In this manner, PRO effectively transforms human alignment into aligning the probability ranking of n responses generated by LLM with the preference ranking of humans towards these responses. Experiments have shown that PRO outperforms baseline algorithms, achieving comparable results to ChatGPT and human responses through automatic-based, reward-based, GPT-4, and human evaluations.

----

## [2117] TACIT: A Target-Agnostic Feature Disentanglement Framework for Cross-Domain Text Classification

**Authors**: *Rui Song, Fausto Giunchiglia, Yingji Li, Mingjie Tian, Hao Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29866](https://doi.org/10.1609/aaai.v38i17.29866)

**Abstract**:

Cross-domain text classification aims to transfer models from label-rich source domains to label-poor target domains, giving it a wide range of practical applications. Many approaches promote cross-domain generalization by capturing domaininvariant features. However, these methods rely on unlabeled samples provided by the target domains, which renders the model ineffective when the target domain is agnostic. Furthermore, the models are easily disturbed by shortcut learning in the source domain, which also hinders the improvement of domain generalization ability. To solve the aforementioned issues, this paper proposes TACIT, a target domain agnostic feature disentanglement framework which adaptively decouples robust and unrobust features by Variational Auto-Encoders. Additionally, to encourage the separation of unrobust features from robust features, we design a feature distillation task that compels unrobust features to approximate the output of the teacher. The teacher model is trained with a few easy samples that are easy to carry potential unknown shortcuts. Experimental results verify that our framework achieves comparable results to state-of-the-art baselines while utilizing only source domain data.

----

## [2118] A Dual-Way Enhanced Framework from Text Matching Point of View for Multimodal Entity Linking

**Authors**: *Shezheng Song, Shan Zhao, Chengyu Wang, Tianwei Yan, Shasha Li, Xiaoguang Mao, Meng Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29867](https://doi.org/10.1609/aaai.v38i17.29867)

**Abstract**:

Multimodal Entity Linking (MEL) aims at linking ambiguous mentions with multimodal information to entity in Knowledge Graph (KG) such as Wikipedia, which plays a key role in many applications. However, existing methods suffer from shortcomings, including modality impurity such as noise in raw image and ambiguous textual entity representation, which puts obstacles to MEL. We formulate multimodal entity linking as a neural text matching problem where each multimodal information (text and image) is treated as a query, and the model learns the mapping from each query to the relevant entity from candidate entities. This paper introduces a dual-way enhanced (DWE) framework for MEL: (1) our model refines queries with multimodal data and addresses semantic gaps using cross-modal enhancers between text and image information. Besides, DWE innovatively leverages fine-grained image attributes, including facial characteristic and scene feature, to enhance and refine visual features. (2)By using Wikipedia descriptions, DWE enriches entity semantics and obtains more comprehensive textual representation, which reduces between textual representation and the entities in KG. Extensive experiments on three public benchmarks demonstrate that our method achieves state-of-the-art (SOTA) performance, indicating the superiority of our model. The code is released on https://github.com/season1blue/DWE.

----

## [2119] RoPDA: Robust Prompt-Based Data Augmentation for Low-Resource Named Entity Recognition

**Authors**: *Sihan Song, Furao Shen, Jian Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29868](https://doi.org/10.1609/aaai.v38i17.29868)

**Abstract**:

Data augmentation has been widely used in low-resource NER tasks to tackle the problem of data sparsity. However, previous data augmentation methods have the disadvantages of disrupted syntactic structures, token-label mismatch, and requirement for external knowledge or manual effort. To address these issues, we propose Robust Prompt-based Data Augmentation (RoPDA) for low-resource NER. Based on pre-trained language models (PLMs) with continuous prompt, RoPDA performs entity augmentation and context augmentation through five fundamental augmentation operations to generate label-flipping and label-preserving examples. To optimize the utilization of the augmented samples, we present two techniques: self-consistency filtering and mixup. The former effectively eliminates low-quality samples with a bidirectional mask, while the latter prevents performance degradation arising from the direct utilization of labelflipping samples. Extensive experiments on three popular benchmarks from different domains demonstrate that RoPDA significantly improves upon strong baselines, and also outperforms state-of-the-art semi-supervised learning methods when unlabeled data is included.

----

## [2120] Wikiformer: Pre-training with Structured Information of Wikipedia for Ad-Hoc Retrieval

**Authors**: *Weihang Su, Qingyao Ai, Xiangsheng Li, Jia Chen, Yiqun Liu, Xiaolong Wu, Shengluan Hou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29869](https://doi.org/10.1609/aaai.v38i17.29869)

**Abstract**:

With the development of deep learning and natural language processing techniques, pre-trained language models have been widely used to solve information retrieval (IR) problems. Benefiting from the pre-training and fine-tuning paradigm, these models achieve state-of-the-art performance. In previous works, plain texts in Wikipedia have been widely used in the pre-training stage. However, the rich structured information in Wikipedia, such as the titles, abstracts, hierarchical heading (multi-level title) structure, relationship between articles, references, hyperlink structures, and the writing organizations, has not been fully explored. In this paper, we devise four pre-training objectives tailored for IR tasks based on the structured knowledge of Wikipedia. Compared to existing pre-training methods, our approach can better capture the semantic knowledge in the training corpus by leveraging the human-edited structured data from Wikipedia. Experimental results on multiple IR benchmark datasets show the superior performance of our model in both zero-shot and fine-tuning settings compared to existing strong retrieval baselines. Besides, experimental results in biomedical and legal domains demonstrate that our approach achieves better performance in vertical domains compared to previous models, especially in scenarios where long text similarity matching is needed. The code is available at https://github.com/oneal2000/Wikiformer.

----

## [2121] SIG: Speaker Identification in Literature via Prompt-Based Generation

**Authors**: *Zhenlin Su, Liyan Xu, Jin Xu, Jiangnan Li, Mingdu Huangfu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29870](https://doi.org/10.1609/aaai.v38i17.29870)

**Abstract**:

Identifying speakers of quotations in narratives is an important task in literary analysis, with challenging scenarios including the out-of-domain inference for unseen speakers, and non-explicit cases where there are no speaker mentions in surrounding context. 
In this work, we propose a simple and effective approach SIG, a generation-based method that verbalizes the task and quotation input based on designed prompt templates, which also enables easy integration of other auxiliary tasks that further bolster the speaker identification performance. The prediction can either come from direct generation by the model, or be determined by the highest generation probability of each speaker candidate. Based on our approach design, SIG supports out-of-domain evaluation, and achieves open-world classification paradigm that is able to accept any forms of candidate input. We perform both cross-domain evaluation and in-domain evaluation on PDNC, the largest dataset of this task, where empirical results suggest that SIG outperforms previous baselines of complicated designs, as well as the zero-shot ChatGPT, especially excelling at those hard non-explicit scenarios by up to 17% improvement. Additional experiments on another dataset WP further corroborate the efficacy of SIG.

----

## [2122] Collaborative Synthesis of Patient Records through Multi-Visit Health State Inference

**Authors**: *Hongda Sun, Hongzhan Lin, Rui Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29871](https://doi.org/10.1609/aaai.v38i17.29871)

**Abstract**:

Electronic health records (EHRs) have become the foundation of machine learning applications in healthcare, while the utility of real patient records is often limited by privacy and security concerns. Synthetic EHR generation provides an additional perspective to compensate for this limitation. Most existing methods synthesize new records based on real EHR data, without consideration of different types of events in EHR data, which cannot control the event combinations in line with medical common sense. In this paper, we propose MSIC,  a Multi-visit health Status Inference model for Collaborative EHR synthesis to address these limitations. First, we formulate the synthetic EHR generation process as a probabilistic graphical model and tightly connect different types of events by modeling the latent health states. Then, we derive a health state inference method tailored for the multi-visit scenario to effectively utilize previous records to synthesize current and future records. Furthermore, we propose to generate medical reports to add textual descriptions for each medical event,  providing broader applications for synthesized EHR data. For generating different paragraphs in each visit, we incorporate a multi-generator deliberation framework to collaborate the message passing of multiple generators and employ a two-phase decoding strategy to generate high-quality reports. Our extensive experiments on the widely used benchmarks, MIMIC-III and MIMIC-IV, demonstrate that MSIC advances state-of-the-art results on the quality of synthetic data while maintaining low privacy risks.

----

## [2123] SciEval: A Multi-Level Large Language Model Evaluation Benchmark for Scientific Research

**Authors**: *Liangtai Sun, Yang Han, Zihan Zhao, Da Ma, Zhennan Shen, Baocai Chen, Lu Chen, Kai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29872](https://doi.org/10.1609/aaai.v38i17.29872)

**Abstract**:

Recently, there has been growing interest in using Large Language Models (LLMs) for scientific research. Numerous benchmarks have been proposed to evaluate the ability of LLMs for scientific research. However, current benchmarks are mostly based on pre-collected objective questions. This design suffers from data leakage problem and lacks the evaluation of subjective Q/A ability. In this paper, we propose SciEval, a comprehensive and multi-disciplinary evaluation benchmark to address these issues. Based on Bloom's taxonomy, SciEval covers four dimensions to systematically evaluate scientific research ability. In particular, we design a "dynamic" subset based on scientific principles to prevent evaluation from potential data leakage. Both objective and subjective questions are included in SciEval. These characteristics make SciEval a more effective benchmark for scientific research ability evaluation of LLMs. Comprehensive experiments on most advanced LLMs show that, although GPT-4 achieves SOTA performance compared to other LLMs, there is still substantial room for improvement, especially for dynamic questions. The codes and data are publicly available on https://github.com/OpenDFM/SciEval.

----

## [2124] UMIE: Unified Multimodal Information Extraction with Instruction Tuning

**Authors**: *Lin Sun, Kai Zhang, Qingyuan Li, Renze Lou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29873](https://doi.org/10.1609/aaai.v38i17.29873)

**Abstract**:

Multimodal information extraction (MIE) gains significant attention as the popularity of multimedia content increases. However, current MIE methods often resort to using task-specific model structures, which results in limited generalizability across tasks and underutilizes shared knowledge across MIE tasks. To address these issues, we propose UMIE, a unified multimodal information extractor to unify three MIE tasks as a generation problem using instruction tuning, being able to effectively extract both textual and visual mentions. Extensive experiments show that our single UMIE outperforms various state-of-the-art (SoTA) methods across six MIE datasets on three tasks. Furthermore, in-depth analysis demonstrates UMIE's strong generalization in the zero-shot setting, robustness to instruction variants, and interpretability. Our research serves as an initial step towards a unified MIE model and initiates the exploration into both instruction tuning and large language models within the MIE domain. Our code, data, and model are available at https://github.com/ZUCC-AI/UMIE.

----

## [2125] InstructDoc: A Dataset for Zero-Shot Generalization of Visual Document Understanding with Instructions

**Authors**: *Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko Saito, Jun Suzuki*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29874](https://doi.org/10.1609/aaai.v38i17.29874)

**Abstract**:

We study the problem of completing various visual document understanding (VDU) tasks, e.g., question answering and information extraction, on real-world documents through human-written instructions. To this end, we propose InstructDoc, the first large-scale collection of 30 publicly available VDU datasets, each with diverse instructions in a unified format, which covers a wide range of 12 tasks and includes open document types/formats. Furthermore, to enhance the generalization performance on VDU tasks, we design a new instruction-based document reading and understanding model, InstructDr, that connects document images, image encoders, and large language models (LLMs) through a trainable bridging module. Experiments demonstrate that InstructDr can effectively adapt to new VDU datasets, tasks, and domains via given instructions and outperforms existing multimodal LLMs and ChatGPT without specific training.

----

## [2126] Graph Neural Prompting with Large Language Models

**Authors**: *Yijun Tian, Huan Song, Zichen Wang, Haozhu Wang, Ziqing Hu, Fang Wang, Nitesh V. Chawla, Panpan Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29875](https://doi.org/10.1609/aaai.v38i17.29875)

**Abstract**:

Large language models (LLMs) have shown remarkable generalization capability with exceptional performance in various language modeling tasks. However, they still exhibit inherent limitations in precisely capturing and returning grounded knowledge. While existing work has explored utilizing knowledge graphs (KGs) to enhance language modeling via joint training and customized model architectures, applying this to LLMs is problematic owing to their large number of parameters and high computational cost. Therefore, how to enhance pre-trained LLMs using grounded knowledge, e.g., retrieval-augmented generation, remains an open question. In this work, we propose Graph Neural Prompting (GNP), a novel plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from KGs. GNP encompasses various designs, including a standard graph neural network encoder, a cross-modality pooling module, a domain projector, and a self-supervised link prediction objective. Extensive experiments on multiple datasets demonstrate the superiority of GNP on both commonsense and biomedical reasoning tasks across different LLM sizes and settings. Code is available at https://github.com/meettyj/GNP.

----

## [2127] Adaptive Graph Learning for Multimodal Conversational Emotion Detection

**Authors**: *Geng Tu, Tian Xie, Bin Liang, Hongpeng Wang, Ruifeng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29876](https://doi.org/10.1609/aaai.v38i17.29876)

**Abstract**:

Multimodal Emotion Recognition in Conversations (ERC) aims to identify the emotions conveyed by each utterance in a conversational video. Current efforts encounter challenges in balancing intra- and inter-speaker context dependencies when tackling intra-modal interactions. This balance is vital as it encompasses modeling self-dependency (emotional inertia) where speakers' own emotions affect them and modeling interpersonal dependencies (empathy) where counterparts' emotions influence a speaker. Furthermore, challenges arise in addressing cross-modal interactions that involve content with conflicting emotions across different modalities. To address this issue, we introduce an adaptive interactive graph network (IGN) called AdaIGN that employs the Gumbel Softmax trick to adaptively select nodes and edges, enhancing intra- and cross-modal interactions. Unlike undirected graphs, we use a directed IGN to prevent future utterances from impacting the current one. Next, we propose Node- and Edge-level Selection Policies (NESP) to guide node and edge selection, along with a Graph-Level Selection Policy (GSP) to integrate the utterance representation from original IGN and NESP-enhanced IGN. Moreover, we design a task-specific loss function that prioritizes text modality and intra-speaker context selection. To reduce computational complexity, we use pre-defined pseudo labels through self-supervised methods to mask unnecessary utterance nodes for selection. Experimental results show that AdaIGN outperforms state-of-the-art methods on two popular datasets. Our code will be available at https://github.com/TuGengs/AdaIGN.

----

## [2128] Dependency Structure-Enhanced Graph Attention Networks for Event Detection

**Authors**: *Qizhi Wan, Changxuan Wan, Keli Xiao, Kun Lu, Chenliang Li, Xiping Liu, Dexi Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29877](https://doi.org/10.1609/aaai.v38i17.29877)

**Abstract**:

Existing models on event detection share three-fold limitations, including (1) insufficient consideration of the structures between dependency relations, (2) limited exploration of the directed-edge semantics, and (3) issues in strengthening the event core arguments. To tackle these problems, we propose a dependency structure-enhanced event detection framework. In addition to the traditional token dependency parsing tree, denoted as TDG, our model considers the dependency edges in it as new nodes and constructs a dependency relation graph (DRG). DRG allows the embedding representations of dependency relations to be updated as nodes rather than edges in a graph neural network. 
Moreover, the levels of core argument nodes in the two graphs are adjusted by dependency relation types in TDG to enhance their status. Subsequently, the two graphs are further encoded and jointly trained in graph attention networks (GAT). Importantly, we design an interaction strategy of node embedding for the two graphs and refine the attention coefficient computational method to encode the semantic meaning of directed edges. Extensive experiments are conducted to validate the effectiveness of our method, and the results confirm its superiority over the state-of-the-art baselines. Our model outperforms the best benchmark with the F1 score increased by 3.5 and 3.4 percentage points on ACE2005 English and Chinese corpus.

----

## [2129] ESRL: Efficient Sampling-Based Reinforcement Learning for Sequence Generation

**Authors**: *Chenglong Wang, Hang Zhou, Yimin Hu, Yifu Huo, Bei Li, Tongran Liu, Tong Xiao, Jingbo Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29878](https://doi.org/10.1609/aaai.v38i17.29878)

**Abstract**:

Applying Reinforcement Learning (RL) to sequence generation models enables the direct optimization of long-term rewards (e.g., BLEU and human feedback), but typically requires large-scale sampling over a space of action sequences. This is a computational challenge as presented by the practice of sequence generation problems, such as machine translation, where we often deal with a large action space (e.g., a vocabulary) and a long action sequence (e.g., a translation). In this work, we introduce two-stage sampling and dynamic sampling approaches to improve the sampling efficiency during training sequence generation models via RL. We experiment with our approaches on the traditional sequence generation tasks, including machine translation and abstractive summarization. Furthermore, we evaluate our approaches in RL from human feedback (RLHF) through training a large language model using the reward model. Experimental results show that the efficient sampling-based RL, referred to as ESRL, can outperform all baselines in terms of both training efficiency and memory consumption. Notably, ESRL yields consistent performance gains over the strong REINFORCE, minimum risk training, and proximal policy optimization methods. The code is available at https://github.com/wangclnlp/DeepSpeed-Chat-Extension/examples/esrl.

----

## [2130] Exploring Equation as a Better Intermediate Meaning Representation for Numerical Reasoning of Large Language Models

**Authors**: *Dingzirui Wang, Longxu Dou, Wenbin Zhang, Junyu Zeng, Wanxiang Che*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29879](https://doi.org/10.1609/aaai.v38i17.29879)

**Abstract**:

Numerical reasoning is a vital capability for natural language processing models to understand and process numerical information in real-world scenarios. Most current methods first generate the Intermediate Meaning Representations (IMRs) of questions and then generate answers. Current SOTA methods generate programs as IMRs with large language models (LLMs). Intuitively, equations have fewer restrictions and closer semantics to the question than programs, leading to higher generation accuracy. However, current LLMs generate equations worse than programs, where we assume that the equation data is rare in pre-training data compared to programs. So in this paper, we try to use equations as IMRs to solve the numerical reasoning task by addressing two problems: (1) Theoretically, how to prove that the equation is an IMR with higher generation accuracy than programs; (2) Empirically, how to improve the generation accuracy of equations with LLMs. For the first problem, we propose and prove a proposition to theoretically compare the generation accuracy of different IMRs. For the second problem, we present a method called Boosting Numerical ReasonIng by Decomposing the Generation of Equations Bridge, which can improve the accuracy of LLMs in generating equations as IMRs by reducing the tendency of generating constant expressions and programs. Our method improves the performance by 2.2%, 0.9%, and 1.7% on GSM8K, SVAMP, and Algebra datasets compared to the previous state-of-the-art methods under the single reasoning path setting. Our code and prompts are available at https://github.com/zirui-HIT/Bridge_for_Numerical_Reasoning}.

----

## [2131] Manifold-Based Verbalizer Space Re-embedding for Tuning-Free Prompt-Based Classification

**Authors**: *Haochun Wang, Sendong Zhao, Chi Liu, Nuwa Xi, Muzhen Cai, Bing Qin, Ting Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29880](https://doi.org/10.1609/aaai.v38i17.29880)

**Abstract**:

Prompt-based classification adapts tasks to a cloze question format utilizing the [MASK] token and the filled tokens are then mapped to labels through pre-defined verbalizers. Recent studies have explored the use of verbalizer embeddings to reduce labor in this process. However, all existing studies require a tuning process for either the pre-trained models or additional trainable embeddings. Meanwhile, the distance between high-dimensional verbalizer embeddings should not be measured by Euclidean distance due to the potential for non-linear manifolds in the representation space. In this study, we propose a tuning-free manifold-based space re-embedding method called Locally Linear Embedding with Intra-class Neighborhood Constraint (LLE-INC) for verbalizer embeddings, which preserves local properties within the same class as guidance for classification. Experimental results indicate that even without tuning any parameters, our LLE-INC is on par with automated verbalizers with parameter tuning. And with the parameter updating, our approach further enhances prompt-based tuning by up to 3.2%. Furthermore, experiments with the LLaMA-7B&13B indicate that LLE-INC is an efficient tuning-free classification approach for the hyper-scale language models.

----

## [2132] Improving the Robustness of Knowledge-Grounded Dialogue via Contrastive Learning

**Authors**: *Jiaan Wang, Jianfeng Qu, Kexin Wang, Zhixu Li, Wen Hua, Ximing Li, An Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29881](https://doi.org/10.1609/aaai.v38i17.29881)

**Abstract**:

Knowledge-grounded dialogue (KGD) learns to generate an informative response based on a given dialogue context and external knowledge (e.g., knowledge graphs; KGs). Recently, the emergence of large language models (LLMs) and pre-training techniques has brought great success to knowledge-grounded dialogue. However, when building KGD systems in real applications, there are various real-world noises that are inevitable to face. For example, the dialogue context might involve perturbations such as misspellings and abbreviations. In addition, KGs typically suffer from incompletion and also might contain erroneous and outdated facts. Such real-world noises pose a challenge to the robustness of KGD systems and hinder their applications in the real world. In this paper, we propose an entity-based contrastive learning framework for improving the robustness of KGD. Specifically, we make use of the entity information in a KGD sample to create both its positive and negative samples which involve semantic-irrelevant and semantic-relevant perturbations, respectively. The contrastive learning framework ensures the KGD model is aware of these two types of perturbations, thus could generate informative responses with the potentially noisy inputs in real applications. Experimental results on three widely-used benchmark datasets show that our method achieves new state-of-the-art performance in terms of automatic evaluation scores, verifying its effectiveness and potentiality. Furthermore, we show that our method is able to generate better responses than comparison models in both the noisy and the few-shot settings.

----

## [2133] Restoring Speaking Lips from Occlusion for Audio-Visual Speech Recognition

**Authors**: *Jiadong Wang, Zexu Pan, Malu Zhang, Robby T. Tan, Haizhou Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29882](https://doi.org/10.1609/aaai.v38i17.29882)

**Abstract**:

Prior studies on audio-visual speech recognition typically assume the visibility of speaking lips, ignoring the fact that visual occlusion occurs in real-world videos, thus adversely affecting recognition performance. To address this issue, we propose a framework that restores occluded lips in a video by utilizing both the video itself and the corresponding noisy audio. Specifically, the framework aims to achieve these three tasks: detecting occluded frames, masking occluded areas, and reconstruction of masked regions. We tackle the first two issues by utilizing the Class Activation Map (CAM) obtained from occluded frame detection to facilitate the masking of occluded areas. Additionally, we introduce a novel synthesis-matching strategy for the reconstruction to ensure the compatibility of audio features with different levels of occlusion. Our framework is evaluated in terms of Word Error Rate (WER) on the original videos, the videos corrupted by concealed lips, and the videos restored using the framework with several existing state-of-the-art audio-visual speech recognition methods. Experimental results substantiate that our framework significantly mitigates performance degradation resulting from lip occlusion. Under -5dB noise conditions, AV-Hubert's WER increases from 10.62% to 13.87% due to lip occlusion, but rebounds to 11.87% in conjunction with the proposed framework. Furthermore, the framework also demonstrates its capacity to produce natural synthesized images in qualitative assessments.

----

## [2134] Learning from Failure: Improving Meeting Summarization without Good Samples

**Authors**: *Ke Wang, Xiutian Zhao, Wei Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29883](https://doi.org/10.1609/aaai.v38i17.29883)

**Abstract**:

Existing methods aligning language models with various human needs are reliant heavily on high-quality and task-specific data. However, industrial deployment of task-specific language models often encounter challenges in the availability of appropriate training samples. Taking meeting summarization for instance, public datasets are scarce, and private corpora are also hard to obtain due to privacy issues or resource-demanding annotation. To improve meeting summarization in the absence of positively-rated (i.e., ``good'') samples, we propose Score Tuning, a cold start tuning framework that leverages bad samples of distinguishable degrees to incrementally enhance the performance of summary generation without an initial presence of good samples. Our method utilizes asynchronous and numerical human feedback that measure the quality of generated summaries. Formulating data into triplets of (transcript, summary, score), our approach instructs a pre-trained model to learn the association between summary qualities and human-rated scores and hence to generate better summaries corresponding to higher scores. The experiment results show that our method is effective in improving meeting summarization on both English and Chinese corpora while requiring less annotated data and training resources compared to existing alignment methods. Additionally, we also preliminarily explore the transferability of our approach in machine translation tasks and demonstrate its potential for future development and usage in other domains.

----

## [2135] T-SciQ: Teaching Multimodal Chain-of-Thought Reasoning via Large Language Model Signals for Science Question Answering

**Authors**: *Lei Wang, Yi Hu, Jiabang He, Xing Xu, Ning Liu, Hui Liu, Heng Tao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29884](https://doi.org/10.1609/aaai.v38i17.29884)

**Abstract**:

Large Language Models (LLMs) have recently demonstrated exceptional performance in various Natural Language Processing (NLP) tasks. They have also shown the ability to perform chain-of-thought (CoT) reasoning to solve complex problems. Recent studies have explored CoT reasoning in complex multimodal scenarios, such as the science question answering task, by fine-tuning multimodal models with high-quality human-annotated CoT rationales. However, collecting high-quality COT rationales is usually time-consuming and costly. Besides, the annotated rationales are hardly accurate due to the external essential information missed. To address these issues, we propose a novel method termed T-SciQ that aims at teaching science question answering with LLM signals. The T-SciQ approach generates high-quality CoT rationales as teaching signals and is advanced to train much smaller models to perform CoT reasoning in complex modalities. Additionally, we introduce a novel data mixing strategy to produce more effective teaching data samples for simple and complex science question answer problems. Extensive experimental results show that our T-SciQ method achieves a new state-of-the-art performance on the ScienceQA benchmark, with an accuracy of 96.18%. Moreover, our approach outperforms the most powerful fine-tuned baseline by 4.5%. The code is publicly available at https://github.com/T-SciQ/T-SciQ.

----

## [2136] Mitigating the Impact of False Negative in Dense Retrieval with Contrastive Confidence Regularization

**Authors**: *Shiqi Wang, Yeqin Zhang, Cam-Tu Nguyen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29885](https://doi.org/10.1609/aaai.v38i17.29885)

**Abstract**:

In open-domain Question Answering (QA), dense text retrieval is crucial for finding relevant passages to generate answers. Typically, contrastive learning is used to train a retrieval model, which maps passages and queries to the same semantic space, making similar ones closer and dissimilar ones further apart. However, training such a system is challenging due to the false negative problem, where relevant passages may be missed during data annotation. Hard negative sampling, commonly used to improve contrastive learning, can introduce more noise in training. This is because hard negatives are those close to a given query, and thus more likely to be false negatives. To address this, we propose a novel contrastive confidence regularizer for Noise Contrastive Estimation (NCE) loss, a commonly used contrastive loss. Our analysis shows that the regularizer helps make the dense retrieval model more robust against false negatives with a theoretical guarantee. Additionally, we propose a model-agnostic method to filter out noisy negative passages in the dataset, improving any downstream dense retrieval models. Through experiments on three datasets, we demonstrate that our method achieves better retrieval performance in comparison to existing state-of-the-art dense retrieval systems.

----

## [2137] DenoSent: A Denoising Objective for Self-Supervised Sentence Representation Learning

**Authors**: *Xinghao Wang, Junliang He, Pengyu Wang, Yunhua Zhou, Tianxiang Sun, Xipeng Qiu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29886](https://doi.org/10.1609/aaai.v38i17.29886)

**Abstract**:

Contrastive-learning-based methods have dominated sentence representation learning. These methods regularize the representation space by pulling similar sentence representations closer and pushing away the dissimilar ones and have been proven effective in various NLP tasks, e.g., semantic textual similarity (STS) tasks. However, it is challenging for these methods to learn fine-grained semantics as they only learn from the inter-sentence perspective, i.e., their supervision signal comes from the relationship between data samples. In this work, we propose a novel denoising objective that inherits from another perspective, i.e., the intra-sentence perspective. By introducing both discrete and continuous noise, we generate noisy sentences and then train our model to restore them to their original form. Our empirical evaluations demonstrate that this approach delivers competitive results on both semantic textual similarity (STS) and a wide range of transfer tasks, standing up well in comparison to contrastive-learning-based methods. Notably, the proposed intra-sentence denoising objective complements existing inter-sentence contrastive methodologies and can be integrated with them to further enhance performance.  Our code is available at https://github.com/xinghaow99/DenoSent.

----

## [2138] LLMRG: Improving Recommendations through Large Language Model Reasoning Graphs

**Authors**: *Yan Wang, Zhixuan Chu, Xin Ouyang, Simeng Wang, Hongyan Hao, Yue Shen, Jinjie Gu, Siqiao Xue, James Zhang, Qing Cui, Longfei Li, Jun Zhou, Sheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29887](https://doi.org/10.1609/aaai.v38i17.29887)

**Abstract**:

Recommendation systems aim to provide users with relevant suggestions, but often lack interpretability and fail to capture higher-level semantic relationships between user behaviors and profiles. In this paper, we propose a novel approach that leverages large language models (LLMs) to construct personalized reasoning graphs. These graphs link a user's profile and behavioral sequences through causal and logical inferences, representing the user's interests in an interpretable way. Our approach, LLM reasoning graphs (LLMRG), has four components: chained graph reasoning, divergent extension, self-verification and scoring, and knowledge base self-improvement. The resulting reasoning graph is encoded using graph neural networks, which serves as additional input to improve conventional recommender systems, without requiring extra user or item information. Our approach demonstrates how LLMs can enable more logical and interpretable recommender systems through personalized reasoning graphs. LLMRG allows recommendations to benefit from both engineered recommendation systems and LLM-derived reasoning graphs. We demonstrate the effectiveness of LLMRG on benchmarks and real-world scenarios in enhancing base recommendation models.

----

## [2139] A Positive-Unlabeled Metric Learning Framework for Document-Level Relation Extraction with Incomplete Labeling

**Authors**: *Ye Wang, Huazheng Pan, Tao Zhang, Wen Wu, Wenxin Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29888](https://doi.org/10.1609/aaai.v38i17.29888)

**Abstract**:

The goal of document-level relation extraction (RE) is to identify relations between entities that span multiple sentences. Recently, incomplete labeling in document-level RE has received increasing attention, and some studies have used methods such as positive-unlabeled learning to tackle this issue, but there is still a lot of room for improvement. Motivated by this, we propose a positive-augmentation and positive-mixup positive-unlabeled metric learning framework (P3M). Specifically, we formulate document-level RE as a metric learning problem. We aim to pull the distance closer between entity pair embedding and their corresponding relation embedding, while pushing it farther away from the none-class relation embedding. Additionally, we adapt the positive-unlabeled learning to this loss objective. In order to improve the generalizability of the model, we use dropout to augment positive samples and propose a positive-none-class mixup method. Extensive experiments show that P3M improves the F1 score by approximately 4-10 points in document-level RE with incomplete labeling, and achieves state-of-the-art results in fully labeled scenarios. Furthermore, P3M has also demonstrated robustness to prior estimation bias in incomplete labeled scenarios.

----

## [2140] Knowledge Graph Prompting for Multi-Document Question Answering

**Authors**: *Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa F. Siu, Ruiyi Zhang, Tyler Derr*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29889](https://doi.org/10.1609/aaai.v38i17.29889)

**Abstract**:

The `pre-train, prompt, predict' paradigm of large language models (LLMs) has achieved remarkable success in open-domain question answering (OD-QA). However, few works explore this paradigm in multi-document question answering (MD-QA), a task demanding a thorough understanding of the logical associations among the contents and structures of documents. To fill this crucial gap, we propose a Knowledge Graph Prompting (KGP) method to formulate the right context in prompting LLMs for MD-QA, which consists of a graph construction module and a graph traversal module. For graph construction, we create a knowledge graph (KG) over multiple documents with nodes symbolizing passages or document structures (e.g., pages/tables), and edges denoting the semantic/lexical similarity between passages or document structural relations. For graph traversal, we design an LLM-based graph traversal agent that navigates across nodes and gathers supporting passages assisting LLMs in MD-QA. The constructed graph serves as the global ruler that regulates the transitional space among passages and reduces retrieval latency. Concurrently, the graph traversal agent acts as a local navigator that gathers pertinent context to progressively approach the question and guarantee retrieval quality. Extensive experiments underscore the efficacy of KGP for MD-QA, signifying the potential of leveraging graphs in enhancing the prompt design and retrieval augmented generation for LLMs. Our code: https://github.com/YuWVandy/KG-LLM-MDQA.

----

## [2141] STAIR: Spatial-Temporal Reasoning with Auditable Intermediate Results for Video Question Answering

**Authors**: *Yueqian Wang, Yuxuan Wang, Kai Chen, Dongyan Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29890](https://doi.org/10.1609/aaai.v38i17.29890)

**Abstract**:

Recently we have witnessed the rapid development of video question answering models. However, most models can only handle simple videos in terms of temporal reasoning, and their performance tends to drop when answering temporal-reasoning questions on long and informative videos. 
To tackle this problem we propose STAIR, a Spatial-Temporal Reasoning model with Auditable Intermediate Results for video question answering. STAIR is a neural module network, which contains a program generator to decompose a given question into a hierarchical combination of several sub-tasks, and a set of lightweight neural modules to complete each of these sub-tasks.
Though neural module networks are already widely studied on image-text tasks, applying them to videos is a non-trivial task, as reasoning on videos requires different abilities. In this paper, we define a set of basic video-text sub-tasks for video question answering and design a set of lightweight modules to complete them.
Different from most prior works, modules of STAIR return intermediate outputs specific to their intentions instead of always returning attention maps, which makes it easier to interpret and collaborate with pre-trained models. We also introduce intermediate supervision to make these intermediate outputs more accurate. We conduct extensive experiments on several video question answering datasets under various settings to show STAIR's performance, explainability, compatibility with pre-trained models, and applicability when program annotations are not available.  Code: https://github.com/yellow-binary-tree/STAIR

----

## [2142] Video Event Extraction with Multi-View Interaction Knowledge Distillation

**Authors**: *Kaiwen Wei, Runyan Du, Li Jin, Jian Liu, Jianhua Yin, Linhao Zhang, Jintao Liu, Nayu Liu, Jingyuan Zhang, Zhi Guo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29891](https://doi.org/10.1609/aaai.v38i17.29891)

**Abstract**:

Video event extraction (VEE) aims to extract key events and generate the event arguments for their semantic roles from the video. Despite promising results have been achieved by existing methods, they still lack an elaborate learning strategy to adequately consider: (1) inter-object interaction, which reflects the relation between objects; (2) inter-modality interaction, which aligns the features from text and video modality. In this paper, we propose a Multi-view Interaction with knowledge Distillation (MID) framework to solve the above problems with the Knowledge Distillation (KD) mechanism. Specifically, we propose the self-Relational KD (self-RKD) to enhance the inter-object interaction, where the relation between objects is measured by distance metric, and the high-level relational knowledge from the deeper layer is taken as the guidance for boosting the shallow layer in the video encoder. Meanwhile, to improve the inter-modality interaction, the Layer-to-layer KD (LKD) is proposed, which integrates additional cross-modal supervisions (i.e., the results of cross-attention) with the textual supervising signal for training each transformer decoder layer. Extensive experiments show that without any additional parameters, MID achieves the state-of-the-art performance compared to other strong methods in VEE.

----

## [2143] ConsistNER: Towards Instructive NER Demonstrations for LLMs with the Consistency of Ontology and Context

**Authors**: *Chenxiao Wu, Wenjun Ke, Peng Wang, Zhizhao Luo, Guozheng Li, Wanyi Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29892](https://doi.org/10.1609/aaai.v38i17.29892)

**Abstract**:

Named entity recognition (NER) aims to identify and classify specific entities mentioned in textual sentences. Most existing superior NER models employ the standard fully supervised paradigm, which requires a large amount of annotated data during training. In order to maintain performance with insufficient annotation resources (i.e., low resources), in-context learning (ICL) has drawn a lot of attention, due to its plug-and-play nature compared to other methods (e.g., meta-learning and prompt learning). In this manner, how to retrieve high-correlated demonstrations for target sentences serves as the key to emerging ICL ability. For the NER task, the correlation implies the consistency of both ontology (i.e., generalized entity type) and context (i.e., sentence semantic), which is ignored by previous NER demonstration retrieval techniques. To address this issue, we propose ConsistNER, a novel three-stage framework that incorporates ontological and contextual information for low-resource NER. Firstly, ConsistNER employs large language models (LLMs) to pre-recognize potential entities in a zero-shot manner. Secondly, ConsistNER retrieves the sentence-specific demonstrations for each target sentence based on the two following considerations: (1) Regarding ontological consistency, demonstrations are filtered into a candidate set based on ontology distribution. (2) Regarding contextual consistency, an entity-aware self-attention mechanism is introduced to focus more on the potential entities and semantic-correlated tokens. Finally, ConsistNER feeds the retrieved demonstrations for all target sentences into LLMs for prediction. We conduct experiments on four widely-adopted NER datasets, including both general and specific domains. Experimental results show that ConsistNER achieves a 6.01%-26.37% and 3.07%-21.18% improvement over the state-of-the-art baselines on Micro-F1 scores under 1- and 5-shot settings, respectively.

----

## [2144] Mitigating Idiom Inconsistency: A Multi-Semantic Contrastive Learning Method for Chinese Idiom Reading Comprehension

**Authors**: *Mingmin Wu, Yuxue Hu, Yongcheng Zhang, Zeng Zhi, Guixin Su, Ying Sha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29893](https://doi.org/10.1609/aaai.v38i17.29893)

**Abstract**:

Chinese idioms pose a significant challenge for machine reading comprehension due to their metaphorical meanings often diverging from their literal counterparts, leading to metaphorical inconsistency. Furthermore, the same idiom can have different meanings in different contexts, resulting in contextual inconsistency. Although deep learning-based methods have achieved some success in idioms reading comprehension, existing approaches still struggle to accurately capture idiom representations due to metaphorical inconsistency and contextual inconsistency of idioms. To address these challenges, we propose a novel model, Multi-Semantic Contrastive Learning Method (MSCLM), which simultaneously addresses metaphorical inconsistency and contextual inconsistency of idioms. To mitigate metaphorical inconsistency, we propose a metaphor contrastive learning module based on the prompt method, bridging the semantic gap between literal and metaphorical meanings of idioms. To mitigate contextual inconsistency, we propose a multi-semantic cross-attention module to explore semantic features between different metaphors of the same idiom in various contexts. Our model has been compared with multiple current latest models (including GPT-3.5) on multiple Chinese idiom reading comprehension datasets, and the experimental results demonstrate that MSCLM outperforms state-of-the-art models.

----

## [2145] Improving Open-Domain Dialogue Response Generation with Multi-Source Multilingual Commonsense Knowledge

**Authors**: *Sixing Wu, Jiong Yu, Jiahao Chen, Xiaofan Deng, Wei Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29894](https://doi.org/10.1609/aaai.v38i17.29894)

**Abstract**:

Knowledge-grounded Dialogue Response Generation (KRG) can facilitate informative and fidelity dialogues using external knowledge. Prior monolingual works can only use the knowledge of the corresponding native language. Thus, due to the prohibitive costs of collecting and constructing external knowledge bases, the limited scale of accessible external knowledge always constrains the ability of KRG, especially in low-resource language scenarios.  To this end, we propose a new task, Multi-Source Multilingual Knowledge-Grounded Response Generation (MMKRG), which simultaneously uses multiple knowledge sources of different languages. We notice that simply combining knowledge of different languages is inefficient due to the Cross-Conflict  issue and Cross-Repetition issue. Thus, we propose a novel approach MMK-BART, which uses a simple but elegant Estimate-Cluster-Penalize mechanism to overcome the mentioned issues and adopts the multilingual language model mBART as the backbone. Meanwhile, based on the recent multilingual corpus XDailyDialog, we propose an MMKRG dataset MMK-DailyDialog, which has been aligned to the large-scale multilingual commonsense knowledge base ConceptNet and supports four languages (English, Chinese, German, and Italian). Extensive experiments have verified the effectiveness of our dataset and approach in monolingual, cross-lingual, and multilingual scenarios.

----

## [2146] On the Affinity, Rationality, and Diversity of Hierarchical Topic Modeling

**Authors**: *Xiaobao Wu, Fengjun Pan, Thong Nguyen, Yichao Feng, Chaoqun Liu, Cong-Duy Nguyen, Anh Tuan Luu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29895](https://doi.org/10.1609/aaai.v38i17.29895)

**Abstract**:

Hierarchical topic modeling aims to discover latent topics from a corpus and organize them into a hierarchy to understand documents with desirable semantic granularity. However, existing work struggles with producing topic hierarchies of low affinity, rationality, and diversity, which hampers document understanding. To overcome these challenges, we in this paper propose Transport Plan and Context-aware Hierarchical Topic Model (TraCo). Instead of early simple topic dependencies, we propose a transport plan dependency method. It constrains dependencies to ensure their sparsity and balance, and also regularizes topic hierarchy building with them. This improves affinity and diversity of hierarchies. We further propose a context-aware disentangled decoder. Rather than previously entangled decoding, it distributes different semantic granularity to topics at different levels by disentangled decoding. This facilitates the rationality of hierarchies. Experiments on benchmark datasets demonstrate that our method surpasses state-of-the-art baselines, effectively improving the affinity, rationality, and diversity of hierarchical topic modeling with better performance on downstream tasks.

----

## [2147] MindMap: Constructing Evidence Chains for Multi-Step Reasoning in Large Language Models

**Authors**: *Yangyu Wu, Xu Han, Wei Song, Miaomiao Cheng, Fei Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29896](https://doi.org/10.1609/aaai.v38i17.29896)

**Abstract**:

Large language models (LLMs) have demonstrated remarkable performance in various natural language processing tasks. However, they still face significant challenges in automated reasoning, particularly in scenarios involving multi-step reasoning. In this paper, we focus on the logical reasoning problem. The main task is to answer a question based on a set of available facts and rules. A lot of work has focused on guiding LLMs to think logically by generating reasoning paths, ignoring the structure among available facts. In this paper, we propose a simple approach MindMap by introducing evidence chains for supporting reasoning. An evidence chain refers to a set of facts that involve the same subject. In this way, we can organize related facts together to avoid missing important information. MindMap can be integrated with existing reasoning framework, such as Chain-of-Thought (CoT) and Selection-Inference (SI), by letting the model select relevant evidence chains instead of independent facts. The experimental results on the bAbI and ProofWriterOWA datasets demonstrate the effectiveness of MindMap.It can significantly improve CoT and SI, especially in multi-step reasoning tasks.

----

## [2148] De-biased Attention Supervision for Text Classification with Causality

**Authors**: *Yiquan Wu, Yifei Liu, Ziyu Zhao, Weiming Lu, Yating Zhang, Changlong Sun, Fei Wu, Kun Kuang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29897](https://doi.org/10.1609/aaai.v38i17.29897)

**Abstract**:

In text classification models, while the unsupervised attention mechanism can enhance performance, it often produces attention distributions that are puzzling to humans, such as assigning high weight to seemingly insignificant conjunctions. Recently, numerous studies have explored Attention Supervision (AS) to guide the model toward more interpretable attention distributions. However, such AS can impact classification performance, especially in specialized domains. In this paper, we address this issue from a causality perspective. Firstly, we leverage the causal graph to reveal two biases in the AS: 1) Bias caused by the label distribution of the dataset. 2) Bias caused by the words' different occurrence ranges that some words can occur across labels while others only occur in a particular label. We then propose a novel De-biased Attention Supervision (DAS) method to eliminate these biases with causal techniques. Specifically, we adopt backdoor adjustment on the label-caused bias and reduce the word-caused bias by subtracting the direct causal effect of the word. Through extensive experiments on two professional text classification datasets (e.g., medicine and law), we demonstrate that our method achieves improved classification accuracy along with more coherent attention distributions.

----

## [2149] Get an A in Math: Progressive Rectification Prompting

**Authors**: *Zhenyu Wu, Meng Jiang, Chao Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29898](https://doi.org/10.1609/aaai.v38i17.29898)

**Abstract**:

Chain-of-Thought (CoT) prompting methods have enabled large language models (LLMs) to generate reasoning paths and solve math word problems (MWPs). However, they are sensitive to mistakes in the paths, as any mistake can result in an incorrect answer. We propose a novel method named Progressive Rectification Prompting (PRP) to improve average accuracy on eight MWP datasets from 77.3 to 90.5. Given an initial answer from CoT, PRP iterates a verify-then-rectify process to progressively identify incorrect answers and rectify the reasoning paths. With the most likely correct answer, the LLM predicts a masked numerical value in the question; if the prediction does not match the masked value, the answer is likely incorrect. Then the LLM is prompted to re-generate the reasoning path hinted with a set of incorrect answers to prevent itself from repeating previous mistakes. PRP achieves the best performance compared against the CoT methods. Our implementation is made publicly available at https://wzy6642.github.io/prp.github.io/.

----

## [2150] DIUSum: Dynamic Image Utilization for Multimodal Summarization

**Authors**: *Min Xiao, Junnan Zhu, Feifei Zhai, Yu Zhou, Chengqing Zong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29899](https://doi.org/10.1609/aaai.v38i17.29899)

**Abstract**:

Existing multimodal summarization approaches focus on fusing image features in the encoding process, ignoring the individualized needs for images when generating different summaries. However, whether intuitively or empirically, not all images can improve summary quality. Therefore, we propose a novel Dynamic Image Utilization framework for multimodal Summarization (DIUSum) to select and utilize valuable images for summarization. First, to predict whether an image helps produce a high-quality summary, we propose an image selector to score the usefulness of each image. Second, to dynamically utilize the multimodal information, we incorporate the hard and soft guidance from the image selector. Under the guidance, the image information is plugged into the decoder to generate a summary. Experimental results have shown that DIUSum outperforms multiple strong baselines and achieves SOTA on two public multimodal summarization datasets. Further analysis demonstrates that the image selector can reflect the improved level of summary quality brought by the images.

----

## [2151] Automated Defect Report Generation for Enhanced Industrial Quality Control

**Authors**: *Jiayuan Xie, Zhiping Zhou, Zihan Wu, Xinting Zhang, Jiexin Wang, Yi Cai, Qing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29900](https://doi.org/10.1609/aaai.v38i17.29900)

**Abstract**:

Defect detection is a pivotal aspect ensuring product quality and production efficiency in industrial manufacturing. Existing studies on defect detection predominantly focus on locating defects through bounding boxes and classifying defect types. However, their methods can only provide limited information and fail to meet the requirements for further processing after detecting defects. To this end, we propose a novel task called defect detection report generation, which aims to provide more comprehensive and informative insights into detected defects in the form of text reports. For this task, we propose some new datasets, which contain 16 different materials and each defect contains a detailed report of human constructs. In addition, we propose a knowledge-aware report generation model as a baseline for future research, which aims to incorporate additional knowledge to generate detailed analysis and subsequent processing related to defect in images. By constructing defect report datasets and proposing corresponding baselines, we chart new directions for future research and practical applications of this task.

----

## [2152] ALISON: Fast and Effective Stylometric Authorship Obfuscation

**Authors**: *Eric Xing, Saranya Venkatraman, Thai Le, Dongwon Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29901](https://doi.org/10.1609/aaai.v38i17.29901)

**Abstract**:

Authorship Attribution (AA) and Authorship Obfuscation (AO) are two competing tasks of increasing importance in privacy research. Modern AA leverages an author's consistent writing style to match a text to its author using an AA classifier. AO is the corresponding adversarial task, aiming to modify a text in such a way that its semantics are preserved, yet an AA model cannot correctly infer its authorship. To address privacy concerns raised by state-of-the-art (SOTA) AA methods,
new AO methods have been proposed but remain largely impractical to use due to their prohibitively slow training and obfuscation speed, often taking hours.
To this challenge, we propose a practical AO method, ALISON, that (1) dramatically reduces training/obfuscation time, demonstrating more than 10x faster obfuscation than SOTA AO methods, (2) achieves better obfuscation success through attacking three transformer-based AA methods on two benchmark datasets, typically performing 15% better than competing methods, (3) does not require direct signals from a target AA classifier during obfuscation, and (4) utilizes unique stylometric features,  allowing sound model interpretation for explainable obfuscation. We also demonstrate that ALISON can effectively prevent four SOTA AA methods from accurately determining the authorship of ChatGPT-generated texts, all while minimally changing the original text semantics. To ensure the reproducibility of our findings, our code and data are available at: https://github.com/EricX003/ALISON.

----

## [2153] SECap: Speech Emotion Captioning with Large Language Model

**Authors**: *Yaoxun Xu, Hangting Chen, Jianwei Yu, Qiaochu Huang, Zhiyong Wu, Shi-Xiong Zhang, Guangzhi Li, Yi Luo, Rongzhi Gu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29902](https://doi.org/10.1609/aaai.v38i17.29902)

**Abstract**:

Speech emotions are crucial in human communication and are extensively used in fields like speech synthesis and natural language understanding. Most prior studies, such as speech emotion recognition, have categorized speech emotions into a fixed set of classes.  Yet, emotions expressed in human speech are often complex, and categorizing them into predefined groups can be insufficient to adequately represent speech emotions. On the contrary, describing speech emotions directly by means of natural language may be a more effective approach. Regrettably, there are not many studies available that have focused on this direction. Therefore, this paper proposes a speech emotion captioning framework named SECap, aiming at effectively describing speech emotions using natural language. Owing to the impressive capabilities of large language models in language comprehension and text generation, SECap employs LLaMA as the text decoder to allow the production of coherent speech emotion captions. In addition, SECap leverages HuBERT as the audio encoder to extract general speech features and Q-Former as the Bridge-Net to provide LLaMA with emotion-related speech features. To accomplish this, Q-Former utilizes mutual information learning to disentangle emotion-related speech features and speech contents, while implementing contrastive learning to extract more emotion-related speech features. The results of objective and subjective evaluations demonstrate that: 1) the SECap framework outperforms the HTSAT-BART baseline in all objective evaluations; 2) SECap can generate high-quality speech emotion captions that attain performance on par with human annotators in subjective mean opinion score tests.

----

## [2154] Question Calibration and Multi-Hop Modeling for Temporal Question Answering

**Authors**: *Chao Xue, Di Liang, Pengfei Wang, Jing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29903](https://doi.org/10.1609/aaai.v38i17.29903)

**Abstract**:

Many models that leverage knowledge graphs (KGs) have recently demonstrated remarkable success in question answering (QA) tasks. In the real world, many facts contained in KGs are time-constrained thus temporal KGQA has received increasing attention. Despite the fruitful efforts of previous models in temporal KGQA, they still have several limitations. (I) They adopt  pre-trained language models (PLMs) to obtain question representations, while PLMs tend to focus on entity information and ignore entity transfer caused by temporal constraints, and finally fail to learn specific temporal representations of entities. (II) They neither emphasize the graph structure between entities nor explicitly model the multi-hop relationship in the graph, which will make it difficult to solve complex multi-hop question answering. To alleviate this problem, we propose a novel Question Calibration and Multi-Hop Modeling (QC-MHM) network. Specifically, We first calibrate the question representation by fusing the question and the time-constrained concepts in KG. Then, we construct the GNN layer to complete multi-hop message passing. Finally, the question representation is combined with the embedding output by the GNN to generate the final prediction. Empirical results verify that the proposed model achieves better performance than the state-of-the-art models in the benchmark dataset. Notably, the Hits@1 and Hits@10 results of QC-MHM on the CronQuestions dataset's complex questions are absolutely improved by 5.1% and 1.2% compared to the best-performing baseline.  Moreover, QC-MHM can generate interpretable and trustworthy predictions.

----

## [2155] Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification

**Authors**: *Xiaojun Xue, Chunxia Zhang, Tianxiang Xu, Zhendong Niu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29904](https://doi.org/10.1609/aaai.v38i17.29904)

**Abstract**:

Few-shot named entity recognition (NER) aims to recognize novel named entities in low-resource domains utilizing existing knowledge. However, the present few-shot NER models assume that the labeled data are all clean without noise or outliers, and there are few works focusing on the robustness of the cross-domain transfer learning ability to textual adversarial attacks in Few-shot NER. In this work, we comprehensively explore and assess the robustness of few-shot NER models under textual adversarial attack scenario, and found the vulnerability of existing few-shot NER models. Furthermore, we propose a robust two-stage few-shot NER method with Boundary Discrimination and Correlation Purification (BDCP). Specifically, in the span detection stage, the entity boundary discriminative module is introduced to provide a highly distinguishing boundary representation space to detect entity spans. In the entity typing stage, the correlations between entities and contexts are purified by minimizing the interference information and facilitating correlation generalization to alleviate the perturbations caused by textual adversarial attacks. In addition, we construct adversarial examples for few-shot NER based on public datasets Few-NERD and Cross-Dataset. Comprehensive evaluations on those two groups of few-shot NER datasets containing adversarial examples demonstrate the robustness and superiority of the proposed method.

----

## [2156] Tackling Vision Language Tasks through Learning Inner Monologues

**Authors**: *Diji Yang, Kezhen Chen, Jinmeng Rao, Xiaoyuan Guo, Yawen Zhang, Jie Yang, Yi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29905](https://doi.org/10.1609/aaai.v38i17.29905)

**Abstract**:

Visual language tasks such as Visual Question Answering (VQA) or Visual Entailment (VE) require AI models to comprehend and reason with both visual and textual content. Driven by the power of Large Language Models (LLMs), two prominent methods have emerged: (1) the hybrid integration between LLMs and Vision-Language Models (VLMs), where visual inputs are firstly converted into language descriptions by VLMs, serving as inputs for LLMs to generate final answer(s); (2) visual feature alignment in language space, where visual inputs are encoded as embeddings and projected to LLMs' language space via further supervised fine-tuning. The first approach provides light training costs and interpretability but is hard to be optimized in an end-to-end fashion. The second approach presents decent performance, but feature alignment usually requires large amounts of training data and lacks interpretability. 
To tackle this dilemma, we propose a novel approach, Inner Monologue Multi-Modal Optimization (IMMO), to solve complex vision language problems by simulating Inner Monologue, a cognitive process in which an individual engages in silent verbal communication with themselves. More specifically, we enable LLMs and VLMs to interact through natural language conversation (i.e., Inner Monologue) and propose to use a two-stage training process to learn how to do Inner Monologue (self-asking questions and answering questions). IMMO is evaluated on two popular tasks and achieves competitive performance with less training data when compared with state-of-the-art models while concurrently keeping the interpretability. The results suggest that by emulating the cognitive phenomenon of internal dialogue, our approach can enhance reasoning and explanation abilities, contributing to the more effective fusion of vision and language models. More importantly, instead of using predefined human-crafted monologues, IMMO learns this process within the deep learning models, broadening its potential applications across various AI challenges beyond vision and language tasks.

----

## [2157] YTCommentQA: Video Question Answerability in Instructional Videos

**Authors**: *Saelyne Yang, Sunghyun Park, Yunseok Jang, Moontae Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29906](https://doi.org/10.1609/aaai.v38i17.29906)

**Abstract**:

Instructional videos provide detailed how-to guides for various tasks, with viewers often posing questions regarding the content. Addressing these questions is vital for comprehending the content, yet receiving immediate answers is difficult. While numerous computational models have been developed for Video Question Answering (Video QA) tasks, they are primarily trained on questions generated based on video content, aiming to produce answers from within the content. However, in real-world situations, users may pose questions that go beyond the video's informational boundaries, highlighting the necessity to determine if a video can provide the answer. Discerning whether a question can be answered by video content is challenging due to the multi-modal nature of videos, where visual and verbal information are intertwined. To bridge this gap, we present the YTCommentQA dataset, which contains naturally-generated questions from YouTube, categorized by their answerability and required modality to answer -- visual, script, or both. Experiments with answerability classification tasks demonstrate the complexity of YTCommentQA and emphasize the need to comprehend the combined role of visual and script information in video reasoning. The dataset is available at https://github.com/lgresearch/YTCommentQA.

----

## [2158] Zhongjing: Enhancing the Chinese Medical Capabilities of Large Language Model through Expert Feedback and Real-World Multi-Turn Dialogue

**Authors**: *Songhua Yang, Hanjie Zhao, Senbin Zhu, Guangyu Zhou, Hongfei Xu, Yuxiang Jia, Hongying Zan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29907](https://doi.org/10.1609/aaai.v38i17.29907)

**Abstract**:

Recent advances in Large Language Models (LLMs) have achieved remarkable breakthroughs in understanding and responding to user intents. However, their performance lag behind general use cases in some expertise domains, such as Chinese medicine. Existing efforts to incorporate Chinese medicine into LLMs rely on Supervised Fine-Tuning (SFT) with single-turn and distilled dialogue data. These models lack the ability for doctor-like proactive inquiry and multi-turn comprehension and cannot align responses with experts' intentions. In this work, we introduce Zhongjing, the first Chinese medical LLaMA-based LLM that implements an entire training pipeline from continuous pre-training, SFT, to Reinforcement Learning from Human Feedback (RLHF). Additionally, we construct a Chinese multi-turn medical dialogue dataset of 70,000 authentic doctor-patient dialogues, CMtMedQA, which significantly enhances the model's capability for complex dialogue and proactive inquiry initiation. We also define a refined annotation rule and evaluation criteria given the unique characteristics of the biomedical domain. Extensive experimental results show that Zhongjing outperforms baselines in various capacities and matches the performance of ChatGPT in some abilities, despite the 100x parameters. Ablation studies also demonstrate the contributions of each component: pre-training enhances medical knowledge, and RLHF further improves instruction-following ability and safety. Our code, datasets, and models are available at https://github.com/SupritYoung/Zhongjing.

----

## [2159] Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation

**Authors**: *Zhewei Yao, Xiaoxia Wu, Cheng Li, Stephen Youn, Yuxiong He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29908](https://doi.org/10.1609/aaai.v38i17.29908)

**Abstract**:

Post-training quantization (PTQ) has emerged as a promising technique for mitigating memory consumption and computational costs in large language models (LLMs). However, a systematic examination of various quantization schemes, model families, and quantization bit precision has been absent from the literature. In this paper, we conduct a comprehensive analysis of these factors by investigating the effects of PTQ on weight-only, activation-only, and weight-and-activation quantization using diverse methods such as round-to-nearest (RTN), GPTQ, ZeroQuant, and their variants. We apply these methods to two distinct model families with parameters ranging from 125M to 176B. Our contributions include: (1) a sensitivity analysis revealing that activation quantization is generally more susceptible to weight quantization, with smaller models often outperforming larger models in terms of activation quantization; (2) an evaluation and comparison of existing PTQ methods to optimize model size reduction while minimizing the impact on accuracy, revealing that none of the current methods can achieve the original model quality for quantization with either INT4-weight or INT4-weight-and-INT8-activation; (3) based on these insights, we propose an optimized method called Low-Rank Compensation (LoRC), which employs low-rank matrices to enhance model quality recovery with a minimal increase in model size.

----

## [2160] Investigating the Effectiveness of Task-Agnostic Prefix Prompt for Instruction Following

**Authors**: *Seonghyeon Ye, Hyeonbin Hwang, Sohee Yang, Hyeongu Yun, Yireun Kim, Minjoon Seo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29909](https://doi.org/10.1609/aaai.v38i17.29909)

**Abstract**:

In this paper, we present our finding that prepending a Task-Agnostic Prefix Prompt (TAPP) to the input improves the instruction-following ability of various Large Language Models (LLMs) during inference. TAPP is different from canonical prompts for LLMs in that it is a fixed prompt prepended to the beginning of every input regardless of the target task for zero-shot generalization. We observe that both base LLMs (i.e. not fine-tuned to follow instructions) and instruction-tuned models benefit from TAPP, resulting in 34.58% and 12.26% improvement on average, respectively. This implies that the instruction-following ability of LLMs can be improved during inference time with a fixed prompt constructed with simple heuristics. We hypothesize that TAPP assists language models to better estimate the output distribution by focusing more on the instruction of the target task during inference. In other words, such ability does not seem to be sufficiently activated in not only base LLMs but also many instruction-fine-tuned LLMs.

----

## [2161] Uni-MIS: United Multiple Intent Spoken Language Understanding via Multi-View Intent-Slot Interaction

**Authors**: *Shangjian Yin, Peijie Huang, Yuhong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29910](https://doi.org/10.1609/aaai.v38i17.29910)

**Abstract**:

So far, multi-intent spoken language understanding (SLU) has become a research hotspot in the field of natural language processing (NLP) due to its ability to recognize and extract multiple intents expressed and annotate corresponding sequence slot tags within a single utterance. Previous research has primarily concentrated on the token-level intent-slot interaction to model joint intent detection and slot filling, which resulted in a failure to fully utilize anisotropic intent-guiding information during joint training. In this work, we present a novel architecture by modeling the multi-intent SLU as a multi-view intent-slot interaction. The architecture resolves the kernel bottleneck of unified multi-intent SLU by effectively modeling the intent-slot relations with utterance, chunk, and token-level interaction. We further develop a neural framework, namely Uni-MIS, in which the unified multi-intent SLU is modeled as a three-view intent-slot interaction fusion to better capture the interaction information after special encoding. A chunk-level intent detection decoder is used to sufficiently capture the multi-intent, and an adaptive intent-slot graph network is used to capture the fine-grained intent information to guide final slot filling. We perform extensive experiments on two widely used benchmark datasets for multi-intent SLU, where our model bets on all the current strong baselines, pushing the state-of-the-art performance of unified multi-intent SLU. Additionally, the ChatGPT benchmark that we have developed demonstrates that there is a considerable amount of potential research value in the field of multi-intent SLU.

----

## [2162] TextGT: A Double-View Graph Transformer on Text for Aspect-Based Sentiment Analysis

**Authors**: *Shuo Yin, Guoqiang Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29911](https://doi.org/10.1609/aaai.v38i17.29911)

**Abstract**:

Aspect-based sentiment analysis (ABSA) is aimed at predicting the sentiment polarities of the aspects included in a sentence instead of the whole sentence itself, and is a fine-grained learning task compared to the conventional text classification. In recent years, on account of the ability to model the connectivity relationships between the words in one sentence, graph neural networks have been more and more popular to handle the natural language processing tasks, and meanwhile many works emerge for the ABSA task. However, most of the works utilizing graph convolution easily incur the over-smoothing problem, while graph Transformer for ABSA has not been explored yet. In addition, although some previous works are dedicated to using both GNN and Transformer to handle text, the methods of tightly combining graph view and sequence view of text is open to research. To address the above issues, we propose a double-view graph Transformer on text (TextGT) for ABSA. In TextGT, the procedure in graph view of text is handled by GNN layers, while Transformer layers deal with the sequence view, and these two processes are tightly coupled, alleviating the over-smoothing problem. Moreover, we propose an algorithm for implementing a kind of densely message passing graph convolution called TextGINConv, to employ edge features in graphs. Extensive experiments demonstrate the effectiveness of our TextGT over the state-of-the-art approaches, and validate the TextGINConv module. The source code is available at https://github.com/shuoyinn/TextGT.

----

## [2163] History Matters: Temporal Knowledge Editing in Large Language Model

**Authors**: *Xunjian Yin, Jin Jiang, Liming Yang, Xiaojun Wan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29912](https://doi.org/10.1609/aaai.v38i17.29912)

**Abstract**:

The imperative task of revising or updating the knowledge stored within large language models arises from two distinct sources: intrinsic errors inherent in the model which should be corrected and outdated knowledge due to external shifts in the real world which should be updated. Prevailing efforts in model editing conflate these two distinct categories of edits arising from distinct reasons and directly modify the original knowledge in models into new knowledge. However, we argue that preserving the model's original knowledge remains pertinent. Specifically, if a model's knowledge becomes outdated due to evolving worldly dynamics, it should retain recollection of the historical knowledge while integrating the newfound knowledge. In this work, we introduce the task of Temporal Knowledge Editing (TKE) and establish a benchmark AToKe (Assessment of TempOral Knowledge Editing) to evaluate current model editing methods. We find that while existing model editing methods are effective at making models remember new knowledge, the edited model catastrophically forgets historical knowledge. To address this gap, we propose a simple and general framework termed Multi-Editing with Time Objective (METO) for enhancing existing editing models, which edits both historical and new knowledge concurrently and optimizes the model's prediction for the time of each fact. Our assessments demonstrate that while AToKe is still difficult, METO maintains the effectiveness of learning new knowledge and meanwhile substantially improves the performance of edited models on utilizing historical knowledge.

----

## [2164] Topic-VQ-VAE: Leveraging Latent Codebooks for Flexible Topic-Guided Document Generation

**Authors**: *Youngjoon Yoo, Jongwon Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29913](https://doi.org/10.1609/aaai.v38i17.29913)

**Abstract**:

This paper introduces a novel approach for topic modeling utilizing latent codebooks from Vector-Quantized Variational Auto-Encoder~(VQ-VAE), discretely encapsulating the rich information of the pre-trained embeddings such as the pre-trained language model.
From the novel interpretation of the latent codebooks and embeddings as conceptual bag-of-words, we propose a new generative topic model called Topic-VQ-VAE~(TVQ-VAE) which inversely generates the original documents related to the respective latent codebook.
The TVQ-VAE can visualize the topics with various generative distributions including the traditional BoW distribution and the autoregressive image generation.
Our experimental results on document analysis and image generation demonstrate that TVQ-VAE effectively captures the topic context which reveals the underlying structures of the dataset and supports flexible forms of document generation.
Official implementation of the proposed TVQ-VAE is available at https://github.com/clovaai/TVQ-VAE.

----

## [2165] CK12: A Rounded K12 Knowledge Graph Based Benchmark for Chinese Holistic Cognition Evaluation

**Authors**: *Weihao You, Pengcheng Wang, Changlong Li, Zhilong Ji, Jinfeng Bai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29914](https://doi.org/10.1609/aaai.v38i17.29914)

**Abstract**:

New NLP benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present a meticulously designed evaluation benchmark that leverages the knowledge graph. This evaluation comprises 584 level-1 knowledge points and 1,989 level-2 knowledge points, thereby encompassing a comprehensive spectrum of the K12 education domain knowledge. The primary objective is to comprehensively assess the high-level comprehension aptitude and reasoning capabilities of LLMs operating within the Chinese context. Our evaluation incorporates five distinct question types with 39,452 questions. We test the current mainstream LLMs by three distinct modes. Firstly, four prompt evaluation modes were employed to assess the fundamental capacity. Additionally, for choice questions, a result-oriented evaluation approach was designed through data augmentation to assess the model's proficiency in advanced knowledge and reasoning. Moreover, a subset with reasoning process is derived, and the process-oriented testing method is used to test the model's interpretability and higher-order reasoning capacity. We further show models' capability in our knowledge points, and anticipate the evaluation can assist in the assessment of the strengths and deficiencies of LLMs on knowledge points, thus fostering their development within the Chinese context. Our Dataset will be publicly available in https://github.com/tal-tech/chinese-k12-evaluation.

----

## [2166] Reliable Data Generation and Selection for Low-Resource Relation Extraction

**Authors**: *Junjie Yu, Xing Wang, Wenliang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29915](https://doi.org/10.1609/aaai.v38i17.29915)

**Abstract**:

Automated construction of annotated data holds significant importance in Relation Extraction (RE) tasks due to the hardness and cost of human annotation. In this work, we propose Self-RDGS, a method for Self-supervised Reliable Data Generation and Selection in low-resource RE tasks. At first, we fully utilize the knowledge of triplets as prompts to generate sentences by employing the Large Language Models (LLMs). Since the auto-generated data contains noise, we then propose a ranking-based data selection method to select reliable sentences. Finally, we integrate the data selection and RE model training within a self-supervised iterative framework. Through experimentation on three datasets with low-resource settings, we demonstrate the effectiveness of our proposed approach in constructing annotated data and achieving noteworthy improvements in comparison to multiple baselines. Code, data and models are available at https://github.com/jjyunlp/GenerationRE.

----

## [2167] MELO: Enhancing Model Editing with Neuron-Indexed Dynamic LoRA

**Authors**: *Lang Yu, Qin Chen, Jie Zhou, Liang He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29916](https://doi.org/10.1609/aaai.v38i17.29916)

**Abstract**:

Large language models (LLMs) have shown great success in various Natural Language Processing (NLP) tasks, whist they still need updates after deployment to fix errors or keep pace with the changing knowledge in the world. Researchers formulate such problem as Model Editing and have developed various editors focusing on different axes of editing properties. However, current editors can hardly support all properties and rely on heavy computational resources. In this paper, we propose a plug-in Model Editing method based on neuron-indexed dynamic LoRA (MELO), which alters the behavior of language models by dynamically activating certain LoRA blocks according to the index built in an inner vector database. Our method satisfies various editing properties with high efficiency and can be easily integrated into multiple LLM backbones. Experimental results show that our proposed MELO achieves state-of-the-art editing performance on three sequential editing tasks (document classification, question answering and hallucination correction), while requires the least trainable parameters and computational cost.

----

## [2168] SeqGPT: An Out-of-the-Box Large Language Model for Open Domain Sequence Understanding

**Authors**: *Tianyu Yu, Chengyue Jiang, Chao Lou, Shen Huang, Xiaobin Wang, Wei Liu, Jiong Cai, Yangning Li, Yinghui Li, Kewei Tu, Hai-Tao Zheng, Ningyu Zhang, Pengjun Xie, Fei Huang, Yong Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29917](https://doi.org/10.1609/aaai.v38i17.29917)

**Abstract**:

Large language models (LLMs) have shown impressive abilities for open-domain NLP tasks. However, LLMs are sometimes too footloose for natural language understanding (NLU) tasks which always have restricted output and input format. Their performances on NLU tasks are highly related to prompts or demonstrations and are shown to be poor at performing several representative NLU tasks, such as event extraction and entity typing. To this end, we present SeqGPT, a bilingual (i.e., English and Chinese) open-source autoregressive model specially enhanced for open-domain natural language understanding. We express all NLU tasks with two atomic tasks, which define fixed instructions to restrict the input and output format but still ``open'' for arbitrarily varied label sets. The model is first instruction-tuned with extremely fine-grained labeled data synthesized by ChatGPT and then further fine-tuned by 233 different atomic tasks from 152 datasets across various domains. The experimental results show that SeqGPT has decent classification and extraction ability, and is capable of performing language understanding tasks on unseen domains. We also conduct empirical studies on the scaling of data and model size as well as on the transfer across tasks. Our models are accessible at https://github.com/Alibaba-NLP/SeqGPT.

----

## [2169] TaskLAMA: Probing the Complex Task Understanding of Language Models

**Authors**: *Quan Yuan, Mehran Kazemi, Xin Xu, Isaac Noble, Vaiva Imbrasaite, Deepak Ramachandran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29918](https://doi.org/10.1609/aaai.v38i17.29918)

**Abstract**:

Structured  Complex  Task  Decomposition  (SCTD)  is  the problem of breaking down a complex real-world task (such as planning a wedding) into a directed acyclic graph over individual steps that contribute to achieving the task, with edges specifying temporal dependencies between steps. SCTD is an important component of assistive planning tools, and a challenge  for  commonsense  reasoning  systems.  We  probe  how accurately SCTD can be done with the knowledge extracted from  pre-trained  Large  Language  Models  (LLMs).  We  introduce a new high-quality human-annotated dataset for this problem and novel metrics to fairly assess  performance of LLMs against several baselines. Our experiments reveal that LLMs  are able to decompose complex tasks into individual steps effectively, with a relative improvement of 15% to 280% over the best baseline. We also propose a number of approaches to further improve their performance, with a relative improvement of 7% to 37%. However, we find that LLMs still struggle to predict pairwise temporal dependencies, which reveals a gap in their understanding of complex tasks.

----

## [2170] An Autoregressive Text-to-Graph Framework for Joint Entity and Relation Extraction

**Authors**: *Urchade Zaratiana, Nadi Tomeh, Pierre Holat, Thierry Charnois*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29919](https://doi.org/10.1609/aaai.v38i17.29919)

**Abstract**:

In this paper, we propose a novel method for joint entity and relation extraction from unstructured text by framing it as a conditional sequence generation problem. In contrast to conventional generative information extraction models that are left-to-right token-level generators, our approach is \textit{span-based}. It generates a linearized graph where nodes represent text spans and edges represent relation triplets. Our method employs a transformer encoder-decoder architecture with pointing mechanism on a dynamic vocabulary of spans and relation types. Our model can capture the structural characteristics and boundaries of entities and relations through span representations while simultaneously grounding the generated output in the original text thanks to the pointing mechanism. Evaluation on benchmark datasets validates the effectiveness of our approach, demonstrating competitive results. Code is available at https://github.com/urchade/ATG.

----

## [2171] Teaching Large Language Models to Translate with Comparison

**Authors**: *Jiali Zeng, Fandong Meng, Yongjing Yin, Jie Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29920](https://doi.org/10.1609/aaai.v38i17.29920)

**Abstract**:

Open-sourced large language models (LLMs) have demonstrated remarkable efficacy in various tasks with instruction tuning. 
However, these models can sometimes struggle with tasks that require more specialized knowledge such as translation. 
One possible reason for such deficiency is that instruction tuning aims to generate fluent and coherent text that continues from a given instruction without being constrained by any task-specific requirements. 
Moreover, it can be more challenging to tune smaller LLMs with lower-quality training data.
To address this issue, we propose a novel framework using examples in comparison to teach LLMs to learn translation. 
Our approach involves output comparison and preference comparison, presenting the model with 
carefully designed examples of correct and incorrect translations and an additional preference loss for better regularization.
Empirical evaluation on four language directions of WMT2022 and FLORES-200 benchmarks shows the superiority of our proposed method over existing methods. 
Our findings offer a new perspective on fine-tuning LLMs for translation tasks and provide a promising solution for generating high-quality translations.
Please refer to Github for more details:
https://github.com/lemon0830/TIM.

----

## [2172] InterpretARA: Enhancing Hybrid Automatic Readability Assessment with Linguistic Feature Interpreter and Contrastive Learning

**Authors**: *Jinshan Zeng, Xianchao Tong, Xianglong Yu, Wenyan Xiao, Qing Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29921](https://doi.org/10.1609/aaai.v38i17.29921)

**Abstract**:

The hybrid automatic readability assessment (ARA) models that combine deep and linguistic features have recently received rising attention due to their impressive performance. However, the utilization of linguistic features is not fully realized, as ARA models frequently concentrate excessively on numerical values of these features, neglecting valuable structural information embedded within them. This leads to limited contribution of linguistic features in these hybrid ARA models, and in some cases, it may even result in counterproductive outcomes. In this paper, we propose a novel hybrid ARA model named InterpretARA through introducing a linguistic interpreter to better comprehend the structural information contained in linguistic features, and leveraging the contrastive learning that enables the model to understand relative difficulty relationships among texts and thus enhances deep representations. Both document-level and segment-level deep representations are extracted and used for the readability assessment. A series of experiments are conducted over four English corpora and one Chinese corpus to demonstrate the effectiveness of the proposed model. Experimental results show that InterpretARA outperforms state-of-the-art models in most corpora, and the introduced linguistic interpreter can provide more useful information than existing ways for ARA.

----

## [2173] ConsistentEE: A Consistent and Hardness-Guided Early Exiting Method for Accelerating Language Models Inference

**Authors**: *Ziqian Zeng, Yihuai Hong, Hongliang Dai, Huiping Zhuang, Cen Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29922](https://doi.org/10.1609/aaai.v38i17.29922)

**Abstract**:

Early Exiting is one of the most popular methods to achieve efficient inference. Current early exiting methods adopt the (weighted) sum of the cross entropy loss of all internal classifiers as the objective function during training, imposing all these classifiers to predict all instances correctly. However, during inference, as long as one internal classifier predicts an instance correctly, it can accelerate without losing accuracy. Thus, there is a notable gap between training and inference. We propose ConsistentEE, an early exiting method that is consistent in training and inference. ConsistentEE formulates the early exiting process as a reinforcement learning problem. A policy network is added to decide whether an instance should exit or continue. The training objective of ConsistentEE only requires each instance to be predicted correctly by one internal classifier. Additionally, we introduce the concept "Memorized Layer" to measure the hardness of an instance. We incorporate the memorized layer into reward function design, which allows "easy'' instances to focus more on acceleration while ``hard'' instances to focus more on accuracy. Experimental results show that our method outperforms other baselines on various natural language understanding and generation tasks using PLMs and LLMs as backbones respectively.

----

## [2174] A Comprehensive Analysis of the Effectiveness of Large Language Models as Automatic Dialogue Evaluators

**Authors**: *Chen Zhang, Luis Fernando D'Haro, Yiming Chen, Malu Zhang, Haizhou Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29923](https://doi.org/10.1609/aaai.v38i17.29923)

**Abstract**:

Automatic evaluation is an integral aspect of dialogue system research. The traditional reference-based NLG metrics are generally found to be unsuitable for dialogue assessment. Consequently, recent studies have suggested various unique, reference-free neural metrics that better align with human evaluations. Notably among them, large language models (LLMs), particularly the instruction-tuned variants like ChatGPT, are shown to be promising substitutes for human judges. Yet, existing works on utilizing LLMs for automatic dialogue evaluation are limited in their scope in terms of the number of meta-evaluation datasets, mode of evaluation, coverage of LLMs, etc. Hence, it remains inconclusive how effective these LLMs are. To this end, we conduct a comprehensive study on the application of LLMs for automatic dialogue evaluation. Specifically, we analyze the multi-dimensional evaluation capability of 30 recently emerged LLMs at both turn and dialogue levels, using a comprehensive set of 12 meta-evaluation datasets. Additionally, we probe the robustness of the LLMs in handling various adversarial perturbations at both turn and dialogue levels. Finally, we explore how model-level and dimension-level ensembles impact the evaluation performance. All resources are available at https://github.com/e0397123/comp-analysis.

----

## [2175] PREFER: Prompt Ensemble Learning via Feedback-Reflect-Refine

**Authors**: *Chenrui Zhang, Lin Liu, Chuyuan Wang, Xiao Sun, Hongyu Wang, Jinpeng Wang, Mingchen Cai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29924](https://doi.org/10.1609/aaai.v38i17.29924)

**Abstract**:

As an effective tool for eliciting the power of Large Language Models (LLMs), prompting has recently demonstrated unprecedented abilities across a variety of complex tasks. To further improve the performance, prompt ensemble has attracted substantial interest for tackling the hallucination and instability of LLMs. However, existing methods usually adopt a two-stage paradigm, which requires a pre-prepared set of prompts with substantial manual effort, and is unable to perform directed optimization for different weak learners. In this paper, we propose a simple, universal, and automatic method named PREFER (Prompt Ensemble learning via Feedback-Reflect-Refine) to address the stated limitations. Specifically, given the fact that weak learners are supposed to focus on hard examples during boosting, PREFER builds a feedback mechanism for reflecting on the inadequacies of existing weak learners. Based on this, the LLM is required to automatically synthesize new prompts for iterative refinement. Moreover, to enhance stability of the prompt effect evaluation, we propose a novel prompt bagging method involving forward and backward thinking, which is superior to majority voting and is beneficial for both feedback and weight calculation in boosting. Extensive experiments demonstrate that our PREFER achieves state-of-the-art performance in multiple types of tasks by a significant margin. We have made our code publicly available.

----

## [2176] Causal Walk: Debiasing Multi-Hop Fact Verification with Front-Door Adjustment

**Authors**: *Congzhi Zhang, Linhai Zhang, Deyu Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29925](https://doi.org/10.1609/aaai.v38i17.29925)

**Abstract**:

Multi-hop fact verification aims to detect the veracity of the given claim by integrating and reasoning over multiple pieces of evidence. Conventional multi-hop fact verification models are prone to rely on spurious correlations from the annotation artifacts, leading to an obvious performance decline on unbiased datasets. Among the various debiasing works, the causal inference-based methods become popular by performing theoretically guaranteed debiasing such as casual intervention or counterfactual reasoning. However, existing causal inference-based debiasing methods, which mainly formulate fact verification as a single-hop reasoning task to tackle shallow bias patterns, cannot deal with the complicated bias patterns hidden in multiple hops of evidence. To address the challenge, we propose Causal Walk, a novel method for debiasing multi-hop fact verification from a causal perspective with front-door adjustment. Specifically, in the structural causal model, the reasoning path between the treatment (the input claim-evidence graph) and the outcome (the veracity label) is introduced as the mediator to block the confounder. With the front-door adjustment, the causal effect between the treatment and the outcome is decomposed into the causal effect between the treatment and the mediator, which is estimated by applying the idea of random walk, and the causal effect between the mediator and the outcome, which is estimated with normalized weighted geometric mean approximation. To investigate the effectiveness of the proposed method, an adversarial multi-hop fact verification dataset and a symmetric multi-hop fact verification dataset are proposed with the help of the large language model. Experimental results show that Causal Walk outperforms some previous debiasing methods on both existing datasets and the newly constructed datasets. Code and data will be released at https://github.com/zcccccz/CausalWalk.

----

## [2177] Visual Hallucination Elevates Speech Recognition

**Authors**: *Fang Zhang, Yongxin Zhu, Xiangxiang Wang, Huang Chen, Xing Sun, Linli Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29926](https://doi.org/10.1609/aaai.v38i17.29926)

**Abstract**:

Due to the detrimental impact of noise on the conventional audio speech recognition (ASR) task, audio-visual speech recognition~(AVSR) has been proposed by incorporating both audio and visual video signals. Although existing methods have demonstrated that the aligned visual input of lip movements can enhance the robustness of AVSR systems against noise, the paired videos are not always 
available during inference, leading to the problem of 
the missing visual modality, which restricts their practicality in real-world scenarios. 

To tackle this problem, we propose a Discrete Feature based Visual Generative Model (DFVGM) which exploits semantic correspondences between the audio and visual modalities 
during training, generating 
visual hallucinations in lieu of
real videos during inference. To achieve that, the 
primary challenge is to generate the visual hallucination 
given the noisy audio while preserving semantic correspondences with the clean speech. To 
tackle this challenge, we 
start with training the audio encoder in the Audio-Only (AO) setting, which generates continuous semantic features closely associated with the linguistic information. Simultaneously, the visual encoder is trained in the Visual-Only (VO) setting, producing visual features that are phonetically related. Next, we employ K-means to 
discretize the continuous audio and visual feature spaces. The discretization step 
allows DFVGM to capture high-level semantic structures that are more resilient to noise and generate 
visual hallucinations with high quality. 
To evaluate the effectiveness and robustness of our approach, we conduct extensive experiments on two publicly available datasets. The results demonstrate that our method achieves a remarkable 53% relative reduction (30.5%->12.9%) in Word Error Rate (WER) on average compared to the current state-of-the-art Audio-Only (AO) baselines while maintaining comparable results (< 5% difference) under the Audio-Visual (AV) setting even without video as input.

----

## [2178] Quantum Interference Model for Semantic Biases of Glosses in Word Sense Disambiguation

**Authors**: *Junwei Zhang, Ruifang He, Fengyu Guo, Chang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29927](https://doi.org/10.1609/aaai.v38i17.29927)

**Abstract**:

Word Sense Disambiguation (WSD) aims to determine the meaning of the target word according to the given context. Currently, a single representation enhanced by glosses from different dictionaries or languages is used to characterize each word sense. By analyzing the similarity between glosses of the same word sense, we find semantic biases among them, revealing that the glosses have their own descriptive perspectives. Therefore, the traditional approach of integrating all glosses by a single representation results in failing to present the unique semantics revealed by the individual glosses. In this paper, a quantum superposition state is employed to formalize the representations of multiple glosses of the same word sense to reveal their distributions. Furthermore, the quantum interference model is leveraged to calculate the probability that the target word belongs to this superposition state. The advantage is that the interference term can be regarded as a confidence level to guide word sense recognition. Finally, experiments are performed under standard WSD evaluation framework and the latest cross-lingual datasets, and the results verify the effectiveness of our model.

----

## [2179] Tree-of-Reasoning Question Decomposition for Complex Question Answering with Large Language Models

**Authors**: *Kun Zhang, Jiali Zeng, Fandong Meng, Yuanzhuo Wang, Shiqi Sun, Long Bai, Huawei Shen, Jie Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29928](https://doi.org/10.1609/aaai.v38i17.29928)

**Abstract**:

Large language models (LLMs) have recently demonstrated remarkable performance across various Natual Language Processing tasks. In the field of multi-hop reasoning, the Chain-of-thought (CoT) prompt method has emerged as a paradigm, using curated stepwise reasoning demonstrations to enhance LLM's ability to reason and produce coherent rational pathways. To ensure the accuracy, reliability, and traceability of the generated answers, many studies have incorporated information retrieval (IR) to provide LLMs with external knowledge. However, existing CoT with IR methods decomposes questions into sub-questions based on a single compositionality type, which limits their effectiveness for questions involving multiple compositionality types. Additionally, these methods suffer from inefficient retrieval, as complex questions often contain abundant information, leading to the retrieval of irrelevant information inconsistent with the query's intent. In this work, we propose a novel question decomposition framework called TRQA for multi-hop question answering, which addresses these limitations. Our framework introduces a reasoning tree (RT) to represent the structure of complex questions. It consists of four components: the Reasoning Tree Constructor (RTC), the Question Generator (QG), the Retrieval and LLM Interaction Module (RAIL), and the Answer Aggregation Module (AAM). Specifically, the RTC predicts diverse sub-question structures to construct the reasoning tree, allowing a more comprehensive representation of complex questions.  The QG generates sub-questions for leaf-node in the reasoning tree, and we explore two methods for QG: prompt-based and T5-based approaches. The IR module retrieves documents aligned with sub-questions, while the LLM formulates answers based on the retrieved information. Finally, the AAM aggregates answers along the reason tree, producing a definitive response from bottom to top.

----

## [2180] What to Remember: Self-Adaptive Continual Learning for Audio Deepfake Detection

**Authors**: *Xiaohui Zhang, Jiangyan Yi, Chenglong Wang, Chu Yuan Zhang, Siding Zeng, Jianhua Tao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29929](https://doi.org/10.1609/aaai.v38i17.29929)

**Abstract**:

The rapid evolution of speech synthesis and voice conversion has raised substantial concerns due to the potential misuse of such technology, prompting a pressing need for effective audio deepfake detection mechanisms. Existing detection models have shown remarkable success in discriminating known deepfake audio, but struggle when encountering new attack types. To address this challenge, one of the emergent effective approaches is continual learning. In this paper, we propose a continual learning approach called Radian Weight Modification (RWM) for audio deepfake detection. The fundamental concept underlying RWM involves categorizing all classes into two groups: those with compact feature distributions across tasks, such as genuine audio, and those with more spread-out distributions, like various types of fake audio. These distinctions are quantified by means of the in-class cosine distance, which subsequently serves as the basis for RWM to introduce a trainable gradient modification direction for distinct data types. Experimental evaluations against mainstream continual learning methods reveal the superiority of RWM in terms of knowledge acquisition and mitigating forgetting in audio deepfake detection. Furthermore, RWM's applicability extends beyond audio deepfake detection, demonstrating its potential significance in diverse machine learning domains such as image recognition.

----

## [2181] A Goal Interaction Graph Planning Framework for Conversational Recommendation

**Authors**: *Xiaotong Zhang, Xuefang Jia, Han Liu, Xinyue Liu, Xianchao Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29930](https://doi.org/10.1609/aaai.v38i17.29930)

**Abstract**:

Multi-goal conversational recommender system (MG-CRS) which is more in line with realistic scenarios has attracted a lot of attention. MG-CRS can dynamically capture the demands of users in conversation, continuously engage their interests, and make recommendations. The key of accomplishing these tasks is to plan a reasonable goal sequence which can naturally guide the user to accept the recommended goal. Previous works have demonstrated that mining the correlations of goals from the goal sequences in the dialogue corpus is helpful for recommending the goal that the user is interested in. However, they independently model correlations for each level of goal (i.e., goal type or entity) and neglect the order of goals appear in the dialogue. In this paper, we propose a goal interaction graph planning framework which constructs a directed heterogeneous graph to flexibly model the correlations between any level of goals and retain the order of goals. We design a goal interaction graph learning module to model the goal correlations and propagate goal representations via directed edges, then use an encoder and a dual-way fusion decoder to extract the most relevant information with the current goal from the conversation and domain knowledge, making the next-goal prediction fully exploit the prior goal correlations and user feedback. Finally we generate engaging responses based on the predicted goal sequence to complete the recommendation task. Experiments on two benchmark datasets show that our method achieves significant improvements in both the goal planning and response generation tasks.

----

## [2182] Personalized LoRA for Human-Centered Text Understanding

**Authors**: *You Zhang, Jin Wang, Liang-Chih Yu, Dan Xu, Xuejie Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29931](https://doi.org/10.1609/aaai.v38i17.29931)

**Abstract**:

Effectively and efficiently adapting a pre-trained language model (PLM) for human-centered text understanding (HCTU) is challenging since user tokens are million-level in most personalized applications and do not have concrete explicit semantics. A standard and parameter-efficient approach (e.g., LoRA) necessitates memorizing numerous suits of adapters for each user. In this work, we introduce a personalized LoRA (PLoRA) with a plug-and-play (PnP) framework for the HCTU task. PLoRA is effective, parameter-efficient, and dynamically deploying in PLMs. Moreover, a personalized dropout and a mutual information maximizing strategies are adopted and hence the proposed PLoRA can be well adapted to few/zero-shot learning scenarios for the cold-start issue. Experiments conducted on four benchmark datasets show that the proposed method outperforms existing methods in full/few/zero-shot learning scenarios for the HCTU task, even though it has fewer trainable parameters. For reproducibility, the code for this paper is available at: https://github.com/yoyo-yun/PLoRA.

----

## [2183] StyleSinger: Style Transfer for Out-of-Domain Singing Voice Synthesis

**Authors**: *Yu Zhang, Rongjie Huang, Ruiqi Li, Jinzheng He, Yan Xia, Feiyang Chen, Xinyu Duan, Baoxing Huai, Zhou Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29932](https://doi.org/10.1609/aaai.v38i17.29932)

**Abstract**:

Style transfer for out-of-domain (OOD) singing voice synthesis (SVS) focuses on generating high-quality singing voices with unseen styles (such as timbre, emotion, pronunciation, and articulation skills) derived from reference singing voice samples. However, the endeavor to model the intricate nuances of singing voice styles is an arduous task, as singing voices possess a remarkable degree of expressiveness. Moreover, existing SVS methods encounter a decline in the quality of synthesized singing voices in OOD scenarios, as they rest upon the assumption that the target vocal attributes are discernible during the training phase. To overcome these challenges, we propose StyleSinger, the first singing voice synthesis model for zero-shot style transfer of out-of-domain reference singing voice samples. StyleSinger incorporates two critical approaches for enhanced effectiveness: 1) the Residual Style Adaptor (RSA) which employs a residual quantization module to capture diverse style characteristics in singing voices, and 2) the Uncertainty Modeling Layer Normalization (UMLN) to perturb the style attributes within the content representation during the training phase and thus improve the model generalization. Our extensive evaluations in zero-shot style transfer undeniably establish that StyleSinger outperforms baseline models in both audio quality and similarity to the reference singing voice samples. Access to singing voice samples can be found at https://stylesinger.github.io/.

----

## [2184] Seed-Guided Fine-Grained Entity Typing in Science and Engineering Domains

**Authors**: *Yu Zhang, Yunyi Zhang, Yanzhen Shen, Yu Deng, Lucian Popa, Larisa Shwartz, ChengXiang Zhai, Jiawei Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29933](https://doi.org/10.1609/aaai.v38i17.29933)

**Abstract**:

Accurately typing entity mentions from text segments is a fundamental task for various natural language processing applications. Many previous approaches rely on massive human-annotated data to perform entity typing. Nevertheless, collecting such data in highly specialized science and engineering domains (e.g., software engineering and security) can be time-consuming and costly, without mentioning the domain gaps between training and inference data if the model needs to be applied to confidential datasets. In this paper, we study the task of seed-guided fine-grained entity typing in science and engineering domains, which takes the name and a few seed entities for each entity type as the only supervision and aims to classify new entity mentions into both seen and unseen types (i.e., those without seed entities). To solve this problem, we propose SEType which first enriches the weak supervision by finding more entities for each seen type from an unlabeled corpus using the contextualized representations of pre-trained language models. It then matches the enriched entities to unlabeled text to get pseudo-labeled samples and trains a textual entailment model that can make inferences for both seen and unseen types. Extensive experiments on two datasets covering four domains demonstrate the effectiveness of SEType in comparison with various baselines. Code and data are available at: https://github.com/yuzhimanhua/SEType.

----

## [2185] LLMEval: A Preliminary Study on How to Evaluate Large Language Models

**Authors**: *Yue Zhang, Ming Zhang, Haipeng Yuan, Shichun Liu, Yongyao Shi, Tao Gui, Qi Zhang, Xuanjing Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29934](https://doi.org/10.1609/aaai.v38i17.29934)

**Abstract**:

Recently, the evaluation of Large Language Models has emerged as a popular area of research. 
The three crucial questions for LLM evaluation are ``what, where, and how to evaluate''.
However, the existing research mainly focuses on the first two questions, which are basically what tasks to give the LLM during testing and what kind of knowledge it should deal with.
As for the third question, which is about what standards to use, the types of evaluators, how to score, and how to rank, there hasn't been much discussion.
In this paper, we analyze evaluation methods by comparing various criteria with both manual and automatic evaluation, utilizing onsite, crowd-sourcing, public annotators and GPT-4, with different scoring methods and ranking systems. 
We propose a new dataset, LLMEval and conduct evaluations on 20 LLMs. 
A total of 2,186 individuals participated, leading to the generation of 243,337 manual annotations and 57,511 automatic evaluation results.
We perform comparisons and analyses of different settings and conduct 10 conclusions that can provide some insights for evaluating LLM in the future. The dataset and the results are publicly available at 
https://github.com/llmeval.
The version with the appendix are publicly available at https://arxiv.org/abs/2312.07398.

----

## [2186] Coreference Graph Guidance for Mind-Map Generation

**Authors**: *Zhuowei Zhang, Mengting Hu, Yinhao Bai, Zhen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29935](https://doi.org/10.1609/aaai.v38i17.29935)

**Abstract**:

Mind-map generation aims to process a document into a hierarchical structure to show its central idea and branches. Such a manner is more conducive to understanding the logic and semantics of the document than plain text. Recently, a state-of-the-art method encodes the sentences of a document sequentially and converts them to a relation graph via sequence-to-graph. Though this method is efficient to generate mind-maps in parallel, its mechanism focuses more on sequential features while hardly capturing structural information. Moreover, it's difficult to model long-range semantic relations. In this work, we propose a coreference-guided mind-map generation network (CMGN) to incorporate external structure knowledge. Specifically, we construct a coreference graph based on the coreference semantic relationship to introduce the graph structure information. Then we employ a coreference graph encoder to mine the potential governing relations between sentences. In order to exclude noise and better utilize the information of the coreference graph, we adopt a graph enhancement module in a contrastive learning manner. Experimental results demonstrate that our model outperforms all the existing methods. The case study further proves that our model can more accurately and concisely reveal the structure and semantics of a document. Code and data are available at https://github.com/Cyno2232/CMGN.

----

## [2187] ExpeL: LLM Agents Are Experiential Learners

**Authors**: *Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, Gao Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29936](https://doi.org/10.1609/aaai.v38i17.29936)

**Abstract**:

The recent surge in research interest in applying large language models (LLMs) to decision-making tasks has flourished by leveraging the extensive world knowledge embedded in LLMs. While there is a growing demand to tailor LLMs for custom decision-making tasks, finetuning them for specific tasks is resource-intensive and may diminish the model's generalization capabilities. Moreover, state-of-the-art language models like GPT-4 and Claude are primarily accessible through API calls, with their parametric weights remaining proprietary and unavailable to the public. This scenario emphasizes the growing need for new methodologies that allow learning from agent experiences without requiring parametric updates. To address these problems, we introduce the Experiential Learning (ExpeL) agent. Our agent autonomously gathers experiences and extracts knowledge using natural language from a collection of training tasks. At inference, the agent recalls its extracted insights and past experiences to make informed decisions. Our empirical results highlight the robust learning efficacy of the ExpeL agent, indicating a consistent enhancement in its performance as it accumulates experiences. We further explore the emerging capabilities and transfer learning potential of the ExpeL agent through qualitative observations and additional experiments.

----

## [2188] Conditional Variational Autoencoder for Sign Language Translation with Cross-Modal Alignment

**Authors**: *Rui Zhao, Liang Zhang, Biao Fu, Cong Hu, Jinsong Su, Yidong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29937](https://doi.org/10.1609/aaai.v38i17.29937)

**Abstract**:

Sign language translation (SLT) aims to convert continuous sign language videos into textual sentences. As a typical multi-modal task, there exists an inherent modality gap between sign language videos and spoken language text, which makes the cross-modal alignment between visual and textual modalities crucial. However, previous studies tend to rely on an intermediate sign gloss representation to help alleviate the cross-modal problem thereby neglecting the alignment across modalities that may lead to compromised results. To address this issue, we propose a novel framework based on Conditional Variational autoencoder for SLT (CV-SLT) that facilitates direct and sufficient cross-modal alignment between sign language videos and spoken language text. Specifically, our CV-SLT consists of two paths with two Kullback-Leibler (KL) divergences to regularize the outputs of the encoder and decoder, respectively. In the prior path, the model solely relies on visual information to predict the target text; whereas in the posterior path, it simultaneously encodes visual information and textual knowledge to reconstruct the target text. The first KL divergence optimizes the conditional variational autoencoder and regularizes the encoder outputs, while the second KL divergence performs a self-distillation from the posterior path to the prior path, ensuring the consistency of decoder outputs.We further enhance the integration of textual information to the posterior path by employing a shared Attention Residual Gaussian Distribution (ARGD), which considers the textual information in the posterior path as a residual component relative to the prior path. Extensive experiments conducted on public datasets demonstrate the effectiveness of our framework, achieving new state-of-the-art results while significantly alleviating the cross-modal representation discrepancy. The code and models are available at https://github.com/rzhao-zhsq/CV-SLT.

----

## [2189] Graph Reasoning Transformers for Knowledge-Aware Question Answering

**Authors**: *Ruilin Zhao, Feng Zhao, Liang Hu, Guandong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29938](https://doi.org/10.1609/aaai.v38i17.29938)

**Abstract**:

Augmenting Language Models (LMs) with structured knowledge graphs (KGs) aims to leverage structured world knowledge to enhance the capability of LMs to complete knowledge-intensive tasks. However, existing methods are unable to effectively utilize the structured knowledge in a KG due to their inability to capture the rich relational semantics of knowledge triplets. Moreover, the modality gap between natural language text and KGs has become a challenging obstacle when aligning and fusing cross-modal information. To address these challenges, we propose a novel knowledge-augmented question answering (QA) model, namely, Graph Reasoning Transformers (GRT). Different from conventional node-level methods, the GRT serves knowledge triplets as atomic knowledge and utilize a triplet-level graph encoder to capture triplet-level graph features. Furthermore, to alleviate the negative effect of the modality gap on joint reasoning, we propose a representation alignment  pretraining to align the cross-modal representations and introduce a cross-modal information fusion module with attention bias to enable fine-grained information fusion. Extensive experiments conducted on three knowledge-intensive QA benchmarks show that the GRT outperforms the state-of-the-art KG-augmented QA systems, demonstrating the effectiveness and adaptation of our proposed model.

----

## [2190] MultiSum: A Multi-Facet Approach for Extractive Social Summarization Utilizing Semantic and Sociological Relationships

**Authors**: *Tanglong Zhao, Ruifang He, Jing Xu, Bo Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29939](https://doi.org/10.1609/aaai.v38i17.29939)

**Abstract**:

Social summarization aims to provide summaries for a large number of social texts (called posts)  about a single topic. To extract a summary, both  the representation of post and summary selection method are crucial. Previous methods introduce social relation to enhance post embedding to mitigate the sparse representation due to its brief and informal expression. However, they ignore that there are multiple relations between posts. Besides, existing graph-based centrality calculation approaches tend to select posts from one aspect. This leads to facet bias especially when there are multiple viewpoints. In this paper, we propose a model named MultiSum to improve social summarization. Specifically, 1) We use graph convolutional networks to fuse text content with social and semantic relations to improve post representation; 2) The similarity between the summary and all aspects is incorporated into the centrality score during the selection phase, encouraging the model to pay attention to different facets. Experimental results on English and Chinese corpora support the effectiveness of this model. Furthermore, external evaluations by human experts and large language models demonstrate the validity of MultiSum in facet coverage and redundancy reduction.

----

## [2191] QPEN: Quantum Projection and Quantum Entanglement Enhanced Network for Cross-Lingual Aspect-Based Sentiment Analysis

**Authors**: *Xingqiang Zhao, Hai Wan, Kunxun Qi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29940](https://doi.org/10.1609/aaai.v38i17.29940)

**Abstract**:

Aspect-based sentiment analysis (ABSA) has attracted much attention due to its wide application scenarios. Most previous studies have focused solely on monolingual ABSA, posing a formidable challenge when extending ABSA applications to multilingual scenarios. In this paper, we study upgrading monolingual ABSA to cross-lingual ABSA. Existing methods usually exploit pre-trained cross-lingual language to model cross-lingual ABSA, and enhance the model with translation data. However, the low-resource languages might be under-represented during the pre-training phase, and the translation-enhanced methods heavily rely on the quality of the translation and label projection. Inspired by the observation that quantum entanglement can correlate multiple single systems, we map the monolingual expression to the quantum Hilbert space as a single quantum system, and then utilize quantum entanglement and quantum measurement to achieve cross-lingual ABSA. Specifically, we propose a novel quantum neural model named QPEN (short for quantum projection and quantum entanglement enhanced network). It is equipped with a proposed quantum projection module that projects aspects as quantum superposition on a complex-valued Hilbert space. Furthermore, a quantum entanglement module is proposed in QPEN to share language-specific features between different languages without transmission. We conducted simulation experiments on the classical computer, and experimental results on SemEval-2016 dataset demonstrate that our method achieves state-of-the-art performance in terms of F1-scores for five languages.

----

## [2192] SENCR: A Span Enhanced Two-Stage Network with Counterfactual Rethinking for Chinese NER

**Authors**: *Hang Zheng, Qingsong Li, Shen Chen, Yuxuan Liang, Li Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29941](https://doi.org/10.1609/aaai.v38i17.29941)

**Abstract**:

Recently, lots of works that incorporate external lexicon information into character-level Chinese named entity recognition(NER) to overcome the lackness of natural delimiters of words, have achieved many advanced performance. However, obtaining and maintaining high-quality lexicons is costly, especially in special domains. In addition, the entity boundary bias caused by high mention coverage in some boundary characters poses a significant challenge to the generalization of NER models but receives little attention in the existing literature. To address these issues, we propose SENCR, a Span Enhanced Two-stage Network with Counterfactual Rethinking for Chinese NER, that contains a boundary detector for boundary supervision, a convolution-based type classifier for better span representation and a counterfactual rethinking(CR) strategy for debiased boundary detection in inference. The proposed boundary detector and type classifier are jointly trained with the same contextual encoder and then the trained boundary detector is debiased by our proposed CR strategy without modifying any model parameters in the inference stage. Extensive experiments on four Chinese NER datasets show the effectiveness of our proposed approach.

----

## [2193] Reverse Multi-Choice Dialogue Commonsense Inference with Graph-of-Thought

**Authors**: *Li Zheng, Hao Fei, Fei Li, Bobo Li, Lizi Liao, Donghong Ji, Chong Teng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29942](https://doi.org/10.1609/aaai.v38i17.29942)

**Abstract**:

With the proliferation of dialogic data across the Internet, the Dialogue Commonsense Multi-choice Question Answering (DC-MCQ) task has emerged as a response to the challenge of comprehending user queries and intentions.
Although prevailing methodologies exhibit effectiveness in addressing single-choice questions, they encounter difficulties in handling multi-choice queries due to the heightened intricacy and informational density. 
In this paper, inspired by the human cognitive process of progressively excluding options, we propose a three-step Reverse Exclusion Graph-of-Thought (ReX-GoT) framework, including Option Exclusion, Error Analysis, and Combine Information.
Specifically, our ReX-GoT mimics human reasoning by gradually excluding irrelevant options and learning the reasons for option errors to choose the optimal path of the GoT and ultimately infer the correct answer.
By progressively integrating intricate clues, our method effectively reduces the difficulty of multi-choice reasoning and provides a novel solution for DC-MCQ.
Extensive experiments on the CICERO and CICERO_v2 datasets validate the significant improvement of our approach on DC-MCQ task.
On zero-shot setting, our model outperform the best baseline by 17.67% in terms of F1 score for the multi-choice task.
Most strikingly, our GPT3.5-based ReX-GoT framework achieves a remarkable 39.44% increase in F1 score.

----

## [2194] FT-GAN: Fine-Grained Tune Modeling for Chinese Opera Synthesis

**Authors**: *Meizhen Zheng, Peng Bai, Xiaodong Shi, Xun Zhou, Yiting Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29943](https://doi.org/10.1609/aaai.v38i17.29943)

**Abstract**:

Although singing voice synthesis (SVS) has made significant progress recently, with its unique styles and various genres, Chinese opera synthesis requires greater attention but is rarely studied for lack of training data and high expressiveness. In this work, we build a high-quality Gezi Opera (a type of Chinese opera popular in Fujian and Taiwan) audio-text alignment dataset and formulate specific data annotation methods applicable to Chinese operas. We propose FT-GAN, an acoustic model for fine-grained tune modeling in Chinese opera synthesis based on the empirical analysis of the differences between Chinese operas and pop songs. To further improve the quality of the synthesized opera, we propose a speech pre-training strategy for additional knowledge injection. The experimental results show that FT-GAN outperforms the strong baselines in SVS on the Gezi Opera synthesis task. Extensive experiments further verify that FT-GAN performs well on synthesis tasks of other operas such as Peking Opera. Audio samples, the dataset, and the codes are available at https://zhengmidon.github.io/FTGAN.github.io/.

----

## [2195] Layer-Wise Representation Fusion for Compositional Generalization

**Authors**: *Yafang Zheng, Lei Lin, Shuangtao Li, Yuxuan Yuan, Zhaohong Lai, Shan Liu, Biao Fu, Yidong Chen, Xiaodong Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29944](https://doi.org/10.1609/aaai.v38i17.29944)

**Abstract**:

Existing neural models are demonstrated to struggle with compositional generalization (CG), i.e., the ability to systematically generalize to unseen compositions of seen components. A key reason for failure on CG is that the syntactic and semantic representations of sequences in both the uppermost layer of the encoder and decoder are entangled. However, previous work concentrates on separating the learning of syntax and semantics instead of exploring the reasons behind the representation entanglement (RE) problem to solve it. We explain why it exists by analyzing the representation evolving mechanism from the bottom to the top of the Transformer layers. We find that the ``shallow'' residual connections within each layer fail to fuse previous layers' information effectively, leading to information forgetting between layers and further the RE problems. Inspired by this, we propose LRF, a novel Layer-wise Representation Fusion framework for CG, which learns to fuse previous layers' information back into the encoding and decoding process effectively through introducing a fuse-attention module at each encoder and decoder layer. LRF achieves promising results on two realistic benchmarks, empirically demonstrating the effectiveness of our proposal. Codes are available at https://github.com/thinkaboutzero/LRF.

----

## [2196] You Only Read Once: Constituency-Oriented Relational Graph Convolutional Network for Multi-Aspect Multi-Sentiment Classification

**Authors**: *Yongqiang Zheng, Xia Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29945](https://doi.org/10.1609/aaai.v38i17.29945)

**Abstract**:

Most of the existing aspect-based sentiment analysis (ABSA) models only predict the sentiment polarity of a single aspect at a time, focusing primarily on enhancing the representation of this single aspect based on the other contexts or aspects. This one-to-one paradigm ignores the fact that multi-aspect, multi-sentiment sentences contain not only distinct specific descriptions for distinct specific aspects, but also shared global context information for multiple aspects. To fully consider these issues, we propose a one-to-many ABSA framework, called You Only Read Once (YORO), that can simultaneously model representations of all aspects based on their specific descriptions and better fuse their relationships using globally shared contextual information in the sentence. Predicting the sentiment polarity of multiple aspects simultaneously is beneficial to improving the efficacy of calculation and prediction. Extensive experiments are conducted on three public datasets (MAMS, Rest14, and Lap14). Experimental results demonstrate the effectiveness of YORO in handling multi-aspect, multi-sentiment scenarios and highlight the promise of one-to-many ABSA in balancing efficiency and accuracy.

----

## [2197] MemoryBank: Enhancing Large Language Models with Long-Term Memory

**Authors**: *Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, Yanlin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29946](https://doi.org/10.1609/aaai.v38i17.29946)

**Abstract**:

Large Language Models (LLMs) have drastically reshaped our interactions with artificial intelligence (AI) systems, showcasing impressive performance across an extensive array of tasks. Despite this, a notable hindrance remains—the deficiency of a long-term memory mechanism within these models. This shortfall becomes increasingly evident in situations demanding sustained interaction, such as personal companion systems, psychological counseling, and secretarial assistance. Recognizing the necessity for long-term memory, we propose MemoryBank, a novel memory mechanism tailored for LLMs. MemoryBank enables the models to summon relevant memories, continually evolve through continuous memory updates, comprehend, and adapt to a user's personality over time by synthesizing information from previous interactions. To mimic anthropomorphic behaviors and selectively preserve memory, MemoryBank incorporates a memory updating mechanism, inspired by the Ebbinghaus Forgetting Curve theory. This mechanism permits the AI to forget and reinforce memory based on time elapsed and the relative significance of the memory, thereby offering a more human-like memory mechanism and enriched user experience. MemoryBank is versatile in accommodating both closed-source models like ChatGPT and open-source models such as ChatGLM. To validate MemoryBank's effectiveness, we exemplify its application through the creation of an LLM-based chatbot named SiliconFriend in a long-term AI Companion scenario. Further tuned with psychological dialog data, SiliconFriend displays heightened empathy and discernment in its interactions. Experiment involves both qualitative analysis with real-world user dialogs and quantitative analysis with simulated dialogs. In the latter, ChatGPT acts as multiple users with diverse characteristics and generates long-term dialog contexts covering a wide array of topics. The results of our analysis reveal that SiliconFriend, equipped with MemoryBank, exhibits a strong capability for long-term companionship as it can provide emphatic response, recall relevant memories and understand user personality.

----

## [2198] Fine-Grained Distillation for Long Document Retrieval

**Authors**: *Yucheng Zhou, Tao Shen, Xiubo Geng, Chongyang Tao, Jianbing Shen, Guodong Long, Can Xu, Daxin Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29947](https://doi.org/10.1609/aaai.v38i17.29947)

**Abstract**:

Long document retrieval aims to fetch query-relevant documents from a large-scale collection, where knowledge distillation has become de facto to improve a retriever by mimicking a heterogeneous yet powerful cross-encoder. However, in contrast to passages or sentences, retrieval on long documents suffers from the \textit{scope hypothesis} that a long document may cover multiple topics. This maximizes their structure heterogeneity and poses a granular-mismatch issue, leading to an inferior distillation efficacy. In this work, we propose a new learning framework, fine-grained distillation (FGD), for long-document retrievers. While preserving the conventional dense retrieval paradigm, it first produces global-consistent representations crossing different fine granularity and then applies multi-granular aligned distillation merely during training. In experiments, we evaluate our framework on two long-document retrieval benchmarks, which show state-of-the-art performance.

----

## [2199] Quantifying and Analyzing Entity-Level Memorization in Large Language Models

**Authors**: *Zhenhong Zhou, Jiuyang Xiang, Chaomeng Chen, Sen Su*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29948](https://doi.org/10.1609/aaai.v38i17.29948)

**Abstract**:

Large language models (LLMs) have been proven capable of memorizing their training data, which can be extracted through specifically designed prompts. As the scale of datasets continues to grow, privacy risks arising from memorization have attracted increasing attention. Quantifying language model memorization helps evaluate potential privacy risks. However, prior works on quantifying memorization require access to the precise original data or incur substantial computational overhead, making it difficult for applications in real-world language models. To this end, we propose a fine-grained, entity-level definition to quantify memorization with conditions and metrics closer to real-world scenarios. In addition, we also present an approach for efficiently extracting sensitive entities from autoregressive language models. We conduct extensive experiments based on the proposed, probing language models' ability to reconstruct sensitive entities under different settings. We find that language models have strong memorization at the entity level and are able to reproduce the training data even with partial leakages. The results demonstrate that LLMs not only memorize their training data but also understand associations between entities. These findings necessitate that trainers of LLMs exercise greater prudence regarding model memorization, adopting memorization mitigation techniques to preclude privacy violations.

----



[Go to the previous page](AAAI-2024-list10.md)

[Go to the next page](AAAI-2024-list12.md)

[Go to the catalog section](README.md)