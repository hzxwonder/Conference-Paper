## [0] A Multi-Modal Contrastive Diffusion Model for Therapeutic Peptide Generation

**Authors**: *Yongkang Wang, Xuan Liu, Feng Huang, Zhankun Xiong, Wen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27749](https://doi.org/10.1609/aaai.v38i1.27749)

**Abstract**:

Therapeutic peptides represent a unique class of pharmaceutical agents crucial for the treatment of human diseases. Recently, deep generative models have exhibited remarkable potential for generating therapeutic peptides, but they only utilize sequence or structure information alone, which hinders the performance in generation. In this study, we propose a Multi-Modal Contrastive Diffusion model (MMCD), fusing both sequence and structure modalities in a diffusion framework to co-generate novel peptide sequences and structures. Specifically, MMCD constructs the sequence-modal and structure-modal diffusion models, respectively, and devises a multi-modal contrastive learning strategy with inter-contrastive and intra-contrastive in each diffusion timestep, aiming to capture the consistency between two modalities and boost model performance. The inter-contrastive aligns sequences and structures of peptides by maximizing the agreement of their embeddings, while the intra-contrastive differentiates therapeutic and non-therapeutic peptides by maximizing the disagreement of their sequence/structure embeddings simultaneously. The extensive experiments demonstrate that MMCD performs better than other state-of-the-art deep generative methods in generating therapeutic peptides across various metrics, including antimicrobial/anticancer score, diversity, and peptide-docking.

----

## [1] Towards Automated RISC-V Microarchitecture Design with Reinforcement Learning

**Authors**: *Chen Bai, Jianwang Zhai, Yuzhe Ma, Bei Yu, Martin D. F. Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27750](https://doi.org/10.1609/aaai.v38i1.27750)

**Abstract**:

Microarchitecture determines the implementation of a microprocessor. Designing a microarchitecture to achieve better performance, power, and area (PPA) trade-off has been increasingly difficult. Previous data-driven methodologies hold inappropriate assumptions and lack more tightly coupling with expert knowledge. This paper proposes a novel reinforcement learning-based (RL) solution that addresses these limitations. With the integration of microarchitecture scaling graph, PPA preference space embedding, and proposed lightweight environment in RL, experiments using commercial electronic design automation (EDA) tools show that our method achieves an average PPA trade-off improvement of 16.03% than previous state-of-the-art approaches with 4.07× higher efficiency. The solution qualities outperform human implementations by at most 2.03× in the PPA trade-off.

----

## [2] Generating Novel Leads for Drug Discovery Using LLMs with Logical Feedback

**Authors**: *Shreyas Bhat Brahmavar, Ashwin Srinivasan, Tirtharaj Dash, Sowmya Ramaswamy Krishnan, Lovekesh Vig, Arijit Roy, Raviprasad Aduri*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27751](https://doi.org/10.1609/aaai.v38i1.27751)

**Abstract**:

Large Language Models (LLMs) can be used as repositories of biological and chemical information to generate pharmacological lead compounds. However, for LLMs to focus on specific drug targets typically requires experimentation with progressively more refined prompts. Results thus become dependent not just on what is known about the target, but also on what is known about the prompt- engineering. In this paper, we separate the prompt into domain-constraints that can be written in a standard logical form and a simple text-based query. We investigate whether LLMs can be guided, not by refining prompts manually, but by refining the logical component automatically, keeping the query unchanged. We describe an iterative procedure LMLF (“Language Model with Logical Feedback”) in which the constraints are progressively refined using a logical notion of generalisation. On any iteration, newly generated instances are verified against the constraint, providing "logical-feedback" for the next iteration's refinement of the constraints. We evaluate LMLF using two well-known  targets (inhibition of the Janus Kinase 2; and Dopamine Receptor D2); and two different LLMs (GPT-3 and PaLM). We show that LMLF, starting with the same logical constraints and query text, can be used to guide both LLMs to generate potential leads. We find: (a)  Binding affinities of LMLF-generated molecules are skewed towards higher binding affinities than those from existing baselines; (b) LMLF results in generating molecules that are skewed towards higher binding affinities than without logical feedback; (c) Assessment by a computational chemist suggests that LMLF generated compounds may be novel inhibitors. These findings suggest that LLMs with logical feedback may provide a mechanism for generating new leads without requiring the domain-specialist to acquire sophisticated skills in prompt-engineering.

----

## [3] SeGA: Preference-Aware Self-Contrastive Learning with Prompts for Anomalous User Detection on Twitter

**Authors**: *Ying-Ying Chang, Wei-Yao Wang, Wen-Chih Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27752](https://doi.org/10.1609/aaai.v38i1.27752)

**Abstract**:

In the dynamic and rapidly evolving world of social media, detecting anomalous users has become a crucial task to address malicious activities such as misinformation and cyberbullying. As the increasing number of anomalous users improves the ability to mimic normal users and evade detection, existing methods only focusing on bot detection are ineffective in terms of capturing subtle distinctions between users. To address these challenges, we proposed SeGA, preference-aware self-contrastive learning for anomalous user detection, which leverages heterogeneous entities and their relations in the Twittersphere to detect anomalous users with different malicious strategies. SeGA utilizes the knowledge of large language models to summarize user preferences via posts. In addition, integrating user preferences with prompts as pseudo-labels for preference-aware self-contrastive learning enables the model to learn multifaceted aspects for describing the behaviors of users. Extensive experiments on the proposed TwBNT benchmark demonstrate that SeGA significantly outperforms the state-of-the-art methods (+3.5% ∼ 27.6%) and empirically validate the effectiveness of the model design and pre-training strategies. Our code and data are publicly available at https://github.com/ying0409/SeGA.

----

## [4] Neural Embeddings for kNN Search in Biological Sequence

**Authors**: *Zhihao Chang, Linzhu Yu, Yanchao Xu, Wentao Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27753](https://doi.org/10.1609/aaai.v38i1.27753)

**Abstract**:

Biological sequence nearest neighbor search plays a fundamental role in bioinformatics. To alleviate the pain of quadratic complexity for conventional distance computation, neural distance embeddings, which project sequences into geometric space, have been recognized as a promising paradigm. To maintain the distance order between sequences, these models all deploy triplet loss and use intuitive methods to select a subset of triplets for training from a vast selection space. However, we observed that such training often enables models to distinguish only a fraction of distance orders, leaving others unrecognized. Moreover, naively selecting more triplets for training under the state-of-the-art network not only adds costs but also hampers model performance.
    In this paper, we introduce Bio-kNN: a kNN search framework for biological sequences. It includes a systematic triplet selection method and a multi-head network, enhancing the discernment of all distance orders without increasing training expenses. Initially, we propose a clustering-based approach to partition all triplets into several clusters with similar properties, and then select triplets from these clusters using an innovative strategy. Meanwhile, we noticed that simultaneously training different types of triplets in the same network cannot achieve the expected performance, thus we propose a multi-head network to tackle this. Our network employs a convolutional neural network(CNN) to extract local features shared by all clusters, and then learns a multi-layer perception(MLP) head for each cluster separately. Besides, we treat CNN as a special head, thereby integrating crucial local features which are neglected in previous models into our model for similarity recognition. Extensive experiments show that our Bio-kNN significantly outperforms the state-of-the-art methods on two large-scale datasets without increasing the training cost.

----

## [5] i-Rebalance: Personalized Vehicle Repositioning for Supply Demand Balance

**Authors**: *Haoyang Chen, Peiyan Sun, Qiyuan Song, Wanyuan Wang, Weiwei Wu, Wencan Zhang, Guanyu Gao, Yan Lyu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27754](https://doi.org/10.1609/aaai.v38i1.27754)

**Abstract**:

Ride-hailing platforms have been facing the challenge of balancing demand and supply. Existing vehicle reposition techniques often treat drivers as homogeneous agents and relocate them deterministically, assuming compliance with the reposition. In this paper, we consider a more realistic and driver-centric scenario where drivers have unique cruising preferences and can decide whether to take the recommendation or not on their own. We propose i-Rebalance, a personalized vehicle reposition technique with deep reinforcement learning (DRL). i-Rebalance estimates drivers' decisions on accepting reposition recommendations through an on-field user study involving 99 real drivers. To optimize supply-demand balance and enhance preference satisfaction simultaneously, i-Rebalance has a sequential reposition strategy with dual DRL agents: Grid Agent to determine the reposition order of idle vehicles, and Vehicle Agent to provide personalized recommendations to each vehicle in the pre-defined order. This sequential learning strategy facilitates more effective policy training within a smaller action space compared to traditional joint-action methods. Evaluation of real-world trajectory data shows that i-Rebalance improves driver acceptance rate by 38.07% and total driver income by 9.97%.

----

## [6] GIN-SD: Source Detection in Graphs with Incomplete Nodes via Positional Encoding and Attentive Fusion

**Authors**: *Le Cheng, Peican Zhu, Keke Tang, Chao Gao, Zhen Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27755](https://doi.org/10.1609/aaai.v38i1.27755)

**Abstract**:

Source detection in graphs has demonstrated robust efficacy in the domain of rumor source identification. Although recent solutions have enhanced performance by leveraging deep neural networks, they often require complete user data. In this paper, we address a more challenging task, rumor source detection with incomplete user data, and propose a novel framework, i.e., Source Detection in Graphs with Incomplete Nodes via Positional Encoding and Attentive Fusion (GIN-SD), to tackle this challenge. Specifically, our approach utilizes a positional embedding module to distinguish nodes that are incomplete and employs a self-attention mechanism to focus on nodes with greater information transmission capacity. To mitigate the prediction bias caused by the significant disparity between the numbers of source and non-source nodes, we also introduce a class-balancing mechanism. Extensive experiments validate the effectiveness of GIN-SD and its superiority to state-of-the-art methods.

----

## [7] Deep Quantum Error Correction

**Authors**: *Yoni Choukroun, Lior Wolf*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27756](https://doi.org/10.1609/aaai.v38i1.27756)

**Abstract**:

Quantum error correction codes (QECC) are a key component for realizing the potential of quantum computing. QECC, as its classical counterpart (ECC), enables the reduction of error rates, by distributing quantum logical information across redundant physical qubits, such that errors can be detected and corrected. In this work, we efficiently train novel end-to-end deep quantum error decoders. We resolve the quantum measurement collapse by augmenting syndrome decoding to predict an initial estimate of the system noise, which is then refined iteratively through a deep neural network. The logical error rates calculated over finite fields are directly optimized via a differentiable objective, enabling efficient decoding under the constraints imposed by the code. Finally, our architecture is extended to support faulty syndrome measurement, by efficient decoding of repeated syndrome sampling. The proposed method demonstrates the power of neural decoders for QECC by achieving state-of-the-art accuracy, outperforming for small distance topological codes, the existing end-to-end neural and classical decoders, which are often computationally prohibitive.

----

## [8] Propagation Tree Is Not Deep: Adaptive Graph Contrastive Learning Approach for Rumor Detection

**Authors**: *Chaoqun Cui, Caiyan Jia*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27757](https://doi.org/10.1609/aaai.v38i1.27757)

**Abstract**:

Rumor detection on social media has become increasingly important. Most existing graph-based models presume rumor propagation trees (RPTs) have deep structures and learn sequential stance features along branches. However, through statistical analysis on real-world datasets, we find RPTs exhibit wide structures, with most nodes being shallow 1-level replies. To focus learning on intensive substructures, we propose Rumor Adaptive Graph Contrastive Learning (RAGCL) method with adaptive view augmentation guided by node centralities. We summarize three principles for RPT augmentation: 1) exempt root nodes, 2) retain deep reply nodes, 3) preserve lower-level nodes in deep sections. We employ node dropping, attribute masking and edge dropping with probabilities from centrality-based importance scores to generate views. A graph contrastive objective then learns robust rumor representations. Extensive experiments on four benchmark datasets demonstrate RAGCL outperforms state-of-the-art methods. Our work reveals the wide-structure nature of RPTs and contributes an effective graph contrastive learning approach tailored for rumor detection through principled adaptive augmentation. The proposed principles and augmentation techniques can potentially benefit other applications involving tree-structured graphs.

----

## [9] Prompt to Transfer: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning

**Authors**: *Longchao Da, Minquan Gao, Hao Mei, Hua Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27758](https://doi.org/10.1609/aaai.v38i1.27758)

**Abstract**:

Numerous solutions are proposed for the Traffic Signal Control (TSC) tasks aiming to provide efficient transportation and alleviate traffic congestion. Recently, promising results have been attained by Reinforcement Learning (RL) methods through trial and error in simulators, bringing confidence in solving cities' congestion problems. However, performance gaps still exist when simulator-trained policies are deployed to the real world. This issue is mainly introduced by the system dynamic difference between the training simulators and the real-world environments. In this work, we leverage the knowledge of Large Language Models (LLMs) to understand and profile the system dynamics by a prompt-based grounded action transformation to bridge the performance gap. Specifically, this paper exploits the pre-trained LLM's inference ability to understand how traffic dynamics change with weather conditions, traffic states, and road types. Being aware of the changes, the policies' action is taken and grounded based on realistic dynamics, thus helping the agent learn a more realistic policy. We conduct experiments on four different scenarios to show the effectiveness of the proposed PromptGAT's ability to mitigate the performance gap of reinforcement learning from simulation to reality (sim-to-real).

----

## [10] Multitarget Device-Free Localization via Cross-Domain Wi-Fi RSS Training Data and Attentional Prior Fusion

**Authors**: *Na Fan, Zeyue Tian, Amartansh Dubey, Samruddhi Deshmukh, Ross D. Murch, Qifeng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27759](https://doi.org/10.1609/aaai.v38i1.27759)

**Abstract**:

Device-free localization (DFL) using easily-obtained Wi-Fi received signal strength (RSS) has wide real-world applications for not requiring people to carry trackable devices.
However, accurate multitarget DFL remains challenging due to the unknown number of targets, multipath interference (MPI), especially between nearby targets, and limited real-world data. 
In this study, we pioneeringly propose a transformer-based learning method with Wi-Fi RSS as input, and an attentional prior fusion module, to simultaneously locate an unknown number of people at random positions. 
To overcome the multitarget data collection challenges, we contribute a large-scale cross-domain real-simulation-augmentation training dataset with one and two real-world nearby non-person objects at limited positions and up to five simulated and augmented randomly distributed targets.
Experimental results demonstrate our method's improved accuracy, generalization ability, and robustness with fewer Wi-Fi nodes than previous methods.

----

## [11] Heterogeneous Graph Reasoning for Fact Checking over Texts and Tables

**Authors**: *Haisong Gong, Weizhi Xu, Shu Wu, Qiang Liu, Liang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27760](https://doi.org/10.1609/aaai.v38i1.27760)

**Abstract**:

Fact checking aims to predict claim veracity by reasoning over multiple evidence pieces. It usually involves evidence retrieval and veracity reasoning. In this paper, we focus on the latter, reasoning over unstructured text and structured table information. Previous works have primarily relied on fine-tuning pretrained language models or training homogeneous-graph-based models. Despite their effectiveness, we argue that they fail to explore the rich semantic information underlying the evidence with different structures. To address this, we propose a novel word-level Heterogeneous-graph-based model for Fact Checking over unstructured and structured information, namely HeterFC. Our approach leverages a heterogeneous evidence graph, with words as nodes and thoughtfully designed edges representing different evidence properties. We perform information propagation via a relational graph neural network, facilitating interactions between claims and evidence. An attention-based method is utilized to integrate information, combined with a language model for generating predictions. We introduce a multitask loss function to account for potential inaccuracies in evidence retrieval. Comprehensive experiments on the large fact checking dataset FEVEROUS demonstrate the effectiveness of HeterFC. Code will be released at: https://github.com/Deno-V/HeterFC.

----

## [12] Text-Guided Molecule Generation with Diffusion Language Model

**Authors**: *Haisong Gong, Qiang Liu, Shu Wu, Liang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27761](https://doi.org/10.1609/aaai.v38i1.27761)

**Abstract**:

Text-guided molecule generation is a task where molecules are generated to match specific textual descriptions. Recently, most existing SMILES-based molecule generation methods rely on an autoregressive architecture. In this work, we propose the Text-Guided Molecule Generation with Diffusion Language Model (TGM-DLM), a novel approach that leverages diffusion models to address the limitations of autoregressive methods. TGM-DLM updates token embeddings within the SMILES string collectively and iteratively, using a two-phase diffusion generation process. The first phase optimizes embeddings from random noise, guided by the text description, while the second phase corrects invalid SMILES strings to form valid molecular representations. We demonstrate that TGM-DLM outperforms MolT5-Base, an autoregressive model, without the need for additional data resources. Our findings underscore the remarkable effectiveness of TGM-DLM in generating coherent and precise molecules with specific properties, opening new avenues in drug discovery and related scientific domains. Code will be released at: https://github.com/Deno-V/tgm-dlm.

----

## [13] Adversarial Robust Safeguard for Evading Deep Facial Manipulation

**Authors**: *Jiazhi Guan, Yi Zhao, Zhuoer Xu, Changhua Meng, Ke Xu, Youjian Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27762](https://doi.org/10.1609/aaai.v38i1.27762)

**Abstract**:

The non-consensual exploitation of facial manipulation has emerged as a pressing societal concern. In tandem with the identification of such fake content, recent research endeavors have advocated countering manipulation techniques through proactive interventions, specifically the incorporation of adversarial noise to impede the manipulation in advance. Nevertheless, with insufficient consideration of robustness, we show that current methods falter in providing protection after simple perturbations, e.g., blur. In addition, traditional optimization-based methods face limitations in scalability as they struggle to accommodate the substantial expansion of data volume, a consequence of the time-intensive iterative pipeline. To solve these challenges, we propose a learning-based model, Adversarial Robust Safeguard (ARS), to generate desirable protection noise in a single forward process, concurrently exhibiting a heightened resistance against prevalent perturbations.
Specifically, our method involves a two-way protection design, characterized by a basic protection component responsible for generating efficacious noise features, coupled with robust protection for further enhancement. In robust protection, we first fuse image features with spatially duplicated noise embedding, thereby accounting for inherent information redundancy. Subsequently, a combination comprising a differentiable perturbation module and an adversarial network is devised to simulate potential information degradation during the training process. To evaluate it, we conduct experiments on four manipulation methods and compare recent works comprehensively. The results of our method exhibit good visual effects with pronounced robustness against varied perturbations at different levels.

----

## [14] FlightBERT++: A Non-autoregressive Multi-Horizon Flight Trajectory Prediction Framework

**Authors**: *Dongyue Guo, Zheng Zhang, Zhen Yan, Jianwei Zhang, Yi Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27763](https://doi.org/10.1609/aaai.v38i1.27763)

**Abstract**:

Flight Trajectory Prediction (FTP) is an essential task in Air Traffic Control (ATC), which can assist air traffic controllers in managing airspace more safely and efficiently. Existing approaches generally perform multi-horizon FTP tasks in an autoregressive manner, thereby suffering from error accumulation and low-efficiency problems.  In this paper, a novel framework, called FlightBERT++, is proposed to i) forecast multi-horizon flight trajectories directly in a non-autoregressive way, and ii) improve the limitation of the binary encoding (BE) representation in the FlightBERT. Specifically, the FlightBERT++ is implemented by a generalized encoder-decoder architecture, in which the encoder learns the temporal-spatial patterns from historical observations and the decoder predicts the flight status for the future horizons. Compared with conventional architecture, an innovative horizon-aware contexts generator is dedicatedly designed to consider the prior horizon information, which further enables non-autoregressive multi-horizon prediction. Moreover, a differential prompted decoder is proposed to enhance the capability of the differential predictions by leveraging the stationarity of the differential sequence. The experimental results on a real-world dataset demonstrated that the FlightBERT++ outperformed the competitive baselines in both FTP performance and computational efficiency.

----

## [15] LogFormer: A Pre-train and Tuning Pipeline for Log Anomaly Detection

**Authors**: *Hongcheng Guo, Jian Yang, Jiaheng Liu, Jiaqi Bai, Boyang Wang, Zhoujun Li, Tieqiao Zheng, Bo Zhang, Junran Peng, Qi Tian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27764](https://doi.org/10.1609/aaai.v38i1.27764)

**Abstract**:

Log anomaly detection is a key component in the field of artificial intelligence for IT operations (AIOps). Considering log data of variant domains, retraining the whole network for unknown domains is inefficient in real industrial scenarios. However, previous deep models merely focused on extracting the semantics of log sequences in the same domain, leading to poor generalization on multi-domain logs. To alleviate this issue, we propose a unified Transformer-based framework for Log anomaly detection (LogFormer) to improve the generalization ability across different domains, where we establish a two-stage process including the pre-training and adapter-based tuning stage. Specifically, our model is first pre-trained on the source domain to obtain shared semantic knowledge of log data. Then, we transfer such knowledge to the target domain via shared parameters. Besides, the Log-Attention module is proposed to supplement the information ignored by the log-paring. The proposed method is evaluated on three public datasets and one real-world dataset. Experimental results on multiple benchmarks demonstrate the effectiveness of our LogFormer  with fewer trainable parameters and lower training costs.

----

## [16] ContraNovo: A Contrastive Learning Approach to Enhance De Novo Peptide Sequencing

**Authors**: *Zhi Jin, Sheng Xu, Xiang Zhang, Tianze Ling, Nanqing Dong, Wanli Ouyang, Zhiqiang Gao, Cheng Chang, Siqi Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27765](https://doi.org/10.1609/aaai.v38i1.27765)

**Abstract**:

De novo peptide sequencing from mass spectrometry (MS) data is a critical task in proteomics research. Traditional de novo algorithms have encountered a bottleneck in accuracy due to the inherent complexity of proteomics data. While deep learning-based methods have shown progress, they reduce the problem to a translation task, potentially overlooking critical nuances between spectra and peptides. In our research, we present ContraNovo, a pioneering algorithm that leverages contrastive learning to extract the relationship between spectra and peptides and incorporates the mass information into peptide decoding, aiming to address these intricacies more efficiently. Through rigorous evaluations on two benchmark datasets, ContraNovo consistently outshines contemporary state-of-the-art solutions, underscoring its promising potential in enhancing de novo peptide sequencing.

----

## [17] Inducing Point Operator Transformer: A Flexible and Scalable Architecture for Solving PDEs

**Authors**: *Seungjun Lee, Taeil Oh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27766](https://doi.org/10.1609/aaai.v38i1.27766)

**Abstract**:

Solving partial differential equations (PDEs) by learning the solution operators has emerged as an attractive alternative to traditional numerical methods. However, implementing such architectures presents two main challenges: flexibility in handling irregular and arbitrary input and output formats and scalability to large discretizations. Most existing architectures are limited by their desired structure or infeasible to scale large inputs and outputs. To address these issues, we introduce an attention-based model called an inducing point operator transformer (IPOT). Inspired by inducing points methods, IPOT is designed to handle any input function and output query while capturing global interactions in a computationally efficient way. By detaching the inputs/outputs discretizations from the processor with a smaller latent bottleneck, IPOT offers flexibility in processing arbitrary discretizations and scales linearly with the size of inputs/outputs. Our experimental results demonstrate that IPOT achieves strong performances with manageable computational complexity on an extensive range of PDE benchmarks and real-world weather forecasting scenarios, compared to state-of-the-art methods. Our code is publicly available at https://github.com/7tl7qns7ch/IPOT.

----

## [18] MASTER: Market-Guided Stock Transformer for Stock Price Forecasting

**Authors**: *Tong Li, Zhaoyang Liu, Yanyan Shen, Xue Wang, Haokun Chen, Sen Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27767](https://doi.org/10.1609/aaai.v38i1.27767)

**Abstract**:

Stock price forecasting has remained an extremely challenging problem for many decades due to the high volatility of the stock market. Recent efforts have been devoted to modeling complex stock correlations toward joint stock price forecasting. Existing works share a common neural architecture that learns temporal patterns from individual stock series and then mixes up temporal representations to establish stock correlations. However, they only consider time-aligned stock correlations stemming from all the input stock features, which suffer from two limitations. First, stock correlations often occur momentarily and in a cross-time manner. Second, the feature effectiveness is dynamic with market variation, which affects both the stock sequential patterns and their correlations. To address the limitations, this paper introduces MASTER, a MArkert-guided Stock TransformER, which models the momentary and cross-time stock correlation and leverages market information for automatic feature selection. MASTER elegantly tackles the complex stock correlation by alternatively engaging in intra-stock and inter-stock information aggregation. Experiments show the superiority of MASTER compared with previous works and visualize the captured realistic stock correlation to provide valuable insights.

----

## [19] Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting

**Authors**: *Yanhong Li, Jack Xu, David C. Anastasiu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27768](https://doi.org/10.1609/aaai.v38i1.27768)

**Abstract**:

In the hydrology field, time series forecasting is crucial for efficient water resource management, improving flood and drought control and increasing the safety and quality of life for the general population. However, predicting long-term streamflow is a complex task due to the presence of extreme events. It requires the capture of long-range dependencies and the modeling of rare but important extreme values. Existing approaches often struggle to tackle these dual challenges simultaneously. In this paper, we specifically delve into these issues and propose Distance-weighted Auto-regularized Neural network (DAN), a novel extreme-adaptive model for long-range forecasting of stremflow enhanced by polar representation learning. DAN utilizes a distance-weighted multi-loss mechanism and stackable blocks to dynamically refine indicator sequences from exogenous data, while also being able to handle uni-variate time-series by employing Gaussian Mixture probability modeling to improve robustness to severe events. We also introduce Kruskal-Wallis sampling and gate control vectors to handle imbalanced extreme data. On four real-life hydrologic streamflow datasets, we demonstrate that DAN significantly outperforms both state-of-the-art hydrologic time series prediction methods and general methods designed for long-term time series prediction.

----

## [20] The Causal Impact of Credit Lines on Spending Distributions

**Authors**: *Yijun Li, Cheuk Hang Leung, Xiangqian Sun, Chaoqun Wang, Yiyan Huang, Xing Yan, Qi Wu, Dongdong Wang, Zhixiang Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27769](https://doi.org/10.1609/aaai.v38i1.27769)

**Abstract**:

Consumer credit services offered by electronic commerce platforms provide customers with convenient loan access during shopping and have the potential to stimulate sales. To understand the causal impact of credit lines on spending, previous studies have employed causal estimators, (e.g., direct regression (DR), inverse propensity weighting (IPW), and double machine learning (DML)) to estimate the treatment effect. However, these estimators do not treat the spending of each individual as a distribution that can capture the range and pattern of amounts spent across different orders. By disregarding the outcome as a distribution, valuable insights embedded within the outcome distribution might be overlooked. This paper thus develops distribution valued estimators which extend from existing real valued DR, IPW, and DML estimators within Rubin’s causal framework. We establish their consistency and apply them to a real dataset from a large electronic commerce platform. Our findings reveal that credit lines generally have a positive impact on spending across all quantiles, but consumers would allocate more to luxuries (higher quantiles) than necessities (lower quantiles) as credit lines increase.

----

## [21] Improving PTM Site Prediction by Coupling of Multi-Granularity Structure and Multi-Scale Sequence Representation

**Authors**: *Zhengyi Li, Menglu Li, Lida Zhu, Wen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27770](https://doi.org/10.1609/aaai.v38i1.27770)

**Abstract**:

Protein post-translational modification (PTM) site prediction is a fundamental task in bioinformatics. Several computational methods have been developed to predict PTM sites. However, existing methods ignore the structure information and merely utilize protein sequences. Furthermore, designing a more fine-grained structure representation learning method is urgently needed as PTM is a biological event that occurs at the atom granularity. In this paper, we propose a PTM site prediction method by Coupling of Multi-Granularity structure and Multi-Scale sequence representation, PTM-CMGMS for brevity. Specifically, multigranularity structure-aware representation learning is designed to learn neighborhood structure representations at the amino acid, atom, and whole protein granularity from AlphaFold predicted structures, followed by utilizing contrastive learning to optimize the structure representations. Additionally, multi-scale sequence representation learning is used to extract context sequence information, and motif generated by aligning all context sequences of PTM sites assists the prediction. Extensive experiments on three datasets show that PTM-CMGMS outperforms the state-of-the-art methods. Source code can be found at https://github.com/LZY-HZAU/PTM-CMGMS.

----

## [22] Joint Learning Neuronal Skeleton and Brain Circuit Topology with Permutation Invariant Encoders for Neuron Classification

**Authors**: *Minghui Liao, Guojia Wan, Bo Du*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27771](https://doi.org/10.1609/aaai.v38i1.27771)

**Abstract**:

Determining the types of neurons within a nervous system plays a significant role in the analysis of brain connectomics and the investigation of neurological diseases. However, the efficiency of utilizing anatomical, physiological, or molecular characteristics of neurons is relatively low and costly. With the advancements in electron microscopy imaging and analysis techniques for brain tissue, we are able to obtain whole-brain connectome consisting neuronal high-resolution morphology and connectivity information. However, few models are built based on such data for automated neuron classification. In this paper, we propose NeuNet, a framework that combines morphological information of neurons obtained from skeleton and topological information between neurons obtained from neural circuit. Specifically, NeuNet consists of three components, namely Skeleton Encoder, Connectome Encoder, and Readout Layer. Skeleton Encoder integrates the local information of neurons in a bottom-up manner, with a one-dimensional convolution in neural skeleton's point data; Connectome Encoder uses a graph neural network to capture the topological information of neural circuit; finally, Readout Layer fuses the above two information and outputs classification results. We reprocess and release two new datasets for neuron classification task from volume electron microscopy(VEM) images of human brain cortex and Drosophila brain. Experiments on these two datasets demonstrated the effectiveness of our model with accuracies of 0.9169 and 0.9363, respectively. Code and data are available at: https://github.com/WHUminghui/NeuNet.

----

## [23] Root Cause Analysis in Microservice Using Neural Granger Causal Discovery

**Authors**: *Cheng-Ming Lin, Ching Chang, Wei-Yao Wang, Kuang-Da Wang, Wen-Chih Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27772](https://doi.org/10.1609/aaai.v38i1.27772)

**Abstract**:

In recent years, microservices have gained widespread adoption in IT operations due to their scalability, maintenance, and flexibility. However, it becomes challenging for site reliability engineers (SREs) to pinpoint the root cause due to the complex relationship in microservices when facing system malfunctions. Previous research employed structure learning methods (e.g., PC-algorithm) to establish causal relationships and derive root causes from causal graphs. Nevertheless, they ignored the temporal order of time series data and failed to leverage the rich information inherent in the temporal relationships. For instance, in cases where there is a sudden spike in CPU utilization, it can lead to an increase in latency for other microservices. However, in this scenario, the anomaly in CPU utilization occurs before the latency increases, rather than simultaneously. As a result, the PC-algorithm fails to capture such characteristics. To address these challenges, we propose RUN, a novel approach for root cause analysis using neural Granger causal discovery with contrastive learning. RUN enhances the backbone encoder by integrating contextual information from time series and leverages a time series forecasting model to conduct neural Granger causal discovery. In addition, RUN incorporates Pagerank with a personalization vector to efficiently recommend the top-k root causes. Extensive experiments conducted on the synthetic and real-world microservice-based datasets demonstrate that RUN noticeably outperforms the state-of-the-art root cause analysis methods. Moreover, we provide an analysis scenario for the sock-shop case to showcase the practicality and efficacy of RUN in microservice-based applications. Our code is publicly available at https://github.com/zmlin1998/RUN.

----

## [24] Model-Driven Deep Neural Network for Enhanced AoA Estimation Using 5G gNB

**Authors**: *Shengheng Liu, Xingkang Li, Zihuan Mao, Peng Liu, Yongming Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27773](https://doi.org/10.1609/aaai.v38i1.27773)

**Abstract**:

High-accuracy positioning has become a fundamental enabler for intelligent connected devices. Nevertheless, the present wireless networks still rely on model-driven approaches to achieve positioning functionality, which are susceptible to performance degradation in practical scenarios, primarily due to hardware impairments. Integrating artificial intelligence into the positioning framework presents a promising solution to revolutionize the accuracy and robustness of location-based services. In this study, we address this challenge by reformulating the problem of angle-of-arrival (AoA) estimation into image reconstruction of spatial spectrum. To this end, we design a model-driven deep neural network (MoD-DNN), which can automatically calibrate the angular-dependent phase error. The proposed MoD-DNN approach employs an iterative optimization scheme between a convolutional neural network and a sparse conjugate gradient algorithm. Simulation and experimental results are presented to demonstrate the effectiveness of the proposed method in enhancing spectrum calibration and AoA estimation.

----

## [25] MID-FiLD: MIDI Dataset for Fine-Level Dynamics

**Authors**: *Jesung Ryu, Seungyeon Rhyu, Hong-Gyu Yoon, Eunchong Kim, Ju Young Yang, Taehyun Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27774](https://doi.org/10.1609/aaai.v38i1.27774)

**Abstract**:

One of the challenges in generating human-like music is articulating musical expressions such as dynamics, phrasing, and timbre, which are difficult for computational models to mimic. Previous efforts to tackle this problem have been insufficient due to a fundamental lack of data containing information about musical expressions. In this paper, we introduce MID-FiLD, a MIDI dataset for learning fine-level dynamics control. Notable properties of MID-FiLD are as follows: (1) All 4,422 MIDI samples are constructed by professional music writers with a strong understanding of composition and musical expression. (2) Each MIDI sample contains four different musical metadata and control change \#1 (CC\#1) value. We verify that our metadata is a key factor in MID-FiLD, exerting a substantial influence over produced CC\#1 values. In addition, we demonstrate the applicability of MID-FiLD to deep learning models by suggesting a token-based encoding methodology and reveal the potential for generating controllable, human-like musical expressions.

----

## [26] PosDiffNet: Positional Neural Diffusion for Point Cloud Registration in a Large Field of View with Perturbations

**Authors**: *Rui She, Sijie Wang, Qiyu Kang, Kai Zhao, Yang Song, Wee Peng Tay, Tianyu Geng, Xingchao Jian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27775](https://doi.org/10.1609/aaai.v38i1.27775)

**Abstract**:

Point cloud registration is a crucial technique in 3D computer vision with a wide range of applications. However, this task can be challenging, particularly in large fields of view with dynamic objects, environmental noise, or other perturbations. To address this challenge, we propose a model called PosDiffNet. Our approach performs hierarchical registration based on window-level, patch-level, and point-level correspondence. We leverage a graph neural partial differential equation (PDE) based on Beltrami flow to obtain high-dimensional features and position embeddings for point clouds. We incorporate position embeddings into a Transformer module based on a neural ordinary differential equation (ODE) to efficiently represent patches within points. We employ the multi-level correspondence derived from the high feature similarity scores to facilitate alignment between point clouds. Subsequently, we use registration methods such as SVD-based algorithms to predict the transformation using corresponding point pairs. We evaluate PosDiffNet on several 3D point cloud datasets, verifying that it achieves state-of-the-art (SOTA) performance for point cloud registration in large fields of view with perturbations. The implementation code of experiments is available at https://github.com/AI-IT-AVs/PosDiffNet.

----

## [27] StegaStyleGAN: Towards Generic and Practical Generative Image Steganography

**Authors**: *Wenkang Su, Jiangqun Ni, Yiyan Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27776](https://doi.org/10.1609/aaai.v38i1.27776)

**Abstract**:

The recent advances in generative image steganography have drawn increasing attention due to their potential for provable security and bulk embedding capacity. However, existing generative steganographic schemes are usually tailored for specific tasks and are hardly applied to applications with practical constraints. To address this issue, this paper proposes a generic generative image steganography scheme called Steganography StyleGAN (StegaStyleGAN) that meets the practical objectives of security, capacity, and robustness within the same framework. In StegaStyleGAN, a novel Distribution-Preserving Secret Data Modulator (DP-SDM) is used to achieve provably secure generative image steganography by preserving the data distribution of the model inputs. Additionally, a generic and efficient Secret Data Extractor (SDE) is invented for accurate secret data extraction. By choosing whether to incorporate the Image Attack Simulator (IAS) during the training process, one can obtain two models with different parameters but the same structure (both generator and extractor) for lossless and lossy channel covert communication, namely StegaStyleGAN-Ls and StegaStyleGAN-Ly. Furthermore, by mating with GAN inversion, conditional generative steganography can be achieved as well. Experimental results demonstrate that, whether for lossless or lossy communication channels, the proposed StegaStyleGAN can significantly outperform the corresponding state-of-the-art schemes.

----

## [28] Dual-Channel Learning Framework for Drug-Drug Interaction Prediction via Relation-Aware Heterogeneous Graph Transformer

**Authors**: *Xiaorui Su, Pengwei Hu, Zhu-Hong You, Philip S. Yu, Lun Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27777](https://doi.org/10.1609/aaai.v38i1.27777)

**Abstract**:

Identifying novel drug-drug interactions (DDIs) is a crucial task in pharmacology, as the interference between pharmacological substances can pose serious medical risks. In recent years, several network-based techniques have emerged for predicting DDIs. However, they primarily focus on local structures within DDI-related networks, often overlooking the significance of indirect connections between pairwise drug nodes from a global perspective. Additionally, effectively handling heterogeneous information present in both biomedical knowledge graphs and drug molecular graphs remains a challenge for improved performance of DDI prediction. To address these limitations, we propose a Transformer-based relatIon-aware Graph rEpresentation leaRning framework (TIGER) for DDI prediction. TIGER leverages the Transformer architecture to effectively exploit the structure of heterogeneous graph, which allows it direct learning of long dependencies and high-order structures. Furthermore, TIGER incorporates a relation-aware self-attention mechanism, capturing a diverse range of semantic relations that exist between pairs of nodes in heterogeneous graph. In addition to these advancements, TIGER enhances predictive accuracy by modeling DDI prediction task using a dual-channel network, where drug molecular graph and biomedical knowledge graph are fed into two respective channels. By incorporating embeddings obtained at graph and node levels, TIGER can benefit from structural properties of drugs as well as rich contextual information provided by biomedical knowledge graph. Extensive experiments conducted on three real-world datasets demonstrate the effectiveness of TIGER in DDI prediction. Furthermore, case studies highlight its ability to provide a deeper understanding of underlying mechanisms of DDIs.

----

## [29] Molecular Optimization Model with Patentability Constraint

**Authors**: *Sally Turutov, Kira Radinsky*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27778](https://doi.org/10.1609/aaai.v38i1.27778)

**Abstract**:

In drug development, molecular optimization is a crucial challenge that involves generating novel molecules given a lead molecule as input. The task requires maintaining molecular similarity to the original molecule while simultaneously optimizing multiple chemical attributes. To aid in this process, numerous generative models have been proposed.
However, in practical applications, it is crucial for these models not only to generate novel molecules with the above constraints but also to generate molecules that significantly differ from any existing patented compounds. 
In this work, we present a multi-optimization molecular framework to address this challenge.
Our framework trains a model to prioritize both enhanced properties and substantial dissimilarity from patented compounds. By jointly learning continuous representations of optimized and patentable molecules, we ensure that the generated molecules are significantly distant from any patented compounds while improving chemical properties.
Through empirical evaluation, we demonstrate the superior performance of our approach compared to state-of-the-art molecular optimization methods both in chemical property optimization and patentability.

----

## [30] Generalizable Sleep Staging via Multi-Level Domain Alignment

**Authors**: *Jiquan Wang, Sha Zhao, Haiteng Jiang, Shijian Li, Tao Li, Gang Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27779](https://doi.org/10.1609/aaai.v38i1.27779)

**Abstract**:

Automatic sleep staging is essential for sleep assessment and disorder diagnosis. Most existing methods depend on one specific dataset and are limited to be generalized to other unseen datasets, for which the training data and testing data are from the same dataset. In this paper, we introduce domain generalization into automatic sleep staging and propose the task of generalizable sleep staging which aims to improve the model generalization ability to unseen datasets. Inspired by existing domain generalization methods, we adopt the feature alignment idea and propose a framework called SleepDG to solve it. Considering both of local salient features and sequential features are important for sleep staging, we propose a Multi-level Feature Alignment combining epoch-level and sequence-level feature alignment to learn domain-invariant feature representations. Specifically, we design an Epoch-level Feature Alignment to align the feature distribution of each single sleep epoch among different domains, and a Sequence-level Feature Alignment to minimize the discrepancy of sequential features among different domains. SleepDG is validated on five public datasets, achieving the state-of-the-art performance.

----

## [31] Inspecting Prediction Confidence for Detecting Black-Box Backdoor Attacks

**Authors**: *Tong Wang, Yuan Yao, Feng Xu, Miao Xu, Shengwei An, Ting Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27780](https://doi.org/10.1609/aaai.v38i1.27780)

**Abstract**:

Backdoor attacks have been shown to be a serious security threat against deep learning models, and various defenses have been proposed to detect whether a model is backdoored or not. However, as indicated by a recent black-box attack, existing defenses can be easily bypassed by implanting the backdoor in the frequency domain.
To this end, we propose a new defense DTInspector against black-box backdoor attacks, based on a new observation related to the prediction confidence of learning models. That is, to achieve a high attack success rate with a small amount of poisoned data, backdoor attacks usually render a model exhibiting statistically higher prediction confidences on the poisoned samples. We provide both theoretical and empirical evidence for the generality of this observation. DTInspector then carefully examines the prediction confidences of data samples, and decides the existence of backdoor using the shortcut nature of backdoor triggers. Extensive evaluations on six backdoor attacks, four datasets, and three advanced attacking types demonstrate the effectiveness of the proposed defense.

----

## [32] Conformal Crystal Graph Transformer with Robust Encoding of Periodic Invariance

**Authors**: *Yingheng Wang, Shufeng Kong, John M. Gregoire, Carla P. Gomes*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27781](https://doi.org/10.1609/aaai.v38i1.27781)

**Abstract**:

Machine learning techniques, especially in the realm of materials design, hold immense promise in predicting the properties of crystal materials and aiding in the discovery of novel crystals with desirable traits. However, crystals possess unique geometric constraints—namely, E(3) invariance for primitive cell and periodic invariance—which need to be accurately reflected in crystal representations. Though past research has explored various construction techniques to preserve periodic invariance in crystal representations, their robustness remains inadequate. Furthermore, effectively capturing angular information within 3D crystal structures continues to pose a significant challenge for graph-based approaches. This study introduces novel solutions to these challenges. We first present a graph construction method that robustly encodes periodic invariance and a strategy to capture angular information in neural networks without compromising efficiency. We further introduce CrystalFormer, a pioneering graph transformer architecture that emphasizes angle preservation and enhances long-range information. Through comprehensive evaluation, we verify our model's superior performance in 5 crystal prediction tasks, reaffirming the efficiency of our proposed methods.

----

## [33] SuperJunction: Learning-Based Junction Detection for Retinal Image Registration

**Authors**: *Yu Wang, Xiaoye Wang, Zaiwang Gu, Weide Liu, Wee Siong Ng, Weimin Huang, Jun Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27782](https://doi.org/10.1609/aaai.v38i1.27782)

**Abstract**:

Keypoints-based approaches have shown to be promising for retinal image registration, which superimpose two or more images from different views based on keypoint detection and description. However, existing approaches suffer from ineffective keypoint detector and descriptor training. Meanwhile, the non-linear mapping from 3D retinal structure to 2D images is often neglected. In this paper, we propose a novel learning-based junction detection approach for retinal image registration, which enhances both the keypoint detector and  descriptor training. To improve the keypoint detection, it uses a multi-task vessel detection to regularize the model training, which helps to learn more representative features and reduce the risk of over-fitting. To achieve effective training for keypoints description, a new constrained negative sampling approach is proposed to compute the descriptor loss. Moreover,  we also consider the non-linearity between retinal images from different views during matching. Experimental results on FIRE dataset show that our method achieves mean area under curve of 0.850, which is 12.6% higher than 0.755 by the state-of-the-art method. All the codes are available at https://github.com/samjcheng/SuperJunction.

----

## [34] Explore 3D Dance Generation via Reward Model from Automatically-Ranked Demonstrations

**Authors**: *Zilin Wang, Haolin Zhuang, Lu Li, Yinmin Zhang, Junjie Zhong, Jun Chen, Yu Yang, Boshi Tang, Zhiyong Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27783](https://doi.org/10.1609/aaai.v38i1.27783)

**Abstract**:

This paper presents an Exploratory 3D Dance generation framework, E3D2, designed to address the exploration capability deficiency in existing music-conditioned 3D dance generation models. Current models often generate monotonous and simplistic dance sequences that misalign with human preferences because they lack exploration capabilities.The E3D2 framework involves a reward model trained from automatically-ranked dance demonstrations, which then guides the reinforcement learning process. This approach encourages the agent to explore and generate high quality and diverse dance movement sequences. The soundness of the reward model is both theoretically and experimentally validated. Empirical experiments demonstrate the effectiveness of E3D2 on the AIST++ dataset.

----

## [35] PSC-CPI: Multi-Scale Protein Sequence-Structure Contrasting for Efficient and Generalizable Compound-Protein Interaction Prediction

**Authors**: *Lirong Wu, Yufei Huang, Cheng Tan, Zhangyang Gao, Bozhen Hu, Haitao Lin, Zicheng Liu, Stan Z. Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27784](https://doi.org/10.1609/aaai.v38i1.27784)

**Abstract**:

Compound-Protein Interaction (CPI) prediction aims to predict the pattern and strength of compound-protein interactions for rational drug discovery. Existing deep learning-based methods utilize only the single modality of protein sequences or structures and lack the co-modeling of the joint distribution of the two modalities, which may lead to significant performance drops in complex real-world scenarios due to various factors, e.g., modality missing and domain shifting. More importantly, these methods only model protein sequences and structures at a single fixed scale, neglecting more fine-grained multi-scale information, such as those embedded in key protein fragments. In this paper, we propose a novel multi-scale Protein Sequence-structure Contrasting framework for CPI prediction (PSC-CPI), which captures the dependencies between protein sequences and structures through both intra-modality and cross-modality contrasting. We further apply length-variable protein augmentation to allow contrasting to be performed at different scales, from the amino acid level to the sequence level. Finally, in order to more fairly evaluate the model generalizability, we split the test data into four settings based on whether compounds and proteins have been observed during the training stage. Extensive experiments have shown that PSC-CPI generalizes well in all four settings, particularly in the more challenging ``Unseen-Both" setting, where neither compounds nor proteins have been observed during training. Furthermore, even when encountering a situation of modality missing, i.e., inference with only single-modality protein data, PSC-CPI still exhibits comparable or even better performance than previous approaches.

----

## [36] Uncertainty Quantification for Forward and Inverse Problems of PDEs via Latent Global Evolution

**Authors**: *Tailin Wu, Willie Neiswanger, Hongtao Zheng, Stefano Ermon, Jure Leskovec*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27785](https://doi.org/10.1609/aaai.v38i1.27785)

**Abstract**:

Deep learning-based surrogate models have demonstrated remarkable advantages over classical solvers in terms of speed, often achieving speedups of 10 to 1000 times over traditional partial differential equation (PDE) solvers. However, a significant challenge hindering their widespread adoption in both scientific and industrial domains is the lack of understanding about their prediction uncertainties, particularly in scenarios that involve critical decision making. To address this limitation, we propose a method that integrates efficient and precise uncertainty quantification into a deep learning-based surrogate model. Our method, termed Latent Evolution of PDEs with Uncertainty Quantification (LE-PDE-UQ), endows deep learning-based surrogate models with robust and efficient uncertainty quantification capabilities for both forward and inverse problems. LE-PDE-UQ leverages latent vectors within a latent space to evolve both the system's state and its corresponding uncertainty estimation. The latent vectors are decoded to provide predictions for the system's state as well as estimates of its uncertainty. In extensive experiments, we demonstrate the accurate uncertainty quantification performance of our approach, surpassing that of strong baselines including deep ensembles, Bayesian neural network layers, and dropout. Our method excels at propagating uncertainty over extended auto-regressive rollouts, making it suitable for scenarios involving long-term predictions. Our code is available at: https://github.com/AI4Science-WestlakeU/le-pde-uq.

----

## [37] Multilevel Attention Network with Semi-supervised Domain Adaptation for Drug-Target Prediction

**Authors**: *Zhousan Xie, Shikui Tu, Lei Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27786](https://doi.org/10.1609/aaai.v38i1.27786)

**Abstract**:

Prediction of drug-target interactions (DTIs) is a crucial step in drug discovery, and deep learning methods have shown great promise on various DTI datasets. However, existing approaches still face several challenges, including limited labeled data, hidden bias issue, and a lack of generalization ability to out-of-domain data. These challenges hinder the model's capacity to learn truly informative interaction features, leading to shortcut learning and inferior predictive performance on novel drug-target pairs. To address these issues, we propose MlanDTI, a semi-supervised domain adaptive multilevel attention network (Mlan) for DTI prediction. We utilize two pre-trained BERT models to acquire bidirectional representations enriched with information from unlabeled data. Then, we introduce a multilevel attention mechanism, enabling the model to learn domain-invariant DTIs at different hierarchical levels. Moreover, we present a simple yet effective semi-supervised pseudo-labeling method to further enhance our model's predictive ability in cross-domain scenarios. Experiments on four datasets show that MlanDTI achieves state-of-the-art performances over other methods under intra-domain settings and outperforms all other approaches under cross-domain settings. The source code is available at https://github.com/CMACH508/MlanDTI.

----

## [38] Geometric-Facilitated Denoising Diffusion Model for 3D Molecule Generation

**Authors**: *Can Xu, Haosen Wang, Weigang Wang, Pengfei Zheng, Hongyang Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27787](https://doi.org/10.1609/aaai.v38i1.27787)

**Abstract**:

Denoising diffusion models have shown great potential in multiple research areas. Existing diffusion-based generative methods on de novo 3D molecule generation face two major challenges. Since majority heavy atoms in molecules allow connections to multiple atoms through single bonds, solely using pair-wise distance to model molecule geometries is insufficient. Therefore, the first one involves proposing an effective neural network as the denoising kernel that is capable to capture complex multi-body interatomic relationships and learn high-quality features. Due to the discrete nature of graphs, mainstream diffusion-based methods for molecules heavily rely on predefined rules and generate edges in an indirect manner. The second challenge involves accommodating molecule generation to diffusion and accurately predicting the existence of bonds. In our research, we view the iterative way of updating molecule conformations in diffusion process is consistent with molecular dynamics and introduce a novel molecule generation method named Geometric-Facilitated Molecular Diffusion (GFMDiff). For the first challenge, we introduce a Dual-track Transformer Network (DTN) to fully excevate global spatial relationships and learn high quality representations which contribute to accurate predictions of features and geometries. As for the second challenge, we design Geometric-facilitated Loss (GFLoss) which intervenes the formation of bonds during the training period, instead of directly embedding edges into the latent space. Comprehensive experiments on current benchmarks demonstrate the superiority of GFMDiff.

----

## [39] GAMC: An Unsupervised Method for Fake News Detection Using Graph Autoencoder with Masking

**Authors**: *Shu Yin, Peican Zhu, Lianwei Wu, Chao Gao, Zhen Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27788](https://doi.org/10.1609/aaai.v38i1.27788)

**Abstract**:

With the rise of social media, the spread of fake news has become a significant concern, potentially misleading public perceptions and impacting social stability. Although deep learning methods like CNNs, RNNs, and Transformer-based models like BERT have enhanced fake news detection. However, they primarily focus on content and do not consider social context during news propagation. Graph-based techniques have incorporated the social context but are limited by the need for large labeled datasets. To address these challenges, this paper introduces GAMC, an unsupervised fake news detection technique using the Graph Autoencoder with Masking and Contrastive learning. By leveraging both the context and content of news propagation as self-supervised signals, our method reduces the dependency on labeled datasets. Specifically, GAMC begins by applying data augmentation to the original news propagation graphs. Subsequently, these augmented graphs are encoded using a graph encoder and subsequently reconstructed via a graph decoder. Finally, a composite loss function that encompasses both reconstruction error and contrastive loss is designed. Firstly, it ensures the model can effectively capture the latent features, based on minimizing the discrepancy between reconstructed and original graph representations. Secondly, it aligns the representations of augmented graphs that originate from the same source. Experiments on the real-world dataset validate the effectiveness of our method.

----

## [40] Unsupervised Gene-Cell Collective Representation Learning with Optimal Transport

**Authors**: *Jixiang Yu, Nanjun Chen, Ming Gao, Xiangtao Li, Ka-Chun Wong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27789](https://doi.org/10.1609/aaai.v38i1.27789)

**Abstract**:

Cell type identification plays a vital role in single-cell RNA sequencing (scRNA-seq) data analysis. Although many deep embedded methods to cluster scRNA-seq data have been proposed, they still fail in elucidating the intrinsic properties of cells and genes. Here, we present a novel end-to-end deep graph clustering model for single-cell transcriptomics data based on unsupervised Gene-Cell Collective representation learning and Optimal Transport (scGCOT) which integrates both cell and gene correlations. Specifically, scGCOT learns the latent embedding of cells and genes simultaneously and reconstructs the cell graph, the gene graph, and the gene expression count matrix. A zero-inflated negative binomial (ZINB) model is estimated via the reconstructed count matrix to capture the essential properties of scRNA-seq data. By leveraging the optimal transport-based joint representation alignment, scGCOT learns the clustering process and the latent representations through a mutually supervised self optimization strategy. Extensive experiments with 14 competing methods on 15 real scRNA-seq datasets demonstrate the competitive edges of scGCOT.

----

## [41] MCSSME: Multi-Task Contrastive Learning for Semi-supervised Singing Melody Extraction from Polyphonic Music

**Authors**: *Shuai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27790](https://doi.org/10.1609/aaai.v38i1.27790)

**Abstract**:

Singing melody extraction is an important task in the field of music information retrieval (MIR). The development of data-driven models for this task have achieved great successes.  However, the existing models have two major limitations: firstly, most of the existing singing melody extraction models have formulated this task as a pixel-level prediction task. The lack of labeling data has limited the model for further improvements. Secondly, the generalization of the existing models are prone to be disturbed by the music genres. To address the issues mentioned above, in this paper, we propose a multi-Task contrastive learning framework for semi-supervised singing melody extraction, termed as MCSSME. 
Specifically, to deal with data scarcity limitation, we propose a self-consistency regularization (SCR) method to train the model on the unlabeled data. Transformations are applied to the raw signal of polyphonic music, which makes the network to improve its representation capability via recognizing the transformations. We further propose a novel multi-task learning (MTL) approach to jointly learn singing melody extraction and classification of transformed data. To deal with generalization limitation, we also propose a contrastive embedding learning, which strengthens the intra-class compactness and inter-class separability. To improve the generalization on different music genres, we also propose a domain classification method to learn task-dependent features by mapping data from different music genres to shared subspace.  MCSSME evaluates on a set of well-known public melody extraction datasets with promising performances. The experimental results demonstrate the effectiveness of the MCSSME framework for singing melody extraction from polyphonic music using very limited labeled data scenarios.

----

## [42] RetroOOD: Understanding Out-of-Distribution Generalization in Retrosynthesis Prediction

**Authors**: *Yemin Yu, Luotian Yuan, Ying Wei, Hanyu Gao, Fei Wu, Zhihua Wang, Xinhai Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27791](https://doi.org/10.1609/aaai.v38i1.27791)

**Abstract**:

Machine learning-assisted retrosynthesis prediction models have been gaining widespread adoption, though their performances oftentimes degrade significantly when deployed in real-world applications embracing out-of-distribution (OOD) molecules or reactions. Despite steady progress on standard benchmarks, our understanding of existing retrosynthesis prediction models under the premise of distribution shifts remains stagnant. To this end, we first formally sort out two types of distribution shifts in retrosynthesis prediction and construct two groups of benchmark datasets. Next, through comprehensive experiments, we systematically compare state-of-the-art retrosynthesis prediction models on the two groups of benchmarks, revealing the limitations of previous in-distribution evaluation and re-examining the advantages of each model. More remarkably, we are motivated by the above empirical insights to propose two model-agnostic techniques that can improve the OOD generalization of arbitrary off-the-shelf retrosynthesis prediction algorithms. Our preliminary experiments show their high potential with an average performance improvement of 4.6%, and the established benchmarks serve as a foothold for further retrosynthesis prediction research towards OOD generalization.

----

## [43] Designing Biological Sequences without Prior Knowledge Using Evolutionary Reinforcement Learning

**Authors**: *Xi Zeng, Xiaotian Hao, Hongyao Tang, Zhentao Tang, Shaoqing Jiao, Dazhi Lu, Jiajie Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27792](https://doi.org/10.1609/aaai.v38i1.27792)

**Abstract**:

Designing novel biological sequences with desired properties is a significant challenge in biological science because of the extra large search space. The traditional design process usually involves multiple rounds of costly wet lab evaluations. To reduce the need for expensive wet lab experiments, machine learning methods are used to aid in designing biological sequences. However, the limited availability of biological sequences with known properties hinders the training of machine learning models, significantly restricting their applicability and performance. To fill this gap, we present ERLBioSeq, an Evolutionary Reinforcement Learning algorithm for BIOlogical SEQuence design. ERLBioSeq leverages the capability of reinforcement learning to learn without prior knowledge and the potential of evolutionary algorithms to enhance the exploration of reinforcement learning in the large search space of biological sequences. Additionally, to enhance the efficiency of biological sequence design, we developed a predictor for sequence screening in the biological sequence design process, which incorporates both the local and global sequence information. We evaluated the proposed method on three main types of biological sequence design tasks, including the design of DNA, RNA, and protein. The results demonstrate that the proposed method achieves significant improvement compared to the existing state-of-the-art methods.

----

## [44] Adversarial Socialbots Modeling Based on Structural Information Principles

**Authors**: *Xianghua Zeng, Hao Peng, Angsheng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27793](https://doi.org/10.1609/aaai.v38i1.27793)

**Abstract**:

The importance of effective detection is underscored by the fact that socialbots imitate human behavior to propagate misinformation, leading to an ongoing competition between socialbots and detectors. Despite the rapid advancement of reactive detectors, the exploration of adversarial socialbot modeling remains incomplete, significantly hindering the development of proactive detectors. To address this issue, we propose a mathematical Structural Information principles-based Adversarial Socialbots Modeling framework, namely SIASM, to enable more accurate and effective modeling of adversarial behaviors. First, a heterogeneous graph is presented to integrate various users and rich activities in the original social network and measure its dynamic uncertainty as structural entropy. By minimizing the high-dimensional structural entropy, a hierarchical community structure of the social network is generated and referred to as the optimal encoding tree. Secondly, a novel method is designed to quantify influence by utilizing the assigned structural entropy, which helps reduce the computational cost of SIASM by filtering out uninfluential users. Besides, a new conditional structural entropy is defined between the socialbot and other users to guide the follower selection for network influence maximization. Extensive and comparative experiments on both homogeneous and heterogeneous social networks demonstrate that, compared with state-of-the-art baselines, the proposed SIASM framework yields substantial performance improvements in terms of network influence (up to 16.32%) and sustainable stealthiness (up to 16.29%) when evaluated against a robust detector with 90% accuracy.

----

## [45] NondBREM: Nondeterministic Offline Reinforcement Learning for Large-Scale Order Dispatching

**Authors**: *Hongbo Zhang, Guang Wang, Xu Wang, Zhengyang Zhou, Chen Zhang, Zheng Dong, Yang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27794](https://doi.org/10.1609/aaai.v38i1.27794)

**Abstract**:

One of the most important tasks in ride-hailing is order dispatching, i.e., assigning unserved orders to available drivers. Recent order dispatching has achieved a significant improvement due to the advance of reinforcement learning, which has been approved to be able to effectively address sequential decision-making problems like order dispatching. However, most existing reinforcement learning methods require agents to learn the optimal policy by interacting with environments online, which is challenging or impractical for real-world deployment due to high costs or safety concerns. For example, due to the spatiotemporally unbalanced supply and demand, online reinforcement learning-based order dispatching may significantly impact the revenue of the ride-hailing platform and passenger experience during the policy learning period. Hence, in this work, we develop an offline deep reinforcement learning framework called NondBREM for large-scale order dispatching, which learns policy from only the accumulated logged data to avoid costly and unsafe interactions with the environment. In NondBREM, a Nondeterministic Batch-Constrained Q-learning (NondBCQ) module is developed to reduce the algorithm extrapolation error and a Random Ensemble Mixture (REM) module that integrates multiple value networks with multi-head networks is utilized to improve the model generalization and robustness. Extensive experiments on large-scale real-world ride-hailing datasets show the superiority of our design.

----

## [46] Scale Optimization Using Evolutionary Reinforcement Learning for Object Detection on Drone Imagery

**Authors**: *Jialu Zhang, Xiaoying Yang, Wentao He, Jianfeng Ren, Qian Zhang, Yitian Zhao, Ruibin Bai, Xiangjian He, Jiang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27795](https://doi.org/10.1609/aaai.v38i1.27795)

**Abstract**:

Object detection in aerial imagery presents a significant challenge due to large scale variations among objects. This paper proposes an evolutionary reinforcement learning agent, integrated within a coarse-to-fine object detection framework, to optimize the scale for more effective detection of objects in such images. Specifically, a set of patches potentially containing objects are first generated. A set of rewards measuring the localization accuracy, the accuracy of predicted labels, and the scale consistency among nearby patches are designed in the agent to guide the scale optimization. The proposed scale-consistency reward ensures similar scales for neighboring objects of the same category. Furthermore, a spatial-semantic attention mechanism is designed to exploit the spatial semantic relations between patches. The agent employs the proximal policy optimization strategy in conjunction with the evolutionary strategy, effectively utilizing both the current patch status and historical experience embedded in the agent. The proposed model is compared with state-of-the-art methods on two benchmark datasets for object detection on drone imagery. It significantly outperforms all the compared methods. Code is available at https://github.com/UNNC-CV/EvOD/.

----

## [47] Adversarial Attacks on Federated-Learned Adaptive Bitrate Algorithms

**Authors**: *Rui-Xiao Zhang, Tianchi Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27796](https://doi.org/10.1609/aaai.v38i1.27796)

**Abstract**:

Learning-based adaptive bitrate (ABR) algorithms have revolutionized video streaming solutions. With the growing demand for data privacy and the rapid development of mobile devices, federated learning (FL) has emerged as a popular training method for neural ABR algorithms in both academia and industry. However, we have discovered that FL-based ABR models are vulnerable to model-poisoning attacks as local updates remain unseen during global aggregation. In response, we propose MAFL (Malicious ABR model based on Federated Learning) to prove that backdooring the learning-based ABR model via FL is practical. Instead of attacking the global policy, MAFL only targets a single ``target client''. Moreover, the unique challenges brought by deep reinforcement learning (DRL) make the attack even more challenging. To address these challenges, MAFL is designed with a two-stage attacking mechanism. Using two representative attack cases with real-world traces, we show that MAFL significantly degrades the model performance on the target client (i.e., increasing rebuffering penalty by 2x and 5x) with a minimal negative impact on benign clients.

----

## [48] Generalize for Future: Slow and Fast Trajectory Learning for CTR Prediction

**Authors**: *Jian Zhu, Congcong Liu, Xue Jiang, Changping Peng, Zhangang Lin, Jingping Shao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27797](https://doi.org/10.1609/aaai.v38i1.27797)

**Abstract**:

Deep neural networks (DNNs) have achieved significant advancements in click-through rate (CTR) prediction by demonstrating strong generalization on training data. However, in real-world scenarios, the assumption of independent and identically distributed (i.i.d.) conditions, which is fundamental to this problem, is often violated due to temporal distribution shifts. This violation can lead to suboptimal model performance when optimizing empirical risk without access to future data, resulting in overfitting on the training data and convergence to a single sharp minimum. To address this challenge, we propose a novel model updating framework called Slow and Fast Trajectory Learning (SFTL) network. SFTL aims to mitigate the discrepancy between past and future domains while quickly adapting to recent changes in small temporal drifts. This mechanism entails two interactions among three complementary learners: (i) the Working Learner, which updates model parameters using modern optimizers (e.g., Adam, Adagrad) and serves as the primary learner in the recommendation system, (ii) the Slow Learner, which is updated in each temporal domain by directly assigning the model weights of the working learner, and (iii) the Fast Learner, which is updated in each iteration by assigning exponentially moving average weights of the working learner. Additionally, we propose a novel rank-based trajectory loss to facilitate interaction between the working learner and trajectory learner, aiming to adapt to temporal drift and enhance performance in the current domain compared to the past. We provide theoretical understanding and conduct extensive experiments on real-world CTR prediction datasets to validate the effectiveness and efficiency of SFTL in terms of both convergence speed and model performance. The results demonstrate the superiority of SFTL over existing approaches.

----

## [49] Hot or Cold? Adaptive Temperature Sampling for Code Generation with Large Language Models

**Authors**: *Yuqi Zhu, Jia Li, Ge Li, Yunfei Zhao, Jia Li, Zhi Jin, Hong Mei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27798](https://doi.org/10.1609/aaai.v38i1.27798)

**Abstract**:

Recently, Large Language Models (LLMs) have shown impressive abilities in code generation. However, existing LLMs' decoding strategies are designed for Natural Language (NL) generation, overlooking the differences between NL and programming languages (PL). Due to this oversight, a better decoding strategy for code generation remains an open question. In this paper, we conduct the first systematic study to explore a decoding strategy specialized in code generation. With an analysis of loss distributions of code tokens, we find that code tokens can be divided into two categories: challenging tokens that are difficult to predict and confident tokens that can be easily inferred. Among them, the challenging tokens mainly appear at the beginning of a code block. Inspired by the above findings, we propose a simple yet effective method: Adaptive Temperature (AdapT) sampling, which dynamically adjusts the temperature coefficient when decoding different tokens. We apply a larger temperature when sampling for challenging tokens, allowing LLMs to explore diverse choices. We employ a smaller temperature for confident tokens avoiding the influence of tail randomness noises. We apply AdapT sampling to LLMs with different sizes and conduct evaluations on two popular datasets. Results show that AdapT sampling significantly outperforms state-of-the-art decoding strategy.

----

## [50] Operationalizing Essential Characteristics of Creativity in a Computational System for Music Composition

**Authors**: *Paul M. Bodily, Dan Ventura*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27799](https://doi.org/10.1609/aaai.v38i1.27799)

**Abstract**:

We address the problem of building and evaluating a computational system whose primary objective is creativity. We illustrate seven characteristics for computational creativity in the context of a system that autonomously composes Western lyrical music. We conduct an external evaluation of the system in which respondents rated the system with regard to each characteristic as well as with regard to overall creativity. Average scores for overall creativity exceeded the ratings for any single characteristic, suggesting that creativity may be an emergent property and that unique research opportunities exist for building CC systems whose design attempts to comprehend all known characteristics of creativity.

----

## [51] Neural Reasoning about Agents' Goals, Preferences, and Actions

**Authors**: *Matteo Bortoletto, Lei Shi, Andreas Bulling*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27800](https://doi.org/10.1609/aaai.v38i1.27800)

**Abstract**:

We propose the Intuitive Reasoning Network (IRENE) - a novel neural model for intuitive psychological reasoning about agents' goals, preferences, and actions that can generalise previous experiences to new situations. IRENE combines a graph neural network for learning agent and world state representations with a transformer to encode the task context. When evaluated on the challenging Baby Intuitions Benchmark, IRENE achieves new state-of-the-art performance on three out of its five tasks - with up to 48.9% improvement. In contrast to existing methods, IRENE is able to bind preferences to specific agents, to better distinguish between rational and irrational agents, and to better understand the role of blocking obstacles. We also investigate, for the first time, the influence of the training tasks on test performance. Our analyses demonstrate the effectiveness of IRENE in combining prior knowledge gained during training for unseen evaluation tasks.

----

## [52] An Empirical Study of CLIP for Text-Based Person Search

**Authors**: *Min Cao, Yang Bai, Ziyin Zeng, Mang Ye, Min Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27801](https://doi.org/10.1609/aaai.v38i1.27801)

**Abstract**:

Text-based Person Search (TBPS) aims to retrieve the person images using natural language descriptions. Recently, Contrastive Language Image Pretraining (CLIP), a universal large cross-modal vision-language pre-training model, has remarkably performed over various cross-modal downstream tasks due to its powerful cross-modal semantic learning capacity. TPBS, as a fine-grained cross-modal retrieval task, is also facing the rise of research on the CLIP-based TBPS. In order to explore the potential of the visual-language pre-training model for downstream TBPS tasks, this paper makes the first attempt to conduct a comprehensive empirical study of CLIP for TBPS and thus contribute a straightforward, incremental, yet strong TBPS-CLIP baseline to the TBPS community. We revisit critical design considerations under CLIP, including data augmentation and loss function. The model, with the aforementioned designs and practical training tricks, can attain satisfactory performance without any sophisticated modules. Also, we conduct the probing experiments of TBPS-CLIP in model generalization and model compression, demonstrating the effectiveness of TBPS-CLIP from various aspects. This work is expected to provide empirical insights and highlight future CLIP-based TBPS research.

----

## [53] Social Physics Informed Diffusion Model for Crowd Simulation

**Authors**: *Hongyi Chen, Jingtao Ding, Yong Li, Yue Wang, Xiao-Ping Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27802](https://doi.org/10.1609/aaai.v38i1.27802)

**Abstract**:

Crowd simulation holds crucial applications in various domains, such as urban planning, architectural design, and traffic arrangement. In recent years, physics-informed machine learning methods have achieved state-of-the-art performance in crowd simulation but fail to model the heterogeneity and multi-modality of human movement comprehensively. In this paper, we propose a social physics-informed diffusion model named SPDiff to mitigate the above gap. SPDiff takes both the interactive and historical information of crowds in the current timeframe to reverse the diffusion process, thereby generating the distribution of pedestrian movement in the subsequent timeframe. Inspired by the well-known social physics model, i.e., Social Force, regarding crowd dynamics, we design a crowd interaction encoder to guide the denoising process and further enhance this module with the equivariant properties of crowd interactions. To mitigate error accumulation in long-term simulations, we propose a multi-frame rollout training algorithm for diffusion modeling. Experiments conducted on two real-world datasets demonstrate the superior performance of SPDiff in terms of both macroscopic and microscopic evaluation metrics. Code and appendix are available at https://github.com/tsinghua-fib-lab/SPDiff.

----

## [54] Trend-Aware Supervision: On Learning Invariance for Semi-supervised Facial Action Unit Intensity Estimation

**Authors**: *Yingjie Chen, Jiarui Zhang, Tao Wang, Yun Liang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27803](https://doi.org/10.1609/aaai.v38i1.27803)

**Abstract**:

With the increasing need for facial behavior analysis, semi-supervised AU intensity estimation using only keyframe annotations has emerged as a practical and effective solution to relieve the burden of annotation. However, the lack of annotations makes the spurious correlation problem caused by AU co-occurrences and subject variation much more prominent, leading to non-robust intensity estimation that is entangled among AUs and biased among subjects. We observe that trend information inherent in keyframe annotations could act as extra supervision and raising the awareness of AU-specific facial appearance changing trends during training is the key to learning invariant AU-specific features. To this end, we propose Trend-AwareSupervision (TAS), which pursues three kinds of trend awareness, including intra-trend ranking awareness, intra-trend speed awareness, and inter-trend subject awareness. TAS alleviates the spurious correlation problem by raising trend awareness during training to learn AU-specific features that represent the corresponding facial appearance changes, to achieve intensity estimation invariance. Experiments conducted on two commonly used AU benchmark datasets, BP4D and DISFA, show the effectiveness of each kind of awareness. And under trend-aware supervision, the performance can be improved without extra computational or storage costs during inference.

----

## [55] Enhancing the Robustness of Spiking Neural Networks with Stochastic Gating Mechanisms

**Authors**: *Jianhao Ding, Zhaofei Yu, Tiejun Huang, Jian K. Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27804](https://doi.org/10.1609/aaai.v38i1.27804)

**Abstract**:

Spiking neural networks (SNNs) exploit neural spikes to provide solutions for low-power intelligent applications on neuromorphic hardware. Although SNNs have high computational efficiency due to spiking communication, they still lack resistance to adversarial attacks and noise perturbations. In the brain, neuronal responses generally possess stochasticity induced by ion channels and synapses, while the role of stochasticity in computing tasks is poorly understood. Inspired by this, we elaborate a stochastic gating spiking neural model for layer-by-layer spike communication, introducing stochasticity to SNNs. Through theoretical analysis, our gating model can be viewed as a regularizer that prevents error amplification under attacks. Meanwhile, our work can explain the robustness of Poisson coding. Experimental results prove that our method can be used alone or with existing robust enhancement algorithms to improve SNN robustness and reduce SNN energy consumption. We hope our work will shed new light on the role of stochasticity in the computation of SNNs. Our code is available at https://github.com/DingJianhao/StoG-meets-SNN/.

----

## [56] Imitation of Life: A Search Engine for Biologically Inspired Design

**Authors**: *Hen Emuna, Nadav Borenstein, Xin Qian, Hyeonsu B. Kang, Joel Chan, Aniket Kittur, Dafna Shahaf*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27805](https://doi.org/10.1609/aaai.v38i1.27805)

**Abstract**:

Biologically Inspired Design (BID), or Biomimicry, is a problem-solving methodology that applies analogies from nature to solve engineering challenges. For example, Speedo engineers designed swimsuits based on shark skin. Finding relevant biological solutions for real-world problems poses significant challenges, both due to the limited biological knowledge engineers and designers typically possess and to the limited BID resources. Existing BID datasets are hand-curated and small, and scaling them up requires costly human annotations.

In this paper, we introduce BARcode (Biological Analogy Retriever), a search engine for automatically mining bio-inspirations from the web at scale. Using advances in natural language understanding and data programming, BARcode identifies potential inspirations for engineering challenges. Our experiments demonstrate that BARcode can retrieve inspirations that are valuable to engineers and designers tackling real-world problems, as well as recover famous historical BID examples. We release data and code; we view BARcode as a step towards addressing the challenges that have historically hindered the practical application of BID to engineering innovation.

----

## [57] An Efficient Knowledge Transfer Strategy for Spiking Neural Networks from Static to Event Domain

**Authors**: *Xiang He, Dongcheng Zhao, Yang Li, Guobin Shen, Qingqun Kong, Yi Zeng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27806](https://doi.org/10.1609/aaai.v38i1.27806)

**Abstract**:

Spiking neural networks (SNNs) are rich in spatio-temporal dynamics and are suitable for processing event-based neuromorphic data.  However, event-based datasets are usually less annotated than static datasets. This small data scale makes SNNs prone to overfitting and limits their performance. In order to improve the generalization ability of SNNs on event-based datasets, we use static images to assist SNN training on event data. In this paper, we first discuss the domain mismatch problem encountered when directly transferring networks trained on static datasets to event data. We argue that the inconsistency of feature distributions becomes a major factor hindering the effective transfer of knowledge from static images to event data. To address this problem, we propose solutions in terms of two aspects: feature distribution and training strategy. Firstly, we propose a knowledge transfer loss, which consists of domain alignment loss and spatio-temporal regularization. The domain alignment loss learns domain-invariant spatial features by reducing the marginal distribution distance between the static image and the event data. Spatio-temporal regularization provides dynamically learnable coefficients for domain alignment loss by using the output features of the event data at each time step as a regularization term. In addition, we propose a sliding training strategy, which gradually replaces static image inputs probabilistically with event data, resulting in a smoother and more stable training for the network. We validate our method on neuromorphic datasets, including N-Caltech101, CEP-DVS, and N-Omniglot. The experimental results show that our proposed method achieves better performance on all datasets compared to the current state-of-the-art methods. 
Code is available at https://github.com/Brain-Cog-Lab/Transfer-for-DVS.

----

## [58] Responding to the Call: Exploring Automatic Music Composition Using a Knowledge-Enhanced Model

**Authors**: *Zhejing Hu, Yan Liu, Gong Chen, Xiao Ma, Shenghua Zhong, Qianwen Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27807](https://doi.org/10.1609/aaai.v38i1.27807)

**Abstract**:

Call-and-response is a musical technique that enriches the creativity of music, crafting coherent musical ideas that mirror the back-and-forth nature of human dialogue with distinct musical characteristics. Although this technique is integral to numerous musical compositions, it remains largely uncharted in automatic music composition. To enhance the creativity of machine-composed music, we first introduce the Call-Response Dataset (CRD) containing 19,155 annotated musical pairs and crafted comprehensive objective evaluation metrics for musical assessment. Then, we design a knowledge-enhanced learning-based method to bridge the gap between human and machine creativity. Specifically, we train the composition module using the call-response pairs, supplementing it with musical knowledge in terms of rhythm, melody, and harmony. Our experimental results underscore that our proposed model adeptly produces a wide variety of creative responses for various musical calls.

----

## [59] Neural Amortized Inference for Nested Multi-Agent Reasoning

**Authors**: *Kunal Jha, Tuan Anh Le, Chuanyang Jin, Yen-Ling Kuo, Joshua B. Tenenbaum, Tianmin Shu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27808](https://doi.org/10.1609/aaai.v38i1.27808)

**Abstract**:

Multi-agent interactions, such as communication, teaching, and bluffing, often rely on higher-order social inference, i.e., understanding how others infer oneself. Such intricate reasoning can be effectively modeled through nested multi-agent reasoning. Nonetheless, the computational complexity escalates exponentially with each level of reasoning, posing a significant challenge. However, humans effortlessly perform complex social inferences as part of their daily lives. To bridge the gap between human-like inference capabilities and computational limitations, we propose a novel approach: leveraging neural networks to amortize high-order social inference, thereby expediting nested multi-agent reasoning. We evaluate our method in two challenging multi-agent interaction domains. The experimental results demonstrate that our method is computationally efficient while exhibiting minimal degradation in accuracy.

----

## [60] Hidden Follower Detection: How Is the Gaze-Spacing Pattern Embodied in Frequency Domain?

**Authors**: *Shu Li, Ruimin Hu, Suhui Li, Liang Liao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27809](https://doi.org/10.1609/aaai.v38i1.27809)

**Abstract**:

Spatiotemporal social behavior analysis is a technique that studies the social behavior patterns of objects and estimates their risks based on their trajectories. In social public scenarios such as train stations, hidden following behavior has become one of the most challenging issues due to its probability of evolving into violent events, which is more than 25%. In recent years, research on hidden following detection (HFD) has focused on differences in time series between hidden followers and normal pedestrians under two temporal characteristics: gaze and spatial distance. However, the time-domain representation for time series is irreversible and usually causes the loss of critical information. In this paper, we deeply study the expression efficiency of time/frequency domain features of time series, by exploring the recovery mechanism of features to source time series, we establish a fidelity estimation method for feature expression and a selection model for frequency-domain features based on the signal-to-distortion ratio (SDR). Experimental results demonstrate the feature fidelity of time series and HFD performance are positively correlated, and the fidelity of frequency-domain features and HFD performance are significantly better than the time-domain features. On both real and simulated datasets, the accuracy of the proposed method is increased by 3%, and the gaze-only module is improved by 10%. Related research has explored new methods for optimal feature selection based on fidelity, new patterns for efficient feature expression of hidden following behavior, and the mechanism of multimodal collaborative identification.

----

## [61] Music Style Transfer with Time-Varying Inversion of Diffusion Models

**Authors**: *Sifei Li, Yuxin Zhang, Fan Tang, Chongyang Ma, Weiming Dong, Changsheng Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27810](https://doi.org/10.1609/aaai.v38i1.27810)

**Abstract**:

With the development of diffusion models, text-guided image style transfer has demonstrated great controllable and high-quality results. However, the utilization of text for diverse music style transfer poses significant challenges, primarily due to the limited availability of matched audio-text datasets. Music, being an abstract and complex art form, exhibits variations and intricacies even within the same genre, thereby making accurate textual descriptions challenging. This paper presents a music style transfer approach that effectively captures musical attributes using minimal data.  We introduce a novel time-varying textual inversion module to precisely capture mel-spectrogram features at different levels. During inference, we utilize a bias-reduced stylization technique to get stable results. Experimental results demonstrate that our method can transfer the style of specific instruments, as well as incorporate natural sounds to compose melodies. Samples and code are available at https://lsfhuihuiff.github.io/MusicTI/.

----

## [62] A Brain-Inspired Way of Reducing the Network Complexity via Concept-Regularized Coding for Emotion Recognition

**Authors**: *Han Lu, Xiahai Zhuang, Qiang Luo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27811](https://doi.org/10.1609/aaai.v38i1.27811)

**Abstract**:

The human brain can effortlessly and reliably perceive emotions, whereas existing facial emotion recognition (FER) methods suffer from drawbacks such as complex model structures, high storage requirements, and poor interpretability. Inspired by the role of emotion concepts in visual perception coding within the human brain, we propose a dual-pathway framework emulating the neural computation of emotion recognition. Specifically, these two pathways are designed to model the representation of emotion concepts in the brain and the visual perception process, respectively. For the former, we adopt a disentangled approach to extract emotion concepts from complex facial geometric attributes; for the latter, we employ an emotional confidence evaluation strategy to determine which concept is optimal for regularizing the perceptual coding. The proposed concept-regularized coding strategy endows the framework with flexibility and interpretability as well as good performances on several benchmarking FER datasets.

----

## [63] Multi-Energy Guided Image Translation with Stochastic Differential Equations for Near-Infrared Facial Expression Recognition

**Authors**: *Bingjun Luo, Zewen Wang, Jinpeng Wang, Junjie Zhu, Xibin Zhao, Yue Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27812](https://doi.org/10.1609/aaai.v38i1.27812)

**Abstract**:

Illumination variation has been a long-term challenge in real-world facial expression recognition (FER). Under uncontrolled or non-visible light conditions, near-infrared (NIR) can provide a simple and alternative solution to obtain high-quality images and supplement the geometric and texture details that are missing in the visible (VIS) domain. Due to the lack of large-scale NIR facial expression datasets, directly extending VIS FER methods to the NIR spectrum may be ineffective. Additionally, previous heterogeneous image synthesis methods are restricted by low controllability without prior task knowledge. To tackle these issues, we present the first approach, called for NIR-FER Stochastic Differential Equations (NFER-SDE), that transforms face expression appearance between heterogeneous modalities to the overfitting problem on small-scale NIR data. NFER-SDE can take the whole VIS source image as input and, together with domain-specific knowledge, guide the preservation of modality-invariant information in the high-frequency content of the image. Extensive experiments and ablation studies show that NFER-SDE significantly improves the performance of NIR FER and achieves state-of-the-art results on the only two available NIR FER datasets, Oulu-CASIA and Large-HFE.

----

## [64] Successive POI Recommendation via Brain-Inspired Spatiotemporal Aware Representation

**Authors**: *Gehua Ma, He Wang, Jingyuan Zhao, Rui Yan, Huajin Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27813](https://doi.org/10.1609/aaai.v38i1.27813)

**Abstract**:

Existing approaches usually perform spatiotemporal representation in the spatial and temporal dimensions, respectively, which isolates the spatial and temporal natures of the target and leads to sub-optimal embeddings. Neuroscience research has shown that the mammalian brain entorhinal-hippocampal system provides efficient graph representations for general knowledge. Moreover, entorhinal grid cells present concise spatial representations, while hippocampal place cells represent perception conjunctions effectively. Thus, the entorhinal-hippocampal system provides a novel angle for spatiotemporal representation, which inspires us to propose the SpatioTemporal aware Embedding framework (STE) and apply it to POIs (STEP). STEP considers two types of POI-specific representations: sequential representation and spatiotemporal conjunctive representation, learned using sparse unlabeled data based on the proposed graph-building policies. Notably, STEP jointly represents the spatiotemporal natures of POIs using both observations and contextual information from integrated spatiotemporal dimensions by constructing a spatiotemporal context graph. Furthermore, we introduce a successive POI recommendation method using STEP, which achieves state-of-the-art performance on two benchmarks. In addition, we demonstrate the excellent performance of the STE representation approach in other spatiotemporal representation-centered tasks through a case study of the traffic flow prediction problem. Therefore, this work provides a novel solution to spatiotemporal representation and paves a new way for spatiotemporal modeling-related tasks.

----

## [65] BDIQA: A New Dataset for Video Question Answering to Explore Cognitive Reasoning through Theory of Mind

**Authors**: *Yuanyuan Mao, Xin Lin, Qin Ni, Liang He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27814](https://doi.org/10.1609/aaai.v38i1.27814)

**Abstract**:

As a foundational component of cognitive intelligence, theory of mind (ToM) can make AI more closely resemble human thought processes, thereby enhancing their interaction and collaboration with human. In particular, it can significantly improve a model's comprehension of videos in complex scenes. However, current video question answer (VideoQA) datasets focus on studying causal reasoning within events, few of them genuinely incorporating human ToM. Consequently, there is a lack of development in ToM reasoning tasks within the area of VideoQA. This paper presents BDIQA, the first benchmark to explore the cognitive reasoning capabilities of VideoQA models in the context of ToM. BDIQA is inspired by the cognitive development of children's ToM and addresses the current deficiencies in machine ToM within datasets and tasks. Specifically, it offers tasks at two difficulty levels, assessing Belief, Desire and Intention (BDI) reasoning in both simple and complex scenarios.  We conduct evaluations on several mainstream methods of VideoQA and diagnose their capabilities with zero-shot, few-shot and supervised learning. We find that the performance of pre-trained models on cognitive reasoning tasks remains unsatisfactory. To counter this challenge, we undertake thorough analysis and experimentation, ultimately presenting two guidelines to enhance cognitive reasoning derived from ablation analysis.

----

## [66] Unveiling the Significance of Toddler-Inspired Reward Transition in Goal-Oriented Reinforcement Learning

**Authors**: *Junseok Park, Yoonsung Kim, Hee bin Yoo, Min Whoo Lee, Kibeom Kim, Won-Seok Choi, Minsu Lee, Byoung-Tak Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27815](https://doi.org/10.1609/aaai.v38i1.27815)

**Abstract**:

Toddlers evolve from free exploration with sparse feedback to exploiting prior experiences for goal-directed learning with denser rewards. Drawing inspiration from this Toddler-Inspired Reward Transition, we set out to explore the implications of varying reward transitions when incorporated into Reinforcement Learning (RL) tasks. Central to our inquiry is the transition from sparse to potential-based dense rewards, which share optimal strategies regardless of reward changes. Through various experiments, including those in egocentric navigation and robotic arm manipulation tasks, we found that proper reward transitions significantly influence sample efficiency and success rates. Of particular note is the efficacy of the toddler-inspired Sparse-to-Dense (S2D) transition. Beyond these performance metrics, using Cross-Density Visualizer technique, we observed that transitions, especially the S2D, smooth the policy loss landscape, promoting wide minima that enhance generalization in RL models.

----

## [67] Gated Attention Coding for Training High-Performance and Efficient Spiking Neural Networks

**Authors**: *Xuerui Qiu, Rui-Jie Zhu, Yuhong Chou, Zhaorui Wang, Liang-Jian Deng, Guoqi Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27816](https://doi.org/10.1609/aaai.v38i1.27816)

**Abstract**:

Spiking neural networks (SNNs) are emerging as an energy-efficient alternative to traditional artificial neural networks (ANNs) due to their unique spike-based event-driven nature. Coding is crucial in SNNs as it converts external input stimuli into spatio-temporal feature sequences.  However, most existing deep SNNs rely on direct coding that generates powerless spike representation and lacks the temporal dynamics inherent in human vision. Hence, we introduce Gated Attention Coding (GAC), a plug-and-play module that leverages the multi-dimensional gated attention unit to efficiently encode inputs into powerful representations before feeding them into the SNN architecture. GAC functions as a preprocessing layer that does not disrupt the spike-driven nature of the SNN, making it amenable to efficient neuromorphic hardware implementation with minimal modifications. Through an observer model theoretical analysis, we demonstrate GAC's attention mechanism improves temporal dynamics and coding efficiency. Experiments on CIFAR10/100 and ImageNet datasets demonstrate that GAC achieves state-of-the-art accuracy with remarkable efficiency. Notably, we improve top-1 accuracy by 3.10% on CIFAR100 with only 6-time steps and 1.07% on ImageNet while reducing energy usage to 66.9% of the previous works. To our best knowledge, it is the first time to explore the attention-based dynamic coding scheme in deep SNNs, with exceptional effectiveness and efficiency on large-scale datasets. Code is available at https://github.com/bollossom/GAC.

----

## [68] Efficient Spiking Neural Networks with Sparse Selective Activation for Continual Learning

**Authors**: *Jiangrong Shen, Wenyao Ni, Qi Xu, Huajin Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27817](https://doi.org/10.1609/aaai.v38i1.27817)

**Abstract**:

The next generation of machine intelligence requires the capability of continual learning to acquire new knowledge without forgetting the old one while conserving limited computing resources. 
Spiking neural networks (SNNs), compared to artificial neural networks (ANNs), have more characteristics that align with biological neurons, which may be helpful as a potential gating function for knowledge maintenance in neural networks. Inspired by the selective sparse activation principle of context gating in biological systems, we present a novel SNN model with selective activation to achieve continual learning. The trace-based K-Winner-Take-All (K-WTA) and variable threshold components are designed to form the sparsity in selective activation in spatial and temporal dimensions of spiking neurons, which promotes the subpopulation of neuron activation to perform specific tasks. As a result, continual learning can be maintained by routing different tasks via different populations of neurons in the network. The experiments are conducted on MNIST and CIFAR10 datasets under the class incremental setting. The results show that the proposed SNN model achieves competitive performance similar to and even surpasses the other regularization-based methods deployed under traditional ANNs.

----

## [69] Boosting Neural Cognitive Diagnosis with Student's Affective State Modeling

**Authors**: *Shanshan Wang, Zhen Zeng, Xun Yang, Ke Xu, Xingyi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27818](https://doi.org/10.1609/aaai.v38i1.27818)

**Abstract**:

Cognitive Diagnosis Modeling aims to infer students' proficiency level on knowledge concepts from their response logs. Existing methods typically model students’ response processes as the interaction between students and exercises or concepts based on hand-crafted or deeply-learned interaction functions. Despite their promising achievements, they fail to consider the relationship between students' cognitive states and affective states in learning, e.g., the feelings of frustration, boredom, or confusion with the learning content, which is insufficient for comprehensive cognitive diagnosis in intelligent education. To fill the research gap, we propose a novel Affect-aware Cognitive Diagnosis (ACD) model which can effectively diagnose the knowledge proficiency levels of students by taking into consideration the affective factors. Specifically, we first design a student affect perception module under the assumption that the affective state is jointly influenced by the student's affect trait and the difficulty of the exercise. Then, our inferred affective distribution is further used to estimate the student's subjective factors, i.e., guessing and slipping, respectively. Finally, we integrate the estimated guessing and slipping parameters with the basic neural cognitive diagnosis framework based on the DINA model, which facilitates the modeling of complex exercising interactions in a more accurate and interpretable fashion. Besides, we also extend our affect perception module in an unsupervised learning setting based on contrastive learning, thus significantly improving the compatibility of our ACD. To the best of our knowledge, we are the first to unify the cognition modeling and affect modeling into the same framework for student cognitive diagnosis. Extensive experiments on real-world datasets clearly demonstrate the effectiveness of our ACD. Our code is available at https://github.com/zeng-zhen/ACD.

----

## [70] DMMR: Cross-Subject Domain Generalization for EEG-Based Emotion Recognition via Denoising Mixed Mutual Reconstruction

**Authors**: *Yiming Wang, Bin Zhang, Yujiao Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27819](https://doi.org/10.1609/aaai.v38i1.27819)

**Abstract**:

Electroencephalography (EEG) has proven to be effective in emotion analysis. However, current methods struggle with individual variations, complicating the generalization of models trained on data from source subjects to unseen target subjects. To tackle this issue, we propose the Denoising Mixed Mutual Reconstruction (DMMR) model, employing a two-stage pre-training followed by fine-tuning approach. During the pre-training phase, DMMR leverages self-supervised learning through a multi-decoder autoencoder, which encodes and reconstructs features of one subject, aiming to generate features resembling those from other subjects within the same category, thereby encouraging the encoder to learn subject-invariant features. We introduce a hidden-layer mixed data augmentation approach to mitigate the limitations posed by the scarcity of source data, thereby extending the method to a two-stage process. To bolster stability against noise, we incorporate a noise injection method, named “Time Steps Shuffling”, into the input data. During the fine-tuning phase, an emotion classifier is integrated to extract emotion-related features. Experimental accuracy on the SEED and SEED-IV datasets reached 88.27% (±5.62) and 72.70% (±8.01), respectively, demonstrating state-of-the-art and comparable performance, thereby showcasing the superiority of DMMR. The proposed data augmentation and noise injection methods were observed to complementarily enhance accuracy and stability, thus alleviating the aforementioned issues.

----

## [71] Transient Glimpses: Unveiling Occluded Backgrounds through the Spike Camera

**Authors**: *Jiyuan Zhang, Shiyan Chen, Yajing Zheng, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27820](https://doi.org/10.1609/aaai.v38i1.27820)

**Abstract**:

The de-occlusion problem, involving extracting clear background images by removing foreground occlusions, holds significant practical importance but poses considerable challenges. Most current research predominantly focuses on generating discrete images from calibrated camera arrays, but this approach often struggles with dense occlusions and fast motions due to limited perspectives and motion blur. To overcome these limitations, an effective solution requires the integration of multi-view visual information. The spike camera, as an innovative neuromorphic sensor, shows promise with its ultra-high temporal resolution and dynamic range. In this study, we propose a novel approach that utilizes a single spike camera for continuous multi-view imaging to address occlusion removal. By rapidly moving the spike camera, we capture a dense stream of spikes from occluded scenes. Our model, SpkOccNet, processes these spikes by integrating multi-view spatial-temporal information via long-short-window feature extractor (LSW) and employs a novel cross-view mutual attention-based module (CVA) for effective fusion and refinement. Additionally, to facilitate research in occlusion removal, we introduce the S-OCC dataset, which consists of real-world spike-based data. Experimental results demonstrate the efficiency and generalization capabilities of our model in effectively removing dense occlusions across diverse scenes. Public project page: https://github.com/Leozhangjiyuan/SpikeDeOcclusion.

----

## [72] Open-Set Facial Expression Recognition

**Authors**: *Yuhang Zhang, Yue Yao, Xuannan Liu, Lixiong Qin, Wenjing Wang, Weihong Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27821](https://doi.org/10.1609/aaai.v38i1.27821)

**Abstract**:

Facial expression recognition (FER) models are typically trained on datasets with a fixed number of seven basic classes. However, recent research works (Cowen et al. 2021; Bryant et al. 2022; Kollias 2023) point out that there are far more expressions than the basic ones. Thus, when these models are deployed in the real world, they may encounter unknown classes, such as compound expressions that cannot be classified into existing basic classes. To address this issue, we propose the open-set FER task for the first time. Though there are many existing open-set recognition methods, we argue that they do not work well for open-set FER because FER data are all human faces with very small inter-class distances, which makes the open-set samples very similar to close-set samples. In this paper, we are the first to transform the disadvantage of small inter-class distance into an advantage by proposing a new way for open-set FER. Specifically, we find that small inter-class distance allows for sparsely distributed pseudo labels of open-set samples, which can be viewed as symmetric noisy labels. Based on this novel observation, we convert the open-set FER to a noisy label detection problem. We further propose a novel method that incorporates attention map consistency and cycle training to detect the open-set samples. Extensive experiments on various FER datasets demonstrate that our method clearly outperforms state-of-the-art open-set recognition methods by large margins. Code is available at https://github.com/zyh-uaiaaaa.

----

## [73] Bootstrapping Cognitive Agents with a Large Language Model

**Authors**: *Feiyu Zhu, Reid G. Simmons*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27822](https://doi.org/10.1609/aaai.v38i1.27822)

**Abstract**:

Large language models contain noisy general knowledge of the world, yet are hard to train or fine-tune. In contrast cognitive architectures have excellent interpretability and are flexible to update but require a lot of manual work to instantiate. In this work, we combine the best of both worlds: bootstrapping a cognitive-based model with the noisy knowledge encoded in large language models. Through an embodied agent doing kitchen tasks, we show that our proposed framework yields better efficiency compared to an agent entirely based on large language models. Our experiments also indicate that the cognitive agent bootstrapped using this framework can generalize to novel environments and be scaled to complex tasks.

----

## [74] Data Augmented Graph Neural Networks for Personality Detection

**Authors**: *Yangfu Zhu, Yue Xia, Meiling Li, Tingting Zhang, Bin Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i1.27823](https://doi.org/10.1609/aaai.v38i1.27823)

**Abstract**:

Personality detection is a fundamental task for user psychology research. One of the biggest challenges in personality detection lies in the quantitative limitation of labeled data collected by completing the personality questionnaire, which is very time-consuming and labor-intensive. Most of the existing works are mainly devoted to learning the rich representations of posts based on labeled data. However, they still suffer from the inherent weakness of the amount limitation of labels, which potentially restricts the capability of the model to deal with unseen data. In this paper, we construct a heterogeneous personality graph for each labeled and unlabeled user and develop a novel psycholinguistic augmented graph neural network to detect personality in a semi-supervised manner, namely Semi-PerGCN. Specifically,  our model first explores a supervised Personality Graph Neural Network (PGNN) to refine labeled user representation on the heterogeneous graph. For the remaining massive unlabeled users, we utilize the empirical psychological knowledge of the Linguistic Inquiry and Word Count (LIWC) lexicon for multi-view graph augmentation and perform unsupervised graph consistent constraints on the parameters shared PGNN.  During the learning process of finite labeled users, noise-invariant learning on a large scale of unlabeled users is combined to enhance the generalization ability. Extensive experiments on three real-world datasets, Youtube, PAN2015, and MyPersonality demonstrate the effectiveness of our Semi-PerGCN in personality detection, especially in scenarios with limited labeled users.

----

## [75] DreamStyler: Paint by Style Inversion with Text-to-Image Diffusion Models

**Authors**: *Namhyuk Ahn, Junsoo Lee, Chunggi Lee, Kunhee Kim, Daesik Kim, Seung-Hun Nam, Kibeom Hong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27824](https://doi.org/10.1609/aaai.v38i2.27824)

**Abstract**:

Recent progresses in large-scale text-to-image models have yielded remarkable accomplishments, finding various applications in art domain.
However, expressing unique characteristics of an artwork (e.g. brushwork, colortone, or composition) with text prompts alone may encounter limitations due to the inherent constraints of verbal description.
To this end, we introduce DreamStyle, a novel framework designed for artistic image synthesis, proficient in both text-to-image synthesis and style transfer.
DreamStyle optimizes a multi-stage textual embedding with a context-aware text prompt, resulting in prominent image quality.
In addition, with content and style guidance, DreamStyle exhibits flexibility to accommodate a range of style references.
Experimental results demonstrate its superior performance across multiple scenarios, suggesting its promising potential in artistic product creation.
Project page: https://nmhkahn.github.io/dreamstyler/

----

## [76] Context Enhanced Transformer for Single Image Object Detection in Video Data

**Authors**: *Seungjun An, Seonghoon Park, Gyeongnyeon Kim, Jeongyeol Baek, Byeongwon Lee, Seungryong Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27825](https://doi.org/10.1609/aaai.v38i2.27825)

**Abstract**:

With the increasing importance of video data in real-world applications, there is a rising need for efficient object detection methods that utilize temporal information. While existing video object detection (VOD) techniques employ various strategies to address this challenge, they typically depend on locally adjacent frames or randomly sampled images within a clip. Although recent Transformer-based VOD methods have shown promising results, their reliance on multiple inputs and additional network complexity to incorporate temporal information limits their practical applicability. In this paper, we propose a novel approach to single image object detection, called Context Enhanced TRansformer (CETR), by incorporating temporal context into DETR using a newly designed memory module. To efficiently store temporal information, we construct a class-wise memory that collects contextual information across data. Additionally, we present a classification-based sampling technique to selectively utilize the relevant memory for the current image. In the testing, We introduce a test-time memory adaptation method that updates individual memory functions by considering the test distribution. Experiments with CityCam and ImageNet VID datasets exhibit the efficiency of the framework on various video systems. The project page and code will be made available at: https://ku-cvlab.github.io/CETR.

----

## [77] SHaRPose: Sparse High-Resolution Representation for Human Pose Estimation

**Authors**: *Xiaoqi An, Lin Zhao, Chen Gong, Nannan Wang, Di Wang, Jian Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27826](https://doi.org/10.1609/aaai.v38i2.27826)

**Abstract**:

High-resolution representation is essential for achieving good performance in human pose estimation models. To obtain such features, existing works utilize high-resolution input images or fine-grained image tokens. However, this dense high-resolution representation brings a significant computational burden. In this paper, we address the following question: "Only sparse human keypoint locations are detected for human pose estimation, is it really necessary to describe the whole image in a dense, high-resolution manner?" Based on dynamic transformer models, we propose a framework that only uses Sparse High-resolution Representations for human Pose estimation (SHaRPose). In detail, SHaRPose consists of two stages. At the coarse stage, the relations between image regions and keypoints are dynamically mined while a coarse estimation is generated. Then, a quality predictor is applied to decide whether the coarse estimation results should be refined. At the fine stage, SHaRPose builds sparse high-resolution representations only on the regions related to the keypoints and provides refined high-precision human pose estimations. Extensive experiments demonstrate the outstanding performance of the proposed method. Specifically, compared to the state-of-the-art method ViTPose, our model SHaRPose-Base achieves 77.4 AP (+0.5 AP) on the COCO validation set and 76.7 AP (+0.5 AP) on the COCO test-dev set, and infers at a speed of 1.4x faster than ViTPose-Base. Code is available at https://github.com/AnxQ/sharpose.

----

## [78] Comparing the Robustness of Modern No-Reference Image- and Video-Quality Metrics to Adversarial Attacks

**Authors**: *Anastasia Antsiferova, Khaled Abud, Aleksandr Gushchin, Ekaterina Shumitskaya, Sergey Lavrushkin, Dmitriy S. Vatolin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27827](https://doi.org/10.1609/aaai.v38i2.27827)

**Abstract**:

Nowadays, neural-network-based image- and video-quality metrics perform better than traditional methods. However, they also became more vulnerable to adversarial attacks that increase metrics' scores without improving visual quality. The existing benchmarks of quality metrics compare their performance in terms of correlation with subjective quality and calculation time. Nonetheless, the adversarial robustness of image-quality metrics is also an area worth researching. This paper analyses modern metrics' robustness to different adversarial attacks. We adapted adversarial attacks from computer vision tasks and compared attacks' efficiency against 15 no-reference image- and video-quality metrics. Some metrics showed high resistance to adversarial attacks, which makes their usage in benchmarks safer than vulnerable metrics. The benchmark accepts submissions of new metrics for researchers who want to make their metrics more robust to attacks or to find such metrics for their needs. The latest results can be found online: https://videoprocessing.ai/benchmarks/metrics-robustness.html.

----

## [79] DocFormerv2: Local Features for Document Understanding

**Authors**: *Srikar Appalaraju, Peng Tang, Qi Dong, Nishant Sankaran, Yichu Zhou, R. Manmatha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27828](https://doi.org/10.1609/aaai.v38i2.27828)

**Abstract**:

We propose DocFormerv2, a multi-modal transformer for Visual Document Understanding (VDU). The VDU domain entails understanding documents (beyond mere OCR predictions) e.g., extracting information from a form, VQA for documents and other tasks. VDU is challenging as it needs a model to make sense of multiple modalities (visual, language and spatial) to make a prediction. Our approach, termed DocFormerv2 is an encoder-decoder transformer which takes as input - vision, language and spatial features. DocFormerv2 is pre-trained with unsupervised tasks employed asymmetrically i.e., two novel document tasks on encoder and one on the auto-regressive decoder. The unsupervised tasks have been carefully designed to ensure that the pre-training encourages local-feature alignment between multiple modalities. DocFormerv2 when evaluated on nine challenging datasets shows state-of-the-art performance on all over strong baselines - On TabFact (+4.3%), InfoVQA (+1.4%), FUNSD (+1.0%). Furthermore, to show generalization capabilities, on three VQA tasks involving scene-text, DocFormerv2 outperforms previous comparably-sized models and even does better than much larger models (such as GIT2, PaLI and Flamingo) on these tasks. Extensive ablations show that due to its novel pre-training tasks, DocFormerv2 understands multiple modalities better than prior-art in VDU.

----

## [80] Exposing the Deception: Uncovering More Forgery Clues for Deepfake Detection

**Authors**: *Zhongjie Ba, Qingyu Liu, Zhenguang Liu, Shuang Wu, Feng Lin, Li Lu, Kui Ren*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27829](https://doi.org/10.1609/aaai.v38i2.27829)

**Abstract**:

Deepfake technology has given rise to a spectrum of novel and compelling applications. Unfortunately, the widespread proliferation of high-fidelity fake videos has led to pervasive confusion and deception, shattering our faith that seeing is believing. One aspect that has been overlooked so far is that current deepfake detection approaches may easily fall into the trap of overfitting, focusing only on forgery clues within one or a few local regions. Moreover, existing works heavily rely on neural networks to extract forgery features, lacking theoretical constraints guaranteeing that sufficient forgery clues are extracted and superfluous features are eliminated. These deficiencies culminate in unsatisfactory accuracy and limited generalizability in real-life scenarios.

In this paper, we try to tackle these challenges through three designs: (1) We present a novel framework to capture broader forgery clues by extracting multiple non-overlapping local representations and fusing them into a global semantic-rich feature. (2) Based on the information bottleneck theory, we derive Local Information Loss to guarantee the orthogonality of local representations while preserving comprehensive task-relevant information. (3) Further, to fuse the local representations and remove task-irrelevant information, we arrive at a Global Information Loss through the theoretical analysis of mutual information. Empirically, our method achieves state-of-the-art performance on five benchmark datasets. Our code is available at https://github.com/QingyuLiu/Exposing-the-Deception, hoping to inspire researchers.

----

## [81] Prompt-Based Distribution Alignment for Unsupervised Domain Adaptation

**Authors**: *Shuanghao Bai, Min Zhang, Wanqi Zhou, Siteng Huang, Zhirong Luan, Donglin Wang, Badong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27830](https://doi.org/10.1609/aaai.v38i2.27830)

**Abstract**:

Recently, despite the unprecedented success of large pre-trained visual-language models (VLMs) on a wide range of downstream tasks, the real-world unsupervised domain adaptation (UDA) problem is still not well explored. Therefore, in this paper, we first experimentally demonstrate that the unsupervised-trained VLMs can significantly reduce the distribution discrepancy between source and target domains, thereby improving the performance of UDA. However, a major challenge for directly deploying such models on downstream UDA tasks is prompt engineering, which requires aligning the domain knowledge of source and target domains, since the performance of UDA is severely influenced by a good domain-invariant representation. We further propose a Prompt-based Distribution Alignment (PDA) method to incorporate the domain knowledge into prompt learning. Specifically, PDA employs a two-branch prompt-tuning paradigm, namely base branch and alignment branch. The base branch focuses on integrating class-related representation into prompts, ensuring discrimination among different classes.  To further minimize domain discrepancy, for the alignment branch, we construct feature banks for both the source and target domains and propose image-guided feature tuning (IFT) to make the input attend to feature banks, which effectively integrates self-enhanced and cross-domain features into the model.  In this way, these two branches can be mutually promoted to enhance the adaptation of VLMs for UDA. We conduct extensive experiments on three benchmarks to demonstrate that our proposed PDA achieves state-of-the-art performance. The code is available at https://github.com/BaiShuanghao/Prompt-based-Distribution-Alignment.

----

## [82] Local-Global Multi-Modal Distillation for Weakly-Supervised Temporal Video Grounding

**Authors**: *Peijun Bao, Yong Xia, Wenhan Yang, Boon Poh Ng, Meng Hwa Er, Alex C. Kot*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27831](https://doi.org/10.1609/aaai.v38i2.27831)

**Abstract**:

This paper for the first time leverages multi-modal videos for weakly-supervised temporal video grounding. As labeling the video moment is labor-intensive and subjective, the weakly-supervised approaches have gained increasing attention in recent years. However, these approaches could inherently compromise performance due to inadequate supervision. Therefore, to tackle this challenge, we for the first time pay attention to exploiting complementary information extracted from multi-modal videos (e.g., RGB frames, optical flows), where richer supervision is naturally introduced in the weaklysupervised context. Our motivation is that by integrating different modalities of the videos, the model is learned from synergic supervision and thereby can attain superior generalization capability. However, addressing multiple modalities† would also inevitably introduce additional computational overhead, and might become inapplicable if a particular modality is inaccessible. To solve this issue, we adopt a novel route: building a multi-modal distillation algorithm to capitalize on the multi-modal knowledge as supervision for model training, while still being able to work with only the single modal input during inference. As such, we can utilize the benefits brought by the supplementary nature of multiple modalities, without compromising the applicability in practical scenarios. Specifically, we first propose a cross-modal mutual learning framework and train a sophisticated teacher model to learn collaboratively from the multi-modal videos. Then we identify two sorts of knowledge from the teacher model, i.e., temporal boundaries and semantic activation map. And we devise a local-global distillation algorithm to transfer this knowledge to a student model of single-modal input at both local and global levels. Extensive experiments on large-scale datasets demonstrate that our method achieves state-of-the-art performance with/without multi-modal inputs.

----

## [83] Omnipotent Distillation with LLMs for Weakly-Supervised Natural Language Video Localization: When Divergence Meets Consistency

**Authors**: *Peijun Bao, Zihao Shao, Wenhan Yang, Boon Poh Ng, Meng Hwa Er, Alex C. Kot*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27832](https://doi.org/10.1609/aaai.v38i2.27832)

**Abstract**:

Natural language video localization plays a pivotal role in video understanding, and leveraging weakly-labeled data is considered a promising approach to circumvent the laborintensive process of manual annotations. However, this approach encounters two significant challenges: 1) limited input distribution, namely that the limited writing styles of the language query, annotated by human annotators, hinder the model’s generalization to real-world scenarios with diverse vocabularies and sentence structures; 2) the incomplete ground truth, whose supervision guidance is insufficient. To overcome these challenges, we propose an omnipotent distillation algorithm with large language models (LLM). The distribution of the input sample is enriched to obtain diverse multi-view versions while a consistency then comes to regularize the consistency of their results for distillation. Specifically, we first train our teacher model with the proposed intra-model agreement, where multiple sub-models are supervised by each other. Then, we leverage the LLM to paraphrase the language query and distill the teacher model to a lightweight student model by enforcing the consistency between the localization results of the paraphrased sentence and the original one. In addition, to assess the generalization of the model across different dimensions of language variation, we create extensive datasets by building upon existing datasets. Our experiments demonstrate substantial performance improvements adaptively to diverse kinds of language queries.

----

## [84] Improving Diffusion-Based Image Restoration with Error Contraction and Error Correction

**Authors**: *Qiqi Bao, Zheng Hui, Rui Zhu, Peiran Ren, Xuansong Xie, Wenming Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27833](https://doi.org/10.1609/aaai.v38i2.27833)

**Abstract**:

Generative diffusion prior captured from the off-the-shelf denoising diffusion generative model has recently attained significant interest. However, several attempts have been made to adopt diffusion models to noisy inverse problems either fail to achieve satisfactory results or require a few thousand iterations to achieve high-quality reconstructions. In this work, we propose a diffusion-based image restoration with error contraction and error correction (DiffECC) method. Two strategies are introduced to contract the restoration error in the posterior sampling process. First, we combine existing CNN-based approaches with diffusion models to ensure data consistency from the beginning. Second, to amplify the error contraction effects of the noise, a restart sampling algorithm is designed. In the error correction strategy, the estimation-correction idea is proposed on both the data term and the prior term. Solving them iteratively within the diffusion sampling framework leads to superior image generation results. Experimental results for image restoration tasks such as super-resolution (SR), Gaussian deblurring, and motion deblurring demonstrate that our approach can reconstruct high-quality images compared with state-of-the-art sampling-based diffusion models.

----

## [85] Relevant Intrinsic Feature Enhancement Network for Few-Shot Semantic Segmentation

**Authors**: *Xiaoyi Bao, Jie Qin, Siyang Sun, Xingang Wang, Yun Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27834](https://doi.org/10.1609/aaai.v38i2.27834)

**Abstract**:

For few-shot semantic segmentation, the primary task is to extract class-specific intrinsic information from limited labeled data. However, the semantic ambiguity and inter-class similarity of previous methods limit the accuracy of pixel-level foreground-background classification. To alleviate these issues, we propose the Relevant Intrinsic Feature Enhancement Network (RiFeNet). To improve the semantic consistency of foreground instances, we propose an unlabeled branch as an efficient data utilization method, which teaches the model how to extract intrinsic features robust to intra-class differences. Notably, during testing, the proposed unlabeled branch is excluded without extra unlabeled data and computation. Furthermore, we extend the inter-class variability between foreground and background by proposing a novel multi-level prototype generation and interaction module. The different-grained complementarity between global and local prototypes allows for better distinction between similar categories. The qualitative and quantitative performance of RiFeNet surpasses the state-of-the-art methods on PASCAL-5i and COCO benchmarks.

----

## [86] Image Safeguarding: Reasoning with Conditional Vision Language Model and Obfuscating Unsafe Content Counterfactually

**Authors**: *Mazal Bethany, Brandon Wherry, Nishant Vishwamitra, Peyman Najafirad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27835](https://doi.org/10.1609/aaai.v38i2.27835)

**Abstract**:

Social media platforms are being increasingly used by malicious actors to share unsafe content, such as images depicting sexual activity, cyberbullying, and self-harm. Consequently, major platforms use artificial intelligence (AI) and human moderation to obfuscate such images to make them safer. Two critical needs for obfuscating unsafe images is that an accurate rationale for obfuscating image regions must be provided, and the sensitive regions should be obfuscated (e.g. blurring) for users' safety. This process involves addressing two key problems: (1) the reason for obfuscating unsafe images demands the platform to provide an accurate rationale that must be grounded in unsafe image-specific attributes, and (2) the unsafe regions in the image must be minimally obfuscated while still depicting the safe regions. In this work, we address these key issues by first performing visual reasoning by designing a visual reasoning model (VLM) conditioned on pre-trained unsafe image classifiers to provide an accurate rationale grounded in unsafe image attributes, and then proposing a counterfactual explanation algorithm that minimally identifies and obfuscates unsafe regions for safe viewing, by first utilizing an unsafe image classifier attribution matrix to guide segmentation for a more optimal subregion segmentation followed by an informed greedy search to determine the minimum number of subregions required to modify the classifier's output based on attribution score. Extensive experiments on uncurated data from social networks emphasize the efficacy of our proposed method. We make our code available at: https://github.com/SecureAIAutonomyLab/ConditionalVLM

----

## [87] DanceAnyWay: Synthesizing Beat-Guided 3D Dances with Randomized Temporal Contrastive Learning

**Authors**: *Aneesh Bhattacharya, Manas Paranjape, Uttaran Bhattacharya, Aniket Bera*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27836](https://doi.org/10.1609/aaai.v38i2.27836)

**Abstract**:

We present DanceAnyWay, a generative learning method to synthesize beat-guided dances of 3D human characters synchronized with music. Our method learns to disentangle the dance movements at the beat frames from the dance movements at all the remaining frames by operating at two hierarchical levels. At the coarser "beat" level, it encodes the rhythm, pitch, and melody information of the input music via dedicated feature representations only at the beat frames. It leverages them to synthesize the beat poses of the target dances using a sequence-to-sequence learning framework. At the finer "repletion" level, our method encodes similar rhythm, pitch, and melody information from all the frames of the input music via dedicated feature representations. It generates the full dance sequences by combining the synthesized beat and repletion poses and enforcing plausibility through an adversarial learning framework. Our training paradigm also enforces fine-grained diversity in the synthesized dances through a randomized temporal contrastive loss, which ensures different segments of the dance sequences have different movements and avoids motion freezing or collapsing to repetitive movements. We evaluate the performance of our approach through extensive experiments on the benchmark AIST++ dataset and observe improvements of about 7%-12% in motion quality metrics and 1.5%-4% in motion diversity metrics over the current baselines, respectively. We also conducted a user study to evaluate the visual quality of our synthesized dances. We noted that, on average, the samples generated by our method were about 9-48% more preferred by the participants and had a 4-27% better five-point Likert-scale score over the best available current baseline in terms of motion quality and synchronization. Our source code and project page are available at https://github.com/aneeshbhattacharya/DanceAnyWay.

----

## [88] DiffSED: Sound Event Detection with Denoising Diffusion

**Authors**: *Swapnil Bhosale, Sauradip Nag, Diptesh Kanojia, Jiankang Deng, Xiatian Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27837](https://doi.org/10.1609/aaai.v38i2.27837)

**Abstract**:

Sound Event Detection (SED) aims to predict the temporal boundaries of all the events of interest and their class labels, given an unconstrained audio sample. Taking either the split-and-classify (i.e., frame-level) strategy or the more principled event-level modeling approach, all existing methods consider the SED problem from the discriminative learning perspective. In this work, we reformulate the SED problem by taking a generative learning perspective. Specifically, we aim to generate sound temporal boundaries from noisy proposals in a denoising diffusion process, conditioned on a target audio sample. During training, our model learns to reverse the noising process by converting noisy latent queries to the ground-truth versions in the elegant Transformer decoder framework. Doing so enables the model generate accurate event boundaries from even noisy queries during inference. Extensive experiments on the Urban-SED and EPIC-Sounds datasets demonstrate that our model significantly outperforms existing alternatives, with 40+% faster convergence in training. Code: https://github.com/Surrey-UPLab/DiffSED

----

## [89] Learning Generalized Segmentation for Foggy-Scenes by Bi-directional Wavelet Guidance

**Authors**: *Qi Bi, Shaodi You, Theo Gevers*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27838](https://doi.org/10.1609/aaai.v38i2.27838)

**Abstract**:

Learning scene semantics that can be well generalized to foggy conditions is important for safety-crucial applications such as autonomous driving. 
Existing methods need both annotated clear images and foggy images to train a curriculum domain adaptation model.
Unfortunately, these methods can only generalize to the target foggy domain that has seen in the training stage, but the foggy domains vary a lot in both urban-scene styles and fog styles.
In this paper, we propose to learn scene segmentation well generalized to foggy-scenes under the domain generalization setting, which does not involve any foggy images in the training stage and can generalize to any arbitrary unseen foggy scenes. 
We argue that an ideal segmentation model that can be well generalized to foggy-scenes need to simultaneously enhance the content, de-correlate the urban-scene style and de-correlate the fog style. 
As the content (e.g., scene semantic) rests more in low-frequency features while the style of urban-scene and fog rests more in high-frequency features, we propose a novel bi-directional wavelet guidance (BWG) mechanism to realize the above three objectives in a divide-and-conquer manner. 
With the aid of Haar wavelet transformation,
the low frequency component is concentrated on the content enhancement self-attention, while the high frequency component is shifted to the style and fog self-attention for de-correlation purpose.
It is integrated into existing mask-level Transformer segmentation pipelines in a learnable fashion.
Large-scale experiments are conducted on four foggy-scene segmentation datasets under a variety of interesting settings.
The proposed method significantly outperforms existing directly-supervised, curriculum domain adaptation and domain generalization segmentation methods. 
Source code is available at https://github.com/BiQiWHU/BWG.

----

## [90] Learning Generalized Medical Image Segmentation from Decoupled Feature Queries

**Authors**: *Qi Bi, Jingjun Yi, Hao Zheng, Wei Ji, Yawen Huang, Yuexiang Li, Yefeng Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27839](https://doi.org/10.1609/aaai.v38i2.27839)

**Abstract**:

Domain generalized medical image segmentation requires models to learn from multiple source domains and generalize well to arbitrary unseen target domain. Such a task is both technically challenging and clinically practical, due to the domain shift problem (i.e., images are collected from different hospitals and scanners). Existing methods focused on either learning shape-invariant representation or reaching consensus among the source domains. An ideal generalized representation is supposed to show similar pattern responses within the same channel for cross-domain images.
However, to deal with the significant distribution discrepancy, the network tends to capture similar patterns by multiple channels, while different cross-domain patterns are also allowed to rest in the same channel. 
To address this issue, we propose to leverage channel-wise decoupled deep features as queries. With the aid of cross-attention mechanism, the long-range dependency between deep and shallow features can be fully mined via self-attention and then guides the learning of generalized representation. Besides, a relaxed deep whitening transformation is proposed to learn channel-wise decoupled features in a feasible way. The proposed decoupled fea-
ture query (DFQ) scheme can be seamlessly integrate into the Transformer segmentation model in an end-to-end manner. 
Extensive experiments show its state-of-the-art performance, notably outperforming the runner-up by 1.31% and 1.98% with DSC metric on generalized fundus and prostate benchmarks, respectively. Source code is available at https://github.com/BiQiWHU/DFQ.

----

## [91] Learning Content-Enhanced Mask Transformer for Domain Generalized Urban-Scene Segmentation

**Authors**: *Qi Bi, Shaodi You, Theo Gevers*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27840](https://doi.org/10.1609/aaai.v38i2.27840)

**Abstract**:

Domain-generalized urban-scene semantic segmentation (USSS) aims to learn generalized semantic predictions across diverse urban-scene styles. Unlike generic domain gap challenges, USSS is unique in that the semantic categories are often similar in different urban scenes, while the styles can vary significantly due to changes in urban landscapes, weather conditions, lighting, and other factors. 
Existing approaches typically rely on convolutional neural networks (CNNs) to learn the content of urban scenes.

In this paper, we propose a Content-enhanced Mask TransFormer (CMFormer) for domain-generalized USSS. The main idea is to enhance the focus of the fundamental component, the mask attention mechanism, in Transformer segmentation models on content information. 
We have observed through empirical analysis that a mask representation effectively captures pixel segments, albeit with reduced robustness to style variations. Conversely, its lower-resolution counterpart exhibits greater ability to accommodate style variations, while being less proficient in representing pixel segments. To harness the synergistic attributes of these two approaches, we introduce a novel content-enhanced mask attention mechanism. It learns mask queries from both the image feature and its down-sampled counterpart, aiming to simultaneously encapsulate the content and address stylistic variations. These features are fused into a Transformer decoder and integrated into a multi-resolution content-enhanced mask attention learning scheme.

Extensive experiments conducted on various domain-generalized urban-scene segmentation datasets demonstrate that the proposed CMFormer significantly outperforms existing CNN-based methods by up to 14.0% mIoU and the contemporary HGFormer by up to 1.7% mIoU. The source code is publicly available at https://github.com/BiQiWHU/CMFormer.

----

## [92] ShapeBoost: Boosting Human Shape Estimation with Part-Based Parameterization and Clothing-Preserving Augmentation

**Authors**: *Siyuan Bian, Jiefeng Li, Jiasheng Tang, Cewu Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27841](https://doi.org/10.1609/aaai.v38i2.27841)

**Abstract**:

Accurate human shape recovery from a monocular RGB image is a challenging task because humans come in different shapes and sizes and wear different clothes. In this paper, we propose ShapeBoost, a new human shape recovery framework that achieves pixel-level alignment even for rare body shapes and high accuracy for people wearing different types of clothes. Unlike previous approaches that rely on the use of PCA-based shape coefficients, we adopt a new human shape parameterization that decomposes the human shape into bone lengths and the mean width of each part slice. This part-based parameterization technique achieves a balance between flexibility and validity using a semi-analytical shape reconstruction algorithm. Based on this new parameterization, a clothing-preserving data augmentation module is proposed to generate realistic images with diverse body shapes and accurate annotations. Experimental results show that our method outperforms other state-of-the-art methods in diverse body shape situations as well as in varied clothing situations.

----

## [93] MICA: Towards Explainable Skin Lesion Diagnosis via Multi-Level Image-Concept Alignment

**Authors**: *Yequan Bie, Luyang Luo, Hao Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27842](https://doi.org/10.1609/aaai.v38i2.27842)

**Abstract**:

Black-box deep learning approaches have showcased significant potential in the realm of medical image analysis. However, the stringent trustworthiness requirements intrinsic to the medical field have catalyzed research into the utilization of Explainable Artificial Intelligence (XAI), with a particular focus on concept-based methods. Existing concept-based methods predominantly apply concept annotations from a single perspective (e.g., global level), neglecting the nuanced semantic relationships between sub-regions and concepts embedded within medical images. This leads to underutilization of the valuable medical information and may cause models to fall short in harmoniously balancing interpretability and performance when employing inherently interpretable architectures such as Concept Bottlenecks. To mitigate these shortcomings, we propose a multi-modal explainable disease diagnosis framework that meticulously aligns medical images and clinical-related concepts semantically at multiple strata, encompassing the image level, token level, and concept level. Moreover, our method allows for model intervention and offers both textual and visual explanations in terms of human-interpretable concepts. Experimental results on three skin image datasets demonstrate that our method, while preserving model interpretability, attains high performance and label efficiency for concept detection and disease diagnosis. The code is available at https://github.com/Tommy-Bie/MICA.

----

## [94] VIXEN: Visual Text Comparison Network for Image Difference Captioning

**Authors**: *Alexander Black, Jing Shi, Yifei Fan, Tu Bui, John P. Collomosse*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27843](https://doi.org/10.1609/aaai.v38i2.27843)

**Abstract**:

We present VIXEN - a technique that succinctly summarizes in text the visual differences between a pair of images in order to highlight any content manipulation present. Our proposed network linearly maps image features in a pairwise manner, constructing a soft prompt for a pretrained large language model. We address the challenge of low volume of training data and lack of manipulation variety in existing image difference captioning (IDC) datasets by training on synthetically manipulated images from the recent InstructPix2Pix dataset generated via prompt-to-prompt editing framework. We augment this dataset with change summaries produced via GPT-3.  We show that VIXEN produces state-of-the-art, comprehensible difference captions for diverse image contents and edit types, offering a potential mitigation against misinformation disseminated via manipulated image content. Code and data are available at http://github.com/alexblck/vixen

----

## [95] SRFormer: Text Detection Transformer with Incorporated Segmentation and Regression

**Authors**: *Qingwen Bu, Sungrae Park, Minsoo Khang, Yichuan Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27844](https://doi.org/10.1609/aaai.v38i2.27844)

**Abstract**:

Existing techniques for text detection can be broadly classified into two primary groups: segmentation-based and regression-based methods. Segmentation models offer enhanced robustness to font variations but require intricate post-processing, leading to high computational overhead. Regression-based methods undertake instance-aware prediction but face limitations in robustness and data efficiency due to their reliance on high-level representations. In our academic pursuit, we propose SRFormer, a unified DETR-based model with amalgamated Segmentation and Regression, aiming at the synergistic harnessing of the inherent robustness in segmentation representations, along with the straightforward post-processing of instance-level regression. Our empirical analysis indicates that favorable segmentation predictions can be obtained at the initial decoder layers. In light of this, we constrain the incorporation of segmentation branches to the first few decoder layers and employ progressive regression refinement in subsequent layers, achieving performance gains while minimizing computational load from the mask. Furthermore, we propose a Mask-informed Query Enhancement module. We take the segmentation result as a natural soft-ROI to pool and extract robust pixel representations, which are then employed to enhance and diversify instance queries. Extensive experimentation across multiple benchmarks has yielded compelling findings, highlighting our method's exceptional robustness, superior training and data efficiency, as well as its state-of-the-art performance. Our code is available at https://github.com/retsuh-bqw/SRFormer-Text-Det.

----

## [96] Orthogonal Dictionary Guided Shape Completion Network for Point Cloud

**Authors**: *Pingping Cai, Deja Scott, Xiaoguang Li, Song Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27845](https://doi.org/10.1609/aaai.v38i2.27845)

**Abstract**:

Point cloud shape completion, which aims to reconstruct the missing regions of the incomplete point clouds with plausible shapes, is an ill-posed and challenging task that benefits many downstream 3D applications. Prior approaches achieve this goal by employing a two-stage completion framework, generating a coarse yet complete seed point cloud through an encoder-decoder network, followed by refinement and upsampling. However, the encoded features suffer from information loss of the missing portion, leading to an inability of the decoder to reconstruct seed points with detailed geometric clues. To tackle this issue, we propose a novel Orthogonal Dictionary Guided Shape Completion Network (ODGNet). The proposed ODGNet consists of a Seed Generation U-Net, which leverages multi-level feature extraction and concatenation to significantly enhance the representation capability of seed points, and Orthogonal Dictionaries that can learn shape priors from training samples and thus compensate for the information loss of the missing portions during inference. Our design is simple but to the point, extensive experiment results indicate that the proposed method can reconstruct point clouds with more details and outperform previous state-of-the-art counterparts. The implementation code is available at https://github.com/corecai163/ODGNet.

----

## [97] Spherical Pseudo-Cylindrical Representation for Omnidirectional Image Super-resolution

**Authors**: *Qing Cai, Mu Li, Dongwei Ren, Jun Lyu, Haiyong Zheng, Junyu Dong, Yee-Hong Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27846](https://doi.org/10.1609/aaai.v38i2.27846)

**Abstract**:

Omnidirectional images have attracted significant attention in recent years due to the rapid development of virtual reality technologies. Equirectangular projection (ERP), a naive form to store and transfer omnidirectional images, however, is challenging for existing two-dimensional (2D) image super-resolution (SR) methods due to its inhomogeneous distributed sampling density and distortion across latitude. In this paper, we make one of the first attempts to design a spherical pseudo-cylindrical representation, which not only allows pixels at different latitudes to adaptively adopt the best distinct sampling density but also is model-agnostic to most off-the-shelf SR methods, enhancing their performances. Specifically, we start by  upsampling each latitude of the input ERP image and design a computationally tractable optimization algorithm to adaptively obtain a (sub)-optimal sampling density for each latitude of the ERP image. Addressing the distortion of ERP, we introduce a new viewport-based training loss based on the original 3D sphere format of the omnidirectional image, which inherently lacks distortion. Finally, we present a simple yet effective recursive progressive omnidirectional SR network to showcase the feasibility of our idea. The experimental results on public datasets demonstrate the effectiveness of the proposed method as well as the consistently superior performance of our method over most state-of-the-art methods both quantitatively and qualitatively.

----

## [98] Disentangled Diffusion-Based 3D Human Pose Estimation with Hierarchical Spatial and Temporal Denoiser

**Authors**: *Qingyuan Cai, Xuecai Hu, Saihui Hou, Li Yao, Yongzhen Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27847](https://doi.org/10.1609/aaai.v38i2.27847)

**Abstract**:

Recently, diffusion-based methods for monocular 3D human pose estimation have achieved state-of-the-art (SOTA) performance by directly regressing the 3D joint coordinates from the 2D pose sequence. Although some methods decompose the task into bone length and bone direction prediction based on the human anatomical skeleton to explicitly incorporate more human body prior constraints, the performance of these methods is significantly lower than that of the SOTA diffusion-based methods. This can be attributed to the tree structure of the human skeleton. Direct application of the disentangled method could amplify the accumulation of hierarchical errors, propagating through each hierarchy. Meanwhile, the hierarchical information has not been fully explored by the previous methods. To address these problems, a Disentangled Diffusion-based 3D human Pose Estimation method with Hierarchical Spatial and Temporal Denoiser is proposed, termed DDHPose. In our approach: (1) We disentangle the 3d pose and diffuse the bone length and bone direction during the forward process of the diffusion model to effectively model the human pose prior. A disentanglement loss is proposed to supervise diffusion model learning. (2) For the reverse process, we propose Hierarchical Spatial and Temporal Denoiser (HSTDenoiser) to improve the hierarchical modelling of each joint. Our HSTDenoiser comprises two components: the Hierarchical-Related Spatial Transformer (HRST) and the Hierarchical-Related Temporal Transformer (HRTT). HRST exploits joint spatial information and the influence of the parent joint on each joint for spatial modeling, while HRTT utilizes information from both the joint and its hierarchical adjacent joints to explore the hierarchical temporal correlations among joints. Extensive experiments on the Human3.6M and MPI-INF-3DHP datasets show that our method outperforms the SOTA disentangled-based, non-disentangled based, and probabilistic approaches by 10.0%, 2.0%, and 1.3%, respectively.

----

## [99] Rethinking the Paradigm of Content Constraints in Unpaired Image-to-Image Translation

**Authors**: *Xiuding Cai, Yaoyao Zhu, Dong Miao, Linjie Fu, Yu Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27848](https://doi.org/10.1609/aaai.v38i2.27848)

**Abstract**:

In an unpaired setting, lacking sufficient content constraints for image-to-image translation (I2I) tasks, GAN-based approaches are usually prone to model collapse. Current solutions can be divided into two categories, reconstruction-based and Siamese network-based. The former requires that the transformed or transforming image can be perfectly converted back to the original image, which is sometimes too strict and limits the generative performance. The latter involves feeding the original and generated images into a feature extractor and then matching their outputs. This is not efficient enough, and a universal feature extractor is not easily available. In this paper, we propose EnCo, a simple but efficient way to maintain the content by constraining the representational similarity in the latent space of patch-level features from the same stage of the encoder and decoder of the generator. For the similarity function, we use a simple MSE loss instead of contrastive loss, which is currently widely used in I2I tasks. Benefits from the design, EnCo training is extremely efficient, while the features from the encoder produce a more positive effect on the decoding, leading to more satisfying generations. In addition, we rethink the role played by discriminators in sampling patches and propose a discriminative attention-guided (DAG) patch sampling strategy to replace random sampling. DAG is parameter-free and only requires negligible computational overhead, while significantly improving the performance of the model. Extensive experiments on multiple datasets demonstrate the effectiveness and advantages of EnCo, and we achieve multiple state-of-the-art compared to previous methods.

----

## [100] FusionFormer: A Concise Unified Feature Fusion Transformer for 3D Pose Estimation

**Authors**: *Yanlu Cai, Weizhong Zhang, Yuan Wu, Cheng Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27849](https://doi.org/10.1609/aaai.v38i2.27849)

**Abstract**:

Depth uncertainty is a core challenge in 3D human pose estimation, especially when the camera parameters are unknown. Previous methods try to reduce the impact of depth uncertainty by multi-view and/or multi-frame feature fusion to utilize more spatial and temporal information. However, they generally lead to marginal improvements and their performance still cannot match the camera-parameter-required methods. The reason is that their handcrafted fusion schemes cannot fuse the features flexibly, e.g., the multi-view and/or multi-frame features are fused separately. Moreover, the diverse and complicated fusion schemes make the principle for developing effective fusion schemes unclear and also raises an open problem that whether there exist more simple and elegant fusion schemes. To address these issues, this paper proposes an extremely concise unified feature fusion transformer (FusionFormer) with minimized handcrafted design for 3D pose estimation. FusionFormer fuses both the multi-view and multi-frame features in a unified fusion scheme, in which all the features are accessible to each other and thus can be fused flexibly.   Experimental results on several mainstream datasets demonstrate that FusionFormer achieves state-of-the-art performance. To our best knowledge, this is the first camera-parameter-free method to outperform the existing camera-parameter-required methods, revealing the tremendous potential of camera-parameter-free models. These impressive experimental results together with our concise feature fusion scheme resolve the above open problem. Another appealing feature of FusionFormer we observe is that benefiting from its effective fusion scheme, we can achieve impressive performance with smaller model size and less FLOPs.

----

## [101] Decoupled Textual Embeddings for Customized Image Generation

**Authors**: *Yufei Cai, Yuxiang Wei, Zhilong Ji, Jinfeng Bai, Hu Han, Wangmeng Zuo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27850](https://doi.org/10.1609/aaai.v38i2.27850)

**Abstract**:

Customized text-to-image generation, which aims to learn user-specified concepts with a few images, has drawn significant attention recently. However, existing methods usually suffer from overfitting issues and entangle the subject-unrelated information (e.g., background and pose) with the learned concept, limiting the potential to compose concept into new scenes. To address these issues, we propose the DETEX, a novel approach that learns the disentangled concept embedding for flexible customized text-to-image generation. Unlike conventional methods that learn a single concept embedding from the given images, our DETEX represents each image using multiple word embeddings during training, i.e., a learnable image-shared subject embedding and several image-specific subject-unrelated embeddings. To decouple irrelevant attributes (i.e., background and pose) from the subject embedding, we further present several attribute mappers that encode each image as several image-specific subject-unrelated embeddings. To encourage these unrelated embeddings to capture the irrelevant information, we incorporate them with corresponding attribute words and propose a joint training strategy to facilitate the disentanglement. During inference, we only use the subject embedding for image generation, while selectively using image-specific embeddings to retain image-specified attributes. Extensive experiments demonstrate that the subject embedding obtained by our method can faithfully represent the target concept, while showing superior editability compared to the state-of-the-art methods. Our code will be available at https://github.com/PrototypeNx/DETEX.

----

## [102] Disguise without Disruption: Utility-Preserving Face De-identification

**Authors**: *Zikui Cai, Zhongpai Gao, Benjamin Planche, Meng Zheng, Terrence Chen, M. Salman Asif, Ziyan Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27851](https://doi.org/10.1609/aaai.v38i2.27851)

**Abstract**:

With the rise of cameras and smart sensors, humanity generates an exponential amount of data. This valuable information, including underrepresented cases like AI in medical settings, can fuel new deep-learning tools. However, data scientists must prioritize ensuring privacy for individuals in these untapped datasets, especially for images or videos with faces, which are prime targets for identification methods. Proposed solutions to de-identify such images often compromise non-identifying facial attributes relevant to downstream tasks.
In this paper, we introduce Disguise, a novel algorithm that seamlessly de-identifies facial images while ensuring the usability of the modified data. Unlike previous approaches, our solution is firmly grounded in the domains of differential privacy and ensemble-learning research. Our method involves extracting and substituting depicted identities with synthetic ones, generated using variational mechanisms to maximize obfuscation and non-invertibility. Additionally, we leverage supervision from a mixture-of-experts to disentangle and preserve other utility attributes. We extensively evaluate our method using multiple datasets, demonstrating a higher de-identification rate and superior consistency compared to prior approaches in various downstream tasks.

----

## [103] Bi-directional Adapter for Multimodal Tracking

**Authors**: *Bing Cao, Junliang Guo, Pengfei Zhu, Qinghua Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27852](https://doi.org/10.1609/aaai.v38i2.27852)

**Abstract**:

Due to the rapid development of computer vision, single-modal (RGB) object tracking has made significant progress in recent years. Considering the limitation of single imaging
sensor, multi-modal images (RGB, infrared, etc.) are introduced to compensate for this deficiency for all-weather object tracking in complex environments. However, as acquiring sufficient multi-modal tracking data is hard while the dominant modality changes with the open environment, most existing techniques fail to extract multi-modal complementary information dynamically, yielding unsatisfactory tracking performance. To handle this problem, we propose a novel multi-modal visual prompt tracking model based on a universal bi-directional adapter, cross-prompting multiple modalities mutually. Our model consists of a universal bi-directional adapter and multiple modality-specific transformer encoder branches with sharing parameters. The encoders extract features of each modality separately by using a frozen, pre-trained foundation model. We develop a simple but effective light feature adapter to transfer modality-specific information from one modality to another, performing visual feature prompt fusion in an adaptive manner. With adding fewer (0.32M) trainable parameters, our model achieves superior tracking performance in comparison with both the full fine-tuning methods and the prompt learning-based methods. Our code is available: https://github.com/SparkTempest/BAT.

----

## [104] Domain-Controlled Prompt Learning

**Authors**: *Qinglong Cao, Zhengqin Xu, Yuntian Chen, Chao Ma, Xiaokang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27853](https://doi.org/10.1609/aaai.v38i2.27853)

**Abstract**:

Large pre-trained vision-language models, such as CLIP, have shown remarkable generalization capabilities across various tasks when appropriate text prompts are provided. However, adapting these models to specific domains, like remote sensing images (RSIs), medical images, etc, remains unexplored and challenging. Existing prompt learning methods often lack domain-awareness or domain-transfer mechanisms, leading to suboptimal performance due to the misinterpretation of specific images in natural image patterns.  To tackle this dilemma, we proposed a Domain-Controlled Prompt Learning for the specific domains. Specifically, the large-scale specific domain foundation model (LSDM) is first introduced to provide essential specific domain knowledge. Using lightweight neural networks, we transfer this knowledge into domain biases, which control both the visual and language branches to obtain domain-adaptive prompts in a directly incorporating manner.  Simultaneously, to overcome the existing overfitting challenge, we propose a novel noisy-adding strategy, without extra trainable parameters, to help the model escape the suboptimal solution in a global domain oscillation manner. Experimental results show our method achieves state-of-the-art performance in specific domain image recognition datasets. Our code is available at https://github.com/caoql98/DCPL.

----

## [105] LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer

**Authors**: *Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27854](https://doi.org/10.1609/aaai.v38i2.27854)

**Abstract**:

Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

----

## [106] Descanning: From Scanned to the Original Images with a Color Correction Diffusion Model

**Authors**: *Junghun Cha, Ali Haider, Seoyun Yang, Hoeyeong Jin, Subin Yang, A. F. M. Shahab Uddin, Jaehyoung Kim, Soo Ye Kim, Sung-Ho Bae*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27855](https://doi.org/10.1609/aaai.v38i2.27855)

**Abstract**:

A significant volume of analog information, i.e., documents and images, have been digitized in the form of scanned copies for storing, sharing, and/or analyzing in the digital world. However, the quality of such contents is severely degraded by various distortions caused by printing, storing, and scanning processes in the physical world. Although restoring high-quality content from scanned copies has become an indispensable task for many products, it has not been systematically explored, and to the best of our knowledge, no public datasets are available. In this paper, we define this problem as Descanning and introduce a new high-quality and large-scale dataset named DESCAN-18K. It contains 18K pairs of original and scanned images collected in the wild containing multiple complex degradations. In order to eliminate such complex degradations, we propose a new image restoration model called DescanDiffusion consisting of a color encoder that corrects the global color degradation and a conditional denoising diffusion probabilistic model (DDPM) that removes local degradations. To further improve the generalization ability of DescanDiffusion, we also design a synthetic data generation scheme by reproducing prominent degradations in scanned images. We demonstrate that our DescanDiffusion outperforms other baselines including commercial restoration products, objectively and subjectively, via comprehensive experiments and analyses.

----

## [107] Fine Structure-Aware Sampling: A New Sampling Training Scheme for Pixel-Aligned Implicit Models in Single-View Human Reconstruction

**Authors**: *Kennard Yanting Chan, Fayao Liu, Guosheng Lin, Chuan Sheng Foo, Weisi Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27856](https://doi.org/10.1609/aaai.v38i2.27856)

**Abstract**:

Pixel-aligned implicit models, such as PIFu, PIFuHD, and ICON, are used for single-view clothed human reconstruction. These models need to be trained using a sampling training scheme. Existing sampling training schemes either fail to capture thin surfaces (e.g. ears, fingers) or cause noisy artefacts in reconstructed meshes. To address these problems, we introduce Fine Structured-Aware Sampling (FSS), a new sampling training scheme to train pixel-aligned implicit models for single-view human reconstruction. FSS resolves the aforementioned problems by proactively adapting to the thickness and complexity of surfaces. In addition, unlike existing sampling training schemes, FSS shows how normals of sample points can be capitalized in the training process to improve results.
Lastly, to further improve the training process, FSS proposes a mesh thickness loss signal for pixel-aligned implicit models. It becomes computationally feasible to introduce this loss once a slight reworking of the pixel-aligned implicit function framework is carried out. Our results show that our methods significantly outperform SOTA methods qualitatively and quantitatively. Our code is publicly available at https://github.com/kcyt/FSS.

----

## [108] CMDA: Cross-Modal and Domain Adversarial Adaptation for LiDAR-Based 3D Object Detection

**Authors**: *Gyusam Chang, Wonseok Roh, Sujin Jang, Dongwook Lee, Daehyun Ji, Gyeongrok Oh, Jinsun Park, Jinkyu Kim, Sangpil Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27857](https://doi.org/10.1609/aaai.v38i2.27857)

**Abstract**:

Recent LiDAR-based 3D Object Detection (3DOD) methods show promising results, but they often do not generalize well to target domains outside the source (or training) data distribution. To reduce such domain gaps and thus to make 3DOD models more generalizable, we introduce a novel unsupervised domain adaptation (UDA) method, called CMDA, which (i) leverages visual semantic cues from an image modality (i.e., camera images) as an effective semantic bridge to close the domain gap in the cross-modal Bird's Eye View (BEV) representations. Further, (ii) we also introduce a self-training-based learning strategy, wherein a model is adversarially trained to generate domain-invariant features, which disrupt the discrimination of whether a feature instance comes from a source or an unseen target domain. Overall, our CMDA framework guides the 3DOD model to generate highly informative and domain-adaptive features for novel data distributions. In our extensive experiments with large-scale benchmarks, such as nuScenes, Waymo, and KITTI, those mentioned above provide significant performance gains for UDA tasks, achieving state-of-the-art performance.

----

## [109] A Hybrid Global-Local Perception Network for Lane Detection

**Authors**: *Qing Chang, Yifei Tong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27858](https://doi.org/10.1609/aaai.v38i2.27858)

**Abstract**:

Lane detection is a critical task in autonomous driving, which requires accurately predicting the complex topology of lanes in various scenarios. While previous methods of lane detection have shown success, challenges still exist, especially in scenarios where lane markings are absent. In this paper, we analyze the role of global and local features in accurately detecting lanes and propose a Hybrid Global-Local Perception Network (HGLNet) to leverage them. Global and local features play distinct roles in lane detection by respectively aiding in the detection of lane instances and the localization of corresponding lanes. HGLNet extracts global semantic context by utilizing a global extraction head that aggregates information about adaptive sampling points around lanes, achieving an optimal trade-off between performance and efficiency. Moreover, we introduce a Multi-hierarchy feature aggregator (MFA) to capture feature hierarchies in both regional and local ranges, elevating the representation of local features. The proposed Hybrid architecture can simultaneously focus on global and local features at different depth levels and efficiently integrate them to sense the global presence of lanes and accurately regress their locations. Experimental results demonstrate that our proposed method improves detection accuracy in various challenging scenarios, outperforming the state-of-the-art lane detection methods.

----

## [110] Improving Robustness for Joint Optimization of Camera Pose and Decomposed Low-Rank Tensorial Radiance Fields

**Authors**: *Bo-Yu Chen, Wei-Chen Chiu, Yu-Lun Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27859](https://doi.org/10.1609/aaai.v38i2.27859)

**Abstract**:

In this paper, we propose an algorithm that allows joint refinement of camera pose and scene geometry represented by decomposed low-rank tensor, using only 2D images as supervision. 

First, we conduct a pilot study based on a 1D signal and relate our findings to 3D scenarios, where the naive joint pose optimization on voxel-based NeRFs can easily lead to sub-optimal solutions.

Moreover, based on the analysis of the frequency spectrum, we propose to apply convolutional Gaussian filters on 2D and 3D radiance fields for a coarse-to-fine training schedule that enables joint camera pose optimization.

Leveraging the decomposition property in decomposed low-rank tensor, our method achieves an equivalent effect to brute-force 3D convolution with only incurring little computational overhead. 

To further improve the robustness and stability of joint optimization, we also propose techniques of smoothed 2D supervision, randomly scaled kernel parameters, and edge-guided loss mask. 

Extensive quantitative and qualitative evaluations demonstrate that our proposed framework achieves superior performance in novel view synthesis as well as rapid convergence for optimization.

The source code is available at https://github.com/Nemo1999/Joint-TensoRF.

----

## [111] Sketch and Refine: Towards Fast and Accurate Lane Detection

**Authors**: *Chao Chen, Jie Liu, Chang Zhou, Jie Tang, Gangshan Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27860](https://doi.org/10.1609/aaai.v38i2.27860)

**Abstract**:

Lane detection is to determine the precise location and shape of lanes on the road. Despite efforts made by current methods, it remains a challenging task due to the complexity of real-world scenarios. Existing approaches, whether proposal-based or keypoint-based, suffer from depicting lanes effectively and efficiently. Proposal-based methods detect lanes by distinguishing and regressing a collection of proposals in a streamlined top-down way, yet lack sufficient flexibility in lane representation. Keypoint-based methods, on the other hand, construct lanes flexibly from local descriptors, which typically entail complicated post-processing. In this paper, we present a “Sketch-and-Refine” paradigm that utilizes the merits of both keypoint-based and proposal-based methods. The motivation is that local directions of lanes are semantically simple and clear. At the “Sketch” stage, local directions of keypoints can be easily estimated by fast convolutional layers. Then we can build a set of lane proposals accordingly with moderate accuracy. At the “Refine” stage, we further optimize these proposals via a novel Lane Segment Association Module (LSAM), which allows adaptive lane segment adjustment. Last but not least, we propose multi-level feature integration to enrich lane feature representations more efficiently. Based on the proposed “Sketch-and-Refine” paradigm, we propose a fast yet effective lane detector dubbed “SRLane”. Experiments show that our SRLane can run at a fast speed (i.e., 278 FPS) while yielding an F1 score of 78.9%. The source code is available at: https://github.com/passerer/SRLane.

----

## [112] Iterative Token Evaluation and Refinement for Real-World Super-resolution

**Authors**: *Chaofeng Chen, Shangchen Zhou, Liang Liao, Haoning Wu, Wenxiu Sun, Qiong Yan, Weisi Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27861](https://doi.org/10.1609/aaai.v38i2.27861)

**Abstract**:

Real-world image super-resolution (RWSR) is a long-standing problem as low-quality (LQ) images often have complex and unidentified degradations. Existing methods such as Generative Adversarial Networks (GANs) or continuous diffusion models present their own issues including GANs being difficult to train while continuous diffusion models requiring numerous inference steps. In this paper, we propose an Iterative Token Evaluation and Refinement (ITER) framework for RWSR, which utilizes a discrete diffusion model operating in the discrete token representation space, i.e., indexes of features extracted from a VQGAN codebook pre-trained with high-quality (HQ) images. We show that ITER is easier to train than GANs and more efficient than continuous diffusion models. Specifically, we divide RWSR into two sub-tasks, i.e., distortion removal and texture generation. Distortion removal involves simple HQ token prediction with LQ images, while texture generation uses a discrete diffusion model to iteratively refine the distortion removal output with a token refinement network. In particular, we propose to include a token evaluation network in the discrete diffusion process. It learns to evaluate which tokens are good restorations and helps to improve the iterative refinement results. Moreover, the evaluation network can first check status of the distortion removal output and then adaptively select total refinement steps needed, thereby maintaining a good balance between distortion removal and texture generation. Extensive experimental results show that ITER is easy to train and performs well within just 8 iterative steps.

----

## [113] FeatWalk: Enhancing Few-Shot Classification through Local View Leveraging

**Authors**: *Dalong Chen, Jianjia Zhang, Wei-Shi Zheng, Ruixuan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27862](https://doi.org/10.1609/aaai.v38i2.27862)

**Abstract**:

Few-shot learning is a challenging task due to the limited availability of training samples. Recent few-shot learning studies with meta-learning and simple transfer learning methods have achieved promising performance. However, the feature extractor pre-trained with the upstream dataset may neglect the extraction of certain features which could be crucial for downstream tasks. In this study, inspired by the process of human learning in few-shot tasks, where humans not only observe the whole image (`global view') but also attend to various local image regions (`local view') for comprehensive understanding of detailed features, we propose a simple yet effective few-shot learning method called FeatWalk which can utilize the complementary nature of global and local views,  therefore providing an intuitive and effective solution to the problem of insufficient local information extraction from the pre-trained feature extractor. Our method can be easily and flexibly combined with various existing methods, further enhancing few-shot learning performance. Extensive experiments on multiple benchmark datasets consistently demonstrate the effectiveness and versatility of our method.The source code is available at https://github.com/exceefind/FeatWalk.

----

## [114] Real3D: The Curious Case of Neural Scene Degeneration

**Authors**: *Dengsheng Chen, Jie Hu, Xiaoming Wei, Enhua Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27863](https://doi.org/10.1609/aaai.v38i2.27863)

**Abstract**:

Despite significant progress in utilizing pre-trained text-to-image diffusion models to guide the creation of 3D scenes, these methods often struggle to generate scenes that are sufficiently realistic, leading to "neural scene degeneration". 
In this work, we propose a new 3D scene generation model called Real3D. 
Specifically, Real3D designs a pipeline from a NeRF-like implicit renderer to a tetrahedrons-based explicit renderer, greatly improving the neural network's ability to generate various neural scenes. 
Moreover, Real3D introduces an additional discriminator to prevent neural scenes from falling into undesirable local optima, thus avoiding the degeneration phenomenon.
Our experimental results demonstrate that Real3D outperforms all existing state-of-the-art text-to-3D generation methods, providing valuable insights to facilitate the development of learning-based 3D scene generation approaches.

----

## [115] DDAE: Towards Deep Dynamic Vision BERT Pretraining

**Authors**: *Honghao Chen, Xiangwen Kong, Xiangyu Zhang, Xin Zhao, Kaiqi Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27864](https://doi.org/10.1609/aaai.v38i2.27864)

**Abstract**:

Recently, masked image modeling (MIM) has demonstrated promising prospects in self-supervised representation learning. However, existing MIM frameworks recover all masked patches equivalently, ignoring that the reconstruction difficulty of different patches can vary sharply due to their diverse distance from visible patches. In this paper, we propose a novel deep dynamic supervision to enable MIM methods to dynamically reconstruct patches with different degrees of difficulty at different pretraining phases and depths of the model. Our deep dynamic supervision helps to provide more locality inductive bias for ViTs especially in deep layers, which inherently makes up for the absence of local prior for self-attention mechanism. Built upon the deep dynamic supervision, we propose Deep Dynamic AutoEncoder (DDAE), a simple yet effective MIM framework that utilizes dynamic mechanisms for pixel regression and feature self-distillation simultaneously. Extensive experiments across a variety of vision tasks including ImageNet classification, semantic segmentation on ADE20K and object detection on COCO demonstrate the effectiveness of our approach.

----

## [116] Rethinking Multi-Scale Representations in Deep Deraining Transformer

**Authors**: *Hongming Chen, Xiang Chen, Jiyang Lu, Yufeng Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27865](https://doi.org/10.1609/aaai.v38i2.27865)

**Abstract**:

Existing Transformer-based image deraining methods depend mostly on fixed single-input single-output U-Net architecture. In fact, this not only neglects the potentially explicit information from multiple image scales, but also lacks the capability of exploring the complementary implicit information across different scales. In this work, we rethink the multi-scale representations and design an effective multi-input multi-output framework that constructs intra- and inter-scale hierarchical modulation to better facilitate rain removal and help image restoration. We observe that rain levels reduce dramatically in coarser image scales, thus proposing to restore rain-free results from the coarsest scale to the finest scale in image pyramid inputs, which also alleviates the difficulty of model learning. Specifically, we integrate a sparsity-compensated Transformer block and a frequency-enhanced convolutional block into a coupled representation module, in order to jointly learn the intra-scale content-aware features. To facilitate representations learned at different scales to communicate with each other, we leverage a gated fusion module to adaptively aggregate the inter-scale spatial-aware features, which are rich in correlated information of rain appearances, leading to high-quality results. Extensive experiments demonstrate that our model achieves consistent gains on five benchmarks.

----

## [117] Unsupervised Group Re-identification via Adaptive Clustering-Driven Progressive Learning

**Authors**: *Hongxu Chen, Quan Zhang, Jian-Huang Lai, Xiaohua Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27866](https://doi.org/10.1609/aaai.v38i2.27866)

**Abstract**:

Group re-identification (G-ReID) aims to correctly associate groups with the same members captured by different cameras. However, supervised approaches for this task often suffer from the high cost of cross-camera sample labeling. Unsupervised methods based on clustering can avoid sample labeling, but the problem of member variations often makes clustering unstable, leading to incorrect pseudo-labels. To address these challenges, we propose an adaptive clustering-driven progressive learning approach (ACPL), which consists of a group adaptive clustering (GAC) module and a global dynamic prototype update (GDPU) module. Specifically, GAC designs the quasi-distance between groups, thus fully capitalizing on both individual-level and holistic information within groups. In the case of great uncertainty in intra-group members, GAC effectively minimizes the impact of non-discriminative features and reduces the noise in the model's pseudo-labels. Additionally, our GDPU devises a dynamic weight to update the prototypes and effectively mine the hard samples with complex member variations, which improves the model's robustness. Extensive experiments conducted on four popular G-ReID datasets demonstrate that our method not only achieves state-of-the-art performance on unsupervised G-ReID but also performs comparably to several fully supervised approaches.

----

## [118] Guiding a Harsh-Environments Robust Detector via RAW Data Characteristic Mining

**Authors**: *Hongyang Chen, Hung-Shuo Tai, Kaisheng Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27867](https://doi.org/10.1609/aaai.v38i2.27867)

**Abstract**:

Consumer-grade cameras capture the RAW physical description of a scene and then process the image signals to obtain high-quality RGB images that are faithful to human visual perception. Conventionally, dense prediction scenes require high-precision recognition of objects in RGB images. However, predicting RGB data to exhibit the expected adaptability and robustness in harsh environments can be challenging. By capitalizing on the broader color gamut and higher bit depth offered by RAW data, in this paper, we demonstrate that RAW data can significantly improve the accuracy and robustness of object detectors in harsh environments. Firstly, we propose a general Pipeline for RAW Detection (PRD), along with a preprocessing strategy tailored to RAW data. Secondly, we design the RAW Corruption Benchmark (RCB) to address the dearth of benchmarks that reflect realistic scenarios in harsh environments. Thirdly, we demonstrate the significant improvement of RAW images in object detection for low-light and corrupt scenes. Specifically, our experiments indicate that PRD (using FCOS) outperforms RGB detection by 13.9mAP on LOD-Snow without generating restored images. Finally, we introduce a new nonlinear method called Functional Regularization (FR), which can effectively mine the unique characteristics of RAW data. The code is available at https://github.com/DreamerCCC/RawMining.

----

## [119] CutFreq: Cut-and-Swap Frequency Components for Low-Level Vision Augmentation

**Authors**: *Hongyang Chen, Kaisheng Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27868](https://doi.org/10.1609/aaai.v38i2.27868)

**Abstract**:

Low-level vision plays a crucial role in a wide range of imaging quality and image recognition applications. However, the limited size, quality, and diversity of datasets often pose significant challenges for low-level tasks. Data augmentation is the most effective and practical way of sample expansion, but the commonly used augmentation methods in high-level tasks have limited improvement in the low-level due to the boundary effects or the non-realistic context information. In this paper, we propose the Cut-and-Swap Frequency Components (CutFreq) method for low-level vision, which aims to preserve high-level representations with directionality and improve image synthesis quality. Observing the significant frequency domain differences between reconstructed images and real ones, in CutFreq, we propose to transform the input and real images separately in the frequency domain, then define two stages for the model training process, and finally swap the specified frequency bands respectively and inversely transform to generate augmented samples. The experimental results show the superior performance of CutFreq on five low-level vision tasks. Moreover, we demonstrate the effectiveness of CutFreq in the low-data regime. Code is available at https://github.com/DreamerCCC/CutFreq.

----

## [120] Null Space Matters: Range-Null Decomposition for Consistent Multi-Contrast MRI Reconstruction

**Authors**: *Jiacheng Chen, Jiawei Jiang, Fei Wu, Jianwei Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27869](https://doi.org/10.1609/aaai.v38i2.27869)

**Abstract**:

Consistency and interpretability have long been the critical issues in MRI reconstruction. While interpretability has been dramatically improved with the employment of deep unfolding networks (DUNs), current methods still suffer from inconsistencies and generate inferior anatomical structure. Especially in multi-contrast scenes, different imaging protocols often exacerbate the concerned issue. In this paper, we propose a range-null decomposition-assisted DUN architecture to ensure consistency while still providing desirable interpretability. Given the input decomposed, we argue that the inconsistency could be analytically relieved by feeding solely the null-space component into proximal mapping, while leaving the range-space counterpart fixed. More importantly, a correlation decoupling scheme is further proposed to narrow the information gap for multi-contrast fusion, which dynamically borrows isotropic features from the opponent while maintaining the modality-specific ones. Specifically, the two features are attached to different frequencies and learned individually by the newly designed isotropy encoder and anisotropy encoder. The former strives for the contrast-shared information, while the latter serves to capture the contrast-specific features. The quantitative and qualitative results show that our proposal outperforms most cutting-edge methods by a large margin. Codes will be released on https://github.com/chenjiachengzzz/RNU.

----

## [121] PNeSM: Arbitrary 3D Scene Stylization via Prompt-Based Neural Style Mapping

**Authors**: *Jiafu Chen, Wei Xing, Jiakai Sun, Tianyi Chu, Yiling Huang, Boyan Ji, Lei Zhao, Huaizhong Lin, Haibo Chen, Zhizhong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27870](https://doi.org/10.1609/aaai.v38i2.27870)

**Abstract**:

3D scene stylization refers to transform the appearance of a 3D scene to match a given style image, ensuring that images rendered from different viewpoints exhibit the same style as the given style image, while maintaining the 3D consistency of the stylized scene. Several existing methods have obtained impressive results in stylizing 3D scenes. However, the mod- els proposed by these methods need to be re-trained when applied to a new scene. In other words, their models are cou- pled with a specific scene and cannot adapt to arbitrary other scenes. To address this issue, we propose a novel 3D scene stylization framework to transfer an arbitrary style to an ar- bitrary scene, without any style-related or scene-related re- training. Concretely, we first map the appearance of the 3D scene into a 2D style pattern space, which realizes complete disentanglement of the geometry and appearance of the 3D scene and makes our model be generalized to arbitrary 3D scenes. Then we stylize the appearance of the 3D scene in the 2D style pattern space via a prompt-based 2D stylization al- gorithm. Experimental results demonstrate that our proposed framework is superior to SOTA methods in both visual qual- ity and generalization.

----

## [122] TagFog: Textual Anchor Guidance and Fake Outlier Generation for Visual Out-of-Distribution Detection

**Authors**: *Jiankang Chen, Tong Zhang, Wei-Shi Zheng, Ruixuan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27871](https://doi.org/10.1609/aaai.v38i2.27871)

**Abstract**:

Out-of-distribution (OOD) detection is crucial in many real-world applications. However, intelligent models are often trained solely on in-distribution (ID) data, leading to overconfidence when misclassifying OOD data as ID classes.  In this study, we propose a new learning framework which leverage simple Jigsaw-based fake OOD data and rich semantic embeddings (`anchors') from the ChatGPT description of ID knowledge to help guide the training of the image encoder. The learning framework can be flexibly combined with existing post-hoc approaches to OOD detection, and extensive empirical evaluations on multiple OOD detection benchmarks demonstrate that rich textual representation of ID knowledge and fake OOD knowledge can well help train a visual encoder for OOD detection. With the learning framework, new state-of-the-art performance was achieved on all the benchmarks. The code is available at https://github.com/Cverchen/TagFog.

----

## [123] EVE: Efficient Vision-Language Pre-training with Masked Prediction and Modality-Aware MoE

**Authors**: *Junyi Chen, Longteng Guo, Jia Sun, Shuai Shao, Zehuan Yuan, Liang Lin, Dongyu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27872](https://doi.org/10.1609/aaai.v38i2.27872)

**Abstract**:

Building scalable vision-language models to learn from diverse, multimodal data remains an open challenge. In this paper, we introduce an Efficient Vision-languagE foundation model, namely EVE, which is one unified multimodal Transformer pre-trained solely by one unified pre-training task. Specifically, EVE encodes both vision and language within a shared Transformer network integrated with modality-aware sparse Mixture-of-Experts (MoE) modules, which capture modality-specific information by selectively switching to different experts. To unify pre-training tasks of vision and language, EVE performs masked signal modeling on image-text pairs to reconstruct masked signals, i.e., image pixels and text tokens, given visible signals. This simple yet effective pre-training objective accelerates training by 4x compared to the model pre-trained with Image-Text Contrastive and Image-Text Matching losses. Owing to the combination of the unified architecture and pre-training task, EVE is easy to scale up, enabling better downstream performance with fewer resources and faster training speed. Despite its simplicity, EVE achieves state-of-the-art performance on various vision-language downstream tasks, including visual question answering, visual reasoning, and image-text retrieval.

----

## [124] CaMIL: Causal Multiple Instance Learning for Whole Slide Image Classification

**Authors**: *Kaitao Chen, Shiliang Sun, Jing Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27873](https://doi.org/10.1609/aaai.v38i2.27873)

**Abstract**:

Whole slide image (WSI) classification is a crucial component in automated pathology analysis. Due to the inherent challenges of high-resolution WSIs and the absence of patch-level labels, most of the proposed methods follow the multiple instance learning (MIL) formulation. While MIL has been equipped with excellent instance feature extractors and aggregators, it is prone to learn spurious associations that undermine the performance of the model. For example, relying solely on color features may lead to erroneous diagnoses due to spurious associations between the disease and the color of patches. To address this issue, we develop a causal MIL framework for WSI classification, effectively distinguishing between causal and spurious associations. Specifically, we use the expectation of the intervention P(Y | do(X)) for bag prediction rather than the traditional likelihood P(Y | X). By applying the front-door adjustment, the spurious association is effectively blocked, where the intervened mediator is aggregated from patch-level features. We evaluate our proposed method on two publicly available WSI datasets, Camelyon16 and TCGA-NSCLC. Our causal MIL framework shows outstanding performance and is plug-and-play, seamlessly integrating with various feature extractors and aggregators.

----

## [125] Multi-Prototype Space Learning for Commonsense-Based Scene Graph Generation

**Authors**: *Lianggangxu Chen, Youqi Song, Yiqing Cai, Jiale Lu, Yang Li, Yuan Xie, Changbo Wang, Gaoqi He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27874](https://doi.org/10.1609/aaai.v38i2.27874)

**Abstract**:

In the domain of scene graph generation, modeling commonsense as a single-prototype representation has been typically employed to facilitate the recognition of infrequent predicates. However, a fundamental challenge lies in the large intra-class variations of the visual appearance of predicates, resulting in subclasses within a predicate class. Such a challenge typically leads to the problem of misclassifying diverse predicates due to the rough predicate space clustering. In this paper, inspired by cognitive science, we maintain multi-prototype representations for each predicate class, which can accurately find the multiple class centers of the predicate space. Technically, we propose a novel multi-prototype learning framework consisting of three main steps: prototype-predicate matching, prototype updating, and prototype space optimization. We first design a triple-level optimal transport to match each predicate feature within the same class to a specific prototype. In addition, the prototypes are updated using momentum updating to find the class centers according to the matching results. Finally, we enhance the inter-class separability of the prototype space through iterations of the inter-class separability loss and intra-class compactness loss. Extensive evaluations demonstrate that our approach significantly outperforms state-of-the-art methods on the Visual Genome dataset.

----

## [126] Kumaraswamy Wavelet for Heterophilic Scene Graph Generation

**Authors**: *Lianggangxu Chen, Youqi Song, Shaohui Lin, Changbo Wang, Gaoqi He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27875](https://doi.org/10.1609/aaai.v38i2.27875)

**Abstract**:

Graph neural networks (GNNs) has demonstrated its capabilities in the field of scene graph generation (SGG) by updating node representations from neighboring nodes. Actually it can be viewed as a form of low-pass filter in the spatial domain, which smooths node feature representation and retains commonalities among nodes. However, spatial GNNs does not work well in the case of heterophilic SGG in which fine-grained predicates are always connected to a large number of coarse-grained predicates. Blind smoothing undermines the discriminative information of the fine-grained predicates, resulting in failure to predict them accurately. To address the heterophily, our key idea is to design tailored filters by wavelet transform from the spectral domain. First, we prove rigorously that when the heterophily on the scene graph increases, the spectral energy gradually shifts towards the high-frequency part. Inspired by this observation, we subsequently propose the Kumaraswamy Wavelet Graph Neural Network (KWGNN). KWGNN leverages complementary multi-group Kumaraswamy wavelets to cover all frequency bands. Finally, KWGNN adaptively generates band-pass filters and then integrates the filtering results to better accommodate varying levels of smoothness on the graph. Comprehensive experiments on the Visual Genome and Open Images datasets show that our method achieves state-of-the-art performance.

----

## [127] ViT-Calibrator: Decision Stream Calibration for Vision Transformer

**Authors**: *Lin Chen, Zhijie Jia, Lechao Cheng, Yang Gao, Jie Lei, Yijun Bei, Zunlei Feng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27876](https://doi.org/10.1609/aaai.v38i2.27876)

**Abstract**:

A surge of interest has emerged in utilizing Transformers in diverse vision tasks owing to its formidable performance. However, existing approaches primarily focus on optimizing internal model architecture designs that often entail significant trial and error with high burdens. In this work, we propose a new paradigm dubbed Decision Stream Calibration that boosts the performance of general Vision Transformers. To achieve this, we shed light on the information propagation mechanism in the learning procedure by exploring the correlation between different tokens and the relevance coefficient of multiple dimensions. Upon further analysis, it was discovered that 1) the final decision is associated with tokens of foreground targets, while token features of foreground target will be transmitted into the next layer as much as possible, and the useless token features of background area will be eliminated gradually in the forward propagation. 2) Each category is solely associated with specific sparse dimensions in the tokens. Based on the discoveries mentioned above, we designed a two-stage calibration scheme, namely ViT-Calibrator, including token propagation calibration stage and dimension propagation calibration stage. Extensive experiments on commonly used datasets show that the proposed approach can achieve promising results.

----

## [128] NeRF-VPT: Learning Novel View Representations with Neural Radiance Fields via View Prompt Tuning

**Authors**: *Linsheng Chen, Guangrun Wang, Liuchun Yuan, Keze Wang, Ken Deng, Philip H. S. Torr*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27877](https://doi.org/10.1609/aaai.v38i2.27877)

**Abstract**:

Neural Radiance Fields (NeRF) have garnered remarkable success in novel view synthesis. Nonetheless, the task of generating high-quality images for novel views persists as a critical challenge. While the existing efforts have exhibited commendable progress, capturing intricate details, enhancing textures, and achieving superior Peak Signal-to-Noise Ratio (PSNR) metrics warrant further focused attention and advancement. In this work, we propose NeRF-VPT, an innovative method for novel view synthesis to address these challenges. Our proposed NeRF-VPT employs a cascading view prompt tuning paradigm, wherein RGB information gained from preceding rendering outcomes serves as instructive visual prompts for subsequent rendering stages, with the aspiration that the prior knowledge embedded in the prompts can facilitate the gradual enhancement of rendered image quality. NeRF-VPT only requires sampling RGB data from previous stage renderings as priors at each training stage, without relying on extra guidance or complex techniques. Thus, our NeRF-VPT is plug-and-play and can be readily integrated into existing methods. By conducting comparative analyses of our NeRF-VPT against several NeRF-based approaches on demanding real-scene benchmarks, such as Realistic Synthetic 360, Real Forward-Facing, Replica dataset, and a user-captured dataset, we substantiate that our NeRF-VPT significantly elevates baseline performance and proficiently generates more high-quality novel view images than all the compared state-of-the-art methods. Furthermore, the cascading learning of NeRF-VPT introduces adaptability to scenarios with sparse inputs, resulting in a significant enhancement of accuracy for sparse-view novel view synthesis. The source code and dataset are available at https://github.com/Freedomcls/NeRF-VPT.

----

## [129] WebVLN: Vision-and-Language Navigation on Websites

**Authors**: *Qi Chen, Dileepa Pitawela, Chongyang Zhao, Gengze Zhou, Hsiang-Ting Chen, Qi Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27878](https://doi.org/10.1609/aaai.v38i2.27878)

**Abstract**:

Vision-and-Language Navigation (VLN) task aims to enable AI agents to accurately understand and follow natural language instructions to navigate through real-world environments, ultimately reaching specific target locations. We recognise a promising opportunity to extend VLN to a comparable navigation task that holds substantial significance in our daily lives, albeit within the virtual realm: navigating websites on the Internet. This paper proposes a new task named Vision-and-Language Navigation on Websites (WebVLN), where we use question-based instructions to train an agent, emulating how users naturally browse websites. Unlike the existing VLN task that only pays attention to vision and instruction (language), the WebVLN agent further considers underlying web-specific content like HTML, which could not be seen on the rendered web pages yet contain rich visual and textual information. Toward this goal, we contribute a dataset, WebVLN-v1, and introduce a novel approach called Website-aware VLN Network (WebVLN-Net), which is built upon the foundation of state-of-the-art VLN techniques. Experimental results show that WebVLN-Net outperforms current VLN and web-related navigation methods. We believe that the introduction of the newWebVLN task and its dataset will establish a new dimension within the VLN domain and contribute to the broader vision-and-language research community. Code is available at: https://github.com/WebVLN/WebVLN.

----

## [130] Learning Multimodal Volumetric Features for Large-Scale Neuron Tracing

**Authors**: *Qihua Chen, Xuejin Chen, Chenxuan Wang, Yixiong Liu, Zhiwei Xiong, Feng Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27879](https://doi.org/10.1609/aaai.v38i2.27879)

**Abstract**:

The current neuron reconstruction pipeline for electron microscopy (EM) data usually includes automatic image segmentation followed by extensive human expert proofreading. In this work, we aim to reduce human workload by predicting connectivity between over-segmented neuron pieces, taking both microscopy image and 3D morphology features into account, similar to human proofreading workflow. To this end, we first construct a dataset, named FlyTracing, that contains millions of pairwise connections of segments expanding the whole fly brain, which is three orders of magnitude larger than existing datasets for neuron segment connection. To learn sophisticated biological imaging features from the connectivity annotations, we propose a novel connectivity-aware contrastive learning method to generate dense volumetric EM image embedding. The learned embeddings can be easily incorporated with any point or voxel-based morphological representations for automatic neuron tracing. Extensive comparisons of different combination schemes of image and morphological representation in identifying split errors across the whole fly brain demonstrate the superiority of the proposed approach, especially for the locations that contain severe imaging artifacts, such as section missing and misalignment. The dataset and code are available at https://github.com/Levishery/Flywire-Neuron-Tracing.

----

## [131] M-BEV: Masked BEV Perception for Robust Autonomous Driving

**Authors**: *Siran Chen, Yue Ma, Yu Qiao, Yali Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27880](https://doi.org/10.1609/aaai.v38i2.27880)

**Abstract**:

3D perception is a critical problem in autonomous driving. Recently, the Bird’s-Eye-View (BEV) approach has attracted extensive attention, due to low-cost deployment and desirable vision detection capacity. However, the existing models ignore a realistic scenario during the driving procedure, i.e., one or more view cameras may be failed, which largely deteriorates their performance. To tackle this problem, we propose a generic Masked BEV (M-BEV) perception framework, which can effectively improve robustness to this challenging scenario, by random masking and reconstructing camera views in the end-to-end training. More specifically, we develop a novel Masked View Reconstruction (MVR) module in our M-BEV. It mimics various missing cases by randomly masking features of different camera views, then leverages the original features of these views as self-supervision and reconstructs the masked ones with the distinct spatio-temporal context across camera views. Via such a plug-and-play MVR, our M-BEV is capable of learning the missing views from the resting ones, and thus well generalized for robust view recovery and accurate perception in the testing. We perform extensive experiments on the popular NuScenes benchmark, where our framework can significantly boost 3D perception performance of the state-of-the-art models on various missing view cases, e.g., for the absence of back view, our M-BEV promotes the PETRv2 model with 10.3% mAP gain.

----

## [132] VPDETR: End-to-End Vanishing Point DEtection TRansformers

**Authors**: *Taiyan Chen, Xianghua Ying, Jinfa Yang, Ruibin Wang, Ruohao Guo, Bowei Xing, Ji Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27881](https://doi.org/10.1609/aaai.v38i2.27881)

**Abstract**:

In the field of vanishing point detection, previous works commonly relied on extracting and clustering straight lines or classifying candidate points as vanishing points. This paper proposes a novel end-to-end framework, called VPDETR (Vanishing Point DEtection TRansformer), that views vanishing point detection as a set prediction problem, applicable to both Manhattan and non-Manhattan world datasets. By using the positional embedding of anchor points as queries in Transformer decoders and dynamically updating them layer by layer, our method is able to directly input images and output their vanishing points without the need for explicit straight line extraction and candidate points sampling. Additionally, we introduce an orthogonal loss and a cross-prediction loss to improve accuracy on the Manhattan world datasets. Experimental results demonstrate that VPDETR achieves competitive performance compared to state-of-the-art methods, without requiring post-processing.

----

## [133] TCI-Former: Thermal Conduction-Inspired Transformer for Infrared Small Target Detection

**Authors**: *Tianxiang Chen, Zhentao Tan, Qi Chu, Yue Wu, Bin Liu, Nenghai Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27882](https://doi.org/10.1609/aaai.v38i2.27882)

**Abstract**:

Infrared small target detection (ISTD) is critical to national security and has been extensively applied in military areas. ISTD aims to segment small target pixels from background. Most ISTD networks focus on designing feature extraction blocks or feature fusion modules, but rarely describe the ISTD process from the feature map evolution perspective. In the ISTD process, the network attention gradually shifts towards target areas. We abstract this process as the directional movement of feature map pixels to target areas through convolution, pooling and interactions with surrounding pixels, which can be analogous to the movement of thermal particles constrained by surrounding variables and particles. In light of this analogy, we propose Thermal Conduction-Inspired Transformer (TCI-Former) based on the theoretical principles of thermal conduction. According to thermal conduction differential equation in heat dynamics, we derive the pixel movement differential equation (PMDE) in the image domain and further develop two modules: Thermal Conduction-Inspired Attention (TCIA) and Thermal Conduction Boundary Module (TCBM). TCIA incorporates finite difference method with PMDE to reach a numerical approximation so that target body features can be extracted. To further remove errors in boundary areas, TCBM is designed and supervised by boundary masks to refine target body features with fine boundary details. Experiments on IRSTD-1k and NUAA-SIRST demonstrate the superiority of our method.

----

## [134] Intrinsic Phase-Preserving Networks for Depth Super Resolution

**Authors**: *Xuanhong Chen, Hang Wang, Jialiang Chen, Kairui Feng, Jinfan Liu, Xiaohang Wang, Weimin Zhang, Bingbing Ni*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27883](https://doi.org/10.1609/aaai.v38i2.27883)

**Abstract**:

Depth map super-resolution (DSR) plays an indispensable role in 3D vision. We discover an non-trivial spectral phenomenon: the components of high-resolution (HR) and low-resolution (LR) depth maps manifest the same intrinsic phase, and the spectral phase of RGB is a superset of them, which suggests that a phase-aware filter can assist in the precise use of RGB cues. Motivated by this, we propose an intrinsic phase-preserving DSR paradigm, named IPPNet, to fully exploit inter-modality collaboration in a mutually guided way. In a nutshell, a novel Phase-Preserving Filtering Module (PPFM) is developed to generate dynamic phase-aware filters according to the LR depth flow to filter out erroneous noisy components contained in RGB and then conduct depth enhancement via the modulation of the phase-preserved RGB signal.  By stacking multiple PPFM blocks, the proposed IPPNet is capable of reaching a highly competitive restoration performance.  Extensive experiments on various benchmark datasets, e.g., NYU v2, RGB-D-D, reach SOTA performance and also well demonstrate the validity of the proposed phase-preserving scheme. Code: https://github.com/neuralchen/IPPNet/.

----

## [135] Box2Poly: Memory-Efficient Polygon Prediction of Arbitrarily Shaped and Rotated Text

**Authors**: *Xuyang Chen, Dong Wang, Konrad Schindler, Mingwei Sun, Yongliang Wang, Nicoló Savioli, Liqiu Meng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27884](https://doi.org/10.1609/aaai.v38i2.27884)

**Abstract**:

Recently, Transformer-based text detection techniques have sought to predict polygons by encoding the coordinates of individual boundary vertices using distinct query features. However, this approach incurs a significant memory overhead and struggles to effectively capture the intricate relationships between vertices belonging to the same instance. Consequently, irregular text layouts often lead to the prediction of outlined vertices, diminishing the quality of results. To address these challenges, we present an innovative approach rooted in Sparse R-CNN: a cascade decoding pipeline for polygon prediction. Our method ensures precision by iteratively refining polygon predictions, considering both the scale and location of preceding results. Leveraging this stabilized regression pipeline, even employing just a single feature vector to guide polygon instance regression yields promising detection results. Simultaneously, the leverage of instance-level feature proposal substantially enhances memory efficiency ( > 50% less vs. the SOTA method DPText-DETR) and reduces inference speed (> 40% less  vs. DPText-DETR) with comparable performance on benchmarks. The code is available at https://github.com/Albertchen98/Box2Poly.git.

----

## [136] FashionERN: Enhance-and-Refine Network for Composed Fashion Image Retrieval

**Authors**: *Yanzhe Chen, Huasong Zhong, Xiangteng He, Yuxin Peng, Jiahuan Zhou, Lele Cheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27885](https://doi.org/10.1609/aaai.v38i2.27885)

**Abstract**:

The goal of composed fashion image retrieval is to locate a target image based on a reference image and modified text. Recent methods utilize symmetric encoders (e.g., CLIP) pre-trained on large-scale non-fashion datasets. However, the input for this task exhibits an asymmetric nature, where the reference image contains rich content while the modified text is often brief. Therefore, methods employing symmetric encoders encounter a severe phenomenon: retrieval results dominated by reference images, leading to the oversight of modified text. We propose a Fashion Enhance-and-Refine Network (FashionERN) centered around two aspects: enhancing the text encoder and refining visual semantics. We introduce a Triple-branch Modifier Enhancement model, which injects relevant information from the reference image and aligns the modified text modality with the target image modality. Furthermore, we propose a Dual-guided Vision Refinement model that retains critical visual information through text-guided refinement and self-guided refinement processes. The combination of these two models significantly mitigates the reference dominance phenomenon, ensuring accurate fulfillment of modifier requirements. Comprehensive experiments demonstrate our approach's state-of-the-art performance on four commonly used datasets.

----

## [137] IT3D: Improved Text-to-3D Generation with Explicit View Synthesis

**Authors**: *Yiwen Chen, Chi Zhang, Xiaofeng Yang, Zhongang Cai, Gang Yu, Lei Yang, Guosheng Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27886](https://doi.org/10.1609/aaai.v38i2.27886)

**Abstract**:

Recent strides in Text-to-3D techniques have been propelled by distilling knowledge from powerful large text-to-image diffusion models (LDMs). Nonetheless, existing Text-to-3D approaches often grapple with challenges such as over-saturation, inadequate detailing, and unrealistic outputs. This study presents a novel strategy that leverages explicitly synthesized multi-view images to address these issues. Our approach involves the utilization of image-to-image pipelines, empowered by LDMs, to generate posed high-quality images based on the renderings of coarse 3D models. Although the generated images mostly alleviate the aforementioned issues, challenges such as view inconsistency and significant content variance persist due to the inherent generative nature of large diffusion models, posing extensive difficulties in leveraging these images effectively. To overcome this hurdle, we advocate integrating a discriminator alongside a novel Diffusion-GAN dual training strategy to guide the training of 3D models. For the incorporated discriminator, the synthesized multi-view images are considered real data, while the renderings of the optimized 3D models function as fake data. We conduct a comprehensive set of experiments that demonstrate the effectiveness of our method over baseline approaches.

----

## [138] Beyond the Label Itself: Latent Labels Enhance Semi-supervised Point Cloud Panoptic Segmentation

**Authors**: *Yujun Chen, Xin Tan, Zhizhong Zhang, Yanyun Qu, Yuan Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27887](https://doi.org/10.1609/aaai.v38i2.27887)

**Abstract**:

As the exorbitant expense of labeling autopilot datasets and the growing trend of utilizing unlabeled data, semi-supervised segmentation on point clouds becomes increasingly imperative. Intuitively, finding out more ``unspoken words'' (i.e., latent instance information) beyond the label itself should be helpful to improve performance. In this paper, we discover two types of latent labels behind the displayed label embedded in LiDAR and image data. First, in the LiDAR Branch, we propose a novel augmentation, Cylinder-Mix, which is able to augment more yet reliable samples for training. Second, in the Image Branch, we propose the Instance Position-scale Learning (IPSL) Module to learn and fuse the information of instance position and scale, which is from a 2D pre-trained detector and a type of latent label obtained from 3D to 2D projection. Finally, the two latent labels are embedded into the multi-modal panoptic segmentation network. The ablation of the IPSL module demonstrates its robust adaptability, and the experiments evaluated on SemanticKITTI and nuScenes demonstrate that our model outperforms the state-of-the-art method, LaserMix.

----

## [139] Visual Chain-of-Thought Prompting for Knowledge-Based Visual Reasoning

**Authors**: *Zhenfang Chen, Qinhong Zhou, Yikang Shen, Yining Hong, Zhiqing Sun, Dan Gutfreund, Chuang Gan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27888](https://doi.org/10.1609/aaai.v38i2.27888)

**Abstract**:

Knowledge-based visual reasoning remains a daunting task since it not only requires machines to interpret the concepts and relationships from visual scenes but also associate them with external world knowledge to conduct a chain of reasoning on open-world questions. Previous works, however, treat visual perception and language-based reasoning as two independent modules, failing to attend to both modules throughout all stages of reasoning. To this end, we propose Visual Chain-of-thought Prompting (VCTP) for knowledge-based reasoning, which involves the interaction between visual content and natural language in an iterative step-by-step reasoning manner. VCTP contains three stages, see, think, and confirm. The see stage scans the image and grounds the visual concept candidates with a visual perception model. The think stage adopts a pre-trained large language model (LLM) to attend to key visual concepts from natural language questions adaptively. It then transforms key visual context into text context for prompting with a visual captioning model, and adopts the LLM to generate the answer. The confirm stage further uses the LLM to generate the supporting rationale to the answer, which is then passed through a cross-modality classifier to verify that it’s consistent with the visual context. We iterate through the think-confirm stages to ensure the verified rationale is consistent with the answer. We conduct experiments on a range of knowledge-based visual reasoning datasets. We found our VCTP enjoys several benefits, 1). it achieves better performance than the previous few-shot learning baselines; 2). it enjoys the total transparency and trustworthiness of the whole reasoning process by providing rationales for each reasoning step; 3). it is computation-efficient compared with other fine-tuning baselines. Our code is available at https://github.com/UMass-Foundation-Model/VisualCoT.git

----

## [140] Blind Face Restoration under Extreme Conditions: Leveraging 3D-2D Prior Fusion for Superior Structural and Texture Recovery

**Authors**: *Zhengrui Chen, Liying Lu, Ziyang Yuan, Yiming Zhu, Yu Li, Chun Yuan, Weihong Deng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27889](https://doi.org/10.1609/aaai.v38i2.27889)

**Abstract**:

Blind face restoration under extreme conditions involves reconstructing high-quality face images from severely degraded inputs. These input images are often in poor quality and have extreme facial poses, leading to errors in facial structure and unnatural artifacts within the restored images. In this paper, we show that utilizing 3D priors effectively compensates for structure knowledge deficiencies in 2D priors while preserving the texture details. Based on this, we introduce FREx (Face Restoration under Extreme conditions) that combines structure-accurate 3D priors and texture-rich 2D priors in pretrained generative networks for blind face restoration under extreme conditions. To fuse the different information in 3D and 2D priors, we introduce an adaptive weight module that adjusts the importance of features based on the input image's condition. With this approach, our model can restore structure-accurate and natural-looking faces even when the images have lost a lot of information due to degradation and extreme pose. Extensive experimental results on synthetic and real-world datasets validate the effectiveness of our methods.

----

## [141] CamoDiffusion: Camouflaged Object Detection via Conditional Diffusion Models

**Authors**: *Zhongxi Chen, Ke Sun, Xianming Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27890](https://doi.org/10.1609/aaai.v38i2.27890)

**Abstract**:

Camouflaged Object Detection (COD) is a challenging task in computer vision due to the high similarity between camouflaged objects and their surroundings. Existing COD methods struggle with nuanced object boundaries and overconfident incorrect predictions. In response, we propose a new paradigm that treats COD as a conditional mask-generation task leveraging diffusion models. Our method, dubbed CamoDiffusion, employs the denoising process to progressively refine predictions while incorporating image conditions. Due to the stochastic sampling process of diffusion, our model is capable of sampling multiple possible predictions, avoiding the problem of overconfident point estimation. Moreover, we develop specialized network architecture, training, and sampling strategies, to enhance the model’s expressive power, refinement capabilities and suppress overconfident mis-segmentations, thus aptly tailoring the diffusion model to the demands of COD. Extensive experiments on three COD datasets attest to the superior performance of our model compared to existing state-of-the-art methods, particularly on the most challenging COD10K dataset, where our approach achieves 0.019 in terms of MAE. Codes and models are available at https://github.com/Rapisurazurite/CamoDiffusion.

----

## [142] DreamIdentity: Enhanced Editability for Efficient Face-Identity Preserved Image Generation

**Authors**: *Zhuowei Chen, Shancheng Fang, Wei Liu, Qian He, Mengqi Huang, Zhendong Mao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27891](https://doi.org/10.1609/aaai.v38i2.27891)

**Abstract**:

While large-scale pre-trained text-to-image models can synthesize diverse and high-quality human-centric images, an intractable problem is how to preserve the face identity and follow the text prompts simultaneously for conditioned input face images and texts. Despite existing encoder-based methods achieving high efficiency and decent face similarity, the generated image often fails to follow the textual prompts. To ease this editability issue, we present DreamIdentity, to learn edit-friendly and accurate face-identity representations in the word embedding space. Specifically, we propose self-augmented editability learning to enhance the editability for projected embedding, which is achieved by constructing paired generated celebrity's face and edited celebrity images for training, aiming at transferring mature editability of off-the-shelf text-to-image models in celebrity to unseen identities. Furthermore, we design a novel dedicated face-identity encoder to learn an accurate representation of human faces, which applies multi-scale ID-aware features followed by a multi-embedding projector to generate the pseudo words in the text embedding space directly. Extensive experiments show that our method can generate more text-coherent and ID-preserved images with negligible time overhead compared to the standard text-to-image generation process.

----

## [143] Deep Linear Array Pushbroom Image Restoration: A Degradation Pipeline and Jitter-Aware Restoration Network

**Authors**: *Zida Chen, Ziran Zhang, Haoying Li, Menghao Li, Yueting Chen, Qi Li, Huajun Feng, Zhihai Xu, Shiqi Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27892](https://doi.org/10.1609/aaai.v38i2.27892)

**Abstract**:

Linear Array Pushbroom (LAP) imaging technology is widely used in the realm of remote sensing. However, images acquired through LAP always suffer from distortion and blur because of camera jitter. Traditional methods for restoring LAP images, such as algorithms estimating the point spread function (PSF), exhibit limited performance. To tackle this issue, we propose a Jitter-Aware Restoration Network (JARNet), to remove the distortion and blur in two stages. In the first stage, we formulate an Optical Flow Correction (OFC) block to refine the optical flow of the degraded LAP images, resulting in pre-corrected images where most of the distortions are alleviated. In the second stage, for further enhancement of the pre-corrected images, we integrate two jitter-aware techniques within the Spatial and Frequency Residual (SFRes) block: 1) introducing Coordinate Attention (CoA) to the SFRes block in order to capture the jitter state in orthogonal direction; 2) manipulating image features in both spatial and frequency domains to leverage local and global priors. Additionally, we develop a data synthesis pipeline, which applies Continue Dynamic Shooting Model (CDSM) to simulate realistic degradation in LAP images. Both the proposed JARNet and LAP image synthesis pipeline establish a foundation for addressing this intricate challenge. Extensive experiments demonstrate that the proposed two-stage method outperforms state-of-the-art image restoration models. Code is available at https://github.com/JHW2000/JARNet.

----

## [144] Context-Aware Iteration Policy Network for Efficient Optical Flow Estimation

**Authors**: *Ri Cheng, Ruian He, Xuhao Jiang, Shili Zhou, Weimin Tan, Bo Yan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27893](https://doi.org/10.1609/aaai.v38i2.27893)

**Abstract**:

Existing recurrent optical flow estimation networks are computationally expensive since they use a fixed large number of iterations to update the flow field for each sample. An efficient network should skip iterations when the flow improvement is limited. In this paper, we develop a Context-Aware Iteration Policy Network for efficient optical flow estimation, which determines the optimal number of iterations per sample. The policy network achieves this by learning contextual information to realize whether flow improvement is bottlenecked or minimal. On the one hand, we use iteration embedding and historical hidden cell, which include previous iterations information, to convey how flow has changed from previous iterations. On the other hand, we use the incremental loss to make the policy network implicitly perceive the magnitude of optical flow improvement in the subsequent iteration. Furthermore, the computational complexity in our dynamic network is controllable, allowing us to satisfy various resource preferences with a single trained model. Our policy network can be easily integrated into state-of-the-art optical flow networks. Extensive experiments show that our method maintains performance while reducing FLOPs by about 40%/20% for the Sintel/KITTI datasets.

----

## [145] SparseGNV: Generating Novel Views of Indoor Scenes with Sparse RGB-D Images

**Authors**: *Weihao Cheng, Yan-Pei Cao, Ying Shan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27894](https://doi.org/10.1609/aaai.v38i2.27894)

**Abstract**:

We study to generate novel views of indoor scenes given sparse input views. The challenge is to achieve both photorealism and view consistency. We present SparseGNV: a learning framework that incorporates 3D structures and image generative models to generate novel views with three modules. The first module builds a neural point cloud as underlying geometry, providing scene context and guidance for the target novel view. The second module utilizes a transformer-based network to map the scene context and the guidance into a shared latent space and autoregressively decodes the target view in the form of discrete image tokens. The third module reconstructs the tokens back to the image of the target view. SparseGNV is trained across a large-scale indoor scene dataset to learn generalizable priors. Once trained, it can efficiently generate novel views of an unseen indoor scene in a feed-forward manner. We evaluate SparseGNV on real-world indoor scenes and demonstrate that it outperforms state-of-the-art methods based on either neural radiance fields or conditional image generation.

----

## [146] Colorizing Monochromatic Radiance Fields

**Authors**: *Yean Cheng, Renjie Wan, Shuchen Weng, Chengxuan Zhu, Yakun Chang, Boxin Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27895](https://doi.org/10.1609/aaai.v38i2.27895)

**Abstract**:

Though Neural Radiance Fields (NeRF) can produce colorful 3D representations of the world by using a set of 2D images, such ability becomes non-existent when only monochromatic images are provided. Since color is necessary in representing the world, reproducing color from monochromatic radiance fields becomes crucial. To achieve this goal, instead of manipulating the monochromatic radiance fields directly, we consider it as a representation-prediction task in the Lab color space. By first constructing the luminance and density representation using monochromatic images, our prediction stage can recreate color representation on the basis of an image colorization module. We then reproduce a colorful implicit model through the representation of luminance, density, and color. Extensive experiments have been conducted to validate the effectiveness of our approaches. Our project page: https://liquidammonia.github.io/color-nerf.

----

## [147] Parallel Vertex Diffusion for Unified Visual Grounding

**Authors**: *Zesen Cheng, Kehan Li, Peng Jin, Siheng Li, Xiangyang Ji, Li Yuan, Chang Liu, Jie Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27896](https://doi.org/10.1609/aaai.v38i2.27896)

**Abstract**:

Unified visual grounding (UVG) capitalizes on a wealth of task-related knowledge across various grounding tasks via one-shot training, which curtails retraining costs and task-specific architecture design efforts. Vertex generation-based UVG methods achieve this versatility by unified modeling object box and contour prediction and provide a text-powered interface to vast related multi-modal tasks, e.g., visual question answering and captioning. However, these methods typically generate vertexes sequentially through autoregression, which is prone to be trapped in error accumulation and heavy computation, especially for high-dimension sequence generation in complex scenarios. In this paper, we develop Parallel Vertex Diffusion (PVD) based on the parallelizability of diffusion models to accurately and efficiently generate vertexes in a parallel and scalable manner. Since the coordinates fluctuate greatly, it typically encounters slow convergence when training diffusion models without geometry constraints. Therefore, we consummate our PVD by two critical components, i.e., center anchor mechanism and angle summation loss, which serve to normalize coordinates and adopt a differentiable geometry descriptor from the point-in-polygon problem of computational geometry to constrain the overall difference of prediction and label vertexes. These innovative designs empower our PVD to demonstrate its superiority with state-of-the-art performance across various grounding tasks.

----

## [148] iDet3D: Towards Efficient Interactive Object Detection for LiDAR Point Clouds

**Authors**: *Dongmin Choi, Wonwoo Cho, Kangyeol Kim, Jaegul Choo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27897](https://doi.org/10.1609/aaai.v38i2.27897)

**Abstract**:

Accurately annotating multiple 3D objects in LiDAR scenes is laborious and challenging. While a few previous studies have attempted to leverage semi-automatic methods for cost-effective bounding box annotation, such methods have limitations in efficiently handling numerous multi-class objects. To effectively accelerate 3D annotation pipelines, we propose iDet3D, an efficient interactive 3D object detector. Supporting a user-friendly 2D interface, which can ease the cognitive burden of exploring 3D space to provide click interactions, iDet3D enables users to annotate the entire objects in each scene with minimal interactions. Taking the sparse nature of 3D point clouds into account, we design a negative click simulation (NCS) to improve accuracy by reducing false-positive predictions. In addition, iDet3D incorporates two click propagation techniques to take full advantage of user interactions: (1) dense click guidance (DCG) for keeping user-provided information throughout the network and (2) spatial click propagation (SCP) for detecting other instances of the same class based on the user-specified objects. Through our extensive experiments, we present that our method can construct precise annotations in a few clicks, which shows the practicality as an efficient annotation tool for 3D object detection.

----

## [149] Fusion-Vital: Video-RF Fusion Transformer for Advanced Remote Physiological Measurement

**Authors**: *Jae-Ho Choi, Ki-Bong Kang, Kyung-Tae Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27898](https://doi.org/10.1609/aaai.v38i2.27898)

**Abstract**:

Remote physiology, which involves monitoring vital signs without the need for physical contact, has great potential for various applications. Current remote physiology methods rely only on a single camera or radio frequency (RF) sensor to capture the microscopic signatures from vital movements. However, our study shows that fusing deep RGB and RF features from both sensor streams can further improve performance. Because these multimodal features are defined in distinct dimensions and have varying contextual importance, the main challenge in the fusion process lies in the effective alignment of them and adaptive integration of features under dynamic scenarios. To address this challenge, we propose a novel vital sensing model, named Fusion-Vital, that combines the RGB and RF modalities through the new introduction of pairwise input formats and transformer-based fusion strategies. We also perform comprehensive experiments based on a newly collected and released remote vital dataset comprising synchronized video-RF sensors, showing the superiority of the fusion approach over the previous single-sensor baselines in various aspects.

----

## [150] MeDM: Mediating Image Diffusion Models for Video-to-Video Translation with Temporal Correspondence Guidance

**Authors**: *Ernie Chu, Tzuhsuan Huang, Shuo-Yen Lin, Jun-Cheng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27899](https://doi.org/10.1609/aaai.v38i2.27899)

**Abstract**:

This study introduces an efficient and effective method, MeDM, that utilizes pre-trained image Diffusion Models for video-to-video translation with consistent temporal flow. The proposed framework can render videos from scene position information, such as a normal G-buffer, or perform text-guided editing on videos captured in real-world scenarios. We employ explicit optical flows to construct a practical coding that enforces physical constraints on generated frames and mediates independent frame-wise scores. By leveraging this coding, maintaining temporal consistency in the generated videos can be framed as an optimization problem with a closed-form solution. To ensure compatibility with Stable Diffusion, we also suggest a workaround for modifying observation-space scores in latent Diffusion Models. Notably, MeDM does not require fine-tuning or test-time optimization of the Diffusion Models. Through extensive qualitative, quantitative, and subjective experiments on various benchmarks, the study demonstrates the effectiveness and superiority of the proposed approach. Our project page can be found at https://medm2023.github.io

----

## [151] Attack Deterministic Conditional Image Generative Models for Diverse and Controllable Generation

**Authors**: *Tianyi Chu, Wei Xing, Jiafu Chen, Zhizhong Wang, Jiakai Sun, Lei Zhao, Haibo Chen, Huaizhong Lin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27900](https://doi.org/10.1609/aaai.v38i2.27900)

**Abstract**:

Existing generative adversarial network (GAN) based conditional image generative models typically produce fixed output for the same conditional input, which is unreasonable for highly subjective tasks, such as large-mask image inpainting or style transfer. On the other hand, GAN-based diverse image generative methods require retraining/fine-tuning the network or designing complex noise injection functions, which is computationally expensive, task-specific, or struggle to generate high-quality results. Given that many deterministic conditional image generative models have been able to produce high-quality yet fixed results, we raise an intriguing question: is it possible for pre-trained deterministic conditional image generative models to generate diverse results without changing network structures or parameters? To answer this question, we re-examine the conditional image generation tasks from the perspective of adversarial attack and propose a simple and efficient plug-in projected gradient descent (PGD) like method for diverse and controllable image generation. The key idea is attacking the pre-trained deterministic generative models by adding a micro perturbation to the input condition. In this way, diverse results can be generated without any adjustment of network structures or fine-tuning of the pre-trained models. In addition, we can also control the diverse results to be generated by specifying the attack direction according to a reference text or image. Our work opens the door to applying adversarial attack to low-level vision tasks, and experiments on various conditional image generation tasks demonstrate the effectiveness and superiority of the proposed method.

----

## [152] NILUT: Conditional Neural Implicit 3D Lookup Tables for Image Enhancement

**Authors**: *Marcos V. Conde, Javier Vazquez-Corral, Michael S. Brown, Radu Timofte*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27901](https://doi.org/10.1609/aaai.v38i2.27901)

**Abstract**:

3D lookup tables (3D LUTs) are a key component for image enhancement. Modern image signal processors (ISPs) have dedicated support for these as part of the camera rendering pipeline. Cameras typically provide multiple options for picture styles, where each style is usually obtained by applying a unique handcrafted 3D LUT. Current approaches for learning and applying 3D LUTs are notably fast, yet not so memory-efficient, as storing multiple 3D LUTs is required. For this reason and other implementation limitations, their use on mobile devices is less popular. In this work, we propose a Neural Implicit LUT (NILUT), an implicitly defined continuous 3D color transformation parameterized by a neural network. We show that NILUTs are capable of accurately emulating real 3D LUTs. Moreover, a NILUT can be extended to incorporate multiple styles into a single network with the ability to blend styles implicitly. Our novel approach is memory-efficient, controllable and can complement previous methods, including learned ISPs. Code at https://github.com/mv-lab/nilut

----

## [153] Decoupled Optimisation for Long-Tailed Visual Recognition

**Authors**: *Cong Cong, Shiyu Xuan, Sidong Liu, Shiliang Zhang, Maurice Pagnucco, Yang Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27902](https://doi.org/10.1609/aaai.v38i2.27902)

**Abstract**:

When training on a long-tailed dataset, conventional learning algorithms tend to exhibit a bias towards classes with a larger sample size. Our investigation has revealed that this biased learning tendency originates from the model parameters, which are trained to disproportionately contribute to the classes characterised by their sample size (e.g., many, medium, and few classes).
To balance the overall parameter contribution across all classes, we investigate the importance of each model parameter to the learning of different class groups, and propose a multistage parameter Decouple and Optimisation (DO) framework that decouples parameters into different groups with each group learning a specific portion of classes. To optimise the parameter learning, we apply different training objectives with a collaborative optimisation step to learn complementary information about each class group. Extensive experiments on long-tailed datasets, including CIFAR100, Places-LT, ImageNet-LT, and iNaturaList 2018, show that our framework achieves competitive performance compared to the state-of-the-art.

----

## [154] Underwater Organism Color Fine-Tuning via Decomposition and Guidance

**Authors**: *Xiaofeng Cong, Jie Gui, Junming Hou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27903](https://doi.org/10.1609/aaai.v38i2.27903)

**Abstract**:

Due to the wavelength dependent light attenuation and scattering, the color of the underwater organism usually appears distorted. The existing underwater image enhancement methods mainly focus on designing networks capable of generating enhanced underwater organisms with fixed color. Due to the complexity of the underwater environment, ground truth labels are difficult to obtain, which results in the non-existence of perfect enhancement effects. Different from the existing methods, this paper proposes an algorithm with color enhancement and color fine-tuning (CECF) capabilities. The color enhancement behavior of CECF is the same as that of existing methods, aiming to restore the color of the distorted underwater organism. Beyond this general purpose, the color fine-tuning behavior of CECF can adjust the color of organisms in a controlled manner, which can generate enhanced organisms with diverse colors. To achieve this purpose, four processes are used in CECF. A supervised enhancement process learns the mapping from a distorted image to an enhanced image by the decomposition of color code. A self reconstruction process and a cross-reconstruction process are used for content-invariant learning. A color fine-tuning process is designed based on the guidance for obtaining various enhanced results with different colors. Experimental results have proven the enhancement ability and color fine-tuning ability of the proposed CECF. The source code is provided in https://github.com/Xiaofeng-life/CECF.

----

## [155] Color Event Enhanced Single-Exposure HDR Imaging

**Authors**: *Mengyao Cui, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27904](https://doi.org/10.1609/aaai.v38i2.27904)

**Abstract**:

Single-exposure high dynamic range (HDR) imaging aims
to reconstruct the wide-range intensities of a scene by using
its single low dynamic range (LDR) image, thus providing
significant efficiency. Existing methods pay high attention to
restoring the luminance by inversing the tone-mapping process,
while the color in the over-/under-exposed area cannot
be well restored due to the information loss of the single
LDR image. To address this issue, we introduce color
events into the imaging pipeline, which record asynchronous
pixel-wise color changes in a high dynamic range, enabling
edge-like scene perception under challenging lighting conditions.
Specifically, we propose a joint framework that incorporates
color events and a single LDR image to restore
both content and color of an HDR image, where an exposureaware
transformer (EaT) module is designed to propagate the
informative hints, provided by the normal-exposed LDR regions
and the event streams, to the missing areas. In this
module, an exposure-aware mask is estimated to suppress
distractive information and strengthen the restoration of the
over-/under-exposed regions. To our knowledge, we are the
first to use color events to enhance single-exposure HDR
imaging. We also contribute corresponding datasets, consisting
of synthesized datasets and a real-world dataset collected
by a DAVIS346-color camera. The datasets can be found at
https://www.kaggle.com/datasets/mengyaocui/ce-hdr. Extensive
experiments demonstrate the effectiveness of the proposed
method.

----

## [156] PHFormer: Multi-Fragment Assembly Using Proxy-Level Hybrid Transformer

**Authors**: *Wenting Cui, Runzhao Yao, Shaoyi Du*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27905](https://doi.org/10.1609/aaai.v38i2.27905)

**Abstract**:

Fragment assembly involves restoring broken objects to their original geometries, and has many applications, such as archaeological restoration. Existing learning based frameworks have shown potential for solving part assembly problems with semantic decomposition, but cannot handle such geometrical decomposition problems. In this work, we propose a novel assembly framework, proxy level hybrid Transformer, with the core idea of using a hybrid graph to model and reason complex structural relationships between patches of fragments, dubbed as proxies. To this end, we propose a hybrid attention module, composed of intra and inter attention layers, enabling capturing of crucial contextual information within fragments and relative structural knowledge across fragments. Furthermore, we propose an adjacency aware hierarchical pose estimator, exploiting a decompose and integrate strategy. It progressively predicts adjacent probability and relative poses between fragments, and then implicitly infers their absolute poses by dynamic information integration. Extensive experimental results demonstrate that our method effectively reduces assembly errors while maintaining fast inference speed. The code is available at https://github.com/521piglet/PHFormer.

----

## [157] Trash to Treasure: Low-Light Object Detection via Decomposition-and-Aggregation

**Authors**: *Xiaohan Cui, Long Ma, Tengyu Ma, Jinyuan Liu, Xin Fan, Risheng Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27906](https://doi.org/10.1609/aaai.v38i2.27906)

**Abstract**:

Object detection in low-light scenarios has attracted much attention in the past few years. A mainstream and representative scheme introduces enhancers as the pre-processing for regular detectors. However, because of the disparity in task objectives between the enhancer and detector, this paradigm cannot shine at its best ability. In this work, we try to arouse the potential of enhancer + detector. Different from existing works, we extend the illumination-based enhancers (our newly designed or existing) as a scene decomposition module, whose removed illumination is exploited as the auxiliary in the detector for extracting detection-friendly features. A semantic aggregation module is further established for integrating multi-scale scene-related semantic information in the context space. Actually, our built scheme successfully transforms the "trash" (i.e., the ignored illumination in the detector) into the "treasure" for the detector. Plenty of experiments are conducted to reveal our superiority against other state-of-the-art methods. The code will be public if it is accepted.

----

## [158] Omni-Kernel Network for Image Restoration

**Authors**: *Yuning Cui, Wenqi Ren, Alois Knoll*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27907](https://doi.org/10.1609/aaai.v38i2.27907)

**Abstract**:

Image restoration aims to reconstruct a high-quality image from a degraded low-quality observation. Recently, Transformer models have achieved promising performance on image restoration tasks due to their powerful ability to model long-range dependencies. However, the quadratically growing complexity with respect to the input size makes them inapplicable to practical applications. In this paper, we develop an efficient convolutional network for image restoration by enhancing multi-scale representation learning. To this end, we propose an omni-kernel module that consists of three branches, i.e., global, large, and local branches, to learn global-to-local feature representations efficiently. Specifically, the global branch achieves a global perceptive field via the dual-domain channel attention and frequency-gated mechanism. Furthermore, to provide multi-grained receptive fields, the large branch is formulated via different shapes of depth-wise convolutions with unusually large kernel sizes. Moreover, we complement local information using a point-wise depth-wise convolution. Finally, the proposed network, dubbed OKNet, is established by inserting the omni-kernel module into the bottleneck position for efficiency. Extensive experiments demonstrate that our network achieves state-of-the-art performance on 11 benchmark datasets for three representative image restoration tasks, including image dehazing, image desnowing, and image defocus deblurring. The code is available at https://github.com/c-yn/OKNet.

----

## [159] Aleth-NeRF: Illumination Adaptive NeRF with Concealing Field Assumption

**Authors**: *Ziteng Cui, Lin Gu, Xiao Sun, Xianzheng Ma, Yu Qiao, Tatsuya Harada*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27908](https://doi.org/10.1609/aaai.v38i2.27908)

**Abstract**:

The standard Neural Radiance Fields (NeRF) paradigm employs a viewer-centered methodology, entangling the aspects of illumination and material reflectance into emission solely from 3D points. This simplified rendering approach presents challenges in accurately modeling images captured under adverse lighting conditions, such as low light or over-exposure. Motivated by the ancient Greek emission theory that posits visual perception as a result of rays emanating from the eyes, we slightly refine the conventional NeRF framework to train NeRF under challenging light conditions and generate normal-light condition novel views unsupervisedly. We introduce the concept of a ``Concealing Field," which assigns transmittance values to the surrounding air to account for illumination effects. In dark scenarios, we assume that object emissions maintain a standard lighting level but are attenuated as they traverse the air during the rendering process. Concealing Field thus compel NeRF to learn reasonable density and colour estimations for objects even in dimly lit situations. Similarly, the Concealing Field can mitigate over-exposed emissions during rendering stage. Furthermore, we present a comprehensive multi-view dataset captured under challenging illumination conditions for evaluation. Our code and proposed dataset are available at https://github.com/cuiziteng/Aleth-NeRF.

----

## [160] Federated Modality-Specific Encoders and Multimodal Anchors for Personalized Brain Tumor Segmentation

**Authors**: *Qian Dai, Dong Wei, Hong Liu, Jinghan Sun, Liansheng Wang, Yefeng Zheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27909](https://doi.org/10.1609/aaai.v38i2.27909)

**Abstract**:

Most existing federated learning (FL) methods for medical image analysis only considered intramodal heterogeneity, limiting their applicability to multimodal imaging applications. In practice, it is not uncommon that some FL participants only possess a subset of the complete imaging modalities, posing inter-modal heterogeneity as a challenge to effectively training a global model on all participants’ data. In addition, each participant would expect to obtain a personalized model tailored for its local data characteristics from the FL in such a scenario. In this work, we propose a new FL framework with federated modality-specific encoders and multimodal anchors (FedMEMA) to simultaneously address the two concurrent issues. Above all, FedMEMA employs an exclusive encoder for each modality to account for the inter-modal heterogeneity in the first place. In the meantime, while the encoders are shared by the participants, the decoders are personalized to meet individual needs. Specifically, a server with full-modal data employs a fusion decoder to aggregate and fuse representations from all modality-specific encoders, thus bridging the modalities to optimize the encoders via backpropagation reversely. Meanwhile, multiple anchors are extracted from the fused multimodal representations and distributed to the clients in addition to the encoder parameters. On the other end, the clients with incomplete modalities calibrate their missing-modal representations toward the global full-modal anchors via scaled dot-product cross-attention, making up the information loss due to absent modalities while adapting the representations of present ones. FedMEMA is validated on the BraTS 2020 benchmark for multimodal brain tumor segmentation. Results show that it outperforms various up-to-date methods for multimodal and personalized FL and that its novel designs are effective. Our code is available.

----

## [161] Generating and Reweighting Dense Contrastive Patterns for Unsupervised Anomaly Detection

**Authors**: *Songmin Dai, Yifan Wu, Xiaoqiang Li, Xiangyang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27910](https://doi.org/10.1609/aaai.v38i2.27910)

**Abstract**:

Recent unsupervised anomaly detection methods often rely on feature extractors pretrained with auxiliary datasets or on well-crafted anomaly-simulated samples. However, this might limit their adaptability to an increasing set of anomaly detection tasks due to the priors in the selection of auxiliary datasets or the strategy of anomaly simulation. To tackle this challenge, we first introduce a prior-less anomaly generation paradigm and subsequently develop an innovative unsupervised anomaly detection framework named GRAD, grounded in this paradigm. GRAD comprises three essential components: (1) a diffusion model (PatchDiff) to generate contrastive patterns by preserving the local structures while disregarding the global structures present in normal images, (2) a self-supervised reweighting mechanism to handle the challenge of long-tailed and unlabeled contrastive patterns generated by PatchDiff, and (3) a lightweight patch-level detector to efficiently distinguish the normal patterns and reweighted contrastive patterns. The generation results of PatchDiff effectively expose various types of anomaly patterns, e.g. structural and logical anomaly patterns. In addition, extensive experiments on both MVTec AD and MVTec LOCO datasets also support the aforementioned observation and demonstrate that GRAD achieves competitive anomaly detection accuracy and superior inference speed.

----

## [162] Noisy Correspondence Learning with Self-Reinforcing Errors Mitigation

**Authors**: *Zhuohang Dang, Minnan Luo, Chengyou Jia, Guang Dai, Xiaojun Chang, Jingdong Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27911](https://doi.org/10.1609/aaai.v38i2.27911)

**Abstract**:

Cross-modal retrieval relies on well-matched large-scale datasets that are laborious in practice. Recently, to alleviate expensive data collection, co-occurring pairs from the Internet are automatically harvested for training.
However, it inevitably includes mismatched pairs, i.e., noisy correspondences, undermining supervision reliability and degrading performance. Current methods leverage deep neural networks' memorization effect to address noisy correspondences, which overconfidently focus on similarity-guided training with hard negatives and suffer from self-reinforcing errors. In light of above, we introduce a novel noisy correspondence learning framework, namely Self-Reinforcing Errors Mitigation (SREM).
Specifically, by viewing sample matching as classification tasks within the batch, we generate classification logits for the given sample. Instead of a single similarity score, we refine sample filtration through energy uncertainty and estimate model's sensitivity of selected clean samples using swapped classification entropy, in view of the overall prediction distribution. Additionally, we propose cross-modal biased complementary learning to leverage negative matches overlooked in hard-negative training, further improving model optimization stability and curbing self-reinforcing errors. Extensive experiments on challenging benchmarks affirm the efficacy and efficiency of SREM.

----

## [163] LDMVFI: Video Frame Interpolation with Latent Diffusion Models

**Authors**: *Duolikun Danier, Fan Zhang, David R. Bull*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27912](https://doi.org/10.1609/aaai.v38i2.27912)

**Abstract**:

Existing works on video frame interpolation (VFI) mostly employ deep neural networks that are trained by minimizing the L1, L2, or deep feature space distance (e.g. VGG loss) between their outputs and ground-truth frames. However, recent works have shown that these metrics are poor indicators of perceptual VFI quality. Towards developing perceptually-oriented VFI methods, in this work we propose latent diffusion model-based VFI, LDMVFI. This approaches the VFI problem from a generative perspective by formulating it as a conditional generation problem. As the first effort to address VFI using latent diffusion models, we rigorously benchmark our method on common test sets used in the existing VFI literature. Our quantitative experiments and user study indicate that LDMVFI is able to interpolate video content with favorable perceptual quality compared to the state of the art, even in the high-resolution regime. Our code is available at https://github.com/danier97/LDMVFI.

----

## [164] No More Shortcuts: Realizing the Potential of Temporal Self-Supervision

**Authors**: *Ishan Rajendrakumar Dave, Simon Jenni, Mubarak Shah*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27913](https://doi.org/10.1609/aaai.v38i2.27913)

**Abstract**:

Self-supervised approaches for video have shown impressive results in video understanding tasks. However, unlike early works that leverage temporal self-supervision, current state-of-the-art methods primarily rely on tasks from the image domain (e.g., contrastive learning) that do not explicitly promote the learning of temporal features. We identify two factors that limit existing temporal self-supervision: 1) tasks are too simple, resulting in saturated training performance, and 2) we uncover shortcuts based on local appearance statistics that hinder the learning of high-level features. To address these issues, we propose 1) a more challenging reformulation of temporal self-supervision as frame-level (rather than clip-level) recognition tasks and 2) an effective augmentation strategy to mitigate shortcuts. Our model extends a representation of single video frames, pre-trained through contrastive learning, with a transformer that we train through temporal self-supervision. We demonstrate experimentally that our more challenging frame-level task formulations and the removal of shortcuts drastically improve the quality of features learned through temporal self-supervision. Our extensive experiments show state-of-the-art performance across 10 video understanding datasets, illustrating the generalization ability and robustness of our learned video representations. Project Page: https://daveishan.github.io/nms-webpage.

----

## [165] A Dynamic GCN with Cross-Representation Distillation for Event-Based Learning

**Authors**: *Yongjian Deng, Hao Chen, Youfu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27914](https://doi.org/10.1609/aaai.v38i2.27914)

**Abstract**:

Recent advances in event-based research prioritize sparsity and temporal precision. Approaches learning sparse point-based representations through graph CNNs (GCN) become more popular. Yet, these graph techniques hold lower performance than their frame-based counterpart due to two issues: (i) Biased graph structures that don't properly incorporate varied attributes (such as semantics, and spatial and temporal signals) for each vertex, resulting in inaccurate graph representations. (ii) A shortage of robust pretrained models. Here we solve the first problem by proposing a new event-based GCN (EDGCN), with a dynamic aggregation module to integrate all attributes of vertices adaptively. To address the second problem, we introduce a novel learning framework called cross-representation distillation (CRD), which leverages the dense representation of events as a cross-representation auxiliary to provide additional supervision and prior knowledge for the event graph. This frame-to-graph distillation allows us to benefit from the large-scale priors provided by CNNs while still retaining the advantages of graph-based models. Extensive experiments show our model and learning framework are effective and generalize well across multiple vision tasks.

----

## [166] ResMatch: Residual Attention Learning for Feature Matching

**Authors**: *Yuxin Deng, Kaining Zhang, Shihua Zhang, Yansheng Li, Jiayi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27915](https://doi.org/10.1609/aaai.v38i2.27915)

**Abstract**:

Attention-based graph neural networks have made great progress in feature matching. However, the literature lacks a comprehensive understanding of how the attention mechanism operates for feature matching. In this paper, we rethink cross- and self-attention from the viewpoint of traditional feature matching and filtering. To facilitate the learning of matching and filtering, we incorporate the similarity of descriptors into cross-attention and relative positions into self-attention. In this way, the attention can concentrate on learning residual matching and filtering functions with reference to the basic functions of measuring visual and spatial correlation. Moreover, we leverage descriptor similarity and relative positions to extract inter- and intra-neighbors. Then sparse attention for each point can be performed only within its neighborhoods to acquire higher computation efficiency. Extensive experiments, including feature matching, pose estimation and visual localization, confirm the superiority of the proposed method. Our codes are available at https://github.com/ACuOoOoO/ResMatch.

----

## [167] SDGMNet: Statistic-Based Dynamic Gradient Modulation for Local Descriptor Learning

**Authors**: *Yuxin Deng, Jiayi Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27916](https://doi.org/10.1609/aaai.v38i2.27916)

**Abstract**:

Rescaling the backpropagated gradient of contrastive loss has made significant progress in descriptor learning. However, current gradient modulation strategies have no regard for the varying distribution of global gradients, so they would suffer from changes in training phases or datasets. In this paper, we propose a dynamic gradient modulation, named SDGMNet, for contrastive local descriptor learning. The core of our method is formulating modulation functions with dynamically estimated statistical characteristics. Firstly, we introduce angle for distance measure after deep analysis on backpropagation of pair-wise loss. On this basis, auto-focus modulation is employed to moderate the impact of statistically uncommon individual pairs in stochastic gradient descent optimization; probabilistic margin cuts off the gradients of proportional triplets that have achieved enough optimization; power adjustment balances the total weights of negative pairs and positive pairs. Extensive experiments demonstrate that our novel descriptor surpasses previous state-of-the-art methods in several tasks including patch verification, retrieval, pose estimation, and 3D reconstruction.

----

## [168] Stereo Vision Conversion from Planar Videos Based on Temporal Multiplane Images

**Authors**: *Shanding Diao, Yuan Chen, Yang Zhao, Wei Jia, Zhao Zhang, Ronggang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27917](https://doi.org/10.1609/aaai.v38i2.27917)

**Abstract**:

With the rapid development of 3D movie and light-field displays, there is a growing demand for stereo videos. However, generating high-quality stereo videos from planar videos remains a challenging task. Traditional depth-image-based rendering techniques struggle to effectively handle the problem of occlusion exposure, which occurs when the occluded contents  become visible in other views. Recently, the single-view multiplane images (MPI) representation has shown promising performance for planar video stereoscopy. However, the MPI still lacks real details that are occluded in the current frame, resulting in blurry artifacts in occlusion exposure regions. In fact, planar videos can leverage complementary information from adjacent frames to predict a more complete scene representation for the current frame. Therefore, this paper extends the MPI from still frames to the temporal domain, introducing the temporal MPI (TMPI). By extracting complementary information from adjacent frames based on optical flow guidance, obscured regions in the current frame can be effectively repaired. Additionally, a new module called masked optical flow warping (MOFW) is introduced to improve the propagation of pixels along optical flow trajectories. Experimental results demonstrate that the proposed method can generate high-quality stereoscopic or light-field videos from a single view and reproduce better occluded details than other state-of-the-art (SOTA) methods. https://github.com/Dio3ding/TMPI

----

## [169] Weak Distribution Detectors Lead to Stronger Generalizability of Vision-Language Prompt Tuning

**Authors**: *Kun Ding, Haojian Zhang, Qiang Yu, Ying Wang, Shiming Xiang, Chunhong Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27918](https://doi.org/10.1609/aaai.v38i2.27918)

**Abstract**:

We propose a generalized method for boosting the generalization ability of pre-trained vision-language models (VLMs) while fine-tuning on downstream few-shot tasks. The idea is realized by exploiting out-of-distribution (OOD) detection to predict whether a sample belongs to a base distribution or a novel distribution and then using the score generated by a dedicated competition based scoring function to fuse the zero-shot and few-shot classifier. The fused classifier is dynamic, which will bias towards the zero-shot classifier if a sample is more likely from the distribution pre-trained on, leading to improved base-to-novel generalization ability. Our method is performed only in test stage, which is applicable to boost existing methods without time-consuming re-training. Extensive experiments show that even weak distribution detectors can still improve VLMs' generalization ability. Specifically, with the help of OOD detectors, the harmonic mean of CoOp and ProGrad increase by 2.6 and 1.5 percentage points over 11 recognition datasets in the base-to-novel setting.

----

## [170] Expressive Forecasting of 3D Whole-Body Human Motions

**Authors**: *Pengxiang Ding, Qiongjie Cui, Haofan Wang, Min Zhang, Mengyuan Liu, Donglin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27919](https://doi.org/10.1609/aaai.v38i2.27919)

**Abstract**:

Human motion forecasting, with the goal of estimating future human behavior over a period of time, is a fundamental task in many real-world applications. However, existing works typically concentrate on foretelling the major joints of the human body without considering the delicate movements of the human hands.
In practical applications, hand gesture plays an important role in human communication with the real world, and expresses the primary intention of human beings. In this work, we are the first to formulate whole-body human pose forecasting task, which jointly predicts future both body and gesture activities. Correspondingly, we propose a novel Encoding-Alignment-Interaction (EAI) framework that aims to predict both coarse (body joints) and fine-grained (gestures) activities collaboratively, enabling expressive and cross-facilitated forecasting of 3D whole-body human motions. Specifically, our model involves two key constituents: cross-context alignment (XCA) and cross-context interaction (XCI). Considering the heterogeneous information within the whole-body, XCA aims to align the latent features of various human components, while XCI focuses on effectively capturing the context interaction among the human components. We conduct extensive experiments on a newly-introduced large-scale benchmark and achieve state-of-the-art performance. The code is public for research purposes at https://github.com/Dingpx/EAI.

----

## [171] Transferable Adversarial Attacks for Object Detection Using Object-Aware Significant Feature Distortion

**Authors**: *Xinlong Ding, Jiansheng Chen, Hongwei Yu, Yu Shang, Yining Qin, Huimin Ma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27920](https://doi.org/10.1609/aaai.v38i2.27920)

**Abstract**:

Transferable black-box adversarial attacks against classifiers by disturbing the intermediate-layer features have been extensively studied in recent years. However, these methods have not yet achieved satisfactory performances when directly applied to object detectors. This is largely because the features of detectors are fundamentally different from that of the classifiers. In this study, we propose a simple but effective method to improve the transferability of adversarial examples for object detectors by leveraging the properties of spatial consistency and limited equivariance of object detectors’ features. Specifically, we combine a novel loss function and deliberately designed data augmentation to distort the backbone features of object detectors by suppressing significant features corresponding to objects and amplifying the surrounding vicinal features corresponding to object boundaries. As such the target object and background area on the generated adversarial samples are more likely to be confused by other detectors. Extensive experimental results show that our proposed method achieves state-of-the-art black-box transferability for untargeted attacks on various models, including one/two-stage, CNN/Transformer-based, and anchor-free/anchor-based detectors.

----

## [172] Hyp-OW: Exploiting Hierarchical Structure Learning with Hyperbolic Distance Enhances Open World Object Detection

**Authors**: *Thang Doan, Xin Li, Sima Behpour, Wenbin He, Liang Gou, Liu Ren*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27921](https://doi.org/10.1609/aaai.v38i2.27921)

**Abstract**:

Open World Object Detection (OWOD) is a challenging and realistic task that extends beyond the scope of standard Object Detection task. It involves detecting both known and unknown objects while integrating learned knowledge for future tasks. However, the level of "unknownness" varies significantly depending on the context. For example, a tree is typically considered part of the background in a self-driving scene, but it may be significant in a household context. We argue that this contextual information should already be embedded within the known classes. In other words, there should be a semantic or latent structure relationship between the known and unknown items to be discovered. Motivated by this observation, we propose Hyp-OW, a method that learns and models hierarchical representation of known items through a SuperClass Regularizer. Leveraging this representation allows us to effectively detect unknown objects using a similarity distance-based relabeling module. Extensive experiments on benchmark datasets demonstrate the effectiveness of Hyp-OW, achieving improvement in both known and unknown detection (up to 6 percent). These findings are particularly pronounced in our newly designed benchmark, where a strong hierarchical structure exists between known and unknown objects.

----

## [173] Exploiting Polarized Material Cues for Robust Car Detection

**Authors**: *Wen Dong, Haiyang Mei, Ziqi Wei, Ao Jin, Sen Qiu, Qiang Zhang, Xin Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27922](https://doi.org/10.1609/aaai.v38i2.27922)

**Abstract**:

Car detection is an important task that serves as a crucial prerequisite for many automated driving functions. The large variations in lighting/weather conditions and vehicle densities of the scenes pose significant challenges to existing car detection algorithms to meet the highly accurate perception demand for safety, due to the unstable/limited color information, which impedes the extraction of meaningful/discriminative features of cars. In this work, we present a novel learning-based car detection method that leverages trichromatic linear polarization as an additional cue to disambiguate such challenging cases. A key observation is that polarization, characteristic of the light wave, can robustly describe intrinsic physical properties of the scene objects in various imaging conditions and is strongly linked to the nature of materials for cars (e.g., metal and glass) and their surrounding environment (e.g., soil and trees), thereby providing reliable and discriminative features for robust car detection in challenging scenes. To exploit polarization cues, we first construct a pixel-aligned RGB-Polarization car detection dataset, which we subsequently employ to train a novel multimodal fusion network. Our car detection network dynamically integrates RGB and polarization features in a request-and-complement manner and can explore the intrinsic material properties of cars across all learning samples. We extensively validate our method and demonstrate that it outperforms state-of-the-art detection methods. Experimental results show that polarization is a powerful cue for car detection. Our code is available at https://github.com/wind1117/AAAI24-PCDNet.

----

## [174] Learning Multi-Modal Cross-Scale Deformable Transformer Network for Unregistered Hyperspectral Image Super-resolution

**Authors**: *Wenqian Dong, Yang Xu, Jiahui Qu, Shaoxiong Hou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27923](https://doi.org/10.1609/aaai.v38i2.27923)

**Abstract**:

Hyperspectral image super-resolution (HSI-SR) is a technology to improve the spatial resolution of HSI. Existing fusion-based SR methods have shown great performance, but still have some problems as follows: 1) existing methods assume that the auxiliary image providing spatial information is strictly registered with the HSI, but images are difficult to be registered finely due to the shooting platforms, shooting viewpoints and the influence of atmospheric turbulence; 2) most of the methods are based on convolutional neural networks (CNNs), which is effective for local features but cannot utilize the global features. To this end, we propose a multi-modal cross-scale deformable transformer network (M2DTN) to achieve unregistered HSI-SR. Specifically, we formulate a spectrum-preserving based spatial-guided registration-SR unified model (SSRU) from the  view of the realistic degradation scenarios. According to SSRU, we propose multi-modal registration deformable module (MMRD) to align features between different modalities by deformation field. In order to efficiently utilize the unique information between different modals, we design multi-scale feature transformer (MSFT) to emphasize the spatial-spectral features at different scales. In addition, we propose the cross-scale feature aggregation module (CSFA) to accurately reconstruct the HSI by aggregating feature information at different scales. Experiments show that M2DTN outperforms the-state-of-the-art HSI-SR methods. Code is obtainable at https://github.com/Jiahuiqu/M2DTN.

----

## [175] Joint Demosaicing and Denoising for Spike Camera

**Authors**: *Yanchen Dong, Ruiqin Xiong, Jing Zhao, Jian Zhang, Xiaopeng Fan, Shuyuan Zhu, Tiejun Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27924](https://doi.org/10.1609/aaai.v38i2.27924)

**Abstract**:

As a neuromorphic camera with high temporal resolution, spike camera can capture dynamic scenes with high-speed motion. Recently, spike camera with a color filter array (CFA) has been developed for color imaging. There are some methods for spike camera demosaicing to reconstruct color images from Bayer-pattern spike streams. However, the demosaicing results are bothered by severe noise in spike streams, to which previous works pay less attention. In this paper, we propose an iterative joint demosaicing and denoising network (SJDD-Net) for spike cameras based on the observation model. Firstly, we design a color spike representation (CSR) to learn latent representation from Bayer-pattern spike streams. In CSR, we propose an offset-sharing deformable convolution module to align temporal features of color channels. Then we develop a spike noise estimator (SNE) to obtain features of the noise distribution. Finally, a color correlation prior (CCP) module is proposed to utilize the color correlation for better details. For training and evaluation, we designed a spike camera simulator to generate Bayer-pattern spike streams with synthesized noise. Besides, we captured some Bayer-pattern spike streams, building the first real-world captured dataset to our knowledge. Experimental results show that our method can restore clean images from Bayer-pattern spike streams. The source codes and dataset are available at https://github.com/csycdong/SJDD-Net.

----

## [176] ChromaFusionNet (CFNet): Natural Fusion of Fine-Grained Color Editing

**Authors**: *Yi Dong, Yuxi Wang, Ruoxi Fan, Wenqi Ouyang, Zhiqi Shen, Peiran Ren, Xuansong Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27925](https://doi.org/10.1609/aaai.v38i2.27925)

**Abstract**:

Digital image enhancement aims to deliver visually striking, pleasing images that align with human perception. While global techniques can elevate the image's overall aesthetics, fine-grained color enhancement can further boost visual appeal and expressiveness. However, colorists frequently face challenges in achieving accurate, localized color adjustments. Direct composition of these local edits can result in spatial color inconsistencies. Existing methods, including color style transfer and image harmonization, exhibit inconsistencies, especially at boundary regions. Addressing this, we present ChromaFusionNet (CFNet), a novel approach that views the color fusion problem through the lens of image color inpainting. Built on the Vision Transformer architecture, CFNet captures global context and delivers high-fidelity outputs, seamlessly blending colors while preserving boundary integrity. Empirical studies on ImageNet and COCO datasets demonstrate CFNet's superiority over existing methods in maintaining color harmony and color fidelity. Robustness evaluations and user studies have further validated the effectiveness of CFNet. In conclusion, CFNet introduces an innovative approach to seamless, fine-grained color fusion, paving the way for advancements in the domain of fine-grained color editing. Code and pretrained models are available at our project page: https://yidong.pro/projects/cfnet.

----

## [177] HybridGait: A Benchmark for Spatial-Temporal Cloth-Changing Gait Recognition with Hybrid Explorations

**Authors**: *Yilan Dong, Chunlin Yu, Ruiyang Ha, Ye Shi, Yuexin Ma, Lan Xu, Yanwei Fu, Jingya Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27926](https://doi.org/10.1609/aaai.v38i2.27926)

**Abstract**:

Existing gait recognition benchmarks mostly include minor clothing variations in the laboratory environments, but lack persistent changes in appearance over time and space. In this paper, we propose the first in-the-wild benchmark CCGait for cloth-changing gait recognition, which incorporates diverse clothing changes, indoor and outdoor scenes, and multi-modal statistics over 92 days. To further address the coupling effect of clothing and viewpoint variations, we propose a hybrid approach HybridGait that exploits both temporal dynamics and the projected 2D information of 3D human meshes. Specifically, we introduce a Canonical Alignment Spatial-Temporal Transformer (CA-STT) module to encode human joint position-aware features, and fully exploit 3D dense priors via a Silhouette-guided Deformation with 3D-2D Appearance Projection (SilD) strategy. Our contributions are twofold: we provide a challenging benchmark CCGait that captures realistic appearance changes over expanded time and space, and we propose a hybrid framework HybridGait that outperforms prior works on CCGait and Gait3D benchmarks. Our project page is available at https://github.com/HCVLab/HybridGait.

----

## [178] PPEA-Depth: Progressive Parameter-Efficient Adaptation for Self-Supervised Monocular Depth Estimation

**Authors**: *Yue-Jiang Dong, Yuan-Chen Guo, Ying-Tian Liu, Fang-Lue Zhang, Song-Hai Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27927](https://doi.org/10.1609/aaai.v38i2.27927)

**Abstract**:

Self-supervised monocular depth estimation is of significant importance with applications spanning across autonomous driving and robotics. However, the reliance on self-supervision introduces a strong static-scene assumption, thereby posing challenges in achieving optimal performance in dynamic scenes, which are prevalent in most real-world situations.
To address these issues, we propose PPEA-Depth, a Progressive Parameter-Efficient Adaptation approach to transfer a pre-trained image model for self-supervised depth estimation. The training comprises two sequential stages: an initial phase trained on a dataset primarily composed of static scenes, succeeded by an expansion to more intricate datasets involving dynamic scenes. To facilitate this process, we design compact encoder and decoder adapters to enable parameter-efficient tuning, allowing the network to adapt effectively. They not only uphold generalized patterns from pre-trained image models but also retain knowledge gained from the preceding phase into the subsequent one. Extensive experiments demonstrate that PPEA-Depth achieves state-of-the-art performance on KITTI, CityScapes and DDAD datasets.

----

## [179] CycleVTON: A Cycle Mapping Framework for Parser-Free Virtual Try-On

**Authors**: *Chenghu Du, Junyin Wang, Yi Rong, Shuqing Liu, Kai Liu, Shengwu Xiong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27928](https://doi.org/10.1609/aaai.v38i2.27928)

**Abstract**:

Image-based virtual try-on aims to transfer a target clothing onto a specific person. A significant challenge is arbitrarily matched clothing and person lack corresponding ground truth to supervised learning. A recent pioneering work leveraged an improved cycleGAN to enable one network to generate the desired image for another network during training. However, there is no difference in the result distribution before and after the clothing changes. Therefore, using two different networks is unnecessary and may even increase the difficulty of convergence. Furthermore, the introduced human parsing used to provide body structure information in the input also have a negative impact on the try-on result. How to employ a single network for supervised learning while eliminating human parsing? To tackle these issues, we present a Cycle mapping Virtual Try-On Network (CycleVTON), which can produce photo-realistic try-on results by using a cycle mapping framework without the parser. In particular, we introduce a flow constraint loss to achieve supervised learning of arbitrarily matched clothing and person as inputs to the deformer, thus naturally mimicking the interaction between clothing and the human body. Additionally, we design a skin generation strategy that can adapt to the shape of the target clothing by dynamically adjusting the skin region, i.e., by first removing and then filling skin areas. Extensive experiments conducted on challenging benchmarks demonstrate that our proposed method exhibits superior performance compared to state-of-the-art methods.

----

## [180] Arbitrary-Scale Point Cloud Upsampling by Voxel-Based Network with Latent Geometric-Consistent Learning

**Authors**: *Hang Du, Xuejun Yan, Jingjing Wang, Di Xie, Shiliang Pu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27929](https://doi.org/10.1609/aaai.v38i2.27929)

**Abstract**:

Recently, arbitrary-scale point cloud upsampling mechanism became increasingly popular due to its efficiency and convenience for practical applications. To achieve this, most previous approaches formulate it as a problem of surface approximation and employ point-based networks to learn surface representations. However, learning surfaces from sparse point clouds is more challenging, and thus they often suffer from the low-fidelity geometry approximation. To address it, we propose an arbitrary-scale Point cloud Upsampling framework using Voxel-based Network (PU-VoxelNet). Thanks to the completeness and regularity inherited from the voxel representation, voxel-based networks are capable of providing predefined grid space to approximate 3D surface, and an arbitrary number of points can be reconstructed according to the predicted density distribution within each grid cell. However, we investigate the inaccurate grid sampling caused by imprecise density predictions. To address this issue, a density-guided grid resampling method is developed to generate high-fidelity points while effectively avoiding sampling outliers. Further, to improve the fine-grained details, we present an auxiliary training supervision to enforce the latent geometric consistency among local surface patches. Extensive experiments indicate the proposed approach outperforms the state-of-the-art approaches not only in terms of fixed upsampling rates but also for arbitrary-scale upsampling. The code is available at https://github.com/hikvision-research/3DVision

----

## [181] CDPNet: Cross-Modal Dual Phases Network for Point Cloud Completion

**Authors**: *Zhenjiang Du, Jiale Dou, Zhitao Liu, Jiwei Wei, Guan Wang, Ning Xie, Yang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27930](https://doi.org/10.1609/aaai.v38i2.27930)

**Abstract**:

Point cloud completion aims at completing shapes from their partial. Most existing methods utilized shape’s priors information for point cloud completion, such as inputting the partial and getting the complete one through an encoder-decoder deep learning structure.  However, it is very often to easily cause the loss of information in the generation process because of the invisibility of missing areas. Unlike most existing methods directly inferring the missing points using shape priors, we address it as a cross-modality task. We propose a new Cross-modal Dual Phases Network (CDPNet) for shape completion. Our key idea is that the global information of the shape is obtained from the extra single-view image, and the partial point clouds provide the geometric information. After that, the multi-modal features jointly guide the specific structural information. To learn the geometric details of the shape, we chose to use patches to preserve the local geometric feature. In this way, we can generate shapes with enough geometric details. Experimental results show that our method achieves state-of-the-art performance on point cloud completion.

----

## [182] Tuning-Free Inversion-Enhanced Control for Consistent Image Editing

**Authors**: *Xiaoyue Duan, Shuhao Cui, Guoliang Kang, Baochang Zhang, Zhengcong Fei, Mingyuan Fan, Junshi Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27931](https://doi.org/10.1609/aaai.v38i2.27931)

**Abstract**:

Consistent editing of real images is a challenging task, as it requires performing non-rigid edits (e.g., changing postures) to the main objects in the input image without changing their identity or attributes. To guarantee consistent attributes, some existing methods fine-tune the entire model or the textual embedding for structural consistency, but they are time-consuming and fail to perform non-rigid edits. Other works are tuning-free, but their performances are weakened by the quality of Denoising Diffusion Implicit Model (DDIM) reconstruction, which often fails in real-world scenarios. In this paper, we present a novel approach called Tuning-free Inversion-enhanced Control (TIC), which directly correlates features from the inversion process with those from the sampling process to mitigate the inconsistency in DDIM reconstruction. Specifically, our method effectively obtains inversion features from the key and value features in the self-attention layers, and enhances the sampling process by these inversion features, thus achieving accurate reconstruction and content-consistent editing. To extend the applicability of our method to general editing scenarios, we also propose a mask-guided attention concatenation strategy that combines contents from both the inversion and the naive DDIM editing processes. Experiments show that the proposed method outperforms previous works in reconstruction and consistent editing, and produces impressive results in various settings.

----

## [183] WeditGAN: Few-Shot Image Generation via Latent Space Relocation

**Authors**: *Yuxuan Duan, Li Niu, Yan Hong, Liqing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27932](https://doi.org/10.1609/aaai.v38i2.27932)

**Abstract**:

In few-shot image generation, directly training GAN models on just a handful of images faces the risk of overfitting. A popular solution is to transfer the models pretrained on large source domains to small target ones. In this work, we introduce WeditGAN, which realizes model transfer by editing the intermediate latent codes w in StyleGANs with learned constant offsets (delta w), discovering and constructing target latent spaces via simply relocating the distribution of source latent spaces. The established one-to-one mapping between latent spaces can naturally prevents mode collapse and overfitting. Besides, we also propose variants of WeditGAN to further enhance the relocation process by regularizing the direction or finetuning the intensity of delta w. Experiments on a collection of widely used source/target datasets manifest the capability of WeditGAN in generating realistic and diverse images, which is simple yet highly effective in the research area of few-shot image generation. Codes are available at https://github.com/Ldhlwh/WeditGAN.

----

## [184] SkeletonGait: Gait Recognition Using Skeleton Maps

**Authors**: *Chao Fan, Jingzhe Ma, Dongyang Jin, Chuanfu Shen, Shiqi Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27933](https://doi.org/10.1609/aaai.v38i2.27933)

**Abstract**:

The choice of the representations is essential for deep gait recognition methods. The binary silhouettes and skeletal coordinates are two dominant representations in recent literature, achieving remarkable advances in many scenarios. However, inherent challenges remain, in which silhouettes are not always guaranteed in unconstrained scenes, and structural cues have not been fully utilized from skeletons. In this paper, we introduce a novel skeletal gait representation named skeleton map, together with SkeletonGait, a skeleton-based method to exploit structural information from human skeleton maps. Specifically, the skeleton map represents the coordinates of human joints as a heatmap with Gaussian approximation, exhibiting a silhouette-like image devoid of exact body structure. Beyond achieving state-of-the-art performances over five popular gait datasets, more importantly, SkeletonGait uncovers novel insights about how important structural features are in describing gait and when they play a role.  Furthermore, we propose a multi-branch architecture, named SkeletonGait++, to make use of complementary features from both skeletons and silhouettes. Experiments indicate that SkeletonGait++ outperforms existing state-of-the-art methods by a significant margin in various scenarios. For instance, it achieves an impressive rank-1 accuracy of over 85% on the challenging GREW dataset. The source code is available at https://github.com/ShiqiYu/OpenGait.

----

## [185] TDeLTA: A Light-Weight and Robust Table Detection Method Based on Learning Text Arrangement

**Authors**: *Yang Fan, Xiangping Wu, Qingcai Chen, Heng Li, Yan Huang, Zhixiang Cai, Qitian Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27934](https://doi.org/10.1609/aaai.v38i2.27934)

**Abstract**:

The diversity of tables makes table detection a great challenge, leading to existing models becoming more tedious and complex. Despite achieving high performance, they often overfit to the table style in training set, and suffer from significant performance degradation when encountering out-of-distribution tables in other domains. To tackle this problem, we start from the essence of the table, which is a set of text arranged in rows and columns. Based on this, we propose a novel, light-weighted and robust Table Detection method based on Learning Text Arrangement, namely TDeLTA. TDeLTA takes the text blocks as input, and then models the arrangement of them with  a sequential encoder and an attention module. To locate the tables precisely, we design a text-classification task, classifying the text blocks into 4 categories according to their semantic roles in the tables. Experiments are conducted on both the text blocks parsed from PDF and extracted by open-source OCR tools, respectively. Compared to several state-of-the-art methods, TDeLTA achieves competitive results with only 3.1M model parameters on the large-scale public datasets. Moreover, when faced with the cross-domain data under the 0-shot setting, TDeLTA outperforms baselines by a large margin of nearly 7%, which shows the strong robustness and transferability of the proposed model.

----

## [186] Collaborative Tooth Motion Diffusion Model in Digital Orthodontics

**Authors**: *Yeying Fan, Guangshun Wei, Chen Wang, Shaojie Zhuang, Wenping Wang, Yuanfeng Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27935](https://doi.org/10.1609/aaai.v38i2.27935)

**Abstract**:

Tooth motion generation is an essential task in digital orthodontic treatment for precise and quick dental healthcare, which aims to generate the whole intermediate tooth motion process given the initial pathological and target ideal tooth alignments. 
Most prior works for multi-agent motion planning problems usually result in complex solutions.
Moreover, the occlusal relationship between upper and lower teeth is often overlooked. 
In this paper, we propose a collaborative tooth motion diffusion model. 
The critical insight is to remodel the problem as a diffusion process. 
In this sense, we model the whole tooth motion distribution with a diffusion model and transform the planning problem into a sampling process from this distribution.
We design a tooth latent representation to provide accurate conditional guides consisting of two key components: the tooth frame represents the position and posture, and the tooth latent shape code represents the geometric morphology. 
Subsequently, we present a collaborative diffusion model to learn the multi-tooth motion distribution based on inter-tooth and occlusal constraints, which are implemented by graph structure and new loss functions, respectively. 
Extensive qualitative and quantitative experiments demonstrate the superiority of our framework in the application of orthodontics compared with state-of-the-art methods.

----

## [187] Everything2Motion: Synchronizing Diverse Inputs via a Unified Framework for Human Motion Synthesis

**Authors**: *Zhaoxin Fan, Longbin Ji, Pengxin Xu, Fan Shen, Kai Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27936](https://doi.org/10.1609/aaai.v38i2.27936)

**Abstract**:

In the dynamic field of film and game development, the emergence of human motion synthesis methods has revolutionized avatar animation. Traditional methodologies, typically reliant on single modality inputs like text or audio, employ modality-specific model frameworks, posing challenges for unified model deployment and application. To address this, we propose Everything2Motion, a unified model framework. Everything2Motion consists of three key modules. The Input-Output Modality Modulation module tailors structures for specific multimodal inputs, eliminating the need for modality-specific frameworks. The Query-aware Autoencoder, based on the transformer encoder-decoder architecture, enables efficient latent motion generation. Lastly, the Prior Motion Distillation Decoder, a pretrained module, enhances the final skeleton sequence's naturalness and fluidity. Comprehensive experiments on several public datasets demonstrate the effectiveness of Everything2Motion, highlighting its potential for practical applications and setting a new benchmark in human motion synthesis.

----

## [188] Variance-Insensitive and Target-Preserving Mask Refinement for Interactive Image Segmentation

**Authors**: *Chaowei Fang, Ziyin Zhou, Junye Chen, Hanjing Su, Qingyao Wu, Guanbin Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27937](https://doi.org/10.1609/aaai.v38i2.27937)

**Abstract**:

Point-based interactive image segmentation can ease the burden of mask annotation in applications such as semantic segmentation and image editing. However, fully extracting the target mask with limited user inputs remains challenging. We introduce a novel method, Variance-Insensitive and Target-Preserving Mask Refinement to enhance segmentation quality with fewer user inputs. Regarding the last segmentation result as the initial mask, an iterative refinement process is commonly employed to continually enhance the initial mask. Nevertheless, conventional techniques suffer from sensitivity to the variance in the initial mask. To circumvent this problem, our proposed method incorporates a mask matching algorithm for ensuring consistent inferences from different types of initial masks. We also introduce a target-aware zooming algorithm to preserve object information during downsampling, balancing efficiency and accuracy. Experiments on GrabCut, Berkeley, SBD, and DAVIS datasets demonstrate our method's state-of-the-art performance in interactive image segmentation.

----

## [189] Evaluate Geometry of Radiance Fields with Low-Frequency Color Prior

**Authors**: *Qihang Fang, Yafei Song, Keqiang Li, Li Shen, Huaiyu Wu, Gang Xiong, Liefeng Bo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27938](https://doi.org/10.1609/aaai.v38i2.27938)

**Abstract**:

A radiance field is an effective representation of 3D scenes, which has been widely adopted in novel-view synthesis and 3D reconstruction. It is still an open and challenging problem to evaluate the geometry, i.e., the density field, as the ground-truth is almost impossible to obtain. One alternative indirect solution is to transform the density field into a point-cloud and compute its Chamfer Distance with the scanned ground-truth. However, many widely-used datasets have no point-cloud ground-truth since the scanning process along with the equipment is expensive and complicated.
To this end, we propose a novel metric, named Inverse Mean Residual Color (IMRC), which can evaluate the geometry only with the observation images. Our key insight is that the better the geometry, the lower-frequency the computed color field. From this insight, given a reconstructed density field and observation images, we design a closed-form method to approximate the color field with low-frequency spherical harmonics, and compute the inverse mean residual color. Then the higher the IMRC, the better the geometry. Qualitative and quantitative experimental results verify the effectiveness of our proposed IMRC metric. We also benchmark several state-of-the-art methods using IMRC to promote future related research. Our code is available at https://github.com/qihangGH/IMRC.

----

## [190] Simple Image-Level Classification Improves Open-Vocabulary Object Detection

**Authors**: *Ruohuan Fang, Guansong Pang, Xiao Bai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27939](https://doi.org/10.1609/aaai.v38i2.27939)

**Abstract**:

Open-Vocabulary Object Detection (OVOD) aims to detect novel objects beyond a given set of base categories on which the detection model is trained. Recent OVOD methods focus on adapting the image-level pre-trained vision-language models (VLMs), such as CLIP, to a region-level object detection task via, eg., region-level knowledge distillation, regional prompt learning, or region-text pre-training, to expand the detection vocabulary. These methods have demonstrated remarkable performance in recognizing regional visual concepts, but they are weak in exploiting the VLMs' powerful global scene understanding ability learned from the billion-scale image-level text descriptions. This limits their capability in detecting hard objects of small, blurred, or occluded appearance from novel/base categories, whose detection heavily relies on contextual information. To address this, we propose a novel approach, namely Simple Image-level Classification for Context-Aware Detection Scoring (SIC-CADS), to leverage the superior global knowledge yielded from CLIP for complementing the current OVOD models from a global perspective. The core of SIC-CADS is a multi-modal multi-label recognition (MLR) module that learns the object co-occurrence-based contextual information from CLIP to recognize all possible object categories in the scene. These image-level MLR scores can then be utilized to refine the instance-level detection scores of the current OVOD models in detecting those hard objects. This is verified by extensive empirical results on two popular benchmarks, OV-LVIS and OV-COCO, which show that SIC-CADS achieves significant and consistent improvement when combined with different types of OVOD models. Further, SIC-CADS also improves the cross-dataset generalization ability on Objects365 and OpenImages. Code is available at https://github.com/mala-lab/SIC-CADS.

----

## [191] Self-Supervised Bird's Eye View Motion Prediction with Cross-Modality Signals

**Authors**: *Shaoheng Fang, Zuhong Liu, Mingyu Wang, Chenxin Xu, Yiqi Zhong, Siheng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27940](https://doi.org/10.1609/aaai.v38i2.27940)

**Abstract**:

Learning the dense bird's eye view (BEV) motion flow in a self-supervised manner is an emerging research for robotics and autonomous driving. Current self-supervised methods mainly rely on point correspondences between point clouds, which may introduce the problems of fake flow and inconsistency, hindering the model’s ability to learn accurate and realistic motion. In this paper, we introduce a novel cross-modality self-supervised training framework that effectively addresses these issues by leveraging multi-modality data to obtain supervision signals. We design three innovative supervision signals to preserve the inherent properties of scene motion, including the masked Chamfer distance loss, the piecewise rigidity loss, and the temporal consistency loss. Through extensive experiments, we demonstrate that our proposed self-supervised framework outperforms all previous self-supervision methods for the motion prediction task.

----

## [192] Fewer Steps, Better Performance: Efficient Cross-Modal Clip Trimming for Video Moment Retrieval Using Language

**Authors**: *Xiang Fang, Daizong Liu, Wanlong Fang, Pan Zhou, Zichuan Xu, Wenzheng Xu, Junyang Chen, Renfu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27941](https://doi.org/10.1609/aaai.v38i2.27941)

**Abstract**:

Given an untrimmed video and a sentence query, video moment retrieval using language (VMR) aims to locate a target query-relevant moment. Since the untrimmed video is overlong, almost all existing VMR methods first sparsely down-sample each untrimmed video into multiple fixed-length video clips and then conduct multi-modal interactions with the query feature and expensive clip features for reasoning, which is infeasible for long real-world videos that span hours. Since the video is downsampled into  fixed-length clips,  some query-related frames may be filtered out, which will blur the specific boundary of the target moment, take the adjacent irrelevant frames as new boundaries, easily leading to cross-modal misalignment and introducing both boundary-bias and reasoning-bias. To this end, in this paper, we propose an efficient approach, SpotVMR, to trim the query-relevant clip. Besides, our proposed SpotVMR can serve as plug-and-play module, which achieves efficiency for state-of-the-art VMR methods while maintaining good retrieval performance. Especially, we first design a novel clip search model that learns to identify promising video regions to search conditioned on the language query. Then, we introduce a  set of low-cost semantic indexing features to capture the context of objects and interactions that suggest where to search the query-relevant moment. Also, the distillation loss is utilized to  address the optimization issues arising from end-to-end joint training of the clip selector and VMR model.
Extensive experiments on three challenging datasets demonstrate its effectiveness.

----

## [193] An Embedding-Unleashing Video Polyp Segmentation Framework via Region Linking and Scale Alignment

**Authors**: *Zhixue Fang, Xinrong Guo, Jingyin Lin, Huisi Wu, Jing Qin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27942](https://doi.org/10.1609/aaai.v38i2.27942)

**Abstract**:

Automatic polyp segmentation from colonoscopy videos is a critical task for the development of computer-aided screening and diagnosis systems. However, accurate and real-time video polyp segmentation (VPS) is a very challenging task due to low contrast between background and polyps and frame-to-frame dramatic variations in colonoscopy videos. We propose a novel embedding-unleashing framework consisting of a proposal-generative network (PGN) and an appearance-embedding network (AEN) to comprehensively address these challenges. Our framework, for the first time, models VPS as an appearance-level semantic embedding process to facilitate generate more global information to counteract background disturbances and dramatic variations. Specifically, PGN is a video segmentation network to obtain segmentation mask proposals, while AEN is a network we specially designed to produce appearance-level embedding semantics for PGN, thereby unleashing the capability of PGN in VPS. Our AEN consists of a cross-scale region linking (CRL) module and a cross-wise scale alignment (CSA) module. The former screens reliable background information against background disturbances by constructing linking of region semantics, while the latter performs the scale alignment to resist dramatic variations by modeling the center-perceived motion dependence with a cross-wise manner. We further introduce a parameter-free semantic interaction to embed the semantics of AEN into PGN to obtain the segmentation results. Extensive experiments on CVC-612 and SUN-SEG demonstrate that our approach achieves better performance than other state-of-the-art methods. Codes are available at https://github.com/zhixue-fang/EUVPS.

----

## [194] Debiased Novel Category Discovering and Localization

**Authors**: *Juexiao Feng, Yuhong Yang, Yanchun Xie, Yaqian Li, Yandong Guo, Yuchen Guo, Yuwei He, Liuyu Xiang, Guiguang Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27943](https://doi.org/10.1609/aaai.v38i2.27943)

**Abstract**:

In recent years, object detection in deep learning has experienced rapid development. However, most existing object detection models perform well only on closed-set datasets, ignoring a large number of potential objects whose categories are not defined in the training set. These objects are often identified as background or incorrectly classified as pre-defined categories by the detectors. In this paper, we focus on the challenging problem of Novel Class Discovery and Localization (NCDL), aiming to train detectors that can detect the categories present in the training data, while also actively discover, localize, and cluster new categories. We analyze existing NCDL methods and identify the core issue: object detectors tend to be biased towards seen objects, and this leads to the neglect of unseen targets. To address this issue, we first propose an Debiased Region Mining (DRM) approach that combines class-agnostic Region Proposal Network (RPN) and class-aware RPN in a complementary manner. Additionally, we suggest to improve the representation network through semi-supervised contrastive learning by leveraging unlabeled data. Finally, we adopt a simple and efficient mini-batch K-means clustering method for novel class discovery. We conduct extensive experiments on the NCDL benchmark, and the results demonstrate that the proposed DRM approach significantly outperforms previous methods, establishing a new state-of-the-art.

----

## [195] Interpretable3D: An Ad-Hoc Interpretable Classifier for 3D Point Clouds

**Authors**: *Tuo Feng, Ruijie Quan, Xiaohan Wang, Wenguan Wang, Yi Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27944](https://doi.org/10.1609/aaai.v38i2.27944)

**Abstract**:

3D decision-critical tasks urgently require research on explanations to ensure system reliability and transparency. Extensive explanatory research has been conducted on 2D images, but there is a lack in the 3D field. Furthermore, the existing explanations for 3D models are post-hoc and can be misleading, as they separate explanations from the original model. To address these issues, we propose an ad-hoc interpretable classifier for 3D point clouds (i.e., Interpretable3D). As an intuitive case-based classifier, Interpretable3D can provide reliable ad-hoc explanations without any embarrassing nuances. It allows users to understand how queries are embedded within past observations in prototype sets. Interpretable3D has two iterative training steps: 1) updating one prototype with the mean of the embeddings within the same sub-class in Prototype Estimation, and 2) penalizing or rewarding the estimated prototypes in Prototype Optimization. The mean of embeddings has a clear statistical meaning, i.e., class sub-centers. Moreover, we update prototypes with their most similar observations in the last few epochs. Finally, Interpretable3D classifies new samples according to prototypes. We evaluate the performance of Interpretable3D on four popular point cloud models: DGCNN, PointNet2, PointMLP, and PointNeXt. Our Interpretable3D demonstrates comparable or superior performance compared to softmax-based black-box models in the tasks of 3D shape classification and part segmentation. Our code is released at: github.com/FengZicai/Interpretable3D.

----

## [196] Mimic: Speaking Style Disentanglement for Speech-Driven 3D Facial Animation

**Authors**: *Hui Fu, Zeqing Wang, Ke Gong, Keze Wang, Tianshui Chen, Haojie Li, Haifeng Zeng, Wenxiong Kang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i2.27945](https://doi.org/10.1609/aaai.v38i2.27945)

**Abstract**:

Speech-driven 3D facial animation aims to synthesize vivid facial animations that accurately synchronize with speech and match the unique speaking style. However, existing works primarily focus on achieving precise lip synchronization while neglecting to model the subject-specific speaking style, often resulting in unrealistic facial animations. To the best of our knowledge, this work makes the first attempt to explore the coupled information between the speaking style and the semantic content in facial motions. Specifically, we introduce an innovative speaking style disentanglement method, which enables arbitrary-subject speaking style encoding and leads to a more realistic synthesis of speech-driven facial animations. Subsequently, we propose a novel framework called Mimic to learn disentangled representations of the speaking style and content from facial motions by building two latent spaces for style and content, respectively. Moreover, to facilitate disentangled representation learning, we introduce four well-designed constraints: an auxiliary style classifier, an auxiliary inverse classifier, a content contrastive loss, and a pair of latent cycle losses, which can effectively contribute to the construction of the identity-related style space and semantic-related content space. Extensive qualitative and quantitative experiments conducted on three publicly available datasets demonstrate that our approach outperforms state-of-the-art methods and is capable of capturing diverse speaking styles for speech-driven 3D facial animation. The source code and supplementary video are publicly available at: https://zeqing-wang.github.io/Mimic/

----

## [197] Fine-Grained Multi-View Hand Reconstruction Using Inverse Rendering

**Authors**: *Qijun Gan, Wentong Li, Jinwei Ren, Jianke Zhu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27946](https://doi.org/10.1609/aaai.v38i3.27946)

**Abstract**:

Reconstructing high-fidelity hand models with intricate textures plays a crucial role in enhancing human-object interaction and advancing real-world applications. Despite the state-of-the-art methods excelling in texture generation and image rendering, they often face challenges in accurately capturing geometric details. Learning-based approaches usually offer better robustness and faster inference, which tend to produce smoother results and require substantial amounts of training data. To address these issues, we present a novel fine-grained multi-view hand mesh reconstruction method that leverages inverse rendering to restore hand poses and intricate details. Firstly, our approach predicts a parametric hand mesh model through Graph Convolutional Networks (GCN) based method from multi-view images. We further introduce a novel Hand Albedo and Mesh (HAM) optimization module to refine both the hand mesh and textures, which is capable of preserving the mesh topology. In addition, we suggest an effective mesh-based neural rendering scheme to simultaneously generate photo-realistic image and optimize mesh geometry by fusing the pre-trained rendering network with vertex features. We conduct the comprehensive experiments on InterHand2.6M, DeepHandMesh and dataset collected by ourself, whose promising results show that our proposed approach outperforms the state-of-the-art methods on both reconstruction accuracy and rendering quality. Code and dataset are publicly available at https://github.com/agnJason/FMHR.

----

## [198] Attacking Transformers with Feature Diversity Adversarial Perturbation

**Authors**: *Chenxing Gao, Hang Zhou, Junqing Yu, Yuteng Ye, Jiale Cai, Junle Wang, Wei Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27947](https://doi.org/10.1609/aaai.v38i3.27947)

**Abstract**:

Understanding the mechanisms behind Vision Transformer (ViT), particularly its vulnerability to adversarial perturbations, is crucial for addressing challenges in its real-world applications. Existing ViT adversarial attackers rely on labels to calculate the gradient for perturbation, and exhibit low transferability to other structures and tasks. In this paper, we present a label-free white-box attack approach for ViT-based models that exhibits strong transferability to various black-box models, including most ViT variants, CNNs, and MLPs, even for models developed for other modalities. Our inspiration comes from the feature collapse phenomenon in ViTs, where the critical attention mechanism overly depends on the low-frequency component of features, causing the features in middle-to-end layers to become increasingly similar and eventually collapse. We propose the feature diversity attacker to naturally accelerate this process and achieve remarkable performance and transferability.

----

## [199] Leveraging Imagery Data with Spatial Point Prior for Weakly Semi-supervised 3D Object Detection

**Authors**: *Hongzhi Gao, Zheng Chen, Zehui Chen, Lin Chen, Jiaming Liu, Shanghang Zhang, Feng Zhao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i3.27948](https://doi.org/10.1609/aaai.v38i3.27948)

**Abstract**:

Training high-accuracy 3D detectors necessitates massive labeled 3D annotations with 7 degree-of-freedom, which is laborious and time-consuming. Therefore, the form of point annotations is proposed to offer significant prospects for practical applications in 3D detection, which is not only more accessible and less expensive but also provides strong spatial information for object localization. In this paper, we empirically discover that it is non-trivial to merely adapt Point-DETR to its 3D form, encountering two main bottlenecks: 1) it fails to encode strong 3D prior into the model, and 2) it generates low-quality pseudo labels in distant regions due to the extreme sparsity of LiDAR points. To overcome these challenges, we introduce Point-DETR3D, a teacher-student framework for weakly semi-supervised 3D detection, designed to fully capitalize on point-wise supervision within a constrained instance-wise annotation budget. Different from Point-DETR which encodes 3D positional information solely through a point encoder, we propose an explicit positional query initialization strategy to enhance the positional prior. Considering the low quality of pseudo labels at distant regions produced by the teacher model, we enhance the detector's perception by incorporating dense imagery data through a novel Cross-Modal Deformable RoI Fusion (D-RoI). Moreover, an innovative point-guided self-supervised learning technique is proposed to allow for fully exploiting point priors, even in student models. Extensive experiments on representative nuScenes dataset demonstrate our Point-DETR3D obtains significant improvements compared to previous works. Notably, with only 5% of labeled data, Point-DETR3D achieves over 90% performance of its fully supervised counterpart.

----



[Go to the next page](AAAI-2024-list02.md)

[Go to the catalog section](README.md)