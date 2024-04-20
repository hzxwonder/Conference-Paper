## [1600] Taxonomizing and Measuring Representational Harms: A Look at Image Tagging

**Authors**: *Jared Katzman, Angelina Wang, Morgan Klaus Scheuerman, Su Lin Blodgett, Kristen Laird, Hanna M. Wallach, Solon Barocas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26670](https://doi.org/10.1609/aaai.v37i12.26670)

**Abstract**:

In this paper, we examine computational approaches for measuring the "fairness" of image tagging systems, finding that they cluster into five distinct categories, each with its own analytic foundation. We also identify a range of normative concerns that are often collapsed under the terms "unfairness," "bias," or even "discrimination" when discussing problematic cases of image tagging. Specifically, we identify four types of representational harms that can be caused by image tagging systems, providing concrete examples of each. We then consider how different computational measurement approaches map to each of these types, demonstrating that there is not a one-to-one mapping. Our findings emphasize that no single measurement approach will be definitive and that it is not possible to infer from the use of a particular measurement approach which type of harm was intended to be measured. Lastly, equipped with this more granular understanding of the types of representational harms that can be caused by image tagging systems, we show that attempts to mitigate some of these types of harms may be in tension with one another.

----

## [1601] Winning the CityLearn Challenge: Adaptive Optimization with Evolutionary Search under Trajectory-Based Guidance

**Authors**: *Vanshaj Khattar, Ming Jin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26671](https://doi.org/10.1609/aaai.v37i12.26671)

**Abstract**:

Modern power systems will have to face difficult challenges in the years to come: frequent blackouts in urban areas caused by high peaks of electricity demand, grid instability exacerbated by the intermittency of renewable generation, and climate change on a global scale amplified by increasing carbon emissions. While current practices are growingly inadequate, the pathway of artificial intelligence (AI)-based methods to widespread adoption is hindered by missing aspects of trustworthiness. The CityLearn Challenge is an exemplary opportunity for researchers from multi-disciplinary fields to investigate the potential of AI to tackle these pressing issues within the energy domain, collectively modeled as a reinforcement learning (RL) task. Multiple real-world challenges faced by contemporary RL techniques are embodied in the problem formulation. In this paper, we present a novel method using the solution function of optimization as policies to compute the actions for sequential decision-making, while notably adapting the parameters of the optimization model from online observations. Algorithmically, this is achieved by an evolutionary algorithm under a novel trajectory-based guidance scheme. Formally, the global convergence property is established. Our agent ranked first in the latest 2021 CityLearn Challenge, being able to achieve superior performance in almost all metrics while maintaining some key aspects of interpretability.

----

## [1602] Robust Planning over Restless Groups: Engagement Interventions for a Large-Scale Maternal Telehealth Program

**Authors**: *Jackson A. Killian, Arpita Biswas, Lily Xu, Shresth Verma, Vineet Nair, Aparna Taneja, Aparna Hegde, Neha Madhiwalla, Paula Rodriguez Diaz, Sonja Johnson-Yu, Milind Tambe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26672](https://doi.org/10.1609/aaai.v37i12.26672)

**Abstract**:

In 2020, maternal mortality in India was estimated to be as high as 130 deaths per 100K live births, nearly twice the UN's target. To improve health outcomes, the non-profit ARMMAN sends automated voice messages to expecting and new mothers across India. However, 38% of mothers stop listening to these calls, missing critical preventative care information. To improve engagement, ARMMAN employs health workers to intervene by making service calls, but workers can only call a fraction of the 100K enrolled mothers. Partnering with ARMMAN, we model the problem of allocating limited interventions across mothers as a restless multi-armed bandit (RMAB), where the realities of large scale and model uncertainty present key new technical challenges. We address these with GROUPS, a double oracle–based algorithm for robust planning in RMABs with scalable grouped arms. Robustness over grouped arms requires several methodological advances. First, to adversarially select stochastic group dynamics, we develop a new method to optimize Whittle indices over transition probability intervals. Second, to learn group-level RMAB policy best responses to these adversarial environments, we introduce a weighted index heuristic. Third, we prove a key theoretical result that planning over grouped arms achieves the same minimax regret--optimal strategy as planning over individual arms, under a technical condition. Finally, using real-world data from ARMMAN, we show that GROUPS produces robust policies that reduce minimax regret by up to 50%, halving the number of preventable missed voice messages to connect more mothers with life-saving maternal health information.

----

## [1603] Equivariant Message Passing Neural Network for Crystal Material Discovery

**Authors**: *Astrid Klipfel, Zied Bouraoui, Olivier Peltre, Yaël Frégier, Najwa Harrati, Adlane Sayede*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26673](https://doi.org/10.1609/aaai.v37i12.26673)

**Abstract**:

Automatic material discovery with desired properties is a fundamental challenge for material sciences. Considerable attention has recently been devoted to generating stable crystal structures. While existing work has shown impressive success on supervised tasks such as property prediction, the progress on unsupervised tasks such as material generation is still hampered by the limited extent to which the equivalent geometric representations of the same crystal are considered. To address this challenge, we propose EPGNN a periodic equivariant message-passing neural network that learns crystal lattice deformation in an unsupervised fashion. Our model equivalently acts on lattice according to the deformation action that must be performed, making it suitable for crystal generation, relaxation and optimisation. We present experimental evaluations that demonstrate the effectiveness of our approach.

----

## [1604] Accurate Fairness: Improving Individual Fairness without Trading Accuracy

**Authors**: *Xuran Li, Peng Wu, Jing Su*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26674](https://doi.org/10.1609/aaai.v37i12.26674)

**Abstract**:

Accuracy and individual fairness are both crucial for trustworthy machine learning, but these two aspects are often incompatible with each other so that enhancing one aspect may sacrifice the other inevitably with side effects of true bias or false fairness. We propose in this paper a new fairness criterion, accurate fairness, to align individual fairness with accuracy. Informally, it requires the treatments of an individual and the individual's similar counterparts to conform to a uniform target, i.e., the ground truth of the individual. We prove that accurate fairness also implies typical group fairness criteria over a union of similar sub-populations. We then present a Siamese fairness in-processing approach to minimize the accuracy and fairness losses of a machine learning model under the accurate fairness constraints. To the best of our knowledge, this is the first time that a Siamese approach is adapted for bias mitigation. We also propose fairness confusion matrix-based metrics, fair-precision, fair-recall, and fair-F1 score, to quantify a trade-off between accuracy and individual fairness. Comparative case studies with popular fairness datasets show that our Siamese fairness approach can achieve on average 1.02%-8.78% higher individual fairness (in terms of fairness through awareness) and 8.38%-13.69% higher accuracy, as well as 10.09%-20.57% higher true fair rate, and 5.43%-10.01% higher fair-F1 score, than the state-of-the-art bias mitigation techniques. This demonstrates that our Siamese fairness approach can indeed improve individual fairness without trading accuracy. Finally, the accurate fairness criterion and Siamese fairness approach are applied to mitigate the possible service discrimination with a real Ctrip dataset, by on average fairly serving 112.33% more customers (specifically, 81.29% more customers in an accurately fair way) than baseline models.

----

## [1605] Point-to-Region Co-learning for Poverty Mapping at High Resolution Using Satellite Imagery

**Authors**: *Zhili Li, Yiqun Xie, Xiaowei Jia, Kara Stuart, Caroline Delaire, Sergii Skakun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26675](https://doi.org/10.1609/aaai.v37i12.26675)

**Abstract**:

Despite improvements in safe water and sanitation services in low-income countries, a substantial proportion of the population in Africa still does not have access to these essential services. Up-to-date fine-scale maps of low-income settlements are urgently needed by authorities  to improve service provision. We aim to develop a cost-effective solution to generate fine-scale maps of these vulnerable populations using multi-source public information. The problem is challenging as ground-truth maps are available at only a limited number of cities, and the patterns are heterogeneous across cities. Recent attempts tackling the spatial heterogeneity issue focus on scenarios where true labels partially exist for each input region, which are unavailable for the present problem. We propose a dynamic point-to-region co-learning framework to learn heterogeneity patterns that cannot be reflected by point-level information and generalize deep learners to new areas with no labels. We also propose an attention-based correction layer to remove spurious signatures, and a region-gate to capture both region-invariant and variant patterns. Experiment results on real-world fine-scale data in three cities of Kenya show that the proposed approach can largely improve model performance on various base network architectures.

----

## [1606] AirFormer: Predicting Nationwide Air Quality in China with Transformers

**Authors**: *Yuxuan Liang, Yutong Xia, Songyu Ke, Yiwei Wang, Qingsong Wen, Junbo Zhang, Yu Zheng, Roger Zimmermann*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26676](https://doi.org/10.1609/aaai.v37i12.26676)

**Abstract**:

Air pollution is a crucial issue affecting human health and livelihoods, as well as one of the barriers to economic growth. Forecasting air quality has become an increasingly important endeavor with significant social impacts, especially in emerging countries. In this paper, we present a novel Transformer termed AirFormer to predict nationwide air quality in China, with an unprecedented fine spatial granularity covering thousands of locations. AirFormer decouples the learning process into two stages: 1) a bottom-up deterministic stage that contains two new types of self-attention mechanisms to efficiently learn spatio-temporal representations; 2) a top-down stochastic stage with latent variables to capture the intrinsic uncertainty of air quality data. We evaluate AirFormer with 4-year data from 1,085 stations in Chinese Mainland. Compared to prior models, AirFormer reduces prediction errors by 5%∼8% on 72-hour future predictions. Our source code is available at https://github.com/yoshall/airformer.

----

## [1607] SimFair: A Unified Framework for Fairness-Aware Multi-Label Classification

**Authors**: *Tianci Liu, Haoyu Wang, Yaqing Wang, Xiaoqian Wang, Lu Su, Jing Gao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26677](https://doi.org/10.1609/aaai.v37i12.26677)

**Abstract**:

Recent years have witnessed increasing concerns towards unfair decisions made by machine learning algorithms. To improve fairness in model decisions, various fairness notions have been proposed and many fairness-aware methods are developed. However, most of existing definitions and methods focus only on single-label classification. Fairness for multi-label classification, where each instance is associated with more than one labels, is still yet to establish. To fill this gap, we study fairness-aware multi-label classification in this paper. We start by extending Demographic Parity (DP) and Equalized Opportunity (EOp), two popular fairness notions, to multi-label classification scenarios. Through a systematic study, we show that on multi-label data, because of unevenly distributed labels, EOp usually fails to construct a reliable estimate on labels with few instances. We then propose a new framework named Similarity s-induced Fairness (sγ -SimFair). This new framework utilizes data that have similar labels when estimating fairness on a particular label group for better stability, and can unify DP and EOp. Theoretical analysis and experimental results on real-world datasets together demonstrate the advantage of sγ -SimFair over existing methods on multi-label classification tasks.

----

## [1608] Human Mobility Modeling during the COVID-19 Pandemic via Deep Graph Diffusion Infomax

**Authors**: *Yang Liu, Yu Rong, Zhuoning Guo, Nuo Chen, Tingyang Xu, Fugee Tsung, Jia Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26678](https://doi.org/10.1609/aaai.v37i12.26678)

**Abstract**:

Non-Pharmaceutical Interventions (NPIs), such as social gathering restrictions, have shown effectiveness to slow the transmission of COVID-19 by reducing the contact of people. To support policy-makers, multiple studies have first modelled human mobility via macro indicators (e.g., average daily travel distance) and then study the effectiveness of NPIs. In this work, we focus on mobility modelling and, from a micro perspective, aim to predict locations that will be visited by COVID-19 cases. Since NPIs generally cause economic and societal loss, such a prediction benefits governments when they design and evaluate them. However, in real-world situations, strict privacy data protection regulations result in severe data sparsity problems (i.e., limited case and location information).
To address these challenges and jointly model variables including a geometric graph, a set of diffusions and a set of locations, we propose a model named Deep Graph Diffusion Infomax (DGDI). We show the maximization of DGDI can be bounded by two tractable components: a univariate Mutual Information (MI) between geometric graph and diffusion representation, and a univariate MI between diffusion representation and location representation. To facilitate the research of COVID-19 prediction, we present two benchmarks that contain geometric graphs and location histories of COVID-19 cases. Extensive experiments on the two benchmarks show that DGDI significantly outperforms other competing methods.

----

## [1609] Interpretable Chirality-Aware Graph Neural Network for Quantitative Structure Activity Relationship Modeling in Drug Discovery

**Authors**: *Yunchao Liu, Yu Wang, Oanh Vu, Rocco Moretti, Bobby Bodenheimer, Jens Meiler, Tyler Derr*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26679](https://doi.org/10.1609/aaai.v37i12.26679)

**Abstract**:

In computer-aided drug discovery, quantitative structure activity relation models are trained to predict biological activity from chemical structure. Despite the recent success of applying graph neural network to this task, important chemical information such as molecular chirality is ignored. To fill this crucial gap, we propose Molecular-Kernel Graph NeuralNetwork (MolKGNN) for molecular representation learning, which features SE(3)-/conformation invariance, chirality-awareness, and interpretability. For our MolKGNN, we first design a molecular graph convolution to capture the chemical pattern by comparing the atom's similarity with the learnable molecular kernels. Furthermore, we propagate the similarity score to capture the higher-order chemical pattern. To assess the method, we conduct a comprehensive evaluation with nine well-curated datasets spanning numerous important drug targets that feature realistic high class imbalance and it demonstrates the superiority of MolKGNN over other graph neural networks in computer-aided drug discovery. Meanwhile, the learned kernels identify patterns that agree with domain knowledge, confirming the pragmatic interpretability of this approach.  Our code and supplementary material are publicly available at https://github.com/meilerlab/MolKGNN.

----

## [1610] Task-Adaptive Meta-Learning Framework for Advancing Spatial Generalizability

**Authors**: *Zhexiong Liu, Licheng Liu, Yiqun Xie, Zhenong Jin, Xiaowei Jia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26680](https://doi.org/10.1609/aaai.v37i12.26680)

**Abstract**:

Spatio-temporal machine learning is critically needed for a variety of societal applications, such as agricultural monitoring, hydrological forecast, and traffic management. These applications greatly rely on regional features that characterize spatial and temporal differences. However, spatio-temporal data often exhibit complex patterns and significant data variability across different locations. The labels in many real-world applications can also be limited, which makes it difficult to separately train independent models for different locations. Although meta learning has shown promise in model adaptation with small samples, existing meta learning methods remain limited in handling a large number of heterogeneous tasks, e.g., a large number of locations with varying data patterns. To bridge the gap, we propose task-adaptive formulations and a model-agnostic meta-learning framework that transforms regionally heterogeneous data into location-sensitive meta tasks. We conduct task adaptation following an easy-to-hard task hierarchy in which different meta models are adapted to tasks of different difficulty levels. One major advantage of our proposed method is that it improves the model adaptation to a large number of heterogeneous tasks. It also enhances the model generalization  by automatically adapting the meta model of the corresponding difficulty level to any new tasks. We demonstrate the superiority of our proposed framework over a diverse set of baselines and state-of-the-art meta-learning frameworks. Our extensive experiments on real crop yield data show the effectiveness of the proposed method in handling spatial-related heterogeneous tasks in real societal applications.

----

## [1611] A Composite Multi-Attention Framework for Intraoperative Hypotension Early Warning

**Authors**: *Feng Lu, Wei Li, Zhiqiang Zhou, Cheng Song, Yifei Sun, Yuwei Zhang, Yufei Ren, Xiaofei Liao, Hai Jin, Ailin Luo, Albert Y. Zomaya*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26681](https://doi.org/10.1609/aaai.v37i12.26681)

**Abstract**:

Intraoperative hypotension (IOH) events warning plays a crucial role in preventing postoperative complications, such as postoperative delirium and mortality. Despite significant efforts, two fundamental problems limit its wide clinical use. The well-established IOH event warning systems are often built on proprietary medical devices that may not be available in all hospitals. The warnings are also triggered mainly through a predefined IOH event that might not be suitable for all patients. This work proposes a composite multi-attention (CMA) framework to tackle these problems by conducting short-term predictions on user-definable IOH events using vital signals in a low sampling rate with demographic characteristics. Our framework leverages a multi-modal fusion network to make four vital signals and three demographic characteristics as input modalities. For each modality, a multi-attention mechanism is used for feature extraction for better model training. Experiments on two large-scale real-world data sets show that our method can achieve up to 94.1% accuracy on IOH events early warning while the signals sampling rate is reduced by 3000 times. Our proposal CMA can achieve a mean absolute error of 4.50 mm Hg in the most challenging 15-minute mean arterial pressure prediction task and the error reduction by 42.9% compared to existing solutions.

----

## [1612] Bugs in the Data: How ImageNet Misrepresents Biodiversity

**Authors**: *Alexandra Sasha Luccioni, David Rolnick*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26682](https://doi.org/10.1609/aaai.v37i12.26682)

**Abstract**:

ImageNet-1k is a dataset often used for benchmarking machine learning (ML) models and evaluating tasks such as image recognition and object detection. Wild animals make up 27% of ImageNet-1k but, unlike classes representing people and objects, these data have not been closely scrutinized. In the current paper, we analyze the 13,450 images from 269 classes that represent wild animals in the ImageNet-1k validation set, with the participation of expert ecologists. We find that many of the classes are ill-defined or overlapping, and that 12% of the images are incorrectly labeled, with some classes having >90% of images incorrect. We also find that both the wildlife-related labels and images included in ImageNet-1k present significant geographical and cultural biases, as well as ambiguities such as artificial animals, multiple species in the same image, or the presence of humans. Our findings highlight serious issues with the extensive use of this dataset for evaluating ML systems, the use of such algorithms in wildlife-related tasks, and more broadly the ways in which ML datasets are commonly created and curated.

----

## [1613] LUCID: Exposing Algorithmic Bias through Inverse Design

**Authors**: *Carmen Mazijn, Carina Prunkl, Andres Algaba, Jan Danckaert, Vincent Ginis*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26683](https://doi.org/10.1609/aaai.v37i12.26683)

**Abstract**:

AI systems can create, propagate, support, and automate bias in decision-making processes. To mitigate biased decisions, we both need to understand the origin of the bias and define what it means for an algorithm to make fair decisions. Most group fairness notions assess a model's equality of outcome by computing statistical metrics on the outputs. We argue that these output metrics encounter intrinsic obstacles and present a complementary approach that aligns with the increasing focus on equality of treatment. By Locating Unfairness through Canonical Inverse Design (LUCID), we generate a canonical set that shows the desired inputs for a model given a preferred output. The canonical set reveals the model's internal logic and exposes potential unethical biases by repeatedly interrogating the decision-making process. We evaluate LUCID on the UCI Adult and COMPAS data sets and find that some biases detected by a canonical set differ from those of output metrics. The results show that by shifting the focus towards equality of treatment and looking into the algorithm's internal workings, the canonical sets are a valuable addition to the toolbox of algorithmic fairness evaluation.

----

## [1614] Neighbor Auto-Grouping Graph Neural Networks for Handover Parameter Configuration in Cellular Network

**Authors**: *Mehrtash Mehrabi, Walid Masoudimansour, Yingxue Zhang, Jie Chuai, Zhitang Chen, Mark Coates, Jianye Hao, Yanhui Geng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26684](https://doi.org/10.1609/aaai.v37i12.26684)

**Abstract**:

The mobile communication enabled by cellular networks is the one of the main foundations of our modern society. Optimizing the performance of cellular networks and providing massive connectivity with improved coverage and user experience has a considerable social and economic impact on our daily life. This performance relies heavily on the configuration of the network parameters. However, with the massive increase in both the size and complexity of cellular networks, network management, especially parameter configuration, is becoming complicated. The current practice, which relies largely on experts' prior knowledge, is not adequate and will require lots of domain experts and high maintenance costs. In this work, we propose a learning-based framework for handover parameter configuration. The key challenge, in this case, is to tackle the complicated dependencies between neighboring cells and jointly optimize the whole network. Our framework addresses this challenge in two ways. First, we introduce a novel approach to imitate how the network responds to different network states and parameter values, called auto-grouping graph convolutional network (AG-GCN). During the parameter configuration stage, instead of solving the global optimization problem, we design a local multi-objective optimization strategy where each cell considers several local performance metrics to balance its own performance and its neighbors. We evaluate our proposed algorithm via a simulator constructed using real network data. We demonstrate that the handover parameters our model can find, achieve better average network throughput compared to those recommended by experts as well as alternative baselines, which can bring better network quality and stability. It has the potential to massively reduce costs arising from human expert intervention and maintenance.

----

## [1615] Help Me Heal: A Reinforced Polite and Empathetic Mental Health and Legal Counseling Dialogue System for Crime Victims

**Authors**: *Kshitij Mishra, Priyanshu Priya, Asif Ekbal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26685](https://doi.org/10.1609/aaai.v37i12.26685)

**Abstract**:

The potential for conversational agents offering mental health and legal counseling in an autonomous, interactive, and vitally accessible environment is getting highlighted due to the increased access to information through the internet and mobile devices. A counseling conversational agent should be able to offer higher engagement mimicking the real-time counseling sessions. The ability to empathize or comprehend and feel another person’s emotions and experiences is a crucial quality that promotes effective therapeutic bonding and rapport-building. Further, the use of polite encoded language in the counseling reflects the nobility and creates a familiar, warm, and comfortable atmosphere to resolve human issues. Therefore, focusing on these two aspects, we propose a Polite and Empathetic Mental Health and Legal Counseling Dialogue System (Po-Em-MHLCDS) for the victims of crimes. To build Po-Em-MHLCDS, we first create a Mental Health and Legal Counseling Dataset (MHLCD) by recruiting six employees who are asked to converse with each other, acting as a victim and the agent interchangeably following a fixed stated guidelines. Second, the MHLCD dataset is annotated with three informative labels, viz. counseling strategies, politeness, and empathy. Lastly, we train the Po-Em-MHLCDS in a reinforcement learning framework by designing an efficient and effective reward function to reinforce correct counseling strategy, politeness and empathy while maintaining contextual-coherence and non-repetitiveness in the generated responses. Our extensive automatic and human evaluation demonstrate the strength of the proposed system. Codes and Data can be accessed at https://www.iitp.ac.in/ ai-nlp-ml/resources.html#MHLCD or https://github.com/Mishrakshitij/Po-Em-MHLCDS

----

## [1616] Carburacy: Summarization Models Tuning and Comparison in Eco-Sustainable Regimes with a Novel Carbon-Aware Accuracy

**Authors**: *Gianluca Moro, Luca Ragazzi, Lorenzo Valgimigli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26686](https://doi.org/10.1609/aaai.v37i12.26686)

**Abstract**:

Generative transformer-based models have reached cutting-edge performance in long document summarization. Nevertheless, this task is witnessing a paradigm shift in developing ever-increasingly computationally-hungry solutions, focusing on effectiveness while ignoring the economic, environmental, and social costs of yielding such results. Accordingly, such extensive resources impact climate change and raise barriers to small and medium organizations distinguished by low-resource regimes of hardware and data. As a result, this unsustainable trend has lifted many concerns in the community, which directs the primary efforts on the proposal of tools to monitor models' energy costs. Despite their importance, no evaluation measure considering models' eco-sustainability exists yet. In this work, we propose Carburacy, the first carbon-aware accuracy measure that captures both model effectiveness and eco-sustainability. We perform a comprehensive benchmark for long document summarization, comparing multiple state-of-the-art quadratic and linear transformers on several datasets under eco-sustainable regimes. Finally, thanks to Carburacy, we found optimal combinations of hyperparameters that let models be competitive in effectiveness with significantly lower costs.

----

## [1617] Joint Self-Supervised Image-Volume Representation Learning with Intra-inter Contrastive Clustering

**Authors**: *Duy M. H. Nguyen, Hoang Nguyen, Truong Thanh Nhat Mai, Tri Cao, Binh T. Nguyen, Nhat Ho, Paul Swoboda, Shadi Albarqouni, Pengtao Xie, Daniel Sonntag*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26687](https://doi.org/10.1609/aaai.v37i12.26687)

**Abstract**:

Collecting large-scale medical datasets with fully annotated samples for training of deep networks is prohibitively expensive, especially for 3D volume data. Recent breakthroughs in self-supervised learning (SSL) offer the ability to overcome the lack of labeled training samples by learning feature representations from unlabeled data. However, most current SSL techniques in the medical field have been designed for either 2D images or 3D volumes. In practice, this restricts the capability to fully leverage unlabeled data from numerous sources, which may include both 2D and 3D data. Additionally, the use of these pre-trained networks is constrained to downstream tasks with compatible data dimensions.
In this paper, we propose a novel framework for unsupervised joint learning on 2D and 3D data modalities. Given a set of 2D images or 2D slices extracted from 3D volumes, we construct an SSL task based on a 2D contrastive clustering problem for distinct classes. The 3D volumes are exploited by computing vectored embedding at each slice and then assembling a holistic feature through deformable self-attention mechanisms in Transformer, allowing incorporating long-range dependencies between slices inside 3D volumes. These holistic features are further utilized to define a novel 3D clustering agreement-based SSL task and masking embedding prediction inspired by pre-trained language models. Experiments on downstream tasks, such as 3D brain segmentation, lung nodule detection, 3D heart structures segmentation, and abnormal chest X-ray detection, demonstrate the effectiveness of our joint 2D and 3D SSL approach. We improve plain 2D Deep-ClusterV2 and SwAV by a significant margin and also surpass various modern 2D and 3D SSL approaches.

----

## [1618] For the Underrepresented in Gender Bias Research: Chinese Name Gender Prediction with Heterogeneous Graph Attention Network

**Authors**: *Zihao Pan, Kai Peng, Shuai Ling, Haipeng Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26688](https://doi.org/10.1609/aaai.v37i12.26688)

**Abstract**:

Achieving gender equality is an important pillar for humankind’s sustainable future. Pioneering data-driven gender bias research is based on large-scale public records such as scientific papers, patents, and company registrations, covering female researchers, inventors and entrepreneurs, and so on. Since gender information is often missing in relevant datasets, studies rely on tools to infer genders from names. However, available open-sourced Chinese gender-guessing tools are not yet suitable for scientific purposes, which may be partially responsible for female Chinese being underrepresented in mainstream gender bias research and affect their universality. Specifically, these tools focus on character-level information while overlooking the fact that the combinations of Chinese characters in multi-character names, as well as the components and pronunciations of characters, convey important messages. As a first effort, we design a Chinese Heterogeneous Graph Attention (CHGAT) model to capture the heterogeneity in component relationships and incorporate the pronunciations of characters. Our model largely surpasses current tools and also outperforms the state-of-the-art algorithm. Last but not least, the most popular Chinese name-gender dataset is single-character based with far less female coverage from an unreliable source, naturally hindering relevant studies. We open-source a more balanced multi-character dataset from an official source together with our code, hoping to help future research promoting gender equality.

----

## [1619] FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms

**Authors**: *Peng Qi, Yuyan Bu, Juan Cao, Wei Ji, Ruihao Shui, Junbin Xiao, Danding Wang, Tat-Seng Chua*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26689](https://doi.org/10.1609/aaai.v37i12.26689)

**Abstract**:

Short video platforms have become an important channel for news sharing, but also a new breeding ground for fake news. To mitigate this problem, research of fake news video detection has recently received a lot of attention. Existing works face two roadblocks: the scarcity of comprehensive and largescale datasets and insufficient utilization of multimodal information. Therefore, in this paper, we construct the largest Chinese short video dataset about fake news named FakeSV, which includes news content, user comments, and publisher profiles simultaneously. To understand the characteristics of fake news videos, we conduct exploratory analysis of FakeSV from different perspectives. Moreover, we provide a new multimodal detection model named SV-FEND, which exploits the cross-modal correlations to select the most informative features and utilizes the social context information for detection. Extensive experiments evaluate the superiority of the proposed method and provide detailed comparisons of different methods and modalities for future works. Our dataset and codes are available in https://github.com/ICTMCG/FakeSV.

----

## [1620] EINNs: Epidemiologically-Informed Neural Networks

**Authors**: *Alexander Rodríguez, Jiaming Cui, Naren Ramakrishnan, Bijaya Adhikari, B. Aditya Prakash*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26690](https://doi.org/10.1609/aaai.v37i12.26690)

**Abstract**:

We introduce EINNs, a framework crafted for epidemic forecasting that builds upon the theoretical grounds provided by mechanistic models as well as the data-driven expressibility afforded by AI models, and their capabilities to ingest heterogeneous information. Although neural forecasting models have been successful in multiple tasks, predictions well-correlated with epidemic trends and long-term predictions remain open challenges. Epidemiological ODE models contain mechanisms that can guide us in these two tasks; however, they have limited capability of ingesting data sources and modeling composite signals. Thus, we propose to leverage work in physics-informed neural networks to learn latent epidemic dynamics and transfer relevant knowledge to another neural network which ingests multiple data sources and has more appropriate inductive bias. In contrast with previous work, we do not assume the observability of complete dynamics and do not need to numerically solve the ODE equations during training. Our thorough experiments on all US states and HHS regions for COVID-19 and influenza forecasting showcase the clear benefits of our approach in both short-term and long-term forecasting as well as in learning the mechanistic dynamics over other non-trivial alternatives.

----

## [1621] Counterfactual Fairness Is Basically Demographic Parity

**Authors**: *Lucas Rosenblatt, R. Teal Witter*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26691](https://doi.org/10.1609/aaai.v37i12.26691)

**Abstract**:

Making fair decisions is crucial to ethically implementing machine learning algorithms in social settings. In this work, we consider the celebrated definition of counterfactual fairness. We begin by showing that an algorithm which satisfies counterfactual fairness also satisfies demographic parity, a far simpler fairness constraint. Similarly, we show that all algorithms satisfying demographic parity can be trivially modified to satisfy counterfactual fairness. Together, our results indicate that counterfactual fairness is basically equivalent to demographic parity, which has important implications for the growing body of work on counterfactual fairness. We then validate our theoretical findings empirically, analyzing three existing algorithms for counterfactual fairness against three simple benchmarks. We find that two simple benchmark algorithms outperform all three existing algorithms---in terms of fairness, accuracy, and efficiency---on several data sets. Our analysis leads us to formalize a concrete fairness goal: to preserve the order of individuals within protected groups. We believe transparency around the ordering of individuals within protected groups makes fair algorithms more trustworthy. By design, the two simple benchmark algorithms satisfy this goal while the existing algorithms do not.

----

## [1622] Detecting Anomalous Networks of Opioid Prescribers and Dispensers in Prescription Drug Data

**Authors**: *Katie Rosman, Daniel B. Neill*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26692](https://doi.org/10.1609/aaai.v37i12.26692)

**Abstract**:

The opioid overdose epidemic represents a serious public health crisis, with fatality rates rising considerably over the past several years. To help address the abuse of prescription opioids, state governments collect data on dispensed prescriptions, yet the use of these data is typically limited to manual searches. In this paper, we propose a novel graph-based framework for detecting anomalous opioid prescribing patterns in state Prescription Drug Monitoring Program (PDMP) data, which could aid governments in deterring opioid diversion and abuse. Specifically, we seek to identify connected networks of opioid prescribers and dispensers who engage in high-risk and possibly illicit activity. We develop and apply a novel extension of the Non-Parametric Heterogeneous Graph Scan (NPHGS) to two years of de-identified PDMP data from the state of Kansas, and find that NPHGS identifies subgraphs that are significantly more anomalous than those detected by other graph-based methods. NPHGS also reveals clusters of potentially illicit activity, which may strengthen state law enforcement and regulatory capabilities. Our paper is the first to demonstrate how prescription data can systematically identify anomalous opioid prescribers and dispensers, as well as illustrating the efficacy of a network-based approach. Additionally, our technical extensions to NPHGS offer both improved flexibility and graph density reduction, enabling the framework to be replicated across jurisdictions and extended to other problem domains.

----

## [1623] Practical Disruption of Image Translation Deepfake Networks

**Authors**: *Nataniel Ruiz, Sarah Adel Bargal, Cihang Xie, Stan Sclaroff*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26693](https://doi.org/10.1609/aaai.v37i12.26693)

**Abstract**:

By harnessing the latest advances in deep learning, image-to-image translation architectures have recently achieved impressive capabilities. Unfortunately, the growing representational power of these architectures has prominent unethical uses. Among these, the threats of (1) face manipulation ("DeepFakes") used for misinformation or pornographic use (2) "DeepNude" manipulations of body images to remove clothes from individuals, etc. Several works tackle the task of disrupting such image translation networks by inserting imperceptible adversarial attacks into the input image. Nevertheless, these works have limitations that may result in disruptions that are not practical in the real world. Specifically, most works generate disruptions in a white-box scenario, assuming perfect knowledge about the image translation network. The few remaining works that assume a black-box scenario require a large number of queries to successfully disrupt the adversary's image translation network. In this work we propose Leaking Transferable Perturbations (LTP), an algorithm that significantly reduces the number of queries needed to disrupt an image translation network by dynamically re-purposing previous disruptions into new query efficient disruptions.

----

## [1624] Daycare Matching in Japan: Transfers and Siblings

**Authors**: *Zhaohong Sun, Yoshihiro Takenami, Daisuke Moriwaki, Yoji Tomita, Makoto Yokoo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26694](https://doi.org/10.1609/aaai.v37i12.26694)

**Abstract**:

In this paper, we study a daycare matching problem in Japan and report the design and implementation of a new centralized algorithm, which is going to be deployed in one municipality in the Tokyo metropolis. There are two features that make this market different from the classical hospital-doctor matching problem: i) some children are initially enrolled and prefer to be transferred to other daycare centers; ii) one family may be associated with two or more children and is allowed to submit preferences over combinations of daycare centers. We revisit some well-studied properties including individual rationality, non-wastefulness, as well as stability, and generalize them to this new setting. We design an algorithm based on integer programming (IP) that captures these properties and conduct experiments on five real-life data sets provided by three municipalities. Experimental results show that i) our algorithm performs at least as well as currently used methods in terms of numbers of matched children and blocking coalition; ii) we can find a stable outcome for all instances, although the existence of such an outcome is not guaranteed in theory.

----

## [1625] City-Scale Pollution Aware Traffic Routing by Sampling Max Flows Using MCMC

**Authors**: *Shreevignesh Suriyanarayanan, Praveen Paruchuri, Girish Varma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26695](https://doi.org/10.1609/aaai.v37i12.26695)

**Abstract**:

A significant cause of air pollution in urban areas worldwide is the high volume of road traffic. Long-term exposure to severe pollution can cause serious health issues. One approach towards tackling this problem is to design a pollution-aware traffic routing policy that balances multiple objectives of i) avoiding extreme pollution in any area ii) enabling short transit times, and iii) making effective use of the road capacities. We propose a novel sampling-based approach for this problem. We give the first construction of a Markov Chain that can sample integer max flow solutions of a planar graph, with theoretical guarantees that the probabilities depend on the aggregate transit length. We designed a traffic policy using diverse samples and simulated traffic on real-world road maps using the SUMO traffic simulator. We observe a considerable decrease in areas with severe pollution when experimented with maps of large cities across the world compared to other approaches.

----

## [1626] Weather2vec: Representation Learning for Causal Inference with Non-local Confounding in Air Pollution and Climate Studies

**Authors**: *Mauricio Tec, James G. Scott, Corwin M. Zigler*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26696](https://doi.org/10.1609/aaai.v37i12.26696)

**Abstract**:

Estimating the causal effects of a spatially-varying intervention on a spatially-varying outcome may be subject to non-local confounding (NLC), a phenomenon that can bias estimates when the treatments and outcomes of a given unit are dictated in part by the covariates of other nearby units. In particular, NLC is a challenge for evaluating the effects of environmental policies and climate events on health-related outcomes such as air pollution exposure. This paper first formalizes NLC using the potential outcomes framework, providing a comparison with the related phenomenon of causal interference. Then, it proposes a broadly applicable framework, termed weather2vec, that uses the theory of balancing scores to learn representations of non-local information into a scalar or vector defined for each observational unit, which is subsequently used to adjust for confounding in conjunction with causal inference methods. The framework is evaluated in a simulation study and two case studies on air pollution where the weather is an (inherently regional) known confounder.

----

## [1627] Evaluating Digital Agriculture Recommendations with Causal Inference

**Authors**: *Ilias Tsoumas, Georgios Giannarakis, Vasileios Sitokonstantinou, Alkiviadis Koukos, Dimitra Loka, Nikolaos S. Bartsotas, Charalampos Kontoes, Ioannis Athanasiadis*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26697](https://doi.org/10.1609/aaai.v37i12.26697)

**Abstract**:

In contrast to the rapid digitalization of several industries, agriculture suffers from low adoption of smart farming tools. Even though recent advancements in AI-driven digital agriculture can offer high-performing predictive functionalities, they lack tangible quantitative evidence on their benefits to the farmers. Field experiments can derive such evidence, but are often costly, time consuming and hence limited in scope and scale of application. To this end, we propose an observational causal inference framework for the empirical evaluation of the impact of digital tools on target farm performance indicators (e.g., yield in this case). This way, we can increase farmers' trust via enhancing the transparency of the digital agriculture market, and in turn accelerate the adoption of technologies that aim to secure farmer income resilience and global agricultural sustainability against a changing climate. As a case study, we designed and implemented a recommendation system for the optimal sowing time of cotton based on numerical weather predictions, which was used by a farmers' cooperative during the growing season of 2021. We then leverage agricultural knowledge, collected yield data, and environmental information to develop a causal graph of the farm system. Using the back-door criterion, we identify the impact of sowing recommendations on the yield and subsequently estimate it using linear regression, matching, inverse propensity score weighting and meta-learners. The results revealed that a field sown according to our recommendations exhibited a statistically significant yield increase that ranged from 12% to 17%, depending on the method. The effect estimates were robust, as indicated by the agreement among the estimation methods and four successful refutation tests. We argue that this approach can be implemented for decision support systems of other fields, extending their evaluation beyond a performance assessment of internal functionalities.

----

## [1628] Everyone's Voice Matters: Quantifying Annotation Disagreement Using Demographic Information

**Authors**: *Ruyuan Wan, Jaehyung Kim, Dongyeop Kang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26698](https://doi.org/10.1609/aaai.v37i12.26698)

**Abstract**:

In NLP annotation, it is common to have multiple annotators label the text and then obtain the ground truth labels based on major annotators’ agreement. However, annotators are individuals with different backgrounds and various voices. When annotation tasks become subjective, such as detecting politeness, offense, and social norms, annotators’ voices differ and vary. Their diverse voices may represent the true distribution of people’s opinions on subjective matters. Therefore, it is crucial to study the disagreement from annotation to understand which content is controversial from the annotators. In our research, we extract disagreement labels from five subjective datasets, then fine-tune language models to predict annotators’ disagreement. Our results show that knowing annotators’ demographic information (e.g., gender, ethnicity, education level), in addition to the task text, helps predict the disagreement. To investigate the effect of annotators’ demographics on their disagreement level, we simulate different combinations of their artificial demographics and explore the variance of the prediction to distinguish the disagreement from the inherent controversy from text content and the disagreement in the annotators’ perspective. Overall, we propose an innovative disagreement prediction mechanism for better design of the annotation process that will achieve more accurate and inclusive results for NLP systems. Our code and dataset are publicly available.

----

## [1629] MixFairFace: Towards Ultimate Fairness via MixFair Adapter in Face Recognition

**Authors**: *Fu-En Wang, Chien-Yi Wang, Min Sun, Shang-Hong Lai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26699](https://doi.org/10.1609/aaai.v37i12.26699)

**Abstract**:

Although significant progress has been made in face recognition, demographic bias still exists in face recognition systems. For instance, it usually happens that the face recognition performance for a certain demographic group is lower than the others. In this paper, we propose MixFairFace framework to improve the fairness in face recognition models. First of all, we argue that the commonly used attribute-based fairness metric is not appropriate for face recognition. A face recognition system can only be considered fair while every person has a close performance. Hence, we propose a new evaluation protocol to fairly evaluate the fairness performance of different approaches. Different from previous approaches that require sensitive attribute labels such as race and gender for reducing the demographic bias, we aim at addressing the identity bias in face representation, i.e., the performance inconsistency between different identities, without the need for sensitive attribute labels. To this end, we propose MixFair Adapter to determine and reduce the identity bias of training samples. Our extensive experiments demonstrate that our MixFairFace approach achieves state-of-the-art fairness performance on all benchmark datasets.

----

## [1630] PateGail: A Privacy-Preserving Mobility Trajectory Generator with Imitation Learning

**Authors**: *Huandong Wang, Changzheng Gao, Yuchen Wu, Depeng Jin, Lina Yao, Yong Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26700](https://doi.org/10.1609/aaai.v37i12.26700)

**Abstract**:

Generating human mobility trajectories is of great importance to solve the lack of large-scale trajectory data in numerous applications, which is caused by privacy concerns. However, existing mobility trajectory generation methods still require real-world human trajectories centrally collected as the training data, where there exists an inescapable risk of privacy leakage. To overcome this limitation, in this paper, we propose PateGail, a privacy-preserving imitation learning model to generate mobility trajectories, which utilizes the powerful generative adversary imitation learning model to simulate the decision-making process of humans. Further, in order to protect user privacy, we train this model collectively based on decentralized mobility data stored in user devices, where personal discriminators are trained locally to distinguish and reward the real and generated human trajectories. In the training process, only the generated trajectories and their rewards obtained based on personal discriminators are shared between the server and devices, whose privacy is further preserved by our proposed perturbation mechanisms with theoretical proof to satisfy differential privacy. Further, to better model the human decision-making process, we propose a novel aggregation mechanism of the rewards obtained from personal discriminators. We theoretically prove that under the reward obtained based on the aggregation mechanism, our proposed model maximizes the lower bound of the discounted total rewards of users. Extensive experiments show that the trajectories generated by our model are able to resemble real-world trajectories in terms of five key statistical metrics, outperforming state-of-the-art algorithms by over 48.03%. Furthermore, we demonstrate that the synthetic trajectories are able to efficiently support practical applications, including mobility prediction and location recommendation.

----

## [1631] Noise Based Deepfake Detection via Multi-Head Relative-Interaction

**Authors**: *Tianyi Wang, Kam-Pui Chow*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26701](https://doi.org/10.1609/aaai.v37i12.26701)

**Abstract**:

Deepfake brings huge and potential negative impacts to our daily lives. As the real-life Deepfake videos circulated on the Internet become more authentic, most existing detection algorithms have failed since few visual differences can be observed between an authentic video and a Deepfake one. However, the forensic traces are always retained within the synthesized videos. In this study, we present a noise-based Deepfake detection model, NoiseDF for short, which focuses on the underlying forensic noise traces left behind the Deepfake videos. In particular, we enhance the RIDNet denoiser to extract noise traces and features from the cropped face and background squares of the video image frames. Meanwhile, we devise a novel Multi-Head Relative-Interaction method to evaluate the degree of interaction between the faces and backgrounds that plays a pivotal role in the Deepfake detection task. Besides outperforming the state-of-the-art models, the visualization of the extracted Deepfake forensic noise traces has further displayed the evidence and proved the robustness of our approach.

----

## [1632] Semi-supervised Credit Card Fraud Detection via Attribute-Driven Graph Representation

**Authors**: *Sheng Xiang, Mingzhi Zhu, Dawei Cheng, Enxia Li, Ruihui Zhao, Yi Ouyang, Ling Chen, Yefeng Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26702](https://doi.org/10.1609/aaai.v37i12.26702)

**Abstract**:

Credit card fraud incurs a considerable cost for both cardholders and issuing banks. Contemporary methods apply machine learning-based classifiers to detect fraudulent behavior from labeled transaction records. But labeled data are usually a small proportion of billions of real transactions due to expensive labeling costs, which implies that they do not well exploit many natural features from unlabeled data. Therefore, we propose a semi-supervised graph neural network for fraud detection. Specifically, we leverage transaction records to construct a temporal transaction graph, which is composed of temporal transactions (nodes) and interactions (edges) among them. Then we pass messages among the nodes through a Gated Temporal Attention Network (GTAN) to learn the transaction representation. We further model the fraud patterns through risk propagation among transactions. The extensive experiments are conducted on a real-world transaction dataset and two publicly available fraud detection datasets. The result shows that our proposed method, namely GTAN, outperforms other state-of-the-art baselines on three fraud detection datasets. Semi-supervised experiments demonstrate the excellent fraud detection performance of our model with only a tiny proportion of labeled data.

----

## [1633] Privacy-Preserved Evolutionary Graph Modeling via Gromov-Wasserstein Autoregression

**Authors**: *Yue Xiang, Dixin Luo, Hongteng Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26703](https://doi.org/10.1609/aaai.v37i12.26703)

**Abstract**:

Real-world graphs like social networks are often evolutionary over time, whose observations at different timestamps lead to graph sequences. Modeling such evolutionary graphs is important for many applications, but solving this problem often requires the correspondence between the graphs at different timestamps, which may leak private node information, e.g., the temporal behavior patterns of the nodes. We proposed a Gromov-Wasserstein Autoregressive (GWAR) model to capture the generative mechanisms of evolutionary graphs, which does not require the correspondence information and thus preserves the privacy of the graphs' nodes. This model consists of two autoregressions, predicting the number of nodes and the probabilities of nodes and edges, respectively. The model takes observed graphs as its input and predicts future graphs via solving a joint graph alignment and merging task. This task leads to a fused Gromov-Wasserstein (FGW) barycenter problem, in which we approximate the alignment of the graphs based on a novel inductive fused Gromov-Wasserstein (IFGW) distance. The IFGW distance is parameterized by neural networks and can be learned under mild assumptions, thus, we can infer the FGW barycenters without iterative optimization and predict future graphs efficiently. Experiments show that our GWAR achieves encouraging performance in modeling evolutionary graphs in privacy-preserving scenarios.

----

## [1634] Auto-CM: Unsupervised Deep Learning for Satellite Imagery Composition and Cloud Masking Using Spatio-Temporal Dynamics

**Authors**: *Yiqun Xie, Zhili Li, Han Bao, Xiaowei Jia, Dongkuan Xu, Xun Zhou, Sergii Skakun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26704](https://doi.org/10.1609/aaai.v37i12.26704)

**Abstract**:

Cloud masking is both a fundamental and a critical task in the vast majority of Earth observation problems across social sectors, including agriculture, energy, water, etc. The sheer volume of satellite imagery to be processed has fast-climbed to a scale (e.g., >10 PBs/year) that is prohibitive for manual processing. Meanwhile, generating reliable cloud masks and image composite is increasingly challenging due to the continued distribution-shifts in the imagery collected by existing sensors and the ever-growing variety of sensors and platforms. Moreover, labeled samples are scarce and geographically limited compared to the needs in real large-scale applications. In related work, traditional remote sensing methods are often physics-based and rely on special spectral signatures from multi- or hyper-spectral bands, which are often not available in data collected by many -- and especially more recent -- high-resolution platforms. Machine learning and deep learning based methods, on the other hand, often require large volumes of up-to-date training data to be reliable and generalizable over space. We propose an autonomous image composition and masking (Auto-CM) framework to learn to solve the fundamental tasks in a label-free manner, by leveraging different dynamics of events in both geographic domains and time-series. Our experiments show that Auto-CM outperforms existing methods on a wide-range of data with different satellite platforms, geographic regions and bands.

----

## [1635] ERASER: AdvERsArial Sensitive Element Remover for Image Privacy Preservation

**Authors**: *Guang Yang, Juan Cao, Danding Wang, Peng Qi, Jintao Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26705](https://doi.org/10.1609/aaai.v37i12.26705)

**Abstract**:

The daily practice of online image sharing enriches our lives, but also raises a severe issue of privacy leakage. To mitigate the privacy risks during image sharing, some researchers modify the sensitive elements in images with visual obfuscation methods including traditional ones like blurring and pixelating, as well as generative ones based on deep learning. However, images processed by such methods may be recovered or recognized by models, which cannot guarantee privacy. Further, traditional methods make the images very unnatural with low image quality. Although generative methods produce better images, most of them suffer from insufficiency in the frequency domain, which influences image quality. Therefore, we propose the AdvERsArial Sensitive Element Remover (ERASER) to guarantee both image privacy and image quality. 1) To preserve image privacy, for the regions containing sensitive elements, ERASER guarantees enough difference after being modified in an adversarial way. Specifically, we take both the region and global content into consideration with a Prior Transformer and obtain the corresponding region prior and global prior. Based on the priors, ERASER is trained with an adversarial Difference Loss to make the content in the regions different. As a result, ERASER can reserve the main structure and change the texture of the target regions for image privacy preservation. 2) To guarantee the image quality, ERASER improves the frequency insufficiency of current generative methods. Specifically, the region prior and global prior are processed with Fast Fourier Convolution to capture characteristics and achieve consistency in both pixel and frequency domains. Quantitative analyses demonstrate that the proposed ERASER achieves a balance between image quality and image privacy preservation, while qualitative analyses demonstrate that ERASER indeed reduces the privacy risk from the visual perception aspect.

----

## [1636] Deep Learning on a Healthy Data Diet: Finding Important Examples for Fairness

**Authors**: *Abdelrahman Zayed, Prasanna Parthasarathi, Gonçalo Mordido, Hamid Palangi, Samira Shabanian, Sarath Chandar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26706](https://doi.org/10.1609/aaai.v37i12.26706)

**Abstract**:

Data-driven predictive solutions predominant in commercial applications tend to suffer from biases and stereotypes, which raises equity concerns. Prediction models may discover, use, or amplify spurious correlations based on gender or other protected personal characteristics, thus discriminating against marginalized groups. Mitigating gender bias has become an important research focus in natural language processing (NLP) and is an area where annotated corpora are available. Data augmentation reduces gender bias by adding counterfactual examples to the training dataset. In this work, we show that some of the examples in the augmented dataset can be not important or even harmful to fairness. We hence propose a general method for pruning both the factual and counterfactual examples to maximize the model’s fairness as measured by the demographic parity, equality of opportunity, and equality of odds. The fairness achieved by our method surpasses that of data augmentation on three text classification datasets, using no more than half of the examples in the augmented dataset. Our experiments are conducted using models of varying sizes and pre-training settings. WARNING: This work uses language that is offensive in nature.

----

## [1637] On the Effectiveness of Curriculum Learning in Educational Text Scoring

**Authors**: *Zijie Zeng, Dragan Gasevic, Guanliang Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26707](https://doi.org/10.1609/aaai.v37i12.26707)

**Abstract**:

Automatic Text Scoring (ATS) is a widely-investigated task in education. Existing approaches often stressed the structure design of an ATS model and neglected the training process of the model. Considering the difficult nature of this task, we argued that the performance of an ATS model could be potentially boosted by carefully selecting data of varying complexities in the training process. Therefore, we aimed to investigate the effectiveness of curriculum learning (CL) in scoring educational text. Specifically, we designed two types of difficulty measurers: (i) pre-defined, calculated by measuring a sample's readability, length, the number of grammatical errors or unique words it contains; and (ii) automatic, calculated based on whether a model in a training epoch can accurately score the samples. These measurers were tested in both the easy-to-hard to hard-to-easy training paradigms. Through extensive evaluations on two widely-used datasets (one for short answer scoring and the other for long essay scoring), we demonstrated that (a) CL indeed could boost the performance of state-of-the-art ATS models, and the maximum improvement could be up to 4.5%, but most improvements were achieved when assessing short and easy answers; (b) the pre-defined measurer calculated based on the number of grammatical errors contained in a text sample tended to outperform the other difficulty measurers across different training paradigms.

----

## [1638] Censored Fairness through Awareness

**Authors**: *Wenbin Zhang, Tina Hernandez-Boussard, Jeremy C. Weiss*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26708](https://doi.org/10.1609/aaai.v37i12.26708)

**Abstract**:

There has been increasing concern within the machine learning community and beyond that Artificial Intelligence (AI) faces a bias and discrimination crisis which needs AI fairness with urgency. As many have begun to work on this problem, most existing work depends on the availability of class label for the given fairness definition and algorithm which may not align with real-world usage. In this work, we study an AI fairness problem that stems from the gap between the design of a "fair" model in the lab and its deployment in the real-world. Specifically, we consider defining and mitigating individual unfairness amidst censorship, where the availability of class label is not always guaranteed due to censorship, which is broadly applicable in a diversity of real-world socially sensitive applications. We show that our method is able to quantify and mitigate individual unfairness in the presence of censorship across three benchmark tasks, which provides the first known results on individual fairness guarantee in analysis of censored data.

----

## [1639] A Continual Pre-training Approach to Tele-Triaging Pregnant Women in Kenya

**Authors**: *Wenbo Zhang, Hangzhi Guo, Prerna Ranganathan, Jay Patel, Sathyanath Rajasekharan, Nidhi Danayak, Manan Gupta, Amulya Yadav*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26709](https://doi.org/10.1609/aaai.v37i12.26709)

**Abstract**:

Access to high-quality maternal health care services is limited in Kenya, which resulted in ∼36,000 maternal and neonatal deaths in 2018. To tackle this challenge, Jacaranda Health (a non-profit organization working on maternal health in Kenya) developed PROMPTS, an SMS based tele-triage system for pregnant and puerperal women, which has more than 350,000 active users in Kenya. PROMPTS empowers pregnant women living far away from doctors and hospitals to send SMS messages to get quick answers (through human helpdesk agents) to questions about their medical symptoms and pregnancy status. Unfortunately, ∼1.1 million SMS messages are received by PROMPTS every month, which makes it challenging for helpdesk agents to ensure that these messages can be interpreted correctly and evaluated by their level of emergency to ensure timely responses and/or treatments for women in need. This paper reports on a collaborative effort with Jacaranda Health to develop a state-of-the-art natural language processing (NLP) framework, TRIM-AI (TRIage for Mothers using AI), which can automatically predict the emergency level (or severity of medical condition) of a pregnant mother based on the content of their SMS messages. TRIM-AI leverages recent advances in multi-lingual pre-training and continual pre-training to tackle code-mixed SMS messages (between English and Swahili), and achieves a weighted F1 score of 0.774 on real-world datasets. TRIM-AI has been successfully deployed in the field since June 2022, and is being used by Jacaranda Health to prioritize the provision of services and care to pregnant women with the most critical medical conditions. Our preliminary A/B tests in the field show that TRIM-AI is ∼17% more accurate at predicting high-risk medical conditions from SMS messages sent by pregnant Kenyan mothers, which reduces the helpdesk’s workload by ∼12%.

----

## [1640] Future Aware Pricing and Matching for Sustainable On-Demand Ride Pooling

**Authors**: *Xianjie Zhang, Pradeep Varakantham, Hao Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26710](https://doi.org/10.1609/aaai.v37i12.26710)

**Abstract**:

The popularity of on-demand ride pooling is owing to the benefits offered to customers (lower prices), taxi drivers (higher revenue), environment (lower carbon footprint due to fewer vehicles) and aggregation companies like Uber (higher revenue). To achieve these benefits, two key interlinked challenges have to be solved effectively: (a) pricing -- setting prices to customer requests for taxis; and (b) matching -- assignment of customers (that accepted the prices) to taxis/cars. Traditionally, both these challenges have been studied individually and using myopic approaches (considering only current requests), without considering the impact of current matching on addressing future requests. In this paper, we develop a novel framework that handles the pricing and matching problems together, while also considering the future impact of the pricing and matching decisions. In our experimental results on a real-world taxi dataset, we demonstrate that our framework can significantly improve revenue (up to 17% and on average 6.4%) in a sustainable manner by reducing the number of vehicles  (up to 14% and on average 10.6%) required to obtain a given fixed revenue and the overall distance travelled by vehicles (up to 11.1% and on average 3.7%). That is to say, we are able to provide an ideal win-win scenario for all stakeholders (customers, drivers, aggregator, environment) involved by obtaining higher revenue for customers, drivers, aggregator (ride pooling company) while being good for the environment (due to fewer number of vehicles on the road and lesser fuel consumed).

----

## [1641] A Crowd-AI Collaborative Duo Relational Graph Learning Framework towards Social Impact Aware Photo Classification

**Authors**: *Yang Zhang, Ziyi Kou, Lanyu Shang, Huimin Zeng, Zhenrui Yue, Dong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26711](https://doi.org/10.1609/aaai.v37i12.26711)

**Abstract**:

In artificial intelligence (AI), negative social impact (NSI) represents the negative effect on the society as a result of mistakes conducted by AI agents. While the photo classification problem has been widely studied in the AI community, the NSI made by photo misclassification is largely ignored due to the lack of quantitative measurements of the NSI and effective approaches to reduce it. In this paper, we focus on an NSI-aware photo classification problem where the goal is to develop a novel crowd-AI collaborative learning framework that leverages online crowd workers to quantitatively estimate and effectively reduce the NSI of misclassified photos. Our problem is motivated by the limitations of current NSI-aware photo classification approaches that either 1) cannot accurately estimate NSI because they simply model NSI as the semantic difference between true and misclassified categories or 2) require costly human annotations to estimate NSI of pairwise class categories. To address such limitations, we develop SocialCrowd, a crowdsourcing-based NSI-aware photo classification framework that explicitly reduces the NSI of photo misclassification by designing a duo relational NSI-aware graph with the NSI estimated by online crowd workers. The evaluation results on two large-scale image datasets show that SocialCrowd not only reduces the NSI of photo misclassification but also improves the classification accuracy on both datasets.

----

## [1642] People Taking Photos That Faces Never Share: Privacy Protection and Fairness Enhancement from Camera to User

**Authors**: *Junjie Zhu, Lin Gu, Xiaoxiao Wu, Zheng Li, Tatsuya Harada, Yingying Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26712](https://doi.org/10.1609/aaai.v37i12.26712)

**Abstract**:

The soaring number of personal mobile devices and public cameras poses a threat to fundamental human rights and ethical principles. For example, the stolen of private information such as face image by malicious third parties will lead to catastrophic consequences. By manipulating appearance of face in the image, most of existing protection algorithms are effective but irreversible. Here, we propose a practical and systematic solution to invertiblely protect face information in the full-process pipeline from camera to final users. Specifically, We design a novel lightweight Flow-based Face Encryption Method (FFEM) on the local embedded system privately connected to the camera,  minimizing the risk of  eavesdropping during data transmission. FFEM uses a flow-based face encoder to encode each face to a Gaussian distribution and encrypts the encoded face feature by random rotating the Gaussian distribution with the rotation matrix is as the password. While encrypted latent-variable face  images  are sent to users through public but less reliable channels, password will be protected through more secure channels through technologies such as asymmetric encryption, blockchain, or other sophisticated security schemes. User could select to decode an image with fake faces from the encrypted image on the public channel. Only trusted users are able to recover the original face  using the encrypted matrix transmitted in secure channel. More interestingly, by  tuning Gaussian ball in latent space, we could control the fairness of the replaced face on attributes such as gender and race. Extensive experiments demonstrate that our solution could protect privacy and enhance fairness with minimal effect on high-level downstream task.

----

## [1643] OpenMapFlow: A Library for Rapid Map Creation with Machine Learning and Remote Sensing Data

**Authors**: *Ivan Zvonkov, Gabriel Tseng, Catherine Nakalembe, Hannah Kerner*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26713](https://doi.org/10.1609/aaai.v37i12.26713)

**Abstract**:

The desired output for most real-world tasks using machine learning (ML) and remote sensing data is a set of dense predictions that form a predicted map for a geographic region. However, most prior work involving ML and remote sensing follows the traditional practice of reporting metrics on a set of independent, geographically-sparse samples and does not perform dense predictions. To reduce the labor of producing dense prediction maps, we present OpenMapFlow---an open-source python library for rapid map creation with ML and remote sensing data. OpenMapFlow provides 1) a data processing pipeline for users to create labeled datasets for any region, 2) code to train state-of-the-art deep learning models on custom or existing datasets, and 3) a cloud-based architecture to deploy models for efficient map prediction. We demonstrate the benefits of OpenMapFlow through experiments on three binary classification tasks: cropland, crop type (maize), and building mapping. We show that OpenMapFlow drastically reduces the time required for dense prediction compared to traditional workflows. We hope this library will stimulate novel research in areas such as domain shift, unsupervised learning, and societally-relevant applications and lessen the barrier to adopting research methods for real-world tasks.

----

## [1644] Formally Verified SAT-Based AI Planning

**Authors**: *Mohammad Abdulaziz, Friedrich Kurz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26714](https://doi.org/10.1609/aaai.v37i12.26714)

**Abstract**:

We present an executable formally verified SAT encoding of ground classical AI planning problems. We use the theorem prover Isabelle/HOL to perform the verification. We experimentally test the verified encoding and show that it can be used for reasonably sized standard planning benchmarks. We also use it as a reference to test a state-of-the-art SAT-based
planner, showing that it sometimes falsely claims that problems have no solutions of certain lengths.

----

## [1645] Shielding in Resource-Constrained Goal POMDPs

**Authors**: *Michal Ajdarów, Simon Brlej, Petr Novotný*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26715](https://doi.org/10.1609/aaai.v37i12.26715)

**Abstract**:

We consider partially observable Markov decision processes (POMDPs) modeling an agent that needs a supply of a certain resource (e.g., electricity stored in batteries) to operate correctly. The resource is consumed by the agent's actions and can be replenished only in certain states. The agent aims to minimize the expected cost of reaching some goal while preventing resource exhaustion, a problem we call resource-constrained goal optimization (RSGO). We take a two-step approach to the RSGO problem. First, using formal methods techniques, we design an algorithm computing a shield for a given scenario: a procedure that observes the agent and prevents it from using actions that might eventually lead to resource exhaustion. Second, we augment the POMCP heuristic search algorithm for POMDP planning with our shields to obtain an algorithm solving the RSGO problem. We implement our algorithm and present experiments showing its applicability to benchmarks from the literature.

----

## [1646] Implicit Bilevel Optimization: Differentiating through Bilevel Optimization Programming

**Authors**: *Francesco Alesiani*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26716](https://doi.org/10.1609/aaai.v37i12.26716)

**Abstract**:

Bilevel Optimization Programming is used to model complex and conflicting interactions between agents, for example in Robust AI or Privacy preserving AI. Integrating bilevel mathematical programming within deep learning is thus an essential objective for the Machine Learning community.  
Previously proposed approaches only consider single-level programming. In this paper, we extend existing single-level optimization programming approaches and thus propose Differentiating through Bilevel Optimization Programming (BiGrad) for end-to-end learning of models that use Bilevel Programming as a layer. 
BiGrad has wide applicability and can be used in modern machine learning frameworks. BiGrad is applicable to both continuous and combinatorial Bilevel optimization problems. We describe a class of gradient estimators for the combinatorial case which reduces the requirements in terms of computation complexity; for the case of the continuous variable, the gradient computation takes advantage of the push-back approach (i.e. vector-jacobian product) for an efficient implementation. Experiments show that the BiGrad successfully extends existing single-level approaches to Bilevel Programming.

----

## [1647] Query-Based Hard-Image Retrieval for Object Detection at Test Time

**Authors**: *Edward Ayers, Jonathan Sadeghi, John Redford, Romain Mueller, Puneet K. Dokania*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26717](https://doi.org/10.1609/aaai.v37i12.26717)

**Abstract**:

There is a longstanding interest in capturing the error behaviour of object detectors by finding images where their performance is likely to be unsatisfactory. In real-world applications such as autonomous driving, it is also crucial to characterise potential failures beyond simple requirements of detection performance. For example, a missed detection of a pedestrian close to an ego vehicle will generally require closer inspection than a missed detection of a car in the distance. The problem of predicting such potential failures at test time  has largely been overlooked in the literature and conventional approaches based on detection uncertainty fall short in that they are agnostic to such fine-grained characterisation of errors. In this work, we propose to reformulate the problem of finding "hard" images as a query-based hard image retrieval task, where queries are specific definitions of "hardness", and offer a simple and intuitive method that can solve this task for a large family of queries. Our method is entirely post-hoc, does not require ground-truth annotations, is independent of the choice of a detector, and relies on an efficient Monte Carlo estimation that uses a simple stochastic model in place of the ground-truth. We show experimentally that it can be applied successfully to a wide variety of queries for which it can reliably identify hard images for a given detector without any labelled data. We provide results on ranking and classification tasks using the widely used RetinaNet, Faster-RCNN, Mask-RCNN, and Cascade Mask-RCNN object detectors. The code for this project is available at https://github.com/fiveai/hardest.

----

## [1648] Probabilities Are Not Enough: Formal Controller Synthesis for Stochastic Dynamical Models with Epistemic Uncertainty

**Authors**: *Thom S. Badings, Licio Romao, Alessandro Abate, Nils Jansen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26718](https://doi.org/10.1609/aaai.v37i12.26718)

**Abstract**:

Capturing uncertainty in models of complex dynamical systems is crucial to designing safe controllers. Stochastic noise causes aleatoric uncertainty, whereas imprecise knowledge of model parameters leads to epistemic uncertainty. Several approaches use formal abstractions to synthesize policies that satisfy temporal specifications related to safety and reachability. However, the underlying models exclusively capture aleatoric but not epistemic uncertainty, and thus require that model parameters are known precisely. Our contribution to overcoming this restriction is a novel abstraction-based controller synthesis method for continuous-state models with stochastic noise and uncertain parameters. By sampling techniques and robust analysis, we capture both aleatoric and epistemic uncertainty, with a user-specified confidence level, in the transition probability intervals of a so-called interval Markov decision process (iMDP). We synthesize an optimal policy on this iMDP, which translates (with the specified confidence level) to a feedback controller for the continuous model with the same performance guarantees. Our experimental benchmarks confirm that accounting for epistemic uncertainty leads to controllers that are more robust against variations in parameter values.

----

## [1649] Accelerating Inverse Learning via Intelligent Localization with Exploratory Sampling

**Authors**: *Sirui Bi, Victor Fung, Jiaxin Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26719](https://doi.org/10.1609/aaai.v37i12.26719)

**Abstract**:

In the scope of "AI for Science", solving inverse problems is a longstanding challenge in materials and drug discovery, where the goal is to determine the hidden structures given a set of desirable properties. Deep generative models are recently proposed to solve inverse problems, but these are currently struggling in expensive forward operators, precisely localizing the exact solutions and fully exploring the parameter spaces without missing solutions. In this work, we propose a novel approach (called iPage) to accelerate the inverse learning process by leveraging probabilistic inference from deep invertible models and deterministic optimization via fast gradient descent.  Given a target property, the learned invertible model provides a posterior over the parameter space; we identify these posterior samples as an intelligent prior initialization which enables us to narrow down the search space. We then perform gradient descent to calibrate the inverse solutions within a local region. Meanwhile, a space-filling sampling is imposed on the latent space to better explore and capture all possible solutions. We evaluate our approach on three benchmark tasks and create two datasets of real-world applications from quantum chemistry and additive manufacturing and find our method achieves superior performance compared to several state-of-the-art baseline methods. The iPage code is available at https://github.com/jxzhangjhu/MatDesINNe.

----

## [1650] Attention-Conditioned Augmentations for Self-Supervised Anomaly Detection and Localization

**Authors**: *Behzad Bozorgtabar, Dwarikanath Mahapatra*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26720](https://doi.org/10.1609/aaai.v37i12.26720)

**Abstract**:

Self-supervised anomaly detection and localization are critical to real-world scenarios in which collecting anomalous samples and pixel-wise labeling is tedious or infeasible, even worse when a wide variety of unseen anomalies could surface at test time. Our approach involves a pretext task in the context of masked image modeling, where the goal is to impose agreement between cluster assignments obtained from the representation of an image view containing saliency-aware masked patches and the uncorrupted image view. We harness the self-attention map extracted from the transformer to mask non-salient image patches without destroying the crucial structure associated with the foreground object. Subsequently, the pre-trained model is fine-tuned to detect and localize simulated anomalies generated under the guidance of the transformer's self-attention map. We conducted extensive validation and ablations on the benchmark of industrial images and achieved superior performance against competing methods. We also show the adaptability of our method to the medical images of the chest X-rays benchmark.

----

## [1651] Robust-by-Design Classification via Unitary-Gradient Neural Networks

**Authors**: *Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio C. Buttazzo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26721](https://doi.org/10.1609/aaai.v37i12.26721)

**Abstract**:

The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of an SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.

----

## [1652] Ensemble-in-One: Ensemble Learning within Random Gated Networks for Enhanced Adversarial Robustness

**Authors**: *Yi Cai, Xuefei Ning, Huazhong Yang, Yu Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26722](https://doi.org/10.1609/aaai.v37i12.26722)

**Abstract**:

Adversarial attacks have threatened modern deep learning systems by crafting adversarial examples with small perturbations to fool the convolutional neural networks (CNNs). To alleviate that, ensemble training methods are proposed to facilitate better adversarial robustness by diversifying the vulnerabilities among the sub-models, simultaneously maintaining comparable natural accuracy as standard training. Previous practices also demonstrate that enlarging the ensemble can improve the robustness. However, conventional ensemble methods are with poor scalability, owing to the rapidly increasing complexity when containing more sub-models in the ensemble. Moreover, it is usually infeasible to train or deploy an ensemble with substantial sub-models, owing to the tight hardware resource budget and latency requirement. In this work, we propose Ensemble-in-One (EIO), a simple but effective method to efficiently enlarge the ensemble with a random gated network (RGN). EIO augments a candidate model by replacing the parametrized layers with multi-path random gated blocks (RGBs) to construct an RGN. The scalability is significantly boosted because the number of paths exponentially increases with the RGN depth. Then by learning from the vulnerabilities of numerous other paths within the RGN, every path obtains better adversarial robustness. Our experiments demonstrate that EIO consistently outperforms previous ensemble training methods with smaller computational overheads, simultaneously achieving better accuracy-robustness trade-offs than adversarial training methods under black-box transfer attacks. Code is available at https://github.com/cai-y13/Ensemble-in-One.git

----

## [1653] Safe Reinforcement Learning via Shielding under Partial Observability

**Authors**: *Steven Carr, Nils Jansen, Sebastian Junges, Ufuk Topcu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26723](https://doi.org/10.1609/aaai.v37i12.26723)

**Abstract**:

Safe exploration is a common problem in reinforcement learning (RL) that aims to prevent agents from making disastrous decisions while exploring their environment. A family of approaches to this problem assume domain knowledge in the form of a (partial) model of this environment to decide upon the safety of an action. A so-called shield forces the RL agent to select only safe actions. However, for adoption in various applications, one must look beyond enforcing safety and also ensure the applicability of RL with good performance. We extend the applicability of shields via tight integration with state-of-the-art deep RL, and provide an extensive, empirical study in challenging, sparse-reward environments under partial observability. We show that a carefully integrated shield ensures safety and can improve the convergence rate and final performance of RL agents. We furthermore show that a shield can be used to bootstrap state-of-the-art RL agents: they remain safe after initial learning in a shielded setting, allowing us to disable a potentially too conservative shield eventually.

----

## [1654] PowRL: A Reinforcement Learning Framework for Robust Management of Power Networks

**Authors**: *Anandsingh Chauhan, Mayank Baranwal, Ansuma Basumatary*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26724](https://doi.org/10.1609/aaai.v37i12.26724)

**Abstract**:

Power grids, across the world, play an important societal and economical role by providing uninterrupted, reliable and transient-free power to several industries, businesses and household consumers. With the advent of renewable power resources and EVs resulting into uncertain generation and highly dynamic load demands, it has become ever so important to ensure robust operation of power networks through suitable management of transient stability issues and localize the events of blackouts. In the light of ever increasing stress on the modern grid infrastructure and the grid operators, this paper presents a reinforcement learning (RL) framework, PowRL, to mitigate the effects of unexpected network events, as well as reliably maintain electricity everywhere on the network at all times. The PowRL leverages a novel heuristic for overload management, along with the RL-guided decision making on optimal topology selection to ensure that the grid is operated safely and reliably (with no overloads). PowRL is benchmarked on a variety of competition datasets hosted by the L2RPN (Learning to Run a Power Network). Even with its reduced action space, PowRL tops the leaderboard in the L2RPN NeurIPS 2020 challenge (Robustness track) at an aggregate level, while also being the top performing agent in the L2RPN WCCI 2020 challenge. Moreover, detailed analysis depicts state-of-the-art performances by the PowRL agent in some of the test scenarios.

----

## [1655] Two Wrongs Don't Make a Right: Combating Confirmation Bias in Learning with Label Noise

**Authors**: *Mingcai Chen, Hao Cheng, Yuntao Du, Ming Xu, Wenyu Jiang, Chongjun Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26725](https://doi.org/10.1609/aaai.v37i12.26725)

**Abstract**:

Noisy labels damage the performance of deep networks.  For robust learning, a prominent two-stage pipeline alternates between eliminating possible incorrect labels and semi-supervised training. However, discarding part of noisy labels could result in a loss of information, especially when the corruption has a dependency on data, e.g., class-dependent or instance-dependent. Moreover, from the training dynamics of a representative two-stage method DivideMix, we identify the domination of confirmation bias: pseudo-labels fail to correct a considerable amount of noisy labels, and consequently, the errors accumulate. To sufficiently exploit information from noisy labels and mitigate wrong corrections, we propose Robust Label Refurbishment (Robust LR)—a new hybrid method that integrates pseudo-labeling and confidence estimation techniques to refurbish noisy labels. We show that our method successfully alleviates the damage of both label noise and confirmation bias. As a result, it achieves state-of-the-art performance across datasets and noise types, namely CIFAR under different levels of synthetic noise and mini-WebVision and ANIMAL-10N with real-world noise.

----

## [1656] Testing the Channels of Convolutional Neural Networks

**Authors**: *Kang Choi, Donghyun Son, Younghoon Kim, Jiwon Seo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26726](https://doi.org/10.1609/aaai.v37i12.26726)

**Abstract**:

Neural networks have complex structures, and thus it is hard to understand their inner workings and ensure correctness. To understand and debug convolutional neural networks (CNNs) we propose techniques for testing the channels of CNNs. We design FtGAN, an extension to GAN, that can generate test data with varying the intensity (i.e., sum of the neurons) of a channel of a target CNN. We also proposed a channel selection algorithm to find representative channels for testing. To efficiently inspect the target CNN’s inference computations, we define unexpectedness score, which estimates how similar the inference computation of the test data is to that of the training data. We evaluated FtGAN with five public datasets and showed that our techniques successfully identify defective channels in five different CNN models.

----

## [1657] Feature-Space Bayesian Adversarial Learning Improved Malware Detector Robustness

**Authors**: *Bao Gia Doan, Shuiqiao Yang, Paul Montague, Olivier Y. de Vel, Tamas Abraham, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26727](https://doi.org/10.1609/aaai.v37i12.26727)

**Abstract**:

We present a new algorithm to train a robust malware detector. Malware is a prolific problem and malware detectors are a front-line defense. Modern detectors rely on machine learning algorithms. Now, the adversarial objective is to devise alterations to the malware code to decrease the chance of being detected whilst preserving the functionality and realism of the malware. Adversarial learning is effective in improving robustness but generating functional and realistic adversarial malware samples is non-trivial. Because: i) in contrast to tasks capable of using gradient-based feedback, adversarial learning in a domain without a differentiable mapping function from the problem space (malware code inputs) to the feature space is hard; and ii) it is difficult to ensure the adversarial malware is realistic and functional. 
This presents a challenge for developing scalable adversarial machine learning algorithms for large datasets at a production or commercial scale to realize robust malware detectors. We propose an alternative; perform adversarial learning in the feature space in contrast to the problem space. We prove the projection of perturbed, yet valid malware, in the problem space into feature space will always be a subset of adversarials generated in the feature space. Hence, by generating a robust network against feature-space adversarial examples, we inherently achieve robustness against problem-space adversarial examples. We formulate a Bayesian adversarial learning objective that captures the distribution of models for improved robustness. 
To explain the robustness of the Bayesian adversarial learning algorithm, we prove that our learning method bounds the difference between the adversarial risk and empirical risk and improves robustness. We show that Bayesian neural networks (BNNs) achieve state-of-the-art results; especially in the False Positive Rate (FPR) regime. Adversarially trained BNNs achieve state-of-the-art robustness. Notably, adversarially trained BNNs are robust against stronger attacks with larger attack budgets by a margin of up to 15% on a recent production-scale malware dataset of more than 20 million samples. Importantly, our efforts create a benchmark for future defenses in the malware domain.

----

## [1658] Correct-by-Construction Reinforcement Learning of Cardiac Pacemakers from Duration Calculus Requirements

**Authors**: *Kalyani Dole, Ashutosh Gupta, John Komp, Shankaranarayanan Krishna, Ashutosh Trivedi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26728](https://doi.org/10.1609/aaai.v37i12.26728)

**Abstract**:

As the complexity of pacemaker devices continues to grow, the importance of capturing its functional correctness requirement formally cannot be overestimated. The pacemaker system specification document by \emph{Boston Scientific} provides a widely accepted set of specifications for pacemakers. 
As these specifications are written in a natural language, they are not amenable for automated verification, synthesis, or reinforcement learning of pacemaker systems. This paper presents a formalization of these requirements for a dual-chamber pacemaker in \emph{duration calculus} (DC), a highly expressive real-time specification language.
The proposed formalization allows us to automatically translate pacemaker requirements into executable specifications as stopwatch automata, which can be used to enable simulation, monitoring, validation, verification and automatic synthesis of pacemaker systems. 
The cyclic nature of the pacemaker-heart closed-loop system results in DC requirements that compile to a decidable subclass of stopwatch automata. We present shield reinforcement learning (shield RL),  a shield synthesis based reinforcement learning algorithm, by automatically constructing safety envelopes from DC specifications.

----

## [1659] SafeLight: A Reinforcement Learning Method toward Collision-Free Traffic Signal Control

**Authors**: *Wenlu Du, Junyi Ye, Jingyi Gu, Jing Li, Hua Wei, Guiling Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26729](https://doi.org/10.1609/aaai.v37i12.26729)

**Abstract**:

Traffic signal control is safety-critical for our daily life. Roughly one-quarter of road accidents in the U.S. happen at intersections due to problematic signal timing, urging the development of safety-oriented intersection control. However, existing studies on adaptive traffic signal control using reinforcement learning technologies have focused mainly on minimizing traffic delay but neglecting the potential exposure to unsafe conditions. We, for the first time, incorporate road safety standards as enforcement to ensure the safety of existing reinforcement learning methods, aiming toward operating intersections with zero collisions. We have proposed a safety-enhanced residual reinforcement learning method (SafeLight) and employed multiple optimization techniques, such as multi-objective loss function and reward shaping for better knowledge integration. Extensive experiments are conducted using both synthetic and real-world benchmark datasets. Results show that our method can significantly reduce collisions while increasing traffic mobility.

----

## [1660] PatchNAS: Repairing DNNs in Deployment with Patched Network Architecture Search

**Authors**: *Yuchu Fang, Wenzhong Li, Yao Zeng, Yang Zheng, Zheng Hu, Sanglu Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26730](https://doi.org/10.1609/aaai.v37i12.26730)

**Abstract**:

Despite being widely deployed in safety-critical applications such as autonomous driving and health care, deep neural networks (DNNs) still suffer from non-negligible reliability issues. Numerous works had reported that DNNs were vulnerable to either natural environmental noises or man-made adversarial noises. How to repair DNNs in deployment with noisy samples is a crucial topic for the robustness of neural networks. While many network repairing methods based on data argumentation and weight adjustment have been proposed, they require retraining and redeploying the whole model, which causes high overhead and is infeasible for varying faulty cases on different deployment environments. In this paper, we propose a novel network repairing framework called PatchNAS from the architecture perspective, where we freeze the pretrained DNNs and introduce a small patch network to deal with failure samples at runtime. PatchNAS introduces a novel network instrumentation method to determine the faulty stage of the network structure given the collected failure samples. A small patch network structure is searched unsupervisedly using neural architecture search (NAS) technique with data samples from deployment environment. The patch network repairs the DNNs by correcting the output feature maps of the faulty stage, which helps to maintain network performance on normal samples and enhance robustness in noisy environments. Extensive experiments based on several DNNs across 15 types of natural noises show that the proposed PatchNAS outperforms the state-of-the-arts with significant performance improvement as well as much lower deployment overhead.

----

## [1661] Similarity Distribution Based Membership Inference Attack on Person Re-identification

**Authors**: *Junyao Gao, Xinyang Jiang, Huishuai Zhang, Yifan Yang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26731](https://doi.org/10.1609/aaai.v37i12.26731)

**Abstract**:

While person Re-identification (Re-ID) has progressed rapidly due to its wide real-world applications, it also causes severe risks of leaking personal information from training data. Thus, this paper focuses on quantifying this risk by membership inference (MI) attack. Most of the existing MI attack algorithms focus on classification models, while Re-ID follows a totally different training and inference paradigm. Re-ID is a fine-grained recognition task with complex feature embedding, and model outputs commonly used by existing MI like logits and losses are not accessible during inference. Since Re-ID focuses on modelling the relative relationship between image pairs instead of individual semantics, we conduct a formal and empirical analysis which validates that the distribution shift of the inter-sample similarity between training and test set is a critical criterion for Re-ID membership inference. As a result, we propose a novel membership inference attack method based on the inter-sample similarity distribution. Specifically, a set of anchor images are sampled to represent the similarity distribution conditioned on a target image, and a neural network with a novel anchor selection module is proposed to predict the membership of the target image. Our experiments validate the effectiveness of the proposed approach on both the Re-ID task and conventional classification task.

----

## [1662] Out-of-Distribution Detection Is Not All You Need

**Authors**: *Joris Guérin, Kevin Delmas, Raul Sena Ferreira, Jérémie Guiochet*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26732](https://doi.org/10.1609/aaai.v37i12.26732)

**Abstract**:

The usage of deep neural networks in safety-critical systems is limited by our ability to guarantee their correct behavior. Runtime monitors are components aiming to identify unsafe predictions and discard them before they can lead to catastrophic consequences. Several recent works on runtime monitoring have focused on out-of-distribution (OOD) detection, i.e., identifying inputs that are different from the training data. In this work, we argue that OOD detection is not a well-suited framework to design efficient runtime monitors and that it is more relevant to evaluate monitors based on their ability to discard incorrect predictions. We call this setting out-of-model-scope detection and discuss the conceptual differences with OOD. We also conduct extensive experiments on popular datasets from the literature to show that studying monitors in the OOD setting can be misleading: 1. very good OOD results can give a false impression of safety, 2. comparison under the OOD setting does not allow identifying the best monitor to detect errors. Finally, we also show that removing erroneous training data samples helps to train better monitors.

----

## [1663] Contrastive Self-Supervised Learning Leads to Higher Adversarial Susceptibility

**Authors**: *Rohit Gupta, Naveed Akhtar, Ajmal Mian, Mubarak Shah*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26733](https://doi.org/10.1609/aaai.v37i12.26733)

**Abstract**:

Contrastive self-supervised learning (CSL) has managed to match or surpass the performance of supervised learning in image and video classification. However, it is still largely unknown if the nature of the representations induced by the two learning paradigms is similar. We investigate this under the lens of adversarial robustness. Our analysis of the problem reveals that CSL has intrinsically higher sensitivity to perturbations over supervised learning. We identify the uniform distribution of data representation over a unit hypersphere in the CSL representation space as the key contributor to this phenomenon. We establish that this is a result of the presence of false negative pairs in the training process, which increases model sensitivity to input perturbations. Our finding is supported by extensive experiments for image and video classification using adversarial perturbations and other input corruptions. We devise a strategy to detect and remove false negative pairs that is simple, yet effective in improving model robustness with CSL training. We close up to 68% of the robustness gap between CSL and its supervised counterpart. Finally, we contribute to adversarial learning by incorporating our method in CSL. We demonstrate an average gain of about 5% over two different state-of-the-art methods in this domain.

----

## [1664] AutoCost: Evolving Intrinsic Cost for Zero-Violation Reinforcement Learning

**Authors**: *Tairan He, Weiye Zhao, Changliu Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26734](https://doi.org/10.1609/aaai.v37i12.26734)

**Abstract**:

Safety is a critical hurdle that limits the application of deep reinforcement learning to real-world control tasks. To this end, constrained reinforcement learning leverages cost functions to improve safety in constrained Markov decision process. However, constrained methods fail to achieve zero violation even when the cost limit is zero. This paper analyzes the reason for such failure, which suggests that a proper cost function plays an important role in constrained RL. Inspired by the analysis, we propose AutoCost, a simple yet effective framework that automatically searches for cost functions that help constrained RL to achieve zero-violation performance. We validate the proposed method and the searched cost function on the safety benchmark Safety Gym. We compare the performance of augmented agents that use our cost function to provide additive intrinsic costs to a Lagrangian-based policy learner and a constrained-optimization policy learner with baseline agents that use the same policy learners but with only extrinsic costs. Results show that the converged policies with intrinsic costs in all environments achieve zero constraint violation and comparable performance with baselines.

----

## [1665] Test Time Augmentation Meets Post-hoc Calibration: Uncertainty Quantification under Real-World Conditions

**Authors**: *Achim Hekler, Titus J. Brinker, Florian Buettner*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26735](https://doi.org/10.1609/aaai.v37i12.26735)

**Abstract**:

Communicating the predictive uncertainty of deep neural networks transparently and reliably is important in many safety-critical applications such as medicine. However, modern neural networks tend to be poorly calibrated, resulting in wrong predictions made with a high confidence. While existing post-hoc calibration methods like temperature scaling or isotonic regression yield strongly calibrated predictions in artificial experimental settings, their efficiency can significantly reduce in real-world applications, where scarcity of labeled data or domain drifts are commonly present. In this paper, we first investigate the impact of these characteristics on post-hoc calibration and introduce an easy-to-implement extension of common post-hoc calibration methods based on test time augmentation. In extensive experiments, we demonstrate that our approach results in substantially better calibration on various architectures. We demonstrate the robustness of our proposed approach on a real-world application for skin cancer classification and show that it facilitates safe decision-making under real-world uncertainties.

----

## [1666] Robust Training of Neural Networks against Bias Field Perturbations

**Authors**: *Patrick Henriksen, Alessio Lomuscio*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26736](https://doi.org/10.1609/aaai.v37i12.26736)

**Abstract**:

We introduce the problem of training neural networks such that they are robust against a class of smooth intensity perturbations modelled by bias fields. We first develop an approach towards this goal based on a state-of-the-art robust training method utilising Interval Bound Propagation (IBP). We analyse the resulting algorithm and observe that IBP often produces very loose bounds for bias field perturbations, which may be detrimental to training. We then propose an alternative approach based on Symbolic Interval Propagation (SIP), which usually results in significantly tighter bounds than IBP. We present ROBNET, a tool implementing these approaches for bias field robust training. In experiments networks trained with the SIP-based approach achieved up to 31% higher certified robustness while also maintaining a better accuracy than networks trained with the IBP approach.

----

## [1667] Redactor: A Data-Centric and Individualized Defense against Inference Attacks

**Authors**: *Geon Heo, Steven Euijong Whang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26737](https://doi.org/10.1609/aaai.v37i12.26737)

**Abstract**:

Information leakage is becoming a critical problem as various information becomes publicly available by mistake, and machine learning models train on that data to provide services. As a result, one's private information could easily be memorized by such trained models. Unfortunately, deleting information is out of the question as the data is already exposed to the Web or third-party platforms. Moreover, we cannot necessarily control the labeling process and the model trainings by other parties either. In this setting, we study the problem of targeted disinformation generation where the goal is to dilute the data and thus make a model safer and more robust against inference attacks on a specific target (e.g., a person's profile) by only inserting new data. Our method finds the closest points to the target in the input space that will be labeled as a different class. Since we cannot control the labeling process, we instead conservatively estimate the labels probabilistically by combining decision boundaries of multiple classifiers using data programming techniques. Our experiments show that a probabilistic decision boundary can be a good proxy for labelers, and that our approach is effective in defending against inference attacks and can scale to large data.

----

## [1668] Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting

**Authors**: *Pengyue Hou, Jie Han, Xingyu Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26738](https://doi.org/10.1609/aaai.v37i12.26738)

**Abstract**:

Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most effective methods. Theoretically, adversarial perturbation in untargeted attacks can be added along arbitrary directions and the predicted labels of untargeted attacks should be unpredictable. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs become virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair losses in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boosts model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks. The code of the proposed SPAT is published at https://github.com/puerrrr/Self-Paced-Adversarial-Training.

----

## [1669] CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models

**Authors**: *Akshita Jha, Chandan K. Reddy*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26739](https://doi.org/10.1609/aaai.v37i12.26739)

**Abstract**:

Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., primarily concerned with the human understanding of code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, Code Attack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. Code Attack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.

----

## [1670] Formalising the Robustness of Counterfactual Explanations for Neural Networks

**Authors**: *Junqi Jiang, Francesco Leofante, Antonio Rago, Francesca Toni*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26740](https://doi.org/10.1609/aaai.v37i12.26740)

**Abstract**:

The use of counterfactual explanations (CFXs) is an increasingly popular explanation strategy for machine learning models. However, recent studies have shown that these explanations may not be robust to changes in the underlying model (e.g., following retraining), which raises questions about their reliability in real-world applications. Existing attempts towards solving this problem are heuristic, and the robustness to model changes of the resulting CFXs is evaluated with only a small number of retrained models, failing to provide exhaustive guarantees. To remedy this, we propose ∆-robustness, the first notion to formally and deterministically assess the robustness (to model changes) of CFXs for neural networks. We introduce an abstraction framework based on interval neural networks 
to verify the ∆-robustness of CFXs against a possibly infinite set of changes to the model parameters, i.e., weights and biases. We then demonstrate the utility of this approach in two distinct ways. First, we  analyse the ∆-robustness of a number of CFX generation methods from the literature and show that they unanimously host significant deficiencies in this regard. Second, we demonstrate how embedding ∆-robustness within existing methods can provide CFXs which are provably robust.

----

## [1671] READ: Aggregating Reconstruction Error into Out-of-Distribution Detection

**Authors**: *Wenyu Jiang, Yuxin Ge, Hao Cheng, Mingcai Chen, Shuai Feng, Chongjun Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26741](https://doi.org/10.1609/aaai.v37i12.26741)

**Abstract**:

Detecting out-of-distribution (OOD) samples is crucial to the safe deployment of a classifier in the real world. However, deep neural networks are known to be overconfident for abnormal data. Existing works directly design score function by mining the inconsistency from classifier for in-distribution (ID) and OOD. In this paper, we further complement this inconsistency with reconstruction error, based on the assumption that an autoencoder trained on ID data cannot reconstruct OOD as well as ID. We propose a novel method, READ (Reconstruction Error Aggregated Detector), to unify inconsistencies from classifier and autoencoder. Specifically, the reconstruction error of raw pixels is transformed to latent space of classifier. We show that the transformed reconstruction error bridges the semantic gap and inherits detection performance from the original. Moreover, we propose an adjustment strategy to alleviate the overconfidence problem of autoencoder according to a fine-grained characterization of OOD data. Under two scenarios of pre-training and retraining, we respectively present two variants of our method, namely READ-MD (Mahalanobis Distance) only based on pre-trained classifier and READ-ED (Euclidean Distance) which retrains the classifier. Our methods do not require access to test time OOD data for fine-tuning hyperparameters. Finally, we demonstrate the effectiveness of the proposed methods through extensive comparisons with state-of-the-art OOD detection algorithms. On a CIFAR-10 pre-trained WideResNet, our method reduces the average FPR@95TPR by up to 9.8% compared with previous state-of-the-art.

----

## [1672] Sample-Dependent Adaptive Temperature Scaling for Improved Calibration

**Authors**: *Tom Joy, Francesco Pinto, Ser-Nam Lim, Philip H. S. Torr, Puneet K. Dokania*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26742](https://doi.org/10.1609/aaai.v37i12.26742)

**Abstract**:

It is now well known that neural networks can be wrong with high confidence in their predictions, leading to poor calibration. The most common post-hoc approach to compensate for this is to perform temperature scaling, which adjusts the confidences of the predictions on any input by scaling the logits by a fixed value. Whilst this approach typically improves the average calibration across the whole test dataset, this improvement typically reduces the individual confidences of the predictions irrespective of whether the classification of a given input is correct or incorrect. With this insight, we base our method on the observation that different samples contribute to the calibration error by varying amounts, with some needing to increase their confidence and others needing to decrease it. Therefore, for each input, we propose to predict a different temperature value, allowing us to adjust the mismatch between confidence and accuracy at a finer granularity. Our method is applied post-hoc, enabling it to be very fast with a negligible memory footprint and is applied to off-the-shelf pre-trained classifiers. We test our method on the ResNet50 and WideResNet28-10 architectures using the CIFAR10/100 and Tiny-ImageNet datasets, showing that producing per-data-point temperatures improves the expected calibration error across the whole test set.

----

## [1673] Heuristic Search in Dual Space for Constrained Fixed-Horizon POMDPs with Durative Actions

**Authors**: *Majid Khonji, Duoaa Khalifa*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26743](https://doi.org/10.1609/aaai.v37i12.26743)

**Abstract**:

The Partially Observable Markov Decision Process (POMDP) is widely used in probabilistic planning for stochastic domains. However, current extensions, such as constrained and chance-constrained POMDPs, have limitations in modeling real-world planning problems because they assume that all actions have a fixed duration. To address this issue, we propose a unified model that encompasses durative POMDP and its constrained extensions. To solve the durative POMDP and its constrained extensions, we first convert them into an Integer Linear Programming (ILP) formulation. This approach leverages existing solvers in the ILP literature and provides a foundation for solving these problems. We then introduce a heuristic search approach that prunes the search space, which is guided by solving successive partial ILP programs. Our empirical evaluation results show that our approach outperforms the current state-of-the-art fixed-horizon chance-constrained POMDP solver.

----

## [1674] Iteratively Enhanced Semidefinite Relaxations for Efficient Neural Network Verification

**Authors**: *Jianglin Lan, Yang Zheng, Alessio Lomuscio*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26744](https://doi.org/10.1609/aaai.v37i12.26744)

**Abstract**:

We propose an enhanced semidefinite program (SDP) relaxation to enable the tight and efficient verification of neural networks (NNs). The tightness improvement is achieved by introducing a nonlinear constraint to existing SDP relaxations previously proposed for NN verification. The efficiency of the proposal stems from the iterative nature of the proposed algorithm in that it solves the resulting non-convex SDP by recursively solving auxiliary convex layer-based SDP problems. We show formally that the solution generated by our algorithm is tighter than state-of-the-art SDP-based solutions for the problem. We also show that the solution sequence converges to the optimal solution of the non-convex enhanced SDP relaxation. The experimental results on standard benchmarks in the area show that our algorithm achieves the state-of-the-art performance whilst maintaining an acceptable computational cost.

----

## [1675] A Semidefinite Relaxation Based Branch-and-Bound Method for Tight Neural Network Verification

**Authors**: *Jianglin Lan, Benedikt Brückner, Alessio Lomuscio*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26745](https://doi.org/10.1609/aaai.v37i12.26745)

**Abstract**:

We introduce a novel method based on semidefinite program (SDP) for the tight and efficient verification of neural networks. The proposed SDP relaxation advances the present state of the art in SDP-based neural network verification by adding a set of linear constraints based on eigenvectors. We extend this novel SDP relaxation by combining it with a branch-and-bound method that can provably close the relaxation gap up to zero. We show formally that the proposed approach leads to a provably tighter solution than the present state of the art. We report experimental results showing that the proposed method outperforms baselines in terms of verified accuracy while retaining an acceptable computational overhead.

----

## [1676] Robust Image Steganography: Hiding Messages in Frequency Coefficients

**Authors**: *Yuhang Lan, Fei Shang, Jianhua Yang, Xiangui Kang, Enping Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26746](https://doi.org/10.1609/aaai.v37i12.26746)

**Abstract**:

Steganography is a technique that hides secret messages into a public multimedia object without raising suspicion from third parties. However, most existing works cannot provide good robustness against lossy JPEG compression while maintaining a relatively large embedding capacity. This paper presents an end-to-end robust steganography system based on the invertible neural network (INN). Instead of hiding in the spatial domain, our method directly hides secret messages into the discrete cosine transform (DCT) coefficients of the cover image, which significantly improves the robustness and anti-steganalysis security. A mutual information loss is first proposed to constrain the flow of information in INN. Besides, a two-way fusion module (TWFM) is implemented, utilizing spatial and DCT domain features as auxiliary information to facilitate message extraction. These two designs aid in recovering secret messages from the DCT coefficients losslessly. Experimental results demonstrate that our method yields significantly lower error rates than other existing hiding methods. For example, our method achieves reliable extraction with 0 error rate for 1 bit per pixel (bpp) embedding payload; and under the JPEG compression with quality factor QF=10, the error rate of our method is about 22% lower than the state-of-the-art robust image hiding methods, which demonstrates remarkable robustness against JPEG compression.

----

## [1677] Quantization-Aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks

**Authors**: *Mathias Lechner, Dorde Zikelic, Krishnendu Chatterjee, Thomas A. Henzinger, Daniela Rus*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26747](https://doi.org/10.1609/aaai.v37i12.26747)

**Abstract**:

We study the problem of training and certifying adversarially robust quantized neural networks (QNNs). Quantization is a technique for making neural networks more efficient by running them using low-bit integer arithmetic and is therefore commonly adopted in industry. Recent work has shown that floating-point neural networks that have been verified to be robust can become vulnerable to adversarial attacks after quantization, and certification of the quantized representation is necessary to guarantee robustness.
In this work, we present quantization-aware interval bound propagation (QA-IBP), a novel method for training robust QNNs.
Inspired by advances in robust learning of non-quantized networks, our training algorithm computes the gradient of an abstract representation of the actual network. Unlike existing approaches, our method can handle the discrete semantics of QNNs. 
Based on QA-IBP, we also develop a complete verification procedure for verifying the adversarial robustness of QNNs, which is guaranteed to terminate and produce a correct answer. Compared to existing approaches, the key advantage of our verification procedure is that it runs entirely on GPU or other accelerator devices. 
We demonstrate experimentally that our approach significantly outperforms existing methods and establish the new state-of-the-art for training and certifying the robustness of QNNs.

----

## [1678] Revisiting the Importance of Amplifying Bias for Debiasing

**Authors**: *Jungsoo Lee, Jeonghoon Park, Daeyoung Kim, Juyoung Lee, Edward Choi, Jaegul Choo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26748](https://doi.org/10.1609/aaai.v37i12.26748)

**Abstract**:

In image classification, debiasing aims to train a classifier to be less susceptible to dataset bias, the strong correlation between peripheral attributes of data samples and a target class. For example, even if the frog class in the dataset mainly consists of frog images with a swamp background (i.e., bias aligned samples), a debiased classifier should be able to correctly classify a frog at a beach (i.e., bias conflicting samples). Recent debiasing approaches commonly use two components for debiasing, a biased model fB and a debiased model fD. fB is trained to focus on bias aligned samples (i.e., overfitted to the bias) while fD is mainly trained with bias conflicting samples by concentrating on samples which fB fails to learn, leading fD to be less susceptible to the dataset bias. While the state of the art debiasing techniques have aimed to better train fD, we focus on training fB, an overlooked component until now. Our empirical analysis reveals that removing the bias conflicting samples from the training set for fB is important for improving the debiasing performance of fD. This is due to the fact that the bias conflicting samples work as noisy samples for amplifying the bias for fB since those samples do not include the bias attribute. To this end, we propose a simple yet effective data sample selection method which removes the bias conflicting samples to construct a bias amplified dataset for training fB. Our data sample selection method can be directly applied to existing reweighting based debiasing approaches, obtaining consistent performance boost and achieving the state of the art performance on both synthetic and real-world datasets.

----

## [1679] WAT: Improve the Worst-Class Robustness in Adversarial Training

**Authors**: *Boqi Li, Weiwei Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26749](https://doi.org/10.1609/aaai.v37i12.26749)

**Abstract**:

Deep Neural Networks (DNN) have been shown to be vulnerable to adversarial examples. Adversarial training (AT) is a popular and effective strategy to defend against adversarial attacks. Recent works have shown that a robust model well-trained by AT exhibits a remarkable robustness disparity among classes, and propose various methods to obtain consistent robust accuracy across classes. Unfortunately, these methods sacrifice a good deal of the average robust accuracy. Accordingly, this paper proposes a novel framework of worst-class adversarial training and leverages no-regret dynamics to solve this problem. Our goal is to obtain a classifier with great performance on worst-class and sacrifice just a little average robust accuracy at the same time. We then rigorously analyze the theoretical properties of our proposed algorithm, and the generalization error bound in terms of the worst-class robust risk. Furthermore, we propose a measurement to evaluate the proposed method in terms of both the average and worst-class accuracies. Experiments on various datasets and networks show that our proposed method outperforms the state-of-the-art approaches.

----

## [1680] PLMmark: A Secure and Robust Black-Box Watermarking Framework for Pre-trained Language Models

**Authors**: *Peixuan Li, Pengzhou Cheng, Fangqi Li, Wei Du, Haodong Zhao, Gongshen Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26750](https://doi.org/10.1609/aaai.v37i12.26750)

**Abstract**:

The huge training overhead, considerable commercial value, and various potential security risks make it urgent to protect the intellectual property (IP) of Deep Neural Networks (DNNs). DNN watermarking has become a plausible method to meet this need. However, most of the existing watermarking schemes focus on image classification tasks. The schemes designed for the textual domain lack security and reliability. Moreover, how to protect the IP of widely-used pre-trained language models (PLMs) remains a blank. 
To fill these gaps, we propose PLMmark, the first secure and robust black-box watermarking framework for PLMs. It consists of three phases: (1) In order to generate watermarks that contain owners’ identity information, we propose a novel encoding method to establish a strong link between a digital signature and trigger words by leveraging the original vocabulary tables of PLMs. Combining this with public key cryptography ensures the security of our scheme. (2) To embed robust, task-agnostic, and highly transferable watermarks in PLMs, we introduce a supervised contrastive loss to deviate the output representations of trigger sets from that of clean samples. In this way, the watermarked models will respond to the trigger sets anomaly and thus can identify the ownership. (3) To make the model ownership verification results reliable, we perform double verification, which guarantees the unforgeability of ownership. Extensive experiments on text classification tasks demonstrate that the embedded watermark can transfer to all the downstream tasks and can be effectively extracted and verified. The watermarking scheme is robust to watermark removing attacks (fine-pruning and re-initializing) and is secure enough to resist forgery attacks.

----

## [1681] Rethinking Label Refurbishment: Model Robustness under Label Noise

**Authors**: *Yangdi Lu, Zhiwei Xu, Wenbo He*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26751](https://doi.org/10.1609/aaai.v37i12.26751)

**Abstract**:

A family of methods that generate soft labels by mixing the hard labels with a certain distribution, namely label refurbishment, are widely used to train deep neural networks. However, some of these methods are still poorly understood in the presence of label noise. In this paper, we revisit four label refurbishment methods and reveal the strong connection between them. We find that they affect the neural network models in different manners. Two of them smooth the estimated posterior for regularization effects, and the other two force the model to produce high-confidence predictions. We conduct extensive experiments to evaluate related methods and observe that both effects improve the model generalization under label noise. Furthermore, we theoretically show that both effects lead to generalization guarantees on the clean distribution despite being trained with noisy labels.

----

## [1682] A Holistic Approach to Undesired Content Detection in the Real World

**Authors**: *Todor Markov, Chong Zhang, Sandhini Agarwal, Florentine Eloundou Nekoul, Theodore Lee, Steven Adler, Angela Jiang, Lilian Weng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26752](https://doi.org/10.1609/aaai.v37i12.26752)

**Abstract**:

We present a holistic approach to building a robust and useful natural language classification system for real-world content moderation. The success of such a system relies on a chain of carefully designed and executed steps, including the design of content taxonomies and labeling instructions, data quality control, an active learning pipeline to capture rare events, and a variety of methods to make the model robust and to avoid overfitting. Our moderation system is trained to detect a broad set of categories of undesired content, including sexual content, hateful content, violence, self-harm, and harassment. This approach generalizes to a wide range of different content taxonomies and can be used to create high-quality content classifiers that outperform off-the-shelf models.

----

## [1683] A Risk-Sensitive Approach to Policy Optimization

**Authors**: *Jared Markowitz, Ryan W. Gardner, Ashley J. Llorens, Raman Arora, I-Jeng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26753](https://doi.org/10.1609/aaai.v37i12.26753)

**Abstract**:

Standard deep reinforcement learning (DRL) aims to maximize expected reward, considering collected experiences equally in formulating a policy. This differs from human decision-making, where gains and losses are valued differently and outlying outcomes are given increased consideration.  It also fails to capitalize on opportunities to improve safety and/or performance through the incorporation of distributional context. Several approaches to distributional DRL have been investigated, with one popular strategy being to evaluate the projected distribution of returns for possible actions.  We propose a more direct approach whereby risk-sensitive objectives, specified in terms of the cumulative distribution function (CDF) of the distribution of full-episode rewards, are optimized. This approach allows for outcomes to be weighed based on relative quality, can be used for both continuous and discrete action spaces, and may naturally be applied in both constrained and unconstrained settings.  We show how to compute an asymptotically consistent estimate of the policy gradient for a broad class of risk-sensitive objectives via sampling, subsequently incorporating variance reduction and regularization measures to facilitate effective on-policy learning.  We then demonstrate that the use of moderately "pessimistic" risk profiles, which emphasize scenarios where the agent performs poorly, leads to enhanced exploration and a continual focus on addressing deficiencies.  We test the approach using different risk profiles in six OpenAI Safety Gym environments, comparing to state of the art on-policy methods.  Without cost constraints, we find that pessimistic risk profiles can be used to reduce cost while improving total reward accumulation.  With cost constraints, they are seen to provide higher positive rewards than risk-neutral approaches at the prescribed allowable cost.

----

## [1684] Anonymization for Skeleton Action Recognition

**Authors**: *Saemi Moon, Myeonghyeon Kim, Zhenyue Qin, Yang Liu, Dongwoo Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26754](https://doi.org/10.1609/aaai.v37i12.26754)

**Abstract**:

Skeleton-based action recognition attracts practitioners and researchers due to the lightweight, compact nature of datasets. Compared with RGB-video-based action recognition, skeleton-based action recognition is a safer way to protect the privacy of subjects while having competitive recognition performance. However, due to improvements in skeleton recognition algorithms as well as motion and depth sensors, more details of motion characteristics can be preserved in the skeleton dataset, leading to potential privacy leakage. We first train classifiers to categorize private information from skeleton trajectories to investigate the potential privacy leakage from skeleton datasets. Our preliminary experiments show that the gender classifier achieves 87% accuracy on average, and the re-identification classifier achieves 80% accuracy on average with three baseline models: Shift-GCN, MS-G3D, and 2s-AGCN. We propose an anonymization framework based on adversarial learning to protect potential privacy leakage from the skeleton dataset. Experimental results show that an anonymized dataset can reduce the risk of privacy leakage while having marginal effects on action recognition performance even with simple anonymizer architectures. The code used in our experiments is available at https://github.com/ml-postech/Skeleton-anonymization/

----

## [1685] Monitoring Model Deterioration with Explainable Uncertainty Estimation via Non-parametric Bootstrap

**Authors**: *Carlos Mougan, Dan Saattrup Nielsen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26755](https://doi.org/10.1609/aaai.v37i12.26755)

**Abstract**:

Monitoring machine learning models once they are deployed
is challenging. It is even more challenging to decide when
to retrain models in real-case scenarios when labeled data is
beyond reach, and monitoring performance metrics becomes
unfeasible. In this work, we use non-parametric bootstrapped
uncertainty estimates and SHAP values to provide explainable
uncertainty estimation as a technique that aims to monitor
the deterioration of machine learning models in deployment
environments, as well as determine the source of model deteri-
oration when target labels are not available. Classical methods
are purely aimed at detecting distribution shift, which can lead
to false positives in the sense that the model has not deterio-
rated despite a shift in the data distribution. To estimate model
uncertainty we construct prediction intervals using a novel
bootstrap method, which improves previous state-of-the-art
work. We show that both our model deterioration detection
system as well as our uncertainty estimation method achieve
better performance than the current state-of-the-art. Finally,
we use explainable AI techniques to gain an understanding
of the drivers of model deterioration. We release an open
source Python package, doubt, which implements our pro-
posed methods, as well as the code used to reproduce our
experiments.

----

## [1686] Certified Policy Smoothing for Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Gaojie Jin, Qiang Ni*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26756](https://doi.org/10.1609/aaai.v37i12.26756)

**Abstract**:

Cooperative multi-agent reinforcement learning (c-MARL) is widely applied in safety-critical scenarios, thus the analysis of robustness for c-MARL models is profoundly important. However, robustness certification for c-MARLs has not yet been explored in the community. In this paper, we propose a novel certification method, which is the first work to leverage a scalable approach for c-MARLs to determine actions with guaranteed certified bounds. c-MARL certification poses two key challenges compared to single-agent systems:  (i) the accumulated uncertainty as the number of agents increases; (ii) the potential lack of impact when changing the action of a single agent into a global team reward. These challenges prevent us from directly using existing algorithms. Hence, we employ the false discovery rate (FDR) controlling procedure considering the importance of each agent to certify per-state robustness. We further propose a tree-search-based algorithm to find a lower bound of the global reward under the minimal certified perturbation. As our method is general, it can also be applied in a single-agent environment. We empirically show that our certification bounds are much tighter than those of state-of-the-art RL certification solutions. We also evaluate our method on two popular c-MARL algorithms: QMIX and VDN, under two different environments, with two and four agents. The experimental results show that our method can certify the robustness of all c-MARL models in various environments. Our tool CertifyCMARL is available at https://github.com/TrustAI/CertifyCMARL.

----

## [1687] Constrained Reinforcement Learning in Hard Exploration Problems

**Authors**: *Pathmanathan Pankayaraj, Pradeep Varakantham*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26757](https://doi.org/10.1609/aaai.v37i12.26757)

**Abstract**:

One approach to guaranteeing safety in Reinforcement Learning is through cost constraints that are dependent on the policy. Recent works in constrained RL have developed methods that ensure  constraints are enforced even at learning time while maximizing the overall value of the policy. Unfortunately, as demonstrated in our experimental results, such approaches do not perform well on complex multi-level tasks, with longer episode lengths or sparse rewards. To that end, we propose a scalable hierarchical approach for constrained RL problems that employs backward cost value functions in the context of task hierarchy and a novel intrinsic reward function in lower levels of the hierarchy to enable cost constraint enforcement. One of our key contributions is in proving that backward value functions are theoretically viable even when there are multiple levels of decision making. We also show that our new approach, referred to as Hierarchically Limited consTraint Enforcement (HiLiTE) significantly improves on state of the art Constrained RL approaches for many  benchmark problems from literature. We further demonstrate that this performance (on value and constraint enforcement) clearly outperforms existing best approaches for constrained RL and hierarchical RL.

----

## [1688] Defending from Physically-Realizable Adversarial Attacks through Internal Over-Activation Analysis

**Authors**: *Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio C. Buttazzo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26758](https://doi.org/10.1609/aaai.v37i12.26758)

**Abstract**:

This work presents Z-Mask, an effective and deterministic strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks.
The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism.
The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches in the real world.
The results confirm that Z-Mask outperforms the state-of-the-art methods in terms of detection accuracy and overall performance of the networks under attack.
Furthermore, Z-Mask preserves its robustness against defense-aware attacks, making it suitable for safe and secure AI applications.

----

## [1689] Formally Verified Solution Methods for Markov Decision Processes

**Authors**: *Maximilian Schäffeler, Mohammad Abdulaziz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26759](https://doi.org/10.1609/aaai.v37i12.26759)

**Abstract**:

We formally verify executable algorithms for solving Markov decision processes (MDPs) in the interactive theorem prover Isabelle/HOL. We build on existing formalizations of probability theory to analyze the expected total reward criterion on finite and infinite-horizon problems. Our developments formalize the Bellman equation and give conditions under which optimal policies exist. Based on this analysis, we verify dynamic programming algorithms to solve tabular MDPs. We evaluate the formally verified implementations experimentally on standard problems, compare them with state-of-the-art systems, and show that they are practical.

----

## [1690] Improving Training and Inference of Face Recognition Models via Random Temperature Scaling

**Authors**: *Lei Shang, Mouxiao Huang, Wu Shi, Yuchen Liu, Yang Liu, Wang Steven, Baigui Sun, Xuansong Xie, Yu Qiao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26760](https://doi.org/10.1609/aaai.v37i12.26760)

**Abstract**:

Data uncertainty is commonly observed in the images for face recognition (FR). However, deep learning algorithms often make predictions with high confidence even for uncertain or irrelevant inputs. Intuitively, FR algorithms can benefit from both the estimation of uncertainty and the detection of out-of-distribution (OOD) samples. Taking a probabilistic view of the current classification model, the temperature scalar is exactly the scale of uncertainty noise implicitly added in the softmax function. Meanwhile, the uncertainty of images in a dataset should follow a prior distribution. Based on the observation, a unified framework for uncertainty modeling and FR, Random Temperature Scaling (RTS), is proposed to learn a reliable FR algorithm. The benefits of RTS are two-fold. (1) In the training phase, it can adjust the learning strength of clean and noisy samples for stability and accuracy. (2) In the test phase, it can provide a score of confidence to detect uncertain, low-quality and even OOD samples, without training on extra labels. Extensive experiments on FR benchmarks demonstrate that the magnitude of variance in RTS, which serves as an OOD detection metric, is closely related to the uncertainty of the input image. RTS can achieve top performance on both the FR and OOD detection tasks. Moreover, the model trained with RTS can perform robustly on datasets with noise. The proposed module is light-weight and only adds negligible computation cost to the model.

----

## [1691] Task and Model Agnostic Adversarial Attack on Graph Neural Networks

**Authors**: *Kartik Sharma, Samidha Verma, Sourav Medya, Arnab Bhattacharya, Sayan Ranu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26761](https://doi.org/10.1609/aaai.v37i12.26761)

**Abstract**:

Adversarial attacks on Graph Neural Networks (GNNs) reveal their security vulnerabilities, limiting their adoption in safety-critical applications. However, existing attack strategies rely on the knowledge of either the GNN model being used or the predictive task being attacked. Is this knowledge necessary? For example, a graph may be used for multiple downstream tasks unknown to a practical attacker. It is thus important to test the vulnerability of GNNs to adversarial perturbations in a model and task-agnostic setting. In this work, we study this problem and show that Gnns remain vulnerable even when the downstream task and model are unknown. The proposed algorithm, TANDIS (Targeted Attack via Neighborhood DIStortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, TANDIS designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep Q-learning. Extensive experiments on real datasets show that, on average, TANDIS is up to 50% more effective than state-of-the-art techniques, while being more than 1000 times faster.

----

## [1692] Robust Sequence Networked Submodular Maximization

**Authors**: *Qihao Shi, Bingyang Fu, Can Wang, Jiawei Chen, Sheng Zhou, Yan Feng, Chun Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26762](https://doi.org/10.1609/aaai.v37i12.26762)

**Abstract**:

In this paper, we study the Robust optimization for sequence Networked submodular maximization (RoseNets) problem. We interweave the robust optimization  with the sequence networked submodular maximization. The elements are connected by a directed acyclic graph and the objective function is not submodular on the elements but on the edges in the graph. Under such networked submodular scenario, the impact of removing an element from a sequence depends both on its position in the sequence and in the network. This makes the existing robust algorithms inapplicable and calls for new robust algorithms. In this paper, we take the first step to study the RoseNets problem. We design a robust greedy algorithms, which is robust against the removal of an arbitrary subset of the selected elements. The approximation ratio of the algorithm depends both on the number of the removed elements and the network topology. We further conduct experiments on real applications of recommendation and link prediction. The experimental results demonstrate the effectiveness of the proposed algorithm.

----

## [1693] Safe Policy Improvement for POMDPs via Finite-State Controllers

**Authors**: *Thiago D. Simão, Marnix Suilen, Nils Jansen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26763](https://doi.org/10.1609/aaai.v37i12.26763)

**Abstract**:

We study safe policy improvement (SPI) for partially observable Markov decision processes (POMDPs). SPI is an offline reinforcement learning (RL) problem that assumes access to (1) historical data about an environment, and (2) the so-called behavior policy that previously generated this data by interacting with the environment. SPI methods neither require access to a model nor the environment itself, and aim to reliably improve upon the behavior policy in an offline manner. Existing methods make the strong assumption that the environment is fully observable. In our novel approach to the SPI problem for POMDPs, we assume that a finite-state controller (FSC) represents the behavior policy and that finite memory is sufficient to derive optimal policies. This assumption allows us to map the POMDP to a finite-state fully observable MDP, the history MDP. We estimate this MDP by combining the historical data and the memory of the FSC, and compute an improved policy using an off-the-shelf SPI algorithm. The underlying SPI method constrains the policy space according to the available data, such that the newly computed policy only differs from the behavior policy when sufficient data is available. We show that this new policy, converted into a new FSC for the (unknown) POMDP, outperforms the behavior policy with high probability. Experimental results on several well-established benchmarks show the applicability of the approach, even in cases where finite memory is not sufficient.

----

## [1694] STL-Based Synthesis of Feedback Controllers Using Reinforcement Learning

**Authors**: *Nikhil Kumar Singh, Indranil Saha*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26764](https://doi.org/10.1609/aaai.v37i12.26764)

**Abstract**:

Deep Reinforcement Learning (DRL) has the potential to be used for synthesizing feedback controllers (agents) for various complex systems with unknown dynamics. These systems are expected to satisfy diverse safety and liveness properties best captured using temporal logic. In RL, the reward function plays a crucial role in specifying the desired behaviour of these agents. However, the problem of designing the reward function for an RL agent to satisfy complex temporal logic specifications has received limited attention in the literature. To address this, we provide a systematic way of generating rewards in real-time by using the quantitative semantics of Signal Temporal Logic (STL), a widely used temporal logic to specify the behaviour of cyber-physical systems. We propose a new quantitative semantics for STL having several desirable properties, making it suitable for reward generation. We evaluate our STL-based reinforcement learning mechanism on several complex continuous control benchmarks and compare our STL semantics with those available in the literature in terms of their efficacy in synthesizing the controller agent. Experimental results establish our new semantics to be the most suitable for synthesizing feedback controllers for complex continuous dynamical systems through reinforcement learning.

----

## [1695] Understanding and Enhancing Robustness of Concept-Based Models

**Authors**: *Sanchit Sinha, Mengdi Huai, Jianhui Sun, Aidong Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26765](https://doi.org/10.1609/aaai.v37i12.26765)

**Abstract**:

Rising usage of deep neural networks to perform decision making in critical applications like medical diagnosis and fi- nancial analysis have raised concerns regarding their reliability and trustworthiness. As automated systems become more mainstream, it is important their decisions be transparent, reliable and understandable by humans for better trust and confidence. To this effect, concept-based models such as Concept Bottleneck Models (CBMs) and Self-Explaining Neural Networks (SENN) have been proposed which constrain the latent space of a model to represent high level concepts easily understood by domain experts in the field. Although concept-based models promise a good approach to both increasing explainability and reliability, it is yet to be shown if they demonstrate robustness and output consistent concepts under systematic perturbations to their inputs. To better understand performance of concept-based models on curated malicious samples, in this paper, we aim to study their robustness to adversarial perturbations, which are also known as the imperceptible changes to the input data that are crafted by an attacker to fool a well-learned concept-based model. Specifically, we first propose and analyze different malicious attacks to evaluate the security vulnerability of concept based models. Subsequently, we propose a potential general adversarial training-based defense mechanism to increase robustness of these systems to the proposed malicious attacks. Extensive experiments on one synthetic and two real-world datasets demonstrate the effectiveness of the proposed attacks and the defense approach. An appendix of the paper with more comprehensive results can also be viewed at https://arxiv.org/abs/2211.16080.

----

## [1696] Misspecification in Inverse Reinforcement Learning

**Authors**: *Joar Skalse, Alessandro Abate*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26766](https://doi.org/10.1609/aaai.v37i12.26766)

**Abstract**:

The aim of Inverse Reinforcement Learning (IRL) is to infer a reward function R from a policy pi. To do this, we need a model of how pi relates to R. In the current literature, the most common models are optimality, Boltzmann rationality, and causal entropy maximisation. One of the primary motivations behind IRL is to infer human preferences from human behaviour. However, the true relationship between human preferences and human behaviour is much more complex than any of the models currently used in IRL. This means that they are misspecified, which raises the worry that they might lead to unsound inferences if applied to real-world data. In this paper, we provide a mathematical analysis of how robust different IRL models are to misspecification, and answer precisely how the demonstrator policy may differ from each of the standard models before that model leads to faulty inferences about the reward function R. We also introduce a framework for reasoning about misspecification in IRL, together with formal tools that can be used to easily derive the misspecification robustness of new IRL models.

----

## [1697] Planning and Learning for Non-markovian Negative Side Effects Using Finite State Controllers

**Authors**: *Aishwarya Srivastava, Sandhya Saisubramanian, Praveen Paruchuri, Akshat Kumar, Shlomo Zilberstein*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26767](https://doi.org/10.1609/aaai.v37i12.26767)

**Abstract**:

Autonomous systems are often deployed in the open world where it is hard to obtain complete specifications of objectives and constraints. Operating based on an incomplete model can produce negative side effects (NSEs), which affect the safety and reliability of the system. We focus on mitigating NSEs in environments modeled as Markov decision processes (MDPs). First, we learn a model of NSEs using observed data that contains state-action trajectories and severity of associated NSEs. Unlike previous works that associate NSEs with state-action pairs, our framework associates NSEs with entire trajectories, which is more general and captures non-Markovian dependence on states and actions. Second, we learn finite state controllers (FSCs) that predict NSE severity for a given trajectory and generalize well to unseen data. Finally, we develop a constrained MDP model that uses information from the underlying MDP and the learned FSC for planning while avoiding NSEs. Our empirical evaluation demonstrates the effectiveness of our approach in learning and mitigating Markovian and non-Markovian NSEs.

----

## [1698] Toward Robust Uncertainty Estimation with Random Activation Functions

**Authors**: *Yana Stoyanova, Soroush Ghandi, Maryam Tavakol*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26768](https://doi.org/10.1609/aaai.v37i12.26768)

**Abstract**:

Deep neural networks are in the limelight of machine learning with their excellent performance in many data-driven applications. However, they can lead to inaccurate predictions when queried in out-of-distribution data points, which can have detrimental effects especially in sensitive domains, such as healthcare and transportation, where erroneous predictions can be very costly and/or dangerous. Subsequently, quantifying the uncertainty of the output of a neural network is often leveraged to evaluate the confidence of its predictions, and ensemble models have proved to be effective in measuring the uncertainty by utilizing the variance of predictions over a pool of models. In this paper, we propose a novel approach for uncertainty quantification via ensembles, called Random Activation Functions (RAFs) Ensemble, that aims at improving the ensemble diversity toward a more robust estimation, by accommodating each neural network with a different (random) activation function. Extensive empirical study demonstrates that RAFs Ensemble outperforms state-of-the-art ensemble uncertainty quantification methods on both synthetic and real-world datasets in a series of regression tasks.

----

## [1699] Improving Robust Fariness via Balance Adversarial Training

**Authors**: *Chunyu Sun, Chenye Xu, Chengyuan Yao, Siyuan Liang, Yichao Wu, Ding Liang, Xianglong Liu, Aishan Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26769](https://doi.org/10.1609/aaai.v37i12.26769)

**Abstract**:

Adversarial training (AT) methods are effective against adversarial attacks, yet they introduce severe disparity of accuracy and robustness between different classes, known as the robust fairness problem. Previously proposed Fair Robust Learning (FRL) adaptively reweights different classes to improve fairness. However, the performance of the better-performed classes decreases, leading to a strong performance drop. In this paper, we observed two unfair phenomena during adversarial training: different difficulties in generating adversarial examples from each class (source-class fairness) and disparate target class tendencies when generating adversarial examples (target-class fairness). From the observations, we propose Balance Adversarial Training (BAT) to address the robust fairness problem. Regarding source-class fairness, we adjust the attack strength and difficulties of each class to generate samples near the decision boundary for easier and fairer model learning; considering target-class fairness, by introducing a uniform distribution constraint, we encourage the adversarial example generation process for each class with a fair tendency. Extensive experiments conducted on multiple datasets (CIFAR-10, CIFAR-100, and ImageNette) demonstrate that our BAT can significantly outperform other baselines in mitigating the robust fairness problem (+5-10\% on the worst class accuracy)(Our codes can be found at https://github.com/silvercherry/Improving-Robust-Fairness-via-Balance-Adversarial-Training).

----

## [1700] DPAUC: Differentially Private AUC Computation in Federated Learning

**Authors**: *Jiankai Sun, Xin Yang, Yuanshun Yao, Junyuan Xie, Di Wu, Chong Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26770](https://doi.org/10.1609/aaai.v37i12.26770)

**Abstract**:

Federated learning (FL) has gained significant attention recently as a privacy-enhancing tool to jointly train a machine learning model by multiple participants. 
The prior work on FL has mostly studied how to protect label privacy during model training. However, model evaluation in FL might also lead to the potential leakage of private label information.
In this work, we propose an evaluation algorithm that can accurately compute the widely used AUC (area under the curve) metric when using the label differential privacy (DP) in FL. Through extensive experiments, we show our algorithms can compute accurate AUCs compared to the ground truth. The code is available at https://github.com/bytedance/fedlearner/tree/master/example/privacy/DPAUC

----

## [1701] Conflicting Interactions among Protection Mechanisms for Machine Learning Models

**Authors**: *Sebastian Szyller, N. Asokan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26771](https://doi.org/10.1609/aaai.v37i12.26771)

**Abstract**:

Nowadays, systems based on machine learning (ML) are widely used in different domains.
Given their popularity, ML models have become targets for various attacks.
As a result, research at the intersection of security/privacy and ML has flourished. Typically such work has focused on individual types of security/privacy concerns and mitigations thereof.

However, in real-life deployments, an ML model will need to be protected against several concerns simultaneously.
A protection mechanism optimal for a specific security or privacy concern may interact negatively with mechanisms intended to address other concerns. Despite its practical relevance, the potential for such conflicts has not been studied adequately.

In this work, we first provide a framework for analyzing such conflicting interactions.
We then focus on systematically analyzing pairwise interactions between protection mechanisms for one concern, model and data ownership verification, with two other classes of ML protection mechanisms: differentially private training, and robustness against model evasion.
We find that several pairwise interactions result in conflicts.

We also explore potential approaches for avoiding such conflicts. First, we study the effect of hyperparameter relaxations, finding that there is no sweet spot balancing the performance of both protection mechanisms.
Second, we explore whether modifying one type of protection mechanism (ownership verification) so as to decouple it from factors that may be impacted by a conflicting mechanism (differentially private training or robustness to model evasion) can avoid conflict.
We show that this approach can indeed avoid the conflict between ownership verification mechanisms when combined with differentially private training, but has no effect on robustness to model evasion. We conclude by identifying the gaps in the landscape of studying interactions between other types of ML protection mechanisms.

----

## [1702] Neural Policy Safety Verification via Predicate Abstraction: CEGAR

**Authors**: *Marcel Vinzent, Siddhant Sharma, Jörg Hoffmann*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26772](https://doi.org/10.1609/aaai.v37i12.26772)

**Abstract**:

Neural networks (NN) are an increasingly important representation of action policies pi. Recent work has extended predicate abstraction to prove safety of such pi, through policy predicate abstraction (PPA) which over-approximates the state space subgraph induced by pi. The advantage of PPA is that reasoning about the NN – calls to SMT solvers – is required only locally, at individual abstract state transitions, in contrast to bounded model checking (BMC) where SMT must reason globally about sequences of NN decisions. Indeed, it has been shown that PPA can outperform a simple BMC implementation. However, the abstractions underlying these results (i.e., the abstraction predicates) were supplied manually. Here we automate this step. We extend counterexample guided abstraction refinement (CEGAR) to PPA. This involves dealing with a new source of spuriousness in abstract unsafe paths, pertaining not to transition behavior but to the decisions of the neural network pi. We introduce two methods tackling this issue based on the states involved, and we show that global SMT calls deciding spuriousness exactly can be avoided. We devise algorithmic enhancements leveraging incremental computation and heuristic search. We show empirically that the resulting verification tool has significant advantages over an encoding into the state-of-the-art model checker nuXmv. In particular, ours is the only approach in our experiments that succeeds in proving policies safe.

----

## [1703] Towards Verifying the Geometric Robustness of Large-Scale Neural Networks

**Authors**: *Fu Wang, Peipei Xu, Wenjie Ruan, Xiaowei Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26773](https://doi.org/10.1609/aaai.v37i12.26773)

**Abstract**:

Deep neural networks (DNNs) are known to be vulnerable to adversarial geometric transformation. This paper aims to verify the robustness of large-scale DNNs against the combination of multiple geometric transformations with a provable guarantee.  Given a set of transformations (e.g., rotation, scaling, etc.), we develop GeoRobust, a black-box robustness analyser built upon a novel global optimisation strategy, for locating the worst-case combination of transformations that affect and even alter a network's output.  GeoRobust can provide provable guarantees on finding the worst-case combination based on recent advances in Lipschitzian theory. Due to its black-box nature, GeoRobust can be deployed on large-scale DNNs regardless of their architectures, activation functions, and the number of neurons. In practice, GeoRobust can locate the worst-case geometric transformation with high precision for the ResNet50 model on ImageNet in a few seconds on average. We examined 18 ImageNet classifiers, including the ResNet family and vision transformers, and found a positive correlation between the geometric robustness of the networks and the parameter numbers. We also observe that increasing the depth of DNN is more beneficial than increasing its width in terms of improving its geometric robustness. Our tool GeoRobust is available at https://github.com/TrustAI/GeoRobust.

----

## [1704] Revisiting Item Promotion in GNN-Based Collaborative Filtering: A Masked Targeted Topological Attack Perspective

**Authors**: *Yongwei Wang, Yong Liu, Zhiqi Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26774](https://doi.org/10.1609/aaai.v37i12.26774)

**Abstract**:

Graph neural networks (GNN) based collaborative filtering (CF) has attracted increasing attention in e-commerce and financial marketing platforms. However, there still lack efforts to evaluate the robustness of such CF systems in deployment. Fundamentally different from existing attacks, this work revisits the item promotion task and reformulates it from a targeted topological attack perspective for the first time. Specifically, we first develop a targeted attack formulation to maximally increase a target item's popularity. We then leverage gradient-based optimizations to find a solution. However, we observe the gradient estimates often appear noisy due to the discrete nature of a graph, which leads to a degradation of attack ability. To resolve noisy gradient effects, we then propose a masked attack objective that can remarkably enhance the topological attack ability. Furthermore, we design a computationally efficient approach to the proposed attack, thus making it feasible to evaluate large-large CF systems. Experiments on two real-world datasets show the effectiveness of our attack in analyzing the robustness of GNN-based CF more practically.

----

## [1705] Robust Average-Reward Markov Decision Processes

**Authors**: *Yue Wang, Alvaro Velasquez, George K. Atia, Ashley Prater-Bennette, Shaofeng Zou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26775](https://doi.org/10.1609/aaai.v37i12.26775)

**Abstract**:

In robust Markov decision processes (MDPs), the uncertainty in the transition kernel is addressed by finding a policy that optimizes the worst-case performance over an uncertainty set of MDPs. While much of the literature has focused on discounted MDPs, robust average-reward MDPs remain largely unexplored. In this paper, we focus on robust average-reward MDPs, where the goal is to find a policy that optimizes the worst-case average reward over an uncertainty set. We first take an approach that approximates average-reward MDPs using discounted MDPs. We prove that the robust discounted value function converges to the robust average-reward as the discount factor goes to 1, and moreover when it is large, any optimal policy of the robust discounted MDP is also an optimal policy of the robust average-reward. We further design a robust dynamic programming approach, and theoretically characterize its convergence to the optimum. Then, we investigate robust average-reward MDPs directly without using discounted MDPs as an intermediate step. We derive the robust Bellman equation for robust average-reward MDPs, prove that the optimal policy can be derived from its solution, and further design a robust relative value iteration algorithm that provably finds its solution, or equivalently, the optimal robust policy.

----

## [1706] Robust Graph Meta-Learning via Manifold Calibration with Proxy Subgraphs

**Authors**: *Zhenzhong Wang, Lulu Cao, Wanyu Lin, Min Jiang, Kay Chen Tan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26776](https://doi.org/10.1609/aaai.v37i12.26776)

**Abstract**:

Graph meta-learning has become a preferable paradigm for graph-based node classification with long-tail distribution, owing to its capability of capturing the intrinsic manifold of support and query nodes. Despite the remarkable success, graph meta-learning suffers from severe performance degradation when training on graph data with structural noise. In this work, we observe that the structural noise may impair the smoothness of the intrinsic manifold supporting the support and query nodes, leading to the poor transferable priori of the meta-learner. To address the issue, we propose a new approach for graph meta-learning that is robust against structural noise, called Proxy subgraph-based Manifold Calibration method (Pro-MC). Concretely, a subgraph generator is designed to generate proxy subgraphs that can calibrate the smoothness of the manifold. The proxy subgraph compromises two types of subgraphs with two biases, thus preventing the manifold from being rugged and straightforward. By doing so, our proposed meta-learner can obtain generalizable and transferable prior knowledge. In addition, we provide a theoretical analysis to illustrate the effectiveness of Pro-MC. Experimental results have demonstrated that our approach can achieve state-of-the-art performance under various structural noises.

----

## [1707] HOTCOLD Block: Fooling Thermal Infrared Detectors with a Novel Wearable Design

**Authors**: *Hui Wei, Zhixiang Wang, Xuemei Jia, Yinqiang Zheng, Hao Tang, Shin'ichi Satoh, Zheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26777](https://doi.org/10.1609/aaai.v37i12.26777)

**Abstract**:

Adversarial attacks on thermal infrared imaging expose the risk of related applications. Estimating the security of these systems is essential for safely deploying them in the real world. In many cases, realizing the attacks in the physical space requires elaborate special perturbations. These solutions are often impractical and attention-grabbing. To address the need for a physically practical and stealthy adversarial attack, we introduce HotCold Block, a novel physical attack for infrared detectors that hide persons utilizing the wearable Warming Paste and Cooling Paste. By attaching these readily available temperature-controlled materials to the body, HotCold Block evades human eyes efficiently. Moreover, unlike existing methods that build adversarial patches with complex texture and structure features, HotCold Block utilizes an SSP-oriented adversarial optimization algorithm that enables attacks with pure color blocks and explores the influence of size, shape, and position on attack performance. Extensive experimental results in both digital and physical environments demonstrate the performance of our proposed HotCold Block. Code is available: https://github.com/weihui1308/HOTCOLDBlock.

----

## [1708] Beyond NaN: Resiliency of Optimization Layers in the Face of Infeasibility

**Authors**: *Wai Tuck Wong, Sarah Eve Kinsey, Ramesha Karunasena, Thanh Hong Nguyen, Arunesh Sinha*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26778](https://doi.org/10.1609/aaai.v37i12.26778)

**Abstract**:

Prior work has successfully incorporated optimization layers as the last layer in neural networks for various problems, thereby allowing joint learning and planning in one neural network forward pass. In this work, we identify a weakness in such a set-up where inputs to the optimization layer lead to undefined output of the neural network. Such undefined decision outputs can lead to possible catastrophic outcomes in critical real time applications. We show that an adversary can cause such failures by forcing rank deficiency on the matrix fed to the optimization layer which results in the optimization failing to produce a solution. We provide a defense for the failure cases by controlling the condition number of the input matrix. We study the problem in the settings of synthetic data, Jigsaw Sudoku, and in speed planning for autonomous driving. We show that our proposed defense effectively prevents the framework from failing with undefined output. Finally, we surface a number of edge cases which lead to serious bugs in popular optimization solvers which can be abused as well.

----

## [1709] DeepGemini: Verifying Dependency Fairness for Deep Neural Network

**Authors**: *Xuan Xie, Fuyuan Zhang, Xinwen Hu, Lei Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26779](https://doi.org/10.1609/aaai.v37i12.26779)

**Abstract**:

Deep neural networks (DNNs) have been widely adopted in many decision-making industrial applications. Their fairness issues, i.e., whether there exist unintended biases in the DNN, receive much attention and become critical concerns, which can directly cause negative impacts in our daily life and potentially undermine the fairness of our society, especially with their increasing deployment at an unprecedented speed. Recently, some early attempts have been made to provide fairness assurance of DNNs, such as fairness testing, which aims at finding discriminatory samples empirically, and fairness certification, which develops sound but not complete analysis to certify the fairness of DNNs. Nevertheless, how to formally compute discriminatory samples and fairness scores (i.e., the percentage of fair input space), is still largely uninvestigated. In this paper, we propose DeepGemini, a novel fairness formal analysis technique for DNNs, which contains two key components: discriminatory sample discovery and fairness score computation. To uncover discriminatory samples, we encode the fairness of DNNs as safety properties and search for discriminatory samples by means of state-of-the-art verification techniques for DNNs. This reduction enables us to be the first to formally compute discriminatory samples. To compute the fairness score, we develop counterexample guided fairness analysis, which utilizes four heuristics to efficiently approximate a lower bound of fairness score. Extensive experimental evaluations demonstrate the effectiveness and efficiency of DeepGemini on commonly-used benchmarks, and DeepGemini outperforms state-of-the-art DNN fairness certification approaches in terms of both efficiency and scalability.

----

## [1710] Auditing and Robustifying COVID-19 Misinformation Datasets via Anticontent Sampling

**Authors**: *Clay H. Yoo, Ashiqur R. KhudaBukhsh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26780](https://doi.org/10.1609/aaai.v37i12.26780)

**Abstract**:

This paper makes two key contributions. First, it argues that highly specialized rare content classifiers trained on small data typically have limited exposure to the richness and topical diversity of the negative class (dubbed anticontent) as observed in the wild. As a result, these classifiers' strong performance observed on the test set may not translate into real-world settings. In the context of COVID-19 misinformation detection, we conduct an in-the-wild audit of multiple datasets and demonstrate that models trained with several prominently cited recent datasets are vulnerable to anticontent when evaluated in the wild. Second, we present a novel active learning pipeline that requires zero manual annotation and iteratively augments the training data with challenging anticontent, robustifying these classifiers.

----

## [1711] User-Oriented Robust Reinforcement Learning

**Authors**: *Haoyi You, Beichen Yu, Haiming Jin, Zhaoxing Yang, Jiahui Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26781](https://doi.org/10.1609/aaai.v37i12.26781)

**Abstract**:

Recently, improving the robustness of policies across different environments attracts increasing attention in the reinforcement learning (RL) community. Existing robust RL methods mostly aim to achieve the max-min robustness by optimizing the policy’s performance in the worst-case environment. However, in practice, a user that uses an RL policy may have different preferences over its performance across environments. Clearly, the aforementioned max-min robustness is oftentimes too conservative to satisfy user preference. Therefore, in this paper, we integrate user preference into policy learning in robust RL, and propose a novel User-Oriented Robust RL (UOR-RL) framework. Specifically, we define a new User-Oriented Robustness (UOR) metric for RL, which allocates different weights to the environments according to user preference and generalizes the max-min robustness metric. To optimize the UOR metric, we develop two different UOR-RL training algorithms for the scenarios with or without a priori known environment distribution, respectively. Theoretically, we prove that our UOR-RL training algorithms converge to near-optimal policies even with inaccurate or completely no knowledge about the environment distribution. Furthermore, we carry out extensive experimental evaluations in 6 MuJoCo tasks. The experimental results demonstrate that UOR-RL is comparable to the state-of-the-art baselines under the average-case and worst-case performance metrics, and more importantly establishes new state-of-the-art performance under the UOR metric.

----

## [1712] Safety Verification of Nonlinear Systems with Bayesian Neural Network Controllers

**Authors**: *Xia Zeng, Zhengfeng Yang, Li Zhang, Xiaochao Tang, Zhenbing Zeng, Zhiming Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26782](https://doi.org/10.1609/aaai.v37i12.26782)

**Abstract**:

Bayesian neural networks (BNNs) retain NN structures with a probability distribution placed over their weights. With the introduced uncertainties and redundancies, BNNs are proper choices of robust controllers for safety-critical control systems. This paper considers the problem of verifying the safety of nonlinear closed-loop systems with BNN controllers over unbounded-time horizon. In essence, we compute a safe weight set such that as long as the BNN controller is always applied with weights sampled from the safe weight set, the controlled system is guaranteed to be safe. We propose a novel two-phase method for the safe weight set computation. First, we construct a reference safe control set that constraints the control inputs, through polynomial approximation to the BNN controller followed by polynomial-optimization-based barrier certificate generation. Then, the computation of safe weight set is reduced to a range inclusion problem of the BNN on the system domain w.r.t. the safe control set, which can be solved incrementally and the set of safe weights can be extracted. Compared with the existing method based on invariant learning and mixed-integer linear programming, we could compute safe weight sets with larger radii on a series of linear benchmarks. Moreover, experiments on a series of widely used nonlinear control tasks show that our method can synthesize large safe weight sets with probability measure as high as 95% even for a large-scale system of dimension 7.

----

## [1713] Reachability Analysis of Neural Network Control Systems

**Authors**: *Chi Zhang, Wenjie Ruan, Peipei Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26783](https://doi.org/10.1609/aaai.v37i12.26783)

**Abstract**:

Neural network controllers (NNCs) have shown great promise in autonomous and cyber-physical systems. Despite the various verification approaches for neural networks, the safety analysis of NNCs remains an open problem. Existing verification approaches for neural network control systems (NNCSs) either can only work on a limited type of activation functions, or result in non-trivial over-approximation errors with time evolving. This paper proposes a verification framework for NNCS based on Lipschitzian optimisation, called DeepNNC. We first prove the Lipschitz continuity of closed-loop NNCSs by unrolling and eliminating the loops. We then reveal the working principles of applying Lipschitzian optimisation on NNCS verification and illustrate it by verifying an adaptive cruise control model. Compared to state-of-the-art verification approaches, DeepNNC shows superior performance in terms of efficiency and accuracy over a wide range of NNCs. We also provide a case study to demonstrate the capability of DeepNNC to handle a real-world, practical, and complex system. Our tool DeepNNC is available at https://github.com/TrustAI/DeepNNC.

----

## [1714] BIFRNet: A Brain-Inspired Feature Restoration DNN for Partially Occluded Image Recognition

**Authors**: *Jiahong Zhang, Lihong Cao, Qiuxia Lai, Bingyao Li, Yunxiao Qin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26784](https://doi.org/10.1609/aaai.v37i12.26784)

**Abstract**:

The partially occluded image recognition (POIR) problem has been a challenge for artificial intelligence for a long time. A common strategy to handle the POIR problem is using the non-occluded features for classification. Unfortunately, this strategy will lose effectiveness when the image is severely occluded, since the visible parts can only provide limited information. Several studies in neuroscience reveal that feature restoration which fills in the occluded information and is called amodal completion is essential for human brains to recognize partially occluded images. However, feature restoration is commonly ignored by CNNs, which may be the reason why CNNs are ineffective for the POIR problem. Inspired by this, we propose a novel brain-inspired feature restoration network (BIFRNet) to solve the POIR problem. It mimics a ventral visual pathway to extract image features and a dorsal visual pathway to distinguish occluded and visible image regions. In addition, it also uses a knowledge module to store classification prior knowledge and uses a completion module to restore occluded features based on visible features and prior knowledge. Thorough experiments on synthetic and real-world occluded image datasets show that BIFRNet outperforms the existing methods in solving the POIR problem. Especially for severely occluded images, BIRFRNet surpasses other methods by a large margin and is close to the human brain performance. Furthermore, the brain-inspired design makes BIFRNet more interpretable.

----

## [1715] Robustness to Spurious Correlations Improves Semantic Out-of-Distribution Detection

**Authors**: *Lily H. Zhang, Rajesh Ranganath*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26785](https://doi.org/10.1609/aaai.v37i12.26785)

**Abstract**:

Methods which utilize the outputs or feature representations of predictive models have emerged as promising approaches for out-of-distribution (OOD) detection of image inputs. However, as demonstrated in previous work, these methods struggle to detect OOD inputs that share nuisance values (e.g. background) with in-distribution inputs. The detection of shared-nuisance OOD (SN-OOD) inputs is particularly relevant in real-world applications, as anomalies and in-distribution inputs tend to be captured in the same settings during deployment. In this work, we provide a possible explanation for these failures and propose nuisance-aware OOD detection to address them. Nuisance-aware OOD detection substitutes a classifier trained via Empirical Risk Minimization (ERM) with one that 1. approximates a distribution where the nuisance-label relationship is broken and 2. yields representations that are independent of the nuisance under this distribution, both marginally and conditioned on the label. We can train a classifier to achieve these objectives using Nuisance-Randomized Distillation (NuRD), an algorithm developed for OOD generalization under spurious correlations. Output- and feature-based nuisance-aware OOD detection perform substantially better than their original counterparts, succeeding even when detection based on domain generalization algorithms fails to improve performance.

----

## [1716] Evaluating Model-Free Reinforcement Learning toward Safety-Critical Tasks

**Authors**: *Linrui Zhang, Qin Zhang, Li Shen, Bo Yuan, Xueqian Wang, Dacheng Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26786](https://doi.org/10.1609/aaai.v37i12.26786)

**Abstract**:

Safety comes first in many real-world applications involving autonomous agents. Despite a large number of reinforcement learning (RL) methods focusing on safety-critical tasks, there is still a lack of high-quality evaluation of those algorithms that adheres to safety constraints at each decision step under complex and unknown dynamics. In this paper, we revisit prior work in this scope from the perspective of state-wise safe RL and categorize them as projection-based, recovery-based, and optimization-based approaches, respectively.  Furthermore, we propose Unrolling Safety Layer (USL), a joint method that combines safety optimization and safety projection. This novel technique explicitly enforces hard constraints via the deep unrolling architecture and enjoys structural advantages in navigating the trade-off between reward improvement and constraint satisfaction. To facilitate further research in this area,  we reproduce related algorithms in a unified pipeline and incorporate them into SafeRL-Kit, a toolkit that provides off-the-shelf interfaces and evaluation utilities for safety-critical tasks. We then perform a comparative study of the involved algorithms on six benchmarks ranging from robotic control to autonomous driving. The empirical results provide an insight into their applicability and robustness in learning zero-cost-return policies without task-dependent handcrafting. The project page is available at https://sites.google.com/view/saferlkit.

----

## [1717] Video-Audio Domain Generalization via Confounder Disentanglement

**Authors**: *Shengyu Zhang, Xusheng Feng, Wenyan Fan, Wenjing Fang, Fuli Feng, Wei Ji, Shuo Li, Li Wang, Shanshan Zhao, Zhou Zhao, Tat-Seng Chua, Fei Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26787](https://doi.org/10.1609/aaai.v37i12.26787)

**Abstract**:

Existing video-audio understanding models are trained and evaluated in an intra-domain setting, facing performance degeneration in real-world applications where multiple domains and distribution shifts naturally exist. The key to video-audio domain generalization (VADG) lies in  alleviating spurious correlations over multi-modal features. To achieve this goal, we resort to causal theory and attribute such correlation to confounders affecting both video-audio features and labels. We propose a DeVADG framework that conducts uni-modal and cross-modal deconfounding through back-door adjustment. DeVADG performs cross-modal disentanglement and obtains fine-grained confounders at both class-level and domain-level using half-sibling regression and unpaired domain transformation, which essentially identifies domain-variant factors and class-shared factors that cause spurious correlations between features and false labels. To promote VADG research, we collect a VADG-Action dataset for video-audio action recognition with over 5,000 video clips across four domains (e.g., cartoon and game) and ten action classes (e.g., cooking and riding). We conduct extensive experiments, i.e., multi-source DG, single-source DG, and qualitative analysis, validating the rationality of our causal analysis and the effectiveness of the DeVADG framework.

----

## [1718] Rethinking Safe Control in the Presence of Self-Seeking Humans

**Authors**: *Zixuan Zhang, Maitham Al-Sunni, Haoming Jing, Hirokazu Shirado, Yorie Nakahira*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26788](https://doi.org/10.1609/aaai.v37i12.26788)

**Abstract**:

Safe control methods are often designed to behave safely even in worst-case human uncertainties. Such design can cause more aggressive human behaviors that exploit its conservatism and result in greater risk for everyone. However, this issue has not been systematically investigated previously. This paper uses an interaction-based payoff structure from evolutionary game theory to model humans’ short-sighted, self-seeking behaviors. The model captures how prior human-machine interaction experience causes behavioral and strategic changes in humans in the long term. We then show that deterministic worst-case safe control techniques and equilibrium-based stochastic methods can have worse safety and performance trade-offs than a basic method that mediates human strategic changes. This finding suggests an urgent need to fundamentally rethink the safe control framework used in human-technology interaction in pursuit of greater safety for all.

----

## [1719] Towards Safe AI: Sandboxing DNNs-Based Controllers in Stochastic Games

**Authors**: *Bingzhuo Zhong, Hongpeng Cao, Majid Zamani, Marco Caccamo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i12.26789](https://doi.org/10.1609/aaai.v37i12.26789)

**Abstract**:

Nowadays, AI-based techniques, such as deep neural networks (DNNs), are widely deployed in autonomous systems for complex mission requirements (e.g., motion planning in robotics). However, DNNs-based controllers are typically very complex, and it is very hard to formally verify their correctness, potentially causing severe risks for safety-critical autonomous systems. In this paper, we propose a construction scheme for a so-called Safe-visor architecture to sandbox DNNs-based controllers. Particularly, we consider the construction under a stochastic game framework to provide a system-level safety guarantee which is robust to noises and disturbances. A supervisor is built to check the control inputs provided by a DNNs-based controller and decide whether to accept them.  Meanwhile, a safety advisor is running in parallel to provide fallback control inputs in case the DNN-based controller is rejected. We demonstrate the proposed approaches on a quadrotor employing an unverified DNNs-based controller.

----

## [1720] Probabilistic Programs as an Action Description Language

**Authors**: *Ronen I. Brafman, David Tolpin, Or Wertheim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26790](https://doi.org/10.1609/aaai.v37i13.26790)

**Abstract**:

Actions description languages (ADLs), such as STRIPS, PDDL, and RDDL specify the input format for planning algorithms. Unfortunately, their syntax is familiar to planning experts only, and not to potential users of planning technology. Moreover, this syntax limits the ability to describe complex and large domains. We argue that programming languages (PLs), and more specifically, probabilistic programming languages (PPLs), provide a more suitable alternative. PLs are familiar to all programmers, support complex data types and rich libraries for their manipulation, and have powerful constructs, such as loops, sub-routines, and local variables with which complex, realistic models and complex objectives can be simply and naturally specified. PPLs, specifically, make it easy to specify distributions, which is essential for stochastic models. The natural objection to this proposal is that PLs are opaque and too expressive, making reasoning about them difficult. However, PPLs also come with efficient inference algorithms, which, coupled with a growing body of work on sampling-based and gradient-based planning, imply that planning and execution monitoring can be carried out efficiently in practice. In this paper, we expand on this proposal, illustrating its potential with  examples.

----

## [1721] Foundations of Cooperative AI

**Authors**: *Vincent Conitzer, Caspar Oesterheld*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26791](https://doi.org/10.1609/aaai.v37i13.26791)

**Abstract**:

AI systems can interact in unexpected ways, sometimes with disastrous consequences. As AI gets to control more of our world, these interactions will become more common and have higher stakes. As AI becomes more advanced, these interactions will become more sophisticated, and game theory will provide the tools for analyzing these interactions. However, AI agents are in some ways unlike the agents traditionally studied in game theory, introducing new challenges as well as opportunities. We propose a research agenda to develop the game theory of highly advanced AI agents, with a focus on achieving cooperation.

----

## [1722] Multimodal Propaganda Processing

**Authors**: *Vincent Ng, Shengjie Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26792](https://doi.org/10.1609/aaai.v37i13.26792)

**Abstract**:

Propaganda campaigns have long been used to influence public opinion via disseminating biased and/or misleading information. Despite the increasing prevalence of propaganda content on the Internet, few attempts have been made by AI researchers to analyze such content. We introduce the task of multimodal propaganda processing, where the goal is to automatically analyze propaganda content. We believe that this task presents a long-term challenge to AI researchers and that successful processing of propaganda could bring machine understanding one important step closer to human understanding. We discuss the technical challenges associated with this task and outline the steps that need to be taken to address it.

----

## [1723] Foundation Model for Material Science

**Authors**: *Seiji Takeda, Akihiro Kishimoto, Lisa Hamada, Daiju Nakano, John R. Smith*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26793](https://doi.org/10.1609/aaai.v37i13.26793)

**Abstract**:

Foundation models (FMs) are achieving remarkable successes to realize complex downstream tasks in domains including natural language and visions. In this paper, we propose building an FM for material science, which is trained with massive data across a wide variety of material domains and data modalities. Nowadays machine learning models play key roles in material discovery, particularly for property prediction and structure generation. However, those models have been independently developed to address only specific tasks without sharing more global knowledge. Development of an FM for material science will enable overarching modeling across material domains and data modalities by sharing their feature representations. We discuss fundamental challenges and required technologies to build an FM from the aspects of data preparation, model development, and downstream tasks.

----

## [1724] QA Is the New KR: Question-Answer Pairs as Knowledge Bases

**Authors**: *William W. Cohen, Wenhu Chen, Michiel de Jong, Nitish Gupta, Alessandro Presta, Pat Verga, John Wieting*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26794](https://doi.org/10.1609/aaai.v37i13.26794)

**Abstract**:

We propose a new knowledge representation (KR) based on knowledge bases (KBs) derived from text, based on question generation and entity linking.  We argue that the proposed type of KB has many of the key advantages of a traditional symbolic KB: in particular, it consists of small modular components, which can be combined compositionally to answer complex queries, including relational queries and queries involving ``multi-hop'' inferences. However, unlike a traditional KB, this information store is well-aligned with common user information needs. We present one such KB, called a QEDB, and give qualitative evidence that the atomic components are high-quality and meaningful, and that atomic components can be combined in ways similar to the triples in a symbolic KB. We also show experimentally that questions reflective of typical user questions are more easily answered with a QEDB than a symbolic KB.

----

## [1725] Customer Service Combining Human Operators and Virtual Agents: A Call for Multidisciplinary AI Research

**Authors**: *Sarit Kraus, Yaniv Oshrat, Yonatan Aumann, Tal Hollander, Oleg Maksimov, Anita Ostroumov, Natali Shechtman*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26795](https://doi.org/10.1609/aaai.v37i13.26795)

**Abstract**:

The use of virtual agents (bots) has become essential for providing online assistance to customers. However, even though a lot of effort has been dedicated to the research, development, and deployment of such virtual agents, customers are frequently frustrated with the interaction with the virtual agent and require a human instead.  We suggest that a holistic approach, combining virtual agents and human operators working together, is the path to providing satisfactory service.  However, implementing such a holistic customer service system will not, and cannot, be achieved using any single AI technology or branch. Rather, such a system will inevitably require the integration of multiple and diverse AI technologies, including natural language processing, multi-agent systems, machine learning, reinforcement learning, and behavioral cloning; in addition to integration with other disciplines such as psychology, business, sociology, economics, operation research, informatics, computer-human interaction, and more. As such, we believe this customer service application offers a rich domain for experimentation and application of multidisciplinary AI.  In this paper, we introduce the holistic customer service application and discuss the key AI technologies and disciplines required for a successful AI solution for this setting.  For each of these AI technologies, we outline the key scientific questions and research avenues stemming from this setting.  We demonstrate that integrating technologies from different fields can lead to a cost-effective successful customer service center.  The challenge is that there is a need for several communities, each with its own language and modeling techniques, different problem-solving methods, and different evaluation methodologies, all of which need to work together. Real cooperation will require the formation of joint methodologies and techniques that could improve the service to customers, but, more importantly, open new directions in cooperation of diverse communities toward solving joint difficult tasks.

----

## [1726] The Many Faces of Adversarial Machine Learning

**Authors**: *Yevgeniy Vorobeychik*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26796](https://doi.org/10.1609/aaai.v37i13.26796)

**Abstract**:

Adversarial machine learning (AML) research is concerned with robustness of machine learning models and algorithms to malicious tampering. Originating at the intersection between machine learning and cybersecurity, AML has come to have broader research appeal, stretching traditional notions of security to include applications of computer vision, natural language processing, and network science. In addition, the problems of strategic classification, algorithmic recourse, and counterfactual explanations have essentially the same core mathematical structure as AML, despite distinct motivations. I give a simplified overview of the central problems in AML, and then discuss both the security-motivated AML domains, and the problems above unrelated to security. These together span a number of important AI subdisciplines, but can all broadly be viewed as concerned with trustworthy AI. My goal is to clarify both the technical connections among these, as well as the substantive differences, suggesting directions for future research.

----

## [1727] Holistic Adversarial Robustness of Deep Learning Models

**Authors**: *Pin-Yu Chen, Sijia Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26797](https://doi.org/10.1609/aaai.v37i13.26797)

**Abstract**:

Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning-based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.

----

## [1728] Can We Trust Fair-AI?

**Authors**: *Salvatore Ruggieri, José M. Álvarez, Andrea Pugnana, Laura State, Franco Turini*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26798](https://doi.org/10.1609/aaai.v37i13.26798)

**Abstract**:

There is a fast-growing literature in addressing the fairness of AI models (fair-AI), with a continuous stream of new conceptual frameworks, methods, and tools. How much can we trust them? How much do they actually impact society? 
We take a critical focus on fair-AI and survey issues, simplifications, and mistakes that researchers and practitioners often underestimate, which in turn can undermine the trust on fair-AI and limit its contribution to society. In particular, we discuss the hyper-focus on fairness metrics and on optimizing their average performances. We instantiate this observation by discussing the Yule's effect of fair-AI tools: being fair on average does not imply being fair in contexts that matter. We conclude that the use of fair-AI methods should be complemented with the design, development, and verification practices that are commonly summarized under the umbrella of trustworthy AI.

----

## [1729] Safety Validation of Learning-Based Autonomous Systems: A Multi-Fidelity Approach

**Authors**: *Ali Baheri*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26799](https://doi.org/10.1609/aaai.v37i13.26799)

**Abstract**:

In recent years, learning-based autonomous systems have emerged as a promising tool for automating many crucial tasks. The key question is how we can build trust in such systems for safety-critical applications. My research aims to focus on the creation and validation of safety frameworks that leverage multiple sources of information. The ultimate goal is to establish a solid foundation for a long-term research program aimed at understanding the role of fidelity in simulators for safety validation and robot learning.

----

## [1730] Probabilistic Reasoning and Learning for Trustworthy AI

**Authors**: *YooJung Choi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26800](https://doi.org/10.1609/aaai.v37i13.26800)

**Abstract**:

As automated decision-making systems are increasingly deployed in areas with personal and societal impacts, there is a growing demand for artificial intelligence and machine learning systems that are fair, robust, interpretable, and generally trustworthy. Ideally we would wish to answer questions regarding these properties and provide guarantees about any automated system to be deployed in the real world. This raises the need for a unified language and framework under which we can reason about and develop trustworthy AI systems. This talk will discuss how tractable probabilistic reasoning and learning provides such framework. 

It is important to note that guarantees regarding fairness, robustness, etc., hold with respect to the distribution of the world in which the decision-making system operates. For example, to see whether automated loan decisions are biased against certain gender, one may compare the average decision for each gender; this requires knowledge of how the features used in the decision are distributed for each gender. Moreover, there are inherent uncertainties in modeling this distribution, in addition to the uncertainties when deploying a system in the real world, such as missing or noisy information. We can handle such uncertainties in a principled way through probabilistic reasoning. Taking fairness-aware learning as an example, we can deal with biased labels in the training data by explicitly modeling the observed labels as being generated from some probabilistic process that injects bias/noise to hidden, fair labels, particularly in a way that best explains the observed data.

A key challenge that still needs to be addressed is that: we need models that can closely fit complex real-world distributions—i.e. expressive—while also being amenable to exact and efficient inference of probabilistic queries—i.e. tractable. I will show that probabilistic circuits, a family of tractable probabilistic models, offer both such benefits. In order to ultimately develop a common framework to study various areas of trustworthy AI (e.g., privacy, fairness, explanations, etc.), we need models that can flexibly answer different questions, even the ones it did not foresee. This talk will thus survey the efforts to expand the horizon of complex reasoning capabilities of probabilistic circuits, especially highlighted by a modular approach that answers various queries via a pipeline of a handful of simple tractable operations.

----

## [1731] The Automatic Computer Scientist

**Authors**: *Andrew Cropper*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26801](https://doi.org/10.1609/aaai.v37i13.26801)

**Abstract**:

Algorithms are ubiquitous: they track our sleep, help us find cheap flights, and even help us see black holes. However, designing novel algorithms is extremely difficult, and we do not have efficient algorithms for many fundamental problems. The goal of my research is to accelerate algorithm discovery by building an automatic computer scientist. To work towards this goal, my research focuses on inductive logic programming, a form of machine learning in which my collaborators and I have demonstrated major advances in automated algorithm discovery over the past five years. In this talk and paper, I survey these advances.

----

## [1732] Perception for General-purpose Robot Manipulation

**Authors**: *Karthik Desingh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26802](https://doi.org/10.1609/aaai.v37i13.26802)

**Abstract**:

To autonomously perform tasks, a robot should continually perceive the state of its environment, reason with the task at hand, plan and execute appropriate actions. In this pipeline, perception is largely unsolved and one of the more challenging problems. Common indoor environments typically pose two main problems: 1) inherent occlusions leading to unreliable observations of objects, and 2) the presence and involvement of a wide range of objects with varying physical and visual attributes (i.e., rigid, articulated, deformable, granular, transparent, etc.). Thus, we need algorithms that can accommodate perceptual uncertainty in the state estimation and generalize to a wide range of objects. Probabilistic inference methods have been highly suitable for modeling perceptual uncertainty, and data-driven approaches using deep learning techniques have shown promising advancements toward generalization. Perception for manipulation is a more intricate setting requiring the best from both worlds. My research aims to develop robot perception algorithms that can generalize over objects and tasks while accommodating perceptual uncertainty to support robust task execution in the real world. In this presentation, I will briefly highlight my research in these two research threads.

----

## [1733] Cooperative Multi-Agent Learning in a Complex World: Challenges and Solutions

**Authors**: *Yali Du*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26803](https://doi.org/10.1609/aaai.v37i13.26803)

**Abstract**:

Over the past few years, artificial intelligence (AI) has achieved great success in a variety of applications, such as image classification and recommendation systems. This success has often been achieved by training machine learning models on static datasets, where inputs and desired outputs are provided.
However, we are now seeing a shift in this paradigm. Instead of learning from static datasets, machine learning models are increasingly being trained through feedback from their interactions with the world. This is particularly important when machine learning models are deployed in the real world, as their decisions can often have an impact on other agents, turning the decision-making process into a multi-agent problem.
As a result, multi-agent learning in complex environments is a critical area of research for the next generation of AI, particularly in the context of cooperative tasks. Cooperative multi-agent learning is an essential problem for practitioners to consider as it has the potential to enable a wide range of multi-agent tasks.
In this presentation, we will review the background and challenges of cooperative multi-agent learning, and survey our research that aims to address  these challenges.

----

## [1734] Distributed Stochastic Nested Optimization for Emerging Machine Learning Models: Algorithm and Theory

**Authors**: *Hongchang Gao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26804](https://doi.org/10.1609/aaai.v37i13.26804)

**Abstract**:

Traditional machine learning models can be formulated as the expected risk minimization (ERM) problem:
minw∈Rd Eξ [l(w; ξ)], where w ∈ Rd denotes the model parameter, ξ represents training samples, l(·) is the loss function. Numerous optimization algorithms, such as stochastic gradient descent (SGD), have been developed to solve the ERM problem. However, a wide range of emerging machine learning models are beyond this class of optimization problems, such as model-agnostic meta-learning (Finn, Abbeel, and Levine 2017). Of particular interest of my research is the stochastic nested optimization (SNO) problem, whose objective function has a nested structure. Specifically, I have been focusing on two instances of this kind of problem: stochastic compositional optimization (SCO) problems, which cover meta-learning, area-under-the-precision recall-curve optimization, contrastive self-supervised learning, etc., and stochastic bilevel optimization (SBO) problems, which can be applied to meta-learning, hyperparameter optimization, neural network architecture search, etc.
With the emergence of large-scale distributed data, such as the user data generated on mobile devices or intelligent hardware, it is imperative to develop distributed optimization algorithms for SNO (Distributed SNO). A significant challenge for optimizing distributed SNO problems lies in that the stochastic (hyper-)gradient is a biased estimation of the full gradient. Thus, existing distributed optimization algorithms when applied to them suffer from slow convergence rates. In this talk, I will discuss my recent works about distributed SCO (Gao and Huang 2021; Gao, Li, and Huang 2022) and distributed SBO (Gao, Gu, and Thai 2022; Gao 2022) under both centralized and decentralized settings, including algorithmic details about reducing the bias of stochastic gradient, theoretical convergence rate, and practical machine learning applications, and then highlight challenges for future research.

----

## [1735] Targeted Knowledge Infusion To Make Conversational AI Explainable and Safe

**Authors**: *Manas Gaur*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26805](https://doi.org/10.1609/aaai.v37i13.26805)

**Abstract**:

Conversational Systems (CSys) represent practical and tangible outcomes of advances in NLP and AI. CSys see continuous improvements through unsupervised training of large language models (LLMs) on a humongous amount of generic training data. However, when these CSys are suggested for use in domains like Mental Health, they fail to match the acceptable standards of clinical care, such as the clinical process in Patient Health Questionnaire (PHQ-9). The talk will present, Knowledge-infused Learning (KiL), a paradigm within NeuroSymbolic AI that focuses on making machine/deep learning models (i) learn over knowledge-enriched data, (ii) learn to follow guidelines in process-oriented tasks for safe and reasonable generation, and (iii) learn to leverage multiple contexts and stratified knowledge to yield user-level explanations. KiL established Knowledge-Intensive Language Understanding, a set of tasks for assessing safety, explainability, and conceptual flow in CSys.

----

## [1736] Accountability Layers: Explaining Complex System Failures by Parts

**Authors**: *Leilani H. Gilpin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26806](https://doi.org/10.1609/aaai.v37i13.26806)

**Abstract**:

With the rise of AI used for critical decision-making, many important predictions are made by complex and opaque AI algorithms.  The aim of eXplainable Artificial Intelligence (XAI) is to make these opaque decision-making algorithms more transparent and trustworthy.  This is often done by constructing an ``explainable model'' for a single modality or subsystem. However, this approach fails for complex systems that are made out of multiple parts.  In this paper, I discuss how to explain complex system failures.  I represent a complex machine as a hierarchical model of introspective sub-systems working together towards a common goal. The subsystems communicate in a common symbolic language.  This work creates a set of explanatory accountability layers for trustworthy AI.

----

## [1737] Generative Decision Making Under Uncertainty

**Authors**: *Aditya Grover*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26807](https://doi.org/10.1609/aaai.v37i13.26807)

**Abstract**:

In the fields of natural language processing (NLP) and computer vision (CV), recent advances in generative modeling have led to powerful machine learning systems that can effectively learn from large labeled and unlabeled datasets. These systems, by and large, apply a uniform pretrain-finetune pipeline on sequential data streams and have achieved state-of-the-art-performance across many tasks and benchmarks. In this talk, we will present recent algorithms that extend this paradigm to sequential decision making, by casting it as an inverse problem that can be solved via deep generative models. These generative approaches are stable to train, provide a flexible interface for single- and multi-task inference, and generalize exceedingly well outside their training datasets. We instantiate these algorithms in the context of reinforcement learning and black-box optimization. Empirically, we demonstrate that these approaches perform exceedingly well on high-dimensional benchmarks outperforming the current state-of-the-art approaches based on forward models.

----

## [1738] Food Information Engineering: A Systematic Literature Review

**Authors**: *Azanzi Jiomekong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26808](https://doi.org/10.1609/aaai.v37i13.26808)

**Abstract**:

In recent years, the research on food information gave rise to the food information engineering domain. The goal of this paper is to provide to the research community with a systematic literature review of methodologies, methods and tools used in this domain.

----

## [1739] Better Environments for Better AI

**Authors**: *Sarah Keren*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26809](https://doi.org/10.1609/aaai.v37i13.26809)

**Abstract**:

Most past research aimed at increasing the capabilities of AI methods has focused exclusively on the AI agent itself, i.e., given some input, what are the improvements to the agent’s reasoning that will yield the best possible output. In my research, I take a novel approach to increasing the capabilities of AI agents via the design of the environments in which they are intended to act. My methods for automated design identify the inherent capabilities and limitations of AI agents with respect to their environment and find the best way to modify the environment to account for those limitations and maximize the agents’ performance.

The future will bring an ever increasing set of interactions between people and automated agents, whether at home, at the workplace, on the road, or across many other everyday settings. Autonomous vehicles, robotic tools, medical devices, and smart
homes, all allow ample opportunity for human-robot and multi-agent interactions. In these settings, recognizing what agents are trying to achieve, providing relevant assistance, and supporting an effective collaboration are essential tasks, and tasks
that can all be enhanced via careful environment design. However, the increasing complexity of the systems we use and the environments in which we operate makes devising good design solutions extremely challenging. This stresses the importance
of developing automated design tools to help determine the most effective ways to apply change and enable robust AI systems. My long-term goal is to provide theoretical foundations for designing AI systems that are capable of effective partnership in sustainable and efficient collaborations of automated agents as well as of automated agents and people.

----

## [1740] Recent Developments in Data-Driven Algorithms for Discrete Optimization

**Authors**: *Elias B. Khalil*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26810](https://doi.org/10.1609/aaai.v37i13.26810)

**Abstract**:

The last few years have witnessed a renewed interest in “data-driven algorithm design” (Balcan 2020), the use of Machine Learning (ML) to tailor an algorithm to a distribution of instances. More than a decade ago, advances in algorithm configuration (Hoos 2011) paved the way for the use of historical data to modify an algorithm’s (typically fixed, static) parameters. In discrete optimization (e.g., satisfiability, integer programming, etc.), exact and inexact algorithms for NP-Hard problems often involve heuristic search decisions (Lodi 2013), abstracted as parameters, that can demonstrably benefit from tuning on historical instances from the application of interest.

While useful, algorithm configuration may be insufficient: setting the parameters of an algorithm upfront of solving the input instance is still a static, high-level decision. In contrast, we have been exploring a suite of ML and Reinforcement Learning (RL) approaches that tune iterative optimization algorithms, such as branch-and-bound for integer programming or construction heuristics, at the iteration-level (Khalil et al. 2016, 2017; Dai et al. 2017; Chmiela et al. 2021; Gupta et al. 2022; Chi et al. 2022; Khalil, Vaezipoor, and Dilkina 2022; Khalil, Morris, and Lodi 2022; Alomrani, Moravej, and Khalil 2022; Cappart et al. 2021; Gupta et al. 2020).

We will survey our most recent work in this area:
1. New methods for learning in MILP branch-and-bound (Gupta et al. 2020, 2022; Chmiela et al. 2021; Khalil, Vaezipoor, and Dilkina 2022; Khalil, Morris, and Lodi 2022);

2. RL for online combinatorial optimization and largescale linear programming (Alomrani, Moravej, and Khalil 2022; Chi et al. 2022);

3. Neural network approximations for stochastic programming (Dumouchelle et al. 2022).

----

## [1741] Advances in AI for Safety, Equity, and Well-Being on Web and Social Media: Detection, Robustness, Attribution, and Mitigation

**Authors**: *Srijan Kumar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26811](https://doi.org/10.1609/aaai.v37i13.26811)

**Abstract**:

In the talk, I shall describe my lab’s recent advances in AI, applied machine learning, and data mining to combat malicious actors (sockpuppets, ban evaders, etc.) and dangerous content (misinformation, hate, etc.) on web and social media platforms. My vision is to create a trustworthy online ecosystem for everyone and create the next generation of socially-aware methods that promote health, equity, and safety. Broadly, in my research, I have created novel graph, content (NLP, multimodality), and adversarial machine learning methods leveraging terabytes of data to detect, predict, and mitigate online threats. I shall describe the advancements made in my group across four key thrusts: (1) Detection of harmful content and malicious actors across platforms, languages, and modalities, (2) Robustifying detection models against adversarial actors by predicting future malicious activities, (3) Attributing the impact of harmful content and the role of recommender systems, and (4) Developing mitigation techniques to counter misinformation by professionals and the crowd.

----

## [1742] Intelligent Planning for Large-Scale Multi-Robot Coordination

**Authors**: *Jiaoyang Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26812](https://doi.org/10.1609/aaai.v37i13.26812)

**Abstract**:

Robots will play a crucial role in the future and need to work as a team in increasingly more complex applications. Advances in robotics have laid the hardware foundations for building large-scale multi-robot systems. But how to coordinate robots intelligently is a difficult problem. We believe that graph-search-based planning can systematically exploit the combinatorial structure of multi-robot coordination problems and efficiently generate solutions with rigorous guarantees on correctness, completeness, and solution quality. We started with one problem that is central to many multi-robot applications. Multi-Agent Path Finding (MAPF) is an NP-hard problem of planning collision-free paths for a team of agents while minimizing their travel times. We addressed the MAPF problem from both (1) a theoretical perspective by developing efficient algorithms to solve large MAPF instances with completeness and optimality guarantees via a variety of AI and optimization technologies, such as constraint reasoning, heuristic search, stochastic local search, and machine learning, and (2) an applicational perspective by developing algorithmic techniques for integrating MAPF with task planning and execution for various multi-robot systems, such as mobile robot coordination, traffic management, drone swarm control, multi-arm assembly, and character control in video games. This paper is part of the AAAI-23 New Faculty Highlights.

----

## [1743] Robust and Adaptive Deep Learning via Bayesian Principles

**Authors**: *Yingzhen Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26813](https://doi.org/10.1609/aaai.v37i13.26813)

**Abstract**:

Deep learning models have achieved tremendous successes in accurate predictions for computer vision, natural language processing and speech recognition applications. However, to succeed in high-risk and safety-critical domains such as healthcare and finance, these deep learning models need to be made reliable and trustworthy. Specifically, they need to be robust and adaptive to real-world environments which can be drastically different from the training settings. In this talk, I will advocate for Bayesian principles to achieve the goal of building robust and adaptive deep learning models. I will introduce a suite of uncertainty quantification methods for Bayesian deep learning, and demonstrate applications en- abled by accurate uncertainty estimates, e.g., robust predic- tion, continual learning and repairing model failures. I will conclude by discussing the research challenges and potential impact for robust and adaptive deep learning models.

This paper is part of the AAAI-23 New Faculty Highlights.

----

## [1744] AAAI New Faculty Highlights: General and Scalable Optimization for Robust AI

**Authors**: *Sijia Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26814](https://doi.org/10.1609/aaai.v37i13.26814)

**Abstract**:

Deep neural networks (DNNs) can easily be manipulated (by an adversary) to output drastically different predictions and can be done so in a controlled and directed way. This process is known as adversarial attack and is considered one of the major hurdles in using DNNs in high-stakes and real-world applications. Although developing methods to secure DNNs against adversaries is now a primary research focus, it suffers from limitations such as lack of optimization generality and lack of optimization scalability. My research highlights will offer a holistic understanding of optimization foundations for robust AI, peer into their emerging challenges, and present recent solutions developed by my research group.

----

## [1745] Combining Runtime Monitoring and Machine Learning with Human Feedback

**Authors**: *Anna Lukina*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26815](https://doi.org/10.1609/aaai.v37i13.26815)

**Abstract**:

State-of-the-art machine-learned controllers for autonomous systems demonstrate unbeatable performance in scenarios known from training. However, in evolving environments---changing weather or unexpected anomalies---, safety and interpretability remain the greatest challenges for autonomous systems to be reliable and are the urgent scientific challenges.

Existing machine-learning approaches focus on recovering lost performance but leave the system open to potential safety violations. Formal methods address this problem by rigorously analysing a smaller representation of the system but they rarely prioritize performance of the controller. 

We propose to combine insights from formal verification and runtime monitoring with interpretable machine-learning design for guaranteeing reliability of autonomous systems.

----

## [1746] Towards Safe and Resilient Autonomy in Multi-Robot Systems

**Authors**: *Wenhao Luo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26816](https://doi.org/10.1609/aaai.v37i13.26816)

**Abstract**:

In the near future, autonomous systems such as multi-robot
systems are envisioned to increasingly co-exist with hu-
mans in our daily lives, from household service to large-
scale warehouse logistics, agriculture environment sampling,
and smart city. In these applications, robots and humans as
networked heterogeneous components will frequently inter-
act with each other in a variety of scenarios under uncer-
tain, rapidly-changing, and possibly hostile environment. On
one hand, harmonious interactions among robots, as well as
between robots and humans, would require safe integration
(e.g. collision-free close-proximity interactions) of heteroge-
neous robots, human, and human-robot autonomy. On the
other hand, reliable interactions among autonomous multi-
robot systems often call for resilient system integrity (e.g.
communication capability with potential robot failures) to re-
tain its capability of accomplishing complex tasks through
coordinated behaviors. In the proposed talk, I will discuss our
recent works towards safe autonomy and resilient autonomy
that aim to facilitate correct-by-design robotic behaviors in a
variety of applications.

----

## [1747] Monitoring and Intervening on Large Populations of Weakly Coupled Processes with Social Impact Applications

**Authors**: *Andrew Perrault*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26817](https://doi.org/10.1609/aaai.v37i13.26817)

**Abstract**:

Many real-world sequential decision problems can be decomposed into processes with independent dynamics that are coupled via the action structure. We discuss recent work on such problems and future directions.

----

## [1748] Internal Robust Representations for Domain Generalization

**Authors**: *Mohammad Rostami*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26818](https://doi.org/10.1609/aaai.v37i13.26818)

**Abstract**:

Model generalization under distributional changes remains a significant challenge for machine learning. We present consolidating the internal representation of the training data in a model as a strategy of improving model generalization.

----

## [1749] Planning and Learning for Reliable Autonomy in the Open World

**Authors**: *Sandhya Saisubramanian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26819](https://doi.org/10.1609/aaai.v37i13.26819)

**Abstract**:

Safe and reliable decision-making is critical for long-term deployment of autonomous systems. Despite the recent advances in artificial intelligence, ensuring safe and reliable operation of human-aligned autonomous systems in open-world environments remains a challenge. My research focuses on developing planning and learning algorithms that support reliable autonomy in fully and partially observable environments, in the presence of uncertainty, limited information, and limited resources. This talk covers a summary of my recent research towards reliable autonomy.

----

## [1750] Dynamics of Cooperation and Conflict in Multiagent Systems

**Authors**: *Fernando P. Santos*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26820](https://doi.org/10.1609/aaai.v37i13.26820)

**Abstract**:

Meeting today’s major scientific and societal challenges requires understanding the dynamics of cooperation, coordination, and conflict in complex adaptive systems (CAS). Artificial Intelligence (AI) is intimately connected with these challenges, both as an application domain and as a source of new computational techniques: On the one hand, AI suggests new algorithmic recommendations and interaction paradigms, offering novel possibilities to engineer cooperation and alleviate conflict in multiagent (hybrid) systems; on the other hand, new learning algorithms provide improved techniques to simulate sophisticated agents and increasingly realistic CAS. My research lies at the interface between CAS and AI: I develop computational methods to understand cooperation and conflict in multiagent systems, and how these depend on systems’ design and incentives. I focus on mapping interaction rules and incentives onto emerging macroscopic patterns and long-term dynamics. Examples of this research agenda, that I will survey in this talk, include modelling (1) the connection between reputation systems and cooperation dynamics, (2) the role of agents with hard-coded strategies in stabilizing fair behaviors in a population, or (3) the impact of recommendation algorithms on potential sources of conflict (e.g., radicalization and polarization) in a system composed of adaptive agents influencing each other over time.

----

## [1751] Combating Disinformation on Social Media and Its Challenges: A Computational Perspective

**Authors**: *Kai Shu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26821](https://doi.org/10.1609/aaai.v37i13.26821)

**Abstract**:

The use of social media has accelerated information sharing and instantaneous communications. The low barrier to entering social media enables more users to participate and keeps them engaged longer, incentivizing individuals with a hidden agenda to spread disinformation online to manipulate information and sway opinion. Disinformation, such as fake news, hoaxes, and conspiracy theories, has increasingly become a hindrance to the functioning of online social media as an effective channel for trustworthy information. Therefore, it is imperative to understand disinformation and systematically investigate how to improve resistance against it. This article highlights relevant theories and recent advancements of detecting disinformation from a computational perspective, and urges the need for future interdisciplinary research.

----

## [1752] Human-Aware AI - A Foundational Framework for Human-AI Interaction

**Authors**: *Sarath Sreedharan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26822](https://doi.org/10.1609/aaai.v37i13.26822)

**Abstract**:

We are living through a revolutionary moment in AI history. We are seeing the development of impressive new AI systems at a rate that was unimaginable just a few years ago. However, AI's true potential to transform society remains unrealized, in no small part due to the inability of current systems to work effectively with people. A major hurdle to achieving such coordination is the inherent asymmetry between the AI system and its users. In this talk, I will discuss how the framework of Human-Aware AI (HAAI) provides us with the tools required to bridge this gap and support fluent and intuitive coordination between the AI system and its users.

----

## [1753] Towards Unified, Explainable, and Robust Multisensory Perception

**Authors**: *Yapeng Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26823](https://doi.org/10.1609/aaai.v37i13.26823)

**Abstract**:

Humans perceive surrounding scenes through multiple senses with multisensory integration. For example, hearing helps capture the spatial location of a racing car behind us; seeing peoples' talking faces can strengthen our perception of their speech. However, today's state-of-the-art scene understanding systems are usually designed to rely on a single audio or visual modality. Ignoring multisensory cooperation has become one of the key bottlenecks in creating intelligent systems with human-level perception capability, which impedes the real-world applications of existing scene understanding models.  To address this limitation, my research has pioneered marrying computer vision with computer audition to create multimodal systems that can learn to understand audio and visual data. In particular, my current research focuses on asking and solving fundamental problems in a fresh research area: audio-visual scene understanding and strives to develop unified, explainable, and robust multisensory perception machines. The three themes are distinct yet interconnected, and all of them are essential for designing powerful and trustworthy perception systems. In my talk, I will give a brief overview about this new research area and then introduce my works in the three research thrusts.

----

## [1754] Reshaping State-Space Search: From Dominance to Contrastive Analysis

**Authors**: *Álvaro Torralba*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26824](https://doi.org/10.1609/aaai.v37i13.26824)

**Abstract**:

State-space search is paramount for intelligent decision making when long-term thinking is needed. We introduce dominance and contrastive analysis methods, which enable reasoning about the relative advantages among different courses of action. This re-shapes how agents reason and leads to new families of state-space search algorithms.

----

## [1755] Artificial Intelligence at the Service of Society to Analyse Human Arguments

**Authors**: *Serena Villata*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26825](https://doi.org/10.1609/aaai.v37i13.26825)

**Abstract**:

Argument(ation) mining (AM) is an area of research in Artificial Intelligence (AI) that aims to identify, analyse and automatically generate arguments in natural language. In a pipeline, the identification and analysis of the arguments and their components (i.e. premises and claims) in texts and the prediction of their relations (i.e. attack and support) are then handled by argument-based reasoning frameworks so that, for example, fallacies and inconsistencies can be automatically identified. Recently, the field of argument mining has tackled new challenges, namely the evaluation of argument quality (e.g. strength, persuasiveness), natural language argument summarisation and retrieval, and natural language argument generation. In this paper, I discuss my main contributions in this area as well as some lines of future research.  This paper is part of the AAAI-23 New Faculty Highlights.

----

## [1756] AI for Equitable, Data-Driven Decisions in Public Health

**Authors**: *Bryan Wilder*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26826](https://doi.org/10.1609/aaai.v37i13.26826)

**Abstract**:

As exemplified by the COVID-19 pandemic, our health and wellbeing depend on a difficult-to-measure web of societal factors and individual behaviors. This effort requires new algorithmic and data-driven paradigms which span the full process of gathering costly data, learning models to understand and predict such interactions, and optimizing the use of limited resources in interventions. In response to these needs, I present methodological developments at the intersection of machine learning, optimization, and social networks which are motivated by on-the-ground collaborations on HIV prevention, tuberculosis treatment, and the COVID-19 response. Here, I give an overview of two lines of work.

----

## [1757] Learning to See the Physical World

**Authors**: *Jiajun Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26827](https://doi.org/10.1609/aaai.v37i13.26827)

**Abstract**:

This paper is part of the AAAI-23 New Faculty Highlights. In my presentation, I will introduce my research goal, which is to build machines that see, interact with, and reason about the physical world just like humans. This problem, which we call physical scene understanding, involves three key topics that bridge research in computer science, AI, robotics, cognitive science, and neuroscience: Perception, Physical Interaction, and Reasoning.

----

## [1758] Enhance Robustness of Machine Learning with Improved Efficiency

**Authors**: *Yan Yan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26828](https://doi.org/10.1609/aaai.v37i13.26828)

**Abstract**:

Robustness of machine learning, often referring to securing performance on different data, is always an active field due to the ubiquitous variety and diversity of data in practice. Many studies have been investigated to enhance the learning process robust in recent years. To this end, there is usually a trade-off that results in somewhat extra cost, e.g., more data samples, more complicated objective functions, more iterations to converge in optimization, etc. Then this problem boils down to finding a better trade-off under some conditions. My recent research focuses on robust machine learning with improved efficiency. Particularly, the efficiency here represents learning speed to find a model, and the number of data required to secure the robustness. In the talk, I will survey three pieces of my recent research by elaborating the algorithmic idea and theoretical analysis as technical contributions --- (i) epoch stochastic gradient descent ascent for min-max problems, (ii) stochastic optimization algorithm for non-convex inf-projection problems, and (iii) neighborhood conformal prediction. In the first two pieces of work, the proposed optimization algorithms are general and cover objective functions for robust machine learning. In the third one, I will elaborate an efficient conformal prediction algorithm that guarantee the robustness of prediction after model is trained. Particularly, the efficiency of conformal prediction is measured by its bandwidth.

----

## [1759] The Analysis of Deep Neural Networks by Information Theory: From Explainability to Generalization

**Authors**: *Shujian Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26829](https://doi.org/10.1609/aaai.v37i13.26829)

**Abstract**:

Despite their great success in many artificial intelligence tasks, deep neural networks (DNNs) still suffer from a few limitations, such as poor generalization behavior for out-of-distribution (OOD) data and the "black-box" nature. Information theory offers fresh insights to solve these challenges. In this short paper, we briefly review the recent developments in this area, and highlight our contributions.

----

## [1760] Towards Societal Impact of AI

**Authors**: *Chuxu Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26830](https://doi.org/10.1609/aaai.v37i13.26830)

**Abstract**:

Artificial intelligence (AI) and Machine Learning (ML) have shown great success in many areas such as computer vision, natural language processing, and knowledge discovery. However, AI research to deliver social benefits and impacts is less explored while imminent needed. Guided by the United Nations’ Sustainable Development Goals, my research involves the development of advanced AI techniques, in particular Deep Graph Learning (DGL), to address the grand societal challenges and further apply them to various social good applications for improving our society and people’s daily life, namely DGL for Social Good (DGL4SG). Achieving the goal is not easy since challenges come from the increasing complexity of many factors including problems, data, and techniques, which require long-term and concentrated effort. DGL presents a good opportunity to build better solutions and tools due to its strong capability in learning and inferring graph data which is ideal for modeling many real-world social good systems. Fortunately, I have been working on DGL with continued contributions and impacts since my graduate study. The special research experience lifts me up to a unique position for conducting research that intersects AI, DGL, and social good, and pushing the field of DGL4SG forward.

----

## [1761] Information Transfer in Multitask Learning, Data Augmentation, and Beyond

**Authors**: *Hongyang R. Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26831](https://doi.org/10.1609/aaai.v37i13.26831)

**Abstract**:

A hallmark of human intelligence is that we continue to learn new information and then extrapolate the learned information onto new tasks and domains (see, e.g., Thrun and Pratt (1998)). While this is a fairly intuitive observation, formulating such ideas has proved to be a challenging research problem and continues to inspire new studies. Recently, there has been increasing interest in AI/ML about building models that generalize across tasks, even when they have some form of distribution shifts. How can we ground this research in a solid framework to develop principled methods for better practice? This talk will present my recent works addressing this research question. My talk will involve three parts: revisiting multitask learning from the lens of deep learning theory, designing principled methods for robust transfer, and algorithmic implications for data augmentation.

----

## [1762] A New Challenge in Policy Evaluation

**Authors**: *Shangtong Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26832](https://doi.org/10.1609/aaai.v37i13.26832)

**Abstract**:

This paper proposes a new challenge in policy evaluation: to improve the online data efficiency of Monte Carlo methods via information extracted from offline data while maintaining the unbiasedness of Monte Carlo methods.

----

## [1763] Building Compositional Robot Autonomy with Modularity and Abstraction

**Authors**: *Yuke Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26833](https://doi.org/10.1609/aaai.v37i13.26833)

**Abstract**:

This paper summarizes my research roadmap for building compositional robot autonomy with the principles of modularity and abstraction.

----

## [1764] Accurate Detection of Weld Seams for Laser Welding in Real-World Manufacturing

**Authors**: *Rabia Ali, Muhammad Sarmad, Jawad Tayyub, Alexander Vogel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26834](https://doi.org/10.1609/aaai.v37i13.26834)

**Abstract**:

Welding is a fabrication process used to join or fuse two mechanical parts. Modern welding machines have automated lasers that follow a pre-defined weld seam path between the two parts to create a bond. Previous efforts have used simple computer vision edge detectors to automatically detect the weld seam edge on an image at the junction of two metals to be welded. However, these systems lack reliability and accuracy resulting in manual human verification of the detected edges. This paper presents a neural network architecture that automatically detects the weld seam edge between two metals with high accuracy. We augment this system with a pre-classifier that filters out anomalous workpieces (e.g., incorrect placement). Finally, we justify our design choices by evaluating against several existing deep network pipelines as well as proof through real-world use. We also describe in detail the process of deploying this system in a real-world shop floor including evaluation and monitoring. We make public a large, well-labeled laser seam dataset to perform deep learning-based edge detection in industrial settings.

----

## [1765] Blending Advertising with Organic Content in E-commerce via Virtual Bids

**Authors**: *Carlos Carrion, Zenan Wang, Harikesh S. Nair, Xianghong Luo, Yulin Lei, Peiqin Gu, Xiliang Lin, Wenlong Chen, Junsheng Jin, Fanan Zhu, Changping Peng, Yongjun Bao, Zhangang Lin, Weipeng Yan, Jingping Shao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26835](https://doi.org/10.1609/aaai.v37i13.26835)

**Abstract**:

It has become increasingly common that sponsored content (i.e., paid ads) and non-sponsored content are jointly displayed to users, especially on e-commerce platforms. Thus, both of these contents may interact together to influence their engagement behaviors. In general, sponsored content helps brands achieve their marketing goals and provides ad revenue to the platforms. In contrast, non-sponsored content contributes to the long-term health of the platform through increasing users' engagement. A key conundrum to platforms is learning how to blend both of these contents allowing their interactions to be considered and balancing these business objectives. This paper proposes a system built for this purpose and applied to product detail pages of JD.COM, an e-commerce company. This system achieves three objectives: (a) Optimization of competing business objectives via Virtual Bids allowing the expressiveness of the valuation of the platform for these objectives. (b) Modeling the users' click behaviors considering explicitly the influence exerted by the sponsored and non-sponsored content displayed alongside through a deep learning approach. (c) Consideration of a Vickrey-Clarke-Groves (VCG) Auction design compatible with the allocation of ads and its induced externalities. Experiments are presented demonstrating the performance of the proposed system. Moreover, our approach is fully deployed and serves all traffic through JD.COM's mobile application.

----

## [1766] Efficient Training of Large-Scale Industrial Fault Diagnostic Models through Federated Opportunistic Block Dropout

**Authors**: *Yuanyuan Chen, Zichen Chen, Sheng Guo, Yansong Zhao, Zelei Liu, Pengcheng Wu, Chengyi Yang, Zengxiang Li, Han Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26836](https://doi.org/10.1609/aaai.v37i13.26836)

**Abstract**:

Artificial intelligence (AI)-empowered industrial fault diagnostics is important in ensuring the safe operation of industrial applications. Since complex industrial systems often involve multiple industrial plants (possibly belonging to different companies or subsidiaries) with sensitive data collected and stored in a distributed manner, collaborative fault diagnostic model training often needs to leverage federated learning (FL). As the scale of the industrial fault diagnostic models are often large and communication channels in such systems are often not exclusively used for FL model training, existing deployed FL model training frameworks cannot train such models efficiently across multiple institutions. In this paper, we report our experience developing and deploying the Federated Opportunistic Block Dropout (FedOBD) approach for industrial fault diagnostic model training. By decomposing large-scale models into semantic blocks and enabling FL participants to opportunistically upload selected important blocks in a quantized manner, it significantly reduces the communication overhead while maintaining model performance. Since its deployment in ENN Group in February 2022, FedOBD has served two coal chemical plants across two cities in China to build industrial fault prediction models. It helped the company reduce the training communication overhead by over 70% compared to its previous AI Engine, while maintaining model performance at over 85% test F1 score. To our knowledge, it is the first successfully deployed dropout-based FL approach.

----

## [1767] AmnioML: Amniotic Fluid Segmentation and Volume Prediction with Uncertainty Quantification

**Authors**: *Daniel Csillag, Lucas Monteiro Paes, Thiago Ramos, João Vitor Romano, Rodrigo Schuller, Roberto de Beauclair Seixas, Roberto I. Oliveira, Paulo Orenstein*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26837](https://doi.org/10.1609/aaai.v37i13.26837)

**Abstract**:

Accurately predicting the volume of amniotic fluid is fundamental to assessing pregnancy risks, though the task usually requires many hours of laborious work by medical experts. In this paper, we present  AmnioML, a machine learning solution that leverages deep learning and conformal prediction to output fast and accurate volume estimates and segmentation masks from fetal MRIs with Dice coefficient over 0.9. Also, we make available a novel, curated dataset for fetal MRIs with 853 exams and benchmark the performance of many recent deep learning architectures. In addition, we introduce a conformal prediction tool that yields narrow predictive intervals with theoretically guaranteed coverage, thus aiding doctors in detecting pregnancy risks and saving lives. A successful case study of AmnioML deployed in a medical setting is also reported. Real-world clinical benefits include up to 20x segmentation time reduction, with most segmentations deemed by doctors as not needing any further manual refinement. Furthermore, AmnioML's volume predictions were found to be highly accurate in practice, with mean absolute error below 56mL and tight predictive intervals, showcasing its impact in reducing pregnancy complications.

----

## [1768] A Robust and Scalable Stacked Ensemble for Day-Ahead Forecasting of Distribution Network Losses

**Authors**: *Gunnar Grotmol, Eivind Hovdegård Furdal, Nisha Dalal, Are Løkken Ottesen, Ella-Lovise Hammervold Rørvik, Martin Mølnå, Gleb Sizov, Odd Erik Gundersen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26838](https://doi.org/10.1609/aaai.v37i13.26838)

**Abstract**:

Accurate day-ahead nominations of grid losses in electrical distribution networks are important to reduce the societal cost of these losses. We present a modification of the CatBoost ensemble-based system for day-ahead grid loss prediction detailed in Dalal et al. (2020), making four main changes. Base models predict on the log-space of the target, to ensure non-negative predictions. The model ensemble is changed to include different model types, for increased ensemble variance. Feature engineering is applied to consumption and weather forecasts, to improve base model performance. Finally, a non-negative least squares-based stacking method that uses as many available models as possible for each prediction is introduced, to achieve an improved model selection that is robust to missing data.
When deployed for over three months in 2022, the resulting system reduced mean absolute error by 10.7% compared to the system from Dalal et al. (2020), a reduction from 5.05 to 4.51 MW. With no tuning of machine learning parameters, the system was also extended to three new grids, where it achieved similar relative error as on the old grids. Our system is robust and easily scalable, and our proposed stacking method could provide improved performance in applications outside grid loss.

----

## [1769] Developing the Wheel Image Similarity Application with Deep Metric Learning: Hyundai Motor Company Case

**Authors**: *Kyung-Pyo Kang, Ga Hyeon Jeong, Jeong Hoon Eom, Soon Beom Kwon, Jae Hong Park*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26839](https://doi.org/10.1609/aaai.v37i13.26839)

**Abstract**:

The global automobile market experiences quick changes in design preferences. In response to the demand shifts, manufacturers now try to apply new technologies to bring a novel design to market faster. In this paper, we introduce a novel application that performs a similarity verification task of wheel designs using an AI model and cloud computing technology. At Jan 2022, we successfully implemented the application to the wheel design process of Hyundai Motor Company’s design team and shortened the similarity verification time by 90% to a maximum of 10 minutes. We believe that this study is the first to build a wheel image database and empirically prove that the cross-entropy loss does similar tasks as the pairwise losses do in the embedding space. As a result, we successfully automated Hyundai Motor’s verification task of wheel design similarity. With a few clicks, the end-users in Hyundai Motor could take advantage of our application.

----

## [1770] Detecting VoIP Data Streams: Approaches Using Hidden Representation Learning

**Authors**: *Maya Kapoor, Michael Napolitano, Jonathan Quance, Thomas Moyer, Siddharth Krishnan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26840](https://doi.org/10.1609/aaai.v37i13.26840)

**Abstract**:

The use of voice-over-IP technology has rapidly expanded over the past several years, and has thus become a significant portion of traffic in the real, complex network environment. Deep packet inspection and middlebox technologies need to analyze call flows in order to perform network management, load-balancing, content monitoring, forensic analysis, and intelligence gathering. Because the session setup and management data can be sent on different ports or out of sync with VoIP call data over the Real-time Transport Protocol (RTP) with low latency, inspection software may miss calls or parts of calls. To solve this problem, we engineered two different deep learning models based on hidden representation learning. MAPLE, a matrix-based encoder which transforms packets into an image  representation, uses convolutional neural networks to determine RTP packets from data flow. DATE is a density-analysis based tensor encoder which transforms packet data into a three-dimensional point cloud representation. We then perform density-based clustering over the point clouds as latent representations of the data, and classify packets as RTP or non-RTP based on their statistical clustering features. In this research, we show that these tools may allow a data collection and analysis pipeline to begin detecting and buffering RTP streams for later session association, solving the initial drop problem. MAPLE achieves over ninety-nine percent accuracy in RTP/non-RTP detection. The results of our experiments show that both models can not only classify RTP versus non-RTP packet streams, but could extend to other network traffic classification problems in real deployments of network analysis pipelines.

----

## [1771] NewsPanda: Media Monitoring for Timely Conservation Action

**Authors**: *Sedrick Scott Keh, Zheyuan Ryan Shi, David J. Patterson, Nirmal Bhagabati, Karun Dewan, Areendran Gopala, Pablo Izquierdo, Debojyoti Mallick, Ambika Sharma, Pooja Shrestha, Fei Fang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26841](https://doi.org/10.1609/aaai.v37i13.26841)

**Abstract**:

Non-governmental organizations for environmental conservation have a significant interest in monitoring conservation-related media and getting timely updates about infrastructure construction projects as they may cause massive impact to key conservation areas. Such monitoring, however, is difficult and time-consuming. We introduce NewsPanda, a toolkit which automatically detects and analyzes online articles related to environmental conservation and infrastructure construction. We fine-tune a BERT-based model using active learning methods and noise correction algorithms to identify articles that are relevant to conservation and infrastructure construction. For the identified articles, we perform further analysis, extracting keywords and finding potentially related sources. NewsPanda has been successfully deployed by the World Wide Fund for Nature teams in the UK, India, and Nepal since February 2022. It currently monitors over 80,000 websites and 1,074 conservation sites across India and Nepal, saving more than 30 hours of human efforts weekly. We have now scaled it up to cover 60,000 conservation sites globally.

----

## [1772] Trustworthy Residual Vehicle Value Prediction for Auto Finance

**Authors**: *Mihye Kim, Jimyung Choi, Jaehyun Kim, Wooyoung Kim, Yeonung Baek, Gisuk Bang, Kwangwoon Son, Yeonman Ryou, Kee-Eung Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26842](https://doi.org/10.1609/aaai.v37i13.26842)

**Abstract**:

The residual value (RV) of a vehicle refers to its estimated worth at some point in the future. It is a core component in every auto financial product, used to determine the credit lines and the leasing rates. As such, an accurate prediction of RV is critical for the auto finance industry, since it can pose a risk of revenue loss by over-prediction or make the financial product incompetent by under-prediction. Although there are a number of prior studies on training machine learning models on a large amount of used car sales data, we had to cope with real-world operational requirements such as compliance with regulations (i.e. monotonicity of output with respect to a subset of features) and generalization to unseen input (i.e. new and rare car models). In this paper, we describe how we coped with these practical challenges and created value for our business at Hyundai Capital Services, the top auto financial service provider in Korea.

----

## [1773] A Dataset and Baseline Approach for Identifying Usage States from Non-intrusive Power Sensing with MiDAS IoT-Based Sensors

**Authors**: *Bharath Muppasani, Cheyyur Jaya Anand, Chinmayi Appajigowda, Biplav Srivastava, Lokesh Johri*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26843](https://doi.org/10.1609/aaai.v37i13.26843)

**Abstract**:

The state identification problem seeks to identify power usage patterns of any system, like buildings or factories, of interest. In this challenge paper, we make power usage dataset available from 8 institutions in manufacturing, education and medical institutions from the US and India, and an initial unsupervised machine learning based solution as a baseline for the community to accelerate research in this area.

----

## [1774] Real-Time Detection of Robotic Traffic in Online Advertising

**Authors**: *Anand Muralidhar, Sharad Chitlangia, Rajat Agarwal, Muneeb Ahmed*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26844](https://doi.org/10.1609/aaai.v37i13.26844)

**Abstract**:

Detecting robotic traffic at scale on online ads needs an approach that is scalable, comprehensive, precise, and can rapidly respond to changing traffic patterns. In this paper we describe SLIDR or SLIce-Level Detection of Robots, a real-time deep neural network model trained with weak supervision to identify invalid clicks on online ads. We ensure fairness across different traffic slices by formulating a convex optimization problem that allows SLIDR to achieve optimal performance on individual traffic slices with a budget on overall false positives. SLIDR has been deployed since 2021 and safeguards advertiser campaigns on Amazon against robots clicking on ads on the e-commerce site. We describe some of the important lessons learned by deploying SLIDR that include guardrails that prevent updates of anomalous models and disaster recovery mechanisms to mitigate or correct decisions made by a faulty model.

----

## [1775] Dynamic Pricing with Volume Discounts in Online Settings

**Authors**: *Marco Mussi, Gianmarco Genalti, Alessandro Nuara, Francesco Trovò, Marcello Restelli, Nicola Gatti*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26845](https://doi.org/10.1609/aaai.v37i13.26845)

**Abstract**:

According to the main international reports, more pervasive industrial and business-process automation, thanks to machine learning and advanced analytic tools, will unlock more than 14 trillion USD worldwide annually by 2030. In the specific case of pricing problems, which constitute the class of problems we investigate in this paper, the estimated unlocked value will be about 0.5 trillion USD per year. In particular, this paper focuses on pricing in e-commerce when the objective function is profit maximization and only transaction data are available. This setting is one of the most common in real-world applications. Our work aims to find a pricing strategy that allows defining optimal prices at different volume thresholds to serve different classes of users. Furthermore, we face the major challenge, common in real-world settings, of dealing with limited data available. We design a two-phase online learning algorithm, namely PVD-B, capable of exploiting the data incrementally in an online fashion. The algorithm first estimates the demand curve and retrieves the optimal average price, and subsequently it offers discounts to differentiate the prices for each volume threshold. We ran a real-world 4-month-long A/B testing experiment in collaboration with an Italian e-commerce company, in which our algorithm PVD-B - corresponding to A configuration - has been compared with human pricing specialists - corresponding to B configuration. At the end of the experiment, our algorithm produced a total turnover of about 300 KEuros, outperforming the B configuration performance by about 55%. The Italian company we collaborated with decided to adopt our algorithm for more than 1,200 products since January 2022.

----

## [1776] An Explainable Forecasting System for Humanitarian Needs Assessment

**Authors**: *Rahul Nair, Bo Schwartz Madsen, Alexander Kjærum*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26846](https://doi.org/10.1609/aaai.v37i13.26846)

**Abstract**:

We present a machine learning system for forecasting forced displacement populations deployed at the Danish Refugee Council (DRC). The system, named Foresight, supports long term forecasts aimed at humanitarian response planning. It is explainable, providing evidence and context supporting the forecast. Additionally, it supports scenarios, whereby analysts are able to generate forecasts under alternative conditions. The system has been in deployment since early 2020 and powers several downstream business functions within DRC. It is central to our annual Global Displacement Report which informs our response planning. We describe the system, key outcomes, lessons learnt, along with technical limitations and challenges in deploying machine learning systems in the humanitarian sector.

----

## [1777] Industry-Scale Orchestrated Federated Learning for Drug Discovery

**Authors**: *Martijn Oldenhof, Gergely Ács, Balázs Pejó, Ansgar Schuffenhauer, Nicholas Holway, Noé Sturm, Arne Dieckmann, Oliver Fortmeier, Eric Boniface, Clément Mayer, Arnaud Gohier, Peter Schmidtke, Ritsuya Niwayama, Dieter Kopecky, Lewis H. Mervin, Prakash Chandra Rathi, Lukas Friedrich, András Formanek, Peter Antal, Jordon Rahaman, Adam Zalewski, Wouter Heyndrickx, Ezron Oluoch, Manuel Stößel, Michal Vanco, David Endico, Fabien Gelus, Thaïs de Boisfossé, Adrien Darbier, Ashley Nicollet, Matthieu Blottière, Maria Telenczuk, Van Tien Nguyen, Thibaud Martinez, Camille Boillet, Kelvin Moutet, Alexandre Picosson, Aurélien Gasser, Inal Djafar, Antoine Simon, Adam Arany, Jaak Simm, Yves Moreau, Ola Engkvist, Hugo Ceulemans, Camille Marini, Mathieu Galtier*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26847](https://doi.org/10.1609/aaai.v37i13.26847)

**Abstract**:

To apply federated learning to drug discovery we developed a novel platform in the context of European Innovative Medicines Initiative (IMI) project MELLODDY (grant n°831472), which was comprised of 10 pharmaceutical companies, academic research labs, large industrial companies and startups. The MELLODDY platform was the first industry-scale platform to enable the creation of a global federated model for drug discovery without sharing the confidential data sets of the individual partners. The federated model was trained on the platform by aggregating the gradients of all contributing partners in a cryptographic, secure way following each training iteration. The platform was deployed on an Amazon Web Services (AWS) multi-account architecture running Kubernetes clusters in private subnets. Organisationally, the roles of the different partners were codified as different rights and permissions on the platform and administrated in a decentralized way. The MELLODDY platform generated new scientific discoveries which are described in a companion paper.

----

## [1778] THMA: Tencent HD Map AI System for Creating HD Map Annotations

**Authors**: *Kun Tang, Xu Cao, Zhipeng Cao, Tong Zhou, Erlong Li, Ao Liu, Shengtao Zou, Chang Liu, Shuqi Mei, Elena Sizikova, Chao Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26848](https://doi.org/10.1609/aaai.v37i13.26848)

**Abstract**:

Nowadays, autonomous vehicle technology is becoming more and more mature. Critical to progress and safety, high-definition (HD) maps, a type of centimeter-level map collected using a laser sensor, provide accurate descriptions of the surrounding environment. The key challenge of HD map production is efficient, high-quality collection and annotation of large-volume datasets. Due to the demand for high quality, HD map production requires significant manual human effort to create annotations, a very time-consuming and costly process for the map industry. In order to reduce manual annotation burdens, many artificial intelligence (AI) algorithms have been developed to pre-label the HD maps. However, there still exists a large gap between AI algorithms and the traditional manual HD map production pipelines in accuracy and robustness. Furthermore, it is also very resource-costly to build large-scale annotated datasets and advanced machine learning algorithms for AI-based HD map automatic labeling systems. In this paper, we introduce the Tencent HD Map AI (THMA) system, an innovative end-to-end, AI-based, active learning HD map labeling system capable of producing and labeling HD maps with a scale of hundreds of thousands of kilometers. In THMA, we train AI models directly from massive HD map datasets via supervised, self-supervised, and weakly supervised learning to achieve high accuracy and efficiency required by downstream users. THMA has been deployed by the Tencent Map team to provide services to downstream companies and users, serving over 1,000 labeling workers and producing more than 30,000 kilometers of HD map data per day at most. More than 90 percent of the HD map data in Tencent Map is labeled automatically by THMA, accelerating the traditional HD map labeling process by more than ten times.

----

## [1779] Increasing Impact of Mobile Health Programs: SAHELI for Maternal and Child Care

**Authors**: *Shresth Verma, Gargi Singh, Aditya Mate, Paritosh Verma, Sruthi Gorantla, Neha Madhiwalla, Aparna Hegde, Divy Thakkar, Manish Jain, Milind Tambe, Aparna Taneja*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26849](https://doi.org/10.1609/aaai.v37i13.26849)

**Abstract**:

Underserved communities face critical health challenges due to lack of access to timely and reliable information. Nongovernmental organizations are leveraging the widespread use of cellphones to combat these healthcare challenges and spread preventative awareness. The health workers at these organizations reach out individually to beneficiaries; however such programs still suffer from declining engagement. We have deployed SAHELI, a system to efficiently utilize the limited availability of health workers for improving maternal and child health in India. SAHELI uses the Restless Multiarmed Bandit (RMAB) framework to identify beneficiaries for outreach. It is the first deployed application for RMABs in public health, and is already in continuous use by our partner NGO, ARMMAN. We have already reached ~100K beneficiaries with SAHELI, and are on track to serve 1 million beneficiaries by the end of 2023. This scale and impact has been achieved through multiple innovations in the RMAB model and its development, in preparation of real world data, and in deployment practices; and through careful consideration of responsible AI practices. Specifically, in this paper, we describe our approach to learn from past data to improve the performance of SAHELI’s RMAB model, the real-world challenges faced during deployment and adoption of SAHELI, and
the end-to-end pipeline.

----

## [1780] MuMIC - Multimodal Embedding for Multi-Label Image Classification with Tempered Sigmoid

**Authors**: *Fengjun Wang, Sarai Mizrachi, Moran Beladev, Guy Nadav, Gil Amsalem, Karen Lastmann Assaraf, Hadas Harush Boker*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26850](https://doi.org/10.1609/aaai.v37i13.26850)

**Abstract**:

Multi-label image classification is a foundational topic in various domains. Multimodal learning approaches have recently achieved outstanding results in image representation and single-label image classification. For instance, Contrastive Language-Image Pretraining (CLIP) demonstrates impressive image-text representation learning abilities and is robust to natural distribution shifts. This success inspires us to leverage multimodal learning for multi-label classification tasks, and benefit from contrastively learnt pretrained models.

We propose the Multimodal Multi-label Image Classification (MuMIC) framework, which utilizes a hardness-aware tempered sigmoid based Binary Cross Entropy loss function, thus enables the optimization on multi-label objectives and transfer learning on CLIP. MuMIC is capable of providing high classification performance, handling real-world noisy data, supporting zero-shot predictions, and producing domain-specific image embeddings.

In this study, a total of 120 image classes are defined, and more than 140K positive annotations are collected on approximately 60K Booking.com images. The final MuMIC model is deployed on Booking.com Content Intelligence Platform, and it outperforms other state-of-the-art models with 85.6% GAP@10 and 83.8% GAP on all 120 classes, as well as a 90.1% macro mAP score across 32 majority classes. We summarize the modelling choices which are extensively tested through ablation studies. To the best of our knowledge, we are the first to adapt contrastively learnt multimodal pretraining for real-world multi-label image classification problems, and the innovation can be transferred to other domains.

----

## [1781] OPRADI: Applying Security Game to Fight Drive under the Influence in Real-World

**Authors**: *Luzhan Yuan, Wei Wang, Gaowei Zhang, Yi Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26851](https://doi.org/10.1609/aaai.v37i13.26851)

**Abstract**:

Driving under the influence (DUI) is one of the main causes of traffic accidents, often leading to severe life and property losses. Setting up sobriety checkpoints on certain roads is the most commonly used practice to identify DUI-drivers in many countries worldwide. However, setting up checkpoints according to the police's experiences may not be effective for ignoring the strategic interactions between the police and DUI-drivers, particularly when inspecting resources are limited. To remedy this situation, we adapt the classic Stackelberg security game (SSG) to a new SSG-DUI game to describe the strategic interactions in catching DUI-drivers. SSG-DUI features drivers' bounded rationality and social knowledge sharing among them, thus realizing improved real-world fidelity. With SSG-DUI, we propose OPRADI, a systematic approach for advising better strategies in setting up checkpoints. We perform extensive experiments to evaluate it in both simulated environments and real-world contexts, in collaborating with a Chinese city's police bureau. The results reveal its effectiveness in improving police's real-world operations, thus having significant practical potentials.

----

## [1782] AHPA: Adaptive Horizontal Pod Autoscaling Systems on Alibaba Cloud Container Service for Kubernetes

**Authors**: *Zhiqiang Zhou, Chaoli Zhang, Lingna Ma, Jing Gu, Huajie Qian, Qingsong Wen, Liang Sun, Peng Li, Zhimin Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26852](https://doi.org/10.1609/aaai.v37i13.26852)

**Abstract**:

The existing resource allocation policy for application instances in Kubernetes cannot dynamically adjust according to the requirement of business, which would cause an enormous waste of resources during fluctuations. Moreover, the emergence of new cloud services puts higher resource management requirements. This paper discusses horizontal POD resources management in Alibaba Cloud Container Services with a newly deployed AI algorithm framework named AHPA - the adaptive horizontal pod auto-scaling system. Based on a robust decomposition forecasting algorithm and performance training model, AHPA offers an optimal pod number adjustment plan that could reduce POD resources
and maintain business stability. Since being deployed in April 2021, this system has expanded to multiple customer scenarios, including logistics, social networks, AI audio and video, e-commerce, etc. Compared with the previous algorithms, AHPA solves the elastic lag problem, increasing CPU usage by 10% and reducing resource cost by more than 20%. In addition, AHPA can automatically perform flexible planning according to the predicted business volume without manual intervention, significantly saving operation and maintenance costs.

----

## [1783] eForecaster: Unifying Electricity Forecasting with Robust, Flexible, and Explainable Machine Learning Algorithms

**Authors**: *Zhaoyang Zhu, Weiqi Chen, Rui Xia, Tian Zhou, Peisong Niu, Bingqing Peng, Wenwei Wang, Hengbo Liu, Ziqing Ma, Qingsong Wen, Liang Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26853](https://doi.org/10.1609/aaai.v37i13.26853)

**Abstract**:

Electricity forecasting is crucial in scheduling and planning of future electric load, so as to improve the reliability and safeness of the power grid. Despite recent developments of forecasting algorithms in the machine learning community, there is a lack of general and advanced algorithms specifically considering requirements from the power industry perspective. In this paper, we present eForecaster, a unified AI platform including robust, flexible, and explainable machine learning algorithms for diversified electricity forecasting applications. Since Oct. 2021, multiple commercial bus load, system load, and renewable energy forecasting systems built upon eForecaster have been deployed in seven provinces of China. The deployed systems consistently reduce the average Mean Absolute Error (MAE) by 39.8% to 77.0%, with reduced manual work and explainable guidance. In particular, eForecaster also integrates multiple interpretation methods to uncover the working mechanism of the predictive models, which significantly improves forecasts adoption and user satisfaction.

----

## [1784] Cosmic Microwave Background Recovery: A Graph-Based Bayesian Convolutional Network Approach

**Authors**: *Jadie Adams, Steven Lu, Krzysztof M. Gorski, Graca Rocha, Kiri L. Wagstaff*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26854](https://doi.org/10.1609/aaai.v37i13.26854)

**Abstract**:

The cosmic microwave background (CMB) is a significant source of knowledge about the origin and evolution of our universe. However, observations of the CMB are contaminated by foreground emissions, obscuring the CMB signal and reducing its efficacy in constraining cosmological parameters. We employ deep learning as a data-driven approach to CMB cleaning from multi-frequency full-sky maps. In particular, we develop a graph-based Bayesian convolutional neural network based on the U-Net architecture that predicts cleaned CMB with pixel-wise uncertainty estimates. We demonstrate the potential of this technique on realistic simulated data based on the Planck mission. We show that our model ac- accurately recovers the cleaned CMB sky map and resulting angular power spectrum while identifying regions of uncertainty. Finally, we discuss the current challenges and the path forward for deploying our model for CMB recovery on real observations.

----

## [1785] Phase-Informed Bayesian Ensemble Models Improve Performance of COVID-19 Forecasts

**Authors**: *Aniruddha Adiga, Gursharn Kaur, Lijing Wang, Benjamin Hurt, Przemyslaw J. Porebski, Srinivasan Venkatramanan, Bryan L. Lewis, Madhav V. Marathe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26855](https://doi.org/10.1609/aaai.v37i13.26855)

**Abstract**:

Despite hundreds of methods published in the literature, forecasting epidemic dynamics remains challenging yet important. The challenges stem from multiple sources, including: the need for timely data, co-evolution of epidemic dynamics with behavioral and immunological adaptations, and the evolution of new pathogen strains. The ongoing COVID-19 pandemic highlighted these challenges; in an important article, Reich et al. did a comprehensive analysis highlighting many of these challenges.

In this paper, we take another step in critically evaluating existing epidemic forecasting methods. Our methods are based on a simple yet crucial observation - epidemic dynamics go through a number of phases (waves). Armed with this understanding, we propose a modification to our deployed Bayesian ensembling case time series forecasting framework. We show that ensembling methods employing the phase information and using different weighting schemes for each phase can produce improved forecasts. We evaluate our proposed method with both the currently deployed model and the COVID-19 forecasthub models. The overall performance of the proposed model is consistent across the pandemic but more importantly, it is ranked third and first during two critical rapid growth phases in cases, regimes where the performance of most models from the CDC forecasting hub dropped significantly.

----

## [1786] Towards Hybrid Automation by Bootstrapping Conversational Interfaces for IT Operation Tasks

**Authors**: *Jayachandu Bandlamudi, Kushal Mukherjee, Prerna Agarwal, Sampath Dechu, Siyu Huo, Vatche Isahagian, Vinod Muthusamy, Naveen Purushothaman, Renuka Sindhgatta*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26856](https://doi.org/10.1609/aaai.v37i13.26856)

**Abstract**:

Process automation has evolved from end-to-end automation of repetitive process branches to hybrid automation where bots perform some activities and humans serve other activities. In the context of knowledge-intensive processes such as IT operations, implementing hybrid automation is a natural choice where robots can perform certain mundane functions, with humans taking over the decision of when and which IT systems need to act. Recently, ChatOps, which refers to conversation-driven collaboration for IT operations, has rapidly accelerated efficiency by providing a cross-organization and cross-domain platform to resolve and manage issues as soon as possible. Hence, providing a natural language interface to bots is a logical progression to enable collaboration between humans and bots. This work presents a no-code approach to provide a conversational interface that enables human workers to collaborate with bots executing automation scripts. The bots identify the intent of users' requests and automatically orchestrate one or more relevant automation tasks to serve the request. We further detail our process of mining the conversations between humans and bots to monitor performance and identify the scope for improvement in service quality.

----

## [1787] Compressing Cross-Lingual Multi-Task Models at Qualtrics

**Authors**: *Daniel Campos, Daniel J. Perry, Samir Joshi, Yashmeet Gambhir, Wei Du, Zhengzheng Xing, Aaron Colak*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26857](https://doi.org/10.1609/aaai.v37i13.26857)

**Abstract**:

Experience management is an emerging business area where organizations focus on understanding the feedback of customers and employees in order to improve their end-to-end experiences.
This results in a unique set of machine learning problems to help understand how people feel, discover issues they care about, and find which actions need to be taken on data that are different in content and distribution from traditional NLP domains.
In this paper, we present a case study of building text analysis applications that perform multiple classification tasks efficiently in 12 languages in the nascent business area of experience management.
In order to scale up modern ML methods on experience data, we leverage cross lingual and multi-task modeling techniques to consolidate our models into a single deployment to avoid overhead.
We also make use of model compression and model distillation to reduce overall inference latency and hardware cost to the level acceptable for business needs while maintaining model prediction quality.
Our findings show that multi-task modeling improves task performance for a subset of experience management tasks in both XLM-R and mBert architectures.
Among the compressed architectures we explored, we found that MiniLM achieved the best compression/performance tradeoff.
Our case study demonstrates a speedup of up to 15.61x with 2.60% average task degradation (or 3.29x speedup with 1.71% degradation) and estimated savings of 44% over using the original full-size model.
These results demonstrate a successful scaling up of text classification for the challenging new area of ML for experience management.

----

## [1788] SolderNet: Towards Trustworthy Visual Inspection of Solder Joints in Electronics Manufacturing Using Explainable Artificial Intelligence

**Authors**: *Hayden Gunraj, Paul Guerrier, Sheldon Fernandez, Alexander Wong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26858](https://doi.org/10.1609/aaai.v37i13.26858)

**Abstract**:

In electronics manufacturing, solder joint defects are a common problem affecting a variety of printed circuit board components. To identify and correct solder joint defects, the solder joints on a circuit board are typically inspected manually by trained human inspectors, which is a very time-consuming and error-prone process. To improve both inspection efficiency and accuracy, in this work we describe an explainable deep learning-based visual quality inspection system tailored for visual inspection of solder joints in electronics manufacturing environments. At the core of this system is an explainable solder joint defect identification system called SolderNet which we design and implement with trust and transparency in mind. While several challenges remain before the full system can be developed and deployed, this study presents important progress towards trustworthy visual inspection of solder joints in electronics manufacturing.

----

## [1789] MobilePTX: Sparse Coding for Pneumothorax Detection Given Limited Training Examples

**Authors**: *Darryl Hannan, Steven C. Nesbit, Ximing Wen, Glen Smith, Qiao Zhang, Alberto Goffi, Vincent Chan, Michael J. Morris, John C. Hunninghake, Nicholas E. Villalobos, Edward Kim, Rosina O. Weber, Christopher J. MacLellan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26859](https://doi.org/10.1609/aaai.v37i13.26859)

**Abstract**:

Point-of-Care Ultrasound (POCUS) refers to clinician-performed and interpreted ultrasonography at the patient's bedside. Interpreting these images requires a high level of expertise, which may not be available during emergencies. In this paper, we support POCUS by developing classifiers that can aid medical professionals by diagnosing whether or not a patient has pneumothorax. We decomposed the task into multiple steps, using YOLOv4 to extract relevant regions of the video and a 3D sparse coding model to represent video features. Given the difficulty in acquiring positive training videos, we trained a small-data classifier with a maximum of 15 positive and 32 negative examples. To counteract this limitation, we leveraged subject matter expert (SME) knowledge to limit the hypothesis space, thus reducing the cost of data collection. We present results using two lung ultrasound datasets and demonstrate that our model is capable of achieving performance on par with SMEs in pneumothorax identification. We then developed an iOS application that runs our full system in less than 4 seconds on an iPad Pro, and less than 8 seconds on an iPhone 13 Pro, labeling key regions in the lung sonogram to provide interpretable diagnoses.

----

## [1790] Vessel-to-Vessel Motion Compensation with Reinforcement Learning

**Authors**: *Sverre Herland, Kerstin Bach*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26860](https://doi.org/10.1609/aaai.v37i13.26860)

**Abstract**:

Actuation delay poses a challenge for robotic arms and cranes. This is especially the case in dynamic environments where the robot arm or the objects it is trying to manipulate are moved by exogenous forces. In this paper, we consider the task of using a robotic arm to compensate for relative motion between two vessels at sea. We construct a hybrid controller that combines an Inverse Kinematic (IK) solver with a Reinforcement Learning (RL) agent that issues small corrections to the IK input. The solution is empirically evaluated in a simulated environment under several sea states and actuation delays. We observe that more intense waves and larger actuation delays have an adverse effect on the IK controller's ability to compensate for vessel motion. The RL agent is shown to be effective at mitigating large parts of these errors, both in the average case and in the worst case. Its modest requirement for sensory information, combined with the inherent safety in only making small adjustments, also makes it a promising approach for real-world deployment.

----

## [1791] Intuitive Access to Smartphone Settings Using Relevance Model Trained by Contrastive Learning

**Authors**: *Joonyoung Kim, Kangwook Lee, Haebin Shin, Hurnjoo Lee, Sechun Kang, Byunguk Choi, Dong Shin, Joohyung Lee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26861](https://doi.org/10.1609/aaai.v37i13.26861)

**Abstract**:

The more new features that are being added to smartphones, the harder it becomes for users to find them. This is because the feature names are usually short and there are just too many of them for the users to remember the exact words.  The users are more comfortable asking contextual queries that describe the features they are looking for, but the standard term frequency-based search cannot process them. This paper presents a novel retrieval system for mobile features that accepts intuitive and contextual search queries. We trained a relevance model via contrastive learning from a pre-trained language model to perceive the contextual relevance between a query embedding and indexed mobile features. Also, to make it efficiently run on-device using minimal resources, we applied knowledge distillation to compress the model without degrading much performance. To verify the feasibility of our method, we collected test queries and conducted comparative experiments with the currently deployed search baselines. The results show that our system outperforms the others on contextual sentence queries and even on usual keyword-based queries.

----

## [1792] Towards Safe Mechanical Ventilation Treatment Using Deep Offline Reinforcement Learning

**Authors**: *Flemming Kondrup, Thomas Jiralerspong, Elaine Lau, Nathan de Lara, Jacob Shkrob, My Duc Tran, Doina Precup, Sumana Basu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26862](https://doi.org/10.1609/aaai.v37i13.26862)

**Abstract**:

Mechanical ventilation is a key form of life support for patients with pulmonary impairment. Healthcare workers are required to continuously adjust ventilator settings for each patient, a challenging and time consuming task. Hence, it would be beneficial to develop an automated decision support tool to optimize ventilation treatment. We present DeepVent, a Conservative Q-Learning (CQL) based offline Deep Reinforcement Learning (DRL) agent that learns to predict the optimal ventilator parameters for a patient to promote 90 day survival. We design a clinically relevant intermediate reward that encourages continuous improvement of the patient vitals as well as addresses the challenge of sparse reward in RL. We find that DeepVent recommends ventilation parameters within safe ranges, as outlined in recent clinical trials. The CQL algorithm offers additional safety by mitigating the overestimation of the value estimates of out-of-distribution states/actions. We evaluate our agent using Fitted Q Evaluation (FQE) and demonstrate that it outperforms physicians from the MIMIC-III dataset.

----

## [1793] Data-Driven Machine Learning Models for a Multi-Objective Flapping Fin Unmanned Underwater Vehicle Control System

**Authors**: *Julian Lee, Kamal Viswanath, Alisha Sharma, Jason Geder, Marius Pruessner, Brian Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26863](https://doi.org/10.1609/aaai.v37i13.26863)

**Abstract**:

Flapping-fin unmanned underwater vehicle (UUV) propulsion systems provide high maneuverability for naval tasks such as surveillance and terrain exploration. Recent work has explored the use of time-series neural network surrogate models to predict thrust from vehicle design and fin kinematics. We develop a search-based inverse model that leverages a kinematics-to-thrust neural network model for control system design. Our inverse model finds a set of fin kinematics with the multi-objective goal of reaching a target thrust and creating a smooth kinematic transition between flapping cycles. We demonstrate how a control system integrating this inverse model can make online, cycle-to-cycle adjustments to prioritize different system objectives.

----

## [1794] AnimateSVG: Autonomous Creation and Aesthetics Evaluation of Scalable Vector Graphics Animations for the Case of Brand Logos

**Authors**: *Deborah Mateja, Rebecca Armbruster, Jonathan Baumert, Tim Bleil, Jakob Langenbahn, Jan Christian Schwedhelm, Sarah Sester, Armin Heinzl*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26864](https://doi.org/10.1609/aaai.v37i13.26864)

**Abstract**:

In the light of the constant battle for attention on digital media, animating digital content plays an increasing role in modern graphic design. In this study, we use artificial intelligence methods to create aesthetic animations along the case of brand logos. With scalable vector graphics as the standard format in modern graphic design, we develop an autonomous end-to-end method using complex machine learning techniques to create brand logo animations as scalable vector graphics from scratch. We acquire data and setup a comprehensive animation space to create novel animations and evaluate them based on their aesthetics. We propose and compare two alternative computational models for automated logo animation and carefully weigh up their idiosyncrasies: on the one hand, we set up an aesthetics evaluation model to train an animation generator and, on the other hand, we combine tree ensembles with global optimization. Indeed, our proposed methods are capable of creating aesthetic logo animations, receiving an average rating of ‘good’ from observers.

----

## [1795] Grape Cold Hardiness Prediction via Multi-Task Learning

**Authors**: *Aseem Saxena, Paola Pesantez-Cabrera, Rohan Ballapragada, Kin-Ho Lam, Markus Keller, Alan Fern*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26865](https://doi.org/10.1609/aaai.v37i13.26865)

**Abstract**:

Cold temperatures during fall and spring have the potential to cause frost damage to grapevines and other fruit plants, which can significantly decrease harvest yields. To help prevent these losses, farmers deploy expensive frost mitigation measures, such as, sprinklers, heaters, and wind machines, when they judge that damage may occur. This judgment, however, is challenging because the cold hardiness of plants changes throughout the dormancy period and it is difficult to directly measure. This has led scientists to develop cold hardiness prediction models that can be tuned to different grape cultivars based on laborious field measurement data. In this paper, we study whether deep-learning models can improve cold hardiness prediction for grapes based on data that has been collected over a 30-year time period. A key challenge is that the amount of data per cultivar is highly variable, with some cultivars having only a small amount. For this purpose, we investigate the use of multi-task learning to leverage data across cultivars in order to improve prediction performance for individual cultivars. We evaluate a number of multi-task learning approaches and show that the highest performing approach is able to significantly improve over learning for single cultivars and outperforms the current state-of-the-art scientific model for most cultivars.

----

## [1796] Reward Design for an Online Reinforcement Learning Algorithm Supporting Oral Self-Care

**Authors**: *Anna L. Trella, Kelly W. Zhang, Inbal Nahum-Shani, Vivek Shetty, Finale Doshi-Velez, Susan A. Murphy*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26866](https://doi.org/10.1609/aaai.v37i13.26866)

**Abstract**:

While dental disease is largely preventable, professional advice on optimal oral hygiene practices is often forgotten or abandoned by patients. Therefore patients may benefit from timely and personalized encouragement to engage in oral self-care behaviors. In this paper, we develop an online reinforcement learning (RL) algorithm for use in optimizing the delivery of mobile-based prompts to encourage oral hygiene behaviors. One of the main challenges in developing such an algorithm is ensuring that the algorithm considers the impact of current actions on the effectiveness of future actions (i.e., delayed effects), especially when the algorithm has been designed to run stably and autonomously in a constrained, real-world setting characterized by highly noisy, sparse data. We address this challenge by designing a quality reward that maximizes the desired health outcome (i.e., high-quality brushing) while minimizing user burden. We also highlight a procedure for optimizing the hyperparameters of the reward by building a simulation environment test bed and evaluating candidates using the test bed. The RL algorithm discussed in this paper will be deployed in Oralytics. To the best of our knowledge, Oralytics is the first mobile health study utilizing an RL algorithm designed to prevent dental disease by optimizing the delivery of motivational messages supporting oral self-care behaviors.

----

## [1797] Embedding a Long Short-Term Memory Network in a Constraint Programming Framework for Tomato Greenhouse Optimisation

**Authors**: *Dirk van Bokkem, Max van den Hemel, Sebastijan Dumancic, Neil Yorke-Smith*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26867](https://doi.org/10.1609/aaai.v37i13.26867)

**Abstract**:

Increasing global food demand, accompanied by the limited number of expert growers, brings the need for more sustainable and efficient horticulture. The controlled environment of greenhouses enable data collection and precise control. For optimally controlling the greenhouse climate, a grower not only looks at crop production, but rather aims at maximising the profit. However this is a complex, long term optimisation task. In this paper, Constraint Programming (CP) is applied to task of optimal greenhouse economic control, by leveraging a learned greenhouse climate model through a CP embedding. In collaboration with an industrial partner, we demonstrate how to model the greenhouse climate with an LSTM model, embed this LSTM into a CP optimisation framework, and optimise the expected profit of the grower. This data-to-decision pipeline is being integrated into a decision support system for multiple greenhouses in the Netherlands.

----

## [1798] Fault Injection Based Interventional Causal Learning for Distributed Applications

**Authors**: *Qing Wang, Jesus Rios, Saurabh Jha, Karthikeyan Shanmugam, Frank Bagehorn, Xi Yang, Robert Filepp, Naoki Abe, Larisa Shwartz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26868](https://doi.org/10.1609/aaai.v37i13.26868)

**Abstract**:

We apply the machinery of interventional causal learning with programmable interventions to the domain of applications management. Modern applications are modularized into interdependent components or services (e.g. microservices) for ease of development and management. The communication graph among such components is a function of application code and is not always known to the platform provider. In our solution we learn this unknown communication graph solely using application logs observed during the execution of the application by using fault injections in a staging environment. Specifically, we have developed an active (or interventional) causal learning algorithm that uses the observations obtained during fault injections to learn a model of error propagation in the communication among the components. The “power of intervention” additionally allows us to address the presence of confounders in unobserved user interactions. We demonstrate the effectiveness of our solution in learning the communication graph of well-known microservice application benchmarks. We also show the efficacy of the solution on a downstream task of fault localization in which the learned graph indeed helps to localize faults at runtime in a production environment (in which the location of the fault is unknown). Additionally, we briefly discuss the implementation and deployment status of a fault injection framework which incorporates the developed technology.

----

## [1799] High-Throughput, High-Performance Deep Learning-Driven Light Guide Plate Surface Visual Quality Inspection Tailored for Real-World Manufacturing Environments

**Authors**: *Carol Xu, Mahmoud Famouri, Gautam Bathla, Mohammad Javad Shafiee, Alexander Wong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26869](https://doi.org/10.1609/aaai.v37i13.26869)

**Abstract**:

Light guide plates are essential optical components widely used in a diverse range of applications ranging from medical lighting fixtures to back-lit TV displays. An essential step in the manufacturing of light guide plates is the quality inspection of defects such as scratches, bright/dark spots, and impurities. This is mainly done in industry through manual visual inspection for plate pattern irregularities, which is time-consuming and prone to human error and thus act as a significant barrier to high-throughput production. Advances in deep learning-driven computer vision has led to the exploration of automated visual quality inspection of light guide plates to improve inspection consistency, accuracy, and efficiency. However, given the computational constraints and high-throughput nature of real-world manufacturing environments, the widespread adoption of deep learning-driven visual inspection systems for inspecting light guide plates in real-world manufacturing environments has been greatly limited due to high computational requirements and integration challenges of existing deep learning approaches in research literature. In this work, we introduce a fully-integrated, high-throughput, high-performance deep learning-driven workflow for light guide plate surface visual quality inspection (VQI) tailored for real-world manufacturing environments. To enable automated VQI on the edge computing within the fully-integrated VQI system, a highly compact deep anti-aliased attention condenser neural network (which we name Light-DefectNet) tailored specifically for light guide plate surface defect detection in resource-constrained scenarios was created via machine-driven design exploration with computational and “best-practices” constraints as well as L1 paired classification discrepancy loss. Experiments show that Light-DetectNet achieves a detection accuracy of ∼98.2% on the LGPSDD benchmark while having just 770K parameters
(∼33× and ∼6.9× lower than ResNet-50 and EfficientNet-B0, respectively) and ∼93M FLOPs (∼88× and ∼8.4× lower than ResNet-50 and EfficientNet-B0, respectively) and ∼8.8× faster inference speed than EfficientNet-B0 on an embedded ARM processor. As such, the proposed deep learning-driven workflow, integrated with the aforementioned LightDefectNet neural network, is highly suited for high-throughput, high-performance light plate surface VQI within real-world manufacturing environments.

----



[Go to the previous page](AAAI-2023-list08.md)

[Go to the next page](AAAI-2023-list10.md)

[Go to the catalog section](README.md)