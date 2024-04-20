## [0] Back to the Future: Toward a Hybrid Architecture for Ad Hoc Teamwork

**Authors**: *Hasra Dodampegama, Mohan Sridharan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25070](https://doi.org/10.1609/aaai.v37i1.25070)

**Abstract**:

State of the art methods for ad hoc teamwork, i.e., for collaboration without prior coordination, often use a long history of prior observations to model the behavior of other agents (or agent types) and to determine the ad hoc agent's behavior. In many practical domains, it is difficult to obtain large training datasets, and necessary to quickly revise the existing models to account for changes in team composition or domain attributes. Our architecture builds on the principles of step-wise refinement and ecological rationality to enable an ad hoc agent to perform non-monotonic logical reasoning with prior commonsense domain knowledge and models learned rapidly from limited examples to predict the behavior of other agents. In the simulated multiagent collaboration domain Fort Attack, we experimentally demonstrate that our architecture enables an ad hoc agent to adapt to changes in the behavior of other agents, and provides enhanced transparency and better performance than a state of the art data-driven baseline.

----

## [1] Reducing ANN-SNN Conversion Error through Residual Membrane Potential

**Authors**: *Zecheng Hao, Tong Bu, Jianhao Ding, Tiejun Huang, Zhaofei Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25071](https://doi.org/10.1609/aaai.v37i1.25071)

**Abstract**:

Spiking Neural Networks (SNNs) have received extensive academic attention due to the unique properties of low power consumption and high-speed computing on neuromorphic chips. Among various training methods of SNNs, ANN-SNN conversion has shown the equivalent level of performance as ANNs on large-scale datasets. However, unevenness error, which refers to the deviation caused by different temporal sequences of spike arrival on activation layers, has not been effectively resolved and seriously suffers the performance of SNNs under the condition of short time-steps. In this paper, we make a detailed analysis of unevenness error and divide it into four categories. We point out that the case of the ANN output being zero while the SNN output being larger than zero accounts for the largest percentage. Based on this, we theoretically prove the sufficient and necessary conditions of this case and propose an optimization strategy based on residual membrane potential to reduce unevenness error. The experimental results show that the proposed method achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet datasets. For example, we reach top-1 accuracy of 64.32% on ImageNet with 10-steps. To the best of our knowledge, this is the first time ANN-SNN conversion can simultaneously achieve high accuracy and ultra-low-latency on the complex dataset. Code is available at https://github.com/hzc1208/ANN2SNN_SRP.

----

## [2] Hierarchical ConViT with Attention-Based Relational Reasoner for Visual Analogical Reasoning

**Authors**: *Wentao He, Jialu Zhang, Jianfeng Ren, Ruibin Bai, Xudong Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25072](https://doi.org/10.1609/aaai.v37i1.25072)

**Abstract**:

Raven’s Progressive Matrices (RPMs) have been widely used to evaluate the visual reasoning ability of humans. To tackle the challenges of visual perception and logic reasoning on RPMs, we propose a Hierarchical ConViT with Attention-based Relational Reasoner (HCV-ARR). Traditional solution methods often apply relatively shallow convolution networks to visually perceive shape patterns in RPM images, which may not fully model the long-range dependencies of complex pattern combinations in RPMs. The proposed ConViT consists of a convolutional block to capture the low-level attributes of visual patterns, and a transformer block to capture the high-level image semantics such as pattern formations. Furthermore, the proposed hierarchical ConViT captures visual features from multiple receptive fields, where the shallow layers focus on the image fine details while the deeper layers focus on the image semantics. To better model the underlying reasoning rules embedded in RPM images, an Attention-based Relational Reasoner (ARR) is proposed to establish the underlying relations among images. The proposed ARR well exploits the hidden relations among question images through the developed element-wise attentive reasoner. Experimental results on three RPM datasets demonstrate that the proposed HCV-ARR achieves a significant performance gain compared with the state-of-the-art models. The source code is available at: https://github.com/wentaoheunnc/HCV-ARR.

----

## [3] Deep Spiking Neural Networks with High Representation Similarity Model Visual Pathways of Macaque and Mouse

**Authors**: *Liwei Huang, Zhengyu Ma, Liutao Yu, Huihui Zhou, Yonghong Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25073](https://doi.org/10.1609/aaai.v37i1.25073)

**Abstract**:

Deep artificial neural networks (ANNs) play a major role in modeling the visual pathways of primate and rodent. However, they highly simplify the computational properties of neurons compared to their biological counterparts. Instead, Spiking Neural Networks (SNNs) are more biologically plausible models since spiking neurons encode information with time sequences of spikes, just like biological neurons do. However, there is a lack of studies on visual pathways with deep SNNs models. In this study, we model the visual cortex with deep SNNs for the first time, and also with a wide range of state-of-the-art deep CNNs and ViTs for comparison. Using three similarity metrics, we conduct neural representation similarity experiments on three neural datasets collected from two species under three types of stimuli. Based on extensive similarity analyses, we further investigate the functional hierarchy and mechanisms across species. Almost all similarity scores of SNNs are higher than their counterparts of CNNs with an average of 6.6%. Depths of the layers with the highest similarity scores exhibit little differences across mouse cortical regions, but vary significantly across macaque regions, suggesting that the visual processing structure of mice is more regionally homogeneous than that of macaques. Besides, the multi-branch structures observed in some top mouse brain-like neural networks provide computational evidence of parallel processing streams in mice, and the different performance in fitting macaque neural representations under different stimuli exhibits the functional specialization of information processing in macaques. Taken together, our study demonstrates that SNNs could serve as promising candidates to better model and explain the functional hierarchy and mechanisms of the visual system.

----

## [4] A Semi-parametric Model for Decision Making in High-Dimensional Sensory Discrimination Tasks

**Authors**: *Stephen Keeley, Benjamin Letham, Craig Sanders, Chase Tymms, Michael Shvartsman*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25074](https://doi.org/10.1609/aaai.v37i1.25074)

**Abstract**:

Psychometric functions typically characterize binary sensory decisions along a single stimulus dimension. However, real-life sensory tasks vary along a greater variety of dimensions (e.g. color, contrast and luminance for visual stimuli). Approaches to characterizing high-dimensional sensory spaces either require strong parametric assumptions about these additional contextual dimensions, or fail to leverage known properties of classical psychometric curves. We overcome both limitations by introducing a semi-parametric model of sensory discrimination that applies traditional psychophysical models along a stimulus intensity dimension, but puts Gaussian process (GP) priors on the parameters of these models with respect to the remaining dimensions. By combining the flexibility of the GP with the deep literature on parametric psychophysics, our semi-parametric models achieve good performance with much less data than baselines on both synthetic and real-world, high-dimensional psychophysics datasets. We additionally show strong performance in a Bayesian active learning setting, and present a novel active learning paradigm for the semi-parametric model.

----

## [5] A Machine with Short-Term, Episodic, and Semantic Memory Systems

**Authors**: *Taewoon Kim, Michael Cochez, Vincent François-Lavet, Mark A. Neerincx, Piek Vossen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25075](https://doi.org/10.1609/aaai.v37i1.25075)

**Abstract**:

Inspired by the cognitive science theory of the explicit human memory systems, we have modeled an agent with short-term, episodic, and semantic memory systems, each of which is modeled with a knowledge graph. To evaluate this system and analyze the behavior of this agent, we designed and released our own reinforcement learning agent environment, “the Room”, where an agent has to learn how to encode, store, and retrieve memories to maximize its return by answering questions. We show that our deep Q-learning based agent successfully learns whether a short-term memory should be forgotten, or rather be stored in the episodic or semantic memory systems. Our experiments indicate that an agent with human-like memory systems can outperform an agent without this memory structure in the environment.

----

## [6] Persuasion Strategies in Advertisements

**Authors**: *Yaman Kumar, Rajat Jha, Arunim Gupta, Milan Aggarwal, Aditya Garg, Tushar Malyan, Ayush Bhardwaj, Rajiv Ratn Shah, Balaji Krishnamurthy, Changyou Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25076](https://doi.org/10.1609/aaai.v37i1.25076)

**Abstract**:

Modeling what makes an advertisement persuasive, i.e., eliciting the desired response from consumer, is critical to the
study of propaganda, social psychology, and marketing. Despite its importance, computational modeling of persuasion
in computer vision is still in its infancy, primarily due to
the lack of benchmark datasets that can provide persuasion-strategy labels associated with ads. Motivated by persuasion literature in social psychology and marketing, we introduce an extensive vocabulary of persuasion strategies and
build the first ad image corpus annotated with persuasion
strategies. We then formulate the task of persuasion strategy prediction with multi-modal learning, where we design
a multi-task attention fusion model that can leverage other
ad-understanding tasks to predict persuasion strategies. The
dataset also provides image segmentation masks, which labels persuasion strategies in the corresponding ad images on
the test split. We publicly release our code and dataset at https://midas-research.github.io/persuasion-advertisements/.

----

## [7] Intensity-Aware Loss for Dynamic Facial Expression Recognition in the Wild

**Authors**: *Hanting Li, Hongjing Niu, Zhaoqing Zhu, Feng Zhao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25077](https://doi.org/10.1609/aaai.v37i1.25077)

**Abstract**:

Compared with the image-based static facial expression recognition (SFER) task, the dynamic facial expression recognition (DFER) task based on video sequences is closer to the natural expression recognition scene. However, DFER is often more challenging. One of the main reasons is that video sequences often contain frames with different expression intensities, especially for the facial expressions in the real-world scenarios, while the images in SFER frequently present uniform and high expression intensities. Nevertheless, if the expressions with different intensities are treated equally, the features learned by the networks will have large intra-class and small inter-class differences, which are harmful to DFER. To tackle this problem, we propose the global convolution-attention block (GCA) to rescale the channels of the feature maps. In addition, we introduce the intensity-aware loss (IAL) in the training process to help the network distinguish the samples with relatively low expression intensities. Experiments on two in-the-wild dynamic facial expression datasets (i.e., DFEW and FERV39k) indicate that our method outperforms the state-of-the-art DFER approaches. The source code will be available at https://github.com/muse1998/IAL-for-Facial-Expression-Recognition.

----

## [8] AVCAffe: A Large Scale Audio-Visual Dataset of Cognitive Load and Affect for Remote Work

**Authors**: *Pritam Sarkar, Aaron Posen, Ali Etemad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25078](https://doi.org/10.1609/aaai.v37i1.25078)

**Abstract**:

We introduce AVCAffe, the first Audio-Visual dataset consisting of Cognitive load and Affect attributes. We record AVCAffe by simulating remote work scenarios over a video-conferencing platform, where subjects collaborate to complete a number of cognitively engaging tasks. AVCAffe is the largest originally collected (not collected from the Internet) affective dataset in English language. We recruit 106 participants from 18 different countries of origin, spanning an age range of 18 to 57 years old, with a balanced male-female ratio. AVCAffe comprises a total of 108 hours of video, equivalent to more than 58,000 clips along with task-based self-reported ground truth labels for arousal, valence, and cognitive load attributes such as mental demand, temporal demand, effort, and a few others. We believe AVCAffe would be a challenging benchmark for the deep learning research community given the inherent difficulty of classifying affect and cognitive load in particular. Moreover, our dataset fills an existing timely gap by facilitating the creation of learning systems for better self-management of remote work meetings, and further study of hypotheses regarding the impact of remote work on cognitive load and affective states.

----

## [9] ESL-SNNs: An Evolutionary Structure Learning Strategy for Spiking Neural Networks

**Authors**: *Jiangrong Shen, Qi Xu, Jian K. Liu, Yueming Wang, Gang Pan, Huajin Tang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25079](https://doi.org/10.1609/aaai.v37i1.25079)

**Abstract**:

Spiking neural networks (SNNs) have manifested remarkable advantages in power consumption and event-driven property during the inference process. To take full advantage of low power consumption and improve the efficiency of these models further, the pruning methods have been explored to find sparse SNNs without redundancy connections after training. However, parameter redundancy still hinders the efficiency of SNNs during training. In the human brain, the rewiring process of neural networks is highly dynamic, while synaptic connections maintain relatively sparse during brain development. Inspired by this, here we propose an efficient evolutionary structure learning (ESL) framework for SNNs, named ESL-SNNs, to implement the sparse SNN training from scratch. The pruning and regeneration of synaptic connections in SNNs evolve dynamically during learning, yet keep the structural sparsity at a certain level. As a result, the ESL-SNNs can search for optimal sparse connectivity by exploring all possible parameters across time. Our experiments show that the proposed ESL-SNNs framework is able to learn SNNs with sparse structures effectively while reducing the limited accuracy. The ESL-SNNs achieve merely 0.28% accuracy loss with 10% connection density on the DVS-Cifar10 dataset. Our work presents a brand-new approach for sparse training of SNNs from scratch with biologically plausible evolutionary mechanisms, closing the gap in the expressibility between sparse training and dense training. Hence, it has great potential for SNN lightweight training and inference with low power consumption and small memory usage.

----

## [10] Zero-Shot Linear Combinations of Grounded Social Interactions with Linear Social MDPs

**Authors**: *Ravi Tejwani, Yen-Ling Kuo, Tianmin Shu, Bennett Stankovits, Dan Gutfreund, Joshua B. Tenenbaum, Boris Katz, Andrei Barbu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25080](https://doi.org/10.1609/aaai.v37i1.25080)

**Abstract**:

Humans and animals engage in rich social interactions. It is often theorized that a relatively small number of basic social interactions give rise to the full range of behavior observed. But no computational theory explaining how social interactions combine together has been proposed before. We do so here. We take a model, the Social MDP, which is able to express a range of social interactions, and extend it to represent linear combinations of social interactions. Practically for robotics applications, such models are now able to not just express that an agent should help another agent, but to express goal-centric social interactions. Perhaps an agent is helping someone get dressed, but preventing them from falling, and is happy to exchange stories in the meantime. How an agent responds socially, should depend on what it thinks the other agent is doing at that point in time. To encode this notion, we take linear combinations of social interactions as defined in Social MDPs, and compute the weights on those combinations on the fly depending on the estimated goals of other agents. This new model, the Linear Social MDP, enables zero-shot reasoning about complex social interactions, provides a mathematical basis for the long-standing intuition that social interactions should compose, and leads to interesting new behaviors that we validate using human observers. Complex social interactions are part of the future of intelligent agents, and having principled mathematical models built on a foundation like MDPs will make it possible to bring social interactions to every robotic application.

----

## [11] Complex Dynamic Neurons Improved Spiking Transformer Network for Efficient Automatic Speech Recognition

**Authors**: *Qingyu Wang, Tielin Zhang, Minglun Han, Yi Wang, Duzhen Zhang, Bo Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25081](https://doi.org/10.1609/aaai.v37i1.25081)

**Abstract**:

The spiking neural network (SNN) using leaky-integrated-and-fire (LIF) neurons has been commonly used in automatic speech recognition (ASR) tasks. However, the LIF neuron is still relatively simple compared to that in the biological brain. Further research on more types of neurons with different scales of neuronal dynamics is necessary. Here we introduce four types of neuronal dynamics to post-process the sequential patterns generated from the spiking transformer to get the complex dynamic neuron improved spiking transformer neural network (DyTr-SNN). We found that the DyTr-SNN could handle the non-toy automatic speech recognition task well, representing a lower phoneme error rate, lower computational cost, and higher robustness. These results indicate that the further cooperation of SNNs and neural dynamics at the neuron and network scales might have much in store for the future, especially on the ASR tasks.

----

## [12] Self-Supervised Graph Learning for Long-Tailed Cognitive Diagnosis

**Authors**: *Shanshan Wang, Zhen Zeng, Xun Yang, Xingyi Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25082](https://doi.org/10.1609/aaai.v37i1.25082)

**Abstract**:

Cognitive diagnosis is a fundamental yet critical research task in the field of intelligent education, which aims to discover the proficiency level of different students on specific knowledge concepts. Despite the effectiveness of existing efforts, previous methods always considered the mastery level on the whole students, so they still suffer from the Long Tail Effect. A large number of students who have sparse interaction records are usually wrongly diagnosed during inference. To relieve the situation, we proposed a Self-supervised Cognitive Diagnosis (SCD) framework which leverages the self-supervised manner to assist the graph-based cognitive diagnosis, then the performance on those students with sparse data can be improved. Specifically, we came up with a graph confusion method that drops edges under some special rules to generate different sparse views of the graph. By maximizing the cross-view consistency of node representations, our model could pay more attention on long-tailed students. Additionally, we proposed an importance-based view generation rule to improve the influence of long-tailed students. Extensive experiments on real-world datasets show the effectiveness of our approach, especially on the students with much sparser interaction records. Our code is available at https://github.com/zeng-zhen/SCD.

----

## [13] CMNet: Contrastive Magnification Network for Micro-Expression Recognition

**Authors**: *Mengting Wei, Xingxun Jiang, Wenming Zheng, Yuan Zong, Cheng Lu, Jiateng Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25083](https://doi.org/10.1609/aaai.v37i1.25083)

**Abstract**:

Micro-Expression Recognition (MER) is challenging because the Micro-Expressions' (ME) motion is too weak to distinguish. This hurdle can be tackled by enhancing intensity for a more accurate acquisition of movements. However, existing magnification strategies tend to use the features of facial images that include not only intensity clues as intensity features, leading to the intensity representation deficient of credibility. In addition, the intensity variation over time, which is crucial for encoding movements, is also neglected. To this end, we provide a reliable scheme to extract intensity clues while considering their variation on the time scale. First, we devise an Intensity Distillation (ID) loss to acquire the intensity clues by contrasting the difference between frames, given that the difference in the same video lies only in the intensity. Then, the intensity clues are calibrated to follow the trend of the original video. Specifically, due to the lack of truth intensity annotation of the original video, we build the intensity tendency by setting each intensity vacancy an uncertain value, which guides the extracted intensity clues to converge towards this trend rather some fixed values. A Wilcoxon rank sum test (Wrst) method is enforced to implement the calibration. Experimental results on three public ME databases i.e. CASME II, SAMM, and SMIC-HS validate the superiority against state-of-the-art methods.

----

## [14] Disentangling Reafferent Effects by Doing Nothing

**Authors**: *Benedict Wilkins, Kostas Stathis*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25084](https://doi.org/10.1609/aaai.v37i1.25084)

**Abstract**:

An agent's ability to distinguish between sensory effects that are self-caused, and those that are not, is instrumental in the achievement of its goals. This ability is thought to be central to a variety of functions in biological organisms, from perceptual stabilisation and accurate motor control, to higher level cognitive functions such as planning, mirroring and the sense of agency. Although many of these functions are well studied in AI, this important distinction is rarely made explicit and the focus tends to be on the associational relationship between action and sensory effect or success. Toward the development of more general agents, we develop a framework that enables agents to disentangle self-caused and externally-caused sensory effects. Informed by relevant models and experiments in robotics, and in the biological and cognitive sciences, we demonstrate the general applicability of this framework through an extensive experimental evaluation over three different environments.

----

## [15] Learning Temporal-Ordered Representation for Spike Streams Based on Discrete Wavelet Transforms

**Authors**: *Jiyuan Zhang, Shanshan Jia, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25085](https://doi.org/10.1609/aaai.v37i1.25085)

**Abstract**:

Spike camera, a new type of neuromorphic visual sensor that imitates the sampling mechanism of the primate fovea, can capture photons and output 40000 Hz binary spike streams. Benefiting from the asynchronous sampling mechanism, the spike camera can record fast-moving objects and clear images can be recovered from the spike stream at any specified timestamps without motion blurring. Despite these, due to the dense time sequence information of the discrete spike stream, it is not easy to directly apply the existing algorithms of traditional cameras to the spike camera. Therefore, it is necessary and interesting to explore a universally effective representation of dense spike streams to better fit various network architectures. In this paper, we propose to mine temporal-robust features of spikes in time-frequency space with wavelet transforms. We present a novel Wavelet-Guided Spike Enhancing (WGSE) paradigm consisting of three consecutive steps: multi-level wavelet transform, CNN-based learnable module, and inverse wavelet transform. With the assistance of WGSE, the new streaming representation of spikes can be learned. We demonstrate the effectiveness of WGSE on two downstream tasks, achieving state-of-the-art performance on the image reconstruction task and getting considerable performance on semantic segmentation. Furthermore, We build a new spike-based synthesized dataset for semantic segmentation. Code and Datasets are available at https://github.com/Leozhangjiyuan/WGSE-SpikeCamera.

----

## [16] ScatterFormer: Locally-Invariant Scattering Transformer for Patient-Independent Multispectral Detection of Epileptiform Discharges

**Authors**: *Ruizhe Zheng, Jun Li, Yi Wang, Tian Luo, Yuguo Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25086](https://doi.org/10.1609/aaai.v37i1.25086)

**Abstract**:

Patient-independent detection of epileptic activities based on visual spectral representation of continuous EEG (cEEG) has been widely used for diagnosing epilepsy. However, precise detection remains a considerable challenge due to subtle variabilities across subjects, channels and time points. Thus, capturing fine-grained, discriminative features of EEG patterns, which is associated with high-frequency textural information, is yet to be resolved. In this work, we propose Scattering Transformer (ScatterFormer), an invariant scattering transform-based hierarchical Transformer that specifically pays attention to subtle features. In particular, the disentangled frequency-aware attention (FAA) enables the Transformer to capture clinically informative high-frequency components, offering a novel clinical explainability based on visual encoding of multichannel EEG signals. Evaluations on two distinct tasks of epileptiform detection demonstrate
the effectiveness our method. Our proposed model achieves median AUCROC and accuracy of 98.14%, 96.39% in patients with Rolandic epilepsy. On a neonatal seizure detection benchmark, it outperforms the state-of-the-art by 9% in terms of average AUCROC.

----

## [17] Progress and Limitations of Deep Networks to Recognize Objects in Unusual Poses

**Authors**: *Amro Abbas, Stéphane Deny*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25087](https://doi.org/10.1609/aaai.v37i1.25087)

**Abstract**:

Deep networks should be robust to rare events if they are to be successfully deployed in high-stakes real-world applications. Here we study the capability of deep networks to recognize objects in unusual poses. We create a synthetic dataset of images of objects in unusual orientations, and evaluate the robustness of a collection of 38 recent and competitive deep networks for image classification. We show that classifying these images is still a challenge for all networks tested, with an average accuracy drop of 29.5% compared to when the objects are presented
upright. This brittleness is largely unaffected by various design choices, such as training losses, architectures, dataset modalities, and data-augmentation schemes. However, networks trained on very large datasets substantially outperform others, with the best network tested—Noisy Student trained on JFT-300M—showing a relatively small accuracy drop of only 14.5% on unusual poses. Nevertheless, a visual inspection of the failures of Noisy Student reveals a remaining gap in robustness with humans. Furthermore, combining multiple object transformations—3D-rotations and scaling—further degrades the performance of all networks. Our results provide another measurement of the robustness of deep networks to consider when using them in the real world. Code and datasets are available at https://github.com/amro-kamal/ObjectPose.

----

## [18] Denoising after Entropy-Based Debiasing a Robust Training Method for Dataset Bias with Noisy Labels

**Authors**: *Sumyeong Ahn, Se-Young Yun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25088](https://doi.org/10.1609/aaai.v37i1.25088)

**Abstract**:

Improperly constructed datasets can result in inaccurate inferences. For instance, models trained on biased datasets perform poorly in terms of generalization (i.e., dataset bias). Recent debiasing techniques have successfully achieved generalization performance by underestimating easy-to-learn samples (i.e., bias-aligned samples) and highlighting difficult-to-learn samples (i.e., bias-conflicting samples). However, these techniques may fail owing to noisy labels, because the trained model recognizes noisy labels as difficult-to-learn and thus highlights them. In this study, we find that earlier approaches that used the provided labels to quantify difficulty could be affected by the small proportion of noisy labels. Furthermore, we find that running denoising algorithms before debiasing is ineffective because denoising algorithms reduce the impact of difficult-to-learn samples, including valuable bias-conflicting samples. Therefore, we propose an approach called denoising after entropy-based debiasing, i.e., DENEB, which has three main stages. (1) The prejudice model is trained by emphasizing (bias-aligned, clean) samples, which are selected using a Gaussian Mixture Model. (2) Using the per-sample entropy from the output of the prejudice model, the sampling probability of each sample that is proportional to the entropy is computed. (3) The final model is trained using existing denoising algorithms with the mini-batches constructed by following the computed sampling probability. Compared to existing debiasing and denoising algorithms, our method achieves better debiasing performance on multiple benchmarks.

----

## [19] Rethinking Interpretation: Input-Agnostic Saliency Mapping of Deep Visual Classifiers

**Authors**: *Naveed Akhtar, Mohammad Amir Asim Khan Jalwana*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25089](https://doi.org/10.1609/aaai.v37i1.25089)

**Abstract**:

Saliency methods provide post-hoc model interpretation by attributing input features to the model outputs. Current methods mainly achieve this using a single input sample, thereby failing to answer input-independent inquiries about the model. We also show that input-specific saliency mapping is intrinsically susceptible to misleading feature attribution. Current attempts to use `general' input features for model interpretation assume  access to a dataset containing  those features, which biases the interpretation.  Addressing the gap, we introduce a new perspective of input-agnostic saliency mapping that computationally estimates the high-level  features attributed by the model to its outputs. These features are geometrically correlated, and are computed by accumulating model's gradient information with respect to an unrestricted data distribution. To compute these features, we nudge independent data points over the model loss surface towards the local minima associated by a human-understandable concept, e.g., class label for classifiers. With a systematic projection, scaling and refinement process, this information is transformed into an interpretable visualization without compromising its model-fidelity. The visualization serves as a stand-alone qualitative interpretation. With an extensive evaluation, we not only demonstrate successful visualizations for a variety of concepts for large-scale models, but also showcase an interesting utility of this new form of saliency mapping by identifying backdoor signatures in compromised classifiers.

----

## [20] Deep Digging into the Generalization of Self-Supervised Monocular Depth Estimation

**Authors**: *Jinwoo Bae, Sungho Moon, Sunghoon Im*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25090](https://doi.org/10.1609/aaai.v37i1.25090)

**Abstract**:

Self-supervised monocular depth estimation has been widely studied recently. Most of the work has focused on improving performance on benchmark datasets, such as KITTI, but has offered a few experiments on generalization performance. In this paper, we investigate the backbone networks (e.g., CNNs, Transformers, and CNN-Transformer hybrid models) toward the generalization of monocular depth estimation. We first evaluate state-of-the-art models on diverse public datasets, which have never been seen during the network training. Next, we investigate the effects of texture-biased and shape-biased representations using the various texture-shifted datasets that we generated. We observe that Transformers exhibit a strong shape bias and CNNs do a strong texture-bias. We also find that shape-biased models show better generalization performance for monocular depth estimation compared to texture-biased models. Based on these observations, we newly design a CNN-Transformer hybrid network with a multi-level adaptive feature fusion module, called MonoFormer. The design intuition behind MonoFormer is to increase shape bias by employing Transformers while compensating for the weak locality bias of Transformers by adaptively fusing multi-level representations. Extensive experiments show that the proposed method achieves state-of-the-art performance with various public datasets. Our method also shows the best generalization ability among the competitive methods.

----

## [21] Self-Contrastive Learning: Single-Viewed Supervised Contrastive Framework Using Sub-network

**Authors**: *Sangmin Bae, Sungnyun Kim, Jongwoo Ko, Gihun Lee, Seungjong Noh, Se-Young Yun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25091](https://doi.org/10.1609/aaai.v37i1.25091)

**Abstract**:

Contrastive loss has significantly improved performance in supervised classification tasks by using a multi-viewed framework that leverages augmentation and label information. The augmentation enables contrast with another view of a single image but enlarges training time and memory usage. To exploit the strength of multi-views while avoiding the high computation cost, we introduce a multi-exit architecture that outputs multiple features of a single image in a single-viewed framework. To this end, we propose Self-Contrastive (SelfCon) learning, which self-contrasts within multiple outputs from the different levels of a single network. The multi-exit architecture efficiently replaces multi-augmented images and leverages various information from different layers of a network. We demonstrate that SelfCon learning improves the classification performance of the encoder network, and empirically analyze its advantages in terms of the single-view and the sub-network. Furthermore, we provide theoretical evidence of the performance increase based on the mutual information bound. For ImageNet classification on ResNet-50, SelfCon improves accuracy by +0.6% with 59% memory and 48% time of Supervised Contrastive learning, and a simple ensemble of multi-exit outputs boosts performance up to +1.5%. Our code is available at https://github.com/raymin0223/self-contrastive-learning.

----

## [22] Layout Representation Learning with Spatial and Structural Hierarchies

**Authors**: *Yue Bai, Dipu Manandhar, Zhaowen Wang, John P. Collomosse, Yun Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25092](https://doi.org/10.1609/aaai.v37i1.25092)

**Abstract**:

We present a novel hierarchical modeling method for layout representation learning, the core of design documents (e.g., user interface, poster, template).
Existing works on layout representation often ignore element hierarchies, which is an important facet of layouts, and mainly rely on the spatial bounding boxes for feature extraction.
This paper proposes a Spatial-Structural Hierarchical Auto-Encoder (SSH-AE) that learns hierarchical representation by treating a hierarchically annotated layout as a tree format.
On the one side, we model SSH-AE from both spatial (semantic views) and structural (organization and relationships) perspectives, which are two complementary aspects to represent a layout.
On the other side, the semantic/geometric properties are associated at multiple resolutions/granularities, naturally handling complex layouts.
Our learned representations are used for effective layout search from both spatial and structural similarity perspectives.
We also newly involve the tree-edit distance (TED) as an evaluation metric to construct a comprehensive evaluation protocol for layout similarity assessment, which benefits a systematic and customized layout search.
We further present a new dataset of POSTER layouts which we believe will be useful for future layout research.
We show that our proposed SSH-AE outperforms the existing methods achieving state-of-the-art performance on two benchmark datasets.
Code is available at github.com/yueb17/SSH-AE.

----

## [23] Cross-Modal Label Contrastive Learning for Unsupervised Audio-Visual Event Localization

**Authors**: *Peijun Bao, Wenhan Yang, Boon Poh Ng, Meng Hwa Er, Alex C. Kot*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25093](https://doi.org/10.1609/aaai.v37i1.25093)

**Abstract**:

This paper for the first time explores audio-visual event localization in an unsupervised manner. Previous methods tackle this problem in a supervised setting and require segment-level or video-level event category ground-truth to train the model. However, building large-scale multi-modality datasets with category annotations is human-intensive and thus not scalable to real-world applications. To this end, we propose cross-modal label contrastive learning to exploit multi-modal information among unlabeled audio and visual streams as self-supervision signals. At the feature representation level, multi-modal representations are collaboratively learned from audio and visual components by using self-supervised representation learning. At the label level, we propose a novel self-supervised pretext task i.e. label contrasting to self-annotate videos with pseudo-labels for localization model training. Note that irrelevant background would hinder the acquisition of high-quality pseudo-labels and thus lead to an inferior localization model. To address this issue, we then propose an expectation-maximization algorithm that optimizes the pseudo-label acquisition and localization model in a coarse-to-fine manner. Extensive experiments demonstrate that our unsupervised approach performs reasonably well compared to the state-of-the-art supervised methods.

----

## [24] Multi-Level Compositional Reasoning for Interactive Instruction Following

**Authors**: *Suvaansh Bhambri, Byeonghwi Kim, Jonghyun Choi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25094](https://doi.org/10.1609/aaai.v37i1.25094)

**Abstract**:

Robotic agents performing domestic chores by natural language directives are required to master the complex job of navigating environment and interacting with objects in the environments. The tasks given to the agents are often composite thus are challenging as completing them require to reason about multiple subtasks, e.g., bring a cup of coffee. To address the challenge, we propose to divide and conquer it by breaking the task into multiple subgoals and attend to them individually for better navigation and interaction. We call it Multi-level Compositional Reasoning Agent (MCR-Agent). Specifically, we learn a three-level action policy. At the highest level, we infer a sequence of human-interpretable subgoals to be executed based on language instructions by a high-level policy composition controller. At the middle level, we discriminatively control the agent’s navigation by a master policy by alternating between a navigation policy and various independent interaction policies. Finally, at the lowest level, we infer manipulation actions with the corresponding object masks using the appropriate interaction policy. Our approach not only generates human interpretable subgoals but also achieves 2.03% absolute gain to comparable state of the arts in the efficiency metric (PLWSR in unseen set) without using rule-based planning or a semantic spatial memory. The
code is available at https://github.com/yonseivnl/mcr-agent.

----

## [25] Self-Supervised Image Local Forgery Detection by JPEG Compression Trace

**Authors**: *Xiuli Bi, Wuqing Yan, Bo Liu, Bin Xiao, Weisheng Li, Xinbo Gao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25095](https://doi.org/10.1609/aaai.v37i1.25095)

**Abstract**:

For image local forgery detection, the existing methods require a large amount of labeled data for training, and most of them cannot detect multiple types of forgery simultaneously. In this paper, we firstly analyzed the JPEG compression traces which are mainly caused by different JPEG compression chains, and designed a trace extractor to learn such traces. Then, we utilized the trace extractor as the backbone and trained self-supervised to strengthen the discrimination ability of learned traces. With its benefits, regions with different JPEG compression chains can easily be distinguished within a forged image. Furthermore, our method does not rely on a large amount of training data, and even does not require any forged images for training. Experiments show that the proposed method can detect image local forgery on different datasets without re-training, and keep stable performance over various types of image local forgery.

----

## [26] VASR: Visual Analogies of Situation Recognition

**Authors**: *Yonatan Bitton, Ron Yosef, Eliyahu Strugo, Dafna Shahaf, Roy Schwartz, Gabriel Stanovsky*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25096](https://doi.org/10.1609/aaai.v37i1.25096)

**Abstract**:

A core process in human cognition is analogical mapping: the ability to identify a similar relational structure between different situations.
We introduce a novel task, Visual Analogies of Situation Recognition, adapting the classical word-analogy task into the visual domain. Given a triplet of images, the task is to select an image candidate B' that completes the analogy (A to A' is like B to what?). Unlike previous work on visual analogy that focused on simple image transformations, we tackle complex analogies requiring understanding of scenes. 

We leverage situation recognition annotations and the CLIP model to generate a large set of 500k candidate analogies. Crowdsourced  annotations for a sample of the data indicate that humans agree with the dataset label ~80% of the time (chance level 25%). Furthermore, we use human annotations to create a gold-standard dataset of 3,820 validated analogies.
Our experiments demonstrate that state-of-the-art models do well when distractors are chosen randomly (~86%), but  struggle with carefully chosen distractors (~53%, compared to 90% human accuracy). We hope our dataset will encourage the development of new analogy-making models. Website: https://vasr-dataset.github.io/

----

## [27] Parametric Surface Constrained Upsampler Network for Point Cloud

**Authors**: *Pingping Cai, Zhenyao Wu, Xinyi Wu, Song Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25097](https://doi.org/10.1609/aaai.v37i1.25097)

**Abstract**:

Designing a point cloud upsampler, which aims to generate a clean and dense point cloud given a sparse point representation, is a fundamental and challenging problem in computer vision. A line of attempts achieves this goal by establishing a point-to-point mapping function via deep neural networks. However, these approaches are prone to produce outlier points due to the lack of explicit surface-level constraints. To solve this problem, we introduce a novel surface regularizer into the upsampler network by forcing the neural network to learn the underlying parametric surface represented by bicubic functions and rotation functions, where the new generated points are then constrained on the underlying surface. These designs are integrated into two different networks for two tasks that take advantages of upsampling layers -- point cloud upsampling and point cloud completion for evaluation. The state-of-the-art experimental results on both tasks demonstrate the effectiveness of the proposed method. The implementation code will be available at https://github.com/corecai163/PSCU.

----

## [28] Explicit Invariant Feature Induced Cross-Domain Crowd Counting

**Authors**: *Yiqing Cai, Lianggangxu Chen, Haoyue Guan, Shaohui Lin, Changhong Lu, Changbo Wang, Gaoqi He*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25098](https://doi.org/10.1609/aaai.v37i1.25098)

**Abstract**:

Cross-domain crowd counting has shown progressively improved performance. However, most methods fail to explicitly consider the transferability of different features between source and target domains. In this paper, we propose an innovative explicit Invariant Feature induced Cross-domain Knowledge Transformation framework to address the inconsistent domain-invariant features of different domains. The main idea is to explicitly extract domain-invariant features from both source and target domains, which builds a bridge to transfer more rich knowledge between two domains. The framework consists of three parts, global feature decoupling (GFD), relation exploration and alignment (REA), and graph-guided knowledge enhancement (GKE). In the GFD module, domain-invariant features are efficiently decoupled from domain-specific ones in two domains, which allows the model to distinguish crowds features from backgrounds in the complex scenes. In the REA module both inter-domain relation graph (Inter-RG) and intra-domain relation graph (Intra-RG) are built. Specifically, Inter-RG aggregates multi-scale domain-invariant features between two domains and further aligns local-level invariant features. Intra-RG preserves taskrelated specific information to assist the domain alignment. Furthermore, GKE strategy models the confidence of pseudolabels to further enhance the adaptability of the target domain. Various experiments show our method achieves state-of-theart performance on the standard benchmarks. Code is available at https://github.com/caiyiqing/IF-CKT.

----

## [29] Painterly Image Harmonization in Dual Domains

**Authors**: *Junyan Cao, Yan Hong, Li Niu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25099](https://doi.org/10.1609/aaai.v37i1.25099)

**Abstract**:

Image harmonization aims to produce visually harmonious composite images by adjusting the foreground appearance to be compatible with the background. When the composite image has photographic foreground and painterly background, the task is called painterly image harmonization. There are only few works on this task, which are either time-consuming or weak in generating well-harmonized results. In this work, we propose a novel painterly harmonization network consisting of a dual-domain generator and a dual-domain discriminator, which harmonizes the composite image in both spatial domain and frequency domain. The dual-domain generator performs harmonization by using AdaIN modules in the spatial domain and our proposed ResFFT modules in the frequency domain. The dual-domain discriminator attempts to distinguish the inharmonious patches based on the spatial feature and frequency feature of each patch, which can enhance the ability of generator in an adversarial manner. Extensive experiments on the benchmark dataset show the effectiveness of our method. Our code and model are available at https://github.com/bcmi/PHDNet-Painterly-Image-Harmonization.

----

## [30] MMTN: Multi-Modal Memory Transformer Network for Image-Report Consistent Medical Report Generation

**Authors**: *Yiming Cao, Lizhen Cui, Lei Zhang, Fuqiang Yu, Zhen Li, Yonghui Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25100](https://doi.org/10.1609/aaai.v37i1.25100)

**Abstract**:

Automatic medical report generation is an essential task in applying artificial intelligence to the medical domain, which can lighten the workloads of doctors and promote clinical automation. The state-of-the-art approaches employ Transformer-based encoder-decoder architectures to generate reports for medical images. However, they do not fully explore the relationships between multi-modal medical data, and generate inaccurate and inconsistent reports. To address these issues, this paper proposes a Multi-modal Memory Transformer Network (MMTN) to cope with multi-modal medical data for generating image-report consistent medical reports. On the one hand, MMTN reduces the occurrence of image-report inconsistencies by designing a unique encoder to associate and memorize the relationship between medical images and medical terminologies. On the other hand, MMTN utilizes the cross-modal complementarity of the medical vision and language for the word prediction, which further enhances the accuracy of generating medical reports. Extensive experiments on three real datasets show that MMTN achieves significant effectiveness over state-of-the-art approaches on both automatic metrics and human evaluation.

----

## [31] KT-Net: Knowledge Transfer for Unpaired 3D Shape Completion

**Authors**: *Zhen Cao, Wenxiao Zhang, Xin Wen, Zhen Dong, Yu-Shen Liu, Xiongwu Xiao, Bisheng Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25101](https://doi.org/10.1609/aaai.v37i1.25101)

**Abstract**:

Unpaired 3D object completion aims to predict a complete 3D shape from an incomplete input without knowing the correspondence between the complete and incomplete shapes. In this paper, we propose the novel KTNet to solve this task from the new perspective of knowledge transfer. KTNet elaborates a teacher-assistant-student network to establish multiple knowledge transfer processes. Specifically, the teacher network takes complete shape as input and learns the knowledge of complete shape. The student network takes the incomplete one as input and restores the corresponding complete shape. And the assistant modules not only help to transfer the knowledge of complete shape from the teacher to the student, but also judge the learning effect of the student network. As a result, KTNet makes use of a more comprehensive understanding to establish the geometric correspondence between complete and incomplete shapes in a perspective of knowledge transfer, which enables more detailed geometric inference for generating high-quality complete shapes. We conduct comprehensive experiments on several datasets, and the results show that our method outperforms previous methods of unpaired point cloud completion by a large margin. Code is available at https://github.com/a4152684/KT-Net.

----

## [32] Deconstructed Generation-Based Zero-Shot Model

**Authors**: *Dubing Chen, Yuming Shen, Haofeng Zhang, Philip H. S. Torr*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25102](https://doi.org/10.1609/aaai.v37i1.25102)

**Abstract**:

Recent research on Generalized Zero-Shot Learning (GZSL) has focused primarily on generation-based methods. However, current literature has overlooked the fundamental principles of these methods and has made limited progress in a complex manner. In this paper, we aim to deconstruct the generator-classifier framework and provide guidance for its improvement and extension. We begin by breaking down the generator-learned unseen class distribution into class-level and instance-level distributions. Through our analysis of the role of these two types of distributions in solving the GZSL problem, we generalize the focus of the generation-based approach, emphasizing the importance of (i) attribute generalization in generator learning and (ii) independent classifier learning with partially biased data. We present a simple method based on this analysis that outperforms SotAs on four public GZSL datasets, demonstrating the validity of our deconstruction. Furthermore, our proposed method remains effective even without a generative model, representing a step towards simplifying the generator-classifier structure. Our code is available at https://github.com/cdb342/DGZ.

----

## [33] Tracking and Reconstructing Hand Object Interactions from Point Cloud Sequences in the Wild

**Authors**: *Jiayi Chen, Mi Yan, Jiazhao Zhang, Yinzhen Xu, Xiaolong Li, Yijia Weng, Li Yi, Shuran Song, He Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25103](https://doi.org/10.1609/aaai.v37i1.25103)

**Abstract**:

In this work, we tackle the challenging task of jointly tracking hand object poses and reconstructing their shapes from depth point cloud sequences in the wild, given the initial poses at frame 0. We for the first time propose a point cloud-based hand joint tracking network, HandTrackNet, to estimate the inter-frame hand joint motion. Our HandTrackNet proposes a novel hand pose canonicalization module to ease the tracking task, yielding accurate and robust hand joint tracking. Our pipeline then reconstructs the full hand via converting the predicted hand joints into a MANO hand. For object tracking, we devise a simple yet effective module that estimates the object SDF from the first frame and performs optimization-based tracking. Finally, a joint optimization step is adopted to perform joint hand and object reasoning, which alleviates the occlusion-induced ambiguity and further refines the hand pose. During training, the whole pipeline only sees purely synthetic data, which are synthesized with sufficient variations and by depth simulation for the ease of generalization. The whole pipeline is pertinent to the generalization gaps and thus directly transferable to real in-the-wild data. We evaluate our method on two real hand object interaction datasets, e.g. HO3D and DexYCB, without any fine-tuning. Our experiments demonstrate that the proposed method significantly outperforms the previous state-of-the-art depth-based hand and object pose estimation and tracking methods, running at a frame rate of 9 FPS. We have released our code on https://github.com/PKU-EPIC/HOTrack.

----

## [34] Amodal Instance Segmentation via Prior-Guided Expansion

**Authors**: *Junjie Chen, Li Niu, Jianfu Zhang, Jianlou Si, Chen Qian, Liqing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25104](https://doi.org/10.1609/aaai.v37i1.25104)

**Abstract**:

Amodal instance segmentation aims to infer the amodal mask, including both the visible part and occluded part of each object instance. Predicting the occluded parts is challenging. Existing methods often produce incomplete amodal boxes and amodal masks, probably due to lacking visual evidences to expand the boxes and masks. To this end, we propose a prior-guided expansion framework, which builds on a two-stage segmentation model (i.e., Mask R-CNN) and performs box-level (resp., pixel-level) expansion for amodal box (resp., mask) prediction, by retrieving regression (resp., flow) transformations from a memory bank of expansion prior. We conduct extensive experiments on KINS, D2SA, and COCOA cls datasets, which show the effectiveness of our method.

----

## [35] SwinRDM: Integrate SwinRNN with Diffusion Model towards High-Resolution and High-Quality Weather Forecasting

**Authors**: *Lei Chen, Fei Du, Yuan Hu, Zhibin Wang, Fan Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25105](https://doi.org/10.1609/aaai.v37i1.25105)

**Abstract**:

Data-driven medium-range weather forecasting has attracted much attention in recent years. However, the forecasting accuracy at high resolution is unsatisfactory currently. Pursuing high-resolution and high-quality weather forecasting, we develop a data-driven model SwinRDM which integrates an improved version of SwinRNN with a diffusion model. SwinRDM performs predictions at 0.25-degree resolution and achieves superior forecasting accuracy to IFS (Integrated Forecast System), the state-of-the-art operational NWP model, on representative atmospheric variables including 500 hPa geopotential (Z500), 850 hPa temperature (T850), 2-m temperature (T2M), and total precipitation (TP), at lead times of up to 5 days. We propose to leverage a two-step strategy to achieve high-resolution predictions at 0.25-degree considering the trade-off between computation memory and forecasting accuracy. Recurrent predictions for future atmospheric fields are firstly performed at 1.40625-degree resolution, and then a diffusion-based super-resolution model is leveraged to recover the high spatial resolution and finer-scale atmospheric details. SwinRDM pushes forward the performance and potential of data-driven models for a large margin towards operational applications.

----

## [36] Take Your Model Further: A General Post-refinement Network for Light Field Disparity Estimation via BadPix Correction

**Authors**: *Rongshan Chen, Hao Sheng, Da Yang, Sizhe Wang, Zhenglong Cui, Ruixuan Cong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25106](https://doi.org/10.1609/aaai.v37i1.25106)

**Abstract**:

Most existing light field (LF) disparity estimation algorithms focus on handling occlusion, texture-less or other areas that harm LF structure to improve accuracy, while ignoring other potential modeling ideas. In this paper, we propose a novel idea called Bad Pixel (BadPix) correction for method modeling, then implement a general post-refinement network for LF disparity estimation: Bad-pixel Correction Network (BpCNet). Given an initial disparity map generated by a specific algorithm, we assume that all BadPixs on it are in a small range. Then BpCNet is modeled as a fine-grained search strategy, and a more accurate result can be obtained by evaluating the consistency of LF images in this limited range. Due to the assumption and the consistency between input and output, BpCNet can perform as a  general  post-refinement network, and can work on almost all existing algorithms iteratively.  We demonstrate the feasibility of our  theory through extensive experiments, and achieve remarkable performance on the HCI 4D Light Field Benchmark.

----

## [37] Improving Dynamic HDR Imaging with Fusion Transformer

**Authors**: *Rufeng Chen, Bolun Zheng, Hua Zhang, Quan Chen, Chenggang Yan, Gregory G. Slabaugh, Shanxin Yuan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25107](https://doi.org/10.1609/aaai.v37i1.25107)

**Abstract**:

Reconstructing a High Dynamic Range (HDR) image from several Low Dynamic Range (LDR) images with different exposures is a challenging task, especially in the presence of camera and object motion. Though existing models using convolutional neural networks (CNNs) have made great progress, challenges still exist, e.g., ghosting artifacts. Transformers, originating from the field of natural language processing, have shown success in computer vision tasks, due to their ability to address a large receptive field even within a single layer. In this paper, we propose a transformer model for HDR imaging. Our pipeline includes three steps: alignment, fusion, and reconstruction. The key component is the HDR transformer module. Through experiments and ablation studies, we demonstrate that our model outperforms the state-of-the-art by large margins on several popular public datasets.

----

## [38] Self-Supervised Joint Dynamic Scene Reconstruction and Optical Flow Estimation for Spiking Camera

**Authors**: *Shiyan Chen, Zhaofei Yu, Tiejun Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25108](https://doi.org/10.1609/aaai.v37i1.25108)

**Abstract**:

Spiking camera, a novel retina-inspired vision sensor, has shown its great potential for capturing high-speed dynamic scenes with a sampling rate of 40,000 Hz. The spiking camera abandons the concept of exposure window, with each of its photosensitive units continuously capturing photons and firing spikes asynchronously. However, the special sampling mechanism prevents the frame-based algorithm from being used to spiking camera. It remains to be a challenge to reconstruct dynamic scenes and perform common computer vision tasks for spiking camera. In this paper, we propose a self-supervised joint learning framework for optical flow estimation and reconstruction of spiking camera. The framework reconstructs clean frame-based spiking representations in a self-supervised manner, and then uses them to train the optical flow networks. We also propose an optical flow based inverse rendering process to achieve self-supervision by minimizing the difference with respect to the original spiking temporal aggregation image. The experimental results demonstrate that our method bridges the gap between synthetic and real-world scenes and achieves desired results in real-world scenarios. To the best of our knowledge, this is the first attempt to jointly reconstruct dynamic scenes and estimate optical flow for spiking camera from a self-supervised learning perspective.

----

## [39] Bidirectional Optical Flow NeRF: High Accuracy and High Quality under Fewer Views

**Authors**: *Shuo Chen, Binbin Yan, Xinzhu Sang, Duo Chen, Peng Wang, Xiao Guo, Chongli Zhong, Huaming Wan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25109](https://doi.org/10.1609/aaai.v37i1.25109)

**Abstract**:

Neural Radiance Fields (NeRF) can implicitly represent 3D-consistent RGB images and geometric by optimizing an underlying continuous volumetric scene function using a sparse set of input views, which has greatly benefited view synthesis tasks. However, NeRF fails to estimate correct geometry when given fewer views, resulting in failure to synthesize novel views. Existing works rely on introducing depth images or adding depth estimation networks to resolve the problem of poor synthetic view in NeRF with fewer views. However, due to the lack of spatial consistency of the single-depth image and the poor performance of depth estimation with fewer views, the existing methods still have challenges in addressing this problem. So this paper proposes Bidirectional Optical Flow NeRF(BOF-NeRF), which addresses this problem by mining optical flow information between 2D images. Our key insight is that utilizing 2D optical flow images to design a loss can effectively guide NeRF to learn the correct geometry and synthesize the right novel view. We also propose a view-enhanced fusion method based on geometry and color consistency to solve the problem of novel view details loss in NeRF. We conduct extensive experiments on the NeRF-LLFF and DTU MVS benchmarks for novel view synthesis tasks with fewer images in different complex real scenes. We further demonstrate the robustness of BOF-NeRF under different baseline distances on the Middlebury dataset. In all cases, BOF-NeRF outperforms current state-of-the-art baselines for novel view synthesis and scene geometry estimation.

----

## [40] Scalable Spatial Memory for Scene Rendering and Navigation

**Authors**: *Wen-Cheng Chen, Chu-Song Chen, Wei-Chen Chiu, Min-Chun Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25110](https://doi.org/10.1609/aaai.v37i1.25110)

**Abstract**:

Neural scene representation and rendering methods have shown promise in learning the implicit form of scene structure without supervision. However, the implicit representation learned in most existing methods is non-expandable and cannot be inferred online for novel scenes, which makes the learned representation difficult to be applied across different reinforcement learning (RL) tasks. In this work, we introduce Scene Memory Network (SMN) to achieve online spatial memory construction and expansion for view rendering in novel scenes. SMN models the camera projection and back-projection as spatially aware memory control processes, where the memory values store the information of the partial 3D area, and the memory keys indicate the position of that area. The memory controller can learn the geometry property from observations without the camera's intrinsic parameters and depth supervision. We further apply the memory constructed by SMN to exploration and navigation tasks. The experimental results reveal the generalization ability of our proposed SMN in large-scale scene synthesis and its potential to improve the performance of spatial RL tasks.

----

## [41] Hybrid CNN-Transformer Feature Fusion for Single Image Deraining

**Authors**: *Xiang Chen, Jinshan Pan, Jiyang Lu, Zhentao Fan, Hao Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25111](https://doi.org/10.1609/aaai.v37i1.25111)

**Abstract**:

Since rain streaks exhibit diverse geometric appearances and irregular overlapped phenomena, these complex characteristics challenge the design of an effective single image deraining model. To this end, rich local-global information representations are increasingly indispensable for better satisfying rain removal. In this paper, we propose a lightweight Hybrid CNN-Transformer Feature Fusion Network (dubbed as HCT-FFN) in a stage-by-stage progressive manner, which can harmonize these two architectures to help image restoration by leveraging their individual learning strengths. Specifically, we stack a sequence of the degradation-aware mixture of experts (DaMoE) modules in the CNN-based stage, where appropriate local experts adaptively enable the model to emphasize spatially-varying rain distribution features. As for the Transformer-based stage, a background-aware vision Transformer (BaViT) module is employed to complement spatially-long feature dependencies of images, so as to achieve global texture recovery while preserving the required structure. Considering the indeterminate knowledge discrepancy among CNN features and Transformer features, we introduce an interactive fusion branch at adjacent stages to further facilitate the reconstruction of high-quality deraining results. Extensive evaluations show the effectiveness and extensibility of our developed HCT-FFN. The source code is available at https://github.com/cschenxiang/HCT-FFN.

----

## [42] MGFN: Magnitude-Contrastive Glance-and-Focus Network for Weakly-Supervised Video Anomaly Detection

**Authors**: *Yingxian Chen, Zhengzhe Liu, Baoheng Zhang, Wilton W. T. Fok, Xiaojuan Qi, Yik-Chung Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25112](https://doi.org/10.1609/aaai.v37i1.25112)

**Abstract**:

Weakly supervised detection of anomalies in surveillance videos is a challenging task. Going beyond existing works that have deficient capabilities to localize anomalies in long videos, we propose a novel glance and focus network to effectively integrate spatial-temporal information for accurate anomaly detection. In addition, we empirically found that existing approaches that use feature magnitudes to represent the degree of anomalies typically ignore the effects of scene variations, and hence result in sub-optimal performance due to the inconsistency of feature magnitudes across scenes. To address this issue, we propose the Feature Amplification Mechanism and a Magnitude Contrastive Loss to enhance the discriminativeness of feature magnitudes for detecting anomalies. Experimental results on two large-scale benchmarks UCF-Crime and XD-Violence manifest that our method outperforms state-of-the-art approaches.

----

## [43] Tagging before Alignment: Integrating Multi-Modal Tags for Video-Text Retrieval

**Authors**: *Yizhen Chen, Jie Wang, Lijian Lin, Zhongang Qi, Jin Ma, Ying Shan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25113](https://doi.org/10.1609/aaai.v37i1.25113)

**Abstract**:

Vision-language alignment learning for video-text retrieval arouses a lot of attention in recent years. Most of the existing methods either transfer the knowledge of image-text pretraining model to video-text retrieval task without fully exploring the multi-modal information of videos, or simply fuse multi-modal features in a brute force manner without explicit guidance. In this paper, we integrate multi-modal information in an explicit manner by tagging, and use the tags as the anchors for better video-text alignment. Various pretrained experts are utilized for extracting the information of multiple modalities, including object, person, motion, audio, etc. To take full advantage of these information, we propose the TABLE (TAgging Before aLignmEnt) network, which consists of a visual encoder, a tag encoder, a text encoder, and a tag-guiding cross-modal encoder for jointly encoding multi-frame visual features and multi-modal tags information. Furthermore, to strengthen the interaction between video and text, we build a joint cross-modal encoder with the triplet input of [vision, tag, text] and perform two additional supervised tasks, Video Text Matching (VTM) and Masked Language Modeling (MLM). Extensive experimental results demonstrate that the TABLE model is capable of achieving State-Of-The-Art (SOTA) performance on various video-text retrieval benchmarks, including MSR-VTT, MSVD, LSMDC and DiDeMo.

----

## [44] DUET: Cross-Modal Semantic Grounding for Contrastive Zero-Shot Learning

**Authors**: *Zhuo Chen, Yufeng Huang, Jiaoyan Chen, Yuxia Geng, Wen Zhang, Yin Fang, Jeff Z. Pan, Huajun Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25114](https://doi.org/10.1609/aaai.v37i1.25114)

**Abstract**:

Zero-shot learning (ZSL) aims to predict unseen classes whose samples have never appeared during training. One of the most effective and widely used semantic information for zero-shot image classification are attributes which are annotations for class-level visual characteristics. However, the current methods often fail to discriminate those subtle visual distinctions between images due to not only the shortage of fine-grained annotations, but also the attribute imbalance and co-occurrence. In this paper, we present a transformer-based end-to-end ZSL method named DUET, which integrates latent semantic knowledge from the pre-trained language models (PLMs) via a self-supervised multi-modal learning paradigm. Specifically, we (1) developed a cross-modal semantic grounding network to investigate the model's capability of disentangling semantic attributes from the images; (2) applied an attribute-level contrastive learning strategy to further enhance the model's discrimination on fine-grained visual characteristics against the attribute co-occurrence and imbalance; (3) proposed a multi-task learning policy for considering multi-model objectives. We find that our DUET can achieve state-of-the-art performance on three standard ZSL benchmarks and a knowledge graph equipped ZSL benchmark. Its components are effective and its predictions are interpretable.

----

## [45] Imperceptible Adversarial Attack via Invertible Neural Networks

**Authors**: *Zihan Chen, Ziyue Wang, Jun-Jie Huang, Wentao Zhao, Xiao Liu, Dejian Guan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25115](https://doi.org/10.1609/aaai.v37i1.25115)

**Abstract**:

Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks. Code is available at https://github.com/jjhuangcs/AdvINN.

----

## [46] Cross-Modality Person Re-identification with Memory-Based Contrastive Embedding

**Authors**: *De Cheng, Xiaolong Wang, Nannan Wang, Zhen Wang, Xiaoyu Wang, Xinbo Gao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25116](https://doi.org/10.1609/aaai.v37i1.25116)

**Abstract**:

Visible-infrared person re-identification (VI-ReID) aims to retrieve the person images of the same identity from the RGB to infrared image space, which is very important for real-world surveillance system. In practice, VI-ReID is more challenging due to the heterogeneous modality discrepancy, which further aggravates the challenges of traditional single-modality person ReID problem, i.e., inter-class confusion and intra-class variations. In this paper, we propose an aggregated memory-based cross-modality deep metric learning framework, which benefits from the increasing number of learned modality-aware and modality-agnostic centroid proxies for cluster contrast and mutual information learning. Furthermore, to suppress the modality discrepancy, the proposed cross-modality alignment objective simultaneously utilizes both historical and up-to-date learned cluster proxies for enhanced cross-modality association. Such training mechanism helps to obtain hard positive references through increased diversity of learned cluster proxies, and finally achieves stronger ``pulling close'' effect between cross-modality image features. Extensive experiment results demonstrate the effectiveness of the proposed method, surpassing state-of-the-art works significantly by a large margin on the commonly used VI-ReID datasets.

----

## [47] User-Controllable Arbitrary Style Transfer via Entropy Regularization

**Authors**: *Jiaxin Cheng, Yue Wu, Ayush Jaiswal, Xu Zhang, Pradeep Natarajan, Prem Natarajan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25117](https://doi.org/10.1609/aaai.v37i1.25117)

**Abstract**:

Ensuring the overall end-user experience is a challenging task in arbitrary style transfer (AST) due to the subjective nature of style transfer quality. A good practice is to provide users many instead of one AST result. However, existing approaches require to run multiple AST models or inference a diversified AST (DAST) solution multiple times, and thus they are either slow in speed or limited in diversity. In this paper, we propose a novel solution ensuring both efficiency and diversity for generating multiple user-controllable AST results by systematically modulating AST behavior at run-time. We begin with reformulating three prominent AST methods into a unified assign-and-mix problem and discover that the entropies of their assignment matrices exhibit a large variance. We then solve the unified problem in an optimal transport framework using the Sinkhorn-Knopp algorithm with a user input ε to control the said entropy and thus modulate stylization. Empirical results demonstrate the superiority of the proposed solution, with speed and stylization quality comparable to or better than existing AST and significantly more diverse than previous DAST works. Code is available at https://github.com/cplusx/eps-Assign-and-Mix.

----

## [48] Neural Architecture Search for Wide Spectrum Adversarial Robustness

**Authors**: *Zhi Cheng, Yanxi Li, Minjing Dong, Xiu Su, Shan You, Chang Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25118](https://doi.org/10.1609/aaai.v37i1.25118)

**Abstract**:

One major limitation of CNNs is that they are vulnerable to adversarial attacks. Currently, adversarial robustness in neural networks is commonly optimized with respect to a small pre-selected adversarial noise strength, causing them to have potentially limited performance when under attack by larger adversarial noises in real-world scenarios. In this research, we aim to find Neural Architectures that have improved robustness on a wide range of adversarial noise strengths through Neural Architecture Search. In detail, we propose a lightweight Adversarial Noise Estimator to reduce the high cost of generating adversarial noise with respect to different strengths. Besides, we construct an Efficient Wide Spectrum Searcher to reduce the cost of adjusting network architecture with the large adversarial validation set during the search. With the two components proposed, the number of adversarial noise strengths searched can be increased significantly while having a limited increase in search time. Extensive experiments on benchmark datasets such as CIFAR and ImageNet demonstrate that with a significantly richer search signal in robustness, our method can find architectures with improved overall robustness while having a limited impact on natural accuracy and around 40% reduction in search time compared with the naive approach of searching. Codes available at: https://github.com/zhicheng2T0/Wsr-NAS.git

----

## [49] Adversarial Alignment for Source Free Object Detection

**Authors**: *Qiaosong Chu, Shuyan Li, Guangyi Chen, Kai Li, Xiu Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25119](https://doi.org/10.1609/aaai.v37i1.25119)

**Abstract**:

Source-free object detection (SFOD) aims to transfer a detector pre-trained on a label-rich source domain to an unlabeled target domain without seeing source data. While most existing SFOD methods generate pseudo labels via a source-pretrained model to guide training, these pseudo labels usually contain high noises due to heavy domain discrepancy. In order to obtain better pseudo supervisions, we divide the target domain into source-similar and source-dissimilar parts and align them in the feature space by adversarial learning.Specifically, we design a detection variance-based criterion to divide the target domain. This criterion is motivated by a finding that larger detection variances denote higher recall and larger similarity to the source domain. Then we incorporate an adversarial module into a mean teacher framework to drive the feature spaces of these two subsets indistinguishable. Extensive experiments on multiple cross-domain object detection datasets demonstrate that our proposed method consistently outperforms the compared SFOD methods. Our implementation is available at https://github.com/ChuQiaosong.

----

## [50] Weakly Supervised 3D Multi-Person Pose Estimation for Large-Scale Scenes Based on Monocular Camera and Single LiDAR

**Authors**: *Peishan Cong, Yiteng Xu, Yiming Ren, Juze Zhang, Lan Xu, Jingya Wang, Jingyi Yu, Yuexin Ma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25120](https://doi.org/10.1609/aaai.v37i1.25120)

**Abstract**:

Depth estimation is usually ill-posed and ambiguous for monocular camera-based 3D multi-person pose estimation. Since LiDAR can capture accurate depth information in long-range scenes, it can benefit both the global localization of individuals and the 3D pose estimation by providing rich geometry features. Motivated by this, we propose a monocular camera and single LiDAR-based method for 3D multi-person pose estimation in large-scale scenes, which is easy to deploy and insensitive to light. Specifically, we design an effective fusion strategy to take advantage of multi-modal input data, including images and point cloud, and make full use of temporal information to guide the network to learn natural and coherent human motions. Without relying on any 3D pose annotations, our method exploits the inherent geometry constraints of point cloud for self-supervision and utilizes 2D keypoints on images for weak supervision. Extensive experiments on public datasets and our newly collected dataset demonstrate the superiority and generalization capability of our proposed method. Project homepage is at \url{https://github.com/4DVLab/FusionPose.git}.

----

## [51] OctFormer: Efficient Octree-Based Transformer for Point Cloud Compression with Local Enhancement

**Authors**: *Mingyue Cui, Junhua Long, Mingjian Feng, Boyang Li, Huang Kai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25121](https://doi.org/10.1609/aaai.v37i1.25121)

**Abstract**:

Point cloud compression with a higher compression ratio and tiny loss is essential for efficient data transportation. However, previous methods that depend on 3D convolution or frequent multi-head self-attention operations bring huge computations. To address this problem, we propose an octree-based Transformer compression method called OctFormer, which does not rely on the occupancy information of sibling nodes. Our method uses non-overlapped context windows to construct octree node sequences and share the result of a multi-head self-attention operation among a sequence of nodes. Besides, we introduce a locally-enhance module for exploiting the sibling features and a positional encoding generator for enhancing the translation invariance of the octree node sequence. Compared to the previous state-of-the-art works, our method obtains up to 17% Bpp savings compared to the voxel-context-based baseline and saves an overall 99% coding time compared to the attention-based baseline.

----

## [52] Dual-Domain Attention for Image Deblurring

**Authors**: *Yuning Cui, Yi Tao, Wenqi Ren, Alois Knoll*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25122](https://doi.org/10.1609/aaai.v37i1.25122)

**Abstract**:

As a long-standing and challenging task, image deblurring aims to reconstruct the latent sharp image from its degraded counterpart. In this study, to bridge the gaps between degraded/sharp image pairs in the spatial and frequency domains simultaneously,  we develop the dual-domain attention mechanism for image deblurring. Self-attention is widely used in vision tasks, however, due to the quadratic complexity, it is not applicable to image deblurring with high-resolution images. To alleviate this issue, we propose a novel spatial attention module by implementing self-attention in the style of dynamic group convolution for integrating information from the local region, enhancing the representation learning capability and reducing computational burden. Regarding frequency domain learning, many frequency-based deblurring approaches either treat the spectrum as a whole or decompose frequency components in a complicated manner. In this work, we devise a frequency attention module to compactly decouple the spectrum into distinct frequency parts and accentuate the informative part with extremely lightweight learnable parameters. Finally, we incorporate attention modules into a U-shaped network. Extensive comparisons with prior arts on the common benchmarks show that our model, named Dual-domain Attention Network (DDANet), obtains comparable results with a significantly improved inference speed.

----

## [53] Multi-Resolution Monocular Depth Map Fusion by Self-Supervised Gradient-Based Composition

**Authors**: *Yaqiao Dai, Renjiao Yi, Chenyang Zhu, Hongjun He, Kai Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25123](https://doi.org/10.1609/aaai.v37i1.25123)

**Abstract**:

Monocular depth estimation is a challenging problem on which deep neural networks have demonstrated great potential. However, depth maps predicted by existing deep models usually lack fine-grained details due to convolution operations and down-samplings in networks. We find that increasing input resolution is helpful to preserve more local details while the estimation at low resolution is more accurate globally. Therefore, we propose a novel depth map fusion module to combine the advantages of estimations with multi-resolution inputs. Instead of merging the low- and high-resolution estimations equally, we adopt the core idea of Poisson fusion, trying to implant the gradient domain of high-resolution depth into the low-resolution depth. While classic Poisson fusion requires a fusion mask as supervision, we propose a self-supervised framework based on guided image filtering. We demonstrate that this gradient-based composition performs much better at noisy immunity, compared with the state-of-the-art depth map fusion method. Our lightweight depth fusion is one-shot and runs in real-time, making it 80X faster than a state-of-the-art depth fusion method. Quantitative evaluations demonstrate that the proposed method can be integrated into many fully convolutional monocular depth estimation backbones with a significant performance boost, leading to state-of-the-art results of detail enhancement on depth maps. Codes are released at https://github.com/yuinsky/gradient-based-depth-map-fusion.

----

## [54] Improving Crowded Object Detection via Copy-Paste

**Authors**: *Jiangfan Deng, Dewen Fan, Xiaosong Qiu, Feng Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25124](https://doi.org/10.1609/aaai.v37i1.25124)

**Abstract**:

Crowdedness caused by overlapping among similar objects is a ubiquitous challenge in the field of 2D visual object detection. In this paper, we first underline two main effects of the crowdedness issue: 1) IoU-confidence correlation disturbances (ICD) and 2) confused de-duplication (CDD). Then we explore a pathway of cracking these nuts from the perspective of data augmentation. Primarily, a particular copy- paste scheme is proposed towards making crowded scenes. Based on this operation, we first design a "consensus learning" method to further resist the ICD problem and then find out the pasting process naturally reveals a pseudo "depth" of object in the scene, which can be potentially used for alleviating CDD dilemma. Both methods are derived from magical using of the copy-pasting without extra cost for hand-labeling. Experiments show that our approach can easily improve the state-of-the-art detector in typical crowded detection task by more than 2% without any bells and whistles. Moreover, this work can outperform existing data augmentation strategies in crowded scenario.

----

## [55] Defending Backdoor Attacks on Vision Transformer via Patch Processing

**Authors**: *Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25125](https://doi.org/10.1609/aaai.v37i1.25125)

**Abstract**:

Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing  attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing.
The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposedds defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.

----

## [56] Head-Free Lightweight Semantic Segmentation with Linear Transformer

**Authors**: *Bo Dong, Pichao Wang, Fan Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25126](https://doi.org/10.1609/aaai.v37i1.25126)

**Abstract**:

Existing semantic segmentation works have been mainly focused on designing effective decoders; however, the computational load introduced by the overall structure has long been ignored, which hinders their applications on resource-constrained hardwares. In this paper, we propose a head-free lightweight architecture specifically for semantic segmentation, named Adaptive Frequency Transformer (AFFormer). AFFormer adopts a parallel architecture to leverage prototype representations as specific learnable local descriptions which replaces the decoder and preserves the rich image semantics on high-resolution features. Although removing the decoder compresses most of the computation, the accuracy of the parallel structure is still limited by low computational resources. Therefore, we employ heterogeneous operators (CNN and vision Transformer) for pixel embedding and prototype representations to further save computational costs. Moreover, it is very difficult to linearize the complexity of the vision Transformer from the perspective of spatial domain. Due to the fact that semantic segmentation is very sensitive to frequency information, we construct a lightweight prototype learning block with adaptive frequency filter of complexity O(n) to replace standard self attention with O(n^2). Extensive experiments on widely adopted datasets demonstrate that AFFormer achieves superior accuracy while retaining only 3M parameters. On the ADE20K dataset, AFFormer achieves 41.8 mIoU and 4.6 GFLOPs, which is 4.4 mIoU higher than Segformer, with  45% less GFLOPs. On the Cityscapes dataset, AFFormer achieves 78.7 mIoU and 34.4 GFLOPs, which is 2.5 mIoU higher than Segformer with 72.5% less GFLOPs. Code is available at https://github.com/dongbo811/AFFormer.

----

## [57] Hierarchical Contrast for Unsupervised Skeleton-Based Action Representation Learning

**Authors**: *Jianfeng Dong, Shengkai Sun, Zhonglin Liu, Shujie Chen, Baolong Liu, Xun Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25127](https://doi.org/10.1609/aaai.v37i1.25127)

**Abstract**:

This paper targets unsupervised skeleton-based action representation learning and proposes a new Hierarchical Contrast (HiCo) framework. Different from the existing contrastive-based solutions that typically represent an input skeleton sequence into instance-level features and perform contrast holistically, our proposed HiCo represents the input into multiple-level features and performs contrast in a hierarchical manner. Specifically, given a human skeleton sequence, we represent it into multiple feature vectors of different granularities from both temporal and spatial domains via sequence-to-sequence (S2S) encoders and unified downsampling modules. Besides, the hierarchical contrast is conducted in terms of four levels: instance level, domain level, clip level, and part level. Moreover, HiCo is orthogonal to the S2S encoder, which allows us to flexibly embrace state-of-the-art S2S encoders. Extensive experiments on four datasets, i.e., NTU-60, NTU-120, PKU-I and PKU-II, show that HiCo achieves a new state-of-the-art for unsupervised skeleton-based action representation learning in two downstream tasks including action recognition and retrieval, and its learned action representation is of good transferability. Besides, we also show that our framework is effective for semi-supervised skeleton-based action recognition. Our code is available at https://github.com/HuiGuanLab/HiCo.

----

## [58] Exploring Tuning Characteristics of Ventral Stream's Neurons for Few-Shot Image Classification

**Authors**: *Lintao Dong, Wei Zhai, Zheng-Jun Zha*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25128](https://doi.org/10.1609/aaai.v37i1.25128)

**Abstract**:

Human has the remarkable ability of learning novel objects by browsing extremely few examples, which may be attributed to the generic and robust feature extracted in the ventral stream of our brain for representing visual objects. In this sense, the tuning characteristics of ventral stream's neurons can be useful prior knowledge to improve few-shot classification. Specifically, we computationally model two groups of neurons found in ventral stream which are respectively sensitive to shape cues and color cues. Then we propose the hierarchical feature regularization method with these neuron models to regularize the backbone of a few-shot model, thus making it produce more generic and robust features for few-shot classification. In addition, to simulate the tuning characteristic that neuron firing at a higher rate in response to foreground stimulus elements compared to background elements, which we call belongingness, we design a foreground segmentation algorithm based on the observation that the foreground object usually does not appear at the edge of the picture, then multiply the foreground mask with the backbone of few-shot model. Our method is model-agnostic and can be applied to few-shot models with different backbones, training paradigms and classifiers.

----

## [59] Incremental-DETR: Incremental Few-Shot Object Detection via Self-Supervised Learning

**Authors**: *Na Dong, Yongqiang Zhang, Mingli Ding, Gim Hee Lee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25129](https://doi.org/10.1609/aaai.v37i1.25129)

**Abstract**:

Incremental few-shot object detection aims at detecting novel classes without forgetting knowledge of the base classes with only a few labeled training data from the novel classes. Most related prior works are on incremental object detection that rely on the availability of abundant training samples per novel class that substantially limits the scalability to real-world setting where novel data can be scarce. In this paper, we propose the Incremental-DETR that does incremental few-shot object detection via fine-tuning and self-supervised learning on the DETR object detector. To alleviate severe over-fitting with few novel class data, we first fine-tune the class-specific components of DETR with self-supervision from additional object proposals generated using Selective Search as pseudo labels. We further introduce an incremental few-shot fine-tuning strategy with knowledge distillation on the class-specific components of DETR to encourage the network in detecting novel classes without forgetting the base classes. Extensive experiments conducted on standard incremental object detection and incremental few-shot object detection settings show that our approach significantly outperforms state-of-the-art methods by a large margin.  Our source code is available at https://github.com/dongnana777/Incremental-DETR.

----

## [60] PeCo: Perceptual Codebook for BERT Pre-training of Vision Transformers

**Authors**: *Xiaoyi Dong, Jianmin Bao, Ting Zhang, Dongdong Chen, Weiming Zhang, Lu Yuan, Dong Chen, Fang Wen, Nenghai Yu, Baining Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25130](https://doi.org/10.1609/aaai.v37i1.25130)

**Abstract**:

This paper explores a better prediction target for  BERT pre-training of vision transformers. We observe that current prediction targets disagree with human perception judgment. This contradiction motivates us to learn a perceptual prediction target. We argue that perceptually similar images should stay close to each other in the prediction target space. We surprisingly find one simple yet effective idea: enforcing perceptual similarity during the dVAE training. Moreover, we adopt a self-supervised transformer model for deep feature extraction and show that it works well for calculating perceptual similarity. We demonstrate that such learned visual tokens indeed exhibit better semantic meanings, and help pre-training achieve superior transfer performance in various downstream tasks. For example, we achieve 84.5% Top-1 accuracy on ImageNet-1K with ViT-B backbone, outperforming the competitive method BEiT by +1.3% under the same pre-training epochs. Our approach also gets significant improvement on object detection and segmentation on COCO and semantic segmentation on ADE20K. Equipped with a larger backbone ViT-H, we achieve the state-of-the-art ImageNet accuracy (88.3%) among methods using only ImageNet-1K data.

----

## [61] Domain-General Crowd Counting in Unseen Scenarios

**Authors**: *Zhipeng Du, Jiankang Deng, Miaojing Shi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25131](https://doi.org/10.1609/aaai.v37i1.25131)

**Abstract**:

Domain shift across crowd data severely hinders crowd counting models to generalize to unseen scenarios. Although domain adaptive crowd counting approaches close this gap to a certain extent, they are still dependent on the target domain data to adapt (e.g. finetune) their models to the specific domain. In this paper, we instead target to train a model based on a single source domain which can generalize well on any unseen domain. This falls into the realm of domain generalization that remains unexplored in crowd counting. We first introduce a dynamic sub-domain division scheme which divides the source domain into multiple sub-domains such that we can initiate a meta-learning framework for domain generalization. The sub-domain division is dynamically refined during the meta-learning. Next, in order to disentangle domain-invariant information from domain-specific information in image features, we design the domain-invariant and -specific crowd memory modules to re-encode image features. Two types of losses, i.e. feature reconstruction and orthogonal losses, are devised to enable this disentanglement. Extensive experiments on several standard crowd counting benchmarks i.e. SHA, SHB, QNRF, and NWPU, show the strong generalizability of our method. Our code is available at: https://github.com/ZPDu/Domain-general-Crowd-Counting-in-Unseen-Scenarios

----

## [62] Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation

**Authors**: *Yuxuan Duan, Yan Hong, Li Niu, Liqing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25132](https://doi.org/10.1609/aaai.v37i1.25132)

**Abstract**:

The performances of defect inspection have been severely hindered by insufficient defect images in industries, which can be alleviated by generating more samples as data augmentation. We propose the first defect image generation method in the challenging few-shot cases. Given just a handful of defect images and relatively more defect-free ones, our goal is to augment the dataset with new defect images. Our method consists of two training stages. First, we train a data-efficient StyleGAN2 on defect-free images as the backbone. Second, we attach defect-aware residual blocks to the backbone, which learn to produce reasonable defect masks and accordingly manipulate the features within the masked regions by training the added modules on limited defect images. Extensive experiments on MVTec AD dataset not only validate the effectiveness of our method in generating realistic and diverse defect images, but also manifest the benefits it brings to downstream defect inspection tasks. Codes are available at https://github.com/Ldhlwh/DFMGAN.

----

## [63] Frido: Feature Pyramid Diffusion for Complex Scene Image Synthesis

**Authors**: *Wan-Cyuan Fan, Yen-Chun Chen, Dongdong Chen, Yu Cheng, Lu Yuan, Yu-Chiang Frank Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25133](https://doi.org/10.1609/aaai.v37i1.25133)

**Abstract**:

Diffusion models (DMs) have shown great potential for high-quality image synthesis. However, when it comes to producing images with complex scenes, how to properly describe both image global structures and object details remains a challenging task. In this paper, we present Frido, a Feature Pyramid Diffusion model performing a multi-scale coarse-to-fine denoising process for image synthesis. Our model decomposes an input image into scale-dependent vector quantized features, followed by a coarse-to-fine gating for producing image output. During the above multi-scale representation learning stage, additional input conditions like text, scene graph, or image layout can be further exploited. Thus, Frido can be also applied for conditional or cross-modality image synthesis. We conduct extensive experiments over various unconditioned and conditional image generation tasks, ranging from text-to-image synthesis, layout-to-image, scene-graph-to-image, to label-to-image. More specifically, we achieved state-of-the-art FID scores on five benchmarks, namely layout-to-image on COCO and OpenImages, scene-graph-to-image on COCO and Visual Genome, and label-to-image on COCO.

----

## [64] Target-Free Text-Guided Image Manipulation

**Authors**: *Wan-Cyuan Fan, Cheng-Fu Yang, Chiao-An Yang, Yu-Chiang Frank Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25134](https://doi.org/10.1609/aaai.v37i1.25134)

**Abstract**:

We tackle the problem of target-free text-guided image manipulation, which requires one to modify the input reference image based on the given text instruction, while no ground truth target image is observed during training. To address this challenging task, we propose a Cyclic-Manipulation GAN (cManiGAN) in this paper, which is able to realize where and how to edit the image regions of interest. Specifically, the image editor in cManiGAN learns to identify and complete the input image, while cross-modal interpreter and reasoner are deployed to verify the semantic correctness of the output image based on the input instruction. While the former utilizes factual/counterfactual description learning for authenticating the image semantics, the latter predicts the "undo" instruction and provides pixel-level supervision for the training of cManiGAN. With the above operational cycle-consistency, our cManiGAN can be trained in the above weakly supervised setting. We conduct extensive experiments on the datasets of CLEVR and COCO datasets, and the effectiveness and generalizability of our proposed method can be successfully verified. Project page: sites.google.com/view/wancyuanfan/projects/cmanigan.

----

## [65] One Is All: Bridging the Gap between Neural Radiance Fields Architectures with Progressive Volume Distillation

**Authors**: *Shuangkang Fang, Weixin Xu, Heng Wang, Yi Yang, Yufeng Wang, Shuchang Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25135](https://doi.org/10.1609/aaai.v37i1.25135)

**Abstract**:

Neural Radiance Fields (NeRF) methods have proved effective as compact, high-quality and versatile representations for 3D scenes, and enable downstream tasks such as editing, retrieval, navigation, etc.  Various neural architectures are vying for the core structure of NeRF, including the plain Multi-Layer Perceptron (MLP), sparse tensors, low-rank tensors, hashtables and their compositions. Each of these representations has its particular set of trade-offs. For example, the hashtable-based representations admit faster training and rendering but their lack of clear geometric meaning hampers downstream tasks like spatial-relation-aware editing. In this paper, we propose Progressive Volume Distillation (PVD), a systematic distillation method that allows any-to-any conversions between different architectures, including MLP, sparse or low-rank tensors, hashtables and their compositions. PVD consequently empowers downstream applications to optimally adapt the neural representations for the task at hand in a post hoc fashion. The conversions are fast, as distillation is progressively performed on different levels of volume representations, from shallower to deeper. We also employ special treatment of density to deal with its specific numerical instability problem. Empirical evidence is presented to validate our method on the NeRF-Synthetic, LLFF and TanksAndTemples datasets. For example, with PVD, an MLP-based NeRF model can be distilled from a hashtable-based Instant-NGP model at a 10~20X faster speed than being trained the original NeRF from scratch, while achieving a superior level of synthesis quality. Code is available at https://github.com/megvii-research/AAAI2023-PVD.

----

## [66] Weakly-Supervised Semantic Segmentation for Histopathology Images Based on Dataset Synthesis and Feature Consistency Constraint

**Authors**: *Zijie Fang, Yang Chen, Yifeng Wang, Zhi Wang, Xiangyang Ji, Yongbing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25136](https://doi.org/10.1609/aaai.v37i1.25136)

**Abstract**:

Tissue segmentation is a critical task in computational pathology due to its desirable ability to indicate the prognosis of cancer patients. Currently, numerous studies attempt to use image-level labels to achieve pixel-level segmentation to reduce the need for fine annotations. However, most of these methods are based on class activation map, which suffers from inaccurate segmentation boundaries. To address this problem, we propose a novel weakly-supervised tissue segmentation framework named PistoSeg, which is implemented under a fully-supervised manner by transferring tissue category labels to pixel-level masks. Firstly, a dataset synthesis method is proposed based on Mosaic transformation to generate synthesized images with pixel-level masks. Next, considering the difference between synthesized and real images, this paper devises an attention-based feature consistency, which directs the training process of a proposed pseudo-mask refining module. Finally, the refined pseudo-masks are used to train a precise segmentation model for testing. Experiments based on WSSS4LUAD and BCSS-WSSS validate that PistoSeg outperforms the state-of-the-art methods. The code is released at https://github.com/Vison307/PistoSeg.

----

## [67] Uncertainty-Aware Image Captioning

**Authors**: *Zhengcong Fei, Mingyuan Fan, Li Zhu, Junshi Huang, Xiaoming Wei, Xiaolin Wei*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25137](https://doi.org/10.1609/aaai.v37i1.25137)

**Abstract**:

It is well believed that the higher uncertainty in a word of the caption, the more inter-correlated context information is required to determine it. However, current image captioning methods usually consider the generation of all words in a sentence sequentially and equally. In this paper, we propose an uncertainty-aware image captioning framework, which parallelly and iteratively operates insertion of discontinuous candidate words between existing words from easy to difficult until converged. We hypothesize that high-uncertainty words in a sentence need more prior information to make a correct decision and should be produced at a later stage. The resulting non-autoregressive hierarchy makes the caption generation explainable and intuitive. Specifically, we utilize an image-conditioned bag-of-word model to measure the word uncertainty and apply a dynamic programming algorithm to construct the training pairs. During inference, we devise an uncertainty-adaptive parallel beam search technique that yields an empirically logarithmic time complexity. Extensive experiments on the MS COCO benchmark reveal that our approach outperforms the strong baseline and related methods on both captioning quality as well as decoding speed.

----

## [68] Unsupervised Domain Adaptation for Medical Image Segmentation by Selective Entropy Constraints and Adaptive Semantic Alignment

**Authors**: *Wei Feng, Lie Ju, Lin Wang, Kaimin Song, Xin Zhao, Zongyuan Ge*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25138](https://doi.org/10.1609/aaai.v37i1.25138)

**Abstract**:

Generalizing a deep learning model to new domains is crucial for computer-aided medical diagnosis systems. Most existing unsupervised domain adaptation methods have made significant progress in reducing the domain distribution gap through adversarial training. However, these methods may still produce overconfident but erroneous results on unseen target images. This paper proposes a new unsupervised domain adaptation framework for cross-modality medical image segmentation. Specifically, We first introduce two data augmentation approaches to generate two sets of semantics-preserving augmented images. Based on the model's predictive consistency on these two sets of augmented images, we identify reliable and unreliable pixels. We then perform a selective entropy constraint: we minimize the entropy of reliable pixels to increase their confidence while maximizing the entropy of unreliable pixels to reduce their confidence. Based on the identified reliable and unreliable pixels, we further propose an adaptive semantic alignment module which performs class-level distribution adaptation by minimizing the distance between same class prototypes between domains, where unreliable pixels are removed to derive more accurate prototypes. We have conducted extensive experiments on the cross-modality cardiac structure segmentation task. The experimental results show that the proposed method significantly outperforms the state-of-the-art comparison algorithms. Our code and data are available at https://github.com/fengweie/SE_ASA.

----

## [69] SEFormer: Structure Embedding Transformer for 3D Object Detection

**Authors**: *Xiaoyu Feng, Heming Du, Hehe Fan, Yueqi Duan, Yongpan Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25139](https://doi.org/10.1609/aaai.v37i1.25139)

**Abstract**:

Effectively  preserving and encoding structure features from objects in irregular and sparse LiDAR points is a crucial challenge to 3D object detection on the point cloud. Recently, Transformer has demonstrated promising performance on many 2D and even 3D vision tasks. Compared with the fixed and rigid convolution kernels, the self-attention mechanism in Transformer can adaptively exclude the unrelated or noisy points and is thus suitable for preserving the local spatial structure in the irregular LiDAR point cloud. However, Transformer only performs a simple sum on the point features, based on the self-attention mechanism, and all the points share the same transformation for value.  A such isotropic operation cannot capture the direction-distance-oriented local structure, which is essential for 3D object detection. In this work, we propose a Structure-Embedding transFormer (SEFormer), which can not only preserve the local structure as a traditional Transformer but also have the ability to encode the local structure. Compared to the self-attention mechanism in traditional Transformer, SEFormer learns different feature transformations for value points based on the relative directions and distances to the query point. Then we propose a SEFormer-based network for high-performance 3D object detection. Extensive experiments show that the proposed architecture can achieve SOTA results on the Waymo Open Dataset, one of the most significant 3D detection benchmarks for autonomous driving. Specifically, SEFormer achieves 79.02% mAP, which is 1.2% higher than existing works. https://github.com/tdzdog/SEFormer.

----

## [70] Exploit Domain-Robust Optical Flow in Domain Adaptive Video Semantic Segmentation

**Authors**: *Yuan Gao, Zilei Wang, Jiafan Zhuang, Yixin Zhang, Junjie Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25140](https://doi.org/10.1609/aaai.v37i1.25140)

**Abstract**:

Domain adaptive semantic segmentation aims to exploit the pixel-level annotated samples on source domain to assist the segmentation of unlabeled samples on target domain. For such a task, the key is to construct reliable supervision signals on target domain. However, existing methods can only provide unreliable supervision signals constructed by segmentation model (SegNet) that are generally domain-sensitive. In this work, we try to find a domain-robust clue to construct more reliable supervision signals. Particularly, we experimentally observe the domain-robustness of optical flow in video tasks as it mainly represents the motion characteristics of scenes. However, optical flow cannot be directly used as supervision signals of semantic segmentation since both of them essentially represent different information. To tackle this issue, we first propose a novel Segmentation-to-Flow Module (SFM) that converts semantic segmentation maps to optical flows, named the segmentation-based flow (SF), and then propose a Segmentation-based Flow Consistency (SFC) method to impose consistency between SF and optical flow, which can implicitly supervise the training of segmentation model. The extensive experiments on two challenging benchmarks demonstrate the effectiveness of our method, and it outperforms previous state-of-the-art methods with considerable performance improvement. Our code is available at https://github.com/EdenHazardan/SFC.

----

## [71] Scene-Level Sketch-Based Image Retrieval with Minimal Pairwise Supervision

**Authors**: *Ce Ge, Jingyu Wang, Qi Qi, Haifeng Sun, Tong Xu, Jianxin Liao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25141](https://doi.org/10.1609/aaai.v37i1.25141)

**Abstract**:

The sketch-based image retrieval (SBIR) task has long been researched at the instance level, where both query sketches and candidate images are assumed to contain only one dominant object. This strong assumption constrains its application, especially with the increasingly popular intelligent terminals and human-computer interaction technology. In this work, a more general scene-level SBIR task is explored, where sketches and images can both contain multiple object instances. The new general task is extremely challenging due to several factors: (i) scene-level SBIR inherently shares sketch-specific difficulties with instance-level SBIR (e.g., sparsity, abstractness, and diversity), (ii) the cross-modal similarity is measured between two partially aligned domains (i.e., not all objects in images are drawn in scene sketches), and (iii) besides instance-level visual similarity, a more complex multi-dimensional scene-level feature matching problem is imposed (including appearance, semantics, layout, etc.). Addressing these challenges, a novel Conditional Graph Autoencoder model is proposed to deal with scene-level sketch-images retrieval. More importantly, the model can be trained with only pairwise supervision, which distinguishes our study from others in that elaborate instance-level annotations (for example, bounding boxes) are no longer required. Extensive experiments confirm the ability of our model to robustly retrieve multiple related objects at the scene level and exhibit superior performance beyond strong competitors.

----

## [72] Causal Intervention for Human Trajectory Prediction with Cross Attention Mechanism

**Authors**: *Chunjiang Ge, Shiji Song, Gao Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25142](https://doi.org/10.1609/aaai.v37i1.25142)

**Abstract**:

Human trajectory Prediction (HTP) in complex social environments plays a crucial and fundamental role in artificial intelligence systems. Conventional methods make use of both history behaviors and social interactions to forecast future trajectories. However, we demonstrate that the social environment is a confounder that misleads the model to learn spurious correlations between history and future trajectories. To end this, we first formulate the social environment, history and future trajectory variables into a structural causal model to analyze the causalities among them. Based on causal intervention rather than conventional likelihood, we propose a Social Environment ADjustment (SEAD) method, to remove the confounding effect of the social environment. The core of our method is implemented by a Social Cross Attention (SCA) module, which is universal, simple and effective. Our method has consistent improvements on ETH-UCY datasets with three baseline models and achieves competitive performances with existing methods.

----

## [73] Point-Teaching: Weakly Semi-supervised Object Detection with Point Annotations

**Authors**: *Yongtao Ge, Qiang Zhou, Xinlong Wang, Chunhua Shen, Zhibin Wang, Hao Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25143](https://doi.org/10.1609/aaai.v37i1.25143)

**Abstract**:

Point annotations are considerably more time-efficient than bounding box annotations. However, how to use cheap point annotations to boost the performance of semi-supervised object detection is still an open question. In this work, we present Point-Teaching, a weakly- and semi-supervised object detection framework to fully utilize the point annotations. Specifically, we propose a Hungarian-based point-matching method to generate pseudo labels for point-annotated images. We further propose multiple instance learning (MIL) approaches at the level of images and points to supervise the object detector with point annotations. Finally, we propose a simple data augmentation, named Point-Guided Copy-Paste, to reduce the impact of those unmatched points. Experiments demonstrate the effectiveness of our method on a few datasets and various data regimes. In particular, Point-Teaching outperforms the previous best method Group R-CNN by 3.1 AP with 5% fully labeled data and 2.3 AP with 30% fully labeled data on the MS COCO dataset. We believe that our proposed framework can largely lower the bar of learning accurate object detectors and pave the way for its broader applications. The code is available at https://github.com/YongtaoGe/Point-Teaching.

----

## [74] Progressive Multi-View Human Mesh Recovery with Self-Supervision

**Authors**: *Xuan Gong, Liangchen Song, Meng Zheng, Benjamin Planche, Terrence Chen, Junsong Yuan, David S. Doermann, Ziyan Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25144](https://doi.org/10.1609/aaai.v37i1.25144)

**Abstract**:

To date, little attention has been given to multi-view 3D human mesh estimation, despite real-life applicability (e.g., motion capture, sport analysis) and robustness to single-view ambiguities. Existing solutions typically suffer from poor generalization performance to new settings, largely due to the limited diversity of image/3D-mesh pairs in multi-view training data. To address this shortcoming, people have explored the use of synthetic images. But besides the usual impact of visual gap between rendered and target data, synthetic-data-driven multi-view estimators also suffer from overfitting to the camera viewpoint distribution sampled during training which usually differs from real-world distributions. Tackling both challenges, we propose a novel simulation-based training pipeline for multi-view human mesh recovery, which (a) relies on intermediate 2D representations which are more robust to synthetic-to-real domain gap; (b) leverages learnable calibration and triangulation to adapt to more diversified camera setups; and (c) progressively aggregates multi-view information in a canonical 3D space to remove ambiguities in 2D representations. Through extensive benchmarking, we demonstrate the superiority of the proposed solution especially for unseen in-the-wild scenarios.

----

## [75] Incremental Image De-raining via Associative Memory

**Authors**: *Yi Gu, Chao Wang, Jie Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25145](https://doi.org/10.1609/aaai.v37i1.25145)

**Abstract**:

While deep learning models have achieved the state-of-the-art performance on single-image rain removal, most methods only consider learning fixed mapping rules on the single synthetic dataset for lifetime. This limits the real-life application as iterative optimization may change mapping rules and training samples. However, when models learn a sequence of datasets in multiple incremental steps, they are susceptible to catastrophic forgetting that adapts to new incremental episodes while failing to preserve previously acquired mapping rules. In this paper, we argue the importance of sample diversity in the episodes on the iterative optimization, and propose a novel memory management method, Associative Memory, to achieve incremental image de-raining. It bridges connections between current and past episodes for feature reconstruction by sampling domain mappings of past learning steps, and guides the learning to trace the current pathway back to the historical environment without storing extra data. Experiments demonstrate that our method can achieve better performance than existing approaches on both inhomogeneous and incremental datasets within the spectrum of highly compact systems.

----

## [76] Flexible 3D Lane Detection by Hierarchical Shape Matching

**Authors**: *Zhihao Guan, Ruixin Liu, Zejian Yuan, Ao Liu, Kun Tang, Tong Zhou, Erlong Li, Chao Zheng, Shuqi Mei*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25146](https://doi.org/10.1609/aaai.v37i1.25146)

**Abstract**:

As one of the basic while vital technologies for HD map construction, 3D lane detection is still an open problem due to varying visual conditions, complex typologies, and strict demands for precision. In this paper, an end-to-end flexible and hierarchical lane detector is proposed to precisely predict 3D lane lines from point clouds. Specifically, we design a hierarchical network predicting flexible representations of lane shapes at different levels, simultaneously collecting global instance semantics and avoiding local errors. In the global scope, we propose to regress parametric curves w.r.t adaptive axes that help to make more robust predictions towards complex scenes, while in the local vision the structure of lane segment is detected in each of the dynamic anchor cells sampled along the global predicted curves. Moreover, corresponding global and local shape matching losses and anchor cell generation strategies are designed. Experiments on two datasets show that we overwhelm current top methods under high precision standards, and full ablation studies also verify each part of our method. Our codes will be released at https://github.com/Doo-do/FHLD.

----

## [77] Underwater Ranker: Learn Which Is Better and How to Be Better

**Authors**: *Chunle Guo, Ruiqi Wu, Xin Jin, Linghao Han, Weidong Zhang, Zhi Chai, Chongyi Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25147](https://doi.org/10.1609/aaai.v37i1.25147)

**Abstract**:

In this paper, we present a ranking-based underwater image quality assessment (UIQA) method, abbreviated as URanker. The URanker is built on the efficient conv-attentional image Transformer. In terms of underwater images, we specially devise (1) the histogram prior that embeds the color distribution of an underwater image as histogram token to attend global degradation and (2) the dynamic cross-scale correspondence to model local degradation. The final prediction depends on the class tokens from different scales, which comprehensively considers multi-scale dependencies. With the margin ranking loss, our URanker can accurately rank the order of underwater images of the same scene enhanced by different underwater image enhancement (UIE) algorithms according to their visual quality. To achieve that, we also contribute a dataset, URankerSet, containing sufficient results enhanced by different UIE algorithms and the corresponding perceptual rankings, to train our URanker. Apart from the good performance of URanker, we found that a simple U-shape UIE network can obtain promising performance when it is coupled with our pre-trained URanker as additional supervision. In addition, we also propose a normalization tail that can significantly improve the performance of UIE networks. Extensive experiments demonstrate the state-of-the-art performance of our method. The key designs of our method are discussed. Our code and dataset are available at https://li-chongyi.github.io/URanker_files/.

----

## [78] ShadowFormer: Global Context Helps Shadow Removal

**Authors**: *Lanqing Guo, Siyu Huang, Ding Liu, Hao Cheng, Bihan Wen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25148](https://doi.org/10.1609/aaai.v37i1.25148)

**Abstract**:

Recent deep learning methods have achieved promising results in image shadow removal. However, most of the existing approaches focus on working locally within shadow and non-shadow regions, resulting in severe artifacts around the shadow boundaries as well as inconsistent illumination between shadow and non-shadow regions. It is still challenging for the deep shadow removal model to exploit the global contextual correlation between shadow and non-shadow regions. In this work, we first propose a Retinex-based shadow model, from which we derive a novel transformer-based network, dubbed ShandowFormer, to exploit non-shadow regions to help shadow region restoration. A multi-scale channel attention framework is employed to hierarchically capture the global information. Based on that, we propose a Shadow-Interaction Module (SIM) with Shadow-Interaction Attention (SIA) in the bottleneck stage to effectively model the context correlation between shadow and non-shadow regions. We conduct extensive experiments on three popular public datasets, including ISTD, ISTD+, and SRD, 
to evaluate the proposed method. Our method achieves state-of-the-art performance by using up to 150X fewer model parameters.

----

## [79] RAFaRe: Learning Robust and Accurate Non-parametric 3D Face Reconstruction from Pseudo 2D&3D Pairs

**Authors**: *Longwei Guo, Hao Zhu, Yuanxun Lu, Menghua Wu, Xun Cao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25149](https://doi.org/10.1609/aaai.v37i1.25149)

**Abstract**:

We propose a robust and accurate non-parametric method for single-view 3D face reconstruction (SVFR). While tremendous efforts have been devoted to parametric SVFR, a visible gap still lies between the result 3D shape and the ground truth. We believe there are two major obstacles: 1) the representation of the parametric model is limited to a certain face database; 2) 2D images and 3D shapes in the fitted datasets are distinctly misaligned. To resolve these issues, a large-scale pseudo 2D&3D dataset is created by first rendering the detailed 3D faces, then swapping the face in the wild images with the rendered face. These pseudo 2D&3D pairs are created from publicly available datasets which eliminate the gaps between 2D and 3D data while covering diverse appearances, poses, scenes, and illumination. We further propose a non-parametric scheme to learn a well-generalized SVFR model from the created dataset, and the proposed hierarchical signed distance function turns out to be effective in predicting middle-scale and small-scale 3D facial geometry. Our model outperforms previous methods on FaceScape-wild/lab and MICC benchmarks and is well generalized to various appearances, poses, expressions, and in-the-wild environments. The code is released at https://github.com/zhuhao-nju/rafare.

----

## [80] RankDNN: Learning to Rank for Few-Shot Learning

**Authors**: *Qianyu Guo, Haotong Gong, Xujun Wei, Yanwei Fu, Yizhou Yu, Wenqiang Zhang, Weifeng Ge*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25150](https://doi.org/10.1609/aaai.v37i1.25150)

**Abstract**:

This paper introduces a new few-shot learning pipeline that
casts relevance ranking for image retrieval as binary ranking
relation classification. In comparison to image classification,
ranking relation classification is sample efficient and
domain agnostic. Besides, it provides a new perspective on
few-shot learning and is complementary to state-of-the-art
methods. The core component of our deep neural network is
a simple MLP, which takes as input an image triplet encoded
as the difference between two vector-Kronecker products,
and outputs a binary relevance ranking order. The proposed
RankMLP can be built on top of any state-of-the-art feature
extractors, and our entire deep neural network is called
the ranking deep neural network, or RankDNN. Meanwhile,
RankDNN can be flexibly fused with other post-processing
methods. During the meta test, RankDNN ranks support images
according to their similarity with the query samples,
and each query sample is assigned the class label of its
nearest neighbor. Experiments demonstrate that RankDNN
can effectively improve the performance of its baselines
based on a variety of backbones and it outperforms previous
state-of-the-art algorithms on multiple few-shot learning
benchmarks, including miniImageNet, tieredImageNet,
Caltech-UCSD Birds, and CIFAR-FS. Furthermore, experiments
on the cross-domain challenge demonstrate the superior
transferability of RankDNN.The code is available at:
https://github.com/guoqianyu-alberta/RankDNN.

----

## [81] Social Relation Reasoning Based on Triangular Constraints

**Authors**: *Yunfei Guo, Fei Yin, Wei Feng, Xudong Yan, Tao Xue, Shuqi Mei, Cheng-Lin Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25151](https://doi.org/10.1609/aaai.v37i1.25151)

**Abstract**:

Social networks are essentially in a graph structure where persons act as nodes and the edges connecting nodes denote social relations. The prediction of social relations, therefore, relies on the context in graphs to model the higher-order constraints among relations, which has not been exploited sufficiently by previous works, however. In this paper, we formulate the paradigm of the higher-order constraints in social relations into triangular relational closed-loop structures, i.e., triangular constraints, and further introduce the triangular reasoning graph attention network (TRGAT). Our TRGAT employs the attention mechanism to aggregate features with triangular constraints in the graph, thereby exploiting the higher-order context to reason social relations iteratively. Besides, to acquire better feature representations of persons, we introduce node contrastive learning into relation reasoning. Experimental results show that our method outperforms existing approaches significantly, with higher accuracy and better consistency in generating social relation graphs.

----

## [82] CALIP: Zero-Shot Enhancement of CLIP with Parameter-Free Attention

**Authors**: *Ziyu Guo, Renrui Zhang, Longtian Qiu, Xianzheng Ma, Xupeng Miao, Xuming He, Bin Cui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25152](https://doi.org/10.1609/aaai.v37i1.25152)

**Abstract**:

Contrastive Language-Image Pre-training (CLIP) has been shown to learn visual representations with promising zero-shot performance. To further improve its downstream accuracy, existing works propose additional learnable modules upon CLIP and fine-tune them by few-shot training sets. However, the resulting extra training cost and data requirement severely hinder the efficiency for model deployment and knowledge transfer. In this paper, we introduce a free-lunch enhancement method, CALIP, to boost CLIP's zero-shot performance via a parameter-free attention module. Specifically, we guide visual and textual representations to interact with each other and explore cross-modal informative features via attention. As the pre-training has largely reduced the embedding distances between two modalities, we discard all learnable parameters in the attention and bidirectionally update the multi-modal features, enabling the whole process to be parameter-free and training-free. In this way, the images are blended with textual-aware signals and the text representations become visual-guided for better adaptive zero-shot alignment. We evaluate CALIP on various benchmarks of 14 datasets for both 2D image and 3D point cloud few-shot classification, showing consistent zero-shot performance improvement over CLIP. Based on that, we further insert a small number of linear layers in CALIP's attention module and verify our robustness under the few-shot settings, which also achieves leading performance compared to existing methods. Those extensive experiments demonstrate the superiority of our approach for efficient enhancement of CLIP. Code is available at https://github.com/ZiyuGuo99/CALIP.

----

## [83] Few-Shot Object Detection via Variational Feature Aggregation

**Authors**: *Jiaming Han, Yuqiang Ren, Jian Ding, Ke Yan, Gui-Song Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25153](https://doi.org/10.1609/aaai.v37i1.25153)

**Abstract**:

As few-shot object detectors are often trained with abundant base samples and fine-tuned on few-shot novel examples, the learned models are usually biased to base classes and sensitive to the variance of novel examples. To address this issue, we propose a meta-learning framework with two novel feature aggregation schemes. More precisely, we first present a Class-Agnostic Aggregation (CAA) method, where the query and support features can be aggregated regardless of their categories. The interactions between different classes encourage class-agnostic representations and reduce confusion between base and novel classes.
Based on the CAA, we then propose a Variational Feature Aggregation (VFA) method, which encodes support examples into class-level support features for robust feature aggregation. We use a variational autoencoder to estimate class distributions and sample variational features from distributions that are more robust to the variance of support examples. Besides, we decouple classification and regression tasks so that VFA is performed on the classification branch without affecting object localization. Extensive experiments on PASCAL VOC and COCO demonstrate that our method significantly outperforms a strong baseline (up to 16%) and previous state-of-the-art methods (4% in average).

----

## [84] Generating Transferable 3D Adversarial Point Cloud via Random Perturbation Factorization

**Authors**: *Bangyan He, Jian Liu, Yiming Li, Siyuan Liang, Jingzhi Li, Xiaojun Jia, Xiaochun Cao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25154](https://doi.org/10.1609/aaai.v37i1.25154)

**Abstract**:

Recent studies have demonstrated that existing deep neural networks (DNNs) on 3D point clouds are vulnerable to adversarial examples, especially under the white-box settings where the adversaries have access to model parameters. However, adversarial 3D point clouds generated by existing white-box methods have limited transferability across different DNN architectures. They have only minor threats in real-world scenarios under the black-box settings where the adversaries can only query the deployed victim model. In this paper, we revisit the transferability of adversarial 3D point clouds. We observe that an adversarial perturbation can be randomly factorized into two sub-perturbations, which are also likely to be adversarial perturbations. It motivates us to consider the effects of the perturbation and its sub-perturbations simultaneously to increase the transferability for sub-perturbations also contain helpful information. In this paper, we propose a simple yet effective attack method to generate more transferable adversarial 3D point clouds. Specifically, rather than simply optimizing the loss of perturbation alone, we combine it with its random factorization. We conduct experiments on benchmark dataset, verifying our method's effectiveness in increasing transferability while preserving high efficiency.

----

## [85] Target-Aware Tracking with Long-Term Context Attention

**Authors**: *Kaijie He, Canlong Zhang, Sheng Xie, Zhixin Li, Zhiwen Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25155](https://doi.org/10.1609/aaai.v37i1.25155)

**Abstract**:

Most deep trackers still follow the guidance of the siamese paradigms and use a template that contains only the target without any contextual information, which makes it difficult for the tracker to cope with large appearance changes, rapid target movement, and attraction from similar objects. To alleviate the above problem, we propose a long-term context attention (LCA) module that can perform extensive information fusion on the target and its context from long-term frames, and calculate the target correlation while enhancing target features. The complete contextual information contains the location of the target as well as the state around the target. LCA uses the target state from the previous frame to exclude the interference of similar objects and complex backgrounds, thus accurately locating the target and enabling the tracker to obtain higher robustness and regression accuracy. By embedding the LCA module in Transformer, we build a powerful online tracker with a target-aware backbone, termed as TATrack. In addition, we propose a dynamic online update algorithm based on the classification confidence of historical information without additional calculation burden. Our tracker achieves state-of-the-art performance on multiple benchmarks, with 71.1% AUC, 89.3% NP, and 73.0% AO on LaSOT, TrackingNet, and GOT-10k. The code and trained models are available on https://github.com/hekaijie123/TATrack.

----

## [86] Weakly-Supervised Camouflaged Object Detection with Scribble Annotations

**Authors**: *Ruozhen He, Qihua Dong, Jiaying Lin, Rynson W. H. Lau*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25156](https://doi.org/10.1609/aaai.v37i1.25156)

**Abstract**:

Existing camouflaged object detection (COD) methods rely heavily on large-scale datasets with pixel-wise annotations. However, due to the ambiguous boundary, annotating camouflage objects pixel-wisely is very time-consuming and labor-intensive, taking ~60mins to label one image. In this paper, we propose the first weakly-supervised COD method, using scribble annotations as supervision. To achieve this, we first relabel 4,040 images in existing camouflaged object datasets with scribbles, which takes ~10s to label one image. As scribble annotations only describe the primary structure of objects without details, for the network to learn to localize the boundaries of camouflaged objects, we propose a novel consistency loss composed of two parts: a cross-view loss to attain reliable consistency over different images, and an inside-view loss to maintain consistency inside a single prediction map. Besides, we observe that humans use semantic information to segment regions near the boundaries of camouflaged objects. Hence, we further propose a feature-guided loss, which includes visual features directly extracted from images and semantically significant features captured by the model. Finally, we propose a novel network for COD via scribble learning on structural information and semantic relations. Our network has two novel modules: the local-context contrasted (LCC) module, which mimics visual inhibition to enhance image contrast/sharpness and expand the scribbles into potential camouflaged regions, and the logical semantic relation (LSR) module, which analyzes the semantic relation to determine the regions representing the camouflaged object. Experimental results show that our model outperforms relevant SOTA methods on three COD benchmarks with an average improvement of 11.0% on MAE, 3.2% on S-measure, 2.5% on E-measure, and 4.4% on weighted F-measure.

----

## [87] Efficient Mirror Detection via Multi-Level Heterogeneous Learning

**Authors**: *Ruozhen He, Jiaying Lin, Rynson W. H. Lau*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25157](https://doi.org/10.1609/aaai.v37i1.25157)

**Abstract**:

We present HetNet (Multi-level Heterogeneous Network), a highly efficient mirror detection network. Current mirror detection methods focus more on performance than efficiency, limiting the real-time applications (such as drones). Their lack of efficiency is aroused by the common design of adopting homogeneous modules at different levels, which ignores the difference between different levels of features. In contrast, HetNet detects potential mirror regions initially through low-level understandings (e.g., intensity contrasts) and then combines with high-level understandings (contextual discontinuity for instance) to finalize the predictions. To perform accurate yet efficient mirror detection, HetNet follows an effective architecture that obtains specific information at different stages to detect mirrors. We further propose a multi-orientation intensity-based contrasted module (MIC) and a reflection semantic logical module (RSL), equipped on HetNet, to predict potential mirror regions by low-level understandings and analyze semantic logic in scenarios by high-level understandings, respectively. Compared to the state-of-the-art method, HetNet runs 664% faster and draws an average performance gain of 8.9% on MAE, 3.1% on IoU, and 2.0% on F-measure on two mirror detection benchmarks. The code is available at https://github.com/Catherine-R-He/HetNet.

----

## [88] TransVCL: Attention-Enhanced Video Copy Localization Network with Flexible Supervision

**Authors**: *Sifeng He, Yue He, Minlong Lu, Chen Jiang, Xudong Yang, Feng Qian, Xiaobo Zhang, Lei Yang, Jiandong Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25158](https://doi.org/10.1609/aaai.v37i1.25158)

**Abstract**:

Video copy localization aims to precisely localize all the copied segments within a pair of untrimmed videos in video retrieval applications. Previous methods typically start from frame-to-frame similarity matrix generated by cosine similarity between frame-level features of the input video pair, and then detect and refine the boundaries of copied segments on similarity matrix under temporal constraints. In this paper, we propose TransVCL: an attention-enhanced video copy localization network, which is optimized directly from initial frame-level features and trained end-to-end with three main components: a customized Transformer for feature enhancement, a correlation and softmax layer for similarity matrix generation, and a temporal alignment module for copied segments localization. In contrast to previous methods demanding the handcrafted similarity matrix, TransVCL incorporates long-range temporal information between feature sequence pair using self- and cross- attention layers. With the joint design and optimization of three components, the similarity matrix can be learned to present more discriminative copied patterns, leading to significant improvements over previous methods on segment-level labeled datasets (VCSL and VCDB). Besides the state-of-the-art performance in fully supervised setting, the attention architecture facilitates TransVCL to further exploit unlabeled or simply video-level labeled data. Additional experiments of supplementing video-level labeled datasets including SVD and FIVR reveal the high flexibility of TransVCL from full supervision to semi-supervision (with or without video-level annotation). Code is publicly available at https://github.com/transvcl/TransVCL.

----

## [89] Open-Vocabulary Multi-Label Classification via Multi-Modal Knowledge Transfer

**Authors**: *Sunan He, Taian Guo, Tao Dai, Ruizhi Qiao, Xiujun Shu, Bo Ren, Shu-Tao Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25159](https://doi.org/10.1609/aaai.v37i1.25159)

**Abstract**:

Real-world recognition system often encounters the challenge of unseen labels. To identify such unseen labels, multi-label zero-shot learning (ML-ZSL) focuses on transferring knowledge by a pre-trained textual label embedding (e.g., GloVe). However, such methods only exploit single-modal knowledge from a language model, while ignoring the rich semantic information inherent in image-text pairs. Instead, recently developed open-vocabulary (OV) based methods succeed in exploiting such information of image-text pairs in object detection, and achieve impressive performance. Inspired by the success of OV-based methods, we propose a novel open-vocabulary framework, named multi-modal knowledge transfer (MKT), for multi-label classification. Specifically, our method exploits multi-modal knowledge of image-text pairs based on a vision and language pre-training (VLP) model. To facilitate transferring the image-text matching ability of VLP model, knowledge distillation is employed to guarantee the consistency of image and label embeddings, along with prompt tuning to further update the label embeddings. To further enable the recognition of multiple objects, a simple but effective two-stream module is developed to capture both local and global features. Extensive experimental results show that our method significantly outperforms state-of-the-art methods on public benchmark datasets.

----

## [90] Parameter-Efficient Model Adaptation for Vision Transformers

**Authors**: *Xuehai He, Chunyuan Li, Pengchuan Zhang, Jianwei Yang, Xin Eric Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25160](https://doi.org/10.1609/aaai.v37i1.25160)

**Abstract**:

In computer vision, it has achieved great transfer learning performance via adapting large-scale pretrained vision models (e.g., vision transformers) to downstream tasks. Common approaches for model adaptation either update all model parameters or leverage linear probes. In this paper, we aim to study parameter-efficient model adaptation strategies for vision transformers on the image classification task. We formulate efficient model adaptation as a subspace training problem and perform a comprehensive benchmarking over different efficient adaptation methods. We conduct an empirical study on each efficient model adaptation method focusing on its performance alongside parameter cost. Furthermore, we propose a parameter-efficient model adaptation framework, which first selects submodules by measuring local intrinsic dimensions and then projects them into subspace for further decomposition via a novel Kronecker Adaptation method. We analyze and compare our method with a diverse set of baseline model adaptation methods (including state-of-the-art methods for pretrained language models). Our method performs the best in terms of the tradeoff between accuracy and parameter efficiency across 20 datasets under the few-shot setting and 7 image classification datasets under the full-shot setting.

----

## [91] DarkFeat: Noise-Robust Feature Detector and Descriptor for Extremely Low-Light RAW Images

**Authors**: *Yuze He, Yubin Hu, Wang Zhao, Jisheng Li, Yong-Jin Liu, Yuxing Han, Jiangtao Wen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25161](https://doi.org/10.1609/aaai.v37i1.25161)

**Abstract**:

Low-light visual perception, such as SLAM or SfM at night, has received increasing attention, in which keypoint detection and local feature description play an important role. Both handcraft designs and machine learning methods have been widely studied for local feature detection and description, however, the performance of existing methods degrades in the extreme low-light scenarios in a certain degree, due to the low signal-to-noise ratio in images. To address this challenge, images in RAW format that retain more raw sensing information have been considered in recent works with a denoise-then-detect scheme. However, existing denoising methods are still insufficient for RAW images and heavily time-consuming, which limits the practical applications of such scheme. In this paper, we propose DarkFeat, a deep learning model which directly detects and describes local features from extreme low-light RAW images in an end-to-end manner.  A novel noise robustness map and selective suppression constraints are proposed to effectively mitigate the influence of noise and extract more reliable keypoints. Furthermore, a customized pipeline of synthesizing dataset containing low-light RAW image matching pairs is proposed to extend end-to-end training. Experimental results show that DarkFeat achieves state-of-the-art performance on both indoor and outdoor parts of the challenging MID benchmark, outperforms the denoise-then-detect methods and significantly reduces computational costs up to 70%. Code is available at https://github.com/THU-LYJ-Lab/DarkFeat.

----

## [92] GAM: Gradient Attention Module of Optimization for Point Clouds Analysis

**Authors**: *Haotian Hu, Fanyi Wang, Zhiwang Zhang, Yaonong Wang, Laifeng Hu, Yanhao Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25162](https://doi.org/10.1609/aaai.v37i1.25162)

**Abstract**:

In the point cloud analysis task, the existing local feature aggregation descriptors (LFAD) do not fully utilize the neighborhood information of center points. Previous methods only use the distance information to constrain the local aggregation process, which is easy to be affected by abnormal points and cannot adequately fit the original geometry of the point cloud. This paper argues that fine-grained geometric information (FGGI) plays an important role in the aggregation of local features. Based on this, we propose a gradient-based local attention module to address the above problem, which is called Gradient Attention Module (GAM). GAM simplifies the process of extracting the gradient information in the neighborhood to explicit representation using the Zenith Angle matrix and Azimuth Angle matrix, which makes the module 35X faster. The comprehensive experiments on the ScanObjectNN dataset, ShapeNet dataset, S3DIS dataset, Modelnet40 dataset, and KITTI dataset demonstrate the effectiveness, efficientness, and generalization of our newly proposed GAM for 3D point cloud analysis. Especially in S3DIS, GAM achieves the highest index in the current point-based model with mIoU/OA/mAcc of 74.4%/90.6%/83.2%.

----

## [93] Self-Supervised Learning for Multilevel Skeleton-Based Forgery Detection via Temporal-Causal Consistency of Actions

**Authors**: *Liang Hu, Dora D. Liu, Qi Zhang, Usman Naseem, Zhong Yuan Lai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25163](https://doi.org/10.1609/aaai.v37i1.25163)

**Abstract**:

Skeleton-based human action recognition and analysis have become increasingly attainable in many areas, such as security surveillance and anomaly detection. Given the prevalence of skeleton-based applications, tampering attacks on human skeletal features have emerged very recently. In particular, checking the temporal inconsistency and/or incoherence (TII) in the skeletal sequence of human action is a principle of forgery detection. To this end, we propose an approach to self-supervised learning of the temporal causality behind human action, which can effectively check TII in skeletal sequences. Especially, we design a multilevel skeleton-based forgery detection framework to recognize the forgery on frame level, clip level, and action level in terms of learning the corresponding temporal-causal skeleton representations for each level. Specifically, a hierarchical graph convolution network architecture is designed to learn low-level skeleton representations based on physical skeleton connections and high-level action representations based on temporal-causal dependencies for specific actions. Extensive experiments consistently show state-of-the-art results on multilevel forgery detection tasks and superior performance of our framework compared to current competing methods.

----

## [94] Self-Emphasizing Network for Continuous Sign Language Recognition

**Authors**: *Lianyu Hu, Liqing Gao, Zekang Liu, Wei Feng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25164](https://doi.org/10.1609/aaai.v37i1.25164)

**Abstract**:

Hand and face play an important role in expressing sign language. Their features are usually especially leveraged to improve system performance. However, to effectively extract visual representations and capture trajectories for hands and face, previous methods always come at high computations with increased training complexity. They usually employ extra heavy pose-estimation networks to locate human body keypoints or rely on additional pre-extracted heatmaps for supervision. To relieve this problem, we propose a self-emphasizing network (SEN) to emphasize informative spatial regions in a self-motivated way, with few extra computations and without additional expensive supervision. Specifically, SEN first employs a lightweight subnetwork to incorporate local spatial-temporal features to identify informative regions, and then dynamically augment original features via attention maps. It's also observed that not all frames contribute equally to recognition. We present a temporal self-emphasizing module to adaptively emphasize those discriminative frames and suppress redundant ones. A comprehensive comparison with previous methods equipped with hand and face features demonstrates the superiority of our method, even though they always require huge computations and rely on expensive extra supervision. Remarkably, with few extra computations, SEN achieves new state-of-the-art accuracy on four large-scale datasets, PHOENIX14, PHOENIX14-T, CSL-Daily, and CSL. Visualizations verify the effects of SEN on emphasizing informative spatial and temporal features. Code is available at https://github.com/hulianyuyy/SEN_CSLR

----

## [95] Store and Fetch Immediately: Everything Is All You Need for Space-Time Video Super-resolution

**Authors**: *Mengshun Hu, Kui Jiang, Zhixiang Nie, Jiahuan Zhou, Zheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25165](https://doi.org/10.1609/aaai.v37i1.25165)

**Abstract**:

Existing space-time video super-resolution (ST-VSR) methods fail to achieve high-quality reconstruction since they fail to fully explore the spatial-temporal correlations, long-range components in particular. Although the recurrent structure for ST-VSR adopts bidirectional propagation to aggregate information from the entire video, collecting the temporal information between the past and future via one-stage representations inevitably loses the long-range relations. To alleviate the limitation, this paper proposes an immediate storeand-fetch network to promote long-range correlation learning, where the stored information from the past and future can be refetched to help the representation of the current frame. Specifically, the proposed network consists of two modules: a backward recurrent module (BRM) and a forward recurrent module (FRM). The former first performs backward inference from future to past, while storing future super-resolution (SR) information for each frame. Following that, the latter performs forward inference from past to future to super-resolve all frames, while storing past SR information for each frame. Since FRM inherits SR information from BRM, therefore, spatial and temporal information from the entire video sequence is immediately stored and fetched, which allows drastic improvement for ST-VSR. Extensive experiments both on ST-VSR and space video super-resolution (S-VSR) as well as time video super-resolution (T-VSR) have demonstrated the effectiveness of our proposed method over other state-of-the-art methods on public datasets. Code is available https://github.com/hhhhhumengshun/SFI-STVR

----

## [96] PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models against Adversarial Examples

**Authors**: *Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25166](https://doi.org/10.1609/aaai.v37i1.25166)

**Abstract**:

Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point.
Extensive experiments against different premier point cloud completion networks show that PointCA can cause the performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept  below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.

----

## [97] High-Resolution Iterative Feedback Network for Camouflaged Object Detection

**Authors**: *Xiaobin Hu, Shuo Wang, Xuebin Qin, Hang Dai, Wenqi Ren, Donghao Luo, Ying Tai, Ling Shao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25167](https://doi.org/10.1609/aaai.v37i1.25167)

**Abstract**:

Spotting camouflaged objects that are visually assimilated into the background is tricky for both object detection algorithms and humans who are usually confused or cheated by the perfectly intrinsic similarities between the foreground objects and the background surroundings. To tackle this challenge, we aim to extract the high-resolution texture details to avoid the detail degradation that causes blurred vision in edges and boundaries. We introduce a novel HitNet to refine the low-resolution representations by high-resolution features in an iterative feedback manner, essentially a global loop-based connection among the multi-scale resolutions. To design better feedback feature ﬂow and avoid the feature corruption caused by recurrent path, an iterative feedback strategy is proposed to impose more constraints on each feedback connection. Extensive experiments on four challenging datasets demonstrate that our HitNet breaks the performance bottleneck and achieves significant improvements compared with 29 state-of-the-art methods. In addition, to address the data scarcity in camouflaged scenarios, we provide an application example to convert the salient objects to camouflaged objects, thereby generating more camouflaged training samples from the diverse salient object datasets. Code will be made publicly available.

----

## [98] Leveraging Sub-class Discimination for Compositional Zero-Shot Learning

**Authors**: *Xiaoming Hu, Zilei Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25168](https://doi.org/10.1609/aaai.v37i1.25168)

**Abstract**:

Compositional Zero-Shot Learning (CZSL) aims at identifying unseen compositions composed of previously seen attributes and objects during the test phase. In real images, the visual appearances of attributes and objects (primitive concepts) generally interact with each other. Namely, the visual appearances of an attribute may change when composed with different objects, and vice versa. But previous works overlook this important property. In this paper, we introduce a simple yet effective approach with leveraging sub-class discrimination. Specifically, we define the primitive concepts in different compositions as sub-classes, and then maintain the sub-class discrimination to address the above challenge. More specifically, inspired by the observation that the composed recognition models could account for the differences across sub-classes, we first propose to impose the embedding alignment between the composed and disentangled recognition to incorporate sub-class discrimination at the feature level. Then we develop the prototype modulator networks to adjust the class prototypes w.r.t. the composition information, which can enhance sub-class discrimination at the classifier level. We conduct extensive experiments on the challenging benchmark datasets, and the considerable performance improvement over state-of-the-art approaches is achieved, which indicates the effectiveness of our method. Our code is available at https://github.com/hxm97/SCD-CZSL.

----

## [99] GPTR: Gestalt-Perception Transformer for Diagram Object Detection

**Authors**: *Xin Hu, Lingling Zhang, Jun Liu, Jinfu Fan, Yang You, Yaqiang Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25169](https://doi.org/10.1609/aaai.v37i1.25169)

**Abstract**:

Diagram object detection is the key basis of practical applications such as textbook question answering. Because the diagram mainly consists of simple lines and color blocks, its visual features are sparser than those of natural images. In addition, diagrams usually express diverse knowledge, in which there are many low-frequency object categories in diagrams. These lead to the fact that traditional data-driven detection model is not suitable for diagrams. In this work, we propose a gestalt-perception transformer model for diagram object detection, which is based on an encoder-decoder architecture. Gestalt perception contains a series of laws to explain human perception, that the human visual system tends to perceive patches in an image that are similar, close or connected without abrupt directional changes as a perceptual whole object. Inspired by these thoughts, we build a gestalt-perception graph in transformer encoder, which is composed of diagram patches as nodes and the relationships between patches as edges. This graph aims to group these patches into objects via laws of similarity, proximity, and smoothness implied in these edges, so that the meaningful objects can be effectively detected. The experimental results demonstrate that the proposed GPTR achieves the best results in the diagram object detection task. Our model also obtains comparable results over the competitors in natural image object detection.

----

## [100] Resolving Task Confusion in Dynamic Expansion Architectures for Class Incremental Learning

**Authors**: *Bingchen Huang, Zhineng Chen, Peng Zhou, Jiayin Chen, Zuxuan Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25170](https://doi.org/10.1609/aaai.v37i1.25170)

**Abstract**:

The dynamic expansion architecture is becoming popular in class incremental learning, mainly due to its advantages in alleviating catastrophic forgetting. However, task confu- sion is not well assessed within this framework, e.g., the discrepancy between classes of different tasks is not well learned (i.e., inter-task confusion, ITC), and certain prior- ity is still given to the latest class batch (i.e., old-new con- fusion, ONC). We empirically validate the side effects of the two types of confusion. Meanwhile, a novel solution called Task Correlated Incremental Learning (TCIL) is pro- posed to encourage discriminative and fair feature utilization across tasks. TCIL performs a multi-level knowledge distil- lation to propagate knowledge learned from old tasks to the new one. It establishes information flow paths at both fea- ture and logit levels, enabling the learning to be aware of old classes. Besides, attention mechanism and classifier re- scoring are applied to generate more fair classification scores. We conduct extensive experiments on CIFAR100 and Ima- geNet100 datasets. The results demonstrate that TCIL con- sistently achieves state-of-the-art accuracy. It mitigates both ITC and ONC, while showing advantages in battle with catas- trophic forgetting even no rehearsal memory is reserved. Source code: https://github.com/YellowPancake/TCIL.

----

## [101] ClassFormer: Exploring Class-Aware Dependency with Transformer for Medical Image Segmentation

**Authors**: *Huimin Huang, Shiao Xie, Lanfen Lin, Ruofeng Tong, Yen-Wei Chen, Hong Wang, Yuexiang Li, Yawen Huang, Yefeng Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25171](https://doi.org/10.1609/aaai.v37i1.25171)

**Abstract**:

Vision Transformers have recently shown impressive performances on medical image segmentation. Despite their strong capability of modeling long-range dependencies, the current methods still give rise to two main concerns in a class-level perspective: (1) intra-class problem: the existing methods lacked in extracting class-specific correspondences of different pixels, which may lead to poor object coverage and/or boundary prediction; (2) inter-class problem: the existing methods failed to model explicit category-dependencies among various objects, which may result in inaccurate localization. In light of these two issues, we propose a novel transformer, called ClassFormer, powered by two appealing transformers, i.e., intra-class dynamic transformer and inter-class interactive transformer, to address the challenge of fully exploration on compactness and discrepancy. Technically, the intra-class dynamic transformer is first designed to decouple representations of different categories with an adaptive selection mechanism for compact learning, which optimally highlights the informative features to reflect the salient keys/values from multiple scales. We further introduce the inter-class interactive transformer to capture the category dependency among different objects, and model class tokens as the representative class centers to guide a global semantic reasoning. As a consequence, the feature consistency is ensured with the expense of intra-class penalization, while inter-class constraint strengthens the feature discriminability between different categories. Extensive empirical evidence shows that ClassFormer can be easily plugged into any architecture, and yields improvements over the state-of-the-art methods in three public benchmarks.

----

## [102] NLIP: Noise-Robust Language-Image Pre-training

**Authors**: *Runhui Huang, Yanxin Long, Jianhua Han, Hang Xu, Xiwen Liang, Chunjing Xu, Xiaodan Liang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25172](https://doi.org/10.1609/aaai.v37i1.25172)

**Abstract**:

Large-scale cross-modal pre-training paradigms have recently shown ubiquitous success on a wide range of downstream tasks, e.g., zero-shot classification, retrieval and image captioning. However, their successes highly rely on the scale and quality of web-crawled data that naturally contain much incomplete and noisy information (e.g., wrong or irrelevant contents). Existing works either design manual rules to clean data or generate pseudo-targets as auxiliary signals for reducing noise impact, which do not explicitly tackle both the incorrect and incomplete challenges at the same time. In this paper, to automatically mitigate the impact of noise by solely mining over existing data, we propose a principled Noise-robust Language-Image Pre-training framework (NLIP) to stabilize pre-training via two schemes: noise-harmonization and noise-completion. First, in noise-harmonization scheme, NLIP estimates the noise probability of each pair according to the memorization effect of cross-modal transformers, then adopts noise-adaptive regularization to harmonize the cross-modal alignments with varying degrees. Second, in noise-completion scheme, to enrich the missing object information of text, NLIP injects a concept-conditioned cross-modal decoder to obtain semantic-consistent synthetic captions to complete noisy ones, which uses the retrieved visual concepts (i.e., objects’ names) for the corresponding image to guide captioning generation. By collaboratively optimizing noise-harmonization and noise-completion schemes, our NLIP can alleviate the common noise effects during image-text pre-training in a more efficient way. Extensive experiments show the significant performance improvements of our NLIP using only 26M data over existing pre-trained models (e.g., CLIP, FILIP and BLIP) on 12 zero-shot classification datasets (e.g., +8.6% over CLIP on average accuracy), MSCOCO image captioning (e.g., +1.9 over BLIP trained with 129M data on CIDEr) and zero-shot image-text retrieval tasks.

----

## [103] Symmetry-Aware Transformer-Based Mirror Detection

**Authors**: *Tianyu Huang, Bowen Dong, Jiaying Lin, Xiaohui Liu, Rynson W. H. Lau, Wangmeng Zuo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25173](https://doi.org/10.1609/aaai.v37i1.25173)

**Abstract**:

Mirror detection aims to identify the mirror regions in the given input image. Existing works mainly focus on integrating the semantic features and structural features to mine specific relations between mirror and non-mirror regions, or introducing mirror properties like depth or chirality to help analyze the existence of mirrors. In this work, we observe that a real object typically forms a loose symmetry relationship with its corresponding reflection in the mirror, which is beneficial in distinguishing mirrors from real objects. Based on this observation, we propose a dual-path Symmetry-Aware Transformer-based mirror detection Network (SATNet), which includes two novel modules: Symmetry-Aware Attention Module (SAAM) and Contrast and Fusion Decoder Module (CFDM). Specifically, we first adopt a transformer backbone to model global information aggregation in images, extracting multi-scale features in two paths. We then feed the high-level dual-path features to SAAMs to capture the symmetry relations. Finally, we fuse the dual-path features and refine our prediction maps progressively with CFDMs to obtain the final mirror mask. Experimental results show that SATNet outperforms both RGB and RGB-D mirror detection methods on all available mirror detection datasets.

----

## [104] AudioEar: Single-View Ear Reconstruction for Personalized Spatial Audio

**Authors**: *Xiaoyang Huang, Yanjun Wang, Yang Liu, Bingbing Ni, Wenjun Zhang, Jinxian Liu, Teng Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25174](https://doi.org/10.1609/aaai.v37i1.25174)

**Abstract**:

Spatial audio, which focuses on immersive 3D sound rendering, is widely applied in the acoustic industry. One of the key problems of current spatial audio rendering methods is the lack of personalization based on different anatomies of individuals, which is essential to produce accurate sound source positions. In this work, we address this problem from an interdisciplinary perspective. The rendering of spatial audio is strongly correlated with the 3D shape of human bodies, particularly ears. To this end, we propose to achieve personalized spatial audio by reconstructing 3D human ears with single-view images. First, to benchmark the ear reconstruction task, we introduce AudioEar3D, a high-quality 3D ear dataset consisting of 112 point cloud ear scans with RGB images. To self-supervisedly train a reconstruction model, we further collect a 2D ear dataset composed of 2,000 images, each one with manual annotation of occlusion and 55 landmarks, named AudioEar2D. To our knowledge, both datasets have the largest scale and best quality of their kinds for public use. Further, we propose AudioEarM, a reconstruction method guided by a depth estimation network that is trained on synthetic data, with two loss functions tailored for ear data. Lastly, to fill the gap between the vision and acoustics community, we develop a pipeline to integrate the reconstructed ear mesh with an off-the-shelf 3D human body and simulate a personalized Head-Related Transfer Function (HRTF), which is the core of spatial audio rendering. Code and data are publicly available in https://github.com/seanywang0408/AudioEar.

----

## [105] Boosting Point Clouds Rendering via Radiance Mapping

**Authors**: *Xiaoyang Huang, Yi Zhang, Bingbing Ni, Teng Li, Kai Chen, Wenjun Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25175](https://doi.org/10.1609/aaai.v37i1.25175)

**Abstract**:

Recent years we have witnessed rapid development in NeRF-based image rendering due to its high quality. However, point clouds rendering is somehow less explored. Compared to NeRF-based rendering which suffers from dense spatial sampling, point clouds rendering is naturally less computation intensive, which enables its deployment in mobile computing device. In this work, we focus on boosting the image quality of point clouds rendering with a compact model design. We first analyze the adaption of the volume rendering formulation on point clouds. Based on the analysis, we simplify the NeRF representation to a spatial mapping function which only requires single evaluation per pixel. Further, motivated by ray marching, we rectify the the noisy raw point clouds to the estimated intersection between rays and surfaces as queried coordinates, which could avoid spatial frequency collapse and neighbor point disturbance. Composed of rasterization, spatial mapping and the refinement stages, our method achieves the state-of-the-art performance on point clouds rendering, outperforming prior works by notable margins, with a smaller model size. We obtain a PSNR of 31.74 on NeRF-Synthetic, 25.88 on ScanNet and 30.81 on DTU. Code and data are publicly available in https://github.com/seanywang0408/RadianceMapping.

----

## [106] FreeEnricher: Enriching Face Landmarks without Additional Cost

**Authors**: *Yangyu Huang, Xi Chen, Jongyoo Kim, Hao Yang, Chong Li, Jiaolong Yang, Dong Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25176](https://doi.org/10.1609/aaai.v37i1.25176)

**Abstract**:

Recent years have witnessed significant growth of face alignment. Though dense facial landmark is highly demanded in various scenarios, e.g., cosmetic medicine and facial beautification, most works only consider sparse face alignment. To address this problem, we present a framework that can enrich landmark density by existing sparse landmark datasets, e.g., 300W with 68 points and WFLW with 98 points. Firstly, we observe that the local patches along each semantic contour are highly similar in appearance. Then, we propose a weakly-supervised idea of learning the refinement ability on original sparse landmarks and adapting this ability to enriched dense landmarks. Meanwhile, several operators are devised and organized together to implement the idea. Finally, the trained model is applied as a plug-and-play module to the existing face alignment networks. To evaluate our method, we manually label the dense landmarks on 300W testset. Our method yields state-of-the-art accuracy not only in newly-constructed dense 300W testset but also in the original sparse 300W and WFLW testsets without additional cost.

----

## [107] PATRON: Perspective-Aware Multitask Model for Referring Expression Grounding Using Embodied Multimodal Cues

**Authors**: *Md Mofijul Islam, Alexi Gladstone, Tariq Iqbal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25177](https://doi.org/10.1609/aaai.v37i1.25177)

**Abstract**:

Humans naturally use referring expressions with verbal utterances and nonverbal gestures to refer to objects and events. As these referring expressions can be interpreted differently from the speaker's or the observer's perspective, people effectively decide on the perspective in comprehending the expressions. However, existing models do not explicitly learn perspective grounding, which often causes the models to perform poorly in understanding embodied referring expressions. To make it exacerbate, these models are often trained on datasets collected in non-embodied settings without nonverbal gestures and curated from an exocentric perspective. To address these issues, in this paper, we present a perspective-aware multitask learning model, called PATRON, for relation and object grounding tasks in embodied settings by utilizing verbal utterances and nonverbal cues. In PATRON, we have developed a guided fusion approach, where a perspective grounding task guides the relation and object grounding task. Through this approach, PATRON learns disentangled task-specific and task-guidance representations, where task-guidance representations guide the extraction of salient multimodal features to ground the relation and object accurately. Furthermore, we have curated a synthetic dataset of embodied referring expressions with multimodal cues, called CAESAR-PRO. The experimental results suggest that PATRON outperforms the evaluated state-of-the-art visual-language models. Additionally, the results indicate that learning to ground perspective helps machine learning models to improve the performance of the relation and object grounding task. Furthermore, the insights from the extensive experimental results and the proposed dataset will enable researchers to evaluate visual-language models' effectiveness in understanding referring expressions in other embodied settings.

----

## [108] Unifying Vision-Language Representation Space with Single-Tower Transformer

**Authors**: *Jiho Jang, Chaerin Kong, Donghyeon Jeon, Seonhoon Kim, Nojun Kwak*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25178](https://doi.org/10.1609/aaai.v37i1.25178)

**Abstract**:

Contrastive learning is a form of distance learning that aims to learn invariant features from two related representations. In this work, we explore the hypothesis that an image and caption can be regarded as two different views of the underlying mutual information, and train a model to learn a unified vision-language representation space that encodes both modalities at once in a modality-agnostic manner. We first identify difficulties in learning a one-tower model for vision-language pretraining (VLP), and propose One Representation (OneR) as a simple yet effective framework for our goal. We discover intriguing properties that distinguish OneR from the previous works that have modality-specific representation spaces such as zero-shot localization, text-guided visual reasoning and multi-modal retrieval, and present analyses to provide insights into this new form of multi-modal representation learning. Thorough evaluations demonstrate the potential of a unified modality-agnostic VLP framework.

----

## [109] Delving Deep into Pixel Alignment Feature for Accurate Multi-View Human Mesh Recovery

**Authors**: *Kai Jia, Hongwen Zhang, Liang An, Yebin Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25179](https://doi.org/10.1609/aaai.v37i1.25179)

**Abstract**:

Regression-based methods have shown high efficiency and effectiveness for multi-view human mesh recovery. The key components of a typical regressor lie in the feature extraction of input views and the fusion of multi-view features. In this paper, we present Pixel-aligned Feedback Fusion (PaFF) for accurate yet efficient human mesh recovery from multi-view images. PaFF is an iterative regression framework that performs feature extraction and fusion alternately. At each iteration, PaFF extracts pixel-aligned feedback features from each input view according to the reprojection of the current estimation and fuses them together with respect to each vertex of the downsampled mesh. In this way, our regressor can not only perceive the misalignment status of each view from the feedback features but also correct the mesh parameters more effectively based on the feature fusion on mesh vertices. Additionally, our regressor disentangles the global orientation and translation of the body mesh from the estimation of mesh parameters such that the camera parameters of input views can be better utilized in the regression process. The efficacy of our method is validated in the Human3.6M dataset via comprehensive ablation experiments, where PaFF achieves 33.02 MPJPE and brings significant improvements over the previous best solutions by more than 29%. The project page with code and video results can be found at https://kairobo.github.io/PaFF/.

----

## [110] Semi-attention Partition for Occluded Person Re-identification

**Authors**: *Mengxi Jia, Yifan Sun, Yunpeng Zhai, Xinhua Cheng, Yi Yang, Ying Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25180](https://doi.org/10.1609/aaai.v37i1.25180)

**Abstract**:

This paper proposes a Semi-Attention Partition (SAP) method to learn well-aligned part features for occluded person re-identification (re-ID). Currently, the mainstream methods employ either external semantic partition or attention-based partition, and the latter manner is usually better than the former one. Under this background, this paper explores a potential that the weak semantic partition can be a good teacher for the strong attention-based partition. In other words, the attention-based student can substantially surpass its noisy semantic-based teacher, contradicting the common sense that the student usually achieves inferior (or comparable) accuracy. A key to this effect is: the proposed SAP encourages the attention-based partition of the (transformer) student to be partially consistent with the semantic-based teacher partition through knowledge distillation, yielding the so-called semi-attention. Such partial consistency allows the student to have both consistency and reasonable conflict with the noisy teacher. More specifically, on the one hand, the attention is guided by the semantic partition from the teacher. On the other hand, the attention mechanism itself still has some degree of freedom to comply with the inherent similarity between different patches, thus gaining resistance against noisy supervision. Moreover, we integrate a battery of well-engineered designs into SAP to reinforce their cooperation (e.g., multiple forms of teacher-student consistency), as well as to promote reasonable conflict (e.g., mutual absorbing partition refinement and a supervision signal dropout strategy). Experimental results confirm that the transformer student achieves substantial improvement after this semi-attention learning scheme, and produces new state-of-the-art accuracy on several standard re-ID benchmarks.

----

## [111] Fast Online Hashing with Multi-Label Projection

**Authors**: *Wenzhe Jia, Yuan Cao, Junwei Liu, Jie Gui*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25181](https://doi.org/10.1609/aaai.v37i1.25181)

**Abstract**:

Hashing has been widely researched to solve the large-scale approximate nearest neighbor search problem owing to its time and storage superiority. In recent years, a number of online hashing methods have emerged, which can update the hash functions to adapt to the new stream data and realize dynamic  retrieval. However, existing online hashing methods are required to update the whole database with the latest hash functions when a query arrives, which leads to low retrieval efficiency with the continuous increase of the stream data. On the other hand, these methods ignore the supervision relationship among the examples, especially in the multi-label case. In this paper, we propose a novel Fast Online Hashing (FOH) method which only updates the binary codes of a small part of the database. To be specific, we first build a query pool in which the nearest neighbors of each central point are recorded. When a new query arrives, only the binary codes of the corresponding potential neighbors are updated. In addition, we create a similarity matrix which takes the multi-label supervision information into account and bring in the multi-label projection loss to further preserve the similarity among the multi-label data. The experimental results on two common benchmarks show that the proposed FOH can achieve dramatic superiority on query time up to 6.28 seconds less than state-of-the-art baselines with competitive retrieval accuracy.

----

## [112] Fourier-Net: Fast Image Registration with Band-Limited Deformation

**Authors**: *Xi Jia, Joseph Bartlett, Wei Chen, Siyang Song, Tianyang Zhang, Xinxing Cheng, Wenqi Lu, Zhaowen Qiu, Jinming Duan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25182](https://doi.org/10.1609/aaai.v37i1.25182)

**Abstract**:

Unsupervised image registration commonly adopts U-Net style networks to predict dense displacement fields in the full-resolution spatial domain. For high-resolution volumetric image data, this process is however resource-intensive and time-consuming. To tackle this problem, we propose the Fourier-Net, replacing the expansive path in a U-Net style network with a parameter-free model-driven decoder. Specifically, instead of our Fourier-Net learning to output a full-resolution displacement field in the spatial domain, we learn its low-dimensional representation in a band-limited Fourier domain. This representation is then decoded by our devised model-driven decoder (consisting of a zero padding layer and an inverse discrete Fourier transform layer) to the dense, full-resolution displacement field in the spatial domain. These changes allow our unsupervised Fourier-Net to contain fewer parameters and computational operations, resulting in faster inference speeds. Fourier-Net is then evaluated on two public 3D brain datasets against various state-of-the-art approaches. For example, when compared to a recent transformer-based method, named TransMorph, our Fourier-Net, which only uses 2.2% of its parameters and 6.66% of the multiply-add operations, achieves a 0.5% higher Dice score and an 11.48 times faster inference speed. Code is available at https://github.com/xi-jia/Fourier-Net.

----

## [113] Semi-supervised Deep Large-Baseline Homography Estimation with Progressive Equivalence Constraint

**Authors**: *Hai Jiang, Haipeng Li, Yuhang Lu, Songchen Han, Shuaicheng Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25183](https://doi.org/10.1609/aaai.v37i1.25183)

**Abstract**:

Homography estimation is erroneous in the case of large-baseline due to the low image overlay and limited receptive field. To address it, we propose a progressive estimation strategy by converting large-baseline homography into multiple intermediate ones, cumulatively multiplying these intermediate items can reconstruct the initial homography. Meanwhile, a semi-supervised homography identity loss, which consists of two components: a supervised objective and an unsupervised objective, is introduced. The first supervised loss is acting to optimize intermediate homographies, while the second unsupervised one helps to estimate a large-baseline homography without photometric losses. To validate our method, we propose a large-scale dataset that covers regular and challenging scenes. Experiments show that our method achieves state-of-the-art performance in large-baseline scenes while keeping competitive performance in small-baseline scenes. Code and dataset are available at https://github.com/megvii-research/LBHomo.

----

## [114] Multi-Modality Deep Network for Extreme Learned Image Compression

**Authors**: *Xuhao Jiang, Weimin Tan, Tian Tan, Bo Yan, Liquan Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25184](https://doi.org/10.1609/aaai.v37i1.25184)

**Abstract**:

Image-based single-modality compression learning approaches have demonstrated exceptionally powerful encoding and decoding capabilities in the past few years , but suffer from blur and severe semantics loss at extremely low bitrates. To address this issue, we propose a multimodal machine learning method for text-guided image compression, in which the semantic information of text is used as prior information to guide image compression for better compression performance. We fully study the role of text description in different components of the codec, and demonstrate its effectiveness. In addition, we adopt the image-text attention module and image-request complement module to better fuse image and text features, and propose an improved multimodal semantic-consistent loss to produce semantically complete reconstructions. Extensive experiments, including a user study, prove that our method can obtain visually pleasing results at extremely low bitrates, and achieves a comparable or even better performance than state-of-the-art methods, even though these methods are at 2x to 4x bitrates of ours.

----

## [115] PolarFormer: Multi-Camera 3D Object Detection with Polar Transformer

**Authors**: *Yanqin Jiang, Li Zhang, Zhenwei Miao, Xiatian Zhu, Jin Gao, Weiming Hu, Yu-Gang Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25185](https://doi.org/10.1609/aaai.v37i1.25185)

**Abstract**:

3D object detection in autonomous driving aims to reason “what” and “where” the objects of interest present in a 3D world. Following the conventional wisdom of previous 2D object detection, existing methods often adopt the canonical Cartesian coordinate system with perpendicular axis. However, we conjugate that this does not fit the nature of the ego car’s perspective, as each onboard camera perceives the world in shape of wedge intrinsic to the imaging geometry with radical (non perpendicular) axis. Hence, in this paper we advocate the exploitation of the Polar coordinate system and propose a new Polar Transformer (PolarFormer) for more accurate 3D object detection in the bird’s-eye-view (BEV) taking as input only multi-camera 2D images. Specifically, we design a cross-attention based Polar detection head without restriction to the shape of input structure to deal with irregular Polar grids. For tackling the unconstrained object scale variations along Polar’s distance dimension, we further introduce a multi-scale Polar representation learning strategy. As a result, our model can make best use of the Polar representation rasterized via attending to the corresponding image observation in a sequence-to-sequence fashion subject to the geometric constraints. Thorough experiments on the nuScenes dataset demonstrate that our PolarFormer outperforms significantly state-of-the-art 3D object detection alternatives.

----

## [116] 3D-TOGO: Towards Text-Guided Cross-Category 3D Object Generation

**Authors**: *Zutao Jiang, Guansong Lu, Xiaodan Liang, Jihua Zhu, Wei Zhang, Xiaojun Chang, Hang Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25186](https://doi.org/10.1609/aaai.v37i1.25186)

**Abstract**:

Abstract
This article has been updated and an error has been fixed in published paper. An Erratum to this article was published on 6 September 2023. 
Text-guided 3D object generation aims to generate 3D objects described by user-defined captions, which paves a flexible way to visualize what we imagined. Although some works have been devoted to solving this challenging task, these works either utilize some explicit 3D representations (e.g., mesh), which lack texture and require post-processing for rendering photo-realistic views; or require individual time-consuming optimization for every single case. Here, we make the first attempt to achieve generic text-guided cross-category 3D object generation via a new 3D-TOGO model, which integrates a text-to-views generation module and a views-to-3D generation module. The text-to-views generation module is designed to generate different views of the target 3D object given an input caption. prior-guidance, caption-guidance and view contrastive learning are proposed for achieving better view-consistency and caption similarity. Meanwhile, a pixelNeRF model is adopted for the views-to-3D generation module to obtain the implicit 3D neural representation from the previously-generated views. Our 3D-TOGO model generates 3D objects in the form of the neural radiance field with good texture and requires no time-cost optimization for every single caption. Besides, 3D-TOGO can control the category, color and shape of generated 3D objects with the input caption. Extensive experiments on the largest 3D object dataset (i.e., ABO) are conducted to verify that 3D-TOGO can better generate high-quality 3D objects according to the input captions across 98 different categories, in terms of PSNR, SSIM, LPIPS and CLIP-score, compared with text-NeRF and Dreamfields.

----

## [117] FacT: Factor-Tuning for Lightweight Adaptation on Vision Transformer

**Authors**: *Shibo Jie, Zhi-Hong Deng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25187](https://doi.org/10.1609/aaai.v37i1.25187)

**Abstract**:

Recent work has explored the potential to adapt a pre-trained vision transformer (ViT) by updating only a few parameters so as to improve storage efficiency, called parameter-efficient transfer learning (PETL). Current PETL methods have shown that by tuning only 0.5% of the parameters, ViT can be adapted to downstream tasks with even better performance than full fine-tuning. In this paper, we aim to further promote the efficiency of PETL to meet the extreme storage constraint in real-world applications. To this end, we propose a tensorization-decomposition framework to store the weight increments, in which the weights of each ViT are tensorized into a single 3D tensor, and their increments are then decomposed into lightweight factors. In the fine-tuning process, only the factors need to be updated and stored, termed Factor-Tuning (FacT). On VTAB-1K benchmark, our method performs on par with NOAH, the state-of-the-art PETL method, while being 5x more parameter-efficient. We also present a tiny version that only uses 8K (0.01% of ViT's parameters) trainable parameters but outperforms full fine-tuning and many other PETL methods such as VPT and BitFit. In few-shot settings, FacT also beats all PETL baselines using the fewest parameters, demonstrating its strong capability in the low-data regime.

----

## [118] Estimating Reflectance Layer from a Single Image: Integrating Reflectance Guidance and Shadow/Specular Aware Learning

**Authors**: *Yeying Jin, Ruoteng Li, Wenhan Yang, Robby T. Tan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25188](https://doi.org/10.1609/aaai.v37i1.25188)

**Abstract**:

Estimating the reflectance layer from a single image is a challenging task. It becomes more challenging when the input image contains shadows or specular highlights, which often render an inaccurate estimate of the reflectance layer. Therefore, we propose a two-stage learning method, including reflectance guidance and a Shadow/Specular-Aware (S-Aware) network to tackle the problem. In the first stage, an initial reflectance layer free from shadows and specularities is obtained with the constraint of novel losses that are guided by prior-based shadow-free and specular-free images. To further enforce the reflectance layer to be independent of shadows and specularities in the second-stage refinement, we introduce an S-Aware network that distinguishes the reflectance image from the input image. Our network employs a classifier to categorize shadow/shadow-free, specular/specular-free classes, enabling the activation features to function as attention maps that focus on shadow/specular regions. Our quantitative and qualitative evaluations show that our method outperforms the state-of-the-art methods in the reflectance layer estimation that is free from shadows and specularities.

----

## [119] Weakly-Guided Self-Supervised Pretraining for Temporal Activity Detection

**Authors**: *Kumara Kahatapitiya, Zhou Ren, Haoxiang Li, Zhenyu Wu, Michael S. Ryoo, Gang Hua*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25189](https://doi.org/10.1609/aaai.v37i1.25189)

**Abstract**:

Temporal Activity Detection aims to predict activity classes per frame, in contrast to video-level predictions in Activity Classification (i.e., Activity Recognition). Due to the expensive frame-level annotations required for detection, the scale of detection datasets is limited. Thus, commonly, previous work on temporal activity detection resorts to fine-tuning a classification model pretrained on large-scale classification datasets (e.g., Kinetics-400). However, such pretrained models are not ideal for downstream detection, due to the disparity between the pretraining and the downstream fine-tuning tasks. In this work, we propose a novel weakly-guided self-supervised pretraining method for detection. We leverage weak labels (classification) to introduce a self-supervised pretext task (detection) by generating frame-level pseudo labels, multi-action frames, and action segments. Simply put, we design a detection task similar to downstream, on large-scale classification data, without extra annotations. We show that the models pretrained with the proposed weakly-guided self-supervised detection task outperform prior work on multiple challenging activity detection benchmarks, including Charades and MultiTHUMOS. Our extensive ablations further provide insights on when and how to use the proposed models for activity detection. Code is available at github.com/kkahatapitiya/SSDet.

----

## [120] Correlation Loss: Enforcing Correlation between Classification and Localization

**Authors**: *Fehmi Kahraman, Kemal Oksuz, Sinan Kalkan, Emre Akbas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25190](https://doi.org/10.1609/aaai.v37i1.25190)

**Abstract**:

Object detectors are conventionally trained by a weighted sum of classification and localization losses. Recent studies (e.g., predicting IoU with an auxiliary head, Generalized Focal Loss, Rank & Sort Loss) have shown that forcing these two loss terms to interact with each other in non-conventional ways creates a useful inductive bias and improves performance. Inspired by these works, we focus on the correlation between classification and localization and make two main contributions: (i) We provide an analysis about the effects of correlation between classification and localization tasks in object detectors.  We identify why correlation affects the performance of various NMS-based and NMS-free detectors, and we devise measures to evaluate the effect of correlation and use them to analyze common detectors. (ii) Motivated by our observations, e.g., that NMS-free detectors can also benefit from correlation, we propose Correlation Loss, a novel plug-in loss function that improves the performance of various object detectors by directly optimizing correlation coefficients: E.g., Correlation Loss on Sparse R-CNN, an NMS-free method, yields 1.6 AP gain on COCO and 1.8 AP gain on Cityscapes dataset. Our best model on Sparse R-CNN reaches 51.0 AP without test-time augmentation on COCO test-dev, reaching state-of-the-art.  Code is available at: https://github.com/fehmikahraman/CorrLoss.

----

## [121] GuidedMixup: An Efficient Mixup Strategy Guided by Saliency Maps

**Authors**: *Minsoo Kang, Suhyun Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25191](https://doi.org/10.1609/aaai.v37i1.25191)

**Abstract**:

Data augmentation is now an essential part of the image training process, as it effectively prevents overfitting and makes the model more robust against noisy datasets. Recent mixing augmentation strategies have advanced to generate the mixup mask that can enrich the saliency information, which is a supervisory signal. However, these methods incur a significant computational burden to optimize the mixup mask. From this motivation, we propose a novel saliency-aware mixup method, GuidedMixup, which aims to retain the salient regions in mixup images with low computational overhead. We develop an efficient pairing algorithm that pursues to minimize the conflict of salient regions of paired images and achieve rich saliency in mixup images. Moreover, GuidedMixup controls the mixup ratio for each pixel to better preserve the salient region by interpolating two paired images smoothly. The experiments on several datasets demonstrate that GuidedMixup provides a good trade-off between augmentation overhead and generalization performance on classification datasets. In addition, our method shows good performance in experiments with corrupted or reduced datasets.

----

## [122] 3D Human Pose Lifting with Grid Convolution

**Authors**: *Yangyuxuan Kang, Yuyang Liu, Anbang Yao, Shandong Wang, Enhua Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25192](https://doi.org/10.1609/aaai.v37i1.25192)

**Abstract**:

Existing lifting networks for regressing 3D human poses from 2D single-view poses are typically constructed with linear layers based on graph-structured representation learning. In sharp contrast to them, this paper presents Grid Convolution (GridConv), mimicking the wisdom of regular convolution operations in image space. GridConv is based on a novel Semantic Grid Transformation (SGT) which leverages a binary assignment matrix to map the irregular graph-structured human pose onto a regular weave-like grid pose representation joint by joint, enabling layer-wise feature learning with GridConv operations. We provide two ways to implement SGT, including handcrafted and learnable designs. Surprisingly, both designs turn out to achieve promising results and the learnable one is better, demonstrating the great potential of this new lifting representation learning formulation. To improve the ability of GridConv to encode contextual cues, we introduce an attention module over the convolutional kernel, making grid convolution operations input-dependent, spatial-aware and grid-specific. We show that our fully convolutional grid lifting network outperforms state-of-the-art methods with noticeable margins under (1) conventional evaluation on Human3.6M and (2) cross-evaluation on MPI-INF-3DHP. Code is available at https://github.com/OSVAI/GridConv.

----

## [123] Bidirectional Domain Mixup for Domain Adaptive Semantic Segmentation

**Authors**: *Daehan Kim, Minseok Seo, Kwanyong Park, Inkyu Shin, Sanghyun Woo, In So Kweon, Dong-Geol Choi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25193](https://doi.org/10.1609/aaai.v37i1.25193)

**Abstract**:

Mixup provides interpolated training samples and allows the model to obtain smoother decision boundaries for better generalization. The idea can be naturally applied to the domain adaptation task, where we can mix the source and target samples to obtain domain-mixed samples for better adaptation. However, the extension of the idea from classification to segmentation (i.e., structured output) is nontrivial. This paper systematically studies the impact of mixup under the domain adaptive semantic segmentation task and presents a simple yet effective mixup strategy called Bidirectional Domain Mixup (BDM). In specific, we achieve domain mixup in two-step: cut and paste. Given the warm-up model trained from any adaptation techniques, we forward the source and target samples and perform a simple threshold-based cut out of the unconfident regions (cut). After then, we fill-in the dropped regions with the other domain region patches (paste). In doing so, we jointly consider class distribution, spatial structure, and pseudo label confidence. Based on our analysis, we found that BDM leaves domain transferable regions by cutting, balances the dataset-level class distribution while preserving natural scene context by pasting. We coupled our proposal with various state-of-the-art adaptation models and observe significant improvement consistently. We also provide extensive ablation experiments to empirically verify our main components of the framework. Visit our project page with the code at https://sites.google.com/view/bidirectional-domain-mixup

----

## [124] Frequency Selective Augmentation for Video Representation Learning

**Authors**: *Jinhyung Kim, Taeoh Kim, Minho Shim, Dongyoon Han, Dongyoon Wee, Junmo Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25194](https://doi.org/10.1609/aaai.v37i1.25194)

**Abstract**:

Recent self-supervised video representation learning methods focus on maximizing the similarity between multiple augmented views from the same video and largely rely on the quality of generated views. However, most existing methods lack a mechanism to prevent representation learning from bias towards static information in the video. In this paper, we propose frequency augmentation (FreqAug), a spatio-temporal data augmentation method in the frequency domain
for video representation learning. FreqAug stochastically removes specific frequency components from the video so that learned representation captures essential features more from the remaining information for various downstream tasks. Specifically, FreqAug pushes the model to focus more on dynamic features rather than static features in the video via dropping spatial or temporal low-frequency components. To verify the generality of the proposed method, we experiment with FreqAug on multiple self-supervised learning frameworks along with standard augmentations. Transferring the improved representation to five video action recognition and two temporal action localization downstream tasks shows consistent improvements over baselines.

----

## [125] Pose-Guided 3D Human Generation in Indoor Scene

**Authors**: *Minseok Kim, Changwoo Kang, Jeongin Park, Kyungdon Joo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25195](https://doi.org/10.1609/aaai.v37i1.25195)

**Abstract**:

In this work, we address the problem of scene-aware 3D human avatar generation based on human-scene interactions. In particular, we pay attention to the fact that physical contact between a 3D human and a scene (i.e., physical human-scene interactions) requires a geometrical alignment to generate natural 3D human avatar. Motivated by this fact, we present a new 3D human generation framework that considers geometric alignment on potential contact areas between 3D human avatars and their surroundings. In addition, we introduce a compact yet effective human pose classifier that classifies the human pose and provides potential contact areas of the 3D human avatar. It allows us to adaptively use geometric alignment loss according to the classified human pose. Compared to state-of-the-art method, our method can generate physically and semantically plausible 3D humans that interact naturally with 3D scenes without additional post-processing. In our evaluations, we achieve the improvements with more plausible interactions and more variety of poses than prior research in qualitative and quantitative analysis. Project page: https://bupyeonghealer.github.io/phin/.

----

## [126] Semantic-Aware Superpixel for Weakly Supervised Semantic Segmentation

**Authors**: *Sangtae Kim, Daeyoung Park, Byonghyo Shim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25196](https://doi.org/10.1609/aaai.v37i1.25196)

**Abstract**:

Weakly-supervised semantic segmentation aims to train a semantic segmentation network using weak labels. Among weak labels, image-level label has been the most popular choice due to its simplicity. However, since image-level labels lack accurate object region information, additional modules such as saliency detector have been exploited in weakly supervised semantic segmentation, which requires pixel-level label for training. In this paper, we explore a self-supervised vision transformer to mitigate the heavy efforts on generation of pixel-level annotations. By exploiting the features obtained from self-supervised vision transformer, our superpixel discovery method finds out the semantic-aware superpixels based on the feature similarity in an unsupervised manner. Once we obtain the superpixels, we train the semantic segmentation network using superpixel-guided seeded region growing method. Despite its simplicity, our approach achieves the competitive result with the state-of-the-arts on PASCAL VOC 2012 and MS-COCO 2014 semantic segmentation datasets for weakly supervised semantic segmentation. Our code is available at https://github.com/st17kim/semantic-aware-superpixel.

----

## [127] Multispectral Invisible Coating: Laminated Visible-Thermal Physical Attack against Multispectral Object Detectors Using Transparent Low-E Films

**Authors**: *Taeheon Kim, Youngjoon Yu, Yong Man Ro*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25197](https://doi.org/10.1609/aaai.v37i1.25197)

**Abstract**:

Multispectral object detection plays a vital role in safety-critical vision systems that require an around-the-clock operation and encounter dynamic real-world situations(e.g., self-driving cars and autonomous surveillance systems). Despite its crucial competence in safety-related applications, its security against physical attacks is severely understudied. We investigate the vulnerability of multispectral detectors against physical attacks by proposing a new physical method: Multispectral Invisible Coating. Utilizing transparent Low-e films, we realize a laminated visible-thermal physical attack by attaching Low-e films over a visible attack printing. Moreover, we apply our physical method to manufacture a Multispectral Invisible Suit that hides persons from the multiple view angles of Multispectral detectors. To simulate our attack under various surveillance scenes, we constructed a large-scale multispectral pedestrian dataset which we will release in public. Extensive experiments show that our proposed method effectively attacks the state-of-the-art multispectral detector both in the digital space and the physical world.

----

## [128] CRAFT: Camera-Radar 3D Object Detection with Spatio-Contextual Fusion Transformer

**Authors**: *Youngseok Kim, Sanmin Kim, Jun Won Choi, Dongsuk Kum*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25198](https://doi.org/10.1609/aaai.v37i1.25198)

**Abstract**:

Camera and radar sensors have significant advantages in cost, reliability, and maintenance compared to LiDAR. Existing fusion methods often fuse the outputs of single modalities at the result-level, called the late fusion strategy. This can benefit from using off-the-shelf single sensor detection algorithms, but late fusion cannot fully exploit the complementary properties of sensors, thus having limited performance despite the huge potential of camera-radar fusion. Here we propose a novel proposal-level early fusion approach that effectively exploits both spatial and contextual properties of camera and radar for 3D object detection. Our fusion framework first associates image proposal with radar points in the polar coordinate system to efficiently handle the discrepancy between the coordinate system and spatial properties. Using this as a first stage, following consecutive cross-attention based feature fusion layers adaptively exchange spatio-contextual information between camera and radar, leading to a robust and attentive fusion. Our camera-radar fusion approach achieves the state-of-the-art 41.1% mAP and 52.3% NDS on the nuScenes test set, which is 8.7 and 10.8 points higher than the camera-only baseline, as well as yielding competitive performance on the LiDAR method.

----

## [129] Simple and Effective Synthesis of Indoor 3D Scenes

**Authors**: *Jing Yu Koh, Harsh Agrawal, Dhruv Batra, Richard Tucker, Austin Waters, Honglak Lee, Yinfei Yang, Jason Baldridge, Peter Anderson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25199](https://doi.org/10.1609/aaai.v37i1.25199)

**Abstract**:

We study the problem of synthesizing immersive 3D indoor scenes from one or a few images. Our aim is to generate high-resolution images and videos from novel viewpoints, including viewpoints that extrapolate far beyond the input images while maintaining 3D consistency. Existing approaches are highly complex, with many separately trained stages and components. We propose a simple alternative: an image-to-image GAN that maps directly from reprojections of incomplete point clouds to full high-resolution RGB-D images. On the Matterport3D and RealEstate10K datasets, our approach significantly outperforms prior work when evaluated by humans, as well as on FID scores. Further, we show that our model is useful for generative data augmentation. A vision-and-language navigation (VLN) agent trained with trajectories spatially-perturbed by our model improves success rate by up to 1.5% over a state of the art baseline on the mature R2R benchmark. Our code will be made available to facilitate generative data augmentation and applications to downstream robotics and embodied AI tasks.

----

## [130] MGTANet: Encoding Sequential LiDAR Points Using Long Short-Term Motion-Guided Temporal Attention for 3D Object Detection

**Authors**: *Junho Koh, Junhyung Lee, Youngwoo Lee, Jaekyum Kim, Jun Won Choi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25200](https://doi.org/10.1609/aaai.v37i1.25200)

**Abstract**:

Most scanning LiDAR sensors generate a sequence of point clouds in real-time. While conventional 3D object detectors use a set of unordered LiDAR points acquired over a fixed time interval, recent studies have revealed that substantial performance improvement can be achieved by exploiting the spatio-temporal context present in a sequence of LiDAR point sets. In this paper, we propose a novel 3D object detection architecture, which can encode LiDAR point cloud sequences acquired by multiple successive scans. The encoding process of the point cloud sequence is performed on two different time scales. We first design a short-term motion-aware voxel encoding that captures the short-term temporal changes of point clouds driven by the motion of objects in each voxel. We also propose long-term motion-guided bird’s eye view (BEV) feature enhancement that adaptively aligns and aggregates the BEV feature maps obtained by the short-term voxel encoding by utilizing the dynamic motion context inferred from the sequence of the feature maps. The experiments conducted on the public nuScenes benchmark demonstrate that the proposed 3D object detector offers significant improvements in performance compared to the baseline methods and that it sets a state-of-the-art performance for certain 3D object detection categories. Code is available at https://github.com/HYjhkoh/MGTANet.git.

----

## [131] InstanceFormer: An Online Video Instance Segmentation Framework

**Authors**: *Rajat Koner, Tanveer Hannan, Suprosanna Shit, Sahand Sharifzadeh, Matthias Schubert, Thomas Seidl, Volker Tresp*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25201](https://doi.org/10.1609/aaai.v37i1.25201)

**Abstract**:

Recent transformer-based offline video instance segmentation (VIS) approaches achieve encouraging results and significantly outperform online approaches. However, their reliance on the whole video and the immense computational complexity caused by full Spatio-temporal attention limit them in real-life applications such as processing lengthy videos. In this paper, we propose a single-stage transformer-based efficient online VIS framework named InstanceFormer, which is especially suitable for long and challenging videos. We propose three novel components to model short-term and long-term dependency and temporal coherence. First, we propagate the representation, location, and semantic information of prior instances to model short-term changes. Second, we propose a novel memory cross-attention in the decoder, which allows the network to look into earlier instances within a certain temporal window. Finally, we employ a temporal contrastive loss to impose coherence in the representation of an instance across all frames. Memory attention and temporal coherence are particularly beneficial to long-range dependency modeling, including challenging scenarios like occlusion. The proposed InstanceFormer outperforms previous online benchmark methods by a large margin across multiple datasets. Most importantly, InstanceFormer surpasses offline approaches for challenging and long datasets such as YouTube-VIS-2021 and OVIS. Code is available at https://github.com/rajatkoner08/InstanceFormer.

----

## [132] Pixel-Wise Warping for Deep Image Stitching

**Authors**: *Hyeokjun Kweon, Hyeonseong Kim, Yoonsu Kang, Youngho Yoon, Wooseong Jeong, Kuk-Jin Yoon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25202](https://doi.org/10.1609/aaai.v37i1.25202)

**Abstract**:

Existing image stitching approaches based on global or local homography estimation are not free from the parallax problem and suffer from undesired artifacts. In this paper, instead of relying on the homography-based warp, we propose a novel deep image stitching framework exploiting the pixel-wise warp field to handle the large-parallax problem. The proposed deep image stitching framework consists of a Pixel-wise Warping Module (PWM) and a Stitched Image Generating Module (SIGMo). For PWM, we obtain pixel-wise warp in a similar manner as estimating an optical flow (OF).
In the stitching scenario, the input images usually include non-overlap (NOV) regions of which warp cannot be directly estimated, unlike the overlap (OV) regions. To help the PWM predict a reasonable warp on the NOV region, we impose two geometrical constraints: an epipolar loss and a line-preservation loss. With the obtained warp field, we relocate the pixels of the target image using forward warping. Finally, the SIGMo is trained by the proposed multi-branch training framework to generate a stitched image from a reference image and a warped target image. For training and evaluating the proposed framework, we build and publish a novel dataset including image pairs with corresponding pixel-wise ground truth warp and stitched result images. We show that the results of the proposed framework are quantitatively and qualitatively superior to those of the conventional methods.

----

## [133] Learning to Learn Better for Video Object Segmentation

**Authors**: *Meng Lan, Jing Zhang, Lefei Zhang, Dacheng Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25203](https://doi.org/10.1609/aaai.v37i1.25203)

**Abstract**:

Recently, the joint learning framework (JOINT) integrates matching based transductive reasoning and online inductive learning to achieve accurate and robust semi-supervised video object segmentation (SVOS). However, using the mask embedding as the label to guide the generation of target features in the two branches may result in inadequate target representation and degrade the performance. Besides, how to reasonably fuse the target features in the two different branches rather than simply adding them together to avoid the adverse effect of one dominant branch has not been investigated. In this paper, we propose a novel framework that emphasizes Learning to Learn Better (LLB) target features for SVOS, termed LLB, where we design the discriminative label generation module (DLGM) and the adaptive fusion module to address these issues. Technically, the DLGM takes the background-filtered frame instead of the target mask as input and adopts a lightweight encoder to generate the target features, which serves as the label of the online few-shot learner and the value of the decoder in the transformer to guide the two branches to learn more discriminative target representation. The adaptive fusion module maintains a learnable gate for each branch, which reweighs the element-wise feature representation and allows an adaptive amount of target information in each branch flowing to the fused target feature, thus preventing one branch from being dominant and making the target feature more robust to distractor. Extensive experiments on public benchmarks show that our proposed LLB method achieves state-of-the-art performance.

----

## [134] Curriculum Multi-Negative Augmentation for Debiased Video Grounding

**Authors**: *Xiaohan Lan, Yitian Yuan, Hong Chen, Xin Wang, Zequn Jie, Lin Ma, Zhi Wang, Wenwu Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25204](https://doi.org/10.1609/aaai.v37i1.25204)

**Abstract**:

Video Grounding (VG) aims to locate the desired segment from a video given a sentence query. Recent studies have found that current VG models are prone to over-rely the groundtruth moment annotation distribution biases in the training set. To discourage the standard VG model's behavior of exploiting such temporal annotation biases and improve the model generalization ability, we propose multiple negative augmentations in a hierarchical way, including cross-video augmentations from clip-/video-level, and self-shuffled augmentations with masks. These augmentations can effectively diversify the data distribution so that the model can make more reasonable predictions instead of merely fitting the temporal biases. However, directly adopting such data augmentation strategy may inevitably carry some noise shown in our cases, since not all of the handcrafted augmentations are semantically irrelevant to the groundtruth video. To further denoise and improve the grounding accuracy, we design a multi-stage curriculum strategy to adaptively train the standard VG model from easy to hard negative augmentations. Experiments on newly collected Charades-CD and ActivityNet-CD datasets demonstrate our proposed strategy can improve the performance of the base model on both i.i.d and o.o.d scenarios.

----

## [135] Weakly Supervised 3D Segmentation via Receptive-Driven Pseudo Label Consistency and Structural Consistency

**Authors**: *Yuxiang Lan, Yachao Zhang, Yanyun Qu, Cong Wang, Chengyang Li, Jia Cai, Yuan Xie, Zongze Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25205](https://doi.org/10.1609/aaai.v37i1.25205)

**Abstract**:

As manual point-wise label is time and labor-intensive for fully supervised large-scale point cloud  semantic segmentation, weakly supervised method is increasingly active. However, existing methods fail to generate high-quality pseudo labels effectively, leading to unsatisfactory results. In this paper, we propose a weakly supervised point cloud semantic segmentation framework via receptive-driven pseudo label consistency and structural consistency to mine potential knowledge. Specifically, we propose three consistency contrains: pseudo label consistency among different scales,  semantic structure consistency between intra-class features and class-level relation structure consistency between pair-wise categories.  Three consistency constraints are jointly used to effectively prepares and utilizes pseudo labels simultaneously for stable training. Finally, extensive experimental results on three challenging datasets demonstrate that our method significantly outperforms state-of-the-art weakly supervised methods and even achieves comparable performance to the fully supervised methods.

----

## [136] MultiAct: Long-Term 3D Human Motion Generation from Multiple Action Labels

**Authors**: *Taeryung Lee, Gyeongsik Moon, Kyoung Mu Lee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25206](https://doi.org/10.1609/aaai.v37i1.25206)

**Abstract**:

We tackle the problem of generating long-term 3D human motion from multiple action labels. Two main previous approaches, such as action- and motion-conditioned methods, have limitations to solve this problem. The action-conditioned methods generate a sequence of motion from a single action. Hence, it cannot generate long-term motions composed of multiple actions and transitions between actions. Meanwhile, the motion-conditioned methods generate future motions from initial motion. The generated future motions only depend on the past, so they are not controllable by the user's desired actions. We present MultiAct, the first framework to generate long-term 3D human motion from multiple action labels. MultiAct takes account of both action and motion conditions with a unified recurrent generation system. It repetitively takes the previous motion and action label; then, it generates a smooth transition and the motion of the given action. As a result, MultiAct produces realistic long-term motion controlled by the given sequence of multiple action labels. The code is publicly available in https://github.com/TaeryungLee/MultiAct RELEASE.

----

## [137] Not All Neighbors Matter: Point Distribution-Aware Pruning for 3D Point Cloud

**Authors**: *Yejin Lee, Donghyun Lee, JungUk Hong, Jae W. Lee, Hongil Yoon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25207](https://doi.org/10.1609/aaai.v37i1.25207)

**Abstract**:

Applying deep neural networks to 3D point cloud processing has demonstrated a rapid pace of advancement in those domains where 3D geometry information can greatly boost task performance, such as AR/VR, robotics, and autonomous driving. However, as the size of both the neural network model and 3D point cloud continues to scale, reducing the entailed computation and memory access overhead is a primary challenge to meet strict latency and energy constraints of practical applications. This paper proposes a new weight pruning technique for 3D point cloud based on spatial point distribution. We identify that particular groups of neighborhood voxels in 3D point cloud contribute more frequently to actual output features than others. Based on this observation, we propose to selectively prune less contributing groups of neighborhood voxels first to reduce the computation overhead while minimizing the impact on model accuracy. We apply our proposal to three representative sparse 3D convolution libraries. Our proposal reduces the inference latency by 1.60× on average and energy consumption by 1.74× on NVIDIA GV100 GPU with no loss in accuracy metric

----

## [138] Symbolic Replay: Scene Graph as Prompt for Continual Learning on VQA Task

**Authors**: *Stan Weixian Lei, Difei Gao, Jay Zhangjie Wu, Yuxuan Wang, Wei Liu, Mengmi Zhang, Mike Zheng Shou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25208](https://doi.org/10.1609/aaai.v37i1.25208)

**Abstract**:

VQA is an ambitious task aiming to answer any image-related question. However, in reality, it is hard to build such a system once for all since the needs of users are continuously updated, and the system has to implement new functions. Thus, Continual Learning (CL) ability is a must in developing advanced VQA systems. Recently, a pioneer work split a VQA dataset into disjoint answer sets to study this topic. However, CL on VQA involves not only the expansion of label sets (new Answer sets). It is crucial to study how to answer questions when deploying VQA systems to new environments (new Visual scenes) and how to answer questions requiring new functions (new Question types). Thus, we propose CLOVE, a benchmark for Continual Learning On Visual quEstion answering, which contains scene- and function-incremental settings for the two aforementioned CL scenarios. In terms of methodology, the main difference between CL on VQA and classification is that the former additionally involves expanding and preventing forgetting of reasoning mechanisms, while the latter focusing on class representation. Thus, we propose a real-data-free replay-based method tailored for CL on VQA, named Scene Graph as Prompt for Symbolic Replay. Using a piece of scene graph as a prompt, it replays pseudo scene graphs to represent the past images, along with correlated QA pairs. A unified VQA model is also proposed to utilize the current and replayed data to enhance its QA ability. Finally, experimental results reveal challenges in CLOVE and demonstrate the effectiveness of our method. Code and data
are available at https://github.com/showlab/CLVQA.

----

## [139] Linking People across Text and Images Based on Social Relation Reasoning

**Authors**: *Yang Lei, Peizhi Zhao, Pijian Li, Yi Cai, Qingbao Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25209](https://doi.org/10.1609/aaai.v37i1.25209)

**Abstract**:

As a sub-task of visual grounding, linking people across text and images aims to localize target people in images with corresponding sentences. Existing approaches tend to capture superficial features of people (e.g., dress and location) that suffer from the incompleteness information across text and images. We observe that humans are adept at exploring social relations to assist identifying people. Therefore, we propose a Social Relation Reasoning (SRR) model to address the aforementioned issues. Firstly, we design a Social Relation Extraction (SRE) module to extract social relations between people in the input sentence. Specially, the SRE module based on zero-shot learning is able to extract social relations even though they are not defined in the existing datasets. A Reasoning based Cross-modal Matching (RCM) module is further used to generate matching matrices by reasoning on the social relations and visual features. Experimental results show that the accuracy of our proposed SRR model outperforms the state-of-the-art models on the challenging datasets Who's Waldo and FL: MSRE, by more than 5\% and 7\%, respectively. Our source code is available at https://github.com/VILAN-Lab/SRR.

----

## [140] ReGANIE: Rectifying GAN Inversion Errors for Accurate Real Image Editing

**Authors**: *Bingchuan Li, Tianxiang Ma, Peng Zhang, Miao Hua, Wei Liu, Qian He, Zili Yi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25210](https://doi.org/10.1609/aaai.v37i1.25210)

**Abstract**:

The StyleGAN family succeed in high-fidelity image generation and allow for flexible and plausible editing of generated images by manipulating the semantic-rich latent style space. However, projecting a real image into its latent space encounters an inherent trade-off between inversion quality and editability. Existing encoder-based or optimization-based StyleGAN inversion methods attempt to mitigate the trade-off but see limited performance. To fundamentally resolve this problem, we propose a novel two-phase framework by designating two separate networks to tackle editing and reconstruction respectively, instead of balancing the two. Specifically, in Phase I, a W-space-oriented StyleGAN inversion network is trained and used to perform image inversion and edit- ing, which assures the editability but sacrifices reconstruction quality. In Phase II, a carefully designed rectifying network is utilized to rectify the inversion errors and perform ideal reconstruction. Experimental results show that our approach yields near-perfect reconstructions without sacrificing the editability, thus allowing accurate manipulation of real images. Further, we evaluate the performance of our rectifying net- work, and see great generalizability towards unseen manipulation types and out-of-domain images.

----

## [141] SWBNet: A Stable White Balance Network for sRGB Images

**Authors**: *Chunxiao Li, Xuejing Kang, Zhifeng Zhang, Anlong Ming*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25211](https://doi.org/10.1609/aaai.v37i1.25211)

**Abstract**:

The white balance methods for sRGB images (sRGB-WB)  aim to directly remove their color temperature shifts. Despite  achieving promising white balance (WB) performance, the  existing methods suffer from WB instability, i.e., their results  are inconsistent for images with different color temperatures. We propose a stable white balance network (SWBNet) to alleviate this problem. It learns the color temperature-insensitive  features to generate white-balanced images, resulting in  consistent WB results. Specifically, the color temperatureinsensitive features are learned by implicitly suppressing lowfrequency information sensitive to color temperatures. Then,  a color temperature contrastive loss is introduced to facilitate the most information shared among features of the same  scene and different color temperatures. This way, features  from the same scene are more insensitive to color temperatures regardless of the inputs. We also present a color temperature sensitivity-oriented transformer that globally perceives multiple color temperature shifts within an image  and corrects them by different weights. It helps to improve  the accuracy of stabilized SWBNet, especially for multiillumination sRGB images. Experiments indicate that our SWBNet achieves stable and remarkable WB performance.

----

## [142] Frequency Domain Disentanglement for Arbitrary Neural Style Transfer

**Authors**: *Dongyang Li, Hao Luo, Pichao Wang, Zhibin Wang, Shang Liu, Fan Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25212](https://doi.org/10.1609/aaai.v37i1.25212)

**Abstract**:

Arbitrary neural style transfer has been a popular research topic due to its rich application scenarios. Effective disentanglement of content and style is the critical factor for synthesizing an image with arbitrary style. The existing methods focus on disentangling feature representations of content and style in the spatial domain where the content and style components are innately entangled and difficult to be disentangled clearly. Therefore, these methods always suffer from low-quality results because of the sub-optimal disentanglement. To address such a challenge, this paper proposes the frequency mixer (FreMixer) module that disentangles and re-entangles the frequency spectrum of content and style components in the frequency domain. Since content and style components have different frequency-domain characteristics (frequency bands and frequency patterns), the FreMixer could well disentangle these two components. Based on the FreMixer module, we design a novel Frequency Domain Disentanglement (FDD) framework for arbitrary neural style transfer. Qualitative and quantitative experiments verify that the proposed method can render better stylized results compared to the state-of-the-art methods.

----

## [143] Pose-Oriented Transformer with Uncertainty-Guided Refinement for 2D-to-3D Human Pose Estimation

**Authors**: *Han Li, Bowen Shi, Wenrui Dai, Hongwei Zheng, Botao Wang, Yu Sun, Min Guo, Chenglin Li, Junni Zou, Hongkai Xiong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25213](https://doi.org/10.1609/aaai.v37i1.25213)

**Abstract**:

There has been a recent surge of interest in introducing transformers to 3D human pose estimation (HPE) due to their powerful capabilities in modeling long-term dependencies. However, existing transformer-based methods treat body joints as equally important inputs and ignore the prior knowledge of human skeleton topology in the self-attention mechanism. To tackle this issue, in this paper, we propose a Pose-Oriented Transformer (POT) with uncertainty guided refinement for 3D HPE. Specifically, we first develop novel pose-oriented self-attention mechanism and distance-related position embedding for POT to explicitly exploit the human skeleton topology. The pose-oriented self-attention mechanism explicitly models the topological interactions between body joints, whereas the distance-related position embedding encodes the distance of joints to the root joint to distinguish groups of joints with different difficulties in regression. Furthermore, we present an Uncertainty-Guided Refinement Network (UGRN) to refine pose predictions from POT, especially for the difficult joints, by considering the estimated uncertainty of each joint with uncertainty-guided sampling strategy and self-attention mechanism. Extensive experiments demonstrate that our method significantly outperforms the state-of-the-art methods with reduced model parameters on 3D HPE benchmarks such as Human3.6M and MPI-INF-3DHP.

----

## [144] CEE-Net: Complementary End-to-End Network for 3D Human Pose Generation and Estimation

**Authors**: *Haolun Li, Chi-Man Pun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25214](https://doi.org/10.1609/aaai.v37i1.25214)

**Abstract**:

The limited number of actors and actions in existing datasets make 3D pose estimators tend to overfit, which can be seen from the performance degradation of the algorithm on cross-datasets, especially for rare and complex poses. Although previous data augmentation works have increased the diversity of the training set, the changes in camera viewpoint and position play a dominant role in improving the accuracy of the estimator, while the generated 3D poses are limited and still heavily rely on the source dataset. In addition, these works do not consider the adaptability of the pose estimator to generated data, and complex poses will cause training collapse. In this paper, we propose the CEE-Net, a Complementary End-to-End Network for 3D human pose generation and estimation. The generator extremely expands the distribution of each joint-angle in the existing dataset and limits them to a reasonable range. By learning the correlations within and between the torso and limbs, the estimator can combine different body-parts more effectively and weaken the influence of specific joint-angle changes on the global pose, improving the generalization ability. Extensive ablation studies show that our pose generator greatly strengthens the joint-angle distribution, and our pose estimator can utilize these poses positively. Compared with the state-of-the-art methods, our method can achieve much better performance on various cross-datasets, rare and complex poses.

----

## [145] Real-World Deep Local Motion Deblurring

**Authors**: *Haoying Li, Ziran Zhang, Tingting Jiang, Peng Luo, Huajun Feng, Zhihai Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25215](https://doi.org/10.1609/aaai.v37i1.25215)

**Abstract**:

Most existing deblurring methods focus on removing global blur caused by camera shake, while they cannot well handle local blur caused by object movements. To fill the vacancy of local deblurring in real scenes, we establish the first real local motion blur dataset (ReLoBlur), which is captured by a synchronized beam-splitting photographing system and corrected by a post-progressing pipeline. Based on ReLoBlur, we propose a Local Blur-Aware Gated network (LBAG) and several local blur-aware techniques to bridge the gap between global and local deblurring: 1) a blur detection approach based on background subtraction to localize blurred regions; 2) a gate mechanism to guide our network to focus on blurred regions; and 3) a blur-aware patch cropping strategy to address data imbalance problem. Extensive experiments prove the reliability of ReLoBlur dataset, and demonstrate that LBAG achieves better performance than state-of-the-art global deblurring methods and our proposed local blur-aware techniques are effective.

----

## [146] Disentangle and Remerge: Interventional Knowledge Distillation for Few-Shot Object Detection from a Conditional Causal Perspective

**Authors**: *Jiangmeng Li, Yanan Zhang, Wenwen Qiang, Lingyu Si, Chengbo Jiao, Xiaohui Hu, Changwen Zheng, Fuchun Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25216](https://doi.org/10.1609/aaai.v37i1.25216)

**Abstract**:

Few-shot learning models learn representations with limited human annotations, and such a learning paradigm demonstrates practicability in various tasks, e.g., image classification, object detection, etc. However, few-shot object detection methods suffer from an intrinsic defect that the limited training data makes the model cannot sufficiently explore semantic information. To tackle this, we introduce knowledge distillation to the few-shot object detection learning paradigm. We further run a motivating experiment, which demonstrates that in the process of knowledge distillation, the empirical error of the teacher model degenerates the prediction performance of the few-shot object detection model as the student. To understand the reasons behind this phenomenon, we revisit the learning paradigm of knowledge distillation on the few-shot object detection task from the causal theoretic standpoint, and accordingly, develop a Structural Causal Model. Following the theoretical guidance, we propose a backdoor adjustment-based knowledge distillation method for the few-shot object detection task, namely Disentangle and Remerge (D&R), to perform conditional causal intervention toward the corresponding Structural Causal Model. Empirically, the experiments on benchmarks demonstrate that D&R can yield significant performance boosts in few-shot object detection. Code is available at https://github.com/ZYN-1101/DandR.git.

----

## [147] Learning Motion-Robust Remote Photoplethysmography through Arbitrary Resolution Videos

**Authors**: *Jianwei Li, Zitong Yu, Jingang Shi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25217](https://doi.org/10.1609/aaai.v37i1.25217)

**Abstract**:

Remote photoplethysmography (rPPG) enables non-contact heart rate (HR) estimation from facial videos which gives significant convenience compared with traditional contact-based measurements. In the real-world long-term health monitoring scenario, the distance of the participants and their head movements usually vary by time, resulting in the inaccurate rPPG measurement due to the varying face resolution and complex motion artifacts. Different from the previous rPPG models designed for a constant distance between camera and participants, in this paper, we propose two plug-and-play blocks (i.e., physiological signal feature extraction block (PFE) and temporal face alignment block (TFA)) to alleviate the degradation of changing distance and head motion.
On one side, guided with representative-area information, PFE adaptively encodes the arbitrary resolution facial frames to the fixed-resolution facial structure features. On the other side, leveraging the estimated optical flow, TFA is able to counteract the rPPG signal confusion caused by the head movement thus benefit the motion-robust rPPG signal recovery. Besides, we also train the model with a cross-resolution constraint using a two-stream dual-resolution framework, which further helps PFE learn resolution-robust facial rPPG features. Extensive experiments on three benchmark datasets (UBFC-rPPG, COHFACE and PURE) demonstrate the superior performance of the proposed method. One highlight is that with PFE and TFA, the off-the-shelf spatio-temporal rPPG models can predict more robust rPPG signals under both varying face resolution and severe head movement scenarios. The codes are available at https://github.com/LJWGIT/Arbitrary_Resolution_rPPG.

----

## [148] FSR: A General Frequency-Oriented Framework to Accelerate Image Super-resolution Networks

**Authors**: *Jinmin Li, Tao Dai, Mingyan Zhu, Bin Chen, Zhi Wang, Shu-Tao Xia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25218](https://doi.org/10.1609/aaai.v37i1.25218)

**Abstract**:

Deep neural networks (DNNs) have witnessed remarkable achievement in image super-resolution (SR), and plenty of DNN-based SR models with elaborated network designs have recently been proposed. However, existing methods usually require substantial computations by operating in spatial domain. To address this issue, we propose a general frequency-oriented framework (FSR) to accelerate SR networks by considering data characteristics in frequency domain. Our FSR mainly contains dual feature aggregation module (DFAM) to extract informative features in both spatial and transform domains, followed by a four-path SR-Module with different capacities to super-resolve in the frequency domain. Specifically, DFAM further consists of a transform attention block (TABlock) and a spatial context block (SCBlock) to extract global spectral information and local spatial information, respectively, while SR-Module is a parallel network container that contains four to-be-accelerated branches. Furthermore, we propose an adaptive weight strategy for a trade-off between image details recovery and visual quality. Extensive experiments show that our FSR can save FLOPs by almost 40% while reducing inference time by 50% for other SR methods (e.g., FSRCNN, CARN, SRResNet and RCAN). Code is available at https://github.com/THU-Kingmin/FSR.

----

## [149] Learning Polysemantic Spoof Trace: A Multi-Modal Disentanglement Network for Face Anti-spoofing

**Authors**: *Kaicheng Li, Hongyu Yang, Binghui Chen, Pengyu Li, Biao Wang, Di Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25219](https://doi.org/10.1609/aaai.v37i1.25219)

**Abstract**:

Along with the widespread use of face recognition systems, their vulnerability has become highlighted. While existing face anti-spoofing methods can be generalized between attack types, generic solutions are still challenging due to the diversity of spoof characteristics. Recently, the spoof trace disentanglement framework has shown great potential for coping with both seen and unseen spoof scenarios, but the performance is largely restricted by the single-modal input. This paper focuses on this issue and presents a multi-modal disentanglement model which targetedly learns polysemantic spoof traces for more accurate and robust generic attack detection. In particular, based on the adversarial learning mechanism, a two-stream disentangling network is designed to estimate spoof patterns from the RGB and depth inputs, respectively. In this case, it captures complementary spoofing clues inhering in different attacks. Furthermore, a fusion module is exploited, which recalibrates both representations at multiple stages to promote the disentanglement in each individual modality. It then performs cross-modality aggregation to deliver a more comprehensive spoof trace representation for prediction. Extensive evaluations are conducted on multiple benchmarks, demonstrating that learning polysemantic spoof traces favorably contributes to anti-spoofing with more perceptible and interpretable results.

----

## [150] Stroke Extraction of Chinese Character Based on Deep Structure Deformable Image Registration

**Authors**: *Meng Li, Yahan Yu, Yi Yang, Guanghao Ren, Jian Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25220](https://doi.org/10.1609/aaai.v37i1.25220)

**Abstract**:

Stroke extraction of Chinese characters plays an important role in the field of character recognition and generation. The most existing character stroke extraction methods focus on image morphological features. These methods usually lead to errors of cross strokes extraction and stroke matching due to rarely using stroke semantics and prior information. In this paper, we propose a deep learning-based character stroke extraction method that takes semantic features and prior information of strokes into consideration. This method consists of three parts: image registration-based stroke registration that establishes the rough registration of the reference strokes and the target as prior information; image semantic segmentation-based stroke segmentation that preliminarily separates target strokes into seven categories; and high-precision extraction of single strokes. In the stroke registration, we propose a structure deformable image registration network to achieve structure-deformable transformation while maintaining the stable morphology of single strokes for character images with complex structures. In order to verify the effectiveness of the method, we construct two datasets respectively for calligraphy characters and regular handwriting characters. The experimental results show that our method strongly outperforms the baselines. Code is available at https://github.com/MengLi-l1/StrokeExtraction.

----

## [151] Spatial-Spectral Transformer for Hyperspectral Image Denoising

**Authors**: *Miaoyu Li, Ying Fu, Yulun Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25221](https://doi.org/10.1609/aaai.v37i1.25221)

**Abstract**:

Hyperspectral image (HSI) denoising is a crucial preprocessing procedure for the subsequent HSI applications. Unfortunately, though witnessing the development of deep learning in HSI denoising area, existing convolution-based methods face the trade-off between computational efficiency and capability to model non-local characteristics of HSI. In this paper, we propose a Spatial-Spectral Transformer (SST) to alleviate this problem. To fully explore intrinsic similarity characteristics in both spatial dimension and spectral dimension, we conduct non-local spatial self-attention and global spectral self-attention with Transformer architecture. The window-based spatial self-attention focuses on the spatial similarity beyond the neighboring region. While, the spectral self-attention exploits the long-range dependencies between highly correlative bands. Experimental results show that our proposed method outperforms the state-of-the-art HSI denoising methods in quantitative quality and visual results. The code is released at https://github.com/MyuLi/SST.

----

## [152] Learning Semantic Alignment with Global Modality Reconstruction for Video-Language Pre-training towards Retrieval

**Authors**: *Mingchao Li, Xiaoming Shi, Haitao Leng, Wei Zhou, Hai-Tao Zheng, Kuncai Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25222](https://doi.org/10.1609/aaai.v37i1.25222)

**Abstract**:

Video-language pre-training for text-based video retrieval tasks is vitally important. Previous pre-training methods suffer from the semantic misalignments. The reason is that these methods ignore sequence alignments but focusing on critical token alignment. To alleviate the problem, we propose a video-language pre-training framework, termed videolanguage pre-training For lEarning sEmantic aLignments (FEEL), to learn semantic alignments at the sequence level. Specifically, the global modality reconstruction and the cross- modal self-contrasting method is utilized to learn the alignments at the sequence level better. Extensive experimental results demonstrate the effectiveness of FEEL on text-based video retrieval and text-based video corpus moment retrieval.

----

## [153] Layout-Aware Dreamer for Embodied Visual Referring Expression Grounding

**Authors**: *Mingxiao Li, Zehao Wang, Tinne Tuytelaars, Marie-Francine Moens*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25223](https://doi.org/10.1609/aaai.v37i1.25223)

**Abstract**:

In this work, we study the problem of Embodied Referring Expression Grounding, where an agent needs to navigate in a previously unseen environment and localize a remote object described by a concise high-level natural language instruction. When facing such a situation, a human tends to imagine what the destination may look like and to explore the environment based on prior knowledge of the environmental layout, such as the fact that a bathroom is more likely to be found near a bedroom than a kitchen. We have designed an autonomous agent called Layout-aware Dreamer (LAD), including two novel modules, that is, the Layout Learner and the Goal Dreamer to mimic this cognitive decision process. The Layout Learner learns to infer the room category distribution of neighboring unexplored areas along the path for coarse layout estimation, which effectively introduces layout common sense of room-to-room transitions to our agent. To learn an effective exploration of the environment, the Goal Dreamer imagines the destination beforehand. Our agent achieves new state-of-the-art performance on the public leaderboard of REVERIE dataset in challenging unseen test environments with improvement on navigation success rate (SR) by 4.02% and remote grounding success (RGS) by 3.43% comparing to previous previous state of the art. The code
is released at https://github.com/zehao-wang/LAD.

----

## [154] NeAF: Learning Neural Angle Fields for Point Normal Estimation

**Authors**: *Shujuan Li, Junsheng Zhou, Baorui Ma, Yu-Shen Liu, Zhizhong Han*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25224](https://doi.org/10.1609/aaai.v37i1.25224)

**Abstract**:

Normal estimation for unstructured point clouds is an important task in 3D computer vision. Current methods achieve encouraging results by mapping local patches to normal vectors or learning local surface fitting using neural networks. However, these methods are not generalized well to unseen scenarios and are sensitive to parameter settings. To resolve these issues, we propose an implicit function to learn an angle field around the normal of each point in the spherical coordinate system, which is dubbed as Neural Angle Fields (NeAF). Instead of directly predicting the normal of an input point, we predict the angle offset between the ground truth normal and a randomly sampled query normal. This strategy pushes the network to observe more diverse samples, which leads to higher prediction accuracy in a more robust manner. To predict normals from the learned angle fields at inference time, we randomly sample query vectors in a unit spherical space and take the vectors with minimal angle values as the predicted normals. To further leverage the prior learned by NeAF, we propose to refine the predicted normal vectors by minimizing the angle offsets. The experimental results with synthetic data and real scans show significant improvements over the state-of-the-art under widely used benchmarks. Project page: https://lisj575.github.io/NeAF/.

----

## [155] CLIP-ReID: Exploiting Vision-Language Model for Image Re-identification without Concrete Text Labels

**Authors**: *Siyuan Li, Li Sun, Qingli Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i1.25225](https://doi.org/10.1609/aaai.v37i1.25225)

**Abstract**:

Pre-trained vision-language models like CLIP have recently shown superior performances on various downstream tasks, including image classification and segmentation. However, in fine-grained image re-identification (ReID), the labels are indexes, lacking concrete text descriptions. Therefore, it remains to be determined how such models could be applied to these tasks. This paper first finds out that simply fine-tuning the visual model initialized by the image encoder in CLIP, has already obtained competitive performances in various ReID tasks. Then we propose a two-stage strategy to facilitate a better visual representation. The key idea is to fully exploit the cross-modal description ability in CLIP through a set of learnable text tokens for each ID and give them to the text encoder to form ambiguous descriptions. In the first training stage, image and text encoders from CLIP keep fixed, and only the text tokens are optimized from scratch by the contrastive loss computed within a batch. In the second stage, the ID-specific text tokens and their encoder become static, providing constraints for fine-tuning the image encoder. With the help of the designed loss in the downstream task, the image encoder is able to represent data as vectors in the feature embedding accurately. The effectiveness of the proposed strategy is validated on several datasets for the person or vehicle ReID tasks. Code is available at https://github.com/Syliz517/CLIP-ReID.

----

## [156] DC-Former: Diverse and Compact Transformer for Person Re-identification

**Authors**: *Wen Li, Cheng Zou, Meng Wang, Furong Xu, Jianan Zhao, Ruobing Zheng, Yuan Cheng, Wei Chu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25226](https://doi.org/10.1609/aaai.v37i2.25226)

**Abstract**:

In person re-identification (ReID) task, it is still challenging to learn discriminative representation by deep learning, due to limited data. Generally speaking, the model will get better performance when increasing the amount of data. The addition of similar classes strengthens the ability of the classifier to identify similar identities, thereby improving the discrimination of representation. In this paper, we propose a Diverse and Compact Transformer (DC-Former) that can achieve a similar effect by splitting embedding space into multiple diverse and compact subspaces. Compact embedding subspace helps model learn more robust and discriminative embedding to identify similar classes. And the fusion of these diverse embeddings containing more fine-grained information can further improve the effect of ReID. Specifically, multiple class tokens are used in vision transformer to represent multiple embedding spaces. Then, a self-diverse constraint (SDC) is applied to these spaces to push them away from each other, which makes each embedding space diverse and compact. Further, a dynamic weight controller (DWC) is further designed for balancing the relative importance among them during training. The experimental results of our method are promising, which surpass previous state-of-the-art methods on several commonly used person ReID benchmarks. Our code is available at https://github.com/ant-research/Diverse-and-Compact-Transformer.

----

## [157] Panoramic Video Salient Object Detection with Ambisonic Audio Guidance

**Authors**: *Xiang Li, Haoyuan Cao, Shijie Zhao, Junlin Li, Li Zhang, Bhiksha Raj*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25227](https://doi.org/10.1609/aaai.v37i2.25227)

**Abstract**:

Video salient object detection (VSOD), as a fundamental computer vision problem, has been extensively discussed in the last decade. However, all existing works focus on addressing the VSOD problem in 2D scenarios. With the rapid development of VR devices, panoramic videos have been a promising alternative to 2D videos to provide immersive feelings of the real world. In this paper, we aim to tackle the video salient object detection problem for panoramic videos, with their corresponding ambisonic audios. A multimodal fusion module equipped with two pseudo-siamese audio-visual context fusion (ACF) blocks is proposed to effectively conduct audio-visual interaction. The ACF block equipped with spherical positional encoding enables the fusion in the 3D context to capture the spatial correspondence between pixels and sound sources from the equirectangular frames and ambisonic audios. Experimental results verify the effectiveness of our proposed components and demonstrate that our method achieves state-of-the-art performance on the ASOD60K dataset.

----

## [158] LWSIS: LiDAR-Guided Weakly Supervised Instance Segmentation for Autonomous Driving

**Authors**: *Xiang Li, Junbo Yin, Botian Shi, Yikang Li, Ruigang Yang, Jianbing Shen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25228](https://doi.org/10.1609/aaai.v37i2.25228)

**Abstract**:

Image instance segmentation is a fundamental research topic in autonomous driving, which is crucial for scene understanding and road safety. Advanced learning-based approaches often rely on the costly 2D mask annotations for training. 
In this paper, we present a more artful framework, LiDAR-guided Weakly Supervised Instance Segmentation (LWSIS), which leverages the off-the-shelf 3D data, i.e., Point Cloud, together with the 3D boxes, as natural weak supervisions for
training the 2D image instance segmentation models. Our LWSIS not only exploits the complementary information in multimodal data during training but also significantly reduces the annotation cost of the dense 2D masks. In detail, LWSIS consists of two crucial modules, Point Label Assignment (PLA) and Graph-based Consistency Regularization (GCR). The former module aims to automatically assign the 3D point cloud as 2D point-wise labels, while the atter further refines the predictions by enforcing geometry and appearance consistency of the multimodal data. Moreover, we conduct a secondary instance segmentation annotation on the nuScenes, named nuInsSeg, to encourage further research on multimodal perception tasks. Extensive experiments on the nuInsSeg, as well as the large-scale Waymo, show that LWSIS can substantially improve existing weakly supervised segmentation models by only involving 3D data during training. Additionally, LWSIS can also be incorporated into 3D object detectors like PointPainting to boost the 3D detection performance for free. The code and dataset are available at https://github.com/Serenos/LWSIS.

----

## [159] Adaptive Texture Filtering for Single-Domain Generalized Segmentation

**Authors**: *Xinhui Li, Mingjia Li, Yaxing Wang, Chuan-Xian Ren, Xiaojie Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25229](https://doi.org/10.1609/aaai.v37i2.25229)

**Abstract**:

Domain generalization in semantic segmentation aims to alleviate the performance degradation on unseen domains through learning domain-invariant features. Existing methods diversify images in the source domain by adding complex or even abnormal textures to reduce the sensitivity to domain-specific features. However, these approaches depends heavily on the richness of the texture bank and training them can be time-consuming. In contrast to importing textures arbitrarily or augmenting styles randomly, we focus on the single source domain itself to achieve the generalization. In this paper, we present a novel adaptive texture filtering mechanism to suppress the influence of texture without using augmentation, thus eliminating the interference of domain-specific features. Further, we design a hierarchical guidance generalization network equipped with structure-guided enhancement modules, which purpose to learn the domain-invariant generalized knowledge. Extensive experiments together with ablation studies on widely-used datasets are conducted to verify the effectiveness of the proposed model, and reveal its superiority over other state-of-the-art alternatives.

----

## [160] MEID: Mixture-of-Experts with Internal Distillation for Long-Tailed Video Recognition

**Authors**: *Xinjie Li, Huijuan Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25230](https://doi.org/10.1609/aaai.v37i2.25230)

**Abstract**:

The long-tailed video recognition problem is especially challenging, as videos tend to be long and untrimmed, and each video may contain multiple classes, causing frame-level class imbalance. The previous method tackles the long-tailed video recognition only through frame-level sampling for class re-balance without distinguishing the frame-level feature representation between head and tail classes. To improve the frame-level feature representation of tail classes, we modulate the frame-level features with an auxiliary distillation loss to reduce the distribution distance between head and tail classes. Moreover, we design a mixture-of-experts framework with two different expert designs, i.e., the first expert with an attention-based classification network handling the original long-tailed distribution, and the second expert dealing with the re-balanced distribution from class-balanced sampling. Notably, in the second expert, we specifically focus on the frames unsolved by the first expert through designing a complementary frame selection module, which inherits the attention weights from the first expert and selects frames with low attention weights, and we also enhance the motion feature representation for these selected frames. To highlight the multi-label challenge in long-tailed video recognition, we create two additional benchmarks based on Charades and CharadesEgo videos with the multi-label property, called CharadesLT and CharadesEgoLT. Extensive experiments are conducted on the existing long-tailed video benchmark VideoLT and the two new benchmarks to verify the effectiveness of our proposed method with state-of-the-art performance. The code and proposed benchmarks are released at https://github.com/VisionLanguageLab/MEID.

----

## [161] Gradient Corner Pooling for Keypoint-Based Object Detection

**Authors**: *Xuyang Li, Xuemei Xie, Mingxuan Yu, Jiakai Luo, Chengwei Rao, Guangming Shi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25231](https://doi.org/10.1609/aaai.v37i2.25231)

**Abstract**:

Detecting objects as multiple keypoints is an important approach in the anchor-free object detection methods while corner pooling is an effective feature encoding method for corner positioning. The corners of the bounding box are located by summing the feature maps which are max-pooled in the x and y directions respectively by corner pooling. In the unidirectional max pooling operation, the features of the densely arranged objects of the same class are prone to occlusion. To this end, we propose a method named Gradient Corner Pooling. The spatial distance information of objects on the feature map is encoded during the unidirectional pooling process, which effectively alleviates the occlusion of the homogeneous object features. Further, the computational complexity of gradient corner pooling is the same as traditional corner pooling and hence it can be implemented efficiently. Gradient corner pooling obtains consistent improvements for various keypoint-based methods by directly replacing corner pooling. We verify the gradient corner pooling algorithm on the dataset and in real scenarios, respectively. The networks with gradient corner pooling located the corner points earlier in the training process and achieve an average accuracy improvement of 0.2%-1.6% on the MS-COCO dataset. The detectors with gradient corner pooling show better angle adaptability for arrayed objects in the actual scene test.

----

## [162] Towards Real-Time Segmentation on the Edge

**Authors**: *Yanyu Li, Changdi Yang, Pu Zhao, Geng Yuan, Wei Niu, Jiexiong Guan, Hao Tang, Minghai Qin, Qing Jin, Bin Ren, Xue Lin, Yanzhi Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25232](https://doi.org/10.1609/aaai.v37i2.25232)

**Abstract**:

The research in real-time segmentation mainly focuses on desktop GPUs. 
However, autonomous driving and many other applications rely on real-time segmentation on the edge, and current arts are far from the goal. 
In addition, recent advances in vision transformers also inspire us to re-design the network architecture for dense prediction task. 
In this work, we propose to combine the self attention block with lightweight convolutions to form new building blocks, and employ latency constraints to search an efficient sub-network. 
We train an MLP latency model based on generated architecture configurations and their latency measured on mobile devices, so that we can predict the latency of subnets during search phase. 
To the best of our knowledge, we are the first to achieve over 74% mIoU on Cityscapes with semi-real-time inference (over 15 FPS) on mobile GPU from an off-the-shelf phone.

----

## [163] BEVDepth: Acquisition of Reliable Depth for Multi-View 3D Object Detection

**Authors**: *Yinhao Li, Zheng Ge, Guanyi Yu, Jinrong Yang, Zengran Wang, Yukang Shi, Jianjian Sun, Zeming Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25233](https://doi.org/10.1609/aaai.v37i2.25233)

**Abstract**:

In this research, we propose a new 3D object detector with a trustworthy depth estimation,  dubbed BEVDepth, for camera-based Bird's-Eye-View~(BEV) 3D object detection. Our work is based on a key observation -- depth estimation in recent approaches is surprisingly inadequate given the fact that depth is essential to camera 3D detection. Our BEVDepth resolves this by leveraging explicit depth supervision. A camera-awareness depth estimation module is also introduced to facilitate the depth predicting capability. Besides, we design a novel Depth Refinement Module to counter the side effects carried by imprecise feature unprojection. Aided by customized Efficient Voxel Pooling and multi-frame mechanism, BEVDepth achieves the new state-of-the-art 60.9% NDS on the challenging nuScenes test set while maintaining high efficiency. For the first time, the NDS score of a camera model reaches 60%. Codes have been released.

----

## [164] BEVStereo: Enhancing Depth Estimation in Multi-View 3D Object Detection with Temporal Stereo

**Authors**: *Yinhao Li, Han Bao, Zheng Ge, Jinrong Yang, Jianjian Sun, Zeming Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25234](https://doi.org/10.1609/aaai.v37i2.25234)

**Abstract**:

Restricted by the ability of depth perception, all Multi-view 3D object detection methods fall into the bottleneck of depth accuracy. By constructing temporal stereo, depth estimation is quite reliable in indoor scenarios. However, there are two difficulties in directly integrating temporal stereo into outdoor multi-view 3D object detectors: 1) The construction of temporal stereos for all views results in high computing costs. 2) Unable to adapt to challenging outdoor scenarios. In this study, we propose an effective method for creating temporal stereo by dynamically determining the center and range of the temporal stereo. The most confident center is found using the EM algorithm. Numerous experiments on nuScenes have shown the BEVStereo's ability to deal with complex outdoor scenarios that other stereo-based methods are unable to handle.  For the first time, a stereo-based approach shows superiority in scenarios like a static ego vehicle and moving objects. BEVStereo achieves the new state-of-the-art in the camera-only track of nuScenes dataset while maintaining memory efficiency.  Codes have been released.

----

## [165] Learning Single Image Defocus Deblurring with Misaligned Training Pairs

**Authors**: *Yu Li, Dongwei Ren, Xinya Shu, Wangmeng Zuo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25235](https://doi.org/10.1609/aaai.v37i2.25235)

**Abstract**:

By adopting popular pixel-wise loss, existing methods for defocus deblurring heavily rely on well aligned training image pairs. Although training pairs of ground-truth and blurry images are carefully collected, e.g., DPDD dataset, misalignment is inevitable between training pairs, making existing methods possibly suffer from deformation artifacts. In this paper, we propose a joint deblurring and reblurring learning (JDRL) framework for single image defocus deblurring with misaligned training pairs. Generally, JDRL consists of a deblurring module and a spatially invariant reblurring module, by which deblurred result can be adaptively supervised by ground-truth image to recover sharp textures while maintaining spatial consistency with the blurry image. First, in the deblurring module, a bi-directional optical flow-based deformation is introduced to tolerate spatial misalignment between deblurred and ground-truth images. Second, in the reblurring module, deblurred result is reblurred to be spatially aligned with blurry image, by predicting a set of isotropic blur kernels and weighting maps. Moreover, we establish a new single image defocus deblurring (SDD) dataset, further validating our JDRL and also benefiting future research. Our JDRL can be applied to boost defocus deblurring networks in terms of both quantitative metrics and visual quality on DPDD, RealDOF and our SDD datasets.

----

## [166] Curriculum Temperature for Knowledge Distillation

**Authors**: *Zheng Li, Xiang Li, Lingfeng Yang, Borui Zhao, Renjie Song, Lei Luo, Jun Li, Jian Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25236](https://doi.org/10.1609/aaai.v37i2.25236)

**Abstract**:

Most existing distillation methods ignore the flexible role of the temperature in the loss function and fix it as a hyper-parameter that can be decided by an inefficient grid search. In general, the temperature controls the discrepancy between two distributions and can faithfully determine the difficulty level of the distillation task. Keeping a constant temperature, i.e., a fixed level of task difficulty, is usually sub-optimal for a growing student during its progressive learning stages. In this paper, we propose a simple curriculum-based technique, termed Curriculum Temperature for Knowledge Distillation (CTKD), which controls the task difficulty level during the student's learning career through a dynamic and learnable temperature. Specifically, following an easy-to-hard curriculum, we gradually increase the distillation loss w.r.t. the temperature, leading to increased distillation difficulty in an adversarial manner. As an easy-to-use plug-in technique, CTKD can be seamlessly integrated into existing knowledge distillation frameworks and brings general improvements at a negligible additional computation cost. Extensive experiments on CIFAR-100, ImageNet-2012, and MS-COCO demonstrate the effectiveness of our method.

----

## [167] Actionness Inconsistency-Guided Contrastive Learning for Weakly-Supervised Temporal Action Localization

**Authors**: *Zhilin Li, Zilei Wang, Qinying Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25237](https://doi.org/10.1609/aaai.v37i2.25237)

**Abstract**:

Weakly-supervised temporal action localization (WTAL) aims to detect action instances given only video-level labels. To address the challenge, recent methods commonly employ a two-branch framework, consisting of a class-aware branch and a class-agnostic branch. In principle, the two branches are supposed to produce the same actionness activation. However, we observe that there are actually many inconsistent activation regions. These inconsistent regions usually contain some challenging segments whose semantic information (action or background) is ambiguous. In this work, we propose a novel Actionness Inconsistency-guided Contrastive Learning (AICL) method which utilizes the consistent segments to boost the representation learning of the inconsistent segments. Specifically, we first define the consistent and inconsistent segments by comparing the predictions of two branches and then construct positive and negative pairs between consistent segments and inconsistent segments for contrastive learning. In addition, to avoid the trivial case where there is no consistent sample, we introduce an action consistency constraint to control the difference between the two branches. We conduct extensive experiments on THUMOS14, ActivityNet v1.2, and ActivityNet v1.3 datasets, and the results show the effectiveness of AICL with state-of-the-art performance. Our code is available at https://github.com/lizhilin-ustc/AAAI2023-AICL.

----

## [168] READ: Large-Scale Neural Scene Rendering for Autonomous Driving

**Authors**: *Zhuopeng Li, Lu Li, Jianke Zhu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25238](https://doi.org/10.1609/aaai.v37i2.25238)

**Abstract**:

With the development of advanced driver assistance systems~(ADAS) and autonomous vehicles, conducting experiments in various scenarios becomes an urgent need. Although having been capable of synthesizing photo-realistic street scenes, conventional image-to-image translation methods cannot produce coherent scenes due to the lack of 3D information. In this paper, a large-scale neural rendering method is proposed to synthesize the autonomous driving scene~(READ), which makes it possible to generate large-scale driving scenes in real time on a PC through a variety of sampling schemes. In order to effectively represent driving scenarios, we propose an ω-net rendering network to learn neural descriptors from sparse point clouds. Our model can not only synthesize photo-realistic driving scenes but also stitch and edit them. The promising experimental results show that our model performs well in large-scale driving scenarios.

----

## [169] CDTA: A Cross-Domain Transfer-Based Attack with Contrastive Learning

**Authors**: *Zihan Li, Weibin Wu, Yuxin Su, Zibin Zheng, Michael R. Lyu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25239](https://doi.org/10.1609/aaai.v37i2.25239)

**Abstract**:

Despite the excellent performance, deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples. Besides, these examples are often transferable among different models. In other words, the same adversarial example can fool multiple models with different architectures at the same time. Based on this property, many black-box transfer-based attack techniques have been developed. However, current transfer-based attacks generally focus on the cross-architecture setting, where the attacker has access to the training data of the target model, which is not guaranteed in realistic situations. In this paper, we design a Cross-Domain Transfer-Based Attack (CDTA), which works in the cross-domain scenario. In this setting, attackers have no information about the target model, such as its architecture and training data. Specifically, we propose a contrastive spectral training method to train a feature extractor on a source domain (e.g., ImageNet) and use it to craft adversarial examples on target domains (e.g., Oxford 102 Flower). Our method corrupts the semantic information of the benign image by scrambling the outputs of both the intermediate feature layers and the final layer of the feature extractor. We evaluate CDTA with 16 target deep models on four datasets with widely varying styles. The results confirm that, in terms of the attack success rate, our approach can consistently outperform the state-of-the-art baselines by an average of 11.45% across all target models. Our code is available at https://github.com/LiulietLee/CDTA.

----

## [170] HybridCap: Inertia-Aid Monocular Capture of Challenging Human Motions

**Authors**: *Han Liang, Yannan He, Chengfeng Zhao, Mutian Li, Jingya Wang, Jingyi Yu, Lan Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25240](https://doi.org/10.1609/aaai.v37i2.25240)

**Abstract**:

Monocular 3D motion capture (mocap) is beneficial to many applications. The use of a single camera, however, often fails to handle occlusions of different body parts and hence it is limited to capture relatively simple movements. We present a light-weight, hybrid mocap technique called HybridCap that augments the camera with only 4 Inertial Measurement Units (IMUs) in a novel learning-and-optimization framework. We first employ a weakly-supervised and hierarchical motion inference module based on cooperative pure residual recurrent blocks that serve as limb, body and root trackers as well as an inverse kinematics solver. Our network effectively narrows the search space of plausible motions via coarse-to-fine pose estimation and manages to tackle challenging movements with high efficiency. We further develop a hybrid optimization scheme that combines inertial feedback and visual cues to improve tracking accuracy. Extensive experiments on various datasets demonstrate HybridCap can robustly handle challenging movements ranging from fitness actions to Latin dance. It also achieves real-time performance up to 60 fps with state-of-the-art accuracy.

----

## [171] Global Dilated Attention and Target Focusing Network for Robust Tracking

**Authors**: *Yun Liang, Qiaoqiao Li, Fumian Long*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25241](https://doi.org/10.1609/aaai.v37i2.25241)

**Abstract**:

Self Attention has shown the excellent performance in tracking due to its global modeling capability. However, it brings two challenges: First, its global receptive field has less attention on local structure and inter-channel associations, which limits the semantics to distinguish objects and backgrounds; Second, its feature fusion with linear process cannot avoid the interference of non-target semantic objects. To solve the above issues, this paper proposes a robust tracking method named GdaTFT by defining the Global Dilated Attention (GDA) and Target Focusing Network (TFN). The GDA provides a new global semantics modeling approach to enhance the semantic objects while eliminating the background. It is defined via the local focusing module, dilated attention and channel adaption module. Thus, it promotes semantics by focusing local key information, building long-range dependencies and enhancing the semantics of channels. Subsequently, to distinguish the target and non-target objects both with rich semantics, the TFN is proposed to accurately focus the target region. Different from the present feature fusion, it uses the template as the query to build a point-to-point correlation between the template and search region, and finally achieves part-level augmentation of target feature in the search region. Thus, the TFN efficiently augments the target embedding while weakening the non-target objects. Experiments on challenging benchmarks (LaSOT, TrackingNet, GOT-10k, OTB-100) demonstrate that the GdaTFT outperforms many state-of-the-art trackers and achieves leading performance. Code will be available.

----

## [172] Only a Few Classes Confusing: Pixel-Wise Candidate Labels Disambiguation for Foggy Scene Understanding

**Authors**: *Liang Liao, Wenyi Chen, Zhen Zhang, Jing Xiao, Yan Yang, Chia-Wen Lin, Shin'ichi Satoh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25242](https://doi.org/10.1609/aaai.v37i2.25242)

**Abstract**:

Not all semantics become confusing when deploying a semantic segmentation model for real-world scene understanding of adverse weather. The true semantics of most pixels have a high likelihood of appearing in the few top classes according to confidence ranking. In this paper, we replace the one-hot pseudo label with a candidate label set (CLS) that consists of only a few ambiguous classes and exploit its effects on self-training-based unsupervised domain adaptation. Specifically, we formulate the problem as a coarse-to-fine process. In the coarse-level process, adaptive CLS selection is proposed to pick a minimal set of confusing candidate labels based on the reliability of label predictions. Then, representation learning and label rectification are iteratively performed to facilitate feature clustering in an embedding space and to disambiguate the confusing semantics. Experimentally, our method outperforms the state-of-the-art methods on three realistic foggy benchmarks.

----

## [173] Actional Atomic-Concept Learning for Demystifying Vision-Language Navigation

**Authors**: *Bingqian Lin, Yi Zhu, Xiaodan Liang, Liang Lin, Jianzhuang Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25243](https://doi.org/10.1609/aaai.v37i2.25243)

**Abstract**:

Vision-Language Navigation (VLN) is a challenging task which requires an agent to align complex visual observations to language instructions to reach the goal position. Most existing VLN agents directly learn to align the raw directional features and visual features trained using one-hot labels to linguistic instruction features. However, the big semantic gap among these multi-modal inputs makes the alignment difficult and therefore limits the navigation performance. In this paper, we propose Actional Atomic-Concept Learning (AACL), which maps visual observations to actional atomic concepts for facilitating the alignment. Specifically, an actional atomic concept is a natural language phrase containing an atomic action and an object, e.g., ``go up stairs''. These actional atomic concepts, which serve as the bridge between observations and instructions, can effectively mitigate the semantic gap and simplify the alignment. AACL contains three core components: 1) a concept mapping module to map the observations to the actional atomic concept representations through the VLN environment and the recently proposed Contrastive Language-Image Pretraining (CLIP) model, 2) a concept refining adapter to encourage more instruction-oriented object concept extraction by re-ranking the predicted object concepts by CLIP, and 3) an observation co-embedding module which utilizes concept representations to regularize the observation representations. Our AACL establishes new state-of-the-art results on both fine-grained (R2R) and high-level (REVERIE and R2R-Last) VLN benchmarks. Moreover, the visualization shows that AACL significantly improves the interpretability in action decision. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/VLN-AACL.

----

## [174] Probability Guided Loss for Long-Tailed Multi-Label Image Classification

**Authors**: *Dekun Lin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25244](https://doi.org/10.1609/aaai.v37i2.25244)

**Abstract**:

Long-tailed learning has attracted increasing attention in very recent years. Long-tailed multi-label image classification is one subtask and remains challenging and poorly researched. In this paper, we provide a fresh perspective from probability to tackle this problem. More specifically, we find that existing cost-sensitive learning methods for long-tailed multi-label classification will affect the predicted probability of positive and negative labels in varying degrees during training, and different processes of probability will affect the final performance in turn. We thus propose a probability guided loss which contains two components to control this process. One is the probability re-balancing which can flexibly adjust the process of training probability. And the other is the adaptive probability-aware focal which can further reduce the probability gap between positive and negative labels. We conduct extensive experiments on two long-tailed multi-label image classification datasets: VOC-LT and COCO-LT. The results demonstrate the rationality and superiority of our strategy.

----

## [175] Self-Supervised Image Denoising Using Implicit Deep Denoiser Prior

**Authors**: *Huangxing Lin, Yihong Zhuang, Xinghao Ding, Delu Zeng, Yue Huang, Xiaotong Tu, John W. Paisley*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25245](https://doi.org/10.1609/aaai.v37i2.25245)

**Abstract**:

We devise a new regularization for denoising with self-supervised learning. The regularization uses a deep image prior learned by the network, rather than a traditional predefined prior. Specifically, we treat the output of the network as a ``prior'' that we again denoise after ``re-noising.'' The network is updated to minimize the discrepancy between the twice-denoised image and its prior. We demonstrate that this regularization enables the network to learn to denoise even if it has not seen any clean images. The effectiveness of our method is based on the fact that CNNs naturally tend to capture low-level image statistics. Since our method utilizes the image prior implicitly captured by the deep denoising CNN to guide denoising, we refer to this training strategy as an Implicit Deep Denoiser Prior (IDDP). IDDP can be seen as a mixture of learning-based methods and traditional model-based denoising methods, in which regularization is adaptively formulated using the output of the network. We apply IDDP to various denoising tasks using only observed corrupted data and show that it achieves better denoising results than other self-supervised denoising methods.

----

## [176] Accelerating the Training of Video Super-resolution Models

**Authors**: *Lijian Lin, Xintao Wang, Zhongang Qi, Ying Shan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25246](https://doi.org/10.1609/aaai.v37i2.25246)

**Abstract**:

Despite that convolution neural networks (CNN) have recently demonstrated high-quality reconstruction for video super-resolution (VSR), efficiently training competitive VSR models remains a challenging problem. It usually takes an order of magnitude more time than training their counterpart image models, leading to long research cycles. Existing VSR methods typically train models with fixed spatial and temporal sizes from beginning to end. The fixed sizes are usually set to large values for good performance, resulting to slow training. However, is such a rigid training strategy necessary for VSR? In this work, we show that it is possible to gradually train video models from small to large spatial/temporal sizes, \ie, in an easy-to-hard manner. In particular, the whole training is divided into several stages and the earlier stage has smaller training spatial shape. Inside each stage, the temporal size also varies from short to long while the spatial size remains unchanged. Training is accelerated by such a multigrid training strategy, as most of computation is performed on smaller spatial and shorter temporal shapes. For further acceleration with GPU parallelization, we also investigate the large minibatch training without the loss in accuracy. Extensive experiments demonstrate that our method is capable of largely speeding up training (up to $6.2\times$ speedup in wall-clock training time) without performance drop for various VSR models.

----

## [177] SelectAugment: Hierarchical Deterministic Sample Selection for Data Augmentation

**Authors**: *Shiqi Lin, Zhizheng Zhang, Xin Li, Zhibo Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25247](https://doi.org/10.1609/aaai.v37i2.25247)

**Abstract**:

Data augmentation (DA) has been extensively studied to facilitate model optimization in many tasks. Prior DA works focus on designing augmentation operations themselves, while leaving selecting suitable samples for augmentation out of consideration. This might incur visual ambiguities and further induce training biases. In this paper, we propose an effective approach, dubbed SelectAugment, to select samples for augmentation in a deterministic and online manner based on the sample contents and the network training status. To facilitate the policy learning, in each batch, we exploit the hierarchy of this task by first determining the augmentation ratio and then deciding whether to augment each training sample under this ratio. We model this process as two-step decision-making and adopt Hierarchical Reinforcement Learning (HRL) to learn the selection policy. In this way, the negative effects of the randomness in selecting samples to augment can be effectively alleviated and the effectiveness of DA is improved. Extensive experiments demonstrate that our proposed SelectAugment significantly improves various off-the-shelf DA methods on image classification and fine-grained image recognition.

----

## [178] AdaCM: Adaptive ColorMLP for Real-Time Universal Photo-Realistic Style Transfer

**Authors**: *Tianwei Lin, Honglin Lin, Fu Li, Dongliang He, Wenhao Wu, Meiling Wang, Xin Li, Yong Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25248](https://doi.org/10.1609/aaai.v37i2.25248)

**Abstract**:

Photo-realistic style transfer aims at migrating the artistic style from an exemplar style image to a content image, producing a result image without spatial distortions or unrealistic artifacts. Impressive results have been achieved by recent deep models. However, deep neural network based methods are too expensive to run in real-time. Meanwhile, bilateral grid based methods are much faster but still contain artifacts like overexposure. In this work, we propose the Adaptive ColorMLP (AdaCM), an effective and efficient framework for universal photo-realistic style transfer. First, we find the complex non-linear color mapping between input and target domain can be efficiently modeled by a small multi-layer perceptron (ColorMLP) model. Then, in AdaCM, we adopt a CNN encoder to adaptively predict all parameters for the ColorMLP conditioned on each input content and style image pair. Experimental results demonstrate that AdaCM can generate vivid and high-quality stylization results. Meanwhile, our AdaCM is ultrafast and can process a 4K resolution image in 6ms on one V100 GPU.

----

## [179] SEPT: Towards Scalable and Efficient Visual Pre-training

**Authors**: *Yiqi Lin, Huabin Zheng, Huaping Zhong, Jinjing Zhu, Weijia Li, Conghui He, Lin Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25249](https://doi.org/10.1609/aaai.v37i2.25249)

**Abstract**:

Recently, the self-supervised pre-training paradigm has shown great potential in leveraging large-scale unlabeled data to improve downstream task performance. However, increasing the scale of unlabeled pre-training data in real-world scenarios requires prohibitive computational costs and faces the challenge of uncurated samples. To address these issues, we build a task-specific self-supervised pre-training framework from a data selection perspective based on a simple hypothesis that pre-training on the unlabeled samples with similar distribution to the target task can bring substantial performance gains. Buttressed by the hypothesis, we propose the first yet novel framework for Scalable and Efficient visual Pre-Training (SEPT) by introducing a retrieval pipeline for data selection. SEPT first leverage a self-supervised pre-trained model to extract the features of the entire unlabeled dataset for retrieval pipeline initialization. Then, for a specific target task, SEPT retrievals the most similar samples from the unlabeled dataset based on feature similarity for each target instance for pre-training. Finally, SEPT pre-trains the target model with the selected unlabeled samples in a self-supervised manner for target data finetuning. By decoupling the scale of pre-training and available upstream data for a target task, SEPT achieves high scalability of the upstream dataset and high efficiency of pre-training, resulting in high model architecture flexibility. Results on various downstream tasks demonstrate that SEPT can achieve competitive or even better performance compared with ImageNet pre-training while reducing the size of training samples by one magnitude without resorting to any extra annotations.

----

## [180] Cross-Modality Earth Mover's Distance for Visible Thermal Person Re-identification

**Authors**: *Yongguo Ling, Zhun Zhong, Zhiming Luo, Fengxiang Yang, Donglin Cao, Yaojin Lin, Shaozi Li, Nicu Sebe*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25250](https://doi.org/10.1609/aaai.v37i2.25250)

**Abstract**:

Visible thermal person re-identification (VT-ReID) suffers from inter-modality discrepancy and intra-identity variations. Distribution alignment is a popular solution for VT-ReID, however, it is usually restricted to the influence of the intra-identity variations. In this paper, we propose the Cross-Modality Earth Mover's Distance (CM-EMD) that can alleviate the impact of the intra-identity variations during modality alignment. CM-EMD selects an optimal transport strategy and assigns high weights to pairs that have a smaller intra-identity variation. In this manner, the model will focus on reducing the inter-modality discrepancy while paying less attention to intra-identity variations, leading to a more effective modality alignment. Moreover, we introduce two techniques to improve the advantage of CM-EMD. First, Cross-Modality Discrimination Learning (CM-DL) is designed to overcome the discrimination degradation problem caused by modality alignment. By reducing the ratio between intra-identity and inter-identity variances, CM-DL leads the model to learn more discriminative representations. Second, we construct the Multi-Granularity Structure (MGS), enabling us to align modalities from both coarse- and fine-grained levels with the proposed CM-EMD. Extensive experiments show the benefits of the proposed CM-EMD and its auxiliary techniques (CM-DL and MGS). Our method achieves state-of-the-art performance on two VT-ReID benchmarks.

----

## [181] Hypotheses Tree Building for One-Shot Temporal Sentence Localization

**Authors**: *Daizong Liu, Xiang Fang, Pan Zhou, Xing Di, Weining Lu, Yu Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25251](https://doi.org/10.1609/aaai.v37i2.25251)

**Abstract**:

Given an untrimmed video, temporal sentence localization (TSL) aims to localize a specific segment according to a given sentence query. Though respectable works have made decent achievements in this task, they severely rely on dense video frame annotations, which require a tremendous amount of human effort to collect. In this paper, we target another more practical and challenging setting: one-shot temporal sentence localization (one-shot TSL), which learns to retrieve the query information among the entire video with only one annotated frame. Particularly, we propose an effective and novel tree-structure baseline for one-shot TSL, called Multiple Hypotheses Segment Tree (MHST), to capture the query-aware discriminative frame-wise information under the insufficient annotations. Each video frame is taken as the leaf-node, and the adjacent frames sharing the same visual-linguistic semantics will be merged into the upper non-leaf node for tree building. At last, each root node is an individual segment hypothesis containing the consecutive frames of its leaf-nodes. During the tree construction, we also introduce a pruning strategy to eliminate the interference of query-irrelevant nodes. With our designed self-supervised loss functions, our MHST is able to generate high-quality segment hypotheses for ranking and selection with the query. Experiments on two challenging datasets demonstrate that MHST achieves competitive performance compared to existing methods.

----

## [182] The Devil Is in the Frequency: Geminated Gestalt Autoencoder for Self-Supervised Visual Pre-training

**Authors**: *Hao Liu, Xinghua Jiang, Xin Li, Antai Guo, Yiqing Hu, Deqiang Jiang, Bo Ren*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25252](https://doi.org/10.1609/aaai.v37i2.25252)

**Abstract**:

The self-supervised Masked Image Modeling (MIM) schema, following "mask-and-reconstruct" pipeline of recovering contents from masked image, has recently captured the increasing interest in the community, owing to the excellent ability of learning visual representation from unlabeled data. Aiming at learning representations with high semantics abstracted, a group of works attempts to reconstruct non-semantic pixels with large-ratio masking strategy, which may suffer from "over-smoothing" problem, while others directly infuse semantics into targets in off-line way requiring extra data. Different from them, we shift the perspective to the Fourier domain which naturally has global perspective and present a new Masked Image Modeling (MIM), termed Geminated Gestalt Autoencoder (Ge^2-AE) for visual pre-training. Specifically, we equip our model with geminated decoders in charge of reconstructing image contents from both pixel and frequency space, where each other serves as not only the complementation but also the reciprocal constraints. Through this way, more robust representations can be learned in the pre-trained encoders, of which the effectiveness is confirmed by the juxtaposing experimental results on downstream recognition tasks. We also conduct several quantitative and qualitative experiments to investigate the learning behavior of our method. To our best knowledge, this is the first MIM work to solve the visual pre-training through the lens of frequency domain.

----

## [183] M3AE: Multimodal Representation Learning for Brain Tumor Segmentation with Missing Modalities

**Authors**: *Hong Liu, Dong Wei, Donghuan Lu, Jinghan Sun, Liansheng Wang, Yefeng Zheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25253](https://doi.org/10.1609/aaai.v37i2.25253)

**Abstract**:

Multimodal magnetic resonance imaging (MRI) provides complementary information for sub-region analysis of brain tumors. Plenty of methods have been proposed for automatic brain tumor segmentation using four common MRI modalities and achieved remarkable performance. In practice, however, it is common to have one or more modalities missing due to image corruption, artifacts, acquisition protocols, allergy to contrast agents, or simply cost. In this work, we propose a novel two-stage framework for brain tumor segmentation with missing modalities. In the first stage, a multimodal masked autoencoder (M3AE) is proposed, where both random modalities (i.e., modality dropout) and random patches of the remaining modalities are masked for a reconstruction task, for self-supervised learning of robust multimodal representations against missing modalities. To this end, we name our framework M3AE. Meanwhile, we employ model inversion to optimize a representative full-modal image at marginal extra cost, which will be used to substitute for the missing modalities and boost performance during inference. Then in the second stage, a memory-efficient self distillation is proposed to distill knowledge between heterogenous missing-modal situations while fine-tuning the model for supervised segmentation. Our M3AE belongs to the ‘catch-all’ genre where a single model can be applied to all possible subsets of modalities, thus is economic for both training and deployment. Extensive experiments on BraTS 2018 and 2020 datasets demonstrate its superior performance to existing state-of-the-art methods with missing modalities, as well as the efficacy of its components. Our code is available at: https://github.com/ccarliu/m3ae.

----

## [184] From Coarse to Fine: Hierarchical Pixel Integration for Lightweight Image Super-resolution

**Authors**: *Jie Liu, Chao Chen, Jie Tang, Gangshan Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25254](https://doi.org/10.1609/aaai.v37i2.25254)

**Abstract**:

Image super-resolution (SR) serves as a fundamental tool for the processing and transmission of multimedia data. Recently, Transformer-based models have achieved competitive performances in image SR. They divide images into fixed-size patches and apply self-attention on these patches to model long-range dependencies among pixels. However, this architecture design is originated for high-level vision tasks, which lacks design guideline from SR knowledge. In this paper, we aim to design a new attention block whose insights are from the interpretation of Local Attribution Map (LAM) for SR networks.  Specifically, LAM presents a hierarchical importance map where the most important pixels are located in a fine area of a patch and some less important pixels are spread in a coarse area of the whole image. To access pixels in the coarse area, instead of using a very large patch size, we propose a lightweight Global Pixel Access (GPA) module that applies cross-attention with the most similar patch in an image. In the fine area, we use an Intra-Patch Self-Attention (IPSA) module to model long-range pixel dependencies in a local patch, and then a spatial convolution is applied to process the finest details. In addition, a Cascaded Patch Division (CPD) strategy is proposed to enhance perceptual quality of recovered images. Extensive experiments suggest that our method outperforms state-of-the-art lightweight SR methods by a large margin. Code is available at https://github.com/passerer/HPINet.

----

## [185] Fast Fluid Simulation via Dynamic Multi-Scale Gridding

**Authors**: *Jinxian Liu, Ye Chen, Bingbing Ni, Wei Ren, Zhenbo Yu, Xiaoyang Huang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25255](https://doi.org/10.1609/aaai.v37i2.25255)

**Abstract**:

Recent works on learning-based frameworks for Lagrangian (i.e., particle-based) fluid simulation, though bypassing iterative pressure projection via efficient convolution operators, are still time-consuming due to excessive amount of particles. To address this challenge, we propose a dynamic multi-scale gridding method to reduce the magnitude of elements that have to be processed, by observing repeated particle motion patterns within certain consistent regions. Specifically, we hierarchically generate multi-scale micelles in Euclidean space by grouping particles that share similar motion patterns/characteristics based on super-light motion and scale estimation modules. With little internal motion variation, each micelle is modeled as a single rigid body with convolution only applied to a single representative particle. In addition, a distance-based interpolation is conducted to propagate relative motion message among micelles. With our efficient design, the network produces high visual fidelity fluid simulations with the inference time to be only 4.24 ms/frame (with 6K fluid particles), hence enables real-time human-computer interaction and animation. Experimental results on multiple datasets show that our work achieves great simulation acceleration with negligible prediction error increase.

----

## [186] TransLO: A Window-Based Masked Point Transformer Framework for Large-Scale LiDAR Odometry

**Authors**: *Jiuming Liu, Guangming Wang, Chaokang Jiang, Zhe Liu, Hesheng Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25256](https://doi.org/10.1609/aaai.v37i2.25256)

**Abstract**:

Recently, transformer architecture has gained great success in the computer vision community, such as image classification, object detection, etc. Nonetheless, its application for 3D vision remains to be explored, given that point cloud is inherently sparse, irregular, and unordered. Furthermore, existing point transformer frameworks usually feed raw point cloud of N×3 dimension into transformers, which limits the point processing scale because of their quadratic computational costs to the input size N. In this paper, we rethink the structure of point transformer. Instead of directly applying transformer to points, our network (TransLO) can process tens of thousands of points simultaneously by projecting points onto a 2D surface and then feeding them into a local transformer with linear complexity. Specifically, it is mainly composed of two components: Window-based Masked transformer with Self Attention (WMSA) to capture long-range dependencies; Masked Cross-Frame Attention (MCFA) to associate two frames and predict pose estimation. To deal with the sparsity issue of point cloud, we propose a binary mask to remove invalid and dynamic points. To our knowledge, this is the first transformer-based LiDAR odometry network. The experiment results on the KITTI odometry dataset show that our average rotation and translation RMSE achieves 0.500°/100m and 0.993% respectively. The performance of our network surpasses all recent learning-based methods and even outperforms LOAM on most evaluation sequences.Codes will be released on https://github.com/IRMVLab/TransLO.

----

## [187] Low-Light Video Enhancement with Synthetic Event Guidance

**Authors**: *Lin Liu, Junfeng An, Jianzhuang Liu, Shanxin Yuan, Xiangyu Chen, Wengang Zhou, Houqiang Li, Yanfeng Wang, Qi Tian*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25257](https://doi.org/10.1609/aaai.v37i2.25257)

**Abstract**:

Low-light video enhancement (LLVE) is an important yet challenging task with many applications such as photographing and autonomous driving. Unlike single image low-light enhancement, most LLVE methods utilize temporal information from adjacent frames to restore the color and remove the noise of the target frame. However, these algorithms, based on the framework of multi-frame alignment and enhancement, may produce multi-frame fusion artifacts when encountering extreme low light or fast motion. In this paper, inspired by the low latency and high dynamic range of events, we use synthetic events from multiple frames to guide the enhancement and restoration of low-light videos. Our method contains three stages: 1) event synthesis and enhancement, 2) event and image fusion, and 3) low-light enhancement. In this framework, we design two novel modules (event-image fusion transform and event-guided dual branch) for the second and third stages, respectively. Extensive experiments show that our method outperforms existing low-light video or single image enhancement approaches on both synthetic and real LLVE datasets. Our code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/LLVE-SEG.

----

## [188] Novel Motion Patterns Matter for Practical Skeleton-Based Action Recognition

**Authors**: *Mengyuan Liu, Fanyang Meng, Chen Chen, Songtao Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25258](https://doi.org/10.1609/aaai.v37i2.25258)

**Abstract**:

Most skeleton-based action recognition methods assume that the same type of action samples in the training set and the test set share similar motion patterns. However, action samples in real scenarios usually contain novel motion patterns which are not involved in the training set. As it is laborious to collect sufficient training samples to enumerate various types of novel motion patterns, this paper presents a practical skeleton-based action recognition task where the training set contains common motion patterns of action samples and the test set contains action samples that suffer from novel motion patterns. For this task, we present a Mask Graph Convolutional Network (Mask-GCN) to focus on learning action-specific skeleton joints that mainly convey action information meanwhile masking action-agnostic skeleton joints that convey rare action information and suffer more from novel motion patterns. Specifically, we design a policy network to learn layer-wise body masks to construct masked adjacency matrices, which guide a GCN-based backbone to learn stable yet informative action features from dynamic graph structure. Extensive experiments on our newly collected dataset verify that Mask-GCN outperforms most GCN-based methods when testing with various novel motion patterns.

----

## [189] EMEF: Ensemble Multi-Exposure Image Fusion

**Authors**: *Renshuai Liu, Chengyang Li, Haitao Cao, Yinglin Zheng, Ming Zeng, Xuan Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25259](https://doi.org/10.1609/aaai.v37i2.25259)

**Abstract**:

Although remarkable progress has been made in recent years, current multi-exposure image fusion (MEF) research is still bounded by the lack of real ground truth, objective evaluation function, and robust fusion strategy. In this paper, we study the MEF problem from a new perspective. We don’t utilize any synthesized ground truth, design any loss function, or develop any fusion strategy. Our proposed method EMEF takes advantage of the wisdom of multiple imperfect MEF contributors including both conventional and deep learning-based methods. Specifically, EMEF consists of two main stages: pre-train an imitator network and tune the imitator in the runtime. In the first stage, we make a unified network imitate different MEF targets in a style modulation way. In the second stage, we tune the imitator network by optimizing the style code, in order to find an optimal fusion result for each input pair. In the experiment, we construct EMEF from four state-of-the-art MEF methods and then make comparisons with the individuals and several other competitive methods on the latest released MEF benchmark dataset. The promising experimental results demonstrate that our ensemble framework can “get the best of all worlds”. The code is available at https://github.com/medalwill/EMEF.

----

## [190] Reducing Domain Gap in Frequency and Spatial Domain for Cross-Modality Domain Adaptation on Medical Image Segmentation

**Authors**: *Shaolei Liu, Siqi Yin, Linhao Qu, Manning Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25260](https://doi.org/10.1609/aaai.v37i2.25260)

**Abstract**:

Unsupervised domain adaptation (UDA) aims to learn a model trained on source domain and performs well on unlabeled target domain. In medical image segmentation field, most existing UDA methods depend on adversarial learning to address the domain gap between different image modalities, which is ineffective due to its complicated training process. In this paper, we propose a simple yet effective UDA method based on frequency and spatial domain transfer under multi-teacher distillation framework. In the frequency domain, we first introduce non-subsampled contourlet transform for identifying domain-invariant and domain-variant frequency components (DIFs and DVFs), and then keep the DIFs unchanged while replacing the DVFs of the source domain images with that of the target domain images to narrow the domain gap. In the spatial domain, we propose a batch momentum update-based histogram matching strategy to reduce the domain-variant image style bias. Experiments on two commonly used cross-modality medical image segmentation datasets show that our proposed method achieves superior performance compared to state-of-the-art methods.

----

## [191] DQ-DETR: Dual Query Detection Transformer for Phrase Extraction and Grounding

**Authors**: *Shilong Liu, Shijia Huang, Feng Li, Hao Zhang, Yaoyuan Liang, Hang Su, Jun Zhu, Lei Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25261](https://doi.org/10.1609/aaai.v37i2.25261)

**Abstract**:

In this paper, we study the problem of visual grounding by considering both phrase extraction and grounding (PEG). In contrast to the previous phrase-known-at-test setting, PEG requires a model to extract phrases from text and locate objects from image simultaneously, which is a more practical setting in real applications. As phrase extraction can be regarded as a 1D text segmentation problem, we formulate PEG as a dual detection problem and propose a novel DQ-DETR model, which introduces dual queries to probe different features from image and text for object prediction and phrase mask prediction. Each pair of dual queries are designed to have shared positional parts but different content parts. Such a design effectively alleviates the difficulty of modality alignment between image and text (in contrast to a single query design) and empowers Transformer decoder to leverage phrase mask-guided attention to improve the performance. To evaluate the performance of PEG, we also propose a new metric CMAP (cross-modal average precision), analogous to the AP metric in object detection. The new metric overcomes the ambiguity of Recall@1 in many-box-to-one-phrase cases in phrase grounding. As a result, our PEG pre-trained DQ-DETR establishes new state-of-the-art results on all visual grounding benchmarks with a ResNet-101 backbone. For example, it achieves 91.04% and 83.51% in terms of recall rate on RefCOCO testA and testB with a ResNet-101 backbone.

----

## [192] Progressive Neighborhood Aggregation for Semantic Segmentation Refinement

**Authors**: *Ting Liu, Yunchao Wei, Yanning Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25262](https://doi.org/10.1609/aaai.v37i2.25262)

**Abstract**:

Multi-scale features from backbone networks have been widely applied to recover object details in segmentation tasks. Generally, the multi-level features are fused in a certain manner for further pixel-level dense prediction. Whereas, the spatial structure information is not fully explored, that is similar nearby pixels can be used to complement each other. In this paper, we investigate a progressive neighborhood aggregation (PNA) framework to refine the semantic segmentation prediction, resulting in an end-to-end solution that can perform the coarse prediction and refinement in a unified network. Specifically, we first present a neighborhood aggregation module, the neighborhood similarity matrices for each pixel are estimated on multi-scale features, which are further used to progressively aggregate the high-level feature for recovering the spatial structure. In addition, to further integrate the high-resolution details into the aggregated feature, we apply a self-aggregation module on the low-level features to emphasize important semantic information for complementing losing spatial details. Extensive experiments on five segmentation datasets, including Pascal VOC 2012, CityScapes, COCO-Stuff 10k, DeepGlobe, and Trans10k, demonstrate that the proposed framework can be cascaded into existing segmentation models providing consistent improvements. In particular, our method achieves new state-of-the-art performances on two challenging datasets, DeepGlobe and Trans10k. The code is available at https://github.com/liutinglt/PNA.

----

## [193] CoordFill: Efficient High-Resolution Image Inpainting via Parameterized Coordinate Querying

**Authors**: *Weihuang Liu, Xiaodong Cun, Chi-Man Pun, Menghan Xia, Yong Zhang, Jue Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25263](https://doi.org/10.1609/aaai.v37i2.25263)

**Abstract**:

Image inpainting aims to fill the missing hole of the input. It is hard to solve this task efficiently when facing high-resolution images due to two reasons: (1) Large reception field needs to be handled for high-resolution image inpainting. (2) The general encoder and decoder network synthesizes many background pixels synchronously due to the form of the image matrix. In this paper, we try to break the above limitations for the first time thanks to the recent development of continuous implicit representation. In detail, we down-sample and encode the degraded image to produce the spatial-adaptive parameters for each spatial patch via an attentional Fast Fourier Convolution (FFC)-based parameter generation network. Then, we take these parameters as the weights and biases of a series of multi-layer perceptron (MLP), where the input is the encoded continuous coordinates and the output is the synthesized color value. Thanks to the proposed structure, we only encode the high-resolution image in a relatively low resolution for larger reception field capturing. Then, the continuous position encoding will be helpful to synthesize the photo-realistic high-frequency textures by re-sampling the coordinate in a higher resolution. Also, our framework enables us to query the coordinates of missing pixels only in parallel, yielding a more efficient solution than the previous methods. Experiments show that the proposed method achieves real-time performance on the 2048X2048 images using a single GTX 2080 Ti GPU and can handle 4096X4096 images, with much better performance than existing state-of-the-art methods visually and numerically. The code is available at: https://github.com/NiFangBaAGe/CoordFill.

----

## [194] CCQ: Cross-Class Query Network for Partially Labeled Organ Segmentation

**Authors**: *Xuyang Liu, Bingbing Wen, Sibei Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25264](https://doi.org/10.1609/aaai.v37i2.25264)

**Abstract**:

Learning multi-organ segmentation from multiple partially-labeled datasets attracts increasing attention. It can be a promising solution for the scarcity of large-scale, fully labeled 3D medical image segmentation datasets. However, existing algorithms of multi-organ segmentation on partially-labeled datasets neglect the semantic relations and anatomical priors between different categories of organs, which is crucial for partially-labeled multi-organ segmentation. In this paper, we tackle the limitations above by proposing the Cross-Class Query Network (CCQ). CCQ consists of an image encoder, a cross-class query learning module, and an attentive refinement segmentation module. More specifically, the image encoder captures the long-range dependency of a single image via the transformer encoder. Cross-class query learning module first generates query vectors that represent semantic concepts of different categories and then utilizes these query vectors to find the class-relevant features of image representation for segmentation. The attentive refinement segmentation module with an attentive skip connection incorporates the high-resolution image details and eliminates the class-irrelevant noise. Extensive experiment results demonstrate that CCQ outperforms all the state-of-the-art models on the MOTS dataset, which consists of seven organ and tumor segmentation tasks. Code is available at https://github.com/Yang-007/CCQ.git.

----

## [195] Counterfactual Dynamics Forecasting - a New Setting of Quantitative Reasoning

**Authors**: *Yanzhu Liu, Ying Sun, Joo-Hwee Lim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25265](https://doi.org/10.1609/aaai.v37i2.25265)

**Abstract**:

Rethinking and introspection are important elements of human intelligence. To mimic these capabilities, counterfactual reasoning has attracted attention of AI researchers recently, which aims to forecast the alternative outcomes for hypothetical scenarios (“what-if”). However, most existing approaches focused on qualitative reasoning (e.g., casual-effect relationship). It lacks a well-defined description of the differences between counterfactuals and facts, as well as how these differences evolve over time. This paper defines a new problem formulation - counterfactual dynamics forecasting - which is described in middle-level abstraction under the structural causal models (SCM) framework and derived as ordinary differential equations (ODEs) as low-level quantitative computation. Based on it, we propose a method to infer counterfactual dynamics considering the factual dynamics as demonstration. Moreover, the evolution of differences between facts and counterfactuals are modelled by an explicit temporal component. The experimental results on two dynamical systems demonstrate the effectiveness of the proposed method.

----

## [196] Self-Decoupling and Ensemble Distillation for Efficient Segmentation

**Authors**: *Yuang Liu, Wei Zhang, Jun Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25266](https://doi.org/10.1609/aaai.v37i2.25266)

**Abstract**:

Knowledge distillation (KD) is a promising teacher-student learning paradigm that transfers information from a cumbersome teacher to a student network. To avoid the training cost of a large teacher network, the recent studies propose to distill knowledge from the student itself, called Self-KD. However, due to the limitations of the performance and capacity of the student, the soft-labels or features distilled by the student barely provide reliable guidance. Moreover, most of the Self-KD algorithms are specific to classification tasks based on soft-labels, and not suitable for semantic segmentation. To alleviate these contradictions, we revisit the label and feature distillation problem in segmentation, and propose Self-Decoupling and Ensemble Distillation for Efficient Segmentation (SDES). Specifically, we design a decoupled prediction ensemble distillation (DPED) algorithm that generates reliable soft-labels with multiple expert decoders, and a decoupled feature ensemble distillation (DFED) mechanism to utilize more important channel-wise feature maps for encoder learning. The extensive experiments on three public segmentation datasets demonstrate the superiority of our approach and the efficacy of each component in the framework through the ablation study.

----

## [197] Token Mixing: Parameter-Efficient Transfer Learning from Image-Language to Video-Language

**Authors**: *Yuqi Liu, Luhui Xu, Pengfei Xiong, Qin Jin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25267](https://doi.org/10.1609/aaai.v37i2.25267)

**Abstract**:

Applying large scale pre-trained image-language model to video-language tasks has recently become a trend, which brings two challenges. One is how to effectively transfer knowledge from static images to dynamic videos, and the other is how to deal with the prohibitive cost of fully fine-tuning due to growing model size. Existing works that attempt to realize parameter-efficient image-language to video-language transfer learning can be categorized into two types: 1) appending a sequence of temporal transformer blocks after the 2D Vision Transformer (ViT), and 2) inserting a temporal block into the ViT architecture. While these two types of methods only require fine-tuning the newly added components, there are still many parameters to update, and they are only validated on a single video-language task. In this work, based on our analysis of the core ideas of different temporal modeling components in existing approaches, we propose a token mixing strategy to enable cross-frame interactions, which enables transferring from the pre-trained image-language model to video-language tasks through selecting and mixing a key set and a value set from the input video samples. As token mixing does not require the addition of any components or modules, we can directly partially fine-tune the pre-trained image-language model to achieve parameter-efficiency. We carry out extensive experiments to compare our proposed token mixing method with other parameter-efficient transfer learning methods. Our token mixing method outperforms other methods on both understanding tasks and generation tasks. Besides, our method achieves new records on multiple video-language tasks. The code is available at https://github.com/yuqi657/video_language_model.

----

## [198] StereoDistill: Pick the Cream from LiDAR for Distilling Stereo-Based 3D Object Detection

**Authors**: *Zhe Liu, Xiaoqing Ye, Xiao Tan, Errui Ding, Xiang Bai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25268](https://doi.org/10.1609/aaai.v37i2.25268)

**Abstract**:

In this paper, we propose a cross-modal distillation method named StereoDistill to narrow the gap between the stereo and LiDAR-based approaches via distilling the stereo detectors from the superior LiDAR model at the response level, which is usually overlooked in 3D object detection distillation.
The key designs of StereoDistill are: the X-component Guided Distillation~(XGD) for regression and the Cross-anchor Logit Distillation~(CLD) for classification. In XGD, instead of empirically adopting a threshold to select the high-quality teacher predictions as soft targets, we decompose the predicted 3D box into 
sub-components and retain the corresponding part for distillation if the teacher component pilot is consistent with ground truth to largely boost the number of positive predictions and alleviate the mimicking difficulty of the student model. For CLD, we aggregate the probability distribution of all anchors at the same position  to encourage the highest probability anchor rather than individually distill the distribution at the anchor level. 
Finally, our StereoDistill achieves state-of-the-art results for stereo-based 3D detection on the KITTI test benchmark and extensive experiments on KITTI and Argoverse Dataset validate the effectiveness.

----

## [199] Good Helper Is around You: Attention-Driven Masked Image Modeling

**Authors**: *Zhengqi Liu, Jie Gui, Hao Luo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i2.25269](https://doi.org/10.1609/aaai.v37i2.25269)

**Abstract**:

It has been witnessed that masked image modeling (MIM) has shown a huge potential in self-supervised learning in the past year. Benefiting from the universal backbone vision transformer, MIM learns self-supervised visual representations through masking a part of patches of the image while attempting to recover the missing pixels. Most previous works mask patches of the image randomly, which underutilizes the semantic information that is beneficial to visual representation learning. On the other hand, due to the large size of the backbone, most previous works have to spend much time on pre-training. In this paper, we propose Attention-driven Masking and Throwing Strategy (AMT), which could solve both problems above. We first leverage the self-attention mechanism to obtain the semantic information of the image during the training process automatically without using any supervised methods. Masking strategy can be guided by that information to mask areas selectively, which is helpful for representation learning. Moreover, a redundant patch throwing strategy is proposed, which makes learning more efficient. As a plug-and-play module for masked image modeling, AMT improves the linear probing accuracy of MAE by 2.9% ~ 5.9% on CIFAR-10/100, STL-10, Tiny ImageNet, and ImageNet-1K, and obtains an improved performance with respect to fine-tuning accuracy of MAE and SimMIM. Moreover, this design also achieves superior performance on downstream detection and segmentation tasks.

----



[Go to the next page](AAAI-2023-list02.md)

[Go to the catalog section](README.md)