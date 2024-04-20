## [2200] MathAttack: Attacking Large Language Models towards Math Solving Ability

**Authors**: *Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29949](https://doi.org/10.1609/aaai.v38i17.29949)

**Abstract**:

With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the robustness of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of robustness in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. The code and dataset is available at: https://github.com/zhouzihao501/MathAttack.

----

## [2201] LimeAttack: Local Explainable Method for Textual Hard-Label Adversarial Attack

**Authors**: *Hai Zhu, Qingyang Zhao, Weiwei Shang, Yuren Wu, Kai Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29950](https://doi.org/10.1609/aaai.v38i17.29950)

**Abstract**:

Natural language processing models are vulnerable to adversarial examples. Previous textual adversarial attacks adopt model internal information (gradients or  confidence scores) to generate adversarial examples. However, this information is unavailable in the real world. Therefore, we focus on a more realistic and challenging setting, named hard-label attack, in which the attacker can only query the model and obtain a discrete prediction label. Existing hard-label attack algorithms tend to initialize adversarial examples by random substitution and then utilize complex heuristic algorithms to optimize the adversarial perturbation. These methods require a lot of  model queries and the attack success rate is restricted by adversary initialization. In this paper,  we propose a novel hard-label attack algorithm named LimeAttack, which leverages a local explainable method to approximate word importance ranking, and then adopts beam search to find the optimal solution. Extensive experiments show that LimeAttack achieves the better attacking performance compared with existing hard-label attack under the same query budget. In addition, we evaluate the effectiveness of LimeAttack on large language models and some defense methods, and results indicate that adversarial examples remain a significant threat to  large language models. The adversarial examples crafted by LimeAttack  are highly transferable and  effectively improve model robustness in adversarial training.

----

## [2202] Multichannel AV-wav2vec2: A Framework for Learning Multichannel Multi-Modal Speech Representation

**Authors**: *Qiushi Zhu, Jie Zhang, Yu Gu, Yuchen Hu, Lirong Dai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29951](https://doi.org/10.1609/aaai.v38i17.29951)

**Abstract**:

Self-supervised speech pre-training methods have developed rapidly in recent years, which show to be very effective for many near-field single-channel speech tasks. However, far-field multichannel speech processing is suffering from the scarcity of labeled multichannel data and complex ambient noises. The efficacy of self-supervised learning for far-field multichannel and multi-modal speech processing has not been well explored. Considering that visual information helps to improve speech recognition performance in noisy scenes, in this work we propose the multichannel multi-modal speech self-supervised learning framework AV-wav2vec2, which utilizes video and multichannel audio data as inputs. First, we propose a multi-path structure to process multi-channel audio streams and a visual stream in parallel, with intra-, and inter-channel contrastive as training targets to fully exploit the rich information in multi-channel speech data. Second, based on contrastive learning, we use additional single-channel audio data, which is trained jointly to improve the performance of multichannel multi-modal representation. Finally, we use a Chinese multichannel multi-modal dataset in real scenarios to validate the effectiveness of the proposed method on audio-visual speech recognition (AVSR), automatic speech recognition (ASR), visual speech recognition (VSR) and audio-visual speaker diarization (AVSD) tasks.

----

## [2203] Aligner²: Enhancing Joint Multiple Intent Detection and Slot Filling via Adjustive and Forced Cross-Task Alignment

**Authors**: *Zhihong Zhu, Xuxin Cheng, Yaowei Li, Hongxiang Li, Yuexian Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29952](https://doi.org/10.1609/aaai.v38i17.29952)

**Abstract**:

Multi-intent spoken language understanding (SLU) has garnered growing attention due to its ability to handle multiple intent utterances, which closely mirrors practical scenarios. Unlike traditional SLU,  each intent in multi-intent SLU corresponds to its designated scope for slots,  which occurs in certain fragments within the utterance. As a result, establishing precise scope alignment to mitigate noise impact emerges as a key challenge in multi-intent SLU. More seriously, they lack alignment between the predictions of the two sub-tasks due to task-independent decoding, resulting in a limitation on the overall performance.  To address these challenges, we propose a novel framework termed Aligner² for multi-intent SLU, which contains an Adjustive Cross-task Aligner (ACA) and a Forced Cross-task Aligner (FCA). ACA utilizes the information conveyed by joint label embeddings to accurately align the scope of intent and corresponding slots, before the interaction of the two subtasks. FCA introduces reinforcement learning, to enforce the alignment of the task-specific hidden states after the interaction, which is explicitly guided by the prediction. Extensive experiments on two public multi-intent SLU datasets demonstrate the superiority of our  Aligner² over state-of-the-art methods. More encouragingly, the proposed method  Aligner² can be easily integrated into existing multi-intent SLU frameworks, to further boost performance.

----

## [2204] Towards Explainable Joint Models via Information Theory for Multiple Intent Detection and Slot Filling

**Authors**: *Xianwei Zhuang, Xuxin Cheng, Yuexian Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29953](https://doi.org/10.1609/aaai.v38i17.29953)

**Abstract**:

Recent joint models for multi-intent detection and slot filling have obtained promising results through modeling the unidirectional or bidirectional guidance between intent and slot. However, existing works design joint models heuristically and lack some theoretical exploration, including (1) theoretical measurement of the joint-interaction quality; (2) explainability of design and optimization methods of joint models, which may limit the performance and efficiency of designs. In this paper, we mathematically define the cross-task information gain (CIG) to measure the quality of joint processes from an information-theoretic perspective and discover an implicit optimization of CIG in previous models. Based on this, we propose a novel multi-stage iterative framework with theoretical effectiveness, explainability, and convergence, which can explicitly optimize information for cross-task interactions. Further, we devise an information-based joint model (InfoJoint) that conforms to this theoretical framework to gradually reduce the cross-task propagation of erroneous semantics through CIG iterative maximization. Extensive experiment results on two public datasets show that InfoJoint outperforms the state-of-the-art models by a large margin.

----

## [2205] Video-Context Aligned Transformer for Video Question Answering

**Authors**: *Linlin Zong, Jiahui Wan, Xianchao Zhang, Xinyue Liu, Wenxin Liang, Bo Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i17.29954](https://doi.org/10.1609/aaai.v38i17.29954)

**Abstract**:

Video question answering involves understanding video content to generate accurate answers to questions. Recent studies have successfully modeled video features and achieved diverse multimodal interaction, yielding impressive outcomes. However, they have overlooked the fact that the video contains richer instances and events beyond the scope of the stated question. Extremely imbalanced alignment of information from both sides leads to significant instability in reasoning. To address this concern, we propose the Video-Context Aligned Transformer (V-CAT), which leverages the context to achieve semantic and content alignment between video and question. Specifically, the video and text are encoded into a shared semantic space initially. We apply contrastive learning to global video token and context token to enhance the semantic alignment. Then, the pooled context feature is utilized to obtain corresponding visual content. Finally, the answer is decoded by integrating the refined video and question features. We evaluate the effectiveness of V-CAT on MSVD-QA and MSRVTT-QA dataset, both achieving state-of-the-art performance. Extended experiments further analyze and demonstrate the effectiveness of each proposed module.

----

## [2206] Quality-Diversity Generative Sampling for Learning with Synthetic Data

**Authors**: *Allen Chang, Matthew C. Fontaine, Serena Booth, Maja J. Mataric, Stefanos Nikolaidis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29955](https://doi.org/10.1609/aaai.v38i18.29955)

**Abstract**:

Generative models can serve as surrogates for some real data sources by creating synthetic training datasets, but in doing so they may transfer biases to downstream tasks. We focus on protecting quality and diversity when generating synthetic training datasets. We propose quality-diversity generative sampling (QDGS), a framework for sampling data uniformly across a user-defined measure space, despite the data coming from a biased generator. QDGS is a model-agnostic framework that uses prompt guidance to optimize a quality objective across measures of diversity for synthetically generated data, without fine-tuning the generative model. Using balanced synthetic datasets generated by QDGS, we first debias classifiers trained on color-biased shape datasets as a proof-of-concept. By applying QDGS to facial data synthesis, we prompt for desired semantic concepts, such as skin tone and age, to create an intersectional dataset with a combined blend of visual features. Leveraging this balanced data for training classifiers improves fairness while maintaining accuracy on facial recognition benchmarks. Code available at: https://github.com/Cylumn/qd-generative-sampling.

----

## [2207] A Cross-View Hierarchical Graph Learning Hypernetwork for Skill Demand-Supply Joint Prediction

**Authors**: *Wenshuo Chao, Zhaopeng Qiu, Likang Wu, Zhuoning Guo, Zhi Zheng, Hengshu Zhu, Hao Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29956](https://doi.org/10.1609/aaai.v38i18.29956)

**Abstract**:

The rapidly changing landscape of technology and industries leads to dynamic skill requirements, making it crucial for employees and employers to anticipate such shifts to maintain a competitive edge in the labor market. Existing efforts in this area either relies on domain-expert knowledge or regarding the skill evolution as a simplified time series forecasting problem. However, both approaches overlook the sophisticated relationships among different skills and the inner-connection between skill demand and supply variations. 
In this paper, we propose a Cross-view Hierarchical Graph learning Hypernetwork (CHGH) framework for joint skill demand-supply prediction. Specifically, CHGH is an encoder-decoder network consisting of i) a cross-view graph encoder to capture the interconnection between skill demand and supply, ii) a hierarchical graph encoder to model the co-evolution of skills from a cluster-wise perspective, and iii) a conditional hyper-decoder to jointly predict demand and supply variations by incorporating historical demand-supply gaps. Extensive experiments on three real-world datasets demonstrate the superiority of the proposed framework compared to seven baselines and the effectiveness of the three modules.

----

## [2208] Conditional Backdoor Attack via JPEG Compression

**Authors**: *Qiuyu Duan, Zhongyun Hua, Qing Liao, Yushu Zhang, Leo Yu Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29957](https://doi.org/10.1609/aaai.v38i18.29957)

**Abstract**:

Deep neural network (DNN) models have been proven vulnerable to backdoor attacks. One trend of backdoor attacks is developing more invisible and dynamic triggers to make attacks stealthier. However, these invisible and dynamic triggers can be inadvertently mitigated by some widely used passive denoising operations, such as image compression, making the efforts under this trend questionable. Another trend is to exploit the full potential of backdoor attacks by proposing new triggering paradigms, such as hibernated or opportunistic backdoors. In line with these trends, our work investigates the first conditional backdoor attack, where the backdoor is activated by a specific condition rather than pre-defined triggers. Specifically, we take the JPEG compression as our condition and jointly optimize the compression operator and the target model's loss function, which can force the target model to accurately learn the JPEG compression behavior as the triggering condition. In this case, besides the conditional triggering feature, our attack is also stealthy and robust to denoising operations. Extensive experiments on the MNIST, GTSRB and CelebA verify our attack's effectiveness, stealthiness and resistance to existing backdoor defenses and denoising operations. As a new triggering paradigm, the conditional backdoor attack brings a new angle for assessing the vulnerability of DNN models, and conditioned over JPEG compression magnifies its threat due to the universal usage of JPEG.

----

## [2209] Complementary Knowledge Distillation for Robust and Privacy-Preserving Model Serving in Vertical Federated Learning

**Authors**: *Dashan Gao, Sheng Wan, Lixin Fan, Xin Yao, Qiang Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29958](https://doi.org/10.1609/aaai.v38i18.29958)

**Abstract**:

Vertical Federated Learning (VFL) enables an active party with labeled data to enhance model performance (utility) by collaborating with multiple passive parties that possess auxiliary features corresponding to the same sample identifiers (IDs). Model serving in VFL is vital for real-world, delay-sensitive applications, and it faces two major challenges: 1) robustness against arbitrarily-aligned data and stragglers; and 2) privacy protection, ensuring minimal label leakage to passive parties. Existing methods fail to transfer knowledge among parties to improve robustness in a privacy-preserving way. In this paper, we introduce a privacy-preserving knowledge transfer framework, Complementary Knowledge Distillation (CKD), designed to enhance the robustness and privacy of multi-party VFL systems. Specifically, we formulate a Complementary Label Coding (CLC) objective to encode only complementary label information of the active party's local model for passive parties to learn. Then, CKD selectively transfers the CLC-encoded complementary knowledge 1) from the passive parties to the active party, and 2) among the passive parties themselves. Experimental results on four real-world datasets demonstrate that CKD outperforms existing approaches in terms of robustness against arbitrarily-aligned data, while also minimizing label privacy leakage.

----

## [2210] Resource Democratization: Is Compute the Binding Constraint on AI Research?

**Authors**: *Rebecca Gelles, Veronica Kinoshita, Micah Musser, James Dunham*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29959](https://doi.org/10.1609/aaai.v38i18.29959)

**Abstract**:

Access to compute is widely viewed as a primary barrier to AI research progress. Compute resource stratification between academic and industry researchers is therefore a source of concern. Yet the experiences of researchers who might encounter resource constraints in their work have received no direct study. We addressed this gap by conducting a large survey of AI researchers that posed questions about project inputs, outcomes, and challenges. Contrary to popular narratives, responses from more than 500 participants revealed more concern about talent and data limitations than compute access. There were few differences between academic and industry researchers in this regard. The exception were researchers who already use large amounts of compute, and expressed a need for more. These findings suggest that interventions to subsidize compute without addressing the limitations on talent and data availability reported by our respondents might cause or exacerbate commonly cited resource inequalities, with unknown impact on the future of equitable research.

----

## [2211] How to Overcome Curse-of-Dimensionality for Out-of-Distribution Detection?

**Authors**: *Soumya Suvra Ghosal, Yiyou Sun, Yixuan Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29960](https://doi.org/10.1609/aaai.v38i18.29960)

**Abstract**:

Machine learning models deployed in the wild can be challenged by out-of-distribution (OOD) data from unknown classes. Recent advances in OOD detection rely on distance measures to distinguish samples that are relatively far away from the in-distribution (ID) data. Despite the promise, distance-based methods can suffer from the curse-of-dimensionality problem, which limits the efficacy in high dimensional feature space. To combat this problem, we propose a novel framework, Subspace Nearest Neighbor (SNN), for OOD detection. In training, our method regularizes the model and its feature representation by leveraging the most relevant subset of dimensions (i.e. subspace). The subspace learning yields highly distinguishable distance measures between ID and OOD data. We provide comprehensive experiments and ablations to validate the efficacy of SNN. Compared to the current best distance-based method, SNN reduces the average FPR95 by 15.96% on the CIFAR-100 benchmark.

----

## [2212] Exploiting Discrepancy in Feature Statistic for Out-of-Distribution Detection

**Authors**: *Xiaoyuan Guan, Jiankang Chen, Shenshen Bu, Yuren Zhou, Wei-Shi Zheng, Ruixuan Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29961](https://doi.org/10.1609/aaai.v38i18.29961)

**Abstract**:

Recent studies on out-of-distribution (OOD) detection focus on designing models or scoring functions that can effectively distinguish between unseen OOD data and in-distribution (ID) data. In this paper, we propose a simple yet novel ap-
proach to OOD detection by leveraging the phenomenon that the average of feature vector elements from convolutional neural network (CNN) is typically larger for ID data than for OOD data. Specifically, the average of feature vector elements is used as part of the scoring function to further separate OOD data from ID data. We also provide mathematical analysis to explain this phenomenon. Experimental evaluations demonstrate that, when combined with a strong baseline, our method can achieve state-of-the-art performance on several OOD detection benchmarks. Furthermore, our method can be easily integrated into various CNN architectures and requires less computation. Source code address: https://github.com/SYSU-MIA-GROUP/statistical_discrepancy_ood.

----

## [2213] Reward Penalties on Augmented States for Solving Richly Constrained RL Effectively

**Authors**: *Hao Jiang, Tien Mai, Pradeep Varakantham, Huy Hoang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29962](https://doi.org/10.1609/aaai.v38i18.29962)

**Abstract**:

Constrained Reinforcement Learning employs trajectory-based cost constraints (such as expected cost, Value at Risk, or Conditional VaR cost) to compute safe policies. The challenge lies in handling these constraints effectively while optimizing expected reward. Existing methods convert such trajectory-based constraints into local cost constraints, but they rely on cost estimates, leading to either aggressive or conservative solutions with regards to cost. We propose an unconstrained formulation that employs reward penalties over states augmented with costs to compute safe policies. Unlike standard primal-dual methods, our approach penalizes only infeasible trajectories through state augmentation. This ensures that increasing the penalty parameter always guarantees a feasible policy, a feature lacking in primal-dual methods. Our approach exhibits strong empirical performance and theoretical properties, offering a fresh paradigm for solving complex Constrained RL problems, including rich constraints like expected cost, Value at Risk, and Conditional Value at Risk. Our experimental results demonstrate superior performance compared to leading approaches across various constraint types on multiple benchmark problems.

----

## [2214] The Logic of Doxastic Strategies

**Authors**: *Junli Jiang, Pavel Naumov*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29963](https://doi.org/10.1609/aaai.v38i18.29963)

**Abstract**:

In many real-world situations, there is often not enough information to know that a certain strategy will succeed in achieving the goal, but there is a good reason to believe that it will. The paper introduces the term "doxastic" for such strategies.

The main technical contribution is a sound and complete logical system that describes the interplay between doxastic strategy and belief modalities.

----

## [2215] MERGE: Fast Private Text Generation

**Authors**: *Zi Liang, Pinghui Wang, Ruofei Zhang, Nuo Xu, Shuo Zhang, Lifeng Xing, Haitao Bai, Ziyang Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29964](https://doi.org/10.1609/aaai.v38i18.29964)

**Abstract**:

The drastic increase in language models' parameters has led to a new trend of deploying models in cloud servers, raising growing concerns about private inference for Transformer-based models. Existing two-party privacy-preserving techniques, however, only take into account natural language understanding (NLU) scenarios. Private inference in natural language generation (NLG), crucial for applications like translation and code completion, remains underexplored. In addition, previous privacy-preserving techniques suffer from convergence issues during model training and exhibit poor inference speed when used with NLG models due to the neglect of time-consuming operations in auto-regressive generations. To address these issues, we propose a fast private text generation framework for Transformer-based language models, namely MERGE. MERGE reuses the output hidden state as the word embedding to bypass the embedding computation and reorganize the linear operations in the Transformer module to accelerate the forward procedure. Extensive experiments show that MERGE achieves a 26.5x speedup to the vanilla encrypted model under the sequence length 512, and reduces 80% communication cost, with an up to 10x speedup to state-of-the-art approximated models.

----

## [2216] Does Few-Shot Learning Suffer from Backdoor Attacks?

**Authors**: *Xinwei Liu, Xiaojun Jia, Jindong Gu, Yuan Xun, Siyuan Liang, Xiaochun Cao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29965](https://doi.org/10.1609/aaai.v38i18.29965)

**Abstract**:

The field of few-shot learning (FSL) has shown promising results in scenarios where training data is limited, but its vulnerability to backdoor attacks remains largely unexplored. We first explore this topic by first evaluating the performance of the existing backdoor attack methods on few-shot learning scenarios. Unlike in standard supervised learning, existing backdoor attack methods failed to perform an effective attack in FSL due to two main issues. Firstly, the model tends to overfit to either benign features or trigger features, causing a tough trade-off between attack success rate and benign accuracy. Secondly, due to the small number of training samples, the dirty label or visible trigger in the support set can be easily detected by victims, which reduces the stealthiness of attacks. It seemed that FSL could survive from backdoor attacks.  However, in this paper, we propose the Few-shot Learning Backdoor Attack (FLBA) to show that FSL can still be vulnerable to backdoor attacks. Specifically, we first generate a trigger to maximize the gap between poisoned and benign features. It enables the model to learn both benign and trigger features, which solves the problem of overfitting. To make it more stealthy, we hide the trigger by optimizing two types of imperceptible perturbation, namely attractive and repulsive perturbation, instead of attaching the trigger directly. Once we obtain the perturbations, we can poison all samples in the benign support set into a hidden poisoned support set and fine-tune the model on it. Our method demonstrates a high Attack Success Rate (ASR) in FSL tasks with different few-shot learning paradigms while preserving clean accuracy and maintaining stealthiness. This study reveals that few-shot learning still suffers from backdoor attacks, and its security should be given attention.

----

## [2217] Towards Model Extraction Attacks in GAN-Based Image Translation via Domain Shift Mitigation

**Authors**: *Di Mi, Yanjun Zhang, Leo Yu Zhang, Shengshan Hu, Qi Zhong, Haizhuan Yuan, Shirui Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29966](https://doi.org/10.1609/aaai.v38i18.29966)

**Abstract**:

Model extraction attacks (MEAs) enable an attacker to replicate the functionality of a victim deep neural network (DNN) model by only querying its API service remotely, posing a severe threat to the security and integrity of pay-per-query DNN-based services. Although the majority of current research on MEAs has primarily concentrated on neural classifiers, there is a growing prevalence of image-to-image translation (I2IT) tasks in our everyday activities. However, techniques developed for MEA of DNN classifiers cannot be directly transferred to the case of I2IT, rendering the vulnerability of I2IT models to MEA attacks often underestimated. This paper unveils the threat of MEA in I2IT tasks from a new perspective. Diverging from the traditional approach of bridging the distribution gap between attacker queries and victim training samples, we opt to mitigate the effect caused by the different distributions, known as the domain shift. This is achieved by introducing a new regularization term that penalizes high-frequency noise, and seeking a flatter minimum to avoid overfitting to the shifted distribution. Extensive experiments on different image translation tasks, including image super-resolution and style transfer, are performed on different backbone victim models, and the new design consistently outperforms the baseline by a large margin across all metrics. A few real-life I2IT APIs are also verified to be extremely vulnerable to our attack, emphasizing the need for enhanced defenses and potentially revised API publishing policies.

----

## [2218] Towards the Robustness of Differentially Private Federated Learning

**Authors**: *Tao Qi, Huili Wang, Yongfeng Huang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29967](https://doi.org/10.1609/aaai.v38i18.29967)

**Abstract**:

Robustness and privacy protection are two important factors of trustworthy federated learning (FL). Existing FL works usually secure data privacy by perturbing local model gradients via the differential privacy (DP) technique, or defend against poisoning attacks by filtering the local gradients in the outlier of the gradient distribution before aggregation. However, these two issues are often addressed independently in existing works, and how to secure federated learning in both privacy and robustness still needs further exploration. In this paper, we unveil that although DP noisy perturbation can improve the learning robustness, DP-FL frameworks are not inherently robust and are vulnerable to a carefully-designed attack method. Furthermore, we reveal that it is challenging for existing robust FL methods to defend against attacks on DP-FL. This can be attributed to the fact that the local gradients of DP-FL are perturbed by random noise, and the selected central gradients inevitably incorporate a higher proportion of poisoned gradients compared to conventional FL. To address this problem, we further propose a new defense method for DP-FL (named Robust-DPFL), which can effectively distinguish poisoned and clean local gradients in DP-FL and robustly update the global model. Experiments on three benchmark datasets demonstrate that baseline methods cannot ensure task accuracy, data privacy, and robustness simultaneously, while Robust-DPFL can effectively enhance the privacy protection and robustness of federated learning meanwhile maintain the task performance.

----

## [2219] Responsibility in Extensive Form Games

**Authors**: *Qi Shi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29968](https://doi.org/10.1609/aaai.v38i18.29968)

**Abstract**:

Two different forms of responsibility, counterfactual and seeing-to-it, have been extensively discussed in philosophy and AI in the context of a single agent or multiple agents acting simultaneously. Although the generalisation of counterfactual responsibility to a setting where multiple agents act in some order is relatively straightforward, the same cannot be said about seeing-to-it responsibility. Two versions of seeing-to-it modality applicable to such settings have been proposed in the literature. Neither of them perfectly captures the intuition of responsibility. The paper proposes a definition of seeing-to-it responsibility for such settings that amalgamate the two modalities.

The paper shows that the newly proposed notion of responsibility and counterfactual responsibility are not definable through each other and studies the responsibility gap for these two forms of responsibility. It shows that although these two forms of responsibility are not enough to ascribe responsibility in each possible situation, this gap does not exist if higher-order responsibility is taken into account.

----

## [2220] Towards Fairness in Online Service with K Servers and Its Application on Fair Food Delivery

**Authors**: *Daman Deep Singh, Amit Kumar, Abhijnan Chakraborty*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29969](https://doi.org/10.1609/aaai.v38i18.29969)

**Abstract**:

The k-SERVER problem is one of the most prominent problems in online algorithms with several variants and extensions. However, simplifying assumptions like instantaneous server movements and zero service time has hitherto limited its applicability to real-world problems. In this paper, we introduce a realistic generalization of k-SERVER without such assumptions – the k-FOOD problem, where requests with source-destination locations and an associated pickup time window arrive in an online fashion, and each has to be served by exactly one of the available k servers. The k-FOOD problem offers the versatility to model a variety of real-world use cases such as food delivery, ride sharing, and quick commerce. Moreover, motivated by the need for fairness in online platforms, we introduce the FAIR k-FOOD problem with the max-min objective. We establish that both k-FOOD and FAIR k-FOOD problems are strongly NP-hard and develop an optimal offline algorithm that arises naturally from a time-expanded flow network. Subsequently, we propose an online algorithm DOC4FOOD involving virtual movements of servers to the nearest request location. Experiments on a real-world food-delivery dataset, alongside synthetic datasets, establish the efficacy of the proposed algorithm against state-of-the-art fair food delivery algorithms.

----

## [2221] Value Kaleidoscope: Engaging AI with Pluralistic Human Values, Rights, and Duties

**Authors**: *Taylor Sorensen, Liwei Jiang, Jena D. Hwang, Sydney Levine, Valentina Pyatkin, Peter West, Nouha Dziri, Ximing Lu, Kavel Rao, Chandra Bhagavatula, Maarten Sap, John Tasioulas, Yejin Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29970](https://doi.org/10.1609/aaai.v38i18.29970)

**Abstract**:

Human values are crucial to human decision-making. Value pluralism is the view that multiple correct values may be held in tension with one another (e.g., when considering lying to a friend to protect their feelings, how does one balance honesty with friendship?). As statistical learners, AI systems fit to averages by default, washing out these potentially irreducible value conflicts. To improve AI systems to better reflect value pluralism, the first-order challenge is to explore the extent to which AI systems can model pluralistic human values, rights, and duties as well as their interaction.

We introduce ValuePrism, a large-scale dataset of 218k values, rights, and duties connected to 31k human-written situations. ValuePrism’s contextualized values are generated by GPT-4 and deemed high-quality by human annotators 91% of the time. We conduct a large-scale study with annotators across diverse social and demographic backgrounds to try to understand whose values are represented.

With ValuePrism, we build Value Kaleidoscope (or Kaleido), an open, light-weight, and structured language-based multi-task model that generates, explains, and assesses the relevance and valence (i.e., support or oppose) of human values, rights, and duties within a specific context. Humans prefer the sets of values output by our system over the teacher GPT- 4, finding them more accurate and with broader coverage. In addition, we demonstrate that Kaleido can help explain variability in human decision-making by outputting contrasting values. Finally, we show that Kaleido’s representations transfer to other philosophical frameworks and datasets, confirming the benefit of an explicit, modular, and interpretable approach to value pluralism. We hope that our work will serve as a step to making more explicit the implicit values behind human decision-making and to steering AI systems to make decisions that are more in accordance with them.

----

## [2222] Moral Uncertainty and the Problem of Fanaticism

**Authors**: *Jazon Szabo, Natalia Criado, Jose Such, Sanjay Modgil*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29971](https://doi.org/10.1609/aaai.v38i18.29971)

**Abstract**:

While there is universal agreement that agents ought to act ethically, there is no agreement as to what constitutes ethical behaviour. To address this problem, recent philosophical approaches to `moral uncertainty' propose aggregation of multiple ethical theories to guide agent behaviour. However, one of the foundational proposals for aggregation - Maximising Expected Choiceworthiness (MEC) - has been criticised as being vulnerable to fanaticism; the problem of an ethical theory dominating agent behaviour despite low credence (confidence) in said theory. Fanaticism thus undermines the `democratic' motivation for accommodating multiple ethical perspectives. The problem of fanaticism has not yet been mathematically defined. Representing moral uncertainty as an instance of social welfare aggregation, this paper contributes to the field of moral uncertainty by 1) formalising the problem of fanaticism as a property of social welfare functionals and  2) providing non-fanatical alternatives to MEC, i.e. Highest k-trimmed Mean and Highest Median.

----

## [2223] U-trustworthy Models Reliability, Competence, and Confidence in Decision-Making

**Authors**: *Ritwik Vashistha, Arya Farahi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29972](https://doi.org/10.1609/aaai.v38i18.29972)

**Abstract**:

With growing concerns regarding bias and discrimination in predictive models, the AI community has increasingly focused on assessing AI system trustworthiness. Conventionally, trustworthy AI literature relies on the probabilistic framework and calibration as prerequisites for trustworthiness. In this work, we depart from this viewpoint by proposing a novel trust framework inspired by the philosophy literature on trust. We present a precise mathematical definition of trustworthiness, termed U-trustworthiness, specifically tailored for a subset of tasks aimed at maximizing a utility function. We argue that a model’s U-trustworthiness is contingent upon its ability to maximize Bayes utility within this task subset. Our first set of results challenges the probabilistic framework by demonstrating its potential to favor less trustworthy models and introduce the risk of misleading trustworthiness assessments. Within the context of U-trustworthiness, we prove that properly-ranked models are inherently U-trustworthy. Furthermore, we advocate for the adoption of the AUC metric as the preferred measure of trustworthiness. By offering both theoretical guarantees and experimental validation, AUC enables robust evaluation of trustworthiness, thereby enhancing model selection and hyperparameter tuning to yield more trustworthy outcomes.

----

## [2224] TraceEvader: Making DeepFakes More Untraceable via Evading the Forgery Model Attribution

**Authors**: *Mengjie Wu, Jingui Ma, Run Wang, Sidan Zhang, Ziyou Liang, Boheng Li, Chenhao Lin, Liming Fang, Lina Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29973](https://doi.org/10.1609/aaai.v38i18.29973)

**Abstract**:

In recent few years, DeepFakes are posing serve threats and concerns to both individuals and celebrities, as realistic DeepFakes facilitate the spread of disinformation. Model attribution techniques aim at attributing the adopted forgery models of DeepFakes for provenance purposes and providing explainable results to DeepFake forensics. However, the existing model attribution techniques rely on the trace left in the DeepFake creation, which can become futile if such traces were disrupted. Motivated by our observation that certain traces served for model attribution appeared in both the high-frequency and low-frequency domains and play a divergent role in model attribution. In this work, for the first time, we propose a novel training-free evasion attack, TraceEvader, in the most practical non-box setting.  Specifically, TraceEvader injects a universal imitated traces learned from wild DeepFakes into the high-frequency component and introduces adversarial blur into the domain of the low-frequency component, where the added distortion confuses the extraction of certain traces for model attribution. The comprehensive evaluation on 4 state-of-the-art (SOTA) model attribution techniques and fake images generated by 8 generative models including generative adversarial networks (GANs) and diffusion models (DMs) demonstrates the effectiveness of our method. Overall, our TraceEvader achieves the highest average attack success rate of 79% and is robust against image transformations and dedicated denoising techniques as well where the average attack success rate is still around 75%. Our TraceEvader confirms the limitations of current model attribution techniques and calls the attention of DeepFake researchers and practitioners for more robust-purpose model attribution techniques.

----

## [2225] SAME: Sample Reconstruction against Model Extraction Attacks

**Authors**: *Yi Xie, Jie Zhang, Shiqian Zhao, Tianwei Zhang, Xiaofeng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29974](https://doi.org/10.1609/aaai.v38i18.29974)

**Abstract**:

While deep learning models have shown significant performance across various domains, their deployment needs extensive resources and advanced computing infrastructure. As a solution, Machine Learning as a Service (MLaaS) has emerged, lowering the barriers for users to release or productize their deep learning models. However, previous studies have highlighted potential privacy and security concerns associated with MLaaS, and one primary threat is model extraction attacks. To address this, there are many defense solutions but they suffer from unrealistic assumptions and generalization issues, making them less practical for reliable protection. Driven by these limitations, we introduce a novel defense mechanism, SAME, based on the concept of sample reconstruction. This strategy imposes minimal prerequisites on the defender's capabilities, eliminating the need for auxiliary Out-of-Distribution (OOD) datasets, user query history, white-box model access, and additional intervention during model training. It is compatible with existing active defense methods. Our extensive experiments corroborate the superior efficacy of SAME over state-of-the-art solutions. Our code is available at https://github.com/xythink/SAME.

----

## [2226] High-Fidelity Gradient Inversion in Distributed Learning

**Authors**: *Zipeng Ye, Wenjian Luo, Qi Zhou, Yubo Tang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29975](https://doi.org/10.1609/aaai.v38i18.29975)

**Abstract**:

Distributed learning frameworks aim to train global models by sharing gradients among clients while preserving the data privacy of each individual client. However, extensive research has demonstrated that these learning frameworks do not absolutely ensure the privacy, as training data can be reconstructed from shared gradients. Nevertheless, the existing privacy-breaking attack methods have certain limitations. Some are applicable only to small models, while others can only recover images in small batch size and low resolutions, or with low fidelity. Furthermore, when there are some data with the same label in a training batch, existing attack methods usually perform poorly. In this work, we successfully address the limitations of existing attacks by two steps. Firstly, we model the coefficient of variation (CV) of features and design an evolutionary algorithm based on the minimum CV to accurately reconstruct the labels of all training data. After that, we propose a stepwise gradient inversion attack, which dynamically adapts the objective function, thereby effectively and rationally promoting the convergence of attack results towards an optimal solution. With these two steps, our method is able to recover high resolution images (224*224 pixel, from ImageNet and Web) with high fidelity in distributed learning scenarios involving complex models and larger batch size. Experiment results demonstrate the superiority of our approach, reveal the potential vulnerabilities of the distributed learning paradigm, and emphasize the necessity of developing more secure mechanisms. Source code is available at https://github.com/MiLab-HITSZ/2023YeHFGradInv.

----

## [2227] Robustness Verification of Deep Reinforcement Learning Based Control Systems Using Reward Martingales

**Authors**: *Dapeng Zhi, Peixin Wang, Cheng Chen, Min Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29976](https://doi.org/10.1609/aaai.v38i18.29976)

**Abstract**:

Deep Reinforcement Learning (DRL) has gained prominence as an effective approach for control systems. However, its practical deployment is impeded by state perturbations that can severely impact system performance. Addressing this critical challenge requires robustness verification about system performance, which involves tackling two quantitative questions: (i) how to establish guaranteed bounds for expected cumulative rewards, and (ii) how to determine tail bounds for cumulative rewards. In this work, we present the first approach for robustness verification of DRL-based control systems by introducing  reward martingales, which offer a rigorous mathematical foundation to characterize the impact of state perturbations on system performance in terms of cumulative rewards. Our verified results provide provably quantitative certificates for the two questions. We then show that reward martingales can be implemented and trained via neural networks, against different types of control policies. Experimental results demonstrate that our certified bounds tightly enclose simulation outcomes on various DRL-based control systems, indicating the effectiveness and generality of the proposed approach.

----

## [2228] Regulating AI: Applying Insights from Behavioural Economics and Psychology to the Application of Article 5 of the EU AI Act

**Authors**: *Huixin Zhong, Eamonn O'Neill, Janina A. Hoffmann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29977](https://doi.org/10.1609/aaai.v38i18.29977)

**Abstract**:

Article 5 of the European Union’s Artificial Intelligence Act is intended to regulate AI use to prevent potentially harmful consequences. Nevertheless, applying this legislation practically is likely to be challenging because of ambiguously used terminologies and because it fails to specify which manipulation techniques may be invoked by AI, potentially leading to significant harm. This paper aims to bridge this gap by defining key terms and demonstrating how AI may invoke these techniques, drawing from insights in psychology and behavioural economics. First, this paper provides definitions of the terms “subliminal techniques”, “manipulative techniques” and “deceptive techniques”. Secondly, we identified from the literature in cognitive psychology and behavioural economics three subliminal and five manipulative techniques and exemplify how AI might implement these techniques to manipulate users in real-world case scenarios. These illustrations may serve as a practical guide for stakeholders to detect cases of AI manipulation and consequently devise preventive measures. Article 5 has also been criticised for offering inadequate protection. We critically assess the protection offered by Article 5, proposing specific revisions to paragraph 1, points (a) and (b) of Article 5 to increase its protective effectiveness.

----

## [2229] Batch Normalization Is Blind to the First and Second Derivatives of the Loss

**Authors**: *Zhanpeng Zhou, Wen Shen, Huixin Chen, Ling Tang, Yuefeng Chen, Quanshi Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29978](https://doi.org/10.1609/aaai.v38i18.29978)

**Abstract**:

We prove that when we do the Taylor series expansion of the loss function, the BN operation will block the influence of the first-order term and most influence of the second-order term of the loss. We also find that such a problem is caused by the standardization phase of the BN operation. We believe that proving the blocking of certain loss terms provides an analytic perspective for potential detects of a deep model with BN operations, although the blocking problem is not fully equivalent to significant damages in all tasks on benchmark datasets. Experiments show that the BN operation significantly affects feature representations in specific tasks.

----

## [2230] Block-Level Goal Recognition Design

**Authors**: *Tsz-Chiu Au*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29979](https://doi.org/10.1609/aaai.v38i18.29979)

**Abstract**:

Existing works on goal recognition design (GRD) consider the underlying domain as a classical planning domain and apply modifications to the domain to minimize the worst case distinctiveness. In this paper, we propose replacing existing modifications with blocks, which group several closely related modifications together such that a block can modify a region in a search space with respect to some design constraints. Moreover, there could be blocks within blocks such that the design space becomes hierarchical for modifications at different levels of granularity. We present 1) a new version of pruned-reduce, a successful pruning rule for GRD, for block-level GRD, and 2) a new pruning rule for pruning some branches in both hierarchical and non-hierarchical design space. Our experiments show that searching in hierarchical design spaces greatly speeds up the redesign process.

----

## [2231] Learning Planning Domains from Non-redundant Fully-Observed Traces: Theoretical Foundations and Complexity Analysis

**Authors**: *Pascal Bachor, Gregor Behnke*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29980](https://doi.org/10.1609/aaai.v38i18.29980)

**Abstract**:

Domain learning is the task of finding an action model that can explain given observed plan executions, so-called traces.                                                                                   
It allows us to automate the identification of actions' preconditions and effects instead of relying on hand-modeled expert knowledge.                                                                      
While previous research has put forth various techniques and covers multiple planning formalisms, the theoretical foundations of domain learning are still in their infancy.
                                                                                                                                                                                                          
We investigate the most basic setting, that is grounded classical planning without negative preconditions or conditional effects with full observability of the state variables.                            
The given traces are assumed to be justified in the sense that either no single action or no set of actions can be removed without violating correctness of the plan.                                       
Furthermore, we might be given additional constraints in the form of a propositional logical formula.                                                                                                       
We show the consequences of these assumptions for the computational complexity of identifying a satisfactory planning domain.

----

## [2232] Dealing with Numeric and Metric Time Constraints in PDDL3 via Compilation to Numeric Planning

**Authors**: *Luigi Bonassi, Alfonso Emilio Gerevini, Enrico Scala*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29981](https://doi.org/10.1609/aaai.v38i18.29981)

**Abstract**:

This paper studies an approach to planning with PDDL3 constraints involving mixed propositional and numeric conditions, as well as metric time constraints.  
We show how the whole PDDL3 with instantaneous actions can be compiled away into a numeric planning problem without PDDL3 constraints, enabling the use of any state-of-the-art numeric planner that is agnostic to the existence of PDDL3. Our solution exploits the concept of regression. In addition to a basic compilation, we present an optimized variant based on the observation that it is possible to make the compilation sensitive to the structure of the problem to solve; this can be done by reasoning on the interactions between the problem actions and the constraints. The resulting optimization substantially reduces the size of the planning task. We experimentally observe that our approach significantly outperforms existing state-of-the-art planners supporting the same class of constraints over known benchmark domains, settling a new state-of-the-art planning system for PDDL3.

----

## [2233] The Complexity of Optimizing Atomic Congestion

**Authors**: *Cornelius Brand, Robert Ganian, Subrahmanyam Kalyanasundaram, Fionn Mc Inerney*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29982](https://doi.org/10.1609/aaai.v38i18.29982)

**Abstract**:

Atomic congestion games are a classic topic in network design, routing, and algorithmic game theory, and are capable of modeling congestion and flow optimization tasks in various application areas. While both the price of anarchy for such games as well as the computational complexity of computing their Nash equilibria are by now well-understood, the computational complexity of computing a system-optimal set of strategies - that is, a centrally planned routing that minimizes the average cost of agents - is severely understudied in the literature. We close this gap by identifying the exact boundaries of tractability for the problem through the lens of the parameterized complexity paradigm. After showing that the problem remains highly intractable even on extremely simple networks, we obtain a set of results which demonstrate that the structural parameters which control the computational (in)tractability of the problem are not vertex-separator based in nature (such as, e.g., treewidth), but rather based on edge separators. We conclude by extending our analysis towards the (even more challenging) min-max variant of the problem.

----

## [2234] Stop! Planner Time: Metareasoning for Probabilistic Planning Using Learned Performance Profiles

**Authors**: *Matthew Budd, Bruno Lacerda, Nick Hawes*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29983](https://doi.org/10.1609/aaai.v38i18.29983)

**Abstract**:

The metareasoning framework aims to enable autonomous agents to factor in planning costs when making decisions. In this work, we develop the first non-myopic metareasoning algorithm for planning with Markov decision processes. Our method learns the behaviour of anytime probabilistic planning algorithms from performance data. Specifically, we propose a novel model for metareasoning, based on contextual performance profiles that predict the value of the planner's current solution given the time spent planning, the state of the planning algorithm's internal parameters, and the difficulty of the planning problem being solved. This model removes the need to assume that the current solution quality is always known, broadening the class of metareasoning problems that can be addressed. We then employ deep reinforcement learning to learn a policy that decides, at each timestep, whether to continue planning or start executing the current plan, and how to set hyperparameters of the planner to enhance its performance. We demonstrate our algorithm's ability to perform effective metareasoning in two domains.

----

## [2235] Can LLMs Fix Issues with Reasoning Models? Towards More Likely Models for AI Planning

**Authors**: *Turgay Caglar, Sirine Belhaj, Tathagata Chakraborti, Michael Katz, Sarath Sreedharan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29984](https://doi.org/10.1609/aaai.v38i18.29984)

**Abstract**:

This is the first work to look at the application of large language models (LLMs) for the purpose of model space edits in automated planning tasks. To set the stage for this union, we explore two different flavors of model space problems that have been studied in the AI planning literature and explore the effect of an LLM on those tasks. We empirically demonstrate how the performance of an LLM contrasts with combinatorial search (CS) – an approach that has been traditionally used to solve model space tasks in planning, both with the LLM in the role of a standalone model space reasoner as well as in the role of a statistical signal in concert with the CS approach as part of a two-stage process. Our experiments show promising results suggesting further forays of LLMs into the exciting world of model space reasoning for planning tasks in the future.

----

## [2236] Symbolic Numeric Planning with Patterns

**Authors**: *Matteo Cardellini, Enrico Giunchiglia, Marco Maratea*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29985](https://doi.org/10.1609/aaai.v38i18.29985)

**Abstract**:

In this paper, we propose a novel  approach for solving linear numeric planning problems, called Symbolic Pattern Planning. Given a planning problem Pi, a bound n and a pattern --defined as an arbitrary sequence of actions-- we encode the problem of finding a plan for Pi with bound n as a formula with fewer variables and/or clauses than the state-of-the-art rolled-up and relaxed-relaxed-exists encodings. More importantly, we prove that for any given bound, it is never the case that the latter two encodings allow finding a valid plan while ours does not. On the experimental side, we consider 6 other planning systems  --including the ones which participated in this year's International Planning Competition (IPC)-- and we show that our planner Patty has remarkably good comparative performances on this year's IPC problems.

----

## [2237] Learning Domain-Independent Heuristics for Grounded and Lifted Planning

**Authors**: *Dillon Ze Chen, Sylvie Thiébaux, Felipe W. Trevizan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29986](https://doi.org/10.1609/aaai.v38i18.29986)

**Abstract**:

We present three novel graph representations of planning tasks suitable for learning domain-independent heuristics using Graph Neural Networks (GNNs) to guide search. In particular, to mitigate the issues caused by large grounded GNNs we present the first method for learning domain-independent heuristics with only the lifted representation of a planning task. We also provide a theoretical analysis of the expressiveness of our models, showing that some are more powerful than STRIPS-HGN, the only other existing model for learning domain-independent heuristics. Our experiments show that our heuristics generalise to much larger problems than those in the training set, vastly surpassing STRIPS-HGN heuristics.

----

## [2238] Approximate Distance Oracle for Fault-Tolerant Geometric Spanners

**Authors**: *Kyungjin Cho, Jihun Shin, Eunjin Oh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29987](https://doi.org/10.1609/aaai.v38i18.29987)

**Abstract**:

In this paper, we present approximate distance and shortest-path oracles for fault-tolerant Euclidean spanners motivated by the routing problem in real-world road networks.
A fault-tolerant Euclidean spanner for a set of points in Euclidean space is a graph
in which, despite the deletion of small number of any points, the distance between any two points in the damaged graph is an approximation of their Euclidean distance. 
Given a fault-tolerant Euclidean spanner and a small approximation factor, 
our data structure allows us to compute an approximate distance between two points in the damaged spanner in constant time when a query involves any two points and a small set of failed points. 
Additionally, by incorporating additional data structures, we can return a path itself in time almost linear in the length of the returned path. 
Both data structures require near-linear space.

----

## [2239] Optimizing the Optimization of Planning Domains by Automatic Action Schema Splitting

**Authors**: *Mojtaba Elahi, Jussi Rintanen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29988](https://doi.org/10.1609/aaai.v38i18.29988)

**Abstract**:

Most planners are based on grounding, that is, generating all instances of a parameterized action during a preprocessing phase.
For some problems the number of ground actions is too high, causing a performance bottleneck.
Building upon an existing approach, we present an enhanced method to split action schemas automatically during the grounding phase, to reduce the number of ground actions.
First, we propose to exploit the structural knowledge of the problems to have a more informative dependency graph.
Then, we suggest a better objective function to define and choose the best split.
Finally, we present a more effective search to find it.
We experimentally measure the impact of each of these improvements, and show that our approach significantly outperforms the state of the art.

----

## [2240] An Effective Polynomial Technique for Compiling Conditional Effects Away

**Authors**: *Alfonso Emilio Gerevini, Francesco Percassi, Enrico Scala*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29989](https://doi.org/10.1609/aaai.v38i18.29989)

**Abstract**:

The paper introduces a novel polynomial compilation technique for the sound and complete removal of conditional effects in classical planning problems. Similar to Nebel's polynomial compilation of conditional effects, our solution also decomposes each action with conditional effects into several simpler actions. However, it does so more effectively by exploiting the actual structure of the given conditional effects. We characterise such a structure using a directed graph and leverage it to significantly reduce the number of additional atoms required, thereby shortening the size of valid plans. Our experimental analysis indicates that this approach enables the effective use of polynomial compilations, offering benefits in terms of modularity and reusability of existing planners. It also demonstrates that a compilation-based approach can be more efficient, either independently or in synergy with state-of-the-art optimal planners that directly support conditional effects.

----

## [2241] GOALNET: Interleaving Neural Goal Predicate Inference with Classical Planning for Generalization in Robot Instruction Following

**Authors**: *Jigyasa Gupta, Shreya Sharma, Shreshth Tuli, Rohan Paul, Mausam*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29990](https://doi.org/10.1609/aaai.v38i18.29990)

**Abstract**:

Our goal is to enable a robot to learn how to sequence its actions to perform high-level tasks specified as natural language instructions, given successful demonstrations from a human partner. Our novel neuro-symbolic solution GOALNET builds an iterative two-step approach that interleaves (i) inferring next subgoal predicate implied by the language instruction, for a given world state, and (ii) synthesizing a feasible subgoal-reaching plan from that state. The agent executes the plan, and the two steps are repeated. GOALNET combines (i) learning, where dense representations are acquired for language instruction and the world state via a neural network prediction model, enabling generalization to novel settings and (ii) planning, where the cause-effect modeling by a classical planner eschews irrelevant predicates, facilitating multi-stage decision making in large domains. GOALNET obtains 78% improvement in the goal reaching rate in comparison to several state-of-the-art approaches on benchmark data with multi-stage instructions. Further, GOALNET can generalize to novel instructions for scenes with unseen objects. Source code available at https://github. com/reail-iitd/goalnet.

----

## [2242] SayCanPay: Heuristic Planning with Large Language Models Using Learnable Domain Knowledge

**Authors**: *Rishi Hazra, Pedro Zuidberg Dos Martires, Luc De Raedt*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29991](https://doi.org/10.1609/aaai.v38i18.29991)

**Abstract**:

Large Language Models (LLMs) have demonstrated impressive planning abilities due to their vast "world knowledge". Yet, obtaining plans that are both feasible (grounded in affordances) and cost-effective (in plan length), remains a challenge, despite recent progress. This contrasts with heuristic planning methods that employ domain knowledge (formalized in action models such as PDDL) and heuristic search to generate feasible, optimal plans. Inspired by this, we propose to combine the power of LLMs and heuristic planning by leveraging the world knowledge of LLMs and the principles of heuristic search. Our approach, SayCanPay, employs LLMs to generate actions (Say) guided by learnable domain knowledge, that evaluates actions' feasibility (Can) and long-term reward/payoff (Pay), and heuristic search to select the best sequence of actions. Our contributions are (1) a novel framing of the LLM planning problem in the context of heuristic planning, (2) integrating grounding and cost-effective elements into the generated plans, and (3) using heuristic search over actions. Our extensive evaluations show that our model surpasses other LLM planning approaches.

----

## [2243] A Surprisingly Simple Continuous-Action POMDP Solver: Lazy Cross-Entropy Search Over Policy Trees

**Authors**: *Marcus Hörger, Hanna Kurniawati, Dirk P. Kroese, Nan Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29992](https://doi.org/10.1609/aaai.v38i18.29992)

**Abstract**:

The Partially Observable Markov Decision Process (POMDP) provides a principled framework for decision making in stochastic partially observable environments. However, computing good solutions for problems with continuous action spaces remains challenging. To ease this challenge, we propose a simple online POMDP solver, called Lazy Cross-Entropy Search Over Policy Trees (LCEOPT). At each planning step, our method uses a novel lazy Cross-Entropy method to search the space of policy trees, which provide a simple policy representation. Specifically, we maintain a distribution on promising finite-horizon policy trees. The distribution is iteratively updated by sampling policies, evaluating them via Monte Carlo simulation, and refitting them to the top-performing ones. Our method is lazy in the sense that it exploits the policy tree representation to avoid redundant computations in policy sampling, evaluation, and distribution update. This leads to computational savings of up to two orders of magnitude. Our LCEOPT is surprisingly simple as compared to existing state-of-the-art methods, yet empirically outperforms them on several continuous-action POMDP problems, particularly for problems with higher-dimensional action spaces.

----

## [2244] Optimizing Local Satisfaction of Long-Run Average Objectives in Markov Decision Processes

**Authors**: *David Klaska, Antonín Kucera, Vojtech Kur, Vít Musil, Vojtech Rehák*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29993](https://doi.org/10.1609/aaai.v38i18.29993)

**Abstract**:

Long-run average optimization problems for Markov decision processes (MDPs) require constructing policies with optimal steady-state behavior, i.e., optimal limit frequency of visits to the states. However, such policies may suffer from local instability in the sense that the frequency of states visited in a bounded time horizon along a run differs significantly from the limit frequency. In this work, we propose an efficient algorithmic solution to this problem.

----

## [2245] Monte Carlo Tree Search in the Presence of Transition Uncertainty

**Authors**: *Farnaz Kohankhaki, Kiarash Aghakasiri, Hongming Zhang, Ting-Han Wei, Chao Gao, Martin Müller*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29994](https://doi.org/10.1609/aaai.v38i18.29994)

**Abstract**:

Monte Carlo Tree Search (MCTS) is an immensely popular search-based framework used for decision making. It is traditionally applied to domains where a perfect simulation model of the environment is available. We study and improve MCTS in the context where the environment model is given but imperfect. We show that the discrepancy between the model and the actual environment can lead to significant performance degradation with standard MCTS. We therefore develop Uncertainty Adapted MCTS (UA-MCTS), a more robust algorithm within the MCTS framework. We estimate the transition uncertainty in the given model, and direct the search towards more certain transitions in the state space. We modify all four MCTS phases to improve the search behavior by considering these estimates. We prove, in the corrupted bandit case, that adding uncertainty information to adapt UCB leads to tighter regret bound than standard UCB. Empirically, we evaluate UA-MCTS and its individual components on the deterministic domains from the MinAtar test suite. Our results demonstrate that UA-MCTS strongly improves MCTS in the presence of model transition errors.

----

## [2246] Learning Safe Action Models with Partial Observability

**Authors**: *Hai S. Le, Brendan Juba, Roni Stern*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29995](https://doi.org/10.1609/aaai.v38i18.29995)

**Abstract**:

A common approach for solving planning problems is to model them in a formal language such as the Planning Domain Definition Language (PDDL), and then use an appropriate PDDL planner. 
Several algorithms for learning PDDL models from observations have been proposed but plans created with these learned models may not be sound. 
We propose two algorithms for learning PDDL models that are guaranteed to be safe to use even when given observations that include partially observable states. 
We analyze these algorithms theoretically, characterizing the sample complexity each algorithm requires to guarantee probabilistic completeness. 
We also show experimentally that our algorithms are often better than FAMA, a state-of-the-art PDDL learning algorithm.

----

## [2247] Generalized Planning for the Abstraction and Reasoning Corpus

**Authors**: *Chao Lei, Nir Lipovetzky, Krista A. Ehinger*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29996](https://doi.org/10.1609/aaai.v38i18.29996)

**Abstract**:

The Abstraction and Reasoning Corpus (ARC) is a general artificial intelligence benchmark that poses difficulties for pure machine learning methods due to its requirement for fluid intelligence with a focus on reasoning and abstraction. In this work, we introduce an ARC solver, Generalized Planning for Abstract Reasoning (GPAR). It casts an ARC problem as a generalized planning (GP) problem, where a solution is formalized as a planning program with pointers. We express each ARC problem using the standard Planning Domain Definition Language (PDDL) coupled with external functions representing object-centric abstractions. We show how to scale up GP solvers via domain knowledge specific to ARC in the form of restrictions over the actions model, predicates, arguments and valid structure of planning programs. Our experiments demonstrate that  GPAR outperforms the state-of-the-art solvers on the object-centric tasks of the ARC, showing the effectiveness of GP and the expressiveness of PDDL to model ARC problems. The challenges provided by the ARC benchmark motivate research to advance existing GP solvers and understand new relations with other planning computational models. Code is available at github.com/you68681/GPAR.

----

## [2248] Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice

**Authors**: *Idan Lev-Yehudi, Moran Barenboim, Vadim Indelman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29997](https://doi.org/10.1609/aaai.v38i18.29997)

**Abstract**:

Solving partially observable Markov decision processes (POMDPs) with high dimensional and continuous observations, such as camera images, is required for many real life robotics and planning problems. Recent researches suggested machine learned probabilistic models as observation models, but their use is currently too computationally expensive for online deployment. We deal with the question of what would be the implication of using simplified observation models for planning, while retaining formal guarantees on the quality of the solution. Our main contribution is a novel probabilistic bound based on a statistical total variation distance of the simplified model. We show that it bounds the theoretical POMDP value w.r.t. original model, from the empirical planned value with the simplified model, by generalizing recent results of particle-belief MDP concentration bounds. Our calculations can be separated into offline and online parts, and we arrive at formal guarantees without having to access the costly model at all during planning, which is also a novel result. Finally, we demonstrate in simulation how to integrate the bound into the routine of an existing continuous online POMDP solver.

----

## [2249] Learning to Optimize Permutation Flow Shop Scheduling via Graph-Based Imitation Learning

**Authors**: *Longkang Li, Siyuan Liang, Zihao Zhu, Chris Ding, Hongyuan Zha, Baoyuan Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29998](https://doi.org/10.1609/aaai.v38i18.29998)

**Abstract**:

The permutation flow shop scheduling (PFSS), aiming at finding the optimal permutation of jobs, is widely used in manufacturing systems. When solving large-scale PFSS problems, traditional optimization algorithms such as heuristics could hardly meet the demands of both solution accuracy and computational efficiency, thus learning-based methods have recently garnered more attention. Some work attempts to solve the problems by reinforcement learning methods, which suffer from slow convergence issues during training and are still not accurate enough regarding the solutions. To that end, we propose to train the model via expert-driven imitation learning, which accelerates convergence more stably and accurately. Moreover, in order to extract better feature representations of input jobs, we incorporate the graph structure as the encoder. The extensive experiments reveal that our proposed model obtains significant promotion and presents excellent generalizability in large-scale problems with up to 1000 jobs. Compared to the state-of-the-art reinforcement learning method, our model's network parameters are reduced to only 37% of theirs, and the solution gap of our model towards the expert solutions decreases from 6.8% to 1.3% on average. The code is available at: https://github.com/longkangli/PFSS-IL.

----

## [2250] NaRuto: Automatically Acquiring Planning Models from Narrative Texts

**Authors**: *Ruiqi Li, Leyang Cui, Songtuan Lin, Patrik Haslum*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.29999](https://doi.org/10.1609/aaai.v38i18.29999)

**Abstract**:

Domain model acquisition has been identified as a bottleneck in the application of planning technology, especially within narrative planning. Learning action models from narrative texts in an automated way is essential to overcome this barrier, but challenging because of the inherent complexities of such texts. We present an evaluation of planning domain models derived from narrative texts using our fully automated, unsupervised system, NaRuto. Our system combines structured event extraction, predictions of commonsense event relations, and textual contradictions and similarities. Evaluation results show that NaRuto generates domain models of significantly better quality than existing fully automated methods, and even sometimes on par with those created by semi-automated methods, with human assistance.

----

## [2251] On the Computational Complexity of Plan Verification, (Bounded) Plan-Optimality Verification, and Bounded Plan Existence

**Authors**: *Songtuan Lin, Conny Olz, Malte Helmert, Pascal Bercher*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30000](https://doi.org/10.1609/aaai.v38i18.30000)

**Abstract**:

In this paper we study the computational complexity of several reasoning tasks centered around the bounded plan existence problem. We do this for standard classical planning and hierarchical task network (HTN) planning and each for a grounded and a lifted representation. Whereas bounded plan existence complexity is known for classical planning, it has not yet been studied for HTN planning. For plan verification, results were available for both formalisms except for the lifted HTN planning. We will present lower and upper bounds of the complexity of plan verification in lifted HTN planning and provide novel insights into its grounded counterpart, in which we show that verification is not just NP-complete in the general case, but already for a severely restricted special case. Finally, we show the complexity concerning verifying the optimality of a given plan and discuss its connection to the bounded plan existence problem.

----

## [2252] PRP Rebooted: Advancing the State of the Art in FOND Planning

**Authors**: *Christian Muise, Sheila A. McIlraith, J. Christopher Beck*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30001](https://doi.org/10.1609/aaai.v38i18.30001)

**Abstract**:

Fully Observable Non-Deterministic (FOND) planning is a variant of classical symbolic planning in which actions are nondeterministic, with an action's outcome known only upon execution. It is a popular planning paradigm with applications ranging from robot planning to dialogue-agent design and reactive synthesis. Over the last 20 years, a number of approaches to FOND planning have emerged. In this work, we establish a new state of the art, following in the footsteps of some of the most powerful FOND planners to date. Our planner, PR2, decisively outperforms the four leading FOND planners, at times by a large margin, in 17 of 18 domains that represent a comprehensive benchmark suite. Ablation studies demonstrate the impact of various techniques we introduce, with the largest improvement coming from our novel FOND-aware heuristic.

----

## [2253] Abstract Action Scheduling for Optimal Temporal Planning via OMT

**Authors**: *Stefan Panjkovic, Andrea Micheli*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30002](https://doi.org/10.1609/aaai.v38i18.30002)

**Abstract**:

Given the model of a system with explicit temporal constraints, optimal temporal planning is the problem of finding a schedule of actions that achieves a certain goal while optimizing an objective function. Recent approaches for optimal planning reduce the problem to a series of queries to an Optimization Modulo Theory (OMT) solver: each query encodes a bounded version of the problem, with additional abstract actions representing an over-approximation of the plans beyond the bound. This technique suffers from performance issues, mainly due to the looseness of the over-approximation, which can include many non-executable plans.
In this paper, we propose a refined abstraction for solving optimal temporal planning via OMT by introducing abstract scheduling constraints, which have a double purpose. First, they enforce a partial ordering of abstract actions based on mutual dependencies between them, which leads to a better makespan estimation and allows to prove optimality sooner. Second, they implicitly forbid circular self-enabling of abstract actions, which is a common cause of spurious models that severely affects performance in existing approaches. We prove the soundness and completeness of the resulting approach and empirically demonstrate its superiority with respect to the state of the art.

----

## [2254] Generalising Planning Environment Redesign

**Authors**: *Alberto Pozanco, Ramon Fraga Pereira, Daniel Borrajo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30003](https://doi.org/10.1609/aaai.v38i18.30003)

**Abstract**:

In Environment Design, one interested party seeks to affect another agent's decisions by applying changes to the environment. Most research on planning environment (re)design assumes the interested party's objective is to facilitate the recognition of goals and plans, and search over the space of environment modifications to find the minimal set of changes that simplify those tasks and optimise a particular metric. This search space is usually intractable, so existing approaches devise metric-dependent pruning techniques for performing search more efficiently. This results in approaches that are not able to generalise across different objectives and/or metrics. In this paper, we argue that the interested party could have objectives and metrics that are not necessarily related to recognising agents' goals or plans. Thus, to generalise the task of Planning Environment Redesign, we develop a general environment redesign approach that is metric-agnostic and leverages recent research on top-quality planning to efficiently redesign planning environments according to any interested party's objective and metric. Experiments over a set of environment redesign benchmarks show that our general approach outperforms existing approaches when using well-known metrics, such as facilitating the recognition of goals, as well as its effectiveness when solving environment redesign tasks that optimise a novel set of different metrics.

----

## [2255] When CEGAR Meets Regression: A Love Story in Optimal Classical Planning

**Authors**: *Martín Pozo, Álvaro Torralba, Carlos Linares López*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30004](https://doi.org/10.1609/aaai.v38i18.30004)

**Abstract**:

Counterexample-Guided Abstraction Refinement (CEGAR) is a prominent technique to generate Cartesian abstractions for guiding search in cost- optimal planning. The core idea is to iteratively refine the abstraction, finding a flaw of the current optimal abstract plan. All existing approaches find these flaws by executing the abstract plan using progression in the original state space.

Instead, we propose to do backward refinements by using regression from the goals. This results in a new type of flaw, that can identify invalid plan suffixes. The resulting abstractions are less focused on the initial state, but more informative on average, significantly improving the performance of current CEGAR-based techniques. Furthermore, they can be combined with forward refinements in several bidirectional strategies that provide the benefits of both methods.

----

## [2256] Efficient Constraint Generation for Stochastic Shortest Path Problems

**Authors**: *Johannes Schmalz, Felipe W. Trevizan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30005](https://doi.org/10.1609/aaai.v38i18.30005)

**Abstract**:

Current methods for solving Stochastic Shortest Path Problems (SSPs) find states’ costs-to-go by applying Bellman backups, where state-of-the-art methods employ heuristics to select states to back up and prune. A fundamental limitation of these algorithms is their need to compute the cost-to-go for every applicable action during each state backup, leading to unnecessary computation for actions identified as sub-optimal. We present new connections between planning and operations research and, using this framework, we address this issue of unnecessary computation by introducing an efficient version of constraint generation for SSPs. This technique allows algorithms to ignore sub-optimal actions and avoid computing their costs-to-go. We also apply our novel technique to iLAO* resulting in a new algorithm, CG-iLAO*. Our experiments show that CG-iLAO* ignores up to 57% of iLAO*’s actions and it solves problems up to 8x and 3x faster than LRTDP and iLAO*.

----

## [2257] Generalized Planning in PDDL Domains with Pretrained Large Language Models

**Authors**: *Tom Silver, Soham Dan, Kavitha Srinivas, Joshua B. Tenenbaum, Leslie Pack Kaelbling, Michael Katz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30006](https://doi.org/10.1609/aaai.v38i18.30006)

**Abstract**:

Recent work has considered whether large language models (LLMs) can function as planners: given a task, generate a plan. We investigate whether LLMs can serve as generalized planners: given a domain and training tasks, generate a program that efficiently produces plans for other tasks in the domain. In particular, we consider PDDL domains and use GPT-4 to synthesize Python programs. We also consider (1) Chain-of-Thought (CoT) summarization, where the LLM is prompted to summarize the domain and propose a strategy in words before synthesizing the program; and (2) automated debugging, where the program is validated with respect to the training tasks, and in case of errors, the LLM is re-prompted with four types of feedback. We evaluate this approach in seven PDDL domains and compare it to four ablations and four baselines. Overall, we find that GPT-4 is a surprisingly powerful generalized planner. We also conclude that automated debugging is very important, that CoT summarization has non-uniform impact, that GPT-4 is far superior to GPT-3.5, and that just two training tasks are often sufficient for strong generalization.

----

## [2258] Equity-Transformer: Solving NP-Hard Min-Max Routing Problems as Sequential Generation with Equity Context

**Authors**: *Jiwoo Son, Minsu Kim, Sanghyeok Choi, Hyeonah Kim, Jinkyoo Park*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30007](https://doi.org/10.1609/aaai.v38i18.30007)

**Abstract**:

Min-max routing problems aim to minimize the maximum tour length among multiple agents as they collaboratively visit all cities, i.e., the completion time. These problems include impactful real-world applications but are known as NP-hard. Existing methods are facing challenges, particularly in large-scale problems that require the coordination of numerous agents to cover thousands of cities. This paper proposes Equity-Transformer to solve large-scale min-max routing problems. First, we model min-max routing problems into sequential planning, reducing the complexity and enabling the use of a powerful Transformer architecture. Second, we propose key inductive biases that ensure equitable workload distribution among agents. The effectiveness of Equity-Transformer is demonstrated through its superior performance in two representative min-max routing tasks: the min-max multi-agent traveling salesman problem (min-max mTSP) and the min-max multi-agent pick-up and delivery problem (min-max mPDP). Notably, our method achieves significant reductions of runtime, approximately 335 times, and cost values of about 53% compared to a competitive heuristic (LKH3) in the case of 100 vehicles with 1,000 cities of mTSP. We provide reproducible source code: https://github.com/kaist-silab/equity-transformer.

----

## [2259] Distilling Autoregressive Models to Obtain High-Performance Non-autoregressive Solvers for Vehicle Routing Problems with Faster Inference Speed

**Authors**: *Yubin Xiao, Di Wang, Boyang Li, Mingzhao Wang, Xuan Wu, Changliang Zhou, You Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30008](https://doi.org/10.1609/aaai.v38i18.30008)

**Abstract**:

Neural construction models have shown promising performance for Vehicle Routing Problems (VRPs) by adopting either the Autoregressive (AR) or Non-Autoregressive (NAR) learning approach. While AR models produce high-quality solutions, they generally have a high inference latency due to their sequential generation nature. Conversely, NAR models generate solutions in parallel with a low inference latency but generally exhibit inferior performance. In this paper, we propose a generic Guided Non-Autoregressive Knowledge Distillation (GNARKD) method to obtain high-performance NAR models having a low inference latency. GNARKD removes the constraint of sequential generation in AR models while preserving the learned pivotal components in the network architecture to obtain the corresponding NAR models through knowledge distillation. We evaluate GNARKD by applying it to three widely adopted AR models to obtain NAR VRP solvers for both synthesized and real-world instances. The experimental results demonstrate that GNARKD significantly reduces the inference time (4-5 times faster) with acceptable performance drop (2-3%). To the best of our knowledge, this study is first-of-its-kind to obtain NAR VRP solvers from AR ones through knowledge distillation.

----

## [2260] GLOP: Learning Global Partition and Local Construction for Solving Large-Scale Routing Problems in Real-Time

**Authors**: *Haoran Ye, Jiarui Wang, Helan Liang, Zhiguang Cao, Yong Li, Fanzhang Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30009](https://doi.org/10.1609/aaai.v38i18.30009)

**Abstract**:

The recent end-to-end neural solvers have shown promise for small-scale routing problems but suffered from limited real-time scaling-up performance. This paper proposes GLOP (Global and Local Optimization Policies), a unified hierarchical framework that efficiently scales toward large-scale routing problems. GLOP hierarchically partitions large routing problems into Travelling Salesman Problems (TSPs) and TSPs into Shortest Hamiltonian Path Problems. For the first time, we hybridize non-autoregressive neural heuristics for coarse-grained problem partitions and autoregressive neural heuristics for fine-grained route constructions, leveraging the scalability of the former and the meticulousness of the latter. Experimental results show that GLOP achieves competitive and state-of-the-art real-time performance on large-scale routing problems, including TSP, ATSP, CVRP, and PCTSP. Our code is available at: https://github.com/henry-yeh/GLOP.

----

## [2261] Learning-Augmented Online Algorithm for Two-Level Ski-Rental Problem

**Authors**: *Keyuan Zhang, Zhongdong Liu, Nakjung Choi, Bo Ji*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30010](https://doi.org/10.1609/aaai.v38i18.30010)

**Abstract**:

In this paper, we study the two-level ski-rental problem, where a user needs to fulfill a sequence of demands for multiple items by choosing one of the three payment options: paying for the on-demand usage (i.e., rent), buying individual items (i.e., single purchase), and buying all the items (i.e., combo purchase). Without knowing future demands, the user aims to minimize the total cost (i.e., the sum of the rental, single purchase, and combo purchase costs) by balancing the trade-off between the expensive upfront costs (for purchase) and the potential future expenses (for rent). We first design a robust online algorithm (RDTSR) that offers a worst-case performance guarantee. While online algorithms are robust against the worst-case scenarios, they are often overly cautious and thus suffer a poor average performance in typical scenarios. On the other hand, Machine Learning (ML) algorithms typically show promising average performance in various applications but lack worst-case performance guarantees. To harness the benefits of both methods, we develop a learning-augmented algorithm (LADTSR) by integrating ML predictions into the robust online algorithm, which outperforms the robust online algorithm under accurate predictions while ensuring worst-case performance guarantees even when predictions are inaccurate. Finally, we conduct numerical experiments on both synthetic and real-world trace data to corroborate the effectiveness of our approach.

----

## [2262] s-ID: Causal Effect Identification in a Sub-population

**Authors**: *Amir Mohammad Abouei, Ehsan Mokhtarian, Negar Kiyavash*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30011](https://doi.org/10.1609/aaai.v38i18.30011)

**Abstract**:

Causal inference in a sub-population involves identifying the causal effect of an intervention on a specific subgroup, which is distinguished from the whole population through the influence of systematic biases in the sampling process. However, ignoring the subtleties introduced by sub-populations can either lead to erroneous inference or limit the applicability of existing methods. We introduce and advocate for a causal inference problem in sub-populations (henceforth called s-ID), in which we merely have access to observational data of the targeted sub-population (as opposed to the entire population). Existing inference problems in sub-populations operate on the premise that the given data distributions originate from the entire population, thus, cannot tackle the s-ID problem. To address this gap, we provide necessary and sufficient conditions that must hold in the causal graph for a causal effect in a sub-population to be identifiable from the observational distribution of that sub-population. Given these conditions, we present a sound and complete algorithm for the s-ID problem.

----

## [2263] On Estimating the Gradient of the Expected Information Gain in Bayesian Experimental Design

**Authors**: *Ziqiao Ao, Jinglai Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30012](https://doi.org/10.1609/aaai.v38i18.30012)

**Abstract**:

Bayesian Experimental Design (BED), which aims to find the optimal experimental conditions for Bayesian inference, is usually posed as to optimize the expected information gain (EIG). The gradient information is often needed for efficient EIG optimization, and as a result the ability to estimate the gradient of EIG is essential for BED problems. The primary goal of this work is to develop methods for estimating the gradient of EIG, which, combined with the stochastic gradient descent algorithms,  result in efficient optimization of EIG. Specifically, we first introduce a posterior expected representation of the EIG gradient with respect to the design variables. Based on this, we propose two methods for estimating the EIG gradient, UEEG-MCMC that leverages posterior samples generated through Markov Chain Monte Carlo (MCMC) to estimate the EIG gradient, and BEEG-AP that focuses on achieving high simulation efficiency by repeatedly using parameter samples. Theoretical analysis and numerical studies illustrate that UEEG-MCMC is robust agains the actual EIG value, while BEEG-AP is more efficient when the EIG value to be optimized is small. Moreover, both methods show superior performance compared to several popular benchmarks in our numerical experiments.

----

## [2264] Backward Responsibility in Transition Systems Using General Power Indices

**Authors**: *Christel Baier, Roxane van den Bossche, Sascha Klüppelholz, Johannes Lehmann, Jakob Piribauer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30013](https://doi.org/10.1609/aaai.v38i18.30013)

**Abstract**:

To improve reliability and the understanding of AI systems, there is increasing interest in the use of formal methods, e.g. model checking. Model checking tools produce a counterexample when a model does not satisfy a property. Understanding these counterexamples is critical for efficient debugging, as it allows the developer to focus  on the parts of the program that caused the issue.

To this end, we present a new technique that ascribes a responsibility value to each state in a transition system that does not satisfy a given safety property. The value is higher if the non-deterministic choices in a state have more power to change the outcome, given the behaviour observed in the counterexample. For this, we employ a concept from cooperative game theory – namely general power indices, such as the Shapley value – to compute the responsibility of the states.

We present an optimistic and pessimistic version of responsibility that differ in how they treat the states that do not lie on the counterexample. We give a characterisation of optimistic responsibility that leads to an efficient algorithm for it and show computational hardness of the pessimistic version. We also present a tool to compute responsibility and show how a stochastic algorithm can be used to approximate responsibility in larger models. These methods can be deployed in the design phase, at runtime and at inspection time to gain insights on causal relations within the behavior of AI systems.

----

## [2265] The Expected Loss of Preconditioned Langevin Dynamics Reveals the Hessian Rank

**Authors**: *Amitay Bar, Rotem Mulayoff, Tomer Michaeli, Ronen Talmon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30014](https://doi.org/10.1609/aaai.v38i18.30014)

**Abstract**:

Langevin dynamics (LD) is widely used for sampling from distributions and for optimization. In this work, we derive a closed-form expression for the expected loss of preconditioned LD near stationary points of the objective function. We use the fact that at the vicinity of such points, LD reduces to an Ornstein–Uhlenbeck process, which is amenable to convenient mathematical treatment. Our analysis reveals that when the preconditioning matrix satisfies a particular relation with respect to the noise covariance, LD's expected loss becomes proportional to the rank of the objective's Hessian. We illustrate the applicability of this result in the context of neural networks, where the Hessian rank has been shown to capture the complexity of the predictor function but is usually computationally hard to probe. Finally, we use our analysis to compare SGD-like and Adam-like preconditioners and identify the regimes under which each of them leads to a lower expected loss.

----

## [2266] Pandora's Problem with Deadlines

**Authors**: *Ben Berger, Tomer Ezra, Michal Feldman, Federico Fusco*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30015](https://doi.org/10.1609/aaai.v38i18.30015)

**Abstract**:

Pandora’s problem is a fundamental model that studies optimal search under costly inspection. In the classic version, there are n boxes, each associated with a known cost and a known distribution over values. A strategy inspects the boxes sequentially and obtains a utility that equals the difference between the maximum value of an inspected box and the total inspection cost. Weitzman (1979) presented a surprisingly simple strategy that obtains the optimal expected utility.

In this work we introduce a new variant of Pandora’s problem in which every box is also associated with a publicly known deadline, indicating the final round by which its value may be chosen. This model captures many real-life scenarios where alternatives admit deadlines, such as candidate interviews and college admissions. Our main result is an efficient threshold-based strategy that achieves a constant approximation relative to the performance of the optimal strategy for the deadlines setting.

----

## [2267] Minibatch Stochastic Three Points Method for Unconstrained Smooth Minimization

**Authors**: *Soumia Boucherouite, Grigory Malinovsky, Peter Richtárik, El Houcine Bergou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30016](https://doi.org/10.1609/aaai.v38i18.30016)

**Abstract**:

We present a new zero-order optimization method called Minibatch Stochastic Three Points (MiSTP), specifically designed to solve stochastic unconstrained minimization problems when only an approximate evaluation of the objective function is possible. MiSTP is an extension of the Stochastic Three Point Method (STP). The key innovation of MiSTP is that it selects the next point solely based on the objective function approximation, without relying on its exact evaluation. At each iteration, MiSTP generates a random search direction and compares the approximations of the objective function at the current point, the randomly generated direction and its opposite. The best of these three points is chosen as the next iterate. We analyze the worst-case complexity of MiSTP in the convex and non-convex cases and demonstrate that it matches the most accurate complexity bounds known in the literature for zero-order optimization methods. We perform extensive numerical evaluations to assess the computational efficiency of MiSTP and compare its performance to other state-of-the-art methods by testing it on several machine learning tasks. The results show that MiSTP outperforms or has comparable performance against state-of-the-art methods indicating its potential for a wide range of practical applications.

----

## [2268] Identification of Causal Structure with Latent Variables Based on Higher Order Cumulants

**Authors**: *Wei Chen, Zhiyi Huang, Ruichu Cai, Zhifeng Hao, Kun Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30017](https://doi.org/10.1609/aaai.v38i18.30017)

**Abstract**:

Causal discovery with latent variables is a crucial but challenging task. Despite the emergence of numerous methods aimed at addressing this challenge, they are not fully identified to the structure that two observed variables are influenced by one latent variable and there might be a directed edge in between. Interestingly, we notice that this structure can be identified through the utilization of higher-order cumulants. By leveraging the higher-order cumulants of non-Gaussian data, we provide an analytical solution for estimating the causal coefficients or their ratios. With the estimated (ratios of) causal coefficients, we propose a novel approach to identify the existence of a causal edge between two observed variables subject to latent variable influence. In case when such a causal edge exits, we introduce an asymmetry criterion to determine the causal direction. The experimental results demonstrate the effectiveness of our proposed method.

----

## [2269] Direct Amortized Likelihood Ratio Estimation

**Authors**: *Adam D. Cobb, Brian Matejek, Daniel Elenius, Anirban Roy, Susmit Jha*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30018](https://doi.org/10.1609/aaai.v38i18.30018)

**Abstract**:

We introduce a new amortized likelihood ratio estimator for likelihood-free simulation-based inference (SBI). Our estimator is simple to train and estimates the likelihood ratio using a single forward pass of the neural estimator. Our approach directly computes the likelihood ratio between two competing parameter sets which is different from the previous approach of comparing two neural network output values. We refer to our model as the direct neural ratio estimator (DNRE). As part of introducing the DNRE, we derive a corresponding Monte Carlo estimate of the posterior. We benchmark our new ratio estimator and compare to previous ratio estimators in the literature. We show that our new ratio estimator often outperforms these previous approaches. As a further contribution, we introduce a new derivative estimator for likelihood ratio estimators that enables us to compare likelihood-free Hamiltonian Monte Carlo (HMC) with random-walk Metropolis-Hastings (MH). We show that HMC is equally competitive, which has not been previously shown. Finally, we include a novel real-world application of SBI by using our neural ratio estimator to design a quadcopter. Code is available at https://github.com/SRI-CSL/dnre.

----

## [2270] Probabilistic Offline Policy Ranking with Approximate Bayesian Computation

**Authors**: *Longchao Da, Porter Jenkins, Trevor Schwantes, Jeffrey Dotson, Hua Wei*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30019](https://doi.org/10.1609/aaai.v38i18.30019)

**Abstract**:

In practice, it is essential to compare and rank candidate policies offline before real-world deployment for safety and reliability. Prior work seeks to solve this offline policy ranking (OPR) problem through value-based methods, such as Off-policy evaluation (OPE). However, they fail to analyze special case performance (e.g., worst or best cases), due to the lack of holistic characterization of policies’ performance. It is even more difficult to estimate precise policy values when the reward is not fully accessible under sparse settings. In this paper, we present Probabilistic Offline Policy Ranking (POPR), a framework to address OPR problems by leveraging expert data to characterize the probability of a candidate policy behaving like experts, and approximating its entire performance posterior distribution to help with ranking. POPR does not rely on value estimation, and the derived performance posterior can be used to distinguish candidates in worst-, best-, and average-cases. To estimate the posterior, we propose POPR-EABC, an Energy-based Approximate Bayesian Computation (ABC) method conducting likelihood-free inference. POPR-EABC reduces the heuristic nature of ABC by a smooth energy function, and improves the sampling efficiency by a pseudo-likelihood. We empirically demonstrate that POPR-EABC is adequate for evaluating policies in both discrete and continuous action spaces across various experiment environments, and facilitates probabilistic comparisons of candidate policies before deployment.

----

## [2271] Generalized Bradley-Terry Models for Score Estimation from Paired Comparisons

**Authors**: *Julien Fageot, Sadegh Farhadkhani, Lê-Nguyên Hoang, Oscar Villemaud*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30020](https://doi.org/10.1609/aaai.v38i18.30020)

**Abstract**:

Many applications, e.g. in content recommendation, sports, or recruitment, leverage the comparisons of alternatives to score those alternatives. The classical Bradley-Terry model and its variants have been widely used to do so. The historical model considers binary comparisons (victory/defeat) between alternatives, while more recent developments allow finer comparisons to be taken into account. In this article, we introduce a probabilistic model encompassing a broad variety of paired comparisons that can take discrete or continuous values. We do so by considering a well-behaved subset of the exponential family, which we call the family of generalized Bradley-Terry (GBT) models, as it includes the classical Bradley-Terry model and many of its variants. Remarkably, we prove that all GBT models are guaranteed to yield a strictly convex negative log-likelihood. Moreover, assuming a Gaussian prior on alternatives' scores, we prove that the maximum a posteriori (MAP) of GBT models, whose existence, uniqueness and fast computation are thus guaranteed, varies monotonically with respect to comparisons (the more A beats B, the better the score of A) and is Lipschitz-resilient with respect to each new comparison (a single new comparison can only have a bounded effect on all the estimated scores). These desirable properties make GBT models appealing for practical use. We illustrate some features of GBT models on simulations.

----

## [2272] Identifiability of Direct Effects from Summary Causal Graphs

**Authors**: *Simon Ferreira, Charles K. Assaad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30021](https://doi.org/10.1609/aaai.v38i18.30021)

**Abstract**:

Dynamic structural causal models (SCMs) are a powerful framework for reasoning in dynamic systems about direct effects which measure how a change in one variable affects another variable while holding all other variables constant. The causal relations in a dynamic structural causal model can be qualitatively represented with an acyclic full-time causal graph. Assuming linearity and no hidden confounding and given the full-time causal graph, the direct causal effect is always identifiable. However, in many application such a graph is not available for various reasons but nevertheless experts have access to the summary causal graph of the full-time causal graph which represents causal relations between time series while omitting temporal information and allowing cycles. This paper presents a complete identifiability result which characterizes all cases for which the direct effect
is graphically identifiable from a summary causal graph and gives two sound finite adjustment sets that can be used to estimate the direct effect whenever it is identifiable.

----

## [2273] Model Counting and Sampling via Semiring Extensions

**Authors**: *Andreas Goral, Joachim Giesen, Mark Blacher, Christoph Staudt, Julien Klaus*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30022](https://doi.org/10.1609/aaai.v38i18.30022)

**Abstract**:

Many decision and optimization problems have natural extensions as counting problems. The best known example is the Boolean satisfiability problem (SAT), where we want to count the satisfying assignments of truth values to the variables, which is known as the #SAT problem. Likewise, for discrete optimization problems, we want to count the states on which the objective function attains the optimal value. Both SAT and discrete optimization can be formulated as selective marginalize a product function (MPF) queries. Here, we show how general selective MPF queries can be extended for model counting. MPF queries are encoded as tensor hypernetworks over suitable semirings that can be solved by generic tensor hypernetwork contraction algorithms. Our model counting extension is again an MPF query, on an extended semiring, that can be solved by the same contraction algorithms. Model counting is required for uniform model sampling. We show how the counting extension can be further extended for model sampling by constructing yet another semiring. We have implemented the model counting and sampling extensions. Experiments show that our generic approach is competitive with the state of the art in model counting and model sampling.

----

## [2274] Identification for Tree-Shaped Structural Causal Models in Polynomial Time

**Authors**: *Aaryan Gupta, Markus Bläser*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30023](https://doi.org/10.1609/aaai.v38i18.30023)

**Abstract**:

Linear structural causal models (SCMs) are used to express and analyze the relationships between random variables. Direct causal effects are represented as directed edges and confounding factors as bidirected edges. Identifying the causal parameters from correlations between the nodes is an open problem in artificial intelligence. In this paper, we study SCMs whose directed component forms a tree. Van der Zander et al. give a PSPACE-algorithm for the identification problem in this case, which is a significant improvement over the general Gröbner basis approach, which has doubly-exponential time complexity in the number of structural parameters. However, they do not show that their algorithm is complete. In this work, we present a randomized polynomial-time algorithm, which solves the identification problem for tree-shaped SCMs. For every structural parameter, our algorithms decides whether it is generically identifiable, generically 2-identifiable, or generically unidentifiable. (No other cases can occur.) In the first two cases, it provides one or two  fractional affine square root terms of polynomials (FASTPs) for the corresponding parameter, respectively. In particular, our algorithm is not only polynomial time, but also complete for for tree-shaped SCMs.

----

## [2275] Learning GAI-Decomposable Utility Models for Multiattribute Decision Making

**Authors**: *Margot Herin, Patrice Perny, Nataliya Sokolovska*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30024](https://doi.org/10.1609/aaai.v38i18.30024)

**Abstract**:

We propose an approach to learn a multiattribute utility function to model, explain or predict the value system of a Decision Maker. The main challenge of the modelling task is to describe human values and preferences in the presence of interacting attributes while keeping the utility function as simple as possible. We focus on the generalized additive decomposable utility model which allows interactions between attributes while preserving some additive decomposability of the evaluation model. We present a learning approach able to identify the factors of interacting attributes and to learn the utility functions defined on these factors. This approach relies on the determination of a sparse representation of the ANOVA decomposition of the multiattribute utility function using multiple kernel learning. It applies to both continuous and discrete attributes. Numerical tests are performed to demonstrate the practical efficiency of the learning approach.

----

## [2276] Uncertainty Quantification in Heterogeneous Treatment Effect Estimation with Gaussian-Process-Based Partially Linear Model

**Authors**: *Shunsuke Horii, Yoichi Chikahara*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30025](https://doi.org/10.1609/aaai.v38i18.30025)

**Abstract**:

Estimating heterogeneous treatment effects across individuals has attracted growing attention as a statistical tool for performing critical decision-making. We propose a Bayesian inference framework that quantifies the uncertainty in treatment effect estimation to support decision-making in a relatively small sample size setting. Our proposed model places Gaussian process priors on the nonparametric components of a semiparametric model called a partially linear model. This model formulation has three advantages. First, we can analytically compute the posterior distribution of a treatment effect without relying on the computationally demanding posterior approximation. Second, we can guarantee that the posterior distribution concentrates around the true one as the sample size goes to infinity. Third, we can incorporate prior knowledge about a treatment effect into the prior distribution, improving the estimation efficiency. Our experimental results show that even in the small sample size setting, our method can accurately estimate the heterogeneous treatment effects and effectively quantify its estimation uncertainty.

----

## [2277] Learning Diffusions under Uncertainty

**Authors**: *Hao Huang, Qian Yan, Keqi Han, Ting Gan, Jiawei Jiang, Quanqing Xu, Chuanhui Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30026](https://doi.org/10.1609/aaai.v38i18.30026)

**Abstract**:

To infer a diffusion network based on observations from historical diffusion processes, existing approaches assume that observation data contain exact occurrence time of each node infection, or at least the eventual infection statuses of nodes in each diffusion process. They determine potential influence relationships between nodes by identifying frequent sequences, or statistical correlations, among node infections. In some real-world settings, such as the spread of epidemics, tracing exact infection times is often infeasible due to a high cost; even obtaining precise infection statuses of nodes is a challenging task, since observable symptoms such as headache only partially reveal a node’s true status. In this work, we investigate how to effectively infer a diffusion network from observation data with uncertainty. Provided with only probabilistic information about node infection statuses, we formulate the problem of diffusion network inference as a constrained nonlinear regression w.r.t. the probabilistic data. An alternating maximization method is designed to solve this regression problem iteratively, and the improvement of solution quality in each iteration can be theoretically guaranteed. Empirical studies are conducted on both synthetic and real-world networks, and the results verify the effectiveness and efficiency of our approach.

----

## [2278] Robustly Improving Bandit Algorithms with Confounded and Selection Biased Offline Data: A Causal Approach

**Authors**: *Wen Huang, Xintao Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30027](https://doi.org/10.1609/aaai.v38i18.30027)

**Abstract**:

This paper studies bandit problems where an agent has access to offline data that might be utilized to potentially improve the estimation of each arm’s reward distribution. A major obstacle in this setting is the existence of compound biases from the observational data. Ignoring these biases and blindly fitting a model with the biased data could even negatively affect the online learning phase. In this work, we formulate this problem from a causal perspective. First, we categorize the biases into confounding bias and selection bias based on the causal structure they imply. Next, we extract the causal bound for each arm that is robust towards compound biases from biased observational data. The derived bounds contain the
ground truth mean reward and can effectively guide the bandit agent to learn a nearly-optimal decision policy. We also conduct regret analysis in both contextual and non-contextual bandit settings and show that prior causal bounds could help
consistently reduce the asymptotic regret.

----

## [2279] Effectiveness of Constant Stepsize in Markovian LSA and Statistical Inference

**Authors**: *Dongyan Lucy Huo, Yudong Chen, Qiaomin Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30028](https://doi.org/10.1609/aaai.v38i18.30028)

**Abstract**:

In this paper, we study the effectiveness of using a constant stepsize in statistical inference via linear stochastic approximation (LSA) algorithms with Markovian data. After establishing a Central Limit Theorem (CLT), we outline an inference procedure that uses averaged LSA iterates to construct confidence intervals (CIs). Our procedure leverages the fast mixing property of constant-stepsize LSA for better covariance estimation and employs Richardson-Romberg (RR) extrapolation to reduce the bias induced by constant stepsize and Markovian data. We develop theoretical results for guiding stepsize selection in RR extrapolation, and identify several important settings where the bias provably vanishes even without extrapolation. We conduct extensive numerical experiments and compare against classical inference approaches. Our results show that using a constant stepsize enjoys easy hyperparameter tuning, fast convergence, and consistently better CI coverage, especially when data is limited.

----

## [2280] Piecewise Linear Transformation - Propagating Aleatoric Uncertainty in Neural Networks

**Authors**: *Thomas Krapf, Michael Hagn, Paul Miethaner, Alexander Schiller, Lucas Luttner, Bernd Heinrich*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30029](https://doi.org/10.1609/aaai.v38i18.30029)

**Abstract**:

Real-world data typically exhibit aleatoric uncertainty which has to be considered during data-driven decision-making to assess the confidence of the decision provided by machine learning models. To propagate aleatoric uncertainty represented by probability distributions (PDs) through neural networks (NNs), both sampling-based and function approximation-based methods have been proposed. However, these methods suffer from significant approximation errors and are not able to accurately represent predictive uncertainty in the NN output. In this paper, we present a novel method, Piecewise Linear Transformation (PLT), for propagating PDs through NNs with piecewise linear activation functions (e.g., ReLU NNs). PLT does not require sampling or specific assumptions about the PDs. Instead, it harnesses the piecewise linear structure of such NNs to determine the propagated PD in the output space. In this way, PLT supports the accurate quantification of predictive uncertainty based on the criterion exactness of the propagated PD. We assess this exactness in theory by showing error bounds for our propagated PD. Further, our experimental evaluation validates that PLT outperforms competing methods on publicly available real-world classification and regression datasets regarding exactness. Thus, the PDs propagated by PLT allow to assess the uncertainty of the provided decisions, offering valuable support.

----

## [2281] Probabilities of Causation with Nonbinary Treatment and Effect

**Authors**: *Ang Li, Judea Pearl*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30030](https://doi.org/10.1609/aaai.v38i18.30030)

**Abstract**:

Probabilities of causation are proven to be critical in modern decision-making. This paper deals with the problem of estimating the probabilities of causation when treatment and effect are not binary. Pearl defined the binary probabilities of causation, such as the probability of necessity and sufficiency (PNS), the probability of sufficiency (PS), and the probability of necessity (PN). Tian and Pearl then derived sharp bounds for these probabilities of causation using experimental and observational data. In this paper, we define and provide theoretical bounds for all types of probabilities of causation with multivalued treatments and effects. We further discuss examples where our bounds guide practical decisions and use simulation studies to evaluate how informative the bounds are for various data combinations.

----

## [2282] Unit Selection with Nonbinary Treatment and Effect

**Authors**: *Ang Li, Judea Pearl*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30031](https://doi.org/10.1609/aaai.v38i18.30031)

**Abstract**:

The unit selection problem aims to identify a set of individuals who are most likely to exhibit a desired mode of behavior or to evaluate the percentage of such individuals in a given population, for example, selecting individuals who would respond one way if encouraged and a different way if not encouraged. Using a combination of experimental and observational data, Li and Pearl solved the binary unit selection problem (binary treatment and effect) by deriving tight bounds on the "benefit function," which is the payoff/cost associated with selecting an individual with given characteristics. This paper extends the benefit function to the general form such that the treatment and effect are not restricted to binary. We then propose an algorithm to test the identifiability of the nonbinary benefit function and an algorithm to compute the bounds of the nonbinary benefit function using experimental and observational data.

----

## [2283] Solving Satisfiability Modulo Counting for Symbolic and Statistical AI Integration with Provable Guarantees

**Authors**: *Jinzhao Li, Nan Jiang, Yexiang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30032](https://doi.org/10.1609/aaai.v38i18.30032)

**Abstract**:

Satisfiability Modulo Counting (SMC) encompasses problems that require both symbolic decision-making and statistical reasoning. Its general formulation captures many real-world problems at the intersection of symbolic and statistical AI. SMC searches for policy interventions to control probabilistic outcomes. Solving SMC is challenging because of its highly intractable nature (NP^PP-complete), incorporating statistical inference and symbolic reasoning. Previous research on SMC solving lacks provable guarantees and/or suffers from suboptimal empirical performance, especially when combinatorial constraints are present. We propose XOR-SMC, a polynomial algorithm with access to NP-oracles, to solve highly intractable SMC problems with constant approximation guarantees. XOR-SMC transforms the highly intractable SMC into satisfiability problems by replacing the model counting in SMC with SAT formulae subject to randomized XOR constraints. Experiments on solving important SMC problems in AI for social good demonstrate that XOR-SMC outperforms several baselines both in solution quality and running time.

----

## [2284] TNPAR: Topological Neural Poisson Auto-Regressive Model for Learning Granger Causal Structure from Event Sequences

**Authors**: *Yuequn Liu, Ruichu Cai, Wei Chen, Jie Qiao, Yuguang Yan, Zijian Li, Keli Zhang, Zhifeng Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30033](https://doi.org/10.1609/aaai.v38i18.30033)

**Abstract**:

Learning Granger causality from event sequences is a challenging but essential task across various applications. Most existing methods rely on the assumption that event sequences are independent and identically distributed (i.i.d.). However, this i.i.d. assumption is often violated due to the inherent dependencies among the event sequences. Fortunately, in practice, we find these dependencies can be modeled by a topological network, suggesting a potential solution to the non-i.i.d. problem by introducing the prior topological network into Granger causal discovery. This observation prompts us to tackle two ensuing challenges: 1) how to model the event sequences while incorporating both the prior topological network and the latent Granger causal structure, and 2) how to learn the Granger causal structure. To this end, we devise a unified topological neural Poisson auto-regressive model with two processes. In the generation process, we employ a variant of the neural Poisson process to model the event sequences, considering influences from both the topological network and the Granger causal structure. In the inference process, we formulate an amortized inference algorithm to infer the latent Granger causal structure. We encapsulate these two processes within a unified likelihood function, providing an end-to-end framework for this task. Experiments on simulated and real-world data demonstrate the effectiveness of our approach.

----

## [2285] Colour Passing Revisited: Lifted Model Construction with Commutative Factors

**Authors**: *Malte Luttermann, Tanya Braun, Ralf Möller, Marcel Gehrke*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30034](https://doi.org/10.1609/aaai.v38i18.30034)

**Abstract**:

Lifted probabilistic inference exploits symmetries in a probabilistic model to allow for tractable probabilistic inference with respect to domain sizes. To apply lifted inference, a lifted representation has to be obtained, and to do so, the so-called colour passing algorithm is the state of the art. The colour passing algorithm, however, is bound to a specific inference algorithm and we found that it ignores commutativity of factors while constructing a lifted representation. We contribute a modified version of the colour passing algorithm that uses logical variables to construct a lifted representation independent of a specific inference algorithm while at the same time exploiting commutativity of factors during an offline-step. Our proposed algorithm efficiently detects more symmetries than the state of the art and thereby drastically increases compression, yielding significantly faster online query times for probabilistic inference when the resulting model is applied.

----

## [2286] Root Cause Explanation of Outliers under Noisy Mechanisms

**Authors**: *Phuoc Nguyen, Truyen Tran, Sunil Gupta, Thin Nguyen, Svetha Venkatesh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30035](https://doi.org/10.1609/aaai.v38i18.30035)

**Abstract**:

Identifying root causes of anomalies in causal processes is vital across disciplines. Once identified, one can isolate the root causes and implement necessary measures to restore the normal operation. Causal processes are often modelled as graphs with entities being nodes and their paths/interconnections as edge. Existing work only consider the contribution of nodes in the generative process, thus can not attribute the outlier score to the edges of the mechanism if the anomaly occurs in the connections. In this paper, we consider both individual edge and node of each mechanism when identifying the root causes. We introduce a noisy functional causal model to account for this purpose. Then, we employ Bayesian learning and inference methods to infer the noises of the nodes and edges. We then represent the functional form of a target outlier leaf as a function of the node and edge noises. Finally, we propose an efficient gradient-based attribution method to compute the anomaly attribution scores which scales linearly with the number of nodes and edges. Experiments on simulated datasets and two real-world scenario datasets show better anomaly attribution performance of the proposed method compared to the baselines. Our method scales to larger graphs with more nodes and edges.

----

## [2287] Identification of Causal Structure in the Presence of Missing Data with Additive Noise Model

**Authors**: *Jie Qiao, Zhengming Chen, Jianhua Yu, Ruichu Cai, Zhifeng Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30036](https://doi.org/10.1609/aaai.v38i18.30036)

**Abstract**:

Missing data are an unavoidable complication frequently encountered in many causal discovery tasks. 
When a missing process depends on the missing values themselves (known as self-masking missingness), the recovery of the joint distribution becomes unattainable, and detecting the presence of such self-masking missingness remains a perplexing challenge. Consequently, due to the inability to reconstruct the original distribution and to discern the underlying missingness mechanism, simply applying existing causal discovery methods would lead to wrong conclusions. In this work, we found that the recent advances additive noise model has the potential for learning causal structure under the existence of the self-masking missingness. With this observation, we aim to investigate the identification problem of learning causal structure from missing data under an additive noise model with different missingness mechanisms, where the `no self-masking missingness' assumption can be eliminated appropriately. 
Specifically, we first elegantly extend the scope of identifiability of causal skeleton to the case with weak self-masking missingness (i.e., no other variable could be the cause of self-masking indicators except itself). We further provide the sufficient and necessary identification conditions of the causal direction under additive noise model and show that the causal structure can be identified up to an IN-equivalent pattern. We finally propose a practical algorithm based on the above theoretical results on learning the causal skeleton and causal direction. Extensive experiments on synthetic and real data demonstrate the efficiency and effectiveness of the proposed algorithms.

----

## [2288] Causal Discovery from Poisson Branching Structural Causal Model Using High-Order Cumulant with Path Analysis

**Authors**: *Jie Qiao, Yu Xiang, Zhengming Chen, Ruichu Cai, Zhifeng Hao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30037](https://doi.org/10.1609/aaai.v38i18.30037)

**Abstract**:

Count data naturally arise in many fields, such as finance, neuroscience, and epidemiology, and discovering causal structure among count data is a crucial task in various scientific and industrial scenarios. One of the most common characteristics of count data is the inherent branching structure described by a binomial thinning operator and an independent Poisson distribution that captures both branching and noise. For instance, in a population count scenario, mortality and immigration contribute to the count, where survival follows a Bernoulli distribution, and immigration follows a Poisson distribution. However, causal discovery from such data is challenging due to the non-identifiability issue: a single causal pair is Markov equivalent, i.e.,  X->Y and Y->X are distributed equivalent. Fortunately, in this work, we found that the causal order from X to its child Y is identifiable if X is a root vertex and has at least two directed paths to Y, or the ancestor of X with the most directed path to X has a directed path to Y without passing X. Specifically, we propose a Poisson Branching Structure Causal Model (PB-SCM) and perform a path analysis on PB-SCM using high-order cumulants. Theoretical results establish the connection between the path and cumulant and demonstrate that the path information can be obtained from the cumulant. With the path information, causal order is identifiable under some graphical conditions. A practical algorithm for learning causal structure under PB-SCM is proposed and the experiments demonstrate and verify the effectiveness of the proposed method.

----

## [2289] A Fixed-Parameter Tractable Algorithm for Counting Markov Equivalence Classes with the Same Skeleton

**Authors**: *Vidya Sagar Sharma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30038](https://doi.org/10.1609/aaai.v38i18.30038)

**Abstract**:

Causal DAGs (also known as Bayesian networks) are a popular tool for encoding
conditional dependencies between random variables. In a causal DAG, the random
variables are modeled as vertices in the DAG, and it is stipulated that every
random variable is independent of its non-descendants conditioned on its parents. It
is possible, however, for two different causal DAGs on the same set of random
variables to encode exactly the same set of conditional dependencies. Such
causal DAGs are said to be Markov equivalent, and equivalence classes of
Markov equivalent DAGs are known as Markov Equivalent Classes (MECs).
Beautiful combinatorial characterizations of MECs have been developed in the
past few decades, and it is known, in particular, that all DAGs in the same MEC
must have the same skeleton (underlying undirected graph) and v-structures (induced subgraph of the form a->b<-c).

These combinatorial characterizations also suggest several natural algorithmic
questions. One of these is: given an undirected graph G as input, how many
distinct Markov equivalence classes have the skeleton G? Much work has been
devoted in the last few years to this and other closely related problems.
However, to the best of our knowledge, a polynomial-time algorithm for the
problem remains unknown.

In this paper, we make progress towards this goal by giving a fixed parameter
tractable algorithm for the above problem, with the parameters being the
treewidth and the maximum degree of the input graph G. The main technical
ingredient in our work is a construction we refer to as shadow,
which lets us create a local description of long-range constraints imposed
by the combinatorial characterizations of MECs.

----

## [2290] Learning Bayesian Network Classifiers to Minimize the Class Variable Parameters

**Authors**: *Shouta Sugahara, Koya Kato, Maomi Ueno*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30039](https://doi.org/10.1609/aaai.v38i18.30039)

**Abstract**:

This study proposes and evaluates a new Bayesian network classifier (BNC) having an I-map structure with the fewest class variable parameters among all structures for which the class variable has no parent. Moreover, a new learning algorithm to learn our proposed model is presented. The proposed method is guaranteed to obtain the true classification probability asymptotically. Moreover, the method has lower computational costs than those of exact learning BNC using marginal likelihood. Comparison experiments have demonstrated the superior performance of the proposed method.

----

## [2291] Bayesian Inference with Complex Knowledge Graph Evidence

**Authors**: *Armin Toroghi, Scott Sanner*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30040](https://doi.org/10.1609/aaai.v38i18.30040)

**Abstract**:

Knowledge Graphs (KGs) provide a widely used format for representing entities and their relationships and have found use in diverse applications including question answering and recommendation.  A majority of current research on KG inference has focused on reasoning with atomic facts (triples) and has disregarded the possibility of making complex evidential observations involving logical operators (negation, conjunction, disjunction) and quantifiers (existential, universal).  Further, while the application of complex evidence has been explored in KG-based query answering (KGQA) research, in many practical online settings, observations are made sequentially.  For example, in KGQA, additional context may be incrementally suggested to narrow down the answer.  Or in interactive recommendation, user critiques may be expressed sequentially in order to narrow down a set of preferred items.  Both settings are indicative of information filtering or tracking tasks that are reminiscent of belief tracking in Bayesian inference.  In fact, in this paper, we precisely cast the problem of belief tracking over unknown KG entities given incremental complex KG evidence as a Bayesian filtering problem.  Specifically, we leverage Knowledge-based Model Construction (KBMC) over the logical KG evidence to instantiate a Markov Random Field (MRF) likelihood representation to perform closed-form Bayesian inference with complex KG evidence (BIKG).  We experimentally evaluate BIKG in incremental KGQA and interactive recommendation tasks demonstrating that it outperforms non-incremental methodologies and leads to better incorporation of conjunctive evidence vs. existing complex KGQA methods like CQD that leverage fuzzy T-norm operators. Overall, this work demonstrates a novel, efficient, and unified perspective of logic, KGs, and online inference through the lens of closed-form BIKG.

----

## [2292] Exact, Fast and Expressive Poisson Point Processes via Squared Neural Families

**Authors**: *Russell Tsuchida, Cheng Soon Ong, Dino Sejdinovic*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30041](https://doi.org/10.1609/aaai.v38i18.30041)

**Abstract**:

We introduce squared neural Poisson point processes (SNEPPPs) by parameterising the intensity function by the squared norm of a two layer neural network.
When the hidden layer is fixed and the second layer has a single neuron, our approach resembles previous uses of squared Gaussian process or kernel methods, but allowing the hidden layer to be learnt allows for additional flexibility.
In many cases of interest, the integrated intensity function admits a closed form and can be computed in quadratic time in the number of hidden neurons.
We enumerate a far more extensive number of such cases than has previously been discussed.
Our approach is more memory and time efficient than naive implementations of squared or exponentiated kernel methods or Gaussian processes.
Maximum likelihood and maximum a posteriori estimates in a reparameterisation of the final layer of the intensity function can be obtained by solving a (strongly) convex optimisation problem using projected gradient descent. 
We demonstrate SNEPPPs on real, and synthetic benchmarks, and provide a software implementation.

----

## [2293] Inference and Learning in Dynamic Decision Networks Using Knowledge Compilation

**Authors**: *Gabriele Venturato, Vincent Derkinderen, Pedro Zuidberg Dos Martires, Luc De Raedt*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30042](https://doi.org/10.1609/aaai.v38i18.30042)

**Abstract**:

Decision making under uncertainty in dynamic environments is a fundamental AI problem in which agents need to determine which decisions (or actions) to make at each time step to maximise their expected utility. Dynamic decision networks (DDNs) are an extension of dynamic Bayesian networks with decisions and utilities. DDNs can be used to compactly represent Markov decision processes (MDPs). We propose a novel algorithm called mapl-cirup that leverages knowledge compilation techniques developed for (dynamic) Bayesian networks to perform inference and gradient-based learning in DDNs. Specifically, we knowledge-compile the Bellman update present in DDNs into dynamic decision circuits and evaluate them within an (algebraic) model counting framework. In contrast to other exact symbolic MDP approaches, we obtain differentiable circuits that enable gradient-based parameter learning.

----

## [2294] Linear-Time Algorithms for Front-Door Adjustment in Causal Graphs

**Authors**: *Marcel Wienöbst, Benito van der Zander, Maciej Liskiewicz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30043](https://doi.org/10.1609/aaai.v38i18.30043)

**Abstract**:

Causal effect estimation from observational data is a fundamental task in empirical sciences. It becomes particularly challenging when unobserved confounders are involved in a system. This paper focuses on front-door adjustment – a classic technique which, using observed mediators allows to identify causal effects even in the presence of unobserved confounding. While the statistical properties of the front-door estimation are quite well understood, its algorithmic aspects remained unexplored for a long time. In 2022, Jeong, Tian, and Bareinboim presented the first polynomial-time algorithm for finding sets satisfying the front-door criterion in a given directed acyclic graph (DAG), with an O(n³(n+m)) run time, where n denotes the number of variables and m the number of edges of the causal graph. In our work, we give the first linear-time, i.e., O(n+m), algorithm for this task, which thus reaches the asymptotically optimal time complexity. This result implies an O(n(n+m)) delay enumeration algorithm of all front-door adjustment sets, again improving previous work by a factor of n³. Moreover, we provide the first linear-time algorithm for finding a minimal front-door adjustment set. We offer implementations of our algorithms in multiple programming languages to facilitate practical usage and empirically validate their feasibility, even for large graphs.

----

## [2295] Neural Causal Abstractions

**Authors**: *Kevin Xia, Elias Bareinboim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30044](https://doi.org/10.1609/aaai.v38i18.30044)

**Abstract**:

The ability of humans to understand the world in terms of cause and effect relationships, as well as their ability to compress information into abstract concepts, are two hallmark features of human intelligence. These two topics have been studied in tandem under the theory of causal abstractions, but it is an open problem how to best leverage abstraction theory in real-world causal inference tasks, where the true model is not known, and limited data is available in most practical settings. In this paper, we focus on a family of causal abstractions constructed by clustering variables and their domains, redefining abstractions to be amenable to individual causal distributions. We show that such abstractions can be learned in practice using Neural Causal Models, allowing us to utilize the deep learning toolkit to solve causal tasks (identification, estimation, sampling) at different levels of abstraction granularity. Finally, we show how representation learning can be used to learn abstractions, which we apply in our experiments to scale causal inferences to high dimensional settings such as with image data.

----

## [2296] Federated Contextual Cascading Bandits with Asynchronous Communication and Heterogeneous Users

**Authors**: *Hantao Yang, Xutong Liu, Zhiyong Wang, Hong Xie, John C. S. Lui, Defu Lian, Enhong Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30045](https://doi.org/10.1609/aaai.v38i18.30045)

**Abstract**:

We study the problem of federated contextual combinatorial cascading bandits, where agents collaborate under the coordination of a central server to provide tailored recommendations to users. Existing works consider either a synchronous framework, necessitating full agent participation and global synchronization, or assume user homogeneity with identical behaviors. We overcome these limitations by considering (1) federated agents operating in an asynchronous communication paradigm, where no mandatory synchronization is required and all agents communicate independently with the server, (2) heterogeneous user behaviors, where users can be stratified into latent user clusters, each exhibiting distinct preferences. For this setting, we propose a UCB-type algorithm with delicate communication protocols. Through theoretical analysis, we give sub-linear regret bounds on par with those achieved in the synchronous framework, while incurring only logarithmic communication costs. Empirical evaluation on synthetic and real-world datasets validates our algorithm's superior performance in terms of regrets and communication costs.

----

## [2297] Causal-Driven Skill Prerequisite Structure Discovery

**Authors**: *Shenbao Yu, Yifeng Zeng, Fan Yang, Yinghui Pan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30046](https://doi.org/10.1609/aaai.v38i18.30046)

**Abstract**:

Knowing a prerequisite structure among skills in a subject domain effectively enables several educational applications, including intelligent tutoring systems and curriculum planning. Traditionally, educators or domain experts use intuition to determine the skills' prerequisite relationships, which is time-consuming and prone to fall into the trap of blind spots. In this paper, we focus on inferring the prerequisite structure given access to students' performance on exercises in a subject. Nevertheless, it is challenging since students' mastery of skills can not be directly observed, but can only be estimated, i.e., its latency in nature. To tackle this problem, we propose a causal-driven skill prerequisite structure discovery (CSPS) method in a two-stage learning framework. In the first stage, we learn the skills' correlation relationships presented in the covariance matrix from the student performance data while, through the predicted covariance matrix in the second stage, we consider a heuristic method based on conditional independence tests and standardized partial variance to discover the prerequisite structure. We demonstrate the performance of the new approach with both simulated and real-world data. The experimental results show the effectiveness of the proposed model for identifying the skills' prerequisite structure.

----

## [2298] Deep Copula-Based Survival Analysis for Dependent Censoring with Identifiability Guarantees

**Authors**: *Weijia Zhang, Chun Kai Ling, Xuanhui Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30047](https://doi.org/10.1609/aaai.v38i18.30047)

**Abstract**:

Censoring is the central problem in survival analysis where either the time-to-event (for instance, death), or the time-to censoring (such as loss of follow-up) is observed for each sample. The majority of existing machine learning-based survival analysis methods assume that survival is conditionally independent of censoring given a set of covariates; an assumption that cannot be verified since only marginal distributions is available from the data. The existence of dependent censoring, along with the inherent bias in current estimators has been demonstrated in a variety of applications, accentuating the need for a more nuanced approach. However, existing methods that adjust for dependent censoring require practitioners to specify the ground truth copula. This requirement poses a significant challenge for practical applications, as model misspecification can lead to substantial bias. In this work, we propose a flexible deep learning-based survival analysis method that simultaneously accommodate for dependent censoring and eliminates the requirement for specifying the ground truth copula. We theoretically prove the identifiability of our model under a broad family of copulas and survival distributions. Experiments results from a wide range of datasets demonstrate that our approach successfully discerns the underlying dependency structure and significantly reduces survival estimation bias when compared to existing methods.

----

## [2299] DOGE-Train: Discrete Optimization on GPU with End-to-End Training

**Authors**: *Ahmed Abbas, Paul Swoboda*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30048](https://doi.org/10.1609/aaai.v38i18.30048)

**Abstract**:

We present a fast, scalable, data-driven approach for solving relaxations of 0-1 integer linear programs. We use a combination of graph neural networks (GNN) and a Lagrange decomposition based algorithm. We make the latter differentiable for end-to-end training and use GNNs to predict its algorithmic parameters. This allows to retain the algorithm's theoretical properties including dual feasibility and guaranteed non-decrease in the lower bound while improving it via training. We overcome suboptimal fixed points of the basic solver by additional non-parametric GNN update steps maintaining dual feasibility. For training we use an unsupervised loss. We train on smaller problems and test on larger ones showing strong generalization performance with a GNN comprising only around 10k parameters. Our solver achieves significantly faster performance and better dual objectives than its non-learned version, achieving close to optimal objective values of LP relaxations of very large structured prediction problems and on selected combinatorial ones. In particular, we achieve better objective values than specialized approximate solvers for specific problem classes while retaining their efficiency. Our solver has better any-time performance over a large time period compared to a commercial solver.

----

## [2300] Delegation-Relegation for Boolean Matrix Factorization

**Authors**: *Florent Avellaneda, Roger Villemaire*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30049](https://doi.org/10.1609/aaai.v38i18.30049)

**Abstract**:

The Boolean Matrix Factorization (BMF) problem aims to represent a n×m Boolean matrix as the Boolean product of two matrices of small rank k, where the product is computed using Boolean algebra operations. However, finding a BMF of minimum rank is known to be NP-hard, posing challenges for heuristic algorithms and exact approaches in terms of rank found and computation time, particularly as matrix size or the number of entries equal to 1 grows.
In this paper, we present a new approach to simplifying the matrix to be factorized by reducing the number of 1-entries, which allows to directly recover a Boolean factorization of the original matrix from its simplified version. We introduce two types of simplification: one that performs numerous simplifications without preserving the original rank and another that performs fewer simplifications but guarantees that an optimal BMF on the simplified matrix yields an optimal BMF on the original matrix. Furthermore, our experiments show that our approach outperforms existing exact BMF algorithms.

----

## [2301] An Interpretable Approach to the Solutions of High-Dimensional Partial Differential Equations

**Authors**: *Lulu Cao, Yufei Liu, Zhenzhong Wang, Dejun Xu, Kai Ye, Kay Chen Tan, Min Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30050](https://doi.org/10.1609/aaai.v38i18.30050)

**Abstract**:

In recent years, machine learning algorithms, especially deep learning, have shown promising prospects in solving Partial Differential Equations (PDEs). However, as the dimension increases, the relationship and interaction between variables become more complex, and existing methods are difficult to provide fast and interpretable solutions for high-dimensional PDEs. To address this issue, we propose a genetic programming symbolic regression algorithm based on transfer learning and automatic differentiation to solve PDEs. This method uses genetic programming to search for a mathematically understandable expression and combines automatic differentiation to determine whether the search result satisfies the PDE and boundary conditions to be solved. To overcome the problem of slow solution speed caused by large search space, we propose a transfer learning mechanism that transfers the structure of one-dimensional PDE analytical solution to the form of high-dimensional PDE solution. We tested three representative types of PDEs, and the results showed that our proposed method can obtain reliable and human-understandable real solutions or algebraic equivalent solutions of PDEs, and the convergence speed is better than the compared methods. Code of this project is at https://github.com/grassdeerdeer/HD-TLGP.

----

## [2302] Sampling for Beyond-Worst-Case Online Ranking

**Authors**: *Qingyun Chen, Sungjin Im, Benjamin Moseley, Chenyang Xu, Ruilong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30051](https://doi.org/10.1609/aaai.v38i18.30051)

**Abstract**:

The feedback arc set problem is one of the most fundamental and well-studied ranking problems where n objects are to be ordered based on their pairwise comparison.  The problem enjoys several efficient approximation algorithms in the offline setting. Unfortunately, online there are strong lower bounds on the competitive ratio establishing that no algorithm can perform well in the worst case.
This paper introduces a new beyond-worst-case model for online feedback arc set. In the model, a sample of the input is given to the algorithm offline before the remaining instance is revealed online.  This models the case in practice where yesterday's data is available and is similar to today's online instance. This sample is drawn from a known distribution which may not be uniform.  We design an online algorithm with strong theoretical guarantees.  The algorithm has a small constant competitive ratio when the sample is uniform---if not, we show we can recover the same result by adding a provably minimal sample. 
Empirical results validate the theory and show that such algorithms can be used on temporal data to obtain strong results.

----

## [2303] Learning Ultrametric Trees for Optimal Transport Regression

**Authors**: *Samantha Chen, Puoya Tabaghi, Yusu Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30052](https://doi.org/10.1609/aaai.v38i18.30052)

**Abstract**:

Optimal transport provides a metric which quantifies the dissimilarity between probability measures. For measures supported in discrete metric spaces, finding the optimal transport distance has cubic time complexity in the size of the space. However, measures supported on trees admit a closed-form optimal transport that can be computed in linear time. In this paper, we aim to find an optimal tree structure for a given discrete metric space so that the tree-Wasserstein distance approximates the optimal transport distance in the original space. One of our key ideas is to cast the problem in ultrametric spaces. This helps us optimize over the space of ultrametric trees --- a mixed-discrete and continuous optimization problem --- via projected gradient decent over the space of ultrametric matrices. During optimization, we project the parameters to the ultrametric space via a hierarchical minimum spanning tree algorithm, equivalent to the closest projection to ultrametrics under the supremum norm. Experimental results on real datasets show that our approach outperforms previous approaches (e.g. Flowtree, Quadtree) in approximating optimal transport distances. Finally, experiments on synthetic data generated on ground truth trees show that our algorithm can accurately uncover the underlying trees.

----

## [2304] Parameterized Approximation Algorithms for Sum of Radii Clustering and Variants

**Authors**: *Xianrun Chen, Dachuan Xu, Yicheng Xu, Yong Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30053](https://doi.org/10.1609/aaai.v38i18.30053)

**Abstract**:

Clustering is one of the most fundamental tools in artificial intelligence, machine learning, and data mining. In this paper, we follow one of the recent mainstream topics of clustering, Sum of Radii (SoR), which naturally arises as a balance between the folklore k-center and k-median. SoR aims to determine a set of k balls, each centered at a point in a given dataset, such that their union covers the entire dataset while minimizing the sum of radii of the k balls. 
We propose a general technical framework to overcome the challenge posed by varying radii in SoR, which yields fixed-parameter tractable (fpt) algorithms with respect to k (i.e., whose running time is f(k) ploy(n) for some f). 
Our framework is versatile and obtains fpt approximation algorithms with constant approximation ratios for SoR as well as its variants in general metrics, such as Fair SoR and Matroid SoR, which significantly improve the previous results.

----

## [2305] Traffic Flow Optimisation for Lifelong Multi-Agent Path Finding

**Authors**: *Zhe Chen, Daniel Harabor, Jiaoyang Li, Peter J. Stuckey*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30054](https://doi.org/10.1609/aaai.v38i18.30054)

**Abstract**:

Multi-Agent Path Finding (MAPF) is a fundamental problem in robotics that asks us to compute collision-free paths for a team of agents, all moving across a shared map. Although many works appear on this topic, all current algorithms struggle as the number of agents grows.  The principal reason is that existing approaches typically plan free-flow optimal paths, which creates congestion. To tackle this issue, we propose a new approach for MAPF where agents are guided to their destination by following congestion-avoiding paths. We evaluate the idea in two large-scale settings: one-shot MAPF, where each agent has a single destination, and lifelong MAPF, where agents are continuously assigned new destinations. Empirically, we report large improvements in solution quality for one-short MAPF and in overall throughput for lifelong MAPF.

----

## [2306] Runtime Analysis of the (μ + 1) GA: Provable Speed-Ups from Strong Drift towards Diverse Populations

**Authors**: *Benjamin Doerr, Aymen Echarghaoui, Mohammed Jamal, Martin S. Krejca*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30055](https://doi.org/10.1609/aaai.v38i18.30055)

**Abstract**:

Most evolutionary algorithms used in practice heavily employ crossover. In contrast, the rigorous understanding of how crossover is beneficial is largely lagging behind. In this work, we make a considerable step forward by analyzing the population dynamics of the (µ+1) genetic algorithm when optimizing the Jump benchmark. We observe (and prove via mathematical means) that once the population contains two different individuals on the local optimum, the diversity in the population increases in expectation. From this drift towards more diverse states, we show that a diversity suitable for crossover to be effective is reached quickly and, more importantly, then persists for a time that is at least exponential in the population size µ. This drastically improves over the previously best known guarantee, which is only quadratic in µ.

Our new understanding of the population dynamics easily gives stronger performance guarantees. In particular, we derive that population sizes logarithmic in the problem size n suffice to gain an Ω(n)-factor runtime improvement from crossover (previous works achieved comparable bounds only with µ = Θ(n) or a non-standard mutation rate).

----

## [2307] Novelty vs Potential Heuristics: A Comparison of Hardness Measures for Satisficing Planning

**Authors**: *Simon Dold, Malte Helmert*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30056](https://doi.org/10.1609/aaai.v38i18.30056)

**Abstract**:

Classical planning considers a given task and searches for a plan to solve it. Some tasks are harder to solve than others. We can measure the 'hardness' of a task with the novelty width and the correlation complexity. In this work, we compare these measures.
Additionally, we introduce the river measure, a new measure that is based on potential heuristics and therefore similar to the correlation complexity but also comparable to the novelty width.
We show that the river measure is upper bounded by the correlation complexity and by the novelty width +1. 
Furthermore, we show that we can convert a planning task with a polynomial blowup of the task size to ensure that a heuristic of dimension 2 exists that gives rise to backtrack-free search.

----

## [2308] Cumulative Regret Analysis of the Piyavskii-Shubert Algorithm and Its Variants for Global Optimization

**Authors**: *Kaan Gökcesu, Hakan Gökcesu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30057](https://doi.org/10.1609/aaai.v38i18.30057)

**Abstract**:

We study the problem of global optimization, where we analyze the performance of the Piyavskii--Shubert algorithm and its variants. For any given time duration T, instead of the extensively studied simple regret (which is the difference of the losses between the best estimate up to T and the global minimum), we study the cumulative regret up to time T. For L-Lipschitz continuous functions, we show that the cumulative regret is O(L logT). For H-Lipschitz smooth functions, we show that the cumulative regret is O(H). We analytically extend our results for functions with Hölder continuous derivatives, which cover both the Lipschitz continuous and the Lipschitz smooth functions, individually. We further show that a simpler variant of the Piyavskii-Shubert algorithm performs just as well as the traditional variants for the Lipschitz continuous or the Lipschitz smooth functions. We further extend our results to broader classes of functions, and show that, our algorithm efficiently determines its queries; and achieves nearly minimax optimal (up to log factors) cumulative regret, for general convex or even concave regularity conditions on the extrema of the objective (which encompasses many preceding regularities). We consider further extensions by investigating the performance of the Piyavskii-Shubert variants in the scenarios with unknown regularity, noisy evaluation and multivariate domain.

----

## [2309] Efficient Constrained K-center Clustering with Background Knowledge

**Authors**: *Longkun Guo, Chaoqi Jia, Kewen Liao, Zhigang Lu, Minhui Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30058](https://doi.org/10.1609/aaai.v38i18.30058)

**Abstract**:

Center-based clustering has attracted significant research interest from both theory and practice. In many practical applications, input data often contain background knowledge that can be used to improve clustering results. In this work, we build on widely adopted k-center clustering and model its input background knowledge as must-link (ML) and cannot-link (CL) constraint sets. However, most clustering problems including k-center are inherently NP-hard, while the more complex constrained variants are known to suffer severer approximation and computation barriers that significantly limit their applicability. By employing a suite of techniques including reverse dominating sets, linear programming (LP) integral polyhedron, and LP duality, we arrive at the first efficient approximation algorithm for constrained k-center with the best possible ratio of 2. We also construct competitive baseline algorithms and empirically evaluate our approximation algorithm against them on a variety of real datasets. The results validate our theoretical findings and demonstrate the great advantages of our algorithm in terms of clustering cost, clustering quality, and running time.

----

## [2310] Limited Query Graph Connectivity Test

**Authors**: *Mingyu Guo, Jialiang Li, Aneta Neumann, Frank Neumann, Hung X. Nguyen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30059](https://doi.org/10.1609/aaai.v38i18.30059)

**Abstract**:

We propose a combinatorial optimisation model called Limited Query Graph Connectivity Test. We consider a graph whose edges have two possible states (On/Off). The edges' states are hidden initially. We could query an edge to reveal its state. Given a source s and a destination t, we aim to test s−t connectivity by identifying either a path (consisting of only On edges) or a cut (consisting of only Off edges). We are limited to B queries, after which we stop regardless of whether graph connectivity is established. We aim to design a query policy that minimizes the expected number of queries.

Our model is mainly motivated by a cyber security use case where we need to establish whether attack paths exist in a given network, between a source (i.e., a compromised user node) and a destination (i.e., a high-privilege admin node).  Edge query is resolved by manual effort from the IT admin, which is the motivation behind query minimization.

Our model is highly related to Stochastic Boolean Function Evaluation (SBFE).  There are two existing exact algorithms for SBFE that are prohibitively expensive. We propose a signifcantly more scalable exact algorithm. While previous exact algorithms only scale for trivial graphs (i.e., past works experimented on at most 20 edges), we empirically demonstrate that our algorithm is scalable for a wide range of much larger practical graphs (i.e., graphs representing Windows domain networks with tens of thousands of edges).

We also propose three heuristics. Our best-performing heuristic is via limiting the planning horizon of the exact algorithm. The other two are via reinforcement learning (RL) and Monte Carlo tree search (MCTS). We also derive an algorithm for computing the performance lower bound. Experimentally, we show that all our heuristics are near optimal.  The heuristic building on the exact algorithm outperforms all other heuristics, surpassing RL, MCTS and eight existing heuristics ported from SBFE and related literature.

----

## [2311] Theoretical Aspects of Generating Instances with Unique Solutions: Pre-assignment Models for Unique Vertex Cover

**Authors**: *Takashi Horiyama, Yasuaki Kobayashi, Hirotaka Ono, Kazuhisa Seto, Ryu Suzuki*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30060](https://doi.org/10.1609/aaai.v38i18.30060)

**Abstract**:

The uniqueness of an optimal solution to a combinatorial optimization problem attracts many fields of researchers' attention because it has a wide range of applications, it is related to important classes in computational complexity, and the existence of only one solution is often critical for algorithm designs in theory. However, as the authors know, there is no major benchmark set consisting of only instances with unique solutions, and no algorithm generating instances with unique solutions is known; a systematic approach to getting a problem instance guaranteed having a unique solution would be helpful. A possible approach is as follows: Given a problem instance, we specify a small part of a solution in advance so that only one optimal solution meets the specification. This paper formulates such a ``pre-assignment'' approach for the vertex cover problem as a typical combinatorial optimization problem and discusses its computational complexity.
First, we show that the problem is ΣP2-complete in general, while the problem becomes NP-complete when an input graph is bipartite. 
We then present an O(2.1996^n)-time algorithm for general graphs and an O(1.9181^n)-time algorithm for bipartite graphs, where n is the number of vertices. The latter is based on an FPT algorithm with O*(3.6791^τ) time for vertex cover number τ. Furthermore, we show that the problem for trees can be solved in O(1.4143^n) time.

----

## [2312] KD-Club: An Efficient Exact Algorithm with New Coloring-Based Upper Bound for the Maximum k-Defective Clique Problem

**Authors**: *Mingming Jin, Jiongzhi Zheng, Kun He*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30061](https://doi.org/10.1609/aaai.v38i18.30061)

**Abstract**:

The Maximum k-Defective Clique Problem (MDCP) aims to find a maximum k-defective clique in a given graph, where a k-defective clique is a relaxation clique missing at most k edges. MDCP is NP-hard and finds many real-world applications in analyzing dense but not necessarily complete subgraphs. Exact algorithms for MDCP mainly follow the Branch-and-bound (BnB) framework, whose performance heavily depends on the quality of the upper bound on the cardinality of a maximum k-defective clique. The state-of-the-art BnB MDCP algorithms calculate the upper bound quickly but conservatively as they ignore many possible missing edges. In this paper, we propose a novel CoLoring-based Upper Bound (CLUB) that uses graph coloring techniques to detect independent sets so as to detect missing edges ignored by the previous methods. We then develop a new BnB algorithm for MDCP, called KD-Club, using CLUB in both the preprocessing stage for graph reduction and the BnB searching process for branch pruning. Extensive experiments show that KD-Club significantly outperforms state-of-the-art BnB MDCP algorithms on the number of solved instances within the cut-off time, having much smaller search tree and shorter solving time on various benchmarks.

----

## [2313] Parallel Beam Search Algorithms for Domain-Independent Dynamic Programming

**Authors**: *Ryo Kuroiwa, J. Christopher Beck*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30062](https://doi.org/10.1609/aaai.v38i18.30062)

**Abstract**:

Domain-independent dynamic programming (DIDP), a model-based paradigm based on dynamic programming, has shown promising performance on multiple combinatorial optimization problems compared with mixed integer programming (MIP) and constraint programming (CP). The current DIDP solvers are based on heuristic search, and the state-of-the-art solver, complete anytime beam search (CABS), uses beam search. However, the current DIDP solvers cannot utilize multiple threads, unlike state-of-the-art MIP and CP solvers. In this paper, we propose three parallel beam search algorithms and develop multi-thread implementations of CABS. With 32 threads, our multi-thread DIDP solvers achieve 9 to 39 times speedup on average and significant performance improvement over the sequential solver, finding the new best solutions for two instances of the traveling salesperson problem with time windows. In addition, our solvers outperform multi-thread MIP and CP solvers in four of the six combinatorial optimization problems evaluated.

----

## [2314] Rectangle Search: An Anytime Beam Search

**Authors**: *Sofia Lemons, Wheeler Ruml, Robert C. Holte, Carlos Linares López*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30063](https://doi.org/10.1609/aaai.v38i18.30063)

**Abstract**:

Anytime heuristic search algorithms try to find a (potentially suboptimal) solution as quickly as possible and then work to find better and better solutions until an optimal solution is obtained or time is exhausted. The most widely-known anytime search algorithms are based on best-first search.  In this paper, we propose a new algorithm, rectangle search, that is instead based on beam search, a variant of breadth-first search.  It repeatedly explores alternatives at all depth levels and is thus best-suited to problems featuring deep local minima.  Experiments using a variety of popular search benchmarks suggest that rectangle search is competitive with fixed-width beam search and often performs better than the previous best anytime search algorithms.

----

## [2315] Learning to Stop Cut Generation for Efficient Mixed-Integer Linear Programming

**Authors**: *Haotian Ling, Zhihai Wang, Jie Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30064](https://doi.org/10.1609/aaai.v38i18.30064)

**Abstract**:

Cutting planes (cuts) play an important role in solving mixed-integer linear programs (MILPs), as they significantly tighten the dual bounds and improve the solving performance. A key problem for cuts is when to stop cuts generation, which is important for the efficiency of solving MILPs. However, many modern MILP solvers employ hard-coded heuristics to tackle this problem, which tends to neglect underlying patterns among MILPs from certain applications. To address this challenge, we formulate the cuts generation stopping problem as a reinforcement learning problem and propose a novel hybrid graph representation model (HYGRO) to learn effective stopping strategies. An appealing feature of HYGRO is that it can effectively capture both the dynamic and static features of MILPs, enabling dynamic decision-making for the stopping strategies. To the best of our knowledge, HYGRO is the first data-driven method to tackle the cuts generation stopping problem. By integrating our approach with modern solvers, experiments demonstrate that HYGRO significantly improves the efficiency of solving MILPs compared to competitive baselines, achieving up to 31% improvement.

----

## [2316] A Fast Exact Solver with Theoretical Analysis for the Maximum Edge-Weighted Clique Problem

**Authors**: *Lu Liu, Mingyu Xiao, Yi Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30065](https://doi.org/10.1609/aaai.v38i18.30065)

**Abstract**:

The maximum vertex-weighted clique problem (MVWCP) and the maximum edge-weighted clique problem (MEWCP) are two natural extensions of the fundamental maximum clique problem. 
In this paper, we systematically study MEWCP and make the following major contributions:
(1) We show that MEWCP is NP-hard even when the minimum degree of the graph is n-2, in contrast to MVWCP which is polynomial-time solvable when the minimum degree of the graph is at least n-3. This result distinguishes the complexity of the two problems for the first time.
(2) To address MEWCP, we develop an efficient branch-and-bound algorithm called MEWCat with both practical and theoretical performance guarantees. In practice, MEWCat utilizes a new upper bound tighter than existing ones, which allows for more efficient pruning of branches. In theory, we prove a running-time bound of O*(1.4423^n) for MEWCat, which breaks the trivial bound of O*(2^n) in the research line of practical exact MEWCP solvers for the first time.
(3) Empirically, we evaluate the performance of MEWCat on various benchmark instances. The experiments demonstrate that MEWCat outperforms state-of-the-art exact solvers significantly. For instance, on 16 DIMACS graphs that the state-of-the-art solver BBEWC fails to solve within 7200 seconds, MEWCat solves all of them with an average time of less than 1000 seconds. On real-world graphs, MEWCat achieves an average speedup of over 36x.

----

## [2317] Towards Running Time Analysis of Interactive Multi-Objective Evolutionary Algorithms

**Authors**: *Tianhao Lu, Chao Bian, Chao Qian*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30066](https://doi.org/10.1609/aaai.v38i18.30066)

**Abstract**:

Evolutionary algorithms (EAs) are widely used for multi-objective optimization due to their population-based nature. Traditional multi-objective EAs (MOEAs) generate a large set of solutions to approximate the Pareto front, leaving a decision maker (DM) with the task of selecting a preferred solution. However, this process can be inefficient and time-consuming, especially when there are many objectives or the DM has subjective preferences. To address this issue, interactive MOEAs (iMOEAs) combine decision making into the optimization process, i.e., update the population with the help of the DM. In contrast to their wide applications, there has existed only two pieces of theoretical works on iMOEAs, which only considered interactive variants of the two simple single-objective algorithms, RLS and (1+1)-EA. This paper provides the first running time analysis (the essential theoretical aspect of EAs) for practical iMOEAs. Specifically, we prove that the expected running time of the well-developed interactive NSGA-II (called R-NSGA-II) for solving the OneMinMax, OneJumpZeroJump problems are all asymptotically faster than the traditional NSGA-II. Meanwhile, we present a variant of OneMinMax, and prove that R-NSGA-II can be exponentially slower than NSGA-II. These results provide theoretical justification for the effectiveness of iMOEAs while identifying situations where they may fail. Experiments are also conducted to validate the theoretical results.

----

## [2318] Accelerating Cutting-Plane Algorithms via Reinforcement Learning Surrogates

**Authors**: *Kyle Mana, Fernando Acero, Stephen Mak, Parisa Zehtabi, Michael Cashmore, Daniele Magazzeni, Manuela Veloso*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30067](https://doi.org/10.1609/aaai.v38i18.30067)

**Abstract**:

Discrete optimization belongs to the set of N P-hard
problems, spanning fields such as mixed-integer
programming and combinatorial optimization. A current
standard approach to solving convex discrete optimization
problems is the use of cutting-plane algorithms, which
reach optimal solutions by iteratively adding inequalities
known as cuts to refine a feasible set. Despite the existence
of a number of general-purpose cut-generating algorithms,
large-scale discrete optimization problems continue to suffer
from intractability. In this work, we propose a method for
accelerating cutting-plane algorithms via reinforcement
learning. Our approach uses learned policies as surrogates
for N P-hard elements of the cut generating procedure
in a way that (i) accelerates convergence, and (ii) retains
guarantees of optimality. We apply our method on two types
of problems where cutting-plane algorithms are commonly
used: stochastic optimization, and mixed-integer quadratic
programming. We observe the benefits of our method when
applied to Benders decomposition (stochastic optimization)
and iterative loss approximation (quadratic programming),
achieving up to 45% faster average convergence when
compared to modern alternative algorithms.

----

## [2319] Paths, Proofs, and Perfection: Developing a Human-Interpretable Proof System for Constrained Shortest Paths

**Authors**: *Konstantin Sidorov, Gonçalo Homem De Almeida Correia, Mathijs de Weerdt, Emir Demirovic*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30068](https://doi.org/10.1609/aaai.v38i18.30068)

**Abstract**:

People want to rely on optimization algorithms for complex decisions but verifying the optimality of the solutions can then become a valid concern, particularly for critical decisions taken by non-experts in optimization. One example is the shortest-path problem on a network, occurring in many contexts from transportation to logistics to telecommunications. While the standard shortest-path problem is both solvable in polynomial time and certifiable by duality, introducing side constraints makes solving and certifying the solutions much harder. We propose a proof system for constrained shortest-path problems, which gives a set of logical rules to derive new facts about feasible solutions. The key trait of the proposed proof system is that it specifically includes high-level graph concepts within its reasoning steps (such as connectivity or path structure), in contrast to, e.g., using linear combinations of model constraints. Thus, using our proof system, we can provide a step-by-step, human-auditable explanation showing that the path given by an external solver cannot be improved. Additionally, to maximize the advantages of this setup, we propose a proof search procedure that specifically aims to find small proofs of this form using a procedure similar to A* search. We evaluate our proof system on constrained shortest path instances generated from real-world road networks and experimentally show that we may indeed derive more interpretable proofs compared to an integer programming approach, in some cases leading to much smaller proofs.

----

## [2320] Learning Encodings for Constructive Neural Combinatorial Optimization Needs to Regret

**Authors**: *Rui Sun, Zhi Zheng, Zhenkun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30069](https://doi.org/10.1609/aaai.v38i18.30069)

**Abstract**:

Deep-reinforcement-learning (DRL) based neural combinatorial optimization (NCO) methods have demonstrated efficiency without relying on the guidance of optimal solutions. As the most mainstream among them, the learning constructive heuristic (LCH) achieves high-quality solutions through a rapid autoregressive solution construction process. However, these LCH-based methods are deficient in convergency, and there is still a performance gap compared to the optimal. Intuitively, learning to regret some steps in the solution construction process is helpful to the training efficiency and network representations. This article proposes a novel regret-based mechanism for an advanced solution construction process. Our method can be applied as a plug-in to any existing LCH-based DRL-NCO method. Experimental results demonstrate the capability of our work to enhance the performance of various NCO models. Results also show that the proposed LCH-Regret outperforms the previous modification methods on several typical combinatorial optimization problems. The code and Supplementary File are available at https://github.com/SunnyR7/LCH-Regret.

----

## [2321] COMBHelper: A Neural Approach to Reduce Search Space for Graph Combinatorial Problems

**Authors**: *Hao Tian, Sourav Medya, Wei Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30070](https://doi.org/10.1609/aaai.v38i18.30070)

**Abstract**:

Combinatorial Optimization (CO) problems over graphs appear routinely in many applications such as in optimizing traffic, viral marketing in social networks, and matching for job allocation. Due to their combinatorial nature, these problems are often NP-hard. Existing approximation algorithms and heuristics rely on the search space to find the solutions and become time-consuming when this space is large. In this paper, we design a neural method called COMBHelper to reduce this space and thus improve the efficiency of the traditional CO algorithms based on node selection. Specifically, it employs a Graph Neural Network (GNN) to identify promising nodes for the solution set. This pruned search space is then fed to the traditional CO algorithms. COMBHelper also uses a Knowledge Distillation (KD) module and a problem-specific boosting module to bring further efficiency and efficacy. Our extensive experiments show that the traditional CO algorithms with COMBHelper are at least 2 times faster than their original versions.

----

## [2322] Improving Neural Network Generalization on Data-Limited Regression with Doubly-Robust Boosting

**Authors**: *Hao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30071](https://doi.org/10.1609/aaai.v38i18.30071)

**Abstract**:

Enhancing the generalization performance of neural networks given limited data availability remains a formidable challenge, due to the model selection trade-off between training error and generalization gap.
To handle this challenge, we present a posterior optimization issue, specifically designed to reduce the generalization error of trained neural networks.
To operationalize this concept, we propose a Doubly-Robust Boosting machine (DRBoost) which consists of a statistical learner and a zero-order optimizer.
The statistical learner reduces the model capacity and thus the generalization gap; the zero-order optimizer minimizes the training error in a gradient-free manner. The two components cooperate to reduce the generalization error of a fully trained neural network in a doubly robust manner.
Furthermore, the statistical learner alleviates the multicollinearity in the discriminative layer and enhances the generalization performance.
The zero-order optimizer eliminates the reliance on gradient calculation and offers more flexibility in learning objective selection.
Experiments demonstrate that DRBoost improves the generalization performance of various prevalent neural network backbones effectively.

----

## [2323] Inertial Algorithm with Dry Fraction and Convolutional Sparse Coding for 3D Localization with Light Field Microscopy

**Authors**: *Xiaofan Wang, Zhiyuan Deng, Changle Wang, Jinjia Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30072](https://doi.org/10.1609/aaai.v38i18.30072)

**Abstract**:

Light field microscopy is a high-speed 3D imaging technique that records the light field from multiple angles by the microlens array(MLA), thus allowing us to obtain information about the light source from a single image only. For the fundamental problem of neuron localization, we improve the method of combining depth-dependent dictionary with sparse coding in this paper. In order to obtain higher localization accuracy and good noise immunity, we propose an inertial proximal gradient acceleration algorithm with dry friction, Fast-IPGDF. By preventing falling into a local minimum, our algorithm achieves better convergence and converges quite fast, which improves the speed and accuracy of obtaining the locolization of the light source based on the matching depth of epipolar plane images (EPI). We demonstrate the effectiveness of the algorithm for localizing non-scattered fluorescent beads in both noisy and non-noisy environments. The experimental results show that our method can achieve simultaneous localization of multiple point sources and effective localization in noisy environments. Compared to existing studies, our method shows significant improvements in both localization accuracy and speed.

----

## [2324] A Novel Skip Orthogonal List for Dynamic Optimal Transport Problem

**Authors**: *Xiaoyang Xu, Hu Ding*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30073](https://doi.org/10.1609/aaai.v38i18.30073)

**Abstract**:

Optimal transport is a fundamental topic that has attracted a great amount of attention from the optimization community in the past decades. In this paper, we consider an interesting discrete dynamic optimal transport problem: can we efficiently update the optimal transport plan when the weights or the locations of the data points change? This problem is naturally motivated by several applications in machine learning. For example, we often need to compute the optimal transport cost between two different data sets; if some changes happen to a few data points, should we re-compute the high complexity cost function or update the cost by some efficient dynamic data structure? We are aware that several dynamic maximum flow algorithms have been proposed before, however, the research on dynamic minimum cost flow problem is still quite limited, to the best of our knowledge. We propose a novel 2D Skip Orthogonal List together with some dynamic tree techniques. Although our algorithm is based on the conventional simplex method, it can efficiently find the variable to pivot within expected O(1) time, and complete each pivoting operation within expected O(|V|) time where V is the set of all supply and demand nodes. Since dynamic modifications typically do not introduce significant changes, our algorithm requires only a few simplex iterations in practice. So our algorithm is more efficient than re-computing the optimal transport cost that needs at least one traversal over all |E|=O(|V|^2) variables, where |E| denotes the number of edges in the network. Our experiments demonstrate that our algorithm significantly outperforms existing algorithms in the dynamic scenarios.

----

## [2325] Sample-and-Bound for Non-convex Optimization

**Authors**: *Yaoguang Zhai, Zhizhen Qin, Sicun Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30074](https://doi.org/10.1609/aaai.v38i18.30074)

**Abstract**:

Standard approaches for global optimization of non-convex functions, such as branch-and-bound, maintain partition trees to systematically prune the domain. The tree size grows exponentially in the number of dimensions. We propose new sampling-based methods for non-convex optimization that adapts Monte Carlo Tree Search (MCTS) to improve efficiency. Instead of the standard use of visitation count in Upper Confidence Bounds, we utilize numerical overapproximations of the objective as an uncertainty metric, and also take into account of sampled estimates of first-order and second-order information. The Monte Carlo tree in our approach avoids the usual fixed combinatorial patterns in growing the tree, and aggressively zooms into the promising regions, while still balancing exploration and exploitation. We evaluate the proposed algorithms on high-dimensional non-convex optimization benchmarks against competitive baselines and analyze the effects of the hyper parameters.

----

## [2326] Threshold-Based Responsive Simulated Annealing for Directed Feedback Vertex Set Problem

**Authors**: *Qingyun Zhang, Yuming Du, Zhouxing Su, Chu-Min Li, Junzhou Xu, Zhihuai Chen, Zhipeng Lü*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30075](https://doi.org/10.1609/aaai.v38i18.30075)

**Abstract**:

As a classical NP-hard problem and the topic of the PACE 2022 competition, the directed feedback vertex set problem (DFVSP) aims to find a minimum subset of vertices such that, when vertices in the subset and all their adjacent edges are removed from the directed graph, the remainder graph is acyclic. In this paper, we propose a threshold-based responsive simulated annealing algorithm called TRSA for solving DFVSP. First, we simplify the problem instances with two new reduction rules proposed in this paper and eight reduction rules from the literature. Then, based on a new solution representation, TRSA solves DFVSP with a fast local search procedure featured by a swap-based neighborhood structure and three neighborhood acceleration strategies. Finally, all these strategies are incorporated into a threshold-based responsive simulated annealing framework. Computational experiments on 140 benchmark instances show that TRSA is highly competitive compared to the state-of-the-art methods. Specifically, TRSA can improve the best known results for 53 instances, while matching the best known results for 79 ones. Furthermore, some important features of TRSA are analyzed to identify its success factors.

----

## [2327] Jointly Improving the Sample and Communication Complexities in Decentralized Stochastic Minimax Optimization

**Authors**: *Xuan Zhang, Gabriel Mancino-Ball, Necdet Serhat Aybat, Yangyang Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30076](https://doi.org/10.1609/aaai.v38i18.30076)

**Abstract**:

We propose a novel single-loop decentralized algorithm, DGDA-VR, for solving the stochastic nonconvex strongly-concave minimax problems over a connected network of agents, which are equipped with stochastic first-order oracles to estimate their local gradients. DGDA-VR, incorporating variance reduction, achieves O(ε^−3) oracle complexity and O(ε^−2) communication complexity without resorting to multi-communication rounds – both are optimal, i.e., matching the lower bounds for this class of problems. Since DGDA-VR does not require multiple communication rounds, it is applicable to a broader range of decentralized computational environments. To the best of our knowledge, this is the first distributed method using a single communication round in each iteration to jointly optimize the oracle and communication complexities for the problem considered here.

----

## [2328] Runtime Analysis of the SMS-EMOA for Many-Objective Optimization

**Authors**: *Weijie Zheng, Benjamin Doerr*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30077](https://doi.org/10.1609/aaai.v38i18.30077)

**Abstract**:

The widely used multiobjective optimizer NSGA-II was recently proven to have considerable difficulties in many-objective optimization. In contrast, experimental results in the literature show a good performance of the SMS-EMOA, which can be seen as a steady-state NSGA-II that uses the hypervolume contribution instead of the crowding distance as the second selection criterion. 

This paper conducts the first rigorous runtime analysis of the SMS-EMOA for many-objective optimization. To this aim, we first propose a many-objective counterpart, the m-objective mOJZJ problem, of the bi-objective OJZJ benchmark, which is the first many-objective multimodal benchmark used in a mathematical runtime analysis. We prove that SMS-EMOA computes the full Pareto front of this benchmark in an expected number of O(M^2 n^k) iterations, where n denotes the problem size (length of the bit-string representation), k the gap size (a difficulty parameter of the problem), and M=(2n/m-2k+3)^(m/2) the size of the Pareto front. This result together with the existing negative result on the original NSGA-II shows that in principle, the general approach of the NSGA-II is suitable for many-objective optimization, but the crowding distance as tie-breaker has deficiencies.

We obtain three additional insights on the SMS-EMOA. Different from a recent result for the bi-objective OJZJ benchmark, the stochastic population update often does not help for mOJZJ. It results in a 1/Θ(min(Mk^(1/2)/2^(k/2),1)) speed-up, which is Θ(1) for large m such as m>k. On the positive side, we prove that heavy-tailed mutation still results in a speed-up of order k^(0.5+k-β). Finally, we conduct the first runtime analyses of the SMS-EMOA on the bi-objective OneMinMax and LOTZ benchmarks and show that it has a performance comparable to the GSEMO and the NSGA-II.

----

## [2329] How to Use the Metropolis Algorithm for Multi-Objective Optimization?

**Authors**: *Weijie Zheng, Mingfeng Li, Renzhong Deng, Benjamin Doerr*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30078](https://doi.org/10.1609/aaai.v38i18.30078)

**Abstract**:

The Metropolis algorithm can cope with local optima by accepting inferior solutions with suitably small probability. That this can work well was not only observed in empirical research, but also via mathematical runtime analyses on single-objective benchmarks. This paper takes several steps towards understanding, again via theoretical means, whether such advantages can also be obtained in multi-objective optimization.

The original Metropolis algorithm has two components, one-bit mutation and the acceptance strategy, which allows accepting inferior solutions. When adjusting the acceptance strategy to multi-objective optimization in the way that an inferior solution that is accepted replaces its parent, then the Metropolis algorithm is not very efficient on our multi-objective version of the multimodal DLB benchmark called DLTB. With one-bit mutation, this multi-objective Metropolis algorithm cannot optimize the DLTB problem, with standard bit-wise mutation it needs at least Ω(n^5) time to cover the full Pareto front. In contrast, we show that many other multi-objective optimizers, namely the GSEMO, SMS-EMOA, and NSGA-II, only need time O(n^4).

When keeping the parent when an inferior point is accepted, the multi-objective Metropolis algorithm both with one-bit or standard bit-wise mutation solves the DLTB problem efficiently, with one-bit mutation experimentally leading to better results than several other algorithms.

Overall, our work suggests that the general mechanism of the Metropolis algorithm can be interesting in multi-objective optimization, but that the implementation details can have a huge impact on the performance.

----

## [2330] Two-Stage Evolutionary Reinforcement Learning for Enhancing Exploration and Exploitation

**Authors**: *Qingling Zhu, Xiaoqiang Wu, Qiuzhen Lin, Wei-Neng Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i18.30079](https://doi.org/10.1609/aaai.v38i18.30079)

**Abstract**:

The integration of Evolutionary Algorithm (EA) and Reinforcement Learning (RL) has emerged as a promising approach for tackling some challenges in RL, such as sparse rewards, lack of exploration, and brittle convergence properties. However, existing methods often employ actor networks as individuals of EA, which may constrain their exploratory capabilities, as the entire actor population will stop evolution when the critic network in RL falls into local optimal. To alleviate this issue, this paper introduces a Two-stage Evolutionary Reinforcement Learning (TERL) framework that maintains a population containing both actor and critic networks. TERL divides the learning process into two stages. In the initial stage, individuals independently learn actor-critic networks, which are optimized alternatively by RL and Particle Swarm Optimization (PSO). This dual optimization fosters greater exploration, curbing susceptibility to local optima. Shared information from a common replay buffer and PSO algorithm substantially mitigates the computational load of training multiple agents. In the subsequent stage, TERL shifts to a refined exploitation phase. Here, only the best individual undergoes further refinement, while the rest individuals continue PSO-based optimization. This allocates more computational resources to the best individual for yielding superior performance. Empirical assessments, conducted across a range of continuous control problems, validate the efficacy of the proposed TERL paradigm.

----

## [2331] ImageCaptioner2: Image Captioner for Image Captioning Bias Amplification Assessment

**Authors**: *Eslam Abdelrahman, Pengzhan Sun, Li Erran Li, Mohamed Elhoseiny*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30080](https://doi.org/10.1609/aaai.v38i19.30080)

**Abstract**:

Most pre-trained learning systems are known to suffer from bias, which typically emerges from the data, the model, or both. Measuring and quantifying bias and its sources is a challenging task and has been extensively studied in image
captioning. Despite the significant effort in this direction, we observed that existing metrics lack consistency in the inclusion of the visual signal. In this paper, we introduce a new bias assessment metric, dubbed ImageCaptioner2, for image captioning. Instead of measuring the absolute bias in the model or the data, ImageCaptioner2pay more attention to the bias introduced by the model w.r.t the data bias, termed bias amplification. Unlike the existing methods, which only evaluate the image captioning algorithms based on the generated captions only, ImageCaptioner2incorporates the image while measuring the bias. In addition, we design a formulation for measuring the bias of generated captions as prompt-based image captioning instead of using language classifiers. Finally, we apply our ImageCaptioner2metric across 11 different image captioning architectures on three different datasets, i.e., MS-COCO caption dataset, Artemis V1, and Artemis V2, and on three different protected attributes, i.e., gender, race, and emotions. Consequently, we verify the effectiveness of our ImageCaptioner2metric by proposing Anonymous-Bench, which is a novel human evaluation paradigm for bias metrics. Our metric shows significant superiority over the recent bias metric; LIC, in terms of human alignment, where the correlation scores are 80% and 54% for our metric and LIC, respectively. The code and more details are available at https://eslambakr.github.io/imagecaptioner2.github.io/.

----

## [2332] A Framework for Data-Driven Explainability in Mathematical Optimization

**Authors**: *Kevin-Martin Aigner, Marc Goerigk, Michael Hartisch, Frauke Liers, Arthur Miehlich*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30081](https://doi.org/10.1609/aaai.v38i19.30081)

**Abstract**:

Advancements in mathematical programming have made it possible to efficiently tackle large-scale real-world problems that were deemed intractable just a few decades ago. However, provably optimal solutions may not be accepted due to the perception of optimization software as a black box. Although well understood by scientists, this lacks easy accessibility for practitioners. Hence, we advocate for introducing the explainability of a solution as another evaluation criterion, next to its objective value, which enables us to find trade-off solutions between these two criteria. Explainability is attained by comparing against (not necessarily optimal) solutions that were implemented in similar situations in the past. Thus, solutions are preferred that exhibit similar features. Although we prove that already in simple cases the explainable model is NP-hard, we characterize relevant polynomially solvable cases such as the explainable shortest path problem. Our numerical experiments on both artificial as well as real-world road networks show the resulting Pareto front. It turns out that the cost of enforcing explainability can be very small.

----

## [2333] On the Importance of Application-Grounded Experimental Design for Evaluating Explainable ML Methods

**Authors**: *Kasun Amarasinghe, Kit T. Rodolfa, Sérgio M. Jesus, Valerie Chen, Vladimir Balayan, Pedro Saleiro, Pedro Bizarro, Ameet Talwalkar, Rayid Ghani*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30082](https://doi.org/10.1609/aaai.v38i19.30082)

**Abstract**:

Most existing evaluations of explainable machine learning (ML) methods rely on simplifying assumptions or proxies that do not reflect real-world use cases; the handful of more robust evaluations on real-world settings have shortcomings in their design, generally leading to overestimation of methods' real-world utility.  In this work, we seek to address this by conducting a study that evaluates post-hoc explainable ML methods in a setting consistent with the application context and provide a template for future evaluation studies. We modify and improve a prior study on e-commerce fraud detection by relaxing the original work's simplifying assumptions that departed from the deployment context. Our study finds no evidence for the utility of the tested explainable ML methods in the context, which is a drastically different conclusion from the earlier work. This highlights how seemingly trivial experimental design choices can yield misleading conclusions about method utility. In addition, our work carries lessons about the necessity of not only evaluating explainable ML methods using tasks, data, users, and metrics grounded in the intended application context but also developing methods tailored to specific applications, moving beyond general-purpose explainable ML methods.

----

## [2334] Risk-Aware Continuous Control with Neural Contextual Bandits

**Authors**: *Jose A. Ayala-Romero, Andres Garcia-Saavedra, Xavier Costa-Pérez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30083](https://doi.org/10.1609/aaai.v38i19.30083)

**Abstract**:

Recent advances in learning techniques have garnered attention for their applicability to a diverse range of real-world sequential decision-making problems. Yet, many practical applications have critical constraints for operation in real environments. Most learning solutions often neglect the risk of failing to meet these constraints, hindering their implementation in real-world contexts. In this paper, we propose a risk-aware decision-making framework for contextual bandit problems, accommodating constraints and continuous action spaces. Our approach employs an actor multi-critic architecture, with each critic characterizing the distribution of performance and constraint metrics. Our framework is designed to cater to various risk levels, effectively balancing constraint satisfaction against performance. To demonstrate the effectiveness of our approach, we first compare it against state-of-the-art baseline methods in a synthetic environment, highlighting the impact of intrinsic environmental noise across different risk configurations. Finally, we evaluate our framework in a real-world use case involving a 5G mobile network where only our approach satisfies consistently the system constraint (a signal processing reliability target) with a small performance toll (8.5% increase in power consumption).

----

## [2335] Robust Uncertainty Quantification Using Conformalised Monte Carlo Prediction

**Authors**: *Daniel Bethell, Simos Gerasimou, Radu Calinescu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30084](https://doi.org/10.1609/aaai.v38i19.30084)

**Abstract**:

Deploying deep learning models in safety-critical applications remains a very challenging task, mandating the provision of assurances for the dependable operation of these models. Uncertainty quantification (UQ) methods estimate the model’s confidence per prediction, informing decision-making by considering the effect of randomness and model misspecification. Despite the advances of state-of-the-art UQ methods, they are computationally expensive or produce conservative prediction sets/intervals. We introduce MC-CP, a novel hybrid UQ method that combines a new adaptive Monte Carlo (MC) dropout method with conformal prediction (CP). MC-CP adaptively modulates the traditional MC dropout at runtime to save memory and computation resources, enabling predictions to be consumed by CP, yielding robust prediction sets/intervals. Throughout comprehensive experiments, we show that MC-CP delivers significant improvements over comparable UQ methods, like MC dropout, RAPS and CQR, both in classification and regression benchmarks. MC-CP can be easily added to existing models, making its deployment simple. The MC-CP code and replication package is available at https://github.com/team-daniel/MC-CP.

----

## [2336] CCTR: Calibrating Trajectory Prediction for Uncertainty-Aware Motion Planning in Autonomous Driving

**Authors**: *Chengtai Cao, Xinhong Chen, Jianping Wang, Qun Song, Rui Tan, Yung-Hui Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30085](https://doi.org/10.1609/aaai.v38i19.30085)

**Abstract**:

Autonomous driving systems rely on precise trajectory prediction for safe and efficient motion planning. Despite considerable efforts to enhance prediction accuracy, inherent uncertainties persist due to data noise and incomplete observations. Many strategies entail formalizing prediction outcomes into distributions and utilizing variance to represent uncertainty. However, our experimental investigation reveals that existing trajectory prediction models yield unreliable uncertainty estimates, necessitating additional customized calibration processes. On the other hand, directly applying current calibration techniques to prediction outputs may yield sub-optimal results due to using a universal scaler for all predictions and neglecting informative data cues. In this paper, we propose Customized Calibration Temperature with Regularizer (CCTR), a generic framework that calibrates the output distribution. Specifically, CCTR 1) employs a calibration-based regularizer to align output variance with the discrepancy between prediction and ground truth and 2) generates a tailor-made temperature scaler for each prediction using a post-processing network guided by context and historical information. Extensive evaluation involving multiple prediction and planning methods demonstrates the superiority of CCTR over existing calibration algorithms and uncertainty-aware methods, with significant improvements of 11%-22% in calibration quality and 17%-46% in motion planning.

----

## [2337] Rethinking the Development of Large Language Models from the Causal Perspective: A Legal Text Prediction Case Study

**Authors**: *Haotian Chen, Lingwei Zhang, Yiran Liu, Yang Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30086](https://doi.org/10.1609/aaai.v38i19.30086)

**Abstract**:

While large language models (LLMs) exhibit impressive performance on a wide range of NLP tasks, most of them fail to learn the causality from correlation, which disables them from learning rationales for predicting. Rethinking the whole developing process of LLMs is of great urgency as they are adopted in various critical tasks that need rationales, including legal text prediction (e.g., legal judgment prediction). In this paper, we first explain the underlying theoretical mechanism of their failure and argue that both the data imbalance and the omission of causality in model design and selection render the current training-testing paradigm failed to select the unique causality-based model from correlation-based models. Second, we take the legal text prediction task as the testbed and reconstruct the developing process of LLMs by simultaneously infusing causality into model architectures and organizing causality-based adversarial attacks for evaluation. Specifically, we base our reconstruction on our theoretical analysis and propose a causality-aware self-attention mechanism (CASAM), which prevents LLMs from entangling causal and non-causal information by restricting the interaction between causal and non-causal words. Meanwhile, we propose eight kinds of legal-specific attacks to form causality-based model selection. Our extensive experimental results demonstrate that our proposed CASAM achieves state-of-the-art (SOTA) performances and the strongest robustness on three commonly used legal text prediction benchmarks. We make our code publicly available at https://github.com/Carrot-Red/Rethink-LLM-development.

----

## [2338] Truth Forest: Toward Multi-Scale Truthfulness in Large Language Models through Intervention without Tuning

**Authors**: *Zhongzhi Chen, Xingwu Sun, Xianfeng Jiao, Fengzong Lian, Zhanhui Kang, Di Wang, Chengzhong Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30087](https://doi.org/10.1609/aaai.v38i19.30087)

**Abstract**:

Despite the great success of large language models (LLMs) in various tasks, they suffer from generating hallucinations. We introduce Truth Forest, a method that enhances truthfulness in LLMs by uncovering hidden truth representations using multi-dimensional orthogonal probes. Specifically, it creates multiple orthogonal bases for modeling truth by incorporating orthogonal constraints into the probes. Moreover, we introduce Random Peek, a systematic technique considering an extended range of positions within the sequence, reducing the gap between discerning and generating truth features in LLMs. By employing this approach, we improved the truthfulness of Llama-2-7B from 40.8% to 74.5% on TruthfulQA. Likewise, significant improvements are observed in fine-tuned models. We conducted a thorough analysis of truth features using probes. Our visualization results show that orthogonal probes capture complementary truth-related features, forming well-defined clusters that reveal the inherent structure of the dataset.

----

## [2339] Constrained Meta-Reinforcement Learning for Adaptable Safety Guarantee with Differentiable Convex Programming

**Authors**: *Minjae Cho, Chuangchuang Sun*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30088](https://doi.org/10.1609/aaai.v38i19.30088)

**Abstract**:

Despite remarkable achievements in artificial intelligence, the deployability of learning-enabled systems in high-stakes real-world environments still faces persistent challenges. For example, in safety-critical domains like autonomous driving, robotic manipulation, and healthcare, it is crucial not only to achieve high performance but also to comply with given constraints. Furthermore, adaptability becomes paramount in non-stationary domains, where environmental parameters are subject to change. While safety and adaptability are recognized as key qualities for the new generation of AI, current approaches have not demonstrated effective adaptable performance in constrained settings. Hence, this paper breaks new ground by studying the unique challenges of ensuring safety in nonstationary environments by solving constrained problems through the lens of the meta-learning approach (learning to learn). While unconstrained meta-learning already encounters complexities in end to end differentiation of the loss due to the bi-level nature, its constrained counterpart introduces an additional layer of difficulty, since the constraints imposed on task-level updates complicate the differentiation process. To address the issue, we first employ successive convex-constrained policy updates across multiple tasks with differentiable convex programming, which allows meta-learning in constrained scenarios by enabling end-to-end differentiation. This approach empowers the agent to rapidly adapt to new tasks under nonstationarity while ensuring compliance with safety constraints. We also provide a theoretical analysis demonstrating guaranteed monotonic improvement of our approach, justifying our algorithmic designs. Extensive simulations across diverse environments provide empirical validation with significant improvement over established benchmarks.

----

## [2340] Conformal Prediction Regions for Time Series Using Linear Complementarity Programming

**Authors**: *Matthew Cleaveland, Insup Lee, George J. Pappas, Lars Lindemann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30089](https://doi.org/10.1609/aaai.v38i19.30089)

**Abstract**:

Conformal prediction is a statistical tool for producing prediction regions of machine learning models that are valid with high probability. However, applying conformal prediction to  time series data leads to  conservative prediction regions. In fact,  to obtain prediction regions over T time steps with confidence 1--delta, previous works require that each individual prediction region is valid with confidence 1--delta/T. We propose an optimization-based method for reducing this conservatism to enable long horizon planning and verification when using learning-enabled time series predictors. Instead of considering prediction errors individually at each time step, we consider a parameterized prediction error over multiple time steps. By optimizing the parameters over an additional dataset, we find  prediction regions that are not conservative. We show that this problem can be cast as a mixed integer linear complementarity program (MILCP), which we then relax into a linear complementarity program (LCP). Additionally, we prove that the relaxed LP has the same optimal cost as the original MILCP. Finally, we demonstrate the efficacy of our method on case studies using pedestrian trajectory predictors and F16 fighter jet altitude predictors.

----

## [2341] TTTS: Tree Test Time Simulation for Enhancing Decision Tree Robustness against Adversarial Examples

**Authors**: *Seffi Cohen, Ofir Arbili, Yisroel Mirsky, Lior Rokach*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30090](https://doi.org/10.1609/aaai.v38i19.30090)

**Abstract**:

Decision trees are widely used for addressing learning tasks involving tabular data. Yet, they are susceptible to adversarial attacks. In this paper, we present Tree Test Time Simulation (TTTS), a novel inference-time methodology that incorporates Monte Carlo simulations into decision trees to enhance their robustness. TTTS introduces a probabilistic modification to the decision path, without altering the underlying tree structure. Our comprehensive empirical analysis of 50 datasets yields promising results. Without the presence of any attacks, TTTS has successfully improved model performance from an AUC of 0.714 to 0.773. Under the challenging conditions of white-box attacks, TTTS demonstrated its robustness by boosting performance from an AUC of 0.337 to 0.680. Even when subjected to black-box attacks, TTTS maintains high accuracy and enhances the model's performance from an AUC of 0.628 to 0.719. Compared to defenses such as Feature Squeezing, TTTS proves to be much more effective. We also found that TTTS exhibits similar robustness in decision forest settings across different attacks.

----

## [2342] Find the Lady: Permutation and Re-synchronization of Deep Neural Networks

**Authors**: *Carl De Sousa Trias, Mihai Petru Mitrea, Attilio Fiandrotti, Marco Cagnazzo, Sumanta Chaudhuri, Enzo Tartaglione*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30091](https://doi.org/10.1609/aaai.v38i19.30091)

**Abstract**:

Deep neural networks are characterized by multiple symmetrical, equi-loss solutions that are redundant. Thus, the order of neurons in a layer and feature maps can be given arbitrary permutations, without affecting (or minimally affecting) their output. If we shuffle these neurons, or if we apply to them some perturbations (like fine-tuning) can we put them back in the original order i.e. re-synchronize? Is there a possible corruption threat? Answering these questions is important for applications like neural network white-box watermarking for ownership tracking and integrity verification.
We advance a method to re-synchronize the order of permuted neurons. Our method is also effective if neurons are further altered by parameter pruning, quantization, and fine-tuning, showing robustness to integrity attacks. Additionally, we provide theoretical and practical evidence for the usual means to corrupt the integrity of the model, resulting in a solution to counter it. We test our approach on popular computer vision datasets and models, and we illustrate the threat and our countermeasure on a popular white-box watermarking method.

----

## [2343] Stability Analysis of Switched Linear Systems with Neural Lyapunov Functions

**Authors**: *Virginie Debauche, Alec Edwards, Raphaël M. Jungers, Alessandro Abate*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30092](https://doi.org/10.1609/aaai.v38i19.30092)

**Abstract**:

Neural-based, data-driven analysis and control of dynamical systems have been recently investigated and have shown great promise, e.g. for safety verification or stability analysis. Indeed, not only do neural networks allow for an entirely model-free, data-driven approach, but also for handling arbitrary complex functions via their power of representation (as opposed to, e.g. algebraic optimization techniques that are restricted to polynomial functions). Whilst classical Lyapunov techniques allow to provide a formal and robust guarantee of stability of a switched dynamical system, very little is yet known about correctness guarantees for Neural Lyapunov functions, nor about their performance (amount of data needed for a certain accuracy). 
We formally introduce Neural Lyapunov functions for the stability analysis of switched linear systems: we benchmark them on this paradigmatic problem, which is notoriously difficult (and in general Turing-undecidable), but which admits existing recently-developed technologies and theoretical results. Inspired by switched systems theory, we provide theoretical guarantees on the representative power of neural networks, leveraging recent results from the ML community. We additionally experimentally display how Neural Lyapunov functions compete with state-of-the-art results and techniques, while admitting a wide range of improvement, both in theory and in practice. This study intends to improve our understanding of the opportunities and current limitations of neural-based data-driven analysis and control of complex dynamical systems.

----

## [2344] Robustness Verification of Multi-Class Tree Ensembles

**Authors**: *Laurens Devos, Lorenzo Cascioli, Jesse Davis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30093](https://doi.org/10.1609/aaai.v38i19.30093)

**Abstract**:

Tree ensembles are one of the most widely used model classes. 
However, these models are susceptible to adversarial examples, which are slightly perturbed examples that elicit a misprediction.
There has been significant research on designing approaches to verify the robustness of tree ensembles to such attacks. However, existing verification algorithms for tree ensembles are only able to analyze binary classifiers and hence address multiclass problems by reducing them to binary ones using a one-versus-other strategy. In this paper, we show that naively applying this strategy can yield incorrect results in certain situations. We address this shortcoming by proposing a novel approximate heuristic approach to verification for multiclass tree ensembles. Our approach is based on a novel generalization of the verification task, which we show emits other relevant verification queries.

----

## [2345] P2BPO: Permeable Penalty Barrier-Based Policy Optimization for Safe RL

**Authors**: *Sumanta Dey, Pallab Dasgupta, Soumyajit Dey*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30094](https://doi.org/10.1609/aaai.v38i19.30094)

**Abstract**:

Safe Reinforcement Learning (SRL) algorithms aim to learn a policy that maximizes the reward while satisfying the safety constraints. One of the challenges in SRL is that it is often difficult to balance the two objectives of reward maximization and safety constraint satisfaction. Existing algorithms utilize constraint optimization techniques like penalty-based, barrier penalty-based, and Lagrangian-based dual or primal policy optimizations methods. However, they suffer from training oscillations and approximation errors, which impact the overall learning objectives.

This paper proposes the Permeable Penalty Barrier-based Policy Optimization (P2BPO) algorithm that addresses this issue by allowing a small fraction of penalty beyond the penalty barrier, and a parameter is used to control this permeability. In addition, an adaptive penalty parameter is used instead of a constant one, which is initialized with a low value and increased gradually as the agent violates the safety constraints. We have also provided a theoretical proof of the proposed method's performance guarantee bound, which ensures that P2BPO can learn a policy satisfying the safety constraints with high probability while achieving a higher expected reward. Furthermore, we compare P2BPO with other SRL algorithms on various SRL tasks and demonstrate that it achieves better rewards while adhering to the constraints.

----

## [2346] Trade-Offs in Fine-Tuned Diffusion Models between Accuracy and Interpretability

**Authors**: *Mischa Dombrowski, Hadrien Reynaud, Johanna P. Müller, Matthew Baugh, Bernhard Kainz*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30095](https://doi.org/10.1609/aaai.v38i19.30095)

**Abstract**:

Recent advancements in diffusion models have significantly impacted the trajectory of generative machine learning re-search, with many adopting the strategy of fine-tuning pre-trained models using domain-specific text-to-image datasets. Notably, this method has been readily employed for medical applications, such as X-ray image synthesis, leveraging the plethora of associated radiology reports. Yet, a prevailing concern is the lack of assurance on whether these models genuinely comprehend their generated content. With the evolution of text conditional image generation, these models have grown potent enough to facilitate object localization scrutiny. Our research underscores this advancement in the critical realm of medical imaging, emphasizing the crucial role of interpretability. We further unravel a consequential trade-off between image fidelity – as gauged by conventional metrics – and model interpretability in generative diffusion models. Specifically, the adoption of learnable text encoders when fine-tuning results in diminished interpretability. Our in-depth exploration uncovers the underlying factors responsible for this divergence. Consequently, we present a set of design principles for the development of truly interpretable generative models. Code is available at https://github.com/MischaD/chest-distillation.

----

## [2347] From Hope to Safety: Unlearning Biases of Deep Models via Gradient Penalization in Latent Space

**Authors**: *Maximilian Dreyer, Frederik Pahde, Christopher J. Anders, Wojciech Samek, Sebastian Lapuschkin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30096](https://doi.org/10.1609/aaai.v38i19.30096)

**Abstract**:

Deep Neural Networks are prone to learning spurious correlations embedded in the training data, leading to potentially biased predictions. This poses risks when deploying these models for high-stake decision-making, such as in medical applications. Current methods for post-hoc model correction either require input-level annotations which are only possible for spatially localized biases, or augment the latent feature space, thereby hoping to enforce the right reasons. We present a novel method for model correction on the concept level that explicitly reduces model sensitivity towards biases via gradient penalization. When modeling biases via Concept Activation Vectors, we highlight the importance of choosing robust directions, as traditional regression-based approaches such as Support Vector Machines tend to result in diverging directions. We effectively mitigate biases in controlled and real-world settings on the ISIC, Bone Age, ImageNet and CelebA datasets using VGG, ResNet and EfficientNet architectures. Code and Appendix are available on https://github.com/frederikpahde/rrclarc.

----

## [2348] Automatically Testing Functional Properties of Code Translation Models

**Authors**: *Hasan Ferit Eniser, Valentin Wüstholz, Maria Christakis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30097](https://doi.org/10.1609/aaai.v38i19.30097)

**Abstract**:

Large language models are becoming increasingly practical for translating code across programming languages, a process known as transpiling. Even though automated transpilation significantly boosts developer productivity, a key concern is whether the generated code is correct. Existing work initially used manually crafted test suites to test the translations of a small corpus of programs; these test suites were later automated. In contrast, we devise the first approach for automated, functional, property-based testing of code translation models. Our general, user-provided specifications about the transpiled code capture a range of properties, from purely syntactic to purely semantic ones. As shown by our experiments, this approach is very effective in detecting property violations in popular code translation models, and therefore, in evaluating model quality with respect to given properties. We also go a step further and explore the usage scenario where a user simply aims to obtain a correct translation of some code with respect to certain properties without necessarily being concerned about the overall quality of the model. To this purpose, we develop the first property-guided search procedure for code translation models, where a model is repeatedly queried with slightly different parameters to produce alternative and potentially more correct translations. Our results show that this search procedure helps to obtain significantly better code translations.

----

## [2349] A Simple and Yet Fairly Effective Defense for Graph Neural Networks

**Authors**: *Sofiane Ennadir, Yassine Abbahaddou, Johannes F. Lutzeyer, Michalis Vazirgiannis, Henrik Boström*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30098](https://doi.org/10.1609/aaai.v38i19.30098)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as the dominant approach for machine learning on graph-structured data. However, concerns have arisen regarding the vulnerability of GNNs to small adversarial perturbations. Existing defense methods against such perturbations suffer from high time complexity and can negatively impact the model's performance on clean graphs. To address these challenges, this paper introduces NoisyGNNs, a novel defense method that incorporates noise into the underlying model's architecture. We establish a theoretical connection between noise injection and the enhancement of GNN robustness, highlighting the effectiveness of our approach. We further conduct extensive empirical evaluations on the node classification task to validate our theoretical findings, focusing on two popular GNNs: the GCN and GIN. The results demonstrate that NoisyGNN achieves superior or comparable defense performance to existing methods while minimizing added time complexity. The NoisyGNN approach is model-agnostic, allowing it to be integrated with different GNN architectures. Successful combinations of our NoisyGNN approach with existing defense techniques demonstrate even further improved adversarial defense results. Our code is publicly available at: https://github.com/Sennadir/NoisyGNN.

----

## [2350] Invisible Backdoor Attack against 3D Point Cloud Classifier in Graph Spectral Domain

**Authors**: *Linkun Fan, Fazhi He, Tongzhen Si, Wei Tang, Bing Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30099](https://doi.org/10.1609/aaai.v38i19.30099)

**Abstract**:

3D point cloud has been wildly used in security crucial domains, such as self-driving and 3D face recognition. Backdoor attack is a serious threat that usually destroy Deep Neural Networks (DNN) in the training stage. Though a few 3D backdoor attacks are designed to achieve guaranteed attack efficiency, their deformation will alarm human inspection. To obtain invisible backdoored point cloud, this paper proposes a novel 3D backdoor attack, named IBAPC, which generates backdoor trigger in the graph spectral domain. The effectiveness is grounded by the advantage of graph spectral signal that it can induce both global structure and local points to be responsible for the caused deformation in spatial domain. In detail, a new backdoor implanting function is proposed whose aim is to transform point cloud to graph spectral signal for conducting backdoor trigger. Then, we design a backdoor training procedure which updates the parameter of backdoor implanting function and victim 3D DNN alternately. Finally, the backdoored 3D DNN and its associated backdoor implanting function is obtained by finishing the backdoor training procedure. Experiment results suggest that IBAPC achieves SOTA attack stealthiness from three aspects including objective distance measurement, subjective human evaluation, graph spectral signal residual. At the same time, it obtains competitive attack efficiency. The code is available at https://github.com/f-lk/IBAPC.

----

## [2351] CASE: Exploiting Intra-class Compactness and Inter-class Separability of Feature Embeddings for Out-of-Distribution Detection

**Authors**: *Shuai Feng, Pengsheng Jin, Chongjun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30100](https://doi.org/10.1609/aaai.v38i19.30100)

**Abstract**:

Detecting out-of-distribution (OOD) inputs is critical for reliable machine learning, but deep neural networks often make overconfident predictions, even for OOD inputs that deviate from the distribution of training data. Prior methods relied on the widely used softmax cross-entropy (CE) loss that is adequate for classifying in-distribution (ID) samples but not optimally designed for OOD detection. To address this issue, we propose CASE, a simple and effective OOD detection method by explicitly improving intra-class Compactness And inter-class Separability of feature Embeddings. To enhance the separation between ID and OOD samples, CASE uses a dual-loss framework, which includes a separability loss that maximizes the inter-class Euclidean distance to promote separability among different class centers, along with a compactness loss that minimizes the intra-class Euclidean distance to encourage samples to be close to their class centers. In particular, the class centers are defined as a free optimization parameter of the model and updated by gradient descent, which is simple and further enhances the OOD detection performance. Extensive experiments demonstrate the superiority of CASE, which reduces the average FPR95 by 37.11% and improves the average AUROC by 15.89% compared to the baseline method using a softmax confidence score on the more challenging CIFAR-100 model.

----

## [2352] Solving Non-rectangular Reward-Robust MDPs via Frequency Regularization

**Authors**: *Uri Gadot, Esther Derman, Navdeep Kumar, Maxence Mohamed Elfatihi, Kfir Levy, Shie Mannor*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30101](https://doi.org/10.1609/aaai.v38i19.30101)

**Abstract**:

In robust Markov decision processes (RMDPs), it is assumed that the reward and the transition dynamics lie in a given uncertainty set. By targeting maximal return under the most adversarial model from that set, RMDPs address performance sensitivity to misspecified environments. Yet, to preserve computational tractability, the uncertainty set is traditionally independently structured for each state. This so-called rectangularity condition is solely motivated by computational concerns. As a result, it lacks a practical incentive and may lead to overly conservative behavior.
In this work, we study coupled reward RMDPs where the transition kernel is fixed, but the reward function lies within an alpha-radius from a nominal one. We draw a direct connection between this type of non-rectangular reward-RMDPs and applying policy visitation frequency regularization. We introduce a policy-gradient method, and prove its convergence. Numerical experiments illustrate the learned policy's robustness and its less conservative behavior when compared to rectangular uncertainty.

----

## [2353] Balance Reward and Safety Optimization for Safe Reinforcement Learning: A Perspective of Gradient Manipulation

**Authors**: *Shangding Gu, Bilgehan Sel, Yuhao Ding, Lu Wang, Qingwei Lin, Ming Jin, Alois Knoll*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30102](https://doi.org/10.1609/aaai.v38i19.30102)

**Abstract**:

Ensuring the safety of Reinforcement Learning (RL) is crucial for its deployment in real-world applications. Nevertheless, managing the trade-off between reward and safety during exploration presents a significant challenge. Improving reward performance through policy adjustments may adversely affect safety performance. In this study, we aim to address this conflicting relation by leveraging the theory of gradient manipulation. Initially, we analyze the conflict between reward and safety gradients. Subsequently, we tackle the balance between reward and safety optimization by proposing a soft switching policy optimization method, for which we provide convergence analysis. Based on our theoretical examination, we provide a safe RL framework to overcome the aforementioned challenge, and we develop a Safety-MuJoCo Benchmark to assess the performance of safe RL algorithms. Finally, we evaluate the effectiveness of our method on the Safety-MuJoCo Benchmark and a popular safe benchmark, Omnisafe. Experimental results demonstrate that our algorithms outperform several state-of-the-art baselines in terms of balancing reward and safety optimization.

----

## [2354] π-Light: Programmatic Interpretable Reinforcement Learning for Resource-Limited Traffic Signal Control

**Authors**: *Yin Gu, Kai Zhang, Qi Liu, Weibo Gao, Longfei Li, Jun Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30103](https://doi.org/10.1609/aaai.v38i19.30103)

**Abstract**:

The recent advancements in Deep Reinforcement Learning (DRL) have significantly enhanced the performance of adaptive Traffic Signal Control (TSC). However, DRL policies are typically represented by neural networks, which are over-parameterized black-box models. As a result, the learned policies often lack interpretability, and cannot be deployed directly in the real-world edge hardware due to resource constraints. In addition, the DRL methods often exhibit limited generalization performance, struggling to generalize the learned policy to other geographical regions. These factors limit the practical application of learning-based approaches. To address these issues, we suggest the use of an inherently interpretable program for representing the control policy. We present a new approach, Programmatic Interpretable reinforcement learning for traffic signal control (π-light), designed to autonomously discover non-differentiable programs. Specifically, we define a Domain Specific Language (DSL) and transformation rules for constructing programs, and utilize Monte Carlo Tree Search (MCTS) to find the optimal program in a discrete space. Extensive experiments demonstrate that our method consistently outperforms baseline approaches. Moreover, π-Light exhibits superior generalization capabilities compared to DRL, enabling training and evaluation across intersections from different cities. Finally, we analyze how the learned program policies can directly deploy on edge devices with extremely limited resources.

----

## [2355] Generative Model for Decision Trees

**Authors**: *Riccardo Guidotti, Anna Monreale, Mattia Setzu, Giulia Volpi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30104](https://doi.org/10.1609/aaai.v38i19.30104)

**Abstract**:

Decision trees are among the most popular supervised models due to their interpretability and knowledge representation resembling human reasoning. Commonly-used decision tree induction algorithms are based on greedy top-down strategies. Although these approaches are known to be an efficient heuristic, the resulting trees are only locally optimal and tend to have overly complex structures. On the other hand, optimal decision tree algorithms attempt to create an entire decision tree at once to achieve global optimality. We place our proposal between these approaches by designing a generative model for decision trees. Our method first learns a latent decision tree space through a variational architecture using pre-trained decision tree models. Then, it adopts a genetic procedure to explore such latent space to find a compact decision tree with good predictive performance. We compare our proposal against classical tree induction methods, optimal approaches, and ensemble models. The results show that our proposal can generate accurate and shallow, i.e., interpretable, decision trees.

----

## [2356] Omega-Regular Decision Processes

**Authors**: *Ernst Moritz Hahn, Mateo Perez, Sven Schewe, Fabio Somenzi, Ashutosh Trivedi, Dominik Wojtczak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30105](https://doi.org/10.1609/aaai.v38i19.30105)

**Abstract**:

Regular decision processes (RDPs) are a subclass of non-Markovian decision processes where the transition and reward functions are guarded by some regular property of the past (a lookback). While RDPs enable intuitive and succinct representation of non-Markovian decision processes, their expressive power coincides with finite-state Markov decision processes (MDPs). We introduce omega-regular decision processes (ODPs) where the non-Markovian aspect of the transition and reward functions are extended to an omega-regular lookahead over the system evolution. Semantically, these lookaheads can be considered as promises made by the decision maker or the learning agent about her future behavior. In particular, we assume that, if the promised lookaheads are not met, then the payoff to the decision maker is falsum (least desirable payoff), overriding any rewards collected by the decision maker. We enable optimization and learning for ODPs under the discounted-reward objective by reducing them to lexicographic optimization and learning over finite MDPs. We present experimental results demonstrating the effectiveness of the proposed reduction.

----

## [2357] Provable Robustness against a Union of L_0 Adversarial Attacks

**Authors**: *Zayd Hammoudeh, Daniel Lowd*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30106](https://doi.org/10.1609/aaai.v38i19.30106)

**Abstract**:

Sparse or L0 adversarial attacks arbitrarily perturb an unknown subset of the features. L0 robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art L0 certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of L0 evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art L0 defenses, FPA is up to 3,000x faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.

----

## [2358] All but One: Surgical Concept Erasing with Model Preservation in Text-to-Image Diffusion Models

**Authors**: *Seunghoo Hong, Juhun Lee, Simon S. Woo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30107](https://doi.org/10.1609/aaai.v38i19.30107)

**Abstract**:

Text-to-Image models such as Stable Diffusion have shown impressive image generation synthesis, thanks to the utilization of large-scale datasets. However, these datasets may contain sexually explicit, copyrighted, or undesirable content, which allows the model to directly generate them. Given that retraining these large models on individual concept deletion requests is infeasible, fine-tuning algorithms have been developed to tackle concept erasing in diffusion models. While these algorithms yield good concept erasure, they all present one of the following issues: 1) the corrupted feature space yields synthesis of disintegrated objects, 2) the initially synthesized content undergoes a divergence in both spatial structure and semantics in the generated images, and 3) sub-optimal training updates heighten the model's susceptibility to utility harm. These issues severely degrade the original utility of generative models. In this work, we present a new approach that solves all of these challenges. We take inspiration from the concept of classifier guidance and propose a surgical update on the classifier guidance term while constraining the drift of the unconditional score term. Furthermore, our algorithm empowers the user to select an alternative to the erasing concept, allowing for more controllability. Our experimental results show that our algorithm not only erases the target concept effectively but also preserves the model’s generation capability.

----

## [2359] Towards Efficient Verification of Quantized Neural Networks

**Authors**: *Pei Huang, Haoze Wu, Yuting Yang, Ieva Daukantas, Min Wu, Yedi Zhang, Clark W. Barrett*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30108](https://doi.org/10.1609/aaai.v38i19.30108)

**Abstract**:

Quantization replaces floating point arithmetic with integer arithmetic in deep neural network models, providing more efficient on-device inference with less power and memory. In this work, we propose a framework for formally verifying the properties of quantized neural networks. Our baseline technique is based on integer linear programming which guarantees both soundness and completeness. We then show how efficiency can be improved by utilizing gradient-based heuristic search methods and also bound-propagation techniques. We evaluate our approach on perception networks quantized with PyTorch. Our results show that we can verify quantized networks with better scalability and efficiency than the previous state of the art.

----

## [2360] On the Concept Trustworthiness in Concept Bottleneck Models

**Authors**: *Qihan Huang, Jie Song, Jingwen Hu, Haofei Zhang, Yong Wang, Mingli Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30109](https://doi.org/10.1609/aaai.v38i19.30109)

**Abstract**:

Concept Bottleneck Models (CBMs), which break down the reasoning process into the input-to-concept mapping and the concept-to-label prediction, have garnered significant attention due to their remarkable interpretability achieved by the interpretable concept bottleneck. However, despite the transparency of the concept-to-label prediction, the mapping from the input to the intermediate concept remains a black box, giving rise to concerns about the trustworthiness of the learned concepts (i.e., these concepts may be predicted based on spurious cues). The issue of concept untrustworthiness greatly hampers the interpretability of CBMs, thereby hindering their further advancement. To conduct a comprehensive analysis on this issue, in this study we establish a benchmark to assess the trustworthiness of concepts in CBMs. A pioneering metric, referred to as concept trustworthiness score, is proposed to gauge whether the concepts are derived from relevant regions. Additionally, an enhanced CBM is introduced, enabling concept predictions to be made specifically from distinct parts of the feature map, thereby facilitating the exploration of their related regions. Besides, we introduce three modules, namely the cross-layer alignment (CLA) module, the cross-image alignment (CIA) module, and the prediction alignment (PA) module, to further enhance the concept trustworthiness within the elaborated CBM. The experiments on five datasets across ten architectures demonstrate that without using any concept localization annotations during training, our model improves the concept trustworthiness by a large margin, meanwhile achieving superior accuracy to the state-of-the-arts. Our code is available at https://github.com/hqhQAQ/ProtoCBM.

----

## [2361] Personalization as a Shortcut for Few-Shot Backdoor Attack against Text-to-Image Diffusion Models

**Authors**: *Yihao Huang, Felix Juefei-Xu, Qing Guo, Jie Zhang, Yutong Wu, Ming Hu, Tianlin Li, Geguang Pu, Yang Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30110](https://doi.org/10.1609/aaai.v38i19.30110)

**Abstract**:

Although recent personalization methods have democratized high-resolution image synthesis by enabling swift concept acquisition with minimal examples and lightweight computation, they also present an exploitable avenue for highly accessible backdoor attacks. This paper investigates a critical and unexplored aspect of text-to-image (T2I) diffusion models - their potential vulnerability to backdoor attacks via personalization. By studying the prompt processing of popular personalization methods (epitomized by Textual Inversion and DreamBooth), we have devised dedicated personalization-based backdoor attacks according to the different ways of dealing with unseen tokens and divide them into two families: nouveau-token and legacy-token backdoor attacks. In comparison to conventional backdoor attacks involving the fine-tuning of the entire text-to-image diffusion model, our proposed personalization-based backdoor attack method can facilitate more tailored, efficient, and few-shot attacks. Through comprehensive empirical study, we endorse the utilization of the nouveau-token backdoor attack due to its impressive effectiveness, stealthiness, and integrity, markedly outperforming the legacy-token backdoor attack.

----

## [2362] Stronger and Transferable Node Injection Attacks

**Authors**: *Samyak Jain, Tanima Dutta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30111](https://doi.org/10.1609/aaai.v38i19.30111)

**Abstract**:

Despite the increasing popularity of graph neural networks (GNNs), the security risks associated with their deployment have not been well explored. Existing works follow the standard adversarial attacks to maximize cross-entropy loss within an L-infinity norm bound. We analyze the robustness of GNNs against node injection attacks (NIAs) in black-box settings by allowing new nodes to be injected and attacked. In this work, we propose to design stronger and transferable NIAs. First, we propose margin aware attack (MAA) that uses a maximum margin loss to generate NIAs. We then propose a novel margin and direction aware attack (MDA) that diversifies the initial directions of MAA attack by minimizing the cosine similarity of the injected nodes with respect to their respective random initialization in addition to the maximization of max-margin loss. This makes the NIAs stronger. We further observe that using L2 norm of gradients in the attack step leads to an enhanced diversity amongst the node features, thereby further enhancing the strength of the attack. We incorporate transferability in NIAs by perturbing the surrogate model before generating the attack. An analysis of eigen spectrum density of the hessian of the loss emphasizes that perturbing the weights of the surrogate model improves the transferability. Our experimental results demonstrate that the proposed resilient node injection attack (R-NIA) consistently outperform PGD by margins about 7-15% on both large and small graph datasets. R-NIA is significantly stronger and transferable than existing NIAs on graph robustness benchmarks.

----

## [2363] Learning Fair Policies for Multi-Stage Selection Problems from Observational Data

**Authors**: *Zhuangzhuang Jia, Grani A. Hanasusanto, Phebe Vayanos, Weijun Xie*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30112](https://doi.org/10.1609/aaai.v38i19.30112)

**Abstract**:

We consider the problem of learning fair policies for multi-stage selection problems from observational data. This problem arises in several high-stakes domains such as company hiring, loan approval, or bail decisions where outcomes (e.g., career success, loan repayment, recidivism) are only observed for those selected. We propose a multi-stage framework that can be augmented with various fairness constraints, such as demographic parity or equal opportunity. This problem is a highly intractable infinite chance-constrained program involving the unknown joint distribution of covariates and outcomes.  Motivated by the potential impact of selection decisions on people’s lives and livelihoods, we propose to focus on interpretable linear selection rules. Leveraging tools from causal inference and sample average approximation, we obtain an asymptotically consistent solution to this selection problem by solving a mixed binary conic optimization problem, which can be solved using standard off-the-shelf solvers. We conduct extensive computational experiments on a variety of datasets adapted from the UCI repository on which we show that our proposed approaches can achieve an 11.6% improvement in precision and a 38% reduction in the measure of unfairness compared to the existing selection policy.

----

## [2364] NeRFail: Neural Radiance Fields-Based Multiview Adversarial Attack

**Authors**: *Wenxiang Jiang, Hanwei Zhang, Xi Wang, Zhongwen Guo, Hao Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30113](https://doi.org/10.1609/aaai.v38i19.30113)

**Abstract**:

Adversarial attacks, i.e., generating adversarial perturbations with a small magnitude to deceive deep neural networks, are important for investigating and improving model trustworthiness. Traditionally, the topic was scoped within 2D images without considering 3D multiview information. Benefiting from Neural Radiance Fields (NeRF), one can easily reconstruct a 3D scene with a Multi-Layer Perceptron (MLP) from given 2D views and synthesize photo-realistic renderings of novel vantages. This opens up a door to discussing the possibility of undertaking to attack multiview NeRF network with downstream tasks from different rendering angles, which we denote Neural Radiance Fiels-based multiview adversarial Attack (NeRFail). The goal is, given one scene and a subset of views, to deceive the recognition results of agnostic view angles as well as given views. To do so, we propose a transformation mapping from pixels to 3D points such that our attack generates multiview adversarial perturbations by attacking a subset of images with different views, intending to prevent the downstream classifier from correctly predicting images rendered by NeRF from other views. Experiments show that our multiview adversarial perturbations successfully obfuscate the downstream classifier at both known and unknown views. Notably, when retraining another NeRF on the perturbed training data, we show that the perturbation can be inherited and reproduced. The code can be found at https://github.com/jiang-wenxiang/NeRFail.

----

## [2365] Analysis of Differentially Private Synthetic Data: A Measurement Error Approach

**Authors**: *Yangdi Jiang, Yi Liu, Xiaodong Yan, Anne-Sophie Charest, Linglong Kong, Bei Jiang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30114](https://doi.org/10.1609/aaai.v38i19.30114)

**Abstract**:

Differentially private (DP) synthetic datasets have been receiving significant attention from academia, industry, and government. However, little is known about how to perform statistical inference using DP synthetic datasets. Naive approaches that do not take into account the induced uncertainty due to the DP mechanism will result in biased estimators and invalid inferences. In this paper, we present a class of maximum likelihood estimator (MLE)-based easy-to-implement bias-corrected DP estimators with valid asymptotic confidence intervals (CI) for parameters in regression settings, by establishing the connection between additive DP mechanisms and measurement error models. Our simulation shows that our estimator has comparable performance to the widely used sufficient statistic perturbation (SSP) algorithm in some scenarios but with the advantage of releasing a synthetic dataset and obtaining statistically valid asymptotic CIs, which can achieve better coverage when compared to the naive CIs obtained by ignoring the DP mechanism.

----

## [2366] Chasing Fairness in Graphs: A GNN Architecture Perspective

**Authors**: *Zhimeng Jiang, Xiaotian Han, Chao Fan, Zirui Liu, Na Zou, Ali Mostafavi, Xia Hu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30115](https://doi.org/10.1609/aaai.v38i19.30115)

**Abstract**:

There has been significant progress in improving the performance of graph neural networks (GNNs) through enhancements in graph data, model architecture design, and training strategies. For fairness in graphs, recent studies achieve fair representations and predictions through either graph data pre-processing (e.g., node feature masking, and topology rewiring) or fair training strategies (e.g., regularization, adversarial debiasing, and fair contrastive learning). How to achieve fairness in graphs from the model architecture perspective is less explored. More importantly, GNNs exhibit worse fairness performance compared to multilayer perception since their model architecture (i.e., neighbor aggregation) amplifies biases. To this end, we aim to achieve fairness via a new GNN architecture. We propose Fair Message Passing (FMP) designed within a unified optimization framework for GNNs. Notably, FMP explicitly renders sensitive attribute usage in forward propagation for node classification task using cross-entropy loss without data pre-processing. In FMP, the aggregation is first adopted to utilize neighbors' information and then the bias mitigation step explicitly pushes demographic group node presentation centers together.
In this way, FMP scheme can aggregate useful information from neighbors and mitigate bias to achieve better fairness and prediction tradeoff performance. 
Experiments on node classification tasks demonstrate that the proposed FMP outperforms several baselines in terms of fairness and accuracy on three real-world datasets. The code is available at https://github.com/zhimengj0326/FMP.

----

## [2367] Assume-Guarantee Reinforcement Learning

**Authors**: *Milad Kazemi, Mateo Perez, Fabio Somenzi, Sadegh Soudjani, Ashutosh Trivedi, Alvaro Velasquez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30116](https://doi.org/10.1609/aaai.v38i19.30116)

**Abstract**:

We present a modular approach to reinforcement learning (RL) in environments consisting of simpler components evolving in parallel. A monolithic view of such modular environments may be prohibitively large to learn, or may require unrealizable communication between the components in the form of a centralized controller. Our proposed approach is based on the assume-guarantee paradigm where the optimal control for the individual components is synthesized in isolation by making assumptions about the behaviors of neighboring components, and providing guarantees about their own behavior. We express these assume-guarantee contracts as regular languages and provide automatic translations to scalar rewards to be used in RL. By combining local probabilities of satisfaction for each component, we provide a lower bound on the probability of satisfaction of the complete system. By solving a Markov game for each component, RL can produce a controller for each component that maximizes this lower bound. The controller utilizes the information it receives through communication, observations, and any knowledge of a coarse model of other agents. We experimentally demonstrate the efficiency of the proposed approach on a variety of case studies.

----

## [2368] DeepBern-Nets: Taming the Complexity of Certifying Neural Networks Using Bernstein Polynomial Activations and Precise Bound Propagation

**Authors**: *Haitham Khedr, Yasser Shoukry*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30117](https://doi.org/10.1609/aaai.v38i19.30117)

**Abstract**:

Formal certification of Neural Networks (NNs) is crucial for ensuring their safety, fairness, and robustness. Unfortunately, on the one hand, sound and complete certification algorithms of ReLU-based NNs do not scale to large-scale NNs. On the other hand, incomplete certification algorithms are easier to compute, but they result in loose bounds that deteriorate with the depth of NN, which diminishes their effectiveness. In this paper, we ask the following question; can we replace the ReLU activation function with one that opens the door to incomplete certification algorithms that are easy to compute but can produce tight bounds on the NN's outputs? We introduce DeepBern-Nets, a class of NNs with activation functions based on Bernstein polynomials instead of the commonly used ReLU activation. Bernstein polynomials are smooth and differentiable functions with desirable properties such as the so-called range enclosure and subdivision properties. We design a novel Interval Bound Propagation (IBP) algorithm, called Bern-IBP, to efficiently compute tight bounds on DeepBern-Nets outputs. Our approach leverages the properties of Bernstein polynomials to improve the tractability of neural network certification tasks while maintaining the accuracy of the trained networks. We conduct experiments in adversarial robustness and reachability analysis settings to assess the effectiveness of the approach. Our proposed framework achieves high certified accuracy for adversarially-trained NNs, which is often a challenging task for certifiers of ReLU-based NNs. This work establishes Bernstein polynomial activation as a promising alternative for improving NN certification tasks across various NNs applications.

----

## [2369] Layer Attack Unlearning: Fast and Accurate Machine Unlearning via Layer Level Attack and Knowledge Distillation

**Authors**: *Hyunjune Kim, Sangyong Lee, Simon S. Woo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30118](https://doi.org/10.1609/aaai.v38i19.30118)

**Abstract**:

Recently, serious concerns have been raised about the privacy issues related to training datasets in machine learning algorithms when including personal data. Various regulations in different countries, including the GDPR grant individuals to have personal data erased, known as ‘the right to be forgotten’ or ‘the right to erasure’. However, there has been less research on effectively and practically deleting the requested personal data from the training set while not jeopardizing the overall machine learning performance. In this work, we propose a fast and novel machine unlearning paradigm at the layer level called layer attack unlearning, which is highly accurate and fast compared to existing machine unlearning algorithms. We introduce the Partial-PGD algorithm to locate the samples to forget efficiently. In addition, we only use the last layer of the model inspired by the Forward-Forward algorithm for unlearning process. Lastly, we use Knowledge Distillation (KD) to reliably learn the decision boundaries from the teacher using soft label information to improve accuracy performance. We conducted extensive experiments with SOTA machine unlearning models and demonstrated the effectiveness of our approach for accuracy and end-to-end unlearning performance.

----

## [2370] Quilt: Robust Data Segment Selection against Concept Drifts

**Authors**: *Minsu Kim, Seonghyeon Hwang, Steven Euijong Whang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30119](https://doi.org/10.1609/aaai.v38i19.30119)

**Abstract**:

Continuous machine learning pipelines are common in industrial settings where models are periodically trained on data streams. Unfortunately, concept drifts may occur in data streams where the joint distribution of the data X and label y, P(X, y), changes over time and possibly degrade model accuracy. Existing concept drift adaptation approaches mostly focus on updating the model to the new data possibly using ensemble techniques of previous models and tend to discard the drifted historical data. However, we contend that explicitly utilizing the drifted data together leads to much better model accuracy and propose Quilt, a data-centric framework for identifying and selecting data segments that maximize model accuracy. To address the potential downside of efficiency, Quilt extends existing data subset selection techniques, which can be used to reduce the training data without compromising model accuracy. These techniques cannot be used as is because they only assume virtual drifts where the posterior probabilities P(y|X) are assumed not to change. In contrast, a key challenge in our setup is to also discard undesirable data segments with concept drifts. Quilt thus discards drifted data segments and selects data segment subsets holistically for accurate and efficient model training. The two operations use gradient-based scores, which have little computation overhead. In our experiments, we show that Quilt outperforms state-of-the-art drift adaptation and data selection baselines on synthetic and real datasets.

----

## [2371] OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples

**Authors**: *Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30120](https://doi.org/10.1609/aaai.v38i19.30120)

**Abstract**:

Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

----

## [2372] Accelerating Adversarially Robust Model Selection for Deep Neural Networks via Racing

**Authors**: *Matthias König, Holger H. Hoos, Jan N. van Rijn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30121](https://doi.org/10.1609/aaai.v38i19.30121)

**Abstract**:

Recent research has introduced several approaches to formally verify the robustness of neural network models against perturbations in their inputs, such as the ones that occur in adversarial attacks. At the same time, this particular verification task is known to be computationally challenging. More specifically, assessing the robustness of a neural network against input perturbations can easily take several hours of compute time per input vector, even when using state-of-the-art verification approaches. In light of this, it becomes challenging to select from a given set of neural network models the one that is best in terms of robust accuracy, i.e., the fraction of instances for which the model is known to be robust against adversarial perturbations, especially when given limited computing resources.
To tackle this problem, we propose a racing method specifically adapted to the domain of robustness verification. This racing method utilises Delta-values, which can be seen as an efficiently computable proxy for the distance of a given input to a neural network model to the decision boundary. We present statistical evidence indicating significant differences in the empirical cumulative distribution between robust and non-robust inputs as a function of Delta-values. Using this information, we show that it is possible to reliably expose vulnerabilities in the model with relatively few input iterations. Overall, when applied to selecting the most robust network from sets of 31 MNIST and 27 CIFAR-10 networks, our proposed method achieves speedups of a factor of 108 and 42, respectively, in terms of cumulative running time compared to standard local robustness verification on the complete testing sets.

----

## [2373] Robust Active Measuring under Model Uncertainty

**Authors**: *Merlijn Krale, Thiago D. Simão, Jana Tumova, Nils Jansen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30122](https://doi.org/10.1609/aaai.v38i19.30122)

**Abstract**:

Partial observability and uncertainty are common problems in sequential decision-making that particularly impede the use of formal models such as Markov decision processes (MDPs). However, in practice, agents may be able to employ costly sensors to measure their environment and resolve partial observability by gathering information. Moreover, imprecise transition functions can capture model uncertainty. We combine these concepts and extend MDPs to robust active-measuring MDPs (RAM-MDPs). We present an active-measure heuristic to solve RAM-MDPs efficiently and show that model uncertainty can, counterintuitively, let agents take fewer measurements. We propose a method to counteract this behavior while only incurring a bounded additional cost. We empirically compare our methods to several baselines and show their superior scalability and performance.

----

## [2374] Towards Large Certified Radius in Randomized Smoothing Using Quasiconcave Optimization

**Authors**: *Bo-Han Kung, Shang-Tse Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30123](https://doi.org/10.1609/aaai.v38i19.30123)

**Abstract**:

Randomized smoothing is currently the state-of-the-art method that provides certified robustness for deep neural networks. However, due to its excessively conservative nature, this method of incomplete verification often cannot achieve an adequate certified radius on real-world datasets. One way to obtain a larger certified radius is to use an input-specific algorithm instead of using a fixed Gaussian filter for all data points. Several methods based on this idea have been proposed, but they either suffer from high computational costs or gain marginal improvement in certified radius. In this work, we show that by exploiting the quasiconvex problem structure, we can find the optimal certified radii for most data points with slight computational overhead. This observation leads to an efficient and effective input-specific randomized smoothing algorithm. We conduct extensive experiments and empirical analysis on CIFAR-10 and ImageNet. The results show that the proposed method significantly enhances the certified radii with low computational overhead.

----

## [2375] Contrastive Credibility Propagation for Reliable Semi-supervised Learning

**Authors**: *Brody Kutt, Pralay Ramteke, Xavier Mignot, Pamela Toman, Nandini Ramanan, Sujit Rokka Chhetri, Shan Huang, Min Du, William Hewlett*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30124](https://doi.org/10.1609/aaai.v38i19.30124)

**Abstract**:

Producing labels for unlabeled data is error-prone, making semi-supervised learning (SSL) troublesome. Often, little is known about when and why an algorithm fails to outperform a supervised baseline. Using benchmark datasets, we craft five common real-world SSL data scenarios: few-label, open-set, noisy-label, and class distribution imbalance/misalignment in the labeled and unlabeled sets. We propose a novel algorithm called Contrastive Credibility Propagation (CCP) for deep SSL via iterative transductive pseudo-label refinement. CCP unifies semi-supervised learning and noisy label learning for the goal of reliably outperforming a supervised baseline in any data scenario. Compared to prior methods which focus on a subset of scenarios, CCP uniquely outperforms the supervised baseline in all scenarios, supporting practitioners when the qualities of labeled or unlabeled data are unknown.

----

## [2376] Exponent Relaxation of Polynomial Zonotopes and Its Applications in Formal Neural Network Verification

**Authors**: *Tobias Ladner, Matthias Althoff*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30125](https://doi.org/10.1609/aaai.v38i19.30125)

**Abstract**:

Formal verification of neural networks is a challenging problem due to the complexity and nonlinearity of neural networks.
    It has been shown that polynomial zonotopes can tightly enclose the output set of a neural network.
    Unfortunately, the tight enclosure comes with additional complexity in the set representation,
    thus, rendering subsequent operations expensive to compute, such as computing interval bounds and intersection checking.
    To address this issue, we present a novel approach to restructure a polynomial zonotope to tightly enclose the original polynomial zonotope
    while drastically reducing its complexity.
    The restructuring is achieved by relaxing the exponents of the dependent factors of polynomial zonotopes and finding an appropriate approximation error.
    We demonstrate the applicability of our approach on output sets of neural networks,
    where we obtain tighter results in various subsequent operations, such as order reduction, zonotope enclosure, and range bounding.

----

## [2377] I Prefer Not to Say: Protecting User Consent in Models with Optional Personal Data

**Authors**: *Tobias Leemann, Martin Pawelczyk, Christian Thomas Eberle, Gjergji Kasneci*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30126](https://doi.org/10.1609/aaai.v38i19.30126)

**Abstract**:

We examine machine learning models in a setup where individuals have the choice to share optional personal information with a decision-making system, as seen in modern insurance pricing models. Some users consent to their data being used whereas others object and keep their data undisclosed. In this work, we show that the decision not to share data can be considered as information in itself that should be protected to respect users' privacy. This observation raises the overlooked problem of how to ensure that users who protect their personal data do not suffer any disadvantages as a result. To address this problem, we formalize protection requirements for models which only use the information for which active user consent was obtained. This excludes implicit information contained in the decision to share data or not. We offer the first solution to this problem by proposing the notion of Protected User Consent (PUC), which we prove to be loss-optimal under our protection requirement. We observe that privacy and performance are not fundamentally at odds with each other and that it is possible for a decision maker to benefit from additional data while respecting users' consent. To learn PUC-compliant models, we devise a model-agnostic data augmentation strategy with finite sample convergence guarantees. Finally, we analyze the implications of PUC on challenging real datasets, tasks, and models.

----

## [2378] Promoting Counterfactual Robustness through Diversity

**Authors**: *Francesco Leofante, Nico Potyka*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30127](https://doi.org/10.1609/aaai.v38i19.30127)

**Abstract**:

Counterfactual explanations shed light on the decisions of black-box models by explaining
how an input can be altered to obtain a favourable decision from the model (e.g., when a loan application has been rejected).
However, as noted recently, counterfactual explainers may lack robustness in the sense that a minor change
in the input can cause a major change in the explanation. This can cause confusion on the user side and
open the door for adversarial attacks. In this paper, we study some sources of non-robustness. 
While there are fundamental reasons for why an explainer that returns a single counterfactual cannot be
robust in all instances, we show that some interesting robustness guarantees can be given by reporting 
multiple rather than a single counterfactual. Unfortunately, the number of counterfactuals that need to
be reported for the theoretical guarantees to hold can be prohibitively large. We therefore propose an approximation
algorithm that uses a diversity criterion to select a feasible number of most relevant explanations and study its robustness empirically. Our experiments indicate that our method improves the
state-of-the-art in generating robust explanations, while maintaining other desirable properties
and providing competitive computational performance.

----

## [2379] Revisiting the Information Capacity of Neural Network Watermarks: Upper Bound Estimation and Beyond

**Authors**: *Fangqi Li, Haodong Zhao, Wei Du, Shilin Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30128](https://doi.org/10.1609/aaai.v38i19.30128)

**Abstract**:

To trace the copyright of deep neural networks, an owner can embed its identity information into its model as a watermark.
The capacity of the watermark quantify the maximal volume of information that can be verified from the watermarked model.
Current studies on capacity focus on the ownership verification accuracy under ordinary removal attacks and fail to capture the relationship between robustness and fidelity.
This paper studies the capacity of deep neural network watermarks from an information theoretical perspective.
We propose a new definition of deep neural network watermark capacity analogous to channel capacity, analyze its properties, and design an algorithm that yields a tight estimation of its upper bound under adversarial overwriting.
We also propose a universal non-invasive method to secure the transmission of the identity message beyond capacity by multiple rounds of ownership verification. 
Our observations provide evidence for neural network owners and defenders that are curious about the tradeoff between the integrity of their ownership and the performance degradation of their products.

----

## [2380] PointCVaR: Risk-Optimized Outlier Removal for Robust 3D Point Cloud Classification

**Authors**: *Xinke Li, Junchi Lu, Henghui Ding, Changsheng Sun, Joey Tianyi Zhou, Yeow Meng Chee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30129](https://doi.org/10.1609/aaai.v38i19.30129)

**Abstract**:

With the growth of 3D sensing technology, the deep learning system for 3D point clouds has become increasingly important, especially in applications such as autonomous vehicles where safety is a primary concern. However, there are growing concerns about the reliability of these systems when they encounter noisy point clouds, either occurring naturally or introduced with malicious intent. This paper highlights the challenges of point cloud classification posed by various forms of noise, from simple background noise to malicious adversarial/backdoor attacks that can intentionally skew model predictions. While there's an urgent need for optimized point cloud denoising, current point outlier removal approaches, an essential step for denoising, rely heavily on handcrafted strategies and are not adapted for higher-level tasks, such as classification. To address this issue, we introduce an innovative point outlier cleansing method that harnesses the power of downstream classification models. Using gradient-based attribution analysis, we define a novel concept: point risk. Drawing inspiration from tail risk minimization in finance, we recast the outlier removal process as an optimization problem, named PointCVaR. Extensive experiments show that our proposed technique not only robustly filters diverse point cloud outliers but also consistently and significantly enhances existing robust methods for point cloud classification. A notable feature of our approach is its effectiveness in defending against the latest threat of backdoor attacks in point clouds.

----

## [2381] Game-Theoretic Unlearnable Example Generator

**Authors**: *Shuang Liu, Yihan Wang, Xiao-Shan Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30130](https://doi.org/10.1609/aaai.v38i19.30130)

**Abstract**:

Unlearnable example attacks are data poisoning attacks aiming to degrade the clean test accuracy of deep learning by adding imperceptible perturbations to the training samples, which can be formulated as a bi-level optimization problem. However, directly solving this optimization problem is intractable for deep neural networks. In this paper, we investigate unlearnable example attacks from a game-theoretic perspective, by formulating the attack as a nonzero sum Stackelberg game. First, the existence of game equilibria is proved under the normal setting and the adversarial training setting.  It is shown that the game equilibrium gives the most powerful poison attack in that the victim has the lowest test accuracy among all networks within the same hypothesis space when certain loss functions are used. Second, we propose a novel attack method, called the Game Unlearnable Example (GUE), which has three main gradients.  (1) The poisons are obtained by directly solving the equilibrium of the Stackelberg game with a first-order algorithm.  (2) We employ an autoencoder-like generative network model as the poison attacker. (3) A novel payoff function is introduced to evaluate the performance of the poison. Comprehensive experiments demonstrate that GUE can effectively poison the model in various scenarios.  Furthermore, the GUE still works by using a relatively small percentage of the training data to train the generator, and the poison generator can generalize to unseen data well. Our implementation code can be found at https://github.com/hong-xian/gue.

----

## [2382] Beyond Traditional Threats: A Persistent Backdoor Attack on Federated Learning

**Authors**: *Tao Liu, Yuhang Zhang, Zhu Feng, Zhiqin Yang, Chen Xu, Dapeng Man, Wu Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30131](https://doi.org/10.1609/aaai.v38i19.30131)

**Abstract**:

Backdoors on federated learning will be diluted by subsequent benign updates. This is reflected in the significant reduction of attack success rate as iterations increase, ultimately failing. We use a new metric to quantify the degree of this weakened backdoor effect, called attack persistence. Given that research to improve this performance has not been widely noted, we propose a Full Combination Backdoor Attack (FCBA) method. It aggregates more combined trigger information for a more complete backdoor pattern in the global model. Trained backdoored global model is more resilient to benign updates, leading to a higher attack success rate on the test set. We test on three datasets and evaluate with two models across various settings. FCBA's persistence outperforms SOTA federated learning backdoor attacks. On GTSRB, post-attack 120 rounds, our attack success rate rose over 50% from baseline. The core code of our method is available at https://github.com/PhD-TaoLiu/FCBA.

----

## [2383] Handling Long and Richly Constrained Tasks through Constrained Hierarchical Reinforcement Learning

**Authors**: *Yuxiao Lu, Arunesh Sinha, Pradeep Varakantham*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30132](https://doi.org/10.1609/aaai.v38i19.30132)

**Abstract**:

Safety in goal directed Reinforcement Learning (RL) settings has typically been handled through constraints over trajectories and have demonstrated good performance in primarily short horizon tasks. In this paper, we are specifically interested in the problem of solving temporally extended decision making problems such as robots cleaning different areas in a house while avoiding slippery and unsafe areas (e.g., stairs) and retaining enough charge to move to a charging dock; in the presence of complex safety constraints.  Our key contribution is a  (safety) Constrained Search with Hierarchical Reinforcement Learning (CoSHRL) mechanism that combines an upper level constrained search agent (which computes a reward maximizing policy from a given start to a far away goal state while satisfying cost constraints) with a low-level goal conditioned RL agent (which estimates cost and reward values to move between nearby states). A major advantage of CoSHRL is that it can handle constraints on the cost value distribution (e.g., on Conditional Value at Risk, CVaR) and can adjust to flexible constraint thresholds without retraining. We perform extensive experiments with different types of safety constraints to demonstrate the utility of our approach over leading approaches in constrained and hierarchical RL.

----

## [2384] Combining Graph Transformers Based Multi-Label Active Learning and Informative Data Augmentation for Chest Xray Classification

**Authors**: *Dwarikanath Mahapatra, Behzad Bozorgtabar, Zongyuan Ge, Mauricio Reyes, Jean-Philippe Thiran*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30133](https://doi.org/10.1609/aaai.v38i19.30133)

**Abstract**:

Informative sample selection in active learning (AL) helps a machine learning system attain optimum performance with minimum labeled samples, thus improving human-in-the-loop computer-aided diagnosis systems with limited labeled data. Data augmentation is highly effective for enlarging datasets with less labeled data. Combining informative sample selection and data augmentation should leverage their respective advantages and improve performance of AL systems. We propose a novel approach to combine informative sample selection and data augmentation for multi-label active learning. Conventional informative sample selection approaches have mostly focused on the single-label case which do not perform optimally in the multi-label setting. We improve upon state-of-the-art multi-label active learning techniques by representing disease labels as graph nodes, use graph attention transformers (GAT) to learn more effective inter-label relationships and identify most informative samples. We generate transformations of these informative samples which are also informative. Experiments on public chest xray datasets show improved results over state-of-the-art multi-label AL techniques in terms of classification performance, learning rates, and robustness. We also perform qualitative analysis to determine the realism of generated images.

----

## [2385] Enumerating Safe Regions in Deep Neural Networks with Provable Probabilistic Guarantees

**Authors**: *Luca Marzari, Davide Corsi, Enrico Marchesini, Alessandro Farinelli, Ferdinando Cicalese*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30134](https://doi.org/10.1609/aaai.v38i19.30134)

**Abstract**:

Identifying safe areas is a key point to guarantee trust for systems that are based on Deep Neural Networks (DNNs). To this end, we introduce the AllDNN-Verification problem: given a safety property and a DNN, enumerate the set of all the regions of the property input domain which are safe, i.e., where the property does hold. Due to the #P-hardness of the problem, we propose an efficient approximation method called ε-ProVe. Our approach exploits a controllable underestimation of the output reachable sets obtained via statistical prediction of tolerance limits, and can provide a tight —with provable probabilistic guarantees— lower estimate of the safe areas. Our empirical evaluation on different standard benchmarks shows the scalability and effectiveness of our method, offering valuable insights for this new type of verification of DNNs.

----

## [2386] Divide-and-Aggregate Learning for Evaluating Performance on Unlabeled Data

**Authors**: *Shuyu Miao, Jian Liu, Lin Zheng, Hong Jin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30135](https://doi.org/10.1609/aaai.v38i19.30135)

**Abstract**:

Artificial Intelligence (AI) models have become an integral part of modern society, significantly improving human lives. However, ensuring the reliability and safety of these models is of paramount importance. One critical aspect is the continuous monitoring and verification of model performance to prevent any potential risks. Real-time online evaluation of AI models is necessary to maintain their effectiveness and mitigate any harm caused by performance degradation. The traditional approach to model evaluation involves supervised methods that rely on manual labeling to compare results with model predictions. Unfortunately, this method is not suitable for online model monitoring due to its inherent lag and high cost. While there have been attempts to explore free-label model evaluation, these approaches often consider only the global features of the entire dataset. Additionally, they can only perform model evaluation based on a single dimension of model confidence or features. In this paper, we propose a novel approach called Divide-and-Aggregate Learning (DAL) for unsupervised model evaluation. Our method addresses the limitations of previous approaches by dividing the output of the model into buckets, capturing local information of the distribution. We then aggregate this local information to obtain global information and further represent the relationship between the distribution and model performance. Importantly, our method can simultaneously handle the confidence distribution and feature distribution of the model output. Extensive experiments have been conducted to demonstrate the effectiveness of our DAL model. The results show that our approach outperforms previous methods on four widely used datasets. We will make our source code publicly available.

----

## [2387] SentinelLMs: Encrypted Input Adaptation and Fine-Tuning of Language Models for Private and Secure Inference

**Authors**: *Abhijit Mishra, Mingda Li, Soham Deo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30136](https://doi.org/10.1609/aaai.v38i19.30136)

**Abstract**:

This paper addresses the privacy and security concerns associated with deep neural language models, which serve as crucial components in various modern AI-based applications. These models are often used after being pre-trained and fine-tuned for specific tasks, with deployment on servers accessed through the internet. However, this introduces two fundamental risks: (a) the transmission of user inputs to the server via the network gives rise to interception vulnerabilities, and (b) privacy concerns emerge as organizations that deploy such models store user data with restricted context. To address this, we propose a novel method to adapt and fine-tune transformer-based language models on passkey-encrypted user-specific text. The original pre-trained language model first undergoes a quick adaptation (without any further pre-training) with a series of irreversible transformations applied to the tokenizer and token embeddings. This enables the model to perform inference on encrypted inputs while preventing reverse engineering of text from model parameters and intermediate outputs. After adaptation, models are fine-tuned on encrypted versions of existing training datasets. Experimental evaluation employing adapted versions of renowned models (e.g., BERT, RoBERTa) across established benchmark English and multilingual datasets for text classification and sequence labeling shows that encrypted models achieve performance parity with their original counterparts. This serves to safeguard performance, privacy, and security cohesively.

----

## [2388] Safeguarded Progress in Reinforcement Learning: Safe Bayesian Exploration for Control Policy Synthesis

**Authors**: *Rohan Mitta, Hosein Hasanbeig, Jun Wang, Daniel Kroening, Yiannis Kantaros, Alessandro Abate*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30137](https://doi.org/10.1609/aaai.v38i19.30137)

**Abstract**:

This paper addresses the problem of maintaining safety during training in Reinforcement Learning (RL), such that the safety constraint violations are bounded at any point during learning. As enforcing safety during training might severely limit the agent’s exploration, we propose here a new architecture that handles the trade-off between efficient progress and safety during exploration. As the exploration progresses, we update via Bayesian inference Dirichlet-Categorical models of the transition probabilities of the Markov decision process that describes the environment dynamics. We then propose a way to approximate moments of belief about the risk associated to the action selection policy. We demonstrate that this approach can be easily interleaved with RL and we present experimental results to showcase the performance of the overall architecture.

----

## [2389] Feature Unlearning for Pre-trained GANs and VAEs

**Authors**: *Saemi Moon, Seunghyuk Cho, Dongwoo Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30138](https://doi.org/10.1609/aaai.v38i19.30138)

**Abstract**:

We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST, CelebA, and FFHQ datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.

----

## [2390] Reward Certification for Policy Smoothed Reinforcement Learning

**Authors**: *Ronghui Mu, Leandro Soriano Marcolino, Yanghao Zhang, Tianle Zhang, Xiaowei Huang, Wenjie Ruan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30139](https://doi.org/10.1609/aaai.v38i19.30139)

**Abstract**:

Reinforcement Learning (RL) has achieved remarkable success in safety-critical areas, but it can be weakened by adversarial attacks. Recent studies have introduced ``smoothed policies" to enhance its robustness. Yet, it is still challenging to establish a provable guarantee to certify the bound of its total reward. Prior methods relied primarily on computing bounds using Lipschitz continuity or calculating the probability of cumulative reward being above specific thresholds. However, these techniques are only suited for continuous perturbations on the RL agent's observations and are restricted to perturbations bounded by the l2-norm. To address these limitations, this paper proposes a general  black-box certification method, called ReCePS, which is capable of directly certifying the cumulative reward of the smoothed policy under various lp-norm bounded perturbations. Furthermore, we extend our methodology to certify perturbations on action spaces. Our approach leverages f-divergence to measure the distinction between the original distribution and the perturbed distribution, subsequently determining the certification bound by solving a convex optimisation problem. We provide a comprehensive theoretical analysis and run experiments in multiple environments. Our results show that our method not only improves the tightness of certified lower bound of the mean cumulative reward but also demonstrates better efficiency than state-of-the-art methods.

----

## [2391] EncryIP: A Practical Encryption-Based Framework for Model Intellectual Property Protection

**Authors**: *Xin Mu, Yu Wang, Zhengan Huang, Junzuo Lai, Yehong Zhang, Hui Wang, Yue Yu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30140](https://doi.org/10.1609/aaai.v38i19.30140)

**Abstract**:

In the rapidly growing digital economy, protecting intellectual property (IP) associated with digital products has become increasingly important.  Within this context, machine learning (ML) models, being highly valuable digital assets, have gained significant attention for IP protection. This paper introduces a practical encryption-based framework called EncryIP, which seamlessly integrates a public-key encryption scheme into the model learning process. This approach enables the protected model to generate randomized and confused labels, ensuring that only individuals with accurate secret keys, signifying authorized users, can decrypt and reveal authentic labels. Importantly, the proposed framework not only facilitates the protected model to multiple authorized users without requiring repetitive training of the original ML model with IP protection methods but also maintains the model's performance without compromising its accuracy. Compared to existing methods like watermark-based, trigger-based, and passport-based approaches, EncryIP demonstrates superior effectiveness in both training protected models and efficiently detecting the unauthorized spread of ML models.

----

## [2392] Neural Closure Certificates

**Authors**: *Alireza Nadali, Vishnu Murali, Ashutosh Trivedi, Majid Zamani*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30141](https://doi.org/10.1609/aaai.v38i19.30141)

**Abstract**:

Notions of transition invariants and closure certificates have seen recent use in the formal verification of controlled dynamical systems against \omega-regular properties.
Unfortunately, existing approaches face limitations in two directions.
First, they require a closed-form mathematical expression representing the model of the system.
Such an expression may be difficult to find, too complex to be of any use, or unavailable due to security or privacy constraints.
Second, finding such invariants typically rely on optimization techniques such as sum-of-squares (SOS) or satisfiability modulo theory (SMT) solvers.
This restricts the classes of systems that need to be formally verified.
To address these drawbacks, we introduce a notion of neural closure certificates.
We present a data-driven algorithm that trains a neural network to represent a closure certificate.
Our approach is formally correct under some mild assumptions, i.e., one is able to formally show that the unknown system  satisfies the \omega-regular property of interest if a neural closure certificate can be computed.
Finally, we demonstrate the efficacy of our approach with relevant case studies.

----

## [2393] SocialStigmaQA: A Benchmark to Uncover Stigma Amplification in Generative Language Models

**Authors**: *Manish Nagireddy, Lamogha Chiazor, Moninder Singh, Ioana Baldini*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30142](https://doi.org/10.1609/aaai.v38i19.30142)

**Abstract**:

Current datasets for unwanted social bias auditing are limited to studying protected demographic features such as race and gender. In this work, we introduce a comprehensive benchmark that is meant to capture the amplification of social bias, via stigmas, in generative language models. Taking inspiration from social science research, we start with a documented list of 93 US-centric stigmas and curate a question-answering (QA) dataset which involves simple social situations. Our benchmark, SocialStigmaQA, contains roughly 10K prompts, with a variety of prompt styles, carefully constructed to systematically test for both social bias and model robustness. We present results for SocialStigmaQA with two open source generative language models and we find that the proportion of socially biased output ranges from 45% to 59% across a variety of decoding strategies and prompting styles. We demonstrate that the deliberate design of the templates in our benchmark (e.g., adding biasing text to the prompt or using different verbs that change the answer that indicates bias) impacts the model tendencies to generate socially biased output. Additionally, through manual evaluation, we discover problematic patterns in the generated chain-of-thought output that range from subtle bias to lack of reasoning.

Warning: This paper contains examples of text which are toxic, biased, and potentially harmful.

----

## [2394] MaxEnt Loss: Constrained Maximum Entropy for Calibration under Out-of-Distribution Shift

**Authors**: *Dexter Neo, Stefan Winkler, Tsuhan Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30143](https://doi.org/10.1609/aaai.v38i19.30143)

**Abstract**:

We present a new loss function that addresses the out-of-distribution (OOD) network calibration problem. While many objective functions have been proposed to effectively calibrate models in-distribution, our findings show that they do not always fare well OOD. Based on the Principle of Maximum Entropy, we incorporate helpful statistical constraints observed during training, delivering better model calibration without sacrificing accuracy. We provide theoretical analysis and show empirically that our method works well in practice, achieving state-of-the-art calibration on both synthetic and real-world benchmarks. Our code is available at https://github.com/dexterdley/MaxEnt-Loss.

----

## [2395] ORES: Open-Vocabulary Responsible Visual Synthesis

**Authors**: *Minheng Ni, Chenfei Wu, Xiaodong Wang, Shengming Yin, Lijuan Wang, Zicheng Liu, Nan Duan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30144](https://doi.org/10.1609/aaai.v38i19.30144)

**Abstract**:

Avoiding synthesizing specific visual concepts is an essential challenge in responsible visual synthesis. However, the visual concept that needs to be avoided for responsible visual synthesis tends to be diverse, depending on the region, context, and usage scenarios. In this work, we formalize a new task, Open-vocabulary Responsible Visual Synthesis (ORES), where the synthesis model is able to avoid forbidden visual concepts while allowing users to input any desired content. To address this problem, we present a Two-stage Intervention (TIN) framework. By introducing 1) rewriting with learnable instruction through a large-scale language model (LLM) and 2) synthesizing with prompt intervention on a diffusion synthesis model, it can effectively synthesize images avoiding any concepts but following the user's query as much as possible. To evaluate on ORES, we provide a publicly available dataset, baseline models, and benchmark. Experimental results demonstrate the effectiveness of our method in reducing risks of image generation. Our work highlights the potential of LLMs in responsible visual synthesis. Our code and dataset is public available in https://github.com/kodenii/ORES.

----

## [2396] Q-SENN: Quantized Self-Explaining Neural Networks

**Authors**: *Thomas Norrenbrock, Marco Rudolph, Bodo Rosenhahn*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30145](https://doi.org/10.1609/aaai.v38i19.30145)

**Abstract**:

Explanations in Computer Vision are often desired, but most Deep Neural Networks can only provide saliency maps with questionable faithfulness. Self-Explaining Neural Networks (SENN) extract interpretable concepts with fidelity, diversity, and grounding to combine them linearly for decision-making. While they can explain what was recognized, initial realizations lack accuracy and general applicability. We propose the Quantized-Self-Explaining Neural Network “Q-SENN”. Q-SENN satisfies or exceeds the desiderata of SENN while being applicable to more complex datasets and maintaining most or all of the accuracy of an uninterpretable baseline model, outperforming previous work in all considered metrics. Q-SENN describes the relationship between every class and feature as either positive, negative or neutral instead of an arbitrary number of possible relations, enforcing more binary human-friendly features. Since every class is assigned just 5 interpretable features on average, Q-SENN shows convincing local and global interpretability. Additionally, we propose a feature alignment method, capable of aligning learned features with human language-based concepts without additional supervision. Thus, what is learned can be more easily verbalized. The code is published: https://github.com/ThomasNorr/Q-SENN

----

## [2397] Understanding Likelihood of Normalizing Flow and Image Complexity through the Lens of Out-of-Distribution Detection

**Authors**: *Genki Osada, Tsubasa Takahashi, Takashi Nishide*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30146](https://doi.org/10.1609/aaai.v38i19.30146)

**Abstract**:

Out-of-distribution (OOD) detection is crucial to safety-critical machine learning applications and has been extensively studied.
While recent studies have predominantly focused on classifier-based methods, research on deep generative model (DGM)-based methods have lagged relatively.
This disparity may be attributed to a perplexing phenomenon: DGMs often assign higher likelihoods to unknown OOD inputs than to their known training data.
This paper focuses on explaining the underlying mechanism of this phenomenon.
We propose a hypothesis that less complex images concentrate in high-density regions in the latent space, resulting in a higher likelihood assignment in the Normalizing Flow (NF).
We experimentally demonstrate its validity for five NF architectures, concluding that their likelihood is untrustworthy.
Additionally, we show that this problem can be alleviated by treating image complexity as an independent variable.
Finally, we provide evidence of the potential applicability of our hypothesis in another DGM, PixelCNN++.

----

## [2398] Adversarial Initialization with Universal Adversarial Perturbation: A New Approach to Fast Adversarial Training

**Authors**: *Chao Pan, Qing Li, Xin Yao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30147](https://doi.org/10.1609/aaai.v38i19.30147)

**Abstract**:

Traditional adversarial training, while effective at improving machine learning model robustness, is computationally intensive. Fast Adversarial Training (FAT) addresses this by using a single-step attack to generate adversarial examples more efficiently. Nonetheless, FAT is susceptible to a phenomenon known as catastrophic overfitting, wherein the model's adversarial robustness abruptly collapses to zero during the training phase. To address this challenge, recent studies have suggested adopting adversarial initialization with Fast Gradient Sign Method Adversarial Training (FGSM-AT), which recycles adversarial perturbations from prior epochs by computing gradient momentum. However, our research has uncovered a flaw in this approach. Given that data augmentation is employed during the training phase, the samples in each epoch are not identical. Consequently, the method essentially yields not the adversarial perturbation of a singular sample, but rather the Universal Adversarial Perturbation (UAP) of a sample and its data augmentation. This insight has led us to explore the potential of using UAPs for adversarial initialization within the context of FGSM-AT. We have devised various strategies for adversarial initialization utilizing UAPs, including single, class-based, and feature-based UAPs. Experiments conducted on three distinct datasets demonstrate that our method achieves an improved trade-off among robustness, computational cost, and memory footprint. Code is available at https://github.com/fzjcdt/fgsm-uap.

----

## [2399] A PAC Learning Algorithm for LTL and Omega-Regular Objectives in MDPs

**Authors**: *Mateo Perez, Fabio Somenzi, Ashutosh Trivedi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i19.30148](https://doi.org/10.1609/aaai.v38i19.30148)

**Abstract**:

Linear temporal logic (LTL) and omega-regular objectives---a superset of LTL---have seen recent use as a way to express non-Markovian objectives in reinforcement learning. We introduce a model-based probably approximately correct (PAC) learning algorithm for omega-regular objectives in Markov decision processes (MDPs). As part of the development of our algorithm, we introduce the epsilon-recurrence time: a measure of the speed at which a policy converges to the satisfaction of the omega-regular objective in the limit. We prove that our algorithm only requires a polynomial number of samples in the relevant parameters, and perform experiments which confirm our theory.

----



[Go to the previous page](AAAI-2024-list11.md)

[Go to the next page](AAAI-2024-list13.md)

[Go to the catalog section](README.md)