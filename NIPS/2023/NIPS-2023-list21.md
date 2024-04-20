## [0] DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions

**Authors**: *Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, Zhaoxiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract-Conference.html)

**Abstract**:

As it is empirically observed that Vision Transformers (ViTs) are quite insensitive to the order of input tokens, the need for an appropriate self-supervised pretext task that enhances the location awareness of ViTs is becoming evident. To address this, we present DropPos, a novel pretext task designed to reconstruct Dropped Positions. The formulation of DropPos is simple: we first drop a large random subset of positional embeddings and then the model classifies the actual position for each non-overlapping patch among all possible positions solely based on their visual appearance. To avoid trivial solutions, we increase the difficulty of this task by keeping only a subset of patches visible. Additionally, considering there may be different patches with similar visual appearances, we propose position smoothing and attentive reconstruction strategies to relax this classification problem, since it is not necessary to reconstruct their exact positions in these cases. Empirical evaluations of DropPos show strong capabilities. DropPos outperforms supervised pre-training and achieves competitive results compared with state-of-the-art self-supervised alternatives on a wide range of downstream benchmarks. This suggests that explicitly encouraging spatial reasoning abilities, as DropPos does, indeed contributes to the improved location awareness of ViTs. The code is publicly available at https://github.com/Haochen-Wang409/DropPos.

----

## [0] Hierarchical VAEs provide a normative account of motion processing in the primate brain

**Authors**: *Hadi Vafaii, Jacob Yates, Daniel Butts*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/909d6b6a7c6ac13ea51de4c4cace35db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/909d6b6a7c6ac13ea51de4c4cace35db-Abstract-Conference.html)

**Abstract**:

The relationship between perception and inference, as postulated by Helmholtz in the 19th century, is paralleled in modern machine learning by generative models like Variational Autoencoders (VAEs) and their hierarchical variants. Here, we evaluate the role of hierarchical inference and its alignment with brain function in the domain of motion perception. We first introduce a novel synthetic data framework, Retinal Optic Flow Learning (ROFL), which enables control over motion statistics and their causes. We then present a new hierarchical VAE and test it against alternative models on two downstream tasks: (i) predicting ground truth causes of retinal optic flow (e.g., self-motion); and (ii) predicting the responses of neurons in the motion processing pathway of primates. We manipulate the model architectures (hierarchical versus non-hierarchical), loss functions, and the causal structure of the motion stimuli. We find that hierarchical latent structure in the model leads to several improvements. First, it improves the linear decodability of ground truth variables and does so in a sparse and disentangled manner. Second, our hierarchical VAE outperforms previous state-of-the-art models in predicting neuronal responses and exhibits sparse latent-to-neuron relationships. These results depend on the causal structure of the world, indicating that alignment between brains and artificial neural networks depends not only on architecture but also on matching ecologically relevant stimulus statistics. Taken together, our results suggest that hierarchical Bayesian inference underlines the brain's understanding of the world, and hierarchical VAEs can effectively model this understanding.

----

## [0] Variational Gaussian Processes with Decoupled Conditionals

**Authors**: *Xinran Zhu, Kaiwen Wu, Natalie Maus, Jacob Gardner, David Bindel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90bfd7201f6717b215e5dcfd987064da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90bfd7201f6717b215e5dcfd987064da-Abstract-Conference.html)

**Abstract**:

Variational Gaussian processes (GPs) approximate exact GP inference by using a small set of inducing points to form a sparse approximation of the true posterior, with the fidelity of the model increasing with additional inducing points. Although the approximation error in principle can be reduced through the use of more inducing points, this leads to scaling optimization challenges and computational complexity. To achieve scalability, inducing point methods typically introduce conditional independencies and then approximations to the training and test conditional distributions. In this paper, we consider an alternative approach to modifying the training and test conditionals, in which we make them more flexible. In particular, we investigate decoupling the parametric form of the predictive mean and covariance in the conditionals, and learn independent parameters for predictive mean and covariance. We derive new evidence lower bounds (ELBO) under these more flexible conditionals, and provide two concrete examples of applying the decoupled conditionals. Empirically, we find this additional flexibility leads to improved model performance on a variety of regression tasks and Bayesian optimization (BO) applications.

----

## [0] EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

**Authors**: *Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce EgoSchema, a very long-form video question-answering dataset, and benchmark to evaluate long video understanding capabilities of modern vision and language systems. Derived from Ego4D, EgoSchema consists of over 5000 human curated multiple choice question answer pairs, spanning over 250 hours of real video data, covering a very broad range of natural human activity and behavior. For each question, EgoSchema requires the correct answer to be selected between five given options based on a three-minute-long video clip. While some prior works have proposed video datasets with long clip lengths, we posit that merely the length of the video clip does not truly capture the temporal difficulty of the video task that is being considered. To remedy this, we introduce temporal certificate sets, a general notion for capturing the intrinsic temporal understanding length associated with a broad range of video understanding tasks & datasets. Based on this metric, we find EgoSchema to have intrinsic temporal lengths over 5.7x longer than the second closest dataset and 10x to 100x longer than any other video understanding dataset. Further, our evaluation of several current state-of-the-art video and language models shows them to be severely lacking in long-term video understanding capabilities. Even models with several billions of parameters achieve QA accuracy less than 33% (random is 20%) on the EgoSchema multi-choice question answering task, while humans achieve about 76% accuracy. We posit that EgoSchema, with its long intrinsic temporal structures and diverse complexity, would serve as a valuable evaluation probe for developing effective long-term video understanding systems in the future. Data and Zero-shot model evaluation code will all be open-sourced under the Ego4D license at http://egoschema.github.io.

----

## [0] TabMT: Generating tabular data with masked transformers

**Authors**: *Manbir Gulati, Paul F. Roysdon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html)

**Abstract**:

Autoregressive and Masked Transformers are incredibly effective as generative models and classifiers.    While these models are most prevalent in NLP, they also exhibit strong performance in other domains, such as vision.     This work contributes to the exploration of transformer-based models in synthetic data generation for diverse application domains.     In this paper, we present TabMT, a novel Masked Transformer design for generating synthetic tabular data.     TabMT effectively addresses the unique challenges posed by heterogeneous data fields and is natively able to handle missing data.     Our design leverages improved masking techniques to allow for generation and demonstrates state-of-the-art performance from extremely small to extremely large tabular datasets.     We evaluate TabMT for privacy-focused applications and find that it is able to generate high quality data with superior privacy tradeoffs.

----

## [0] Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images

**Authors**: *Gabriel Sarch, Michael J. Tarr, Katerina Fragkiadaki, Leila Wehbe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90e06fe49254204248cb12562528b952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90e06fe49254204248cb12562528b952-Abstract-Conference.html)

**Abstract**:

The alignment between deep neural network (DNN) features and cortical responses currently provides the most accurate quantitative explanation for higher visual areas. At the same time, these model features have been critiqued as uninterpretable explanations, trading one black box (the human brain) for another (a neural network). In this paper, we train networks to directly predict, from scratch, brain responses to images from a large-scale dataset of natural scenes (Allen et. al., 2021). We then use "network dissection" (Bau et. al., 2017), an explainable AI technique used for enhancing neural network interpretability by identifying and localizing the most significant features in images for individual units of a trained network, and which has been used to study category selectivity in the human brain (Khosla & Wehbe, 2022). We adapt this approach to create a hypothesis-neutral model that is then used to explore the tuning properties of specific visual regions beyond category selectivity, which we call "brain dissection". We use brain dissection to examine a range of ecologically important, intermediate properties, including depth, surface normals, curvature, and object relations across sub-regions of the parietal, lateral, and ventral visual streams, and scene-selective regions. Our findings reveal distinct preferences in brain regions for interpreting visual scenes, with ventro-lateral areas favoring closer and curvier features, medial and parietal areas opting for more varied and flatter 3D elements, and the parietal region uniquely preferring spatial relations. Scene-selective regions exhibit varied preferences, as the retrosplenial complex prefers distant and outdoor features, while the occipital and parahippocampal place areas favor proximity, verticality, and in the case of the OPA, indoor elements. Such findings show the potential of using explainable AI to uncover spatial feature selectivity across the visual cortex, contributing to a deeper, more fine-grained understanding of the functional characteristics of human visual cortex when viewing natural scenes.

----

## [0] Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models

**Authors**: *Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html)

**Abstract**:

Unlike most reinforcement learning agents which require an unrealistic amount of environment interactions to learn a new behaviour, humans excel at learning quickly by merely observing and imitating others. This ability highly depends on the fact that humans have a model of their own embodiment that allows them to infer the most likely actions that led to the observed behaviour. In this paper, we propose Action Inference by Maximising Evidence (AIME) to replicate this behaviour using world models. AIME consists of two distinct phases. In the first phase, the agent learns a world model from its past experience to understand its own body by maximising the ELBO. While in the second phase, the agent is given some observation-only demonstrations of an expert performing a novel task and tries to imitate the expert's behaviour. AIME achieves this by defining a policy as an inference model and maximising the evidence of the demonstration under the policy and world model. Our method is "zero-shot" in the sense that it does not require further training for the world model or online interactions with the environment after given the demonstration. We empirically validate the zero-shot imitation performance of our method on the Walker and Cheetah embodiment of the DeepMind Control Suite and find it outperforms the state-of-the-art baselines. Code is available at: https://github.com/argmax-ai/aime.

----

## [0] ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion

**Authors**: *Yingjun Du, Zehao Xiao, Shengcai Liao, Cees Snoek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/911dd89c81efc624c4e1c39381179505-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/911dd89c81efc624c4e1c39381179505-Abstract-Conference.html)

**Abstract**:

Prototype-based meta-learning has emerged as a powerful technique for addressing few-shot learning challenges. However, estimating a deterministic prototype using a simple average function from a limited number of examples remains a fragile process. To overcome this limitation, we introduce ProtoDiff, a novel framework that leverages a task-guided diffusion model during the meta-training phase to gradually generate prototypes, thereby providing efficient class representations. Specifically,  a set of prototypes is optimized to achieve per-task prototype overfitting, enabling accurately obtaining the overfitted prototypes for individual tasks.Furthermore, we introduce a task-guided diffusion process within the prototype space, enabling the meta-learning of a generative process that transitions from a vanilla prototype to an overfitted prototype. ProtoDiff gradually generates task-specific prototypes from random noise during the meta-test stage, conditioned on the limited samples available for the new task. Furthermore, to expedite training and enhance ProtoDiff's performance, we propose the utilization of residual prototype learning, which leverages the sparsity of the residual prototype. We conduct thorough ablation studies to demonstrate its ability to accurately capture the underlying prototype distribution and enhance generalization. The new state-of-the-art performance on within-domain, cross-domain, and few-task few-shot classiﬁcation further substantiates the beneﬁt of ProtoDiff.

----

## [0] Synthetic Experience Replay

**Authors**: *Cong Lu, Philip J. Ball, Yee Whye Teh, Jack Parker-Holder*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html)

**Abstract**:

A key theme in the past decade has been that when large neural networks and large datasets combine they can produce remarkable results. In deep reinforcement learning (RL), this paradigm is commonly made possible through experience replay, whereby a dataset of past experiences is used to train a policy or value function. However, unlike in supervised or self-supervised learning, an RL agent has to collect its own data, which is often limited. Thus, it is challenging to reap the benefits of deep learning, and even small neural networks can overfit at the start of training. In this work, we leverage the tremendous recent progress in generative modeling and propose Synthetic Experience Replay (SynthER), a diffusion-based approach to flexibly upsample an agent's collected experience. We show that SynthER is an effective method for training RL agents across offline and online settings, in both proprioceptive and pixel-based environments. In offline settings, we observe drastic improvements when upsampling small offline datasets and see that additional synthetic data also allows us to effectively train larger networks. Furthermore, SynthER enables online agents to train with a much higher update-to-data ratio than before, leading to a significant increase in sample efficiency, without any algorithmic changes. We believe that synthetic training data could open the door to realizing the full potential of deep learning for replay-based RL algorithms from limited data. Finally, we open-source our code at https://github.com/conglu1997/SynthER.

----

## [0] Learning to Tokenize for Generative Retrieval

**Authors**: *Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, Zhaochun Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html)

**Abstract**:

As a new paradigm in information retrieval, generative retrieval directly generates a ranked list of document identifiers (docids) for a given query using generative language models (LMs).How to assign each document a unique docid (denoted as document tokenization) is a critical problem, because it determines whether the generative retrieval model can precisely retrieve any document by simply decoding its docid.Most existing methods adopt rule-based tokenization, which is ad-hoc and does not generalize well.In contrast, in this paper we propose a novel document tokenization learning method, GenRet, which learns to encode the complete document semantics into docids.GenRet learns to tokenize documents into short discrete representations (i.e., docids) via a discrete auto-encoding approach.We develop a progressive training scheme to capture the autoregressive nature of docids and diverse clustering techniques to stabilize the training process.Based on the semantic-embedded docids of any set of documents, the generative retrieval model can learn to generate the most relevant docid only according to the docids' semantic relevance to the queries.We conduct experiments on the NQ320K, MS MARCO, and BEIR datasets.GenRet establishes the new state-of-the-art on the NQ320K dataset.Compared to generative retrieval baselines, GenRet can achieve significant improvements on unseen documents.Moreover, GenRet can also outperform comparable baselines on MS MARCO and BEIR, demonstrating the method's generalizability.

----

## [0] A Reduction-based Framework for Sequential Decision Making with Delayed Feedback

**Authors**: *Yunchang Yang, Han Zhong, Tianhao Wu, Bin Liu, Liwei Wang, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/915125efea950af378435518b3542e6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/915125efea950af378435518b3542e6a-Abstract-Conference.html)

**Abstract**:

We study stochastic delayed feedback in general single-agent and multi-agent sequential decision making, which includes bandits, single-agent Markov decision processes (MDPs), and Markov games (MGs). We propose a novel reduction-based framework, which turns any multi-batched algorithm for sequential decision making with instantaneous feedback into a sample-efficient algorithm that can handle stochastic delays in sequential decision making. By plugging different multi-batched algorithms into our framework, we provide several examples demonstrating that our framework not only matches or improves existing results for bandits, tabular MDPs, and tabular MGs, but also provides the first line of studies on delays in sequential decision making with function approximation. In summary, we provide a complete set of sharp results for single-agent and multi-agent sequential decision making with delayed feedback.

----

## [0] Efficient RL with Impaired Observability: Learning to Act with Delayed and Missing State Observations

**Authors**: *Minshuo Chen, Yu Bai, H. Vincent Poor, Mengdi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html)

**Abstract**:

In real-world reinforcement learning (RL) systems, various forms of {\it impaired observability} can complicate matters. These situations arise when an agent is unable to observe the most recent state of the system due to latency or lossy channels, yet the agent must still make real-time decisions. This paper introduces a theoretical investigation into efficient RL in control systems where agents must act with delayed and missing state observations. We establish near-optimal regret bounds, of the form $\tilde{\mathcal{O}}(\sqrt{{\rm poly}(H) SAK})$, for RL in both the delayed and missing observation settings. Despite impaired observability posing significant challenges to the policy class and planning, our results demonstrate that learning remains efficient, with the regret bound optimally depending on the state-action size of the original system. Additionally, we provide a characterization of the performance of the optimal policy under impaired observability, comparing it to the optimal value obtained with full observability.

----

## [0] Unified 3D Segmenter As Prototypical Classifiers

**Authors**: *Zheyun Qin, Cheng Han, Qifan Wang, Xiushan Nie, Yilong Yin, Xiankai Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/916cb4e1aeafaa0757953c9bacd17337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/916cb4e1aeafaa0757953c9bacd17337-Abstract-Conference.html)

**Abstract**:

The task of point cloud segmentation, comprising semantic, instance, and panoptic segmentation, has been mainly tackled by designing task-specific network architectures, which often lack the flexibility to generalize across tasks, thus resulting in a fragmented research landscape. In this paper, we introduce ProtoSEG, a prototype-based model that unifies semantic, instance, and panoptic segmentation tasks. Our approach treats these three homogeneous tasks as a classification problem with different levels of granularity. By leveraging a Transformer architecture, we extract point embeddings to optimize prototype-class distances and dynamically learn class prototypes to accommodate the end tasks. Our prototypical design enjoys simplicity and transparency, powerful representational learning, and ad-hoc explainability. Empirical results demonstrate that ProtoSEG outperforms concurrent well-known specialized architectures on 3D point cloud benchmarks, achieving 72.3%, 76.4% and 74.2% mIoU for semantic segmentation on S3DIS, ScanNet V2 and SemanticKITTI, 66.8% mCov and 51.2% mAP for instance segmentation on S3DIS and ScanNet V2, 62.4% PQ for panoptic segmentation on SemanticKITTI, validating the strength of our concept and the effectiveness of our algorithm.  The code and models are available at https://github.com/zyqin19/PROTOSEG.

----

## [0] Cola: A Benchmark for Compositional Text-to-image Retrieval

**Authors**: *Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan A. Plummer, Ranjay Krishna, Kate Saenko*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/917cd410aa55b61594fa2a6f6e5a9e94-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/917cd410aa55b61594fa2a6f6e5a9e94-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Compositional reasoning is a hallmark of human visual intelligence. Yet, despite the size of large vision-language models, they struggle to represent simple compositions by combining objects with their attributes. To measure this lack of compositional capability, we design Cola, a text-to-image retrieval benchmark to Compose Objects Localized with Attributes. To solve Cola, a model must retrieve images with the correct configuration of attributes and objects and avoid choosing a distractor image with the same objects and attributes but in the wrong configuration. Cola contains about 1.2k composed queries of 168 objects and 197 attributes on around 30K images. Our human evaluation finds that Cola is 83.33% accurate, similar to contemporary compositionality benchmarks. Using Cola as a testbed, we explore empirical modeling designs to adapt pre-trained vision-language models to reason compositionally. We explore 6 adaptation strategies on 2 seminal vision-language models, using compositionality-centric test benchmarks - Cola and CREPE. We find the optimal adaptation strategy is to train a multi-modal attention layer that jointly attends over the frozen pre-trained image and language features. Surprisingly, training multimodal layers on CLIP performs better than tuning a larger FLAVA model with already pre-trained multimodal layers. Furthermore, our adaptation strategy improves CLIP and FLAVA to comparable levels, suggesting that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained. Lastly, we show that Cola is harder than a closely related contemporary benchmark, CREPE, since simpler fine-tuning strategies without multimodal layers suffice on CREPE, but not on Cola. However, we still see a significant gap between our best adaptation and human accuracy, suggesting considerable room for further research. Project page: https://cs-people.bu.edu/array/research/cola/

----

## [0] Estimating Causal Effects Identifiable from a Combination of Observations and Experiments

**Authors**: *Yonghan Jung, Ivan Diaz, Jin Tian, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/917d55788726131e3bb21bf39d477f58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/917d55788726131e3bb21bf39d477f58-Abstract-Conference.html)

**Abstract**:

Learning cause and effect relations is arguably one of the central challenges found throughout the data sciences.Formally, determining whether a collection of observational and interventional distributions can be combined to learn a target causal relation is known as the problem of generalized identification (or g-identification) [Lee et al., 2019]. Although g-identification has been well understood and solved in theory, it turns out to be challenging to apply these results in practice, in particular when considering the estimation of the target distribution from finite samples. In this paper, we develop a new, general estimator that exhibits multiply robustness properties for g-identifiable causal functionals. Specifically, we show that any g-identifiable causal effect can be expressed as a function of generalized multi-outcome sequential back-door adjustments that are amenable to estimation. We then construct a corresponding estimator for the g-identification expression that exhibits robustness properties to bias. We analyze the asymptotic convergence properties of the estimator. Finally, we illustrate the use of the proposed estimator in experimental studies. Simulation results corroborate the theory.

----

## [0] LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition

**Authors**: *Haoxuan Qu, Xiaofei Hui, Yujun Cai, Jun Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91813e5ddd9658b99be4c532e274b49c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91813e5ddd9658b99be4c532e274b49c-Abstract-Conference.html)

**Abstract**:

Open-set object recognition aims to identify if an object is from a class that has been encountered during training or not. To perform open-set object recognition accurately, a key challenge is how to reduce the reliance on spurious-discriminative features. In this paper, motivated by that different large models pre-trained through different paradigms can possess very rich while distinct implicit knowledge, we propose a novel framework named Large Model Collaboration (LMC) to tackle the above challenge via collaborating different off-the-shelf large models in a training-free manner. Moreover, we also incorporate the proposed framework with several novel designs to effectively extract implicit knowledge from large models. Extensive experiments demonstrate the efficacy of our proposed framework. Code is available \href{https://github.com/Harryqu123/LMC}{here}.

----

## [0] TaskMet: Task-driven Metric Learning for Model Learning

**Authors**: *Dishank Bansal, Ricky T. Q. Chen, Mustafa Mukadam, Brandon Amos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91a5742235f70ae846436d9780e9f1d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91a5742235f70ae846436d9780e9f1d4-Abstract-Conference.html)

**Abstract**:

Deep learning models are often used with some downstream task. Models solely trained to achieve accurate predictions may struggle to perform well on the desired downstream tasks. We propose using the task loss to learn a metric which parameterizes a loss to train the model. This approach does not alter the optimal prediction model itself, but rather changes the model learning to emphasize the information important for the downstream task. This enables us to achieve the best of both worlds: a prediction model trained in the original prediction space while also being valuable for the desired downstream task. We validate our approach through experiments conducted in two main settings: 1) decision-focused model learning scenarios involving portfolio optimization and budget allocation, and 2) reinforcement learning in noisy environments with distracting states.

----

## [0] Pairwise Causality Guided Transformers for Event Sequences

**Authors**: *Xiao Shou, Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Oktie Hassanzadeh, Kristin P. Bennett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91b047c5f5bd41ef56bfaf4ad0bd19e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91b047c5f5bd41ef56bfaf4ad0bd19e3-Abstract-Conference.html)

**Abstract**:

Although pairwise causal relations have been extensively studied in observational longitudinal analyses across many disciplines, incorporating knowledge of causal pairs into deep learning models for temporal event sequences remains largely unexplored. In this paper, we propose a novel approach for enhancing the performance of transformer-based models in multivariate event sequences by injecting pairwise qualitative causal knowledge such as `event Z amplifies future occurrences of event Y'. We establish a new framework for causal inference in temporal event sequences using a transformer architecture, providing a theoretical justification for our approach, and show how to obtain unbiased estimates of the proposed measure. Experimental results demonstrate that our approach outperforms several state-of-the-art models in terms of prediction accuracy by effectively leveraging knowledge about causal pairs. We also consider a unique application where we extract knowledge around sequences of societal events by generating them from a large language model, and demonstrate how a causal knowledge graph can help with event prediction in such sequences. Overall, our framework offers a practical means of improving the performance of transformer-based models in multivariate event sequences by explicitly exploiting pairwise causal information.

----

## [0] Self-Refine: Iterative Refinement with Self-Feedback

**Authors**: *Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)

**Abstract**:

Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides *feedback* for its output and uses it to *refine* itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider.  We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by $\sim$20\% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test-time using our simple, standalone approach.

----

## [0] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors**: *Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences.To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions.We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them.We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform.Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80\% agreement, the same level of agreement between humans.Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain.Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna.The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.

----

## [0] Causal Discovery in Semi-Stationary Time Series

**Authors**: *Shanyun Gao, Raghavendra Addanki, Tong Yu, Ryan A. Rossi, Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91f9fb16b5679115a777ade51af87e48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91f9fb16b5679115a777ade51af87e48-Abstract-Conference.html)

**Abstract**:

Discovering causal relations from observational time series without making the stationary assumption is a significant challenge. In practice, this challenge is common in many areas, such as retail sales, transportation systems, and medical science. Here, we consider this problem for a class of non-stationary time series. The structural causal model (SCM) of this type of time series, called the semi-stationary time series, exhibits that a finite number of different causal mechanisms occur sequentially and periodically across time. This model holds considerable practical utility because it can represent periodicity, including common occurrences such as seasonality and diurnal variation. We propose a constraint-based, non-parametric algorithm for discovering causal relations in this setting. The resulting algorithm, PCMCI$_{\Omega}$, can capture the alternating and recurring changes in the causal mechanisms and then identify the underlying causal graph with conditional independence (CI) tests. We show that this algorithm is sound in identifying causal relations on discrete time series. We validate the algorithm with extensive experiments on continuous and discrete simulated data. We also apply our algorithm to a real-world climate dataset.

----

## [0] Fine-grained Expressivity of Graph Neural Networks

**Authors**: *Jan Böker, Ron Levie, Ningyuan Huang, Soledad Villar, Christopher Morris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9200d97ca2bf3a26db7b591844014f00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9200d97ca2bf3a26db7b591844014f00-Abstract-Conference.html)

**Abstract**:

Numerous recent works have analyzed the expressive power of message-passing graph neural networks (MPNNs), primarily utilizing combinatorial techniques such as the $1$-dimensional Weisfeiler--Leman test ($1$-WL) for the graph isomorphism problem. However, the graph isomorphism objective is inherently binary, not giving insights into the degree of similarity between two given graphs. This work resolves this issue by considering continuous extensions of both $1$-WL and MPNNs to graphons. Concretely, we show that the continuous variant of $1$-WL delivers an accurate topological characterization of the expressive power of MPNNs on graphons, revealing which graphs these networks can distinguish and the level of difficulty in separating them. We identify the finest topology where MPNNs separate points and prove a universal approximation theorem. Consequently, we provide a theoretical framework for graph and graphon similarity combining various topological variants of classical characterizations of the $1$-WL. In particular, we characterize the expressive power of MPNNs in terms of the tree distance, which is a graph distance based on the concept of fractional isomorphisms, and substructure counts via tree homomorphisms, showing that these concepts have the same expressive power as the $1$-WL and MPNNs on graphons. Empirically, we validate our theoretical findings by showing that randomly initialized MPNNs, without training, exhibit competitive performance compared to their trained counterparts. Moreover, we evaluate different MPNN architectures based on their ability to preserve graph distances, highlighting the significance of our continuous $1$-WL test in understanding MPNNs' expressivity.

----

## [0] CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

**Authors**: *Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Code completion models have made significant progress in recent years, yet current popular evaluation datasets, such as HumanEval and MBPP, predominantly focus on code completion tasks within a single file. This over-simplified setting falls short of representing the real-world software development scenario where repositories span multiple files with numerous cross-file dependencies, and accessing and understanding cross-file context is often required to complete the code correctly. To fill in this gap, we propose CrossCodeEval, a diverse and multilingual code completion benchmark that necessitates an in-depth cross-file contextual understanding to complete the code accurately. CrossCodeEval is built on a diverse set of real-world, open-sourced, permissively-licensed repositories in four popular programming languages: Python, Java, TypeScript, and C#. To create examples that strictly require cross-file context for accurate completion, we propose a straightforward yet efficient static-analysis-based approach to pinpoint the use of cross-file context within the current file. Extensive experiments on state-of-the-art code language models like CodeGen and StarCoder demonstrate that CrossCodeEval is extremely challenging when the relevant cross-file context is absent, and we see clear improvements when adding these context into the prompt. However, despite such improvements,  the pinnacle of performance remains notably unattained even with the highest-performing model,  indicating that CrossCodeEval is also capable of assessing model's capability in leveraging extensive context to make better code completion. Finally, we benchmarked various methods in retrieving cross-file context, and show that CrossCodeEval can also be used to measure the capability of code retrievers.

----

## [0] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning

**Authors**: *Yangru Huang, Peixi Peng, Yifan Zhao, Haoran Xu, Mengyue Geng, Yonghong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9213010cbcd6ba8e1f1cf1533835d51c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9213010cbcd6ba8e1f1cf1533835d51c-Abstract-Conference.html)

**Abstract**:

Integrating RGB frames with alternative modality inputs is gaining increasing traction in many vision-based reinforcement learning (RL) applications. Existing multi-modal vision-based RL methods usually follow a Global Value Estimation (GVE) pipeline, which uses a fused modality feature to obtain a unified global environmental description. However, such a feature-level fusion paradigm with a single critic may fall short in policy learning as it tends to overlook the distinct values of each modality. To remedy this, this paper proposes a Local modality-customized Value Estimation (LVE) paradigm, which dynamically estimates the contribution and adjusts the importance weight of each modality from a value-level perspective. Furthermore, a task-contextual re-fusion process is developed to achieve a task-level re-balance of estimations from both feature and value levels. To this end, a Hierarchical Adaptive Value Estimation (HAVE) framework is formed, which adaptively coordinates the contributions of individual modalities as well as their collective efficacy. Agents trained by HAVE are able to exploit the unique characteristics of various modalities while capturing their intricate interactions, achieving substantially improved performance. We specifically highlight the potency of our approach within the challenging landscape of autonomous driving, utilizing the CARLA benchmark with neuromorphic event and depth data to demonstrate HAVE's capability and the effectiveness of its distinct components.

----

## [0] Small Total-Cost Constraints in Contextual Bandits with Knapsacks, with Application to Fairness

**Authors**: *Evgenii Chzhen, Christophe Giraud, Zhen Li, Gilles Stoltz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/921dcb622bd0119c8f4f34644ce87ee0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/921dcb622bd0119c8f4f34644ce87ee0-Abstract-Conference.html)

**Abstract**:

We consider contextual bandit problems with knapsacks [CBwK], a problem where at each round, a scalar reward is obtained and vector-valued costs are suffered. The learner aims to maximize the cumulative rewards while ensuring that the cumulative costs are lower than some predetermined cost constraints. We assume that contexts come from a continuous set, that costs can be signed, and that the expected reward and cost functions, while unknown, may be uniformly estimated---a typical assumption in the literature. In this setting, total cost constraints had so far to be at least of order $T^{3/4}$, where $T$ is the number of rounds, and were even typically assumed to depend linearly on $T$. We are however motivated to use CBwK to impose a fairness constraint of equalized average costs between groups: the budget associated with the corresponding cost constraints should be as close as possible to the natural deviations, of order $\sqrt{T}$. To that end, we introduce a dual strategy based on projected-gradient-descent updates, that is able to deal with total-cost constraints of the order of $\sqrt{T}$ up to poly-logarithmic terms. This strategy is more direct and simpler than existing strategies in the literature. It relies on a careful, adaptive, tuning of the step size.

----

## [0] CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation

**Authors**: *Pengyun Zhu, Long Wen, Jinfei Liu, Feng Xue, Jian Lou, Zhibo Wang, Kui Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92225ec7e87b97a9e007ca6ab7944b14-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/92225ec7e87b97a9e007ca6ab7944b14-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A privacy policy serves as an online internet protocol crafted by service providers, which details how service providers collect, process, store, manage, and use personal information when users engage with applications. However, these privacy policies are often filled with technobabble and legalese, making them "incomprehensible''. As a result, users often agree to all terms unknowingly, even some terms may conflict with the law, thereby posing a considerable risk to personal privacy information. One potential solution to alleviate this challenge is to automatically summarize privacy policies using NLP techniques. However, existing techniques primarily focus on extracting key sentences, resulting in comparatively shorter agreements, but failing to address the poor readability caused by the "incomprehensible'' of technobabble and legalese. Moreover, research on Chinese application privacy policy summarization is currently almost nonexistent, and there is a lack of a high-quality corpus suitable for addressing readability issues. To tackle these challenges, we introduce a fine-grained CAPP-130 corpus and a TCSI-pp framework. CAPP-130 contains 130 Chinese privacy policies from popular applications that have been carefully annotated and interpreted by legal experts, resulting in 52,489 annotations and 20,555 rewritten sentences. TCSI-pp first extracts sentences related to the topic specified by users and then uses a generative model to rewrite the sentences into comprehensible summarization. Built upon TSCI-pp, we construct a summarization tool TSCI-pp-zh by selecting RoBERTa from six classification models for sentence extraction and selecting mT5 from five generative models for sentence rewriting. Experimental results show that TCSI-pp-zh outperforms GPT-4 and other baselines in Chinese application privacy policy summarization, demonstrating exceptional readability and reliability. Our data, annotation guidelines, benchmark models, and source code are publicly available at https://github.com/EnlightenedAI/CAPP-130.

----

## [0] Neural Oscillators are Universal

**Authors**: *Samuel Lanthaler, T. Konstantin Rusch, Siddhartha Mishra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/923285deb805c3e14e1aeebc9854d644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/923285deb805c3e14e1aeebc9854d644-Abstract-Conference.html)

**Abstract**:

Coupled oscillators are being increasingly used as the basis of machine learning (ML) architectures, for instance in sequence modeling, graph representation learning and in physical neural networks that are used in analog ML devices. We introduce an abstract class of neural oscillators that encompasses these architectures and prove that neural oscillators are universal, i.e, they can approximate any continuous and casual operator mapping between time-varying functions, to desired accuracy. This universality result provides theoretical justification for the use of oscillator based ML systems. The proof builds on a fundamental result of independent interest, which shows that a combination of forced harmonic oscillators with a nonlinear read-out suffices to approximate the underlying operators.

----

## [0] PAC-Bayes Generalization Certificates for Learned Inductive Conformal Prediction

**Authors**: *Apoorva Sharma, Sushant Veer, Asher Hancock, Heng Yang, Marco Pavone, Anirudha Majumdar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9235c376df778f1aaf486a882afb7471-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9235c376df778f1aaf486a882afb7471-Abstract-Conference.html)

**Abstract**:

Inductive Conformal Prediction (ICP) provides a practical and effective approach for equipping deep learning models with uncertainty estimates in the form of set-valued predictions which are guaranteed to contain the ground truth with high probability.Despite the appeal of this coverage guarantee, these sets may not be efficient: the size and contents of the prediction sets are not directly controlled, and instead depend on the underlying model and choice of score function.To remedy this, recent work has proposed learning model and score function parameters using data to directly optimize the efficiency of the ICP prediction sets.While appealing, the generalization theory for such an approach is lacking: direct optimization of empirical efficiency may yield prediction sets that are either no longer efficient on test data, or no longer obtain the required coverage on test data.In this work, we use PAC-Bayes theory to obtain generalization bounds on both the coverage and the efficiency of set-valued predictors which can be directly optimized to maximize efficiency while satisfying a desired test coverage.In contrast to prior work, our framework allows us to utilize the entire calibration dataset to learn the parameters of the model and score function, instead of requiring a separate hold-out set for obtaining test-time coverage guarantees.We leverage these theoretical results to provide a practical algorithm for using calibration data to simultaneously fine-tune the parameters of a model and score function while guaranteeing test-time coverage and efficiency of the resulting prediction sets.We evaluate the approach on regression and classification tasks, and outperform baselines calibrated using a Hoeffding bound-based PAC guarantee on ICP, especially in the low-data regime.

----

## [0] Image Captioners Are Scalable Vision Learners Too

**Authors**: *Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92369a01fbe8046a093746389b2c413e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92369a01fbe8046a093746389b2c413e-Abstract-Conference.html)

**Abstract**:

Contrastive pretraining on image-text pairs from the web is one of the most popular large-scale pretraining strategies for vision backbones, especially in the context of large multimodal models. At the same time, image captioning on this type of data is commonly considered an inferior pretraining strategy. In this paper, we perform a fair comparison of these two pretraining strategies, carefully matching training data, compute, and model capacity. Using a standard encoder-decoder transformer, we find that captioning alone is surprisingly effective: on classification tasks, captioning produces vision encoders competitive with contrastively pretrained encoders, while surpassing them on vision & language tasks. We further analyze the effect of the model architecture and scale, as well as the pretraining data on the representation quality, and find that captioning exhibits the same or better scaling behavior along these axes. Overall our results show that plain image captioning is a more powerful pretraining strategy than was previously believed. Code is available at https://github.com/google-research/big_vision.

----

## [0] Diplomat: A Dialogue Dataset for Situated PragMATic Reasoning

**Authors**: *Hengli Li, Song-Chun Zhu, Zilong Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/924303c6a45685510877ee018cdc8f80-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/924303c6a45685510877ee018cdc8f80-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The ability to discern and comprehend pragmatic meanings is a cornerstone of social and emotional intelligence, referred to as pragmatic reasoning. Despite the strides made in the development of Large Language Models (LLMs), such as ChatGPT, these models grapple with capturing the nuanced and ambiguous facets of language, falling short of the aspiration to build human-like conversational agents. In this work, we introduce a novel benchmark, the DiPlomat, which delves into the fundamental components of conversational pragmatic reasoning, encompassing situational context reasoning, open-world knowledge acquisition, and unified figurative language understanding. We start by collecting a new human-annotated dialogue dataset, composed of 4,177 multi-turn dialogues and a vocabulary of 48,900 words. Along with the dataset, two tasks are proposed to evaluate machines' pragmatic reasoning capabilities, namely, Pragmatic Reasoning and Identification(PIR) and Conversational Question Answering (CQA). Furthermore, we probe into a zero-shot natural language inference task, where the significance of context in pragmatic reasoning is underscored. Experimental findings illustrate the existing limitations of current prevailing LLMs in the realm of pragmatic reasoning, shedding light on the pressing need for further research to facilitate the emergence of emotional intelligence within human-like conversational agents.

----

## [0] CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement

**Authors**: *Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html)

**Abstract**:

Recently, multivariate time series (MTS) forecasting techniques have seen rapid development and widespread applications across various fields. Transformer-based and GNN-based methods have shown promising potential due to their strong ability to model interaction of time and variables. However, by conducting a comprehensive analysis of the real-world data, we observe that the temporal fluctuations and heterogeneity between variables are not well handled by existing methods. To address the above issues, we propose CrossGNN, a linear complexity GNN model to refine the cross-scale and cross-variable interaction for MTS. To deal with the unexpected noise in time dimension, an adaptive multi-scale identifier (AMSI) is leveraged to construct multi-scale time series with reduced noise. A Cross-Scale GNN is proposed to extract the scales with clearer trend and weaker noise. Cross-Variable GNN is proposed to utilize the homogeneity and heterogeneity between different variables. By simultaneously focusing on edges with higher saliency scores and constraining those edges with lower scores, the time and space complexity (i.e., $O(L)$) of CrossGNN can be linear with the input sequence length $L$. Extensive experimental results on 8 real-world MTS datasets demonstrate the effectiveness of CrossGNN compared with state-of-the-art methods.

----

## [0] Structured Prediction with Stronger Consistency Guarantees

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/927962d8866377a07ee3150d2d691319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/927962d8866377a07ee3150d2d691319-Abstract-Conference.html)

**Abstract**:

We present an extensive study of surrogate losses for structured prediction supported by *$H$-consistency bounds*. These are recently introduced guarantees that are more relevant to learning than Bayes-consistency, since they are not asymptotic and since they take into account the hypothesis set $H$ used. We first show that no non-trivial $H$-consistency bound can be derived for widely used surrogate structured prediction losses. We then define several new families of surrogate losses, including *structured comp-sum losses* and *structured constrained losses*, for which we prove $H$-consistency bounds and thus Bayes-consistency. These loss functions readily lead to new structured prediction algorithms with stronger theoretical guarantees, based on their minimization. We describe efficient algorithms for minimizing several of these surrogate losses, including a new *structured logistic loss*.

----

## [0] Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark

**Authors**: *Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Shangzhe Wu, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92a821f6c25b29241df6985ceb673a85-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/92a821f6c25b29241df6985ceb673a85-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Stanford-ORB, a new real-world 3D Object inverse Rendering Benchmark. Recent advances in inverse rendering have enabled a wide range of real-world applications in 3D content generation, moving rapidly from research and commercial use cases to consumer devices. While the results continue to improve, there is no real-world benchmark that can quantitatively assess and compare the performance of various inverse rendering methods. Existing real-world datasets typically only consist of the shape and multi-view images of objects, which are not sufficient for evaluating the quality of material recovery and object relighting. Methods capable of recovering material and lighting often resort to synthetic data for quantitative evaluation, which on the other hand does not guarantee generalization to complex real-world environments. We introduce a new dataset of real-world objects captured under a variety of natural scenes with ground-truth 3D scans, multi-view images, and environment lighting. Using this dataset, we establish the first comprehensive real-world evaluation benchmark for object inverse rendering tasks from in-the-wild scenes, and compare the performance of various existing methods. All data, code, and models can be accessed at https://stanfordorb.github.io/

----

## [0] Explainable Brain Age Prediction using coVariance Neural Networks

**Authors**: *Saurabh Sihag, Gonzalo Mateos, Corey McMillan, Alejandro Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92bb2145c74b7d10fbb61aba315b5010-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92bb2145c74b7d10fbb61aba315b5010-Abstract-Conference.html)

**Abstract**:

In computational neuroscience, there has been an increased interest in developing machine learning algorithms that leverage brain imaging data to provide estimates of "brain age" for an individual. Importantly, the discordance between brain age and chronological age (referred to as "brain age gap") can capture accelerated aging due to adverse health conditions and therefore, can reflect increased vulnerability towards neurological disease or cognitive impairments. However, widespread adoption of brain age for clinical decision support has been hindered due to lack of transparency and methodological justifications in most existing brain age prediction algorithms. In this paper, we leverage coVariance neural networks (VNN) to propose an explanation-driven and anatomically interpretable framework for brain age prediction using cortical thickness features. Specifically, our brain age prediction framework extends beyond the coarse metric of brain age gap in Alzheimerâ€™s disease (AD) and we make two important observations: (i) VNNs can assign anatomical interpretability to elevated brain age gap in AD by identifying contributing brain regions, (ii) the interpretability offered by VNNs is contingent on their ability to exploit specific eigenvectors of the anatomical covariance matrix. Together, these observations facilitate an explainable and anatomically interpretable perspective to the task of brain age prediction.

----

## [0] Adversarial Examples Might be Avoidable: The Role of Data Concentration in Adversarial Robustness

**Authors**: *Ambar Pal, Jeremias Sulam, René Vidal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92d21245424f3898b7110f555a00e829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92d21245424f3898b7110f555a00e829-Abstract-Conference.html)

**Abstract**:

The susceptibility of modern machine learning classifiers to adversarial examples has motivated theoretical results suggesting that these might be unavoidable. However, these results can be too general to be applicable to natural data distributions. Indeed, humans are quite robust for tasks involving vision. This apparent conflict motivates a deeper dive into the question: Are adversarial examples truly unavoidable? In this work, we theoretically demonstrate that a key property of the data distribution -- concentration on small-volume subsets of the input space -- determines whether a robust classifier exists. We further demonstrate that, for a data distribution concentrated on a union of low-dimensional linear subspaces, utilizing structure in data naturally leads to classifiers that enjoy data-dependent polyhedral robustness guarantees, improving upon methods for provable certification in certain regimes.

----

## [0] Structured State Space Models for In-Context Reinforcement Learning

**Authors**: *Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob N. Foerster, Satinder Singh, Feryal M. P. Behbahani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html)

**Abstract**:

Structured state space sequence (S4) models have recently achieved state-of-the-art performance on long-range sequence modeling tasks. These models also have fast inference speeds and parallelisable training, making them potentially useful in many reinforcement learning settings. We propose a  modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks. We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task. We evaluate our modified architecture on a set of partially-observable environments and find that, in practice, our model outperforms RNN's while also running over five times faster. Then, by leveraging the modelâ€™s ability to handle long-range sequences, we achieve strong performance on a challenging meta-learning task in which the agent is given a randomly-sampled continuous control environment, combined with a randomly-sampled linear projection of the environment's observations and actions. Furthermore, we show the resulting model can adapt to out-of-distribution held-out tasks. Overall, the results presented in this paper show that structured state space models are fast and performant for in-context reinforcement learning tasks. We provide code at https://github.com/luchris429/s5rl.

----

## [0] Sharpness-Aware Minimization Leads to Low-Rank Features

**Authors**: *Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92dd1adab39f362046f99dfe3c39d90f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92dd1adab39f362046f99dfe3c39d90f-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) is a recently proposed method that minimizes the sharpness of the training loss of a neural network. While its generalization improvement is well-known and is the primary motivation, we uncover an additional intriguing effect of SAM: reduction of the feature rank which happens at different layers of a neural network. We show that this low-rank effect occurs very broadly: for different architectures such as fully-connected networks, convolutional networks, vision transformers and for different objectives such as regression, classification, language-image contrastive training. To better understand this phenomenon, we provide a mechanistic understanding of how low-rank features arise in a simple two-layer network. We observe that a significant number of activations gets entirely pruned by SAM which directly contributes to the rank reduction. We confirm this effect theoretically and check that it can also occur in deep networks, although the overall rank reduction mechanism can be more complex, especially for deep networks with pre-activation skip connections and self-attention layers.

----

## [0] A Spectral Theory of Neural Prediction and Alignment

**Authors**: *Abdulkadir Canatar, Jenelle Feather, Albert J. Wakhloo, SueYeon Chung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9308d1b7d4ae2d3e2e67ae94b1078bf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9308d1b7d4ae2d3e2e67ae94b1078bf7-Abstract-Conference.html)

**Abstract**:

The representations of neural networks are often compared to those of biological systems by performing regression between the neural network responses and those measured from biological systems. Many different state-of-the-art deep neural networks yield similar neural predictions, but it remains unclear how to differentiate among models that perform equally well at predicting neural responses. To gain insight into this, we use a recent theoretical framework that relates the generalization error from regression to the spectral properties of the model and the target. We apply this theory to the case of regression between model activations and neural responses and decompose the neural prediction error in terms of the model eigenspectra, alignment of model eigenvectors and neural responses, and the training set size. Using this decomposition, we introduce geometrical measures to interpret the neural prediction error. We test a large number of deep neural networks that predict visual cortical activity and show that there are multiple types of geometries that result in low neural prediction error as measured via regression. The work demonstrates that carefully decomposing representational metrics can provide interpretability of how models are capturing neural activity and points the way towards improved models of neural activity.

----

## [0] Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning

**Authors**: *Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Gaetan Lin, Hao Chen, Liwei Wu, Ning Jia, Shiji Song, Gao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)

**Abstract**:

Offline-to-online reinforcement learning (RL) is a training paradigm that combines pre-training on a pre-collected dataset with fine-tuning in an online environment. However, the incorporation of online fine-tuning can intensify the well-known distributional shift problem. Existing solutions tackle this problem by imposing a policy constraint on the policy improvement objective in both offline and online learning. They typically advocate a single balance between policy improvement and constraints across diverse data collections. This one-size-fits-all manner may not optimally leverage each collected sample due to the significant variation in data quality across different states. To this end, we introduce Family Offline-to-Online RL (FamO2O), a simple yet effective framework that empowers existing algorithms to determine state-adaptive improvement-constraint balances. FamO2O utilizes a universal model to train a family of policies with different improvement/constraint intensities, and a balance model to select a suitable policy for each state. Theoretically, we prove that state-adaptive balances are necessary for achieving a higher policy performance upper bound. Empirically, extensive experiments show that FamO2O offers a statistically significant improvement over various existing methods, achieving state-of-the-art performance on the D4RL benchmark. Codes are available at https://github.com/LeapLabTHU/FamO2O.

----

## [0] Test-Time Distribution Normalization for Contrastively Learned Visual-language Models

**Authors**: *Yifei Zhou, Juntao Ren, Fengyu Li, Ramin Zabih, Ser Nam Lim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/931db0b5a61f9db6c97c7e4bf068147d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/931db0b5a61f9db6c97c7e4bf068147d-Abstract-Conference.html)

**Abstract**:

Advances in the field of visual-language contrastive learning have made it possible for many downstream applications to be carried out efficiently and accurately by simply taking the dot product between image and text representations.  One of the most representative approaches proposed recently known as CLIP has quickly garnered widespread adoption due to its effectiveness. CLIP is trained with an InfoNCE loss that takes into account both positive and negative samples to help learn a much more robust representation space. This paper however reveals that the common downstream practice of taking a dot product is only a zeroth-order approximation of the optimization goal, resulting in a loss of information during test-time. Intuitively, since the model has been optimized based on the InfoNCE loss, test-time procedures should ideally also be in alignment. The question lies in how one can retrieve any semblance of negative samples information during inference in a computationally efficient way. We propose Distribution Normalization (DN), where we approximate the mean representation of a batch of test samples and use such a mean to represent what would be analogous to negative samples in the InfoNCE loss. DN requires no retraining or fine-tuning and can be effortlessly applied during inference. Extensive experiments on a wide variety of downstream tasks exhibit a clear advantage of DN over the dot product on top of other existing test-time augmentation methods.

----

## [0] Propagating Knowledge Updates to LMs Through Distillation

**Authors**: *Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, Eunsol Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/932147114c48f8b04d41aebc0c631158-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/932147114c48f8b04d41aebc0c631158-Abstract-Conference.html)

**Abstract**:

Modern language models have the capacity to store and use immense amounts of knowledge about real-world entities, but it remains unclear how to update such knowledge stored in model parameters. While prior methods for updating knowledge in LMs successfully inject atomic facts, updated LMs fail to make inferences based on injected facts. In this work, we demonstrate that a context distillation-based approach can both impart knowledge about entities \emph{and} propagate that knowledge to enable broader inferences. Our approach consists of two stages: transfer set generation and distillation on the transfer set. We first generate a transfer set by prompting a language model to generate continuations from the entity definition. Then, we update the model parameters so that the distribution of the LM (the 'student') matches the distribution of the LM conditioned on the definition (the 'teacher') on the transfer set. Our experiments demonstrate that this approach is more effective at propagating knowledge updates than fine-tuning and other gradient-based knowledge-editing methods. Moreover, it does not  compromise performance in other contexts, even when injecting the definitions of up to 150 entities at once.

----

## [0] ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

**Authors**: *Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract-Conference.html)

**Abstract**:

Modeling continuous-time dynamics on irregular time series is critical to account for data evolution and correlations that occur continuously. Traditional methods including recurrent neural networks or Transformer models leverage inductive bias via powerful neural architectures to capture complex patterns. However, due to their discrete characteristic, they have limitations in generalizing to continuous-time data paradigms. Though neural ordinary differential equations (Neural ODEs) and their variants have shown promising results in dealing with irregular time series, they often fail to capture the intricate correlations within these sequences. It is challenging yet demanding to concurrently model the relationship between input data points and capture the dynamic changes of the continuous-time system. To tackle this problem, we propose ContiFormer that extends the relation modeling of vanilla Transformer to the continuous-time domain, which explicitly incorporates the modeling abilities of continuous dynamics of Neural ODEs with the attention mechanism of Transformers. We mathematically characterize the expressive power of ContiFormer and illustrate that, by curated designs of function hypothesis, many Transformer variants specialized in irregular time series modeling can be covered as a special case of ContiFormer. A wide range of experiments on both synthetic and real-world datasets have illustrated the superior modeling capacities and prediction performance of ContiFormer on irregular time series data. The project link is https://seqml.github.io/contiformer/.

----

## [0] Differentiable Random Partition Models

**Authors**: *Thomas M. Sutter, Alain Ryser, Joram Liebeskind, Julia E. Vogt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/933b5d002cf251b3e854d586e55ac58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/933b5d002cf251b3e854d586e55ac58c-Abstract-Conference.html)

**Abstract**:

Partitioning a set of elements into an unknown number of mutually exclusive subsets is essential in many machine learning problems.However, assigning elements, such as samples in a dataset or neurons in a network layer, to an unknown and discrete number of subsets is inherently non-differentiable, prohibiting end-to-end gradient-based optimization of parameters.We overcome this limitation by proposing a novel two-step method for inferring partitions, which allows its usage in variational inference tasks.This new approach enables reparameterized gradients with respect to the parameters of the new random partition model.Our method works by inferring the number of elements per subset and, second, by filling these subsets in a learned order.We highlight the versatility of our general-purpose approach on three different challenging experiments: variational clustering, inference of shared and independent generative factors under weak supervision, and multitask learning.

----

## [0] Connecting Pre-trained Language Model and Downstream Task via Properties of Representation

**Authors**: *Chenwei Wu, Holden Lee, Rong Ge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html)

**Abstract**:

Recently, researchers have found that representations learned by large-scale pre-trained language models are useful in various downstream tasks. However, there is little theoretical understanding of how pre-training performance is related to downstream task performance. In this paper, we analyze how this performance transfer depends on the properties of the downstream task and the structure of the representations. We consider a log-linear model where a word can be predicted from its context through a network having softmax as its last layer. We show that even if the downstream task is highly structured and depends on a simple function of the hidden representation, there are still cases when a low pre-training loss cannot guarantee good performance on the downstream task. On the other hand, we propose and empirically validate the existence of an ``anchor vector'' in the representation space, and show that this assumption, together with properties of the downstream task,  guarantees performance transfer.

----

## [0] Generalizable One-shot 3D Neural Head Avatar

**Authors**: *Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, Jan Kautz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/937ae0e83eb08d2cb8627fe1def8c751-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/937ae0e83eb08d2cb8627fe1def8c751-Abstract-Conference.html)

**Abstract**:

We present a method that reconstructs and animates a 3D head avatar from a single-view portrait image. Existing methods either involve time-consuming optimization for a specific person with multiple images, or they struggle to synthesize intricate appearance details beyond the facial region. To address these limitations, we propose a framework that not only generalizes to unseen identities based on a single-view image without requiring person-specific optimization, but also captures characteristic details within and beyond the face area (e.g. hairstyle, accessories, etc.). At the core of our method are three branches that produce three tri-planes representing the coarse 3D geometry, detailed appearance of a source image, as well as the expression of a target image. By applying volumetric rendering to the combination of the three tri-planes followed by a super-resolution module, our method yields a high fidelity image of the desired identity, expression and pose. Once trained, our model enables efficient 3D head avatar reconstruction and animation via a single forward pass through a network. Experiments show that the proposed approach generalizes well to unseen validation datasets, surpassing SOTA baseline methods by a large margin on head avatar reconstruction and animation.

----

## [0] Equivariant Single View Pose Prediction Via Induced and Restriction Representations

**Authors**: *Owen Howell, David Klee, Ondrej Biza, Linfeng Zhao, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93b3d975f9a2448964a906199db98a9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93b3d975f9a2448964a906199db98a9d-Abstract-Conference.html)

**Abstract**:

Learning about the three-dimensional world from two-dimensional images is a fundamental problem in computer vision. An ideal neural network architecture for such tasks would leverage the fact that objects can be rotated and translated in three dimensions to make predictions about novel images. However, imposing $SO(3)$-equivariance on two-dimensional inputs is difficult because the group of three-dimensional rotations does not have a natural action on the two-dimensional plane. Specifically, it is possible that an element of $SO(3)$ will rotate an image out of plane. We show that an algorithm that learns a three-dimensional representation of the world from two dimensional images must satisfy certain consistency properties which we formulate as $SO(2)$-equivariance constraints. We use the induced representation of $SO(2)$ on $SO(3)$ to construct and classify architectures that have two-dimensional inputs and which satisfy these consistency constraints. We prove that any architecture which respects said consistency constraints can be realized as an instance of our construction. We show that three previously proposed neural architectures for 3D pose prediction are special cases of our construction. We propose a new algorithm that is a learnable generalization of previously considered methods. We test our architecture on three pose predictions task and achieve SOTA results on both the PASCAL3D+ and SYMSOL pose estimation tasks.

----

## [0] Unsupervised Learning for Solving the Travelling Salesman Problem

**Authors**: *Yimeng Min, Yiwei Bai, Carla P. Gomes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93b8618a9061f8a55825c13ecf28392b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93b8618a9061f8a55825c13ecf28392b-Abstract-Conference.html)

**Abstract**:

We propose UTSP, an Unsupervised Learning (UL) framework for solving the Travelling Salesman Problem (TSP). We train a Graph Neural Network (GNN) using a surrogate loss. The GNN outputs a heat map representing the probability for each edge to be part of the optimal path. We then apply local search to generate our final prediction based on the heat map. Our loss function consists of two parts: one pushes the model to find the shortest path and the other serves as a surrogate for the constraint that the route should form a Hamiltonian Cycle. Experimental results show that UTSP outperforms the existing data-driven TSP heuristics.Our approach is parameter efficient as well as data efficient: the model takes  $\sim$ 10\% of the number of parameters and $\sim$ 0.2\% of training samples compared with Reinforcement Learning or Supervised Learning methods.

----

## [0] ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion

**Authors**: *Wei Xing, Yuxin Wang, Zheng Xing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93cf20db85fabb0fd4bb89346510629c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93cf20db85fabb0fd4bb89346510629c-Abstract-Conference.html)

**Abstract**:

Multi-fidelity fusion has become an important surrogate technique, which provides insights into expensive computer simulations and effectively improves decision-making, e.g., optimization, with less computational cost. Multi-fidelity fusion is much more computationally efficient compared to traditional single-fidelity surrogates. Despite the fast advancement of multi-fidelity fusion techniques, they lack a systematic framework to make use of the fidelity indicator, deal with high-dimensional and arbitrary data structure, and scale well to infinite-fidelity problems. In this work, we first generalize the popular autoregression (AR) to derive a novel linear fidelity differential equation (FiDE), paving the way to tractable infinite-fidelity fusion. We generalize FiDE to a high-dimensional system, which also provides a unifying framework to seemly bridge the gap between many multi- and single-fidelity GP-based models. We then propose ContinuAR, a rank-1 approximation solution to FiDEs, which is tractable to train, compatible with arbitrary multi-fidelity data structure, linearly scalable to the output dimension, and most importantly, delivers consistent SOTA performance with a significant margin over the baseline methods. Compared to the SOTA infinite-fidelity fusion, IFC, ContinuAR achieves up to 4x improvement in accuracy and 62,500x speedup in training time.

----

## [0] FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

**Authors**: *Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas N. Diggavi, Mani B. Srivastava, Tarek F. Abdelzaher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html)

**Abstract**:

This paper proposes a novel contrastive learning framework, called FOCAL, for extracting comprehensive features from multimodal time-series sensing signals through self-supervised training. Existing multimodal contrastive frameworks mostly rely on the shared information between sensory modalities, but do not explicitly consider the exclusive modality information that could be critical to understanding the underlying sensing physics. Besides, contrastive frameworks for time series have not handled the temporal information locality appropriately. FOCAL solves these challenges by making the following contributions: First, given multimodal time series, it encodes each modality into a factorized latent space consisting of shared features and private features that are orthogonal to each other. The shared space emphasizes feature patterns consistent across sensory modalities through a modal-matching objective. In contrast, the private space extracts modality-exclusive information through a transformation-invariant objective. Second, we propose a temporal structural constraint for modality features, such that the average distance between temporally neighboring samples is no larger than that of temporally distant samples. Extensive evaluations are performed on four multimodal sensing datasets with two backbone encoders and two classifiers to demonstrate the superiority of FOCAL. It consistently outperforms the state-of-the-art baselines in downstream tasks with a clear margin, under different ratios of available labels. The code and self-collected dataset are available at https://github.com/tomoyoshki/focal.

----

## [0] Assumption violations in causal discovery and the robustness of score matching

**Authors**: *Francesco Montagna, Atalanti-Anastasia Mastakouri, Elias Eulig, Nicoletta Noceti, Lorenzo Rosasco, Dominik Janzing, Bryon Aragam, Francesco Locatello*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93ed74938a54a73b5e4c52bbaf42ca8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93ed74938a54a73b5e4c52bbaf42ca8e-Abstract-Conference.html)

**Abstract**:

When domain knowledge is limited and experimentation is restricted by ethical, financial, or time constraints, practitioners turn to observational causal discovery methods to recover the causal structure, exploiting the statistical properties of their data. Because causal discovery without further assumptions is an ill-posed problem, each algorithm comes with its own set of usually untestable assumptions, some of which are hard to meet in real datasets. Motivated by these considerations, this paper extensively benchmarks the empirical performance of recent causal discovery methods on observational iid data generated under different background conditions, allowing for violations of the critical assumptions required by each selected approach. Our experimental findings show that score matching-based methods demonstrate surprising performance in the false positive and false negative rate of the inferred graph in these challenging scenarios, and we provide theoretical insights into their performance. This work is also the first effort to benchmark the stability of causal discovery algorithms with respect to the values of their hyperparameters. Finally, we hope this paper will set a new standard for the evaluation of causal discovery methods and can serve as an accessible entry point for practitioners interested in the field, highlighting the empirical implications of different algorithm choices.

----

## [0] Normalizing flow neural networks by JKO scheme

**Authors**: *Chen Xu, Xiuyuan Cheng, Yao Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93fce71def4e3cf418918805455d436f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93fce71def4e3cf418918805455d436f-Abstract-Conference.html)

**Abstract**:

Normalizing flow is a class of deep generative models for efficient sampling and likelihood estimation, which achieves attractive performance, particularly in high dimensions. The flow is often implemented using a sequence of invertible residual blocks. Existing works adopt special network architectures and regularization of flow trajectories. In this paper, we develop a neural ODE flow network called JKO-iFlow, inspired by the Jordan-Kinderleherer-Otto (JKO) scheme, which unfolds the discrete-time dynamic of the Wasserstein gradient flow. The proposed method stacks residual blocks one after another, allowing efficient block-wise training of the residual blocks, avoiding sampling SDE trajectories and score matching or variational learning, thus reducing the memory load and difficulty in end-to-end training. We also develop adaptive time reparameterization of the flow network with a progressive refinement of the induced trajectory in probability space to improve the model accuracy further. Experiments with synthetic and real data show that the proposed JKO-iFlow network achieves competitive performance compared with existing flow and diffusion models at a significantly reduced computational and memory cost.

----

## [0] Stability-penalty-adaptive follow-the-regularized-leader: Sparsity, game-dependency, and best-of-both-worlds

**Authors**: *Taira Tsuchiya, Shinji Ito, Junya Honda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9408564a4229f4a933ac9bd09a29ee96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9408564a4229f4a933ac9bd09a29ee96-Abstract-Conference.html)

**Abstract**:

Adaptivity to the difficulties of a problem is a key property in sequential decision-making problems to broaden the applicability of algorithms. Follow-the-regularized-leader (FTRL) has recently emerged as one of the most promising approaches for obtaining various types of adaptivity in bandit problems. Aiming to further generalize this adaptivity, we develop a generic adaptive learning rate, called stability-penalty-adaptive (SPA) learning rate for FTRL. This learning rate yields a regret bound jointly depending on stability and penalty of the algorithm, into which the regret of FTRL is typically decomposed. With this result, we establish several algorithms with three types of adaptivity: sparsity, game-dependency, and best-of-both-worlds (BOBW). Despite the fact that sparsity appears frequently in real problems, existing sparse multi-armed bandit algorithms with $k$-arms assume that the sparsity level $s \leq k$ is known in advance, which is often not the case in real-world scenarios. To address this issue, we first establish $s$-agnostic algorithms with regret bounds of $\tilde{O}(\sqrt{sT})$ in the adversarial regime for $T$ rounds, which matches the existing lower bound up to a logarithmic factor. Meanwhile, BOBW algorithms aim to achieve a near-optimal regret in both the stochastic and adversarial regimes. Leveraging the SPA learning rate and the technique for $s$-agnostic algorithms combined with a new analysis to bound the variation in FTRL output in response to changes in a regularizer, we establish the first BOBW algorithm with a sparsity-dependent bound. Additionally, we explore partial monitoring and demonstrate that the proposed SPA learning rate framework allows us to achieve a game-dependent bound and the BOBW simultaneously.

----

## [0] Domain Agnostic Fourier Neural Operators

**Authors**: *Ning Liu, Siavash Jafarzadeh, Yue Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html)

**Abstract**:

Fourier neural operators (FNOs) can learn highly nonlinear mappings between function spaces, and have recently become a popular tool for learning responses of complex physical systems. However, to achieve good accuracy and efficiency, FNOs rely on the Fast Fourier transform (FFT), which is restricted to modeling problems on rectangular domains. To lift such a restriction and permit FFT on irregular geometries as well as topology changes, we introduce domain agnostic Fourier neural operator (DAFNO), a novel neural operator architecture for learning surrogates with irregular geometries and evolving domains. The key idea is to incorporate a smoothed characteristic function in the integral layer architecture of FNOs, and leverage FFT to achieve rapid computations, in such a way that the geometric information is explicitly encoded in the architecture. In our empirical evaluation, DAFNO has achieved state-of-the-art accuracy as compared to baseline neural operator models on two benchmark datasets of material modeling and airfoil simulation. To further demonstrate the capability and generalizability of DAFNO in handling complex domains with topology changes, we consider a brittle material fracture evolution problem. With only one training crack simulation sample, DAFNO has achieved generalizability to unseen loading scenarios and substantially different crack patterns from the trained scenario. Our code and data accompanying this paper are available at https://github.com/ningliu-iga/DAFNO.

----

## [0] A2CiD2: Accelerating Asynchronous Communication in Decentralized Deep Learning

**Authors**: *Adel Nabli, Eugene Belilovsky, Edouard Oyallon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/940f1d0760ca52c8b21ef3b661357ec2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/940f1d0760ca52c8b21ef3b661357ec2-Abstract-Conference.html)

**Abstract**:

Distributed training of Deep Learning models has been critical to many recent successes in the field. Current standard methods primarily rely on synchronous centralized algorithms which induce major communication bottlenecks and synchronization locks at scale. Decentralized asynchronous algorithms are emerging as a potential alternative but their practical applicability still lags. In order to mitigate the increase in communication cost that naturally comes with scaling the number of workers, we introduce a principled asynchronous, randomized, gossip-based optimization algorithm which works thanks to a continuous local momentum named $\textbf{A}^2\textbf{CiD}^2$. Our method allows each worker to continuously process mini-batches without stopping, and run a peer-to-peer averaging routine in parallel, reducing idle time. In addition to inducing a significant communication acceleration at no cost other than adding a local momentum variable, minimal adaptation is required to incorporate $\textbf{A}^2\textbf{CiD}^2$ to standard asynchronous approaches. Our theoretical analysis proves accelerated rates compared to previous asynchronous decentralized baselines and we empirically show that using our $\textbf{A}^2\textbf{CiD}^2$ momentum significantly decrease communication costs in poorly connected networks. In particular, we show consistent improvement on the ImageNet dataset using up to 64 asynchronous workers (A100 GPUs) and various communication network topologies.

----

## [0] MathNAS: If Blocks Have a Role in Mathematical Architecture Design

**Authors**: *Qinsi Wang, Jinghan Ke, Zhi Liang, Sihai Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9410d94d47adfb07b41a0b226270f068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9410d94d47adfb07b41a0b226270f068-Abstract-Conference.html)

**Abstract**:

Neural Architecture Search (NAS) has emerged as a favoured method for unearthing effective neural architectures. Recent development of large models has intensified the demand for faster search speeds and more accurate search results. However, designing large models by NAS is challenging due to the dramatical increase of search space and the associated huge performance evaluation cost. Consider a typical modular search space widely used in NAS, in which a neural architecture consists of $m$ block nodes and a block node has $n$ alternative blocks. Facing the space containing $n^m$ candidate networks, existing NAS methods attempt to find the best one by searching and evaluating candidate networks directly.Different from the general strategy that takes architecture search as a whole problem, we propose a novel divide-and-conquer strategy by making use of the modular nature of the search space.Here, we introduce MathNAS, a general NAS framework based on mathematical programming.  In MathNAS, the performances of all possible building blocks in the search space are calculated first, and then the performance of a network is directly predicted based on the performances of its building blocks.Although estimating block performances involves network training, just as what happens for network performance evaluation in existing NAS methods, predicting network performance is completely training-free and thus extremely fast. In contrast to the $n^m$ candidate networks to evaluate in existing NAS methods, which requires training and a formidable computational burden, there are only $m*n$ possible blocks to handle in MathNAS.Therefore, our approach effectively reduces the complexity of network performance evaluation. The superiority of MathNAS is validated on multiple large-scale CV and NLP benchmark datasets. Notably on ImageNet-1k, MathNAS achieves 82.5\% top-1 accuracy, 1.2\% and 0.96\% higher than Swin-T and LeViT-256, respectively. In addition, when deployed on mobile device, MathNAS achieves real-time search and dynamic network switching within 1s (0.4s on TX2 GPU), surpassing baseline dynamic networks in on-device performance.

----

## [0] Block Broyden's Methods for Solving Nonlinear Equations

**Authors**: *Chengchang Liu, Cheng Chen, Luo Luo, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9417a5154519e370fd64e5a65e7dc59b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9417a5154519e370fd64e5a65e7dc59b-Abstract-Conference.html)

**Abstract**:

This paper studies quasi-Newton methods for solving nonlinear equations. We propose block variants of both good and bad Broyden's methods, which enjoy explicit local superlinear convergence rates. Our block good Broyden's method has faster condition-number-free convergence rate than existing Broyden's methods because it takes the advantage of multiple rank modification on the Jacobian estimator. On the other hand, our block bad Broyden's method directly estimates the inverse of the Jacobian provably, which reduces the computational cost of the iteration. Our theoretical results provide some new insights on why good Broyden's method outperforms bad Broyden's method in most of the cases. The empirical results also demonstrate the superiority of our methods and validate our theoretical analysis.

----

## [0] Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

**Authors**: *Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html)

**Abstract**:

Diffusion models have been shown to be capable of generating high-quality images, suggesting that they could contain meaningful internal representations. Unfortunately, the feature maps that encode a diffusion model's internal information are spread not only over layers of the network, but also over diffusion timesteps, making it challenging to extract useful descriptors. We propose Diffusion Hyperfeatures, a framework for consolidating  multi-scale and multi-timestep feature maps into per-pixel feature descriptors that can be used for downstream tasks. These descriptors can be extracted for both synthetic and real images using the generation and inversion processes. We evaluate the utility of our Diffusion Hyperfeatures on the task of semantic keypoint correspondence: our method achieves superior performance on the SPair-71k real image benchmark. We also demonstrate that our method is flexible and transferable: our feature aggregation network trained on the inversion features of real image pairs can be used on the generation features of synthetic image pairs with unseen objects and compositions. Our code is available at https://diffusion-hyperfeatures.github.io.

----

## [0] No Change, No Gain: Empowering Graph Neural Networks with Expected Model Change Maximization for Active Learning

**Authors**: *Zixing Song, Yifei Zhang, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are crucial for machine learning applications with graph-structured data, but their success depends on sufficient labeled data. We present a novel active learning (AL) method for GNNs, extending the Expected Model Change Maximization (EMCM) principle to improve prediction performance on unlabeled data. By presenting a Bayesian interpretation for the node embeddings generated by GNNs under the semi-supervised setting, we efficiently compute the closed-form EMCM acquisition function as the selection criterion for AL without re-training. Our method establishes a direct connection with expected prediction error minimization, offering theoretical guarantees for AL performance. Experiments demonstrate our method's effectiveness compared to existing approaches, in terms of both accuracy and efficiency.

----

## [0] Scaling Laws for Hyperparameter Optimization

**Authors**: *Arlind Kadra, Maciej Janowski, Martin Wistuba, Josif Grabocka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/945c781d7194ea81026148838af95af7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/945c781d7194ea81026148838af95af7-Abstract-Conference.html)

**Abstract**:

Hyperparameter optimization is an important subfield of machine learning that focuses on tuning the hyperparameters of a chosen algorithm to achieve peak performance. Recently, there has been a stream of methods that tackle the issue of hyperparameter optimization, however, most of the methods do not exploit the dominant power law nature of learning curves for Bayesian optimization. In this work, we propose Deep Power Laws (DPL), an ensemble of neural network models conditioned to yield predictions that follow a power-law scaling pattern. Our method dynamically decides which configurations to pause and train incrementally by making use of gray-box evaluations. We compare our method against 7 state-of-the-art competitors on 3 benchmarks related to tabular, image, and NLP datasets covering 59 diverse tasks. Our method achieves the best results across all benchmarks by obtaining the best any-time results compared to all competitors.

----

## [0] A Robust and Opponent-Aware League Training Method for StarCraft II

**Authors**: *Ruozi Huang, Xipeng Wu, Hongsheng Yu, Zhong Fan, Haobo Fu, Qiang Fu, Wei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94796017d01c5a171bdac520c199d9ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94796017d01c5a171bdac520c199d9ed-Abstract-Conference.html)

**Abstract**:

It is extremely difficult to train a superhuman Artificial Intelligence (AI) for games of similar size to StarCraft II. AlphaStar is the first AI that beat human professionals in the full game of StarCraft II, using a league training framework that is inspired by a game-theoretic approach. In this paper, we improve AlphaStar's league training in two significant aspects.  We train goal-conditioned exploiters, whose abilities of spotting weaknesses in the main agent and the entire league are greatly improved compared to the unconditioned exploiters in AlphaStar. In addition, we endow the agents in the league with the new ability of opponent modeling, which makes the agent more responsive to the opponent's real-time strategy. Based on these improvements, we train a better and superhuman AI with orders of magnitude less resources than AlphaStar (see Table 1 for a full comparison). Considering the iconic role of StarCraft II in game AI research, we believe our method and results on StarCraft II provide valuable design principles on how one would utilize the general league training framework for obtaining a least-exploitable strategy in various, large-scale, real-world games.

----

## [0] Causal Fairness for Outcome Control

**Authors**: *Drago Plecko, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html)

**Abstract**:

As society transitions towards an AI-based decision-making infrastructure, an ever-increasing number of decisions once under control of humans are now delegated to automated systems. Even though such developments make various parts of society more efficient, a large body of evidence suggests that a great deal of care needs to be taken to make such automated decision-making systems fair and equitable, namely, taking into account sensitive attributes such as gender, race, and religion. In this paper, we study a specific decision-making task called outcome control in which an automated system aims to optimize an outcome variable $Y$ while being fair and equitable. The interest in such a setting ranges from interventions related to criminal justice and welfare, all the way to clinical decision-making and public health. In this paper, we first analyze through causal lenses the notion of benefit, which captures how much a specific individual would benefit from a positive decision, counterfactually speaking, when contrasted with an alternative, negative one. We introduce the notion of benefit fairness, which can be seen as the minimal fairness requirement in decision-making, and develop an algorithm for satisfying it. We then note that the benefit itself may be influenced by the protected attribute, and propose causal tools which can be used to analyze this. Finally, if some of the variations of the protected attribute in the benefit are considered as discriminatory, the notion of benefit fairness may need to be strengthened, which leads us to articulating a notion of causal benefit fairness. Using this notion, we develop a new optimization procedure capable of maximizing $Y$ while ascertaining causal fairness in the decision process.

----

## [0] DeepPCR: Parallelizing Sequential Operations in Neural Networks

**Authors**: *Federico Danieli, Miguel Sarabia, Xavier Suau Cuadros, Pau Rodríguez, Luca Zappella*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html)

**Abstract**:

Parallelization techniques have become ubiquitous for accelerating inference and training of deep neural networks. Despite this, several operations are still performed in a sequential manner. For instance, the forward and backward passes are executed layer-by-layer, and the output of diffusion models is produced by applying a sequence of denoising steps. This sequential approach results in a computational cost proportional to the number of steps involved, presenting a potential bottleneck as the number of steps increases. In this work, we introduce DeepPCR, a novel algorithm which parallelizes typically sequential operations in order to speed up inference and training of neural networks. DeepPCR is based on interpreting a sequence of $L$ steps as the solution of a specific system of equations, which we recover using the Parallel Cyclic Reduction algorithm. This reduces the complexity of computing the sequential operations from $\mathcal{O}(L)$ to $\mathcal{O}(\log_2L)$, thus yielding a speedup for large $L$. To verify the theoretical lower complexity of the algorithm, and to identify regimes for speedup, we test the effectiveness of DeepPCR in parallelizing the forward and backward pass in multi-layer perceptrons, and reach speedups of up to $30\times$ for the forward and $200\times$ for the backward pass. We additionally showcase the flexibility of DeepPCR by parallelizing training of ResNets with as many as 1024 layers, and generation in diffusion models, enabling up to $7\times$ faster training and $11\times$ faster generation, respectively, when compared to the sequential approach.

----

## [0] DELTA: Diverse Client Sampling for Fasting Federated Learning

**Authors**: *Lin Wang, Yongxin Guo, Tao Lin, Xiaoying Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html)

**Abstract**:

Partial client participation has been widely adopted in Federated Learning (FL) to reduce the communication burden efficiently. However, an inadequate client sampling scheme can lead to the selection of unrepresentative subsets, resulting in significant variance in model updates and slowed convergence. Existing sampling methods are either biased or can be further optimized for faster convergence.In this paper, we present DELTA, an unbiased sampling scheme designed to alleviate these issues. DELTA characterizes the effects of client diversity and local variance, and samples representative clients with valuable information for global model updates. In addition, DELTA is a proven optimal unbiased sampling scheme that minimizes variance caused by partial client participation and outperforms other unbiased sampling schemes in terms of convergence.  Furthermore, to address full-client gradient dependence, we provide a practical version of DELTA depending on the available clients' information, and also analyze its convergence. Our results are validated through experiments on both synthetic and real-world datasets.

----

## [0] OpenAssistant Conversations - Democratizing Large Language Model Alignment

**Authors**: *Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Aligning large language models (LLMs) with human preferences has proven to drastically improve usability and has driven rapid adoption as demonstrated by ChatGPT.Alignment techniques such as supervised fine-tuning (\textit{SFT}) and  reinforcement learning from human feedback (\textit{RLHF}) greatly reduce the required skill and domain knowledge to effectively harness the capabilities of LLMs, increasing their accessibility and utility across various domains.However, state-of-the-art alignment techniques like \textit{RLHF} rely on high-quality human feedback data, which is expensive to create and often remains proprietary.In an effort to democratize research on large-scale alignment, we release OpenAssistant Conversations, a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 complete and fully annotated conversation trees.The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.Models trained on OpenAssistant Conversations show consistent improvements on standard benchmarks over respective base models.We release our code\footnote{\git} and data\footnote{\data} under a fully permissive licence.

----

## [0] Conformal Meta-learners for Predictive Inference of Individual Treatment Effects

**Authors**: *Ahmed M. Alaa, Zaid Ahmad, Mark J. van der Laan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94ab02a30b0e4a692a42ccd0b4c55399-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94ab02a30b0e4a692a42ccd0b4c55399-Abstract-Conference.html)

**Abstract**:

We investigate the problem of machine learning-based (ML) predictive inference on individual treatment effects (ITEs). Previous work has focused primarily on developing ML-based “meta-learners” that can provide point estimates of the conditional average treatment effect (CATE)—these are model-agnostic approaches for combining intermediate nuisance estimates to produce estimates of CATE. In this paper, we develop conformal meta-learners, a general framework for issuing predictive intervals for ITEs by applying the standard conformal prediction (CP) procedure on top of CATE meta-learners. We focus on a broad class of meta-learners based on two-stage pseudo-outcome regression and develop a stochastic ordering framework to study their validity. We show that inference with conformal meta-learners is marginally valid if their (pseudo-outcome) conformity scores stochastically dominate “oracle” conformity scores evaluated on the unobserved ITEs. Additionally, we prove that commonly used CATE meta-learners, such as the doubly-robust learner, satisfy a model- and distribution-free stochastic (or convex) dominance condition, making their conformal inferences valid for practically-relevant levels of target coverage. Whereas existing procedures conduct inference on nuisance parameters (i.e., potential outcomes) via weighted CP, conformal meta-learners enable direct inference on the target parameter (ITE). Numerical experiments show that conformal meta-learners provide valid intervals with competitive efficiency while retaining the favorable point estimation properties of CATE meta-learners.

----

## [0] Simple and Controllable Music Generation

**Authors**: *Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html)

**Abstract**:

We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen can generate high-quality samples, both mono and stereo, while being conditioned on textual description or melodic features, allowing better controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark. Through ablation studies, we shed light over the importance of each of the components comprising MusicGen. Music samples, code, and models are available at https://github.com/facebookresearch/audiocraft

----

## [0] Temporal Robustness against Data poisoning

**Authors**: *Wenxiao Wang, Soheil Feizi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html)

**Abstract**:

Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples when the attacks are temporally bounded. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal threat model for data poisoning.

----

## [0] Optimal Treatment Regimes for Proximal Causal Learning

**Authors**: *Tao Shen, Yifan Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94ccfdb2ca14f33a86a0b9b7d0c1bfb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94ccfdb2ca14f33a86a0b9b7d0c1bfb1-Abstract-Conference.html)

**Abstract**:

A common concern when a policymaker draws causal inferences from and makes decisions based on observational data is that the measured covariates are insufficiently rich to account for all sources of confounding, i.e., the standard no confoundedness assumption fails to hold. The recently proposed proximal causal inference framework shows that proxy variables that abound in real-life scenarios can be leveraged to identify causal effects and therefore facilitate decision-making. Building upon this line of work, we propose a novel optimal individualized treatment regime based on so-called outcome and treatment confounding bridges. We then show that the value function of this new optimal treatment regime is superior to that of existing ones in the literature. Theoretical guarantees, including identification, superiority, excess value bound, and consistency of the estimated regime, are established. Furthermore, we demonstrate the proposed optimal regime via numerical experiments and a real data application.

----

## [0] Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models

**Authors**: *Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson, Fei Sha, Leonardo Zepeda-Núñez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94d13c2401fe119e57ba325b6fe526e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94d13c2401fe119e57ba325b6fe526e0-Abstract-Conference.html)

**Abstract**:

We introduce a two-stage probabilistic framework for statistical downscaling using unpaired data. Statistical downscaling seeks a probabilistic map to transform low-resolution data from a biased coarse-grained numerical scheme to high-resolution data that is consistent with a high-fidelity scheme. Our framework tackles the problem bycomposing two transformations: (i) a debiasing step via an optimal transport map, and (ii) an upsampling step achieved by a probabilistic diffusion model with a posteriori conditional sampling. This approach characterizes a conditional distribution without needing paired data, and faithfully recovers relevant physical statistics from biased samples. We demonstrate the utility of the proposed approach on one- and two-dimensional fluid flow problems, which are representative of the core difficulties present in numerical simulations of weather and climate. Our method produces realistic high-resolution outputs from low-resolution inputs, by upsampling resolutions of $8\times$ and $16\times$. Moreover, our procedure correctly matches the statistics of physical quantities, even when the low-frequency content of the inputs and outputs do not match, a crucial but difficult-to-satisfy assumption needed by current state-of-the-art alternatives. Code for this work is available at: https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion.

----

## [0] Transformers over Directed Acyclic Graphs

**Authors**: *Yuankai Luo, Veronika Thost, Lei Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html)

**Abstract**:

Transformer models have recently gained popularity in graph representation learning as they have the potential to learn complex relationships beyond the ones captured by regular graph neural networks.The main research question is how to inject the structural bias of graphs into the transformer architecture,and several proposals have been made for undirected molecular graphs and, recently, also for larger network graphs.In this paper, we study transformers over directed acyclic graphs (DAGs) and propose architecture adaptations tailored to DAGs: (1) An attention mechanism that is considerably more efficient than the regular quadratic complexity of transformers and at the same time faithfully captures the DAG structure, and (2) a positional encoding of the DAG's partial order, complementing the former.We rigorously evaluate our approach over various types of tasks, ranging from classifying source code graphs to nodes in citation networks, and show that it is effective in two important aspects: in making graph transformers generally outperform graph neural networks tailored to DAGs and in improving SOTA graph transformer performance in terms of both quality and efficiency.

----

## [0] Understanding and Mitigating Copying in Diffusion Models

**Authors**: *Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html)

**Abstract**:

Images generated by diffusion models like Stable Diffusion are increasingly widespread. Recent works and even lawsuits have shown that these models are prone to replicating their training data, unbeknownst to the user. In this paper, we first analyze this memorization problem in text-to-image diffusion models.  While it is widely believed that duplicated images in the training set are responsible for content replication at inference time, we observe that the text conditioning of the model plays a similarly important role. In fact, we see in our experiments that data replication often does not happen for unconditional models, while it is common in the text-conditional case. Motivated by our findings, we then propose several techniques for reducing data replication at both training and inference time by randomizing and augmenting image captions in the training set. Code is available at https://github.com/somepago/DCR.

----

## [0] Credal Marginal MAP

**Authors**: *Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Fabio Cozman, Alexander G. Gray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/953390c834451505703c9da45de634d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/953390c834451505703c9da45de634d8-Abstract-Conference.html)

**Abstract**:

Credal networks extend Bayesian networks to allow for imprecision in probability values. Marginal MAP is a widely applicable mixed inference task that identifies the most likely assignment for a subset of variables (called MAP variables). However, the  task is extremely difficult to solve in credal networks particularly because the evaluation of each complete MAP assignment involves exact likelihood computations (combinatorial sums) over the vertices of a complex joint credal set representing the space of all possible marginal distributions of the MAP variables. In this paper, we explore Credal Marginal MAP inference and develop new exact methods based on variable elimination and depth-first search as well as several approximation schemes based on the mini-bucket partitioning and stochastic local search. An extensive empirical evaluation demonstrates the effectiveness of our new methods on random as well as real-world benchmark problems.

----

## [0] Multi-task Representation Learning for Pure Exploration in Bilinear Bandits

**Authors**: *Subhojyoti Mukherjee, Qiaomin Xie, Josiah Hanna, Robert D. Nowak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95464e2e49103dc560091ed2c64a5b12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95464e2e49103dc560091ed2c64a5b12-Abstract-Conference.html)

**Abstract**:

We study multi-task representation learning for the problem of pure exploration in bilinear bandits. In bilinear bandits, an action takes theform of a pair of arms from two different entity types and the reward is a bilinear function of the known feature vectors of the arms. In the \textit{multi-task bilinear bandit problem}, we aim to find optimal actions for multiple tasks that share a common low-dimensional linear representation. The objective is to leverage this characteristic to expedite the process of identifying the best pair of arms for all tasks. We propose the algorithm GOBLIN that uses an experimental design approach to optimize sample allocations for learning the global representation as well as minimize the number of samples needed to identify the optimal pair of arms in individual tasks. To the best of our knowledge, this is the first study to give sample complexity analysis for pure exploration in bilinear bandits with shared representation. Our results demonstrate that by learning the shared representation across tasks, we achieve significantly improved sample complexity compared to the traditional approach of solving tasks independently.

----

## [0] Mechanic: A Learning Rate Tuner

**Authors**: *Ashok Cutkosky, Aaron Defazio, Harsh Mehta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html)

**Abstract**:

We introduce a technique for tuning the learning rate scale factor of any base optimization algorithm and schedule automatically, which we call Mechanic. Our method provides a practical realization of recent theoretical reductions for accomplishing a similar goal in online convex optimization. We rigorously evaluate Mechanic on a range of large scale deep learning tasks with varying batch sizes, schedules, and base optimization algorithms. These experiments demonstrate that depending on the problem, Mechanic either comes very close to, matches or even improves upon manual tuning of learning rates.

----

## [0] Compositional Policy Learning in Stochastic Control Systems with Formal Guarantees

**Authors**: *Dorde Zikelic, Mathias Lechner, Abhinav Verma, Krishnendu Chatterjee, Thomas A. Henzinger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95827e011b9e899f189a01fe2f4ef316-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95827e011b9e899f189a01fe2f4ef316-Abstract-Conference.html)

**Abstract**:

Reinforcement learning has shown promising results in learning neural network policies for complicated control tasks. However, the lack of formal guarantees about the behavior of such policies remains an impediment to their deployment. We propose a novel method for learning a composition of neural network policies in stochastic environments, along with a formal certificate which guarantees that a specification over the policy's behavior is satisfied with the desired probability. Unlike prior work on verifiable RL, our approach leverages the compositional nature of logical specifications provided in SpectRL, to learn over graphs of probabilistic reach-avoid specifications. The formal guarantees are provided by learning neural network policies together with reach-avoid supermartingales (RASM) for the graph’s sub-tasks and then composing them into a global policy. We also derive a tighter lower bound compared to previous work on the probability of reach-avoidance implied by a RASM, which is required to find a compositional policy with an acceptable probabilistic threshold for complex tasks with multiple edge policies. We implement a prototype of our approach and evaluate it on a Stochastic Nine Rooms environment.

----

## [0] Fast Exact Leverage Score Sampling from Khatri-Rao Products with Applications to Tensor Decomposition

**Authors**: *Vivek Bharadwaj, Osman Asif Malik, Riley Murray, Laura Grigori, Aydin Buluç, James Demmel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html)

**Abstract**:

We present a data structure to randomly sample rows from the Khatri-Rao product of several matrices according to the exact distribution of its leverage scores. Our proposed sampler draws each row in time logarithmic in the height of the Khatri-Rao product and quadratic in its column count, with persistent space overhead at most the size of the input matrices. As a result, it tractably draws samples even when the matrices forming the Khatri-Rao product have tens of millions of rows each. When used to sketch the linear least-squares problems arising in Candecomp / PARAFAC decomposition, our method achieves lower asymptotic complexity per solve than recent state-of-the-art methods. Experiments on billion-scale sparse tensors and synthetic data validate our theoretical claims, with our algorithm achieving higher accuracy than competing methods as the decomposition rank grows.

----

## [0] Online Performative Gradient Descent for Learning Nash Equilibria in Decision-Dependent Games

**Authors**: *Zihan Zhu, Ethan Fang, Zhuoran Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html)

**Abstract**:

We study the multi-agent game within the innovative framework of decision-dependent games, which establishes a feedback mechanism that population data reacts to agentsâ€™ actions and further characterizes the strategic interactions between agents. We focus on finding the Nash equilibrium of decision-dependent games in the bandit feedback setting. However, since agents are strategically coupled, traditional gradient-based methods are infeasible without the gradient oracle. To overcome this challenge, we model the strategic interactions by a general parametric model and propose a novel online algorithm, Online Performative Gradient Descent (OPGD), which leverages the ideas of online stochastic approximation and projected gradient descent to learn the Nash equilibrium in the context of function approximation for the unknown gradient. In particular, under mild assumptions on the function classes defined in the parametric model, we prove that OPGD can find the Nash equilibrium efficiently for strongly monotone decision-dependent games. Synthetic numerical experiments validate our theory.

----

## [0] AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset

**Authors**: *Jiakang Yuan, Bo Zhang, Xiangchao Yan, Botian Shi, Tao Chen, Yikang Li, Yu Qiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95ab5c3e26fd82c7de3230bbad087d2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95ab5c3e26fd82c7de3230bbad087d2d-Abstract-Conference.html)

**Abstract**:

It is a long-term vision for Autonomous Driving (AD) community that the perception models can learn from a large-scale point cloud dataset, to obtain unified representations that can achieve promising results on different tasks or benchmarks. Previous works mainly focus on the self-supervised pre-training pipeline, meaning that they perform the pre-training and fine-tuning on the same benchmark, which is difficult to attain the performance scalability and cross-dataset application for the pre-training checkpoint.  In this paper, for the first time, we are committed to building a large-scale pre-training point-cloud dataset with diverse data distribution, and meanwhile learning generalizable representations from such a diverse pre-training dataset. We formulate the point-cloud pre-training task as a semi-supervised problem, which leverages the few-shot labeled and massive unlabeled point-cloud data to generate the unified backbone representations that can be directly applied to many baseline models and benchmarks, decoupling the AD-related pre-training process and downstream fine-tuning task. During the period of backbone pre-training, by enhancing the scene- and instance-level distribution diversity and exploiting the backbone's ability to learn from unknown instances, we achieve significant performance gains on a series of downstream perception benchmarks including Waymo, nuScenes, and KITTI, under different baseline models like PV-RCNN++, SECOND, CenterPoint.

----

## [0] Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors

**Authors**: *Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html)

**Abstract**:

Deployed language models decay over time due to shifting inputs, changing user needs, or emergent world-knowledge gaps. When such problems are identified, we want to make targeted edits while avoiding expensive retraining. However, current model editors, which modify such behaviors of pre-trained models, degrade model performance quickly across multiple, sequential edits. We propose GRACE, a \textit{lifelong} model editing method, which implements spot-fixes on streaming errors of a deployed model, ensuring minimal impact on unrelated inputs. GRACE writes new mappings into a pre-trained model's latent space, creating a discrete, local codebook of edits without altering model weights. This is the first method enabling thousands of sequential edits using only streaming errors. Our experiments on T5, BERT, and GPT models show GRACE's state-of-the-art performance in making and retaining edits, while generalizing to unseen inputs. Our code is available at github.com/thartvigsen/grace.

----

## [0] On the Identifiability of Sparse ICA without Assuming Non-Gaussianity

**Authors**: *Ignavier Ng, Yujia Zheng, Xinshuai Dong, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95b7a93e60fdfd10cc202f44fd6adf5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95b7a93e60fdfd10cc202f44fd6adf5f-Abstract-Conference.html)

**Abstract**:

Independent component analysis (ICA) is a fundamental statistical tool used to reveal hidden generative processes from observed data. However, traditional ICA approaches struggle with the rotational invariance inherent in Gaussian distributions, often necessitating the assumption of non-Gaussianity in the underlying sources. This may limit their applicability in broader contexts. To accommodate Gaussian sources, we develop an identifiability theory that relies on second-order statistics without imposing further preconditions on the distribution of sources, by introducing novel assumptions on the connective structure from sources to observed variables. Different from recent work that focuses on potentially restrictive connective structures, our proposed assumption of structural variability is both considerably less restrictive and provably necessary. Furthermore, we propose two estimation methods based on second-order statistics and sparsity constraint. Experimental results are provided to validate our identifiability theory and estimation methods.

----

## [0] Unbiased Compression Saves Communication in Distributed Optimization: When and How Much?

**Authors**: *Yutong He, Xinmeng Huang, Kun Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9602d22a8c791f23f8e4d1398e3fb5be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9602d22a8c791f23f8e4d1398e3fb5be-Abstract-Conference.html)

**Abstract**:

Communication compression is a common technique in distributed optimizationthat can alleviate communication overhead by transmitting compressed gradientsand model parameters. However, compression can introduce information distortion,which slows down convergence and incurs more communication rounds to achievedesired solutions. Given the trade-off between lower per-round communicationcosts and additional rounds of communication, it is unclear whether communicationcompression reduces the total communication cost.This paper explores the conditions under which unbiased compression, a widelyused form of compression, can reduce the total communication cost, as well as theextent to which it can do so. To this end, we present the first theoretical formulationfor characterizing the total communication cost in distributed optimization withunbiased compressors. We demonstrate that unbiased compression alone does notnecessarily save the total communication cost, but this outcome can be achievedif the compressors used by all workers are further assumed independent. Weestablish lower bounds on the communication rounds required by algorithms usingindependent unbiased compressors to minimize smooth convex functions andshow that these lower bounds are tight by refining the analysis for ADIANA.Our results reveal that using independent unbiased compression can reduce thetotal communication cost by a factor of up to $\Theta(\sqrt{\min\\{n,\kappa\\}})$ when all localsmoothness constants are constrained by a common upper bound, where $n$ is thenumber of workers and $\kappa$ is the condition number of the functions being minimized.These theoretical findings are supported by experimental results.

----

## [0] Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck

**Authors**: *Benjamin L. Edelman, Surbhi Goel, Sham M. Kakade, Eran Malach, Cyril Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/960573a3b797441aec39caa9f74bc793-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/960573a3b797441aec39caa9f74bc793-Abstract-Conference.html)

**Abstract**:

In modern deep learning, algorithmic choices (such as width, depth, and learning rate) are known to modulate nuanced resource tradeoffs. This work investigates how these complexities necessarily arise for feature learning in the presence of computational-statistical gaps. We begin by considering offline sparse parity learning, a supervised classification problem which admits a statistical query lower bound for gradient-based training of a multilayer perceptron. This lower bound can be interpreted as a multi-resource tradeoff frontier: successful learning can only occur if one is sufficiently rich (large model), knowledgeable (large dataset), patient (many training iterations), or lucky (many random guesses). We show, theoretically and experimentally, that sparse initialization and increasing network width yield significant improvements in sample efficiency in this setting. Here, width plays the role of parallel search: it amplifies the probability of finding "lottery ticket" neurons, which learn sparse features more sample-efficiently. Finally, we show that the synthetic sparse parity task can be useful as a proxy for real problems requiring axis-aligned feature learning. We demonstrate improved sample efficiency on tabular classification benchmarks by using wide, sparsely-initialized MLP models; these networks sometimes outperform tuned random forests.

----

## [0] Reliable learning in challenging environments

**Authors**: *Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96189e90e599ccc43f00434ff3ed0312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96189e90e599ccc43f00434ff3ed0312-Abstract-Conference.html)

**Abstract**:

The problem of designing learners that provide guarantees that their predictions are provably correct is of increasing importance in machine learning. However, learning theoretic guarantees have only been considered in very specific settings.  In this work, we consider the design and analysis of reliable learners in challenging test-time environments as encountered in modern machine learning problems: namely adversarial test-time attacks (in several variations) and natural distribution shifts.  In this work, we provide a reliable learner with provably optimal guarantees in such settings. We discuss computationally feasible implementations of the learner and further show that our algorithm achieves strong positive performance guarantees on several natural examples: for example, linear separators under log-concave distributions or smooth boundary classifiers under smooth probability distributions.

----

## [0] Retaining Beneficial Information from Detrimental Data for Neural Network Repair

**Authors**: *Long-Kai Huang, Peilin Zhao, Junzhou Huang, Sinno Jialin Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html)

**Abstract**:

The performance of deep learning models heavily relies on the quality of the training data. Inadequacies in the training data, such as corrupt input or noisy labels, can lead to the failure of model generalization. Recent studies propose repairing the model by identifying the training samples that contribute to the failure and removing their influence from the model. However, it is important to note that the identified data may contain both beneficial and detrimental information. Simply erasing the information of the identified data from the model can have a negative impact on its performance, especially when accurate data is mistakenly identified as detrimental and removed. To overcome this challenge, we propose a novel approach that leverages the knowledge obtained from a retained clean set. Our method first identifies harmful data by utilizing the clean set, then separates the beneficial and detrimental information within the identified data. Finally, we utilize the extracted beneficial information to enhance the model's performance. Through empirical evaluations, we demonstrate that our method outperforms baseline approaches in both identifying harmful data and rectifying model failures. Particularly in scenarios where identification is challenging and a significant amount of benign data is involved, our method improves performance while the baselines deteriorate due to the erroneous removal of beneficial information.

----

## [0] Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera

**Authors**: *Lujie Xia, Ziluo Ding, Rui Zhao, Jiyuan Zhang, Lei Ma, Zhaofei Yu, Tiejun Huang, Ruiqin Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html)

**Abstract**:

Efficiently selecting an appropriate spike stream data length to extract precise information is the key to the spike vision tasks. To address this issue, we propose a dynamic timing representation for spike streams. Based on multi-layers architecture, it applies dilated convolutions on temporal dimension to extract features on multi-temporal scales with few parameters. And we design layer attention to dynamically fuse these features. Moreover, we propose an unsupervised learning method for optical flow estimation in a spike-based manner to break the dependence on labeled data. In addition, to verify the robustness, we also build a spike-based synthetic validation dataset for extreme scenarios in autonomous driving, denoted as SSES dataset. It consists of various corner cases. Experiments show that our method can predict optical flow from spike streams in different high-speed scenes, including real scenes. For instance, our method achieves $15\%$ and $19\%$ error reduction on PHM dataset compared to the best spike-based work, SCFlow, in $\Delta t=10$ and $\Delta t=20$ respectively, using the same settings as in previous works. The source code and dataset are available at \href{https://github.com/Bosserhead/USFlow}{https://github.com/Bosserhead/USFlow}.

----

## [0] No-regret Algorithms for Fair Resource Allocation

**Authors**: *Abhishek Sinha, Ativ Joshi, Rajarshi Bhattacharjee, Cameron Musco, Mohammad Hajiesmaili*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96842011407c2691ab4eefff48fc864d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96842011407c2691ab4eefff48fc864d-Abstract-Conference.html)

**Abstract**:

We consider a fair resource allocation problem in the no-regret setting against an unrestricted adversary. The objective is to allocate resources equitably among several agents in an online fashion so that the difference of the aggregate $\alpha$-fair utilities of the agents achieved by an optimal static clairvoyant allocation and the online policy grows sublinearly with time. The problem inherits its difficulty from the non-separable nature of the global $\alpha$-fairness function. Previously, it was shown that no online policy could achieve a sublinear standard regret in this problem. In this paper, we propose an efficient online resource allocation policy, called Online Fair Allocation ($\texttt{OFA}$), that achieves sublinear $c_\alpha$-approximate regret with approximation factor $c_\alpha=(1-\alpha)^{-(1-\alpha)}\leq 1.445,$ for $0\leq \alpha < 1$. Our upper bound on the $c_\alpha$-regret for this problem exhibits a surprising \emph{phase transition} phenomenon -- transitioning from a power-law to a constant at the critical exponent $\alpha=\frac{1}{2}.$  Our result also resolves an open problem in designing an efficient no-regret policy for the online job scheduling problem in certain parameter regimes. Along the way, we introduce new algorithmic and analytical techniques, including greedy estimation of the future gradients for non-additive global reward functions and bootstrapping second-order regret bounds, which may be of independent interest.

----

## [0] Bypass Exponential Time Preprocessing: Fast Neural Network Training via Weight-Data Correlation Preprocessing

**Authors**: *Josh Alman, Jiehao Liang, Zhao Song, Ruizhe Zhang, Danyang Zhuo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9690d4746230cfea3d067fca695ba648-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9690d4746230cfea3d067fca695ba648-Abstract-Conference.html)

**Abstract**:

Over the last decade, deep neural networks have transformed our society, and they are already widely applied in various machine learning applications. State-of-the-art deep neural networks are becoming larger in size every year to deliver increasing model accuracy, and as a result, model training consumes substantial computing resources and will only consume more in the future.Using current training methods, in each iteration, to process a data point $x \in \mathbb{R}^d$ in a layer, we need to spend $\Theta(md)$ time to evaluate all the $m$ neurons in the layer. This means processing the entire layer takes $\Theta(nmd)$ time for $n$ data points. Recent work [Song, Yang and Zhang, NeurIPS 2021] reduces this time per iteration to $o(nmd)$, but requires exponential time to preprocess either the data or the neural network weights, making it unlikely to have practical usage. In this work, we present a new preprocessing method that simply stores the weight-data correlation in a tree data structure in order to quickly and dynamically detect which neurons fire at each iteration. Our method requires only $O(nmd)$ time in preprocessing and still achieves $o(nmd)$ time per iteration. We complement our new algorithm with a lower bound, proving that assuming a popular conjecture from complexity theory, one could not substantially speed up our algorithm for dynamic detection of firing neurons.

----

## [0] Online PCA in Converging Self-consistent Field Equations

**Authors**: *Xihan Li, Xiang Chen, Rasul Tutunov, Haitham Bou-Ammar, Lei Wang, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/969c14957c0df5ce2db642b3a5fa985c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/969c14957c0df5ce2db642b3a5fa985c-Abstract-Conference.html)

**Abstract**:

Self-consistent Field (SCF) equation is a type of nonlinear eigenvalue problem in which the matrix to be eigen-decomposed is a function of its own eigenvectors. It is of great significance in computational science for its connection to the Schr√∂dinger equation. Traditional fixed-point iteration methods for solving such equations suffer from non-convergence issues. In this work, we present a novel perspective on such SCF equations as a principal component analysis (PCA) for non-stationary time series, in which a distribution and its own top principal components are mutually updated over time, and the equilibrium state of the model corresponds to the solution of the SCF equations. By the new perspective, online PCA techniques are able to engage in so as to enhance the convergence of the model towards the equilibrium state, acting as a new set of tools for converging the SCF equations. With several numerical adaptations, we then develop a new algorithm for converging the SCF equation, and demonstrated its high convergence capacity with experiments on both synthesized and real electronic structure scenarios.

----

## [0] DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing

**Authors**: *Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96a54c09569ebbdd9ecb22f5012e6b66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96a54c09569ebbdd9ecb22f5012e6b66-Abstract-Conference.html)

**Abstract**:

Proteins play a critical role in carrying out biological functions, and their 3D structures are essential in determining their functions. Accurately predicting the conformation of protein side-chains given their backbones is important for applications in protein structure prediction, design and protein-protein interactions. Traditional methods are computationally intensive and have limited accuracy, while existing machine learning methods treat the problem as a regression task and overlook the restrictions imposed by the constant covalent bond lengths and angles. In this work, we present DiffPack, a torsional diffusion model that learns the joint distribution of side-chain torsional angles, the only degrees of freedom in side-chain packing, by diffusing and denoising on the torsional space. To avoid issues arising from simultaneous perturbation of all four torsional angles, we propose autoregressively generating the four torsional angles from $\chi_1$ to $\chi_4$ and training diffusion models for each torsional angle. We evaluate the method on several benchmarks for protein side-chain packing and show that our method achieves improvements of 11.9% and 13.5% in angle accuracy on CASP13 and CASP14, respectively, with a significantly smaller model size ($60\times$ fewer parameters). Additionally, we show the effectiveness of our method  in enhancing side-chain predictions in the AlphaFold2 model. Code is available at https://github.com/DeepGraphLearning/DiffPack.

----

## [0] ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections

**Authors**: *Chun-Han Yao, Amit Raj, Wei-Chih Hung, Michael Rubinstein, Yuanzhen Li, Ming-Hsuan Yang, Varun Jampani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html)

**Abstract**:

Estimating 3D articulated shapes like animal bodies from monocular images is inherently challenging due to the ambiguities of camera viewpoint, pose, texture, lighting, etc. We propose ARTIC3D, a self-supervised framework to reconstruct per-instance 3D shapes from a sparse image collection in-the-wild. Specifically, ARTIC3D is built upon a skeleton-based surface representation and is further guided by 2D diffusion priors from Stable Diffusion. First, we enhance the input images with occlusions/truncation via 2D diffusion to obtain cleaner mask estimates and semantic features. Second, we perform diffusion-guided 3D optimization to estimate shape and texture that are of high-fidelity and faithful to input images. We also propose a novel technique to calculate more stable image-level gradients via diffusion models compared to existing alternatives. Finally, we produce realistic animations by fine-tuning the rendered shape and texture under rigid part transformations. Extensive evaluations on multiple existing datasets as well as newly introduced noisy web image collections with occlusions and truncation demonstrate that ARTIC3D outputs are more robust to noisy images, higher quality in terms of shape and texture details, and more realistic when animated.

----

## [0] Noether Embedding: Efficient Learning of Temporal Regularities

**Authors**: *Chi Gao, Zidong Zhou, Luping Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html)

**Abstract**:

Learning to detect and encode temporal regularities (TRs) in events is a prerequisite for human-like intelligence. These regularities should be formed from limited event samples and stored as easily retrievable representations. Existing event embeddings, however, cannot effectively decode TR validity with well-trained vectors, let alone satisfy the efficiency requirements. We develop Noether Embedding (NE) as the first efficient TR learner with event embeddings. Specifically, NE possesses the intrinsic time-translation symmetries of TRs indicated as conserved local energies in the embedding space. This structural bias reduces the calculation of each TR validity to embedding each event sample, enabling NE to achieve data-efficient TR formation insensitive to sample size and time-efficient TR retrieval in constant time complexity. To comprehensively evaluate the TR learning capability of embedding models, we define complementary tasks of TR detection and TR query, formulate their evaluation metrics, and assess embeddings on classic ICEWS14, ICEWS18, and GDELT datasets. Our experiments demonstrate that NE consistently achieves about double the F1 scores for detecting valid TRs compared to classic embeddings, and it provides over ten times higher confidence scores for querying TR intervals. Additionally, we showcase NE's potential applications in social event prediction, personal decision-making, and memory-constrained scenarios.

----

## [0] TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning

**Authors**: *Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html)

**Abstract**:

Despite recent progress in reinforcement learning (RL) from raw pixel data, sample inefficiency continues to present a substantial obstacle. Prior works have attempted to address this challenge by creating self-supervised auxiliary tasks, aiming to enrich the agent's learned representations with control-relevant information for future state prediction.However, these objectives are often insufficient to learn representations that can represent the optimal policy or value function, and they often consider tasks with small, abstract discrete action spaces and thus overlook the importance of action representation learning in continuous control.In this paper, we introduce $\texttt{TACO}$: $\textbf{T}$emporal $\textbf{A}$ction-driven $\textbf{CO}$ntrastive Learning, a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. $\texttt{TACO}$ simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states. Theoretically, $\texttt{TACO}$ can be shown to learn state and action representations that encompass sufficient information for control, thereby improving sample efficiency.For online RL, $\texttt{TACO}$ achieves 40% performance boost after one million environment interaction steps on average across nine challenging visual continuous control tasks from Deepmind Control Suite. In addition, we show that $\texttt{TACO}$ can also serve as a plug-and-play module adding to existing offline visual RL methods to establish the new state-of-the-art performance for offline visual RL across offline datasets with varying quality.

----

## [0] On the choice of Perception Loss Function for Learned Video Compression

**Authors**: *Sadaf Salehkalaibar, Buu Phan, Jun Chen, Wei Yu, Ashish Khisti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html)

**Abstract**:

We study causal, low-latency, sequential video compression when the output is subjected to both a mean squared-error (MSE) distortion loss as well as a perception loss to target realism. Motivated by prior approaches, we consider two different perception loss functions (PLFs). The first, PLF-JD,  considers the joint distribution (JD) of all the video frames up to the current one, while the second metric, PLF-FMD,  considers the framewise marginal distributions (FMD) between the source and reconstruction. Using information theoretic analysis and deep-learning based experiments, we demonstrate that the choice of PLF can have a significant effect on the reconstruction, especially at low-bit rates. In particular, while the reconstruction based on PLF-JD can better preserve the temporal correlation across frames, it also imposes a significant penalty in distortion  compared to PLF-FMD and further makes it more difficult to recover from errors  made in the earlier output frames. Although the choice of PLF decisively affects  reconstruction quality, we also demonstrate that it may not be essential to commit to a particular PLF during encoding and the choice of PLF can be delegated to the decoder. In particular, encoded representations generated by training a system to minimize the MSE (without requiring either PLF) can be  {\em near universal}  and can generate close to optimal reconstructions for either choice of PLF at the decoder.  We validate our results using (one-shot) information-theoretic analysis, detailed study of the rate-distortion-perception tradeoff of the Gauss-Markov source model as well as deep-learning based experiments on moving MNIST and KTH datasets.

----

## [0] Imitation Learning from Vague Feedback

**Authors**: *Xin-Qiang Cai, Yu-Jie Zhang, Chao-Kai Chiang, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96e35b532b4932a86cce8c929ff3f960-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96e35b532b4932a86cce8c929ff3f960-Abstract-Conference.html)

**Abstract**:

Imitation learning from human feedback studies how to train well-performed imitation agents with an annotator's relative comparison of two demonstrations (one demonstration is better/worse than the other), which is usually easier to collect than the perfect expert data required by traditional imitation learning. However, in many real-world applications, it is still expensive or even impossible to provide a clear pairwise comparison between two demonstrations with similar quality. This motivates us to study the problem of imitation learning with vague feedback, where the data annotator can only distinguish the paired demonstrations correctly when their quality differs significantly, i.e., one from the expert and another from the non-expert. By modeling the underlying demonstration pool as a mixture of expert and non-expert data, we show that the expert policy distribution can be recovered when the proportion $\alpha$ of expert data is known. We also propose a mixture proportion estimation method for the unknown $\alpha$ case. Then, we integrate the recovered expert policy distribution with generative adversarial imitation learning to form an end-to-end algorithm. Experiments show that our methods outperform standard and preference-based imitation learning methods on various tasks.

----

## [0] Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination

**Authors**: *Yuchen Bai, Jean-Baptiste Durand, Grégoire Vincent, Florence Forbes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html)

**Abstract**:

Lidar (Light Detection and Ranging) has become an essential part of the remote sensing toolbox used for biosphere monitoring. In particular, Lidar provides the opportunity to map forest leaf area with unprecedented accuracy, while leaf area has remained an important source of uncertainty affecting models of gas exchanges between the vegetation and the atmosphere. Unmanned Aerial Vehicles (UAV) are easy to mobilize and therefore allow frequent revisits to track the response of vegetation to climate change. However, miniature sensors embarked on UAVs usually provide point clouds of limited density, which are further affected by a strong decrease in density from top to bottom of the canopy due to progressively stronger occlusion. In such a context, discriminating leaf points from wood points presents a significant challenge due in particular to strong class imbalance and spatially irregular sampling intensity. Here we introduce a neural network model based on the Pointnet ++ architecture which makes use of point geometry only (excluding any spectral information). To cope with local data sparsity, we propose an innovative sampling scheme which strives to preserve local important geometric information. We also propose a loss function adapted to the severe class imbalance. We show that our model outperforms state-of-the-art alternatives on UAV point clouds. We discuss future possible improvements, particularly regarding much denser point clouds acquired from below the canopy.

----

## [0] Max-Margin Token Selection in Attention Mechanism

**Authors**: *Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, Samet Oymak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html)

**Abstract**:

Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(X)=\langle Xv, \texttt{softmax}(XWp)\rangle$, where $X$ is the token sequence and $(v,W,p)$ are trainable parameters. We prove that running gradient descent on $p$, or equivalently $W$, converges in direction to a max-margin solution that separates *locally-optimal* tokens from non-optimal ones. This clearly formalizes attention as an optimal token selection mechanism. Remarkably, our results are applicable to general data and precisely characterize *optimality* of tokens in terms of the value embeddings $Xv$ and problem geometry. We also provide a broader regularization path analysis that establishes the margin maximizing nature of attention even for nonlinear prediction heads. When optimizing $v$ and $p$ simultaneously with logistic loss, we identify conditions under which the regularization paths directionally converge to their respective hard-margin SVM solutions where $v$ separates the input features based on their labels. Interestingly, the SVM formulation of $p$ is influenced by the support vector geometry of $v$. Finally, we verify our theoretical findings via numerical experiments and provide insights.

----

## [0] Locality-Aware Generalizable Implicit Neural Representation

**Authors**: *Doyup Lee, Chiheon Kim, Minsu Cho, Wook-Shin Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html)

**Abstract**:

Generalizable implicit neural representation (INR) enables a single continuous function, i.e., a coordinate-based neural network, to represent multiple data instances by modulating its weights or intermediate features using latent codes. However, the expressive power of the state-of-the-art modulation is limited due to its inability to localize and capture fine-grained details of data entities such as specific pixels and rays. To address this issue, we propose a novel framework for generalizable INR that combines a transformer encoder with a locality-aware INR decoder. The transformer encoder predicts a set of latent tokens from a data instance to encode local information into each latent token. The locality-aware INR decoder extracts a modulation vector by selectively aggregating the latent tokens via cross-attention for a coordinate input and then predicts the output by progressively decoding with coarse-to-fine modulation through multiple frequency bandwidths. The selective token aggregation and the multi-band feature modulation enable us to learn locality-aware representation in spatial and spectral aspects, respectively. Our framework significantly outperforms previous generalizable INRs and validates the usefulness of the locality-aware latents for downstream tasks such as image generation.

----

## [0] StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners

**Authors**: *Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html)

**Abstract**:

We investigate the potential of learning visual representations using synthetic images generated by text-to-image models. This is a natural question in the light of the excellent performance of such models in generating high-quality images. We consider specifically the Stable Diffusion, one of the leading open source text-to-image models. We show that (1) when the generative model is properly configured, training self-supervised methods on synthetic images can match or beat the real image counterpart;(2) by treating the multiple images generated from the same text prompt as positives for each other, we develop a multi-positive contrastive learning method, which we call StableRep. With solely synthetic images, the representations learned by StableRep surpass the performance of representations learned by SimCLR and CLIP using the same set of text prompts and corresponding real images, on large scale datasets. When we further add language supervision, \name~trained with 20M synthetic images (10M captions) achieves better accuracy than CLIP trained with 50M real images (50M captions).

----

## [0] DropCompute: simple and more robust distributed synchronous training via compute variance reduction

**Authors**: *Niv Giladi, Shahar Gottlieb, Moran Shkolnik, Asaf Karnieli, Ron Banner, Elad Hoffer, Kfir Y. Levy, Daniel Soudry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html)

**Abstract**:

Background: Distributed training is essential for large scale training of deep neural networks (DNNs). The dominant methods for large scale DNN training are synchronous (e.g. All-Reduce), but these require waiting for all workers in each step. Thus, these methods are limited by the delays caused by straggling workers.Results: We study a typical scenario in which workers are straggling due to variability in compute time. We find an analytical relation between compute time properties and scalability limitations, caused by such straggling workers. With these findings, we propose a simple yet effective decentralized method to reduce the variation among workers and thus improve the robustness of synchronous training. This method can be integrated with the widely used All-Reduce. Our findings are validated on large-scale training tasks using 200 Gaudi Accelerators.

----

## [0] A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

**Authors**: *Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/973a0f50d43cf99118cdab456edcacda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/973a0f50d43cf99118cdab456edcacda-Abstract-Conference.html)

**Abstract**:

Real-world datasets are typically imbalanced in the sense that only a few classes have numerous samples, while many classes are associated with only a few samples. As a result, a naive ERM learning process will be biased towards the majority classes, making it difficult to generalize to the minority classes. To address this issue, one simple but effective approach is to modify the loss function to emphasize the learning on minority classes, such as re-weighting the losses or adjusting the logits via class-dependent terms. However, existing generalization analysis of such losses is still coarse-grained and fragmented, failing to explain some empirical results. To bridge this gap between theory and practice, we propose a novel technique named data-dependent contraction to capture how these modified losses handle different classes. On top of this technique, a fine-grained generalization bound is established for imbalanced learning, which helps reveal the mystery of re-weighting and logit-adjustment in a unified manner. Furthermore, a principled learning algorithm is developed based on the theoretical insights. Finally, the empirical results on benchmark datasets not only validate the theoretical results but also demonstrate the effectiveness of the proposed method.

----



[Go to the previous page](NIPS-2023-list2000.md)

[Go to the next page](NIPS-2023-list2002.md)

[Go to the catalog section](README.md)