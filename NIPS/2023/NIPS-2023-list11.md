## [2000] DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions

**Authors**: *Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, Zhaoxiang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9098e2901b4eb54772f83535f89cb8ac-Abstract-Conference.html)

**Abstract**:

As it is empirically observed that Vision Transformers (ViTs) are quite insensitive to the order of input tokens, the need for an appropriate self-supervised pretext task that enhances the location awareness of ViTs is becoming evident. To address this, we present DropPos, a novel pretext task designed to reconstruct Dropped Positions. The formulation of DropPos is simple: we first drop a large random subset of positional embeddings and then the model classifies the actual position for each non-overlapping patch among all possible positions solely based on their visual appearance. To avoid trivial solutions, we increase the difficulty of this task by keeping only a subset of patches visible. Additionally, considering there may be different patches with similar visual appearances, we propose position smoothing and attentive reconstruction strategies to relax this classification problem, since it is not necessary to reconstruct their exact positions in these cases. Empirical evaluations of DropPos show strong capabilities. DropPos outperforms supervised pre-training and achieves competitive results compared with state-of-the-art self-supervised alternatives on a wide range of downstream benchmarks. This suggests that explicitly encouraging spatial reasoning abilities, as DropPos does, indeed contributes to the improved location awareness of ViTs. The code is publicly available at https://github.com/Haochen-Wang409/DropPos.

----

## [2001] Hierarchical VAEs provide a normative account of motion processing in the primate brain

**Authors**: *Hadi Vafaii, Jacob Yates, Daniel Butts*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/909d6b6a7c6ac13ea51de4c4cace35db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/909d6b6a7c6ac13ea51de4c4cace35db-Abstract-Conference.html)

**Abstract**:

The relationship between perception and inference, as postulated by Helmholtz in the 19th century, is paralleled in modern machine learning by generative models like Variational Autoencoders (VAEs) and their hierarchical variants. Here, we evaluate the role of hierarchical inference and its alignment with brain function in the domain of motion perception. We first introduce a novel synthetic data framework, Retinal Optic Flow Learning (ROFL), which enables control over motion statistics and their causes. We then present a new hierarchical VAE and test it against alternative models on two downstream tasks: (i) predicting ground truth causes of retinal optic flow (e.g., self-motion); and (ii) predicting the responses of neurons in the motion processing pathway of primates. We manipulate the model architectures (hierarchical versus non-hierarchical), loss functions, and the causal structure of the motion stimuli. We find that hierarchical latent structure in the model leads to several improvements. First, it improves the linear decodability of ground truth variables and does so in a sparse and disentangled manner. Second, our hierarchical VAE outperforms previous state-of-the-art models in predicting neuronal responses and exhibits sparse latent-to-neuron relationships. These results depend on the causal structure of the world, indicating that alignment between brains and artificial neural networks depends not only on architecture but also on matching ecologically relevant stimulus statistics. Taken together, our results suggest that hierarchical Bayesian inference underlines the brain's understanding of the world, and hierarchical VAEs can effectively model this understanding.

----

## [2002] Variational Gaussian Processes with Decoupled Conditionals

**Authors**: *Xinran Zhu, Kaiwen Wu, Natalie Maus, Jacob Gardner, David Bindel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90bfd7201f6717b215e5dcfd987064da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90bfd7201f6717b215e5dcfd987064da-Abstract-Conference.html)

**Abstract**:

Variational Gaussian processes (GPs) approximate exact GP inference by using a small set of inducing points to form a sparse approximation of the true posterior, with the fidelity of the model increasing with additional inducing points. Although the approximation error in principle can be reduced through the use of more inducing points, this leads to scaling optimization challenges and computational complexity. To achieve scalability, inducing point methods typically introduce conditional independencies and then approximations to the training and test conditional distributions. In this paper, we consider an alternative approach to modifying the training and test conditionals, in which we make them more flexible. In particular, we investigate decoupling the parametric form of the predictive mean and covariance in the conditionals, and learn independent parameters for predictive mean and covariance. We derive new evidence lower bounds (ELBO) under these more flexible conditionals, and provide two concrete examples of applying the decoupled conditionals. Empirically, we find this additional flexibility leads to improved model performance on a variety of regression tasks and Bayesian optimization (BO) applications.

----

## [2003] EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding

**Authors**: *Karttikeya Mangalam, Raiymbek Akshulakov, Jitendra Malik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/90ce332aff156b910b002ce4e6880dec-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce EgoSchema, a very long-form video question-answering dataset, and benchmark to evaluate long video understanding capabilities of modern vision and language systems. Derived from Ego4D, EgoSchema consists of over 5000 human curated multiple choice question answer pairs, spanning over 250 hours of real video data, covering a very broad range of natural human activity and behavior. For each question, EgoSchema requires the correct answer to be selected between five given options based on a three-minute-long video clip. While some prior works have proposed video datasets with long clip lengths, we posit that merely the length of the video clip does not truly capture the temporal difficulty of the video task that is being considered. To remedy this, we introduce temporal certificate sets, a general notion for capturing the intrinsic temporal understanding length associated with a broad range of video understanding tasks & datasets. Based on this metric, we find EgoSchema to have intrinsic temporal lengths over 5.7x longer than the second closest dataset and 10x to 100x longer than any other video understanding dataset. Further, our evaluation of several current state-of-the-art video and language models shows them to be severely lacking in long-term video understanding capabilities. Even models with several billions of parameters achieve QA accuracy less than 33% (random is 20%) on the EgoSchema multi-choice question answering task, while humans achieve about 76% accuracy. We posit that EgoSchema, with its long intrinsic temporal structures and diverse complexity, would serve as a valuable evaluation probe for developing effective long-term video understanding systems in the future. Data and Zero-shot model evaluation code will all be open-sourced under the Ego4D license at http://egoschema.github.io.

----

## [2004] TabMT: Generating tabular data with masked transformers

**Authors**: *Manbir Gulati, Paul F. Roysdon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90debc7cedb5cac83145fc8d18378dc5-Abstract-Conference.html)

**Abstract**:

Autoregressive and Masked Transformers are incredibly effective as generative models and classifiers.    While these models are most prevalent in NLP, they also exhibit strong performance in other domains, such as vision.     This work contributes to the exploration of transformer-based models in synthetic data generation for diverse application domains.     In this paper, we present TabMT, a novel Masked Transformer design for generating synthetic tabular data.     TabMT effectively addresses the unique challenges posed by heterogeneous data fields and is natively able to handle missing data.     Our design leverages improved masking techniques to allow for generation and demonstrates state-of-the-art performance from extremely small to extremely large tabular datasets.     We evaluate TabMT for privacy-focused applications and find that it is able to generate high quality data with superior privacy tradeoffs.

----

## [2005] Brain Dissection: fMRI-trained Networks Reveal Spatial Selectivity in the Processing of Natural Images

**Authors**: *Gabriel Sarch, Michael J. Tarr, Katerina Fragkiadaki, Leila Wehbe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90e06fe49254204248cb12562528b952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90e06fe49254204248cb12562528b952-Abstract-Conference.html)

**Abstract**:

The alignment between deep neural network (DNN) features and cortical responses currently provides the most accurate quantitative explanation for higher visual areas. At the same time, these model features have been critiqued as uninterpretable explanations, trading one black box (the human brain) for another (a neural network). In this paper, we train networks to directly predict, from scratch, brain responses to images from a large-scale dataset of natural scenes (Allen et. al., 2021). We then use "network dissection" (Bau et. al., 2017), an explainable AI technique used for enhancing neural network interpretability by identifying and localizing the most significant features in images for individual units of a trained network, and which has been used to study category selectivity in the human brain (Khosla & Wehbe, 2022). We adapt this approach to create a hypothesis-neutral model that is then used to explore the tuning properties of specific visual regions beyond category selectivity, which we call "brain dissection". We use brain dissection to examine a range of ecologically important, intermediate properties, including depth, surface normals, curvature, and object relations across sub-regions of the parietal, lateral, and ventral visual streams, and scene-selective regions. Our findings reveal distinct preferences in brain regions for interpreting visual scenes, with ventro-lateral areas favoring closer and curvier features, medial and parietal areas opting for more varied and flatter 3D elements, and the parietal region uniquely preferring spatial relations. Scene-selective regions exhibit varied preferences, as the retrosplenial complex prefers distant and outdoor features, while the occipital and parahippocampal place areas favor proximity, verticality, and in the case of the OPA, indoor elements. Such findings show the potential of using explainable AI to uncover spatial feature selectivity across the visual cortex, contributing to a deeper, more fine-grained understanding of the functional characteristics of human visual cortex when viewing natural scenes.

----

## [2006] Action Inference by Maximising Evidence: Zero-Shot Imitation from Observation with World Models

**Authors**: *Xingyuan Zhang, Philip Becker-Ehmck, Patrick van der Smagt, Maximilian Karl*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html)

**Abstract**:

Unlike most reinforcement learning agents which require an unrealistic amount of environment interactions to learn a new behaviour, humans excel at learning quickly by merely observing and imitating others. This ability highly depends on the fact that humans have a model of their own embodiment that allows them to infer the most likely actions that led to the observed behaviour. In this paper, we propose Action Inference by Maximising Evidence (AIME) to replicate this behaviour using world models. AIME consists of two distinct phases. In the first phase, the agent learns a world model from its past experience to understand its own body by maximising the ELBO. While in the second phase, the agent is given some observation-only demonstrations of an expert performing a novel task and tries to imitate the expert's behaviour. AIME achieves this by defining a policy as an inference model and maximising the evidence of the demonstration under the policy and world model. Our method is "zero-shot" in the sense that it does not require further training for the world model or online interactions with the environment after given the demonstration. We empirically validate the zero-shot imitation performance of our method on the Walker and Cheetah embodiment of the DeepMind Control Suite and find it outperforms the state-of-the-art baselines. Code is available at: https://github.com/argmax-ai/aime.

----

## [2007] ProtoDiff: Learning to Learn Prototypical Networks by Task-Guided Diffusion

**Authors**: *Yingjun Du, Zehao Xiao, Shengcai Liao, Cees Snoek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/911dd89c81efc624c4e1c39381179505-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/911dd89c81efc624c4e1c39381179505-Abstract-Conference.html)

**Abstract**:

Prototype-based meta-learning has emerged as a powerful technique for addressing few-shot learning challenges. However, estimating a deterministic prototype using a simple average function from a limited number of examples remains a fragile process. To overcome this limitation, we introduce ProtoDiff, a novel framework that leverages a task-guided diffusion model during the meta-training phase to gradually generate prototypes, thereby providing efficient class representations. Specifically,  a set of prototypes is optimized to achieve per-task prototype overfitting, enabling accurately obtaining the overfitted prototypes for individual tasks.Furthermore, we introduce a task-guided diffusion process within the prototype space, enabling the meta-learning of a generative process that transitions from a vanilla prototype to an overfitted prototype. ProtoDiff gradually generates task-specific prototypes from random noise during the meta-test stage, conditioned on the limited samples available for the new task. Furthermore, to expedite training and enhance ProtoDiff's performance, we propose the utilization of residual prototype learning, which leverages the sparsity of the residual prototype. We conduct thorough ablation studies to demonstrate its ability to accurately capture the underlying prototype distribution and enhance generalization. The new state-of-the-art performance on within-domain, cross-domain, and few-task few-shot classiﬁcation further substantiates the beneﬁt of ProtoDiff.

----

## [2008] Synthetic Experience Replay

**Authors**: *Cong Lu, Philip J. Ball, Yee Whye Teh, Jack Parker-Holder*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/911fc798523e7d4c2e9587129fcf88fc-Abstract-Conference.html)

**Abstract**:

A key theme in the past decade has been that when large neural networks and large datasets combine they can produce remarkable results. In deep reinforcement learning (RL), this paradigm is commonly made possible through experience replay, whereby a dataset of past experiences is used to train a policy or value function. However, unlike in supervised or self-supervised learning, an RL agent has to collect its own data, which is often limited. Thus, it is challenging to reap the benefits of deep learning, and even small neural networks can overfit at the start of training. In this work, we leverage the tremendous recent progress in generative modeling and propose Synthetic Experience Replay (SynthER), a diffusion-based approach to flexibly upsample an agent's collected experience. We show that SynthER is an effective method for training RL agents across offline and online settings, in both proprioceptive and pixel-based environments. In offline settings, we observe drastic improvements when upsampling small offline datasets and see that additional synthetic data also allows us to effectively train larger networks. Furthermore, SynthER enables online agents to train with a much higher update-to-data ratio than before, leading to a significant increase in sample efficiency, without any algorithmic changes. We believe that synthetic training data could open the door to realizing the full potential of deep learning for replay-based RL algorithms from limited data. Finally, we open-source our code at https://github.com/conglu1997/SynthER.

----

## [2009] Learning to Tokenize for Generative Retrieval

**Authors**: *Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, Zhaochun Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91228b942a4528cdae031c1b68b127e8-Abstract-Conference.html)

**Abstract**:

As a new paradigm in information retrieval, generative retrieval directly generates a ranked list of document identifiers (docids) for a given query using generative language models (LMs).How to assign each document a unique docid (denoted as document tokenization) is a critical problem, because it determines whether the generative retrieval model can precisely retrieve any document by simply decoding its docid.Most existing methods adopt rule-based tokenization, which is ad-hoc and does not generalize well.In contrast, in this paper we propose a novel document tokenization learning method, GenRet, which learns to encode the complete document semantics into docids.GenRet learns to tokenize documents into short discrete representations (i.e., docids) via a discrete auto-encoding approach.We develop a progressive training scheme to capture the autoregressive nature of docids and diverse clustering techniques to stabilize the training process.Based on the semantic-embedded docids of any set of documents, the generative retrieval model can learn to generate the most relevant docid only according to the docids' semantic relevance to the queries.We conduct experiments on the NQ320K, MS MARCO, and BEIR datasets.GenRet establishes the new state-of-the-art on the NQ320K dataset.Compared to generative retrieval baselines, GenRet can achieve significant improvements on unseen documents.Moreover, GenRet can also outperform comparable baselines on MS MARCO and BEIR, demonstrating the method's generalizability.

----

## [2010] A Reduction-based Framework for Sequential Decision Making with Delayed Feedback

**Authors**: *Yunchang Yang, Han Zhong, Tianhao Wu, Bin Liu, Liwei Wang, Simon S. Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/915125efea950af378435518b3542e6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/915125efea950af378435518b3542e6a-Abstract-Conference.html)

**Abstract**:

We study stochastic delayed feedback in general single-agent and multi-agent sequential decision making, which includes bandits, single-agent Markov decision processes (MDPs), and Markov games (MGs). We propose a novel reduction-based framework, which turns any multi-batched algorithm for sequential decision making with instantaneous feedback into a sample-efficient algorithm that can handle stochastic delays in sequential decision making. By plugging different multi-batched algorithms into our framework, we provide several examples demonstrating that our framework not only matches or improves existing results for bandits, tabular MDPs, and tabular MGs, but also provides the first line of studies on delays in sequential decision making with function approximation. In summary, we provide a complete set of sharp results for single-agent and multi-agent sequential decision making with delayed feedback.

----

## [2011] Efficient RL with Impaired Observability: Learning to Act with Delayed and Missing State Observations

**Authors**: *Minshuo Chen, Yu Bai, H. Vincent Poor, Mengdi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html)

**Abstract**:

In real-world reinforcement learning (RL) systems, various forms of {\it impaired observability} can complicate matters. These situations arise when an agent is unable to observe the most recent state of the system due to latency or lossy channels, yet the agent must still make real-time decisions. This paper introduces a theoretical investigation into efficient RL in control systems where agents must act with delayed and missing state observations. We establish near-optimal regret bounds, of the form $\tilde{\mathcal{O}}(\sqrt{{\rm poly}(H) SAK})$, for RL in both the delayed and missing observation settings. Despite impaired observability posing significant challenges to the policy class and planning, our results demonstrate that learning remains efficient, with the regret bound optimally depending on the state-action size of the original system. Additionally, we provide a characterization of the performance of the optimal policy under impaired observability, comparing it to the optimal value obtained with full observability.

----

## [2012] Unified 3D Segmenter As Prototypical Classifiers

**Authors**: *Zheyun Qin, Cheng Han, Qifan Wang, Xiushan Nie, Yilong Yin, Xiankai Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/916cb4e1aeafaa0757953c9bacd17337-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/916cb4e1aeafaa0757953c9bacd17337-Abstract-Conference.html)

**Abstract**:

The task of point cloud segmentation, comprising semantic, instance, and panoptic segmentation, has been mainly tackled by designing task-specific network architectures, which often lack the flexibility to generalize across tasks, thus resulting in a fragmented research landscape. In this paper, we introduce ProtoSEG, a prototype-based model that unifies semantic, instance, and panoptic segmentation tasks. Our approach treats these three homogeneous tasks as a classification problem with different levels of granularity. By leveraging a Transformer architecture, we extract point embeddings to optimize prototype-class distances and dynamically learn class prototypes to accommodate the end tasks. Our prototypical design enjoys simplicity and transparency, powerful representational learning, and ad-hoc explainability. Empirical results demonstrate that ProtoSEG outperforms concurrent well-known specialized architectures on 3D point cloud benchmarks, achieving 72.3%, 76.4% and 74.2% mIoU for semantic segmentation on S3DIS, ScanNet V2 and SemanticKITTI, 66.8% mCov and 51.2% mAP for instance segmentation on S3DIS and ScanNet V2, 62.4% PQ for panoptic segmentation on SemanticKITTI, validating the strength of our concept and the effectiveness of our algorithm.  The code and models are available at https://github.com/zyqin19/PROTOSEG.

----

## [2013] Cola: A Benchmark for Compositional Text-to-image Retrieval

**Authors**: *Arijit Ray, Filip Radenovic, Abhimanyu Dubey, Bryan A. Plummer, Ranjay Krishna, Kate Saenko*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/917cd410aa55b61594fa2a6f6e5a9e94-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/917cd410aa55b61594fa2a6f6e5a9e94-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Compositional reasoning is a hallmark of human visual intelligence. Yet, despite the size of large vision-language models, they struggle to represent simple compositions by combining objects with their attributes. To measure this lack of compositional capability, we design Cola, a text-to-image retrieval benchmark to Compose Objects Localized with Attributes. To solve Cola, a model must retrieve images with the correct configuration of attributes and objects and avoid choosing a distractor image with the same objects and attributes but in the wrong configuration. Cola contains about 1.2k composed queries of 168 objects and 197 attributes on around 30K images. Our human evaluation finds that Cola is 83.33% accurate, similar to contemporary compositionality benchmarks. Using Cola as a testbed, we explore empirical modeling designs to adapt pre-trained vision-language models to reason compositionally. We explore 6 adaptation strategies on 2 seminal vision-language models, using compositionality-centric test benchmarks - Cola and CREPE. We find the optimal adaptation strategy is to train a multi-modal attention layer that jointly attends over the frozen pre-trained image and language features. Surprisingly, training multimodal layers on CLIP performs better than tuning a larger FLAVA model with already pre-trained multimodal layers. Furthermore, our adaptation strategy improves CLIP and FLAVA to comparable levels, suggesting that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained. Lastly, we show that Cola is harder than a closely related contemporary benchmark, CREPE, since simpler fine-tuning strategies without multimodal layers suffice on CREPE, but not on Cola. However, we still see a significant gap between our best adaptation and human accuracy, suggesting considerable room for further research. Project page: https://cs-people.bu.edu/array/research/cola/

----

## [2014] Estimating Causal Effects Identifiable from a Combination of Observations and Experiments

**Authors**: *Yonghan Jung, Ivan Diaz, Jin Tian, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/917d55788726131e3bb21bf39d477f58-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/917d55788726131e3bb21bf39d477f58-Abstract-Conference.html)

**Abstract**:

Learning cause and effect relations is arguably one of the central challenges found throughout the data sciences.Formally, determining whether a collection of observational and interventional distributions can be combined to learn a target causal relation is known as the problem of generalized identification (or g-identification) [Lee et al., 2019]. Although g-identification has been well understood and solved in theory, it turns out to be challenging to apply these results in practice, in particular when considering the estimation of the target distribution from finite samples. In this paper, we develop a new, general estimator that exhibits multiply robustness properties for g-identifiable causal functionals. Specifically, we show that any g-identifiable causal effect can be expressed as a function of generalized multi-outcome sequential back-door adjustments that are amenable to estimation. We then construct a corresponding estimator for the g-identification expression that exhibits robustness properties to bias. We analyze the asymptotic convergence properties of the estimator. Finally, we illustrate the use of the proposed estimator in experimental studies. Simulation results corroborate the theory.

----

## [2015] LMC: Large Model Collaboration with Cross-assessment for Training-Free Open-Set Object Recognition

**Authors**: *Haoxuan Qu, Xiaofei Hui, Yujun Cai, Jun Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91813e5ddd9658b99be4c532e274b49c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91813e5ddd9658b99be4c532e274b49c-Abstract-Conference.html)

**Abstract**:

Open-set object recognition aims to identify if an object is from a class that has been encountered during training or not. To perform open-set object recognition accurately, a key challenge is how to reduce the reliance on spurious-discriminative features. In this paper, motivated by that different large models pre-trained through different paradigms can possess very rich while distinct implicit knowledge, we propose a novel framework named Large Model Collaboration (LMC) to tackle the above challenge via collaborating different off-the-shelf large models in a training-free manner. Moreover, we also incorporate the proposed framework with several novel designs to effectively extract implicit knowledge from large models. Extensive experiments demonstrate the efficacy of our proposed framework. Code is available \href{https://github.com/Harryqu123/LMC}{here}.

----

## [2016] TaskMet: Task-driven Metric Learning for Model Learning

**Authors**: *Dishank Bansal, Ricky T. Q. Chen, Mustafa Mukadam, Brandon Amos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91a5742235f70ae846436d9780e9f1d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91a5742235f70ae846436d9780e9f1d4-Abstract-Conference.html)

**Abstract**:

Deep learning models are often used with some downstream task. Models solely trained to achieve accurate predictions may struggle to perform well on the desired downstream tasks. We propose using the task loss to learn a metric which parameterizes a loss to train the model. This approach does not alter the optimal prediction model itself, but rather changes the model learning to emphasize the information important for the downstream task. This enables us to achieve the best of both worlds: a prediction model trained in the original prediction space while also being valuable for the desired downstream task. We validate our approach through experiments conducted in two main settings: 1) decision-focused model learning scenarios involving portfolio optimization and budget allocation, and 2) reinforcement learning in noisy environments with distracting states.

----

## [2017] Pairwise Causality Guided Transformers for Event Sequences

**Authors**: *Xiao Shou, Debarun Bhattacharjya, Tian Gao, Dharmashankar Subramanian, Oktie Hassanzadeh, Kristin P. Bennett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91b047c5f5bd41ef56bfaf4ad0bd19e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91b047c5f5bd41ef56bfaf4ad0bd19e3-Abstract-Conference.html)

**Abstract**:

Although pairwise causal relations have been extensively studied in observational longitudinal analyses across many disciplines, incorporating knowledge of causal pairs into deep learning models for temporal event sequences remains largely unexplored. In this paper, we propose a novel approach for enhancing the performance of transformer-based models in multivariate event sequences by injecting pairwise qualitative causal knowledge such as `event Z amplifies future occurrences of event Y'. We establish a new framework for causal inference in temporal event sequences using a transformer architecture, providing a theoretical justification for our approach, and show how to obtain unbiased estimates of the proposed measure. Experimental results demonstrate that our approach outperforms several state-of-the-art models in terms of prediction accuracy by effectively leveraging knowledge about causal pairs. We also consider a unique application where we extract knowledge around sequences of societal events by generating them from a large language model, and demonstrate how a causal knowledge graph can help with event prediction in such sequences. Overall, our framework offers a practical means of improving the performance of transformer-based models in multivariate event sequences by explicitly exploiting pairwise causal information.

----

## [2018] Self-Refine: Iterative Refinement with Self-Feedback

**Authors**: *Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)

**Abstract**:

Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce Self-Refine, an approach for improving initial outputs from LLMs through iterative feedback and refinement. The main idea is to generate an initial output using an LLMs; then, the same LLMs provides *feedback* for its output and uses it to *refine* itself, iteratively. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead uses a single LLM as the generator, refiner and the feedback provider.  We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, improving by $\sim$20\% absolute on average in task performance. Our work demonstrates that even state-of-the-art LLMs like GPT-4 can be further improved at test-time using our simple, standalone approach.

----

## [2019] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**Authors**: *Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Evaluating large language model (LLM) based chat assistants is challenging due to their broad capabilities and the inadequacy of existing benchmarks in measuring human preferences.To address this, we explore using strong LLMs as judges to evaluate these models on more open-ended questions.We examine the usage and limitations of LLM-as-a-judge, including position, verbosity, and self-enhancement biases, as well as limited reasoning ability, and propose solutions to mitigate some of them.We then verify the agreement between LLM judges and human preferences by introducing two benchmarks: MT-bench, a multi-turn question set; and Chatbot Arena, a crowdsourced battle platform.Our results reveal that strong LLM judges like GPT-4 can match both controlled and crowdsourced human preferences well, achieving over 80\% agreement, the same level of agreement between humans.Hence, LLM-as-a-judge is a scalable and explainable way to approximate human preferences, which are otherwise very expensive to obtain.Additionally, we show our benchmark and traditional benchmarks complement each other by evaluating several variants of LLaMA and Vicuna.The MT-bench questions, 3K expert votes, and 30K conversations with human preferences are publicly available at https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge.

----

## [2020] Causal Discovery in Semi-Stationary Time Series

**Authors**: *Shanyun Gao, Raghavendra Addanki, Tong Yu, Ryan A. Rossi, Murat Kocaoglu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/91f9fb16b5679115a777ade51af87e48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/91f9fb16b5679115a777ade51af87e48-Abstract-Conference.html)

**Abstract**:

Discovering causal relations from observational time series without making the stationary assumption is a significant challenge. In practice, this challenge is common in many areas, such as retail sales, transportation systems, and medical science. Here, we consider this problem for a class of non-stationary time series. The structural causal model (SCM) of this type of time series, called the semi-stationary time series, exhibits that a finite number of different causal mechanisms occur sequentially and periodically across time. This model holds considerable practical utility because it can represent periodicity, including common occurrences such as seasonality and diurnal variation. We propose a constraint-based, non-parametric algorithm for discovering causal relations in this setting. The resulting algorithm, PCMCI$_{\Omega}$, can capture the alternating and recurring changes in the causal mechanisms and then identify the underlying causal graph with conditional independence (CI) tests. We show that this algorithm is sound in identifying causal relations on discrete time series. We validate the algorithm with extensive experiments on continuous and discrete simulated data. We also apply our algorithm to a real-world climate dataset.

----

## [2021] Fine-grained Expressivity of Graph Neural Networks

**Authors**: *Jan Böker, Ron Levie, Ningyuan Huang, Soledad Villar, Christopher Morris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9200d97ca2bf3a26db7b591844014f00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9200d97ca2bf3a26db7b591844014f00-Abstract-Conference.html)

**Abstract**:

Numerous recent works have analyzed the expressive power of message-passing graph neural networks (MPNNs), primarily utilizing combinatorial techniques such as the $1$-dimensional Weisfeiler--Leman test ($1$-WL) for the graph isomorphism problem. However, the graph isomorphism objective is inherently binary, not giving insights into the degree of similarity between two given graphs. This work resolves this issue by considering continuous extensions of both $1$-WL and MPNNs to graphons. Concretely, we show that the continuous variant of $1$-WL delivers an accurate topological characterization of the expressive power of MPNNs on graphons, revealing which graphs these networks can distinguish and the level of difficulty in separating them. We identify the finest topology where MPNNs separate points and prove a universal approximation theorem. Consequently, we provide a theoretical framework for graph and graphon similarity combining various topological variants of classical characterizations of the $1$-WL. In particular, we characterize the expressive power of MPNNs in terms of the tree distance, which is a graph distance based on the concept of fractional isomorphisms, and substructure counts via tree homomorphisms, showing that these concepts have the same expressive power as the $1$-WL and MPNNs on graphons. Empirically, we validate our theoretical findings by showing that randomly initialized MPNNs, without training, exhibit competitive performance compared to their trained counterparts. Moreover, we evaluate different MPNN architectures based on their ability to preserve graph distances, highlighting the significance of our continuous $1$-WL test in understanding MPNNs' expressivity.

----

## [2022] CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion

**Authors**: *Yangruibo Ding, Zijian Wang, Wasi Uddin Ahmad, Hantian Ding, Ming Tan, Nihal Jain, Murali Krishna Ramanathan, Ramesh Nallapati, Parminder Bhatia, Dan Roth, Bing Xiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/920f2dced7d32ab2ba2f1970bc306af6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Code completion models have made significant progress in recent years, yet current popular evaluation datasets, such as HumanEval and MBPP, predominantly focus on code completion tasks within a single file. This over-simplified setting falls short of representing the real-world software development scenario where repositories span multiple files with numerous cross-file dependencies, and accessing and understanding cross-file context is often required to complete the code correctly. To fill in this gap, we propose CrossCodeEval, a diverse and multilingual code completion benchmark that necessitates an in-depth cross-file contextual understanding to complete the code accurately. CrossCodeEval is built on a diverse set of real-world, open-sourced, permissively-licensed repositories in four popular programming languages: Python, Java, TypeScript, and C#. To create examples that strictly require cross-file context for accurate completion, we propose a straightforward yet efficient static-analysis-based approach to pinpoint the use of cross-file context within the current file. Extensive experiments on state-of-the-art code language models like CodeGen and StarCoder demonstrate that CrossCodeEval is extremely challenging when the relevant cross-file context is absent, and we see clear improvements when adding these context into the prompt. However, despite such improvements,  the pinnacle of performance remains notably unattained even with the highest-performing model,  indicating that CrossCodeEval is also capable of assessing model's capability in leveraging extensive context to make better code completion. Finally, we benchmarked various methods in retrieving cross-file context, and show that CrossCodeEval can also be used to measure the capability of code retrievers.

----

## [2023] Hierarchical Adaptive Value Estimation for Multi-modal Visual Reinforcement Learning

**Authors**: *Yangru Huang, Peixi Peng, Yifan Zhao, Haoran Xu, Mengyue Geng, Yonghong Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9213010cbcd6ba8e1f1cf1533835d51c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9213010cbcd6ba8e1f1cf1533835d51c-Abstract-Conference.html)

**Abstract**:

Integrating RGB frames with alternative modality inputs is gaining increasing traction in many vision-based reinforcement learning (RL) applications. Existing multi-modal vision-based RL methods usually follow a Global Value Estimation (GVE) pipeline, which uses a fused modality feature to obtain a unified global environmental description. However, such a feature-level fusion paradigm with a single critic may fall short in policy learning as it tends to overlook the distinct values of each modality. To remedy this, this paper proposes a Local modality-customized Value Estimation (LVE) paradigm, which dynamically estimates the contribution and adjusts the importance weight of each modality from a value-level perspective. Furthermore, a task-contextual re-fusion process is developed to achieve a task-level re-balance of estimations from both feature and value levels. To this end, a Hierarchical Adaptive Value Estimation (HAVE) framework is formed, which adaptively coordinates the contributions of individual modalities as well as their collective efficacy. Agents trained by HAVE are able to exploit the unique characteristics of various modalities while capturing their intricate interactions, achieving substantially improved performance. We specifically highlight the potency of our approach within the challenging landscape of autonomous driving, utilizing the CARLA benchmark with neuromorphic event and depth data to demonstrate HAVE's capability and the effectiveness of its distinct components.

----

## [2024] Small Total-Cost Constraints in Contextual Bandits with Knapsacks, with Application to Fairness

**Authors**: *Evgenii Chzhen, Christophe Giraud, Zhen Li, Gilles Stoltz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/921dcb622bd0119c8f4f34644ce87ee0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/921dcb622bd0119c8f4f34644ce87ee0-Abstract-Conference.html)

**Abstract**:

We consider contextual bandit problems with knapsacks [CBwK], a problem where at each round, a scalar reward is obtained and vector-valued costs are suffered. The learner aims to maximize the cumulative rewards while ensuring that the cumulative costs are lower than some predetermined cost constraints. We assume that contexts come from a continuous set, that costs can be signed, and that the expected reward and cost functions, while unknown, may be uniformly estimated---a typical assumption in the literature. In this setting, total cost constraints had so far to be at least of order $T^{3/4}$, where $T$ is the number of rounds, and were even typically assumed to depend linearly on $T$. We are however motivated to use CBwK to impose a fairness constraint of equalized average costs between groups: the budget associated with the corresponding cost constraints should be as close as possible to the natural deviations, of order $\sqrt{T}$. To that end, we introduce a dual strategy based on projected-gradient-descent updates, that is able to deal with total-cost constraints of the order of $\sqrt{T}$ up to poly-logarithmic terms. This strategy is more direct and simpler than existing strategies in the literature. It relies on a careful, adaptive, tuning of the step size.

----

## [2025] CAPP-130: A Corpus of Chinese Application Privacy Policy Summarization and Interpretation

**Authors**: *Pengyun Zhu, Long Wen, Jinfei Liu, Feng Xue, Jian Lou, Zhibo Wang, Kui Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92225ec7e87b97a9e007ca6ab7944b14-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/92225ec7e87b97a9e007ca6ab7944b14-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

A privacy policy serves as an online internet protocol crafted by service providers, which details how service providers collect, process, store, manage, and use personal information when users engage with applications. However, these privacy policies are often filled with technobabble and legalese, making them "incomprehensible''. As a result, users often agree to all terms unknowingly, even some terms may conflict with the law, thereby posing a considerable risk to personal privacy information. One potential solution to alleviate this challenge is to automatically summarize privacy policies using NLP techniques. However, existing techniques primarily focus on extracting key sentences, resulting in comparatively shorter agreements, but failing to address the poor readability caused by the "incomprehensible'' of technobabble and legalese. Moreover, research on Chinese application privacy policy summarization is currently almost nonexistent, and there is a lack of a high-quality corpus suitable for addressing readability issues. To tackle these challenges, we introduce a fine-grained CAPP-130 corpus and a TCSI-pp framework. CAPP-130 contains 130 Chinese privacy policies from popular applications that have been carefully annotated and interpreted by legal experts, resulting in 52,489 annotations and 20,555 rewritten sentences. TCSI-pp first extracts sentences related to the topic specified by users and then uses a generative model to rewrite the sentences into comprehensible summarization. Built upon TSCI-pp, we construct a summarization tool TSCI-pp-zh by selecting RoBERTa from six classification models for sentence extraction and selecting mT5 from five generative models for sentence rewriting. Experimental results show that TCSI-pp-zh outperforms GPT-4 and other baselines in Chinese application privacy policy summarization, demonstrating exceptional readability and reliability. Our data, annotation guidelines, benchmark models, and source code are publicly available at https://github.com/EnlightenedAI/CAPP-130.

----

## [2026] Neural Oscillators are Universal

**Authors**: *Samuel Lanthaler, T. Konstantin Rusch, Siddhartha Mishra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/923285deb805c3e14e1aeebc9854d644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/923285deb805c3e14e1aeebc9854d644-Abstract-Conference.html)

**Abstract**:

Coupled oscillators are being increasingly used as the basis of machine learning (ML) architectures, for instance in sequence modeling, graph representation learning and in physical neural networks that are used in analog ML devices. We introduce an abstract class of neural oscillators that encompasses these architectures and prove that neural oscillators are universal, i.e, they can approximate any continuous and casual operator mapping between time-varying functions, to desired accuracy. This universality result provides theoretical justification for the use of oscillator based ML systems. The proof builds on a fundamental result of independent interest, which shows that a combination of forced harmonic oscillators with a nonlinear read-out suffices to approximate the underlying operators.

----

## [2027] PAC-Bayes Generalization Certificates for Learned Inductive Conformal Prediction

**Authors**: *Apoorva Sharma, Sushant Veer, Asher Hancock, Heng Yang, Marco Pavone, Anirudha Majumdar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9235c376df778f1aaf486a882afb7471-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9235c376df778f1aaf486a882afb7471-Abstract-Conference.html)

**Abstract**:

Inductive Conformal Prediction (ICP) provides a practical and effective approach for equipping deep learning models with uncertainty estimates in the form of set-valued predictions which are guaranteed to contain the ground truth with high probability.Despite the appeal of this coverage guarantee, these sets may not be efficient: the size and contents of the prediction sets are not directly controlled, and instead depend on the underlying model and choice of score function.To remedy this, recent work has proposed learning model and score function parameters using data to directly optimize the efficiency of the ICP prediction sets.While appealing, the generalization theory for such an approach is lacking: direct optimization of empirical efficiency may yield prediction sets that are either no longer efficient on test data, or no longer obtain the required coverage on test data.In this work, we use PAC-Bayes theory to obtain generalization bounds on both the coverage and the efficiency of set-valued predictors which can be directly optimized to maximize efficiency while satisfying a desired test coverage.In contrast to prior work, our framework allows us to utilize the entire calibration dataset to learn the parameters of the model and score function, instead of requiring a separate hold-out set for obtaining test-time coverage guarantees.We leverage these theoretical results to provide a practical algorithm for using calibration data to simultaneously fine-tune the parameters of a model and score function while guaranteeing test-time coverage and efficiency of the resulting prediction sets.We evaluate the approach on regression and classification tasks, and outperform baselines calibrated using a Hoeffding bound-based PAC guarantee on ICP, especially in the low-data regime.

----

## [2028] Image Captioners Are Scalable Vision Learners Too

**Authors**: *Michael Tschannen, Manoj Kumar, Andreas Steiner, Xiaohua Zhai, Neil Houlsby, Lucas Beyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92369a01fbe8046a093746389b2c413e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92369a01fbe8046a093746389b2c413e-Abstract-Conference.html)

**Abstract**:

Contrastive pretraining on image-text pairs from the web is one of the most popular large-scale pretraining strategies for vision backbones, especially in the context of large multimodal models. At the same time, image captioning on this type of data is commonly considered an inferior pretraining strategy. In this paper, we perform a fair comparison of these two pretraining strategies, carefully matching training data, compute, and model capacity. Using a standard encoder-decoder transformer, we find that captioning alone is surprisingly effective: on classification tasks, captioning produces vision encoders competitive with contrastively pretrained encoders, while surpassing them on vision & language tasks. We further analyze the effect of the model architecture and scale, as well as the pretraining data on the representation quality, and find that captioning exhibits the same or better scaling behavior along these axes. Overall our results show that plain image captioning is a more powerful pretraining strategy than was previously believed. Code is available at https://github.com/google-research/big_vision.

----

## [2029] Diplomat: A Dialogue Dataset for Situated PragMATic Reasoning

**Authors**: *Hengli Li, Song-Chun Zhu, Zilong Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/924303c6a45685510877ee018cdc8f80-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/924303c6a45685510877ee018cdc8f80-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The ability to discern and comprehend pragmatic meanings is a cornerstone of social and emotional intelligence, referred to as pragmatic reasoning. Despite the strides made in the development of Large Language Models (LLMs), such as ChatGPT, these models grapple with capturing the nuanced and ambiguous facets of language, falling short of the aspiration to build human-like conversational agents. In this work, we introduce a novel benchmark, the DiPlomat, which delves into the fundamental components of conversational pragmatic reasoning, encompassing situational context reasoning, open-world knowledge acquisition, and unified figurative language understanding. We start by collecting a new human-annotated dialogue dataset, composed of 4,177 multi-turn dialogues and a vocabulary of 48,900 words. Along with the dataset, two tasks are proposed to evaluate machines' pragmatic reasoning capabilities, namely, Pragmatic Reasoning and Identification(PIR) and Conversational Question Answering (CQA). Furthermore, we probe into a zero-shot natural language inference task, where the significance of context in pragmatic reasoning is underscored. Experimental findings illustrate the existing limitations of current prevailing LLMs in the realm of pragmatic reasoning, shedding light on the pressing need for further research to facilitate the emergence of emotional intelligence within human-like conversational agents.

----

## [2030] CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement

**Authors**: *Qihe Huang, Lei Shen, Ruixin Zhang, Shouhong Ding, Binwu Wang, Zhengyang Zhou, Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html)

**Abstract**:

Recently, multivariate time series (MTS) forecasting techniques have seen rapid development and widespread applications across various fields. Transformer-based and GNN-based methods have shown promising potential due to their strong ability to model interaction of time and variables. However, by conducting a comprehensive analysis of the real-world data, we observe that the temporal fluctuations and heterogeneity between variables are not well handled by existing methods. To address the above issues, we propose CrossGNN, a linear complexity GNN model to refine the cross-scale and cross-variable interaction for MTS. To deal with the unexpected noise in time dimension, an adaptive multi-scale identifier (AMSI) is leveraged to construct multi-scale time series with reduced noise. A Cross-Scale GNN is proposed to extract the scales with clearer trend and weaker noise. Cross-Variable GNN is proposed to utilize the homogeneity and heterogeneity between different variables. By simultaneously focusing on edges with higher saliency scores and constraining those edges with lower scores, the time and space complexity (i.e., $O(L)$) of CrossGNN can be linear with the input sequence length $L$. Extensive experimental results on 8 real-world MTS datasets demonstrate the effectiveness of CrossGNN compared with state-of-the-art methods.

----

## [2031] Structured Prediction with Stronger Consistency Guarantees

**Authors**: *Anqi Mao, Mehryar Mohri, Yutao Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/927962d8866377a07ee3150d2d691319-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/927962d8866377a07ee3150d2d691319-Abstract-Conference.html)

**Abstract**:

We present an extensive study of surrogate losses for structured prediction supported by *$H$-consistency bounds*. These are recently introduced guarantees that are more relevant to learning than Bayes-consistency, since they are not asymptotic and since they take into account the hypothesis set $H$ used. We first show that no non-trivial $H$-consistency bound can be derived for widely used surrogate structured prediction losses. We then define several new families of surrogate losses, including *structured comp-sum losses* and *structured constrained losses*, for which we prove $H$-consistency bounds and thus Bayes-consistency. These loss functions readily lead to new structured prediction algorithms with stronger theoretical guarantees, based on their minimization. We describe efficient algorithms for minimizing several of these surrogate losses, including a new *structured logistic loss*.

----

## [2032] Stanford-ORB: A Real-World 3D Object Inverse Rendering Benchmark

**Authors**: *Zhengfei Kuang, Yunzhi Zhang, Hong-Xing Yu, Samir Agarwala, Shangzhe Wu, Jiajun Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92a821f6c25b29241df6985ceb673a85-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/92a821f6c25b29241df6985ceb673a85-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Stanford-ORB, a new real-world 3D Object inverse Rendering Benchmark. Recent advances in inverse rendering have enabled a wide range of real-world applications in 3D content generation, moving rapidly from research and commercial use cases to consumer devices. While the results continue to improve, there is no real-world benchmark that can quantitatively assess and compare the performance of various inverse rendering methods. Existing real-world datasets typically only consist of the shape and multi-view images of objects, which are not sufficient for evaluating the quality of material recovery and object relighting. Methods capable of recovering material and lighting often resort to synthetic data for quantitative evaluation, which on the other hand does not guarantee generalization to complex real-world environments. We introduce a new dataset of real-world objects captured under a variety of natural scenes with ground-truth 3D scans, multi-view images, and environment lighting. Using this dataset, we establish the first comprehensive real-world evaluation benchmark for object inverse rendering tasks from in-the-wild scenes, and compare the performance of various existing methods. All data, code, and models can be accessed at https://stanfordorb.github.io/

----

## [2033] Explainable Brain Age Prediction using coVariance Neural Networks

**Authors**: *Saurabh Sihag, Gonzalo Mateos, Corey McMillan, Alejandro Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92bb2145c74b7d10fbb61aba315b5010-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92bb2145c74b7d10fbb61aba315b5010-Abstract-Conference.html)

**Abstract**:

In computational neuroscience, there has been an increased interest in developing machine learning algorithms that leverage brain imaging data to provide estimates of "brain age" for an individual. Importantly, the discordance between brain age and chronological age (referred to as "brain age gap") can capture accelerated aging due to adverse health conditions and therefore, can reflect increased vulnerability towards neurological disease or cognitive impairments. However, widespread adoption of brain age for clinical decision support has been hindered due to lack of transparency and methodological justifications in most existing brain age prediction algorithms. In this paper, we leverage coVariance neural networks (VNN) to propose an explanation-driven and anatomically interpretable framework for brain age prediction using cortical thickness features. Specifically, our brain age prediction framework extends beyond the coarse metric of brain age gap in Alzheimerâ€™s disease (AD) and we make two important observations: (i) VNNs can assign anatomical interpretability to elevated brain age gap in AD by identifying contributing brain regions, (ii) the interpretability offered by VNNs is contingent on their ability to exploit specific eigenvectors of the anatomical covariance matrix. Together, these observations facilitate an explainable and anatomically interpretable perspective to the task of brain age prediction.

----

## [2034] Adversarial Examples Might be Avoidable: The Role of Data Concentration in Adversarial Robustness

**Authors**: *Ambar Pal, Jeremias Sulam, René Vidal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92d21245424f3898b7110f555a00e829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92d21245424f3898b7110f555a00e829-Abstract-Conference.html)

**Abstract**:

The susceptibility of modern machine learning classifiers to adversarial examples has motivated theoretical results suggesting that these might be unavoidable. However, these results can be too general to be applicable to natural data distributions. Indeed, humans are quite robust for tasks involving vision. This apparent conflict motivates a deeper dive into the question: Are adversarial examples truly unavoidable? In this work, we theoretically demonstrate that a key property of the data distribution -- concentration on small-volume subsets of the input space -- determines whether a robust classifier exists. We further demonstrate that, for a data distribution concentrated on a union of low-dimensional linear subspaces, utilizing structure in data naturally leads to classifiers that enjoy data-dependent polyhedral robustness guarantees, improving upon methods for provable certification in certain regimes.

----

## [2035] Structured State Space Models for In-Context Reinforcement Learning

**Authors**: *Chris Lu, Yannick Schroecker, Albert Gu, Emilio Parisotto, Jakob N. Foerster, Satinder Singh, Feryal M. P. Behbahani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92d3d2a9801211ca3693ccb2faa1316f-Abstract-Conference.html)

**Abstract**:

Structured state space sequence (S4) models have recently achieved state-of-the-art performance on long-range sequence modeling tasks. These models also have fast inference speeds and parallelisable training, making them potentially useful in many reinforcement learning settings. We propose a  modification to a variant of S4 that enables us to initialise and reset the hidden state in parallel, allowing us to tackle reinforcement learning tasks. We show that our modified architecture runs asymptotically faster than Transformers in sequence length and performs better than RNN's on a simple memory-based task. We evaluate our modified architecture on a set of partially-observable environments and find that, in practice, our model outperforms RNN's while also running over five times faster. Then, by leveraging the modelâ€™s ability to handle long-range sequences, we achieve strong performance on a challenging meta-learning task in which the agent is given a randomly-sampled continuous control environment, combined with a randomly-sampled linear projection of the environment's observations and actions. Furthermore, we show the resulting model can adapt to out-of-distribution held-out tasks. Overall, the results presented in this paper show that structured state space models are fast and performant for in-context reinforcement learning tasks. We provide code at https://github.com/luchris429/s5rl.

----

## [2036] Sharpness-Aware Minimization Leads to Low-Rank Features

**Authors**: *Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/92dd1adab39f362046f99dfe3c39d90f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/92dd1adab39f362046f99dfe3c39d90f-Abstract-Conference.html)

**Abstract**:

Sharpness-aware minimization (SAM) is a recently proposed method that minimizes the sharpness of the training loss of a neural network. While its generalization improvement is well-known and is the primary motivation, we uncover an additional intriguing effect of SAM: reduction of the feature rank which happens at different layers of a neural network. We show that this low-rank effect occurs very broadly: for different architectures such as fully-connected networks, convolutional networks, vision transformers and for different objectives such as regression, classification, language-image contrastive training. To better understand this phenomenon, we provide a mechanistic understanding of how low-rank features arise in a simple two-layer network. We observe that a significant number of activations gets entirely pruned by SAM which directly contributes to the rank reduction. We confirm this effect theoretically and check that it can also occur in deep networks, although the overall rank reduction mechanism can be more complex, especially for deep networks with pre-activation skip connections and self-attention layers.

----

## [2037] A Spectral Theory of Neural Prediction and Alignment

**Authors**: *Abdulkadir Canatar, Jenelle Feather, Albert J. Wakhloo, SueYeon Chung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9308d1b7d4ae2d3e2e67ae94b1078bf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9308d1b7d4ae2d3e2e67ae94b1078bf7-Abstract-Conference.html)

**Abstract**:

The representations of neural networks are often compared to those of biological systems by performing regression between the neural network responses and those measured from biological systems. Many different state-of-the-art deep neural networks yield similar neural predictions, but it remains unclear how to differentiate among models that perform equally well at predicting neural responses. To gain insight into this, we use a recent theoretical framework that relates the generalization error from regression to the spectral properties of the model and the target. We apply this theory to the case of regression between model activations and neural responses and decompose the neural prediction error in terms of the model eigenspectra, alignment of model eigenvectors and neural responses, and the training set size. Using this decomposition, we introduce geometrical measures to interpret the neural prediction error. We test a large number of deep neural networks that predict visual cortical activity and show that there are multiple types of geometries that result in low neural prediction error as measured via regression. The work demonstrates that carefully decomposing representational metrics can provide interpretability of how models are capturing neural activity and points the way towards improved models of neural activity.

----

## [2038] Train Once, Get a Family: State-Adaptive Balances for Offline-to-Online Reinforcement Learning

**Authors**: *Shenzhi Wang, Qisen Yang, Jiawei Gao, Matthieu Gaetan Lin, Hao Chen, Liwei Wu, Ning Jia, Shiji Song, Gao Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)

**Abstract**:

Offline-to-online reinforcement learning (RL) is a training paradigm that combines pre-training on a pre-collected dataset with fine-tuning in an online environment. However, the incorporation of online fine-tuning can intensify the well-known distributional shift problem. Existing solutions tackle this problem by imposing a policy constraint on the policy improvement objective in both offline and online learning. They typically advocate a single balance between policy improvement and constraints across diverse data collections. This one-size-fits-all manner may not optimally leverage each collected sample due to the significant variation in data quality across different states. To this end, we introduce Family Offline-to-Online RL (FamO2O), a simple yet effective framework that empowers existing algorithms to determine state-adaptive improvement-constraint balances. FamO2O utilizes a universal model to train a family of policies with different improvement/constraint intensities, and a balance model to select a suitable policy for each state. Theoretically, we prove that state-adaptive balances are necessary for achieving a higher policy performance upper bound. Empirically, extensive experiments show that FamO2O offers a statistically significant improvement over various existing methods, achieving state-of-the-art performance on the D4RL benchmark. Codes are available at https://github.com/LeapLabTHU/FamO2O.

----

## [2039] Test-Time Distribution Normalization for Contrastively Learned Visual-language Models

**Authors**: *Yifei Zhou, Juntao Ren, Fengyu Li, Ramin Zabih, Ser Nam Lim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/931db0b5a61f9db6c97c7e4bf068147d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/931db0b5a61f9db6c97c7e4bf068147d-Abstract-Conference.html)

**Abstract**:

Advances in the field of visual-language contrastive learning have made it possible for many downstream applications to be carried out efficiently and accurately by simply taking the dot product between image and text representations.  One of the most representative approaches proposed recently known as CLIP has quickly garnered widespread adoption due to its effectiveness. CLIP is trained with an InfoNCE loss that takes into account both positive and negative samples to help learn a much more robust representation space. This paper however reveals that the common downstream practice of taking a dot product is only a zeroth-order approximation of the optimization goal, resulting in a loss of information during test-time. Intuitively, since the model has been optimized based on the InfoNCE loss, test-time procedures should ideally also be in alignment. The question lies in how one can retrieve any semblance of negative samples information during inference in a computationally efficient way. We propose Distribution Normalization (DN), where we approximate the mean representation of a batch of test samples and use such a mean to represent what would be analogous to negative samples in the InfoNCE loss. DN requires no retraining or fine-tuning and can be effortlessly applied during inference. Extensive experiments on a wide variety of downstream tasks exhibit a clear advantage of DN over the dot product on top of other existing test-time augmentation methods.

----

## [2040] Propagating Knowledge Updates to LMs Through Distillation

**Authors**: *Shankar Padmanabhan, Yasumasa Onoe, Michael J. Q. Zhang, Greg Durrett, Eunsol Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/932147114c48f8b04d41aebc0c631158-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/932147114c48f8b04d41aebc0c631158-Abstract-Conference.html)

**Abstract**:

Modern language models have the capacity to store and use immense amounts of knowledge about real-world entities, but it remains unclear how to update such knowledge stored in model parameters. While prior methods for updating knowledge in LMs successfully inject atomic facts, updated LMs fail to make inferences based on injected facts. In this work, we demonstrate that a context distillation-based approach can both impart knowledge about entities \emph{and} propagate that knowledge to enable broader inferences. Our approach consists of two stages: transfer set generation and distillation on the transfer set. We first generate a transfer set by prompting a language model to generate continuations from the entity definition. Then, we update the model parameters so that the distribution of the LM (the 'student') matches the distribution of the LM conditioned on the definition (the 'teacher') on the transfer set. Our experiments demonstrate that this approach is more effective at propagating knowledge updates than fine-tuning and other gradient-based knowledge-editing methods. Moreover, it does not  compromise performance in other contexts, even when injecting the definitions of up to 150 entities at once.

----

## [2041] ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling

**Authors**: *Yuqi Chen, Kan Ren, Yansen Wang, Yuchen Fang, Weiwei Sun, Dongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9328208f88ec69420031647e6ff97727-Abstract-Conference.html)

**Abstract**:

Modeling continuous-time dynamics on irregular time series is critical to account for data evolution and correlations that occur continuously. Traditional methods including recurrent neural networks or Transformer models leverage inductive bias via powerful neural architectures to capture complex patterns. However, due to their discrete characteristic, they have limitations in generalizing to continuous-time data paradigms. Though neural ordinary differential equations (Neural ODEs) and their variants have shown promising results in dealing with irregular time series, they often fail to capture the intricate correlations within these sequences. It is challenging yet demanding to concurrently model the relationship between input data points and capture the dynamic changes of the continuous-time system. To tackle this problem, we propose ContiFormer that extends the relation modeling of vanilla Transformer to the continuous-time domain, which explicitly incorporates the modeling abilities of continuous dynamics of Neural ODEs with the attention mechanism of Transformers. We mathematically characterize the expressive power of ContiFormer and illustrate that, by curated designs of function hypothesis, many Transformer variants specialized in irregular time series modeling can be covered as a special case of ContiFormer. A wide range of experiments on both synthetic and real-world datasets have illustrated the superior modeling capacities and prediction performance of ContiFormer on irregular time series data. The project link is https://seqml.github.io/contiformer/.

----

## [2042] Differentiable Random Partition Models

**Authors**: *Thomas M. Sutter, Alain Ryser, Joram Liebeskind, Julia E. Vogt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/933b5d002cf251b3e854d586e55ac58c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/933b5d002cf251b3e854d586e55ac58c-Abstract-Conference.html)

**Abstract**:

Partitioning a set of elements into an unknown number of mutually exclusive subsets is essential in many machine learning problems.However, assigning elements, such as samples in a dataset or neurons in a network layer, to an unknown and discrete number of subsets is inherently non-differentiable, prohibiting end-to-end gradient-based optimization of parameters.We overcome this limitation by proposing a novel two-step method for inferring partitions, which allows its usage in variational inference tasks.This new approach enables reparameterized gradients with respect to the parameters of the new random partition model.Our method works by inferring the number of elements per subset and, second, by filling these subsets in a learned order.We highlight the versatility of our general-purpose approach on three different challenging experiments: variational clustering, inference of shared and independent generative factors under weak supervision, and multitask learning.

----

## [2043] Connecting Pre-trained Language Model and Downstream Task via Properties of Representation

**Authors**: *Chenwei Wu, Holden Lee, Rong Ge*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93712c59f6a81bd92040facf04c8b308-Abstract-Conference.html)

**Abstract**:

Recently, researchers have found that representations learned by large-scale pre-trained language models are useful in various downstream tasks. However, there is little theoretical understanding of how pre-training performance is related to downstream task performance. In this paper, we analyze how this performance transfer depends on the properties of the downstream task and the structure of the representations. We consider a log-linear model where a word can be predicted from its context through a network having softmax as its last layer. We show that even if the downstream task is highly structured and depends on a simple function of the hidden representation, there are still cases when a low pre-training loss cannot guarantee good performance on the downstream task. On the other hand, we propose and empirically validate the existence of an ``anchor vector'' in the representation space, and show that this assumption, together with properties of the downstream task,  guarantees performance transfer.

----

## [2044] Generalizable One-shot 3D Neural Head Avatar

**Authors**: *Xueting Li, Shalini De Mello, Sifei Liu, Koki Nagano, Umar Iqbal, Jan Kautz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/937ae0e83eb08d2cb8627fe1def8c751-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/937ae0e83eb08d2cb8627fe1def8c751-Abstract-Conference.html)

**Abstract**:

We present a method that reconstructs and animates a 3D head avatar from a single-view portrait image. Existing methods either involve time-consuming optimization for a specific person with multiple images, or they struggle to synthesize intricate appearance details beyond the facial region. To address these limitations, we propose a framework that not only generalizes to unseen identities based on a single-view image without requiring person-specific optimization, but also captures characteristic details within and beyond the face area (e.g. hairstyle, accessories, etc.). At the core of our method are three branches that produce three tri-planes representing the coarse 3D geometry, detailed appearance of a source image, as well as the expression of a target image. By applying volumetric rendering to the combination of the three tri-planes followed by a super-resolution module, our method yields a high fidelity image of the desired identity, expression and pose. Once trained, our model enables efficient 3D head avatar reconstruction and animation via a single forward pass through a network. Experiments show that the proposed approach generalizes well to unseen validation datasets, surpassing SOTA baseline methods by a large margin on head avatar reconstruction and animation.

----

## [2045] Equivariant Single View Pose Prediction Via Induced and Restriction Representations

**Authors**: *Owen Howell, David Klee, Ondrej Biza, Linfeng Zhao, Robin Walters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93b3d975f9a2448964a906199db98a9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93b3d975f9a2448964a906199db98a9d-Abstract-Conference.html)

**Abstract**:

Learning about the three-dimensional world from two-dimensional images is a fundamental problem in computer vision. An ideal neural network architecture for such tasks would leverage the fact that objects can be rotated and translated in three dimensions to make predictions about novel images. However, imposing $SO(3)$-equivariance on two-dimensional inputs is difficult because the group of three-dimensional rotations does not have a natural action on the two-dimensional plane. Specifically, it is possible that an element of $SO(3)$ will rotate an image out of plane. We show that an algorithm that learns a three-dimensional representation of the world from two dimensional images must satisfy certain consistency properties which we formulate as $SO(2)$-equivariance constraints. We use the induced representation of $SO(2)$ on $SO(3)$ to construct and classify architectures that have two-dimensional inputs and which satisfy these consistency constraints. We prove that any architecture which respects said consistency constraints can be realized as an instance of our construction. We show that three previously proposed neural architectures for 3D pose prediction are special cases of our construction. We propose a new algorithm that is a learnable generalization of previously considered methods. We test our architecture on three pose predictions task and achieve SOTA results on both the PASCAL3D+ and SYMSOL pose estimation tasks.

----

## [2046] Unsupervised Learning for Solving the Travelling Salesman Problem

**Authors**: *Yimeng Min, Yiwei Bai, Carla P. Gomes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93b8618a9061f8a55825c13ecf28392b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93b8618a9061f8a55825c13ecf28392b-Abstract-Conference.html)

**Abstract**:

We propose UTSP, an Unsupervised Learning (UL) framework for solving the Travelling Salesman Problem (TSP). We train a Graph Neural Network (GNN) using a surrogate loss. The GNN outputs a heat map representing the probability for each edge to be part of the optimal path. We then apply local search to generate our final prediction based on the heat map. Our loss function consists of two parts: one pushes the model to find the shortest path and the other serves as a surrogate for the constraint that the route should form a Hamiltonian Cycle. Experimental results show that UTSP outperforms the existing data-driven TSP heuristics.Our approach is parameter efficient as well as data efficient: the model takes  $\sim$ 10\% of the number of parameters and $\sim$ 0.2\% of training samples compared with Reinforcement Learning or Supervised Learning methods.

----

## [2047] ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion

**Authors**: *Wei Xing, Yuxin Wang, Zheng Xing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93cf20db85fabb0fd4bb89346510629c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93cf20db85fabb0fd4bb89346510629c-Abstract-Conference.html)

**Abstract**:

Multi-fidelity fusion has become an important surrogate technique, which provides insights into expensive computer simulations and effectively improves decision-making, e.g., optimization, with less computational cost. Multi-fidelity fusion is much more computationally efficient compared to traditional single-fidelity surrogates. Despite the fast advancement of multi-fidelity fusion techniques, they lack a systematic framework to make use of the fidelity indicator, deal with high-dimensional and arbitrary data structure, and scale well to infinite-fidelity problems. In this work, we first generalize the popular autoregression (AR) to derive a novel linear fidelity differential equation (FiDE), paving the way to tractable infinite-fidelity fusion. We generalize FiDE to a high-dimensional system, which also provides a unifying framework to seemly bridge the gap between many multi- and single-fidelity GP-based models. We then propose ContinuAR, a rank-1 approximation solution to FiDEs, which is tractable to train, compatible with arbitrary multi-fidelity data structure, linearly scalable to the output dimension, and most importantly, delivers consistent SOTA performance with a significant margin over the baseline methods. Compared to the SOTA infinite-fidelity fusion, IFC, ContinuAR achieves up to 4x improvement in accuracy and 62,500x speedup in training time.

----

## [2048] FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space

**Authors**: *Shengzhong Liu, Tomoyoshi Kimura, Dongxin Liu, Ruijie Wang, Jinyang Li, Suhas N. Diggavi, Mani B. Srivastava, Tarek F. Abdelzaher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93e98ddf39a9beb0a97fbbe56a986c80-Abstract-Conference.html)

**Abstract**:

This paper proposes a novel contrastive learning framework, called FOCAL, for extracting comprehensive features from multimodal time-series sensing signals through self-supervised training. Existing multimodal contrastive frameworks mostly rely on the shared information between sensory modalities, but do not explicitly consider the exclusive modality information that could be critical to understanding the underlying sensing physics. Besides, contrastive frameworks for time series have not handled the temporal information locality appropriately. FOCAL solves these challenges by making the following contributions: First, given multimodal time series, it encodes each modality into a factorized latent space consisting of shared features and private features that are orthogonal to each other. The shared space emphasizes feature patterns consistent across sensory modalities through a modal-matching objective. In contrast, the private space extracts modality-exclusive information through a transformation-invariant objective. Second, we propose a temporal structural constraint for modality features, such that the average distance between temporally neighboring samples is no larger than that of temporally distant samples. Extensive evaluations are performed on four multimodal sensing datasets with two backbone encoders and two classifiers to demonstrate the superiority of FOCAL. It consistently outperforms the state-of-the-art baselines in downstream tasks with a clear margin, under different ratios of available labels. The code and self-collected dataset are available at https://github.com/tomoyoshki/focal.

----

## [2049] Assumption violations in causal discovery and the robustness of score matching

**Authors**: *Francesco Montagna, Atalanti-Anastasia Mastakouri, Elias Eulig, Nicoletta Noceti, Lorenzo Rosasco, Dominik Janzing, Bryon Aragam, Francesco Locatello*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93ed74938a54a73b5e4c52bbaf42ca8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93ed74938a54a73b5e4c52bbaf42ca8e-Abstract-Conference.html)

**Abstract**:

When domain knowledge is limited and experimentation is restricted by ethical, financial, or time constraints, practitioners turn to observational causal discovery methods to recover the causal structure, exploiting the statistical properties of their data. Because causal discovery without further assumptions is an ill-posed problem, each algorithm comes with its own set of usually untestable assumptions, some of which are hard to meet in real datasets. Motivated by these considerations, this paper extensively benchmarks the empirical performance of recent causal discovery methods on observational iid data generated under different background conditions, allowing for violations of the critical assumptions required by each selected approach. Our experimental findings show that score matching-based methods demonstrate surprising performance in the false positive and false negative rate of the inferred graph in these challenging scenarios, and we provide theoretical insights into their performance. This work is also the first effort to benchmark the stability of causal discovery algorithms with respect to the values of their hyperparameters. Finally, we hope this paper will set a new standard for the evaluation of causal discovery methods and can serve as an accessible entry point for practitioners interested in the field, highlighting the empirical implications of different algorithm choices.

----

## [2050] Normalizing flow neural networks by JKO scheme

**Authors**: *Chen Xu, Xiuyuan Cheng, Yao Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/93fce71def4e3cf418918805455d436f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/93fce71def4e3cf418918805455d436f-Abstract-Conference.html)

**Abstract**:

Normalizing flow is a class of deep generative models for efficient sampling and likelihood estimation, which achieves attractive performance, particularly in high dimensions. The flow is often implemented using a sequence of invertible residual blocks. Existing works adopt special network architectures and regularization of flow trajectories. In this paper, we develop a neural ODE flow network called JKO-iFlow, inspired by the Jordan-Kinderleherer-Otto (JKO) scheme, which unfolds the discrete-time dynamic of the Wasserstein gradient flow. The proposed method stacks residual blocks one after another, allowing efficient block-wise training of the residual blocks, avoiding sampling SDE trajectories and score matching or variational learning, thus reducing the memory load and difficulty in end-to-end training. We also develop adaptive time reparameterization of the flow network with a progressive refinement of the induced trajectory in probability space to improve the model accuracy further. Experiments with synthetic and real data show that the proposed JKO-iFlow network achieves competitive performance compared with existing flow and diffusion models at a significantly reduced computational and memory cost.

----

## [2051] Stability-penalty-adaptive follow-the-regularized-leader: Sparsity, game-dependency, and best-of-both-worlds

**Authors**: *Taira Tsuchiya, Shinji Ito, Junya Honda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9408564a4229f4a933ac9bd09a29ee96-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9408564a4229f4a933ac9bd09a29ee96-Abstract-Conference.html)

**Abstract**:

Adaptivity to the difficulties of a problem is a key property in sequential decision-making problems to broaden the applicability of algorithms. Follow-the-regularized-leader (FTRL) has recently emerged as one of the most promising approaches for obtaining various types of adaptivity in bandit problems. Aiming to further generalize this adaptivity, we develop a generic adaptive learning rate, called stability-penalty-adaptive (SPA) learning rate for FTRL. This learning rate yields a regret bound jointly depending on stability and penalty of the algorithm, into which the regret of FTRL is typically decomposed. With this result, we establish several algorithms with three types of adaptivity: sparsity, game-dependency, and best-of-both-worlds (BOBW). Despite the fact that sparsity appears frequently in real problems, existing sparse multi-armed bandit algorithms with $k$-arms assume that the sparsity level $s \leq k$ is known in advance, which is often not the case in real-world scenarios. To address this issue, we first establish $s$-agnostic algorithms with regret bounds of $\tilde{O}(\sqrt{sT})$ in the adversarial regime for $T$ rounds, which matches the existing lower bound up to a logarithmic factor. Meanwhile, BOBW algorithms aim to achieve a near-optimal regret in both the stochastic and adversarial regimes. Leveraging the SPA learning rate and the technique for $s$-agnostic algorithms combined with a new analysis to bound the variation in FTRL output in response to changes in a regularizer, we establish the first BOBW algorithm with a sparsity-dependent bound. Additionally, we explore partial monitoring and demonstrate that the proposed SPA learning rate framework allows us to achieve a game-dependent bound and the BOBW simultaneously.

----

## [2052] Domain Agnostic Fourier Neural Operators

**Authors**: *Ning Liu, Siavash Jafarzadeh, Yue Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/940a7634dab556b67af15bacd337f7db-Abstract-Conference.html)

**Abstract**:

Fourier neural operators (FNOs) can learn highly nonlinear mappings between function spaces, and have recently become a popular tool for learning responses of complex physical systems. However, to achieve good accuracy and efficiency, FNOs rely on the Fast Fourier transform (FFT), which is restricted to modeling problems on rectangular domains. To lift such a restriction and permit FFT on irregular geometries as well as topology changes, we introduce domain agnostic Fourier neural operator (DAFNO), a novel neural operator architecture for learning surrogates with irregular geometries and evolving domains. The key idea is to incorporate a smoothed characteristic function in the integral layer architecture of FNOs, and leverage FFT to achieve rapid computations, in such a way that the geometric information is explicitly encoded in the architecture. In our empirical evaluation, DAFNO has achieved state-of-the-art accuracy as compared to baseline neural operator models on two benchmark datasets of material modeling and airfoil simulation. To further demonstrate the capability and generalizability of DAFNO in handling complex domains with topology changes, we consider a brittle material fracture evolution problem. With only one training crack simulation sample, DAFNO has achieved generalizability to unseen loading scenarios and substantially different crack patterns from the trained scenario. Our code and data accompanying this paper are available at https://github.com/ningliu-iga/DAFNO.

----

## [2053] A2CiD2: Accelerating Asynchronous Communication in Decentralized Deep Learning

**Authors**: *Adel Nabli, Eugene Belilovsky, Edouard Oyallon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/940f1d0760ca52c8b21ef3b661357ec2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/940f1d0760ca52c8b21ef3b661357ec2-Abstract-Conference.html)

**Abstract**:

Distributed training of Deep Learning models has been critical to many recent successes in the field. Current standard methods primarily rely on synchronous centralized algorithms which induce major communication bottlenecks and synchronization locks at scale. Decentralized asynchronous algorithms are emerging as a potential alternative but their practical applicability still lags. In order to mitigate the increase in communication cost that naturally comes with scaling the number of workers, we introduce a principled asynchronous, randomized, gossip-based optimization algorithm which works thanks to a continuous local momentum named $\textbf{A}^2\textbf{CiD}^2$. Our method allows each worker to continuously process mini-batches without stopping, and run a peer-to-peer averaging routine in parallel, reducing idle time. In addition to inducing a significant communication acceleration at no cost other than adding a local momentum variable, minimal adaptation is required to incorporate $\textbf{A}^2\textbf{CiD}^2$ to standard asynchronous approaches. Our theoretical analysis proves accelerated rates compared to previous asynchronous decentralized baselines and we empirically show that using our $\textbf{A}^2\textbf{CiD}^2$ momentum significantly decrease communication costs in poorly connected networks. In particular, we show consistent improvement on the ImageNet dataset using up to 64 asynchronous workers (A100 GPUs) and various communication network topologies.

----

## [2054] MathNAS: If Blocks Have a Role in Mathematical Architecture Design

**Authors**: *Qinsi Wang, Jinghan Ke, Zhi Liang, Sihai Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9410d94d47adfb07b41a0b226270f068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9410d94d47adfb07b41a0b226270f068-Abstract-Conference.html)

**Abstract**:

Neural Architecture Search (NAS) has emerged as a favoured method for unearthing effective neural architectures. Recent development of large models has intensified the demand for faster search speeds and more accurate search results. However, designing large models by NAS is challenging due to the dramatical increase of search space and the associated huge performance evaluation cost. Consider a typical modular search space widely used in NAS, in which a neural architecture consists of $m$ block nodes and a block node has $n$ alternative blocks. Facing the space containing $n^m$ candidate networks, existing NAS methods attempt to find the best one by searching and evaluating candidate networks directly.Different from the general strategy that takes architecture search as a whole problem, we propose a novel divide-and-conquer strategy by making use of the modular nature of the search space.Here, we introduce MathNAS, a general NAS framework based on mathematical programming.  In MathNAS, the performances of all possible building blocks in the search space are calculated first, and then the performance of a network is directly predicted based on the performances of its building blocks.Although estimating block performances involves network training, just as what happens for network performance evaluation in existing NAS methods, predicting network performance is completely training-free and thus extremely fast. In contrast to the $n^m$ candidate networks to evaluate in existing NAS methods, which requires training and a formidable computational burden, there are only $m*n$ possible blocks to handle in MathNAS.Therefore, our approach effectively reduces the complexity of network performance evaluation. The superiority of MathNAS is validated on multiple large-scale CV and NLP benchmark datasets. Notably on ImageNet-1k, MathNAS achieves 82.5\% top-1 accuracy, 1.2\% and 0.96\% higher than Swin-T and LeViT-256, respectively. In addition, when deployed on mobile device, MathNAS achieves real-time search and dynamic network switching within 1s (0.4s on TX2 GPU), surpassing baseline dynamic networks in on-device performance.

----

## [2055] Block Broyden's Methods for Solving Nonlinear Equations

**Authors**: *Chengchang Liu, Cheng Chen, Luo Luo, John C. S. Lui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9417a5154519e370fd64e5a65e7dc59b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9417a5154519e370fd64e5a65e7dc59b-Abstract-Conference.html)

**Abstract**:

This paper studies quasi-Newton methods for solving nonlinear equations. We propose block variants of both good and bad Broyden's methods, which enjoy explicit local superlinear convergence rates. Our block good Broyden's method has faster condition-number-free convergence rate than existing Broyden's methods because it takes the advantage of multiple rank modification on the Jacobian estimator. On the other hand, our block bad Broyden's method directly estimates the inverse of the Jacobian provably, which reduces the computational cost of the iteration. Our theoretical results provide some new insights on why good Broyden's method outperforms bad Broyden's method in most of the cases. The empirical results also demonstrate the superiority of our methods and validate our theoretical analysis.

----

## [2056] Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence

**Authors**: *Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, Trevor Darrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/942032b61720a3fd64897efe46237c81-Abstract-Conference.html)

**Abstract**:

Diffusion models have been shown to be capable of generating high-quality images, suggesting that they could contain meaningful internal representations. Unfortunately, the feature maps that encode a diffusion model's internal information are spread not only over layers of the network, but also over diffusion timesteps, making it challenging to extract useful descriptors. We propose Diffusion Hyperfeatures, a framework for consolidating  multi-scale and multi-timestep feature maps into per-pixel feature descriptors that can be used for downstream tasks. These descriptors can be extracted for both synthetic and real images using the generation and inversion processes. We evaluate the utility of our Diffusion Hyperfeatures on the task of semantic keypoint correspondence: our method achieves superior performance on the SPair-71k real image benchmark. We also demonstrate that our method is flexible and transferable: our feature aggregation network trained on the inversion features of real image pairs can be used on the generation features of synthetic image pairs with unseen objects and compositions. Our code is available at https://diffusion-hyperfeatures.github.io.

----

## [2057] No Change, No Gain: Empowering Graph Neural Networks with Expected Model Change Maximization for Active Learning

**Authors**: *Zixing Song, Yifei Zhang, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) are crucial for machine learning applications with graph-structured data, but their success depends on sufficient labeled data. We present a novel active learning (AL) method for GNNs, extending the Expected Model Change Maximization (EMCM) principle to improve prediction performance on unlabeled data. By presenting a Bayesian interpretation for the node embeddings generated by GNNs under the semi-supervised setting, we efficiently compute the closed-form EMCM acquisition function as the selection criterion for AL without re-training. Our method establishes a direct connection with expected prediction error minimization, offering theoretical guarantees for AL performance. Experiments demonstrate our method's effectiveness compared to existing approaches, in terms of both accuracy and efficiency.

----

## [2058] Scaling Laws for Hyperparameter Optimization

**Authors**: *Arlind Kadra, Maciej Janowski, Martin Wistuba, Josif Grabocka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/945c781d7194ea81026148838af95af7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/945c781d7194ea81026148838af95af7-Abstract-Conference.html)

**Abstract**:

Hyperparameter optimization is an important subfield of machine learning that focuses on tuning the hyperparameters of a chosen algorithm to achieve peak performance. Recently, there has been a stream of methods that tackle the issue of hyperparameter optimization, however, most of the methods do not exploit the dominant power law nature of learning curves for Bayesian optimization. In this work, we propose Deep Power Laws (DPL), an ensemble of neural network models conditioned to yield predictions that follow a power-law scaling pattern. Our method dynamically decides which configurations to pause and train incrementally by making use of gray-box evaluations. We compare our method against 7 state-of-the-art competitors on 3 benchmarks related to tabular, image, and NLP datasets covering 59 diverse tasks. Our method achieves the best results across all benchmarks by obtaining the best any-time results compared to all competitors.

----

## [2059] A Robust and Opponent-Aware League Training Method for StarCraft II

**Authors**: *Ruozi Huang, Xipeng Wu, Hongsheng Yu, Zhong Fan, Haobo Fu, Qiang Fu, Wei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94796017d01c5a171bdac520c199d9ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94796017d01c5a171bdac520c199d9ed-Abstract-Conference.html)

**Abstract**:

It is extremely difficult to train a superhuman Artificial Intelligence (AI) for games of similar size to StarCraft II. AlphaStar is the first AI that beat human professionals in the full game of StarCraft II, using a league training framework that is inspired by a game-theoretic approach. In this paper, we improve AlphaStar's league training in two significant aspects.  We train goal-conditioned exploiters, whose abilities of spotting weaknesses in the main agent and the entire league are greatly improved compared to the unconditioned exploiters in AlphaStar. In addition, we endow the agents in the league with the new ability of opponent modeling, which makes the agent more responsive to the opponent's real-time strategy. Based on these improvements, we train a better and superhuman AI with orders of magnitude less resources than AlphaStar (see Table 1 for a full comparison). Considering the iconic role of StarCraft II in game AI research, we believe our method and results on StarCraft II provide valuable design principles on how one would utilize the general league training framework for obtaining a least-exploitable strategy in various, large-scale, real-world games.

----

## [2060] Causal Fairness for Outcome Control

**Authors**: *Drago Plecko, Elias Bareinboim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html)

**Abstract**:

As society transitions towards an AI-based decision-making infrastructure, an ever-increasing number of decisions once under control of humans are now delegated to automated systems. Even though such developments make various parts of society more efficient, a large body of evidence suggests that a great deal of care needs to be taken to make such automated decision-making systems fair and equitable, namely, taking into account sensitive attributes such as gender, race, and religion. In this paper, we study a specific decision-making task called outcome control in which an automated system aims to optimize an outcome variable $Y$ while being fair and equitable. The interest in such a setting ranges from interventions related to criminal justice and welfare, all the way to clinical decision-making and public health. In this paper, we first analyze through causal lenses the notion of benefit, which captures how much a specific individual would benefit from a positive decision, counterfactually speaking, when contrasted with an alternative, negative one. We introduce the notion of benefit fairness, which can be seen as the minimal fairness requirement in decision-making, and develop an algorithm for satisfying it. We then note that the benefit itself may be influenced by the protected attribute, and propose causal tools which can be used to analyze this. Finally, if some of the variations of the protected attribute in the benefit are considered as discriminatory, the notion of benefit fairness may need to be strengthened, which leads us to articulating a notion of causal benefit fairness. Using this notion, we develop a new optimization procedure capable of maximizing $Y$ while ascertaining causal fairness in the decision process.

----

## [2061] DeepPCR: Parallelizing Sequential Operations in Neural Networks

**Authors**: *Federico Danieli, Miguel Sarabia, Xavier Suau Cuadros, Pau Rodríguez, Luca Zappella*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/948d8ba4e30c8c3a800cf436b31f376e-Abstract-Conference.html)

**Abstract**:

Parallelization techniques have become ubiquitous for accelerating inference and training of deep neural networks. Despite this, several operations are still performed in a sequential manner. For instance, the forward and backward passes are executed layer-by-layer, and the output of diffusion models is produced by applying a sequence of denoising steps. This sequential approach results in a computational cost proportional to the number of steps involved, presenting a potential bottleneck as the number of steps increases. In this work, we introduce DeepPCR, a novel algorithm which parallelizes typically sequential operations in order to speed up inference and training of neural networks. DeepPCR is based on interpreting a sequence of $L$ steps as the solution of a specific system of equations, which we recover using the Parallel Cyclic Reduction algorithm. This reduces the complexity of computing the sequential operations from $\mathcal{O}(L)$ to $\mathcal{O}(\log_2L)$, thus yielding a speedup for large $L$. To verify the theoretical lower complexity of the algorithm, and to identify regimes for speedup, we test the effectiveness of DeepPCR in parallelizing the forward and backward pass in multi-layer perceptrons, and reach speedups of up to $30\times$ for the forward and $200\times$ for the backward pass. We additionally showcase the flexibility of DeepPCR by parallelizing training of ResNets with as many as 1024 layers, and generation in diffusion models, enabling up to $7\times$ faster training and $11\times$ faster generation, respectively, when compared to the sequential approach.

----

## [2062] DELTA: Diverse Client Sampling for Fasting Federated Learning

**Authors**: *Lin Wang, Yongxin Guo, Tao Lin, Xiaoying Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/949c57d30f8791e3ae42646081b3c102-Abstract-Conference.html)

**Abstract**:

Partial client participation has been widely adopted in Federated Learning (FL) to reduce the communication burden efficiently. However, an inadequate client sampling scheme can lead to the selection of unrepresentative subsets, resulting in significant variance in model updates and slowed convergence. Existing sampling methods are either biased or can be further optimized for faster convergence.In this paper, we present DELTA, an unbiased sampling scheme designed to alleviate these issues. DELTA characterizes the effects of client diversity and local variance, and samples representative clients with valuable information for global model updates. In addition, DELTA is a proven optimal unbiased sampling scheme that minimizes variance caused by partial client participation and outperforms other unbiased sampling schemes in terms of convergence.  Furthermore, to address full-client gradient dependence, we provide a practical version of DELTA depending on the available clients' information, and also analyze its convergence. Our results are validated through experiments on both synthetic and real-world datasets.

----

## [2063] OpenAssistant Conversations - Democratizing Large Language Model Alignment

**Authors**: *Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi Rui Tam, Keith Stevens, Abdullah Barhoum, Duc Nguyen, Oliver Stanley, Richárd Nagyfi, Shahul ES, Sameer Suri, David Glushkov, Arnav Dantuluri, Andrew Maguire, Christoph Schuhmann, Huu Nguyen, Alexander Mattick*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/949f0f8f32267d297c2d4e3ee10a2e7e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Aligning large language models (LLMs) with human preferences has proven to drastically improve usability and has driven rapid adoption as demonstrated by ChatGPT.Alignment techniques such as supervised fine-tuning (\textit{SFT}) and  reinforcement learning from human feedback (\textit{RLHF}) greatly reduce the required skill and domain knowledge to effectively harness the capabilities of LLMs, increasing their accessibility and utility across various domains.However, state-of-the-art alignment techniques like \textit{RLHF} rely on high-quality human feedback data, which is expensive to create and often remains proprietary.In an effort to democratize research on large-scale alignment, we release OpenAssistant Conversations, a human-generated, human-annotated assistant-style conversation corpus consisting of 161,443 messages in 35 different languages, annotated with 461,292 quality ratings, resulting in over 10,000 complete and fully annotated conversation trees.The corpus is a product of a worldwide crowd-sourcing effort involving over 13,500 volunteers.Models trained on OpenAssistant Conversations show consistent improvements on standard benchmarks over respective base models.We release our code\footnote{\git} and data\footnote{\data} under a fully permissive licence.

----

## [2064] Conformal Meta-learners for Predictive Inference of Individual Treatment Effects

**Authors**: *Ahmed M. Alaa, Zaid Ahmad, Mark J. van der Laan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94ab02a30b0e4a692a42ccd0b4c55399-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94ab02a30b0e4a692a42ccd0b4c55399-Abstract-Conference.html)

**Abstract**:

We investigate the problem of machine learning-based (ML) predictive inference on individual treatment effects (ITEs). Previous work has focused primarily on developing ML-based “meta-learners” that can provide point estimates of the conditional average treatment effect (CATE)—these are model-agnostic approaches for combining intermediate nuisance estimates to produce estimates of CATE. In this paper, we develop conformal meta-learners, a general framework for issuing predictive intervals for ITEs by applying the standard conformal prediction (CP) procedure on top of CATE meta-learners. We focus on a broad class of meta-learners based on two-stage pseudo-outcome regression and develop a stochastic ordering framework to study their validity. We show that inference with conformal meta-learners is marginally valid if their (pseudo-outcome) conformity scores stochastically dominate “oracle” conformity scores evaluated on the unobserved ITEs. Additionally, we prove that commonly used CATE meta-learners, such as the doubly-robust learner, satisfy a model- and distribution-free stochastic (or convex) dominance condition, making their conformal inferences valid for practically-relevant levels of target coverage. Whereas existing procedures conduct inference on nuisance parameters (i.e., potential outcomes) via weighted CP, conformal meta-learners enable direct inference on the target parameter (ITE). Numerical experiments show that conformal meta-learners provide valid intervals with competitive efficiency while retaining the favorable point estimation properties of CATE meta-learners.

----

## [2065] Simple and Controllable Music Generation

**Authors**: *Jade Copet, Felix Kreuk, Itai Gat, Tal Remez, David Kant, Gabriel Synnaeve, Yossi Adi, Alexandre Défossez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html)

**Abstract**:

We tackle the task of conditional music generation. We introduce MusicGen, a single Language Model (LM) that operates over several streams of compressed discrete music representation, i.e., tokens. Unlike prior work, MusicGen is comprised of a single-stage transformer LM together with efficient token interleaving patterns, which eliminates the need for cascading several models, e.g., hierarchically or upsampling. Following this approach, we demonstrate how MusicGen can generate high-quality samples, both mono and stereo, while being conditioned on textual description or melodic features, allowing better controls over the generated output. We conduct extensive empirical evaluation, considering both automatic and human studies, showing the proposed approach is superior to the evaluated baselines on a standard text-to-music benchmark. Through ablation studies, we shed light over the importance of each of the components comprising MusicGen. Music samples, code, and models are available at https://github.com/facebookresearch/audiocraft

----

## [2066] Temporal Robustness against Data poisoning

**Authors**: *Wenxiao Wang, Soheil Feizi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94bcb01789fccf15afe2764d8fe0f40e-Abstract-Conference.html)

**Abstract**:

Data poisoning considers cases when an adversary manipulates the behavior of machine learning algorithms through malicious training data. Existing threat models of data poisoning center around a single metric, the number of poisoned samples. In consequence, if attackers can poison more samples than expected with affordable overhead, as in many practical scenarios, they may be able to render existing defenses ineffective in a short time. To address this issue, we leverage timestamps denoting the birth dates of data, which are often available but neglected in the past. Benefiting from these timestamps, we propose a temporal threat model of data poisoning with two novel metrics, earliness and duration, which respectively measure how long an attack started in advance and how long an attack lasted. Using these metrics, we define the notions of temporal robustness against data poisoning, providing a meaningful sense of protection even with unbounded amounts of poisoned samples when the attacks are temporally bounded. We present a benchmark with an evaluation protocol simulating continuous data collection and periodic deployments of updated models, thus enabling empirical evaluation of temporal robustness. Lastly, we develop and also empirically verify a baseline defense, namely temporal aggregation, offering provable temporal robustness and highlighting the potential of our temporal threat model for data poisoning.

----

## [2067] Optimal Treatment Regimes for Proximal Causal Learning

**Authors**: *Tao Shen, Yifan Cui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94ccfdb2ca14f33a86a0b9b7d0c1bfb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94ccfdb2ca14f33a86a0b9b7d0c1bfb1-Abstract-Conference.html)

**Abstract**:

A common concern when a policymaker draws causal inferences from and makes decisions based on observational data is that the measured covariates are insufficiently rich to account for all sources of confounding, i.e., the standard no confoundedness assumption fails to hold. The recently proposed proximal causal inference framework shows that proxy variables that abound in real-life scenarios can be leveraged to identify causal effects and therefore facilitate decision-making. Building upon this line of work, we propose a novel optimal individualized treatment regime based on so-called outcome and treatment confounding bridges. We then show that the value function of this new optimal treatment regime is superior to that of existing ones in the literature. Theoretical guarantees, including identification, superiority, excess value bound, and consistency of the estimated regime, are established. Furthermore, we demonstrate the proposed optimal regime via numerical experiments and a real data application.

----

## [2068] Debias Coarsely, Sample Conditionally: Statistical Downscaling through Optimal Transport and Probabilistic Diffusion Models

**Authors**: *Zhong Yi Wan, Ricardo Baptista, Anudhyan Boral, Yi-Fan Chen, John Anderson, Fei Sha, Leonardo Zepeda-Núñez*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94d13c2401fe119e57ba325b6fe526e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94d13c2401fe119e57ba325b6fe526e0-Abstract-Conference.html)

**Abstract**:

We introduce a two-stage probabilistic framework for statistical downscaling using unpaired data. Statistical downscaling seeks a probabilistic map to transform low-resolution data from a biased coarse-grained numerical scheme to high-resolution data that is consistent with a high-fidelity scheme. Our framework tackles the problem bycomposing two transformations: (i) a debiasing step via an optimal transport map, and (ii) an upsampling step achieved by a probabilistic diffusion model with a posteriori conditional sampling. This approach characterizes a conditional distribution without needing paired data, and faithfully recovers relevant physical statistics from biased samples. We demonstrate the utility of the proposed approach on one- and two-dimensional fluid flow problems, which are representative of the core difficulties present in numerical simulations of weather and climate. Our method produces realistic high-resolution outputs from low-resolution inputs, by upsampling resolutions of $8\times$ and $16\times$. Moreover, our procedure correctly matches the statistics of physical quantities, even when the low-frequency content of the inputs and outputs do not match, a crucial but difficult-to-satisfy assumption needed by current state-of-the-art alternatives. Code for this work is available at: https://github.com/google-research/swirl-dynamics/tree/main/swirl_dynamics/projects/probabilistic_diffusion.

----

## [2069] Transformers over Directed Acyclic Graphs

**Authors**: *Yuankai Luo, Veronika Thost, Lei Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html)

**Abstract**:

Transformer models have recently gained popularity in graph representation learning as they have the potential to learn complex relationships beyond the ones captured by regular graph neural networks.The main research question is how to inject the structural bias of graphs into the transformer architecture,and several proposals have been made for undirected molecular graphs and, recently, also for larger network graphs.In this paper, we study transformers over directed acyclic graphs (DAGs) and propose architecture adaptations tailored to DAGs: (1) An attention mechanism that is considerably more efficient than the regular quadratic complexity of transformers and at the same time faithfully captures the DAG structure, and (2) a positional encoding of the DAG's partial order, complementing the former.We rigorously evaluate our approach over various types of tasks, ranging from classifying source code graphs to nodes in citation networks, and show that it is effective in two important aspects: in making graph transformers generally outperform graph neural networks tailored to DAGs and in improving SOTA graph transformer performance in terms of both quality and efficiency.

----

## [2070] Understanding and Mitigating Copying in Diffusion Models

**Authors**: *Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, Tom Goldstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html)

**Abstract**:

Images generated by diffusion models like Stable Diffusion are increasingly widespread. Recent works and even lawsuits have shown that these models are prone to replicating their training data, unbeknownst to the user. In this paper, we first analyze this memorization problem in text-to-image diffusion models.  While it is widely believed that duplicated images in the training set are responsible for content replication at inference time, we observe that the text conditioning of the model plays a similarly important role. In fact, we see in our experiments that data replication often does not happen for unconditional models, while it is common in the text-conditional case. Motivated by our findings, we then propose several techniques for reducing data replication at both training and inference time by randomizing and augmenting image captions in the training set. Code is available at https://github.com/somepago/DCR.

----

## [2071] Credal Marginal MAP

**Authors**: *Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Fabio Cozman, Alexander G. Gray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/953390c834451505703c9da45de634d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/953390c834451505703c9da45de634d8-Abstract-Conference.html)

**Abstract**:

Credal networks extend Bayesian networks to allow for imprecision in probability values. Marginal MAP is a widely applicable mixed inference task that identifies the most likely assignment for a subset of variables (called MAP variables). However, the  task is extremely difficult to solve in credal networks particularly because the evaluation of each complete MAP assignment involves exact likelihood computations (combinatorial sums) over the vertices of a complex joint credal set representing the space of all possible marginal distributions of the MAP variables. In this paper, we explore Credal Marginal MAP inference and develop new exact methods based on variable elimination and depth-first search as well as several approximation schemes based on the mini-bucket partitioning and stochastic local search. An extensive empirical evaluation demonstrates the effectiveness of our new methods on random as well as real-world benchmark problems.

----

## [2072] Multi-task Representation Learning for Pure Exploration in Bilinear Bandits

**Authors**: *Subhojyoti Mukherjee, Qiaomin Xie, Josiah Hanna, Robert D. Nowak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95464e2e49103dc560091ed2c64a5b12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95464e2e49103dc560091ed2c64a5b12-Abstract-Conference.html)

**Abstract**:

We study multi-task representation learning for the problem of pure exploration in bilinear bandits. In bilinear bandits, an action takes theform of a pair of arms from two different entity types and the reward is a bilinear function of the known feature vectors of the arms. In the \textit{multi-task bilinear bandit problem}, we aim to find optimal actions for multiple tasks that share a common low-dimensional linear representation. The objective is to leverage this characteristic to expedite the process of identifying the best pair of arms for all tasks. We propose the algorithm GOBLIN that uses an experimental design approach to optimize sample allocations for learning the global representation as well as minimize the number of samples needed to identify the optimal pair of arms in individual tasks. To the best of our knowledge, this is the first study to give sample complexity analysis for pure exploration in bilinear bandits with shared representation. Our results demonstrate that by learning the shared representation across tasks, we achieve significantly improved sample complexity compared to the traditional approach of solving tasks independently.

----

## [2073] Mechanic: A Learning Rate Tuner

**Authors**: *Ashok Cutkosky, Aaron Defazio, Harsh Mehta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/955499a8e2860ed746717c1374224c43-Abstract-Conference.html)

**Abstract**:

We introduce a technique for tuning the learning rate scale factor of any base optimization algorithm and schedule automatically, which we call Mechanic. Our method provides a practical realization of recent theoretical reductions for accomplishing a similar goal in online convex optimization. We rigorously evaluate Mechanic on a range of large scale deep learning tasks with varying batch sizes, schedules, and base optimization algorithms. These experiments demonstrate that depending on the problem, Mechanic either comes very close to, matches or even improves upon manual tuning of learning rates.

----

## [2074] Compositional Policy Learning in Stochastic Control Systems with Formal Guarantees

**Authors**: *Dorde Zikelic, Mathias Lechner, Abhinav Verma, Krishnendu Chatterjee, Thomas A. Henzinger*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95827e011b9e899f189a01fe2f4ef316-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95827e011b9e899f189a01fe2f4ef316-Abstract-Conference.html)

**Abstract**:

Reinforcement learning has shown promising results in learning neural network policies for complicated control tasks. However, the lack of formal guarantees about the behavior of such policies remains an impediment to their deployment. We propose a novel method for learning a composition of neural network policies in stochastic environments, along with a formal certificate which guarantees that a specification over the policy's behavior is satisfied with the desired probability. Unlike prior work on verifiable RL, our approach leverages the compositional nature of logical specifications provided in SpectRL, to learn over graphs of probabilistic reach-avoid specifications. The formal guarantees are provided by learning neural network policies together with reach-avoid supermartingales (RASM) for the graph’s sub-tasks and then composing them into a global policy. We also derive a tighter lower bound compared to previous work on the probability of reach-avoidance implied by a RASM, which is required to find a compositional policy with an acceptable probabilistic threshold for complex tasks with multiple edge policies. We implement a prototype of our approach and evaluate it on a Stochastic Nine Rooms environment.

----

## [2075] Fast Exact Leverage Score Sampling from Khatri-Rao Products with Applications to Tensor Decomposition

**Authors**: *Vivek Bharadwaj, Osman Asif Malik, Riley Murray, Laura Grigori, Aydin Buluç, James Demmel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html)

**Abstract**:

We present a data structure to randomly sample rows from the Khatri-Rao product of several matrices according to the exact distribution of its leverage scores. Our proposed sampler draws each row in time logarithmic in the height of the Khatri-Rao product and quadratic in its column count, with persistent space overhead at most the size of the input matrices. As a result, it tractably draws samples even when the matrices forming the Khatri-Rao product have tens of millions of rows each. When used to sketch the linear least-squares problems arising in Candecomp / PARAFAC decomposition, our method achieves lower asymptotic complexity per solve than recent state-of-the-art methods. Experiments on billion-scale sparse tensors and synthetic data validate our theoretical claims, with our algorithm achieving higher accuracy than competing methods as the decomposition rank grows.

----

## [2076] Online Performative Gradient Descent for Learning Nash Equilibria in Decision-Dependent Games

**Authors**: *Zihan Zhu, Ethan Fang, Zhuoran Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html)

**Abstract**:

We study the multi-agent game within the innovative framework of decision-dependent games, which establishes a feedback mechanism that population data reacts to agentsâ€™ actions and further characterizes the strategic interactions between agents. We focus on finding the Nash equilibrium of decision-dependent games in the bandit feedback setting. However, since agents are strategically coupled, traditional gradient-based methods are infeasible without the gradient oracle. To overcome this challenge, we model the strategic interactions by a general parametric model and propose a novel online algorithm, Online Performative Gradient Descent (OPGD), which leverages the ideas of online stochastic approximation and projected gradient descent to learn the Nash equilibrium in the context of function approximation for the unknown gradient. In particular, under mild assumptions on the function classes defined in the parametric model, we prove that OPGD can find the Nash equilibrium efficiently for strongly monotone decision-dependent games. Synthetic numerical experiments validate our theory.

----

## [2077] AD-PT: Autonomous Driving Pre-Training with Large-scale Point Cloud Dataset

**Authors**: *Jiakang Yuan, Bo Zhang, Xiangchao Yan, Botian Shi, Tao Chen, Yikang Li, Yu Qiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95ab5c3e26fd82c7de3230bbad087d2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95ab5c3e26fd82c7de3230bbad087d2d-Abstract-Conference.html)

**Abstract**:

It is a long-term vision for Autonomous Driving (AD) community that the perception models can learn from a large-scale point cloud dataset, to obtain unified representations that can achieve promising results on different tasks or benchmarks. Previous works mainly focus on the self-supervised pre-training pipeline, meaning that they perform the pre-training and fine-tuning on the same benchmark, which is difficult to attain the performance scalability and cross-dataset application for the pre-training checkpoint.  In this paper, for the first time, we are committed to building a large-scale pre-training point-cloud dataset with diverse data distribution, and meanwhile learning generalizable representations from such a diverse pre-training dataset. We formulate the point-cloud pre-training task as a semi-supervised problem, which leverages the few-shot labeled and massive unlabeled point-cloud data to generate the unified backbone representations that can be directly applied to many baseline models and benchmarks, decoupling the AD-related pre-training process and downstream fine-tuning task. During the period of backbone pre-training, by enhancing the scene- and instance-level distribution diversity and exploiting the backbone's ability to learn from unknown instances, we achieve significant performance gains on a series of downstream perception benchmarks including Waymo, nuScenes, and KITTI, under different baseline models like PV-RCNN++, SECOND, CenterPoint.

----

## [2078] Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors

**Authors**: *Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, Marzyeh Ghassemi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html)

**Abstract**:

Deployed language models decay over time due to shifting inputs, changing user needs, or emergent world-knowledge gaps. When such problems are identified, we want to make targeted edits while avoiding expensive retraining. However, current model editors, which modify such behaviors of pre-trained models, degrade model performance quickly across multiple, sequential edits. We propose GRACE, a \textit{lifelong} model editing method, which implements spot-fixes on streaming errors of a deployed model, ensuring minimal impact on unrelated inputs. GRACE writes new mappings into a pre-trained model's latent space, creating a discrete, local codebook of edits without altering model weights. This is the first method enabling thousands of sequential edits using only streaming errors. Our experiments on T5, BERT, and GPT models show GRACE's state-of-the-art performance in making and retaining edits, while generalizing to unseen inputs. Our code is available at github.com/thartvigsen/grace.

----

## [2079] On the Identifiability of Sparse ICA without Assuming Non-Gaussianity

**Authors**: *Ignavier Ng, Yujia Zheng, Xinshuai Dong, Kun Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/95b7a93e60fdfd10cc202f44fd6adf5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/95b7a93e60fdfd10cc202f44fd6adf5f-Abstract-Conference.html)

**Abstract**:

Independent component analysis (ICA) is a fundamental statistical tool used to reveal hidden generative processes from observed data. However, traditional ICA approaches struggle with the rotational invariance inherent in Gaussian distributions, often necessitating the assumption of non-Gaussianity in the underlying sources. This may limit their applicability in broader contexts. To accommodate Gaussian sources, we develop an identifiability theory that relies on second-order statistics without imposing further preconditions on the distribution of sources, by introducing novel assumptions on the connective structure from sources to observed variables. Different from recent work that focuses on potentially restrictive connective structures, our proposed assumption of structural variability is both considerably less restrictive and provably necessary. Furthermore, we propose two estimation methods based on second-order statistics and sparsity constraint. Experimental results are provided to validate our identifiability theory and estimation methods.

----

## [2080] Unbiased Compression Saves Communication in Distributed Optimization: When and How Much?

**Authors**: *Yutong He, Xinmeng Huang, Kun Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9602d22a8c791f23f8e4d1398e3fb5be-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9602d22a8c791f23f8e4d1398e3fb5be-Abstract-Conference.html)

**Abstract**:

Communication compression is a common technique in distributed optimizationthat can alleviate communication overhead by transmitting compressed gradientsand model parameters. However, compression can introduce information distortion,which slows down convergence and incurs more communication rounds to achievedesired solutions. Given the trade-off between lower per-round communicationcosts and additional rounds of communication, it is unclear whether communicationcompression reduces the total communication cost.This paper explores the conditions under which unbiased compression, a widelyused form of compression, can reduce the total communication cost, as well as theextent to which it can do so. To this end, we present the first theoretical formulationfor characterizing the total communication cost in distributed optimization withunbiased compressors. We demonstrate that unbiased compression alone does notnecessarily save the total communication cost, but this outcome can be achievedif the compressors used by all workers are further assumed independent. Weestablish lower bounds on the communication rounds required by algorithms usingindependent unbiased compressors to minimize smooth convex functions andshow that these lower bounds are tight by refining the analysis for ADIANA.Our results reveal that using independent unbiased compression can reduce thetotal communication cost by a factor of up to $\Theta(\sqrt{\min\\{n,\kappa\\}})$ when all localsmoothness constants are constrained by a common upper bound, where $n$ is thenumber of workers and $\kappa$ is the condition number of the functions being minimized.These theoretical findings are supported by experimental results.

----

## [2081] Pareto Frontiers in Deep Feature Learning: Data, Compute, Width, and Luck

**Authors**: *Benjamin L. Edelman, Surbhi Goel, Sham M. Kakade, Eran Malach, Cyril Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/960573a3b797441aec39caa9f74bc793-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/960573a3b797441aec39caa9f74bc793-Abstract-Conference.html)

**Abstract**:

In modern deep learning, algorithmic choices (such as width, depth, and learning rate) are known to modulate nuanced resource tradeoffs. This work investigates how these complexities necessarily arise for feature learning in the presence of computational-statistical gaps. We begin by considering offline sparse parity learning, a supervised classification problem which admits a statistical query lower bound for gradient-based training of a multilayer perceptron. This lower bound can be interpreted as a multi-resource tradeoff frontier: successful learning can only occur if one is sufficiently rich (large model), knowledgeable (large dataset), patient (many training iterations), or lucky (many random guesses). We show, theoretically and experimentally, that sparse initialization and increasing network width yield significant improvements in sample efficiency in this setting. Here, width plays the role of parallel search: it amplifies the probability of finding "lottery ticket" neurons, which learn sparse features more sample-efficiently. Finally, we show that the synthetic sparse parity task can be useful as a proxy for real problems requiring axis-aligned feature learning. We demonstrate improved sample efficiency on tabular classification benchmarks by using wide, sparsely-initialized MLP models; these networks sometimes outperform tuned random forests.

----

## [2082] Reliable learning in challenging environments

**Authors**: *Maria-Florina Balcan, Steve Hanneke, Rattana Pukdee, Dravyansh Sharma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96189e90e599ccc43f00434ff3ed0312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96189e90e599ccc43f00434ff3ed0312-Abstract-Conference.html)

**Abstract**:

The problem of designing learners that provide guarantees that their predictions are provably correct is of increasing importance in machine learning. However, learning theoretic guarantees have only been considered in very specific settings.  In this work, we consider the design and analysis of reliable learners in challenging test-time environments as encountered in modern machine learning problems: namely adversarial test-time attacks (in several variations) and natural distribution shifts.  In this work, we provide a reliable learner with provably optimal guarantees in such settings. We discuss computationally feasible implementations of the learner and further show that our algorithm achieves strong positive performance guarantees on several natural examples: for example, linear separators under log-concave distributions or smooth boundary classifiers under smooth probability distributions.

----

## [2083] Retaining Beneficial Information from Detrimental Data for Neural Network Repair

**Authors**: *Long-Kai Huang, Peilin Zhao, Junzhou Huang, Sinno Jialin Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html)

**Abstract**:

The performance of deep learning models heavily relies on the quality of the training data. Inadequacies in the training data, such as corrupt input or noisy labels, can lead to the failure of model generalization. Recent studies propose repairing the model by identifying the training samples that contribute to the failure and removing their influence from the model. However, it is important to note that the identified data may contain both beneficial and detrimental information. Simply erasing the information of the identified data from the model can have a negative impact on its performance, especially when accurate data is mistakenly identified as detrimental and removed. To overcome this challenge, we propose a novel approach that leverages the knowledge obtained from a retained clean set. Our method first identifies harmful data by utilizing the clean set, then separates the beneficial and detrimental information within the identified data. Finally, we utilize the extracted beneficial information to enhance the model's performance. Through empirical evaluations, we demonstrate that our method outperforms baseline approaches in both identifying harmful data and rectifying model failures. Particularly in scenarios where identification is challenging and a significant amount of benign data is involved, our method improves performance while the baselines deteriorate due to the erroneous removal of beneficial information.

----

## [2084] Unsupervised Optical Flow Estimation with Dynamic Timing Representation for Spike Camera

**Authors**: *Lujie Xia, Ziluo Ding, Rui Zhao, Jiyuan Zhang, Lei Ma, Zhaofei Yu, Tiejun Huang, Ruiqin Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html)

**Abstract**:

Efficiently selecting an appropriate spike stream data length to extract precise information is the key to the spike vision tasks. To address this issue, we propose a dynamic timing representation for spike streams. Based on multi-layers architecture, it applies dilated convolutions on temporal dimension to extract features on multi-temporal scales with few parameters. And we design layer attention to dynamically fuse these features. Moreover, we propose an unsupervised learning method for optical flow estimation in a spike-based manner to break the dependence on labeled data. In addition, to verify the robustness, we also build a spike-based synthetic validation dataset for extreme scenarios in autonomous driving, denoted as SSES dataset. It consists of various corner cases. Experiments show that our method can predict optical flow from spike streams in different high-speed scenes, including real scenes. For instance, our method achieves $15\%$ and $19\%$ error reduction on PHM dataset compared to the best spike-based work, SCFlow, in $\Delta t=10$ and $\Delta t=20$ respectively, using the same settings as in previous works. The source code and dataset are available at \href{https://github.com/Bosserhead/USFlow}{https://github.com/Bosserhead/USFlow}.

----

## [2085] No-regret Algorithms for Fair Resource Allocation

**Authors**: *Abhishek Sinha, Ativ Joshi, Rajarshi Bhattacharjee, Cameron Musco, Mohammad Hajiesmaili*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96842011407c2691ab4eefff48fc864d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96842011407c2691ab4eefff48fc864d-Abstract-Conference.html)

**Abstract**:

We consider a fair resource allocation problem in the no-regret setting against an unrestricted adversary. The objective is to allocate resources equitably among several agents in an online fashion so that the difference of the aggregate $\alpha$-fair utilities of the agents achieved by an optimal static clairvoyant allocation and the online policy grows sublinearly with time. The problem inherits its difficulty from the non-separable nature of the global $\alpha$-fairness function. Previously, it was shown that no online policy could achieve a sublinear standard regret in this problem. In this paper, we propose an efficient online resource allocation policy, called Online Fair Allocation ($\texttt{OFA}$), that achieves sublinear $c_\alpha$-approximate regret with approximation factor $c_\alpha=(1-\alpha)^{-(1-\alpha)}\leq 1.445,$ for $0\leq \alpha < 1$. Our upper bound on the $c_\alpha$-regret for this problem exhibits a surprising \emph{phase transition} phenomenon -- transitioning from a power-law to a constant at the critical exponent $\alpha=\frac{1}{2}.$  Our result also resolves an open problem in designing an efficient no-regret policy for the online job scheduling problem in certain parameter regimes. Along the way, we introduce new algorithmic and analytical techniques, including greedy estimation of the future gradients for non-additive global reward functions and bootstrapping second-order regret bounds, which may be of independent interest.

----

## [2086] Bypass Exponential Time Preprocessing: Fast Neural Network Training via Weight-Data Correlation Preprocessing

**Authors**: *Josh Alman, Jiehao Liang, Zhao Song, Ruizhe Zhang, Danyang Zhuo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9690d4746230cfea3d067fca695ba648-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9690d4746230cfea3d067fca695ba648-Abstract-Conference.html)

**Abstract**:

Over the last decade, deep neural networks have transformed our society, and they are already widely applied in various machine learning applications. State-of-the-art deep neural networks are becoming larger in size every year to deliver increasing model accuracy, and as a result, model training consumes substantial computing resources and will only consume more in the future.Using current training methods, in each iteration, to process a data point $x \in \mathbb{R}^d$ in a layer, we need to spend $\Theta(md)$ time to evaluate all the $m$ neurons in the layer. This means processing the entire layer takes $\Theta(nmd)$ time for $n$ data points. Recent work [Song, Yang and Zhang, NeurIPS 2021] reduces this time per iteration to $o(nmd)$, but requires exponential time to preprocess either the data or the neural network weights, making it unlikely to have practical usage. In this work, we present a new preprocessing method that simply stores the weight-data correlation in a tree data structure in order to quickly and dynamically detect which neurons fire at each iteration. Our method requires only $O(nmd)$ time in preprocessing and still achieves $o(nmd)$ time per iteration. We complement our new algorithm with a lower bound, proving that assuming a popular conjecture from complexity theory, one could not substantially speed up our algorithm for dynamic detection of firing neurons.

----

## [2087] Online PCA in Converging Self-consistent Field Equations

**Authors**: *Xihan Li, Xiang Chen, Rasul Tutunov, Haitham Bou-Ammar, Lei Wang, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/969c14957c0df5ce2db642b3a5fa985c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/969c14957c0df5ce2db642b3a5fa985c-Abstract-Conference.html)

**Abstract**:

Self-consistent Field (SCF) equation is a type of nonlinear eigenvalue problem in which the matrix to be eigen-decomposed is a function of its own eigenvectors. It is of great significance in computational science for its connection to the Schr√∂dinger equation. Traditional fixed-point iteration methods for solving such equations suffer from non-convergence issues. In this work, we present a novel perspective on such SCF equations as a principal component analysis (PCA) for non-stationary time series, in which a distribution and its own top principal components are mutually updated over time, and the equilibrium state of the model corresponds to the solution of the SCF equations. By the new perspective, online PCA techniques are able to engage in so as to enhance the convergence of the model towards the equilibrium state, acting as a new set of tools for converging the SCF equations. With several numerical adaptations, we then develop a new algorithm for converging the SCF equation, and demonstrated its high convergence capacity with experiments on both synthesized and real electronic structure scenarios.

----

## [2088] DiffPack: A Torsional Diffusion Model for Autoregressive Protein Side-Chain Packing

**Authors**: *Yangtian Zhang, Zuobai Zhang, Bozitao Zhong, Sanchit Misra, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96a54c09569ebbdd9ecb22f5012e6b66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96a54c09569ebbdd9ecb22f5012e6b66-Abstract-Conference.html)

**Abstract**:

Proteins play a critical role in carrying out biological functions, and their 3D structures are essential in determining their functions. Accurately predicting the conformation of protein side-chains given their backbones is important for applications in protein structure prediction, design and protein-protein interactions. Traditional methods are computationally intensive and have limited accuracy, while existing machine learning methods treat the problem as a regression task and overlook the restrictions imposed by the constant covalent bond lengths and angles. In this work, we present DiffPack, a torsional diffusion model that learns the joint distribution of side-chain torsional angles, the only degrees of freedom in side-chain packing, by diffusing and denoising on the torsional space. To avoid issues arising from simultaneous perturbation of all four torsional angles, we propose autoregressively generating the four torsional angles from $\chi_1$ to $\chi_4$ and training diffusion models for each torsional angle. We evaluate the method on several benchmarks for protein side-chain packing and show that our method achieves improvements of 11.9% and 13.5% in angle accuracy on CASP13 and CASP14, respectively, with a significantly smaller model size ($60\times$ fewer parameters). Additionally, we show the effectiveness of our method  in enhancing side-chain predictions in the AlphaFold2 model. Code is available at https://github.com/DeepGraphLearning/DiffPack.

----

## [2089] ARTIC3D: Learning Robust Articulated 3D Shapes from Noisy Web Image Collections

**Authors**: *Chun-Han Yao, Amit Raj, Wei-Chih Hung, Michael Rubinstein, Yuanzhen Li, Ming-Hsuan Yang, Varun Jampani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96aca14d6c4dcd3adf54bc2c5ad7f138-Abstract-Conference.html)

**Abstract**:

Estimating 3D articulated shapes like animal bodies from monocular images is inherently challenging due to the ambiguities of camera viewpoint, pose, texture, lighting, etc. We propose ARTIC3D, a self-supervised framework to reconstruct per-instance 3D shapes from a sparse image collection in-the-wild. Specifically, ARTIC3D is built upon a skeleton-based surface representation and is further guided by 2D diffusion priors from Stable Diffusion. First, we enhance the input images with occlusions/truncation via 2D diffusion to obtain cleaner mask estimates and semantic features. Second, we perform diffusion-guided 3D optimization to estimate shape and texture that are of high-fidelity and faithful to input images. We also propose a novel technique to calculate more stable image-level gradients via diffusion models compared to existing alternatives. Finally, we produce realistic animations by fine-tuning the rendered shape and texture under rigid part transformations. Extensive evaluations on multiple existing datasets as well as newly introduced noisy web image collections with occlusions and truncation demonstrate that ARTIC3D outputs are more robust to noisy images, higher quality in terms of shape and texture details, and more realistic when animated.

----

## [2090] Noether Embedding: Efficient Learning of Temporal Regularities

**Authors**: *Chi Gao, Zidong Zhou, Luping Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html)

**Abstract**:

Learning to detect and encode temporal regularities (TRs) in events is a prerequisite for human-like intelligence. These regularities should be formed from limited event samples and stored as easily retrievable representations. Existing event embeddings, however, cannot effectively decode TR validity with well-trained vectors, let alone satisfy the efficiency requirements. We develop Noether Embedding (NE) as the first efficient TR learner with event embeddings. Specifically, NE possesses the intrinsic time-translation symmetries of TRs indicated as conserved local energies in the embedding space. This structural bias reduces the calculation of each TR validity to embedding each event sample, enabling NE to achieve data-efficient TR formation insensitive to sample size and time-efficient TR retrieval in constant time complexity. To comprehensively evaluate the TR learning capability of embedding models, we define complementary tasks of TR detection and TR query, formulate their evaluation metrics, and assess embeddings on classic ICEWS14, ICEWS18, and GDELT datasets. Our experiments demonstrate that NE consistently achieves about double the F1 scores for detecting valid TRs compared to classic embeddings, and it provides over ten times higher confidence scores for querying TR intervals. Additionally, we showcase NE's potential applications in social event prediction, personal decision-making, and memory-constrained scenarios.

----

## [2091] TACO: Temporal Latent Action-Driven Contrastive Loss for Visual Reinforcement Learning

**Authors**: *Ruijie Zheng, Xiyao Wang, Yanchao Sun, Shuang Ma, Jieyu Zhao, Huazhe Xu, Hal Daumé III, Furong Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96d00450ed65531ffe2996daed487536-Abstract-Conference.html)

**Abstract**:

Despite recent progress in reinforcement learning (RL) from raw pixel data, sample inefficiency continues to present a substantial obstacle. Prior works have attempted to address this challenge by creating self-supervised auxiliary tasks, aiming to enrich the agent's learned representations with control-relevant information for future state prediction.However, these objectives are often insufficient to learn representations that can represent the optimal policy or value function, and they often consider tasks with small, abstract discrete action spaces and thus overlook the importance of action representation learning in continuous control.In this paper, we introduce $\texttt{TACO}$: $\textbf{T}$emporal $\textbf{A}$ction-driven $\textbf{CO}$ntrastive Learning, a simple yet powerful temporal contrastive learning approach that facilitates the concurrent acquisition of latent state and action representations for agents. $\texttt{TACO}$ simultaneously learns a state and an action representation by optimizing the mutual information between representations of current states paired with action sequences and representations of the corresponding future states. Theoretically, $\texttt{TACO}$ can be shown to learn state and action representations that encompass sufficient information for control, thereby improving sample efficiency.For online RL, $\texttt{TACO}$ achieves 40% performance boost after one million environment interaction steps on average across nine challenging visual continuous control tasks from Deepmind Control Suite. In addition, we show that $\texttt{TACO}$ can also serve as a plug-and-play module adding to existing offline visual RL methods to establish the new state-of-the-art performance for offline visual RL across offline datasets with varying quality.

----

## [2092] On the choice of Perception Loss Function for Learned Video Compression

**Authors**: *Sadaf Salehkalaibar, Buu Phan, Jun Chen, Wei Yu, Ashish Khisti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96d328a1f6d8396d8c8a62f2beee252a-Abstract-Conference.html)

**Abstract**:

We study causal, low-latency, sequential video compression when the output is subjected to both a mean squared-error (MSE) distortion loss as well as a perception loss to target realism. Motivated by prior approaches, we consider two different perception loss functions (PLFs). The first, PLF-JD,  considers the joint distribution (JD) of all the video frames up to the current one, while the second metric, PLF-FMD,  considers the framewise marginal distributions (FMD) between the source and reconstruction. Using information theoretic analysis and deep-learning based experiments, we demonstrate that the choice of PLF can have a significant effect on the reconstruction, especially at low-bit rates. In particular, while the reconstruction based on PLF-JD can better preserve the temporal correlation across frames, it also imposes a significant penalty in distortion  compared to PLF-FMD and further makes it more difficult to recover from errors  made in the earlier output frames. Although the choice of PLF decisively affects  reconstruction quality, we also demonstrate that it may not be essential to commit to a particular PLF during encoding and the choice of PLF can be delegated to the decoder. In particular, encoded representations generated by training a system to minimize the MSE (without requiring either PLF) can be  {\em near universal}  and can generate close to optimal reconstructions for either choice of PLF at the decoder.  We validate our results using (one-shot) information-theoretic analysis, detailed study of the rate-distortion-perception tradeoff of the Gauss-Markov source model as well as deep-learning based experiments on moving MNIST and KTH datasets.

----

## [2093] Imitation Learning from Vague Feedback

**Authors**: *Xin-Qiang Cai, Yu-Jie Zhang, Chao-Kai Chiang, Masashi Sugiyama*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/96e35b532b4932a86cce8c929ff3f960-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/96e35b532b4932a86cce8c929ff3f960-Abstract-Conference.html)

**Abstract**:

Imitation learning from human feedback studies how to train well-performed imitation agents with an annotator's relative comparison of two demonstrations (one demonstration is better/worse than the other), which is usually easier to collect than the perfect expert data required by traditional imitation learning. However, in many real-world applications, it is still expensive or even impossible to provide a clear pairwise comparison between two demonstrations with similar quality. This motivates us to study the problem of imitation learning with vague feedback, where the data annotator can only distinguish the paired demonstrations correctly when their quality differs significantly, i.e., one from the expert and another from the non-expert. By modeling the underlying demonstration pool as a mixture of expert and non-expert data, we show that the expert policy distribution can be recovered when the proportion $\alpha$ of expert data is known. We also propose a mixture proportion estimation method for the unknown $\alpha$ case. Then, we integrate the recovered expert policy distribution with generative adversarial imitation learning to form an end-to-end algorithm. Experiments show that our methods outperform standard and preference-based imitation learning methods on various tasks.

----

## [2094] Semantic segmentation of sparse irregular point clouds for leaf/wood discrimination

**Authors**: *Yuchen Bai, Jean-Baptiste Durand, Grégoire Vincent, Florence Forbes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html)

**Abstract**:

Lidar (Light Detection and Ranging) has become an essential part of the remote sensing toolbox used for biosphere monitoring. In particular, Lidar provides the opportunity to map forest leaf area with unprecedented accuracy, while leaf area has remained an important source of uncertainty affecting models of gas exchanges between the vegetation and the atmosphere. Unmanned Aerial Vehicles (UAV) are easy to mobilize and therefore allow frequent revisits to track the response of vegetation to climate change. However, miniature sensors embarked on UAVs usually provide point clouds of limited density, which are further affected by a strong decrease in density from top to bottom of the canopy due to progressively stronger occlusion. In such a context, discriminating leaf points from wood points presents a significant challenge due in particular to strong class imbalance and spatially irregular sampling intensity. Here we introduce a neural network model based on the Pointnet ++ architecture which makes use of point geometry only (excluding any spectral information). To cope with local data sparsity, we propose an innovative sampling scheme which strives to preserve local important geometric information. We also propose a loss function adapted to the severe class imbalance. We show that our model outperforms state-of-the-art alternatives on UAV point clouds. We discuss future possible improvements, particularly regarding much denser point clouds acquired from below the canopy.

----

## [2095] Max-Margin Token Selection in Attention Mechanism

**Authors**: *Davoud Ataee Tarzanagh, Yingcong Li, Xuechen Zhang, Samet Oymak*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/970f59b22f4c72aec75174aae63c7459-Abstract-Conference.html)

**Abstract**:

Attention mechanism is a central component of the transformer architecture which led to the phenomenal success of large language models. However, the theoretical principles underlying the attention mechanism are poorly understood, especially its nonconvex optimization dynamics. In this work, we explore the seminal softmax-attention model $f(X)=\langle Xv, \texttt{softmax}(XWp)\rangle$, where $X$ is the token sequence and $(v,W,p)$ are trainable parameters. We prove that running gradient descent on $p$, or equivalently $W$, converges in direction to a max-margin solution that separates *locally-optimal* tokens from non-optimal ones. This clearly formalizes attention as an optimal token selection mechanism. Remarkably, our results are applicable to general data and precisely characterize *optimality* of tokens in terms of the value embeddings $Xv$ and problem geometry. We also provide a broader regularization path analysis that establishes the margin maximizing nature of attention even for nonlinear prediction heads. When optimizing $v$ and $p$ simultaneously with logistic loss, we identify conditions under which the regularization paths directionally converge to their respective hard-margin SVM solutions where $v$ separates the input features based on their labels. Interestingly, the SVM formulation of $p$ is influenced by the support vector geometry of $v$. Finally, we verify our theoretical findings via numerical experiments and provide insights.

----

## [2096] Locality-Aware Generalizable Implicit Neural Representation

**Authors**: *Doyup Lee, Chiheon Kim, Minsu Cho, Wook-Shin Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9713d53ee4f31781304b1ca43266f8d1-Abstract-Conference.html)

**Abstract**:

Generalizable implicit neural representation (INR) enables a single continuous function, i.e., a coordinate-based neural network, to represent multiple data instances by modulating its weights or intermediate features using latent codes. However, the expressive power of the state-of-the-art modulation is limited due to its inability to localize and capture fine-grained details of data entities such as specific pixels and rays. To address this issue, we propose a novel framework for generalizable INR that combines a transformer encoder with a locality-aware INR decoder. The transformer encoder predicts a set of latent tokens from a data instance to encode local information into each latent token. The locality-aware INR decoder extracts a modulation vector by selectively aggregating the latent tokens via cross-attention for a coordinate input and then predicts the output by progressively decoding with coarse-to-fine modulation through multiple frequency bandwidths. The selective token aggregation and the multi-band feature modulation enable us to learn locality-aware representation in spatial and spectral aspects, respectively. Our framework significantly outperforms previous generalizable INRs and validates the usefulness of the locality-aware latents for downstream tasks such as image generation.

----

## [2097] StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners

**Authors**: *Yonglong Tian, Lijie Fan, Phillip Isola, Huiwen Chang, Dilip Krishnan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/971f1e59cd956cc094da4e2f78c6ea7c-Abstract-Conference.html)

**Abstract**:

We investigate the potential of learning visual representations using synthetic images generated by text-to-image models. This is a natural question in the light of the excellent performance of such models in generating high-quality images. We consider specifically the Stable Diffusion, one of the leading open source text-to-image models. We show that (1) when the generative model is properly configured, training self-supervised methods on synthetic images can match or beat the real image counterpart;(2) by treating the multiple images generated from the same text prompt as positives for each other, we develop a multi-positive contrastive learning method, which we call StableRep. With solely synthetic images, the representations learned by StableRep surpass the performance of representations learned by SimCLR and CLIP using the same set of text prompts and corresponding real images, on large scale datasets. When we further add language supervision, \name~trained with 20M synthetic images (10M captions) achieves better accuracy than CLIP trained with 50M real images (50M captions).

----

## [2098] DropCompute: simple and more robust distributed synchronous training via compute variance reduction

**Authors**: *Niv Giladi, Shahar Gottlieb, Moran Shkolnik, Asaf Karnieli, Ron Banner, Elad Hoffer, Kfir Y. Levy, Daniel Soudry*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html)

**Abstract**:

Background: Distributed training is essential for large scale training of deep neural networks (DNNs). The dominant methods for large scale DNN training are synchronous (e.g. All-Reduce), but these require waiting for all workers in each step. Thus, these methods are limited by the delays caused by straggling workers.Results: We study a typical scenario in which workers are straggling due to variability in compute time. We find an analytical relation between compute time properties and scalability limitations, caused by such straggling workers. With these findings, we propose a simple yet effective decentralized method to reduce the variation among workers and thus improve the robustness of synchronous training. This method can be integrated with the widely used All-Reduce. Our findings are validated on large-scale training tasks using 200 Gaudi Accelerators.

----

## [2099] A Unified Generalization Analysis of Re-Weighting and Logit-Adjustment for Imbalanced Learning

**Authors**: *Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/973a0f50d43cf99118cdab456edcacda-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/973a0f50d43cf99118cdab456edcacda-Abstract-Conference.html)

**Abstract**:

Real-world datasets are typically imbalanced in the sense that only a few classes have numerous samples, while many classes are associated with only a few samples. As a result, a naive ERM learning process will be biased towards the majority classes, making it difficult to generalize to the minority classes. To address this issue, one simple but effective approach is to modify the loss function to emphasize the learning on minority classes, such as re-weighting the losses or adjusting the logits via class-dependent terms. However, existing generalization analysis of such losses is still coarse-grained and fragmented, failing to explain some empirical results. To bridge this gap between theory and practice, we propose a novel technique named data-dependent contraction to capture how these modified losses handle different classes. On top of this technique, a fine-grained generalization bound is established for imbalanced learning, which helps reveal the mystery of re-weighting and logit-adjustment in a unified manner. Furthermore, a principled learning algorithm is developed based on the theoretical insights. Finally, the empirical results on benchmark datasets not only validate the theoretical results but also demonstrate the effectiveness of the proposed method.

----

## [2100] Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming

**Authors**: *Gregory Dexter, Petros Drineas, David P. Woodruff, Taisuke Yasuda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9768645621c2cd6c5b851a06205b92cf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9768645621c2cd6c5b851a06205b92cf-Abstract-Conference.html)

**Abstract**:

Sketching algorithms have recently proven to be a powerful approach both for designing low-space streaming algorithms as well as fast polynomial time approximation schemes (PTAS). In this work, we develop new techniques to extend the applicability of sketching-based approaches to the sparse dictionary learning and the Euclidean $k$-means clustering problems. In particular, we initiate the study of the challenging setting where the dictionary/clustering assignment for each of the $n$ input points must be output, which has surprisingly received little attention in prior work. On the fast algorithms front, we obtain a new approach for designing PTAS's for the $k$-means clustering problem, which generalizes to the first PTAS for the sparse dictionary learning problem. On the streaming algorithms front, we obtain new upper bounds and lower bounds for dictionary learning and $k$-means clustering. In particular, given a design matrix $\mathbf A\in\mathbb R^{n\times d}$ in a turnstile stream, we show an $\tilde O(nr/\epsilon^2 + dk/\epsilon)$ space upper bound for $r$-sparse dictionary learning of size $k$, an $\tilde O(n/\epsilon^2 + dk/\epsilon)$ space upper bound for $k$-means clustering, as well as an $\tilde O(n)$ space upper bound for $k$-means clustering on random order row insertion streams with a natural "bounded sensitivity" assumption. On the lower bounds side, we obtain a general $\tilde\Omega(n/\epsilon + dk/\epsilon)$ lower bound for $k$-means clustering, as well as an $\tilde\Omega(n/\epsilon^2)$ lower bound for algorithms which can estimate the cost of a single fixed set of candidate centers.

----

## [2101] Annotator: A Generic Active Learning Baseline for LiDAR Semantic Segmentation

**Authors**: *Binhui Xie, Shuang Li, Qingju Guo, Chi Harold Liu, Xinjing Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/976cc04f0cbaad7790ce0d665e44f90f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/976cc04f0cbaad7790ce0d665e44f90f-Abstract-Conference.html)

**Abstract**:

Active learning, a label-efficient paradigm, empowers models to interactively query an oracle for labeling new data. In the realm of LiDAR semantic segmentation, the challenges stem from the sheer volume of point clouds, rendering annotation labor-intensive and cost-prohibitive. This paper presents Annotator, a general and efficient active learning baseline, in which a voxel-centric online selection strategy is tailored to efficiently probe and annotate the salient and exemplar voxel girds within each LiDAR scan, even under distribution shift. Concretely, we first execute an in-depth analysis of several common selection strategies such as Random, Entropy, Margin, and then develop voxel confusion degree (VCD) to exploit the local topology relations and structures of point clouds. Annotator excels in diverse settings, with a particular focus on active learning (AL), active source-free domain adaptation (ASFDA), and active domain adaptation (ADA). It consistently delivers exceptional performance across LiDAR semantic segmentation benchmarks, spanning both simulation-to-real and real-to-real scenarios. Surprisingly, Annotator exhibits remarkable efficiency, requiring significantly fewer annotations, e.g., just labeling five voxels per scan in the SynLiDAR â†’ SemanticKITTI task. This results in impressive performance, achieving 87.8% fully-supervised performance under AL, 88.5% under ASFDA, and 94.4% under ADA. We envision that Annotator will offer a simple, general, and efficient solution for label-efficient 3D applications.

----

## [2102] Kissing to Find a Match: Efficient Low-Rank Permutation Representation

**Authors**: *Hannah Dröge, Zorah Lähner, Yuval Bahat, Onofre Martorell Nadal, Felix Heide, Michael Moeller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97826456fb8c02fa368d673a49bbc563-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97826456fb8c02fa368d673a49bbc563-Abstract-Conference.html)

**Abstract**:

Permutation matrices play a key role in matching and assignment problems across the fields, especially in computer vision and robotics. However, memory for explicitly representing permutation matrices grows quadratically with the size of the problem, prohibiting large problem instances. In this work, we propose to tackle the curse of dimensionality of large  permutation matrices by approximating them using low-rank matrix factorization, followed by a nonlinearity. To this end, we rely on the Kissing number theory to infer the minimal rank required for representing a permutation matrix of a given size, which is significantly smaller than the problem size. This leads to a drastic reduction in computation and memory costs, e.g., up to $3$ orders of magnitude less memory for a problem of size $n=20000$, represented using $8.4\times10^5$ elements in two small matrices instead of using a single huge matrix with $4\times 10^8$ elements. The proposed representation allows for accurate representations of large permutation matrices, which in turn enables handling large problems that would have been infeasible otherwise. We demonstrate the applicability and merits of the proposed approach through a series of experiments on a range of problems that involve predicting permutation matrices, from linear and quadratic assignment to shape matching problems.

----

## [2103] Creating Multi-Level Skill Hierarchies in Reinforcement Learning

**Authors**: *Joshua B. Evans, Özgür Simsek*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97b73904e88cc1dc0a3485595eda3753-Abstract-Conference.html)

**Abstract**:

What is a useful skill hierarchy for an autonomous agent? We propose an answer based on a graphical representation of how the interaction between an agent and its environment may unfold. Our approach uses modularity maximisation as a central organising principle to expose the structure of the interaction graph at multiple levels of abstraction. The result is a collection of skills that operate at varying time scales, organised into a hierarchy, where skills that operate over longer time scales are composed of skills that operate over shorter time scales. The entire skill hierarchy is generated automatically, with no human input, including the skills themselves (their behaviour, when they can be called, and when they terminate) as well as the dependency structure between them. In a wide range of environments, this approach generates skill hierarchies that are intuitively appealing and that considerably improve the learning performance of the agent.

----

## [2104] Winner Takes It All: Training Performant RL Populations for Combinatorial Optimization

**Authors**: *Nathan Grinsztajn, Daniel Furelos-Blanco, Shikha Surana, Clément Bonnet, Tom Barrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html)

**Abstract**:

Applying reinforcement learning (RL) to combinatorial optimization problems is attractive as it removes the need for expert knowledge or pre-solved instances. However, it is unrealistic to expect an agent to solve these (often NP-)hard problems in a single shot at inference due to their inherent complexity. Thus, leading approaches often implement additional search strategies, from stochastic sampling and beam-search to explicit fine-tuning. In this paper, we argue for the benefits of learning a population of complementary policies, which can be simultaneously rolled out at inference. To this end, we introduce Poppy, a simple training procedure for populations. Instead of relying on a predefined or hand-crafted notion of diversity, Poppy induces an unsupervised specialization targeted solely at maximizing the performance of the population. We show that Poppy produces a set of complementary policies, and obtains state-of-the-art RL results on three popular NP-hard problems: traveling salesman, capacitated vehicle routing, and job-shop scheduling.

----

## [2105] Revisiting Scalarization in Multi-Task Learning: A Theoretical Perspective

**Authors**: *Yuzheng Hu, Ruicheng Xian, Qilong Wu, Qiuling Fan, Lang Yin, Han Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html)

**Abstract**:

Linear scalarization, i.e., combining all loss functions by a weighted sum, has been the default choice in the literature of multi-task learning (MTL) since its inception. In recent years, there is a surge of interest in developing Specialized Multi-Task Optimizers (SMTOs) that treat MTL as a multi-objective optimization problem. However, it remains open whether there is a fundamental advantage of SMTOs over scalarization. In fact, heated debates exist in the community comparing these two types of algorithms, mostly from an empirical perspective. To approach the above question, in this paper, we revisit scalarization from a theoretical perspective. We focus on linear MTL models and study whether scalarization is capable of fully exploring the Pareto front. Our findings reveal that, in contrast to recent works that claimed empirical advantages of scalarization, scalarization is inherently incapable of full exploration, especially for those Pareto optimal solutions that strike the balanced trade-offs between multiple tasks. More concretely, when the model is under-parametrized, we reveal a multi-surface structure of the feasible region and identify necessary and sufficient conditions for full exploration. This leads to the conclusion that scalarization is in general incapable of tracing out the Pareto front. Our theoretical results partially answer the open questions in Xin et al. (2021), and provide a more intuitive explanation on why scalarization fails beyond non-convexity. We additionally perform experiments on a real-world dataset using both scalarization and state-of-the-art SMTOs. The experimental results not only corroborate our theoretical findings, but also unveil the potential of SMTOs in finding balanced solutions, which cannot be achieved by scalarization.

----

## [2106] Provable Guarantees for Generative Behavior Cloning: Bridging Low-Level Stability and High-Level Behavior

**Authors**: *Adam Block, Ali Jadbabaie, Daniel Pfrommer, Max Simchowitz, Russ Tedrake*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97c903fbf21a7d863af2015d8803ca8f-Abstract-Conference.html)

**Abstract**:

We propose a theoretical framework for studying behavior cloning of complex expert demonstrations using generative modeling.Our framework invokes low-level controllers - either learned or implicit in position-command control - to stabilize imitation around expert demonstrations. We show that with (a) a suitable low-level stability guarantee and (b) a powerful enough generative model as our imitation learner,  pure supervised behavior cloning can generate trajectories matching the per-time step distribution of essentially arbitrary expert trajectories in an optimal transport cost. Our analysis relies on a stochastic continuity property of the learned policy we call "total variation continuity" (TVC). We then show that TVC can be ensured with minimal degradation of accuracy by combining a popular data-augmentation regimen with a novel algorithmic trick: adding augmentation noise at execution time. We instantiate our guarantees for policies parameterized by diffusion models and prove that if the learner accurately estimates the score of the (noise-augmented) expert policy, then the distribution of imitator trajectories is close to the demonstrator distribution in a natural optimal transport distance. Our analysis constructs intricate couplings between noise-augmented trajectories, a technique that may be of independent interest. We conclude by empirically validating our algorithmic recommendations, and discussing implications for future research directions for better behavior cloning with generative modeling.

----

## [2107] Prefix-Tree Decoding for Predicting Mass Spectra from Molecules

**Authors**: *Samuel Goldman, John Bradshaw, Jiayi Xin, Connor W. Coley*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html)

**Abstract**:

Computational predictions of mass spectra from molecules have enabled the discovery of clinically relevant metabolites. However, such predictive tools are still limited as they occupy one of two extremes, either operating  (a) by fragmenting molecules combinatorially with overly rigid constraints on potential rearrangements and poor time complexity or (b) by decoding lossy and nonphysical discretized spectra vectors. In this work, we use a new intermediate strategy for predicting mass spectra from molecules by treating mass spectra as sets of molecular formulae, which are themselves multisets of atoms. After first encoding an input molecular graph, we decode a set of molecular subformulae, each of which specify a predicted peak in the mass spectrum, the intensities of which are predicted by a second model. Our key insight is to overcome the combinatorial possibilities for molecular subformulae by decoding the formula set using a prefix tree structure, atom-type by atom-type, representing a general method for ordered multiset decoding. We show promising empirical results on mass spectra prediction tasks.

----

## [2108] Knowledge-Augmented Reasoning Distillation for Small Language Models in Knowledge-Intensive Tasks

**Authors**: *Minki Kang, Seanie Lee, Jinheon Baek, Kenji Kawaguchi, Sung Ju Hwang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97faedc90260eae5c400f92d5831c3d7-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) have shown promising performance in knowledge-intensive reasoning tasks that require a compound understanding of knowledge. However, deployment of the LLMs in real-world applications can be challenging due to their high computational requirements and concerns on data privacy.Previous studies have focused on building task-specific small Language Models (LMs) by fine-tuning them with labeled data or distilling LLMs. However, these approaches are ill-suited for knowledge-intensive reasoning tasks due to the limited capacity of small LMs in memorizing the knowledge required.Motivated by our theoretical analysis on memorization, we propose Knowledge-Augmented Reasoning Distillation (KARD), a novel method that fine-tunes small LMs to generate rationales obtained from LLMs with augmented knowledge retrieved from an external knowledge base. Moreover, we further propose a neural reranker to obtain documents relevant to rationale generation. We empirically show that KARD significantly improves the performance of small T5 and GPT models on the challenging knowledge-intensive reasoning datasets, namely MedQA-USMLE, StrategyQA, and OpenbookQA.Notably, our method makes the 250M T5 models achieve superior performance against the fine-tuned 3B models, having 12 times larger parameters, on both MedQA-USMLE and StrategyQA benchmarks.

----

## [2109] Nonparametric Identifiability of Causal Representations from Unknown Interventions

**Authors**: *Julius von Kügelgen, Michel Besserve, Wendong Liang, Luigi Gresele, Armin Kekic, Elias Bareinboim, David M. Blei, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/97fe251c25b6f99a2a23b330a75b11d4-Abstract-Conference.html)

**Abstract**:

We study causal representation learning, the task of inferring latent causal variables and their causal relations from high-dimensional functions (“mixtures”) of the variables. Prior work relies on weak supervision, in the form of counterfactual pre- and post-intervention views or temporal structure; places restrictive assumptions, such as linearity, on the mixing function or latent causal model; or requires partial knowledge of the generative process, such as the causal graph or intervention targets. We instead consider the general setting in which both the causal model and the mixing function are nonparametric. The learning signal takes the form of multiple datasets, or environments, arising from unknown interventions in the underlying causal model. Our goal is to identify both the ground truth latents and their causal graph up to a set of ambiguities which we show to be irresolvable from interventional data. We study the fundamental setting of two causal variables and prove that the observational distribution and one perfect intervention per node suffice for identifiability, subject to a genericity condition. This condition rules out spurious solutions that involve fine-tuning of the intervened and observational distributions, mirroring similar conditions for nonlinear cause-effect inference. For an arbitrary number of variables, we show that at least one pair of distinct perfect interventional domains per node guarantees identifiability. Further, we demonstrate that the strengths of causal influences among the latent variables are preserved by all equivalent solutions, rendering the inferred representation appropriate for drawing causal conclusions from new data. Our study provides the first identifiability results for the general nonparametric setting with unknown interventions, and elucidates what is possible and impossible for causal representation learning without more direct supervision.

----

## [2110] Dual Mean-Teacher: An Unbiased Semi-Supervised Framework for Audio-Visual Source Localization

**Authors**: *Yuxin Guo, Shijie Ma, Hu Su, Zhiqing Wang, Yuhao Zhao, Wei Zou, Siyang Sun, Yun Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html)

**Abstract**:

Audio-Visual Source Localization (AVSL) aims to locate sounding objects within video frames given the paired audio clips. Existing methods predominantly rely on self-supervised contrastive learning of audio-visual correspondence. Without any bounding-box annotations, they struggle to achieve precise localization, especially for small objects, and suffer from blurry boundaries and false positives. Moreover, the naive semi-supervised method is poor in effectively utilizing the abundance of unlabeled audio-visual pairs. In this paper, we propose a novel Semi-Supervised Learning framework for AVSL, namely Dual Mean-Teacher (DMT), comprising two teacher-student structures to circumvent the confirmation bias issue. Specifically, two teachers, pre-trained on limited labeled data, are employed to filter out noisy samples via the consensus between their predictions, and then generate high-quality pseudo-labels by intersecting their confidence maps. The optimal utilization of both labeled and unlabeled data combined with this unbiased framework enable DMT to outperform current state-of-the-art methods by a large margin, with CIoU of $\textbf{90.4\%}$ and $\textbf{48.8\%}$ on Flickr-SoundNet and VGG-Sound Source, obtaining $\textbf{8.9\%}$ and $\textbf{9.6\%}$ improvements respectively, given only $3\%$ of data positional-annotated. We also extend our framework to some existing AVSL methods and consistently boost their performance. Our code is publicly available at https://github.com/gyx-gloria/DMT.

----

## [2111] Hierarchical Gaussian Mixture based Task Generative Model for Robust Meta-Learning

**Authors**: *Yizhou Zhang, Jingchao Ni, Wei Cheng, Zhengzhang Chen, Liang Tong, Haifeng Chen, Yan Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/982ca2640e64bf7a1908b028ebc8734a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/982ca2640e64bf7a1908b028ebc8734a-Abstract-Conference.html)

**Abstract**:

Meta-learning enables quick adaptation of machine learning models to new tasks with limited data. While tasks could come from varying distributions in reality, most of the existing meta-learning methods consider both training and testing tasks as from the same uni-component distribution, overlooking two critical needs of a practical solution: (1) the various sources of tasks may compose a multi-component mixture distribution, and (2) novel tasks may come from a distribution that is unseen during meta-training. In this paper, we demonstrate these two challenges can be solved jointly by modeling the density of task instances. We develop a meta-training framework underlain by a novel Hierarchical Gaussian Mixture based Task Generative Model (HTGM). HTGM extends the widely used empirical process of sampling tasks to a theoretical model, which learns task embeddings, fits the mixture distribution of tasks, and enables density-based scoring of novel tasks. The framework is agnostic to the encoder and scales well with large backbone networks. The model parameters are learned end-to-end by maximum likelihood estimation via an Expectation-Maximization (EM) algorithm. Extensive experiments on benchmark datasets indicate the effectiveness of our method for both sample classification and novel task detection.

----

## [2112] Temporal Dynamic Quantization for Diffusion Models

**Authors**: *Junhyuk So, Jungwon Lee, Daehyun Ahn, Hyungjun Kim, Eunhyeok Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/983591c3e9a0dc94a99134b3238bbe52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/983591c3e9a0dc94a99134b3238bbe52-Abstract-Conference.html)

**Abstract**:

Diffusion model has gained popularity in vision applications due to its remarkable generative performance and versatility. However, its high storage and computation demands, resulting from the model size and iterative generation, hinder its use on mobile devices. Existing quantization techniques struggle to maintain performance even in 8-bit precision due to the diffusion model's unique property of temporal variation in activation. We introduce a novel quantization method that dynamically adjusts the quantization interval based on time step information, significantly improving output quality. Unlike conventional dynamic quantization techniques, our approach has no computational overhead during inference and is compatible with both post-training quantization (PTQ) and quantization-aware training (QAT). Our extensive experiments demonstrate substantial improvements in output quality with the quantized model across various configurations.

----

## [2113] Learning Interpretable Low-dimensional Representation via Physical Symmetry

**Authors**: *Xuanjie Liu, Daniel Chin, Yichen Huang, Gus Xia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9850e6a5410331290dc1deefb7514448-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9850e6a5410331290dc1deefb7514448-Abstract-Conference.html)

**Abstract**:

We have recently seen great progress in learning interpretable music representations, ranging from basic factors, such as pitch and timbre, to high-level concepts, such as chord and texture. However, most methods rely heavily on music domain knowledge. It remains an open question what general computational principles give rise to interpretable representations, especially low-dim factors that agree with human perception. In this study, we take inspiration from modern physics and use physical symmetry as a self-consistency constraint for the latent space. Specifically, it requires the prior model that characterises the dynamics of the latent states to be equivariant with respect to certain group transformations. We show that physical symmetry leads the model to learn a linear pitch factor from unlabelled monophonic music audio in a self-supervised fashion. In addition, the same methodology can be applied to computer vision, learning a 3D Cartesian space from videos of a simple moving object without labels. Furthermore, physical symmetry naturally leads to counterfactual representation augmentation, a new technique which improves sample efficiency.

----

## [2114] ImageBrush: Learning Visual In-Context Instructions for Exemplar-Based Image Manipulation

**Authors**: *Yasheng Sun, Yifan Yang, Houwen Peng, Yifei Shen, Yuqing Yang, Han Hu, Lili Qiu, Hideki Koike*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98530736e5d94e62b689dfc1fda89bd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98530736e5d94e62b689dfc1fda89bd1-Abstract-Conference.html)

**Abstract**:

While language-guided image manipulation has made remarkable progress, the challenge of how to instruct the manipulation process faithfully reflecting human intentions persists. An accurate and comprehensive description of a manipulation task using natural language is laborious and sometimes even impossible, primarily due to the inherent uncertainty and ambiguity present in linguistic expressions. Is it feasible to accomplish image manipulation without resorting to external cross-modal language information? If this possibility exists, the inherent modality gap would be effortlessly eliminated. In this paper, we propose a novel  manipulation methodology, dubbed ImageBrush, that learns visual instructions for more accurate image editing.Our key idea is to employ a pair of transformation images as visual instructions, which not only precisely captures human intention but also facilitates accessibility in real-world scenarios. Capturing visual instructions is particularly challenging because it involves extracting the underlying intentions solely from visual demonstrations and then applying this operation to a new image. To address this challenge, we formulate visual instruction learning as a diffusion-based inpainting problem, where the contextual information is fully exploited through an iterative process of generation. A visual prompting encoder is carefully devised to enhance the model's capacity in uncovering human intent behind the visual instructions. Extensive experiments show that our method generates engaging manipulation results conforming to the transformations entailed in demonstrations. Moreover, our model exhibits robust generalization capabilities on various downstream tasks such as pose transfer, image translation and video inpainting.

----

## [2115] Meek Separators and Their Applications in Targeted Causal Discovery

**Authors**: *Kirankumar Shiragur, Jiaqi Zhang, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/985786d06c1e45e9e8c65f7aca3547e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/985786d06c1e45e9e8c65f7aca3547e4-Abstract-Conference.html)

**Abstract**:

Learning causal structures from interventional data is a fundamental problem with broad applications across various fields. While many previous works have focused on recovering the entire causal graph, in practice, there are scenarios where learning only part of the causal graph suffices. This is called \emph{targeted} causal discovery. In our work, we focus on two such well-motivated problems: subset search and causal matching. We aim to minimize the number of interventions in both cases.Towards this, we introduce the \emph{Meek separator}, which is a subset of vertices that, when intervened, decomposes the remaining unoriented edges into smaller connected components. We then present an efficient algorithm to find Meek separators that are of small sizes. Such a procedure is helpful in designing various divide-and-conquer-based approaches. In particular, we propose two randomized algorithms that achieve logarithmic approximation for subset search and causal matching, respectively. Our results provide the first known average-case provable guarantees for both problems. We believe that this opens up possibilities to design near-optimal methods for many other targeted causal structure learning problems arising from various applications.

----

## [2116] CLeAR: Continual Learning on Algorithmic Reasoning for Human-like Intelligence

**Authors**: *Bong Gyun Kang, HyunGi Kim, Dahuin Jung, Sungroh Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/986e0caad271b59417287737416d8594-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/986e0caad271b59417287737416d8594-Abstract-Conference.html)

**Abstract**:

Continual learning (CL) aims to incrementally learn multiple tasks that are presented sequentially. The significance of CL lies not only in the practical importance but also in studying the learning mechanisms of humans who are excellent continual learners. While most research on CL has been done on structured data such as images, there is a lack of research on CL for abstract logical concepts such as counting, sorting, and arithmetic, which humans learn gradually over time in the real world. In this work, for the first time, we introduce novel algorithmic reasoning (AR) methodology for continual tasks of abstract concepts: CLeAR. Our methodology proposes a one-to-many mapping of input distribution to a shared mapping space, which allows the alignment of various tasks of different dimensions and shared semantics. Our tasks of abstract logical concepts, in the form of formal language, can be classified into Chomsky hierarchies based on their difficulty. In this study, we conducted extensive experiments consisting of 15 tasks with various levels of Chomsky hierarchy, ranging from in-hierarchy to inter-hierarchy scenarios. CLeAR not only achieved near zero forgetting but also improved accuracy during following tasks, a phenomenon known as backward transfer, while previous CL methods designed for image classification drastically failed.

----

## [2117] SLIBO-Net: Floorplan Reconstruction via Slicing Box Representation with Local Geometry Regularization

**Authors**: *Jheng-Wei Su, Kuei-Yu Tung, Chi-Han Peng, Peter Wonka, Hung-Kuo Chu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html)

**Abstract**:

This paper focuses on improving the reconstruction of 2D floorplans from unstructured 3D point clouds. We identify opportunities for enhancement over the existing methods in three main areas: semantic quality, efficient representation, and local geometric details. To address these, we presents SLIBO-Net, an innovative approach to reconstructing 2D floorplans from unstructured 3D point clouds. We propose a novel transformer-based architecture that employs an efficient floorplan representation, providing improved room shape supervision and allowing for manageable token numbers. By incorporating geometric priors as a regularization mechanism and post-processing step, we enhance the capture of local geometric details. We also propose a scale-independent evaluation metric, correcting the discrepancy in error treatment between varying floorplan sizes. Our approach notably achieves a new state-of-the-art on the Structure3D dataset. The resultant floorplans exhibit enhanced semantic plausibility, substantially improving the overall quality and realism of the reconstructions. Our code and dataset are available online.

----

## [2118] Fantastic Robustness Measures: The Secrets of Robust Generalization

**Authors**: *Hoki Kim, Jinseong Park, Yujin Choi, Jaewook Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html)

**Abstract**:

Adversarial training has become the de-facto standard method for improving the robustness of models against adversarial examples. However, robust overfitting remains a significant challenge, leading to a large gap between the robustness on the training and test datasets. To understand and improve robust generalization, various measures have been developed, including margin, smoothness, and flatness-based measures. In this study, we present a large-scale analysis of robust generalization to empirically verify whether the relationship between these measures and robust generalization remains valid in diverse settings. We demonstrate when and how these measures effectively capture the robust generalization gap by comparing over 1,300 models trained on CIFAR-10 under the $L_\infty$ norm and further validate our findings through an evaluation of more than 100 models from RobustBench across CIFAR-10, CIFAR-100, and ImageNet. We hope this work can help the community better understand adversarial robustness and motivate the development of more robust defense methods against adversarial attacks.

----

## [2119] A Spectral Algorithm for List-Decodable Covariance Estimation in Relative Frobenius Norm

**Authors**: *Ilias Diakonikolas, Daniel Kane, Jasper C. H. Lee, Ankit Pensia, Thanasis Pittas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html)

**Abstract**:

We study the problem of list-decodable Gaussian covariance estimation. Given a multiset $T$ of $n$ points in $\mathbb{R}^d$ such that an unknown $\alpha<1/2$ fraction of points in $T$ are i.i.d. samples from an unknown Gaussian $\mathcal{N}(\mu, \Sigma)$, the goal is to output a list of $O(1/\alpha)$ hypotheses at least one of which is close to $\Sigma$ in relative Frobenius norm. Our main result is a $\mathrm{poly}(d,1/\alpha)$ sample and time algorithm for this task that guarantees relative Frobenius norm error of $\mathrm{poly}(1/\alpha)$. Importantly, our algorithm relies purely on spectral techniques. As a corollary, we obtain an efficient spectral algorithm for robust partial clustering of Gaussian mixture models (GMMs) --- a key ingredient in the recent work of [BakDJKKV22] on robustly learning arbitrary GMMs. Combined with the other components of [BakDJKKV22], our new method yields the first Sum-of-Squares-free algorithm for robustly learning GMMs, resolving an open problem proposed by Vempala and Kothari. At the technical level, we develop a novel multi-filtering method for list-decodable covariance estimation that may be useful in other settings.

----

## [2120] Diff-Foley: Synchronized Video-to-Audio Synthesis with Latent Diffusion Models

**Authors**: *Simian Luo, Chuanhao Yan, Chenxu Hu, Hang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98c50f47a37f63477c01558600dd225a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98c50f47a37f63477c01558600dd225a-Abstract-Conference.html)

**Abstract**:

The Video-to-Audio (V2A) model has recently gained attention for its practical application in generating audio directly from silent videos, particularly in video/film production. However, previous methods in V2A have limited generation quality in terms of temporal synchronization and audio-visual relevance. We present Diff-Foley, a synchronized Video-to-Audio synthesis method with a latent diffusion model (LDM) that generates high-quality audio with improved synchronization and audio-visual relevance. We adopt contrastive audio-visual pretraining (CAVP) to learn more temporally and semantically aligned features, then train an LDM with CAVP-aligned visual features on spectrogram latent space. The CAVP-aligned features enable LDM to capture the subtler audio-visual correlation via a cross-attention module. We further significantly improve sample quality with `double guidance'. Diff-Foley achieves state-of-the-art V2A performance on current large scale V2A dataset. Furthermore, we demonstrate Diff-Foley practical applicability and adaptability via customized downstream finetuning. Project Page: https://diff-foley.github.io/

----

## [2121] The Drunkard's Odometry: Estimating Camera Motion in Deforming Scenes

**Authors**: *David Recasens, Martin R. Oswald, Marc Pollefeys, Javier Civera*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98c9b79e9c686aadd4d81e34a7773dd1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/98c9b79e9c686aadd4d81e34a7773dd1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Estimating camera motion in deformable scenes poses a complex and open research challenge. Most existing non-rigid structure from motion techniques assume to observe also static scene parts besides deforming scene parts in order to establish an anchoring reference. However, this assumption does not hold true in certain relevant application cases such as endoscopies. Deformable odometry and SLAM pipelines, which tackle the most challenging scenario of exploratory trajectories, suffer from a lack of robustness and proper quantitative evaluation methodologies. To tackle this issue with a common benchmark, we introduce the Drunkard's Dataset, a challenging collection of synthetic data targeting visual navigation and reconstruction in deformable environments. This dataset is the first large set of exploratory camera trajectories with ground truth inside 3D scenes where every surface exhibits non-rigid deformations over time. Simulations in realistic 3D buildings lets us obtain a vast amount of data and ground truth labels, including camera poses, RGB images and depth, optical flow and normal maps at high resolution and quality. We further present a novel deformable odometry method, dubbed the Drunkard’s Odometry, which decomposes optical flow estimates into rigid-body camera motion and non-rigid scene deformations. In order to validate our data, our work contains an evaluation of several baselines as well as a novel tracking error metric which does not require ground truth data. Dataset and code: https://davidrecasens.github.io/TheDrunkard'sOdometry/

----

## [2122] Optimal Treatment Allocation for Efficient Policy Evaluation in Sequential Decision Making

**Authors**: *Ting Li, Chengchun Shi, Jianing Wang, Fan Zhou, Hongtu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98d0ad88db1e51bd0aa341a823290ece-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98d0ad88db1e51bd0aa341a823290ece-Abstract-Conference.html)

**Abstract**:

A/B testing is critical for modern technological companies to evaluate the effectiveness of newly developed products against standard baselines. This paper studies optimal designs that aim to maximize the amount of information obtained from online experiments to estimate treatment effects accurately. We propose three optimal allocation strategies in a dynamic setting where treatments are sequentially assigned over time. These strategies are designed to minimize the variance of the treatment effect estimator when data follow a non Markov decision process or a (time-varying) Markov decision process. We further develop estimation procedures based on existing off-policy evaluation (OPE) methods and conduct extensive experiments in various environments to demonstrate the effectiveness of the proposed methodologies. In theory, we prove the optimality of the proposed treatment allocation design and establish upper bounds for the mean squared errors of the resulting treatment effect estimators.

----

## [2123] Advancing Bayesian Optimization via Learning Correlated Latent Space

**Authors**: *Seunghun Lee, Jaewon Chu, Sihyeon Kim, Juyeon Ko, Hyunwoo J. Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98e967164ae2f6811b975d686dece3eb-Abstract-Conference.html)

**Abstract**:

Bayesian optimization is a powerful method for optimizing black-box functions with limited function evaluations. Recent works have shown that optimization in a latent space through deep generative models such as variational autoencoders leads to effective and efficient Bayesian optimization for structured or discrete data. However, as the optimization does not take place in the input space, it leads to an inherent gap that results in potentially suboptimal solutions. To alleviate the discrepancy, we propose Correlated latent space Bayesian Optimization (CoBO), which focuses on learning correlated latent spaces characterized by a strong correlation between the distances in the latent space and the distances within the objective function. Specifically, our method introduces Lipschitz regularization, loss weighting, and trust region recoordination to minimize the inherent gap around the promising areas. We demonstrate the effectiveness of our approach on several optimization tasks in discrete data, such as molecule design and arithmetic expression fitting, and achieve high performance within a small budget.

----

## [2124] Generalization bounds for neural ordinary differential equations and deep residual networks

**Authors**: *Pierre Marion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98ed250b203d1ac6b24bbcf263e3d4a7-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equations (neural ODEs) are a popular family of continuous-depth deep learning models.  In this work, we consider a large family of parameterized ODEs with continuous-in-time parameters, which include time-dependent neural ODEs.  We derive a generalization bound for this class by a Lipschitz-based argument. By leveraging the analogy between neural ODEs and deep residual networks, our approach yields in particular a generalization bound for a class of deep residual networks. The bound involves the magnitude of the difference between successive weight matrices. We illustrate numerically how this quantity affects the generalization capability of neural networks.

----

## [2125] Global Update Tracking: A Decentralized Learning Algorithm for Heterogeneous Data

**Authors**: *Sai Aparna Aketi, Abolfazl Hashemi, Kaushik Roy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html)

**Abstract**:

Decentralized learning enables the training of deep learning models over large distributed datasets generated at different locations, without the need for a central server. However, in practical scenarios, the data distribution across these devices can be significantly different, leading to a degradation in model performance. In this paper, we focus on designing a decentralized learning algorithm that is less susceptible to variations in data distribution across devices. We propose Global Update Tracking (GUT), a novel tracking-based method that aims to mitigate the impact of heterogeneous data in decentralized learning without introducing any communication overhead. We demonstrate the effectiveness of the proposed technique through an exhaustive set of experiments on various Computer Vision datasets (CIFAR-10, CIFAR-100, Fashion MNIST, and ImageNette), model architectures, and network topologies. Our experiments show that the proposed method achieves state-of-the-art performance for decentralized learning on heterogeneous data via a 1-6% improvement in test accuracy compared to other existing techniques.

----

## [2126] QuadAttacK: A Quadratic Programming Approach to Learning Ordered Top-K Adversarial Attacks

**Authors**: *Thomas Paniagua, Ryan Grainger, Tianfu Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9902a53031ebbbab73898028073d4790-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9902a53031ebbbab73898028073d4790-Abstract-Conference.html)

**Abstract**:

The adversarial vulnerability of Deep Neural Networks (DNNs) has been well-known and widely concerned, often under the context of learning top-$1$ attacks (e.g., fooling a DNN to classify a cat image as dog). This paper shows that the concern is much more serious by learning significantly more aggressive ordered top-$K$ clear-box targeted attacks proposed in~\citep{zhang2020learning}. We propose a novel and rigorous quadratic programming (QP) method of learning ordered top-$K$ attacks with low computing cost, dubbed as \textbf{QuadAttac$K$}. Our QuadAttac$K$ directly solves the QP to satisfy the attack constraint in the feature embedding space (i.e., the input space to the final linear classifier), which thus exploits the semantics of the feature embedding space (i.e., the principle of class coherence). With the optimized feature embedding  vector perturbation, it then computes the adversarial perturbation in the data space via the vanilla one-step back-propagation. In experiments, the proposed QuadAttac$K$ is tested in the ImageNet-1k  classification using ResNet-50, DenseNet-121, and Vision Transformers (ViT-B and DEiT-S). It successfully pushes the boundary of successful ordered top-$K$ attacks from $K=10$ up to $K=20$ at a cheap budget ($1\times 60$) and further improves attack success rates for $K=5$ for all tested models, while retaining the performance for $K=1$.

----

## [2127] Predicting mutational effects on protein-protein binding via a side-chain diffusion probabilistic model

**Authors**: *Shiwei Liu, Tian Zhu, Milong Ren, Chungong Yu, Dongbo Bu, Haicang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html)

**Abstract**:

Many crucial biological processes rely on networks of protein-protein interactions. Predicting the effect of amino acid mutations on protein-protein binding is important in protein engineering, including therapeutic discovery. However, the scarcity of annotated experimental data on binding energy poses a significant challenge for developing computational approaches, particularly deep learning-based methods. In this work, we propose SidechainDiff, a novel representation learning-based approach that leverages unlabelled experimental protein structures. SidechainDiff utilizes a Riemannian diffusion model to learn the generative process of side-chain conformations and can also give the structural context representations of mutations on the protein-protein interface. Leveraging the learned representations, we achieve state-of-the-art performance in predicting the mutational effects on protein-protein binding. Furthermore, SidechainDiff is the first diffusion-based generative model for side-chains, distinguishing it from prior efforts that have predominantly focused on the generation of protein backbone structures.

----

## [2128] PETAL: Physics Emulation Through Averaged Linearizations for Solving Inverse Problems

**Authors**: *Jihui Jin, Etienne Ollivier, Richard Touret, Matthew McKinley, Karim Sabra, Justin Romberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/991c9324ca71aa85ab4dd11146b35fc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/991c9324ca71aa85ab4dd11146b35fc3-Abstract-Conference.html)

**Abstract**:

Inverse problems describe the task of recovering an underlying signal of interest given observables. Typically, the observables are related via some non-linear forward model applied to the underlying unknown signal. Inverting the non-linear forward model can be computationally expensive, as it often involves computing and inverting a linearization at a series of estimates. Rather than inverting the physics-based model, we instead train a surrogate forward model (emulator) and leverage modern auto-grad libraries to solve for the input within a classical optimization framework. Current methods to train emulators are done in a black box supervised machine learning fashion and fail to take advantage of any existing knowledge of the forward model. In this article, we propose a simple learned weighted average model that embeds linearizations of the forward model around various reference points into the model itself, explicitly incorporating known physics. Grounding the learned model with physics based linearizations improves the forward modeling accuracy and provides richer physics based gradient information during the inversion process leading to more accurate signal recovery. We demonstrate the efficacy on an ocean acoustic tomography (OAT) example that aims to recover ocean sound speed profile (SSP) variations from acoustic observations (e.g. eigenray arrival times) within simulation of ocean dynamics in the Gulf of Mexico.

----

## [2129] RealTime QA: What's the Answer Right Now?

**Authors**: *Jungo Kasai, Keisuke Sakaguchi, Yoichi Takahashi, Ronan Le Bras, Akari Asai, Xinyan Yu, Dragomir Radev, Noah A. Smith, Yejin Choi, Kentaro Inui*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9941624ef7f867a502732b5154d30cb7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9941624ef7f867a502732b5154d30cb7-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce RealTime QA, a dynamic question answering (QA) platform that announces questions and evaluates systems on a regular basis (weekly in this version). RealTime QA inquires about the current world, and QA systems need to answer questions about novel events or information. It therefore challenges static, conventional assumptions in open-domain QA datasets and pursues instantaneous applications. We build strong baseline models upon large pretrained language models, including GPT-3 and T5. Our benchmark is an ongoing effort, and this paper presents real-time evaluation results over the past year. Our experimental results show that GPT-3 can often properly update its generation results, based on newly-retrieved documents, highlighting the importance of up-to-date information retrieval. Nonetheless, we find that GPT-3 tends to return outdated answers when retrieved documents do not provide sufficient information to find an answer. This suggests an important avenue for future research: can an open-domain QA system identify such unanswerable cases and communicate with the user or even the retrieval module to modify the retrieval results? We hope that RealTime QA will spur progress in instantaneous applications of question answering and beyond.

----

## [2130] Learning Transformer Programs

**Authors**: *Dan Friedman, Alexander Wettig, Danqi Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract-Conference.html)

**Abstract**:

Recent research in mechanistic interpretability has attempted to reverse-engineer Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of providing complete, faithful descriptions of the underlying algorithms. In this work, we introduce a procedure for training Transformers that are mechanistically interpretable by design. We build on RASP [Weiss et al., 2021], a programming language that can be compiled into Transformer weights. Instead of compiling human-written programs into Transformers, we design a modified Transformer that can be trained using gradient-based optimization and then automatically converted into a discrete, human-readable program. We refer to these models as Transformer Programs. To validate our approach, we learn Transformer Programs for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck languages), and NLP tasks including named entity recognition and text classification. The Transformer Programs can automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To demonstrate these advantages, we convert Transformers into Python programs and use off-the-shelf code analysis tools to debug model errors and identify the “circuits” used to solve different sub-problems. We hope that Transformer Programs open a new path toward the goal of intrinsically interpretable machine learning.

----

## [2131] An Inverse Scaling Law for CLIP Training

**Authors**: *Xianhang Li, Zeyu Wang, Cihang Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html)

**Abstract**:

CLIP, one of the pioneering foundation models that connect images and text, has enabled many recent breakthroughs in computer vision. However, its associated training cost is prohibitively high, imposing a significant barrier to its widespread exploration. In this paper, we present a surprising finding that there exists an inverse scaling law for CLIP training, whereby the larger the image/text encoders used, the shorter the sequence length of image/text tokens that can be applied in training. Moreover, we showcase that the strategy for reducing image/text token length plays a crucial role in determining the quality of this scaling law.As a result of this finding, we are able to successfully train CLIP even with limited computational resources. For example, using 8 A100 GPUs, our CLIP models achieve zero-shot top-1 ImageNet-1k accuracies of 63.2% in ~2 days, 67.8% in ~3 days, and 69.3% in ~4 days. Our method also works well when scaling up --- with G/14, we register a new record of 83.0% ImageNet-1k zero-shot accuracy, and meanwhile accelerate the training by ~33x compared to its OpenCLIP counterpart.By reducing the computation barrier associated with CLIP, we hope to inspire more research in this field, particularly from academics. Our code is available at https://github.com/UCSC-VLAA/CLIPA.

----

## [2132] Sequential Preference Ranking for Efficient Reinforcement Learning from Human Feedback

**Authors**: *Minyoung Hwang, Gunmin Lee, Hogun Kee, Chan Woo Kim, Kyungjae Lee, Songhwai Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99766cda865be123d55a1d9666c7b9fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99766cda865be123d55a1d9666c7b9fc-Abstract-Conference.html)

**Abstract**:

Reinforcement learning from human feedback (RLHF) alleviates the problem of designing a task-specific reward function in reinforcement learning by learning it from human preference. However, existing RLHF models are considered inefficient as they produce only a single preference data from each human feedback. To tackle this problem, we propose a novel RLHF framework called SeqRank, that uses sequential preference ranking to enhance the feedback efficiency. Our method samples trajectories in a sequential manner by iteratively selecting a defender from the set of previously chosen trajectories $\mathcal{K}$ and a challenger from the set of unchosen trajectories $\mathcal{U}\setminus\mathcal{K}$, where $\mathcal{U}$ is the replay buffer. We propose two trajectory comparison methods with different defender sampling strategies: (1) sequential pairwise comparison that selects the most recent trajectory and (2) root pairwise comparison that selects the most preferred trajectory from $\mathcal{K}$. We construct a data structure and rank trajectories by preference to augment additional queries. The proposed method results in at least 39.2% higher average feedback efficiency than the baseline and also achieves a balance between feedback efficiency and data dependency. We examine the convergence of the empirical risk and the generalization bound of the reward model with Rademacher complexity. While both trajectory comparison methods outperform conventional pairwise comparison, root pairwise comparison improves the average reward in locomotion tasks and the average success rate in manipulation tasks by 29.0% and 25.0%, respectively. The source code and the videos are provided in the supplementary material.

----

## [2133] Diffusion-SS3D: Diffusion Model for Semi-supervised 3D Object Detection

**Authors**: *Cheng-Ju Ho, Chen-Hsuan Tai, Yen-Yu Lin, Ming-Hsuan Yang, Yi-Hsuan Tsai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99786eed5e16920f908572fb00e151c3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99786eed5e16920f908572fb00e151c3-Abstract-Conference.html)

**Abstract**:

Semi-supervised object detection is crucial for 3D scene understanding, efficiently addressing the limitation of acquiring large-scale 3D bounding box annotations. Existing methods typically employ a teacher-student framework with pseudo-labeling to leverage unlabeled point clouds. However, producing reliable pseudo-labels in a diverse 3D space still remains challenging. In this work, we propose Diffusion-SS3D, a new perspective of enhancing the quality of pseudo-labels via the diffusion model for semi-supervised 3D object detection. Specifically, we include noises to produce corrupted 3D object size and class label distributions, and then utilize the diffusion model as a denoising process to obtain bounding box outputs. Moreover, we integrate the diffusion model into the teacher-student framework, so that the denoised bounding boxes can be used to improve pseudo-label generation, as well as the entire semi-supervised learning process. We conduct experiments on the ScanNet and SUN RGB-D benchmark datasets to demonstrate that our approach achieves state-of-the-art performance against existing methods. We also present extensive analysis to understand how our diffusion model design affects performance in semi-supervised learning. The source code will be available at https://github.com/luluho1208/Diffusion-SS3D.

----

## [2134] Aligning Language Models with Human Preferences via a Bayesian Approach

**Authors**: *Jiashuo Wang, Haozhao Wang, Shichao Sun, Wenjie Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html)

**Abstract**:

In the quest to advance human-centric natural language generation (NLG) systems, ensuring alignment between NLG models and human preferences is crucial. For this alignment, current popular methods leverage a reinforcement learning (RL) approach with a reward model trained on feedback from humans. However, inherent disagreements due to the subjective nature of human preferences pose a significant challenge for training the reward model, resulting in a deterioration of the NLG performance. To tackle this issue, previous approaches typically rely on majority voting or averaging to consolidate multiple inconsistent preferences into a merged one. Although straightforward to understand and execute, such methods suffer from an inability to capture the nuanced degrees of disaggregation among humans and may only represent a specialized subset of individuals, thereby lacking the ability to quantitatively disclose the universality of human preferences. To address this challenge, this paper proposes a novel approach, which employs a Bayesian framework to account for the distribution of disagreements among human preferences as training a preference model, and names it as $\textbf{d-PM}$. Besides, considering the RL strategy's inefficient and complex training process over the training efficiency, we further propose utilizing the contrastive learning strategy to train the NLG model with the preference scores derived from the d-PM model. Extensive experiments on two human-centric NLG tasks, i.e., emotional support conversation and integrity ``Rule-of-Thumb'' generation, show that our method consistently exceeds previous SOTA models in both automatic and human evaluations.

----

## [2135] A Smooth Binary Mechanism for Efficient Private Continual Observation

**Authors**: *Joel Daniel Andersson, Rasmus Pagh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99c41fb9fd53abfdd4a0259560ef1c9d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99c41fb9fd53abfdd4a0259560ef1c9d-Abstract-Conference.html)

**Abstract**:

In privacy under continual observation we study how to release differentially private estimates based on a dataset that evolves over time. The problem of releasing private prefix sums of $x_1, x_2, x_3,\dots\in${$0,1$} (where the value of each $x_i$ is to be private) is particularly well-studied, and a generalized form is used in state-of-the-art methods for private stochastic gradient descent (SGD).The seminal binary mechanism privately releases the first $t$ prefix sums with noise of variance polylogarithmic in $t$. Recently, Henzinger et al. and Denisov et al. showed that it is possible to improve on the binary mechanism in two ways: The variance of the noise can be reduced by a (large) constant factor, and also made more even across time steps. However, their algorithms for generating the noise distribution are not as efficient as one would like in terms of computation time and (in particular) space.We address the efficiency problem by presenting a simple alternative to the binary mechanism in which 1) generating the noise takes constant average time per value, 2) the variance is reduced by a factor about 4 compared to the binary mechanism, and 3) the noise distribution at each step is identical. Empirically, a simple Python implementation of our approach outperforms the running time of the approach of Henzinger et al., as well as an attempt to improve their algorithm using high-performance algorithms for multiplication with Toeplitz matrices.

----

## [2136] Training Transformers with 4-bit Integers

**Authors**: *Haocheng Xi, Changhao Li, Jianfei Chen, Jun Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/99fc8bc48b917c301a80cb74d91c0c06-Abstract-Conference.html)

**Abstract**:

Quantizing the activation, weight, and gradient to 4-bit is promising to accelerate neural network training. However, existing 4-bit training methods require custom numerical formats  which are not supported by contemporary hardware. In this work, we propose a training method for transformers with all matrix multiplications implemented with the INT4 arithmetic. Training with an ultra-low INT4 precision is challenging. To achieve this, we carefully analyze the specific structures of activation and gradients in transformers to propose dedicated quantizers for them. For forward propagation, we identify the challenge of outliers and propose a Hadamard quantizer to suppress the outliers. For backpropagation, we leverage the structural sparsity of gradients by proposing bit splitting and leverage score sampling techniques to quantize gradients accurately. Our algorithm achieves competitive accuracy on a wide range of tasks including natural language understanding, machine translation, and image classification. Unlike previous 4-bit training methods, our algorithm can be implemented on the current generation of GPUs. Our prototypical linear operator implementation is up to 2.2 times faster than the FP16 counterparts and speeds up the training by 17.8\% on average for sufficiently large models. Our code is available at https://github.com/xijiu9/Train_Transformers_with_INT4.

----

## [2137] TD Convergence: An Optimization Perspective

**Authors**: *Kavosh Asadi, Shoham Sabach, Yao Liu, Omer Gottesman, Rasool Fakoor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a08fbb992f15faa695c42b6a2c8e000-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a08fbb992f15faa695c42b6a2c8e000-Abstract-Conference.html)

**Abstract**:

We study the convergence behavior of the celebrated temporal-difference (TD) learning algorithm. By looking at the algorithm through the lens of optimization, we first argue that TD can be viewed as an iterative optimization algorithm where the function to be minimized changes per iteration. By carefully investigating the divergence displayed by TD on a classical counter example, we identify two forces that determine the convergent or divergent behavior of the algorithm. We next formalize our discovery in the linear TD setting with quadratic loss and prove that convergence of TD hinges on the interplay between these two forces. We extend this optimization perspective to prove convergence of TD in a much broader setting than just linear approximation and squared loss. Our results provide a theoretical explanation for the successful application of TD in reinforcement learning.

----

## [2138] Time Series as Images: Vision Transformer for Irregularly Sampled Time Series

**Authors**: *Zekun Li, Shiyang Li, Xifeng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a17c1eb808cf012065e9db47b7ca80d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a17c1eb808cf012065e9db47b7ca80d-Abstract-Conference.html)

**Abstract**:

Irregularly sampled time series are increasingly prevalent, particularly in medical domains. While various specialized methods have been developed to handle these irregularities, effectively modeling their complex dynamics and pronounced sparsity remains a challenge. This paper introduces a novel perspective by converting irregularly sampled time series into line graph images, then utilizing powerful pre-trained vision transformers for time series classification in the same way as image classification. This method not only largely simplifies specialized algorithm designs but also presents the potential to serve as a universal framework for time series modeling. Remarkably, despite its simplicity, our approach outperforms state-of-the-art specialized algorithms on several popular healthcare and human activity datasets. Especially in the rigorous leave-sensors-out setting where a portion of variables is omitted during testing, our method exhibits strong robustness against varying degrees of missing observations, achieving an impressive improvement of 42.8% in absolute F1 score points over leading specialized baselines even with half the variables masked. Code and data are available at https://github.com/Leezekun/ViTST.

----

## [2139] Symbolic Discovery of Optimization Algorithms

**Authors**: *Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a39b4925e35cf447ccba8757137d84f-Abstract-Conference.html)

**Abstract**:

We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training. We leverage efficient search techniques to explore an infinite and sparse program space. To bridge the large generalization gap between proxy and target tasks, we also introduce program selection and simplification strategies.Our method discovers a simple and effective optimization algorithm, $\textbf{Lion}$ ($\textit{Evo$\textbf{L}$ved S$\textbf{i}$gn M$\textbf{o}$me$\textbf{n}$tum}$). It is more memory-efficient than Adam as it only keeps track of the momentum. Different from adaptive optimizers, its update has the same magnitude for each parameter calculated through the sign operation.We compare Lion with widely used optimizers, such as Adam and Adafactor, for training a variety of models on different tasks. On image classification, Lion boosts the accuracy of ViT by up to 2\% on ImageNet and saves up to 5x the pre-training compute on JFT. On vision-language contrastive learning, we achieve 88.3\% $\textit{zero-shot}$ and 91.1\% $\textit{fine-tuning}$ accuracy on ImageNet, surpassing the previous best results by 2\% and 0.1\%, respectively. On diffusion models, Lion outperforms Adam by achieving a better FID score and reducing the training compute by up to 2.3x. For autoregressive, masked language modeling, and fine-tuning, Lion exhibits a similar or better performance compared to Adam. Our analysis of Lion reveals that its performance gain grows with the training batch size. It also requires a smaller learning rate than Adam due to the larger norm of the update produced by the sign function. Additionally, we examine the limitations of Lion and identify scenarios where its improvements are small or not statistically significant.

----

## [2140] On Calibrating Diffusion Probabilistic Models

**Authors**: *Tianyu Pang, Cheng Lu, Chao Du, Min Lin, Shuicheng Yan, Zhijie Deng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a645c38d4ec6f94633a35aeb2079596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a645c38d4ec6f94633a35aeb2079596-Abstract-Conference.html)

**Abstract**:

Recently, diffusion probabilistic models (DPMs) have achieved promising results in diverse generative tasks. A typical DPM framework includes a forward process that gradually diffuses the data distribution and a reverse process that recovers the data distribution from time-dependent data scores. In this work, we observe that the stochastic reverse process of data scores is a martingale, from which concentration bounds and the optional stopping theorem for data scores can be derived. Then, we discover a simple way for calibrating an arbitrary pretrained DPM, with which the score matching loss can be reduced and the lower bounds of model likelihood can consequently be increased. We provide general calibration guidelines under various model parametrizations. Our calibration method is performed only once and the resulting models can be used repeatedly for sampling. We conduct experiments on multiple datasets to empirically validate our proposal. Our code is available at https://github.com/thudzj/Calibrated-DPMs.

----

## [2141] InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

**Authors**: *Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, Steven C. H. Hoi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a6a435e75419a836fe47ab6793623e6-Abstract-Conference.html)

**Abstract**:

Large-scale pre-training and instruction tuning have been successful at creating general-purpose language models with broad competence. However, building general-purpose vision-language models is challenging due to the rich input distributions and task diversity resulting from the additional visual input. Although vision-language pretraining has been widely studied, vision-language instruction tuning remains under-explored. In this paper, we conduct a systematic and comprehensive study on vision-language instruction tuning based on the pretrained BLIP-2 models. We gather 26 publicly available datasets, covering a wide variety of tasks and capabilities, and transform them into instruction tuning format. Additionally, we introduce an instruction-aware Query Transformer, which extracts informative features tailored to the given instruction. Trained on 13 held-in datasets, InstructBLIP attains state-of-the-art zero-shot performance across all 13 held-out datasets, substantially outperforming BLIP-2 and larger Flamingo models. Our models also lead to state-of-the-art performance when finetuned on individual downstream tasks (e.g., 90.7% accuracy on ScienceQA questions with image contexts). Furthermore, we qualitatively demonstrate the advantages of InstructBLIP over concurrent multimodal models. All InstructBLIP models are open-source.

----

## [2142] Privacy Auditing with One (1) Training Run

**Authors**: *Thomas Steinke, Milad Nasr, Matthew Jagielski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a6f6e0d6781d1cb8689192408946d73-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a6f6e0d6781d1cb8689192408946d73-Abstract-Conference.html)

**Abstract**:

We propose a scheme for auditing differentially private machine learning systems with a single training run. This exploits the parallelism of being able to add or remove multiple training examples independently. We analyze this using the connection between differential privacy and statistical generalization, which avoids the cost of group privacy. Our auditing scheme requires minimal assumptions about the algorithm and can be applied in the black-box or white-box setting. We demonstrate the effectiveness of our framework by applying it to DP-SGD, where we can achieve meaningful empirical privacy lower bounds by training only one model. In contrast, standard methods would require training hundreds of models.

----

## [2143] Kernel Stein Discrepancy thinning: a theoretical perspective of pathologies and a practical fix with regularization

**Authors**: *Clément Bénard, Brian Staber, Sébastien Da Veiga*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html)

**Abstract**:

Stein thinning is a promising algorithm proposed by (Riabiz et al., 2022) for post-processing outputs of Markov chain Monte Carlo (MCMC). The main principle is to greedily minimize the kernelized Stein discrepancy (KSD), which only requires the gradient of the log-target distribution, and is thus well-suited for Bayesian inference. The main advantages of Stein thinning are the automatic remove of the burn-in period, the correction of the bias introduced by recent MCMC algorithms, and the asymptotic properties of convergence towards the target distribution. Nevertheless, Stein thinning suffers from several empirical pathologies, which may result in poor approximations, as observed in the literature. In this article, we conduct a theoretical analysis of these pathologies, to clearly identify the mechanisms at stake, and suggest improved strategies. Then, we introduce the regularized Stein thinning algorithm to alleviate the identified pathologies. Finally, theoretical guarantees and extensive experiments show the high efficiency of the proposed algorithm. An implementation of regularized Stein thinning as the kernax library in python and JAX is available at https://gitlab.com/drti/kernax.

----

## [2144] Punctuation-level Attack: Single-shot and Single Punctuation Can Fool Text Models

**Authors**: *Wenqiang Wang, Chongyang Du, Tao Wang, Kaihao Zhang, Wenhan Luo, Lin Ma, Wei Liu, Xiaochun Cao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9a9f4e15ad0d680429a3e0570a96f763-Abstract-Conference.html)

**Abstract**:

The adversarial attacks have attracted increasing attention in various fields including natural language processing. The current textual attacking models primarily focus on fooling models by adding character-/word-/sentence-level perturbations, ignoring their influence on human perception. In this paper, for the first time in the community, we propose a novel mode of textual attack, punctuation-level attack. With various types of perturbations, including insertion, displacement, deletion, and replacement, the punctuation-level attack achieves promising fooling rates against SOTA models on typical textual tasks and maintains minimal influence on human perception and understanding of the text by mere perturbation of single-shot single punctuation. Furthermore, we propose a search method named Text Position Punctuation Embedding and Paraphrase (TPPEP) to accelerate the pursuit of optimal position to deploy the attack, without exhaustive search, and we present a mathematical interpretation of TPPEP. Thanks to the integrated Text Position Punctuation Embedding (TPPE), the punctuation attack can be applied at a constant cost of time. Experimental results on public datasets and SOTA models demonstrate the effectiveness of the punctuation attack and the proposed TPPE. We additionally apply the single punctuation attack to summarization, semantic-similarity-scoring, and text-to-image tasks, and achieve encouraging results.

----

## [2145] Towards Hybrid-grained Feature Interaction Selection for Deep Sparse Network

**Authors**: *Fuyuan Lyu, Xing Tang, Dugang Liu, Chen Ma, Weihong Luo, Liang Chen, Xiuqiang He, Xue (Steve) Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ab8da29b1eb3bec912a06e0879065cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ab8da29b1eb3bec912a06e0879065cd-Abstract-Conference.html)

**Abstract**:

Deep sparse networks are widely investigated as a neural network architecture for prediction tasks with high-dimensional sparse features, with which feature interaction selection is a critical component. While previous methods primarily focus on how to search feature interaction in a coarse-grained space, less attention has been given to a finer granularity. In this work, we introduce a hybrid-grained feature interaction selection approach that targets both feature field and feature value for deep sparse networks. To explore such expansive space, we propose a decomposed space which is calculated on the fly. We then develop a selection algorithm called OptFeature, which efficiently selects the feature interaction from both the feature field and the feature value simultaneously. Results from experiments on three large real-world benchmark datasets demonstrate that OptFeature performs well in terms of accuracy and efficiency. Additional studies support the feasibility of our method. All source code are publicly available\footnote{https://anonymous.4open.science/r/OptFeature-Anonymous}.

----

## [2146] On the Asymptotic Learning Curves of Kernel Ridge Regression under Power-law Decay

**Authors**: *Yicheng Li, Haobo Zhang, Qian Lin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9adc8ada9183f4b9a007a02773fd8114-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9adc8ada9183f4b9a007a02773fd8114-Abstract-Conference.html)

**Abstract**:

The widely observed 'benign overfitting phenomenon' in the neural network literature raises the challenge to the `bias-variance trade-off' doctrine in the statistical learning theory.Since the generalization ability of the 'lazy trained' over-parametrized neural network can be well approximated by that of the neural tangent kernel regression,the curve of the excess risk (namely, the learning curve) of kernel ridge regression attracts increasing attention recently.However, most recent arguments on the learning curve are heuristic and are based on the 'Gaussian design' assumption.In this paper, under mild and more realistic assumptions, we rigorously provide a full characterization of the learning curve in the asymptotic senseunder a power-law decay condition of the eigenvalues of the kernel and also the target function.The learning curve elaborates the effect and the interplay of the choice of the regularization parameter, the source condition and the noise.In particular, our results suggest that the 'benign overfitting phenomenon' exists in over-parametrized neural networks only when the noise level is small.

----

## [2147] Mechanism Design for Collaborative Normal Mean Estimation

**Authors**: *Yiding Chen, Jerry Zhu, Kirthevasan Kandasamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html)

**Abstract**:

We study collaborative normal mean estimation, where $m$ strategic agents collect i.i.d samples from a normal distribution $\mathcal{N}(\mu, \sigma^2)$ at a cost. They all wish to estimate the mean $\mu$. By sharing data with each other, agents can obtain better estimates while keeping the cost of data collection small. To facilitate this collaboration, we wish to design mechanisms that encourage agents to collect a sufficient amount of data and share it truthfully, so that they are all better off than working alone. In naive mechanisms, such as simply pooling and sharing all the data, an individual agent might find it beneficial to under-collect and/or fabricate data, which can lead to poor social outcomes. We design a novel mechanism that overcomes these challenges via two key techniques: first, when sharing the others' data with an agent, the mechanism corrupts this dataset proportional to how much the data reported by the agent differs from the others; second, we design minimax optimal estimators for the corrupted dataset. Our mechanism, which is Nash incentive compatible and individually rational, achieves a social penalty (sum of all agents' estimation errors and data collection costs) that is at most a factor 2 of the global minimum. When applied to high dimensional (non-Gaussian) distributions with bounded variance, this mechanism retains these three properties, but with slightly weaker results. Finally, in two special cases where we restrict the strategy space of the agents, we design mechanisms that essentially achieve the global minimum.

----

## [2148] DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation

**Authors**: *Kaipeng Zheng, Huishuai Zhang, Weiran Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html)

**Abstract**:

Few-shot learning aims to adapt models trained on the base dataset to novel tasks where the categories were not seen by the model before. This often leads to a relatively concentrated distribution of feature values across channels on novel classes, posing challenges in determining channel importance for novel tasks. Standard few-shot learning methods employ geometric similarity metrics such as cosine similarity and negative Euclidean distance to gauge the semantic relatedness between two features. However, features with high geometric similarities may carry distinct semantics, especially in the context of few-shot learning. In this paper, we demonstrate that the importance ranking of feature channels is a more reliable indicator for few-shot learning than geometric similarity metrics. We observe that replacing the geometric similarity metric with Kendall’s rank correlation only during inference is able to improve the performance of few-shot learning across a wide range of methods and datasets with different domains. Furthermore, we propose a carefully designed differentiable loss for meta-training to address the non-differentiability issue of Kendall’s rank correlation. By replacing geometric similarity with differentiable Kendall’s rank correlation, our method can integrate with numerous existing few-shot approaches and is ready for integrating with future state-of-the-art methods that rely on geometric similarity metrics. Extensive experiments validate the efficacy of the rank-correlation-based approach, showcasing a significant improvement in few-shot learning.

----

## [2149] High-dimensional Contextual Bandit Problem without Sparsity

**Authors**: *Junpei Komiyama, Masaaki Imaizumi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b35a0a20d617dc68ae98a7a57df2f51-Abstract-Conference.html)

**Abstract**:

In this research, we investigate the high-dimensional linear contextual bandit problem where the number of features $p$ is greater than the budget $T$, or it may even be infinite. Differing from the majority of previous works in this field, we do not impose sparsity on the regression coefficients. Instead, we rely on recent findings on overparameterized models, which enables us to analyze the performance of the minimum-norm interpolating estimator when data distributions have small effective ranks. We propose an explore-then-commit (EtC) algorithm to address this problem and examine its performance. Through our analysis, we derive the optimal rate of the ETC algorithm in terms of $T$ and show that this rate can be achieved by balancing exploration and exploitation. Moreover, we introduce an adaptive explore-then-commit (AEtC) algorithm that adaptively finds the optimal balance. We assess the performance of the proposed algorithms through a series of simulations.

----

## [2150] VidChapters-7M: Video Chapters at Scale

**Authors**: *Antoine Yang, Arsha Nagrani, Ivan Laptev, Josef Sivic, Cordelia Schmid*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b5c3e00d6ed30aad7adac9e7a664de1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Segmenting untrimmed videos into chapters enables users to quickly navigate to the information of their interest. This important topic has been understudied due to the lack of publicly released datasets. To address this issue, we present VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. VidChapters-7M is automatically created from videos online in a scalable manner by scraping user-annotated chapters and hence without any additional manual annotation. We introduce the following three tasks based on this data. First, the video chapter generation task consists of temporally segmenting the video and generating a chapter title for each segment. To further dissect the problem, we also define two variants of this task: video chapter generation given ground-truth boundaries, which requires generating a chapter title given an annotated video segment, and video chapter grounding, which requires temporally localizing a chapter given its annotated title. We benchmark both simple baselines as well as state-of-the-art video-language models on these three tasks. We also show that pretraining on VidChapters-7M transfers well to dense video captioning tasks, largely improving the state of the art on the YouCook2 and ViTT benchmarks. Finally, our experiments reveal that downstream performance scales well with the size of the pretraining dataset.

----

## [2151] Energy-Based Models for Anomaly Detection: A Manifold Diffusion Recovery Approach

**Authors**: *Sangwoong Yoon, Young-Uk Jin, Yung-Kyun Noh, Frank C. Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html)

**Abstract**:

We present a new method of training energy-based models (EBMs) for anomaly detection that leverages low-dimensional structures within data. The proposed algorithm, Manifold Projection-Diffusion Recovery (MPDR), first perturbs a data point along a low-dimensional manifold that approximates the training dataset. Then, EBM is trained to maximize the probability of recovering the original data. The training involves the generation of negative samples via MCMC, as in conventional EBM training, but from a different distribution concentrated near the manifold. The resulting near-manifold negative samples are highly informative, reflecting relevant modes of variation in data. An energy function of MPDR effectively learns accurate boundaries of the training data distribution and excels at detecting out-of-distribution samples. Experimental results show that MPDR exhibits strong performance across various anomaly detection tasks involving diverse data types, such as images, vectors, and acoustic signals.

----

## [2152] Characterizing the Optimal 0-1 Loss for Multi-class Classification with a Test-time Attacker

**Authors**: *Sihui Dai, Wenxin Ding, Arjun Nitin Bhagoji, Daniel Cullina, Heather Zheng, Ben Zhao, Prateek Mittal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b867f0e56c4c085ef1cfdad691db5f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b867f0e56c4c085ef1cfdad691db5f6-Abstract-Conference.html)

**Abstract**:

Finding classifiers robust to adversarial examples is critical for their safedeployment. Determining the robustness of the best possible classifier under agiven threat model for a fixed data distribution and comparing it to thatachieved by state-of-the-art training methods is thus an important diagnostictool. In this paper, we find achievable information-theoretic lower bounds onrobust loss in the presence of a test-time attacker for *multi-classclassifiers on any discrete dataset*. We provide a general framework for findingthe optimal $0-1$ loss that revolves around the construction of a conflicthypergraph from the data and adversarial constraints. The prohibitive cost ofthis formulation in practice leads us to formulate other variants of the attacker-classifiergame that more efficiently determine the range of the optimal loss. Ourvaluation shows, for the first time, an analysis of the gap to optimalrobustness for classifiers in the multi-class setting on benchmark datasets.

----

## [2153] Truncated Affinity Maximization: One-class Homophily Modeling for Graph Anomaly Detection

**Authors**: *Hezhe Qiao, Guansong Pang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b905031125e56a557db38dff4fa8d21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b905031125e56a557db38dff4fa8d21-Abstract-Conference.html)

**Abstract**:

We reveal a one-class homophily phenomenon, which is one prevalent property we find empirically in real-world graph anomaly detection (GAD) datasets, i.e., normal nodes tend to have strong connection/affinity with each other, while the homophily in abnormal nodes is significantly weaker than normal nodes. However, this anomaly-discriminative property is ignored by existing GAD methods that are typically built using a conventional anomaly detection objective, such as data reconstruction.In this work, we explore this property to introduce a novel unsupervised anomaly scoring measure for GAD -- local node affinity-- that assigns a larger anomaly score to nodes that are less affiliated with their neighbors, with the affinity defined as similarity on node attributes/representations.  We further propose Truncated Affinity Maximization (TAM) that learns tailored node representations for our anomaly measure by maximizing the local affinity of nodes to their neighbors. Optimizing on the original graph structure can be biased by non-homophily edges(i.e., edges connecting normal and abnormal nodes). Thus, TAM is instead optimized on truncated graphs where non-homophily edges are removed iteratively to mitigate this bias. The learned representations result in significantly stronger local affinity for normal nodes than abnormal nodes. Extensive empirical results on 10 real-world GAD datasets show that TAM substantially outperforms seven competing models, achieving over 10% increase in AUROC/AUPRC compared to the best contenders on challenging datasets. Our code is available at https://github.com/mala-lab/TAM-master/.

----

## [2154] Sample-Conditioned Hypothesis Stability Sharpens Information-Theoretic Generalization Bounds

**Authors**: *Ziqiao Wang, Yongyi Mao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b912f91a5e299472764377db6ca2431-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b912f91a5e299472764377db6ca2431-Abstract-Conference.html)

**Abstract**:

We present new information-theoretic generalization guarantees through the a novel construction of the "neighboring-hypothesis" matrix and a new family of stability notions termed sample-conditioned hypothesis (SCH) stability.  Our approach yields sharper bounds that improve upon previous information-theoretic bounds in various learning scenarios. Notably, these bounds address the limitations of existing information-theoretic bounds in the context of stochastic convex optimization (SCO) problems, as explored in the recent work by Haghifam et al. (2023).

----

## [2155] Exploiting Contextual Objects and Relations for 3D Visual Grounding

**Authors**: *Li Yang, Chunfeng Yuan, Ziqi Zhang, Zhongang Qi, Yan Xu, Wei Liu, Ying Shan, Bing Li, Weiping Yang, Peng Li, Yan Wang, Weiming Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9b91ee0da3bcd61905fcd89e770168fc-Abstract-Conference.html)

**Abstract**:

3D visual grounding, the task of identifying visual objects in 3D scenes based on natural language inputs, plays a critical role in enabling machines to understand and engage with the real-world environment. However, this task is challenging due to the necessity to capture 3D contextual information to distinguish target objects from complex 3D scenes. The absence of annotations for contextual objects and relations further exacerbates the difficulties. In this paper, we propose a novel model, CORE-3DVG, to address these challenges by explicitly learning about contextual objects and relations. Our method accomplishes 3D visual grounding via three sequential modular networks, including a text-guided object detection network, a relation matching network, and a target identification network. During training, we introduce a pseudo-label self-generation strategy and a weakly-supervised method to facilitate the learning of contextual objects and relations, respectively. The proposed techniques allow the networks to focus more effectively on referred objects within 3D scenes by understanding their context better. We validate our model on the challenging Nr3D, Sr3D, and ScanRefer datasets and demonstrate state-of-the-art performance. Our code will be public at https://github.com/yangli18/CORE-3DVG.

----

## [2156] Learning to Search Feasible and Infeasible Regions of Routing Problems with Flexible Neural k-Opt

**Authors**: *Yining Ma, Zhiguang Cao, Yeow Meng Chee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bae70d354793a95fa18751888cea07d-Abstract-Conference.html)

**Abstract**:

In this paper, we present Neural k-Opt (NeuOpt), a novel learning-to-search (L2S) solver for routing problems. It learns to perform flexible k-opt exchanges based on a tailored action factorization method and a customized recurrent dual-stream decoder. As a pioneering work to circumvent the pure feasibility masking scheme and enable the autonomous exploration of both feasible and infeasible regions, we then propose the Guided Infeasible Region Exploration (GIRE) scheme, which supplements the NeuOpt policy network with feasibility-related features and leverages reward shaping to steer reinforcement learning more effectively. Additionally, we equip NeuOpt with Dynamic Data Augmentation (D2A) for more diverse searches during inference. Extensive experiments on the Traveling Salesman Problem (TSP) and Capacitated Vehicle Routing Problem (CVRP) demonstrate that our NeuOpt not only significantly outstrips existing (masking-based) L2S solvers, but also showcases superiority over the learning-to-construct (L2C) and learning-to-predict (L2P) solvers. Notably, we offer fresh perspectives on how neural solvers can handle VRP constraints. Our code is available: https://github.com/yining043/NeuOpt.

----

## [2157] Importance Weighted Actor-Critic for Optimal Conservative Offline Reinforcement Learning

**Authors**: *Hanlin Zhu, Paria Rashidinejad, Jiantao Jiao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bb93a3c1a424654aaea6f5b594e94d5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bb93a3c1a424654aaea6f5b594e94d5-Abstract-Conference.html)

**Abstract**:

We propose A-Crab (Actor-Critic Regularized by Average Bellman error), a new practical algorithm for offline reinforcement learning (RL) in complex environments with insufficient data coverage. Our algorithm combines the marginalized importance sampling framework with the actor-critic paradigm, where the critic returns evaluations of the actor (policy) that are pessimistic relative to the offline data and have a small average (importance-weighted) Bellman error. Compared to existing methods, our algorithm simultaneously offers a number of advantages:(1) It achieves the optimal statistical rate of $1/\sqrt{N}$---where $N$ is the size of offline dataset---in converging to the best policy covered in the offline dataset, even when combined with general function approximators.(2) It relies on a weaker \textit{average} notion of policy coverage (compared to the $\ell_\infty$ single-policy concentrability) that exploits the structure of policy visitations.(3) It outperforms the data-collection behavior policy over a wide range of specific hyperparameters. We provide both theoretical analysis and experimental results to validate the effectiveness of our proposed algorithm. The code is available at https://github.com/zhuhl98/ACrab.

----

## [2158] Hierarchical Semi-Implicit Variational Inference with Application to Diffusion Model Acceleration

**Authors**: *Longlin Yu, Tianyu Xie, Yu Zhu, Tong Yang, Xiangyu Zhang, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bbb3c3aa33616c55521e2f826c132bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bbb3c3aa33616c55521e2f826c132bd-Abstract-Conference.html)

**Abstract**:

Semi-implicit variational inference (SIVI) has been introduced to expand the analytical variational families by defining expressive semi-implicit distributions in a hierarchical manner. However, the single-layer architecture commonly used in current SIVI methods can be insufficient when the target posterior has complicated structures. In this paper, we propose hierarchical semi-implicit variational inference, called HSIVI, which generalizes SIVI to allow more expressive multi-layer construction of semi-implicit distributions. By introducing auxiliary distributions that interpolate between a simple base distribution and the target distribution, the conditional layers can be trained by progressively matching these auxiliary distributions one layer after another. Moreover, given pre-trained score networks, HSIVI can be used to accelerate the sampling process of diffusion models with the score matching objective. We show that HSIVI significantly enhances the expressiveness of SIVI on several Bayesian inference problems with complicated target distributions. When used for diffusion model acceleration, we show that HSIVI can produce high quality samples comparable to or better than the existing fast diffusion model based samplers with a small number of function evaluations on various datasets.

----

## [2159] Geometry-Aware Adaptation for Pretrained Models

**Authors**: *Nicholas Roberts, Xintong Li, Dyah Adila, Sonia Cromp, Tzu-Heng Huang, Jitian Zhao, Frederic Sala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bbc8b6038603e6170e35f89e3c3e296-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bbc8b6038603e6170e35f89e3c3e296-Abstract-Conference.html)

**Abstract**:

Machine learning models---including prominent zero-shot models---are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes---or, in the case of zero-shot prediction, to improve its performance---without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping $\text{argmax}$ with the Fr√©chet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes for when it is not possible to predict the entire range of unobserved classes. Empirically, using easily-available external metrics, our proposed approach, Loki, gains up to 29.7% relative improvement over SimCLR on ImageNet and scales to hundreds of thousands of classes. When no such metric is available, Loki can use self-derived metrics from class embeddings and obtains a 10.5% improvement on pretrained zero-shot models such as CLIP.

----

## [2160] JourneyDB: A Benchmark for Generative Image Understanding

**Authors**: *Keqiang Sun, Junting Pan, Yuying Ge, Hao Li, Haodong Duan, Xiaoshi Wu, Renrui Zhang, Aojun Zhou, Zipeng Qin, Yi Wang, Jifeng Dai, Yu Qiao, Limin Wang, Hongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bc59aff4685e39e1a8175d5303248a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bc59aff4685e39e1a8175d5303248a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While recent advancements in vision-language models have had a transformative impact on multi-modal comprehension, the extent to which these models possess the ability to comprehend generated images remains uncertain. Synthetic images, in comparison to real data, encompass a higher level of diversity in terms of both content and style, thereby presenting significant challenges for the models to fully grasp. In light of this challenge, we introduce a comprehensive dataset, referred to as JourneyDB, that caters to the domain of generative images within the context of multi-modal visual understanding. Our meticulously curated dataset comprises 4 million distinct and high-quality generated images, each paired with the corresponding text prompts that were employed in their creation. Furthermore, we additionally introduce an external subset with results of another 22 text-to-image generative models, which makes JourneyDB a comprehensive benchmark for evaluating the comprehension of generated images. On our dataset, we have devised four benchmarks to assess the performance of generated image comprehension in relation to both content and style interpretation. These benchmarks encompass prompt inversion, style retrieval, image captioning, and visual question answering. Lastly, we evaluate the performance of state-of-the-art multi-modal models when applied to the JourneyDB dataset, providing a comprehensive analysis of their strengths and limitations in comprehending generated content. We anticipate that the proposed dataset and benchmarks will facilitate further research in the field of generative content understanding. The dataset is publicly available at https://journeydb.github.io.

----

## [2161] A fast heuristic to optimize time-space tradeoff for large models

**Authors**: *Akifumi Imanishi, Zijian Xu, Masayuki Takagi, Sixue Wang, Emilio Castillo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9be39b35906526b8d240056daac72c6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9be39b35906526b8d240056daac72c6f-Abstract-Conference.html)

**Abstract**:

Training large-scale neural networks is heavily constrained by GPU memory. In order to circumvent this limitation, gradient checkpointing, or recomputation is a powerful technique. There is active research in this area with methods such as Checkmake or Moccasin. However, both Checkmate and Moccasin rely on mixed integer linear programming or constraint programming, resulting in limited scalability due to their exponentially large search space.This paper proposes a novel algorithm for recomputation (FastSA) based on a simulated annealing heuristic that achieves comparable or even better solutions than state-of-the-art alternatives. FastSA can optimize computational graphs with thousands of nodes within 3 to 30 seconds, several orders of magnitude faster than current solutions.We applied FastSA to PyTorch models and verified its effectiveness through popular large vision and text models, including recent language models with the transformer architecture. The results demonstrate significant memory reductions by 73% with extra 18% computational overheads on average. Our experiments demonstrate the practicality and effectiveness of our recomputation algorithm, further highlighting its potential for wide application in various deep learning domains.

----

## [2162] A Unified Conditional Framework for Diffusion-based Image Restoration

**Authors**: *Yi Zhang, Xiaoyu Shi, Dasong Li, Xiaogang Wang, Jian Wang, Hongsheng Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html)

**Abstract**:

Diffusion Probabilistic Models (DPMs) have recently shown remarkable performance in image generation tasks, which are capable of generating highly realistic images. When adopting DPMs for image restoration tasks, the crucial aspect lies in how to integrate the conditional information to guide the DPMs to generate accurate and natural output, which has been largely overlooked in existing works. In this paper, we present a unified conditional framework based on diffusion models for image restoration. We leverage a lightweight UNet to predict initial guidance and the diffusion model to learn the residual of the guidance. By carefully designing the basic module and integration module for the diffusion model block, we integrate the guidance and other auxiliary conditional information into every block of the diffusion model to achieve spatially-adaptive generation conditioning. To handle high-resolution images, we propose a simple yet effective inter-step patch-splitting strategy to produce arbitrary-resolution images without grid artifacts. We evaluate our conditional framework on three challenging tasks: extreme low-light denoising, deblurring, and JPEG restoration, demonstrating its significant improvements in perceptual quality and the generalization to restoration tasks. The code will be released at https://zhangyi-3.github.io/project/UCDIR/.

----

## [2163] Environment-Aware Dynamic Graph Learning for Out-of-Distribution Generalization

**Authors**: *Haonan Yuan, Qingyun Sun, Xingcheng Fu, Ziwei Zhang, Cheng Ji, Hao Peng, Jianxin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf12308ece130daa083fb21f7faf1b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf12308ece130daa083fb21f7faf1b6-Abstract-Conference.html)

**Abstract**:

Dynamic graph neural networks (DGNNs) are increasingly pervasive in exploiting spatio-temporal patterns on dynamic graphs. However, existing works fail to generalize under distribution shifts, which are common in real-world scenarios. As the generation of dynamic graphs is heavily influenced by latent environments, investigating their impacts on the out-of-distribution (OOD) generalization is critical. However, it remains unexplored with the following two major challenges: (1) How to properly model and infer the complex environments on dynamic graphs with distribution shifts? (2) How to discover invariant patterns given inferred spatio-temporal environments? To solve these challenges, we propose a novel Environment-Aware dynamic Graph LEarning (EAGLE) framework for OOD generalization by modeling complex coupled environments and exploiting spatio-temporal invariant patterns. Specifically, we first design the environment-aware EA-DGNN to model environments by multi-channel environments disentangling. Then, we propose an environment instantiation mechanism for environment diversification with inferred distributions. Finally, we discriminate spatio-temporal invariant patterns for out-of-distribution prediction by the invariant pattern recognition mechanism and perform fine-grained causal interventions node-wisely with a mixture of instantiated environment samples. Experiments on real-world and synthetic dynamic graph datasets demonstrate the superiority of our method against state-of-the-art baselines under distribution shifts. To the best of our knowledge, we are the first to study OOD generalization on dynamic graphs from the environment learning perspective.

----

## [2164] Provably Fast Finite Particle Variants of SVGD via Virtual Particle Stochastic Approximation

**Authors**: *Aniket Das, Dheeraj Nagaraj*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bf1962c5b65a243ee243bb03ff2c506-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bf1962c5b65a243ee243bb03ff2c506-Abstract-Conference.html)

**Abstract**:

Stein Variational Gradient Descent (SVGD) is a popular particle-based variational inference algorithm with impressive empirical performance across various domains. Although the population (i.e, infinite-particle) limit dynamics of SVGD is well characterized, its behavior in the finite-particle regime is far less understood. To this end, our work introduces the notion of *virtual particles* to develop novel stochastic approximations of population-limit SVGD dynamics in the space of probability measures, that are exactly realizable using finite particles. As a result, we design two computationally efficient variants of SVGD, namely VP-SVGD and GB-SVGD, with provably fast finite-particle convergence rates. Our algorithms can be viewed as specific random-batch approximations of SVGD, which are computationally more efficient than ordinary SVGD. We show that the $n$ particles output by VP-SVGD and GB-SVGD, run for $T$ steps with batch-size $K$, are at-least as good as i.i.d samples from a distribution whose Kernel Stein Discrepancy to the target is at most $O(\tfrac{d^{1/3}}{(KT)^{1/6}})$ under standard assumptions. Our results also hold under a mild growth condition on the potential function, which is much weaker than the isoperimetric (e.g. Poincare Inequality) or information-transport conditions (e.g. Talagrand's Inequality $\mathsf{T}_1$) generally considered in prior works. As a corollary, we analyze the convergence of the empirical measure (of the particles output by VP-SVGD and GB-SVGD) to the target distribution and demonstrate a **double exponential improvement** over the best known finite-particle analysis of SVGD. Beyond this, our results present the **first known oracle complexities for this setting with polynomial dimension dependence**, thereby completely eliminating the curse of dimensionality exhibited by previously known finite-particle rates.

----

## [2165] Flow Factorized Representation Learning

**Authors**: *Yue Song, Andy Keller, Nicu Sebe, Max Welling*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9bfc2c20fa2f56a18397eafe1be8a50a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9bfc2c20fa2f56a18397eafe1be8a50a-Abstract-Conference.html)

**Abstract**:

A prominent goal of representation learning research is to achieve representations which are factorized in a useful manner with respect to the ground truth factors of variation. The fields of disentangled and equivariant representation learning have approached this ideal from a range of complimentary perspectives; however, to date, most approaches have proven to either be ill-specified or insufficiently flexible to effectively separate all realistic factors of interest in a learned latent space. In this work, we propose an alternative viewpoint on such structured representation learning which we call Flow Factorized Representation Learning, and demonstrate it to learn both more efficient and more usefully structured representations than existing frameworks. Specifically, we introduce a generative model which specifies a distinct set of latent probability paths that define different input transformations. Each latent flow is generated by the gradient field of a learned potential following dynamic optimal transport. Our novel setup brings new understandings to both \textit{disentanglement} and \textit{equivariance}. We show that our model achieves higher likelihoods on standard representation learning benchmarks while simultaneously being closer to approximately equivariant models. Furthermore, we demonstrate that the transformations learned by our model are flexibly composable and can also extrapolate to new data, implying a degree of robustness and generalizability approaching the ultimate goal of usefully factorized representation learning.

----

## [2166] Hierarchical Randomized Smoothing

**Authors**: *Yan Scholten, Jan Schuchardt, Aleksandar Bojchevski, Stephan Günnemann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c0efc0d84c263972af72bf70a2de533-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c0efc0d84c263972af72bf70a2de533-Abstract-Conference.html)

**Abstract**:

Real-world data is complex and often consists of objects that can be decomposed into multiple entities (e.g. images into pixels, graphs into interconnected nodes). Randomized smoothing is a powerful framework for making models provably robust against small changes to their inputs - by guaranteeing robustness of the majority vote when randomly adding noise before classification. Yet, certifying robustness on such complex data via randomized smoothing is challenging when adversaries do not arbitrarily perturb entire objects (e.g. images) but only a subset of their entities (e.g. pixels). As a solution, we introduce hierarchical randomized smoothing: We partially smooth objects by adding random noise only on a randomly selected subset of their entities. By adding noise in a more targeted manner than existing methods we obtain stronger robustness guarantees while maintaining high accuracy. We initialize hierarchical smoothing using different noising distributions, yielding novel robustness certificates for discrete and continuous domains. We experimentally demonstrate the importance of hierarchical smoothing in image and node classification, where it yields superior robustness-accuracy trade-offs. Overall, hierarchical smoothing is an important contribution towards models that are both - certifiably robust to perturbations and accurate.

----

## [2167] BenchCLAMP: A Benchmark for Evaluating Language Models on Syntactic and Semantic Parsing

**Authors**: *Subhro Roy, Samuel Thomson, Tongfei Chen, Richard Shin, Adam Pauls, Jason Eisner, Benjamin Van Durme*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recent work has shown that generation from a prompted or fine-tuned language model can perform well at semantic parsing when the output is constrained to be a valid semantic representation. We introduce BenchCLAMP, a Benchmark to evaluate Constrained LAnguage Model Parsing, that includes context-free grammars for seven semantic parsing datasets and two syntactic parsing datasets with varied output meaning representations, as well as a constrained decoding interface to generate only valid outputs covered by these grammars. We provide low, medium, and high resource splits for each dataset, allowing accurate comparison of various language models under different data regimes. Our benchmark supports evaluation of language models using prompt-based learning as well as fine-tuning. We benchmark seven language models, including two GPT-3 variants available only through an API. Our experiments show that encoder-decoder pretrained language models can achieve similar performance or even surpass state-of-the-art methods for both syntactic and semantic parsing when the model output is constrained to be valid.

----

## [2168] Stable Nonconvex-Nonconcave Training via Linear Interpolation

**Authors**: *Thomas Pethick, Wanyun Xie, Volkan Cevher*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html)

**Abstract**:

This paper presents a theoretical analysis of linear interpolation as a principled method for stabilizing (large-scale) neural network training. We argue that instabilities in the optimization process are often caused by the nonmonotonicity of the loss landscape and show how linear interpolation can help by leveraging the theory of nonexpansive operators. We construct a new optimization scheme called relaxed approximate proximal point (RAPP), which is the first 1-SCLI method to achieve last iterate convergence rates for $\rho$-comonotone problems while only requiring $\rho > -\tfrac{1}{2L}$. The construction extends to constrained and regularized settings. By replacing the inner optimizer in RAPP we rediscover the family of Lookahead algorithms for which we establish convergence in cohypomonotone problems even when the base optimizer is taken to be gradient descent ascent. The range of cohypomonotone problems in which Lookahead converges is further expanded by exploiting that Lookahead inherits the properties of the base optimizer. We corroborate the results with experiments on generative adversarial networks which demonstrates the benefits of the linear interpolation present in both RAPP and Lookahead.

----

## [2169] UniPC: A Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models

**Authors**: *Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, Jiwen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c2aa1e456ea543997f6927295196381-Abstract-Conference.html)

**Abstract**:

Diffusion probabilistic models (DPMs) have demonstrated a very promising ability in high-resolution image synthesis. However, sampling from a pre-trained DPM is time-consuming due to the multiple evaluations of the denoising network, making it more and more important to accelerate the sampling of DPMs. Despite recent progress in designing fast samplers, existing methods still cannot generate satisfying images in many applications where fewer steps (e.g., $<$10) are favored. In this paper, we develop a unified corrector (UniC) that can be applied after any existing DPM sampler to increase the order of accuracy without extra model evaluations, and derive a unified predictor (UniP) that supports arbitrary order as a byproduct. Combining UniP and UniC, we propose a unified predictor-corrector framework called UniPC for the fast sampling of DPMs, which has a unified analytical form for any order and can significantly improve the sampling quality over previous methods, especially in extremely few steps. We evaluate our methods through extensive experiments including both unconditional and conditional sampling using pixel-space and latent-space DPMs. Our UniPC can achieve 3.87 FID on CIFAR10 (unconditional) and 7.51 FID on ImageNet 256$\times$256 (conditional) with only 10 function evaluations. Code is available at https://github.com/wl-zhao/UniPC.

----

## [2170] Coop: Memory is not a Commodity

**Authors**: *Jianhao Zhang, Shihan Ma, Peihong Liu, Jinhui Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c534edc7ac1d6438216311be6d42eb2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c534edc7ac1d6438216311be6d42eb2-Abstract-Conference.html)

**Abstract**:

Tensor rematerialization allows the training of deep neural networks (DNNs) under limited memory budgets by checkpointing the models and recomputing the evicted tensors as needed. However, the existing tensor rematerialization techniques overlook the memory system in deep learning frameworks and implicitly assume that free memory blocks at different addresses are identical. Under this flawed assumption, discontiguous tensors are evicted, among which some are not used to allocate the new tensor. This leads to severe memory fragmentation and increases the cost of potential rematerializations.To address this issue, we propose to evict tensors within a sliding window to ensure all evictions are contiguous and are immediately used. Furthermore, we proposed cheap tensor partitioning and recomputable in-place to further reduce the rematerialization cost by optimizing the tensor allocation.We named our method Coop as it is a co-optimization of tensor allocation and tensor rematerialization. We evaluated Coop on eight representative DNNs. The experimental results demonstrate that Coop achieves up to $2\times$ memory saving and hugely reduces compute overhead, search latency, and memory fragmentation compared to the state-of-the-art baselines.

----

## [2171] Learning with Explanation Constraints

**Authors**: *Rattana Pukdee, Dylan Sam, J. Zico Kolter, Maria-Florina Balcan, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c537882044c8b5352c363e840872ddb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c537882044c8b5352c363e840872ddb-Abstract-Conference.html)

**Abstract**:

As larger deep learning models are hard to interpret, there has been a recent focus on generating explanations of these black-box models. In contrast, we may have apriori explanations of how models should behave. In this paper, we formalize this notion as learning from explanation constraints and provide a learning theoretic framework to analyze how such explanations can improve the learning of our models.  One may naturally ask, "When would these explanations be helpful?"Our first key contribution addresses this question via a class of models that satisfies these explanation constraints in expectation over new data. We provide a characterization of the benefits of these models (in terms of the reduction of their Rademacher complexities) for a canonical class of explanations given by gradient information in the settings of both linear models and two layer neural networks. In addition, we provide an algorithmic solution for our framework, via a variational approximation that achieves better performance and satisfies these constraints more frequently, when compared to simpler augmented Lagrangian methods to incorporate these explanations. We demonstrate the benefits of our approach over a large array of synthetic and real-world experiments.

----

## [2172] On the Interplay between Social Welfare and Tractability of Equilibria

**Authors**: *Ioannis Anagnostides, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c6d29852a049218d70108bbf5c48dfe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c6d29852a049218d70108bbf5c48dfe-Abstract-Conference.html)

**Abstract**:

Computational tractability and social welfare (aka. efficiency) of equilibria are two fundamental but in general orthogonal considerations in algorithmic game theory. Nevertheless, we show that when (approximate) full efficiency can be guaranteed via a smoothness argument a la Roughgarden, Nash equilibria are approachable under a family of no-regret learning algorithms, thereby enabling fast and decentralized computation. We leverage this connection to obtain new convergence results in large games---wherein the number of players $n \gg 1$---under the well-documented property of full efficiency via smoothness in the limit. Surprisingly, our framework unifies equilibrium computation in disparate classes of problems including games with vanishing strategic sensitivity and two-player zero-sum games, illuminating en route an immediate but overlooked equivalence between smoothness and a well-studied condition in the optimization literature known as the Minty property. Finally, we establish that a family of no-regret dynamics attains a welfare bound that improves over the smoothness framework while at the same time guaranteeing convergence to the set of coarse correlated equilibria. We show this by employing the clairvoyant mirror descent algortihm recently introduced by Piliouras et al.

----

## [2173] Solving Linear Inverse Problems Provably via Posterior Sampling with Latent Diffusion Models

**Authors**: *Litu Rout, Negin Raoof, Giannis Daras, Constantine Caramanis, Alex Dimakis, Sanjay Shakkottai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c70cfa2e7d9328c649c94d50cbf8faf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c70cfa2e7d9328c649c94d50cbf8faf-Abstract-Conference.html)

**Abstract**:

We present the first framework to solve linear inverse problems leveraging pre-trained \textit{latent} diffusion models.  Previously proposed algorithms (such as DPS and DDRM) only apply to \textit{pixel-space} diffusion models.  We theoretically analyze our algorithm showing provable sample recovery in a linear model setting. The algorithmic insight obtained from our analysis extends to more general settings often considered in practice. Experimentally, we outperform previously proposed posterior sampling algorithms in a wide variety of problems including random inpainting, block inpainting, denoising, deblurring, destriping, and super-resolution.

----

## [2174] Maximum State Entropy Exploration using Predecessor and Successor Representations

**Authors**: *Arnav Kumar Jain, Lucas Lehnert, Irina Rish, Glen Berseth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c7900fac04a701cbed83256b76dbaa3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c7900fac04a701cbed83256b76dbaa3-Abstract-Conference.html)

**Abstract**:

Animals have a developed ability to explore that aids them in important tasks such as locating food, exploring for shelter, and finding misplaced items. These exploration skills necessarily track where they have been so that they can plan for finding items with relative efficiency. Contemporary exploration algorithms often learn a less efficient exploration strategy because they either condition only on the current state or simply rely on making random open-loop exploratory moves. In this work, we propose $\eta\psi$-Learning, a method to learn efficient exploratory policies by conditioning on past episodic experience to make the next exploratory move. Specifically, $\eta\psi$-Learning learns an exploration policy that maximizes the entropy of the state visitation distribution of a single trajectory. Furthermore, we demonstrate how variants of the predecessor representation and successor representations can be combined to predict the state visitation entropy. Our experiments demonstrate the efficacy of $\eta\psi$-Learning to strategically explore the environment and maximize the state coverage with limited samples.

----

## [2175] From Distribution Learning in Training to Gradient Search in Testing for Combinatorial Optimization

**Authors**: *Yang Li, Jinpei Guo, Runzhong Wang, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html)

**Abstract**:

Extensive experiments have gradually revealed the potential performance bottleneck of modeling Combinatorial Optimization (CO) solving as neural solution prediction tasks. The neural networks, in their pursuit of minimizing the average objective score across the distribution of historical problem instances, diverge from the core target of CO of seeking optimal solutions for every test instance. This calls for an effective search on each problem instance, while the model should serve to provide supporting knowledge that benefits the search. To this end, we propose T2T (Training to Testing) framework that first leverages the generative modeling to estimate the high-quality solution distribution for each instance during training, and then conducts a gradient-based search within the solution space during testing. The proposed neural search paradigm consistently leverages generative modeling, specifically diffusion, for graduated solution improvement. It disrupts the local structure of the given solution by introducing noise and reconstructs a lower-cost solution guided by the optimization objective.  Experimental results on Traveling Salesman Problem (TSP) and Maximal Independent Set (MIS) show the significant superiority of T2T, demonstrating an average performance gain of 49.15% for TSP solving and 17.27% for MIS solving compared to the previous state-of-the-art.

----

## [2176] Learning Curves for Noisy Heterogeneous Feature-Subsampled Ridge Ensembles

**Authors**: *Benjamin S. Ruben, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9c940ba3be5bc9020ec74279d6e37c8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9c940ba3be5bc9020ec74279d6e37c8a-Abstract-Conference.html)

**Abstract**:

Feature bagging is a well-established ensembling method which aims to reduceprediction variance by combining predictions of many estimators trained on subsetsor projections of features. Here, we develop a theory of feature-bagging in noisyleast-squares ridge ensembles and simplify the resulting learning curves in the specialcase of equicorrelated data. Using analytical learning curves, we demonstratethat subsampling shifts the double-descent peak of a linear predictor. This leadsus to introduce heterogeneous feature ensembling, with estimators built on varyingnumbers of feature dimensions, as a computationally efficient method to mitigatedouble-descent. Then, we compare the performance of a feature-subsamplingensemble to a single linear predictor, describing a trade-off between noise amplificationdue to subsampling and noise reduction due to ensembling. Our qualitativeinsights carry over to linear classifiers applied to image classification tasks withrealistic datasets constructed using a state-of-the-art deep learning feature map.

----

## [2177] Neural MMO 20: A Massively Multi-task Addition to Massively Multi-agent Learning

**Authors**: *Joseph Suarez, David Bloomin, Kyoung Whan Choe, Hao Xiang Li, Ryan Sullivan, Nishaanth Kanna, Daniel Scott, Rose S. Shuman, Herbie Bradley, Louis Castricato, Phillip Isola, Chenghui Yu, Yuhao Jiang, Qimai Li, Jiaxin Chen, Xiaolong Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Neural MMO 2.0 is a massively multi-agent and multi-task environment for reinforcement learning research. This version features a novel task-system that broadens the range of training settings and poses a new challenge in generalization: evaluation on and against tasks, maps, and opponents never seen during training. Maps are procedurally generated with 128 agents in the standard setting and 1-1024 supported overall. Version 2.0 is a complete rewrite of its predecessor with three-fold improved performance, effectively addressing simulation bottlenecks in online training. Enhancements to compatibility enable training with standard reinforcement learning frameworks designed for much simpler environments. Neural MMO 2.0 is free and open-source with comprehensive documentation available at neuralmmo.github.io and an active community Discord. To spark initial research on this new platform, we are concurrently running a competition at NeurIPS 2023.

----

## [2178] Zero-shot Visual Relation Detection via Composite Visual Cues from Large Language Models

**Authors**: *Lin Li, Jun Xiao, Guikun Chen, Jian Shao, Yueting Zhuang, Long Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9ca825deb6ce588c96f880728d3b8aea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9ca825deb6ce588c96f880728d3b8aea-Abstract-Conference.html)

**Abstract**:

Pretrained vision-language models, such as CLIP, have demonstrated strong generalization capabilities, making them promising tools in the realm of zero-shot visual recognition. Visual relation detection (VRD) is a typical task that identifies relationship (or interaction) types between object pairs within an image. However, naively utilizing CLIP with prevalent class-based prompts for zero-shot VRD has several weaknesses, e.g., it struggles to distinguish between different fine-grained relation types and it neglects essential spatial information of two objects. To this end, we propose a novel method for zero-shot VRD: RECODE, which solves RElation detection via COmposite DEscription prompts. Specifically, RECODE first decomposes each predicate category into subject, object, and spatial components. Then, it leverages large language models (LLMs) to generate description-based prompts (or visual cues) for each component. Different visual cues enhance the discriminability of similar relation categories from different perspectives, which significantly boosts performance in VRD. To dynamically fuse different cues, we further introduce a chain-of-thought method that prompts LLMs to generate reasonable weights for different visual cues. Extensive experiments on four VRD benchmarks have demonstrated the effectiveness and interpretability of RECODE.

----

## [2179] ToolQA: A Dataset for LLM Question Answering with External Tools

**Authors**: *Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, Chao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9cb2a7495900f8b602cb10159246a016-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9cb2a7495900f8b602cb10159246a016-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks, but they still suffer from challenges such as hallucination and weak numerical reasoning. To overcome these challenges, external tools can be used to enhance LLMs' question-answering abilities. However, current evaluation methods do not distinguish between questions that can be answered using LLMs' internal knowledge and those that require external information through tool use. To address this issue, we introduce a new dataset called ToolQA, which is designed to faithfully evaluate LLMs' ability to use external tools for question answering. Our development of ToolQA involved a scalable, automated process for dataset curation, along with 13 specialized tools designed for interaction with external knowledge in order to answer questions. Importantly, we strive to minimize the overlap between our benchmark data and LLMs' pre-training data, enabling a more precise evaluation of LLMs' tool-use reasoning abilities. We conducted an in-depth diagnosis of existing tool-use LLMs to highlight their strengths, weaknesses, and potential improvements. Our findings set a new benchmark for evaluating LLMs and suggest new directions for future advancements. Our data and code are freely available for the broader scientific community on GitHub.

----

## [2180] BiSLS/SPS: Auto-tune Step Sizes for Stable Bi-level Optimization

**Authors**: *Chen Fan, Gaspard Choné-Ducasse, Mark Schmidt, Christos Thrampoulidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html)

**Abstract**:

The popularity of bi-level optimization (BO) in deep learning has spurred a growing interest in studying gradient-based BO algorithms.However, existing algorithms involve two coupled learning rates that can be affected by approximation errors when computing hypergradients, making careful fine-tuning necessary to ensure fast convergence. To alleviate this issue, we investigate the use of recently proposed adaptive step-size methods, namely stochastic line search (SLS) and stochastic Polyak step size (SPS), for computing both the upper and lower-level learning rates. First, we revisit the use of SLS and SPS in single-level optimization without the additional interpolation condition that is typically assumed in prior works. For such settings, we investigate new variants of SLS and SPS that improve upon existing suggestions in the literature and are simpler to implement. Importantly, these two variants can be seen as special instances of general family of methods with an envelope-type step-size. This unified envelope strategy allows for the extension of the algorithms and their convergence guarantees to BO settings. Finally, our extensive experiments demonstrate that the new algorithms, which are available in both SGD and Adam versions, can find large learning rates with minimal tuning and converge faster than corresponding vanilla SGD or Adam BO algorithms that require fine-tuning.

----

## [2181] Compositional Abilities Emerge Multiplicatively: Exploring Diffusion Models on a Synthetic Task

**Authors**: *Maya Okawa, Ekdeep Singh Lubana, Robert P. Dick, Hidenori Tanaka*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d0f188c7947eacb0c07f709576824f6-Abstract-Conference.html)

**Abstract**:

Modern generative models exhibit unprecedented capabilities to generate extremely realistic data. However, given the inherent compositionality of the real world, reliable use of these models in practical applications requires that they exhibit the capability to compose a novel set of concepts to generate outputs not seen in the training data set. Prior work demonstrates that recent diffusion models do exhibit intriguing compositional generalization abilities, but also fail unpredictably. Motivated by this, we perform a controlled study for understanding compositional generalization in conditional diffusion models in a synthetic setting, varying different attributes of the training data and measuring the model's ability to generate samples out-of-distribution. Our results show: (i) the order in which the ability to generate samples from a concept and compose them emerges is governed by the structure of the underlying data-generating process; (ii) performance on compositional tasks exhibits a sudden "emergence" due to multiplicative reliance on the performance of constituent tasks, partially explaining emergent phenomena seen in generative models; and (iii) composing concepts with lower frequency in the training data to generate out-of-distribution samples requires considerably more optimization steps compared to generating in-distribution samples. Overall, our study lays a foundation for understanding emergent capabilities and compositionality in generative models from a data-centric perspective.

----

## [2182] Extracting Reward Functions from Diffusion Models

**Authors**: *Felipe Nuti, Tim Franzmeyer, João F. Henriques*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d23562fcedc078e27a3be813ff6feb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d23562fcedc078e27a3be813ff6feb5-Abstract-Conference.html)

**Abstract**:

Diffusion models have achieved remarkable results in image generation, and have similarly been used to learn high-performing policies in sequential decision-making tasks. Decision-making diffusion models can be trained on lower-quality data, and then be steered with a reward function to generate near-optimal trajectories.We consider the problem of extracting a reward function by comparing a decision-making diffusion model that models low-reward behavior and one that models high-reward behavior; a setting related to inverse reinforcement learning. We first define the notion of a \emph{relative reward function of two diffusion models} and show conditions under which it exists and is unique. We then devise a practical learning algorithm for extracting it by aligning the gradients of a reward function -- parametrized by a neural network -- to the difference in outputs of both diffusion models.Our method finds correct reward functions in navigation environments, and we demonstrate that steering the base model with the learned reward functions results in significantly increased performance in standard locomotion benchmarks.Finally, we demonstrate that our approach generalizes beyond sequential decision-making by learning a reward-like function from two large-scale image generation diffusion models. The extracted reward function successfully assigns lower rewards to harmful images.

----

## [2183] Disentangling Voice and Content with Self-Supervision for Speaker Recognition

**Authors**: *Tianchi Liu, Kong Aik Lee, Qiongqiong Wang, Haizhou Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html)

**Abstract**:

For speaker recognition, it is difficult to extract an accurate  speaker representation from speech because of its mixture of speaker traits and content. This paper proposes a disentanglement framework that simultaneously models speaker traits and content variability in speech. It is realized with the use of three Gaussian inference layers, each consisting of a learnable transition model that extracts distinct speech components. Notably, a strengthened transition model is specifically designed to model complex speech dynamics. We also propose a self-supervision method to dynamically disentangle content without the use of labels other than speaker identities.  The efficacy of the proposed framework is validated via experiments conducted on the VoxCeleb and SITW datasets with 9.56\% and 8.24\% average reductions in EER and minDCF, respectively. Since neither additional model training nor data is specifically needed, it is easily applicable in practical use.

----

## [2184] Automatic Integration for Spatiotemporal Neural Point Processes

**Authors**: *Zihao Zhou, Rose Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d30c2def27b5c6a5fb21a9aa5c16f8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d30c2def27b5c6a5fb21a9aa5c16f8f-Abstract-Conference.html)

**Abstract**:

Learning continuous-time point processes is essential to many discrete event forecasting tasks. However, integration poses a major challenge, particularly for spatiotemporal point processes (STPPs), as it involves calculating the likelihood through triple integrals over space and time. Existing methods for integrating STPP either assume a parametric form of the intensity function, which lacks flexibility; or approximating the intensity with Monte Carlo sampling, which introduces numerical errors. Recent work by Omi et al. proposes a dual network approach for efficient integration of flexible intensity function. However, their method only focuses on the 1D temporal point process. In this paper, we introduce a novel paradigm: Auto-STPP (Automatic Integration for Spatiotemporal Neural Point Processes) that extends the dual network approach to 3D STPP. While previous work provides a foundation, its direct extension overly restricts the intensity function and leads to computational challenges. In response, we introduce a decomposable parametrization for the integral network using ProdNet. This approach, leveraging the product of simplified univariate graphs, effectively sidesteps the computational complexities inherent in multivariate computational graphs. We prove the consistency of Auto-STPP and validate it on synthetic data and benchmark real-world datasets. Auto-STPP shows a significant advantage in recovering complex intensity functions from irregular spatiotemporal events, particularly when the intensity is sharply localized. Our code is open-source at https://github.com/Rose-STL-Lab/AutoSTPP.

----

## [2185] Identifiability Guarantees for Causal Disentanglement from Soft Interventions

**Authors**: *Jiaqi Zhang, Kristjan H. Greenewald, Chandler Squires, Akash Srivastava, Karthikeyan Shanmugam, Caroline Uhler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d3a4cdf6f70559e8c6fe02170fba568-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d3a4cdf6f70559e8c6fe02170fba568-Abstract-Conference.html)

**Abstract**:

Causal disentanglement aims to uncover a representation of data using latent variables that are interrelated through a causal model. Such a representation is identifiable if the latent model that explains the data is unique. In this paper, we focus on the scenario where unpaired observational and interventional data are available, with each intervention changing the mechanism of a latent variable. When the causal variables are fully observed, statistically consistent algorithms have been developed to identify the causal model under faithfulness assumptions. We here show that identifiability can still be achieved with unobserved causal variables, given a generalized notion of faithfulness. Our results guarantee that we can recover the latent causal model up to an equivalence class and predict the effect of unseen combinations of interventions, in the limit of infinite data. We implement our causal disentanglement framework by developing an autoencoding variational Bayes algorithm and apply it to the problem of predicting combinatorial perturbation effects in genomics.

----

## [2186] Equivariant Adaptation of Large Pretrained Models

**Authors**: *Arnab Kumar Mondal, Siba Smarak Panigrahi, Oumar Kaba, Sai Mudumba, Siamak Ravanbakhsh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d5856318032ef3630cb580f4e24f823-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d5856318032ef3630cb580f4e24f823-Abstract-Conference.html)

**Abstract**:

Equivariant networks are specifically designed to ensure consistent behavior with respect to a set of input transformations, leading to higher sample efficiency and more accurate and robust predictions. However, redesigning each component of prevalent deep neural network architectures to achieve chosen equivariance is a difficult problem and can result in a computationally expensive network during both training and inference. A recently proposed alternative towards equivariance that removes the architectural constraints is to use a simple canonicalization network that transforms the input to a canonical form before feeding it to an unconstrained prediction network. We show here that this approach can effectively be used to make a large pretrained network equivariant. However, we observe that the produced canonical orientations can be misaligned with those of the training distribution, hindering performance. Using dataset-dependent priors to inform the canonicalization function, we are able to make large pretrained models equivariant while maintaining their performance. This significantly improves the robustness of these models to deterministic transformations of the data, such as rotations. We believe this equivariant adaptation of large pretrained models can help their domain-specific applications with known symmetry priors.

----

## [2187] HT-Step: Aligning Instructional Articles with How-To Videos

**Authors**: *Triantafyllos Afouras, Effrosyni Mavroudi, Tushar Nagarajan, Huiyu Wang, Lorenzo Torresani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d58d85bfc041b4f901c62ba37a3f322-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d58d85bfc041b4f901c62ba37a3f322-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce HT-Step, a large-scale dataset containing temporal annotations of instructional article steps in cooking videos. It includes 122k segment-level annotations over 20k narrated videos (approximately 2.3k hours) of the HowTo100M dataset.Each annotation provides a temporal interval, and a categorical step label from a taxonomy of 4,958 unique steps automatically mined from wikiHow articles which include rich descriptions of each step.Our dataset significantly surpasses existing labeled step datasets in terms of scale, number of tasks, and richness of natural language step descriptions. Based on these annotations, we introduce a strongly supervised benchmark for aligning instructional articles with how-to videos and present a comprehensive evaluation of baseline methods for this task.By publicly releasing these annotations and defining rigorous evaluation protocols and metrics,we hope to significantly accelerate research in the field of procedural activity understanding.

----

## [2188] Provable Training for Graph Contrastive Learning

**Authors**: *Yue Yu, Xiao Wang, Mengmei Zhang, Nian Liu, Chuan Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d75de47462ffe77addaa7b985fc6d8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d75de47462ffe77addaa7b985fc6d8e-Abstract-Conference.html)

**Abstract**:

Graph Contrastive Learning (GCL) has emerged as a popular training approach for learning node embeddings from augmented graphs without labels. Despite the key principle that maximizing the similarity between positive node pairs while minimizing it between negative node pairs is well established, some fundamental problems are still unclear. Considering the complex graph structure, are some nodes consistently well-trained and following this principle even with different graph augmentations? Or are there some nodes more likely to be untrained across graph augmentations and violate the principle? How to distinguish these nodes and further guide the training of GCL? To answer these questions, we first present experimental evidence showing that the training of GCL is indeed imbalanced across all nodes. To address this problem, we propose the metric "node compactness", which is the lower bound of how a node follows the GCL principle related to the range of augmentations. We further derive the form of node compactness theoretically through bound propagation, which can be integrated into binary cross-entropy as a regularization. To this end, we propose the PrOvable Training (POT) for GCL, which regularizes the training of GCL to encode node embeddings that follows the GCL principle better. Through extensive experiments on various benchmarks, POT consistently improves the existing GCL approaches, serving as a friendly plugin.

----

## [2189] Generalized Weighted Path Consistency for Mastering Atari Games

**Authors**: *Dengwei Zhao, Shikui Tu, Lei Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d87a0c38431d0ec8d8b8ece95198c04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d87a0c38431d0ec8d8b8ece95198c04-Abstract-Conference.html)

**Abstract**:

Reinforcement learning with the help of neural-guided search consumes huge computational resources to achieve remarkable performance. Path consistency (PC), i.e., $f$ values on one optimal path should be identical, was previously imposed on MCTS by PCZero to improve the learning efficiency of AlphaZero. Not only  PCZero still lacks a theoretical support but also considers  merely board games. In this paper,  PCZero is  generalized into GW-PCZero for  real applications with non-zero immediate reward. A weighting mechanism is introduced to reduce the variance caused by scouting's uncertainty on the $f$ value estimation. For the first time, it is theoretically proved that neural-guided MCTS is guaranteed to find the optimal solution under the constraint of PC. Experiments are conducted on the Atari $100$k benchmark with $26$ games and GW-PCZero achieves $198\%$ mean human performance, higher than the state-of-the-art EfficientZero's $194\\%$, while consuming only $25\\%$ of the computational resources consumed by EfficientZero.

----

## [2190] Scaling Data-Constrained Language Models

**Authors**: *Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, Colin A. Raffel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html)

**Abstract**:

The current trend of scaling language models involves increasing both parameter count and training dataset size. Extrapolating this trend suggests that training dataset size may soon be limited by the amount of text data available on the internet. Motivated by this limit, we investigate scaling language models in data-constrained regimes. Specifically, we run a large set of experiments varying the extent of data repetition and compute budget, ranging up to 900 billion training tokens and 9 billion parameter models. We find that with constrained data for a fixed compute budget, training with up to 4 epochs of repeated data yields negligible changes to loss compared to having unique data. However, with more repetition, the value of adding compute eventually decays to zero. We propose and empirically validate a scaling law for compute optimality that accounts for the decreasing value of repeated tokens and excess parameters. Finally, we experiment with approaches mitigating data scarcity, including augmenting the training dataset with code data or removing commonly used filters. Models and datasets from our 400 training runs are freely available at https://github.com/huggingface/datablations.

----

## [2191] A Definition of Continual Reinforcement Learning

**Authors**: *David Abel, André Barreto, Benjamin Van Roy, Doina Precup, Hado Philip van Hasselt, Satinder Singh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d8cf1247786d6dfeefeeb53b8b5f6d7-Abstract-Conference.html)

**Abstract**:

In a standard view of the reinforcement learning problem, an agent’s goal is to efficiently identify a policy that maximizes long-term reward. However, this perspective is based on a restricted view of learning as finding a solution, rather than treating learning as endless adaptation. In contrast, continual reinforcement learning refers to the setting in which the best agents never stop learning. Despite the importance of continual reinforcement learning, the community lacks a simple definition of the problem that highlights its commitments and makes its primary concepts precise and clear. To this end, this paper is dedicated to carefully defining the continual reinforcement learning problem. We formalize the notion of agents that “never stop learning” through a new mathematical language for analyzing and cataloging agents. Using this new language, we define a continual learning agent as one that can be understood as carrying out an implicit search process indefinitely, and continual reinforcement learning as the setting in which the best agents are all continual learning agents. We provide two motivating examples, illustrating that traditional views of multi-task reinforcement learning and continual supervised learning are special cases of our definition. Collectively, these definitions and perspectives formalize many intuitive concepts at the heart of learning, and open new research pathways surrounding continual learning agents.

----

## [2192] A Dual-Stream Neural Network Explains the Functional Segregation of Dorsal and Ventral Visual Pathways in Human Brains

**Authors**: *Minkyu Choi, Kuan Han, Xiaokai Wang, Yizhen Zhang, Zhongming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9d8ed3c9e27a9265ee60c8edba3dec1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9d8ed3c9e27a9265ee60c8edba3dec1d-Abstract-Conference.html)

**Abstract**:

The human visual system uses two parallel pathways for spatial processing and object recognition. In contrast, computer vision systems tend to use a single feedforward pathway, rendering them less robust, adaptive, or efficient than human vision. To bridge this gap, we developed a dual-stream vision model inspired by the human eyes and brain. At the input level, the model samples two complementary visual patterns to mimic how the human eyes use magnocellular and parvocellular retinal ganglion cells to separate retinal inputs to the brain. At the backend, the model processes the separate input patterns through two branches of convolutional neural networks (CNN) to mimic how the human brain uses the dorsal and ventral cortical pathways for parallel visual processing. The first branch (WhereCNN) samples a global view to learn spatial attention and control eye movements. The second branch (WhatCNN) samples a local view to represent the object around the fixation. Over time, the two branches interact recurrently to build a scene representation from moving fixations. We compared this model with the human brains processing the same movie and evaluated their functional alignment by linear transformation. The WhereCNN and WhatCNN branches were found to differentially match the dorsal and ventral pathways of the visual cortex, respectively, primarily due to their different learning objectives, rather than their distinctions in retinal sampling or sensitivity to attention-driven eye movements. These model-based results lead us to speculate that the distinct responses and representations of the ventral and dorsal streams are more influenced by their distinct goals in visual attention and object recognition than by their specific bias or selectivity in retinal inputs. This dual-stream model takes a further step in brain-inspired computer vision, enabling parallel neural networks to actively explore and understand the visual surroundings.

----

## [2193] When Do Transformers Shine in RL? Decoupling Memory from Credit Assignment

**Authors**: *Tianwei Ni, Michel Ma, Benjamin Eysenbach, Pierre-Luc Bacon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dc5accb1e4f4a9798eae145f2e4869b-Abstract-Conference.html)

**Abstract**:

Reinforcement learning (RL) algorithms face two distinct challenges: learning effective representations of past and present observations, and determining how actions influence future returns. Both challenges involve modeling long-term dependencies. The Transformer architecture has been very successful to solve problems that involve long-term dependencies, including in the RL domain. However, the underlying reason for the strong performance of Transformer-based RL methods remains unclear: is it because they learn effective memory, or because they perform effective credit assignment? After introducing formal definitions of memory length and credit assignment length, we design simple configurable tasks to measure these distinct quantities. Our empirical results reveal that Transformers can enhance the memory capability of RL algorithms, scaling up to tasks that require memorizing observations $1500$ steps ago. However, Transformers do not improve long-term credit assignment. In summary, our results provide an explanation for the success of Transformers in RL, while also highlighting an important area for future research and benchmark design. Our code is open-sourced at https://github.com/twni2016/Memory-RL.

----

## [2194] Hypothesis Selection with Memory Constraints

**Authors**: *Maryam Aliakbarpour, Mark Bun, Adam Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dd67d30e0edd53581363c1b49006e1d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dd67d30e0edd53581363c1b49006e1d-Abstract-Conference.html)

**Abstract**:

Hypothesis selection is a fundamental problem in learning theory and statistics. Given a dataset and a finite set of candidate distributions, the goal is to select a distribution that matches the data as well as possible. More specifically, suppose we have sample access to an unknown distribution $P$ over a domain $\mathcal{X}$ that we know is well-approximated by one of a a class of $n$  distributions (a.k.a. hypotheses), $\mathcal{H} \coloneqq \{H_1, H_2, \ldots, H_n\}$. The goal is to design an algorithm that outputs a distribution $\hat{H} \in \mathcal{H}$ whose total variation distance from $P$ is nearly minimal.In this work, we study the hypothesis selection problem under memory constraints. We consider a model where samples from $P$ are presented in a stream and we access each sample $x$ via ``PDF-comparison'' queries that allow us to compare the probability densities of any pair of hypothesesat the domain point $x$ (i.e., is $H_i(x) < H_j(x)$?). This model allows us to study how much memory is needed at any point in time to store information about the portion of the stream seen so far.Our main result is an algorithm that achieves a nearly optimal tradeoff between memory usage and the number of samples required. In particular, given $b$ bits of memory (for $b$ roughly between $\log n$ and $n$), our algorithm solves the hypothesis selection problem with $s$ samples, where $b \cdot s = O(n \log n)$. This result is optimal up to an $O(\log n)$ factor, for all $b$.

----

## [2195] Optimization or Architecture: How to Hack Kalman Filtering

**Authors**: *Ido Greenberg, Netanel Yannay, Shie Mannor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9dfcc83c01e94d02c751c47517855c9f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9dfcc83c01e94d02c751c47517855c9f-Abstract-Conference.html)

**Abstract**:

In non-linear filtering, it is traditional to compare non-linear architectures such as neural networks to the standard linear Kalman Filter (KF). We observe that this mixes the evaluation of two separate components: the non-linear architecture, and the parameters optimization method. In particular, the non-linear model is often optimized, whereas the reference KF model is not. We argue that both should be optimized similarly, and to that end present the Optimized KF (OKF). We demonstrate that the KF may become competitive to neural models â€“ if optimized using OKF. This implies that experimental conclusions of certain previous studies were derived from a flawed process. The advantage of OKF over the standard KF is further studied theoretically and empirically, in a variety of problems. Conveniently, OKF can replace the KF in real-world systems by merely updating the parameters.

----

## [2196] Online robust non-stationary estimation

**Authors**: *Abishek Sankararaman, Balakrishnan Narayanaswamy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e15d892c63903ecc278e0dd05536951-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e15d892c63903ecc278e0dd05536951-Abstract-Conference.html)

**Abstract**:

The real-time estimation of time-varying parameters from high-dimensional, heavy-tailed and corrupted data-streams is a common sub-routine in systems ranging from those for network monitoring and anomaly detection to those for traffic scheduling in data-centers. For estimation tasks that can be cast as minimizing a strongly convex loss function, we prove that an appropriately tuned version of the {\ttfamily clipped Stochastic Gradient Descent} (SGD) is simultaneously {\em(i)} adaptive to drift, {\em (ii)} robust to heavy-tailed inliers and arbitrary corruptions,  {\em(iii)} requires no distributional knowledge and {\em (iv)} can be implemented in an online streaming fashion. All prior estimation algorithms have only been proven to posses a subset of these practical desiderata. A observation we make is that, neither the $\mathcal{O}\left(\frac{1}{t}\right)$ learning rate for {\ttfamily clipped SGD} known to be optimal for strongly convex loss functions of a \emph{stationary} data-stream, nor the $\mathcal{O}(1)$ learning rate known to be optimal for being adaptive to drift in a \emph{noiseless} environment can be used. Instead, a learning rate of $T^{-\alpha}$ for $ \alpha < 1$ where $T$ is the stream-length is needed to balance adaptivity to potential drift and to combat noise. We develop a new inductive argument and combine it with a martingale concentration result to derive high-probability under \emph{any learning rate} on data-streams exhibiting \emph{arbitrary distribution shift} - a proof strategy that may be of independent interest. Further, using the classical doubling-trick, we relax the knowledge of the stream length $T$. Ours is the first online estimation algorithm that is provably robust to heavy-tails, corruptions and distribution shift simultaneously. We complement our theoretical results empirically on synthetic and real data.

----

## [2197] POP-3D: Open-Vocabulary 3D Occupancy Prediction from Images

**Authors**: *Antonín Vobecký, Oriane Siméoni, David Hurych, Spyridon Gidaris, Andrei Bursuc, Patrick Pérez, Josef Sivic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e30acdeff572463c1db9b7de59de64c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e30acdeff572463c1db9b7de59de64c-Abstract-Conference.html)

**Abstract**:

We describe an approach to predict open-vocabulary 3D semantic voxel occupancy map from input 2D images with the objective of enabling 3D grounding, segmentation and retrieval of free-form language queries. This is a challenging problem because of the 2D-3D ambiguity and the open-vocabulary nature of the target tasks, where obtaining annotated training data in 3D is difficult. The contributions of this work are three-fold. First, we design a new model architecture for open-vocabulary 3D semantic occupancy prediction.  The architecture consists of a 2D-3D encoder together with occupancy prediction and 3D-language heads. The output is a dense voxel map of 3D grounded language embeddings enabling a range of open-vocabulary tasks. Second, we develop a tri-modal self-supervised learning algorithm that leverages three modalities: (i) images, (ii) language and (iii) LiDAR point clouds, and enables training the proposed architecture using a strong pre-trained vision-language model without the need for any 3D manual language annotations. Finally, we demonstrate quantitatively the strengths of the proposed model on several open-vocabulary tasks:Zero-shot 3D semantic segmentation using existing datasets; 3D grounding and retrieval of free-form language queries, using a small dataset that we propose as an extension of nuScenes. You can find the project page here https://vobecant.github.io/POP3D.

----

## [2198] Faster Relative Entropy Coding with Greedy Rejection Coding

**Authors**: *Gergely Flamich, Stratis Markou, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e720fce64f91114c49cfd640d821da3-Abstract-Conference.html)

**Abstract**:

Relative entropy coding (REC) algorithms encode a sample from a target distribution $Q$ using a proposal distribution $P$ using as few bits as possible. Unlike entropy coding, REC does not assume discrete distributions and require quantisation.As such, it can be naturally integrated into communication pipelines such as learnt compression and differentially private federated learning. Unfortunately, despite their practical benefits, REC algorithms have not seen widespread application, due to their prohibitively slow runtimes or restrictive assumptions. In this paper, we make progress towards addressing these issues. We introduce Greedy Rejection Coding (GRC), which generalises the rejection sampling-based algorithm of Harsha et al. (2007) to arbitrary probability spaces and partitioning schemes. We first show that GRC terminates almost surely and returns unbiased samples from $Q$, and then focus on two variants of GRC, namely GRCS and GRCD. We show that for continuous $Q$ and $P$ over $\mathbb{R}$ with unimodal $dQ/dP$, the expected runtime of GRCS is upper bounded by $\beta D_{KL}(Q||P) + \mathcal{O}(1)$ where $\beta \approx 4.82$, and its expected codelength is optimal. This makes GRCS the first REC algorithm with guaranteed optimal runtime for this class of distributions, up to the multiplicative constant $\beta$. This significantly improves upon the previous state-of-the-art method, A* coding (Flamich et al., 2022). Under the same assumptions, we experimentally observe and conjecture that the expected runtime and codelength of GRCD are upper bounded by $D_{KL}(Q||P) + \mathcal{O}(1)$. Finally, we evaluate GRC in a compression pipeline with variational autoencoders on MNIST, and show that a modified training objective and a codelength-compression method can further improve compression efficiency.

----

## [2199] Sparse Parameterization for Epitomic Dataset Distillation

**Authors**: *Xing Wei, Anjia Cao, Funing Yang, Zhiheng Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9e8889198d16fb79926e71adbe38cae4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9e8889198d16fb79926e71adbe38cae4-Abstract-Conference.html)

**Abstract**:

The success of deep learning relies heavily on large and diverse datasets, but the storage, preprocessing, and training of such data present significant challenges. To address these challenges, dataset distillation techniques have been proposed to obtain smaller synthetic datasets that capture the essential information of the originals. In this paper, we introduce a Sparse Parameterization for Epitomic datasEt Distillation (SPEED) framework, which leverages the concept of dictionary learning and sparse coding to distill epitomes that represent pivotal information of the dataset. SPEED prioritizes proper parameterization of the synthetic dataset and introduces techniques to capture spatial redundancy within and between synthetic images. We propose Spatial-Agnostic Epitomic Tokens (SAETs) and Sparse Coding Matrices (SCMs) to efficiently represent and select significant features. Additionally, we build a Feature-Recurrent Network (FReeNet) to generate hierarchical features with high compression and storage efficiency. Experimental results demonstrate the superiority of SPEED in handling high-resolution datasets, achieving state-of-the-art performance on multiple benchmarks and downstream applications. Our framework is compatible with a variety of dataset matching approaches, generally enhancing their performance. This work highlights the importance of proper parameterization in epitomic dataset distillation and opens avenues for efficient representation learning. Source code is available at https://github.com/MIV-XJTU/SPEED.

----



[Go to the previous page](NIPS-2023-list10.md)

[Go to the next page](NIPS-2023-list12.md)

[Go to the catalog section](README.md)