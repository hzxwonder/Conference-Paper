## [1600] Adversarial Auto-Augment with Label Preservation: A Representation Learning Principle Guided Approach

        **Authors**: *Kaiwen Yang, Yanchao Sun, Jiahao Su, Fengxiang He, Xinmei Tian, Furong Huang, Tianyi Zhou, Dacheng Tao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8a1c4a54d73728d4d61701e320687c6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8a1c4a54d73728d4d61701e320687c6d-Abstract-Conference.html)

        **Abstract**:

        Data augmentation is a critical contributing factor to the success of deep learning but heavily relies on prior domain knowledge which is not always available. Recent works on automatic data augmentation learn a policy to form a sequence of augmentation operations, which are still pre-defined and restricted to limited options. In this paper, we show that a prior-free autonomous data augmentation's objective can be derived from a representation learning principle that aims to preserve the minimum sufficient information of the labels. Given an example, the objective aims at creating a distant ``hard positive example'' as the augmentation, while still preserving the original label. We then propose a practical surrogate to the objective that can be optimized efficiently and integrated seamlessly into existing methods for a broad class of machine learning tasks, e.g., supervised, semi-supervised, and noisy-label learning. Unlike previous works, our method does not require training an extra generative model but instead leverages the intermediate layer representations of the end-task model for generating data augmentations. In experiments, we show that our method consistently brings non-trivial improvements to the three aforementioned learning tasks from both efficiency and final performance, either or not combined with pre-defined augmentations, e.g., on medical images when domain knowledge is unavailable and the existing augmentation techniques perform poorly. Code will be released publicly.

        ----

        ## [1601] Coordinate Linear Variance Reduction for Generalized Linear Programming

        **Authors**: *Chaobing Song, Cheuk Yin Lin, Stephen J. Wright, Jelena Diakonikolas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8a54a80ffc2834689ffdd0920202018e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8a54a80ffc2834689ffdd0920202018e-Abstract-Conference.html)

        **Abstract**:

        We study a class of generalized linear programs (GLP) in a large-scale setting, which includes simple, possibly nonsmooth convex regularizer and simple convex set constraints. By reformulating (GLP) as an equivalent convex-concave min-max problem, we show that the linear structure in the problem can be used to design an efficient, scalable first-order algorithm, to which we give the name Coordinate Linear Variance Reduction (CLVR; pronounced ``clever''). CLVR yields improved complexity results for (GLP) that depend on the max row norm of the linear constraint matrix in (GLP) rather than the spectral norm. When the regularization terms and constraints are separable, CLVR admits an efficient lazy update strategy that makes its complexity bounds scale with the number of nonzero elements of the linear constraint matrix in (GLP) rather than the matrix dimensions. On the other hand, for the special case of linear programs, by exploiting sharpness, we propose a restart scheme for CLVR to obtain empirical linear convergence. Then we show that Distributionally Robust Optimization (DRO) problems with ambiguity sets based on both $f$-divergence and Wasserstein metrics can be reformulated as (GLPs) by introducing sparsely connected auxiliary variables. We complement our theoretical guarantees with numerical experiments that verify our algorithm's practical effectiveness, in terms of wall-clock time and number of data passes.

        ----

        ## [1602] Probabilistic Missing Value Imputation for Mixed Categorical and Ordered Data

        **Authors**: *Yuxuan Zhao, Alex Townsend, Madeleine Udell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8a7e7f5ed2aee24e98d65b5efdde8e1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8a7e7f5ed2aee24e98d65b5efdde8e1f-Abstract-Conference.html)

        **Abstract**:

        Many real-world datasets contain missing entries and mixed data types including categorical and ordered (e.g. continuous and ordinal) variables. Imputing the missing entries is necessary, since many data analysis pipelines require complete data, but challenging especially for mixed data. This paper proposes a probabilistic imputation method using an extended Gaussian copula model that supports both single and multiple imputation. The method models mixed categorical and ordered data using a latent Gaussian distribution. The unordered characteristics of categorical variables is explicitly modeled using the argmax operator. The method makes no assumptions on the data marginals nor does it require tuning any hyperparameters. Experimental results on synthetic and real datasets show that imputation with the extended Gaussian copula outperforms the current state-of-the-art for both categorical and ordered variables in mixed data.

        ----

        ## [1603] Gaussian Copula Embeddings

        **Authors**: *Chien Lu, Jaakko Peltonen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ae260afda41b45ed77be58358a6c519-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ae260afda41b45ed77be58358a6c519-Abstract-Conference.html)

        **Abstract**:

        Learning latent vector representations via embedding models has been shown promising in machine learning. However, most of the embedding models are still limited to a single type of observation data. We propose a Gaussian copula embedding model to learn latent vector representations of items in a heterogeneous data setting. The proposed model can effectively incorporate different types of observed data and, at the same time, yield robust embeddings. We demonstrate the proposed model can effectively learn in many different scenarios, outperforming competing models in modeling quality and task performance.

        ----

        ## [1604] SGAM: Building a Virtual 3D World through Simultaneous Generation and Mapping

        **Authors**: *Yuan Shen, Wei-Chiu Ma, Shenlong Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ae9cf363ea625161f885b798c1f1f78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ae9cf363ea625161f885b798c1f1f78-Abstract-Conference.html)

        **Abstract**:

        We present simultaneous generation and mapping (SGAM), a novel 3D scene generation algorithm. Our goal is to produce a realistic, globally consistent 3D world on a large scale. Achieving this goal is challenging and goes beyond the capacities of existing 3D generation or video generation approaches, which fail to scale up to create large, globally consistent 3D scene structures. Towards tackling the challenges, we take a hybrid approach that integrates generative sensor model- ing with 3D reconstruction. Our proposed approach is an autoregressive generative framework that simultaneously generates sensor data at novel viewpoints and builds a 3D map at each timestamp. Given an arbitrary camera trajectory, our method repeatedly applies this generation-and-mapping process for thousands of steps, allowing us to create a gigantic virtual world. Our model can be trained from RGB-D sequences without having access to the complete 3D scene structure. The generated scenes are readily compatible with various interactive environments and rendering engines. Experiments on CLEVER and GoogleEarth datasets demon- strates ours can generate consistent, realistic, and geometrically-plausible scenes that compare favorably to existing view synthesis methods. Our project page is available at https://yshen47.github.io/sgam.

        ----

        ## [1605] Finding Naturally Occurring Physical Backdoors in Image Datasets

        **Authors**: *Emily Wenger, Roma Bhattacharjee, Arjun Nitin Bhagoji, Josephine Passananti, Emilio Andere, Heather Zheng, Ben Y. Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8af749935131cc8ea5dae4f6d8cdb304-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8af749935131cc8ea5dae4f6d8cdb304-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Extensive literature on backdoor poison attacks has studied attacks and defenses for backdoors using  “digital trigger patterns.” In contrast, “physical backdoors” use physical objects as triggers, have only recently been identified, and are qualitatively different enough to resist most defenses targeting digital trigger backdoors. Research on physical backdoors is limited by access to large datasets containing real images of physical objects co-located with misclassification targets. Building these datasets is time- and labor-intensive.This work seeks to address the challenge of accessibility for research on physical backdoor attacks. We hypothesize that there may be naturally occurring physically co-located objects already present in popular datasets such as ImageNet. Once identified, a careful relabeling of these data can transform them into training samples for physical backdoor attacks. We propose a method to scalably identify these subsets of potential triggers in existing datasets, along with the specific classes they can poison. We call these naturally occurring trigger-class subsets natural backdoor datasets. Our techniques successfully identify natural backdoors in widely-available datasets, and produce models behaviorally equivalent to those trained on manually curated datasets. We release our code to allow the research community to create their own datasets for research on physical backdoor attacks.

        ----

        ## [1606] Unsupervised Representation Learning from Pre-trained Diffusion Probabilistic Models

        **Authors**: *Zijian Zhang, Zhou Zhao, Zhijie Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8aff4ffcf2a9d41692a805b3987e29ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8aff4ffcf2a9d41692a805b3987e29ea-Abstract-Conference.html)

        **Abstract**:

        Diffusion Probabilistic Models (DPMs) have shown a powerful capacity of generating high-quality image samples. Recently, diffusion autoencoders (Diff-AE) have been proposed to explore DPMs for representation learning via autoencoding. Their key idea is to jointly train an encoder for discovering meaningful representations from images and a conditional DPM as the decoder for reconstructing images. Considering that training DPMs from scratch will take a long time and there have existed numerous pre-trained DPMs, we propose \textbf{P}re-trained \textbf{D}PM \textbf{A}uto\textbf{E}ncoding (\textbf{PDAE}), a general method to adapt existing pre-trained DPMs to the decoders for image reconstruction, with better training efficiency and performance than Diff-AE. Specifically, we find that the reason that pre-trained DPMs fail to reconstruct an image from its latent variables is due to the information loss of forward process, which causes a gap between their predicted posterior mean and the true one. From this perspective, the classifier-guided sampling method can be explained as computing an extra mean shift to fill the gap, reconstructing the lost class information in samples. These imply that the gap corresponds to the lost information of the image, and we can reconstruct the image by filling the gap. Drawing inspiration from this, we employ a trainable model to predict a mean shift according to encoded representation and train it to fill as much gap as possible, in this way, the encoder is forced to learn as much information as possible from images to help the filling. By reusing a part of network of pre-trained DPMs and redesigning the weighting scheme of diffusion loss, PDAE can learn meaningful representations from images efficiently. Extensive experiments demonstrate the effectiveness, efficiency and flexibility of PDAE.

        ----

        ## [1607] Learning Causally Invariant Representations for Out-of-Distribution Generalization on Graphs

        **Authors**: *Yongqiang Chen, Yonggang Zhang, Yatao Bian, Han Yang, Kaili Ma, Binghui Xie, Tongliang Liu, Bo Han, James Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8b21a7ea42cbcd1c29a7a88c444cce45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8b21a7ea42cbcd1c29a7a88c444cce45-Abstract-Conference.html)

        **Abstract**:

        Despite recent success in using the invariance principle for out-of-distribution (OOD) generalization on Euclidean data (e.g., images), studies on graph data are still limited. Different from images, the complex nature of graphs poses unique challenges to adopting the invariance principle. In particular, distribution shifts on graphs can appear in a variety of forms such as attributes and structures, making it difficult to identify the invariance. Moreover, domain or environment partitions, which are often required by OOD methods on Euclidean data, could be highly expensive to obtain for graphs. To bridge this gap, we propose a new framework, called Causality Inspired Invariant Graph LeArning (CIGA), to capture the invariance of graphs for guaranteed OOD generalization under various distribution shifts. Specifically, we characterize potential distribution shifts on graphs with causal models, concluding that OOD generalization on graphs is achievable when models focus only on subgraphs containing the most information about the causes of labels. Accordingly, we propose an information-theoretic objective to extract the desired subgraphs that maximally preserve the invariant intra-class information. Learning with these subgraphs is immune to distribution shifts. Extensive experiments on 16 synthetic or real-world datasets, including a challenging setting -- DrugOOD, from AI-aided drug discovery, validate the superior OOD performance of CIGA.

        ----

        ## [1608] To update or not to update? Neurons at equilibrium in deep models

        **Authors**: *Andrea Bragagnolo, Enzo Tartaglione, Marco Grangetto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8b2fc235787852ead92da2268cd9e90c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8b2fc235787852ead92da2268cd9e90c-Abstract-Conference.html)

        **Abstract**:

        Recent advances in deep learning optimization showed that, with some a-posteriori information on fully-trained models, it is possible to match the same performance by simply training a subset of their parameters. Such a discovery has a broad impact from theory to applications, driving the research towards methods to identify the minimum subset of parameters to train without look-ahead information exploitation. However, the methods proposed do not match the state-of-the-art performance, and rely on unstructured sparsely connected models.In this work we shift our focus from the single parameters to the behavior of the whole neuron, exploiting the concept of neuronal equilibrium (NEq). When a neuron is in a configuration at equilibrium (meaning that it has learned a specific input-output relationship), we can halt its update; on the contrary, when a neuron is at non-equilibrium, we let its state evolve towards an equilibrium state, updating its parameters. The proposed approach has been tested on different state-of-the-art learning strategies and tasks, validating NEq and observing that the neuronal equilibrium depends on the specific learning setup.

        ----

        ## [1609] Non-convex online learning via algorithmic equivalence

        **Authors**: *Udaya Ghai, Zhou Lu, Elad Hazan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8b40b4984e6c09ee49333ddd2dc719d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8b40b4984e6c09ee49333ddd2dc719d4-Abstract-Conference.html)

        **Abstract**:

        We study an algorithmic equivalence technique between non-convex gradient descent and convex mirror descent. We start by looking at a harder problem of regret minimization in online non-convex optimization. We show that under certain geometric and smoothness conditions, online gradient descent applied to  non-convex  functions is an approximation of online mirror descent applied to convex functions under reparameterization. In continuous time, the gradient flow with this reparameterization was shown to be \emph{exactly} equivalent to continuous-time mirror descent by Amid and Warmuth, but theory for the analogous discrete time algorithms is left as an open problem. We prove an $O(T^{\frac{2}{3}})$ regret bound for non-convex online gradient descent in this setting, answering this open problem. Our analysis is based on a new and simple algorithmic equivalence method.

        ----

        ## [1610] TA-MoE: Topology-Aware Large Scale Mixture-of-Expert Training

        **Authors**: *Chang Chen, Min Li, Zhihua Wu, Dianhai Yu, Chao Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8b465dd58ac50e1b0b22894fd581f62f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8b465dd58ac50e1b0b22894fd581f62f-Abstract-Conference.html)

        **Abstract**:

        Sparsely gated Mixture-of-Expert (MoE) has demonstrated its effectiveness in scaling up deep neural networks to an extreme scale. Despite that numerous efforts have been made to improve the performance of MoE from the model design or system optimization perspective, existing MoE dispatch patterns are still not able to fully exploit the underlying heterogeneous network environments. In this paper, we propose TA-MoE, a topology-aware routing strategy for large-scale MoE trainging, from a model-system co-design perspective, which can dynamically adjust the MoE dispatch pattern according to the network topology. Based on communication modeling, we abstract the dispatch problem into an optimization objective and obtain the approximate dispatch pattern under different topologies. On top of that, we design a topology-aware auxiliary loss, which can adaptively route the data to fit in the underlying topology without sacrificing the model accuracy. Experiments show that TA-MoE can substantially outperform its counterparts on various hardware and model configurations, with roughly 1.01x-1.61x, 1.01x-4.77x, 1.25x-1.54x improvements over the popular DeepSpeed-MoE, FastMoE and FasterMoE systems.

        ----

        ## [1611] A Combinatorial Perspective on the Optimization of Shallow ReLU Networks

        **Authors**: *Michael Matena, Colin Raffel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8b8fe72f3193fe78ac353ebcc686b395-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8b8fe72f3193fe78ac353ebcc686b395-Abstract-Conference.html)

        **Abstract**:

        The NP-hard problem of optimizing a shallow ReLU network can be characterized as a combinatorial search over each training exampleâ€™s activation pattern followed by a constrained convex problem given a fixed set of activation patterns. We explore the implications of this combinatorial aspect of ReLU optimization in this work. We show that it can be naturally modeled via a geometric and combinatoric object known as a zonotope with its vertex set isomorphic to the set of feasible activation patterns. This assists in analysis and provides a foundation for further research. We demonstrate its usefulness when we explore the sensitivity of the optimal loss to perturbations of the training data. Later we discuss methods of zonotope vertex selection and its relevance to optimization. Overparameterization assists in training by making a randomly chosen vertex more likely to contain a good solution. We then introduce a novel polynomial-time vertex selection procedure that provably picks a vertex containing the global optimum using only double the minimum number of parameters required to fit the data. We further introduce a local greedy search heuristic over zonotope vertices and demonstrate that it outperforms gradient descent on underparameterized problems.

        ----

        ## [1612] Large Language Models are Zero-Shot Reasoners

        **Authors**: *Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, Yusuke Iwasawa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8bb0d291acd4acf06ef112099c16f326-Abstract-Conference.html)

        **Abstract**:

        Pretrained large language models (LLMs) are widely used in many sub-fields of natural language processing (NLP) and generally known as excellent few-shot learners with task-specific exemplars. Notably, chain of thought (CoT) prompting, a recent technique for eliciting complex multi-step reasoning through step-by-step answer examples, achieved the state-of-the-art performances in arithmetics and symbolic reasoning, difficult system-2 tasks that do not follow the standard scaling laws for LLMs. While these successes are often attributed to LLMs' ability for few-shot learning, we show that LLMs are decent zero-shot reasoners by simply adding ``Let's think step by step'' before each answer. Experimental results demonstrate that our Zero-shot-CoT, using the same single prompt template, significantly outperforms zero-shot LLM performances on diverse benchmark reasoning tasks including arithmetics (MultiArith, GSM8K, AQUA-RAT, SVAMP), symbolic reasoning (Last Letter, Coin Flip), and other logical reasoning tasks (Date Understanding, Tracking Shuffled Objects),  without any hand-crafted few-shot examples, e.g. increasing the accuracy on MultiArith from 17.7% to 78.7% and GSM8K from 10.4% to 40.7% with large-scale InstructGPT model (text-davinci-002), as well as similar magnitudes of improvements with another off-the-shelf large model, 540B parameter PaLM. The versatility of this single prompt across very diverse reasoning tasks hints at untapped and understudied fundamental zero-shot capabilities of LLMs, suggesting high-level, multi-task broad cognitive capabilities may be extracted by simple prompting. We hope our work not only serves as the minimal strongest zero-shot baseline for the challenging reasoning benchmarks, but also highlights the importance of carefully exploring and analyzing the enormous zero-shot knowledge hidden inside LLMs before crafting finetuning datasets or few-shot exemplars.

        ----

        ## [1613] Trading off Utility, Informativeness, and Complexity in Emergent Communication

        **Authors**: *Mycal Tucker, Roger Levy, Julie A. Shah, Noga Zaslavsky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8bb5f66371c7e4cbf6c223162c62c0f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8bb5f66371c7e4cbf6c223162c62c0f4-Abstract-Conference.html)

        **Abstract**:

        Emergent communication (EC) research often focuses on optimizing task-specific utility as a driver for communication. However, there is increasing evidence that human languages are shaped by task-general communicative constraints and evolve under pressure to optimize the Information Bottleneck (IB) tradeoff between the informativeness and complexity of the lexicon. Here, we integrate these two approaches by trading off utility, informativeness, and complexity in EC. To this end, we propose Vector-Quantized Variational Information Bottleneck (VQ-VIB), a method for training neural agents to encode inputs into discrete signals embedded in a continuous space. We evaluate our approach in multi-agent reinforcement learning settings and in color reference games and show that: (1) VQ-VIB agents can continuously adapt to changing communicative needs and, in the color domain, align with human languages; (2) the emergent VQ-VIB embedding spaces are semantically meaningful and perceptually grounded; and (3) encouraging informativeness leads to faster convergence rates and improved utility, both in VQ-VIB and in prior neural architectures for symbolic EC, with VQ-VIB achieving higher utility for any given complexity. This work offers a new framework for EC that is grounded in information-theoretic principles that are believed to characterize human language evolution and that may facilitate human-agent interaction.

        ----

        ## [1614] FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation

        **Authors**: *Mehmet Ozgur Turkoglu, Alexander Becker, Hüseyin Anil Gündüz, Mina Rezaei, Bernd Bischl, Rodrigo Caye Daudt, Stefano D'Aronco, Jan D. Wegner, Konrad Schindler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html)

        **Abstract**:

        The ability to estimate epistemic uncertainty is often crucial when deploying machine learning in the real world, but modern methods often produce overconfident, uncalibrated uncertainty predictions. A common approach to quantify epistemic uncertainty, usable across a wide class of prediction models, is to train a model ensemble. In a naive implementation, the ensemble approach has high computational cost and high memory demand. This challenges in particular modern deep learning, where even a single deep network is already demanding in terms of compute and memory, and has given rise to a number of attempts to emulate the model ensemble without actually instantiating separate ensemble members. We introduce FiLM-Ensemble, a deep, implicit ensemble method based on the concept of Feature-wise Linear Modulation (FiLM). That technique was originally developed for multi-task learning, with the aim of decoupling different tasks. We show that the idea can be extended to uncertainty quantification: by modulating the network activations of a single deep network with FiLM, one obtains a model ensemble with high diversity, and consequently well-calibrated estimates of epistemic uncertainty, with low computational overhead in comparison. Empirically, FiLM-Ensemble outperforms other implicit ensemble methods, and it comes very close to the upper bound of an explicit ensemble of networks (sometimes even beating it), at a fraction of the memory cost.

        ----

        ## [1615] Meta-DMoE: Adapting to Domain Shift by Meta-Distillation from Mixture-of-Experts

        **Authors**: *Tao Zhong, Zhixiang Chi, Li Gu, Yang Wang, Yuanhao Yu, Jin Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8bd4f1dbc7a70c6b80ce81b8b4fdc0b2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8bd4f1dbc7a70c6b80ce81b8b4fdc0b2-Abstract-Conference.html)

        **Abstract**:

        In this paper, we tackle the problem of domain shift. Most existing methods perform training on multiple source domains using a single model, and the same trained model is used on all unseen target domains. Such solutions are sub-optimal as each target domain exhibits its own specialty, which is not adapted. Furthermore, expecting single-model training to learn extensive knowledge from multiple source domains is counterintuitive. The model is more biased toward learning only domain-invariant features and may result in negative knowledge transfer. In this work, we propose a novel framework for unsupervised test-time adaptation, which is formulated as a knowledge distillation process to address domain shift. Specifically, we incorporate Mixture-of-Experts (MoE) as teachers, where each expert is separately trained on different source domains to maximize their specialty. Given a test-time target domain, a small set of unlabeled data is sampled to query the knowledge from MoE. As the source domains are correlated to the target domains, a transformer-based aggregator then combines the domain knowledge by examining the interconnection among them. The output is treated as a supervision signal to adapt a student prediction network toward the target domain. We further employ meta-learning to enforce the aggregator to distill positive knowledge and the student network to achieve fast adaptation. Extensive experiments demonstrate that the proposed method outperforms the state-of-the-art and validates the effectiveness of each proposed component. Our code is available at https://github.com/n3il666/Meta-DMoE.

        ----

        ## [1616] Beyond Time-Average Convergence: Near-Optimal Uncoupled Online Learning via Clairvoyant Multiplicative Weights Update

        **Authors**: *Georgios Piliouras, Ryann Sim, Stratis Skoulakis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8bd5148caced2d73cea7b6961a874a49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8bd5148caced2d73cea7b6961a874a49-Abstract-Conference.html)

        **Abstract**:

        In this paper we provide a novel and simple algorithm, Clairvoyant Multiplicative Weights Updates (CMWU), for convergence to \textit{Coarse Correlated Equilibria} (CCE) in general games. CMWU effectively corresponds to the standard MWU algorithm but where all agents, when updating their mixed strategies, use the payoff profiles based on tomorrow's behavior, i.e. the agents are clairvoyant. CMWU achieves constant regret of $\ln(m)/\eta$ in all normal-form games with m actions and fixed step-sizes $\eta$. Although CMWU encodes in its definition a fixed point computation, which in principle could result in dynamics that are neither computationally efficient nor uncoupled, we show that both of these issues can be largely circumvented. Specifically, as long as the step-size $\eta$ is upper bounded by $\frac{1}{(n-1)V}$, where $n$ is the number of agents and $[0,V]$ is the payoff range, then the CMWU updates can be computed linearly fast via a contraction map. This implementation results in an uncoupled online learning dynamic that admits a $O(\log T)$-sparse sub-sequence where each agent experiences at most $O(nV\log m)$ regret. This implies that the CMWU dynamics converge with rate $O(nV \log m \log T / T)$ to a CCE and improves on the current state-of-the-art convergence rate.

        ----

        ## [1617] Meta-Reward-Net: Implicitly Differentiable Reward Learning for Preference-based Reinforcement Learning

        **Authors**: *Runze Liu, Fengshuo Bai, Yali Du, Yaodong Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8be9c134bb193d8bd3827d4df8488228-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8be9c134bb193d8bd3827d4df8488228-Abstract-Conference.html)

        **Abstract**:

        Setting up a well-designed reward function has been challenging for many reinforcement learning applications. Preference-based reinforcement learning (PbRL) provides a new framework that avoids reward engineering by leveraging human preferences (i.e., preferring apples over oranges) as the reward signal. Therefore, improving the efficacy of data usage for preference data becomes critical. In this work, we propose Meta-Reward-Net (MRN), a data-efficient PbRL framework that incorporates bi-level optimization for both reward and policy learning. The key idea of MRN is to adopt the performance of the Q-function as the learning target. Based on this, MRN learns the Q-function and the policy in the inner level while updating the reward function adaptively according to the performance of the Q-function on the preference data in the outer level. Our experiments on robotic simulated manipulation tasks and locomotion tasks demonstrate that MRN outperforms prior methods in the case of few preference labels and significantly improves data efficiency, achieving state-of-the-art in preference-based RL. Ablation studies further demonstrate that MRN learns a more accurate Q-function compared to prior work and shows obvious advantages when only a small amount of human feedback is available. The source code and videos of this project are released at https://sites.google.com/view/meta-reward-net.

        ----

        ## [1618] One-shot Neural Backdoor Erasing via Adversarial Weight Masking

        **Authors**: *Shuwen Chai, Jinghui Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c0f7107ab85892ccf51f0a814957af1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c0f7107ab85892ccf51f0a814957af1-Abstract-Conference.html)

        **Abstract**:

        Recent studies show that despite achieving high accuracy on a number of real-world applications, deep neural networks (DNNs) can be backdoored: by injecting triggered data samples into the training dataset, the adversary can mislead the trained model into classifying any test data to the target class as long as the trigger pattern is presented. To nullify such backdoor threats, various methods have been proposed. Particularly, a line of research aims to purify the potentially compromised model. However, one major limitation of this line of work is the requirement to access sufficient original training data: the purifying performance is a lot worse when the available training data is limited. In this work, we propose Adversarial Weight Masking (AWM), a novel method capable of erasing the neural backdoors even in the one-shot setting. The key idea behind our method is to formulate this into a min-max optimization problem: first, adversarially recover the non-robust perturbation patterns and then (soft) mask the network weights that are sensitive to the recovered patterns. Comprehensive evaluations of several benchmark datasets suggest that AWM can largely improve the purifying effects over other state-of-the-art methods on various available training dataset sizes.

        ----

        ## [1619] Revisiting Neural Scaling Laws in Language and Vision

        **Authors**: *Ibrahim M. Alabdulmohsin, Behnam Neyshabur, Xiaohua Zhai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html)

        **Abstract**:

        The remarkable progress in deep learning in recent years is largely driven by improvements in scale, where bigger models are trained on larger datasets for longer schedules. To predict the benefit of scale empirically, we argue for a more rigorous methodology based on the extrapolation loss, instead of reporting the best-fitting (interpolating) parameters. We then present a recipe for estimating scaling law parameters reliably from learning curves. We demonstrate that it extrapolates more accurately than previous methods in a wide range of architecture families across several domains, including image classification, neural machine translation (NMT) and  language modeling, in addition to tasks from the BIG-Bench evaluation benchmark. Finally, we release a benchmark dataset comprising of 90 evaluation tasks to facilitate research in this domain.

        ----

        ## [1620] Efficient Phi-Regret Minimization in Extensive-Form Games via Online Mirror Descent

        **Authors**: *Yu Bai, Chi Jin, Song Mei, Ziang Song, Tiancheng Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c263f70550cc7d69dba3fc170a23e77-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c263f70550cc7d69dba3fc170a23e77-Abstract-Conference.html)

        **Abstract**:

        A conceptually appealing approach for learning Extensive-Form Games (EFGs) is to convert them to Normal-Form Games (NFGs). This approach enables us to directly translate state-of-the-art techniques and analyses in NFGs to learning EFGs, but typically suffers from computational intractability due to the exponential blow-up of the game size introduced by the conversion. In this paper, we address this problem in natural and important setups for the \emph{$\Phi$-Hedge} algorithm---A generic algorithm capable of learning a large class of equilibria for NFGs. We show that $\Phi$-Hedge can be directly used to learn Nash Equilibria (zero-sum settings), Normal-Form Coarse Correlated Equilibria (NFCCE), and Extensive-Form Correlated Equilibria (EFCE) in EFGs. We prove that, in those settings, the \emph{$\Phi$-Hedge} algorithms are equivalent to standard Online Mirror Descent (OMD) algorithms for EFGs with suitable dilated regularizers, and run in polynomial time. This new connection further allows us to design and analyze a new class of OMD algorithms based on modifying its log-partition function. In particular, we design an improved algorithm with balancing techniques that achieves a sharp $\widetilde{\mathcal{O}}(\sqrt{XAT})$ EFCE-regret under bandit-feedback in an EFG with $X$ information sets, $A$ actions, and $T$ episodes. To our best knowledge, this is the first such rate and matches the information-theoretic lower bound.

        ----

        ## [1621] Long Range Graph Benchmark

        **Authors**: *Vijay Prakash Dwivedi, Ladislav Rampásek, Michael Galkin, Ali Parviz, Guy Wolf, Anh Tuan Luu, Dominique Beaini*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Graph Neural Networks (GNNs) that are based on the message passing (MP) paradigm generally exchange information between 1-hop neighbors to build node representations at each layer. In principle, such networks are not able to capture long-range interactions (LRI) that may be desired or necessary for learning a given task on graphs. Recently, there has been an increasing interest in development of Transformer-based methods for graphs that can consider full node connectivity beyond the original sparse structure, thus enabling the modeling of LRI. However, MP-GNNs that simply rely on 1-hop message passing often fare better in several existing graph benchmarks when combined with positional feature representations, among other innovations, hence limiting the perceived utility and ranking of Transformer-like architectures. Here, we present the Long Range Graph Benchmark (LRGB) with 5 graph learning datasets: $\texttt{PascalVOC-SP}$, $\texttt{COCO-SP}$, $\texttt{PCQM-Contact}$, $\texttt{Peptides-func}$ and $\texttt{Peptides-struct}$ that arguably require LRI reasoning to achieve strong performance in a given task. We benchmark both baseline GNNs and Graph Transformer networks to verify that the models which capture long-range dependencies perform significantly better on these tasks. Therefore, these datasets are suitable for benchmarking and exploration of MP GNNs and Graph Transformer architectures that are intended to capture LRI.

        ----

        ## [1622] On the inability of Gaussian process regression to optimally learn compositional functions

        **Authors**: *Matteo Giordano, Kolyan Ray, Johannes Schmidt-Hieber*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c420176b45e923cf99dee1d7356a763-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c420176b45e923cf99dee1d7356a763-Abstract-Conference.html)

        **Abstract**:

        We rigorously prove that deep Gaussian process priors can outperform Gaussian process priors if the target function has a compositional structure. To this end, we study information-theoretic lower bounds for posterior contraction rates for Gaussian process regression in a continuous regression model. We show that if the true function is a generalized additive function, then the posterior based on any mean-zero Gaussian process can only recover the truth at a rate that is strictly slower than the minimax rate by a factor that is polynomially suboptimal in the sample size $n$.

        ----

        ## [1623] Active Learning Through a Covering Lens

        **Authors**: *Ofer Yehuda, Avihu Dekel, Guy Hacohen, Daphna Weinshall*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html)

        **Abstract**:

        Deep active learning aims to reduce the annotation cost for the training of deep models, which is notoriously data-hungry. Until recently, deep active learning methods were ineffectual in the low-budget regime, where only a small number of examples are annotated. The situation has been alleviated by recent advances in representation and self-supervised learning, which impart the geometry of the data representation with rich information about the points. Taking advantage of this progress, we study the problem of subset selection for annotation through a “covering” lens, proposing ProbCover – a new active learning algorithm for the low budget regime, which seeks to maximize Probability Coverage. We then describe a dual way to view the proposed formulation, from which one can derive strategies suitable for the high budget regime of active learning, related to existing methods like Coreset. We conclude with extensive experiments, evaluating ProbCover in the low-budget regime. We show that our principled active learning strategy improves the state-of-the-art in the low-budget regime in several image recognition benchmarks. This method is especially beneficial in the semi-supervised setting, allowing state-of-the-art semi-supervised methods to match the performance of fully supervised methods, while using much fewer labels nonetheless. Code is available at https://github.com/avihu111/TypiClust.

        ----

        ## [1624] Uplifting Bandits

        **Authors**: *Yu-Guan Hsieh, Shiva Prasad Kasiviswanathan, Branislav Kveton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c69d058c83362ee123b5e2c37d6296a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c69d058c83362ee123b5e2c37d6296a-Abstract-Conference.html)

        **Abstract**:

        We introduce a new multi-armed bandit model where the reward is a sum of multiple random variables, and each action only alters the distributions of some of these variables. Upon taking an action, the agent observes the realizations of all variables. This model is motivated by marketing campaigns and recommender systems, where the variables represent outcomes on individual customers, such as clicks. We propose UCB-style algorithms that estimate the uplifts of the actions over a baseline. We study multiple variants of the problem, including when the baseline and affected variables are unknown, and prove sublinear regret bounds for all of these. In addition, we provide regret lower bounds that justify the necessity of our modeling assumptions. Experiments on synthetic and real-world datasets demonstrate the benefit of methods that estimate the uplifts over policies that do not use this structure.

        ----

        ## [1625] Training Uncertainty-Aware Classifiers with Conformalized Deep Learning

        **Authors**: *Bat-Sheva Einbinder, Yaniv Romano, Matteo Sesia, Yanfei Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks are powerful tools to detect hidden patterns in data and leverage them to make predictions, but they are not designed to understand uncertainty and estimate reliable probabilities. In particular, they tend to be overconfident. We begin to address this problem in the context of multi-class classification by developing a novel training algorithm producing models with more dependable uncertainty estimates, without sacrificing predictive power. The idea is to mitigate overconfidence by minimizing a loss function, inspired by advances in conformal inference, that quantifies model uncertainty by carefully leveraging hold-out data. Experiments with synthetic and real data demonstrate this method can lead to smaller conformal prediction sets with higher conditional coverage, after exact calibration with hold-out data, compared to state-of-the-art alternatives.

        ----

        ## [1626] A general approximation lower bound in $L^p$ norm, with applications to feed-forward neural networks

        **Authors**: *El Mehdi Achour, Armand Foucault, Sébastien Gerchinovitz, François Malgouyres*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8caa10fb546ae38b3d3f0d32ecc866f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8caa10fb546ae38b3d3f0d32ecc866f0-Abstract-Conference.html)

        **Abstract**:

        We study the fundamental limits to the expressive power of neural networks. Given two sets $F$, $G$ of real-valued functions, we first prove a general lower bound on how well functions in $F$ can be approximated in $L^p(\mu)$ norm by functions in $G$, for any $p \geq 1$ and any probability measure $\mu$. The lower bound depends on the packing number of $F$, the range of $F$, and the fat-shattering dimension of $G$. We then instantiate this bound to the case where $G$ corresponds to a piecewise-polynomial feedforward neural network, and describe in details the application to two sets $F$: Hölder balls and multivariate monotonic functions. Beside matching (known or new) upper bounds up to log factors, our lower bounds shed some light on the similarities or differences between approximation in $L^p$ norm or in sup norm, solving an open question by DeVore et al. (2021). Our proof strategy differs from the sup norm case and uses a key probability result of Mendelson (2002).

        ----

        ## [1627] EnvPool: A Highly Parallel Reinforcement Learning Environment Execution Engine

        **Authors**: *Jiayi Weng, Min Lin, Shengyi Huang, Bo Liu, Denys Makoviichuk, Viktor Makoviychuk, Zichen Liu, Yufan Song, Ting Luo, Yukun Jiang, Zhongwen Xu, Shuicheng Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8caaf08e49ddbad6694fae067442ee21-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        There has been significant progress in developing reinforcement learning (RL) training systems. Past works such as IMPALA, Apex, Seed RL, Sample Factory, and others, aim to improve the system's overall throughput. In this paper, we aim to address a common bottleneck in the RL training system, i.e., parallel environment execution, which is often the slowest part of the whole system but receives little attention. With a curated design for paralleling RL environments, we have improved the RL environment simulation speed across different hardware setups, ranging from a laptop and a modest workstation, to a high-end machine such as NVIDIA DGX-A100. On a high-end machine, EnvPool achieves one million frames per second for the environment execution on Atari environments and three million frames per second on MuJoCo environments. When running EnvPool on a laptop, the speed is 2.8x that of the Python subprocess. Moreover, great compatibility with existing RL training libraries has been demonstrated in the open-sourced community, including CleanRL, rl_games, DeepMind Acme, etc. Finally, EnvPool allows researchers to iterate their ideas at a much faster pace and has great potential to become the de facto RL environment execution engine. Example runs show that it only takes five minutes to train agents to play Atari Pong and MuJoCo Ant on a laptop.  EnvPool is open-sourced at https://github.com/sail-sg/envpool.

        ----

        ## [1628] Generative Visual Prompt: Unifying Distributional Control of Pre-Trained Generative Models

        **Authors**: *Chen Henry Wu, Saman Motamed, Shaunak Srivastava, Fernando De la Torre*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8cb1c53863b290ee09b94d17f16ef355-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8cb1c53863b290ee09b94d17f16ef355-Abstract-Conference.html)

        **Abstract**:

        Generative models (e.g., GANs, diffusion models) learn the underlying data distribution in an unsupervised manner. However, many applications of interest require sampling from a particular region of the output space or sampling evenly over a range of characteristics. For efficient sampling in these scenarios, we propose Generative Visual Prompt (PromptGen), a framework for distributional control over pre-trained generative models by incorporating knowledge of other off-the-shelf models. PromptGen defines control as energy-based models (EBMs) and samples images in a feed-forward manner by approximating the EBM with invertible neural networks, avoiding optimization at inference. Our experiments demonstrate how PromptGen can efficiently sample from several unconditional generative models (e.g., StyleGAN2, StyleNeRF, diffusion autoencoder, NVAE) in a controlled or/and de-biased manner using various off-the-shelf models: (1) with the CLIP model as control, PromptGen can sample images guided by text, (2) with image classifiers as control, PromptGen can de-bias generative models across a set of attributes or attribute combinations, and (3) with inverse graphics models as control, PromptGen can sample images of the same identity in different poses. (4) Finally, PromptGen reveals that the CLIP model shows a "reporting bias" when used as control, and PromptGen can further de-bias this controlled distribution in an iterative manner. The code is available at https://github.com/ChenWu98/Generative-Visual-Prompt.

        ----

        ## [1629] Implicit Warping for Animation with Image Sets

        **Authors**: *Arun Mallya, Ting-Chun Wang, Ming-Yu Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8cb31912235561112339f04903657f72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8cb31912235561112339f04903657f72-Abstract-Conference.html)

        **Abstract**:

        We present a new implicit warping framework for image animation using sets of source images through the transfer of motion of a driving video. A single cross-modal attention layer is used to find correspondences between the source images and the driving image, choose the most appropriate features from different source images, and warp the selected features. This is in contrast to the existing methods that use explicit flow-based warping, which is designed for animation using a single source and does not extend well to multiple sources. The pick-and-choose capability of our framework helps it achieve state-of-the-art results on multiple datasets for image animation using both single and multiple source images.

        ----

        ## [1630] FNeVR: Neural Volume Rendering for Face Animation

        **Authors**: *Bohan Zeng, Boyu Liu, Hong Li, Xuhui Liu, Jianzhuang Liu, Dapeng Chen, Wei Peng, Baochang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8cc7e1509fbfee9cabaacd3ab0bfe2b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8cc7e1509fbfee9cabaacd3ab0bfe2b1-Abstract-Conference.html)

        **Abstract**:

        Face animation, one of the hottest topics in computer vision, has achieved a promising performance with the help of generative models. However, it remains a critical challenge to generate identity preserving and photo-realistic images due to the sophisticated motion deformation and complex facial detail modeling. To address these problems, we propose a Face Neural Volume Rendering (FNeVR) network to fully explore the potential of 2D motion warping and 3D volume rendering in a unified framework. In FNeVR, we design a 3D Face Volume Rendering (FVR) module to enhance the facial details for image rendering. Specifically, we first extract 3D information with a well designed architecture, and then introduce an orthogonal adaptive ray-sampling module for efficient rendering. We also design a lightweight pose editor, enabling FNeVR to edit the facial pose in a simple yet effective way. Extensive experiments show that our FNeVR obtains the best overall quality and performance on widely used talking-head benchmarks.

        ----

        ## [1631] Ontologue: Declarative Benchmark Construction for Ontological Multi-Label Classification

        **Authors**: *Sean Yang, Bernease Herman, Bill Howe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We describe a customizable benchmark for hierarchical and ontological multi-label classification, a task where labels are equipped with a graph structure and data items can be assigned multiple labels.  We find that current benchmarks do not adequately represent the problem space, casting doubt on the generalizability of current results. We consider three dimensions of the problem space: context (availability of rich features on the data and labels), distribution of labels over data, and graph structure. For context, the lack of complex features on the labels (and in some cases, the data) artificially prevent the use of modern representation learning techniques as an appropriate baseline.  For distribution, we find the long tail of labels over data constitute a few-shot learning problem that artificially confounds the results: for most common benchmarks, over 40% of the labels have fewer than 5 data points in the training set.  For structure, we find that the correlation between performance and the height of the tree can explain some of the variation in performance, informing practical utility. In this paper, we demonstrate how the lack of diversity in benchmarks can confound performance analysis, then present a declarative query system called Ontologue for generating custom benchmarks with specific properties, then use this system to design 4 new benchmarks extracted from DBPedia that better represent the problem space. We evaluate state-of-the-art algorithms on both existing and new benchmarks and show that the performance conclusions can vary significantly depending on the dimensions we consider.  We intend the system and derived benchmarks to improve the analysis of generalizability for these problems.

        ----

        ## [1632] Make an Omelette with Breaking Eggs: Zero-Shot Learning for Novel Attribute Synthesis

        **Authors**: *Yu Hsuan Li, Tzu-Yin Chao, Ching-Chun Huang, Pin-Yu Chen, Wei-Chen Chiu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8cf3760422b9d4505589a97c8f9569e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8cf3760422b9d4505589a97c8f9569e7-Abstract-Conference.html)

        **Abstract**:

        Most of the existing algorithms for zero-shot classification problems typically rely on the attribute-based semantic relations among categories to realize the classification of novel categories without observing any of their instances. However, training the zero-shot classification models still requires attribute labeling for each class (or even instance) in the training dataset, which is also expensive. To this end, in this paper, we bring up a new problem scenario: ''Can we derive zero-shot learning for novel attribute detectors/classifiers and use them to automatically annotate the dataset for labeling efficiency?'' Basically, given only a small set of detectors that are learned to recognize some manually annotated attributes (i.e., the seen attributes), we aim to synthesize the detectors of novel attributes in a zero-shot learning manner. Our proposed method, Zero-Shot Learning for Attributes (ZSLA), which is the first of its kind to the best of our knowledge, tackles this new research problem by applying the set operations to first decompose the seen attributes into their basic attributes and then recombine these basic attributes into the novel ones. Extensive experiments are conducted to verify the capacity of our synthesized detectors for accurately capturing the semantics of the novel attributes and show their superior performance in terms of detection and localization compared to other baseline approaches. Moreover, we demonstrate the application of automatic annotation using our synthesized detectors on Caltech-UCSD Birds-200-2011 dataset. Various generalized zero-shot classification algorithms trained upon the dataset re-annotated by ZSLA shows comparable performance with those trained with the manual ground-truth annotations.

        ----

        ## [1633] Exploiting Semantic Relations for Glass Surface Detection

        **Authors**: *Jiaying Lin, Yuen Hei Yeung, Rynson W. H. Lau*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d162f48c816af5f8c114eb437e8b28b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d162f48c816af5f8c114eb437e8b28b-Abstract-Conference.html)

        **Abstract**:

        Glass surfaces are omnipresent in our daily lives and often go unnoticed by the majority of us. While humans are generally able to infer their locations and thus avoid collisions, it can be difficult for current object detection systems to handle them due to the transparent nature of glass surfaces. Previous methods approached the problem by extracting global context information to obtain priors such as object boundaries and reflections. However, their performances cannot be guaranteed when these deterministic features are not available. We observe that humans often reason through the semantic context of the environment, which offers insights into the categories of and proximity between entities that are expected to appear in the surrounding. For example, the odds of co-occurrence of glass windows with walls and curtains are generally higher than that with other objects such as cars and trees, which have relatively less semantic relevance. Based on this observation, we propose a model ('GlassSemNet') that integrates the contextual relationship of the scenes for glass surface detection with two novel modules: (1) Scene Aware Activation (SAA) Module to adaptively filter critical channels with respect to spatial and semantic features, and (2) Context Correlation Attention (CCA) Module to progressively learn the contextual correlations among objects both spatially and semantically. In addition, we propose a large-scale glass surface detection dataset named {\it Glass Surface Detection - Semantics} ('GSD-S'), which contains 4,519 real-world RGB glass surface images from diverse real-world scenes with detailed annotations for both glass surface detection and semantic segmentation. Experimental results show that our model outperforms contemporary works, especially with 42.6\% MAE improvement on our proposed GSD-S dataset. Code, dataset, and models are available at https://jiaying.link/neurips2022-gsds/

        ----

        ## [1634] Differentially Private Generalized Linear Models Revisited

        **Authors**: *Raman Arora, Raef Bassily, Cristóbal Guzmán, Michael Menart, Enayat Ullah*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d321ebb82b58987509b8624cbb85d65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d321ebb82b58987509b8624cbb85d65-Abstract-Conference.html)

        **Abstract**:

        We study the problem of $(\epsilon,\delta)$-differentially private learning of linear predictors with convex losses. We provide results for two subclasses of loss functions. The first case is when the loss is smooth and non-negative but not necessarily Lipschitz (such as the squared loss). For this case, we establish an  upper bound on the excess population risk of $\tilde{O}\left(\frac{\Vert w^*\Vert}{\sqrt{n}} + \min\left\{\frac{\Vert w^* \Vert^2}{(n\epsilon)^{2/3}},\frac{\sqrt{d}\Vert w^*\Vert^2}{n\epsilon}\right\}\right)$, where $n$ is the number of samples, $d$ is the dimension of the problem, and $w^*$ is the minimizer of the population risk. Apart from the dependence on $\Vert w^\ast\Vert$, our bound is essentially tight in all parameters. In particular, we show a lower bound of $\tilde{\Omega}\left(\frac{1}{\sqrt{n}} + {\min\left\{\frac{\Vert w^*\Vert^{4/3}}{(n\epsilon)^{2/3}}, \frac{\sqrt{d}\Vert w^*\Vert}{n\epsilon}\right\}}\right)$. We also revisit the previously studied case of Lipschitz losses \cite{SSTT21}.  For this case, we close the gap in the existing work and show that the optimal rate is (up to log factors) $\Theta\left(\frac{\Vert w^*\Vert}{\sqrt{n}} + \min\left\{\frac{\Vert w^*\Vert}{\sqrt{n\epsilon}},\frac{\sqrt{\text{rank}}\Vert w^*\Vert}{n\epsilon}\right\}\right)$, where $\text{rank}$ is the rank of the design matrix. This improves over existing work in the high privacy regime. Finally, our algorithms involve a private model selection approach that we develop to enable attaining the stated rates without a-priori knowledge of $\Vert w^*\Vert$.

        ----

        ## [1635] GREED: A Neural Framework for Learning Graph Distance Functions

        **Authors**: *Rishabh Ranjan, Siddharth Grover, Sourav Medya, Venkatesan T. Chakaravarthy, Yogish Sabharwal, Sayan Ranu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d492b8a6201d83d1015af9e264f0bf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d492b8a6201d83d1015af9e264f0bf2-Abstract-Conference.html)

        **Abstract**:

        Similarity search in graph databases is one of the most fundamental operations in graph analytics. Among various distance functions, graph and subgraph edit distances (GED and SED respectively) are two of the most popular and expressive measures. Unfortunately, exact computations for both are NP-hard. To overcome this computational bottleneck, neural approaches to learn and predict edit distance in polynomial time have received much interest. While considerable progress has been made, there exist limitations that need to be addressed. First, the efficacy of an approximate distance function lies not only in its approximation accuracy, but also in the preservation of its properties. To elaborate, although GED is a metric, its neural approximations do not provide such a guarantee. This prohibits their usage in higher order tasks that rely on metric distance functions, such as clustering or indexing. Second, several existing frameworks for GED do not extend to SED due to SED being asymmetric. In this work, we design a novel siamese graph neural network called Greed, which through a carefully crafted inductive bias, learns GED and SED in a property-preserving manner. Through extensive experiments across $10$ real graph datasets containing up to $7$ million edges, we establish that Greed is not only more accurate than the state of the art, but also up to $3$ orders of magnitude faster. Even more significantly, due to preserving the triangle inequality, the generated embeddings are indexable and consequently, even in a CPU-only environment, Greed is up to $50$ times faster than GPU-powered computations of the closest baseline.

        ----

        ## [1636] Domain Adaptation under Open Set Label Shift

        **Authors**: *Saurabh Garg, Sivaraman Balakrishnan, Zachary C. Lipton*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d5f526a31d3731a30eb58d5874cf5b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d5f526a31d3731a30eb58d5874cf5b1-Abstract-Conference.html)

        **Abstract**:

        We introduce the problem of domain adaptation under Open Set Label Shift (OSLS), where the label distribution can change arbitrarily and a new class may arrive during deployment, but the class-conditional distributions $p(x|y)$ are domain-invariant. OSLS subsumes domain adaptation under label shift and Positive-Unlabeled (PU) learning. The learner's goals here are two-fold: (a) estimate the target label distribution, including the novel class; and (b) learn a target classifier. First, we establish the necessary and sufficient for identifying these quantities. Second, motivated by advances in label shift and PU learning, we propose practical methods for both tasks that leverage black-box predictors. Unlike typical Open Set Domain Adaptation (OSDA) problems, which tend to be ill-posed and amenable only to heuristics, OSLS offers a well-posed problem amenable to more principled machinery. Experiments across numerous semi-synthetic benchmarks on vision, language, and medical datasets demonstrate that our methods consistently outperform OSDA baselines, achieving $10$--$25\%$ improvements in target domain accuracy. Finally, we analyze the proposed methods, establishing finite-sample convergence to the true label marginal and convergence to optimal classifier for linear models in a Gaussian setup. Code is available at https://github.com/acmi-lab/Open-Set-Label-Shift.

        ----

        ## [1637] Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning

        **Authors**: *Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html)

        **Abstract**:

        Recent studies reveal that a well-trained deep reinforcement learning (RL) policy can be particularly vulnerable to adversarial perturbations on input observations. Therefore, it is crucial to train RL agents that are robust against any attacks with a bounded budget. Existing robust training methods in deep RL either treat correlated steps separately, ignoring the robustness of long-term rewards, or train the agents and RL-based attacker together, doubling the computational burden and sample complexity of the training process. In this work, we propose a strong and efficient robust training framework for RL, named Worst-case-aware Robust RL (WocaR-RL) that directly estimates and optimizes the worst-case reward of a policy under bounded l_p attacks without requiring extra samples for learning an attacker. Experiments on multiple environments show that WocaR-RL achieves state-of-the-art performance under various strong attacks, and obtains significantly higher training efficiency than prior state-of-the-art robust training methods. The code of this work is available at https://github.com/umd-huang-lab/WocaR-RL.

        ----

        ## [1638] Alleviating "Posterior Collapse" in Deep Topic Models via Policy Gradient

        **Authors**: *Yewen Li, Chaojie Wang, Zhibin Duan, Dongsheng Wang, Bo Chen, Bo An, Mingyuan Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8d7baf888ca264fd5f2b0d478882b6a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8d7baf888ca264fd5f2b0d478882b6a2-Abstract-Conference.html)

        **Abstract**:

        Deep topic models have been proven as a promising way to extract hierarchical latent representations from documents represented as high-dimensional bag-of-words vectors.However, the representation capability of existing deep topic models is still limited by the phenomenon of "posterior collapse", which has been widely criticized in deep generative models, resulting in the higher-level latent representations exhibiting similar or meaningless patterns.To this end, in this paper, we first develop a novel deep-coupling generative process for existing deep topic models, which incorporates skip connections into the generation of documents, enforcing strong links between the document and its multi-layer latent representations.After that, utilizing data augmentation techniques, we reformulate the deep-coupling generative process as a Markov decision process and develop a corresponding Policy Gradient (PG) based training algorithm, which can further alleviate the information reduction at higher layers.Extensive experiments demonstrate that our developed methods can effectively alleviate "posterior collapse" in deep topic models, contributing to providing higher-quality latent document representations.

        ----

        ## [1639] Tracking Functional Changes in Nonstationary Signals with Evolutionary Ensemble Bayesian Model for Robust Neural Decoding

        **Authors**: *Xinyun Zhu, Yu Qi, Gang Pan, Yueming Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html)

        **Abstract**:

        Neural signals are typical nonstationary data where the functional mapping between neural activities and the intentions (such as the velocity of movements) can occasionally change. Existing studies mostly use a fixed neural decoder, thus suffering from an unstable performance given neural functional changes. We propose a novel evolutionary ensemble framework (EvoEnsemble) to dynamically cope with changes in neural signals by evolving the decoder model accordingly. EvoEnsemble integrates evolutionary computation algorithms in a Bayesian framework where the fitness of models can be sequentially computed with their likelihoods according to the incoming data at each time slot, which enables online tracking of time-varying functions. Two strategies of evolve-at-changes and history-model-archive are designed to further improve efficiency and stability. Experiments with simulations and neural signals demonstrate that EvoEnsemble can track the changes in functions effectively thus improving the accuracy and robustness of neural decoding. The improvement is most significant in neural signals with functional changes.

        ----

        ## [1640] Zeroth-Order Hard-Thresholding: Gradient Error vs Expansivity

        **Authors**: *William de Vazelhes, Hualin Zhang, Huimin Wu, Xiaotong Yuan, Bin Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8de5384f522efff26884559599c09312-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8de5384f522efff26884559599c09312-Abstract-Conference.html)

        **Abstract**:

        $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem.To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with  a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions.   Importantly, we   reveal a conflict between  the deviation of  ZO estimators and  the expansivity of the hard-thresholding operator,  and provide a theoretical   minimal value of the number of random directions in ZO gradients. In addition,  we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings.  Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.

        ----

        ## [1641] Collaborative Linear Bandits with Adversarial Agents: Near-Optimal Regret Bounds

        **Authors**: *Aritra Mitra, Arman Adibi, George J. Pappas, Hamed Hassani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html)

        **Abstract**:

        We consider a linear stochastic bandit problem involving $M$ agents that can collaborate via a central server to minimize regret. A fraction $\alpha$ of these agents are adversarial and can act arbitrarily, leading to the following tension: while collaboration can potentially reduce regret, it can also disrupt the process of learning due to adversaries. In this work, we provide a fundamental understanding of this tension by designing new algorithms that balance the exploration-exploitation trade-off via carefully constructed robust confidence intervals. We also complement our algorithms with tight analyses. First, we develop a robust collaborative phased elimination algorithm that achieves $\tilde{O}\left(\alpha+ 1/\sqrt{M}\right) \sqrt{dT}$ regret for each good agent; here, $d$ is the model-dimension and $T$ is the horizon. For small $\alpha$, our result thus reveals a clear benefit of collaboration despite adversaries. Using an information-theoretic argument, we then prove a matching lower bound, thereby providing the first set of tight, near-optimal regret bounds for collaborative linear bandits with adversaries. Furthermore, by leveraging recent advances in high-dimensional robust statistics, we significantly extend our algorithmic ideas and results to (i) the generalized linear bandit model that allows for non-linear observation maps; and (ii) the contextual bandit setting that allows for time-varying feature vectors.

        ----

        ## [1642] Differentially Private Graph Learning via Sensitivity-Bounded Personalized PageRank

        **Authors**: *Alessandro Epasto, Vahab Mirrokni, Bryan Perozzi, Anton Tsitsulin, Peilin Zhong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html)

        **Abstract**:

        Personalized PageRank (PPR) is a fundamental tool in unsupervised learning of graph representations such as node ranking, labeling, and graph embedding. However, while data privacy is one of the most important recent concerns, existing PPR algorithms are not designed to protect user privacy. PPR is highly sensitive to the input graph edges: the difference of only one edge may cause a big change in the PPR vector, potentially leaking private user data.In this work, we propose an algorithm which outputs an approximate PPR and has provably bounded sensitivity to input edges. In addition, we prove that our algorithm achieves  similar accuracy to non-private algorithms when the input graph has large degrees. Our sensitivity-bounded PPR directly implies private algorithms for several tools of graph learning, such as, differentially private (DP) PPR ranking, DP node classification, and DP node embedding. To complement our theoretical analysis, we also empirically verify the practical performances of our algorithms.

        ----

        ## [1643] How Well Do Unsupervised Learning Algorithms Model Human Real-time and Life-long Learning?

        **Authors**: *Chengxu Zhuang, Ziyu Xiang, Yoon Bai, Xiaoxuan Jia, Nicholas B. Turk-Browne, Kenneth A. Norman, James J. DiCarlo, Dan Yamins*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Humans learn from visual inputs at multiple timescales, both rapidly and flexibly acquiring visual knowledge over short periods, and robustly accumulating online learning progress over longer periods. Modeling these powerful learning capabilities is an important problem for computational visual cognitive science, and models that could replicate them would be of substantial utility in real-world computer vision settings. In this work, we establish benchmarks for both real-time and life-long continual visual learning. Our real-time learning benchmark measures a model's ability to match the rapid visual behavior changes of real humans over the course of minutes and hours, given a stream of visual inputs. Our life-long learning benchmark evaluates the performance of models in a purely online learning curriculum obtained directly from child visual experience over the course of years of development. We evaluate a spectrum of recent deep self-supervised visual learning algorithms on both benchmarks, finding that none of them perfectly match human performance, though some algorithms perform substantially better than others. Interestingly, algorithms embodying recent trends in self-supervised learning -- including BYOL, SwAV and MAE -- are substantially worse on our benchmarks than an earlier generation of self-supervised algorithms such as SimCLR and MoCo-v2. We present analysis indicating that the failure of these newer algorithms is primarily due to their inability to handle the kind of sparse low-diversity datastreams that naturally arise in the real world, and that actively leveraging memory through negative sampling -- a mechanism eschewed by these newer algorithms -- appears useful for facilitating learning in such low-diversity environments. We also illustrate a complementarity between the short and long timescales in the two benchmarks, showing how requiring a single learning algorithm to be locally context-sensitive enough to match real-time learning changes while stable enough to avoid catastrophic forgetting over the long term induces a trade-off that human-like algorithms may have to straddle. Taken together, our benchmarks establish a quantitative way to directly compare learning between neural networks models and human learners, show how choices in the mechanism by which such algorithms handle sample comparison and memory strongly impact their ability to match human learning abilities, and expose an open problem space for identifying more flexible and robust visual self-supervision algorithms.

        ----

        ## [1644] Stochastic Multiple Target Sampling Gradient Descent

        **Authors**: *Hoang Phan, Ngoc Tran, Trung Le, Toan Tran, Nhat Ho, Dinh Phung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html)

        **Abstract**:

        Sampling from an unnormalized target distribution is an essential problem with many applications in probabilistic inference. Stein Variational Gradient Descent (SVGD) has been shown to be a powerful method that iteratively updates a set of particles to approximate the distribution of interest. Furthermore, when analysing its asymptotic properties, SVGD reduces exactly to a single-objective optimization problem and can be viewed as a probabilistic version of this single-objective optimization problem. A natural question then arises: ``Can we derive a probabilistic version of the multi-objective optimization?''. To answer this question, we propose Stochastic Multiple Target Sampling Gradient Descent (MT-SGD), enabling us to sample from multiple unnormalized target distributions. Specifically, our MT-SGD conducts a flow of intermediate distributions gradually orienting to multiple target distributions, which allows the sampled particles to move to the joint high-likelihood region of the target distributions. Interestingly, the asymptotic analysis shows that our approach reduces exactly to the multiple-gradient descent algorithm for multi-objective optimization, as expected. Finally, we conduct comprehensive experiments to demonstrate the merit of our approach to multi-task learning.

        ----

        ## [1645] Towards Out-of-Distribution Sequential Event Prediction: A Causal Treatment

        **Authors**: *Chenxiao Yang, Qitian Wu, Qingsong Wen, Zhiqiang Zhou, Liang Sun, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html)

        **Abstract**:

        The goal of sequential event prediction is to estimate the next event based on a sequence of historical events, with applications to sequential recommendation, user behavior analysis and clinical treatment. In practice, the next-event prediction models are trained with sequential data collected at one time and need to generalize to newly arrived sequences in remote future, which requires models to handle temporal distribution shift from training to testing. In this paper, we first take a data-generating perspective to reveal a negative result that existing approaches with maximum likelihood estimation would fail for distribution shift due to the latent context confounder, i.e., the common cause for the historical events and the next event. Then we devise a new learning objective based on backdoor adjustment and further harness variational inference to make it tractable for sequence learning problems. On top of that, we propose a framework with hierarchical branching structures for learning context-specific representations. Comprehensive experiments on diverse tasks (e.g., sequential recommendation) demonstrate the effectiveness, applicability and scalability of our method with various off-the-shelf models as backbones.

        ----

        ## [1646] Hybrid Neural Autoencoders for Stimulus Encoding in Visual and Other Sensory Neuroprostheses

        **Authors**: *Jacob Granley, Lucas Relic, Michael Beyeler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html)

        **Abstract**:

        Sensory neuroprostheses are emerging as a promising technology to restore lost sensory function or augment human capabilities. However, sensations elicited by current devices often appear artificial and distorted. Although current models can predict the neural or perceptual response to an electrical stimulus, an optimal stimulation strategy solves the inverse problem: what is the required stimulus to produce a desired response? Here, we frame this as an end-to-end optimization problem, where a deep neural network stimulus encoder is trained to invert a known and fixed forward model that approximates the underlying biological system. As a proof of concept, we demonstrate the effectiveness of this Hybrid Neural Autoencoder (HNA) in visual neuroprostheses. We find that HNA produces high-fidelity patient-specific stimuli representing handwritten digits and segmented images of everyday objects, and significantly outperforms conventional encoding strategies across all simulated patients. Overall this is an important step towards the long-standing challenge of restoring high-quality vision to people living with incurable blindness and may prove a promising solution for a variety of neuroprosthetic technologies.

        ----

        ## [1647] Tractable Function-Space Variational Inference in Bayesian Neural Networks

        **Authors**: *Tim G. J. Rudner, Zonghao Chen, Yee Whye Teh, Yarin Gal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ea50bf458f6070548b11babbe0bf89b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ea50bf458f6070548b11babbe0bf89b-Abstract-Conference.html)

        **Abstract**:

        Reliable predictive uncertainty estimation plays an important role in enabling the deployment of neural networks to safety-critical settings. A popular approach for estimating the predictive uncertainty of neural networks is to define a prior distribution over the network parameters, infer an approximate posterior distribution, and use it to make stochastic predictions. However, explicit inference over neural network parameters makes it difficult to incorporate meaningful prior information about the data-generating process into the model. In this paper, we pursue an alternative approach. Recognizing that the primary object of interest in most settings is the distribution over functions induced by the posterior distribution over neural network parameters, we frame Bayesian inference in neural networks explicitly as inferring a posterior distribution over functions and propose a scalable function-space variational inference method that allows incorporating prior information and results in reliable predictive uncertainty estimates. We show that the proposed method leads to state-of-the-art uncertainty estimation and predictive performance on a range of prediction tasks and demonstrate that it performs well on a challenging safety-critical medical diagnosis task in which reliable uncertainty estimation is essential.

        ----

        ## [1648] BMU-MoCo: Bidirectional Momentum Update for Continual Video-Language Modeling

        **Authors**: *Yizhao Gao, Nanyi Fei, Haoyu Lu, Zhiwu Lu, Hao Jiang, Yijie Li, Zhao Cao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html)

        **Abstract**:

        Video-language models suffer from forgetting old/learned knowledge when trained with streaming data. In this work, we thus propose a continual video-language modeling (CVLM) setting, where models are supposed to be sequentially trained on five widely-used video-text datasets with different data distributions. Although most of existing continual learning methods have achieved great success by exploiting extra information (e.g., memory data of past tasks) or dynamically extended networks, they cause enormous resource consumption when transferred to our CVLM setting. To overcome the challenges (i.e., catastrophic forgetting and heavy resource consumption) in CVLM, we propose a novel cross-modal MoCo-based model with bidirectional momentum update (BMU), termed BMU-MoCo. Concretely, our BMU-MoCo has two core designs: (1) Different from the conventional MoCo, we apply the momentum update to not only momentum encoders but also encoders (i.e., bidirectional) at each training step, which enables the model to review the learned knowledge retained in the momentum encoders. (2) To further enhance our BMU-MoCo by utilizing earlier knowledge, we additionally maintain a pair of global momentum encoders (only initialized at the very beginning) with the same BMU strategy. Extensive results show that our BMU-MoCo remarkably outperforms recent competitors w.r.t. video-text retrieval performance and forgetting rate, even without using any extra data or dynamic networks.

        ----

        ## [1649] Can Hybrid Geometric Scattering Networks Help Solve the Maximum Clique Problem?

        **Authors**: *Yimeng Min, Frederik Wenkel, Michael Perlmutter, Guy Wolf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ec88961d36d9a87ac24baf45402744f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ec88961d36d9a87ac24baf45402744f-Abstract-Conference.html)

        **Abstract**:

        We propose a geometric scattering-based graph neural network (GNN) for approximating solutions of the NP-hard maximum clique (MC) problem. We construct a loss function with two terms, one which encourages the network to find highly connected nodes and the other which acts as a surrogate for the constraint that the nodes form a clique. We then use this loss to train an efficient GNN architecture that outputs a vector representing the probability for each node to be part of the MC and apply a rule-based decoder to make our final prediction. The incorporation of the scattering transform alleviates the so-called oversmoothing problem that is often encountered in GNNs and would degrade the performance of our proposed setup. Our empirical results demonstrate that our method outperforms representative GNN baselines in terms of solution accuracy and inference speed as well as conventional solvers like Gurobi with limited time budgets. Furthermore, our scattering model is very parameter efficient with only $\sim$ 0.1\% of the number of parameters compared to previous GNN baseline models.

        ----

        ## [1650] Expected Improvement for Contextual Bandits

        **Authors**: *Hung Tran-The, Sunil Gupta, Santu Rana, Tuan Truong, Long Tran-Thanh, Svetha Venkatesh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f0942c43fcfba4cc66a859b9fcb1bba-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f0942c43fcfba4cc66a859b9fcb1bba-Abstract-Conference.html)

        **Abstract**:

        The expected improvement (EI) is a popular technique to handle the tradeoff between exploration and exploitation under uncertainty. This technique has been widely used in Bayesian optimization but it is not applicable for the contextual bandit problem which is a generalization of the standard bandit and Bayesian optimization. In this paper, we initiate and study the EI technique for contextual bandits from both theoretical and practical perspectives. We propose two novel EI-based algorithms, one when the reward function is assumed to be linear and the other for more general reward functions. With linear reward functions, we demonstrate that our algorithm achieves a near-optimal regret. Notably, our regret improves that of LinTS \cite{agrawal13} by a factor $\sqrt{d}$ while avoiding to solve a NP-hard problem at each iteration as in LinUCB \cite{Abbasi11}. For more general reward functions which are modeled by deep neural networks, we prove that our algorithm achieves a $\tilde{\mathcal O} (\tilde{d}\sqrt{T})$ regret, where $\tilde{d}$ is the effective dimension of a neural tangent kernel (NTK) matrix, and $T$ is the number of iterations. Our experiments on various benchmark datasets show that both proposed algorithms work well and consistently outperform existing approaches, especially in high dimensions.

        ----

        ## [1651] Redistribution of Weights and Activations for AdderNet Quantization

        **Authors**: *Ying Nie, Kai Han, Haikang Diao, Chuanjian Liu, Enhua Wu, Yunhe Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f15e0b418ccdefec8313affc897dc8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f15e0b418ccdefec8313affc897dc8c-Abstract-Conference.html)

        **Abstract**:

        Adder Neural Network (AdderNet) provides a new way for developing energy-efficient neural networks by replacing the expensive multiplications in convolution with cheaper additions (i.e., L1-norm). To achieve higher hardware efficiency, it is necessary to further study the low-bit quantization of AdderNet. Due to the limitation that the commutative law in multiplication does not hold in L1-norm, the well-established quantization methods on convolutional networks cannot be applied on AdderNets. Thus, the existing AdderNet quantization techniques propose to use only one shared scale to quantize both the weights and activations simultaneously. Admittedly, such an approach can keep the commutative law in the  L1-norm quantization process, while the accuracy drop after low-bit quantization cannot be ignored. To this end, we first thoroughly analyze the difference on distributions of weights and activations in AdderNet and then propose a new quantization algorithm by redistributing the weights and the activations. Specifically, the pre-trained full-precision weights in different kernels are clustered into different groups, then the intra-group sharing and inter-group independent scales can be adopted. To further compensate the accuracy drop caused by the distribution difference, we then develop a lossless range clamp scheme for weights and a simple yet effective outliers clamp strategy for activations. Thus, the functionality of full-precision weights and the representation ability of full-precision activations can be fully preserved. The effectiveness of the proposed quantization method for AdderNet is well verified on several benchmarks, e.g., our 4-bit post-training quantized adder ResNet-18 achieves an 66.5% top-1 accuracy on the ImageNet with comparable energy efficiency,  which is about 8.5% higher than that of the previous AdderNet quantization methods. Code will be available at https://gitee.com/mindspore/models/tree/master/research/cv/AdderQuant.

        ----

        ## [1652] Physically-Based Face Rendering for NIR-VIS Face Recognition

        **Authors**: *Yunqi Miao, Alexandros Lattas, Jiankang Deng, Jungong Han, Stefanos Zafeiriou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html)

        **Abstract**:

        Near infrared (NIR) to Visible (VIS) face matching is challenging due to the significant domain gaps as well as a lack of sufficient data for cross-modality model training. To overcome this problem, we propose a novel method for paired NIR-VIS facial image generation. Specifically, we reconstruct 3D face shape and reflectance from a large 2D facial dataset and introduce a novel method of transforming the VIS reflectance to NIR reflectance. We then use a physically-based renderer to generate a vast, high-resolution and photorealistic dataset consisting of various poses and identities in the NIR and VIS spectra. Moreover, to facilitate the identity feature learning, we propose an IDentity-based Maximum Mean Discrepancy (ID-MMD) loss, which not only reduces the modality gap between NIR and VIS images at the domain level but encourages the network to focus on the identity features instead of facial details, such as poses and accessories. Extensive experiments conducted on four challenging NIR-VIS face recognition benchmarks demonstrate that the proposed method can achieve comparable performance with the state-of-the-art (SOTA) methods without requiring any existing NIR-VIS face recognition datasets. With slightly fine-tuning on the target NIR-VIS face recognition datasets, our method can significantly surpass the SOTA performance. Code and pretrained models are released under the insightface GitHub.

        ----

        ## [1653] DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection

        **Authors**: *Xuanwen Huang, Yang Yang, Yang Wang, Chunping Wang, Zhisheng Zhang, Jiarong Xu, Lei Chen, Michalis Vazirgiannis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f1918f71972789db39ec0d85bb31110-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Graph Anomaly Detection (GAD) has recently become a hot research spot due to its practicability and theoretical value. Since GAD emphasizes the application and the rarity of anomalous samples, enriching the varieties of its datasets is fundamental. Thus, this paper present DGraph, a real-world dynamic graph in the finance domain. DGraph overcomes many limitations of current GAD datasets. It contains about 3M nodes, 4M dynamic edges, and 1M ground-truth nodes. We provide a comprehensive observation of DGraph, revealing that anomalous nodes and normal nodes generally have different structures, neighbor distribution, and temporal dynamics. Moreover, it suggests that 2M background nodes are also essential for detecting fraudsters. Furthermore, we conduct extensive experiments on DGraph. Observation and experiments demonstrate that DGraph is propulsive to advance GAD research and enable in-depth exploration of anomalous nodes.

        ----

        ## [1654] Globally Convergent Policy Search for Output Estimation

        **Authors**: *Jack Umenberger, Max Simchowitz, Juan C. Perdomo, Kaiqing Zhang, Russ Tedrake*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f41d5802bea87ab45425fbcf78349c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f41d5802bea87ab45425fbcf78349c0-Abstract-Conference.html)

        **Abstract**:

        We introduce the first direct policy search algorithm which provably converges to the globally optimal dynamic filter for the classical problem of predicting the outputs of a linear dynamical system, given noisy, partial observations. Despite the ubiquity of partial observability in practice, theoretical guarantees for direct policy search algorithms, one of the backbones of modern reinforcement learning, have proven difficult to achieve. This is primarily due to the degeneracies which arise when optimizing over filters that maintain an internal state. In this paper, we provide a new perspective on this challenging problem based on the notion of informativity, which intuitively requires that all components of a filter’s internal state are representative of the true state of the underlying dynamical system. We show that informativity overcomes the aforementioned degeneracy. Specifically, we propose a regularizer which explicitly enforces informativity, and establish that gradient descent on this regularized objective - combined with a “reconditioning step” – converges to the globally optimal cost at a $O(1/T)$ rate.

        ----

        ## [1655] Escaping Saddle Points for Effective Generalization on Class-Imbalanced Data

        **Authors**: *Harsh Rangwani, Sumukh K. Aithal, Mayank Mishra, Venkatesh Babu R.*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f4d70db9ecec97b6723a86f1cd9cb4b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f4d70db9ecec97b6723a86f1cd9cb4b-Abstract-Conference.html)

        **Abstract**:

        Real-world datasets exhibit imbalances of varying types and degrees. Several techniques based on re-weighting and margin adjustment of loss are often used to enhance the performance of neural networks, particularly on minority classes. In this work, we analyze the class-imbalanced learning problem by examining the loss landscape of neural networks trained with re-weighting and margin based techniques. Specifically, we examine the spectral density of Hessian of class-wise loss, through which we observe that the network weights converges to a saddle point in the loss landscapes of minority classes. Following this observation, we also find that optimization methods designed to escape from saddle points can be effectively used to improve generalization on minority classes. We further theoretically and empirically demonstrate that Sharpness-Aware Minimization (SAM), a recent technique that encourages convergence to a flat minima, can be effectively used to escape saddle points for minority classes. Using SAM results in a 6.2\% increase in accuracy on the minority classes over the state-of-the-art Vector Scaling Loss, leading to an overall average increase of 4\% across imbalanced datasets. The code is available at https://github.com/val-iisc/Saddle-LongTail.

        ----

        ## [1656] Continuous Deep Q-Learning in Optimal Control Problems: Normalized Advantage Functions Analysis

        **Authors**: *Anton Plaksin, Stepan Martyanov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f688ba732c27f76542cad77f0fa2e27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f688ba732c27f76542cad77f0fa2e27-Abstract-Conference.html)

        **Abstract**:

        One of the most effective continuous deep reinforcement learning algorithms is normalized advantage functions (NAF). The main idea of NAF consists in the approximation of the Q-function by functions quadratic with respect to the action variable. This idea allows to apply the algorithm to continuous reinforcement learning problems, but on the other hand, it brings up the question of classes of problems in which this approximation is acceptable. The presented paper describes one such class. We consider reinforcement learning problems obtained by the discretization of certain optimal control problems. Based on the idea of NAF, we present a new family of quadratic functions and prove its suitable approximation properties. Taking these properties into account, we provide several ways to improve NAF. The experimental results confirm the efficiency of our improvements.

        ----

        ## [1657] Partial Identification of Treatment Effects with Implicit Generative Models

        **Authors**: *Vahid Balazadeh Meresht, Vasilis Syrgkanis, Rahul G. Krishnan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f6b3692297e49e5d5c91ba00281379c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f6b3692297e49e5d5c91ba00281379c-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of partial identification, the estimation of bounds on the treatment effects from observational data. Although studied using discrete treatment variables or in specific causal graphs (e.g., instrumental variables), partial identification has been recently explored using tools from deep generative modeling. We propose a new method for partial identification of average treatment effects (ATEs) in general causal graphs using implicit generative models comprising continuous and discrete random variables. Since ATE with continuous treatment is generally non-regular, we leverage the partial derivatives of response functions to define a regular approximation of ATE, a quantity we call uniform average treatment derivative (UATD). We prove that our algorithm converges to tight bounds on ATE in linear structural causal models (SCMs). For nonlinear SCMs, we empirically show that using UATD leads to tighter and more stable bounds than methods that directly optimize the ATE.

        ----

        ## [1658] Robust Model Selection and Nearly-Proper Learning for GMMs

        **Authors**: *Allen Liu, Jerry Li, Ankur Moitra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8f75af4704feac629a560f4ad6b67cef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8f75af4704feac629a560f4ad6b67cef-Abstract-Conference.html)

        **Abstract**:

        In learning theory, a standard assumption is that the data is generated from a finite mixture model. But what happens when the number of components is not known in advance? The problem of estimating the number of components, also called model selection, is important in its own right but there are essentially no known efficient algorithms with provable guarantees.  In this work, we study the problem of model selection for univariate Gaussian mixture models (GMMs). Given $\textsf{poly}(k/\epsilon)$ samples from a distribution that is $\epsilon$-close in TV distance to a GMM with $k$ components, we can construct a GMM with $\widetilde{O}(k)$ components that approximates the distribution to within $\widetilde{O}(\epsilon)$ in $\textsf{poly}(k/\epsilon)$ time.  Thus we are able to approximately determine the minimum number of components needed to fit the distribution within a logarithmic factor.  Moreover, by adapting the techniques we obtain similar results for reconstructing Fourier-sparse signals.  Prior to our work, the only known algorithms for learning arbitrary univariate GMMs either output significantly more than $k$ components (e.g. $k/\epsilon^2$ components for kernel density estimates) or run in time exponential in $k$.

        ----

        ## [1659] A2: Efficient Automated Attacker for Boosting Adversarial Training

        **Authors**: *Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming Gu, Yihua Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8fc54b95eb361d109f3a564f2a0cb516-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8fc54b95eb361d109f3a564f2a0cb516-Abstract-Conference.html)

        **Abstract**:

        Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models.However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.

        ----

        ## [1660] Shape, Light, and Material Decomposition from Images using Monte Carlo Rendering and Denoising

        **Authors**: *Jon Hasselgren, Nikolai Hofmann, Jacob Munkberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8fcb27984bf16ca03cad643244ec470d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8fcb27984bf16ca03cad643244ec470d-Abstract-Conference.html)

        **Abstract**:

        Recent advances in differentiable rendering have enabled high-quality reconstruction of 3D scenes from multi-view images. Most methods rely on simple rendering algorithms: pre-filtered direct lighting or learned representations of irradiance. We show that a more realistic shading model, incorporating ray tracing and Monte Carlo integration, substantially improves decomposition into shape, materials & lighting. Unfortunately, Monte Carlo integration provides estimates with significant noise, even at large sample counts, which makes gradient-based inverse rendering very challenging. To address this, we incorporate multiple importance sampling and denoising in a novel inverse rendering pipeline. This improves convergence and enables gradient-based optimization at low sample counts. We present an efficient method to jointly reconstruct geometry (explicit triangle meshes), materials, and lighting, which substantially improves material and light separation compared to previous work. We argue that denoising can become an integral part of high quality inverse rendering pipelines.

        ----

        ## [1661] Convergence for score-based generative modeling with polynomial complexity

        **Authors**: *Holden Lee, Jianfeng Lu, Yixin Tan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/8ff87c96935244b63503f542472462b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/8ff87c96935244b63503f542472462b3-Abstract-Conference.html)

        **Abstract**:

        Score-based generative modeling (SGM) is a highly successful approach for learning a probability distribution from data and generating further samples. We prove the first polynomial convergence guarantees for the core mechanic behind SGM: drawing samples from a probability density $p$ given a score estimate (an estimate of $\nabla \ln p$) that is accurate in $L^2(p)$. Compared to previous works, we do not incur error that grows exponentially in time or that suffers from a curse of dimensionality. Our guarantee works for any smooth distribution and depends polynomially on its log-Sobolev constant. Using our guarantee, we give a theoretical analysis of score-based generative modeling, which transforms white-noise input into samples from a learned data distribution given score estimates at different noise scales. Our analysis gives theoretical grounding to the observation that an annealed procedure is required in practice to generate good samples, as our proof depends essentially on using annealing to obtain a warm start at each step. Moreover, we show that a predictor-corrector algorithm gives better convergence than using either portion alone.

        ----

        ## [1662] Kernel Interpolation with Sparse Grids

        **Authors**: *Mohit Yadav, Daniel R. Sheldon, Cameron Musco*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html)

        **Abstract**:

        Structured kernel interpolation (SKI) accelerates Gaussian processes (GP) inference by interpolating the kernel covariance function using a dense grid of inducing points, whose corresponding kernel matrix is highly structured and thus amenable to fast linear algebra. Unfortunately, SKI scales poorly in the dimension of the input points, since the dense grid size grows exponentially with the dimension. To mitigate this issue, we propose the use of sparse grids within the SKI framework. These grids enable accurate interpolation, but with a number of points growing more slowly with dimension. We contribute a novel nearly linear time matrix-vector multiplication algorithm for the sparse grid kernel matrix. We also describe how sparse grids can be combined with an efficient interpolation scheme based on simplicial complexes. With these modifications, we demonstrate that SKI can be scaled to higher dimensions while maintaining accuracy, for both synthetic and real datasets.

        ----

        ## [1663] Contrastive Language-Image Pre-Training with Knowledge Graphs

        **Authors**: *Xuran Pan, Tianzhu Ye, Dongchen Han, Shiji Song, Gao Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/904aac1c930c196f1c71533d4d9dc31a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/904aac1c930c196f1c71533d4d9dc31a-Abstract-Conference.html)

        **Abstract**:

        Recent years have witnessed the fast development of large-scale pre-training frameworks that can extract multi-modal representations in a unified form and achieve promising performances when transferred to downstream tasks. Nevertheless, existing approaches mainly focus on pre-training with simple image-text pairs, while neglecting the semantic connections between concepts from different modalities. In this paper, we propose a knowledge-based pre-training framework, dubbed Knowledge-CLIP, which injects semantic information into the widely used CLIP model. Through introducing knowledge-based objectives in the pre-training process and utilizing different types of knowledge graphs as training data, our model can semantically align the representations in vision and language with higher quality, and enhance the reasoning ability across scenarios and modalities. Extensive experiments on various vision-language downstream tasks demonstrate the effectiveness of Knowledge-CLIP compared with the original CLIP and competitive baselines.

        ----

        ## [1664] Reconstructing Training Data From Trained Neural Networks

        **Authors**: *Niv Haim, Gal Vardi, Gilad Yehudai, Ohad Shamir, Michal Irani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/906927370cbeb537781100623cca6fa6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/906927370cbeb537781100623cca6fa6-Abstract-Conference.html)

        **Abstract**:

        Understanding to what extent neural networks memorize training data is an intriguing question with practical and theoretical implications. In this paper we show that in some cases a significant fraction of the training data can in fact be reconstructed from the parameters of a trained neural network classifier.We propose a novel reconstruction scheme that stems from recent theoretical results about the implicit bias in training neural networks with gradient-based methods.To the best of our knowledge, our results are the first to show that reconstructing a large portion of the actual training samples from a trained neural network classifier is generally possible.This has negative implications on privacy, as it can be used as an attack for revealing sensitive training data. We demonstrate our method for binary MLP classifiers on a few standard computer vision datasets.

        ----

        ## [1665] Hierarchical Agglomerative Graph Clustering in Poly-Logarithmic Depth

        **Authors**: *Laxman Dhulipala, David Eisenstat, Jakub Lacki, Vahab Mirrokni, Jessica Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/909de96145d97514b143dfde03e6cd2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/909de96145d97514b143dfde03e6cd2b-Abstract-Conference.html)

        **Abstract**:

        Obtaining scalable algorithms for \emph{hierarchical agglomerative clustering} (HAC) is of significant interest due to the massive size of real-world datasets. At the same time, efficiently parallelizing HAC is difficult due to the seemingly sequential nature of the algorithm. In this paper, we address this issue and present ParHAC, the first efficient parallel HAC algorithm with sublinear depth for the widely-used average-linkage function. In particular, we provide a $(1+\epsilon)$-approximation algorithm for this problem on $m$ edge graphs using $\tilde{O}(m)$ work and poly-logarithmic depth. Moreover, we show that obtaining similar bounds for \emph{exact} average-linkage HAC is not possible under standard complexity-theoretic assumptions.We complement our theoretical results with a comprehensive study of the ParHAC algorithm in terms of its scalability, performance, and quality, and compare with several state-of-the-art sequential and parallel baselines. On a broad set of large publicly-available real-world datasets, we find that ParHAC obtains a 50.1x speedup on average over the best sequential baseline, while achieving quality similar to the exact HAC algorithm. We also show that ParHAC can cluster one of the largest publicly available graph datasets with 124 billion edges in a little over three hours using a commodity multicore machine.

        ----

        ## [1666] On-Device Training Under 256KB Memory

        **Authors**: *Ji Lin, Ligeng Zhu, Wei-Ming Chen, Wei-Chen Wang, Chuang Gan, Song Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/90c56c77c6df45fc8e556a096b7a2b2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/90c56c77c6df45fc8e556a096b7a2b2e-Abstract-Conference.html)

        **Abstract**:

        On-device training enables the model to adapt to new data collected from the sensors by fine-tuning a pre-trained model. Users can benefit from customized AI models without having to transfer the data to the cloud, protecting the privacy. However, the training memory consumption is prohibitive for IoT devices that have tiny memory resources. We propose an algorithm-system co-design framework to make on-device training possible with only 256KB of memory. On-device training faces two unique challenges: (1) the quantized graphs of neural networks are hard to optimize due to low bit-precision and the lack of normalization; (2) the limited hardware resource (memory and computation) does not allow full backpropagation. To cope with the optimization difficulty, we propose Quantization- Aware Scaling to calibrate the gradient scales and stabilize 8-bit quantized training. To reduce the memory footprint, we propose Sparse Update to skip the gradient computation of less important layers and sub-tensors. The algorithm innovation is implemented by a lightweight training system, Tiny Training Engine, which prunes the backward computation graph to support sparse updates and offload the runtime auto-differentiation to compile time. Our framework is the first practical solution for on-device transfer learning of visual recognition on tiny IoT devices (e.g., a microcontroller with only 256KB SRAM), using less than 1/1000 of the memory of PyTorch and TensorFlow while matching the accuracy. Our study enables IoT devices not only to perform inference but also to continuously adapt to new data for on-device lifelong learning. A video demo can be found here: https://youtu.be/XaDCO8YtmBw.

        ----

        ## [1667] Behavior Transformers: Cloning $k$ modes with one stone

        **Authors**: *Nur Muhammad Shafiullah, Zichen Jeff Cui, Ariuntuya Altanzaya, Lerrel Pinto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/90d17e882adbdda42349db6f50123817-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/90d17e882adbdda42349db6f50123817-Abstract-Conference.html)

        **Abstract**:

        While behavior learning has made impressive progress in recent times, it lags behind computer vision and natural language processing due to its inability to leverage large, human-generated datasets. Human behavior has a wide variance, multiple modes, and human demonstrations naturally do not come with reward labels. These properties limit the applicability of current methods in Offline RL and Behavioral Cloning to learn from large, pre-collected datasets. In this work, we present Behavior Transformer (BeT), a new technique to model unlabeled demonstration data with multiple modes. BeT retrofits standard transformer architectures with action discretization coupled with a multi-task action correction inspired by offset prediction in object detection. This allows us to leverage the multi-modal modeling ability of modern transformers to predict multi-modal continuous actions. We experimentally evaluate BeT on a variety of robotic manipulation and self-driving behavior datasets. We show that BeT significantly improves over prior state-of-the-art work on solving demonstrated tasks while capturing the major modes present in the pre-collected datasets. Finally, through an extensive ablation study, we further analyze the importance of every crucial component in BeT. Videos of behavior generated by BeT are available here: https://mahis.life/bet

        ----

        ## [1668] Performative Power

        **Authors**: *Moritz Hardt, Meena Jagadeesan, Celestine Mendler-Dünner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Abstract-Conference.html)

        **Abstract**:

        We introduce the notion of performative power, which measures the ability of a firm operating an algorithmic system, such as a digital content recommendation platform, to cause change in a population of participants. We relate performative power to the economic study of competition in digital economies. Traditional economic concepts struggle with identifying anti-competitive patterns in digital platforms not least due to the complexity of market definition. In contrast, performative power is a causal notion that is identifiable with minimal knowledge of the market, its internals, participants, products, or prices.Low performative power implies that a firm can do no better than to optimize their objective on current data. In contrast, firms of high performative power stand to benefit from steering the population towards more profitable behavior. We confirm in a simple theoretical model that monopolies maximize performative power. A firm's ability to personalize increases performative power, while competition and outside options decrease performative power. On the empirical side, we propose an observational causal design to identify performative power from discontinuities in how digital platforms display content. This allows to repurpose causal effects from various studies about digital platforms as lower bounds on performative power. Finally, we speculate about the role that performative power might play in competition policy and antitrust enforcement in digital marketplaces.

        ----

        ## [1669] Diagonal State Spaces are as Effective as Structured State Spaces

        **Authors**: *Ankit Gupta, Albert Gu, Jonathan Berant*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9156b0f6dfa9bbd18c79cc459ef5d61c-Abstract-Conference.html)

        **Abstract**:

        Modeling long range dependencies in sequential data is a fundamental step towards attaining human-level performance in many modalities such as text, vision, audio and video. While attention-based models are a popular and effective choice in modeling short-range interactions, their performance on tasks requiring long range reasoning has been largely inadequate. In an exciting result, Gu et al. (ICLR 2022) proposed the $\textit{Structured State Space}$ (S4) architecture delivering large gains over state-of-the-art models on several long-range tasks across various modalities. The core proposition of S4 is the parameterization of state matrices via a diagonal plus low rank structure, allowing efficient computation. In this work, we show that one can match the performance of S4 even without the low rank correction and thus assuming the state matrices to be diagonal. Our $\textit{Diagonal State Space}$ (DSS) model matches the performance of S4 on Long Range Arena tasks, speech classification on Speech Commands dataset, while being conceptually simpler and straightforward to implement.

        ----

        ## [1670] Deep Fourier Up-Sampling

        **Authors**: *Man Zhou, Hu Yu, Jie Huang, Feng Zhao, Jinwei Gu, Chen Change Loy, Deyu Meng, Chongyi Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91a23b3e6a2ebaad62e17d0269f88c6b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91a23b3e6a2ebaad62e17d0269f88c6b-Abstract-Conference.html)

        **Abstract**:

        Existing convolutional neural networks widely adopt spatial down-/up-sampling for multi-scale modeling. However, spatial up-sampling operators (e.g., interpolation, transposed convolution, and un-pooling) heavily depend on local pixel attention, incapably exploring the global dependency. In contrast,  the Fourier domain is in accordance with the nature of global modeling according to the spectral convolution theorem. Unlike the spatial domain that easily performs  up-sampling with the property of local similarity, up-sampling in the Fourier domain is more challenging as it does not follow such a local property. In this study, we propose a theoretically feasible Deep Fourier Up-Sampling (FourierUp) to solve these issues. We revisit the relationships between spatial and Fourier domains and reveal the transform rules on the features of different resolutions in the Fourier domain, which provide key insights for FourierUp's designs. FourierUp as a generic operator consists of three key components: 2D discrete Fourier transform,  Fourier dimension increase rules, and 2D inverse Fourier transform, which can be directly integrated with existing networks. Extensive experiments across multiple computer vision tasks, including object detection, image segmentation, image de-raining, image dehazing, and guided image super-resolution, demonstrate the consistent performance gains obtained by introducing our FourierUp. Code will be publicly available.

        ----

        ## [1671] Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement

        **Authors**: *Yan Li, Xinjiang Lu, Yaqing Wang, Dejing Dou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91a85f3fb8f570e6be52b333b5ab017a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91a85f3fb8f570e6be52b333b5ab017a-Abstract-Conference.html)

        **Abstract**:

        Time series forecasting has been a widely explored task of great importance in many applications. However, it is common that real-world time series data are recorded in a short time period, which results in a big gap between the deep model and the limited and noisy time series. In this work, we propose to address the time series forecasting problem with generative modeling and propose a bidirectional variational auto-encoder (BVAE) equipped with diffusion, denoise, and disentanglement, namely D3VAE. Specifically, a coupled diffusion probabilistic model is proposed to augment the time series data without increasing the aleatoric uncertainty and implement a more tractable inference process with BVAE. To ensure the generated series move toward the true target, we further propose to adapt and integrate the multiscale denoising score matching into the diffusion process for time series forecasting. In addition, to enhance the interpretability and stability of the prediction, we treat the latent variable in a multivariate manner and disentangle them on top of minimizing total correlation. Extensive experiments on synthetic and real-world data show that D3VAE outperforms competitive algorithms with remarkable margins. Our implementation is available at https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/D3VAE.

        ----

        ## [1672] The Missing Invariance Principle found - the Reciprocal Twin of Invariant Risk Minimization

        **Authors**: *Dongsung Huh, Avinash Baidya*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91b482312a0845ed86e244adbd9935e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91b482312a0845ed86e244adbd9935e4-Abstract-Conference.html)

        **Abstract**:

        Machine learning models often generalize poorly to out-of-distribution (OOD) data as a result of relying on features that are spuriously correlated with the label during training. Recently, the technique of Invariant Risk Minimization (IRM) was proposed to learn predictors that only use invariant features by conserving the feature-conditioned label expectation $\mathbb{E}_e[y|f(x)]$ across environments. However, more recent studies have demonstrated that IRM-v1, a practical version of IRM, can fail in various settings. Here, we identify a fundamental flaw of IRM formulation that causes the failure. We then introduce a complementary notion of invariance, MRI, based on conserving the label-conditioned feature expectation $\mathbb{E}_e[f(x)|y]$, which is free of this flaw. Further, we introduce a simplified, practical version of the MRI formulation called MRI-v1. We prove that for general linear problems, MRI-v1 guarantees invariant predictors given sufficient number of environments. We also empirically demonstrate that MRI-v1 strongly out-performs IRM-v1 and consistently achieves near-optimal OOD generalization in  image-based nonlinear problems.

        ----

        ## [1673] Signal Recovery with Non-Expansive Generative Network Priors

        **Authors**: *Jorio Cocola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91d193b65d0b120d29503590827de1ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91d193b65d0b120d29503590827de1ea-Abstract-Conference.html)

        **Abstract**:

        We study compressive sensing with a deep generative network prior. Initial theoretical guarantees for efficient recovery from compressed linear measurements have been developed for signals in the range of a ReLU network with Gaussian weights and logarithmic expansivity: that is when each layer is larger than the previous one by a logarithmic factor. It was later shown that constant expansivity is sufficient for recovery. It has remained open whether the expansivity can be relaxed, allowing for networks with contractive layers (as often the case of real generators). In this work we answer this question, proving that a signal in the range of a Gaussian generative network can be recovered from few linear measurements provided that the width of the layers is proportional to the input layer size (up to log factors). This condition allows the generative network to have contractive layers. Our result is based on showing that Gaussian matrices satisfy a matrix concentration inequality which we term Range Restricted Weight Distribution Condition (R2WDC) and which weakens the Weight Distribution Condition (WDC) upon which previous theoretical guarantees were based. The WDC has also been used to analyze other signal recovery problems with generative network priors. By replacing the WDC with the R2WDC, we are able to extend previous results for signal recovery with expansive generative network priors to non-expansive ones. We discuss these extensions for phase retrieval, denoising, and spiked matrix recovery.

        ----

        ## [1674] Towards Understanding the Mixture-of-Experts Layer in Deep Learning

        **Authors**: *Zixiang Chen, Yihe Deng, Yue Wu, Quanquan Gu, Yuanzhi Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html)

        **Abstract**:

        The Mixture-of-Experts (MoE) layer, a sparsely-activated model controlled by a router, has achieved great success in deep learning. However, the understanding of such architecture remains elusive. In this paper, we formally study how the MoE layer improves the performance of neural network learning and why the mixture model will not collapse into a single model. Our empirical results suggest that the cluster structure of the underlying problem and the non-linearity of the expert are pivotal to the success of MoE. This motivates us to consider a challenging classification problem with intrinsic cluster structures. Theoretically, we proved that this problem is hard to solve by a single expert such as a two-layer convolutional neural network (CNN).  Yet with the MoE layer with each expert being a two-layer CNN, the problem can be solved successfully. In particular, our theory shows that the router can learn the cluster-center features, which helps divide the input complex problem into simpler classification sub-problems that individual experts can conquer. To our knowledge, this is the first theoretical result toward formally understanding the mechanism of the MoE layer for deep learning.

        ----

        ## [1675] Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples

        **Authors**: *Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/91ffdc5e2f12436d99914418e38d0a09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/91ffdc5e2f12436d99914418e38d0a09-Abstract-Conference.html)

        **Abstract**:

        Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations.Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner.In this work, we overcome these limitations by: (i) categorizing   attack failures based on how they affect the optimization of gradient-based attacks, while also  unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel \emph{indicators of failure}, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.

        ----

        ## [1676] Beyond accuracy: generalization properties of bio-plausible temporal credit assignment rules

        **Authors**: *Yuhan Helena Liu, Arna Ghosh, Blake A. Richards, Eric Shea-Brown, Guillaume Lajoie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9226f8122feb9c229c1efd9270ce7021-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9226f8122feb9c229c1efd9270ce7021-Abstract-Conference.html)

        **Abstract**:

        To unveil how the brain learns, ongoing work seeks  biologically-plausible approximations of gradient descent algorithms for training recurrent neural networks (RNNs). Yet, beyond task accuracy, it is unclear if such learning rules converge to solutions that exhibit different levels of generalization than their non-biologically-plausible counterparts. Leveraging results from deep learning theory based on loss landscape curvature, we ask: how do biologically-plausible gradient approximations affect generalization? We first demonstrate that state-of-the-art biologically-plausible learning rules for training RNNs exhibit worse and more variable generalization performance compared to their machine learning counterparts that follow the true gradient more closely. Next, we verify that such generalization performance is correlated significantly with loss landscape curvature, and we show that biologically-plausible learning rules tend to approach high-curvature regions in synaptic weight space. Using tools from dynamical systems, we derive theoretical arguments and present a theorem explaining this phenomenon. This predicts our numerical results, and explains why biologically-plausible rules lead to worse and more variable generalization properties. Finally, we suggest potential remedies that could be used by the brain to mitigate this effect. To our knowledge, our analysis is the first to identify the reason for this generalization gap between artificial and biologically-plausible learning rules, which can help guide future investigations into how the brain learns solutions that generalize.

        ----

        ## [1677] Parameter tuning and model selection in Optimal Transport with semi-dual Brenier formulation

        **Authors**: *Adrien Vacher, François-Xavier Vialard*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9230b34134929c69b14dc37990634122-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9230b34134929c69b14dc37990634122-Abstract-Conference.html)

        **Abstract**:

        Over the past few years, numerous computational models have been developed to solve Optimal Transport (OT) in a stochastic setting, where distributions are represented by samples and where the goal is to find the closest map to the ground truth OT map, unknown in practical settings. So far, no quantitative criterion has yet been put forward to tune the parameter of these models and select maps that best approximate the ground truth. To perform this task, we propose to leverage the Brenier formulation of OT. Theoretically, we show that this formulation guarantees that, up to sharp a distortion parameter depending on the smoothness/strong convexity and a statistical deviation term, the selected map achieves the lowest quadratic error to the ground truth. This criterion, estimated via convex optimization, enables parameter tuning and model selection among entropic regularization of OT, input convex neural networks and smooth and strongly convex nearest-Brenier (SSNB) models.We also use this criterion to question the use of OT in Domain-Adaptation (DA). In a standard DA experiment, it enables us to identify the potential that is closest to the true OT map between the source and the target. Yet, we observe that this selected potential is far from being the one that performs best for the downstream transfer classification task.

        ----

        ## [1678] VITA: Video Instance Segmentation via Object Token Association

        **Authors**: *Miran Heo, Sukjun Hwang, Seoung Wug Oh, Joon-Young Lee, Seon Joo Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9232d474be0a4e5f1e1bcb0765f17f9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9232d474be0a4e5f1e1bcb0765f17f9a-Abstract-Conference.html)

        **Abstract**:

        We introduce a novel paradigm for offline Video Instance Segmentation (VIS), based on the hypothesis that explicit object-oriented information can be a strong clue for understanding the context of the entire sequence. To this end, we propose VITA, a simple structure built on top of an off-the-shelf Transformer-based image instance segmentation model. Specifically, we use an image object detector as a means of distilling object-specific contexts into object tokens. VITA accomplishes video-level understanding by associating frame-level object tokens without using spatio-temporal backbone features. By effectively building relationships between objects using the condensed information, VITA achieves the state-of-the-art on VIS benchmarks with a ResNet-50 backbone: 49.8 AP, 45.7 AP on YouTube-VIS 2019 & 2021, and 19.6 AP on OVIS. Moreover, thanks to its object token-based structure that is disjoint from the backbone features, VITA shows several practical advantages that previous offline VIS methods have not explored - handling long and high-resolution videos with a common GPU, and freezing a frame-level detector trained on image domain. Code is available at the link.

        ----

        ## [1679] Bridging the Gap: Unifying the Training and Evaluation of Neural Network Binary Classifiers

        **Authors**: *Nathan Tsoi, Kate Candon, Deyuan Li, Yofti Milkessa, Marynel Vázquez*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/92440ec643f4e9f17409557b6516566e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/92440ec643f4e9f17409557b6516566e-Abstract-Conference.html)

        **Abstract**:

        While neural network binary classifiers are often evaluated on metrics such as Accuracy and $F_1$-Score, they are commonly trained with a cross-entropy objective. How can this training-evaluation gap be addressed? While specific techniques have been adopted to optimize certain confusion matrix based metrics, it is challenging or impossible in some cases to generalize the techniques to other metrics. Adversarial learning approaches have also been proposed to optimize networks via confusion matrix based metrics, but they tend to be much slower than common training methods. In this work, we propose a unifying approach to training neural network binary classifiers that combines a differentiable approximation of the Heaviside function with a probabilistic view of the typical confusion matrix values using soft sets. Our theoretical analysis shows the benefit of using our method to optimize for a given evaluation metric, such as $F_1$-Score, with soft sets, and our extensive experiments show the effectiveness of our approach in several domains.

        ----

        ## [1680] Truncated proposals for scalable and hassle-free simulation-based inference

        **Authors**: *Michael Deistler, Pedro J. Gonçalves, Jakob H. Macke*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html)

        **Abstract**:

        Simulation-based inference (SBI) solves statistical inverse problems by repeatedly running a stochastic simulator and inferring posterior distributions from model-simulations. To improve simulation efficiency, several inference methods take a sequential approach and iteratively adapt the proposal distributions from which model simulations are generated. However, many of these sequential methods are difficult to use in practice, both because the resulting optimisation problems can be challenging and efficient diagnostic tools are lacking. To overcome these issues, we present Truncated Sequential Neural Posterior Estimation (TSNPE). TSNPE performs sequential inference with truncated proposals, sidestepping the optimisation issues of alternative approaches. In addition, TSNPE allows to efficiently perform coverage tests that can scale to complex models with many parameters. We demonstrate that TSNPE performs on par with previous methods on established benchmark tasks. We then apply TSNPE to two challenging problems from neuroscience and show that TSNPE can successfully obtain the posterior distributions, whereas previous methods fail. Overall, our results demonstrate that TSNPE is an efficient, accurate, and robust inference method that can scale to challenging scientific models.

        ----

        ## [1681] When to Update Your Model: Constrained Model-based Reinforcement Learning

        **Authors**: *Tianying Ji, Yu Luo, Fuchun Sun, Mingxuan Jing, Fengxiang He, Wenbing Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/927eae0f3d1c89cc39398022f436c472-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/927eae0f3d1c89cc39398022f436c472-Abstract-Conference.html)

        **Abstract**:

        Designing and analyzing model-based RL (MBRL) algorithms with guaranteed monotonic improvement has been challenging, mainly due to the interdependence between policy optimization and model learning. Existing discrepancy bounds generally ignore the impacts of model shifts, and their corresponding algorithms are prone to degrade performance by drastic model updating. In this work, we first propose a novel and general theoretical scheme for a non-decreasing performance guarantee of MBRL. Our follow-up derived bounds reveal the relationship between model shifts and performance improvement. These discoveries encourage us to formulate a constrained lower-bound optimization problem to permit the monotonicity of MBRL. A further example demonstrates that learning models from a dynamically-varying number of explorations benefit the eventual returns. Motivated by these analyses, we design a simple but effective algorithm CMLO (Constrained Model-shift Lower-bound Optimization), by introducing an event-triggered mechanism that flexibly determines when to update the model.  Experiments show that CMLO surpasses other state-of-the-art methods and produces a boost when various policy optimization methods are employed.

        ----

        ## [1682] Noise Attention Learning: Enhancing Noise Robustness by Gradient Scaling

        **Authors**: *Yangdi Lu, Yang Bo, Wenbo He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/92864e1191ed272deb0914b3bb50f97c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/92864e1191ed272deb0914b3bb50f97c-Abstract-Conference.html)

        **Abstract**:

        Machine learning has been highly successful in data-driven applications but is often hampered when the data contains noise, especially label noise. When trained on noisy labels, deep neural networks tend to fit all noisy labels, resulting in poor generalization. To handle this problem, a common idea is to force the model to fit only clean samples rather than mislabeled ones. In this paper, we propose a simple yet effective method that automatically distinguishes the mislabeled samples and prevents the model from memorizing them, named Noise Attention Learning. In our method, we introduce an attention branch to produce attention weights based on representations of samples. This attention branch is learned to divide the samples according to the predictive power in their representations. We design the corresponding loss function that incorporates the attention weights for training the model without affecting the original learning direction. Empirical results show that most of the mislabeled samples yield significantly lower weights than the clean ones. Furthermore, our theoretical analysis shows that the gradients of training samples are dynamically scaled by the attention weights, implicitly preventing memorization of the mislabeled samples. Experimental results on two benchmarks (CIFAR-10 and CIFAR-100) with simulated label noise and three real-world noisy datasets (ANIMAL-10N, Clothing1M and Webvision) demonstrate that our approach outperforms state-of-the-art methods.

        ----

        ## [1683] Iso-Dream: Isolating and Leveraging Noncontrollable Visual Dynamics in World Models

        **Authors**: *Minting Pan, Xiangming Zhu, Yunbo Wang, Xiaokang Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9316769afaaeeaad42a9e3633b14e801-Abstract-Conference.html)

        **Abstract**:

        World models learn the consequences of actions in vision-based interactive systems. However, in practical scenarios such as autonomous driving, there commonly exists noncontrollable dynamics independent of the action signals, making it difficult to learn effective world models. Naturally, therefore, we need to enable the world models to decouple the controllable and noncontrollable dynamics from the entangled spatiotemporal data. To this end, we present a reinforcement learning approach named Iso-Dream, which expands the Dream-to-Control framework in two aspects. First, the world model contains a three-branch neural architecture. By solving the inverse dynamics problem, it learns to factorize latent representations according to the responses to action signals. Second, in the process of behavior learning, we estimate the state values by rolling-out a sequence of noncontrollable states (less related to the actions) into the future and associate the current controllable state with them. In this way, the isolation of mixed dynamics can greatly facilitate long-horizon decision-making tasks in realistic scenes, such as avoiding potential future risks by predicting the movement of other vehicles in autonomous driving. Experiments show that Iso-Dream is effective in decoupling the mixed dynamics and remarkably outperforms existing approaches in a wide range of visual control and prediction domains.

        ----

        ## [1684] PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies

        **Authors**: *Guocheng Qian, Yuchen Li, Houwen Peng, Jinjie Mai, Hasan Hammoud, Mohamed Elhoseiny, Bernard Ghanem*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9318763d049edf9a1f2779b2a59911d3-Abstract-Conference.html)

        **Abstract**:

        PointNet++ is one of the most influential neural architectures for point cloud understanding. Although the accuracy of PointNet++ has been largely surpassed by recent networks such as PointMLP and Point Transformer, we find that a large portion of the performance gain is due to improved training strategies, i.e. data augmentation and optimization techniques, and increased model sizes rather than architectural innovations. Thus, the full potential of PointNet++ has yet to be explored. In this work, we revisit the classical PointNet++ through a systematic study of model training and scaling strategies, and offer two major contributions. First, we propose a set of improved training strategies that significantly improve PointNet++ performance. For example, we show that, without any change in architecture, the overall accuracy (OA) of PointNet++ on ScanObjectNN object classification can be raised from 77.9% to 86.1%, even outperforming state-of-the-art PointMLP. Second, we introduce an inverted residual bottleneck design and separable MLPs into PointNet++ to enable efficient and effective model scaling and propose PointNeXt, the next version of PointNets. PointNeXt can be flexibly scaled up and outperforms state-of-the-art methods on both 3D classification and segmentation tasks. For classification, PointNeXt reaches an overall accuracy of 87.7 on ScanObjectNN, surpassing PointMLP by 2.3%, while being 10x faster in inference. For semantic segmentation, PointNeXt establishes a new state-of-the-art performance with 74.9% mean IoU on S3DIS (6-fold cross-validation), being superior to the recent Point Transformer. The code and models are available at https://github.com/guochengqian/pointnext.

        ----

        ## [1685] Quantum Algorithms for Sampling Log-Concave Distributions and Estimating Normalizing Constants

        **Authors**: *Andrew M. Childs, Tongyang Li, Jin-Peng Liu, Chunhao Wang, Ruizhe Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/933e953353c25ec70477ef28e45a2dcc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/933e953353c25ec70477ef28e45a2dcc-Abstract-Conference.html)

        **Abstract**:

        Given a convex function $f\colon\mathbb{R}^{d}\to\mathbb{R}$, the problem of sampling from a distribution $\propto e^{-f(x)}$ is called log-concave sampling. This task has wide applications in machine learning, physics, statistics, etc. In this work, we develop quantum algorithms for sampling log-concave distributions and for estimating their normalizing constants $\int_{\mathbb{R}^d}e^{-f(x)}\mathrm{d} x$. First, we use underdamped Langevin diffusion to develop quantum algorithms that match the query complexity (in terms of the condition number $\kappa$ and dimension $d$) of analogous classical algorithms that use gradient (first-order) queries, even though the quantum algorithms use only evaluation (zeroth-order) queries. For estimating normalizing constants, these algorithms also achieve quadratic speedup in the multiplicative error $\epsilon$. Second, we develop quantum Metropolis-adjusted Langevin algorithms with query complexity $\widetilde{O}(\kappa^{1/2}d)$ and $\widetilde{O}(\kappa^{1/2}d^{3/2}/\epsilon)$ for log-concave sampling and normalizing constant estimation, respectively, achieving polynomial speedups in $\kappa,d,\epsilon$ over the best known classical algorithms by exploiting quantum analogs of the Monte Carlo method and quantum walks. We also prove a $1/\epsilon^{1-o(1)}$ quantum lower bound for estimating normalizing constants, implying near-optimality of our quantum algorithms in $\epsilon$.

        ----

        ## [1686] Physics-Embedded Neural Networks: Graph Neural PDE Solvers with Mixed Boundary Conditions

        **Authors**: *Masanobu Horie, Naoto Mitsume*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93476ae409ae3246e22a9d4b931f84ed-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/93476ae409ae3246e22a9d4b931f84ed-Abstract-Conference.html)

        **Abstract**:

        Graph neural network (GNN) is a promising approach to learning and predicting physical phenomena described in boundary value problems, such as partial differential equations (PDEs) with boundary conditions. However, existing models inadequately treat boundary conditions essential for the reliable prediction of such problems. In addition, because of the locally connected nature of GNNs, it is difficult to accurately predict the state after a long time, where interaction between vertices tends to be global. We present our approach termed physics-embedded neural networks that considers boundary conditions and predicts the state after a long time using an implicit method. It is built based on an $\mathrm{E}(n)$-equivariant GNN, resulting in high generalization performance on various shapes. We demonstrate that our model learns flow phenomena in complex shapes and outperforms a well-optimized classical solver and a state-of-the-art machine learning model in speed-accuracy trade-off. Therefore, our model can be a useful standard for realizing reliable, fast, and accurate GNN-based PDE solvers. The code is available at https://github.com/yellowshippo/penn-neurips2022.

        ----

        ## [1687] Mismatched No More: Joint Model-Policy Optimization for Model-Based RL

        **Authors**: *Benjamin Eysenbach, Alexander Khazatsky, Sergey Levine, Ruslan Salakhutdinov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/935151cc6cb5d8b6816133b75233775a-Abstract-Conference.html)

        **Abstract**:

        Many model-based reinforcement learning (RL) methods follow a similar template: fit a model to previously observed data, and then use data from that model for RL or planning. However, models that achieve better training performance (e.g., lower MSE) are not necessarily better for control: an RL agent may seek out the small fraction of states where an accurate model makes mistakes, or it might act in ways that do not expose the errors of an inaccurate model. As noted in prior work, there is an objective mismatch: models are useful if they yield good policies, but they are trained to maximize their accuracy, rather than the performance of the policies that result from them.  In this work, we propose a single objective for jointly training the model and the policy, such that updates to either component increase a lower bound on expected return. To the best of our knowledge, this is the first lower bound for model-based RL that holds globally and can be efficiently estimated in continuous settings; it is the only lower bound that mends the objective mismatch problem. A version of this bound becomes tight under certain assumptions. Optimizing this bound resembles a GAN: a classifier distinguishes between real and fake transitions, the model is updated to produce transitions that look realistic, and the policy is updated to avoid states where the model predictions are unrealistic. Numerical simulations demonstrate that optimizing this bound yields reward maximizing policies and yields dynamics that (perhaps surprisingly) can aid in exploration. We also show that a deep RL algorithm loosely based on our lower bound can achieve performance competitive with prior model-based methods, and better performance on certain hard exploration tasks.

        ----

        ## [1688] Phase diagram of Stochastic Gradient Descent in high-dimensional two-layer neural networks

        **Authors**: *Rodrigo Veiga, Ludovic Stephan, Bruno Loureiro, Florent Krzakala, Lenka Zdeborová*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/939bb847ebfd14c6e4d3b5705e562054-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/939bb847ebfd14c6e4d3b5705e562054-Abstract-Conference.html)

        **Abstract**:

        Despite the non-convex optimization landscape, over-parametrized shallow networks are able to achieve global convergence under gradient descent. The picture can be radically different for narrow networks, which tend to get stuck in badly-generalizing local minima. Here we investigate the cross-over between these two regimes in the high-dimensional setting, and in particular investigate the connection between the so-called mean-field/hydrodynamic regime and the seminal approach of Saad \& Solla. Focusing on the case of Gaussian data, we study the interplay between the learning rate, the time scale, and the number of hidden units in the high-dimensional dynamics of stochastic gradient descent (SGD). Our work builds on a deterministic description of SGD in high-dimensions from statistical physics, which we extend and for which we provide rigorous convergence rates.

        ----

        ## [1689] Constrained Stochastic Nonconvex Optimization with State-dependent Markov Data

        **Authors**: *Abhishek Roy, Krishnakumar Balasubramanian, Saeed Ghadimi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93b11b5128ced940120f41ce9b216f39-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/93b11b5128ced940120f41ce9b216f39-Abstract-Conference.html)

        **Abstract**:

        We study stochastic optimization algorithms for constrained nonconvex stochastic optimization problems with Markovian data. In particular, we focus on the case when the transition kernel of the Markov chain is state-dependent. Such stochastic optimization problems arise in various machine learning problems including strategic classification and reinforcement learning. For this problem, we study both projection-based and projection-free algorithms. In both cases, we establish that the number of calls to the stochastic first-order oracle to obtain an appropriately defined $\epsilon$-stationary point is of the order $\mathcal{O}(1/\epsilon^{2.5})$. In the projection-free setting we additionally establish that the number of calls to the linear minimization oracle is of order $\mathcal{O}(1/\epsilon^{5.5})$. We also empirically demonstrate the performance of our algorithm on the problem of strategic classification with neural networks.

        ----

        ## [1690] Adapting Self-Supervised Vision Transformers by Probing Attention-Conditioned Masking Consistency

        **Authors**: *Viraj Prabhu, Sriram Yenamandra, Aaditya Singh, Judy Hoffman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93b4d708976a1d9b1250c400e7fda811-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/93b4d708976a1d9b1250c400e7fda811-Abstract-Conference.html)

        **Abstract**:

        Visual domain adaptation (DA) seeks to transfer trained models to unseen, unlabeled domains across distribution shift, but approaches typically focus on adapting convolutional neural network architectures initialized with supervised ImageNet representations. In this work, we shift focus to adapting modern architectures for object recognition -- the increasingly popular Vision Transformer (ViT) -- initialized with modern pretraining based on self-supervised learning (SSL). Inspired by the design of recent SSL approaches based on learning from partial image inputs generated via masking or cropping -- either by learning to predict the missing pixels, or learning representational invariances to such augmentations -- we propose PACMAC, a two-stage adaptation algorithm for self-supervised ViTs. PACMAC first performs in-domain SSL on pooled source and target data to learn task-discriminative features, and then probes the model's predictive consistency across a set of partial target inputs generated via a novel attention-conditioned masking strategy, to identify reliable candidates for self-training. Our simple approach leads to consistent performance gains over competing methods that use ViTs and self-supervised initializations on standard object recognition benchmarks. Our code is available at https://github.com/virajprabhu/PACMAC.

        ----

        ## [1691] MaskTune: Mitigating Spurious Correlations by Forcing to Explore

        **Authors**: *Saeid Asgari Taghanaki, Aliasghar Khani, Fereshte Khani, Ali Gholami, Linh Tran, Ali Mahdavi-Amiri, Ghassan Hamarneh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/93be245fce00a9bb2333c17ceae4b732-Abstract-Conference.html)

        **Abstract**:

        A fundamental challenge of over-parameterized deep learning models is learning meaningful data representations that yield good performance on a downstream task without over-fitting spurious input features. This work proposes MaskTune, a masking strategy that prevents over-reliance on spurious (or a limited number of) features. MaskTune forces the trained model to explore new features during a single epoch finetuning by masking previously discovered features. MaskTune, unlike earlier approaches for mitigating shortcut learning, does not require any supervision, such as annotating spurious features or labels for subgroup samples in a dataset. Our empirical results on biased MNIST, CelebA, Waterbirds, and ImagenNet-9L datasets show that MaskTune is effective on tasks that often suffer from the existence of spurious correlations. Finally, we show that \method{} outperforms or achieves similar performance to the competing methods when applied to the selective classification (classification with rejection option) task. Code for MaskTune is available at https://github.com/aliasgharkhani/Masktune.

        ----

        ## [1692] A Dataset for Efforts Towards Achieving the Sustainable Development Goal of Safe Working Environments

        **Authors**: *Eirik Lund Flogard, Ole Jakob Mengshoel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93e4d161bdd93d1dc0202b4044159edb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/93e4d161bdd93d1dc0202b4044159edb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Among United Nations' 17 Sustainable Development Goals (SDGs), we highlight SDG 8 on Decent Work and Economic Growth.  Specifically, we consider how to achieve subgoal 8.8, "protect labour rights and promote safe working environments for all workers [...]", in light of poor health, safety and environment (HSE) conditions being a widespread problem at workplaces. In EU alone, it is estimated that more than 4000 deaths occur each year due to poor working conditions. To handle the problem and achieve SDG 8, governmental agencies conduct labour inspections and it is therefore essential that these are carried out efficiently. Current research suggests that machine learning (ML) can be used to improve labour inspections, for instance by selecting organisations for inspections more effectively. However, the research in this area is very limited, in part due to a lack of publicly available data. Consequently, we introduce a new dataset called the Labour Inspection Checklists Dataset (LICD), which we have made publicly available. LICD consists of 63634 instances where each instance is an inspection conducted by the Norwegian Labour Inspection Authority. LICD has 577 features and labels. The dataset provides several ML research opportunities; we discuss two demonstration experiments. One experiment deals with the problem of selecting a relevant checklist for inspecting a given target organisation. The other experiment concerns the problem of predicting HSE violations, given a specific checklist and a target organisation. Our experimental results, while promising, suggest that achieving good ML classification performance is difficult for both problems. This motivates future research to improve ML performance, inspire other data analysis efforts, and ultimately achieve SDG 8.

        ----

        ## [1693] Decomposing NeRF for Editing via Feature Field Distillation

        **Authors**: *Sosuke Kobayashi, Eiichi Matsumoto, Vincent Sitzmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/93f250215e4889119807b6fac3a57aec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/93f250215e4889119807b6fac3a57aec-Abstract-Conference.html)

        **Abstract**:

        Emerging neural radiance fields (NeRF) are a promising scene representation for computer graphics, enabling high-quality 3D reconstruction and novel view synthesis from image observations.However, editing a scene represented by a NeRF is challenging, as the underlying connectionist representations such as MLPs or voxel grids are not object-centric or compositional.In particular, it has been difficult to selectively edit specific regions or objects.In this work, we tackle the problem of semantic scene decomposition of NeRFs to enable query-based local editing of the represented 3D scenes.We propose to distill the knowledge of off-the-shelf, self-supervised 2D image feature extractors such as CLIP-LSeg or DINO into a 3D feature field optimized in parallel to the radiance field.Given a user-specified query of various modalities such as text, an image patch, or a point-and-click selection, 3D feature fields semantically decompose 3D space without the need for re-training, and enables us to semantically select and edit regions in the radiance field.Our experiments validate that the distilled feature fields can transfer recent progress in 2D vision and language foundation models to 3D scene representations, enabling convincing 3D segmentation and selective editing of emerging neural graphics representations.

        ----

        ## [1694] Sample-Then-Optimize Batch Neural Thompson Sampling

        **Authors**: *Zhongxiang Dai, Yao Shu, Bryan Kian Hsiang Low, Patrick Jaillet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/940f8a526b8b36f110265f4b6059d81b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/940f8a526b8b36f110265f4b6059d81b-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimization (BO), which uses a Gaussian process (GP) as a surrogate to model its objective function, is popular for black-box optimization. However, due to the limitations of GPs, BO underperforms in some problems such as those with categorical, high-dimensional or image inputs. To this end, recent works have used the highly expressive neural networks (NNs) as the surrogate model and derived theoretical guarantees using the theory of neural tangent kernel (NTK). However, these works suffer from the limitations of the requirement to invert an extremely large parameter matrix and the restriction to the sequential (rather than batch) setting. To overcome these limitations, we introduce two algorithms based on the Thompson sampling (TS) policy named Sample-Then-Optimize Batch Neural TS (STO-BNTS) and STO-BNTS-Linear. To choose an input query, we only need to train an NN (resp. a linear model) and then choose the query by maximizing the trained NN (resp. linear model), which is equivalently sampled from the GP posterior with the NTK as the kernel function. As a result, our algorithms sidestep the need to invert the large parameter matrix yet still preserve the validity of the TS policy. Next, we derive regret upper bounds for our algorithms with batch evaluations, and use insights from batch BO and NTK to show that they are asymptotically no-regret under certain conditions. Finally, we verify their empirical effectiveness using practical AutoML and reinforcement learning experiments.

        ----

        ## [1695] Video-based Human-Object Interaction Detection from Tubelet Tokens

        **Authors**: *Danyang Tu, Wei Sun, Xiongkuo Min, Guangtao Zhai, Wei Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9415416201aa201902d1743c7e65787b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9415416201aa201902d1743c7e65787b-Abstract-Conference.html)

        **Abstract**:

        We present a novel vision Transformer, named TUTOR, which is able to learn tubelet tokens, served as highly-abstracted spatial-temporal representations, for video-based human-object interaction (V-HOI) detection. The tubelet tokens structurize videos by agglomerating and linking semantically-related patch tokens along spatial and temporal domains, which enjoy two benefits: 1) Compactness: each token is learned by a selective attention mechanism to reduce redundant dependencies from others; 2) Expressiveness: each token is enabled to align with a semantic instance, i.e., an object or a human, thanks to agglomeration and linking. The effectiveness and efficiency of TUTOR are verified by extensive experiments. Results show our method outperforms existing works by large margins, with a relative mAP gain of $16.14\%$ on VidHOI and a 2 points gain on CAD-120 as well as a $4 \times$ speedup.

        ----

        ## [1696] Efficient and Stable Fully Dynamic Facility Location

        **Authors**: *Sayan Bhattacharya, Silvio Lattanzi, Nikos Parotsidis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/943d6dca1884955e645d8997ae2fa938-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/943d6dca1884955e645d8997ae2fa938-Abstract-Conference.html)

        **Abstract**:

        We consider the classic facility location problem in fully dynamic data streams, where elements can be both inserted and deleted. In this problem, one is interested in maintaining a stable and high quality solution throughout the data stream while using only little time per update (insertion or deletion). We study the problem and provide the first algorithm that at the same time maintains a constant approximation and incurs polylogarithmic amortized recourse per update. We complement our theoretical results with an experimental analysis showing the practical efficiency of our method.

        ----

        ## [1697] MCVD - Masked Conditional Video Diffusion for Prediction, Generation, and Interpolation

        **Authors**: *Vikram Voleti, Alexia Jolicoeur-Martineau, Chris Pal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/944618542d80a63bbec16dfbd2bd689a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/944618542d80a63bbec16dfbd2bd689a-Abstract-Conference.html)

        **Abstract**:

        Video prediction is a challenging task. The quality of video frames from current state-of-the-art (SOTA) generative models tends to be poor and generalization beyond the training data is difficult. Furthermore, existing prediction frameworks are typically not capable of simultaneously handling other video-related tasks such as unconditional generation or interpolation. In this work, we devise a general-purpose framework called Masked Conditional Video Diffusion (MCVD) for all of these video synthesis tasks using a probabilistic conditional score-based denoising diffusion model, conditioned on past and/or future frames. We train the model in a manner where we randomly and independently mask all the past frames or all the future frames. This novel but straightforward setup allows us to train a single model that is capable of executing a broad range of video tasks, specifically: future/past prediction -- when only future/past frames are masked; unconditional generation -- when both past and future frames are masked; and interpolation -- when neither past nor future frames are masked. Our experiments show that this approach can generate high-quality frames for diverse types of videos. Our MCVD models are built from simple non-recurrent 2D-convolutional architectures, conditioning on blocks of frames and generating blocks of frames. We generate videos of arbitrary lengths autoregressively in a block-wise manner. Our approach yields SOTA results across standard video prediction and interpolation benchmarks, with computation times for training models measured in 1-12 days using $\le$ 4 GPUs. Project page: \url{https://mask-cond-video-diffusion.github.io}Code: \url{https://mask-cond-video-diffusion.github.io/}

        ----

        ## [1698] Addressing Leakage in Concept Bottleneck Models

        **Authors**: *Marton Havasi, Sonali Parbhoo, Finale Doshi-Velez*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/944ecf65a46feb578a43abfd5cddd960-Abstract-Conference.html)

        **Abstract**:

        Concept bottleneck models (CBMs) enhance the interpretability of their predictions by first predicting high-level concepts given features, and subsequently predicting outcomes on the basis of these concepts.  Recently, it was demonstrated that training the label predictor directly on the probabilities produced by the concept predictor as opposed to the ground-truth concepts, improves label predictions. However, this results in corruptions in the concept predictions that impact the concept accuracy as well as our ability to intervene on the concepts -- a key proposed benefit of CBMs. In this work, we investigate and address two issues with CBMs that cause this disparity in performance: having an insufficient concept set and using inexpressive concept predictor. With our modifications, CBMs become competitive in terms of predictive performance, with models that otherwise leak additional information in the concept probabilities, while having dramatically increased concept accuracy and intervention accuracy.

        ----

        ## [1699] Adaptive Oracle-Efficient Online Learning

        **Authors**: *Guanghui Wang, Zihao Hu, Vidya Muthukumar, Jacob D. Abernethy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/948106cb5a114684a64c89a1e517e3fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/948106cb5a114684a64c89a1e517e3fe-Abstract-Conference.html)

        **Abstract**:

        The classical algorithms for online learning and decision-making have the benefit of achieving the optimal performance guarantees, but suffer from computational complexity limitations when implemented at scale. More recent sophisticated techniques, which we refer to as $\textit{oracle-efficient}$ methods, address this problem by dispatching to an $\textit{offline optimization oracle}$ that can search through an exponentially-large (or even infinite) space of decisions and select that which performed the best on any dataset. But despite the benefits of computational feasibility, most oracle-efficient algorithms exhibit one major limitation: while performing well in worst-case settings, they do not adapt well to friendly environments. In this paper we consider two such friendly scenarios, (a) "small-loss" problems and (b) IID data. We provide a new framework for designing follow-the-perturbed-leader algorithms that are oracle-efficient and adapt well to the small-loss environment, under a particular condition which we call $\textit{approximability}$ (which is spiritually related to sufficient conditions provided in (Dud√≠k et al., 2020)). We identify a series of real-world settings, including online auctions and transductive online classification, for which approximability holds. We also extend the algorithm to an IID data setting and establish a "best-of-both-worlds" bound in the oracle-efficient setting.

        ----

        ## [1700] MoVQ: Modulating Quantized Vectors for High-Fidelity Image Generation

        **Authors**: *Chuanxia Zheng, Tung-Long Vuong, Jianfei Cai, Dinh Phung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94840c41497ead6a84f493f029eba7fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/94840c41497ead6a84f493f029eba7fa-Abstract-Conference.html)

        **Abstract**:

        Although two-stage Vector Quantized (VQ) generative models allow for synthesizing high-fidelity and high-resolution images, their quantization operator encodes similar patches within an image into the same index, resulting in a repeated artifact for similar adjacent regions using existing decoder architectures. To address this issue, we propose to incorporate the spatially conditional normalization to modulate the quantized vectors so as to insert spatially variant information to the embedded index maps, encouraging the decoder to generate more photorealistic images. Moreover, we use multichannel quantization to increase the recombination capability of the discrete codes without increasing the cost of model and codebook. Additionally, to generate discrete tokens at the second stage, we adopt a Masked Generative Image Transformer (MaskGIT) to learn an underlying prior distribution in the compressed latent space, which is much faster than the conventional autoregressive model. Experiments on two benchmark datasets demonstrate that our proposed modulated VQGAN is able to greatly improve the reconstructed image quality as well as provide high-fidelity image generation.

        ----

        ## [1701] Meta-Auto-Decoder for Solving Parametric Partial Differential Equations

        **Authors**: *Xiang Huang, Zhanhong Ye, Hongsheng Liu, Shi Ji, Zidong Wang, Kang Yang, Yang Li, Min Wang, Haotian Chu, Fan Yu, Bei Hua, Lei Chen, Bin Dong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/948552777302d3abf92415b1d7e9de70-Abstract-Conference.html)

        **Abstract**:

        Many important problems in science and engineering require solving the so-called parametric partial differential equations (PDEs), i.e., PDEs with different physical parameters, boundary conditions, shapes of computation domains, etc.  Recently, building learning-based numerical solvers for parametric PDEs has become an emerging new field.  One category of methods such as the Deep Galerkin Method (DGM) and Physics-Informed Neural Networks (PINNs) aim to approximate the solution of the PDEs. They are typically unsupervised and mesh-free, but require going through the time-consuming network training process from scratch for each set of parameters of the PDE.  Another category of methods such as Fourier Neural Operator (FNO) and Deep Operator Network (DeepONet) try to approximate the solution mapping directly.  Being fast with only one forward inference for each PDE parameter without retraining, they often require a large corpus of paired input-output observations drawn from numerical simulations, and most of them need a predefined mesh as well.  In this paper, we propose Meta-Auto-Decoder (MAD), a mesh-free and unsupervised deep learning method that enables the pre-trained model to be quickly adapted to equation instances by implicitly encoding (possibly heterogenous) PDE parameters as latent vectors.  The proposed method MAD can be interpreted by manifold learning in infinite-dimensional spaces, granting it a geometric insight.  Extensive numerical experiments show that the MAD method exhibits faster convergence speed without losing accuracy than other deep learning-based methods.

        ----

        ## [1702] Sharpness-Aware Training for Free

        **Authors**: *Jiawei Du, Daquan Zhou, Jiashi Feng, Vincent Y. F. Tan, Joey Tianyi Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/948b1c9d660d7286dd767cd07dabd487-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/948b1c9d660d7286dd767cd07dabd487-Abstract-Conference.html)

        **Abstract**:

        Modern deep neural networks (DNNs) have achieved state-of-the-art performances but are typically over-parameterized. The over-parameterization may result in undesirably large generalization error in the absence of other customized training strategies. Recently, a line of research under the name of Sharpness-Aware Minimization (SAM) has shown that minimizing a sharpness measure, which reflects the geometry of the loss landscape, can significantly reduce the generalization error. However, SAM-like methods incur a two-fold computational overhead of the given base optimizer (e.g. SGD) for approximating the sharpness measure. In this paper, we propose Sharpness-Aware Training for Free, or SAF, which mitigates the sharp landscape at almost zero additional computational cost over the base optimizer. Intuitively, SAF achieves this by avoiding sudden drops in the loss in the sharp local minima throughout the trajectory of the updates of the weights. Specifically, we suggest a novel trajectory loss, based on the KL-divergence between the outputs of DNNs with the current weights and past weights, as a replacement of the SAM's sharpness measure. This loss captures the rate of change of the training loss along the model's update trajectory. By minimizing it, SAF ensures the convergence to a flat minimum with improved generalization capabilities. Extensive empirical results show that SAF minimizes the sharpness in the same way that SAM does, yielding better results on the ImageNet dataset with essentially the same computational cost as the base optimizer.

        ----

        ## [1703] Generalization Error Bounds on Deep Learning with Markov Datasets

        **Authors**: *Lan V. Truong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/949b3011c50300a2b4e60377466f52a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/949b3011c50300a2b4e60377466f52a8-Abstract-Conference.html)

        **Abstract**:

        In this paper, we derive upper bounds on generalization errors for deep neural networks with Markov datasets. These bounds are developed based on Koltchinskii and Panchenko's approach for bounding the generalization error of combined classifiers with i.i.d. datasets. The development of new symmetrization inequalities in high-dimensional probability for Markov chains is a key element in our extension, where the spectral gap of the infinitesimal generator of the Markov chain plays a key parameter in these inequalities. We also propose a simple method to convert these bounds and other similar ones in traditional deep learning and machine learning to Bayesian counterparts for both i.i.d. and Markov datasets. Extensions to $m$-order homogeneous Markov chains such as AR and ARMA models and mixtures of several Markov data services are given.

        ----

        ## [1704] AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier-Stokes Solutions

        **Authors**: *Florent Bonnet, Jocelyn Ahmed Mazari, Paola Cinnella, Patrick Gallinari*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94ab7b23a345f93333eac8748a66c763-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/94ab7b23a345f93333eac8748a66c763-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Surrogate models are necessary to optimize meaningful quantities in physical dynamics as their recursive numerical resolutions are often prohibitively expensive. It is mainly the case for fluid dynamics and the resolution of Navier–Stokes equations. However, despite the fast-growing field of data-driven models for physical systems, reference datasets representing real-world phenomena are lacking. In this work, we develop \textsc{AirfRANS}, a dataset for studying the two-dimensional incompressible steady-state Reynolds-Averaged Navier–Stokes equations over airfoils at a subsonic regime and for different angles of attacks. We also introduce metrics on the stress forces at the surface of geometries and visualization of boundary layers to assess the capabilities of models to accurately predict the meaningful information of the problem. Finally, we propose deep learning baselines on four machine learning tasks to study \textsc{AirfRANS} under different constraints for generalization considerations: big and scarce data regime, Reynolds number, and angle of attack extrapolation.

        ----

        ## [1705] Generalization for multiclass classification with overparameterized linear models

        **Authors**: *Vignesh Subramanian, Rahul Arya, Anant Sahai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/94b472a1842cd7c56dcb125fb2765fbd-Abstract-Conference.html)

        **Abstract**:

        Via an overparameterized linear model with Gaussian features, we provide conditions for good generalization for multiclass classification of minimum-norm interpolating solutions in an asymptotic setting where both the number of underlying features and the number of classes scale with the number of training points. The survival/contamination analysis framework for understanding the behavior of overparameterized learning problems is adapted to this setting, revealing that multiclass classification qualitatively behaves like binary classification in that, as long as there are not too many classes (made precise in the paper), it is possible to generalize well even in settings where regression tasks would not generalize. Besides various technical challenges, it turns out that the key difference from the binary classification setting is that there are relatively fewer training examples of each class in the multiclass setting as the number of classes increases, making the multiclass problem ``harder'' than the binary one.

        ----

        ## [1706] Inception Transformer

        **Authors**: *Chenyang Si, Weihao Yu, Pan Zhou, Yichen Zhou, Xinchao Wang, Shuicheng Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/94e85561a342de88b559b72c9b29f638-Abstract-Conference.html)

        **Abstract**:

        Recent studies show that transformer has strong capability of building long-range dependencies, yet is incompetent in capturing high frequencies that predominantly convey local information. To tackle this issue, we present a novel and general-purpose $\textit{Inception Transformer}$, or $\textit{iFormer}$ for short, that effectively learns comprehensive features with both high- and low-frequency information in visual data. Specifically,  we design an Inception mixer to explicitly graft the advantages of convolution and max-pooling for capturing the high-frequency information to transformers. Different from recent hybrid frameworks, the Inception mixer brings greater efficiency through a channel splitting mechanism to adopt parallel convolution/max-pooling path and self-attention path as high- and low-frequency mixers, while having the flexibility to model discriminative information scattered within a wide frequency range. Considering that bottom layers play more roles in capturing high-frequency details while top layers more in modeling low-frequency global information, we further introduce a frequency ramp structure, i.e., gradually decreasing the dimensions fed to the high-frequency mixer and increasing those to the low-frequency mixer, which can effectively trade-off high- and low-frequency components across different layers. We benchmark the iFormer on a series of vision tasks, and showcase that it achieves impressive performance on  image classification, COCO detection and ADE20K segmentation. For example, our iFormer-S hits the top-1 accuracy of 83.4% on ImageNet-1K, much higher than DeiT-S by 3.6%, and even slightly better than much bigger model Swin-B (83.3%) with only 1/4 parameters and 1/3 FLOPs. Code and models are released at https://github.com/sail-sg/iFormer.

        ----

        ## [1707] ElasticMVS: Learning elastic part representation for self-supervised multi-view stereopsis

        **Authors**: *Jinzhi Zhang, Ruofan Tang, Zheng Cao, Jing Xiao, Ruqi Huang, Lu Fang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94ef721705ea95d6981632be62bb66e2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/94ef721705ea95d6981632be62bb66e2-Abstract-Conference.html)

        **Abstract**:

        Self-supervised multi-view stereopsis (MVS) attracts increasing attention for learning dense surface predictions from only a set of images without onerous ground-truth 3D training data for supervision. However, existing methods highly rely on the local photometric consistency, which fails to identify accurately dense correspondence in broad textureless and reflectance areas.In this paper, we show that geometric proximity such as surface connectedness and occlusion boundaries implicitly inferred from images could serve as reliable guidance for pixel-wise multi-view correspondences. With this insight, we present a novel elastic part representation which encodes physically-connected part segmentations with elastically-varying scales, shapes and boundaries. Meanwhile, a self-supervised MVS framework namely ElasticMVS is proposed to learn the representation and estimate per-view depth following a part-aware propagation and evaluation scheme. Specifically, the pixel-wise part representation is trained by a contrastive learning-based strategy, which increases the representation compactness in geometrically concentrated areas and contrasts otherwise. ElasticMVS iteratively optimizes a part-level consistency loss and a surface smoothness loss, based on a set of depth hypotheses propagated from the geometrically concentrated parts. Extensive evaluations convey the superiority of ElasticMVS in the reconstruction completeness and accuracy, as well as the efficiency and scalability. Particularly, for the challenging large-scale reconstruction benchmark, ElasticMVS demonstrates significant performance gain over both the supervised and self-supervised approaches.

        ----

        ## [1708] Adaptive Stochastic Variance Reduction for Non-convex Finite-Sum Minimization

        **Authors**: *Ali Kavis, Stratis Skoulakis, Kimon Antonakopoulos, Leello Tadesse Dadi, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/94f625dcdec313cd432d65f96fcc51c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/94f625dcdec313cd432d65f96fcc51c8-Abstract-Conference.html)

        **Abstract**:

        We propose an adaptive variance-reduction method, called AdaSpider, for minimization of $L$-smooth, non-convex functions with a finite-sum structure. In essence, AdaSpider combines an AdaGrad-inspired (Duchi et al., 2011), but a fairly distinct, adaptive step-size schedule with the recursive \textit{stochastic path integrated estimator} proposed in (Fang et al., 2018). To our knowledge, AdaSpider is the first parameter-free non-convex variance-reduction method in the sense that it does not require the knowledge of problem-dependent parameters, such as smoothness constant $L$, target accuracy $\epsilon$ or any bound on gradient norms. In doing so, we are able to compute an $\epsilon$-stationary point with $\tilde{O}\left(n + \sqrt{n}/\epsilon^2\right)$ oracle-calls, which matches the respective lower bound up to logarithmic factors.

        ----

        ## [1709] Thinking Outside the Ball: Optimal Learning with Gradient Descent for Generalized Linear Stochastic Convex Optimization

        **Authors**: *Idan Amir, Roi Livni, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9521b6e7f33e039e7d92e23f5e37bbf4-Abstract-Conference.html)

        **Abstract**:

        We consider linear prediction with a convex Lipschitz loss, or more generally, stochastic convex optimization problems of generalized linear form, i.e.~where each instantaneous loss is a scalar convex function of a linear function.  We show that in this setting, early stopped Gradient Descent (GD), without any explicit regularization or projection, ensures excess error at most $\varepsilon$ (compared to the best possible with unit Euclidean norm) with an optimal, up to logarithmic factors, sample complexity of $\tilde{O}(1/\varepsilon^2)$ and only $\tilde{O}(1/\varepsilon^2)$ iterations.  This contrasts with general stochastic convex optimization, where $\Omega(1/\varepsilon^4)$ iterations are needed Amir et al. 2021. The lower iteration complexity is ensured by leveraging uniform convergence rather than stability.  But instead of uniform convergence in a norm ball, which we show can guarantee suboptimal learning using $\Theta(1/\varepsilon^4)$ samples, we rely on uniform convergence in a distribution-dependent ball.

        ----

        ## [1710] Generalization Properties of NAS under Activation and Skip Connection Search

        **Authors**: *Zhenyu Zhu, Fanghui Liu, Grigorios Chrysos, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/952b691c116bf753daafa6ce274e81bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/952b691c116bf753daafa6ce274e81bb-Abstract-Conference.html)

        **Abstract**:

        Neural Architecture Search (NAS) has fostered the automatic discovery of state-of-the-art neural architectures. Despite the progress achieved with NAS, so far there is little attention to theoretical guarantees on NAS. In this work, we study the generalization properties of NAS under a unifying framework enabling (deep) layer skip connection search and activation function search. To this end, we derive the lower (and upper) bounds of the minimum eigenvalue of the Neural Tangent Kernel (NTK) under the (in)finite-width regime using a certain search space including mixed activation functions, fully connected, and residual neural networks. We use the minimum eigenvalue to establish generalization error bounds of NAS in the stochastic gradient descent training. Importantly, we theoretically and experimentally show how the derived results can guide NAS to select the top-performing architectures, even in the case without training, leading to a train-free algorithm based on our theory. Accordingly, our numerical validation shed light on the design of computationally efficient methods for NAS. Our analysis is non-trivial due to the coupling of various architectures and activation functions under the unifying framework and has its own interest in providing the lower bound of the minimum eigenvalue of NTK in deep learning theory.

        ----

        ## [1711] Mesoscopic modeling of hidden spiking neurons

        **Authors**: *Shuqi Wang, Valentin Schmutz, Guillaume Bellec, Wulfram Gerstner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/953e742190ca02fc8f9f710052f2fead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/953e742190ca02fc8f9f710052f2fead-Abstract-Conference.html)

        **Abstract**:

        Can we use spiking neural networks (SNN) as generative models of multi-neuronal recordings, while taking into account that most neurons are unobserved? Modeling the unobserved neurons with large pools of hidden spiking neurons leads to severely underconstrained problems that are hard to tackle with maximum likelihood estimation. In this work, we use coarse-graining and mean-field approximations to derive a bottom-up, neuronally-grounded latent variable model (neuLVM), where the activity of the unobserved neurons is reduced to a low-dimensional mesoscopic description. In contrast to previous latent variable models, neuLVM can be explicitly mapped to a recurrent, multi-population SNN, giving it a transparent biological interpretation. We show, on synthetic spike trains, that a few observed neurons are sufficient for neuLVM to perform efficient model inversion of large SNNs, in the sense that it can recover connectivity parameters, infer single-trial latent population activity, reproduce ongoing metastable dynamics, and generalize when subjected to perturbations mimicking optogenetic stimulation.

        ----

        ## [1712] SageMix: Saliency-Guided Mixup for Point Clouds

        **Authors**: *Sanghyeok Lee, Minkyu Jeon, Injae Kim, Yunyang Xiong, Hyunwoo J. Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9543942c237ded1b39b1fd37259ff88e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9543942c237ded1b39b1fd37259ff88e-Abstract-Conference.html)

        **Abstract**:

        Data augmentation is key to improving the generalization ability of deep learning models. Mixup is a simple and widely-used data augmentation technique that has proven effective in alleviating the problems of overfitting and data scarcity. Also, recent studies of saliency-aware Mixup in the image domain show that preserving discriminative parts is beneficial to improving the generalization performance. However, these Mixup-based data augmentations are underexplored in 3D vision, especially in point clouds. In this paper, we propose SageMix, a saliency-guided Mixup for point clouds to preserve salient local structures. Specifically, we extract salient regions from two point clouds and smoothly combine them into one continuous shape. With a simple sequential sampling by re-weighted saliency scores, SageMix preserves the local structure of salient regions. Extensive experiments demonstrate that the proposed method consistently outperforms existing Mixup methods in various benchmark point cloud datasets. With PointNet++, our method achieves an accuracy gain of 2.6% and 4.0% over standard training in ModelNet40 and ScanObjectNN, respectively. In addition to generalization performance, SageMix improves robustness and uncertainty calibration. Moreover, when adopting our method to various tasks including part segmentation and standard image classification, our method achieves competitive performance. Code is available at https://github.com/mlvlab/SageMix.

        ----

        ## [1713] Denoising Diffusion Restoration Models

        **Authors**: *Bahjat Kawar, Michael Elad, Stefano Ermon, Jiaming Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95504595b6169131b6ed6cd72eb05616-Abstract-Conference.html)

        **Abstract**:

        Many interesting tasks in image restoration can be cast as linear inverse problems. A recent family of approaches for solving these problems uses stochastic algorithms that sample from the posterior distribution of natural images given the measurements. However, efficient solutions often require problem-specific supervised training to model the posterior, whereas unsupervised methods that are not problem-specific typically rely on inefficient iterative methods. This work addresses these issues by introducing Denoising Diffusion Restoration Models (DDRM), an efficient, unsupervised posterior sampling method. Motivated by variational inference, DDRM takes advantage of a pre-trained denoising diffusion generative model for solving any linear inverse problem. We demonstrate DDRM's versatility on several image datasets for super-resolution, deblurring, inpainting, and colorization under various amounts of measurement noise. DDRM outperforms the current leading unsupervised methods on the diverse ImageNet dataset in reconstruction quality, perceptual quality, and runtime, being $5\times$ faster than the nearest competitor. DDRM also generalizes well for natural images out of the distribution of the observed ImageNet training set.

        ----

        ## [1714] On Translation and Reconstruction Guarantees of the Cycle-Consistent Generative Adversarial Networks

        **Authors**: *Anish Chakrabarty, Swagatam Das*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/958b631012454391121f96fdc719d034-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/958b631012454391121f96fdc719d034-Abstract-Conference.html)

        **Abstract**:

        The task of unpaired image-to-image translation has witnessed a revolution with the introduction of the cycle-consistency loss to Generative Adversarial Networks (GANs). Numerous variants, with Cycle-Consistent Adversarial Network (CycleGAN) at their forefront, have shown remarkable empirical performance. The involvement of two unalike data spaces and the existence of multiple solution maps between them are some of the facets that make such architectures unique. In this study, we investigate the statistical properties of such unpaired data translator networks between distinct spaces, bearing the additional responsibility of cycle-consistency. In a density estimation setup, we derive sharp non-asymptotic bounds on the translation errors under suitably characterized models. This, in turn, points out sufficient regularity conditions that maps must obey to carry out successful translations. We further show that cycle-consistency is achieved as a consequence of the data being successfully generated in each space based on observations from the other. In a first-of-its-kind attempt, we also provide deterministic bounds on the cumulative reconstruction error. In the process, we establish tolerable upper bounds on the discrepancy responsible for ill-posedness in such networks.

        ----

        ## [1715] Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks

        **Authors**: *Jianan Zhou, Jianing Zhu, Jingfeng Zhang, Tongliang Liu, Gang Niu, Bo Han, Masashi Sugiyama*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/959f70ee50044bed305e48e3484005a7-Abstract-Conference.html)

        **Abstract**:

        Adversarial training (AT) with imperfect supervision is significant but receives limited attention. To push AT towards more practical scenarios, we explore a brand new yet challenging setting, i.e., AT with complementary labels (CLs), which specify a class that a data sample does not belong to. However, the direct combination of AT with existing methods for CLs results in consistent failure, but not on a simple baseline of two-stage training. In this paper, we further explore the phenomenon and identify the underlying challenges of AT with CLs as intractable adversarial optimization and low-quality adversarial examples. To address the above problems, we propose a new learning strategy using gradually informative attacks, which consists of two critical components: 1) Warm-up Attack (Warm-up) gently raises the adversarial perturbation budgets to ease the adversarial optimization with CLs; 2) Pseudo-Label Attack (PLA) incorporates the progressively informative model predictions into a corrected complementary loss. Extensive experiments are conducted to demonstrate the effectiveness of our method on a range of benchmarked datasets. The code is publicly available at: https://github.com/RoyalSkye/ATCL.

        ----

        ## [1716] Tractable Optimality in Episodic Latent MABs

        **Authors**: *Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, Shie Mannor*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95a6fcdc0c8458baa9c6e14736a644f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95a6fcdc0c8458baa9c6e14736a644f8-Abstract-Conference.html)

        **Abstract**:

        We consider a multi-armed bandit problem with $M$ latent contexts, where an agent interacts with the environment for an episode of $H$ time steps. Depending on the length of the episode, the learner may not be able to estimate accurately the latent context. The resulting partial observation of the environment makes the learning task significantly more challenging. Without any additional structural assumptions, existing techniques to tackle partially observed settings imply the decision maker can learn a near-optimal policy with $O(A)^H$ episodes, but do not promise more. In this work, we show that learning with {\em polynomial} samples in $A$ is possible. We achieve this by using techniques from experiment design. Then, through a method-of-moments approach, we design a procedure that provably learns a near-optimal policy with $O(\poly(A) + \poly(M,H)^{\min(M,H)})$ interactions. In practice, we show that we can formulate the moment-matching via maximum likelihood estimation. In our experiments, this significantly outperforms the worst-case guarantees, as well as existing practical methods.

        ----

        ## [1717] A Characterization of Semi-Supervised Adversarially Robust PAC Learnability

        **Authors**: *Idan Attias, Steve Hanneke, Yishay Mansour*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95a704bd2fdf8ef8242b4adcc7ce3c93-Abstract-Conference.html)

        **Abstract**:

        We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model.We address the question of how many labeled and unlabeled examples are required to ensure learning.We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require),the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity.This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.

        ----

        ## [1718] Data-IQ: Characterizing subgroups with heterogeneous outcomes in tabular data

        **Authors**: *Nabeel Seedat, Jonathan Crabbé, Ioana Bica, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95b6e2ff961580e03c0a662a63a71812-Abstract-Conference.html)

        **Abstract**:

        High model performance, on average, can hide that models may systematically underperform on subgroups of the data. We consider the tabular setting, which surfaces the unique issue of outcome heterogeneity - this is prevalent in areas such as healthcare, where patients with similar features can have different outcomes, thus making reliable predictions challenging. To tackle this, we propose Data-IQ, a framework to systematically stratify examples into subgroups with respect to their outcomes. We do this by analyzing the behavior of individual examples during training, based on their predictive confidence and, importantly, the aleatoric (data) uncertainty. Capturing the aleatoric uncertainty permits a principled characterization and then subsequent stratification of data examples into three distinct subgroups (Easy, Ambiguous, Hard). We experimentally demonstrate the benefits of Data-IQ on four real-world medical datasets. We show that Data-IQ's characterization of examples is most robust to variation across similarly performant (yet different models), compared to baselines. Since Data-IQ can be used with any ML model (including neural networks, gradient boosting etc.), this property ensures consistency of data characterization, while allowing flexible model selection. Taking this a step further, we demonstrate that the subgroups enable us to construct new approaches to both feature acquisition and dataset selection. Furthermore, we highlight how the subgroups can inform reliable model usage, noting the significant impact of the Ambiguous subgroup on model generalization.

        ----

        ## [1719] Task-Free Continual Learning via Online Discrepancy Distance Learning

        **Authors**: *Fei Ye, Adrian G. Bors*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95c6ae3f3393786203a4b6dcb9df1036-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95c6ae3f3393786203a4b6dcb9df1036-Abstract-Conference.html)

        **Abstract**:

        Learning from non-stationary data streams, also called Task-Free Continual Learning (TFCL) remains challenging due to the absence of explicit task information in most applications. Even though recently some algorithms have been proposed for TFCL, these methods lack theoretical guarantees. Moreover, there are no theoretical studies about forgetting during TFCL. This paper develops a new theoretical analysis framework that derives generalization bounds based on the discrepancy distance between the visited samples and the entire information made available for training the model. This analysis provides new insights into the forgetting behaviour in classification tasks. Inspired by this theoretical model, we propose a new approach enabled with the dynamic component expansion mechanism for a mixture model, namely Online Discrepancy Distance Learning (ODDL). ODDL estimates the discrepancy between the current memory and the already accumulated knowledge as an expansion signal aiming to ensure a compact network architecture with optimal performance. We then propose a new sample selection approach that selectively stores the samples into the memory buffer through the discrepancy-based measure, further improving the performance. We perform several TFCL experiments with the proposed methodology, which demonstrate that the proposed approach achieves the state of the art performance.

        ----

        ## [1720] BinauralGrad: A Two-Stage Conditional Diffusion Probabilistic Model for Binaural Audio Synthesis

        **Authors**: *Yichong Leng, Zehua Chen, Junliang Guo, Haohe Liu, Jiawei Chen, Xu Tan, Danilo P. Mandic, Lei He, Xiangyang Li, Tao Qin, Sheng Zhao, Tie-Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95f03faf3763e1b1ce2c3de62da8f090-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/95f03faf3763e1b1ce2c3de62da8f090-Abstract-Conference.html)

        **Abstract**:

        Binaural audio plays a significant role in constructing immersive augmented and virtual realities. As it is expensive to record binaural audio from the real world, synthesizing them from mono audio has attracted increasing attention. This synthesis process involves not only the basic physical warping of the mono audio, but also room reverberations and head/ear related filtration, which, however, are difficult to accurately simulate in traditional digital signal processing. In this paper, we formulate the synthesis process from a different perspective by decomposing the binaural audio into a common part that shared by the left and right channels as well as a specific part that differs in each channel. Accordingly, we propose BinauralGrad, a novel two-stage framework equipped with diffusion models to synthesize them respectively. Specifically, in the first stage, the common information of the binaural audio is generated with a single-channel diffusion model conditioned on the mono audio, based on which the binaural audio is generated by a two-channel diffusion model in the second stage. Combining this novel perspective of two-stage synthesis with advanced generative models (i.e., the diffusion models), the proposed BinauralGrad is able to generate accurate and high-fidelity binaural audio samples. Experiment results show that on a benchmark dataset, BinauralGrad outperforms the existing baselines by a large margin in terms of both object and subject evaluation metrics (Wave L2: $0.128$ vs. $0.157$, MOS: $3.80$ vs. $3.61$). The generated audio samples\footnote{\url{https://speechresearch.github.io/binauralgrad}} and code\footnote{\url{https://github.com/microsoft/NeuralSpeech/tree/master/BinauralGrad}} are available online.

        ----

        ## [1721] ConfLab: A Data Collection Concept, Dataset, and Benchmark for Machine Analysis of Free-Standing Social Interactions in the Wild

        **Authors**: *Chirag Raman, José Vargas Quiros, Stephanie Tan, Ashraful Islam, Ekin Gedik, Hayley Hung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/95f9ad2e251e9014697589037450f9bb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/95f9ad2e251e9014697589037450f9bb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recording the dynamics of unscripted human interactions in the wild is challenging due to the delicate trade-offs between several factors: participant privacy, ecological validity, data fidelity, and logistical overheads. To address these, following a 'datasets for the community by the community' ethos, we propose the Conference Living Lab (ConfLab): a new concept for multimodal multisensor data collection of in-the-wild free-standing social conversations. For the first instantiation of ConfLab described here, we organized a real-life professional networking event at a major international conference. Involving 48 conference attendees, the dataset captures a diverse mix of status, acquaintance, and networking motivations. Our capture setup improves upon the data fidelity of prior in-the-wild datasets while retaining privacy sensitivity: 8 videos (1920x1080, 60 fps) from a non-invasive overhead view, and custom wearable sensors with onboard recording of body motion (full 9-axis IMU), privacy-preserving low-frequency audio (1250 Hz), and Bluetooth-based proximity. Additionally, we developed custom solutions for distributed hardware synchronization at acquisition, and time-efficient continuous annotation of body keypoints and actions at high sampling rates. Our benchmarks showcase some of the open research tasks related to in-the-wild privacy-preserving social data analysis: keypoints detection from overhead camera views, skeleton-based no-audio speaker detection, and F-formation detection.

        ----

        ## [1722] Flamingo: a Visual Language Model for Few-Shot Learning

        **Authors**: *Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, Andy Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, Karén Simonyan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/960a172bc7fbf0177ccccbb411a7d800-Abstract-Conference.html)

        **Abstract**:

        Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. We propose key architectural innovations to: (i) bridge powerful pretrained vision-only and language-only models, (ii) handle sequences of arbitrarily interleaved visual and textual data, and (iii) seamlessly ingest images or videos as inputs. Thanks to their flexibility, Flamingo models can be trained on large-scale multimodal web corpora containing arbitrarily interleaved text and images, which is key to endow them with in-context few-shot learning capabilities. We perform a thorough evaluation of our models, exploring and measuring their ability to rapidly adapt to a variety of image and video tasks. These include open-ended tasks such as visual question-answering, where the model is prompted with a question which it has to answer, captioning tasks, which evaluate the ability to describe a scene or an event, and close-ended tasks such as multiple-choice visual question-answering. For tasks lying anywhere on this spectrum, a single Flamingo model can achieve a new state of the art with few-shot learning, simply by prompting the model with task-specific examples. On numerous benchmarks, Flamingo outperforms models fine-tuned on thousands of times more task-specific data.

        ----

        ## [1723] Understanding the Eluder Dimension

        **Authors**: *Gene Li, Pritish Kamath, Dylan J. Foster, Nati Srebro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/960cfbb846aff424ac20aadce6fa6530-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/960cfbb846aff424ac20aadce6fa6530-Abstract-Conference.html)

        **Abstract**:

        We provide new insights on eluder dimension, a complexity measure that has been extensively used to bound the regret of algorithms for online bandits and reinforcement learning with function approximation. First, we study the relationship between the eluder dimension for a function class and a generalized notion of \emph{rank}, defined for any monotone ``activation'' $\sigma : \mathbb{R}\to \mathbb{R}$, which corresponds to the minimal dimension required to represent the class as a generalized linear model. It is known that when $\sigma$ has derivatives bounded away from $0$, $\sigma$-rank gives rise to an upper bound on eluder dimension for any function class; we show however that eluder dimension can be exponentially smaller than $\sigma$-rank. We also show that the condition on the derivative is necessary; namely, when $\sigma$ is the $\mathsf{relu}$ activation, the eluder dimension can be exponentially larger than $\sigma$-rank. For Boolean-valued function classes, we obtain a characterization of the eluder dimension in terms of star number and threshold dimension, quantities which are relevant in active learning and online learning respectively.

        ----

        ## [1724] Predictive Querying for Autoregressive Neural Sequence Models

        **Authors**: *Alex Boyd, Samuel Showalter, Stephan Mandt, Padhraic Smyth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9622163c87b67fd5a4a0ec3247cf356e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9622163c87b67fd5a4a0ec3247cf356e-Abstract-Conference.html)

        **Abstract**:

        In reasoning about sequential events it is natural to pose probabilistic queries such as “when will event A occur next” or “what is the probability of A occurring before B”, with applications in areas such as user modeling, language models, medicine, and finance. These types of queries are complex to answer compared to next-event prediction, particularly for neural autoregressive models such as recurrent neural networks and transformers. This is in part due to the fact that future querying involves marginalization over large path spaces, which is not straightforward to do efficiently in such  models. In this paper we introduce a general typology for predictive queries in neural autoregressive sequence models and show that such queries can be systematically represented by sets of elementary building blocks. We leverage this typology to develop new query estimation methods based on beam search, importance sampling, and hybrids. Across four large-scale sequence datasets from different application domains, as well as for the GPT-2 language model, we demonstrate the ability to make query answering tractable for arbitrary queries in exponentially-large predictive path-spaces, and find clear differences in cost-accuracy tradeoffs between search and sampling methods.

        ----

        ## [1725] Learning State-Aware Visual Representations from Audible Interactions

        **Authors**: *Himangi Mittal, Pedro Morgado, Unnat Jain, Abhinav Gupta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9647157086adf5aa2c0217fb7f82bb19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9647157086adf5aa2c0217fb7f82bb19-Abstract-Conference.html)

        **Abstract**:

        We propose a self-supervised algorithm to learn representations from egocentric video data. Recently, significant efforts have been made to capture humans interacting with their own environments as they go about their daily activities. In result, several large egocentric datasets of interaction-rich multi-modal data have emerged. However, learning representations from videos can be challenging. First, given the uncurated nature of long-form continuous videos, learning effective representations require focusing on moments in time when interactions take place. Second, visual representations of daily activities should be sensitive to changes in the state of the environment. However, current successful multi-modal learning frameworks encourage representation invariance over time. To address these challenges, we leverage audio signals to identify moments of likely interactions which are conducive to better learning. We also propose a novel self-supervised objective that learns from audible state changes caused by interactions. We validate these contributions extensively on two large-scale egocentric datasets, EPIC-Kitchens-100 and the recently released Ego4D, and show improvements on several downstream tasks, including action recognition, long-term action anticipation, and object state change classification.

        ----

        ## [1726] Context-Based Dynamic Pricing with Partially Linear Demand Model

        **Authors**: *Jinzhi Bu, David Simchi-Levi, Chonghuan Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/964892fb1437e73ef14f305df9bf5e7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/964892fb1437e73ef14f305df9bf5e7b-Abstract-Conference.html)

        **Abstract**:

        In todayâ€™s data-rich environment, context-based dynamic pricing has gained much attention. To model the demand as a function of price and context, the existing literature either adopts a parametric  model or a non-parametric model. The former is easier to implement but may suffer from model mis-specification, whereas the latter is more robust but does not leverage many structural properties of the underlying problem. This paper combines these two approaches by studying the context-based dynamic pricing with online learning, where the unknown expected demand admits a semi-parametric partially linear structure. Specifically, we consider two demand models, whose expected demand at price $p$ and context $x \in \mathbb{R}^d$ is given by $bp+g(x)$ and $ f(p)+ a^\top x$ respectively. We assume that $g(x)$ is $\beta$-H{\"o}lder continuous in the first model, and $f(p)$ is $k$th-order smooth with an additional parameter $\delta$ in the second model. For both models, we design an efficient online learning algorithm with provable regret upper bounds, and establish matching lower bounds. This enables us to characterize the statistical complexity for the two learning models, whose optimal regret rates are $\widetilde \Theta(\sqrt T \vee T^{\frac{d}{d+2\beta}})$ and $\widetilde \Theta(\sqrt T \vee (\delta T^{k+1})^{\frac{1}{2k+1}})$ respectively. The numerical results demonstrate that our learning algorithms are more effective than benchmark algorithms, and also reveal the effects of parameters $d$, $\beta$ and $\delta$ on the algorithm's empirical regret, which are consistent with our theoretical findings.

        ----

        ## [1727] NeuroSchedule: A Novel Effective GNN-based Scheduling Method for High-level Synthesis

        **Authors**: *Jun Zeng, Mingyang Kou, Hailong Yao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/964b1c8dd5667fd647c09c8772829fd1-Abstract-Conference.html)

        **Abstract**:

        High-level synthesis (HLS) is widely used for transferring behavior-level specifications into circuit-level implementations. As a critical step in HLS, scheduling arranges the execution order of operations for enhanced performance. However, existing scheduling methods suffer from either exponential runtime or poor quality of solutions. This paper proposes an efficient and effective GNN-based scheduling method called NeuroSchedule, with both fast runtime and enhanced solution quality. Major features are as follows: (1) The learning problem for HLS scheduling is formulated for the first time, and a new machine learning framework is proposed. (2) Pre-training models are adopted to further enhance the scalability for various scheduling problems with different settings. Experimental results show that NeuroSchedule obtains near-optimal solutions while achieving more than 50,000x improvement in runtime compared with the ILP-based scheduling method. At the same time, NeuroSchedule improves the scheduling results by 6.10% on average compared with state-of-the-art entropy-directed method. To the best of our knowledge, this is the first GNN-based scheduling method for HLS.

        ----

        ## [1728] Hand-Object Interaction Image Generation

        **Authors**: *Hezhen Hu, Weilun Wang, Wengang Zhou, Houqiang Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96810b6d4752abe7bfb91f234c51e9e6-Abstract-Conference.html)

        **Abstract**:

        In this work, we are dedicated to a new task, i.e., hand-object interaction image generation, which aims to conditionally generate the hand-object image under the given hand, object and their interaction status. This task is challenging and research-worthy in many potential application scenarios, such as AR/VR games and online shopping, etc. To address this problem, we propose a novel HOGAN framework, which utilizes the expressive model-aware hand-object representation and leverages its inherent topology to build the unified surface space. In this space, we explicitly consider the complex self- and mutual occlusion during interaction. During final image synthesis, we consider different characteristics of hand and object and generate the target image in a split-and-combine manner. For evaluation, we build a comprehensive protocol to access both the fidelity and structure preservation of the generated image. Extensive experiments on two large-scale datasets, i.e., HO3Dv3 and DexYCB, demonstrate the effectiveness and superiority of our framework both quantitatively and qualitatively. The code will be available at https://github.com/play-with-HOI-generation/HOIG.

        ----

        ## [1729] DISCO: Adversarial Defense with Local Implicit Functions

        **Authors**: *Chih-Hui Ho, Nuno Vasconcelos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96930636e3fb63935e2af153d1cc40a3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96930636e3fb63935e2af153d1cc40a3-Abstract-Conference.html)

        **Abstract**:

        The problem of adversarial defenses for image classification, where the goal is to robustify a classifier against adversarial examples, is considered. Inspired by the hypothesis that these examples lie beyond the natural image manifold, a novel aDversarIal defenSe with local impliCit functiOns (DISCO) is proposed to remove adversarial perturbations by localized manifold projections. DISCO consumes an adversarial image and a query pixel location and outputs a clean RGB value at the location. It is implemented with an encoder and a local implicit module, where the former produces per-pixel deep features and the latter uses the features in the neighborhood of query pixel for predicting the clean RGB value. Extensive experiments demonstrate that both DISCO and its cascade version outperform prior defenses, regardless of whether the defense is known to the attacker. DISCO is also shown to be data and parameter efficient and to mount defenses that transfers across datasets, classifiers and attacks.

        ----

        ## [1730] Distinguishing discrete and continuous behavioral variability using warped autoregressive HMMs

        **Authors**: *Julia Costacurta, Lea Duncker, Blue Sheffer, Winthrop Gillis, Caleb Weinreb, Jeffrey E. Markowitz, Sandeep R. Datta, Alex H. Williams, Scott W. Linderman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96b3aa81a9e593ca5e9b184756034a43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96b3aa81a9e593ca5e9b184756034a43-Abstract-Conference.html)

        **Abstract**:

        A core goal in systems neuroscience and neuroethology is to understand how neural circuits generate naturalistic behavior. One foundational idea is that complex naturalistic behavior may be composed of sequences of stereotyped behavioral syllables, which combine to generate rich sequences of actions. To investigate this, a common approach is to use autoregressive hidden Markov models (ARHMMs) to segment video into discrete behavioral syllables. While these approaches have been successful in extracting syllables that are interpretable, they fail to account for other forms of behavioral variability, such as differences in speed, which may be better described as continuous in nature. To overcome these limitations, we introduce a class of warped ARHMMs (WARHMM). As is the case in the ARHMM, behavior is modeled as a mixture of autoregressive dynamics. However, the dynamics under each discrete latent state (i.e. each behavioral syllable) are additionally modulated by a continuous latent ``warping variable.'' We present two versions of warped ARHMM in which the warping variable affects the dynamics of each syllable either linearly or nonlinearly. Using depth-camera recordings of freely moving mice, we demonstrate that the failure of ARHMMs to account for continuous behavioral variability results in duplicate cluster assignments. WARHMM achieves similar performance to the standard ARHMM while using fewer behavioral syllables. Further analysis of behavioral measurements in mice demonstrates that WARHMM identifies structure relating to response vigor.

        ----

        ## [1731] RORL: Robust Offline Reinforcement Learning via Conservative Smoothing

        **Authors**: *Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96bbdd0ed2a9e7cd2fb7caf2fae15f3d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96bbdd0ed2a9e7cd2fb7caf2fae15f3d-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.

        ----

        ## [1732] Optimal Scaling for Locally Balanced Proposals in Discrete Spaces

        **Authors**: *Haoran Sun, Hanjun Dai, Dale Schuurmans*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96c6f409a374b5c81d2efa4bc5526f27-Abstract-Conference.html)

        **Abstract**:

        Optimal scaling has been well studied for Metropolis-Hastings (M-H) algorithms in continuous spaces, but a similar understanding has been lacking in discrete spaces.Recently, a family of locally balanced proposals (LBP) for discrete spaces has been proved to be asymptotically optimal, but the question of optimal scaling has remained open.In this paper, we establish, for the first time, that the efficiency of M-H in discrete spaces can also be characterized by an asymptotic acceptance rate that is independent of the target distribution. Moreover, we verify, both theoretically and empirically, that the optimal acceptance rates for LBP and random walk Metropolis (RWM) are $0.574$ and $0.234$ respectively. These results also help establish that LBP is asymptotically $O(N^\frac{2}{3})$ more efficient than RWM with respect to model dimension $N$. Knowledge of the optimal acceptance rate allows one to automatically tune the neighborhood size of a proposal distribution in a discrete space, directly analogous to step-size control in continuous spaces.We demonstrate empirically that such adaptive M-H sampling can robustly improve sampling in a variety of target distributions in discrete spaces, including training deep energy based models.

        ----

        ## [1733] The Impact of Task Underspecification in Evaluating Deep Reinforcement Learning

        **Authors**: *Vindula Jayawardana, Catherine Tang, Sirui Li, Dajiang Suo, Cathy Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96ca792fddef7c1e3366c405022463cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96ca792fddef7c1e3366c405022463cb-Abstract-Conference.html)

        **Abstract**:

        Evaluations of Deep Reinforcement Learning (DRL) methods are an integral part of scientific progress of the field. Beyond designing DRL methods for general intelligence, designing task-specific methods is becoming increasingly prominent for real-world applications. In these settings, the standard evaluation practice involves using a few instances of Markov Decision Processes (MDPs) to represent the task. However, many tasks induce a large family of MDPs owing to variations in the underlying environment, particularly in real-world contexts. For example, in traffic signal control, variations may stem from intersection geometries and traffic flow levels. The select MDP instances may thus inadvertently cause overfitting, lacking the statistical power to draw conclusions about the method's true performance across the family. In this article, we augment DRL evaluations to consider parameterized families of MDPs. We show that in comparison to evaluating DRL methods on select MDP instances, evaluating the MDP family often yields a substantially different relative ranking of methods, casting doubt on what methods should be considered state-of-the-art. We validate this phenomenon in standard control benchmarks and the real-world application of traffic signal control. At the same time, we show that accurately evaluating on an MDP family is nontrivial. Overall, this work identifies new challenges for empirical rigor in reinforcement learning, especially as the outcomes of DRL trickle into downstream decision-making.

        ----

        ## [1734] Zero-Shot 3D Drug Design by Sketching and Generating

        **Authors**: *Siyu Long, Yi Zhou, Xinyu Dai, Hao Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/96ddbf813f042e8ff891b4d6f7149bb6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/96ddbf813f042e8ff891b4d6f7149bb6-Abstract-Conference.html)

        **Abstract**:

        Drug design is a crucial step in the drug discovery cycle. Recently, various deep learning-based methods design drugs by generating novel molecules from scratch, avoiding traversing large-scale drug libraries. However, they depend on scarce experimental data or time-consuming docking simulation, leading to overfitting issues with limited training data and slow generation speed. In this study, we propose the zero-shot drug design method DESERT (Drug dEsign by SkEtching and geneRaTing). Specifically, DESERT splits the design process into two stages: sketching and generating, and bridges them with the molecular shape. The two-stage fashion enables our method to utilize the large-scale molecular database to reduce the need for experimental data and docking simulation. Experiments show that DESERT achieves a new state-of-the-art at a fast speed.

        ----

        ## [1735] Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning

        **Authors**: *Xiang Chen, Lei Li, Ningyu Zhang, Xiaozhuan Liang, Shumin Deng, Chuanqi Tan, Fei Huang, Luo Si, Huajun Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/97011c648eda678424f9292dadeae72e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/97011c648eda678424f9292dadeae72e-Abstract-Conference.html)

        **Abstract**:

        Prompt learning approaches have made waves in natural language processing by inducing better few-shot performance while they still follow a parametric-based learning paradigm; the oblivion and rote memorization problems in learning may encounter unstable generalization issues. Specifically, vanilla prompt learning may struggle to utilize atypical instances by rote during fully-supervised training or overfit shallow patterns with low-shot data. To alleviate such limitations, we develop RetroPrompt with the motivation of decoupling knowledge from memorization to help the model strike a balance between generalization and memorization. In contrast with vanilla prompt learning, RetroPrompt constructs an open-book knowledge-store from training instances and implements a retrieval mechanism during the process of input, training and inference, thus equipping the model with the ability to retrieve related contexts from the training corpus as cues for enhancement. Extensive experiments demonstrate that RetroPrompt can obtain better performance in both few-shot and zero-shot settings. Besides, we further illustrate that our proposed RetroPrompt can yield better generalization abilities with new datasets. Detailed analysis of memorization indeed reveals RetroPrompt can reduce the reliance of language models on memorization; thus, improving generalization for downstream tasks. Code is available in https://github.com/zjunlp/PromptKG/tree/main/research/RetroPrompt.

        ----

        ## [1736] Polynomial time guarantees for the Burer-Monteiro method

        **Authors**: *Diego Cifuentes, Ankur Moitra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9708c7d3a0fef3710f33ba05a74e10b3-Abstract-Conference.html)

        **Abstract**:

        The Burer-Monteiro method is one of the most widely used techniques for solving large-scale semidefinite programs (SDP). The basic idea is to solve a nonconvex program in $Y$, where $Y$ is an $n \times p$ matrix such that $X = Y Y^T$. We show that this method can solve SDPs in polynomial time in a smoothed analysis setting. More precisely, we consider an SDP whose domain satisfies some compactness and smoothness assumptions, and slightly perturb the cost matrix and the constraints. We show that if $p \gtrsim \sqrt{2(1{+}\eta)m}$, where $m$ is the number of constraints and $\eta>0$ is any fixed constant, then the Burer-Monteiro method can solve SDPs to any desired accuracy in polynomial time, in the setting of smooth analysis. The bound on $p$ approaches the celebrated Barvinok-Pataki bound in the limit as $\eta$ goes to zero, beneath which it the nonconvex program can be suboptimal. Our main technical contribution, which is key for our tight bound on $p$, is to connect spurious approximately critical points of the nonconvex program to tubular neighborhoods of certain algebraic varieties, and then estimate the volume of such tubes.

        ----

        ## [1737] Optimal Comparator Adaptive Online Learning with Switching Cost

        **Authors**: *Zhiyu Zhang, Ashok Cutkosky, Yannis Paschalidis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/972cd27c994a806e187ef1c2f5254059-Abstract-Conference.html)

        **Abstract**:

        Practical online learning tasks are often naturally defined on unconstrained domains, where optimal algorithms for general convex losses are characterized by the notion of comparator adaptivity. In this paper, we design such algorithms in the presence of switching cost - the latter penalizes the typical optimism in adaptive algorithms, leading to a delicate design trade-off. Based on a novel dual space scaling strategy discovered by a continuous-time analysis, we propose a simple algorithm that improves the existing comparator adaptive regret bound [ZCP22a] to the optimal rate. The obtained benefits are further extended to the expert setting, and the practicality of the proposed algorithm is demonstrated through a sequential investment task.

        ----

        ## [1738] A Robust Phased Elimination Algorithm for Corruption-Tolerant Gaussian Process Bandits

        **Authors**: *Ilija Bogunovic, Zihan Li, Andreas Krause, Jonathan Scarlett*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9739fdfbecb84b2cab3ba06f3ee5498b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9739fdfbecb84b2cab3ba06f3ee5498b-Abstract-Conference.html)

        **Abstract**:

        We consider the sequential optimization of an unknown, continuous, and expensive to evaluate reward function, from noisy and adversarially corrupted observed rewards. When the corruption attacks are subject to a suitable budget $C$ and the function lives in a Reproducing Kernel Hilbert Space (RKHS), the problem can be posed as {\em corrupted Gaussian process (GP) bandit optimization}. We propose a novel robust elimination-type algorithm that runs in epochs, combines exploration with infrequent switching to select a small subset of actions, and plays each action for multiple time instants. Our algorithm, {\em Robust GP Phased Elimination (RGP-PE)}, successfully balances robustness to corruptions with exploration and exploitation such that its performance degrades minimally in the presence (or absence) of adversarial corruptions. When $T$ is the number of samples and $\gamma_T$ is the maximal information gain, the corruption-dependent term in our regret bound is $O(C \gamma_T^{3/2})$, which is significantly tighter than the existing $O(C \sqrt{T \gamma_T})$ for several commonly-considered kernels. We perform the first empirical study of robustness in the corrupted GP bandit setting, and show that our algorithm is robust against a variety of adversarial attacks.

        ----

        ## [1739] Fair Rank Aggregation

        **Authors**: *Diptarka Chakraborty, Syamantak Das, Arindam Khan, Aditya Subramanian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/974309ef51ebd89034adc64a57e304f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/974309ef51ebd89034adc64a57e304f2-Abstract-Conference.html)

        **Abstract**:

        Ranking algorithms find extensive usage in diverse areas such as web search, employment, college    admission, voting, etc.  The related rank aggregation problem deals with combining multiple    rankings into a single aggregate ranking.  However, algorithms for both these problems might be    biased against some individuals or groups due to implicit prejudice or marginalization in the    historical data.  We study ranking and rank aggregation problems from a fairness or diversity    perspective, where the candidates (to be ranked) may belong to different groups and each group    should have a fair representation in the final ranking. We allow the designer to set the    parameters that define fair representation. These parameters specify the allowed range of the    number of candidates from a particular group in the top-$k$ positions of the ranking.  Given any    ranking, we provide a fast and exact algorithm for finding the closest fair ranking for the    Kendall tau metric under {\em strong fairness}, i.e., when the final ranking is fair for all    values of $k$. We also provide an exact algorithm for finding the closest fair ranking for the    Ulam metric under strong fairness when there are only $O(1)$ number of groups.  Our    algorithms are simple, fast, and might be extendable to other relevant metrics. We also give a    novel  meta-algorithm for the general rank aggregation problem under the fairness framework.    Surprisingly, this meta-algorithm works for any generalized mean objective (including center and    median problems) and any fairness criteria. As a byproduct, we obtain 3-approximation algorithms    for both center and median problems, under both Kendall tau and Ulam metrics. Furthermore, using    sophisticated techniques we obtain a $(3-\varepsilon)$-approximation algorithm, for a constant    $\varepsilon>0$,  for the Ulam metric under strong fairness.

        ----

        ## [1740] Beyond IID: data-driven decision-making in heterogeneous environments

        **Authors**: *Omar Besbes, Will Ma, Omar Mouchtaki*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/974ff7b5bf08dbf9400b5d599a39c77f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/974ff7b5bf08dbf9400b5d599a39c77f-Abstract-Conference.html)

        **Abstract**:

        In this work, we study data-driven decision-making and depart from the classical identically and independently distributed (i.i.d.) assumption.  We present a new framework in which  historical samples   are generated from unknown and different distributions, which we dub  \textit{heterogeneous environments}.  These distributions are assumed to lie in a heterogeneity ball with known radius and centered around the (also) unknown future (out-of-sample) distribution on which the performance of a decision will be evaluated. We quantify the asymptotic worst-case regret that is achievable by central data-driven policies such as Sample Average Approximation, but also by rate-optimal ones,   as a function of the radius of the heterogeneity ball. Our work shows that the type of achievable performance varies considerably across different combinations of problem classes and notions of heterogeneity. We demonstrate the versatility of our framework by comparing achievable guarantees for the heterogeneous version of widely studied  data-driven problems such as  pricing, ski-rental, and newsvendor. En route, we establish a new connection between data-driven decision-making and distributionally robust optimization.

        ----

        ## [1741] Neur2SP: Neural Two-Stage Stochastic Programming

        **Authors**: *Rahul Patel, Justin Dumouchelle, Elias B. Khalil, Merve Bodur*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9793671e4be9858a69a32545204d59d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9793671e4be9858a69a32545204d59d1-Abstract-Conference.html)

        **Abstract**:

        Stochastic Programming is a powerful modeling framework for decision-making under uncertainty. In this work, we tackle two-stage stochastic programs (2SPs), the most widely used class of stochastic programming models. Solving 2SPs exactly requires optimizing over an expected value function that is computationally intractable. Having a mixed-integer linear program (MIP) or a nonlinear program (NLP) in the second stage further aggravates the intractability, even when specialized algorithms that exploit problem structure are employed.Finding high-quality (first-stage) solutions -- without leveraging problem structure -- can be crucial in such settings. We develop Neur2SP, a new method that approximates the expected value function via a neural network to obtain a surrogate model that can be solved more efficiently than the traditional extensive formulation approach. Neur2SP makes no assumptions about the problem structure, in particular about the second-stage problem, and can be implemented using an off-the-shelf MIP solver. Our extensive computational experiments on four benchmark 2SP problem classes with different structures (containing MIP and NLP second-stage problems) demonstrate the efficiency (time) and efficacy (solution quality) of Neur2SP. In under 1.66 seconds, Neur2SP finds high-quality solutions across all problems even as the number of scenarios increases, an ideal property that is difficult to have for traditional 2SP solution techniques. Namely, the most generic baseline method typically requires minutes to hours to find solutions of comparable quality.

        ----

        ## [1742] SQ Lower Bounds for Learning Single Neurons with Massart Noise

        **Authors**: *Ilias Diakonikolas, Daniel Kane, Lisheng Ren, Yuxin Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/97b983c974551153d20ddfabb62a5203-Abstract-Conference.html)

        **Abstract**:

        We study the problem of PAC learning a single neuron in the presence of Massart noise. Specifically, for a known activation function $f: \mathbb{R}\to \mathbb{R}$, the learner is given access to labeled examples $(\mathbf{x}, y) \in \mathbb{R}^d \times \mathbb{R}$, where the marginal distribution of $\mathbf{x}$ is arbitrary and the corresponding label $y$ is a Massart corruption of $f(\langle \mathbf{w}, \mathbf{x} \rangle)$. The goal of the learner is to output a hypothesis $h: \mathbb{R}^d \to \mathbb{R}$ with small squared loss. For a range of activation functions, including ReLUs, we establish super-polynomial Statistical Query (SQ) lower bounds for this learning problem. In more detail, we prove that no efficient SQ algorithm can approximate the optimal error within any constant factor. Our main technical contribution is a novel SQ-hard construction for learning $\{ \pm 1\}$-weight Massart halfspaces on the Boolean hypercube that is interesting on its own right.

        ----

        ## [1743] MaskPlace: Fast Chip Placement via Reinforced Visual Representation Learning

        **Authors**: *Yao Lai, Yao Mu, Ping Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/97c8a8eb0e5231d107d0da51b79e09cb-Abstract-Conference.html)

        **Abstract**:

        Placement is an essential task in modern chip design, aiming at placing millions of circuit modules on a 2D chip canvas. Unlike the human-centric solution, which requires months of intense effort by hardware engineers to produce a layout to minimize delay and energy consumption, deep reinforcement learning has become an emerging autonomous tool. However, the learning-centric method is still in its early stage, impeded by a massive design space of size ten to the order of a few thousand. This work presents MaskPlace to automatically generate a valid chip layout design within a few hours, whose performance can be superior or comparable to recent advanced approaches. It has several appealing benefits that prior arts do not have. Firstly, MaskPlace recasts placement as a problem of learning pixel-level visual representation to comprehensively describe millions of modules on a chip,  enabling placement in a high-resolution canvas and a large action space. It outperforms recent methods that represent a chip as a hypergraph. Secondly, it enables training the policy network by an intuitive reward function with dense reward, rather than a complicated reward function with sparse reward from previous methods. Thirdly, extensive experiments on many public benchmarks show that MaskPlace outperforms existing RL approaches in all key performance metrics, including wirelength, congestion, and density. For example, it achieves 60%-90% wirelength reduction and guarantees zero overlaps. We believe MaskPlace can improve AI-assisted chip layout design. The deliverables are released at https://laiyao1.github.io/maskplace.

        ----

        ## [1744] List-Decodable Sparse Mean Estimation

        **Authors**: *Shiwei Zeng, Jie Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/97d596ca21d0751ba2c633bad696cf7f-Abstract-Conference.html)

        **Abstract**:

        Robust mean estimation is one of the most important problems in statistics: given a set of samples in $\mathbb{R}^d$ where an $\alpha$ fraction are drawn from some distribution $D$ and the rest are adversarially corrupted, we aim to estimate the mean of $D$. A surge of recent research interest has been focusing on the list-decodable setting where $\alpha \in (0, \frac12]$, and the goal is to output a finite number of estimates among which at least one approximates the target mean. In this paper, we consider that the underlying distribution $D$ is Gaussian with $k$-sparse mean. Our main contribution is the first polynomial-time algorithm that enjoys sample complexity $O\big(\mathrm{poly}(k, \log d)\big)$, i.e. poly-logarithmic in the dimension. One of our core algorithmic ingredients is using low-degree {\em sparse polynomials} to filter outliers, which may find more applications.

        ----

        ## [1745] Structure-Aware Image Segmentation with Homotopy Warping

        **Authors**: *Xiaoling Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98143953a7fd1319175b491888fc8df5-Abstract-Conference.html)

        **Abstract**:

        Besides per-pixel accuracy, topological correctness is also crucial for the segmentation of images with fine-scale structures, e.g., satellite images and biomedical images. In this paper, by leveraging the theory of digital topology, we identify pixels in an image that are critical for topology. By focusing on these critical pixels, we propose a new \textbf{homotopy warping loss} to train deep image segmentation networks for better topological accuracy. To efficiently identify these topologically critical pixels, we propose a new algorithm exploiting the distance transform. The proposed algorithm, as well as the loss function, naturally generalize to different topological structures in both 2D and 3D settings. The proposed loss function helps deep nets achieve better performance in terms of topology-aware metrics, outperforming state-of-the-art structure/topology-aware segmentation methods.

        ----

        ## [1746] Positive-Unlabeled Learning using Random Forests via Recursive Greedy Risk Minimization

        **Authors**: *Jonathan Wilton, Abigail M. Y. Koay, Ryan K. L. Ko, Miao Xu, Nan Ye*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98257285340854262185500e59bc0f28-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98257285340854262185500e59bc0f28-Abstract-Conference.html)

        **Abstract**:

        The need to learn from positive and unlabeled data, or PU learning, arises in many applications and has attracted increasing interest. While random forests are known to perform well on many tasks with positive and negative data, recent PU algorithms are generally based on deep neural networks, and the potential of tree-based PU learning is under-explored. In this paper, we propose new random forest algorithms for PU-learning. Key to our approach is a new interpretation of decision tree algorithms for positive and negative data as \emph{recursive greedy risk minimization algorithms}. We extend this perspective to the PU setting to develop new decision tree learning algorithms that directly minimizes PU-data based estimators for the expected risk. This allows us to develop an efficient PU random forest algorithm, PU extra trees. Our approach features three desirable properties: it is robust to the choice of the loss function in the sense that various loss functions lead to the same decision trees; it requires little hyperparameter tuning as compared to neural network based PU learning; it supports a feature importance that directly measures a feature's contribution to risk minimization. Our algorithms demonstrate strong performance on several datasets. Our code is available at \url{https://github.com/puetpaper/PUExtraTrees}.

        ----

        ## [1747] Extracting computational mechanisms from neural data using low-rank RNNs

        **Authors**: *Adrian Valente, Jonathan W. Pillow, Srdjan Ostojic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9877d915a4b4f00e85e7b4cfdf41e450-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9877d915a4b4f00e85e7b4cfdf41e450-Abstract-Conference.html)

        **Abstract**:

        An influential framework within systems neuroscience posits that neural computations can be understood in terms of low-dimensional dynamics in recurrent circuits. A number of methods have thus been developed to extract latent dynamical systems from neural recordings, but inferring models that are both predictive and interpretable remains a difficult challenge. Here we propose a new method called Low-rank Inference from Neural Trajectories (LINT), based on a class of low-rank recurrent neural networks (lrRNNs) for which a link between connectivity and dynamics has been previously demonstrated. By fitting such networks to trajectories of neural activity, LINT yields a mechanistic model of latent dynamics, as well as a set of axes for dimensionality reduction and verifiable predictions for inactivations of specific populations of neurons. Here, we first demonstrate the consistency of our method and apply it to two use cases: (i) we reverse-engineer "black-box" vanilla RNNs trained to perform cognitive tasks, and (ii) we infer latent dynamics and neural contributions from electrophysiological recordings of nonhuman primates performing a similar task.

        ----

        ## [1748] Pluralistic Image Completion with Gaussian Mixture Models

        **Authors**: *Xiaobo Xia, Wenhao Yang, Jie Ren, Yewen Li, Yibing Zhan, Bo Han, Tongliang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/987913de7a2963359196d4491d0fd4e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/987913de7a2963359196d4491d0fd4e7-Abstract-Conference.html)

        **Abstract**:

        Pluralistic image completion focuses on generating both visually realistic and diverse results for image completion. Prior methods enjoy the empirical successes of this task. However, their used constraints for pluralistic image completion are argued to be not well interpretable and unsatisfactory from two aspects. First, the constraints for visual reality can be weakly correlated to the objective of image completion or even redundant. Second, the constraints for diversity are designed to be task-agnostic, which causes the constraints to not work well. In this paper, to address the issues, we propose an end-to-end probabilistic method. Specifically, we introduce a unified probabilistic graph model that represents the complex interactions in image completion. The entire procedure of image completion is then mathematically divided into several sub-procedures, which helps efficient enforcement of constraints. The sub-procedure directly related to pluralistic results is identified, where the interaction is established by a Gaussian mixture model (GMM). The inherent parameters of GMM are task-related, which are optimized adaptively during training, while the number of its primitives can control the diversity of results conveniently. We formally establish the effectiveness of our method and demonstrate it with comprehensive experiments. The implementationis available at https://github.com/tmllab/PICMM.

        ----

        ## [1749] A Fast Post-Training Pruning Framework for Transformers

        **Authors**: *Woosuk Kwon, Sehoon Kim, Michael W. Mahoney, Joseph Hassoun, Kurt Keutzer, Amir Gholami*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/987bed997ab668f91c822a09bce3ea12-Abstract-Conference.html)

        **Abstract**:

        Pruning is an effective way to reduce the huge inference cost of Transformer models. However, prior work on pruning Transformers requires retraining the models. This can add high training cost and high complexity to model deployment, making it difficult to use in many practical situations. To address this, we propose a fast post-training pruning framework for Transformers that does not require any retraining. Given a resource constraint and a sample dataset, our framework automatically prunes the Transformer model using structured sparsity methods. To retain high accuracy without retraining, we introduce three novel techniques: (i) a lightweight mask search algorithm that finds which heads and filters to prune based on the Fisher information; (ii) mask rearrangement that complements the search algorithm; and (iii) mask tuning that reconstructs the output activations for each layer. We apply our method to BERT-base and DistilBERT, and we evaluate its effectiveness on GLUE and SQuAD benchmarks. Our framework achieves up to 2.0x reduction in FLOPs and 1.56x speedup in inference latency, while maintaining < 1% loss in accuracy. Importantly, our framework prunes Transformers in less than 3 minutes on a single GPU, which is over two orders of magnitude faster than existing pruning approaches that retrain the models.

        ----

        ## [1750] SurDis: A Surface Discontinuity Dataset for Wearable Technology to Assist Blind Navigation in Urban Environments

        **Authors**: *Kuan Yew Leong, Siew Mooi Lim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/988120df77d6c767995febd7ff616517-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/988120df77d6c767995febd7ff616517-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        According to World Health Organization, there is an estimated 2.2 billion people with a near or distance vision impairment worldwide. Difficulty in self-navigation is one of the greatest challenges to independence for the blind and low vision (BLV) people. Through consultations with several BLV service providers, we realized that negotiating surface discontinuities is one of the very prominent challenges when navigating an outdoor environment within the urban. Surface discontinuities are commonly formed by rises and drop-offs along a pathway. They could be a threat to balancing during a walk and perceiving such a threat is highly challenging to the BLVs. In this paper, we introduce SurDis, a novel dataset of depth maps and stereo images that exemplifies the issue of surface discontinuity in the urban areas of Klang Valley, Malaysia. We seek to address the limitation of existing datasets of such nature in these areas. Current mobility tools for the BLVs predominantly focus on furniture, indoor built environments, traffic signs, vehicles, humans and various types of objects' detection above the surface of a pathway. We emphasize a specific purpose for SurDis â€“ to support the development of assistive wearable technology for the BLVs to negotiate surface discontinuity. We consulted BLV volunteers on the specifications of surface condition that could become hazardous for navigation using 3D printed replicas of actual scaled-down scenes, and identified locations that are frequented by the BLVs as our target data collection fields. With feedback from these volunteers, we developed a lightweight, small and unobtrusive prototype equipped with a tiny stereo camera and an embedded system on a single board computer to capture the samples from 10 different locations. We describe instrument development, data collection, preprocessing, annotation, and experiments conducted. The dataset contains: (1) more than 17000 depth maps generated from 200 sets of stereo image sequences, (2) annotations of surface discontinuity in the depth maps, and (3) bitmap stereo image pairs corresponding to the depth maps in (1).

        ----

        ## [1751] Interventions, Where and How? Experimental Design for Causal Models at Scale

        **Authors**: *Panagiotis Tigas, Yashas Annadani, Andrew Jesson, Bernhard Schölkopf, Yarin Gal, Stefan Bauer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98a5c0470e57d518ade4e56c6ee0b363-Abstract-Conference.html)

        **Abstract**:

        Causal discovery from observational and interventional data is challenging due to limited data and non-identifiability which introduces uncertainties in estimating the underlying structural causal model (SCM). Incorporating these uncertainties and selecting optimal experiments (interventions) to perform can help to identify the true SCM faster. Existing methods in experimental design for causal discovery from limited data either rely on linear assumptions for the SCM or select only the intervention target. In this paper, we incorporate recent advances in Bayesian causal discovery into the Bayesian optimal experimental design framework, which allows for active causal discovery of nonlinear, large SCMs, while selecting both the target and the value to intervene with. We demonstrate the performance of the proposed method on synthetic graphs (Erdos-Rènyi, Scale Free) for both linear and nonlinear SCMs as well as on the \emph{in-silico} single-cell gene regulatory network dataset, DREAM.

        ----

        ## [1752] Dual-discriminative Graph Neural Network for Imbalanced Graph-level Anomaly Detection

        **Authors**: *Ge Zhang, Zhenyu Yang, Jia Wu, Jian Yang, Shan Xue, Hao Peng, Jianlin Su, Chuan Zhou, Quan Z. Sheng, Leman Akoglu, Charu C. Aggarwal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98a625423070cfc6ae3d82d4b59408a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98a625423070cfc6ae3d82d4b59408a0-Abstract-Conference.html)

        **Abstract**:

        Graph-level anomaly detection aims to distinguish anomalous graphs in a graph dataset from normal graphs. Anomalous graphs represent a very few but essential patterns in the real world. The anomalous property of a graph may be referable to its anomalous attributes of particular nodes and anomalous substructures that refer to a subset of nodes and edges in the graph. In addition, due to the imbalance nature of anomaly problem, anomalous information will be diluted by normal graphs with overwhelming quantities. Various anomaly notions in the attributes and/or substructures and the imbalance nature together make detecting anomalous graphs a non-trivial task. In this paper, we propose a graph neural network for graph-level anomaly detection, namely iGAD. Specifically, an anomalous graph attribute-aware graph convolution and an anomalous graph substructure-aware deep Random Walk Kernel (deep RWK) are welded into a graph neural network to achieve the dual-discriminative ability on anomalous attributes and substructures. Deep RWK in iGAD makes up for the deficiency of graph convolution in distinguishing structural information caused by the simple neighborhood aggregation mechanism. Further, we propose a Point Mutual Information (PMI)-based loss function to target the problems caused by imbalance distributions. PMI-based loss function enables iGAD to capture essential correlation between input graphs and their anomalous/normal properties. We evaluate iGAD on four real-world graph datasets. Extensive experiments demonstrate the superiority of iGAD on the graph-level anomaly detection task.

        ----

        ## [1753] Posted Pricing and Dynamic Prior-independent Mechanisms with Value Maximizers

        **Authors**: *Yuan Deng, Vahab Mirrokni, Hanrui Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98b2b307aa4aa323df2ba3a83460f25e-Abstract-Conference.html)

        **Abstract**:

        We study posted price auctions and dynamic prior-independent mechanisms for (ROI-constrained) value maximizers. In contrast to classic (quasi-linear) utility maximizers, these agents aim to maximize their total value subject to a minimum ratio of value per unit of payment made. When personalized posted prices are allowed, posted price auctions for value maximizers can be reduced to posted price auctions for utility maximizers. However, for anonymous posted prices, the well-known $\frac 1 2$ approximation for utility maximizers is impossible for value maximizers and we provide a posted price mechanism with $\frac12(1 - 1/e)$ approximation. Moreover, we demonstrate how to apply our results to design prior-independent mechanisms in a dynamic environment; and to the best of our knowledge, this gives the first constant revenue approximation with multiple value maximizers. Finally, we provide an extension to combinatorial auctions with submodular / XOS agents.

        ----

        ## [1754] Curious Exploration via Structured World Models Yields Zero-Shot Object Manipulation

        **Authors**: *Cansu Sancaktar, Sebastian Blaes, Georg Martius*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98ecdc722006c2959babbdbdeb22eb75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98ecdc722006c2959babbdbdeb22eb75-Abstract-Conference.html)

        **Abstract**:

        It has been a long-standing dream to design artificial agents that explore their environment efficiently via intrinsic motivation, similar to how children perform curious free play. Despite recent advances in intrinsically motivated reinforcement learning (RL), sample-efficient exploration in object manipulation scenarios remains a significant challenge as most of the relevant information lies in the sparse agent-object and object-object interactions. In this paper, we propose to use structured world models to incorporate relational inductive biases in the control loop to achieve sample-efficient and interaction-rich exploration in compositional multi-object environments. By planning for future novelty inside structured world models, our method generates free-play behavior that starts to interact with objects early on and develops more complex behavior over time. Instead of using models only to compute intrinsic rewards, as commonly done, our method showcases that the self-reinforcing cycle between good models and good exploration also opens up another avenue: zero-shot generalization to downstream tasks via model-based planning. After the entirely intrinsic task-agnostic exploration phase, our method solves challenging downstream tasks such as stacking, flipping, pick & place, and throwing that generalizes to unseen numbers and arrangements of objects without any additional training.

        ----

        ## [1755] Estimating Noise Transition Matrix with Label Correlations for Noisy Multi-Label Learning

        **Authors**: *Shikun Li, Xiaobo Xia, Hansong Zhang, Yibing Zhan, Shiming Ge, Tongliang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/98f8c89ae042c512e6c87e0e0c2a0f98-Abstract-Conference.html)

        **Abstract**:

        In label-noise learning, the noise transition matrix, bridging the class posterior for noisy and clean data, has been widely exploited to learn statistically consistent classifiers. The effectiveness of these algorithms relies heavily on estimating the transition matrix. Recently, the problem of label-noise learning in multi-label classification has received increasing attention, and these consistent algorithms can be applied in multi-label cases. However, the estimation of transition matrices in noisy multi-label learning has not been studied and remains challenging, since most of the existing estimators in noisy multi-class learning depend on the existence of anchor points and the accurate fitting of noisy class posterior. To address this problem, in this paper, we first study the identifiability problem of the class-dependent transition matrix in noisy multi-label learning, and then inspired by the identifiability results, we propose a new estimator by exploiting label correlations without neither anchor points nor accurate fitting of noisy class posterior. Specifically, we estimate the occurrence probability of two noisy labels to get noisy label correlations. Then, we perform sample selection to further extract information that implies clean label correlations, which is used to estimate the occurrence probability of one noisy label when a certain clean label appears. By utilizing the mismatch of label correlations implied in these occurrence probabilities, the transition matrix is identifiable, and can then be acquired by solving a simple bilinear decomposition problem. Empirical results demonstrate the effectiveness of our estimator to estimate the transition matrix with label correlations, leading to better classification performance. Source codes are available at https://github.com/tmllab/Multi-Label-T.

        ----

        ## [1756] Active Learning Polynomial Threshold Functions

        **Authors**: *Omri Ben-Eliezer, Max Hopkins, Chutong Yang, Hantao Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/99015a2974664cb9db56844d0f27b5a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/99015a2974664cb9db56844d0f27b5a9-Abstract-Conference.html)

        **Abstract**:

        We initiate the study of active learning polynomial threshold functions (PTFs). While traditional lower bounds imply that even univariate quadratics cannot be non-trivially actively learned, we show that allowing the learner basic access to the derivatives of the underlying classifier circumvents this issue and leads to a computationally efficient algorithm for active learning degree-$d$ univariate PTFs in $\tilde{O}(d^3\log(1/\varepsilon\delta))$ queries. We extend this result to the batch active setting, providing a smooth transition between query complexity and rounds of adaptivity, and also provide near-optimal algorithms for active learning PTFs in several average case settings. Finally, we prove that access to derivatives is insufficient for active learning multivariate PTFs, even those of just two variables.

        ----

        ## [1757] Single-phase deep learning in cortico-cortical networks

        **Authors**: *Will Greedy, Heng Wei Zhu, Joseph Pemberton, Jack Mellor, Rui Ponte Costa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/99088dffd5eab0babebcda4bc58bbcea-Abstract-Conference.html)

        **Abstract**:

        The error-backpropagation (backprop) algorithm remains the most common solution to the credit assignment problem in artificial neural networks. In neuroscience, it is unclear whether the brain could adopt a similar strategy to correctly modify its synapses. Recent models have attempted to bridge this gap while being consistent with a range of experimental observations. However, these models are either unable to effectively backpropagate error signals across multiple layers or require a multi-phase learning process, neither of which are reminiscent of learning in the brain. Here, we introduce a new model, Bursting Cortico-Cortical Networks (BurstCCN), which solves these issues by integrating known properties of cortical networks namely bursting activity, short-term plasticity (STP) and dendrite-targeting interneurons. BurstCCN relies on burst multiplexing via connection-type-specific STP to propagate backprop-like error signals within deep cortical networks. These error signals are encoded at distal dendrites and induce burst-dependent plasticity as a result of excitatory-inhibitory top-down inputs. First, we demonstrate that our model can effectively backpropagate errors through multiple layers using a single-phase learning process. Next, we show both empirically and analytically that learning in our model approximates backprop-derived gradients. Finally, we demonstrate that our model is capable of learning complex image classification tasks (MNIST and CIFAR-10). Overall, our results suggest that cortical features across sub-cellular, cellular, microcircuit and systems levels jointly underlie single-phase efficient deep learning in the brain.

        ----

        ## [1758] Domain Generalization by Learning and Removing Domain-specific Features

        **Authors**: *Yu Ding, Lei Wang, Bin Liang, Shuming Liang, Yang Wang, Fang Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9941833e8327910ef25daeb9005e4748-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9941833e8327910ef25daeb9005e4748-Abstract-Conference.html)

        **Abstract**:

        Deep Neural Networks (DNNs) suffer from domain shift when the test dataset follows a distribution different from the training dataset. Domain generalization aims to tackle this issue by learning a model that can generalize to unseen domains. In this paper, we propose a new approach that aims to explicitly remove domain-specific features for domain generalization. Following this approach, we propose a novel framework called Learning and Removing Domain-specific features for Generalization (LRDG) that learns a domain-invariant model by tactically removing domain-specific features from the input images. Specifically, we design a classifier to effectively learn the domain-specific features for each source domain, respectively. We then develop an encoder-decoder network to map each input image into a new image space where the learned domain-specific features are removed. With the images output by the encoder-decoder network, another classifier is designed to learn the domain-invariant features to conduct image classification. Extensive experiments demonstrate that our framework achieves superior performance compared with state-of-the-art methods.

        ----

        ## [1759] Torsional Diffusion for Molecular Conformer Generation

        **Authors**: *Bowen Jing, Gabriele Corso, Jeffrey Chang, Regina Barzilay, Tommi S. Jaakkola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/994545b2308bbbbc97e3e687ea9e464f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/994545b2308bbbbc97e3e687ea9e464f-Abstract-Conference.html)

        **Abstract**:

        Molecular conformer generation is a fundamental task in computational chemistry. Several machine learning approaches have been developed, but none have outperformed state-of-the-art cheminformatics methods. We propose torsional diffusion, a novel diffusion framework that operates on the space of torsion angles via a diffusion process on the hypertorus and an extrinsic-to-intrinsic score model. On a standard benchmark of drug-like molecules, torsional diffusion generates superior conformer ensembles compared to machine learning and cheminformatics methods in terms of both RMSD and chemical properties, and is orders of magnitude faster than previous diffusion-based models. Moreover, our model provides exact likelihoods, which we employ to build the first generalizable Boltzmann generator. Code is available at https://github.com/gcorso/torsional-diffusion.

        ----

        ## [1760] LiteTransformerSearch: Training-free Neural Architecture Search for Efficient Language Models

        **Authors**: *Mojan Javaheripi, Gustavo de Rosa, Subhabrata Mukherjee, Shital Shah, Tomasz Religa, Caio Cesar Teodoro Mendes, Sébastien Bubeck, Farinaz Koushanfar, Debadeepta Dey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9949e6906be6448230cdba9a4cb2d564-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9949e6906be6448230cdba9a4cb2d564-Abstract-Conference.html)

        **Abstract**:

        The Transformer architecture is ubiquitously used as the building block of largescale autoregressive language models. However, finding architectures with the optimal trade-off between task performance (perplexity) and hardware constraints like peak memory utilization and latency is non-trivial. This is exacerbated by the proliferation of various hardware. We leverage the somewhat surprising empirical observation that the number of decoder parameters in autoregressive Transformers has a high rank correlation with task performance, irrespective of the architecture topology. This observation organically induces a simple Neural Architecture Search (NAS) algorithm that uses decoder parameters as a proxy for perplexity without need for any model training. The search phase of our training-free algorithm, dubbed Lightweight Transformer Search (LTS), can be run directly on target devices since it does not require GPUs. Using on-target device measurements, LTS extracts the Pareto-frontier of perplexity versus any hardware performance cost. We evaluate LTS on diverse devices from ARM CPUs to NVIDIA GPUs and two popular autoregressive Transformer backbones: GPT-2 and Transformer-XL. Results show that the perplexity of 16-layer GPT-2 and Transformer-XL can be achieved with up to 1.5×, 2.5× faster runtime and 1.2×, 2.0× lower peak memory utilization. When evaluated in zero and one-shot settings, LTS Pareto-frontier models achieve higher average accuracy compared to the 350M parameter OPT across 14 tasks, with up to 1.6× lower latency. LTS extracts the Pareto-frontier in under 3 hours while running on a commodity laptop. We effectively remove the carbon footprint of hundreds of GPU hours of training during search, offering a strong simple baseline for future NAS methods in autoregressive language modeling.

        ----

        ## [1761] AgraSSt: Approximate Graph Stein Statistics for Interpretable Assessment of Implicit Graph Generators

        **Authors**: *Wenkai Xu, Gesine D. Reinert*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/996e2b446391fcb8bf32a3d1645cc799-Abstract-Conference.html)

        **Abstract**:

        We propose and analyse a novel statistical procedure, coined AgraSSt, to assess the quality of graph generators which may not be available in explicit forms. In particular, AgraSSt can be used to determine whether a learned graph generating process is capable of generating graphs which resemble a given input graph. Inspired by Stein operators for random graphs, the key idea of AgraSSt is the construction of a kernel discrepancy based on an operator obtained from the graph generator. AgraSSt can provide interpretable criticisms for a graph generator training procedure and help identify reliable sample batches for downstream tasks. We give theoretical guarantees for a broad class of random graph models. Moreover, we provide empirical results on both synthetic input graphs with known graph generation procedures, and real-world input graphs that the state-of-the-art (deep) generative models for graphs are trained on.

        ----

        ## [1762] On the Limitations of Stochastic Pre-processing Defenses

        **Authors**: *Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/997089469acbeb410405e43f0011be1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/997089469acbeb410405e43f0011be1f-Abstract-Conference.html)

        **Abstract**:

        Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.

        ----

        ## [1763] Iterative Scene Graph Generation

        **Authors**: *Siddhesh Khandelwal, Leonid Sigal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/99831104028c3b7e6079fd8bdcc42c8f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/99831104028c3b7e6079fd8bdcc42c8f-Abstract-Conference.html)

        **Abstract**:

        The task of scene graph generation entails identifying object entities and their corresponding interaction predicates in a given image (or video). Due to the combinatorially large solution space, existing approaches to scene graph generation assume certain factorization of the joint distribution to make the estimation feasible (e.g., assuming that objects are conditionally independent of predicate predictions). However, this fixed factorization is not ideal under all scenarios (e.g., for images where an object entailed in interaction is small and not discernible on its own). In this work, we propose a novel framework for scene graph generation that addresses this limitation, as well as introduces dynamic conditioning on the image, using message passing in a Markov Random Field. This is implemented as an iterative refinement procedure wherein each modification is conditioned on the graph generated in the previous iteration. This conditioning across refinement steps allows joint reasoning over entities and relations. This framework is realized via a novel and end-to-end trainable transformer-based architecture. In addition, the proposed framework can improve existing approach performance. Through extensive experiments on Visual Genome and Action Genome benchmark datasets we show improved performance on the scene graph generation.

        ----

        ## [1764] Proximal Point Imitation Learning

        **Authors**: *Luca Viano, Angeliki Kamoutsi, Gergely Neu, Igor Krawczuk, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9988f2c8e07c1f98af7ba9ca31ccae0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9988f2c8e07c1f98af7ba9ca31ccae0b-Abstract-Conference.html)

        **Abstract**:

        This work develops new algorithms with rigorous efficiency guarantees for infinite horizon imitation learning (IL) with linear function approximation without restrictive coherence assumptions. We begin with the minimax formulation of the problem and then outline how to leverage classical tools from optimization, in particular, the proximal-point method (PPM) and dual smoothing, for online and offline IL, respectively. Thanks to PPM, we avoid nested policy evaluation and cost updates for online IL appearing in the prior literature. In particular, we do away with the conventional alternating updates by the optimization of a single convex and smooth objective over both cost and $Q$-functions. When solved inexactly, we relate the optimization errors to the suboptimality of the recovered policy. As an added bonus, by re-interpreting PPM as dual smoothing with the expert policy as a center point, we also obtain an offline IL algorithm enjoying theoretical guarantees in terms of required expert trajectories. Finally, we achieve convincing empirical performance for both linear and neural network function approximation.

        ----

        ## [1765] Heatmap Distribution Matching for Human Pose Estimation

        **Authors**: *Haoxuan Qu, Li Xu, Yujun Cai, Lin Geng Foo, Jun Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/999fcab97007ebef0cda9949550b4a9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/999fcab97007ebef0cda9949550b4a9e-Abstract-Conference.html)

        **Abstract**:

        For tackling the task of 2D human pose estimation, the great majority of the recent methods regard this task as a heatmap estimation problem, and optimize the heatmap prediction using the Gaussian-smoothed heatmap as the optimization objective and using the pixel-wise loss (e.g. MSE) as the loss function. In this paper, we show that optimizing the heatmap prediction in such a way, the model performance of body joint localization, which is the intrinsic objective of this task, may not be consistently improved during the optimization process of the heatmap prediction. To address this problem, from a novel perspective, we propose to formulate the optimization of the heatmap prediction as a distribution matching problem between the predicted heatmap and the dot annotation of the body joint directly. By doing so, our proposed method does not need to construct the Gaussian-smoothed heatmap and can achieve a more consistent model performance improvement during the optimization of the heatmap prediction. We show the effectiveness of our proposed method through extensive experiments on the COCO dataset and the MPII dataset.

        ----

        ## [1766] Mining Unseen Classes via Regional Objectness: A Simple Baseline for Incremental Segmentation

        **Authors**: *Zekang Zhang, Guangyu Gao, Zhiyuan Fang, Jianbo Jiao, Yunchao Wei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/99b419554537c66bf27e5eb7a74c7de4-Abstract-Conference.html)

        **Abstract**:

        Incremental or continual learning has been extensively studied for image classification tasks to alleviate catastrophic forgetting, a phenomenon in which earlier learned knowledge is forgotten when learning new concepts. For class incremental semantic segmentation, such a phenomenon often becomes much worse due to the semantic shift of the background class, \ie, some concepts learned at previous stages are assigned to the background class at the current training stage, therefore, significantly reducing the performance of these old concepts. To address this issue, we propose a simple yet effective method in this paper, named Mining unseen Classes via Regional Objectness (MicroSeg). Our MicroSeg is based on the assumption that \emph{background regions with strong objectness possibly belong to those concepts in the historical or future stages}. Therefore, to avoid forgetting old knowledge at the current training stage, our MicroSeg first splits the given image into hundreds of segment proposals with a proposal generator. Those segment proposals with strong objectness from the background are then clustered and assigned new defined labels during the optimization. In this way, the distribution characterizes of old concepts in the feature space could be better perceived, relieving the catastrophic forgetting caused by the semantic shift of the background class accordingly.  We conduct extensive experiments on Pascal VOC and ADE20K, and competitive results well demonstrate the effectiveness of our MicroSeg. Code is available at \href{https://github.com/zkzhang98/MicroSeg}{\textcolor{orange}{\texttt{https://github.com/zkzhang98/MicroSeg}}}.

        ----

        ## [1767] Bezier Gaussian Processes for Tall and Wide Data

        **Authors**: *Martin Jørgensen, Michael A. Osborne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/99c80ceb10cb674110f03b2def6a5b76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/99c80ceb10cb674110f03b2def6a5b76-Abstract-Conference.html)

        **Abstract**:

        Modern approximations to Gaussian processes are suitable for tall data'', with a cost that scales well in the number of observations, but under-performs onwide data'', scaling poorly in the number of input features. That is, as the number of input features grows, good predictive performance requires the number of summarising variables, and their associated cost, to grow rapidly. We introduce a kernel that allows the number of summarising variables to grow exponentially with the number of input features, but requires only linear cost in both number of observations and input features. This scaling is achieved through our introduction of the ``Bezier buttress'', which allows approximate inference without computing matrix inverses or determinants. We show that our kernel has close similarities to some of the most used kernels in Gaussian process regression, and empirically demonstrate the kernel's ability to scale to both tall and wide datasets.

        ----

        ## [1768] Smoothed Embeddings for Certified Few-Shot Learning

        **Authors**: *Mikhail Pautov, Olesya Kuznetsova, Nurislam Tursynbek, Aleksandr Petiushko, Ivan V. Oseledets*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a07bb7288caaea2ecc4c367188bc6db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a07bb7288caaea2ecc4c367188bc6db-Abstract-Conference.html)

        **Abstract**:

        Randomized smoothing is considered to be the state-of-the-art provable defense against adversarial perturbations. However, it heavily exploits the fact that classifiers map input objects to class probabilities and do not focus on the ones that learn a metric space in which classification is performed by computing distances to embeddings of class prototypes. In this work, we extend randomized smoothing to few-shot learning models that map inputs to normalized embeddings. We provide analysis of the Lipschitz continuity of such models and  derive a robustness certificate against $\ell_2$-bounded perturbations that may be useful in few-shot learning scenarios. Our theoretical results are confirmed by experiments on different datasets.

        ----

        ## [1769] Finite Sample Analysis Of Dynamic Regression Parameter Learning

        **Authors**: *Mark Kozdoba, Edward Moroshko, Shie Mannor, Yacov Crammer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a0c3a83cadca7c5a7355074ae5a7569-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a0c3a83cadca7c5a7355074ae5a7569-Abstract-Conference.html)

        **Abstract**:

        We consider the dynamic linear regression problem, where the predictor vector may vary with time. This problem can be modeled as a linear dynamical system, with non-constant observation operator, where the parameters that need to be learned are the variance of both the process noise and the observation noise. While variance estimation for dynamic regression is a natural problem, with a variety of applications, existing approaches to this problem either lack guarantees altogether, or only have asymptotic guarantees without explicit rates. In particular, existing literature does not provide any clues to the following  fundamental question: In terms of data characteristics, what does the convergence rate depend on?  In this paper we study the global system operator -- the operator that maps the  noise vectors to the output. We obtain estimates on its spectrum, and as a result derive the first known variance estimators with finite sample complexity guarantees. The proposed bounds depend on the shape of a certain spectrum related to the system operator, and thus provide the first known explicit geometric parameter of the data that can be used to bound estimation errors. In addition, the results hold for arbitrary sub Gaussian distributions of noise terms.  We evaluate the approach on synthetic and real-world benchmarks.

        ----

        ## [1770] Group Meritocratic Fairness in Linear Contextual Bandits

        **Authors**: *Riccardo Grazzi, Arya Akhavan, John Isak Texas Falk, Leonardo Cella, Massimiliano Pontil*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a1dab894ce96cb8339c2fadd85a100b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a1dab894ce96cb8339c2fadd85a100b-Abstract-Conference.html)

        **Abstract**:

        We study the linear contextual bandit problem where an agent has to select one candidate from a pool and each candidate belongs to a sensitive group. In this setting, candidates' rewards may not be directly comparable between groups, for example when the agent is an employer hiring candidates from different ethnic groups and some groups have a lower reward due to discriminatory bias and/or social injustice. We propose a notion of fairness that states that the agent's policy is fair when it selects a candidate with highest relative rank, which measures how good the reward is when compared to candidates from the same group. This is a very strong notion of fairness, since the relative rank is not directly observed by the agent and depends on the underlying reward model and on the distribution of rewards. Thus we study the problem of learning a policy which approximates a fair policy under the condition that the contexts are independent between groups and the distribution of rewards of each group is absolutely continuous. In particular, we design a greedy policy which at each round constructs a ridge regression estimate from the observed context-reward pairs, and then computes an estimate of the relative rank of each candidate using the empirical cumulative distribution function. We prove that, despite its simplicity and the lack of an initial exploration phase, the greedy policy achieves, up to log factors and with high probability, a fair pseudo-regret of order $\sqrt{dT}$ after $T$ rounds, where $d$ is the dimension of the context vectors. The policy also satisfies demographic parity at each round when averaged over all possible information available before the selection. Finally, we use simulated settings and experiments on the US census data to show that our policy achieves sub-linear fair pseudo-regret also in practice.

        ----

        ## [1771] New Lower Bounds for Private Estimation and a Generalized Fingerprinting Lemma

        **Authors**: *Gautam Kamath, Argyris Mouzakis, Vikrant Singhal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a6b278218966499194491f55ccf8b75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a6b278218966499194491f55ccf8b75-Abstract-Conference.html)

        **Abstract**:

        We prove new lower bounds for statistical estimation tasks under the constraint of $(\varepsilon,\delta)$-differential privacy. First, we provide tight lower bounds for private covariance estimation of Gaussian distributions. We show that estimating the covariance matrix in Frobenius norm requires $\Omega(d^2)$ samples, and in spectral norm requires $\Omega(d^{3/2})$ samples, both matching upper bounds up to logarithmic factors. We prove these bounds via our main technical contribution, a broad generalization of the fingerprinting method to exponential families. Additionally, using the private Assouad method of Acharya, Sun, and Zhang, we show a tight $\Omega(d/(\alpha^2 \varepsilon))$ lower bound for estimating the mean of a distribution with bounded covariance to $\alpha$-error in $\ell_2$-distance. Prior known lower bounds for all these problems were either polynomially weaker or held under the stricter condition of $(\varepsilon,0)$-differential privacy.

        ----

        ## [1772] Lower Bounds on Randomly Preconditioned Lasso via Robust Sparse Designs

        **Authors**: *Jonathan A. Kelner, Frederic Koehler, Raghu Meka, Dhruv Rohatgi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a8d52eb05eb7b13f54b3d9eada667b7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a8d52eb05eb7b13f54b3d9eada667b7-Abstract-Conference.html)

        **Abstract**:

        Sparse linear regression with ill-conditioned Gaussian random covariates is widely believed to exhibit a statistical/computational gap, but there is surprisingly little formal evidence for this belief. Recent work has shown that, for certain covariance matrices, the broad class of Preconditioned Lasso programs provably cannot succeed on polylogarithmically sparse signals with a sublinear number of samples. However, this lower bound only holds against deterministic preconditioners, and in many contexts randomization is crucial to the success of preconditioners. We prove a stronger lower bound that rules out randomized preconditioners. For an appropriate covariance matrix, we construct a single signal distribution on which any invertibly-preconditioned Lasso program fails with high probability, unless it receives a linear number of samples. Surprisingly, at the heart of our lower bound is a new robustness result in compressed sensing. In particular, we study recovering a sparse signal when a few measurements can be erased adversarially. To our knowledge, this natural question has not been studied before for sparse measurements. We surprisingly show that standard sparse Bernoulli measurements are almost-optimally robust to adversarial erasures: if $b$ measurements are erased, then all but $O(b)$ of the coordinates of the signal are identifiable.

        ----

        ## [1773] Model-based Safe Deep Reinforcement Learning via a Constrained Proximal Policy Optimization Algorithm

        **Authors**: *Ashish Kumar Jayant, Shalabh Bhatnagar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a8eb202c060b7d81f5889631cbcd47e-Abstract-Conference.html)

        **Abstract**:

        During initial iterations of training in most Reinforcement Learning (RL) algorithms, agents perform a significant number of random exploratory steps. In the real world, this can limit the practicality of these algorithms as it can lead to potentially dangerous behavior. Hence safe exploration is a critical issue in applying RL algorithms in the real world. This problem has been recently well studied under the Constrained Markov Decision Process (CMDP) Framework, where in addition to single-stage rewards, an agent receives single-stage costs or penalties as well depending on the state transitions. The prescribed  cost functions are responsible for mapping undesirable behavior at any given time-step to a scalar value. The goal then is to find a feasible policy that maximizes reward returns while constraining the cost returns to be below a prescribed threshold during training as well as deployment.We propose an On-policy Model-based Safe Deep RL algorithm in which we learn the transition dynamics of the environment in an online manner as well as find a feasible optimal policy using the Lagrangian Relaxation-based Proximal Policy Optimization. We use an ensemble of neural networks with different initializations to tackle epistemic and aleatoric uncertainty issues faced during environment model learning.  We compare our approach with relevant model-free and model-based approaches in Constrained RL using the  challenging Safe Reinforcement Learning benchmark - the Open AI Safety Gym.  We demonstrate that our algorithm is more sample efficient and results in lower  cumulative hazard violations as compared to constrained model-free approaches. Further, our approach shows better reward performance than other constrained model-based approaches in the literature.

        ----

        ## [1774] Exact Solutions of a Deep Linear Network

        **Authors**: *Ziyin Liu, Botao Li, Xiangming Meng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a940e858b17f01c402e164835140c4a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a940e858b17f01c402e164835140c4a-Abstract-Conference.html)

        **Abstract**:

        This work finds the analytical expression of the global minima of a deep linear network with weight decay and stochastic neurons, a fundamental model for understanding the landscape of neural networks. Our result implies that zero is a special point in deep neural network architecture. We show that weight decay strongly interacts with the model architecture and can create bad minima at zero in a network with more than $1$ hidden layer, qualitatively different from a network with only $1$ hidden layer. Practically, our result implies that common deep learning initialization methods are insufficient to ease the optimization of neural networks in general.

        ----

        ## [1775] An Adaptive Kernel Approach to Federated Learning of Heterogeneous Causal Effects

        **Authors**: *Thanh Vinh Vo, Arnab Bhattacharyya, Young Lee, Tze-Yun Leong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9a9afa70eead1805f00e3a0df2a41157-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9a9afa70eead1805f00e3a0df2a41157-Abstract-Conference.html)

        **Abstract**:

        We propose a new causal inference framework to learn causal effects from multiple, decentralized data sources in a federated setting. We introduce an adaptive transfer algorithm that learns the similarities among the data sources by utilizing Random Fourier Features to disentangle the loss function into multiple components, each of which is associated with a data source. The data sources may have different distributions; the causal effects are independently and systematically incorporated. The proposed method estimates the similarities among the sources through transfer coefficients, and hence requiring no prior information about the similarity measures. The heterogeneous causal effects can be estimated with no sharing of the raw training data among the sources, thus minimizing the risk of privacy leak. We also provide minimax lower bounds to assess the quality of the parameters learned from the disparate sources. The proposed method is empirically shown to outperform the baselines on decentralized data sources with dissimilar distributions.

        ----

        ## [1776] FeLMi : Few shot Learning with hard Mixup

        **Authors**: *Aniket Roy, Anshul Shah, Ketul Shah, Prithviraj Dhar, Anoop Cherian, Rama Chellappa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html)

        **Abstract**:

        Learning from a few examples is a challenging computer vision task. Traditionally,meta-learning-based methods have shown promise towards solving this problem.Recent approaches show benefits by learning a feature extractor on the abundantbase examples and transferring these to the fewer novel examples. However, thefinetuning stage is often prone to overfitting due to the small size of the noveldataset. To this end, we propose Few shot Learning with hard Mixup (FeLMi)using manifold mixup to synthetically generate samples that helps in mitigatingthe data scarcity issue. Different from a naïve mixup, our approach selects the hardmixup samples using an uncertainty-based criteria. To the best of our knowledge,we are the first to use hard-mixup for the few-shot learning problem. Our approachallows better use of the pseudo-labeled base examples through base-novel mixupand entropy-based filtering. We evaluate our approach on several common few-shotbenchmarks - FC-100, CIFAR-FS, miniImageNet and tieredImageNet and obtainimprovements in both 1-shot and 5-shot settings. Additionally, we experimented onthe cross-domain few-shot setting (miniImageNet → CUB) and obtain significantimprovements.

        ----

        ## [1777] EpiGRAF: Rethinking training of 3D GANs

        **Authors**: *Ivan Skorokhodov, Sergey Tulyakov, Yiqun Wang, Peter Wonka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b01333262789ea3a65a5fab4c22feae-Abstract-Conference.html)

        **Abstract**:

        A recent trend in generative modeling is building 3D-aware generators from 2D image collections. To induce the 3D bias, such models typically rely on volumetric rendering, which is expensive to employ at high resolutions. Over the past months, more than ten works have addressed this scaling issue by training a separate 2D decoder to upsample a low-resolution image (or a feature tensor) produced from a pure 3D generator.  But this solution comes at a cost: not only does it break multi-view consistency (i.e., shape and texture change when the camera moves), but it also learns geometry in low fidelity. In this work, we show that obtaining a high-resolution 3D generator with SotA image quality is possible by following a completely different route of simply training the model patch-wise. We revisit and improve this optimization scheme in two ways. First, we design a location- and scale-aware discriminator to work on patches of different proportions and spatial positions. Second, we modify the patch sampling strategy based on an annealed beta distribution to stabilize training and accelerate the convergence. The resulting model, named EpiGRAF, is an efficient, high-resolution, pure 3D generator, and we test it on four datasets (two introduced in this work) at (256^2) and (512^2) resolutions. It obtains state-of-the-art image quality, high-fidelity geometry and trains ({\approx})2.5 faster than the upsampler-based counterparts. Code/data/visualizations: https://universome.github.io/epigraf.

        ----

        ## [1778] Towards Trustworthy Automatic Diagnosis Systems by Emulating Doctors' Reasoning with Deep Reinforcement Learning

        **Authors**: *Arsène Fansi Tchango, Rishab Goel, Julien Martel, Zhi Wen, Gaétan Marceau-Caron, Joumana Ghosn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b6c8c4a5aeb6a37c9efa963e30993d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b6c8c4a5aeb6a37c9efa963e30993d9-Abstract-Conference.html)

        **Abstract**:

        The automation of the medical evidence acquisition and diagnosis process has recently attracted increasing attention in order to reduce the workload of doctors and democratize access to medical care. However, most works proposed in the machine learning literature focus solely on improving the prediction accuracy of a patient's pathology. We argue that this objective is insufficient to ensure doctors' acceptability of such systems. In their initial interaction with patients, doctors do not only focus on identifying the pathology a patient is suffering from; they instead generate a differential diagnosis (in the form of a short list of plausible diseases) because the medical evidence collected from patients is often insufficient to establish a final diagnosis. Moreover, doctors explicitly explore severe pathologies before potentially ruling them out from the differential, especially in acute care settings. Finally, for doctors to trust a system's recommendations, they need to understand how the gathered evidences led to the predicted diseases. In particular, interactions between a system and a patient need to emulate the reasoning of doctors. We therefore propose to model the evidence acquisition and automatic diagnosis tasks using a deep reinforcement learning framework that considers three essential aspects of a doctor's reasoning, namely generating a differential diagnosis using an exploration-confirmation approach while prioritizing severe pathologies. We propose metrics for evaluating interaction quality based on these three aspects. We show that our approach performs better than existing models while maintaining competitive pathology prediction accuracy.

        ----

        ## [1779] Towards Improving Faithfulness in Abstractive Summarization

        **Authors**: *Xiuying Chen, Mingzhe Li, Xin Gao, Xiangliang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b6d7202750e8e32cd5270eb7fc131f7-Abstract-Conference.html)

        **Abstract**:

        Despite the success achieved in neural abstractive summarization based on pre-trained language models, one unresolved issue is that the generated summaries are not always faithful to the input document.There are two possible causes of the unfaithfulness problem: (1) the summarization model fails to understand or capture the gist of the input text, and (2) the model over-relies on the language model to generate fluent but inadequate words.In this work, we propose a Faithfulness Enhanced Summarization model (FES), which is designed for addressing these two problems and improving faithfulness in abstractive summarization.For the first problem, we propose to use question-answering (QA) to examine whether the encoder fully grasps the input document and can answer the questions on the key information in the input. The QA attention on the proper input words can also be used to stipulate how the decoder should attend to the source.For the second problem, we introduce a max-margin loss defined on the difference between the language and the summarization model, aiming to prevent the overconfidence of the language model.Extensive experiments on two benchmark summarization datasets, CNN/DM and XSum, demonstrate that our model significantly outperforms strong baselines.The evaluation of factual consistency also shows that our model generates more faithful summaries than baselines.

        ----

        ## [1780] ZIN: When and How to Learn Invariance Without Environment Partition?

        **Authors**: *Yong Lin, Shengyu Zhu, Lu Tan, Peng Cui*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b77f07301b1ef1fe810aae96c12cb7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b77f07301b1ef1fe810aae96c12cb7b-Abstract-Conference.html)

        **Abstract**:

        It is commonplace to encounter heterogeneous data, of which some aspects of the data distribution may vary  but the underlying causal mechanisms remain constant.  When data are divided into distinct environments according to the heterogeneity, recent invariant learning methods have proposed to learn robust and invariant models using this environment partition. It is hence tempting to utilize the inherent heterogeneity even when environment partition is not provided. Unfortunately, in this work, we show that learning invariant features under this circumstance is fundamentally impossible without further inductive biases or additional information. Then, we propose a framework to jointly learn environment partition and invariant representation, assisted by additional auxiliary information. We derive sufficient and necessary conditions for our framework to provably identify invariant features under a fairly general setting. Experimental results on both synthetic and real world datasets validate our analysis and demonstrate an improved performance of the proposed framework. Our findings also raise the need of making the role of  inductive biases more explicit when learning invariant models without environment partition in future works. Codes are available at https://github.com/linyongver/ZIN_official .

        ----

        ## [1781] Random Sharpness-Aware Minimization

        **Authors**: *Yong Liu, Siqi Mai, Minhao Cheng, Xiangning Chen, Cho-Jui Hsieh, Yang You*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b79416c0dc4b09feaa169ed5cdd63d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b79416c0dc4b09feaa169ed5cdd63d4-Abstract-Conference.html)

        **Abstract**:

        Currently, Sharpness-Aware Minimization (SAM) is proposed to seek the parameters that lie in a flat region to improve the generalization when training neural networks. In particular, a minimax optimization objective is defined to find the maximum loss value centered on the weight, out of the purpose of simultaneously minimizing loss value and loss sharpness. For the sake of simplicity, SAM applies one-step gradient ascent to approximate the solution of the inner maximization.  However, one-step gradient ascent may not be sufficient and multi-step gradient ascents will cause additional training costs.  Based on this observation, we propose a novel random smoothing based SAM (R-SAM) algorithm. To be specific, R-SAM essentially smooths the loss landscape, based on which we are able to apply the one-step gradient ascent on the smoothed weights to improve the approximation of the inner maximization. Further, we evaluate our proposed R-SAM on CIFAR and ImageNet datasets. The experimental results illustrate that R-SAM can consistently improve the performance on ResNet and Vision Transformer (ViT) training.

        ----

        ## [1782] Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation

        **Authors**: *Jannik Kossen, Sebastian Farquhar, Yarin Gal, Thomas Rainforth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9b9cfd5428153ccfbd4ba34b7e007305-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9b9cfd5428153ccfbd4ba34b7e007305-Abstract-Conference.html)

        **Abstract**:

        We propose Active Surrogate Estimators (ASEs), a new method for label-efficient model evaluation. Evaluating model performance is a challenging and important problem when labels are expensive. ASEs address this active testing problem using a surrogate-based estimation approach that interpolates the errors of points with unknown labels, rather than forming a Monte Carlo estimator. ASEs actively learn the underlying surrogate, and we propose a novel acquisition strategy, XWED, that tailors this learning to the final estimation task. We find that ASEs offer greater label-efficiency than the current state-of-the-art when applied to challenging model evaluation problems for deep neural networks.

        ----

        ## [1783] HAPI: A Large-scale Longitudinal Dataset of Commercial ML API Predictions

        **Authors**: *Lingjiao Chen, Zhihua Jin, Sabri Eyuboglu, Christopher Ré, Matei Zaharia, James Y. Zou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9bcd0bdb2777fe8c729b682f07e993f1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9bcd0bdb2777fe8c729b682f07e993f1-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Commercial ML APIs offered by providers such as Google, Amazon and Microsoft have dramatically simplified ML adoptions in many applications. Numerous companies and academics pay to use ML APIs for tasks such as object detection, OCR and sentiment analysis. Different ML APIs tackling the same task can have very heterogeneous performances. Moreover, the ML models underlying the APIs also evolve over time. As ML APIs rapidly become a valuable marketplace and an integral part of analytics, it is critical to systematically study and compare different APIs with each other and to characterize how individual APIs change over time. However, this practically important topic is currently underexplored due to the lack of data. In this paper, we present HAPI (History of APIs), a longitudinal dataset of 1,761,417 instances of commercial ML API applications (involving APIs from Amazon, Google, IBM, Microsoft and other providers) across diverse tasks including image tagging, speech recognition, and text mining from 2020 to 2022. Each instance consists of a query input for an API (e.g., an image or text) along with the API’s output prediction/annotation and confidence scores. HAPI is the first large-scale dataset of ML API usages and is a unique resource for studying ML  as-a-service (MLaaS). As examples of the types of analyses that HAPI enables, we show that ML APIs’ performance changes substantially over time—several APIs’ accuracies dropped on specific benchmark datasets. Even when the API’s aggregate performance stays steady, its error modes can shift across different subtypes of data between 2020 and 2022. Such changes can substantially impact the entire analytics pipelines that use some ML API as a component. We further use HAPI to study commercial APIs’ performance disparities across demographic subgroups over time. HAPI can stimulate more research in the growing field of MLaaS.

        ----

        ## [1784] Near-Optimal Regret Bounds for Multi-batch Reinforcement Learning

        **Authors**: *Zihan Zhang, Yuhang Jiang, Yuan Zhou, Xiangyang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9bcd1fa0c05e5f25ba7a1261f1852e82-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9bcd1fa0c05e5f25ba7a1261f1852e82-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the episodic reinforcement learning (RL) problem modeled by finite-horizon Markov Decision Processes (MDPs) with constraint on the number of batches. The multi-batch reinforcement learning framework, where the agent is required to provide a time schedule to update policy before everything, which is particularly suitable for the scenarios where the agent suffers extensively from changing the policy adaptively. Given a finite-horizon MDP with $S$ states, $A$ actions and planning horizon $H$, we design a computational efficient algorithm to achieve near-optimal regret of $\tilde{O}(\sqrt{SAH^3K\ln(1/\delta)})$\footnote{$\tilde{O}(\cdot)$ hides logarithmic terms of $(S,A,H,K)$} in $K$ episodes using $O\left(H+\log_2\log_2(K) \right)$ batches with confidence parameter $\delta$. To our best of knowledge, it is the first $\tilde{O}(\sqrt{SAH^3K})$ regret bound with $O(H+\log_2\log_2(K))$ batch complexity. Meanwhile, we show that to achieve $\tilde{O}(\mathrm{poly}(S,A,H)\sqrt{K})$ regret, the number of batches is at least $\Omega\left(H/\log_A(K)+ \log_2\log_2(K) \right)$, which matches our upper bound up to logarithmic terms.Our technical contribution are two-fold: 1) a near-optimal design scheme to explore over the unlearned states; 2) an computational efficient algorithm to explore certain directions with an approximated transition model.ion model.

        ----

        ## [1785] OST: Improving Generalization of DeepFake Detection via One-Shot Test-Time Training

        **Authors**: *Liang Chen, Yong Zhang, Yibing Song, Jue Wang, Lingqiao Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9bf0810a4a1597a36d27ceea58667d92-Abstract-Conference.html)

        **Abstract**:

        State-of-the-art deepfake detectors perform well in identifying forgeries when they are evaluated on a test set similar to the training set, but struggle to maintain good performance when the test forgeries exhibit different characteristics from the training images e.g., forgeries are created by unseen deepfake methods. Such a weak generalization capability hinders the applicability of deepfake detectors. In this paper, we introduce a new learning paradigm specially designed for the generalizable deepfake detection task. Our key idea is to construct a test-sample-specific auxiliary task to update the model before applying it to the sample. Specifically, we synthesize pseudo-training samples from each test image and create a test-time training objective to update the model. Moreover, we proposed to leverage meta-learning to ensure that a fast single-step test-time gradient descent, dubbed one-shot test-time training (OST), can be sufficient for good deepfake detection performance. Extensive results across several benchmark datasets demonstrate that our approach performs favorably against existing arts in terms of generalization to unseen data and robustness to different post-processing steps.

        ----

        ## [1786] The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games

        **Authors**: *Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre M. Bayen, Yi Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Proximal Policy Optimization (PPO) is a ubiquitous on-policy reinforcement learning algorithm but is significantly less utilized than off-policy learning algorithms in multi-agent settings. This is often due to the belief that PPO is significantly less sample efficient than off-policy methods in multi-agent systems. In this work, we carefully study the performance of PPO in cooperative multi-agent settings. We show that PPO-based multi-agent algorithms achieve surprisingly strong performance in four popular multi-agent testbeds: the particle-world environments, the StarCraft multi-agent challenge, the Hanabi challenge, and Google Research Football, with minimal hyperparameter tuning and without any domain-specific algorithmic modifications or architectures. Importantly, compared to competitive off-policy methods, PPO often achieves competitive or superior results in both final returns and sample efficiency. Finally, through ablation studies, we analyze implementation and hyperparameter factors that are critical to PPO's empirical performance, and give concrete practical suggestions regarding these factors. Our results show that when using these practices, simple PPO-based methods are a strong baseline in cooperative multi-agent reinforcement learning. Source code is released at https://github.com/marlbenchmark/on-policy.

        ----

        ## [1787] Resolving the data ambiguity for periodic crystals

        **Authors**: *Daniel Widdowson, Vitaliy Kurlin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c256fa1965318b7fcb9ed104c265540-Abstract-Conference.html)

        **Abstract**:

        The fundamental model of all solid crystalline materials is a periodic set of atomic centers considered up to rigid motion in Euclidean space. The major obstacle to materials discovery was highly ambiguous representations of periodic crystals that didn't allow fast and reliable comparisons and led to numerous (near-) duplicates in many databases of experimental and simulated crystals. This paper exemplarily resolves the ambiguity by invariants, which are descriptors without false negatives.The new Pointwise Distance Distributions (PDD) is a numerical matrix with a near-linear time complexity and an exactly computable metric. The strongest theoretical result is generic completeness (absence of false positives) for all finite and periodic sets of points in any dimension. The strength of PDD is shown by 200B+ pairwise comparisons of all periodic structures in the world's largest collection (Cambridge Structural Database) of existing materials over two days on a modest desktop.

        ----

        ## [1788] Video PreTraining (VPT): Learning to Act by Watching Unlabeled Online Videos

        **Authors**: *Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, Jeff Clune*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c7008aff45b5d8f0973b23e1a22ada0-Abstract-Conference.html)

        **Abstract**:

        Pretraining on noisy, internet-scale datasets has been heavily studied as a technique for training models with broad, general capabilities for text, images, and other modalities. However, for many sequential decision domains such as robotics, video games, and computer use, publicly available data does not contain the labels required to train behavioral priors in the same way. We extend the internet-scale pretraining paradigm to sequential decision domains through semi-supervised imitation learning wherein agents learn to act by watching online unlabeled videos. Specifically, we show that with a small amount of labeled data we can train an inverse dynamics model accurate enough to label a huge unlabeled source of online data -- here, online videos of people playing Minecraft -- from which we can then train a general behavioral prior. Despite using the native human interface (mouse and keyboard at 20Hz), we show that this behavioral prior has nontrivial zero-shot capabilities and that it can be fine-tuned, with both imitation learning and reinforcement learning, to hard-exploration tasks that are impossible to learn from scratch via reinforcement learning. For many tasks our models exhibit human-level performance, and we are the first to report computer agents that can craft diamond tools, which can take proficient humans upwards of 20 minutes (24,000 environment actions) of gameplay to accomplish.

        ----

        ## [1789] GLOBEM Dataset: Multi-Year Datasets for Longitudinal Human Behavior Modeling Generalization

        **Authors**: *Xuhai Xu, Han Zhang, Yasaman S. Sefidgar, Yiyi Ren, Xin Liu, Woosuk Seo, Jennifer Brown, Kevin S. Kuehn, Mike A. Merrill, Paula S. Nurius, Shwetak N. Patel, Tim Althoff, Margaret E. Morris, Eve A. Riskin, Jennifer Mankoff, Anind K. Dey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c7e8a0821dfcb58a9a83cbd37cc8131-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c7e8a0821dfcb58a9a83cbd37cc8131-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Recent research has demonstrated the capability of behavior signals captured by smartphones and wearables for longitudinal behavior modeling. However, there is a lack of a comprehensive public dataset that serves as an open testbed for fair comparison among algorithms. Moreover, prior studies mainly evaluate algorithms using data from a single population within a short period, without measuring the cross-dataset generalizability of these algorithms. We present the first multi-year passive sensing datasets, containing over 700 user-years and 497 unique users’ data collected from mobile and wearable sensors, together with a wide range of well-being metrics. Our datasets can support multiple cross-dataset evaluations of behavior modeling algorithms’ generalizability across different users and years. As a starting point, we provide the benchmark results of 18 algorithms on the task of depression detection. Our results indicate that both prior depression detection algorithms and domain generalization techniques show potential but need further research to achieve adequate cross-dataset generalizability. We envision our multi-year datasets can support the ML community in developing generalizable longitudinal behavior modeling algorithms.

        ----

        ## [1790] Shape And Structure Preserving Differential Privacy

        **Authors**: *Carlos Soto, Karthik Bharath, Matthew Reimherr, Aleksandra B. Slavkovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c84feb75eae1ef6389f31b3ef050b6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c84feb75eae1ef6389f31b3ef050b6a-Abstract-Conference.html)

        **Abstract**:

        It is common for data structures such as images and shapes of 2D objects to be represented as points on a manifold. The utility of a mechanism to produce sanitized differentially private estimates from such data is intimately linked to how compatible it is with the underlying structure and geometry of the space. In particular, as recently shown, utility of the Laplace mechanism on a positively curved manifold, such as Kendall’s 2D shape space, is significantly influenced by the curvature. Focusing on the problem of sanitizing the Fr\'echet mean of a sample of points on a manifold, we exploit the characterization of the mean as the minimizer of an objective function comprised of the sum of squared distances and develop a K-norm gradient mechanism on Riemannian manifolds that favors values that produce gradients close to the the zero of the objective function. For the case of positively curved manifolds, we describe how using the gradient of the squared distance function offers better control over sensitivity than the Laplace mechanism, and demonstrate this numerically on a dataset of shapes of corpus callosa. Further illustrations of the mechanism’s utility on a sphere and the manifold of symmetric positive definite matrices are also presented.

        ----

        ## [1791] Transformers meet Stochastic Block Models: Attention with Data-Adaptive Sparsity and Cost

        **Authors**: *Sungjun Cho, Seonwoo Min, Jinwoo Kim, Moontae Lee, Honglak Lee, Seunghoon Hong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9c93b3cd3bc60c0fe7b0c2d74a2da966-Abstract-Conference.html)

        **Abstract**:

        To overcome the quadratic cost of self-attention, recent works have proposed various sparse attention modules, most of which fall under one of two groups: 1) sparse attention under a hand-crafted patterns and 2) full attention followed by a sparse variant of softmax such as $\alpha$-entmax. Unfortunately, the first group lacks adaptability to data while the second still requires quadratic cost in training. In this work, we propose SBM-Transformer, a model that resolves both problems by endowing each attention head with a mixed-membership Stochastic Block Model (SBM). Then, each attention head data-adaptively samples a bipartite graph, the adjacency of which is used as an attention mask for each input. During backpropagation, a straight-through estimator is used to flow gradients beyond the discrete sampling step and adjust the probabilities of sampled edges based on the predictive loss. The forward and backward cost are thus linear to the number of edges, which each attention head can also choose flexibly based on the input. By assessing the distribution of graphs, we theoretically show that SBM-Transformer is a universal approximator for arbitrary sequence-to-sequence functions in expectation. Empirical evaluations under the LRA and GLUE benchmarks demonstrate that our model outperforms previous efficient variants as well as the original Transformer with full attention. Our implementation can be found in https://github.com/sc782/SBM-Transformer.

        ----

        ## [1792] Characteristics of Harmful Text: Towards Rigorous Benchmarking of Language Models

        **Authors**: *Maribeth Rauh, John Mellor, Jonathan Uesato, Po-Sen Huang, Johannes Welbl, Laura Weidinger, Sumanth Dathathri, Amelia Glaese, Geoffrey Irving, Iason Gabriel, William Isaac, Lisa Anne Hendricks*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9ca22870ae0ba55ee50ce3e2d269e5de-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Large language models produce human-like text that drive a growing number of applications.  However, recent literature and, increasingly, real world observations, have demonstrated that these models can generate language that is toxic, biased, untruthful or otherwise harmful.  Though work to evaluate language model harms is under way, translating foresight about which harms may arise into rigorous benchmarks is not straightforward.  To facilitate this translation, we outline six ways of characterizing harmful text which merit explicit consideration when designing new benchmarks.  We then use these characteristics as a lens to identify trends and gaps in existing benchmarks. Finally, we apply them in a case study of the Perspective API, a toxicity classifier that is widely used in harm benchmarks.  Our characteristics provide one piece of the bridge that translates between foresight and effective evaluation.

        ----

        ## [1793] APG: Adaptive Parameter Generation Network for Click-Through Rate Prediction

        **Authors**: *Bencheng Yan, Pengjie Wang, Kai Zhang, Feng Li, Hongbo Deng, Jian Xu, Bo Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9cd0c57170f48520749d5ae62838241f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9cd0c57170f48520749d5ae62838241f-Abstract-Conference.html)

        **Abstract**:

        In many web applications, deep learning-based CTR prediction models (deep CTR models for short) are widely adopted. Traditional deep CTR models learn patterns in a static manner, i.e., the network parameters are the same across all the instances. However, such a manner can hardly characterize each of the instances which may have different underlying distributions. It actually limits the representation power of deep CTR models, leading to sub-optimal results. In this paper, we propose an efficient, effective, and universal module, named as Adaptive Parameter Generation network (APG), which can dynamically generate parameters for deep CTR models on-the-fly based on different instances. Extensive experimental evaluation results show that APG can be applied to a variety of deep CTR models and significantly improve their performance. Meanwhile, APG can reduce the time cost by 38.7\% and memory usage by 96.6\% compared to a regular deep CTR model.We have deployed APG in the industrial sponsored search system and achieved 3\% CTR gain and 1\% RPM gain respectively.

        ----

        ## [1794] NeoRL: A Near Real-World Benchmark for Offline Reinforcement Learning

        **Authors**: *Rongjun Qin, Xingyuan Zhang, Songyi Gao, Xiong-Hui Chen, Zewen Li, Weinan Zhang, Yang Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9cd828eb8dc81a84fb6bf89a94263e1b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/9cd828eb8dc81a84fb6bf89a94263e1b-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Offline reinforcement learning (RL) aims at learning effective policies from historical data without extra environment interactions. During our experience of applying offline RL, we noticed that previous offline RL benchmarks commonly involve significant reality gaps, which we have identified include rich and overly exploratory datasets, degraded baseline, and missing policy validation. In many real-world situations, to ensure system safety, running an overly exploratory policy to collect various data is prohibited, thus only a narrow data distribution is available. The resulting policy is regarded as effective if it is better than the working behavior policy; the policy model can be deployed only if it has been well validated, rather than accomplished the training. In this paper, we present a Near real-world offline RL benchmark, named NeoRL, to reflect these properties. NeoRL datasets are collected with a more conservative strategy. Moreover, NeoRL contains the offline training and offline validation pipeline before the online test, corresponding to real-world situations. We then evaluate recent state-of-the-art offline RL algorithms in NeoRL. The empirical results demonstrate that some offline RL algorithms are less competitive to the behavior cloning and the deterministic behavior policy, implying that they could be less effective in real-world tasks than in the previous benchmarks. We also disclose that current offline policy evaluation methods could hardly select the best policy. We hope this work will shed some light on future research and deploying RL in real-world systems.

        ----

        ## [1795] Stochastic Halpern Iteration with Variance Reduction for Stochastic Monotone Inclusions

        **Authors**: *Xufeng Cai, Chaobing Song, Cristóbal Guzmán, Jelena Diakonikolas*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9cf5fff2f85310e6ece5bc3a8489b6fa-Abstract-Conference.html)

        **Abstract**:

        We study stochastic monotone inclusion problems, which widely appear in machine learning applications, including robust regression and adversarial learning. We propose novel variants of stochastic Halpern iteration with recursive variance reduction. In the cocoercive---and more generally Lipschitz-monotone---setup, our algorithm attains $\epsilon$ norm of the operator with $\mathcal{O}(\frac{1}{\epsilon^3})$ stochastic operator evaluations, which significantly improves over state of the art $\mathcal{O}(\frac{1}{\epsilon^4})$ stochastic operator evaluations required for existing monotone inclusion solvers applied to the same problem classes. We further show how to couple one of the proposed variants of stochastic Halpern iteration with a scheduled restart scheme to solve stochastic monotone inclusion problems with ${\mathcal{O}}(\frac{\log(1/\epsilon)}{\epsilon^2})$ stochastic operator evaluations under additional sharpness or strong monotonicity assumptions.

        ----

        ## [1796] SNN-RAT: Robustness-enhanced Spiking Neural Network through Regularized Adversarial Training

        **Authors**: *Jianhao Ding, Tong Bu, Zhaofei Yu, Tiejun Huang, Jian K. Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9cf904c86cc5f9ac95646c07d2cfa241-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9cf904c86cc5f9ac95646c07d2cfa241-Abstract-Conference.html)

        **Abstract**:

        Spiking neural networks (SNNs) are promising to be widely deployed in real-time and safety-critical applications with the advance of neuromorphic computing. Recent work has demonstrated the insensitivity of SNNs to small random perturbations due to the discrete internal information representation. The variety of training algorithms and the involvement of the temporal dimension pose more threats to the robustness of SNNs than that of typical neural networks. We account for the vulnerability of SNNs by constructing adversaries based on different differentiable approximation techniques. By deriving a Lipschitz constant specifically for the spike representation, we first theoretically answer the question of how much adversarial invulnerability is retained in SNNs. Hence, to defend against the broad attack methods, we propose a regularized adversarial training scheme with low computational overheads. SNNs can benefit from the constraint of the perturbed spike distance's amplification and the generalization on multiple adversarial $\epsilon$-neighbourhoods. Our experiments on the image recognition benchmarks have proven that our training scheme can defend against powerful adversarial attacks crafted from strong differentiable approximations. To be specific, our approach makes the black-box attacks of the Projected Gradient Descent attack nearly ineffective. We believe that our work will facilitate the spread of SNNs for safety-critical applications and help understand the robustness of the human brain.

        ----

        ## [1797] The Mechanism of Prediction Head in Non-contrastive Self-supervised Learning

        **Authors**: *Zixin Wen, Yuanzhi Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d276b0a087efdd2404f3295b26c24c1-Abstract-Conference.html)

        **Abstract**:

        The surprising discovery of the BYOL method shows the negative samples can be replaced by adding the prediction head to the network.  It is mysterious why even when there exist trivial collapsed global optimal solutions, neural networks trained by (stochastic) gradient descent can still learn competitive representations. In this work, we present our empirical and theoretical discoveries on non-contrastive self-supervised learning. Empirically, we find that when the prediction head is initialized as an identity matrix with only its off-diagonal entries being trainable, the network can learn competitive representations even though the trivial optima still exist in the training objective. Theoretically, we characterized the substitution effect and acceleration effect of the trainable, but identity-initialized prediction head. The substitution effect happens when learning the stronger features in some neurons can substitute for learning these features in other neurons through updating the prediction head. And the acceleration effect happens when the substituted features can accelerate the learning of other weaker features to prevent them from being ignored. These two effects enable the neural networks to learn diversified features rather than focus only on learning the strongest features, which is likely the cause of the dimensional collapse phenomenon. To the best of our knowledge, this is also the first end-to-end optimization guarantee for non-contrastive methods using nonlinear neural networks with a trainable prediction head and normalization.

        ----

        ## [1798] Counterfactual Temporal Point Processes

        **Authors**: *Kimia Noorbakhsh, Manuel Gomez-Rodriguez*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d3faa41886997cfc2128b930077fa49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d3faa41886997cfc2128b930077fa49-Abstract-Conference.html)

        **Abstract**:

        Machine learning models based on temporal point processes are the state of the art in a wide variety of applications involving discrete events in continuous time. However, these models lack the ability to answer counterfactual questions, which are increasingly relevant as these models are being used to inform targeted interventions. In this work, our goal is to fill this gap. To this end, we first develop a causal model of thinning for temporal point processes that builds upon the Gumbel-Max structural causal model. This model satisfies a desirable counterfactual monotonicity condition, which is sufficient to identify counterfactual dynamics in the process of thinning. Then, given an observed realization of a temporal point process with a given intensity function, we develop a sampling algorithm that uses the above causal model of thinning and the superposition theorem to simulate counterfactual realizations of the temporal point process under a given alternative intensity function. Simulation experiments using synthetic and real epidemiological data show that the counterfactual realizations provided by our algorithm may give valuable insights to enhance targeted interventions.

        ----

        ## [1799] Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

        **Authors**: *Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)

        **Abstract**:

        We explore how generating a chain of thought---a series of intermediate reasoning steps---significantly improves the ability of large language models to perform complex reasoning. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.

        ----

        

[Go to the previous page](NIPS-2022-list08.md)

[Go to the next page](NIPS-2022-list10.md)

[Go to the catalog section](README.md)