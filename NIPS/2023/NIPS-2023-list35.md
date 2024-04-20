## [0] Anytime-Competitive Reinforcement Learning with Policy Prior

**Authors**: *Jianyi Yang, Pengfei Li, Tongxin Li, Adam Wierman, Shaolei Ren*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f53437debdd397c42929d929614bc705-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f53437debdd397c42929d929614bc705-Abstract-Conference.html)

**Abstract**:

This paper studies the problem of Anytime-Competitive Markov Decision Process (A-CMDP). Existing works on Constrained Markov Decision Processes (CMDPs) aim to optimize the expected reward while constraining the expected cost over random dynamics, but the cost in a specific episode can still be unsatisfactorily high. In contrast, the goal of A-CMDP is to optimize the expected reward while guaranteeing a bounded cost in each round of any episode against a policy prior. We propose a new algorithm, called Anytime-Competitive Reinforcement Learning (ACRL), which provably guarantees the anytime cost constraints. The regret analysis shows the policy asymptotically matches the optimal reward achievable under the anytime competitive constraints. Experiments on the application of carbon-intelligent computing verify the reward performance and cost constraint guarantee of ACRL.

----

## [0] Metis: Understanding and Enhancing In-Network Regular Expressions

**Authors**: *Zhengxin Zhang, Yucheng Huang, Guanglin Duan, Qing Li, Dan Zhao, Yong Jiang, Lianbo Ma, Xi Xiao, Hengyang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f54bd48aba0dff7acdac86123188f1b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f54bd48aba0dff7acdac86123188f1b6-Abstract-Conference.html)

**Abstract**:

Regular expressions (REs) offer one-shot solutions for many networking tasks, e.g., network intrusion detection. However, REs purely rely on expert knowledge and cannot utilize labeled data for better accuracy. Today, neural networks (NNs) have shown superior accuracy and flexibility, thanks to their ability to learn from rich labeled data. Nevertheless, NNs are often incompetent in cold-start scenarios and too complex for deployment on network devices. In this paper, we propose Metis, a general framework that converts REs to network device affordable models for superior accuracy and throughput by taking advantage of REs' expert knowledge and NNs' learning ability. In Metis, we convert REs to byte-level recurrent neural networks (BRNNs) without training. The BRNNs preserve expert knowledge from REs and offer adequate accuracy in cold-start scenarios. When rich labeled data is available, the performance of BRNNs can be improved by training. Furthermore, we design a semi-supervised knowledge distillation to transform the BRNNs into pooling soft random forests (PSRFs) that can be deployed on network devices. To the best of our knowledge, this is the first method to employ model inference as an alternative to RE matching in network scenarios. We collect network traffic data on our campus for three weeks and evaluate Metis on them. Experimental results show that Metis is more accurate than original REs and other baselines, achieving superior throughput when deployed on network devices.

----

## [0] Adaptive Test-Time Personalization for Federated Learning

**Authors**: *Wenxuan Bao, Tianxin Wei, Haohan Wang, Jingrui He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f555b62384279b98732204cb1a670a23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f555b62384279b98732204cb1a670a23-Abstract-Conference.html)

**Abstract**:

Personalized federated learning algorithms have shown promising results in adapting models to various distribution shifts. However, most of these methods require labeled data on testing clients for personalization, which is usually unavailable in real-world scenarios. In this paper, we introduce a novel setting called test-time personalized federated learning (TTPFL), where clients locally adapt a global model in an unsupervised way without relying on any labeled data during test-time. While traditional test-time adaptation (TTA) can be used in this scenario, most of them inherently assume training data come from a single domain, while they come from multiple clients (source domains) with different distributions. Overlooking these domain interrelationships can result in suboptimal generalization. Moreover, most TTA algorithms are designed for a specific kind of distribution shift and lack the flexibility to handle multiple kinds of distribution shifts in FL. In this paper, we find that this lack of flexibility partially results from their pre-defining which modules to adapt in the model. To tackle this challenge, we propose a novel algorithm called ATP to adaptively learns the adaptation rates for each module in the model from distribution shifts among source domains. Theoretical analysis proves the strong generalization of ATP. Extensive experiments demonstrate its superiority in handling various distribution shifts including label shift, image corruptions, and domain shift, outperforming existing TTA methods across multiple datasets and model architectures. Our code is available at https://github.com/baowenxuan/ATP.

----

## [0] Context-lumpable stochastic bandits

**Authors**: *Chung-Wei Lee, Qinghua Liu, Yasin Abbasi-Yadkori, Chi Jin, Tor Lattimore, Csaba Szepesvári*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f564a952c1b86684baf7d7241ae27ac8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f564a952c1b86684baf7d7241ae27ac8-Abstract-Conference.html)

**Abstract**:

We consider a contextual bandit problem with $S $ contexts and $K $ actions. In each round $t=1,2,\dots$ the learnerobserves a random context and chooses an action based on its past experience. The learner then observes a random reward whose mean is a function of the context and the action for the round. Under the assumption that the contexts can be lumped into $r\le \min(S ,K)$ groups such that the mean reward for the various actions is the same for any two contexts that are in the same group, we give an algorithm that outputs an $\epsilon$-optimal policy after using at most $\widetilde O(r (S +K )/\epsilon^2)$ samples with high probability and provide a matching $\widetilde\Omega(r (S +K )/\epsilon^2)$ lower bound. In the regret minimization setting, we give an algorithm whose cumulative regret up to time $T$ is bounded by $\widetilde O(\sqrt{r ^3(S +K )T})$. To the best of our knowledge, we are the first to show the near-optimal sample complexity in the PAC setting and $\widetilde O{\sqrt{\text{poly}(r)(S+K)T}}$ minimax regret in the online setting for this problem.  We also show our algorithms can be applied to more general low-rank bandits and get improved regret bounds in some scenarios.

----

## [0] Text Alignment Is An Efficient Unified Model for Massive NLP Tasks

**Authors**: *Yuheng Zha, Yichi Yang, Ruichen Li, Zhiting Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f5708199bdc013c5b56406db305b991e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f5708199bdc013c5b56406db305b991e-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs), typically designed as a function of next-word prediction, have excelled across extensive NLP tasks. Despite the generality, next-word prediction is often not an efficient formulation for many of the tasks, demanding an extreme scale of model parameters (10s or 100s of billions) and sometimes yielding suboptimal performance.In practice, it is often desirable to build more efficient models---despite being less versatile, they still apply to a substantial subset of problems, delivering on par or even superior performance with much smaller model sizes.In this paper, we propose text alignment as an efficient unified model for a wide range of crucial tasks involving text entailment, similarity, question answering (and answerability), factual consistency, and so forth. Given a pair of texts, the model measures the degree of alignment between their information. We instantiate an alignment model through lightweight finetuning of RoBERTa (355M parameters) using 5.9M examples from 28 datasets. Despite its compact size, extensive experiments show the model's efficiency and strong performance: (1) On over 20 datasets of aforementioned diverse tasks, the model matches or surpasses FLAN-T5 models that have around 2x or 10x more parameters; the single unified model also outperforms task-specific models finetuned on individual datasets; (2) When applied to evaluate factual consistency of language generation on 23 datasets, our model improves over various baselines, including the much larger GPT-3.5 (ChatGPT) and sometimes even GPT-4; (3) The lightweight model can also serve as an add-on component for LLMs such as GPT-3.5 in question answering tasks, improving the average exact match (EM) score by 17.94 and F1 score by 15.05 through identifying unanswerable questions.

----

## [0] Learning from Active Human Involvement through Proxy Value Propagation

**Authors**: *Zhenghao Mark Peng, Wenjie Mo, Chenda Duan, Quanyi Li, Bolei Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f57ffe47d0b528fbb97901d16bd4eba2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f57ffe47d0b528fbb97901d16bd4eba2-Abstract-Conference.html)

**Abstract**:

Learning from active human involvement enables the human subject to actively intervene and demonstrate to the AI agent during training. The interaction and corrective feedback from human brings safety and AI alignment to the learning process. In this work, we propose a new reward-free active human involvement method called Proxy Value Propagation for policy optimization. Our key insight is that a proxy value function can be designed to express human intents, wherein state- action pairs in the human demonstration are labeled with high values, while those agents’ actions that are intervened receive low values. Through the TD-learning framework, labeled values of demonstrated state-action pairs are further propagated to other unlabeled data generated from agents’ exploration. The proxy value function thus induces a policy that faithfully emulates human behaviors. Human- in-the-loop experiments show the generality and efficiency of our method. With minimal modification to existing reinforcement learning algorithms, our method can learn to solve continuous and discrete control tasks with various human control devices, including the challenging task of driving in Grand Theft Auto V. Demo video and code are available at: https://metadriverse.github.io/pvp.

----

## [0] PrObeD: Proactive Object Detection Wrapper

**Authors**: *Vishal Asnani, Abhinav Kumar, Suya You, Xiaoming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f5846131aa6a72d1df3bd6d43a4a960b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f5846131aa6a72d1df3bd6d43a4a960b-Abstract-Conference.html)

**Abstract**:

Previous research in $2D$ object detection focuses on various tasks, including detecting objects in generic and camouflaged images. These works are regarded as passive works for object detection as they take the input image as is. However, convergence to global minima is not guaranteed to be optimal in neural networks; therefore, we argue that the trained weights in the object detector are not optimal. To rectify this problem, we propose a wrapper based on proactive schemes, PrObeD, which enhances the performance of these object detectors by learning a signal. PrObeD consists of an encoder-decoder architecture, where the encoder network generates an image-dependent signal termed templates to encrypt the input images, and the decoder recovers this template from the encrypted images. We propose that learning the optimum template results in an object detector with an improved detection performance. The template acts as a mask to the input images to highlight semantics useful for the object detector. Finetuning the object detector with these encrypted images enhances the detection performance for both generic and camouflaged. Our experiments on MS-COCO, CAMO, COD$10$K, and NC$4$K datasets show improvement over different detectors after applying PrObeD. Our models/codes are available at https://github.com/vishal3477/Proactive-Object-Detection.

----

## [0] Waypoint Transformer: Reinforcement Learning via Supervised Learning with Intermediate Targets

**Authors**: *Anirudhan Badrinath, Yannis Flet-Berliac, Allen Nie, Emma Brunskill*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f58c24798220ba724fe05c0fa786227d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f58c24798220ba724fe05c0fa786227d-Abstract-Conference.html)

**Abstract**:

Despite the recent advancements in offline reinforcement learning via supervised learning (RvS) and the success of the decision transformer (DT) architecture in various domains, DTs have fallen short in several challenging benchmarks. The root cause of this underperformance lies in their inability to seamlessly connect segments of suboptimal trajectories. To overcome this limitation, we present a novel approach to enhance RvS methods by integrating intermediate targets. We introduce the Waypoint Transformer (WT), using an architecture that builds upon the DT framework and  conditioned on automatically-generated waypoints. The results show a significant increase in the final return compared to existing RvS methods, with performance on par or greater than existing state-of-the-art temporal difference learning-based methods. Additionally, the performance and stability improvements are largest in the most challenging environments and data configurations, including AntMaze Large Play/Diverse and Kitchen Mixed/Partial.

----

## [0] Should Under-parameterized Student Networks Copy or Average Teacher Weights?

**Authors**: *Berfin Simsek, Amire Bendjeddou, Wulfram Gerstner, Johanni Brea*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f5ccb3ab757131a93586ef61ec701533-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f5ccb3ab757131a93586ef61ec701533-Abstract-Conference.html)

**Abstract**:

Any continuous function $f^*$ can be approximated arbitrarily well by a neural network with sufficiently many neurons $k$. We consider the case when $f^*$ itself is a neural network with one hidden layer and $k$ neurons. Approximating $f^*$ with a neural network with $n< k$ neurons can thus be seen as fitting an under-parameterized "student" network with $n$ neurons to a "teacher" network with $k$ neurons. As the student has fewer neurons than the teacher, it is unclear, whether each of the $n$ student neurons should copy one of the teacher neurons or rather average a group of teacher neurons. For shallow neural networks with erf activation function and for the standard Gaussian input distribution, we prove that "copy-average" configurations are critical points if the teacher's incoming vectors are orthonormal and its outgoing weights are unitary. Moreover, the optimum among such configurations is reached when $n-1$ student neurons each copy one teacher neuron and the $n$-th student neuron averages the remaining $k-n+1$ teacher neurons. For the student network with $n=1$ neuron, we provide additionally a closed-form solution of the non-trivial critical point(s) for commonly used activation functions through solving an equivalent constrained optimization problem. Empirically, we find for the erf activation function that gradient flow converges either to the optimal copy-average critical point or to another point where each student neuron approximately copies a different teacher neuron. Finally, we find similar results for the ReLU activation function, suggesting that the optimal solution of underparameterized networks has a universal structure.

----

## [0] A Dataset for Analyzing Streaming Media Performance over HTTP/3 Browsers

**Authors**: *Sapna Chaudhary, Mukulika Maity, Sandip Chakraborty, Naval Shukla*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f5da8ac52cf8857157c63c4803b6690b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f5da8ac52cf8857157c63c4803b6690b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

HTTP/3 is a new application layer protocol supported by most browsers. It uses QUIC as an underlying transport protocol. QUIC provides multiple benefits, like faster connection establishment, reduced latency, and improved connection migration. Hence, most popular browsers like Chrome/Chromium, Microsoft Edge, Apple Safari, and Mozilla Firefox have started supporting it. In this paper, we present an HTTP/3-supported browser dataset collection tool named H3B. It collects the application and network-level logs during YouTube streaming. We consider YouTube, as it  the most popular video streaming application supporting QUIC. Using this tool, we collected a dataset of over 5936 YouTube sessions covering 5464 hours of streaming over 5 different geographical locations and 5 different bandwidth patterns. We believe our tool and as well as the dataset could be used in multiple applications such as a better configuration of application/transport protocols based on the network conditions, intelligent integration of network and application, predicting YouTube's QoE etc. We analyze the dataset and observe that during an HTTP/3 streaming not all requests are served by HTTP/3. Instead whenever the network condition is not favorable the browser chooses to fallback, and the application requests are transmitted using HTTP/2 over the old-standing transport protocol TCP. We observe that such switching of protocols impacts the performance of video streaming applications.

----

## [0] Learning from Rich Semantics and Coarse Locations for Long-tailed Object Detection

**Authors**: *Lingchen Meng, Xiyang Dai, Jianwei Yang, Dongdong Chen, Yinpeng Chen, Mengchen Liu, Yi-Ling Chen, Zuxuan Wu, Lu Yuan, Yu-Gang Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f5fcd88d3deb97bb62559208cfa0ab62-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f5fcd88d3deb97bb62559208cfa0ab62-Abstract-Conference.html)

**Abstract**:

Long-tailed object detection (LTOD) aims to handle the extreme data imbalance in real-world datasets, where many tail classes have scarce instances. One popular strategy is to explore extra data with image-level labels, yet it produces limited results due to (1) semantic ambiguity---an image-level label only captures a salient part of the image, ignoring the remaining rich semantics within the image; and (2) location sensitivity---the label highly depends on the locations and crops of the original image, which may change after data transformations like random cropping.To remedy this, we propose RichSem, a simple but effective method, which is robust to learn rich semantics from coarse locations without the need of accurate bounding boxes. RichSem leverages rich semantics from images, which are then served as additional ``soft supervision'' for training detectors. Specifically, we add a semantic branch to our detector to learn these soft semantics and enhance feature representations for long-tailed object detection. The semantic branch is only used for training and is removed during inference. RichSem achieves consistent improvements on both overall and rare-category of LVIS under different backbones and detectors. Our method achieves state-of-the-art performance without requiring complex training and testing procedures. Moreover, we show the effectiveness of our method on other long-tailed datasets with additional experiments.

----

## [0] StoryBench: A Multifaceted Benchmark for Continuous Story Visualization

**Authors**: *Emanuele Bugliarello, H. Hernan Moraldo, Ruben Villegas, Mohammad Babaeizadeh, Mohammad Taghi Saffar, Han Zhang, Dumitru Erhan, Vittorio Ferrari, Pieter-Jan Kindermans, Paul Voigtlaender*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f63f5fbed1a4ef08c857c5f377b5d33a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f63f5fbed1a4ef08c857c5f377b5d33a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Generating video stories from text prompts is a complex task. In addition to having high visual quality, videos need to realistically adhere to a sequence of text prompts whilst being consistent throughout the frames. Creating a benchmark for video generation requires data annotated over time, which contrasts with the single caption used often in video datasets. To fill this gap, we collect comprehensive human annotations on three existing datasets, and introduce StoryBench: a new, challenging multi-task benchmark to reliably evaluate forthcoming text-to-video models. Our benchmark includes three video generation tasks of increasing difficulty: action execution, where the next action must be generated starting from a conditioning video; story continuation, where a sequence of actions must be executed starting from a conditioning video; and story generation, where a video must be generated from only text prompts. We evaluate small yet strong text-to-video baselines, and show the benefits of training on story-like data algorithmically generated from existing video captions. Finally, we establish guidelines for human evaluation of video stories, and reaffirm the need of better automatic metrics for video generation. StoryBench aims at encouraging future research efforts in this exciting new area.

----

## [0] DiffInfinite: Large Mask-Image Synthesis via Parallel Random Patch Diffusion in Histopathology

**Authors**: *Marco Aversa, Gabriel Nobis, Miriam Hägele, Kai Standvoss, Mihaela Chirica, Roderick Murray-Smith, Ahmed M. Alaa, Lukas Ruff, Daniela Ivanova, Wojciech Samek, Frederick Klauschen, Bruno Sanguinetti, Luis Oala*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f64927f5de00c47899e6e58c731966b6-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f64927f5de00c47899e6e58c731966b6-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present DiffInfinite, a hierarchical diffusion model that generates arbitrarily large histological images while preserving long-range correlation structural information. Our approach first generates synthetic segmentation masks, subsequently used as conditions for the high-fidelity generative diffusion process. The proposed sampling method can be scaled up to any desired image size while only requiring small patches for fast training. Moreover, it can be parallelized more efficiently than previous large-content generation methods while avoiding tiling artifacts. The training leverages classifier-free guidance to augment a small, sparsely annotated dataset with unlabelled data. Our method alleviates unique challenges in histopathological imaging practice: large-scale information, costly manual annotation, and protective data handling. The biological plausibility of DiffInfinite data is evaluated in a survey by ten experienced pathologists as well as a downstream classification and segmentation task. Samples from the model score strongly on anti-copying metrics which is relevant for the protection of patient data.

----

## [0] Benchmarking Foundation Models with Language-Model-as-an-Examiner

**Authors**: *Yushi Bai, Jiahao Ying, Yixin Cao, Xin Lv, Yuze He, Xiaozhi Wang, Jifan Yu, Kaisheng Zeng, Yijia Xiao, Haozhe Lyu, Jiayin Zhang, Juanzi Li, Lei Hou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f64e55d03e2fe61aa4114e49cb654acb-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Numerous benchmarks have been established to assess the performance of foundation models on open-ended question answering, which serves as a comprehensive test of a model's ability to understand and generate language in a manner similar to humans.Most of these works focus on proposing new datasets, however, we see two main issues within previous benchmarking pipelines, namely testing leakage and evaluation automation. In this paper, we propose a novel benchmarking framework, Language-Model-as-an-Examiner, where the LM serves as a knowledgeable examiner that formulates questions based on its knowledge and evaluates responses in a reference-free manner. Our framework allows for effortless extensibility as various LMs can be adopted as the examiner, and the questions can be constantly updated given more diverse trigger topics. For a more comprehensive and equitable evaluation, we devise three strategies: (1) We instruct the LM examiner to generate questions across a multitude of domains to probe for a broad acquisition, and raise follow-up questions to engage in a more in-depth assessment. (2) Upon evaluation, the examiner combines both scoring and ranking measurements, providing a reliable result as it aligns closely with human annotations. (3) We additionally propose a decentralized Peer-examination method to address the biases in a single examiner. Our data and benchmarking results are available at: http://lmexam.xlore.cn.

----

## [0] Granger Components Analysis: Unsupervised learning of latent temporal dependencies

**Authors**: *Jacek Dmochowski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f66340d6f28dae6aab0176892c9065e7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f66340d6f28dae6aab0176892c9065e7-Abstract-Conference.html)

**Abstract**:

A new technique for unsupervised learning of time series data based on the notion of Granger causality is presented. The technique learns pairs of projections of a multivariate data set such that the resulting components -- "driving" and "driven" -- maximize the strength of the Granger causality between the latent time series (how strongly the past of the driving signal predicts the present of the driven signal). A coordinate descent algorithm that learns pairs of coefficient vectors in an alternating fashion is developed and shown to blindly identify the underlying sources (up to scale) on simulated vector autoregressive (VAR) data. The technique is tested on scalp electroencephalography (EEG) data from a motor imagery experiment where the resulting components lateralize with the side of the cued hand, and also on functional magnetic resonance imaging (fMRI) data, where the recovered components express previously reported resting-state networks.

----

## [0] Monte Carlo Tree Search with Boltzmann Exploration

**Authors**: *Michael Painter, Mohamed Baioumy, Nick Hawes, Bruno Lacerda*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f670ef96387d9a5a8a51e2ed80cb148d-Abstract-Conference.html)

**Abstract**:

Monte-Carlo Tree Search (MCTS) methods, such as Upper Confidence Bound applied to Trees (UCT), are instrumental to automated planning techniques. However, UCT can be slow to explore an optimal action when it initially appears inferior to other actions. Maximum ENtropy Tree-Search (MENTS) incorporates the maximum entropy principle into an MCTS approach, utilising Boltzmann policies to sample actions, naturally encouraging more exploration. In this paper, we highlight a major limitation of MENTS: optimal actions for the maximum entropy objective do not necessarily correspond to optimal actions for the original objective. We introduce two algorithms, Boltzmann Tree Search (BTS) and Decaying ENtropy Tree-Search (DENTS), that address these limitations and preserve the benefits of Boltzmann policies, such as allowing actions to be sampled faster by using the Alias method. Our empirical analysis shows that our algorithms show consistent high performance across several benchmark domains, including the game of Go.

----

## [0] False Discovery Proportion control for aggregated Knockoffs

**Authors**: *Alexandre Blain, Bertrand Thirion, Olivier Grisel, Pierre Neuvial*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6712d5191d2501dfc7024389f7bfcdd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6712d5191d2501dfc7024389f7bfcdd-Abstract-Conference.html)

**Abstract**:

Controlled variable selection is an important analytical step in various scientific fields, such as brain imaging or genomics. In these high-dimensional data settings, considering too many variables leads to poor models and high costs, hence the need for statistical guarantees on false positives. Knockoffs are a popular statistical tool for conditional variable selection in high dimension. However, they control for the expected proportion of false discoveries (FDR) and not the actual proportion of false discoveries (FDP). We present a new method, KOPI, that controls the proportion of false discoveries for Knockoff-based inference. The proposed method also relies on a new type of aggregation to address the undesirable randomness associated with classical Knockoff inference. We demonstrate FDP control and substantial power gains over existing Knockoff-based methods in various simulation settings and achieve good sensitivity/specificity tradeoffs on brain imaging data.

----

## [0] Interpretability at Scale: Identifying Causal Mechanisms in Alpaca

**Authors**: *Zhengxuan Wu, Atticus Geiger, Thomas Icard, Christopher Potts, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6a8b109d4d4fd64c75e94aaf85d9697-Abstract-Conference.html)

**Abstract**:

Obtaining human-interpretable explanations of large, general-purpose language models is an urgent goal for AI safety. However, it is just as important that our interpretability methods are faithful to the causal dynamics underlying model behavior and able to robustly generalize to unseen inputs. Distributed Alignment Search (DAS) is a powerful gradient descent method grounded in a theory of causal abstraction that uncovered perfect alignments between interpretable symbolic algorithms and small deep learning models fine-tuned for specific tasks. In the present paper, we scale DAS significantly by replacing the remaining brute-force search steps with learned parameters -- an approach we call Boundless DAS. This enables us to efficiently search for interpretable causal structure in large language models while they follow instructions. We apply Boundless DAS to the Alpaca model (7B parameters), which, off the shelf, solves a simple numerical reasoning problem. With Boundless DAS, we discover that Alpaca does this by implementing a causal model with two interpretable boolean variables. Furthermore, we find that the alignment of neural representations with these variables is robust to changes in inputs and instructions. These findings mark a first step toward deeply understanding the inner-workings of our largest and most widely deployed language models.

----

## [0] Large Language Models Are Semi-Parametric Reinforcement Learning Agents

**Authors**: *Danyang Zhang, Lu Chen, Situo Zhang, Hongshen Xu, Zihan Zhao, Kai Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6b22ac37beb5da61efd4882082c9ecd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6b22ac37beb5da61efd4882082c9ecd-Abstract-Conference.html)

**Abstract**:

Inspired by the insights in cognitive science with respect to human memory and reasoning mechanism, a novel evolvable LLM-based (Large Language Model) agent framework is proposed as Rememberer. By equipping the LLM with a long-term experience memory, Rememberer is capable of exploiting the experiences from the past episodes even for different task goals, which excels an LLM-based agent with fixed exemplars or equipped with a transient working memory. We further introduce Reinforcement Learning with Experience Memory (RLEM) to update the memory. Thus, the whole system can learn from the experiences of both success and failure, and evolve its capability without fine-tuning the parameters of the LLM. In this way, the proposed Rememberer constitutes a semi-parametric RL agent. Extensive experiments are conducted on two RL task sets to evaluate the proposed framework. The average results with different initialization and training sets exceed the prior SOTA by 4% and 2% for the success rate on two task sets and demonstrate the superiority and robustness of Rememberer.

----

## [0] BIOT: Biosignal Transformer for Cross-data Learning in the Wild

**Authors**: *Chaoqi Yang, M. Brandon Westover, Jimeng Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)

**Abstract**:

Biological signals, such as electroencephalograms (EEG), play a crucial role in numerous clinical applications, exhibiting diverse data formats and quality profiles. Current deep learning models for biosignals (based on CNN, RNN, and Transformers) are typically specialized for specific datasets and clinical settings, limiting their broader applicability. This paper explores the development of a flexible biosignal encoder architecture that can enable pre-training on multiple datasets and fine-tuned on downstream biosignal tasks with different formats.To overcome the unique challenges associated with biosignals of various formats, such as mismatched channels, variable sample lengths, and prevalent missing val- ues, we propose Biosignal Transformer (BIOT). The proposed BIOT model can enable cross-data learning with mismatched channels, variable lengths, and missing values by tokenizing different biosignals into unified "sentences" structure. Specifically, we tokenize each channel separately into fixed-length segments containing local signal features and then rearrange the segments to form a long "sentence". Channel embeddings and relative position embeddings are added to each segment (viewed as "token") to preserve spatio-temporal features.The BIOT model is versatile and applicable to various biosignal learning settings across different datasets, including joint pre-training for larger models. Comprehensive evaluations on EEG, electrocardiogram (ECG), and human activity sensory signals demonstrate that BIOT outperforms robust baselines in common settings and facilitates learning across multiple datasets with different formats. Using CHB-MIT seizure detection task as an example, our vanilla BIOT model shows 3% improvement over baselines in balanced accuracy, and the pre-trained BIOT models (optimized from other data sources) can further bring up to 4% improvements. Our repository is public at https://github.com/ycq091044/BIOT.

----

## [0] Interactive Multi-fidelity Learning for Cost-effective Adaptation of Language Model with Sparse Human Supervision

**Authors**: *Jiaxin Zhang, Zhuohang Li, Kamalika Das, Kumar Sricharan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6c1843f11d34312b11ec5ff9a10c5a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6c1843f11d34312b11ec5ff9a10c5a6-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have demonstrated remarkable capabilities in various tasks. However, their suitability for domain-specific tasks, is limited due to their immense scale at deployment, susceptibility to misinformation, and more importantly, high data annotation costs. We propose a novel Interactive Multi-Fidelity Learning (IMFL) framework for cost-effective development of small domain-specific LMs under limited annotation budgets. Our approach formulates the domain-specific fine-tuning process as a multi-fidelity learning problem, focusing on identifying the optimal acquisition strategy that balances between low-fidelity automatic LLM annotations and high-fidelity human annotations to maximize model performance. We further propose an exploration-exploitation query strategy that enhances annotation diversity and informativeness, incorporating two innovative designs: 1) prompt retrieval that selects in-context examples from human-annotated samples to improve LLM annotation, and 2) variable batch size that controls the order for choosing each fidelity to facilitate knowledge distillation, ultimately enhancing annotation quality. Extensive experiments on financial and medical tasks demonstrate that IMFL achieves superior performance compared with single fidelity annotations. Given a limited budget of human annotation, IMFL significantly outperforms the $\bf 3\times$ human annotation baselines in all four tasks and achieves very close performance as $\bf 5\times$ human annotation on two of the tasks. These promising results suggest that the high human annotation costs in domain-specific tasks can be significantly reduced by employing IMFL, which utilizes fewer human annotations, supplemented with cheaper and faster LLM (e.g., GPT-3.5) annotations to achieve comparable performance.

----

## [0] OceanBench: The Sea Surface Height Edition

**Authors**: *J. Emmanuel Johnson, Quentin Febvre, Anastasiia Gorbunova, Sammy Metref, Maxime Ballarotta, Julien Le Sommer, Ronan Fablet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f6ccbf94fa57c2ae372ece91b537574d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f6ccbf94fa57c2ae372ece91b537574d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The ocean is a crucial component of the Earth's system. It profoundly influences human activities and plays a critical role in climate regulation. Our understanding has significantly improved over the last decades with the advent of satellite remote sensing data, allowing us to capture essential sea surface quantities over the globe, e.g., sea surface height (SSH). Despite their ever-increasing abundance, ocean satellite data presents challenges for information extraction due to their sparsity and irregular sampling, signal complexity, and noise. Machine learning (ML) techniques have demonstrated their capabilities in dealing with large-scale, complex signals. Therefore we see an opportunity for these ML models to harness the full extent of the information contained in ocean satellite data. However, data representation and relevant evaluation metrics can be the defining factors when determining the success of applied ML. The processing steps from the raw observation data to a ML-ready state and from model outputs to interpretable quantities require domain expertise, which can be a significant barrier to entry for ML researchers. In addition, imposing fixed processing steps, like committing to specific variables, regions, and geometries, will narrow the scope of ML models and their potential impact on real-world applications. OceanBench is a unifying framework that provides standardized processing steps that comply with domain-expert standards. It is designed with a flexible and pedagogical abstraction: it a) provides plug-and-play data and pre-configured pipelines for ML researchers to benchmark their models w.r.t. ML and domain-related baselines and b) provides a transparent and configurable framework for researchers to customize and extend the pipeline for their tasks. In this work, we demonstrate the OceanBench framework through a first edition dedicated to SSH interpolation challenges. We provide datasets and ML-ready benchmarking pipelines for the long-standing problem of interpolating observations from simulated ocean satellite data, multi-modal and multi-sensor fusion issues, and transfer-learning to real ocean satellite observations. The  OceanBench framework is available at https://github.com/jejjohnson/oceanbench and the dataset registry is available at https://github.com/quentinf00/oceanbench-data-registry.

----

## [0] Amortized Reparametrization: Efficient and Scalable Variational Inference for Latent SDEs

**Authors**: *Kevin Course, Prasanth B. Nair*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f72d4fdfd5eb425cd81df9fe6272a533-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f72d4fdfd5eb425cd81df9fe6272a533-Abstract-Conference.html)

**Abstract**:

We consider the problem of inferring latent stochastic differential equations (SDEs) with a time and memory cost that scales independently with the amount of data, the total length of the time series, and the stiffness of the approximate differential equations. This is in stark contrast to typical methods for inferring latent differential equations which, despite their constant memory cost, have a time complexity that is heavily dependent on the stiffness of the approximate differential equation. We achieve this computational advancement by removing the need to solve differential equations when approximating gradients using a novel amortization strategy coupled with a recently derived reparametrization of expectations under linear SDEs. We show that, in practice, this allows us to achieve similar performance to methods based on adjoint sensitivities with more than an order of magnitude fewer evaluations of the model in training.

----

## [0] Boundary Guided Learning-Free Semantic Control with Diffusion Models

**Authors**: *Ye Zhu, Yu Wu, Zhiwei Deng, Olga Russakovsky, Yan Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f737da5ea0e122870fad209509f87d5b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f737da5ea0e122870fad209509f87d5b-Abstract-Conference.html)

**Abstract**:

Applying pre-trained generative denoising diffusion models (DDMs) for downstream tasks such as image semantic editing usually requires either fine-tuning DDMs or learning auxiliary editing networks in the existing literature. In this work, we present our BoundaryDiffusion method for efficient, effective and light-weight semantic control with frozen pre-trained DDMs, without learning any extra networks. As one of the first learning-free diffusion editing works, we start by seeking a more comprehensive understanding of the intermediate high-dimensional latent spaces by theoretically and empirically analyzing their probabilistic and geometric behaviors in the Markov chain. We then propose to further explore the critical step in the denoising trajectory that characterizes the convergence of a pre-trained DDM and introduce an automatic search method. Last but not least, in contrast to the conventional understanding that DDMs have relatively poor semantic behaviors (in generic latent spaces), we prove that the critical latent space we found already forms semantic subspace boundaries at the generic level in unconditional DDMs, which allows us to do controllable manipulation by guiding the denoising trajectory towards the targeted boundary via a single-step operation. We conduct extensive experiments on multiple DPMs architectures (DDPM, iDDPM) and datasets (CelebA, CelebA-HQ, LSUN-church, LSUN-bedroom, AFHQ-dog) with different resolutions (64, 256), achieving superior or state-of-the-art performance in various task scenarios (image semantic editing, text-based editing, unconditional semantic control) to demonstrate the effectiveness.

----

## [0] Kiki or Bouba? Sound Symbolism in Vision-and-Language Models

**Authors**: *Morris Alper, Hadar Averbuch-Elor*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f74054328beeb0c21a9b8e99da557f5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f74054328beeb0c21a9b8e99da557f5a-Abstract-Conference.html)

**Abstract**:

Although the mapping between sound and meaning in human language is assumed to be largely arbitrary, research in cognitive science has shown that there are non-trivial correlations between particular sounds and meanings across languages and demographic groups, a phenomenon known as sound symbolism. Among the many dimensions of meaning, sound symbolism is particularly salient and well-demonstrated with regards to cross-modal associations between language and the visual domain. In this work, we address the question of whether sound symbolism is reflected in vision-and-language models such as CLIP and Stable Diffusion. Using zero-shot knowledge probing to investigate the inherent knowledge of these models, we find strong evidence that they do show this pattern, paralleling the well-known kiki-bouba effect in psycholinguistics. Our work provides a novel method for demonstrating sound symbolism and understanding its nature using computational tools. Our code will be made publicly available.

----

## [0] MoCa: Measuring Human-Language Model Alignment on Causal and Moral Judgment Tasks

**Authors**: *Allen Nie, Yuhui Zhang, Atharva Amdekar, Chris Piech, Tatsunori B. Hashimoto, Tobias Gerstenberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f751c6f8bfb52c60f43942896fe65904-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f751c6f8bfb52c60f43942896fe65904-Abstract-Conference.html)

**Abstract**:

Human commonsense understanding of the physical and social world is organized around intuitive theories. These theories support making causal and moral judgments. When something bad happens, we naturally ask: who did what, and why? A rich literature in cognitive science has studied people's causal and moral intuitions. This work has revealed a number of factors that systematically influence people's judgments, such as the violation of norms and whether the harm is avoidable or inevitable. We collected a dataset of stories from 24 cognitive science papers and developed a system to annotate each story with the factors they investigated. Using this dataset, we test whether large language models (LLMs) make causal and moral judgments about text-based scenarios that align with those of human participants. On the aggregate level, alignment has improved with more recent LLMs. However, using statistical analyses, we find that LLMs weigh the different factors quite differently from human participants. These results show how curated, challenge datasets combined with insights from cognitive science can help us go beyond comparisons based merely on aggregate metrics: we uncover LLMs implicit tendencies and show to what extent these align with human intuitions.

----

## [0] Collapsed Inference for Bayesian Deep Learning

**Authors**: *Zhe Zeng, Guy Van den Broeck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f763f7c9a6599e14b07add5937d8189c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f763f7c9a6599e14b07add5937d8189c-Abstract-Conference.html)

**Abstract**:

Bayesian neural networks (BNNs) provide a formalism to quantify and calibrate uncertainty in deep learning. Current inference approaches for BNNs often resort to few-sample estimation for scalability, which can harm predictive performance, while its alternatives tend to be computationally prohibitively expensive. We tackle this challenge by revealing a previously unseen connection between inference on BNNs and volume computation problems. With this observation, we introduce a novel collapsed inference scheme that performs Bayesian model averaging using collapsed samples. It improves over a Monte-Carlo sample by limiting sampling to a subset of the network weights while pairing it with some closed-form conditional distribution over the rest. A collapsed sample represents uncountably many models drawn from the approximate posterior and thus yields higher sample efficiency. Further, we show that the marginalization of a collapsed sample can be solved analytically and efficiently despite the non-linearity of neural networks by leveraging existing volume computation solvers. Our proposed use of collapsed samples achieves a balance between scalability and accuracy. On various regression and classification tasks, our collapsed Bayesian deep learning approach demonstrates significant improvements over existing methods and sets a new state of the art in terms of uncertainty estimation as well as predictive performance.

----

## [0] Contextual Stochastic Bilevel Optimization

**Authors**: *Yifan Hu, Jie Wang, Yao Xie, Andreas Krause, Daniel Kuhn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f77d9409647c096789067c09455858a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f77d9409647c096789067c09455858a2-Abstract-Conference.html)

**Abstract**:

We introduce contextual stochastic bilevel optimization (CSBO) -- a stochastic bilevel optimization framework with the lower-level problem minimizing an expectation conditioned on some contextual information and the upper-level decision variable. This framework extends classical stochastic bilevel optimization when the lower-level decision maker responds optimally not only to the decision of the upper-level decision maker but also to some side information and when there are multiple or even infinite many followers. It captures important applications such as meta-learning, personalized federated learning, end-to-end learning, and Wasserstein distributionally robust optimization with side information (WDRO-SI). Due to the presence of contextual information, existing single-loop methods for classical stochastic bilevel optimization are unable to converge. To overcome this challenge, we introduce an efficient double-loop gradient method based on the Multilevel Monte-Carlo (MLMC) technique and establish its sample and computational complexities. When specialized to stochastic nonconvex optimization, our method matches  existing lower bounds. For meta-learning, the complexity of our method does not depend on the number of tasks. Numerical experiments further validate our theoretical results.

----

## [0] Learning Invariant Molecular Representation in Latent Discrete Space

**Authors**: *Xiang Zhuang, Qiang Zhang, Keyan Ding, Yatao Bian, Xiao Wang, Jingsong Lv, Hongyang Chen, Huajun Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f780a86b7145988ac219d49d8e37a58f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f780a86b7145988ac219d49d8e37a58f-Abstract-Conference.html)

**Abstract**:

Molecular representation learning lays the foundation for drug discovery. However, existing methods suffer from poor out-of-distribution (OOD) generalization, particularly when data for training and testing originate from different environments. To address this issue, we propose a new framework for learning molecular representations that exhibit invariance and robustness against distribution shifts. Specifically, we propose a strategy called  ``first-encoding-then-separation'' to identify invariant molecule features in the latent space, which deviates from conventional practices. Prior to the separation step, we introduce a residual vector quantization module that mitigates the over-fitting to training data distributions while preserving the expressivity of encoders. Furthermore, we design a task-agnostic self-supervised learning objective to encourage precise invariance identification, which enables our method widely applicable to a variety of tasks, such as regression and multi-label classification. Extensive experiments on 18 real-world molecular datasets demonstrate that our model achieves stronger generalization against state-of-the-art baselines in the presence of various distribution shifts.  Our code is available at https://github.com/HICAI-ZJU/iMoLD.

----

## [0] Accelerating Motion Planning via Optimal Transport

**Authors**: *An T. Le, Georgia Chalvatzaki, Armin Biess, Jan Peters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7a94134f1c726796c6f81fb946e489d-Abstract-Conference.html)

**Abstract**:

Motion planning is still an open problem for many disciplines, e.g., robotics, autonomous driving, due to their need for high computational resources that hinder real-time, efficient decision-making. A class of methods striving to provide smooth solutions is gradient-based trajectory optimization. However, those methods usually suffer from bad local minima, while for many settings, they may be inapplicable due to the absence of easy-to-access gradients of the optimization objectives. In response to these issues, we introduce Motion Planning via Optimal Transport (MPOT)---a \textit{gradient-free} method that optimizes a batch of smooth trajectories over highly nonlinear costs, even for high-dimensional tasks, while imposing smoothness through a Gaussian Process dynamics prior via the planning-as-inference perspective. To facilitate batch trajectory optimization, we introduce an original zero-order and highly-parallelizable update rule----the Sinkhorn Step, which uses the regular polytope family for its search directions. Each regular polytope, centered on trajectory waypoints, serves as a local cost-probing neighborhood, acting as a \textit{trust region} where the Sinkhorn Step ``transports'' local waypoints toward low-cost regions. We theoretically show that Sinkhorn Step guides the optimizing parameters toward local minima regions of non-convex objective functions. We then show the efficiency of MPOT in a range of problems from low-dimensional point-mass navigation to high-dimensional whole-body robot motion planning, evincing its superiority compared to popular motion planners, paving the way for new applications of optimal transport in motion planning.

----

## [0] M5HisDoc: A Large-scale Multi-style Chinese Historical Document Analysis Benchmark

**Authors**: *Yongxin Shi, Chongyu Liu, Dezhi Peng, Cheng Jian, Jiarong Huang, Lianwen Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7b424d242cc6bb7708cff241367334d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7b424d242cc6bb7708cff241367334d-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recognizing and organizing text in correct reading order plays a crucial role in historical document analysis and preservation. While existing methods have shown promising performance, they often struggle with challenges such as diverse layouts, low image quality, style variations, and distortions. This is primarily due to the lack of consideration for these issues in the current benchmarks, which hinders the development and evaluation of historical document analysis and recognition (HDAR) methods in complex real-world scenarios. To address this gap, this paper introduces a complex multi-style Chinese historical document analysis benchmark, named M5HisDoc. The M5 indicates five properties of style, ie., Multiple layouts, Multiple document types, Multiple calligraphy styles, Multiple backgrounds, and Multiple challenges. The M5HisDoc dataset consists of two subsets, M5HisDoc-R (Regular) and M5HisDoc-H (Hard). The M5HisDoc-R subset comprises 4,000 historical document images. To ensure high-quality annotations, we meticulously perform manual annotation and triple-checking. To replicate real-world conditions for historical document analysis applications, we incorporate image rotation, distortion, and resolution reduction into M5HisDoc-R subset to form a new challenging subset named M5HisDoc-H, which contains the same number of images as M5HisDoc-R. The dataset exhibits diverse styles, significant scale variations, dense texts, and an extensive character set. We conduct benchmarking experiments on five tasks: text line detection, text line recognition, character detection, character recognition, and reading order prediction. We also conduct cross-validation with other benchmarks. Experimental results demonstrate that the M5HisDoc dataset can offer new challenges and great opportunities for future research in this field, thereby providing deep insights into the solution for HDAR. The dataset is available at https://github.com/HCIILAB/M5HisDoc.

----

## [0] CWCL: Cross-Modal Transfer with Continuously Weighted Contrastive Loss

**Authors**: *Rakshith Sharma Srinivasa, Jaejin Cho, Chouchang Yang, Yashas Malur Saidutta, Ching Hua Lee, Yilin Shen, Hongxia Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7b77476d89d5fb58aeb77691d2f40f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7b77476d89d5fb58aeb77691d2f40f5-Abstract-Conference.html)

**Abstract**:

This paper considers contrastive training for cross-modal 0-shot transfer wherein a pre-trained model in one modality is used for representation learning in another domain using pairwise data. The learnt models in the latter domain can then be used for a diverse set of tasks in a 0-shot way, similar to Contrastive Language-Image Pre-training (CLIP) and Locked-image Tuning (LiT) that have recently gained considerable attention. Classical contrastive training employs sets of positive and negative examples to align similar and repel dissimilar training data samples. However, similarity amongst training examples has a more continuous nature, thus calling for a more `non-binary' treatment. To address this, we propose a new contrastive loss function called Continuously Weighted Contrastive Loss (CWCL) that employs a continuous measure of similarity. With CWCL, we seek to transfer the structure of the embedding space from one modality to another. Owing to the continuous nature of similarity in the proposed loss function, these models outperform existing methods for 0-shot transfer across multiple models, datasets and modalities. By using publicly available datasets, we achieve 5-8% (absolute) improvement over previous state-of-the-art methods in 0-shot image classification and 20-30% (absolute) improvement in 0-shot speech-to-intent classification and keyword classification.

----

## [0] Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning

**Authors**: *Zikang Tian, Ruizhi Chen, Xing Hu, Ling Li, Rui Zhang, Fan Wu, Shaohui Peng, Jiaming Guo, Zidong Du, Qi Guo, Yunji Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7d3cef7ff579f2f903c8f458e730cae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7d3cef7ff579f2f903c8f458e730cae-Abstract-Conference.html)

**Abstract**:

In recent years, Multi-Agent Reinforcement Learning (MARL) techniques have made significant strides in achieving high asymptotic performance in single task. However, there has been limited exploration of model transferability across tasks. Training a model from scratch for each task can be time-consuming and expensive, especially for large-scale Multi-Agent Systems. Therefore, it is crucial to develop methods for generalizing the model across tasks. Considering that there exist task-independent subtasks across MARL tasks, a model that can decompose such subtasks from the source task could generalize to target tasks. However, ensuring true task-independence of subtasks poses a challenge. In this paper, we propose to \textbf{d}ecompose a \textbf{t}ask in\textbf{to} a series of \textbf{g}eneralizable \textbf{s}ubtasks (DT2GS), a novel framework that addresses this challenge by utilizing a scalable subtask encoder and an adaptive subtask semantic module. We show that these components endow subtasks with two properties critical for task-independence: avoiding overfitting to the source task and maintaining consistent yet scalable semantics across tasks. Empirical results demonstrate that DT2GS possesses sound zero-shot generalization capability across tasks, exhibits sufficient transferability, and outperforms existing methods in both multi-task and single-task problems.

----

## [0] The Equivalence of Dynamic and Strategic Stability under Regularized Learning in Games

**Authors**: *Victor Boone, Panayotis Mertikopoulos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7e8bc4c853e3e58bc487e213c79c587-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7e8bc4c853e3e58bc487e213c79c587-Abstract-Conference.html)

**Abstract**:

In this paper, we examine the long-run behavior of regularized, no-regret learning in finite N-player games. A well-known result in the field states that the empirical frequencies of play under no-regret learning converge to the game’s set of coarse correlated equilibria; however, our understanding of how the players' actual strategies evolve over time is much more limited – and, in many cases, non-existent. This issue is exacerbated further by a series of recent results showing that only strict Nash equilibria are stable and attracting under regularized learning, thus making the relation between learning and pointwise solution concepts particularly elusive. In lieu of this, we take a more general approach and instead seek to characterize the setwise rationality properties of the players' day-to-day trajectory of play. To do so, we focus on one of the most stringent criteria of setwise strategic stability, namely that any unilateral deviation from the set in question incurs a cost to the deviator – a property known as closedness under better replies (club). In so doing, we obtain a remarkable equivalence between strategic and dynamic stability: a product of pure strategies is closed under better replies if and only if its span is stable and attracting under regularized learning. In addition, we estimate the rate of convergence to such sets, and we show that methods based on entropic regularization (like the exponential weights algorithm) converge at a geometric rate, while projection-based methods converge within a finite number of iterations, even with bandit, payoff-based feedback.

----

## [0] HubRouter: Learning Global Routing via Hub Generation and Pin-hub Connection

**Authors**: *Xingbo Du, Chonghua Wang, Ruizhe Zhong, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7f98663c516fceb582354ee2d9d274d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7f98663c516fceb582354ee2d9d274d-Abstract-Conference.html)

**Abstract**:

Global Routing (GR) is a core yet time-consuming task in VLSI systems. It recently attracted efforts from the machine learning community, especially generative models, but they suffer from the non-connectivity of generated routes. We argue that the inherent non-connectivity can harm the advantage of its one-shot generation and has to be post-processed by traditional approaches. Thus, we propose a novel definition, called hub, which represents the key point in the route. Equipped with hubs, global routing is transferred from a pin-pin connection problem to a hub-pin connection problem. Specifically, to generate definitely-connected routes, this paper proposes a two-phase learning scheme named HubRouter, which includes 1) hub-generation phase: A condition-guided hub generator using deep generative models; 2) pin-hub-connection phase: An RSMT construction module that connects the hubs and pins using an actor-critic model. In the first phase, we incorporate typical generative models into a multi-task learning framework to perform hub generation and address the impact of sensitive noise points with stripe mask learning. During the second phase, HubRouter employs an actor-critic model to finish the routing, which is efficient and has very slight errors. Experiments on simulated and real-world global routing benchmarks are performed to show our approach's efficiency, particularly HubRouter outperforms the state-of-the-art generative global routing methods in wirelength, overflow, and running time. Moreover, HubRouter also shows strength in other applications, such as RSMT construction and interactive path replanning.

----

## [0] L2-Uniform Stability of Randomized Learning Algorithms: Sharper Generalization Bounds and Confidence Boosting

**Authors**: *Xiaotong Yuan, Ping Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7fc38fdd95fd146a471791b93ff9f12-Abstract-Conference.html)

**Abstract**:

Exponential generalization bounds with near-optimal rates have recently been established for uniformly stable algorithms~\citep{feldman2019high,bousquet2020sharper}. We seek to extend these best known high probability bounds from deterministic learning algorithms to the regime of randomized learning. One simple approach for achieving this goal is to define the stability for the expectation over the algorithm's randomness, which may result in sharper parameter but only leads to guarantees regarding the on-average generalization error. Another natural option is to consider the stability conditioned on the algorithm's randomness, which is way more stringent but may lead to generalization with high probability jointly over the randomness of sample and algorithm. The present paper addresses such a tension between these two alternatives and makes progress towards relaxing it inside a classic framework of confidence-boosting. To this end, we first introduce a novel concept of $L_2$-uniform stability that holds uniformly over data but in second-moment over the algorithm's randomness. Then as a core contribution of this work, we prove a strong exponential bound on the first-moment of generalization error under the notion of $L_2$-uniform stability. As an interesting consequence of the bound, we show that a bagging-based meta algorithm leads to near-optimal generalization with high probability jointly over the randomness of data and algorithm. We further substantialize these generic results to stochastic gradient descent (SGD) to derive sharper exponential bounds for convex or non-convex optimization with natural time-decaying learning rates, which have not been possible to prove with the existing stability-based generalization guarantees.

----

## [0] Neural Sampling in Hierarchical Exponential-family Energy-based Models

**Authors**: *Xingsi Dong, Si Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f7fdebf712db182eddaee2eb02af91e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f7fdebf712db182eddaee2eb02af91e0-Abstract-Conference.html)

**Abstract**:

Bayesian brain theory suggests that the brain employs generative models to understand the external world. The sampling-based perspective posits that the brain infers the posterior distribution through samples of stochastic neuronal responses. Additionally, the brain continually updates its generative model to approach the true distribution of the external world. In this study, we introduce the Hierarchical Exponential-family Energy-based (HEE) model, which captures the dynamics of inference and learning. In the HEE model, we decompose the partition function into individual layers and leverage a group of neurons with shorter time constants to sample the gradient of the decomposed normalization term. This allows our model to estimate the partition function and perform inference simultaneously, circumventing the negative phase encountered in conventional energy-based models (EBMs). As a result, the learning process is localized both in time and space, and the model is easy to converge. To match the brain's rapid computation, we demonstrate that neural adaptation can serve as a momentum term, significantly accelerating the inference process. On natural image datasets, our model exhibits representations akin to those observed in the biological visual system. Furthermore, for the machine learning community, our model can generate observations through joint or marginal generation. We show that marginal generation outperforms joint generation and achieves performance on par with other EBMs.

----

## [0] Block Coordinate Plug-and-Play Methods for Blind Inverse Problems

**Authors**: *Weijie Gan, Shirin Shoushtari, Yuyang Hu, Jiaming Liu, Hongyu An, Ulugbek Kamilov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f810c2ba07bae78dfe9d25c5d40c5536-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f810c2ba07bae78dfe9d25c5d40c5536-Abstract-Conference.html)

**Abstract**:

Plug-and-play (PnP) prior is a well-known class of methods for solving imaging inverse problems by computing fixed-points of operators combining physical measurement models and learned image denoisers. While PnP methods have been extensively used for image recovery with known measurement operators, there is little work on PnP for solving blind inverse problems. We address this gap by presenting a new block-coordinate PnP (BC-PnP) method that efficiently solves this joint estimation problem by introducing learned denoisers as priors on both the unknown image and the unknown measurement operator. We present a new convergence theory for BC-PnP compatible with blind inverse problems by considering nonconvex data-fidelity terms and expansive denoisers. Our theory analyzes the convergence of BC-PnP to a stationary point of an implicit function associated with an approximate minimum mean-squared error (MMSE) denoiser. We numerically validate our method on two blind inverse problems: automatic coil sensitivity estimation in magnetic resonance imaging (MRI) and blind image deblurring. Our results show that BC-PnP provides an efficient and principled framework for using denoisers as PnP priors for jointly estimating measurement operators and images.

----

## [0] PreDiff: Precipitation Nowcasting with Latent Diffusion Models

**Authors**: *Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle C. Maddix, Yi Zhu, Mu Li, Yuyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f82ba6a6b981fbbecf5f2ee5de7db39c-Abstract-Conference.html)

**Abstract**:

Earth system forecasting has traditionally relied on complex physical models that are computationally expensive and require significant domain expertise.In the past decade, the unprecedented increase in spatiotemporal Earth observation data has enabled data-driven forecasting models using deep learning techniques.These models have shown promise for diverse Earth system forecasting tasks but either struggle with handling uncertainty or neglect domain-specific prior knowledge, resulting in averaging possible futures to blurred forecasts or generating physically implausible predictions.To address these limitations, we propose a two-stage pipeline for probabilistic spatiotemporal forecasting: 1) We develop PreDiff, a conditional latent diffusion model capable of probabilistic forecasts. 2) We incorporate an explicit knowledge alignment mechanism to align forecasts with domain-specific physical constraints. This is achieved by estimating the deviation from imposed constraints at each denoising step and adjusting the transition distribution accordingly.We conduct empirical studies on two datasets: N-body MNIST, a synthetic dataset with chaotic behavior, and SEVIR, a real-world precipitation nowcasting dataset. Specifically, we impose the law of conservation of energy in N-body MNIST and anticipated precipitation intensity in SEVIR. Experiments demonstrate the effectiveness of PreDiff in handling uncertainty, incorporating domain-specific prior knowledge, and generating forecasts that exhibit high operational utility.

----

## [0] All Points Matter: Entropy-Regularized Distribution Alignment for Weakly-supervised 3D Segmentation

**Authors**: *Liyao Tang, Zhe Chen, Shanshan Zhao, Chaoyue Wang, Dacheng Tao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f86c5c4d4dca70d30b1c12a33a2bc1a4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f86c5c4d4dca70d30b1c12a33a2bc1a4-Abstract-Conference.html)

**Abstract**:

Pseudo-labels are widely employed in weakly supervised 3D segmentation tasks where only sparse ground-truth labels are available for learning.Existing methods often rely on empirical label selection strategies, such as confidence thresholding, to generate beneficial pseudo-labels for model training.This approach may, however, hinder the comprehensive exploitation of unlabeled data points.We hypothesize that this selective usage arises from the noise in pseudo-labels generated on unlabeled data. The noise in pseudo-labels may result in significant discrepancies between pseudo-labels and model predictions, thus confusing and affecting the model training greatly.To address this issue, we propose a novel learning strategy to regularize the generated pseudo-labels and effectively narrow the gaps between pseudo-labels and model predictions.More specifically, our method introduces an Entropy Regularization loss and a Distribution Alignment loss for weakly supervised learning in 3D segmentation tasks, resulting in an ERDA learning strategy.Interestingly, by using KL distance to formulate the distribution alignment loss, it reduces to a deceptively simple cross-entropy-based loss which optimizes both the pseudo-label generation network and the 3D segmentation network simultaneously.Despite the simplicity, our method promisingly improves the performance.We validate the effectiveness through extensive experiments on various baselines and large-scale datasets.Results show that ERDA effectively enables the effective usage of all unlabeled data points for learning and achieves state-of-the-art performance under different settings.Remarkably, our method can outperform fully-supervised baselines using only 1\% of true annotations.Code and model will be made publicly available at https://github.com/LiyaoTang/ERDA.

----

## [0] SimMMDG: A Simple and Effective Framework for Multi-modal Domain Generalization

**Authors**: *Hao Dong, Ismail Nejjar, Han Sun, Eleni N. Chatzi, Olga Fink*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f88bec15cc4cb56b432ee040bb63f94f-Abstract-Conference.html)

**Abstract**:

In real-world scenarios, achieving domain generalization (DG) presents significant challenges as models are required to generalize to unknown target distributions. Generalizing to unseen multi-modal distributions poses even greater difficulties due to the distinct properties exhibited by different modalities. To overcome the challenges of achieving domain generalization in multi-modal scenarios, we propose SimMMDG, a simple yet effective multi-modal DG framework. We argue that mapping features from different modalities into the same embedding space impedes model generalization. To address this, we propose splitting the features within each modality into modality-specific and modality-shared components. We employ supervised contrastive learning on the modality-shared features to ensure they possess joint properties and impose distance constraints on modality-specific features to promote diversity. In addition, we introduce a cross-modal translation module to regularize the learned features, which can also be used for missing-modality generalization. We demonstrate that our framework is theoretically well-supported and achieves strong performance in multi-modal DG on the EPIC-Kitchens dataset and the novel Human-Animal-Cartoon (HAC) dataset introduced in this paper. Our source code and HAC dataset are available at https://github.com/donghao51/SimMMDG.

----

## [0] Bounding training data reconstruction in DP-SGD

**Authors**: *Jamie Hayes, Borja Balle, Saeed Mahloujifar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8928b073ccbec15d35f2a9d39430bfd-Abstract-Conference.html)

**Abstract**:

Differentially private training offers a protection which is usually interpreted as a guarantee against membership inference attacks. By proxy, this guarantee extends to other threats like reconstruction attacks attempting to extract complete training examples. Recent works provide evidence that if one does not need to protect against membership attacks but instead only wants to protect against a training data reconstruction, then utility of private models can be improved because less noise is required to protect against these more ambitious attacks. We investigate this question further in the context of DP-SGD, a standard algorithm for private deep learning, and provide an upper bound on the success of any reconstruction attack against DP-SGD together with an attack that empirically matches the predictions of our bound. Together, these two results open the door to fine-grained investigations on how to set the privacy parameters of DP-SGD in practice to protect against reconstruction attacks. Finally, we use our methods to demonstrate that different settings of the DP-SGD parameters leading to same DP guarantees can results in significantly different success rates for reconstruction, indicating that the DP guarantee alone might not be a good proxy for controlling the protection against reconstruction attacks.

----

## [0] T2I-CompBench: A Comprehensive Benchmark for Open-world Compositional Text-to-image Generation

**Authors**: *Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, Xihui Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8ad010cdd9143dbb0e9308c093aff24-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8ad010cdd9143dbb0e9308c093aff24-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Despite the stunning ability to generate high-quality images by recent text-to-image models, current approaches often struggle to effectively compose objects with different attributes and relationships into a complex and coherent scene. We propose T2I-CompBench, a comprehensive benchmark for open-world compositional text-to-image generation, consisting of 6,000 compositional text prompts from 3 categories (attribute binding, object relationships, and complex compositions) and 6 sub-categories (color binding, shape binding, texture binding, spatial relationships, non-spatial relationships, and complex compositions). We further propose several evaluation metrics specifically designed to evaluate compositional text-to-image generation and explore the potential and limitations of multimodal LLMs for evaluation. We introduce a new approach, Generative mOdel finetuning with Reward-driven Sample selection (GORS), to boost the compositional text-to-image generation abilities of pretrained text-to-image models. Extensive experiments and evaluations are conducted to benchmark previous methods on T2I-CompBench, and to validate the effectiveness of our proposed evaluation metrics and GORS approach. Project page is available at https://karine-h.github.io/T2I-CompBench/.

----

## [0] Neural Processes with Stability

**Authors**: *Huafeng Liu, Liping Jing, Jian Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8cea6a15db693dc525cde5e688410a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8cea6a15db693dc525cde5e688410a9-Abstract-Conference.html)

**Abstract**:

Unlike traditional statistical models depending on hand-specified priors, neural processes (NPs) have recently emerged as a class of powerful neural statistical models that combine the strengths of neural networks and stochastic processes. NPs can define a flexible class of stochastic processes well suited for highly non-trivial functions by encoding contextual knowledge into the function space. However, noisy context points introduce challenges to the algorithmic stability that small changes in training data may significantly change the models and yield lower generalization performance. In this paper, we provide theoretical guidelines for deriving stable solutions with high generalization by introducing the notion of algorithmic stability into NPs, which can be flexible to work with various NPs and achieves less biased approximation with theoretical guarantees. To illustrate the superiority of the proposed model, we perform experiments on both synthetic and real-world data, and the results demonstrate that our approach not only helps to achieve more accurate performance but also improves model robustness.

----

## [0] Multi-Agent Learning with Heterogeneous Linear Contextual Bandits

**Authors**: *Anh Do, Thanh Nguyen-Tang, Raman Arora*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8d39584f87944e5dbe46ec76f19e20a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8d39584f87944e5dbe46ec76f19e20a-Abstract-Conference.html)

**Abstract**:

As trained intelligent systems become increasingly pervasive, multiagent learning has emerged as a popular framework for studying complex interactions between autonomous agents. Yet, a formal understanding of how and when learners in heterogeneous environments benefit from sharing their respective experiences is far from complete. In this paper, we seek answers to these questions in the context of linear contextual bandits. We present a novel distributed learning algorithm based on the upper confidence bound (UCB) algorithm, which we refer to as H-LINUCB, wherein agents cooperatively minimize the group regret under the coordination of a central server. In the setting where the level of heterogeneity or dissimilarity across the environments is known to the agents, we show that H-LINUCB is provably optimal in regimes where the tasks are highly similar or highly dissimilar.

----

## [0] A polar prediction model for learning to represent visual transformations

**Authors**: *Pierre-Étienne H. Fiquet, Eero P. Simoncelli*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8e55d98b0c2569bd0aa25b076e6b3f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8e55d98b0c2569bd0aa25b076e6b3f8-Abstract-Conference.html)

**Abstract**:

All organisms make temporal predictions, and their evolutionary fitness level depends on the accuracy of these predictions. In the context of visual perception, the motions of both the observer and objects in the scene structure the dynamics of sensory signals, allowing for partial prediction of future signals based on past ones. Here, we propose a self-supervised representation-learning framework that extracts and exploits the regularities of natural videos to compute accurate predictions. We motivate the polar architecture by appealing to the Fourier shift theorem and its group-theoretic generalization, and we optimize its parameters on next-frame prediction. Through controlled experiments, we demonstrate that this approach can discover the representation of simple transformation groups acting in data. When trained on natural video datasets, our framework achieves better prediction performance than traditional motion compensation and rivals conventional deep networks, while maintaining interpretability and speed. Furthermore, the polar computations can be restructured into components resembling normalized simple and direction-selective complex cell models of primate V1 neurons. Thus, polar prediction offers a principled framework for understanding how the visual system represents sensory inputs in a form that simplifies temporal prediction.

----

## [0] MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

**Authors**: *Lili Yu, Daniel Simig, Colin Flaherty, Armen Aghajanyan, Luke Zettlemoyer, Mike Lewis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f8f78f8043f35890181a824e53a57134-Abstract-Conference.html)

**Abstract**:

Autoregressive transformers are spectacular models for short sequences but scale poorly to long sequences such as high-resolution images, podcasts, code, or books. We proposed Megabyte, a multi-scale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes. Megabyte segments sequences into patches and uses a local submodel within patches and a global model between patches. This enables sub-quadratic self-attention, much larger feedforward layers for the same compute, and improved parallelism during decoding---unlocking  better performance at reduced cost for both training and generation. Extensive experiments show that Megabyte allows byte-level models to perform competitively with subword models on long context language modeling, achieve state-of-the-art density estimation on ImageNet, and model audio from raw files. Together, these results establish the  viability of tokenization-free autoregressive sequence modeling at scale.

----

## [0] CQM: Curriculum Reinforcement Learning with a Quantized World Model

**Authors**: *Seungjae Lee, Daesol Cho, Jonghae Park, H. Jin Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f93df618c6907bc0a03222040d70d004-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f93df618c6907bc0a03222040d70d004-Abstract-Conference.html)

**Abstract**:

Recent curriculum Reinforcement Learning (RL) has shown notable progress in solving complex tasks by proposing sequences of surrogate tasks. However, the previous approaches often face challenges when they generate curriculum goals in a high-dimensional space. Thus, they usually rely on manually specified goal spaces. To alleviate this limitation and improve the scalability of the curriculum, we propose a novel curriculum method that automatically defines the semantic goal space which contains vital information for the curriculum process, and suggests curriculum goals over it. To define the semantic goal space, our method discretizes continuous observations via vector quantized-variational autoencoders (VQ-VAE) and restores the temporal relations between the discretized observations by a graph. Concurrently, ours suggests uncertainty and temporal distance-aware curriculum goals that converges to the final goals over the automatically composed goal space. We demonstrate that the proposed method allows efficient explorations in an uninformed environment with raw goal examples only. Also, ours outperforms the state-of-the-art curriculum RL methods on data efficiency and performance, in various goal-reaching tasks even with ego-centric visual inputs.

----

## [0] Debiasing Conditional Stochastic Optimization

**Authors**: *Lie He, Shiva Prasad Kasiviswanathan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f944a7bcfe9e76b34490ebe4e29196d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f944a7bcfe9e76b34490ebe4e29196d9-Abstract-Conference.html)

**Abstract**:

In this paper, we study the conditional stochastic optimization (CSO) problem which  covers a variety of applications including  portfolio selection, reinforcement learning, robust learning, causal inference, etc. The sample-averaged gradient of the CSO objective is biased due to its nested structure, and therefore requires a high sample complexity for convergence. We introduce a general stochastic extrapolation technique that effectively reduces the bias. We show that for nonconvex smooth objectives, combining this extrapolation with variance reduction techniques can achieve a significantly better sample complexity than the existing bounds. Additionally, we  develop new algorithms for the  finite-sum variant of the CSO problem that also significantly improve upon existing results. Finally, we believe that our debiasing technique has the potential to be a useful tool for addressing similar challenges in other stochastic optimization problems.

----

## [0] Cascading Bandits: Optimizing Recommendation Frequency in Delayed Feedback Environments

**Authors**: *Dairui Wang, Junyu Cao, Yan Zhang, Wei Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f95606d8e870020085990d9650b4f2a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f95606d8e870020085990d9650b4f2a1-Abstract-Conference.html)

**Abstract**:

Delayed feedback is a critical problem in dynamic recommender systems. In practice, the feedback result often depends on the frequency of recommendation. Most existing online learning literature fails to consider optimization of the recommendation frequency, and regards the reward from each successfully recommended message to be equal. In this paper, we consider a novel cascading bandits setting, where individual messages from a selected list are sent to a user periodically. Whenever a user does not like a message, she may abandon the system with a probability positively correlated with the recommendation frequency.  A learning agent needs to learn both the underlying message attraction probabilities and users' abandonment probabilities through the randomly delayed feedback. We first show a dynamic programming solution to finding the optimal message sequence in deterministic scenarios, in which the reward is allowed to vary with different messages. Then we propose a polynomial time UCB-based offline learning algorithm, and discuss its performance by characterizing its regret bound. For the online setting, we propose a learning algorithm which allows adaptive content for a given user. Numerical experiment on AmEx dataset confirms the effectiveness of our algorithms.

----

## [0] CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference

**Authors**: *Wenxuan Zeng, Meng Li, Haichuan Yang, Wen-jie Lu, Runsheng Wang, Ru Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f96839fc751b67492e17e70f5c9730e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f96839fc751b67492e17e70f5c9730e4-Abstract-Conference.html)

**Abstract**:

Deep neural network (DNN) inference based on secure 2-party computation (2PC) can offer cryptographically-secure privacy protection but suffers from orders of magnitude latency overhead due to enormous communication. Previous works heavily rely on a proxy metric of ReLU counts to approximate the communication overhead and focus on reducing the ReLUs to improve the communication efficiency. However, we observe these works achieve limited communication reduction for state-of-the-art (SOTA) 2PC protocols due to the ignorance of other linear and non-linear operations, which now contribute to the majority of communication. In this work, we present CoPriv, a framework that jointly optimizes the 2PC inference protocol and the DNN architecture. CoPriv features a new 2PC protocol for convolution based on Winograd transformation and develops DNN-aware optimization to significantly reduce the inference communication. CoPriv further develops a 2PC-aware network optimization algorithm that is compatible with the proposed protocol and simultaneously reduces the communication for all the linear and non-linear operations. We compare CoPriv with the SOTA 2PC protocol, CrypTFlow2, and demonstrate 2.1× communication reduction for both ResNet-18 and ResNet-32 on CIFAR-100. We also compare CoPriv with SOTA network optimization methods, including SNL, MetaPruning, etc. CoPriv achieves 9.98× and 3.88× online and total communication reduction with a higher accuracy compare to SNL, respectively. CoPriv also achieves 3.87× online communication reduction with more than 3% higher accuracy compared to MetaPruning.

----

## [0] Generalized equivalences between subsampling and ridge regularization

**Authors**: *Pratik Patil, Jin-Hong Du*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f976982cd1c1b9e076c096787ef6652e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f976982cd1c1b9e076c096787ef6652e-Abstract-Conference.html)

**Abstract**:

We establish precise structural and risk equivalences between subsampling and ridge regularization for ensemble ridge estimators. Specifically, we prove that linear and quadratic functionals of subsample ridge estimators, when fitted with different ridge regularization levels $\lambda$ and subsample aspect ratios $\psi$, are asymptotically equivalent along specific paths in the $(\lambda,\psi)$-plane (where $\psi$ is the ratio of the feature dimension to the subsample size). Our results only require bounded moment assumptions on feature and response distributions and allow for arbitrary joint distributions. Furthermore, we provide a data-dependent method to determine the equivalent paths of $(\lambda,\psi)$. An indirect implication of our equivalences is that optimally tuned ridge regression exhibits a monotonic prediction risk in the data aspect ratio. This resolves a recent open problem raised by Nakkiran et al. for general data distributions under proportional asymptotics, assuming a mild regularity condition that maintains regression hardness through linearized signal-to-noise ratios.

----

## [0] D4Explainer: In-distribution Explanations of Graph Neural Network via Discrete Denoising Diffusion

**Authors**: *Jialin Chen, Shirley Wu, Abhijit Gupta, Rex Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f978c8f3b5f399cae464e85f72e28503-Abstract-Conference.html)

**Abstract**:

The widespread deployment of Graph Neural Networks (GNNs) sparks significant interest in their explainability, which plays a vital role in model auditing and ensuring trustworthy graph learning. The objective of GNN explainability is to discern the underlying graph structures that have the most significant impact on model predictions. Ensuring that explanations generated are reliable necessitates consideration of the in-distribution property, particularly due to the vulnerability of GNNs to out-of-distribution data. Unfortunately, prevailing explainability methods tend to constrain the generated explanations to the structure of the original graph, thereby downplaying the significance of the in-distribution property and resulting in explanations that lack reliability.To address these challenges, we propose D4Explainer, a novel approach that provides in-distribution GNN explanations for both counterfactual and model-level explanation scenarios. The proposed D4Explainer incorporates generative graph distribution learning into the optimization objective, which accomplishes two goals: 1) generate a collection of diverse counterfactual graphs that conform to the in-distribution property for a given instance, and 2) identify the most discriminative graph patterns that contribute to a specific class prediction, thus serving as model-level explanations. It is worth mentioning that D4Explainer is the first unified framework that combines both counterfactual and model-level explanations.Empirical evaluations conducted on synthetic and real-world datasets provide compelling evidence of the state-of-the-art performance achieved by D4Explainer in terms of explanation accuracy, faithfulness, diversity, and robustness.

----

## [0] Core-sets for Fair and Diverse Data Summarization

**Authors**: *Sepideh Mahabadi, Stojan Trajanovski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f980ba94f513168f2b292f58aef929ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f980ba94f513168f2b292f58aef929ec-Abstract-Conference.html)

**Abstract**:

We study core-set construction algorithms for the task of Diversity Maximization under fairness/partition constraint. Given a set of points $P$ in a metric space partitioned into $m$ groups, and given $k_1,\ldots,k_m$, the goal of this problem is to pick $k_i$ points from each group $i$ such that the overall diversity of the $k=\sum_i k_i$ picked points is maximized. We consider two natural diversity measures: sum-of-pairwise distances and sum-of-nearest-neighbor distances, and show improved core-set construction algorithms with respect to these measures. More precisely, we show the first constant factor core-set w.r.t. sum-of-pairwise distances whose size is independent of the size of the dataset and the aspect ratio. Second, we show the first core-set w.r.t. the sum-of-nearest-neighbor distances. Finally, we run several experiments showing the effectiveness of our core-set approach. In particular, we apply constrained diversity maximization to summarize a set of timed messages that takes into account the messages' recency. Specifically, the summary should include more recent messages compared to older ones. This is a real task in one of the largest communication platforms, affecting the experience of hundreds of millions daily active users. By utilizing our core-set method for this task, we achieve a 100x speed-up while losing the diversity by only a few percent. Moreover, our approach allows us to improve the space usage of the algorithm in the streaming setting.

----

## [0] Energy-Efficient Scheduling with Predictions

**Authors**: *Eric Balkanski, Noémie Périvier, Clifford Stein, Hao-Ting Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f99bb39502f09c4825e89760b4e1ad04-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f99bb39502f09c4825e89760b4e1ad04-Abstract-Conference.html)

**Abstract**:

An  important goal of modern scheduling systems is to efficiently manage power usage. In energy-efficient scheduling,   the operating system controls the speed at which a machine is processing jobs  with the dual objective of  minimizing energy consumption and optimizing the quality of service cost of the resulting schedule. Since  machine-learned predictions  about future  requests can often be learned from historical data, a recent line of work  on learning-augmented algorithms aims to achieve improved performance guarantees by leveraging predictions.   In particular, for energy-efficient scheduling, Bamas et. al. [NeurIPS '20] and Antoniadis et. al. [SWAT '22]  designed algorithms with predictions for the  energy minimization with deadlines problem and achieved an improved competitive ratio when the prediction error is small while also maintaining  worst-case bounds even when the prediction error is arbitrarily large.In this paper, we consider a general setting for energy-efficient scheduling and provide a flexible learning-augmented algorithmic framework that takes as input an offline and an online algorithm for the desired energy-efficient scheduling problem. We show that, when the prediction error is small, this framework gives improved competitive ratios for many different energy-efficient scheduling problems, including  energy minimization with deadlines, while also maintaining a bounded competitive ratio regardless of the prediction error. Finally, we empirically demonstrate that this framework achieves an improved performance on real and synthetic datasets.

----

## [0] Diversify Your Vision Datasets with Automatic Diffusion-based Augmentation

**Authors**: *Lisa Dunlap, Alyssa Umino, Han Zhang, Jiezhi Yang, Joseph E. Gonzalez, Trevor Darrell*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f99f7b22ad47fa6ce151730cf8d17911-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f99f7b22ad47fa6ce151730cf8d17911-Abstract-Conference.html)

**Abstract**:

Many fine-grained classification tasks, like rare animal identification, have limited training data and consequently classifiers trained on these datasets often fail to  generalize to variations in the domain like changes in weather or location.  As such, we explore how natural language descriptions of the domains seen in training data can be used with large vision models trained on diverse pretraining datasets to generate useful variations of the training data. We introduce ALIA (Automated Language-guided Image Augmentation), a method which utilizes large vision and language models to automatically generate natural language descriptions of a dataset's domains and augment the training data via language-guided image editing. To maintain data integrity, a model trained on the original dataset filters out minimal image edits and those which corrupt class-relevant information. The resulting dataset is visually consistent with the original training data and offers significantly enhanced diversity. We show that ALIA is able to surpasses traditional data augmentation and text-to-image generated data on fine-grained classification tasks, including cases of domain generalization and contextual bias. Code is available at https://github.com/lisadunlap/ALIA.

----

## [0] DISCS: A Benchmark for Discrete Sampling

**Authors**: *Katayoon Goshvadi, Haoran Sun, Xingchao Liu, Azade Nova, Ruqi Zhang, Will Grathwohl, Dale Schuurmans, Hanjun Dai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f9ad87c1ebbae8a3555adb31dbcacf44-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/f9ad87c1ebbae8a3555adb31dbcacf44-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Sampling in discrete spaces, with critical applications in simulation and optimization, has recently been boosted by significant advances in gradient-based approaches that exploit modern accelerators like GPUs. However, two key challenges are hindering further advancement in research on discrete sampling. First, since there is no consensus on experimental settings and evaluation setups, the empirical results in different research papers are often not comparable. Second, implementing samplers and target distributions often requires a nontrivial amount of effort in terms of calibration and parallelism. To tackle these challenges, we propose DISCS (DISCrete Sampling), a tailored package and benchmark that supports unified and efficient experiment implementation and evaluations for discrete sampling in three types of tasks: sampling from classical graphical models and energy based generative models, and sampling for solving combinatorial optimization. Throughout the comprehensive evaluations in DISCS, we gained new insights into scalability, design principles for proposal distributions, and lessons for adaptive sampling design. DISCS efficiently implements representative discrete samplers in existing research works as baselines and offers a simple interface that researchers can conveniently add new discrete samplers and directly compare their performance with the benchmark result in a calibrated setup.

----

## [0] Improving Robustness with Adaptive Weight Decay

**Authors**: *Amin Ghiasi, Ali Shafahi, Reza Ardekani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f9d7d6c695bc983fcfb5b70a5fbdfd2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f9d7d6c695bc983fcfb5b70a5fbdfd2f-Abstract-Conference.html)

**Abstract**:

We propose adaptive weight decay, which automatically tunes the hyper-parameter for weight decay during each training iteration. For classification problems, we propose changing the value of the weight decay hyper-parameter on the fly based on the strength of updates from the classification loss (i.e., gradient of cross-entropy), and the regularization loss (i.e., $\ell_2$-norm of the weights). We show that this simple modification can result in large improvements in adversarial robustness — an area which suffers from robust overfitting — without requiring extra data accros various datasets and architecture choices. For example, our reformulation results in 20\% relative robustness improvement for CIFAR-100, and 10\% relative robustness improvement on CIFAR-10 comparing to the best tuned hyper-parameters of traditional weight decay resulting in models that have comparable performance to SOTA robustness methods. In addition, this method has other desirable properties, such as less sensitivity to learning rate, and smaller weight norms, which the latter contributes to robustness to overfitting to label noise, and pruning.

----

## [0] Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning

**Authors**: *Lin Guan, Karthik Valmeekam, Sarath Sreedharan, Subbarao Kambhampati*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f9f54762cbb4fe4dbffdd4f792c31221-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f9f54762cbb4fe4dbffdd4f792c31221-Abstract-Conference.html)

**Abstract**:

There is a growing interest in applying pre-trained large language models (LLMs) to planning problems. However, methods that use LLMs directly as planners are currently impractical due to several factors, including limited correctness of plans, strong reliance on feedback from interactions with simulators or even the actual environment, and the inefficiency in utilizing human feedback. In this work, we introduce a novel alternative paradigm that constructs an explicit world (domain) model in planning domain definition language (PDDL) and then uses it to plan with sound domain-independent planners. To address the fact that LLMs may not generate a fully functional PDDL model initially, we employ LLMs as an interface between PDDL and sources of corrective feedback, such as PDDL validators and humans. For users who lack a background in PDDL, we show that LLMs can translate PDDL into natural language and effectively encode corrective feedback back to the underlying domain model. Our framework not only enjoys the correctness guarantee offered by the external planners but also reduces human involvement by allowing users to correct domain models at the beginning, rather than inspecting and correcting (through interactive prompting) every generated plan as in previous work. On two IPC domains and a Household domain that is more complicated than commonly used benchmarks such as ALFWorld, we demonstrate that GPT-4 can be leveraged to produce high-quality PDDL models for over 40 actions, and the corrected PDDL models are then used to successfully solve 48 challenging planning tasks. Resources, including the source code, are released at: https://guansuns.github.io/pages/llm-dm.

----

## [0] Described Object Detection: Liberating Object Detection with Flexible Expressions

**Authors**: *Chi Xie, Zhao Zhang, Yixuan Wu, Feng Zhu, Rui Zhao, Shuang Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/f9fd24fd32eccc14cd3ecd3716a1cbf8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/f9fd24fd32eccc14cd3ecd3716a1cbf8-Abstract-Conference.html)

**Abstract**:

Detecting objects based on language information is a popular task that includes Open-Vocabulary object Detection (OVD) and Referring Expression Comprehension (REC). In this paper, we advance them to a more practical setting called *Described Object Detection* (DOD) by expanding category names to flexible language expressions for OVD and overcoming the limitation of REC only grounding the pre-existing object. We establish the research foundation for DOD by constructing a *Description Detection Dataset* ($D^3$). This dataset features flexible language expressions, whether short category names or long descriptions, and annotating all described objects on all images without omission. By evaluating previous SOTA methods on $D^3$, we find some troublemakers that fail current REC, OVD, and bi-functional methods. REC methods struggle with confidence scores, rejecting negative instances, and multi-target scenarios, while OVD methods face constraints with long and complex descriptions. Recent bi-functional methods also do not work well on DOD due to their separated training procedures and inference strategies for REC and OVD tasks. Building upon the aforementioned findings, we propose a baseline that largely improves REC methods by reconstructing the training data and introducing a binary classification sub-task, outperforming existing methods. Data and code are available at https://github.com/shikras/d-cube and related works are tracked in https://github.com/Charles-Xie/awesome-described-object-detection.

----

## [0] Learning Cuts via Enumeration Oracles

**Authors**: *Daniel Thuerck, Boro Sofranac, Marc E. Pfetsch, Sebastian Pokutta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa0126bb7ebad258bf4ffdbbac2dd787-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa0126bb7ebad258bf4ffdbbac2dd787-Abstract-Conference.html)

**Abstract**:

Cutting-planes are one of the most important building blocks for solving large-scale integer programming (IP) problems to (near) optimality. The majority of cutting plane approaches rely on explicit rules to derive valid inequalities that can separate the target point from the feasible set. Local cuts, on the other hand, seek to directly derive the facets of the underlying polyhedron and use them as cutting planes. However, current approaches rely on solving Linear Programming (LP) problems in order to derive such a hyperplane. In this paper, we present a novel generic approach for learning the facets of the underlying polyhedron by accessing it implicitly via an enumeration oracle in a reduced dimension. This is achieved by embedding the oracle in a variant of the Frank-Wolfe algorithm which is capable of generating strong cutting planes, effectively turning the enumeration oracle into a separation oracle. We demonstrate the effectiveness of our approach with a case study targeting the multidimensional knapsack problem (MKP).

----

## [0] Learning Descriptive Image Captioning via Semipermeable Maximum Likelihood Estimation

**Authors**: *Zihao Yue, Anwen Hu, Liang Zhang, Qin Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa1cfe4e956d85e016b1f8f49b189a0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa1cfe4e956d85e016b1f8f49b189a0b-Abstract-Conference.html)

**Abstract**:

Image captioning aims to describe visual content in natural language. As 'a picture is worth a thousand words', there could be various correct descriptions for an image. However, with maximum likelihood estimation as the training objective, the captioning model is penalized whenever its prediction mismatches with the label. For instance, when the model predicts a word expressing richer semantics than the label, it will be penalized and optimized to prefer more concise expressions, referred to as conciseness optimization. In contrast, predictions that are more concise than labels lead to richness optimization. Such conflicting optimization directions could eventually result in the model generating general descriptions. In this work, we introduce Semipermeable MaxImum Likelihood Estimation (SMILE), which allows richness optimization while blocking conciseness optimization, thus encouraging the model to generate longer captions with more details. Extensive experiments on two mainstream image captioning datasets MSCOCO and Flickr30K demonstrate that SMILE significantly enhances the descriptiveness of generated captions. We further provide in-depth investigations to facilitate a better understanding of how SMILE works.

----

## [0] Alternating Gradient Descent and Mixture-of-Experts for Integrated Multimodal Perception

**Authors**: *Hassan Akbari, Dan Kondratyuk, Yin Cui, Rachel Hornung, Huisheng Wang, Hartwig Adam*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa384d5f9e85380833d523766af5941c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa384d5f9e85380833d523766af5941c-Abstract-Conference.html)

**Abstract**:

We present Integrated Multimodal Perception (IMP), a simple and scalable multimodal multi-task training and modeling approach. IMP integrates multimodal inputs including image, video, text, and audio into a single Transformer encoder with minimal modality-specific components. IMP makes use of a novel design that combines Alternating Gradient Descent (AGD) and Mixture-of-Experts (MoE) for efficient model & task scaling. We conduct extensive empirical studies and reveal the following key insights:    1) performing gradient descent updates by alternating on diverse modalities, loss functions, and tasks, with varying input resolutions, efficiently improves the model.    2) sparsification with MoE on a single modality-agnostic encoder substantially improves the performance, outperforming dense models that use modality-specific encoders or additional fusion layers and greatly mitigating the conflicts between modalities. IMP achieves competitive performance on a wide range of downstream tasks including video classification, image classification, image-text, and video-text retrieval. Most notably, we train a sparse IMP-MoE-L focusing on video tasks that achieves new state-of-the-art in zero-shot video classification: 77.0% on Kinetics-400, 76.8% on Kinetics-600, and 68.3% on Kinetics-700, improving the previous state-of-the-art by +5%, +6.7%, and +5.8%, respectively, while using only 15% of their total training computational cost.

----

## [0] The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data Only

**Authors**: *Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Hamza Alobeidli, Alessandro Cappelli, Baptiste Pannier, Ebtesam Almazrouei, Julien Launay*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa3ed726cc5073b9c31e3e49a807789c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa3ed726cc5073b9c31e3e49a807789c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Large language models are commonly trained on a mixture of filtered web data and curated ``high-quality'' corpora, such as social media conversations, books, or technical papers. This curation process is believed to be necessary to produce performant models with broad zero-shot generalization abilities. However, as larger models requiring pretraining on trillions of tokens are considered, it is unclear how scalable is curation, and whether we will run out of unique high-quality data soon.  At variance with previous beliefs, we show that properly filtered and deduplicated web data alone can lead to powerful models; even significantly outperforming models trained on The Pile. Despite extensive filtering, the high-quality data we extract from the web is still plentiful, and we are able to obtain five trillion tokens from CommonCrawl. We publicly release an extract of 500 billion tokens from our RefinedWeb dataset, and 1.3/7.5B parameters language models trained on it.

----

## [0] Self-Correcting Bayesian Optimization through Bayesian Active Learning

**Authors**: *Carl Hvarfner, Erik Hellsten, Frank Hutter, Luigi Nardi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa55bf1947530fc9567059ff42a806c2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa55bf1947530fc9567059ff42a806c2-Abstract-Conference.html)

**Abstract**:

Gaussian processes are the model of choice in Bayesian optimization and active learning. Yet, they are highly dependent on cleverly chosen hyperparameters to reach their full potential, and little effort is devoted to finding good hyperparameters in the literature. We demonstrate the impact of selecting good hyperparameters for GPs and present two acquisition functions that explicitly prioritize hyperparameter learning. Statistical distance-based Active Learning (SAL) considers the average disagreement between samples from the posterior, as measured by a statistical distance. SAL outperforms the state-of-the-art in Bayesian active learning on several test functions. We then introduce Self-Correcting Bayesian Optimization (SCoreBO), which extends SAL to perform Bayesian optimization and active learning simultaneously. SCoreBO learns the model hyperparameters at improved rates compared to vanilla BO, while outperforming the latest Bayesian optimization methods on traditional benchmarks. Moreover, we demonstrate the importance of self-correction on atypical Bayesian optimization tasks.

----

## [0] SE(3) Equivariant Augmented Coupling Flows

**Authors**: *Laurence I. Midgley, Vincent Stimper, Javier Antorán, Emile Mathieu, Bernhard Schölkopf, José Miguel Hernández-Lobato*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa55eb802a531c8087e225ecf2dcfbca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa55eb802a531c8087e225ecf2dcfbca-Abstract-Conference.html)

**Abstract**:

Coupling normalizing flows allow for fast sampling and density evaluation, making them the tool of choice for probabilistic modeling of physical systems. However, the standard coupling architecture precludes endowing flows that operate on the Cartesian coordinates of atoms with the SE(3) and permutation invariances of physical systems. This work proposes a coupling flow that preserves SE(3) and permutation equivariance by performing coordinate splits along additional augmented dimensions. At each layer, the flow maps atoms' positions into learned SE(3) invariant bases, where we apply standard flow transformations, such as monotonic rational-quadratic splines, before returning to the original basis.Crucially, our flow preserves fast sampling and density evaluation, and may be used to produce unbiased estimates of expectations with respect to the target distribution via importance sampling.When trained on the DW4, LJ13, and QM9-positional datasets, our flow is competitive with equivariant continuous normalizing flows and diffusion models, while allowing sampling more than an order of magnitude faster.Moreover, to the best of our knowledge, we are the first to learn the full Boltzmann distribution of alanine dipeptide by only modeling the Cartesian positions of its atoms.Lastly, we demonstrate that our flow can be trained to approximately sample from the Boltzmann distribution of the DW4 and LJ13 particle systems using only their energy functions.

----

## [0] Bridging the Domain Gap: Self-Supervised 3D Scene Understanding with Foundation Models

**Authors**: *Zhimin Chen, Longlong Jing, Yingwei Li, Bing Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa5b423e24b442180bcd4e13ae75a27f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa5b423e24b442180bcd4e13ae75a27f-Abstract-Conference.html)

**Abstract**:

Foundation models have achieved remarkable results in 2D and language tasks like image segmentation, object detection, and visual-language understanding. However, their potential to enrich 3D scene representation learning is largely untapped due to the existence of the domain gap. In this work, we propose an innovative methodology called Bridge3D to address this gap by pre-training 3D models using features, semantic masks, and captions sourced from foundation models. Specifically, our method employs semantic masks from foundation models to guide the masking and reconstruction process for the masked autoencoder, enabling more focused attention on foreground representations. Moreover, we bridge the 3D-text gap at the scene level using image captioning foundation models, thereby facilitating scene-level knowledge distillation. We further extend this bridging effort by introducing an innovative object-level knowledge distillation method that harnesses highly accurate object-level masks and semantic text data from foundation models. Our methodology significantly surpasses the performance of existing state-of-the-art methods in 3D object detection and semantic segmentation tasks. For instance, on the ScanNet dataset, Bridge3D improves the baseline by a notable margin of 6.3%. Code will be available at: https://github.com/Zhimin-C/Bridge3D

----

## [0] Imagine That! Abstract-to-Intricate Text-to-Image Synthesis with Scene Graph Hallucination Diffusion

**Authors**: *Shengqiong Wu, Hao Fei, Hanwang Zhang, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa64505ebdc94531087bc81251ce2376-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa64505ebdc94531087bc81251ce2376-Abstract-Conference.html)

**Abstract**:

In this work, we investigate the task of text-to-image (T2I) synthesis under the abstract-to-intricate setting, i.e., generating intricate visual content from simple abstract text prompts. Inspired by human imagination intuition, we propose a novel scene-graph hallucination (SGH) mechanism for effective abstract-to-intricate T2I synthesis. SGH carries out scene hallucination by expanding the initial scene graph (SG) of the input prompt with more feasible specific scene structures, in which the structured semantic representation of SG ensures high controllability of the intrinsic scene imagination. To approach the T2I synthesis, we deliberately build an SG-based hallucination diffusion system. First, we implement the SGH module based on the discrete diffusion technique, which evolves the SG structure by iteratively adding new scene elements. Then, we utilize another continuous-state diffusion model as the T2I synthesizer, where the overt image-generating process is navigated by the underlying semantic scene structure induced from the SGH module. On the benchmark COCO dataset, our system outperforms the existing best-performing T2I model by a significant margin, especially improving on the abstract-to-intricate T2I generation. Further in-depth analyses reveal how our methods advance.

----

## [0] A unified framework for information-theoretic generalization bounds

**Authors**: *Yifeng Chu, Maxim Raginsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa67d13ba6c73637593bbcc92f6400ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa67d13ba6c73637593bbcc92f6400ff-Abstract-Conference.html)

**Abstract**:

This paper presents a general methodology for deriving information-theoretic generalization bounds for learning algorithms. The main technical tool is a probabilistic decorrelation lemma based on a change of measure and a relaxation of Young's inequality in $L_{\psi_p}$ Orlicz spaces. Using the decorrelation lemma in combination with other techniques, such as symmetrization, couplings, and chaining in the space of probability measures, we obtain new upper bounds on the generalization error, both in expectation and in high probability, and recover as special cases many of the existing generalization bounds, including the ones based on mutual information, conditional mutual information, stochastic chaining, and PAC-Bayes inequalities. In addition, the Fernique--Talagrand upper bound on the expected supremum of a subgaussian process emerges as a special case.

----

## [0] Diverse Shape Completion via Style Modulated Generative Adversarial Networks

**Authors**: *Wesley Khademi, Fuxin Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa68ea2ff794ce792a688dec82c04f49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa68ea2ff794ce792a688dec82c04f49-Abstract-Conference.html)

**Abstract**:

Shape completion aims to recover the full 3D geometry of an object from a partial observation. This problem is inherently multi-modal since there can be many ways to plausibly complete the missing regions of a shape. Such diversity would be indicative of the underlying uncertainty of the shape and could be preferable for downstream tasks such as planning. In this paper, we propose a novel conditional generative adversarial network that can produce many diverse plausible completions of a partially observed point cloud. To enable our network to produce multiple completions for the same partial input, we introduce stochasticity into our network via style modulation. By extracting style codes from complete shapes during training, and learning a distribution over them, our style codes can explicitly carry shape category information leading to better completions. We further introduce diversity penalties and discriminators at multiple scales to prevent conditional mode collapse and to train without the need for multiple ground truth completions for each partial input. Evaluations across several synthetic and real datasets demonstrate that our method achieves significant improvements in respecting the partial observations while obtaining greater diversity in completions.

----

## [0] On the Role of Randomization in Adversarially Robust Classification

**Authors**: *Lucas Gnecco Heredia, Muni Sreenivas Pydi, Laurent Meunier, Benjamin Négrevergne, Yann Chevaleyre*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fa9755043814e7f08d859a286bb83c35-Abstract-Conference.html)

**Abstract**:

Deep neural networks are known to be vulnerable to small adversarial perturbations in test data. To defend against adversarial attacks, probabilistic classifiers have been proposed as an alternative to deterministic ones. However, literature has conflicting findings on the effectiveness of probabilistic classifiers in comparison to deterministic ones. In this paper, we clarify the role of randomization in building adversarially robust classifiers.Given a base hypothesis set of deterministic classifiers, we show the conditions under which a randomized ensemble outperforms the hypothesis set in adversarial risk, extending previous results.Additionally, we show that for any probabilistic binary classifier (including randomized ensembles), there exists a deterministic classifier that outperforms it. Finally, we give an explicit description of the deterministic hypothesis set that contains such a deterministic classifier for many types of commonly used probabilistic classifiers, i.e. randomized ensembles and parametric/input noise injection.

----

## [0] Controlling Text-to-Image Diffusion by Orthogonal Finetuning

**Authors**: *Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html)

**Abstract**:

Large text-to-image diffusion models have impressive capabilities in generating photorealistic images from text prompts. How to effectively guide or control these powerful models to perform different downstream tasks becomes an important open problem. To tackle this challenge, we introduce a principled finetuning method -- Orthogonal Finetuning (OFT), for adapting text-to-image diffusion models to downstream tasks. Unlike existing methods, OFT can provably preserve hyperspherical energy which characterizes the pairwise neuron relationship on the unit hypersphere. We find that this property is crucial for preserving the semantic generation ability of text-to-image diffusion models. To improve finetuning stability, we further propose Constrained Orthogonal Finetuning (COFT) which imposes an additional radius constraint to the hypersphere. Specifically, we consider two important finetuning text-to-image tasks: subject-driven generation where the goal is to generate subject-specific images given a few images of a subject and a text prompt, and controllable generation where the goal is to enable the model to take in additional control signals. We empirically show that our OFT framework outperforms existing methods in generation quality and convergence speed.

----

## [0] NCDL: A Framework for Deep Learning on non-Cartesian Lattices

**Authors**: *Joshua Horacsek, Usman R. Alim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fab489de1a3224f0394d8f1d3c3213a8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fab489de1a3224f0394d8f1d3c3213a8-Abstract-Conference.html)

**Abstract**:

The use of non-Cartesian grids is a niche but important topic in sub-fields of the numerical sciences such as simulation and scientific visualization. However, non-Cartesian approaches are virtually unexplored in machine learning. This is likely due to the difficulties in the representation of data on non-Cartesian domains and the lack of support for standard machine learning operations on non-Cartesian data. This paper proposes a new data structure called the lattice tensor which generalizes traditional tensor spatio-temporal operations to lattice tensors, enabling the use of standard machine learning algorithms on non-Cartesian data. However, data need not reside on a non-Cartesian structure, we use non-Dyadic downsampling schemes to bring Cartesian data into a non-Cartesian space for further processing.   We introduce a software library that implements the lattice tensor container (with some common machine learning operations), and   demonstrate its effectiveness. Our method provides a general framework for machine learning on non-Cartesian domains, addressing the challenges mentioned above and filling a gap in the current literature.

----

## [0] Human-in-the-Loop Optimization for Deep Stimulus Encoding in Visual Prostheses

**Authors**: *Jacob Granley, Tristan Fauvel, Matthew Chalk, Michael Beyeler*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb06bc3abcece7b8725a8b83b8fa3632-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb06bc3abcece7b8725a8b83b8fa3632-Abstract-Conference.html)

**Abstract**:

Neuroprostheses show potential in restoring lost sensory function and enhancing human capabilities, but the sensations produced by current devices often seem unnatural or distorted. Exact placement of implants and differences in individual perception lead to significant variations in stimulus response, making personalized stimulus optimization a key challenge. Bayesian optimization could be usedto optimize patient-specific stimulation parameters with limited noisy observations, but is not feasible for high-dimensional stimuli. Alternatively, deep learning models can optimize stimulus encoding strategies, but typically assume perfect knowledge of patient-specific variations. Here we propose a novel, practically feasible approach that overcomes both of these fundamental limitations. First, a deep encoder network is trained to produce optimal stimuli for any individual patient by inverting a forward model mapping electrical stimuli to visual percepts. Second, a preferential Bayesian optimization strategy utilizes this encoder to learn the optimal patient-specific parameters for a new patient, using a minimal number of pairwise comparisons between candidate stimuli. We demonstrate the viability of this approach on a novel, state-of-the-art visual prosthesis model. Our approach quickly learns a personalized stimulus encoder and leads to dramatic improvements in the quality of restored vision, outperforming existing encoding strategies. Further, this approach is robust to noisy patient feedback and misspecifications in the underlying forward model. Overall, our results suggest that combining the strengths of deep learning and Bayesian optimization could significantly improve the perceptual experience of patients fitted with visual prostheses and may prove a viable solution for a range of neuroprosthetic technologies

----

## [0] CAST: Cross-Attention in Space and Time for Video Action Recognition

**Authors**: *Dongho Lee, Jongseo Lee, Jinwoo Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb1b83b35e96998ddfc0ce1dab635445-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb1b83b35e96998ddfc0ce1dab635445-Abstract-Conference.html)

**Abstract**:

Recognizing human actions in videos requires spatial and temporal understanding. Most existing action recognition models lack a balanced spatio-temporal understanding of videos. In this work, we propose a novel two-stream architecture, called Cross-Attention in Space and Time (CAST), that achieves a balanced spatio-temporal understanding of videos using only RGB input. Our proposed bottleneck cross-attention mechanism enables the spatial and temporal expert models to exchange information and make synergistic predictions, leading to improved performance. We validate the proposed method with extensive experiments on public benchmarks with different characteristics: EPIC-Kitchens-100, Something-Something-V2, and Kinetics-400. Our method consistently shows favorable performance across these datasets, while the performance of existing methods fluctuates depending on the dataset characteristics. The code is available at https://github.com/KHU-VLL/CAST.

----

## [0] Faster Differentially Private Convex Optimization via Second-Order Methods

**Authors**: *Arun Ganesh, Mahdi Haghifam, Thomas Steinke, Abhradeep Guha Thakurta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb1d9c3fc2161e12aa71cdcab74b9d2c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb1d9c3fc2161e12aa71cdcab74b9d2c-Abstract-Conference.html)

**Abstract**:

Differentially private (stochastic) gradient descent is the workhorse of DP private machine learning in both the convex and non-convex settings. Without privacy constraints, second-order methods, like Newton's method, converge faster than first-order methods like gradient descent. In this work, we investigate the prospect of using the second-order information from the loss function to accelerate DP convex optimization. We first develop a private variant of the regularized cubic Newton method of Nesterov and Polyak, and show that for the class of strongly convex loss functions, our algorithm has quadratic convergence and achieves the optimal excess loss. We then design a practical second-order DP algorithm for the unconstrained logistic regression problem. We theoretically and empirically study the performance of our algorithm. Empirical results show our algorithm consistently achieves the best excess loss compared to other baselines and is 10-40x faster than DP-GD/DP-SGD for challenging datasets.

----

## [0] Auditing for Human Expertise

**Authors**: *Rohan Alur, Loren Laine, Darrick K. Li, Manish Raghavan, Devavrat Shah, Dennis L. Shung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html)

**Abstract**:

High-stakes prediction tasks (e.g., patient diagnosis) are often handled by trained human experts. A common source of concern about automation in these settings is that experts may exercise intuition that is difficult to model and/or have access to information (e.g., conversations with a patient) that is simply unavailable to a would-be algorithm. This raises a natural question whether human experts add value which could not be captured by an algorithmic predictor.We develop a statistical framework under which we can pose this question as a natural hypothesis test. Indeed, as our framework highlights, detecting human expertise is more subtle than simply comparing the accuracy of expert predictions to those made by a particular learning algorithm. Instead, we propose a simple procedure which tests whether expert predictions are statistically independent from the outcomes of interest after conditioning on the available inputs (‘features’). A rejection of our test thus suggests that human experts may add value to any algorithm trained on the available data, and has direct implications for whether human-AI ‘complementarity’ is achievable in a given prediction task.We highlight the utility of our procedure using admissions data collected from the emergency department of a large academic hospital system, where we show that physicians’ admit/discharge decisions for patients with acute gastrointestinal bleeding (AGIB) appear to be incorporating information that is not available to a standard algorithmic screening tool. This is despite the fact that the screening tool is arguably more accurate than physicians’ discretionary decisions, highlighting that – even absent normative concerns about accountability or interpretability – accuracy is insufficient to justify algorithmic automation.

----

## [0] Smooth, exact rotational symmetrization for deep learning on point clouds

**Authors**: *Sergey Pozdnyakov, Michele Ceriotti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb4a7e3522363907b26a86cc5be627ac-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb4a7e3522363907b26a86cc5be627ac-Abstract-Conference.html)

**Abstract**:

Point clouds are versatile representations of 3D objects and have found widespread application in science and engineering. Many successful deep-learning models have been proposed that use them as input. The domain of chemical and materials modeling is especially challenging because exact compliance with physical constraints is highly desirable for a model to be usable in practice. These constraints include smoothness and invariance with respect to translations, rotations, and permutations of identical atoms. If these requirements are not rigorously fulfilled, atomistic simulations might lead to absurd outcomes even if the model has excellent accuracy. Consequently, dedicated architectures, which achieve invariance by restricting their design space, have been developed. General-purpose point-cloud models are more varied but often disregard rotational symmetry. We propose a general symmetrization method that adds rotational equivariance to any given model while preserving all the other requirements.Our approach simplifies the development of better atomic-scale machine-learning schemes by relaxing the constraints on the design space and making it possible to incorporate ideas that proved effective in other domains.We demonstrate this idea by introducing the Point Edge Transformer (PET) architecture, which is not intrinsically equivariant but achieves state-of-the-art performance on several benchmark datasets of molecules and solids. A-posteriori application of our general protocol makes PET exactly equivariant, with minimal changes to its accuracy.

----

## [0] Functional Equivalence and Path Connectivity of Reducible Hyperbolic Tangent Networks

**Authors**: *Matthew Farrugia-Roberts*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb64a43508e0cfe53ee6179ff31ea900-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb64a43508e0cfe53ee6179ff31ea900-Abstract-Conference.html)

**Abstract**:

Understanding the learning process of artificial neural networks requires clarifying the structure of the parameter space within which learning takes place. A neural network parameter's functional equivalence class is the set of parameters implementing the same input--output function. For many architectures, almost all parameters have a simple and well-documented functional equivalence class. However, there is also a vanishing minority of reducible parameters, with richer functional equivalence classes caused by redundancies among the network's units.In this paper, we give an algorithmic characterisation of unit redundancies and reducible functional equivalence classes for a single-hidden-layer hyperbolic tangent architecture. We show that such functional equivalence classes are piecewise-linear path-connected sets, and that for parameters with a majority of redundant units, the sets have a diameter of at most 7 linear segments.

----

## [0] On Robust Streaming for Learning with Experts: Algorithms and Lower Bounds

**Authors**: *David P. Woodruff, Fred Zhang, Samson Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb71332951af4ae27fbd457daadc5341-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb71332951af4ae27fbd457daadc5341-Abstract-Conference.html)

**Abstract**:

In the online learning with experts problem, an algorithm makes predictions about an outcome on each of $T$ days, given a set of $n$ experts who make predictions on each day. The algorithm is given feedback on the outcomes of each day, including the cost of its prediction and the cost of the expert predictions, and the goal is to make a prediction with the minimum cost, compared to the best expert in hindsight. However, often the predictions made by experts or algorithms at some time influence future outcomes, so that the input is adaptively generated. In this paper, we study robust algorithms for the experts problem under memory constraints. We first give a randomized algorithm that is robust to adaptive inputs that uses $\widetilde{O}\left(\frac{n}{R\sqrt{T}}\right)$ space for  $M=O\left(\frac{R^2 T}{\log^2 n}\right)$, thereby showing a smooth space-regret trade-off. We then show a space lower bound of $\widetilde{\Omega}\left(\frac{nM}{RT}\right)$ for any randomized algorithm that achieves regret $R$ with probability $1-2^{-\Omega(T)}$, when the best expert makes $M$ mistakes. Our result implies that the natural deterministic algorithm, which iterates through pools of experts until each expert in the pool has erred, is optimal up to polylogarithmic factors. Finally, we empirically demonstrate the benefit of using robust procedures against a white-box adversary that has access to the internal state of the algorithm.

----

## [0] Stochastic Optimal Control for Collective Variable Free Sampling of Molecular Transition Paths

**Authors**: *Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Bernd Ensing, Max Welling*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb7f55f36c53247a704792a721272706-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb7f55f36c53247a704792a721272706-Abstract-Conference.html)

**Abstract**:

We consider the problem of sampling transition paths between two given metastable states of a molecular system, eg. a folded and unfolded protein or products and reactants of a chemical reaction. Due to the existence of high energy barriers separating the states, these transition paths are unlikely to be sampled with standard Molecular Dynamics (MD) simulation. Traditional methods to augment MD with a bias potential to increase the probability of the transition rely on a dimensionality reduction step based on Collective Variables (CVs). Unfortunately, selecting appropriate CVs requires chemical intuition and traditional methods are therefore not always applicable to larger systems. Additionally, when incorrect CVs are used, the bias potential might not be minimal and bias the system along dimensions irrelevant to the transition. Showing a formal relation between the problem of sampling molecular transition paths, the Schrodinger bridge problem and stochastic optimal control with neural network policies, we propose a machine learning method for sampling said transitions. Unlike previous non-machine learning approaches our method, named PIPS, does not depend on CVs. We show that our method successful generates low energy transitions for Alanine Dipeptide as well as the larger Polyproline and Chignolin proteins.

----

## [0] RangePerception: Taming LiDAR Range View for Efficient and Accurate 3D Object Detection

**Authors**: *Yeqi Bai, Ben Fei, Youquan Liu, Tao Ma, Yuenan Hou, Botian Shi, Yikang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb8e52adcd9b59bad73f109c53afc43a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb8e52adcd9b59bad73f109c53afc43a-Abstract-Conference.html)

**Abstract**:

LiDAR-based 3D detection methods currently use bird's-eye view (BEV) or range view (RV) as their primary basis. The former relies on voxelization and 3D convolutions, resulting in inefficient training and inference processes. Conversely, RV-based methods demonstrate higher efficiency due to their compactness and compatibility with 2D convolutions, but their performance still trails behind that of BEV-based methods. To eliminate this performance gap while preserving the efficiency of RV-based methods, this study presents an efficient and accurate RV-based 3D object detection framework termed RangePerception. Through meticulous analysis, this study identifies two critical challenges impeding the performance of existing RV-based methods: 1) there exists a natural domain gap between the 3D world coordinate used in output and 2D range image coordinate used in input, generating difficulty in information extraction from range images; 2) native range images suffer from vision corruption issue, affecting the detection accuracy of the objects located on the margins of the range images. To address the key challenges above, we propose two novel algorithms named Range Aware Kernel (RAK) and Vision Restoration Module (VRM), which facilitate information flow from range image representation and world-coordinate 3D detection results. With the help of RAK and VRM, our RangePerception achieves 3.25/4.18 higher averaged L1/L2 AP compared to previous state-of-the-art RV-based method RangeDet, on Waymo Open Dataset. For the first time as an RV-based 3D detection method, RangePerception achieves slightly superior averaged AP compared with the well-known BEV-based method CenterPoint and the inference speed of RangePerception is 1.3 times as fast as CenterPoint.

----

## [0] One-for-All: Bridge the Gap Between Heterogeneous Architectures in Knowledge Distillation

**Authors**: *Zhiwei Hao, Jianyuan Guo, Kai Han, Yehui Tang, Han Hu, Yunhe Wang, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb8e5f198c7a5dcd48860354e38c0edc-Abstract-Conference.html)

**Abstract**:

Knowledge distillation (KD) has proven to be a highly effective approach for enhancing model performance through a teacher-student training scheme. However, most existing distillation methods are designed under the assumption that the teacher and student models belong to the same model family, particularly the hint-based approaches. By using centered kernel alignment (CKA) to compare the learned features between heterogeneous teacher and student models, we observe significant feature divergence. This divergence illustrates the ineffectiveness of previous hint-based methods in cross-architecture distillation. To tackle the challenge in distilling heterogeneous models, we propose a simple yet effective one-for-all KD framework called OFA-KD, which significantly improves the distillation performance between heterogeneous architectures.  Specifically, we project intermediate features into an aligned latent space such as the logits space, where architecture-specific information is discarded. Additionally, we introduce an adaptive target enhancement scheme to prevent the student from being disturbed by irrelevant information. Extensive experiments with various architectures, including CNN, Transformer, and MLP, demonstrate the superiority of our OFA-KD framework in enabling distillation between heterogeneous architectures. Specifically, when equipped with our OFA-KD, the student models achieve notable performance improvements, with a maximum gain of 8.0% on the CIFAR-100 dataset and 0.7% on the ImageNet-1K dataset. PyTorch code and checkpoints can be found at https://github.com/Hao840/OFAKD.

----

## [0] The Graph Pencil Method: Mapping Subgraph Densities to Stochastic Block Models

**Authors**: *Lee M. Gunderson, Gecia Bravo Hermsdorff, Peter Orbanz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fb9f53edbfd80b3a543f7963b63363ff-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fb9f53edbfd80b3a543f7963b63363ff-Abstract-Conference.html)

**Abstract**:

In this work, we describe a method that determines an exact map from a finite set of subgraph densities to the parameters of a stochastic block model (SBM) matching these densities. Given a number K of blocks, the subgraph densities of a finite number of stars and bistars uniquely determines a single element of the class of all degree-separated stochastic block models with K blocks. Our method makes it possible to translate estimates of these subgraph densities into model parameters, and hence to use subgraph densities directly for inference. The computational overhead is negligible; computing the translation map is polynomial in K, but independent of the graph size once the subgraph densities are given.

----

## [0] Cross-links Matter for Link Prediction: Rethinking the Debiased GNN from a Data Perspective

**Authors**: *Zihan Luo, Hong Huang, Jianxun Lian, Xiran Song, Xing Xie, Hai Jin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fba4a59c7a569fce120eea9aa9227052-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fba4a59c7a569fce120eea9aa9227052-Abstract-Conference.html)

**Abstract**:

Recently, the bias-related issues in GNN-based link prediction have raised widely spread concerns. In this paper, we emphasize the bias on links across different node clusters, which we call cross-links, after considering its significance in both easing information cocoons and preserving graph connectivity. Instead of following the objective-oriented mechanism in prior works with compromised utility, we empirically find that existing GNN models face severe data bias between internal-links (links within the same cluster) and cross-links, and this inspires us to rethink the bias issue on cross-links from a data perspective. Specifically, we design a simple yet effective twin-structure framework, which can be easily applied to most of GNNs to mitigate the bias as well as boost their utility in an end-to-end manner. The basic idea is to generate debiased node embeddings as demonstrations, and fuse them into the embeddings of original GNNs. In particular, we learn debiased node embeddings with the help of augmented supervision signals, and a novel dynamic training strategy is designed to effectively fuse debiased node embeddings with the original node embeddings. Experiments on three datasets with six common GNNs show that our framework can not only alleviate the bias between internal-links and cross-links, but also boost the overall accuracy. Comparisons with other state-of-the-art methods also verify the superiority of our method.

----

## [0] On the Robustness of Removal-Based Feature Attributions

**Authors**: *Chris Lin, Ian Covert, Su-In Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fbbda4e85a6641bf425be3a6cfd84d20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fbbda4e85a6641bf425be3a6cfd84d20-Abstract-Conference.html)

**Abstract**:

To explain predictions made by complex machine learning models, many feature attribution methods have been developed that assign importance scores to input features. Some recent work challenges the robustness of these methods by showing that they are sensitive to input and model perturbations, while other work addresses this issue by proposing robust attribution methods. However, previous work on attribution robustness has focused primarily on gradient-based feature attributions, whereas the robustness of removal-based attribution methods is not currently well understood. To bridge this gap, we theoretically characterize the robustness properties of removal-based feature attributions. Specifically, we provide a unified analysis of such methods and derive upper bounds for the difference between intact and perturbed attributions, under settings of both input and model perturbations. Our empirical results on synthetic and real-world data validate our theoretical results and demonstrate their practical implications, including the ability to increase attribution robustness by improving the modelâ€™s Lipschitz regularity.

----

## [0] Sample-efficient Multi-objective Molecular Optimization with GFlowNets

**Authors**: *Yiheng Zhu, Jialu Wu, Chaowen Hu, Jiahuan Yan, Chang-Yu Hsieh, Tingjun Hou, Jian Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fbc9981dd6316378aee7fd5975250f21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fbc9981dd6316378aee7fd5975250f21-Abstract-Conference.html)

**Abstract**:

Many crucial scientific problems involve designing novel molecules with desired properties, which can be formulated as a black-box optimization problem over the discrete chemical space. In practice, multiple conflicting objectives and costly evaluations (e.g., wet-lab experiments) make the diversity of candidates paramount. Computational methods have achieved initial success but still struggle with considering diversity in both objective and search space. To fill this gap, we propose a multi-objective Bayesian optimization (MOBO) algorithm leveraging the hypernetwork-based GFlowNets (HN-GFN) as an acquisition function optimizer, with the purpose of sampling a diverse batch of candidate molecular graphs from an approximate Pareto front. Using a single preference-conditioned hypernetwork, HN-GFN learns to explore various trade-offs between objectives. We further propose a hindsight-like off-policy strategy to share high-performing molecules among different preferences in order to speed up learning for HN-GFN. We empirically illustrate that HN-GFN has adequate capacity to generalize over preferences. Moreover, experiments in various real-world MOBO settings demonstrate that our framework predominantly outperforms existing methods in terms of candidate quality and sample efficiency. The code is available at https://github.com/violet-sto/HN-GFN.

----

## [0] DeepSimHO: Stable Pose Estimation for Hand-Object Interaction via Physics Simulation

**Authors**: *Rong Wang, Wei Mao, Hongdong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fbdaea4878318e214c0577dae4b8bc43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fbdaea4878318e214c0577dae4b8bc43-Abstract-Conference.html)

**Abstract**:

This paper addresses the task of 3D pose estimation for a hand interacting with an object from a single image observation. When modeling hand-object interaction, previous works mainly exploit proximity cues, while overlooking the dynamical nature that the hand must stably grasp the object to counteract gravity and thus preventing the object from slipping or falling. These works fail to leverage dynamical constraints in the estimation and consequently often produce unstable results. Meanwhile, refining unstable configurations with physics-based reasoning remains challenging, both by the complexity of contact dynamics and by the lack of effective and efficient physics inference in the data-driven learning framework. To address both issues, we present DeepSimHO: a novel deep-learning pipeline that combines forward physics simulation and backward gradient approximation with a neural network. Specifically, for an initial hand-object pose estimated by a base network, we forward it to a physics simulator to evaluate its stability. However, due to non-smooth contact geometry and penetration, existing differentiable simulators can not provide reliable state gradient. To remedy this, we further introduce a deep network to learn the stability evaluation process from the simulator, while smoothly approximating its gradient and thus enabling effective back-propagation. Extensive experiments show that our method noticeably improves the stability of the estimation and achieves superior efficiency over test-time optimization. The code is available at https://github.com/rongakowang/DeepSimHO.

----

## [0] Towards Data-Algorithm Dependent Generalization: a Case Study on Overparameterized Linear Regression

**Authors**: *Jing Xu, Jiaye Teng, Yang Yuan, Andrew C. Yao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fbe30aab28ad7148bc73804689ac0bd7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fbe30aab28ad7148bc73804689ac0bd7-Abstract-Conference.html)

**Abstract**:

One of the major open problems in machine learning is to characterize generalization in the overparameterized regime, where most traditional generalization bounds become inconsistent even for overparameterized linear regression. In many scenarios, this failure can be attributed to obscuring the crucial interplay between the training algorithm and the underlying data distribution. This paper demonstrate that the generalization behavior of overparameterized model should be analyzed in a both data-relevant and algorithm-relevant manner. To make a formal characterization, We introduce a notion called data-algorithm compatibility, which considers the generalization behavior of the entire data-dependent training trajectory, instead of traditional last-iterate analysis.  We validate our claim by studying the setting of solving overparameterized linear regression with gradient descent. Specifically, we perform a data-dependent trajectory analysis and derive a sufficient condition for compatibility in such a setting. Our theoretical results demonstrate that if we take early stopping iterates into consideration, generalization can hold with significantly weaker restrictions on the problem instance than the previous last-iterate analysis.

----

## [0] Global Structure-Aware Diffusion Process for Low-light Image Enhancement

**Authors**: *Jinhui Hou, Zhiyu Zhu, Junhui Hou, Hui Liu, Huanqiang Zeng, Hui Yuan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc034d186280f55370b6aca7a3285a65-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc034d186280f55370b6aca7a3285a65-Abstract-Conference.html)

**Abstract**:

This paper studies a diffusion-based framework to address the low-light image enhancement problem. To harness the capabilities of diffusion models, we delve into this intricate process and advocate for the regularization of its inherent ODE-trajectory. To be specific, inspired by the recent research that low curvature ODE-trajectory results in a stable and effective diffusion process, we formulate a curvature regularization term anchored in the intrinsic non-local structures of image data, i.e., global structure-aware regularization, which gradually facilitates the preservation of complicated details and the augmentation of contrast during the diffusion process. This incorporation mitigates the adverse effects of noise and artifacts resulting from the diffusion process, leading to a more precise and flexible enhancement. To additionally promote learning in challenging regions, we introduce an uncertainty-guided regularization technique, which wisely relaxes constraints on the most extreme regions of the image. Experimental evaluations reveal that the proposed diffusion-based framework, complemented by rank-informed regularization, attains distinguished performance in low-light enhancement. The outcomes indicate substantial advancements in image quality, noise suppression, and contrast amplification in comparison with state-of-the-art methods. We believe this innovative approach will stimulate further exploration and advancement in low-light image processing, with potential implications for other applications of diffusion models. The code is publicly available at https://github.com/jinnh/GSAD.

----

## [0] FedGCN: Convergence-Communication Tradeoffs in Federated Training of Graph Convolutional Networks

**Authors**: *Yuhang Yao, Weizhao Jin, Srivatsan Ravi, Carlee Joe-Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc07feae9af49dd3f1a1e049b77f4e17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc07feae9af49dd3f1a1e049b77f4e17-Abstract-Conference.html)

**Abstract**:

Methods for training models on graphs distributed across multiple clients have recently grown in popularity, due to the size of these graphs as well as regulations on keeping data where it is generated. However, the cross-client edges naturally exist among clients. Thus, distributed methods for training a model on a single graph incur either significant communication overhead between clients or a loss of available information to the training. We introduce the Federated Graph Convolutional Network (FedGCN) algorithm, which uses federated learning to train GCN models for semi-supervised node classification with fast convergence and little communication. Compared to prior methods that require extra communication among clients at each training round, FedGCN clients only communicate with the central server in one pre-training step, greatly reducing communication costs and allowing the use of homomorphic encryption to further enhance privacy. We theoretically analyze the tradeoff between FedGCN's convergence rate and communication cost under different data distributions. Experimental results show that our FedGCN algorithm achieves better model accuracy with 51.7\% faster convergence on average and at least 100$\times$ less communication compared to prior work.

----

## [0] SegRefiner: Towards Model-Agnostic Segmentation Refinement with Discrete Diffusion Process

**Authors**: *Mengyu Wang, Henghui Ding, Jun Hao Liew, Jiajun Liu, Yao Zhao, Yunchao Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc0cc55dca3d791c4a0bb2d8ddeefe4f-Abstract-Conference.html)

**Abstract**:

In this paper, we explore a principal way to enhance the quality of object masks produced by different segmentation models. We propose a model-agnostic solution called SegRefiner, which offers a novel perspective on this problem by interpreting segmentation refinement as a data generation process. As a result, the refinement process can be smoothly implemented through a series of denoising diffusion steps. Specifically, SegRefiner takes coarse masks as inputs and refines them using a discrete diffusion process. By predicting the label and corresponding states-transition probabilities for each pixel, SegRefiner progressively refines the noisy masks in a conditional denoising manner. To assess the effectiveness of SegRefiner, we conduct comprehensive experiments on various segmentation tasks, including semantic segmentation, instance segmentation, and dichotomous image segmentation. The results demonstrate the superiority of our SegRefiner from multiple aspects. Firstly, it consistently improves both the segmentation metrics and boundary metrics across different types of coarse masks. Secondly, it outperforms previous model-agnostic refinement methods by a significant margin. Lastly, it exhibits a strong capability to capture extremely fine details when refining high-resolution images. The source code and trained models are available at SegRefiner.git

----

## [0] On Masked Pre-training and the Marginal Likelihood

**Authors**: *Pablo Moreno-Muñoz, Pol Garcia Recasens, Søren Hauberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc0e3f908a2116ba529ad0a1530a3675-Abstract-Conference.html)

**Abstract**:

Masked pre-training removes random input dimensions and learns a model that can predict the missing values. Empirical results indicate that this intuitive form of self-supervised learning yields models that generalize very well to new domains. A theoretical understanding is, however, lacking. This paper shows that masked pre-training with a suitable cumulative scoring function corresponds to maximizing the model's marginal likelihood, which is de facto the Bayesian model selection measure of generalization. Beyond shedding light on the success of masked pre-training, this insight also suggests that Bayesian models can be trained with appropriately designed self-supervision. Empirically, we confirm the developed theory and explore the main learning principles of masked pre-training in large language models.

----

## [0] AQuA: A Benchmarking Tool for Label Quality Assessment

**Authors**: *Mononito Goswami, Vedant Sanil, Arjun Choudhry, Arvind Srinivasan, Chalisa Udompanyawit, Artur Dubrawski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc20ea8d104cab737a5561096f9bde9b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning (ML) models are only as good as the data they are trained on. But recent studies have found datasets widely used to train and evaluate ML models, e.g. ImageNet, to have pervasive labeling errors. Erroneous labels on the train set hurt ML models' ability to generalize, and they impact evaluation and model selection using the test set. Consequently, learning in the presence of labeling errors is an active area of research, yet this field lacks a comprehensive benchmark to evaluate these methods. Most of these methods are evaluated on a few computer vision datasets with significant variance in the experimental protocols. With such a large pool of methods and inconsistent evaluation, it is also unclear how ML practitioners can choose the right models to assess label quality in their data. To this end, we propose a benchmarking environment AQuA to rigorously evaluate methods that enable machine learning in the presence of label noise. We also introduce a design space to delineate concrete design choices of label error detection models. We hope that our proposed design space and benchmark enable practitioners to choose the right tools to improve their label quality and that our benchmark enables objective and rigorous evaluation of machine learning tools facing mislabeled data.

----

## [0] Smoothed Analysis of Sequential Probability Assignment

**Authors**: *Alankrita Bhatt, Nika Haghtalab, Abhishek Shetty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc30caeb45721bab13507c50199e6403-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc30caeb45721bab13507c50199e6403-Abstract-Conference.html)

**Abstract**:

We initiate the study of smoothed analysis for the sequential probability assignment problem with contexts. We study information-theoretically optimal minmax rates as well as a framework for algorithmic reduction involving the maximum likelihood estimator oracle. Our approach establishes a general-purpose reduction from minimax rates for sequential probability assignment for smoothed adversaries to minimax rates for transductive learning. This leads to optimal (logarithmic) fast rates for parametric classes and classes with finite VC dimension. On the algorithmic front, we develop an algorithm that efficiently taps into the MLE oracle, for general classes of functions. We show that under general conditions this algorithmic approach yields sublinear regret.

----

## [0] Invariant Learning via Probability of Sufficient and Necessary Causes

**Authors**: *Mengyue Yang, Yonggang Zhang, Zhen Fang, Yali Du, Furui Liu, Jean-Francois Ton, Jianhong Wang, Jun Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc657b7fd7b9aaa462f2ef9f0362b273-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc657b7fd7b9aaa462f2ef9f0362b273-Abstract-Conference.html)

**Abstract**:

Out-of-distribution (OOD) generalization is indispensable for learning models in the wild, where testing distribution typically unknown and different from the training. Recent methods derived from causality have shown great potential in achieving OOD generalization. However, existing methods mainly focus on the invariance property of causes, while largely overlooking the property of sufficiency and necessity conditions. Namely, a necessary but insufficient cause (feature) is invariant to distribution shift, yet it may not have required accuracy. By contrast, a sufficient yet unnecessary cause (feature) tends to fit specific data well but may have a risk of adapting to a new domain. To capture the information of sufficient and necessary causes, we employ a classical concept, the probability of sufficiency and necessary causes (PNS), which indicates the probability of whether one is the necessary and sufficient cause. To associate PNS with OOD generalization, we propose PNS risk and formulate an algorithm to learn representation with a high PNS value. We theoretically analyze and prove the generalizability of the PNS risk. Experiments on both synthetic and real-world benchmarks demonstrate the effectiveness of the proposed method. The detailed implementation can be found at the GitHub repository: https://github.com/ymy4323460/CaSN.

----

## [0] Reinforcement Learning for Fine-tuning Text-to-Image Diffusion Models

**Authors**: *Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, Kimin Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc65fab891d83433bd3c8d966edde311-Abstract-Conference.html)

**Abstract**:

Learning from human feedback has been shown to improve text-to-image models. These techniques first learn a reward function that captures what humans care about in the task and then improve the models based on the learned reward function. Even though relatively simple approaches (e.g., rejection sampling based on reward scores) have been investigated, fine-tuning text-to-image models with the reward function remains challenging. In this work, we propose using online reinforcement learning (RL) to fine-tune text-to-image models. We focus on diffusion models, defining the fine-tuning task as an RL problem, and updating the pre-trained text-to-image diffusion models using policy gradient to maximize the feedback-trained reward. Our approach, coined DPOK, integrates policy optimization with KL regularization. We conduct an analysis of KL regularization for both RL fine-tuning and supervised fine-tuning. In our experiments, we show that DPOK is generally superior to supervised fine-tuning with respect to both image-text alignment and image quality. Our code is available at https://github.com/google-research/google-research/tree/master/dpok.

----

## [0] Online POMDP Planning with Anytime Deterministic Guarantees

**Authors**: *Moran Barenboim, Vadim Indelman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc6bd0eef19459655d5b097af783661d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc6bd0eef19459655d5b097af783661d-Abstract-Conference.html)

**Abstract**:

Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that iseasier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior node. Then, since a complete belief update may be computationally demanding, we extend the bounds to support reduction of both the state and the observation spaces. We demonstrate how our guarantees can be integrated with existing state-of-the-art solvers that sample a subset of states and observations. As a result, the returned solution holds deterministic bounds relative to the optimal policy. Lastly, we substantiate our findings with supporting experimental results.

----

## [0] The Curious Price of Distributional Robustness in Reinforcement Learning with a Generative Model

**Authors**: *Laixi Shi, Gen Li, Yuting Wei, Yuxin Chen, Matthieu Geist, Yuejie Chi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fc8ee7c7ab5b5f6b1615045dfb617ed6-Abstract-Conference.html)

**Abstract**:

This paper investigates model robustness in reinforcement learning (RL) via the framework of distributionally robust Markov decision processes (RMDPs). Despite recent efforts, the sample complexity of RMDPs is much less understood regardless of the uncertainty set in use; in particular, there exist large gaps between existing upper and lower bounds, and it is unclear if distributional robustness bears any statistical implications when benchmarked against standard RL. In this paper, assuming access to a generative model, we derive the sample complexity of RMDPs---when the uncertainty set is measured via either total variation or $\chi^2$ divergence over the full range of uncertainty levels---using a model-based algorithm called distributionally robust value iteration, and develop  minimax lower bounds to benchmark its tightness. Our results not only strengthen the prior art in both directions of upper and lower bounds, but also deliver surprising messages that learning RMDPs is not necessarily easier or more difficult than standard MDPs. In the case of total variation, we establish the minimax-optimal sample complexity of RMDPs which is always smaller than that of standard MDPs. In the case of $\chi^2$ divergence, we establish the sample complexity of RMDPs that is tight up to polynomial factors of the effective horizon, and grows linearly with respect to the uncertainty level when it approaches infinity.

----

## [0] Strategic Apple Tasting

**Authors**: *Keegan Harris, Chara Podimata, Steven Z. Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/fcd3909db30887ce1da519c4468db668-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/fcd3909db30887ce1da519c4468db668-Abstract-Conference.html)

**Abstract**:

Algorithmic decision-making in high-stakes domains often involves assigning decisions to agents with incentives to strategically modify their input to the algorithm. In addition to dealing with incentives, in many domains of interest (e.g. lending and hiring) the decision-maker only observes feedback regarding their policy for rounds in which they assign a positive decision to the agent; this type of feedback is often referred to as apple tasting (or one-sided) feedback. We formalize this setting as an online learning problem with apple-tasting feedback where a principal makes decisions about a sequence of $T$ agents, each of which is represented by a context that may be strategically modified. Our goal is to achieve sublinear strategic regret, which compares the performance of the principal to that of the best fixed policy in hindsight, if the agents were truthful when revealing their contexts. Our main result is a learning algorithm which incurs $\tilde{\mathcal{O}}(\sqrt{T})$ strategic regret when the sequence of agents is chosen stochastically. We also give an algorithm capable of handling adversarially-chosen agents, albeit at the cost of $\tilde{\mathcal{O}}(T^{(d+1)/(d+2)})$ strategic regret (where $d$ is the dimension of the context). Our algorithms can be easily adapted to the setting where the principal receives bandit feedback---this setting generalizes both the linear contextual bandit problem (by considering agents with incentives) and the strategic classification problem (by allowing for partial feedback).

----



[Go to the previous page](NIPS-2023-list3400.md)

[Go to the next page](NIPS-2023-list3402.md)

[Go to the catalog section](README.md)