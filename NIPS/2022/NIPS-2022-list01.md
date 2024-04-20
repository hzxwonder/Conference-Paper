## [0] Federated Submodel Optimization for Hot and Cold Data Features

        **Authors**: *Yucheng Ding, Chaoyue Niu, Fan Wu, Shaojie Tang, Chengfei Lyu, Yanghe Feng, Guihai Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/002262941c9edfd472a79298b2ac5e17-Abstract-Conference.html)

        **Abstract**:

        We focus on federated learning in practical recommender systems and natural language processing scenarios. The global model for federated optimization typically contains a large and sparse embedding layer, while each client’s local data tend to interact with part of features, updating only a small submodel with the feature-related embedding vectors. We identify a new and important issue that distinct data features normally involve different numbers of clients, generating the differentiation of hot and cold features. We further reveal that the classical federated averaging algorithm (FedAvg) or its variants, which randomly selects clients to participate and uniformly averages their submodel updates, will be severely slowed down, because different parameters of the global model are optimized at different speeds. More specifically, the model parameters related to hot (resp., cold) features will be updated quickly (resp., slowly). We thus propose federated submodel averaging (FedSubAvg), which introduces the number of feature-related clients as the metric of feature heat to correct the aggregation of submodel updates. We prove that due to the dispersion of feature heat, the global objective is ill-conditioned, and FedSubAvg works as a suitable diagonal preconditioner. We also rigorously analyze FedSubAvg’s convergence rate to stationary points. We finally evaluate FedSubAvg over several public and industrial datasets. The evaluation results demonstrate that FedSubAvg significantly outperforms FedAvg and its variants.

        ----

        ## [1] On Kernelized Multi-Armed Bandits with Constraints

        **Authors**: *Xingyu Zhou, Bo Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/00295cede6e1600d344b5cd6d9fd4640-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/00295cede6e1600d344b5cd6d9fd4640-Abstract-Conference.html)

        **Abstract**:

        We study a stochastic bandit problem with a general unknown reward function and a general unknown constraint function. Both functions can be non-linear (even non-convex) and are assumed to lie in a reproducing kernel Hilbert space (RKHS) with a bounded norm. This kernelized bandit setup strictly generalizes standard multi-armed bandits and linear bandits. In contrast to safety-type hard constraints studied in prior works, we consider soft constraints that may be violated in any round as long as the cumulative violations are small, which is motivated by various practical applications. Our ultimate goal is to study how to utilize the nature of soft constraints to attain a finer complexity-regret-constraint trade-off in the kernelized bandit setting. To this end, leveraging primal-dual optimization, we propose a general framework for both algorithm design and performance analysis. This framework builds upon a novel sufficient condition, which not only is satisfied under general exploration strategies, including \emph{upper confidence bound} (UCB), \emph{Thompson sampling} (TS), and new ones based on \emph{random exploration}, but also enables a unified analysis for showing both sublinear regret and sublinear or even zero constraint violation. We demonstrate the superior performance of our proposed algorithms via numerical experiments based on both synthetic and real-world datasets. Along the way, we also make the first detailed comparison between two popular methods for analyzing constrained bandits and Markov decision processes (MDPs) by discussing the key difference and some subtleties in the analysis, which could be of independent interest to the communities.

        ----

        ## [2] Geometric Order Learning for Rank Estimation

        **Authors**: *Seon-Ho Lee, Nyeong-Ho Shin, Chang-Su Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/00358de35a101a372ea0412bed913c86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/00358de35a101a372ea0412bed913c86-Abstract-Conference.html)

        **Abstract**:

        A novel approach to rank estimation, called geometric order learning (GOL), is proposed in this paper. First, we construct an embedding space, in which the direction and distance between objects represent order and metric relations between their ranks, by enforcing two geometric constraints: the order constraint compels objects to be sorted according to their ranks, while the metric constraint makes the distance between objects reflect their rank difference. Then, we perform the simple $k$ nearest neighbor ($k$-NN) search in the embedding space to estimate the rank of a test object. Moreover, to assess the quality of embedding spaces for rank estimation, we propose a metric called discriminative ratio for ranking (DRR). Extensive experiments on facial age estimation, historical color image (HCI) classification, and aesthetic score regression demonstrate that GOL constructs effective embedding spaces and thus yields excellent rank estimation performances. The source codes are available at https://github.com/seon92/GOL

        ----

        ## [3] Structured Recognition for Generative Models with Explaining Away

        **Authors**: *Changmin Yu, Hugo Soulat, Neil Burgess, Maneesh Sahani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/003a96110b7134d678cb675c6aea6c7d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/003a96110b7134d678cb675c6aea6c7d-Abstract-Conference.html)

        **Abstract**:

        A key goal of unsupervised learning is to go beyond density estimation and sample generation to reveal the structure inherent within observed data. Such structure can be expressed in the pattern of interactions between explanatory latent variables captured through a probabilistic graphical model. Although the learning of structured graphical models has a long history, much recent work in unsupervised modelling has instead emphasised flexible deep-network-based generation, either transforming independent latent generators to model complex data or assuming that distinct observed variables are derived from different latent nodes. Here, we extend amortised variational inference to incorporate structured factors over multiple variables, able to capture the observation-induced posterior dependence between latents that results from “explaining away” and thus allow complex observations to depend on multiple nodes of a structured graph. We show that appropriately parametrised factors can be combined efficiently with variational message passing in rich graphical structures. We instantiate the framework in nonlinear Gaussian Process Factor Analysis, evaluating the structured recognition framework using synthetic data from known generative processes. We fit the GPFA model to high-dimensional neural spike data from the hippocampus of freely moving rodents, where the model successfully identifies latent signals that correlate with behavioural covariates.

        ----

        ## [4] NAS-Bench-Graph: Benchmarking Graph Neural Architecture Search

        **Authors**: *Yijian Qin, Ziwei Zhang, Xin Wang, Zeyang Zhang, Wenwu Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/004bed4e186fdd7ebb73aad6e97c2332-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/004bed4e186fdd7ebb73aad6e97c2332-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Graph neural architecture search (GraphNAS) has recently aroused considerable attention in both academia and industry. However, two key challenges seriously hinder the further research of GraphNAS. First, since there is no consensus for the experimental setting, the empirical results in different research papers are often not comparable and even not reproducible, leading to unfair comparisons. Secondly, GraphNAS often needs extensive computations, which makes it highly inefficient and inaccessible to researchers without access to large-scale computation. To solve these challenges, we propose NAS-Bench-Graph, a tailored benchmark that supports unified, reproducible, and efficient evaluations for GraphNAS. Specifically, we construct a unified, expressive yet compact search space, covering 26,206 unique graph neural network (GNN) architectures and propose a principled evaluation protocol. To avoid unnecessary repetitive training, we have trained and evaluated all of these architectures on nine representative graph datasets, recording detailed metrics including train, validation, and test performance in each epoch, the latency, the number of parameters, etc. Based on our proposed benchmark, the performance of GNN architectures can be directly obtained by a look-up table without any further computation, which enables fair, fully reproducible, and efficient comparisons.  To demonstrate its usage, we make in-depth analyses of our proposed NAS-Bench-Graph, revealing several interesting findings for GraphNAS. We also showcase how the benchmark can be easily compatible with GraphNAS open libraries such as AutoGL and NNI. To the best of our knowledge, our work is the first benchmark for graph neural architecture search.

        ----

        ## [5] Fast Bayesian Coresets via Subsampling and Quasi-Newton Refinement

        **Authors**: *Cian Naik, Judith Rousseau, Trevor Campbell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/005413e90d003d13886019607b037f52-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/005413e90d003d13886019607b037f52-Abstract-Conference.html)

        **Abstract**:

        Bayesian coresets approximate a posterior distribution by building a small weighted subset of the data points. Any inference procedure that is too computationally expensive to be run on the full posterior can instead be run inexpensively on the coreset, with results that approximate those on the full data. However, current approaches are limited by either a significant run-time or the need for the user to specify a low-cost approximation to the full posterior. We propose a Bayesian coreset construction algorithm that first selects a uniformly random subset of data, and then optimizes the weights using a novel quasi-Newton method. Our algorithm is a simple to implement, black-box method, that does not require the user to specify a low-cost posterior approximation. It is the first to come with a general high-probability bound on the KL divergence of the output coreset posterior. Experiments demonstrate that our method provides significant improvements in coreset quality against alternatives with comparable construction times, with far less storage cost and user input required.

        ----

        ## [6] What You See is What You Classify: Black Box Attributions

        **Authors**: *Steven Stalder, Nathanaël Perraudin, Radhakrishna Achanta, Fernando Pérez-Cruz, Michele Volpi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0073cc73e1873b35345209b50a3dab66-Abstract-Conference.html)

        **Abstract**:

        An important step towards explaining deep image classifiers lies in the identification of image regions that contribute to individual class scores in the model's output. However, doing this accurately is a difficult task due to the black-box nature of such networks. Most existing approaches find such attributions either using activations and gradients or by repeatedly perturbing the input. We instead address this challenge by training a second deep network, the Explainer, to predict attributions for a pre-trained black-box classifier, the Explanandum. These attributions are provided in the form of masks that only show the classifier-relevant parts of an image, masking out the rest. Our approach produces sharper and more boundary-precise masks when compared to the saliency maps generated by other methods. Moreover, unlike most existing approaches, ours is capable of directly generating very distinct class-specific masks in a single forward pass. This makes the proposed method very efficient during inference. We show that our attributions are superior to established methods both visually and quantitatively with respect to the PASCAL VOC-2007 and Microsoft COCO-2014 datasets.

        ----

        ## [7] Adaptive Interest for Emphatic Reinforcement Learning

        **Authors**: *Martin Klissarov, Rasool Fakoor, Jonas W. Mueller, Kavosh Asadi, Taesup Kim, Alexander J. Smola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/008079ec00eec9760ee93af5434ee932-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/008079ec00eec9760ee93af5434ee932-Abstract-Conference.html)

        **Abstract**:

        Emphatic algorithms have shown great promise in stabilizing and improving reinforcement learning by selectively emphasizing the update rule. Although the emphasis fundamentally depends on an interest function which defines the intrinsic importance of each state, most approaches simply adopt a uniform interest over all states (except where a hand-designed interest is possible based on domain knowledge). In this paper, we investigate adaptive methods that allow the interest function to dynamically vary over states and iterations. In particular, we leverage meta-gradients to automatically discover online an interest function that would accelerate the agentâ€™s learning process. Empirical evaluations on a wide range of environments show that adapting the interest is key to provide significant gains. Qualitative analysis indicates that the learned interest function emphasizes states of particular importance, such as bottlenecks, which can be especially useful in a transfer learning setting.

        ----

        ## [8] Scaling & Shifting Your Features: A New Baseline for Efficient Model Tuning

        **Authors**: *Dongze Lian, Daquan Zhou, Jiashi Feng, Xinchao Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/00bb4e415ef117f2dee2fc3b778d806d-Abstract-Conference.html)

        **Abstract**:

        Existing fine-tuning methods either tune all parameters of the pre-trained model (full fine-tuning), which is not efficient, or only tune the last linear layer (linear probing), which suffers a significant accuracy drop compared to the full fine-tuning. In this paper, we propose a new parameter-efficient fine-tuning method termed as SSF, representing that researchers only need to Scale and Shift the deep Features extracted by a pre-trained model to catch up with the performance of full fine-tuning. In this way, SSF also surprisingly outperforms other parameter-efficient fine-tuning approaches even with a smaller number of tunable parameters. Furthermore, different from some existing parameter-efficient fine-tuning methods (e.g., Adapter or VPT) that introduce the extra parameters and computational cost in the training and inference stages, SSF only adds learnable parameters during the training stage, and these additional parameters can be merged into the original pre-trained model weights via re-parameterization in the inference phase. With the proposed SSF, our model obtains 2.46% (90.72% vs. 88.54%) and 11.48% (73.10% vs. 65.57%) performance improvement on FGVC and VTAB-1k in terms of Top-1 accuracy compared to the full fine-tuning but only fine-tuning about 0.3M parameters. We also conduct amounts of experiments in various model families (CNNs, Transformers, and MLPs) and datasets. Results on 26 image classification datasets in total and 3 robustness & out-of-distribution datasets show the effectiveness of SSF. Code is available at https://github.com/dongzelian/SSF.

        ----

        ## [9] Zero-Shot Video Question Answering via Frozen Bidirectional Language Models

        **Authors**: *Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, Cordelia Schmid*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/00d1f03b87a401b1c7957e0cc785d0bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/00d1f03b87a401b1c7957e0cc785d0bc-Abstract-Conference.html)

        **Abstract**:

        Video question answering (VideoQA) is a complex task that requires diverse multi-modal data for training. Manual annotation of question and answers for videos, however, is tedious and prohibits scalability. To tackle this problem, recent methods consider zero-shot settings with no manual annotation of visual question-answer. In particular, a promising approach adapts frozen autoregressive language models pretrained on Web-scale text-only data to multi-modal inputs. In contrast, we here build on frozen bidirectional language models (BiLM) and show that such an approach provides a stronger and cheaper alternative for zero-shot VideoQA. In particular, (i) we combine visual inputs with the frozen BiLM using light trainable modules, (ii) we train such modules using Web-scraped multi-modal data, and finally (iii) we perform zero-shot VideoQA inference through masked language modeling, where the masked text is the answer to a given question. Our proposed approach, FrozenBiLM, outperforms the state of the art in zero-shot VideoQA by a significant margin on a variety of datasets, including LSMDC-FiB, iVQA, MSRVTT-QA, MSVD-QA, ActivityNet-QA, TGIF-FrameQA, How2QA and TVQA. It also demonstrates competitive performance in the few-shot and fully-supervised setting. Our code and models are publicly available at https://github.com/antoyang/FrozenBiLM.

        ----

        ## [10] Active Learning with Neural Networks: Insights from Nonparametric Statistics

        **Authors**: *Yinglun Zhu, Robert Nowak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01025a4e79355bb37a10ba39605944b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01025a4e79355bb37a10ba39605944b5-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks have great representation power, but typically require large numbers of training examples. This motivates deep active learning methods that can significantly reduce the amount of labeled training data. Empirical successes of deep active learning have been recently reported in the literature, however, rigorous label complexity guarantees of deep active learning have remained elusive. This constitutes a significant gap between theory and practice. This paper tackles this gap by providing the first near-optimal label complexity guarantees for deep active learning. The key insight is to study deep active learning from the nonparametric classification perspective. Under standard low noise conditions, we show that active learning with neural networks can provably achieve the minimax label complexity, up to disagreement coefficient and other logarithmic terms. When equipped with an abstention option, we further develop an efficient deep active learning algorithm that achieves $\mathsf{polylog}(\frac{1}{\varepsilon})$ label complexity, without any low noise assumptions.  We also provide extensions of our results beyond the commonly studied Sobolev/H\"older spaces and develop label complexity guarantees for learning in Radon $\mathsf{BV}^2$ spaces, which have recently been proposed as natural function spaces associated with neural networks.

        ----

        ## [11] IM-Loss: Information Maximization Loss for Spiking Neural Networks

        **Authors**: *Yufei Guo, Yuanpei Chen, Liwen Zhang, Xiaode Liu, YingLei Wang, Xuhui Huang, Zhe Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/010c5ba0cafc743fece8be02e7adb8dd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/010c5ba0cafc743fece8be02e7adb8dd-Abstract-Conference.html)

        **Abstract**:

        Spiking Neural Network (SNN), recognized as a type of biologically plausible architecture, has recently drawn much research attention. It transmits information by $0/1$ spikes. This bio-mimetic mechanism of SNN demonstrates extreme energy efficiency since it avoids any multiplications on neuromorphic hardware. However, the forward-passing $0/1$ spike quantization will cause information loss and accuracy degradation. To deal with this problem, the Information maximization loss (IM-Loss) that aims at maximizing the information flow in the SNN is proposed in the paper. The IM-Loss not only enhances the information expressiveness of an SNN directly but also plays a part of the role of normalization without introducing any additional operations (\textit{e.g.}, bias and scaling) in the inference phase. Additionally, we introduce a novel differentiable spike activity estimation, Evolutionary Surrogate Gradients (ESG) in SNNs. By appointing automatic evolvable surrogate gradients for spike activity function, ESG can ensure sufficient model updates at the beginning and accurate gradients at the end of the training, resulting in both easy convergence and high task performance. Experimental results on both popular non-spiking static and neuromorphic datasets show that the SNN models trained by our method outperform the current state-of-the-art algorithms.

        ----

        ## [12] Using natural language and program abstractions to instill human inductive biases in machines

        **Authors**: *Sreejan Kumar, Carlos G. Correa, Ishita Dasgupta, Raja Marjieh, Michael Y. Hu, Robert D. Hawkins, Jonathan D. Cohen, Nathaniel D. Daw, Karthik Narasimhan, Tom Griffiths*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html)

        **Abstract**:

        Strong inductive biases give humans the ability to quickly learn to perform a variety of tasks. Although meta-learning is a method to endow neural networks with useful inductive biases, agents trained by meta-learning may sometimes acquire very different strategies from humans. We show that co-training these agents on predicting representations from natural language task descriptions and programs induced to generate such tasks guides them toward more human-like inductive biases. Human-generated language descriptions and program induction models that add new learned primitives both contain abstract concepts that can compress description length. Co-training on these representations result in more human-like behavior in downstream meta-reinforcement learning agents than less abstract controls (synthetic language descriptions, program induction without learned primitives), suggesting that the abstraction supported by these representations is key.

        ----

        ## [13] Second Thoughts are Best: Learning to Re-Align With Human Values from Text Edits

        **Authors**: *Ruibo Liu, Chenyan Jia, Ge Zhang, Ziyu Zhuang, Tony X. Liu, Soroush Vosoughi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01c4593d60a020fed5607944330106b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01c4593d60a020fed5607944330106b1-Abstract-Conference.html)

        **Abstract**:

        We present Second Thoughts, a new learning paradigm that enables language models (LMs) to re-align with human values. By modeling the chain-of-edits between value-unaligned and value-aligned text, with LM fine-tuning and additional refinement through reinforcement learning, Second Thoughts not only achieves superior performance in three value alignment benchmark datasets but also shows strong human-value transfer learning ability in few-shot scenarios. The generated editing steps also offer better interpretability and ease for interactive error correction. Extensive human evaluations further confirm its effectiveness.

        ----

        ## [14] SatMAE: Pre-training Transformers for Temporal and Multi-Spectral Satellite Imagery

        **Authors**: *Yezhen Cong, Samar Khanna, Chenlin Meng, Patrick Liu, Erik Rozi, Yutong He, Marshall Burke, David B. Lobell, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html)

        **Abstract**:

        Unsupervised pre-training methods for large vision models have shown to enhance performance on downstream supervised tasks. Developing similar techniques for satellite imagery presents significant opportunities as unlabelled data is plentiful and the inherent temporal and multi-spectral structure provides avenues to further improve existing pre-training strategies. In this paper, we present SatMAE, a pre-training framework for temporal or multi-spectral satellite imagery based on Masked Autoencoder (MAE). To leverage temporal information,  we include a temporal embedding along with independently masking image patches across time. In addition, we demonstrate that encoding multi-spectral data as groups of bands with distinct spectral positional encodings is beneficial. Our approach yields strong improvements over previous state-of-the-art techniques, both in terms of supervised learning performance on benchmark datasets (up to $\uparrow$ 7%), and transfer learning performance on downstream remote sensing tasks, including land cover classification (up to $\uparrow$ 14%) and semantic segmentation. Code and data are available on the project website: https://sustainlab-group.github.io/SatMAE/

        ----

        ## [15] On Sample Optimality in Personalized Collaborative and Federated Learning

        **Authors**: *Mathieu Even, Laurent Massoulié, Kevin Scaman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01cea7793f3c68af2e4989fc66bf8fb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01cea7793f3c68af2e4989fc66bf8fb0-Abstract-Conference.html)

        **Abstract**:

        In personalized federated learning, each member of a potentially large set of agents aims to train a model minimizing its loss function averaged over its local data distribution. We study this problem under the lens of stochastic optimization, focusing on a scenario with a large number of agents, that each possess very few data samples from their local data distribution. Specifically, we prove novel matching lower and upper bounds on the number of samples required from all agents to approximately minimize the generalization error of a fixed agent. We provide strategies matching these lower bounds, based on a gradient filtering approach: given prior knowledge on some notion of distance between local data distributions, agents filter and aggregate stochastic gradients received from other agents, in order to achieve an optimal bias-variance trade-off. Finally, we quantify the impact of using rough estimations of the distances between local distributions of agents, based on a very small number of local samples.

        ----

        ## [16] Offline Multi-Agent Reinforcement Learning with Knowledge Distillation

        **Authors**: *Wei-Cheng Tseng, Tsun-Hsuan Johnson Wang, Yen-Chen Lin, Phillip Isola*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html)

        **Abstract**:

        We introduce an offline multi-agent reinforcement learning ( offline MARL) framework that utilizes previously collected data without additional online data collection. Our method reformulates offline MARL as a sequence modeling problem and thus builds on top of the simplicity and scalability of the Transformer architecture. In the fashion of centralized training and decentralized execution, we propose to first train a teacher policy as if the MARL dataset is generated by a single agent. After the teacher policy has identified and recombined the "good" behavior in the dataset, we create separate student policies and distill not only the teacher policy's features but also its structural relations among different agents' features to student policies. Despite its simplicity, the proposed method outperforms state-of-the-art model-free offline MARL baselines while being more robust to demonstration's quality on several environments.

        ----

        ## [17] Decentralized Gossip-Based Stochastic Bilevel Optimization over Communication Networks

        **Authors**: *Shuoguang Yang, Xuezhou Zhang, Mengdi Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01db36a646c07c64dd39a92b4eceb417-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01db36a646c07c64dd39a92b4eceb417-Abstract-Conference.html)

        **Abstract**:

        Bilevel optimization have gained growing interests, with numerous applications found in meta learning, minimax games, reinforcement learning, and nested composition optimization. This paper studies the problem of decentralized distributed bilevel optimization over a network where agents can only communicate with neighbors, and gives examples from multi-task, multi-agent learning and federated learning.In this paper, we propose a gossip-based distributed bilevel learning algorithm that allows networked agents to solve both the inner and outer optimization problems in a single timescale and share information through network propagation. We show that our algorithm enjoys the $\mathcal{O}(\frac{1}{K \epsilon^2})$ per-agent sample complexity for general nonconvex bilevel optimization and $\mathcal{O}(\frac{1}{K \epsilon})$ for Polyak-≈Åojasiewicz objective, achieving a speedup that scales linearly with the network size $K$. The sample complexities are optimal in both $\epsilon$ and $K$.We test our algorithm on the examples of hyperparameter tuning and decentralized reinforcement learning. Simulated experiments confirmed that our algorithm achieves the state-of-the-art training efficiency and test accuracy.

        ----

        ## [18] Conditional Meta-Learning of Linear Representations

        **Authors**: *Giulia Denevi, Massimiliano Pontil, Carlo Ciliberto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/01ecd39ca49ddecc5729ca996304781b-Abstract-Conference.html)

        **Abstract**:

        Standard meta-learning for representation learning aims to find a common representation to be shared across multiple tasks. The effectiveness of these methods is often limited when the nuances of the tasks’ distribution cannot be captured by a single representation. In this work we overcome this issue by inferring a conditioning function, mapping the tasks’ side information (such as the tasks’ training dataset itself) into a representation tailored to the task at hand. We study environments in which our conditional strategy outperforms standard meta-learning, such as those in which tasks can be organized in separate clusters according to the representation they share. We then propose a meta-algorithm capable of leveraging this advantage in practice. In the unconditional setting, our method yields a new estimator enjoying faster learning rates and requiring less hyper-parameters to tune than current state-of-the-art methods. Our results are supported by preliminary experiments.

        ----

        ## [19] Theory and Approximate Solvers for Branched Optimal Transport with Multiple Sources

        **Authors**: *Peter Lippmann, Enrique Fita Sanmartín, Fred A. Hamprecht*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0206c1c20a18915da23df5e61966fc6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0206c1c20a18915da23df5e61966fc6a-Abstract-Conference.html)

        **Abstract**:

        Branched optimal transport (BOT) is a generalization of optimal transport in which transportation costs along an edge are subadditive. This subadditivity models an increase in transport efficiency when shipping mass along the same route, favoring branched transportation networks. We here study the NP-hard optimization of BOT networks connecting a finite number of sources and sinks in $\mathbb{R}^2$. First, we show how to efficiently find the best geometry of a BOT network for many sources and sinks, given a topology. Second, we argue that a topology with more than three edges meeting at a branching point is never optimal. Third, we show that the results obtained for the Euclidean plane generalize directly to optimal transportation networks on two-dimensional Riemannian manifolds. Finally, we present a simple but effective approximate BOT solver combining geometric optimization with a combinatorial optimization of the network topology.

        ----

        ## [20] CHIMLE: Conditional Hierarchical IMLE for Multimodal Conditional Image Synthesis

        **Authors**: *Shichong Peng, Seyed Alireza Moazenipourasil, Ke Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0207c9ea9faf66c6e892c3fa3c167b75-Abstract-Conference.html)

        **Abstract**:

        A persistent challenge in conditional image synthesis has been to generate diverse output images from the same input image despite only one output image being observed per input image. GAN-based methods are prone to mode collapse, which leads to low diversity. To get around this, we leverage Implicit Maximum Likelihood Estimation (IMLE) which can overcome mode collapse fundamentally. IMLE uses the same generator as GANs but trains it with a different, non-adversarial objective which ensures each observed image has a generated sample nearby. Unfortunately, to generate high-fidelity images, prior IMLE-based methods require a large number of samples, which is expensive. In this paper, we propose a new method to get around this limitation, which we dub Conditional Hierarchical IMLE (CHIMLE), which can generate high-fidelity images without requiring many samples. We show CHIMLE significantly outperforms the prior best IMLE, GAN and diffusion-based methods in terms of image fidelity and mode coverage across four tasks, namely night-to-day, 16x single image super-resolution, image colourization and image decompression. Quantitatively, our method improves Fr√©chet Inception Distance (FID) by 36.9% on average compared to the prior best IMLE-based method, and by 27.5% on average compared to the best non-IMLE-based general-purpose methods. More results and code are available on the project website at https://niopeng.github.io/CHIMLE/.

        ----

        ## [21] Active Ranking without Strong Stochastic Transitivity

        **Authors**: *Hao Lou, Tao Jin, Yue Wu, Pan Xu, Quanquan Gu, Farzad Farnoud*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/020e313d40a7c060ed07a10cef287750-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/020e313d40a7c060ed07a10cef287750-Abstract-Conference.html)

        **Abstract**:

        Ranking from noisy comparisons is of great practical interest in machine learning. In this paper, we consider the problem of recovering the exact full ranking for a list of items under ranking models that do *not* assume the Strong Stochastic Transitivity property. We propose a $$\delta$$-correct algorithm, Probe-Rank, that actively learns the ranking of the items from noisy pairwise comparisons. We prove a sample complexity upper bound for Probe-Rank, which only depends on the preference probabilities between items that are adjacent in the true ranking. This improves upon existing sample complexity results that depend on the preference probabilities for all pairs of items. Probe-Rank thus outperforms existing methods over a large collection of instances that do not satisfy Strong Stochastic Transitivity. Thorough numerical experiments in various settings are conducted, demonstrating that Probe-Rank is significantly more sample-efficient than the state-of-the-art active ranking method.

        ----

        ## [22] Offline Goal-Conditioned Reinforcement Learning via $f$-Advantage Regression

        **Authors**: *Yecheng Jason Ma, Jason Yan, Dinesh Jayaraman, Osbert Bastani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/022a39052abf9ca467e268923057dfc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/022a39052abf9ca467e268923057dfc0-Abstract-Conference.html)

        **Abstract**:

        Offline goal-conditioned reinforcement learning (GCRL) promises general-purpose skill learning in the form of reaching diverse goals from purely offline datasets. We propose $\textbf{Go}$al-conditioned $f$-$\textbf{A}$dvantage $\textbf{R}$egression (GoFAR), a novel regression-based offline GCRL algorithm derived from a state-occupancy matching perspective; the key intuition is that the goal-reaching task can be formulated as a state-occupancy matching problem between a dynamics-abiding imitator agent and an expert agent that directly teleports to the goal. In contrast to prior approaches, GoFAR does not require any hindsight relabeling and enjoys uninterleaved optimization for its value and policy networks. These distinct features confer GoFAR with much better offline performance and stability as well as statistical performance guarantee that is unattainable for prior methods. Furthermore, we demonstrate that GoFAR's training objectives can be re-purposed to learn an agent-independent goal-conditioned planner from purely offline source-domain data, which enables zero-shot transfer to new target domains. Through extensive experiments, we validate GoFAR's effectiveness in various problem settings and tasks, significantly outperforming prior state-of-art. Notably, on a real robotic dexterous manipulation task, while no other method makes meaningful progress, GoFAR acquires complex manipulation behavior that successfully accomplishes diverse goals.

        ----

        ## [23] Rethinking and Improving Robustness of Convolutional Neural Networks: a Shapley Value-based Approach in Frequency Domain

        **Authors**: *Yiting Chen, Qibing Ren, Junchi Yan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/022abe84083d235f7572ca5cba24c51c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/022abe84083d235f7572ca5cba24c51c-Abstract-Conference.html)

        **Abstract**:

        The existence of adversarial examples poses concerns for the robustness of convolutional neural networks (CNN), for which a popular hypothesis is about the frequency bias phenomenon: CNNs rely more on high-frequency components (HFC) for classification than humans, which causes the brittleness of CNNs. However, most previous works manually select and roughly divide the image frequency spectrum and conduct qualitative analysis. In this work, we introduce Shapley value, a metric of cooperative game theory, into the frequency domain and propose to quantify the positive (negative) impact of every frequency component of data on CNNs. Based on the Shapley value, we quantify the impact in a fine-grained way and show intriguing instance disparity. Statistically, we investigate adversarial training(AT) and the adversarial attack in the frequency domain. The observations motivate us to perform an in-depth analysis and lead to multiple novel hypotheses about i) the cause of adversarial robustness of the AT model; ii) the fairness problem of AT between different classes in the same dataset; iii) the attack bias on different frequency components. Finally, we propose a Shapley-value guided data augmentation technique for improving the robustness. Experimental results on image classification benchmarks show its effectiveness.

        ----

        ## [24] Adversarial Style Augmentation for Domain Generalized Urban-Scene Segmentation

        **Authors**: *Zhun Zhong, Yuyang Zhao, Gim Hee Lee, Nicu Sebe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/023d94f44110b9a3c62329beec739772-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/023d94f44110b9a3c62329beec739772-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider the problem of domain generalization in semantic segmentation, which aims to learn a robust model using only labeled synthetic (source) data. The model is expected to perform well on unseen real (target) domains. Our study finds that the image style variation can largely influence the model's performance and the style features can be well represented by the channel-wise mean and standard deviation of images. Inspired by this, we propose a novel adversarial style augmentation (AdvStyle) approach, which can dynamically generate hard stylized images during training and thus can effectively prevent the model from overfitting on the source domain. Specifically, AdvStyle regards the style feature as a learnable parameter and updates it by adversarial training. The learned adversarial style feature is used to construct an adversarial image for robust model training. AdvStyle is easy to implement and can be readily applied to different models. Experiments on two synthetic-to-real semantic segmentation benchmarks demonstrate that AdvStyle can significantly improve the model performance on unseen real domains and show that we can achieve the state of the art. Moreover, AdvStyle can be employed to domain generalized image classification and produces a clear improvement on the considered datasets.

        ----

        ## [25] Fully Sparse 3D Object Detection

        **Authors**: *Lue Fan, Feng Wang, Naiyan Wang, Zhaoxiang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0247fa3c511bbc415c8b768ee7b32f9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0247fa3c511bbc415c8b768ee7b32f9e-Abstract-Conference.html)

        **Abstract**:

        As the perception range of LiDAR increases, LiDAR-based 3D object detection becomes a dominant task in the long-range perception task of autonomous driving. The mainstream 3D object detectors usually build dense feature maps in the network backbone and prediction head. However, the computational and spatial costs on the dense feature map are quadratic to the perception range, which makes them hardly scale up to the long-range setting. To enable efficient long-range LiDAR-based object detection, we build a fully sparse 3D object detector (FSD). The computational and spatial cost of FSD is roughly linear to the number of points and independent of the perception range. FSD is built upon the general sparse voxel encoder and a novel sparse instance recognition (SIR) module.  SIR first groups the points into instances and then applies instance-wise feature extraction and prediction. In this way, SIR resolves the issue of center feature missing, which hinders the design of the fully sparse architecture for all center-based or anchor-based detectors. Moreover, SIR avoids the time-consuming neighbor queries in previous point-based methods by grouping points into instances. We conduct extensive experiments on the large-scale Waymo Open Dataset to reveal the working mechanism of FSD, and state-of-the-art performance is reported. To demonstrate the superiority of FSD in long-range detection, we also conduct experiments on Argoverse 2 Dataset, which has a much larger perception range ($200m$) than Waymo Open Dataset ($75m$).  On such a large perception range, FSD achieves state-of-the-art performance and is 2.4$\times$ faster than the dense counterpart. Codes will be released.

        ----

        ## [26] Diffusion Visual Counterfactual Explanations

        **Authors**: *Maximilian Augustin, Valentyn Boreiko, Francesco Croce, Matthias Hein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/025f7165a452e7d0b57f1397fed3b0fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/025f7165a452e7d0b57f1397fed3b0fd-Abstract-Conference.html)

        **Abstract**:

        Visual Counterfactual Explanations (VCEs) are an important tool to understand the decisions of an image classifier. They are “small” but “realistic” semantic changes of the image changing the classifier decision. Current approaches for the generation of VCEs are restricted to adversarially robust models and often contain non-realistic artefacts, or are limited to image classification problems with few classes. In this paper, we overcome this by generating Diffusion Visual Counterfactual Explanations (DVCEs) for arbitrary ImageNet classifiers via a diffusion process. Two modifications to the diffusion process are key for our DVCEs: first, an adaptive parameterization, whose hyperparameters generalize across images and models, together with distance regularization and late start of the diffusion process, allow us to generate images with minimal semantic changes to the original ones but different classification. Second, our cone regularization via an adversarially robust model ensures that the diffusion process does not converge to trivial non-semantic changes, but instead produces realistic images of the target class which achieve high confidence by the classifier.

        ----

        ## [27] Recurrent Video Restoration Transformer with Guided Deformable Attention

        **Authors**: *Jingyun Liang, Yuchen Fan, Xiaoyu Xiang, Rakesh Ranjan, Eddy Ilg, Simon Green, Jiezhang Cao, Kai Zhang, Radu Timofte, Luc Van Gool*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/02687e7b22abc64e651be8da74ec610e-Abstract-Conference.html)

        **Abstract**:

        Video restoration aims at restoring multiple high-quality frames from multiple low-quality frames. Existing video restoration methods generally fall into two extreme cases, i.e., they either restore all frames in parallel or restore the video frame by frame in a recurrent way, which would result in different merits and drawbacks. Typically, the former has the advantage of temporal information fusion. However, it suffers from large model size and intensive memory consumption; the latter has a relatively small model size as it shares parameters across frames; however, it lacks long-range dependency modeling ability and parallelizability. In this paper, we attempt to integrate the advantages of the two cases by proposing a recurrent video restoration transformer, namely RVRT. RVRT processes local neighboring frames in parallel within a globally recurrent framework which can achieve a good trade-off between model size, effectiveness, and efficiency. Specifically, RVRT divides the video into multiple clips and uses the previously inferred clip feature to estimate the subsequent clip feature. Within each clip, different frame features are jointly updated with implicit feature aggregation. Across different clips, the guided deformable attention is designed for clip-to-clip alignment, which predicts multiple relevant locations from the whole inferred clip and aggregates their features by the attention mechanism. Extensive experiments on video super-resolution, deblurring, and denoising show that the proposed RVRT achieves state-of-the-art performance on benchmark datasets with balanced model size, testing memory and runtime.

        ----

        ## [28] A Consolidated Cross-Validation Algorithm for Support Vector Machines via Data Reduction

        **Authors**: *Boxiang Wang, Archer Y. Yang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/026aff87942ce636ada884d934cde0ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/026aff87942ce636ada884d934cde0ae-Abstract-Conference.html)

        **Abstract**:

        We propose a consolidated cross-validation (CV) algorithm for training and tuning the support vector machines (SVM) on reproducing kernel Hilbert spaces. Our consolidated CV algorithm utilizes a recently proposed exact leave-one-out formula for the SVM and accelerates the SVM computation via a data reduction strategy. In addition, to compute the SVM with the bias term (intercept), which is not handled by the existing data reduction methods, we propose a novel two-stage consolidated CV algorithm. With numerical studies, we demonstrate that our algorithm is about an order of magnitude faster than the two mainstream SVM solvers, kernlab and LIBSVM, with almost the same accuracy.

        ----

        ## [29] On-Demand Sampling: Learning Optimally from Multiple Distributions

        **Authors**: *Nika Haghtalab, Michael I. Jordan, Eric Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/02917acec264a52a729b99d9bc857909-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/02917acec264a52a729b99d9bc857909-Abstract-Conference.html)

        **Abstract**:

        Societal and real-world considerations such as robustness, fairness, social welfare and multi-agent tradeoffs have given rise to multi-distribution learning paradigms, such as collaborative [Blum et al. 2017], group distributionally robust [Sagawa et al. 2019], and fair federated learning [Mohri et al. 2019]. In each of these settings, a learner seeks to minimize its worstcase loss over a set of $n$ predefined distributions, while using as few samples as possible. In this paper, we establish the optimal sample complexity of these learning paradigms and give algorithms that meet this sample complexity. Importantly, our sample complexity bounds exceed that of the sample complexity of learning a single distribution only by an additive factor of $\frac{n\log(n)}{\epsilon^2}$. These improve upon the best known sample complexity of agnostic federated learning by Mohri et al. 2019 by a multiplicative factor of $n$, the sample complexity of collaborative learning by Nguyen and Zakynthinou 2018 by a multiplicative factor $\frac{\log(n)}{\epsilon^3}$, and give the first sample complexity bounds for the group DRO objective of Sagawa et al. 2019. To achieve optimal sample complexity, our algorithms learn to sample and learn from distributions on demand. Our algorithm design and analysis extends stochastic optimization techniques to solve zero-sum games in a new stochastic setting.

        ----

        ## [30] Asynchronous SGD Beats Minibatch SGD Under Arbitrary Delays

        **Authors**: *Konstantin Mishchenko, Francis R. Bach, Mathieu Even, Blake E. Woodworth*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/029df12a9363313c3e41047844ecad94-Abstract-Conference.html)

        **Abstract**:

        The existing analysis of asynchronous stochastic gradient descent (SGD) degrades dramatically when any delay is large, giving the impression that performance depends primarily on the delay. On the contrary, we prove much better guarantees for the same asynchronous SGD algorithm regardless of the delays in the gradients, depending instead just on the number of parallel devices used to implement the algorithm. Our guarantees are strictly better than the existing analyses, and we also argue that asynchronous SGD outperforms synchronous minibatch SGD in the settings we consider. For our analysis, we introduce a novel recursion based on ``virtual iterates'' and delay-adaptive stepsizes, which allow us to derive state-of-the-art guarantees for both convex and non-convex objectives.

        ----

        ## [31] Coresets for Relational Data and The Applications

        **Authors**: *Jiaxiang Chen, Qingyuan Yang, Ruomin Huang, Hu Ding*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/029f82afd78288059dc946b105c451fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/029f82afd78288059dc946b105c451fd-Abstract-Conference.html)

        **Abstract**:

        A coreset is a small set that can approximately preserve the structure of the original input data set. Therefore we can run our algorithm on a coreset so as to reduce the total computational complexity. Conventional coreset techniques assume that the input data set is available to process explicitly. However, this assumption may not hold in real-world scenarios. In this paper, we consider the problem of coresets construction over relational data. Namely, the data is decoupled into several relational tables, and it could be very expensive to directly materialize the data matrix by joining the tables. We propose a novel approach called ``aggregation tree with pseudo-cube'' that can build a coreset from bottom to up. Moreover, our approach can neatly circumvent several troublesome issues of relational learning problems [Khamis et al., PODS 2019]. Under some mild assumptions, we show that our coreset approach can be applied for the machine learning tasks, such as clustering, logistic regression and SVM.

        ----

        ## [32] Model-Based Offline Reinforcement Learning with Pessimism-Modulated Dynamics Belief

        **Authors**: *Kaiyang Guo, Yunfeng Shao, Yanhui Geng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03469b1a66e351b18272be23baf3b809-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03469b1a66e351b18272be23baf3b809-Abstract-Conference.html)

        **Abstract**:

        Model-based offline reinforcement learning (RL) aims to find highly rewarding policy, by leveraging a previously collected static dataset and a dynamics model. While the dynamics model learned through reuse of the static dataset, its generalization ability hopefully promotes policy learning if properly utilized. To that end, several works propose to quantify the uncertainty of predicted dynamics, and explicitly apply it to penalize reward. However, as the dynamics and the reward are  intrinsically different factors in context of MDP, characterizing the impact of dynamics uncertainty through reward penalty may incur unexpected tradeoff between model utilization and risk avoidance. In this work, we instead maintain a belief distribution over dynamics, and evaluate/optimize policy through biased sampling from the belief. The sampling procedure, biased towards pessimism, is derived based on an alternating Markov game formulation of offline RL. We formally show that the biased sampling naturally induces an updated dynamics belief with policy-dependent reweighting factor, termed Pessimism-Modulated Dynamics Belief. To improve policy, we devise an iterative regularized policy optimization algorithm for the game, with guarantee of monotonous improvement under certain condition. To make practical, we further devise an offline RL algorithm to approximately find the solution. Empirical results show that the proposed approach achieves state-of-the-art performance on a wide range of benchmark tasks.

        ----

        ## [33] Generating Training Data with Language Models: Towards Zero-Shot Language Understanding

        **Authors**: *Yu Meng, Jiaxin Huang, Yu Zhang, Jiawei Han*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0346c148ba1c21c6b4780a961ea141dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0346c148ba1c21c6b4780a961ea141dc-Abstract-Conference.html)

        **Abstract**:

        Pretrained language models (PLMs) have demonstrated remarkable performance in various natural language processing tasks: Unidirectional PLMs (e.g., GPT) are well known for their superior text generation capabilities; bidirectional PLMs (e.g., BERT) have been the prominent choice for natural language understanding (NLU) tasks. While both types of models have achieved promising few-shot learning performance, their potential for zero-shot learning has been underexplored. In this paper, we present a simple approach that uses both types of PLMs for fully zero-shot learning of NLU tasks without requiring any task-specific data: A unidirectional PLM generates class-conditioned texts guided by prompts, which are used as the training data for fine-tuning a bidirectional PLM. With quality training data selected based on the generation probability and regularization techniques (label smoothing and temporal ensembling) applied to the fine-tuning stage for better generalization and stability, our approach demonstrates strong performance across seven classification tasks of the GLUE benchmark (e.g., 72.3/73.8 on MNLI-m/mm and 92.8 on SST-2), significantly outperforming zero-shot prompting methods and achieving even comparable results to strong few-shot approaches using 32 training samples per class.

        ----

        ## [34] Wavelet Score-Based Generative Modeling

        **Authors**: *Florentin Guth, Simon Coste, Valentin De Bortoli, Stéphane Mallat*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03474669b759f6d38cdca6fb4eb905f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03474669b759f6d38cdca6fb4eb905f4-Abstract-Conference.html)

        **Abstract**:

        Score-based generative models (SGMs) synthesize new data samples from Gaussian white noise by running a time-reversed Stochastic Differential Equation (SDE) whose drift coefficient depends on some probabilistic score. The discretization of such SDEs typically requires a large number of time steps and hence a high computational cost. This is because of ill-conditioning properties of the score that we analyze mathematically. Previous approaches have relied on multiscale generation to considerably accelerate SGMs. We explain how this acceleration results from an implicit factorization of the data distribution into a product of conditional probabilities of wavelet coefficients across scales. The resulting Wavelet Score-based Generative Model (WSGM) synthesizes wavelet coefficients with the same number of time steps at all scales, and its time complexity therefore grows linearly with the image size. This is proved mathematically for Gaussian distributions, and shown numerically for physical processes at phase transition and natural image datasets.

        ----

        ## [35] Robust Binary Models by Pruning Randomly-initialized Networks

        **Authors**: *Chen Liu, Ziqi Zhao, Sabine Süsstrunk, Mathieu Salzmann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/035f23c0ac4cf2b73b9365ba5a98ad56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/035f23c0ac4cf2b73b9365ba5a98ad56-Abstract-Conference.html)

        **Abstract**:

        Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks. Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either +1 or −1, keep them fixed, and find a subnetwork structure that is robust to attacks. Our method confirms the Strong Lottery Ticket Hypothesis in the presence of adversarial attacks, and extends this to binary networks. Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability. Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets. Finally, we show the structured patterns of our pruned binary networks.

        ----

        ## [36] Why do tree-based models still outperform deep learning on typical tabular data?

        **Authors**: *Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        While deep learning has enabled tremendous progress on text and image datasets, its superiority on tabular data is not clear. We contribute extensive benchmarks of standard and novel deep learning methods as well as tree-based models such as XGBoost and Random Forests, across a large number of datasets and hyperparameter combinations. We define a standard set of 45 datasets from varied domains with clear characteristics of tabular data and a benchmarking methodology accounting for both fitting models and finding good hyperparameters. Results show that tree-based models remain state-of-the-art on medium-sized data ($\sim$10K samples) even without accounting for their superior speed. To understand this gap, we conduct an empirical investigation into the differing inductive biases of tree-based models and neural networks. This leads to a series of challenges which should guide researchers aiming to build tabular-specific neural network: 1) be robust to uninformative features, 2) preserve the orientation of the data, and 3) be able to easily learn irregular functions. To stimulate research on tabular architectures, we contribute a standard benchmark and raw data for baselines: every point of a 20\,000 compute hours hyperparameter search for each learner.

        ----

        ## [37] Generalizing Consistent Multi-Class Classification with Rejection to be Compatible with Arbitrary Losses

        **Authors**: *Yuzhou Cao, Tianchi Cai, Lei Feng, Lihong Gu, Jinjie Gu, Bo An, Gang Niu, Masashi Sugiyama*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03a90e1bb2ceb2ea165424f2d96aa3a1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03a90e1bb2ceb2ea165424f2d96aa3a1-Abstract-Conference.html)

        **Abstract**:

        \emph{Classification with rejection} (CwR) refrains from making a prediction to avoid critical misclassification when encountering test samples that are difficult to classify. Though previous methods for CwR have been provided with theoretical guarantees, they are only compatible with certain loss functions, making them not flexible enough when the loss needs to be changed with the dataset in practice. In this paper, we derive a novel formulation for CwR that can be equipped with arbitrary loss functions while maintaining the theoretical guarantees. First, we show that $K$-class CwR is equivalent to a $(K\!+\!1)$-class classification problem on the original data distribution with an augmented class, and propose an empirical risk minimization formulation to solve this problem with an estimation error bound. Then, we find necessary and sufficient conditions for the learning \emph{consistency} of the surrogates constructed on our proposed formulation equipped with any classification-calibrated multi-class losses, where consistency means the surrogate risk minimization implies the target risk minimization for CwR. Finally, experiments on benchmark datasets validate the effectiveness of our proposed method.

        ----

        ## [38] Markovian Interference in Experiments

        **Authors**: *Vivek F. Farias, Andrew A. Li, Tianyi Peng, Andrew Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03a9a9c1e15850439653bb971a4ad4b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03a9a9c1e15850439653bb971a4ad4b3-Abstract-Conference.html)

        **Abstract**:

        We consider experiments in dynamical systems where interventions on some experimental units impact other units through a limiting constraint (such as a limited supply of products). Despite outsize practical importance, the best estimators for this `Markovian' interference problem are largely heuristic in nature, and their bias is not well understood. We formalize the problem of inference in such experiments as one of policy evaluation. Off-policy estimators, while unbiased, apparently incur a large penalty in variance relative to state-of-the-art heuristics. We introduce an on-policy estimator: the Differences-In-Q's (DQ) estimator. We show that the DQ estimator can in general have exponentially smaller variance than off-policy evaluation. At the same time, its bias is second order in the impact of the intervention. This yields a striking bias-variance tradeoff so that the DQ estimator effectively dominates state-of-the-art alternatives. From a theoretical perspective, we introduce three separate novel techniques that are of independent interest in the theory of Reinforcement Learning (RL). Our empirical evaluation includes a set of experiments on a city-scale ride-hailing simulator.

        ----

        ## [39] Identifiability and generalizability from multiple experts in Inverse Reinforcement Learning

        **Authors**: *Paul Rolland, Luca Viano, Norman Schürhoff, Boris Nikolov, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03bdba50e3741ac5e3eaa0e55423587e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03bdba50e3741ac5e3eaa0e55423587e-Abstract-Conference.html)

        **Abstract**:

        While Reinforcement Learning (RL) aims to train an agent from a reward function in a given environment, Inverse Reinforcement Learning (IRL) seeks to recover the reward function from observing an expert's behavior. It is well known that, in general, various reward functions can lead to the same optimal policy, and hence, IRL is ill-defined. However, \cite{cao2021identifiability} showed that, if we observe two or more experts with different discount factors or acting in different environments, the reward function can under certain conditions be identified up to a constant. This work starts by showing an equivalent identifiability statement from multiple experts in tabular MDPs based on a rank condition, which is easily verifiable and is shown to be also necessary. We then extend our result to various different scenarios, i.e., we characterize reward identifiability in the case where the reward function can be represented as a linear combination of given features, making it more interpretable, or when we have access to approximate transition matrices. Even when the reward is not identifiable, we provide conditions characterizing when data on multiple experts in a given environment allows to generalize and train an optimal agent in a new environment. Our theoretical results on reward identifiability and generalizability are validated in various numerical experiments.

        ----

        ## [40] Parallel Tempering With a Variational Reference

        **Authors**: *Nikola Surjanovic, Saifuddin Syed, Alexandre Bouchard-Côté, Trevor Campbell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03cd3cf3f74d4f9ce5958de269960884-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03cd3cf3f74d4f9ce5958de269960884-Abstract-Conference.html)

        **Abstract**:

        Sampling from complex target distributions is a challenging task fundamental to Bayesian inference. Parallel tempering (PT) addresses this problem by constructing a Markov chain on the expanded state space of a sequence of distributions interpolating between the posterior distribution and a fixed reference distribution, which is typically chosen to be the prior. However, in the typical case where the prior and posterior are nearly mutually singular, PT methods are computationally prohibitive. In this work we address this challenge by constructing a generalized annealing path connecting the posterior to an adaptively tuned variational reference. The reference distribution is tuned to minimize the forward (inclusive) KL divergence to the posterior distribution using a simple, gradient-free moment-matching procedure. We show that our adaptive procedure converges to the forward KL minimizer, and that the forward KL divergence serves as a good proxy to a previously developed measure of PT performance. We also show that in the large-data limit in typical Bayesian models, the proposed  method improves in performance, while traditional PT deteriorates arbitrarily. Finally, we introduce PT with two references---one fixed, one  variational---with a novel split annealing path that ensures stable variational reference adaptation. The paper concludes with experiments that demonstrate the large empirical gains achieved by our method in a wide range of realistic Bayesian inference scenarios.

        ----

        ## [41] Provably Efficient Reinforcement Learning in Partially Observable Dynamical Systems

        **Authors**: *Masatoshi Uehara, Ayush Sekhari, Jason D. Lee, Nathan Kallus, Wen Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03d7e13f0092405804f3a381ade8f3f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03d7e13f0092405804f3a381ade8f3f0-Abstract-Conference.html)

        **Abstract**:

        We study Reinforcement Learning for partially observable systems using function approximation. We propose a new PO-bilinear framework, that is general enough to include models such as undercomplete tabular Partially Observable Markov Decision Processes (POMDPs), Linear Quadratic Gaussian (LQG), Predictive State Representations (PSRs),  as well as a newly introduced model Hilbert Space Embeddings of POMDPs. Under this framework, we propose an actor-critic style algorithm that is capable to performing agnostic policy learning. Given a policy class that consists of memory based policies (i.e., policy that looks at a fixed-length window of recent observations), and a value function class that consists of functions taking both memory and future observations as inputs, our algorithm learns to compete against the best memory-based policy among the policy class. For certain examples such as undercomplete POMDPs and LQGs, by leveraging their special properties, our algorithm is even capable of competing against the globally optimal policy without paying an exponential dependence on the horizon.

        ----

        ## [42] Off-Policy Evaluation for Episodic Partially Observable Markov Decision Processes under Non-Parametric Models

        **Authors**: *Rui Miao, Zhengling Qi, Xiaoke Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03dfa2a7755635f756b160e9f4c6b789-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03dfa2a7755635f756b160e9f4c6b789-Abstract-Conference.html)

        **Abstract**:

        We study the problem of off-policy evaluation (OPE) for episodic Partially Observable Markov Decision Processes (POMDPs) with continuous states. Motivated by the recently proposed proximal causal inference framework, we develop a non-parametric identification result for estimating the policy value via a sequence of so-called V-bridge functions with the help of time-dependent proxy variables. We then develop a fitted-Q-evaluation-type algorithm to estimate V-bridge functions recursively, where a non-parametric instrumental variable (NPIV) problem is solved at each step. By analyzing this challenging sequential NPIV estimation, we establish the finite-sample error bounds for estimating the V-bridge functions and accordingly that for evaluating the policy value, in terms of the sample size, length of horizon and so-called (local) measure of ill-posedness at each step. To the best of our knowledge, this is the first finite-sample error bound for OPE in POMDPs under non-parametric models.

        ----

        ## [43] Efficient Knowledge Distillation from Model Checkpoints

        **Authors**: *Chaofei Wang, Qisen Yang, Rui Huang, Shiji Song, Gao Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/03e0712bf85ebe7cec4f1a7fc53216c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/03e0712bf85ebe7cec4f1a7fc53216c9-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation is an effective approach to learn compact models (students) with the supervision of large and strong models (teachers). As empirically there exists a strong correlation between the performance of teacher and student models, it is commonly believed that a high performing teacher is preferred. Consequently, practitioners tend to use a well trained network or an ensemble of them as the teacher. In this paper, we observe that an intermediate model, i.e., a checkpoint in the middle of the training procedure, often serves as a better teacher compared to the fully converged model, although the former has much lower accuracy. More surprisingly, a weak snapshot ensemble of several intermediate models from a same training trajectory can outperform a strong ensemble of independently trained and fully converged models, when they are used as teachers. We show that this phenomenon can be partially explained by the information bottleneck principle: the feature representations of intermediate models can have higher mutual information regarding the input, and thus contain more ``dark knowledge'' for effective distillation. We further propose an optimal intermediate teacher selection algorithm based on maximizing the total task-related mutual information. Experiments verify its effectiveness and applicability. Our code is available at https://github.com/LeapLabTHU/CheckpointKD.

        ----

        ## [44] Decoupled Self-supervised Learning for Graphs

        **Authors**: *Teng Xiao, Zhengyu Chen, Zhimeng Guo, Zeyang Zhuang, Suhang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/040c816286b3844fd78f2124eec75f2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/040c816286b3844fd78f2124eec75f2e-Abstract-Conference.html)

        **Abstract**:

        This paper studies the problem of conducting self-supervised learning for node representation learning on  graphs. Most existing self-supervised learning methods assume the graph is homophilous, where linked nodes often belong to the same class or have similar features. However, such assumptions of homophily do not always hold in real-world graphs. We address this problem by developing a decoupled self-supervised learning (DSSL) framework for graph neural networks. DSSL imitates a generative process of nodes and links from latent variable modeling of the semantic structure, which decouples different underlying semantics between different neighborhoods into the self-supervised learning process. Our DSSL framework is agnostic to the encoders and does not need prefabricated augmentations, thus is flexible to different graphs. To effectively optimize the framework,  we derive the evidence lower bound of the self-supervised objective and develop a scalable training algorithm with variational inference. We provide a theoretical analysis to justify that DSSL enjoys the better downstream performance. Extensive experiments on various types of graph benchmarks demonstrate that our proposed framework can  achieve better performance compared with competitive  baselines.

        ----

        ## [45] Shadow Knowledge Distillation: Bridging Offline and Online Knowledge Transfer

        **Authors**: *Lujun Li, Zhe Jin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/040d3b6af368bf71f952c18da5713b48-Abstract-Conference.html)

        **Abstract**:

        Knowledge distillation can be generally divided into offline and online categories according to whether teacher model is pre-trained and persistent during the distillation process.  Offline distillation can employ existing models yet always demonstrates inferior performance than online ones.  In this paper, we first empirically show that the essential factor for their performance gap lies in the reversed distillation from student to teacher, rather than the training fashion.  Offline distillation can achieve competitive performance gain by fine-tuning pre-trained teacher to adapt student with such reversed distillation.  However, this fine-tuning process still costs lots of training budgets.  To alleviate this dilemma, we propose SHAKE, a simple yet effective SHAdow KnowlEdge transfer framework to bridge offline and online distillation, which trades the accuracy with efficiency.  Specifically, we build an extra shadow head on the backbone to mimic the predictions of pre-trained teacher as its shadow.  Then, this shadow head is leveraged as a proxy teacher to perform bidirectional distillation with student on the fly.  In this way, SHAKE not only updates this student-aware proxy teacher with the knowledge of pre-trained model, but also greatly optimizes costs of augmented reversed distillation.  Extensive experiments on classification and object detection tasks demonstrate that our technique achieves state-of-the-art results with different CNNs and Vision Transformer models.  Additionally, our method shows strong compatibility with multi-teacher and augmentation strategies by gaining additional performance improvement.  Code is made publicly available at https://lilujunai.github.io/SHAKE/.

        ----

        ## [46] ComENet: Towards Complete and Efficient Message Passing for 3D Molecular Graphs

        **Authors**: *Limei Wang, Yi Liu, Yuchao Lin, Haoran Liu, Shuiwang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0418973e545b932939302cb605d06f43-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0418973e545b932939302cb605d06f43-Abstract-Conference.html)

        **Abstract**:

        Many real-world data can be modeled as 3D graphs, but learning representations that incorporates 3D information completely and efficiently is challenging. Existing methods either use partial 3D information, or suffer from excessive computational cost. To incorporate 3D information completely and efficiently, we propose a novel message passing scheme that operates within 1-hop neighborhood. Our method guarantees full completeness of 3D information on 3D graphs by achieving global and local completeness. Notably, we propose the important rotation angles to fulfill global completeness. Additionally, we show that our method is orders of magnitude faster than prior methods. We provide rigorous proof of completeness and analysis of time complexity for our methods. As molecules are in essence quantum systems, we build the \underline{com}plete and \underline{e}fficient graph neural network (ComENet) by combing quantum inspired basis functions and the proposed message passing scheme. Experimental results demonstrate the capability and efficiency of ComENet, especially on real-world datasets that are large in both numbers and sizes of graphs. Our code is publicly available as part of the DIG library (\url{https://github.com/divelab/DIG}).

        ----

        ## [47] VLMbench: A Compositional Benchmark for Vision-and-Language Manipulation

        **Authors**: *Kaizhi Zheng, Xiaotong Chen, Odest Chadwicke Jenkins, Xin Eric Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/04543a88eae2683133c1acbef5a6bf77-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Benefiting from language flexibility and compositionality, humans naturally intend to use language to command an embodied agent for complex tasks such as navigation and object manipulation. In this work, we aim to fill the blank of the last mile of embodied agents---object manipulation by following human guidance, e.g., “move the red mug next to the box while keeping it upright.” To this end, we introduce an Automatic Manipulation Solver (AMSolver) system and build a Vision-and-Language Manipulation benchmark (VLMbench) based on it, containing various language instructions on categorized robotic manipulation tasks. Specifically, modular rule-based task templates are created to automatically generate robot demonstrations with language instructions, consisting of diverse object shapes and appearances, action types, and motion constraints. We also develop a keypoint-based model 6D-CLIPort to deal with multi-view observations and language input and output a sequence of 6 degrees of freedom (DoF) actions. We hope the new simulator and benchmark will facilitate future research on language-guided robotic manipulation.

        ----

        ## [48] Tiered Reinforcement Learning: Pessimism in the Face of Uncertainty and Constant Regret

        **Authors**: *Jiawei Huang, Li Zhao, Tao Qin, Wei Chen, Nan Jiang, Tie-Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0463ec87d0ac1e98a6cbe3d95d4e3e35-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0463ec87d0ac1e98a6cbe3d95d4e3e35-Abstract-Conference.html)

        **Abstract**:

        We propose a new learning framework that captures the tiered structure of many real-world user-interaction applications, where the users can be divided into two groups based on their different tolerance on exploration risks and should be treated separately. In this setting, we simultaneously maintain two policies $\pi^{\text{O}}$ and $\pi^{\text{E}}$: $\pi^{\text{O}}$ (``O'' for ``online'') interacts with more risk-tolerant users from the first tier and minimizes regret by balancing exploration and exploitation as usual, while $\pi^{\text{E}}$ (``E'' for ``exploit'') exclusively focuses on exploitation for risk-averse users from the second tier utilizing the data collected so far. An important question is whether such a separation yields advantages over the standard online setting (i.e., $\pi^{\text{E}}=\pi^{\text{O}}$) for the risk-averse users. We individually consider the gap-independent vs.~gap-dependent settings. For the former, we prove that the separation is indeed not beneficial from a minimax perspective. For the latter, we show that if choosing Pessimistic Value Iteration as the exploitation algorithm to produce $\pi^{\text{E}}$, we can achieve a constant regret for risk-averse users independent of the number of episodes $K$, which is in sharp contrast to the $\Omega(\log K)$ regret for any online RL algorithms in the same setting, while the regret of $\pi^{\text{O}}$ (almost) maintains its online regret optimality and does not need to compromise for the success of $\pi^{\text{E}}$.

        ----

        ## [49] Between Stochastic and Adversarial Online Convex Optimization: Improved Regret Bounds via Smoothness

        **Authors**: *Sarah Sachs, Hédi Hadiji, Tim van Erven, Cristóbal Guzmán*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/047aa59e51e3ac7a2422a55468feefd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/047aa59e51e3ac7a2422a55468feefd5-Abstract-Conference.html)

        **Abstract**:

        Stochastic and adversarial data are two widely studied settings in online learning. But many optimizationtasks are neither i.i.d. nor fully adversarial, which makes it of  fundamental interest to get a better theoretical understanding of the world between these extremes. In this work we establish novel regret bounds for online convex optimization in a setting that interpolates between stochastic i.i.d. and fully adversarial losses. By exploiting smoothness of the expected losses, these bounds replace a dependence on the maximum gradient length by the variance of the gradients, which was previously known only for linear losses. In addition, they weaken the i.i.d. assumption by allowing, for example, adversarially poisoned rounds, which were previously considered in the expert and bandit setting. Our results extend this to the online convex optimization framework.  In the fully i.i.d. case, our bounds match the rates one would expect from results in stochastic acceleration, and in the fully adversarial case they gracefully deteriorate to match the minimax regret.  We further provide lower bounds showing that our regret upper bounds aretight for all intermediate regimes in terms of the stochastic variance and theadversarial variation of the loss gradients.

        ----

        ## [50] Differentially Private Learning Needs Hidden State (Or Much Faster Convergence)

        **Authors**: *Jiayuan Ye, Reza Shokri*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04b42392f9a3a16aea012395359b8148-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/04b42392f9a3a16aea012395359b8148-Abstract-Conference.html)

        **Abstract**:

        Prior work on differential privacy analysis of randomized SGD algorithms relies on composition theorems, where the implicit (unrealistic) assumption is that the internal state of the iterative algorithm is revealed to the adversary. As a result, the R\'enyi DP bounds derived by such composition-based analyses linearly grow with the number of training epochs. When the internal state of the algorithm is hidden, we prove a converging privacy bound for noisy stochastic gradient descent (on strongly convex smooth loss functions). We show how to take advantage of privacy amplification by sub-sampling and randomized post-processing, and prove the dynamics of privacy bound for shuffle and partition'' andsample without replacement'' stochastic mini-batch gradient descent schemes. We prove that, in these settings, our privacy bound converges exponentially fast and is substantially smaller than the composition bounds, notably after a few number of training epochs. Thus, unless the DP algorithm converges fast, our privacy analysis shows that hidden state analysis can significantly amplify differential privacy.

        ----

        ## [51] BR-SNIS: Bias Reduced Self-Normalized Importance Sampling

        **Authors**: *Gabriel Cardoso, Sergey Samsonov, Achille Thin, Eric Moulines, Jimmy Olsson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/04bd683d5428d91c5fbb5a7d2c27064d-Abstract-Conference.html)

        **Abstract**:

        Importance Sampling (IS) is a method for approximating expectations with respect to a target distribution using independent samples from a proposal distribution and the associated to importance weights. In many cases, the target distribution is known up to a normalization constant and self-normalized IS (SNIS) is then used. While the use of self-normalization can have a positive effect on the dispersion of the estimator, it introduces bias. In this work, we propose a new method BR-SNIS whose complexity is essentially the same as SNIS and which significantly reduces bias. This method is a wrapper, in the sense that it uses the same proposal samples and importance weights but makes a clever use of iterated sampling-importance-resampling (i-SIR) to form a bias-reduced version of the estimator. We derive the proposed algorithm with rigorous theoretical results, including novel bias, variance, and high-probability bounds. We illustrate our findings with numerical examples.

        ----

        ## [52] Learning to Configure Computer Networks with Neural Algorithmic Reasoning

        **Authors**: *Luca Beurer-Kellner, Martin T. Vechev, Laurent Vanbever, Petar Velickovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04cc90ec6868b97b7423dc38ced1e35c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/04cc90ec6868b97b7423dc38ced1e35c-Abstract-Conference.html)

        **Abstract**:

        We present a new method for scaling automatic configuration of computer networks. The key idea is to relax the computationally hard search problem of finding a configuration that satisfies a given specification into an approximate objective amenable to learning-based techniques. Based on this idea, we train a neural algorithmic model which learns to generate configurations likely to (fully or partially) satisfy a given specification under existing routing protocols. By relaxing the rigid satisfaction guarantees, our approach (i) enables greater flexibility: it is protocol-agnostic, enables cross-protocol reasoning, and does not depend on hardcoded rules; and (ii) finds configurations for much larger computer networks than previously possible. Our learned synthesizer is up to 490x faster than state-of-the-art SMT-based methods, while producing configurations which on average satisfy more than 93% of the provided requirements.

        ----

        ## [53] Early Stage Convergence and Global Convergence of Training Mildly Parameterized Neural Networks

        **Authors**: *Mingze Wang, Chao Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04cda3a5ef307978cb5dbef6ab649380-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/04cda3a5ef307978cb5dbef6ab649380-Abstract-Conference.html)

        **Abstract**:

        The convergence of GD and SGD when training mildly parameterized neural networks starting from random initialization is studied. For a broad range of models and loss functions, including the widely used square loss and cross entropy loss, we prove an ''early stage convergence'' result. We show that the loss is decreased by a significant amount in the early stage of the training, and this decreasing is fast. Furthurmore, for exponential type loss functions, and under some assumptions on the training data, we show global convergence of GD. Instead of relying on extreme over-parameterization, our study is based on a microscopic analysis of the activation patterns for the neurons, which helps us derive gradient lower bounds. The results on activation patterns, which we call ``neuron partition'', help build intuitions for understanding the behavior of neural networks' training dynamics, and may be of independent interest.

        ----

        ## [54] On Divergence Measures for Bayesian Pseudocoresets

        **Authors**: *Balhae Kim, Jungwon Choi, Seanie Lee, Yoonho Lee, Jung-Woo Ha, Juho Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/04f8311e7e22eac15d67fe45c242ead8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/04f8311e7e22eac15d67fe45c242ead8-Abstract-Conference.html)

        **Abstract**:

        A Bayesian pseudocoreset is a small synthetic dataset for which the posterior over parameters approximates that of the original dataset. While promising, the scalability of Bayesian pseudocoresets is not yet validated in large-scale problems such as image classification with deep neural networks. On the other hand, dataset distillation methods similarly construct a small dataset such that the optimization with the synthetic dataset converges to a solution similar to optimization with full data. Although dataset distillation has been empirically verified in large-scale settings, the framework is restricted to point estimates, and their adaptation to Bayesian inference has not been explored. This paper casts two representative dataset distillation algorithms as approximations to methods for constructing pseudocoresets by minimizing specific divergence measures: reverse KL divergence and Wasserstein distance. Furthermore, we provide a unifying view of such divergence measures in Bayesian pseudocoreset construction. Finally, we propose a novel Bayesian pseudocoreset algorithm based on minimizing forward KL divergence. Our empirical results demonstrate that the pseudocoresets constructed from these methods reflect the true posterior even in large-scale Bayesian inference problems.

        ----

        ## [55] Unsupervised Learning of Equivariant Structure from Sequences

        **Authors**: *Takeru Miyato, Masanori Koyama, Kenji Fukumizu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0503f5dce343a1d06d16ba103dd52db1-Abstract-Conference.html)

        **Abstract**:

        In this study, we present \textit{meta-sequential prediction} (MSP), an unsupervised framework to learn the symmetry from the time sequence of length at least three. Our method leverages the stationary property~(e.g. constant velocity, constant acceleration) of the time sequence to learn the underlying equivariant structure of the dataset by simply training the encoder-decoder model to be able to predict the future observations. We will demonstrate that, with our framework, the hidden disentangled structure of the dataset naturally emerges as a by-product by applying \textit{simultaneous block-diagonalization} to the transition operators in the latent space, the procedure which is commonly used in representation theory to decompose the feature-space based on the type of response to group actions.We will showcase our method from both empirical and theoretical perspectives.Our result suggests that finding a simple structured relation and learning a model with extrapolation capability are two sides of the same coin. The code is available at https://github.com/takerum/metasequentialprediction.

        ----

        ## [56] Multi-Class $H$-Consistency Bounds

        **Authors**: *Pranjal Awasthi, Anqi Mao, Mehryar Mohri, Yutao Zhong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/051f3997af1dd65da8e14397b6a72f8e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/051f3997af1dd65da8e14397b6a72f8e-Abstract-Conference.html)

        **Abstract**:

        We present an extensive study of $H$-consistency bounds for multi-class classification. These are upper bounds on the target loss estimation error of a predictor in a hypothesis set $H$, expressed in terms of the surrogate loss estimation error of that predictor. They are stronger and more significant guarantees than Bayes-consistency, $H$-calibration or $H$-consistency, and more informative than excess error bounds derived for $H$ being the family of all measurable functions. We give a series of new $H$-consistency bounds for surrogate multi-class losses, including max losses, sum losses, and constrained losses, both in the non-adversarial and adversarial cases, and for different differentiable or convex auxiliary functions used. We also prove that no non-trivial $H$-consistency bound can be given in some cases. To our knowledge, these are the first $H$-consistency bounds proven for the multi-class setting. Our proof techniques are also novel and likely to be useful in the analysis of other such guarantees.

        ----

        ## [57] On the Frequency-bias of Coordinate-MLPs

        **Authors**: *Sameera Ramasinghe, Lachlan E. MacDonald, Simon Lucey*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0525fa17a8dbea687359116d01732e12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0525fa17a8dbea687359116d01732e12-Abstract-Conference.html)

        **Abstract**:

        We show that typical implicit regularization assumptions for deep neural networks (for regression) do not hold for coordinate-MLPs, a family of MLPs that are now ubiquitous in computer vision for representing high-frequency signals. Lack of such implicit bias disrupts smooth interpolations between training samples, and hampers generalizing across signal regions with different spectra. We investigate this behavior through a Fourier lens and uncover that as the bandwidth of a coordinate-MLP is enhanced, lower frequencies tend to get suppressed unless a suitable prior is provided explicitly. Based on these insights, we propose a simple regularization technique that can mitigate the above problem, which can be  incorporated into existing networks without any architectural modifications.

        ----

        ## [58] DC-BENCH: Dataset Condensation Benchmark

        **Authors**: *Justin Cui, Ruochen Wang, Si Si, Cho-Jui Hsieh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/052e22cfdd344c79634f7ec76fa03e22-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/052e22cfdd344c79634f7ec76fa03e22-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Dataset Condensation is a newly emerging technique aiming at learning a tiny dataset that captures the rich information encoded in the original dataset. As the size of datasets contemporary machine learning models rely on becomes increasingly large, condensation methods become a prominent direction for accelerating network training and reducing data storage. Despite numerous methods have been proposed in this rapidly growing field, evaluating and comparing different condensation methods is non-trivial and still remains an open issue. The quality of condensed dataset are often shadowed by many critical contributing factors to the end performance, such as data augmentation and model architectures. The lack of a systematic way to evaluate and compare condensation methods not only hinders our understanding of existing techniques, but also discourages practical usage of the synthesized datasets. This work provides the first large-scale standardized benchmark on Dataset Condensation. It consists of a suite of evaluations to comprehensively reflect the generability and effectiveness of condensation methods through the lens of their generated dataset. Leveraging this benchmark, we conduct a large-scale study of current condensation methods, and report many insightful findings that open up new possibilities for future development. The benchmark library, including evaluators, baseline methods, and generated datasets, is open-sourced to facilitate future research and application.

        ----

        ## [59] Mask Matching Transformer for Few-Shot Segmentation

        **Authors**: *Siyu Jiao, Gengwei Zhang, Shant Navasardyan, Ling Chen, Yao Zhao, Yunchao Wei, Honghui Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/053a18c03e0844d0c484ba2861f8ae6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/053a18c03e0844d0c484ba2861f8ae6c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we aim to tackle the challenging few-shot segmentation task from a new perspective. Typical methods follow the paradigm to firstly learn prototypical features from support images and then match query features in pixel-level to obtain segmentation results. However, to obtain satisfactory segments, such a paradigm needs to couple the learning of the matching operations with heavy segmentation modules, limiting the flexibility of design and increasing the learning complexity. To alleviate this issue, we propose Mask Matching Transformer (MM-Former), a new paradigm for the few-shot segmentation task. Specifically, MM-Former first uses a class-agnostic segmenter to decompose the query image into multiple segment proposals. Then, a simple matching mechanism is applied to merge the related segment proposals into the final mask guided by the support images. The advantages of our MM-Former are two-fold. First, the MM-Former follows the paradigm of 'decompose first and then blend', allowing our method to benefit from the advanced potential objects segmenter to produce high-quality mask proposals for query images. Second, the mission of prototypical features is relaxed to learn coefficients to fuse correct ones within a proposal pool, making the MM-Former be well generalized to complex scenarios or cases. We conduct extensive experiments on the popular COCO-$20^i$ and Pascal-$5^i$ benchmarks. Competitive results well demonstrate the effectiveness and the generalization ability of our MM-Former. Code is available at https://github.com/Picsart-AI-Research/Mask-Matching-Transformer.

        ----

        ## [60] Queue Up Your Regrets: Achieving the Dynamic Capacity Region of Multiplayer Bandits

        **Authors**: *Ilai Bistritz, Nicholas Bambos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/056e8e9c8ca9929cb6cf198952bf1dbb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/056e8e9c8ca9929cb6cf198952bf1dbb-Abstract-Conference.html)

        **Abstract**:

        Abstract Consider $N$ cooperative agents such that for $T$ turns, each agent n takes an action $a_{n}$ and receives a stochastic reward $r_{n}\left(a_{1},\ldots,a_{N}\right)$. Agents cannot observe the actions of other agents and do not know even their own reward function. The agents can communicate with their neighbors on a connected graph $G$ with diameter $d\left(G\right)$. We want each agent $n$ to achieve an expected average reward of at least $\lambda_{n}$ over time, for a given quality of service (QoS) vector $\boldsymbol{\lambda}$. A QoS vector $\boldsymbol{\lambda}$ is not necessarily achievable. By giving up on immediate reward, knowing that the other agents will compensate later, agents can improve their achievable capacity region. Our main observation is that the gap between $\lambda_{n}t$ and the accumulated reward of agent $n$, which we call the QoS regret, behaves like a queue. Inspired by this observation, we propose a distributed algorithm that aims to learn a max-weight matching of agents to actions. In each epoch, the algorithm employs a consensus phase where the agents agree on a certain weighted sum of rewards by communicating only $O\left(d\left(G\right)\right)$ numbers every turn. Then, the algorithm uses distributed successive elimination on a random subset of action profiles to approximately maximize this weighted sum of rewards. We prove a bound on the accumulated sum of expected QoS regrets of all agents, that holds if $\boldsymbol{\lambda}$ is a safety margin $\varepsilon_{T}$ away from the boundary of the capacity region, where $\varepsilon_{T}\rightarrow0$ as $T\rightarrow\infty$. This bound implies that, for large $T$, our algorithm can achieve any $\boldsymbol{\lambda}$ in the interior of the dynamic capacity region, while all agents are guaranteed an empirical average expected QoS regret of $\tilde{O}\left(1\right)$ over $t=1,\ldots,T$ which never exceeds $\tilde{O}\left(\sqrt{t}\right)$ for any $t$. We then extend our result to time-varying i.i.d. communication graphs.

        ----

        ## [61] Differentially Private Covariance Revisited

        **Authors**: *Wei Dong, Yuting Liang, Ke Yi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/057405fd73dd7ba7f32a7cb34fb7c7f5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/057405fd73dd7ba7f32a7cb34fb7c7f5-Abstract-Conference.html)

        **Abstract**:

        In this paper, we present two new algorithms for covariance estimation under concentrated differential privacy (zCDP).  The first algorithm achieves a Frobenius error of $\tilde{O}(d^{1/4}\sqrt{\mathrm{tr}}/\sqrt{n} + \sqrt{d}/n)$, where $\mathrm{tr}$ is the trace of the covariance matrix.  By taking $\mathrm{tr}=1$, this also implies a worst-case error bound of $\tilde{O}(d^{1/4}/\sqrt{n})$, which improves the standard Gaussian mechanism's $\tilde{O}(d/n)$ for the regime $d>\widetilde{\Omega}(n^{2/3})$.  Our second algorithm offers a tail-sensitive bound that could be much better on skewed data.  The corresponding algorithms are also simple and efficient. Experimental results show that they offer significant improvements over prior work.

        ----

        ## [62] Trimmed Maximum Likelihood Estimation for Robust Generalized Linear Model

        **Authors**: *Pranjal Awasthi, Abhimanyu Das, Weihao Kong, Rajat Sen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/05b12f103c9e613efc4c85674cdc9066-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/05b12f103c9e613efc4c85674cdc9066-Abstract-Conference.html)

        **Abstract**:

        We study the problem of learning generalized linear models under adversarial corruptions.We analyze a classical heuristic called the \textit{iterative trimmed maximum likelihood estimator} which is known to be effective against \textit{label corruptions} in practice. Under label corruptions, we prove that this simple estimator achieves minimax near-optimal risk on a wide range of generalized linear models, including Gaussian regression, Poisson regression and Binomial regression. Finally, we extend the estimator to the much more challenging setting of \textit{label and covariate corruptions} and demonstrate its robustness and optimality in that setting as well.

        ----

        ## [63] Causal Discovery in Linear Latent Variable Models Subject to Measurement Error

        **Authors**: *Yuqin Yang, AmirEmad Ghassami, Mohamed S. Nafea, Negar Kiyavash, Kun Zhang, Ilya Shpitser*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/05b63fa06784b71aab3939004e0f0a0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/05b63fa06784b71aab3939004e0f0a0d-Abstract-Conference.html)

        **Abstract**:

        We focus on causal discovery in the presence of measurement error in linear systems where the mixing matrix, i.e., the matrix indicating the independent exogenous noise terms pertaining to the observed variables, is identified up to permutation and scaling of the columns. We demonstrate a somewhat surprising connection between this problem and causal discovery in the presence of unobserved parentless causes, in the sense that there is a mapping, given by the mixing matrix, between the underlying models to be inferred in these problems. Consequently, any identifiability result based on the mixing matrix for one model translates to an identifiability result for the other model. We characterize to what extent the causal models can be identified under a two-part faithfulness assumption. Under only the first part of the assumption (corresponding to the conventional definition of faithfulness), the structure can be learned up to the causal ordering among an ordered grouping of the variables but not all the edges across the groups can be identified. We further show that if both parts of the faithfulness assumption are imposed, the structure can be learned up to a more refined ordered grouping. As a result of this refinement, for the latent variable model with unobserved parentless causes, the structure can be identified. Based on our theoretical results, we propose causal structure learning methods for both models, and evaluate their performance on synthetic data.

        ----

        ## [64] Density-driven Regularization for Out-of-distribution Detection

        **Authors**: *Wenjian Huang, Hao Wang, Jiahao Xia, Chengyan Wang, Jianguo Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/05b69cc4c8ff6e24c5de1ecd27223d37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/05b69cc4c8ff6e24c5de1ecd27223d37-Abstract-Conference.html)

        **Abstract**:

        Detecting out-of-distribution (OOD) samples is essential for reliably deploying deep learning classifiers in open-world applications. However, existing detectors relying on discriminative probability suffer from the overconfident posterior estimate for OOD data. Other reported approaches either impose strong unproven parametric assumptions to estimate OOD sample density or develop empirical detectors lacking clear theoretical motivations. To address these issues, we propose a theoretical probabilistic framework for OOD detection in deep classification networks, in which two regularization constraints are constructed to reliably calibrate and estimate sample density to identify OOD. Specifically, the density consistency regularization enforces the agreement between analytical and empirical densities of observable low-dimensional categorical labels. The contrastive distribution regularization separates the densities between in distribution (ID) and distribution-deviated samples. A simple and robust implementation algorithm is also provided, which can be used for any pre-trained neural network classifiers. To the best of our knowledge, we have conducted the most extensive evaluations and comparisons on computer vision benchmarks. The results show that our method significantly outperforms state-of-the-art detectors, and even achieves comparable or better performance than methods utilizing additional large-scale outlier exposure datasets.

        ----

        ## [65] Sparsity in Continuous-Depth Neural Networks

        **Authors**: *Hananeh Aliee, Till Richter, Mikhail Solonin, Ignacio Ibarra, Fabian J. Theis, Niki Kilbertus*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0626822954674a06ccd9c234e3f0d572-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0626822954674a06ccd9c234e3f0d572-Abstract-Conference.html)

        **Abstract**:

        Neural Ordinary Differential Equations (NODEs) have proven successful in learning dynamical systems in terms of accurately recovering the observed trajectories. While different types of sparsity have been proposed to improve robustness, the generalization properties of NODEs for dynamical systems beyond the observed data are underexplored. We systematically study the influence of weight and feature sparsity on forecasting as well as on identifying the underlying dynamical laws. Besides assessing existing methods, we propose a regularization technique to sparsify ``input-output connections'' and extract relevant features during training. Moreover, we curate real-world datasets including human motion capture and human hematopoiesis single-cell RNA-seq data to realistically analyze different levels of out-of-distribution (OOD) generalization in forecasting and dynamics identification respectively. Our extensive empirical evaluation on these challenging benchmarks suggests that weight sparsity improves generalization in the presence of noise or irregular sampling. However, it does not prevent learning spurious feature dependencies in the inferred dynamics, rendering them impractical for predictions under interventions, or for inferring the true underlying dynamics. Instead, feature sparsity can indeed help with recovering sparse ground-truth dynamics compared to unregularized NODEs.

        ----

        ## [66] Environment Diversification with Multi-head Neural Network for Invariant Learning

        **Authors**: *Bo-Wei Huang, Keng-Te Liao, Chang-Sheng Kao, Shou-De Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/062d711fb777322e2152435459e6e9d9-Abstract-Conference.html)

        **Abstract**:

        Neural networks are often trained with empirical risk minimization; however, it has been shown that a shift between training and testing distributions can cause unpredictable performance degradation. On this issue, a research direction, invariant learning, has been proposed to extract causal features insensitive to the distributional changes. This work proposes EDNIL, an invariant learning framework containing a multi-head neural network to absorb data biases. We show that this framework does not require prior knowledge about environments or strong assumptions about the pre-trained model. We also reveal that the proposed algorithm has theoretical connections to recent studies discussing properties of variant and invariant features. Finally, we demonstrate that models trained with EDNIL are empirically more robust against distributional shifts.

        ----

        ## [67] Learning Probabilistic Models from Generator Latent Spaces with Hat EBM

        **Authors**: *Mitch Hill, Erik Nijkamp, Jonathan Mitchell, Bo Pang, Song-Chun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/062f9525a7476942f61a6c3b42d0a63f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/062f9525a7476942f61a6c3b42d0a63f-Abstract-Conference.html)

        **Abstract**:

        This work proposes a method for using any generator network as the foundation of an Energy-Based Model (EBM). Our formulation posits that observed images are the sum of unobserved latent variables passed through the generator network and a residual random variable that spans the gap between the generator output and the image manifold. One can then define an EBM that includes the generator as part of its forward pass, which we call the Hat EBM. The model can be trained without inferring the latent variables of the observed data or calculating the generator Jacobian determinant. This enables explicit probabilistic modeling of the output distribution of any type of generator network. Experiments show strong performance of the proposed method on (1) unconditional ImageNet synthesis at 128$\times$128 resolution, (2) refining the output of existing generators, and (3) learning EBMs that incorporate non-probabilistic generators. Code and pretrained models to reproduce our results are available at https://github.com/point0bar1/hat-ebm.

        ----

        ## [68] Learning Best Combination for Efficient N: M Sparsity

        **Authors**: *Yuxin Zhang, Mingbao Lin, Zhihang Lin, Yiting Luo, Ke Li, Fei Chao, Yongjian Wu, Rongrong Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/06589ec9d86876508600a678f9c8f51d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/06589ec9d86876508600a678f9c8f51d-Abstract-Conference.html)

        **Abstract**:

        By forcing N out of M consecutive weights to be non-zero, the recent N:M fine-grained network sparsity has received increasing attention with its two attractive advantages over traditional irregular network sparsity methods: 1) Promising performance at a high sparsity. 2) Significant speedups when performed on NVIDIA A100 GPUs. Current implementation on N:M sparsity requires a tedious pre-training phase or computationally heavy from-scratch training. To circumvent these problems, this paper presents an efficient solution for achieving N:M fine-grained sparsity from scratch. Specifically, we first make a re-formulation to convert the N:M fine-grained sparsity into a combinatorial problem, in which, the object falls into choosing the best weight combination among $C_M^N$ candidates. Then, we equip each combination with a learnable importance score, which can be jointly optimized along with its associated weights. Through rigorous proof, we demonstrate that the magnitude of the optimized score well reflects the importance of its corresponding weights combination to the training loss. Therefore, by gradually removing combinations with smaller scores till the best one is left, N:M fine-grained sparsity can be efficiently optimized during the normal training phase without any extra expenditure. Comprehensive experimental results have demonstrated that our proposed method for learning best combination, dubbed as LBC, consistently increases the efficacy of the off-the-shelf N:M methods across varying networks and datasets. Our project is released at https://github.com/zyxxmu/LBC.

        ----

        ## [69] Why Do Artificially Generated Data Help Adversarial Robustness

        **Authors**: *Yue Xing, Qifan Song, Guang Cheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/065e259a1d2d955e63b99aac6a3a3081-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/065e259a1d2d955e63b99aac6a3a3081-Abstract-Conference.html)

        **Abstract**:

        In the adversarial training framework of \cite{carmon2019unlabeled,gowal2021improving}, people use generated/real unlabeled data with pseudolabels to improve adversarial robustness. We provide statistical insights to explain why the artificially generated data improve adversarial training. In particular, we study how the attack strength and the quality of the unlabeled data affect adversarial robustness in this framework. Our results show that with a high-quality unlabeled data generator, adversarial training can benefit greatly from this framework under large attack strength, while a poor generator can still help to some extent. To make adaptions concerning the quality of generated data, we propose an algorithm that performs online adjustment to the weight between the labeled real data and the generated data, aiming to optimize the adversarial risk. Numerical studies are conducted to verify our theories and show the effectiveness of the proposed algorithm.

        ----

        ## [70] Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera

        **Authors**: *Hongrui Cai, Wanquan Feng, Xuetao Feng, Yan Wang, Juyong Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html)

        **Abstract**:

        We propose Neural-DynamicReconstruction (NDR), a template-free method to recover high-fidelity geometry and motions of a dynamic scene from a monocular RGB-D camera. In NDR, we adopt the neural implicit function for surface representation and rendering such that the captured color and depth can be fully utilized to jointly optimize the surface and deformations. To represent and constrain the non-rigid deformations, we propose a novel neural invertible deforming network such that the cycle consistency between arbitrary two frames is automatically satisfied. Considering that the surface topology of dynamic scene might change over time, we employ a topology-aware strategy to construct the topology-variant correspondence for the fused frames. NDR also further refines the camera poses in a global optimization manner. Experiments on public datasets and our collected dataset demonstrate that NDR outperforms existing monocular dynamic reconstruction methods.

        ----

        ## [71] Global Optimal K-Medoids Clustering of One Million Samples

        **Authors**: *Jiayang Ren, Kaixun Hua, Yankai Cao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/06abed94583030dd50abe6767bd643b1-Abstract-Conference.html)

        **Abstract**:

        We study the deterministic global optimization of the K-Medoids clustering problem. This work proposes a branch and bound (BB) scheme, in which a tailored Lagrangian relaxation method proposed in the 1970s is used to provide a lower bound at each BB node. The lower bounding method already guarantees the maximum gap at the root node. A closed-form solution to the lower bound can be derived analytically without explicitly solving any optimization problems, and its computation can be easily parallelized. Moreover, with this lower bounding method, finite convergence to the global optimal solution can be guaranteed by branching only on the regions of medoids. We also present several tailored bound tightening techniques to reduce the search space and computational cost. Extensive computational studies on 28 machine learning datasets demonstrate that our algorithm can provide a provable global optimal solution with an optimality gap of 0.1\% within 4 hours on datasets with up to one million samples. Besides, our algorithm can obtain better or equal objective values than the heuristic method. A theoretical proof of global convergence for our algorithm is also presented.

        ----

        ## [72] Batch Multi-Fidelity Active Learning with Budget Constraints

        **Authors**: *Shibo Li, Jeff M. Phillips, Xin Yu, Robert M. Kirby, Shandian Zhe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/06ea400b9b7cfce6428ec27a371632eb-Abstract-Conference.html)

        **Abstract**:

        Learning functions with high-dimensional outputs is critical in many applications, such as physical simulation and engineering design. However, collecting training examples for these applications is often costly, e.g., by running numerical solvers. The recent work (Li et al., 2022) proposes the first multi-fidelity active learning approach for high-dimensional outputs, which can acquire examples at different fidelities to reduce the cost while improving the learning performance. However,  this method only queries at one pair of fidelity and input at a time, and hence has a risk of bringing in strongly correlated examples to reduce the learning efficiency. In this paper, we propose Batch Multi-Fidelity Active Learning with Budget Constraints (BMFAL-BC), which can promote the diversity of training examples to improve the benefit-cost ratio, while respecting a given budget constraint for batch queries. Hence, our method can be more practically useful. Specifically, we propose a novel batch acquisition function that measures the mutual information between a batch of multi-fidelity queries and the target function, so as to penalize highly correlated queries and encourages diversity. The optimization of the batch acquisition function is challenging in that it involves a combinatorial search over many fidelities while subject to the budget constraint. To address this challenge, we develop a weighted greedy algorithm that can sequentially identify each (fidelity, input) pair, while achieving a near $(1 - 1/e)$-approximation of the optimum. We show the advantage of our method in several computational physics and engineering applications.

        ----

        ## [73] UniCLIP: Unified Framework for Contrastive Language-Image Pre-training

        **Authors**: *Janghyeon Lee, Jongsuk Kim, Hyounguk Shon, Bumsoo Kim, Seung Hwan Kim, Honglak Lee, Junmo Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/072fd0525592b43da661e254bbaadc27-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/072fd0525592b43da661e254bbaadc27-Abstract-Conference.html)

        **Abstract**:

        Pre-training vision-language models with contrastive objectives has shown promising results that are both scalable to large uncurated datasets and transferable to many downstream applications. Some following works have targeted to improve data efficiency by adding self-supervision terms, but inter-domain (image-text) contrastive loss and intra-domain (image-image) contrastive loss are defined on individual spaces in those works, so many feasible combinations of supervision are overlooked. To overcome this issue, we propose UniCLIP, a Unified framework for Contrastive Language-Image Pre-training. UniCLIP integrates the contrastive loss of both inter-domain pairs and intra-domain pairs into a single universal space. The discrepancies that occur when integrating contrastive loss between different domains are resolved by the three key components of UniCLIP: (1) augmentation-aware feature embedding, (2) MP-NCE loss, and (3) domain dependent similarity measure. UniCLIP outperforms previous vision-language pre-training methods on various single- and multi-modality downstream tasks. In our experiments, we show that each component that comprises UniCLIP contributes well to the final performance.

        ----

        ## [74] Efficient Multi-agent Communication via Self-supervised Information Aggregation

        **Authors**: *Cong Guan, Feng Chen, Lei Yuan, Chenghe Wang, Hao Yin, Zongzhang Zhang, Yang Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/075b2875e2b671ddd74aeec0ac9f0357-Abstract-Conference.html)

        **Abstract**:

        Utilizing messages from teammates can improve coordination in cooperative Multi-agent Reinforcement Learning (MARL). To obtain meaningful information for decision-making, previous works typically combine raw messages generated by teammates with local information as inputs for policy. However, neglecting the aggregation of multiple messages poses great inefficiency for policy learning. Motivated by recent advances in representation learning, we argue that efficient message aggregation is essential for good coordination in MARL. In this paper, we propose Multi-Agent communication via Self-supervised Information Aggregation (MASIA), with which agents can aggregate the received messages into compact representations with high relevance to augment the local policy. Specifically, we design a permutation invariant message encoder to generate common information aggregated representation from raw messages and optimize it via reconstructing and shooting future information in a self-supervised manner. Each agent would utilize the most relevant parts of the aggregated representation for decision-making by a novel message extraction mechanism. Empirical results demonstrate that our method significantly outperforms strong baselines on multiple cooperative MARL tasks for various task settings.

        ----

        ## [75] Accelerated Training of Physics-Informed Neural Networks (PINNs) using Meshless Discretizations

        **Authors**: *Ramansh Sharma, Varun Shankar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0764db1151b936aca59249e2c1386101-Abstract-Conference.html)

        **Abstract**:

        Physics-informed neural networks (PINNs) are neural networks trained by using physical laws in the form of partial differential equations (PDEs) as soft constraints. We present a new technique for the accelerated training of PINNs that combines modern scientific computing techniques with machine learning: discretely-trained PINNs (DT-PINNs). The repeated computation of the partial derivative terms in the PINN loss functions via automatic differentiation during training is known to be computationally expensive, especially for higher-order derivatives. DT-PINNs are trained by replacing these exact spatial derivatives with high-order accurate numerical discretizations computed using meshless radial basis function-finite differences (RBF-FD) and applied via sparse-matrix vector multiplication. While in principle any high-order discretization may be used, the use of RBF-FD allows for DT-PINNs to be trained even on point cloud samples placed on irregular domain geometries. Additionally, though traditional PINNs (vanilla-PINNs) are typically stored and trained in 32-bit floating-point (fp32) on the GPU, we show that for DT-PINNs, using fp64 on the GPU leads to significantly faster training times than fp32 vanilla-PINNs with comparable accuracy. We demonstrate the efficiency and accuracy of DT-PINNs via a series of experiments. First, we explore the effect of network depth on both numerical and automatic differentiation of a neural network with random weights and show that RBF-FD approximations of third-order accuracy and above are more efficient while being sufficiently accurate. We then compare the DT-PINNs to vanilla-PINNs on both linear and nonlinear Poisson equations and show that DT-PINNs achieve similar losses with 2-4x faster training times on a consumer GPU. Finally, we also demonstrate that similar results can be obtained for the PINN solution to the heat equation (a space-time problem) by discretizing the spatial derivatives using RBF-FD and using automatic differentiation for the temporal derivative. Our results show that fp64 DT-PINNs offer a superior cost-accuracy profile to fp32 vanilla-PINNs, opening the door to a new paradigm of leveraging scientific computing techniques to support machine learning.

        ----

        ## [76] DOPE: Doubly Optimistic and Pessimistic Exploration for Safe Reinforcement Learning

        **Authors**: *Archana Bura, Aria HasanzadeZonuzy, Dileep Kalathil, Srinivas Shakkottai, Jean-François Chamberland*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/076a93fd42aa85f5ccee921a01d77dd5-Abstract-Conference.html)

        **Abstract**:

        Safe reinforcement learning is extremely challenging--not only must the agent explore an unknown environment, it must do so while ensuring no safety constraint violations. We formulate this safe  reinforcement learning (RL) problem using the framework of a finite-horizon Constrained Markov Decision Process (CMDP) with an unknown transition probability function, where we model the safety requirements as constraints on the expected cumulative costs that must be satisfied during all episodes of learning.  We propose a model-based safe RL algorithm that we call Doubly Optimistic and Pessimistic Exploration (DOPE), and show that it achieves an objective regret $\tilde{O}(|\mathcal{S}|\sqrt{|\mathcal{A}| K})$ without violating the safety constraints during learning, where  $|\mathcal{S}|$ is the number of states, $|\mathcal{A}|$ is the number of actions, and $K$ is the number of learning episodes.  Our key idea is to combine a reward bonus for exploration (optimism) with a conservative constraint (pessimism), in addition to the standard optimistic model-based exploration.  DOPE is not only able to improve the objective regret bound, but also shows a significant empirical performance improvement as compared to earlier optimism-pessimism approaches.

        ----

        ## [77] Improved Regret Analysis for Variance-Adaptive Linear Bandits and Horizon-Free Linear Mixture MDPs

        **Authors**: *Yeoneung Kim, Insoon Yang, Kwang-Sung Jun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/078fa8f77ce55ef6e9cf79275b88acb0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/078fa8f77ce55ef6e9cf79275b88acb0-Abstract-Conference.html)

        **Abstract**:

        In online learning problems, exploiting low variance plays an important role in obtaining tight performance guarantees yet is challenging because variances are often not known a priori.  Recently, considerable progress has been made by Zhang et al. (2021) where they obtain a variance-adaptive regret bound for linear bandits without knowledge of the variances and a horizon-free regret bound for linear mixture Markov decision processes (MDPs).  In this paper, we present novel analyses that improve their regret bounds significantly.  For linear bandits, we achieve $\tilde O(\min\{d\sqrt{K}, d^{1.5}\sqrt{\sum_{k=1}^K \sigma_k^2}\} + d^2)$ where $d$ is the dimension of the features, $K$ is the time horizon, and $\sigma_k^2$ is the noise variance at time step $k$, and $\tilde O$ ignores polylogarithmic dependence, which is a factor of $d^3$ improvement.  For linear mixture MDPs with the assumption of maximum cumulative reward in an episode being in $[0,1]$, we achieve a horizon-free regret bound of $\tilde O(d \sqrt{K} + d^2)$ where $d$ is the number of base models and $K$ is the number of episodes.  This is a factor of $d^{3.5}$ improvement in the leading term and $d^7$ in the lower order term.  Our analysis critically relies on a novel peeling-based regret analysis that leverages the elliptical potential `count' lemma.

        ----

        ## [78] Communication-Efficient Topologies for Decentralized Learning with $O(1)$ Consensus Rate

        **Authors**: *Zhuoqing Song, Weijian Li, Kexin Jin, Lei Shi, Ming Yan, Wotao Yin, Kun Yuan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0790ef700dd0072f4940abda9b7d0005-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0790ef700dd0072f4940abda9b7d0005-Abstract-Conference.html)

        **Abstract**:

        Decentralized optimization is an emerging paradigm in distributed learning in which agents achieve network-wide solutions by  peer-to-peer communication without the central server. Since communication tends to be slower than computation, when each agent communicates with only a few neighboring agents per iteration, they can complete iterations faster than with more agents or a central server. However, the total number of iterations to reach a network-wide solution is affected by the speed at which the information of the agents is ``mixed'' by communication. We found that popular communication topologies either have large degrees (such as stars and complete graphs) or are ineffective at mixing information (such as rings and grids). To address this problem, we propose a new family of topologies, EquiTopo, which has an (almost) constant degree and network-size-independent consensus rate which is used to measure the mixing efficiency.In the proposed family, EquiStatic has a degree of $\Theta(\ln(n))$, where $n$ is the network size, and a series of time-varying one-peer topologies, EquiDyn, has a constant degree of 1. We generate EquiDyn through a certain random sampling procedure. Both of them achieve $n$-independent consensus rate. We apply them to decentralized SGD and decentralized gradient tracking and obtain faster communication and better convergence, both theoretically and empirically. Our code is implemented through BlueFog and available at https://github.com/kexinjinnn/EquiTopo.

        ----

        ## [79] Moderate-fitting as a Natural Backdoor Defender for Pre-trained Language Models

        **Authors**: *Biru Zhu, Yujia Qin, Ganqu Cui, Yangyi Chen, Weilin Zhao, Chong Fu, Yangdong Deng, Zhiyuan Liu, Jingang Wang, Wei Wu, Maosong Sun, Ming Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0799492e7be38b66d10ead5e8809616d-Abstract-Conference.html)

        **Abstract**:

        Despite the great success of pre-trained language models (PLMs) in a large set of natural language processing (NLP) tasks, there has been a growing concern about their security in real-world applications. Backdoor attack, which poisons a small number of training samples by inserting backdoor triggers, is a typical threat to security. Trained on the poisoned dataset, a victim model would perform normally on benign samples but predict the attacker-chosen label on samples containing pre-defined triggers. The vulnerability of PLMs under backdoor attacks has been proved with increasing evidence in the literature. In this paper, we present several simple yet effective training strategies that could effectively defend against such attacks. To the best of our knowledge, this is the first work to explore the possibility of backdoor-free adaptation for PLMs. Our motivation is based on the observation that, when trained on the poisoned dataset, the PLM's adaptation follows a strict order of two stages: (1) a moderate-fitting stage, where the model mainly learns the major features corresponding to the original task instead of subsidiary features of backdoor triggers, and (2) an overfitting stage, where both features are learned adequately. Therefore, if we could properly restrict the PLM's adaptation to the moderate-fitting stage, the model would neglect the backdoor triggers but still achieve satisfying performance on the original task. To this end, we design three methods to defend against backdoor attacks by reducing the model capacity, training epochs, and learning rate, respectively. Experimental results demonstrate the effectiveness of our methods in defending against several representative NLP backdoor attacks. We also perform visualization-based analysis to attain a deeper understanding of how the model learns different features, and explore the effect of the poisoning ratio. Finally, we explore whether our methods could defend against backdoor attacks for the pre-trained CV model. The codes are publicly available at https://github.com/thunlp/Moderate-fitting.

        ----

        ## [80] Dataset Distillation via Factorization

        **Authors**: *Songhua Liu, Kai Wang, Xingyi Yang, Jingwen Ye, Xinchao Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/07bc722f08f096e6ea7ee99349ff0a86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/07bc722f08f096e6ea7ee99349ff0a86-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study dataset distillation (DD), from a novel perspective and introduce a \emph{dataset factorization} approach, termed \emph{HaBa}, which is a plug-and-play strategy portable to any existing DD baseline. Unlike conventional DD approaches that aim to produce distilled and representative samples, \emph{HaBa} explores decomposing a dataset into two components: data \emph{Ha}llucination networks and \emph{Ba}ses, where the latter is fed into the former to reconstruct image samples. The flexible combinations between bases and hallucination networks, therefore, equip the distilled data with exponential informativeness gain, which largely increase the representation capability of distilled datasets. To furthermore increase the data efficiency of compression results, we further introduce a pair of adversarial contrastive \xw{constraints} on the resultant hallucination networks and bases, which increase the diversity of generated images and inject more discriminant information into the factorization. Extensive comparisons and experiments demonstrate that our method can yield significant improvement on downstream classification tasks compared with previous state of the arts, while reducing the total number of compressed parameters by up to 65\%. Moreover, distilled datasets by our approach also achieve \textasciitilde10\% higher accuracy than baseline methods in cross-architecture generalization. Our code is available \href{https://github.com/Huage001/DatasetFactorization}{here}.

        ----

        ## [81] Adaptive Sampling for Discovery

        **Authors**: *Ziping Xu, Eunjae Shim, Ambuj Tewari, Paul M. Zimmerman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/07bc8125400bf4b140c332010756bd9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/07bc8125400bf4b140c332010756bd9b-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study a sequential decision-making problem, called Adaptive Sampling for Discovery (ASD). Starting with a large unlabeled dataset, algorithms for ASD adaptively label the points with the goal to maximize the sum of responses.This problem has wide applications to real-world discovery problems, for example drug discovery with the help of machine learning models. ASD algorithms face the well-known exploration-exploitation dilemma. The algorithm needs to choose points that yield information to improve model estimates but it also needs to exploit the model. We rigorously formulate the problem and propose a general information-directed sampling (IDS) algorithm. We provide theoretical guarantees for the performance of IDS in linear, graph and low-rank models. The benefits of IDS are shown in both simulation experiments and real-data experiments for discovering chemical reaction conditions.

        ----

        ## [82] A Large Scale Search Dataset for Unbiased Learning to Rank

        **Authors**: *Lixin Zou, Haitao Mao, Xiaokai Chu, Jiliang Tang, Wenwen Ye, Shuaiqiang Wang, Dawei Yin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/07f560092a0edceabf55af32a40eaee3-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/07f560092a0edceabf55af32a40eaee3-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        The unbiased learning to rank (ULTR) problem has been greatly advanced by recent deep learning techniques and well-designed debias algorithms. However, promising results on the existing benchmark datasets may not be extended to the practical scenario due to some limitations of existing datasets. First, their semantic feature extractions are outdated while state-of-the-art large-scale pre-trained language models like BERT cannot be utilized due to the lack of original text. Second, display features are incomplete; thus in-depth study on ULTR is impossible such as the displayed abstract for analyzing the click necessary bias. Third, synthetic user feedback has been adopted by most existing datasets and real-world user feedback is greatly missing. To overcome these disadvantages, we introduce the Baidu-ULTR dataset. It involves randomly sampled 1.2 billion searching sessions and 7,008 expert annotated queries(397,572 query document pairs). Baidu-ULTR is the first billion-level dataset for ULTR. Particularly, it offers: (1)the original semantic features and pre-trained language models of different sizes; (2)sufficient display information such as position, displayed height, and displayed abstract, enabling the comprehensive study of multiple displayed biases; and (3)rich user feedback on search result pages (SERPs) like dwelling time, allowing for user engagement optimization and promoting the exploration of multi-task learning in ULTR. Furthermore, we present the design principle of Baidu-ULTR and the performance of representative ULTR algorithms on Baidu-ULTR. The Baidu-ULTR dataset and corresponding baseline implementations are available at https://github.com/ChuXiaokai/baiduultrdataset. The dataset homepage is available at https://searchscience.baidu.com/dataset.html.

        ----

        ## [83] SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation

        **Authors**: *Meng-Hao Guo, Cheng-Ze Lu, Qibin Hou, Zhengning Liu, Ming-Ming Cheng, Shi-Min Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08050f40fff41616ccfc3080e60a301a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08050f40fff41616ccfc3080e60a301a-Abstract-Conference.html)

        **Abstract**:

        We present SegNeXt, a simple convolutional network architecture for semantic segmentation. Recent transformer-based models have dominated the field of se- mantic segmentation due to the efficiency of self-attention in encoding spatial information. In this paper, we show that convolutional attention is a more efficient and effective way to encode contextual information than the self-attention mech- anism in transformers. By re-examining the characteristics owned by successful segmentation models, we discover several key components leading to the perfor- mance improvement of segmentation models. This motivates us to design a novel convolutional attention network that uses cheap convolutional operations. Without bells and whistles, our SegNeXt significantly improves the performance of previous state-of-the-art methods on popular benchmarks, including ADE20K, Cityscapes, COCO-Stuff, Pascal VOC, Pascal Context, and iSAID. Notably, SegNeXt out- performs EfficientNet-L2 w/ NAS-FPN and achieves 90.6% mIoU on the Pascal VOC 2012 test leaderboard using only 1/10 parameters of it. On average, SegNeXt achieves about 2.0% mIoU improvements compared to the state-of-the-art methods on the ADE20K datasets with the same or fewer computations.

        ----

        ## [84] Understanding Hyperdimensional Computing for Parallel Single-Pass Learning

        **Authors**: *Tao Yu, Yichi Zhang, Zhiru Zhang, Christopher De Sa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/080be5eb7e887319ff30c792c2cbc28c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/080be5eb7e887319ff30c792c2cbc28c-Abstract-Conference.html)

        **Abstract**:

        Hyperdimensional computing (HDC) is an emerging learning paradigm that computes with high dimensional binary vectors. There is an active line of research on HDC in the community of emerging hardware because of its energy efficiency and ultra-low latency---but HDC suffers from low model accuracy, with little theoretical understanding of what limits its performance. We propose a new theoretical analysis of the limits of HDC via a consideration of what similarity matrices can be expressed'' by binary vectors, and we show how the limits of HDC can be approached using random Fourier features (RFF). We extend our analysis to the more general class of vector symbolic architectures (VSA), which compute with high-dimensional vectors (hypervectors) that are not necessarily binary. We propose a new class of VSAs, finite group VSAs, which surpass the limits of HDC. Using representation theory, we characterize which similarity matrices can beexpressed'' by finite group VSA hypervectors, and we show how these VSAs can be constructed. Experimental results show that our RFF method and group VSA can both outperform the state-of-the-art HDC model by up to 7.6\% while maintaining hardware efficiency. This work aims to inspire a future interest on HDC in the ML community and connect to the hardware community.

        ----

        ## [85] Syndicated Bandits: A Framework for Auto Tuning Hyper-parameters in Contextual Bandit Algorithms

        **Authors**: *Qin Ding, Yue Kang, Yi-Wei Liu, Thomas Chun Man Lee, Cho-Jui Hsieh, James Sharpnack*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/082e82cae0232f45f27fdd2612c31f8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/082e82cae0232f45f27fdd2612c31f8a-Abstract-Conference.html)

        **Abstract**:

        The stochastic contextual bandit problem, which models the trade-off between exploration and exploitation, has many real applications, including recommender systems, online advertising and clinical trials. As many other machine learning algorithms, contextual bandit algorithms often have one or more hyper-parameters. As an example, in most optimal stochastic contextual bandit algorithms, there is an unknown exploration parameter which controls the trade-off between exploration and exploitation. A proper choice of the hyper-parameters is essential for contextual bandit algorithms to perform well. However, it is infeasible to use offline tuning methods to select hyper-parameters in contextual bandit environment since there is no pre-collected dataset and the decisions have to be made in real time. To tackle this problem, we first propose a two-layer bandit structure for auto tuning the exploration parameter and further generalize it to the Syndicated Bandits framework which can learn multiple hyper-parameters dynamically in contextual bandit environment. We derive the regret bounds of our proposed Syndicated Bandits framework and show it can avoid its regret dependent exponentially in the number of hyper-parameters to be tuned. Moreover, it achieves optimal regret bounds under certain scenarios. Syndicated Bandits framework is general enough to handle the tuning tasks in many popular contextual bandit algorithms, such as LinUCB, LinTS, UCB-GLM, etc. Experiments on both synthetic and real datasets validate the effectiveness of our proposed framework.

        ----

        ## [86] Benign, Tempered, or Catastrophic: Toward a Refined Taxonomy of Overfitting

        **Authors**: *Neil Mallinar, James B. Simon, Amirhesam Abedsoltan, Parthe Pandit, Misha Belkin, Preetum Nakkiran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08342dc6ab69f23167b4123086ad4d38-Abstract-Conference.html)

        **Abstract**:

        The practical success of overparameterized neural networks has motivated the recent scientific study of \emph{interpolating methods}-- learning methods which are able fit their training data perfectly. Empirically, certain interpolating methods can fit noisy training data without catastrophically bad test performance, which defies standard intuitions from statistical learning theory. Aiming to explain this, a large body of recent work has studied \emph{benign overfitting}, a behavior seen in certain asymptotic settings under which interpolating methods approach Bayes-optimality, even in the presence of noise. In this work, we argue that, while benign overfitting has been instructive to study, real interpolating methods like deep networks do not fit benignly. That is, noise in the train set leads to suboptimal generalization, suggesting that these methods fall in an intermediate regime between benign and catastrophic overfitting, in which asymptotic risk is neither is neither Bayes-optimal nor unbounded, with the confounding effect of the noise being ``tempered" but non-negligible. We call this behavior \textit{tempered overfitting}. We first provide broad empirical evidence for our three-part taxonomy, demonstrating that deep neural networks and kernel machines fit to noisy data can be reasonably well classified as benign, tempered, or catastrophic. We then specialize to kernel (ridge) regression (KR), obtaining conditions on the ridge parameter and kernel eigenspectrum under which KR exhibits each of the three behaviors, demonstrating the consequences for KR with common kernels and trained neural networks of infinite width using experiments on natural and synthetic datasets.

        ----

        ## [87] Pre-trained Adversarial Perturbations

        **Authors**: *Yuanhao Ban, Yinpeng Dong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/084727e8abf90a8365b940036329cb6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/084727e8abf90a8365b940036329cb6f-Abstract-Conference.html)

        **Abstract**:

        Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against the fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared to the state-of-the-art methods.

        ----

        ## [88] An Empirical Study on Disentanglement of Negative-free Contrastive Learning

        **Authors**: *Jinkun Cao, Ruiqian Nai, Qing Yang, Jialei Huang, Yang Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0850e04a62e0f3407780852581c5fcf4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0850e04a62e0f3407780852581c5fcf4-Abstract-Conference.html)

        **Abstract**:

        Negative-free contrastive learning methods have attracted a lot of attention with simplicity and impressive performances for large-scale pretraining. However, its disentanglement property remains unexplored. In this paper, we examine negative-free contrastive learning methods to study the disentanglement property empirically. We find that existing disentanglement metrics fail to make meaningful measurements for high-dimensional representation models, so we propose a new disentanglement metric based on Mutual Information between latent representations and data factors. With this proposed metric, we benchmark the disentanglement property of negative-free contrastive learning on both popular synthetic datasets and a real-world dataset CelebA. Our study shows that the investigated methods can learn a well-disentangled subset of representation. As far as we know, we are the first to extend the study of disentangled representation learning to high-dimensional representation space and introduce negative-free contrastive learning methods into this area. The source code of this paper is available at https://github.com/noahcao/disentanglementlibmed.

        ----

        ## [89] MABSplit: Faster Forest Training Using Multi-Armed Bandits

        **Authors**: *Mo Tiwari, Ryan Kang, Jaeyong Lee, Chris Piech, Ilan Shomorony, Sebastian Thrun, Martin J. Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08857467641ad82f635023d530605b4c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08857467641ad82f635023d530605b4c-Abstract-Conference.html)

        **Abstract**:

        Random forests are some of the most widely used machine learning models today, especially in domains that necessitate interpretability. We present an algorithm that accelerates the training of random forests and other popular tree-based learning methods. At the core of our algorithm is a novel node-splitting subroutine, dubbed MABSplit, used to efficiently find split points when constructing decision trees. Our algorithm borrows techniques from the multi-armed bandit literature to judiciously determine how to allocate samples and computational power across candidate split points. We provide theoretical guarantees that MABSplit improves the sample complexity of each node split from linear to logarithmic in the number of data points. In some settings, MABSplit leads to 100x faster training (an 99% reduction in training time) without any decrease in generalization performance. We demonstrate similar speedups when MABSplit is used across a variety of forest-based variants, such as Extremely Random Forests and Random Patches. We also show our algorithm can be used in both classification and regression tasks. Finally, we show that MABSplit outperforms existing methods in generalization performance and feature importance calculations under a fixed computational budget. All of our experimental results are reproducible via a one-line script at https://github.com/ThrunGroup/FastForest.

        ----

        ## [90] Counterfactual Fairness with Partially Known Causal Graph

        **Authors**: *Aoqi Zuo, Susan Wei, Tongliang Liu, Bo Han, Kun Zhang, Mingming Gong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08887999616116910fccec17a63584b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08887999616116910fccec17a63584b5-Abstract-Conference.html)

        **Abstract**:

        Fair machine learning aims to avoid treating individuals or sub-populations unfavourably based on \textit{sensitive attributes}, such as gender and race. Those methods in fair machine learning that are built on causal inference ascertain discrimination and bias through causal effects. Though causality-based fair learning is attracting increasing attention, current methods assume the true causal graph is fully known. This paper proposes a general method to achieve the notion of counterfactual fairness when the true causal graph is unknown. To select features that lead to counterfactual fairness, we derive the conditions and algorithms to identify ancestral relations between variables on a \textit{Partially Directed Acyclic Graph (PDAG)}, specifically, a class of causal DAGs that can be learned from observational data combined with domain knowledge. Interestingly, we find that counterfactual fairness can be achieved as if the true causal graph were fully known, when specific background knowledge is provided: the sensitive attributes do not have ancestors in the causal graph. Results on both simulated and real-world datasets demonstrate the effectiveness of our method.

        ----

        ## [91] Controlled Sparsity via Constrained Optimization or: How I Learned to Stop Tuning Penalties and Love Constraints

        **Authors**: *Jose Gallego-Posada, Juan Ramirez, Akram Erraqabi, Yoshua Bengio, Simon Lacoste-Julien*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/089b592cccfafdca8e0178e85b609f19-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/089b592cccfafdca8e0178e85b609f19-Abstract-Conference.html)

        **Abstract**:

        The performance of trained neural networks is robust to harsh levels of pruning. Coupled with the ever-growing size of deep learning models, this observation has motivated extensive research on learning sparse models. In this work, we focus on the task of controlling the level of sparsity when performing sparse learning. Existing methods based on sparsity-inducing penalties involve expensive trial-and-error tuning of the penalty factor, thus lacking direct control of the resulting model sparsity. In response, we adopt a constrained formulation: using the gate mechanism proposed by Louizos et al. (2018), we formulate a constrained optimization problem where sparsification is guided by the training objective and the desired sparsity target in an end-to-end fashion. Experiments on CIFAR-{10, 100}, TinyImageNet, and ImageNet using WideResNet and ResNet{18, 50} models validate the effectiveness of our proposal and demonstrate that we can reliably achieve pre-determined sparsity targets without compromising on predictive performance.

        ----

        ## [92] Algorithms and Hardness for Learning Linear Thresholds from Label Proportions

        **Authors**: *Rishi Saket*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08a9e28c96d016dd63903ab51cd085b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08a9e28c96d016dd63903ab51cd085b0-Abstract-Conference.html)

        **Abstract**:

        We study the learnability of linear threshold functions (LTFs) in the learning from label proportions (LLP) framework. In this, the feature-vector classifier is learnt from bags of feature-vectors and their corresponding observed label proportions which are satisfied by (i.e., consistent with) some unknown LTF. This problem has been investigated in recent work (Saket21)  which gave an algorithm to produce an LTF that satisfies at least $(2/5)$-fraction of a satisfiable collection of bags, each of size $\leq 2$, by solving and rounding a natural SDP relaxation. However, this SDP relaxation is specific to at most $2$-sized bags and does not apply to bags of larger size.     In this work we provide a fairly non-trivial SDP relaxation of a  non-quadratic formulation for bags of size $3$. We analyze its rounding procedure using novel matrix decomposition techniques to obtain an algorithm which outputs an LTF satisfying at least $(1/12)$-fraction of the bags of size $\leq 3$. We also apply our techniques to bags of size $q \geq 4$ to provide a $\Omega\left(1/q\right)$-approximation guarantee for a weaker notion of satisfiability. We include comparative experiments on simulated data demonstrating the applicability of our algorithmic techniques.    From the complexity side we provide a hardness reduction to produce instances with bags of any constant size $q$. Our reduction proves the NP-hardness of satisfying  more than $({1}/{q}) + o(1)$ fraction of a satisfiable collection of such bags using as hypothesis any function of constantly many LTFs, showing thereby that the problem is harder to approximate as the bag size $q$ increases. Using a strengthened analysis, for $q=2$ we obtain a $({4}/{9}) +o(1)$ hardness factor for this problem, improving upon the $({1}/{2}) + o(1)$ factor shown by Saket21.

        ----

        ## [93] Predictive Coding beyond Gaussian Distributions

        **Authors**: *Luca Pinchetti, Tommaso Salvatori, Yordan Yordanov, Beren Millidge, Yuhang Song, Thomas Lukasiewicz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08f9de0232c0b485110237f6e6cf88f1-Abstract-Conference.html)

        **Abstract**:

        A large amount of recent research has the far-reaching goal of finding training methods for deep neural networks that can serve as alternatives to backpropagation~(BP). A prominent example is predictive coding (PC), which is a neuroscience-inspired method that performs inference on hierarchical Gaussian generative models. These methods, however, fail to keep up with modern neural networks, as they are unable to replicate the dynamics of complex layers and activation functions. In this work, we solve this problem by generalizing PC to arbitrary probability distributions, enabling the training of architectures, such as transformers, that are hard to approximate with only Gaussian assumptions. We perform three experimental analyses. First, we study the gap between our method and the standard formulation of PC on multiple toy examples. Second, we test the reconstruction quality on variational autoencoders, where our method reaches the same reconstruction quality as BP. Third, we show that our method allows us to train transformer networks and achieve performance comparable with BP on conditional language models. More broadly, this method allows neuroscience-inspired  learning to be applied to multiple domains, since the internal distributions can be flexibly adapted to the data, tasks, and architectures used.

        ----

        ## [94] Semi-supervised Active Linear Regression

        **Authors**: *Nived Rajaraman, Devvrit, Pranjal Awasthi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08fe4b20d554296e503f5a43795c78d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08fe4b20d554296e503f5a43795c78d6-Abstract-Conference.html)

        **Abstract**:

        Labeled data often comes at a high cost as it may require recruiting human labelers or running costly experiments. At the same time, in many practical scenarios, one already has access to a partially labeled, potentially biased dataset that can help with the learning task at hand. Motivated by such settings, we formally initiate a study of ``semi-supervised active learning'' through the frame of linear regression. Here, the learner has access to a dataset $X \in \mathbb{R}^{(n_{\text{un}}+n_{\text{lab}}) \times d}$ composed of $n_{\text{un}}$ unlabeled examples that a learner can actively query, and $n_{\text{lab}}$ examples labeled a priori. Denoting the true labels by $Y \in \mathbb{R}^{n_{\text{un}} + n_{\text{lab}}}$, the learner's objective is to find $\widehat{\beta} \in \mathbb{R}^d$ such that,$$\| X \widehat{\beta} - Y \|_2^2 \le (1 + \epsilon) \min_{\beta \in \mathbb{R}^d} \| X \beta - Y \|_2^2$$while querying the labels of as few unlabeled points as possible. In this paper, we introduce an instance dependent parameter called the reduced rank, denoted $\text{R}_X$, and propose an efficient algorithm with query complexity $O(\text{R}_X/\epsilon)$. This result directly implies improved upper bounds for two important special cases: $(i)$ active ridge regression, and $(ii)$ active kernel ridge regression, where the reduced-rank equates to the ``statistical dimension'', $\textsf{sd}_\lambda$ and ``effective dimension'', $d_\lambda$ of the problem respectively, where $\lambda \ge 0$ denotes the regularization parameter. Finally, we introduce a distributional version of the problem as a special case of the agnostic formulation we consider earlier; here, for every $X$, we prove a matching instance-wise lower bound of $\Omega (\text{R}_X / \epsilon)$ on the query complexity of any algorithm.

        ----

        ## [95] Boosting Barely Robust Learners: A New Perspective on Adversarial Robustness

        **Authors**: *Avrim Blum, Omar Montasser, Greg Shakhnarovich, Hongyang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/08fe50bf209c57eecf0804f9f9ed639f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/08fe50bf209c57eecf0804f9f9ed639f-Abstract-Conference.html)

        **Abstract**:

        We present an oracle-efficient algorithm for boosting the adversarial robustness of barely robust learners. Barely robust learning algorithms learn predictors that are adversarially robust only on a small fraction $\beta \ll 1$ of the data distribution. Our proposed notion of barely robust learning requires robustness with respect to a ``larger'' perturbation set; which we show is necessary for strongly robust learning, and that weaker relaxations are not sufficient for strongly robust learning. Our results reveal a qualitative and quantitative equivalence between two seemingly unrelated problems: strongly robust learning and barely robust learning.

        ----

        ## [96] Decision-Focused Learning without Decision-Making: Learning Locally Optimized Decision Losses

        **Authors**: *Sanket Shah, Kai Wang, Bryan Wilder, Andrew Perrault, Milind Tambe*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0904c7edde20d7134a77fc7f9cd86ea2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0904c7edde20d7134a77fc7f9cd86ea2-Abstract-Conference.html)

        **Abstract**:

        Decision-Focused Learning (DFL) is a paradigm for tailoring a predictive model to a downstream optimization task that uses its predictions in order to perform better \textit{on that specific task}. The main technical challenge associated with DFL is that it requires being able to differentiate through the optimization problem, which is difficult due to discontinuous solutions and other challenges. Past work has largely gotten around this this issue by \textit{handcrafting} task-specific surrogates to the original optimization problem that provide informative gradients when differentiated through. However, the need to handcraft surrogates for each new task limits the usability of DFL. In addition, there are often no guarantees about the convexity of the resulting surrogates and, as a result, training a predictive model using them can lead to inferior local optima. In this paper, we do away with surrogates altogether and instead \textit{learn} loss functions that capture task-specific information. To the best of our knowledge, ours is the first approach that entirely replaces the optimization component of decision-focused learning with a loss that is automatically learned. Our approach (a) only requires access to a black-box oracle that can solve the optimization problem and is thus \textit{generalizable}, and (b) can be \textit{convex by construction} and so can be easily optimized over. We evaluate our approach on three resource allocation problems from the literature and find that our approach outperforms learning without taking into account task-structure in all three domains, and even hand-crafted surrogates from the literature.

        ----

        ## [97] Neural Stochastic PDEs: Resolution-Invariant Learning of Continuous Spatiotemporal Dynamics

        **Authors**: *Cristopher Salvi, Maud Lemercier, Andris Gerasimovics*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/091166620a04a289c555f411d8899049-Abstract-Conference.html)

        **Abstract**:

        Stochastic partial differential equations (SPDEs) are the mathematical tool of choice for modelling spatiotemporal PDE-dynamics under the influence of randomness. Based on the notion of mild solution of an SPDE, we introduce a novel neural architecture to learn solution operators of PDEs with (possibly stochastic) forcing from partially observed data. The proposed Neural SPDE model provides an extension to two popular classes of physics-inspired architectures. On the one hand, it extends Neural CDEs and variants -- continuous-time analogues of RNNs -- in that it is capable of processing incoming sequential information arriving at arbitrary spatial resolutions. On the other hand, it extends Neural Operators -- generalizations of neural networks to model mappings between spaces of functions -- in that it can parameterize solution operators of SPDEs depending simultaneously on the initial condition and a realization of the driving noise. By performing operations in the spectral domain, we show how a Neural SPDE can be evaluated in two ways, either by calling an ODE solver (emulating a spectral Galerkin scheme), or by solving a fixed point problem. Experiments on various semilinear SPDEs, including the stochastic Navier-Stokes equations, demonstrate how the Neural SPDE model is capable of learning complex spatiotemporal dynamics in a resolution-invariant way, with better accuracy and lighter training data requirements compared to alternative models, and up to 3 orders of magnitude faster than traditional solvers.

        ----

        ## [98] Okapi: Generalising Better by Making Statistical Matches Match

        **Authors**: *Myles Bartlett, Sara Romiti, Viktoriia Sharmanska, Novi Quadrianto*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0918183ced31affb7ce0345e45ac1943-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0918183ced31affb7ce0345e45ac1943-Abstract-Conference.html)

        **Abstract**:

        We propose Okapi, a simple, efficient, and general method for robust semi-supervised learning based on online statistical matching. Our method uses a nearest-neighbours-based matching procedure to generate cross-domain views for a consistency loss, while eliminating statistical outliers. In order to perform the online matching in a runtime- and memory-efficient way, we draw upon the self-supervised literature and combine a memory bank with a slow-moving momentum encoder. The consistency loss is applied within the feature space, rather than on the predictive distribution, making the method agnostic to both the modality and the task in question. We experiment on the WILDS 2.0 datasets Sagawa et al., which significantly expands the range of modalities, applications, and shifts available for studying and benchmarking real-world unsupervised adaptation. Contrary to Sagawa et al., we show that it is in fact possible to leverage additional unlabelled data to improve upon empirical risk minimisation (ERM) results with the right method. Our method outperforms the baseline methods in terms of out-of-distribution (OOD) generalisation on the iWildCam (a multi-class classification task) and PovertyMap (a regression task) image datasets as well as the CivilComments (a binary classification task) text dataset. Furthermore, from a qualitative perspective, we show the matches obtained from the learned encoder are strongly semantically related. Code for our paper is publicly available at https://github.com/wearepal/okapi/.

        ----

        ## [99] Revisiting Heterophily For Graph Neural Networks

        **Authors**: *Sitao Luan, Chenqing Hua, Qincheng Lu, Jiaqi Zhu, Mingde Zhao, Shuyuan Zhang, Xiao-Wen Chang, Doina Precup*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/092359ce5cf60a80e882378944bf1be4-Abstract-Conference.html)

        **Abstract**:

        Graph Neural Networks (GNNs) extend basic Neural Networks (NNs) by using graph structures based on the relational inductive bias (homophily assumption). While GNNs have been commonly believed to outperform NNs in real-world tasks, recent work has identified a non-trivial set of datasets where their performance compared to NNs is not satisfactory. Heterophily has been considered the main cause of this empirical observation and numerous works have been put forward to address it. In this paper, we first revisit the widely used homophily metrics and point out that their consideration of only graph-label consistency is a shortcoming. Then, we study heterophily from the  perspective of post-aggregation node similarity and define new homophily metrics, which are potentially advantageous compared to existing ones. Based on this investigation, we prove that some harmful cases of heterophily can be effectively addressed by local diversification operation. Then, we propose the Adaptive Channel Mixing (ACM), a framework to adaptively exploit aggregation, diversification and identity channels to extract richer localized information in each baseline GNN layer. ACM is more powerful than the commonly used uni-channel framework for node classification tasks on heterophilic graphs. When evaluated on 10 benchmark node classification tasks, ACM-augmented baselines consistently achieve significant performance gain, exceeding state-of-the-art GNNs on most  tasks without incurring significant computational burden. (Code: https://github.com/SitaoLuan/ACM-GNN)

        ----

        ## [100] Museformer: Transformer with Fine- and Coarse-Grained Attention for Music Generation

        **Authors**: *Botao Yu, Peiling Lu, Rui Wang, Wei Hu, Xu Tan, Wei Ye, Shikun Zhang, Tao Qin, Tie-Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/092c2d45005ea2db40fc24c470663416-Abstract-Conference.html)

        **Abstract**:

        Symbolic music generation aims to generate music scores automatically. A recent trend is to use Transformer or its variants in music generation, which is, however, suboptimal, because the full attention cannot efficiently model the typically long music sequences (e.g., over 10,000 tokens), and the existing models have shortcomings in generating musical repetition structures. In this paper, we propose Museformer, a Transformer with a novel fine- and coarse-grained attention for music generation. Specifically, with the fine-grained attention, a token of a specific bar directly attends to all the tokens of the bars that are most relevant to music structures (e.g., the previous 1st, 2nd, 4th and 8th bars, selected via similarity statistics); with the coarse-grained attention, a token only attends to the summarization of the other bars rather than each token of them so as to reduce the computational cost. The advantages are two-fold. First, it can capture both music structure-related correlations via the fine-grained attention, and other contextual information via the coarse-grained attention. Second, it is efficient and can model over 3X longer music sequences compared to its full-attention counterpart. Both objective and subjective experimental results demonstrate its ability to generate long music sequences with high quality and better structures.

        ----

        ## [101] Emergent Communication: Generalization and Overfitting in Lewis Games

        **Authors**: *Mathieu Rita, Corentin Tallec, Paul Michel, Jean-Bastien Grill, Olivier Pietquin, Emmanuel Dupoux, Florian Strub*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract-Conference.html)

        **Abstract**:

        Lewis signaling games are a class of simple communication games for simulating the emergence of language. In these games, two agents must agree on a communication protocol in order to solve a cooperative task. Previous work has shown that agents trained to play this game with reinforcement learning tend to develop languages that display undesirable properties from a linguistic point of view (lack of generalization, lack of compositionality, etc). In this paper, we aim to provide better understanding of this phenomenon by analytically studying the learning problem in Lewis games. As a core contribution, we demonstrate that the standard objective in Lewis games can be decomposed in two components: a co-adaptation loss and an information loss. This decomposition enables us to surface two potential sources of overfitting, which we show may undermine the emergence of a structured communication protocol. In particular, when we control for overfitting on the co-adaptation loss, we recover desired properties in the emergent languages: they are more compositional and  generalize better.

        ----

        ## [102] Towards Efficient Post-training Quantization of Pre-trained Language Models

        **Authors**: *Haoli Bai, Lu Hou, Lifeng Shang, Xin Jiang, Irwin King, Michael R. Lyu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/096347b4efc264ae7f07742fea34af1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/096347b4efc264ae7f07742fea34af1f-Abstract-Conference.html)

        **Abstract**:

        Network quantization has gained increasing attention with the rapid growth of large pre-trained language models~(PLMs). However, most existing quantization methods for PLMs follow quantization-aware training~(QAT) that requires end-to-end training with full access to the entire dataset. Therefore, they suffer from slow training, large memory overhead, and data accessibility issues. In this paper, we study post-training quantization~(PTQ) of PLMs, and propose module-wise quantization error minimization~(MREM), an efficient solution to mitigate these issues. By partitioning the PLM into multiple modules, we minimize the reconstruction error incurred by quantization for each module. In addition, we design a new model parallel training strategy such that each module can be trained locally on separate computing devices without waiting for preceding modules, which brings nearly the theoretical training speed-up (e.g., $4\times$ on $4$ GPUs). Experiments on GLUE and SQuAD benchmarks show that our proposed PTQ solution not only performs close to QAT, but also enjoys significant reductions in training time, memory overhead, and data consumption.

        ----

        ## [103] TweetNERD - End to End Entity Linking Benchmark for Tweets

        **Authors**: *Shubhanshu Mishra, Aman Saini, Raheleh Makki, Sneha Mehta, Aria Haghighi, Ali Mollahosseini*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/09723c9f291f6056fd1885081859c186-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Named Entity Recognition and Disambiguation (NERD) systems are foundational for information retrieval, question answering, event detection, and other natural language processing (NLP) applications. We introduce TweetNERD, a dataset of 340K+ Tweets across 2010-2021, for benchmarking NERD systems on Tweets. This is the largest and most temporally diverse open sourced dataset benchmark for NERD on Tweets and can be used to facilitate research in this area. We describe evaluation setup with TweetNERD for three NERD tasks: Named Entity Recognition (NER), Entity Linking with True Spans (EL), and End to End Entity Linking (End2End); and provide performance of existing publicly available methods on specific TweetNERD splits. TweetNERD is available at: https://doi.org/10.5281/zenodo.6617192 under Creative Commons Attribution 4.0 International (CC BY 4.0) license. Check out more details at https://github.com/twitter-research/TweetNERD.

        ----

        ## [104] Riemannian Neural SDE: Learning Stochastic Representations on Manifolds

        **Authors**: *Sung Woo Park, Hyomin Kim, Kyungjae Lee, Junseok Kwon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/098491b37deebbe6c007e69815729e09-Abstract-Conference.html)

        **Abstract**:

        In recent years, the neural stochastic differential equation (NSDE) has gained attention for modeling stochastic representations with great success in various types of applications. However, it typically loses expressivity when the data representation is manifold-valued. To address this issue, we suggest a principled method for expressing the stochastic representation with the Riemannian neural SDE (RNSDE), which extends the conventional Euclidean NSDE. Empirical results for various tasks demonstrate that the proposed method significantly outperforms baseline methods.

        ----

        ## [105] OccGen: Selection of Real-world Multilingual Parallel Data Balanced in Gender within Occupations

        **Authors**: *Marta R. Costa-jussà, Christine Basta, Oriol Domingo, André Rubungo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/09933f07ae2ccbca7212bb4e43de8db0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/09933f07ae2ccbca7212bb4e43de8db0-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        This paper describes the OCCGEN toolkit, which allows extracting multilingual parallel data balanced in gender within occupations. OCCGEN can extract datasets that reflect gender diversity (beyond binary) more fairly in society to be further used to explicitly mitigate occupational gender stereotypes. We propose two use cases that extract evaluation datasets for machine translation in four high-resourcelanguages from different linguistic families and in a low-resource African language. Our analysis of these use cases shows that translation outputs in high-resource languages tend to worsen in feminine subsets (compared to masculine). This can be explained because less attention is paid to the source sentence. Then, more attention is given to the target prefix overgeneralizing to the most frequent masculine forms.

        ----

        ## [106] Learning in Observable POMDPs, without Computationally Intractable Oracles

        **Authors**: *Noah Golowich, Ankur Moitra, Dhruv Rohatgi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/099607cd970f4e1ac2fdd30624dffff8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/099607cd970f4e1ac2fdd30624dffff8-Abstract-Conference.html)

        **Abstract**:

        Much of reinforcement learning theory is built on top of oracles that are computationally hard to implement. Specifically for learning near-optimal policies in Partially Observable Markov Decision Processes (POMDPs), existing algorithms either need to make strong assumptions about the model dynamics (e.g. deterministic transitions) or assume access to an oracle for solving a hard optimistic planning or estimation problem as a subroutine. In this work we develop the first oracle-free learning algorithm for POMDPs under reasonable assumptions. Specifically, we give a quasipolynomial-time end-to-end algorithm for learning in ``observable'' POMDPs, where observability is the assumption that well-separated distributions over states induce well-separated distributions over observations. Our techniques circumvent the more traditional approach of using the principle of optimism under uncertainty to promote exploration, and instead give a novel application of barycentric spanners to constructing policy covers.

        ----

        ## [107] Cross-Image Context for Single Image Inpainting

        **Authors**: *Tingliang Feng, Wei Feng, Weiqi Li, Di Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/09b6e009612875dd0a7291d5f4fd8b49-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/09b6e009612875dd0a7291d5f4fd8b49-Abstract-Conference.html)

        **Abstract**:

        Visual context is of crucial importance for image inpainting. The contextual information captures the appearance and semantic correlation between the image regions, helping to propagate the information of the complete regions for reasoning the content of the corrupted regions. Many inpainting methods compute the visual context based on the regions within the single image. In this paper, we propose the Cross-Image Context Memory (CICM) for learning and using the cross-image context to recover the corrupted regions. CICM consists of multiple sets of the cross-image representations learned from the image regions with different visual patterns. The regional representations are learned across different images, thus providing richer context that benefit the inpainting task. The experimental results demonstrate the effectiveness and generalization of CICM, which achieves state-of-the-art performances on various datasets for single image inpainting.

        ----

        ## [108] Efficient and Effective Augmentation Strategy for Adversarial Training

        **Authors**: *Sravanti Addepalli, Samyak Jain, Venkatesh Babu R.*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/09d22e4155aa4fdadf3dac8c6bd940fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/09d22e4155aa4fdadf3dac8c6bd940fe-Abstract-Conference.html)

        **Abstract**:

        Adversarial training of Deep Neural Networks is known to be significantly more data-hungry when compared to standard training. Furthermore, complex data augmentations such as AutoAugment, which have led to substantial gains in standard training of image classifiers, have not been successful with Adversarial Training. We first explain this contrasting behavior by viewing augmentation during training as a problem of domain generalization, and further propose Diverse Augmentation-based Joint Adversarial Training (DAJAT) to use data augmentations effectively in adversarial training. We aim to handle the conflicting goals of enhancing the diversity of the training dataset and training with data that is close to the test distribution by using a combination of simple and complex augmentations with separate batch normalization layers during training. We further utilize the popular Jensen-Shannon divergence loss to encourage the \emph{joint} learning of the \emph{diverse augmentations}, thereby allowing simple augmentations to guide the learning of complex ones. Lastly, to improve the computational efficiency of the proposed method, we propose and utilize a two-step defense, Ascending Constraint Adversarial Training (ACAT), that uses an increasing epsilon schedule and weight-space smoothing to prevent gradient masking. The proposed method DAJAT achieves substantially better robustness-accuracy trade-off when compared to existing methods on the RobustBench Leaderboard on ResNet-18 and WideResNet-34-10. The code for implementing DAJAT is available here: https://github.com/val-iisc/DAJAT

        ----

        ## [109] Multi-Sample Training for Neural Image Compression

        **Authors**: *Tongda Xu, Yan Wang, Dailan He, Chenjian Gao, Han Gao, Kunzan Liu, Hongwei Qin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/09e7121c046e0ad54aada522d3e1f967-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/09e7121c046e0ad54aada522d3e1f967-Abstract-Conference.html)

        **Abstract**:

        This paper considers the problem of lossy neural image compression (NIC). Current state-of-the-art (SOTA) methods adopt uniform posterior to approximate quantization noise, and single-sample pathwise estimator to approximate the gradient of evidence lower bound (ELBO). In this paper, we propose to train NIC with multiple-sample importance weighted autoencoder (IWAE) target, which is tighter than ELBO and converges to log likelihood as sample size increases. First, we identify that the uniform posterior of NIC has special properties, which affect the variance and bias of pathwise and score function estimators of the IWAE target. Moreover, we provide insights on a commonly adopted trick in NIC from gradient variance perspective. Based on those analysis, we further propose multiple-sample NIC (MS-NIC), an enhanced IWAE target for NIC. Experimental results demonstrate that it improves SOTA NIC methods. Our MS-NIC is plug-and-play, and can be easily extended to neural video compression.

        ----

        ## [110] Adaptive Data Debiasing through Bounded Exploration

        **Authors**: *Yifan Yang, Yang Liu, Parinaz Naghizadeh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a166a3d98720697d9028bbe592fa177-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a166a3d98720697d9028bbe592fa177-Abstract-Conference.html)

        **Abstract**:

        Biases in existing datasets used to train algorithmic decision rules can raise ethical and economic concerns due to the resulting disparate treatment of different groups. We propose an algorithm for sequentially debiasing such datasets through adaptive and bounded exploration in a classification problem with costly and censored feedback. Exploration in this context means that at times, and to a judiciously-chosen extent, the decision maker deviates from its (current) loss-minimizing rule, and instead accepts some individuals that would otherwise be rejected, so as to reduce statistical data biases. Our proposed algorithm includes parameters that can be used to balance between the ultimate goal of removing data biases -- which will in turn lead to more accurate and fair decisions, and the exploration risks incurred to achieve this goal. We analytically show that such exploration can help debias data in certain distributions. We further investigate how fairness criteria can work in conjunction with our data debiasing algorithm. We illustrate the performance of our algorithm using experiments on synthetic and real-world datasets.

        ----

        ## [111] Learning to Navigate Wikipedia by Taking Random Walks

        **Authors**: *Manzil Zaheer, Kenneth Marino, Will Grathwohl, John Schultz, Wendy Shang, Sheila Babayan, Arun Ahuja, Ishita Dasgupta, Christine Kaeser-Chen, Rob Fergus*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a245311a23460d1846043d4156445d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a245311a23460d1846043d4156445d6-Abstract-Conference.html)

        **Abstract**:

        A fundamental ability of an intelligent web-based agent is seeking out and acquiring new information. Internet search engines reliably find the correct vicinity but the top results may be a few links away from the desired target. A complementary approach is navigation via hyperlinks, employing a policy that comprehends local content and selects a link that moves it closer to the target. In this paper, we show that behavioral cloning of randomly sampled trajectories is sufficient to learn an effective link selection policy. We demonstrate the approach on a graph version of Wikipedia with 38M nodes and 387M edges. The model is able to efficiently navigate between nodes 5 and 20 steps apart 96% and 92% of the time, respectively. We then use the resulting embeddings and policy in downstream fact verification and question answering tasks where, in combination with basic TF-IDF search and ranking methods, they are competitive results to the state-of-the-art methods.

        ----

        ## [112] When does return-conditioned supervised learning work for offline reinforcement learning?

        **Authors**: *David Brandfonbrener, Alberto Bietti, Jacob Buckman, Romain Laroche, Joan Bruna*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a2f65c9d2313b71005e600bd23393fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a2f65c9d2313b71005e600bd23393fe-Abstract-Conference.html)

        **Abstract**:

        Several recent works have proposed a class of algorithms for the offline reinforcement learning (RL) problem that we will refer to as return-conditioned supervised learning (RCSL). RCSL algorithms learn the distribution of actions conditioned on both the state and the return of the trajectory. Then they define a policy by conditioning on achieving high return. In this paper, we provide a rigorous study of the capabilities and limitations of RCSL something which is crucially missing in previous work. We find that RCSL returns the optimal policy under a set of assumptions that are stronger than those needed for the more traditional dynamic programming-based algorithms. We provide specific examples of MDPs and datasets that illustrate the necessity of these assumptions and the limits of RCSL. Finally, we present empirical evidence that these limitations will also cause issues in practice by providing illustrative experiments in simple point-mass environments and on datasets from the D4RL benchmark.

        ----

        ## [113] Provable Subspace Identification Under Post-Nonlinear Mixtures

        **Authors**: *Qi Lyu, Xiao Fu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a6059857ae5c82ea9726ee9282a7145-Abstract-Conference.html)

        **Abstract**:

        Unsupervised mixture learning (UML) aims at identifying linearly or nonlinearly mixed latent components in a blind manner. UML is known to be challenging: Even learning linear mixtures requires highly nontrivial analytical tools, e.g., independent component analysis or nonnegative matrix factorization. In this work, the post-nonlinear (PNL) mixture model---where {\it unknown} element-wise nonlinear functions are imposed onto a linear mixture---is revisited. The PNL model is widely employed in different fields ranging from brain signal classification, speech separation, remote sensing, to causal discovery. To identify and remove the unknown nonlinear functions, existing works often assume different properties on the latent components (e.g., statistical independence or probability-simplex structures). This work shows that under a carefully designed UML criterion, the existence of a nontrivial {\it null space} associated with the underlying mixing system suffices to guarantee identification/removal of the unknown nonlinearity. Compared to prior works, our finding largely relaxes the conditions of attaining PNL identifiability, and thus may benefit applications where no strong structural information on the latent components is known. A finite-sample analysis is offered to characterize the performance of the proposed approach under realistic settings. To implement the proposed learning criterion, a block coordinate descent algorithm is proposed. A series of numerical experiments corroborate our theoretical claims.

        ----

        ## [114] S3-NeRF: Neural Reflectance Field from Shading and Shadow under a Single Viewpoint

        **Authors**: *Wenqi Yang, Guanying Chen, Chaofeng Chen, Zhenfang Chen, Kwan-Yee K. Wong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a630402ee92620dc2de3b704181de9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a630402ee92620dc2de3b704181de9b-Abstract-Conference.html)

        **Abstract**:

        In this paper, we address the "dual problem" of multi-view scene reconstruction in which we utilize single-view images captured under different point lights to learn a neural scene representation. Different from existing single-view methods which can only recover a 2.5D scene representation (i.e., a normal / depth map for the visible surface), our method learns a neural reflectance field to represent the 3D geometry and BRDFs of a scene. Instead of relying on multi-view photo-consistency, our method exploits two information-rich monocular cues, namely shading and shadow, to infer scene geometry. Experiments on multiple challenging datasets show that our method is capable of recovering 3D geometry, including both visible and invisible parts, of a scene from single-view images. Thanks to the neural reflectance field representation, our method is robust to depth discontinuities. It supports applications like novel-view synthesis and relighting. Our code and model can be found at https://ywq.github.io/s3nerf.

        ----

        ## [115] AdaFocal: Calibration-aware Adaptive Focal Loss

        **Authors**: *Arindam Ghosh, Thomas Schaaf, Matthew R. Gormley*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a692a24dbc744fca340b9ba33bc6522-Abstract-Conference.html)

        **Abstract**:

        Much recent work has been devoted to the problem of ensuring that a neural network's confidence scores match the true probability of being correct, i.e. the calibration problem. Of note, it was found that training with focal loss leads to better calibration than cross-entropy while achieving similar level of accuracy \cite{mukhoti2020}. This success stems from focal loss regularizing the entropy of the model's prediction (controlled by the parameter $\gamma$), thereby reining in the model's overconfidence. Further improvement is expected if $\gamma$ is selected independently for each training sample (Sample-Dependent Focal Loss (FLSD-53) \cite{mukhoti2020}). However, FLSD-53 is based on heuristics and does not generalize well. In this paper, we propose a calibration-aware adaptive focal loss called AdaFocal that utilizes the calibration properties of focal (and inverse-focal) loss and adaptively modifies $\gamma_t$ for different groups of samples based on $\gamma_{t-1}$ from the previous step and the knowledge of model's under/over-confidence on the validation set. We evaluate AdaFocal on various image recognition and one NLP task, covering a wide variety of network architectures, to confirm the improvement in calibration while achieving similar levels of accuracy. Additionally, we show that models trained with AdaFocal achieve a significant boost in out-of-distribution detection.

        ----

        ## [116] PDEBench: An Extensive Benchmark for Scientific Machine Learning

        **Authors**: *Makoto Takamoto, Timothy Praditia, Raphael Leiteritz, Daniel MacKinlay, Francesco Alesiani, Dirk Pflüger, Mathias Niepert*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a9747136d411fb83f0cf81820d44afb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Machine learning-based modeling of physical systems has experienced increased interest in recent years. Despite some impressive progress, there is still a lack of benchmarks for Scientific ML that are easy to use but still challenging and repre- sentative of a wide range of problems. We introduce PDEBENCH, a benchmark suite of time-dependent simulation tasks based on Partial Differential Equations (PDEs). PDEBENCH comprises both code and data to benchmark the performance of novel machine learning models against both classical numerical simulations and machine learning baselines. Our proposed set of benchmark problems con- tribute the following unique features: (1) A much wider range of PDEs compared to existing benchmarks, ranging from relatively common examples to more real- istic and difficult problems; (2) much larger ready-to-use datasets compared to prior work, comprising multiple simulation runs across a larger number of ini- tial and boundary conditions and PDE parameters; (3) more extensible source codes with user-friendly APIs for data generation and baseline results with popular machine learning models (FNO, U-Net, PINN, Gradient-Based Inverse Method). PDEBENCH allows researchers to extend the benchmark freely for their own pur- poses using a standardized API and to compare the performance of new models to existing baseline methods. We also propose new evaluation metrics with the aim to provide a more holistic understanding of learning methods in the context of Scientific ML. With those metrics we identify tasks which are challenging for recent ML methods and propose these tasks as future challenges for the community. The code is available at https://github.com/pdebench/PDEBench.

        ----

        ## [117] Learning Robust Dynamics through Variational Sparse Gating

        **Authors**: *Arnav Kumar Jain, Shivakanth Sujit, Shruti Joshi, Vincent Michalski, Danijar Hafner, Samira Ebrahimi Kahou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0a97df4ce5b403ea87645010e9005130-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0a97df4ce5b403ea87645010e9005130-Abstract-Conference.html)

        **Abstract**:

        Learning world models from their sensory inputs enables agents to plan for actions by imagining their future outcomes. World models have previously been shown to improve sample-efficiency in simulated environments with few objects, but have not yet been applied successfully to environments with many objects. In environments with many objects, often only a small number of them are moving or interacting at the same time. In this paper, we investigate integrating this inductive bias of sparse interactions into the latent dynamics of world models trained from pixels. First, we introduce Variational Sparse Gating (VSG), a latent dynamics model that updates its feature dimensions sparsely through stochastic binary gates. Moreover, we propose a simplified architecture Simple Variational Sparse Gating (SVSG) that removes the deterministic pathway of previous models, resulting in a fully stochastic transition function that leverages the VSG mechanism. We evaluate the two model architectures in the BringBackShapes (BBS) environment that features a large number of moving objects and partial observability, demonstrating clear improvements over prior models.

        ----

        ## [118] Where to Pay Attention in Sparse Training for Feature Selection?

        **Authors**: *Ghada Sokar, Zahra Atashgahi, Mykola Pechenizkiy, Decebal Constantin Mocanu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0aa800df4298539770b57824afc77a89-Abstract-Conference.html)

        **Abstract**:

        A new line of research for feature selection based on neural networks has recently emerged. Despite its superiority to classical methods, it requires many training iterations to converge and detect the informative features. For datasets with a large number of samples or a very high dimensional feature space, the computational time becomes prohibitively long. In this paper, we present a new efficient unsupervised method for feature selection based on sparse autoencoders. In particular, we propose a new sparse training algorithm that optimizes a model's sparse topology during training to quickly pay attention to informative features. The attention-based adaptation of the sparse topology enables fast detection of informative features after a few training iterations. We performed extensive experiments on 10 datasets of different types, including image, speech, text, artificial, and biological. They cover a wide range of characteristics, such as low and high-dimensional feature spaces, as well as few and large training samples. Our proposed approach outperforms the state-of-the-art methods in terms of the selection of informative features while reducing training iterations and computational costs substantially. Moreover, the experiments show the robustness of our method in extremely noisy environments.

        ----

        ## [119] Maximizing Revenue under Market Shrinkage and Market Uncertainty

        **Authors**: *Maria-Florina Balcan, Siddharth Prasad, Tuomas Sandholm*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0aeb9a0f0a9715e853953ceb96531473-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0aeb9a0f0a9715e853953ceb96531473-Abstract-Conference.html)

        **Abstract**:

        A shrinking market is a ubiquitous challenge faced by various industries. In this paper we formulate the first formal model of shrinking markets in multi-item settings, and study how mechanism design and machine learning can help preserve revenue in an uncertain, shrinking market. Via a sample-based learning mechanism, we prove the first guarantees on how much revenue can be preserved by truthful multi-item, multi-bidder auctions (for limited supply) when only a random unknown fraction of the population participates in the market. We first present a general reduction that converts any sufficiently rich auction class into a randomized auction robust to market shrinkage. Our main technique is a novel combinatorial construction called a winner diagram that concisely represents all possible executions of an auction on an uncertain set of bidders. Via a probabilistic analysis of winner diagrams, we derive a general possibility result: a sufficiently rich class of auctions always contains an auction that is robust to market shrinkage and market uncertainty. Our result has applications to important practically-constrained settings such as auctions with a limited number of winners. We then show how to efficiently learn an auction that is robust to market shrinkage by leveraging practically-efficient routines for solving the winner determination problem.

        ----

        ## [120] General Cutting Planes for Bound-Propagation-Based Neural Network Verification

        **Authors**: *Huan Zhang, Shiqi Wang, Kaidi Xu, Linyi Li, Bo Li, Suman Jana, Cho-Jui Hsieh, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b06c8673ebb453e5e468f7743d8f54e-Abstract-Conference.html)

        **Abstract**:

        Bound propagation methods, when combined with branch and bound, are among the most effective methods to formally verify properties of deep neural networks such as correctness, robustness, and safety. However, existing works cannot handle the general form of cutting plane constraints widely accepted in traditional solvers, which are crucial for strengthening verifiers with tightened convex relaxations. In this paper, we generalize the bound propagation procedure to allow the addition of arbitrary cutting plane constraints, including those involving relaxed integer variables that do not appear in existing bound propagation formulations. Our generalized bound propagation method, GCP-CROWN, opens up the opportunity to apply general cutting plane methods for neural network verification while benefiting from the efficiency and GPU acceleration of bound propagation methods. As a case study, we investigate the use of cutting planes generated by off-the-shelf mixed integer programming (MIP) solver. We find that MIP solvers can generate high-quality cutting planes for strengthening bound-propagation-based verifiers using our new formulation. Since the branching-focused bound propagation procedure and the cutting-plane-focused MIP solver can run in parallel utilizing different types of hardware (GPUs and CPUs), their combination can quickly explore a large number of branches with strong cutting planes, leading to strong verification performance. Experiments demonstrate that our method is the first verifier that can completely solve the oval20 benchmark and verify twice as many instances on the oval21 benchmark compared to the best tool in VNN-COMP 2021, and also noticeably outperforms state-of-the-art verifiers on a wide range of benchmarks. GCP-CROWN is part of the $\alpha,\beta$-CROWN verifier, the VNN-COMP 2022 winner. Code is available at http://PaperCode.cc/GCP-CROWN.

        ----

        ## [121] StrokeRehab: A Benchmark Dataset for Sub-second Action Identification

        **Authors**: *Aakash Kaku, Kangning Liu, Avinash Parnandi, Haresh Rengaraj Rajamohan, Kannan Venkataramanan, Anita Venkatesan, Audre Wirtanen, Natasha Pandit, Heidi M. Schambra, Carlos Fernandez-Granda*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b11fce9fb449c4171dbec167bf63e12-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b11fce9fb449c4171dbec167bf63e12-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Automatic action identification from video and kinematic data is an important machine learning problem with applications ranging from robotics to smart health. Most existing works focus on identifying coarse actions such as running, climbing,  or cutting vegetables, which have relatively long durations and a complex series of motions. This is an important limitation for applications that require identification of more elemental motions at high temporal resolution. For example, in the rehabilitation of arm impairment after stroke, quantifying the training dose (number of repetitions) requires differentiating motions with sub-second durations. Our goal is to bridge this gap. To this end, we introduce a large-scale, multimodal dataset, StrokeRehab, as a new action-recognition benchmark that includes elemental short-duration actions labeled at a high temporal resolution. StrokeRehab consists of a high-quality inertial measurement unit sensor and video data of 51 stroke-impaired patients and 20 healthy subjects performing activities of daily living like feeding, brushing teeth, etc. Because it contains data from both healthy and impaired individuals, StrokeRehab can be used to study the influence of distribution shift in action-recognition tasks. When evaluated on StrokeRehab, current state-of-the-art models for action segmentation produce noisy predictions, which reduces their accuracy in identifying the corresponding sequence of actions. To address this, we propose a novel approach for high-resolution action identification, inspired by speech-recognition techniques, which is based on a sequence-to-sequence model that directly predicts the sequence of actions. This approach outperforms current state-of-the-art methods on StrokeRehab, as well as on the standard benchmark datasets 50Salads, Breakfast, and Jigsaws.

        ----

        ## [122] An $\alpha$-regret analysis of Adversarial Bilateral Trade

        **Authors**: *Yossi Azar, Amos Fiat, Federico Fusco*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b2832072ff6df19e586c74e27d90f12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b2832072ff6df19e586c74e27d90f12-Abstract-Conference.html)

        **Abstract**:

        We study sequential bilateral trade where sellers and buyers valuations are completely arbitrary ({\sl i.e.}, determined by an adversary). Sellers and buyers are strategic agents with private valuations for the good and the goal is to design a mechanism that maximizes efficiency (or gain from trade) while being incentive compatible, individually rational and budget balanced. In this paper we consider gain from trade which is harder to approximate than social welfare.We consider a variety of feedback scenarios and distinguish the cases where the mechanism posts one price and when it can post different prices for buyer and seller. We show several surprising results about the separation between the different scenarios. In particular we show that (a) it is impossible to achieve sublinear $\alpha$-regret for any $\alpha<2$, (b) but with full feedback sublinear $2$-regret is achievable (c) with a single price and partial feedback one cannot get sublinear $\alpha$ regret for any constant $\alpha$  (d) nevertheless, posting two prices even with one-bit feedback achieves sublinear $2$-regret, and (e) there is a provable separation in the $2$-regret bounds between full and partial feedback.

        ----

        ## [123] LDSA: Learning Dynamic Subtask Assignment in Cooperative Multi-Agent Reinforcement Learning

        **Authors**: *Mingyu Yang, Jian Zhao, Xunhan Hu, Wengang Zhou, Jiangcheng Zhu, Houqiang Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b4145b562cc22fb7fa50a2cd17c191d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b4145b562cc22fb7fa50a2cd17c191d-Abstract-Conference.html)

        **Abstract**:

        Cooperative multi-agent reinforcement learning (MARL) has made prominent progress in recent years. For training efficiency and scalability, most of the MARL algorithms make all agents share the same policy or value network. However, in many complex multi-agent tasks, different agents are expected to possess specific abilities to handle different subtasks. In those scenarios, sharing parameters indiscriminately may lead to similar behavior across all agents, which will limit the exploration efficiency and degrade the final performance. To balance the training complexity and the diversity of agent behavior, we propose a novel framework to learn dynamic subtask assignment (LDSA) in cooperative MARL. Specifically, we first introduce a subtask encoder to construct a vector representation for each subtask according to its identity. To reasonably assign agents to different subtasks, we propose an ability-based subtask selection strategy, which can dynamically group agents with similar abilities into the same subtask. In this way, agents dealing with the same subtask share their learning of specific abilities and different subtasks correspond to different specific abilities. We further introduce two regularizers to increase the representation difference between subtasks and stabilize the training by discouraging agents from frequently changing subtasks, respectively. Empirical results show that LDSA learns reasonable and effective subtask assignment for better collaboration and significantly improves the learning performance on the challenging StarCraft II micromanagement benchmark and Google Research Football.

        ----

        ## [124] Mildly Conservative Q-Learning for Offline Reinforcement Learning

        **Authors**: *Jiafei Lyu, Xiaoteng Ma, Xiu Li, Zongqing Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b5669c3b07bb8429af19a7919376ff5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b5669c3b07bb8429af19a7919376ff5-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (RL) defines the task of learning from a static logged dataset without continually interacting with the environment. The distribution shift between the learned policy and the behavior policy makes it necessary for the value function to stay conservative such that out-of-distribution (OOD) actions will not be severely overestimated. However, existing approaches, penalizing the unseen actions or regularizing with the behavior policy, are too pessimistic, which suppresses the generalization of the value function and hinders the performance improvement. This paper explores mild but enough conservatism for offline learning while not harming generalization. We propose Mildly Conservative Q-learning (MCQ), where OOD actions are actively trained by assigning them proper pseudo Q values. We theoretically show that MCQ induces a policy that behaves at least as well as the behavior policy and no erroneous overestimation will occur for OOD actions. Experimental results on the D4RL benchmarks demonstrate that MCQ achieves remarkable performance compared with prior work. Furthermore, MCQ shows superior generalization ability when transferring from offline to online, and significantly outperforms baselines. Our code is publicly available at https://github.com/dmksjfl/MCQ.

        ----

        ## [125] Iterative Feature Matching: Toward Provable Domain Generalization with Logarithmic Environments

        **Authors**: *Yining Chen, Elan Rosenfeld, Mark Sellke, Tengyu Ma, Andrej Risteski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html)

        **Abstract**:

        Domain generalization aims at performing well on unseen test environments with data from a limited number of training environments. Despite a proliferation of proposed algorithms for this task, assessing their performance both theoretically and empirically is still very challenging. Distributional matching algorithms such as (Conditional) Domain Adversarial Networks [Ganin et al., 2016, Long et al., 2018] are popular and enjoy empirical success, but they lack formal guarantees. Other approaches such as Invariant Risk Minimization (IRM) require a prohibitively large number of training environments---linear in the dimension of the spurious feature space $d_s$---even on simple data models like the one proposed by [Rosenfeld et al., 2021]. Under a variant of this model, we show that ERM and IRM can fail to find the optimal invariant predictor with $o(d_s)$ environments. We then present an iterative feature matching algorithm that is guaranteed with high probability to find the optimal invariant predictor after seeing only $O(\log d_s)$ environments. Our results provide the first theoretical justification for distribution-matching algorithms widely used in practice under a concrete nontrivial data model.

        ----

        ## [126] Certifying Robust Graph Classification under Orthogonal Gromov-Wasserstein Threats

        **Authors**: *Hongwei Jin, Zishun Yu, Xinhua Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b6b00f384aa33fec1f3d6bcf9550224-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b6b00f384aa33fec1f3d6bcf9550224-Abstract-Conference.html)

        **Abstract**:

        Graph classifiers are vulnerable to topological attacks. Although certificates of robustness have been recently developed, their threat model only counts local and global edge perturbations, which effectively ignores important graph structures such as isomorphism. To address this issue, we propose measuring the perturbation with the orthogonal Gromov-Wasserstein discrepancy, and building its Fenchel biconjugate to facilitate convex optimization. Our key insight is drawn from the matching loss whose root connects two variables via a monotone operator, and it yields a tight outer convex approximation for resistance distance on graph nodes. When applied to graph classification by graph convolutional networks, both our certificate and attack algorithm are demonstrated effective.

        ----

        ## [127] Functional Ensemble Distillation

        **Authors**: *Coby Penso, Idan Achituve, Ethan Fetaya*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b7f639ef28a9035a71f7e0c04c1d681-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b7f639ef28a9035a71f7e0c04c1d681-Abstract-Conference.html)

        **Abstract**:

        Bayesian models have many desirable properties, most notable is their ability to generalize from limited data and to properly estimate the uncertainty in their predictions. However, these benefits come at a steep computational cost as Bayesian inference, in most cases, is computationally intractable. One popular approach to alleviate this problem is using a Monte-Carlo estimation with an ensemble of models sampled from the posterior. However, this approach still comes at a significant computational cost, as one needs to store and run multiple models at test time. In this work, we investigate how to best distill an ensemble's predictions using an efficient model. First, we argue that current approaches are limited as they are constrained to classification and the Dirichlet distribution. Second, in many limited data settings, all ensemble members achieve nearly zero training loss, namely, they produce near-identical predictions on the training set which results in sub-optimal distilled models. To address both problems, we propose a novel and general distillation approach, named Functional Ensemble Distillation (FED), and we investigate how to best distill an ensemble in this setting. We find that learning the distilled model via a simple augmentation scheme in the form of mixup  augmentation significantly boosts the performance. We evaluated our method on several tasks and showed that it achieves superior results in both accuracy and uncertainty estimation compared to current approaches.

        ----

        ## [128] Use-Case-Grounded Simulations for Explanation Evaluation

        **Authors**: *Valerie Chen, Nari Johnson, Nicholay Topin, Gregory Plumb, Ameet Talwalkar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0b9536e186a77feff516893a5f393f7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0b9536e186a77feff516893a5f393f7a-Abstract-Conference.html)

        **Abstract**:

        A growing body of research runs human subject evaluations to study whether providing users with explanations of machine learning models can help them with practical real-world use cases. However, running user studies is challenging and costly, and consequently each study typically only evaluates a limited number of different settings, e.g., studies often only evaluate a few arbitrarily selected model explanation methods.  To address these challenges and aid user study design, we introduce Simulated Evaluations (SimEvals). SimEvals involve training algorithmic agents that take as input the information content (such as model explanations) that would be presented to the user, to predict answers to the use case of interest.  The algorithmic agent's test set accuracy provides a measure of the predictiveness of the information content for the downstream use case. We run a comprehensive evaluation on three real-world use cases (forward simulation, model debugging, and counterfactual reasoning) to demonstrate that SimEvals can effectively identify which explanation methods will help humans for each use case.  These results provide evidence that \simevals{} can be used to efficiently screen an important set of user study design decisions, e.g., selecting which explanations should be presented to the user, before running a potentially costly user study.

        ----

        ## [129] Lethal Dose Conjecture on Data Poisoning

        **Authors**: *Wenxiao Wang, Alexander Levine, Soheil Feizi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0badcb4e95306df76a719409155e46e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0badcb4e95306df76a719409155e46e8-Abstract-Conference.html)

        **Abstract**:

        Data poisoning considers an adversary that distorts the training set of machine learning algorithms for malicious purposes. In this work, we bring to light one conjecture regarding the fundamentals of data poisoning, which we call the Lethal Dose Conjecture. The conjecture states: If $n$ clean training samples are needed for accurate predictions, then in a size-$N$ training set, only $\Theta(N/n)$ poisoned samples can be tolerated while ensuring accuracy. Theoretically, we verify this conjecture in multiple cases. We also offer a more general perspective of this conjecture through distribution discrimination. Deep Partition Aggregation (DPA) and its extension, Finite Aggregation (FA) are recent approaches for provable defenses against data poisoning, where they predict through the majority vote of many base models trained from different subsets of training set using a given learner. The conjecture implies that both DPA and FA are (asymptotically) optimal---if we have the most data-efficient learner, they can turn it into one of the most robust defenses against data poisoning. This outlines a practical approach to developing stronger defenses against poisoning via finding data-efficient learners. Empirically, as a proof of concept, we show that by simply using different data augmentations for base learners, we can respectively double and triple the certified robustness of DPA on CIFAR-10 and GTSRB without sacrificing accuracy.

        ----

        ## [130] Online Decision Mediation

        **Authors**: *Daniel Jarrett, Alihan Hüyük, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0bc795afae289ed465a65a3b4b1f4eb7-Abstract-Conference.html)

        **Abstract**:

        Consider learning a decision support assistant to serve as an intermediary between (oracle) expert behavior and (imperfect) human behavior: At each time, the algorithm observes an action chosen by a fallible agent, and decides whether to accept that agent's decision, intervene with an alternative, or request the expert's opinion. For instance, in clinical diagnosis, fully-autonomous machine behavior is often beyond ethical affordances, thus real-world decision support is often limited to monitoring and forecasting. Instead, such an intermediary would strike a prudent balance between the former (purely prescriptive) and latter (purely descriptive) approaches, while providing an efficient interface between human mistakes and expert feedback. In this work, we first formalize the sequential problem of online decision mediation---that is, of simultaneously learning and evaluating mediator policies from scratch with abstentive feedback: In each round, deferring to the oracle obviates the risk of error, but incurs an upfront penalty, and reveals the otherwise hidden expert action as a new training data point. Second, we motivate and propose a solution that seeks to trade off (immediate) loss terms against (future) improvements in generalization error; in doing so, we identify why conventional bandit algorithms may fail. Finally, through experiments and sensitivities on a variety of datasets, we illustrate consistent gains over applicable benchmarks on performance measures with respect to the mediator policy, the learned model, and the decision-making system as a whole.

        ----

        ## [131] Neural-Symbolic Entangled Framework for Complex Query Answering

        **Authors**: *Zezhong Xu, Wen Zhang, Peng Ye, Hui Chen, Huajun Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0bcfb525c8f8f07ae10a93d0b2a40e00-Abstract-Conference.html)

        **Abstract**:

        Answering complex queries over knowledge graphs (KG) is an important yet challenging task because of the KG incompleteness issue and cascading errors during reasoning. Recent query embedding (QE) approaches embed the entities and relations in a KG and the first-order logic (FOL) queries into a low dimensional space, making the query can be answered by dense similarity searching. However, previous works mainly concentrate on the target answers, ignoring intermediate entities' usefulness, which is essential for relieving the cascading error problem in logical query answering. In addition, these methods are usually designed with their own geometric or distributional embeddings to handle logical operators like union, intersection, and negation, with the sacrifice of the accuracy of the basic operator -- projection, and they could not absorb other embedding methods to their models. In this work, we propose a Neural and Symbolic Entangled framework (ENeSy) for complex query answering, which enables the neural and symbolic reasoning to enhance each other to alleviate the cascading error and KG incompleteness. The projection operator in ENeSy could be any embedding method with the capability of link prediction, and the other FOL operators are handled without parameters. With both neural and symbolic reasoning results contained, ENeSy answers queries in ensembles. We evaluate ENeSy on complex query answering benchmarks, and ENeSy achieves the state-of-the-art, especially in the setting of training model only with the link prediction task.

        ----

        ## [132] Reinforcement Learning with Automated Auxiliary Loss Search

        **Authors**: *Tairan He, Yuge Zhang, Kan Ren, Minghuan Liu, Che Wang, Weinan Zhang, Yuqing Yang, Dongsheng Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0be44cc1d459731928501cae5699f57a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0be44cc1d459731928501cae5699f57a-Abstract-Conference.html)

        **Abstract**:

        A good state representation is crucial to solving complicated reinforcement learning (RL) challenges. Many recent works focus on designing auxiliary losses for learning informative representations. Unfortunately, these handcrafted objectives rely heavily on expert knowledge and may be sub-optimal. In this paper, we propose a principled and universal method for learning better representations with auxiliary loss functions, named Automated Auxiliary Loss Search (A2LS), which automatically searches for top-performing auxiliary loss functions for RL. Specifically, based on the collected trajectory data, we define a general auxiliary loss space of size $7.5 \times 10^{20}$ and explore the space with an efficient evolutionary search strategy. Empirical results show that the discovered auxiliary loss (namely, A2-winner) significantly improves the performance on both high-dimensional (image) and low-dimensional (vector) unseen tasks with much higher efficiency, showing promising generalization ability to different settings and even different benchmark domains. We conduct a statistical analysis to reveal the relations between patterns of auxiliary losses and RL performance.

        ----

        ## [133] FinRL-Meta: Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning

        **Authors**: *Xiao-Yang Liu, Ziyi Xia, Jingyang Rui, Jiechao Gao, Hongyang Yang, Ming Zhu, Christina Dan Wang, Zhaoran Wang, Jian Guo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0bf54b80686d2c4dc0808c2e98d430f7-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Finance is a particularly challenging playground for deep reinforcement learning. However, establishing high-quality market environments and benchmarks for financial reinforcement learning is challenging due to three major factors, namely, low signal-to-noise ratio of financial data, survivorship bias of historical data, and backtesting overfitting. In this paper, we present an openly accessible FinRL-Meta library that has been actively maintained by the AI4Finance community. First, following a DataOps paradigm, we will provide hundreds of market environments through an automatic data curation pipeline that processes dynamic datasets from real-world markets into gym-style market environments. Second, we reproduce popular papers as stepping stones for users to design new trading strategies. We also deploy the library on cloud platforms so that users can visualize their own results and assess the relative performance via community-wise competitions. Third, FinRL-Meta provides tens of Jupyter/Python demos organized into a curriculum and a documentation website to serve the rapidly growing community. FinRL-Meta is available at: \url{https://github.com/AI4Finance-Foundation/FinRL-Meta}

        ----

        ## [134] TempEL: Linking Dynamically Evolving and Newly Emerging Entities

        **Authors**: *Klim Zaporojets, Lucie-Aimée Kaffee, Johannes Deleu, Thomas Demeester, Chris Develder, Isabelle Augenstein*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0c3464f16c854d395b880cf9e7bcaf2f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0c3464f16c854d395b880cf9e7bcaf2f-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        In our continuously evolving world, entities change over time and new, previously non-existing or unknown, entities appear. We study how this evolutionary scenario impacts the performance on a well established entity linking (EL) task. For that study, we introduce TempEL, an entity linking dataset that consists of time-stratified English Wikipedia snapshots from 2013 to 2022, from which we collect both anchor mentions of entities, and these target entities’ descriptions. By capturing such temporal aspects, our newly introduced TempEL resource contrasts with currently existing entity linking datasets, which are composed of fixed mentions linked to a single static version of a target Knowledge Base (e.g., Wikipedia 2010 for CoNLL-AIDA). Indeed, for each of our collected temporal snapshots, TempEL contains links to entities that are continual, i.e., occur in all of the years, as well as completely new entities that appear for the first time at some point. Thus, we enable to quantify the performance of current state-of-the-art EL models for: (i) entities that are subject to changes over time in their Knowledge Base descriptions as well as their mentions’ contexts, and (ii) newly created entities that were previously non-existing (e.g., at the time the EL model was trained). Our experimental results show that in terms of temporal performance degradation, (i) continual entities suffer a decrease of up to 3.1% EL accuracy, while (ii) for new entities this accuracy drop is up to 17.9%. This highlights the challenge of the introduced TempEL dataset and opens new research prospects in the area of time-evolving entity disambiguation.

        ----

        ## [135] M$^4$I: Multi-modal Models Membership Inference

        **Authors**: *Pingyi Hu, Zihan Wang, Ruoxi Sun, Hu Wang, Minhui Xue*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0c79d6ed1788653643a1ac67b6ea32a7-Abstract-Conference.html)

        **Abstract**:

        With the development of machine learning techniques, the attention of research has been moved from single-modal learning to multi-modal learning, as real-world data exist in the form of different modalities. However, multi-modal models often carry more information than single-modal models and they are usually applied in sensitive scenarios, such as medical report generation or disease identification. Compared with the existing membership inference against machine learning classifiers, we focus on the problem that the input and output of the multi-modal models are in different modalities, such as image captioning. This work studies the privacy leakage of multi-modal models through the lens of membership inference attack, a process of determining whether a data record involves in the model training process or not. To achieve this, we propose Multi-modal Models Membership Inference (M$^4$I) with two attack methods to infer the membership status, named metric-based (MB) M$^4$I and feature-based (FB) M$^4$I, respectively. More specifically, MB M$^4$I adopts similarity metrics while attacking to infer target data membership. FB M$^4$I uses a pre-trained shadow multi-modal feature extractor to achieve the purpose of data inference attack by comparing the similarities from extracted input and output features. Extensive experimental results show that both attack methods can achieve strong performances. Respectively, 72.5% and 94.83% of attack success rates on average can be obtained under unrestricted scenarios. Moreover, we evaluate multiple defense mechanisms against our attacks. The source code of M$^4$I attacks is publicly available at https://github.com/MultimodalMI/Multimodal-membership-inference.git.

        ----

        ## [136] Best of Both Worlds Model Selection

        **Authors**: *Aldo Pacchiano, Christoph Dann, Claudio Gentile*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0c8d3770cbb759430f4f4679abe3ab80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0c8d3770cbb759430f4f4679abe3ab80-Abstract-Conference.html)

        **Abstract**:

        We study the problem of model selection in bandit scenarios in the presence of nested policy classes, with the goal of obtaining simultaneous adversarial and stochastic (``best of both worlds") high-probability regret guarantees. Our approach requires that each base learner comes with a candidate regret bound that may or may not hold, while our meta algorithm plays each base learner according to a schedule that keeps the base learner's candidate regret bounds balanced until they are detected to violate their guarantees. We develop careful mis-specification tests specifically designed to blend the above model selection criterion with the ability to leverage the (potentially benign) nature of the environment. We recover the model selection guarantees of the CORRAL algorithm for adversarial environments, but with the additional benefit of achieving high probability regret bounds. More importantly, our model selection results also hold simultaneously in stochastic environments under gap assumptions. These are the first theoretical results that achieve best-of-both world (stochastic and adversarial) guarantees while performing model selection in contextual bandit scenarios.

        ----

        ## [137] The Unreasonable Effectiveness of Fully-Connected Layers for Low-Data Regimes

        **Authors**: *Peter Kocsis, Peter Súkeník, Guillem Brasó, Matthias Nießner, Laura Leal-Taixé, Ismail Elezi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cc21b418ec126f005c7fe8157432339-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cc21b418ec126f005c7fe8157432339-Abstract-Conference.html)

        **Abstract**:

        Convolutional neural networks were the standard for solving many computer vision tasks until recently, when Transformers of MLP-based architectures have started to show competitive performance. These architectures typically have a vast number of weights and need to be trained on massive datasets; hence, they are not suitable for their use in low-data regimes. In this work, we propose a simple yet effective framework to improve generalization from small amounts of data. We augment modern CNNs with fully-connected (FC) layers and show the massive impact this architectural change has in low-data regimes. We further present an online joint knowledge-distillation method to utilize the extra FC layers at train time but avoid them during test time. This allows us to improve the generalization of a CNN-based model without any increase in the number of weights at test time. We perform classification experiments for a large range of network backbones and several standard datasets on supervised learning and active learning. Our experiments significantly outperform the networks without fully-connected layers, reaching a relative improvement of up to $16\%$ validation accuracy in the supervised setting without adding any extra parameters during inference.

        ----

        ## [138] Augmentations in Hypergraph Contrastive Learning: Fabricated and Generative

        **Authors**: *Tianxin Wei, Yuning You, Tianlong Chen, Yang Shen, Jingrui He, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cd1eec0eeaf5ce1bf6d8875a7c1d095-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cd1eec0eeaf5ce1bf6d8875a7c1d095-Abstract-Conference.html)

        **Abstract**:

        This paper targets at improving the generalizability of hypergraph neural networks in the low-label regime, through applying the contrastive learning approach from images/graphs (we refer to it as HyperGCL). We focus on the following question: How to construct contrastive views for hypergraphs via augmentations? We provide the solutions in two folds. First, guided by domain knowledge, we fabricate two schemes to augment hyperedges with higher-order relations encoded, and adopt three vertex augmentation strategies from graph-structured data. Second, in search of more effective views in a data-driven manner, we for the first time propose a hypergraph generative model to  generate augmented views, and then an end-to-end differentiable pipeline to jointly learn hypergraph augmentations and model parameters. Our technical innovations are reflected in designing both fabricated and generative augmentations of hypergraphs. The experimental findings include: (i) Among fabricated augmentations in HyperGCL, augmenting hyperedges provides the most numerical gains, implying that higher-order information in structures is usually more downstream-relevant; (ii) Generative augmentations do better in preserving higher-order information to further benefit generalizability; (iii) HyperGCL also boosts robustness and fairness in hypergraph representation learning. Codes are released at https://github.com/weitianxin/HyperGCL.

        ----

        ## [139] On the Global Convergence Rates of Decentralized Softmax Gradient Play in Markov Potential Games

        **Authors**: *Runyu Zhang, Jincheng Mei, Bo Dai, Dale Schuurmans, Na Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cd4c8c7ba098b199242c6634f43f653-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cd4c8c7ba098b199242c6634f43f653-Abstract-Conference.html)

        **Abstract**:

        Softmax policy gradient is a popular algorithm for policy optimization in single-agent reinforcement learning, particularly since projection is not needed for each gradient update. However, in multi-agent systems, the lack of central coordination introduces significant additional difficulties in the convergence analysis. Even for a stochastic game with identical interest, there can be multiple Nash Equilibria (NEs), which disables proof techniques that rely on the existence of a unique global optimum. Moreover, the softmax parameterization introduces non-NE policies with zero gradient, making it difficult for gradient-based algorithms in seeking NEs. In this paper, we study the finite time convergence of decentralized softmax gradient play in a special form of game, Markov Potential Games (MPGs), which includes the identical interest game as a special case. We investigate both gradient play and natural gradient play, with and without $\log$-barrier regularization. The established convergence rates for the unregularized cases contain a trajectory dependent constant that can be \emph{arbitrarily large}, whereas the $\log$-barrier regularization overcomes this drawback, with the cost of slightly worse dependence on other factors such as the action set size. An empirical study on an identical interest matrix game confirms the theoretical findings.

        ----

        ## [140] Sym-NCO: Leveraging Symmetricity for Neural Combinatorial Optimization

        **Authors**: *Minsu Kim, Junyoung Park, Jinkyoo Park*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cddb777d3441326544e21b67f41bdc8-Abstract-Conference.html)

        **Abstract**:

        Deep reinforcement learning (DRL)-based combinatorial optimization (CO) methods (i.e., DRL-NCO) have shown significant merit over the conventional CO solvers as DRL-NCO is capable of learning CO solvers less relying on problem-specific expert domain knowledge (heuristic method) and supervised labeled data (supervised learning method). This paper presents a novel training scheme, Sym-NCO, which is a regularizer-based training scheme that leverages universal symmetricities in various CO problems and solutions. Leveraging symmetricities such as rotational and reflectional invariance can greatly improve the generalization capability of DRL-NCO because it allows the learned solver to exploit the commonly shared symmetricities in the same CO problem class. Our experimental results verify that our Sym-NCO greatly improves the performance of DRL-NCO methods in four CO tasks, including the traveling salesman problem (TSP), capacitated vehicle routing problem (CVRP), prize collecting TSP (PCTSP), and orienteering problem (OP), without utilizing problem-specific expert domain knowledge. Remarkably, Sym-NCO outperformed not only the existing DRL-NCO methods but also a competitive conventional solver, the iterative local search (ILS), in PCTSP at 240$\times$ faster speed. Our source code is available at https://github.com/alstn12088/Sym-NCO.

        ----

        ## [141] Few-Shot Parameter-Efficient Fine-Tuning is Better and Cheaper than In-Context Learning

        **Authors**: *Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, Colin Raffel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cde695b83bd186c1fd456302888454c-Abstract-Conference.html)

        **Abstract**:

        Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)^3 that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters. We also propose a simple recipe based on the T0 model called T-Few that can be applied to new tasks without task-specific tuning or modifications. We validate the effectiveness of T-Few on completely unseen tasks by applying it to the RAFT benchmark, attaining super-human performance for the first time and outperforming the state-of-the-art by 6% absolute. All of the code used in our experiments will be publicly available.

        ----

        ## [142] HF-NeuS: Improved Surface Reconstruction Using High-Frequency Details

        **Authors**: *Yiqun Wang, Ivan Skorokhodov, Peter Wonka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0ce8e3434c7b486bbddff9745b2a1722-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0ce8e3434c7b486bbddff9745b2a1722-Abstract-Conference.html)

        **Abstract**:

        Neural rendering can be used to reconstruct implicit representations of shapes without 3D supervision. However, current neural surface reconstruction methods have difficulty learning high-frequency geometry details, so the reconstructed shapes are often over-smoothed. We develop HF-NeuS, a novel method to improve the quality of surface reconstruction in neural rendering. We follow recent work to model surfaces as signed distance functions (SDFs). First, we offer a derivation to analyze the relationship between the SDF, the volume density, the transparency function, and the weighting function used in the volume rendering equation and propose to model transparency as a transformed SDF. Second, we observe that attempting to jointly encode high-frequency and low-frequency components in a single SDF leads to unstable optimization. We propose to decompose the SDF into base and displacement functions with a coarse-to-fine strategy to increase the high-frequency details gradually. Finally, we design an adaptive optimization strategy that makes the training process focus on improving those regions near the surface where the SDFs have artifacts. Our qualitative and quantitative results show that our method can reconstruct fine-grained surface details and obtain better surface reconstruction quality than the current state of the art. Code available at https://github.com/yiqun-wang/HFS.

        ----

        ## [143] On the Epistemic Limits of Personalized Prediction

        **Authors**: *Lucas Monteiro Paes, Carol Xuan Long, Berk Ustun, Flávio P. Calmon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0cfc9404f89400c5ed897035e0d3748c-Abstract-Conference.html)

        **Abstract**:

        Machine learning models are often personalized by using group attributes that encode personal characteristics (e.g., sex, age group, HIV status). In such settings, individuals expect to receive more accurate predictions in return for disclosing group attributes to the personalized model. We study when we can tell that a personalized model upholds this principle for every group who provides personal data. We introduce a metric called the benefit of personalization (BoP) to measure the smallest gain in accuracy that any group expects to receive from a personalized model. We describe how the BoP can be used to carry out basic routines to audit a personalized model, including: (i) hypothesis tests to check that a personalized model improves performance for every group; (ii) estimation procedures to bound the minimum gain in personalization. We characterize the reliability of these routines in a finite-sample regime and present minimax bounds on both the probability of error for BoP hypothesis tests and the mean-squared error of BoP estimates. Our results show that we can only claim that personalization improves performance for each group who provides data when we explicitly limit the number of group attributes used by a personalized model. In particular, we show that it is impossible to reliably verify that a personalized classifier with $k \geq 19$ binary group attributes will benefit every group who provides personal data using a dataset of $n = 8\times10^9$ samples -- one for each person in the world.

        ----

        ## [144] DeepInteraction: 3D Object Detection via Modality Interaction

        **Authors**: *Zeyu Yang, Jiaqi Chen, Zhenwei Miao, Wei Li, Xiatian Zhu, Li Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0d18ab3b5fabfa6fe47c62e711af02f0-Abstract-Conference.html)

        **Abstract**:

        Existing top-performance 3D object detectors typically rely on the multi-modal fusion strategy. This design is however fundamentally restricted due to overlooking the modality-specific useful information and finally hampering the model performance. To address this limitation, in this work we introduce a novel modality interaction strategy where individual per-modality representations are learned and maintained throughout for enabling their unique characteristics to be exploited during object detection. To realize this proposed strategy, we design a DeepInteraction architecture characterized by a multi-modal representational interaction encoder and a multi-modal predictive interaction decoder. Experiments on the large-scale nuScenes dataset show that our proposed method surpasses all prior arts often by a large margin. Crucially, our method is ranked at the first position at the highly competitive nuScenes object detection leaderboard.

        ----

        ## [145] Deep Differentiable Logic Gate Networks

        **Authors**: *Felix Petersen, Christian Borgelt, Hilde Kuehne, Oliver Deussen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0d3496dd0cec77a999c98d35003203ca-Abstract-Conference.html)

        **Abstract**:

        Recently, research has increasingly focused on developing efficient neural network architectures. In this work, we explore logic gate networks for machine learning tasks by learning combinations of logic gates. These networks comprise logic gates such as "AND" and "XOR", which allow for very fast execution. The difficulty in learning logic gate networks is that they are conventionally non-differentiable and therefore do not allow training with gradient descent. Thus, to allow for effective training, we propose differentiable logic gate networks, an architecture that combines real-valued logics and a continuously parameterized relaxation of the network. The resulting discretized logic gate networks achieve fast inference speeds, e.g., beyond a million images of MNIST per second on a single CPU core.

        ----

        ## [146] Maximizing and Satisficing in Multi-armed Bandits with Graph Information

        **Authors**: *Parth Thaker, Mohit Malu, Nikhil Rao, Gautam Dasarathy*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0d561979f0f4bc6127cfcfe9c46ee205-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0d561979f0f4bc6127cfcfe9c46ee205-Abstract-Conference.html)

        **Abstract**:

        Pure exploration in multi-armed bandits has emerged as an important framework for modeling decision making and search under uncertainty. In modern applications however, one is often faced with a tremendously large number of options and even obtaining one observation per option may be too costly rendering traditional pure exploration algorithms ineffective. Fortunately, one often has access to similarity relationships amongst the options that can be leveraged. In this paper, we consider the pure exploration problem in stochastic multi-armed bandits where the similarities between the arms is captured by a graph and the rewards may be represented as a smooth signal on this graph. In particular, we consider the problem of finding the arm with the maximum reward (i.e., the maximizing problem) or one that has sufficiently high reward (i.e., the satisficing problem) under this model. We propose novel algorithms GRUB (GRaph based UcB) and zeta-GRUB for these problems and provide theoretical characterization of their performance which specifically elicits the benefit of the graph side information. We also prove a lower bound on the data requirement that shows a large class of problems where these algorithms are near-optimal. We complement our theory with experimental results that show the benefit of capitalizing on such side information.

        ----

        ## [147] MoGDE: Boosting Mobile Monocular 3D Object Detection with Ground Depth Estimation

        **Authors**: *Yunsong Zhou, Quan Liu, Hongzi Zhu, Yunzhe Li, Shan Chang, Minyi Guo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0d81e6f2511fc78631ee0315fafeef9e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0d81e6f2511fc78631ee0315fafeef9e-Abstract-Conference.html)

        **Abstract**:

        Monocular 3D object detection (Mono3D) in mobile settings (e.g., on a vehicle, a drone, or a robot) is an important yet challenging task. Due to the near-far disparity phenomenon of monocular vision and the ever-changing camera pose, it is hard to acquire high detection accuracy, especially for far objects. Inspired by the insight that the depth of an object can be well determined according to the depth of the ground where it stands, in this paper, we propose a novel Mono3D framework, called MoGDE, which constantly estimates the corresponding ground depth of an image and then utilizes the estimated ground depth information to guide Mono3D. To this end, we utilize a pose detection network to estimate the pose of the camera and then construct a feature map portraying pixel-level ground depth according to the 3D-to-2D perspective geometry. Moreover, to improve Mono3D with the estimated ground depth, we design an RGB-D feature fusion network based on the transformer structure, where the long-range self-attention mechanism is utilized to effectively identify ground-contacting points and pin the corresponding ground depth to the image feature map. We conduct extensive experiments on the real-world KITTI dataset. The results demonstrate that MoGDE can effectively improve the Mono3D accuracy and robustness for both near and far objects. MoGDE yields the best performance compared with the state-of-the-art methods by a large margin and is ranked number one on the KITTI 3D benchmark.

        ----

        ## [148] Causality Preserving Chaotic Transformation and Classification using Neurochaos Learning

        **Authors**: *Harikrishnan N. B., Aditi Kathpalia, Nithin Nagaraj*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0d9057d84a9fc37523bf826232ea6820-Abstract-Conference.html)

        **Abstract**:

        Discovering cause and effect variables from observational data is an important but challenging problem in science and engineering. In this work, a recently proposed brain inspired learning algorithm namely-\emph{Neurochaos Learning} (NL) is used for the classification of cause and effect time series generated using coupled autoregressive processes, coupled 1D chaotic skew tent maps, coupled 1D chaotic logistic maps and a real-world prey-predator system. In the case of coupled skew tent maps, the proposed method consistently outperforms a five layer Deep Neural Network (DNN) and Long Short Term Memory (LSTM) architecture for unidirectional coupling coefficient values ranging from $0.1$ to $0.7$. Further, we investigate the preservation of causality in the feature extracted space of NL using Granger Causality for coupled autoregressive processes and Compression-Complexity Causality for coupled chaotic systems and real-world prey-predator dataset. Unlike DNN, LSTM and 1D Convolutional Neural Network, it is found that NL preserves the inherent causal structures present in the input timeseries data. These findings are promising for the theory and applications of causal machine learning and open up the possibility to explore the potential of NL for more sophisticated causal learning tasks.

        ----

        ## [149] GOOD: A Graph Out-of-Distribution Benchmark

        **Authors**: *Shurui Gui, Xiner Li, Limei Wang, Shuiwang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/0dc91de822b71c66a7f54fa121d8cbb9-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Out-of-distribution (OOD) learning deals with scenarios in which training and test data follow different distributions. Although general OOD problems have been intensively studied in machine learning, graph OOD is only an emerging area of research. Currently, there lacks a systematic benchmark tailored to graph OOD method evaluation. In this work, we aim at developing an OOD benchmark, known as GOOD, for graphs specifically. We explicitly make distinctions between covariate and concept shifts and design data splits that accurately reflect different shifts. We consider both graph and node prediction tasks as there are key differences in designing shifts. Overall, GOOD contains 11 datasets with 17 domain selections. When combined with covariate, concept, and no shifts, we obtain 51 different splits. We provide performance results on 10 commonly used baseline methods with 10 random runs. This results in 510 dataset-model combinations in total. Our results show significant performance gaps between in-distribution and OOD settings. Our results also shed light on different performance trends between covariate and concept shifts by different methods. Our GOOD benchmark is a growing project and expects to expand in both quantity and variety of resources as the area develops. The GOOD benchmark can be accessed via https://github.com/divelab/GOOD/.

        ----

        ## [150] CascadeXML: Rethinking Transformers for End-to-end Multi-resolution Training in Extreme Multi-label Classification

        **Authors**: *Siddhant Kharbanda, Atmadeep Banerjee, Erik Schultheis, Rohit Babbar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0e0157ce5ea15831072be4744cbd5334-Abstract-Conference.html)

        **Abstract**:

        Extreme Multi-label Text Classification (XMC) involves learning a classifier that can assign an input with a subset of most relevant labels from millions of label choices. Recent approaches, such as XR-Transformer and LightXML, leverage a transformer instance to achieve state-of-the-art performance. However, in this process, these approaches need to make various trade-offs between performance and computational requirements. A major shortcoming, as compared to the Bi-LSTM based AttentionXML, is that they fail to keep separate feature representations for each resolution in a label tree. We thus propose CascadeXML, an end-to-end multi-resolution learning pipeline, which can harness the multi-layered architecture of a transformer model for attending to different label resolutions with separate feature representations. CascadeXML significantly outperforms all existing approaches with non-trivial gains obtained on benchmark datasets consisting of up to three million labels. Code for CascadeXML will be made publicly available at https://github.com/xmc-aalto/cascadexml.

        ----

        ## [151] VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?

        **Authors**: *Jiawei Jiang, Lukas Burkhalter, Fangcheng Fu, Bolin Ding, Bo Du, Anwar Hithnawi, Bo Li, Ce Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0e1a2388cd2f78069f4d048d935cb218-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0e1a2388cd2f78069f4d048d935cb218-Abstract-Conference.html)

        **Abstract**:

        Vertical Federated Learning (VFL), that trains federated models over vertically partitioned data, has emerged as an important learning paradigm. However, existing VFL methods are facing two challenges: (1) scalability when # participants grows to even modest scale and (2) diminishing return w.r.t. # participants: not all participants are equally important and many will not introduce quality improvement in a large consortium. Inspired by these two challenges, in this paper, we ask: How can we select l out of m participants, where l ≪ m, that are most important?We call this problem Vertically Federated Participant Selection, and model it with a principled mutual information-based view. Our first technical contribution is VF-MINE—a Vertically Federated Mutual INformation Estimator—that uses one of the most celebrated algorithms in database theory—Fagin’s algorithm as a building block. Our second contribution is to further optimize VF-MINE to enable VF-PS, a group testing-based participant selection framework. We empirically show that vertically federated participation selection can be orders of magnitude faster than training a full-fledged VFL model, while being able to identify the most important subset of participants that often lead to a VFL model of similar quality.

        ----

        ## [152] PopArt: Efficient Sparse Regression and Experimental Design for Optimal Sparse Linear Bandits

        **Authors**: *Kyoungseok Jang, Chicheng Zhang, Kwang-Sung Jun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0e5cce15e1bfc6b3d7b71f24cc5da821-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0e5cce15e1bfc6b3d7b71f24cc5da821-Abstract-Conference.html)

        **Abstract**:

        In sparse linear bandits, a learning agent sequentially selects an action from a fixed action set and receives reward feedback, and the reward function depends linearly on a few coordinates of the covariates of the actions. This has applications in many real-world sequential decision making problems. In this paper, we devise a simple, novel sparse linear estimation method called $\textrm{PopArt}$ that enjoys a tighter $\ell_1$ recovery guarantee compared to Lasso (Tibshirani, 1996). Our bound naturally motivates an experimental design criterion that is convex and thus computationally efficient to solve. Based on our novel estimator and design criterion, we derive sparse linear bandit algorithms that enjoy improved regret upper bounds upon the state of the art (Hao et al., 2020), especially in terms of the geometry of the given action set. Finally, we prove a matching lower bound for sparse linear bandits in the data-poor regime, which closes the gap between upper and lower bounds in prior work.

        ----

        ## [153] Augmenting Online Algorithms with $\varepsilon$-Accurate Predictions

        **Authors**: *Anupam Gupta, Debmalya Panigrahi, Bernardo Subercaseaux, Kevin Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0ea048312aa812b2711fe765a9e9ef05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0ea048312aa812b2711fe765a9e9ef05-Abstract-Conference.html)

        **Abstract**:

        The growing body of work in learning-augmented online algorithms studies how online algorithms can be improved when given access to ML predictions about the future. Motivated by ML models that give a confidence parameter for their predictions, we study online algorithms with predictions that are $\epsilon$-accurate: namely, each prediction is correct with probability (at least) $\epsilon$, but can be arbitrarily inaccurate with the remaining probability. We show that even with predictions that are accurate with a small probability and arbitrarily inaccurate otherwise, we can dramatically outperform worst-case bounds for a range of classical online problems including caching, online set cover, and online facility location. Our main results are an $O(\log(1/\varepsilon))$-competitive algorithm for caching, and a simple $O(1/\varepsilon)$-competitive algorithm for a large family of covering problems, including set cover and facility location, with $\epsilon$-accurate predictions.

        ----

        ## [154] Unsupervised Multi-Object Segmentation by Predicting Probable Motion Patterns

        **Authors**: *Laurynas Karazija, Subhabrata Choudhury, Iro Laina, Christian Rupprecht, Andrea Vedaldi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0eaf2c04280c7fecc8b26762dd4ab6da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0eaf2c04280c7fecc8b26762dd4ab6da-Abstract-Conference.html)

        **Abstract**:

        We propose a new approach to learn to segment multiple image objects without manual supervision. The method can extract objects form still images, but uses videos for supervision. While prior works have considered motion for segmentation, a key insight is that, while motion can be used to identify objects, not all objects are necessarily in motion: the absence of motion does not imply the absence of objects. Hence, our model learns to predict image regions that are likely to contain motion patterns characteristic of objects moving rigidly. It does not predict specific motion, which cannot be done unambiguously from a still image, but a distribution of possible motions, which includes the possibility that an object does not move at all. We demonstrate the advantage of this approach over its deterministic counterpart and show state-of-the-art unsupervised object segmentation performance on simulated and real-world benchmarks, surpassing methods that use motion even at test time. As our approach is applicable to variety of network architectures that segment the scenes, we also apply it to existing image reconstruction-based models showing drastic improvement. Project page and code: https://www.robots.ox.ac.uk/~vgg/research/ppmp.

        ----

        ## [155] Incrementality Bidding via Reinforcement Learning under Mixed and Delayed Rewards

        **Authors**: *Ashwinkumar Badanidiyuru Varadaraja, Zhe Feng, Tianxi Li, Haifeng Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0ee633a6ade45eab4276352b3ee79c7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0ee633a6ade45eab4276352b3ee79c7a-Abstract-Conference.html)

        **Abstract**:

        Incrementality, which  measures the causal effect of showing an ad to a potential customer (e.g. a user in an internet platform) versus not, is a central object for advertisers in online advertising platforms. This paper  investigates the problem of how an advertiser can learn to optimize the bidding sequence in an online manner \emph{without} knowing the incrementality parameters in advance. We formulate the offline version of this problem as a specially structured episodic Markov Decision Process (MDP) and then, for its online learning counterpart,  propose a novel reinforcement learning (RL) algorithm with regret at most $\widetilde{O}(H^2\sqrt{T})$, which depends on the number of rounds $H$ and number of episodes $T$, but does not depend on the number of actions (i.e., possible bids). A fundamental difference between our learning problem from standard RL problems is that the realized reward feedback from conversion incrementality is \emph{mixed} and \emph{delayed}. To handle this difficulty we propose and analyze a novel pairwise moment-matching algorithm to learn the conversion incrementality, which we believe is of independent  interest.

        ----

        ## [156] Masked Generative Adversarial Networks are Data-Efficient Generation Learners

        **Authors**: *Jiaxing Huang, Kaiwen Cui, Dayan Guan, Aoran Xiao, Fangneng Zhan, Shijian Lu, Shengcai Liao, Eric P. Xing*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0efcb1885b8534109f95ca82a5319d25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0efcb1885b8534109f95ca82a5319d25-Abstract-Conference.html)

        **Abstract**:

        This paper shows that masked generative adversarial network (MaskedGAN) is robust image generation learners with limited training data. The idea of MaskedGAN is simple: it randomly masks out certain image information for effective GAN training with limited data. We develop two masking strategies that work along orthogonal dimensions of training images, including a shifted spatial masking that masks the images in spatial dimensions with random shifts, and a balanced spectral masking that masks certain image spectral bands with self-adaptive probabilities. The two masking strategies complement each other which together encourage more challenging holistic learning from limited training data, ultimately suppressing trivial solutions and failures in GAN training. Albeit simple, extensive experiments show that MaskedGAN achieves superior performance consistently across different network architectures (e.g., CNNs including BigGAN and StyleGAN-v2 and Transformers including TransGAN and GANformer) and datasets (e.g., CIFAR-10, CIFAR-100, ImageNet, 100-shot, AFHQ, FFHQ and Cityscapes).

        ----

        ## [157] What You See is What You Get: Principled Deep Learning via Distributional Generalization

        **Authors**: *Bogdan Kulynych, Yao-Yuan Yang, Yaodong Yu, Jaroslaw Blasiok, Preetum Nakkiran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f4bbaaaf1e167f79134dd4cf11e3ed4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f4bbaaaf1e167f79134dd4cf11e3ed4-Abstract-Conference.html)

        **Abstract**:

        Having similar behavior at training time and test time—what we call a “What You See Is What You Get” (WYSIWYG) property—is desirable in machine learning. Models trained with standard stochastic gradient descent (SGD), however, do not necessarily have this property, as their complex behaviors such as robustness or subgroup performance can differ drastically between training and test time. In contrast, we show that Differentially-Private (DP) training provably ensures the high-level WYSIWYG property, which we quantify using a notion of distributional generalization. Applying this connection, we introduce new conceptual tools for designing deep-learning methods by reducing generalization concerns to optimization ones: to mitigate unwanted behavior at test time, it is provably sufficient to mitigate this behavior on the training data. By applying this novel design principle, which bypasses “pathologies” of SGD, we construct simple algorithms that are competitive with SOTA in several distributional-robustness applications, significantly improve the privacy vs. disparate impact trade-off of DP-SGD, and mitigate robust overfitting in adversarial training. Finally, we also improve on theoretical bounds relating DP, stability, and distributional generalization.

        ----

        ## [158] Towards Understanding the Condensation of Neural Networks at Initial Training

        **Authors**: *Hanxu Zhou, Qixuan Zhou, Tao Luo, Yaoyu Zhang, Zhi-Qin John Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f4d1fc085b7504c140e66bb26ed8842-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f4d1fc085b7504c140e66bb26ed8842-Abstract-Conference.html)

        **Abstract**:

        Empirical works show that for ReLU neural networks (NNs) with small initialization, input weights of hidden neurons (the input weight of a hidden neuron consists of the weight from its input layer to the hidden neuron and its bias term) condense onto isolated orientations. The condensation dynamics implies that the training implicitly regularizes a NN towards one with much smaller effective size. In this work, we illustrate the formation of the condensation in multi-layer fully connected NNs and show that the maximal number of condensed orientations in the initial training stage is twice the multiplicity of the activation function, where ``multiplicity'' indicates the multiple roots of activation function at origin. Our theoretical analysis confirms experiments for two cases, one is for the activation function of multiplicity one with arbitrary dimension input, which contains many common activation functions, and the other is for the layer with one-dimensional input and arbitrary multiplicity. This work makes a step towards understanding how small initialization leads NNs to condensation at the initial training stage.

        ----

        ## [159] CoNT: Contrastive Neural Text Generation

        **Authors**: *Chenxin An, Jiangtao Feng, Kai Lv, Lingpeng Kong, Xipeng Qiu, Xuanjing Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f5fcf4bff73a3537e0813a38f0d3f76-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f5fcf4bff73a3537e0813a38f0d3f76-Abstract-Conference.html)

        **Abstract**:

        Recently, contrastive learning attracts increasing interests in neural text generation as a new solution to alleviate the exposure bias problem.  It introduces a sequence-level training signal which is crucial to generation tasks that always rely on auto-regressive decoding. However, previous methods using contrastive learning in neural text generation usually lead to inferior performance. In this paper, we analyse the underlying reasons and propose a new Contrastive Neural Text generation framework, CoNT.  CoNT addresses bottlenecks that prevent contrastive learning from being widely adopted in generation tasks from three aspects -- the construction of contrastive examples, the choice of the contrastive loss, and the strategy in decoding. We validate CoNT on five generation tasks with ten benchmarks, including machine translation, summarization, code comment generation, data-to-text generation and commonsense generation.  Experimental results show that CoNT clearly outperforms its baseline on all the ten benchmarks with a convincing margin.  Especially, CoNT surpasses previous the most competitive contrastive learning method for text generation, by 1.50 BLEU on machine translation and 1.77 ROUGE-1 on summarization, respectively. It achieves new state-of-the-art on summarization, code comment generation (without external data) and data-to-text generation.

        ----

        ## [160] GAPX: Generalized Autoregressive Paraphrase-Identification X

        **Authors**: *Yifei Zhou, Renyu Li, Hayden Housen, Ser Nam Lim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f6cc80ad86e553d085842308e0fd2cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f6cc80ad86e553d085842308e0fd2cb-Abstract-Conference.html)

        **Abstract**:

        Paraphrase Identification is a fundamental task in Natural Language Processing. While much progress has been made in the field, the performance of many state-of- the-art models often suffer from distribution shift during inference time. We verify that a major source of this performance drop comes from biases introduced by negative examples. To overcome these biases, we propose in this paper to train two separate models, one that only utilizes the positive pairs and the other the negative pairs. This enables us the option of deciding how much to utilize the negative model, for which we introduce a perplexity based out-of-distribution metric that we show can effectively and automatically determine how much weight it should be given during inference. We support our findings with strong empirical results.

        ----

        ## [161] Scalable Infomin Learning

        **Authors**: *Yanzhi Chen, Weihao Sun, Yingzhen Li, Adrian Weller*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f7e4bb7a35dd4cb426203c91a4bfa10-Abstract-Conference.html)

        **Abstract**:

        The task of infomin learning aims to learn a representation with high utility while being uninformative about a specified target, with the latter achieved by minimising the mutual information between the representation and the target. It has broad applications, ranging from training fair prediction models against protected attributes, to unsupervised learning with disentangled representations. Recent works on infomin learning mainly use adversarial training, which involves training a neural network to estimate mutual information or its proxy and thus is slow and difficult to optimise. Drawing on recent advances in slicing techniques, we propose a new infomin learning approach, which uses a novel proxy metric to mutual information. We further derive an accurate and analytically computable approximation to this proxy metric, thereby removing the need of constructing neural network-based mutual information estimators. Compared to baselines, experiments on algorithmic fairness, disentangled representation learning and domain adaptation verify that our method can more effectively remove unwanted information with limited time budget.

        ----

        ## [162] Learning to Accelerate Partial Differential Equations via Latent Global Evolution

        **Authors**: *Tailin Wu, Takashi Maruyama, Jure Leskovec*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f817dcbad81afb21fb695f1b2e55e44-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f817dcbad81afb21fb695f1b2e55e44-Abstract-Conference.html)

        **Abstract**:

        Simulating the time evolution of Partial Differential Equations (PDEs) of large-scale systems is crucial in many scientific and engineering domains such as fluid dynamics, weather forecasting and their inverse optimization problems. However, both classical solvers and recent deep learning-based surrogate models are typically extremely computationally intensive, because of their local evolution: they need to update the state of each discretized cell at each time step during inference. Here we develop Latent Evolution of PDEs (LE-PDE), a simple, fast and scalable method to accelerate the simulation and inverse optimization of PDEs. LE-PDE learns a compact, global representation of the system and efficiently evolves it fully in the latent space with learned latent evolution models. LE-PDE achieves speedup by having a much smaller latent dimension to update during long rollout as compared to updating in the input space. We introduce new learning objectives to effectively learn such latent dynamics to ensure long-term stability. We further introduce techniques for speeding-up inverse optimization of boundary conditions for PDEs via backpropagation through time in latent space, and an annealing technique to address the non-differentiability and sparse interaction of boundary conditions. We test our method in a 1D benchmark of nonlinear PDEs, 2D  Navier-Stokes flows into turbulent phase and an inverse optimization of boundary conditions in 2D Navier-Stokes flow. Compared to state-of-the-art deep learning-based surrogate models and other strong baselines, we demonstrate up to 128x reduction in the dimensions to update, and up to 15x improvement in speed, while achieving competitive accuracy.

        ----

        ## [163] Monte Carlo Augmented Actor-Critic for Sparse Reward Deep Reinforcement Learning from Suboptimal Demonstrations

        **Authors**: *Albert Wilcox, Ashwin Balakrishna, Jules Dedieu, Wyame Benslimane, Daniel S. Brown, Ken Goldberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f94c552e5fe82bc152494985e34bd48-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f94c552e5fe82bc152494985e34bd48-Abstract-Conference.html)

        **Abstract**:

        Providing densely shaped reward functions for RL algorithms is often exceedingly challenging, motivating the development of RL algorithms that can learn from easier-to-specify sparse reward functions. This sparsity poses new exploration challenges. One common way to address this problem is using demonstrations to provide initial signal about regions of the state space with high rewards. However, prior RL from demonstrations algorithms introduce significant complexity and many hyperparameters, making them hard to implement and tune. We introduce Monte Carlo Actor-Critic (MCAC), a parameter free modification to standard actor-critic algorithms which initializes the replay buffer with demonstrations and computes a modified $Q$-value by taking the maximum of the standard temporal distance (TD) target and a Monte Carlo estimate of the reward-to-go. This encourages exploration in the neighborhood of high-performing trajectories by encouraging high $Q$-values in corresponding regions of the state space. Experiments across $5$ continuous control domains suggest that MCAC can be used to significantly increase learning efficiency across $6$ commonly used RL and RL-from-demonstrations algorithms. See https://sites.google.com/view/mcac-rl for code and supplementary material.

        ----

        ## [164] Not too little, not too much: a theoretical analysis of graph (over)smoothing

        **Authors**: *Nicolas Keriven*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f956ca6f667c62e0f71511773c86a59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f956ca6f667c62e0f71511773c86a59-Abstract-Conference.html)

        **Abstract**:

        We analyze graph smoothing with mean aggregation, where each node successively receives the average of the features of its neighbors. Indeed, it has quickly been observed that Graph Neural Networks (GNNs), which generally follow some variant of Message-Passing (MP) with repeated aggregation, may be subject to the oversmoothing phenomenon: by performing too many rounds of MP, the node features tend to converge to a non-informative limit. In the case of mean aggregation, for connected graphs, the node features become constant across the whole graph. At the other end of the spectrum, it is intuitively obvious that some MP rounds are necessary, but existing analyses do not exhibit both phenomena at once: beneficial ``finite'' smoothing and oversmoothing in the limit. In this paper, we consider simplified linear GNNs, and rigorously analyze two examples for which a finite number of mean aggregation steps provably improves the learning performance, before oversmoothing kicks in. We consider a latent space random graph model, where node features are partial observations of the latent variables and the graph contains pairwise relationships between them. We show that graph smoothing restores some of the lost information, up to a certain point, by two phenomena: graph smoothing shrinks non-principal directions in the data faster than principal ones, which is useful for regression, and shrinks nodes within communities faster than they collapse together, which improves classification.

        ----

        ## [165] Kernel similarity matching with Hebbian networks

        **Authors**: *Kyle Luther, H. Sebastian Seung*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0f98645119923217a245735c2c4d23f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0f98645119923217a245735c2c4d23f4-Abstract-Conference.html)

        **Abstract**:

        Recent works have derived neural networks with online correlation-based learning rules to perform \textit{kernel similarity matching}. These works applied existing linear similarity matching algorithms to nonlinear features generated with random Fourier methods. In this paper attempt to perform kernel similarity matching by directly learning the nonlinear features. Our algorithm proceeds by deriving and then minimizing an upper bound for the sum of squared errors between output and input kernel similarities. The construction of our upper bound leads to online correlation-based learning rules which can be implemented with a 1 layer recurrent neural network. In addition to generating high-dimensional linearly separable representations, we show that our upper bound naturally yields representations which are sparse and selective for specific input patterns. We compare the approximation quality of our method to neural random Fourier method and variants of the popular but non-biological ``Nystr{\"o}m'' method for approximating the kernel matrix. Our method appears to be comparable or better than randomly sampled Nystr{\"o}m methods when the outputs are relatively low dimensional (although still potentially higher dimensional than the inputs) but less faithful when the outputs are very high dimensional.

        ----

        ## [166] HumanLiker: A Human-like Object Detector to Model the Manual Labeling Process

        **Authors**: *Haoran Wei, Ping Guo, Yangguang Zhu, Chenglong Liu, Peng Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0fb98d483fa580e0354bcdd3a003a3f3-Abstract-Conference.html)

        **Abstract**:

        Popular object detection models generate bounding boxes in a different way than we humans. As an example, modern detectors yield object box either upon the regression of its center and width/height (center-guided detector), or by grouping paired estimated corners (corner-guided detector). However, that is not the pattern we manually label an object due to high degrees of freedom in searching centers or low efficiency of grouping corners. Empirically, humans run two steps to locate an object bounding box manually: 1) click the mouse at the top-left corner of object, and then drag the mouse to the bottom-right corner; 2) refine the corner positions to make the bounding box more precisely, if necessary.  Inspired by this manual labeling process, we propose a novel human-like detector, termed as HumanLiker, which is devised as a two-stage end-to-end detector to simulate the two aforementioned. Like we humans in manual labeling, HumanLiker can effectively avert both the thorny center searching and heuristic corner grouping. Different from the mainstream detector branches, i.e., the center/corner-guided methods, the HumanLiker provides a new paradigm which integrates the advantages of both branches to balance the detection efficiency and bounding box quality. On MS-COCO test-dev set, HumanLiker can achieve 50.2%/51.6% and 53.8%/55.6% in term of AP with ResNeXt-101 and SwinTransformer backbones in single/multi-scale testing, outperforming current popular center/corner-guided baselines (e.g., DETR/CornerNet) by a large margin, with much less training epochs and higher inference FPS.  Code will be available soon.

        ----

        ## [167] Scalable Representation Learning in Linear Contextual Bandits with Constant Regret Guarantees

        **Authors**: *Andrea Tirinzoni, Matteo Papini, Ahmed Touati, Alessandro Lazaric, Matteo Pirotta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0fd489e5e393f61b355be86ed4c24a54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0fd489e5e393f61b355be86ed4c24a54-Abstract-Conference.html)

        **Abstract**:

        We study the problem of representation learning in stochastic contextual linear bandits. While the primary concern in this domain is usually to find \textit{realizable} representations (i.e., those that allow predicting the reward function at any context-action pair exactly), it has been recently shown that representations with certain spectral properties (called \textit{HLS}) may be more effective for the exploration-exploitation task, enabling \textit{LinUCB} to achieve constant (i.e., horizon-independent) regret. In this paper, we propose \textsc{BanditSRL}, a representation learning algorithm that combines a novel constrained optimization problem to learn a realizable representation with good spectral properties with a generalized likelihood ratio test to exploit the recovered representation and avoid excessive exploration. We prove that \textsc{BanditSRL} can be paired with any no-regret algorithm and achieve constant regret whenever an \textit{HLS} representation is available. Furthermore, \textsc{BanditSRL} can be easily combined with deep neural networks and we show how regularizing towards \textit{HLS} representations is beneficial in standard benchmarks.

        ----

        ## [168] DevFly: Bio-Inspired Development of Binary Connections for Locality Preserving Sparse Codes

        **Authors**: *Tianqi Wei, Rana Alkhoury Maroun, Qinghai Guo, Barbara Webb*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0fed4ca757f63257370f456def09d3eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0fed4ca757f63257370f456def09d3eb-Abstract-Conference.html)

        **Abstract**:

        Neural circuits undergo developmental processes which can be influenced by experience. Here we explore a bio-inspired development process to form the connections in a network used for locality sensitive hashing. The network is a simplified model of the insect mushroom body, which has sparse connections from the input layer to a second layer of higher dimension, forming a sparse code. In previous versions of this model, connectivity between the layers is random. We investigate whether the performance of the hash, evaluated in nearest neighbour query tasks, can be improved by process of developing the connections, in which the strongest input dimensions in successive samples are wired to each successive coding dimension. Experiments show that the accuracy of searching for nearest neighbours is improved, although performance is dependent on the parameter values and datasets used. Our approach is also much faster than alternative methods that have been proposed for training the connections in this model. Importantly, the development process does not impact connections built at an earlier stage, which should provide stable coding results for simultaneous learning in a downstream network.

        ----

        ## [169] Why neural networks find simple solutions: The many regularizers of geometric complexity

        **Authors**: *Benoit Dherin, Michael Munn, Mihaela Rosca, David Barrett*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0ff3502bb29570b219967278db150a50-Abstract-Conference.html)

        **Abstract**:

        In many contexts, simpler models are preferable to more complex models and the control of this model complexity is the goal for many methods in machine learning such as regularization, hyperparameter tuning and architecture design. In deep learning, it has been difficult to understand the underlying mechanisms of complexity control, since many traditional measures are not naturally suitable for deep neural networks. Here we develop the notion of geometric complexity, which is a measure of the variability of the model function, computed using a discrete Dirichlet energy. Using a combination of theoretical arguments and empirical results, we show that many common training heuristics such as parameter norm regularization, spectral norm regularization, flatness regularization, implicit gradient regularization, noise regularization and the choice of parameter initialization all act to control geometric complexity, providing a unifying framework in which to characterize the behavior of deep learning models.

        ----

        ## [170] Efficiently Computing Local Lipschitz Constants of Neural Networks via Bound Propagation

        **Authors**: *Zhouxing Shi, Yihan Wang, Huan Zhang, J. Zico Kolter, Cho-Jui Hsieh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/0ff54b4ec4f70b3ae12c8621ca8a49f4-Abstract-Conference.html)

        **Abstract**:

        Lipschitz constants are connected to many properties  of neural networks, such as robustness, fairness, and generalization. Existing methods for computing Lipschitz constants either produce relatively loose upper bounds or are limited to small networks. In this paper, we develop an efficient framework for computing the $\ell_\infty$ local Lipschitz constant of a neural network by tightly upper bounding the norm of Clarke Jacobian via linear bound propagation. We formulate the computation of local Lipschitz constants with a linear bound propagation process on a high-order backward graph induced by the chain rule of Clarke Jacobian. To enable linear bound propagation, we derive tight linear relaxations for specific nonlinearities in Clarke Jacobian. This formulate unifies existing ad-hoc approaches such as RecurJac, which can be seen as a special case of ours with weaker relaxations. The bound propagation framework also allows us to easily borrow the popular Branch-and-Bound (BaB) approach from neural network verification to further tighten Lipschitz constants. Experiments show that on tiny models, our method produces comparable bounds compared to exact methods that cannot scale to slightly larger models; on larger models, our method efficiently produces tighter results than existing relaxed or naive methods, and our method scales to much larger practical models that previous works could not handle. We also demonstrate an application on provable monotonicity analysis. Code is available at https://github.com/shizhouxing/Local-Lipschitz-Constants.

        ----

        ## [171] A Causal Analysis of Harm

        **Authors**: *Sander Beckers, Hana Chockler, Joseph Y. Halpern*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/100c1f131893d3b4b34bb8db49bef79f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/100c1f131893d3b4b34bb8db49bef79f-Abstract-Conference.html)

        **Abstract**:

        As autonomous systems rapidly become ubiquitous, there is a growing need for a legal and regulatory framework toaddress when and how such a system harms someone. There have been several attempts within the philosophy literature to define harm, but none of them has proven capable of dealing with with the many examples that have been presented, leading some to suggest that the notion of harm should be abandoned and ``replaced by more well-behaved notions''. As harm is generally something that is caused, most of these definitions have involved causality at some level. Yet surprisingly, none of them makes use of causal models and the definitions of actual causality that they can express. In this paper we formally define a qualitative notion of harm that uses causal models and is based on a well-known definition of actual causality (Halpern, 2016). The key novelty of our definition is that it is based on contrastive causation and uses a default utility to which the utility of actual outcomes is compared. We show that our definition is able to handle the examples from the literature, and illustrate its importance for reasoning about situations involving autonomous systems.

        ----

        ## [172] Seeing the forest and the tree: Building representations of both individual and collective dynamics with transformers

        **Authors**: *Ran Liu, Mehdi Azabou, Max Dabagia, Jingyun Xiao, Eva L. Dyer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/1022661f3f43406065641f16ce25eafa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/1022661f3f43406065641f16ce25eafa-Abstract-Conference.html)

        **Abstract**:

        Complex time-varying systems are often studied by abstracting away from the dynamics of individual components to build a model of the population-level dynamics from the start. However, when building a population-level description, it can be easy to lose sight of each individual and how they contribute to the larger picture. In this paper, we present a novel transformer architecture for learning from time-varying data that builds descriptions of both the individual as well as the collective population dynamics. Rather than combining all of our data into our model at the onset, we develop a separable architecture that operates on individual time-series first before passing them forward; this induces a permutation-invariance property and can be used to transfer across systems of different size and order. After demonstrating that our model can be applied to successfully recover complex interactions and dynamics in many-body systems, we apply our approach to populations of neurons in the nervous system. On neural activity datasets, we show that our model not only yields robust decoding performance, but also provides impressive performance in transfer across recordings of different animals without any neuron-level correspondence. By enabling flexible pre-training that can be transferred to neural recordings of different size and order, our work provides a first step towards creating a foundation model for neural decoding.

        ----

        ## [173] Semi-Supervised Learning with Decision Trees: Graph Laplacian Tree Alternating Optimization

        **Authors**: *Arman Zharmagambetov, Miguel Á. Carreira-Perpiñán*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/104f7b25495a0e40e65fb7c7eee37ed9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/104f7b25495a0e40e65fb7c7eee37ed9-Abstract-Conference.html)

        **Abstract**:

        Semi-supervised learning seeks to learn a machine learning model when only a small amount of the available data is labeled. The most widespread approach uses a graph prior, which encourages similar instances to have similar predictions. This has been very successful with models ranging from kernel machines to neural networks, but has remained inapplicable to decision trees, for which the optimization problem is much harder. We solve this based on a reformulation of the problem which requires iteratively solving two simpler problems: a supervised tree learning problem, which can be solved by the Tree Alternating Optimization algorithm; and a label smoothing problem, which can be solved through a sparse linear system. The algorithm is scalable and highly effective even with very few labeled instances, and makes it possible to learn accurate, interpretable models based on decision trees in such situations.

        ----

        ## [174] Riemannian Score-Based Generative Modelling

        **Authors**: *Valentin De Bortoli, Emile Mathieu, Michael J. Hutchinson, James Thornton, Yee Whye Teh, Arnaud Doucet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/105112d52254f86d5854f3da734a52b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/105112d52254f86d5854f3da734a52b4-Abstract-Conference.html)

        **Abstract**:

        Score-based generative models (SGMs) are a powerful class of generative models that exhibit remarkable empirical performance.Score-based generative modelling (SGM) consists of a noising'' stage, whereby a diffusion is used to gradually add Gaussian noise to data, and a generative model, which entails adenoising'' process defined by approximating the time-reversal of the diffusion. Existing SGMs assume that data is supported on a Euclidean space, i.e. a manifold with flat geometry.  In many domains such as robotics, geoscience or protein modelling,  data is often naturally described by distributions living on Riemannian manifolds and current SGM techniques are not appropriate. We introduce here \emph{Riemannian Score-based Generative Models} (RSGMs), a class of generative models extending SGMs to Riemannian manifolds.  We demonstrate our approach on a variety of compact manifolds, and in particular with earth and climate science spherical data.

        ----

        ## [175] Intra-agent speech permits zero-shot task acquisition

        **Authors**: *Chen Yan, Federico Carnevale, Petko Georgiev, Adam Santoro, Aurelia Guy, Alistair Muldal, Chia-Chun Hung, Josh Abramson, Timothy P. Lillicrap, Gregory Wayne*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/1074541383db5ef12d6ac66d2f8e8d34-Abstract-Conference.html)

        **Abstract**:

        Human language learners are exposed to a trickle of informative, context-sensitive language, but a flood of raw sensory data. Through both social language use and internal processes of rehearsal and practice, language learners are able to build high-level, semantic representations that explain their perceptions. Here, we take inspiration from such processes of "inner speech" in humans (Vygotsky, 1934) to better understand the role of intra-agent speech in embodied behavior. First, we formally pose intra-agent speech as a semi-supervised problem and develop two algorithms that enable visually grounded captioning with little labeled language data. We then experimentally compute scaling curves over different amounts of labeled data and compare the data efficiency against a supervised learning baseline. Finally, we incorporate intra-agent speech into an embodied, mobile manipulator agent operating in a 3D virtual world, and show that with as few as 150 additional image captions, intra-agent speech endows the agent with the ability to manipulate and answer questions about a new object without any related task-directed experience (zero-shot). Taken together, our experiments suggest that modelling intra-agent speech is effective in enabling embodied agents to learn new tasks efficiently and without direct interaction experience.

        ----

        ## [176] Free Probability for predicting the performance of feed-forward fully connected neural networks

        **Authors**: *Reda Chhaibi, Tariq Daouda, Ezechiel Kahn*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/10826a1a80f816ea98d559d7c7a97973-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/10826a1a80f816ea98d559d7c7a97973-Abstract-Conference.html)

        **Abstract**:

        Gradient descent during the learning process of a neural network can be subject to many instabilities. The spectral density of the Jacobian is a key component for analyzing stability. Following the works of Pennington et al., such Jacobians are modeled using free multiplicative convolutions from Free Probability Theory (FPT).We present a reliable and very fast method for computing the associated spectral densities, for given architecture and initialization. This method has a controlled and proven convergence. Our technique is based on an homotopy method: it is an adaptative Newton-Raphson scheme which chains basins of attraction. We find contiguous lilypad-like basins and step from one to the next, heading towards the objective.In order to demonstrate the relevance of our method we show that the relevant FPT metrics computed before training are highly correlated to final test losses â€“ up to 85%. We also give evidence that a very desirable feature for neural networks is the hyperbolicity of their Jacobian at initialization, while remaining at the edge of chaos.

        ----

        ## [177] The Minority Matters: A Diversity-Promoting Collaborative Metric Learning Algorithm

        **Authors**: *Shilong Bao, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, Qingming Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/109cf25cbc36037deecdbeabfa199956-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/109cf25cbc36037deecdbeabfa199956-Abstract-Conference.html)

        **Abstract**:

        Collaborative Metric Learning (CML) has recently emerged as a popular method in recommendation systems (RS), closing the gap between metric learning and Collaborative Filtering. Following the convention of RS, existing methods exploit unique user representation in their model design. This paper focuses on a challenging scenario where a user has multiple categories of interests. Under this setting, we argue that the unique user representation might induce preference bias, especially when the item category distribution is imbalanced. To address this issue, we propose a novel method called Diversity-Promoting Collaborative Metric Learning (DPCML), with the hope of considering the commonly ignored minority interest of the user. The key idea behind DPCML is to include a multiple set of representations for each user in the system. Based on this embedding paradigm, user preference toward an item is aggregated from different embeddings by taking the minimum item-user distance among the user embedding set. Furthermore, we observe that the diversity of the embeddings for the same user also plays an essential role in the model. To this end, we propose a diversity control regularization term to accommodate the multi-vector representation strategy better. Theoretically, we show that DPCML could generalize well to unseen test data by tackling the challenge of the annoying operation that comes from the minimum value. Experiments over a range of benchmark datasets speak to the efficacy of DPCML.

        ----

        ## [178] Open-Ended Reinforcement Learning with Neural Reward Functions

        **Authors**: *Robert Meier, Asier Mujika*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/10a6bdcabbd5a3d36b760daa295f63c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/10a6bdcabbd5a3d36b760daa295f63c1-Abstract-Conference.html)

        **Abstract**:

        Inspired by the great success of unsupervised learning in Computer Vision and Natural Language Processing, the Reinforcement Learning community has recently started to focus more on unsupervised discovery of skills. Most current approaches, like DIAYN or DADS, optimize some form of mutual information objective. We propose a different approach that uses reward functions encoded by neural networks. These are trained iteratively to reward more complex behavior. In high-dimensional robotic environments our approach learns a wide range of interesting skills including front-flips for Half-Cheetah and one-legged running for Humanoid. It is the first skill discovery algorithm that can learn such skills without relying on any form of feature engineering. In the pixel-based Montezuma's Revenge environment our method also works with minimal changes and it learns complex skills that involve interacting with items and visiting diverse locations.

        ----

        ## [179] A Reduction to Binary Approach for Debiasing Multiclass Datasets

        **Authors**: *Ibrahim M. Alabdulmohsin, Jessica Schrouff, Sanmi Koyejo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/10eaa0aae94b34308e9b3fa7b677cbe1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/10eaa0aae94b34308e9b3fa7b677cbe1-Abstract-Conference.html)

        **Abstract**:

        We propose a novel reduction-to-binary (R2B) approach that enforces demographic parity for multiclass classification with non-binary sensitive attributes via a reduction to a sequence of binary debiasing tasks. We prove that R2B satisfies optimality and bias guarantees and demonstrate empirically that it can lead to an improvement over two baselines: (1) treating multiclass problems  as multi-label by debiasing labels independently and (2) transforming the features instead of the labels. Surprisingly, we also demonstrate that independent label debiasing yields competitive results in most (but not all) settings. We validate these conclusions on synthetic and real-world datasets from social science, computer vision, and healthcare.

        ----

        ## [180] Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks

        **Authors**: *Zhiyang Chen, Yousong Zhu, Zhaowen Li, Fan Yang, Wei Li, Haixin Wang, Chaoyang Zhao, Liwei Wu, Rui Zhao, Jinqiao Wang, Ming Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/112bfcff816203efbb986bc178380ef2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/112bfcff816203efbb986bc178380ef2-Abstract-Conference.html)

        **Abstract**:

        Visual tasks vary a lot in their output formats and concerned contents, therefore it is hard to process them with an identical structure. One main obstacle lies in the high-dimensional outputs in object-level visual tasks. In this paper, we propose an object-centric vision framework, Obj2Seq. Obj2Seq takes objects as basic units, and regards most object-level visual tasks as sequence generation problems of objects. Therefore, these visual tasks can be decoupled into two steps. First recognize objects of given categories, and then generate a sequence for each of these objects. The definition of the output sequences varies for different tasks, and the model is supervised by matching these sequences with ground-truth targets. Obj2Seq is able to flexibly determine input categories to satisfy customized requirements, and be easily extended to different visual tasks. When experimenting on MS COCO, Obj2Seq achieves 45.7% AP on object detection, 89.0% AP on multi-label classification and 65.0% AP on human pose estimation. These results demonstrate its potential to be generally applied to different visual tasks. Code has been made available at: https://github.com/CASIA-IVA-Lab/Obj2Seq.

        ----

        ## [181] Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering

        **Authors**: *Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, Ashwin Kalyan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11332b6b6cf4485b84afadb1352d3a9a-Abstract-Conference.html)

        **Abstract**:

        When answering a question, humans utilize the information available across different modalities to synthesize a consistent and complete chain of thought (CoT). This process is normally a black box in the case of deep learning models like large-scale language models. Recently, science question benchmarks have been used to diagnose the multi-hop reasoning ability and interpretability of an AI system. However, existing datasets fail to provide annotations for the answers, or are restricted to the textual-only modality, small scales, and limited domain diversity. To this end, we present Science Question Answering (ScienceQA), a new benchmark that consists of ~21k multimodal multiple choice questions with a diverse set of science topics and annotations of their answers with corresponding lectures and explanations. We further design language models to learn to generate lectures and explanations as the chain of thought (CoT) to mimic the multi-hop reasoning process when answering ScienceQA questions. ScienceQA demonstrates the utility of CoT in language models, as CoT improves the question answering performance by 1.20% in few-shot GPT-3 and 3.99% in fine-tuned UnifiedQA. We also explore the upper bound for models to leverage explanations by feeding those in the input; we observe that it improves the few-shot performance of GPT-3 by 18.96%. Our analysis further shows that language models, similar to humans, benefit from explanations to learn from fewer data and achieve the same performance with just 40% of the data. The data and code are available at https://scienceqa.github.io.

        ----

        ## [182] Few-Shot Audio-Visual Learning of Environment Acoustics

        **Authors**: *Sagnik Majumder, Changan Chen, Ziad Al-Halah, Kristen Grauman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/113ae3a9762ca2168f860a8501d6ae25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/113ae3a9762ca2168f860a8501d6ae25-Abstract-Conference.html)

        **Abstract**:

        Room impulse response (RIR) functions capture how the surrounding physical environment transforms the sounds heard by a listener, with implications for various applications in AR, VR, and robotics. Whereas traditional methods to estimate RIRs assume dense geometry and/or sound measurements throughout the environment, we explore how to infer RIRs based on a sparse set of images and echoes observed in the space.  Towards that goal, we introduce a transformer-based method that uses self-attention to build a rich acoustic context, then predicts RIRs of arbitrary query source-receiver locations through cross-attention. Additionally, we design a novel training objective that improves the match in the acoustic signature between the RIR predictions and the targets. In experiments using a state-of-the-art audio-visual simulator for 3D environments, we demonstrate that our method successfully generates arbitrary RIRs, outperforming state-of-the-art methods and---in a major departure from traditional methods---generalizing to novel environments in a few-shot manner. Project: http://vision.cs.utexas.edu/projects/fs_rir

        ----

        ## [183] The Phenomenon of Policy Churn

        **Authors**: *Tom Schaul, André Barreto, John Quan, Georg Ostrovski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/114292cf3f930ba157ed33f66997fee2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/114292cf3f930ba157ed33f66997fee2-Abstract-Conference.html)

        **Abstract**:

        We identify and study the phenomenon of policy churn, that is, the rapid change of the greedy policy in value-based reinforcement learning. Policy churn operates at a surprisingly rapid pace, changing the greedy action in a large fraction of states within a handful of learning updates (in a typical deep RL set-up such as DQN on Atari). We characterise the phenomenon empirically, verifying that it is not limited to specific algorithm or environment properties. A number of ablations help whittle down the plausible explanations on why churn occurs to just a handful, all related to deep learning. Finally, we hypothesise that policy churn is a beneficial but overlooked form of implicit exploration that casts $\epsilon$-greedy exploration in a fresh light, namely that $\epsilon$-noise plays a much smaller role than expected.

        ----

        ## [184] Molecule Generation by Principal Subgraph Mining and Assembling

        **Authors**: *Xiangzhe Kong, Wenbing Huang, Zhixing Tan, Yang Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/1160792eab11de2bbaf9e71fce191e8c-Abstract-Conference.html)

        **Abstract**:

        Molecule generation is central to a variety of applications. Current attention has been paid to approaching the generation task as subgraph prediction and assembling. Nevertheless, these methods usually rely on hand-crafted or external subgraph construction, and the subgraph assembling depends solely on local arrangement. In this paper, we define a novel notion, principal subgraph that is closely related to the informative pattern within molecules. Interestingly, our proposed merge-and-update subgraph extraction method can automatically discover frequent principal subgraphs from the dataset, while previous methods are incapable of. Moreover, we develop a two-step subgraph assembling strategy, which first predicts a set of subgraphs in a sequence-wise manner and then assembles all generated subgraphs globally as the final output molecule.  Built upon graph variational auto-encoder, our model is demonstrated to be effective in terms of several evaluation metrics and efficiency, compared with state-of-the-art methods on distribution learning and (constrained) property optimization tasks.

        ----

        ## [185] Implicit Neural Representations with Levels-of-Experts

        **Authors**: *Zekun Hao, Arun Mallya, Serge J. Belongie, Ming-Yu Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/1165af8b913fb836c6280b42d6e0084f-Abstract-Conference.html)

        **Abstract**:

        Coordinate-based networks, usually in the forms of MLPs, have been successfully applied to the task of predicting high-frequency but low-dimensional signals using coordinate inputs. To scale them to model large-scale signals, previous works resort to hybrid representations, combining a coordinate-based network with a grid-based representation, such as sparse voxels. However, such approaches lack a compact global latent representation in its grid, making it difficult to model a distribution of signals, which is important for generalization tasks. To address the limitation, we propose the Levels-of-Experts (LoE) framework, which is a novel coordinate-based representation consisting of an MLP with periodic, position-dependent weights arranged hierarchically. For each linear layer of the MLP, multiple candidate values of its weight matrix are tiled and replicated across the input space, with different layers replicating at different frequencies. Based on the input, only one of the weight matrices is chosen for each layer. This greatly increases the model capacity without incurring extra computation or compromising generalization capability. We show that the new representation is an efficient and competitive drop-in replacement for a wide range of tasks, including signal fitting, novel view synthesis, and generative modeling.

        ----

        ## [186] Planning for Sample Efficient Imitation Learning

        **Authors**: *Zhao-Heng Yin, Weirui Ye, Qifeng Chen, Yang Gao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11715d433f6f8b9106baae0df023deb3-Abstract-Conference.html)

        **Abstract**:

        Imitation learning is a class of promising policy learning algorithms that is free from many practical issues with reinforcement learning, such as the reward design issue and the exploration hardness. However, the current imitation algorithm struggles to achieve both high performance and high in-environment sample efficiency simultaneously. Behavioral Cloning (BC) does not need in-environment interactions, but it suffers from the covariate shift problem which harms its performance. Adversarial Imitation Learning (AIL) turns imitation learning into a distribution matching problem. It can achieve better performance on some tasks but it requires a large number of in-environment interactions. Inspired by the recent success of EfficientZero in RL, we propose EfficientImitate (EI), a planning-based imitation learning method that can achieve high in-environment sample efficiency and performance simultaneously. Our algorithmic contribution in this paper is two-fold. First, we extend AIL into the MCTS-based RL. Second, we show the seemingly incompatible two classes of imitation algorithms (BC and AIL) can be naturally unified under our framework, enjoying the benefits of both. We benchmark our method not only on the state-based DeepMind Control Suite but also on the image version which many previous works find highly challenging. Experimental results show that EI achieves state-of-the-art results in performance and sample efficiency. EI shows over 4x gain in performance in the limited sample setting on state-based and image-based tasks and can solve challenging problems like Humanoid, where previous methods fail with a small amount of interactions. Our code is available at https://github.com/zhaohengyin/EfficientImitate.

        ----

        ## [187] Concept Activation Regions: A Generalized Framework For Concept-Based Explanations

        **Authors**: *Jonathan Crabbé, Mihaela van der Schaar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11a7f429d75f9f8c6e9c630aeb6524b5-Abstract-Conference.html)

        **Abstract**:

        Concept-based explanations permit to understand the predictions of a deep neural network (DNN) through the lens of concepts specified by users. Existing methods assume that the examples illustrating a concept are mapped in a fixed direction of the DNN's latent space. When this holds true, the concept can be represented by a concept activation vector (CAV) pointing in that direction. In this work, we propose to relax this assumption by allowing concept examples to be scattered across different clusters in the DNN's latent space. Each concept is then represented by a region of the DNN's latent space that includes these clusters and that we call concept activation region (CAR). To formalize this idea, we introduce an extension of the CAV formalism that is based on the kernel trick and support vector classifiers. This CAR formalism yields global concept-based explanations and local concept-based feature importance. We prove that CAR explanations built with radial kernels are invariant under latent space isometries. In this way, CAR assigns the same explanations to latent spaces that have the same geometry. We further demonstrate empirically that CARs offer (1) more accurate descriptions of how concepts are scattered in the DNN's latent space; (2) global explanations that are closer to human concept annotations and (3) concept-based feature importance that meaningfully relate concepts with each other. Finally, we use CARs to show that DNNs can autonomously rediscover known scientific concepts, such as the prostate cancer grading system.

        ----

        ## [188] Towards Safe Reinforcement Learning with a Safety Editor Policy

        **Authors**: *Haonan Yu, Wei Xu, Haichao Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11afefdd848d1bc9ac9f1604d9f45817-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11afefdd848d1bc9ac9f1604d9f45817-Abstract-Conference.html)

        **Abstract**:

        We consider the safe reinforcement learning (RL) problem of maximizing utility with extremely low constraint violation rates. Assuming no prior knowledge or pre-training of the environment safety model given a task, an agent has to learn, via exploration, which states and actions are safe. A popular approach in this line of research is to combine a model-free RL algorithm with the Lagrangian method to adjust the weight of the constraint reward relative to the utility reward dynamically. It relies on a single policy to handle the conflict between utility and constraint rewards, which is often challenging. We present SEditor, a two-policy approach that learns a safety editor policy transforming potentially unsafe actions proposed by a utility maximizer policy into safe ones. The safety editor is trained to maximize the constraint reward while minimizing a hinge loss of the utility state-action values before and after an action is edited. SEditor extends existing safety layer designs that assume simplified safety models, to general safe RL scenarios where the safety model can in theory be arbitrarily complex. As a first-order method, it is easy to implement and efficient for both inference and training. On 12 Safety Gym tasks and 2 safe racing tasks, SEditor obtains much a higher overall safety-weighted-utility (SWU) score than the baselines, and demonstrates outstanding utility performance with constraint violation rates as low as once per 2k time steps, even in obstacle-dense environments. On some tasks, this low violation rate is up to 200 times lower than that of an unconstrained RL method with similar utility performance. Code is available at https://github.com/hnyu/seditor.

        ----

        ## [189] Understanding Cross-Domain Few-Shot Learning Based on Domain Similarity and Few-Shot Difficulty

        **Authors**: *Jaehoon Oh, Sungnyun Kim, Namgyu Ho, Jin-Hwa Kim, Hwanjun Song, Se-Young Yun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11b3ae28275461741026c46c0c786711-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11b3ae28275461741026c46c0c786711-Abstract-Conference.html)

        **Abstract**:

        Cross-domain few-shot learning (CD-FSL) has drawn increasing attention for handling large differences between the source and target domains--an important concern in real-world scenarios. To overcome these large differences, recent works have considered exploiting small-scale unlabeled data from the target domain during the pre-training stage. This data enables self-supervised pre-training on the target domain, in addition to supervised pre-training on the source domain. In this paper, we empirically investigate which pre-training is preferred based on domain similarity and few-shot difficulty of the target domain. We discover that the performance gain of self-supervised pre-training over supervised pre-training becomes large when the target domain is dissimilar to the source domain, or the target domain itself has low few-shot difficulty. We further design two pre-training schemes, mixed-supervised and two-stage learning, that improve performance. In this light, we present six findings for CD-FSL, which are supported by extensive experiments and analyses on three source and eight target benchmark datasets with varying levels of domain similarity and few-shot difficulty. Our code is available at https://github.com/sungnyun/understanding-cdfsl.

        ----

        ## [190] Ambiguous Images With Human Judgments for Robust Visual Event Classification

        **Authors**: *Kate Sanders, Reno Kriz, Anqi Liu, Benjamin Van Durme*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/11e3e0f1b29dcd31bd0952bfc1357f68-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Contemporary vision benchmarks predominantly consider tasks on which humans can achieve near-perfect performance. However, humans are frequently presented with visual data that they cannot classify with 100% certainty, and models trained on standard vision benchmarks achieve low performance when evaluated on this data. To address this issue, we introduce a procedure for creating datasets of ambiguous images and use it to produce SQUID-E ("Squidy"), a collection of noisy images extracted from videos. All images are annotated with ground truth values and a test set is annotated with human uncertainty judgments. We use this dataset to characterize human uncertainty in vision tasks and evaluate existing visual event classification models. Experimental results suggest that existing vision models are not sufficiently equipped to provide meaningful outputs for ambiguous images and that datasets of this nature can be used to assess and improve such models through model training and direct evaluation of model calibration. These findings motivate large-scale ambiguous dataset creation and further research focusing on noisy visual data.

        ----

        ## [191] Sustainable Online Reinforcement Learning for Auto-bidding

        **Authors**: *Zhiyu Mou, Yusen Huo, Rongquan Bai, Mingzhou Xie, Chuan Yu, Jian Xu, Bo Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11faf17bf7e5412d9cded369f97db23d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11faf17bf7e5412d9cded369f97db23d-Abstract-Conference.html)

        **Abstract**:

        Recently, auto-bidding technique has become an essential tool to increase the revenue of advertisers. Facing the complex and ever-changing bidding environments in the real-world advertising system (RAS), state-of-the-art auto-bidding policiesÂ usually leverage reinforcement learning (RL) algorithms to generate real-time bids on behalf of the advertisers. Due to safety concerns, it was believed that the RL training process can only be carried out in an offline virtual advertising system (VAS) that is built based on the historical data generated in the RAS. In this paper, we argue that there exists significant gaps between the VAS and RAS, making the RL training process suffer from the problem of inconsistency between online and offline (IBOO). Firstly, we formally define the IBOO and systematically analyze its causes and influences. Then, to avoid the IBOO, we propose a sustainable online RL (SORL) framework that trains the auto-bidding policy by directly interacting with the RAS, instead of learning in the VAS. Specifically, based on our proof of the Lipschitz smooth property of the Q function, we design a safe and efficient online exploration (SER) policy for continuously collecting data from the RAS. Meanwhile, we derive the theoretical lower bound on the safety degree of the SER policy. We also develop a variance-suppressed conservative Q-learning (V-CQL) method to effectively and stably learn the auto-bidding policy with the collected data. Finally, extensive simulated and real-world experiments validate the superiority of our approach over the state-of-the-art auto-bidding algorithm.

        ----

        ## [192] Uni-Perceiver-MoE: Learning Sparse Generalist Models with Conditional MoEs

        **Authors**: *Jinguo Zhu, Xizhou Zhu, Wenhai Wang, Xiaohua Wang, Hongsheng Li, Xiaogang Wang, Jifeng Dai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/11fc8c98b46d4cbdfe8157267228f7d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/11fc8c98b46d4cbdfe8157267228f7d7-Abstract-Conference.html)

        **Abstract**:

        To build an artificial neural network like the biological intelligence system, recent works have unified numerous tasks into a generalist model, which can process various tasks with shared parameters and do not have any task-specific modules. While generalist models achieve promising results on various benchmarks, they have performance degradation on some tasks compared with task-specialized models. In this work, we find that interference among different tasks and modalities is the main factor to this phenomenon. To mitigate such interference, we introduce the Conditional Mixture-of-Experts (Conditional MoEs) to generalist models. Routing strategies under different levels of conditions are proposed to take both the training/inference cost and generalization ability into account. By incorporating the proposed Conditional MoEs, the recently proposed generalist model Uni-Perceiver can effectively mitigate the interference across tasks and modalities, and achieves state-of-the-art results on a series of downstream tasks via prompt tuning on 1% of downstream data. Moreover, the introduction of Conditional MoEs still holds the generalization ability of generalist models to conduct zero-shot inference on new tasks, e.g., videotext retrieval and video caption. Code and pre-trained generalist models are publicly released at https://github.com/fundamentalvision/Uni-Perceiver.

        ----

        ## [193] Improved Coresets for Euclidean k-Means

        **Authors**: *Vincent Cohen-Addad, Kasper Green Larsen, David Saulpic, Chris Schwiegelshohn, Omar Ali Sheikh-Omar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/120c9ab5c58ba0fa9dd3a22ace1de245-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/120c9ab5c58ba0fa9dd3a22ace1de245-Abstract-Conference.html)

        **Abstract**:

        Given a set of $n$ points in $d$ dimensions, the Euclidean $k$-means problem (resp. Euclidean $k$-median) consists of finding $k$ centers such that the sum of squared distances (resp. sum of distances) from every point to its closest center is minimized. The arguably most popular way of dealing with this problem in the big data setting is to first compress the data by computing a weighted subset known as a coreset and then run any algorithm on this subset. The guarantee of the coreset is that for any candidate solution, the ratio between coreset cost and the cost of the original instance is less than a $(1\pm \varepsilon)$ factor. The current state of the art coreset size is $\tilde O(\min(k^{2} \cdot \varepsilon^{-2},k\cdot \varepsilon^{-4}))$ for Euclidean $k$-means and $\tilde O(\min(k^{2} \cdot \varepsilon^{-2},k\cdot \varepsilon^{-3}))$ for Euclidean $k$-median. The best known lower bound for both problems is $\Omega(k\varepsilon^{-2})$. In this paper, we improve these bounds to $\tilde O(\min(k^{3/2} \cdot \varepsilon^{-2},k\cdot \varepsilon^{-4}))$ for Euclidean $k$-means and $\tilde O(\min(k^{4/3} \cdot \varepsilon^{-2},k\cdot \varepsilon^{-3}))$ for Euclidean $k$-median. In particular, ours is the first provable bound that breaks through the $k^2$ barrier while retaining an optimal dependency on $\varepsilon$.

        ----

        ## [194] Accelerated Linearized Laplace Approximation for Bayesian Deep Learning

        **Authors**: *Zhijie Deng, Feng Zhou, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/12143893d9d37c3569dda800b95cabd9-Abstract-Conference.html)

        **Abstract**:

        Laplace approximation (LA) and its linearized variant (LLA) enable effortless adaptation of pretrained deep neural networks to Bayesian neural networks. The generalized Gauss-Newton (GGN) approximation is typically introduced to improve their tractability. However, LA and LLA are still confronted with non-trivial inefficiency issues and should rely on Kronecker-factored, diagonal, or even last-layer approximate GGN matrices in practical use. These approximations are likely to harm the fidelity of learning outcomes. To tackle this issue, inspired by the connections between LLA and neural target kernels (NTKs), we develop a Nystrom approximation to NTKs to accelerate LLA. Our method benefits from the capability of popular deep learning libraries for forward mode automatic differentiation, and enjoys reassuring theoretical guarantees. Extensive studies reflect the merits of the proposed method in aspects of both scalability and performance. Our method can even scale up to architectures like vision transformers. We also offer valuable ablation studies to diagnose our method. Code is available at https://github.com/thudzj/ELLA.

        ----

        ## [195] Learning to Reason with Neural Networks: Generalization, Unseen Data and Boolean Measures

        **Authors**: *Emmanuel Abbe, Samy Bengio, Elisabetta Cornacchia, Jon M. Kleinberg, Aryo Lotfi, Maithra Raghu, Chiyuan Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/12202970782399ee67981dc5269c3b8a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/12202970782399ee67981dc5269c3b8a-Abstract-Conference.html)

        **Abstract**:

        This paper considers the Pointer Value Retrieval (PVR) benchmark introduced in [ZRKB21], where a `reasoning' function acts on a string of digits to produce the label. More generally, the paper considers the learning of logical functions with gradient descent (GD) on neural networks. It is first shown that in order to learn logical functions with gradient descent on symmetric neural networks, the generalization error can be lower-bounded in terms of the noise-stability of the target function, supporting a conjecture made in [ZRKB21]. It is then shown that in the distribution shift setting, when the data withholding corresponds to freezing a single feature (referred to as canonical holdout), the generalization error of gradient descent admits a tight characterization in terms of the Boolean influence for several relevant architectures. This is shown on linear models and supported experimentally on other models such as MLPs and Transformers. In particular, this puts forward the hypothesis that for such architectures and for learning logical functions such as PVR functions, GD tends to have an implicit bias towards low-degree representations, which in turn gives the Boolean influence for the generalization error under quadratic loss.

        ----

        ## [196] Using Partial Monotonicity in Submodular Maximization

        **Authors**: *Loay Mualem, Moran Feldman*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/1227a7a80529ecfe033065b9fcc5a042-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/1227a7a80529ecfe033065b9fcc5a042-Abstract-Conference.html)

        **Abstract**:

        Over the last two decades, submodular function maximization has been the workhorse of many discrete optimization problems in machine learning applications. Traditionally, the study of submodular functions was based on binary function properties, but recent works began to consider continuous function properties such as the submodularity ratio and the curvature. The monotonicity property of set functions plays a central role in submodular maximization. Nevertheless, no continuous version of this property has been suggested to date (as far as we know), which is unfortunate since submoduar functions that are almost monotone often arise in machine learning applications. In this work we fill this gap by defining the monotonicity ratio, which is a continuous version of the monotonicity property. We then show that for many standard submodular maximization algorithms one can prove new approximation guarantees that depend on the monotonicity ratio; leading to improved approximation ratios for the common machine learning applications of movie recommendation, quadratic programming, image summarization and ride-share optimization.

        ----

        ## [197] Enhanced Meta Reinforcement Learning via Demonstrations in Sparse Reward Environments

        **Authors**: *Desik Rengarajan, Sapana Chaudhary, Jaewon Kim, Dileep Kalathil, Srinivas Shakkottai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/122f45f4d451617ac87adf7024ee14cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/122f45f4d451617ac87adf7024ee14cd-Abstract-Conference.html)

        **Abstract**:

        Meta reinforcement learning (Meta-RL) is an approach wherein the experience gained from solving a variety of tasks is distilled into a meta-policy. The meta-policy, when adapted over only a small (or just a single) number of steps, is able to perform near-optimally on a new, related task.  However, a major challenge to adopting this approach to solve real-world problems is that they are often associated with sparse reward functions that only indicate whether a task is completed partially or fully. We consider the situation where some data, possibly generated by a sub-optimal agent, is available for each task. We then develop a class of algorithms entitled Enhanced Meta-RL via Demonstrations (EMRLD) that exploit this information---even if sub-optimal---to obtain guidance during training. We show how EMRLD jointly utilizes RL and supervised learning over the offline data to generate a meta-policy that demonstrates monotone performance improvements. We also develop a warm started variant called EMRLD-WS that is particularly efficient for sub-optimal demonstration data. Finally, we show that our EMRLD algorithms significantly outperform existing approaches in a variety of sparse reward environments, including that of a mobile robot.

        ----

        ## [198] Riemannian Diffusion Models

        **Authors**: *Chin-Wei Huang, Milad Aghajohari, Joey Bose, Prakash Panangaden, Aaron C. Courville*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/123d3e814e257e0781e5d328232ead9b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/123d3e814e257e0781e5d328232ead9b-Abstract-Conference.html)

        **Abstract**:

        Diffusion models are recent state-of-the-art methods for image generation and likelihood estimation. In this work, we generalize continuous-time diffusion models to arbitrary Riemannian manifolds and derive a variational framework for likelihood estimation. Computationally, we propose new methods for computing the Riemannian divergence which is needed for likelihood estimation. Moreover, in generalizing the Euclidean case, we prove that maximizing this variational lower-bound is equivalent to Riemannian score matching. Empirically, we demonstrate the expressive power of Riemannian diffusion models on a wide spectrum of smooth manifolds, such as spheres, tori, hyperboloids, and orthogonal groups. Our proposed method achieves new state-of-the-art likelihoods on all benchmarks.

        ----

        ## [199] Training and Inference on Any-Order Autoregressive Models the Right Way

        **Authors**: *Andy Shih, Dorsa Sadigh, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/123fd8a56501194823c8e0dca00733df-Abstract-Conference.html)

        **Abstract**:

        Conditional inference on arbitrary subsets of variables is a core problem in probabilistic inference with important applications such as masked language modeling and image inpainting. In recent years, the family of Any-Order Autoregressive Models (AO-ARMs) -- closely related to popular models such as BERT and XLNet -- has shown breakthrough performance in arbitrary conditional tasks across a sweeping range of domains. But, in spite of their success, in this paper we identify significant improvements to be made to previous formulations of AO-ARMs. First, we show that AO-ARMs suffer from redundancy in their probabilistic model, i.e., they define the same distribution in multiple different ways. We alleviate this redundancy by training on a smaller set of univariate conditionals that still maintains support for efficient arbitrary conditional inference. Second, we upweight the training loss for univariate conditionals that are evaluated more frequently during inference. Our method leads to improved performance with no compromises on tractability, giving state-of-the-art likelihoods in arbitrary conditional modeling on text (Text8), image (CIFAR10, ImageNet32), and continuous tabular data domains.

        ----

        

[Go to the next page](NIPS-2022-list02.md)

[Go to the catalog section](README.md)