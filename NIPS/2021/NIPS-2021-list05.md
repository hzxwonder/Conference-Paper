## [800] Scalable Quasi-Bayesian Inference for Instrumental Variable Regression

        **Authors**: *Ziyu Wang, Yuhao Zhou, Tongzheng Ren, Jun Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56a3107cad6611c8337ee36d178ca129-Abstract.html)

        **Abstract**:

        Recent years have witnessed an upsurge of interest in employing flexible machine learning models for instrumental variable (IV) regression, but the development of uncertainty quantification methodology is still lacking.  In this work we present a scalable quasi-Bayesian procedure for IV regression, building upon the recently developed kernelized IV models.  Contrary to Bayesian modeling for IV, our approach does not require additional assumptions on the data generating process, and leads to a scalable approximate inference algorithm with time cost comparable to the corresponding point estimation methods.  Our algorithm can be further extended to work with neural network models.  We analyze the theoretical properties of the proposed quasi-posterior, and demonstrate through empirical evaluation the competitive performance of our method.

        ----

        ## [801] Kernel Identification Through Transformers

        **Authors**: *Fergus Simpson, Ian Davies, Vidhi Lalchand, Alessandro Vullo, Nicolas Durrande, Carl Edward Rasmussen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56c3b2c6ea3a83aaeeff35eeb45d700d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56c3b2c6ea3a83aaeeff35eeb45d700d-Abstract.html)

        **Abstract**:

        Kernel selection plays a central role in determining the performance of Gaussian Process (GP) models, as the chosen kernel determines both the inductive biases and prior support of functions under the GP prior. This work addresses the challenge of constructing custom kernel functions for high-dimensional GP regression models. Drawing inspiration from recent progress in deep learning, we introduce a novel approach named KITT: Kernel Identification Through Transformers. KITT exploits a transformer-based architecture to generate kernel recommendations in under 0.1 seconds, which is several orders of magnitude faster than conventional kernel search algorithms. We train our model using synthetic data generated from priors over a vocabulary of known kernels. By exploiting the nature of the self-attention mechanism, KITT is able to process datasets with inputs of arbitrary dimension. We demonstrate that kernels chosen by KITT yield strong performance over a diverse collection of regression benchmarks.

        ----

        ## [802] Curriculum Design for Teaching via Demonstrations: Theory and Applications

        **Authors**: *Gaurav Yengera, Rati Devidze, Parameswaran Kamalaruban, Adish Singla*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56c51a39a7c77d8084838cc920585bd0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56c51a39a7c77d8084838cc920585bd0-Abstract.html)

        **Abstract**:

        We consider the problem of teaching via demonstrations in sequential decision-making settings. In particular, we study how to design a personalized curriculum over demonstrations to speed up the learner's convergence. We provide a unified curriculum strategy for two popular learner models: Maximum Causal Entropy Inverse Reinforcement Learning (MaxEnt-IRL) and Cross-Entropy Behavioral Cloning (CrossEnt-BC). Our unified strategy induces a ranking over demonstrations based on a notion of difficulty scores computed w.r.t. the teacher's optimal policy and the learner's current policy. Compared to the state of the art, our strategy doesn't require access to the learner's internal dynamics and still enjoys similar convergence guarantees under mild technical conditions. Furthermore, we adapt our curriculum strategy to the setting where no teacher agent is present using task-specific difficulty scores. Experiments on a synthetic car driving environment and navigation-based environments demonstrate the effectiveness of our curriculum strategy.

        ----

        ## [803] Revenue maximization via machine learning with noisy data

        **Authors**: *Ellen Vitercik, Tom Yan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56d33021e640f5d64a611a71b5dc30a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56d33021e640f5d64a611a71b5dc30a3-Abstract.html)

        **Abstract**:

        Increasingly, copious amounts of consumer data are used to learn high-revenue mechanisms via machine learning. Existing research on mechanism design via machine learning assumes that there is a distribution over the buyers' values for the items for sale and that the learning algorithm's input is a training set sampled from this distribution. This setup makes the strong assumption that no noise is introduced during data collection. In order to help place mechanism design via machine learning on firm foundations, we investigate the extent to which this learning process is robust to noise. Optimizing revenue using noisy data is challenging because revenue functions are extremely volatile: an infinitesimal change in the buyers' values can cause a steep drop in revenue. Nonetheless, we provide guarantees when arbitrarily correlated noise is added to the training set; we only require that the noise has bounded magnitude or is sub-Gaussian. We conclude with an application of our guarantees to multi-task mechanism design, where there are multiple distributions over buyers' values and the goal is to learn a high-revenue mechanism per distribution. To our knowledge, we are the first to study mechanism design via machine learning with noisy data as well as multi-task mechanism design.

        ----

        ## [804] Exploiting Data Sparsity in Secure Cross-Platform Social Recommendation

        **Authors**: *Jinming Cui, Chaochao Chen, Lingjuan Lyu, Carl Yang, Li Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56db57b4db0a6fcb7f9e0c0b504f6472-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56db57b4db0a6fcb7f9e0c0b504f6472-Abstract.html)

        **Abstract**:

        Social recommendation has shown promising improvements over traditional systems since it leverages social correlation data as an additional input. Most existing work assumes that all data are available to the recommendation platform. However, in practice, user-item interaction data (e.g.,rating) and user-user social data are usually generated by different platforms, and both of which contain sensitive information.  Therefore, "How to perform secure and efficient social recommendation across different platforms, where the data are highly-sparse in nature" remains an important challenge. In this work, we bring secure computation techniques into social recommendation, and propose S3Rec, a sparsity-aware secure cross-platform social recommendation framework. As a result, our model can not only improve the recommendation performance of the rating platform by incorporating the sparse social data on the social platform, but also protect data privacy of both platforms. Moreover, to further improve model training efficiency, we propose two secure sparse matrix multiplication protocols based on homomorphic encryption and private information retrieval. Our experiments on two benchmark datasets demonstrate the effectiveness of S3Rec.

        ----

        ## [805] Parallelizing Thompson Sampling

        **Authors**: *Amin Karbasi, Vahab S. Mirrokni, Mohammad Shadravan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/56f0b515214a7ec9f08a4bbf9a56f7ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/56f0b515214a7ec9f08a4bbf9a56f7ba-Abstract.html)

        **Abstract**:

        How can we make use of information parallelism in online decision-making problems while efficiently balancing the exploration-exploitation trade-off? In this paper, we introduce a batch Thompson Sampling framework for two canonical online decision-making problems with partial feedback, namely,  stochastic multi-arm bandit and linear contextual bandit. Over a time horizon $T$,  our batch Thompson Sampling policy achieves the same  (asymptotic) regret bound of a fully sequential one while carrying out only   $O(\log T)$ batch queries.  To achieve this exponential reduction, i.e., reducing the number of interactions from $T$ to $O(\log T)$, our batch policy dynamically determines the duration of each batch in order to balance the exploration-exploitation trade-off. We also demonstrate experimentally that dynamic batch allocation outperforms natural baselines.

        ----

        ## [806] Dynamic Causal Bayesian Optimization

        **Authors**: *Virginia Aglietti, Neil Dhir, Javier González, Theodoros Damoulas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/577bcc914f9e55d5e4e4f82f9f00e7d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/577bcc914f9e55d5e4e4f82f9f00e7d4-Abstract.html)

        **Abstract**:

        We study the problem of performing a sequence of optimal interventions in a dynamic causal system where both the target variable of interest, and the inputs, evolve over time. This problem arises in a variety of domains including healthcare, operational research and policy design. Our approach, which we call Dynamic Causal Bayesian Optimisation (DCBO), brings together ideas from decision making, causal inference and Gaussian process (GP) emulation. DCBO is useful in scenarios where the causal effects are changing over time. Indeed, at every time step, DCBO identifies a local optimal intervention by integrating both observational and past interventional data collected from the system. We give theoretical results detailing how one can transfer interventional information across time steps and define a dynamic causal GP model which can be used to find optimal interventions in practice. Finally, we demonstrate how DCBO identifies optimal interventions faster than competing approaches in multiple settings and applications.

        ----

        ## [807] Local Differential Privacy for Regret Minimization in Reinforcement Learning

        **Authors**: *Evrard Garcelon, Vianney Perchet, Ciara Pike-Burke, Matteo Pirotta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/580760fb5def6e2ca8eaf601236d5b08-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/580760fb5def6e2ca8eaf601236d5b08-Abstract.html)

        **Abstract**:

        Reinforcement learning algorithms are widely used in domains where it is desirable to provide a personalized service. In these domains it is common that user data contains sensitive information that needs to be protected from third parties. Motivated by this, we study privacy in the context of finite-horizon Markov Decision Processes (MDPs) by requiring information to be obfuscated on the user side. We formulate this notion of privacy for RL by leveraging the local differential privacy (LDP) framework. We establish a lower bound for regret minimization in finite-horizon MDPs with LDP guarantees which shows that guaranteeing privacy has a multiplicative effect on the regret. This result shows that while LDP is an appealing notion of privacy, it makes the learning problem significantly more complex. Finally, we present an optimistic algorithm that simultaneously satisfies $\varepsilon$-LDP requirements, and achieves $\sqrt{K}/\varepsilon$ regret in any finite-horizon MDP after $K$ episodes,  matching the lower bound dependency on the number of episodes $K$.

        ----

        ## [808] Emergent Discrete Communication in Semantic Spaces

        **Authors**: *Mycal Tucker, Huao Li, Siddharth Agrawal, Dana Hughes, Katia P. Sycara, Michael Lewis, Julie A. Shah*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5812f92450ccaf17275500841c70924a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5812f92450ccaf17275500841c70924a-Abstract.html)

        **Abstract**:

        Neural agents trained in reinforcement learning settings can learn to communicate among themselves via discrete tokens, accomplishing as a team what agents would be unable to do alone. However, the current standard of using one-hot vectors as discrete communication tokens prevents agents from acquiring more desirable aspects of communication such as zero-shot understanding. Inspired by word embedding techniques from natural language processing, we propose neural agent architectures that enables them to communicate via discrete tokens derived from a learned, continuous space. We show in a decision theoretic framework that our technique optimizes communication over a wide range of scenarios, whereas one-hot tokens are only optimal under restrictive assumptions. In self-play experiments, we validate that our trained agents learn to cluster tokens in semantically-meaningful ways, allowing them communicate in noisy environments where other techniques fail. Lastly, we demonstrate both that agents using our method can effectively respond to novel human communication and that humans can understand unlabeled emergent agent communication, outperforming the use of one-hot communication.

        ----

        ## [809] Drop, Swap, and Generate: A Self-Supervised Approach for Generating Neural Activity

        **Authors**: *Ran Liu, Mehdi Azabou, Max Dabagia, Chi-Heng Lin, Mohammad Gheshlaghi Azar, Keith B. Hengen, Michal Valko, Eva L. Dyer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/58182b82110146887c02dbd78719e3d5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/58182b82110146887c02dbd78719e3d5-Abstract.html)

        **Abstract**:

        Meaningful and simplified representations of neural activity can yield insights into how and what information is being processed within a neural circuit. However, without labels, finding representations that reveal the link between the brain and behavior can be challenging. Here, we introduce a novel unsupervised approach for learning disentangled representations of neural activity called Swap-VAE. Our approach combines a generative modeling framework with an instance-specific alignment loss that tries to maximize the representational similarity between transformed views of the input (brain state). These transformed (or augmented) views are created by dropping out neurons and jittering samples in time, which intuitively should lead the network to a representation that maintains both temporal consistency and invariance to the specific neurons used to represent the neural state. Through evaluations on both synthetic data and neural recordings from hundreds of neurons in different primate brains, we show that it is possible to build representations that disentangle neural datasets along relevant latent dimensions linked to behavior.

        ----

        ## [810] Equivariant Manifold Flows

        **Authors**: *Isay Katsman, Aaron Lou, Derek Lim, Qingxuan Jiang, Ser-Nam Lim, Christopher De Sa*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/581b41df0cd50ace849e061ef74827fc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/581b41df0cd50ace849e061ef74827fc-Abstract.html)

        **Abstract**:

        Tractably modelling distributions over manifolds has long been an important goal in the natural sciences. Recent work has focused on developing general machine learning models to learn such distributions. However, for many applications these distributions must respect manifold symmetriesâ€”a trait which most previous models disregard. In this paper, we lay the theoretical foundations for learning symmetry-invariant distributions on arbitrary manifolds via equivariant manifold flows. We demonstrate the utility of our approach by learning quantum field theory-motivated invariant SU(n) densities and by correcting meteor impact dataset bias.

        ----

        ## [811] Scalable Bayesian GPFA with automatic relevance determination and discrete noise models

        **Authors**: *Kristopher T. Jensen, Ta-Chu Kao, Jasmine Stone, Guillaume Hennequin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/58238e9ae2dd305d79c2ebc8c1883422-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/58238e9ae2dd305d79c2ebc8c1883422-Abstract.html)

        **Abstract**:

        Latent variable models are ubiquitous in the exploratory analysis of neural population recordings, where they allow researchers to summarize the activity of large populations of neurons in lower dimensional ‘latent’ spaces. Existing methods can generally be categorized into (i) Bayesian methods that facilitate flexible incorporation of prior knowledge and uncertainty estimation, but which typically do not scale to large datasets; and (ii) highly parameterized methods without explicit priors that scale better but often struggle in the low-data regime. Here, we bridge this gap by developing a fully Bayesian yet scalable version of Gaussian process factor analysis (bGPFA), which models neural data as arising from a set of inferred latent processes with a prior that encourages smoothness over time. Additionally, bGPFA uses automatic relevance determination to infer the dimensionality of neural activity directly from the training data during optimization. To enable the analysis of continuous recordings without trial structure, we introduce a novel variational inference strategy that scales near-linearly in time and also allows for non-Gaussian noise models appropriate for electrophysiological recordings. We apply bGPFA to continuous recordings spanning 30 minutes with over 14 million data points from primate motor and somatosensory cortices during a self-paced reaching task. We show that neural activity progresses from an initial state at target onset to a reach- specific preparatory state well before movement onset. The distance between these initial and preparatory latent states is predictive of reaction times across reaches, suggesting that such preparatory dynamics have behavioral relevance despite the lack of externally imposed delay periods. Additionally, bGPFA discovers latent processes that evolve over slow timescales on the order of several seconds and contain complementary information about reaction time. These timescales are longer than those revealed by methods which focus on individual movement epochs and may reflect fluctuations in e.g. task engagement.

        ----

        ## [812] Recurrence along Depth: Deep Convolutional Neural Networks with Recurrent Layer Aggregation

        **Authors**: *Jingyu Zhao, Yanwen Fang, Guodong Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/582967e09f1b30ca2539968da0a174fa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/582967e09f1b30ca2539968da0a174fa-Abstract.html)

        **Abstract**:

        This paper introduces a concept of layer aggregation to describe how information from previous layers can be reused to better extract features at the current layer. While DenseNet is a typical example of the layer aggregation mechanism, its redundancy has been commonly criticized in the literature. This motivates us to propose a very light-weighted module, called recurrent layer aggregation (RLA), by making use of the sequential structure of layers in a deep CNN. Our RLA module is compatible with many mainstream deep CNNs, including ResNets, Xception and MobileNetV2, and its effectiveness is verified by our extensive experiments on image classification, object detection and instance segmentation tasks. Specifically, improvements can be uniformly observed on CIFAR, ImageNet and MS COCO datasets, and the corresponding RLA-Nets can surprisingly boost the performances by 2-3% on the object detection task. This evidences the power of our RLA module in helping main CNNs better learn structural information in images.

        ----

        ## [813] Independent Prototype Propagation for Zero-Shot Compositionality

        **Authors**: *Frank Ruis, Gertjan J. Burghouts, Doina Bucur*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/584b98aac2dddf59ee2cf19ca4ccb75e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/584b98aac2dddf59ee2cf19ca4ccb75e-Abstract.html)

        **Abstract**:

        Humans are good at compositional zero-shot reasoning; someone who has never seen a zebra before could nevertheless recognize one when we tell them it looks like a horse with black and white stripes. Machine learning systems, on the other hand, usually leverage spurious correlations in the training data, and while such correlations can help recognize objects in context, they hurt generalization. To be able to deal with underspecified datasets while still leveraging contextual clues during classification, we propose ProtoProp, a novel prototype propagation graph method. First we learn prototypical representations of objects (e.g., zebra) that are independent w.r.t. their attribute labels (e.g., stripes) and vice versa. Next we propagate the independent prototypes through a compositional graph, to learn compositional prototypes of novel attribute-object combinations that reflect the dependencies of the target distribution. The method does not rely on any external data, such as class hierarchy graphs or pretrained word embeddings. We evaluate our approach on AO-Clevr, a synthetic and strongly visual dataset with clean labels, UT-Zappos, a noisy real-world dataset of fine-grained shoe types, and C-GQA, a large-scale object detection dataset modified for compositional zero-shot learning. We show that in the generalized compositional zero-shot setting we outperform state-of-the-art results, and through ablations we show the importance of each part of the method and their contribution to the final results. The code is available on github.

        ----

        ## [814] Universal Graph Convolutional Networks

        **Authors**: *Di Jin, Zhizhi Yu, Cuiying Huo, Rui Wang, Xiao Wang, Dongxiao He, Jiawei Han*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5857d68cd9280bc98d079fa912fd6740-Abstract.html)

        **Abstract**:

        Graph Convolutional Networks (GCNs), aiming to obtain the representation of a node by aggregating its neighbors, have demonstrated great power in tackling various analytics tasks on graph (network) data. The remarkable performance of GCNs typically relies on the homophily assumption of networks, while such assumption cannot always be satisfied, since the heterophily or randomness are also widespread in real-world. This gives rise to one fundamental question: whether networks with different structural properties should adopt different propagation mechanisms? In this paper, we first conduct an experimental investigation. Surprisingly, we discover that there are actually segmentation rules for the propagation mechanism, i.e., 1-hop, 2-hop and $k$-nearest neighbor ($k$NN) neighbors are more suitable as neighborhoods of network with complete homophily, complete heterophily and randomness, respectively. However, the real-world networks are complex, and may present diverse structural properties, e.g., the network dominated by homophily may contain a small amount of randomness. So can we reasonably utilize these segmentation rules to design a universal propagation mechanism independent of the network structural assumption? To tackle this challenge, we develop a new universal GCN framework, namely U-GCN. It first introduces a multi-type convolution to extract information from 1-hop, 2-hop and $k$NN networks simultaneously, and then designs a discriminative aggregation to sufficiently fuse them aiming to given learning objectives. Extensive experiments demonstrate the superiority of U-GCN over state-of-the-arts. The code and data are available at https://github.com/jindi-tju.

        ----

        ## [815] Adversarial Feature Desensitization

        **Authors**: *Pouya Bashivan, Reza Bayat, Adam Ibrahim, Kartik Ahuja, Mojtaba Faramarzi, Touraj Laleh, Blake A. Richards, Irina Rish*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/587b7b833034299fdd5f4b10e7dc9fca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/587b7b833034299fdd5f4b10e7dc9fca-Abstract.html)

        **Abstract**:

        Neural networks are known to be vulnerable to adversarial attacks -- slight but carefully constructed perturbations of the inputs which can drastically impair the network's performance. Many defense methods have been proposed for improving robustness of  deep networks by training them on adversarially perturbed inputs. However, these models often remain vulnerable to new types of attacks not seen during training, and even to slightly stronger versions of previously seen  attacks. In this work, we propose a novel approach to  adversarial robustness, which builds upon the insights from the domain adaptation field. Our method, called Adversarial Feature Desensitization (AFD), aims at learning  features that are invariant towards adversarial perturbations of the inputs. This is achieved through a game where we learn features that are both predictive and robust (insensitive to adversarial attacks), i.e. cannot be used to discriminate between natural and adversarial data. Empirical results on several benchmarks  demonstrate the effectiveness of the proposed approach against a wide range of attack types and attack strengths. Our code is available at https://github.com/BashivanLab/afd.

        ----

        ## [816] Few-Shot Data-Driven Algorithms for Low Rank Approximation

        **Authors**: *Piotr Indyk, Tal Wagner, David P. Woodruff*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/588da7a73a2e919a23cb9a419c4c6d44-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/588da7a73a2e919a23cb9a419c4c6d44-Abstract.html)

        **Abstract**:

        Recently, data-driven and learning-based algorithms for low rank matrix approximation were shown to outperform classical data-oblivious algorithms by wide margins in terms of accuracy.  Those algorithms are based on the optimization of sparse sketching matrices, which lead to large savings in time and memory during testing. However, they require long training times on a large amount of existing data, and rely on access to specialized hardware and software. In this work, we develop new data-driven low rank approximation algorithms with better computational efficiency in the training phase, alleviating these drawbacks. Furthermore, our methods are interpretable: while previous algorithms choose the sketching matrix either at random or by black-box learning, we show that it can be set (or initialized) to clearly interpretable values extracted from the dataset. Our experiments show that our algorithms, either by themselves or in combination with previous methods, achieve significant empirical advantage over previous work, improving training times by up to an order of magnitude toward achieving the same target accuracy.

        ----

        ## [817] Neural-PIL: Neural Pre-Integrated Lighting for Reflectance Decomposition

        **Authors**: *Mark Boss, Varun Jampani, Raphael Braun, Ce Liu, Jonathan T. Barron, Hendrik P. A. Lensch*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/58ae749f25eded36f486bc85feb3f0ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/58ae749f25eded36f486bc85feb3f0ab-Abstract.html)

        **Abstract**:

        Decomposing a scene into its shape, reflectance and illumination is a fundamental problem in computer vision and graphics. Neural approaches such as NeRF have achieved remarkable success in view synthesis, but do not explicitly perform decomposition and instead operate exclusively on radiance (the product of reflectance and illumination). Extensions to NeRF, such as NeRD, can perform decomposition but struggle to accurately recover detailed illumination, thereby significantly limiting realism. We propose a novel reflectance decomposition network that can estimate shape, BRDF, and per-image illumination given a set of object images captured under varying illumination. Our key technique is a novel illumination integration network called Neural-PIL that replaces a costly illumination integral operation in the rendering with a simple network query. In addition, we also learn deep low-dimensional priors on BRDF and illumination representations using novel smooth manifold auto-encoders. Our decompositions can result in considerably better BRDF and light estimates enabling more accurate novel view-synthesis and relighting compared to prior art. Project page: https://markboss.me/publication/2021-neural-pil/

        ----

        ## [818] Asymptotics of the Bootstrap via Stability with Applications to Inference with Model Selection

        **Authors**: *Morgane Austern, Vasilis Syrgkanis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/58b7483ba899e0ce4d97ac5eecf6fa99-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/58b7483ba899e0ce4d97ac5eecf6fa99-Abstract.html)

        **Abstract**:

        One of the most commonly used methods for forming confidence intervals is the empirical bootstrap, which is especially expedient when the limiting distribution of the estimator is unknown. However, despite its ubiquitous role in machine learning, its theoretical properties are still not well understood. Recent developments in probability have provided new tools to study the bootstrap method. However, they have been applied only to specific applications and contexts, and it is unclear whether these techniques are applicable to the understanding of the consistency of the bootstrap in machine learning pipelines. In this paper, we derive general stability conditions under which the empirical bootstrap estimator is consistent and quantify the speed of convergence. Moreover, we propose alternative ways to use the bootstrap method to build confidence intervals with coverage guarantees. Finally, we illustrate the generality and tightness of our results by examples of interest for machine learning including for two-sample kernel tests after kernel selection and the empirical risk of stacked estimators.

        ----

        ## [819] Dynamic influence maximization

        **Authors**: *Binghui Peng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/58ec72df0caca51df569d0b497c33805-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/58ec72df0caca51df569d0b497c33805-Abstract.html)

        **Abstract**:

        We initiate a systematic study on {\em dynamic influence maximization} (DIM). In the DIM problem, one maintains a seed set $S$ of at most $k$ nodes in a dynamically involving social network, with the goal of maximizing the expected influence spread while minimizing the amortized updating cost. We consider two evolution models. In the {\em incremental model}, the social network gets enlarged over time and one only introduces new users and establishes new social links, we design an algorithm that achieves $(1-1/e-\epsilon)$-approximation to the optimal solution and has $k \cdot\mathsf{poly}(\log n, \epsilon^{-1})$ amortized running time, which matches the state-of-art offline algorithm with only poly-logarithmic overhead. In the fully dynamic model, users join in and leave, influence propagation gets strengthened or weakened in real time, we prove that under the Strong Exponential Time Hypothesis (SETH), no algorithm can achieve $2^{-(\log n)^{1-o(1)}}$-approximation unless the amortized running time is $n^{1-o(1)}$.  On the technical side, we exploit novel adaptive sampling approaches that reduce DIM to the dynamic MAX-k coverage problem, and design an efficient $(1-1/e-\epsilon)$-approximation algorithm for it. Our lower bound leverages the recent developed distributed PCP framework.

        ----

        ## [820] Risk Monotonicity in Statistical Learning

        **Authors**: *Zakaria Mhammedi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5907c88df2965e500c98e948dfae20c0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5907c88df2965e500c98e948dfae20c0-Abstract.html)

        **Abstract**:

        Acquisition of data is a difficult task in many applications of machine learning, and it is only natural that one hopes and expects the population risk to decrease (better performance) monotonically with increasing data points. It turns out, somewhat surprisingly, that this is not the case even for the most standard algorithms that minimize the empirical risk. Non-monotonic behavior of the risk and instability in training have manifested and appeared in the popular deep learning paradigm under the description of double descent. These problems highlight the current lack of understanding of learning algorithms and generalization. It is, therefore, crucial to pursue this concern and provide a characterization of such behavior. In this paper, we derive the first consistent and risk-monotonic (in high probability) algorithms for a general statistical learning setting under weak assumptions, consequently answering some questions posed by Viering et. al. 2019 on how to avoid non-monotonic behavior of risk curves. We further show that risk monotonicity need not necessarily come at the price of worse excess risk rates. To achieve this, we derive new empirical Bernstein-like concentration inequalities of independent interest that hold for certain non-i.i.d.~processes such as Martingale Difference Sequences.

        ----

        ## [821] Information is Power: Intrinsic Control via Information Capture

        **Authors**: *Nicholas Rhinehart, Jenny Wang, Glen Berseth, John D. Co-Reyes, Danijar Hafner, Chelsea Finn, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/59112692262234e3fad47fa8eabf03a4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/59112692262234e3fad47fa8eabf03a4-Abstract.html)

        **Abstract**:

        Humans and animals explore their environment and acquire useful skills even in the absence of clear goals, exhibiting intrinsic motivation. The study of intrinsic motivation in artificial agents is concerned with the following question: what is a good general-purpose objective for an agent? We study this question in dynamic partially-observed environments, and argue that a compact and general learning objective is to minimize the entropy of the agent's state visitation estimated using a latent state-space model. This objective induces an agent to both gather information about its environment, corresponding to reducing uncertainty, and to gain control over its environment, corresponding to reducing the unpredictability of future world states. We instantiate this approach as a deep reinforcement learning agent equipped with a deep variational Bayes filter. We find that our agent learns to discover, represent, and exercise control of dynamic objects in a variety of partially-observed environments sensed with visual observations without extrinsic reward.

        ----

        ## [822] Extracting Deformation-Aware Local Features by Learning to Deform

        **Authors**: *Guilherme A. Potje, Renato Martins, Felipe C. Chamone, Erickson R. Nascimento*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5934c1ec0cd31e12bd9084d106bc2e32-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5934c1ec0cd31e12bd9084d106bc2e32-Abstract.html)

        **Abstract**:

        Despite the advances in extracting local features achieved by handcrafted and learning-based descriptors, they are still limited by the lack of invariance to non-rigid transformations. In this paper, we present a new approach to compute features from still images that are robust to non-rigid deformations to circumvent the problem of matching deformable surfaces and objects. Our deformation-aware local descriptor, named DEAL, leverages a polar sampling and a spatial transformer warping to provide invariance to rotation, scale, and image deformations. We train the model architecture end-to-end by applying isometric non-rigid deformations to objects in a simulated environment as guidance to provide highly discriminative local features. The experiments show that our method outperforms state-of-the-art handcrafted, learning-based image, and RGB-D descriptors in different datasets with both real and realistic synthetic deformable objects in still images. The source code and trained model of the descriptor are publicly available at https://www.verlab.dcc.ufmg.br/descriptors/neurips2021.

        ----

        ## [823] Object-Centric Representation Learning with Generative Spatial-Temporal Factorization

        **Authors**: *Nanbo Li, Muhammad Ahmed Raza, Wenbin Hu, Zhaole Sun, Robert B. Fisher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/593906af0d138e69f49d251d3e7cbed0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/593906af0d138e69f49d251d3e7cbed0-Abstract.html)

        **Abstract**:

        Learning object-centric scene representations is essential for attaining structural understanding and abstraction of complex scenes. Yet, as current approaches for unsupervised object-centric representation learning are built upon either a stationary observer assumption or a static scene assumption, they often: i) suffer single-view spatial ambiguities, or ii) infer incorrectly or inaccurately object representations from dynamic scenes. To address this, we propose Dynamics-aware Multi-Object Network (DyMON), a method that broadens the scope of multi-view object-centric representation learning to dynamic scenes. We train DyMON on multi-view-dynamic-scene data and show that DyMON learns---without supervision---to factorize the entangled effects of observer motions and scene object dynamics from a sequence of observations, and constructs scene object spatial representations suitable for rendering at arbitrary times (querying across time) and from arbitrary viewpoints (querying across space). We also show that the factorized scene representations (w.r.t. objects) support querying about a single object by space and time independently.

        ----

        ## [824] Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization

        **Authors**: *Zhenghao Peng, Quanyi Li, Ka-Ming Hui, Chunxiao Liu, Bolei Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/594ca7adb3277c51a998252e2d4c906e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/594ca7adb3277c51a998252e2d4c906e-Abstract.html)

        **Abstract**:

        Self-Driven Particles (SDP) describe a category of multi-agent systems common in everyday life, such as flocking birds and traffic flows. In a SDP system, each agent pursues its own goal and constantly changes its cooperative or competitive behaviors with its nearby agents. Manually designing the controllers for such SDP system is time-consuming, while the resulting emergent behaviors are often not realistic nor generalizable. Thus the realistic simulation of SDP systems remains challenging. Reinforcement learning provides an appealing alternative for automating the development of the controller for SDP. However, previous multi-agent reinforcement learning (MARL) methods define the agents to be teammates or enemies before hand, which fail to capture the essence of SDP where the role of each agent varies to be cooperative or competitive even within one episode. To simulate SDP with MARL, a key challenge is to coordinate agents' behaviors while still maximizing individual objectives. Taking traffic simulation as the testing bed, in this work we develop a novel MARL method called Coordinated Policy Optimization (CoPO), which incorporates social psychology principle to learn neural controller for SDP. Experiments show that the proposed method can achieve superior performance compared to MARL baselines in various metrics. Noticeably the trained vehicles exhibit complex and diverse social behaviors that improve performance and safety of the population as a whole. Demo video and source code are available at: https://decisionforce.github.io/CoPO/

        ----

        ## [825] Gradient-based Hyperparameter Optimization Over Long Horizons

        **Authors**: *Paul Micaelli, Amos J. Storkey*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/596dedf4498e258e4bdc9fd70df9a859-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/596dedf4498e258e4bdc9fd70df9a859-Abstract.html)

        **Abstract**:

        Gradient-based hyperparameter optimization has earned a widespread popularity in the context of few-shot meta-learning, but remains broadly impractical for tasks with long horizons (many gradient steps), due to memory scaling and gradient degradation issues. A common workaround is to learn hyperparameters online, but this introduces greediness which comes with a significant performance drop. We propose forward-mode differentiation with sharing (FDS), a simple and efficient algorithm which tackles memory scaling issues with forward-mode differentiation, and gradient degradation issues by sharing hyperparameters that are contiguous in time. We provide theoretical guarantees about the noise reduction properties of our algorithm, and demonstrate its efficiency empirically by differentiating through $\sim 10^4$ gradient steps of unrolled optimization. We consider large hyperparameter search ranges on CIFAR-10 where we significantly outperform greedy gradient-based alternatives, while achieving $\times 20$ speedups compared to the state-of-the-art black-box methods.

        ----

        ## [826] Stochastic Bias-Reduced Gradient Methods

        **Authors**: *Hilal Asi, Yair Carmon, Arun Jambulapati, Yujia Jin, Aaron Sidford*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/597c7b407a02cc0a92167e7a371eca25-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/597c7b407a02cc0a92167e7a371eca25-Abstract.html)

        **Abstract**:

        We develop a new primitive for stochastic optimization: a low-bias, low-cost  estimator of the minimizer $x_\star$ of any Lipschitz strongly-convex function $f$. In particular, we use a multilevel Monte-Carlo approach due to Blanchet and Glynn to turn any optimal stochastic gradient method into an estimator of $x_\star$ with bias $\delta$, variance $O(\log(1/\delta))$, and an expected sampling cost of $O(\log(1/\delta))$ stochastic gradient evaluations. As an immediate consequence, we obtain cheap and nearly unbiased gradient estimators for the Moreau envelope of any Lipschitz convex function. We demonstrate the potential of our estimator through four applications. First, we develop a method for minimizing the maximum of $N$ functions, improving on recent results and matching a lower bound up to logarithmic factors. Second and third, we recover state-of-the-art rates for projection-efficient and gradient-efficient optimization using simple algorithms with a transparent analysis. Finally, we show that an improved version of our estimator would yield a nearly linear-time, optimal-utility, differentially-private non-smooth stochastic optimization method.

        ----

        ## [827] The Causal-Neural Connection: Expressiveness, Learnability, and Inference

        **Authors**: *Kevin Xia, Kai-Zhan Lee, Yoshua Bengio, Elias Bareinboim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5989add1703e4b0480f75e2390739f34-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5989add1703e4b0480f75e2390739f34-Abstract.html)

        **Abstract**:

        One of the central elements of any causal inference is an object called structural causal model (SCM), which represents a collection of mechanisms and exogenous sources of random variation of the system under investigation (Pearl, 2000). An important property of many kinds of neural networks is universal approximability: the ability to approximate any function to arbitrary precision. Given this property, one may be tempted to surmise that a collection of neural nets is capable of learning any SCM by training on data generated by that SCM. In this paper, we show this is not the case by disentangling the notions of expressivity and learnability. Specifically, we show that the causal hierarchy theorem (Thm. 1, Bareinboim et al., 2020), which describes the limits of what can be learned from data, still holds for neural models. For instance, an arbitrarily complex and expressive neural net is unable to predict the effects of interventions given observational data alone. Given this result, we introduce a special type of SCM called a neural causal model (NCM), and formalize a new type of inductive bias to encode structural constraints necessary for performing causal inferences. Building on this new class of models, we focus on solving two canonical tasks found in the literature known as causal  identification and estimation. Leveraging the neural toolbox, we develop an algorithm that is both sufficient and necessary to determine whether a causal effect can be learned from data (i.e., causal identifiability); it then estimates the effect whenever identifiability holds (causal estimation). Simulations corroborate the proposed approach.

        ----

        ## [828] Validation Free and Replication Robust Volume-based Data Valuation

        **Authors**: *Xinyi Xu, Zhaoxuan Wu, Chuan Sheng Foo, Bryan Kian Hsiang Low*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/59a3adea76fadcb6dd9e54c96fc155d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/59a3adea76fadcb6dd9e54c96fc155d1-Abstract.html)

        **Abstract**:

        Data valuation arises as a non-trivial challenge in real-world use cases such as collaborative machine learning, federated learning, trusted data sharing, data marketplaces. The value of data is often associated with the learning performance (e.g., validation accuracy) of a model trained on the data, which introduces a close coupling between data valuation and validation.  However, a validation set may notbe available in practice and it can be challenging for the data providers to reach an agreement on the choice of the validation set. Another practical issue is that of data replication: Given the value of some data points, a dishonest data provider may replicate these data points to exploit the valuation for a larger reward/payment. We observe that the diversity of the data points is an inherent property of a dataset that is independent of validation. We formalize diversity via the volume of the data matrix (i.e., determinant of its left Gram), which allows us to establish a formal connection between the diversity of data and learning performance without requiring validation. Furthermore, we propose a robust volume measure with a theoretical guarantee on the replication robustness by following the intuition that copying the same data points does not increase the diversity of data.  We perform extensive experiments to demonstrate its consistency in valuation and practical advantages over existing baselines and show that our method is model- and task-agnostic and can be flexibly adapted to handle various neural networks.

        ----

        ## [829] Implicit Finite-Horizon Approximation and Efficient Optimal Algorithms for Stochastic Shortest Path

        **Authors**: *Liyu Chen, Mehdi Jafarnia-Jahromi, Rahul Jain, Haipeng Luo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/59b1deff341edb0b76ace57820cef237-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/59b1deff341edb0b76ace57820cef237-Abstract.html)

        **Abstract**:

        We introduce a generic template for developing regret minimization algorithms in the Stochastic Shortest Path (SSP) model, which achieves minimax optimal regret as long as certain properties are ensured. The key of our analysis is a new technique called implicit finite-horizon approximation, which approximates the SSP model by a finite-horizon counterpart only in the analysis without explicit implementation. Using this template, we develop two new algorithms: the first one is model-free (the first in the literature to our knowledge) and minimax optimal under strictly positive costs; the second one is model-based and minimax optimal even with zero-cost state-action pairs, matching the best existing result from [Tarbouriech et al., 2021b]. Importantly, both algorithms admit highly sparse updates, making them  computationally more efficient than all existing algorithms. Moreover, both can be made completely parameter-free.

        ----

        ## [830] A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks

        **Authors**: *Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Guha Thakurta*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a499f6e26313e19bd4049009bbed5bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a499f6e26313e19bd4049009bbed5bd-Abstract.html)

        **Abstract**:

        Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Most of these attacks require the full knowledge of training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that \emph{full information} adversaries (that craft poisoning examples based on the rest of the training data) are provably much more devastating compared to the optimal attacker that is \emph{oblivious} to the training set yet has access to the distribution of the data.  Our separation result shows that the two settings of data-aware and data-oblivious are fundamentally different and we cannot hope to achieve the same attack or defense results in these scenarios.

        ----

        ## [831] Deep Learning Through the Lens of Example Difficulty

        **Authors**: *Robert J. N. Baldock, Hartmut Maennel, Behnam Neyshabur*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a4b25aaed25c2ee1b74de72dc03c14e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a4b25aaed25c2ee1b74de72dc03c14e-Abstract.html)

        **Abstract**:

        Existing work on understanding deep learning often employs measures that compress all data-dependent information into a few numbers. In this work, we adopt a perspective based on the role of individual examples. We introduce a measure of the computational difficulty of making a prediction for a given input: the (effective) prediction depth. Our extensive investigation reveals surprising yet simple relationships between the prediction depth of a given input and the modelâ€™s uncertainty, confidence, accuracy and speed of learning for that data point. We further categorize difficult examples into three interpretable groups, demonstrate how these groups are processed differently inside deep models and showcase how this understanding allows us to improve prediction accuracy. Insights from our study lead to a coherent view of a number of separately reported phenomena in the literature: early layers generalize while later layers memorize; early layers converge faster and networks learn easy data and simple functions first.

        ----

        ## [832] R-Drop: Regularized Dropout for Neural Networks

        **Authors**: *Xiaobo Liang, Lijun Wu, Juntao Li, Yue Wang, Qi Meng, Tao Qin, Wei Chen, Min Zhang, Tie-Yan Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html)

        **Abstract**:

        Dropout is a powerful and widely used technique to regularize the training of deep neural networks. Though effective and performing well, the randomness introduced by dropout causes unnegligible inconsistency between training and inference. In this paper, we introduce a simple consistency training strategy to regularize dropout, namely R-Drop, which forces the output distributions of different sub models generated by dropout to be consistent with each other. Specifically, for each training sample, R-Drop minimizes the bidirectional KL-divergence between the output distributions of two sub models sampled by dropout. Theoretical analysis reveals that R-Drop reduces the above inconsistency. Experiments on $\bf{5}$ widely used deep learning tasks ($\bf{18}$ datasets in total), including neural machine translation, abstractive summarization, language understanding, language modeling, and image classification, show that R-Drop is universally effective. In particular, it yields substantial improvements when applied to fine-tune large-scale pre-trained models, e.g., ViT, RoBERTa-large, and BART, and achieves state-of-the-art (SOTA) performances with the vanilla Transformer model on WMT14 English$\to$German translation ($\bf{30.91}$ BLEU) and WMT14 English$\to$French translation ($\bf{43.95}$ BLEU), even surpassing models trained with extra large-scale data and expert-designed advanced variants of Transformer models. Our code is available at GitHub\footnote{\url{https://github.com/dropreg/R-Drop}}.

        ----

        ## [833] Diversity Enhanced Active Learning with Strictly Proper Scoring Rules

        **Authors**: *Wei Tan, Lan Du, Wray L. Buntine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a7b238ba0f6502e5d6be14424b20ded-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a7b238ba0f6502e5d6be14424b20ded-Abstract.html)

        **Abstract**:

        We study acquisition functions for active learning (AL) for text classification. The Expected Loss Reduction (ELR) method focuses on a Bayesian estimate of the reduction in classification error, recently updated with Mean Objective Cost of Uncertainty (MOCU).  We convert the ELR framework to estimate the increase in (strictly proper) scores like log probability or negative mean square error, which we call Bayesian Estimate of Mean Proper Scores (BEMPS). We also prove convergence results borrowing techniques used with MOCU. In order to allow better experimentation with the new acquisition functions,  we develop a complementary batch AL algorithm, which encourages diversity in the vector of expected changes in scores for unlabelled data. To allow high performance text classifiers, we combine ensembling and dynamic validation set construction on pretrained language models.  Extensive experimental evaluation then explores how these different acquisition functions perform. The results show that the use of mean square error and log probability with BEMPS yields robust acquisition functions, which consistently outperform the others tested.

        ----

        ## [834] SSUL: Semantic Segmentation with Unknown Label for Exemplar-based Class-Incremental Learning

        **Authors**: *Sungmin Cha, Beomyoung Kim, Youngjoon Yoo, Taesup Moon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a9542c773018268fc6271f7afeea969-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a9542c773018268fc6271f7afeea969-Abstract.html)

        **Abstract**:

        We consider a class-incremental semantic segmentation (CISS) problem. While some recently proposed algorithms utilized variants of knowledge distillation (KD) technique to tackle the problem, they only partially addressed the key additional challenges in CISS that causes the catastrophic forgetting; \textit{i.e.}, the semantic drift of the background class and multi-label prediction issue. To better address these challenges, we propose a new method, dubbed as SSUL-M (Semantic Segmentation with Unknown Label with Memory), by carefully combining several techniques tailored for semantic segmentation. More specifically, we make three main contributions; (1) modeling \textit{unknown} class within the background class to help learning future classes (help plasticity), (2) \textit{freezing} backbone network and past classifiers with binary cross-entropy loss and pseudo-labeling to overcome catastrophic forgetting (help stability), and (3) utilizing \textit{tiny exemplar memory} for the first time in CISS to improve \textit{both} plasticity and stability. As a result, we show our method achieves significantly better performance than the recent state-of-the-art baselines on the standard benchmark datasets. Furthermore, we justify our contributions with thorough and extensive ablation analyses and discuss different natures of the CISS problem compared to the standard class-incremental learning for classification. The official code is available at https://github.com/clovaai/SSUL.

        ----

        ## [835] Lower and Upper Bounds on the Pseudo-Dimension of Tensor Network Models

        **Authors**: *Behnoush Khavari, Guillaume Rabusseau*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5a9d8bf5b7a4b35f3110dde8673bdda2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5a9d8bf5b7a4b35f3110dde8673bdda2-Abstract.html)

        **Abstract**:

        Tensor network methods have been a key ingredient of advances in condensed matter physics and have recently sparked interest in the machine learning community for their ability to compactly represent very high-dimensional objects. Tensor network methods can for example be used to efficiently learn linear models in exponentially large feature spaces [Stoudenmire and Schwab, 2016]. In this work, we derive upper and lower bounds on the VC dimension and pseudo-dimension of a large class of tensor network models for classification, regression and completion. Our upper bounds hold for linear models parameterized by arbitrary tensor network structures, and we derive lower bounds for common  tensor decomposition models~(CP, Tensor Train, Tensor Ring and Tucker) showing the tightness of our general upper bound. These results are used to derive a generalization bound which can be applied to classification with low rank matrices as well as linear classifiers based on any of the commonly used tensor decomposition models. As a corollary of our results, we obtain a bound on the VC dimension of the matrix product state classifier introduced in [Stoudenmire and Schwab, 2016] as a function of the so-called bond dimension~(i.e. tensor train rank), which  answers an open problem listed by Cirac, Garre-Rubio and Pérez-García in [Cirac et al., 2019].

        ----

        ## [836] What Makes Multi-Modal Learning Better than Single (Provably)

        **Authors**: *Yu Huang, Chenzhuang Du, Zihui Xue, Xuanyao Chen, Hang Zhao, Longbo Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5aa3405a3f865c10f420a4a7b55cbff3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5aa3405a3f865c10f420a4a7b55cbff3-Abstract.html)

        **Abstract**:

        The world provides us with data of multiple modalities. Intuitively, models fusing data from different modalities outperform their uni-modal counterparts, since more information is aggregated. Recently, joining the success of deep learning, there is an influential line of work on deep multi-modal learning, which has remarkable empirical results on various applications. However, theoretical justifications in this field are notably lacking.                        Can multi-modal learning provably perform better than uni-modal?In this paper, we answer this question under a most popular multi-modal fusion framework, which firstly encodes features from different modalities into a common latent space and seamlessly maps the latent representations into the task space. We prove that learning with multiple modalities achieves a  smaller population risk than only using its subset of modalities. The main intuition is that the former has a more accurate estimate of the latent space representation. To the best of our knowledge, this is the first theoretical treatment to capture important qualitative phenomena observed in real multi-modal applications from the generalization perspective. Combining with experiment results, we show that multi-modal learning does possess an appealing formal guarantee.

        ----

        ## [837] Quantifying and Improving Transferability in Domain Generalization

        **Authors**: *Guojun Zhang, Han Zhao, Yaoliang Yu, Pascal Poupart*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5adaacd4531b78ff8b5cedfe3f4d5212-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5adaacd4531b78ff8b5cedfe3f4d5212-Abstract.html)

        **Abstract**:

        Out-of-distribution generalization is one of the key challenges when transferring a model from the lab to the real world.  Existing efforts mostly focus on building invariant features among source and target domains. Based on invariant features, a high-performing classifier on source domains could hopefully behave equally well on a target domain. In other words, we hope the invariant features to be \emph{transferable}. However, in practice, there are no perfectly transferable features, and some algorithms seem to learn ``more transferable'' features than others. How can we understand and quantify such \emph{transferability}? In this paper, we formally define transferability that one can quantify and compute in domain generalization. We point out the difference and connection with common discrepancy measures between domains, such as total variation and Wasserstein distance. We then prove that our transferability can be estimated with enough samples and give a new upper bound for the target error based on our transferability. Empirically, we evaluate the transferability of the feature embeddings learned by existing algorithms for domain generalization. Surprisingly, we find that many algorithms are not quite learning transferable features, although few could still survive. In light of this, we propose a new algorithm for learning transferable features and test it over various benchmark datasets, including RotatedMNIST, PACS, Office-Home and WILDS-FMoW. Experimental results show that the proposed algorithm achieves consistent improvement over many state-of-the-art algorithms, corroborating our theoretical findings.

        ----

        ## [838] Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification

        **Authors**: *Youngseog Chung, Willie Neiswanger, Ian Char, Jeff Schneider*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5b168fdba5ee5ea262cc2d4c0b457697-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5b168fdba5ee5ea262cc2d4c0b457697-Abstract.html)

        **Abstract**:

        Among the many ways of quantifying uncertainty in a regression setting, specifying the full quantile function is attractive, as quantiles are amenable to interpretation and evaluation. A model that predicts the true conditional quantiles for each input, at all quantile levels, presents a correct and efficient representation of the underlying uncertainty. To achieve this, many current quantile-based methods focus on optimizing the pinball loss. However, this loss restricts the scope of applicable regression models, limits the ability to target many desirable properties (e.g. calibration, sharpness, centered intervals), and may produce poor conditional quantiles. In this work, we develop new quantile methods that address these shortcomings. In particular, we propose methods that can apply to any class of regression model, select an explicit balance between calibration and sharpness, optimize for calibration of centered intervals, and produce more accurate conditional quantiles. We provide a thorough experimental evaluation of our methods, which includes a high dimensional uncertainty quantification task in nuclear fusion.

        ----

        ## [839] Dynamic Inference with Neural Interpreters

        **Authors**: *Nasim Rahaman, Muhammad Waleed Gondal, Shruti Joshi, Peter V. Gehler, Yoshua Bengio, Francesco Locatello, Bernhard Schölkopf*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5b4e9aa703d0bfa11041debaa2d1b633-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5b4e9aa703d0bfa11041debaa2d1b633-Abstract.html)

        **Abstract**:

        Modern neural network architectures can leverage large amounts of data to generalize well within the training distribution. However, they are less capable of systematic generalization to data drawn from unseen but related distributions, a feat that is hypothesized to require compositional reasoning and reuse of knowledge. In this work, we present Neural Interpreters, an architecture that factorizes inference in a self-attention network as a system of modules, which we call functions. Inputs to the model are routed through a sequence of functions in a way that is end-to-end learned. The proposed architecture can flexibly compose computation along width and depth, and lends itself well to capacity extension after training. To demonstrate the versatility of Neural Interpreters, we evaluate it in two distinct settings: image classification and visual abstract reasoning on Raven Progressive Matrices. In the former, we show that Neural Interpreters perform on par with the vision transformer using fewer parameters, while being transferrable to a new task in a sample efficient manner. In the latter, we find that Neural Interpreters are competitive with respect to the state-of-the-art in terms of systematic generalization.

        ----

        ## [840] Leveraging Recursive Gumbel-Max Trick for Approximate Inference in Combinatorial Spaces

        **Authors**: *Kirill Struminsky, Artyom Gadetsky, Denis Rakitin, Danil Karpushkin, Dmitry P. Vetrov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5b658d2a925565f0755e035597f8d22f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5b658d2a925565f0755e035597f8d22f-Abstract.html)

        **Abstract**:

        Structured latent variables allow incorporating meaningful prior knowledge into deep learning models. However, learning with such variables remains challenging because of their discrete nature. Nowadays, the standard learning approach is to define a latent variable as a perturbed algorithm output and to use a differentiable surrogate for training. In general, the surrogate puts additional constraints on the model and inevitably leads to biased gradients. To alleviate these shortcomings, we extend the Gumbel-Max trick to define distributions over structured domains. We avoid the differentiable surrogates by leveraging the score function estimators for optimization. In particular, we highlight a family of recursive algorithms with a common feature we call stochastic invariant. The feature allows us to construct reliable gradient estimates and control variates without additional constraints on the model. In our experiments, we consider various structured latent variable models and achieve results competitive with relaxation-based counterparts.

        ----

        ## [841] Hamiltonian Dynamics with Non-Newtonian Momentum for Rapid Sampling

        **Authors**: *Greg Ver Steeg, Aram Galstyan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5b970a1d9be0fd100063fd6cd688b73e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5b970a1d9be0fd100063fd6cd688b73e-Abstract.html)

        **Abstract**:

        Sampling from an unnormalized probability distribution is a fundamental problem in machine learning with applications including Bayesian modeling, latent factor inference, and energy-based model training. After decades of research, variations of MCMC remain the default approach to sampling despite slow convergence. Auxiliary neural models can learn to speed up MCMC, but the overhead for training the extra model can be prohibitive. We propose a fundamentally different approach to this problem via a new Hamiltonian dynamics with a non-Newtonian momentum. In contrast to MCMC approaches like Hamiltonian Monte Carlo, no stochastic step is required. Instead, the proposed deterministic dynamics in an extended state space exactly sample the target distribution, specified by an energy function, under an assumption of ergodicity. Alternatively, the dynamics can be interpreted as a normalizing flow that samples a specified energy model without training. The proposed Energy Sampling Hamiltonian (ESH) dynamics have a simple form that can be solved with existing ODE solvers, but we derive a specialized solver that exhibits much better performance. ESH dynamics converge faster than their MCMC competitors enabling faster, more stable training of neural network energy models.

        ----

        ## [842] Dynamic Normalization and Relay for Video Action Recognition

        **Authors**: *Dongqi Cai, Anbang Yao, Yurong Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5bd529d5b07b647a8863cf71e98d651a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5bd529d5b07b647a8863cf71e98d651a-Abstract.html)

        **Abstract**:

        Convolutional Neural Networks (CNNs) have been the dominant model for video action recognition. Due to the huge memory and compute demand, popular action recognition networks need to be trained with small batch sizes, which makes learning discriminative spatial-temporal representations for videos become a challenging problem. In this paper, we present Dynamic Normalization and Relay (DNR), an improved normalization design, to augment the spatial-temporal representation learning of any deep action recognition model, adapting to small batch size training settings. We observe that state-of-the-art action recognition networks usually apply the same normalization parameters to all video data, and ignore the dependencies of the estimated normalization parameters between neighboring frames (at the same layer) and between neighboring layers (with all frames of a video clip). Inspired by this, DNR introduces two dynamic normalization relay modules to explore the potentials of cross-temporal and cross-layer feature distribution dependencies for estimating accurate layer-wise normalization parameters. These two DNR modules are instantiated as a light-weight recurrent structure conditioned on the current input features, and the normalization parameters estimated from the neighboring frames based features at the same layer or from the whole video clip based features at the preceding layers. We first plug DNR into prevailing 2D CNN backbones and test its performance on public action recognition datasets including Kinetics and Something-Something. Experimental results show that DNR brings large performance improvements to the baselines, achieving over 4.4% absolute margins in top-1 accuracy without training bells and whistles. More experiments on 3D backbones and several latest 2D spatial-temporal networks further validate its effectiveness. Code will be available at https://github.com/caidonkey/dnr.

        ----

        ## [843] Robust Visual Reasoning via Language Guided Neural Module Networks

        **Authors**: *Arjun R. Akula, Varun Jampani, Soravit Changpinyo, Song-Chun Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5bd53571b97884635d13910db49626bc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5bd53571b97884635d13910db49626bc-Abstract.html)

        **Abstract**:

        Neural module networks (NMN) are a popular approach for solving multi-modal tasks such as visual question answering (VQA) and visual referring expression recognition (REF). A key limitation in prior implementations of NMN is that the neural modules do not effectively capture the association between the visual input and the relevant neighbourhood context of the textual input. This limits their generalizability. For instance, NMN fail to understand new concepts such as “yellow sphere to the left" even when it is a combination of known concepts from train data: “blue sphere", “yellow cube", and “metallic cube to the left". In this paper, we address this limitation by introducing a language-guided adaptive convolution layer (LG-Conv) into NMN, in which the filter weights of convolutions are explicitly multiplied with a spatially varying language-guided kernel. Our model allows the neural module to adaptively co-attend over potential objects of interest from the visual and textual inputs. Extensive experiments on VQA and REF tasks demonstrate the effectiveness of our approach. Additionally, we propose a new challenging out-of-distribution test split for REF task, which we call C3-Ref+, for explicitly evaluating the NMN’s ability to generalize well to adversarial perturbations and unseen combinations of known concepts. Experiments on C3-Ref+ further demonstrate the generalization capabilities of our approach.

        ----

        ## [844] True Few-Shot Learning with Language Models

        **Authors**: *Ethan Perez, Douwe Kiela, Kyunghyun Cho*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c04925674920eb58467fb52ce4ef728-Abstract.html)

        **Abstract**:

        Pretrained language models (LMs) perform well on many tasks even when learning from a few examples, but prior work uses many held-out examples to tune various aspects of learning, such as hyperparameters, training objectives, and natural language templates ("prompts"). Here, we evaluate the few-shot ability of LMs when such held-out examples are unavailable, a setting we call true few-shot learning. We test two model selection criteria, cross-validation and minimum description length, for choosing LM prompts and hyperparameters in the true few-shot setting. On average, both marginally outperform random selection and greatly underperform selection based on held-out examples. Moreover, selection criteria often prefer models that perform significantly worse than randomly-selected ones. We find similar results even when taking into account our uncertainty in a model's true performance during selection, as well as when varying the amount of computation and number of examples used for selection. Overall, our findings suggest that prior work significantly overestimated the true few-shot ability of LMs given the difficulty of few-shot model selection.

        ----

        ## [845] Selective Sampling for Online Best-arm Identification

        **Authors**: *Romain Camilleri, Zhihan Xiong, Maryam Fazel, Lalit Jain, Kevin G. Jamieson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c333c4ffd55c7a3576e6a614d81af82-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c333c4ffd55c7a3576e6a614d81af82-Abstract.html)

        **Abstract**:

        This work considers the problem of selective-sampling for best-arm identification. Given a set of potential options $\mathcal{Z}\subset\mathbb{R}^d$, a learner aims to compute with probability greater than $1-\delta$, $\arg\max_{z\in \mathcal{Z}} z^{\top}\theta_{\ast}$ where $\theta_{\ast}$ is unknown. At each time step, a potential measurement $x_t\in \mathcal{X}\subset\mathbb{R}^d$ is drawn IID and the learner can either choose to take the measurement, in which case they observe a noisy measurement of $x^{\top}\theta_{\ast}$, or to abstain from taking the measurement and wait for a potentially more informative point to arrive in the stream. Hence the learner faces a fundamental trade-off between the number of labeled samples they take and when they have collected enough evidence to declare the best arm and stop sampling. The main results of this work precisely characterize this trade-off between labeled samples and stopping time and provide an algorithm that nearly-optimally achieves the minimal label complexity given a desired stopping time. In addition, we show that the optimal decision rule has a simple geometric form based on deciding whether a point is in an ellipse or not. Finally, our framework is general enough to capture binary classification improving upon previous works.

        ----

        ## [846] Multi-task Learning of Order-Consistent Causal Graphs

        **Authors**: *Xinshi Chen, Haoran Sun, Caleb Ellington, Eric P. Xing, Le Song*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c3a3b139a11689e0bc55abd95e20e39-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c3a3b139a11689e0bc55abd95e20e39-Abstract.html)

        **Abstract**:

        We consider the problem of discovering $K$ related Gaussian directed acyclic graphs (DAGs), where the involved graph structures share a consistent causal order and sparse unions of supports. Under the multi-task learning setting, we propose a $l_1/l_2$-regularized maximum likelihood estimator (MLE) for learning $K$ linear structural equation models. We theoretically show that the joint estimator, by leveraging data across related tasks, can achieve a better sample complexity for recovering the causal order (or topological order) than separate estimations. Moreover, the joint estimator is able to recover non-identifiable DAGs, by estimating them together with some identifiable DAGs.  Lastly, our analysis also shows the consistency of union support recovery of the structures. To allow practical implementation, we design a continuous optimization problem whose optimizer is the same as the joint estimator and can be approximated efficiently by an iterative algorithm. We validate the theoretical analysis and the effectiveness of the joint estimator in experiments.

        ----

        ## [847] Learning to Iteratively Solve Routing Problems with Dual-Aspect Collaborative Transformer

        **Authors**: *Yining Ma, Jingwen Li, Zhiguang Cao, Wen Song, Le Zhang, Zhenghua Chen, Jing Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c53292c032b6cb8510041c54274e65f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c53292c032b6cb8510041c54274e65f-Abstract.html)

        **Abstract**:

        Recently, Transformer has become a prevailing deep architecture for solving vehicle routing problems (VRPs). However, it is less effective in learning improvement models for VRP because its positional encoding (PE) method is not suitable in representing VRP solutions. This paper presents a novel Dual-Aspect Collaborative Transformer (DACT) to learn embeddings for the node and positional features separately, instead of fusing them together as done in existing ones, so as to avoid potential noises and incompatible correlations. Moreover, the positional features are embedded through a novel cyclic positional encoding (CPE) method to allow Transformer to effectively capture the circularity and symmetry of VRP solutions (i.e., cyclic sequences). We train DACT using Proximal Policy Optimization and design a curriculum learning strategy for better sample efficiency. We apply DACT to solve the traveling salesman problem (TSP) and capacitated vehicle routing problem (CVRP). Results show that our DACT outperforms existing Transformer based improvement models, and exhibits much better generalization performance across different problem sizes on synthetic and benchmark instances, respectively.

        ----

        ## [848] Learning interaction rules from multi-animal trajectories via augmented behavioral models

        **Authors**: *Keisuke Fujii, Naoya Takeishi, Kazushi Tsutsui, Emyo Fujioka, Nozomi Nishiumi, Ryoya Tanaka, Mika Fukushiro, Kaoru Ide, Hiroyoshi Kohno, Ken Yoda, Susumu Takahashi, Shizuko Hiryu, Yoshinobu Kawahara*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c572eca050594c7bc3c36e7e8ab9550-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c572eca050594c7bc3c36e7e8ab9550-Abstract.html)

        **Abstract**:

        Extracting the interaction rules of biological agents from movement sequences pose challenges in various domains. Granger causality is a practical framework for analyzing the interactions from observed time-series data; however, this framework ignores the structures and assumptions of the generative process in animal behaviors, which may lead to interpretational problems and sometimes erroneous assessments of causality. In this paper, we propose a new framework for learning Granger causality from multi-animal trajectories via augmented theory-based behavioral models with interpretable data-driven models. We adopt an approach for augmenting incomplete multi-agent behavioral models described by time-varying dynamical systems with neural networks. For efficient and interpretable learning, our model leverages theory-based architectures separating navigation and motion processes, and the theory-guided regularization for reliable behavioral modeling. This can provide interpretable signs of Granger-causal effects over time, i.e., when specific others cause the approach or separation. In experiments using synthetic datasets, our method achieved better performance than various baselines. We then analyzed multi-animal datasets of mice, flies, birds, and bats, which verified our method and obtained novel biological insights.

        ----

        ## [849] Differentiable Synthesis of Program Architectures

        **Authors**: *Guofeng Cui, He Zhu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c5a93a042235058b1ef7b0ac1e11b67-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c5a93a042235058b1ef7b0ac1e11b67-Abstract.html)

        **Abstract**:

        Differentiable programs have recently attracted much interest due to their interpretability, compositionality, and their efficiency to leverage differentiable training. However, synthesizing differentiable programs requires optimizing over a combinatorial, rapidly exploded space of program architectures. Despite the development of effective pruning heuristics, previous works essentially enumerate the discrete search space of program architectures, which is inefficient. We propose to encode program architecture search as learning the probability distribution over all possible program derivations induced by a context-free grammar. This allows the search algorithm to efficiently prune away unlikely program derivations to synthesize optimal program architectures. To this end, an efficient gradient-descent based method is developed to conduct program architecture search in a continuous relaxation of the discrete space of grammar rules. Experiment results on four sequence classification tasks demonstrate that our program synthesizer excels in discovering program architectures that lead to differentiable programs with higher F1 scores, while being more efficient than state-of-the-art program synthesis methods.

        ----

        ## [850] Make Sure You're Unsure: A Framework for Verifying Probabilistic Specifications

        **Authors**: *Leonard Berrada, Sumanth Dathathri, Krishnamurthy Dvijotham, Robert Stanforth, Rudy Bunel, Jonathan Uesato, Sven Gowal, M. Pawan Kumar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c5bc7df3d37b2a7ea29e1b47b2bd4ab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c5bc7df3d37b2a7ea29e1b47b2bd4ab-Abstract.html)

        **Abstract**:

        Most real world applications require dealing with stochasticity like sensor noise or predictive uncertainty, where formal specifications of desired behavior are inherently probabilistic.  Despite the promise of formal verification in ensuring the reliability of neural networks, progress in the direction of probabilistic specifications has been limited. In this direction, we first introduce a general formulation of probabilistic specifications for neural networks, which captures both probabilistic networks (e.g., Bayesian neural networks, MC-Dropout networks) and uncertain inputs (distributions over inputs arising from sensor noise or other perturbations). We then propose a general technique to verify such specifications by generalizing the notion of Lagrangian duality, replacing standard Lagrangian multipliers with "functional multipliers" that can be arbitrary functions of the activations at a given layer. We show that an optimal choice of functional multipliers leads to exact verification (i.e.,  sound and complete verification),  and for specific forms of multipliers, we develop tractable practical verification algorithms. We empirically validate our algorithms by applying them to Bayesian Neural Networks (BNNs) and MC Dropout Networks, and certifying properties such as adversarial robustness and robust detection of out-of-distribution (OOD) data. On these tasks we are able to provide significantly stronger guarantees when compared to prior work -- for instance, for a VGG-64 MC-Dropout CNN trained on CIFAR-10 in a verification-agnostic manner,  we improve the certified AUC (a verified lower bound on the true AUC) for robust OOD detection (on CIFAR-100) from $0 \% \rightarrow 29\%$. Similarly, for a BNN trained on MNIST, we improve on the $\ell_\infty$ robust accuracy from $60.2 \% \rightarrow 74.6\%$. Further, on a novel specification -- distributionally robust OOD detection -- we improve on the certified AUC from $5\% \rightarrow 23\%$.

        ----

        ## [851] Oracle-Efficient Regret Minimization in Factored MDPs with Unknown Structure

        **Authors**: *Aviv Rosenberg, Yishay Mansour*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5c936263f3428a40227908d5a3847c0b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5c936263f3428a40227908d5a3847c0b-Abstract.html)

        **Abstract**:

        We study regret minimization in non-episodic factored Markov decision processes (FMDPs), where all existing algorithms make the strong assumption that the factored structure of the FMDP is known to the learner in advance. In this paper, we provide the first algorithm that learns the structure of the FMDP while minimizing the regret. Our algorithm is based on the optimism in face of uncertainty principle, combined with a simple statistical method for structure learning, and can be implemented efficiently given oracle-access to an FMDP planner. Moreover, we give a variant of our algorithm that remains efficient even when the oracle is limited to non-factored actions, which is the case with almost all existing approximate planners. Finally, we leverage our techniques to prove a novel lower bound for the known structure case, closing the gap to the regret bound of Chen et al. [2021].

        ----

        ## [852] Linear-Time Probabilistic Solution of Boundary Value Problems

        **Authors**: *Nicholas Krämer, Philipp Hennig*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html)

        **Abstract**:

        We propose a fast algorithm for the probabilistic solution of boundary value problems (BVPs), which are ordinary differential equations subject to boundary conditions.  In contrast to previous work, we introduce a Gauss-Markov prior and tailor it specifically to BVPs, which allows computing a posterior distribution over the solution in linear time, at a quality and cost comparable to that of well-established, non-probabilistic methods.  Our model further delivers uncertainty quantification, mesh refinement, and hyperparameter adaptation. We demonstrate how these practical considerations positively impact the efficiency of the scheme. Altogether, this results in a practically usable probabilistic BVP solver that is (in contrast to non-probabilistic algorithms) natively compatible with other parts of the statistical modelling tool-chain.

        ----

        ## [853] Lifelong Domain Adaptation via Consolidated Internal Distribution

        **Authors**: *Mohammad Rostami*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5caf41d62364d5b41a893adc1a9dd5d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5caf41d62364d5b41a893adc1a9dd5d4-Abstract.html)

        **Abstract**:

        We develop an algorithm to address unsupervised domain adaptation (UDA) in continual learning (CL) settings.  The goal is to update a model continually to learn distributional shifts across sequentially arriving tasks with unlabeled data while retaining the knowledge about the past learned tasks. Existing  UDA  algorithms address the challenge of domain shift, but they require simultaneous access to the datasets of the source and the target domains. On the other hand, existing works on CL can handle tasks with labeled data.  Our solution is based on consolidating the learned internal distribution for improved model generalization on new domains and benefitting from experience replay to overcome catastrophic forgetting.

        ----

        ## [854] Counterbalancing Learning and Strategic Incentives in Allocation Markets

        **Authors**: *Jamie Kang, Faidra Monachou, Moran Koren, Itai Ashlagi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5cc3749a6e56ef6d656735dff9176074-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5cc3749a6e56ef6d656735dff9176074-Abstract.html)

        **Abstract**:

        Motivated by the high discard rate of donated organs in the United States, we study an allocation problem in the presence of learning and strategic incentives. We consider a setting where a benevolent social planner decides whether and how to allocate a single indivisible object to a queue of strategic agents.  The object has a common true quality, good or bad,  which is ex-ante unknown to everyone. Each agent holds an informative, yet noisy, private signal about the quality. To make a correct allocation decision the planner attempts to learn the object quality by truthfully eliciting agents' signals. Under the commonly applied sequential offering mechanism, we show that learning is hampered by the presence of strategic incentives as herding may emerge. This can result in incorrect allocation and welfare loss. To overcome these issues, we propose a novel class of incentive-compatible mechanisms. Our mechanism involves a batch-by-batch, dynamic voting process using a majority rule. We prove that the proposed voting mechanisms improve the probability of correct allocation whenever agents are sufficiently well informed. Particularly, we show that such an improvement can be achieved via a simple greedy algorithm. We quantify the improvement using simulations.

        ----

        ## [855] Controlling Neural Networks with Rule Representations

        **Authors**: *Sungyong Seo, Sercan Ö. Arik, Jinsung Yoon, Xiang Zhang, Kihyuk Sohn, Tomas Pfister*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5cd5058bca53951ffa7801bcdf421651-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5cd5058bca53951ffa7801bcdf421651-Abstract.html)

        **Abstract**:

        We propose a novel training method that integrates rules into deep learning, in a way the strengths of the rules are controllable at inference. Deep Neural Networks with Controllable Rule Representations (DeepCTRL) incorporates a rule encoder into the model coupled with a rule-based objective, enabling a shared representation for decision making. DeepCTRL is agnostic to data type and model architecture. It can be applied to any kind of rule defined for inputs and outputs. The key aspect of DeepCTRL is that it does not require retraining to adapt the rule strength -- at inference, the user can adjust it based on the desired operation point on accuracy vs. rule verification ratio. In real-world domains where incorporating rules is critical -- such as Physics, Retail and Healthcare -- we show the effectiveness of DeepCTRL in teaching rules for deep learning. DeepCTRL improves the trust and reliability of the trained models by significantly increasing their rule verification ratio, while also providing accuracy gains at downstream tasks. Additionally, DeepCTRL enables novel use cases such as hypothesis testing of the rules on data samples, and unsupervised adaptation based on shared rules between datasets.

        ----

        ## [856] Making the most of your day: online learning for optimal allocation of time

        **Authors**: *Etienne Boursier, Tristan Garrec, Vianney Perchet, Marco Scarsini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5d2c2cee8ab0b9a36bd1ed7196bd6c4a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5d2c2cee8ab0b9a36bd1ed7196bd6c4a-Abstract.html)

        **Abstract**:

        We study online learning for optimal allocation when the resource to be allocated is time. An agent receives task proposals sequentially according to a Poisson process and can either accept or reject a proposed task. If she accepts the proposal, she is busy for the duration of the task and obtains a reward that depends on the task duration. If she rejects it, she remains on hold until a new task proposal arrives. We study the regret incurred by the agent first when she knows her reward function but does not know the distribution of the task duration, and then when she does not know her reward function, either. Faster rates are finally obtained by adding structural assumptions on the distribution of rides or on the reward function. This natural setting bears similarities with contextual (one-armed) bandits, but with the crucial difference that the normalized reward associated to a context depends on the whole distribution of contexts.

        ----

        ## [857] Federated Reconstruction: Partially Local Federated Learning

        **Authors**: *Karan Singhal, Hakim Sidahmed, Zachary Garrett, Shanshan Wu, John Rush, Sushant Prakash*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5d44a2b0d85aa1a4dd3f218be6422c66-Abstract.html)

        **Abstract**:

        Personalization methods in federated learning aim to balance the benefits of federated and local training for data availability, communication cost, and robustness to client heterogeneity. Approaches that require clients to communicate all model parameters can be undesirable due to privacy and communication constraints. Other approaches require always-available or stateful clients, impractical in large-scale cross-device settings. We introduce Federated Reconstruction, the first model-agnostic framework for partially local federated learning suitable for training and inference at scale. We motivate the framework via a connection to model-agnostic meta learning, empirically demonstrate its performance over existing approaches for collaborative filtering and next word prediction, and release an open-source library for evaluating approaches in this setting. We also describe the successful deployment of this approach at scale for federated collaborative filtering in a mobile keyboard application.

        ----

        ## [858] Optimal prediction of Markov chains with and without spectral gap

        **Authors**: *Yanjun Han, Soham Jana, Yihong Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5d69dc892ba6e79fda0c6a1e286f24c5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5d69dc892ba6e79fda0c6a1e286f24c5-Abstract.html)

        **Abstract**:

        We study the following learning problem with dependent data: Given a trajectory of length $n$ from a stationary Markov chain with $k$ states, the goal is to predict the distribution of the next state. For $3 \leq k \leq O(\sqrt{n})$, the optimal prediction risk in the Kullback-Leibler divergence is shown to be $\Theta(\frac{k^2}{n}\log \frac{n}{k^2})$, in contrast to the optimal rate of $\Theta(\frac{\log \log n}{n})$ for $k=2$ previously shown in Falahatgar et al in 2016. These nonparametric rates can be attributed to the memory in the data, as the spectral gap of the Markov chain can be arbitrarily small. To quantify the memory effect, we study irreducible reversible chains with a prescribed spectral gap. In addition to characterizing the optimal prediction risk for two states, we show that, as long as the spectral gap is not excessively small, the prediction risk in the Markov model is $O(\frac{k^2}{n})$, which coincides with that of an iid model with the same number of parameters.

        ----

        ## [859] Subquadratic Overparameterization for Shallow Neural Networks

        **Authors**: *Chaehwan Song, Ali Ramezani-Kebrya, Thomas Pethick, Armin Eftekhari, Volkan Cevher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5d9e4a04afb9f3608ccc76c1ffa7573e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5d9e4a04afb9f3608ccc76c1ffa7573e-Abstract.html)

        **Abstract**:

        Overparameterization refers to the important phenomenon where the width of a neural network is chosen such that learning algorithms can provably attain zero loss in nonconvex training. The existing theory establishes such global convergence using various initialization strategies, training modifications, and width scalings. In particular, the state-of-the-art results require the width to scale quadratically with the number of training data under standard initialization strategies used in practice for best generalization performance. In contrast, the most recent results obtain linear scaling either with requiring initializations that lead to the "lazy-training",  or training only a single layer. In this work, we provide an analytical framework that allows us to adopt standard initialization strategies, possibly avoid lazy training, and train all layers simultaneously in basic shallow neural networks while attaining  a desirable subquadratic scaling on the network width. We achieve the desiderata via Polyak-Lojasiewicz condition, smoothness, and standard assumptions on data, and use tools from random matrix theory.

        ----

        ## [860] Continuous Doubly Constrained Batch Reinforcement Learning

        **Authors**: *Rasool Fakoor, Jonas Mueller, Kavosh Asadi, Pratik Chaudhari, Alexander J. Smola*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5da713a690c067105aeb2fae32403405-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5da713a690c067105aeb2fae32403405-Abstract.html)

        **Abstract**:

        Reliant on too many experiments to learn good actions, current Reinforcement Learning (RL) algorithms have limited applicability in real-world settings, which can be too expensive to allow exploration. We propose an algorithm for batch RL, where effective policies are learned using only a fixed offline dataset instead of online interactions with the environment. The limited data in batch RL produces inherent uncertainty in value estimates of states/actions that were insufficiently represented in the training data. This leads to particularly severe extrapolation when our candidate policies diverge from one that generated the data. We propose to mitigate this issue via two straightforward penalties: a policy-constraint to reduce this divergence and a value-constraint that discourages overly optimistic estimates. Over a comprehensive set of $32$ continuous-action batch RL benchmarks, our approach compares favorably to state-of-the-art methods, regardless of how the offline data were collected.

        ----

        ## [861] Bridging Explicit and Implicit Deep Generative Models via Neural Stein Estimators

        **Authors**: *Qitian Wu, Rui Gao, Hongyuan Zha*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5db60c98209913790e4fcce4597ee37c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5db60c98209913790e4fcce4597ee37c-Abstract.html)

        **Abstract**:

        There are two types of deep generative models: explicit and implicit. The former defines an explicit density form that allows likelihood inference; while the latter targets a flexible transformation from random noise to generated samples.  While the two classes of generative models have shown great power in many applications, both of them, when used alone, suffer from respective limitations and drawbacks. To take full advantages of both models and enable mutual compensation, we propose a novel joint training framework that bridges an explicit (unnormalized) density estimator and an implicit sample generator via Stein discrepancy. We show that our method 1) induces novel mutual regularization via kernel Sobolev norm penalization and Moreau-Yosida regularization, and 2) stabilizes the training dynamics. Empirically, we demonstrate that proposed method can facilitate the density estimator to more accurately identify data modes and guide the generator to output higher-quality samples, comparing with training a single counterpart. The new approach also shows promising results when the training samples are contaminated or limited.

        ----

        ## [862] Score-based Generative Modeling in Latent Space

        **Authors**: *Arash Vahdat, Karsten Kreis, Jan Kautz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5dca4c6b9e244d24a30b4c45601d9720-Abstract.html)

        **Abstract**:

        Score-based generative models (SGMs) have recently demonstrated impressive results in terms of both sample quality and distribution coverage. However, they are usually applied directly in data space and often require thousands of network evaluations for sampling. Here, we propose the Latent Score-based Generative Model (LSGM), a novel approach that trains SGMs in a latent space, relying on the variational autoencoder framework. Moving from data to latent space allows us to train more expressive generative models, apply SGMs to non-continuous data, and learn smoother SGMs in a smaller space, resulting in fewer network evaluations and faster sampling. To enable training LSGMs end-to-end in a scalable and stable manner, we (i) introduce a new score-matching objective suitable to the LSGM setting, (ii) propose a novel parameterization of the score function that allows SGM to focus on the mismatch of the target distribution with respect to a simple Normal one, and (iii) analytically derive multiple techniques for variance reduction of the training objective. LSGM obtains a state-of-the-art FID score of 2.10 on CIFAR-10, outperforming all existing generative results on this dataset. On CelebA-HQ-256, LSGM is on a par with previous SGMs in sample quality while outperforming them in sampling time by two orders of magnitude. In modeling binary images, LSGM achieves state-of-the-art likelihood on the binarized OMNIGLOT dataset.

        ----

        ## [863] Deep Conditional Gaussian Mixture Model for Constrained Clustering

        **Authors**: *Laura Manduchi, Kieran Chin-Cheong, Holger Michel, Sven Wellmann, Julia E. Vogt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5dd9db5e033da9c6fb5ba83c7a7ebea9-Abstract.html)

        **Abstract**:

        Constrained clustering has gained significant attention in the field of machine learning as it can leverage prior information on a growing amount of only partially labeled data. Following recent advances in deep generative models, we propose a novel framework for constrained clustering that is intuitive, interpretable, and can be trained efficiently in the framework of stochastic gradient variational inference. By explicitly integrating domain knowledge in the form of probabilistic relations, our proposed model (DC-GMM) uncovers the underlying distribution of data conditioned on prior clustering preferences, expressed as \textit{pairwise constraints}. These constraints guide the clustering process towards a desirable partition of the data by indicating which samples should or should not belong to the same cluster. We provide extensive experiments to demonstrate that DC-GMM shows superior clustering performances and robustness compared to state-of-the-art deep constrained clustering methods on a wide range of data sets. We further demonstrate the usefulness of our approach on two challenging real-world applications.

        ----

        ## [864] Bootstrap Your Object Detector via Mixed Training

        **Authors**: *Mengde Xu, Zheng Zhang, Fangyun Wei, Yutong Lin, Yue Cao, Stephen Lin, Han Hu, Xiang Bai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5e15fb59326e7a9c3d6558ca74621683-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5e15fb59326e7a9c3d6558ca74621683-Abstract.html)

        **Abstract**:

        We introduce MixTraining, a new training paradigm for object detection that can improve the performance of existing detectors for free. MixTraining enhances data augmentation by utilizing augmentations of different strengths while excluding the strong augmentations of certain training samples that may be detrimental to training. In addition, it addresses localization noise and missing labels in human annotations by incorporating pseudo boxes that can compensate for these errors. Both of these MixTraining capabilities are made possible through bootstrapping on the detector, which can be used to predict the difficulty of training on a strong augmentation, as well as to generate reliable pseudo boxes thanks to the robustness of neural networks to labeling error. MixTraining is found to bring consistent improvements across various detectors on the COCO dataset. In particular, the performance of Faster R-CNN~\cite{ren2015faster} with a ResNet-50~\cite{he2016deep} backbone is improved from 41.7 mAP to 44.0 mAP, and the accuracy of Cascade-RCNN~\cite{cai2018cascade} with a Swin-Small~\cite{liu2021swin} backbone is raised from 50.9 mAP to 52.8 mAP.

        ----

        ## [865] Tensor decompositions of higher-order correlations by nonlinear Hebbian plasticity

        **Authors**: *Gabriel Koch Ocker, Michael A. Buice*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5e34a2b4c23f4de585fb09a7f546f527-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5e34a2b4c23f4de585fb09a7f546f527-Abstract.html)

        **Abstract**:

        Biological synaptic plasticity exhibits nonlinearities that are not accounted for by classic Hebbian learning rules. Here, we introduce a simple family of generalized nonlinear Hebbian learning rules. We study the computations implemented by their dynamics in the simple setting of a neuron receiving feedforward inputs. These nonlinear Hebbian rules allow a neuron to learn tensor decompositions of its higher- order input correlations. The particular input correlation decomposed and the form of the decomposition depend on the location of nonlinearities in the plasticity rule. For simple, biologically motivated parameters, the neuron learns eigenvectors of higher-order input correlation tensors. We prove that tensor eigenvectors are attractors and determine their basins of attraction. We calculate the volume of those basins, showing that the dominant eigenvector has the largest basin of attraction. We then study arbitrary learning rules and find that any learning rule that admits a finite Taylor expansion into the neural input and output also has stable equilibria at generalized eigenvectors of higher-order input correlation tensors. Nonlinearities in synaptic plasticity thus allow a neuron to encode higher-order input correlations in a simple fashion.

        ----

        ## [866] Online Adaptation to Label Distribution Shift

        **Authors**: *Ruihan Wu, Chuan Guo, Yi Su, Kilian Q. Weinberger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5e6bd7a6970cd4325e587f02667f7f73-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5e6bd7a6970cd4325e587f02667f7f73-Abstract.html)

        **Abstract**:

        Machine learning models often encounter distribution shifts when deployed in the real world.  In this paper, we focus on adaptation to label distribution shift in the online setting, where the test-time label distribution is continually changing and the model must dynamically adapt to it without observing the true label. This setting is common in many real world scenarios such as medical diagnosis, where disease prevalences can vary substantially at different times of the year. Leveraging a novel analysis, we show that the lack of true label does not hinder estimation of the expected test loss, which enables the reduction of online label shift adaptation to conventional online learning. Informed by this observation, we propose adaptation algorithms inspired by classical online learning techniques such as Follow The Leader (FTL) and Online Gradient Descent (OGD) and derive their regret bounds. We empirically verify our findings under both simulated and real world label distribution shifts and show that OGD is particularly effective and robust to a variety of challenging label shift scenarios.

        ----

        ## [867] One Explanation is Not Enough: Structured Attention Graphs for Image Classification

        **Authors**: *Vivswan Shitole, Fuxin Li, Minsuk Kahng, Prasad Tadepalli, Alan Fern*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5e751896e527c862bf67251a474b3819-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5e751896e527c862bf67251a474b3819-Abstract.html)

        **Abstract**:

        Attention maps are popular tools for explaining the decisions of convolutional neural networks (CNNs) for image classification. Typically, for each image of interest, a single attention map is produced, which assigns weights to pixels based on their importance to the classification. We argue that a single attention map provides an incomplete understanding since there are often many other maps that explain a classification equally well. In this paper, we propose to utilize a beam search algorithm to systematically search for multiple explanations for each image. Results show that there are indeed multiple relatively localized explanations for many images. However, naively showing multiple explanations to users can be overwhelming and does not reveal their common and distinct structures. We introduce structured attention graphs (SAGs), which compactly represent sets of attention maps for an image by visualizing how different combinations of image regions impact the confidence of a classifier. An approach to computing a compact and representative SAG for visualization is proposed via diverse sampling. We conduct a user study comparing the use of SAGs to traditional attention maps for answering comparative counterfactual questions about image classifications. Our results show that the users are significantly more accurate when presented with SAGs compared to standard attention map baselines.

        ----

        ## [868] Integrating Expert ODEs into Neural ODEs: Pharmacology and Disease Progression

        **Authors**: *Zhaozhi Qian, William R. Zame, Lucas M. Fleuren, Paul W. G. Elbers, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5ea1649a31336092c05438df996a3e59-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5ea1649a31336092c05438df996a3e59-Abstract.html)

        **Abstract**:

        Modeling a system's temporal behaviour in reaction to external stimuli is a fundamental problem in many areas. Pure Machine Learning (ML) approaches often fail in the small sample regime and cannot provide actionable insights beyond predictions. A promising modification has been to incorporate expert domain knowledge into ML models. The application we consider is predicting the patient health status and disease progression over time, where a wealth of domain knowledge is available from pharmacology. Pharmacological models describe the dynamics of carefully-chosen medically meaningful variables in terms of systems of Ordinary Differential Equations (ODEs). However, these models only describe a limited collection of variables, and these variables are often not observable in clinical environments. To close this gap, we propose the latent hybridisation model (LHM) that integrates a system of expert-designed ODEs with machine-learned Neural ODEs to fully describe the dynamics of the system and to link the expert and latent variables to observable quantities. We evaluated LHM on synthetic data as well as real-world intensive care data of COVID-19 patients. LHM consistently outperforms previous works, especially when few training samples are available such as at the beginning of the pandemic.

        ----

        ## [869] Shifted Chunk Transformer for Spatio-Temporal Representational Learning

        **Authors**: *Xuefan Zha, Wentao Zhu, Xun Lv, Sen Yang, Ji Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5edc4f7dce28c711afc6265b4f99bf57-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5edc4f7dce28c711afc6265b4f99bf57-Abstract.html)

        **Abstract**:

        Spatio-temporal representational learning has been widely adopted in various fields such as action recognition, video object segmentation, and action anticipation.Previous spatio-temporal representational learning approaches primarily employ ConvNets or sequential models, e.g., LSTM, to learn the intra-frame and inter-frame features.  Recently, Transformer models have successfully dominated the study of natural language processing (NLP), image classification, etc. However, the pure-Transformer based spatio-temporal learning can be prohibitively costly on memory and computation to extract fine-grained features from a tiny patch. To tackle the training difficulty and enhance the spatio-temporal learning, we construct a shifted chunk Transformer with pure self-attention blocks. Leveraging the recent efficient Transformer design in NLP, this shifted chunk Transformer can learn hierarchical spatio-temporal features from a local tiny patch to a global videoclip. Our shifted self-attention can also effectively model complicated inter-frame variances. Furthermore, we build a clip encoder based on Transformer to model long-term temporal dependencies. We conduct thorough ablation studies to validate each component and hyper-parameters in our shifted chunk Transformer, and it outperforms previous state-of-the-art approaches on Kinetics-400, Kinetics-600,UCF101, and HMDB51.

        ----

        ## [870] Faster proximal algorithms for matrix optimization using Jacobi-based eigenvalue methods

        **Authors**: *Hamza Fawzi, Harry Goulbourne*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5ef78f63ba22e7dfb2fa44613311b932-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5ef78f63ba22e7dfb2fa44613311b932-Abstract.html)

        **Abstract**:

        We consider proximal splitting algorithms for convex optimization problems over matrices. A significant computational bottleneck in many of these algorithms is the need to compute a full eigenvalue or singular value decomposition at each iteration for the evaluation of a proximal operator.In this paper we propose to use an old and surprisingly simple method due to Jacobi to compute these eigenvalue and singular value decompositions, and we demonstrate that it can lead to substantial gains in terms of computation time compared to standard approaches. We rely on three essential properties of this method: (a) its ability to exploit an approximate decomposition as an initial point, which in the case of iterative optimization algorithms can be obtained from the previous iterate; (b) its parallel nature which makes it a great fit for hardware accelerators such as GPUs, now common in machine learning, and (c) its simple termination criterion which allows us to trade-off accuracy with computation time. We demonstrate the efficacy of this approach on a variety of algorithms and problems, and show that, on a GPU, we can obtain 5 to 10x speed-ups in the evaluation of proximal operators compared to standard CPU or GPU linear algebra routines. Our findings are supported by new theoretical results providing guarantees on the approximation quality of proximal operators obtained using approximate eigenvalue or singular value decompositions.

        ----

        ## [871] Decrypting Cryptic Crosswords: Semantically Complex Wordplay Puzzles as a Target for NLP

        **Authors**: *Josh Rozner, Christopher Potts, Kyle Mahowald*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5f1d3986fae10ed2994d14ecd89892d7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5f1d3986fae10ed2994d14ecd89892d7-Abstract.html)

        **Abstract**:

        Cryptic crosswords, the dominant crossword variety in the UK, are a promising target for advancing NLP systems that seek to process semantically complex, highly compositional language. Cryptic clues read like fluent natural language but are adversarially composed of two parts: a definition and a wordplay cipher requiring character-level manipulations. Expert humans use creative intelligence to solve cryptics, flexibly combining linguistic, world, and domain knowledge. In this paper, we make two main contributions. First, we present a dataset of cryptic clues as a challenging new benchmark for NLP systems that seek to process compositional language in more creative, human-like ways. After showing that three non-neural approaches and T5, a state-of-the-art neural language model, do not achieve good performance, we make our second main contribution: a novel curriculum approach, in which the model is first fine-tuned on related tasks such as unscrambling words. We also introduce a challenging data split, examine the meta-linguistic capabilities of subword-tokenized models, and investigate model systematicity by perturbing the wordplay part of clues, showing that T5 exhibits behavior partially consistent with human solving strategies. Although our curricular approach considerably improves on the T5 baseline, our best-performing model still fails to generalize to the extent that humans can. Thus, cryptic crosswords remain an unsolved challenge for NLP systems and a potential source of future innovation.

        ----

        ## [872] An Improved Analysis of Gradient Tracking for Decentralized Machine Learning

        **Authors**: *Anastasia Koloskova, Tao Lin, Sebastian U. Stich*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5f25fbe144e4a81a1b0080b6c1032778-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5f25fbe144e4a81a1b0080b6c1032778-Abstract.html)

        **Abstract**:

        We consider decentralized machine learning over a network where the training data is distributed across $n$ agents, each of which can compute stochastic model updates on their local data. The agent's common goal is to find a model that minimizes the average of all local loss functions. While gradient tracking (GT) algorithms can overcome a key challenge, namely accounting for differences between workers' local data distributions, the known convergence rates for GT algorithms are not optimal with respect to their dependence on the mixing parameter $p$ (related to the spectral gap of the connectivity matrix).We provide a tighter analysis of the GT method in the stochastic strongly convex, convex and non-convex settings. We improve the dependency on $p$ from $\mathcal{O}(p^{-2})$ to $\mathcal{O}(p^{-1}c^{-1})$ in the noiseless case and from $\mathcal{O}(p^{-3/2})$ to $\mathcal{O}(p^{-1/2}c^{-1})$ in the general stochastic case, where $c \geq p$ is related to the negative eigenvalues of the connectivity matrix (and is a constant in most practical applications). This improvement was possible due to a new proof technique which could be of independent interest.

        ----

        ## [873] Entropic Desired Dynamics for Intrinsic Control

        **Authors**: *Steven Hansen, Guillaume Desjardins, Kate Baumli, David Warde-Farley, Nicolas Heess, Simon Osindero, Volodymyr Mnih*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5f7f02b7e4ade23430f345f954c938c1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5f7f02b7e4ade23430f345f954c938c1-Abstract.html)

        **Abstract**:

        An agent might be said, informally, to have mastery of its environment when it has maximised the effective number of states it can reliably reach. In practice, this often means maximizing the number of latent codes that can be discriminated from future states under some short time horizon (e.g. \cite{eysenbach2018diversity}). By situating these latent codes in a globally consistent coordinate system, we show that agents can reliably reach more states in the long term while still optimizing a local objective. A simple instantiation of this idea, \textbf{E}ntropic \textbf{D}esired \textbf{D}ynamics for \textbf{I}ntrinsic \textbf{C}on\textbf{T}rol (EDDICT), assumes fixed additive latent dynamics, which results in tractable learning and an interpretable latent space. Compared to prior methods, EDDICT's globally consistent codes allow it to be far more exploratory, as demonstrated by improved state coverage and increased unsupervised performance on hard exploration games such as Montezuma's Revenge.

        ----

        ## [874] Exploring Cross-Video and Cross-Modality Signals for Weakly-Supervised Audio-Visual Video Parsing

        **Authors**: *Yan-Bo Lin, Hung-Yu Tseng, Hsin-Ying Lee, Yen-Yu Lin, Ming-Hsuan Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5f93f983524def3dca464469d2cf9f3e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5f93f983524def3dca464469d2cf9f3e-Abstract.html)

        **Abstract**:

        The audio-visual video parsing task aims to temporally parse a video into audio or visual event categories. However, it is labor intensive to temporally annotate audio and visual events and thus hampers the learning of a parsing model. To this end, we propose to explore additional cross-video and cross-modality supervisory signals to facilitate weakly-supervised audio-visual video parsing. The proposed method exploits both the common and diverse event semantics across videos to identify audio or visual events. In addition, our method explores event co-occurrence across audio, visual, and audio-visual streams. We leverage the explored cross-modality co-occurrence to localize segments of target events while excluding irrelevant ones. The discovered supervisory signals across different videos and modalities can greatly facilitate the training with only video-level annotations. Quantitative and qualitative results demonstrate that the proposed method performs favorably against existing methods on weakly-supervised audio-visual video parsing.

        ----

        ## [875] Littlestone Classes are Privately Online Learnable

        **Authors**: *Noah Golowich, Roi Livni*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fbb4eb0e7c2cedf731ec7c18e344141-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fbb4eb0e7c2cedf731ec7c18e344141-Abstract.html)

        **Abstract**:

        We consider the problem of online classification under a privacy constraint. In this setting a learner observes sequentially a stream of labelled examples $(x_t, y_t)$, for $1 \leq t \leq T$, and returns at each iteration $t$ a hypothesis $h_t$ which is used to predict the label of each new example $x_t$. The learner's performance is measured by her regret against a known hypothesis class $\mathcal{H}$. We require that the algorithm satisfies the following privacy constraint: the sequence $h_1, \ldots, h_T$ of hypotheses output by the algorithm needs to be an $(\epsilon, \delta)$-differentially private function of the whole input sequence $(x_1, y_1), \ldots, (x_T, y_T)$.We provide the first non-trivial regret bound for the realizable setting. Specifically, we show that if the class $\mathcal{H}$ has constant Littlestone dimension then, given an oblivious sequence of labelled examples, there is a private learner that makes in expectation at most $O(\log T)$ mistakes -- comparable to the optimal mistake bound in the non-private case, up to a logarithmic factor. Moreover, for general values of the Littlestone dimension $d$, the same mistake bound holds but with a doubly-exponential in $d$ factor.     A recent line of work has demonstrated a strong connection between classes that are online learnable and those that are differentially-private learnable. Our results strengthen this connection and show that an online learning algorithm can in fact be directly privatized (in the realizable setting).We also discuss an adaptive setting and provide a sublinear regret bound of $O(\sqrt{T})$.

        ----

        ## [876] Dual Parameterization of Sparse Variational Gaussian Processes

        **Authors**: *Vincent Adam, Paul E. Chang, Mohammad Emtiyaz Khan, Arno Solin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fcc629edc0cfa360016263112fe8058-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fcc629edc0cfa360016263112fe8058-Abstract.html)

        **Abstract**:

        Sparse variational Gaussian process (SVGP) methods are a common choice for non-conjugate Gaussian process inference because of their computational benefits. In this paper, we improve their computational efficiency by using a dual parameterization where each data example is assigned dual parameters, similarly to site parameters used in expectation propagation. Our dual parameterization speeds-up inference using natural gradient descent, and provides a tighter evidence lower bound for hyperparameter learning. The approach has the same memory cost as the current SVGP methods, but it is faster and more accurate.

        ----

        ## [877] Learning to dehaze with polarization

        **Authors**: *Chu Zhou, Minggui Teng, Yufei Han, Chao Xu, Boxin Shi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fd0b37cd7dbbb00f97ba6ce92bf5add-Abstract.html)

        **Abstract**:

        Haze, a common kind of bad weather caused by atmospheric scattering, decreases the visibility of scenes and degenerates the performance of computer vision algorithms. Single-image dehazing methods have shown their effectiveness in a large variety of scenes, however, they are based on handcrafted priors or learned features, which do not generalize well to real-world images. Polarization information can be used to relieve its ill-posedness, however, real-world images are still challenging since existing polarization-based methods usually assume that the transmitted light is not significantly polarized, and they require specific clues to estimate necessary physical parameters. In this paper, we propose a generalized physical formation model of hazy images and a robust polarization-based dehazing pipeline without the above assumption or requirement, along with a neural network tailored to the pipeline. Experimental results show that our approach achieves state-of-the-art performance on both synthetic data and real-world hazy images.

        ----

        ## [878] Conservative Data Sharing for Multi-Task Offline Reinforcement Learning

        **Authors**: *Tianhe Yu, Aviral Kumar, Yevgen Chebotar, Karol Hausman, Sergey Levine, Chelsea Finn*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fd2c06f558321eff612bbbe455f6fbd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fd2c06f558321eff612bbbe455f6fbd-Abstract.html)

        **Abstract**:

        Offline reinforcement learning (RL) algorithms have shown promising results in domains where abundant pre-collected data is available. However, prior methods focus on solving individual problems from scratch with an offline dataset without considering how an offline RL agent can acquire multiple skills. We argue that a natural use case of offline RL is in settings where we can pool large amounts of data collected in various scenarios for solving different tasks, and utilize all of this data to learn behaviors for all the tasks more effectively rather than training each one in isolation. However, sharing data across all tasks in multi-task offline RL performs surprisingly poorly in practice. Thorough empirical analysis, we find that sharing data can actually exacerbate the distributional shift between the learned policy and the dataset, which in turn can lead to divergence of the learned policy and poor performance. To address this challenge, we develop a simple technique for data- sharing in multi-task offline RL that routes data based on the improvement over the task-specific data. We call this approach conservative data sharing (CDS), and it can be applied with multiple single-task offline RL methods. On a range of challenging multi-task locomotion, navigation, and vision-based robotic manipulation problems, CDS achieves the best or comparable performance compared to prior offline multi- task RL methods and previous data sharing approaches.

        ----

        ## [879] Universal Rate-Distortion-Perception Representations for Lossy Compression

        **Authors**: *George Zhang, Jingjing Qian, Jun Chen, Ashish Khisti*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fde40544cff0001484ecae2466ce96e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fde40544cff0001484ecae2466ce96e-Abstract.html)

        **Abstract**:

        In the context of lossy compression, Blau \& Michaeli (2019) adopt a mathematical notion of perceptual quality and define the information rate-distortion-perception function, generalizing the classical rate-distortion tradeoff. We consider the notion of universal representations in which one may fix an encoder and vary the decoder to achieve any point within a collection of distortion and perception constraints. We prove that the corresponding information-theoretic universal rate-distortion-perception function is operationally achievable in an approximate sense. Under MSE distortion, we show that the entire distortion-perception tradeoff of a Gaussian source can be achieved by a single encoder of the same rate asymptotically. We then characterize the achievable distortion-perception region for a fixed representation in the case of arbitrary distributions, and identify conditions under which the aforementioned results continue to hold approximately. This motivates the study of practical constructions that are approximately universal across the RDP tradeoff, thereby alleviating the need to design a new encoder for each objective. We provide experimental results on MNIST and SVHN suggesting that on image compression tasks, the operational tradeoffs achieved by machine learning models with a fixed encoder suffer only a small penalty when compared to their variable encoder counterparts.

        ----

        ## [880] What's a good imputation to predict with missing values?

        **Authors**: *Marine Le Morvan, Julie Josse, Erwan Scornet, Gaël Varoquaux*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5fe8fdc79ce292c39c5f209d734b7206-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5fe8fdc79ce292c39c5f209d734b7206-Abstract.html)

        **Abstract**:

        How to learn a good predictor on data with missing values? Most efforts focus on first imputing as well as possible and second learning on the completed data to predict the outcome. Yet, this widespread practice has no theoretical grounding. Here we show that for almost all imputation functions, an impute-then-regress procedure with a powerful learner is Bayes optimal. This result holds for all missing-values mechanisms, in contrast with the classic statistical results that require missing-at-random settings to use imputation in probabilistic modeling. Moreover, it implies that perfect conditional imputation is not needed for good prediction asymptotically. In fact, we show that on perfectly imputed data the best regression function will generally be discontinuous, which makes it hard to learn. Crafting instead the imputation so as to leave the regression function unchanged simply shifts the problem to learning discontinuous imputations. Rather, we suggest that it is easier to learn imputation and regression jointly. We propose such a procedure, adapting NeuMiss, a neural network capturing the conditional links across observed and unobserved variables whatever the missing-value pattern. Our experiments confirm that joint imputation and regression through NeuMiss is better than various two step procedures in a finite-sample regime.

        ----

        ## [881] Replacing Rewards with Examples: Example-Based Policy Search via Recursive Classification

        **Authors**: *Ben Eysenbach, Sergey Levine, Ruslan Salakhutdinov*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/5ffaa9f5182c2a36843f438bb1fdbdea-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/5ffaa9f5182c2a36843f438bb1fdbdea-Abstract.html)

        **Abstract**:

        Reinforcement learning (RL) algorithms assume that users specify tasks by manually writing down a reward function. However, this process can be laborious and demands considerable technical expertise. Can we devise RL algorithms that instead enable users to specify tasks simply by providing examples of successful outcomes? In this paper, we derive a control algorithm that maximizes the future probability of these successful outcome examples. Prior work has approached similar problems with a two-stage process, first learning a reward function and then optimizing this reward function using another reinforcement learning algorithm. In contrast, our method directly learns a value function from transitions and successful outcomes, without learning this intermediate reward function. Our method therefore requires fewer hyperparameters to tune and lines of code to debug. We show that our method satisfies a new data-driven Bellman equation, where examples take the place of the typical reward function term. Experiments show that our approach outperforms prior methods that learn explicit reward functions.

        ----

        ## [882] Hierarchical Skills for Efficient Exploration

        **Authors**: *Jonas Gehring, Gabriel Synnaeve, Andreas Krause, Nicolas Usunier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60106888f8977b71e1f15db7bc9a88d1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60106888f8977b71e1f15db7bc9a88d1-Abstract.html)

        **Abstract**:

        In reinforcement learning, pre-trained low-level skills have the potential to greatly facilitate exploration. However, prior knowledge of the downstream task is required to strike the right balance between generality (fine-grained control) and specificity (faster learning) in skill design. In previous work on continuous control, the sensitivity of methods to this trade-off has not been addressed explicitly, as locomotion provides a suitable prior for navigation tasks, which have been of foremost interest. In this work, we analyze this trade-off for low-level policy pre-training with a new benchmark suite of  diverse, sparse-reward tasks for bipedal robots. We alleviate the need for prior knowledge by proposing a hierarchical skill learning framework that acquires skills of varying complexity in an unsupervised manner. For utilization on downstream tasks, we present a three-layered hierarchical learning algorithm to automatically trade off between general and specific skills as required by the respective task. In our experiments, we show that our approach performs this trade-off effectively and achieves better results than current state-of-the-art methods for end-to-end hierarchical reinforcement learning and unsupervised skill discovery.

        ----

        ## [883] Evidential Softmax for Sparse Multimodal Distributions in Deep Generative Models

        **Authors**: *Phil Chen, Masha Itkina, Ransalu Senanayake, Mykel J. Kochenderfer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60243f9b1ac2dba11ff8131c8f4431e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60243f9b1ac2dba11ff8131c8f4431e0-Abstract.html)

        **Abstract**:

        Many applications of generative models rely on the marginalization of their high-dimensional output probability distributions. Normalization functions that yield sparse probability distributions can make exact marginalization more computationally tractable. However, sparse normalization functions usually require alternative loss functions for training since the log-likelihood is undefined for sparse probability distributions. Furthermore, many sparse normalization functions often collapse the multimodality of distributions. In this work, we present ev-softmax, a sparse normalization function that preserves the multimodality of probability distributions. We derive its properties, including its gradient in closed-form, and introduce a continuous family of approximations to ev-softmax that have full support and can be trained with probabilistic loss functions such as negative log-likelihood and Kullback-Leibler divergence. We evaluate our method on a variety of generative models, including variational autoencoders and auto-regressive architectures. Our method outperforms existing dense and sparse normalization techniques in distributional accuracy. We demonstrate that ev-softmax successfully reduces the dimensionality of probability distributions while maintaining multimodality.

        ----

        ## [884] Submodular + Concave

        **Authors**: *Siddharth Mitra, Moran Feldman, Amin Karbasi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/602443a3d6907117d8b4a308844e963e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/602443a3d6907117d8b4a308844e963e-Abstract.html)

        **Abstract**:

        It has been well established that first order optimization methods can converge to the maximal objective value of concave functions and provide constant factor approximation guarantees for (non-convex/non-concave) continuous submodular functions. In this work, we initiate the study of the maximization of functions of the form $F(x) = G(x) +C(x)$ over a solvable convex body $P$, where $G$ is a smooth DR-submodular function and $C$ is a smooth concave function. This class of functions is a strict extension of both concave and continuous DR-submodular functions for which no theoretical guarantee is known. We provide a suite of Frank-Wolfe style algorithms, which, depending on the nature of the objective function (i.e., if $G$ and $C$ are monotone or not, and non-negative or not) and on the nature of the set $P$ (i.e., whether it is downward closed or not), provide $1-1/e$, $1/e$, or $1/2$ approximation guarantees. We then use our algorithms to get a framework to smoothly interpolate between choosing a diverse set of elements from a given ground set (corresponding to the mode of a determinantal point process) and choosing a clustered set of elements (corresponding to the maxima of a suitable concave function). Additionally, we apply our algorithms to various functions in the above class (DR-submodular + concave) in both constrained and unconstrained settings, and show that our algorithms consistently outperform natural baselines.

        ----

        ## [885] DeepGEM: Generalized Expectation-Maximization for Blind Inversion

        **Authors**: *Angela F. Gao, Jorge C. Castellanos, Yisong Yue, Zachary E. Ross, Katherine L. Bouman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/606c90a06173d69682feb83037a68fec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/606c90a06173d69682feb83037a68fec-Abstract.html)

        **Abstract**:

        Typically, inversion algorithms assume that a forward model, which relates a source to its resulting measurements, is known and fixed. Using collected indirect measurements and the forward model, the goal becomes to recover the source. When the forward model is unknown, or imperfect, artifacts due to model mismatch occur in the recovery of the source. In this paper, we study the problem of blind inversion: solving an inverse problem with unknown or imperfect knowledge of the forward model parameters. We propose DeepGEM, a variational Expectation-Maximization (EM) framework that can be used to solve for the unknown parameters of the forward model in an unsupervised manner. DeepGEM makes use of a normalizing flow generative network to efficiently capture complex posterior distributions, which leads to more accurate evaluation of the source's posterior distribution used in EM. We showcase the effectiveness of our DeepGEM approach by achieving strong performance on the challenging problem of blind seismic tomography, where we significantly outperform the standard method used in seismology.  We also demonstrate the generality of DeepGEM by applying it to a simple case of blind deconvolution.

        ----

        ## [886] Learning to Generate Visual Questions with Noisy Supervision

        **Authors**: *Kai Shen, Lingfei Wu, Siliang Tang, Yueting Zhuang, Zhen He, Zhuoye Ding, Yun Xiao, Bo Long*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60792d855cd8a912a97711f91a1f155c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60792d855cd8a912a97711f91a1f155c-Abstract.html)

        **Abstract**:

        The task of visual question generation (VQG) aims to generate human-like neural questions from an image and potentially other side information (e.g., answer type or the answer itself). Existing works often suffer from the severe one image to many questions mapping problem, which generates uninformative and non-referential questions. Recent work has demonstrated that by leveraging double visual and answer hints, a model can faithfully generate much better quality questions. However, visual hints are not available naturally. Despite they proposed a simple rule-based similarity matching method to obtain candidate visual hints, they could be very noisy practically and thus restrict the quality of generated questions. In this paper, we present a novel learning approach for double-hints based VQG, which can be cast as a weakly supervised learning problem with noises. The key rationale is that the salient visual regions of interest can be viewed as a constraint to improve the generation procedure for producing high-quality questions. As a result, given the predicted salient visual regions of interest, we can focus on estimating the probability of being ground-truth questions, which in turn implicitly measures the quality of predicted visual hints. Experimental results on two benchmark datasets show that our proposed method outperforms the state-of-the-art approaches by a large margin on a variety of metrics, including both automatic machine metrics and human evaluation.

        ----

        ## [887] Pure Exploration in Kernel and Neural Bandits

        **Authors**: *Yinglun Zhu, Dongruo Zhou, Ruoxi Jiang, Quanquan Gu, Rebecca Willett, Robert Nowak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6084e82a08cb979cf75ae28aed37ecd4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6084e82a08cb979cf75ae28aed37ecd4-Abstract.html)

        **Abstract**:

        We study pure exploration in bandits, where the dimension of the feature representation can be much larger than the number of arms. To overcome the curse of dimensionality, we propose to adaptively embed the feature representation of each arm into a lower-dimensional space and carefully deal with the induced model misspecifications. Our approach is conceptually very different from existing works that can either only handle low-dimensional linear bandits or passively deal with model misspecifications. We showcase the application of our approach to two pure exploration settings that were previously under-studied: (1) the reward function belongs to a possibly infinite-dimensional Reproducing Kernel Hilbert Space, and (2) the reward function is nonlinear and can be approximated by neural networks. Our main results provide sample complexity guarantees that only depend on the effective dimension of the feature spaces in the kernel or neural representations. Extensive experiments conducted on both synthetic and real-world datasets demonstrate the efficacy of our methods.

        ----

        ## [888] Numerical Composition of Differential Privacy

        **Authors**: *Sivakanth Gopi, Yin Tat Lee, Lukas Wutschitz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6097d8f3714205740f30debe1166744e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6097d8f3714205740f30debe1166744e-Abstract.html)

        **Abstract**:

        We give a fast algorithm to compose privacy guarantees of differentially private (DP) algorithms to arbitrary accuracy. Our method is based on the notion of privacy loss random variables to quantify the privacy loss of DP algorithms. The running time and memory needed for our algorithm to approximate the privacy curve of a DP algorithm composed with itself $k$ times is $\tilde{O}(\sqrt{k})$. This improves over the best prior method by Koskela et al. (2020) which requires $\tilde{\Omega}(k^{1.5})$ running time. We demonstrate the utility of our algorithm by accurately computing the privacy loss of DP-SGD algorithm of Abadi et al. (2016) and showing that our algorithm speeds up the privacy computations by a few orders of magnitude compared to prior work, while maintaining similar accuracy.

        ----

        ## [889] Coresets for Classification - Simplified and Strengthened

        **Authors**: *Tung Mai, Cameron Musco, Anup Rao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6098ed616e715171f0dabad60a8e5197-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6098ed616e715171f0dabad60a8e5197-Abstract.html)

        **Abstract**:

        We give relative error coresets for training linear classifiers with a broad class of loss functions, including the logistic loss and hinge loss. Our construction achieves $(1\pm \epsilon)$ relative error with $\tilde O(d \cdot \mu_y(X)^2/\epsilon^2)$ points, where $\mu_y(X)$ is a natural complexity measure of the data matrix $X \in \mathbb{R}^{n \times d}$ and label vector $y \in \{-1,1\}^n$, introduced by Munteanu et al. 2018. Our result is based on subsampling data points with probabilities proportional to their  $\ell_1$ $Lewis$ $weights$. It significantly improves on existing theoretical bounds and performs  well in practice, outperforming uniform subsampling along with other importance sampling methods. Our sampling distribution does not depend on the labels, so can be used for active learning. It also does not depend on the specific loss function, so a single coreset can be used  in multiple training scenarios.

        ----

        ## [890] Sequential Algorithms for Testing Closeness of Distributions

        **Authors**: *Aadil Oufkir, Omar Fawzi, Nicolas Flammarion, Aurélien Garivier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/609c5e5089a9aa967232aba2a4d03114-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/609c5e5089a9aa967232aba2a4d03114-Abstract.html)

        **Abstract**:

        What advantage do sequential procedures provide over batch algorithms for testing properties of unknown distributions? Focusing on the problem of testing whether two distributions $\mathcal{D}_1$ and $\mathcal{D}_2$ on $\{1,\dots, n\}$ are equal or $\epsilon$-far, we give several answers to this question. We show that for a small alphabet size $n$, there is a sequential algorithm that outperforms any batch algorithm by a factor of at least $4$ in terms sample complexity. For a general alphabet size $n$, we give a sequential algorithm that uses no more samples than its batch counterpart, and possibly fewer if the actual distance between $\mathcal{D}_1$ and $\mathcal{D}_2$ is larger than $\epsilon$. As a corollary, letting $\epsilon$ go to $0$, we obtain a sequential algorithm for testing closeness (with no a priori bound on the distance between $\mathcal{D}_1$ and $\mathcal{D}_2$) with a sample complexity $\tilde{\mathcal{O}}(\frac{n^{2/3}}{TV(\mathcal{D}_1, \mathcal{D}_2)^{4/3}})$: this improves over the $\tilde{\mathcal{O}}(\frac{n/\log n}{TV(\mathcal{D}_1, \mathcal{D}_2)^{2} })$ tester of [Daskalakis and Kawase 2017]  and is optimal up to multiplicative constants. We also establish limitations of sequential algorithms for the problem of testing closeness: they can improve the worst case number of samples by at most a constant factor.

        ----

        ## [891] Overlapping Spaces for Compact Graph Representations

        **Authors**: *Kirill Shevkunov, Liudmila Prokhorenkova*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60b2149f6bafd1cc9d505496f09160ba-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60b2149f6bafd1cc9d505496f09160ba-Abstract.html)

        **Abstract**:

        Various non-trivial spaces are becoming popular for embedding structured data such as graphs, texts, or images. Following spherical and hyperbolic spaces, more general product spaces have been proposed. However, searching for the best configuration of a product space is a resource-intensive procedure, which reduces the practical applicability of the idea. We generalize the concept of product space and introduce an overlapping space that does not have the configuration search problem. The main idea is to allow subsets of coordinates to be shared between spaces of different types (Euclidean, hyperbolic, spherical). As a result, we often need fewer coordinates to store the objects. Additionally, we propose an optimization algorithm that automatically learns the optimal configuration. Our experiments confirm that overlapping spaces outperform the competitors in graph embedding tasks with different evaluation metrics. We also perform an empirical analysis in a realistic information retrieval setup, where we compare all spaces by incorporating them into DSSM. In this case, the proposed overlapping space consistently achieves nearly optimal results without any configuration tuning. This allows for reducing training time, which can be essential in large-scale applications.

        ----

        ## [892] Hyperparameter Tuning is All You Need for LISTA

        **Authors**: *Xiaohan Chen, Jialin Liu, Zhangyang Wang, Wotao Yin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60c97bef031ec312b512c08565c1868e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60c97bef031ec312b512c08565c1868e-Abstract.html)

        **Abstract**:

        Learned Iterative Shrinkage-Thresholding Algorithm (LISTA) introduces the concept of unrolling an iterative algorithm and training it like a neural network. It has had great success on sparse recovery. In this paper, we show that adding momentum to intermediate variables in the LISTA network achieves a better convergence rate and, in particular, the network with instance-optimal parameters is superlinearly convergent. Moreover, our new theoretical results lead to a practical approach of automatically and adaptively calculating the parameters of a LISTA network layer based on its previous layers. Perhaps most surprisingly, such an adaptive-parameter procedure reduces the training of LISTA to tuning only three hyperparameters from data: a new record set in the context of the recent advances on trimming down LISTA complexity. We call this new ultra-light weight network HyperLISTA. Compared to state-of-the-art LISTA models, HyperLISTA achieves almost the same performance on seen data distributions and performs better when tested on unseen distributions (speciÔ¨Åcally, those with different sparsity levels and nonzero magnitudes). Code is available: https://github.com/VITA-Group/HyperLISTA.

        ----

        ## [893] Foundations of Symbolic Languages for Model Interpretability

        **Authors**: *Marcelo Arenas, Daniel Báez, Pablo Barceló, Jorge Pérez, Bernardo Subercaseaux*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60cb558c40e4f18479664069d9642d5a-Abstract.html)

        **Abstract**:

        Several queries and scores have recently been proposed to explain individual predictions over ML models. Examples include queries based on “anchors”, which are parts of an instance that are sufficient to justify its classification, and “feature-perturbation” scores such as SHAP. Given the need for flexible, reliable, and easy-to-apply interpretability methods for ML models, we foresee the need for developing declarative languages to naturally specify different explainability queries. We do this in a principled way by rooting such a language in a logic called FOIL, which allows for expressing many simple but important explainability queries, and might serve as a core for more expressive interpretability languages. We study the computational complexity of FOIL queries over two classes of ML models often deemed to be easily interpretable: decision trees and more general decision diagrams. Since the number of possible inputs for an ML model is exponential in its dimension, tractability of the FOIL evaluation problem is delicate but can be achieved by either restricting the structure of the models, or the fragment of FOIL being evaluated.  We also present a prototype implementation of FOIL wrapped in a high-level declarative language and perform experiments showing that such a language can be used in practice.

        ----

        ## [894] Bridging Offline Reinforcement Learning and Imitation Learning: A Tale of Pessimism

        **Authors**: *Paria Rashidinejad, Banghua Zhu, Cong Ma, Jiantao Jiao, Stuart Russell*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/60ce36723c17bbac504f2ef4c8a46995-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/60ce36723c17bbac504f2ef4c8a46995-Abstract.html)

        **Abstract**:

        Offline (or batch) reinforcement learning (RL) algorithms seek to learn an optimal policy from a fixed dataset without active data collection. Based on the composition of the offline dataset, two main methods are used: imitation learning which is suitable for expert datasets, and vanilla offline RL which often requires uniform coverage datasets. From a practical standpoint, datasets often deviate from these two extremes and the exact data composition is usually unknown. To bridge this gap, we present a new offline RL framework that smoothly interpolates between the two extremes of data composition, hence unifying imitation learning and vanilla offline RL. The new framework is centered around a weak version of the concentrability coefficient that measures the deviation of the behavior policy from the expert policy alone. Under this new framework, we ask: can one develop an algorithm that achieves a minimax optimal rate adaptive to unknown data composition? To address this question, we consider a lower confidence bound (LCB) algorithm developed based on pessimism in the face of uncertainty in offline RL. We study finite-sample properties of LCB as well as information-theoretic limits in multi-armed bandits, contextual bandits, and Markov decision processes (MDPs). Our analysis reveals surprising facts about optimality rates. In particular, in both contextual bandits and RL, LCB achieves a faster rate of $1/N$ for nearly-expert datasets compared to the usual rate of $1/\sqrt{N}$ in offline RL, where $N$ is the batch dataset sample size. In contextual bandits with at least two contexts, we prove that LCB is adaptively optimal for the entire data composition range, achieving a smooth transition from imitation learning to offline RL. We further show that LCB is almost adaptively optimal in MDPs.

        ----

        ## [895] Impression learning: Online representation learning with synaptic plasticity

        **Authors**: *Colin Bredenberg, Benjamin Lyo, Eero P. Simoncelli, Cristina Savin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/615299acbbac3e21302bbc435091ad9f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/615299acbbac3e21302bbc435091ad9f-Abstract.html)

        **Abstract**:

        Understanding how the brain constructs statistical models of the sensory world remains a longstanding challenge for computational neuroscience. Here, we derive an unsupervised local synaptic plasticity rule that trains neural circuits to infer latent structure from sensory stimuli via a novel loss function for approximate online Bayesian inference. The learning algorithm is driven by a local error signal computed between two factors that jointly contribute to neural activity: stimulus drive and internal predictions --- the network's 'impression' of the stimulus. Physiologically, we associate these two components with the basal and apical dendrites of pyramidal neurons, respectively. We show that learning can be implemented online, is capable of capturing temporal dependencies in continuous input streams, and generalizes to hierarchical architectures. Furthermore, we demonstrate both analytically and empirically that the algorithm is more data-efficient than a three-factor plasticity alternative, enabling it to learn statistics of high-dimensional, naturalistic inputs. Overall, the model provides a bridge from mechanistic accounts of synaptic plasticity to algorithmic descriptions of unsupervised probabilistic learning and inference.

        ----

        ## [896] How Well do Feature Visualizations Support Causal Understanding of CNN Activations?

        **Authors**: *Roland S. Zimmermann, Judy Borowski, Robert Geirhos, Matthias Bethge, Thomas S. A. Wallis, Wieland Brendel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/618faa1728eb2ef6e3733645273ab145-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/618faa1728eb2ef6e3733645273ab145-Abstract.html)

        **Abstract**:

        A precise understanding of why units in an artificial network respond to certain stimuli would constitute a big step towards explainable artificial intelligence. One widely used approach towards this goal is to visualize unit responses via activation maximization. These feature visualizations are purported to provide humans with precise information about the image features that cause a unit to be activated - an advantage over other alternatives like strongly activating dataset samples. If humans indeed gain causal insight from visualizations, this should enable them to predict the effect of an intervention, such as how occluding a certain patch of the image (say, a dog's head) changes a unit's activation. Here, we test this hypothesis by asking humans to decide which of two square occlusions causes a larger change to a unit's activation.Both a large-scale crowdsourced experiment and measurements with experts show that on average the extremely activating feature visualizations by Olah et al. (2017) indeed help humans on this task ($68 \pm 4$% accuracy; baseline performance without any visualizations is $60 \pm 3$%). However, they do not provide any substantial advantage over other visualizations (such as e.g. dataset samples), which yield similar performance ($66\pm3$% to $67 \pm3$% accuracy). Taken together, we propose an objective psychophysical task to quantify the benefit of unit-level interpretability methods for humans, and find no evidence that a widely-used feature visualization method provides humans with better "causal understanding" of unit activations than simple alternative visualizations.

        ----

        ## [897] Fixes That Fail: Self-Defeating Improvements in Machine-Learning Systems

        **Authors**: *Ruihan Wu, Chuan Guo, Awni Y. Hannun, Laurens van der Maaten*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/619427579e7b067421f6aa89d4a8990c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/619427579e7b067421f6aa89d4a8990c-Abstract.html)

        **Abstract**:

        Machine-learning systems such as self-driving cars or virtual assistants are composed of a large number of machine-learning models that recognize image content, transcribe speech, analyze natural language, infer preferences, rank options, etc. Models in these systems are often developed and trained independently, which raises an obvious concern: Can improving a machine-learning model make the overall system worse? We answer this question affirmatively by showing that improving a model can deteriorate the performance of downstream models, even after those downstream models are retrained. Such self-defeating improvements are the result of entanglement between the models in the system. We perform an error decomposition of systems with multiple machine-learning models, which sheds light on the types of errors that can lead to self-defeating improvements. We also present the results of experiments which show that self-defeating improvements emerge in a realistic stereo-based detection system for cars and pedestrians.

        ----

        ## [898] Coarse-to-fine Animal Pose and Shape Estimation

        **Authors**: *Chen Li, Gim Hee Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6195f47dcff14b8f242aa333cdb2703e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6195f47dcff14b8f242aa333cdb2703e-Abstract.html)

        **Abstract**:

        Most existing animal pose and shape estimation approaches reconstruct animal meshes with a parametric SMAL model. This is because the low-dimensional pose and shape parameters of the SMAL model makes it easier for deep networks to learn the high-dimensional animal meshes. However, the SMAL model is learned from scans of toy animals with limited pose and shape variations, and thus may not be able to represent highly varying real animals well. This may result in poor fittings of the estimated meshes to the 2D evidences, e.g. 2D keypoints or silhouettes.  To mitigate this problem, we propose a coarse-to-fine approach to reconstruct 3D animal mesh from a single image. The coarse estimation stage first estimates the pose, shape and translation parameters of the SMAL model. The estimated meshes are then used as a starting point by a graph convolutional network (GCN) to predict a per-vertex deformation in the refinement stage. This combination of SMAL-based and vertex-based representations benefits from both parametric and non-parametric representations. We design our mesh refinement GCN (MRGCN) as an encoder-decoder structure with hierarchical feature representations to overcome the limited receptive field of traditional GCNs. Moreover, we observe that the global image feature used by existing animal mesh reconstruction works is unable to capture detailed shape information for mesh refinement. We thus introduce a local feature extractor to retrieve a vertex-level feature and use it together with the global feature as the input of the MRGCN. We test our approach on the StanfordExtra dataset and achieve state-of-the-art results. Furthermore, we test the generalization capacity of our approach on the Animal Pose and BADJA datasets. Our code is available at the project website.

        ----

        ## [899] Meta-Learning Sparse Implicit Neural Representations

        **Authors**: *Jaeho Lee, Jihoon Tack, Namhoon Lee, Jinwoo Shin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/61b1fb3f59e28c67f3925f3c79be81a1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/61b1fb3f59e28c67f3925f3c79be81a1-Abstract.html)

        **Abstract**:

        Implicit neural representations are a promising new avenue of representing general signals by learning a continuous function that, parameterized as a neural network, maps the domain of a signal to its codomain; the mapping from spatial coordinates of an image to its pixel values, for example. Being capable of conveying fine details in a high dimensional signal, unboundedly of its domain, implicit neural representations ensure many advantages over conventional discrete representations. However, the current approach is difficult to scale for a large number of signals or a data set, since learning a neural representation---which is parameter heavy by itself---for each signal individually requires a lot of memory and computations. To address this issue, we propose to leverage a meta-learning approach in combination with network compression under a sparsity constraint, such that it renders a well-initialized sparse parameterization that evolves quickly to represent a set of unseen signals in the subsequent training. We empirically demonstrate that meta-learned sparse neural representations achieve a much smaller loss than dense meta-learned models with the same number of parameters, when trained to fit each signal using the same number of optimization steps.

        ----

        ## [900] Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation

        **Authors**: *Ho Kei Cheng, Yu-Wing Tai, Chi-Keung Tang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/61b4a64be663682e8cb037d9719ad8cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/61b4a64be663682e8cb037d9719ad8cd-Abstract.html)

        **Abstract**:

        This paper presents a simple yet effective approach to modeling space-time correspondences in the context of video object segmentation. Unlike most existing approaches, we establish correspondences directly between frames without re-encoding the mask features for every object, leading to a highly efficient and robust framework. With the correspondences, every node in the current query frame is inferred by aggregating features from the past in an associative fashion. We cast the aggregation process as a voting problem and find that the existing inner-product affinity leads to poor use of memory with a small (fixed) subset of memory nodes dominating the votes, regardless of the query. In light of this phenomenon, we propose using the negative squared Euclidean distance instead to compute the affinities. We validated that every memory node now has a chance to contribute, and experimentally showed that such diversified voting is beneficial to both memory efficiency and inference accuracy. The synergy of correspondence networks and diversified voting works exceedingly well, achieves new state-of-the-art results on both DAVIS and YouTubeVOS datasets while running significantly faster at 20+ FPS for multiple objects without bells and whistles.

        ----

        ## [901] Sparse Spiking Gradient Descent

        **Authors**: *Nicolas Perez Nieves, Dan F. M. Goodman*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/61f2585b0ebcf1f532c4d1ec9a7d51aa-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/61f2585b0ebcf1f532c4d1ec9a7d51aa-Abstract.html)

        **Abstract**:

        There is an increasing interest in emulating Spiking Neural Networks (SNNs) on neuromorphic computing devices due to their low energy consumption. Recent advances have allowed training SNNs to a point where they start to compete with traditional Artificial Neural Networks (ANNs) in terms of accuracy, while at the same time being energy efficient when run on neuromorphic hardware. However, the process of training SNNs is still based on dense tensor operations originally developed for ANNs which do not leverage the spatiotemporally sparse nature of SNNs. We present here the first sparse SNN backpropagation algorithm which achieves the same or better accuracy as current state of the art methods while being significantly faster and more memory efficient. We show the effectiveness of our method on real datasets of varying complexity (Fashion-MNIST, Neuromophic-MNIST and Spiking Heidelberg Digits) achieving a speedup in the backward pass of up to $150$x, and $85\%$ more memory efficient, without losing accuracy.

        ----

        ## [902] Rethinking Calibration of Deep Neural Networks: Do Not Be Afraid of Overconfidence

        **Authors**: *Deng-Bao Wang, Lei Feng, Min-Ling Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/61f3a6dbc9120ea78ef75544826c814e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/61f3a6dbc9120ea78ef75544826c814e-Abstract.html)

        **Abstract**:

        Capturing accurate uncertainty quantification of the prediction from deep neural networks is important in many real-world decision-making applications. A reliable predictor is expected to be accurate when it is confident about its predictions and indicate high uncertainty when it is likely to be inaccurate. However, modern neural networks have been found to be poorly calibrated, primarily in the direction of overconfidence. In recent years, there is a surge of research on model calibration by leveraging implicit or explicit regularization techniques during training, which obtain well calibration by avoiding overconfident outputs. In our study, we empirically found that despite the predictions obtained from these regularized models are better calibrated, they suffer from not being as calibratable, namely, it is harder to further calibrate their predictions with post-hoc calibration methods like temperature scaling and histogram binning. We conduct a series of empirical studies showing that overconfidence may not hurt final calibration performance if post-hoc calibration is allowed, rather, the penalty of confident outputs will compress the room of potential improvements in post-hoc calibration phase. Our experimental findings point out a new direction to improve calibration of DNNs by considering main training and post-hoc calibration as a unified framework.

        ----

        ## [903] Towards Efficient and Effective Adversarial Training

        **Authors**: *Gaurang Sriramanan, Sravanti Addepalli, Arya Baburaj, Venkatesh Babu R.*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/62889e73828c756c961c5a6d6c01a463-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/62889e73828c756c961c5a6d6c01a463-Abstract.html)

        **Abstract**:

        The vulnerability of Deep Neural Networks to adversarial attacks has spurred immense interest towards improving their robustness. However, present state-of-the-art adversarial defenses involve the use of 10-step adversaries during training, which renders them computationally infeasible for application to large-scale datasets. While the recent single-step defenses show promising direction, their robustness is not on par with multi-step training methods. In this work, we bridge this performance gap by introducing a novel Nuclear-Norm regularizer on network predictions to enforce function smoothing in the vicinity of data samples.  While prior works consider each data sample independently, the proposed regularizer uses the joint statistics of adversarial samples across a training minibatch to enhance optimization during both attack generation and training, obtaining state-of-the-art results amongst efficient defenses. We achieve further gains by incorporating exponential averaging of network weights over training iterations. We finally introduce a Hybrid training approach that combines the effectiveness of a two-step variant of the proposed defense with the efficiency of a single-step defense. We demonstrate superior results when compared to multi-step defenses such as TRADES and PGD-AT as well, at a significantly lower computational cost.

        ----

        ## [904] Intriguing Properties of Contrastive Losses

        **Authors**: *Ting Chen, Calvin Luo, Lala Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/628f16b29939d1b060af49f66ae0f7f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/628f16b29939d1b060af49f66ae0f7f8-Abstract.html)

        **Abstract**:

        We study three intriguing properties of contrastive learning. First, we generalize the standard contrastive loss to a broader family of losses, and we find that various instantiations of the generalized loss perform similarly under the presence of a multi-layer non-linear projection head. Second, we study if instance-based contrastive learning (with a global image representation) can learn well on images with multiple objects present. We find that meaningful hierarchical local features can be learned despite the fact that these objectives operate on global instance-level features. Finally, we study the phenomenon of feature suppression among competing features shared across augmented views, such as "color distribution" vs "object class". We construct datasets with explicit and controllable competing features, and show that, for contrastive learning, a few bits of easy-to-learn shared features can suppress, and even fully prevent, the learning of other sets of competing features. In scenarios where there are multiple objects in an image, the dominant object would suppress the learning of smaller objects. Existing contrastive learning methods critically rely on data augmentation to favor certain sets of features over others, and could suffer from learning saturation for scenarios where existing augmentations cannot fully address the feature suppression. This poses open challenges to existing contrastive learning techniques.

        ----

        ## [905] Detecting Moments and Highlights in Videos via Natural Language Queries

        **Authors**: *Jie Lei, Tamara L. Berg, Mohit Bansal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/62e0973455fd26eb03e91d5741a4a3bb-Abstract.html)

        **Abstract**:

        Detecting customized moments and highlights from videos given natural language (NL) user queries is an important but under-studied topic. One of the challenges in pursuing this direction is the lack of annotated data. To address this issue, we present the Query-based Video Highlights (QVHighlights) dataset. It consists of over 10,000 YouTube videos, covering a wide range of topics, from everyday activities and travel in lifestyle vlog videos to social and political activities in news videos. Each video in the dataset is annotated with: (1) a human-written free-form NL query, (2) relevant moments in the video w.r.t. the query, and (3) five-point scale saliency scores for all query-relevant clips. This comprehensive annotation enables us to develop and evaluate systems that detect relevant moments as well as salient highlights for diverse, flexible user queries. We also present a strong baseline for this task, Moment-DETR, a transformer encoder-decoder model that views moment retrieval as a direct set prediction problem, taking extracted video and query representations as inputs and predicting moment coordinates and saliency scores end-to-end. While our model does not utilize any human prior, we show that it performs competitively when compared to well-engineered architectures. With weakly supervised pretraining using ASR captions, Moment-DETR substantially outperforms previous methods. Lastly, we present several ablations and visualizations of Moment-DETR. Data and code is publicly available at https://github.com/jayleicn/moment_detr.

        ----

        ## [906] Stochastic optimization under time drift: iterate averaging, step-decay schedules, and high probability guarantees

        **Authors**: *Joshua Cutler, Dmitriy Drusvyatskiy, Zaïd Harchaoui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/62e7f2e090fe150ef8deb4466fdc81b3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/62e7f2e090fe150ef8deb4466fdc81b3-Abstract.html)

        **Abstract**:

        We consider the problem of minimizing a convex function that is evolving in time according to unknown and possibly stochastic dynamics. Such problems abound in the machine learning and signal processing literature, under the names of concept drift and stochastic tracking. We provide novel non-asymptotic convergence guarantees for stochastic algorithms with iterate averaging, focusing on bounds valid both in expectation and with high probability. Notably, we show that the tracking efficiency of the proximal stochastic gradient method depends only logarithmically on the initialization quality when equipped with a step-decay schedule.

        ----

        ## [907] Learning Stable Deep Dynamics Models for Partially Observed or Delayed Dynamical Systems

        **Authors**: *Andreas Schlaginhaufen, Philippe Wenk, Andreas Krause, Florian Dörfler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6332a8f62e3a9d5831724f2ffe55cae0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6332a8f62e3a9d5831724f2ffe55cae0-Abstract.html)

        **Abstract**:

        Learning how complex dynamical systems evolve over time is a key challenge in system identification. For safety critical systems, it is often crucial that the learned model is guaranteed to converge to some equilibrium point. To this end, neural ODEs regularized with neural Lyapunov functions are a promising approach when states are fully observed. For practical applications however, {\em partial observations} are the norm. As we will demonstrate, initialization of unobserved augmented states can become a key problem for neural ODEs. To alleviate this issue, we propose to augment the system's state with its history. Inspired by state augmentation in discrete-time systems, we thus obtain {\em neural delay differential equations}. Based on classical time delay stability analysis, we then show how to ensure stability of the learned models, and theoretically analyze our approach. Our experiments demonstrate its applicability to stable system identification of partially observed systems and learning a stabilizing feedback policy in delayed feedback control.

        ----

        ## [908] An Uncertainty Principle is a Price of Privacy-Preserving Microdata

        **Authors**: *John M. Abowd, Robert Ashmead, Ryan Cumings-Menon, Simson L. Garfinkel, Daniel Kifer, Philip Leclerc, William Sexton, Ashley Simpson, Christine Task, Pavel Zhuravlev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/639d79cc857a6c76c2723b7e014fccb0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/639d79cc857a6c76c2723b7e014fccb0-Abstract.html)

        **Abstract**:

        Privacy-protected microdata are often the desired output of a differentially private algorithm since  microdata is familiar and convenient for downstream users. However, there is a statistical price for this kind of convenience. We show that an uncertainty principle governs the trade-off between accuracy for a population of interest (``sum query'') vs. accuracy for its component sub-populations (``point queries''). Compared to differentially private query answering systems that are not required to produce microdata, accuracy can degrade by a logarithmic factor. For example, in the case of pure differential privacy, without the microdata requirement, one can provide noisy answers to the sum query and all point queries while guaranteeing that each answer has squared error $O(1/\epsilon^2)$. With the microdata requirement, one must choose between allowing an additional $\log^2(d)$ factor ($d$ is the number of point queries) for some point queries or allowing an extra $O(d^2)$ factor for the sum query. We present lower bounds for pure, approximate, and concentrated differential privacy. We propose mitigation strategies and create a collection of benchmark datasets that can be used for public study of this problem.

        ----

        ## [909] Fairness in Ranking under Uncertainty

        **Authors**: *Ashudeep Singh, David Kempe, Thorsten Joachims*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html)

        **Abstract**:

        Fairness has emerged as an important consideration in algorithmic decision making. Unfairness occurs when an agent with higher merit obtains a worse outcome than an agent with lower merit. Our central point is that a primary cause of unfairness is uncertainty. A principal or algorithm making decisions never has access to the agents' true merit, and instead uses proxy features that only imperfectly predict merit (e.g., GPA, star ratings, recommendation letters). None of these ever fully capture an agent's merit; yet existing approaches have mostly been defining fairness notions directly based on observed features and outcomes.Our primary point is that it is more principled to acknowledge and model the uncertainty explicitly. The role of observed features is to give rise to a posterior distribution of the agents' merits. We use this viewpoint to define a notion of approximate fairness in ranking. We call an algorithm $\phi$-fair (for $\phi \in [0,1]$) if it has the following property for all agents $x$ and all $k$: if agent $x$ is among the top $k$ agents with respect to merit with probability at least $\rho$ (according to the posterior merit distribution), then the algorithm places the agent among the top $k$ agents in its ranking with probability at least $\phi \rho$.We show how to compute rankings that optimally trade off approximate fairness against utility to the principal. In addition to the theoretical characterization, we present an empirical analysis of the potential impact of the approach in simulation studies. For real-world validation, we applied the approach in the context of a paper recommendation system that we built and fielded at the KDD 2020 conference.

        ----

        ## [910] Generalized Proximal Policy Optimization with Sample Reuse

        **Authors**: *James Queeney, Yannis Paschalidis, Christos G. Cassandras*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/63c4b1baf3b4460fa9936b1a20919bec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/63c4b1baf3b4460fa9936b1a20919bec-Abstract.html)

        **Abstract**:

        In real-world decision making tasks, it is critical for data-driven reinforcement learning methods to be both stable and sample efficient. On-policy methods typically generate reliable policy improvement throughout training, while off-policy methods make more efficient use of data through sample reuse. In this work, we combine the theoretically supported stability benefits of on-policy algorithms with the sample efficiency of off-policy algorithms. We develop policy improvement guarantees that are suitable for the off-policy setting, and connect these bounds to the clipping mechanism used in Proximal Policy Optimization. This motivates an off-policy version of the popular algorithm that we call Generalized Proximal Policy Optimization with Sample Reuse. We demonstrate both theoretically and empirically that our algorithm delivers improved performance by effectively balancing the competing goals of stability and sample efficiency.

        ----

        ## [911] Mosaicking to Distill: Knowledge Distillation from Out-of-Domain Data

        **Authors**: *Gongfan Fang, Yifan Bao, Jie Song, Xinchao Wang, Donglin Xie, Chengchao Shen, Mingli Song*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/63dc7ed1010d3c3b8269faf0ba7491d4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/63dc7ed1010d3c3b8269faf0ba7491d4-Abstract.html)

        **Abstract**:

        Knowledge distillation~(KD) aims to craft a compact student model that imitates the behavior of a pre-trained teacher in a target domain. Prior KD approaches, despite their gratifying results, have largely relied on the premise that \emph{in-domain} data is available to carry out the knowledge transfer. Such an assumption, unfortunately, in many cases violates the practical setting, since the original training data or even the data domain is often unreachable due to privacy or copyright reasons. In this paper, we attempt to tackle an ambitious task, termed as \emph{out-of-domain} knowledge distillation~(OOD-KD), which allows us to conduct KD using only OOD data that can be readily obtained at a very low cost. Admittedly,  OOD-KD is by nature a highly challenging task due to the agnostic domain gap. To this end, we introduce a handy yet surprisingly efficacious approach, dubbed as~\textit{MosaicKD}. The key insight behind MosaicKD lies in that, samples from various domains share common local patterns, even though their global semantic may vary significantly; these shared local patterns, in turn, can be re-assembled analogous to mosaic tiling, to approximate the in-domain data and to further alleviating the domain discrepancy. In MosaicKD, this is achieved through a four-player min-max game, in which a generator, a discriminator, a student network,  are collectively trained in an adversarial manner, partially under the guidance of a pre-trained teacher. We validate MosaicKD over {classification and semantic segmentation tasks} across various benchmarks, and demonstrate that it yields results much superior to the state-of-the-art counterparts on OOD data. Our code is available at \url{https://github.com/zju-vipa/MosaicKD}.

        ----

        ## [912] Batch Active Learning at Scale

        **Authors**: *Gui Citovsky, Giulia DeSalvo, Claudio Gentile, Lazaros Karydas, Anand Rajagopalan, Afshin Rostamizadeh, Sanjiv Kumar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64254db8396e404d9223914a0bd355d2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64254db8396e404d9223914a0bd355d2-Abstract.html)

        **Abstract**:

        The ability to train complex and highly effective models often requires an abundance of training data, which can easily become a bottleneck in cost, time, and computational resources. Batch active learning, which adaptively issues batched queries to a labeling oracle, is a common approach for addressing this problem. The practical benefits of batch sampling come with the downside of less adaptivity and the risk of sampling redundant examples within a batch -- a risk that grows with the batch size. In this work, we analyze an efficient active learning algorithm, which focuses on the large batch setting. In particular, we show that our sampling method, which combines notions of uncertainty and diversity, easily scales to batch sizes (100K-1M) several orders of magnitude larger than used in previous studies and provides significant improvements in model training efficiency compared to recent baselines. Finally, we provide an initial theoretical analysis, proving label complexity guarantees for a related sampling method, which we show is approximately equivalent to our sampling method in specific settings.

        ----

        ## [913] Joint Semantic Mining for Weakly Supervised RGB-D Salient Object Detection

        **Authors**: *Jingjing Li, Wei Ji, Qi Bi, Cheng Yan, Miao Zhang, Yongri Piao, Huchuan Lu, Li Cheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/642e92efb79421734881b53e1e1b18b6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/642e92efb79421734881b53e1e1b18b6-Abstract.html)

        **Abstract**:

        Training saliency detection models with weak supervisions, e.g., image-level tags or captions, is appealing as it removes the costly demand of per-pixel annotations. Despite the rapid progress of RGB-D saliency detection in fully-supervised setting, it however remains an unexplored territory when only weak supervision signals are available. This paper is set to tackle the problem of weakly-supervised RGB-D salient object detection. The key insight in this effort is the idea of maintaining per-pixel pseudo-labels with iterative refinements by reconciling the multimodal input signals in our joint semantic mining (JSM). Considering the large variations in the raw depth map and the lack of explicit pixel-level supervisions, we propose spatial semantic modeling (SSM) to capture saliency-specific depth cues from the raw depth and produce depth-refined pseudo-labels. Moreover, tags and captions are incorporated via a fill-in-the-blank training in our textual semantic modeling (TSM) to estimate the confidences of competing pseudo-labels. At test time, our model involves only a light-weight sub-network of the training pipeline, i.e., it requires only an RGB image as input, thus allowing efficient inference. Extensive evaluations demonstrate the effectiveness of our approach under the weakly-supervised setting. Importantly, our method could also be adapted to work in both fully-supervised and unsupervised paradigms. In each of these scenarios, superior performance has been attained by our approach with comparing to the state-of-the-art dedicated methods. As a by-product, a CapS dataset is constructed by augmenting existing benchmark training set with additional image tags and captions.

        ----

        ## [914] Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition

        **Authors**: *Yulin Wang, Rui Huang, Shiji Song, Zeyi Huang, Gao Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64517d8435994992e682b3e4aa0a0661-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64517d8435994992e682b3e4aa0a0661-Abstract.html)

        **Abstract**:

        Vision Transformers (ViT) have achieved remarkable success in large-scale image recognition. They split every 2D image into a fixed number of patches, each of which is treated as a token. Generally, representing an image with more tokens would lead to higher prediction accuracy, while it also results in drastically increased computational cost. To achieve a decent trade-off between accuracy and speed, the number of tokens is empirically set to 16x16 or 14x14. In this paper, we argue that every image has its own characteristics, and ideally the token number should be conditioned on each individual input. In fact, we have observed that there exist a considerable number of “easy” images which can be accurately predicted with a mere number of 4x4 tokens, while only a small fraction of “hard” ones need a finer representation. Inspired by this phenomenon, we propose a Dynamic Transformer to automatically configure a proper number of tokens for each input image. This is achieved by cascading multiple Transformers with increasing numbers of tokens, which are sequentially activated in an adaptive fashion at test time, i.e., the inference is terminated once a sufficiently confident prediction is produced. We further design efficient feature reuse and relationship reuse mechanisms across different components of the Dynamic Transformer to reduce redundant computations. Extensive empirical results on ImageNet, CIFAR-10, and CIFAR-100 demonstrate that our method significantly outperforms the competitive baselines in terms of both theoretical computational efficiency and practical inference speed. Code and pre-trained models (based on PyTorch and MindSpore) are available at https://github.com/blackfeather-wang/Dynamic-Vision-Transformer and https://github.com/blackfeather-wang/Dynamic-Vision-Transformer-MindSpore.

        ----

        ## [915] Contrastive Learning for Neural Topic Model

        **Authors**: *Thong Nguyen, Anh Tuan Luu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6467c327eaf8940b4dd07a08c63c5e85-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6467c327eaf8940b4dd07a08c63c5e85-Abstract.html)

        **Abstract**:

        Recent empirical studies show that adversarial topic models (ATM) can successfully capture semantic patterns of the document by differentiating a document with another dissimilar sample. However, utilizing that discriminative-generative architecture has two important drawbacks: (1) the architecture does not relate similar documents, which has the same document-word distribution of salient words; (2) it restricts the ability to integrate external information, such as sentiments of the document, which has been shown to benefit the training of neural topic model. To address those issues, we revisit the adversarial topic architecture in the view point of mathematical analysis, propose a novel approach to re-formulate discriminative goal as an optimization problem, and design a novel sampling method which facilitates the integration of external variables. The reformulation encourages the model to incorporate the relations among similar samples and enforces the constraint on the similarity among dissimilar ones; while the sampling method, which is based on the internal input and reconstructed output, helps inform the model of salient words contributing to the main topic. Experimental results show that our framework outperforms other state-of-the-art neural topic models in three common benchmark datasets that belong to various domains, vocabulary sizes, and document lengths in terms of topic coherence.

        ----

        ## [916] Learning in two-player zero-sum partially observable Markov games with perfect recall

        **Authors**: *Tadashi Kozuno, Pierre Ménard, Rémi Munos, Michal Valko*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/646c9941d7fb1bc793a7929328ae3f2f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/646c9941d7fb1bc793a7929328ae3f2f-Abstract.html)

        **Abstract**:

        We study the problem of learning a Nash equilibrium (NE) in an extensive game with imperfect information (EGII) through self-play. Precisely, we focus on two-player, zero-sum, episodic, tabular EGII under the \textit{perfect-recall} assumption where the only feedback is realizations of the game (bandit feedback). In particular the \textit{dynamics of the EGII is not known}---we can only access it by sampling or interacting with a game simulator. For this learning setting, we provide the Implicit Exploration Online Mirror Descent (IXOMD) algorithm. It is a model-free algorithm with a high-probability bound on convergence rate to the NE of order $1/\sqrt{T}$ where~$T$ is the number of played games. Moreover IXOMD is computationally efficient as it needs to perform the updates only along the sampled trajectory.

        ----

        ## [917] A Geometric Structure of Acceleration and Its Role in Making Gradients Small Fast

        **Authors**: *Jongmin Lee, Chanwoo Park, Ernest K. Ryu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/647c722bf90a49140184672e0d3723e3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/647c722bf90a49140184672e0d3723e3-Abstract.html)

        **Abstract**:

        Since Nesterov's seminal 1983 work, many accelerated first-order optimization methods have been proposed, but their analyses lacks a common unifying structure. In this work, we identify a geometric structure satisfied by a wide range of first-order accelerated methods. Using this geometric insight, we present several novel generalizations of accelerated methods. Most interesting among them is a method that reduces the squared gradient norm with $\mathcal{O}(1/K^4)$ rate in the prox-grad setup, faster than the $\mathcal{O}(1/K^3)$ rates of Nesterov's FGM or Kim and Fessler's FPGM-m.

        ----

        ## [918] ATISS: Autoregressive Transformers for Indoor Scene Synthesis

        **Authors**: *Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis, Andreas Geiger, Sanja Fidler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64986d86a17424eeac96b08a6d519059-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64986d86a17424eeac96b08a6d519059-Abstract.html)

        **Abstract**:

        The ability to synthesize realistic and diverse indoor furniture layouts automatically or based on partial input, unlocks many applications, from better interactive 3D tools to data synthesis for training and simulation. In this paper, we present ATISS, a novel autoregressive transformer architecture for creating diverse and plausible synthetic indoor environments, given only the room type and its floor plan. In contrast to prior work, which poses scene synthesis as sequence generation, our model generates rooms as unordered sets of objects. We argue that this formulation is more natural, as it makes ATISS generally useful beyond fully automatic room layout synthesis. For example, the same trained model can be used in interactive applications for general scene completion, partial room re-arrangement with any objects specified by the user, as well as object suggestions for any partial room. To enable this, our model leverages the permutation equivariance of the transformer when conditioning on the partial scene, and is trained to be permutation-invariant across object orderings. Our model is trained end-to-end as an autoregressive generative model using only labeled 3D bounding boxes as supervision. Evaluations on four room types in the 3D-FRONT dataset demonstrate that our model consistently generates plausible room layouts that are more realistic than existing methods.In addition, it has fewer parameters, is simpler to implement and train and runs up to 8 times faster than existing methods.

        ----

        ## [919] Generalized Depthwise-Separable Convolutions for Adversarially Robust and Efficient Neural Networks

        **Authors**: *Hassan Dbouk, Naresh R. Shanbhag*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/649adc59afdef2a8b9e943f94a04b02f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/649adc59afdef2a8b9e943f94a04b02f-Abstract.html)

        **Abstract**:

        Despite their tremendous successes, convolutional neural networks (CNNs) incur high computational/storage costs and are vulnerable to adversarial perturbations. Recent works on robust model compression address these challenges by combining model compression techniques with adversarial training. But these methods are unable to improve throughput (frames-per-second) on real-life hardware while simultaneously preserving robustness to adversarial perturbations. To overcome this problem, we propose the method of Generalized Depthwise-Separable (GDWS) convolution - an efficient, universal, post-training approximation of a standard 2D convolution. GDWS dramatically improves the throughput of a standard pre-trained network on real-life hardware while preserving its robustness. Lastly, GDWS is scalable to large problem sizes since it operates on pre-trained models and doesn't require any additional training. We establish the optimality of GDWS as a 2D convolution approximator and present exact algorithms for constructing optimal GDWS convolutions under complexity and error constraints. We demonstrate the effectiveness of GDWS via extensive experiments on CIFAR-10, SVHN, and ImageNet datasets. Our code can be found at https://github.com/hsndbk4/GDWS.

        ----

        ## [920] A Provably Efficient Model-Free Posterior Sampling Method for Episodic Reinforcement Learning

        **Authors**: *Christoph Dann, Mehryar Mohri, Tong Zhang, Julian Zimmert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/649d45bf179296e31731adfd4df25588-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/649d45bf179296e31731adfd4df25588-Abstract.html)

        **Abstract**:

        Thompson Sampling is one of the most effective methods for contextual bandits and has been generalized to posterior sampling for certain MDP settings. However, existing posterior sampling methods for reinforcement learning are limited by being model-based or lack worst-case theoretical guarantees beyond linear MDPs. This paper proposes a new model-free formulation of posterior sampling that applies to more general episodic reinforcement learning problems with theoretical guarantees. We introduce novel proof techniques to show that under suitable conditions, the worst-case regret of our posterior sampling method matches the best known results of optimization based methods. In the linear MDP setting with dimension, the regret of our algorithm scales linearly with the dimension as compared to a quadratic dependence of the existing posterior sampling-based exploration algorithms.

        ----

        ## [921] Fast Federated Learning in the Presence of Arbitrary Device Unavailability

        **Authors**: *Xinran Gu, Kaixuan Huang, Jingzhao Zhang, Longbo Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64be20f6dd1dd46adf110cf871e3ed35-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64be20f6dd1dd46adf110cf871e3ed35-Abstract.html)

        **Abstract**:

        Federated learning (FL) coordinates with numerous heterogeneous devices to collaboratively train a shared model while preserving user privacy. Despite its multiple advantages, FL faces new challenges. One challenge arises when devices drop out of the training process. In this case, the convergence of popular FL algorithms such as FedAvg is severely influenced by the straggling devices. To tackle this challenge, we study federated learning algorithms in the presence of arbitrary device unavailability and propose an algorithm named Memory-augmented Impatient Federated Averaging (MIFA). Our algorithm efficiently avoids excessive latency induced by inactive devices, and corrects the gradient bias using the memorized latest updates from them. We prove that MIFA achieves minimax optimal convergence rates on non-i.i.d. data for both strongly convex and non-convex smooth functions. We also provide an explicit characterization of the improvement over baseline algorithms through a case study, and validate the results by numerical experiments on real-world datasets.

        ----

        ## [922] On The Structure of Parametric Tournaments with Application to Ranking from Pairwise Comparisons

        **Authors**: *Vishnu Veerathu, Arun Rajkumar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64dafb11e52edd3cd840bf24e56ddce6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64dafb11e52edd3cd840bf24e56ddce6-Abstract.html)

        **Abstract**:

        We consider the classical problem of finding the minimum feedback arc set on tournaments (MFAST). The problem is NP-hard in general and we study it for important classes of tournaments that arise naturally in the problem of learning to rank from pairwise comparisons. Specifically, we consider tournaments classes that arise out of parametric preference matrices that can lead to cyclic preference relations. We investigate their structural properties via forbidden sub tournament configurations.  Towards this, we introduce \emph{Tournament Dimension} - a combinatorial parameter that characterizes the size of a forbidden configuration for rank $r$ tournament classes i.e., classes that arise out pairwise preference matrices which lead to rank $r$ skew-symmetric matrices under a suitable link function. Our main result is a polynomial-time algorithm - \texttt{Rank2Rank} - that solves the MFAST problem for the rank $2$ tournament class. This is achieved via a  geometric characterization that relies on our explicit construction of a forbidden configuration for this class.   Building on our understanding of the rank-$2$ tournament class, we propose a very general and flexible parametric pairwise preference model called the local-global model which subsumes the popular Bradley-Terry-Luce/Thurstone classes to capture locally cyclic as well as globally acyclic preference relations. We develop a polynomial-time algorithm - \texttt{BlockRank2Rank}- to solve the MFAST problem on the associated Block-Rank $2$ tournament class.  As an application, we study the problem of learning to rank from pairwise comparisons under the proposed local-global preference model. Exploiting our structural characterization, we propose  \texttt{PairwiseBlockRank} - a pairwise ranking algorithm for this class. We show sample complexity bounds of \texttt{PairwiseBlockRank}  to learn a good ranking under the proposed model.  Finally, we conduct experiments on synthetic and real-world datasets to show the efficacy of the proposed algorithm.

        ----

        ## [923] SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

        **Authors**: *Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, José M. Álvarez, Ping Luo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html)

        **Abstract**:

        We present SegFormer, a simple, efficient yet powerful semantic segmentation framework which unifies Transformers with lightweight multilayer perceptron (MLP) decoders. SegFormer has two appealing features: 1) SegFormer comprises a novel hierarchically structured Transformer encoder which outputs multiscale features. It does not need positional encoding, thereby avoiding the interpolation of positional codes which leads to decreased performance when the testing resolution differs from training. 2) SegFormer avoids complex decoders. The proposed MLP decoder aggregates information from different layers, and thus combining both local attention and global attention to render powerful representations. We show that this simple and lightweight design is the key to efficient segmentation on Transformers.  We scale our approach up to obtain a series of models from SegFormer-B0 to Segformer-B5, which reaches much better performance and efficiency than previous counterparts.For example, SegFormer-B4 achieves 50.3% mIoU on ADE20K with 64M parameters, being 5x smaller and 2.2% better than the previous best method. Our best model, SegFormer-B5, achieves 84.0% mIoU on Cityscapes validation set and shows excellent zero-shot robustness on Cityscapes-C.

        ----

        ## [924] Fairness via Representation Neutralization

        **Authors**: *Mengnan Du, Subhabrata Mukherjee, Guanchu Wang, Ruixiang Tang, Ahmed Hassan Awadallah, Xia Ben Hu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/64ff7983a47d331b13a81156e2f4d29d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/64ff7983a47d331b13a81156e2f4d29d-Abstract.html)

        **Abstract**:

        Existing bias mitigation methods for DNN models primarily work on learning debiased encoders. This process not only requires a lot of instance-level annotations for sensitive attributes, it also does not guarantee that all fairness sensitive information has been removed from the encoder. To address these limitations, we explore the following research question: Can we reduce the discrimination of DNN models by only debiasing the classification head, even with biased representations as inputs? To this end, we propose a new mitigation technique, namely, Representation Neutralization for Fairness (RNF) that achieves fairness by debiasing only the task-specific classification head of DNN models. To this end, we leverage samples with the same ground-truth label but different sensitive attributes, and use their neutralized representations to train the classification head of the DNN model. The key idea of RNF is to discourage the classification head from capturing spurious correlation between fairness sensitive information in encoder representations with specific class labels. To address low-resource settings with no access to sensitive attribute annotations, we leverage a bias-amplified model to generate proxy annotations for sensitive attributes. Experimental results over several benchmark datasets demonstrate our RNF framework to effectively reduce discrimination of DNN models with minimal degradation in task-specific performance.

        ----

        ## [925] Residual Relaxation for Multi-view Representation Learning

        **Authors**: *Yifei Wang, Zhengyang Geng, Feng Jiang, Chuming Li, Yisen Wang, Jiansheng Yang, Zhouchen Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6516c28727509c3db6280ae16254e916-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6516c28727509c3db6280ae16254e916-Abstract.html)

        **Abstract**:

        Multi-view methods learn representations by aligning multiple views of the same image and their performance largely depends on the choice of data augmentation. In this paper, we notice that some other useful augmentations, such as image rotation, are harmful for multi-view methods because they cause a semantic shift that is too large to be aligned well. This observation motivates us to relax the exact alignment objective to better cultivate stronger augmentations. Taking image rotation as a case study, we develop a generic approach, Pretext-aware Residual Relaxation (Prelax), that relaxes the exact alignment by allowing an adaptive residual vector between different views and encoding the semantic shift through pretext-aware learning. Extensive experiments on different backbones show that our method can not only improve multi-view methods with existing augmentations, but also benefit from stronger image augmentations like rotation.

        ----

        ## [926] Do Vision Transformers See Like Convolutional Neural Networks?

        **Authors**: *Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, Alexey Dosovitskiy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/652cf38361a209088302ba2b8b7f51e0-Abstract.html)

        **Abstract**:

        Convolutional neural networks (CNNs) have so far been the de-facto model for visual data. Recent work has shown that (Vision) Transformer models (ViT) can achieve comparable or even superior performance on image classification tasks. This raises a central question: how are Vision Transformers solving these tasks? Are they acting like convolutional networks, or learning entirely different visual representations? Analyzing the internal representation structure of ViTs and CNNs on image classification benchmarks, we find striking differences between the two architectures, such as ViT having more uniform representations across all layers. We explore how these differences arise, finding crucial roles played by self-attention, which enables early aggregation of global information, and ViT residual connections, which strongly propagate features from lower to higher layers. We study the ramifications for spatial localization, demonstrating ViTs successfully preserve input spatial information, with noticeable effects from different classification methods. Finally, we study the effect of (pretraining) dataset scale on intermediate features and transfer learning, and conclude with a discussion on connections to new architectures such as the MLP-Mixer.

        ----

        ## [927] Optimization-Based Algebraic Multigrid Coarsening Using Reinforcement Learning

        **Authors**: *Ali Taghibakhshi, Scott P. MacLachlan, Luke N. Olson, Matthew West*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6531b32f8d02fece98ff36a64a7c8260-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6531b32f8d02fece98ff36a64a7c8260-Abstract.html)

        **Abstract**:

        Large sparse linear systems of equations are ubiquitous in science and engineering, such as those arising from discretizations of partial differential equations. Algebraic multigrid (AMG) methods are one of the most common methods of solving such linear systems, with an extensive body of underlying mathematical theory. A system of linear equations defines a graph on the set of unknowns and each level of a multigrid solver requires the selection of an appropriate coarse graph along with restriction and interpolation operators that map to and from the coarse representation. The efficiency of the multigrid solver depends critically on this selection and many selection methods have been developed over the years. Recently, it has been demonstrated that it is possible to directly learn the AMG interpolation and restriction operators, given a coarse graph selection. In this paper, we consider the complementary problem of learning to coarsen graphs for a multigrid solver, a necessary step in developing fully learnable AMG methods. We propose a method using a reinforcement learning (RL) agent based on graph neural networks (GNNs), which can learn to perform graph coarsening on small planar training graphs and then be applied to unstructured large planar graphs, assuming bounded node degree. We demonstrate that this method can produce better coarse graphs than existing algorithms, even as the graph size increases and other properties of the graph are varied. We also propose an efficient inference procedure for performing graph coarsening that results in linear time complexity in graph size.

        ----

        ## [928] Delayed Propagation Transformer: A Universal Computation Engine towards Practical Control in Cyber-Physical Systems

        **Authors**: *Wenqing Zheng, Qiangqiang Guo, Hao Yang, Peihao Wang, Zhangyang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/654516d1b4df6917094de807156adc14-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/654516d1b4df6917094de807156adc14-Abstract.html)

        **Abstract**:

        Multi-agent control is a central theme in the Cyber-Physical Systems (CPS). However, current control methods either receive non-Markovian states due to insufficient sensing and decentralized design, or suffer from poor convergence. This paper presents the Delayed Propagation Transformer (DePT), a new transformer-based model that specializes in the global modeling of CPS while taking into account the immutable constraints from the physical world. DePT induces a cone-shaped spatial-temporal attention prior, which injects the information propagation and aggregation principles and enables a global view. With physical constraint inductive bias baked into its design, our DePT is ready to plug and play for a broad class of multi-agent systems. The experimental results on one of the most challenging CPS -- network-scale traffic signal control system in the open world -- show that our model outperformed the state-of-the-art expert methods on synthetic and real-world datasets. Our codes are released at: https://github.com/VITA-Group/DePT.

        ----

        ## [929] Explaining Latent Representations with a Corpus of Examples

        **Authors**: *Jonathan Crabbé, Zhaozhi Qian, Fergus Imrie, Mihaela van der Schaar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65658fde58ab3c2b6e5132a39fae7cb9-Abstract.html)

        **Abstract**:

        Modern machine learning models are complicated. Most of them rely on convoluted latent representations of their input to issue a prediction. To achieve greater transparency than a black-box that connects inputs to predictions, it is necessary to gain a deeper understanding of these latent representations. To that aim, we propose SimplEx: a user-centred method that provides example-based explanations with reference to a freely selected set of examples, called the corpus. SimplEx uses the corpus to improve the userâ€™s understanding of the latent space with post-hoc explanations answering two questions: (1) Which corpus examples explain the prediction issued for a given test example? (2) What features of these corpus examples are relevant for the model to relate them to the test example? SimplEx provides an answer by reconstructing the test latent representation as a mixture of corpus latent representations. Further, we propose a novel approach, the integrated Jacobian, that allows SimplEx to make explicit the contribution of each corpus feature in the mixture. Through experiments on tasks ranging from mortality prediction to image classification, we demonstrate that these decompositions are robust and accurate. With illustrative use cases in medicine, we show that SimplEx empowers the user by highlighting relevant patterns in the corpus that explain model representations. Moreover, we demonstrate how the freedom in choosing the corpus allows the user to have personalized explanations in terms of examples that are meaningful for them.

        ----

        ## [930] Explaining heterogeneity in medial entorhinal cortex with task-driven neural networks

        **Authors**: *Aran Nayebi, Alexander Attinger, Malcolm Campbell, Kiah Hardcastle, Isabel Low, Caitlin S. Mallory, Gabriel Mel, Ben Sorscher, Alex H. Williams, Surya Ganguli, Lisa M. Giocomo, Daniel L. K. Yamins*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/656f0dbf9392657eed7feefc486781fb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/656f0dbf9392657eed7feefc486781fb-Abstract.html)

        **Abstract**:

        Medial entorhinal cortex (MEC) supports a wide range of navigational and memory related behaviors.Well-known experimental results have revealed specialized cell types in MEC --- e.g. grid, border, and head-direction cells --- whose highly stereotypical response profiles are suggestive of the role they might play in supporting MEC functionality. However, the majority of MEC neurons do not exhibit stereotypical firing patterns.How should the response profiles of these more "heterogeneous" cells be described, and how do they contribute to behavior?In this work, we took a computational approach to addressing these questions.We first performed a statistical analysis that shows that heterogeneous MEC cells are just as reliable in their response patterns as the more stereotypical cell types, suggesting that they have a coherent functional role.Next, we evaluated a spectrum of candidate models in terms of their ability to describe the response profiles of both stereotypical and heterogeneous MEC cells.We found that recently developed task-optimized neural network models are substantially better than traditional grid cell-centric models at matching most MEC neuronal response profiles --- including those of grid cells themselves --- despite not being explicitly trained for this purpose.Specific choices of network architecture (such as gated nonlinearities and an explicit intermediate place cell representation) have an important effect on the ability of the model to generalize to novel scenarios, with the best of these models closely approaching the noise ceiling of the data itself.We then performed in silico experiments on this model to address questions involving the relative functional relevance of various cell types, finding that heterogeneous cells are likely to be just as involved in downstream functional outcomes (such as path integration) as grid and border cells.Finally, inspired by recent data showing that, going beyond their spatial response selectivity, MEC cells are also responsive to non-spatial rewards, we introduce a new MEC model that performs reward-modulated path integration.We find that this unified model matches neural recordings across all variable-reward conditions.Taken together, our results point toward a conceptually principled goal-driven modeling approach for moving future experimental and computational efforts beyond overly-simplistic single-cell stereotypes.

        ----

        ## [931] Beyond Smoothness: Incorporating Low-Rank Analysis into Nonparametric Density Estimation

        **Authors**: *Robert A. Vandermeulen, Antoine Ledent*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6591d327f6f731e589b0e869adadf940-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6591d327f6f731e589b0e869adadf940-Abstract.html)

        **Abstract**:

        The construction and theoretical analysis of the most popular universally consistent nonparametric density estimators hinge on one functional property: smoothness. In this paper we investigate the theoretical implications of incorporating a multi-view latent variable model, a type of low-rank model, into nonparametric density estimation. To do this we perform extensive analysis on histogram-style estimators that integrate a multi-view model. Our analysis culminates in showing that there exists a universally consistent histogram-style estimator that converges to any multi-view model with a finite number of Lipschitz continuous components at a rate of $\widetilde{O}(1/\sqrt[3]{n})$ in $L^1$ error. In contrast, the standard histogram estimator can converge at a rate slower than $1/\sqrt[d]{n}$ on the same class of densities. We also introduce a new nonparametric latent variable model based on the Tucker decomposition. A rudimentary implementation of our estimators experimentally demonstrates a considerable performance improvement over the standard histogram estimator. We also provide a thorough analysis of the sample complexity of our Tucker decomposition-based model and a variety of other results. Thus, our paper provides solid theoretical foundations for extending low-rank techniques to the nonparametric setting.

        ----

        ## [932] Multi-View Representation Learning via Total Correlation Objective

        **Authors**: *HyeongJoo Hwang, Geon-Hyeong Kim, Seunghoon Hong, Kee-Eung Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65a99bb7a3115fdede20da98b08a370f-Abstract.html)

        **Abstract**:

        Multi-View Representation Learning (MVRL) aims to discover a shared representation of observations from different views with the complex underlying correlation. In this paper, we propose a variational approach which casts MVRL as maximizing the amount of total correlation reduced by the representation, aiming to learn a shared latent representation that is informative yet succinct to capture the correlation among multiple views. To this end, we introduce a tractable surrogate objective function under the proposed framework, which allows our method to fuse and calibrate the observations in the representation space. From the information-theoretic perspective, we show that our framework subsumes existing multi-view generative models. Lastly, we show that our approach straightforwardly extends to the Partial MVRL (PMVRL) setting, where the observations are missing without any regular pattern. We demonstrate the effectiveness of our approach in the multi-view translation and classification tasks, outperforming strong baseline methods.

        ----

        ## [933] FACMAC: Factored Multi-Agent Centralised Policy Gradients

        **Authors**: *Bei Peng, Tabish Rashid, Christian Schröder de Witt, Pierre-Alexandre Kamienny, Philip H. S. Torr, Wendelin Boehmer, Shimon Whiteson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65b9eea6e1cc6bb9f0cd2a47751a186f-Abstract.html)

        **Abstract**:

        We propose FACtored Multi-Agent Centralised policy gradients (FACMAC), a new method for cooperative multi-agent reinforcement learning in both discrete and continuous action spaces. Like MADDPG, a popular multi-agent actor-critic method, our approach uses deep deterministic policy gradients to learn policies. However, FACMAC learns a centralised but factored critic, which combines per-agent utilities into the joint action-value function via a non-linear monotonic function, as in QMIX, a popular multi-agent $Q$-learning algorithm. However, unlike QMIX, there are no inherent constraints on factoring the critic. We thus also employ a nonmonotonic factorisation and empirically demonstrate that its increased representational capacity allows it to solve some tasks that cannot be solved with monolithic, or monotonically factored critics. In addition, FACMAC uses a centralised policy gradient estimator that optimises over the entire joint action space, rather than optimising over each agent's action space separately as in MADDPG. This allows for more coordinated policy changes and fully reaps the benefits of a centralised critic. We evaluate FACMAC on variants of the multi-agent particle environments, a novel multi-agent MuJoCo benchmark, and a challenging set of StarCraft II micromanagement tasks. Empirical results demonstrate FACMAC's superior performance over MADDPG and other baselines on all three domains.

        ----

        ## [934] EDGE: Explaining Deep Reinforcement Learning Policies

        **Authors**: *Wenbo Guo, Xian Wu, Usmann Khan, Xinyu Xing*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65c89f5a9501a04c073b354f03791b1f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65c89f5a9501a04c073b354f03791b1f-Abstract.html)

        **Abstract**:

        With the rapid development of deep reinforcement learning (DRL) techniques, there is an increasing need to understand and interpret DRL policies. While recent research has developed explanation methods to interpret how an agent determines its moves, they cannot capture the importance of actions/states to a game's final result. In this work, we propose a novel self-explainable model that augments a Gaussian process with a customized kernel function and an interpretable predictor. Together with the proposed model, we also develop a parameter learning procedure that leverages inducing points and variational inference to improve learning efficiency. Using our proposed model, we can predict an agent's final rewards from its game episodes and extract time step importance within episodes as strategy-level explanations for that agent. Through experiments on Atari and MuJoCo games, we verify the explanation fidelity of our method and demonstrate how to employ interpretation to understand agent behavior, discover policy vulnerabilities, remediate policy errors, and even defend against adversarial attacks.

        ----

        ## [935] Learning to Assimilate in Chaotic Dynamical Systems

        **Authors**: *Michael McCabe, Jed Brown*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65cc2c8205a05d7379fa3a6386f710e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65cc2c8205a05d7379fa3a6386f710e1-Abstract.html)

        **Abstract**:

        The accuracy of simulation-based forecasting in chaotic systems is heavily dependent on high-quality estimates of the system state at the beginning of the forecast. Data assimilation methods are used to infer these initial conditions by systematically combining noisy, incomplete observations and numerical models of system dynamics to produce highly effective estimation schemes. We introduce a self-supervised framework, which we call \textit{amortized assimilation}, for learning to assimilate in dynamical systems. Amortized assimilation combines deep learning-based denoising with differentiable simulation, using independent neural networks to assimilate specific observation types while connecting the gradient flow between these sub-tasks with differentiable simulation and shared recurrent memory. This hybrid architecture admits a self-supervised training objective which is minimized by an unbiased estimator of the true system state even in the presence of only noisy training data. Numerical experiments across several chaotic benchmark systems highlight the improved effectiveness of our approach compared to widely-used data assimilation methods.

        ----

        ## [936] Object-aware Contrastive Learning for Debiased Scene Representation

        **Authors**: *Sangwoo Mo, Hyunwoo Kang, Kihyuk Sohn, Chun-Liang Li, Jinwoo Shin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65d2ea03425887a717c435081cfc5dbb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65d2ea03425887a717c435081cfc5dbb-Abstract.html)

        **Abstract**:

        Contrastive self-supervised learning has shown impressive results in learning visual representations from unlabeled images by enforcing invariance against different data augmentations. However, the learned representations are often contextually biased to the spurious scene correlations of different objects or object and background, which may harm their generalization on the downstream tasks. To tackle the issue, we develop a novel object-aware contrastive learning framework that first (a) localizes objects in a self-supervised manner and then (b) debias scene correlations via appropriate data augmentations considering the inferred object locations. For (a), we propose the contrastive class activation map (ContraCAM), which finds the most discriminative regions (e.g., objects) in the image compared to the other images using the contrastively trained models. We further improve the ContraCAM to detect multiple objects and entire shapes via an iterative refinement procedure. For (b), we introduce two data augmentations based on ContraCAM, object-aware random crop and background mixup, which reduce contextual and background biases during contrastive self-supervised learning, respectively. Our experiments demonstrate the effectiveness of our representation learning framework, particularly when trained under multi-object images or evaluated under the background (and distribution) shifted images. Code is available at https://github.com/alinlab/object-aware-contrastive.

        ----

        ## [937] Evaluating Efficient Performance Estimators of Neural Architectures

        **Authors**: *Xuefei Ning, Changcheng Tang, Wenshuo Li, Zixuan Zhou, Shuang Liang, Huazhong Yang, Yu Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65d90fc6d307590b14e9e1800d4e8eab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65d90fc6d307590b14e9e1800d4e8eab-Abstract.html)

        **Abstract**:

        Conducting efficient performance estimations of neural architectures is a major challenge in neural architecture search (NAS). To reduce the architecture training costs in NAS, one-shot estimators (OSEs) amortize the architecture training costs by sharing the parameters of one supernet between all architectures. Recently, zero-shot estimators (ZSEs) that involve no training are proposed to further reduce the architecture evaluation cost. Despite the high efficiency of these estimators, the quality of such estimations has not been thoroughly studied. In this paper, we conduct an extensive and organized assessment of OSEs and ZSEs on five NAS benchmarks: NAS-Bench-101/201/301, and NDS ResNet/ResNeXt-A. Specifically, we employ a set of NAS-oriented criteria to study the behavior of OSEs and ZSEs, and reveal their biases and variances. After analyzing how and why the OSE estimations are unsatisfying, we explore how to mitigate the correlation gap of OSEs from three perspectives. Through our analysis, we give out suggestions for future application and development of efficient architecture performance estimators. Furthermore, the analysis framework proposed in our work could be utilized in future research to give a more comprehensive understanding of newly designed architecture performance estimators. The code is available at https://github.com/walkerning/aw_nas.

        ----

        ## [938] A-NeRF: Articulated Neural Radiance Fields for Learning Human Shape, Appearance, and Pose

        **Authors**: *Shih-Yang Su, Frank Yu, Michael Zollhöfer, Helge Rhodin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/65fc9fb4897a89789352e211ca2d398f-Abstract.html)

        **Abstract**:

        While deep learning reshaped the classical motion capture pipeline with feed-forward networks, generative models are required to recover fine alignment via iterative refinement. Unfortunately, the existing models are usually hand-crafted or learned in controlled conditions, only applicable to limited domains. We propose a method to learn a generative neural body model from unlabelled monocular videos by extending Neural Radiance Fields (NeRFs). We equip them with a skeleton to apply to time-varying and articulated motion. A key insight is that implicit models require the inverse of the forward kinematics used in explicit surface models. Our reparameterization defines spatial latent variables relative to the pose of body parts and thereby overcomes ill-posed inverse operations with an overparameterization. This enables learning volumetric body shape and appearance from scratch while jointly refining the articulated pose; all without ground truth labels for appearance, pose, or 3D shape on the input videos. When used for novel-view-synthesis and motion capture, our neural model improves accuracy on diverse datasets.

        ----

        ## [939] Differential Privacy Over Riemannian Manifolds

        **Authors**: *Matthew Reimherr, Karthik Bharath, Carlos Soto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6600e06fe9350b62c1e343504d4a7b86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6600e06fe9350b62c1e343504d4a7b86-Abstract.html)

        **Abstract**:

        In this work we consider the problem of releasing a differentially private statistical summary that resides on a Riemannian manifold.  We present an extension of the Laplace or K-norm mechanism that utilizes intrinsic distances and volumes on the manifold.  We also consider in detail the specific case where the summary is the Fr\'echet mean of data residing on a manifold.  We demonstrate that our mechanism is rate optimal and depends only on the dimension of the manifold, not on the dimension of any ambient space, while also showing how ignoring the manifold structure can decrease the utility of the sanitized summary.  We illustrate our framework in two examples of particular interest in statistics: the space of symmetric positive definite matrices, which is used for covariance matrices, and the sphere, which can be used as a space for modeling discrete distributions.

        ----

        ## [940] How can classical multidimensional scaling go wrong?

        **Authors**: *Rishi Sonthalia, Greg Van Buskirk, Benjamin Raichel, Anna C. Gilbert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/66121d1f782d29b62a286909165517bc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/66121d1f782d29b62a286909165517bc-Abstract.html)

        **Abstract**:

        Given a matrix $D$ describing the pairwise dissimilarities of a data set, a common task is to embed the data points into Euclidean space. The classical multidimensional scaling (cMDS) algorithm is a widespread method to do this. However, theoretical analysis of the robustness of the algorithm and an in-depth analysis of its performance on non-Euclidean metrics is lacking. In this paper, we derive a formula, based on the eigenvalues of a matrix obtained from $D$, for the Frobenius norm of the difference between $D$ and the metric $D_{\text{cmds}}$ returned by cMDS. This error analysis leads us to the conclusion that when the derived matrix has a significant number of negative eigenvalues, then $\|D-D_{\text{cmds}}\|_F$, after initially decreasing, willeventually increase as we increase the dimension. Hence, counterintuitively, the quality of the embedding degrades as we increase the dimension. We empirically verify that the Frobenius norm increases as we increase the dimension for a variety of non-Euclidean metrics. We also show on several benchmark datasets that this degradation in the embedding results in the classification accuracy of both simple (e.g., 1-nearest neighbor) and complex (e.g., multi-layer neural nets) classifiers decreasing as we increase the embedding dimension.Finally, our analysis leads us to a new efficiently computable algorithm that returns a matrix $D_l$ that is at least as close to the original distances as $D_t$ (the Euclidean metric closest in $\ell_2$ distance). While $D_l$ is not metric, when given as input to cMDS instead of $D$, it empirically results in solutions whose distance to $D$ does not increase when we increase the dimension and the classification accuracy degrades less than the cMDS solution.

        ----

        ## [941] Modeling Heterogeneous Hierarchies with Relation-specific Hyperbolic Cones

        **Authors**: *Yushi Bai, Zhitao Ying, Hongyu Ren, Jure Leskovec*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/662a2e96162905620397b19c9d249781-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/662a2e96162905620397b19c9d249781-Abstract.html)

        **Abstract**:

        Hierarchical relations are prevalent and indispensable for organizing human knowledge captured by a knowledge graph (KG). The key property of hierarchical relations is that they induce a partial ordering over the entities, which needs to be modeled in order to allow for hierarchical reasoning. However, current KG embeddings can model only a single global hierarchy (single global partial ordering) and fail to model multiple heterogeneous hierarchies that exist in a single KG. Here we present ConE (Cone Embedding), a KG embedding model that is able to simultaneously model multiple hierarchical as well as non-hierarchical relations in a knowledge graph. ConE embeds entities into hyperbolic cones and models relations as transformations between the cones. In particular, ConE uses cone containment constraints in different subspaces of the hyperbolic embedding space to capture multiple heterogeneous hierarchies. Experiments on standard knowledge graph benchmarks show that ConE obtains state-of-the-art performance on hierarchical reasoning tasks as well as knowledge graph completion task on hierarchical graphs. In particular, our approach yields new state-of-the-art Hits@1 of 45.3% on WN18RR and 16.1% on DDB14 (0.231 MRR). As for hierarchical reasoning task, our approach outperforms previous best results by an average of 20% across the three datasets.

        ----

        ## [942] Non-asymptotic Error Bounds for Bidirectional GANs

        **Authors**: *Shiao Liu, Yunfei Yang, Jian Huang, Yuling Jiao, Yang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/66be31e4c40d676991f2405aaecc6934-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/66be31e4c40d676991f2405aaecc6934-Abstract.html)

        **Abstract**:

        We derive nearly sharp bounds for the bidirectional GAN (BiGAN) estimation error under the Dudley distance between the latent joint distribution and the data joint distribution with appropriately specified  architecture of the neural networks used in the model. To the best of our knowledge, this is the first theoretical guarantee for the bidirectional GAN learning approach. An appealing feature of our results is that they do not assume the reference and the data distributions to have the same dimensions or these distributions to have bounded support. These assumptions are commonly assumed in the existing convergence analysis of the unidirectional GANs but may not be satisfied in practice. Our results are also applicable to the Wasserstein bidirectional GAN if the target distribution is assumed to have a bounded support. To prove these results, we construct neural network functions that push forward an empirical distribution to another arbitrary empirical distribution on a possibly different-dimensional space. We also develop a novel decomposition of the integral probability metric for the error analysis of bidirectional GANs. These basic theoretical results are of independent interest and can be applied to other related learning problems.

        ----

        ## [943] Confidence-Aware Imitation Learning from Demonstrations with Varying Optimality

        **Authors**: *Songyuan Zhang, Zhangjie Cao, Dorsa Sadigh, Yanan Sui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/670e8a43b246801ca1eaca97b3e19189-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/670e8a43b246801ca1eaca97b3e19189-Abstract.html)

        **Abstract**:

        Most existing imitation learning approaches assume the demonstrations are drawn from experts who are optimal, but relaxing this assumption enables us to use a wider range of data. Standard imitation learning may learn a suboptimal policy from demonstrations with varying optimality. Prior works use confidence scores or rankings to capture beneficial information from demonstrations with varying optimality, but they suffer from many limitations, e.g., manually annotated confidence scores or high average optimality of demonstrations. In this paper, we propose a general framework to learn from demonstrations with varying optimality that jointly learns the confidence score and a well-performing policy. Our approach, Confidence-Aware Imitation Learning (CAIL) learns a well-performing policy from confidence-reweighted demonstrations, while using an outer loss to track the performance of our model and to learn the confidence. We provide theoretical guarantees on the convergence of CAIL and evaluate its performance in both simulated and real robot experiments.Our results show that CAIL significantly outperforms other imitation learning methods from demonstrations with varying optimality. We further show that even without access to any optimal demonstrations, CAIL can still learn a successful policy, and outperforms prior work.

        ----

        ## [944] Answering Complex Causal Queries With the Maximum Causal Set Effect

        **Authors**: *Zachary Markovich*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/670f0c94cc5271fe6017eeffa642b7d3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/670f0c94cc5271fe6017eeffa642b7d3-Abstract.html)

        **Abstract**:

        The standard tools of causal inference have been developed to answer simple causal queries which can be easily formalized as a small number of statistical estimands in the context of a particular structural causal model (SCM); however, scientific theories often make diffuse predictions about a large number of causal variables. This article proposes a framework for parameterizing such complex causal queries as the maximum difference in causal effects associated with two sets of causal variables that have a researcher specified probability of occurring. We term this estimand the Maximum Causal Set Effect (MCSE) and develop an estimator for it that is asymptotically consistent and conservative in finite samples under assumptions that are standard in the causal inference literature. This estimator is also asymptotically normal and amenable to the non-parametric bootstrap, facilitating classical statistical inference about this novel estimand. We compare this estimator to more common latent variable approaches and find that it can uncover larger causal effects in both real world and simulated data.

        ----

        ## [945] Identifiability in inverse reinforcement learning

        **Authors**: *Haoyang Cao, Samuel N. Cohen, Lukasz Szpruch*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/671f0311e2754fcdd37f70a8550379bc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/671f0311e2754fcdd37f70a8550379bc-Abstract.html)

        **Abstract**:

        Inverse reinforcement learning attempts to reconstruct the reward function in a Markov decision problem, using observations of agent actions. As already observed in Russell [1998] the problem is ill-posed, and the reward function is not identifiable, even under the presence of perfect information about optimal behavior. We provide a resolution to this non-identifiability for problems with entropy regularization. For a given environment, we fully characterize the reward functions leading to a given policy and demonstrate that, given demonstrations of actions for the same reward under two distinct discount factors, or under sufficiently different environments, the unobserved reward can be recovered up to a constant. We also give general necessary and sufficient conditions for reconstruction of time-homogeneous rewards on finite horizons, and for action-independent rewards, generalizing recent results of Kim et al. [2021] and Fu et al. [2018].

        ----

        ## [946] A Probabilistic State Space Model for Joint Inference from Differential Equations and Data

        **Authors**: *Jonathan Schmidt, Nicholas Krämer, Philipp Hennig*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6734fa703f6633ab896eecbdfad8953a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6734fa703f6633ab896eecbdfad8953a-Abstract.html)

        **Abstract**:

        Mechanistic models with differential equations are a key component of scientific applications of machine learning. Inference in such models is usually computationally demanding because it involves repeatedly solving the differential equation. The main problem here is that the numerical solver is hard to combine with standard inference techniques. Recent work in probabilistic numerics has developed a new class of solvers for ordinary differential equations (ODEs) that phrase the solution process directly in terms of Bayesian filtering. We here show that this allows such methods to be combined very directly, with conceptual and numerical ease, with latent force models in the ODE itself. It then becomes possible to perform approximate Bayesian inference on the latent force as well as the ODE solution in a single, linear complexity pass of an extended Kalman filter / smoother — that is, at the cost of computing a single ODE solution. We demonstrate the expressiveness and performance of the algorithm by training, among others, a non-parametric SIRD model on data from the COVID-19 outbreak.

        ----

        ## [947] On Plasticity, Invariance, and Mutually Frozen Weights in Sequential Task Learning

        **Authors**: *Julian G. Zilly, Alessandro Achille, Andrea Censi, Emilio Frazzoli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6738fc33dd0b3906cd3626397cd247a7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6738fc33dd0b3906cd3626397cd247a7-Abstract.html)

        **Abstract**:

        Plastic neural networks have the ability to adapt to new tasks. However, in a continual learning setting, the configuration of parameters learned in previous tasks can severely reduce the adaptability to future tasks. In particular, we show that, when using weight decay, weights in successive layers of a deep network may become "mutually frozen". This has a double effect: on the one hand, it makes the network updates more invariant to nuisance factors, providing a useful bias for future tasks. On the other hand, it can prevent the network from learning new tasks that require significantly different features. In this context, we find that the local input sensitivity of a deep model is correlated with its ability to adapt, thus leading to an intriguing trade-off between adaptability and invariance when training a deep model more than once. We then show that a simple intervention that "resets" the mutually frozen connections can improve transfer learning on a variety of visual classification tasks. The efficacy of "resetting" itself depends on the size of the target dataset and the difference of the pre-training and target domains, allowing us to achieve state-of-the-art results on some datasets.

        ----

        ## [948] Provably Efficient Black-Box Action Poisoning Attacks Against Reinforcement Learning

        **Authors**: *Guanlin Liu, Lifeng Lai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/678004486c119599ed7d199f47da043a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/678004486c119599ed7d199f47da043a-Abstract.html)

        **Abstract**:

        Due to the broad range of applications of reinforcement learning (RL), understanding the effects of adversarial attacks against RL model is essential for the safe applications of this model. Prior theoretical works on adversarial attacks against RL mainly focus on either reward poisoning attacks or environment poisoning attacks. In this paper, we introduce a new class of attacks named action poisoning attacks, where an adversary can change the action signal selected by the agent. Compared with existing attack models, the attackerâ€™s ability in the proposed action poisoning attack model is more restricted, which brings some design challenges. We study the action poisoning attack in both white-box and black-box settings. We introduce an adaptive attack scheme called LCB-H, which works for most RL agents in the black-box setting. We prove that LCB-H attack can force any efficient RL agent, whose dynamic regret scales sublinearly with the total number of steps taken, to choose actions according to a policy selected by the attacker very frequently, with only sublinear cost. In addition, we apply LCB-H attack against a very popular model-free RL algorithm: UCB-H. We show that, even in black-box setting, by spending only logarithm cost, the proposed LCB-H attack scheme can force the UCB-H agent to choose actions according to the policy selected by the attacker very frequently.

        ----

        ## [949] Fast Approximation of the Sliced-Wasserstein Distance Using Concentration of Random Projections

        **Authors**: *Kimia Nadjahi, Alain Durmus, Pierre E. Jacob, Roland Badeau, Umut Simsekli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6786f3c62fbf9021694f6e51cc07fe3c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6786f3c62fbf9021694f6e51cc07fe3c-Abstract.html)

        **Abstract**:

        The Sliced-Wasserstein distance (SW) is being increasingly used in machine learning applications as an alternative to the Wasserstein distance and offers significant computational and statistical benefits. Since it is defined as an expectation over random projections, SW is commonly approximated by Monte Carlo. We adopt a new perspective to approximate SW by making use of the concentration of measure phenomenon: under mild assumptions, one-dimensional projections of a high-dimensional random vector are approximately Gaussian. Based on this observation, we develop a simple deterministic approximation for SW. Our method does not require sampling a number of random projections, and is therefore both accurate and easy to use compared to the usual Monte Carlo approximation. We derive nonasymptotical guarantees for our approach, and show that the approximation error goes to zero as the dimension increases, under a weak dependence condition on the data distribution. We validate our theoretical findings on synthetic datasets, and illustrate the proposed approximation on a generative modeling problem.

        ----

        ## [950] Causal Navigation by Continuous-time Neural Networks

        **Authors**: *Charles Vorbach, Ramin M. Hasani, Alexander Amini, Mathias Lechner, Daniela Rus*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67ba02d73c54f0b83c05507b7fb7267f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67ba02d73c54f0b83c05507b7fb7267f-Abstract.html)

        **Abstract**:

        Imitation learning enables high-fidelity, vision-based learning of policies within rich, photorealistic environments. However, such techniques often rely on traditional discrete-time neural models and face difficulties in generalizing to domain shifts by failing to account for the causal relationships between the agent and the environment. In this paper, we propose a theoretical and experimental framework for learning causal representations using continuous-time neural networks, specifically over their discrete-time counterparts. We evaluate our method in the context of visual-control learning of drones over a series of complex tasks, ranging from short- and long-term navigation, to chasing static and dynamic objects through photorealistic environments. Our results demonstrate that causal continuous-time deep models can perform robust navigation tasks, where advanced recurrent models fail. These models learn complex causal control representations directly from raw visual inputs and scale to solve a variety of tasks using imitation learning.

        ----

        ## [951] Global Convergence of Online Optimization for Nonlinear Model Predictive Control

        **Authors**: *Sen Na*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67d16d00201083a2b118dd5128dd6f59-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67d16d00201083a2b118dd5128dd6f59-Abstract.html)

        **Abstract**:

        We study a real-time iteration (RTI) scheme for solving online optimization problem appeared in nonlinear optimal control. The proposed RTI scheme modifies the existing RTI-based model predictive control (MPC) algorithm, by selecting the stepsize of each Newton step at each sampling time using a differentiable exact augmented Lagrangian. The scheme can adaptively select the penalty parameters of augmented Lagrangian on the fly, which are shown to be stabilized after certain time periods. We prove under generic assumptions that, by involving stepsize selection instead of always using a full Newton step (like what most of the existing RTIs do), the scheme converges globally: for any initial point, the KKT residuals of the subproblems converge to zero. A key step is to show that augmented Lagrangian keeps decreasing as horizon moves forward. We demonstrate the global convergence behavior of the proposed RTI scheme in a numerical experiment.

        ----

        ## [952] Argmax Flows and Multinomial Diffusion: Learning Categorical Distributions

        **Authors**: *Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forré, Max Welling*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67d96d458abdef21792e6d8e590244e7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67d96d458abdef21792e6d8e590244e7-Abstract.html)

        **Abstract**:

        Generative flows and diffusion models have been predominantly trained on ordinal data, for example natural images. This paper introduces two extensions of flows and diffusion for categorical data such as language or image segmentation: Argmax Flows and Multinomial Diffusion. Argmax Flows are defined by a composition of a continuous distribution (such as a normalizing flow), and an argmax function. To optimize this model, we learn a probabilistic inverse for the argmax that lifts the categorical data to a continuous space. Multinomial Diffusion gradually adds categorical noise in a diffusion process, for which the generative denoising process is learned. We demonstrate that our method outperforms existing dequantization approaches on text modelling and modelling on image segmentation maps in log-likelihood.

        ----

        ## [953] Learning with User-Level Privacy

        **Authors**: *Daniel Levy, Ziteng Sun, Kareem Amin, Satyen Kale, Alex Kulesza, Mehryar Mohri, Ananda Theertha Suresh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67e235e7f2fa8800d8375409b566e6b6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67e235e7f2fa8800d8375409b566e6b6-Abstract.html)

        **Abstract**:

        We propose and analyze algorithms to solve a range of learning tasks under user-level differential privacy constraints. Rather than guaranteeing only the privacy of individual samples, user-level DP protects a user's entire contribution ($m \ge 1$ samples), providing more stringent but more realistic protection against information leaks.  We show that for high-dimensional meanestimation, empirical risk minimization with smooth losses, stochastic convex optimization, and learning hypothesis classes with finite metric entropy, the privacy cost decreases as $O(1/\sqrt{m})$ as users provide more samples. In contrast, when increasing the number of users $n$, the privacy cost decreases at a faster $O(1/n)$ rate.  We complement these results with lower bounds showing the minimax optimality of our algorithms for mean estimation and stochastic convex optimization. Our algorithms rely on novel techniques for private mean estimation in arbitrary dimension with error scaling as the concentration radius $\tau$ of the distribution rather than the entire range.

        ----

        ## [954] Don't Generate Me: Training Differentially Private Generative Models with Sinkhorn Divergence

        **Authors**: *Tianshi Cao, Alex Bie, Arash Vahdat, Sanja Fidler, Karsten Kreis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67ed94744426295f96268f4ac1881b46-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67ed94744426295f96268f4ac1881b46-Abstract.html)

        **Abstract**:

        Although machine learning models trained on massive data have led to breakthroughs in several areas, their deployment in privacy-sensitive domains remains limited due to restricted access to data. Generative models trained with privacy constraints on private data can sidestep this challenge, providing indirect access to private data instead. We propose DP-Sinkhorn, a novel optimal transport-based generative method for learning data distributions from private data with differential privacy. DP-Sinkhorn minimizes the Sinkhorn divergence, a computationally efficient approximation to the exact optimal transport distance, between the model and data in a differentially private manner and uses a novel technique for controlling the bias-variance trade-off of gradient estimates. Unlike existing approaches for training differentially private generative models, which are mostly based on generative adversarial networks, we do not rely on adversarial objectives, which are notoriously difficult to optimize, especially in the presence of noise imposed by privacy constraints. Hence, DP-Sinkhorn is easy to train and deploy. Experimentally, we improve upon the state-of-the-art on multiple image modeling benchmarks and show differentially private synthesis of informative RGB images.

        ----

        ## [955] Keeping Your Eye on the Ball: Trajectory Attention in Video Transformers

        **Authors**: *Mandela Patrick, Dylan Campbell, Yuki M. Asano, Ishan Misra, Florian Metze, Christoph Feichtenhofer, Andrea Vedaldi, João F. Henriques*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/67f7fb873eaf29526a11a9b7ac33bfac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/67f7fb873eaf29526a11a9b7ac33bfac-Abstract.html)

        **Abstract**:

        In video transformers, the time dimension is often treated in the same way as the two spatial dimensions. However, in a scene where objects or the camera may move, a physical point imaged at one location in frame $t$ may be entirely unrelated to what is found at that location in frame $t+k$. These temporal correspondences should be modeled to facilitate learning about dynamic scenes. To this end, we propose a new drop-in block for video transformers - trajectory attention - that aggregates information along implicitly determined motion paths. We additionally propose a new method to address the quadratic dependence of computation and memory on the input size, which is particularly important for high resolution or long videos. While these ideas are useful in a range of settings, we apply them to the specific task of video action recognition with a transformer model and obtain state-of-the-art results on the Kinetics, Something-Something V2, and Epic-Kitchens datasets.

        ----

        ## [956] Variational Bayesian Optimistic Sampling

        **Authors**: *Brendan O'Donoghue, Tor Lattimore*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/680390c55bbd9ce416d1d69a9ab4760d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/680390c55bbd9ce416d1d69a9ab4760d-Abstract.html)

        **Abstract**:

        We consider online sequential decision problems where an agent must balance  exploration and exploitation. We derive a set of Bayesian `optimistic' policies  which, in the stochastic multi-armed bandit case, includes the Thompson sampling  policy. We provide a new analysis showing that any algorithm producing policies in the optimistic set enjoys $\tilde O(\sqrt{AT})$ Bayesian regret for a problem with $A$ actions after $T$ rounds. We extend the regret analysis for optimistic policies to bilinear saddle-point problems which include zero-sum matrix games and constrained bandits as special cases. In this case we show that Thompson sampling can produce policies outside of the optimistic set and suffer linear regret in some instances. Finding a policy inside the optimistic set amounts to solving a convex optimization problem and we call the resulting algorithm `variational Bayesian optimistic sampling' (VBOS). The procedure works for any posteriors, \ie, it does not require the posterior to have any special properties, such as log-concavity, unimodality, or smoothness. The variational view of the problem has many useful properties, including the ability to tune the exploration-exploitation tradeoff, add regularization, incorporate constraints, and linearly parameterize the policy.

        ----

        ## [957] Cross-modal Domain Adaptation for Cost-Efficient Visual Reinforcement Learning

        **Authors**: *Xiong-Hui Chen, Shengyi Jiang, Feng Xu, Zongzhang Zhang, Yang Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/68264bdb65b97eeae6788aa3348e553c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/68264bdb65b97eeae6788aa3348e553c-Abstract.html)

        **Abstract**:

        In visual-input sim-to-real scenarios, to overcome the reality gap between images rendered in simulators and those from the real world, domain adaptation, i.e., learning an aligned representation space between simulators and the real world, then training and deploying policies in the aligned representation, is a promising direction. Previous methods focus on same-modal domain adaptation. However, those methods require building and running simulators that render high-quality images, which can be difficult and costly. In this paper, we consider a more cost-efficient setting of visual-input sim-to-real where only low-dimensional states are simulated. We first point out that the objective of learning mapping functions in previous methods that align the representation spaces is ill-posed, prone to yield an incorrect mapping. When the mapping crosses modalities, previous methods are easier to fail. Our algorithm, Cross-mOdal Domain Adaptation with Sequential structure (CODAS), mitigates the ill-posedness by utilizing the sequential nature of the data sampling process in RL tasks. Experiments on MuJoCo and Hand Manipulation Suite tasks show that the agents deployed with our method achieve similar performance as it has in the source domain, while those deployed with previous methods designed for same-modal domain adaptation suffer a larger performance gap.

        ----

        ## [958] D2C: Diffusion-Decoding Models for Few-Shot Conditional Generation

        **Authors**: *Abhishek Sinha, Jiaming Song, Chenlin Meng, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/682e0e796084e163c5ca053dd8573b0c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/682e0e796084e163c5ca053dd8573b0c-Abstract.html)

        **Abstract**:

        Conditional generative models of high-dimensional images have many applications, but supervision signals from conditions to images can be expensive to acquire. This paper describes Diffusion-Decoding models with Contrastive representations (D2C), a paradigm for training unconditional variational autoencoders (VAE) for few-shot conditional image generation. D2C uses a learned diffusion-based prior over the latent representations to improve generation and contrastive self-supervised learning to improve representation quality. D2C can adapt to novel generation tasks, conditioned on labels or manipulation constraints, by learning from as few as 100 labeled examples. On conditional generation from new labels, D2C achieves superior performance over state-of-the-art VAEs and diffusion models. On conditional image manipulation, D2C generations are two orders of magnitude faster to produce over StyleGAN2 ones and are preferred by 50% - 60% of the human evaluators in a double-blind study. We release our code at https://github.com/jiamings/d2c.

        ----

        ## [959] Continual Auxiliary Task Learning

        **Authors**: *Matthew McLeod, Chunlok Lo, Matthew Schlegel, Andrew Jacobsen, Raksha Kumaraswamy, Martha White, Adam White*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/68331ff0427b551b68e911eebe35233b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/68331ff0427b551b68e911eebe35233b-Abstract.html)

        **Abstract**:

        Learning auxiliary tasks, such as multiple predictions about the world, can provide many benefits to reinforcement learning systems. A variety of off-policy learning algorithms have been developed to learn such predictions, but as yet there is little work on how to adapt the behavior to gather useful data for those off-policy predictions. In this work, we investigate a reinforcement learning system designed to learn a collection of auxiliary tasks, with a behavior policy learning to take actions to improve those auxiliary predictions. We highlight the inherent non-stationarity in this continual auxiliary task learning problem, for both prediction learners and the behavior learner. We develop an algorithm based on successor features that facilitates tracking under non-stationary rewards, and prove the separation into learning successor features and rewards provides convergence rate improvements. We conduct an in-depth study into the resulting multi-prediction learning system.

        ----

        ## [960] Two-step lookahead Bayesian optimization with inequality constraints

        **Authors**: *Yunxiang Zhang, Xiangyu Zhang, Peter I. Frazier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/685217557383cd194b4f10ae4b39eebf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/685217557383cd194b4f10ae4b39eebf-Abstract.html)

        **Abstract**:

        Recent advances in computationally efficient non-myopic Bayesian optimization offer improved query efficiency over traditional myopic methods like expected improvement, with only a modest increase in computational cost. These advances have been largely limited to unconstrained BO methods with only a few exceptions which require heavy computation. For instance, one existing multi-step lookahead constrained BO method (Lam & Willcox, 2017) relies on computationally expensive unreliable brute force derivative-free optimization of a Monte Carlo rollout acquisition function. Methods that use the reparameterization trick for more efficient derivative-based optimization of non-myopic acquisition functions in the unconstrained setting, like sample average approximation and infinitesimal perturbation analysis, do not extend: constraints introduce discontinuities in the sampled acquisition function surface. Moreover, we argue here that being non-myopic is even more important in constrained problems because fear of violating constraints pushes myopic methods away from sampling the boundary between feasible and infeasible regions, slowing the discovery of optimal solutions with tight constraints. In this paper, we propose a computationally efficient two-step lookahead constrained Bayesian optimization acquisition function (2-OPT-C) supporting both sequential and batch settings. To enable fast acquisition function optimization, we develop a novel likelihood ratio-based unbiased estimator of the gradient of the two-step optimal acquisition function that does not use the reparameterization trick. In numerical experiments, 2-OPT-C typically improves query efficiency by 2x or more over previous methods, and in some cases by 10x or more.

        ----

        ## [961] Learning with Labeling Induced Abstentions

        **Authors**: *Kareem Amin, Giulia DeSalvo, Afshin Rostamizadeh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/689041c2baed0f6d91050495d632d6e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/689041c2baed0f6d91050495d632d6e0-Abstract.html)

        **Abstract**:

        Consider a setting where we wish to automate an expensive task with a machine learning algorithm using a limited labeling resource. In such settings, examples routed for labeling are often out of scope for the machine learning algorithm. For example, in a spam detection setting, human reviewers not only provide labeled data but are such high-quality detectors of spam that examples routed to them no longer require machine evaluation. As a consequence, the distribution of examples routed to the machine is intimately tied to the process generating labels. We introduce a formalization of this setting, and give an algorithm that simultaneously learns a model and decides when to request a label by leveraging ideas from both the abstention and active learning literatures. We prove an upper bound on the algorithm's label complexity and a matching lower bound for any algorithm in this setting. We conduct a thorough set of experiments including an ablation study to test different components of our algorithm. We demonstrate the effectiveness of an efficient version of our algorithm over margin sampling on a variety of datasets.

        ----

        ## [962] SQALER: Scaling Question Answering by Decoupling Multi-Hop and Logical Reasoning

        **Authors**: *Mattia Atzeni, Jasmina Bogojeska, Andreas Loukas*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/68bd22864919297c8c8a8c32378e89b4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/68bd22864919297c8c8a8c32378e89b4-Abstract.html)

        **Abstract**:

        State-of-the-art approaches to reasoning and question answering over knowledge graphs (KGs) usually scale with the number of edges and can only be applied effectively on small instance-dependent subgraphs. In this paper, we address this issue by showing that multi-hop and more complex logical reasoning can be accomplished separately without losing expressive power. Motivated by this insight, we propose an approach to multi-hop reasoning that scales linearly with the number of relation types in the graph, which is usually significantly smaller than the number of edges or nodes. This produces a set of candidate solutions that can be provably refined to recover the solution to the original problem. Our experiments on knowledge-based question answering show that our approach solves the multi-hop MetaQA dataset, achieves a new state-of-the-art on the more challenging WebQuestionsSP, is orders of magnitude more scalable than competitive approaches, and can achieve compositional generalization out of the training distribution.

        ----

        ## [963] Out-of-Distribution Generalization in Kernel Regression

        **Authors**: *Abdulkadir Canatar, Blake Bordelon, Cengiz Pehlevan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/691dcb1d65f31967a874d18383b9da75-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/691dcb1d65f31967a874d18383b9da75-Abstract.html)

        **Abstract**:

        In real word applications, data generating process for training a machine learning model often differs from what the model encounters in the test stage. Understanding how and whether machine learning models generalize  under such distributional shifts have been a theoretical challenge. Here, we study generalization in kernel regression when the training and test distributions are different using methods from statistical physics. Using the replica method, we derive an analytical formula for the out-of-distribution  generalization error applicable to any kernel and real datasets. We identify an overlap matrix that quantifies the mismatch between distributions for a given kernel as a key determinant of generalization performance under distribution shift. Using our analytical expressions we elucidate various generalization phenomena including possible improvement in generalization when there is a mismatch. We develop procedures for optimizing training and test distributions for a given data budget to find best and worst case generalizations under the shift.  We present applications of our theory to real and synthetic datasets and for many kernels. We compare results of our theory applied to Neural Tangent Kernel with simulations of wide networks and show agreement. We analyze linear regression in further depth.

        ----

        ## [964] FL-WBC: Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective

        **Authors**: *Jingwei Sun, Ang Li, Louis DiValentin, Amin Hassanzadeh, Yiran Chen, Hai Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/692baebec3bb4b53d7ebc3b9fabac31b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/692baebec3bb4b53d7ebc3b9fabac31b-Abstract.html)

        **Abstract**:

        Federated learning (FL) is a popular distributed learning framework that trains a global model through iterative communications between a central server and edge devices. Recent works have demonstrated that FL is vulnerable to model poisoning attacks. Several server-based defense approaches (e.g. robust aggregation), have been proposed to mitigate such attacks. However, we empirically show that under extremely strong attacks, these defensive methods fail to guarantee the robustness of FL. More importantly, we observe that as long as the global model is polluted, the impact of attacks on the global model will remain in subsequent rounds even if there are no subsequent attacks. In this work, we propose a client-based defense, named White Blood Cell for Federated Learning (FL-WBC), which can mitigate model poisoning attacks that have already polluted the global model. The key idea of FL-WBC is to identify the parameter space where long-lasting attack effect on parameters resides and perturb that space during local training. Furthermore, we derive a certified robustness guarantee against model poisoning attacks and a convergence guarantee to FedAvg after applying our FL-WBC. We conduct experiments on FasionMNIST and CIFAR10 to evaluate the defense against state-of-the-art model poisoning attacks. The results demonstrate that our method can effectively mitigate model poisoning attack impact on the global model within 5 communication rounds with nearly no accuracy drop under both IID and Non-IID settings. Our defense is also complementary to existing server-based robust aggregation approaches and can further improve the robustness of FL under extremely strong attacks.

        ----

        ## [965] Chebyshev-Cantelli PAC-Bayes-Bennett Inequality for the Weighted Majority Vote

        **Authors**: *Yi-Shan Wu, Andrés R. Masegosa, Stephan Sloth Lorenzen, Christian Igel, Yevgeny Seldin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)

        **Abstract**:

        We present a new second-order oracle bound for the expected risk of a weighted majority vote. The bound is based on a novel parametric form of the Chebyshev-Cantelli inequality (a.k.a. one-sided Chebyshev’s), which is amenable to efficient minimization. The new form resolves the optimization challenge faced by prior oracle bounds based on the Chebyshev-Cantelli inequality, the C-bounds [Germain et al., 2015], and, at the same time, it improves on the oracle bound based on second order Markov’s inequality introduced by Masegosa et al. [2020]. We also derive a new concentration of measure inequality, which we name PAC-Bayes-Bennett, since it combines PAC-Bayesian bounding with Bennett’s inequality. We use it for empirical estimation of the oracle bound. The PAC-Bayes-Bennett inequality improves on the PAC-Bayes-Bernstein inequality of Seldin et al. [2012]. We provide an empirical evaluation demonstrating that the new bounds can improve on the work of Masegosa et al. [2020]. Both the parametric form of the Chebyshev-Cantelli inequality and the PAC-Bayes-Bennett inequality may be of independent interest for the study of concentration of measure in other domains.

        ----

        ## [966] A Multi-Implicit Neural Representation for Fonts

        **Authors**: *Pradyumna Reddy, Zhifei Zhang, Zhaowen Wang, Matthew Fisher, Hailin Jin, Niloy J. Mitra*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6948bd44c91acd2b54ecdd1b132f10fb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6948bd44c91acd2b54ecdd1b132f10fb-Abstract.html)

        **Abstract**:

        Fonts are ubiquitous across documents and come in a variety of styles.  They are either represented in a native vector format or rasterized to produce fixed resolution images. In the first case, the non-standard representation prevents benefiting from latest network architectures for neural representations; while, in the latter case, the rasterized representation, when encoded via networks, results in loss of data fidelity, as font-specific discontinuities like edges and corners are difficult to represent using neural networks. Based on the observation that complex fonts can be represented by a superposition of a set of simpler occupancy functions, we introduce multi-implicits to represent fonts as a permutation-invariant set of learned implict functions, without losing features (e.g., edges and corners). However, while multi-implicits locally preserve font features, obtaining supervision in the form of ground truth multi-channel signals is a problem in itself. Instead, we propose how to train such a representation with only local  supervision, while the proposed neural architecture directly finds globally consistent multi-implicits for font families. We extensively evaluate the proposed representation for various tasks including reconstruction, interpolation, and synthesis to demonstrate clear advantages with existing alternatives. Additionally, the representation naturally enables glyph completion, wherein a single characteristic font is used to synthesize a whole font family in the target style.

        ----

        ## [967] OctField: Hierarchical Implicit Functions for 3D Modeling

        **Authors**: *Jia-Heng Tang, Weikai Chen, Jie Yang, Bo Wang, Songrun Liu, Bo Yang, Lin Gao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/698d51a19d8a121ce581499d7b701668-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/698d51a19d8a121ce581499d7b701668-Abstract.html)

        **Abstract**:

        Recent advances in localized implicit functions have enabled neural implicit representation to be scalable to large scenes.However, the regular subdivision of 3D space employed by these approaches fails to take into account the sparsity of the surface occupancy and the varying granularities of geometric details. As a result, its memory footprint grows cubically with the input volume, leading to a prohibitive computational cost even at a moderately dense decomposition. In this work, we present a learnable hierarchical implicit representation for 3D surfaces, coded OctField, that allows high-precision encoding of intricate surfaces with low memory and computational budget. The key to our approach is an adaptive decomposition of 3D scenes that only distributes local implicit functions around the surface of interest. We achieve this goal by introducing a hierarchical octree structure to adaptively subdivide the 3D space according to the surface occupancy and the richness of part geometry. As octree is discrete and non-differentiable, we further propose a novel hierarchical network that models the subdivision of octree cells as a probabilistic process and recursively encodes and decodes both octree structure and surface geometry in a differentiable manner. We demonstrate the value of OctField for a range of shape modeling and reconstruction tasks, showing superiority over alternative approaches.

        ----

        ## [968] The Inductive Bias of Quantum Kernels

        **Authors**: *Jonas M. Kübler, Simon Buchholz, Bernhard Schölkopf*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69adc1e107f7f7d035d7baf04342e1ca-Abstract.html)

        **Abstract**:

        It has been hypothesized that quantum computers may lend themselves well to applications in machine learning. In the present work, we analyze function classes defined via quantum kernels. Quantum computers offer the possibility to efficiently compute inner products of exponentially large density operators that are classically hard to compute. However, having an exponentially large feature space renders the problem of generalization hard. Furthermore, being able to evaluate inner products in high dimensional spaces efficiently by itself does not guarantee a quantum advantage, as already classically tractable kernels can correspond to high- or infinite-dimensional reproducing kernel Hilbert spaces (RKHS).   We analyze the spectral properties of quantum kernels and find that we can expect an advantage if their RKHS is low dimensional and contains functions that are hard to compute classically. If the target function is known to lie in this class, this implies a quantum advantage, as the quantum computer can encode this inductive bias, whereas there is no classically efficient way to constrain the function class in the same way. However, we show that finding suitable quantum kernels is not easy because the kernel evaluation might require exponentially many measurements.   In conclusion, our message is a somewhat sobering one: we conjecture that quantum machine learning models can offer speed-ups only if we manage to encode knowledge about the problem at hand into quantum circuits, while encoding the same bias into a classical model would be hard. These situations may plausibly occur when learning on data generated by a quantum process, however, they appear to be harder to come by for classical datasets.

        ----

        ## [969] An Exponential Improvement on the Memorization Capacity of Deep Threshold Networks

        **Authors**: *Shashank Rajput, Kartik Sreenivasan, Dimitris S. Papailiopoulos, Amin Karbasi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69dd2eff9b6a421d5ce262b093bdab23-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69dd2eff9b6a421d5ce262b093bdab23-Abstract.html)

        **Abstract**:

        It is well known that modern deep neural networks are powerful enough to memorize datasets even when the labels have been randomized. Recently, Vershynin(2020) settled a long standing question by Baum(1988), proving that deep threshold networks can memorize $n$ points in $d$ dimensions using $\widetilde{\mathcal{O}}(e^{1/\delta^2}+\sqrt{n})$ neurons and $\widetilde{\mathcal{O}}(e^{1/\delta^2}(d+\sqrt{n})+n)$ weights, where $\delta$ is the minimum distance between the points. In this work, we improve the dependence on $\delta$ from exponential to almost linear, proving that $\widetilde{\mathcal{O}}(\frac{1}{\delta}+\sqrt{n})$ neurons and $\widetilde{\mathcal{O}}(\frac{d}{\delta}+n)$ weights are sufficient. Our construction uses Gaussian random weights only in the first layer, while all the subsequent layers use binary or integer weights. We also prove new lower bounds by connecting memorization in neural networks to the purely geometric problem of separating $n$ points on a sphere using hyperplanes.

        ----

        ## [970] Pretraining Representations for Data-Efficient Reinforcement Learning

        **Authors**: *Max Schwarzer, Nitarshan Rajkumar, Michael Noukhovitch, Ankesh Anand, Laurent Charlin, R. Devon Hjelm, Philip Bachman, Aaron C. Courville*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69eba34671b3ef1ef38ee85caae6b2a1-Abstract.html)

        **Abstract**:

        Data efficiency is a key challenge for deep reinforcement learning. We address this problem by using unlabeled data to pretrain an encoder which is then finetuned on a small amount of task-specific data. To encourage learning representations which capture diverse aspects of the underlying MDP, we employ a combination of latent dynamics modelling and unsupervised goal-conditioned RL. When limited to 100k steps of interaction on Atari games (equivalent to two hours of human experience), our approach significantly surpasses prior work combining offline representation pretraining with task-specific finetuning, and compares favourably with other pretraining methods that require orders of magnitude more data. Our approach shows particular promise when combined with larger models as well as more diverse, task-aligned observational data -- approaching human-level performance and data-efficiency on Atari in our best setting.

        ----

        ## [971] Universal Approximation Using Well-Conditioned Normalizing Flows

        **Authors**: *Holden Lee, Chirag Pabbaraju, Anish Prasad Sevekari, Andrej Risteski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69ec5030f78a9b735402d133317bf5f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69ec5030f78a9b735402d133317bf5f6-Abstract.html)

        **Abstract**:

        Normalizing flows are a widely used class of latent-variable generative models with a tractable likelihood. Affine-coupling models [Dinh et al., 2014, 2016] are a particularly common type of normalizing flows, for which the Jacobian of the latent-to-observable-variable transformation is triangular, allowing the likelihood to be computed in linear time. Despite the widespread usage of affine couplings, the special structure of the architecture makes understanding their representational power challenging. The question of universal approximation was only recently resolved by three parallel papers [Huang et al., 2020, Zhang et al., 2020, Koehler et al., 2020] – who showed reasonably regular distributions can be approximated arbitrarily well using affine couplings – albeit with networks with a nearly-singular Jacobian. As ill-conditioned Jacobians are an obstacle for likelihood-based training, the fundamental question remains: which distributions can be approximated using well-conditioned affine coupling flows? In this paper, we show that any log-concave distribution can be approximated using well-conditioned affine-coupling flows.  In terms of proof techniques, we uncover and leverage deep connections between affine coupling architectures, underdamped Langevin dynamics (a stochastic differential equation often used to sample from Gibbs measures) and Hénon maps (a structured dynamical system that appears in the study of symplectic diffeomorphisms). In terms of informing practice, we approximate a padded version of the input distribution with iid Gaussians – a strategy which Koehler et al. [2020] empirically observed to result in better-conditioned flows, but had hitherto no theoretical grounding. Our proof can thus be seen as providing theoretical evidence for the benefits of Gaussian padding when training normalizing flows.

        ----

        ## [972] On the Validity of Modeling SGD with Stochastic Differential Equations (SDEs)

        **Authors**: *Zhiyuan Li, Sadhika Malladi, Sanjeev Arora*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69f62956429865909921fa916d61c1f8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69f62956429865909921fa916d61c1f8-Abstract.html)

        **Abstract**:

        It is generally recognized that finite learning rate (LR), in contrast to infinitesimal LR, is important for good generalization in real-life deep nets. Most attempted explanations propose approximating finite-LR SGD with Itô Stochastic Differential Equations (SDEs), but formal justification for this approximation (e.g., Li et al., 2019) only applies to SGD with tiny LR. Experimental verification of the approximation appears computationally infeasible. The current paper clarifies the picture with the following contributions: (a) An efficient simulation algorithm SVAG that provably converges to the conventionally used Itô SDE approximation. (b) A theoretically motivated testable necessary condition for the SDE approximation and its most famous implication, the linear scaling rule (Goyal et al., 2017), to hold.(c) Experiments using this simulation to demonstrate that the previously proposed SDE approximation can meaningfully capture the training and generalization properties of common deep nets.

        ----

        ## [973] Proportional Participatory Budgeting with Additive Utilities

        **Authors**: *Dominik Peters, Grzegorz Pierczynski, Piotr Skowron*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/69f8ea31de0c00502b2ae571fbab1f95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/69f8ea31de0c00502b2ae571fbab1f95-Abstract.html)

        **Abstract**:

        We study voting rules for participatory budgeting, where a group of voters collectively decides which projects should be funded using a common budget. We allow the projects to have arbitrary costs, and the voters to have arbitrary additive valuations over the projects. We formulate two axioms that guarantee proportional representation to groups of voters with common interests. To the best of our knowledge, all known rules for participatory budgeting do not satisfy either of the two axioms; in addition we show that the most prominent proportional rule for committee elections, Proportional Approval Voting, cannot be adapted to arbitrary costs nor to additive valuations so that it would satisfy our axioms of proportionality. We construct a simple and attractive voting rule that satisfies one of our axioms (for arbitrary costs and arbitrary additive valuations), and that can be evaluated in polynomial time. We prove that our other stronger axiom is also satisfiable, though by a computationally more expensive and less natural voting rule.

        ----

        ## [974] Disentangling the Roles of Curation, Data-Augmentation and the Prior in the Cold Posterior Effect

        **Authors**: *Lorenzo Noci, Kevin Roth, Gregor Bachmann, Sebastian Nowozin, Thomas Hofmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a12d7ebc27cae44623468302c47ad74-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a12d7ebc27cae44623468302c47ad74-Abstract.html)

        **Abstract**:

        The “cold posterior effect” (CPE) in Bayesian deep learning describes the disturbing observation that the predictive performance of Bayesian neural networks can be significantly improved if the Bayes posterior is artificially sharpened using a temperature parameter T <1.  The CPE is problematic in theory and practice and since the effect was identified many researchers have proposed hypotheses to explain the phenomenon. However, despite this intensive research effort the effect remains poorly understood. In this work we provide novel and nuanced evidence relevant to existing explanations for the cold posterior effect, disentangling three hypotheses: 1. The dataset curation hypothesis of Aitchison (2020): we show empirically that the CPE does not arise in a real curated data set but can be produced in a controlled experiment with varying curation strength. 2. The data augmentation hypothesis of Izmailov et al. (2021) and Fortuin et al. (2021): we show empirically that data augmentation is sufficient but not necessary for the CPE to be present. 3. The bad prior hypothesis of Wenzel et al. (2020): we use a simple experiment evaluating the relative importance of the prior and the likelihood, strongly linking the CPE to the prior. Our results demonstrate how the CPE can arise in isolation from synthetic curation, data augmentation, and bad priors. Cold posteriors observed “in the wild” are therefore unlikely to arise from a single simple cause; as a result, we do not expect a simple “fix” for cold posteriors.

        ----

        ## [975] Sanity Checks for Lottery Tickets: Does Your Winning Ticket Really Win the Jackpot?

        **Authors**: *Xiaolong Ma, Geng Yuan, Xuan Shen, Tianlong Chen, Xuxi Chen, Xiaohan Chen, Ning Liu, Minghai Qin, Sijia Liu, Zhangyang Wang, Yanzhi Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a130f1dc6f0c829f874e92e5458dced-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a130f1dc6f0c829f874e92e5458dced-Abstract.html)

        **Abstract**:

        There have been long-standing controversies and inconsistencies over the experiment setup and criteria for identifying the "winning ticket" in literature. To reconcile such, we revisit the definition of lottery ticket hypothesis, with comprehensive and more rigorous conditions. Under our new definition, we show concrete evidence to clarify whether the winning ticket exists across the major DNN architectures and/or applications. Through extensive experiments, we perform quantitative analysis on the correlations between winning tickets and various experimental factors, and empirically study the patterns of our observations. We find that the key training hyperparameters, such as learning rate and training epochs, as well as the architecture characteristics such as capacities and residual connections, are all highly correlated with whether and when the winning tickets can be identified. Based on our analysis, we summarize a guideline for parameter settings in regards of specific architecture characteristics, which we hope to catalyze the research progress on the topic of lottery ticket hypothesis. Our codes are publicly available at: https://github.com/boone891214/sanity-check-LTH.

        ----

        ## [976] Collaborative Causal Discovery with Atomic Interventions

        **Authors**: *Raghavendra Addanki, Shiva Prasad Kasiviswanathan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a1a681b16826ba2e48fedb229db3b65-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a1a681b16826ba2e48fedb229db3b65-Abstract.html)

        **Abstract**:

        We introduce a new Collaborative Causal Discovery problem, through which we model a common scenario in which we have multiple independent entities each with their own causal graph, and the goal is to simultaneously learn all these causal graphs. We study this problem without the causal sufficiency assumption, using Maximal Ancestral Graphs (MAG) to model the causal graphs, and assuming that we have the ability to actively perform independent single vertex (or atomic) interventions on the entities. If the $M$ underlying (unknown) causal graphs of the entities satisfy a natural notion of clustering, we give algorithms that leverage this property and recovers all the causal graphs using roughly logarithmic in $M$ number of atomic interventions per entity. These are significantly fewer than $n$ atomic interventions per entity required to learn each causal graph separately, where $n$ is the number of observable nodes in the causal graph. We complement our results with a lower bound and discuss various extensions of our collaborative setting.

        ----

        ## [977] Towards optimally abstaining from prediction with OOD test examples

        **Authors**: *Adam Kalai, Varun Kanade*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a26c75d6a576c94654bfc4dda548c72-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a26c75d6a576c94654bfc4dda548c72-Abstract.html)

        **Abstract**:

        A common challenge across all areas of machine learning is that training data is not distributed like test data, due to natural shifts or adversarial examples; such examples are referred to as out-of-distribution (OOD) test examples. We consider a model where one may abstain from predicting, at a fixed cost. In particular, our transductive abstention algorithm takes labeled training examples and unlabeled test examples as input, and provides predictions with optimal prediction loss guarantees. The loss bounds match standard generalization bounds when test examples are i.i.d. from the training distribution, but add an additional term that is the cost of abstaining times the statistical distance between the train and test distribution (or the fraction of adversarial examples). For linear regression, we give a polynomial-time algorithm based on Celis-Dennis-Tapia optimization algorithms. For binary classification, we show how to efficiently implement it using a proper agnostic learner (i.e., an Empirical Risk Minimizer) for the class of interest. Our work builds on recent work of Goldwasser, Kalais, and Montasser (2020) who gave error and abstention guarantees for transductive binary classification.

        ----

        ## [978] TokenLearner: Adaptive Space-Time Tokenization for Videos

        **Authors**: *Michael S. Ryoo, A. J. Piergiovanni, Anurag Arnab, Mostafa Dehghani, Anelia Angelova*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a30e32e56fce5cf381895dfe6ca7b6f-Abstract.html)

        **Abstract**:

        In this paper, we introduce a novel visual representation learning which relies on a handful of adaptively learned tokens, and which is applicable to both image and video understanding tasks. Instead of relying on hand-designed splitting strategies to obtain visual tokens and processing a large number of densely sampled patches for attention, our approach learns to mine important tokens in visual data. This results in efficiently and effectively finding a few important visual tokens and enables modeling of pairwise attention between such tokens, over a longer temporal horizon for videos, or the spatial content in image frames. Our experiments demonstrate strong performance on several challenging benchmarks for video recognition tasks. Importantly, due to our tokens being adaptive, we accomplish competitive results at significantly reduced computational cost. We establish new state-of-the-arts on multiple video datasets, including Kinetics-400, Kinetics-600, Charades, and AViD.

        ----

        ## [979] Learning in Multi-Stage Decentralized Matching Markets

        **Authors**: *Xiaowu Dai, Michael I. Jordan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a571fe98a2ba453e84923b447d79cff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a571fe98a2ba453e84923b447d79cff-Abstract.html)

        **Abstract**:

        Matching markets are often organized in a multi-stage and decentralized manner.  Moreover, participants in real-world matching markets often have uncertain preferences. This article develops a framework for learning optimal strategies in such settings, based on a nonparametric statistical approach and variational analysis.  We propose an efficient algorithm, built upon concepts of "lower uncertainty bound" and "calibrated decentralized matching," for maximizing the participants' expected payoff. We show that there exists a welfare-versus-fairness trade-off that is characterized by the uncertainty level of acceptance.  Participants will strategically act in favor of a low uncertainty level to reduce competition and increase expected payoff. We prove that participants can be better off with multi-stage matching compared to single-stage matching. We demonstrate aspects of the theoretical predictions through simulations and an experiment using real data from college admissions.

        ----

        ## [980] Non-asymptotic convergence bounds for Wasserstein approximation using point clouds

        **Authors**: *Quentin Mérigot, Filippo Santambrogio, Clément Sarrazin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a61d423d02a1c56250dc23ae7ff12f3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a61d423d02a1c56250dc23ae7ff12f3-Abstract.html)

        **Abstract**:

        Several issues in machine learning and inverse problems require to generate discrete data, as if sampled from a model probabilitydistribution. A common way to do so relies on the construction of a uniform probability distribution over a set of $N$ points whichminimizes the Wasserstein distance to the model distribution. This minimization problem, where the unknowns are the positions of the atoms, is non-convex. Yet, in most cases, a suitably adjusted version of Lloyd's algorithm in which Voronoi cells are replaced by Power cells, leads to configurations with small Wasserstein error. This is surprising because, again, of the non-convex nature of the problem, which moreover admits spurious critical points. We provide explicit upper bounds for the convergence speed of this Lloyd-type algorithm, starting from a cloud of points sufficiently far from each other. This already works after one step of the iteration procedure, and similar bounds can be deduced, for the corresponding gradient descent. These bounds naturally lead to a sort of Poliak-Łojasiewicz inequality for the Wasserstein distance cost, with an error term depending on the distances between Dirac masses in the discrete distribution.

        ----

        ## [981] Understanding Interlocking Dynamics of Cooperative Rationalization

        **Authors**: *Mo Yu, Yang Zhang, Shiyu Chang, Tommi S. Jaakkola*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a711a119a8a7a9f877b5f379bfe9ea2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a711a119a8a7a9f877b5f379bfe9ea2-Abstract.html)

        **Abstract**:

        Selective rationalization explains the prediction of complex neural networks by finding a small subset of the input that is sufficient to predict the neural model output. The selection mechanism is commonly integrated into the model itself by specifying a two-component cascaded system consisting of a rationale generator, which makes a binary selection of the input features (which is the rationale), and a predictor, which predicts the output based only on the selected features. The components are trained jointly to optimize prediction performance. In this paper, we reveal a major problem with such cooperative rationalization paradigm --- model interlocking. Inter-locking arises when the predictor overfits to the features selected by the generator thus reinforcing the generator's selection even if the selected rationales are sub-optimal. The fundamental cause of the interlocking problem is that the rationalization objective to be minimized is concave with respect to the generatorâ€™s selection policy. We propose a new rationalization framework, called A2R, which introduces a third component into the architecture, a predictor driven by soft attention as opposed to selection. The generator now realizes both soft and hard attention over the features and these are fed into the two different predictors. While the generator still seeks to support the original predictor performance, it also minimizes a gap between the two predictors. As we will show theoretically, since the attention-based predictor exhibits a better convexity property, A2R can overcome the concavity barrier. Our experiments on two synthetic benchmarks and two real datasets demonstrate that A2R can significantly alleviate the interlock problem and find explanations that better align with human judgments.

        ----

        ## [982] Adversarial Robustness without Adversarial Training: A Teacher-Guided Curriculum Learning Approach

        **Authors**: *Anindya Sarkar, Anirban Sarkar, Sowrya Gali, Vineeth N. Balasubramanian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6a971e08a01e6676d0f1a6e0dacbbd67-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6a971e08a01e6676d0f1a6e0dacbbd67-Abstract.html)

        **Abstract**:

        Current SOTA adversarially robust models are mostly based on adversarial training (AT) and differ only by some regularizers either at inner maximization or outer minimization steps. Being repetitive in nature during the inner maximization step, they take a huge time to train. We propose a non-iterative method that enforces the following ideas during training. Attribution maps are more aligned to the actual object in the image for adversarially robust models compared to naturally trained models. Also, the allowed set of pixels to perturb an image (that changes model decision) should be restricted to the object pixels only, which reduces the attack strength by limiting the attack space. Our method achieves significant performance gains with a little extra effort (10-20%) over existing AT models and outperforms all other methods in terms of adversarial as well as natural accuracy. We have performed extensive experimentation with CIFAR-10, CIFAR-100, and TinyImageNet datasets and reported results against many popular strong adversarial attacks to prove the effectiveness of our method.

        ----

        ## [983] Tactical Optimism and Pessimism for Deep Reinforcement Learning

        **Authors**: *Ted Moskovitz, Jack Parker-Holder, Aldo Pacchiano, Michael Arbel, Michael I. Jordan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6abcc8f24321d1eb8c95855eab78ee95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6abcc8f24321d1eb8c95855eab78ee95-Abstract.html)

        **Abstract**:

        In recent years, deep off-policy actor-critic algorithms have become a dominant approach to reinforcement learning for continuous control. One of the primary drivers of this improved performance is the use of pessimistic value updates to address function approximation errors, which previously led to disappointing performance. However, a direct consequence of pessimism is reduced exploration, running counter to theoretical support for the efficacy of optimism in the face of uncertainty. So which approach is best? In this work, we show that the most effective degree of optimism can vary both across tasks and over the course of learning. Inspired by this insight, we introduce a novel deep actor-critic framework, Tactical Optimistic and Pessimistic (TOP) estimation, which switches between optimistic and pessimistic value learning online.  This is achieved by formulating the selection as a multi-arm bandit problem. We show in a series of continuous control tasks that TOP outperforms existing methods which rely on a fixed degree of optimism, setting a new state of the art in challenging pixel-based environments. Since our changes are simple to implement, we believe these insights can easily be incorporated into a multitude of off-policy algorithms.

        ----

        ## [984] Towards Hyperparameter-free Policy Selection for Offline Reinforcement Learning

        **Authors**: *Siyuan Zhang, Nan Jiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6add07cf50424b14fdf649da87843d01-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6add07cf50424b14fdf649da87843d01-Abstract.html)

        **Abstract**:

        How to select between policies and value functions produced by different training algorithms in offline reinforcement learning (RL)---which is crucial for hyperparameter tuning---is an important open question. Existing approaches based on off-policy evaluation (OPE) often require additional function approximation and hence hyperparameters, creating a chicken-and-egg situation.  In this paper, we design  hyperparameter-free algorithms for policy selection based on BVFT [XJ21], a recent theoretical advance in value-function selection, and demonstrate their effectiveness in discrete-action benchmarks such as Atari. To address performance degradation due to poor critics in continuous-action domains, we further combine BVFT with OPE to get the best of both worlds, and obtain a hyperparameter-tuning method for $Q$-function based OPE with theoretical guarantees as a side product.

        ----

        ## [985] FjORD: Fair and Accurate Federated Learning under heterogeneous targets with Ordered Dropout

        **Authors**: *Samuel Horváth, Stefanos Laskaridis, Mário Almeida, Ilias Leontiadis, Stylianos I. Venieris, Nicholas D. Lane*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6aed000af86a084f9cb0264161e29dd3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6aed000af86a084f9cb0264161e29dd3-Abstract.html)

        **Abstract**:

        Federated Learning (FL) has been gaining significant traction across different ML tasks, ranging from vision to keyboard predictions. In large-scale deployments, client heterogeneity is a fact and constitutes a primary problem for fairness, training performance and accuracy. Although significant efforts have been made into tackling statistical data heterogeneity, the diversity in the processing capabilities and network bandwidth of clients, termed system heterogeneity, has remained largely unexplored. Current solutions either disregard a large portion of available devices or set a uniform limit on the model's capacity, restricted by the least capable participants.In this work, we introduce Ordered Dropout, a mechanism that achieves an ordered, nested representation of knowledge in Neural Networks and enables the extraction of lower footprint submodels without the need for retraining. We further show that for linear maps our Ordered Dropout is equivalent to SVD.  We employ this technique, along with a self-distillation methodology, in the realm of FL in a framework called FjORD. FjORD alleviates the problem of client system heterogeneity by tailoring the model width to the client's capabilities. Extensive evaluation on both CNNs and RNNs across diverse modalities shows that FjORD consistently leads to significant performance gains over state-of-the-art baselines while maintaining its nested structure.

        ----

        ## [986] Optimal Uniform OPE and Model-based Offline Reinforcement Learning in Time-Homogeneous, Reward-Free and Task-Agnostic Settings

        **Authors**: *Ming Yin, Yu-Xiang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6b3c49bdba5be0d322334e30c459f8bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6b3c49bdba5be0d322334e30c459f8bd-Abstract.html)

        **Abstract**:

        This work studies the statistical limits of uniform convergence for offline policy evaluation (OPE) problems with model-based methods (for episodic MDP) and provides a unified framework towards optimal learning for several well-motivated offline tasks. Uniform OPE $\sup_\Pi|Q^\pi-\hat{Q}^\pi|<\epsilon$ is a stronger measure than the point-wise OPE and ensures offline learning when $\Pi$ contains all policies (the global class). In this paper, we establish an $\Omega(H^2 S/d_m\epsilon^2)$ lower bound (over model-based family) for the global uniform OPE and our main result establishes an upper bound of $\tilde{O}(H^2/d_m\epsilon^2)$ for the \emph{local} uniform convergence that applies to all \emph{near-empirically optimal} policies for the MDPs with \emph{stationary} transition. Here $d_m$ is the minimal marginal state-action probability. Critically, the highlight in achieving the optimal rate $\tilde{O}(H^2/d_m\epsilon^2)$ is our design of \emph{singleton absorbing MDP}, which is a new sharp analysis tool that works with the model-based approach. We generalize such a model-based framework to the new settings: offline task-agnostic and the offline reward-free with optimal complexity $\tilde{O}(H^2\log(K)/d_m\epsilon^2)$ ($K$ is the number of tasks) and $\tilde{O}(H^2S/d_m\epsilon^2)$ respectively. These results provide a unified solution for simultaneously solving different offline RL problems.

        ----

        ## [987] MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data

        **Authors**: *Zhibo Zhu, Ziqi Liu, Ge Jin, Zhiqiang Zhang, Lei Chen, Jun Zhou, Jianyong Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6b5754d737784b51ec5075c0dc437bf0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6b5754d737784b51ec5075c0dc437bf0-Abstract.html)

        **Abstract**:

        Time series forecasting is widely used in business intelligence, e.g., forecast stock market price, sales, and help the analysis of data trend. Most time series of interest are macroscopic time series that are aggregated from microscopic data. However, instead of directly modeling the macroscopic time series, rare literature studied the forecasting of macroscopic time series by leveraging data on the microscopic level. In this paper, we assume that the microscopic time series follow some unknown mixture probabilistic distributions. We theoretically show that as we identify the ground truth latent mixture components, the estimation of time series from each component could be improved because of lower variance, thus benefitting the estimation of macroscopic time series as well. Inspired by the power of Seq2seq and its variants on the modeling of time series data, we propose Mixture of Seq2seq (MixSeq), an end2end mixture model to cluster microscopic time series, where all the components come from a family of Seq2seq models parameterized by different parameters. Extensive experiments on both synthetic and real-world data show the superiority of our approach.

        ----

        ## [988] Pareto Domain Adaptation

        **Authors**: *Fangrui Lv, Jian Liang, Kaixiong Gong, Shuang Li, Chi Harold Liu, Han Li, Di Liu, Guoren Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6ba3af5d7b2790e73f0de32e5c8c1798-Abstract.html)

        **Abstract**:

        Domain adaptation (DA) attempts to transfer the knowledge from a labeled source domain to an unlabeled target domain that follows different distribution from the source. To achieve this, DA methods include a source classification objective to extract the source knowledge and a domain alignment objective to diminish the domain shift, ensuring knowledge transfer. Typically, former DA methods adopt some weight hyper-parameters to linearly combine the training objectives to form an overall objective. However, the gradient directions of these objectives may conflict with each other due to domain shift. Under such circumstances, the linear optimization scheme might decrease the overall objective value at the expense of damaging one of the training objectives, leading to restricted solutions. In this paper, we rethink the optimization scheme for DA from a gradient-based perspective. We propose a Pareto Domain Adaptation (ParetoDA) approach to control the overall optimization direction, aiming to cooperatively optimize all training objectives. Specifically, to reach a desirable solution on the target domain, we design a surrogate loss mimicking target classification. To improve target-prediction accuracy to support the mimicking, we propose a target-prediction refining mechanism which exploits domain labels via Bayesâ€™ theorem. On the other hand, since prior knowledge of weighting schemes for objectives is often unavailable to guide optimization to approach the optimal solution on the target domain, we propose a dynamic preference mechanism to dynamically guide our cooperative optimization by the gradient of the surrogate loss on a held-out unlabeled target dataset. Our theoretical analyses show that the held-out data can guide but will not be over-fitted by the optimization. Extensive experiments on image classification and semantic segmentation benchmarks demonstrate the effectiveness of ParetoDA

        ----

        ## [989] Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals

        **Authors**: *Lang Liu, Krishna Pillutla, Sean Welleck, Sewoong Oh, Yejin Choi, Zaïd Harchaoui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6bf733bb7f81e866306e9b5f012419cb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6bf733bb7f81e866306e9b5f012419cb-Abstract.html)

        **Abstract**:

        The spectacular success of deep generative models calls for quantitative tools to measure their statistical performance. Divergence frontiers have recently been proposed as an evaluation framework for generative models, due to their ability to measure the quality-diversity trade-off inherent to deep generative modeling. We establish non-asymptotic bounds on the sample complexity of divergence frontiers. We also introduce frontier integrals which provide summary statistics of divergence frontiers. We show how smoothed estimators such as Good-Turing or Krichevsky-Trofimov can overcome the missing mass problem and lead to faster rates of convergence. We illustrate the theoretical results with numerical examples from natural language processing and computer vision.

        ----

        ## [990] Consistency Regularization for Variational Auto-Encoders

        **Authors**: *Samarth Sinha, Adji Bousso Dieng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c19e0a6da12dc02239312f151072ddd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c19e0a6da12dc02239312f151072ddd-Abstract.html)

        **Abstract**:

        Variational Auto-Encoders (VAEs) are a powerful approach to unsupervised learning. They enable scalable approximate posterior inference in latent-variable models using variational inference. A VAE posits a variational family parameterized by a deep neural network---called an encoder---that takes data as input. This encoder is shared across all the observations, which amortizes the cost of inference. However the encoder of a VAE has the undesirable property that it maps a given observation and a semantics-preserving transformation of it to different latent representations. This "inconsistency" of the encoder lowers the quality of the learned representations, especially for downstream tasks, and also negatively affects generalization. In this paper, we propose a regularization method to enforce consistency in VAEs. The idea is to minimize the Kullback-Leibler (KL) divergence between the variational distribution when conditioning on the observation and the variational distribution when conditioning on a random semantics-preserving transformation of this observation. This regularization is applicable to any VAE. In our experiments we apply it to four different VAE variants on several benchmark datasets and found it always improves the quality of the learned representations but also leads to better generalization. In particular, when applied to the Nouveau VAE (NVAE), our regularization method yields state-of-the-art performance on MNIST, CIFAR-10, and CELEBA. We also applied our method to 3D data and found it learns representations of superior quality as measured by accuracy on a downstream classification task. Finally, we show our method can even outperform the triplet loss, an advanced and popular contrastive learning-based method for representation learning.

        ----

        ## [991] Score-based Generative Neural Networks for Large-Scale Optimal Transport

        **Authors**: *Grady Daniels, Tyler Maunu, Paul Hand*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c2e49911b68d315555d5b3eb0dd45bf-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c2e49911b68d315555d5b3eb0dd45bf-Abstract.html)

        **Abstract**:

        We consider the fundamental problem of sampling the optimal transport coupling between given source and target distributions. In certain cases, the optimal transport plan takes the form of a one-to-one mapping from the source support to the target support, but learning or even approximating such a map is computationally challenging for large and high-dimensional datasets due to the high cost of linear programming routines and an intrinsic curse of dimensionality. We study instead the Sinkhorn problem, a regularized form of optimal transport whose solutions are couplings between the source and the target distribution. We introduce a novel framework for learning the Sinkhorn coupling between two distributions in the form of a score-based generative model. Conditioned on source data, our procedure iterates Langevin Dynamics to sample target data according to the regularized optimal coupling. Key to this approach is a neural network parametrization of the Sinkhorn problem, and we prove convergence of gradient descent with respect to network parameters in this formulation. We demonstrate its empirical success on a variety of large scale optimal transport tasks.

        ----

        ## [992] Interactive Label Cleaning with Example-based Explanations

        **Authors**: *Stefano Teso, Andrea Bontempelli, Fausto Giunchiglia, Andrea Passerini*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c349155b122aa8ad5c877007e05f24f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c349155b122aa8ad5c877007e05f24f-Abstract.html)

        **Abstract**:

        We tackle sequential learning under label noise in applications where a human supervisor can be queried to relabel suspicious examples. Existing approaches are flawed, in that they only relabel incoming examples that look "suspicious" to the model. As a consequence, those mislabeled examples that elude (or don't undergo) this cleaning step end up tainting the training data and the model with no further chance of being cleaned. We propose CINCER, a novel approach that cleans both new and past data by identifying \emph{pairs of mutually incompatible examples}. Whenever it detects a suspicious example, CINCER identifies a counter-example in the training set that - according to the model - is maximally incompatible with the suspicious example, and asks the annotator to relabel either or both examples, resolving this possible inconsistency. The counter-examples are chosen to be maximally incompatible, so to serve as \emph{explanations} of the model's suspicion, and highly influential, so to convey as much information as possible if relabeled. CINCER achieves this by leveraging an efficient and robust approximation of influence functions based on the Fisher information matrix (FIM). Our extensive empirical evaluation shows that clarifying the reasons behind the model's suspicions by cleaning the counter-examples helps in acquiring substantially better data and models, especially when paired with our FIM approximation.

        ----

        ## [993] Gradient Descent on Two-layer Nets: Margin Maximization and Simplicity Bias

        **Authors**: *Kaifeng Lyu, Zhiyuan Li, Runzhe Wang, Sanjeev Arora*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c351da15b5e8a743a21ee96a86e25df-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c351da15b5e8a743a21ee96a86e25df-Abstract.html)

        **Abstract**:

        The generalization mystery of overparametrized deep nets has motivated efforts to understand how gradient descent (GD) converges to low-loss solutions that generalize well. Real-life neural networks are initialized from small random values and trained with cross-entropy loss for classification (unlike the "lazy" or "NTK" regime of training where analysis was more successful), and a recent sequence of results (Lyu and Li, 2020; Chizat and Bach, 2020; Ji and Telgarsky, 2020) provide theoretical evidence that GD may converge to the "max-margin" solution with zero loss, which presumably generalizes well. However, the global optimality of margin is proved only in some settings where neural nets are infinitely or exponentially wide. The current paper is able to establish this global optimality for two-layer Leaky ReLU nets trained with gradient flow on linearly separable and symmetric data, regardless of the width. The analysis also gives some theoretical justification for recent empirical findings (Kalimeris et al., 2019) on the so-called simplicity bias of GD towards linear or other "simple" classes of solutions, especially early in training. On the pessimistic side, the paper suggests that such results are fragile. A simple data manipulation can make gradient flow converge to a linear classifier with suboptimal margin.

        ----

        ## [994] Glance-and-Gaze Vision Transformer

        **Authors**: *Qihang Yu, Yingda Xia, Yutong Bai, Yongyi Lu, Alan L. Yuille, Wei Shen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c524f9d5d7027454a783c841250ba71-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c524f9d5d7027454a783c841250ba71-Abstract.html)

        **Abstract**:

        Recently, there emerges a series of vision Transformers, which show superior performance with a more compact model size than conventional convolutional neural networks, thanks to the strong ability of Transformers to model long-range dependencies. However, the advantages of vision Transformers also come with a price: Self-attention, the core part of Transformer, has a quadratic complexity to the input sequence length. This leads to a dramatic increase of computation and memory cost with the increase of sequence length, thus introducing difficulties when applying Transformers to the vision tasks that require dense predictions based on high-resolution feature maps.In this paper, we propose a new vision Transformer, named Glance-and-Gaze Transformer (GG-Transformer), to address the aforementioned issues. It is motivated by the Glance and Gaze behavior of human beings when recognizing objects in natural scenes, with the ability to efficiently model both long-range dependencies and local context. In GG-Transformer, the Glance and Gaze behavior is realized by two parallel branches: The Glance branch is achieved by performing self-attention on the adaptively-dilated partitions of the input, which leads to a linear complexity while still enjoying a global receptive field; The Gaze branch is implemented by a simple depth-wise convolutional layer, which compensates local image context to the features obtained by the Glance mechanism. We empirically demonstrate our method achieves consistently superior performance over previous state-of-the-art Transformers on various vision tasks and benchmarks.

        ----

        ## [995] Stochastic $L^\natural$-convex Function Minimization

        **Authors**: *Haixiang Zhang, Zeyu Zheng, Javad Lavaei*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6c81c83c4bd0b58850495f603ab45a93-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6c81c83c4bd0b58850495f603ab45a93-Abstract.html)

        **Abstract**:

        We study an extension of the stochastic submodular minimization problem, namely, the stochastic $L^\natural$-convex minimization problem. We develop the first polynomial-time algorithms that return a near-optimal solution with high probability. We design a novel truncation operation to further reduce the computational complexity of the proposed algorithms. When applied to a stochastic submodular function, the computational complexity of the proposed algorithms is lower than that of the existing stochastic submodular minimization algorithms. In addition, we provide a strongly polynomial approximate algorithm. The algorithm execution also does not require any prior knowledge about the objective function except the $L^\natural$-convexity. A lower bound on the computational complexity that is required to achieve a high probability error bound is also derived. Numerical experiments are implemented to demonstrate the efficiency of our theoretical findings.

        ----

        ## [996] Self-Supervised GANs with Label Augmentation

        **Authors**: *Liang Hou, Huawei Shen, Qi Cao, Xueqi Cheng*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6cb5da3513bd26085ee3fad631ebb37a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6cb5da3513bd26085ee3fad631ebb37a-Abstract.html)

        **Abstract**:

        Recently, transformation-based self-supervised learning has been applied to generative adversarial networks (GANs) to mitigate catastrophic forgetting in the discriminator by introducing a stationary learning environment. However, the separate self-supervised tasks in existing self-supervised GANs cause a goal inconsistent with generative modeling due to the fact that their self-supervised classifiers are agnostic to the generator distribution. To address this problem, we propose a novel self-supervised GAN that unifies the GAN task with the self-supervised task by augmenting the GAN labels (real or fake) via self-supervision of data transformation. Specifically, the original discriminator and self-supervised classifier are unified into a label-augmented discriminator that predicts the augmented labels to be aware of both the generator distribution and the data distribution under every transformation, and then provide the discrepancy between them to optimize the generator. Theoretically, we prove that the optimal generator could converge to replicate the real data distribution. Empirically, we show that the proposed method significantly outperforms previous self-supervised and data augmentation GANs on both generative modeling and representation learning across benchmark datasets.

        ----

        ## [997] Shape As Points: A Differentiable Poisson Solver

        **Authors**: *Songyou Peng, Chiyu Jiang, Yiyi Liao, Michael Niemeyer, Marc Pollefeys, Andreas Geiger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6cd9313ed34ef58bad3fdd504355e72c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6cd9313ed34ef58bad3fdd504355e72c-Abstract.html)

        **Abstract**:

        In recent years, neural implicit representations gained popularity in 3D reconstruction due to their expressiveness and flexibility. However, the implicit nature of neural implicit representations results in slow inference times and requires careful initialization. In this paper, we revisit the classic yet ubiquitous point cloud representation and introduce a differentiable point-to-mesh layer using a differentiable formulation of Poisson Surface Reconstruction (PSR) which allows for a GPU-accelerated fast solution of the indicator function given an oriented point cloud. The differentiable PSR layer allows us to efficiently and differentiably bridge the explicit 3D point representation with the 3D mesh via the implicit indicator field, enabling end-to-end optimization of surface reconstruction metrics such as Chamfer distance. This duality between points and meshes hence allows us to represent shapes as oriented point clouds, which are explicit, lightweight and expressive. Compared to neural implicit representations, our Shape-As-Points (SAP) model is more interpretable, lightweight, and accelerates inference time by one order of magnitude. Compared to other explicit representations such as points, patches, and meshes, SAP produces topology-agnostic, watertight manifold surfaces. We demonstrate the effectiveness of SAP on the task of surface reconstruction from unoriented point clouds and learning-based reconstruction.

        ----

        ## [998] Outcome-Driven Reinforcement Learning via Variational Inference

        **Authors**: *Tim G. J. Rudner, Vitchyr Pong, Rowan McAllister, Yarin Gal, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6cdd60ea0045eb7a6ec44c54d29ed402-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6cdd60ea0045eb7a6ec44c54d29ed402-Abstract.html)

        **Abstract**:

        While reinforcement learning algorithms provide automated acquisition of optimal policies, practical application of such methods requires a number of design decisions, such as manually designing reward functions that not only define the task, but also provide sufficient shaping to accomplish it. In this paper, we view reinforcement learning as inferring policies that achieve desired outcomes, rather than as a problem of maximizing rewards. To solve this inference problem, we establish a novel variational inference formulation that allows us to derive a well-shaped reward function which can be learned directly from environment interactions. From the corresponding variational objective, we also derive a new probabilistic Bellman backup operator and use it to develop an off-policy algorithm to solve goal-directed tasks. We empirically demonstrate that this method eliminates the need to hand-craft reward functions for a suite of diverse manipulation and locomotion tasks and leads to effective goal-directed behaviors.

        ----

        ## [999] Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks

        **Authors**: *Yonggan Fu, Qixuan Yu, Yang Zhang, Shang Wu, Xu Ouyang, David D. Cox, Yingyan Lin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/6ce8d8f3b038f737cefcdafcf3752452-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/6ce8d8f3b038f737cefcdafcf3752452-Abstract.html)

        **Abstract**:

        Deep Neural Networks (DNNs) are known to be vulnerable to adversarial attacks, i.e., an imperceptible perturbation to the input can mislead DNNs trained on clean images into making erroneous predictions. To tackle this, adversarial training is currently the most effective defense method, by augmenting the training set with adversarial samples generated on the fly. \textbf{Interestingly, we discover for the first time that there exist subnetworks with inborn robustness, matching or surpassing the robust accuracy of the adversarially trained networks with comparable model sizes, within randomly initialized networks without any model training}, indicating that adversarial training on model weights is not indispensable towards adversarial robustness. We name such subnetworks Robust Scratch Tickets (RSTs), which are also by nature efficient. Distinct from the popular lottery ticket hypothesis, neither the original dense networks nor the identified RSTs need to be trained. To validate and understand this fascinating finding, we further conduct extensive experiments to study the existence and properties of RSTs under different models, datasets, sparsity patterns, and attacks, drawing insights regarding the relationship between DNNsâ€™ robustness and their initialization/overparameterization. Furthermore, we identify the poor adversarial transferability between RSTs of different sparsity ratios drawn from the same randomly initialized dense network, and propose a Random RST Switch (R2S) technique, which randomly switches between different RSTs, as a novel defense method built on top of RSTs. We believe our findings about RSTs have opened up a new perspective to study model robustness and extend the lottery ticket hypothesis.

        ----

        

[Go to the previous page](NIPS-2021-list04.md)

[Go to the next page](NIPS-2021-list06.md)

[Go to the catalog section](README.md)