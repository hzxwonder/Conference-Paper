## [2400] Exponential Separations in Symmetric Neural Networks

        **Authors**: *Aaron Zweig, Joan Bruna*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html)

        **Abstract**:

        In this work we demonstrate a novel separation between symmetric neural network architectures.  Specifically, we consider the Relational Network~\parencite{santoro2017simple} architecture as a natural generalization of the DeepSets~\parencite{zaheer2017deep} architecture, and study their representational gap. Under the restriction to analytic activation functions, we construct a symmetric function acting on sets of size $N$ with elements in dimension $D$, which can be efficiently approximated by the former architecture, but provably requires width exponential in $N$ and $D$ for the latter.

        ----

        ## [2401] Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks

        **Authors**: *Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d66d8164cfbf012cac2866edbb375035-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d66d8164cfbf012cac2866edbb375035-Abstract-Conference.html)

        **Abstract**:

        Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.

        ----

        ## [2402] Oscillatory Tracking of Continuous Attractor Neural Networks Account for Phase Precession and Procession of Hippocampal Place Cells

        **Authors**: *Tianhao Chu, Zilong Ji, Junfeng Zuo, Wenhao Zhang, Tiejun Huang, Yuanyuan Mi, Si Wu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6797a91df19b768409b5178642dcb26-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6797a91df19b768409b5178642dcb26-Abstract-Conference.html)

        **Abstract**:

        Hippocampal place cells of freely moving rodents display an intriguing temporal organization in their responses known as `theta phase precession', in which individual neurons fire at progressively earlier phases in successive theta cycles as the animal traverses the place fields. Recent experimental studies found that in addition to phase precession, many place cells also exhibit accompanied phase procession, but the underlying neural mechanism remains unclear. Here, we propose a neural circuit model to elucidate the generation of both kinds of phase shift in place cells' firing. Specifically, we consider a continuous attractor neural network (CANN) with feedback inhibition, which is inspired by the reciprocal interaction between the hippocampus and the medial septum. The feedback inhibition induces intrinsic mobility of the CANN which competes with the extrinsic mobility arising from the external drive. Their interplay generates an oscillatory tracking state, that is, the network bump state (resembling the decoded virtual position of the animal) sweeps back and forth around the external moving input (resembling the physical position of the animal). We show that this oscillatory tracking naturally explains the forward and backward sweeps of the decoded position during the animal's locomotion.  At the single neuron level, the forward and backward sweeps account for, respectively, theta phase precession and procession. Furthermore, by tuning the feedback inhibition strength, we also explain the emergence of bimodal cells and unimodal cells, with the former having co-existed phase precession and procession, and the latter having only significant phase precession. We hope that this study facilitates our understanding of hippocampal temporal coding and lays foundation for unveiling their computational functions.

        ----

        ## [2403] Finding and Listing Front-door Adjustment Sets

        **Authors**: *Hyunchai Jeong, Jin Tian, Elias Bareinboim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d67ac621411d6fa4dca682ce62f84ea7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d67ac621411d6fa4dca682ce62f84ea7-Abstract-Conference.html)

        **Abstract**:

        Identifying the effects of new interventions from data is a significant challenge found across a wide range of the empirical sciences. A well-known strategy for identifying such effects is Pearl's front-door (FD) criterion. The definition of the FD criterion is declarative, only allowing one to decide whether a specific set satisfies the criterion. In this paper, we present algorithms for finding and enumerating possible sets satisfying the FD criterion in a given causal diagram. These results are useful in facilitating the practical applications of the FD criterion for causal effects estimation and helping scientists to select estimands with desired properties, e.g., based on cost, feasibility of measurement, or statistical power.

        ----

        ## [2404] MorphTE: Injecting Morphology in Tensorized Embeddings

        **Authors**: *Guobing Gan, Peng Zhang, Sunzhu Li, Xiuqing Lu, Benyou Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d68b4e80fd0dd8ac72092b3acd418f75-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d68b4e80fd0dd8ac72092b3acd418f75-Abstract-Conference.html)

        **Abstract**:

        In the era of deep learning, word embeddings are essential when dealing with text tasks. However, storing and accessing these embeddings requires a large amount of space. This is not conducive to the deployment of these models on resource-limited devices. Combining the powerful compression capability of tensor products, we propose a word embedding compression method with morphological augmentation,  Morphologically-enhanced Tensorized Embeddings (MorphTE). A word consists of one or more morphemes, the smallest units that bear meaning or have a grammatical function. MorphTE represents a word embedding as an entangled form of its morpheme vectors via the tensor product, which injects prior semantic and grammatical knowledge into the learning of embeddings. Furthermore, the dimensionality of the morpheme vector and the number of morphemes are much smaller than those of words, which greatly reduces the parameters of the word embeddings. We conduct experiments on tasks such as machine translation and question answering. Experimental results on four translation datasets of different languages show that MorphTE can compress word embedding parameters by about $20$ times without performance loss and significantly outperforms related embedding compression methods.

        ----

        ## [2405] Active Learning with Safety Constraints

        **Authors**: *Romain Camilleri, Andrew Wagenmaker, Jamie H. Morgenstern, Lalit Jain, Kevin G. Jamieson*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6929af3791b2cec21c136b573aa87f2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6929af3791b2cec21c136b573aa87f2-Abstract-Conference.html)

        **Abstract**:

        Active learning methods have shown great promise in reducing the number of samples necessary for learning. As automated learning systems are adopted into real-time, real-world decision-making pipelines, it is increasingly important that such algorithms are designed with safety in mind. In this work we investigate the complexity of learning the best safe decision in interactive environments. We reduce this problem to a safe linear bandits problem, where our goal is to find the best arm satisfying certain (unknown) safety constraints. We propose an adaptive experimental design-based algorithm, which we show efficiently trades off between the difficulty of showing an arm is unsafe vs suboptimal. To our knowledge, our results are the first on best-arm identification in linear bandits with safety constraints. In  practice, we demonstrate that this approach performs well on synthetic and real world datasets.

        ----

        ## [2406] Is one annotation enough? - A data-centric image classification benchmark for noisy and ambiguous label estimation

        **Authors**: *Lars Schmarje, Vasco Grossmann, Claudius Zelenka, Sabine Dippel, Rainer Kiko, Mariusz Oszust, Matti Pastell, Jenny Stracke, Anna Valros, Nina Volkmann, Reinhard Koch*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6c03035b8bc551f474f040fe8607cab-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6c03035b8bc551f474f040fe8607cab-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        High-quality data is necessary for modern machine learning. However, the acquisition of such data is difficult due to noisy and ambiguous annotations of humans. The aggregation of such annotations to determine the label of an image leads to a lower data quality. We propose a data-centric image classification benchmark with nine real-world datasets and multiple annotations per image to allow researchers to investigate and quantify the impact of such data quality issues. With the benchmark we can study the impact of annotation costs and (semi-)supervised methods on the data quality for image classification by applying a novel methodology to a range of different algorithms and diverse datasets. Our benchmark uses a two-phase approach via a data label improvement method in the first phase and a fixed evaluation model in the second phase. Thereby, we give a measure for the relation between the input labeling effort and the performance of (semi-)supervised algorithms to enable a deeper insight into how labels should be created for effective model training. Across thousands of experiments, we show that one annotation is not enough and that the inclusion of multiple annotations allows for a better approximation of the real underlying class distribution. We identify that hard labels can not capture the ambiguity of the data and this might lead to the common issue of overconfident models. Based on the presented datasets, benchmarked methods, and analysis, we create multiple research opportunities for the future directed at the improvement of label noise estimation approaches, data annotation schemes, realistic (semi-)supervised learning, or more reliable image collection.

        ----

        ## [2407] Sobolev Acceleration and Statistical Optimality for Learning Elliptic Equations via Gradient Descent

        **Authors**: *Yiping Lu, José H. Blanchet, Lexing Ying*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6c53fe062716387ff0df73cc53de60c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6c53fe062716387ff0df73cc53de60c-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the statistical limits in terms of Sobolev norms of gradient descent for solving inverse problem from randomly sampled noisy observations using a general class of objective functions. Our class of objective functions includes Sobolev training for kernel regression, Deep Ritz Methods (DRM), and Physics Informed Neural Networks (PINN) for solving elliptic partial differential equations (PDEs) as special cases. We consider a potentially infinite-dimensional parameterization of our model using a suitable Reproducing Kernel Hilbert Space and a continuous parameterization of problem hardness through the definition of kernel integral operators. We prove that gradient descent over this objective function can also achieve statistical optimality and the optimal number of passes over the data increases with sample size. Based on our theory, we explain an implicit acceleration of using a Sobolev norm as the objective function for training, inferring that the optimal number of epochs of DRM becomes larger than the number of PINN when both the data size and the hardness of tasks increase, although both DRM and PINN can achieve statistical optimality.

        ----

        ## [2408] Block-Recurrent Transformers

        **Authors**: *DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, Behnam Neyshabur*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6e0bbb9fc3f4c10950052ec2359355c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6e0bbb9fc3f4c10950052ec2359355c-Abstract-Conference.html)

        **Abstract**:

        We introduce the Block-Recurrent Transformer, which applies a transformer layer in a recurrent fashion along a sequence, and has linear complexity with respect to sequence length. Our recurrent cell operates on blocks of tokens rather than single tokens during training, and leverages parallel computation within a block in order to make efficient use of accelerator hardware.  The cell itself is strikingly simple. It is merely a transformer layer: it uses self-attention and cross-attention to efficiently compute a recurrent function over a large set of state vectors and tokens.  Our design was inspired in part by LSTM cells, and it uses LSTM-style gates, but it scales the typical LSTM cell up by several orders of magnitude.  Our implementation of recurrence has the same cost in both computation time and parameter count as a conventional transformer layer, but offers dramatically improved perplexity in language modeling tasks over very long sequences. Our model out-performs a long-range Transformer XL baseline by a wide margin, while running twice as fast.  We demonstrate its effectiveness on PG19 (books), arXiv papers, and GitHub source code.  Our code has been released as open source.

        ----

        ## [2409] Learning Two-Player Markov Games: Neural Function Approximation and Correlated Equilibrium

        **Authors**: *Chris Junchi Li, Dongruo Zhou, Quanquan Gu, Michael I. Jordan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d6f681da2151687df12cc21a1c1e3527-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d6f681da2151687df12cc21a1c1e3527-Abstract-Conference.html)

        **Abstract**:

        We consider learning Nash equilibria in two-player zero-sum Markov Games with nonlinear function approximation, where the action-value function is approximated by a function in a Reproducing Kernel Hilbert Space (RKHS). The key challenge is how to do exploration in the high-dimensional function space. We propose a novel online learning algorithm to find a Nash equilibrium by minimizing the duality gap. At the core of our algorithms are upper and lower confidence bounds that are derived based on the principle of optimism in the face of uncertainty. We prove that our algorithm is able to attain an $O(\sqrt{T})$ regret with polynomial computational complexity, under very mild assumptions on the reward function and the underlying dynamic of the Markov Games. We also propose several extensions of our algorithm, including an algorithm with Bernstein-type bonus that can achieve a tighter regret bound, and another algorithm for model misspecification that can be applied to neural network function approximation.

        ----

        ## [2410] Dense Interspecies Face Embedding

        **Authors**: *Sejong Yang, Subin Jeon, Seonghyeon Nam, Seon Joo Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d71a4a6c796cacd9b8a298589943cdf3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d71a4a6c796cacd9b8a298589943cdf3-Abstract-Conference.html)

        **Abstract**:

        Dense Interspecies Face Embedding (DIFE) is a new direction for understanding faces of various animals by extracting common features among animal faces including human face. There are three main obstacles for interspecies face understanding: (1) lack of animal data compared to human, (2) ambiguous connection between faces of various animals, and (3) extreme shape and style variance. To cope with the lack of data, we utilize multi-teacher knowledge distillation of CSE and StyleGAN2 requiring no additional data or label. Then we synthesize pseudo pair images through the latent space exploration of StyleGAN2 to find implicit associations between different animal faces. Finally, we introduce the semantic matching loss to overcome the problem of extreme shape differences between species. To quantitatively evaluate our method over possible previous methodologies like unsupervised keypoint detection, we perform interspecies facial keypoint transfer on MAFL and AP-10K. Furthermore, the results of other applications like interspecies face image manipulation and dense keypoint transfer are provided. The code is available at https://github.com/kingsj0405/dife.

        ----

        ## [2411] Efficient Scheduling of Data Augmentation for Deep Reinforcement Learning

        **Authors**: *Byungchan Ko, Jungseul Ok*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d74d002a9154b4cc433a234feb27c5f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d74d002a9154b4cc433a234feb27c5f4-Abstract-Conference.html)

        **Abstract**:

        In deep reinforcement learning (RL), data augmentation is widely considered as a tool to induce a set of useful priors about semantic consistency and improve sample efficiency and generalization performance. However, even when the prior is useful for generalization, distilling it to RL agent often interferes with RL training and degenerates sample efficiency. Meanwhile, the agent is forgetful of the prior due to the non-stationary nature of RL. These observations suggest two extreme schedules of distillation: (i) over the entire training; or (ii) only at the end. Hence, we devise a stand-alone network distillation method to inject the consistency prior at any time (even after RL), and a simple yet efficient framework to automatically schedule the distillation. Specifically, the proposed framework first focuses on mastering train environments regardless of generalization by adaptively deciding which {\it or no} augmentation to be used for the training. After this, we add the distillation to extract the remaining benefits for generalization from all the augmentations, which requires no additional new samples. In our experiments, we demonstrate the utility of the proposed framework, in particular, that considers postponing the augmentation to the end of RL training.

        ----

        ## [2412] Mix and Reason: Reasoning over Semantic Topology with Data Mixing for Domain Generalization

        **Authors**: *Chaoqi Chen, Luyao Tang, Feng Liu, Gangming Zhao, Yue Huang, Yizhou Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d75f561eaaf2cb754bc8d7e36d8af362-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d75f561eaaf2cb754bc8d7e36d8af362-Abstract-Conference.html)

        **Abstract**:

        Domain generalization (DG) enables generalizing a learning machine from multiple seen source domains to an unseen target one. The general objective of DG methods is to learn semantic representations that are independent of domain labels, which is theoretically sound but empirically challenged due to the complex mixture of common and domain-specific factors. Although disentangling the representations into two disjoint parts has been gaining momentum in DG, the strong presumption over the data limits its efficacy in many real-world scenarios. In this paper, we propose Mix and Reason (MiRe), a new DG framework that learns semantic representations via enforcing the structural invariance of semantic topology. MiRe consists of two key components, namely,  Category-aware Data Mixing (CDM) and Adaptive Semantic Topology Refinement (ASTR). CDM mixes two images from different domains in virtue of activation maps generated by two complementary classification losses, making the classifier focus on the representations of semantic objects. ASTR introduces relation graphs to represent semantic topology, which is progressively refined via the interactions between local feature aggregation and global cross-domain relational reasoning. Experiments on multiple DG benchmarks validate the effectiveness and robustness of the proposed MiRe.

        ----

        ## [2413] Personalized Online Federated Learning with Multiple Kernels

        **Authors**: *Pouya M. Ghari, Yanning Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d78cc4e15f8fbdb0dd77e551601f572c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d78cc4e15f8fbdb0dd77e551601f572c-Abstract-Conference.html)

        **Abstract**:

        Multi-kernel learning (MKL) exhibits well-documented performance in online non-linear function approximation. Federated learning enables a group of learners (called clients) to train an MKL model on the data distributed among clients to perform online non-linear function approximation. There are some challenges in online federated MKL that need to be addressed: i) Communication efficiency especially when a large number of kernels are considered ii) Heterogeneous data distribution among clients. The present paper develops an algorithmic framework to enable clients to communicate with the server to send their updates with affordable communication cost while clients employ a large dictionary of kernels. Utilizing random feature (RF) approximation, the present paper proposes scalable online federated MKL algorithm. We prove that using the proposed online federated MKL algorithm, each client enjoys sub-linear regret with respect to the RF approximation of its best kernel in hindsight, which indicates that the proposed algorithm can effectively deal with heterogeneity of the data distributed among clients. Experimental results on real datasets showcase the advantages of the proposed algorithm compared with other online federated kernel learning ones.

        ----

        ## [2414] Point Transformer V2: Grouped Vector Attention and Partition-based Pooling

        **Authors**: *Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d78ece6613953f46501b958b7bb4582f-Abstract-Conference.html)

        **Abstract**:

        As a pioneering work exploring transformer architecture for 3D point cloud understanding, Point Transformer achieves impressive results on multiple highly competitive benchmarks. In this work, we analyze the limitations of the Point Transformer and propose our powerful and efficient Point Transformer V2 model with novel designs that overcome the limitations of previous work. In particular, we first propose group vector attention, which is more effective than the previous version of vector attention. Inheriting the advantages of both learnable weight encoding and multi-head attention, we present a highly effective implementation of grouped vector attention with a novel grouped weight encoding layer. We also strengthen the position information for attention by an additional position encoding multiplier. Furthermore, we design novel and lightweight partition-based pooling methods which enable better spatial alignment and more efficient sampling. Extensive experiments show that our model achieves better performance than its predecessor and achieves state-of-the-art on several challenging 3D point cloud understanding benchmarks, including 3D point cloud segmentation on ScanNet v2 and S3DIS and 3D point cloud classification on ModelNet40. Our code will be available at https://github.com/Gofinge/PointTransformerV2.

        ----

        ## [2415] Learning Concept Credible Models for Mitigating Shortcuts

        **Authors**: *Jiaxuan Wang, Sarah Jabbour, Maggie Makar, Michael W. Sjoding, Jenna Wiens*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d791394d32c428aecc7a5b101fb47799-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d791394d32c428aecc7a5b101fb47799-Abstract-Conference.html)

        **Abstract**:

        During training, models can exploit spurious correlations as shortcuts, resulting in poor generalization performance when shortcuts do not persist. In this work, assuming access to a representation based on domain knowledge (i.e., known concepts) that is invariant to shortcuts, we aim to learn robust and accurate models from biased training data. In contrast to previous work, we do not rely solely on known concepts, but allow the model to also learn unknown concepts. We propose two approaches for mitigating shortcuts that incorporate domain knowledge, while accounting for potentially important yet unknown concepts. The first approach is two-staged. After fitting a model using known concepts, it accounts for the residual using unknown concepts. While flexible, we show that this approach is vulnerable when shortcuts are correlated with the unknown concepts. This limitation is addressed by our second approach that extends a recently proposed regularization penalty. Applied to two real-world datasets, we demonstrate that both approaches can successfully mitigate shortcut learning.

        ----

        ## [2416] Neural Approximation of Graph Topological Features

        **Authors**: *Zuoyu Yan, Tengfei Ma, Liangcai Gao, Zhi Tang, Yusu Wang, Chao Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d7ce06e9293c3d8e6cb3f80b4157f875-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d7ce06e9293c3d8e6cb3f80b4157f875-Abstract-Conference.html)

        **Abstract**:

        Topological features based on persistent homology capture high-order structural information so as to augment graph neural network methods. However, computing extended persistent homology summaries remains slow for large and dense graphs and can be a serious bottleneck for the learning pipeline. Inspired by recent success in neural algorithmic reasoning, we propose a novel graph neural network to estimate extended persistence diagrams (EPDs) on graphs efficiently. Our model is built on algorithmic insights, and benefits from better supervision and closer alignment with the EPD computation algorithm. We validate our method with convincing empirical results on approximating EPDs and downstream graph representation learning tasks. Our method is also efficient; on large and dense graphs, we accelerate the computation by nearly 100 times.

        ----

        ## [2417] MOVE: Unsupervised Movable Object Segmentation and Detection

        **Authors**: *Adam Bielski, Paolo Favaro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d7eb232f196124894f2e65b9010a5c57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d7eb232f196124894f2e65b9010a5c57-Abstract-Conference.html)

        **Abstract**:

        We introduce MOVE, a novel method to segment objects without any form of supervision. MOVE exploits the fact that foreground objects can be shifted locally relative to their initial position and result in realistic (undistorted) new images. This property allows us to train a segmentation model on a dataset of images without annotation and to achieve state of the art (SotA) performance on several evaluation datasets for unsupervised salient object detection and segmentation. In unsupervised single object discovery, MOVE gives an average CorLoc improvement of 7.2% over the SotA, and in unsupervised class-agnostic object detection it gives a relative AP improvement of 53% on average. Our approach is built on top of self-supervised features (e.g. from DINO or MAE), an inpainting network (based on the Masked AutoEncoder) and adversarial training.

        ----

        ## [2418] FreGAN: Exploiting Frequency Components for Training GANs under Limited Data

        **Authors**: *Mengping Yang, Zhe Wang, Ziqiu Chi, Yanbing Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d804cef41362be39d3972c1a71cfc4e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d804cef41362be39d3972c1a71cfc4e9-Abstract-Conference.html)

        **Abstract**:

        Training GANs under limited data often leads to discriminator overfitting and memorization issues, causing divergent training. Existing approaches mitigate the overfitting by employing data augmentations, model regularization, or attention mechanisms. However, they ignore the frequency bias of GANs and take poor consideration towards frequency information, especially high-frequency signals that contain rich details. To fully utilize the frequency information of limited data, this paper proposes FreGAN, which raises the model's frequency awareness and draws more attention to synthesising high-frequency signals, facilitating high-quality generation. In addition to exploiting both real and generated images' frequency information, we also involve the frequency signals of real images as a self-supervised constraint, which alleviates the GAN disequilibrium and encourages the generator to synthesis adequate rather than arbitrary frequency signals. Extensive results demonstrate the superiority and effectiveness of our FreGAN in ameliorating generation quality in the low-data regime (especially when training data is less than 100). Besides, FreGAN can be seamlessly applied to existing regularization and attention mechanism models to further boost the performance.

        ----

        ## [2419] LBD: Decouple Relevance and Observation for Individual-Level Unbiased Learning to Rank

        **Authors**: *Mouxiang Chen, Chenghao Liu, Zemin Liu, Jianling Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d81cb1f4dc6e13aeb45553f80b3d6837-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d81cb1f4dc6e13aeb45553f80b3d6837-Abstract-Conference.html)

        **Abstract**:

        Using Unbiased Learning to Rank (ULTR) to train the ranking model with biased click logs has attracted increased research interest. The key idea is to explicitly model the user's observation behavior when building the ranker with a large number of click logs. Considering the simplicity, recent efforts are mainly based on the position bias hypothesis, in which the observation only depends on the position. However, this hypothesis does not hold in many scenarios due to the neglect of the distinct characteristics of individuals in the same position. On the other hand, directly modeling observation bias for each individual is quite challenging, since the effects of each individual's features on relevance and observation are entangled. It is difficult to ravel out this coupled effect and thus obtain a correct relevance model from click data. To address this issue, we first present the concept of coupling effect for individual-level ULTR. Then, we develop the novel Lipschitz and Bernoulli Decoupling (LBD) model to decouple the effects on relevance and observation at the individual level. We prove theoretically that our proposed method could recover the correct relevance order for the ranking objective. Empirical results on two LTR benchmark datasets show that the proposed model outperforms the state-of-the-art baselines and verify its effectiveness in debiasing data.

        ----

        ## [2420] MetricFormer: A Unified Perspective of Correlation Exploring in Similarity Learning

        **Authors**: *Jiexi Yan, Erkun Yang, Cheng Deng, Heng Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d81cd83e7f6748af351485d73f305483-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d81cd83e7f6748af351485d73f305483-Abstract-Conference.html)

        **Abstract**:

        Similarity learning can be significantly advanced by informative relationships among different samples and features. The current methods try to excavate the multiple correlations in different aspects, but cannot integrate them into a unified framework. In this paper, we provide to consider the multiple correlations from a unified perspective and propose a new method called MetricFormer, which can effectively capture and model the multiple correlations with an elaborate metric transformer. In MetricFormer, the feature decoupling block is adopted to learn an ensemble of distinct and diverse features with different discriminative characteristics. After that, we apply the batch-wise correlation block into the batch dimension of each mini-batch to implicitly explore sample relationships. Finally, the feature-wise correlation block is performed to discover the intrinsic structural pattern of the ensemble of features and obtain the aggregated feature embedding for similarity measuring. With three kinds of transformer blocks, we can learn more representative features through the proposed MetricFormer. Moreover, our proposed method can be flexibly integrated with any metric learning framework.  Extensive experiments on three widely-used datasets demonstrate the superiority of our proposed method over state-of-the-art methods.

        ----

        ## [2421] Toward a realistic model of speech processing in the brain with self-supervised learning

        **Authors**: *Juliette Millet, Charlotte Caucheteux, Pierre Orhan, Yves Boubenec, Alexandre Gramfort, Ewan Dunbar, Christophe Pallier, Jean-Remi King*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d81ecfc8fb18e833a3fa0a35d92532b8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d81ecfc8fb18e833a3fa0a35d92532b8-Abstract-Conference.html)

        **Abstract**:

        Several deep neural networks have recently been shown to generate activations similar to those of the brain in response to the same input. These algorithms, however, remain largely implausible: they require (1) extraordinarily large amounts of data, (2) unobtainable supervised labels, (3) textual rather than raw sensory input, and / or (4) implausibly large memory (e.g. thousands of contextual words). These elements highlight the need to identify algorithms that, under these limitations, would suffice to account for both behavioral and brain responses. Focusing on speech processing, we here hypothesize that self-supervised algorithms trained on the raw waveform constitute a promising candidate. Specifically, we compare a recent self-supervised model, wav2vec 2.0, to the brain activity of 412 English, French, and Mandarin individuals recorded with functional Magnetic Resonance Imaging (fMRI), while they listened to approximately one hour of audio books. First, we show that this algorithm learns brain-like representations with as little as 600 hours of unlabelled speech -- a quantity comparable to what infants can be exposed to during language acquisition. Second, its functional hierarchy aligns with the cortical hierarchy of speech processing. Third, different training regimes reveal a functional specialization akin to the cortex: wav2vec 2.0 learns sound-generic, speech-specific and language-specific representations similar to those of the prefrontal and temporal cortices. Fourth, we confirm the similarity of this specialization with the behavior of 386 additional participants. These elements, resulting from the largest neuroimaging benchmark to date, show how self-supervised learning can account for a rich organization of speech processing in the brain, and thus delineate a path to identify the laws of language acquisition which shape the human brain.

        ----

        ## [2422] Distributed Inverse Constrained Reinforcement Learning for Multi-agent Systems

        **Authors**: *Shicheng Liu, Minghui Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)

        **Abstract**:

        This paper considers the problem of recovering the policies of multiple interacting experts by estimating their reward functions and constraints where the demonstration data of the experts is distributed to a group of learners. We formulate this problem as a distributed bi-level optimization problem and propose a novel bi-level ``distributed inverse constrained reinforcement learning" (D-ICRL) algorithm that allows the learners to collaboratively estimate the constraints in the outer loop and learn the corresponding policies and reward functions in the inner loop from the distributed demonstrations through intermittent communications. We formally guarantee that the distributed learners asymptotically achieve consensus which belongs to the set of stationary points of the bi-level optimization problem.

        ----

        ## [2423] Collaborative Decision Making Using Action Suggestions

        **Authors**: *Dylan M. Asmar, Mykel J. Kochenderfer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d85030334fadbd55043c911076caf0ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d85030334fadbd55043c911076caf0ae-Abstract-Conference.html)

        **Abstract**:

        The level of autonomy is increasing in systems spanning multiple domains, but these systems still experience failures. One way to mitigate the risk of failures is to integrate human oversight of the autonomous systems and rely on the human to take control when the autonomy fails. In this work, we formulate a method of collaborative decision making through action suggestions that improves action selection without taking control of the system. Our approach uses each suggestion efficiently by incorporating the implicit information shared through suggestions to modify the agent's belief and achieves better performance with fewer suggestions than naively following the suggested actions. We assume collaborative agents share the same objective and communicate through valid actions. By assuming the suggested action is dependent only on the state, we can incorporate the suggested action as an independent observation of the environment. The assumption of a collaborative environment enables us to use the agent's policy to estimate the distribution over action suggestions. We propose two methods that use suggested actions and demonstrate the approach through simulated experiments. The proposed methodology results in increased performance while also being robust to suboptimal suggestions.

        ----

        ## [2424] Near-Optimal Regret for Adversarial MDP with Delayed Bandit Feedback

        **Authors**: *Tiancheng Jin, Tal Lancewicki, Haipeng Luo, Yishay Mansour, Aviv Rosenberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d850b7e0cdc7f1c0820c6ad85405ae94-Abstract-Conference.html)

        **Abstract**:

        The standard assumption in reinforcement learning (RL) is that agents observe feedback for their actions immediately. However, in practice feedback is often observed in delay. This paper studies online learning in episodic Markov decision process (MDP) with unknown transitions, adversarially changing costs, and unrestricted delayed bandit feedback. More precisely, the feedback for the agent in episode $k$ is revealed only in the end of episode $k + d^k$, where the delay $d^k$ can be changing over episodes and chosen by an oblivious adversary. We present the first algorithms that achieve near-optimal $\sqrt{K + D}$ regret, where $K$ is the number of episodes and $D = \sum_{k=1}^K d^k$ is the total delay, significantly improving upon the best known regret bound of $(K + D)^{2/3}$.

        ----

        ## [2425] Provably sample-efficient RL with side information about latent dynamics

        **Authors**: *Yao Liu, Dipendra Misra, Miro Dudík, Robert E. Schapire*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d8684e49752e06ac5e4b554b60ad212a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d8684e49752e06ac5e4b554b60ad212a-Abstract-Conference.html)

        **Abstract**:

        We study reinforcement learning (RL) in settings where observations are high-dimensional, but where an RL agent has access to abstract knowledge about the structure of the state space, as is the case, for example, when a robot is tasked to go to a specific room in a building using observations from its own camera, while having access to the floor plan. We formalize this setting as transfer reinforcement learning from an "abstract simulator," which we assume is deterministic (such as a simple model of moving around the floor plan), but which is only required to capture the target domain's latent-state dynamics approximately up to unknown (bounded) perturbations (to account for environment stochasticity). Crucially, we assume no prior knowledge about the structure of observations in the target domain except that they can be used to identify the latent states (but the decoding map is unknown). Under these assumptions, we present an algorithm, called TASID, that learns a robust policy in the target domain, with sample complexity that is polynomial in the horizon, and independent of the number of states, which is not possible without access to some prior knowledge. In synthetic experiments, we verify various properties of our algorithm and show that it empirically outperforms transfer RL algorithms that require access to "full simulators" (i.e., those that also simulate observations).

        ----

        ## [2426] Optimal Gradient Sliding and its Application to Optimal Distributed Optimization Under Similarity

        **Authors**: *Dmitry Kovalev, Aleksandr Beznosikov, Ekaterina Borodich, Alexander V. Gasnikov, Gesualdo Scutari*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d88f6f81e1aaf606776ffdd06fdf24ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d88f6f81e1aaf606776ffdd06fdf24ef-Abstract-Conference.html)

        **Abstract**:

        We study structured convex  optimization problems, with additive objective   $r:=p + q$, where $r$ is ($\mu$-strongly) convex, $q$ is $L_q$-smooth and convex, and $p$ is $L_p$-smooth, possibly nonconvex. For such a class of problems, we proposed an inexact accelerated gradient sliding method that can skip the gradient computation for one of these   components while still achieving optimal   complexity of gradient calls of $p$ and $q$, that is, $\mathcal{O}(\sqrt{L_p/\mu})$ and $\mathcal{O}(\sqrt{L_q/\mu})$, respectively. This result is much sharper than the classic black-box  complexity $\mathcal{O}(\sqrt{(L_p+L_q)/\mu})$,   especially when  the difference between $L_p$ and $L_q$ is large. We then apply the proposed method to solve distributed optimization problems over master-worker architectures, under agents' function similarity, due to statistical data similarity or otherwise. The distributed algorithm achieves for the first time lower complexity bounds on both communication and local  gradient calls, with the former having being a long-standing open problem. Finally the method is extended to distributed saddle-problems (under function similarity) by means of solving a class of variational inequalities, achieving lower communication and computation complexity bounds.

        ----

        ## [2427] Universally Expressive Communication in Multi-Agent Reinforcement Learning

        **Authors**: *Matthew Morris, Thomas D. Barrett, Arnu Pretorius*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d8a19c815a8bef25e6094e87f963d28e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d8a19c815a8bef25e6094e87f963d28e-Abstract-Conference.html)

        **Abstract**:

        Allowing agents to share information through communication is crucial for solving complex tasks in multi-agent reinforcement learning. In this work, we consider the question of whether a given communication protocol can express an arbitrary policy. By observing that many existing protocols can be viewed as instances of graph neural networks (GNNs), we demonstrate the equivalence of joint action selection to node labelling. With standard GNN approaches provably limited in their expressive capacity, we draw from existing GNN literature and consider augmenting agent observations with: (1) unique agent IDs and (2) random noise. We provide a theoretical analysis as to how these approaches yield universally expressive communication, and also prove them capable of targeting arbitrary sets of actions for identical agents. Empirically, these augmentations are found to improve performance on tasks where expressive communication is required, whilst, in general, the optimal communication protocol is found to be task-dependent.

        ----

        ## [2428] Fast Stochastic Composite Minimization and an Accelerated Frank-Wolfe Algorithm under Parallelization

        **Authors**: *Benjamin Dubois-Taine, Francis R. Bach, Quentin Berthet, Adrien B. Taylor*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d903c4bb4e77f5de1ec92da2cf9dc8db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d903c4bb4e77f5de1ec92da2cf9dc8db-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of minimizing the sum of two convex functions. One of those functions has Lipschitz-continuous gradients, and can be accessed via stochastic oracles, whereas the other is ``simple''. We provide a Bregman-type algorithm with accelerated convergence in function values to a ball containing the minimum. The radius of this ball depends on problem-dependent constants, including the variance of the stochastic oracle. We further show that this algorithmic setup naturally leads to a variant of Frank-Wolfe achieving acceleration under parallelization. More precisely, when minimizing a smooth convex function on a bounded domain, we show that one can achieve an $\epsilon$ primal-dual gap (in expectation) in $\tilde{O}(1 /\sqrt{\epsilon})$ iterations, by only accessing gradients of the original function and a linear maximization oracle with $O(1 / \sqrt{\epsilon})$ computing units in parallel. We illustrate this fast convergence on synthetic numerical experiments.

        ----

        ## [2429] Multi-Granularity Cross-modal Alignment for Generalized Medical Visual Representation Learning

        **Authors**: *Fuying Wang, Yuyin Zhou, Shujun Wang, Varut Vardhanabhuti, Lequan Yu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d925bda407ada0df3190df323a212661-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d925bda407ada0df3190df323a212661-Abstract-Conference.html)

        **Abstract**:

        Learning medical visual representations directly from paired radiology reports has become an emerging topic in representation learning. However, existing medical image-text joint learning methods are limited by instance or local supervision analysis, ignoring disease-level semantic correspondences. In this paper, we present a novel Multi-Granularity Cross-modal Alignment (MGCA) framework for generalized medical visual representation learning by harnessing the naturally exhibited semantic correspondences between medical image and radiology reports at three different levels, i.e., pathological region-level, instance-level, and disease-level. Specifically, we first incorporate the instance-wise alignment module by maximizing the agreement between image-report pairs. Further, for token-wise alignment, we introduce a bidirectional cross-attention strategy to explicitly learn the matching between fine-grained visual tokens and text tokens, followed by contrastive learning to align them. More important, to leverage the high-level inter-subject relationship semantic (e.g., disease) correspondences, we design a novel cross-modal disease-level alignment paradigm to enforce the cross-modal cluster assignment consistency. Extensive experimental results on seven downstream medical image datasets covering image classification, object detection, and semantic segmentation tasks demonstrate the stable and superior performance of our framework.

        ----

        ## [2430] Learning Robust Rule Representations for Abstract Reasoning via Internal Inferences

        **Authors**: *Wenbo Zhang, Likai Tang, Site Mo, Xianggen Liu, Sen Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d951f73c521d069fefbb73396df01424-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d951f73c521d069fefbb73396df01424-Abstract-Conference.html)

        **Abstract**:

        Abstract reasoning, as one of the hallmarks of human intelligence, involves collecting information, identifying abstract rules, and applying the rules to solve new problems. Although neural networks have achieved human-level performances in several tasks, the abstract reasoning techniques still far lag behind due to the complexity of learning and applying the logic rules, especially in an unsupervised manner. In this work, we propose a novel framework, ARII, that learns rule representations for Abstract Reasoning via Internal Inferences. The key idea is to repeatedly apply a rule to different instances in hope of having a comprehensive understanding (i.e., representations) of the rule. Specifically, ARII consists of a rule encoder, a reasoner, and an internal referrer. Based on the representations produced by the rule encoder, the reasoner draws the conclusion while the referrer performs internal inferences to regularize rule representations to be robust and generalizable. We evaluate ARII on two benchmark datasets, including PGM and I-RAVEN. We observe that ARII achieves new state-of-the-art records on the majority of the reasoning tasks, including most of the generalization tests in PGM. Our codes are available at https://github.com/Zhangwenbo0324/ARII.

        ----

        ## [2431] Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation

        **Authors**: *Sérgio M. Jesus, José Pombal, Duarte Alves, André Ferreira Cruz, Pedro Saleiro, Rita P. Ribeiro, João Gama, Pedro Bizarro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/d9696563856bd350e4e7ac5e5812f23c-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Evaluating new techniques on realistic datasets plays a crucial role in the development of ML research and its broader adoption by practitioners. In recent years, there has been a significant increase of publicly available unstructured data resources for computer vision and NLP tasks. However, tabular data — which is prevalent in many high-stakes domains — has been lagging behind. To bridge this gap, we present Bank Account Fraud (BAF), the first publicly available 1 privacy-preserving, large-scale, realistic suite of tabular datasets. The suite was generated by applying state-of-the-art tabular data generation techniques on an anonymized,real-world bank account opening fraud detection dataset. This setting carries a set of challenges that are commonplace in real-world applications, including temporal dynamics and significant class imbalance. Additionally, to allow practitioners to stress test both performance and fairness of ML methods, each dataset variant of BAF contains specific types of data bias. With this resource, we aim to provide the research community with a more realistic, complete, and robust test bed to evaluate novel and existing methods.

        ----

        ## [2432] DropCov: A Simple yet Effective Method for Improving Deep Architectures

        **Authors**: *Qilong Wang, Mingze Gao, Zhaolin Zhang, Jiangtao Xie, Peihua Li, Qinghua Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d9888cc7baa04c2e44e8115588133515-Abstract-Conference.html)

        **Abstract**:

        Previous works show global covariance pooling (GCP) has great potential to improve deep architectures especially on visual recognition tasks, where post-normalization of GCP plays a very important role in final performance. Although several post-normalization strategies have been studied, these methods pay more close attention to effect of normalization on covariance representations rather than the whole GCP networks, and their effectiveness requires further understanding. Meanwhile, existing effective post-normalization strategies (e.g., matrix power normalization) usually suffer from high computational complexity (e.g., $O(d^{3})$ for $d$-dimensional inputs). To handle above issues, this work first analyzes the effect of post-normalization from the perspective of training GCP networks. Particularly, we for the first time show that \textit{effective post-normalization can make a good trade-off between representation decorrelation and information preservation for GCP, which are crucial to alleviate over-fitting and increase representation ability of deep GCP networks, respectively}. Based on this finding, we can improve existing post-normalization methods with some small modifications, providing further support to our observation. Furthermore, this finding encourages us to propose a novel pre-normalization method for GCP (namely DropCov), which develops an adaptive channel dropout on features right before GCP, aiming to reach trade-off between representation decorrelation and information preservation in a more efficient way. Our DropCov only has a linear complexity of $O(d)$, while being free for inference. Extensive experiments on various benchmarks (i.e., ImageNet-1K, ImageNet-C, ImageNet-A, Stylized-ImageNet, and iNat2017) show our DropCov is superior to the counterparts in terms of efficiency and effectiveness, and provides a simple yet effective method to improve performance of deep architectures involving both deep convolutional neural networks (CNNs) and vision transformers (ViTs).

        ----

        ## [2433] A Unifying Framework for Online Optimization with Long-Term Constraints

        **Authors**: *Matteo Castiglioni, Andrea Celli, Alberto Marchesi, Giulia Romano, Nicola Gatti*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d9b564716709357b4bccec9fc9ad04d2-Abstract-Conference.html)

        **Abstract**:

        We study online learning problems in which a decision maker has to take a sequence of decisions subject to $m$ long-term constraints. The goal of the decision maker is to maximize their total reward, while at the same time achieving small cumulative constraints violations across the $T$ rounds. We present the first best-of-both-world type algorithm for this general class of problems, with no-regret guarantees both in the case in which rewards and constraints are selected according to an unknown stochastic model, and in the case in which they are selected at each round by an adversary. Our algorithm is the first to provide guarantees in the adversarial setting with respect to the optimal fixed strategy that satisfies the long-term constraints. In particular, it guarantees a $\rho/(1+\rho)$ fraction of the optimal utility and sublinear regret, where $\rho$ is a feasibility parameter related to the existence of strictly feasible solutions. Our framework employs traditional regret minimizers as black-box components. Therefore, by instantiating it with an appropriate choice of regret minimizers it can handle both the full-feedback as well as the bandit-feedback setting. Moreover, it allows the decision maker to seamlessly handle scenarios with non-convex reward and constraints. We show how our framework may be applied in the context of budget-management mechanisms for repeated auctions in order to guarantee long-term constraints which are not packing (e.g., ROI constraints).

        ----

        ## [2434] The least-control principle for local learning at equilibrium

        **Authors**: *Alexander Meulemans, Nicolas Zucchet, Seijin Kobayashi, Johannes von Oswald, João Sacramento*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d9c7c8bd6ad4cebb7d006e5109e0b682-Abstract-Conference.html)

        **Abstract**:

        Equilibrium systems are a powerful way to express neural computations. As special cases, they include models of great current interest in both neuroscience and machine learning, such as deep neural networks, equilibrium recurrent neural networks, deep equilibrium models, or meta-learning. Here, we present a new principle for learning such systems with a temporally- and spatially-local rule. Our principle casts learning as a \emph{least-control} problem, where we first introduce an optimal controller to lead the system towards a solution state, and then define learning as reducing the amount of control needed to reach such a state. We show that incorporating learning signals within a dynamics as an optimal control enables transmitting activity-dependent credit assignment information, avoids storing intermediate states in memory, and does not rely on infinitesimal learning signals. In practice, our principle leads to strong performance matching that of leading gradient-based learning methods when applied to an array of problems involving recurrent neural networks and meta-learning. Our results shed light on how the brain might learn and offer new ways of approaching a broad class of machine learning problems.

        ----

        ## [2435] Optimizing Relevance Maps of Vision Transformers Improves Robustness

        **Authors**: *Hila Chefer, Idan Schwartz, Lior Wolf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/d9fa720cf96f7c18ac4d9e04270f0bbf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/d9fa720cf96f7c18ac4d9e04270f0bbf-Abstract-Conference.html)

        **Abstract**:

        It has been observed that visual classification models often rely mostly on spurious cues such as the image background, which hurts their robustness to distribution changes.  To alleviate this shortcoming, we propose to monitor the model's relevancy signal and direct the model to base its prediction on the foreground object.This is done as a finetuning step, involving relatively few samples consisting of pairs of images and their associated foreground masks. Specifically, we encourage the model's relevancy map (i) to assign lower relevance to background regions, (ii) to consider as much information as possible from the foreground, and (iii) we encourage the decisions to have high confidence. When applied to Vision Transformer (ViT) models, a marked improvement in robustness to domain-shifts is observed. Moreover, the foreground masks can be obtained automatically, from a self-supervised variant of the ViT model itself; therefore no additional supervision is required. Our code is available at: https://github.com/hila-chefer/RobustViT.

        ----

        ## [2436] Stability and Generalization of Kernel Clustering: from Single Kernel to Multiple Kernel

        **Authors**: *Weixuan Liang, Xinwang Liu, Yong Liu, Sihang Zhou, Jun-Jie Huang, Siwei Wang, Jiyuan Liu, Yi Zhang, En Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da0945c639b376b84941256c7371dd45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da0945c639b376b84941256c7371dd45-Abstract-Conference.html)

        **Abstract**:

        Multiple kernel clustering (MKC) is an important research topic that has been widely studied for decades. However, current methods still face two problems: inefficient when handling out-of-sample data points and lack of theoretical study of the stability and generalization of clustering. In this paper, we propose a novel method that can efficiently compute the embedding of out-of-sample data with a solid generalization guarantee. Specifically, we approximate the eigen functions of the integral operator associated with the linear combination of base kernel functions to construct low-dimensional embeddings of out-of-sample points for efficient multiple kernel clustering. In addition, we, for the first time, theoretically study the stability of clustering algorithms and prove that the single-view version of the proposed method has uniform stability as $\mathcal{O}\left(Kn^{-3/2}\right)$ and establish an upper bound of excess risk as $\widetilde{\mathcal{O}}\left(Kn^{-3/2}+n^{-1/2}\right)$, where $K$ is the cluster number and $n$ is the number of samples. We then extend the theoretical results to multiple kernel scenarios and find that the stability of MKC depends on kernel weights. As an example, we apply our method to a novel MKC algorithm termed SimpleMKKM and derive the upper bound of its excess clustering risk, which is tighter than the current results. Extensive experimental results validate the effectiveness and efficiency of the proposed method.

        ----

        ## [2437] Deep Ensembles Work, But Are They Necessary?

        **Authors**: *Taiga Abe, Estefany Kelly Buchanan, Geoff Pleiss, Richard S. Zemel, John P. Cunningham*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da18c47118a2d09926346f33bebde9f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da18c47118a2d09926346f33bebde9f4-Abstract-Conference.html)

        **Abstract**:

        Ensembling neural networks is an effective way to increase accuracy, and can often match the performance of individual larger models. This observation poses a natural question: given the choice between a deep ensemble and a single neural network with similar accuracy, is one preferable over the other? Recent work suggests that deep ensembles may offer distinct benefits beyond predictive power: namely, uncertainty quantification and robustness to dataset shift. In this work, we demonstrate limitations to these purported benefits, and show that a single (but larger) neural network can replicate these qualities. First, we show that ensemble diversity, by any metric, does not meaningfully contribute to an ensemble's ability to detect out-of-distribution (OOD) data, but is instead highly correlated with the relative improvement of a single larger model. Second, we show that the OOD performance afforded by ensembles is strongly determined by their in-distribution (InD) performance, and - in this sense - is not indicative of any "effective robustness." While deep ensembles are a practical way to achieve improvements to predictive power, uncertainty quantification, and robustness, our results show that these improvements can be replicated by a (larger) single model.

        ----

        ## [2438] VICE: Variational Interpretable Concept Embeddings

        **Authors**: *Lukas Muttenthaler, Charles Y. Zheng, Patrick McClure, Robert A. Vandermeulen, Martin N. Hebart, Francisco Pereira*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da1a97b53eec1c763c6d06835538fe3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da1a97b53eec1c763c6d06835538fe3e-Abstract-Conference.html)

        **Abstract**:

        A central goal in the cognitive sciences is the development of numerical models for mental representations of object concepts. This paper introduces Variational Interpretable Concept Embeddings (VICE), an approximate Bayesian method for embedding object concepts in a vector space using data collected from humans in a triplet odd-one-out task. VICE uses variational inference to obtain sparse, non-negative representations of object concepts with uncertainty estimates for the embedding values. These estimates are used to automatically select the dimensions that best explain the data. We derive a PAC learning bound for VICE that can be used to estimate generalization performance or determine a sufficient sample size for experimental design. VICE rivals or outperforms its predecessor, SPoSE, at predicting human behavior in the triplet odd-one-out task. Furthermore, VICE's object representations are more reproducible and consistent across random initializations, highlighting the unique advantage of using VICE for deriving interpretable embeddings from human behavior.

        ----

        ## [2439] Random Normalization Aggregation for Adversarial Defense

        **Authors**: *Minjing Dong, Xinghao Chen, Yunhe Wang, Chang Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da3d4d2e9b37f78ec3e7d0428c9b819a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da3d4d2e9b37f78ec3e7d0428c9b819a-Abstract-Conference.html)

        **Abstract**:

        The vulnerability of deep neural networks has been widely found in various models as well as tasks where slight perturbations on the inputs could lead to incorrect predictions. These perturbed inputs are known as adversarial examples and one of the intriguing properties of them is Adversarial Transfersability, i.e. the capability of adversarial examples to fool other models. Traditionally, this transferability is always regarded as a critical threat to the defense against adversarial attacks, however, we argue that the network robustness can be significantly boosted by utilizing adversarial transferability from a new perspective. In this work, we first discuss the influence of different popular normalization layers on the adversarial transferability, and then provide both empirical evidence and theoretical analysis to shed light on the relationship between normalization types and transferability. Based on our theoretical analysis, we propose a simple yet effective module named Random Normalization Aggregation (RNA) which replaces the batch normalization layers in the networks and aggregates different selected normalization types to form a huge random space. Specifically, a random path is sampled during each inference procedure so that the network itself can be treated as an ensemble of a wide range of different models. Since the entire random space is designed with low adversarial transferability, it is difficult to perform effective attacks even when the network parameters are accessible. We conduct extensive experiments on various models and datasets, and demonstrate the strong superiority of proposed algorithm. The PyTorch code is available at https://github.com/UniSerj/Random-Norm-Aggregation and the MindSpore code is available at https://gitee.com/mindspore/models/tree/master/research/cv/RNA.

        ----

        ## [2440] Distributionally Robust Optimization with Data Geometry

        **Authors**: *Jiashuo Liu, Jiayun Wu, Bo Li, Peng Cui*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da535999561b932f56efdd559498282e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da535999561b932f56efdd559498282e-Abstract-Conference.html)

        **Abstract**:

        Distributionally Robust Optimization (DRO) serves as a robust alternative to empirical risk minimization (ERM), which optimizes the worst-case distribution in an uncertainty set typically specified by distance metrics including $f$-divergence and the Wasserstein distance. The metrics defined in the ostensible high dimensional space lead to exceedingly large uncertainty sets, resulting in the underperformance of most existing DRO methods. It has been well documented that high dimensional data approximately resides on low dimensional manifolds. In this work, to further constrain the uncertainty set, we incorporate data geometric properties into the design of distance metrics, obtaining our novel Geometric Wasserstein DRO (GDRO). Empowered by Gradient Flow, we derive a generically applicable approximate algorithm for the optimization of GDRO, and provide the bounded error rate of the approximation as well as the convergence rate of our algorithm. We also theoretically characterize the edge cases where certain existing DRO methods are the degeneracy of GDRO. Extensive experiments justify the superiority of our GDRO to existing DRO methods in multiple settings with strong distributional shifts, and confirm that the uncertainty set of GDRO adapts to data geometry.

        ----

        ## [2441] Near-Optimal Correlation Clustering with Privacy

        **Authors**: *Vincent Cohen-Addad, Chenglin Fan, Silvio Lattanzi, Slobodan Mitrovic, Ashkan Norouzi-Fard, Nikos Parotsidis, Jakub Tarnawski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da645920dcd3bd35b0dae329894bad80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da645920dcd3bd35b0dae329894bad80-Abstract-Conference.html)

        **Abstract**:

        Correlation clustering is a central problem in unsupervised learning, with applications spanning community detection, duplicate detection, automated labeling and many more. In the correlation clustering problem one receives as input a set of nodes and for each node a list of co-clustering preferences, and the goal is to output a clustering that minimizes the disagreement with the specified nodes' preferences. In this paper, we introduce a simple and computationally efficient algorithm for the correlation clustering problem with provable privacy guarantees. Our additive error is stronger than those obtained in prior work and is optimal up to polylogarithmic factors for fixed privacy parameters.

        ----

        ## [2442] Knowledge Distillation from A Stronger Teacher

        **Authors**: *Tao Huang, Shan You, Fei Wang, Chen Qian, Chang Xu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html)

        **Abstract**:

        Unlike existing knowledge distillation methods focus on the baseline settings, where the teacher models and training strategies are not that strong and competing as state-of-the-art approaches, this paper presents a method dubbed DIST to distill better from a stronger teacher. We empirically find that the discrepancy of predictions between the student and a stronger teacher may tend to be fairly severer. As a result, the exact match of predictions in KL divergence would disturb the training and make existing methods perform poorly. In this paper, we show that simply preserving the relations between the predictions of teacher and student would suffice, and propose a correlation-based loss to capture the intrinsic inter-class relations from the teacher explicitly. Besides, considering that different instances have different semantic similarities to each class, we also extend this relational match to the intra-class level. Our method is simple yet practical, and extensive experiments demonstrate that it adapts well to various architectures, model sizes and training strategies, and can achieve state-of-the-art performance consistently on image classification, object detection, and semantic segmentation tasks. Code is available at: https://github.com/hunto/DIST_KD.

        ----

        ## [2443] Optimal Transport of Classifiers to Fairness

        **Authors**: *Maarten Buyl, Tijl De Bie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da75d2bbf862b86f10241d0887613b41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da75d2bbf862b86f10241d0887613b41-Abstract-Conference.html)

        **Abstract**:

        In past work on fairness in machine learning, the focus has been on forcing the prediction of classifiers to have similar statistical properties for people of different demographics. To reduce the violation of these properties, fairness methods usually simply rescale the classifier scores, ignoring similarities and dissimilarities between members of different groups. Yet, we hypothesize that such information is relevant in quantifying the unfairness of a given classifier. To validate this hypothesis, we introduce Optimal Transport to Fairness (OTF), a method that quantifies the violation of fairness constraints as the smallest Optimal Transport cost between a probabilistic classifier and any score function that satisfies these constraints. For a flexible class of linear fairness constraints, we construct a practical way to compute OTF as a differentiable fairness regularizer that can be added to any standard classification setting. Experiments show that OTF can be used to achieve an improved trade-off between predictive power and fairness.

        ----

        ## [2444] Rethinking the compositionality of point clouds through regularization in the hyperbolic space

        **Authors**: *Antonio Montanaro, Diego Valsesia, Enrico Magli*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/da8f9fc2b555d122369f36a9684415c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/da8f9fc2b555d122369f36a9684415c1-Abstract-Conference.html)

        **Abstract**:

        Point clouds of 3D objects exhibit an inherent compositional nature where simple parts can be assembled into progressively more complex shapes to form whole objects. Explicitly capturing such part-whole hierarchy is a long-sought objective in order to build effective models, but its tree-like nature has made the task elusive. In this paper, we propose to embed the features of a point cloud classifier into the hyperbolic space and explicitly regularize the space to account for the part-whole hierarchy. The hyperbolic space is the only space that can successfully embed the tree-like nature of the hierarchy. This leads to substantial improvements in the performance of state-of-art supervised models for point cloud classification.

        ----

        ## [2445] ReCo: Retrieve and Co-segment for Zero-shot Transfer

        **Authors**: *Gyungin Shin, Weidi Xie, Samuel Albanie*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/daabe43c3e1d06980aa23880bfbe1f45-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/daabe43c3e1d06980aa23880bfbe1f45-Abstract-Conference.html)

        **Abstract**:

        Semantic segmentation has a broad range of applications, but its real-world impact has been significantly limited by the prohibitive annotation costs necessary to enable deployment. Segmentation methods that forgo supervision can side-step these costs, but exhibit the inconvenient requirement to provide labelled examples from the target distribution to assign concept names to predictions. An alternative line of work in language-image pre-training has recently demonstrated the potential to produce models that can both assign names across large vocabularies of concepts and enable zero-shot transfer for classification, but do not demonstrate commensurate segmentation abilities.We leverage the retrieval abilities of one such language-image pre-trained model, CLIP, to dynamically curate training sets from unlabelled images for arbitrary collections of concept names, and leverage the robust correspondences offered by modern image representations to co-segment entities among the resulting collections. The synthetic segment collections are then employed to construct a segmentation model (without requiring pixel labels) whose knowledge of concepts is inherited from the scalable pre-training process of CLIP. We demonstrate that our approach, termed Retrieve and Co-segment (ReCo) performs favourably to conventional unsupervised segmentation approaches while inheriting the convenience of nameable predictions and zero-shot transfer. We also demonstrate ReCoâ€™s ability to generate specialist segmenters for extremely rare objects.

        ----

        ## [2446] Monocular Dynamic View Synthesis: A Reality Check

        **Authors**: *Hang Gao, Ruilong Li, Shubham Tulsiani, Bryan Russell, Angjoo Kanazawa*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dab5a29f6614ec47ea0ca85c140226fd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dab5a29f6614ec47ea0ca85c140226fd-Abstract-Conference.html)

        **Abstract**:

        We study the recent progress on dynamic view synthesis (DVS) from monocular video. Though existing approaches have demonstrated impressive results, we show a discrepancy between the practical capture process and the existing experimental protocols, which effectively leaks in multi-view signals during training. We define effective multi-view factors (EMFs) to quantify the amount of multi-view signal present in the input capture sequence based on the relative camera-scene motion. We introduce two new metrics: co-visibility masked image metrics and correspondence accuracy, which overcome the issue in existing protocols. We also propose a new iPhone dataset that includes more diverse real-life deformation sequences. Using our proposed experimental protocol, we show that the state-of-the-art approaches observe a 1-2 dB drop in masked PSNR in the absence of multi-view cues and 4-5 dB drop when modeling complex motion. Code and data can be found at http://hangg7.com/dycheck.

        ----

        ## [2447] Bridging the Gap between Object and Image-level Representations for Open-Vocabulary Detection

        **Authors**: *Hanoona Abdul Rasheed, Muhammad Maaz, Muhammad Uzair Khattak, Salman H. Khan, Fahad Shahbaz Khan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dabf612543b97ea9c8f46d058d33cf74-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dabf612543b97ea9c8f46d058d33cf74-Abstract-Conference.html)

        **Abstract**:

        Existing open-vocabulary object detectors typically enlarge their vocabulary sizes by leveraging different forms of weak supervision. This helps generalize to novel objects at inference. Two popular forms of weak-supervision used in open-vocabulary detection (OVD) include pretrained CLIP model and image-level supervision. We note that both these modes of supervision are not optimally aligned for the detection task: CLIP is trained with image-text pairs and lacks precise localization of objects while the image-level supervision has been used with heuristics that do not accurately specify local object regions. In this work, we propose to address this problem by performing object-centric alignment  of the language embeddings from the CLIP model. Furthermore, we visually ground the objects with only image-level supervision using a pseudo-labeling process that provides high-quality object proposals and helps expand the vocabulary during training. We establish a bridge between the above two object-alignment strategies via a novel weight transfer function that aggregates their complimentary strengths. In essence, the proposed model seeks to minimize the gap between object and image-centric representations in the OVD setting. On the COCO benchmark, our proposed approach achieves 36.6 AP50 on novel classes, an absolute 8.2 gain over the previous best performance. For LVIS, we surpass the state-of-the-art ViLD model by 5.0 mask AP for rare categories and 3.4 overall. Code: https://github.com/hanoonaR/object-centric-ovd.

        ----

        ## [2448] A Simple and Optimal Policy Design for Online Learning with Safety against Heavy-tailed Risk

        **Authors**: *David Simchi-Levi, Zeyu Zheng, Feng Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dae8afc6b990aa0b3b5efaa096fbd7fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dae8afc6b990aa0b3b5efaa096fbd7fa-Abstract-Conference.html)

        **Abstract**:

        We consider the classical multi-armed bandit problem and design simple-to-implement new policies that simultaneously enjoy two properties: worst-case optimality for the expected regret, and safety against heavy-tailed risk for the regret distribution. Recently, Fan and Glynn (2021) showed that information-theoretic optimized bandit policies as well as standard UCB policies suffer from some serious heavy-tailed risk; that is, the probability of incurring a linear regret slowly decays at a polynomial rate of $1/T$, as $T$ (the time horizon) increases. Inspired by their result, we further show that any policy that incurs an instance-dependent $O(\ln T)$ regret must incur a linear regret with probability $\Omega(\mathrm{poly}(1/T))$ and that the heavy-tailed risk actually exists for all "instance-dependent consistent" policies. Next, for the two-armed bandit setting, we provide a simple policy design that (i) has the worst-case optimality for the expected regret at order $\tilde O(\sqrt{T})$ and (ii) has the worst-case tail probability of incurring a linear regret decay at an exponential rate $\exp(-\Omega(\sqrt{T}))$. We further prove that this exponential decaying rate of the tail probability is optimal across all policies that have worst-case optimality for the expected regret. Finally, we generalize the policy design and analysis to the general setting with an arbitrary $K$ number of arms. We provide detailed characterization of the tail probability bound for any regret threshold under our policy design. Numerical experiments are conducted to illustrate the theoretical findings. Our results reveal insights on the incompatibility between consistency and light-tailed risk, whereas indicate that worst-case optimality on expected regret and light-tailed risk are compatible.

        ----

        ## [2449] A Boosting Approach to Reinforcement Learning

        **Authors**: *Nataly Brukhim, Elad Hazan, Karan Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/daf8364f0715a41a469c677c0adc4754-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/daf8364f0715a41a469c677c0adc4754-Abstract-Conference.html)

        **Abstract**:

        Reducing reinforcement learning to supervised learning is a well-studied and effective approach that leverages the benefits of compact function approximation to deal with large-scale Markov decision processes. Independently, the boosting methodology (e.g. AdaBoost) has proven to be indispensable in designing efficient and accurate classification algorithms by combining rough and inaccurate rules-of-thumb.In this paper, we take a further step: we reduce reinforcement learning to a sequence of weak learning problems. Since weak learners perform only marginally better than random guesses, such subroutines constitute a weaker assumption than the availability of an accurate supervised learning oracle. We prove that the sample complexity and running time bounds of the proposed method do not explicitly depend on the number of states.While existing results on boosting operate on convex losses, the value function over policies is non-convex. We show how to use a non-convex variant of the Frank-Wolfe method for boosting, that additionally improves upon the known sample complexity and running time bounds even for reductions to supervised learning.

        ----

        ## [2450] Relaxing Equivariance Constraints with Non-stationary Continuous Filters

        **Authors**: *Tycho F. A. van der Ouderaa, David W. Romero, Mark van der Wilk*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dafd116ac8c735f149558b79fd48e090-Abstract-Conference.html)

        **Abstract**:

        Equivariances provide useful inductive biases in neural network modeling, with the translation equivariance of convolutional neural networks being a canonical example. Equivariances can be embedded in architectures through weight-sharing and place symmetry constraints on the functions a neural network can represent. The type of symmetry is typically fixed and has to be chosen in advance. Although some tasks are inherently equivariant, many tasks do not strictly follow such symmetries. In such cases, equivariance constraints can be overly restrictive. In this work, we propose a parameter-efficient relaxation of equivariance that can effectively interpolate between a (i) non-equivariant linear product, (ii) a strict-equivariant convolution, and (iii) a strictly-invariant mapping. The proposed parameterisation can be thought of as a building block to allow adjustable symmetry structure in neural networks. In addition, we demonstrate that the amount of equivariance can be learned from the training data using backpropagation. Gradient-based learning of equivariance achieves similar or improved performance compared to the best value found by cross-validation and outperforms baselines with partial or strict equivariance on CIFAR-10 and CIFAR-100 image classification tasks.

        ----

        ## [2451] The Franz-Parisi Criterion and Computational Trade-offs in High Dimensional Statistics

        **Authors**: *Afonso S. Bandeira, Ahmed El Alaoui, Samuel B. Hopkins, Tselil Schramm, Alexander S. Wein, Ilias Zadik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/daff682411a64632e083b9d6665b1d30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/daff682411a64632e083b9d6665b1d30-Abstract-Conference.html)

        **Abstract**:

        Many high-dimensional statistical inference problems are believed to possess inherent computational hardness. Various frameworks have been proposed to give rigorous evidence for such hardness, including lower bounds against restricted models of computation (such as low-degree functions), as well as methods rooted in statistical physics that are based on free energy landscapes. This paper aims to make a rigorous connection between the seemingly different low-degree and free-energy based approaches. We define a free-energy based criterion for hardness and formally connect it to the well-established notion of low-degree hardness for a broad class of statistical problems, namely all Gaussian additive models and certain models with a sparse planted signal. By leveraging these rigorous connections we are able to: establish that for Gaussian additive models the "algebraic" notion of low-degree hardness implies failure of "geometric" local MCMC algorithms, and provide new low-degree lower bounds for sparse linear regression which seem difficult to prove directly. These results provide both conceptual insights into the connections between different notions of hardness, as well as concrete technical tools such as new methods for proving low-degree lower bounds.

        ----

        ## [2452] Robust Neural Posterior Estimation and Statistical Model Criticism

        **Authors**: *Daniel Ward, Patrick Cannon, Mark Beaumont, Matteo Fasiolo, Sebastian M. Schmon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db0eac6747e3631eb91095cd76065611-Abstract-Conference.html)

        **Abstract**:

        Computer simulations have proven a valuable tool for understanding complex phenomena across the sciences. However, the utility of simulators for modelling and forecasting purposes is often restricted by low data quality, as well as practical limits to model fidelity. In order to circumvent these difficulties, we argue that modellers must treat simulators as idealistic representations of the true data generating process, and consequently should thoughtfully consider the risk of model misspecification. In this work we revisit neural posterior estimation (NPE), a class of algorithms that enable black-box parameter inference in simulation models, and consider the implication of a simulation-to-reality gap. While recent works have demonstrated reliable performance of these methods, the analyses have been performed using synthetic data generated by the simulator model itself, and have therefore only addressed the well-specified case. In this paper, we find that the presence of misspecification, in contrast, leads to unreliable inference when NPE is used naïvely. As a remedy we argue that principled scientific inquiry with simulators should incorporate a model criticism component, to facilitate interpretable identification of misspecification and a robust inference component, to fit ‘wrong but useful’ models. We propose robust neural posterior estimation (RNPE), an extension of NPE to simultaneously achieve both these aims, through explicitly modelling the discrepancies between simulations and the observed data. We assess the approach on a range of artificially misspecified examples, and find RNPE performs well across the tasks, whereas naïvely using NPE leads to misleading and erratic posteriors.

        ----

        ## [2453] Why do We Need Large Batchsizes in Contrastive Learning? A Gradient-Bias Perspective

        **Authors**: *Changyou Chen, Jianyi Zhang, Yi Xu, Liqun Chen, Jiali Duan, Yiran Chen, Son Tran, Belinda Zeng, Trishul Chilimbi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db174d373133dcc6bf83bc98e4b681f8-Abstract-Conference.html)

        **Abstract**:

        Contrastive learning (CL) has been the de facto technique for self-supervised representation learning (SSL), with impressive empirical success such as multi-modal representation learning. However, traditional CL loss only considers negative samples from a minibatch, which could cause biased gradients due to the non-decomposibility of the loss. For the first time, we consider optimizing a more generalized contrastive loss, where each data sample is associated with an infinite number of negative samples. We show that directly using minibatch stochastic optimization could lead to gradient bias. To remedy this, we propose an efficient Bayesian data augmentation technique to augment the contrastive loss into a decomposable one, where standard stochastic optimization can be directly applied without gradient bias. Specifically, our augmented loss defines a joint distribution over the model parameters and the augmented parameters, which can be conveniently optimized by a proposed stochastic expectation-maximization algorithm. Our framework is more general and is related to several popular SSL algorithms. We verify our framework on both small scale models and several large foundation models, including SSL of ImageNet and SSL for vision-language representation learning. Experiment results indicate the existence of gradient bias in all cases, and demonstrate the effectiveness of the proposed method on improving previous state of the arts. Remarkably, our method can outperform the strong MoCo-v3 under the same hyper-parameter setting with only around half of the minibatch size; and also obtains strong results in the recent public benchmark ELEVATER for few-shot image classification.

        ----

        ## [2454] Randomized Channel Shuffling: Minimal-Overhead Backdoor Attack Detection without Clean Datasets

        **Authors**: *Ruisi Cai, Zhenyu Zhang, Tianlong Chen, Xiaohan Chen, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db1d5c63576587fc1d40d33a75190c71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db1d5c63576587fc1d40d33a75190c71-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) typically require massive data to train on, which is a hurdle for numerous practical domains. Facing the data shortfall, one viable option is to acquire domain-specific training data from external uncensored sources, such as open webs or third-party data collectors. However, the quality of such acquired data is often not rigorously scrutinized, and one cannot easily rule out the risk of `"poisoned" examples being included in such unreliable datasets, resulting in unreliable trained models which pose potential risks to many high-stake applications. While existing options usually suffer from high computational costs or assumptions on clean data access, this paper attempts to detect backdoors for potential victim models with minimal prior knowledge. In particular, provided with a trained model, users are assumed to (1) have no prior knowledge of whether it is already poisoned, or what the target class/percentage of samples is poisoned, and (2) have no access to a clean sample set from the same training distribution, nor any trusted model trained on such clean data. To tackle this challenging scenario, we first observe the contrasting channel-level statistics between the backdoor trigger and clean image features, and consequently, how they can be differentiated by progressive channel shuffling. We then propose the randomized channel shuffling method for backdoor-targeted class detection, which requires only a few feed-forward passes. It thus incurs minimal overheads and demands no clean sample nor prior knowledge. We further explore a “full” clean data-free setting, where neither the target class detection nor the trigger recovery can access the clean data. Extensive experiments are conducted with three datasets (CIFAR-10,  GTSRB, Tiny ImageNet), three architectures (AlexNet, ResNet-20, SENet-18), and three attacks (BadNets, clean label attack, and WaNet). Results consistently endorse the effectiveness of our proposed technique in backdoor model detection,  with margins of 0.291 ～ 0.640 AUROC over the current state-of-the-arts. Codes are available at https://github.com/VITA-Group/Random-Shuffling-BackdoorDetect.

        ----

        ## [2455] Structural Analysis of Branch-and-Cut and the Learnability of Gomory Mixed Integer Cuts

        **Authors**: *Maria-Florina Balcan, Siddharth Prasad, Tuomas Sandholm, Ellen Vitercik*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db2cbf43a349bc866111e791b58c7bf4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db2cbf43a349bc866111e791b58c7bf4-Abstract-Conference.html)

        **Abstract**:

        The incorporation of cutting planes within the branch-and-bound algorithm, known as branch-and-cut, forms the backbone of modern integer programming solvers. These solvers are the foremost method for solving discrete optimization problems and thus have a vast array of applications in machine learning, operations research, and many other fields. Choosing cutting planes effectively is a major research topic in the theory and practice of integer programming. We conduct a novel structural analysis of branch-and-cut that pins down how every step of the algorithm is affected by changes in the parameters defining the cutting planes added to the input integer program. Our main application of this analysis is to derive sample complexity guarantees for using machine learning to determine which cutting planes to apply during branch-and-cut. These guarantees apply to infinite families of cutting planes, such as the family of Gomory mixed integer cuts, which are responsible for the main breakthrough speedups of integer programming solvers. We exploit geometric and combinatorial structure of branch-and-cut in our analysis, which provides a key missing piece for the recent generalization theory of branch-and-cut.

        ----

        ## [2456] Finite-Time Last-Iterate Convergence for Learning in Multi-Player Games

        **Authors**: *Yang Cai, Argyris Oikonomou, Weiqiang Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db2d2001f63e83214b08948b459f69f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db2d2001f63e83214b08948b459f69f0-Abstract-Conference.html)

        **Abstract**:

        We study the question of last-iterate convergence rate of the extragradient algorithm by Korpelevich [1976] and the optimistic gradient algorithm by Popov [1980] in multi-player games. We show that both algorithms with constant step-size have last-iterate convergence rate of $O(\frac{1}{\sqrt{T}})$ to a Nash equilibrium in terms of the gap function in smooth monotone games, where each player's action set is an arbitrary convex set. Previous results only study the unconstrained setting, where each player's action set is the entire Euclidean space.  Our results address an open question raised in several recent work by Hsieh et al. [2019], Golowich et al. [2020a,b], who ask for last-iterate convergence rate of either the extragradient or the optimistic gradient algorithm in the constrained setting. Our convergence rates for both algorithms are tight and match the lower bounds by Golowich et al. [2020a,b]. At the core of our results lies a new notion -- the tangent residual, which we use to measure the proximity to equilibrium. We use the tangent residual (or a slight variation of the tangent residual) as the the potential function in our analysis of the extragradient algorithm (or the optimistic gradient algorithm) and prove that it is non-increasing between two consecutive iterates.

        ----

        ## [2457] When are Local Queries Useful for Robust Learning?

        **Authors**: *Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db6461eaf0eaeaad1d9c4a70e4818cbd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db6461eaf0eaeaad1d9c4a70e4818cbd-Abstract-Conference.html)

        **Abstract**:

        Distributional assumptions have been shown to be necessary for the robust learnability of concept classes when considering the exact-in-the-ball robust risk and access to random examples by Gourdeau et al. (2019). In this paper, we study learning models where the learner is given more power through the use of local queries, and give the first distribution-free algorithms that perform robust empirical risk minimization (ERM) for this notion of robustness. The first learning model we consider uses local membership queries (LMQ), where the learner can query the label of points near the training sample. We show that, under the uniform distribution, LMQs do not increase the robustness threshold of conjunctions and any superclass, e.g., decision lists and halfspaces. Faced with this negative result, we introduce the local equivalence query (LEQ) oracle, which returns whether the hypothesis and target concept agree in the perturbation region around a point in the training sample, as well as a counterexample if it exists. We show a separation result: on one hand, if the query radius $\lambda$ is strictly smaller than the adversary's perturbation budget $\rho$, then distribution-free robust learning is impossible for a wide variety of concept classes; on the other hand, the setting $\lambda=\rho$ allows us to develop robust ERM algorithms. We then bound the query complexity of these algorithms based on online learning guarantees and further improve these bounds for the special case of conjunctions. We finish by giving robust learning algorithms for halfspaces with margins on both $\{0,1\}^n$ and $\mathbb{R}^n$.

        ----

        ## [2458] SPoVT: Semantic-Prototype Variational Transformer for Dense Point Cloud Semantic Completion

        **Authors**: *Sheng-Yu Huang, Hao-Yu Hsu, Yu-Chiang Frank Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db6caae0f83e45e454e2215f07e7c5af-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db6caae0f83e45e454e2215f07e7c5af-Abstract-Conference.html)

        **Abstract**:

        Point cloud completion is an active research topic for 3D vision and has been widelystudied in recent years. Instead of directly predicting missing point cloud fromthe partial input, we introduce a Semantic-Prototype Variational Transformer(SPoVT) in this work, which takes both partial point cloud and their semanticlabels as the inputs for semantic point cloud object completion. By observingand attending at geometry and semantic information as input features, our SPoVTwould derive point cloud features and their semantic prototypes for completionpurposes. As a result, our SPoVT not only performs point cloud completion withvarying resolution, it also allows manipulation of different semantic parts of anobject. Experiments on benchmark datasets would quantitatively and qualitativelyverify the effectiveness and practicality of our proposed model.

        ----

        ## [2459] Improving Intrinsic Exploration with Language Abstractions

        **Authors**: *Jesse Mu, Victor Zhong, Roberta Raileanu, Minqi Jiang, Noah D. Goodman, Tim Rocktäschel, Edward Grefenstette*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/db8cf88ced2536017980998929ee0fdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/db8cf88ced2536017980998929ee0fdf-Abstract-Conference.html)

        **Abstract**:

        Reinforcement learning (RL) agents are particularly hard to train when rewards are sparse. One common solution is to use intrinsic rewards to encourage agents to explore their environment. However, recent intrinsic exploration methods often use state-based novelty measures which reward low-level exploration and may not scale to domains requiring more abstract skills. Instead, we explore natural language as a general medium for highlighting relevant abstractions in an environment. Unlike previous work, we evaluate whether language can improve over existing exploration methods by directly extending (and comparing to) competitive intrinsic exploration baselines: AMIGo (Campero et al., 2021) and NovelD (Zhang et al., 2021). These language-based variants outperform their non-linguistic forms by 47-85% across 13 challenging tasks from the MiniGrid and MiniHack environment suites.

        ----

        ## [2460] The trade-offs of model size in large recommendation models : 100GB to 10MB Criteo-tb DLRM model

        **Authors**: *Aditya Desai, Anshumali Shrivastava*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbae915128892556134f1c5375855590-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbae915128892556134f1c5375855590-Abstract-Conference.html)

        **Abstract**:

        Embedding tables dominate industrial-scale recommendation model sizes, using up to terabytes of memory. A popular and the largest publicly available machine learning MLPerf benchmark on recommendation data is a Deep Learning Recommendation Model (DLRM) trained on a terabyte of click-through data. It contains 100GB of embedding memory (25+Billion parameters). DLRMs, due to their sheer size and the associated volume of data, face difficulty in training, deploying for inference, and memory bottlenecks due to large embedding tables. This paper analyzes and extensively evaluates a generic parameter-sharing setup (PSS) for compressing DLRM models. We show theoretical upper bounds on the learnable memory requirements for achieving approximations to the embedding table. Our bounds indicate exponentially fewer parameters suffice for a good approximation. To this end, we demonstrate a PSS DLRM reaching 10000$\times$ compression on criteo-tb without losing quality. Such a compression, however, comes with a caveat. It requires 4.5 $\times$ more iterations to achieve the same saturation quality. The paper argues that this tradeoff needs more investigation as it might be significantly favorable. Leveraging the small size of the compressed model, we show a 4.3$\times$ improvement in training latency leading to similar overall training times. Thus, in the tradeoff between the system advantage of a small DLRM model vs. slower convergence, we show that scales are tipped towards having a smaller DLRM model, leading to the same quality, faster inference, easier deployment, and similar training times.

        ----

        ## [2461] Near-Optimal Goal-Oriented Reinforcement Learning in Non-Stationary Environments

        **Authors**: *Liyu Chen, Haipeng Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbb5180957513805ebeea787b8c66ac9-Abstract-Conference.html)

        **Abstract**:

        We initiate the study of dynamic regret minimization for goal-oriented reinforcement learning modeled by a non-stationary stochastic shortest path problem with changing cost and transition functions.We start by establishing a lower bound $\Omega((B_{\star} SAT_{\star}(\Delta_c + B_{\star}^2\Delta_P))^{1/3}K^{2/3})$, where $B_{\star}$ is the maximum expected cost of the optimal policy of any episode starting from any state, $T_{\star}$ is the maximum hitting time of the optimal policy of any episode starting from the initial state, $SA$ is the number of state-action pairs, $\Delta_c$ and $\Delta_P$ are the amount of changes of the cost and transition functions respectively, and $K$ is the number of episodes.The different roles of $\Delta_c$ and $\Delta_P$ in this lower bound inspire us to design algorithms that estimate costs and transitions separately.Specifically, assuming the knowledge of $\Delta_c$ and $\Delta_P$, we develop a simple but sub-optimal algorithm and another more involved minimax optimal algorithm (up to logarithmic terms).These algorithms combine the ideas of finite-horizon approximation [Chen et al., 2021b], special Bernstein-style bonuses of the MVP algorithm [Zhang et al., 2020], adaptive confidence widening [Wei and Luo, 2021], as well as some new techniques such as properly penalizing long-horizon policies.Finally, when $\Delta_c$ and $\Delta_P$ are unknown, we develop a variant of the MASTER algorithm [Wei and Luo, 2021] and integrate the aforementioned ideas into it to achieve $\widetilde{O}(\min\{B_{\star} S\sqrt{ALK}, (B_{\star}^2S^2AT_{\star}(\Delta_c+B_{\star}\Delta_P))^{1/3}K^{2/3}\})$ regret, where $L$ is the unknown number of changes of the environment.

        ----

        ## [2462] A Unified Framework for Deep Symbolic Regression

        **Authors**: *Mikel Landajuela, Chak Shing Lee, Jiachen Yang, Ruben Glatt, Cláudio P. Santiago, Ignacio Aravena, Terrell Nathan Mundhenk, Garrett Mulcahy, Brenden K. Petersen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbca58f35bddc6e4003b2dd80e42f838-Abstract-Conference.html)

        **Abstract**:

        The last few years have witnessed a surge in methods for symbolic regression, from advances in traditional evolutionary approaches to novel deep learning-based systems. Individual works typically focus on advancing the state-of-the-art for one particular class of solution strategies, and there have been few attempts to investigate the benefits of hybridizing or integrating multiple strategies. In this work, we identify five classes of symbolic regression solution strategies---recursive problem simplification, neural-guided search, large-scale pre-training, genetic programming, and linear models---and propose a strategy to hybridize them into a single modular, unified symbolic regression framework. Based on empirical evaluation using SRBench, a new community tool for benchmarking symbolic regression methods, our unified framework achieves state-of-the-art performance in its ability to (1) symbolically recover analytical expressions, (2) fit datasets with high accuracy, and (3) balance accuracy-complexity trade-offs, across 252 ground-truth and black-box benchmark problems, in both noiseless settings and across various noise levels. Finally, we provide practical use case-based guidance for constructing hybrid symbolic regression algorithms, supported by extensive, combinatorial ablation studies.

        ----

        ## [2463] VoxGRAF: Fast 3D-Aware Image Synthesis with Sparse Voxel Grids

        **Authors**: *Katja Schwarz, Axel Sauer, Michael Niemeyer, Yiyi Liao, Andreas Geiger*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbdc7a9779ce0278c6e43b62c7e97759-Abstract-Conference.html)

        **Abstract**:

        State-of-the-art 3D-aware generative models rely on coordinate-based MLPs to parameterize 3D radiance fields. While demonstrating impressive results, querying an MLP for every sample along each ray leads to slow rendering.Therefore, existing approaches often render low-resolution feature maps and process them with an upsampling network to obtain the final image. Albeit efficient, neural rendering often entangles viewpoint and content such that changing the camera pose results in unwanted changes of geometry or appearance.Motivated by recent results in voxel-based novel view synthesis, we investigate the utility of sparse voxel grid representations for fast and 3D-consistent generative modeling in this paper.Our results demonstrate that monolithic MLPs can indeed be replaced by 3D convolutions when combining sparse voxel grids with progressive growing, free space pruning and appropriate regularization.To obtain a compact representation of the scene and allow for scaling to higher voxel resolutions, our model disentangles the foreground object (modeled in 3D) from the background (modeled in 2D).In contrast to existing approaches, our method requires only a single forward pass to generate a full 3D scene. It hence allows for efficient rendering from arbitrary viewpoints while yielding 3D consistent results with high visual fidelity. Code and models are available at https://github.com/autonomousvision/voxgraf.

        ----

        ## [2464] Sparse Hypergraph Community Detection Thresholds in Stochastic Block Model

        **Authors**: *Erchuan Zhang, David Suter, Giang Truong, Syed Zulqarnain Gilani*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbdea7859f1d2fc10f2c9e79b8f5ae54-Abstract-Conference.html)

        **Abstract**:

        Community detection in random graphs or hypergraphs is an interesting fundamental problem in statistics, machine learning and computer vision. When the hypergraphs are generated by a {\em stochastic block model}, the existence of a sharp threshold on the model parameters for community detection was conjectured by Angelini et al. 2015. In this paper, we confirm the positive part of the conjecture, the possibility of non-trivial reconstruction above the threshold, for the case of two blocks. We do so by comparing the hypergraph stochastic block model with its Erd{\"o}s-R{\'e}nyi counterpart. We also obtain estimates for the parameters of the hypergraph stochastic block model. The methods developed in this paper are generalised from the study of sparse random graphs by Mossel et al. 2015 and are motivated by the work of Yuan et al. 2022. Furthermore, we present some discussion on the negative part of the conjecture, i.e., non-reconstruction of community structures.

        ----

        ## [2465] Understanding Non-linearity in Graph Neural Networks from the Bayesian-Inference Perspective

        **Authors**: *Rongzhe Wei, Haoteng Yin, Junteng Jia, Austin R. Benson, Pan Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbe0e575e4604367a989e850c9b28401-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbe0e575e4604367a989e850c9b28401-Abstract-Conference.html)

        **Abstract**:

        Graph neural networks (GNNs) have shown superiority in many prediction tasks over graphs due to their impressive capability of capturing nonlinear relations in graph-structured data. However, for node classification tasks, often, only marginal improvement of GNNs has been observed in practice over their linear counterparts. Previous works provide very few understandings of this phenomenon. In this work, we resort to Bayesian learning to give an in-depth investigation of the functions of non-linearity in GNNs for node classification tasks. Given a graph generated from the statistical model CSBM, we observe that the max-a-posterior estimation of a node label given its own and neighbors' attributes consists of two types of non-linearity, the transformation of node attributes and a ReLU-activated feature aggregation from neighbors. The latter surprisingly matches the type of non-linearity used in many GNN models. By further imposing Gaussian assumption on node attributes, we prove that the superiority of those ReLU activations is only significant when the node attributes are far more informative than the graph structure, which nicely explains previous empirical observations. A similar argument is derived when there is a distribution shift of node attributes between the training and testing datasets. Finally, we verify our theory on both synthetic and real-world networks. Our code is available at https://github.com/Graph-COM/Bayesian_inference_based_GNN.git.

        ----

        ## [2466] Explainable Reinforcement Learning via Model Transforms

        **Authors**: *Mira Finkelstein, Nitsan Levy Schlot, Lucy Liu, Yoav Kolumbus, David C. Parkes, Jeffrey S. Rosenschein, Sarah Keren*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dbef234be68d8b170240511639610fd1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dbef234be68d8b170240511639610fd1-Abstract-Conference.html)

        **Abstract**:

        Understanding emerging behaviors of reinforcement learning (RL) agents may be difficult since such agents are often trained in complex environments using highly complex decision making procedures. This has given rise to a variety of approaches to explainability in RL that aim to reconcile discrepancies that may arise between the behavior of an agent and the behavior that is anticipated by an observer. Most recent approaches have relied either on domain knowledge, that may not always be available, on an analysis of the agentâ€™s policy, or on an analysis of specific elements of the underlying environment, typically modeled as a Markov Decision Process (MDP). Our key claim is that even if the underlying model is not fully known (e.g., the transition probabilities have not been accurately learned) or is not maintained by the agent (i.e., when using model-free methods), the model can nevertheless be exploited to automatically generate explanations. For this purpose, we suggest using formal MDP abstractions and transforms, previously used in the literature for expediting the search for optimal policies, to automatically produce explanations. Since such transforms are typically based on a symbolic representation of the environment, they can provide meaningful explanations for gaps between the anticipated and actual agent behavior. We formally define the explainability problem, suggest a class of transforms that can be used for explaining emergent behaviors, and suggest methods that enable efficient search for an explanation. We demonstrate the approach on a set of standard benchmarks.

        ----

        ## [2467] Real-Valued Backpropagation is Unsuitable for Complex-Valued Neural Networks

        **Authors**: *Zhi-Hao Tan, Yi Xie, Yuan Jiang, Zhi-Hua Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc06d4d2792265fb5454a6092bfd5c6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc06d4d2792265fb5454a6092bfd5c6a-Abstract-Conference.html)

        **Abstract**:

        Recently complex-valued neural networks have received increasing attention due to successful applications in various tasks and the potential advantages of better theoretical properties and richer representational capacity. However, the training dynamics of complex networks compared to real networks remains an open problem. In this paper, we investigate the dynamics of deep complex networks during real-valued backpropagation in the infinite-width limit via neural tangent kernel (NTK). We first extend the Tensor Program to the complex domain, to show that the dynamics of any basic complex network architecture is governed by its NTK under real-valued backpropagation. Then we propose a way to investigate the comparison of training dynamics between complex and real networks by studying their NTKs. As a result, we surprisingly prove that for most complex activation functions, the commonly used real-valued backpropagation reduces the training dynamics of complex networks to that of ordinary real networks as the widths tend to infinity, thus eliminating the characteristics of complex-valued neural networks. Finally, the experiments validate our theoretical findings numerically.

        ----

        ## [2468] Learning dynamics of deep linear networks with multiple pathways

        **Authors**: *Jianghong Shi, Eric Shea-Brown, Michael A. Buice*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc3ca8bcd613e43ce540352b58d55d6d-Abstract-Conference.html)

        **Abstract**:

        Not only have deep networks become standard in machine learning, they are increasingly of interest in neuroscience as models of cortical computation that capture relationships between structural and functional properties.  In addition they are a useful target of theoretical research into the properties of network computation.  Deep networks typically have a serial or approximately serial organization across layers, and this is often mirrored in models that purport to represent computation in mammalian brains.  There are, however, multiple examples of parallel pathways in mammalian brains.  In some cases, such as the mouse, the entire visual system appears arranged in a largely parallel, rather than serial fashion.  While these pathways may be formed by differing cost functions that drive different computations, here we present a new mathematical analysis of learning dynamics in networks that have parallel computational pathways driven by the same cost function.  We use the approximation of deep linear networks with large hidden layer sizes to show that, as the depth of the parallel pathways increases, different features of the training set (defined by the singular values of the input-output correlation) will typically concentrate in one of the pathways.  This result is derived analytically and demonstrated with numerical simulation.  Thus, rather than sharing stimulus and task features across multiple pathways, parallel network architectures learn to produce sharply diversified representations with specialized and specific pathways, a mechanism which may hold important consequences for codes in both biological and artificial systems.

        ----

        ## [2469] Self-Supervised Aggregation of Diverse Experts for Test-Agnostic Long-Tailed Recognition

        **Authors**: *Yifan Zhang, Bryan Hooi, Lanqing Hong, Jiashi Feng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc6319dde4fb182b22fb902da9418566-Abstract-Conference.html)

        **Abstract**:

        Existing long-tailed recognition methods, aiming to train class-balanced models from long-tailed data, generally assume the models would be evaluated on the uniform test class distribution. However, practical test class distributions often violate this assumption (e.g., being either long-tailed or even inversely long-tailed), which may lead existing methods to fail in real applications. In this paper, we study a more practical yet challenging task, called test-agnostic long-tailed recognition, where the training class distribution is long-tailed while the test class distribution is agnostic and not necessarily uniform. In addition to the issue of class imbalance, this task poses another challenge: the class distribution shift between the training and test data is unknown. To tackle this task, we propose a novel approach, called Self-supervised Aggregation of Diverse Experts, which consists of two strategies: (i) a new skill-diverse expert learning strategy that trains multiple experts from a single and stationary long-tailed dataset to separately handle different class distributions; (ii) a novel test-time expert aggregation strategy that leverages self-supervision to aggregate the learned multiple experts for handling unknown test class distributions. We theoretically show that our self-supervised strategy has a provable ability to simulate test-agnostic class distributions. Promising empirical results demonstrate the effectiveness of our method on both vanilla and test-agnostic long-tailed recognition. The source code is available at https://github.com/Vanint/SADE-AgnosticLT.

        ----

        ## [2470] Self-Supervised Learning via Maximum Entropy Coding

        **Authors**: *Xin Liu, Zhongdao Wang, Yali Li, Shengjin Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc709714c52b35f2f34aca2a92b06bc8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc709714c52b35f2f34aca2a92b06bc8-Abstract-Conference.html)

        **Abstract**:

        A mainstream type of current self-supervised learning methods pursues a general-purpose representation that can be well transferred to downstream tasks, typically by optimizing on a given pretext task such as instance discrimination. In this work, we argue that existing pretext tasks inevitably introduce biases into the learned representation, which in turn leads to biased transfer performance on various downstream tasks. To cope with this issue, we propose Maximum Entropy Coding (MEC), a more principled objective that explicitly optimizes on the structure of the representation, so that the learned representation is less biased and thus generalizes better to unseen downstream tasks. Inspired by the principle of maximum entropy in information theory, we hypothesize that a generalizable representation should be the one that admits the maximum entropy among all plausible representations. To make the objective end-to-end trainable, we propose to leverage the minimal coding length in lossy data coding as a computationally tractable surrogate for the entropy, and further derive a scalable reformulation of the objective that allows fast computation. Extensive experiments demonstrate that MEC learns a more generalizable representation than previous methods based on specific pretext tasks. It achieves state-of-the-art performance consistently on various downstream tasks, including not only ImageNet linear probe, but also semi-supervised classification, object detection, instance segmentation, and object tracking. Interestingly, we show that existing batch-wise and feature-wise self-supervised objectives could be seen equivalent to low-order approximations of MEC. Code and pre-trained models are available at https://github.com/xinliu20/MEC.

        ----

        ## [2471] A Practical, Progressively-Expressive GNN

        **Authors**: *Lingxiao Zhao, Neil Shah, Leman Akoglu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc89a0709f213fd0ac4b1172719b2c38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc89a0709f213fd0ac4b1172719b2c38-Abstract-Conference.html)

        **Abstract**:

        Message passing neural networks (MPNNs) have become a dominant flavor of graph neural networks (GNNs) in recent years. Yet, MPNNs come with notable limitations; namely, they are at most as powerful as the 1-dimensional Weisfeiler-Leman (1-WL) test in distinguishing graphs in a graph isomorphism testing frame-work. To this end, researchers have drawn inspiration from the k-WL hierarchy to develop more expressive GNNs. However, current k-WL-equivalent GNNs are not practical for even small values of k, as k-WL becomes combinatorially more complex as k grows. At the same time, several works have found great empirical success in graph learning tasks without highly expressive models, implying that chasing expressiveness with a “coarse-grained ruler” of expressivity like k-WL is often unneeded in practical tasks. To truly understand the expressiveness-complexity tradeoff, one desires a more “fine-grained ruler,” which can more gradually increase expressiveness. Our work puts forth such a proposal: Namely, we first propose the (k, c)(≤)-SETWL hierarchy with greatly reduced complexity from k-WL, achieved by moving from k-tuples of nodes to sets with ≤k nodes defined over ≤c connected components in the induced original graph. We show favorable theoretical results for this model in relation to k-WL, and concretize it via (k, c)(≤)-SETGNN, which is as expressive as (k, c)(≤)-SETWL. Our model is practical and progressively-expressive, increasing in power with k and c. We demonstrate effectiveness on several benchmark datasets, achieving several state-of-the-art results with runtime and memory usage applicable to practical graphs. We open source our implementation at https://github.com/LingxiaoShawn/KCSetGNN.

        ----

        ## [2472] On Learning Fairness and Accuracy on Multiple Subgroups

        **Authors**: *Changjian Shui, Gezheng Xu, Qi Chen, Jiaqi Li, Charles X. Ling, Tal Arbel, Boyu Wang, Christian Gagné*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dc96134e169de5aea1ba1fc34dfb8419-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dc96134e169de5aea1ba1fc34dfb8419-Abstract-Conference.html)

        **Abstract**:

        We propose an analysis in fair learning that preserves the utility of the data while reducing prediction disparities under the criteria of group sufficiency. We focus on the scenario where the data contains multiple or even many subgroups, each with limited number of samples. As a result, we present a principled method for learning a fair predictor for all subgroups via formulating it as a bilevel objective. Specifically, the subgroup specific predictors are learned in the lower-level through a small amount of data and the fair predictor. In the upper-level, the fair predictor is updated to be close to all subgroup specific predictors. We further prove that such a bilevel objective can effectively control the group sufficiency and generalization error. We evaluate the proposed framework on real-world datasets. Empirical evidence suggests the consistently improved fair predictions, as well as the comparable accuracy to the baselines.

        ----

        ## [2473] Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition

        **Authors**: *Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dccbeb7a8df3065c4646928985edf435-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dccbeb7a8df3065c4646928985edf435-Abstract-Conference.html)

        **Abstract**:

        Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.

        ----

        ## [2474] Understanding Aesthetics with Language: A Photo Critique Dataset for Aesthetic Assessment

        **Authors**: *Daniel Vera Nieto, Luigi Celona, Clara Fernandez-Labrador*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/dcd18e50ebca0af89187c6e35dabb584-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Computational inference of aesthetics is an ill-defined task due to its subjective nature. Many datasets have been proposed to tackle the problem by providing pairs of images and aesthetic scores based on human ratings. However, humans are better at expressing their opinion, taste, and emotions by means of language rather than summarizing them in a single number. In fact, photo critiques provide much richer information as they reveal how and why users rate the aesthetics of visual stimuli. In this regard, we propose the Reddit Photo Critique Dataset (RPCD), which contains tuples of image and photo critiques. RPCD consists of 74K images and 220K comments and is collected from a Reddit community used by hobbyists and professional photographers to improve their photography skills by leveraging constructive community feedback. The proposed dataset differs from previous aesthetics datasets mainly in three aspects, namely (i) the large scale of the dataset and the extension of the comments criticizing different aspects of the image, (ii) it contains mostly UltraHD images, and (iii) it can easily be extended to new data as it is collected through an automatic pipeline. To the best of our knowledge, in this work, we propose the first attempt to estimate the aesthetic quality of visual stimuli from the critiques. To this end, we exploit the polarity of the sentiment of criticism as an indicator of aesthetic judgment. We demonstrate how sentiment polarity correlates positively with the aesthetic judgment available for two aesthetic assessment benchmarks. Finally, we experiment with several models by using the sentiment scores as a target for ranking images. Dataset and baselines are available https://github.com/mediatechnologycenter/aestheval.

        ----

        ## [2475] Structuring Representations Using Group Invariants

        **Authors**: *Mehran Shakerinava, Arnab Kumar Mondal, Siamak Ravanbakhsh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dcd297696d0bb304ba426b3c5a679c37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dcd297696d0bb304ba426b3c5a679c37-Abstract-Conference.html)

        **Abstract**:

        A finite set of invariants can identify many interesting transformation groups. For example, distances, inner products and angles are preserved by Euclidean, Orthogonal and Conformal transformations, respectively. In an equivariant representation, the group invariants should remain constant on the embedding as we transform the input. This gives a procedure for learning equivariant representations without knowing the possibly nonlinear action of the group in the input space. Rather than enforcing such hard invariance constraints on the latent space, we show how to use invariants for "symmetry regularization" of the latent, while guaranteeing equivariance through other means. We also show the feasibility of learning disentangled representations using this approach and provide favorable qualitative and quantitative results on downstream tasks, including world modeling and reinforcement learning.

        ----

        ## [2476] Black-box coreset variational inference

        **Authors**: *Dionysis Manousakas, Hippolyt Ritter, Theofanis Karaletsos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dce5d66bf88b7b7d0f11fbd9c9f16620-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dce5d66bf88b7b7d0f11fbd9c9f16620-Abstract-Conference.html)

        **Abstract**:

        Recent advances in coreset methods have shown that a selection of representative datapoints can replace massive volumes of data for Bayesian inference, preserving the relevant statistical information and significantly accelerating subsequent downstream tasks. Existing variational coreset constructions rely on either selecting subsets of the observed datapoints, or jointly performing approximate inference and optimizing pseudodata in the observed space akin to inducing points methods in Gaussian Processes. So far, both approaches are limited by complexities in evaluating their objectives for general purpose models, and require generating samples from a typically intractable posterior over the coreset throughout inference and testing.  In this work, we present a black-box variational inference framework for coresets that overcomes these constraints and enables principled application of variational coresets to intractable models, such as Bayesian neural networks. We apply our techniques to supervised learning problems, and compare them with existing approaches in the literature for data summarization and inference.

        ----

        ## [2477] Distilling Representations from GAN Generator via Squeeze and Span

        **Authors**: *Yu Yang, Xiaotian Cheng, Chang Liu, Hakan Bilen, Xiangyang Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd0151081b1e80e93f1b8aaf0b684c18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd0151081b1e80e93f1b8aaf0b684c18-Abstract-Conference.html)

        **Abstract**:

        In recent years, generative adversarial networks (GANs) have been an actively studied topic and shown to successfully produce high-quality realistic images in various domains. The controllable synthesis ability of GAN generators suggests that they maintain informative, disentangled, and explainable image representations, but leveraging and transferring their representations to downstream tasks is largely unexplored. In this paper, we propose to distill knowledge from GAN generators by squeezing and spanning their representations. We \emph{squeeze} the generator features into representations that are invariant to semantic-preserving transformations through a network before they are distilled into the student network. We \emph{span} the distilled representation of the synthetic domain to the real domain by also using real training data to remedy the mode collapse of GANs and boost the student network performance in a real domain. Experiments justify the efficacy of our method and reveal its great significance in self-supervised representation learning. Code is available at https://github.com/yangyu12/squeeze-and-span.

        ----

        ## [2478] A Quantitative Geometric Approach to Neural-Network Smoothness

        **Authors**: *Zi Wang, Gautam Prakriya, Somesh Jha*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd1322ce23cbbdd9d7ebb0ad1223c27a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd1322ce23cbbdd9d7ebb0ad1223c27a-Abstract-Conference.html)

        **Abstract**:

        Fast and precise Lipschitz constant estimation of neural networks is an important task for deep learning. Researchers have recently found an intrinsic trade-off between the accuracy and smoothness of neural networks, so training a network with a loose Lipschitz constant estimation imposes a strong regularization, and can hurt the model accuracy significantly. In this work, we provide a unified theoretical framework, a quantitative geometric approach, to address the Lipschitz constant estimation. By adopting this framework, we can immediately obtain several theoretical results, including the computational hardness of Lipschitz constant estimation and its approximability. We implement the algorithms induced from this quantitative geometric approach, which are based on semidefinite programming (SDP). Our empirical evaluation demonstrates that they are more scalable and precise than existing tools on Lipschitz constant estimation for $\ell_\infty$-perturbations. Furthermore, we also show their intricate relations with other recent SDP-based techniques, both theoretically and empirically. We believe that this unified quantitative geometric perspective can bring new insights and theoretical tools to the investigation of neural-network smoothness and robustness.

        ----

        ## [2479] AutoMTL: A Programming Framework for Automating Efficient Multi-Task Learning

        **Authors**: *Lijun Zhang, Xiao Liu, Hui Guan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd3bd4e35cdd224ed2153a996f41077f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd3bd4e35cdd224ed2153a996f41077f-Abstract-Conference.html)

        **Abstract**:

        Multi-task learning (MTL) jointly learns a set of tasks by sharing parameters among tasks. It is a promising approach for reducing storage costs while improving task accuracy for many computer vision tasks. The effective adoption of MTL faces two main challenges. The first challenge is to determine what parameters to share across tasks to optimize for both memory efficiency and task accuracy. The second challenge is to automatically apply MTL algorithms to an arbitrary CNN backbone without requiring time-consuming manual re-implementation and significant domain expertise. This paper addresses the challenges by developing the first programming framework AutoMTL that automates efficient MTL model development for vision tasks. AutoMTL takes as inputs an arbitrary backbone convolutional neural network (CNN) and a set of tasks to learn, and automatically produces a multi-task model that achieves high accuracy and small memory footprint simultaneously. Experiments on three popular MTL benchmarks (CityScapes, NYUv2, Tiny-Taskonomy) demonstrate the effectiveness of AutoMTL over state-of-the-art approaches as well as the generalizability of AutoMTL across CNNs. AutoMTL is open-sourced and available at https://github.com/zhanglijun95/AutoMTL.

        ----

        ## [2480] Log-Concave and Multivariate Canonical Noise Distributions for Differential Privacy

        **Authors**: *Jordan Awan, Jinshuo Dong*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd73933d99ccd7ffe2306adb95ec5d02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd73933d99ccd7ffe2306adb95ec5d02-Abstract-Conference.html)

        **Abstract**:

        A canonical noise distribution (CND) is an additive mechanism designed to satisfy $f$-differential privacy ($f$-DP), without any wasted privacy budget. $f$-DP is a hypothesis testing-based formulation of privacy phrased in terms of tradeoff functions, which captures the difficulty of a hypothesis test. In this paper, we consider the existence and construction of both log-concave CNDs and multivariate CNDs. Log-concave distributions are important to ensure that higher outputs of the mechanism correspond to higher input values, whereas multivariate noise distributions are important to ensure that a joint release of multiple outputs has a tight privacy characterization. We show that the existence and construction of CNDs for both types of problems is related to whether the tradeoff function can be decomposed by functional composition (related to group privacy) or mechanism composition. In particular, we show that pure $\epsilon$-DP cannot be decomposed in either way and that there is neither a log-concave CND nor any multivariate CND for $\epsilon$-DP. On the other hand, we show that Gaussian-DP, $(0,\delta)$-DP, and Laplace-DP each have both log-concave and multivariate CNDs.

        ----

        ## [2481] On the generalization of learning algorithms that do not converge

        **Authors**: *Nisha Chandramoorthy, Andreas Loukas, Khashayar Gatmiry, Stefanie Jegelka*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd73f39426a03131c38c8d943153d44b-Abstract-Conference.html)

        **Abstract**:

        Generalization analyses of deep learning typically assume that the training converges to a fixed point. But, recent results indicate that in practice, the weights of deep neural networks optimized with stochastic gradient descent often oscillate indefinitely. To reduce this discrepancy between theory and practice, this paper focuses on the generalization of neural networks whose training dynamics do not necessarily converge to fixed points.  Our main contribution is to propose a notion of statistical algorithmic stability (SAS) that extends classical algorithmic stability to non-convergent algorithms and to study its connection to generalization. This ergodic-theoretic approach leads to new insights when compared to the traditional optimization and learning theory perspectives. We prove that the stability of the time-asymptotic behavior of a learning algorithm relates to its generalization and empirically demonstrate how loss dynamics can provide clues to generalization performance. Our findings provide evidence that networks that ``train stably generalize better'' even when the training continues indefinitely and the weights do not converge.

        ----

        ## [2482] PALMER: Perception - Action Loop with Memory for Long-Horizon Planning

        **Authors**: *Onur Beker, Mohammad Mohammadi, Amir Zamir*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dd7a48c862b800f0537fe1d506e641b5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dd7a48c862b800f0537fe1d506e641b5-Abstract-Conference.html)

        **Abstract**:

        To achieve autonomy in a priori unknown real-world scenarios, agents should be able to: i) act from high-dimensional sensory observations (e.g., images), ii) learn from past experience to adapt and improve, and iii) be capable of long horizon planning. Classical planning algorithms (e.g. PRM, RRT) are proficient at handling long-horizon planning. Deep learning based methods in turn can provide the necessary representations to address the others, by modeling statistical contingencies between observations. In this direction, we introduce a general-purpose planning algorithm called PALMER that combines classical sampling-based planning algorithms with learning-based perceptual representations. For training these perceptual representations, we combine Q-learning with contrastive representation learning to create a latent space where the distance between the embeddings of two states captures how easily an optimal policy can traverse between them. For planning with these perceptual representations, we re-purpose classical sampling-based planning algorithms to retrieve previously observed trajectory segments from a replay buffer and restitch them into approximately optimal paths that connect any given pair of start and goal states. This creates a tight feedback loop between representation learning, memory, reinforcement learning, and sampling-based planning. The end result is an experiential framework for long-horizon planning that is significantly more robust and sample efficient compared to existing methods.

        ----

        ## [2483] Leveraging Factored Action Spaces for Efficient Offline Reinforcement Learning in Healthcare

        **Authors**: *Shengpu Tang, Maggie Makar, Michael W. Sjoding, Finale Doshi-Velez, Jenna Wiens*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html)

        **Abstract**:

        Many reinforcement learning (RL) applications have combinatorial action spaces, where each action is a composition of sub-actions. A standard RL approach ignores this inherent factorization structure, resulting in a potential failure to make meaningful inferences about rarely observed sub-action combinations; this is particularly problematic for offline settings, where data may be limited. In this work, we propose a form of linear Q-function decomposition induced by factored action spaces. We study the theoretical properties of our approach, identifying scenarios where it is guaranteed to lead to zero bias when used to approximate the Q-function. Outside the regimes with theoretical guarantees, we show that our approach can still be useful because it leads to better sample efficiency without necessarily sacrificing policy optimality, allowing us to achieve a better bias-variance trade-off. Across several offline RL problems using simulators and real-world datasets motivated by healthcare, we demonstrate that incorporating factored action spaces into value-based RL can result in better-performing policies. Our approach can help an agent make more accurate inferences within underexplored regions of the state-action space when applying RL to observational datasets.

        ----

        ## [2484] Visual correspondence-based explanations improve AI robustness and human-AI team accuracy

        **Authors**: *Mohammad Reza Taesiri, Giang Nguyen, Anh Nguyen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ddb8486bf9ee0fdeca1866a13a96e98e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ddb8486bf9ee0fdeca1866a13a96e98e-Abstract-Conference.html)

        **Abstract**:

        Explaining artificial intelligence (AI) predictions is increasingly important and even imperative in many high-stake applications where humans are the ultimate decision-makers. In this work, we propose two novel architectures of explainable image classifiers that first explain, and then predict (as opposed to post-hoc explanation methods). Our models first rank the training-set images by their distance with the query in an image-level deep feature space. And then, we re-rank the top-50 shortlisted candidates using patch-wise similarity of 5 highest-similarity pairs of patches between the query and every candidate. On ImageNet, our models improve (by 1-4 points) the out-of-distribution accuracy on several datasets including Adversarial Patch and ImageNet-R while performing marginally worse (by 1-2 points) on ImageNet to the baselines (ResNet-50 pre-trained ImageNet). A consistent trend is observed on CUB. Via a large-scale, human study (~60 users per method per dataset) on ImageNet and CUB, we find our proposed correspondence-based explanations led to human-alone image classification accuracy and human-AI team accuracy that are consistently better than those of k-NN. Our correspondence-based explanations help users better correctly reject AI's wrong decisions than all other tested methods.Interestingly, for the first time, we show that it is possible to achieve complementary human-AI team accuracy (i.e. that is higher than either AI-alone or human-alone), in both image classification tasks.

        ----

        ## [2485] Invariance-Aware Randomized Smoothing Certificates

        **Authors**: *Jan Schuchardt, Stephan Günnemann*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ddd45979547a35db2471e69cbf3bca54-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ddd45979547a35db2471e69cbf3bca54-Abstract-Conference.html)

        **Abstract**:

        Building models that comply with the invariances inherent to different domains, such as invariance under translation or rotation, is a key aspect of applying machine learning to real world problems like molecular property prediction, medical imaging, protein folding or LiDAR classification. For the first time, we study how the invariances of a model can be leveraged to provably guarantee the robustness of its predictions. We propose a gray-box approach, enhancing the powerful black-box randomized smoothing technique with white-box knowledge about invariances. First, we develop gray-box certificates based on group orbits, which can be applied to arbitrary models with invariance under permutation and Euclidean isometries. Then, we derive provably tight gray-box certificates. We experimentally demonstrate that the provably tight certificates can offer much stronger guarantees, but that in practical scenarios the orbit-based method is a good approximation.

        ----

        ## [2486] On Image Segmentation With Noisy Labels: Characterization and Volume Properties of the Optimal Solutions to Accuracy and Dice

        **Authors**: *Marcus Nordström, Henrik Hult, Fredrik Löfman, Jonas Söderberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ddf6eeeaa92957d3100b217a4428d819-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ddf6eeeaa92957d3100b217a4428d819-Abstract-Conference.html)

        **Abstract**:

        We study two of the most popular performance metrics in medical image segmentation, Accuracy and Dice, when the target labels are noisy. For both metrics, several statements related to characterization and volume properties of the set of optimal segmentations are proved, and associated experiments are provided. Our main insights are: (i) the volume of the solutions to both metrics may deviate significantly from the expected volume of the target, (ii) the volume of a solution to Accuracy is always less than or equal to the volume of a solution to Dice and (iii) the optimal solutions to both of these metrics coincide when the set of feasible segmentations is constrained to the set of segmentations with the volume equal to the expected volume of the target.

        ----

        ## [2487] Blessing of Depth in Linear Regression: Deeper Models Have Flatter Landscape Around the True Solution

        **Authors**: *Jianhao Ma, Salar Fattahi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html)

        **Abstract**:

        This work characterizes the effect of depth on the optimization landscape of linear regression, showing that, despite their nonconvexity, deeper models have more desirable optimization landscape. We consider a robust and over-parameterized setting, where a subset of measurements are grossly corrupted with noise, and the true linear model is captured via an $N$-layer diagonal linear neural network. On the negative side, we show that this problem does not have a benign landscape: given any $N\geq 1$, with constant probability, there exists a solution corresponding to the ground truth that is neither local nor global minimum. However, on the positive side, we prove that, for any $N$-layer model with $N\geq 2$, a simple sub-gradient method becomes oblivious to such “problematic” solutions; instead, it converges to a balanced solution that is not only close to the ground truth but also enjoys a flat local landscape, thereby eschewing the need for “early stopping”. Lastly, we empirically verify that the desirable optimization landscape of deeper models extends to other robust learning tasks, including deep matrix recovery and deep ReLU networks with $\ell_1$-loss.

        ----

        ## [2488] Fairness Reprogramming

        **Authors**: *Guanhua Zhang, Yihua Zhang, Yang Zhang, Wenqi Fan, Qing Li, Sijia Liu, Shiyu Chang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de08b3ee7c0043a76ee4a44fe68e90bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de08b3ee7c0043a76ee4a44fe68e90bc-Abstract-Conference.html)

        **Abstract**:

        Despite a surge of recent advances in promoting machine Learning (ML) fairness, the existing mainstream approaches mostly require training or finetuning the entire weights of the neural network to meet the fairness criteria.  However, this is often infeasible in practice for those large-scale trained models due to large computational and storage costs, low data efficiency, and model privacy issues.  In this paper, we propose a new generic fairness learning paradigm, called FairReprogram, which incorporates the model reprogramming technique.  Specifically, FairReprogram considers the case where models can not be changed and appends to the input a set of perturbations, called the fairness trigger, which is tuned towards the fairness criteria under a min-max formulation.  We further introduce an information-theoretic framework that explains why and under what conditions fairness goals can be achieved using the fairness trigger.  We show both theoretically and empirically that the fairness trigger can effectively obscure demographic biases in the output prediction of fixed ML models by providing false demographic information that hinders the model from utilizing the correct demographic information to make the prediction.  Extensive experiments on both NLP and CV datasets demonstrate that our method can achieve better fairness improvements than retraining-based methods with far less data dependency under two widely-used fairness criteria. Codes are available at https://github.com/UCSB-NLP-Chang/Fairness-Reprogramming.git.

        ----

        ## [2489] WeightedSHAP: analyzing and improving Shapley based feature attributions

        **Authors**: *Yongchan Kwon, James Y. Zou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de1739eba209c682a90ec3669229ab2d-Abstract-Conference.html)

        **Abstract**:

        Shapley value is a popular approach for measuring the influence of individual features. While Shapley feature attribution is built upon desiderata from game theory, some of its constraints may be less natural in certain machine learning settings, leading to unintuitive model interpretation. In particular, the Shapley value uses the same weight for all marginal contributions---i.e. it gives the same importance when a large number of other features are given versus when a small number of other features are given. This property can be problematic if larger feature sets are more or less informative than smaller feature sets. Our work performs a rigorous analysis of the potential limitations of Shapley feature attribution. We identify simple settings where the Shapley value is mathematically suboptimal by assigning larger attributions for less influential features. Motivated by this observation, we propose WeightedSHAP, which generalizes the Shapley value and learns which marginal contributions to focus directly from data. On several real-world datasets, we demonstrate that the influential features identified by WeightedSHAP are better able to recapitulate the model's predictions compared to the features identified by the Shapley value.

        ----

        ## [2490] Temporal Effective Batch Normalization in Spiking Neural Networks

        **Authors**: *Chaoteng Duan, Jianhao Ding, Shiyan Chen, Zhaofei Yu, Tiejun Huang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html)

        **Abstract**:

        Spiking Neural Networks (SNNs) are promising in neuromorphic hardware owing to utilizing spatio-temporal information and sparse event-driven signal processing. However, it is challenging to train SNNs due to the non-differentiable nature of the binary firing function. The surrogate gradients alleviate the training problem and make SNNs obtain comparable performance as Artificial Neural Networks (ANNs) with the same structure. Unfortunately, batch normalization, contributing to the success of ANNs, does not play a prominent role in SNNs because of the additional temporal dimension. To this end, we propose an effective normalization method called temporal effective batch normalization (TEBN). By rescaling the presynaptic inputs with different weights at every time-step, temporal distributions become smoother and uniform. Theoretical analysis shows that TEBN can be viewed as a smoother of SNN's optimization landscape and could help stabilize the gradient norm. Experimental results on both static and neuromorphic datasets show that SNNs with TEBN outperform the state-of-the-art accuracy with fewer time-steps, and achieve better robustness to hyper-parameters than other normalizations.

        ----

        ## [2491] Remember the Past: Distilling Datasets into Addressable Memories for Neural Networks

        **Authors**: *Zhiwei Deng, Olga Russakovsky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de3d2bb604cfc43c81edd2a31b257f03-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de3d2bb604cfc43c81edd2a31b257f03-Abstract-Conference.html)

        **Abstract**:

        We propose an algorithm that compresses the critical information of a large dataset into compact addressable memories. These memories can then be recalled to quickly re-train a neural network and recover the performance (instead of storing and re-training on the full original dataset). Building upon the dataset distillation framework, we make a key observation that a shared common representation allows for more efficient and effective distillation. Concretely, we learn a set of bases (aka ``memories'') which are shared between classes and combined through learned flexible addressing functions to generate a diverse set of training examples. This leads to several benefits: 1) the size of compressed data does not necessarily grow linearly with the number of classes; 2) an overall higher compression rate with more effective distillation is achieved; and 3) more generalized queries are allowed beyond recalling the original classes. We demonstrate state-of-the-art results on the dataset distillation task across five benchmarks, including up to 16.5% and 9.7% accuracy improvement when distilling CIFAR10 and CIFAR100 respectively. We then leverage our framework to perform continual learning, achieving state-of-the-art results on four benchmarks, with 23.2% accuracy improvement on MANY.

        ----

        ## [2492] Robustness Analysis of Video-Language Models Against Visual and Language Perturbations

        **Authors**: *Madeline Schiappa, Shruti Vyas, Hamid Palangi, Yogesh S. Rawat, Vibhav Vineet*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/de6ff07cbd222c10d694c2b2f732aceb-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Joint visual and language modeling on large-scale datasets has recently shown good progress in multi-modal tasks when compared to single modal learning. However, robustness of these  approaches against real-world perturbations has not been studied. In this work, we perform the first extensive robustness study of video-language models against various real-world perturbations. We focus on text-to-video retrieval and propose two large-scale benchmark datasets, MSRVTT-P and YouCook2-P, which utilize 90 different visual and 35 different text perturbations. The study reveals some interesting initial findings from the studied models: 1) models are more robust when text is perturbed versus when video is perturbed, 2) models that are pre-trained are more robust than those trained from scratch, 3) models attend more to scene and objects rather than motion and action. We hope this study will serve as a benchmark and guide future research in robust video-language learning. The benchmark introduced in this study along with the code and datasets is available at https://bit.ly/3CNOly4.

        ----

        ## [2493] Hyperbolic Feature Augmentation via Distribution Estimation and Infinite Sampling on Manifolds

        **Authors**: *Zhi Gao, Yuwei Wu, Yunde Jia, Mehrtash Harandi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de7858e3e7f9f0f7b2c7bfdc86f6d928-Abstract-Conference.html)

        **Abstract**:

        Learning in hyperbolic spaces has attracted growing attention recently, owing to their capabilities in capturing hierarchical structures of data. However, existing learning algorithms in the hyperbolic space tend to overfit when limited data is given. In this paper, we propose a hyperbolic feature augmentation method that generates diverse and discriminative features in the hyperbolic space to combat overfitting. We employ a wrapped hyperbolic normal distribution to model augmented features, and use a neural ordinary differential equation module that benefits from meta-learning to estimate the distribution. This is to reduce the bias of estimation caused by the scarcity of data. We also derive an upper bound of the augmentation loss, which enables us to train a hyperbolic model by using an infinite number of augmentations. Experiments on few-shot learning and continual learning tasks show that our method significantly improves the performance of hyperbolic algorithms in scarce data regimes.

        ----

        ## [2494] PatchComplete: Learning Multi-Resolution Patch Priors for 3D Shape Completion on Unseen Categories

        **Authors**: *Yuchen Rao, Yinyu Nie, Angela Dai*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/de7dc701a2882088f3136139949e1d05-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/de7dc701a2882088f3136139949e1d05-Abstract-Conference.html)

        **Abstract**:

        While 3D shape representations enable powerful reasoning in many visual and perception applications, learning  3D shape priors tends to be constrained to the specific categories trained on, leading to an inefficient learning process, particularly for general applications with unseen categories. Thus, we propose PatchComplete, which learns effective shape priors based on multi-resolution local patches, which are often more general than full shapes (e.g., chairs and tables often both share legs) and thus enable geometric reasoning about unseen class categories. To learn these shared substructures, we learn multi-resolution patch priors across all train categories, which are then associated to input partial shape observations by attention across the patch priors, and finally decoded into a complete shape reconstruction. Such patch-based priors avoid overfitting to specific train categories and enable reconstruction on entirely unseen categories at test time. We demonstrate the effectiveness of our approach on synthetic ShapeNet data as well as challenging real-scanned objects from ScanNet, which include noise and clutter, improving over state of the art in novel-category shape completion by 19.3% in chamfer distance on ShapeNet, and 9.0% for ScanNet.

        ----

        ## [2495] Q-ViT: Accurate and Fully Quantized Low-bit Vision Transformer

        **Authors**: *Yanjing Li, Sheng Xu, Baochang Zhang, Xianbin Cao, Peng Gao, Guodong Guo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/deb921bff461a7b0a5c344a4871e7101-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/deb921bff461a7b0a5c344a4871e7101-Abstract-Conference.html)

        **Abstract**:

        The large pre-trained vision transformers (ViTs) have demonstrated remarkable performance on various visual tasks, but suffer from expensive computational and memory cost problems when deployed on resource-constrained devices. Among the powerful compression approaches, quantization extremely reduces the computation and memory consumption by low-bit parameters and bit-wise operations. However, low-bit ViTs remain largely unexplored and usually suffer from a significant performance drop compared with the real-valued counterparts. In this work, through extensive empirical analysis, we first identify the bottleneck  for  severe performance drop comes from  the information distortion of the low-bit quantized self-attention map. We then develop an information rectification module (IRM) and a distribution guided distillation (DGD) scheme for fully quantized vision transformers (Q-ViT) to effectively eliminate such distortion, leading to a fully quantized ViTs. We evaluate our methods on popular DeiT and Swin backbones. Extensive experimental results show that our method achieves a much better performance than the prior arts. For example, our Q-ViT can theoretically accelerates the ViT-S by 6.14x and achieves about 80.9% Top-1 accuracy, even surpassing the full-precision counterpart by 1.0% on ImageNet dataset. Our codes and models are attached on https://github.com/YanjingLi0202/Q-ViT

        ----

        ## [2496] Enhancing Safe Exploration Using Safety State Augmentation

        **Authors**: *Aivar Sootla, Alexander I. Cowen-Rivers, Jun Wang, Haitham Bou-Ammar*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/debd0ae2083160397a22a4a8831c7230-Abstract-Conference.html)

        **Abstract**:

        Safe exploration is a challenging and important problem in model-free reinforcement learning (RL). Often the safety cost is sparse and unknown, which unavoidably leads to constraint violations - a phenomenon ideally to be avoided in safety-critical applications. We tackle this problem by augmenting the state-space with a safety state, which is nonnegative if and only if the constraint is satisfied. The value of this state also serves as a distance toward constraint violation, while its initial value indicates the available safety budget. This idea allows us to derive policies for scheduling the safety budget during training. We call our approach Simmer (Safe policy IMproveMEnt for RL) to reflect the careful nature of these schedules. We apply this idea to two safe RL problems: RL with constraints imposed on an average cost, and RL with constraints imposed on a cost with probability one. Our experiments suggest that "simmering" a safe algorithm can improve safety during training for both settings. We further show that Simmer can stabilize training and improve the performance of safe RL with average constraints.

        ----

        ## [2497] Unsupervised Reinforcement Learning with Contrastive Intrinsic Control

        **Authors**: *Michael Laskin, Hao Liu, Xue Bin Peng, Denis Yarats, Aravind Rajeswaran, Pieter Abbeel*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/debf482a7dbdc401f9052dbe15702837-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/debf482a7dbdc401f9052dbe15702837-Abstract-Conference.html)

        **Abstract**:

        We introduce Contrastive Intrinsic Control (CIC), an unsupervised reinforcement learning (RL) algorithm that maximizes the mutual information between state-transitions and latent skill vectors. CIC utilizes contrastive learning between state-transitions and skills vectors to learn behaviour embeddings and maximizes the entropy of these embeddings as an intrinsic reward to encourage behavioural diversity. We evaluate our algorithm on the Unsupervised RL Benchmark (URLB) in the asymptotic state-based setting, which consists of a long reward-free pre-training phase followed by a short adaptation phase to downstream tasks with extrinsic rewards. We find that CIC improves over prior exploration algorithms in terms of adaptation efficiency to downstream tasks on state-based URLB.

        ----

        ## [2498] Distributed Online Convex Optimization with Compressed Communication

        **Authors**: *Zhipeng Tu, Xi Wang, Yiguang Hong, Lei Wang, Deming Yuan, Guodong Shi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dececdcbf0ea0162234a8fb4ab051415-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dececdcbf0ea0162234a8fb4ab051415-Abstract-Conference.html)

        **Abstract**:

        We consider a distributed online convex optimization problem when streaming data are distributed among computing agents over a connected communication network. Since the data are high-dimensional or the network is large-scale, communication load can be a bottleneck for the efficiency of distributed algorithms. To tackle this bottleneck, we apply the state-of-art data compression scheme to the fundamental GD-based distributed online algorithms. Three algorithms with difference-compressed communication are proposed for full information feedback (DC-DOGD), one-point bandit feedback (DC-DOBD), and two-point bandit feedback (DC-DO2BD), respectively. We obtain regret bounds explicitly in terms of time horizon, compression ratio, decision dimension, agent number, and network parameters. Our algorithms are proved to be no-regret and match the same regret bounds, w.r.t. time horizon, with their uncompressed versions for both convex and strongly convex losses. Numerical experiments are given to validate the theoretical findings and illustrate that the proposed algorithms can effectively reduce the total transmitted bits for distributed online training compared with the uncompressed baseline.

        ----

        ## [2499] Local Latent Space Bayesian Optimization over Structured Inputs

        **Authors**: *Natalie Maus, Haydn Jones, Juston Moore, Matt J. Kusner, John Bradshaw, Jacob R. Gardner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/ded98d28f82342a39f371c013dfb3058-Abstract-Conference.html)

        **Abstract**:

        Bayesian optimization over the latent spaces of deep autoencoder models (DAEs) has recently emerged as a promising new approach for optimizing challenging black-box functions over structured, discrete, hard-to-enumerate search spaces (e.g., molecules). Here the DAE dramatically simplifies the search space by mapping inputs into a continuous latent space where familiar Bayesian optimization tools can be more readily applied. Despite this simplification, the latent space typically remains high-dimensional. Thus, even with a well-suited latent space, these approaches do not necessarily provide a complete solution, but may rather shift the structured optimization problem to a high-dimensional one. In this paper, we propose LOL-BO, which adapts the notion of trust regions explored in recent work on high-dimensional Bayesian optimization to the structured setting. By reformulating the encoder to function as both an encoder for the DAE globally and as a deep kernel for the surrogate model within a trust region, we better align the notion of local optimization in the latent space with local optimization in the input space. LOL-BO achieves as much as 20 times improvement over state-of-the-art latent space Bayesian optimization methods across six real-world benchmarks, demonstrating that improvement in optimization strategies is as important as developing better DAE models.

        ----

        ## [2500] Graph Neural Network Bandits

        **Authors**: *Parnian Kassraie, Andreas Krause, Ilija Bogunovic*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dee8f820d86aca28ab0328a9243020f9-Abstract-Conference.html)

        **Abstract**:

        We consider the bandit optimization problem with the reward function defined over graph-structured data. This problem has important applications in molecule design and drug discovery, where the reward is naturally invariant to graph permutations. The key challenges in this setting are scaling to large domains, and to graphs with many nodes. We resolve these challenges by embedding the permutation invariance into our model. In particular, we show that graph neural networks (GNNs) can be used to estimate the reward function, assuming it resides in the Reproducing Kernel Hilbert Space of a permutation-invariant additive kernel. By establishing a novel connection between such kernels and the graph neural tangent kernel (GNTK), we introduce the first GNN confidence bound and use it to design a phased-elimination algorithm with sublinear regret. Our regret bound depends on the GNTK's maximum information gain, which we also provide a bound for. Perhaps surprisingly, even though the reward function depends on all $N$ node features, our guarantees are independent of the number of graph nodes $N$. Empirically, our approach exhibits competitive performance and scales well on graph-structured domains.

        ----

        ## [2501] Concrete Score Matching: Generalized Score Matching for Discrete Data

        **Authors**: *Chenlin Meng, Kristy Choi, Jiaming Song, Stefano Ermon*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df04a35d907e894d59d4eab1f92bc87b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df04a35d907e894d59d4eab1f92bc87b-Abstract-Conference.html)

        **Abstract**:

        Representing probability distributions by the gradient of their density functions has proven effective in modeling a wide range of continuous data modalities. However, this representation is not applicable in discrete domains where the gradient is undefined.   To this end, we propose an analogous score function called the “Concrete score”, a generalization of the (Stein) score for discrete settings. Given a predefined neighborhood structure, the Concrete score of any input is defined by the rate of change of the probabilities with respect to local directional changes of the input. This formulation allows us to recover the (Stein) score in continuous domains when measuring such changes by the Euclidean distance, while using the Manhattan distance leads to our novel score function in discrete domains. Finally, we introduce a new framework to learn such scores from samples called Concrete Score Matching (CSM), and propose an efficient training objective to scale our approach to high dimensions. Empirically, we demonstrate the efficacy of CSM on density estimation tasks on a mixture of synthetic, tabular, and high-dimensional image datasets, and demonstrate that it performs favorably relative to existing baselines for modeling discrete data.

        ----

        ## [2502] Regularized Gradient Descent Ascent for Two-Player Zero-Sum Markov Games

        **Authors**: *Sihan Zeng, Thinh T. Doan, Justin Romberg*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df2a0ada77d0d126841ba2a2f67f875e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df2a0ada77d0d126841ba2a2f67f875e-Abstract-Conference.html)

        **Abstract**:

        We study the problem of finding the Nash equilibrium in a two-player zero-sum Markov game. Due to its formulation as a minimax optimization program, a natural approach to solve the problem is to perform gradient descent/ascent with respect to each player in an alternating fashion. However, due to the non-convexity/non-concavity of the underlying objective function, theoretical understandings of this method are limited. In our paper, we consider solving an entropy-regularized variant of the Markov game. The regularization introduces structures into the optimization landscape that make the solutions more identifiable and allow the problem to be solved more efficiently. Our main contribution is to show that under proper choices of the regularization parameter, the gradient descent ascent algorithm converges to the Nash equilibrium of the original unregularized problem. We explicitly characterize the finite-time performance of the last iterate of our algorithm, which vastly improves over the existing convergence bound of the gradient descent ascent algorithm without regularization. Finally, we complement the analysis with numerical simulations that illustrate the accelerated convergence of the algorithm.

        ----

        ## [2503] Chefs' Random Tables: Non-Trigonometric Random Features

        **Authors**: *Valerii Likhosherstov, Krzysztof Marcin Choromanski, Kumar Avinava Dubey, Frederick Liu, Tamás Sarlós, Adrian Weller*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df2d62b96a4003203450cf89cd338bb7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df2d62b96a4003203450cf89cd338bb7-Abstract-Conference.html)

        **Abstract**:

        We introduce chefs' random tables (CRTs), a new class of non-trigonometric random features (RFs) to approximate Gaussian and softmax kernels. CRTs are an alternative to standard random kitchen sink (RKS) methods, which inherently rely on the trigonometric maps. We present variants of CRTs where RFs are positive, a key requirement for applications in recent low-rank Transformers. Further variance reduction is possible by leveraging statistics which are simple to compute. One instantiation of CRTs, the optimal positive random features (OPRFs), is to our knowledge the first RF method for unbiased softmax kernel estimation with positive and bounded RFs, resulting in exponentially small tails and much lower variance than its counterparts. As we show, orthogonal random features applied in OPRFs provide additional variance reduction for any dimensionality $d$ (not only asymptotically for sufficiently large $d$, as for RKS). We test CRTs on many tasks ranging from non-parametric classification to training Transformers for text, speech and image data, obtaining new state-of-the-art results for low-rank text Transformers, while providing linear space and time complexity.

        ----

        ## [2504] CS-Shapley: Class-wise Shapley Values for Data Valuation in Classification

        **Authors**: *Stephanie Schoch, Haifeng Xu, Yangfeng Ji*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df334022279996b07e0870a629c18857-Abstract-Conference.html)

        **Abstract**:

        Data valuation, or the valuation of individual datum contributions, has seen growing interest in machine learning due to its demonstrable efficacy for tasks such as noisy label detection. In particular, due to the desirable axiomatic properties, several Shapley value approximations have been proposed. In these methods, the value function is usually defined as the predictive accuracy over the entire development set. However, this limits the ability to differentiate between training instances that are helpful or harmful to their own classes. Intuitively, instances that harm their own classes may be noisy or mislabeled and should receive a lower valuation than helpful instances. In this work, we propose CS-Shapley, a Shapley value with a new value function that discriminates between training instances’ in-class and out-of-class contributions. Our theoretical analysis shows the proposed value function is (essentially) the unique function that satisfies two desirable properties for evaluating data values in classification. Further, our experiments on two benchmark evaluation tasks (data removal and noisy label detection) and four classifiers demonstrate the effectiveness of CS-Shapley over existing methods. Lastly, we evaluate the “transferability” of data values estimated from one classifier to others, and our results suggest Shapley-based data valuation is transferable for application across different models.

        ----

        ## [2505] Factuality Enhanced Language Models for Open-Ended Text Generation

        **Authors**: *Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, Bryan Catanzaro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df438caa36714f69277daa92d608dd63-Abstract-Conference.html)

        **Abstract**:

        Pretrained language models (LMs) are susceptible to generate text with nonfactual information.  In this work, we measure and improve the factual accuracy of large-scale LMs for open-ended text generation.  We design the FactualityPrompts test set and metrics to measure the factuality of LM generations.  Based on that, we study the factual accuracy of LMs with parameter sizes ranging from 126M to 530B.   Interestingly, we find that larger LMs are more factual than smaller ones, although a previous study suggests that larger LMs can be less truthful in terms of misconceptions.  In addition, popular sampling algorithms (e.g., top-p) in open-ended text generation can harm the factuality due to the ``uniform randomness'' introduced at every sampling step.  We propose the factual-nucleus sampling algorithm that dynamically adapts the randomness to improve the factuality of generation while maintaining quality.  Furthermore, we analyze the inefficiencies of the standard training method in learning correct associations between entities from factual text corpus (e.g., Wikipedia).   We propose a factuality-enhanced training method that uses TopicPrefix for better awareness of facts and sentence completion as the training objective, which can vastly reduce the factual errors.

        ----

        ## [2506] On the Representation Collapse of Sparse Mixture of Experts

        **Authors**: *Zewen Chi, Li Dong, Shaohan Huang, Damai Dai, Shuming Ma, Barun Patra, Saksham Singhal, Payal Bajaj, Xia Song, Xian-Ling Mao, Heyan Huang, Furu Wei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df4f371f1f89ec8ba5014b3310578048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df4f371f1f89ec8ba5014b3310578048-Abstract-Conference.html)

        **Abstract**:

        Sparse mixture of experts provides larger model capacity while requiring a constant computational overhead. It employs the routing mechanism to distribute input tokens to the best-matched experts according to their hidden representations. However, learning such a routing mechanism encourages token clustering around expert centroids, implying a trend toward representation collapse. In this work, we propose to estimate the routing scores between tokens and experts on a low-dimensional hypersphere. We conduct extensive experiments on cross-lingual language model pre-training and fine-tuning on downstream tasks. Experimental results across seven multilingual benchmarks show that our method achieves consistent gains. We also present a comprehensive analysis on the representation and routing behaviors of our models. Our method alleviates the representation collapse issue and achieves more consistent routing than the baseline mixture-of-experts methods.

        ----

        ## [2507] Nearly Optimal Algorithms for Linear Contextual Bandits with Adversarial Corruptions

        **Authors**: *Jiafan He, Dongruo Zhou, Tong Zhang, Quanquan Gu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/df5f94d6ac6e13d830d70536cde9f0d2-Abstract-Conference.html)

        **Abstract**:

        We study the linear contextual bandit problem in the presence of adversarial corruption, where the reward at each round is corrupted by an adversary, and the corruption level (i.e., the sum of corruption magnitudes over the horizon) is $C\geq 0$. The best-known algorithms in this setting are limited in that they either are computationally inefficient or require a strong assumption on the corruption, or their regret is at least $C$ times worse than the regret without corruption. In this paper, to overcome these limitations, we propose a new algorithm based on the principle of optimism in the face of uncertainty. At the core of our algorithm is a weighted ridge regression where the weight of each chosen action depends on its confidence up to some threshold. We show that for both known $C$ and unknown $C$ cases, our algorithm with proper choice of hyperparameter achieves a regret that nearly matches the lower bounds. Thus, our algorithm is nearly optimal up to logarithmic factors for both cases. Notably, our algorithm achieves the near-optimal regret for both corrupted and uncorrupted cases ($C=0$) simultaneously.

        ----

        ## [2508] Implicit Bias of Gradient Descent on Reparametrized Models: On Equivalence to Mirror Descent

        **Authors**: *Zhiyuan Li, Tianhao Wang, Jason D. Lee, Sanjeev Arora*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dfa1106ea7065899b13f2be9da04efb4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dfa1106ea7065899b13f2be9da04efb4-Abstract-Conference.html)

        **Abstract**:

        As part of the effort to understand implicit bias of gradient descent in overparametrized models, several results have shown how the training trajectory on the overparametrized model can be understood as mirror descent on a different objective. The main result here is a complete characterization of this phenomenon under a notion termed commuting parametrization, which encompasses all the previous results in this setting. It is shown that gradient flow with any commuting parametrization is equivalent to continuous mirror descent with a related mirror map. Conversely,  continuous mirror descent with any mirror map can be viewed as gradient flow with a related commuting parametrization. The latter result relies upon Nash's embedding theorem.

        ----

        ## [2509] HYPRO: A Hybridly Normalized Probabilistic Model for Long-Horizon Prediction of Event Sequences

        **Authors**: *Siqiao Xue, Xiaoming Shi, James Y. Zhang, Hongyuan Mei*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dfbb3d1807b21dadee735eb75069ada4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dfbb3d1807b21dadee735eb75069ada4-Abstract-Conference.html)

        **Abstract**:

        In this paper, we tackle the important yet under-investigated problem of making long-horizon prediction of event sequences. Existing state-of-the-art models do not perform well at this task due to their autoregressive structure. We propose HYPRO, a hybridly normalized probabilistic model that naturally fits this task: its first part is an autoregressive base model that learns to propose predictions; its second part is an energy function that learns to reweight the proposals such that more realistic predictions end up with higher probabilities. We also propose efficient training and inference algorithms for this model. Experiments on multiple real-world datasets demonstrate that our proposed HYPRO model can significantly outperform previous models at making long-horizon predictions of future events. We also conduct a range of ablation studies to investigate the effectiveness of each component of our proposed methods.

        ----

        ## [2510] Towards Understanding Grokking: An Effective Theory of Representation Learning

        **Authors**: *Ziming Liu, Ouail Kitouni, Niklas Nolte, Eric J. Michaud, Max Tegmark, Mike Williams*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dfc310e81992d2e4cedc09ac47eff13e-Abstract-Conference.html)

        **Abstract**:

        We aim to understand grokking, a phenomenon where models generalize long after overfitting their training set. We present both a microscopic analysis anchored by an effective theory and a macroscopic analysis of phase diagrams describing learning performance across hyperparameters. We find that generalization originates from structured representations, whose training dynamics and dependence on training set size can be predicted by our effective theory (in a toy setting). We observe empirically the presence of four learning phases: comprehension, grokking, memorization, and confusion. We find representation learning to occur only in a "Goldilocks zone" (including comprehension and grokking) between memorization and confusion. Compared to the comprehension phase, the grokking phase stays closer to the memorization phase, leading to delayed generalization. The Goldilocks phase is reminiscent of "intelligence from starvation" in Darwinian evolution, where resource limitations drive discovery of more efficient solutions. This study not only provides intuitive explanations of the origin of grokking, but also highlights the usefulness of physics-inspired tools, e.g., effective theories and phase diagrams, for understanding deep learning.

        ----

        ## [2511] Asymptotic Behaviors of Projected Stochastic Approximation: A Jump Diffusion Perspective

        **Authors**: *Jiadong Liang, Yuze Han, Xiang Li, Zhihua Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dfdc9c54cd62f2b2bfd8b090b3489b7f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dfdc9c54cd62f2b2bfd8b090b3489b7f-Abstract-Conference.html)

        **Abstract**:

        In this paper, we consider linearly constrained stochastic approximation problems with federated learning (FL) as a special case. We propose a stochastic approximation algorithm named by LPSA with probabilistic projections to ensure feasibility so that projections are performed with probability $p_n$ at the $n$-th iteration. Considering a specific family of the probability $p_n$ and step size $\eta_n$, we analyze our algorithm from an asymptotic and continuous perspective. Using a novel jump diffusion approximation, we show that the trajectories consisting of properly rescaled last iterates weakly converge to the solution of specific SDEs. By analyzing the SDEs, we identify the asymptotic behaviors of LPSA for different choices of $(p_n, \eta_n)$. We find the algorithm presents an intriguing asymptotic bias-variance trade-off according to the relative magnitude of $p_n$ w.r.t. $\eta_n$. It provides insights on how to choose appropriate $\{(p_n, \eta_n)\}_{n \geq 1}$ to minimize the projection complexity.

        ----

        ## [2512] Towards Practical Few-shot Query Sets: Transductive Minimum Description Length Inference

        **Authors**: *Ségolène Martin, Malik Boudiaf, Émilie Chouzenoux, Jean-Christophe Pesquet, Ismail Ben Ayed*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dff528ce3e1390c88f10bbf5e722a241-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dff528ce3e1390c88f10bbf5e722a241-Abstract-Conference.html)

        **Abstract**:

        Standard few-shot benchmarks are often built upon simplifying assumptions on the query sets, which may not always hold in practice. In particular, for each task at testing time, the classes effectively present in the unlabeled query set are known a priori, and correspond exactly to the set of classes represented in the labeled support set. We relax these assumptions and extend current benchmarks, so that the query-set classes of a given task are unknown, but just belong to a much larger set of possible classes. Our setting could be viewed as an instance of the challenging yet practical problem of extremely imbalanced $K$-way classification, $K$ being much larger than the values typically used in standard benchmarks, and with potentially irrelevant supervision from the support set. Expectedly, our setting incurs drops in the performances of state-of-the-art methods. Motivated by these observations, we introduce a \textbf{P}rim\textbf{A}l \textbf{D}ual Minimum \textbf{D}escription \textbf{LE}ngth (\textbf{PADDLE}) formulation, which balances data-fitting accuracy and model complexity for a given few-shot task, under supervision constraints from the support set. Our constrained MDL-like objective promotes competition among a large set of possible classes, preserving only effective classes that befit better the data of a few-shot task. It is hyper-parameter free, and could be applied on top of any base-class training. Furthermore, we derive a fast block coordinate descent algorithm for optimizing our objective, with convergence guarantee, and a linear computational complexity at each iteration. Comprehensive experiments over the standard few-shot datasets and the more realistic and challenging \textit{i-Nat} dataset show highly competitive performances of our method, more so when the numbers of possible classes in the tasks increase. Our code is publicly available at \url{https://github.com/SegoleneMartin/PADDLE}.

        ----

        ## [2513] Understanding the Generalization Benefit of Normalization Layers: Sharpness Reduction

        **Authors**: *Kaifeng Lyu, Zhiyuan Li, Sanjeev Arora*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/dffd1c523512e557f4e75e8309049213-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/dffd1c523512e557f4e75e8309049213-Abstract-Conference.html)

        **Abstract**:

        Normalization layers (e.g., Batch Normalization, Layer Normalization) were introduced to help with optimization difficulties in very deep nets, but they clearly also help generalization, even in not-so-deep nets. Motivated by the long-held belief that flatter minima lead to better generalization, this paper gives mathematical analysis and supporting experiments suggesting that normalization (together with accompanying weight-decay) encourages GD to reduce the sharpness of loss surface. Here ``sharpness'' is carefully defined given that the loss is scale-invariant, a known consequence of normalization. Specifically, for a fairly broad class of neural nets with normalization, our theory explains how GD with a finite learning rate enters the so-called Edge of Stability (EoS) regime, and characterizes the trajectory of GD in this regime via a continuous sharpness-reduction flow.

        ----

        ## [2514] Learning Manifold Dimensions with Conditional Variational Autoencoders

        **Authors**: *Yijia Zheng, Tong He, Yixuan Qiu, David P. Wipf*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e04101138a3c94544760c1dbdf2c7a2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e04101138a3c94544760c1dbdf2c7a2d-Abstract-Conference.html)

        **Abstract**:

        Although the variational autoencoder (VAE) and its conditional extension (CVAE) are capable of state-of-the-art results across multiple domains, their precise behavior is still not fully understood, particularly in the context of data (like images) that lie on or near a low-dimensional manifold. For example, while prior work has suggested that the globally optimal VAE solution can learn the correct manifold dimension, a necessary (but not sufficient) condition for producing samples from the true data distribution, this has never been rigorously proven.  Moreover, it remains unclear how such considerations would change when various types of conditioning variables are introduced, or when the data support is extended to a union of manifolds (e.g., as is likely the case for MNIST digits and related).  In this work, we address these points by first proving that VAE global minima are indeed capable of recovering the correct manifold dimension.  We then extend this result to more general CVAEs, demonstrating practical scenarios whereby the conditioning variables allow the model to adaptively learn manifolds of varying dimension across samples.  Our analyses, which have practical implications for various CVAE design choices, are also supported by numerical results on both synthetic and real-world datasets.

        ----

        ## [2515] Multi-modal Grouping Network for Weakly-Supervised Audio-Visual Video Parsing

        **Authors**: *Shentong Mo, Yapeng Tian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e095c0a3717629aa5497601985bfcf0e-Abstract-Conference.html)

        **Abstract**:

        The audio-visual video parsing task aims to parse a video into modality- and category-aware temporal segments. Previous work mainly focuses on weakly-supervised approaches, which learn from video-level event labels. During training, they do not know which modality perceives and meanwhile which temporal segment contains the video event. Since there is no explicit grouping in the existing frameworks, the modality and temporal uncertainties make these methods suffer from false predictions. For instance, segments in the same category could be predicted in different event classes. Learning compact and discriminative multi-modal subspaces is essential for mitigating the issue. To this end, in this paper, we propose a novel Multi-modal Grouping Network, namely MGN, for explicitly semantic-aware grouping. Specifically, MGN aggregates event-aware unimodal features through unimodal grouping in terms of learnable categorical embedding tokens. Furthermore, it leverages the cross-modal grouping for modality-aware prediction to match the video-level target. Our simple framework achieves improving results against previous baselines on weakly-supervised audio-visual video parsing. In addition, our MGN is much more lightweight, using only 47.2% of the parameters of baselines (17 MB vs. 36 MB). Code is available at https://github.com/stoneMo/MGN.

        ----

        ## [2516] Optimal-er Auctions through Attention

        **Authors**: *Dmitry Ivanov, Iskander Safiulin, Igor Filippov, Ksenia Balabaeva*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0c07bb70721255482020afca44cabf2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0c07bb70721255482020afca44cabf2-Abstract-Conference.html)

        **Abstract**:

        RegretNet is a recent breakthrough in the automated design of revenue-maximizing auctions. It combines the flexibility of deep learning with the regret-based approach to relax the Incentive Compatibility (IC) constraint (that participants prefer to bid truthfully) in order to approximate optimal auctions. We propose two independent improvements of RegretNet. The first is a neural architecture denoted as RegretFormer that is based on attention layers. The second is a loss function that requires explicit specification of an acceptable IC violation denoted as regret budget. We investigate both modifications in an extensive experimental study that includes settings with constant and inconstant numbers of items and participants, as well as novel validation procedures tailored to regret-based approaches. We find that RegretFormer consistently outperforms RegretNet in revenue (i.e. is optimal-er) and that our loss function both simplifies hyperparameter tuning and allows to unambiguously control the revenue-regret trade-off by selecting the regret budget.

        ----

        ## [2517] Bootstrapped Transformer for Offline Reinforcement Learning

        **Authors**: *Kerong Wang, Hanye Zhao, Xufang Luo, Kan Ren, Weinan Zhang, Dongsheng Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0ccda3cb17b084a6f43c62cfac4784b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0ccda3cb17b084a6f43c62cfac4784b-Abstract-Conference.html)

        **Abstract**:

        Offline reinforcement learning (RL) aims at learning policies from previously collected static trajectory data without interacting with the real environment. Recent works provide a novel perspective by viewing offline RL as a generic sequence generation problem, adopting sequence models such as Transformer architecture to model distributions over trajectories and repurposing beam search as a planning algorithm. However, the training datasets utilized in general offline RL tasks are quite limited and often suffering from insufficient distribution coverage, which could me harmful to training sequence generation models yet has not drawn enough attention in the previous works. In this paper, we propose a novel algorithm named Bootstrapped Transformer, which incorporates the idea of bootstrapping and leverages the learned model to self-generate more offline data to further boost the training of sequence model. We conduct extensive experiments on two offline RL benchmarks and demonstrate that our model can largely remedy the limitations of the existing offline RL training and beat other strong baseline methods. We also analyze the generated pseudo data and the revealed characteristics may shed some light on offline RL training.

        ----

        ## [2518] How to talk so AI will learn: Instructions, descriptions, and autonomy

        **Authors**: *Theodore R. Sumers, Robert D. Hawkins, Mark K. Ho, Tom Griffiths, Dylan Hadfield-Menell*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0cfde0ff720fa9674bb976e7f1b99d4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0cfde0ff720fa9674bb976e7f1b99d4-Abstract-Conference.html)

        **Abstract**:

        From the earliest years of our lives, humans use language to express our beliefs and desires. Being able to talk to artificial agents about our preferences would thus fulfill a central goal of value alignment. Yet today, we lack computational models explaining such language use. To address this challenge, we formalize learning from language in a contextual bandit setting and ask how a human might communicate preferences over behaviors. We study two distinct types of language: instructions, which provide information about the desired policy, and descriptions, which provide information about the reward function. We show that the agent's degree of autonomy determines which form of language is optimal: instructions are better in low-autonomy settings, but descriptions are better when the agent will need to act independently. We then define a pragmatic listener agent that robustly infers the speaker's reward function by reasoning about how the speaker expresses themselves. We validate our models with a behavioral experiment, demonstrating that (1) our speaker model predicts human behavior, and (2) our pragmatic listener successfully recovers humans' reward functions. Finally, we show that this form of social learning can integrate with and reduce regret in traditional reinforcement learning. We hope these insights facilitate a shift from developing agents that obey language to agents that learn from it.

        ----

        ## [2519] Learning on the Edge: Online Learning with Stochastic Feedback Graphs

        **Authors**: *Emmanuel Esposito, Federico Fusco, Dirk van der Hoeven, Nicolò Cesa-Bianchi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0e956681b04ac126679e8c7dd706b2e-Abstract-Conference.html)

        **Abstract**:

        The framework of feedback graphs is a generalization of sequential decision-making with bandit or full information feedback. In this work, we study an extension where the directed feedback graph is stochastic, following a distribution similar to the classical Erdős-Rényi model. Specifically, in each round every edge in the graph is either realized or not with a distinct probability for each edge. We prove nearly optimal regret bounds of order $\min\bigl\{\min_{\varepsilon} \sqrt{(\alpha_\varepsilon/\varepsilon) T},\, \min_{\varepsilon} (\delta_\varepsilon/\varepsilon)^{1/3} T^{2/3}\bigr\}$ (ignoring logarithmic factors), where $\alpha_{\varepsilon}$ and $\delta_{\varepsilon}$ are graph-theoretic quantities measured on the support of the stochastic feedback graph $\mathcal{G}$ with edge probabilities thresholded at $\varepsilon$. Our result, which holds without any preliminary knowledge about $\mathcal{G}$, requires the learner to observe only the realized out-neighborhood of the chosen action. When the learner is allowed to observe the realization of the entire graph (but only the losses in the out-neighborhood of the chosen action), we derive a more efficient algorithm featuring a dependence on weighted versions of the independence and weak domination numbers that exhibits improved bounds for some special cases.

        ----

        ## [2520] Globally Gated Deep Linear Networks

        **Authors**: *Qianyi Li, Haim Sompolinsky*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0f393e7980a24fd12fa6f15adfa25fb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0f393e7980a24fd12fa6f15adfa25fb-Abstract-Conference.html)

        **Abstract**:

        Recently proposed Gated Linear Networks (GLNs) present a tractable nonlinear network architecture, and exhibit interesting capabilities such as learning with local error signals and reduced forgetting in sequential learning. In this work, we introduce a novel gating architecture, named Globally Gated Deep Linear Networks (GGDLNs) where gating units are shared among all processing units in each layer, thereby decoupling the architectures of the nonlinear but unlearned gating and the learned linear processing motifs. We derive exact equations for the generalization properties of Bayesian Learning in these networks in the finite-width thermodynamic limit, defined by $N, P\rightarrow\infty$ while $P/N=O(1)$ where $N$ and $P$ are the hidden layers' width and size of training data sets respectfully. We find that the statistics of the network predictor can be expressed in terms of kernels that undergo shape renormalization through a data-dependent order-parameter matrix compared to the infinite-width Gaussian Process (GP) kernels. Our theory accurately captures the behavior of finite width GGDLNs trained with gradient descent (GD) dynamics. We show that kernel shape renormalization gives rise to rich generalization properties w.r.t. network width, depth, and $L_2$ regularization amplitude. Interestingly, networks with a large number of gating units behave similarly to standard ReLU architectures. Although gating units in the model do not participate in supervised learning, we show the utility of unsupervised learning of the gating parameters. Additionally, our theory allows the evaluation of the network capacity for learning multiple tasks by incorporating task-relevant information into the gating units. In summary, our work is the first exact theoretical solution of learning in a family of nonlinear networks with finite width. The rich and diverse behavior of the GGDLNs suggests that they are helpful analytically tractable models of learning single and multiple tasks, in finite-width nonlinear deep networks.

        ----

        ## [2521] Markov Chain Score Ascent: A Unifying Framework of Variational Inference with Markovian Gradients

        **Authors**: *Kyurae Kim, Jisu Oh, Jacob R. Gardner, Adji Bousso Dieng, Hongseok Kim*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e0fbc0f2e35e58aeffe5524a69ba90e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e0fbc0f2e35e58aeffe5524a69ba90e5-Abstract-Conference.html)

        **Abstract**:

        Minimizing the inclusive Kullback-Leibler (KL) divergence with stochastic gradient descent (SGD) is challenging since its gradient is defined as an integral over the posterior. Recently, multiple methods have been proposed to run SGD with biased gradient estimates obtained from a Markov chain. This paper provides the first non-asymptotic convergence analysis of these methods by establishing their mixing rate and gradient variance. To do this, we demonstrate that these methods—which we collectively refer to as Markov chain score ascent (MCSA) methods—can be cast as special cases of the Markov chain gradient descent framework. Furthermore, by leveraging this new understanding, we develop a novel MCSA scheme, parallel MCSA (pMCSA), that achieves a tighter bound on the gradient variance. We demonstrate that this improved theoretical result translates to superior empirical performance.

        ----

        ## [2522] Cache-Augmented Inbatch Importance Resampling for Training Recommender Retriever

        **Authors**: *Jin Chen, Defu Lian, Yucheng Li, Baoyun Wang, Kai Zheng, Enhong Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1194b07221de8783a02087bb899d69f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1194b07221de8783a02087bb899d69f-Abstract-Conference.html)

        **Abstract**:

        Recommender retrievers aim to rapidly retrieve a fraction of items from the entire item corpus when a user query requests, with the representative two-tower model trained with the log softmax loss. For efficiently training recommender retrievers on modern hardwares, inbatch sampling, where the items in the mini-batch are shared as negatives to estimate the softmax function, has attained growing interest. However, existing inbatch sampling based strategies just correct the sampling bias of inbatch items with item frequency, being unable to distinguish the user queries within the mini-batch and still incurring significant bias from the softmax. In this paper, we propose a Cache-Augmented Inbatch Importance Resampling (XIR) for training recommender retrievers, which not only offers different negatives to user queries with inbatch items, but also adaptively achieves a more accurate estimation of the softmax distribution. Specifically, XIR resamples items from the given mini-batch training pairs based on certain probabilities, where a cache with more frequently sampled items is adopted to augment the candidate item set, with the purpose of reusing the historical informative samples. XIR enables to sample query-dependent negatives based on inbatch items and to capture dynamic changes of model training, which leads to a better approximation of the softmax and further contributes to better convergence. Finally, we conduct experiments to validate the superior performance of the proposed XIR compared with competitive approaches.

        ----

        ## [2523] Doubly Robust Counterfactual Classification

        **Authors**: *Kwangho Kim, Edward H. Kennedy, José R. Zubizarreta*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e124f1547f7ac87e33d348b827d4291b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e124f1547f7ac87e33d348b827d4291b-Abstract-Conference.html)

        **Abstract**:

        We study counterfactual classification as a new tool for decision-making under hypothetical (contrary to fact) scenarios. We propose a doubly-robust nonparametric estimator for a general counterfactual classifier, where we can incorporate flexible constraints by casting the classification problem as a nonlinear mathematical program involving counterfactuals. We go on to analyze the rates of convergence of the estimator and provide a closed-form expression for its asymptotic distribution. Our analysis shows that the proposed estimator is robust against nuisance model misspecification, and can attain fast $\sqrt{n}$ rates with tractable inference even when using nonparametric machine learning approaches. We study the empirical performance of our methods by simulation and apply them for recidivism risk prediction.

        ----

        ## [2524] Rethinking Value Function Learning for Generalization in Reinforcement Learning

        **Authors**: *Seungyong Moon, JunYeong Lee, Hyun Oh Song*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e19ab2dde2e60cf68d1ded18c38938f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e19ab2dde2e60cf68d1ded18c38938f4-Abstract-Conference.html)

        **Abstract**:

        Our work focuses on training RL agents on multiple visually diverse environments to improve observational generalization performance. In prior methods, policy and value networks are separately optimized using a disjoint network architecture to avoid interference and obtain a more accurate value function. We identify that a value network in the multi-environment setting is more challenging to optimize and prone to memorizing the training data than in the conventional single-environment setting. In addition, we find that appropriate regularization on the value network is necessary to improve both training and test performance. To this end, we propose Delayed-Critic Policy Gradient (DCPG), a policy gradient algorithm that implicitly penalizes value estimates by optimizing the value network less frequently with more training data than the policy network. This can be implemented using a single unified network architecture. Furthermore, we introduce a simple self-supervised task that learns the forward and inverse dynamics of environments using a single discriminator, which can be jointly optimized with the value network. Our proposed algorithms significantly improve observational generalization performance and sample efficiency on the Procgen Benchmark.

        ----

        ## [2525] Improving Certified Robustness via Statistical Learning with Logical Reasoning

        **Authors**: *Zhuolin Yang, Zhikuan Zhao, Boxin Wang, Jiawei Zhang, Linyi Li, Hengzhi Pei, Bojan Karlas, Ji Liu, Heng Guo, Ce Zhang, Bo Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1b248453bca182b6138b8c14a75340d-Abstract-Conference.html)

        **Abstract**:

        Intensive algorithmic efforts have been made to enable the rapid improvements of certificated robustness for complex ML models recently. However, current robustness certification methods are only able to certify under a limited perturbation radius. Given that existing pure data-driven statistical approaches have reached a bottleneck, in this paper, we propose to integrate statistical ML models with knowledge (expressed as logical rules) as a reasoning component using Markov logic networks (MLN), so as to further improve the overall certified robustness. This opens new research questions about certifying the robustness of such a paradigm, especially the reasoning component (e.g., MLN). As the first step towards understanding these questions, we first prove that the computational complexity of certifying the robustness of MLN is #P-hard. Guided by this hardness result, we then derive the first certified robustness bound for MLN by carefully analyzing different model regimes. Finally, we conduct extensive experiments on five datasets including both high-dimensional images and natural language texts, and we show that the certified robustness with knowledge-based logical reasoning indeed significantly outperforms that of the state-of-the-arts.

        ----

        ## [2526] Transformer-based Working Memory for Multiagent Reinforcement Learning with Action Parsing

        **Authors**: *Yaodong Yang, Guangyong Chen, Weixun Wang, Xiaotian Hao, Jianye Hao, Pheng-Ann Heng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1cf57f1e104c6c05e31894c15a65e99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1cf57f1e104c6c05e31894c15a65e99-Abstract-Conference.html)

        **Abstract**:

        Learning in real-world multiagent tasks is challenging due to the usual partial observability of each agent. Previous efforts alleviate the partial observability by historical hidden states with Recurrent Neural Networks, however, they do not consider the multiagent characters that either the multiagent observation consists of a number of object entities or the action space shows clear entity interactions. To tackle these issues, we propose the Agent Transformer Memory (ATM) network with a transformer-based memory. First, ATM utilizes the transformer to enable the unified processing of the factored environmental entities and memory. Inspired by the human’s working memory process where a limited capacity of information temporarily held in mind can effectively guide the decision-making, ATM updates its fixed-capacity memory with the working memory updating schema. Second, as agents' each action has its particular interaction entities in the environment, ATM parses the action space to introduce this action’s semantic inductive bias by binding each action with its specified involving entity to predict the state-action value or logit. Extensive experiments on the challenging SMAC and Level-Based Foraging environments validate that ATM could boost existing multiagent RL algorithms with impressive learning acceleration and performance improvement.

        ----

        ## [2527] Recruitment Strategies That Take a Chance

        **Authors**: *Gregory Kehne, Ariel D. Procaccia, Jingyan Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1f351706ab4417524e1b17f9adcc657-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1f351706ab4417524e1b17f9adcc657-Abstract-Conference.html)

        **Abstract**:

        In academic recruitment settings, including faculty hiring and PhD admissions, committees aim to maximize the overall quality of recruited candidates, but there is uncertainty about whether a candidate would accept an offer if given one. Previous work has considered algorithms that make offers sequentially and are subject to a hard budget constraint. We argue that these modeling choices may be inconsistent with the practice of academic recruitment. Instead, we restrict ourselves to a single batch of offers, and we treat the target number of positions as a soft constraint, so we risk overshooting or undershooting the target. Specifically, our objective is to select a subset of candidates that maximizes the overall expected value associated with candidates who accept, minus an expected penalty for deviating from the target. We first analyze the guarantees provided by natural greedy heuristics, showing their desirable properties despite the simplicity. Depending on the structure of the penalty function, we further develop algorithms that provide fully polynomial-time approximation schemes and constant-factor approximations to this objective. Empirical evaluation of our algorithms corroborates these theoretical results.

        ----

        ## [2528] Fully Convolutional One-Stage 3D Object Detection on LiDAR Range Images

        **Authors**: *Zhi Tian, Xiangxiang Chu, Xiaoming Wang, Xiaolin Wei, Chunhua Shen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1f418450107c4a0ddc16d008d131573-Abstract-Conference.html)

        **Abstract**:

        We present a simple yet effective fully convolutional one-stage 3D object detector for LiDAR point clouds of autonomous driving scenes, termed FCOS-LiDAR. Unlike the dominant methods that use the bird-eye view (BEV), our proposed detector detects objects from the range view (RV, a.k.a. range image) of the LiDAR points. Due to the range view's compactness and compatibility with the LiDAR sensors' sampling process on self-driving cars, the range view-based object detector can be realized by solely exploiting the vanilla 2D convolutions, departing from the BEV-based methods which often involve complicated voxelization operations and sparse convolutions.  For the first time, we show that an RV-based 3D detector with standard 2D convolutions alone can achieve comparable performance to state-of-the-art BEV-based detectors while being significantly faster and simpler. More importantly, almost all previous range view-based detectors only focus on single-frame point clouds since it is challenging to fuse multi-frame point clouds into a single range view. In this work, we tackle this challenging issue with a novel range view projection mechanism, and for the first time demonstrate the benefits of fusing multi-frame point clouds for a range-view based detector. Extensive experiments on nuScenes show the superiority of our proposed method and we believe that our work can be strong evidence that an RV-based 3D detector can compare favourably with the current mainstream BEV-based detectors. Code will be made publicly available.

        ----

        ## [2529] Understanding Robust Learning through the Lens of Representation Similarities

        **Authors**: *Christian Cianfarani, Arjun Nitin Bhagoji, Vikash Sehwag, Ben Y. Zhao, Heather Zheng, Prateek Mittal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e1fa017a312368906411501bbd27a1d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e1fa017a312368906411501bbd27a1d6-Abstract-Conference.html)

        **Abstract**:

        Representation learning, \textit{i.e.} the generation of representations useful for downstream applications, is a task of fundamental importance that underlies much of the success of deep neural networks (DNNs). Recently, \emph{robustness to adversarial examples} has emerged as a desirable property for DNNs, spurring the development of robust training methods that account for adversarialexamples. In this paper, we aim to understand how the properties of representations learned by robust training differ from those obtained from standard, non-robust training. This is critical to diagnosing numerous salient pitfalls in robust networks, such as, degradation of performance on benign inputs, poor generalization of robustness, and increase in over-fitting. We utilize a powerful set of tools known as representation similarity metrics, across 3 vision datasets, to obtain layer-wise comparisons between robust and non-robust DNNs with different architectures, training procedures and adversarial constraints. Our experiments highlight hitherto unseen properties of robust representations that we posit underlie the behavioral differences of robust networks. We discover a lack of specialization in robust networks' representations along with a disappearance of `block structure'. We also find overfitting during robust training largely impacts deeper layers. These, along with other findings, suggest ways forward for the design and training of better robust networks.

        ----

        ## [2530] Chartalist: Labeled Graph Datasets for UTXO and Account-based Blockchains

        **Authors**: *Kiarash Shamsi, Friedhelm Victor, Murat Kantarcioglu, Yulia R. Gel, Cuneyt Gurcan Akcora*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e245189a86310b6667ac633dbb922d50-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e245189a86310b6667ac633dbb922d50-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Machine learning on blockchain graphs is an emerging field with many applications such as ransomware payment tracking, price manipulation analysis, and money laundering detection. However, analyzing blockchain data requires domain expertise and computational resources, which pose a significant barrier and hinder advancement in this field. We introduce Chartalist, the first comprehensive platform to methodically access and use machine learning across a large selection of blockchains to address this challenge. Chartalist contains ML-ready datasets from unspent transaction output (UTXO) (e.g., Bitcoin) and account-based blockchains (e.g., Ethereum). We envision that Chartalist can facilitate data modeling, analysis, and representation of blockchain data and attract a wider community of scientists to analyze blockchains. Chartalist is an open-science initiative at https://github.com/cakcora/Chartalist.

        ----

        ## [2531] Multi-Instance Causal Representation Learning for Instance Label Prediction and Out-of-Distribution Generalization

        **Authors**: *Weijia Zhang, Xuanhui Zhang, Hanwen Deng, Min-Ling Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e261e92e1cfb820da930ad8c38d0aead-Abstract-Conference.html)

        **Abstract**:

        Multi-instance learning (MIL) deals with objects represented as bags of instances and can predict instance labels from bag-level supervision. However, significant performance gaps exist between instance-level MIL algorithms and supervised learners since the instance labels are unavailable in MIL. Most existing MIL algorithms tackle the problem by treating multi-instance bags as harmful ambiguities and predicting instance labels by reducing the supervision inexactness. This work studies MIL from a new perspective by considering bags as auxiliary information, and utilize it to identify instance-level causal representations from bag-level weak supervision. We propose the CausalMIL algorithm, which not only excels at instance label prediction but also provides robustness to distribution change by synergistically integrating MIL with identifiable variational autoencoder. Our approach is based on a practical and general assumption: the prior distribution over the instance latent representations belongs to the non-factorized exponential family conditioning on the multi-instance bags. Experiments on synthetic and real-world datasets demonstrate that our approach significantly outperforms various baselines on instance label prediction and out-of-distribution generalization tasks.

        ----

        ## [2532] PerfectDou: Dominating DouDizhu with Perfect Information Distillation

        **Authors**: *Guan Yang, Minghuan Liu, Weijun Hong, Weinan Zhang, Fei Fang, Guangjun Zeng, Yue Lin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e26f31de8b13ec569bf507e6ae2cd952-Abstract-Conference.html)

        **Abstract**:

        As a challenging multi-player card game, DouDizhu has recently drawn much attention for analyzing competition and collaboration in imperfect-information games. In this paper, we propose PerfectDou, a state-of-the-art Doudizhu AI system that summits the game, in an actor-critic framework with a proposed technique named perfect information distillation.In detail, we adopt a perfect-training-imperfection-execution framework that allows the agents to utilize the global information to guide the training of the policies as if it is a perfect information game and the trained policies can be used to play the imperfect information game during the actual gameplay. Correspondingly, we characterize card and game features for DouDizhu to represent the perfect and imperfect information. To train our system, we adopt proximal policy optimization with generalized advantage estimation in a parallel training paradigm. In experiments we show how and why PerfectDou beats all existing programs, and achieves state-of-the-art performance.

        ----

        ## [2533] On the Double Descent of Random Features Models Trained with SGD

        **Authors**: *Fanghui Liu, Johan A. K. Suykens, Volkan Cevher*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e271e30de7a2e462ca1f85cefa816380-Abstract-Conference.html)

        **Abstract**:

        We study generalization properties of random features (RF) regression in high dimensions optimized by stochastic gradient descent (SGD) in under-/over-parameterized regime. In this work, we derive precise non-asymptotic error bounds of RF regression under both constant and polynomial-decay step-size SGD setting, and observe the double descent phenomenon both theoretically and empirically. Our analysis shows how to cope with multiple randomness sources of initialization, label noise, and data sampling (as well as stochastic gradients) with no closed-form solution, and also goes beyond the commonly-used Gaussian/spherical data assumption. Our theoretical results demonstrate that, with SGD training, RF regression still generalizes well for interpolation learning, and is able to characterize the double descent behavior by the unimodality of variance and monotonic decrease of bias. Besides, we also prove that the constant step-size SGD setting incurs no loss in convergence rate when compared to the exact minimum-norm interpolator, as a theoretical justification of using SGD in practice.

        ----

        ## [2534] SALSA: Attacking Lattice Cryptography with Transformers

        **Authors**: *Emily Wenger, Mingjie Chen, François Charton, Kristin E. Lauter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e28b3369186459f57c94a9ec9137fac9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e28b3369186459f57c94a9ec9137fac9-Abstract-Conference.html)

        **Abstract**:

        Currently deployed public-key cryptosystems will be vulnerable to attacks by full-scale quantum computers. Consequently, "quantum resistant" cryptosystems are in high demand, and lattice-based cryptosystems, based on a hard problem known as Learning With Errors (LWE), have emerged as strong contenders for standardization. In this work, we train transformers to perform modular arithmetic and mix half-trained models and statistical cryptanalysis techniques to propose SALSA: a machine learning attack on LWE-based cryptographic schemes. SALSA can fully recover secrets for small-to-mid size LWE instances with sparse binary secrets, and may scale to attack real world LWE-based cryptosystems.

        ----

        ## [2535] Variable-rate hierarchical CPC leads to acoustic unit discovery in speech

        **Authors**: *Santiago Cuervo, Adrian Lancucki, Ricard Marxer, Pawel Rychlikowski, Jan Chorowski*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e2b0a30ea6a67cba58134e57348afb91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e2b0a30ea6a67cba58134e57348afb91-Abstract-Conference.html)

        **Abstract**:

        The success of deep learning comes from its ability to capture the hierarchical structure of data by learning high-level representations defined in terms of low-level ones. In this paper we explore self-supervised learning of hierarchical representations of speech by applying multiple levels of Contrastive Predictive Coding (CPC). We observe that simply stacking two CPC models does not yield significant improvements over single-level architectures. Inspired by the fact that speech is often described as a sequence of discrete units unevenly distributed in time, we propose a model in which the output of a low-level CPC module is non-uniformly downsampled to directly minimize the loss of a high-level CPC module. The latter is designed to also enforce a prior of separability and discreteness in its representations by enforcing dissimilarity of successive high-level representations through focused negative sampling, and by quantization of the prediction targets. Accounting for the structure of the speech signal improves upon single-level CPC features and enhances the disentanglement of the learned representations, as measured by downstream speech recognition tasks, while resulting in a meaningful segmentation of the signal that closely resembles phone boundaries.

        ----

        ## [2536] Learning to Attack Federated Learning: A Model-based Reinforcement Learning Attack Framework

        **Authors**: *Henger Li, Xiaolin Sun, Zizhan Zheng*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e2ef0cae667dbe9bfdbcaed1bd91807b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e2ef0cae667dbe9bfdbcaed1bd91807b-Abstract-Conference.html)

        **Abstract**:

        We propose a model-based reinforcement learning framework to derive untargeted poisoning attacks against federated learning (FL) systems. Our framework first approximates the distribution of the clients' aggregated data using model updates from the server. The learned distribution is then used to build a simulator of the FL environment, which is utilized to learn an adaptive attack policy through reinforcement learning. Our framework is capable of learning strong attacks automatically even when the server adopts a robust aggregation rule. We further derive an upper bound on the attacker's performance loss due to inaccurate distribution estimation. Experimental results on real-world datasets demonstrate that the proposed attack framework significantly outperforms state-of-the-art poisoning attacks. This indicates the importance of developing adaptive defenses for FL systems.

        ----

        ## [2537] Learning Neural Set Functions Under the Optimal Subset Oracle

        **Authors**: *Zijing Ou, Tingyang Xu, Qinliang Su, Yingzhen Li, Peilin Zhao, Yatao Bian*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e332505c4c80ad1d9dc0af26103b672b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e332505c4c80ad1d9dc0af26103b672b-Abstract-Conference.html)

        **Abstract**:

        Learning set functions becomes increasingly important in many applications like product recommendation and compound selection in AI-aided drug discovery. The majority of existing works study methodologies of set function learning under the function value oracle, which, however, requires expensive supervision signals. This renders it impractical for applications with only weak supervisions under the Optimal Subset (OS) oracle, the study of which is surprisingly overlooked. In this work, we present a principled yet practical maximum likelihood learning framework, termed as EquiVSet,  that simultaneously meets the following desiderata of learning neural set functions under the OS oracle: i) permutation invariance of the set mass function being modeled; ii) permission of varying ground set; iii) minimum prior and iv) scalability. The main components of our framework involve: an energy-based treatment of the set mass function, DeepSet-style architectures to handle permutation invariance, mean-field variational inference, and its amortized variants. Thanks to the delicate combination of these advanced architectures, empirical studies on three real-world applications (including  Amazon product recommendation, set anomaly detection, and compound selection for virtual screening) demonstrate that EquiVSet outperforms the baselines by a large margin.

        ----

        ## [2538] A Near-Optimal Best-of-Both-Worlds Algorithm for Online Learning with Feedback Graphs

        **Authors**: *Chloé Rouyer, Dirk van der Hoeven, Nicolò Cesa-Bianchi, Yevgeny Seldin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e36da7acd188c6655792270b38830124-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e36da7acd188c6655792270b38830124-Abstract-Conference.html)

        **Abstract**:

        We consider online learning with feedback graphs, a sequential decision-making framework where the learner's feedback is determined by a directed graph over the action set. We present a computationally-efficient algorithm for learning in this framework that simultaneously achieves near-optimal regret bounds in both stochastic and adversarial environments. The bound against oblivious adversaries is $\tilde{O} (\sqrt{\alpha T})$, where $T$ is the time horizon and $\alpha$ is the independence number of the feedback graph. The bound against stochastic environments is $O\big((\ln T)^2 \max_{S\in \mathcal I(G)} \sum_{i \in S} \Delta_i^{-1}\big)$ where $\mathcal I(G)$ is the family of all independent sets in a suitably defined undirected version of the graph and $\Delta_i$ are the suboptimality gaps.The algorithm combines ideas from the EXP3++ algorithm for stochastic and adversarial bandits and the EXP3.G algorithm for feedback graphs with a novel exploration scheme. The scheme, which exploits the structure of the graph to reduce exploration, is key to obtain best-of-both-worlds guarantees with feedback graphs. We also extend our algorithm and results to a setting where the feedback graphs are allowed to change over time.

        ----

        ## [2539] Optimal Query Complexities for Dynamic Trace Estimation

        **Authors**: *David P. Woodruff, Fred Zhang, Richard Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e3abc125ecacb71786cefb9f67b08c5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e3abc125ecacb71786cefb9f67b08c5d-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of minimizing the number of matrix-vector queries needed for accurate trace estimation in the dynamic setting where our underlying matrix is changing slowly, such as during an optimization process. Specifically, for any $m$ matrices $\mathbf{A}_1,...,\mathbf{A}_m$ with consecutive differences bounded in Schatten-$1$ norm by $\alpha$, we provide a novel binary tree summation procedure that simultaneously estimates all $m$ traces up to $\epsilon$ error with $\delta$ failure probability with an optimal query complexity of $\widetilde{O}(m \alpha\sqrt{\log(1/\delta)}/\epsilon + m\log(1/\delta))$, improving the dependence on both $\alpha$ and $\delta$ from Dharangutte and Musco (NeurIPS, 2021). Our procedure works without additional norm bounds on $\mathbf{A}_i$ and can be generalized to a bound for the $p$-th Schatten norm for $p \in [1,2]$, giving a complexity of $\widetilde{O}(m \alpha(\sqrt{\log(1/\delta)}/\epsilon)^p +m \log(1/\delta))$. By using novel reductions to communication complexity and information-theoretic analyses of Gaussian matrices, we provide matching lower bounds for static and dynamic trace estimation in all relevant parameters, including the failure probability. Our lower bounds (1) give the first tight bounds for Hutchinson's estimator in the matrix-vector product model with Frobenius norm error {\it even in the static setting}, and (2) are the first unconditional lower bounds for dynamic trace estimation, resolving open questions of prior work.

        ----

        ## [2540] Learning from Stochastically Revealed Preference

        **Authors**: *John R. Birge, Xiaocheng Li, Chunlin Sun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e3c3473812173643147170188ef2b141-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e3c3473812173643147170188ef2b141-Abstract-Conference.html)

        **Abstract**:

        We study the learning problem of revealed preference in a stochastic setting: a learner observes the utility-maximizing actions of a set of agents whose utility follows some unknown distribution, and the learner aims to infer the distribution through the observations of actions. The problem can be viewed as a single-constraint special case of the inverse linear optimization problem. Existing works all assume that all the agents share one common utility which can easily be violated under practical contexts. In this paper, we consider two settings for the underlying utility distribution: a Gaussian setting where the customer utility follows the von Mises-Fisher distribution, and a $\delta$-corruption setting where the customer utility distribution concentrates on one fixed vector with high probability and is arbitrarily corrupted otherwise. We devise Bayesian approaches for parameter estimation and develop theoretical guarantees for the recovery of the true parameter. We illustrate the algorithm performance through numerical experiments.

        ----

        ## [2541] Mutual Information Divergence: A Unified Metric for Multimodal Generative Models

        **Authors**: *Jin-Hwa Kim, Yunji Kim, Jiyoung Lee, Kang Min Yoo, Sang-Woo Lee*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e40b60677880e7e74f8a081f65703f0d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e40b60677880e7e74f8a081f65703f0d-Abstract-Conference.html)

        **Abstract**:

        Text-to-image generation and image captioning are recently emerged as a new experimental paradigm to assess machine intelligence. They predict continuous quantity accompanied by their sampling techniques in the generation, making evaluation complicated and intractable to get marginal distributions. Based on a recent trend that multimodal generative evaluations exploit a vison-and-language pre-trained model, we propose the negative Gaussian cross-mutual information using the CLIP features as a unified metric, coined by Mutual Information Divergence (MID). To validate, we extensively compare it with competing metrics using carefully-generated or human-annotated judgments in text-to-image generation and image captioning tasks. The proposed MID significantly outperforms the competitive methods by having consistency across benchmarks, sample parsimony, and robustness toward the exploited CLIP model. We look forward to seeing the underrepresented implications of the Gaussian cross-mutual information in multimodal representation learning and future works based on this novel proposition.

        ----

        ## [2542] Delving into Out-of-Distribution Detection with Vision-Language Representations

        **Authors**: *Yifei Ming, Ziyang Cai, Jiuxiang Gu, Yiyou Sun, Wei Li, Yixuan Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e43a33994a28f746dcfd53eb51ed3c2d-Abstract-Conference.html)

        **Abstract**:

        Recognizing out-of-distribution (OOD) samples is critical for machine learning systems deployed in the open world. The vast majority of OOD detection methods are driven by a single modality (e.g., either vision or language), leaving the rich information in multi-modal representations untapped. Inspired by the recent success of vision-language pre-training, this paper enriches the landscape of OOD detection from a single-modal to a multi-modal regime. Particularly, we propose Maximum Concept Matching (MCM), a simple yet effective zero-shot OOD detection method based on aligning visual features with textual concepts.  We contribute in-depth analysis and theoretical insights to understand the effectiveness of MCM. Extensive experiments demonstrate that MCM achieves superior performance on a wide variety of real-world tasks. MCM with vision-language features outperforms a common baseline with pure visual features on a hard OOD task with semantically similar classes by 13.1% (AUROC) Code is available at https://github.com/deeplearning-wisc/MCM.

        ----

        ## [2543] OnePose++: Keypoint-Free One-Shot Object Pose Estimation without CAD Models

        **Authors**: *Xingyi He, Jiaming Sun, Yuang Wang, Di Huang, Hujun Bao, Xiaowei Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e43f900f571de6c96a70d5724a0fb565-Abstract-Conference.html)

        **Abstract**:

        We propose a new method for object pose estimation without CAD models. The previous feature-matching-based method OnePose has shown promising results under a one-shot setting which eliminates the need for CAD models or object-specific training. However, OnePose relies on detecting repeatable image keypoints and is thus prone to failure on low-textured objects. We propose a keypoint-free pose estimation pipeline to remove the need for repeatable keypoint detection. Built upon the detector-free feature matching method LoFTR, we devise a new keypoint-free SfM method to reconstruct a semi-dense point-cloud model for the object. Given a query image for object pose estimation, a 2D-3D matching network directly establishes 2D-3D correspondences between the query image and the reconstructed point-cloud model without first detecting keypoints in the image. Experiments show that the proposed pipeline outperforms existing one-shot CAD-model-free methods by a large margin and is comparable to CAD-model-based methods on LINEMOD even for low-textured objects. We also collect a new dataset composed of 80 sequences of 40 low-textured objects to facilitate future research on one-shot object pose estimation. The supplementary material, code and dataset are available on the project page: https://zju3dv.github.io/oneposeplusplus/.

        ----

        ## [2544] Confidence-based Reliable Learning under Dual Noises

        **Authors**: *Peng Cui, Yang Yue, Zhijie Deng, Jun Zhu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e444859b2a22df6b56af9381ad1e9480-Abstract-Conference.html)

        **Abstract**:

        Deep neural networks (DNNs) have achieved remarkable success in a variety of computer vision tasks, where massive labeled images are routinely required for model optimization. Yet, the data collected from the open world are unavoidably polluted by noise, which may significantly undermine the efficacy of the learned models. Various attempts have been made to reliably train DNNs under data noise, but they separately account for either the noise existing in the labels or that existing in the images. A naive combination of the two lines of works would suffer from the limitations in both sides, and miss the opportunities to handle the two kinds of noise in parallel. This works provides a first, unified framework for reliable learning under the joint (image, label)-noise. Technically, we develop a confidence-based sample filter to progressively filter out noisy data without the need of pre-specifying noise ratio. Then, we penalize the model uncertainty of the detected noisy data instead of letting the model continue over-fitting the misleading information in them. Experimental results on various challenging synthetic and real-world noisy datasets verify that the proposed method can outperform competing baselines in the aspect of classification performance.

        ----

        ## [2545] Multitasking Models are Robust to Structural Failure: A Neural Model for Bilingual Cognitive Reserve

        **Authors**: *Giannis Daras, Negin Raoof, Zoi Gkalitsiou, Alex Dimakis*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e45caa3d5273d105b8d045e748636957-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e45caa3d5273d105b8d045e748636957-Abstract-Conference.html)

        **Abstract**:

        We find a surprising connection between multitask learning and robustness to neuron failures. Our experiments show that bilingual language models retain higher performance under various neuron perturbations, such as random deletions, magnitude pruning and weight noise. Our study is motivated by research in cognitive science showing that symptoms of dementia and cognitive decline appear later in bilingual speakers compared to monolingual patients with similar brain damage, a phenomenon called bilingual cognitive reserve. Our language model experiments replicate this phenomenon on bilingual GPT-2 and other models.We provide a theoretical justification of this robustness by mathematically analyzing linear representation learning and showing that multitasking creates more robust representations. We open-source our code and models in the following URL: https://github.com/giannisdaras/multilingual_robustness.

        ----

        ## [2546] Learning (Very) Simple Generative Models Is Hard

        **Authors**: *Sitan Chen, Jerry Li, Yuanzhi Li*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e463a2a3daee91a6dd0e0c41cd2f9f7a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e463a2a3daee91a6dd0e0c41cd2f9f7a-Abstract-Conference.html)

        **Abstract**:

        Motivated by the recent empirical successes of deep generative models, we study the computational complexity of the following unsupervised learning problem. For an unknown neural network $F:\mathbb{R}^d\to\mathbb{R}^{d'}$, let $D$ be the distribution over $\mathbb{R}^{d'}$ given by pushing the standard Gaussian $\mathcal{N}(0,\textrm{Id}_d)$ through $F$. Given i.i.d. samples from $D$, the goal is to output *any* distribution close to $D$ in statistical distance.    We show under the statistical query (SQ) model that no polynomial-time algorithm can solve this problem even when the output coordinates of $F$ are one-hidden-layer ReLU networks with $\log(d)$ neurons. Previously, the best lower bounds for this problem simply followed from lower bounds for *supervised learning* and required at least two hidden layers and $\textrm{poly}(d)$ neurons [Daniely-Vardi '21, Chen-Gollakota-Klivans-Meka '22].    The key ingredient in our proof is an ODE-based construction of a compactly supported, piecewise-linear function $f$ with polynomially-bounded slopes such that the pushforward of $\mathcal{N}(0,1)$ under $f$ matches all low-degree moments of $\mathcal{N}(0,1)$.

        ----

        ## [2547] PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding

        **Authors**: *Minghao Xu, Zuobai Zhang, Jiarui Lu, Zhaocheng Zhu, Yangtian Zhang, Chang Ma, Runcheng Liu, Jian Tang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e467582d42d9c13fa9603df16f31de6d-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        We are now witnessing significant progress of deep learning methods in a variety of tasks (or datasets) of proteins. However, there is a lack of a standard benchmark to evaluate the performance of different methods, which hinders the progress of deep learning in this field. In this paper, we propose such a benchmark called PEER, a comprehensive and multi-task benchmark for Protein sEquence undERstanding. PEER provides a set of diverse protein understanding tasks including protein function prediction, protein localization prediction, protein structure prediction, protein-protein interaction prediction, and protein-ligand interaction prediction. We evaluate different types of sequence-based methods for each task including traditional feature engineering approaches, different sequence encoding methods as well as large-scale pre-trained protein language models. In addition, we also investigate the performance of these methods under the multi-task learning setting. Experimental results show that large-scale pre-trained protein language models achieve the best performance for most individual tasks, and jointly training multiple tasks further boosts the performance. The datasets and source codes of this benchmark will be open-sourced soon.

        ----

        ## [2548] Bring Your Own Algorithm for Optimal Differentially Private Stochastic Minimax Optimization

        **Authors**: *Liang Zhang, Kiran Koshy Thekumparampil, Sewoong Oh, Niao He*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e46fc33e80e9fa2febcdb058fba4beca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e46fc33e80e9fa2febcdb058fba4beca-Abstract-Conference.html)

        **Abstract**:

        We study differentially private (DP) algorithms for smooth stochastic minimax optimization, with stochastic minimization as a byproduct. The holy grail of these settings is to guarantee the optimal trade-off between the privacy and the excess population loss, using an algorithm with a linear time-complexity in the number of training samples. We provide a general framework for solving differentially private stochastic minimax optimization (DP-SMO) problems, which enables the practitioners to bring their own base optimization algorithm and use it as a black-box to obtain the near-optimal privacy-loss trade-off. Our framework is inspired from the recently proposed Phased-ERM method [22] for nonsmooth differentially private stochastic convex optimization (DP-SCO), which exploits the stability of the empirical risk minimization (ERM) for the privacy guarantee. The flexibility of our approach enables us to sidestep the requirement that the base algorithm needs to have bounded sensitivity, and allows the use of sophisticated variance-reduced accelerated methods to achieve near-linear time-complexity. To the best of our knowledge, these are the first near-linear time algorithms with near-optimal guarantees on the population duality gap for smooth DP-SMO, when the objective is (strongly-)convex--(strongly-)concave. Additionally, based on our flexible framework, we enrich the family of near-linear time algorithms for smooth DP-SCO with the near-optimal privacy-loss trade-off.

        ----

        ## [2549] A Regret-Variance Trade-Off in Online Learning

        **Authors**: *Dirk van der Hoeven, Nikita Zhivotovskiy, Nicolò Cesa-Bianchi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e473f29459a4a006d4e968537b135e40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e473f29459a4a006d4e968537b135e40-Abstract-Conference.html)

        **Abstract**:

        We consider prediction with expert advice for strongly convex and bounded losses, and investigate trade-offs between regret and ``variance'' (i.e., squared difference of learner's predictions and best expert predictions).With $K$ experts, the Exponentially Weighted Average (EWA) algorithm is known to achieve $O(\log K)$ regret.We prove that a variant of EWA either achieves a \textsl{negative} regret (i.e., the algorithm outperforms the best expert), or guarantees a $O(\log K)$ bound on \textsl{both} variance and regret.Building on this result, we show several examples of how variance of predictions can be exploited in learning.In the online to batch analysis, we show that a large empirical variance allows to stop the online to batch conversion early and outperform the risk of the best predictor in the class. We also recover the optimal rate of model selection aggregation when we do not consider early stopping.In online prediction with corrupted losses, we show that the effect of corruption on the regret can be compensated by a large variance.In online selective sampling, we design an algorithm that samples less when the variance is large, while guaranteeing the optimal regret bound in expectation.In online learning with abstention, we use a similar term as the variance to derive the first high-probability $O(\log K)$ regret bound in this setting.Finally, we extend our results to the setting of online linear regression.

        ----

        ## [2550] Alternating Mirror Descent for Constrained Min-Max Games

        **Authors**: *Andre Wibisono, Molei Tao, Georgios Piliouras*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e496e0ce207ba9cdcc7d79bd499db67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e496e0ce207ba9cdcc7d79bd499db67e-Abstract-Conference.html)

        **Abstract**:

        In this paper we study two-player bilinear zero-sum games with constrained strategy spaces. An instance of natural occurrences of such constraints is when mixed strategies are used, which correspond to a probability simplex constraint. We propose and analyze the alternating mirror descent algorithm, in which each player takes turns to take action following the mirror descent algorithm for constrained optimization. We interpret alternating mirror descent as an alternating discretization of a skew-gradient flow in the dual space, and use tools from convex optimization and modified energy function to establish an $O(K^{-2/3})$ bound on its average regret after $K$ iterations. This quantitatively verifies the algorithm's  better behavior than the simultaneous version of mirror descent algorithm, which is known to diverge and yields an $O(K^{-1/2})$ average regret bound. In the special case of an unconstrained setting, our results recover the behavior of alternating gradient descent algorithm for zero-sum games which was studied in (Bailey et al., COLT 2020).

        ----

        ## [2551] Deep Counterfactual Estimation with Categorical Background Variables

        **Authors**: *Edward De Brouwer*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e4a0d8aef3567f742b0794844d9b5847-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e4a0d8aef3567f742b0794844d9b5847-Abstract-Conference.html)

        **Abstract**:

        Referred to as the third rung of the causal inference ladder, counterfactual queries typically ask the "What if ?" question retrospectively. The standard approach to estimate counterfactuals resides in using a structural equation model that accurately reflects the underlying data generating process. However, such models are seldom available in practice and one usually wishes to infer them from observational data alone. Unfortunately, the correct structural equation model is in general not identifiable from the observed factual distribution. Nevertheless, in this work, we show that under the assumption that the main latent contributors to the treatment responses are categorical, the counterfactuals can be still reliably predicted. Building upon this assumption, we introduce CounterFactual Query Prediction (\method), a novel method to infer counterfactuals from continuous observations when the background variables are categorical. We show that our method significantly outperforms previously available deep-learning-based counterfactual methods, both theoretically and empirically on time series and image data. Our code is available at https://github.com/edebrouwer/cfqp.

        ----

        ## [2552] SnAKe: Bayesian Optimization with Pathwise Exploration

        **Authors**: *Jose Pablo Folch, Shiqiang Zhang, Robert M. Lee, Behrang Shafei, David Walz, Calvin Tsay, Mark van der Wilk, Ruth Misener*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e4bab1843c8d5a69f5abfd0824593493-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e4bab1843c8d5a69f5abfd0824593493-Abstract-Conference.html)

        **Abstract**:

        "Bayesian Optimization is a very effective tool for optimizing expensive black-box functions. Inspired by applications developing and characterizing reaction chemistry using droplet microfluidic reactors, we consider a novel setting where the expense of evaluating the function can increase significantly when making large input changes between iterations. We further assume we are working asynchronously, meaning we have to decide on new queries before we finish evaluating previous experiments. This paper investigates the problem and introduces 'Sequential Bayesian Optimization via Adaptive Connecting Samples' (SnAKe), which provides a solution by considering large batches of queries and preemptively building optimization paths that minimize input costs. We investigate some convergence properties and empirically show that the algorithm is able to achieve regret similar to classical Bayesian Optimization algorithms in both the synchronous and asynchronous settings, while reducing the input costs significantly. We show the method is robust to the choice of its single hyper-parameter and provide a parameter-free alternative."

        ----

        ## [2553] Self-Supervised Learning with an Information Maximization Criterion

        **Authors**: *Serdar Ozsoy, Shadi Hamdan, Sercan Ö. Arik, Deniz Yuret, Alper T. Erdogan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html)

        **Abstract**:

        Self-supervised learning allows AI systems to learn effective representations from large amounts of data using tasks that do not require costly labeling. Mode collapse, i.e., the model producing identical representations for all inputs, is a central problem to many self-supervised learning approaches, making self-supervised tasks, such as matching distorted variants of the inputs, ineffective. In this article, we argue that a straightforward application of information maximization among alternative latent representations of the same input naturally solves the collapse problem and achieves competitive empirical results. We propose a self-supervised learning method, CorInfoMax, that uses a second-order statistics-based mutual information measure that reflects the level of correlation among its arguments. Maximizing this correlative information measure between alternative representations of the same input serves two purposes: (1) it avoids the collapse problem by generating feature vectors with non-degenerate covariances; (2) it establishes relevance among alternative representations by increasing the linear dependence among them. An approximation of the proposed information maximization objective simplifies to a Euclidean distance-based objective function regularized by the log-determinant of the feature covariance matrix. The regularization term acts as a natural barrier against feature space degeneracy. Consequently, beyond avoiding complete output collapse to a single point, the proposed approach also prevents dimensional collapse by encouraging the spread of information across the whole feature space. Numerical experiments demonstrate that CorInfoMax achieves better or competitive performance results relative to the state-of-the-art SSL approaches.

        ----

        ## [2554] TwiBot-22: Towards Graph-Based Twitter Bot Detection

        **Authors**: *Shangbin Feng, Zhaoxuan Tan, Herun Wan, Ningnan Wang, Zilong Chen, Binchi Zhang, Qinghua Zheng, Wenqian Zhang, Zhenyu Lei, Shujie Yang, Xinshun Feng, Qingyue Zhang, Hongrui Wang, Yuhan Liu, Yuyang Bai, Heng Wang, Zijian Cai, Yanbo Wang, Lijing Zheng, Zihan Ma, Jundong Li, Minnan Luo*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e4fd610b1d77699a02df07ae97de992a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e4fd610b1d77699a02df07ae97de992a-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Twitter bot detection has become an increasingly important task to combat misinformation, facilitate social media moderation, and preserve the integrity of the online discourse. State-of-the-art bot detection methods generally leverage the graph structure of the Twitter network, and they exhibit promising performance when confronting novel Twitter bots that traditional methods fail to detect. However, very few of the existing Twitter bot detection datasets are graph-based, and even these few graph-based datasets suffer from limited dataset scale, incomplete graph structure, as well as low annotation quality. In fact, the lack of a large-scale graph-based Twitter bot detection benchmark that addresses these issues has seriously hindered the development and evaluation of novel graph-based bot detection approaches. In this paper, we propose TwiBot-22, a comprehensive graph-based Twitter bot detection benchmark that presents the largest dataset to date, provides diversified entities and relations on the Twitter network, and has considerably better annotation quality than existing datasets. In addition, we re-implement 35 representative Twitter bot detection baselines and evaluate them on 9 datasets, including TwiBot-22, to promote a fair comparison of model performance and a holistic understanding of research progress. To facilitate further research, we consolidate all implemented codes and datasets into the TwiBot-22 evaluation framework, where researchers could consistently evaluate new models and datasets. The TwiBot-22 Twitter bot detection benchmark and evaluation framework are publicly available at \url{https://twibot22.github.io/}.

        ----

        ## [2555] Listen to Interpret: Post-hoc Interpretability for Audio Networks with NMF

        **Authors**: *Jayneel Parekh, Sanjeel Parekh, Pavlo Mozharovskyi, Florence d'Alché-Buc, Gaël Richard*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e53280d73dd5389e820f4a6250365b0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e53280d73dd5389e820f4a6250365b0e-Abstract-Conference.html)

        **Abstract**:

        This paper tackles post-hoc interpretability for audio processing networks. Our goal is to interpret decisions of a trained network in terms of high-level audio objects that are also listenable for the end-user. To this end, we propose a novel interpreter design that incorporates non-negative matrix factorization (NMF). In particular, a regularized interpreter module is trained to take hidden layer representations of the targeted network as input and produce time activations of pre-learnt NMF components as intermediate outputs. Our methodology allows us to generate intuitive audio-based interpretations that explicitly enhance parts of the input signal most relevant for a network's decision. We demonstrate our method's applicability on popular benchmarks, including a real-world multi-label classification task.

        ----

        ## [2556] Model-based RL with Optimistic Posterior Sampling: Structural Conditions and Sample Complexity

        **Authors**: *Alekh Agarwal, Tong Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e536e43b01a4387a2282c2b04103c802-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e536e43b01a4387a2282c2b04103c802-Abstract-Conference.html)

        **Abstract**:

        We propose a general framework to design posterior sampling methods for model-based RL. We show that the proposed algorithms can be analyzed by reducing regret to Hellinger distance in conditional probability estimation. We further show that optimistic posterior sampling can control this Hellinger distance, when we measure model error via data likelihood. This technique allows us to design and analyze unified posterior sampling algorithms with state-of-the-art sample complexity guarantees for many model-based RL settings. We illustrate our general result in many special cases, demonstrating the versatility of our framework.

        ----

        ## [2557] Deep Architecture Connectivity Matters for Its Convergence: A Fine-Grained Analysis

        **Authors**: *Wuyang Chen, Wei Huang, Xinyu Gong, Boris Hanin, Zhangyang Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e54e6eef11f87a874bf1e4551fc6d04e-Abstract-Conference.html)

        **Abstract**:

        Advanced deep neural networks (DNNs), designed by either human or AutoML algorithms, are growing increasingly complex. Diverse operations are connected by complicated connectivity patterns, e.g., various types of skip connections. Those topological compositions are empirically effective and observed to smooth the loss landscape and facilitate the gradient flow in general. However, it remains elusive to derive any principled understanding of their effects on the DNN capacity or trainability, and to understand why or in which aspect one specific connectivity pattern is better than another. In this work, we theoretically characterize the impact of connectivity patterns on the convergence of DNNs under gradient descent training in fine granularity. By analyzing a wide network's Neural Network Gaussian Process (NNGP), we are able to depict how the spectrum of an NNGP kernel propagates through a particular connectivity pattern, and how that affects the bound of convergence rates. As one practical implication of our results, we show that by a simple filtration of "unpromising" connectivity patterns, we can trim down the number of models to evaluate, and significantly accelerate the large-scale neural architecture search without any overhead.

        ----

        ## [2558] OrdinalCLIP: Learning Rank Prompts for Language-Guided Ordinal Regression

        **Authors**: *Wanhua Li, Xiaoke Huang, Zheng Zhu, Yansong Tang, Xiu Li, Jie Zhou, Jiwen Lu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e55b33430e344a1ee23710415b1c9d87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e55b33430e344a1ee23710415b1c9d87-Abstract-Conference.html)

        **Abstract**:

        This paper presents a language-powered paradigm for ordinal regression. Existing methods usually treat each rank as a category and employ a set of weights to learn these concepts. These methods are easy to overfit and usually attain unsatisfactory performance as the learned concepts are mainly derived from the training set. Recent large pre-trained vision-language models like CLIP have shown impressive performance on various visual tasks. In this paper, we propose to learn the rank concepts from the rich semantic CLIP latent space. Specifically, we reformulate this task as an image-language matching problem with a contrastive objective, which regards labels as text and obtains a language prototype from a text encoder for each rank. While prompt engineering for CLIP is extremely time-consuming, we propose OrdinalCLIP, a differentiable prompting method for adapting CLIP for ordinal regression. OrdinalCLIP consists of learnable context tokens and learnable rank embeddings. The learnable rank embeddings are constructed by explicitly modeling numerical continuity, resulting in well-ordered, compact language prototypes in the CLIP space. Once learned, we can only save the language prototypes and discard the huge language model, resulting in zero additional computational overhead compared with the linear head counterpart. Experimental results show that our paradigm achieves competitive performance in general ordinal regression tasks, and gains improvements in few-shot and distribution shift settings for age estimation. The code is available at https://github.com/xk-huang/OrdinalCLIP.

        ----

        ## [2559] Kernel Memory Networks: A Unifying Framework for Memory Modeling

        **Authors**: *Georgios Iatropoulos, Johanni Brea, Wulfram Gerstner*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e55d081280e79e714debf2902e18eb69-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e55d081280e79e714debf2902e18eb69-Abstract-Conference.html)

        **Abstract**:

        We consider the problem of training a neural network to store a set of patterns with maximal noise robustness. A solution, in terms of optimal weights and state update rules, is derived by training each individual neuron to perform either kernel classification or interpolation with a minimum weight norm. By applying this method to feed-forward and recurrent networks, we derive optimal models, termed kernel memory networks, that include, as special cases, many of the hetero- and auto-associative memory models that have been proposed over the past years, such as modern Hopfield networks and Kanerva's sparse distributed memory. We modify Kanerva's model and demonstrate a simple way to design a kernel memory network that can store an exponential number of continuous-valued patterns with a finite basin of attraction. The framework of kernel memory networks offers a simple and intuitive way to understand the storage capacity of previous memory models, and allows for new biological interpretations in terms of dendritic non-linearities and synaptic cross-talk.

        ----

        ## [2560] The First Optimal Acceleration of High-Order Methods in Smooth Convex Optimization

        **Authors**: *Dmitry Kovalev, Alexander V. Gasnikov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e56f394bbd4f0ec81393d767caa5a31b-Abstract-Conference.html)

        **Abstract**:

        In this paper, we study the fundamental open question of finding the optimal high-order algorithm for solving smooth convex minimization problems. Arjevani et al. (2019) established the lower bound $\Omega\left(\epsilon^{-2/(3p+1)}\right)$ on the number of the $p$-th order oracle calls required by an algorithm to find an $\epsilon$-accurate solution to the problem, where the $p$-th order oracle stands for the computation of the objective function value and the derivatives up to the order $p$. However, the existing state-of-the-art high-order methods of Gasnikov et al. (2019b); Bubeck et al. (2019); Jiang et al. (2019) achieve the oracle complexity $\mathcal{O}\left(\epsilon^{-2/(3p+1)} \log (1/\epsilon)\right)$, which does not match the lower bound. The reason for this is that these algorithms require performing a complex binary search procedure, which makes them neither optimal nor practical. We fix this fundamental issue by providing the first algorithm with $\mathcal{O}\left(\epsilon^{-2/(3p+1)}\right)$ $p$-th order oracle complexity.

        ----

        ## [2561] CCCP is Frank-Wolfe in disguise

        **Authors**: *Alp Yurtsever, Suvrit Sra*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e589022774244df75b97eae18bb3628d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e589022774244df75b97eae18bb3628d-Abstract-Conference.html)

        **Abstract**:

        This paper uncovers a simple but rather surprising connection: it shows that the well-known convex-concave procedure (CCCP) and its generalization to constrained problems are both special cases of the Frank-Wolfe (FW) method. This connection not only provides insight of deep (in our opinion) pedagogical value, but also transfers the recently discovered convergence theory of nonconvex Frank-Wolfe methods immediately to CCCP, closing a long-standing gap in its non-asymptotic convergence theory. We hope the viewpoint uncovered by this paper spurs the transfer of other advances made for FW to both CCCP and its generalizations.

        ----

        ## [2562] Uni[MASK]: Unified Inference in Sequential Decision Problems

        **Authors**: *Micah Carroll, Orr Paradise, Jessy Lin, Raluca Georgescu, Mingfei Sun, David Bignell, Stephanie Milani, Katja Hofmann, Matthew J. Hausknecht, Anca D. Dragan, Sam Devlin*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e58fa6a7b431e634e0fd125e225ad10c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e58fa6a7b431e634e0fd125e225ad10c-Abstract-Conference.html)

        **Abstract**:

        Randomly masking and predicting word tokens has been a successful approach in pre-training language models for a variety of downstream tasks. In this work, we observe that the same idea also applies naturally to sequential decision making, where many well-studied tasks like behavior cloning, offline RL, inverse dynamics, and waypoint conditioning correspond to different sequence maskings over a sequence of states, actions, and returns. We introduce the UniMASK framework, which provides a unified way to specify models which can be trained on many different sequential decision making tasks. We show that a single UniMASK model is often capable of carrying out many tasks with performance similar to or better than single-task models. Additionally, after fine-tuning, our UniMASK models consistently outperform comparable single-task models.

        ----

        ## [2563] Efficient Active Learning with Abstention

        **Authors**: *Yinglun Zhu, Robert Nowak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e5aa7171449b83f8b4eec1623eac9906-Abstract-Conference.html)

        **Abstract**:

        The goal of active learning is to achieve the same accuracy achievable by passive learning, while using much fewer labels. Exponential savings in terms of label complexity have been proved in very special cases, but fundamental lower bounds show that such improvements are impossible in general. This suggests a need to explore alternative goals for active learning. Learning with abstention is one such alternative.  In this setting, the active learning algorithm may abstain from prediction and incur an error that is marginally smaller than random guessing. We develop the first computationally efficient active learning algorithm with abstention. Our algorithm provably achieves $\mathsf{polylog}(\frac{1}{\varepsilon})$ label complexity, without any low noise conditions. Such performance guarantee reduces the label complexity by an exponential factor, relative to passive learning and active learning that is not allowed to abstain. Furthermore, our algorithm is guaranteed to only abstain on hard examples (where the true label distribution is close to a fair coin), a novel property we term \emph{proper abstention} that also leads to a host of other desirable characteristics (e.g., recovering minimax guarantees in the standard setting, and avoiding the undesirable ``noise-seeking'' behavior often seen in active learning). We also provide novel extensions of our algorithm that achieve \emph{constant} label complexity and deal with model misspecification.

        ----

        ## [2564] SInGE: Sparsity via Integrated Gradients Estimation of Neuron Relevance

        **Authors**: *Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, Kevin Bailly*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e5b0ec2c61957bfd6cf88e118107cc71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e5b0ec2c61957bfd6cf88e118107cc71-Abstract-Conference.html)

        **Abstract**:

        The leap in performance in state-of-the-art computer vision methods is attributed to the development of deep neural networks. However it often comes at a computational price which may hinder their deployment. To alleviate this limitation, structured pruning is a well known technique which consists in removing channels, neurons or filters, and is commonly applied in order to produce more compact models. In most cases, the computations to remove are selected based on a relative importance criterion. At the same time, the need for explainable predictive models has risen tremendously and motivated the development of robust attribution methods that highlight the relative importance of pixels of an input image or feature map. In this work, we discuss the limitations of existing pruning heuristics, among which magnitude and gradient-based methods. We draw inspiration from attribution methods to design a novel integrated gradient pruning criterion, in which the relevance of each neuron is defined as the integral of the gradient variation on a path towards this neuron removal. Furthermore, We propose an entwined DNN pruning and fine-tuning flowchart to better preserve DNN accuracy while removing parameters. We show through extensive validation on several datasets, architectures as well as pruning scenarios that the proposed method, dubbed SInGE, significantly outperforms existing state-of-the-art DNN pruning methods.

        ----

        ## [2565] On the Complexity of Adversarial Decision Making

        **Authors**: *Dylan J. Foster, Alexander Rakhlin, Ayush Sekhari, Karthik Sridharan*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e5daf532498ebba4fe6544d49c811678-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e5daf532498ebba4fe6544d49c811678-Abstract-Conference.html)

        **Abstract**:

        A central problem in online learning and decision making---from bandits to reinforcement learning---is to understand what modeling assumptions lead to sample-efficient learning guarantees. We consider a general adversarial decision making framework that encompasses (structured) bandit problems with adversarial rewards and reinforcement learning problems with adversarial dynamics. Our main result is to show---via new upper and lower bounds---that the Decision-Estimation Coefficient, a complexity measure introduced by Foster et al. in the stochastic counterpart to our setting, is necessary and sufficient to obtain low regret for adversarial decision making. However, compared to the stochastic setting, one must apply the Decision-Estimation Coefficient to the convex hull of the class of models (or, hypotheses) under consideration. This establishes that the price of accommodating adversarial rewards or dynamics is governed by the behavior of the model class under convexification, and recovers a number of existing results --both positive and negative. En route to obtaining these guarantees, we provide new structural results that connect the Decision-Estimation Coefficient to variants of other well-known complexity measures, including the Information Ratio of Russo and Van Roy and the Exploration-by-Optimization objective of Lattimore and Gy√∂rgy.

        ----

        ## [2566] MoCapAct: A Multi-Task Dataset for Simulated Humanoid Control

        **Authors**: *Nolan Wagener, Andrey Kolobov, Felipe Vieira Frujeri, Ricky Loynd, Ching-An Cheng, Matthew J. Hausknecht*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e5dd4fbb6fb4cb805b982bfb41c20aad-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e5dd4fbb6fb4cb805b982bfb41c20aad-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Simulated humanoids are an appealing research domain due to their physical capabilities. Nonetheless, they are also challenging to control, as a policy must drive an unstable, discontinuous, and high-dimensional physical system. One widely studied approach is to utilize motion capture (MoCap) data to teach the humanoid agent low-level skills (e.g., standing, walking, and running) that can then be re-used to synthesize high-level behaviors. However, even with MoCap data, controlling simulated humanoids remains very hard, as MoCap data offers only kinematic information. Finding physical control inputs to realize the demonstrated motions requires computationally intensive methods like reinforcement learning. Thus, despite the publicly available MoCap data, its utility has been limited to institutions with large-scale compute. In this work, we dramatically lower the barrier for productive research on this topic by training and releasing high-quality agents that can track over three hours of MoCap data for a simulated humanoid in the dmcontrol physics-based environment. We release MoCapAct (Motion Capture with Actions), a dataset of these expert agents and their rollouts, which contain proprioceptive observations and actions. We demonstrate the utility of MoCapAct by using it to train a single hierarchical policy capable of tracking the entire MoCap dataset within dmcontrol and show the learned low-level component can be re-used to efficiently learn downstream high-level tasks. Finally, we use MoCapAct to train an autoregressive GPT model and show that it can control a simulated humanoid to perform natural motion completion given a motion prompt.Videos of the results and links to the code and dataset are available at https://microsoft.github.io/MoCapAct.

        ----

        ## [2567] On the Effectiveness of Persistent Homology

        **Authors**: *Renata Turkes, Guido F. Montúfar, Nina Otter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e637029c42aa593850eeebf46616444d-Abstract-Conference.html)

        **Abstract**:

        Persistent homology (PH) is one of the most popular methods in Topological Data Analysis. Even though PH has been used in many different types of applications, the reasons behind its success remain elusive; in particular, it is not known for which classes of problems it is most effective, or to what extent it can detect geometric or topological features. The goal of this work is to identify some types of problems where PH performs well or even better than other methods in data analysis. We consider three fundamental shape analysis tasks: the detection of the number of holes, curvature and convexity from 2D and 3D point clouds sampled from shapes. Experiments demonstrate that PH is successful in these tasks, outperforming several baselines, including PointNet, an architecture inspired precisely by the properties of point clouds. In addition, we observe that PH remains effective for limited computational resources and limited training data, as well as out-of-distribution test data, including various data transformations and noise. For convexity detection, we provide a theoretical guarantee that PH is effective for this task in $\mathbb{R}^d$, and demonstrate the detection of a convexity measure on the FLAVIA dataset of plant leaf images. Due to the crucial role of shape classification in understanding mathematical and physical structures and objects, and in many applications, the findings of this work will provide some knowledge about the types of problems that are appropriate for PH, so that it can --- to borrow the words from Wigner 1960 --- ``remain valid in future research, and extend, to our pleasure", but to our lesser bafflement, to a variety of applications.

        ----

        ## [2568] An Adaptive Deep RL Method for Non-Stationary Environments with Piecewise Stable Context

        **Authors**: *Xiaoyu Chen, Xiangming Zhu, Yufeng Zheng, Pushi Zhang, Li Zhao, Wenxue Cheng, Peng Cheng, Yongqiang Xiong, Tao Qin, Jianyu Chen, Tie-Yan Liu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e67a667df97d44fa43ffe8d179b44dd8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e67a667df97d44fa43ffe8d179b44dd8-Abstract-Conference.html)

        **Abstract**:

        One of the key challenges in deploying RL to real-world applications is to adapt to variations of unknown environment contexts, such as changing terrains in robotic tasks and fluctuated bandwidth in congestion control. Existing works on adaptation to unknown environment contexts either assume the contexts are the same for the whole episode or assume the context variables are Markovian. However, in many real-world applications, the environment context usually stays stable for a stochastic period and then changes in an abrupt and unpredictable manner within an episode, resulting in a segment structure, which existing works fail to address. To leverage the segment structure of piecewise stable context in real-world applications, in this paper, we propose a \textit{\textbf{Se}gmented \textbf{C}ontext \textbf{B}elief \textbf{A}ugmented \textbf{D}eep~(SeCBAD)} RL method. Our method can jointly infer the belief distribution over latent context with the posterior over segment length and perform more accurate belief context inference with observed data within the current context segment. The inferred belief context can be leveraged to augment the state, leading to a policy that can adapt to abrupt variations in context. We demonstrate empirically that SeCBAD can infer context segment length accurately and outperform existing methods on a toy grid world environment and Mujuco tasks with piecewise-stable context.

        ----

        ## [2569] Expediting Large-Scale Vision Transformer for Dense Prediction without Fine-tuning

        **Authors**: *Weicong Liang, Yuhui Yuan, Henghui Ding, Xiao Luo, Weihong Lin, Ding Jia, Zheng Zhang, Chao Zhang, Han Hu*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e6c2e85db1f1039177c4495ccd399ac4-Abstract-Conference.html)

        **Abstract**:

        Vision transformers have recently achieved competitive results across various vision tasks but still suffer from heavy computation costs when processing a large number of tokens. Many advanced approaches have been developed to reduce the total number of tokens in the large-scale vision transformers, especially for image classification tasks. Typically, they select a small group of essential tokens according to their relevance with the [\texttt{class}] token, then fine-tune the weights of the vision transformer. Such fine-tuning is less practical for dense prediction due to the much heavier computation and GPU memory cost than image classification.In this paper, we focus on a more challenging problem, \ie, accelerating large-scale vision transformers for dense prediction without any additional re-training or fine-tuning. In response to the fact that high-resolution representations are necessary for dense prediction, we present two non-parametric operators, a \emph{token clustering layer} to decrease the number of tokens and a \emph{token reconstruction layer} to increase the number of tokens. The following steps are performed to achieve this: (i) we use the token clustering layer to cluster the neighboring tokens together, resulting in low-resolution representations that maintain the spatial structures; (ii) we apply the following transformer layers only to these low-resolution representations or clustered tokens; and (iii) we use the token reconstruction layer to re-create the high-resolution representations from the refined low-resolution representations. The results obtained by our method are promising on five dense prediction tasks including object detection, semantic segmentation, panoptic segmentation, instance segmentation, and depth estimation. Accordingly, our method accelerates $40\%\uparrow$ FPS and saves $30\%\downarrow$ GFLOPs of ``Segmenter+ViT-L/$16$'' while maintaining $99.5\%$ of the performance on ADE$20$K without fine-tuning the official weights.

        ----

        ## [2570] Flowification: Everything is a normalizing flow

        **Authors**: *Bálint Máté, Samuel Klein, Tobias Golling, François Fleuret*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e6c5195dac675f03d0fcf3955bcdd3c9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e6c5195dac675f03d0fcf3955bcdd3c9-Abstract-Conference.html)

        **Abstract**:

        The two key characteristics of a normalizing flow is that it is invertible (in particular, dimension preserving) and that it monitors the amount by which it changes the likelihood of data points as samples are propagated along the network. Recently, multiple generalizations of normalizing flows have been introduced that relax these two conditions \citep{nielsen2020survae,huang2020augmented}. On the other hand, neural networks only perform a forward pass on the input, there is neither a notion of an inverse of a neural network nor is there one of its likelihood contribution. In this paper we argue that certain neural network architectures can be enriched with a stochastic inverse pass and that their likelihood contribution can be monitored in a way that they fall under the generalized notion of a normalizing flow mentioned above. We term this enrichment \emph{flowification}. We prove that neural networks only containing linear and convolutional layers and invertible activations such as LeakyReLU can be flowified and evaluate them in the generative setting on image datasets.

        ----

        ## [2571] A time-resolved theory of information encoding in recurrent neural networks

        **Authors**: *Rainer Engelken, Sven Goedeke*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e6f29fb27bb400f89f5584c175005679-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e6f29fb27bb400f89f5584c175005679-Abstract-Conference.html)

        **Abstract**:

        Information encoding in neural circuits depends on how well time-varying stimuli are encoded by neural populations.Slow neuronal timescales, noise and network chaos can compromise reliable and rapid population response to external stimuli.A dynamic balance of externally incoming currents by strong recurrent inhibition was previously proposed as a mechanism to accurately and robustly encode a time-varying stimulus in balanced networks of binary neurons, but a theory for recurrent rate networks was missing. Here, we develop a non-stationary dynamic mean-field theory that transparently explains how a tight balance of excitatory currents by recurrent inhibition improves information encoding. We demonstrate that the mutual information rate of a time-varying input increases linearly with the tightness of balance, both in the presence of additive noise and with recurrently generated chaotic network fluctuations. We corroborated our findings in numerical experiments and demonstrated that recurrent networks with positive firing rates trained to transmit a time-varying stimulus generically use recurrent inhibition to increase the information rate. We also found that networks trained to transmit multiple independent time-varying signals spontaneously form multiple local inhibitory clusters, one for each input channel.Our findings suggest that feedforward excitatory input and local recurrent inhibition - as observed in many biological circuits - is a generic circuit motif for encoding and transmitting time-varying information in recurrent neural circuits.

        ----

        ## [2572] A Unified Analysis of Mixed Sample Data Augmentation: A Loss Function Perspective

        **Authors**: *Chanwoo Park, Sangdoo Yun, Sanghyuk Chun*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e6f32e64b9c27d153b46c94f0fe22b56-Abstract-Conference.html)

        **Abstract**:

        We propose the first unified theoretical analysis of mixed sample data augmentation (MSDA), such as Mixup and CutMix. Our theoretical results show that regardless of the choice of the mixing strategy, MSDA behaves as a pixel-level regularization of the underlying training loss and a regularization of the first layer parameters. Similarly, our theoretical results support that the MSDA training strategy can improve adversarial robustness and generalization compared to the vanilla training strategy. Using the theoretical results, we provide a high-level understanding of how different design choices of MSDA work differently. For example, we show that the most popular MSDA methods, Mixup and CutMix, behave differently, e.g., CutMix regularizes the input gradients by pixel distances, while Mixup regularizes the input gradients regardless of pixel distances. Our theoretical results also show that the optimal MSDA strategy depends on tasks, datasets, or model parameters. From these observations, we propose generalized MSDAs, a Hybrid version of Mixup and CutMix  (HMix) and Gaussian Mixup (GMix), simple extensions of Mixup and CutMix. Our implementation can leverage the advantages of Mixup and CutMix, while our implementation is very efficient, and the computation cost is almost neglectable as Mixup and CutMix. Our empirical study shows that our HMix and GMix outperform the previous state-of-the-art MSDA methods in CIFAR-100 and ImageNet classification tasks.

        ----

        ## [2573] Recursive Reinforcement Learning

        **Authors**: *Ernst Moritz Hahn, Mateo Perez, Sven Schewe, Fabio Somenzi, Ashutosh Trivedi, Dominik Wojtczak*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e6f8759254d86ea9c197d30b92b313ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e6f8759254d86ea9c197d30b92b313ca-Abstract-Conference.html)

        **Abstract**:

        Recursion is the fundamental paradigm to finitely describe potentially infinite objects. As state-of-the-art reinforcement learning (RL) algorithms cannot directly reason about recursion, they must rely on the practitioner's ingenuity in designing a suitable "flat" representation of the environment. The resulting manual feature constructions and approximations are cumbersome and error-prone; their lack of transparency hampers scalability. To overcome these challenges, we develop RL algorithms capable of computing optimal policies in environments described as a collection of Markov decision processes (MDPs) that can recursively invoke one another. Each constituent MDP is characterized by several entry and exit points that correspond to input and output values of these invocations. These recursive MDPs (or RMDPs)  are expressively equivalent to probabilistic pushdown systems (with call-stack playing the role of the pushdown stack), and can model probabilistic programs with recursive procedural calls. We introduce Recursive Q-learning---a model-free RL algorithm for RMDPs---and prove that it converges for finite, single-exit and deterministic multi-exit RMDPs under mild assumptions.

        ----

        ## [2574] Non-Linguistic Supervision for Contrastive Learning of Sentence Embeddings

        **Authors**: *Yiren Jian, Chongyang Gao, Soroush Vosoughi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e708577c4a0802320da036532281bc3b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e708577c4a0802320da036532281bc3b-Abstract-Conference.html)

        **Abstract**:

        Semantic representation learning for sentences is an important and well-studied problem in NLP. The current trend for this task involves training a Transformer-based sentence encoder through a contrastive objective with text, i.e., clustering sentences with semantically similar meanings and scattering others. In this work, we find the performance of Transformer models as sentence encoders can be improved by training with multi-modal multi-task losses, using unpaired examples from another modality (e.g., sentences and unrelated image/audio data). In particular, besides learning by the contrastive loss on text, our model clusters examples from a non-linguistic domain (e.g., visual/audio) with a similar contrastive loss at the same time.  The reliance of our framework on unpaired non-linguistic data makes it language-agnostic, enabling it to be widely applicable beyond English NLP. Experiments on 7 semantic textual similarity benchmarks reveal that models trained with the additional non-linguistic (images/audio) contrastive objective lead to higher quality sentence embeddings. This indicates that Transformer models are able to generalize better by doing a similar task (i.e., clustering) with \textit{unpaired} examples from different modalities in a multi-task fashion. The code is available at https://github.com/yiren-jian/NonLing-CSE.

        ----

        ## [2575] Towards Video Text Visual Question Answering: Benchmark and Baseline

        **Authors**: *Minyi Zhao, Bingjia Li, Jie Wang, Wanqing Li, Wenjing Zhou, Lan Zhang, Shijie Xuyang, Zhihang Yu, Xinkun Yu, Guangze Li, Aobotao Dai, Shuigeng Zhou*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e726197ffd401df4013cd9f81007b5cf-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e726197ffd401df4013cd9f81007b5cf-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        There are already some text-based visual question answering (TextVQA) benchmarks for developing machine's ability to answer questions based on texts in images in recent years. However, models developed on these benchmarks cannot work effectively in many real-life scenarios (e.g. traffic monitoring, shopping ads and e-learning videos) where temporal reasoning ability is required. To this end, we propose a new task named Video Text Visual Question Answering (ViteVQA in short) that aims at answering questions by reasoning texts and visual information spatiotemporally in a given video. In particular, on the one hand, we build the first ViteVQA benchmark dataset named M4-ViteVQA --- the abbreviation of Multi-category Multi-frame Multi-resolution Multi-modal benchmark for ViteVQA, which contains 7,620 video clips of 9 categories (i.e., shopping, traveling, driving, vlog, sport, advertisement, movie, game and talking) and 3 kinds of resolutions (i.e., 720p, 1080p and 1176x664), and 25,123 question-answer pairs. On the other hand, we develop a baseline method named T5-ViteVQA for the ViteVQA task. T5-ViteVQA consists of five transformers. It first extracts optical character recognition (OCR) tokens, question features, and video representations via two OCR transformers, one language transformer and one video-language transformer, respectively. Then, a multimodal fusion transformer and an answer generation module are applied to fuse multimodal information and generate the final prediction. Extensive experiments on M4-ViteVQA demonstrate the superiority of T5-ViteVQA to the existing approaches of TextVQA and VQA tasks. The ViteVQA benchmark is available in https://github.com/bytedance/VTVQA.

        ----

        ## [2576] 4D Unsupervised Object Discovery

        **Authors**: *Yuqi Wang, Yuntao Chen, Zhaoxiang Zhang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7407ab5e89c405d28ff6807ffec594a-Abstract-Conference.html)

        **Abstract**:

        Object discovery is a core task in computer vision. While fast progresses have been made in supervised object detection, its unsupervised counterpart remains largely unexplored. With the growth of data volume, the expensive cost of annotations is the major limitation hindering further study.  Therefore, discovering objects without annotations has great significance. However, this task seems impractical on still-image or point cloud alone due to the lack of discriminative information. Previous studies underlook the crucial temporal information and constraints naturally behind multi-modal inputs. In this paper, we propose 4D unsupervised object discovery, jointly discovering objects from 4D data -- 3D point clouds and 2D RGB images with temporal information. We present the first practical approach for this task by proposing a ClusterNet on 3D point clouds, which is jointly iteratively optimized with a 2D localization network. Extensive experiments on the large-scale Waymo Open Dataset suggest that the localization network and ClusterNet achieve competitive performance on both class-agnostic 2D object detection and 3D instance segmentation, bridging the gap between unsupervised methods and full supervised ones. Codes and models will be made available at https://github.com/Robertwyq/LSMOL.

        ----

        ## [2577] Faster Linear Algebra for Distance Matrices

        **Authors**: *Piotr Indyk, Sandeep Silwal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7599c4b309e39e444a7dcf92572fae1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7599c4b309e39e444a7dcf92572fae1-Abstract-Conference.html)

        **Abstract**:

        The distance matrix of a dataset $X$ of $n$ points with respect to a distance function $f$ represents all pairwise distances between points in $X$ induced by $f$. Due to their wide applicability, distance matrices and related families of matrices have been the focus of many recent algorithmic works. We continue this line of research and take a broad view of algorithm design for distance matrices with the goal of designing fast algorithms, which are specifically tailored for distance matrices, for fundamental linear algebraic primitives. Our results include efficient algorithms for computing matrix-vector products for a wide class of distance matrices, such as the $\ell_1$ metric for which we get a linear runtime, as well as an $\Omega(n^2)$ lower bound for any algorithm which computes a matrix-vector product for the $\ell_{\infty}$ case, showing a separation between the $\ell_1$ and the $\ell_{\infty}$ metrics. Our upper bound results in conjunction with recent works on the matrix-vector query model have many further downstream applications, including the fastest algorithm for computing a relative error low-rank approximation for the distance matrix induced by $\ell_1$ and $\ell_2^2$ functions and the fastest algorithm for computing an additive error low-rank approximation for the $\ell_2$ metric, in addition to applications for fast matrix multiplication among others. We also give algorithms for constructing distance matrices and show that one can construct an approximate $\ell_2$ distance matrix in time faster than the bound implied by the Johnson-Lindenstrauss lemma.

        ----

        ## [2578] Contextual Bandits with Knapsacks for a Conversion Model

        **Authors**: *Zhen Li, Gilles Stoltz*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e75a1c8af8d9438df1057fdaa42913eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e75a1c8af8d9438df1057fdaa42913eb-Abstract-Conference.html)

        **Abstract**:

        We consider contextual bandits with knapsacks, with an underlying structure between rewards generated and cost vectors suffered. We do so motivated by sales with commercial discounts. At each round, given the stochastic i.i.d.\ context $\mathbf{x}_t$ and the arm picked $a_t$ (corresponding, e.g., to a discount level), a customer conversion may be obtained, in which case a reward $r(a,\mathbf{x}_t)$ is gained and vector costs $\mathbf{c}(a_t,\mathbf{x}_t)$ are suffered (corresponding, e.g., to losses of earnings). Otherwise, in the absence of a conversion, the reward and costs are null. The reward and costs achieved are thus coupled through the binary variable measuring conversion or the absence thereof. This underlying structure between rewards and costs is different from the linear structures considered by Agrawal and Devanur [2016] (but we show that the techniques introduced in the present article may also be applied to the case of these linear structures). The adaptive policies exhibited in this article solve at each round a linear program based on upper-confidence estimates of the probabilities of conversion given $a$ and $\mathbf{x}$. This kind of policy is most natural and achieves a regret bound of the typical order $(\mathrm{OPT}/B) \smash{\sqrt{T}}$, where $B$ is the total budget allowed, $\mathrm{OPT}$ is the optimal expected reward achievable by a static policy, and $T$ is the number of rounds.

        ----

        ## [2579] Contrastive Learning as Goal-Conditioned Reinforcement Learning

        **Authors**: *Benjamin Eysenbach, Tianjun Zhang, Sergey Levine, Ruslan Salakhutdinov*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7663e974c4ee7a2b475a4775201ce1f-Abstract-Conference.html)

        **Abstract**:

        In reinforcement learning (RL), it is easier to solve a task if given a good representation. While deep RL should automatically acquire such good representations, prior work often finds that learning representations in an end-to-end fashion is unstable and instead equip RL algorithms with additional representation learning parts (e.g., auxiliary losses, data augmentation). How can we design RL algorithms that directly acquire good representations? In this paper, instead of adding representation learning parts to an existing RL algorithm, we show (contrastive) representation learning methods are already RL algorithms in their own right. To do this, we build upon prior work and apply contrastive representation learning to action-labeled trajectories, in such a way that the (inner product of) learned representations exactly corresponds to a goal-conditioned value function. We use this idea to reinterpret a prior RL method as performing contrastive learning, and then use the idea to propose a much simpler method that achieves similar performance. Across a range of goal-conditioned RL tasks, we demonstrate that contrastive RL methods achieve higher success rates than prior non-contrastive methods. We also show that contrastive RL outperforms prior methods on image-based tasks, without using data augmentation or auxiliary objectives

        ----

        ## [2580] The Pitfalls of Regularization in Off-Policy TD Learning

        **Authors**: *Gaurav Manek, J. Zico Kolter*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e78457d4a04b8565f1fe5077df13cddb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e78457d4a04b8565f1fe5077df13cddb-Abstract-Conference.html)

        **Abstract**:

        Temporal Difference (TD) learning is ubiquitous in reinforcement learning, where it is often combined with off-policy sampling and function approximation.  Unfortunately learning with this combination (known as the deadly triad), exhibits instability and unbounded error.  To account for this, modern Reinforcement Learning methods often implicitly (or sometimes explicitly) assume that regularization is sufficient to mitigate the problem in practice; indeed, the standard deadly triad examples from the literature can be ``fixed'' via proper regularization. In this paper, we introduce a series of new counterexamples to show that the instability and unbounded error of TD methods is not solved by regularization. We demonstrate that, in the off-policy setting with linear function approximation, TD methods can fail to learn a non-trivial value function under any amount of regularization; we further show that regularization can induce divergence under common conditions; and we show that one of the most promising methods to mitigate this divergence (Emphatic TD algorithms) may also diverge under regularization. We further demonstrate such divergence when using neural networks as function approximators.  Thus, we argue that the role of regularization in TD methods needs to be reconsidered, given that it is insufficient to prevent divergence and may itself introduce instability. There needs to be much more care in the practical and theoretical application of regularization to Reinforcement Learning methods.

        ----

        ## [2581] MCMAE: Masked Convolution Meets Masked Autoencoders

        **Authors**: *Peng Gao, Teli Ma, Hongsheng Li, Ziyi Lin, Jifeng Dai, Yu Qiao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7938ede51225b490bb69f7b361a9259-Abstract-Conference.html)

        **Abstract**:

        Vision Transformers (ViT) become widely-adopted architectures for various vision tasks. Masked auto-encoding for feature pretraining and multi-scale hybrid convolution-transformer architectures can further unleash the potentials of ViT, leading to state-of-the-art performances on image classification, detection and semantic segmentation. In this paper, our MCMAE framework demonstrates that multi-scale hybrid convolution-transformer can learn more discriminative representations via the mask auto-encoding scheme. However, directly using the original masking strategy leads to the heavy computational cost and pretraining-finetuning discrepancy. To tackle the issue, we adopt the masked convolution to prevent information leakage in the convolution blocks. A simple block-wise masking strategy is proposed to ensure computational efficiency. We also propose to more directly supervise the multi-scale features of the encoder to boost multi-scale features. Based on our pretrained MCMAE models, MCMAE-Base improves ImageNet-1K finetuning accuracy by 1.4% compared with MAE-Base. On object detection, MCMAE-Base finetuned for only 25 epochs surpasses MAE-Base fined-tuned for 100 epochs by 2.9% box AP and 2.2% mask AP respectively. Code and pretrained models are available at \url{https://github.com/Alpha-VL/ConvMAE}.

        ----

        ## [2582] Robustness to Label Noise Depends on the Shape of the Noise Distribution

        **Authors**: *Diane Oyen, Michal Kucer, Nicolas W. Hengartner, Har Simrat Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7a217c3389b323fe156046ed3aa1e0e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7a217c3389b323fe156046ed3aa1e0e-Abstract-Conference.html)

        **Abstract**:

        Machine learning classifiers have been demonstrated, both empirically and theoretically, to be robust to label noise under certain conditions --- notably the typical assumption is that label noise is independent of the features given the class label. We provide a theoretical framework that generalizes beyond this typical assumption by modeling label noise as a distribution over feature space. We show that both the scale and the \emph{shape} of the noise distribution influence the posterior likelihood; and the shape of the noise distribution has a stronger impact on classification performance if the noise is concentrated in feature space where the decision boundary can be moved. For the special case of uniform label noise (independent of features and the class label), we show that the Bayes optimal classifier for $c$ classes is robust to label noise until the ratio of noisy samples goes above $\frac{c-1}{c}$ (e.g. 90\% for 10 classes), which we call the \emph{tipping point}. However, for the special case of class-dependent label noise (independent of features given the class label), the tipping point can be as low as 50\%. Most importantly, we show that when the noise distribution targets decision boundaries (label noise is directly dependent on feature space), classification robustness can drop off even at a small scale of noise. Even when evaluating recent label-noise mitigation methods we see reduced accuracy when label noise is dependent on features. These findings explain why machine learning often handles label noise well if the noise distribution is uniform in feature-space; yet it also points to the difficulty of overcoming label noise when it is concentrated in a region of feature space where a decision boundary can move.

        ----

        ## [2583] Fast Neural Kernel Embeddings for General Activations

        **Authors**: *Insu Han, Amir Zandieh, Jaehoon Lee, Roman Novak, Lechao Xiao, Amin Karbasi*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7be1f4c6212c24919cd743512477c13-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7be1f4c6212c24919cd743512477c13-Abstract-Conference.html)

        **Abstract**:

        Infinite width limit has shed light on generalization and optimization aspects of deep learning by establishing connections between neural networks and kernel methods. Despite their importance, the utility of these kernel methods was limited in large-scale learning settings due to their (super-)quadratic runtime and memory complexities. Moreover, most prior works on neural kernels have focused on the ReLU activation, mainly due to its popularity but also due to the difficulty of computing such kernels for general activations. In this work, we overcome such difficulties by providing methods to work with general activations. First, we compile and expand the list of activation functions admitting exact dual activation expressions to compute neural kernels. When the exact computation is unknown, we present methods to effectively approximate them. We propose a fast sketching method that approximates any multi-layered Neural Network Gaussian Process (NNGP) kernel and Neural Tangent Kernel (NTK) matrices for a wide range of activation functions, going beyond the commonly analyzed ReLU activation. This is done by showing how to approximate the neural kernels using the truncated Hermite expansion of any desired activation functions. While most prior works require data points on the unit sphere, our methods do not suffer from such limitations and are applicable to any dataset of points in $\mathbb{R}^d$. Furthermore, we provide a subspace embedding for NNGP and NTK matrices with near input-sparsity runtime and near-optimal target dimension which applies to any \emph{homogeneous} dual activation functions with rapidly convergent Taylor expansion. Empirically, with respect to exact convolutional NTK (CNTK) computation, our method achieves $106\times$ speedup for approximate CNTK of a 5-layer Myrtle network on CIFAR-10 dataset.

        ----

        ## [2584] Deep invariant networks with differentiable augmentation layers

        **Authors**: *Cédric Rommel, Thomas Moreau, Alexandre Gramfort*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7d019329e662fe4685be505befca3bb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7d019329e662fe4685be505befca3bb-Abstract-Conference.html)

        **Abstract**:

        Designing learning systems which are invariant to certain data transformations is critical in machine learning. Practitioners can typically enforce a desired invariance on the trained model through the choice of a network architecture, e.g. using convolutions for translations, or using data augmentation. Yet, enforcing true invariance in the network can be difficult, and data invariances are not always known a piori. State-of-the-art methods for learning data augmentation policies require held-out data and are based on bilevel optimization problems, which are complex to solve and often computationally demanding. In this work we investigate new ways of learning invariances only from the training data. Using learnable augmentation layers built directly in the network, we demonstrate that our method is very versatile. It can incorporate any type of differentiable augmentation and be applied to a broad class of learning problems beyond computer vision. We provide empirical evidence showing that our approach is easier and faster to train than modern automatic data augmentation techniques based on bilevel optimization, while achieving comparable results. Experiments show that while the invariances transferred to a model through automatic data augmentation are limited by the model expressivity, the invariance yielded by our approach is insensitive to it by design.

        ----

        ## [2585] Factorized-FL: Personalized Federated Learning with Parameter Factorization & Similarity Matching

        **Authors**: *Wonyong Jeong, Sung Ju Hwang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e7feb9dbd9a94b6c552fc403fcebf2ef-Abstract-Conference.html)

        **Abstract**:

        In real-world federated learning scenarios, participants could have their own personalized labels incompatible with those from other clients, due to using different label permutations or tackling completely different tasks or domains. However, most existing FL approaches cannot effectively tackle such extremely heterogeneous scenarios since they often assume that (1) all participants use a synchronized set of labels, and (2) they train on the same tasks from the same domain. In this work, to tackle these challenges, we introduce Factorized-FL, which allows to effectively tackle label- and task-heterogeneous federated learning settings by factorizing the model parameters into a pair of rank-1 vectors, where one captures the common knowledge across different labels and tasks and the other captures knowledge specific to the task for each local model. Moreover, based on the distance in the client-specific vector space, Factorized-FL performs a selective aggregation scheme to utilize only the knowledge from the relevant participants for each client. We extensively validate our method on both label- and domain-heterogeneous settings, on which it outperforms the state-of-the-art personalized federated learning methods. The code is available at https://github.com/wyjeong/Factorized-FL.

        ----

        ## [2586] Reinforcement Learning with a Terminator

        **Authors**: *Guy Tennenholtz, Nadav Merlis, Lior Shani, Shie Mannor, Uri Shalit, Gal Chechik, Assaf Hallak, Gal Dalal*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e83b86156555ab9692743f9f8f67adf1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e83b86156555ab9692743f9f8f67adf1-Abstract-Conference.html)

        **Abstract**:

        We present the problem of reinforcement learning with exogenous termination. We define the Termination Markov Decision Process (TerMDP), an extension of the MDP framework, in which episodes may be interrupted by an external non-Markovian observer. This formulation accounts for numerous real-world situations, such as a human interrupting an autonomous driving agent for reasons of discomfort. We learn the parameters of the TerMDP and leverage the structure of the estimation problem to provide state-wise confidence bounds. We use these to construct a provably-efficient algorithm, which accounts for termination, and bound its regret. Motivated by our theoretical analysis, we design and implement a scalable approach, which combines optimism (w.r.t. termination) and a dynamic discount factor, incorporating the termination probability. We deploy our method on high-dimensional driving and MinAtar benchmarks. Additionally, we test our approach on human data in a driving setting. Our results demonstrate fast convergence and significant improvement over various baseline approaches.

        ----

        ## [2587] How Transferable are Video Representations Based on Synthetic Data?

        **Authors**: *Yo-whan Kim, Samarth Mishra, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Kate Saenko, Aude Oliva, Rogério Feris*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8507db80464ced5658d16b49bd458b9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8507db80464ced5658d16b49bd458b9-Abstract-Datasets_and_Benchmarks.html)

        **Abstract**:

        Action recognition has improved dramatically with massive-scale video datasets. Yet, these datasets are accompanied with issues related to curation cost, privacy, ethics, bias, and copyright. Compared to that, only minor efforts have been devoted toward exploring the potential of synthetic video data. In this work, as a stepping stone towards addressing these shortcomings, we study the transferability of video representations learned solely from synthetically-generated video clips, instead of real data. We propose SynAPT, a novel benchmark for action recognition based on a combination of existing synthetic datasets, in which a model is pre-trained on synthetic videos rendered by various graphics simulators, and then transferred to a set of downstream action recognition datasets, containing different categories than the synthetic data. We provide an extensive baseline analysis on SynAPT revealing that the simulation-to-real gap is minor for datasets with low object and scene bias, where models pre-trained with synthetic data even outperform their real data counterparts. We posit that the gap between real and synthetic action representations can be attributed to contextual bias and static objects related to the action, instead of the temporal dynamics of the action itself. The SynAPT benchmark is available at https://github.com/mintjohnkim/SynAPT.

        ----

        ## [2588] Sub-exponential time Sum-of-Squares lower bounds for Principal Components Analysis

        **Authors**: *Aaron Potechin, Goutham Rajendran*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e85454a113e8b41e017c81875ae68d47-Abstract-Conference.html)

        **Abstract**:

        Principal Components Analysis (PCA) is a dimension-reduction technique widely used in machine learning and statistics. However, due to the dependence of the principal components on all the dimensions, the components are notoriously hard to interpret. Therefore, a variant known as sparse PCA is often preferred. Sparse PCA learns principal components of the data but enforces that such components must be sparse. This has applications in diverse fields such as computational biology and image processing. To learn sparse principal components, it's well known that standard PCA will not work, especially in high dimensions, and therefore algorithms for sparse PCA are often studied as a separate endeavor. Various algorithms have been proposed for Sparse PCA over the years, but given how fundamental it is for applications in science, the limits of efficient algorithms are only partially understood. In this work, we study the limits of the powerful Sum of Squares (SoS) family of algorithms for Sparse PCA. SoS algorithms have recently revolutionized robust statistics, leading to breakthrough algorithms for long-standing open problems in machine learning, such as optimally learning mixtures of gaussians, robust clustering, robust regression, etc. Moreover, it is believed to be the optimal robust algorithm for many statistical problems. Therefore, for sparse PCA, it's plausible that it can beat simpler algorithms such as diagonal thresholding that have been traditionally used. In this work, we show that this is not the case, by exhibiting strong tradeoffs between the number of samples required, the sparsity and the ambient dimension, for which SoS algorithms, even if allowed sub-exponential time, will fail to optimally recover the component. Our results are complemented by known algorithms in literature, thereby painting an almost complete picture of the behavior of efficient algorithms for sparse PCA. Since SoS algorithms encapsulate many algorithmic techniques such as spectral or statistical query algorithms, this solidifies the message that  known algorithms are optimal for sparse PCA. Moreover, our techniques are strong enough to obtain similar tradeoffs for Tensor PCA, another important higher order variant of PCA with applications in topic modeling, video processing, etc.

        ----

        ## [2589] A Multilabel Classification Framework for Approximate Nearest Neighbor Search

        **Authors**: *Ville Hyvönen, Elias Jääsaari, Teemu Roos*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8752f3e51f33a2e06daf044c40ce412-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8752f3e51f33a2e06daf044c40ce412-Abstract-Conference.html)

        **Abstract**:

        Both supervised and unsupervised machine learning algorithms have been used to learn partition-based index structures for approximate nearest neighbor (ANN) search. Existing supervised algorithms formulate the learning task as finding a partition in which the nearest neighbors of a training set point belong to the same partition element as the point itself, so that the nearest neighbor candidates can be retrieved by naive lookup or backtracking search. We formulate candidate set selection in ANN search directly as a multilabel classification problem where the labels correspond to the nearest neighbors of the query point, and interpret the partitions as partitioning classifiers for solving this task. Empirical results suggest that the natural classifier based on this interpretation leads to strictly improved performance when combined with any unsupervised or supervised partitioning strategy. We also prove a sufficient condition for consistency of a partitioning classifier for ANN search, and illustrate the result by verifying this condition for chronological $k$-d trees.

        ----

        ## [2590] Learnable Polyphase Sampling for Shift Invariant and Equivariant Convolutional Networks

        **Authors**: *Renan A. Rojas-Gomez, Teck-Yian Lim, Alexander G. Schwing, Minh N. Do, Raymond A. Yeh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e87b1e06be8c3594c810e8991e77ea40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e87b1e06be8c3594c810e8991e77ea40-Abstract-Conference.html)

        **Abstract**:

        We propose learnable polyphase sampling (LPS), a pair of learnable down/upsampling layers that enable truly shift-invariant and equivariant convolutional networks. LPS can be trained end-to-end from data and generalizes existing handcrafted downsampling layers. It is widely applicable as it can be integrated into any convolutional network by replacing down/upsampling layers. We evaluate LPS on image classification and semantic segmentation. Experiments show that LPS is on-par with or outperforms existing methods in both performance and shift consistency. For the first time, we achieve true shift-equivariance on semantic segmentation (PASCAL VOC), i.e., 100% shift consistency, outperforming baselines by an absolute 3.3%.

        ----

        ## [2591] Not All Bits have Equal Value: Heterogeneous Precisions via Trainable Noise

        **Authors**: *Pedro Savarese, Xin Yuan, Yanjing Li, Michael Maire*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e894de44f7587d5ea723120f4d0b8689-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e894de44f7587d5ea723120f4d0b8689-Abstract-Conference.html)

        **Abstract**:

        We study the problem of training deep networks while quantizing parameters and activations into low-precision numeric representations, a setting central to reducing energy consumption and inference time of deployed models. We propose a method that learns different precisions, as measured by bits in numeric representations, for different weights in a neural network, yielding a heterogeneous allocation of bits across parameters. Learning precisions occurs alongside learning weight values, using a strategy derived from a novel framework wherein the intractability of optimizing discrete precisions is approximated by training per-parameter noise magnitudes. We broaden this framework to also encompass learning precisions for hidden state activations, simultaneously with weight precisions and values.  Our approach exposes the objective of constructing a low-precision inference-efficient model to the entirety of the training process. Experiments show that it finds highly heterogeneous precision assignments for CNNs trained on CIFAR and ImageNet, improving upon previous state-of-the-art quantization methods.  Our improvements extend to the challenging scenario of learning reduced-precision GANs.

        ----

        ## [2592] Tensor Program Optimization with Probabilistic Programs

        **Authors**: *Junru Shao, Xiyou Zhou, Siyuan Feng, Bohan Hou, Ruihang Lai, Hongyi Jin, Wuwei Lin, Masahiro Masuda, Cody Hao Yu, Tianqi Chen*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e894eafae43e68b4c8dfdacf742bcbf3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e894eafae43e68b4c8dfdacf742bcbf3-Abstract-Conference.html)

        **Abstract**:

        Automatic optimization for tensor programs becomes increasingly important as we deploy deep learning in various environments, and efficient optimization relies on a rich search space and effective search. Most existing efforts adopt a search space which lacks the ability to efficiently enable domain experts to grow the search space. This paper introduces MetaSchedule, a domain-specific probabilistic programming language abstraction to construct a rich search space of tensor programs. Our abstraction allows domain experts to analyze the program, and easily propose stochastic choices in a modular way to compose program transformation accordingly. We also build an end-to-end learning-driven framework to find an optimized program for a given search space. Experimental results show that MetaSchedule can cover the search space used in the state-of-the-art tensor program optimization frameworks in a modular way. Additionally, it empowers domain experts to conveniently grow the search space and modularly enhance the system, which brings 48% speedup on end-to-end deep learning workloads.

        ----

        ## [2593] Deep Generative Model for Periodic Graphs

        **Authors**: *Shiyu Wang, Xiaojie Guo, Liang Zhao*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e89e8f84626197942b36a82e524c2529-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e89e8f84626197942b36a82e524c2529-Abstract-Conference.html)

        **Abstract**:

        Periodic graphs are graphs consisting of repetitive local structures, such as crystal nets and polygon mesh. Their generative modeling has great potential in real-world applications such as material design and graphics synthesis. Classical models either rely on domain-specific predefined generation principles (e.g., in crystal net design), or follow geometry-based prescribed rules. Recently, deep generative models have shown great promise in automatically generating general graphs. However, their advancement into periodic graphs has not been well explored due to several key challenges in 1) maintaining graph periodicity; 2) disentangling local and global patterns; and 3) efficiency in learning repetitive patterns. To address them, this paper proposes Periodical-Graph Disentangled Variational Auto-encoder (PGD-VAE), a new deep generative model for periodic graphs that can automatically learn, disentangle, and generate local and global graph patterns. Specifically, we develop a new periodic graph encoder consisting of global-pattern encoder and local-pattern encoder that ensures to disentangle the representation into global and local semantics. We then propose a new periodic graph decoder consisting of local structure decoder, neighborhood decoder, and global structure decoder, as well as the assembler of their outputs that guarantees periodicity. Moreover, we design a new model learning objective that helps ensure the invariance of local-semantic representations for the graphs with the same local structure. Comprehensive experimental evaluations have been conducted to demonstrate the effectiveness of the proposed method.

        ----

        ## [2594] Exploring the Limits of Domain-Adaptive Training for Detoxifying Large-Scale Language Models

        **Authors**: *Boxin Wang, Wei Ping, Chaowei Xiao, Peng Xu, Mostofa Patwary, Mohammad Shoeybi, Bo Li, Anima Anandkumar, Bryan Catanzaro*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8c20cafe841cba3e31a17488dc9c3f1-Abstract-Conference.html)

        **Abstract**:

        Pre-trained language models (LMs) are shown to easily generate toxic language. In this work, we systematically explore domain-adaptive training to reduce the toxicity of language models. We conduct this study on three dimensions: training corpus, model size, and parameter efficiency. For the training corpus, we demonstrate that using self-generated datasets consistently outperforms the existing baselines across various model sizes on both automatic and human evaluations, even when it uses a 3 1 smaller training corpus. We then comprehensively study detoxifying LMs with parameter sizes ranging from 126M up to 530B (3Ã— larger than GPT3), a scale that has never been studied before. We find that i) large LMs have similar toxicity levels as smaller ones given the same pre-training corpus, and ii) large LMs require more endeavor to unlearn the toxic content seen at pretraining. We also explore parameter-efficient training methods for detoxification. We demonstrate that adding and training adapter-only layers in LMs not only saves a lot of parameters but also achieves a better trade-off between toxicity and perplexity than whole model adaptation for large-scale models. Our code will be available at: https://github.com/NVIDIA/Megatron-LM/.

        ----

        ## [2595] Relational Reasoning via Set Transformers: Provable Efficiency and Applications to MARL

        **Authors**: *Fengzhuo Zhang, Boyi Liu, Kaixin Wang, Vincent Y. F. Tan, Zhuoran Yang, Zhaoran Wang*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8da56eb93676e8f60ed2b696e44e7dc-Abstract-Conference.html)

        **Abstract**:

        The cooperative Multi-Agent Reinforcement Learning (MARL) with permutation invariant agents framework has achieved tremendous empirical successes in real-world applications. Unfortunately, the theoretical understanding of this MARL problem is lacking due to the curse of many agents and the limited exploration of the relational reasoning in existing works. In this paper, we verify that the transformer implements complex relational reasoning, and we propose and analyze model-free and model-based offline MARL algorithms with the transformer approximators. We prove that the suboptimality gaps of the model-free and model-based algorithms are independent of and logarithmic in the number of agents respectively, which mitigates the curse of many agents. These results are consequences of a  novel generalization error bound of the transformer and a novel analysis of the Maximum Likelihood Estimate (MLE) of the system dynamics with the transformer. Our model-based algorithm is the first provably efficient MARL algorithm that explicitly exploits the permutation invariance of the agents. Our improved generalization bound may be of independent interest and is applicable  to other regression problems related to the transformer beyond MARL.

        ----

        ## [2596] Missing Data Imputation and Acquisition with Deep Hierarchical Models and Hamiltonian Monte Carlo

        **Authors**: *Ignacio Peis, Chao Ma, José Miguel Hernández-Lobato*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8dbeb1c947a30576c699e7f5c73d3e3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8dbeb1c947a30576c699e7f5c73d3e3-Abstract-Conference.html)

        **Abstract**:

        Variational Autoencoders (VAEs) have recently been highly successful at imputing and acquiring heterogeneous missing data. However, within this specific application domain, existing VAE methods are restricted by using only one layer of latent variables and strictly Gaussian posterior approximations. To address these limitations, we present HH-VAEM, a Hierarchical VAE model for mixed-type incomplete data that uses Hamiltonian Monte Carlo with automatic hyper-parameter tuning for improved approximate inference. Our experiments show that HH-VAEM outperforms existing baselines in the tasks of missing data imputation and supervised learning with missing features. Finally, we also present a sampling-based approach for efficiently computing the information gain when missing features are to be acquired with HH-VAEM. Our experiments show that this sampling-based approach is superior to alternatives based on Gaussian approximations.

        ----

        ## [2597] Score-Based Generative Models Detect Manifolds

        **Authors**: *Jakiw Pidstrigach*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e8fb575e3ede31f9b8c05d53514eb7c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e8fb575e3ede31f9b8c05d53514eb7c6-Abstract-Conference.html)

        **Abstract**:

        Score-based generative models (SGMs) need to approximate the scores $\nabla \log p_t$ of the intermediate distributions as well as the final distribution $p_T$ of the forward process. The theoretical underpinnings of the effects of these approximations are still lacking. We find precise conditions under which SGMs are able to produce samples from an underlying (low-dimensional) data manifold $\mathcal{M}$. This assures us that SGMs are able to generate the "right kind of samples". For example, taking $\mathcal{M}$ to be the subset of images of faces, we provide conditions under which the SGM robustly produces an image of a face, even though the relative frequencies of these images might not accurately represent the true data generating distribution. Moreover, this analysis is a first step towards understanding the generalization properties of SGMs: Taking $\mathcal{M}$ to be the set of all training samples, our results provide a precise description of when the SGM memorizes its training data.

        ----

        ## [2598] Distributionally Robust Optimization via Ball Oracle Acceleration

        **Authors**: *Yair Carmon, Danielle Hausler*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e90b00adc3ba130eb2510d93ba3ff250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e90b00adc3ba130eb2510d93ba3ff250-Abstract-Conference.html)

        **Abstract**:

        We develop and analyze algorithms for distributionally robust optimization (DRO) of convex losses. In particular, we consider group-structured and bounded $f$-divergence uncertainty sets. Our approach relies on an accelerated method that queries a ball optimization oracle, i.e., a subroutine that minimizes the objective within a small ball around the query point. Our main contribution is efficient implementations of this oracle for DRO objectives. For DRO with $N$ non-smooth loss functions, the resulting algorithms find an $\epsilon$-accurate solution with  $\widetilde{O}\left(N\epsilon^{-2/3} + \epsilon^{-2}\right)$ first-order oracle queries to individual loss functions. Compared to existing algorithms for this problem, we improve complexity by a factor of up to $\epsilon^{-4/3}$.

        ----

        ## [2599] Palm up: Playing in the Latent Manifold for Unsupervised Pretraining

        **Authors**: *Hao Liu, Tom Zahavy, Volodymyr Mnih, Satinder Singh*

        **Conference**: *nips 2022*

        **URL**: [http://papers.nips.cc/paper_files/paper/2022/hash/e92381dba235a8309f08ce46376189a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/e92381dba235a8309f08ce46376189a9-Abstract-Conference.html)

        **Abstract**:

        Large and diverse datasets have been the cornerstones of many impressive advancements in artificial intelligence. Intelligent creatures, however, learn by interacting with the environment, which changes the input sensory signals and the state of the environment. In this work, we aim to bring the best of both worlds and propose an algorithm that exhibits an  exploratory behavior whilst it utilizes large diverse datasets. Our key idea is to leverage deep generative models that are pretrained on static datasets and introduce a dynamic model in the latent space. The transition dynamics simply mixes an action and a random sampled latent. It then applies an exponential moving average for temporal persistency, the resulting latent is decoded to image using pretrained generator. We then employ an unsupervised reinforcement learning algorithm to explore in this environment and perform unsupervised representation learning on the collected data. We further leverage the temporal information of this data to pair data points as a natural supervision for representation learning. Our experiments suggest that the learned representations can be successfully transferred to downstream tasks in both vision and reinforcement learning domains.

        ----

        

[Go to the previous page](NIPS-2022-list12.md)

[Go to the next page](NIPS-2022-list14.md)

[Go to the catalog section](README.md)