## [2800] Generator Identification for Linear SDEs with Additive and Multiplicative Noise

**Authors**: *Yuanyuan Wang, Xi Geng, Wei Huang, Biwei Huang, Mingming Gong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca642f8e1174012d67c05c1c9f969644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca642f8e1174012d67c05c1c9f969644-Abstract-Conference.html)

**Abstract**:

In this paper, we present conditions for identifying the generator of a linear stochastic differential equation (SDE) from the distribution of its solution process with a given fixed initial state. These identifiability conditions are crucial in causal inference using linear SDEs as they enable the identification of the post-intervention distributions from its observational distribution. Specifically, we derive a sufficient and necessary condition for identifying the generator of linear SDEs with additive noise, as well as a sufficient condition for identifying the generator of linear SDEs with multiplicative noise. We show that the conditions derived for both types of SDEs are generic. Moreover, we offer geometric interpretations of the derived identifiability conditions to enhance their understanding. To validate our theoretical results, we perform a series of simulations, which support and substantiate the established findings.

----

## [2801] Post-processing Private Synthetic Data for Improving Utility on Selected Measures

**Authors**: *Hao Wang, Shivchander Sudalairaj, John Henning, Kristjan H. Greenewald, Akash Srivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca6980a3dba7fb3e4e66925656dba68b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca6980a3dba7fb3e4e66925656dba68b-Abstract-Conference.html)

**Abstract**:

Existing private synthetic data generation algorithms are agnostic to downstream tasks. However, end users may have specific requirements that the synthetic data must satisfy. Failure to meet these requirements could significantly reduce the utility of the data for downstream use. We introduce a post-processing technique that improves the utility of the synthetic data with respect to measures selected by the end user, while preserving strong privacy guarantees and dataset quality. Our technique involves resampling from the synthetic data to filter out samples that do not meet the selected utility measures, using an efficient stochastic first-order algorithm to find optimal resampling weights. Through comprehensive numerical experiments, we demonstrate that our approach consistently improves the utility of synthetic data across multiple benchmark datasets and state-of-the-art synthetic data generation algorithms.

----

## [2802] A Bayesian Approach To Analysing Training Data Attribution In Deep Learning

**Authors**: *Elisa Nguyen, Minjoon Seo, Seong Joon Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html)

**Abstract**:

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable to deep models in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining deep model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap. Code is provided at https://github.com/ElisaNguyen/bayesian-tda.

----

## [2803] GPEX, A Framework For Interpreting Artificial Neural Networks

**Authors**: *Amir Akbarnejad, Gilbert Bigras, Nilanjan Ray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca8c6f28d8ba1e732e3f217ab05c4ec0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca8c6f28d8ba1e732e3f217ab05c4ec0-Abstract-Conference.html)

**Abstract**:

The analogy between Gaussian processes (GPs) and deep artificial neural networks (ANNs) has received a lot of interest, and has shown promise to unbox the blackbox of deep ANNs. Existing theoretical works put strict assumptions on the ANN (e.g. requiring all intermediate layers to be wide, or using specific activation functions). Accommodating those theoretical assumptions is hard in recent deep architectures, and those theoretical conditions need refinement as new deep architectures emerge. In this paper we derive an evidence lower-bound that encourages the GP's posterior to match the ANN's output without any requirement on the ANN. Using our method we find out that on 5 datasets, only a subset of those theoretical assumptions are sufficient. Indeed, in our experiments we used  a normal ResNet-18 or feed-forward backbone with a single wide layer in the end. One limitation of training GPs is the lack of scalability with respect to the number of inducing points. We use novel computational techniques that allow us to train GPs with hundreds of thousands of inducing points and with GPU acceleration. As shown in our experiments, doing so has been essential to get a close match between the GPs and the ANNs on 5 datasets. We implement our method as a publicly available tool called GPEX: https://github.com/amirakbarnejad/gpex. On 5 datasets (4 image datasets, and 1 biological dataset) and ANNs with 2 types of functionality (classifier or attention-mechanism) we were able to find GPs whose outputs closely match those of the corresponding ANNs. After matching the GPs to the ANNs, we used the GPs' kernel functions to explain the ANNs' decisions. We provide more than 200 explanations (around 30 in the paper and the rest in the supplementary) which are highly interpretable by humans and show the ability of the obtained GPs to unbox the ANNs' decisions.

----

## [2804] On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training

**Authors**: *Jieyu Zhang, Bohan Wang, Zhengyu Hu, Pang Wei Koh, Alexander J. Ratner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca9567d8ef6b2ea2da0d7eed57b933ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca9567d8ef6b2ea2da0d7eed57b933ee-Abstract-Conference.html)

**Abstract**:

Pre-training datasets are critical for building state-of-the-art machine learning models, motivating rigorous study on their impact on downstream tasks. In this work, we study the impact of the trade-off between the intra-class diversity (the number of samples per class) and the inter-class diversity (the number of classes) of a supervised pre-training dataset. Empirically, we found that with the size of the pre-training dataset fixed, the best downstream performance comes with a balance on the intra-/inter-class diversity. To understand the underlying mechanism, we show theoretically that the downstream performance depends monotonically on both types of diversity. Notably, our theory reveals that the optimal class-to-sample ratio (#classes / #samples per class) is invariant to the size of the pre-training dataset, which motivates an application of predicting the optimal number of pre-training classes. We demonstrate the effectiveness of this application by an improvement of around 2 points on the downstream tasks when using ImageNet as the pre-training dataset.

----

## [2805] An information-theoretic quantification of the content of communication between brain regions

**Authors**: *Marco Celotto, Jan Bím, Alejandro Tlaie, Vito De Feo, Alessandro Toso, Stefan Lemke, Daniel Chicharro, Hamed Nili, Malte Bieler, Ileana L. Hanganu-Opatz, Tobias Donner, Andrea Brovelli, Stefano Panzeri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca9eaef07eca2a50fc626cb929617b1c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca9eaef07eca2a50fc626cb929617b1c-Abstract-Conference.html)

**Abstract**:

Quantifying the amount, content and direction of communication between brain regions is key to understanding brain function. Traditional methods to analyze brain activity based on the Wiener-Granger causality principle quantify the overall information propagated by neural activity between simultaneously recorded brain regions, but do not reveal the information flow about specific features of interest (such as sensory stimuli). Here, we develop a new information theoretic measure termed Feature-specific Information Transfer (FIT), quantifying how much information about a specific feature flows between two regions. FIT merges the Wiener-Granger causality principle with information-content specificity. We first derive FIT and prove analytically its key properties. We then illustrate and test them with simulations of neural activity, demonstrating that FIT identifies, within the total information propagated between regions, the information that is transmitted about specific features. We then analyze three neural datasets obtained with different recording methods, magneto- and electro-encephalography, and spiking activity, to demonstrate the ability of FIT to uncover the content and direction of information flow between brain regions beyond what can be discerned with traditional analytical methods. FIT can improve our understanding of how brain regions communicate by uncovering previously unaddressed feature-specific information flow.

----

## [2806] Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models

**Authors**: *Anant Raj, Umut Simsekli, Alessandro Rudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html)

**Abstract**:

This paper deals with the problem of efficient sampling from a stochastic differential equation, given the drift function and the diffusion matrix. The proposed approach leverages a recent model for probabilities (Rudi and Ciliberto, 2021) (the positive semi-definite -- PSD model) from which it is possible to obtain independent and identically distributed (i.i.d.) samples at precision $\varepsilon$ with a cost that is $m^2 d \log(1/\varepsilon)$ where $m$ is the dimension of the model, $d$ the dimension of the space. The proposed approach consists in: first, computing the PSD model that satisfies the Fokker-Planck equation (or its fractional variant) associated with the SDE, up to error $\varepsilon$, and then sampling from the resulting PSD model. Assuming some regularity of the Fokker-Planck solution (i.e. $\beta$-times differentiability plus some geometric condition on its zeros) We obtain an algorithm that: (a) in the preparatory phase obtains a PSD model with L2 distance $\varepsilon$ from the solution of the equation, with a model of dimension $m = \varepsilon^{-(d+1)/(\beta-2s)} (\log(1/\varepsilon))^{d+1}$ where $1/2\leq s\leq1$ is the fractional power to the Laplacian, and total computational complexity of $O(m^{3.5} \log(1/\varepsilon))$ and then (b) for Fokker-Planck equation, it is able to produce i.i.d.\ samples with error $\varepsilon$ in Wasserstein-1 distance, with a cost that is $O(d \varepsilon^{-2(d+1)/\beta-2} \log(1/\varepsilon)^{2d+3})$ per sample. This means that, if the probability associated with the SDE is somewhat regular, i.e. $\beta \geq 4d+2$, then the algorithm requires $O(\varepsilon^{-0.88} \log(1/\varepsilon)^{4.5d})$ in the preparatory phase, and $O(\varepsilon^{-1/2}\log(1/\varepsilon)^{2d+2})$ for each sample. Our results suggest that as the true solution gets smoother, we can circumvent the curse of dimensionality without requiring any sort of convexity.

----

## [2807] TopoSRL: Topology preserving self-supervised Simplicial Representation Learning

**Authors**: *Hiren Madhu, Sundeep Prabhakar Chepuri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/caba69fbc9fa0b06241b98a44cab8b31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/caba69fbc9fa0b06241b98a44cab8b31-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce $\texttt{TopoSRL}$, a novel self-supervised learning (SSL) method for simplicial complexes to effectively capture higher-order interactions and preserve topology in the learned representations. $\texttt{TopoSRL}$ addresses the limitations of existing graph-based SSL methods that typically concentrate on pairwise relationships, neglecting long-range dependencies crucial to capture topological information. We propose a new simplicial augmentation technique that generates two views of the simplicial complex that enriches the representations while being efficient. Next, we propose a new simplicial contrastive loss function that contrasts the generated simplices to preserve local and global information present in the simplicial complexes. Extensive experimental results demonstrate the superior performance of $\texttt{TopoSRL}$ compared to state-of-the-art graph SSL techniques and supervised simplicial neural models across various datasets corroborating the efficacy of $\texttt{TopoSRL}$ in processing simplicial complex data in a self-supervised setting.

----

## [2808] Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

**Authors**: *Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Robotic perception requires the modeling of both 3D geometry and semantics. Existing methods typically focus on estimating 3D bounding boxes, neglecting finer geometric details and struggling to handle general, out-of-vocabulary objects. 3D occupancy prediction, which estimates the detailed occupancy states and semantics of a scene, is an emerging task to overcome these limitations.To support 3D occupancy prediction, we develop a label generation pipeline that produces dense, visibility-aware labels for any given scene. This pipeline comprises three stages: voxel densification, occlusion reasoning, and image-guided voxel refinement. We establish two benchmarks, derived from the Waymo Open Dataset and the nuScenes Dataset, namely Occ3D-Waymo and Occ3D-nuScenes benchmarks. Furthermore, we provide an extensive analysis of the proposed dataset with various baseline models. Lastly, we propose a new model, dubbed Coarse-to-Fine Occupancy (CTF-Occ) network, which demonstrates superior performance on the Occ3D benchmarks.The code, data, and benchmarks are released at \url{https://tsinghua-mars-lab.github.io/Occ3D/}.

----

## [2809] ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design

**Authors**: *Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Han Spinner, Nathan J. Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, Debora S. Marks*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Predicting the effects of mutations in proteins is critical to many applications, from understanding genetic disease to designing novel proteins to address our most pressing challenges in climate, agriculture and healthcare. Despite an increase in machine learning-based protein modeling methods, assessing their effectiveness is problematic due to the use of distinct, often contrived, experimental datasets and variable performance across different protein families. Addressing these challenges requires scale. To that end we introduce ProteinGym v1.0, a large-scale and holistic set of benchmarks specifically designed for protein fitness prediction and design. It encompasses both a broad collection of over 250 standardized deep mutational scanning assays, spanning millions of mutated sequences, as well as curated clinical datasets providing high-quality expert annotations about mutation effects. We devise a robust evaluation framework that combines metrics for both fitness prediction and design, factors in known limitations of the underlying experimental methods, and covers both zero-shot and supervised settings. We report the performance of a diverse set of over 40 high-performing models from various subfields (eg., mutation effects, inverse folding) into a unified benchmark. We open source the corresponding codebase, datasets, MSAs, structures, predictions and develop a user-friendly website that facilitates comparisons across all settings.

----

## [2810] On the spectral bias of two-layer linear networks

**Authors**: *Aditya Vardhan Varre, Maria-Luiza Vladarean, Loucas Pillaud-Vivien, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cad2fd66cf88226d868f90a7cbaa4a53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cad2fd66cf88226d868f90a7cbaa4a53-Abstract-Conference.html)

**Abstract**:

This paper studies the behaviour of two-layer fully connected networks with linear activations trained with gradient flow on the square loss. We show how the optimization process carries an implicit bias on the parameters that depends on the scale of its initialization.  The main result of the paper is a variational characterization of the loss minimizers retrieved by the gradient flow for a specific initialization shape. This characterization reveals that, in the small scale initialization regime, the linear neural network's hidden layer is biased toward having a low-rank structure. To complement our results, we showcase a hidden mirror flow that tracks the dynamics of the singular values of the weights matrices and describe their time evolution. We support our findings with numerical experiments illustrating the phenomena.

----

## [2811] GMSF: Global Matching Scene Flow

**Authors**: *Yushan Zhang, Johan Edstedt, Bastian Wandt, Per-Erik Forssén, Maria Magnusson, Michael Felsberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)

**Abstract**:

We tackle the task of scene flow estimation from point clouds. Given a source and a target point cloud, the objective is to estimate a translation from each point in the source point cloud to the target, resulting in a 3D motion vector field. Previous dominant scene flow estimation methods require complicated coarse-to-fine or recurrent architectures as a multi-stage refinement. In contrast, we propose a significantly simpler single-scale one-shot global matching to address the problem. Our key finding is that reliable feature similarity between point pairs is essential and sufficient to estimate accurate scene flow. We thus propose to decompose the feature extraction step via a hybrid local-global-cross transformer architecture which is crucial to accurate and robust feature representations. Extensive experiments show that the proposed Global Matching Scene Flow (GMSF) sets a new state-of-the-art on multiple scene flow estimation benchmarks. On FlyingThings3D, with the presence of occlusion points, GMSF reduces the outlier percentage from the previous best performance of 27.4% to 5.6%. On KITTI Scene Flow, without any fine-tuning, our proposed method shows state-of-the-art performance. On the Waymo-Open dataset, the proposed method outperforms previous methods by a large margin. The code is available at https://github.com/ZhangYushan3/GMSF.

----

## [2812] Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks

**Authors**: *Ziyi Huang, Henry Lam, Haofeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb2266111eadcfa2c02187ace64e2183-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb2266111eadcfa2c02187ace64e2183-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification (UQ) is important for reliability assessment and enhancement of machine learning models. In deep learning, uncertainties arise not only from data, but also from the training procedure that often injects substantial noises and biases. These hinder the attainment of statistical guarantees and, moreover, impose computational challenges on UQ due to the need for repeated network retraining. Building upon the recent neural tangent kernel theory, we create statistically guaranteed schemes to principally \emph{characterize}, and \emph{remove}, the uncertainty of over-parameterized neural networks with very low computation effort. In particular, our approach, based on what we call a procedural-noise-correcting (PNC) predictor, removes the procedural uncertainty by using only \emph{one} auxiliary network that is trained on a suitably labeled dataset, instead of many retrained networks employed in deep ensembles. Moreover, by combining our PNC predictor with suitable light-computation resampling methods, we build several approaches to construct asymptotically exact-coverage confidence intervals using as low as four trained networks without additional overheads.

----

## [2813] LuminAIRe: Illumination-Aware Conditional Image Repainting for Lighting-Realistic Generation

**Authors**: *Jiajun Tang, Haofeng Zhong, Shuchen Weng, Boxin Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb3658b9983f677670a246c46ece553d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb3658b9983f677670a246c46ece553d-Abstract-Conference.html)

**Abstract**:

We present the ilLumination-Aware conditional Image Repainting (LuminAIRe) task to address the unrealistic lighting effects in recent conditional image repainting (CIR) methods. The environment lighting and 3D geometry conditions are explicitly estimated from given background images and parsing masks using a parametric lighting representation and learning-based priors. These 3D conditions are then converted into illumination images through the proposed physically-based illumination rendering and illumination attention module. With the injection of illumination images, physically-correct lighting information is fed into the lighting-realistic generation process and repainted images with harmonized lighting effects in both foreground and background regions can be acquired, whose superiority over the results of state-of-the-art methods is confirmed through extensive experiments. For facilitating and validating the LuminAIRe task, a new dataset Car-LuminAIRe with lighting annotations and rich appearance variants is collected.

----

## [2814] A graphon-signal analysis of graph neural networks

**Authors**: *Ron Levie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb7943be26bb34f036c7e4068c490903-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb7943be26bb34f036c7e4068c490903-Abstract-Conference.html)

**Abstract**:

We present an approach for analyzing message passing graph neural networks (MPNNs) based on an extension of graphon analysis to a so called graphon-signal analysis. A MPNN is a function that takes a graph and a signal on the graph (a graph-signal) and returns some value. Since the input space of MPNNs is non-Euclidean, i.e., graphs can be of any size and topology, properties such as generalization are less well understood for MPNNs than for Euclidean neural networks. We claim that one important missing ingredient in past work is a meaningful notion of graph-signal similarity measure, that endows the space of inputs to MPNNs with a regular structure. We present such a similarity measure, called the graphon-signal cut distance, which makes the space of all graph-signals a dense subset of a compact metric space -- the graphon-signal space. Informally, two deterministic graph-signals are close in cut-distance if they ``look like'' they were sampled from the same random graph-signal model. Hence, our cut distance is a natural notion of graph-signal similarity, which allows comparing any pair of graph-signals of any size and topology. We prove that MPNNs are Lipschitz continuous functions over the graphon-signal metric space. We then give two applications of this result: 1) a generalization bound for MPNNs, and, 2) the stability of MPNNs to subsampling of graph-signals. Our results apply to any regular enough MPNN on any distribution of graph-signals, making the analysis rather universal.

----

## [2815] Lo-Hi: Practical ML Drug Discovery Benchmark

**Authors**: *Simon Steshin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb82f1f97ad0ca1d92df852a44a3bd73-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb82f1f97ad0ca1d92df852a44a3bd73-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Finding new drugs is getting harder and harder. One of the hopes of drug discovery is to use machine learning models to predict molecular properties. That is why models for molecular property prediction are being developed and tested on benchmarks such as MoleculeNet. However, existing benchmarks are unrealistic and are too different from applying the models in practice. We have created a new practical \emph{Lo-Hi} benchmark consisting of two tasks: Lead Optimization (Lo) and Hit Identification (Hi), corresponding to the real drug discovery process. For the Hi task, we designed a novel molecular splitting algorithm that solves the Balanced Vertex Minimum $k$-Cut problem.  We tested state-of-the-art and classic ML models, revealing which works better under practical settings. We analyzed modern benchmarks and showed that they are unrealistic and overoptimistic.Review: https://openreview.net/forum?id=H2Yb28qGLVLo-Hi benchmark: https://github.com/SteshinSS/lohi_neurips2023Lo-Hi splitter library: https://github.com/SteshinSS/lohi_splitter

----

## [2816] Class-Conditional Conformal Prediction with Many Classes

**Authors**: *Tiffany Ding, Anastasios Angelopoulos, Stephen Bates, Michael I. Jordan, Ryan J. Tibshirani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html)

**Abstract**:

Standard conformal prediction methods provide a marginal coverage guarantee,which means that for a random test point, the conformal prediction set contains the true label with a user-specified probability. In many classificationproblems, we would like to obtain a stronger guarantee--that for test pointsof a specific class, the prediction set contains the true label with thesame user-chosen probability. For the latter goal, existing conformal predictionmethods do not work well when there is a limited amount of labeled data perclass, as is often the case in real applications where the number of classes islarge. We propose a method called clustered conformal prediction thatclusters together classes having "similar" conformal scores and performs conformal prediction at the cluster level. Based on empirical evaluation acrossfour image data sets with many (up to 1000) classes, we find that clusteredconformal typically outperforms existing methods in terms of class-conditionalcoverage and set size metrics.

----

## [2817] Reward Imputation with Sketching for Contextual Batched Bandits

**Authors**: *Xiao Zhang, Ninglu Shao, Zihua Si, Jun Xu, Wenhan Wang, Hanjing Su, Ji-Rong Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cba76ef96c4cd625631ab4d33285b045-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cba76ef96c4cd625631ab4d33285b045-Abstract-Conference.html)

**Abstract**:

Contextual batched bandit (CBB) is a setting where a batch of rewards is observed from the environment at the end of each episode, but the rewards of the non-executed actions are unobserved, resulting in partial-information feedback. Existing approaches for CBB often ignore the rewards of the non-executed actions, leading to underutilization of feedback information. In this paper, we propose an efficient approach called Sketched Policy Updating with Imputed Rewards (SPUIR) that completes the unobserved rewards using sketching, which approximates the full-information feedbacks. We formulate reward imputation as an imputation regularized ridge regression problem that captures the feedback mechanisms of both executed and non-executed actions. To reduce time complexity, we solve the regression problem using randomized sketching. We prove that our approach achieves an instantaneous regret with controllable bias and smaller variance than approaches without reward imputation. Furthermore, our approach enjoys a sublinear regret bound against the optimal policy. We also present two extensions, a rate-scheduled version and a version for nonlinear rewards, making our approach more practical. Experimental results show that SPUIR outperforms state-of-the-art baselines on synthetic, public benchmark, and real-world datasets.

----

## [2818] A Unified Model and Dimension for Interactive Estimation

**Authors**: *Nataly Brukhim, Miro Dudík, Aldo Pacchiano, Robert E. Schapire*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html)

**Abstract**:

We study an abstract framework for interactive learning called interactive estimation in which the goal is to estimate a target from its ``similarity'' to points queried by the learner.We introduce a combinatorial measure called Dissimilarity dimension which largely captures learnability in our model.We present a simple, general, and broadly-applicable algorithm, for which we obtain both regret and PAC generalization bounds that are polynomial in the new dimension. We show that our framework subsumes and thereby unifies two classic learning models:statistical-query learning and structured bandits. We also delineate how the Dissimilarity dimension is related to well-known parameters for both frameworks, in some cases yielding significantly improved analyses.

----

## [2819] Simple, Scalable and Effective Clustering via One-Dimensional Projections

**Authors**: *Moses Charikar, Monika Henzinger, Lunjia Hu, Maximilian Vötsch, Erik Waingarten*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html)

**Abstract**:

Clustering is a fundamental problem in unsupervised machine learning with many applications in data analysis. Popular clustering algorithms such as Lloyd's algorithm and $k$-means++ can take $\Omega(ndk)$ time when clustering $n$ points in a $d$-dimensional space (represented by an $n\times d$ matrix $X$) into $k$ clusters. On massive datasets with moderate to large $k$, the multiplicative $k$ factor can become very expensive. We introduce a simple randomized clustering algorithm that provably runs in expected time $O(\mathsf{nnz}(X) + n\log n)$ for arbitrary $k$. Here $\mathsf{nnz}(X)$ is the total number of non-zero entries in the input dataset $X$, which is upper bounded by $nd$ and can be significantly smaller for sparse datasets. We prove that our algorithm achieves approximation ratio $\widetilde{O}(k^4)$ on any input dataset for the $k$-means objective, and our experiments show that the quality of the clusters found by our algorithm is usually much better than this worst-case bound. We use our algorithm for $k$-means clustering and for coreset construction; our experiments show that it gives a new tradeoff between running time and cluster quality compared to previous state-of-the-art methods for these tasks. Our theoretical analysis is based on novel results of independent interest. We show that the approximation ratio achieved after a random one-dimensional projection can be lifted to the original points and that $k$-means++ seeding can be implemented in expected time $O(n\log n)$ in one dimension.

----

## [2820] Streaming PCA for Markovian Data

**Authors**: *Syamantak Kumar, Purnamrita Sarkar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbb1fa8e7f515e796cda6621a703492f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbb1fa8e7f515e796cda6621a703492f-Abstract-Conference.html)

**Abstract**:

Since its inception in 1982, Oja's algorithm has become an established method for streaming principle component analysis (PCA). We study the problem of streaming PCA, where the data-points are sampled from an irreducible, aperiodic, and reversible Markov chain starting in stationarity. Our goal is to estimate the top eigenvector of the unknown covariance matrix of the stationary distribution. This setting has implications in scenarios where data can solely be sampled from a Markov Chain Monte Carlo (MCMC) type algorithm, and the objective is to perform inference on parameters of the stationary distribution. Most convergence guarantees for Oja's algorithm in the literature assume that the data-points are sampled IID. For data streams with Markovian dependence, one typically downsamples the data to get a "nearly" independent data stream. In this paper, we obtain the first near-optimal rate for Oja's algorithm on the entire data, where we remove the logarithmic dependence on the sample size, $n$, resulting from throwing data away in downsampling strategies.

----

## [2821] Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models

**Authors**: *Beier Zhu, Kaihua Tang, Qianru Sun, Hanwang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbe1fd3136e0f049bb8bc104231ccb99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbe1fd3136e0f049bb8bc104231ccb99-Abstract-Conference.html)

**Abstract**:

Foundation models like CLIP allow zero-shot transfer on various tasks without additional training data. Yet, the zero-shot performance is less competitive than a fully supervised one. Thus, to enhance the performance, fine-tuning and ensembling are also commonly adopted to better fit the downstream tasks. However, we argue that such prior work has overlooked the inherent biases in foundation models. Due to the highly imbalanced Web-scale training set, these foundation models are inevitably skewed toward frequent semantics, and thus the subsequent fine-tuning or ensembling is still biased. In this study, we systematically examine the biases in foundation models and demonstrate the efficacy of our proposed Generalized Logit Adjustment (GLA) method. Note that bias estimation in foundation models is challenging, as most pre-train data cannot be explicitly assessed like in traditional long-tailed classification tasks.To this end, GLA has an optimization-based bias estimation approach for debiasing foundation models. As our work resolves a fundamental flaw in the pre-training, the proposed GLA demonstrates significant improvements across a diverse range of tasks: it achieves 1.5 pp accuracy gains on ImageNet, an large average improvement (1.4-4.6 pp) on 11 few-shot datasets, 2.4 pp gains on long-tailed classification. Codes are in https://github.com/BeierZhu/GLA.

----

## [2822] Learning to Parameterize Visual Attributes for Open-set Fine-grained Retrieval

**Authors**: *Shijie Wang, Jianlong Chang, Haojie Li, Zhihui Wang, Wanli Ouyang, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html)

**Abstract**:

Open-set fine-grained retrieval is an emerging challenging task that allows to retrieve unknown categories beyond the training set. The best solution for handling unknown categories is to represent them using a set of visual attributes learnt from known categories, as widely used in zero-shot learning. Though important, attribute modeling usually requires significant manual annotations and thus is labor-intensive. Therefore, it is worth to investigate how to transform retrieval models trained by image-level supervision from category semantic extraction to attribute modeling. To this end, we propose a novel Visual Attribute Parameterization Network (VAPNet) to learn visual attributes from known categories and parameterize them into the retrieval model, without the involvement of any attribute annotations.In this way, VAPNet could utilize its parameters to parse a set of visual attributes from unknown categories and precisely represent them.Technically, VAPNet explicitly attains some semantics with rich details via making use of local image patches and distills the visual attributes from these discovered semantics. Additionally, it integrates the online refinement of these visual attributes into the training process to iteratively enhance their quality. Simultaneously, VAPNet treats these attributes as supervisory signals to tune the retrieval models, thereby achieving attribute parameterization. Extensive experiments on open-set fine-grained retrieval datasets validate the superior performance of our VAPNet over existing solutions.

----

## [2823] Learning List-Level Domain-Invariant Representations for Ranking

**Authors**: *Ruicheng Xian, Honglei Zhuang, Zhen Qin, Hamed Zamani, Jing Lu, Ji Ma, Kai Hui, Han Zhao, Xuanhui Wang, Michael Bendersky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc473bb3ec4176a5e640c3a6b5fb5239-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc473bb3ec4176a5e640c3a6b5fb5239-Abstract-Conference.html)

**Abstract**:

Domain adaptation aims to transfer the knowledge learned on (data-rich) source domains to (low-resource) target domains, and a popular method is invariant representation learning, which matches and aligns the data distributions on the feature space. Although this method is studied extensively and applied on classification and regression problems, its adoption on ranking problems is sporadic, and the few existing implementations lack theoretical justifications. This paper revisits invariant representation learning for ranking. Upon reviewing prior work, we found that they implement what we call item-level alignment, which aligns the distributions of the items being ranked from all lists in aggregate but ignores their list structure. However, the list structure should be leveraged, because it is intrinsic to ranking problems where the data and the metrics are defined and computed on lists, not the items by themselves. To close this discrepancy, we propose list-level alignmentâ€”learning domain-invariant representations at the higher level of lists. The benefits are twofold: it leads to the first domain adaptation generalization bound for ranking, in turn providing theoretical support for the proposed method, and it achieves better empirical transfer performance for unsupervised domain adaptation on ranking tasks, including passage reranking.

----

## [2824] Homotopy-based training of NeuralODEs for accurate dynamics discovery

**Authors**: *Joon-Hyuk Ko, Hankyul Koh, Nojun Park, Wonho Jhe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc56ae4929d792351a66c39aafb4a34d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc56ae4929d792351a66c39aafb4a34d-Abstract-Conference.html)

**Abstract**:

Neural Ordinary Differential Equations (NeuralODEs) present an attractive way to extract dynamical laws from time series data, as they bridge neural networks with the differential equation-based modeling paradigm of the physical sciences. However, these models often display long training times and suboptimal results, especially for longer duration data. While a common strategy in the literature imposes strong constraints to the NeuralODE architecture to inherently promote stable model dynamics, such methods are ill-suited for dynamics discovery as the unknown governing equation is not guaranteed to satisfy the assumed constraints. In this paper, we develop a new training method for NeuralODEs, based on synchronization and homotopy optimization, that does not require changes to the model architecture. We show that synchronizing the model dynamics and the training data tames the originally irregular loss landscape, which homotopy optimization can then leverage to enhance training. Through benchmark experiments, we demonstrate our method achieves competitive or better training loss while often requiring less than half the number of training epochs compared to other model-agnostic techniques. Furthermore, models trained with our method display better extrapolation capabilities, highlighting the effectiveness of our method.

----

## [2825] Simplifying and Empowering Transformers for Large-Graph Representations

**Authors**: *Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc57fac10eacadb3b72a907ac48f9a98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc57fac10eacadb3b72a907ac48f9a98-Abstract-Conference.html)

**Abstract**:

Learning representations on large-sized graphs is a long-standing challenge due to the inter-dependence nature involved in massive data points. Transformers, as an emerging class of foundation encoders for graph-structured data, have shown promising performance on small graphs due to its global attention capable of capturing all-pair influence beyond neighboring nodes. Even so, existing approaches tend to inherit the spirit of Transformers in language and vision tasks, and embrace complicated models by stacking deep multi-head attentions. In this paper, we critically demonstrate that even using a one-layer attention can bring up surprisingly competitive performance across node property prediction benchmarks where node numbers range from thousand-level to billion-level. This encourages us to rethink the design philosophy for Transformers on large graphs, where the global attention is a computation overhead hindering the scalability. We frame the proposed scheme as Simplified Graph Transformers (SGFormer), which is empowered by a simple attention model that can efficiently propagate information among arbitrary nodes in one layer. SGFormer requires none of positional encodings, feature/graph pre-processing or augmented loss. Empirically, SGFormer successfully scales to the web-scale graph ogbn-papers100M and yields up to 141x inference acceleration over SOTA Transformers on medium-sized graphs. Beyond current results, we believe the proposed methodology alone enlightens a new technical path of independent interest for building Transformers on large graphs.

----

## [2826] Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions

**Authors**: *Jun Xia, Lecheng Zhang, Xiao Zhu, Yue Liu, Zhangyang Gao, Bozhen Hu, Cheng Tan, Jiangbin Zheng, Siyuan Li, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html)

**Abstract**:

Molecular Property Prediction (MPP) is a crucial task in the AI-driven Drug Discovery (AIDD) pipeline, which has recently gained considerable attention thanks to advancements in deep learning. However, recent research has revealed that deep models struggle to beat traditional non-deep ones on MPP. In this study, we benchmark 12 representative models (3 non-deep models and 9 deep models) on 15 molecule datasets. Through the most comprehensive study to date, we make the following key observations: \textbf{(\romannumeral 1)} Deep models are generally unable to outperform non-deep ones; \textbf{(\romannumeral 2)} The failure of deep models on MPP cannot be solely attributed to the small size of molecular datasets; \textbf{(\romannumeral 3)} In particular, some traditional models including XGB and RF that use molecular fingerprints as inputs tend to perform better than other competitors. Furthermore, we conduct extensive empirical investigations into the unique patterns of molecule data and inductive biases of various models underlying these phenomena. These findings stimulate us to develop a simple-yet-effective feature mapping method for molecule data prior to feeding them into deep models. Empirically, deep models equipped with this mapping method can beat non-deep ones in most MoleculeNet datasets. Notably, the effectiveness is further corroborated by extensive experiments on cutting-edge dataset related to COVID-19 and activity cliff datasets.

----

## [2827] Neural Circuits for Fast Poisson Compressed Sensing in the Olfactory Bulb

**Authors**: *Jacob A. Zavatone-Veth, Paul Masset, William L. Tong, Joseph D. Zak, Venkatesh Murthy, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc8638553a347b1834d98be7613fa3f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc8638553a347b1834d98be7613fa3f0-Abstract-Conference.html)

**Abstract**:

Within a single sniff, the mammalian olfactory system can decode the identity and concentration of odorants wafted on turbulent plumes of air. Yet, it must do so given access only to the noisy, dimensionally-reduced representation of the odor world provided by olfactory receptor neurons. As a result, the olfactory system must solve a compressed sensing problem, relying on the fact that only a handful of the millions of possible odorants are present in a given scene. Inspired by this principle, past works have proposed normative compressed sensing models for olfactory decoding. However, these models have not captured the unique anatomy and physiology of the olfactory bulb, nor have they shown that sensing can be achieved within the 100-millisecond timescale of a single sniff. Here, we propose a rate-based Poisson compressed sensing circuit model for the olfactory bulb. This model maps onto the neuron classes of the olfactory bulb, and recapitulates salient features of their connectivity and physiology. For circuit sizes comparable to the human olfactory bulb, we show that this model can accurately detect tens of odors within the timescale of a single sniff. We also show that this model can perform Bayesian posterior sampling for accurate uncertainty estimation. Fast inference is possible only if the geometry of the neural code is chosen to match receptor properties, yielding a distributed neural code that is not axis-aligned to individual odor identities. Our results illustrate how normative modeling can help us map function onto specific neural circuits to generate new hypotheses.

----

## [2828] SUPA: A Lightweight Diagnostic Simulator for Machine Learning in Particle Physics

**Authors**: *Atul Kumar Sinha, Daniele Paliotta, Bálint Máté, John A. Raine, Tobias Golling, François Fleuret*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cca79c22037280d066fbd8bc35ac2e72-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cca79c22037280d066fbd8bc35ac2e72-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Deep learning methods have gained popularity in high energy physics for fast modeling of particle showers in detectors. Detailed simulation frameworks such as the gold standard \textsc{Geant4} are computationally intensive, and current deep generative architectures work on discretized, lower resolution versions of the detailed simulation. The development of models that work at higher spatial resolutions is currently hindered by the complexity of the full simulation data, and by the lack of simpler, more interpretable benchmarks. Our contribution is \textsc{SUPA}, the SUrrogate PArticle propagation simulator, an algorithm and software package for generating data by simulating simplified particle propagation, scattering and shower development in matter. The generation is extremely fast and easy to use compared to \textsc{Geant4}, but still exhibits the key characteristics and challenges of the detailed simulation. The proposed simulator generates thousands of particle showers per second on a desktop machine, a speed up of up to 6 orders of magnitudes over \textsc{Geant4}, and stores detailed geometric information about the shower propagation. \textsc{\textsc{SUPA}} provides much greater flexibility for setting initial conditions and defining multiple benchmarks for the development of models. Moreover, interpreting particle showers as point clouds creates a connection to geometric machine learning and provides challenging and fundamentally new datasets for the field.

----

## [2829] LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

**Authors**: *Artur P. Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus A. Adams*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning has been successfully applied to grid-based PDE modeling in various scientific applications. However, learned PDE solvers based on Lagrangian particle discretizations, which are the preferred approach to problems with free surfaces or complex physics, remain largely unexplored. We present LagrangeBench, the first benchmarking suite for Lagrangian particle problems, focusing on temporal coarse-graining. In particular, our contribution is: (a) seven new fluid mechanics datasets (four in 2D and three in 3D) generated with the Smoothed Particle Hydrodynamics (SPH) method including the Taylor-Green vortex, lid-driven cavity, reverse Poiseuille flow, and dam break, each of which includes different physics like solid wall interactions or free surface, (b) efficient JAX-based API with various recent training strategies and three neighbor search routines, and (c) JAX implementation of established Graph Neural Networks (GNNs) like GNS and SEGNN with baseline results. Finally, to measure the performance of learned surrogates we go beyond established position errors and introduce physical metrics like kinetic energy MSE and Sinkhorn distance for the particle distribution. Our codebase is available under the URL: https://github.com/tumaer/lagrangebench.

----

## [2830] Group Fairness in Peer Review

**Authors**: *Haris Aziz, Evi Micha, Nisarg Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccba10dd4e80e7276054222bb95d467c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccba10dd4e80e7276054222bb95d467c-Abstract-Conference.html)

**Abstract**:

Large conferences such as NeurIPS and AAAI serve as crossroads  of various AI fields, since they attract submissions from a vast number of communities. However, in some cases, this has resulted in a poor reviewing experience for some communities, whose submissions get assigned to less qualified reviewers outside of their communities. An often-advocated solution is to break up any such large conference into smaller conferences, but this can lead to isolation of communities and harm interdisciplinary research. We tackle this challenge by introducing a  notion of group fairness, called the core, which requires that every possible community (subset of researchers) to be treated in a way that prevents them from unilaterally benefiting by  withdrawing from a large conference. We study a simple peer review model, prove that it always admits a reviewing assignment in the core, and design an efficient algorithm to find one such assignment. We use real data from CVPR and ICLR conferences to compare our algorithm to existing reviewing assignment algorithms on a number of metrics.

----

## [2831] Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning

**Authors**: *Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html)

**Abstract**:

Diffusion models have demonstrated highly-expressive generative capabilities in vision and NLP. Recent studies in reinforcement learning (RL) have shown that diffusion models are also powerful in modeling complex policies or trajectories in offline datasets. However, these works have been limited to single-task settings where a generalist agent capable of addressing multi-task predicaments is absent. In this paper, we aim to investigate the effectiveness of a single diffusion model in modeling large-scale multi-task offline data, which can be challenging due to diverse and multimodal data distribution. Specifically, we propose Multi-Task Diffusion Model (\textsc{MTDiff}), a diffusion-based method that incorporates Transformer backbones and prompt learning for generative planning and data synthesis in multi-task offline settings. \textsc{MTDiff} leverages vast amounts of knowledge available in multi-task data and performs implicit knowledge sharing among tasks. For generative planning, we find \textsc{MTDiff} outperforms state-of-the-art algorithms across 50 tasks on Meta-World and 8 maps on Maze2D. For data synthesis, \textsc{MTDiff} generates high-quality data for testing tasks given a single demonstration as a prompt, which enhances the low-quality datasets for even unseen tasks.

----

## [2832] Single-Call Stochastic Extragradient Methods for Structured Non-monotone Variational Inequalities: Improved Analysis under Weaker Conditions

**Authors**: *Sayantan Choudhury, Eduard Gorbunov, Nicolas Loizou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf02786d28730e8311676ffa842e216-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf02786d28730e8311676ffa842e216-Abstract-Conference.html)

**Abstract**:

Single-call stochastic extragradient methods, like stochastic past extragradient (SPEG) and stochastic optimistic gradient (SOG),  have gained a lot of interest in recent years and are one of the most efficient algorithms for solving large-scale min-max optimization and variational inequalities problems (VIP) appearing in various machine learning tasks. However, despite their undoubted popularity, current convergence analyses of SPEG and SOG require strong assumptions like bounded variance or growth conditions. In addition, several important questions regarding the convergence properties of these methods are still open, including mini-batching, efficient step-size selection, and convergence guarantees under different sampling strategies. In this work, we address these questions and provide convergence guarantees for two large classes of structured non-monotone VIPs: (i) quasi-strongly monotone problems (a generalization of strongly monotone problems) and (ii) weak Minty variational inequalities (a generalization of monotone and Minty VIPs). We introduce the expected residual condition, explain its benefits, and show how it allows us to obtain a strictly weaker bound than previously used growth conditions, expected co-coercivity, or bounded variance assumptions. Finally, our convergence analysis holds under the arbitrary sampling paradigm, which includes importance sampling and various mini-batching strategies as special cases.

----

## [2833] Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment

**Authors**: *Tianhe Wu, Shuwei Shi, Haoming Cai, Mingdeng Cao, Jing Xiao, Yinqiang Zheng, Yujiu Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf4a7323b9ee3e54bf77f0e876b3f8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf4a7323b9ee3e54bf77f0e876b3f8b-Abstract-Conference.html)

**Abstract**:

Blind Omnidirectional Image Quality Assessment (BOIQA) aims to objectively assess the human perceptual quality of omnidirectional images (ODIs) without relying on pristine-quality image information. It is becoming more significant with the increasing advancement of virtual reality (VR) technology. However, the quality assessment of ODIs is severely hampered by the fact that the existing BOIQA pipeline lacks the modeling of the observer's browsing process. To tackle this issue, we propose a novel multi-sequence network for BOIQA called Assessor360, which is derived from the realistic multi-assessor ODI quality assessment procedure. Specifically, we propose a generalized Recursive Probability Sampling (RPS) method for the BOIQA task, combining content and details information to generate multiple pseudo viewport sequences from a given starting point. Additionally, we design a Multi-scale Feature Aggregation (MFA) module with a Distortion-aware Block (DAB) to fuse distorted and semantic features of each viewport. We also devise Temporal Modeling Module (TMM) to learn the viewport transition in the temporal domain. Extensive experimental results demonstrate that Assessor360 outperforms state-of-the-art methods on multiple OIQA datasets. The code and models are available at https://github.com/TianheWu/Assessor360.

----

## [2834] Lossy Image Compression with Conditional Diffusion Models

**Authors**: *Ruihan Yang, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html)

**Abstract**:

This paper outlines an end-to-end optimized lossy image compression framework using diffusion generative models. The approach relies on the transform coding paradigm, where an image is mapped into a latent space for entropy coding and, from there, mapped back to the data space for reconstruction. In contrast to VAE-based neural compression, where the (mean) decoder is a deterministic neural network, our decoder is a conditional diffusion model. Our approach thus introduces an additional "content" latent variable on which the reverse diffusion process is conditioned and uses this variable to store information about the image. The remaining ``texture'' variables characterizing the diffusion process are synthesized at decoding time. We show that the model's performance can be tuned toward perceptual metrics of interest. Our extensive experiments involving multiple datasets and image quality assessment metrics show that our approach yields stronger reported FID scores than the GAN-based model, while also yielding competitive performance with VAE-based models in several distortion metrics. Furthermore, training the diffusion with  $\mathcal{X}$-parameterization enables high-quality reconstructions in only a handful of decoding steps, greatly affecting the model's practicality. Our code is available at: https://github.com/buggyyang/CDC_compression

----

## [2835] Leveraging the two-timescale regime to demonstrate convergence of neural networks

**Authors**: *Pierre Marion, Raphaël Berthier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html)

**Abstract**:

We study the training dynamics of shallow neural networks, in a two-timescale regime in which the stepsizes for the inner layer are much smaller than those for the outer layer. In this regime, we prove convergence of the gradient flow to a global optimum of the non-convex optimization problem in a simple univariate setting. The number of neurons need not be asymptotically large for our result to hold, distinguishing our result from popular recent approaches such as the neural tangent kernel or mean-field regimes. Experimental illustration is provided, showing that the stochastic gradient descent behaves according to our description of the gradient flow and thus converges to a global optimum in the two-timescale regime, but can fail outside of this regime.

----

## [2836] Grammar Prompting for Domain-Specific Language Generation with Large Language Models

**Authors**: *Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, Yoon Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs)  can learn to perform a wide range of natural language tasks from just a  handful of in-context examples. However, for generating strings from highly structured  languages (e.g., semantic parsing to complex domain-specific languages), it is challenging for the LLM to generalize from just a few exemplars. We propose \emph{grammar prompting}, a simple approach to enable LLMs to use external knowledge and domain-specific constraints, expressed through a grammar in Backus--Naur Form (BNF), during in-context learning. Grammar prompting augments each demonstration example with a specialized grammar that is minimally sufficient for generating the particular output example, where the specialized grammar is a subset of the full DSL grammar.  For inference, the LLM first predicts a BNF grammar given a test input, and then generates the output according to the rules of the grammar. Experiments demonstrate that grammar prompting can enable LLMs to perform competitively on a diverse set of DSL generation tasks, including semantic parsing (SMCalFlow, Overnight, GeoQuery), PDDL planning, and SMILES-based molecule generation.

----

## [2837] Don't just prune by magnitude! Your mask topology is a secret weapon

**Authors**: *Duc Hoang, Souvik Kundu, Shiwei Liu, Zhangyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed significant progress in understanding the relationship between the connectivity of a deep network's architecture as a graph, and the network's performance. A few prior arts connected deep architectures to expander graphs or Ramanujan graphs, and particularly,[7] demonstrated the use of such graph connectivity measures with ranking and relative performance of various obtained sparse sub-networks  (i.e. models with prune masks) without the need for training. However, no prior work explicitly explores the role of parameters in the graph's connectivity, making the graph-based understanding of prune masks and the magnitude/gradient-based pruning practice isolated from one another. This paper strives to fill in this gap, by analyzing the Weighted Spectral Gap of Ramanujan structures in sparse neural networks and investigates its correlation with final performance. We specifically examine the evolution of sparse structures under a popular dynamic sparse-to-sparse network training scheme, and intriguingly find that the generated random topologies inherently maximize Ramanujan graphs. We also identify a strong correlation between masks, performance, and the weighted spectral gap. Leveraging this observation, we propose to construct a new "full-spectrum coordinate'' aiming to comprehensively characterize a sparse neural network's promise. Concretely, it consists of the classical Ramanujan's gap (structure), our proposed weighted spectral gap (parameters), and the constituent nested regular graphs within. In this new coordinate system, a sparse subnetwork's L2-distance from its original initialization is found to have nearly linear correlated with its performance. Eventually, we apply this unified perspective to develop a new actionable pruning method, by sampling sparse masks to maximize the L2-coordinate distance. Our method can be augmented with the "pruning at initialization" (PaI) method, and significantly outperforms existing PaI methods. With only a few iterations of training (e.g 500 iterations), we can get LTH-comparable performance as that yielded via "pruning after training", significantly saving pre-training costs. Codes can be found at:  https://github.com/VITA-Group/FullSpectrum-PAI.

----

## [2838] Learning Human Action Recognition Representations Without Real Humans

**Authors**: *Howard Zhong, Samarth Mishra, Donghyun Kim, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Aude Oliva, Rogério Feris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd556f38dba3a6c367c42fa85fc0801c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd556f38dba3a6c367c42fa85fc0801c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Pre-training on massive video datasets has become essential to achieve high action recognition performance on smaller downstream datasets. However, most large-scale video datasets contain images of people and hence are accompanied with issues related to privacy, ethics, and data protection, often preventing them from being publicly shared for reproducible research. Existing work has attempted to alleviate these problems by blurring faces, downsampling videos, or training on synthetic data. On the other hand, analysis on the {\em transferability} of privacy-preserving pre-trained models to downstream tasks has been limited. In this work, we study this problem by first asking the question: can we pre-train models for human action recognition with data that does not include real humans? To this end, we present, for the first time, a benchmark that leverages real-world videos with {\em humans removed} and synthetic data containing virtual humans to pre-train a model. We then evaluate the transferability of the representation learned on this data to a diverse set of downstream action recognition benchmarks. Furthermore, we propose a novel pre-training strategy, called Privacy-Preserving MAE-Align, to effectively combine synthetic data and human-removed real data. Our approach outperforms previous baselines by up to 5\% and closes the performance gap between human and no-human action recognition representations on downstream tasks, for both linear probing and fine-tuning. Our benchmark, code, and models are available at https://github.com/howardzh01/PPMA.

----

## [2839] Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation

**Authors**: *Yingyi Chen, Qinghua Tao, Francesco Tonin, Johan A. K. Suykens*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd687a58a13b673eea3fc1b2e4944cf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd687a58a13b673eea3fc1b2e4944cf7-Abstract-Conference.html)

**Abstract**:

Recently, a new line of works has emerged to understand and improve self-attention in Transformers by treating it as a kernel machine. However, existing works apply the methods for symmetric kernels to the asymmetric self-attention, resulting in a nontrivial gap between the analytical understanding and numerical implementation. In this paper, we provide a new perspective to represent and optimize self-attention through asymmetric Kernel Singular Value Decomposition (KSVD), which is also motivated by the low-rank property of self-attention normally observed in deep layers. Through asymmetric KSVD, i) a primal-dual representation of self-attention is formulated, where the optimization objective is cast to maximize the projection variances in the attention outputs; ii) a novel attention mechanism, i.e., Primal-Attention, is proposed via the primal representation of KSVD, avoiding explicit computation of the kernel matrix in the dual; iii) with KKT conditions, we prove that the stationary solution to the KSVD optimization in Primal-Attention yields a zero-value objective. In this manner, KSVD optimization can be implemented by simply minimizing a regularization loss, so that low-rank property is promoted without extra decomposition. Numerical experiments show state-of-the-art performance of our Primal-Attention with improved efficiency. Moreover, we demonstrate that the deployed KSVD optimization regularizes Primal-Attention with a sharper singular value decay than that of the canonical self-attention, further verifying the great potential of our method. To the best of our knowledge, this is the first work that provides a primal-dual representation for the asymmetric kernel in self-attention and successfully applies it to modelling and optimization.

----

## [2840] End-To-End Latent Variational Diffusion Models for Inverse Problems in High Energy Physics

**Authors**: *Alexander Shmakov, Kevin Greif, Michael James Fenton, Aishik Ghosh, Pierre Baldi, Daniel Whiteson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd830afc6208a346e4ec5caf1b08b4b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd830afc6208a346e4ec5caf1b08b4b4-Abstract-Conference.html)

**Abstract**:

High-energy collisions at the Large Hadron Collider (LHC) provide valuable insights into open questions in particle physics. However, detector effects must be corrected before measurements can be compared to certain theoretical predictions or measurements from other detectors. Methods to solve this inverse problem of mapping detector observations to theoretical quantities of the underlying collision are essential parts of many physics analyses at the LHC. We investigate and compare various generative deep learning methods to approximate this inverse mapping. We introduce a novel unified architecture, termed latent variational diffusion models, which combines the latent learning of cutting-edge generative art approaches with an end-to-end variational framework. We demonstrate the effectiveness of this approach for reconstructing global distributions of theoretical kinematic quantities, as well as for ensuring the adherence of the learned posterior distributions to known physics constraints. Our unified approach achieves a distribution-free distance to the truth of over 20 times smaller than non-latent state-of-the-art baseline and 3 times smaller than traditional latent diffusion models.

----

## [2841] AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web

**Authors**: *Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Existing datasets for automated fact-checking have substantial limitations, such as relying on artificial claims, lacking annotations for evidence and intermediate reasoning, or including evidence published after the claim. In this paper we introduce AVeriTeC, a new dataset of 4,568 real-world claims covering fact-checks by 50 different organizations. Each claim is annotated with question-answer pairs supported by evidence available online, as well as textual justifications explaining how the evidence combines to produce a verdict. Through a multi-round annotation process, we avoid common pitfalls including context dependence, evidence insufficiency, and temporal leakage, and reach a substantial inter-annotator agreement of $\kappa=0.619$ on verdicts. We develop a baseline as well as an evaluation scheme for verifying claims through question-answering against the open web.

----

## [2842] DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model

**Authors**: *Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, James Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html)

**Abstract**:

Pervasive integration of GPS-enabled devices and data acquisition technologies has led to an exponential increase in GPS trajectory data, fostering advancements in spatial-temporal data mining research. Nonetheless, GPS trajectories contain personal geolocation information, rendering serious privacy concerns when working with raw data. A promising approach to address this issue is trajectory generation, which involves replacing original data with generated, privacy-free alternatives. Despite the potential of trajectory generation, the complex nature of human behavior and its inherent stochastic characteristics pose challenges in generating high-quality trajectories. In this work, we propose a spatial-temporal diffusion probabilistic model for trajectory generation (DiffTraj). This model effectively combines the generative abilities of diffusion models with the spatial-temporal features derived from real trajectories. The core idea is to reconstruct and synthesize geographic trajectories from white noise through a reverse trajectory denoising process. Furthermore, we propose a Trajectory UNet (Traj-UNet) deep neural network to embed conditional information and accurately estimate noise levels during the reverse process. Experiments on two real-world datasets show that DiffTraj can be intuitively applied to generate high-fidelity trajectories while retaining the original distributions. Moreover, the generated results can support downstream trajectory analysis tasks and significantly outperform other methods in terms of geo-distribution evaluations.

----

## [2843] Meta-in-context learning in large language models

**Authors**: *Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt M. Botvinick, Jane X. Wang, Eric Schulz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cda04d7ea67ea1376bf8c6962d8541e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cda04d7ea67ea1376bf8c6962d8541e0-Abstract-Conference.html)

**Abstract**:

Large language models have shown tremendous performance in a variety of tasks.  In-context learning  -- the ability to improve at a task after being provided with a number of demonstrations -- is seen as one of the main contributors to their success. In the present paper, we demonstrate that the in-context learning abilities of large language models can be recursively improved via in-context learning itself. We coin this phenomenon meta-in-context learning. Looking at two idealized domains, a one-dimensional regression task and a two-armed bandit task, we show that meta-in-context learning adaptively reshapes a large language model's priors over expected tasks. Furthermore, we find that meta-in-context learning modifies the in-context learning strategies of such models. Finally, we broaden the scope of our investigation to encompass two diverse benchmarks: one focusing on real-world regression problems and the other encompassing multiple NLP tasks. In both cases, we observe competitive performance comparable to that of traditional learning algorithms. Taken together, our work improves our understanding of in-context learning and paves the way toward adapting large language models to the environment they are applied purely through meta-in-context learning rather than traditional finetuning.

----

## [2844] Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers

**Authors**: *Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurélien Lucchi, Thomas Hofmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdaac2a02c4fdcae77ba083b110efcc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdaac2a02c4fdcae77ba083b110efcc3-Abstract-Conference.html)

**Abstract**:

Autoregressive Transformers adopted in Large Language Models (LLMs) are hard to scale to long sequences. Despite several works trying to reduce their computational cost, most of LLMs still adopt attention layers between all pairs of tokens in the sequence, thus incurring a quadratic cost. In this study, we present a novel approach that dynamically prunes contextual information while preserving the model's expressiveness, resulting in reduced memory and computational requirements during inference. Our method employs a learnable mechanism that determines which uninformative tokens can be dropped from the context at any point across the generation process. By doing so, our approach not only addresses performance concerns but also enhances interpretability, providing valuable insight into the model's decision-making process. Our technique can be applied to existing pre-trained models through a straightforward fine-tuning process, and the pruning strength can be specified by a sparsity parameter. Notably, our empirical findings demonstrate that we can effectively prune up to 80\% of the context without significant performance degradation on downstream tasks, offering a valuable tool for mitigating inference costs. Our reference implementation achieves up to $2\times$ increase in inference throughput and even greater memory savings.

----

## [2845] SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning

**Authors**: *Dohyeok Lee, Seungyub Han, Taehyun Cho, Jungwoo Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdcaf772b4f8eda0385d0930517de64a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdcaf772b4f8eda0385d0930517de64a-Abstract-Conference.html)

**Abstract**:

Alleviating overestimation bias is a critical challenge for deep reinforcement learning to achieve successful performance on more complex tasks or offline datasets containing out-of-distribution data. In order to overcome overestimation bias, ensemble methods for Q-learning have been investigated to exploit the diversity of multiple Q-functions. Since network initialization has been the predominant approach to promote diversity in Q-functions, heuristically designed diversity injection methods have been studied in the literature. However, previous studies have not attempted to approach guaranteed independence over an ensemble from a theoretical perspective. By introducing a novel regularization loss for Q-ensemble independence based on random matrix theory, we propose spiked Wishart Q-ensemble independence regularization (SPQR) for reinforcement learning. Specifically, we modify the intractable hypothesis testing criterion for the Q-ensemble independence into a tractable KL divergence between the spectral distribution of the Q-ensemble and the target Wigner's semicircle distribution. We implement SPQR in several online and offline ensemble Q-learning algorithms. In the experiments, SPQR outperforms the baseline algorithms in both online and offline RL benchmarks.

----

## [2846] SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models

**Authors**: *Xiaosong Ma, Jie Zhang, Song Guo, Wenchao Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html)

**Abstract**:

Test-time adaptation (TTA) is a special and practical setting in unsupervised domain adaptation, which allows a pre-trained model in a source domain to adapt to unlabeled test data in another target domain. To avoid the computation-intensive backbone fine-tuning process, the zero-shot generalization potentials of the emerging pre-trained vision-language models (e.g., CLIP, CoOp) are leveraged to only tune the run-time prompt for unseen test domains. However, existing solutions have yet to fully exploit the representation capabilities of pre-trained models as they only focus on the entropy-based optimization and the performance is far below the supervised prompt adaptation methods, e.g., CoOp. In this paper, we propose SwapPrompt, a novel framework that can effectively leverage the self-supervised contrastive learning to facilitate the test-time prompt adaptation. SwapPrompt employs a dual prompts paradigm, i.e., an online prompt and a target prompt that averaged from the online prompt to retain historical information. In addition, SwapPrompt applies a swapped prediction mechanism, which takes advantage of the representation capabilities of pre-trained models to enhance the online prompt via contrastive learning. Specifically, we use the online prompt together with an augmented view of the input image to predict the class assignment generated by the target prompt together with an alternative augmented view of the same image. The proposed SwapPrompt can be easily deployed on vision-language models without additional requirement, and experimental results show that it achieves state-of-the-art test-time adaptation performance on ImageNet and nine other datasets. It is also shown that SwapPrompt can even achieve comparable performance with supervised prompt adaptation methods.

----

## [2847] Does a sparse ReLU network training problem always admit an optimum ?

**Authors**: *Quoc-Tung Le, Rémi Gribonval, Elisa Riccietti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdda0657a9f32bc7ddd4343686e7371e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdda0657a9f32bc7ddd4343686e7371e-Abstract-Conference.html)

**Abstract**:

Given a training set, a loss function, and a neural network architecture, it is often taken for granted that optimal network parameters exist, and a common practice is to apply available optimization algorithms to search for them. In this work, we show that the existence of an optimal solution is not always guaranteed, especially in the context of sparse ReLU neural networks.In particular, we first show that optimization problems involving deep networks with certain sparsity patterns do not always have optimal parameters, and that optimization algorithms may then diverge. Via a new topological relation between sparse ReLU neural networks and their linear counterparts, we derive --using existing tools from real algebraic geometry-- an algorithm to verify that a given sparsity pattern suffers from this issue. Then, the existence of a global optimum is proved for every concrete optimization problem involving a shallow sparse ReLU neural network of output dimension one. Overall, the analysis is based on the investigation of two topological properties of the space of functions implementable as sparse ReLU neural networks: a best approximation property, and a closedness property, both in the uniform norm. This is studied both for (finite) domains corresponding to practical training on finite training sets, and for more general domains such as the unit cube. This allows us to provide conditions for the guaranteed existence of an optimum given a sparsity pattern. The results apply not only to several sparsity patterns proposed in recent works on network pruning/sparsification, but also to classical dense neural networks, including architectures not covered by existing results.

----

## [2848] Knowledge Diffusion for Distillation

**Authors**: *Tao Huang, Yuan Zhang, Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdddf13f06182063c4dbde8cbd5a5c21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdddf13f06182063c4dbde8cbd5a5c21-Abstract-Conference.html)

**Abstract**:

The representation gap between teacher and student is an emerging topic in knowledge distillation (KD). To reduce the gap and improve the performance, current methods often resort to complicated training schemes, loss functions, and feature alignments, which are task-specific and feature-specific. In this paper, we state that the essence of these methods is to discard the noisy information and distill the valuable information in the feature, and propose a novel KD method dubbed DiffKD, to explicitly denoise and match features using diffusion models. Our approach is based on the observation that student features typically contain more noises than teacher features due to the smaller capacity of student model. To address this, we propose to denoise student features using a diffusion model trained by teacher features. This allows us to perform better distillation between the refined clean feature and teacher feature. Additionally, we introduce a light-weight diffusion model with a linear autoencoder to reduce the computation cost and an adaptive noise matching module to improve the denoising performance. Extensive experiments demonstrate that DiffKD is effective across various types of features and achieves state-of-the-art performance consistently on image classification, object detection, and semantic segmentation tasks. Code is available at https://github.com/hunto/DiffKD.

----

## [2849] BayesTune: Bayesian Sparse Deep Model Fine-tuning

**Authors**: *Minyoung Kim, Timothy M. Hospedales*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cde2dc73e0ad650176cdfa9b779eefc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cde2dc73e0ad650176cdfa9b779eefc7-Abstract-Conference.html)

**Abstract**:

Deep learning practice is increasingly driven by powerful foundation models (FM), pre-trained at scale and then fine-tuned for specific tasks of interest. A key property of this workflow is the efficacy of performing sparse or parameter-efficient fine-tuning, meaning that by updating only a tiny fraction of the whole FM parameters on a downstream task can lead to surprisingly good performance, often even superior to a full model update. However, it is not clear what is the optimal and principled way to select which parameters to update. Although a growing number of sparse fine-tuning ideas have been proposed, they are mostly not  satisfactory, relying on hand-crafted heuristics or heavy approximation. In this paper we propose a novel Bayesian sparse fine-tuning algorithm: we place a (sparse) Laplace prior for each parameter of the FM, with the mean equal to the initial value and the scale parameter having a hyper-prior that encourages small scale. Roughly speaking, the posterior means of the scale parameters indicate how important it is to update the corresponding parameter away from its initial value when solving the downstream task. Given the sparse prior, most scale parameters are small a posteriori, and the few large-valued scale parameters identify those FM parameters that crucially need to be updated away from their initial values. Based on this, we can threshold the scale parameters to decide which parameters to update or freeze, leading to a principled sparse fine-tuning strategy. To efficiently infer the posterior distribution of the scale parameters, we adopt the Langevin MCMC sampler, requiring only two times the complexity of the vanilla SGD. Tested on popular NLP benchmarks as well as the VTAB vision tasks, our approach shows significant improvement over the state-of-the-arts (e.g., 1% point higher than the best SOTA when fine-tuning RoBERTa for GLUE and SuperGLUE benchmarks).

----

## [2850] Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

**Authors**: *Yaoyu Zhu, Wei Fang, Xiaodong Xie, Tiejun Huang, Zhaofei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cde874a797a8300da693d5e412b7fdc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cde874a797a8300da693d5e412b7fdc0-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) are considered promising brain-inspired energy-efficient models due to their event-driven computing paradigm.The spatiotemporal spike patterns used to convey information in SNNs consist of both rate coding and temporal coding, where the temporal coding is crucial to biological-plausible learning rules such as spike-timing-dependent-plasticity.The time-based training strategy is proposed to better utilize the temporal information in SNNs and learn in an asynchronous fashion.However, some recent works train SNNs by the time-based scheme with rate-coding-dominated loss functions.In this paper, we first map rate-based loss functions to time-based counterparts and explain why they are also applicable to the time-based training scheme.After that, we infer that loss functions providing adequate positive overall gradients help training by theoretical analysis.Based on this, we propose the enhanced counting loss to replace the commonly used mean square counting loss.In addition, we transfer the training of scale factor in weight standardization into thresholds.Experiments show that our approach outperforms previous time-based training methods in most datasets. Our work provides insights for training SNNs with time-based schemes and offers a fresh perspective on the correlation between rate coding and temporal coding.Our code is available at https://github.com/zhuyaoyu/SNN-temporal-training-losses.

----

## [2851] Learning Rate Free Bayesian Inference in Constrained Domains

**Authors**: *Louis Sharrock, Lester Mackey, Christopher Nemeth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdee6c3eaa2adc285f11da7711a75c12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdee6c3eaa2adc285f11da7711a75c12-Abstract-Conference.html)

**Abstract**:

We introduce a suite of new particle-based algorithms for sampling in constrained domains which are entirely learning rate free. Our approach leverages coin betting ideas from convex optimisation, and the viewpoint of constrained sampling as a mirrored optimisation problem on the space of probability measures. Based on this viewpoint, we also introduce a unifying framework for several existing constrained sampling algorithms, including mirrored Langevin dynamics and mirrored Stein variational gradient descent. We demonstrate the performance of our algorithms on a range of numerical examples, including sampling from targets on the simplex, sampling with fairness constraints, and constrained sampling problems in post-selection inference. Our results indicate that our algorithms achieve competitive performance with existing constrained sampling methods, without the need to tune any hyperparameters.

----

## [2852] Volume Feature Rendering for Fast Neural Radiance Field Reconstruction

**Authors**: *Kang Han, Wei Xiang, Lu Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce182e31662883d4decc84a0255335b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce182e31662883d4decc84a0255335b6-Abstract-Conference.html)

**Abstract**:

Neural radiance fields (NeRFs) are able to synthesize realistic novel views from multi-view images captured from distinct positions and perspectives. In NeRF's rendering pipeline, neural networks are used to represent a scene independently or transform queried learnable feature vector of a point to the expected color or density. With the aid of geometry guides either in the form of occupancy grids or proposal networks, the number of color neural network evaluations can be reduced from hundreds to dozens in the standard volume rendering framework. However, many evaluations of the color neural network are still a bottleneck for fast NeRF reconstruction. This paper revisits volume feature rendering (VFR) for the purpose of fast NeRF reconstruction. The VFR integrates the queried feature vectors of a ray into one feature vector, which is then transformed to the final pixel color by a color neural network. This fundamental change to the standard volume rendering framework requires only one single color neural network evaluation to render a pixel, which substantially lowers the high computational complexity of the rendering framework attributed to a large number of color neural network evaluations. Consequently, we can use a comparably larger color neural network to achieve a better rendering quality while maintaining the same training and rendering time costs. This approach achieves the state-of-the-art rendering quality on both synthetic and real-world datasets while requiring less training time compared with existing methods.

----

## [2853] Offline RL with Discrete Proxy Representations for Generalizability in POMDPs

**Authors**: *Pengjie Gu, Xinyu Cai, Dong Xing, Xinrun Wang, Mengchen Zhao, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html)

**Abstract**:

Offline Reinforcement Learning (RL) has demonstrated promising results in various applications by learning policies from previously collected datasets, reducing the need for online exploration and interactions. However, real-world scenarios usually involve partial observability, which brings crucial challenges of the deployment of offline RL methods: i) the policy trained on data with full observability is not robust against the masked observations during execution, and ii) the information of which parts of observations are masked is usually unknown during training. In order to address these challenges, we present Offline RL with DiscrEte pRoxy representations (ORDER), a probabilistic framework which leverages novel state representations to improve the robustness against diverse masked observabilities. Specifically, we propose a discrete representation of the states and use a proxy representation to recover the states from masked partial observable trajectories. The training of ORDER can be compactly described as the following three steps. i) Learning the discrete state representations on data with full observations, ii) Training the decision module based on the discrete representations, and iii) Training the proxy discrete representations on the data with various partial observations, aligning with the discrete representations. We conduct extensive experiments to evaluate ORDER, showcasing its effectiveness in offline RL for diverse partially observable scenarios and highlighting the significance of discrete proxy representations in  generalization performance.ORDER is a flexible framework to employ any offline RL algorithms and we hope that ORDER can pave the way for the deployment of RL policy against various partial observabilities in the real world.

----

## [2854] Meta-AdaM: An Meta-Learned Adaptive Optimizer with Momentum for Few-Shot Learning

**Authors**: *Siyuan Sun, Hongyang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce26d21662c979d515164b416d4571fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce26d21662c979d515164b416d4571fe-Abstract-Conference.html)

**Abstract**:

We introduce Meta-AdaM, a meta-learned adaptive optimizer with momentum, designed for few-shot learning tasks that pose significant challenges to deep learning models due to the limited number of labeled examples. Meta-learning has been successfully employed to address these challenges by transferring meta-learned prior knowledge to new tasks. Most existing works focus on meta-learning an optimal model initialization or an adaptive learning rate learner for rapid convergence. However, these approaches either neglect to consider weight-update history for the adaptive learning rate learner or fail to effectively integrate momentum for fast convergence, as seen in many-shot learning settings. To tackle these limitations, we propose a meta-learned learning rate learner that utilizes weight-update history as input to predict more appropriate learning rates for rapid convergence. Furthermore, for the first time, our approach incorporates momentum into the optimization process of few-shot learning via a double look-ahead mechanism, enabling rapid convergence similar to many-shot settings. Extensive experimental results on benchmark datasets demonstrate the effectiveness of the proposed Meta-AdaM.

----

## [2855] Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments

**Authors**: *Thanh-Dat Truong, Hoang-Quan Nguyen, Bhiksha Raj, Khoa Luu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce3cf998b7f59271e80ce03fb74a7115-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce3cf998b7f59271e80ce03fb74a7115-Abstract-Conference.html)

**Abstract**:

Continual semantic segmentation aims to learn new classes while maintaining the information from the previous classes. Although prior studies have shown impressive progress in recent years, the fairness concern in the continual semantic segmentation needs to be better addressed. Meanwhile, fairness is one of the most vital factors in deploying the deep learning model, especially in human-related or safety applications. In this paper,  we present a novel Fairness Continual Learning approach to the semantic segmentation problem.In particular, under the fairness objective, a new fairness continual learning framework is proposed based on class distributions.Then, a novel Prototypical Contrastive Clustering loss is proposed to address the significant challenges in continual learning, i.e., catastrophic forgetting and background shift. Our proposed loss has also been proven as a novel, generalized learning paradigm of knowledge distillation commonly used in continual learning. Moreover, the proposed Conditional Structural Consistency loss further regularized the structural constraint of the predicted segmentation. Our proposed approach has achieved State-of-the-Art performance on three standard scene understanding benchmarks, i.e., ADE20K, Cityscapes, and Pascal VOC, and promoted the fairness of the segmentation model.

----

## [2856] Post Hoc Explanations of Language Models Can Improve Language Models

**Authors**: *Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, Himabindu Lakkaraju*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce65173b994cf7c925c71b482ee14a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce65173b994cf7c925c71b482ee14a8d-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) have demonstrated remarkable capabilities in performing complex tasks. Moreover, recent research has shown that incorporating human-annotated rationales (e.g., Chain-of-Thought prompting) during in-context learning can significantly enhance the performance of these models, particularly on tasks that require reasoning capabilities. However, incorporating such rationales poses challenges in terms of scalability as this requires a high degree of human involvement. In this work, we present a novel framework, Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY), which addresses the aforementioned challenges by automating the process of rationale generation. To this end, we leverage post hoc explanation methods which output attribution scores (explanations) capturing the influence of each of the input features on model predictions. More specifically, we construct automated natural language rationales that embed insights from post hoc explanations to provide corrective signals to LLMs. Extensive experimentation with real-world datasets demonstrates that our framework, AMPLIFY, leads to prediction accuracy improvements of about 10-25% over a wide range of tasks, including those where prior approaches which rely on human-annotated rationales such as Chain-of-Thought prompting fall short. Our work makes one of the first attempts at highlighting the potential of post hoc explanations as valuable tools for enhancing the effectiveness of LLMs. Furthermore, we conduct additional empirical analyses and ablation studies to demonstrate the impact of each of the components of AMPLIFY, which, in turn, lead to critical insights for refining in context learning.

----

## [2857] Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation

**Authors**: *Diederik P. Kingma, Ruiqi Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce79fbf9baef726645bc2337abb0ade2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce79fbf9baef726645bc2337abb0ade2-Abstract-Conference.html)

**Abstract**:

To achieve the highest perceptual quality, state-of-the-art diffusion models are optimized with objectives that typically look very different from the maximum likelihood and the Evidence Lower Bound (ELBO) objectives. In this work, we reveal that diffusion model objectives are actually closely related to the ELBO.Specifically, we show that all commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Under the condition of monotonic weighting, the connection is even closer: the diffusion objective then equals the ELBO, combined with simple data augmentation, namely Gaussian noise perturbation. We show that this condition holds for a number of state-of-the-art diffusion models. In experiments, we explore new monotonic weightings and demonstrate their effectiveness, achieving state-of-the-art FID scores on the high-resolution ImageNet benchmark.

----

## [2858] Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline

**Authors**: *Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, Yang You*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce7ff3405c782f761fac7f849b41ae9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce7ff3405c782f761fac7f849b41ae9a-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.

----

## [2859] When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning

**Authors**: *Siliang Zeng, Chenliang Li, Alfredo Garcia, Mingyi Hong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce9d3c592712d23f2ec3671941d67fa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce9d3c592712d23f2ec3671941d67fa1-Abstract-Conference.html)

**Abstract**:

Offline inverse reinforcement learning (Offline IRL) aims to recover the structure of rewards and environment dynamics that underlie observed actions in a fixed, finite set of demonstrations from an expert agent. Accurate models of expertise in executing a task has applications in safety-sensitive applications such as clinical decision making and autonomous driving. However, the structure of an expert's preferences implicit in observed actions is closely linked to the expert's model of the environment dynamics (i.e. the ``world''). Thus, inaccurate models of the world obtained from finite data with limited coverage could compound inaccuracy in estimated rewards. To address this issue, we propose a bi-level optimization formulation of the estimation task wherein the upper level is likelihood maximization based upon a conservative model of the expert's policy (lower level). The policy model is conservative in that it maximizes reward subject to a penalty that is increasing in the uncertainty of the estimated model of the world. We propose a new algorithmic framework to solve the bi-level optimization problem formulation and provide statistical and computational guarantees of performance for the associated optimal reward estimator. Finally,  we demonstrate that the proposed algorithm outperforms the state-of-the-art offline IRL and imitation learning benchmarks by a large margin, over the continuous control tasks in MuJoCo and different datasets in the D4RL benchmark.

----

## [2860] Neural Priming for Sample-Efficient Adaptation

**Authors**: *Matthew Wallingford, Vivek Ramanujan, Alex Fang, Aditya Kusupati, Roozbeh Mottaghi, Aniruddha Kembhavi, Ludwig Schmidt, Ali Farhadi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cea5bc68b890bffb10f18aaaab2becb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cea5bc68b890bffb10f18aaaab2becb1-Abstract-Conference.html)

**Abstract**:

We propose Neural Priming, a technique for adapting large pretrained models to distribution shifts and downstream tasks given few or no labeled examples. Presented with class names or unlabeled test samples, Neural Priming enables the model to recall and conditions its parameters on relevant data seen throughout pretraining, thereby priming it for the test distribution. Neural Priming can be performed at test time in even for pretraining datasets as large as LAION-2B. Performing lightweight updates on the recalled data significantly improves accuracy across a variety of distribution shift and transfer learning benchmarks. Concretely, in the zero-shot setting, we see a 2.45% improvement in accuracy on ImageNet and 3.81% accuracy improvement on average across standard transfer learning benchmarks. Further, using our test time inference scheme, we see a 1.41% accuracy improvement on ImageNetV2. These results demonstrate the effectiveness of Neural Priming in addressing the common challenge of limited labeled data and changing distributions. Code and models are open-sourced at https://www.github.com/RAIVNLab/neural-priming.

----

## [2861] Derandomized novelty detection with FDR control via conformal e-values

**Authors**: *Meshi Bashari, Amir Epstein, Yaniv Romano, Matteo Sesia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cec8ad7715d0d13899d5d7d31970f527-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cec8ad7715d0d13899d5d7d31970f527-Abstract-Conference.html)

**Abstract**:

Conformal inference provides a general distribution-free method to rigorously calibrate the output of any machine learning algorithm for novelty detection. While this approach has many strengths, it has the limitation of being randomized, in the sense that it may lead to different results when analyzing twice the same data and this can hinder the interpretation of any findings. We propose to make conformal inferences more stable by leveraging suitable conformal e-values instead of p-values to quantify statistical significance. This solution allows the evidence gathered from multiple analyses of the same data to be aggregated effectively while provably controlling the false discovery rate. Further, we show that the proposed method can reduce randomness without much loss of power compared to standard conformal inference, partly thanks to an innovative way of weighting conformal e-values based on additional side information carefully extracted from the same data. Simulations with synthetic and real data confirm this solution can be effective at eliminating random noise in the inferences obtained with state-of-the-art alternative techniques, sometimes also leading to higher power.

----

## [2862] ZipLM: Inference-Aware Structured Pruning of Language Models

**Authors**: *Eldar Kurtic, Elias Frantar, Dan Alistarh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ced46a50befedcb884ccf0cbe8c3ad23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ced46a50befedcb884ccf0cbe8c3ad23-Abstract-Conference.html)

**Abstract**:

The breakthrough performance of large language models (LLMs) comes with major computational footprints and high deployment costs. In this paper, we progress towards resolving this problem by proposing a novel structured compression approach for LLMs, called ZipLM. ZipLM achieves state-of-the-art accuracy-vs-speedup, while matching a set of desired target runtime speedups in any given inference environment. Specifically, given a model, a dataset, an inference environment, as well as a set of speedup targets, ZipLM iteratively identifies and removes components with the worst loss-runtime trade-off. Unlike prior methods that specialize in either the post-training/one-shot or the gradual compression setting, and only for specific families of models such as BERT (encoder) or GPT (decoder), ZipLM produces state-of-the-art compressed models across all these settings. Furthermore, ZipLM achieves superior results for a fraction of the computational cost relative to prior distillation and pruning techniques, making it a cost-effective approach for generating an entire family of smaller, faster, and highly accurate models, guaranteed to meet the desired inference specifications. In particular, ZipLM outperforms all prior BERT-base distillation and pruning techniques, such as CoFi, MiniLM, and TinyBERT. Moreover, it matches the performance of the heavily optimized MobileBERT model, obtained via extensive architecture search, by simply pruning the baseline BERT-large model. When compressing GPT2, ZipLM outperforms DistilGPT2 while being 60\% smaller and 30\% faster. Our code is available at: https://github.com/IST-DASLab/ZipLM.

----

## [2863] Private (Stochastic) Non-Convex Optimization Revisited: Second-Order Stationary Points and Excess Risks

**Authors**: *Daogao Liu, Arun Ganesh, Sewoong Oh, Abhradeep Guha Thakurta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cede701f00079e43d053ac57b1e75c3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cede701f00079e43d053ac57b1e75c3e-Abstract-Conference.html)

**Abstract**:

We reconsider the challenge of non-convex optimization under differential privacy constraint. Building upon the previous variance-reduced algorithm SpiderBoost, we propose a novel framework that employs two types of gradient oracles: one that estimates the gradient at a single point and a more cost-effective option that calculates the gradient difference between two points. Our framework can ensure continuous accuracy of gradient estimations and subsequently enhances the rates of identifying second-order stationary points.Additionally, we consider a more challenging task by attempting to locate the global minima of a non-convex objective via the exponential mechanism without almost any assumptions. Our preliminary results suggest that the regularized exponential mechanism can effectively emulate previous empirical and population risk bounds, negating the need for smoothness assumptions for algorithms with polynomial running time. Furthermore, with running time factors excluded, the exponential mechanism demonstrates promising population risk bound performance, and we provide a nearly matching lower bound.

----

## [2864] Revealing the unseen: Benchmarking video action recognition under occlusion

**Authors**: *Shresth Grover, Vibhav Vineet, Yogesh S. Rawat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cef53466b62aebbcf8aa2210a89b33a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cef53466b62aebbcf8aa2210a89b33a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In this work, we study the effect of occlusion on video action recognition. Tofacilitate this study, we propose three benchmark datasets and experiment withseven different video action recognition models. These datasets include two synthetic benchmarks, UCF-101-O and K-400-O, which enabled understanding the effects of fundamental properties of occlusion via controlled experiments. We also propose a real-world occlusion dataset, UCF-101-Y-OCC, which helps in further validating the findings of this study. We find several interesting insights such as 1) transformers are more robust than CNN counterparts, 2) pretraining make modelsrobust against occlusions, and 3) augmentation helps, but does not generalize well to real-world occlusions. In addition, we propose a simple transformer based compositional model, termed as CTx-Net, which generalizes well under this distribution shift. We observe that CTx-Net outperforms models which are trained using occlusions as augmentation, performing significantly better under natural occlusions. We believe this benchmark will open up interesting future research in robust video action recognition

----

## [2865] TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models

**Authors**: *Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Bölöni, Qian Lou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf04d01a0e76f8b13095349d9caca033-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf04d01a0e76f8b13095349d9caca033-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated.   Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks.  Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

----

## [2866] Minimax Forward and Backward Learning of Evolving Tasks with Performance Guarantees

**Authors**: *Verónica Álvarez, Santiago Mazuelas, Jose A. Lozano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf4114c34a2b93019aa6e70f99680fae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf4114c34a2b93019aa6e70f99680fae-Abstract-Conference.html)

**Abstract**:

For a sequence of classification tasks that arrive over time, it is common that tasks are evolving in the sense that consecutive tasks often have a higher similarity. The incremental learning of a growing sequence of tasks holds promise to enable accurate classification even with few samples per task by leveraging information from all the tasks in the sequence (forward and backward learning). However, existing techniques developed for continual learning and concept drift adaptation are either designed for tasks with time-independent similarities or only aim to learn the last task in the sequence. This paper presents incremental minimax risk classifiers (IMRCs) that effectively exploit forward and backward learning and account for evolving tasks. In addition, we analytically characterize the performance improvement provided by forward and backward learning in terms of the tasksâ€™ expected quadratic change and the number of tasks. The experimental evaluation shows that IMRCs can result in a significant performance improvement, especially for reduced sample sizes.

----

## [2867] Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms

**Authors**: *Dheeraj Baby, Saurabh Garg, Tzu-Ching Yen, Sivaraman Balakrishnan, Zachary C. Lipton, Yu-Xiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html)

**Abstract**:

This paper focuses on supervised and unsupervised online label shift,where the class marginals $Q(y)$ variesbut the class-conditionals $Q(x|y)$ remain invariant. In the unsupervised setting, our goal is to adapt a learner, trained on some offline labeled data, to changing label distributions given unlabeled online data. In the supervised setting, we must both learn a classifier and adapt to the dynamically evolving class marginals given only labeled online data. We develop novel algorithms that reduce the adaptation problem to online regression and guarantee optimal dynamic regret without any prior knowledge of the extent of drift in the label distribution. Our solution is based on bootstrapping the estimates of *online regression oracles* that track the drifting proportions. Experiments across numerous simulated and real-world online label shift scenarios demonstrate the superior performance of our proposed approaches, often achieving 1-3% improvement in accuracy while being sample and computationally efficient. Code is publicly available at https://github.com/Anon-djiwh/OnlineLabelShift

----

## [2868] Self-supervised video pretraining yields robust and more human-aligned visual representations

**Authors**: *Nikhil Parthasarathy, S. M. Ali Eslami, João Carreira, Olivier J. Hénaff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf57022dff0929796f85ac99d7cefa86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf57022dff0929796f85ac99d7cefa86-Abstract-Conference.html)

**Abstract**:

Humans learn powerful representations of objects and scenes by observing how they evolve over time. Yet, outside of specific tasks that require explicit temporal understanding, static image pretraining remains the dominant paradigm for learning visual foundation models. We question this mismatch, and ask whether video pretraining can yield visual representations that bear the hallmarks of human perception: generalisation across tasks, robustness to perturbations, and consistency with human judgements. To that end we propose a novel procedure for curating videos, and develop a contrastive framework which learns from the complex transformations therein. This simple paradigm for distilling knowledge from videos, called VITO, yields general representations that far outperform prior video pretraining methods on image understanding tasks, and image pretraining methods on video understanding tasks. Moreover, VITO representations are significantly more robust to natural and synthetic deformations than image-, video-, and adversarially-trainedones. Finally, VITOâ€™s predictions are strongly aligned with human judgements, surpassing models that were specifically trained for that purpose. Together, these results suggest that video pretraining could be a simple way of learning unified, robust, and human-aligned representations of the visual world.

----

## [2869] Slot-guided Volumetric Object Radiance Fields

**Authors**: *Di Qi, Tong Yang, Xiangyu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html)

**Abstract**:

We present a novel framework for 3D object-centric representation learning. Our approach effectively decomposes complex scenes into individual objects from a single image in an unsupervised fashion. This method, called \underline{s}lot-guided \underline{V}olumetric \underline{O}bject \underline{R}adiance \underline{F}ields~(sVORF), composes volumetric object radiance fields with object slots as a guidance to implement unsupervised 3D scene decomposition. Specifically, sVORF obtains object slots from a single image via a transformer module, maps these slots to volumetric object radiance fields with a hypernetwork and composes object radiance fields with the guidance of object slots at a 3D location. Moreover, sVORF significantly reduces memory requirement due to small-sized pixel rendering during training. We demonstrate the effectiveness of our approach by showing top results in scene decomposition and generation tasks of complex synthetic datasets (e.g., Room-Diverse). Furthermore, we also confirm the potential of sVORF to segment objects in real-world scenes (e.g., the LLFF dataset).  We hope our approach can provide preliminary understanding of the physical world and help ease future research in 3D object-centric representation learning.

----

## [2870] Riemannian SAM: Sharpness-Aware Minimization on Riemannian Manifolds

**Authors**: *Jihun Yun, Eunho Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf701db0e3b4d0b8681ca6915ac3e87e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf701db0e3b4d0b8681ca6915ac3e87e-Abstract-Conference.html)

**Abstract**:

Contemporary advances in the field of deep learning have embarked upon an exploration of the underlying geometric properties of data, thus encouraging the investigation of techniques that consider general manifolds, for example, hyperbolic or orthogonal neural networks. However, the optimization algorithms for training such geometric deep learning models still remain highly under-explored. In this paper, we introduce Riemannian SAM by generalizing conventional Euclidean SAM to Riemannian manifolds. We successfully formulate the sharpness-aware minimization on Riemannian manifolds, leading to one of a novel instantiation, Lorentz SAM. In addition, SAM variants proposed in previous studies such as Fisher SAM can be derived as special examples under our Riemannian SAM framework. We provide the convergence analysis of Riemannian SAM under a less aggressively decaying ascent learning rate than Euclidean SAM. Our analysis serves as a theoretically sound contribution encompassing a diverse range of manifolds, also providing the guarantees for SAM variants such as Fisher SAM, whose convergence analyses are absent. Lastly, we illustrate the superiority of Riemannian SAM in terms of generalization over previous Riemannian optimization algorithms through experiments on knowledge graph completion and machine translation tasks.

----

## [2871] ODE-based Recurrent Model-free Reinforcement Learning for POMDPs

**Authors**: *Xuanle Zhao, Duzhen Zhang, Liyuan Han, Tielin Zhang, Bo Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equations (ODEs) are widely recognized as the standard for modeling physical mechanisms, which help to perform approximate inference in unknown physical or biological environments. In partially observable (PO) environments, how to infer unseen information from raw observations puzzled the agents. By using a recurrent policy with a compact context, context-based reinforcement learning provides a flexible way to extract unobservable information from historical transitions. To help the agent extract more dynamics-related information, we present a novel ODE-based recurrent model combines with model-free reinforcement learning (RL) framework to solve partially observable Markov decision processes (POMDPs). We experimentally demonstrate the efficacy of our methods across various PO continuous control and meta-RL tasks. Furthermore, our experiments illustrate that our method is robust against irregular observations, owing to the ability of ODEs to model irregularly-sampled time series.

----

## [2872] Deep Contract Design via Discontinuous Networks

**Authors**: *Tonghan Wang, Paul Duetting, Dmitry Ivanov, Inbal Talgam-Cohen, David C. Parkes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7700139af1fa346d2f57f1f5c26c18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7700139af1fa346d2f57f1f5c26c18-Abstract-Conference.html)

**Abstract**:

Contract design involves a principal who establishes contractual agreements about payments for outcomes that arise from the actions of an agent. In this paper, we initiate the study of deep learning for the automated design of optimal contracts. We introduce a novel representation: the Discontinuous ReLU (DeLU) network, which models the principal's utility as a discontinuous piecewise affine function of the design of a contract where each piece corresponds to the agent taking a particular action. DeLU networks implicitly learn closed-form expressions for the incentive compatibility constraints of the agent and the utility maximization objective of the principal, and support parallel inference on each piece through linear programming or interior-point methods that solve for optimal contracts. We provide empirical results that demonstrate success in approximating the principal's utility function with a small number of training samples and scaling to find approximately optimal contracts on problems with a large number of actions and outcomes.

----

## [2873] Temporal Continual Learning with Prior Compensation for Human Motion Prediction

**Authors**: *Jianwei Tang, Jiangxin Sun, Xiaotong Lin, Lifang Zhang, Wei-Shi Zheng, Jian-Fang Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html)

**Abstract**:

Human Motion Prediction (HMP) aims to predict future poses at different moments according to past motion sequences. Previous approaches have treated the prediction of various moments equally, resulting in two main limitations: the learning of short-term predictions is hindered by the focus on long-term predictions, and the incorporation of prior information from past predictions into subsequent predictions is limited. In this paper, we introduce a novel multi-stage training framework called Temporal Continual Learning (TCL) to address the above challenges. To better preserve prior information, we introduce the Prior Compensation Factor (PCF). We incorporate it into the model training to compensate for the lost prior information. Furthermore, we derive a more reasonable optimization objective through theoretical derivation. It is important to note that our TCL framework can be easily integrated with different HMP backbone models and adapted to various datasets and applications. Extensive experiments on four HMP benchmark datasets demonstrate the effectiveness and flexibility of TCL. The code is available at https://github.com/hyqlat/TCL.

----

## [2874] Kernel Quadrature with Randomly Pivoted Cholesky

**Authors**: *Ethan Epperly, Elvira Moreno*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7ba4b2d14e0f6a0e8247af77745094-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7ba4b2d14e0f6a0e8247af77745094-Abstract-Conference.html)

**Abstract**:

This paper presents new quadrature rules for functions in a reproducing kernel Hilbert space using nodes drawn by a sampling algorithm known as randomly pivoted Cholesky. The resulting computational procedure compares favorably to previous kernel quadrature methods, which either achieve low accuracy or require solving a computationally challenging sampling problem. Theoretical and numerical results show that randomly pivoted Cholesky is fast and achieves comparable quadrature error rates to more computationally expensive quadrature schemes based on continuous volume sampling, thinning, and recombination. Randomly pivoted Cholesky is easily adapted to complicated geometries with arbitrary kernels, unlocking new potential for kernel quadrature.

----

## [2875] Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods

**Authors**: *Tobit Klug, Dogukan Atik, Reinhard Heckel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfaea3a519edf73c3a0480ae8f00bc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfaea3a519edf73c3a0480ae8f00bc4e-Abstract-Conference.html)

**Abstract**:

Supervised training of deep neural networks on pairs of clean image and noisy measurement achieves state-of-the-art performance for many image reconstruction tasks, but such training pairs are difficult to collect. Self-supervised methods enable training based on noisy measurements only, without clean images. In this work, we investigate the cost of self-supervised training in terms of sample complexity for a class of self-supervised methods that enable the computation of unbiased estimates of gradients of the supervised loss, including noise2noise methods. We analytically show that a model trained with such self-supervised training is as good as the same model trained in a supervised fashion, but self-supervised training requires more examples than supervised training. We then study self-supervised denoising and accelerated MRI empirically and characterize the cost of self-supervised training in terms of the number of additional samples required, and find that the performance gap between self-supervised and supervised training vanishes as a function of the training examples, at a problem-dependent rate, as predicted by our theory.

----

## [2876] FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression

**Authors**: *Youguang Chen, George Biros*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfcadfe84ee49908cde1fc2992c38d20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfcadfe84ee49908cde1fc2992c38d20-Abstract-Conference.html)

**Abstract**:

We investigate theory and algorithms for pool-based active learning for multiclass classification using multinomial logistic regression.  Using finite sample analysis, we prove that the Fisher Information Ratio (FIR)  lower and upper bounds  the excess risk. Based on our theoretical analysis, we propose an active learning algorithm that  employs regret minimization to minimize the FIR. To verify our derived excess risk bounds, we conduct experiments on synthetic datasets. Furthermore, we compare FIRAL with five other methods and found that our scheme  outperforms them: it consistently produces the smallest classification error in the multiclass logistic regression setting, as demonstrated through experiments on MNIST, CIFAR-10, and 50-class ImageNet.

----

## [2877] AMDP: An Adaptive Detection Procedure for False Discovery Rate Control in High-Dimensional Mediation Analysis

**Authors**: *Jiarong Ding, Xuehu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfce727868dcaf5295c0125f9d6fbc0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfce727868dcaf5295c0125f9d6fbc0b-Abstract-Conference.html)

**Abstract**:

High-dimensional mediation analysis is often associated with a multiple testing problem for detecting significant mediators. Assessing the uncertainty of this detecting process via false discovery rate (FDR) has garnered great interest. To control the FDR in multiple testing, two essential steps are involved: ranking and selection. Existing approaches either construct p-values without calibration or disregard the joint information across tests, leading to conservation in FDR control or non-optimal ranking rules for multiple hypotheses. In this paper, we develop an adaptive mediation detection procedure (referred to as "AMDP") to identify relevant mediators while asymptotically controlling the FDR in high-dimensional mediation analysis. AMDP produces the optimal rule for ranking hypotheses and proposes a data-driven strategy to determine the threshold for mediator selection. This novel method captures information from the proportions of composite null hypotheses and the distribution of p-values, which turns the high dimensionality into an advantage instead of a limitation. The numerical studies on synthetic and real data sets illustrate the performances of AMDP compared with existing approaches.

----

## [2878] Strong and Precise Modulation of Human Percepts via Robustified ANNs

**Authors**: *Guy Gaziv, Michael J. Lee, James J. DiCarlo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d00904cebc0d5b69fada8ad33d0f1422-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d00904cebc0d5b69fada8ad33d0f1422-Abstract-Conference.html)

**Abstract**:

The visual object category reports of artificial neural networks (ANNs) are notoriously sensitive to tiny, adversarial image perturbations. Because human category reports (aka human percepts) are thought to be insensitive to those same small-norm perturbations -- and locally stable in general -- this argues that ANNs are incomplete scientific models of human visual perception. Consistent with this, we show that when small-norm image perturbations are generated by standard ANN models, human object category percepts are indeed highly stable.  However, in this very same "human-presumed-stable" regime, we find that robustified ANNs reliably discover low-norm image perturbations that strongly disrupt human percepts. These previously undetectable human perceptual disruptions are massive in amplitude, approaching the same level of sensitivity seen in robustified ANNs.  Further, we show that robustified ANNs support precise perceptual state interventions: they guide the construction of low-norm image perturbations that strongly alter human category percepts toward specific prescribed percepts.  In sum, these contemporary models of biological visual processing are now accurate enough to guide strong and precise interventions on human perception.

----

## [2879] MomentDiff: Generative Video Moment Retrieval from Random to Real

**Authors**: *Pandeng Li, Chen-Wei Xie, Hongtao Xie, Liming Zhao, Lei Zhang, Yun Zheng, Deli Zhao, Yongdong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d01bda31bbcd780774ff15b534e03c40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d01bda31bbcd780774ff15b534e03c40-Abstract-Conference.html)

**Abstract**:

Video moment retrieval pursues an efficient and generalized solution to identify the specific temporal segments within an untrimmed video that correspond to a given language description.To achieve this goal, we provide a generative diffusion-based framework called MomentDiff, which simulates a typical human retrieval process from random browsing to gradual localization.Specifically, we first diffuse the real span to random noise, and learn to denoise the random noise to the original span with the guidance of similarity between text and video.This allows the model to learn a mapping from arbitrary random locations to real moments, enabling the ability to locate segments from random initialization.Once trained, MomentDiff could sample random temporal segments as initial guesses and iteratively refine them to generate an accurate temporal boundary.Different from discriminative works (e.g., based on learnable proposals or queries), MomentDiff with random initialized spans could resist the temporal location biases from datasets.To evaluate the influence of the temporal location biases, we propose two ``anti-bias'' datasets with location distribution shifts, named Charades-STA-Len and Charades-STA-Mom.The experimental results demonstrate that our efficient framework consistently outperforms state-of-the-art methods on three public benchmarks, and exhibits better generalization and robustness on the proposed anti-bias datasets. The code, model, and anti-bias evaluation datasets will be released publicly.

----

## [2880] Experimental Designs for Heteroskedastic Variance

**Authors**: *Justin Weltz, Tanner Fiez, Alexander Volfovsky, Eric Laber, Blake Mason, Houssam Nassif, Lalit Jain*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d01db5cd2555ba11f75da0454d57b903-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d01db5cd2555ba11f75da0454d57b903-Abstract-Conference.html)

**Abstract**:

Most linear experimental design problems assume homogeneous variance, while the presence of heteroskedastic noise is present in many realistic settings. Let a learner have access to a finite set of measurement vectors $\mathcal{X}\subset \mathbb{R}^d$ that can be probed to receive noisy linear responses of the form $y=x^{\top}\theta^{\ast}+\eta$. Here $\theta^{\ast}\in \mathbb{R}^d$ is an unknown parameter vector, and $\eta$ is independent mean-zero $\sigma_x^2$-sub-Gaussian noise defined by a flexible heteroskedastic variance model, $\sigma_x^2 = x^{\top}\Sigma^{\ast}x$. Assuming that $\Sigma^{\ast}\in \mathbb{R}^{d\times d}$ is an unknown matrix, we propose, analyze and empirically evaluate a novel design for uniformly bounding estimation error of the variance parameters, $\sigma_x^2$. We demonstrate this method on two adaptive experimental design problems under heteroskedastic noise, fixed confidence transductive best-arm identification and level-set identification and prove the first instance-dependent lower bounds in these settings.Lastly, we construct near-optimal algorithms and demonstrate the large improvements in sample complexity gained from accounting for heteroskedastic variance in these designs empirically.

----

## [2881] NeuralGF: Unsupervised Point Normal Estimation by Learning Neural Gradient Function

**Authors**: *Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d027a5c93d484a4312cc486d399c62c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d027a5c93d484a4312cc486d399c62c1-Abstract-Conference.html)

**Abstract**:

Normal estimation for 3D point clouds is a fundamental task in 3D geometry processing. The state-of-the-art methods rely on priors of fitting local surfaces learned from normal supervision. However, normal supervision in benchmarks comes from synthetic shapes and is usually not available from real scans, thereby limiting the learned priors of these methods. In addition, normal orientation consistency across shapes remains difficult to achieve without a separate post-processing procedure. To resolve these issues, we propose a novel method for estimating oriented normals directly from point clouds without using ground truth normals as supervision. We achieve this by introducing a new paradigm for learning neural gradient functions, which encourages the neural network to fit the input point clouds and yield unit-norm gradients at the points. Specifically, we introduce loss functions to facilitate query points to iteratively reach the moving targets and aggregate onto the approximated surface, thereby learning a global surface representation of the data. Meanwhile, we incorporate gradients into the surface approximation to measure the minimum signed deviation of queries, resulting in a consistent gradient field associated with the surface. These techniques lead to our deep unsupervised oriented normal estimator that is robust to noise, outliers and density variations. Our excellent results on widely used benchmarks demonstrate that our method can learn more accurate normals for both unoriented and oriented normal estimation tasks than the latest methods. The source code and pre-trained model are publicly available.

----

## [2882] Gacs-Korner Common Information Variational Autoencoder

**Authors**: *Michael Kleinman, Alessandro Achille, Stefano Soatto, Jonathan C. Kao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d04f08ccf582011f43af91ee1c1956d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d04f08ccf582011f43af91ee1c1956d2-Abstract-Conference.html)

**Abstract**:

We propose a notion of common information that allows one to quantify and separate the information that is shared between two random variables from the information that is unique to each. Our notion of common information is defined by an optimization problem over a family of functions and recovers the G\'acs-K\"orner common information as a special case. Importantly, our notion can be approximated empirically using samples from the underlying data distribution. We then provide a method to partition and quantify the common and unique information using a simple modification of a traditional variational auto-encoder. Empirically, we demonstrate that our formulation allows us to learn semantically meaningful common and unique factors of variation even on high-dimensional data such as images and videos. Moreover, on datasets where ground-truth latent factors are known, we show that we can accurately quantify the common information between the random variables.

----

## [2883] LEACE: Perfect linear concept erasure in closed form

**Authors**: *Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)

**Abstract**:

Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called concept scrubbing, which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Our code is available at https://github.com/EleutherAI/concept-erasure.

----

## [2884] Robust low-rank training via approximate orthonormal constraints

**Authors**: *Dayana Savostianova, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d073692637b4fb8c4eb4b81f0fa2df7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d073692637b4fb8c4eb4b81f0fa2df7b-Abstract-Conference.html)

**Abstract**:

With the growth of model and data sizes, a broad effort has been made to design pruning techniques that reduce the resource demand of deep learning pipelines, while retaining model performance. In order to reduce both inference and training costs, a prominent line of work uses low-rank matrix factorizations to represent the network weights. Although able to retain accuracy,  we observe that low-rank methods tend to compromise model robustness against adversarial perturbations. By modeling robustness in terms of the condition number of the neural network, we argue that this loss of robustness is due to the exploding singular values of the low-rank weight matrices. Thus, we introduce a robust low-rank training algorithm that maintains the network's weights on the low-rank matrix manifold while  simultaneously enforcing approximate orthonormal constraints. The resulting model reduces both training and inference costs while ensuring well-conditioning and thus better adversarial robustness, without compromising model accuracy. This is shown by extensive numerical evidence and by our main approximation theorem that shows the computed robust low-rank network well-approximates the ideal full model, provided a highly performing low-rank sub-network exists.

----

## [2885] Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials

**Authors**: *Shengchao Liu, Weitao Du, Yanjing Li, Zhuoxinran Li, Zhiling Zheng, Chenru Duan, Zhi-Ming Ma, Omar Yaghi, Animashree Anandkumar, Christian Borgs, Jennifer T. Chayes, Hongyu Guo, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d07379f3acf3af51dfc8598862cadfa0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d07379f3acf3af51dfc8598862cadfa0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Artificial intelligence for scientific discovery has recently generated significant interest within the machine learning and scientific communities, particularly in the domains of chemistry, biology, and material discovery. For these scientific problems, molecules serve as the fundamental building blocks, and machine learning has emerged as a highly effective and powerful tool for modeling their geometric structures. Nevertheless, due to the rapidly evolving process of the field and the knowledge gap between science ({\eg}, physics,  chemistry, \& biology) and machine learning communities, a benchmarking study on geometrical representation for such data has not been conducted. To address such an issue, in this paper, we first provide a unified view of the current symmetry-informed geometric methods, classifying them into three main categories: invariance, equivariance with spherical frame basis, and equivariance with vector frame basis. Then we propose a platform, coined Geom3D, which enables benchmarking the effectiveness of geometric strategies. Geom3D contains 16 advanced symmetry-informed geometric representation models and 14 geometric pretraining methods over 52 diverse tasks, including small molecules, proteins, and crystalline materials. We hope that Geom3D can, on the one hand, eliminate barriers for machine learning researchers interested in exploring scientific problems; and, on the other hand, provide valuable guidance for researchers in computational chemistry, structural biology, and materials science, aiding in the informed selection of representation techniques for specific applications. The source code is available on \href{https://github.com/chao1224/Geom3D}{the GitHub repository}.

----

## [2886] Feature Selection in the Contrastive Analysis Setting

**Authors**: *Ethan Weinberger, Ian Covert, Su-In Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d083980ec9f874025550136b776a96a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d083980ec9f874025550136b776a96a9-Abstract-Conference.html)

**Abstract**:

Contrastive analysis (CA) refers to the exploration of variations uniquely enriched in a target dataset as compared to a corresponding background dataset generated from sources of variation that are irrelevant to a given task. For example, a biomedical data analyst may wish to find a small set of genes to use as a proxy for variations in genomic data only present among patients with a given disease (target) as opposed to healthy control subjects (background). However, as of yet the problem of feature selection in the CA setting has received little attention from the machine learning community. In this work we present contrastive feature selection (CFS),a method for performing feature selection in the CA setting. We motivate our approach with a novel information-theoretic analysis of representation learning in the CA setting, and we empirically validate CFS on a semi-synthetic dataset and four real-world biomedical datasets. We find that our method consistently outperforms previously proposed state-of-the-art supervised and fully unsupervised feature selection methods not designed for the CA setting. An open-source implementation of our method is available at https://github.com/suinleelab/CFS.

----

## [2887] GeoDE: a Geographically Diverse Evaluation Dataset for Object Recognition

**Authors**: *Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron Adcock, Laurens van der Maaten, Deepti Ghadiyaram, Olga Russakovsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d08b6801f24dda81199079a3371d77f9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d08b6801f24dda81199079a3371d77f9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Current dataset collection methods typically scrape large amounts of data from the web. While this technique is extremely scalable, data collected in this way tends to reinforce stereotypical biases, can contain personally identifiable information, and typically originates from Europe and North America. In this work, we rethink the dataset collection paradigm and introduce GeoDE, a geographically diverse dataset with 61,940 images from 40 classes and 6 world regions, and no personally identifiable information, collected by soliciting images from people across the world. We analyse GeoDE to understand differences in images collected in this manner compared to web-scraping. Despite the smaller size of this dataset, we demonstrate its use as both an evaluation and training dataset, allowing us to highlight shortcomings in current models, as well as demonstrate improved performance even when training on this small dataset. We release the full dataset and code at https://geodiverse-data-collection.cs.princeton.edu/

----

## [2888] Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs

**Authors**: *Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, Alejandro Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0949cbcec31c09431610553a284f94a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0949cbcec31c09431610553a284f94a-Abstract-Conference.html)

**Abstract**:

We study the problem of computing an optimal policy of an infinite-horizon discounted constrained Markov decision process (constrained MDP). Despite the popularity of Lagrangian-based policy search methods used in practice, the oscillation of policy iterates in these methods has not been fully understood, bringing out issues such as violation of constraints and sensitivity to hyper-parameters. To fill this gap, we employ the Lagrangian method to cast a constrained MDP into a constrained saddle-point problem in which max/min players correspond to primal/dual variables, respectively, and develop two single-time-scale policy-based primal-dual algorithms with non-asymptotic convergence of their policy iterates to an optimal constrained policy. Specifically, we first propose a regularized policy gradient primal-dual (RPG-PD) method that updates the policy using an entropy-regularized policy gradient, and the dual variable via a quadratic-regularized gradient ascent, simultaneously. We prove that the policy primal-dual iterates of RPG-PD converge to a regularized saddle point with a sublinear rate, while the policy iterates converge sublinearly to an optimal constrained policy. We further instantiate RPG-PD in large state or action spaces by including function approximation in policy parametrization, and establish similar sublinear last-iterate policy convergence. Second, we propose an optimistic policy gradient primal-dual (OPG-PD) method that employs the optimistic gradient method to update primal/dual variables, simultaneously. We prove that the policy primal-dual iterates of OPG-PD converge to a saddle point that contains an optimal constrained policy, with a linear rate. To the best of our knowledge, this work appears to be the first non-asymptotic policy last-iterate convergence result for single-time-scale algorithms in constrained MDPs. We further validate the merits and the effectiveness of our methods in computational experiments.

----

## [2889] Unleashing the Power of Randomization in Auditing Differentially Private ML

**Authors**: *Krishna Pillutla, Galen Andrew, Peter Kairouz, H. Brendan McMahan, Alina Oprea, Sewoong Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d09ef5264966e17adffd3157265c9946-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d09ef5264966e17adffd3157265c9946-Abstract-Conference.html)

**Abstract**:

We present a rigorous methodology for auditing differentially private machine learning by adding multiple carefully designed examples called canaries. We take a first principles approach based on three key components. First, we introduce Lifted Differential Privacy (LiDP) that expands the definition of differential privacy to handle randomized datasets. This gives us the freedom to design randomized canaries. Second, we audit LiDP by trying to distinguish between the model trained with $K$ canaries versus $K-1$ canaries in the dataset, leaving one canary out. By drawing the canaries i.i.d., LiDP can leverage the symmetry in the design and reuse each privately trained model to run multiple statistical tests, one for each canary. Third, we introduce novel confidence intervals that take advantage of the multiple test statistics by adapting to the empirical higher-order correlations. Together, this new recipe demonstrates significant improvements in sample complexity, both theoretically and empirically, using synthetic and real data. Further, recent advances in designing stronger canaries can be readily incorporated in the new framework.

----

## [2890] Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations

**Authors**: *Piotr Indyk, Haike Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0ac28b79816b51124fcc804b2496a36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0ac28b79816b51124fcc804b2496a36-Abstract-Conference.html)

**Abstract**:

Graph-based approaches to nearest neighbor search are popular and powerful tools for handling large datasets in practice, but they have limited theoretical guarantees. We study the worst-case performance of recent graph-based approximate nearest neighbor search algorithms, such as HNSW, NSG and DiskANN. For DiskANN, we show that its "slow preprocessing'' version provably supports approximate nearest neighbor search query with constant approximation ratio and poly-logarithmic query time, on data sets with bounded "intrinsic'' dimension. For the other data structure variants studied, including DiskANN with "fast preprocessing'',  HNSW and NSG,  we present a family of instances on which the empirical query time required to achieve a "reasonable'' accuracy is linear in instance size. For example, for DiskANN, we show that the query procedure can take at least  $0.1 n$ steps on instances of size $n$ before it encounters any of the $5$ nearest neighbors of the query.

----

## [2891] On the Overlooked Structure of Stochastic Gradients

**Authors**: *Zeke Xie, Qian-Yuan Tang, Mingming Sun, Ping Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0b2eda0386f477ab14d7e181e16c899-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0b2eda0386f477ab14d7e181e16c899-Abstract-Conference.html)

**Abstract**:

Stochastic gradients closely relate to both optimization and generalization of deep neural networks (DNNs). Some works attempted to explain the success of stochastic optimization for deep learning by the arguably heavy-tail properties of gradient noise, while other works presented theoretical and empirical evidence against the heavy-tail hypothesis on gradient noise. Unfortunately, formal statistical tests for analyzing the structure and heavy tails of stochastic gradients in deep learning are still under-explored. In this paper, we mainly make two contributions. First, we conduct formal statistical tests on the distribution of stochastic gradients and gradient noise across both parameters and iterations. Our statistical tests reveal that dimension-wise gradients usually exhibit power-law heavy tails, while iteration-wise gradients and stochastic gradient noise caused by minibatch training usually do not exhibit power-law heavy tails. Second, we further discover that the covariance spectra of stochastic gradients have the power-law structures overlooked by previous studies and present its theoretical implications for training of DNNs. While previous studies believed that the anisotropic structure of stochastic gradients matters to deep learning, they did not expect the gradient covariance can have such an elegant mathematical structure. Our work challenges the existing belief and provides novel insights on the structure of stochastic gradients in deep learning.

----

## [2892] ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram

**Authors**: *Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-Myoung Kwon, Edward Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0b67349dd16b83b2cf6167fb4e2be50-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0b67349dd16b83b2cf6167fb4e2be50-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Question answering (QA) in the field of healthcare has received much attention due to significant advancements in natural language processing. However, existing healthcare QA datasets primarily focus on medical images, clinical notes, or structured electronic health record tables. This leaves the vast potential of combining electrocardiogram (ECG) data with these systems largely untapped. To address this gap, we present ECG-QA, the first QA dataset specifically designed for ECG analysis. The dataset comprises a total of 70 question templates that cover a wide range of clinically relevant ECG topics, each validated by an ECG expert to ensure their clinical utility. As a result, our dataset includes diverse ECG interpretation questions, including those that require a comparative analysis of two different ECGs. In addition, we have conducted numerous experiments to provide valuable insights for future research directions. We believe that ECG-QA will serve as a valuable resource for the development of intelligent QA systems capable of assisting clinicians in ECG interpretations.

----

## [2893] Provable convergence guarantees for black-box variational inference

**Authors**: *Justin Domke, Robert M. Gower, Guillaume Garrigos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0bcff6425bbf850ec87d5327a965db9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0bcff6425bbf850ec87d5327a965db9-Abstract-Conference.html)

**Abstract**:

Black-box variational inference is widely used in situations where there is no proof that its stochastic optimization succeeds. We suggest this is due to a theoretical gap in existing stochastic optimization proofsâ€”namely the challenge of gradient estimators with unusual noise bounds, and a composite non-smooth objective. For dense Gaussian variational families, we observe that existing gradient estimators based on reparameterization satisfy a quadratic noise bound and give novel convergence guarantees for proximal and projected stochastic gradient descent using this bound. This provides rigorous guarantees that methods similar to those used in practice converge on realistic inference problems.

----

## [2894] Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation

**Authors**: *Wenhao Ding, Laixi Shi, Yuejie Chi, Ding Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0c3841867db2c516454845a450ca885-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0c3841867db2c516454845a450ca885-Abstract-Conference.html)

**Abstract**:

Robustness has been extensively studied in reinforcement learning (RL) to handle various forms of uncertainty such as random perturbations, rare events, and malicious attacks. In this work, we consider one critical type of robustness against spurious correlation, where different portions of the state do not have correlations induced by unobserved confounders. These spurious correlations are ubiquitous in real-world tasks, for instance, a self-driving car usually observes heavy traffic in the daytime and light traffic at night due to unobservable human activity. A model that learns such useless or even harmful correlation could catastrophically fail when the confounder in the test case deviates from the training one. Although motivated, enabling robustness against spurious correlation poses significant challenges since the uncertainty set, shaped by the unobserved confounder and causal structure, is difficult to characterize and identify. Existing robust algorithms that assume simple and unstructured uncertainty sets are therefore inadequate to address this challenge. To solve this issue, we propose Robust State-Confounded Markov Decision Processes (RSC-MDPs) and theoretically demonstrate its superiority in avoiding learning spurious correlations compared with other robust RL counterparts. We also design an empirical algorithm to learn the robust optimal policy for RSC-MDPs, which outperforms all baselines in eight realistic self-driving and manipulation tasks.

----

## [2895] IBA: Towards Irreversible Backdoor Attacks in Federated Learning

**Authors**: *Thuy Dung Nguyen, Tuan Nguyen, Anh Tran, Khoa D. Doan, Kok-Seng Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a distributed learning approach that enables machine learning models to be trained on decentralized data without compromising end devices' personal, potentially sensitive data. However, the distributed nature and uninvestigated data intuitively introduce new security vulnerabilities, including backdoor attacks. In this scenario, an adversary implants backdoor functionality into the global model during training, which can be activated to cause the desired misbehaviors for any input with a specific adversarial pattern. Despite having remarkable success in triggering and distorting model behavior, prior backdoor attacks in FL often hold impractical assumptions, limited imperceptibility, and durability. Specifically, the adversary needs to control a sufficiently large fraction of clients or know the data distribution of other honest clients. In many cases, the trigger inserted is often visually apparent, and the backdoor effect is quickly diluted if the adversary is removed from the training process. To address these limitations, we propose a novel backdoor attack framework in FL, the Irreversible Backdoor Attack (IBA), that jointly learns the optimal and visually stealthy trigger and then gradually implants the backdoor into a global model. This approach allows the adversary to execute a backdoor attack that can evade both human and machine inspections. Additionally, we enhance the efficiency and durability of the proposed attack by selectively poisoning the model's parameters that are least likely updated by the main task's learning process and constraining the poisoned model update to the vicinity of the global model. Finally, we evaluate the proposed attack framework on several benchmark datasets, including MNIST, CIFAR-10, and Tiny ImageNet, and achieved high success rates while simultaneously bypassing existing backdoor defenses and achieving a more durable backdoor effect compared to other backdoor attacks. Overall, IBA offers a more effective, stealthy, and durable approach to backdoor attacks in FL. The code associated with this paper is available on GitHub.

----

## [2896] Spontaneous symmetry breaking in generative diffusion models

**Authors**: *Gabriel Raya, Luca Ambrogioni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0da30e312b75a3fffd9e9191f8bc1b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0da30e312b75a3fffd9e9191f8bc1b0-Abstract-Conference.html)

**Abstract**:

Generative diffusion models have recently emerged as a leading approach for generating high-dimensional data. In this paper, we show that the dynamics of these models exhibit a spontaneous symmetry breaking that divides the generative dynamics into two distinct phases: 1) A linear steady-state dynamics around a central fixed-point and 2) an attractor dynamics directed towards the data manifold. These two "phases'' are separated by the change in stability of the central fixed-point, with the resulting window of instability being responsible for the diversity of the generated samples. Using both theoretical and empirical evidence, we show that an accurate simulation of the early dynamics does not significantly contribute to the final generation, since early fluctuations are reverted to the central fixed point. To leverage this insight, we propose a Gaussian late initialization scheme, which significantly improves model performance, achieving up to 3x FID improvements on fast samplers, while also increasing sample diversity (e.g., racial composition of generated CelebA images). Our work offers a new way to understand the generative dynamics of diffusion models that has the potential to bring about higher performance and less biased fast-samplers.

----

## [2897] RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing

**Authors**: *Antoine Scardigli, Lukas Cavigelli, Lorenz K. Müller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1422213c9f2bdd5178b77d166fba86a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1422213c9f2bdd5178b77d166fba86a-Abstract-Conference.html)

**Abstract**:

Monte-Carlo path tracing is a powerful technique for realistic image synthesis but suffers from high levels of noise at low sample counts, limiting its use in real-time applications. To address this, we propose a framework with end-to-end training of a sampling importance network, a latent space encoder network, and a denoiser network. Our approach uses reinforcement learning to optimize the sampling importance network, thus avoiding explicit numerically approximated gradients. Our method does not aggregate the sampled values per pixel by averaging but keeps all sampled values which are then fed into the latent space encoder. The encoder replaces handcrafted spatiotemporal heuristics by learned representations in a latent space. Finally, a neural denoiser is trained to refine the output image. Our approach increases visual quality on several challenging datasets and reduces rendering times for equal quality by a factor of 1.6x compared to the previous state-of-the-art, making it a promising solution for real-time applications.

----

## [2898] A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks

**Authors**: *Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, Salman Avestimehr*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html)

**Abstract**:

Deep learning models often suffer from forgetting previously learned information when trained on new data. This problem is exacerbated in federated learning (FL), where the data is distributed and can change independently for each user. Many solutions are proposed to resolve this catastrophic forgetting in a centralized setting. However, they do not apply directly to FL because of its unique complexities, such as privacy concerns and resource limitations. To overcome these challenges, this paper presents a framework for \textbf{federated class incremental learning} that utilizes a generative model to synthesize samples from past distributions. This data can be later exploited alongside the training data to mitigate catastrophic forgetting. To preserve privacy, the generative model is trained on the server using data-free methods at the end of each task without requesting data from clients. Moreover, our solution does not demand the users to store old data or models, which gives them the freedom to join/leave the training at any time. Additionally, we introduce SuperImageNet, a new regrouping of the ImageNet dataset specifically tailored for federated continual learning. We demonstrate significant improvements compared to existing baselines through extensive experiments on multiple datasets.

----

## [2899] On the Connection between Pre-training Data Diversity and Fine-tuning Robustness

**Authors**: *Vivek Ramanujan, Thao Nguyen, Sewoong Oh, Ali Farhadi, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1786f5246c67eefde011599d31b2006-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1786f5246c67eefde011599d31b2006-Abstract-Conference.html)

**Abstract**:

Pre-training has been widely adopted in deep learning to improve model performance, especially when the training data for a target task is limited. In our work, we seek to understand the implications of this training strategy on the generalization properties of downstream models. More specifically, we ask the following question: how do properties of the pre-training distribution affect the robustness of a fine-tuned model? The properties we explore include the label space, label semantics, image diversity, data domains, and data quantity of the pre-training distribution. We find that the primary factor influencing downstream effective robustness (Taori et al., 2020) is data quantity, while other factors have limited significance. For example, reducing the number of ImageNet pre-training classes by 4x while increasing the number of images per class by 4x (that is, keeping total data quantity fixed) does not impact the robustness of fine-tuned models. We demonstrate our findings on pre-training distributions drawn from various natural and synthetic data sources, primarily using the iWildCam-WILDS distribution shift as a test for robustness.

----

## [2900] Structured Neural Networks for Density Estimation and Causal Inference

**Authors**: *Asic Q. Chen, Ruian Shi, Xiang Gao, Ricardo Baptista, Rahul G. Krishnan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1881b5125b4e9cf42f6d6d0b6575934-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1881b5125b4e9cf42f6d6d0b6575934-Abstract-Conference.html)

**Abstract**:

Injecting structure into neural networks enables learning functions that satisfy invariances with respect to subsets of inputs. For instance, when learning generative models using neural networks, it is advantageous to encode the conditional independence structure of observed variables, often in the form of Bayesian networks. We propose the Structured Neural Network (StrNN), which injects structure through masking pathways in a neural network. The masks are designed via a novel relationship we explore between neural network architectures and binary matrix factorization, to ensure that the desired independencies are respected. We devise and study practical algorithms for this otherwise NP-hard design problem based on novel objectives that control the model architecture. We demonstrate the utility of StrNN in three applications: (1) binary and Gaussian density estimation with StrNN, (2) real-valued density estimation with Structured Autoregressive Flows (StrAFs) and Structured Continuous Normalizing Flows (StrCNF), and (3) interventional and counterfactual analysis with StrAFs for causal inference. Our work opens up new avenues for learning neural networks that enable data-efficient generative modeling and the use of normalizing flows for causal effect estimation.

----

## [2901] Decision-Aware Actor-Critic with Function Approximation and Theoretical Guarantees

**Authors**: *Sharan Vaswani, Amirreza Kazemi, Reza Babanezhad Harikandeh, Nicolas Le Roux*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d18d208fa9c333483e5724ade7beff0f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d18d208fa9c333483e5724ade7beff0f-Abstract-Conference.html)

**Abstract**:

Actor-critic (AC) methods are widely used in reinforcement learning (RL), and benefit from the flexibility of using any policy gradient method as the actor and value-based method as the critic. The critic is usually trained by minimizing the TD error, an objective that is potentially decorrelated with the true goal of achieving a high reward with the actor. We address this mismatch by designing a joint objective for training the actor and critic in a decision-aware fashion. We use the proposed objective to design a generic, AC algorithm that can easily handle any function approximation. We explicitly characterize the conditions under which the resulting algorithm guarantees monotonic policy improvement, regardless of the choice of the policy and critic parameterization. Instantiating the generic algorithm results in an actor that involves maximizing a sequence of surrogate functions (similar to TRPO, PPO), and a critic that involves minimizing a closely connected objective. Using simple bandit examples, we provably establish the benefit of the proposed critic objective over the standard squared error. Finally, we empirically demonstrate the benefit of our decision-aware actor-critic framework on simple RL problems.

----

## [2902] Label-Retrieval-Augmented Diffusion Models for Learning from Noisy Labels

**Authors**: *Jian Chen, Ruiyi Zhang, Tong Yu, Rohan Sharma, Zhiqiang Xu, Tong Sun, Changyou Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d191ba4c8923ed8fd8935b7c98658b5f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d191ba4c8923ed8fd8935b7c98658b5f-Abstract-Conference.html)

**Abstract**:

Learning from noisy labels is an important and long-standing problem in machine learning for real applications. One of the main research lines focuses on learning a label corrector to purify potential noisy labels. However, these methods typically rely on strict assumptions and are limited to certain types of label noise. In this paper, we reformulate the label-noise problem from a generative-model perspective, i.e., labels are generated by gradually refining an initial random guess. This new perspective immediately enables existing powerful diffusion models to seamlessly learn the stochastic generative process. Once the generative uncertainty is modeled, we can perform classification inference using maximum likelihood estimation of labels. To mitigate the impact of noisy labels, we propose the Label-Retrieval-Augmented (LRA) diffusion model, which leverages neighbor consistency to effectively construct pseudo-clean labels for diffusion training. Our model is flexible and general, allowing easy incorporation of different types of conditional information, e.g., use of pre-trained models, to further boost model performance. Extensive experiments are conducted for evaluation. Our model achieves new state-of-the-art (SOTA) results on all the standard real-world benchmark datasets. Remarkably, by incorporating conditional information from the powerful CLIP model, our method can boost the current SOTA accuracy by 10-20 absolute points in many cases. Code is available: https://anonymous.4open.science/r/LRA-diffusion-5F2F

----

## [2903] Cheaply Estimating Inference Efficiency Metrics for Autoregressive Transformer Models

**Authors**: *Deepak Narayanan, Keshav Santhanam, Peter Henderson, Rishi Bommasani, Tony Lee, Percy Liang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1a14493e5f84d6c6129414f0cd1a7c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1a14493e5f84d6c6129414f0cd1a7c6-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) are highly capable but also computationally expensive. Characterizing the fundamental tradeoff between inference efficiency and model capabilities is thus important, but requires an efficiency metric that is comparable across models from different providers.Unfortunately, raw runtimes measured through black-box APIs do not satisfy this property: model providers can implement software and hardware optimizations orthogonal to the model, and shared infrastructure introduces performance contention.We propose a new metric for inference efficiency called idealized runtime, that puts models on equal footing as though they were served on uniform hardware and software without performance contention, and a cost model to efficiently estimate this metric for autoregressive Transformer models.We also propose variants of the idealized runtime that incorporate the number and type of accelerators needed to serve the model.Using these metrics, we compare ten LLMs developed in 2022 to provide the first analysis of inference efficiency-capability tradeoffs; we make several observations from this analysis, including the fact that the superior inference runtime performance of certain APIs is often a byproduct of optimizations within the API rather than the underlying model.Our code is open sourced at https://github.com/stanford-crfm/helm-efficiency.

----

## [2904] Differentiable sorting for censored time-to-event data

**Authors**: *Andre Vauvelle, Benjamin Wild, Roland Eils, Spiros C. Denaxas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1a25d7e93f06cb422b3a74a0aa3bf3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1a25d7e93f06cb422b3a74a0aa3bf3f-Abstract-Conference.html)

**Abstract**:

Survival analysis is a crucial semi-supervised task in machine learning with significant real-world applications, especially in healthcare. The most common approach to survival analysis, Coxâ€™s partial likelihood, can be interpreted as a ranking model optimized on a lower bound of the concordance index.  We follow these connections further, with listwise ranking losses that allow for a relaxation of the pairwise independence assumption. Given the inherent transitivity of ranking, we explore differentiable sorting networks as a means to introduce a stronger transitive inductive bias during optimization. Despite their potential, current differentiable sorting methods cannot account for censoring, a crucial aspect of many real-world datasets. We propose a novel method, Diffsurv, to overcome this limitation by extending differentiable sorting methods to handle censored tasks. Diffsurv predicts matrices of possible permutations that accommodate the label uncertainty introduced by censored samples. Our experiments reveal that Diffsurv outperforms established baselines in various simulated and real-world risk prediction scenarios. Furthermore, we demonstrate the algorithmic advantages of Diffsurv by presenting a novel method for top-k risk prediction that surpasses current methods.

----

## [2905] Multi-Agent Meta-Reinforcement Learning: Sharper Convergence Rates with Task Similarity

**Authors**: *Weichao Mao, Haoran Qiu, Chen Wang, Hubertus Franke, Zbigniew Kalbarczyk, Ravishankar K. Iyer, Tamer Basar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1b1a091088904cbc7f7faa2b45c8f36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1b1a091088904cbc7f7faa2b45c8f36-Abstract-Conference.html)

**Abstract**:

Multi-agent reinforcement learning (MARL) has primarily focused on solving a single task in isolation, while in practice the environment is often evolving, leaving many related tasks to be solved. In this paper, we investigate the benefits of meta-learning in solving multiple MARL tasks collectively. We establish the first line of theoretical results for meta-learning in a wide range of fundamental MARL settings, including learning Nash equilibria in two-player zero-sum Markov games and Markov potential games, as well as learning coarse correlated equilibria in general-sum Markov games. Under natural notions of task similarity, we show that meta-learning achieves provable sharper convergence to various game-theoretical solution concepts than learning each task separately. As an important intermediate step, we develop multiple MARL algorithms with initialization-dependent convergence guarantees. Such algorithms integrate optimistic policy mirror descents with stage-based value updates, and their refined convergence guarantees (nearly) recover the best known results even when a good initialization is unknown. To our best knowledge, such results are also new and might be of independent interest. We further provide numerical simulations to corroborate our theoretical findings.

----

## [2906] Feature Learning for Interpretable, Performant Decision Trees

**Authors**: *Jack H. Good, Torin Kovach, Kyle Miller, Artur Dubrawski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1b4076ae067dd23bad5ac2693547a01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1b4076ae067dd23bad5ac2693547a01-Abstract-Conference.html)

**Abstract**:

Decision trees are regarded for high interpretability arising from their hierarchical partitioning structure built on simple decision rules. However, in practice, this is not realized because axis-aligned partitioning of realistic data results in deep trees, and because ensemble methods are used to mitigate overfitting. Even then, model complexity and performance remain sensitive to transformation of the input, and extensive expert crafting of features from the raw data is common. We propose the first system to alternate sparse feature learning with differentiable decision tree construction to produce small, interpretable trees with good performance. We benchmark against conventional tree-based models and demonstrate several notions of interpretation of a model and its predictions.

----

## [2907] Particle-based Variational Inference with Generalized Wasserstein Gradient Flow

**Authors**: *Ziheng Cheng, Shiyue Zhang, Longlin Yu, Cheng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1c1f61023dca672117b58f813a12d99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1c1f61023dca672117b58f813a12d99-Abstract-Conference.html)

**Abstract**:

Particle-based variational inference methods (ParVIs) such as Stein variational gradient descent (SVGD) update the particles based on the kernelized Wasserstein gradient flow for the Kullback-Leibler (KL) divergence. However, the design of kernels is often non-trivial and can be restrictive for the flexibility of the method. Recent works show that functional gradient flow approximations with quadratic form regularization terms can improve performance. In this paper, we propose a ParVI framework, called generalized Wasserstein gradient descent (GWG), based on a generalized Wasserstein gradient flow of the KL divergence, which can be viewed as a functional gradient method with a broader class of regularizers induced by convex functions. We show that GWG exhibits strong convergence guarantees. We also provide an adaptive version that automatically chooses Wasserstein metric to accelerate convergence. In experiments, we demonstrate the effectiveness and efficiency of the proposed framework on both simulated and real data problems.

----

## [2908] PAC Learning Linear Thresholds from Label Proportions

**Authors**: *Anand Brahmbhatt, Rishi Saket, Aravindan Raghuveer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1d3cdc9e28b0c67b9df90fca4d1c1b3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1d3cdc9e28b0c67b9df90fca4d1c1b3-Abstract-Conference.html)

**Abstract**:

Learning from label proportions (LLP) is a generalization of supervised learning in which the training data is available as sets or bags of feature-vectors (instances) along with the average instance-label of each bag. The goal is to train a good instance classifier. While most previous works on LLP have focused on training models on such training data, computational learnability of LLP was onlyrecently explored by Saket (2021, 2022) who showed worst case intractability of properly learning linear threshold functions (LTFs) from label proportions. However, their work did not rule out efficient algorithms for this problem for natural distributions.In this work we show that it is indeed possible to efficiently learn LTFs using LTFs when given access to random bags of some label proportion in which feature-vectors are, conditioned on their labels, independently sampled from a Gaussian distribution $N(µ, Σ)$. Our work shows that a certain matrix – formed using covariances of the differences of feature-vectors sampled from the bags with and without replacement – necessarily has its principal component, after a transformation, in the direction of the normal vector of the LTF. Our algorithm estimates the means and covariance matrices using subgaussian concentration bounds which we show can be applied to efficiently sample bags for approximating the normal direction. Using this in conjunction with novel generalization error bounds in the bag setting, we show that a low error hypothesis LTF can be identified. For some special cases of the $N(0, I)$ distribution we provide a simpler mean estimation based algorithm. We include an experimental evaluation of our learning algorithms along with a comparison with those of Saket (2021, 2022) and random LTFs, demonstrating the effectiveness of our techniques.

----

## [2909] A new perspective on building efficient and expressive 3D equivariant graph neural networks

**Authors**: *Weitao Du, Yuanqi Du, Limei Wang, Dieqiao Feng, Guifeng Wang, Shuiwang Ji, Carla P. Gomes, Zhi-Ming Ma*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d212c6c26603c0eb3c9a6b6432386a1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d212c6c26603c0eb3c9a6b6432386a1f-Abstract-Conference.html)

**Abstract**:

Geometric deep learning enables the encoding of physical symmetries in modeling 3D objects. Despite rapid progress in encoding 3D symmetries into Graph Neural Networks (GNNs), a comprehensive evaluation of the expressiveness of these network architectures through a local-to-global analysis lacks today. In this paper, we propose a local hierarchy of 3D isomorphism to evaluate the expressive power of equivariant GNNs and investigate the process of representing global geometric information from local patches. Our work leads to two crucial modules for designing expressive and efficient geometric GNNs; namely local substructure encoding (\textbf{LSE}) and frame transition encoding (\textbf{FTE}). To demonstrate the applicability of our theory, we propose LEFTNet which effectively implements these modules and achieves state-of-the-art performance on both scalar-valued and vector-valued molecular property prediction tasks. We further point out future design space for 3D equivariant graph neural networks. Our codes are available at \url{https://github.com/yuanqidu/LeftNet}.

----

## [2910] Scalable Fair Influence Maximization

**Authors**: *Xiaobin Rui, Zhixiao Wang, Jiayu Zhao, Lichao Sun, Wei Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d242dafdb2c5407ae420bc54c9325fdf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d242dafdb2c5407ae420bc54c9325fdf-Abstract-Conference.html)

**Abstract**:

Given a graph $G$, a community structure $\mathcal{C}$, and a budget $k$, the fair influence maximization problem aims to select a seed set $S$ ($|S|\leq k$) that maximizes the influence spread while narrowing the influence gap between different communities. While various fairness notions exist, the welfare fairness notion, which balances fairness level and influence spread, has shown promising effectiveness. However, the lack of efficient algorithms for optimizing the welfare fairness objective function restricts its application to small-scale networks with only a few hundred nodes. In this paper, we adopt the objective function of welfare fairness to maximize the exponentially weighted summation over the influenced fraction of all communities. We first introduce an unbiased estimator for the fractional power of the arithmetic mean. Then, by adapting the reverse influence sampling (RIS) approach, we convert the optimization problem to a weighted maximum coverage problem. We also analyze the number of reverse reachable sets needed to approximate the fair influence at a high probability. Further, we present an efficient algorithm that guarantees $1-1/e - \varepsilon$ approximation.

----

## [2911] Calibrate and Boost Logical Expressiveness of GNN Over Multi-Relational and Temporal Graphs

**Authors**: *Yeyuan Chen, Dingmin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2706f9149856b6f7016ebf270dd9f25-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2706f9149856b6f7016ebf270dd9f25-Abstract-Conference.html)

**Abstract**:

As a powerful framework for graph representation learning, Graph Neural Networks (GNNs) have garnered significant attention in recent years. However, to the best of our knowledge, there has been no formal analysis of the logical expressiveness of GNNs as Boolean node classifiers over multi-relational graphs, where each edge carries a specific relation type. In this paper, we investigate $\mathcal{FOC}_2$, a fragment of first-order logic with two variables and counting quantifiers. On the negative side, we demonstrate that the R$^2$-GNN architecture, which extends the local message passing GNN by incorporating global readout, fails to capture $\mathcal{FOC}_2$ classifiers in the general case. Nevertheless, on the positive side, we establish that R$^2$-GNNs models are equivalent to $\mathcal{FOC}_2$ classifiers under certain restricted yet reasonable scenarios. To address the limitations of R$^2$-GNNs regarding expressiveness, we propose a simple graph transformation technique, akin to a preprocessing step, which can be executed in linear time. This transformation enables R$^2$-GNNs to effectively capture any $\mathcal{FOC}_2$ classifiers when applied to the "transformed" input graph. Moreover, we extend our analysis of expressiveness and graph transformation to temporal graphs, exploring several temporal GNN architectures and providing an expressiveness hierarchy for them. To validate our findings, we implement R$^2$-GNNs and the graph transformation technique and conduct empirical tests in node classification tasks against various well-known GNN architectures that support multi-relational or temporal graphs. Our experimental results consistently demonstrate that R$^2$-GNN with the graph transformation outperforms the baseline methods on both synthetic and real-world datasets

----

## [2912] Task Arithmetic in the Tangent Space: Improved Editing of Pre-Trained Models

**Authors**: *Guillermo Ortiz-Jiménez, Alessandro Favero, Pascal Frossard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d28077e5ff52034cd35b4aa15320caea-Abstract-Conference.html)

**Abstract**:

Task arithmetic has recently emerged as a cost-effective and scalable approach to edit pre-trained models directly in weight space: By adding the fine-tuned weights of different tasks, the model's performance can be improved on these tasks, while negating them leads to task forgetting. Yet, our understanding of the effectiveness of task arithmetic and its underlying principles remains limited. We present a comprehensive study of task arithmetic in vision-language models and show that weight disentanglement is the crucial factor that makes it effective. This property arises during pre-training and manifests when distinct directions in weight space govern separate, localized regions in function space associated with the tasks. Notably, we show that fine-tuning models in their tangent space by linearizing them amplifies weight disentanglement. This leads to substantial performance improvements across multiple task arithmetic benchmarks and diverse models. Building on these findings, we provide theoretical and empirical analyses of the neural tangent kernel (NTK) of these models and establish a compelling link between task arithmetic and the spatial localization of the NTK eigenfunctions. Overall, our work uncovers novel insights into the fundamental mechanisms of task arithmetic and offers a more reliable and effective approach to edit pre-trained models through the NTK linearization.

----

## [2913] ParaFuzz: An Interpretability-Driven Technique for Detecting Poisoned Samples in NLP

**Authors**: *Lu Yan, Zhuo Zhang, Guanhong Tao, Kaiyuan Zhang, Xuan Chen, Guangyu Shen, Xiangyu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2b752ed4726286a4b488ae16e091d64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2b752ed4726286a4b488ae16e091d64-Abstract-Conference.html)

**Abstract**:

Backdoor attacks have emerged as a prominent threat to natural language processing (NLP) models, where the presence of specific triggers in the input can lead poisoned models to misclassify these inputs to predetermined target classes. Current detection mechanisms are limited by their inability to address more covert backdoor strategies, such as style-based attacks. In this work, we propose an innovative test-time poisoned sample detection framework that hinges on the interpretability of model predictions, grounded in the semantic meaning of inputs.We contend that triggers (e.g., infrequent words) are not supposed to fundamentally alter the underlying semantic meanings of poisoned samples as they want to stay stealthy. Based on this observation, we hypothesize that while the model's predictions for paraphrased clean samples should remain stable, predictions for poisoned samples should revert to their true labels upon the mutations applied to triggers during the paraphrasing process.We employ ChatGPT, a state-of-the-art large language model, as our paraphraser and formulate the trigger-removal task as a prompt engineering problem. We adopt fuzzing, a technique commonly used for unearthing software vulnerabilities, to discover optimal paraphrase prompts that can effectively eliminate triggers while concurrently maintaining input semantics.Experiments on 4 types of backdoor attacks, including the subtle style backdoors, and 4 distinct datasets demonstrate that our approach surpasses baseline methods, including STRIP, RAP, and ONION, in precision and recall.

----

## [2914] DiViNeT: 3D Reconstruction from Disparate Views using Neural Template Regularization

**Authors**: *Aditya Vora, Akshay Gadi Patil, Hao Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2bdcd4f51eea138365af22b50f3bf0a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2bdcd4f51eea138365af22b50f3bf0a-Abstract-Conference.html)

**Abstract**:

We present a volume rendering-based neural surface reconstruction method that takes as few as three disparate RGB images as input. Our key idea is to regularize the reconstruction, which is severely ill-posed and leaving significant gaps between the sparse views, by learning a set of neural templates that act as surface priors. Our method, coined DiViNet, operates in two stages. The first stage learns the templates, in the form of 3D Gaussian functions, across different scenes, without 3D supervision. In the reconstruction stage, our predicted templates serve as anchors to help “stitch” the surfaces over sparse regions. We demonstrate that our approach is not only able to complete the surface geometry but also reconstructs surface details to a reasonable extent from few disparate input views. On the DTU and BlendedMVS datasets, our approach achieves the best reconstruction quality among existing methods in the presence of such sparse views and performs on par, if not better, with competing methods when dense views are employed as inputs.

----

## [2915] Efficient Model-Free Exploration in Low-Rank MDPs

**Authors**: *Zakaria Mhammedi, Adam Block, Dylan J. Foster, Alexander Rakhlin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2dc4d6c7b102d05f111c02a32e7c6bc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2dc4d6c7b102d05f111c02a32e7c6bc-Abstract-Conference.html)

**Abstract**:

A major challenge in reinforcement learning is to develop practical, sample-efficient algorithms for exploration in high-dimensional domains where generalization and function approximation is required. Low-Rank Markov Decision Processes---where transition probabilities admit a low-rank factorization based on an unknown feature embedding---offer a simple, yet expressive framework for RL with function approximation, yet existing algorithms either (1) are computationally intractable, or (2) require restrictive statistical assumptions such as latent variable structure or access to model-based function approximation. In this work, we propose the first provably sample-efficient algorithm for exploration in Low-Rank MDPs that is both computationally efficient and model-free, allowing for general function approximation while requiring no structural assumptions beyond a reachability condition that we show is substantially weaker than that assumed in prior work. Our algorithm, SpanRL, uses the notion of a barycentric spanner for the feature embedding as an efficiently computable basis for exploration, performing efficient spanner computation by interleaving representation learning and policy optimization subroutines. Our analysis---which is appealingly simple and modular---carefully combines several techniques, including a new approach to error-tolerant barycentric spanner computation, and a new analysis of a certain minimax representation learning objective found in prior work.

----

## [2916] Convex-Concave Zero-Sum Stochastic Stackelberg Games

**Authors**: *Denizalp Goktas, Arjun Prakash, Amy Greenwald*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2f6f1dfbf9cd89a78c5a58ef0dec245-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2f6f1dfbf9cd89a78c5a58ef0dec245-Abstract-Conference.html)

**Abstract**:

Zero-sum Markov Stackelberg games can be used to model myriad problems, in domains ranging from economics to human robot interaction. In this paper, we develop policy gradient methods that solve these games in continuous state and action settings using noisy gradient estimates computed from observed trajectories of play. When the games are convex-concave, we prove that our algorithms converge to Stackelberg equilibrium in polynomial time. We also show that reach-avoid problems are naturally modeled as convex-concave zero-sum Markov Stackelberg games, and that Stackelberg equilibrium policies are more effective than their Nash counterparts in these problems.

----

## [2917] Near-Linear Time Algorithm for the Chamfer Distance

**Authors**: *Ainesh Bakshi, Piotr Indyk, Rajesh Jayaram, Sandeep Silwal, Erik Waingarten*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d2fe3a5711a6d488da9e9a78b84ee24c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d2fe3a5711a6d488da9e9a78b84ee24c-Abstract-Conference.html)

**Abstract**:

For any two point sets $A,B \subset \mathbb{R}^d$ of size up to $n$, the Chamfer distance from $A$ to $B$ is defined as $\texttt{CH}(A,B)=\sum_{a \in A} \min_{b \in B} d_X(a,b)$, where $d_X$ is the underlying distance measure (e.g., the Euclidean or Manhattan distance). The Chamfer distance is a popular measure of dissimilarity between point clouds, used in many machine learning, computer vision, and graphics applications, and admits a straightforward $O(d n^2)$-time brute force algorithm. Further, Chamfer distance is often used as a  proxy for the more computationally demanding Earth-Mover (Optimal Transport) Distance. However, the \emph{quadratic} dependence on $n$ in the running time  makes the naive approach intractable for large datasets.We overcome this bottleneck and present the first $(1+\epsilon)$-approximate algorithm for estimating Chamfer distance with a near-linear running time. Specifically, our algorithm runs in time $O(nd \log (n)/\epsilon^2)$ and is implementable. Our experiments demonstrate that it is both accurate and fast on large high-dimensional datasets. We believe that our algorithm will open new avenues for analyzing large high-dimensional point clouds. We also give evidence that if the goal is to report a $(1+\epsilon)$-approximate mapping from $A$ to $B$ (as opposed to just its value), then any sub-quadratic  time algorithm is unlikely to exist.

----

## [2918] Double Pessimism is Provably Efficient for Distributionally Robust Offline Reinforcement Learning: Generic Algorithm and Robust Partial Coverage

**Authors**: *Jose H. Blanchet, Miao Lu, Tong Zhang, Han Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d31b005d817e9c635ec8ffb0fb90190e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d31b005d817e9c635ec8ffb0fb90190e-Abstract-Conference.html)

**Abstract**:

We study distributionally robust offline reinforcement learning (RL), which seeks to find an optimal robust policy purely from an offline dataset that can perform well in perturbed environments. We propose a generic algorithm framework Doubly Pessimistic Model-based Policy Optimization ($\texttt{P}^2\texttt{MPO}$) for robust offline RL, which features a novel combination of a flexible model estimation subroutine and a doubly pessimistic policy optimization step. Here the double pessimism principle is crucial to overcome the distribution shift incurred by i) the mismatch between behavior policy and the family of target policies; and ii) the perturbation of the nominal model. Under certain accuracy assumptions on the model estimation subroutine, we show that $\texttt{P}^2\texttt{MPO}$ is provably sample-efficient with robust partial coverage data, which means that the offline dataset has good coverage of the distributions induced by the optimal robust policy and perturbed models around the nominal model. By tailoring specific model estimation subroutines for concrete examples including tabular Robust Markov Decision Process (RMDP), factored RMDP, and RMDP with kernel and neural function approximations, we show that $\texttt{P}^2\texttt{MPO}$ enjoys a $\tilde{\mathcal{O}}(n^{-1/2})$ convergence rate, where $n$ is the number of trajectories in the offline dataset. Notably, these models, except for the tabular case, are first identified and proven tractable by this paper. To the best of our knowledge, we first propose a general learning principle --- double pessimism --- for robust offline RL and show that it is provably efficient in the context of general function approximations.

----

## [2919] StyleDrop: Text-to-Image Synthesis of Any Style

**Authors**: *Kihyuk Sohn, Lu Jiang, Jarred Barber, Kimin Lee, Nataniel Ruiz, Dilip Krishnan, Huiwen Chang, Yuanzhen Li, Irfan Essa, Michael Rubinstein, Yuan Hao, Glenn Entis, Irina Blok, Daniel Castro Chin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d33b177b69425e7685b0b1c05bd2a5e4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d33b177b69425e7685b0b1c05bd2a5e4-Abstract-Conference.html)

**Abstract**:

Pre-trained large text-to-image models synthesize impressive images with an appropriate use of text prompts. However, ambiguities inherent in natural language, and out-of-distribution effects make it hard to synthesize arbitrary image styles, leveraging a specific design pattern, texture or material. In this paper, we introduce StyleDrop, a method that enables the synthesis of images that faithfully follow a specific style using a text-to-image model. StyleDrop is extremely versatile and captures nuances and details of a user-provided style, such as color schemes, shading, design patterns, and local and global effects. StyleDrop works by efficiently learning a new style by fine-tuning very few trainable parameters (less than 1\% of total model parameters), and improving the quality via iterative training with either human or automated feedback. Better yet, StyleDrop is able to deliver impressive results even when the user supplies only a single image specifying the desired style. An extensive study shows that, for the task of style tuning text-to-image models, StyleDrop on Muse convincingly outperforms other methods, including DreamBooth and textual inversion on Imagen or Stable Diffusion. More results are available at our project website: https://styledrop.github.io.

----

## [2920] Random Cuts are Optimal for Explainable k-Medians

**Authors**: *Konstantin Makarychev, Liren Shan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d3408794e41dd23e34634344d662f5e9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d3408794e41dd23e34634344d662f5e9-Abstract-Conference.html)

**Abstract**:

We show that the RandomCoordinateCut algorithm gives the optimal competitive ratio for explainable $k$-medians in $\ell_1$. The problem of explainable $k$-medians was introduced by Dasgupta, Frost, Moshkovitz, and Rashtchian in 2020. Several groups of authors independently proposed a simple polynomial-time randomized algorithm for the problem and showed that this algorithm is $O(\log k \log\log k)$ competitive.  We provide a tight analysis of the algorithm and prove that its competitive ratio is upper bounded by $2\ln k+2$. This bound matches the $\Omega(\log k)$ lower bound by Dasgupta et al (2020).

----

## [2921] Order Matters in the Presence of Dataset Imbalance for Multilingual Learning

**Authors**: *Dami Choi, Derrick Xin, Hamid Dadkhahi, Justin Gilmer, Ankush Garg, Orhan Firat, Chih-Kuan Yeh, Andrew M. Dai, Behrooz Ghorbani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d346609ec2fefd3938c898a0dda4a480-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d346609ec2fefd3938c898a0dda4a480-Abstract-Conference.html)

**Abstract**:

In this paper, we empirically study the optimization dynamics of multi-task learning, particularly focusing on those that govern a collection of tasks with significant data imbalance. We present a simple yet effective method of pre-training on high-resource tasks, followed by fine-tuning on a mixture of high/low-resource tasks. We provide a thorough empirical study and analysis of this method's benefits showing that it achieves consistent improvements relative to the performance trade-off profile of standard static weighting. We analyze under what data regimes this method is applicable and show its improvements empirically in neural machine translation (NMT) and multi-lingual language modeling.

----

## [2922] Optimizing Prompts for Text-to-Image Generation

**Authors**: *Yaru Hao, Zewen Chi, Li Dong, Furu Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d346d91999074dd8d6073d4c3b13733b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d346d91999074dd8d6073d4c3b13733b-Abstract-Conference.html)

**Abstract**:

Well-designed prompts can guide text-to-image models to generate amazing images. However, the performant prompts are often model-specific and misaligned with user input. Instead of laborious human engineering, we propose prompt adaptation, a general framework that automatically adapts original user input to model-preferred prompts. Specifically, we first perform supervised fine-tuning with a pretrained language model on a small collection of manually engineered prompts. Then we use reinforcement learning to explore better prompts. We define a reward function that encourages the policy to generate more aesthetically pleasing images while preserving the original user intentions. Experimental results on Stable Diffusion show that our method outperforms manual prompt engineering in terms of both automatic metrics and human preference ratings. Moreover, reinforcement learning further boosts performance, especially on out-of-domain prompts.

----

## [2923] Improved Bayes Risk Can Yield Reduced Social Welfare Under Competition

**Authors**: *Meena Jagadeesan, Michael I. Jordan, Jacob Steinhardt, Nika Haghtalab*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d3602fc92fb8b9e0d55356c9e8815e2b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d3602fc92fb8b9e0d55356c9e8815e2b-Abstract-Conference.html)

**Abstract**:

As the scale of machine learning models increases, trends such as scaling laws anticipate consistent downstream improvements in predictive accuracy. However, these trends take the perspective of a single model-provider in isolation, while in reality providers often compete with each other for users. In this work, we demonstrate that competition can fundamentally alter the behavior of these scaling trends, even causing overall predictive accuracy across users to be non-monotonic or decreasing with scale. We define a model of competition for classification tasks, and use data representations as a lens for studying the impact of increases in scale. We find many settings where improving data representation quality (as measured by Bayes risk) decreases the overall predictive accuracy across users (i.e., social welfare) for a marketplace of competing model-providers. Our examples range from closed-form formulas in simple settings to simulations with pretrained representations on CIFAR-10. At a conceptual level, our work suggests that favorable scaling trends for individual model-providers need not translate to downstream improvements in  social welfare in marketplaces with  multiple model providers.

----

## [2924] Inverse Dynamics Pretraining Learns Good Representations for Multitask Imitation

**Authors**: *David Brandfonbrener, Ofir Nachum, Joan Bruna*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference.html)

**Abstract**:

In recent years, domains such as natural language processing and image recognition have popularized the paradigm of using large datasets to pretrain representations that can be effectively transferred to downstream tasks. In this work we evaluate how such a paradigm should be done in imitation learning, where both pretraining and finetuning data are trajectories collected by experts interacting with an unknown environment. Namely, we consider a setting where the pretraining corpus consists of multitask demonstrations and the task for each demonstration is set by an unobserved latent context variable. The goal is to use the pretraining corpus to learn a low dimensional representation of the high dimensional (e.g., visual) observation space which can be transferred to a novel context for finetuning on a limited dataset of demonstrations. Among a variety of possible pretraining objectives, we argue that inverse dynamics modeling -- i.e.,  predicting an action given the observations appearing before and after it in the demonstration -- is well-suited to this setting. We provide empirical evidence of this claim through evaluations on a variety of simulated visuomotor manipulation problems. While previous work has attempted various theoretical explanations regarding the benefit of inverse dynamics modeling, we find that these arguments are insufficient to explain the empirical advantages often observed in our settings, and so we derive a novel analysis using a simple but general environment model.

----

## [2925] Exploiting hidden structures in non-convex games for convergence to Nash equilibrium

**Authors**: *Iosif Sakos, Emmanouil V. Vlatakis-Gkaragkounis, Panayotis Mertikopoulos, Georgios Piliouras*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d37c9ad425fe5b65304d500c6edcba00-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d37c9ad425fe5b65304d500c6edcba00-Abstract-Conference.html)

**Abstract**:

A wide array of modern machine learning applications – from adversarial models to multi-agent reinforcement learning – can be formulated as non-cooperative games whose Nash equilibria represent the system’s desired operational states. Despite having a highly non-convex loss landscape, many cases of interest possess a latent convex structure that could potentially be leveraged to yield convergence to an equilibrium. Driven by this observation, our paper proposes a flexible first-order method that successfully exploits such “hidden structures” and achieves convergence under minimal assumptions for the transformation connecting the players’ control variables to the game’s latent, convex-structured layer. The proposed method – which we call preconditioned hidden gradient descent (PHGD) – hinges on a judiciously chosen gradient preconditioning scheme related to natural gradient methods. Importantly, we make no separability assumptions for the game’s hidden structure, and we provide explicit convergence rate guarantees for both deterministic and stochastic environments.

----

## [2926] Secure Out-of-Distribution Task Generalization with Energy-Based Models

**Authors**: *Shengzhuang Chen, Long-Kai Huang, Jonathan Richard Schwarz, Yilun Du, Ying Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d39e3ae9a11b79691709a7a6e06a63d9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d39e3ae9a11b79691709a7a6e06a63d9-Abstract-Conference.html)

**Abstract**:

The success of meta-learning on out-of-distribution (OOD) tasks in the wild has proved to be hit-and-miss.To safeguard the generalization capability of the meta-learned prior knowledge to OOD tasks, in particularly safety-critical applications, necessitates detection of an OOD task followed by adaptation of the task towards the prior. Nonetheless, the reliability of estimated uncertainty on OOD tasks by existing Bayesian meta-learning methods is restricted by incomplete coverage of the feature distribution shift and insufficient expressiveness of the meta-learned prior. Besides, they struggle to adapt an OOD task, running parallel to the line of cross-domain task adaptation solutions which are vulnerable to overfitting.To this end, we build a single coherent framework that supports both detection and adaptation of OOD tasks, while remaining compatible with off-the-shelf meta-learning backbones. The proposed Energy-Based Meta-Learning (EBML) framework learns to characterize any arbitrary meta-training task distribution with the composition of two expressive neural-network-based energy functions. We deploy the sum of the two energy functions, being proportional to the joint distribution of a task, as a reliable score for detecting OOD tasks; during meta-testing, we adapt the OOD task to in-distribution tasks by energy minimization.Experiments on four regression and classification datasets  demonstrate the effectiveness of our proposal.

----

## [2927] Autodecoding Latent 3D Diffusion Models

**Authors**: *Evangelos Ntavelis, Aliaksandr Siarohin, Kyle Olszewski, Chaoyang Wang, Luc Van Gool, Sergey Tulyakov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d3b93537b521f15613524415dfe43f37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d3b93537b521f15613524415dfe43f37-Abstract-Conference.html)

**Abstract**:

Diffusion-based methods have shown impressive visual results in the text-to-image domain. They first learn a latent space using an autoencoder, then run a denoising process on the bottleneck to generate new samples. However, learning an autoencoder requires substantial data in the target domain. Such data is scarce for 3D generation, prohibiting the learning of large-scale diffusion models for 3D synthesis. We present a novel approach to the generation of static and articulated 3D assets that has a 3D autodecoder at its core. The 3D autodecoder framework embeds properties learned from the target dataset in the latent space, which can then be decoded into a volumetric representation for rendering view-consistent appearance and geometry. We then identify the appropriate intermediate volumetric latent space, and introduce robust normalization and de-normalization operations to learn a 3D diffusion from 2D images or monocular videos of rigid or articulated objects. Our approach is flexible enough to use either existing camera supervision or no camera information at all -- instead efficiently learning it during training. Our evaluations demonstrate that our generation results outperform state-of-the-art alternatives on various benchmark datasets and metrics, including multi-view image datasets of synthetic objects, real in-the-wild videos of moving people, and a large-scale, real video dataset of static objects.

----

## [2928] Physion++: Evaluating Physical Scene Understanding that Requires Online Inference of Different Physical Properties

**Authors**: *Hsiao-Yu Tung, Mingyu Ding, Zhenfang Chen, Daniel Bear, Chuang Gan, Josh Tenenbaum, Dan Yamins, Judith E. Fan, Kevin A. Smith*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d3e8011c912e651ab2a76e7935a1e464-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d3e8011c912e651ab2a76e7935a1e464-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

General physical scene understanding requires more than simply localizing and recognizing objects -- it requires knowledge that objects can have different latent properties (e.g., mass or elasticity), and that those properties affect the outcome of physical events. While there has been great progress in physical and video prediction models in recent years, benchmarks to test their performance typically do not require an understanding that objects have individual physical properties, or at best test only those properties that are directly observable (e.g., size or color). This work proposes a novel dataset and benchmark, termed Physion++, that rigorously evaluates visual physical prediction in artificial systems under circumstances where those predictions rely on accurate estimates of the latent physical properties of objects in the scene. Specifically, we test scenarios where accurate prediction relies on estimates of properties such as mass, friction, elasticity, and deformability, and where the values of those properties can only be inferred by observing how objects move and interact with other objects or fluids. We evaluate the performance of a number of state-of-the-art prediction models that span a variety of levels of learning vs. built-in knowledge, and compare that performance to a set of human predictions. We find that models that have been trained using standard regimes and datasets do not spontaneously learn to make inferences about latent properties, but also that models that encode objectness and physical states tend to make better predictions. However, there is still a huge gap between all models and human performance, and all models' predictions correlate poorly with those made by humans, suggesting that no state-of-the-art model is learning to make physical predictions in a human-like way. These results show that current deep learning models that succeed in some settings nevertheless fail to achieve human-level physical prediction in other cases, especially those where latent property inference is required. Project page: https://dingmyu.github.io/physion_v2/

----

## [2929] HA-ViD: A Human Assembly Video Dataset for Comprehensive Assembly Knowledge Understanding

**Authors**: *Hao Zheng, Regina Lee, Yuqian Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d40e6e4b3ee6c24f2bf2cb72c2412f4b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d40e6e4b3ee6c24f2bf2cb72c2412f4b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Understanding comprehensive assembly knowledge from videos is critical for futuristic ultra-intelligent industry. To enable technological breakthrough, we present HA-ViD â€“ the first human assembly video dataset that features representative industrial assembly scenarios, natural procedural knowledge acquisition process, and consistent human-robot shared annotations. Specifically, HA-ViD captures diverse collaboration patterns of real-world assembly, natural human behaviors and learning progression during assembly, and granulate action annotations to subject, action verb, manipulated object, target object, and tool. We provide 3222 multi-view and multi-modality videos), 1.5M frames, 96K temporal labels and 2M spatial labels. We benchmark four foundational video understanding tasks: action recognition, action segmentation, object detection and multi-object tracking. Importantly, we analyze their performance and the further reasoning steps for comprehending knowledge in assembly progress, process efficiency, task collaboration, skill parameters and human intention. Details of HA-ViD is available at: https://iai-hrc.github.io/ha-vid.

----

## [2930] Classical Simulation of Quantum Circuits: Parallel Environments and Benchmark

**Authors**: *Xiao-Yang Liu, Zeliang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d41b70011dd21ec3de5e019302279551-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d41b70011dd21ec3de5e019302279551-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Google's  quantum supremacy announcement has received broad questions from academia and industry due to the debatable estimate of 10,000 years' running time for the classical simulation task on the Summit supercomputer. Has quantum supremacy already come? Or will it come in one or two decades later? To avoid hasty advertisements of  quantum supremacy by tech giants or quantum startups and eliminate the cost of dedicating a team to the classical simulation task, we advocate an open-source approach to maintain a trustable benchmark performance.  In this paper, we take a reinforcement learning approach for the classical simulation of quantum circuits and demonstrate its great potential by reporting an estimated simulation time of less than 4 days, a speedup of 5.40x over the state-of-the-art method.  Specifically, we formulate the classical simulation task as a tensor network contraction ordering problem using the K-spin Ising model and employ a novel Hamiltonina-based reinforcement learning algorithm. Then, we establish standard criteria to evaluate the performance of classical simulation of quantum circuits.  We develop a dozen of massively parallel environments to simulate quantum circuits.  We open-source our parallel gym environments and benchmarks. We hope the AI/ML community and quantum physics community will collaborate to maintain reference curves for validating an unequivocal first demonstration of empirical quantum supremacy.

----

## [2931] A General Framework for Robust G-Invariance in G-Equivariant Networks

**Authors**: *Sophia Sanborn, Nina Miolane*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d42523d621194ba54dda098669645f91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d42523d621194ba54dda098669645f91-Abstract-Conference.html)

**Abstract**:

We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also \textit{complete}. Many commonly used invariant maps\textemdash such as the \texttt{max}\textemdash are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups\textemdash $SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups)\textemdash acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.

----

## [2932] EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models

**Authors**: *Michael Wornow, Rahul Thapa, Ethan Steinberg, Jason A. Fries, Nigam Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d42db1f74df54cb992b3956eb7f15a6f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d42db1f74df54cb992b3956eb7f15a6f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

While the general machine learning (ML) community has benefited from public datasets, tasks, and models, the progress of ML in healthcare has been hampered by a lack of such shared assets. The success of foundation models creates new challenges for healthcare ML by requiring access to shared pretrained models to validate performance benefits. We help address these challenges through three contributions. First, we publish a new dataset, EHRSHOT, which contains de-identified structured data from the electronic health records (EHRs) of 6,739 patients from Stanford Medicine. Unlike MIMIC-III/IV and other popular EHR datasets, EHRSHOT is longitudinal and not restricted to ICU/ED patients. Second, we publish the weights of CLMBR-T-base, a 141M parameter clinical foundation model pretrained on the structured EHR data of 2.57M patients. We are one of the first to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data  (e.g. GatorTron, ClinicalBERT) only work with unstructured text and cannot process the rich, structured data within an EHR. We provide an end-to-end pipeline for the community to validate and build upon its performance. Third, we define 15 few-shot clinical prediction tasks, enabling evaluation of foundation models on benefits such as sample efficiency and task adaptation. Our model and dataset are available via a research data use agreement from here: https://stanfordaimi.azurewebsites.net/. Code to reproduce our results is available here: https://github.com/som-shahlab/ehrshot-benchmark.

----

## [2933] SEVA: Leveraging sketches to evaluate alignment between human and machine visual abstraction

**Authors**: *Kushin Mukherjee, Holly Huey, Xuanchen Lu, Yael Vinker, Rio Aguina-Kang, Ariel Shamir, Judith E. Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d43621ff2dfe39d298dcd4a41937c912-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d43621ff2dfe39d298dcd4a41937c912-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Sketching is a powerful tool for creating abstract images that are sparse but meaningful. Sketch understanding poses fundamental challenges for general-purpose vision algorithms because it requires robustness to the sparsity of sketches relative to natural visual inputs and because it demands tolerance for semantic ambiguity, as sketches can reliably evoke multiple meanings. While current vision algorithms have achieved high performance on a variety of visual tasks, it remains unclear to what extent they understand sketches in a human-like way. Here we introduce $\texttt{SEVA}$, a new benchmark dataset containing approximately 90K human-generated sketches of 128 object concepts produced under different time constraints, and thus systematically varying in sparsity. We evaluated a suite of state-of-the-art vision algorithms on their ability to correctly identify the target concept depicted in these sketches and to generate responses that are strongly aligned with human response patterns on the same sketch recognition task. We found that vision algorithms that better predicted human sketch recognition performance also better approximated human uncertainty about sketch meaning, but there remains a sizable gap between model and human response patterns. To explore the potential of models that emulate human visual abstraction in generative tasks, we conducted further evaluations of a recently developed sketch generation algorithm (Vinker et al., 2022) capable of generating sketches that vary in sparsity. We hope that public release of this dataset and evaluation protocol will catalyze progress towards algorithms with enhanced capacities for human-like visual abstraction.

----

## [2934] State2Explanation: Concept-Based Explanations to Benefit Agent Learning and User Understanding

**Authors**: *Devleena Das, Sonia Chernova, Been Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html)

**Abstract**:

As more non-AI experts use complex AI systems for daily tasks, there has been an increasing effort to develop methods that produce explanations of AI decision making that are understandable by non-AI experts. Towards this effort, leveraging higher-level concepts and producing concept-based explanations have become a popular method. Most concept-based explanations have been developed for classification techniques, and we posit that the few existing methods for sequential decision making are limited in scope. In this work, we first contribute a desiderata for defining ``concepts'' in sequential decision making settings. Additionally, inspired by the Protege Effect which states explaining knowledge often reinforces one's self-learning, we explore how concept-based explanations of an RL agent's decision making can in turn improve the agent's learning rate, as well as improve end-user understanding of the agent's decision making. To this end, we contribute a unified framework, State2Explanation (S2E), that involves learning a joint embedding model between state-action pairs and concept-based explanations, and leveraging such learned model to both (1) inform reward shaping during an agent's training, and (2) provide explanations to end-users at deployment for improved task performance. Our experimental validations, in Connect 4 and Lunar Lander, demonstrate the success of S2E in providing a dual-benefit, successfully informing reward shaping and improving agent learning rate, as well as significantly improving end user task performance at deployment time.

----

## [2935] Machine learning detects terminal singularities

**Authors**: *Tom Coates, Alexander M. Kasprzyk, Sara Veneziale*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d453490ada2b1991852f053fbd213a6a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d453490ada2b1991852f053fbd213a6a-Abstract-Conference.html)

**Abstract**:

Algebraic varieties are the geometric shapes defined by systems of polynomial equations; they are ubiquitous across mathematics and science. Amongst these algebraic varieties are Q-Fano varieties: positively curved shapes which have Q-factorial terminal singularities. Q-Fano varieties are of fundamental importance in geometry as they are `atomic pieces’ of more complex shapes – the process of breaking a shape into simpler pieces in this sense is called the Minimal Model Programme.Despite their importance, the classification of Q-Fano varieties remains unknown. In this paper we demonstrate that machine learning can be used to understand this classification. We focus on eight-dimensional positively-curved algebraic varieties that have toric symmetry and Picard rank two, and develop a neural network classifier that predicts with 95% accuracy whether or not such an algebraic variety is Q-Fano. We use this to give a first sketch of the landscape of Q-Fano varieties in dimension eight.How the neural network is able to detect Q-Fano varieties with such accuracy remains mysterious, and hints at some deep mathematical theory waiting to be uncovered. Furthermore, when visualised using the quantum period, an invariant that has played an important role in recent theoretical developments, we observe that the classification as revealed by ML appears to fall within a bounded region, and is stratified by the Fano index. This suggests that it may be possible to state and prove conjectures on completeness in the future.Inspired by the ML analysis, we formulate and prove a new global combinatorial criterion for a positively curved toric variety of Picard rank two to have terminal singularities. Together with the first sketch of the landscape of Q-Fano varieties in higher dimensions, this gives strong new evidence that machine learning can be an essential tool in developing mathematical conjectures and accelerating theoretical discovery.

----

## [2936] Efficient Diffusion Policies For Offline Reinforcement Learning

**Authors**: *Bingyi Kang, Xiao Ma, Chao Du, Tianyu Pang, Shuicheng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d45e0bfb5a39477d56b55c0824200008-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d45e0bfb5a39477d56b55c0824200008-Abstract-Conference.html)

**Abstract**:

Offline reinforcement learning (RL) aims to learn optimal policies from offline datasets, where the parameterization of policies is crucial but often overlooked. Recently, Diffsuion-QL significantly boosts the performance of offline RL by representing a policy with a diffusion model, whose success relies on a parametrized Markov Chain with hundreds of steps for sampling. However, Diffusion-QL suffers from two critical limitations. 1) It is computationally inefficient to forward and backward through the whole Markov chain during training. 2) It is incompatible with maximum likelihood-based RL algorithms (e.g., policy gradient methods) as the likelihood of diffusion models is intractable. Therefore, we propose efficient diffusion policy (EDP) to overcome these two challenges. EDP approximately constructs actions from corrupted ones at training to avoid running the sampling chain. We conduct extensive experiments on the D4RL benchmark. The results show that EDP can reduce the diffusion policy training time from 5 days to 5 hours on gym-locomotion tasks. Moreover, we show that EDP is compatible with various offline RL algorithms (TD3, CRR, and IQL) and achieves new state-of-the-art on D4RL by large margins over previous methods.

----

## [2937] Selective Sampling and Imitation Learning via Online Regression

**Authors**: *Ayush Sekhari, Karthik Sridharan, Wen Sun, Runzhe Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d470d6e007a19ff1666386562c77517c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d470d6e007a19ff1666386562c77517c-Abstract-Conference.html)

**Abstract**:

We consider the problem of Imitation Learning (IL) by actively querying noisy expert for feedback. While imitation learning has been empirically successful, much of prior work assumes access to noiseless expert feedback which is not practical in many applications. In fact, when one only has access to noisy expert feedback, algorithms that rely on purely offline data (non-interactive IL) can be shown to need a prohibitively large number of samples to be successful. In contrast, in this work, we provide an interactive algorithm for IL that uses selective sampling to actively query the noisy expert for feedback. Our contributions are twofold: First,  we provide a new selective sampling algorithm that works with general function classes and multiple actions, and obtains the best-known bounds for the regret and the number of queries. Next, we extend this analysis to the problem of IL with noisy expert feedback and provide a new IL algorithm that  makes limited queries.  Our algorithm for selective sampling leverages function approximation, and relies on an online regression oracle w.r.t.~the given model class to predict actions, and to decide whether to query the expert for its label. On the theoretical side, the regret bound of our algorithm is upper bounded by the regret of the online regression oracle, while the query complexity additionally depends on the eluder dimension of the model class. We complement this with a  lower bound that demonstrates that our results are tight. We extend our selective sampling algorithm for IL with general function approximation and provide bounds on both the regret and the number of queries made to the noisy expert. A key novelty here is that our regret and query complexity bounds only depend on the number of times the optimal policy (and not the noisy expert, or the learner) go to states that have a small margin.

----

## [2938] CamoPatch: An Evolutionary Strategy for Generating Camoflauged Adversarial Patches

**Authors**: *Phoenix Williams, Ke Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d482f1362bd6a8448d7c35e717c7063a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d482f1362bd6a8448d7c35e717c7063a-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) have demonstrated vulnerabilities to adversarial examples, which raises concerns about their reliability in safety-critical applications. While the majority of existing methods generate adversarial examples by making small modifications to the entire image, recent research has proposed a practical alternative known as adversarial patches. Adversarial patches have shown to be highly effective in causing DNNs to misclassify by distorting a localized area (patch) of the image. However, existing methods often produce clearly visible distortions since they do not consider the visibility of the patch. To address this, we propose a novel method for constructing adversarial patches that approximates the appearance of the area it covers. We achieve this by using a set of semi-transparent, RGB-valued circles, drawing inspiration from the computational art community. We utilize an evolutionary strategy to optimize the properties of each shape, and employ a simulated annealing approach to optimize the patch's location. Our approach achieves better or comparable performance to state-of-the-art methods on ImageNet DNN classifiers while achieving a lower $l_2$ distance from the original image. By minimizing the visibility of the patch, this work further highlights the vulnerabilities of DNNs to adversarial patches.

----

## [2939] MADLAD-400: A Multilingual And Document-Level Large Audited Dataset

**Authors**: *Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d49042a5d49818711c401d34172f9900-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce MADLAD-400, a manually audited, general domain 3T token monolingual dataset based on CommonCrawl, spanning 419 languages. We discuss the limitations revealed by self-auditing MADLAD-400, and the role data auditing had in the dataset creation process. We then train and release a 10.7B-parameter multilingual machine translation model on 250 billion tokens covering over 450 languages using publicly available data, and find that it is competitive with models that are significantly larger, and report the results on different domains. In addition, we train a 8B-parameter language model, and assess the results on few-shot translation. We make the baseline models available to the research community.

----

## [2940] Learning Regularized Monotone Graphon Mean-Field Games

**Authors**: *Fengzhuo Zhang, Vincent Y. F. Tan, Zhaoran Wang, Zhuoran Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d4c2f25bf0c33065b7d4fb9be2a9add1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d4c2f25bf0c33065b7d4fb9be2a9add1-Abstract-Conference.html)

**Abstract**:

This paper studies two fundamental problems in regularized Graphon Mean-Field Games (GMFGs). First, we establish the existence of a Nash Equilibrium (NE)  of any $\lambda$-regularized GMFG   (for  $\lambda\geq 0$). This result relies on weaker conditions than previous works analyzing both unregularized GMFGs ($\lambda=0$) and $\lambda$-regularized MFGs, which are special cases of GMFGs. Second, we propose provably efficient algorithms to learn the NE in  weakly monotone GMFGs, motivated by Lasry and Lions (2007). Previous literature either only analyzed continuous-time algorithms or required extra conditions to analyze discrete-time algorithms. In contrast, we design a discrete-time algorithm and derive its convergence rate  solely under  weakly monotone conditions. Furthermore, we develop and analyze the action-value function estimation procedure during the online learning process, which is absent from algorithms for monotone GMFGs. This serves as a sub-module in our optimization algorithm. The efficiency of the designed algorithm is corroborated by empirical evaluations.

----

## [2941] From Cloze to Comprehension: Retrofitting Pre-trained Masked Language Models to Pre-trained Machine Reader

**Authors**: *Weiwen Xu, Xin Li, Wenxuan Zhang, Meng Zhou, Wai Lam, Luo Si, Lidong Bing*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d4e1c24ac41ff0b82ca1b171731f0b23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d4e1c24ac41ff0b82ca1b171731f0b23-Abstract-Conference.html)

**Abstract**:

We present Pre-trained Machine Reader (PMR), a novel method for retrofitting pre-trained masked language models (MLMs) to pre-trained machine reading comprehension (MRC) models without acquiring labeled data.PMR can resolve the discrepancy between model pre-training and downstream fine-tuning of existing MLMs.To build the proposed PMR, we constructed a large volume of general-purpose and high-quality MRC-style training data by using Wikipedia hyperlinks and designed a Wiki Anchor Extraction task to guide the MRC-style pre-training.Apart from its simplicity, PMR effectively solves extraction tasks, such as Extractive Question Answering and Named Entity Recognition. PMR shows tremendous improvements over existing approaches, especially in low-resource scenarios.When applied to the sequence classification task in the MRC formulation, PMR enables the extraction of high-quality rationales to explain the classification process, thereby providing greater prediction explainability. PMR also has the potential to serve as a unified model for tackling various extraction and classification tasks in the MRC formulation.

----

## [2942] DaTaSeg: Taming a Universal Multi-Dataset Multi-Task Segmentation Model

**Authors**: *Xiuye Gu, Yin Cui, Jonathan Huang, Abdullah Rashwan, Xuan Yang, Xingyi Zhou, Golnaz Ghiasi, Weicheng Kuo, Huizhong Chen, Liang-Chieh Chen, David A. Ross*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d4eed238cf5807c6b75face996302892-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d4eed238cf5807c6b75face996302892-Abstract-Conference.html)

**Abstract**:

Observing the close relationship among panoptic, semantic and instance segmentation tasks, we propose to train a universal multi-dataset multi-task segmentation model: DaTaSeg. We use a shared representation (mask proposals with class predictions) for all tasks. To tackle task discrepancy, we adopt different merge operations and post-processing for different tasks. We also leverage weak-supervision, allowing our segmentation model to benefit from cheaper bounding box annotations. To share knowledge across datasets, we use text embeddings from the same semantic embedding space as classifiers and share all network parameters among datasets. We train DaTaSeg on ADE semantic, COCO panoptic, and Objects365 detection datasets. DaTaSeg improves performance on all datasets, especially small-scale datasets, achieving 54.0 mIoU on ADE semantic and 53.5 PQ on COCO panoptic. DaTaSeg also enables weakly-supervised knowledge transfer on ADE panoptic and Objects365 instance segmentation. Experiments show DaTaSeg scales with the number of training datasets and enables open-vocabulary segmentation through direct transfer. In addition, we annotate an Objects365 instance segmentation set of 1,000 images and release it as a public evaluation benchmark on https://laoreja.github.io/dataseg.

----

## [2943] SituatedGen: Incorporating Geographical and Temporal Contexts into Generative Commonsense Reasoning

**Authors**: *Yunxiang Zhang, Xiaojun Wan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d4f2bc9885ecbe30f65031819ef8699f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d4f2bc9885ecbe30f65031819ef8699f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Recently, commonsense reasoning in text generation has attracted much attention. Generative commonsense reasoning is the task that requires machines, given a group of keywords, to compose a single coherent sentence with commonsense plausibility. While existing datasets targeting generative commonsense reasoning focus on everyday scenarios, it is unclear how well machines reason under specific geographical and temporal contexts. We formalize this challenging task as SituatedGen, where machines with commonsense should generate a pair of contrastive sentences given a group of keywords including geographical or temporal entities. We introduce a corresponding English dataset consisting of 8,268 contrastive sentence pairs, which are built upon several existing commonsense reasoning benchmarks with minimal manual labor. Experiments show that state-of-the-art generative language models struggle to generate sentences with commonsense plausibility and still lag far behind human performance. Our dataset is publicly available at https://github.com/yunx-z/situated_gen.

----

## [2944] Multi-scale Diffusion Denoised Smoothing

**Authors**: *Jongheon Jeong, Jinwoo Shin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d51e2a4628b15518f58bd1056b2d9124-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d51e2a4628b15518f58bd1056b2d9124-Abstract-Conference.html)

**Abstract**:

Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers adversarial robustness to models at scale, e.g., those of large pre-trained models. Specifically, one can perform randomized smoothing on any classifier via a simple "denoise-and-classify" pipeline, so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. In this paper, we present scalable methods to address the current trade-off between certified robustness and accuracy in denoised smoothing. Our key idea is to "selectively" apply smoothing among multiple noise scales, coined multi-scale smoothing, which can be efficiently implemented with a single diffusion model. This approach also suggests a new objective to compare the collective robustness of multi-scale smoothed classifiers, and questions which representation of diffusion model would maximize the objective. To address this, we propose to further fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, but (b) to generate rather diverse outputs otherwise. Our experiments show that the proposed multi-scale smoothing scheme, combined with diffusion fine-tuning, not only allows strong certified robustness at high noise scales but also maintains accuracy close to non-smoothed classifiers. Code is available at https://github.com/jh-jeong/smoothing-multiscale.

----

## [2945] PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers

**Authors**: *Phillip Lippe, Bas Veeling, Paris Perdikaris, Richard E. Turner, Johannes Brandstetter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d529b943af3dba734f8a7d49efcb6d09-Abstract-Conference.html)

**Abstract**:

Time-dependent partial differential equations (PDEs) are ubiquitous in science and engineering. Recently, mostly due to the high computational cost of traditional solution techniques, deep neural network based surrogates have gained increased interest. The practical utility of such neural PDE solvers relies on their ability to provide accurate, stable predictions over long time horizons, which is a notoriously hard problem. In this work, we present a large-scale analysis of common temporal rollout strategies, identifying the neglect of non-dominant spatial frequency information, often associated with high frequencies in PDE solutions, as the primary pitfall limiting stable, accurate rollout performance.  Based on these insights, we draw inspiration from recent advances in diffusion models to introduce PDE-Refiner; a novel model class that enables more accurate modeling of all frequency components via a multistep refinement process. We validate PDE-Refiner on challenging benchmarks of complex fluid dynamics, demonstrating stable and accurate rollouts that consistently outperform state-of-the-art models, including neural, numerical, and hybrid neural-numerical architectures. We further demonstrate that PDE-Refiner greatly enhances data efficiency, since the denoising objective implicitly induces a novel form of spectral data augmentation. Finally, PDE-Refiner's connection to diffusion models enables an accurate and efficient assessment of the model's predictive uncertainty, allowing us to estimate when the surrogate becomes inaccurate.

----

## [2946] Accelerating Exploration with Unlabeled Prior Data

**Authors**: *Qiyang Li, Jason Zhang, Dibya Ghosh, Amy Zhang, Sergey Levine*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d53d51e88d92d3723755f6d425bc513b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d53d51e88d92d3723755f6d425bc513b-Abstract-Conference.html)

**Abstract**:

Learning to solve tasks from a sparse reward signal is a major challenge for standard reinforcement learning (RL) algorithms. However, in the real world, agents rarely need to solve sparse reward tasks entirely from scratch. More often, we might possess prior experience to draw on that provides considerable guidance about which actions and outcomes are possible in the world, which we can use to explore more effectively for new tasks. In this work, we study how prior data without reward labels may be used to guide and accelerate exploration for an agent solving a new sparse reward task. We propose a simple approach that learns a reward model from online experience, labels the unlabeled prior data with optimistic rewards, and then uses it concurrently alongside the online data for downstream policy and critic optimization. This general formula leads to rapid exploration in several challenging sparse-reward domains where tabula rasa exploration is insufficient, including the AntMaze domain, Adroit hand manipulation domain, and a visual simulated robotic manipulation domain. Our results highlight the ease of incorporating unlabeled prior data into existing online RL algorithms, and the (perhaps surprising) effectiveness of doing so.

----

## [2947] Towards a Unified Framework of Contrastive Learning for Disentangled Representations

**Authors**: *Stefan Matthes, Zhiwei Han, Hao Shen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5470483dd38f71f7bd9e68ce1b94145-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5470483dd38f71f7bd9e68ce1b94145-Abstract-Conference.html)

**Abstract**:

Contrastive learning has recently emerged as a promising approach for learning data representations that discover and disentangle the explanatory factors of the data.Previous analyses of such approaches have largely focused on individual contrastive losses, such as noise-contrastive estimation (NCE) and InfoNCE, and rely on specific assumptions about the data generating process.This paper extends the theoretical guarantees for disentanglement to a broader family of contrastive methods, while also relaxing the assumptions about the data distribution.Specifically, we prove identifiability of the true latents for four contrastive losses studied in this paper, without imposing common independence assumptions.The theoretical findings are validated on several benchmark datasets.Finally, practical limitations of these methods are also investigated.

----

## [2948] An Improved Relaxation for Oracle-Efficient Adversarial Contextual Bandits

**Authors**: *Kiarash Banihashem, MohammadTaghi Hajiaghayi, Suho Shin, Max Springer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d54e440c92affd396117e161bbab5e78-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d54e440c92affd396117e161bbab5e78-Abstract-Conference.html)

**Abstract**:

We present an oracle-efficient relaxation  for the adversarial contextual bandits problem,  where the contexts are sequentially drawn i.i.d from a known distribution and the cost sequence is chosen by an online adversary.  Our algorithm has a regret bound of $O(T^{\frac{2}{3}}(K\log(|\Pi|))^{\frac{1}{3}})$ and makes at most $O(K)$ calls per round to an offline optimization oracle,  where $K$ denotes the number of actions, $T$ denotes the number of rounds and $\Pi$ denotes   the set of policies.  This is the first result to improve the prior best bound of $O((TK)^{\frac{2}{3}}(\log(|\Pi|))^{\frac{1}{3}})$ as obtained by   Syrgkanis et al.  at NeurIPS 2016, and the first to match the original bound of   Langford and Zhang at NeurIPS 2007  which was obtained for the stochastic case.

----

## [2949] Sequential Subset Matching for Dataset Distillation

**Authors**: *Jiawei Du, Qin Shi, Joey Tianyi Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d553f0e0abb80e2a60328d634583bd2e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d553f0e0abb80e2a60328d634583bd2e-Abstract-Conference.html)

**Abstract**:

Dataset distillation is a newly emerging task that synthesizes a small-size dataset used in training deep neural networks (DNNs) for reducing data storage and model training costs. The synthetic datasets are expected to capture the essence of the knowledge contained in real-world datasets such that the former yields a similar performance as the latter. Recent advancements in distillation methods have produced notable improvements in generating synthetic datasets. However, current state-of-the-art methods treat the entire synthetic dataset as a unified entity and optimize each synthetic instance equally . This static optimization approach may lead to performance degradation in dataset distillation. Specifically, we argue that static optimization can give rise to a coupling issue within the synthetic data, particularly when a larger amount of synthetic data is being optimized. This coupling issue, in turn, leads to the failure of the distilled dataset to extract the high-level features learned by the deep neural network (DNN) in the latter epochs.In this study, we propose a new dataset distillation strategy called Sequential Subset Matching (SeqMatch), which tackles this problem by adaptively optimizing the synthetic data to encourage sequential acquisition of knowledge during dataset distillation. Our analysis indicates that SeqMatch effectively addresses the coupling issue by sequentially generating the synthetic instances, thereby enhancing its performance significantly. Our proposed SeqMatch outperforms state-of-the-art methods in various datasets, including SVNH, CIFAR-10, CIFAR-100, and Tiny ImageNet.

----

## [2950] SLaM: Student-Label Mixing for Distillation with Unlabeled Examples

**Authors**: *Vasilis Kontonis, Fotis Iliopoulos, Khoa Trinh, Cenk Baykal, Gaurav Menghani, Erik Vee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d56b84c063265da949fe0feb815dcce8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d56b84c063265da949fe0feb815dcce8-Abstract-Conference.html)

**Abstract**:

Knowledge distillation with unlabeled examples is a powerful training paradigm for generating compact and lightweight student models in applications where the amount of labeled data is limited but one has access to a large pool of unlabeled data. In this setting, a large teacher model generates "soft" pseudo-labels for the unlabeled dataset which are then used for training the student model. Despite its success in a wide variety of applications, a  shortcoming of this approach is that the teacher's pseudo-labels are often noisy, leading to impaired student performance. In this paper, we present a principled method for knowledge distillation with unlabeled examples that we call Student-Label Mixing (SLaM) and we show that it consistently improves over prior approaches by evaluating it on several standard benchmarks. Finally, we show that SLaM comes with theoretical guarantees; along the way we give an algorithm improving the best-known sample complexity for learning halfspaces with margin under random classification noise, and provide the first convergence analysis for so-called ``forward loss-adjustment" methods.

----

## [2951] Mitigating the Popularity Bias of Graph Collaborative Filtering: A Dimensional Collapse Perspective

**Authors**: *Yifei Zhang, Hao Zhu, Yankai Chen, Zixing Song, Piotr Koniusz, Irwin King*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5753be6f71fbfefaf47aa27ec41279c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5753be6f71fbfefaf47aa27ec41279c-Abstract-Conference.html)

**Abstract**:

Graph-based Collaborative Filtering (GCF) is widely used in personalized recommendation systems. However, GCF suffers from a fundamental problem where features tend to occupy the embedding space inefficiently (by spanning only a low-dimensional subspace). Such an effect is characterized in GCF by the embedding space being dominated by a few of popular items with the user embeddings highly concentrated around them. This enhances the so-called Matthew effect of the popularity bias where popular items are highly recommend whereas remaining items are ignored. In this paper, we analyze the above effect in GCF and reveal that the simplified graph convolution operation (typically used in GCF) shrinks the singular space of the feature matrix. As typical approaches  (i.e., optimizing the uniformity term) fail to prevent the embedding space degradation, we propose a decorrelation-enhanced GCF objective that promotes feature diversity by leveraging the so-called principle of redundancy reduction in embeddings. However, unlike conventional methods that use the Euclidean geometry to relax hard constraints for decorrelation, we exploit non-Euclidean geometry. Such a choice helps  maintain the range space of the matrix and obtain small condition number, which prevents the embedding space degradation. Our  method  outperforms contrastive-based GCF models on several benchmark datasets and improves the performance for unpopular items.

----

## [2952] The Rise of AI Language Pathologists: Exploring Two-level Prompt Learning for Few-shot Weakly-supervised Whole Slide Image Classification

**Authors**: *Linhao Qu, Xiaoyuan Luo, Kexue Fu, Manning Wang, Zhijian Song*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d599b81036fd1a3b3949b7d444f31082-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d599b81036fd1a3b3949b7d444f31082-Abstract-Conference.html)

**Abstract**:

This paper introduces the novel concept of few-shot weakly supervised learning for pathology Whole Slide Image (WSI) classification, denoted as FSWC. A solution is proposed based on prompt learning and the utilization of a large language model, GPT-4. Since a WSI is too large and needs to be divided into patches for processing, WSI classification is commonly approached as a Multiple Instance Learning (MIL) problem. In this context, each WSI is considered a bag, and the obtained patches are treated as instances. The objective of FSWC is to classify both bags and instances with only a limited number of labeled bags. Unlike conventional few-shot learning problems, FSWC poses additional challenges due to its weak bag labels within the MIL framework. Drawing inspiration from the recent achievements of vision-language models (V-L models) in downstream few-shot classification tasks, we propose a two-level prompt learning MIL framework tailored for pathology, incorporating language prior knowledge. Specifically, we leverage CLIP to extract instance features for each patch, and introduce a prompt-guided pooling strategy to aggregate these instance features into a bag feature. Subsequently, we employ a small number of labeled bags to facilitate few-shot prompt learning based on the bag features. Our approach incorporates the utilization of GPT-4 in a question-and-answer mode to obtain language prior knowledge at both the instance and bag levels, which are then integrated into the instance and bag level language prompts. Additionally, a learnable component of the language prompts is trained using the available few-shot labeled data. We conduct extensive experiments on three real WSI datasets encompassing breast cancer, lung cancer, and cervical cancer, demonstrating the notable performance of the proposed method in bag and instance classification. All codes will be made publicly accessible.

----

## [2953] State Sequences Prediction via Fourier Transform for Representation Learning

**Authors**: *Mingxuan Ye, Yufei Kuang, Jie Wang, Yang Rui, Wengang Zhou, Houqiang Li, Feng Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5b94ca503b33d07f9bef8ed8ee4678b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5b94ca503b33d07f9bef8ed8ee4678b-Abstract-Conference.html)

**Abstract**:

While deep reinforcement learning (RL) has been demonstrated effective in solving complex control tasks, sample efficiency remains a key challenge due to the large amounts of data required for remarkable performance. Existing research explores the application of representation learning for data-efficient RL, e.g., learning predictive representations by predicting long-term future states. However, many existing methods do not fully exploit the structural information inherent in sequential state signals, which can potentially improve the quality of long-term decision-making but is difficult to discern in the time domain. To tackle this problem, we propose State Sequences Prediction via Fourier Transform (SPF), a novel method that exploits the frequency domain of state sequences to extract the underlying patterns in time series data for learning expressive representations efficiently. Specifically, we theoretically analyze the existence of structural information in state sequences, which is closely related to policy performance and signal regularity, and then propose to predict the Fourier transform of infinite-step future state sequences to extract such information. One of the appealing features of SPF is that it is simple to implement while not requiring storage of infinite-step future states as prediction targets. Experiments demonstrate that the proposed method outperforms several state-of-the-art algorithms in terms of both sample efficiency and performance.

----

## [2954] Beyond Myopia: Learning from Positive and Unlabeled Data through Holistic Predictive Trends

**Authors**: *Xinrui Wang, Wenhai Wan, Chuanxing Geng, Shaoyuan Li, Songcan Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5c0f9585592bad5251133813893a6c0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5c0f9585592bad5251133813893a6c0-Abstract-Conference.html)

**Abstract**:

Learning binary classifiers from positive and unlabeled data (PUL) is vital in many real-world applications, especially when verifying negative examples is difficult. Despite the impressive empirical performance of recent PUL methods, challenges like accumulated errors and increased estimation bias persist due to the absence of negative labels. In this paper, we unveil an intriguing yet long-overlooked observation in PUL: \textit{resampling the positive data in each training iteration to ensure a balanced distribution between positive and unlabeled examples results in strong early-stage performance. Furthermore, predictive trends for positive and negative classes display distinctly different patterns.} Specifically, the scores (output probability) of unlabeled negative examples consistently decrease, while those of unlabeled positive examples show largely chaotic trends. Instead of focusing on classification within individual time frames, we innovatively adopt a holistic approach, interpreting the scores of each example as a temporal point process (TPP). This reformulates the core problem of PUL as recognizing trends in these scores. We then propose a novel TPP-inspired measure for trend detection and prove its asymptotic unbiasedness in predicting changes. Notably, our method accomplishes PUL without requiring additional parameter tuning or prior assumptions, offering an alternative perspective for tackling this problem. Extensive experiments verify the superiority of our method, particularly in a highly imbalanced real-world setting, where it achieves improvements of up to $11.3\%$ in key metrics.

----

## [2955] Fast Approximation of Similarity Graphs with Kernel Density Estimation

**Authors**: *Peter Macgregor, He Sun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5c56ec4f69c9a473089b16000d3f8cd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5c56ec4f69c9a473089b16000d3f8cd-Abstract-Conference.html)

**Abstract**:

Constructing a similarity graph from a set $X$ of data points in $ \mathbb{R}^d$ is the first step of many modern clustering algorithms. However, typical constructions of a similarity graph have high time complexity, and a quadratic space dependency with respect to $|X|$. We address this limitation and present a new algorithmic framework that constructs a sparse approximation of the fully connected similarity graph while preserving its cluster structure. Our presented algorithm is based on the kernel density estimation problem, and is applicable for arbitrary kernel functions. We compare our designed algorithm with the  well-known implementations from the scikit-learn library and the FAISS library,  and find that our method significantly outperforms the implementation from both libraries on a variety of datasets.

----

## [2956] An Efficient Dataset Condensation Plugin and Its Application to Continual Learning

**Authors**: *Enneng Yang, Li Shen, Zhenyi Wang, Tongliang Liu, Guibing Guo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d5f34e7e70d80f5037ab16a48e2d186e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d5f34e7e70d80f5037ab16a48e2d186e-Abstract-Conference.html)

**Abstract**:

Dataset condensation (DC) distills a large real-world dataset into a small synthetic dataset, with the goal of training a network from scratch on the latter that performs similarly to the former. State-of-the-art (SOTA) DC methods have achieved satisfactory results through techniques such as accuracy, gradient, training trajectory, or distribution matching. However, these works all perform matching in the high-dimension pixel spaces, ignoring that natural images are usually locally connected and have lower intrinsic dimensions, resulting in low condensation efficiency.  In this work, we propose a simple-yet-efficient dataset condensation plugin that matches the raw and synthetic datasets in a low-dimensional manifold. Specifically, our plugin condenses raw images into two low-rank matrices instead of parameterized image matrices. Our plugin can be easily incorporated into existing DC methods, thereby containing richer raw dataset information at limited storage costs to improve the downstream applications' performance.  We verify on multiple public datasets that when the proposed plugin is combined with SOTA DC methods, the performance of the network trained on synthetic data is significantly improved compared to traditional DC methods. Moreover, when applying the DC methods as a plugin to continual learning tasks, we observed that our approach effectively mitigates catastrophic forgetting of old tasks under limited memory buffer constraints and avoids the problem of raw data privacy leakage.

----

## [2957] Bootstrapped Training of Score-Conditioned Generator for Offline Design of Biological Sequences

**Authors**: *Minsu Kim, Federico Berto, Sungsoo Ahn, Jinkyoo Park*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d601a9b708cacfad167f6c6c45647a18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d601a9b708cacfad167f6c6c45647a18-Abstract-Conference.html)

**Abstract**:

We study the problem of optimizing biological sequences, e.g., proteins, DNA, and RNA, to maximize a black-box score function that is only evaluated in an offline dataset. We propose a novel solution, bootstrapped training of score-conditioned generator (BootGen) algorithm. Our algorithm repeats a two-stage process. In the first stage, our algorithm trains the biological sequence generator with rank-based weights to enhance the accuracy of sequence generation based on high scores. The subsequent stage involves bootstrapping, which augments the training dataset with self-generated data labeled by a proxy score function. Our key idea is to align the score-based generation with a proxy score function, which distills the knowledge of the proxy score function to the generator. After training, we aggregate samples from multiple bootstrapped generators and proxies to produce a diverse design. Extensive experiments show that our method outperforms competitive baselines on biological sequential design tasks. We provide reproducible source code: https://github.com/kaist-silab/bootgen.

----

## [2958] Statistically Valid Variable Importance Assessment through Conditional Permutations

**Authors**: *Ahmad Chamma, Denis A. Engemann, Bertrand Thirion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d60e14c19cd6e0fc38556ad29ac8fbc9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d60e14c19cd6e0fc38556ad29ac8fbc9-Abstract-Conference.html)

**Abstract**:

Variable importance assessment has become a crucial step in machine-learning applications when using complex learners, such as deep neural networks, on large-scale data. Removal-based importance assessment is currently the reference approach, particularly when statistical guarantees are sought to justify variable inclusion. It is often implemented with variable permutation schemes. On the flip side, these approaches risk misidentifying unimportant variables as important in the presence of correlations among covariates. Here we develop a systematic approach for studying Conditional Permutation Importance (CPI) that is model agnostic and computationally lean,  as well as reusable benchmarks of state-of-the-art variable importance estimators. We show theoretically and empirically that \textit{CPI} overcomes the limitations of standard permutation importance by providing accurate type-I error control. When used with a deep neural network, \textit{CPI} consistently showed top accuracy across benchmarks. An experiment on real-world data analysis in a large-scale medical dataset showed that \textit{CPI} provides a more parsimonious selection of statistically significant variables. Our results suggest that \textit{CPI} can be readily used as drop-in replacement for permutation-based methods.

----

## [2959] Towards Better Dynamic Graph Learning: New Architecture and Unified Library

**Authors**: *Le Yu, Leilei Sun, Bowen Du, Weifeng Lv*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d611019afba70d547bd595e8a4158f55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d611019afba70d547bd595e8a4158f55-Abstract-Conference.html)

**Abstract**:

We propose DyGFormer, a new Transformer-based architecture for dynamic graph learning. DyGFormer is conceptually simple and only needs to learn from nodes' historical first-hop interactions by: (1) a neighbor co-occurrence encoding scheme that explores the correlations of the source node and destination node based on their historical sequences; (2) a patching technique that divides each sequence into multiple patches and feeds them to Transformer, allowing the model to effectively and efficiently benefit from longer histories. We also introduce DyGLib, a unified library with standard training pipelines, extensible coding interfaces, and comprehensive evaluating protocols to promote reproducible, scalable, and credible dynamic graph learning research. By performing exhaustive experiments on thirteen datasets for dynamic link prediction and dynamic node classification tasks, we find that DyGFormer achieves state-of-the-art performance on most of the datasets, demonstrating its effectiveness in capturing nodes' correlations and long-term temporal dependencies. Moreover, some results of baselines are inconsistent with previous reports, which may be caused by their diverse but less rigorous implementations, showing the importance of DyGLib. All the used resources are publicly available at https://github.com/yule-BUAA/DyGLib.

----

## [2960] Implicit Manifold Gaussian Process Regression

**Authors**: *Bernardo Fichera, Slava Borovitskiy, Andreas Krause, Aude Gemma Billard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d611d06e3207330555fbc10810e70163-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d611d06e3207330555fbc10810e70163-Abstract-Conference.html)

**Abstract**:

Gaussian process regression is widely used because of its ability to provide well-calibrated uncertainty estimates and handle small or sparse datasets. However, it struggles with high-dimensional data. One possible way to scale this technique to higher dimensions is to leverage the implicit low-dimensional manifold upon which the data actually lies, as postulated by the manifold hypothesis. Prior work ordinarily requires the manifold structure to be explicitly provided though, i.e. given by a mesh or be known to be one of the well-known manifolds like the sphere. In contrast, in this paper we propose a Gaussian process regression technique capable of inferring implicit structure directly from data (labeled and unlabeled) in a fully differentiable way. For the resulting model, we discuss its convergence to the Mat√©rn Gaussian process on the assumed manifold. Our technique scales up to hundreds of thousands of data points, and improves the predictive performance and calibration of the standard Gaussian process regression in some high-dimensional settings.

----

## [2961] UDC-SIT: A Real-World Dataset for Under-Display Cameras

**Authors**: *Kyusu Ahn, Byeonghyun Ko, HyunGyu Lee, Chanwoo Park, Jaejin Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d612971396f825dbf8e0e736f99a1955-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d612971396f825dbf8e0e736f99a1955-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Under Display Camera (UDC) is a novel imaging system that mounts a digital camera lens beneath a display panel with the panel covering the camera. However, the display panel causes severe degradation to captured images, such as low transmittance, blur, noise, and flare. The restoration of UDC-degraded images is challenging because of the unique luminance and diverse patterns of flares. Existing UDC dataset studies focus on unrealistic or synthetic UDC degradation rather than real-world UDC images. In this paper, we propose a real-world UDC dataset called UDC-SIT. To obtain the non-degraded and UDC-degraded images for the same scene, we propose an image-capturing system and an image alignment technique that exploits discrete Fourier transform (DFT) to align a pair of captured images. UDC-SIT also includes comprehensive annotations missing from other UDC datasets, such as light source, day/night, indoor/outdoor, and flare components (e.g.,  shimmers, streaks, and glares). We compare UDC-SIT with four existing representative UDC datasets and present the problems with existing UDC datasets. To show UDC-SIT's effectiveness, we compare UDC-SIT and a representative synthetic UDC dataset using four representative learnable image restoration models. The result indicates that the models trained with the synthetic UDC dataset are impractical because the synthetic UDC dataset does not reflect the actual characteristics of UDC-degraded images. UDC-SIT can enable further exploration in the UDC image restoration area and provide better insights into the problem. UDC-SIT is available at: https://github.com/mcrl/UDC-SIT.

----

## [2962] The Crucial Role of Normalization in Sharpness-Aware Minimization

**Authors**: *Yan Dai, Kwangjun Ahn, Suvrit Sra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d616a353c711f11c722e3f28d2d9e956-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d616a353c711f11c722e3f28d2d9e956-Abstract-Conference.html)

**Abstract**:

Sharpness-Aware Minimization (SAM) is a recently proposed gradient-based optimizer (Foret et al., ICLR 2021) that greatly improves the prediction performance of deep neural networks. Consequently, there has been a surge of interest in explaining its empirical success. We focus, in particular, on understanding the role played by normalization, a key component of the SAM updates. We theoretically and empirically study the effect of normalization in SAM for both convex and non-convex functions, revealing two key roles played by normalization: i) it helps in stabilizing the algorithm; and ii) it enables the algorithm to drift along a continuum (manifold) of minima -- a property identified by recent theoretical works that is the key to better performance. We further argue that these two properties of normalization make SAM robust against the choice of hyper-parameters, supporting the practicality of SAM. Our conclusions are backed by various experiments.

----

## [2963] Policy Space Diversity for Non-Transitive Games

**Authors**: *Jian Yao, Weiming Liu, Haobo Fu, Yaodong Yang, Stephen McAleer, Qiang Fu, Wei Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d61819e9b4a607b8448de762235148c4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d61819e9b4a607b8448de762235148c4-Abstract-Conference.html)

**Abstract**:

Policy-Space Response Oracles (PSRO) is an influential algorithm framework for approximating a Nash Equilibrium (NE) in multi-agent non-transitive games. Many previous studies have been trying to promote policy diversity in PSRO. A major weakness with existing diversity metrics is that a more diverse (according to their diversity metrics) population does not necessarily mean (as we proved in the paper) a better approximation to a NE. To alleviate this problem, we propose a new diversity metric, the improvement of which guarantees a better approximation to a NE. Meanwhile, we develop a practical and well-justified method to optimize our diversity metric using only state-action samples. By incorporating our diversity regularization into the best response solving of PSRO, we obtain a new PSRO variant, \textit{Policy Space Diversity} PSRO (PSD-PSRO). We present the convergence property of PSD-PSRO. Empirically, extensive experiments on single-state games, Leduc, and Goofspiel demonstrate that PSD-PSRO is more effective in producing significantly less exploitable policies than state-of-the-art PSRO variants.

----

## [2964] COOM: A Game Benchmark for Continual Reinforcement Learning

**Authors**: *Tristan Tomilin, Meng Fang, Yudi Zhang, Mykola Pechenizkiy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d61d9f4fe4357296cb658795fd7999f0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d61d9f4fe4357296cb658795fd7999f0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The advancement of continual reinforcement learning (RL) has been facing various obstacles, including standardized metrics and evaluation protocols, demanding computational requirements, and a lack of widely accepted standard benchmarks. In response to these challenges, we present COOM ($\textbf{C}$ontinual D$\textbf{OOM}$), a continual RL benchmark tailored for embodied pixel-based RL. COOM presents a meticulously crafted suite of task sequences set within visually distinct 3D environments, serving as a robust evaluation framework to assess crucial aspects of continual RL, such as catastrophic forgetting, knowledge transfer, and sample-efficient learning. Following an in-depth empirical evaluation of popular continual learning (CL) methods, we pinpoint their limitations, provide valuable insight into the benchmark and highlight unique algorithmic challenges. This makes our work the first to benchmark image-based CRL in 3D environments with embodied perception. The primary objective of the COOM benchmark is to offer the research community a valuable and cost-effective challenge. It seeks to deepen our comprehension of the capabilities and limitations of current and forthcoming CL methods in an RL setting. The code and environments are open-sourced and accessible on GitHub.

----

## [2965] Video-Mined Task Graphs for Keystep Recognition in Instructional Videos

**Authors**: *Kumar Ashutosh, Santhosh Kumar Ramakrishnan, Triantafyllos Afouras, Kristen Grauman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d62e65cfdba247e0cd7cac5964f9fbd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d62e65cfdba247e0cd7cac5964f9fbd9-Abstract-Conference.html)

**Abstract**:

Procedural activity understanding requires perceiving human actions in terms of a broader task, where multiple keysteps are performed in sequence across a long video to reach a final goal state---such as the steps of a recipe or the steps of a DIY fix-it task.  Prior work largely treats keystep recognition in isolation of this broader structure, or else rigidly confines keysteps to align with a particular sequential script.  We propose discovering a task graph automatically from how-to videos to represent probabilistically how people tend to execute keysteps, then leverage this graph to regularize keystep recognition in novel videos.  On multiple datasets of real-world instructional video, we show the impact: more reliable zero-shot keystep localization and improved video representation learning, exceeding the state of the art.

----

## [2966] Affinity-Aware Graph Networks

**Authors**: *Ameya Velingker, Ali Kemal Sinop, Ira Ktena, Petar Velickovic, Sreenivas Gollapudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d642b0633afad94f660554e05b40608e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d642b0633afad94f660554e05b40608e-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have emerged as a powerful technique for learning on relational data. Owing to the relatively limited number of message passing steps they perform—and hence a smaller receptive field—there has been significant interest in improving their expressivity by incorporating structural aspects of the underlying graph. In this paper, we explore the use of affinity measures as features in graph neural networks, in particular measures arising from random walks, including effective resistance, hitting and commute times. We propose message passing networks based on these features and evaluate their performance on a variety of node and graph property prediction tasks. Our architecture has low computational complexity, while our features are invariant to the permutations of the underlying graph. The measures we compute allow the network to exploit the connectivity properties of the graph, thereby allowing us to outperform relevant benchmarks for a wide variety of tasks, often with significantly fewer message passing steps. On one of the largest publicly available graph regression datasets, OGB-LSC-PCQM4Mv1, we obtain the best known single-model validation MAE at the time of writing.

----

## [2967] Eliminating Catastrophic Overfitting Via Abnormal Adversarial Examples Regularization

**Authors**: *Runqi Lin, Chaojian Yu, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d65befe6b80ecf7f180b4def503d7776-Abstract-Conference.html)

**Abstract**:

Single-step adversarial training (SSAT) has demonstrated the potential to achieve both efficiency and robustness. However, SSAT suffers from catastrophic overfitting (CO), a phenomenon that leads to a severely distorted classifier, making it vulnerable to multi-step adversarial attacks. In this work, we observe that some adversarial examples generated on the SSAT-trained network exhibit anomalous behaviour, that is, although these training samples are generated by the inner maximization process, their associated loss decreases instead, which we named abnormal adversarial examples (AAEs). Upon further analysis, we discover a close relationship between AAEs and classifier distortion, as both the number and outputs of AAEs undergo a significant variation with the onset of CO. Given this observation, we re-examine the SSAT process and uncover that before the occurrence of CO, the classifier already displayed a slight distortion, indicated by the presence of few AAEs. Furthermore, the classifier directly optimizing these AAEs will accelerate its distortion, and correspondingly, the variation of AAEs will sharply increase as a result. In such a vicious circle, the classifier rapidly becomes highly distorted and manifests as CO within a few iterations. These observations motivate us to eliminate CO by hindering the generation of AAEs. Specifically, we design a novel method, termed Abnormal Adversarial Examples Regularization (AAER), which explicitly regularizes the variation of AAEs to hinder the classifier from becoming distorted. Extensive experiments demonstrate that our method can effectively eliminate CO and further boost adversarial robustness with negligible additional computational overhead. Our implementation can be found at https://github.com/tmllab/2023NeurIPSAAER.

----

## [2968] Hardware Resilience Properties of Text-Guided Image Classifiers

**Authors**: *Syed Talal Wasim, Kabila Haile Soboka, Abdulrahman Mahmoud, Salman H. Khan, David Brooks, Gu-Yeon Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d664de48bf5ad8e8e48c77e175eb0e80-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d664de48bf5ad8e8e48c77e175eb0e80-Abstract-Conference.html)

**Abstract**:

This paper presents a novel method to enhance the reliability of image classification models during deployment in the face of transient hardware errors. By utilizing enriched text embeddings derived from GPT-3 with question prompts per class and CLIP pretrained text encoder, we investigate their impact as an initialization for the classification layer. Our approach achieves a remarkable $5.5\times$ average increase in hardware reliability (and up to $14\times$) across various architectures in the most critical layer, with minimal accuracy drop ($0.3\%$ on average) compared to baseline PyTorch models. Furthermore, our method seamlessly integrates with any image classification backbone, showcases results across various network architectures, decreases parameter and FLOPs overhead, and follows a consistent training recipe. This research offers a practical and efficient solution to bolster the robustness of image classification models against hardware failures, with potential implications for future studies in this domain. Our code and models are released at https://github.com/TalalWasim/TextGuidedResilience.

----

## [2969] Reliable Off-Policy Learning for Dosage Combinations

**Authors**: *Jonas Schweisthal, Dennis Frauen, Valentyn Melnychuk, Stefan Feuerriegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d69103d7895f4e2083f24b664003d386-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d69103d7895f4e2083f24b664003d386-Abstract-Conference.html)

**Abstract**:

Decision-making in personalized medicine such as cancer therapy or critical care must often make choices for dosage combinations, i.e., multiple continuous treatments. Existing work for this task has modeled the effect of multiple treatments independently, while estimating the joint effect has received little attention but comes with non-trivial challenges. In this paper, we propose a novel method for reliable off-policy learning for dosage combinations. Our method proceeds along three steps: (1) We develop a tailored neural network that estimates the individualized dose-response function while accounting for the joint effect of multiple dependent dosages. (2) We estimate the generalized propensity score using conditional normalizing flows in order to detect regions with limited overlap in the shared covariate-treatment space. (3) We present a gradient-based learning algorithm to find the optimal, individualized dosage combinations. Here, we ensure reliable estimation of the policy value by avoiding regions with limited overlap. We finally perform an extensive evaluation of our method to show its effectiveness. To the best of our knowledge, ours is the first work to provide a method for reliable off-policy learning for optimal dosage combinations.

----

## [2970] A Unified Algorithm Framework for Unsupervised Discovery of Skills based on Determinantal Point Process

**Authors**: *Jiayu Chen, Vaneet Aggarwal, Tian Lan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6938c8e88ef62394d2f4f3fd428e036-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6938c8e88ef62394d2f4f3fd428e036-Abstract-Conference.html)

**Abstract**:

Learning rich skills under the option framework without supervision of external rewards is at the frontier of reinforcement learning research. Existing works mainly fall into two distinctive categories: variational option discovery that maximizes the diversity of the options through a mutual information loss (while ignoring coverage) and Laplacian-based methods that focus on improving the coverage of options by increasing connectivity of the state space (while ignoring diversity). In this paper, we show that diversity and coverage in unsupervised option discovery can indeed be unified under the same mathematical framework. To be specific, we explicitly quantify the diversity and coverage of the learned options through a novel use of Determinantal Point Process (DPP) and optimize these objectives to discover options with both superior diversity and coverage. Our proposed algorithm, ODPP, has undergone extensive evaluation on challenging tasks created with Mujoco and Atari. The results demonstrate that our algorithm outperforms state-of-the-art baselines in both diversity- and coverage-driven categories.

----

## [2971] Discovering Intrinsic Spatial-Temporal Logic Rules to Explain Human Actions

**Authors**: *Chengzhi Cao, Chao Yang, Ruimao Zhang, Shuang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d69fdbe4d13080bb7fa33249ca136976-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d69fdbe4d13080bb7fa33249ca136976-Abstract-Conference.html)

**Abstract**:

We propose an interpretable model to uncover the behavioral patterns of human movements by analyzing their trajectories. Our approach is based on the belief that human actions are driven by intentions and are influenced by environmental factors such as spatial relationships with surrounding objects. To model this, we use a set of spatial-temporal logic rules that include intention variables as principles. These rules are automatically discovered and used to capture the dynamics of human actions. To learn the model parameters and rule content, we design an EM learning algorithm that treats the unknown rule content as a latent variable. In the E-step, we evaluate the posterior over the latent rule content, and in the M-step, we optimize the rule generator and model parameters by maximizing the expected log-likelihood. Our model has wide-ranging applications in areas such as sports analytics, robotics, and autonomous cars. We demonstrate the model's superior interpretability and prediction performance on both pedestrian and NBA basketball player datasets, achieving promising results.

----

## [2972] DiT-3D: Exploring Plain Diffusion Transformers for 3D Shape Generation

**Authors**: *Shentong Mo, Enze Xie, Ruihang Chu, Lanqing Hong, Matthias Nießner, Zhenguo Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6c01b025cad37d5c8bab4ba18846c02-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6c01b025cad37d5c8bab4ba18846c02-Abstract-Conference.html)

**Abstract**:

Recent Diffusion Transformers (i.e., DiT) have demonstrated their powerful effectiveness in generating high-quality 2D images. However, it is unclear how the Transformer architecture performs equally well in 3D shape generation, as previous 3D diffusion methods mostly adopted the U-Net architecture. To bridge this gap, we propose a novel Diffusion Transformer for 3D shape generation, named DiT-3D, which can directly operate the denoising process on voxelized point clouds using plain Transformers. Compared to existing U-Net approaches, our DiT-3D is more scalable in model size and produces much higher quality generations.Specifically, the DiT-3D adopts the design philosophy of DiT but modifies it by incorporating 3D positional and patch embeddings to aggregate input from voxelized point clouds.To reduce the computational cost of self-attention in 3D shape generation, we incorporate 3D window attention into Transformer blocks, as the increased 3D token length resulting from the additional dimension of voxels can lead to high computation.Finally, linear and devoxelization layers are used to predict the denoised point clouds. In addition, we empirically observe that the pre-trained DiT-2D checkpoint on ImageNet can significantly improve DiT-3D on ShapeNet.Experimental results on the ShapeNet dataset demonstrate that the proposed DiT-3D achieves state-of-the-art performance in high-fidelity and diverse 3D point cloud generation.

----

## [2973] DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries

**Authors**: *Joshua Engels, Benjamin Coleman, Vihan Lakshman, Anshumali Shrivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6cc45de2e2dea14b96c1eba88fd8ef7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6cc45de2e2dea14b96c1eba88fd8ef7-Abstract-Conference.html)

**Abstract**:

We study the problem of $\text{\emph{vector set search}}$ with $\text{\emph{vector set queries}}$. This task is analogous to traditional near-neighbor search, with the exception that both the query and each element in the collection are $\text{\textit{sets}}$ of vectors. We identify this problem as a core subroutine for semantic search applications and find that existing solutions are unacceptably slow. Towards this end, we present a new approximate search algorithm, DESSERT ($\text{\bf D}$ESSERT $\text{\bf E}$ffeciently $\text{\bf S}$earches $\text{\bf S}$ets of $\text{\bf E}$mbeddings via $\text{\bf R}$etrieval $\text{\bf T}$ables). DESSERT is a general tool with strong theoretical guarantees and excellent empirical performance. When we integrate DESSERT into ColBERT, a state-of-the-art semantic search model, we find a 2-5x speedup on the MS MARCO and LoTTE retrieval benchmarks with minimal loss in recall, underscoring the effectiveness and practical applicability of our proposal.

----

## [2974] Dynamic Sparsity Is Channel-Level Sparsity Learner

**Authors**: *Lu Yin, Gen Li, Meng Fang, Li Shen, Tianjin Huang, Zhangyang Wang, Vlado Menkovski, Xiaolong Ma, Mykola Pechenizkiy, Shiwei Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6d0e41e0b1ed38c76d13c9e417a8f1f-Abstract-Conference.html)

**Abstract**:

Sparse training has received an upsurging interest in machine learning due to its tantalizing saving potential for both the entire training process as well as the inference. Dynamic sparse training (DST) as a leading approach can train deep neural networks at high sparsity from scratch to match the performance of their dense counterparts. However, most if not all DST prior arts demonstrate their effectiveness on unstructured sparsity with highly irregular sparse patterns, which receives limited support in common hardware. This limitation hinders the usage of DST in practice. In this paper, we propose Channel-aware dynamic sparse (Chase), that for the first time seamlessly translates the promise of unstructured dynamic sparsity to GPU-friendly channel-level sparsity (not fine-grained N:M or group sparsity) during one end-to-end training process, without any ad-hoc operations. The resulting small sparse networks can be directly accelerated by commodity hardware, without using any particularly sparsity-aware hardware accelerators. This appealing outcome is partially motivated by a hidden phenomenon of dynamic sparsity: off-the-shelf unstructured DST implicitly involves biased parameter reallocation across channels, with a large fraction of channels (up to 60%) being sparser than others. By progressively identifying and removing these channels during training, our approach transfers unstructured sparsity to channel-wise sparsity.  Our experimental results demonstrate that Chase achieves 1.7x inference throughput speedup on common GPU devices without compromising accuracy with ResNet-50 on ImageNet. We release our code in https://github.com/luuyin/chase.

----

## [2975] Bayesian nonparametric (non-)renewal processes for analyzing neural spike train variability

**Authors**: *David Liu, Máté Lengyel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6db7eb6245ec0c6e45f445956994143-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6db7eb6245ec0c6e45f445956994143-Abstract-Conference.html)

**Abstract**:

Neural spiking activity is generally variable, non-stationary, and exhibits complex dependencies on covariates, such as sensory input or behavior. These dependencies have been proposed to be signatures of specific computations, and so characterizing them with quantitative rigor is critical for understanding neural computations. Approaches based on point processes provide a principled statistical framework for modeling neural spiking activity. However, currently, they only allow the instantaneous mean, but not the instantaneous variability, of responses to depend on covariates. To resolve this limitation, we propose a scalable Bayesian approach generalizing modulated renewal processes using sparse variational Gaussian processes. We leverage pathwise conditioning for computing nonparametric priors over conditional interspike interval distributions and rely on automatic relevance determination to detect lagging interspike interval dependencies beyond renewal order.  After systematically validating our method on synthetic data, we apply it to two foundational datasets of animal navigation: head direction cells in freely moving mice and hippocampal place cells in rats running along a linear track. Our model exhibits competitive or better predictive power compared to state-of-the-art baselines, and outperforms them in terms of capturing interspike interval statistics. These results confirm the importance of modeling covariate-dependent spiking variability, and further analyses of our fitted models reveal rich patterns of variability modulation beyond the temporal resolution of flexible count-based approaches.

----

## [2976] Evaluating Self-Supervised Learning for Molecular Graph Embeddings

**Authors**: *Hanchen Wang, Jean Kaddour, Shengchao Liu, Jian Tang, Joan Lasenby, Qi Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6dc15cc2442a40904e704d624d1fbe8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6dc15cc2442a40904e704d624d1fbe8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Graph Self-Supervised Learning (GSSL) provides a robust pathway for acquiring embeddings without expert labelling, a capability that carries profound implications for molecular graphs due to the staggering number of potential molecules and the high cost of obtaining labels. However, GSSL methods are designed not for optimisation within a specific domain but rather for transferability across a variety of downstream tasks. This broad applicability complicates their evaluation. Addressing this challenge, we present "Molecular Graph Representation Evaluation" (MOLGRAPHEVAL), generating detailed profiles of molecular graph embeddings with interpretable and diversified attributes. MOLGRAPHEVAL offers a suite of probing tasks grouped into three categories: (i) generic graph, (ii) molecular substructure, and (iii) embedding space properties. By leveraging MOLGRAPHEVAL to benchmark existing GSSL methods against both current downstream datasets and our suite of tasks, we uncover significant inconsistencies between inferences drawn solely from existing datasets and those derived from more nuanced probing. These findings suggest that current evaluation methodologies fail to capture the entirety of the landscape.

----

## [2977] SEEDS: Exponential SDE Solvers for Fast High-Quality Sampling from Diffusion Models

**Authors**: *Martin Gonzalez, Nelson Fernandez Pinto, Thuy Tran, Elies Gherbi, Hatem Hajri, Nader Masmoudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6f764aae383d9ff28a0f89f71defbd9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6f764aae383d9ff28a0f89f71defbd9-Abstract-Conference.html)

**Abstract**:

A potent class of generative models known as Diffusion Probabilistic Models(DPMs) has become prominent. A forward diffusion process adds gradually noiseto data, while a model learns to gradually denoise. Sampling from pre-trainedDPMs is obtained by solving differential equations (DE) defined by the learntmodel, a process which has shown to be prohibitively slow. Numerous efforts onspeeding-up this process have consisted on crafting powerful ODE solvers.Despite being quick, such solvers do not usually reach the optimal qualityachieved by available slow SDE solvers. Our goal is to propose SDE solvers thatreach optimal quality without requiring several hundreds or thousands of NFEsto achieve that goal. We propose Stochastic Explicit ExponentialDerivative-free Solvers (SEEDS), improving and generalizing ExponentialIntegrator approaches to the stochastic case on several frameworks. After carefully analyzing the formulation of exactsolutions of diffusion SDEs, we craft SEEDS to analytically compute the linearpart of such solutions. Inspired by the Exponential Time-Differencing method,SEEDS use a novel treatment of the stochastic components of solutions,enabling the analytical computation of their variance, and contains high-orderterms allowing to reach optimal quality sampling $\sim3$-$5\times$ faster than previousSDE methods. We validate our approach on several image generation benchmarks,showing that SEEDS outperform or are competitive with previous SDE solvers.Contrary to the latter, SEEDS are derivative and training free, and we fullyprove strong convergence guarantees for them.

----

## [2978] Robust Multi-Agent Reinforcement Learning via Adversarial Regularization: Theoretical Foundation and Stable Algorithms

**Authors**: *Alexander Bukharin, Yan Li, Yue Yu, Qingru Zhang, Zhehui Chen, Simiao Zuo, Chao Zhang, Songan Zhang, Tuo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d6f8517fceeca1e2cd61721dff786c14-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d6f8517fceeca1e2cd61721dff786c14-Abstract-Conference.html)

**Abstract**:

Multi-Agent Reinforcement Learning (MARL) has shown promising results across several domains. Despite this promise, MARL policies often lack robustness and are therefore sensitive to small changes in their environment. This presents a serious concern for the real world deployment of MARL algorithms, where the testing environment may slightly differ from the training environment. In this work we show that we can gain robustness by controlling a policy’s Lipschitz constant, and under mild conditions, establish the existence of a Lipschitz and close-to-optimal policy. Motivated by these insights, we propose a new robust MARL framework, ERNIE, that promotes the Lipschitz continuity of the policies with respect to the state observations and actions by adversarial regularization. The ERNIE framework provides robustness against noisy observations, changing transition dynamics, and malicious actions of agents. However, ERNIE’s adversarial regularization may introduce some training instability. To reduce this instability, we reformulate adversarial regularization as a Stackelberg game. We demonstrate the effectiveness of the proposed framework with extensive experiments in traffic light control and particle environments. In addition, we extend ERNIE to mean-field MARL with a formulation based on distributionally robust optimization that outperforms its non-robust counterpart and is of independent interest. Our code is available at https://github.com/abukharin3/ERNIE.

----

## [2979] Private estimation algorithms for stochastic block models and mixture models

**Authors**: *Hongjie Chen, Vincent Cohen-Addad, Tommaso d'Orsi, Alessandro Epasto, Jacob Imola, David Steurer, Stefan Tiegel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d702d78b2468d2bc80b22a2fc3e59faf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d702d78b2468d2bc80b22a2fc3e59faf-Abstract-Conference.html)

**Abstract**:

We introduce general tools for designing efficient private estimation algorithms, in the high-dimensional settings, whose statistical guarantees almost match those of the best known non-private algorithms.To illustrate our techniques, we consider two problems: recovery of stochastic block models and learning mixtures of spherical Gaussians.For the former, we present the first efficient $(\epsilon, \delta)$-differentially private algorithm for both weak recovery and exact recovery. Previously known algorithms achieving comparable guarantees required quasi-polynomial time. For the latter, we design an  $(\epsilon, \delta)$-differentially private algorithm that recovers the centers of the $k$-mixture when the minimum separation is at least $O(k^{1/t}\sqrt{t})$. For all choices of $t$, this algorithm requires sample complexity $n\geq k^{O(1)}d^{O(t)}$ and time complexity $(nd)^{O(t)}$. Prior work required either an additional additive $\Omega(\sqrt{\log n})$ term in the minimum separation or an explicit upper bound on the Euclidean norm of the centers.

----

## [2980] UP-NeRF: Unconstrained Pose Prior-Free Neural Radiance Field

**Authors**: *Injae Kim, Minhyuk Choi, Hyunwoo J. Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d705dd6e77decdc399162d6d5b92f6e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d705dd6e77decdc399162d6d5b92f6e8-Abstract-Conference.html)

**Abstract**:

Neural Radiance Field (NeRF) has enabled novel view synthesis with high fidelity given images and camera poses. Subsequent works even succeeded in eliminating the necessity of pose priors by jointly optimizing NeRF and camera pose. However, these works are limited to relatively simple settings such as photometrically consistent and occluder-free image collections or a sequence of images from a video. So they have difficulty handling unconstrained images with varying illumination and transient occluders. In this paper, we propose UP-NeRF (Unconstrained Pose-prior-free Neural Radiance Fields) to optimize NeRF with unconstrained image collections without camera pose prior. We tackle these challenges with surrogate tasks that optimize color-insensitive feature fields and a separate module for transient occluders to block their influence on pose estimation. In addition, we introduce a candidate head to enable more robust pose estimation and transient-aware depth supervision to minimize the effect of incorrect prior. Our experiments verify the superior performance of our method compared to the baselines including BARF and its variants in a challenging internet photo collection, Phototourism dataset. The code of UP-NeRF is available at https://github.com/mlvlab/UP-NeRF.

----

## [2981] Nearly Optimal Bounds for Cyclic Forgetting

**Authors**: *William Swartworth, Deanna Needell, Rachel A. Ward, Mark Kong, Halyun Jeong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d72ae75abaa70a3b19c5d4f436c680d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d72ae75abaa70a3b19c5d4f436c680d1-Abstract-Conference.html)

**Abstract**:

We provide theoretical bounds on the forgetting quantity in the continual learning setting for linear tasks, where each round of learning corresponds to projecting onto a linear subspace. For a cyclic task ordering on $T$ tasks repeated $m$ times each, we prove the best known upper bound of $O(T^2/m)$ on the forgetting. Notably, our bound holds uniformly over all choices of tasks and is independent of the ambient dimension. Our main technical contribution is a characterization of the union of all numerical ranges of products of $T$ (real or complex) projections as a sinusoidal spiral, which may be of independent interest.

----

## [2982] ProteinInvBench: Benchmarking Protein Inverse Folding on Diverse Tasks, Models, and Metrics

**Authors**: *Zhangyang Gao, Cheng Tan, Yijie Zhang, Xingran Chen, Lirong Wu, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d73078d49799693792fb0f3f32c57fc8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d73078d49799693792fb0f3f32c57fc8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Protein inverse folding has attracted increasing attention in recent years. However, we observe that current methods are usually limited to the CATH dataset and the recovery metric. The lack of a unified framework for ensembling and comparing different methods hinders the comprehensive investigation. In this paper, we propose ProteinBench, a new benchmark for protein design, which comprises extended protein design tasks, integrated models, and diverse evaluation metrics. We broaden the application of methods originally designed for single-chain protein design to new scenarios of multi-chain and \textit{de novo} protein design. Recent impressive methods, including GraphTrans, StructGNN, GVP, GCA, AlphaDesign, ProteinMPNN, PiFold and KWDesign are integrated into our framework. In addition to the recovery, we also evaluate the confidence, diversity, sc-TM, efficiency, and robustness to thoroughly revisit current protein design approaches and inspire future work. As a result, we establish the first comprehensive benchmark for protein design, which is publicly available at \url{https://github.com/A4Bio/OpenCPD}.

----

## [2983] Understanding and Improving Feature Learning for Out-of-Distribution Generalization

**Authors**: *Yongqiang Chen, Wei Huang, Kaiwen Zhou, Yatao Bian, Bo Han, James Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d73d5645ddbb9ada6c862116435574f6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d73d5645ddbb9ada6c862116435574f6-Abstract-Conference.html)

**Abstract**:

A common explanation for the failure of out-of-distribution (OOD) generalization is that the model trained with empirical risk minimization (ERM) learns spurious features instead of invariant features. However, several recent studies challenged this explanation and found that deep networks may have already learned sufficiently good features for OOD generalization. Despite the contradictions at first glance, we theoretically show that ERM essentially learns both spurious and invariant features, while ERM tends to learn spurious features faster if the spurious correlation is stronger. Moreover, when fed the ERM learned features to the OOD objectives, the invariant feature learning quality significantly affects the final OOD performance, as OOD objectives rarely learn new features. Therefore, ERM feature learning can be a bottleneck to OOD generalization. To alleviate the reliance, we propose Feature Augmented Training (FeAT), to enforce the model to learn richer features ready for OOD generalization. FeAT iteratively augments the model to learn new features while retaining the already learned features. In each round, the retention and augmentation operations are performed on different subsets of the training data that capture distinct features. Extensive experiments show that FeAT effectively learns richer features thus boosting the performance of various OOD objectives.

----

## [2984] Train Hard, Fight Easy: Robust Meta Reinforcement Learning

**Authors**: *Ido Greenberg, Shie Mannor, Gal Chechik, Eli A. Meirom*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d74e6bfe9ce029526e69db14d2c281ec-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d74e6bfe9ce029526e69db14d2c281ec-Abstract-Conference.html)

**Abstract**:

A major challenge of reinforcement learning (RL) in real-world applications is the variation between environments, tasks or clients. Meta-RL (MRL) addresses this issue by learning a meta-policy that adapts to new tasks. Standard MRL methods optimize the average return over tasks, but often suffer from poor results in tasks of high risk or difficulty. This limits system reliability since test tasks are not known in advance. In this work, we define a robust MRL objective with a controlled robustness level. Optimization of analogous robust objectives in RL is known to lead to both biased gradients and data inefficiency. We prove that the gradient bias disappears in our proposed MRL framework. The data inefficiency is addressed via the novel Robust Meta RL algorithm (RoML). RoML is a meta-algorithm that generates a robust version of any given MRL algorithm, by identifying and over-sampling harder tasks throughout training. We demonstrate that RoML achieves robust returns on multiple navigation and continuous control benchmarks.

----

## [2985] Towards Semi-Structured Automatic ICD Coding via Tree-based Contrastive Learning

**Authors**: *Chang Lu, Chandan K. Reddy, Ping Wang, Yue Ning*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d74f9efa1d8ca30b31d65cef8de7c2bf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d74f9efa1d8ca30b31d65cef8de7c2bf-Abstract-Conference.html)

**Abstract**:

Automatic coding of International Classification of Diseases (ICD) is a multi-label text categorization task that involves extracting disease or procedure codes from clinical notes. Despite the application of state-of-the-art natural language processing (NLP) techniques, there are still challenges including limited availability of data due to privacy constraints and the high variability of clinical notes caused by different writing habits of medical professionals and various pathological features of patients. In this work, we investigate the semi-structured nature of clinical notes and propose an automatic algorithm to segment them into sections. To address the variability issues in existing ICD coding models with limited data, we introduce a contrastive pre-training approach on sections using a soft multi-label similarity metric based on tree edit distance. Additionally, we design a masked section training strategy to enable ICD coding models to locate sections related to ICD codes. Extensive experimental results demonstrate that our proposed training strategies effectively enhance the performance of existing ICD coding methods.

----

## [2986] Stable Vectorization of Multiparameter Persistent Homology using Signed Barcodes as Measures

**Authors**: *David Loiseaux, Luis Scoccola, Mathieu Carrière, Magnus Bakke Botnan, Steve Oudot*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d75c474bc01735929a1fab5d0de3b189-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d75c474bc01735929a1fab5d0de3b189-Abstract-Conference.html)

**Abstract**:

Persistent homology (PH) provides  topological descriptors for geometric data, such as weighted graphs, which are interpretable, stable to perturbations, and invariant under, e.g., relabeling. Most applications of PH focus on the one-parameter case---where the descriptors summarize the changes in topology of data as it is filtered by a single quantity of interest---and there is now a wide array of methods enabling the use of one-parameter PH descriptors in data science, which rely on the stable vectorization of these descriptors as elements of a Hilbert space. Although the multiparameter PH (MPH) of data that is filtered by several quantities of interest encodes much richer information than its one-parameter counterpart, the scarceness of stability results for MPH descriptors has so far limited the available options for the stable vectorization of MPH. In this paper, we aim to bring together the best of both worlds by showing how the interpretation of signed barcodes---a recent family of MPH descriptors---as signed Radon measures leads to natural extensions of vectorization strategies from one parameter to multiple parameters. The resulting feature vectors are easy to define and to compute, and provably stable. While, as a proof of concept, we focus on simple choices of signed barcodes and vectorizations, we already see notable performance improvements when comparing our feature vectors to state-of-the-art topology-based methods on various types of data.

----

## [2987] Subclass-Dominant Label Noise: A Counterexample for the Success of Early Stopping

**Authors**: *Yingbin Bai, Zhongyi Han, Erkun Yang, Jun Yu, Bo Han, Dadong Wang, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d763b4a2dde0ae7b77498516ce9f439e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d763b4a2dde0ae7b77498516ce9f439e-Abstract-Conference.html)

**Abstract**:

In this paper, we empirically investigate a previously overlooked and widespread type of label noise, subclass-dominant label noise (SDN). Our findings reveal that, during the early stages of training, deep neural networks can rapidly memorize mislabeled examples in SDN. This phenomenon poses challenges in effectively selecting confident examples using conventional early stopping techniques. To address this issue, we delve into the properties of SDN and observe that long-trained representations are superior at capturing the high-level semantics of mislabeled examples, leading to a clustering effect where similar examples are grouped together. Based on this observation, we propose a novel method called NoiseCluster that leverages the geometric structures of long-trained representations to identify and correct SDN. Our experiments demonstrate that NoiseCluster outperforms state-of-the-art baselines on both synthetic and real-world datasets, highlighting the importance of addressing SDN in learning with noisy labels. The code is available at https://github.com/tmllab/2023NeurIPSSDN.

----

## [2988] OpenMask3D: Open-Vocabulary 3D Instance Segmentation

**Authors**: *Ayça Takmaz, Elisabetta Fedele, Robert W. Sumner, Marc Pollefeys, Federico Tombari, Francis Engelmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d77b5482e38339a8068791d939126be2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d77b5482e38339a8068791d939126be2-Abstract-Conference.html)

**Abstract**:

We introduce the task of open-vocabulary 3D instance segmentation. Current approaches for 3D instance segmentation can typically only recognize object categories from a pre-defined closed set of classes that are annotated in the training datasets. This results in important limitations for real-world applications where one might need to perform tasks guided by novel, open-vocabulary queries related to a wide variety of objects. Recently, open-vocabulary 3D scene understanding methods have emerged to address this problem by learning queryable features for each point in the scene. While such a representation can be directly employed to perform semantic segmentation, existing methods cannot separate multiple object instances. In this work, we address this limitation, and propose OpenMask3D, which is a zero-shot approach for open-vocabulary 3D instance segmentation. Guided by predicted class-agnostic 3D instance masks, our model aggregates per-mask features via multi-view fusion of CLIP-based image embeddings. Experiments and ablation studies on ScanNet200 and Replica show that OpenMask3D outperforms other open-vocabulary methods, especially on the long-tail distribution. Qualitative experiments further showcase OpenMask3Dâ€™s ability to segment object properties based on free-form queries describing geometry, affordances, and materials.

----

## [2989] Model-Based Reparameterization Policy Gradient Methods: Theory and Practical Algorithms

**Authors**: *Shenao Zhang, Boyi Liu, Zhaoran Wang, Tuo Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d78e9e4316e1714fbb0f20be66f8044c-Abstract-Conference.html)

**Abstract**:

ReParameterization (RP) Policy Gradient Methods (PGMs) have been widely adopted for continuous control tasks in robotics and computer graphics. However, recent studies have revealed that, when applied to long-term reinforcement learning problems, model-based RP PGMs may experience chaotic and non-smooth optimization landscapes with exploding gradient variance, which leads to slow convergence. This is in contrast to the conventional belief that reparameterization methods have low gradient estimation variance in problems such as training deep generative models. To comprehend this phenomenon, we conduct a theoretical examination of model-based RP PGMs and search for solutions to the optimization difficulties. Specifically, we analyze the convergence of the model-based RP PGMs and pinpoint the smoothness of function approximators as a major factor that affects the quality of gradient estimation. Based on our analysis, we propose a spectral normalization method to mitigate the exploding variance issue caused by long model unrolls. Our experimental results demonstrate that proper normalization significantly reduces the gradient variance of model-based RP PGMs. As a result, the performance of the proposed method is comparable or superior to other gradient estimators, such as the Likelihood Ratio (LR) gradient estimator. Our code is available at https://github.com/agentification/RP_PGM.

----

## [2990] Bitstream-Corrupted Video Recovery: A Novel Benchmark Dataset and Method

**Authors**: *Tianyi Liu, Kejun Wu, Yi Wang, Wenyang Liu, Kim-Hui Yap, Lap-Pui Chau*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d7928f6dfb0c30d6a6917587dacbe4bc-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d7928f6dfb0c30d6a6917587dacbe4bc-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The past decade has witnessed great strides in video recovery by specialist technologies, like video inpainting, completion, and error concealment. However, they typically simulate the missing content by manual-designed error masks, thus failing to fill in the realistic video loss in video communication (e.g., telepresence, live streaming, and internet video) and multimedia forensics. To address this, we introduce the bitstream-corrupted video (BSCV) benchmark, the first benchmark dataset with more than 28,000 video clips, which can be used for bitstream-corrupted video recovery in the real world. The BSCV is a collection of 1) a proposed three-parameter corruption model for video bitstream, 2) a large-scale dataset containing rich error patterns, multiple corruption levels, and flexible dataset branches, and 3) a new video recovery framework that serves as a benchmark. We evaluate state-of-the-art video inpainting methods on the BSCV dataset, demonstrating existing approaches' limitations and our framework's advantages in solving the bitstream-corrupted video recovery problem. The benchmark and dataset are released at https://github.com/LIUTIGHE/BSCV-Dataset.

----

## [2991] Structured Neural-PI Control with End-to-End Stability and Output Tracking Guarantees

**Authors**: *Wenqi Cui, Yan Jiang, Baosen Zhang, Yuanyuan Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d79c1390baa2e4835586b094d82e5ffb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d79c1390baa2e4835586b094d82e5ffb-Abstract-Conference.html)

**Abstract**:

We study the optimal control of multiple-input and multiple-output dynamical systems via the design of neural network-based controllers with  stability and output tracking guarantees. While neural network-based nonlinear controllers have shown superior performance in various applications, their lack of provable guarantees has restricted their adoption in high-stake real-world applications. This paper bridges the gap between neural network-based controllers and the need for stabilization guarantees. Using equilibrium-independent passivity, a property present in a wide range of physical systems, we propose neural Proportional-Integral (PI) controllers that have provable guarantees of stability and zero steady-state output tracking error. The key structure is the strict monotonicity on proportional and integral terms, which is parameterized as gradients of strictly convex neural networks (SCNN). We construct SCNN with tunable softplus-$\beta$ activations, which yields universal approximation capability and is also useful in incorporating communication constraints. In addition, the SCNNs serve as Lyapunov functions, giving us end-to-end performance guarantees.  Experiments on traffic and power networks demonstrate that the proposed approach improves both transient and steady-state performances, while unstructured neural networks lead to unstable behaviors.

----

## [2992] DäRF: Boosting Radiance Fields from Sparse Input Views with Monocular Depth Adaptation

**Authors**: *Jiuhn Song, Seonghoon Park, Honggyu An, Seokju Cho, Minseop Kwak, Sungjin Cho, Seungryong Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d7a6f4830a18b6974326310478bfa489-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d7a6f4830a18b6974326310478bfa489-Abstract-Conference.html)

**Abstract**:

Neural radiance field (NeRF) shows powerful performance in novel view synthesis and 3D geometry reconstruction, but it suffers from critical performance degradation when the number of known viewpoints is drastically reduced. Existing works attempt to overcome this problem by employing external priors, but their success is limited to certain types of scenes or datasets. Employing monocular depth estimation (MDE) networks, pretrained on large-scale RGB-D datasets, with powerful generalization capability may be a key to solving this problem: however, using MDE in conjunction with NeRF comes with a new set of challenges due to various ambiguity problems exhibited by monocular depths. In this light, we propose a novel framework, dubbed D채RF, that achieves robust NeRF reconstruction with a handful of real-world images by combining the strengths of NeRF and monocular depth estimation through online complementary training. Our framework imposes the MDE network's powerful geometry prior to NeRF representation at both seen and unseen viewpoints to enhance its robustness and coherence. In addition, we overcome the ambiguity problems of monocular depths through patch-wise scale-shift fitting and geometry distillation, which adapts the MDE network to produce depths aligned accurately with NeRF geometry. Experiments show our framework achieves state-of-the-art results both quantitatively and qualitatively, demonstrating consistent and reliable performance in both indoor and outdoor real-world datasets.

----

## [2993] Enhancing Knowledge Transfer for Task Incremental Learning with Data-free Subnetwork

**Authors**: *Qiang Gao, Xiaojun Shan, Yuchen Zhang, Fan Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d7b3cef7c31b94a4a533db83d01a8882-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d7b3cef7c31b94a4a533db83d01a8882-Abstract-Conference.html)

**Abstract**:

As there exist competitive subnetworks within a dense network in concert with Lottery Ticket Hypothesis, we introduce a novel neuron-wise task incremental learning method, namely Data-free Subnetworks (DSN), which attempts to enhance the elastic knowledge transfer across the tasks that sequentially arrive. Specifically, DSN primarily seeks to transfer knowledge to the new coming task from the learned tasks by selecting the affiliated weights of a small set of neurons to be activated, including the reused neurons from prior tasks via neuron-wise masks. And it also transfers possibly valuable knowledge to the earlier tasks via data-free replay. Especially, DSN inherently relieves the catastrophic forgetting and the unavailability of past data or possible privacy concerns. The comprehensive experiments conducted on four benchmark datasets demonstrate the effectiveness of the proposed DSN in the context of task-incremental learning by comparing it to several state-of-the-art baselines. In particular, DSN enables the knowledge transfer to the earlier tasks, which is often overlooked by prior efforts.

----

## [2994] rPPG-Toolbox: Deep Remote PPG Toolbox

**Authors**: *Xin Liu, Girish Narayanswamy, Akshay Paruchuri, Xiaoyu Zhang, Jiankai Tang, Yuzhe Zhang, Roni Sengupta, Shwetak N. Patel, Yuntao Wang, Daniel McDuff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d7d0d548a6317407e02230f15ce75817-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Camera-based physiological measurement is a fast growing field of computer vision. Remote photoplethysmography (rPPG) utilizes imaging devices (e.g., cameras) to measure the peripheral blood volume pulse (BVP) via photoplethysmography, and enables cardiac measurement via webcams and smartphones. However, the task is non-trivial with important pre-processing, modeling and post-processing steps required to obtain state-of-the-art results. Replication of results and benchmarking of new models is critical for scientific progress; however, as with many other applications of deep learning, reliable codebases are not easy to find or use. We present a comprehensive toolbox, rPPG-Toolbox, unsupervised and supervised rPPG models with support for public benchmark datasets, data augmentation and systematic evaluation: https://github.com/ubicomplab/rPPG-Toolbox.

----

## [2995] Proximity-Informed Calibration for Deep Neural Networks

**Authors**: *Miao Xiong, Ailin Deng, Pang Wei W. Koh, Jiaying Wu, Shen Li, Jianqing Xu, Bryan Hooi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d826f5aadb26db488b8686097ceea2d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d826f5aadb26db488b8686097ceea2d1-Abstract-Conference.html)

**Abstract**:

Confidence calibration is central to providing accurate and interpretable uncertainty estimates, especially under safety-critical scenarios. However, we find that existing calibration algorithms often overlook the issue of proximity bias, a phenomenon where models tend to be more overconfident in low proximity data (i.e., data lying in the sparse region of the data distribution) compared to high proximity samples, and thus suffer from inconsistent miscalibration across different proximity samples. We examine the problem over $504$ pretrained ImageNet models and observe that: 1) Proximity bias exists across a wide variety of model architectures and sizes; 2) Transformer-based models are relatively more susceptible to proximity bias than CNN-based models; 3) Proximity bias persists even after performing popular calibration algorithms like temperature scaling; 4) Models tend to overfit more heavily on low proximity samples than on high proximity samples. Motivated by the empirical findings, we propose ProCal, a plug-and-play algorithm with a theoretical guarantee to adjust sample confidence based on proximity. To further quantify the effectiveness of calibration algorithms in mitigating proximity bias, we introduce proximity-informed expected calibration error (PIECE) with theoretical analysis. We show that ProCal is effective in addressing proximity bias and improving calibration on balanced, long-tail, and distribution-shift settings under four metrics over various model architectures. We believe our findings on proximity bias will guide the development of fairer and better-calibrated} models, contributing to the broader pursuit of trustworthy AI.

----

## [2996] Toolformer: Language Models Can Teach Themselves to Use Tools

**Authors**: *Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html)

**Abstract**:

Language models (LMs) exhibit remarkable abilities to solve new tasks from just a few examples or textual instructions, especially at scale. They also, paradoxically, struggle with basic functionality, such as arithmetic or factual lookup, where much simpler and smaller specialized models excel. In this paper, we show that LMs can teach themselves to use external tools via simple APIs and achieve the best of both worlds. We introduce Toolformer, a model trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction. This is done in a self-supervised way, requiring nothing more than a handful of demonstrations for each API. We incorporate a range of tools, including a calculator, a Q&A system, a search engine, a translation system, and a calendar. Toolformer achieves substantially improved zero-shot performance across a variety of downstream tasks, often competitive with much larger models, without sacrificing its core language modeling abilities.

----

## [2997] The probability flow ODE is provably fast

**Authors**: *Sitan Chen, Sinho Chewi, Holden Lee, Yuanzhi Li, Jianfeng Lu, Adil Salim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d84a27ff694345aacc21c72097a69ea2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d84a27ff694345aacc21c72097a69ea2-Abstract-Conference.html)

**Abstract**:

We provide the first polynomial-time convergence guarantees for the probabilistic flow ODE implementation (together with a corrector step) of score-based generative modeling. Our analysis is carried out in the wake of recent results obtaining such guarantees for the SDE-based implementation (i.e., denoising diffusion probabilistic modeling or DDPM), but requires the development of novel techniques for studying deterministic dynamics without contractivity. Through the use of a specially chosen corrector step based on the underdamped Langevin diffusion, we obtain better dimension dependence than prior works on DDPM ($O(\sqrt d)$ vs. $O(d)$, assuming smoothness of the data distribution), highlighting potential advantages of the ODE framework.

----

## [2998] Faster Discrete Convex Function Minimization with Predictions: The M-Convex Case

**Authors**: *Taihei Oki, Shinsaku Sakaue*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d84c0dd9b1bfeee361f3268dcaebf849-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d84c0dd9b1bfeee361f3268dcaebf849-Abstract-Conference.html)

**Abstract**:

Recent years have seen a growing interest in accelerating optimization algorithms with machine-learned predictions. Sakaue and Oki (NeurIPS 2022) have developed a general framework that warm-starts the L-convex function minimization method with predictions, revealing the idea's usefulness for various discrete optimization problems. In this paper, we present a framework for using predictions to accelerate M-convex function minimization, thus complementing previous research and extending the range of discrete optimization algorithms that can benefit from predictions. Our framework is particularly effective for an important subclass called laminar convex minimization, which appears in many operations research applications. Our methods can improve time complexity bounds upon the best worst-case results by using predictions and even have potential to go beyond a lower-bound result.

----

## [2999] Using Imperfect Surrogates for Downstream Inference: Design-based Supervised Learning for Social Science Applications of Large Language Models

**Authors**: *Naoki Egami, Musashi Hinck, Brandon M. Stewart, Hanying Wei*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d862f7f5445255090de13b825b880d59-Abstract-Conference.html)

**Abstract**:

In computational social science (CSS), researchers analyze documents to explain social and political phenomena. In most scenarios, CSS researchers first obtain labels for documents and then explain labels using interpretable regression analyses in the second step. One increasingly common way to annotate documents cheaply at scale is through large language models (LLMs). However, like other scalable ways of producing annotations, such surrogate labels are often imperfect and biased. We present a new algorithm for using imperfect annotation surrogates for downstream statistical analyses while guaranteeing statistical properties—like asymptotic unbiasedness and proper uncertainty quantification—which are fundamental to CSS research. We show that direct use of surrogate labels in downstream statistical analyses leads to substantial bias and invalid confidence intervals, even with high surrogate accuracy of 80-90\%. To address this, we build on debiased machine learning to propose the design-based supervised learning (DSL) estimator. DSL employs a doubly-robust procedure to combine surrogate labels with a smaller number of high-quality, gold-standard labels. Our approach guarantees valid inference for downstream statistical analyses, even when surrogates are arbitrarily biased and without requiring stringent assumptions, by controlling the probability of sampling documents for gold-standard labeling. Both our theoretical analysis and experimental results show that DSL provides valid statistical inference while achieving root mean squared errors comparable to existing alternatives that focus only on prediction without inferential guarantees.

----



[Go to the previous page](NIPS-2023-list14.md)

[Go to the next page](NIPS-2023-list16.md)

[Go to the catalog section](README.md)