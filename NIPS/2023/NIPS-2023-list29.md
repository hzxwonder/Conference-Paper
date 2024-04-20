## [0] Generator Identification for Linear SDEs with Additive and Multiplicative Noise

**Authors**: *Yuanyuan Wang, Xi Geng, Wei Huang, Biwei Huang, Mingming Gong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca642f8e1174012d67c05c1c9f969644-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca642f8e1174012d67c05c1c9f969644-Abstract-Conference.html)

**Abstract**:

In this paper, we present conditions for identifying the generator of a linear stochastic differential equation (SDE) from the distribution of its solution process with a given fixed initial state. These identifiability conditions are crucial in causal inference using linear SDEs as they enable the identification of the post-intervention distributions from its observational distribution. Specifically, we derive a sufficient and necessary condition for identifying the generator of linear SDEs with additive noise, as well as a sufficient condition for identifying the generator of linear SDEs with multiplicative noise. We show that the conditions derived for both types of SDEs are generic. Moreover, we offer geometric interpretations of the derived identifiability conditions to enhance their understanding. To validate our theoretical results, we perform a series of simulations, which support and substantiate the established findings.

----

## [0] Post-processing Private Synthetic Data for Improving Utility on Selected Measures

**Authors**: *Hao Wang, Shivchander Sudalairaj, John Henning, Kristjan H. Greenewald, Akash Srivastava*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca6980a3dba7fb3e4e66925656dba68b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca6980a3dba7fb3e4e66925656dba68b-Abstract-Conference.html)

**Abstract**:

Existing private synthetic data generation algorithms are agnostic to downstream tasks. However, end users may have specific requirements that the synthetic data must satisfy. Failure to meet these requirements could significantly reduce the utility of the data for downstream use. We introduce a post-processing technique that improves the utility of the synthetic data with respect to measures selected by the end user, while preserving strong privacy guarantees and dataset quality. Our technique involves resampling from the synthetic data to filter out samples that do not meet the selected utility measures, using an efficient stochastic first-order algorithm to find optimal resampling weights. Through comprehensive numerical experiments, we demonstrate that our approach consistently improves the utility of synthetic data across multiple benchmark datasets and state-of-the-art synthetic data generation algorithms.

----

## [0] A Bayesian Approach To Analysing Training Data Attribution In Deep Learning

**Authors**: *Elisa Nguyen, Minjoon Seo, Seong Joon Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html)

**Abstract**:

Training data attribution (TDA) techniques find influential training data for the model's prediction on the test data of interest. They approximate the impact of down- or up-weighting a particular training sample. While conceptually useful, they are hardly applicable to deep models in practice, particularly because of their sensitivity to different model initialisation. In this paper, we introduce a Bayesian perspective on the TDA task, where the learned model is treated as a Bayesian posterior and the TDA estimates as random variables. From this novel viewpoint, we observe that the influence of an individual training sample is often overshadowed by the noise stemming from model initialisation and SGD batch composition. Based on this observation, we argue that TDA can only be reliably used for explaining deep model predictions that are consistently influenced by certain training data, independent of other noise factors. Our experiments demonstrate the rarity of such noise-independent training-test data pairs but confirm their existence. We recommend that future researchers and practitioners trust TDA estimates only in such cases. Further, we find a disagreement between ground truth and estimated TDA distributions and encourage future work to study this gap. Code is provided at https://github.com/ElisaNguyen/bayesian-tda.

----

## [0] GPEX, A Framework For Interpreting Artificial Neural Networks

**Authors**: *Amir Akbarnejad, Gilbert Bigras, Nilanjan Ray*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca8c6f28d8ba1e732e3f217ab05c4ec0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca8c6f28d8ba1e732e3f217ab05c4ec0-Abstract-Conference.html)

**Abstract**:

The analogy between Gaussian processes (GPs) and deep artificial neural networks (ANNs) has received a lot of interest, and has shown promise to unbox the blackbox of deep ANNs. Existing theoretical works put strict assumptions on the ANN (e.g. requiring all intermediate layers to be wide, or using specific activation functions). Accommodating those theoretical assumptions is hard in recent deep architectures, and those theoretical conditions need refinement as new deep architectures emerge. In this paper we derive an evidence lower-bound that encourages the GP's posterior to match the ANN's output without any requirement on the ANN. Using our method we find out that on 5 datasets, only a subset of those theoretical assumptions are sufficient. Indeed, in our experiments we used  a normal ResNet-18 or feed-forward backbone with a single wide layer in the end. One limitation of training GPs is the lack of scalability with respect to the number of inducing points. We use novel computational techniques that allow us to train GPs with hundreds of thousands of inducing points and with GPU acceleration. As shown in our experiments, doing so has been essential to get a close match between the GPs and the ANNs on 5 datasets. We implement our method as a publicly available tool called GPEX: https://github.com/amirakbarnejad/gpex. On 5 datasets (4 image datasets, and 1 biological dataset) and ANNs with 2 types of functionality (classifier or attention-mechanism) we were able to find GPs whose outputs closely match those of the corresponding ANNs. After matching the GPs to the ANNs, we used the GPs' kernel functions to explain the ANNs' decisions. We provide more than 200 explanations (around 30 in the paper and the rest in the supplementary) which are highly interpretable by humans and show the ability of the obtained GPs to unbox the ANNs' decisions.

----

## [0] On the Trade-off of Intra-/Inter-class Diversity for Supervised Pre-training

**Authors**: *Jieyu Zhang, Bohan Wang, Zhengyu Hu, Pang Wei Koh, Alexander J. Ratner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca9567d8ef6b2ea2da0d7eed57b933ee-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca9567d8ef6b2ea2da0d7eed57b933ee-Abstract-Conference.html)

**Abstract**:

Pre-training datasets are critical for building state-of-the-art machine learning models, motivating rigorous study on their impact on downstream tasks. In this work, we study the impact of the trade-off between the intra-class diversity (the number of samples per class) and the inter-class diversity (the number of classes) of a supervised pre-training dataset. Empirically, we found that with the size of the pre-training dataset fixed, the best downstream performance comes with a balance on the intra-/inter-class diversity. To understand the underlying mechanism, we show theoretically that the downstream performance depends monotonically on both types of diversity. Notably, our theory reveals that the optimal class-to-sample ratio (#classes / #samples per class) is invariant to the size of the pre-training dataset, which motivates an application of predicting the optimal number of pre-training classes. We demonstrate the effectiveness of this application by an improvement of around 2 points on the downstream tasks when using ImageNet as the pre-training dataset.

----

## [0] An information-theoretic quantification of the content of communication between brain regions

**Authors**: *Marco Celotto, Jan Bím, Alejandro Tlaie, Vito De Feo, Alessandro Toso, Stefan Lemke, Daniel Chicharro, Hamed Nili, Malte Bieler, Ileana L. Hanganu-Opatz, Tobias Donner, Andrea Brovelli, Stefano Panzeri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ca9eaef07eca2a50fc626cb929617b1c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ca9eaef07eca2a50fc626cb929617b1c-Abstract-Conference.html)

**Abstract**:

Quantifying the amount, content and direction of communication between brain regions is key to understanding brain function. Traditional methods to analyze brain activity based on the Wiener-Granger causality principle quantify the overall information propagated by neural activity between simultaneously recorded brain regions, but do not reveal the information flow about specific features of interest (such as sensory stimuli). Here, we develop a new information theoretic measure termed Feature-specific Information Transfer (FIT), quantifying how much information about a specific feature flows between two regions. FIT merges the Wiener-Granger causality principle with information-content specificity. We first derive FIT and prove analytically its key properties. We then illustrate and test them with simulations of neural activity, demonstrating that FIT identifies, within the total information propagated between regions, the information that is transmitted about specific features. We then analyze three neural datasets obtained with different recording methods, magneto- and electro-encephalography, and spiking activity, to demonstrate the ability of FIT to uncover the content and direction of information flow between brain regions beyond what can be discerned with traditional analytical methods. FIT can improve our understanding of how brain regions communicate by uncovering previously unaddressed feature-specific information flow.

----

## [0] Efficient Sampling of Stochastic Differential Equations with Positive Semi-Definite Models

**Authors**: *Anant Raj, Umut Simsekli, Alessandro Rudi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cab5ae2704d3e01f06a92512a5376b87-Abstract-Conference.html)

**Abstract**:

This paper deals with the problem of efficient sampling from a stochastic differential equation, given the drift function and the diffusion matrix. The proposed approach leverages a recent model for probabilities (Rudi and Ciliberto, 2021) (the positive semi-definite -- PSD model) from which it is possible to obtain independent and identically distributed (i.i.d.) samples at precision $\varepsilon$ with a cost that is $m^2 d \log(1/\varepsilon)$ where $m$ is the dimension of the model, $d$ the dimension of the space. The proposed approach consists in: first, computing the PSD model that satisfies the Fokker-Planck equation (or its fractional variant) associated with the SDE, up to error $\varepsilon$, and then sampling from the resulting PSD model. Assuming some regularity of the Fokker-Planck solution (i.e. $\beta$-times differentiability plus some geometric condition on its zeros) We obtain an algorithm that: (a) in the preparatory phase obtains a PSD model with L2 distance $\varepsilon$ from the solution of the equation, with a model of dimension $m = \varepsilon^{-(d+1)/(\beta-2s)} (\log(1/\varepsilon))^{d+1}$ where $1/2\leq s\leq1$ is the fractional power to the Laplacian, and total computational complexity of $O(m^{3.5} \log(1/\varepsilon))$ and then (b) for Fokker-Planck equation, it is able to produce i.i.d.\ samples with error $\varepsilon$ in Wasserstein-1 distance, with a cost that is $O(d \varepsilon^{-2(d+1)/\beta-2} \log(1/\varepsilon)^{2d+3})$ per sample. This means that, if the probability associated with the SDE is somewhat regular, i.e. $\beta \geq 4d+2$, then the algorithm requires $O(\varepsilon^{-0.88} \log(1/\varepsilon)^{4.5d})$ in the preparatory phase, and $O(\varepsilon^{-1/2}\log(1/\varepsilon)^{2d+2})$ for each sample. Our results suggest that as the true solution gets smoother, we can circumvent the curse of dimensionality without requiring any sort of convexity.

----

## [0] TopoSRL: Topology preserving self-supervised Simplicial Representation Learning

**Authors**: *Hiren Madhu, Sundeep Prabhakar Chepuri*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/caba69fbc9fa0b06241b98a44cab8b31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/caba69fbc9fa0b06241b98a44cab8b31-Abstract-Conference.html)

**Abstract**:

In this paper, we introduce $\texttt{TopoSRL}$, a novel self-supervised learning (SSL) method for simplicial complexes to effectively capture higher-order interactions and preserve topology in the learned representations. $\texttt{TopoSRL}$ addresses the limitations of existing graph-based SSL methods that typically concentrate on pairwise relationships, neglecting long-range dependencies crucial to capture topological information. We propose a new simplicial augmentation technique that generates two views of the simplicial complex that enriches the representations while being efficient. Next, we propose a new simplicial contrastive loss function that contrasts the generated simplices to preserve local and global information present in the simplicial complexes. Extensive experimental results demonstrate the superior performance of $\texttt{TopoSRL}$ compared to state-of-the-art graph SSL techniques and supervised simplicial neural models across various datasets corroborating the efficacy of $\texttt{TopoSRL}$ in processing simplicial complex data in a self-supervised setting.

----

## [0] Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving

**Authors**: *Xiaoyu Tian, Tao Jiang, Longfei Yun, Yucheng Mao, Huitong Yang, Yue Wang, Yilun Wang, Hang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cabfaeecaae7d6540ee797a66f0130b0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Robotic perception requires the modeling of both 3D geometry and semantics. Existing methods typically focus on estimating 3D bounding boxes, neglecting finer geometric details and struggling to handle general, out-of-vocabulary objects. 3D occupancy prediction, which estimates the detailed occupancy states and semantics of a scene, is an emerging task to overcome these limitations.To support 3D occupancy prediction, we develop a label generation pipeline that produces dense, visibility-aware labels for any given scene. This pipeline comprises three stages: voxel densification, occlusion reasoning, and image-guided voxel refinement. We establish two benchmarks, derived from the Waymo Open Dataset and the nuScenes Dataset, namely Occ3D-Waymo and Occ3D-nuScenes benchmarks. Furthermore, we provide an extensive analysis of the proposed dataset with various baseline models. Lastly, we propose a new model, dubbed Coarse-to-Fine Occupancy (CTF-Occ) network, which demonstrates superior performance on the Occ3D benchmarks.The code, data, and benchmarks are released at \url{https://tsinghua-mars-lab.github.io/Occ3D/}.

----

## [0] ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design

**Authors**: *Pascal Notin, Aaron Kollasch, Daniel Ritter, Lood van Niekerk, Steffanie Paul, Han Spinner, Nathan J. Rollins, Ada Shaw, Rose Orenbuch, Ruben Weitzman, Jonathan Frazer, Mafalda Dias, Dinko Franceschi, Yarin Gal, Debora S. Marks*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cac723e5ff29f65e3fcbb0739ae91bee-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Predicting the effects of mutations in proteins is critical to many applications, from understanding genetic disease to designing novel proteins to address our most pressing challenges in climate, agriculture and healthcare. Despite an increase in machine learning-based protein modeling methods, assessing their effectiveness is problematic due to the use of distinct, often contrived, experimental datasets and variable performance across different protein families. Addressing these challenges requires scale. To that end we introduce ProteinGym v1.0, a large-scale and holistic set of benchmarks specifically designed for protein fitness prediction and design. It encompasses both a broad collection of over 250 standardized deep mutational scanning assays, spanning millions of mutated sequences, as well as curated clinical datasets providing high-quality expert annotations about mutation effects. We devise a robust evaluation framework that combines metrics for both fitness prediction and design, factors in known limitations of the underlying experimental methods, and covers both zero-shot and supervised settings. We report the performance of a diverse set of over 40 high-performing models from various subfields (eg., mutation effects, inverse folding) into a unified benchmark. We open source the corresponding codebase, datasets, MSAs, structures, predictions and develop a user-friendly website that facilitates comparisons across all settings.

----

## [0] On the spectral bias of two-layer linear networks

**Authors**: *Aditya Vardhan Varre, Maria-Luiza Vladarean, Loucas Pillaud-Vivien, Nicolas Flammarion*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cad2fd66cf88226d868f90a7cbaa4a53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cad2fd66cf88226d868f90a7cbaa4a53-Abstract-Conference.html)

**Abstract**:

This paper studies the behaviour of two-layer fully connected networks with linear activations trained with gradient flow on the square loss. We show how the optimization process carries an implicit bias on the parameters that depends on the scale of its initialization.  The main result of the paper is a variational characterization of the loss minimizers retrieved by the gradient flow for a specific initialization shape. This characterization reveals that, in the small scale initialization regime, the linear neural network's hidden layer is biased toward having a low-rank structure. To complement our results, we showcase a hidden mirror flow that tracks the dynamics of the singular values of the weights matrices and describe their time evolution. We support our findings with numerical experiments illustrating the phenomena.

----

## [0] GMSF: Global Matching Scene Flow

**Authors**: *Yushan Zhang, Johan Edstedt, Bastian Wandt, Per-Erik Forssén, Maria Magnusson, Michael Felsberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb1c4782f159b55380b4584671c4fd88-Abstract-Conference.html)

**Abstract**:

We tackle the task of scene flow estimation from point clouds. Given a source and a target point cloud, the objective is to estimate a translation from each point in the source point cloud to the target, resulting in a 3D motion vector field. Previous dominant scene flow estimation methods require complicated coarse-to-fine or recurrent architectures as a multi-stage refinement. In contrast, we propose a significantly simpler single-scale one-shot global matching to address the problem. Our key finding is that reliable feature similarity between point pairs is essential and sufficient to estimate accurate scene flow. We thus propose to decompose the feature extraction step via a hybrid local-global-cross transformer architecture which is crucial to accurate and robust feature representations. Extensive experiments show that the proposed Global Matching Scene Flow (GMSF) sets a new state-of-the-art on multiple scene flow estimation benchmarks. On FlyingThings3D, with the presence of occlusion points, GMSF reduces the outlier percentage from the previous best performance of 27.4% to 5.6%. On KITTI Scene Flow, without any fine-tuning, our proposed method shows state-of-the-art performance. On the Waymo-Open dataset, the proposed method outperforms previous methods by a large margin. The code is available at https://github.com/ZhangYushan3/GMSF.

----

## [0] Efficient Uncertainty Quantification and Reduction for Over-Parameterized Neural Networks

**Authors**: *Ziyi Huang, Henry Lam, Haofeng Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb2266111eadcfa2c02187ace64e2183-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb2266111eadcfa2c02187ace64e2183-Abstract-Conference.html)

**Abstract**:

Uncertainty quantification (UQ) is important for reliability assessment and enhancement of machine learning models. In deep learning, uncertainties arise not only from data, but also from the training procedure that often injects substantial noises and biases. These hinder the attainment of statistical guarantees and, moreover, impose computational challenges on UQ due to the need for repeated network retraining. Building upon the recent neural tangent kernel theory, we create statistically guaranteed schemes to principally \emph{characterize}, and \emph{remove}, the uncertainty of over-parameterized neural networks with very low computation effort. In particular, our approach, based on what we call a procedural-noise-correcting (PNC) predictor, removes the procedural uncertainty by using only \emph{one} auxiliary network that is trained on a suitably labeled dataset, instead of many retrained networks employed in deep ensembles. Moreover, by combining our PNC predictor with suitable light-computation resampling methods, we build several approaches to construct asymptotically exact-coverage confidence intervals using as low as four trained networks without additional overheads.

----

## [0] LuminAIRe: Illumination-Aware Conditional Image Repainting for Lighting-Realistic Generation

**Authors**: *Jiajun Tang, Haofeng Zhong, Shuchen Weng, Boxin Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb3658b9983f677670a246c46ece553d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb3658b9983f677670a246c46ece553d-Abstract-Conference.html)

**Abstract**:

We present the ilLumination-Aware conditional Image Repainting (LuminAIRe) task to address the unrealistic lighting effects in recent conditional image repainting (CIR) methods. The environment lighting and 3D geometry conditions are explicitly estimated from given background images and parsing masks using a parametric lighting representation and learning-based priors. These 3D conditions are then converted into illumination images through the proposed physically-based illumination rendering and illumination attention module. With the injection of illumination images, physically-correct lighting information is fed into the lighting-realistic generation process and repainted images with harmonized lighting effects in both foreground and background regions can be acquired, whose superiority over the results of state-of-the-art methods is confirmed through extensive experiments. For facilitating and validating the LuminAIRe task, a new dataset Car-LuminAIRe with lighting annotations and rich appearance variants is collected.

----

## [0] A graphon-signal analysis of graph neural networks

**Authors**: *Ron Levie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb7943be26bb34f036c7e4068c490903-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb7943be26bb34f036c7e4068c490903-Abstract-Conference.html)

**Abstract**:

We present an approach for analyzing message passing graph neural networks (MPNNs) based on an extension of graphon analysis to a so called graphon-signal analysis. A MPNN is a function that takes a graph and a signal on the graph (a graph-signal) and returns some value. Since the input space of MPNNs is non-Euclidean, i.e., graphs can be of any size and topology, properties such as generalization are less well understood for MPNNs than for Euclidean neural networks. We claim that one important missing ingredient in past work is a meaningful notion of graph-signal similarity measure, that endows the space of inputs to MPNNs with a regular structure. We present such a similarity measure, called the graphon-signal cut distance, which makes the space of all graph-signals a dense subset of a compact metric space -- the graphon-signal space. Informally, two deterministic graph-signals are close in cut-distance if they ``look like'' they were sampled from the same random graph-signal model. Hence, our cut distance is a natural notion of graph-signal similarity, which allows comparing any pair of graph-signals of any size and topology. We prove that MPNNs are Lipschitz continuous functions over the graphon-signal metric space. We then give two applications of this result: 1) a generalization bound for MPNNs, and, 2) the stability of MPNNs to subsampling of graph-signals. Our results apply to any regular enough MPNN on any distribution of graph-signals, making the analysis rather universal.

----

## [0] Lo-Hi: Practical ML Drug Discovery Benchmark

**Authors**: *Simon Steshin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb82f1f97ad0ca1d92df852a44a3bd73-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb82f1f97ad0ca1d92df852a44a3bd73-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Finding new drugs is getting harder and harder. One of the hopes of drug discovery is to use machine learning models to predict molecular properties. That is why models for molecular property prediction are being developed and tested on benchmarks such as MoleculeNet. However, existing benchmarks are unrealistic and are too different from applying the models in practice. We have created a new practical \emph{Lo-Hi} benchmark consisting of two tasks: Lead Optimization (Lo) and Hit Identification (Hi), corresponding to the real drug discovery process. For the Hi task, we designed a novel molecular splitting algorithm that solves the Balanced Vertex Minimum $k$-Cut problem.  We tested state-of-the-art and classic ML models, revealing which works better under practical settings. We analyzed modern benchmarks and showed that they are unrealistic and overoptimistic.Review: https://openreview.net/forum?id=H2Yb28qGLVLo-Hi benchmark: https://github.com/SteshinSS/lohi_neurips2023Lo-Hi splitter library: https://github.com/SteshinSS/lohi_splitter

----

## [0] Class-Conditional Conformal Prediction with Many Classes

**Authors**: *Tiffany Ding, Anastasios Angelopoulos, Stephen Bates, Michael I. Jordan, Ryan J. Tibshirani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cb931eddd563f8d473c355518ce8601c-Abstract-Conference.html)

**Abstract**:

Standard conformal prediction methods provide a marginal coverage guarantee,which means that for a random test point, the conformal prediction set contains the true label with a user-specified probability. In many classificationproblems, we would like to obtain a stronger guarantee--that for test pointsof a specific class, the prediction set contains the true label with thesame user-chosen probability. For the latter goal, existing conformal predictionmethods do not work well when there is a limited amount of labeled data perclass, as is often the case in real applications where the number of classes islarge. We propose a method called clustered conformal prediction thatclusters together classes having "similar" conformal scores and performs conformal prediction at the cluster level. Based on empirical evaluation acrossfour image data sets with many (up to 1000) classes, we find that clusteredconformal typically outperforms existing methods in terms of class-conditionalcoverage and set size metrics.

----

## [0] Reward Imputation with Sketching for Contextual Batched Bandits

**Authors**: *Xiao Zhang, Ninglu Shao, Zihua Si, Jun Xu, Wenhan Wang, Hanjing Su, Ji-Rong Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cba76ef96c4cd625631ab4d33285b045-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cba76ef96c4cd625631ab4d33285b045-Abstract-Conference.html)

**Abstract**:

Contextual batched bandit (CBB) is a setting where a batch of rewards is observed from the environment at the end of each episode, but the rewards of the non-executed actions are unobserved, resulting in partial-information feedback. Existing approaches for CBB often ignore the rewards of the non-executed actions, leading to underutilization of feedback information. In this paper, we propose an efficient approach called Sketched Policy Updating with Imputed Rewards (SPUIR) that completes the unobserved rewards using sketching, which approximates the full-information feedbacks. We formulate reward imputation as an imputation regularized ridge regression problem that captures the feedback mechanisms of both executed and non-executed actions. To reduce time complexity, we solve the regression problem using randomized sketching. We prove that our approach achieves an instantaneous regret with controllable bias and smaller variance than approaches without reward imputation. Furthermore, our approach enjoys a sublinear regret bound against the optimal policy. We also present two extensions, a rate-scheduled version and a version for nonlinear rewards, making our approach more practical. Experimental results show that SPUIR outperforms state-of-the-art baselines on synthetic, public benchmark, and real-world datasets.

----

## [0] A Unified Model and Dimension for Interactive Estimation

**Authors**: *Nataly Brukhim, Miro Dudík, Aldo Pacchiano, Robert E. Schapire*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbabc2f70de2dd09f491a8715ec3e80f-Abstract-Conference.html)

**Abstract**:

We study an abstract framework for interactive learning called interactive estimation in which the goal is to estimate a target from its ``similarity'' to points queried by the learner.We introduce a combinatorial measure called Dissimilarity dimension which largely captures learnability in our model.We present a simple, general, and broadly-applicable algorithm, for which we obtain both regret and PAC generalization bounds that are polynomial in the new dimension. We show that our framework subsumes and thereby unifies two classic learning models:statistical-query learning and structured bandits. We also delineate how the Dissimilarity dimension is related to well-known parameters for both frameworks, in some cases yielding significantly improved analyses.

----

## [0] Simple, Scalable and Effective Clustering via One-Dimensional Projections

**Authors**: *Moses Charikar, Monika Henzinger, Lunjia Hu, Maximilian Vötsch, Erik Waingarten*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbaffeeda13dbd8bf9489feb3f198ff4-Abstract-Conference.html)

**Abstract**:

Clustering is a fundamental problem in unsupervised machine learning with many applications in data analysis. Popular clustering algorithms such as Lloyd's algorithm and $k$-means++ can take $\Omega(ndk)$ time when clustering $n$ points in a $d$-dimensional space (represented by an $n\times d$ matrix $X$) into $k$ clusters. On massive datasets with moderate to large $k$, the multiplicative $k$ factor can become very expensive. We introduce a simple randomized clustering algorithm that provably runs in expected time $O(\mathsf{nnz}(X) + n\log n)$ for arbitrary $k$. Here $\mathsf{nnz}(X)$ is the total number of non-zero entries in the input dataset $X$, which is upper bounded by $nd$ and can be significantly smaller for sparse datasets. We prove that our algorithm achieves approximation ratio $\widetilde{O}(k^4)$ on any input dataset for the $k$-means objective, and our experiments show that the quality of the clusters found by our algorithm is usually much better than this worst-case bound. We use our algorithm for $k$-means clustering and for coreset construction; our experiments show that it gives a new tradeoff between running time and cluster quality compared to previous state-of-the-art methods for these tasks. Our theoretical analysis is based on novel results of independent interest. We show that the approximation ratio achieved after a random one-dimensional projection can be lifted to the original points and that $k$-means++ seeding can be implemented in expected time $O(n\log n)$ in one dimension.

----

## [0] Streaming PCA for Markovian Data

**Authors**: *Syamantak Kumar, Purnamrita Sarkar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbb1fa8e7f515e796cda6621a703492f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbb1fa8e7f515e796cda6621a703492f-Abstract-Conference.html)

**Abstract**:

Since its inception in 1982, Oja's algorithm has become an established method for streaming principle component analysis (PCA). We study the problem of streaming PCA, where the data-points are sampled from an irreducible, aperiodic, and reversible Markov chain starting in stationarity. Our goal is to estimate the top eigenvector of the unknown covariance matrix of the stationary distribution. This setting has implications in scenarios where data can solely be sampled from a Markov Chain Monte Carlo (MCMC) type algorithm, and the objective is to perform inference on parameters of the stationary distribution. Most convergence guarantees for Oja's algorithm in the literature assume that the data-points are sampled IID. For data streams with Markovian dependence, one typically downsamples the data to get a "nearly" independent data stream. In this paper, we obtain the first near-optimal rate for Oja's algorithm on the entire data, where we remove the logarithmic dependence on the sample size, $n$, resulting from throwing data away in downsampling strategies.

----

## [0] Generalized Logit Adjustment: Calibrating Fine-tuned Models by Removing Label Bias in Foundation Models

**Authors**: *Beier Zhu, Kaihua Tang, Qianru Sun, Hanwang Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cbe1fd3136e0f049bb8bc104231ccb99-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cbe1fd3136e0f049bb8bc104231ccb99-Abstract-Conference.html)

**Abstract**:

Foundation models like CLIP allow zero-shot transfer on various tasks without additional training data. Yet, the zero-shot performance is less competitive than a fully supervised one. Thus, to enhance the performance, fine-tuning and ensembling are also commonly adopted to better fit the downstream tasks. However, we argue that such prior work has overlooked the inherent biases in foundation models. Due to the highly imbalanced Web-scale training set, these foundation models are inevitably skewed toward frequent semantics, and thus the subsequent fine-tuning or ensembling is still biased. In this study, we systematically examine the biases in foundation models and demonstrate the efficacy of our proposed Generalized Logit Adjustment (GLA) method. Note that bias estimation in foundation models is challenging, as most pre-train data cannot be explicitly assessed like in traditional long-tailed classification tasks.To this end, GLA has an optimization-based bias estimation approach for debiasing foundation models. As our work resolves a fundamental flaw in the pre-training, the proposed GLA demonstrates significant improvements across a diverse range of tasks: it achieves 1.5 pp accuracy gains on ImageNet, an large average improvement (1.4-4.6 pp) on 11 few-shot datasets, 2.4 pp gains on long-tailed classification. Codes are in https://github.com/BeierZhu/GLA.

----

## [0] Learning to Parameterize Visual Attributes for Open-set Fine-grained Retrieval

**Authors**: *Shijie Wang, Jianlong Chang, Haojie Li, Zhihui Wang, Wanli Ouyang, Qi Tian*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc19e4ffde5540ac3fcda240e6d975cb-Abstract-Conference.html)

**Abstract**:

Open-set fine-grained retrieval is an emerging challenging task that allows to retrieve unknown categories beyond the training set. The best solution for handling unknown categories is to represent them using a set of visual attributes learnt from known categories, as widely used in zero-shot learning. Though important, attribute modeling usually requires significant manual annotations and thus is labor-intensive. Therefore, it is worth to investigate how to transform retrieval models trained by image-level supervision from category semantic extraction to attribute modeling. To this end, we propose a novel Visual Attribute Parameterization Network (VAPNet) to learn visual attributes from known categories and parameterize them into the retrieval model, without the involvement of any attribute annotations.In this way, VAPNet could utilize its parameters to parse a set of visual attributes from unknown categories and precisely represent them.Technically, VAPNet explicitly attains some semantics with rich details via making use of local image patches and distills the visual attributes from these discovered semantics. Additionally, it integrates the online refinement of these visual attributes into the training process to iteratively enhance their quality. Simultaneously, VAPNet treats these attributes as supervisory signals to tune the retrieval models, thereby achieving attribute parameterization. Extensive experiments on open-set fine-grained retrieval datasets validate the superior performance of our VAPNet over existing solutions.

----

## [0] Learning List-Level Domain-Invariant Representations for Ranking

**Authors**: *Ruicheng Xian, Honglei Zhuang, Zhen Qin, Hamed Zamani, Jing Lu, Ji Ma, Kai Hui, Han Zhao, Xuanhui Wang, Michael Bendersky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc473bb3ec4176a5e640c3a6b5fb5239-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc473bb3ec4176a5e640c3a6b5fb5239-Abstract-Conference.html)

**Abstract**:

Domain adaptation aims to transfer the knowledge learned on (data-rich) source domains to (low-resource) target domains, and a popular method is invariant representation learning, which matches and aligns the data distributions on the feature space. Although this method is studied extensively and applied on classification and regression problems, its adoption on ranking problems is sporadic, and the few existing implementations lack theoretical justifications. This paper revisits invariant representation learning for ranking. Upon reviewing prior work, we found that they implement what we call item-level alignment, which aligns the distributions of the items being ranked from all lists in aggregate but ignores their list structure. However, the list structure should be leveraged, because it is intrinsic to ranking problems where the data and the metrics are defined and computed on lists, not the items by themselves. To close this discrepancy, we propose list-level alignmentâ€”learning domain-invariant representations at the higher level of lists. The benefits are twofold: it leads to the first domain adaptation generalization bound for ranking, in turn providing theoretical support for the proposed method, and it achieves better empirical transfer performance for unsupervised domain adaptation on ranking tasks, including passage reranking.

----

## [0] Homotopy-based training of NeuralODEs for accurate dynamics discovery

**Authors**: *Joon-Hyuk Ko, Hankyul Koh, Nojun Park, Wonho Jhe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc56ae4929d792351a66c39aafb4a34d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc56ae4929d792351a66c39aafb4a34d-Abstract-Conference.html)

**Abstract**:

Neural Ordinary Differential Equations (NeuralODEs) present an attractive way to extract dynamical laws from time series data, as they bridge neural networks with the differential equation-based modeling paradigm of the physical sciences. However, these models often display long training times and suboptimal results, especially for longer duration data. While a common strategy in the literature imposes strong constraints to the NeuralODE architecture to inherently promote stable model dynamics, such methods are ill-suited for dynamics discovery as the unknown governing equation is not guaranteed to satisfy the assumed constraints. In this paper, we develop a new training method for NeuralODEs, based on synchronization and homotopy optimization, that does not require changes to the model architecture. We show that synchronizing the model dynamics and the training data tames the originally irregular loss landscape, which homotopy optimization can then leverage to enhance training. Through benchmark experiments, we demonstrate our method achieves competitive or better training loss while often requiring less than half the number of training epochs compared to other model-agnostic techniques. Furthermore, models trained with our method display better extrapolation capabilities, highlighting the effectiveness of our method.

----

## [0] Simplifying and Empowering Transformers for Large-Graph Representations

**Authors**: *Qitian Wu, Wentao Zhao, Chenxiao Yang, Hengrui Zhang, Fan Nie, Haitian Jiang, Yatao Bian, Junchi Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc57fac10eacadb3b72a907ac48f9a98-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc57fac10eacadb3b72a907ac48f9a98-Abstract-Conference.html)

**Abstract**:

Learning representations on large-sized graphs is a long-standing challenge due to the inter-dependence nature involved in massive data points. Transformers, as an emerging class of foundation encoders for graph-structured data, have shown promising performance on small graphs due to its global attention capable of capturing all-pair influence beyond neighboring nodes. Even so, existing approaches tend to inherit the spirit of Transformers in language and vision tasks, and embrace complicated models by stacking deep multi-head attentions. In this paper, we critically demonstrate that even using a one-layer attention can bring up surprisingly competitive performance across node property prediction benchmarks where node numbers range from thousand-level to billion-level. This encourages us to rethink the design philosophy for Transformers on large graphs, where the global attention is a computation overhead hindering the scalability. We frame the proposed scheme as Simplified Graph Transformers (SGFormer), which is empowered by a simple attention model that can efficiently propagate information among arbitrary nodes in one layer. SGFormer requires none of positional encodings, feature/graph pre-processing or augmented loss. Empirically, SGFormer successfully scales to the web-scale graph ogbn-papers100M and yields up to 141x inference acceleration over SOTA Transformers on medium-sized graphs. Beyond current results, we believe the proposed methodology alone enlightens a new technical path of independent interest for building Transformers on large graphs.

----

## [0] Understanding the Limitations of Deep Models for Molecular property prediction: Insights and Solutions

**Authors**: *Jun Xia, Lecheng Zhang, Xiao Zhu, Yue Liu, Zhangyang Gao, Bozhen Hu, Cheng Tan, Jiangbin Zheng, Siyuan Li, Stan Z. Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc83e97320000f4e08cb9e293b12cf7e-Abstract-Conference.html)

**Abstract**:

Molecular Property Prediction (MPP) is a crucial task in the AI-driven Drug Discovery (AIDD) pipeline, which has recently gained considerable attention thanks to advancements in deep learning. However, recent research has revealed that deep models struggle to beat traditional non-deep ones on MPP. In this study, we benchmark 12 representative models (3 non-deep models and 9 deep models) on 15 molecule datasets. Through the most comprehensive study to date, we make the following key observations: \textbf{(\romannumeral 1)} Deep models are generally unable to outperform non-deep ones; \textbf{(\romannumeral 2)} The failure of deep models on MPP cannot be solely attributed to the small size of molecular datasets; \textbf{(\romannumeral 3)} In particular, some traditional models including XGB and RF that use molecular fingerprints as inputs tend to perform better than other competitors. Furthermore, we conduct extensive empirical investigations into the unique patterns of molecule data and inductive biases of various models underlying these phenomena. These findings stimulate us to develop a simple-yet-effective feature mapping method for molecule data prior to feeding them into deep models. Empirically, deep models equipped with this mapping method can beat non-deep ones in most MoleculeNet datasets. Notably, the effectiveness is further corroborated by extensive experiments on cutting-edge dataset related to COVID-19 and activity cliff datasets.

----

## [0] Neural Circuits for Fast Poisson Compressed Sensing in the Olfactory Bulb

**Authors**: *Jacob A. Zavatone-Veth, Paul Masset, William L. Tong, Joseph D. Zak, Venkatesh Murthy, Cengiz Pehlevan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cc8638553a347b1834d98be7613fa3f0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cc8638553a347b1834d98be7613fa3f0-Abstract-Conference.html)

**Abstract**:

Within a single sniff, the mammalian olfactory system can decode the identity and concentration of odorants wafted on turbulent plumes of air. Yet, it must do so given access only to the noisy, dimensionally-reduced representation of the odor world provided by olfactory receptor neurons. As a result, the olfactory system must solve a compressed sensing problem, relying on the fact that only a handful of the millions of possible odorants are present in a given scene. Inspired by this principle, past works have proposed normative compressed sensing models for olfactory decoding. However, these models have not captured the unique anatomy and physiology of the olfactory bulb, nor have they shown that sensing can be achieved within the 100-millisecond timescale of a single sniff. Here, we propose a rate-based Poisson compressed sensing circuit model for the olfactory bulb. This model maps onto the neuron classes of the olfactory bulb, and recapitulates salient features of their connectivity and physiology. For circuit sizes comparable to the human olfactory bulb, we show that this model can accurately detect tens of odors within the timescale of a single sniff. We also show that this model can perform Bayesian posterior sampling for accurate uncertainty estimation. Fast inference is possible only if the geometry of the neural code is chosen to match receptor properties, yielding a distributed neural code that is not axis-aligned to individual odor identities. Our results illustrate how normative modeling can help us map function onto specific neural circuits to generate new hypotheses.

----

## [0] SUPA: A Lightweight Diagnostic Simulator for Machine Learning in Particle Physics

**Authors**: *Atul Kumar Sinha, Daniele Paliotta, Bálint Máté, John A. Raine, Tobias Golling, François Fleuret*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cca79c22037280d066fbd8bc35ac2e72-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cca79c22037280d066fbd8bc35ac2e72-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Deep learning methods have gained popularity in high energy physics for fast modeling of particle showers in detectors. Detailed simulation frameworks such as the gold standard \textsc{Geant4} are computationally intensive, and current deep generative architectures work on discretized, lower resolution versions of the detailed simulation. The development of models that work at higher spatial resolutions is currently hindered by the complexity of the full simulation data, and by the lack of simpler, more interpretable benchmarks. Our contribution is \textsc{SUPA}, the SUrrogate PArticle propagation simulator, an algorithm and software package for generating data by simulating simplified particle propagation, scattering and shower development in matter. The generation is extremely fast and easy to use compared to \textsc{Geant4}, but still exhibits the key characteristics and challenges of the detailed simulation. The proposed simulator generates thousands of particle showers per second on a desktop machine, a speed up of up to 6 orders of magnitudes over \textsc{Geant4}, and stores detailed geometric information about the shower propagation. \textsc{\textsc{SUPA}} provides much greater flexibility for setting initial conditions and defining multiple benchmarks for the development of models. Moreover, interpreting particle showers as point clouds creates a connection to geometric machine learning and provides challenging and fundamentally new datasets for the field.

----

## [0] LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

**Authors**: *Artur P. Toshev, Gianluca Galletti, Fabian Fritz, Stefan Adami, Nikolaus A. Adams*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccac3b120c7dc86d45f56830732b62be-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Machine learning has been successfully applied to grid-based PDE modeling in various scientific applications. However, learned PDE solvers based on Lagrangian particle discretizations, which are the preferred approach to problems with free surfaces or complex physics, remain largely unexplored. We present LagrangeBench, the first benchmarking suite for Lagrangian particle problems, focusing on temporal coarse-graining. In particular, our contribution is: (a) seven new fluid mechanics datasets (four in 2D and three in 3D) generated with the Smoothed Particle Hydrodynamics (SPH) method including the Taylor-Green vortex, lid-driven cavity, reverse Poiseuille flow, and dam break, each of which includes different physics like solid wall interactions or free surface, (b) efficient JAX-based API with various recent training strategies and three neighbor search routines, and (c) JAX implementation of established Graph Neural Networks (GNNs) like GNS and SEGNN with baseline results. Finally, to measure the performance of learned surrogates we go beyond established position errors and introduce physical metrics like kinetic energy MSE and Sinkhorn distance for the particle distribution. Our codebase is available under the URL: https://github.com/tumaer/lagrangebench.

----

## [0] Group Fairness in Peer Review

**Authors**: *Haris Aziz, Evi Micha, Nisarg Shah*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccba10dd4e80e7276054222bb95d467c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccba10dd4e80e7276054222bb95d467c-Abstract-Conference.html)

**Abstract**:

Large conferences such as NeurIPS and AAAI serve as crossroads  of various AI fields, since they attract submissions from a vast number of communities. However, in some cases, this has resulted in a poor reviewing experience for some communities, whose submissions get assigned to less qualified reviewers outside of their communities. An often-advocated solution is to break up any such large conference into smaller conferences, but this can lead to isolation of communities and harm interdisciplinary research. We tackle this challenge by introducing a  notion of group fairness, called the core, which requires that every possible community (subset of researchers) to be treated in a way that prevents them from unilaterally benefiting by  withdrawing from a large conference. We study a simple peer review model, prove that it always admits a reviewing assignment in the core, and design an efficient algorithm to find one such assignment. We use real data from CVPR and ICLR conferences to compare our algorithm to existing reviewing assignment algorithms on a number of metrics.

----

## [0] Diffusion Model is an Effective Planner and Data Synthesizer for Multi-Task Reinforcement Learning

**Authors**: *Haoran He, Chenjia Bai, Kang Xu, Zhuoran Yang, Weinan Zhang, Dong Wang, Bin Zhao, Xuelong Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccda3c632cc8590ee60ca5ba226a4c30-Abstract-Conference.html)

**Abstract**:

Diffusion models have demonstrated highly-expressive generative capabilities in vision and NLP. Recent studies in reinforcement learning (RL) have shown that diffusion models are also powerful in modeling complex policies or trajectories in offline datasets. However, these works have been limited to single-task settings where a generalist agent capable of addressing multi-task predicaments is absent. In this paper, we aim to investigate the effectiveness of a single diffusion model in modeling large-scale multi-task offline data, which can be challenging due to diverse and multimodal data distribution. Specifically, we propose Multi-Task Diffusion Model (\textsc{MTDiff}), a diffusion-based method that incorporates Transformer backbones and prompt learning for generative planning and data synthesis in multi-task offline settings. \textsc{MTDiff} leverages vast amounts of knowledge available in multi-task data and performs implicit knowledge sharing among tasks. For generative planning, we find \textsc{MTDiff} outperforms state-of-the-art algorithms across 50 tasks on Meta-World and 8 maps on Maze2D. For data synthesis, \textsc{MTDiff} generates high-quality data for testing tasks given a single demonstration as a prompt, which enhances the low-quality datasets for even unseen tasks.

----

## [0] Single-Call Stochastic Extragradient Methods for Structured Non-monotone Variational Inequalities: Improved Analysis under Weaker Conditions

**Authors**: *Sayantan Choudhury, Eduard Gorbunov, Nicolas Loizou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf02786d28730e8311676ffa842e216-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf02786d28730e8311676ffa842e216-Abstract-Conference.html)

**Abstract**:

Single-call stochastic extragradient methods, like stochastic past extragradient (SPEG) and stochastic optimistic gradient (SOG),  have gained a lot of interest in recent years and are one of the most efficient algorithms for solving large-scale min-max optimization and variational inequalities problems (VIP) appearing in various machine learning tasks. However, despite their undoubted popularity, current convergence analyses of SPEG and SOG require strong assumptions like bounded variance or growth conditions. In addition, several important questions regarding the convergence properties of these methods are still open, including mini-batching, efficient step-size selection, and convergence guarantees under different sampling strategies. In this work, we address these questions and provide convergence guarantees for two large classes of structured non-monotone VIPs: (i) quasi-strongly monotone problems (a generalization of strongly monotone problems) and (ii) weak Minty variational inequalities (a generalization of monotone and Minty VIPs). We introduce the expected residual condition, explain its benefits, and show how it allows us to obtain a strictly weaker bound than previously used growth conditions, expected co-coercivity, or bounded variance assumptions. Finally, our convergence analysis holds under the arbitrary sampling paradigm, which includes importance sampling and various mini-batching strategies as special cases.

----

## [0] Assessor360: Multi-sequence Network for Blind Omnidirectional Image Quality Assessment

**Authors**: *Tianhe Wu, Shuwei Shi, Haoming Cai, Mingdeng Cao, Jing Xiao, Yinqiang Zheng, Yujiu Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf4a7323b9ee3e54bf77f0e876b3f8b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf4a7323b9ee3e54bf77f0e876b3f8b-Abstract-Conference.html)

**Abstract**:

Blind Omnidirectional Image Quality Assessment (BOIQA) aims to objectively assess the human perceptual quality of omnidirectional images (ODIs) without relying on pristine-quality image information. It is becoming more significant with the increasing advancement of virtual reality (VR) technology. However, the quality assessment of ODIs is severely hampered by the fact that the existing BOIQA pipeline lacks the modeling of the observer's browsing process. To tackle this issue, we propose a novel multi-sequence network for BOIQA called Assessor360, which is derived from the realistic multi-assessor ODI quality assessment procedure. Specifically, we propose a generalized Recursive Probability Sampling (RPS) method for the BOIQA task, combining content and details information to generate multiple pseudo viewport sequences from a given starting point. Additionally, we design a Multi-scale Feature Aggregation (MFA) module with a Distortion-aware Block (DAB) to fuse distorted and semantic features of each viewport. We also devise Temporal Modeling Module (TMM) to learn the viewport transition in the temporal domain. Extensive experimental results demonstrate that Assessor360 outperforms state-of-the-art methods on multiple OIQA datasets. The code and models are available at https://github.com/TianheWu/Assessor360.

----

## [0] Lossy Image Compression with Conditional Diffusion Models

**Authors**: *Ruihan Yang, Stephan Mandt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ccf6d8b4a1fe9d9c8192f00c713872ea-Abstract-Conference.html)

**Abstract**:

This paper outlines an end-to-end optimized lossy image compression framework using diffusion generative models. The approach relies on the transform coding paradigm, where an image is mapped into a latent space for entropy coding and, from there, mapped back to the data space for reconstruction. In contrast to VAE-based neural compression, where the (mean) decoder is a deterministic neural network, our decoder is a conditional diffusion model. Our approach thus introduces an additional "content" latent variable on which the reverse diffusion process is conditioned and uses this variable to store information about the image. The remaining ``texture'' variables characterizing the diffusion process are synthesized at decoding time. We show that the model's performance can be tuned toward perceptual metrics of interest. Our extensive experiments involving multiple datasets and image quality assessment metrics show that our approach yields stronger reported FID scores than the GAN-based model, while also yielding competitive performance with VAE-based models in several distortion metrics. Furthermore, training the diffusion with  $\mathcal{X}$-parameterization enables high-quality reconstructions in only a handful of decoding steps, greatly affecting the model's practicality. Our code is available at: https://github.com/buggyyang/CDC_compression

----

## [0] Leveraging the two-timescale regime to demonstrate convergence of neural networks

**Authors**: *Pierre Marion, Raphaël Berthier*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd062f8003e38f55dcb93df55b2683d6-Abstract-Conference.html)

**Abstract**:

We study the training dynamics of shallow neural networks, in a two-timescale regime in which the stepsizes for the inner layer are much smaller than those for the outer layer. In this regime, we prove convergence of the gradient flow to a global optimum of the non-convex optimization problem in a simple univariate setting. The number of neurons need not be asymptotically large for our result to hold, distinguishing our result from popular recent approaches such as the neural tangent kernel or mean-field regimes. Experimental illustration is provided, showing that the stochastic gradient descent behaves according to our description of the gradient flow and thus converges to a global optimum in the two-timescale regime, but can fail outside of this regime.

----

## [0] Grammar Prompting for Domain-Specific Language Generation with Large Language Models

**Authors**: *Bailin Wang, Zi Wang, Xuezhi Wang, Yuan Cao, Rif A. Saurous, Yoon Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd40d0d65bfebb894ccc9ea822b47fa8-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs)  can learn to perform a wide range of natural language tasks from just a  handful of in-context examples. However, for generating strings from highly structured  languages (e.g., semantic parsing to complex domain-specific languages), it is challenging for the LLM to generalize from just a few exemplars. We propose \emph{grammar prompting}, a simple approach to enable LLMs to use external knowledge and domain-specific constraints, expressed through a grammar in Backus--Naur Form (BNF), during in-context learning. Grammar prompting augments each demonstration example with a specialized grammar that is minimally sufficient for generating the particular output example, where the specialized grammar is a subset of the full DSL grammar.  For inference, the LLM first predicts a BNF grammar given a test input, and then generates the output according to the rules of the grammar. Experiments demonstrate that grammar prompting can enable LLMs to perform competitively on a diverse set of DSL generation tasks, including semantic parsing (SMCalFlow, Overnight, GeoQuery), PDDL planning, and SMILES-based molecule generation.

----

## [0] Don't just prune by magnitude! Your mask topology is a secret weapon

**Authors**: *Duc Hoang, Souvik Kundu, Shiwei Liu, Zhangyang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd5404354496e39d37b7947d8a0d7b72-Abstract-Conference.html)

**Abstract**:

Recent years have witnessed significant progress in understanding the relationship between the connectivity of a deep network's architecture as a graph, and the network's performance. A few prior arts connected deep architectures to expander graphs or Ramanujan graphs, and particularly,[7] demonstrated the use of such graph connectivity measures with ranking and relative performance of various obtained sparse sub-networks  (i.e. models with prune masks) without the need for training. However, no prior work explicitly explores the role of parameters in the graph's connectivity, making the graph-based understanding of prune masks and the magnitude/gradient-based pruning practice isolated from one another. This paper strives to fill in this gap, by analyzing the Weighted Spectral Gap of Ramanujan structures in sparse neural networks and investigates its correlation with final performance. We specifically examine the evolution of sparse structures under a popular dynamic sparse-to-sparse network training scheme, and intriguingly find that the generated random topologies inherently maximize Ramanujan graphs. We also identify a strong correlation between masks, performance, and the weighted spectral gap. Leveraging this observation, we propose to construct a new "full-spectrum coordinate'' aiming to comprehensively characterize a sparse neural network's promise. Concretely, it consists of the classical Ramanujan's gap (structure), our proposed weighted spectral gap (parameters), and the constituent nested regular graphs within. In this new coordinate system, a sparse subnetwork's L2-distance from its original initialization is found to have nearly linear correlated with its performance. Eventually, we apply this unified perspective to develop a new actionable pruning method, by sampling sparse masks to maximize the L2-coordinate distance. Our method can be augmented with the "pruning at initialization" (PaI) method, and significantly outperforms existing PaI methods. With only a few iterations of training (e.g 500 iterations), we can get LTH-comparable performance as that yielded via "pruning after training", significantly saving pre-training costs. Codes can be found at:  https://github.com/VITA-Group/FullSpectrum-PAI.

----

## [0] Learning Human Action Recognition Representations Without Real Humans

**Authors**: *Howard Zhong, Samarth Mishra, Donghyun Kim, SouYoung Jin, Rameswar Panda, Hilde Kuehne, Leonid Karlinsky, Venkatesh Saligrama, Aude Oliva, Rogério Feris*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd556f38dba3a6c367c42fa85fc0801c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd556f38dba3a6c367c42fa85fc0801c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Pre-training on massive video datasets has become essential to achieve high action recognition performance on smaller downstream datasets. However, most large-scale video datasets contain images of people and hence are accompanied with issues related to privacy, ethics, and data protection, often preventing them from being publicly shared for reproducible research. Existing work has attempted to alleviate these problems by blurring faces, downsampling videos, or training on synthetic data. On the other hand, analysis on the {\em transferability} of privacy-preserving pre-trained models to downstream tasks has been limited. In this work, we study this problem by first asking the question: can we pre-train models for human action recognition with data that does not include real humans? To this end, we present, for the first time, a benchmark that leverages real-world videos with {\em humans removed} and synthetic data containing virtual humans to pre-train a model. We then evaluate the transferability of the representation learned on this data to a diverse set of downstream action recognition benchmarks. Furthermore, we propose a novel pre-training strategy, called Privacy-Preserving MAE-Align, to effectively combine synthetic data and human-removed real data. Our approach outperforms previous baselines by up to 5\% and closes the performance gap between human and no-human action recognition representations on downstream tasks, for both linear probing and fine-tuning. Our benchmark, code, and models are available at https://github.com/howardzh01/PPMA.

----

## [0] Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation

**Authors**: *Yingyi Chen, Qinghua Tao, Francesco Tonin, Johan A. K. Suykens*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd687a58a13b673eea3fc1b2e4944cf7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd687a58a13b673eea3fc1b2e4944cf7-Abstract-Conference.html)

**Abstract**:

Recently, a new line of works has emerged to understand and improve self-attention in Transformers by treating it as a kernel machine. However, existing works apply the methods for symmetric kernels to the asymmetric self-attention, resulting in a nontrivial gap between the analytical understanding and numerical implementation. In this paper, we provide a new perspective to represent and optimize self-attention through asymmetric Kernel Singular Value Decomposition (KSVD), which is also motivated by the low-rank property of self-attention normally observed in deep layers. Through asymmetric KSVD, i) a primal-dual representation of self-attention is formulated, where the optimization objective is cast to maximize the projection variances in the attention outputs; ii) a novel attention mechanism, i.e., Primal-Attention, is proposed via the primal representation of KSVD, avoiding explicit computation of the kernel matrix in the dual; iii) with KKT conditions, we prove that the stationary solution to the KSVD optimization in Primal-Attention yields a zero-value objective. In this manner, KSVD optimization can be implemented by simply minimizing a regularization loss, so that low-rank property is promoted without extra decomposition. Numerical experiments show state-of-the-art performance of our Primal-Attention with improved efficiency. Moreover, we demonstrate that the deployed KSVD optimization regularizes Primal-Attention with a sharper singular value decay than that of the canonical self-attention, further verifying the great potential of our method. To the best of our knowledge, this is the first work that provides a primal-dual representation for the asymmetric kernel in self-attention and successfully applies it to modelling and optimization.

----

## [0] End-To-End Latent Variational Diffusion Models for Inverse Problems in High Energy Physics

**Authors**: *Alexander Shmakov, Kevin Greif, Michael James Fenton, Aishik Ghosh, Pierre Baldi, Daniel Whiteson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd830afc6208a346e4ec5caf1b08b4b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd830afc6208a346e4ec5caf1b08b4b4-Abstract-Conference.html)

**Abstract**:

High-energy collisions at the Large Hadron Collider (LHC) provide valuable insights into open questions in particle physics. However, detector effects must be corrected before measurements can be compared to certain theoretical predictions or measurements from other detectors. Methods to solve this inverse problem of mapping detector observations to theoretical quantities of the underlying collision are essential parts of many physics analyses at the LHC. We investigate and compare various generative deep learning methods to approximate this inverse mapping. We introduce a novel unified architecture, termed latent variational diffusion models, which combines the latent learning of cutting-edge generative art approaches with an end-to-end variational framework. We demonstrate the effectiveness of this approach for reconstructing global distributions of theoretical kinematic quantities, as well as for ensuring the adherence of the learned posterior distributions to known physics constraints. Our unified approach achieves a distribution-free distance to the truth of over 20 times smaller than non-latent state-of-the-art baseline and 3 times smaller than traditional latent diffusion models.

----

## [0] AVeriTeC: A Dataset for Real-world Claim Verification with Evidence from the Web

**Authors**: *Michael Schlichtkrull, Zhijiang Guo, Andreas Vlachos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd86a30526cd1aff61d6f89f107634e4-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Existing datasets for automated fact-checking have substantial limitations, such as relying on artificial claims, lacking annotations for evidence and intermediate reasoning, or including evidence published after the claim. In this paper we introduce AVeriTeC, a new dataset of 4,568 real-world claims covering fact-checks by 50 different organizations. Each claim is annotated with question-answer pairs supported by evidence available online, as well as textual justifications explaining how the evidence combines to produce a verdict. Through a multi-round annotation process, we avoid common pitfalls including context dependence, evidence insufficiency, and temporal leakage, and reach a substantial inter-annotator agreement of $\kappa=0.619$ on verdicts. We develop a baseline as well as an evaluation scheme for verifying claims through question-answering against the open web.

----

## [0] DiffTraj: Generating GPS Trajectory with Diffusion Probabilistic Model

**Authors**: *Yuanshao Zhu, Yongchao Ye, Shiyao Zhang, Xiangyu Zhao, James Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cd9b4a28fb9eebe0430c3312a4898a41-Abstract-Conference.html)

**Abstract**:

Pervasive integration of GPS-enabled devices and data acquisition technologies has led to an exponential increase in GPS trajectory data, fostering advancements in spatial-temporal data mining research. Nonetheless, GPS trajectories contain personal geolocation information, rendering serious privacy concerns when working with raw data. A promising approach to address this issue is trajectory generation, which involves replacing original data with generated, privacy-free alternatives. Despite the potential of trajectory generation, the complex nature of human behavior and its inherent stochastic characteristics pose challenges in generating high-quality trajectories. In this work, we propose a spatial-temporal diffusion probabilistic model for trajectory generation (DiffTraj). This model effectively combines the generative abilities of diffusion models with the spatial-temporal features derived from real trajectories. The core idea is to reconstruct and synthesize geographic trajectories from white noise through a reverse trajectory denoising process. Furthermore, we propose a Trajectory UNet (Traj-UNet) deep neural network to embed conditional information and accurately estimate noise levels during the reverse process. Experiments on two real-world datasets show that DiffTraj can be intuitively applied to generate high-fidelity trajectories while retaining the original distributions. Moreover, the generated results can support downstream trajectory analysis tasks and significantly outperform other methods in terms of geo-distribution evaluations.

----

## [0] Meta-in-context learning in large language models

**Authors**: *Julian Coda-Forno, Marcel Binz, Zeynep Akata, Matt M. Botvinick, Jane X. Wang, Eric Schulz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cda04d7ea67ea1376bf8c6962d8541e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cda04d7ea67ea1376bf8c6962d8541e0-Abstract-Conference.html)

**Abstract**:

Large language models have shown tremendous performance in a variety of tasks.  In-context learning  -- the ability to improve at a task after being provided with a number of demonstrations -- is seen as one of the main contributors to their success. In the present paper, we demonstrate that the in-context learning abilities of large language models can be recursively improved via in-context learning itself. We coin this phenomenon meta-in-context learning. Looking at two idealized domains, a one-dimensional regression task and a two-armed bandit task, we show that meta-in-context learning adaptively reshapes a large language model's priors over expected tasks. Furthermore, we find that meta-in-context learning modifies the in-context learning strategies of such models. Finally, we broaden the scope of our investigation to encompass two diverse benchmarks: one focusing on real-world regression problems and the other encompassing multiple NLP tasks. In both cases, we observe competitive performance comparable to that of traditional learning algorithms. Taken together, our work improves our understanding of in-context learning and paves the way toward adapting large language models to the environment they are applied purely through meta-in-context learning rather than traditional finetuning.

----

## [0] Dynamic Context Pruning for Efficient and Interpretable Autoregressive Transformers

**Authors**: *Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurélien Lucchi, Thomas Hofmann*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdaac2a02c4fdcae77ba083b110efcc3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdaac2a02c4fdcae77ba083b110efcc3-Abstract-Conference.html)

**Abstract**:

Autoregressive Transformers adopted in Large Language Models (LLMs) are hard to scale to long sequences. Despite several works trying to reduce their computational cost, most of LLMs still adopt attention layers between all pairs of tokens in the sequence, thus incurring a quadratic cost. In this study, we present a novel approach that dynamically prunes contextual information while preserving the model's expressiveness, resulting in reduced memory and computational requirements during inference. Our method employs a learnable mechanism that determines which uninformative tokens can be dropped from the context at any point across the generation process. By doing so, our approach not only addresses performance concerns but also enhances interpretability, providing valuable insight into the model's decision-making process. Our technique can be applied to existing pre-trained models through a straightforward fine-tuning process, and the pruning strength can be specified by a sparsity parameter. Notably, our empirical findings demonstrate that we can effectively prune up to 80\% of the context without significant performance degradation on downstream tasks, offering a valuable tool for mitigating inference costs. Our reference implementation achieves up to $2\times$ increase in inference throughput and even greater memory savings.

----

## [0] SPQR: Controlling Q-ensemble Independence with Spiked Random Model for Reinforcement Learning

**Authors**: *Dohyeok Lee, Seungyub Han, Taehyun Cho, Jungwoo Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdcaf772b4f8eda0385d0930517de64a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdcaf772b4f8eda0385d0930517de64a-Abstract-Conference.html)

**Abstract**:

Alleviating overestimation bias is a critical challenge for deep reinforcement learning to achieve successful performance on more complex tasks or offline datasets containing out-of-distribution data. In order to overcome overestimation bias, ensemble methods for Q-learning have been investigated to exploit the diversity of multiple Q-functions. Since network initialization has been the predominant approach to promote diversity in Q-functions, heuristically designed diversity injection methods have been studied in the literature. However, previous studies have not attempted to approach guaranteed independence over an ensemble from a theoretical perspective. By introducing a novel regularization loss for Q-ensemble independence based on random matrix theory, we propose spiked Wishart Q-ensemble independence regularization (SPQR) for reinforcement learning. Specifically, we modify the intractable hypothesis testing criterion for the Q-ensemble independence into a tractable KL divergence between the spectral distribution of the Q-ensemble and the target Wigner's semicircle distribution. We implement SPQR in several online and offline ensemble Q-learning algorithms. In the experiments, SPQR outperforms the baseline algorithms in both online and offline RL benchmarks.

----

## [0] SwapPrompt: Test-Time Prompt Adaptation for Vision-Language Models

**Authors**: *Xiaosong Ma, Jie Zhang, Song Guo, Wenchao Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html)

**Abstract**:

Test-time adaptation (TTA) is a special and practical setting in unsupervised domain adaptation, which allows a pre-trained model in a source domain to adapt to unlabeled test data in another target domain. To avoid the computation-intensive backbone fine-tuning process, the zero-shot generalization potentials of the emerging pre-trained vision-language models (e.g., CLIP, CoOp) are leveraged to only tune the run-time prompt for unseen test domains. However, existing solutions have yet to fully exploit the representation capabilities of pre-trained models as they only focus on the entropy-based optimization and the performance is far below the supervised prompt adaptation methods, e.g., CoOp. In this paper, we propose SwapPrompt, a novel framework that can effectively leverage the self-supervised contrastive learning to facilitate the test-time prompt adaptation. SwapPrompt employs a dual prompts paradigm, i.e., an online prompt and a target prompt that averaged from the online prompt to retain historical information. In addition, SwapPrompt applies a swapped prediction mechanism, which takes advantage of the representation capabilities of pre-trained models to enhance the online prompt via contrastive learning. Specifically, we use the online prompt together with an augmented view of the input image to predict the class assignment generated by the target prompt together with an alternative augmented view of the same image. The proposed SwapPrompt can be easily deployed on vision-language models without additional requirement, and experimental results show that it achieves state-of-the-art test-time adaptation performance on ImageNet and nine other datasets. It is also shown that SwapPrompt can even achieve comparable performance with supervised prompt adaptation methods.

----

## [0] Does a sparse ReLU network training problem always admit an optimum ?

**Authors**: *Quoc-Tung Le, Rémi Gribonval, Elisa Riccietti*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdda0657a9f32bc7ddd4343686e7371e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdda0657a9f32bc7ddd4343686e7371e-Abstract-Conference.html)

**Abstract**:

Given a training set, a loss function, and a neural network architecture, it is often taken for granted that optimal network parameters exist, and a common practice is to apply available optimization algorithms to search for them. In this work, we show that the existence of an optimal solution is not always guaranteed, especially in the context of sparse ReLU neural networks.In particular, we first show that optimization problems involving deep networks with certain sparsity patterns do not always have optimal parameters, and that optimization algorithms may then diverge. Via a new topological relation between sparse ReLU neural networks and their linear counterparts, we derive --using existing tools from real algebraic geometry-- an algorithm to verify that a given sparsity pattern suffers from this issue. Then, the existence of a global optimum is proved for every concrete optimization problem involving a shallow sparse ReLU neural network of output dimension one. Overall, the analysis is based on the investigation of two topological properties of the space of functions implementable as sparse ReLU neural networks: a best approximation property, and a closedness property, both in the uniform norm. This is studied both for (finite) domains corresponding to practical training on finite training sets, and for more general domains such as the unit cube. This allows us to provide conditions for the guaranteed existence of an optimum given a sparsity pattern. The results apply not only to several sparsity patterns proposed in recent works on network pruning/sparsification, but also to classical dense neural networks, including architectures not covered by existing results.

----

## [0] Knowledge Diffusion for Distillation

**Authors**: *Tao Huang, Yuan Zhang, Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Chang Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdddf13f06182063c4dbde8cbd5a5c21-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdddf13f06182063c4dbde8cbd5a5c21-Abstract-Conference.html)

**Abstract**:

The representation gap between teacher and student is an emerging topic in knowledge distillation (KD). To reduce the gap and improve the performance, current methods often resort to complicated training schemes, loss functions, and feature alignments, which are task-specific and feature-specific. In this paper, we state that the essence of these methods is to discard the noisy information and distill the valuable information in the feature, and propose a novel KD method dubbed DiffKD, to explicitly denoise and match features using diffusion models. Our approach is based on the observation that student features typically contain more noises than teacher features due to the smaller capacity of student model. To address this, we propose to denoise student features using a diffusion model trained by teacher features. This allows us to perform better distillation between the refined clean feature and teacher feature. Additionally, we introduce a light-weight diffusion model with a linear autoencoder to reduce the computation cost and an adaptive noise matching module to improve the denoising performance. Extensive experiments demonstrate that DiffKD is effective across various types of features and achieves state-of-the-art performance consistently on image classification, object detection, and semantic segmentation tasks. Code is available at https://github.com/hunto/DiffKD.

----

## [0] BayesTune: Bayesian Sparse Deep Model Fine-tuning

**Authors**: *Minyoung Kim, Timothy M. Hospedales*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cde2dc73e0ad650176cdfa9b779eefc7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cde2dc73e0ad650176cdfa9b779eefc7-Abstract-Conference.html)

**Abstract**:

Deep learning practice is increasingly driven by powerful foundation models (FM), pre-trained at scale and then fine-tuned for specific tasks of interest. A key property of this workflow is the efficacy of performing sparse or parameter-efficient fine-tuning, meaning that by updating only a tiny fraction of the whole FM parameters on a downstream task can lead to surprisingly good performance, often even superior to a full model update. However, it is not clear what is the optimal and principled way to select which parameters to update. Although a growing number of sparse fine-tuning ideas have been proposed, they are mostly not  satisfactory, relying on hand-crafted heuristics or heavy approximation. In this paper we propose a novel Bayesian sparse fine-tuning algorithm: we place a (sparse) Laplace prior for each parameter of the FM, with the mean equal to the initial value and the scale parameter having a hyper-prior that encourages small scale. Roughly speaking, the posterior means of the scale parameters indicate how important it is to update the corresponding parameter away from its initial value when solving the downstream task. Given the sparse prior, most scale parameters are small a posteriori, and the few large-valued scale parameters identify those FM parameters that crucially need to be updated away from their initial values. Based on this, we can threshold the scale parameters to decide which parameters to update or freeze, leading to a principled sparse fine-tuning strategy. To efficiently infer the posterior distribution of the scale parameters, we adopt the Langevin MCMC sampler, requiring only two times the complexity of the vanilla SGD. Tested on popular NLP benchmarks as well as the VTAB vision tasks, our approach shows significant improvement over the state-of-the-arts (e.g., 1% point higher than the best SOTA when fine-tuning RoBERTa for GLUE and SuperGLUE benchmarks).

----

## [0] Exploring Loss Functions for Time-based Training Strategy in Spiking Neural Networks

**Authors**: *Yaoyu Zhu, Wei Fang, Xiaodong Xie, Tiejun Huang, Zhaofei Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cde874a797a8300da693d5e412b7fdc0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cde874a797a8300da693d5e412b7fdc0-Abstract-Conference.html)

**Abstract**:

Spiking Neural Networks (SNNs) are considered promising brain-inspired energy-efficient models due to their event-driven computing paradigm.The spatiotemporal spike patterns used to convey information in SNNs consist of both rate coding and temporal coding, where the temporal coding is crucial to biological-plausible learning rules such as spike-timing-dependent-plasticity.The time-based training strategy is proposed to better utilize the temporal information in SNNs and learn in an asynchronous fashion.However, some recent works train SNNs by the time-based scheme with rate-coding-dominated loss functions.In this paper, we first map rate-based loss functions to time-based counterparts and explain why they are also applicable to the time-based training scheme.After that, we infer that loss functions providing adequate positive overall gradients help training by theoretical analysis.Based on this, we propose the enhanced counting loss to replace the commonly used mean square counting loss.In addition, we transfer the training of scale factor in weight standardization into thresholds.Experiments show that our approach outperforms previous time-based training methods in most datasets. Our work provides insights for training SNNs with time-based schemes and offers a fresh perspective on the correlation between rate coding and temporal coding.Our code is available at https://github.com/zhuyaoyu/SNN-temporal-training-losses.

----

## [0] Learning Rate Free Bayesian Inference in Constrained Domains

**Authors**: *Louis Sharrock, Lester Mackey, Christopher Nemeth*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cdee6c3eaa2adc285f11da7711a75c12-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cdee6c3eaa2adc285f11da7711a75c12-Abstract-Conference.html)

**Abstract**:

We introduce a suite of new particle-based algorithms for sampling in constrained domains which are entirely learning rate free. Our approach leverages coin betting ideas from convex optimisation, and the viewpoint of constrained sampling as a mirrored optimisation problem on the space of probability measures. Based on this viewpoint, we also introduce a unifying framework for several existing constrained sampling algorithms, including mirrored Langevin dynamics and mirrored Stein variational gradient descent. We demonstrate the performance of our algorithms on a range of numerical examples, including sampling from targets on the simplex, sampling with fairness constraints, and constrained sampling problems in post-selection inference. Our results indicate that our algorithms achieve competitive performance with existing constrained sampling methods, without the need to tune any hyperparameters.

----

## [0] Volume Feature Rendering for Fast Neural Radiance Field Reconstruction

**Authors**: *Kang Han, Wei Xiang, Lu Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce182e31662883d4decc84a0255335b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce182e31662883d4decc84a0255335b6-Abstract-Conference.html)

**Abstract**:

Neural radiance fields (NeRFs) are able to synthesize realistic novel views from multi-view images captured from distinct positions and perspectives. In NeRF's rendering pipeline, neural networks are used to represent a scene independently or transform queried learnable feature vector of a point to the expected color or density. With the aid of geometry guides either in the form of occupancy grids or proposal networks, the number of color neural network evaluations can be reduced from hundreds to dozens in the standard volume rendering framework. However, many evaluations of the color neural network are still a bottleneck for fast NeRF reconstruction. This paper revisits volume feature rendering (VFR) for the purpose of fast NeRF reconstruction. The VFR integrates the queried feature vectors of a ray into one feature vector, which is then transformed to the final pixel color by a color neural network. This fundamental change to the standard volume rendering framework requires only one single color neural network evaluation to render a pixel, which substantially lowers the high computational complexity of the rendering framework attributed to a large number of color neural network evaluations. Consequently, we can use a comparably larger color neural network to achieve a better rendering quality while maintaining the same training and rendering time costs. This approach achieves the state-of-the-art rendering quality on both synthetic and real-world datasets while requiring less training time compared with existing methods.

----

## [0] Offline RL with Discrete Proxy Representations for Generalizability in POMDPs

**Authors**: *Pengjie Gu, Xinyu Cai, Dong Xing, Xinrun Wang, Mengchen Zhao, Bo An*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce1c1ff5d94079dea348a2317a889281-Abstract-Conference.html)

**Abstract**:

Offline Reinforcement Learning (RL) has demonstrated promising results in various applications by learning policies from previously collected datasets, reducing the need for online exploration and interactions. However, real-world scenarios usually involve partial observability, which brings crucial challenges of the deployment of offline RL methods: i) the policy trained on data with full observability is not robust against the masked observations during execution, and ii) the information of which parts of observations are masked is usually unknown during training. In order to address these challenges, we present Offline RL with DiscrEte pRoxy representations (ORDER), a probabilistic framework which leverages novel state representations to improve the robustness against diverse masked observabilities. Specifically, we propose a discrete representation of the states and use a proxy representation to recover the states from masked partial observable trajectories. The training of ORDER can be compactly described as the following three steps. i) Learning the discrete state representations on data with full observations, ii) Training the decision module based on the discrete representations, and iii) Training the proxy discrete representations on the data with various partial observations, aligning with the discrete representations. We conduct extensive experiments to evaluate ORDER, showcasing its effectiveness in offline RL for diverse partially observable scenarios and highlighting the significance of discrete proxy representations in  generalization performance.ORDER is a flexible framework to employ any offline RL algorithms and we hope that ORDER can pave the way for the deployment of RL policy against various partial observabilities in the real world.

----

## [0] Meta-AdaM: An Meta-Learned Adaptive Optimizer with Momentum for Few-Shot Learning

**Authors**: *Siyuan Sun, Hongyang Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce26d21662c979d515164b416d4571fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce26d21662c979d515164b416d4571fe-Abstract-Conference.html)

**Abstract**:

We introduce Meta-AdaM, a meta-learned adaptive optimizer with momentum, designed for few-shot learning tasks that pose significant challenges to deep learning models due to the limited number of labeled examples. Meta-learning has been successfully employed to address these challenges by transferring meta-learned prior knowledge to new tasks. Most existing works focus on meta-learning an optimal model initialization or an adaptive learning rate learner for rapid convergence. However, these approaches either neglect to consider weight-update history for the adaptive learning rate learner or fail to effectively integrate momentum for fast convergence, as seen in many-shot learning settings. To tackle these limitations, we propose a meta-learned learning rate learner that utilizes weight-update history as input to predict more appropriate learning rates for rapid convergence. Furthermore, for the first time, our approach incorporates momentum into the optimization process of few-shot learning via a double look-ahead mechanism, enabling rapid convergence similar to many-shot settings. Extensive experimental results on benchmark datasets demonstrate the effectiveness of the proposed Meta-AdaM.

----

## [0] Fairness Continual Learning Approach to Semantic Scene Understanding in Open-World Environments

**Authors**: *Thanh-Dat Truong, Hoang-Quan Nguyen, Bhiksha Raj, Khoa Luu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce3cf998b7f59271e80ce03fb74a7115-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce3cf998b7f59271e80ce03fb74a7115-Abstract-Conference.html)

**Abstract**:

Continual semantic segmentation aims to learn new classes while maintaining the information from the previous classes. Although prior studies have shown impressive progress in recent years, the fairness concern in the continual semantic segmentation needs to be better addressed. Meanwhile, fairness is one of the most vital factors in deploying the deep learning model, especially in human-related or safety applications. In this paper,  we present a novel Fairness Continual Learning approach to the semantic segmentation problem.In particular, under the fairness objective, a new fairness continual learning framework is proposed based on class distributions.Then, a novel Prototypical Contrastive Clustering loss is proposed to address the significant challenges in continual learning, i.e., catastrophic forgetting and background shift. Our proposed loss has also been proven as a novel, generalized learning paradigm of knowledge distillation commonly used in continual learning. Moreover, the proposed Conditional Structural Consistency loss further regularized the structural constraint of the predicted segmentation. Our proposed approach has achieved State-of-the-Art performance on three standard scene understanding benchmarks, i.e., ADE20K, Cityscapes, and Pascal VOC, and promoted the fairness of the segmentation model.

----

## [0] Post Hoc Explanations of Language Models Can Improve Language Models

**Authors**: *Satyapriya Krishna, Jiaqi Ma, Dylan Slack, Asma Ghandeharioun, Sameer Singh, Himabindu Lakkaraju*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce65173b994cf7c925c71b482ee14a8d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce65173b994cf7c925c71b482ee14a8d-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) have demonstrated remarkable capabilities in performing complex tasks. Moreover, recent research has shown that incorporating human-annotated rationales (e.g., Chain-of-Thought prompting) during in-context learning can significantly enhance the performance of these models, particularly on tasks that require reasoning capabilities. However, incorporating such rationales poses challenges in terms of scalability as this requires a high degree of human involvement. In this work, we present a novel framework, Amplifying Model Performance by Leveraging In-Context Learning with Post Hoc Explanations (AMPLIFY), which addresses the aforementioned challenges by automating the process of rationale generation. To this end, we leverage post hoc explanation methods which output attribution scores (explanations) capturing the influence of each of the input features on model predictions. More specifically, we construct automated natural language rationales that embed insights from post hoc explanations to provide corrective signals to LLMs. Extensive experimentation with real-world datasets demonstrates that our framework, AMPLIFY, leads to prediction accuracy improvements of about 10-25% over a wide range of tasks, including those where prior approaches which rely on human-annotated rationales such as Chain-of-Thought prompting fall short. Our work makes one of the first attempts at highlighting the potential of post hoc explanations as valuable tools for enhancing the effectiveness of LLMs. Furthermore, we conduct additional empirical analyses and ablation studies to demonstrate the impact of each of the components of AMPLIFY, which, in turn, lead to critical insights for refining in context learning.

----

## [0] Understanding Diffusion Objectives as the ELBO with Simple Data Augmentation

**Authors**: *Diederik P. Kingma, Ruiqi Gao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce79fbf9baef726645bc2337abb0ade2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce79fbf9baef726645bc2337abb0ade2-Abstract-Conference.html)

**Abstract**:

To achieve the highest perceptual quality, state-of-the-art diffusion models are optimized with objectives that typically look very different from the maximum likelihood and the Evidence Lower Bound (ELBO) objectives. In this work, we reveal that diffusion model objectives are actually closely related to the ELBO.Specifically, we show that all commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Under the condition of monotonic weighting, the connection is even closer: the diffusion objective then equals the ELBO, combined with simple data augmentation, namely Gaussian noise perturbation. We show that this condition holds for a number of state-of-the-art diffusion models. In experiments, we explore new monotonic weightings and demonstrate their effectiveness, achieving state-of-the-art FID scores on the high-resolution ImageNet benchmark.

----

## [0] Response Length Perception and Sequence Scheduling: An LLM-Empowered LLM Inference Pipeline

**Authors**: *Zangwei Zheng, Xiaozhe Ren, Fuzhao Xue, Yang Luo, Xin Jiang, Yang You*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce7ff3405c782f761fac7f849b41ae9a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce7ff3405c782f761fac7f849b41ae9a-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) have revolutionized the field of AI, demonstrating unprecedented capacity across various tasks. However, the inference process for LLMs comes with significant computational costs. In this paper, we propose an efficient LLM inference pipeline that harnesses the power of LLMs. Our approach begins by tapping into the potential of LLMs to accurately perceive and predict the response length with minimal overhead. By leveraging this information, we introduce an efficient sequence scheduling technique that groups queries with similar response lengths into micro-batches. We evaluate our approach on real-world instruction datasets using the LLaMA-based model, and our results demonstrate an impressive 86% improvement in inference throughput without compromising effectiveness. Notably, our method is orthogonal to other inference acceleration techniques, making it a valuable addition to many existing toolkits (e.g., FlashAttention, Quantization) for LLM inference.

----

## [0] When Demonstrations meet Generative World Models: A Maximum Likelihood Framework for Offline Inverse Reinforcement Learning

**Authors**: *Siliang Zeng, Chenliang Li, Alfredo Garcia, Mingyi Hong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ce9d3c592712d23f2ec3671941d67fa1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ce9d3c592712d23f2ec3671941d67fa1-Abstract-Conference.html)

**Abstract**:

Offline inverse reinforcement learning (Offline IRL) aims to recover the structure of rewards and environment dynamics that underlie observed actions in a fixed, finite set of demonstrations from an expert agent. Accurate models of expertise in executing a task has applications in safety-sensitive applications such as clinical decision making and autonomous driving. However, the structure of an expert's preferences implicit in observed actions is closely linked to the expert's model of the environment dynamics (i.e. the ``world''). Thus, inaccurate models of the world obtained from finite data with limited coverage could compound inaccuracy in estimated rewards. To address this issue, we propose a bi-level optimization formulation of the estimation task wherein the upper level is likelihood maximization based upon a conservative model of the expert's policy (lower level). The policy model is conservative in that it maximizes reward subject to a penalty that is increasing in the uncertainty of the estimated model of the world. We propose a new algorithmic framework to solve the bi-level optimization problem formulation and provide statistical and computational guarantees of performance for the associated optimal reward estimator. Finally,  we demonstrate that the proposed algorithm outperforms the state-of-the-art offline IRL and imitation learning benchmarks by a large margin, over the continuous control tasks in MuJoCo and different datasets in the D4RL benchmark.

----

## [0] Neural Priming for Sample-Efficient Adaptation

**Authors**: *Matthew Wallingford, Vivek Ramanujan, Alex Fang, Aditya Kusupati, Roozbeh Mottaghi, Aniruddha Kembhavi, Ludwig Schmidt, Ali Farhadi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cea5bc68b890bffb10f18aaaab2becb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cea5bc68b890bffb10f18aaaab2becb1-Abstract-Conference.html)

**Abstract**:

We propose Neural Priming, a technique for adapting large pretrained models to distribution shifts and downstream tasks given few or no labeled examples. Presented with class names or unlabeled test samples, Neural Priming enables the model to recall and conditions its parameters on relevant data seen throughout pretraining, thereby priming it for the test distribution. Neural Priming can be performed at test time in even for pretraining datasets as large as LAION-2B. Performing lightweight updates on the recalled data significantly improves accuracy across a variety of distribution shift and transfer learning benchmarks. Concretely, in the zero-shot setting, we see a 2.45% improvement in accuracy on ImageNet and 3.81% accuracy improvement on average across standard transfer learning benchmarks. Further, using our test time inference scheme, we see a 1.41% accuracy improvement on ImageNetV2. These results demonstrate the effectiveness of Neural Priming in addressing the common challenge of limited labeled data and changing distributions. Code and models are open-sourced at https://www.github.com/RAIVNLab/neural-priming.

----

## [0] Derandomized novelty detection with FDR control via conformal e-values

**Authors**: *Meshi Bashari, Amir Epstein, Yaniv Romano, Matteo Sesia*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cec8ad7715d0d13899d5d7d31970f527-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cec8ad7715d0d13899d5d7d31970f527-Abstract-Conference.html)

**Abstract**:

Conformal inference provides a general distribution-free method to rigorously calibrate the output of any machine learning algorithm for novelty detection. While this approach has many strengths, it has the limitation of being randomized, in the sense that it may lead to different results when analyzing twice the same data and this can hinder the interpretation of any findings. We propose to make conformal inferences more stable by leveraging suitable conformal e-values instead of p-values to quantify statistical significance. This solution allows the evidence gathered from multiple analyses of the same data to be aggregated effectively while provably controlling the false discovery rate. Further, we show that the proposed method can reduce randomness without much loss of power compared to standard conformal inference, partly thanks to an innovative way of weighting conformal e-values based on additional side information carefully extracted from the same data. Simulations with synthetic and real data confirm this solution can be effective at eliminating random noise in the inferences obtained with state-of-the-art alternative techniques, sometimes also leading to higher power.

----

## [0] ZipLM: Inference-Aware Structured Pruning of Language Models

**Authors**: *Eldar Kurtic, Elias Frantar, Dan Alistarh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/ced46a50befedcb884ccf0cbe8c3ad23-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/ced46a50befedcb884ccf0cbe8c3ad23-Abstract-Conference.html)

**Abstract**:

The breakthrough performance of large language models (LLMs) comes with major computational footprints and high deployment costs. In this paper, we progress towards resolving this problem by proposing a novel structured compression approach for LLMs, called ZipLM. ZipLM achieves state-of-the-art accuracy-vs-speedup, while matching a set of desired target runtime speedups in any given inference environment. Specifically, given a model, a dataset, an inference environment, as well as a set of speedup targets, ZipLM iteratively identifies and removes components with the worst loss-runtime trade-off. Unlike prior methods that specialize in either the post-training/one-shot or the gradual compression setting, and only for specific families of models such as BERT (encoder) or GPT (decoder), ZipLM produces state-of-the-art compressed models across all these settings. Furthermore, ZipLM achieves superior results for a fraction of the computational cost relative to prior distillation and pruning techniques, making it a cost-effective approach for generating an entire family of smaller, faster, and highly accurate models, guaranteed to meet the desired inference specifications. In particular, ZipLM outperforms all prior BERT-base distillation and pruning techniques, such as CoFi, MiniLM, and TinyBERT. Moreover, it matches the performance of the heavily optimized MobileBERT model, obtained via extensive architecture search, by simply pruning the baseline BERT-large model. When compressing GPT2, ZipLM outperforms DistilGPT2 while being 60\% smaller and 30\% faster. Our code is available at: https://github.com/IST-DASLab/ZipLM.

----

## [0] Private (Stochastic) Non-Convex Optimization Revisited: Second-Order Stationary Points and Excess Risks

**Authors**: *Daogao Liu, Arun Ganesh, Sewoong Oh, Abhradeep Guha Thakurta*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cede701f00079e43d053ac57b1e75c3e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cede701f00079e43d053ac57b1e75c3e-Abstract-Conference.html)

**Abstract**:

We reconsider the challenge of non-convex optimization under differential privacy constraint. Building upon the previous variance-reduced algorithm SpiderBoost, we propose a novel framework that employs two types of gradient oracles: one that estimates the gradient at a single point and a more cost-effective option that calculates the gradient difference between two points. Our framework can ensure continuous accuracy of gradient estimations and subsequently enhances the rates of identifying second-order stationary points.Additionally, we consider a more challenging task by attempting to locate the global minima of a non-convex objective via the exponential mechanism without almost any assumptions. Our preliminary results suggest that the regularized exponential mechanism can effectively emulate previous empirical and population risk bounds, negating the need for smoothness assumptions for algorithms with polynomial running time. Furthermore, with running time factors excluded, the exponential mechanism demonstrates promising population risk bound performance, and we provide a nearly matching lower bound.

----

## [0] Revealing the unseen: Benchmarking video action recognition under occlusion

**Authors**: *Shresth Grover, Vibhav Vineet, Yogesh S. Rawat*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cef53466b62aebbcf8aa2210a89b33a1-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/cef53466b62aebbcf8aa2210a89b33a1-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

In this work, we study the effect of occlusion on video action recognition. Tofacilitate this study, we propose three benchmark datasets and experiment withseven different video action recognition models. These datasets include two synthetic benchmarks, UCF-101-O and K-400-O, which enabled understanding the effects of fundamental properties of occlusion via controlled experiments. We also propose a real-world occlusion dataset, UCF-101-Y-OCC, which helps in further validating the findings of this study. We find several interesting insights such as 1) transformers are more robust than CNN counterparts, 2) pretraining make modelsrobust against occlusions, and 3) augmentation helps, but does not generalize well to real-world occlusions. In addition, we propose a simple transformer based compositional model, termed as CTx-Net, which generalizes well under this distribution shift. We observe that CTx-Net outperforms models which are trained using occlusions as augmentation, performing significantly better under natural occlusions. We believe this benchmark will open up interesting future research in robust video action recognition

----

## [0] TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models

**Authors**: *Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Bölöni, Qian Lou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf04d01a0e76f8b13095349d9caca033-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf04d01a0e76f8b13095349d9caca033-Abstract-Conference.html)

**Abstract**:

Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated.   Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks.  Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

----

## [0] Minimax Forward and Backward Learning of Evolving Tasks with Performance Guarantees

**Authors**: *Verónica Álvarez, Santiago Mazuelas, Jose A. Lozano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf4114c34a2b93019aa6e70f99680fae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf4114c34a2b93019aa6e70f99680fae-Abstract-Conference.html)

**Abstract**:

For a sequence of classification tasks that arrive over time, it is common that tasks are evolving in the sense that consecutive tasks often have a higher similarity. The incremental learning of a growing sequence of tasks holds promise to enable accurate classification even with few samples per task by leveraging information from all the tasks in the sequence (forward and backward learning). However, existing techniques developed for continual learning and concept drift adaptation are either designed for tasks with time-independent similarities or only aim to learn the last task in the sequence. This paper presents incremental minimax risk classifiers (IMRCs) that effectively exploit forward and backward learning and account for evolving tasks. In addition, we analytically characterize the performance improvement provided by forward and backward learning in terms of the tasksâ€™ expected quadratic change and the number of tasks. The experimental evaluation shows that IMRCs can result in a significant performance improvement, especially for reduced sample sizes.

----

## [0] Online Label Shift: Optimal Dynamic Regret meets Practical Algorithms

**Authors**: *Dheeraj Baby, Saurabh Garg, Tzu-Ching Yen, Sivaraman Balakrishnan, Zachary C. Lipton, Yu-Xiang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf42f133f355e0e07a8957b508b26a1b-Abstract-Conference.html)

**Abstract**:

This paper focuses on supervised and unsupervised online label shift,where the class marginals $Q(y)$ variesbut the class-conditionals $Q(x|y)$ remain invariant. In the unsupervised setting, our goal is to adapt a learner, trained on some offline labeled data, to changing label distributions given unlabeled online data. In the supervised setting, we must both learn a classifier and adapt to the dynamically evolving class marginals given only labeled online data. We develop novel algorithms that reduce the adaptation problem to online regression and guarantee optimal dynamic regret without any prior knowledge of the extent of drift in the label distribution. Our solution is based on bootstrapping the estimates of *online regression oracles* that track the drifting proportions. Experiments across numerous simulated and real-world online label shift scenarios demonstrate the superior performance of our proposed approaches, often achieving 1-3% improvement in accuracy while being sample and computationally efficient. Code is publicly available at https://github.com/Anon-djiwh/OnlineLabelShift

----

## [0] Self-supervised video pretraining yields robust and more human-aligned visual representations

**Authors**: *Nikhil Parthasarathy, S. M. Ali Eslami, João Carreira, Olivier J. Hénaff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf57022dff0929796f85ac99d7cefa86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf57022dff0929796f85ac99d7cefa86-Abstract-Conference.html)

**Abstract**:

Humans learn powerful representations of objects and scenes by observing how they evolve over time. Yet, outside of specific tasks that require explicit temporal understanding, static image pretraining remains the dominant paradigm for learning visual foundation models. We question this mismatch, and ask whether video pretraining can yield visual representations that bear the hallmarks of human perception: generalisation across tasks, robustness to perturbations, and consistency with human judgements. To that end we propose a novel procedure for curating videos, and develop a contrastive framework which learns from the complex transformations therein. This simple paradigm for distilling knowledge from videos, called VITO, yields general representations that far outperform prior video pretraining methods on image understanding tasks, and image pretraining methods on video understanding tasks. Moreover, VITO representations are significantly more robust to natural and synthetic deformations than image-, video-, and adversarially-trainedones. Finally, VITOâ€™s predictions are strongly aligned with human judgements, surpassing models that were specifically trained for that purpose. Together, these results suggest that video pretraining could be a simple way of learning unified, robust, and human-aligned representations of the visual world.

----

## [0] Slot-guided Volumetric Object Radiance Fields

**Authors**: *Di Qi, Tong Yang, Xiangyu Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf66f995883298c4db2f0dcba28fb211-Abstract-Conference.html)

**Abstract**:

We present a novel framework for 3D object-centric representation learning. Our approach effectively decomposes complex scenes into individual objects from a single image in an unsupervised fashion. This method, called \underline{s}lot-guided \underline{V}olumetric \underline{O}bject \underline{R}adiance \underline{F}ields~(sVORF), composes volumetric object radiance fields with object slots as a guidance to implement unsupervised 3D scene decomposition. Specifically, sVORF obtains object slots from a single image via a transformer module, maps these slots to volumetric object radiance fields with a hypernetwork and composes object radiance fields with the guidance of object slots at a 3D location. Moreover, sVORF significantly reduces memory requirement due to small-sized pixel rendering during training. We demonstrate the effectiveness of our approach by showing top results in scene decomposition and generation tasks of complex synthetic datasets (e.g., Room-Diverse). Furthermore, we also confirm the potential of sVORF to segment objects in real-world scenes (e.g., the LLFF dataset).  We hope our approach can provide preliminary understanding of the physical world and help ease future research in 3D object-centric representation learning.

----

## [0] Riemannian SAM: Sharpness-Aware Minimization on Riemannian Manifolds

**Authors**: *Jihun Yun, Eunho Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf701db0e3b4d0b8681ca6915ac3e87e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf701db0e3b4d0b8681ca6915ac3e87e-Abstract-Conference.html)

**Abstract**:

Contemporary advances in the field of deep learning have embarked upon an exploration of the underlying geometric properties of data, thus encouraging the investigation of techniques that consider general manifolds, for example, hyperbolic or orthogonal neural networks. However, the optimization algorithms for training such geometric deep learning models still remain highly under-explored. In this paper, we introduce Riemannian SAM by generalizing conventional Euclidean SAM to Riemannian manifolds. We successfully formulate the sharpness-aware minimization on Riemannian manifolds, leading to one of a novel instantiation, Lorentz SAM. In addition, SAM variants proposed in previous studies such as Fisher SAM can be derived as special examples under our Riemannian SAM framework. We provide the convergence analysis of Riemannian SAM under a less aggressively decaying ascent learning rate than Euclidean SAM. Our analysis serves as a theoretically sound contribution encompassing a diverse range of manifolds, also providing the guarantees for SAM variants such as Fisher SAM, whose convergence analyses are absent. Lastly, we illustrate the superiority of Riemannian SAM in terms of generalization over previous Riemannian optimization algorithms through experiments on knowledge graph completion and machine translation tasks.

----

## [0] ODE-based Recurrent Model-free Reinforcement Learning for POMDPs

**Authors**: *Xuanle Zhao, Duzhen Zhang, Liyuan Han, Tielin Zhang, Bo Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equations (ODEs) are widely recognized as the standard for modeling physical mechanisms, which help to perform approximate inference in unknown physical or biological environments. In partially observable (PO) environments, how to infer unseen information from raw observations puzzled the agents. By using a recurrent policy with a compact context, context-based reinforcement learning provides a flexible way to extract unobservable information from historical transitions. To help the agent extract more dynamics-related information, we present a novel ODE-based recurrent model combines with model-free reinforcement learning (RL) framework to solve partially observable Markov decision processes (POMDPs). We experimentally demonstrate the efficacy of our methods across various PO continuous control and meta-RL tasks. Furthermore, our experiments illustrate that our method is robust against irregular observations, owing to the ability of ODEs to model irregularly-sampled time series.

----

## [0] Deep Contract Design via Discontinuous Networks

**Authors**: *Tonghan Wang, Paul Duetting, Dmitry Ivanov, Inbal Talgam-Cohen, David C. Parkes*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7700139af1fa346d2f57f1f5c26c18-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7700139af1fa346d2f57f1f5c26c18-Abstract-Conference.html)

**Abstract**:

Contract design involves a principal who establishes contractual agreements about payments for outcomes that arise from the actions of an agent. In this paper, we initiate the study of deep learning for the automated design of optimal contracts. We introduce a novel representation: the Discontinuous ReLU (DeLU) network, which models the principal's utility as a discontinuous piecewise affine function of the design of a contract where each piece corresponds to the agent taking a particular action. DeLU networks implicitly learn closed-form expressions for the incentive compatibility constraints of the agent and the utility maximization objective of the principal, and support parallel inference on each piece through linear programming or interior-point methods that solve for optimal contracts. We provide empirical results that demonstrate success in approximating the principal's utility function with a small number of training samples and scaling to find approximately optimal contracts on problems with a large number of actions and outcomes.

----

## [0] Temporal Continual Learning with Prior Compensation for Human Motion Prediction

**Authors**: *Jianwei Tang, Jiangxin Sun, Xiaotong Lin, Lifang Zhang, Wei-Shi Zheng, Jian-Fang Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7a83a5342befd11d3d65beba1be5b0-Abstract-Conference.html)

**Abstract**:

Human Motion Prediction (HMP) aims to predict future poses at different moments according to past motion sequences. Previous approaches have treated the prediction of various moments equally, resulting in two main limitations: the learning of short-term predictions is hindered by the focus on long-term predictions, and the incorporation of prior information from past predictions into subsequent predictions is limited. In this paper, we introduce a novel multi-stage training framework called Temporal Continual Learning (TCL) to address the above challenges. To better preserve prior information, we introduce the Prior Compensation Factor (PCF). We incorporate it into the model training to compensate for the lost prior information. Furthermore, we derive a more reasonable optimization objective through theoretical derivation. It is important to note that our TCL framework can be easily integrated with different HMP backbone models and adapted to various datasets and applications. Extensive experiments on four HMP benchmark datasets demonstrate the effectiveness and flexibility of TCL. The code is available at https://github.com/hyqlat/TCL.

----

## [0] Kernel Quadrature with Randomly Pivoted Cholesky

**Authors**: *Ethan Epperly, Elvira Moreno*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cf7ba4b2d14e0f6a0e8247af77745094-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cf7ba4b2d14e0f6a0e8247af77745094-Abstract-Conference.html)

**Abstract**:

This paper presents new quadrature rules for functions in a reproducing kernel Hilbert space using nodes drawn by a sampling algorithm known as randomly pivoted Cholesky. The resulting computational procedure compares favorably to previous kernel quadrature methods, which either achieve low accuracy or require solving a computationally challenging sampling problem. Theoretical and numerical results show that randomly pivoted Cholesky is fast and achieves comparable quadrature error rates to more computationally expensive quadrature schemes based on continuous volume sampling, thinning, and recombination. Randomly pivoted Cholesky is easily adapted to complicated geometries with arbitrary kernels, unlocking new potential for kernel quadrature.

----

## [0] Analyzing the Sample Complexity of Self-Supervised Image Reconstruction Methods

**Authors**: *Tobit Klug, Dogukan Atik, Reinhard Heckel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfaea3a519edf73c3a0480ae8f00bc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfaea3a519edf73c3a0480ae8f00bc4e-Abstract-Conference.html)

**Abstract**:

Supervised training of deep neural networks on pairs of clean image and noisy measurement achieves state-of-the-art performance for many image reconstruction tasks, but such training pairs are difficult to collect. Self-supervised methods enable training based on noisy measurements only, without clean images. In this work, we investigate the cost of self-supervised training in terms of sample complexity for a class of self-supervised methods that enable the computation of unbiased estimates of gradients of the supervised loss, including noise2noise methods. We analytically show that a model trained with such self-supervised training is as good as the same model trained in a supervised fashion, but self-supervised training requires more examples than supervised training. We then study self-supervised denoising and accelerated MRI empirically and characterize the cost of self-supervised training in terms of the number of additional samples required, and find that the performance gap between self-supervised and supervised training vanishes as a function of the training examples, at a problem-dependent rate, as predicted by our theory.

----

## [0] FIRAL: An Active Learning Algorithm for Multinomial Logistic Regression

**Authors**: *Youguang Chen, George Biros*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfcadfe84ee49908cde1fc2992c38d20-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfcadfe84ee49908cde1fc2992c38d20-Abstract-Conference.html)

**Abstract**:

We investigate theory and algorithms for pool-based active learning for multiclass classification using multinomial logistic regression.  Using finite sample analysis, we prove that the Fisher Information Ratio (FIR)  lower and upper bounds  the excess risk. Based on our theoretical analysis, we propose an active learning algorithm that  employs regret minimization to minimize the FIR. To verify our derived excess risk bounds, we conduct experiments on synthetic datasets. Furthermore, we compare FIRAL with five other methods and found that our scheme  outperforms them: it consistently produces the smallest classification error in the multiclass logistic regression setting, as demonstrated through experiments on MNIST, CIFAR-10, and 50-class ImageNet.

----

## [0] AMDP: An Adaptive Detection Procedure for False Discovery Rate Control in High-Dimensional Mediation Analysis

**Authors**: *Jiarong Ding, Xuehu Zhu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/cfce727868dcaf5295c0125f9d6fbc0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/cfce727868dcaf5295c0125f9d6fbc0b-Abstract-Conference.html)

**Abstract**:

High-dimensional mediation analysis is often associated with a multiple testing problem for detecting significant mediators. Assessing the uncertainty of this detecting process via false discovery rate (FDR) has garnered great interest. To control the FDR in multiple testing, two essential steps are involved: ranking and selection. Existing approaches either construct p-values without calibration or disregard the joint information across tests, leading to conservation in FDR control or non-optimal ranking rules for multiple hypotheses. In this paper, we develop an adaptive mediation detection procedure (referred to as "AMDP") to identify relevant mediators while asymptotically controlling the FDR in high-dimensional mediation analysis. AMDP produces the optimal rule for ranking hypotheses and proposes a data-driven strategy to determine the threshold for mediator selection. This novel method captures information from the proportions of composite null hypotheses and the distribution of p-values, which turns the high dimensionality into an advantage instead of a limitation. The numerical studies on synthetic and real data sets illustrate the performances of AMDP compared with existing approaches.

----

## [0] Strong and Precise Modulation of Human Percepts via Robustified ANNs

**Authors**: *Guy Gaziv, Michael J. Lee, James J. DiCarlo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d00904cebc0d5b69fada8ad33d0f1422-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d00904cebc0d5b69fada8ad33d0f1422-Abstract-Conference.html)

**Abstract**:

The visual object category reports of artificial neural networks (ANNs) are notoriously sensitive to tiny, adversarial image perturbations. Because human category reports (aka human percepts) are thought to be insensitive to those same small-norm perturbations -- and locally stable in general -- this argues that ANNs are incomplete scientific models of human visual perception. Consistent with this, we show that when small-norm image perturbations are generated by standard ANN models, human object category percepts are indeed highly stable.  However, in this very same "human-presumed-stable" regime, we find that robustified ANNs reliably discover low-norm image perturbations that strongly disrupt human percepts. These previously undetectable human perceptual disruptions are massive in amplitude, approaching the same level of sensitivity seen in robustified ANNs.  Further, we show that robustified ANNs support precise perceptual state interventions: they guide the construction of low-norm image perturbations that strongly alter human category percepts toward specific prescribed percepts.  In sum, these contemporary models of biological visual processing are now accurate enough to guide strong and precise interventions on human perception.

----

## [0] MomentDiff: Generative Video Moment Retrieval from Random to Real

**Authors**: *Pandeng Li, Chen-Wei Xie, Hongtao Xie, Liming Zhao, Lei Zhang, Yun Zheng, Deli Zhao, Yongdong Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d01bda31bbcd780774ff15b534e03c40-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d01bda31bbcd780774ff15b534e03c40-Abstract-Conference.html)

**Abstract**:

Video moment retrieval pursues an efficient and generalized solution to identify the specific temporal segments within an untrimmed video that correspond to a given language description.To achieve this goal, we provide a generative diffusion-based framework called MomentDiff, which simulates a typical human retrieval process from random browsing to gradual localization.Specifically, we first diffuse the real span to random noise, and learn to denoise the random noise to the original span with the guidance of similarity between text and video.This allows the model to learn a mapping from arbitrary random locations to real moments, enabling the ability to locate segments from random initialization.Once trained, MomentDiff could sample random temporal segments as initial guesses and iteratively refine them to generate an accurate temporal boundary.Different from discriminative works (e.g., based on learnable proposals or queries), MomentDiff with random initialized spans could resist the temporal location biases from datasets.To evaluate the influence of the temporal location biases, we propose two ``anti-bias'' datasets with location distribution shifts, named Charades-STA-Len and Charades-STA-Mom.The experimental results demonstrate that our efficient framework consistently outperforms state-of-the-art methods on three public benchmarks, and exhibits better generalization and robustness on the proposed anti-bias datasets. The code, model, and anti-bias evaluation datasets will be released publicly.

----

## [0] Experimental Designs for Heteroskedastic Variance

**Authors**: *Justin Weltz, Tanner Fiez, Alexander Volfovsky, Eric Laber, Blake Mason, Houssam Nassif, Lalit Jain*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d01db5cd2555ba11f75da0454d57b903-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d01db5cd2555ba11f75da0454d57b903-Abstract-Conference.html)

**Abstract**:

Most linear experimental design problems assume homogeneous variance, while the presence of heteroskedastic noise is present in many realistic settings. Let a learner have access to a finite set of measurement vectors $\mathcal{X}\subset \mathbb{R}^d$ that can be probed to receive noisy linear responses of the form $y=x^{\top}\theta^{\ast}+\eta$. Here $\theta^{\ast}\in \mathbb{R}^d$ is an unknown parameter vector, and $\eta$ is independent mean-zero $\sigma_x^2$-sub-Gaussian noise defined by a flexible heteroskedastic variance model, $\sigma_x^2 = x^{\top}\Sigma^{\ast}x$. Assuming that $\Sigma^{\ast}\in \mathbb{R}^{d\times d}$ is an unknown matrix, we propose, analyze and empirically evaluate a novel design for uniformly bounding estimation error of the variance parameters, $\sigma_x^2$. We demonstrate this method on two adaptive experimental design problems under heteroskedastic noise, fixed confidence transductive best-arm identification and level-set identification and prove the first instance-dependent lower bounds in these settings.Lastly, we construct near-optimal algorithms and demonstrate the large improvements in sample complexity gained from accounting for heteroskedastic variance in these designs empirically.

----

## [0] NeuralGF: Unsupervised Point Normal Estimation by Learning Neural Gradient Function

**Authors**: *Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d027a5c93d484a4312cc486d399c62c1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d027a5c93d484a4312cc486d399c62c1-Abstract-Conference.html)

**Abstract**:

Normal estimation for 3D point clouds is a fundamental task in 3D geometry processing. The state-of-the-art methods rely on priors of fitting local surfaces learned from normal supervision. However, normal supervision in benchmarks comes from synthetic shapes and is usually not available from real scans, thereby limiting the learned priors of these methods. In addition, normal orientation consistency across shapes remains difficult to achieve without a separate post-processing procedure. To resolve these issues, we propose a novel method for estimating oriented normals directly from point clouds without using ground truth normals as supervision. We achieve this by introducing a new paradigm for learning neural gradient functions, which encourages the neural network to fit the input point clouds and yield unit-norm gradients at the points. Specifically, we introduce loss functions to facilitate query points to iteratively reach the moving targets and aggregate onto the approximated surface, thereby learning a global surface representation of the data. Meanwhile, we incorporate gradients into the surface approximation to measure the minimum signed deviation of queries, resulting in a consistent gradient field associated with the surface. These techniques lead to our deep unsupervised oriented normal estimator that is robust to noise, outliers and density variations. Our excellent results on widely used benchmarks demonstrate that our method can learn more accurate normals for both unoriented and oriented normal estimation tasks than the latest methods. The source code and pre-trained model are publicly available.

----

## [0] Gacs-Korner Common Information Variational Autoencoder

**Authors**: *Michael Kleinman, Alessandro Achille, Stefano Soatto, Jonathan C. Kao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d04f08ccf582011f43af91ee1c1956d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d04f08ccf582011f43af91ee1c1956d2-Abstract-Conference.html)

**Abstract**:

We propose a notion of common information that allows one to quantify and separate the information that is shared between two random variables from the information that is unique to each. Our notion of common information is defined by an optimization problem over a family of functions and recovers the G\'acs-K\"orner common information as a special case. Importantly, our notion can be approximated empirically using samples from the underlying data distribution. We then provide a method to partition and quantify the common and unique information using a simple modification of a traditional variational auto-encoder. Empirically, we demonstrate that our formulation allows us to learn semantically meaningful common and unique factors of variation even on high-dimensional data such as images and videos. Moreover, on datasets where ground-truth latent factors are known, we show that we can accurately quantify the common information between the random variables.

----

## [0] LEACE: Perfect linear concept erasure in closed form

**Authors**: *Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html)

**Abstract**:

Concept erasure aims to remove specified features from a representation. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the representation as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called concept scrubbing, which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings. Our code is available at https://github.com/EleutherAI/concept-erasure.

----

## [0] Robust low-rank training via approximate orthonormal constraints

**Authors**: *Dayana Savostianova, Emanuele Zangrando, Gianluca Ceruti, Francesco Tudisco*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d073692637b4fb8c4eb4b81f0fa2df7b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d073692637b4fb8c4eb4b81f0fa2df7b-Abstract-Conference.html)

**Abstract**:

With the growth of model and data sizes, a broad effort has been made to design pruning techniques that reduce the resource demand of deep learning pipelines, while retaining model performance. In order to reduce both inference and training costs, a prominent line of work uses low-rank matrix factorizations to represent the network weights. Although able to retain accuracy,  we observe that low-rank methods tend to compromise model robustness against adversarial perturbations. By modeling robustness in terms of the condition number of the neural network, we argue that this loss of robustness is due to the exploding singular values of the low-rank weight matrices. Thus, we introduce a robust low-rank training algorithm that maintains the network's weights on the low-rank matrix manifold while  simultaneously enforcing approximate orthonormal constraints. The resulting model reduces both training and inference costs while ensuring well-conditioning and thus better adversarial robustness, without compromising model accuracy. This is shown by extensive numerical evidence and by our main approximation theorem that shows the computed robust low-rank network well-approximates the ideal full model, provided a highly performing low-rank sub-network exists.

----

## [0] Symmetry-Informed Geometric Representation for Molecules, Proteins, and Crystalline Materials

**Authors**: *Shengchao Liu, Weitao Du, Yanjing Li, Zhuoxinran Li, Zhiling Zheng, Chenru Duan, Zhi-Ming Ma, Omar Yaghi, Animashree Anandkumar, Christian Borgs, Jennifer T. Chayes, Hongyu Guo, Jian Tang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d07379f3acf3af51dfc8598862cadfa0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d07379f3acf3af51dfc8598862cadfa0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Artificial intelligence for scientific discovery has recently generated significant interest within the machine learning and scientific communities, particularly in the domains of chemistry, biology, and material discovery. For these scientific problems, molecules serve as the fundamental building blocks, and machine learning has emerged as a highly effective and powerful tool for modeling their geometric structures. Nevertheless, due to the rapidly evolving process of the field and the knowledge gap between science ({\eg}, physics,  chemistry, \& biology) and machine learning communities, a benchmarking study on geometrical representation for such data has not been conducted. To address such an issue, in this paper, we first provide a unified view of the current symmetry-informed geometric methods, classifying them into three main categories: invariance, equivariance with spherical frame basis, and equivariance with vector frame basis. Then we propose a platform, coined Geom3D, which enables benchmarking the effectiveness of geometric strategies. Geom3D contains 16 advanced symmetry-informed geometric representation models and 14 geometric pretraining methods over 52 diverse tasks, including small molecules, proteins, and crystalline materials. We hope that Geom3D can, on the one hand, eliminate barriers for machine learning researchers interested in exploring scientific problems; and, on the other hand, provide valuable guidance for researchers in computational chemistry, structural biology, and materials science, aiding in the informed selection of representation techniques for specific applications. The source code is available on \href{https://github.com/chao1224/Geom3D}{the GitHub repository}.

----

## [0] Feature Selection in the Contrastive Analysis Setting

**Authors**: *Ethan Weinberger, Ian Covert, Su-In Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d083980ec9f874025550136b776a96a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d083980ec9f874025550136b776a96a9-Abstract-Conference.html)

**Abstract**:

Contrastive analysis (CA) refers to the exploration of variations uniquely enriched in a target dataset as compared to a corresponding background dataset generated from sources of variation that are irrelevant to a given task. For example, a biomedical data analyst may wish to find a small set of genes to use as a proxy for variations in genomic data only present among patients with a given disease (target) as opposed to healthy control subjects (background). However, as of yet the problem of feature selection in the CA setting has received little attention from the machine learning community. In this work we present contrastive feature selection (CFS),a method for performing feature selection in the CA setting. We motivate our approach with a novel information-theoretic analysis of representation learning in the CA setting, and we empirically validate CFS on a semi-synthetic dataset and four real-world biomedical datasets. We find that our method consistently outperforms previously proposed state-of-the-art supervised and fully unsupervised feature selection methods not designed for the CA setting. An open-source implementation of our method is available at https://github.com/suinleelab/CFS.

----

## [0] GeoDE: a Geographically Diverse Evaluation Dataset for Object Recognition

**Authors**: *Vikram V. Ramaswamy, Sing Yu Lin, Dora Zhao, Aaron Adcock, Laurens van der Maaten, Deepti Ghadiyaram, Olga Russakovsky*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d08b6801f24dda81199079a3371d77f9-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d08b6801f24dda81199079a3371d77f9-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Current dataset collection methods typically scrape large amounts of data from the web. While this technique is extremely scalable, data collected in this way tends to reinforce stereotypical biases, can contain personally identifiable information, and typically originates from Europe and North America. In this work, we rethink the dataset collection paradigm and introduce GeoDE, a geographically diverse dataset with 61,940 images from 40 classes and 6 world regions, and no personally identifiable information, collected by soliciting images from people across the world. We analyse GeoDE to understand differences in images collected in this manner compared to web-scraping. Despite the smaller size of this dataset, we demonstrate its use as both an evaluation and training dataset, allowing us to highlight shortcomings in current models, as well as demonstrate improved performance even when training on this small dataset. We release the full dataset and code at https://geodiverse-data-collection.cs.princeton.edu/

----

## [0] Last-Iterate Convergent Policy Gradient Primal-Dual Methods for Constrained MDPs

**Authors**: *Dongsheng Ding, Chen-Yu Wei, Kaiqing Zhang, Alejandro Ribeiro*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0949cbcec31c09431610553a284f94a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0949cbcec31c09431610553a284f94a-Abstract-Conference.html)

**Abstract**:

We study the problem of computing an optimal policy of an infinite-horizon discounted constrained Markov decision process (constrained MDP). Despite the popularity of Lagrangian-based policy search methods used in practice, the oscillation of policy iterates in these methods has not been fully understood, bringing out issues such as violation of constraints and sensitivity to hyper-parameters. To fill this gap, we employ the Lagrangian method to cast a constrained MDP into a constrained saddle-point problem in which max/min players correspond to primal/dual variables, respectively, and develop two single-time-scale policy-based primal-dual algorithms with non-asymptotic convergence of their policy iterates to an optimal constrained policy. Specifically, we first propose a regularized policy gradient primal-dual (RPG-PD) method that updates the policy using an entropy-regularized policy gradient, and the dual variable via a quadratic-regularized gradient ascent, simultaneously. We prove that the policy primal-dual iterates of RPG-PD converge to a regularized saddle point with a sublinear rate, while the policy iterates converge sublinearly to an optimal constrained policy. We further instantiate RPG-PD in large state or action spaces by including function approximation in policy parametrization, and establish similar sublinear last-iterate policy convergence. Second, we propose an optimistic policy gradient primal-dual (OPG-PD) method that employs the optimistic gradient method to update primal/dual variables, simultaneously. We prove that the policy primal-dual iterates of OPG-PD converge to a saddle point that contains an optimal constrained policy, with a linear rate. To the best of our knowledge, this work appears to be the first non-asymptotic policy last-iterate convergence result for single-time-scale algorithms in constrained MDPs. We further validate the merits and the effectiveness of our methods in computational experiments.

----

## [0] Unleashing the Power of Randomization in Auditing Differentially Private ML

**Authors**: *Krishna Pillutla, Galen Andrew, Peter Kairouz, H. Brendan McMahan, Alina Oprea, Sewoong Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d09ef5264966e17adffd3157265c9946-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d09ef5264966e17adffd3157265c9946-Abstract-Conference.html)

**Abstract**:

We present a rigorous methodology for auditing differentially private machine learning by adding multiple carefully designed examples called canaries. We take a first principles approach based on three key components. First, we introduce Lifted Differential Privacy (LiDP) that expands the definition of differential privacy to handle randomized datasets. This gives us the freedom to design randomized canaries. Second, we audit LiDP by trying to distinguish between the model trained with $K$ canaries versus $K-1$ canaries in the dataset, leaving one canary out. By drawing the canaries i.i.d., LiDP can leverage the symmetry in the design and reuse each privately trained model to run multiple statistical tests, one for each canary. Third, we introduce novel confidence intervals that take advantage of the multiple test statistics by adapting to the empirical higher-order correlations. Together, this new recipe demonstrates significant improvements in sample complexity, both theoretically and empirically, using synthetic and real data. Further, recent advances in designing stronger canaries can be readily incorporated in the new framework.

----

## [0] Worst-case Performance of Popular Approximate Nearest Neighbor Search Implementations: Guarantees and Limitations

**Authors**: *Piotr Indyk, Haike Xu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0ac28b79816b51124fcc804b2496a36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0ac28b79816b51124fcc804b2496a36-Abstract-Conference.html)

**Abstract**:

Graph-based approaches to nearest neighbor search are popular and powerful tools for handling large datasets in practice, but they have limited theoretical guarantees. We study the worst-case performance of recent graph-based approximate nearest neighbor search algorithms, such as HNSW, NSG and DiskANN. For DiskANN, we show that its "slow preprocessing'' version provably supports approximate nearest neighbor search query with constant approximation ratio and poly-logarithmic query time, on data sets with bounded "intrinsic'' dimension. For the other data structure variants studied, including DiskANN with "fast preprocessing'',  HNSW and NSG,  we present a family of instances on which the empirical query time required to achieve a "reasonable'' accuracy is linear in instance size. For example, for DiskANN, we show that the query procedure can take at least  $0.1 n$ steps on instances of size $n$ before it encounters any of the $5$ nearest neighbors of the query.

----

## [0] On the Overlooked Structure of Stochastic Gradients

**Authors**: *Zeke Xie, Qian-Yuan Tang, Mingming Sun, Ping Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0b2eda0386f477ab14d7e181e16c899-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0b2eda0386f477ab14d7e181e16c899-Abstract-Conference.html)

**Abstract**:

Stochastic gradients closely relate to both optimization and generalization of deep neural networks (DNNs). Some works attempted to explain the success of stochastic optimization for deep learning by the arguably heavy-tail properties of gradient noise, while other works presented theoretical and empirical evidence against the heavy-tail hypothesis on gradient noise. Unfortunately, formal statistical tests for analyzing the structure and heavy tails of stochastic gradients in deep learning are still under-explored. In this paper, we mainly make two contributions. First, we conduct formal statistical tests on the distribution of stochastic gradients and gradient noise across both parameters and iterations. Our statistical tests reveal that dimension-wise gradients usually exhibit power-law heavy tails, while iteration-wise gradients and stochastic gradient noise caused by minibatch training usually do not exhibit power-law heavy tails. Second, we further discover that the covariance spectra of stochastic gradients have the power-law structures overlooked by previous studies and present its theoretical implications for training of DNNs. While previous studies believed that the anisotropic structure of stochastic gradients matters to deep learning, they did not expect the gradient covariance can have such an elegant mathematical structure. Our work challenges the existing belief and provides novel insights on the structure of stochastic gradients in deep learning.

----

## [0] ECG-QA: A Comprehensive Question Answering Dataset Combined With Electrocardiogram

**Authors**: *Jungwoo Oh, Gyubok Lee, Seongsu Bae, Joon-Myoung Kwon, Edward Choi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0b67349dd16b83b2cf6167fb4e2be50-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0b67349dd16b83b2cf6167fb4e2be50-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Question answering (QA) in the field of healthcare has received much attention due to significant advancements in natural language processing. However, existing healthcare QA datasets primarily focus on medical images, clinical notes, or structured electronic health record tables. This leaves the vast potential of combining electrocardiogram (ECG) data with these systems largely untapped. To address this gap, we present ECG-QA, the first QA dataset specifically designed for ECG analysis. The dataset comprises a total of 70 question templates that cover a wide range of clinically relevant ECG topics, each validated by an ECG expert to ensure their clinical utility. As a result, our dataset includes diverse ECG interpretation questions, including those that require a comparative analysis of two different ECGs. In addition, we have conducted numerous experiments to provide valuable insights for future research directions. We believe that ECG-QA will serve as a valuable resource for the development of intelligent QA systems capable of assisting clinicians in ECG interpretations.

----

## [0] Provable convergence guarantees for black-box variational inference

**Authors**: *Justin Domke, Robert M. Gower, Guillaume Garrigos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0bcff6425bbf850ec87d5327a965db9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0bcff6425bbf850ec87d5327a965db9-Abstract-Conference.html)

**Abstract**:

Black-box variational inference is widely used in situations where there is no proof that its stochastic optimization succeeds. We suggest this is due to a theoretical gap in existing stochastic optimization proofsâ€”namely the challenge of gradient estimators with unusual noise bounds, and a composite non-smooth objective. For dense Gaussian variational families, we observe that existing gradient estimators based on reparameterization satisfy a quadratic noise bound and give novel convergence guarantees for proximal and projected stochastic gradient descent using this bound. This provides rigorous guarantees that methods similar to those used in practice converge on realistic inference problems.

----

## [0] Seeing is not Believing: Robust Reinforcement Learning against Spurious Correlation

**Authors**: *Wenhao Ding, Laixi Shi, Yuejie Chi, Ding Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0c3841867db2c516454845a450ca885-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0c3841867db2c516454845a450ca885-Abstract-Conference.html)

**Abstract**:

Robustness has been extensively studied in reinforcement learning (RL) to handle various forms of uncertainty such as random perturbations, rare events, and malicious attacks. In this work, we consider one critical type of robustness against spurious correlation, where different portions of the state do not have correlations induced by unobserved confounders. These spurious correlations are ubiquitous in real-world tasks, for instance, a self-driving car usually observes heavy traffic in the daytime and light traffic at night due to unobservable human activity. A model that learns such useless or even harmful correlation could catastrophically fail when the confounder in the test case deviates from the training one. Although motivated, enabling robustness against spurious correlation poses significant challenges since the uncertainty set, shaped by the unobserved confounder and causal structure, is difficult to characterize and identify. Existing robust algorithms that assume simple and unstructured uncertainty sets are therefore inadequate to address this challenge. To solve this issue, we propose Robust State-Confounded Markov Decision Processes (RSC-MDPs) and theoretically demonstrate its superiority in avoiding learning spurious correlations compared with other robust RL counterparts. We also design an empirical algorithm to learn the robust optimal policy for RSC-MDPs, which outperforms all baselines in eight realistic self-driving and manipulation tasks.

----

## [0] IBA: Towards Irreversible Backdoor Attacks in Federated Learning

**Authors**: *Thuy Dung Nguyen, Tuan Nguyen, Anh Tran, Khoa D. Doan, Kok-Seng Wong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0c6bc641a56bebee9d985b937307367-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) is a distributed learning approach that enables machine learning models to be trained on decentralized data without compromising end devices' personal, potentially sensitive data. However, the distributed nature and uninvestigated data intuitively introduce new security vulnerabilities, including backdoor attacks. In this scenario, an adversary implants backdoor functionality into the global model during training, which can be activated to cause the desired misbehaviors for any input with a specific adversarial pattern. Despite having remarkable success in triggering and distorting model behavior, prior backdoor attacks in FL often hold impractical assumptions, limited imperceptibility, and durability. Specifically, the adversary needs to control a sufficiently large fraction of clients or know the data distribution of other honest clients. In many cases, the trigger inserted is often visually apparent, and the backdoor effect is quickly diluted if the adversary is removed from the training process. To address these limitations, we propose a novel backdoor attack framework in FL, the Irreversible Backdoor Attack (IBA), that jointly learns the optimal and visually stealthy trigger and then gradually implants the backdoor into a global model. This approach allows the adversary to execute a backdoor attack that can evade both human and machine inspections. Additionally, we enhance the efficiency and durability of the proposed attack by selectively poisoning the model's parameters that are least likely updated by the main task's learning process and constraining the poisoned model update to the vicinity of the global model. Finally, we evaluate the proposed attack framework on several benchmark datasets, including MNIST, CIFAR-10, and Tiny ImageNet, and achieved high success rates while simultaneously bypassing existing backdoor defenses and achieving a more durable backdoor effect compared to other backdoor attacks. Overall, IBA offers a more effective, stealthy, and durable approach to backdoor attacks in FL. The code associated with this paper is available on GitHub.

----

## [0] Spontaneous symmetry breaking in generative diffusion models

**Authors**: *Gabriel Raya, Luca Ambrogioni*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d0da30e312b75a3fffd9e9191f8bc1b0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d0da30e312b75a3fffd9e9191f8bc1b0-Abstract-Conference.html)

**Abstract**:

Generative diffusion models have recently emerged as a leading approach for generating high-dimensional data. In this paper, we show that the dynamics of these models exhibit a spontaneous symmetry breaking that divides the generative dynamics into two distinct phases: 1) A linear steady-state dynamics around a central fixed-point and 2) an attractor dynamics directed towards the data manifold. These two "phases'' are separated by the change in stability of the central fixed-point, with the resulting window of instability being responsible for the diversity of the generated samples. Using both theoretical and empirical evidence, we show that an accurate simulation of the early dynamics does not significantly contribute to the final generation, since early fluctuations are reverted to the central fixed point. To leverage this insight, we propose a Gaussian late initialization scheme, which significantly improves model performance, achieving up to 3x FID improvements on fast samplers, while also increasing sample diversity (e.g., racial composition of generated CelebA images). Our work offers a new way to understand the generative dynamics of diffusion models that has the potential to bring about higher performance and less biased fast-samplers.

----

## [0] RL-based Stateful Neural Adaptive Sampling and Denoising for Real-Time Path Tracing

**Authors**: *Antoine Scardigli, Lukas Cavigelli, Lorenz K. Müller*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1422213c9f2bdd5178b77d166fba86a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1422213c9f2bdd5178b77d166fba86a-Abstract-Conference.html)

**Abstract**:

Monte-Carlo path tracing is a powerful technique for realistic image synthesis but suffers from high levels of noise at low sample counts, limiting its use in real-time applications. To address this, we propose a framework with end-to-end training of a sampling importance network, a latent space encoder network, and a denoiser network. Our approach uses reinforcement learning to optimize the sampling importance network, thus avoiding explicit numerically approximated gradients. Our method does not aggregate the sampled values per pixel by averaging but keeps all sampled values which are then fed into the latent space encoder. The encoder replaces handcrafted spatiotemporal heuristics by learned representations in a latent space. Finally, a neural denoiser is trained to refine the output image. Our approach increases visual quality on several challenging datasets and reduces rendering times for equal quality by a factor of 1.6x compared to the previous state-of-the-art, making it a promising solution for real-time applications.

----

## [0] A Data-Free Approach to Mitigate Catastrophic Forgetting in Federated Class Incremental Learning for Vision Tasks

**Authors**: *Sara Babakniya, Zalan Fabian, Chaoyang He, Mahdi Soltanolkotabi, Salman Avestimehr*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d160ea01902c33e30660851dfbac5980-Abstract-Conference.html)

**Abstract**:

Deep learning models often suffer from forgetting previously learned information when trained on new data. This problem is exacerbated in federated learning (FL), where the data is distributed and can change independently for each user. Many solutions are proposed to resolve this catastrophic forgetting in a centralized setting. However, they do not apply directly to FL because of its unique complexities, such as privacy concerns and resource limitations. To overcome these challenges, this paper presents a framework for \textbf{federated class incremental learning} that utilizes a generative model to synthesize samples from past distributions. This data can be later exploited alongside the training data to mitigate catastrophic forgetting. To preserve privacy, the generative model is trained on the server using data-free methods at the end of each task without requesting data from clients. Moreover, our solution does not demand the users to store old data or models, which gives them the freedom to join/leave the training at any time. Additionally, we introduce SuperImageNet, a new regrouping of the ImageNet dataset specifically tailored for federated continual learning. We demonstrate significant improvements compared to existing baselines through extensive experiments on multiple datasets.

----

## [0] On the Connection between Pre-training Data Diversity and Fine-tuning Robustness

**Authors**: *Vivek Ramanujan, Thao Nguyen, Sewoong Oh, Ali Farhadi, Ludwig Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/d1786f5246c67eefde011599d31b2006-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/d1786f5246c67eefde011599d31b2006-Abstract-Conference.html)

**Abstract**:

Pre-training has been widely adopted in deep learning to improve model performance, especially when the training data for a target task is limited. In our work, we seek to understand the implications of this training strategy on the generalization properties of downstream models. More specifically, we ask the following question: how do properties of the pre-training distribution affect the robustness of a fine-tuned model? The properties we explore include the label space, label semantics, image diversity, data domains, and data quantity of the pre-training distribution. We find that the primary factor influencing downstream effective robustness (Taori et al., 2020) is data quantity, while other factors have limited significance. For example, reducing the number of ImageNet pre-training classes by 4x while increasing the number of images per class by 4x (that is, keeping total data quantity fixed) does not impact the robustness of fine-tuned models. We demonstrate our findings on pre-training distributions drawn from various natural and synthetic data sources, primarily using the iWildCam-WILDS distribution shift as a test for robustness.

----



[Go to the previous page](NIPS-2023-list2800.md)

[Go to the next page](NIPS-2023-list2802.md)

[Go to the catalog section](README.md)