## [0] Beyond Value-Function Gaps: Improved Instance-Dependent Regret Bounds for Episodic Reinforcement Learning

        **Authors**: *Christoph Dann, Teodor Vanislavov Marinov, Mehryar Mohri, Julian Zimmert*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/000c076c390a4c357313fca29e390ece-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/000c076c390a4c357313fca29e390ece-Abstract.html)

        **Abstract**:

        We provide improved gap-dependent regret bounds for reinforcement learning in finite episodic Markov decision processes. Compared to prior work, our bounds depend on alternative definitions of gaps. These definitions are based on the insight that, in order to achieve a favorable regret, an algorithm does not need to learn how to behave optimally in states that are not reached by an optimal policy. We prove tighter upper regret bounds for optimistic algorithms and accompany them with new information-theoretic lower bounds for a large class of MDPs. Our results show that optimistic algorithms can not achieve the information-theoretic lower bounds even in deterministic MDPs unless there is a unique optimal policy.

        ----

        ## [1] Learning One Representation to Optimize All Rewards

        **Authors**: *Ahmed Touati, Yann Ollivier*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/003dd617c12d444ff9c80f717c3fa982-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/003dd617c12d444ff9c80f717c3fa982-Abstract.html)

        **Abstract**:

        We introduce the forward-backward (FB) representation of the dynamics of a reward-free Markov decision process. It provides explicit near-optimal policies for any reward specified a posteriori. During an unsupervised phase, we use reward-free interactions with the environment to learn two representations via off-the-shelf deep learning methods and temporal difference (TD) learning. In the test phase, a reward representation is estimated either from reward observations or an explicit reward description (e.g., a target state). The optimal policy for thatreward is directly obtained from these representations, with no planning. We assume access to an exploration scheme or replay buffer for the first phase.The corresponding unsupervised loss is well-principled: if training is perfect, the policies obtained are provably optimal for any reward function.  With imperfect training, the sub-optimality is proportional to the unsupervised approximation error. The FB representation learns long-range relationships between states and actions, via a predictive occupancy map, without having to synthesize states as in model-based approaches.This is a step towards learning controllable agents in arbitrary black-box stochastic environments. This approach compares well to goal-oriented RL algorithms on discrete and continuous mazes, pixel-based MsPacman, and the FetchReach virtual robot arm. We also illustrate how the agent can immediately adapt to new tasks beyond goal-oriented RL.

        ----

        ## [2] Matrix factorisation and the interpretation of geodesic distance

        **Authors**: *Nick Whiteley, Annie Gray, Patrick Rubin-Delanchy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/007ff380ee5ac49ffc34442f5c2a2b86-Abstract.html)

        **Abstract**:

        Given a graph or similarity matrix, we consider the problem of recovering a notion of true distance between the nodes, and so their true positions. We show that this can be accomplished in two steps: matrix factorisation, followed by nonlinear dimension reduction. This combination is effective because the point cloud obtained in the first step lives close to a manifold in which latent distance is encoded as geodesic distance. Hence, a nonlinear dimension reduction tool, approximating geodesic distance, can recover the latent positions, up to a simple transformation. We give a detailed account of the case where spectral embedding is used, followed by Isomap, and provide encouraging experimental evidence for other combinations of techniques.

        ----

        ## [3] UniDoc: Unified Pretraining Framework for Document Understanding

        **Authors**: *Jiuxiang Gu, Jason Kuen, Vlad I. Morariu, Handong Zhao, Rajiv Jain, Nikolaos Barmpalios, Ani Nenkova, Tong Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0084ae4bc24c0795d1e6a4f58444d39b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0084ae4bc24c0795d1e6a4f58444d39b-Abstract.html)

        **Abstract**:

        Document intelligence automates the extraction of information from documents and supports many business applications. Recent self-supervised learning methods on large-scale unlabeled document datasets have opened up promising directions towards reducing annotation efforts by training models with self-supervised objectives. However, most of the existing document pretraining methods are still language-dominated. We present UDoc, a new unified pretraining framework for document understanding. UDoc is designed to support most document understanding tasks, extending the Transformer to take multimodal embeddings as input. Each input element is composed of words and visual features from a semantic region of the input document image. An important feature of UDoc is that it learns a generic representation by making use of three self-supervised losses, encouraging the representation to model sentences, learn similarities, and align modalities. Extensive empirical analysis demonstrates that the pretraining procedure learns better joint representations and leads to improvements in downstream tasks.

        ----

        ## [4] Finding Discriminative Filters for Specific Degradations in Blind Super-Resolution

        **Authors**: *Liangbin Xie, Xintao Wang, Chao Dong, Zhongang Qi, Ying Shan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/008bd5ad93b754d500338c253d9c1770-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/008bd5ad93b754d500338c253d9c1770-Abstract.html)

        **Abstract**:

        Recent blind super-resolution (SR) methods typically consist of two branches, one for degradation prediction and the other for conditional restoration. However, our experiments show that a one-branch network can achieve comparable performance to the two-branch scheme. Then we wonder: how can one-branch networks automatically learn to distinguish degradations? To find the answer, we propose a new diagnostic tool -- Filter Attribution method based on Integral Gradient (FAIG). Unlike previous integral gradient methods, our FAIG aims at finding the most discriminative filters instead of input pixels/features for degradation removal in blind SR networks. With the discovered filters, we further develop a simple yet effective method to predict the degradation of an input image. Based on FAIG, we show that, in one-branch blind SR networks, 1) we could find a very small number of (1%) discriminative filters for each specific degradation; 2) The weights, locations and connections of the discovered filters are all important to determine the specific network function. 3) The task of degradation prediction can be implicitly realized by these discriminative filters without explicit supervised learning. Our findings can not only help us better understand network behaviors inside one-branch blind SR networks, but also provide guidance on designing more efficient architectures and diagnosing networks for blind SR.

        ----

        ## [5] Counterfactual Explanations Can Be Manipulated

        **Authors**: *Dylan Slack, Anna Hilgard, Himabindu Lakkaraju, Sameer Singh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/009c434cab57de48a31f6b669e7ba266-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/009c434cab57de48a31f6b669e7ba266-Abstract.html)

        **Abstract**:

        Counterfactual explanations are emerging as an attractive option for providing recourse to individuals adversely impacted by algorithmic decisions.  As they are deployed in critical applications (e.g. law enforcement, financial lending), it becomes important to ensure that we clearly understand the vulnerabilties of these methods and find ways to address them. However, there is little understanding of the vulnerabilities and shortcomings of counterfactual explanations. In this work, we introduce the first framework that describes the vulnerabilities of counterfactual explanations and shows how they can be manipulated. More specifically, we show counterfactual explanations may converge to drastically different counterfactuals under a small perturbation indicating they are not robust.  Leveraging this insight, we introduce a novel objective to train seemingly fair models where counterfactual explanations find much lower cost recourse under a slight perturbation.  We describe how these models can unfairly provide low-cost recourse for specific subgroups in the data while appearing fair to auditors. We perform experiments on loan and violent crime prediction data sets where certain subgroups achieve up to 20x lower cost recourse under the perturbation. These results raise concerns regarding the dependability of current counterfactual explanation techniques, which we hope will inspire investigations in robust counterfactual explanations.

        ----

        ## [6] From Canonical Correlation Analysis to Self-supervised Graph Neural Networks

        **Authors**: *Hengrui Zhang, Qitian Wu, Junchi Yan, David Wipf, Philip S. Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/00ac8ed3b4327bdd4ebbebcb2ba10a00-Abstract.html)

        **Abstract**:

        We introduce a conceptually simple yet effective model for self-supervised representation learning with graph data. It follows the previous methods that generate two views of an input graph through data augmentation. However, unlike contrastive methods that focus on instance-level discrimination, we optimize an innovative feature-level objective inspired by classical Canonical Correlation Analysis. Compared with other works, our approach requires none of the parameterized mutual information estimator, additional projector, asymmetric structures, and most importantly, negative samples which can be costly. We show that the new objective essentially 1) aims at discarding augmentation-variant information by learning invariant representations, and 2) can prevent degenerated solutions by decorrelating features in different dimensions. Our theoretical analysis further provides an understanding for the new objective which can be equivalently seen as an instantiation of the Information Bottleneck Principle under the self-supervised setting. Despite its simplicity, our method performs competitively on seven public graph datasets.

        ----

        ## [7] BAST: Bayesian Additive Regression Spanning Trees for Complex Constrained Domain

        **Authors**: *Zhao Tang Luo, Huiyan Sang, Bani K. Mallick*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/00b76fddeaaa7d8c2c43d504b2babd8a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/00b76fddeaaa7d8c2c43d504b2babd8a-Abstract.html)

        **Abstract**:

        Nonparametric regression on complex domains has been a challenging task as most existing methods, such as ensemble models based on binary decision trees, are not designed to account for intrinsic geometries and domain boundaries. This article proposes a Bayesian additive regression spanning trees (BAST) model for nonparametric regression on manifolds, with an emphasis on complex constrained domains or irregularly shaped spaces embedded in Euclidean spaces. Our model is built upon a random spanning tree manifold partition model as each weak learner, which is capable of capturing any irregularly shaped spatially contiguous partitions while respecting intrinsic geometries and domain boundary constraints. Utilizing many nice properties of spanning tree structures, we design an efficient Bayesian inference algorithm. Equipped with a soft prediction scheme, BAST is demonstrated to significantly outperform other competing methods in simulation experiments and in an application to the chlorophyll data in Aral Sea, due to its strong local adaptivity to different levels of smoothness.

        ----

        ## [8] Hyperbolic Busemann Learning with Ideal Prototypes

        **Authors**: *Mina Ghadimi Atigh, Martin Keller-Ressel, Pascal Mettes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01259a0cb2431834302abe2df60a1327-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01259a0cb2431834302abe2df60a1327-Abstract.html)

        **Abstract**:

        Hyperbolic space has become a popular choice of manifold for representation learning of various datatypes from tree-like structures and text to graphs. Building on the success of deep learning with prototypes in Euclidean and hyperspherical spaces, a few recent works have proposed hyperbolic prototypes for classification. Such approaches enable effective learning in low-dimensional output spaces and can exploit hierarchical relations amongst classes, but require privileged information about class labels to position the hyperbolic prototypes. In this work, we propose Hyperbolic Busemann Learning. The main idea behind our approach is to position prototypes on the ideal boundary of the Poincar\'{e} ball, which does not require prior label knowledge. To be able to compute proximities to ideal prototypes, we introduce the penalised Busemann loss. We provide theory supporting the use of ideal prototypes and the proposed loss by proving its equivalence to logistic regression in the one-dimensional case. Empirically, we show that our approach provides a natural interpretation of classification confidence, while outperforming recent hyperspherical and hyperbolic prototype approaches.

        ----

        ## [9] Backward-Compatible Prediction Updates: A Probabilistic Approach

        **Authors**: *Frederik Träuble, Julius von Kügelgen, Matthäus Kleindessner, Francesco Locatello, Bernhard Schölkopf, Peter V. Gehler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/012d9fe15b2493f21902cd55603382ec-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/012d9fe15b2493f21902cd55603382ec-Abstract.html)

        **Abstract**:

        When machine learning systems meet real world applications, accuracy is only one of several requirements. In this paper, we assay a complementary perspective originating from the increasing availability of pre-trained and regularly improving state-of-the-art models. While new improved models develop at a fast pace, downstream tasks vary more slowly or stay constant. Assume that we have a large unlabelled data set for which we want to maintain accurate predictions. Whenever a new and presumably better ML models becomes available, we encounter two problems: (i) given a limited budget, which data points should be re-evaluated using the new model?; and (ii) if the new predictions differ from the current ones, should we update? Problem (i) is about compute cost, which matters for very large data sets and models. Problem (ii) is about maintaining consistency of the predictions, which can be highly relevant for downstream applications; our demand is to avoid negative flips, i.e., changing correct to incorrect predictions. In this paper, we formalize the Prediction Update Problem and present an efficient probabilistic approach as answer to the above questions. In extensive experiments on standard classification benchmark data sets, we show that our method outperforms alternative strategies along key metrics for backward-compatible prediction updates.

        ----

        ## [10] Truncated Marginal Neural Ratio Estimation

        **Authors**: *Benjamin Kurt Miller, Alex Cole, Patrick Forré, Gilles Louppe, Christoph Weniger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01632f7b7a127233fa1188bd6c2e42e1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01632f7b7a127233fa1188bd6c2e42e1-Abstract.html)

        **Abstract**:

        Parametric stochastic simulators are ubiquitous in science, often featuring high-dimensional input parameters and/or an intractable likelihood. Performing Bayesian parameter inference in this context can be challenging. We present a neural simulation-based inference algorithm which simultaneously offers simulation efficiency and fast empirical posterior testability, which is unique among modern algorithms. Our approach is simulation efficient by simultaneously estimating low-dimensional marginal posteriors instead of the joint posterior and by proposing simulations targeted to an observation of interest via a prior suitably truncated by an indicator function.  Furthermore, by estimating a locally amortized posterior our algorithm enables efficient empirical tests of the robustness of the inference results. Since scientists cannot access the ground truth, these tests are necessary for trusting inference in real-world applications. We perform experiments on a marginalized version of the simulation-based inference benchmark and two complex and narrow posteriors, highlighting the simulator efficiency of our algorithm as well as the quality of the estimated marginal posteriors.

        ----

        ## [11] ReAct: Out-of-distribution Detection With Rectified Activations

        **Authors**: *Yiyou Sun, Chuan Guo, Yixuan Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01894d6f048493d2cacde3c579c315a3-Abstract.html)

        **Abstract**:

        Out-of-distribution (OOD) detection has received much attention lately due to its practical importance in enhancing the safe deployment of neural networks. One of the primary challenges is that models often produce highly confident predictions on OOD data, which undermines the driving principle in OOD detection that the model should only be confident about in-distribution samples. In this work, we propose ReAct—a simple and effective technique for reducing model overconfidence on OOD data. Our method is motivated by novel analysis on internal activations of neural networks, which displays highly distinctive signature patterns for OOD distributions. Our method can generalize effectively to different network architectures and different OOD detection scores. We empirically demonstrate that ReAct achieves competitive detection performance on a comprehensive suite of benchmark datasets, and give theoretical explication for our method’s efficacy. On the ImageNet benchmark, ReAct reduces the false positive rate (FPR95) by 25.05% compared to the previous best method.

        ----

        ## [12] Non-local Latent Relation Distillation for Self-Adaptive 3D Human Pose Estimation

        **Authors**: *Jogendra Nath Kundu, Siddharth Seth, Anirudh Jamkhandi, Pradyumna YM, Varun Jampani, Anirban Chakraborty, Venkatesh Babu R.*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/018b59ce1fd616d874afad0f44ba338d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/018b59ce1fd616d874afad0f44ba338d-Abstract.html)

        **Abstract**:

        Available 3D human pose estimation approaches leverage different forms of strong (2D/3D pose) or weak (multi-view or depth) paired supervision. Barring synthetic or in-studio domains, acquiring such supervision for each new target environment is highly inconvenient. To this end, we cast 3D pose learning as a self-supervised adaptation problem that aims to transfer the task knowledge from a labeled source domain to a completely unpaired target. We propose to infer image-to-pose via two explicit mappings viz. image-to-latent and latent-to-pose where the latter is a pre-learned decoder obtained from a prior-enforcing generative adversarial auto-encoder. Next, we introduce relation distillation as a means to align the unpaired cross-modal samples i.e., the unpaired target videos and unpaired 3D pose sequences. To this end, we propose a new set of non-local relations in order to characterize long-range latent pose interactions, unlike general contrastive relations where positive couplings are limited to a local neighborhood structure. Further, we provide an objective way to quantify non-localness in order to select the most effective relation set. We evaluate different self-adaptation settings and demonstrate state-of-the-art 3D human pose estimation performance on standard benchmarks.

        ----

        ## [13] Fast Training of Neural Lumigraph Representations using Meta Learning

        **Authors**: *Alexander W. Bergman, Petr Kellnhofer, Gordon Wetzstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01931a6925d3de09e5f87419d9d55055-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01931a6925d3de09e5f87419d9d55055-Abstract.html)

        **Abstract**:

        Novel view synthesis is a long-standing problem in machine learning and computer vision. Significant progress has recently been made in developing neural scene representations and rendering techniques that synthesize photorealistic images from arbitrary views. These representations, however, are extremely slow to train and often also slow to render. Inspired by neural variants of image-based rendering, we develop a new neural rendering approach with the goal of quickly learning a high-quality representation which can also be rendered in real-time. Our approach, MetaNLR++, accomplishes this by using a unique combination of a neural shape representation and 2D CNN-based image feature extraction, aggregation, and re-projection. To push representation convergence times down to minutes, we leverage meta learning to learn neural shape and image feature priors which accelerate training. The optimized shape and image features can then be extracted using traditional graphics techniques and rendered in real time. We show that MetaNLR++ achieves similar or better novel view synthesis results in a fraction of the time that competing methods require.

        ----

        ## [14] Analytical Study of Momentum-Based Acceleration Methods in Paradigmatic High-Dimensional Non-Convex Problems

        **Authors**: *Stefano Sarao Mannelli, Pierfrancesco Urbani*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/019f8b946a256d9357eadc5ace2c8678-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/019f8b946a256d9357eadc5ace2c8678-Abstract.html)

        **Abstract**:

        The optimization step in many machine learning problems rarely relies on vanilla gradient descent but it is common practice to use momentum-based accelerated methods. Despite these algorithms being widely applied to arbitrary loss functions, their behaviour in generically non-convex, high dimensional landscapes is poorly understood.In this work, we use dynamical mean field theory techniques to describe analytically the average dynamics of these methods in a prototypical non-convex model: the (spiked) matrix-tensor model. We derive a closed set of equations that describe the behaviour of heavy-ball momentum and Nesterov acceleration in the infinite dimensional limit. By numerical integration of these equations, we observe that these methods speed up the dynamics but do not improve the algorithmic threshold with respect to gradient descent in the spiked model.

        ----

        ## [15] Multimodal Few-Shot Learning with Frozen Language Models

        **Authors**: *Maria Tsimpoukelli, Jacob Menick, Serkan Cabi, S. M. Ali Eslami, Oriol Vinyals, Felix Hill*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01b7575c38dac42f3cfb7d500438b875-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01b7575c38dac42f3cfb7d500438b875-Abstract.html)

        **Abstract**:

        When trained at sufficient scale, auto-regressive language models exhibit the notable ability to learn a new language task after being prompted with just a few examples. Here, we present a simple, yet effective, approach for transferring this few-shot learning ability to a multimodal setting (vision and language). Using aligned image and caption data, we train a vision encoder to represent each image as a sequence of continuous embeddings, such that a pre-trained, frozen language model presented with this prefix generates the appropriate caption. The resulting system is a multimodal few-shot learner, with the surprising ability to learn a variety of new tasks when conditioned on examples, represented as a sequence of any number of interleaved image and text embeddings. We demonstrate that it can rapidly learn words for new objects and novel visual categories, do visual question-answering with only a handful of examples, and make use of outside knowledge, by measuring a single model on a variety of established and new benchmarks.

        ----

        ## [16] Approximating the Permanent with Deep Rejection Sampling

        **Authors**: *Juha Harviainen, Antti Röyskö, Mikko Koivisto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01d8bae291b1e4724443375634ccfa0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01d8bae291b1e4724443375634ccfa0e-Abstract.html)

        **Abstract**:

        We present a randomized approximation scheme for the permanent of a matrix with nonnegative entries. Our scheme extends a recursive rejection sampling method of Huber and Law (SODA 2008) by replacing the permanent upper bound with a linear combination of the subproblem bounds at a moderately large depth of the recursion tree. This method, we call deep rejection sampling, is empirically shown to outperform the basic, depth-zero variant, as well as a related method by Kuck et al. (NeurIPS 2019).  We analyze the expected running time of the scheme on random $(0, 1)$-matrices where each entry is independently $1$ with probability $p$. Our bound is superior to a previous one for $p$ less than $1/5$, matching another bound that was only known to hold when every row and column has density exactly $p$.

        ----

        ## [17] Revisiting Model Stitching to Compare Neural Representations

        **Authors**: *Yamini Bansal, Preetum Nakkiran, Boaz Barak*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01ded4259d101feb739b06c399e9cd9c-Abstract.html)

        **Abstract**:

        We revisit and extend model stitching (Lenc & Vedaldi 2015) as a methodology to study the internal representations of neural networks. Given two trained and frozen models $A$ and $B$, we consider a "stitched model" formed by connecting the bottom-layers of $A$ to the top-layers of $B$, with a simple trainable layer between them.  We argue that model stitching is a powerful and perhaps under-appreciated tool, which reveals aspects of representations that measures such as centered kernel alignment (CKA) cannot. Through extensive experiments, we use model stitching to obtain quantitative verifications for intuitive statements such as "good networks learn similar representations", by demonstrating that good networks of the same architecture, but trained in very different ways (eg: supervised vs. self-supervised learning), can be stitched to each other without drop in performance. We also give evidence for the intuition that "more is better" by showing that representations learnt with (1) more data, (2) bigger width, or (3) more training time can be "plugged in" to weaker models to improve performance. Finally, our experiments reveal a new structural property of SGD which we call "stitching connectivity", akin to mode-connectivity: typical minima reached by SGD are all "stitching-connected" to each other.

        ----

        ## [18] AugMax: Adversarial Composition of Random Augmentations for Robust Training

        **Authors**: *Haotao Wang, Chaowei Xiao, Jean Kossaifi, Zhiding Yu, Anima Anandkumar, Zhangyang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/01e9565cecc4e989123f9620c1d09c09-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/01e9565cecc4e989123f9620c1d09c09-Abstract.html)

        **Abstract**:

        Data augmentation is a simple yet effective way to improve the robustness of deep neural networks (DNNs). Diversity and hardness are two complementary dimensions of data augmentation to achieve robustness. For example, AugMix explores random compositions of a diverse set of augmentations to enhance broader coverage, while adversarial training generates adversarially hard samples to spot the weakness. Motivated by this, we propose a data augmentation framework, termed AugMax, to unify the two aspects of diversity and hardness. AugMax first randomly samples multiple augmentation operators and then learns an adversarial mixture of the selected operators. Being a stronger form of data augmentation, AugMax leads to a significantly augmented input distribution which makes model training more challenging. To solve this problem, we further design a disentangled normalization module, termed DuBIN (Dual-Batch-and-Instance Normalization), that disentangles the instance-wise feature heterogeneity arising from AugMax. Experiments show that AugMax-DuBIN leads to significantly improved out-of-distribution robustness, outperforming prior arts by 3.03%, 3.49%, 1.82% and 0.71% on CIFAR10-C, CIFAR100-C, Tiny ImageNet-C and ImageNet-C. Codes and pretrained models are available: https://github.com/VITA-Group/AugMax.

        ----

        ## [19] Habitat 20: Training Home Assistants to Rearrange their Habitat

        **Authors**: *Andrew Szot, Alexander Clegg, Eric Undersander, Erik Wijmans, Yili Zhao, John M. Turner, Noah Maestre, Mustafa Mukadam, Devendra Singh Chaplot, Oleksandr Maksymets, Aaron Gokaslan, Vladimir Vondrus, Sameer Dharur, Franziska Meier, Wojciech Galuba, Angel X. Chang, Zsolt Kira, Vladlen Koltun, Jitendra Malik, Manolis Savva, Dhruv Batra*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html)

        **Abstract**:

        We introduce Habitat 2.0 (H2.0), a simulation platform for training virtual robots in interactive 3D environments and complex physics-enabled scenarios. We make comprehensive contributions to all levels of the embodied AI stack – data, simulation, and benchmark tasks. Specifically, we present: (i) ReplicaCAD: an artist-authored, annotated, reconfigurable 3D dataset of apartments (matching real spaces) with articulated objects (e.g. cabinets and drawers that can open/close); (ii) H2.0: a high-performance physics-enabled 3D simulator with speeds exceeding 25,000 simulation steps per second (850x real-time) on an 8-GPU node, representing 100x speed-ups over prior work; and, (iii) Home Assistant Benchmark (HAB): a suite of common tasks for assistive robots (tidy the house, stock groceries, set the table) that test a range of mobile manipulation capabilities. These large-scale engineering contributions allow us to systematically compare deep reinforcement learning (RL) at scale and classical sense-plan-act (SPA) pipelines in long-horizon structured tasks, with an emphasis on generalization to new objects, receptacles, and layouts. We find that (1) flat RL policies struggle on HAB compared to hierarchical ones; (2) a hierarchy with independent skills suffers from ‘hand-off problems’, and (3) SPA pipelines are more brittle than RL policies.

        ----

        ## [20] Time Discretization-Invariant Safe Action Repetition for Policy Gradient Methods

        **Authors**: *Seohong Park, Jaekyeom Kim, Gunhee Kim*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/024677efb8e4aee2eaeef17b54695bbe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/024677efb8e4aee2eaeef17b54695bbe-Abstract.html)

        **Abstract**:

        In reinforcement learning, continuous time is often discretized by a time scale $\delta$, to which the resulting performance is known to be highly sensitive. In this work, we seek to find a $\delta$-invariant algorithm for policy gradient (PG) methods, which performs well regardless of the value of $\delta$. We first identify the underlying reasons that cause PG methods to fail as $\delta \to 0$, proving that the variance of the PG estimator can diverge to infinity in stochastic environments under a certain assumption of stochasticity. While durative actions or action repetition can be employed to have $\delta$-invariance, previous action repetition methods cannot immediately react to unexpected situations in stochastic environments. We thus propose a novel $\delta$-invariant method named Safe Action Repetition (SAR) applicable to any existing PG algorithm. SAR can handle the stochasticity of environments by adaptively reacting to changes in states during action repetition. We empirically show that our method is not only $\delta$-invariant but also robust to stochasticity, outperforming previous $\delta$-invariant approaches on eight MuJoCo environments with both deterministic and stochastic settings. Our code is available at https://vision.snu.ac.kr/projects/sar.

        ----

        ## [21] Meta-Learning Reliable Priors in the Function Space

        **Authors**: *Jonas Rothfuss, Dominique Heyn, Jinfan Chen, Andreas Krause*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/024d2d699e6c1a82c9ba986386f4d824-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/024d2d699e6c1a82c9ba986386f4d824-Abstract.html)

        **Abstract**:

        Meta-Learning promises to enable more data-efficient inference by harnessing previous experience from related learning tasks. While existing meta-learning methods help us to improve the accuracy of our predictions in face of data scarcity, they fail to supply reliable uncertainty estimates, often being grossly overconfident in their predictions. Addressing these shortcomings, we introduce a novel meta-learning framework, called F-PACOH, that treats meta-learned priors as stochastic processes and performs meta-level regularization directly in the function space. This allows us to directly steer the probabilistic predictions of the meta-learner towards high epistemic uncertainty in regions of insufficient meta-training data and, thus, obtain well-calibrated uncertainty estimates. Finally, we showcase how our approach can be integrated with sequential decision making, where reliable uncertainty quantification is imperative. In our benchmark study on meta-learning for Bayesian Optimization (BO), F-PACOH significantly outperforms all other meta-learners and standard baselines.  Even in a challenging lifelong BO setting, where optimization tasks arrive one at a time and the meta-learner needs to build up informative prior knowledge incrementally, our proposed method demonstrates strong positive transfer.

        ----

        ## [22] VoiceMixer: Adversarial Voice Style Mixup

        **Authors**: *Sang-Hoon Lee, Ji-Hoon Kim, Hyunseung Chung, Seong-Whan Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0266e33d3f546cb5436a10798e657d97-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0266e33d3f546cb5436a10798e657d97-Abstract.html)

        **Abstract**:

        Although recent advances in voice conversion have shown significant improvement, there still remains a gap between the converted voice and target voice. A key factor that maintains this gap is the insufficient decomposition of content and voice style from the source speech. This insufficiency leads to the converted speech containing source speech style or losing source speech content. In this paper, we present VoiceMixer which can effectively decompose and transfer voice style through a novel information bottleneck and adversarial feedback. With self-supervised representation learning, the proposed information bottleneck can decompose the content and style with only a small loss of content information. Also, for adversarial feedback of each information, the discriminator is decomposed into content and style discriminator with self-supervision, which enable our model to achieve better generalization to the voice style of the converted speech. The experimental results show the superiority of our model in disentanglement and transfer performance, and improve audio quality by preserving content information.

        ----

        ## [23] Predicting What You Already Know Helps: Provable Self-Supervised Learning

        **Authors**: *Jason D. Lee, Qi Lei, Nikunj Saunshi, Jiacheng Zhuo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/02e656adee09f8394b402d9958389b7d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/02e656adee09f8394b402d9958389b7d-Abstract.html)

        **Abstract**:

        Self-supervised representation learning solves auxiliary prediction tasks (known as pretext tasks), that do not require labeled data, to learn semantic representations. These pretext tasks are created solely using the input features, such as predicting a missing image patch, recovering the color channels of an image from context, or predicting missing words, yet predicting this \textit{known} information helps in learning representations effective for downstream prediction tasks. This paper posits a mechanism based on approximate conditional independence to formalize how solving certain pretext tasks can learn representations that provably decrease the sample complexity of downstream supervised tasks. Formally, we quantify how the approximate independence between the components of the pretext task (conditional on the label and latent variables) allows us to learn representations that can solve the downstream task with drastically reduced sample complexity by just training a linear layer on top of the learned representation.

        ----

        ## [24] Oracle Complexity in Nonsmooth Nonconvex Optimization

        **Authors**: *Guy Kornowski, Ohad Shamir*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/030e65da2b1c944090548d36b244b28d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/030e65da2b1c944090548d36b244b28d-Abstract.html)

        **Abstract**:

        It is well-known that given a smooth, bounded-from-below, and possibly nonconvex function, standard gradient-based methods can find $\epsilon$-stationary points (with gradient norm less than $\epsilon$) in $\mathcal{O}(1/\epsilon^2)$ iterations. However, many important nonconvex optimization problems, such as those associated with training modern neural networks, are inherently not smooth, making these results inapplicable. In this paper, we study nonsmooth nonconvex optimization from an oracle complexity viewpoint, where the algorithm is assumed to be given access only to local information about the function at various points. We provide two main results (under mild assumptions): First, we consider the problem of getting \emph{near} $\epsilon$-stationary points. This is perhaps the most natural relaxation of \emph{finding} $\epsilon$-stationary points, which is impossible in the nonsmooth nonconvex case. We prove that this relaxed goal cannot be achieved efficiently, for any distance and $\epsilon$ smaller than some constants. Our second result deals with the possibility of tackling nonsmooth nonconvex optimization by reduction to smooth optimization: Namely, applying smooth optimization methods on a smooth approximation of the objective function. For this approach, we prove an inherent trade-off between oracle complexity and smoothness: On the one hand, smoothing a nonsmooth nonconvex function can be done very efficiently (e.g., by randomized smoothing), but with dimension-dependent factors in the smoothness parameter, which can strongly affect iteration complexity when plugging into standard smooth optimization methods. On the other hand, these dimension factors can be  eliminated with suitable smoothing methods, but only by making the oracle complexity of the smoothing process exponentially large.

        ----

        ## [25] CentripetalText: An Efficient Text Instance Representation for Scene Text Detection

        **Authors**: *Tao Sheng, Jie Chen, Zhouhui Lian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/03227b950778ab86436ff79fe975b596-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/03227b950778ab86436ff79fe975b596-Abstract.html)

        **Abstract**:

        Scene text detection remains a grand challenge due to the variation in text curvatures, orientations, and aspect ratios. One of the hardest problems in this task is how to represent text instances of arbitrary shapes. Although many methods have been proposed to model irregular texts in a flexible manner, most of them lose simplicity and robustness. Their complicated post-processings and the regression under Dirac delta distribution undermine the detection performance and the generalization ability. In this paper, we propose an efficient text instance representation named CentripetalText (CT), which decomposes text instances into the combination of text kernels and centripetal shifts. Specifically, we utilize the centripetal shifts to implement pixel aggregation, guiding the external text pixels to the internal text kernels. The relaxation operation is integrated into the dense regression for centripetal shifts, allowing the correct prediction in a range instead of a specific value. The convenient reconstruction of text contours and the tolerance of prediction errors in our method guarantee the high detection accuracy and the fast inference speed, respectively. Besides, we shrink our text detector into a proposal generation module, namely CentripetalText Proposal Network (CPN), replacing Segmentation Proposal Network (SPN) in Mask TextSpotter v3 and producing more accurate proposals. To validate the effectiveness of our method, we conduct experiments on several commonly used scene text benchmarks, including both curved and multi-oriented text datasets. For the task of scene text detection, our approach achieves superior or competitive performance compared to other existing methods, e.g., F-measure of 86.3% at 40.0 FPS on Total-Text, F-measure of 86.1% at 34.8 FPS on MSRA-TD500, etc. For the task of end-to-end scene text recognition, our method outperforms Mask TextSpotter v3 by 1.1% in F-measure on Total-Text.

        ----

        ## [26] Learning to Select Exogenous Events for Marked Temporal Point Process

        **Authors**: *Ping Zhang, Rishabh K. Iyer, Ashish Tendulkar, Gaurav Aggarwal, Abir De*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/032abcd424b4312e7087f434ef1c0094-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/032abcd424b4312e7087f434ef1c0094-Abstract.html)

        **Abstract**:

        Marked temporal point processes (MTPPs) have emerged as a powerful modelingtool for a wide variety of applications which are characterized using discreteevents localized in continuous time. In this context, the events are of two typesendogenous events which occur due to the influence of the previous events andexogenous events which occur due to the effect of the externalities. However, inpractice, the events do not come with endogenous or exogenous labels. To thisend, our goal in this paper is to identify the set of exogenous events from a set ofunlabelled events. To do so, we first formulate the parameter estimation problemin conjunction with exogenous event set selection problem and show that thisproblem is NP hard. Next, we prove that the underlying objective is a monotoneand \alpha-submodular set function, with respect to the candidate set of exogenousevents. Such a characterization subsequently allows us to use a stochastic greedyalgorithm which was originally proposed in~\cite{greedy}for submodular maximization.However, we show that it also admits an approximation guarantee for maximizing\alpha-submodular set function, even when the learning algorithm provides an imperfectestimates of the trained parameters. Finally, our experiments with synthetic andreal data show that our method performs better than the existing approaches builtupon superposition of endogenous and exogenous MTPPs.

        ----

        ## [27] DRIVE: One-bit Distributed Mean Estimation

        **Authors**: *Shay Vargaftik, Ran Ben-Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben-Itzhak, Michael Mitzenmacher*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0397758f8990c1b41b81b43ac389ab9f-Abstract.html)

        **Abstract**:

        We consider the problem where $n$ clients transmit $d$-dimensional real-valued vectors using $d(1+o(1))$ bits each, in a manner that allows the receiver to approximately reconstruct their mean. Such compression problems naturally arise in distributed and federated learning. We provide novel mathematical results and derive computationally efficient algorithms that are more accurate than previous compression techniques.  We evaluate our methods on a collection of distributed and federated learning tasks, using a variety of datasets, and show a consistent improvement over the state of the art.

        ----

        ## [28] Learning Space Partitions for Path Planning

        **Authors**: *Kevin Yang, Tianjun Zhang, Chris Cummins, Brandon Cui, Benoit Steiner, Linnan Wang, Joseph E. Gonzalez, Dan Klein, Yuandong Tian*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/03a3655fff3e9bdea48de9f49e938e32-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/03a3655fff3e9bdea48de9f49e938e32-Abstract.html)

        **Abstract**:

        Path planning, the problem of efficiently discovering high-reward trajectories, often requires optimizing a high-dimensional and multimodal reward function. Popular approaches like CEM and CMA-ES greedily focus on promising regions of the search space and may get trapped in local maxima. DOO and VOOT balance exploration and exploitation, but use space partitioning strategies independent of the reward function to be optimized. Recently, LaMCTS empirically learns to partition the search space in a reward-sensitive manner for black-box optimization. In this paper, we develop a novel formal regret analysis for when and why such an adaptive region partitioning scheme works. We also propose a new path planning method LaP3 which improves the function value estimation within each sub-region, and uses a latent representation of the search space. Empirically, LaP3 outperforms existing path planning methods in 2D navigation tasks, especially in the presence of difficult-to-escape local optima, and shows benefits when plugged into the planning components of model-based RL such as PETS. These gains transfer to highly multimodal real-world tasks, where we outperform strong baselines in compiler phase ordering by up to 39% on average across 9 tasks, and in molecular design by up to 0.4 on properties on a 0-1 scale. Code is available at https://github.com/yangkevin2/neurips2021-lap3.

        ----

        ## [29] Progressive Feature Interaction Search for Deep Sparse Network

        **Authors**: *Chen Gao, Yinfeng Li, Quanming Yao, Depeng Jin, Yong Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/03b2ceb73723f8b53cd533e4fba898ee-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/03b2ceb73723f8b53cd533e4fba898ee-Abstract.html)

        **Abstract**:

        Deep sparse networks (DSNs), of which the crux is exploring the high-order feature interactions, have become the state-of-the-art on the prediction task with high-sparsity features. However, these models suffer from low computation efficiency, including large model size and slow model inference, which largely limits these models' application value. In this work, we approach this problem with neural architecture search by automatically searching the critical component in DSNs, the feature-interaction layer. We propose a distilled search space to cover the desired architectures with fewer parameters. We then develop a progressive search algorithm for efficient search on the space and well capture the order-priority property in sparse prediction tasks. Experiments on three real-world benchmark datasets show promising results of PROFIT in both accuracy and efficiency. Further studies validate the feasibility of our designed search space and search algorithm.

        ----

        ## [30] Local Explanation of Dialogue Response Generation

        **Authors**: *Yi-Lin Tuan, Connor Pryor, Wenhu Chen, Lise Getoor, William Yang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/03b92cd507ff5870df0db7f074728830-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/03b92cd507ff5870df0db7f074728830-Abstract.html)

        **Abstract**:

        In comparison to the interpretation of classification models, the explanation of sequence generation models is also an important problem, however it has seen little attention. In this work, we study model-agnostic explanations of a representative text generation task -- dialogue response generation. Dialog response generation is challenging with its open-ended sentences and multiple acceptable responses. To gain insights into the reasoning process of a generation model, we propose a new method, local explanation of response generation (LERG) that regards the explanations as the mutual interaction of segments in input and output sentences. LERG views the sequence prediction as uncertainty estimation of a human response and then creates explanations by perturbing the input and calculating the certainty change over the human response. We show that LERG adheres to desired properties of explanations for text generation including unbiased approximation, consistency and cause identification. Empirically, our results show that our method consistently improves other widely used methods on proposed automatic- and human- evaluation metrics for this new task by $4.4$-$12.8$\%. Our analysis demonstrates that LERG can extract both explicit and implicit relations between input and output segments.

        ----

        ## [31] Scalable Inference in SDEs by Direct Matching of the Fokker-Planck-Kolmogorov Equation

        **Authors**: *Arno Solin, Ella Tamir, Prakhar Verma*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/03e4d3f831100d4355663f3d425d716b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/03e4d3f831100d4355663f3d425d716b-Abstract.html)

        **Abstract**:

        Simulation-based techniques such as variants of stochastic Runge–Kutta are the de facto approach for inference with stochastic differential equations (SDEs) in machine learning. These methods are general-purpose and used with parametric and non-parametric models, and neural SDEs. Stochastic Runge–Kutta relies on the use of sampling schemes that can be inefficient in high dimensions. We address this issue by revisiting the classical SDE literature and derive direct approximations to the (typically intractable) Fokker–Planck–Kolmogorov equation by matching moments. We show how this workflow is fast, scales to high-dimensional latent spaces, and is applicable to scarce-data applications, where a non-parametric SDE with a driving Gaussian process velocity field specifies the model.

        ----

        ## [32] The Complexity of Bayesian Network Learning: Revisiting the Superstructure

        **Authors**: *Robert Ganian, Viktoriia Korchemna*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/040a99f23e8960763e680041c601acab-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/040a99f23e8960763e680041c601acab-Abstract.html)

        **Abstract**:

        We investigate the parameterized complexity of Bayesian Network Structure Learning (BNSL), a classical problem that has received significant attention in empirical but also purely theoretical studies. We follow up on previous works that have analyzed the complexity of BNSL w.r.t. the so-called superstructure of the input. While known results imply that BNSL is unlikely to be fixed-parameter tractable even when parameterized by the size of a vertex cover in the superstructure, here we show that a different kind of parameterization - notably by the size of a feedback edge set - yields fixed-parameter tractability. We proceed by showing that this result can be strengthened to a localized version of the feedback edge set, and provide corresponding lower bounds that complement previous results to provide a complexity classification of BNSL w.r.t. virtually all well-studied graph parameters.We then analyze how the complexity of BNSL depends on the representation of the input. In particular, while the bulk of past theoretical work on the topic assumed the use of the so-called non-zero representation, here we prove that if an additive representation can be used instead then BNSL becomes fixed-parameter tractable even under significantly milder restrictions to the superstructure, notably when parameterized by the treewidth alone. Last but not least, we show how our results can be extended to the closely related problem of Polytree Learning.

        ----

        ## [33] Fast Tucker Rank Reduction for Non-Negative Tensors Using Mean-Field Approximation

        **Authors**: *Kazu Ghalamkari, Mahito Sugiyama*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/040ca38cefb1d9226d79c05dd25469cb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/040ca38cefb1d9226d79c05dd25469cb-Abstract.html)

        **Abstract**:

        We present an efficient low-rank approximation algorithm for non-negative tensors. The algorithm is derived from our two findings: First,  we show that rank-1 approximation for tensors can be viewed as a mean-field approximation by treating each tensor as a probability distribution. Second, we theoretically provide a sufficient condition for distribution parameters to reduce Tucker ranks of tensors; interestingly, this sufficient condition can be achieved by iterative application of the mean-field approximation. Since the mean-field approximation is always given as a closed formula, our findings lead to a fast low-rank approximation algorithm without using a gradient method. We empirically demonstrate that our algorithm is faster than the existing non-negative Tucker rank reduction methods and achieves competitive or better approximation of given tensors.

        ----

        ## [34] Learning Stochastic Majority Votes by Minimizing a PAC-Bayes Generalization Bound

        **Authors**: *Valentina Zantedeschi, Paul Viallard, Emilie Morvant, Rémi Emonet, Amaury Habrard, Pascal Germain, Benjamin Guedj*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0415740eaa4d9decbc8da001d3fd805f-Abstract.html)

        **Abstract**:

        We investigate a stochastic counterpart of majority votes over finite ensembles of classifiers, and study its generalization properties. While our approach holds for arbitrary distributions, we instantiate it with Dirichlet distributions: this allows for a closed-form and differentiable expression for the expected risk, which then turns the generalization bound into a tractable training objective.The resulting stochastic majority vote learning algorithm achieves state-of-the-art accuracy and benefits from (non-vacuous) tight generalization bounds, in a series of numerical experiments when compared to competing algorithms which also minimize PAC-Bayes objectives -- both with uninformed (data-independent) and informed (data-dependent) priors.

        ----

        ## [35] Numerical influence of ReLU'(0) on backpropagation

        **Authors**: *David Bertoin, Jérôme Bolte, Sébastien Gerchinovitz, Edouard Pauwels*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/043ab21fc5a1607b381ac3896176dac6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/043ab21fc5a1607b381ac3896176dac6-Abstract.html)

        **Abstract**:

        In theory, the choice of ReLU(0) in [0, 1] for a neural network has a negligible influence both on backpropagation and training. Yet, in the real world, 32 bits default precision combined with the size of deep learning problems makes it a hyperparameter of training methods. We investigate the importance of the value of ReLU'(0) for several precision levels (16, 32, 64 bits), on various networks (fully connected, VGG, ResNet) and datasets (MNIST, CIFAR10, SVHN, ImageNet). We observe considerable variations of backpropagation outputs which occur around half of the time in 32 bits precision. The effect disappears with double precision, while it is systematic at 16 bits. For vanilla SGD training, the choice ReLU'(0) = 0 seems to be the most efficient. For our experiments on ImageNet the gain in test accuracy over ReLU'(0) = 1 was more than 10 points (two runs). We also evidence that reconditioning approaches as batch-norm or ADAM tend to buffer the influence of ReLU'(0)’s value. Overall, the message we convey is that algorithmic differentiation of nonsmooth problems potentially hides parameters that could be tuned advantageously.

        ----

        ## [36] A Contrastive Learning Approach for Training Variational Autoencoder Priors

        **Authors**: *Jyoti Aneja, Alexander G. Schwing, Jan Kautz, Arash Vahdat*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0496604c1d80f66fbeb963c12e570a26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0496604c1d80f66fbeb963c12e570a26-Abstract.html)

        **Abstract**:

        Variational autoencoders (VAEs) are one of the powerful likelihood-based generative models with applications in many domains. However, they struggle to generate high-quality images, especially when samples are obtained from the prior without any tempering. One explanation for VAEs' poor generative quality is the prior hole problem: the prior distribution fails to match the aggregate approximate posterior. Due to this mismatch, there exist areas in the latent space with high density under the prior that do not correspond to any encoded image. Samples from those areas are decoded to corrupted images. To tackle this issue, we propose an energy-based prior defined by the product of a base prior distribution and a reweighting factor, designed to bring the base closer to the aggregate posterior. We train the reweighting factor by noise contrastive estimation, and we generalize it to hierarchical VAEs with many latent variable groups. Our experiments confirm that the proposed noise contrastive priors improve the generative performance of state-of-the-art VAEs by a large margin on the MNIST, CIFAR-10, CelebA 64, and CelebA HQ 256 datasets. Our method is simple and can be applied to a wide variety of VAEs to improve the expressivity of their prior distribution.

        ----

        ## [37] What training reveals about neural network complexity

        **Authors**: *Andreas Loukas, Marinos Poiitis, Stefanie Jegelka*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/04a1bf2d968f1ce381cf1f9184a807a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/04a1bf2d968f1ce381cf1f9184a807a9-Abstract.html)

        **Abstract**:

        This work explores the Benevolent Training Hypothesis (BTH) which argues that the complexity of the function a deep neural network (NN) is learning can be deduced by its training dynamics. Our analysis provides evidence for BTH by relating the NN's Lipschitz constant at different regions of the input space with the behavior of the stochastic training procedure.  We first observe that the Lipschitz constant close to the training data affects various aspects of the parameter trajectory, with more complex networks having a longer trajectory, bigger variance, and often veering further from their initialization. We then show that NNs whose 1st layer bias is trained more steadily (i.e., slowly and with little variation) have bounded complexity even in regions of the input space that are far from any training point. Finally, we find that steady training with Dropout implies a training- and data-dependent generalization bound that grows poly-logarithmically with the number of parameters. Overall, our results support the intuition that good training behavior can be a useful bias towards good generalization.

        ----

        ## [38] Class-agnostic Reconstruction of Dynamic Objects from Videos

        **Authors**: *Zhongzheng Ren, Xiaoming Zhao, Alexander G. Schwing*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/04da4aea8e38ac933ab23cb2389dddef-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/04da4aea8e38ac933ab23cb2389dddef-Abstract.html)

        **Abstract**:

        We introduce REDO, a class-agnostic framework to REconstruct the Dynamic Objects from RGBD or calibrated videos. Compared to prior work, our problem setting is more realistic yet more challenging for three reasons: 1) due to occlusion or camera settings an object of interest may never be entirely visible, but we aim to reconstruct the complete shape; 2) we aim to handle different object dynamics including rigid motion, non-rigid motion, and articulation; 3) we aim to reconstruct different  categories  of  objects  with  one  unified  framework. To  address  these challenges, we develop two novel modules.  First, we introduce a canonical 4D implicit function which is pixel-aligned with aggregated temporal visual cues. Second, we develop a 4D transformation module which captures object dynamics to support temporal propagation and aggregation. We study the efficacy of REDO in extensive experiments on synthetic RGBD video datasets SAIL-VOS 3D and DeformingThings4D++,  and on real-world video data 3DPW. We find REDO outperforms state-of-the-art dynamic reconstruction methods by a margin. In ablation studies we validate each developed component.

        ----

        ## [39] Unique sparse decomposition of low rank matrices

        **Authors**: *Dian Jin, Xin Bing, Yuqian Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/051928341be67dcba03f0e04104d9047-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/051928341be67dcba03f0e04104d9047-Abstract.html)

        **Abstract**:

        The problem of finding the unique low dimensional decomposition of a given matrix has been a fundamental and recurrent problem in many areas. In this paper, we study the problem of seeking a unique decomposition of a low-rank matrix $Y\in \mathbb{R}^{p\times n}$ that admits a sparse representation. Specifically, we consider $ Y =  AX\in  \mathbb{R}^{p\times n}$ where the matrix $A\in  \mathbb{R}^{p\times r}$ has full column rank, with $r < \min\{n,p\}$, and the matrix $X\in  \mathbb{R}^{r\times n}$ is element-wise sparse.  We prove that this sparse decomposition of $Y$ can be uniquely identified by recovering ground-truth $A$ column by column, up to some intrinsic signed permutation. Our approach relies on solving a nonconvex optimization problem constrained over the unit sphere. Our geometric analysis for the nonconvex optimization landscape shows that any {\em strict} local solution is close to the ground truth solution, and can be recovered by a simple data-driven initialization followed with any second-order descent algorithm. At last, we corroborate these theoretical results with numerical experiments

        ----

        ## [40] Neighborhood Reconstructing Autoencoders

        **Authors**: *Yonghyeon Lee, Hyeokjun Kwon, Frank C. Park*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05311655a15b75fab86956663e1819cd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05311655a15b75fab86956663e1819cd-Abstract.html)

        **Abstract**:

        Vanilla autoencoders often produce manifolds that overfit to noisy training data, or have the wrong local connectivity and geometry. Autoencoder regularization techniques, e.g., the denoising autoencoder, have had some success in reducing overfitting, whereas recent graph-based methods that exploit local connectivity information provided by neighborhood graphs have had some success in mitigating local connectivity errors. Neither of these two approaches satisfactorily reduce both overfitting and connectivity errors; moreover, graph-based methods typically involve considerable preprocessing and tuning. To simultaneously address the two issues of overfitting and local connectivity, we propose a new graph-based autoencoder, the Neighborhood Reconstructing Autoencoder (NRAE). Unlike existing graph-based methods that attempt to encode the training data to some prescribed latent space distribution -- one consequence being that only the encoder is the object of the regularization -- NRAE merges local connectivity information contained in the neighborhood graphs with local quadratic approximations of the decoder function to formulate a new neighborhood reconstruction loss. Compared to existing graph-based methods, our new loss function is simple and easy to implement, and the resulting algorithm is scalable and computationally efficient; the only required preprocessing step is the construction of the neighborhood graph. Extensive experiments with standard datasets demonstrate that, compared to existing methods, NRAE improves both overfitting and local connectivity in the learned manifold, in some cases by significant margins. Code for NRAE is available at https://github.com/Gabe-YHLee/NRAE-public.

        ----

        ## [41] TopicNet: Semantic Graph-Guided Topic Discovery

        **Authors**: *Zhibin Duan, Yishi Xu, Bo Chen, Dongsheng Wang, Chaojie Wang, Mingyuan Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0537fb40a68c18da59a35c2bfe1ca554-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0537fb40a68c18da59a35c2bfe1ca554-Abstract.html)

        **Abstract**:

        Existing deep hierarchical topic models are able to extract semantically meaningful topics from a text corpus  in an unsupervised manner and automatically organize them into a topic hierarchy.  However, it is unclear how to incorporate prior belief such as knowledge graph to guide the learning of the topic hierarchy. To address this issue, we introduce TopicNet as a deep hierarchical topic model that can inject prior structural knowledge as inductive bias to influence the learning. TopicNet represents each topic as a Gaussian-distributed embedding vector, projects the topics of all layers into a shared embedding space, and explores both the symmetric and asymmetric similarities between Gaussian embedding vectors to incorporate prior semantic hierarchies. With a variational auto-encoding inference network,  the model parameters are optimized by minimizing the evidence lower bound and supervised loss via stochastic gradient descent. Experiments on widely used benchmark show that TopicNet outperforms related deep topic models on discovering deeper interpretable topics and mining better document representations.

        ----

        ## [42] (Almost) Free Incentivized Exploration from Decentralized Learning Agents

        **Authors**: *Chengshuai Shi, Haifeng Xu, Wei Xiong, Cong Shen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/054ab897023645cd7ad69525c46992a0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/054ab897023645cd7ad69525c46992a0-Abstract.html)

        **Abstract**:

        Incentivized exploration in multi-armed bandits (MAB) has witnessed increasing interests and many progresses in recent years, where a principal offers bonuses to agents to do explorations on her behalf. However, almost all existing studies are confined to temporary myopic agents. In this work, we break this barrier and study incentivized exploration with multiple and long-term strategic agents, who have more complicated behaviors that often appear in real-world applications. An important observation of this work is that strategic agents' intrinsic needs of learning benefit (instead of harming) the principal's explorations by providing "free pulls". Moreover, it turns out that increasing the population of agents significantly lowers the principal's burden of incentivizing. The key and somewhat surprising insight revealed from our results is that when there are sufficiently many learning agents involved, the exploration process of the principal can be (almost) free. Our main results are built upon three novel components which may be of independent interest: (1) a simple yet provably effective incentive-provision strategy; (2) a carefully crafted best arm identification algorithm for rewards aggregated under unequal confidences; (3) a high-probability finite-time lower bound of UCB algorithms. Experimental results are provided to complement the theoretical analysis.

        ----

        ## [43] Combining Recurrent, Convolutional, and Continuous-time Models with Linear State Space Layers

        **Authors**: *Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, Christopher Ré*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05546b0e38ab9175cd905eebcc6ebb76-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05546b0e38ab9175cd905eebcc6ebb76-Abstract.html)

        **Abstract**:

        Recurrent neural networks (RNNs), temporal convolutions, and neural differential equations (NDEs) are popular families of deep learning models for time-series data, each with unique strengths and tradeoffs in modeling power and computational efficiency.  We introduce a simple sequence model inspired by control systems that generalizes these approaches while addressing their shortcomings.  The Linear State-Space Layer (LSSL) maps a sequence $u \mapsto y$ by simply simulating a linear continuous-time state-space representation $\dot{x} = Ax + Bu, y = Cx + Du$.  Theoretically, we show that LSSL models are closely related to the three aforementioned families of models and inherit their strengths.  For example, they generalize convolutions to continuous-time, explain common RNN heuristics, and share features of NDEs such as time-scale adaptation.  We then incorporate and generalize recent theory on continuous-time memorization to introduce a trainable subset of structured matrices $A$ that endow LSSLs with long-range memory.  Empirically, stacking LSSL layers into a simple deep neural network obtains state-of-the-art results across time series benchmarks for long dependencies in sequential image classification, real-world healthcare regression tasks, and speech.  On a difficult speech classification task with length-16000 sequences, LSSL outperforms prior approaches by 24 accuracy points, and even outperforms baselines that use hand-crafted features on 100x shorter sequences.

        ----

        ## [44] Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness

        **Authors**: *Zifeng Wang, Tong Jian, Aria Masoomi, Stratis Ioannidis, Jennifer G. Dy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/055e31fa43e652cb4ab6c0ee845c8d36-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/055e31fa43e652cb4ab6c0ee845c8d36-Abstract.html)

        **Abstract**:

        We investigate the HSIC (Hilbert-Schmidt independence criterion) bottleneck as a regularizer for learning an adversarially robust deep neural network classifier. In addition to the usual cross-entropy loss, we add regularization terms for every intermediate layer to ensure that the latent representations retain useful information for output prediction while reducing redundant information. We show that the HSIC bottleneck enhances robustness to adversarial attacks both theoretically and experimentally. In particular, we prove that the HSIC bottleneck regularizer reduces the sensitivity of the classifier to adversarial examples. Our experiments on multiple benchmark datasets and architectures demonstrate that incorporating an HSIC bottleneck regularizer attains competitive natural accuracy and improves adversarial robustness, both with and without adversarial examples during training. Our code and adversarially robust models are publicly available.

        ----

        ## [45] T-LoHo: A Bayesian Regularization Model for Structured Sparsity and Smoothness on Graphs

        **Authors**: *Changwoo J. Lee, Zhao Tang Luo, Huiyan Sang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05a70454516ecd9194c293b0e415777f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05a70454516ecd9194c293b0e415777f-Abstract.html)

        **Abstract**:

        Graphs have been commonly used to represent complex data structures. In models dealing with graph-structured data, multivariate parameters may not only exhibit sparse patterns but have structured sparsity and smoothness in the sense that both zero and non-zero parameters tend to cluster together. We propose a new prior for high-dimensional parameters with graphical relations, referred to as the Tree-based Low-rank Horseshoe (T-LoHo) model, that generalizes the popular univariate Bayesian horseshoe shrinkage prior to the multivariate setting to detect structured sparsity and smoothness simultaneously. The T-LoHo prior can be embedded in many high-dimensional hierarchical models. To illustrate its utility, we apply it to regularize a Bayesian high-dimensional regression problem where the regression coefficients are linked by a graph, so that the resulting clusters have flexible shapes and satisfy the cluster contiguity constraint with respect to the graph. We design an efficient Markov chain Monte Carlo algorithm that delivers full Bayesian inference with uncertainty measures for model parameters such as the number of clusters. We offer theoretical investigations of the clustering effects and posterior concentration results. Finally, we illustrate the performance of the model with simulation studies and a real data application for anomaly detection on a road network. The results indicate substantial improvements over other competing methods such as the sparse fused lasso.

        ----

        ## [46] The Utility of Explainable AI in Ad Hoc Human-Machine Teaming

        **Authors**: *Rohan R. Paleja, Muyleng Ghuy, Nadun Ranawaka Arachchige, Reed Jensen, Matthew C. Gombolay*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05d74c48b5b30514d8e9bd60320fc8f6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05d74c48b5b30514d8e9bd60320fc8f6-Abstract.html)

        **Abstract**:

        Recent advances in machine learning have led to growing interest in Explainable AI (xAI) to enable humans to gain insight into the decision-making of machine learning models. Despite this recent interest, the utility of xAI techniques has not yet been characterized in human-machine teaming. Importantly, xAI offers the promise of enhancing team situational awareness (SA) and shared mental model development, which are the key characteristics of effective human-machine teams. Rapidly developing such mental models is especially critical in ad hoc human-machine teaming, where agents do not have a priori knowledge of others' decision-making strategies. In this paper, we present two novel human-subject experiments quantifying the benefits of deploying xAI techniques within a human-machine teaming scenario. First, we show that xAI techniques can support SA ($p<0.05)$. Second, we examine how different SA levels induced via a collaborative AI policy abstraction affect ad hoc human-machine teaming performance. Importantly, we find that the benefits of xAI are not universal, as there is a strong dependence on the composition of the human-machine team. Novices benefit from xAI providing increased SA ($p<0.05$) but are susceptible to cognitive overhead ($p<0.05$). On the other hand, expert performance degrades with the addition of xAI-based support ($p<0.05$), indicating that the cost of paying attention to the xAI outweighs the benefits obtained from being provided additional information to enhance SA. Our results demonstrate that researchers must deliberately design and deploy the right xAI techniques in the right scenario by carefully considering human-machine team composition and how the xAI method augments SA.

        ----

        ## [47] Subgoal Search For Complex Reasoning Tasks

        **Authors**: *Konrad Czechowski, Tomasz Odrzygózdz, Marek Zbysinski, Michal Zawalski, Krzysztof Olejnik, Yuhuai Wu, Lukasz Kucinski, Piotr Milos*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05d8cccb5f47e5072f0a05b5f514941a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05d8cccb5f47e5072f0a05b5f514941a-Abstract.html)

        **Abstract**:

        Humans excel in solving complex reasoning tasks through a mental process of moving from one idea to a related one. Inspired by this, we propose Subgoal Search (kSubS) method. Its key component is a learned subgoal generator that produces a diversity of subgoals that are both achievable and closer to the solution. Using subgoals reduces the search space and induces a high-level search graph suitable for efficient planning. In this paper, we implement kSubS using a transformer-based subgoal module coupled with the classical best-first search framework. We show that a simple approach of generating $k$-th step ahead subgoals is surprisingly efficient on three challenging domains: two popular puzzle games, Sokoban and the Rubik's Cube, and an inequality proving benchmark INT. kSubS achieves strong results including state-of-the-art on INT within a modest computational budget.

        ----

        ## [48] MCMC Variational Inference via Uncorrected Hamiltonian Annealing

        **Authors**: *Tomas Geffner, Justin Domke*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/05f971b5ec196b8c65b75d2ef8267331-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/05f971b5ec196b8c65b75d2ef8267331-Abstract.html)

        **Abstract**:

        Given an unnormalized target distribution we want to obtain approximate samples from it and a tight lower bound on its (log) normalization constant log Z. Annealed Importance Sampling (AIS) with Hamiltonian MCMC is a powerful method that can be used to do this. Its main drawback is that it uses non-differentiable transition kernels, which makes tuning its many parameters hard. We propose a framework to use an AIS-like procedure with Uncorrected Hamiltonian MCMC, called Uncorrected Hamiltonian Annealing. Our method leads to tight and differentiable lower bounds on log Z. We show empirically that our method yields better performances than other competing approaches, and that the ability to tune its parameters using reparameterization gradients may lead to large performance improvements.

        ----

        ## [49] Landmark-RxR: Solving Vision-and-Language Navigation with Fine-Grained Alignment Supervision

        **Authors**: *Keji He, Yan Huang, Qi Wu, Jianhua Yang, Dong An, Shuanglin Sima, Liang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0602940f23884f782058efac46f64b0f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0602940f23884f782058efac46f64b0f-Abstract.html)

        **Abstract**:

        In Vision-and-Language Navigation (VLN) task, an agent is asked to navigate inside 3D indoor environments following given instructions. Cross-modal alignment is one of the most critical challenges in VLN because the predicted trajectory needs to match the given instruction accurately. In this paper, we address the cross-modal alignment challenge from the perspective of fine-grain. Firstly, to alleviate weak cross-modal alignment supervision from coarse-grained data, we introduce a human-annotated fine-grained VLN dataset, namely Landmark-RxR. Secondly, to further enhance local cross-modal alignment under fine-grained supervision, we investigate the focal-oriented rewards with soft and hard forms, by focusing on the critical points sampled from fine-grained Landmark-RxR. Moreover, to fully evaluate the navigation process, we also propose a re-initialization mechanism that makes metrics insensitive to difficult points, which can cause the agent to deviate from the correct trajectories. Experimental results show that our agent has superior navigation performance on Landmark-RxR, en-RxR and R2R. Our dataset and code are available at https://github.com/hekj/Landmark-RxR.

        ----

        ## [50] A Winning Hand: Compressing Deep Networks Can Improve Out-of-Distribution Robustness

        **Authors**: *James Diffenderfer, Brian R. Bartoldson, Shreya Chaganti, Jize Zhang, Bhavya Kailkhura*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0607f4c705595b911a4f3e7a127b44e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0607f4c705595b911a4f3e7a127b44e0-Abstract.html)

        **Abstract**:

        Successful adoption of deep learning (DL) in the wild requires models to be: (1) compact, (2) accurate, and (3) robust to distributional shifts. Unfortunately, efforts towards simultaneously meeting these requirements have mostly been unsuccessful. This raises an important question: Is the inability to create Compact, Accurate, and Robust Deep neural networks (CARDs) fundamental? To answer this question, we perform a large-scale analysis of popular model compression techniques which uncovers several intriguing patterns. Notably, in contrast to traditional pruning approaches (e.g., fine tuning and gradual magnitude pruning), we find that ``lottery ticket-style'' approaches can surprisingly be used to produce CARDs, including binary-weight CARDs. Specifically, we are able to create extremely compact CARDs that, compared to their larger counterparts, have similar test accuracy and matching (or better) robustness---simply by pruning and (optionally) quantizing. Leveraging the compactness of CARDs, we develop a simple domain-adaptive test-time ensembling approach (CARD-Decks) that uses a gating module to dynamically select appropriate CARDs from the CARD-Deck based on their spectral-similarity with test samples. The proposed approach builds a "winning hand'' of CARDs that establishes a new state-of-the-art (on RobustBench) on CIFAR-10-C accuracies (i.e., 96.8% standard and 92.75% robust) and CIFAR-100-C accuracies (80.6% standard and 71.3% robust) with better memory usage than non-compressed baselines (pretrained CARDs and CARD-Decks available at https://github.com/RobustBench/robustbench). Finally, we provide theoretical support for our empirical findings.

        ----

        ## [51] On the Importance of Gradients for Detecting Distributional Shifts in the Wild

        **Authors**: *Rui Huang, Andrew Geng, Yixuan Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/063e26c670d07bb7c4d30e6fc69fe056-Abstract.html)

        **Abstract**:

        Detecting out-of-distribution (OOD) data has become a critical component in ensuring the safe deployment of machine learning models in the real world. Existing OOD detection approaches primarily rely on the output or feature space for deriving OOD scores, while largely overlooking information from the gradient space. In this paper, we present GradNorm, a simple and effective approach for detecting OOD inputs by utilizing information extracted from the gradient space. GradNorm directly employs the vector norm of gradients, backpropagated from the KL divergence between the softmax output and a uniform probability distribution. Our key idea is that the magnitude of gradients is higher for in-distribution (ID) data than that for OOD data, making it informative for OOD detection. GradNorm demonstrates superior performance, reducing the average FPR95 by up to 16.33% compared to the previous best method.

        ----

        ## [52] Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods

        **Authors**: *Terrance Liu, Giuseppe Vietri, Steven Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0678c572b0d5597d2d4a6b5bd135754c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0678c572b0d5597d2d4a6b5bd135754c-Abstract.html)

        **Abstract**:

        We study private synthetic data generation for query release, where the goal is to construct a sanitized version of a sensitive dataset, subject to differential privacy, that approximately preserves the answers to a large collection of statistical queries. We first present an algorithmic framework that unifies a long line of iterative algorithms in the literature. Under this framework, we propose two new methods. The first method, private entropy projection (PEP), can be viewed as an advanced variant of MWEM that adaptively reuses past query measurements to boost accuracy. Our second method, generative networks with the exponential mechanism (GEM), circumvents computational bottlenecks in algorithms such as MWEM and PEP by optimizing over generative models parameterized by neural networks, which capture a rich family of distributions while enabling fast gradient-based optimization. We demonstrate that PEP and GEM empirically outperform existing algorithms. Furthermore, we show that GEM nicely incorporates prior information from public data while overcoming limitations of PMW^Pub, the existing state-of-the-art method that also leverages public data.

        ----

        ## [53] Understanding End-to-End Model-Based Reinforcement Learning Methods as Implicit Parameterization

        **Authors**: *Clement Gehring, Kenji Kawaguchi, Jiaoyang Huang, Leslie Pack Kaelbling*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/067a26d87265ea39030f5bd82408ce7c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/067a26d87265ea39030f5bd82408ce7c-Abstract.html)

        **Abstract**:

        Estimating the per-state expected cumulative rewards is a critical aspect of reinforcement learning approaches, however the experience is obtained, but standard deep neural-network function-approximation methods are often inefficient in this setting. An alternative approach, exemplified by value iteration networks, is to learn transition and reward models of a latent Markov decision process whose value predictions fit the data. This approach has been shown empirically to converge faster to a more robust solution in many cases, but there has been little theoretical study of this phenomenon. In this paper, we explore such implicit representations of value functions via theory and focused experimentation. We prove that, for a linear parametrization, gradient descent converges to global optima despite non-linearity and non-convexity introduced by the implicit representation. Furthermore, we derive convergence rates for both cases which allow us to identify conditions under which stochastic gradient descent (SGD) with this implicit representation converges substantially faster than its explicit counterpart. Finally, we provide empirical results in some simple domains that illustrate the theoretical findings.

        ----

        ## [54] Mirror Langevin Monte Carlo: the Case Under Isoperimetry

        **Authors**: *Qijia Jiang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/069090145d54bf4aa3894133f7e89873-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/069090145d54bf4aa3894133f7e89873-Abstract.html)

        **Abstract**:

        Motivated by the connection between sampling and optimization, we study a mirror descent analogue of Langevin dynamics and analyze three different discretization schemes, giving nonasymptotic convergence rate under functional inequalities such as Log-Sobolev in the corresponding metric. Compared to the Euclidean setting, the result reveals intricate relationship between the underlying geometry and the target distribution and suggests that care might need to be taken in order for the discretized algorithm to achieve vanishing bias with diminishing stepsize for sampling from potentials under weaker smoothness/convexity regularity conditions.

        ----

        ## [55] Do Different Tracking Tasks Require Different Appearance Models?

        **Authors**: *Zhongdao Wang, Hengshuang Zhao, Ya-Li Li, Shengjin Wang, Philip H. S. Torr, Luca Bertinetto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06997f04a7db92466a2baa6ebc8b872d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06997f04a7db92466a2baa6ebc8b872d-Abstract.html)

        **Abstract**:

        Tracking objects of interest in a video is one of the most popular and widely applicable problems in computer vision. However, with the years, a Cambrian explosion of use cases and benchmarks has fragmented the problem in a multitude of different experimental setups. As a consequence, the literature has fragmented too, and now novel approaches proposed by the community are usually specialised to fit only one specific setup. To understand to what extent this specialisation is necessary, in this work we present UniTrack, a solution to address five different tasks within the same framework. UniTrack consists of a single and task-agnostic appearance model, which can be learned in a supervised or self-supervised fashion, and multiple ``heads'' that address individual tasks and do not require training. We show how most tracking tasks can be solved within this framework, and that the same appearance model can be successfully used to obtain results that are competitive against specialised methods for most of the tasks considered. The framework also allows us to analyse appearance models obtained with the most recent self-supervised methods, thus extending their evaluation and comparison to a larger variety of important problems.

        ----

        ## [56] Towards robust vision by multi-task learning on monkey visual cortex

        **Authors**: *Shahd Safarani, Arne Nix, Konstantin Willeke, Santiago A. Cadena, Kelli Restivo, George H. Denfield, Andreas S. Tolias, Fabian H. Sinz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06a9d51e04213572ef0720dd27a84792-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06a9d51e04213572ef0720dd27a84792-Abstract.html)

        **Abstract**:

        Deep neural networks set the state-of-the-art across many tasks in computer vision, but their generalization ability to simple image distortions is surprisingly fragile. In contrast, the mammalian visual system is robust to a wide range of perturbations. Recent work suggests that this generalization ability can be explained by useful inductive biases encoded in the representations of visual stimuli throughout the visual cortex. Here, we successfully leveraged these inductive biases with a multi-task learning approach: we jointly trained a deep network to perform image classification and to predict neural activity in macaque primary visual cortex (V1) in response to the same natural stimuli. We measured the out-of-distribution generalization abilities of our resulting network by testing its robustness to common image distortions. We found that co-training on monkey V1 data indeed leads to increased robustness despite the absence of those distortions during training. Additionally, we showed that our network's robustness is often very close to that of an Oracle network where parts of the architecture are directly trained on noisy images. Our results also demonstrated that the network's representations become more brain-like as their robustness improves. Using a novel constrained reconstruction analysis, we investigated what makes our brain-regularized network more robust. We found that our monkey co-trained network is more sensitive to content than noise when compared to a Baseline network that we trained for image classification alone. Using DeepGaze-predicted saliency maps for ImageNet images, we found that the monkey co-trained network tends to be more sensitive to salient regions in a scene, reminiscent of existing theories on the role of V1 in the detection of object borders and bottom-up saliency. Overall, our work expands the promising research avenue of transferring inductive biases from biological to artificial neural networks on the representational level, and provides a novel analysis of the effects of our transfer.

        ----

        ## [57] Arbitrary Conditional Distributions with Energy

        **Authors**: *Ryan R. Strauss, Junier B. Oliva*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06c284d3f757b15c02f47f3ff06dc275-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06c284d3f757b15c02f47f3ff06dc275-Abstract.html)

        **Abstract**:

        Modeling distributions of covariates, or density estimation, is a core challenge in unsupervised learning. However, the majority of work only considers the joint distribution, which has limited relevance to practical situations. A more general and useful problem is arbitrary conditional density estimation, which aims to model any possible conditional distribution over a set of covariates, reflecting the more realistic setting of inference based on prior knowledge. We propose a novel method, Arbitrary Conditioning with Energy (ACE), that can simultaneously estimate the distribution $p(\mathbf{x}_u \mid \mathbf{x}_o)$ for all possible subsets of unobserved features $\mathbf{x}_u$ and observed features $\mathbf{x}_o$. ACE is designed to avoid unnecessary bias and complexity --- we specify densities with a highly expressive energy function and reduce the problem to only learning one-dimensional conditionals (from which more complex distributions can be recovered during inference). This results in an approach that is both simpler and higher-performing than prior methods. We show that ACE achieves state-of-the-art for arbitrary conditional likelihood estimation and data imputation on standard benchmarks.

        ----

        ## [58] Learning Domain Invariant Representations in Goal-conditioned Block MDPs

        **Authors**: *Beining Han, Chongyi Zheng, Harris Chan, Keiran Paster, Michael R. Zhang, Jimmy Ba*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06d172404821f7d01060cc9629171b2e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06d172404821f7d01060cc9629171b2e-Abstract.html)

        **Abstract**:

        Deep Reinforcement Learning (RL) is successful in solving many complex Markov Decision Processes (MDPs) problems. However, agents often face unanticipated environmental changes after deployment in the real world. These changes are often spurious and unrelated to the underlying problem, such as background shifts for visual input agents. Unfortunately, deep RL policies are usually sensitive to these changes and fail to act robustly against them. This resembles the problem of domain generalization in supervised learning. In this work, we study this problem for goal-conditioned RL agents. We propose a theoretical framework in the Block MDP setting that characterizes the generalizability of goal-conditioned policies to new environments. Under this framework, we develop a practical method PA-SkewFit that enhances domain generalization. The empirical evaluation shows that our goal-conditioned RL agent can perform well in various unseen test environments, improving by 50\% over baselines.

        ----

        ## [59] Near-Optimal Multi-Perturbation Experimental Design for Causal Structure Learning

        **Authors**: *Scott Sussex, Caroline Uhler, Andreas Krause*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06d5ae105ea1bea4d800bc96491876e9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06d5ae105ea1bea4d800bc96491876e9-Abstract.html)

        **Abstract**:

        Causal structure learning is a key problem in many domains. Causal structures can be learnt by performing experiments on the system of interest. We address the largely unexplored problem of designing a batch of experiments that each simultaneously intervene on multiple variables. While potentially more informative than the commonly considered single-variable interventions, selecting such interventions is algorithmically much more challenging, due to the doubly-exponential combinatorial search space over sets of composite interventions. In this paper, we develop efficient algorithms for optimizing different objective functions quantifying the informativeness of a budget-constrained batch of experiments. By establishing novel submodularity properties of these objectives, we provide approximation guarantees for our algorithms. Our algorithms empirically perform superior to both random interventions and algorithms that only select single-variable interventions.

        ----

        ## [60] Fuzzy Clustering with Similarity Queries

        **Authors**: *Wasim Huleihel, Arya Mazumdar, Soumyabrata Pal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06f2e099b4f87109d52e15d7c05f0084-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06f2e099b4f87109d52e15d7c05f0084-Abstract.html)

        **Abstract**:

        The fuzzy or soft $k$-means objective is a popular generalization of the well-known $k$-means problem, extending the clustering capability of the $k$-means to datasets that are uncertain, vague and otherwise hard to cluster. In this paper, we propose a semi-supervised active clustering framework, where the learner is allowed to interact with an oracle (domain expert), asking for the similarity between a certain set of chosen items. We study the query and computational complexities of clustering in this framework. We prove that having a few of such similarity queries enables one to get a polynomial-time approximation algorithm to an otherwise conjecturally NP-hard problem. In particular, we provide algorithms for fuzzy clustering in this setting that ask $O(\mathsf{poly}(k)\log n)$ similarity queries and run with polynomial-time-complexity, where $n$ is the number of items. The fuzzy $k$-means objective is nonconvex, with $k$-means as a special case, and is equivalent to some other generic nonconvex problem such as non-negative matrix factorization. The ubiquitous Lloyd-type algorithms (or alternating-minimization algorithms) can get stuck at a local minima. Our results show that by making few similarity queries, the problem becomes easier to solve. Finally, we test our algorithms over real-world datasets, showing their effectiveness in real-world applications.

        ----

        ## [61] Improving black-box optimization in VAE latent space using decoder uncertainty

        **Authors**: *Pascal Notin, José Miguel Hernández-Lobato, Yarin Gal*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/06fe1c234519f6812fc4c1baae25d6af-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/06fe1c234519f6812fc4c1baae25d6af-Abstract.html)

        **Abstract**:

        Optimization in the latent space of variational autoencoders is a promising approach to generate high-dimensional discrete objects that maximize an expensive black-box property (e.g., drug-likeness in molecular generation, function approximation with arithmetic expressions). However, existing methods lack robustness as they may decide to explore areas of the latent space for which no data was available during training and where the decoder can be unreliable, leading to the generation of unrealistic or invalid objects. We propose to leverage the epistemic uncertainty of the decoder to guide the optimization process. This is not trivial though, as a naive estimation of uncertainty in the high-dimensional and structured settings we consider would result in high estimator variance. To solve this problem, we introduce an importance sampling-based estimator that provides more robust estimates of epistemic uncertainty. Our uncertainty-guided optimization approach does not require modifications of the model architecture nor the training process. It produces samples with a better trade-off between black-box objective and validity of the generated samples, sometimes improving both simultaneously. We illustrate these advantages across several experimental settings in digit generation, arithmetic expression approximation and molecule generation for drug design.

        ----

        ## [62] Sample Selection for Fair and Robust Training

        **Authors**: *Yuji Roh, Kangwook Lee, Steven Whang, Changho Suh*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html)

        **Abstract**:

        Fairness and robustness are critical elements of Trustworthy AI that need to be addressed together. Fairness is about learning an unbiased model while robustness is about learning from corrupted data, and it is known that addressing only one of them may have an adverse affect on the other. In this work, we propose a sample selection-based algorithm for fair and robust training. To this end, we formulate a combinatorial optimization problem for the unbiased selection of samples in the presence of data corruption. Observing that solving this optimization problem is strongly NP-hard, we propose a greedy algorithm that is efficient and effective in practice. Experiments show that our method obtains fairness and robustness that are better than or comparable to the state-of-the-art technique, both on synthetic and benchmark real datasets. Moreover, unlike other fair and robust training baselines, our algorithm can be used by only modifying the sampling step in batch selection without changing the training algorithm or leveraging additional clean data.

        ----

        ## [63] NeurWIN: Neural Whittle Index Network For Restless Bandits Via Deep RL

        **Authors**: *Khaled Nakhleh, Santosh Ganji, Ping-Chun Hsieh, I-Hong Hou, Srinivas Shakkottai*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0768281a05da9f27df178b5c39a51263-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0768281a05da9f27df178b5c39a51263-Abstract.html)

        **Abstract**:

        Whittle index policy is a powerful tool to obtain asymptotically optimal solutions for the notoriously intractable problem of restless bandits. However, finding the Whittle indices remains a difficult problem for many practical restless bandits with convoluted transition kernels. This paper proposes NeurWIN, a neural Whittle index network that seeks to learn the Whittle indices for any restless bandits by leveraging mathematical properties of the Whittle indices. We show that a neural network that produces the Whittle index is also one that produces the optimal control for a set of Markov decision problems. This property motivates using deep reinforcement learning for the training of NeurWIN. We demonstrate the utility of NeurWIN by evaluating its performance for three recently studied restless bandit problems.Our experiment results show that the performance of NeurWIN is significantly better than other RL algorithms.

        ----

        ## [64] Sageflow: Robust Federated Learning against Both Stragglers and Adversaries

        **Authors**: *Jungwuk Park, Dong-Jun Han, Minseok Choi, Jaekyun Moon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/076a8133735eb5d7552dc195b125a454-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/076a8133735eb5d7552dc195b125a454-Abstract.html)

        **Abstract**:

        While federated learning (FL) allows efficient model training with local data at edge devices, among major issues still to be resolved are: slow devices known as stragglers and malicious attacks launched by adversaries.   While the presence of both of these issues raises serious concerns in practical FL systems, no known schemes or combinations of schemes effectively address them at the same time. We propose Sageflow, staleness-aware grouping with entropy-based filtering and loss-weighted averaging, to handle both stragglers and adversaries simultaneously. Model grouping and weighting according to staleness (arrival delay) provides robustness against stragglers, while entropy-based filtering and loss-weighted averaging, working in a highly complementary fashion at each grouping stage,  counter a wide range of adversary attacks. A theoretical bound is established to provide key insights into the convergence behavior of Sageflow. Extensive experimental results show that Sageflow outperforms various existing methods aiming to handle stragglers/adversaries.

        ----

        ## [65] Alias-Free Generative Adversarial Networks

        **Authors**: *Tero Karras, Miika Aittala, Samuli Laine, Erik Härkönen, Janne Hellsten, Jaakko Lehtinen, Timo Aila*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/076ccd93ad68be51f23707988e934906-Abstract.html)

        **Abstract**:

        We observe that despite their hierarchical convolutional nature, the synthesis process of typical generative adversarial networks depends on absolute pixel coordinates in an unhealthy manner. This manifests itself as, e.g., detail appearing to be glued to image coordinates instead of the surfaces of depicted objects. We trace the root cause to careless signal processing that causes aliasing in the generator network. Interpreting all signals in the network as continuous, we derive generally applicable, small architectural changes that guarantee that unwanted information cannot leak into the hierarchical synthesis process. The resulting networks match the FID of StyleGAN2 but differ dramatically in their internal representations, and they are fully equivariant to translation and rotation even at subpixel scales. Our results pave the way for generative models better suited for video and animation.

        ----

        ## [66] Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images

        **Authors**: *Kwanyoung Kim, Jong Chul Ye*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/077b83af57538aa183971a2fe0971ec1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/077b83af57538aa183971a2fe0971ec1-Abstract.html)

        **Abstract**:

        Recently, there has  been extensive research interest in training  deep networks to denoise images without clean reference.However, the representative approaches such as Noise2Noise, Noise2Void, Stein's unbiased risk estimator (SURE), etc.  seem to differ from one another and it is difficult to find the coherent mathematical structure. To address this, here we present a novel approach, called Noise2Score, which reveals a missing link in order to unite these seemingly different approaches.Specifically, we  show that   image denoising  problems  without clean images can be addressed by finding the mode of the posterior distribution and that the Tweedie's formula offers an explicit solution through the score function (i.e. the gradient of loglikelihood). Our method then uses the  recent finding that  the score function  can be stably estimated from the noisy images using the amortized residual denoising autoencoder, the method of which is closely related to Noise2Noise or Nose2Void. Our Noise2Score approach is so universal  that the same network training can be used to remove noises from images that are corrupted by any exponential family distributions and noise parameters. Using extensive  experiments with Gaussian, Poisson, and Gamma noises, we show  that  Noise2Score significantly outperforms the state-of-the-art self-supervised denoising methods in the benchmark data set such as (C)BSD68, Set12, and Kodak, etc.

        ----

        ## [67] Continuous Mean-Covariance Bandits

        **Authors**: *Yihan Du, Siwei Wang, Zhixuan Fang, Longbo Huang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07811dc6c422334ce36a09ff5cd6fe71-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07811dc6c422334ce36a09ff5cd6fe71-Abstract.html)

        **Abstract**:

        Existing risk-aware multi-armed bandit models typically focus on risk measures of individual options such as variance. As a result, they cannot be directly applied to important real-world online decision making problems with correlated options. In this paper, we propose a novel Continuous Mean-Covariance Bandit (CMCB) model to explicitly take into account option correlation. Specifically, in CMCB, there is a learner who sequentially chooses weight vectors on given options and observes random feedback according to the decisions. The agent's  objective is to  achieve the best trade-off between reward and risk, measured with option covariance. To capture different reward observation scenarios in practice, we consider three feedback settings, i.e., full-information, semi-bandit and full-bandit feedback. We propose novel algorithms with optimal regrets (within logarithmic factors), and provide matching lower bounds to validate their optimalities. The experimental results also demonstrate the superiority of our algorithms.  To the best of our knowledge, this is the first work that considers option correlation in risk-aware bandits and explicitly quantifies how arbitrary covariance structures impact the learning performance.The novel analytical techniques we developed for exploiting the estimated covariance to build concentration and bounding the risk of selected actions based on sampling strategy properties can likely find applications in other bandit analysis and be of independent interests.

        ----

        ## [68] Dynamic Visual Reasoning by Learning Differentiable Physics Models from Video and Language

        **Authors**: *Mingyu Ding, Zhenfang Chen, Tao Du, Ping Luo, Josh Tenenbaum, Chuang Gan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07845cd9aefa6cde3f8926d25138a3a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07845cd9aefa6cde3f8926d25138a3a2-Abstract.html)

        **Abstract**:

        In this work, we propose a unified framework, called Visual Reasoning with Differ-entiable Physics (VRDP), that can jointly learn visual concepts and infer physics models of objects and their interactions from videos and language. This is achieved by seamlessly integrating three components: a visual perception module, a concept learner, and a differentiable physics engine. The visual perception module parses each video frame into object-centric trajectories and represents them as latent scene representations. The concept learner grounds visual concepts (e.g., color, shape, and material) from these object-centric representations based on the language, thus providing prior knowledge for the physics engine. The differentiable physics model, implemented as an impulse-based differentiable rigid-body simulator, performs differentiable physical simulation based on the grounded concepts to infer physical properties, such as mass, restitution, and velocity, by fitting the simulated trajectories into the video observations. Consequently, these learned concepts and physical models can explain what we have seen and imagine what is about to happen in future and counterfactual scenarios. Integrating differentiable physics into the dynamic reasoning framework offers several appealing benefits.  More accurate dynamics prediction in learned physics models enables state-of-the-art performance on both synthetic and real-world benchmarks while still maintaining high transparency and interpretability; most notably, VRDP improves the accuracy of predictive and counterfactual questions by 4.5% and 11.5% compared to its best counterpart. VRDP is also highly data-efficient: physical parameters can be optimized from very few videos, and even a single video can be sufficient. Finally, with all physical parameters inferred, VRDP can quickly learn new concepts from a few examples.

        ----

        ## [69] Solving Soft Clustering Ensemble via $k$-Sparse Discrete Wasserstein Barycenter

        **Authors**: *Ruizhe Qin, Mengying Li, Hu Ding*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07a4e20a7bbeeb7a736682b26b16ebe8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07a4e20a7bbeeb7a736682b26b16ebe8-Abstract.html)

        **Abstract**:

        Clustering ensemble is one of the most important problems in  ensemble learning. Though it has been extensively studied in the past decades, the existing methods often suffer from the issues like high computational complexity and the difficulty on understanding the consensus. In this paper, we study the more general soft clustering ensemble problem where each individual solution is a soft clustering. We connect it to the well-known discrete Wasserstein barycenter problem in geometry. Based on some novel geometric insights in high dimensions, we propose the sampling-based algorithms with provable quality guarantees. We also provide the systematical analysis on the consensus of our model. Finally, we conduct the experiments  to evaluate our proposed algorithms.

        ----

        ## [70] Bayesian Adaptation for Covariate Shift

        **Authors**: *Aurick Zhou, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07ac7cd13fd0eb1654ccdbd222b81437-Abstract.html)

        **Abstract**:

        When faced with distribution shift at test time, deep neural networks often make inaccurate predictions with unreliable uncertainty estimates.While improving the robustness of neural networks is one promising approach to mitigate this issue, an appealing alternate to robustifying networks against all possible test-time shifts is to instead directly adapt them to unlabeled inputs from the particular distribution shift we encounter at test time.However, this poses a challenging question: in the standard Bayesian model for supervised learning, unlabeled inputs are conditionally independent of model parameters when the labels are unobserved, so what can unlabeled data tell us about the model parameters at test-time? In this paper, we derive a Bayesian model that provides for a well-defined relationship between unlabeled inputs under distributional shift and model parameters, and show how approximate inference in this model can be instantiated with a simple regularized entropy minimization procedure at test-time. We evaluate our method on a variety of distribution shifts for image classification, including image corruptions, natural distribution shifts, and domain adaptation settings, and show that our method improves both accuracy and uncertainty estimation.

        ----

        ## [71] Perturb-and-max-product: Sampling and learning in discrete energy-based models

        **Authors**: *Miguel Lázaro-Gredilla, Antoine Dedieu, Dileep George*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07b1c04a30f798b5506c1ec5acfb9031-Abstract.html)

        **Abstract**:

        Perturb-and-MAP offers an elegant approach to approximately sample from a energy-based model (EBM) by computing the maximum-a-posteriori (MAP) configuration of a perturbed version of the model. Sampling in turn enables learning. However, this line of research has been hindered by the general intractability of the MAP computation. Very few works venture outside tractable models, and when they do, they use linear programming approaches, which as we will show, have several limitations. In this work we present perturb-and-max-product (PMP), a parallel and scalable mechanism for sampling and learning in discrete EBMs. Models can be arbitrary as long as they are built using tractable factors. We show that (a) for Ising models, PMP is orders of magnitude faster than Gibbs and Gibbs-with-Gradients (GWG) at learning and generating samples of similar or better quality; (b) PMP is able to learn and sample from RBMs; (c) in a large, entangled graphical model in which Gibbs and GWG fail to mix, PMP succeeds.

        ----

        ## [72] Towards Unifying Behavioral and Response Diversity for Open-ended Learning in Zero-sum Games

        **Authors**: *Xiangyu Liu, Hangtian Jia, Ying Wen, Yujing Hu, Yingfeng Chen, Changjie Fan, Zhipeng Hu, Yaodong Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07bba581a2dd8d098a3be0f683560643-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07bba581a2dd8d098a3be0f683560643-Abstract.html)

        **Abstract**:

        Measuring and promoting policy diversity is critical for solving games with strong non-transitive dynamics where strategic cycles exist, and there is no consistent winner (e.g., Rock-Paper-Scissors). With that in mind, maintaining a pool of diverse policies via open-ended learning is an attractive solution, which can generate auto-curricula to avoid being exploited. However, in conventional open-ended learning algorithms, there are no widely accepted definitions for diversity, making it hard to construct and evaluate the diverse policies. In this work, we summarize previous concepts of diversity and work towards offering a unified measure of diversity in multi-agent open-ended learning to include all elements in Markov games, based on both Behavioral Diversity (BD) and Response Diversity (RD). At the trajectory distribution level, we re-define BD in the state-action space as the discrepancies of occupancy measures. For the reward dynamics, we propose RD to characterize diversity through the responses of policies when encountering different opponents. We also show that many current diversity measures fall in one of the categories of BD or RD but not both. With this unified diversity measure, we design the corresponding diversity-promoting objective and population effectivity when seeking the best responses in open-ended learning. We validate our methods in both relatively simple games like matrix game, non-transitive mixture model, and the complex \textit{Google Research Football} environment. The population found by our methods reveals the lowest exploitability, highest population effectivity in matrix game and non-transitive mixture model, as well as the largest goal difference when interacting with opponents of various levels in \textit{Google Research Football}.

        ----

        ## [73] Towards Better Understanding of Training Certifiably Robust Models against Adversarial Examples

        **Authors**: *Sungyoon Lee, Woojin Lee, Jinseong Park, Jaewook Lee*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07c5807d0d927dcd0980f86024e5208b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07c5807d0d927dcd0980f86024e5208b-Abstract.html)

        **Abstract**:

        We study the problem of training certifiably robust models against adversarial examples. Certifiable training minimizes an upper bound on the worst-case loss over the allowed perturbation, and thus the tightness of the upper bound is an important factor in building certifiably robust models. However, many studies have shown that Interval Bound Propagation (IBP) training uses much looser bounds but outperforms other models that use tighter bounds. We identify another key factor that influences the performance of certifiable training: \textit{smoothness of the loss landscape}. We find significant differences in the loss landscapes across many linear relaxation-based methods, and that the current state-of-the-arts method often has a landscape with favorable optimization properties. Moreover, to test the claim, we design a new certifiable training method with the desired properties. With the tightness and the smoothness, the proposed method achieves a decent performance under a wide range of perturbations, while others with only one of the two factors can perform well only for a specific range of perturbations. Our code is available at \url{https://github.com/sungyoon-lee/LossLandscapeMatters}.

        ----

        ## [74] Mitigating Covariate Shift in Imitation Learning via Offline Data With Partial Coverage

        **Authors**: *Jonathan D. Chang, Masatoshi Uehara, Dhruv Sreenivas, Rahul Kidambi, Wen Sun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07d5938693cc3903b261e1a3844590ed-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07d5938693cc3903b261e1a3844590ed-Abstract.html)

        **Abstract**:

        This paper studies offline Imitation Learning (IL) where an agent learns to imitate an expert demonstrator without additional online environment interactions. Instead, the learner is presented with a static offline dataset of state-action-next state triples from a potentially less proficient behavior policy. We introduce Model-based IL from Offline data (MILO): an algorithmic framework that utilizes the static dataset to solve the offline IL problem efficiently both in theory and in practice. In theory, even if the behavior policy is highly sub-optimal compared to the expert, we show that as long as the data from the behavior policy provides sufficient coverage on the expert state-action traces (and with no necessity for a global coverage over the entire state-action space), MILO can provably combat the covariate shift issue in IL. Complementing our theory results, we also demonstrate that a practical implementation of our approach mitigates covariate shift on benchmark MuJoCo continuous control tasks. We demonstrate that with behavior policies whose performances are less than half of that of the expert, MILO still successfully imitates with an extremely low number of expert state-action pairs while traditional offline IL methods such as behavior cloning (BC) fail completely. Source code is provided at https://github.com/jdchang1/milo.

        ----

        ## [75] Global Filter Networks for Image Classification

        **Authors**: *Yongming Rao, Wenliang Zhao, Zheng Zhu, Jiwen Lu, Jie Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/07e87c2f4fc7f7c96116d8e2a92790f5-Abstract.html)

        **Abstract**:

        Recent advances in self-attention and pure multi-layer perceptrons (MLP) models for vision have shown great potential in achieving promising performance with fewer inductive biases. These models are generally based on learning interaction among spatial locations from raw data. The complexity of self-attention and MLP grows quadratically as the image size increases, which makes these models hard to scale up when high-resolution features are required. In this paper, we present the Global Filter Network (GFNet), a conceptually simple yet computationally efficient architecture, that learns long-term spatial dependencies in the frequency domain with log-linear complexity. Our architecture replaces the self-attention layer in vision transformers with three key operations: a 2D discrete Fourier transform, an element-wise multiplication between frequency-domain features and learnable global filters, and a 2D inverse Fourier transform. We exhibit favorable accuracy/complexity trade-offs of our models on both ImageNet and downstream tasks. Our results demonstrate that GFNet can be a very competitive alternative to transformer-style models and CNNs in efficiency, generalization ability and robustness. Code is available at https://github.com/raoyongming/GFNet

        ----

        ## [76] Catastrophic Data Leakage in Vertical Federated Learning

        **Authors**: *Xiao Jin, Pin-Yu Chen, Chia-Yi Hsu, Chia-Mu Yu, Tianyi Chen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08040837089cdf46631a10aca5258e16-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08040837089cdf46631a10aca5258e16-Abstract.html)

        **Abstract**:

        Recent studies show that private training data can be leaked through the gradients sharing mechanism deployed in distributed machine learning systems, such as federated learning (FL). Increasing batch size to complicate data recovery is often viewed as a promising defense strategy against data leakage. In this paper, we revisit this defense premise and propose an advanced data leakage attack with theoretical justification to efficiently recover batch data from the shared aggregated gradients. We name our proposed method as catastrophic data leakage in vertical federated learning (CAFE). Comparing to existing data leakage attacks, our extensive experimental results on vertical FL settings demonstrate the effectiveness of CAFE to perform large-batch data leakage attack with improved data recovery quality. We also propose a practical countermeasure to mitigate CAFE. Our results suggest that private data participated in standard FL, especially the vertical case, have a high risk of being leaked from the training gradients. Our analysis implies unprecedented and practical data leakage risks in those learning settings. The code of our work is available at https://github.com/DeRafael/CAFE.

        ----

        ## [77] Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee

        **Authors**: *Flint Xiaofeng Fan, Yining Ma, Zhongxiang Dai, Wei Jing, Cheston Tan, Bryan Kian Hsiang Low*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/080acdcce72c06873a773c4311c2e464-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/080acdcce72c06873a773c4311c2e464-Abstract.html)

        **Abstract**:

        The growing literature of Federated Learning (FL) has recently inspired Federated Reinforcement Learning (FRL) to encourage multiple agents to federatively build a better decision-making policy without sharing raw trajectories. Despite its promising applications, existing works on FRL fail to I) provide theoretical analysis on its convergence, and II) account for random system failures and adversarial attacks. Towards this end, we propose the first FRL framework the convergence of which is guaranteed and tolerant to less than half of the participating agents being random system failures or adversarial attackers. We prove that the sample efficiency of the proposed framework is guaranteed to improve with the number of agents and is able to account for such potential failures or attacks. All theoretical results are empirically verified on various RL benchmark tasks.

        ----

        ## [78] Compacter: Efficient Low-Rank Hypercomplex Adapter Layers

        **Authors**: *Rabeeh Karimi Mahabadi, James Henderson, Sebastian Ruder*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/081be9fdff07f3bc808f935906ef70c0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/081be9fdff07f3bc808f935906ef70c0-Abstract.html)

        **Abstract**:

        Adapting large-scale pretrained language models to downstream tasks via fine-tuning is the standard method for achieving state-of-the-art performance on NLP benchmarks. However, fine-tuning all weights of models with millions or billions of parameters is sample-inefficient, unstable in low-resource settings, and wasteful as it requires storing a separate copy of the model for each task. Recent work has developed parameter-efficient fine-tuning methods,  but these approaches either still require a relatively large number of parameters or underperform standard fine-tuning. In this work, we propose Compacter, a method for fine-tuning large-scale language models with a better trade-off between task performance and the number of trainable parameters than prior work. Compacter accomplishes this by building on top of ideas from adapters, low-rank optimization, and parameterized hypercomplex multiplication layers.Specifically, Compacter inserts task-specific weight matrices into a pretrained model's weights, which are computed efficiently as a sum of Kronecker products between shared slow'' weights andfast'' rank-one matrices defined per Compacter layer. By only training 0.047% of a pretrained model's parameters, Compacter performs on par with standard fine-tuning on GLUE and outperforms standard fine-tuning on SuperGLUE and low-resource settings. Our code is publicly available at https://github.com/rabeehk/compacter.

        ----

        ## [79] Distilling Image Classifiers in Object Detectors

        **Authors**: *Shuxuan Guo, Jose M. Alvarez, Mathieu Salzmann*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/082a8bbf2c357c09f26675f9cf5bcba3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/082a8bbf2c357c09f26675f9cf5bcba3-Abstract.html)

        **Abstract**:

        Knowledge distillation constitutes a simple yet effective way to improve the performance of a compact student network by exploiting the knowledge of a more powerful teacher. Nevertheless, the knowledge distillation literature remains limited to the scenario where the student and the teacher tackle the same task. Here, we investigate the problem of transferring knowledge not only across architectures but also across tasks. To this end, we study the case of object detection and, instead of following the standard detector-to-detector distillation approach, introduce a classifier-to-detector knowledge transfer framework. In particular, we propose strategies to exploit the classification teacher to improve both the detector's recognition accuracy and localization performance. Our experiments on several detectors with different backbones demonstrate the effectiveness of our approach, allowing us to outperform the state-of-the-art detector-to-detector distillation methods.

        ----

        ## [80] Subgroup Generalization and Fairness of Graph Neural Networks

        **Authors**: *Jiaqi Ma, Junwei Deng, Qiaozhu Mei*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08425b881bcde94a383cd258cea331be-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08425b881bcde94a383cd258cea331be-Abstract.html)

        **Abstract**:

        Despite enormous successful applications of graph neural networks (GNNs), theoretical understanding of their generalization ability, especially for node-level tasks where data are not independent and identically-distributed (IID), has been sparse. The theoretical investigation of the generalization performance is beneficial for understanding fundamental issues (such as fairness) of GNN models and designing better learning methods. In this paper, we present a novel PAC-Bayesian analysis for GNNs under a non-IID semi-supervised learning setup. Moreover, we analyze the generalization performances on different subgroups of unlabeled nodes, which allows us to further study an accuracy-(dis)parity-style (un)fairness of GNNs from a theoretical perspective. Under reasonable assumptions, we demonstrate that the distance between a test subgroup and the training set can be a key factor affecting the GNN performance on that subgroup, which calls special attention to the training node selection for fair learning. Experiments across multiple GNN models and datasets support our theoretical results.

        ----

        ## [81] Scaling Neural Tangent Kernels via Sketching and Random Features

        **Authors**: *Amir Zandieh, Insu Han, Haim Avron, Neta Shoham, Chaewon Kim, Jinwoo Shin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08ae6a26b7cb089ea588e94aed36bd15-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08ae6a26b7cb089ea588e94aed36bd15-Abstract.html)

        **Abstract**:

        The Neural Tangent Kernel (NTK) characterizes the behavior of infinitely-wide neural networks trained under least squares loss by gradient descent. Recent works also report that NTK regression can outperform finitely-wide neural networks trained on small-scale datasets. However, the computational complexity of kernel methods has limited its use in large-scale learning tasks. To accelerate learning with NTK, we design a near input-sparsity time approximation algorithm for NTK, by sketching the polynomial expansions of arc-cosine kernels: our sketch for the convolutional counterpart of NTK (CNTK) can transform any image using a linear runtime in the number of pixels. Furthermore, we prove a spectral approximation guarantee for the NTK matrix, by combining random features (based on leverage score sampling) of the arc-cosine kernels with a sketching algorithm. We benchmark our methods on various large-scale regression and classification tasks and show that a linear regressor trained on our CNTK features matches the accuracy of exact CNTK on CIFAR-10 dataset while achieving 150x speedup.

        ----

        ## [82] BatchQuant: Quantized-for-all Architecture Search with Robust Quantizer

        **Authors**: *Haoping Bai, Meng Cao, Ping Huang, Jiulong Shan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08aee6276db142f4b8ac98fb8ee0ed1b-Abstract.html)

        **Abstract**:

        As the applications of deep learning models on edge devices increase at an accelerating pace, fast adaptation to various scenarios with varying resource constraints has become a crucial aspect of model deployment. As a result, model optimization strategies with adaptive configuration are becoming increasingly popular. While single-shot quantized neural architecture search enjoys flexibility in both model architecture and quantization policy, the combined search space comes with many challenges, including instability when training the weight-sharing supernet and difficulty in navigating the exponentially growing search space. Existing methods tend to either limit the architecture search space to a small set of options or limit the quantization policy search space to fixed precision policies. To this end, we propose BatchQuant, a robust quantizer formulation that allows fast and stable training of a compact, single-shot, mixed-precision, weight-sharing supernet. We employ BatchQuant to train a compact supernet (offering over $10^{76}$ quantized subnets) within substantially fewer GPU hours than previous methods. Our approach, Quantized-for-all (QFA), is the first to seamlessly extend one-shot weight-sharing NAS supernet to support subnets with arbitrary ultra-low bitwidth mixed-precision quantization policies without retraining. QFA opens up new possibilities in joint hardware-aware neural architecture search and quantization. We demonstrate the effectiveness of our method on ImageNet and achieve SOTA Top-1 accuracy under a low complexity constraint (<20 MFLOPs).

        ----

        ## [83] Long Short-Term Transformer for Online Action Detection

        **Authors**: *Mingze Xu, Yuanjun Xiong, Hao Chen, Xinyu Li, Wei Xia, Zhuowen Tu, Stefano Soatto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08b255a5d42b89b0585260b6f2360bdd-Abstract.html)

        **Abstract**:

        We present Long Short-term TRansformer (LSTR), a temporal modeling algorithm for online action detection, which employs a long- and short-term memory mechanism to model prolonged sequence data. It consists of an LSTR encoder that dynamically leverages coarse-scale historical information from an extended temporal window (e.g., 2048 frames spanning of up to 8 minutes), together with an LSTR decoder that focuses on a short time window (e.g., 32 frames spanning 8 seconds) to model the fine-scale characteristics of the data. Compared to prior work, LSTR provides an effective and efficient method to model long videos with fewer heuristics, which is validated by extensive empirical analysis. LSTR achieves state-of-the-art performance on three standard online action detection benchmarks, THUMOS'14, TVSeries, and HACS Segment. Code has been made available at: https://xumingze0308.github.io/projects/lstr.

        ----

        ## [84] Near Optimal Policy Optimization via REPS

        **Authors**: *Aldo Pacchiano, Jonathan N. Lee, Peter L. Bartlett, Ofir Nachum*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08d562c1eedd30b15b51e35d8486d14c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08d562c1eedd30b15b51e35d8486d14c-Abstract.html)

        **Abstract**:

        Since its introduction a decade ago, relative entropy policy search (REPS) has demonstrated successful policy learning on a number of simulated and real-world robotic domains, not to mention providing algorithmic components used by many recently proposed reinforcement learning (RL) algorithms. While REPS is commonly known in the community, there exist no guarantees on its performance when using stochastic and gradient-based solvers. In this paper we aim to fill this gap by providing guarantees and convergence rates for the sub-optimality of a policy learned using first-order optimization methods applied to the REPS objective. We first consider the setting in which we are given access to exact gradients and demonstrate how near-optimality of the objective translates to near-optimality of the policy. We then consider the practical setting of stochastic gradients, and introduce a technique that uses generative access to the underlying Markov decision process to compute parameter updates that maintain favorable convergence to the optimal regularized policy.

        ----

        ## [85] Self-Consistent Models and Values

        **Authors**: *Gregory Farquhar, Kate Baumli, Zita Marinho, Angelos Filos, Matteo Hessel, Hado Philip van Hasselt, David Silver*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08f0efebb1c51aada9430a089a2050cc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08f0efebb1c51aada9430a089a2050cc-Abstract.html)

        **Abstract**:

        Learned models of the environment provide reinforcement learning (RL) agents with flexible ways of making predictions about the environment.Models enable planning, i.e. using more computation to improve value functions or policies, without requiring additional environment interactions.In this work, we investigate a way of augmenting model-based RL, by additionally encouraging a learned model and value function to be jointly \emph{self-consistent}.This lies in contrast to classic planning methods like Dyna, which only update the value function to be consistent with the model.We propose a number of possible self-consistency updates, study them empirically in both the tabular and function approximation settings, and find that with appropriate choices self-consistency can be useful both for policy evaluation and control.

        ----

        ## [86] Learning on Random Balls is Sufficient for Estimating (Some) Graph Parameters

        **Authors**: *Takanori Maehara, Hoang NT*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08f36fcf88c0a84c19a6ed437b9cbcc9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08f36fcf88c0a84c19a6ed437b9cbcc9-Abstract.html)

        **Abstract**:

        Theoretical analyses for graph learning methods often assume a complete observation of the input graph. Such an assumption might not be useful for handling any-size graphs due to the scalability issues in practice. In this work, we develop a theoretical framework for graph classification problems in the partial observation setting (i.e., subgraph samplings). Equipped with insights from graph limit theory, we propose a new graph classification model that works on a randomly sampled subgraph and a novel topology to characterize the representability of the model. Our theoretical framework contributes a theoretical validation of mini-batch learning on graphs and leads to new learning-theoretic results on generalization bounds as well as size-generalizability without assumptions on the input.

        ----

        ## [87] Risk-Averse Bayes-Adaptive Reinforcement Learning

        **Authors**: *Marc Rigter, Bruno Lacerda, Nick Hawes*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/08f90c1a417155361a5c4b8d297e0d78-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/08f90c1a417155361a5c4b8d297e0d78-Abstract.html)

        **Abstract**:

        In this work, we address risk-averse Bayes-adaptive reinforcement learning. We pose the problem of optimising the conditional value at risk (CVaR) of the total return in Bayes-adaptive Markov decision processes (MDPs).  We show that a policy optimising CVaR in this setting is risk-averse to both the epistemic uncertainty due to the prior distribution over MDPs, and the aleatoric uncertainty due to the inherent stochasticity of MDPs. We reformulate the problem as a two-player stochastic game and propose an approximate algorithm based on Monte Carlo tree search and Bayesian optimisation. Our experiments demonstrate that our approach significantly outperforms baseline approaches for this problem.

        ----

        ## [88] Iterative Connecting Probability Estimation for Networks

        **Authors**: *Yichen Qin, Linhan Yu, Yang Li*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0919b5c38396c3f0c41f1112d538e42c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0919b5c38396c3f0c41f1112d538e42c-Abstract.html)

        **Abstract**:

        Estimating the probabilities of connections between vertices in a random network using an observed adjacency matrix is an important task for network data analysis. Many existing estimation methods are based on certain assumptions on network structure, which limit their applicability in practice. Without making strong assumptions, we develop an iterative connecting probability estimation method based on neighborhood averaging. Starting at a random initial point or an existing estimate, our method iteratively updates the pairwise vertex distances, the sets of similar vertices, and connecting probabilities to improve the precision of the estimate. We propose a two-stage neighborhood selection procedure to achieve the trade-off between smoothness of the estimate and the ability to discover local structure. The tuning parameters can be selected by cross-validation. We establish desirable theoretical properties for our method, and further justify its superior performance by comparing with existing methods in simulation and real data analysis.

        ----

        ## [89] Learning to Adapt via Latent Domains for Adaptive Semantic Segmentation

        **Authors**: *Yunan Liu, Shanshan Zhang, Yang Li, Jian Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/092cb13c22d51c22b9035a2b4fe76b00-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/092cb13c22d51c22b9035a2b4fe76b00-Abstract.html)

        **Abstract**:

        Domain adaptive semantic segmentation aims to transfer knowledge learned from labeled source domain to unlabeled target domain. To narrow down the domain gap and ease adaptation difficulty, some recent methods translate source images to target-like images (latent domains), which are used as supplement or substitute to the original source data. Nevertheless, these methods neglect to explicitly model the relationship of knowledge transferring across different domains. Alternatively, in this work we break through the standard “source-target” one pair adaptation framework and construct multiple adaptation pairs (e.g. “source-latent” and “latent-target”). The purpose is to use the meta-knowledge (how to adapt) learned from one pair as guidance to assist the adaptation of another pair under a meta-learning framework. Furthermore, we extend our method to a more practical setting of open compound domain adaptation (a.k.a multiple-target domain adaptation), where the target is a compound of multiple domains without domain labels. In this setting, we embed an additional pair of “latent-latent” to reduce the domain gap between the source and different latent domains, allowing the model to adapt well on multiple target domains simultaneously. When evaluated on standard benchmarks, our method is superior to the state-of-the-art methods in both the single target and multiple-target domain adaptation settings.

        ----

        ## [90] Single Layer Predictive Normalized Maximum Likelihood for Out-of-Distribution Detection

        **Authors**: *Koby Bibas, Meir Feder, Tal Hassner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/093b60fd0557804c8ba0cbf1453da22f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/093b60fd0557804c8ba0cbf1453da22f-Abstract.html)

        **Abstract**:

        Detecting out-of-distribution (OOD) samples is vital for developing machine learning based models for critical safety systems. Common approaches for OOD detection assume access to some OOD samples during training which may not be available in a real-life scenario. Instead, we utilize the {\em predictive normalized maximum likelihood} (pNML) learner, in which no assumptions are made on the tested input. We derive an explicit expression of the pNML and its generalization error, denoted as the regret, for a single layer neural network (NN). We show that this learner generalizes well when (i) the test vector resides in a subspace spanned by the eigenvectors associated with the large eigenvalues of the empirical correlation matrix of the training data, or (ii) the test sample is far from the decision boundary. Furthermore, we describe how to efficiently apply the derived pNML regret to any pretrained deep NN, by employing the explicit pNML for the last layer, followed by the softmax function. Applying the derived regret to deep NN requires neither additional tunable parameters nor extra data. We extensively evaluate our approach on 74 OOD detection benchmarks using DenseNet-100, ResNet-34, and WideResNet-40 models trained with CIFAR-100, CIFAR-10, SVHN, and ImageNet-30 showing a significant improvement of up to 15.6% over recent leading methods.

        ----

        ## [91] Prototypical Cross-Attention Networks for Multiple Object Tracking and Segmentation

        **Authors**: *Lei Ke, Xia Li, Martin Danelljan, Yu-Wing Tai, Chi-Keung Tang, Fisher Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html)

        **Abstract**:

        Multiple object tracking and segmentation requires detecting, tracking, and segmenting objects belonging to a set of given classes. Most approaches only exploit the temporal dimension to address the association problem, while relying on single frame predictions for the segmentation mask itself. We propose Prototypical Cross-Attention Network (PCAN), capable of leveraging rich spatio-temporal information for online multiple object tracking and segmentation. PCAN first distills a space-time memory into a set of prototypes and then employs cross-attention to retrieve rich information from the past frames. To segment each object, PCAN adopts a prototypical appearance module to learn a set of contrastive foreground and background prototypes, which are then propagated over time. Extensive experiments demonstrate that PCAN outperforms current video instance tracking and segmentation competition winners on both Youtube-VIS and BDD100K datasets, and shows efficacy to both one-stage and two-stage segmentation frameworks. Code and video resources are available at http://vis.xyz/pub/pcan.

        ----

        ## [92] Algorithmic Instabilities of Accelerated Gradient Descent

        **Authors**: *Amit Attia, Tomer Koren*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/094bb65ef46d3eb4be0a87877ec333eb-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/094bb65ef46d3eb4be0a87877ec333eb-Abstract.html)

        **Abstract**:

        We study the algorithmic stability of Nesterov's accelerated gradient method. For convex quadratic objectives, Chen et al. (2018) proved that the uniform stability of the method grows quadratically with the number of optimization steps, and conjectured that the same is true for the general convex and smooth case. We disprove this conjecture and show, for two notions of algorithmic stability (including uniform stability), that the stability of Nesterov's accelerated method in fact deteriorates exponentially fast with the number of gradient steps. This stands in sharp contrast to the bounds in the quadratic case, but also to known results for non-accelerated gradient methods where stability typically grows linearly with the number of steps.

        ----

        ## [93] Learning Optimal Predictive Checklists

        **Authors**: *Haoran Zhang, Quaid Morris, Berk Ustun, Marzyeh Ghassemi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09676fac73eda6cac726c43e43e86c58-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09676fac73eda6cac726c43e43e86c58-Abstract.html)

        **Abstract**:

        Checklists are simple decision aids that are often used to promote safety and reliability in clinical applications. In this paper, we present a method to learn checklists for clinical decision support. We represent predictive checklists as discrete linear classifiers with binary features and unit weights. We then learn globally optimal predictive checklists from data by solving an integer programming problem. Our method allows users to customize checklists to obey complex constraints, including constraints to enforce group fairness and to binarize real-valued features at training time. In addition, it pairs models with an optimality gap that can inform model development and determine the feasibility of learning sufficiently accurate checklists on a given dataset. We pair our method with specialized techniques that speed up its ability to train a predictive checklist that performs well and has a small optimality gap. We benchmark the performance of our method on seven clinical classification problems, and demonstrate its practical benefits by training a short-form checklist for PTSD screening. Our results show that our method can fit simple predictive checklists that perform well and that can easily be customized to obey a rich class of custom constraints.

        ----

        ## [94] Finite Sample Analysis of Average-Reward TD Learning and $Q$-Learning

        **Authors**: *Sheng Zhang, Zhe Zhang, Siva Theja Maguluri*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/096ffc299200f51751b08da6d865ae95-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/096ffc299200f51751b08da6d865ae95-Abstract.html)

        **Abstract**:

        The focus of this paper is on sample complexity guarantees of average-reward reinforcement learning algorithms, which are known to be more challenging to study than their discounted-reward counterparts. To the best of our knowledge, we provide the first known finite sample guarantees using both constant and diminishing step sizes of (i) average-reward TD($\lambda$) with linear function approximation for policy evaluation and (ii) average-reward $Q$-learning in the tabular setting to find the optimal policy. A major challenge is that since the value functions are agnostic to an additive constant, the corresponding Bellman operators are no longer contraction mappings under any norm. We obtain the results for TD($\lambda$) by working in an appropriately defined subspace that ensures uniqueness of the solution. For $Q$-learning, we exploit the span seminorm contractive property of the Bellman operator, and construct a novel Lyapunov function obtained by infimal convolution of a generalized Moreau envelope and the indicator function of a set.

        ----

        ## [95] Generalization Bounds for Graph Embedding Using Negative Sampling: Linear vs Hyperbolic

        **Authors**: *Atsushi Suzuki, Atsushi Nitanda, Jing Wang, Linchuan Xu, Kenji Yamanishi, Marc Cavazza*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09779bb7930c8a0a44360e12b538ae3c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09779bb7930c8a0a44360e12b538ae3c-Abstract.html)

        **Abstract**:

        Graph embedding, which represents real-world entities in a mathematical space, has enabled numerous applications such as analyzing natural languages, social networks, biochemical networks, and knowledge bases.It has been experimentally shown that graph embedding in hyperbolic space can represent hierarchical tree-like data more effectively than embedding in linear space, owing to hyperbolic space's exponential growth property. However, since the theoretical comparison has been limited to ideal noiseless settings, the potential for the hyperbolic space's property to worsen the generalization error for practical data has not been analyzed.In this paper, we provide a generalization error bound applicable for graph embedding both in linear and hyperbolic spaces under various negative sampling settings that appear in graph embedding. Our bound states that error is polynomial and exponential with respect to the embedding space's radius in linear and hyperbolic spaces, respectively, which implies that hyperbolic space's exponential growth property worsens the error.Using our bound, we clarify the data size condition on which graph embedding in hyperbolic space can represent a tree better than in Euclidean space by discussing the bias-variance trade-off.Our bound also shows that imbalanced data distribution, which often appears in graph embedding, can worsen the error.

        ----

        ## [96] Gradient Starvation: A Learning Proclivity in Neural Networks

        **Authors**: *Mohammad Pezeshki, Sékou-Oumar Kaba, Yoshua Bengio, Aaron C. Courville, Doina Precup, Guillaume Lajoie*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html)

        **Abstract**:

        We identify and formalize a fundamental gradient descent phenomenon resulting in a learning proclivity in over-parameterized neural networks. Gradient Starvation arises when cross-entropy loss is minimized by capturing only a subset of features relevant for the task, despite the presence of other predictive features that fail to be discovered. This work provides a theoretical explanation for the emergence of such feature imbalance in neural networks. Using tools from Dynamical Systems theory, we identify simple properties of learning dynamics during gradient descent that lead to this imbalance, and prove that such a situation can be expected given certain statistical structure in training data. Based on our proposed formalism, we develop guarantees for a novel regularization method aimed at decoupling feature learning dynamics, improving accuracy and robustness in cases hindered by gradient starvation. We illustrate our findings with simple and real-world out-of-distribution (OOD) generalization experiments.

        ----

        ## [97] Offline Reinforcement Learning as One Big Sequence Modeling Problem

        **Authors**: *Michael Janner, Qiyang Li, Sergey Levine*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/099fe6b0b444c23836c4a5d07346082b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/099fe6b0b444c23836c4a5d07346082b-Abstract.html)

        **Abstract**:

        Reinforcement learning (RL) is typically viewed as the problem of estimating single-step policies (for model-free RL) or single-step models (for model-based RL), leveraging the Markov property to factorize the problem in time. However, we can also view RL as a sequence modeling problem: predict a sequence of actions that leads to a sequence of high rewards. Viewed in this way, it is tempting to consider whether powerful, high-capacity sequence prediction models that work well in other supervised learning domains, such as natural-language processing, can also provide simple and effective solutions to the RL problem. To this end, we explore how RL can be reframed as "one big sequence modeling" problem, using state-of-the-art Transformer architectures to model distributions over sequences of states, actions, and rewards. Addressing RL as a sequence modeling problem significantly simplifies a range of design decisions: we no longer require separate behavior policy constraints, as is common in prior work on offline model-free RL, and we no longer require ensembles or other epistemic uncertainty estimators, as is common in prior work on model-based RL. All of these roles are filled by the same Transformer sequence model. In our experiments, we demonstrate the flexibility of this approach across imitation learning, goal-conditioned RL, and offline RL.

        ----

        ## [98] Optimality and Stability in Federated Learning: A Game-theoretic Approach

        **Authors**: *Kate Donahue, Jon M. Kleinberg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09a5e2a11bea20817477e0b1dfe2cc21-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09a5e2a11bea20817477e0b1dfe2cc21-Abstract.html)

        **Abstract**:

        Federated learning is a distributed learning paradigm where multiple agents, each only with access to local data, jointly learn a global model. There has recently been an explosion of research aiming not only to improve the accuracy rates of federated learning, but also provide certain guarantees around social good properties such as total error. One branch of this research has taken a game-theoretic approach, and in particular, prior work has viewed federated learning as a hedonic game, where error-minimizing players arrange themselves into federating coalitions. This past work proves the existence of stable coalition partitions, but leaves open a wide range of questions, including how far from optimal these stable solutions are. In this work, we motivate and define a notion of optimality given by the average error rates among federating agents (players). First, we provide and prove the correctness of an efficient algorithm to calculate an optimal (error minimizing) arrangement of players. Next, we analyze the relationship between the stability and optimality of an arrangement. First, we show that for some regions of parameter space, all stable arrangements are optimal (Price of Anarchy equal to 1). However, we show this is not true for all settings: there exist examples of stable arrangements with higher cost than optimal (Price of Anarchy greater than 1). Finally, we give the first constant-factor bound on the performance gap between stability and optimality, proving that the total error of the worst stable solution can be no higher than 9 times the total error of an optimal solution (Price of Anarchy bound of 9).

        ----

        ## [99] Understanding Deflation Process in Over-parametrized Tensor Decomposition

        **Authors**: *Rong Ge, Yunwei Ren, Xiang Wang, Mo Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09a630e07af043e4cae879dd60db1cac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09a630e07af043e4cae879dd60db1cac-Abstract.html)

        **Abstract**:

        In this paper we study the training dynamics for gradient flow on over-parametrized tensor decomposition problems. Empirically, such training process often first fits larger components and then discovers smaller components, which is similar to a tensor deflation process that is commonly used in tensor decomposition algorithms. We prove that for orthogonally decomposable tensor, a slightly modified version of gradient flow would follow a tensor deflation process and recover all the tensor components. Our proof suggests that for orthogonal tensors, gradient flow dynamics works similarly as greedy low-rank learning in the matrix setting, which is a first step towards understanding the implicit regularization effect of over-parametrized models for low-rank tensors.

        ----

        ## [100] Privately Learning Subspaces

        **Authors**: *Vikrant Singhal, Thomas Steinke*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09b69adcd7cbae914c6204984097d2da-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09b69adcd7cbae914c6204984097d2da-Abstract.html)

        **Abstract**:

        Private data analysis suffers a costly curse of dimensionality. However, the data often has an underlying low-dimensional structure. For example, when optimizing via gradient descent, the gradients often lie in or near a low-dimensional subspace. If that low-dimensional structure can be identified, then we can avoid paying (in terms of privacy or accuracy) for the high ambient dimension. We present differentially private algorithms that take input data sampled from a low-dimensional linear subspace (possibly with a small amount of error) and output that subspace (or an approximation to it). These algorithms can serve as a pre-processing step for other procedures.

        ----

        ## [101] On the Value of Interaction and Function Approximation in Imitation Learning

        **Authors**: *Nived Rajaraman, Yanjun Han, Lin Yang, Jingbo Liu, Jiantao Jiao, Kannan Ramchandran*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09dbc1177211571ef3e1ca961cc39363-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09dbc1177211571ef3e1ca961cc39363-Abstract.html)

        **Abstract**:

        We study the statistical guarantees for the Imitation Learning (IL) problem in episodic MDPs.Rajaraman et al. (2020) show an information theoretic lower bound that in the worst case, a learner which can even actively query the expert policy suffers from a suboptimality growing quadratically in the length of the horizon, $H$. We study imitation learning under the $\mu$-recoverability assumption of Ross et al. (2011) which assumes that the difference in the $Q$-value under the expert policy across different actions in a state do not deviate beyond $\mu$ from the maximum. We show that the reduction proposed by Ross et al. (2010) is statistically optimal: the resulting algorithm upon interacting with the MDP for $N$ episodes results in a suboptimality bound of $\widetilde{\mathcal{O}} \left( \mu |\mathcal{S}| H / N \right)$ which we show is optimal up to log-factors. In contrast, we show that any algorithm which does not interact with the MDP and uses an offline dataset of $N$ expert trajectories must incur suboptimality growing as $\gtrsim |\mathcal{S}| H^2/N$ even under the $\mu$-recoverability assumption. This establishes a clear and provable separation of the minimax rates between the active setting and the no-interaction setting. We also study IL with linear function approximation. When the expert plays actions according to a linear classifier of known state-action features, we use the reduction to multi-class classification to show that with high probability, the suboptimality of behavior cloning is  $\widetilde{O}(dH^2/N)$ given $N$ rollouts from the optimal policy. This is optimal up to log-factors but can be improved to $\widetilde{O}(dH/N)$ if we have a linear expert with parameter-sharing across time steps. In contrast, when the MDP transition structure is known to the learner such as in the case of simulators, we demonstrate fundamental differences compared to the tabular setting in terms of the performance of an optimal algorithm, Mimic-MD (Rajaraman et al. (2020)) when extended to the function approximation setting. Here, we introduce a new problem called confidence set linear classification, that can be used to construct sample-efficient IL algorithms.

        ----

        ## [102] Shapeshifter: a Parameter-efficient Transformer using Factorized Reshaped Matrices

        **Authors**: *Aliakbar Panahi, Seyran Saeedi, Tom Arodz*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09def3ebbc44ff3426b28fcd88c83554-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09def3ebbc44ff3426b28fcd88c83554-Abstract.html)

        **Abstract**:

        Language models employ a very large number of trainable parameters. Despite being highly overparameterized, these networks often achieve good out-of-sample test performance on the original task and easily fine-tune to related tasks. Recent observations involving, for example, intrinsic dimension of the objective landscape and  the lottery ticket hypothesis, indicate that often training actively involves only a small fraction of the parameter space. Thus, a question remains how large a parameter space needs to be in the first place â€“- the evidence from recent work on model compression, parameter sharing, factorized representations, and knowledge distillation increasingly shows that models can be made much smaller and still perform well. Here, we focus on factorized representations of matrices that underpin dense, embedding, and self-attention layers. We use low-rank factorized representation of a reshaped and rearranged original matrix to achieve space efficient and expressive linear layers. We prove that stacking such low-rank layers increases their expressiveness, providing theoretical understanding for their effectiveness in deep networks. In Transformer models, our approach leads to more than ten-fold reduction in the number of total trainable parameters, including embedding, attention, and feed-forward layers, with little degradation in on-task performance. The approach operates out-of-the-box,  replacing each parameter matrix with its compact equivalent while maintaining the architecture of the network.

        ----

        ## [103] The Adaptive Doubly Robust Estimator and a Paradox Concerning Logging Policy

        **Authors**: *Masahiro Kato, Kenichiro McAlinn, Shota Yasui*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/09e7655fc1dc8fa7c9d6c4478313d5e6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/09e7655fc1dc8fa7c9d6c4478313d5e6-Abstract.html)

        **Abstract**:

        The doubly robust (DR) estimator, which consists of two nuisance parameters, the conditional mean outcome and the logging policy (the probability of choosing an action), is crucial in causal inference. This paper proposes a DR estimator for dependent samples obtained from adaptive experiments. To obtain an asymptotically normal semiparametric estimator from dependent samples without non-Donsker nuisance estimators, we propose adaptive-fitting as a variant of sample-splitting. We also report an empirical paradox that our proposed DR estimator tends to show better performances compared to other estimators utilizing the true logging policy. While a similar phenomenon is known for estimators with i.i.d. samples, traditional explanations based on asymptotic efficiency cannot elucidate our case with dependent samples. We confirm this hypothesis through simulation studies.

        ----

        ## [104] Regularized Softmax Deep Multi-Agent Q-Learning

        **Authors**: *Ling Pan, Tabish Rashid, Bei Peng, Longbo Huang, Shimon Whiteson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0a113ef6b61820daa5611c870ed8d5ee-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0a113ef6b61820daa5611c870ed8d5ee-Abstract.html)

        **Abstract**:

        Tackling overestimation in $Q$-learning is an important problem that has been extensively studied in single-agent reinforcement learning, but has received comparatively little attention in the multi-agent setting. In this work, we empirically demonstrate that QMIX, a popular $Q$-learning algorithm for cooperative multi-agent reinforcement learning (MARL), suffers from a more severe overestimation in practice than previously acknowledged, and is not mitigated by existing approaches. We rectify this with a novel regularization-based update scheme that penalizes large joint action-values that deviate from a baseline and demonstrate its effectiveness in stabilizing learning. Furthermore, we propose to employ a softmax operator, which we efficiently approximate in a novel way in the multi-agent setting, to further reduce the potential overestimation bias. Our approach, Regularized Softmax (RES) Deep Multi-Agent $Q$-Learning, is general and can be applied to any $Q$-learning based MARL algorithm. We demonstrate that, when applied to QMIX, RES avoids severe overestimation and significantly improves performance, yielding state-of-the-art results in a variety of cooperative multi-agent tasks, including the challenging StarCraft II micromanagement benchmarks.

        ----

        ## [105] Physics-Aware Downsampling with Deep Learning for Scalable Flood Modeling

        **Authors**: *Niv Giladi, Zvika Ben-Haim, Sella Nevo, Yossi Matias, Daniel Soudry*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0a3b5a7a477d359746061d41c3a04fd6-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0a3b5a7a477d359746061d41c3a04fd6-Abstract.html)

        **Abstract**:

        Background. Floods are the most common natural disaster in the world, affecting the lives of hundreds of millions. Flood forecasting is therefore a vitally important endeavor, typically achieved using physical water flow simulations, which rely on accurate terrain elevation maps. However, such simulations, based on solving partial differential equations, are computationally prohibitive on a large scale. This scalability issue is commonly alleviated using a coarse grid representation of the elevation map, though this representation may distort crucial terrain details, leading to significant inaccuracies in the simulation.\Contributions. We train a deep neural network to perform physics-informed downsampling of the terrain map: we optimize the coarse grid representation of the terrain maps, so that the flood prediction will match the fine grid solution. For the learning process to succeed, we configure a dataset specifically for this task. We demonstrate that with this method, it is possible to achieve a significant reduction in computational cost, while maintaining an accurate solution. A reference implementation accompanies the paper as well as documentation and code for dataset reproduction.

        ----

        ## [106] Systematic Generalization with Edge Transformers

        **Authors**: *Leon Bergen, Timothy J. O'Donnell, Dzmitry Bahdanau*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0a4dc6dae338c9cb08947c07581f77a2-Abstract.html)

        **Abstract**:

        Recent research suggests that systematic generalization in natural language understanding remains a challenge for state-of-the-art neural models such as Transformers and Graph Neural Networks. To tackle this challenge, we propose Edge Transformer, a new model that combines inspiration from Transformers and rule-based symbolic AI. The first key idea in Edge Transformers is to associate vector states with every edge, that is, with every pair of input nodes---as opposed to just every node, as it is done in the Transformer model. The second major innovation is a triangular attention mechanism that updates edge representations in a way that is inspired by unification from logic programming. We evaluate Edge Transformer on compositional generalization benchmarks in relational reasoning, semantic parsing, and dependency parsing. In all three settings, the Edge Transformer outperforms Relation-aware, Universal and classical Transformer baselines.

        ----

        ## [107] TransformerFusion: Monocular RGB Scene Reconstruction using Transformers

        **Authors**: *Aljaz Bozic, Pablo R. Palafox, Justus Thies, Angela Dai, Matthias Nießner*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0a87257e5308197df43230edf4ad1dae-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0a87257e5308197df43230edf4ad1dae-Abstract.html)

        **Abstract**:

        We introduce TransformerFusion, a transformer-based 3D scene reconstruction approach. From an input monocular RGB video, the video frames are processed by a transformer network that fuses the observations into a volumetric feature grid representing the scene; this feature grid is then decoded into an implicit 3D scene representation. Key to our approach is the transformer architecture that enables the network to learn to attend to the most relevant image frames for each 3D location in the scene, supervised only by the scene reconstruction task. Features are fused in a coarse-to-fine fashion, storing fine-level features only where needed, requiring lower memory storage and enabling fusion at interactive rates. The feature grid is then decoded to a higher-resolution scene reconstruction, using an MLP-based surface occupancy prediction from interpolated coarse-to-fine 3D features. Our approach results in an accurate surface reconstruction, outperforming state-of-the-art multi-view stereo depth estimation methods, fully-convolutional 3D reconstruction approaches, and approaches using LSTM- or GRU-based recurrent networks for video sequence fusion.

        ----

        ## [108] Maximum Likelihood Training of Score-Based Diffusion Models

        **Authors**: *Yang Song, Conor Durkan, Iain Murray, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0a9fdbb17feb6ccb7ec405cfb85222c4-Abstract.html)

        **Abstract**:

        Score-based diffusion models synthesize samples by reversing a stochastic process that diffuses data to noise, and are trained by minimizing a weighted combination of score matching losses. The log-likelihood of score-based diffusion models can be tractably computed through a connection to continuous normalizing flows, but log-likelihood is not directly optimized by the weighted combination of score matching losses. We show that for a specific weighting scheme, the objective upper bounds the negative log-likelihood, thus enabling approximate maximum likelihood training of score-based diffusion models. We empirically observe that maximum likelihood training consistently improves the likelihood of score-based diffusion models across multiple datasets, stochastic processes, and model architectures. Our best models achieve negative log-likelihoods of 2.83 and 3.76 bits/dim on CIFAR-10 and ImageNet $32\times 32$ without any data augmentation, on a par with state-of-the-art autoregressive models on these tasks.

        ----

        ## [109] Global Convergence of Gradient Descent for Asymmetric Low-Rank Matrix Factorization

        **Authors**: *Tian Ye, Simon S. Du*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0af854284f4ab0cfea8fcfd889cbb41a-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0af854284f4ab0cfea8fcfd889cbb41a-Abstract.html)

        **Abstract**:

        We study the asymmetric low-rank factorization problem:\[\min_{\mathbf{U} \in \mathbb{R}^{m \times d}, \mathbf{V} \in \mathbb{R}^{n \times d}} \frac{1}{2}\|\mathbf{U}\mathbf{V}^\top -\mathbf{\Sigma}\|_F^2\]where $\mathbf{\Sigma}$ is a given matrix of size $m \times n$ and rank $d$. This is a canonical problem that admits two difficulties in optimization: 1) non-convexity and 2) non-smoothness (due to unbalancedness of $\mathbf{U}$ and $\mathbf{V}$). This is also a prototype for more complex problems such as asymmetric matrix sensing and matrix completion. Despite being non-convex and non-smooth, it has been observed empirically that the randomly initialized gradient descent algorithm can solve this problem in polynomial time. Existing theories to explain this phenomenon all require artificial modifications of the algorithm, such as adding noise in each iteration and adding a balancing regularizer to balance the $\mathbf{U}$ and $\mathbf{V}$.This paper presents the first proof that shows randomly initialized gradient descent converges to a global minimum of the asymmetric low-rank factorization problem with a polynomial rate. For the proof, we develop 1) a new symmetrization technique to capture the magnitudes of the symmetry and asymmetry, and 2) a quantitative perturbation analysis to approximate matrix derivatives.  We believe both are useful for other related non-convex problems.

        ----

        ## [110] Adaptive Data Augmentation on Temporal Graphs

        **Authors**: *Yiwei Wang, Yujun Cai, Yuxuan Liang, Henghui Ding, Changhu Wang, Siddharth Bhatia, Bryan Hooi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b0b0994d12ad343511adfbfc364256e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b0b0994d12ad343511adfbfc364256e-Abstract.html)

        **Abstract**:

        Temporal Graph Networks (TGNs) are powerful on modeling temporal graph data based on their increased complexity. Higher complexity carries with it a higher risk of overfitting, which makes TGNs capture random noise instead of essential semantic information. To address this issue, our idea is to transform the temporal graphs using data augmentation (DA) with adaptive magnitudes, so as to effectively augment the input features and preserve the essential semantic information. Based on this idea, we present the MeTA (Memory Tower Augmentation) module: a multi-level module that processes the augmented graphs of different magnitudes on separate levels, and performs message passing across levels to provide adaptively augmented inputs for every prediction. MeTA can be flexibly applied to the training of popular TGNs to improve their effectiveness without increasing their time complexity. To complement MeTA, we propose three DA strategies to realistically model noise by modifying both the temporal and topological features. Empirical results on standard datasets show that MeTA yields significant gains for the popular TGN models on edge prediction and node classification in an efficient manner.

        ----

        ## [111] Regularized Frank-Wolfe for Dense CRFs: Generalizing Mean Field and Beyond

        **Authors**: *D. Khuê Lê-Huu, Karteek Alahari*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b0d29e5d5c8a7a25dced6405bd022a9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b0d29e5d5c8a7a25dced6405bd022a9-Abstract.html)

        **Abstract**:

        We introduce regularized Frank-Wolfe, a general and effective algorithm for inference and learning of dense conditional random fields (CRFs). The algorithm optimizes a nonconvex continuous relaxation of the CRF inference problem using vanilla Frank-Wolfe with approximate updates, which are equivalent to minimizing a regularized energy function. Our proposed method is a generalization of existing algorithms such as mean field or concave-convex procedure. This perspective not only offers a unified analysis of these algorithms, but also allows an easy way of exploring different variants that potentially yield better performance. We illustrate this in our empirical results on standard semantic segmentation datasets, where several instantiations of our regularized Frank-Wolfe outperform mean field inference, both as a standalone component and as an end-to-end trainable layer in a neural network. We also show that dense CRFs, coupled with our new algorithms, produce significant improvements over strong CNN baselines.

        ----

        ## [112] Terra: Imperative-Symbolic Co-Execution of Imperative Deep Learning Programs

        **Authors**: *Taebum Kim, Eunji Jeong, Geon-Woo Kim, Yunmo Koo, Sehoon Kim, Gyeong-In Yu, Byung-Gon Chun*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b32f1a9efe5edf3dd2f38b0c0052bfe-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b32f1a9efe5edf3dd2f38b0c0052bfe-Abstract.html)

        **Abstract**:

        Imperative programming allows users to implement their deep neural networks (DNNs) easily and has become an essential part of recent deep learning (DL) frameworks. Recently, several systems have been proposed to combine the usability of imperative programming with the optimized performance of symbolic graph execution. Such systems convert imperative Python DL programs to optimized symbolic graphs and execute them. However, they cannot fully support the usability of imperative programming. For example, if an imperative DL program contains a Python feature with no corresponding symbolic representation (e.g., third-party library calls or unsupported dynamic control flows) they fail to execute the program. To overcome this limitation, we propose Terra, an imperative-symbolic co-execution system that can handle any imperative DL programs while achieving the optimized performance of symbolic graph execution. To achieve this, Terra builds a symbolic graph by decoupling DL operations from Python features. Then, Terra conducts the imperative execution to support all Python features, while delegating the decoupled operations to the symbolic execution. We evaluated Terraâ€™s performance improvement and coverage with ten imperative DL programs for several DNN architectures. The results show that Terra can speed up the execution of all ten imperative DL programs, whereas AutoGraph, one of the state-of-the-art systems, fails to execute five of them.

        ----

        ## [113] Uniform Sampling over Episode Difficulty

        **Authors**: *Sébastien M. R. Arnold, Guneet S. Dhillon, Avinash Ravichandran, Stefano Soatto*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b3f44d9054402de39441e165a4bdfe0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b3f44d9054402de39441e165a4bdfe0-Abstract.html)

        **Abstract**:

        Episodic training is a core ingredient of few-shot learning to train models on tasks with limited labelled data. Despite its success, episodic training remains largely understudied, prompting us to ask the question: what is the best way to sample episodes? In this paper, we first propose a method to approximate episode sampling distributions based on their difficulty. Building on this method, we perform an extensive analysis and find that sampling uniformly over episode difficulty outperforms other sampling schemes, including curriculum and easy-/hard-mining. As the proposed sampling method is algorithm agnostic, we can leverage these insights to improve few-shot learning accuracies across many episodic training algorithms. We demonstrate the efficacy of our method across popular few-shot learning datasets, algorithms, network architectures, and protocols.

        ----

        ## [114] Scalable Intervention Target Estimation in Linear Models

        **Authors**: *Burak Varici, Karthikeyan Shanmugam, Prasanna Sattigeri, Ali Tajer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b94ce08688c6389ce7b68c52ce3f8c7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b94ce08688c6389ce7b68c52ce3f8c7-Abstract.html)

        **Abstract**:

        This paper considers the problem of estimating the unknown intervention targets in a causal directed acyclic graph from observational and interventional data. The focus is on soft interventions in linear structural equation models (SEMs). Current approaches to causal structure learning either work with known intervention targets or use hypothesis testing to discover the unknown intervention targets even for linear SEMs. This severely limits their scalability and sample complexity. This paper proposes a scalable and efficient algorithm that consistently identifies all intervention targets. The pivotal idea is to estimate the intervention sites from the difference between the precision matrices associated with the observational and interventional datasets. It involves repeatedly estimating such sites in different subsets of variables. The proposed algorithm can be used to also update a given observational Markov equivalence class into the interventional Markov equivalence class. Consistency, Markov equivalency, and sample complexity are established analytically. Finally, simulation results on both real and synthetic data demonstrate the gains of the proposed approach for scalable causal structure recovery. Implementation of the algorithm and the code to reproduce the simulation results are available at \url{https://github.com/bvarici/intervention-estimation}.

        ----

        ## [115] Play to Grade: Testing Coding Games as Classifying Markov Decision Process

        **Authors**: *Allen Nie, Emma Brunskill, Chris Piech*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b9b6d6d154e98ce34b3f2e4ef76eae9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b9b6d6d154e98ce34b3f2e4ef76eae9-Abstract.html)

        **Abstract**:

        Contemporary coding education often presents students with the task of developing programs that have user interaction and complex dynamic systems, such as mouse based games. While pedagogically compelling, there are no contemporary autonomous methods for providing feedback. Notably, interactive programs are impossible to grade by traditional unit tests. In this paper we formalize the challenge of providing feedback to interactive programs as a task of classifying Markov Decision Processes (MDPs). Each student's program fully specifies an MDP where the agent needs to operate and decide, under reasonable generalization, if the dynamics and reward model of the input MDP should be categorized as correct or broken. We demonstrate that by designing a cooperative objective between an agent and an autoregressive model, we can use the agent to sample differential trajectories from the input MDP that allows a classifier to determine membership: Play to Grade. Our method enables an automatic feedback system for interactive code assignments. We release a dataset of 711,274 anonymized student submissions to a single assignment with hand-coded bug labels to support future research.

        ----

        ## [116] Distributional Reinforcement Learning for Multi-Dimensional Reward Functions

        **Authors**: *Pushi Zhang, Xiaoyu Chen, Li Zhao, Wei Xiong, Tao Qin, Tie-Yan Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0b9e57c46de934cee33b0e8d1839bfc2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0b9e57c46de934cee33b0e8d1839bfc2-Abstract.html)

        **Abstract**:

        A growing trend for value-based reinforcement learning (RL) algorithms is to capture more information than scalar value functions in the value network. One of the most well-known methods in this branch is distributional RL, which models return distribution instead of scalar value. In another line of work, hybrid reward architectures (HRA) in RL have studied to model source-specific value functions for each source of reward, which is also shown to be beneficial in performance. To fully inherit the benefits of distributional RL and hybrid reward architectures, we introduce Multi-Dimensional Distributional DQN (MD3QN), which extends distributional RL to model the joint return distribution from multiple reward sources. As a by-product of joint distribution modeling, MD3QN can capture not only the randomness in returns for each source of reward, but also the rich reward correlation between the randomness of different sources. We prove the convergence for the joint distributional Bellman operator and build our empirical algorithm by minimizing the Maximum Mean Discrepancy between joint return distribution and its Bellman target. In experiments, our method accurately models the joint return distribution in environments with richly correlated reward functions, and outperforms previous RL methods utilizing multi-dimensional reward functions in the control setting.

        ----

        ## [117] Differentiable Unsupervised Feature Selection based on a Gated Laplacian

        **Authors**: *Ofir Lindenbaum, Uri Shaham, Erez Peterfreund, Jonathan Svirsky, Nicolas Casey, Yuval Kluger*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0bc10d8a74dbafbf242e30433e83aa56-Abstract.html)

        **Abstract**:

        Scientific observations may consist of a large number of variables (features). Selecting a subset of meaningful features is often crucial for identifying patterns hidden in the ambient space. In this paper, we present a method for unsupervised feature selection, and we demonstrate its advantage in clustering, a common unsupervised task. We propose a differentiable loss that combines a graph Laplacian-based score that favors low-frequency features with a gating mechanism for removing nuisance features. Our method improves upon the naive graph Laplacian score by replacing it with a gated variant computed on a subset of low-frequency features. We identify this subset by learning the parameters of continuously relaxed Bernoulli variables, which gate the entire feature space. We mathematically motivate the proposed approach and demonstrate that it is crucial to compute the graph Laplacian on the gated inputs rather than on the full feature set in the high noise regime. Using several real-world examples, we demonstrate the efficacy and advantage of the proposed approach over leading baselines.

        ----

        ## [118] Smooth Bilevel Programming for Sparse Regularization

        **Authors**: *Clarice Poon, Gabriel Peyré*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0bed45bd5774ffddc95ffe500024f628-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0bed45bd5774ffddc95ffe500024f628-Abstract.html)

        **Abstract**:

        Iteratively reweighted least square (IRLS) is a popular approach to solve sparsity-enforcing regression problems in machine learning. State of the art approaches are more efficient but typically rely on specific coordinate pruning schemes. In this work, we show how a surprisingly simple re-parametrization of IRLS, coupled with a bilevel resolution (instead of an alternating scheme) is able to achieve top performances on a wide range of sparsity (such as Lasso, group Lasso and trace norm regularizations), regularization strength (including hard constraints), and design matrices (ranging from correlated designs to differential operators). Similarly to IRLS, our method only involves linear systems resolutions, but in sharp contrast, corresponds to the minimization of a smooth function. Despite being non-convex, we show that there is no spurious minima and that saddle points are "ridable'', so that there always exists a descent direction.  We thus advocate for the use of a BFGS quasi-Newton solver, which makes our approach  simple, robust and efficient. We perform a numerical benchmark of the convergence speed of our algorithm against state of the art solvers for Lasso, group Lasso, trace norm and linearly constrained problems. These results highlight the versatility of our approach, removing the need to use different solvers depending on the specificity of the ML problem under study.

        ----

        ## [119] Grounding Representation Similarity Through Statistical Testing

        **Authors**: *Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97-Abstract.html)

        **Abstract**:

        To understand neural network behavior, recent works quantitatively compare different networks' learned representations using canonical correlation analysis (CCA), centered kernel alignment (CKA), and other dissimilarity measures. Unfortunately, these widely used measures often disagree on fundamental observations, such as whether deep networks differing only in random initialization learn similar representations. These disagreements raise the question: which, if any, of these dissimilarity measures should we believe? We provide a framework to ground this question through a concrete test: measures should have \emph{sensitivity} to changes that affect functional behavior, and \emph{specificity} against changes that do not. We quantify this through a variety of functional behaviors including probing accuracy and robustness to distribution shift, and examine changes such as varying random initialization and deleting principal components. We find that current metrics exhibit different weaknesses, note that a classical baseline performs surprisingly well, and highlight settings where all metrics appear to fail, thus providing a challenge set for further improvement.

        ----

        ## [120] A Consciousness-Inspired Planning Agent for Model-Based Reinforcement Learning

        **Authors**: *Mingde Zhao, Zhen Liu, Sitao Luan, Shuyuan Zhang, Doina Precup, Yoshua Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0c215f194276000be6a6df6528067151-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0c215f194276000be6a6df6528067151-Abstract.html)

        **Abstract**:

        We present an end-to-end, model-based deep reinforcement learning agent which dynamically attends to relevant parts of its state during planning. The agent uses a bottleneck mechanism over a set-based representation to force the number of entities to which the agent attends at each planning step to be small. In experiments, we investigate the bottleneck mechanism with several sets of customized environments featuring different challenges. We consistently observe that the design allows the planning agents to generalize their learned task-solving abilities in compatible unseen environments by attending to the relevant objects, leading to better out-of-distribution generalization performance.

        ----

        ## [121] Reward-Free Model-Based Reinforcement Learning with Linear Function Approximation

        **Authors**: *Weitong Zhang, Dongruo Zhou, Quanquan Gu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0cb929eae7a499e50248a3a78f7acfc7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0cb929eae7a499e50248a3a78f7acfc7-Abstract.html)

        **Abstract**:

        We study the model-based reward-free reinforcement learning with linear function approximation for episodic Markov decision processes (MDPs). In this setting, the agent works in two phases. In the exploration phase, the agent interacts with the environment and collects samples without the reward. In the planning phase, the agent is given a specific reward function and uses samples collected from the exploration phase to learn a good policy. We propose a new provably efficient algorithm, called UCRL-RFE under the Linear Mixture MDP assumption, where the transition probability kernel of the MDP can be parameterized by a linear function over certain feature mappings defined on the triplet of state, action, and next state. We show that to obtain an $\epsilon$-optimal policy for arbitrary reward function, UCRL-RFE needs to sample at most $\tilde O(H^5d^2\epsilon^{-2})$ episodes during the exploration phase. Here, $H$ is the length of the episode, $d$ is the dimension of the feature mapping. We also propose a variant of UCRL-RFE using Bernstein-type bonus and show that it needs to sample at most $\tilde O(H^4d(H + d)\epsilon^{-2})$ to achieve an $\epsilon$-optimal policy. By constructing a special class of linear Mixture MDPs, we also prove that for any reward-free algorithm, it needs to sample at least $\tilde \Omega(H^2d\epsilon^{-2})$ episodes to obtain an $\epsilon$-optimal policy. Our upper bound matches the lower bound in terms of the dependence on $\epsilon$ and the dependence on $d$ if $H \ge d$.

        ----

        ## [122] Beltrami Flow and Neural Diffusion on Graphs

        **Authors**: *Ben Chamberlain, James Rowbottom, Davide Eynard, Francesco Di Giovanni, Xiaowen Dong, Michael M. Bronstein*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0cbed40c0d920b94126eaf5e707be1f5-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0cbed40c0d920b94126eaf5e707be1f5-Abstract.html)

        **Abstract**:

        We propose a novel class of graph neural networks based on the discretized Beltrami flow, a non-Euclidean diffusion PDE. In our model, node features are supplemented with  positional encodings derived from the graph topology and jointly evolved by the Beltrami flow,  producing simultaneously continuous feature learning, topology evolution. The resulting model generalizes many popular graph neural networks and achieves state-of-the-art results on several benchmarks.

        ----

        ## [123] Think Big, Teach Small: Do Language Models Distil Occam's Razor?

        **Authors**: *Gonzalo Jaimovitch-Lopez, David Castellano Falcón, César Ferri, José Hernández-Orallo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0cd6a652ed1f7811192db1f700c8f0e7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0cd6a652ed1f7811192db1f700c8f0e7-Abstract.html)

        **Abstract**:

        Large language models have recently shown a remarkable ability for few-shot learning, including patterns of algorithmic nature. However, it is still an open question to determine what kind of patterns these models can capture and how many examples they need in their prompts. We frame this question as a teaching problem with strong priors, and study whether language models can identify simple algorithmic concepts from small witness sets. In particular, we explore how several GPT architectures, program induction systems and humans perform in terms of the complexity of the concept and the number of additional examples, and how much their behaviour differs. This first joint analysis of language models and machine teaching can address key questions for artificial intelligence and machine learning, such as whether some strong priors, and Occam’s razor in particular, can be distilled from data, making learning from a few examples possible.

        ----

        ## [124] Disentangling Identifiable Features from Noisy Data with Structured Nonlinear ICA

        **Authors**: *Hermanni Hälvä, Sylvain Le Corff, Luc Lehéricy, Jonathan So, Yongjie Zhu, Elisabeth Gassiat, Aapo Hyvärinen*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0cdbb4e65815fbaf79689b15482e7575-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0cdbb4e65815fbaf79689b15482e7575-Abstract.html)

        **Abstract**:

        We introduce a new general identifiable framework for principled disentanglement referred to as Structured Nonlinear Independent Component Analysis (SNICA). Our contribution is to extend the identifiability theory of deep generative models for a very broad class of structured models. While previous works have shown identifiability for specific classes of time-series models, our theorems extend this to more general temporal structures as well as to models with more complex  structures such as spatial dependencies. In particular, we establish the major result that identifiability for this framework holds even in the presence of noise of unknown distribution. Finally, as an example of our framework's flexibility, we introduce the first nonlinear ICA model for time-series that combines the following very useful properties: it accounts for both nonstationarity and autocorrelation in a fully unsupervised setting;  performs dimensionality reduction;  models hidden states; and  enables principled estimation and inference by variational maximum-likelihood.

        ----

        ## [125] Conditionally Parameterized, Discretization-Aware Neural Networks for Mesh-Based Modeling of Physical Systems

        **Authors**: *Jiayang Xu, Aniruddhe Pradhan, Karthik Duraisamy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0cddb7c06f1cd518e1efdc0e20b70c31-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0cddb7c06f1cd518e1efdc0e20b70c31-Abstract.html)

        **Abstract**:

        Simulations of complex physical systems are typically realized by discretizing partial differential equations (PDEs) on unstructured meshes. While neural networks have recently been explored for the surrogate and reduced order modeling of PDE solutions, they often ignore interactions or hierarchical relations between input features, and process them as concatenated mixtures. We generalize the idea of conditional parameterization -- using trainable functions of input parameters to generate the weights of a neural network, and extend them in a flexible way to encode critical information. Inspired by discretized numerical methods, choices of the parameters include physical quantities and mesh topology features. The functional relation between the modeled features and the parameters is built into the network architecture. The method is implemented on different networks and applied to frontier scientific machine learning tasks including the discovery of unmodeled physics, super-resolution of coarse fields, and the simulation of unsteady flows with chemical reactions. The results show that the conditionally-parameterized networks provide superior performance compared to their traditional counterparts. The CP-GNet - an architecture that can be trained on very few data snapshots - is proposed as the first deep learning model capable of standalone prediction of reacting flows on irregular meshes.

        ----

        ## [126] USCO-Solver: Solving Undetermined Stochastic Combinatorial Optimization Problems

        **Authors**: *Guangmo Tong*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d3180d672e08b4c5312dcdafdf6ef36-Abstract.html)

        **Abstract**:

        Real-world decision-making systems are often subject to uncertainties that have to be resolved through observational data. Therefore, we are frequently confronted with combinatorial optimization problems of which the objective function is unknown and thus has to be debunked using empirical evidence. In contrast to the common practice that relies on a learning-and-optimization strategy, we consider the regression between combinatorial spaces, aiming to infer high-quality optimization solutions from samples of input-solution pairs -- without the need to learn the objective function. Our main deliverable is a universal solver that is able to handle abstract undetermined stochastic combinatorial optimization problems. For learning foundations, we present learning-error analysis under the PAC-Bayesian framework using a new margin-based analysis. In empirical studies, we demonstrate our design using proof-of-concept experiments, and compare it with other methods that are potentially applicable. Overall, we obtain highly encouraging experimental results for several classic combinatorial problems on both synthetic and real-world datasets.

        ----

        ## [127] Adaptive Conformal Inference Under Distribution Shift

        **Authors**: *Isaac Gibbs, Emmanuel J. Candès*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html)

        **Abstract**:

        We develop methods for forming prediction sets in an online setting where the data generating distribution is allowed to vary over time in an unknown fashion. Our framework builds on ideas from conformal inference to provide a general wrapper that can be combined with any black box method that produces point predictions of the unseen label or estimated quantiles of its distribution. While previous conformal inference methods rely on the assumption that the data are exchangeable, our adaptive approach provably achieves the desired coverage frequency over long-time intervals irrespective of the true data generating process. We accomplish this by modelling the distribution shift as a learning problem in a single parameter whose optimal value is varying over time and must be continuously re-estimated. We test our method, adaptive conformal inference, on two real world datasets and find that its predictions are robust to visible and significant distribution shifts.

        ----

        ## [128] Periodic Activation Functions Induce Stationarity

        **Authors**: *Lassi Meronen, Martin Trapp, Arno Solin*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d5a4a5a748611231b945d28436b8ece-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d5a4a5a748611231b945d28436b8ece-Abstract.html)

        **Abstract**:

        Neural network models are known to reinforce hidden data biases, making them unreliable and difficult to interpret. We seek to build models that `know what they do not know' by introducing inductive biases in the function space. We show that periodic activation functions in Bayesian neural networks establish a connection between the prior on the network weights and translation-invariant, stationary Gaussian process priors. Furthermore, we show that this link goes beyond sinusoidal (Fourier) activations by also covering triangular wave and periodic ReLU activation functions. In a series of experiments, we show that periodic activation functions obtain comparable performance for in-domain data and capture sensitivity to perturbed inputs in deep neural networks for out-of-domain detection.

        ----

        ## [129] Towards Optimal Strategies for Training Self-Driving Perception Models in Simulation

        **Authors**: *David Acuna, Jonah Philion, Sanja Fidler*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d5bd023a3ee11c7abca5b42a93c4866-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d5bd023a3ee11c7abca5b42a93c4866-Abstract.html)

        **Abstract**:

        Autonomous driving relies on a huge volume of real-world data to be labeled to high precision.  Alternative solutions seek to exploit driving simulators that can generate large amounts of labeled data with a plethora of content variations. However, the domain gap between the synthetic and real data remains,  raising the following important question: What are the best way to utilize a self-driving simulator for perception tasks?. In this work, we build on top of recent advances in domain-adaptation theory, and from this perspective, propose ways to minimize the reality gap. We primarily focus on the use of labels in the synthetic domain alone. Our approach introduces both a principled way to learn neural-invariant representations and a  theoretically inspired view on how to sample the data from the simulator. Our method is easy to implement in practice as it is agnostic of the network architecture and the choice of the simulator.   We showcase our approach on the bird's-eye-view vehicle segmentation task with multi-sensor data (cameras, lidar) using an open-source simulator (CARLA), and evaluate the entire framework on a real-world dataset (nuScenes). Last but not least, we show what types of variations (e.g. weather conditions, number of assets, map design and color diversity) matter to perception networks when trained with driving simulators, and which ones can be compensated for with our domain adaptation technique.

        ----

        ## [130] KS-GNN: Keywords Search over Incomplete Graphs via Graphs Neural Network

        **Authors**: *Yu Hao, Xin Cao, Yufan Sheng, Yixiang Fang, Wei Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d7363894acdee742caf7fe4e97c4d49-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d7363894acdee742caf7fe4e97c4d49-Abstract.html)

        **Abstract**:

        Keyword search is a fundamental task to retrieve information that is the most relevant to the query keywords. Keyword search over graphs aims to find subtrees or subgraphs containing all query keywords ranked according to some criteria. Existing studies all assume that the graphs have complete information. However, real-world graphs may contain some missing information (such as edges or keywords), thus making the problem much more challenging. To solve the problem of keyword search over incomplete graphs, we propose a novel model named KS-GNN based on the graph neural network and the auto-encoder. By considering the latent relationships and the frequency of different keywords, the proposed KS-GNN aims to alleviate the effect of missing information and is able to learn low-dimensional representative node embeddings that preserve both graph structure and keyword features. Our model can effectively answer keyword search queries with linear time complexity over incomplete graphs. The experiments on four real-world datasets show that our model consistently achieves better performance than state-of-the-art baseline methods in graphs having missing information.

        ----

        ## [131] Reconstruction for Powerful Graph Representations

        **Authors**: *Leonardo Cotta, Christopher Morris, Bruno Ribeiro*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d8080853a54f8985276b0130266a657-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d8080853a54f8985276b0130266a657-Abstract.html)

        **Abstract**:

        Graph neural networks (GNNs) have limited expressive power, failing to represent many graph classes correctly. While more expressive graph representation learning (GRL) alternatives can distinguish some of these classes, they are significantly harder to implement, may not scale well, and have not been shown to outperform well-tuned GNNs in real-world tasks. Thus, devising simple, scalable, and expressive GRL architectures that also achieve real-world improvements remains an open challenge. In this work, we show the extent to which graph reconstruction---reconstructing a graph from its subgraphs---can mitigate the theoretical and practical problems currently faced by GRL architectures. First, we leverage graph reconstruction to build two new classes of expressive graph representations. Secondly, we show how graph reconstruction boosts the expressive power of any GNN architecture while being a (provably) powerful inductive bias for invariances to vertex removals. Empirically,  we show how reconstruction can boost GNN's expressive power---while maintaining its invariance to permutations of the vertices---by solving seven graph property tasks not solvable by the original GNN. Further, we demonstrate how it boosts state-of-the-art GNN's performance across nine real-world benchmark datasets.

        ----

        ## [132] Revealing and Protecting Labels in Distributed Training

        **Authors**: *Trung Dang, Om Thakkar, Swaroop Ramaswamy, Rajiv Mathews, Peter Chin, Françoise Beaufays*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0d924f0e6b3fd0d91074c22727a53966-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0d924f0e6b3fd0d91074c22727a53966-Abstract.html)

        **Abstract**:

        Distributed learning paradigms such as federated learning often involve transmission of model updates, or gradients, over a network, thereby avoiding transmission of private data. However, it is possible for sensitive information about the training data to be revealed from such gradients. Prior works have demonstrated that labels can be revealed analytically from the last layer of certain models (e.g., ResNet), or they can be reconstructed jointly with model inputs by using Gradients Matching [Zhu et al.] with additional knowledge about the current state of the model. In this work, we propose a method to discover the set of labels of training samples from only the gradient of the last layer and the id to label mapping. Our method is applicable to a wide variety of model architectures across multiple domains. We demonstrate the effectiveness of our method for model training in two domains - image classification, and automatic speech recognition. Furthermore, we show that existing reconstruction techniques improve their efficacy when used in conjunction with our method. Conversely, we demonstrate that gradient quantization and sparsification can significantly reduce the success of the attack.

        ----

        ## [133] Solving Graph-based Public Goods Games with Tree Search and Imitation Learning

        **Authors**: *Victor-Alexandru Darvariu, Stephen Hailes, Mirco Musolesi*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0db2e204010400f5c506620adcd1ae68-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0db2e204010400f5c506620adcd1ae68-Abstract.html)

        **Abstract**:

        Public goods games represent insightful settings for studying incentives for individual agents to make contributions that, while costly for each of them, benefit the wider society. In this work, we adopt the perspective of a central planner with a global view of a network of self-interested agents and the goal of maximizing some desired property in the context of a best-shot public goods game. Existing algorithms for this known NP-complete problem find solutions that are sub-optimal and cannot optimize for criteria other than social welfare.In order to efficiently solve public goods games, our proposed method directly exploits the correspondence between equilibria and the Maximal Independent Set (mIS) structural property of graphs. In particular, we define a Markov Decision Process which incrementally generates an mIS, and adopt a planning method to search for equilibria, outperforming existing methods. Furthermore, we devise a graph imitation learning technique that uses demonstrations of the search to obtain a graph neural network parametrized policy which quickly generalizes to unseen game instances. Our evaluation results show that this policy is able to reach 99.5\% of the performance of the planning method while being three orders of magnitude faster to evaluate on the largest graphs tested. The methods presented in this work can be applied to a large class of public goods games of potentially high societal impact and more broadly to other graph combinatorial optimization problems.

        ----

        ## [134] Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence

        **Authors**: *Qi Qi, Youzhi Luo, Zhao Xu, Shuiwang Ji, Tianbao Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0dd1bc593a91620daecf7723d2235624-Abstract.html)

        **Abstract**:

        Areas under ROC (AUROC) and precision-recall curves (AUPRC) are common metrics for evaluating classification performance for imbalanced problems. Compared with AUROC, AUPRC is a more appropriate metric for highly imbalanced datasets. While stochastic optimization of AUROC has been studied extensively, principled stochastic optimization of AUPRC has been rarely explored. In this work, we propose a principled technical method to optimize AUPRC for deep learning. Our approach is based on maximizing the averaged precision (AP), which is an unbiased point estimator of AUPRC. We cast the objective into a sum of dependent compositional functions with inner functions dependent on random variables of the outer level. We propose efficient adaptive and non-adaptive stochastic algorithms named SOAP with provable convergence guarantee under mild conditions by leveraging recent advances in stochastic compositional optimization. Extensive experimental results on image and graph datasets demonstrate that our proposed method outperforms prior methods on imbalanced problems in terms of AUPRC. To the best of our knowledge, our work represents the first attempt to optimize AUPRC with provable convergence. The SOAP has been implemented in the libAUC library at https://libauc.org/.

        ----

        ## [135] Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization

        **Authors**: *Qi Zhu, Carl Yang, Yidan Xu, Haonan Wang, Chao Zhang, Jiawei Han*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0dd6049f5fa537d41753be6d37859430-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0dd6049f5fa537d41753be6d37859430-Abstract.html)

        **Abstract**:

        Graph neural networks (GNNs) have achieved superior performance in various applications, but training dedicated GNNs can be costly for large-scale graphs. Some recent work started to study the pre-training of GNNs. However, none of them provide theoretical insights into the design of their frameworks, or clear requirements and guarantees towards their transferability. In this work, we establish a theoretically grounded and practically useful framework for the transfer learning of GNNs. Firstly, we propose a novel view towards the essential graph information and advocate the capturing of it as the goal of transferable GNN training, which motivates the design of EGI (Ego-Graph Information maximization) to analytically achieve this goal. Secondly,when node features are structure-relevant, we conduct an analysis of EGI transferability regarding the difference between the local graph Laplacians of the source and target graphs. We conduct controlled synthetic experiments to directly justify our theoretical conclusions. Comprehensive experiments on two real-world network datasets show consistent results in the analyzed setting of direct-transfering, while those on large-scale knowledge graphs show promising results in the more practical setting of transfering with fine-tuning.

        ----

        ## [136] You are caught stealing my winning lottery ticket! Making a lottery ticket claim its ownership

        **Authors**: *Xuxi Chen, Tianlong Chen, Zhenyu Zhang, Zhangyang Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0dfd8a39e2a5dd536c185e19a804a73b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0dfd8a39e2a5dd536c185e19a804a73b-Abstract.html)

        **Abstract**:

        Despite tremendous success in many application scenarios, the training and inference costs of using deep learning are also rapidly increasing over time. The lottery ticket hypothesis (LTH) emerges as a promising framework to leverage a special sparse subnetwork (i.e., $\textit{winning ticket}$) instead of a full model for both training and inference, that can lower both costs without sacrificing the performance. The main resource bottleneck of LTH is however the extraordinary cost to find the sparse mask of the winning ticket. That makes the found winning ticket become a valuable asset to the owners, highlighting the necessity of protecting its copyright. Our setting adds a new dimension to the recently soaring interest in protecting against the intellectual property (IP) infringement of deep models and verifying their ownerships, since they take owners' massive/unique resources to develop or train. While existing methods explored encrypted weights or predictions, we investigate a unique way to leverage sparse topological information to perform $\textit{lottery verification}$, by developing several graph-based signatures that can be embedded as credentials. By further combining trigger set-based methods, our proposal can work in both white-box and black-box verification scenarios. Through extensive experiments, we demonstrate the effectiveness of lottery verification in diverse models (ResNet-20, ResNet-18, ResNet-50) on CIFAR-10 and CIFAR-100. Specifically, our verification is shown to be robust to removal attacks such as model fine-tuning and pruning, as well as several ambiguity attacks. Our codes are available at https://github.com/VITA-Group/NO-stealing-LTH.

        ----

        ## [137] Complexity Lower Bounds for Nonconvex-Strongly-Concave Min-Max Optimization

        **Authors**: *Haochuan Li, Yi Tian, Jingzhao Zhang, Ali Jadbabaie*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e105949d99a32ca1751703e94ece601-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e105949d99a32ca1751703e94ece601-Abstract.html)

        **Abstract**:

        We provide a first-order oracle complexity lower bound for finding stationary points of min-max optimization problems where the objective function is smooth, nonconvex in the minimization variable, and strongly concave in the maximization variable. We establish a lower bound of $\Omega\left(\sqrt{\kappa}\epsilon^{-2}\right)$ for deterministic oracles, where $\epsilon$ defines the level of approximate stationarity and $\kappa$ is the condition number. Our lower bound matches the best existing upper bound in the $\epsilon$ and $\kappa$ dependence up to logarithmic factors. For stochastic oracles, we provide a lower bound of $\Omega\left(\sqrt{\kappa}\epsilon^{-2} + \kappa^{1/3}\epsilon^{-4}\right)$. It suggests that there is a gap between the best existing upper bound $\mathcal{O}(\kappa^3 \epsilon^{-4})$ and our lower bound in the condition number dependence.

        ----

        ## [138] Early-stopped neural networks are consistent

        **Authors**: *Ziwei Ji, Justin D. Li, Matus Telgarsky*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e1ebad68af7f0ae4830b7ac92bc3c6f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e1ebad68af7f0ae4830b7ac92bc3c6f-Abstract.html)

        **Abstract**:

        This work studies the behavior of shallow ReLU networks trained with the logistic loss via gradient descent on binary classification data where the underlying data distribution is general, and the (optimal) Bayes risk is not necessarily zero.  In this setting, it is shown that gradient descent with early stopping achieves population risk arbitrarily close to optimal in terms of not just logistic and misclassification losses, but also in terms of calibration, meaning the sigmoid mapping of its outputs approximates the true underlying conditional distribution arbitrarily finely.  Moreover, the necessary iteration, sample, and architectural complexities of this analysis all scale naturally with a certain complexity measure of the true conditional model.  Lastly, while it is not shown that early stopping is necessary, it is shown that any classifier satisfying a basic local interpolation property is inconsistent.

        ----

        ## [139] NxMTransformer: Semi-Structured Sparsification for Natural Language Understanding via ADMM

        **Authors**: *Connor Holmes, Minjia Zhang, Yuxiong He, Bo Wu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e4f5cc9f4f3f7f1651a6b9f9214e5b1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e4f5cc9f4f3f7f1651a6b9f9214e5b1-Abstract.html)

        **Abstract**:

        Natural Language Processing (NLP) has recently achieved great success by using huge pre-trained Transformer networks. However, these models often contain hundreds of millions or even billions of parameters, bringing challenges to online deployment due to latency constraints. Recently, hardware manufacturers have introduced dedicated hardware for NxM sparsity to provide the flexibility of unstructured pruning with the runtime efficiency of structured approaches. NxM sparsity permits arbitrarily selecting M parameters to retain from a contiguous group of N in the dense representation. However, due to the extremely high complexity of pre-trained models, the standard sparse fine-tuning techniques often fail to generalize well on downstream tasks, which have limited data resources. To address such an issue in a principled manner, we introduce a new learning framework, called NxMTransformer, to induce NxM semi-structured sparsity on pretrained language models for natural language understanding to obtain better performance. In particular, we propose to formulate the NxM sparsity as a constrained optimization problem and use Alternating Direction Method of Multipliers (ADMM) to optimize the downstream tasks while taking the underlying hardware constraints into consideration. ADMM decomposes the NxM sparsification problem into two sub-problems that can be solved sequentially, generating sparsified Transformer networks that achieve high accuracy while being able to effectively execute on newly released hardware. We apply our approach to a wide range of NLP tasks, and our proposed method is able to achieve 1.7 points higher accuracy in GLUE score than current best practices. Moreover, we perform detailed analysis on our approach and shed light on how ADMM affects fine-tuning accuracy for downstream tasks. Finally, we illustrate how NxMTransformer achieves additional performance improvement with knowledge distillation based methods.

        ----

        ## [140] Reliable Decisions with Threshold Calibration

        **Authors**: *Roshni Sahoo, Shengjia Zhao, Alyssa Chen, Stefano Ermon*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e65972dce68dad4d52d063967f0a705-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e65972dce68dad4d52d063967f0a705-Abstract.html)

        **Abstract**:

        Decision makers rely on probabilistic forecasts to predict the loss of different decision rules before deployment. When the forecasted probabilities match the true frequencies, predicted losses will be accurate. Although perfect forecasts are typically impossible, probabilities can be calibrated to match the true frequencies on average. However, we find that this \textit{average} notion of calibration, which is typically used in practice, does not necessarily guarantee accurate decision loss prediction. Specifically in the regression setting, the loss of threshold decisions, which are decisions based on whether the forecasted outcome falls above or below a cutoff, might not be predicted accurately. We propose a stronger notion of calibration called threshold calibration, which is exactly the condition required to ensure that decision loss is predicted accurately for threshold decisions. We provide an efficient algorithm which takes an uncalibrated forecaster as input and provably outputs a threshold-calibrated forecaster. Our procedure allows downstream decision makers to confidently estimate the loss of any threshold decision under any threshold loss function. Empirically, threshold calibration improves decision loss prediction without compromising on the quality of the decisions in two real-world settings: hospital scheduling decisions and resource allocation decisions.

        ----

        ## [141] End-to-End Weak Supervision

        **Authors**: *Salva Rühling Cachay, Benedikt Boecking, Artur Dubrawski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e674a918ebca3f78bfe02e2f387689d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e674a918ebca3f78bfe02e2f387689d-Abstract.html)

        **Abstract**:

        Aggregating multiple sources of weak supervision (WS) can ease the data-labeling bottleneck prevalent in many machine learning applications, by replacing the tedious manual collection of ground truth labels. Current state of the art approaches that do not use any labeled training data, however, require two separate modeling steps: Learning a probabilistic latent variable model based on the WS sources -- making assumptions that rarely hold in practice -- followed by downstream model training. Importantly, the first step of modeling does not consider the performance of the downstream model.To address these caveats we propose an end-to-end approach for directly learning the downstream model by maximizing its agreement with probabilistic labels generated by reparameterizing previous probabilistic posteriors with a neural network. Our results show improved performance over prior work in terms of end model performance on downstream test sets, as well as in terms of improved robustness to dependencies among weak supervision sources.

        ----

        ## [142] Shift Invariance Can Reduce Adversarial Robustness

        **Authors**: *Vasu Singla, Songwei Ge, Ronen Basri, David W. Jacobs*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e7c7d6c41c76b9ee6445ae01cc0181d-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e7c7d6c41c76b9ee6445ae01cc0181d-Abstract.html)

        **Abstract**:

        Shift invariance is a critical property of CNNs that improves performance on classification.  However, we show that invariance to circular shifts can also lead to greater sensitivity to adversarial attacks.  We first characterize the margin between classes when a shift-invariant {\em linear} classifier is used. We show that the margin can only depend on the DC component of the signals.  Then, using results about infinitely wide networks, we show that in some simple cases, fully connected and shift-invariant neural networks produce linear decision boundaries.  Using this, we prove that shift invariance in neural networks produces adversarial examples for the simple case of two classes, each consisting of a single image with a black or white dot on a gray background.  This is more than a curiosity; we show empirically that with real datasets and realistic architectures, shift invariance reduces adversarial robustness.  Finally, we describe initial experiments using synthetic data to probe the source of this connection.

        ----

        ## [143] Wisdom of the Crowd Voting: Truthful Aggregation of Voter Information and Preferences

        **Authors**: *Grant Schoenebeck, Biaoshuai Tao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e900ad84f63618452210ab8baae0218-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e900ad84f63618452210ab8baae0218-Abstract.html)

        **Abstract**:

        We consider two-alternative elections where voters' preferences depend on a state variable that is not directly observable. Each voter receives a private signal that is correlated to the state variable. As a special case, our model captures the common scenario where voters can be categorized into three types: those who always prefer one alternative, those who always prefer the other, and those contingent voters whose preferences depends on the state.  In this setting, even if every voter is a contingent voter, agents voting according to their private information need not result in the adoption of the universally preferred alternative, because the signals can be systematically biased.We present a mechanism that elicits and aggregates the private signals from the voters, and outputs the alternative that is favored by the majority.  In particular, voters truthfully reporting their signals forms a strong Bayes Nash equilibrium (where no coalition of voters can deviate and receive a better outcome).

        ----

        ## [144] Replay-Guided Adversarial Environment Design

        **Authors**: *Minqi Jiang, Michael Dennis, Jack Parker-Holder, Jakob N. Foerster, Edward Grefenstette, Tim Rocktäschel*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html)

        **Abstract**:

        Deep reinforcement learning (RL) agents may successfully generalize to new settings if trained on an appropriately diverse set of environment and task configurations. Unsupervised Environment Design (UED) is a promising self-supervised RL paradigm, wherein the free parameters of an underspecified environment are automatically adapted during training to the agent's capabilities, leading to the emergence of diverse training environments. Here, we cast Prioritized Level Replay (PLR), an empirically successful but theoretically unmotivated method that selectively samples randomly-generated training levels, as UED. We argue that by curating completely random levels, PLR, too, can generate novel and complex levels for effective training. This insight reveals a natural class of UED methods we call Dual Curriculum Design (DCD). Crucially, DCD includes both PLR and a popular UED algorithm, PAIRED, as special cases and inherits similar theoretical guarantees. This connection allows us to develop novel theory for PLR, providing a version with a robustness guarantee at Nash equilibria. Furthermore, our theory suggests a highly counterintuitive improvement to PLR: by stopping the agent from updating its policy on uncurated levels (training on less data), we can improve the convergence to Nash equilibria. Indeed, our experiments confirm that our new method, PLR$^{\perp}$, obtains better results on a suite of out-of-distribution, zero-shot transfer tasks, in addition to demonstrating that PLR$^{\perp}$ improves the performance of PAIRED, from which it inherited its theoretical framework.

        ----

        ## [145] There Is No Turning Back: A Self-Supervised Approach for Reversibility-Aware Reinforcement Learning

        **Authors**: *Nathan Grinsztajn, Johan Ferret, Olivier Pietquin, Philippe Preux, Matthieu Geist*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e98aeeb54acf612b9eb4e48a269814c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e98aeeb54acf612b9eb4e48a269814c-Abstract.html)

        **Abstract**:

        We propose to learn to distinguish reversible from irreversible actions for better informed decision-making in Reinforcement Learning (RL). From theoretical considerations, we show that approximate reversibility can be learned through a simple surrogate task: ranking randomly sampled trajectory events in chronological order. Intuitively, pairs of events that are always observed in the same order are likely to be separated by an irreversible sequence of actions. Conveniently, learning the temporal order of events can be done in a fully self-supervised way, which we use to estimate the reversibility of actions from experience, without any priors.We propose two different strategies that incorporate reversibility in RL agents, one strategy for exploration (RAE) and one strategy for control (RAC). We demonstrate the potential of reversibility-aware agents in several environments, including the challenging Sokoban game. In synthetic tasks, we show that we can learn control policies that never fail and reduce to zero the side-effects of interactions, even without access to the reward function.

        ----

        ## [146] Learning to Execute: Efficient Learning of Universal Plan-Conditioned Policies in Robotics

        **Authors**: *Ingmar Schubert, Danny Driess, Ozgur S. Oguz, Marc Toussaint*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0e9b734aa25ca8096cb7b56dc0dd8929-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0e9b734aa25ca8096cb7b56dc0dd8929-Abstract.html)

        **Abstract**:

        Applications of Reinforcement Learning (RL) in robotics are often limited by high data demand. On the other hand, approximate models are readily available in many robotics scenarios, making model-based approaches like planning a data-efficient alternative. Still, the performance of these methods suffers if the model is imprecise or wrong. In this sense, the respective strengths and weaknesses of RL and model-based planners are complementary. In the present work, we investigate how both approaches can be integrated into one framework that combines their strengths. We introduce Learning to Execute (L2E), which leverages information contained in approximate plans to learn universal policies that are conditioned on plans. In our robotic manipulation experiments, L2E exhibits increased performance when compared to pure RL, pure planning, or baseline methods combining learning and planning.

        ----

        ## [147] Self-Diagnosing GAN: Diagnosing Underrepresented Samples in Generative Adversarial Networks

        **Authors**: *Jinhee Lee, Haeri Kim, Youngkyu Hong, Hye Won Chung*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0ebcc77dc72360d0eb8e9504c78d38bd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0ebcc77dc72360d0eb8e9504c78d38bd-Abstract.html)

        **Abstract**:

        Despite remarkable performance in producing realistic samples, Generative Adversarial Networks (GANs) often produce low-quality samples near low-density regions of the data manifold, e.g., samples of minor groups. Many techniques have been developed to improve the quality of generated samples, either by post-processing generated samples or by pre-processing the empirical data distribution, but at the cost of reduced diversity. To promote diversity in sample generation without degrading the overall quality, we propose a simple yet effective method to diagnose and emphasize underrepresented samples during training of a GAN. The main idea is to use the statistics of the discrepancy between the data distribution and the model distribution at each data instance. Based on the observation that the underrepresented samples have a high average discrepancy or high variability in discrepancy, we propose a method to emphasize those samples during training of a GAN. Our experimental results demonstrate that the proposed method improves GAN performance on various datasets, and it is especially effective in improving the quality and diversity of sample generation for minor groups.

        ----

        ## [148] Online Multi-Armed Bandits with Adaptive Inference

        **Authors**: *Maria Dimakopoulou, Zhimei Ren, Zhengyuan Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0ec04cb3912c4f08874dd03716f80df1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0ec04cb3912c4f08874dd03716f80df1-Abstract.html)

        **Abstract**:

        During online decision making in Multi-Armed Bandits (MAB), one needs to conduct inference on the true mean reward of each arm based on data collected so far at each step. However, since the arms are adaptively selected--thereby yielding non-iid data--conducting inference accurately is not straightforward. In particular, sample averaging, which is used in the family of UCB and Thompson sampling (TS) algorithms, does not provide a good choice as it suffers from bias and a lack of good statistical properties (e.g.  asymptotic normality). Our thesis in this paper is that more sophisticated inference schemes that take into account the adaptive nature of the sequentially collected data can unlock further performance gains, even though both UCB and TS type algorithms are optimal in the worst case. In particular, we propose a variant of TS-style algorithms--which we call doubly adaptive TS--that leverages recent advances in causal inference and adaptively reweights the terms of a doubly robust estimator on the true mean reward of each arm. Through 20 synthetic domain experiments and a semi-synthetic experiment based on data from an A/B test of a web service, we demonstrate that using an adaptive inferential scheme (while still retaining the exploration efficacy of TS) provides clear benefits in online decision making: the proposed DATS algorithm has superior empirical performance to existing baselines (UCB and TS) in terms of regret and sample complexity in identifying the best arm. In addition, we also provide a finite-time regret bound of doubly adaptive TS that matches (up to log factors) those of UCB and TS algorithms, thereby establishing that its improved practical benefits do not come at the expense of worst-case suboptimality.

        ----

        ## [149] Efficient Truncated Linear Regression with Unknown Noise Variance

        **Authors**: *Constantinos Daskalakis, Patroklos Stefanou, Rui Yao, Emmanouil Zampetakis*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0ed8861dc36bee580d100f91283d0559-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0ed8861dc36bee580d100f91283d0559-Abstract.html)

        **Abstract**:

        Truncated linear regression is a classical challenge in Statistics, wherein a label, $y = w^T x + \varepsilon$, and its corresponding feature vector, $x \in \mathbb{R}^k$, are only observed if the label falls in some subset $S \subseteq \mathbb{R}$; otherwise the existence of the pair $(x, y)$ is hidden from observation. Linear regression with truncated observations has remained a challenge, in its general form, since the early works of [Tobin'58, Amemiya '73]. When the distribution of the error is normal with known variance, recent work of [Daskalakis et al. '19] provides computationally and statistically efficient estimators of the linear model, $w$. In this paper, we provide the first computationally and statistically efficient estimators for truncated linear regression when the noise variance is unknown, estimating both the linear model and the variance of the noise. Our estimator is based on an efficient implementation of Projected Stochastic Gradient Descent on the negative log-likelihood of the truncated sample. Importantly, we show that the error of our estimates is asymptotically normal, and we use this to provide explicit confidence regions for our estimates.

        ----

        ## [150] Breaking the Dilemma of Medical Image-to-image Translation

        **Authors**: *Lingke Kong, Chenyu Lian, Detian Huang, Zhenjiang Li, Yanle Hu, Qichao Zhou*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0f2818101a7ac4b96ceeba38de4b934c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0f2818101a7ac4b96ceeba38de4b934c-Abstract.html)

        **Abstract**:

        Supervised Pix2Pix and unsupervised Cycle-consistency are two modes that dominate the field of medical image-to-image translation. However, neither modes are ideal. The Pix2Pix mode has excellent performance. But it requires paired and well pixel-wise aligned images, which may not always be achievable due to respiratory motion or anatomy change between times that paired images are acquired. The Cycle-consistency mode is less stringent with training data and works well on unpaired or misaligned images. But its performance may not be optimal. In order to break the dilemma of the existing modes, we propose a new unsupervised mode called RegGAN for medical image-to-image translation. It is based on the theory of "loss-correction". In RegGAN, the misaligned target images are considered as noisy labels and the generator is trained with an additional registration network to fit the misaligned noise distribution adaptively. The goal is to search for the common optimal solution to both image-to-image translation and registration tasks. We incorporated RegGAN into a few state-of-the-art image-to-image translation methods and demonstrated that RegGAN could be easily combined with these methods to improve their performances. Such as a simple CycleGAN in our mode surpasses latest NICEGAN even though using less network parameters. Based on our results, RegGAN outperformed both Pix2Pix on aligned data and Cycle-consistency on misaligned or unpaired data. RegGAN is insensitive to noises which makes it a better choice for a wide range of scenarios, especially for medical image-to-image translation tasks in which well pixel-wise aligned data are not available. Code and dataset are available at https://github.com/Kid-Liet/Reg-GAN.

        ----

        ## [151] Temporally Abstract Partial Models

        **Authors**: *Khimya Khetarpal, Zafarali Ahmed, Gheorghe Comanici, Doina Precup*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0f3d014eead934bbdbacb62a01dc4831-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0f3d014eead934bbdbacb62a01dc4831-Abstract.html)

        **Abstract**:

        Humans and animals have the ability to reason and make predictions about different courses of action at many time scales. In reinforcement learning, option models (Sutton, Precup \& Singh, 1999; Precup, 2000) provide the framework for this kind of temporally abstract prediction and reasoning. Natural intelligent agents are also able to focus their attention on courses of action that are relevant or feasible in a given situation, sometimes termed affordable actions. In this paper, we define a notion of affordances for options, and develop temporally abstract partial option models, that take into account the fact that an option might be affordable only in certain situations. We analyze the trade-offs between estimation and approximation error in planning and learning when using such models, and identify some interesting special cases. Additionally, we empirically demonstrate the ability to learn both affordances and partial option models online resulting in improved sample efficiency and planning time in the Taxi domain.

        ----

        ## [152] TransMatcher: Deep Image Matching Through Transformers for Generalizable Person Re-identification

        **Authors**: *Shengcai Liao, Ling Shao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0f49c89d1e7298bb9930789c8ed59d48-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0f49c89d1e7298bb9930789c8ed59d48-Abstract.html)

        **Abstract**:

        Transformers have recently gained increasing attention in computer vision. However, existing studies mostly use Transformers for feature representation learning, e.g. for image classification and dense predictions, and the generalizability of Transformers is unknown. In this work, we further investigate the possibility of applying Transformers for image matching and metric learning given pairs of images. We find that the Vision Transformer (ViT) and the vanilla Transformer with decoders are not adequate for image matching due to their lack of image-to-image attention. Thus, we further design two naive solutions, i.e. query-gallery concatenation in ViT, and query-gallery cross-attention in the vanilla Transformer. The latter improves the performance, but it is still limited. This implies that the attention mechanism in Transformers is primarily designed for global feature aggregation, which is not naturally suitable for image matching. Accordingly, we propose a new simplified decoder, which drops the full attention implementation with the softmax weighting, keeping only the query-key similarity computation. Additionally, global max pooling and a multilayer perceptron (MLP) head are applied to decode the matching result. This way, the simplified decoder is computationally more efficient, while at the same time more effective for image matching. The proposed method, called TransMatcher, achieves state-of-the-art performance in generalizable person re-identification, with up to 6.1% and 5.7% performance gains in Rank-1 and mAP, respectively, on several popular datasets. Code is available at https://github.com/ShengcaiLiao/QAConv.

        ----

        ## [153] Multi-Objective SPIBB: Seldonian Offline Policy Improvement with Safety Constraints in Finite MDPs

        **Authors**: *Harsh Satija, Philip S. Thomas, Joelle Pineau, Romain Laroche*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0f65caf0a7d00afd2b87c028e88fe931-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0f65caf0a7d00afd2b87c028e88fe931-Abstract.html)

        **Abstract**:

        We study the problem of Safe Policy Improvement (SPI) under constraints in the offline Reinforcement Learning (RL) setting. We consider the scenario where: (i) we have a dataset collected under a known baseline policy, (ii) multiple reward signals are received from the environment inducing as many objectives to optimize. We present an SPI formulation for this RL setting that takes into account the preferences of the algorithmâ€™s user for handling the trade-offs for different reward signals while ensuring that the new policy performs at least as well as the baseline policy along each individual objective. We build on traditional SPI algorithms and propose a novel method based on Safe Policy Iteration with Baseline Bootstrapping (SPIBB, Laroche et al., 2019) that provides high probability guarantees on the performance of the agent in the true environment. We show the effectiveness of our method on a synthetic grid-world safety task as well as in a real-world critical care context to learn a policy for the administration of IV fluids and vasopressors to treat sepsis.

        ----

        ## [154] Is Automated Topic Model Evaluation Broken? The Incoherence of Coherence

        **Authors**: *Alexander Miserlis Hoyle, Pranav Goel, Andrew Hian-Cheong, Denis Peskov, Jordan L. Boyd-Graber, Philip Resnik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0f83556a305d789b1d71815e8ea4f4b0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0f83556a305d789b1d71815e8ea4f4b0-Abstract.html)

        **Abstract**:

        Topic model evaluation, like evaluation of other unsupervised methods, can be contentious. However, the field has coalesced around automated estimates of topic coherence, which rely on the frequency of word co-occurrences in a reference corpus. Contemporary neural topic models surpass classical ones according to these metrics. At the same time, topic model evaluation suffers from a validation gap: automated coherence, developed for classical models, has not been validated using human experimentation for neural models. In addition, a meta-analysis of topic modeling literature reveals a substantial standardization gap in automated topic modeling benchmarks. To address the validation gap, we compare automated coherence with the two most widely accepted human judgment tasks: topic rating and word intrusion. To address the standardization gap, we systematically evaluate a dominant classical model and two state-of-the-art neural models on two commonly used datasets. Automated evaluations declare a winning model when corresponding human evaluations do not, calling into question the validity of fully automatic evaluations independent of human judgments.

        ----

        ## [155] INDIGO: GNN-Based Inductive Knowledge Graph Completion Using Pair-Wise Encoding

        **Authors**: *Shuwen Liu, Bernardo Cuenca Grau, Ian Horrocks, Egor V. Kostylev*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0fd600c953cde8121262e322ef09f70e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0fd600c953cde8121262e322ef09f70e-Abstract.html)

        **Abstract**:

        The aim of knowledge graph (KG) completion is to extend an incomplete KG with missing triples. Popular approaches based on graph embeddings typically work by first representing the KG in a vector space, and then applying a predefined scoring function to the resulting vectors to complete the KG. These approaches work well in transductive settings, where predicted triples involve only constants seen during training; however, they are not applicable in inductive settings, where the KG on which the model was trained is extended with new constants or merged with other KGs. The use of Graph Neural Networks (GNNs) has recently been proposed as a way to overcome these limitations; however, existing approaches do not fully exploit the capabilities of GNNs and still rely on heuristics and ad-hoc scoring functions. In this paper, we propose a novel approach, where the KG is fully encoded into a GNN in a transparent way, and where the predicted triples can be read out directly from the last layer of the GNN without the need for additional components or scoring functions. Our experiments show that our model outperforms state-of-the-art approaches on inductive KG completion benchmarks.

        ----

        ## [156] Do Input Gradients Highlight Discriminative Features?

        **Authors**: *Harshay Shah, Prateek Jain, Praneeth Netrapalli*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/0fe6a94848e5c68a54010b61b3e94b0e-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/0fe6a94848e5c68a54010b61b3e94b0e-Abstract.html)

        **Abstract**:

        Post-hoc gradient-based interpretability methods [Simonyan et al., 2013, Smilkov et al., 2017] that provide instance-specific explanations of model predictions are often based on assumption (A): magnitude of input gradients—gradients of logits with respect to input—noisily highlight discriminative task-relevant features. In this work, we test the validity of assumption (A) using a three-pronged approach:1. We develop an evaluation framework, DiffROAR, to test assumption (A) on four image classification benchmarks. Our results suggest that (i) input gradients of standard models (i.e., trained on original data) may grossly violate (A), whereas (ii) input gradients of adversarially robust models satisfy (A).2. We then introduce BlockMNIST, an MNIST-based semi-real dataset, that by design encodes a priori knowledge of discriminative features. Our analysis on BlockMNIST leverages this information to validate as well as characterize differences between input gradient attributions of standard and robust models.3. Finally, we theoretically prove that our empirical findings hold on a simplified version of the BlockMNIST dataset. Specifically, we prove that input gradients of standard one-hidden-layer MLPs trained on this dataset do not highlight instance-specific signal coordinates, thus grossly violating assumption (A).Our findings motivate the need to formalize and test common assumptions in interpretability in a falsifiable manner [Leavitt and Morcos, 2020]. We believe that the DiffROAR evaluation framework and BlockMNIST-based datasets can serve as sanity checks to audit instance-specific interpretability methods; code and data available at https://github.com/harshays/inputgradients.

        ----

        ## [157] Improving Conditional Coverage via Orthogonal Quantile Regression

        **Authors**: *Shai Feldman, Stephen Bates, Yaniv Romano*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1006ff12c465532f8c574aeaa4461b16-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1006ff12c465532f8c574aeaa4461b16-Abstract.html)

        **Abstract**:

        We develop a method to generate prediction intervals that have a user-specified coverage level across all regions of feature-space, a property called conditional coverage. A typical approach to this task is to estimate the conditional quantiles with quantile regression---it is well-known that this leads to correct coverage in the large-sample limit, although it may not be accurate in finite samples. We find in experiments that traditional quantile regression can have poor conditional coverage. To remedy this, we modify the loss function to promote independence between the size of the intervals and the indicator of a miscoverage event. For the true conditional quantiles, these two quantities are independent (orthogonal), so the modified loss function continues to be valid. Moreover, we empirically show that the modified loss function leads to improved conditional coverage, as evaluated by several metrics. We also introduce two new metrics that check conditional coverage by looking at the strength of the dependence between the interval size and the indicator of miscoverage.

        ----

        ## [158] Minimizing Polarization and Disagreement in Social Networks via Link Recommendation

        **Authors**: *Liwang Zhu, Qi Bao, Zhongzhi Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/101951fe7ebe7bd8c77d14f75746b4bc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/101951fe7ebe7bd8c77d14f75746b4bc-Abstract.html)

        **Abstract**:

        Individual's opinions are fundamentally shaped  and evolved by their interactions with other people, and social phenomena such as disagreement and polarization are now tightly woven into daily life. The quantification and optimization of these concepts have been the subject of much recent research behind a wealth of high-impact data mining applications. In particular, researchers have addressed the question of how such concepts can be optimized by influencing the opinion of a small number of individuals or by designing the network from scratch.Here, rather than  a “design-from-scratch” approach or altering the initial opinion, we study the optimization problem of recommending $k$ new links to minimize the sum of polarization and disagreement in a social network with $n$ nodes and $m$ edges.  We show that our objective function of this combinatorial optimization problem is not submodular, although it is monotone. We propose a simple greedy algorithm with a constant-factor approximation that  solves the problem in cubic running time, and we provide  theoretical analysis of the approximation guarantee for the algorithm. To overcome the computation challenge for large networks, we also provide a fast algorithm with computation complexity $\Otil (mk\eps^{-2})$ for any $\eps>0$,  where the $\Otil (\cdot)$ notation suppresses the ${\rm poly} (\log n)$ factors. Extensive experiments on real datasets demonstrate both the efficiency and effectiveness of our algorithms.

        ----

        ## [159] Adversarial Attacks on Black Box Video Classifiers: Leveraging the Power of Geometric Transformations

        **Authors**: *Shasha Li, Abhishek Aich, Shitong Zhu, M. Salman Asif, Chengyu Song, Amit K. Roy-Chowdhury, Srikanth V. Krishnamurthy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/103303dd56a731e377d01f6a37badae3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/103303dd56a731e377d01f6a37badae3-Abstract.html)

        **Abstract**:

        When compared to the image classification models, black-box adversarial attacks against video classification models have been largely understudied. This could be possible because, with video, the temporal dimension poses significant additional challenges in gradient estimation. Query-efficient black-box attacks rely on effectively estimated gradients towards maximizing the probability of misclassifying the target video. In this work, we demonstrate that such effective gradients can be searched for by parameterizing the temporal structure of the search space with geometric transformations. Specifically, we design a novel iterative algorithm GEOmetric TRAnsformed Perturbations (GEO-TRAP), for attacking video classification models. GEO-TRAP employs standard geometric transformation operations to reduce the search space for effective gradients into searching for a small group of parameters that define these operations. This group of parameters describes the geometric progression of gradients, resulting in a reduced and structured search space. Our algorithm inherently leads to successful perturbations with surprisingly few queries. For example, adversarial examples generated from GEO-TRAP have better attack success rates with ~73.55% fewer queries compared to the state-of-the-art method for video adversarial attacks on the widely used Jester dataset. Overall, our algorithm exposes vulnerabilities of diverse video classification models and achieves new state-of-the-art results under black-box settings on two large datasets.

        ----

        ## [160] Optimal Rates for Random Order Online Optimization

        **Authors**: *Uri Sherman, Tomer Koren, Yishay Mansour*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/107030ca685076c0ed5e054e2c3ed940-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/107030ca685076c0ed5e054e2c3ed940-Abstract.html)

        **Abstract**:

        We study online convex optimization in the random order model, recently proposed by Garber et al. (2020), where the loss functions may be chosen by an adversary, but are then presented to the online algorithm in a uniformly random order. Focusing on the scenario where the cumulative loss function is (strongly) convex, yet individual loss functions are smooth but might be non-convex, we give algorithms that achieve the optimal bounds and significantly outperform the results of Garber et al. (2020), completely removing the dimension dependence and improve their scaling with respect to the strong convexity parameter. Our analysis relies on novel connections between algorithmic stability and generalization for sampling without-replacement analogous to those studied in the with-replacement i.i.d. setting, as well as on a refined average stability analysis of stochastic gradient descent.

        ----

        ## [161] Discrete-Valued Neural Communication

        **Authors**: *Dianbo Liu, Alex Lamb, Kenji Kawaguchi, Anirudh Goyal, Chen Sun, Michael C. Mozer, Yoshua Bengio*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/10907813b97e249163587e6246612e21-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/10907813b97e249163587e6246612e21-Abstract.html)

        **Abstract**:

        Deep learning has advanced from fully connected architectures to structured models organized into components, e.g., the transformer composed of positional elements, modular architectures divided into slots, and graph neural nets made up of nodes. The nature of structured models is that communication among the components has a bottleneck, typically achieved by restricted connectivity and attention. In this work, we further tighten the bottleneck via discreteness of the representations transmitted between components. We hypothesize that this constraint serves as a useful form of inductive bias. Our hypothesis is motivated by past empirical work showing the benefits of discretization in non-structured architectures as well as our own theoretical results showing that discretization increases noise robustness and reduces the underlying dimensionality of the model. Building on an existing technique for discretization from the VQ-VAE, we consider multi-headed discretization with shared codebooks as the output of each architectural component. One motivating intuition is human language in which communication occurs through multiple discrete symbols. This form of communication is hypothesized to facilitate transmission of information between functional components of the brain by providing a common interlingua, just as it does for human-to-human communication. Our experiments show that discrete-valued neural communication (DVNC) substantially improves systematic generalization in a variety of architecturesâ€”transformers, modular architectures, and graph neural networks. We also show that the DVNC is robust to the choice of hyperparameters, making the method useful in practice.

        ----

        ## [162] Skyformer: Remodel Self-Attention with Gaussian Kernel and Nystr\"om Method

        **Authors**: *Yifan Chen, Qi Zeng, Heng Ji, Yun Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/10a7cdd970fe135cf4f7bb55c0e3b59f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/10a7cdd970fe135cf4f7bb55c0e3b59f-Abstract.html)

        **Abstract**:

        Transformers are expensive to train due to the quadratic time and space complexity in the self-attention mechanism. On the other hand, although kernel machines suffer from the same computation bottleneck in pairwise dot products, several approximation schemes have been successfully incorporated to considerably reduce their computational cost without sacrificing too much accuracy. In this work, we leverage the computation methods for kernel machines to alleviate the high computational cost and introduce Skyformer, which replaces the softmax structure with a Gaussian kernel to stabilize the model training and adapts the Nystr√∂m method to a non-positive semidefinite matrix to accelerate the computation. We further conduct theoretical analysis by showing that the matrix approximation error of our proposed method is small in the spectral norm. Experiments on Long Range Arena benchmark show that the proposed method is sufficient in getting comparable or even better performance than the full self-attention while requiring fewer computation resources.

        ----

        ## [163] TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification

        **Authors**: *Zhuchen Shao, Hao Bian, Yang Chen, Yifeng Wang, Jian Zhang, Xiangyang Ji, Yongbing Zhang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)

        **Abstract**:

        Multiple instance learning (MIL) is a powerful tool to solve the weakly supervised classification in whole slide image (WSI) based pathology diagnosis. However, the current MIL methods are usually based on independent and identical distribution hypothesis, thus neglect the correlation among different instances. To address this problem, we proposed a new framework, called correlated MIL, and provided a proof for convergence. Based on this framework, we devised a Transformer based MIL (TransMIL), which explored both morphological and spatial information. The proposed TransMIL can effectively deal with unbalanced/balanced and binary/multiple classification with great visualization and interpretability. We conducted various experiments for three different computational pathology problems and achieved better performance and faster convergence compared with state-of-the-art methods. The test AUC for the binary tumor classification can be up to 93.09% over CAMELYON16 dataset. And the AUC over the cancer subtypes classification can be up to 96.03% and 98.82% over TCGA-NSCLC dataset and TCGA-RCC dataset, respectively. Implementation is available at: https://github.com/szc19990412/TransMIL.

        ----

        ## [164] Multi-view Contrastive Graph Clustering

        **Authors**: *Erlin Pan, Zhao Kang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/10c66082c124f8afe3df4886f5e516e0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/10c66082c124f8afe3df4886f5e516e0-Abstract.html)

        **Abstract**:

        With the explosive growth of information technology, multi-view graph data have become increasingly prevalent and valuable. Most existing multi-view clustering techniques either focus on the scenario of multiple graphs or multi-view attributes. In this paper, we propose a generic framework to cluster multi-view attributed graph data. Specifically, inspired by the success of contrastive learning, we propose multi-view contrastive graph clustering (MCGC) method to learn a consensus graph since the original graph could be noisy or incomplete and is not directly applicable. Our method composes of two key steps: we first filter out the undesirable high-frequency noise while preserving the graph geometric features via graph filtering and obtain a smooth representation of nodes; we then learn a consensus graph regularized by graph contrastive loss. Results on several benchmark datasets show the superiority of our method with respect to state-of-the-art approaches. In particular, our simple approach outperforms existing deep learning-based methods.

        ----

        ## [165] Inverse-Weighted Survival Games

        **Authors**: *Xintian Han, Mark Goldstein, Aahlad Manas Puli, Thomas Wies, Adler J. Perotte, Rajesh Ranganath*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/10fb6cfa4c990d2bad5ddef4f70e8ba2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/10fb6cfa4c990d2bad5ddef4f70e8ba2-Abstract.html)

        **Abstract**:

        Deep models trained through maximum likelihood have achieved state-of-the-art results for survival analysis. Despite this training scheme, practitioners evaluate models under other criteria, such as binary classification losses at a chosen set of time horizons, e.g. Brier score (BS) and Bernoulli log likelihood (BLL). Models trained with maximum likelihood may have poor BS or BLL since maximum likelihood does not directly optimize these criteria. Directly optimizing criteria like BS requires inverse-weighting by the censoring distribution. However, estimating the censoring model under these metrics requires inverse-weighting by the failure distribution. The objective for each model requires the other, but neither are known. To resolve this dilemma, we introduce Inverse-Weighted Survival Games. In these games, objectives for each model are built from re-weighted estimates featuring the other model, where the latter is held fixed during training. When the loss is proper, we show that the games always have the true failure and censoring distributions as a stationary point. This means models in the game do not leave the correct distributions once reached. We construct one case where this stationary point is unique. We show that these games optimize BS on simulations and then apply these principles on real world cancer and critically-ill patient data.

        ----

        ## [166] Generalization Bounds for Meta-Learning via PAC-Bayes and Uniform Stability

        **Authors**: *Alec Farid, Anirudha Majumdar*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1102a326d5f7c9e04fc3c89d0ede88c9-Abstract.html)

        **Abstract**:

        We are motivated by the problem of providing strong generalization guarantees in the context of meta-learning. Existing generalization bounds are either challenging to evaluate or provide vacuous guarantees in even relatively simple settings. We derive a probably approximately correct (PAC) bound for gradient-based meta-learning using two different generalization frameworks in order to deal with the qualitatively different challenges of generalization at the "base" and "meta" levels. We employ bounds for uniformly stable algorithms at the base level and bounds from the PAC-Bayes framework at the meta level. The result of this approach is a novel PAC bound that is tighter when the base learner adapts quickly, which is precisely the goal of meta-learning. We show that our bound provides a tighter guarantee than other bounds on a toy non-convex problem on the unit sphere and a text-based classification example. We also present a practical regularization scheme motivated by the bound in settings where the bound is loose and demonstrate improved performance over baseline techniques.

        ----

        ## [167] Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement

        **Authors**: *Samuel Daulton, Maximilian Balandat, Eytan Bakshy*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html)

        **Abstract**:

        Optimizing multiple competing black-box objectives is a challenging problem in many fields, including science, engineering, and machine learning. Multi-objective Bayesian optimization (MOBO) is a sample-efficient approach for identifying the optimal trade-offs between the objectives. However, many existing methods perform poorly when the observations are corrupted by noise. We propose a novel acquisition function, NEHVI, that overcomes this important practical limitation by applying a Bayesian treatment to the popular expected hypervolume improvement (EHVI) criterion and integrating over this uncertainty in the Pareto frontier. We argue that, even in the noiseless setting, generating multiple candidates in parallel is an incarnation of EHVI with uncertainty in the Pareto frontier and therefore can be addressed using the same underlying technique. Through this lens, we derive a natural parallel variant, qNEHVI, that reduces computational complexity of parallel EHVI from exponential to polynomial with respect to the batch size. qNEHVI is one-step Bayes-optimal for hypervolume maximization in both noisy and noiseless environments, and we show that it can be optimized effectively with gradient-based methods via sample average approximation. Empirically, we demonstrate not only that qNEHVI is substantially more robust to observation noise than existing MOBO approaches, but also that it achieves state-of-the-art optimization performance and competitive wall-times in large-batch environments.

        ----

        ## [168] Evolution Gym: A Large-Scale Benchmark for Evolving Soft Robots

        **Authors**: *Jagdeep Singh Bhatia, Holly Jackson, Yunsheng Tian, Jie Xu, Wojciech Matusik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/118921efba23fc329e6560b27861f0c2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/118921efba23fc329e6560b27861f0c2-Abstract.html)

        **Abstract**:

        Both the design and control of a robot play equally important roles in its task performance. However, while optimal control is well studied in the machine learning and robotics community, less attention is placed on finding the optimal robot design. This is mainly because co-optimizing design and control in robotics is characterized as a challenging problem, and more importantly, a comprehensive evaluation benchmark for co-optimization does not exist. In this paper, we propose Evolution Gym, the first large-scale benchmark for co-optimizing the design and control of soft robots. In our benchmark, each robot is composed of different types of voxels (e.g., soft, rigid, actuators), resulting in a modular and expressive robot design space. Our benchmark environments span a wide range of tasks, including locomotion on various types of terrains and manipulation. Furthermore, we develop several robot co-evolution algorithms by combining state-of-the-art design optimization methods and deep reinforcement learning techniques. Evaluating the algorithms on our benchmark platform, we observe robots exhibiting increasingly complex behaviors as evolution progresses, with the best evolved designs solving many of our proposed tasks. Additionally, even though robot designs are evolved autonomously from scratch without prior knowledge, they often grow to resemble existing natural creatures while outperforming hand-designed robots. Nevertheless, all tested algorithms fail to find robots that succeed in our hardest environments. This suggests that more advanced algorithms are required to explore the high-dimensional design space and evolve increasingly intelligent robots -- an area of research in which we hope Evolution Gym will accelerate progress. Our website with code, environments, documentation, and tutorials is available at http://evogym.csail.mit.edu/.

        ----

        ## [169] On Calibration and Out-of-Domain Generalization

        **Authors**: *Yoav Wald, Amir Feder, Daniel Greenfeld, Uri Shalit*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/118bd558033a1016fcc82560c65cca5f-Abstract.html)

        **Abstract**:

        Out-of-domain (OOD) generalization is a significant challenge for machine learning models. Many techniques have been proposed to overcome this challenge, often focused on learning models with certain invariance properties. In this work, we draw a link between OOD performance and model calibration, arguing that calibration across multiple domains can be viewed as a special case of an invariant representation leading to better OOD generalization. Specifically, we show that under certain conditions, models which achieve \emph{multi-domain calibration} are provably free of spurious correlations. This leads us to propose multi-domain calibration as a measurable and trainable surrogate for the OOD performance of a classifier. We therefore introduce methods that are easy to apply and allow practitioners to improve multi-domain calibration by training or modifying an existing model, leading to better performance on unseen domains. Using four datasets from the recently proposed WILDS OOD benchmark, as well as the Colored MNIST, we demonstrate that training or tuning models so they are calibrated across multiple domains leads to significantly improved performance on unseen test domains. We believe this intriguing connection between calibration and OOD generalization is promising from both a practical and theoretical point of view.

        ----

        ## [170] On the Convergence and Sample Efficiency of Variance-Reduced Policy Gradient Method

        **Authors**: *Junyu Zhang, Chengzhuo Ni, Zheng Yu, Csaba Szepesvári, Mengdi Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/11c484ea9305ea4c7bb6b2e6d570d466-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/11c484ea9305ea4c7bb6b2e6d570d466-Abstract.html)

        **Abstract**:

        Policy gradient (PG) gives rise to a rich class of reinforcement learning (RL) methods. Recently, there has been an emerging trend to augment the existing PG methods such as REINFORCE by the \emph{variance reduction} techniques.  However, all existing variance-reduced PG methods heavily rely on an uncheckable importance weight assumption made for every single iteration of the algorithms. In this paper, a simple gradient truncation mechanism is proposed to address this issue. Moreover, we design a Truncated Stochastic Incremental Variance-Reduced Policy Gradient (TSIVR-PG) method, which is able to maximize not only a cumulative sum of rewards but also a general utility function over a policy's long-term visiting distribution.  We show an $\tilde{\mathcal{O}}(\epsilon^{-3})$ sample complexity for TSIVR-PG to find an $\epsilon$-stationary policy. By assuming the \emph{overparameterization} of policy and exploiting the \emph{hidden convexity} of the problem, we further show that TSIVR-PG converges to global $\epsilon$-optimal policy with $\tilde{\mathcal{O}}(\epsilon^{-2})$ samples.

        ----

        ## [171] Circa: Stochastic ReLUs for Private Deep Learning

        **Authors**: *Zahra Ghodsi, Nandan Kumar Jha, Brandon Reagen, Siddharth Garg*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/11eba2991cc62daa4a85be5c0cfdae97-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/11eba2991cc62daa4a85be5c0cfdae97-Abstract.html)

        **Abstract**:

        The simultaneous rise of machine learning as a service and concerns over user privacy have increasingly motivated the need for private inference (PI). While recent work demonstrates PI is possible using cryptographic primitives, the computational overheads render it impractical. State-of-art deep networks are inadequate in this context because the source of slowdown in PI stems from the ReLU operations whereas optimizations for plaintext inference focus on reducing FLOPs. In this paper we re-think ReLU computations and propose optimizations for PI tailored to properties of neural networks. Specifically, we reformulate ReLU as an approximate sign test and introduce a novel truncation method for the sign test that significantly reduces the cost per ReLU. These optimizations result in a specific type of stochastic ReLU. The key observation is that the stochastic fault behavior is well suited for the fault-tolerant properties of neural network inference. Thus, we provide significant savings without impacting accuracy. We collectively call the optimizations Circa and demonstrate improvements of up to 4.7$\times$ storage and 3$\times$ runtime over baseline implementations; we further show that Circa can be used on top of recent PI optimizations to obtain 1.8$\times$ additional speedup.

        ----

        ## [172] Reinforcement Learning in Reward-Mixing MDPs

        **Authors**: *Jeongyeol Kwon, Yonathan Efroni, Constantine Caramanis, Shie Mannor*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/11f9e78e4899a78dedd439fc583b6693-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/11f9e78e4899a78dedd439fc583b6693-Abstract.html)

        **Abstract**:

        Learning a near optimal policy in a partially observable system remains an elusive challenge in contemporary reinforcement learning. In this work, we consider episodic reinforcement learning in a reward-mixing Markov decision process (MDP). There, a reward function is drawn from one of $M$ possible reward models at the beginning of every episode, but the identity of the chosen reward model is not revealed to the agent. Hence, the latent state space, for which the dynamics are Markovian, is not given to the agent. We study the problem of learning a near optimal policy for two reward-mixing MDPs. Unlike existing approaches that rely on strong assumptions on the dynamics, we make no assumptions and study the problem in full generality. Indeed, with no further assumptions, even for two switching reward-models, the problem requires several new ideas beyond existing algorithmic and analysis techniques for efficient exploration. We provide the first polynomial-time algorithm that finds an $\epsilon$-optimal policy after exploring $\tilde{O}(poly(H,\epsilon^{-1}) \cdot S^2 A^2)$ episodes, where $H$ is time-horizon and $S, A$ are the number of states and actions respectively. This is the first efficient algorithm that does not require any assumptions in partially observed environments where the observation space is smaller than the latent state space.

        ----

        ## [173] A Gang of Adversarial Bandits

        **Authors**: *Mark Herbster, Stephen Pasteris, Fabio Vitale, Massimiliano Pontil*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/124461dcd3571e6674ec4e0e140cc298-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/124461dcd3571e6674ec4e0e140cc298-Abstract.html)

        **Abstract**:

        We consider running multiple instances of multi-armed bandit (MAB) problems in parallel. A main motivation for this study are online recommendation systems, in which each of $N$ users is associated with a MAB problem and the goal is to exploit users' similarity in order to learn users' preferences to $K$ items more efficiently. We consider the adversarial MAB setting, whereby an adversary is free to choose which user and which loss to present to the learner during the learning process. Users are in a social network and the learner is aided by a-priori knowledge of the strengths of the social links between all pairs of users. It is assumed that if the social link between two users is strong then they tend to share the same action. The regret is measured relative to an arbitrary function which maps users to actions. The smoothness of the function is captured by a resistance-based dispersion measure $\Psi$. We present two learning algorithms, GABA-I and GABA-II, which exploit the network structure to bias towards functions of low $\Psi$  values. We show that GABA-I has an expected regret bound of $\mathcal{O}(\sqrt{\ln(NK/\Psi)\Psi KT})$ and per-trial time complexity of $\mathcal{O}(K\ln(N))$, whilst GABA-II has a weaker $\mathcal{O}(\sqrt{\ln(N/\Psi)\ln(NK/\Psi)\Psi KT})$ regret, but a better $\mathcal{O}(\ln(K)\ln(N))$ per-trial time complexity. We highlight improvements of both algorithms over running independent standard MABs across users.

        ----

        ## [174] Explaining Hyperparameter Optimization via Partial Dependence Plots

        **Authors**: *Julia Moosbauer, Julia Herbinger, Giuseppe Casalicchio, Marius Lindauer, Bernd Bischl*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/12ced2db6f0193dda91ba86224ea1cd8-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/12ced2db6f0193dda91ba86224ea1cd8-Abstract.html)

        **Abstract**:

        Automated hyperparameter optimization (HPO) can support practitioners to obtain peak performance in machine learning models.However, there is often a lack of valuable insights into the effects of different hyperparameters on the final model performance.This lack of explainability makes it difficult to trust and understand the automated HPO process and its results.We suggest using interpretable machine learning (IML) to gain insights from the experimental data obtained during HPO with Bayesian optimization (BO).BO tends to focus on promising regions with potential high-performance configurations and thus induces a sampling bias.Hence, many IML techniques, such as the partial dependence plot (PDP), carry the risk of generating biased interpretations.By leveraging the posterior uncertainty of the BO surrogate model, we introduce a variant of the PDP with estimated confidence bands.We propose to partition the hyperparameter space to obtain more confident and reliable PDPs in relevant sub-regions.In an experimental study, we provide quantitative evidence for the increased quality of the PDPs within sub-regions.

        ----

        ## [175] Robustifying Algorithms of Learning Latent Trees with Vector Variables

        **Authors**: *Fengzhuo Zhang, Vincent Y. F. Tan*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/12e086066892a311b752673a28583d3f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/12e086066892a311b752673a28583d3f-Abstract.html)

        **Abstract**:

        We consider learning the structures of Gaussian latent tree models with vector observations when a subset of them are arbitrarily corrupted. First, we present the sample complexities of Recursive Grouping (RG) and Chow-Liu Recursive Grouping (CLRG) without the assumption that the effective depth is bounded in the number of observed nodes, significantly generalizing the results in Choi et al. (2011). We show that Chow-Liu initialization in CLRG greatly reduces the sample complexity of RG from being exponential in the diameter of the tree to only logarithmic in the diameter for the hidden Markov model (HMM). Second, we robustify RG, CLRG, Neighbor Joining (NJ) and Spectral NJ (SNJ) by using the truncated inner product. These robustified algorithms can tolerate a number of corruptions up to the square root of the number of clean samples. Finally, we derive the first known instance-dependent impossibility result for structure learning of latent trees. The optimalities of the robust version of CLRG and NJ are verified by comparing their sample complexities and the impossibility result.

        ----

        ## [176] Representation Learning on Spatial Networks

        **Authors**: *Zheng Zhan, Liang Zhao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/12e35d9186dd72fe62fd039385890b9c-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/12e35d9186dd72fe62fd039385890b9c-Abstract.html)

        **Abstract**:

        Spatial networks are networks for which the nodes and edges are constrained by geometry and embedded in real space, which has crucial effects on their topological properties. Although tremendous success has been achieved in spatial and network representation separately in recent years, there exist very little works on the representation of spatial networks. Extracting powerful representations from spatial networks requires the development of appropriate tools to uncover the pairing of both spatial and network information in the appearance of node permutation invariant, and rotation and translation invariant. Hence it can not be modeled merely with either spatial or network models individually. To address these challenges, this paper proposes a generic framework for spatial network representation learning. Specifically, a provably information-lossless and roto-translation invariant representation of spatial information on networks is presented. Then a higher-order spatial network convolution operation that adapts to our proposed representation is introduced. To ensure efficiency, we also propose a new approach that relied on sampling random spanning trees to reduce the time and memory complexity from $O(N^3)$ to $O(N)$. We demonstrate the strength of our proposed framework through extensive experiments on both synthetic and real-world datasets. The code for the proposed model is available at \url{https://github.com/rollingstonezz/SGMP_code}.

        ----

        ## [177] Continuous-time edge modelling using non-parametric point processes

        **Authors**: *Xuhui Fan, Bin Li, Feng Zhou, Scott A. Sisson*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1301962d8b7bd03fffaa27119aa7fc2b-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1301962d8b7bd03fffaa27119aa7fc2b-Abstract.html)

        **Abstract**:

        The mutually-exciting Hawkes process (ME-HP) is a natural choice to model reciprocity, which is an important attribute of continuous-time edge (dyadic) data. However, existing ways of implementing the ME-HP for such data are either inflexible, as the exogenous (background) rate functions are typically constant and the endogenous (excitation) rate functions are specified parametrically, or inefficient, as inference usually relies on Markov chain Monte Carlo methods with high computational costs. To address these limitations, we discuss various approaches to model design, and develop three variants of non-parametric point processes for continuous-time edge modelling (CTEM). The resulting models are highly adaptable as they generate intensity functions through sigmoidal Gaussian processes, and so provide greater modelling flexibility than parametric forms. The models are implemented via a fast variational inference method enabled by a novel edge modelling construction. The superior performance of the proposed CTEM models is demonstrated through extensive experimental evaluations on four real-world continuous-time edge data sets.

        ----

        ## [178] Deep inference of latent dynamics with spatio-temporal super-resolution using selective backpropagation through time

        **Authors**: *Feng Zhu, Andrew R. Sedler, Harrison A. Grier, Nauman Ahad, Mark A. Davenport, Matthew T. Kaufman, Andrea Giovannucci, Chethan Pandarinath*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1325cdae3b6f0f91a1b629307bf2d498-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1325cdae3b6f0f91a1b629307bf2d498-Abstract.html)

        **Abstract**:

        Modern neural interfaces allow access to the activity of up to a million neurons within brain circuits. However, bandwidth limits often create a trade-off between greater spatial sampling (more channels or pixels) and the temporal frequency of sampling. Here we demonstrate that it is possible to obtain spatio-temporal super-resolution in neuronal time series by exploiting relationships among neurons, embedded in latent low-dimensional population dynamics. Our novel neural network training strategy, selective backpropagation through time (SBTT), enables learning of deep generative models of latent dynamics from data in which the set of observed variables changes at each time step. The resulting models are able to infer activity for missing samples by combining observations with learned latent dynamics. We test SBTT applied to sequential autoencoders and demonstrate more efficient and higher-fidelity characterization of neural population dynamics in electrophysiological and calcium imaging data. In electrophysiology, SBTT enables accurate inference of neuronal population dynamics with lower interface bandwidths, providing an avenue to significant power savings for implanted neuroelectronic interfaces. In applications to two-photon calcium imaging, SBTT accurately uncovers high-frequency temporal structure underlying neural population activity, substantially outperforming the current state-of-the-art. Finally, we demonstrate that performance could be further improved by using limited, high-bandwidth sampling to pretrain dynamics models, and then using SBTT to adapt these models for sparsely-sampled data.

        ----

        ## [179] Memory-efficient Patch-based Inference for Tiny Deep Learning

        **Authors**: *Ji Lin, Wei-Ming Chen, Han Cai, Chuang Gan, Song Han*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1371bccec2447b5aa6d96d2a540fb401-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1371bccec2447b5aa6d96d2a540fb401-Abstract.html)

        **Abstract**:

        Tiny deep learning on microcontroller units (MCUs) is challenging due to the limited memory size. We find that the memory bottleneck is due to the imbalanced memory distribution in convolutional neural network (CNN) designs:  the first several blocks have an order of magnitude larger memory usage than the rest of the network. To alleviate this issue, we propose a generic patch-by-patch inference scheduling, which operates only on a small spatial region of the feature map and significantly cuts down the peak memory. However, naive implementation brings overlapping patches and computation overhead. We further propose receptive field redistribution to shift the receptive field and FLOPs to the later stage and reduce the computation overhead. Manually redistributing the receptive field is difficult. We automate the process with neural architecture search to jointly optimize the neural architecture and inference scheduling, leading to MCUNetV2. Patch-based inference effectively reduces the peak memory usage of existing networks by4-8Ã—.  Co-designed with neural networks, MCUNetV2 sets a record ImageNetaccuracy on MCU (71.8%) and achieves >90% accuracy on the visual wake words dataset under only 32kB SRAM. MCUNetV2 also unblocks object detection on tiny devices, achieving 16.9% higher mAP on Pascal VOC compared to the state-of-the-art result. Our study largely addressed the memory bottleneck in tinyML and paved the way for various vision applications beyond image classification.

        ----

        ## [180] Self-Interpretable Model with Transformation Equivariant Interpretation

        **Authors**: *Yipei Wang, Xiaoqian Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1387a00f03b4b423e63127b08c261bdc-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1387a00f03b4b423e63127b08c261bdc-Abstract.html)

        **Abstract**:

        With the proliferation of machine learning applications in the real world, the demand for explaining machine learning predictions continues to grow especially in high-stakes fields. Recent studies have found that interpretation methods can be sensitive and unreliable, where the interpretations can be disturbed by perturbations or transformations of input data. To address this issue, we propose to learn robust interpretation through transformation equivariant regularization in a self-interpretable model. The resulting model is capable of capturing valid interpretation that is equivariant to geometric transformations. Moreover, since our model is self-interpretable, it enables faithful interpretations that reflect the true predictive mechanism. Unlike existing self-interpretable models, which usually sacrifice expressive power for the sake of interpretation quality, our model preserves the high expressive capability comparable to the state-of-the-art deep learning models in complex tasks, while providing visualizable and faithful high-quality interpretation. We compare with various related methods and validate the interpretation quality and consistency of our model.

        ----

        ## [181] Solving Min-Max Optimization with Hidden Structure via Gradient Descent Ascent

        **Authors**: *Emmanouil V. Vlatakis-Gkaragkounis, Lampros Flokas, Georgios Piliouras*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/13bf4a96378f3854bcd9792d132eff9f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/13bf4a96378f3854bcd9792d132eff9f-Abstract.html)

        **Abstract**:

        Many recent AI architectures are inspired by zero-sum games, however, the behavior of their dynamics is still not well understood. Inspired by this, we study standard gradient descent ascent (GDA) dynamics in a specific class of non-convex non-concave zero-sum games, that we call hidden zero-sum games. In this class, players control the inputs of smooth but possibly non-linear functions whose outputs are being applied as inputs to a convex-concave game. Unlike general zero-sum games, these games have a well-defined notion of solution; outcomes that implement the von-Neumann equilibrium of the ``hidden" convex-concave game. We provide conditions under which vanilla GDA provably converges not merely to local Nash, but the actual von-Neumann solution. If the hidden game lacks strict convexity properties, GDA may fail to converge to any equilibrium, however, by applying standard regularization techniques we can prove convergence to a von-Neumann solution of a slightly perturbed zero-sum game. Our convergence results are non-local despite working in the setting of non-convex non-concave games. Critically, under proper assumptions we combine the Center-Stable Manifold Theorem along with novel type of initialization dependent Lyapunov functions to prove that almost all initial conditions converge to the solution. Finally, we discuss diverse applications of our framework ranging from generative adversarial networks to evolutionary biology.

        ----

        ## [182] Preserved central model for faster bidirectional compression in distributed settings

        **Authors**: *Constantin Philippenko, Aymeric Dieuleveut*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/13d63838ef1fb6f34ca2dc6821c60e49-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/13d63838ef1fb6f34ca2dc6821c60e49-Abstract.html)

        **Abstract**:

        We develop a new approach to tackle communication constraints in a distributed learning problem with a central server. We propose and analyze a new algorithm that performs bidirectional compression and achieves the same convergence rate as algorithms using only uplink (from the local workers to the central server) compression. To obtain this improvement, we design MCM, an algorithm such that the downlink compression only impacts local models, while the global model is preserved. As a result, and contrary to previous works, the gradients on local servers are computed on perturbed models. Consequently, convergence proofs are more challenging and require a precise control of this perturbation. To ensure it, MCM additionally combines model compression with a memory mechanism. This analysis opens new doors, e.g. incorporating worker dependent randomized-models and partial participation.

        ----

        ## [183] Understanding Instance-based Interpretability of Variational Auto-Encoders

        **Authors**: *Zhifeng Kong, Kamalika Chaudhuri*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/13d7dc096493e1f77fb4ccf3eaf79df1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/13d7dc096493e1f77fb4ccf3eaf79df1-Abstract.html)

        **Abstract**:

        Instance-based interpretation methods have been widely studied for supervised learning methods as they help explain how black box neural networks predict. However, instance-based interpretations remain ill-understood in the context of unsupervised learning. In this paper, we investigate influence functions [Koh and Liang, 2017], a popular instance-based interpretation method, for a class of deep generative models called variational auto-encoders (VAE). We formally frame the counter-factual question answered by influence functions in this setting, and through theoretical analysis, examine what they reveal about the impact of training samples on classical unsupervised learning methods. We then introduce VAE- TracIn, a computationally efficient and theoretically sound solution based on Pruthi et al. [2020], for VAEs. Finally, we evaluate VAE-TracIn on several real world datasets with extensive quantitative and qualitative analysis.

        ----

        ## [184] Voxel-based 3D Detection and Reconstruction of Multiple Objects from a Single Image

        **Authors**: *Feng Liu, Xiaoming Liu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1415db70fe9ddb119e23e9b2808cde38-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1415db70fe9ddb119e23e9b2808cde38-Abstract.html)

        **Abstract**:

        Inferring 3D locations and shapes of multiple objects from a single 2D image is a long-standing objective of computer vision. Most of the existing works either predict one of these 3D properties or focus on solving both for a single object. One fundamental challenge lies in how to learn an effective representation of the image that is well-suited for 3D detection and reconstruction. In this work, we propose to learn a regular grid of 3D voxel features from the input image which is aligned with 3D scene space via a 3D feature lifting operator. Based on the 3D voxel features, our novel CenterNet-3D detection head formulates the 3D detection as keypoint detection in the 3D space. Moreover, we devise an efficient coarse-to-fine reconstruction module, including coarse-level voxelization and a novel local PCA-SDF shape representation, which enables fine detail reconstruction and two orders of magnitude faster inference than prior methods. With complementary supervision from both 3D detection and reconstruction, one enables the 3D voxel features to be geometry and context preserving, benefiting both tasks. The effectiveness of our approach is demonstrated through 3D detection and reconstruction on single-object and multiple-object scenarios.

        ----

        ## [185] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization

        **Authors**: *Yusuke Iwasawa, Yutaka Matsuo*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1415fe9fea0fa1e45dddcff5682239a0-Abstract.html)

        **Abstract**:

        This paper presents a new algorithm for domain generalization (DG), \textit{test-time template adjuster (T3A)}, aiming to robustify a model to unknown distribution shift. Unlike existing methods that focus on \textit{training phase}, our method focuses \textit{test phase}, i.e., correcting its prediction by itself during test time. Specifically, T3A adjusts a trained linear classifier (the last layer of deep neural networks) with the following procedure:  (1) compute a pseudo-prototype representation for each class using online unlabeled data augmented by the base classifier trained in the source domains, (2) and then classify each sample based on its distance to the pseudo-prototypes. T3A is back-propagation-free and modifies only the linear layer; therefore, the increase in computational cost during inference is negligible and avoids the catastrophic failure might caused by stochastic optimization. Despite its simplicity, T3A can leverage knowledge about the target domain by using off-the-shelf test-time data and improve performance. We tested our method on four domain generalization benchmarks, namely PACS, VLCS, OfficeHome, and TerraIncognita, along with various backbone networks including ResNet18, ResNet50, Big Transfer (BiT), Vision Transformers (ViT), and MLP-Mixer. The results show T3A stably improves performance on unseen domains across choices of backbone networks, and outperforms existing domain generalization methods.

        ----

        ## [186] Luna: Linear Unified Nested Attention

        **Authors**: *Xuezhe Ma, Xiang Kong, Sinong Wang, Chunting Zhou, Jonathan May, Hao Ma, Luke Zettlemoyer*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/14319d9cfc6123106878dc20b94fbaf3-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/14319d9cfc6123106878dc20b94fbaf3-Abstract.html)

        **Abstract**:

        The quadratic computational and memory complexities of the Transformer's attention mechanism have limited its scalability for modeling long sequences.  In this paper, we propose Luna, a linear unified nested attention mechanism that approximates softmax attention with two nested linear attention functions, yielding only linear (as opposed to quadratic) time and space complexity. Specifically, with the first attention function, Luna packs the input sequence into a sequence of fixed length. Then, the packed sequence is unpacked using the second attention function. As compared to a more traditional attention mechanism, Luna introduces an additional sequence with a fixed length as input and an additional corresponding output, which allows Luna to perform attention operation linearly, while also storing adequate contextual information. We perform extensive evaluations on three benchmarks of sequence modeling tasks: long-context sequence modelling, neural machine translation and masked language modeling for large-scale pretraining. Competitive or even better experimental results demonstrate both the effectiveness and efficiency of Luna compared to a variety of strong baseline methods including the full-rank attention and other efficient sparse and dense attention methods.

        ----

        ## [187] Iterative Causal Discovery in the Possible Presence of Latent Confounders and Selection Bias

        **Authors**: *Raanan Y. Rohekar, Shami Nisimov, Yaniv Gurwicz, Gal Novik*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/144a3f71a03ab7c4f46f9656608efdb2-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/144a3f71a03ab7c4f46f9656608efdb2-Abstract.html)

        **Abstract**:

        We present a sound and complete algorithm, called iterative causal discovery (ICD), for recovering causal graphs in the presence of latent confounders and selection bias. ICD relies on the causal Markov and faithfulness assumptions and recovers the equivalence class of the underlying causal graph. It starts with a complete graph, and consists of a single iterative stage that gradually refines this graph by identifying conditional independence (CI) between connected nodes. Independence and causal relations entailed after any iteration are correct, rendering ICD anytime. Essentially, we tie the size of the CI conditioning set to its distance on the graph from the tested nodes, and increase this value in the successive iteration. Thus, each iteration refines a graph that was recovered by previous iterations having smaller conditioning sets---a higher statistical power---which contributes to stability. We demonstrate empirically that ICD requires significantly fewer CI tests and learns more accurate causal graphs compared to FCI, FCI+, and RFCI algorithms.

        ----

        ## [188] Hindsight Task Relabelling: Experience Replay for Sparse Reward Meta-RL

        **Authors**: *Charles Packer, Pieter Abbeel, Joseph E. Gonzalez*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1454ca2270599546dfcd2a3700e4d2f1-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1454ca2270599546dfcd2a3700e4d2f1-Abstract.html)

        **Abstract**:

        Meta-reinforcement learning (meta-RL) has proven to be a successful framework for leveraging experience from prior tasks to rapidly learn new related tasks, however, current meta-RL approaches struggle to learn in sparse reward environments. Although existing meta-RL algorithms can learn strategies for adapting to new sparse reward tasks, the actual adaptation strategies are learned using hand-shaped reward functions, or require simple environments where random exploration is sufficient to encounter sparse reward. In this paper we present a formulation of hindsight relabelling for meta-RL, which relabels experience during meta-training to enable learning to learn entirely using sparse reward. We demonstrate the effectiveness of our approach on a suite of challenging sparse reward environments that previously required dense reward during meta-training to solve. Our approach solves these environments using the true sparse reward function, with performance comparable to training with a proxy dense reward function.

        ----

        ## [189] A Bayesian-Symbolic Approach to Reasoning and Learning in Intuitive Physics

        **Authors**: *Kai Xu, Akash Srivastava, Dan Gutfreund, Felix Sosa, Tomer D. Ullman, Josh Tenenbaum, Charles Sutton*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/147540e129e096fa91700e9db6588354-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/147540e129e096fa91700e9db6588354-Abstract.html)

        **Abstract**:

        Humans can reason about intuitive physics in fully or partially observed environments even after being exposed to a very limited set of observations. This sample-efficient intuitive physical reasoning is considered a core domain of human common sense knowledge. One hypothesis to explain this remarkable capacity, posits that humans quickly learn approximations to the laws of physics that govern the dynamics of the environment. In this paper, we propose a Bayesian-symbolic framework (BSP) for physical reasoning and learning that is close to human-level sample-efficiency and accuracy. In BSP, the environment is represented by a top-down generative model of entities, which are assumed to interact with each other under unknown force laws over their latent and observed properties. BSP models each of these entities as random variables, and uses Bayesian inference to estimate their unknown properties. For learning the unknown forces, BSP leverages symbolic regression on a novel grammar of Newtonian physics in a bilevel optimization setup. These inference and regression steps are performed in an iterative manner using expectation-maximization, allowing BSP to simultaneously learn force laws while maintaining uncertainty over entity properties. We show that BSP is more sample-efficient compared to neural alternatives on controlled synthetic datasets, demonstrate BSP's applicability to real-world common sense scenes and study BSP's performance on tasks previously used to study human physical reasoning.

        ----

        ## [190] Associating Objects with Transformers for Video Object Segmentation

        **Authors**: *Zongxin Yang, Yunchao Wei, Yi Yang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/147702db07145348245dc5a2f2fe5683-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/147702db07145348245dc5a2f2fe5683-Abstract.html)

        **Abstract**:

        This paper investigates how to realize better and more efficient embedding learning to tackle the semi-supervised video object segmentation under challenging multi-object scenarios. The state-of-the-art methods learn to decode features with a single positive object and thus have to match and segment each target separately under multi-object scenarios, consuming multiple times computing resources. To solve the problem, we propose an Associating Objects with Transformers (AOT) approach to match and decode multiple objects uniformly. In detail, AOT employs an identification mechanism to associate multiple targets into the same high-dimensional embedding space. Thus, we can simultaneously process multiple objects' matching and segmentation decoding as efficiently as processing a single object. For sufficiently modeling multi-object association, a Long Short-Term Transformer is designed for constructing hierarchical matching and propagation. We conduct extensive experiments on both multi-object and single-object benchmarks to examine AOT variant networks with different complexities. Particularly, our R50-AOT-L outperforms all the state-of-the-art competitors on three popular benchmarks, i.e., YouTube-VOS (84.1% J&F), DAVIS 2017 (84.9%), and DAVIS 2016 (91.1%), while keeping more than 3X faster multi-object run-time. Meanwhile, our AOT-T can maintain real-time multi-object speed on the above benchmarks. Based on AOT, we ranked 1st in the 3rd Large-scale VOS Challenge.

        ----

        ## [191] Automatic Symmetry Discovery with Lie Algebra Convolutional Network

        **Authors**: *Nima Dehmamy, Robin Walters, Yanchen Liu, Dashun Wang, Rose Yu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/148148d62be67e0916a833931bd32b26-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/148148d62be67e0916a833931bd32b26-Abstract.html)

        **Abstract**:

        Existing equivariant neural networks require prior knowledge of the symmetry group and discretization for continuous groups. We propose to work with Lie algebras (infinitesimal generators) instead of Lie groups. Our model, the Lie algebra convolutional network (L-conv) can automatically discover symmetries and does not require discretization of the group. We show that L-conv can serve as a building block to construct any group equivariant feedforward architecture. Both CNNs and Graph Convolutional Networks can be expressed as L-conv with appropriate groups. We discover direct connections between L-conv and physics: (1) group invariant loss generalizes field theory (2) Euler-Lagrange equation measures the robustness, and (3) equivariance leads to conservation laws and Noether current. These connections open up new avenues for designing more general equivariant networks and applying them to important problems in physical sciences.

        ----

        ## [192] Zero Time Waste: Recycling Predictions in Early Exit Neural Networks

        **Authors**: *Maciej Wolczyk, Bartosz Wójcik, Klaudia Balazy, Igor T. Podolak, Jacek Tabor, Marek Smieja, Tomasz Trzcinski*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/149ef6419512be56a93169cd5e6fa8fd-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/149ef6419512be56a93169cd5e6fa8fd-Abstract.html)

        **Abstract**:

        The problem of reducing processing time of large deep learning models is a fundamental challenge in many real-world applications. Early exit methods strive towards this goal by attaching additional Internal Classifiers (ICs) to intermediate layers of a neural network. ICs can quickly return predictions for easy examples and, as a result, reduce the average inference time of the whole model. However, if a particular IC does not decide to return an answer early, its predictions are discarded, with its computations effectively being wasted. To solve this issue, we introduce Zero Time Waste (ZTW), a novel approach in which each IC reuses predictions returned by its predecessors by (1) adding direct connections between ICs and (2) combining previous outputs in an ensemble-like manner. We conduct extensive experiments across various datasets and architectures to demonstrate that ZTW achieves a significantly better accuracy vs. inference time trade-off than other recently proposed early exit methods.

        ----

        ## [193] On Model Calibration for Long-Tailed Object Detection and Instance Segmentation

        **Authors**: *Tai-Yu Pan, Cheng Zhang, Yandong Li, Hexiang Hu, Dong Xuan, Soravit Changpinyo, Boqing Gong, Wei-Lun Chao*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/14ad095ecc1c3e1b87f3c522836e9158-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/14ad095ecc1c3e1b87f3c522836e9158-Abstract.html)

        **Abstract**:

        Vanilla models for object detection and instance segmentation suffer from the heavy bias toward detecting frequent objects in the long-tailed setting. Existing methods address this issue mostly during training, e.g., by re-sampling or re-weighting. In this paper, we investigate a largely overlooked approach --- post-processing calibration of confidence scores. We propose NorCal, Normalized Calibration for long-tailed object detection and instance segmentation, a simple and straightforward recipe that reweighs the predicted scores of each class by its training sample size. We show that separately handling the background class and normalizing the scores over classes for each proposal are keys to achieving superior performance. On the LVIS dataset, NorCal can effectively improve nearly all the baseline models not only on rare classes but also on common and frequent classes.  Finally, we conduct extensive analysis and ablation studies to offer insights into various modeling choices and mechanisms of our approach. Our code is publicly available at https://github.com/tydpan/NorCal.

        ----

        ## [194] ReSSL: Relational Self-Supervised Learning with Weak Augmentation

        **Authors**: *Mingkai Zheng, Shan You, Fei Wang, Chen Qian, Changshui Zhang, Xiaogang Wang, Chang Xu*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/14c4f36143b4b09cbc320d7c95a50ee7-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/14c4f36143b4b09cbc320d7c95a50ee7-Abstract.html)

        **Abstract**:

        Self-supervised Learning (SSL) including the mainstream contrastive learning has achieved great success in learning visual representations without data annotations. However, most of methods mainly focus on the instance level information (\ie, the different augmented images of the same instance should have the same feature or cluster into the same class), but there is a lack of attention on the relationships between different instances. In this paper, we introduced a novel SSL paradigm, which we term as relational self-supervised learning  (ReSSL) framework that learns representations by modeling the relationship between different instances. Specifically, our proposed method employs sharpened distribution of pairwise similarities among different instances as \textit{relation} metric, which is thus utilized to match the feature embeddings of different augmentations. Moreover, to boost the performance, we argue that weak augmentations matter to represent a more reliable relation, and leverage momentum strategy for practical efficiency. Experimental results show that our proposed ReSSL significantly outperforms the previous state-of-the-art algorithms in terms of both performance and training efficiency.

        ----

        ## [195] Learning to See by Looking at Noise

        **Authors**: *Manel Baradad Jurjo, Jonas Wulff, Tongzhou Wang, Phillip Isola, Antonio Torralba*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/14f2ebeab937ca128186e7ba876faef9-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/14f2ebeab937ca128186e7ba876faef9-Abstract.html)

        **Abstract**:

        Current vision systems are trained on huge datasets, and these datasets come with costs: curation is expensive, they inherit human biases, and there are concerns over privacy and usage rights. To counter these costs, interest has surged in learning from cheaper data sources, such as unlabeled images. In this paper we go a step further and ask if we can do away with real image datasets entirely, instead learning from procedural noise processes. We investigate a suite of image generation models that produce images from simple random processes. These are then used as training data for a visual representation learner with a contrastive loss. In particular, we study statistical image models, randomly initialized deep generative models, and procedural graphics models.Our findings show that it is important for the noise to capture certain structural properties of real data but that good performance can be achieved even with processes that are far from realistic. We also find that diversity is a key property to learn good representations.

        ----

        ## [196] Explicit loss asymptotics in the gradient descent training of neural networks

        **Authors**: *Maksim Velikanov, Dmitry Yarotsky*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/14faf969228fc18fcd4fcf59437b0c97-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/14faf969228fc18fcd4fcf59437b0c97-Abstract.html)

        **Abstract**:

        Current theoretical results on optimization trajectories of neural networks trained by gradient descent typically have the form of rigorous but potentially loose bounds on the loss values. In the present work we take a different approach and show that the learning trajectory of a wide network in a lazy training regime can be characterized by an explicit asymptotic at large training times. Specifically, the leading term in the asymptotic expansion of the loss behaves as a power law $L(t) \sim C t^{-\xi}$ with exponent $\xi$ expressed only through the data dimension, the smoothness of the activation function, and the class of function being approximated. Our results are based on spectral analysis of the integral operator representing the linearized evolution of a large network trained on the expected loss. Importantly, the techniques we employ do not require a specific form of the data distribution, for example Gaussian, thus making our findings sufficiently universal.

        ----

        ## [197] Test-Time Personalization with a Transformer for Human Pose Estimation

        **Authors**: *Yizhuo Li, Miao Hao, Zonglin Di, Nitesh B. Gundavarapu, Xiaolong Wang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1517c8664be296f0d87d9e5fc54fdd60-Abstract.html)

        **Abstract**:

        We propose to personalize a 2D human pose estimator given a set of test images of a person without using any manual annotations. While there is a significant advancement in human pose estimation, it is still very challenging for a model to generalize to different unknown environments and unseen persons. Instead of using a fixed model for every test case, we adapt our pose estimator during test time to exploit person-specific information. We first train our model on diverse data with both a supervised and a self-supervised pose estimation objectives jointly. We use a Transformer model to build a transformation between the self-supervised keypoints and the supervised keypoints. During test time, we personalize and adapt our model by fine-tuning with the self-supervised objective. The pose is then improved by transforming the updated self-supervised keypoints. We experiment with multiple datasets and show significant improvements on pose estimations with our self-supervised personalization. Project page with code is available at https://liyz15.github.io/TTP/.

        ----

        ## [198] Towards Scalable Unpaired Virtual Try-On via Patch-Routed Spatially-Adaptive GAN

        **Authors**: *Zhenyu Xie, Zaiyu Huang, Fuwei Zhao, Haoye Dong, Michael Kampffmeyer, Xiaodan Liang*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/151de84cca69258b17375e2f44239191-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/151de84cca69258b17375e2f44239191-Abstract.html)

        **Abstract**:

        Image-based virtual try-on is one of the most promising applications of human-centric image generation due to its tremendous real-world potential. Yet, as most try-on approaches fit in-shop garments onto a target person, they require the laborious and restrictive construction of a paired training dataset, severely limiting their scalability. While a few recent works attempt to transfer garments directly from one person to another, alleviating the need to collect paired datasets, their performance is impacted by the lack of paired (supervised) information.  In particular, disentangling style and spatial information of the garment becomes a challenge, which existing methods either address by requiring auxiliary data or extensive online optimization procedures, thereby still inhibiting their scalability. To achieve a scalable virtual try-on system that can transfer arbitrary garments between a source and a target person in an unsupervised manner, we thus propose a texture-preserving end-to-end network, the PAtch-routed SpaTially-Adaptive GAN (PASTA-GAN), that facilitates real-world unpaired virtual try-on. Specifically, to disentangle the style and spatial information of each garment, PASTA-GAN consists of an innovative patch-routed disentanglement module for successfully retaining garment texture and shape characteristics.  Guided by the source person's keypoints, the patch-routed disentanglement module first decouples garments into normalized patches, thus eliminating the inherent spatial information of the garment, and then reconstructs the normalized patches to the warped garment complying with the target person pose. Given the warped garment, PASTA-GAN further introduces novel spatially-adaptive residual blocks that guide the generator to synthesize more realistic garment details. Extensive comparisons with paired and unpaired approaches demonstrate the superiority of PASTA-GAN, highlighting its ability to generate high-quality try-on images when faced with a large variety of garments(e.g. vests, shirts, pants), taking a crucial step towards real-world scalable try-on.

        ----

        ## [199] Bias Out-of-the-Box: An Empirical Analysis of Intersectional Occupational Biases in Popular Generative Language Models

        **Authors**: *Hannah Rose Kirk, Yennie Jun, Filippo Volpin, Haider Iqbal, Elias Benussi, Frédéric A. Dreyer, Aleksandar Shtedritski, Yuki M. Asano*

        **Conference**: *nips 2021*

        **URL**: [https://proceedings.neurips.cc/paper/2021/hash/1531beb762df4029513ebf9295e0d34f-Abstract.html](https://proceedings.neurips.cc/paper/2021/hash/1531beb762df4029513ebf9295e0d34f-Abstract.html)

        **Abstract**:

        The capabilities of natural language models trained on large-scale data have increased immensely over the past few years. Open source libraries such as HuggingFace have made these models easily available and accessible. While prior research has identified biases in large language models, this paper considers biases contained in the most popular versions of these models when applied `out-of-the-box' for downstream tasks. We focus on generative language models as they are well-suited for extracting biases inherited from training data. Specifically, we conduct an in-depth analysis of GPT-2, which is the most downloaded text generation model on HuggingFace, with over half a million downloads per month. We assess biases related to occupational associations for different protected categories by intersecting gender with religion, sexuality, ethnicity, political affiliation, and continental name origin. Using a template-based data collection pipeline, we collect 396K sentence completions made by GPT-2 and find: (i) The machine-predicted jobs are less diverse and more stereotypical for women than for men, especially for intersections; (ii) Intersectional interactions are highly relevant for occupational associations, which we quantify by fitting 262 logistic models; (iii) For most occupations, GPT-2 reflects the skewed gender and ethnicity distribution found in US Labor Bureau data, and even pulls the societally-skewed distribution towards gender parity in cases where its predictions deviate from real labor market observations. This raises the normative question of what language models \textit{should} learn - whether they should reflect or correct for existing inequalities.

        ----

        

[Go to the next page](NIPS-2021-list02.md)

[Go to the catalog section](README.md)