## [0] RVD: A Handheld Device-Based Fundus Video Dataset for Retinal Vessel Segmentation

**Authors**: *Md. Wahiduzzaman Khan, Hongwei Sheng, Hu Zhang, Heming Du, Sen Wang, Minas Theodore Coroneo, Farshid Hajati, Sahar Shariflou, Michael Kalloniatis, Jack Phu, Ashish Agar, Zi Huang, S. Mojtaba Golzan, Xin Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a71ee306d6991f2f87dd414e0bdf851-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a71ee306d6991f2f87dd414e0bdf851-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Retinal vessel segmentation is generally grounded in image-based datasets collected with bench-top devices. The static images naturally lose the dynamic characteristics of retina fluctuation, resulting in diminished dataset richness, and the usage of bench-top devices further restricts dataset scalability due to its limited accessibility. Considering these limitations, we introduce the first video-based retinal dataset by employing handheld devices for data acquisition. The dataset comprises 635 smartphone-based fundus videos collected from four different clinics, involving 415 patients from 50 to 75 years old. It delivers comprehensive and precise annotations of retinal structures in both spatial and temporal dimensions, aiming to advance the landscape of vasculature segmentation. Specifically, the dataset provides three levels of spatial annotations: binary vessel masks for overall retinal structure delineation, general vein-artery masks for distinguishing the vein and artery, and fine-grained vein-artery masks for further characterizing the granularities of each artery and vein. In addition, the dataset offers temporal annotations that capture the vessel pulsation characteristics, assisting in detecting ocular diseases that require fine-grained recognition of hemodynamic fluctuation. In application, our dataset exhibits a significant domain shift with respect to data captured by bench-top devices, thus posing great challenges to existing methods. Thanks to rich annotations and data scales, our dataset potentially paves the path for more advanced retinal analysis and accurate disease diagnosis. In the experiments, we provide evaluation metrics and benchmark results on our dataset, reflecting both the potential and challenges it offers for vessel segmentation tasks. We hope this challenging dataset would significantly contribute to the development of eye disease diagnosis and early prevention.

----

## [0] LayoutGPT: Compositional Visual Planning and Generation with Large Language Models

**Authors**: *Weixi Feng, Wanrong Zhu, Tsu-Jui Fu, Varun Jampani, Arjun R. Akula, Xuehai He, Sugato Basu, Xin Eric Wang, William Yang Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3a7f9e485845dac27423375c934cb4db-Abstract-Conference.html)

**Abstract**:

Attaining a high degree of user controllability in visual generation often requires intricate, fine-grained inputs like layouts. However, such inputs impose a substantial burden on users when compared to simple text inputs. To address the issue, we study how Large Language Models (LLMs) can serve as visual planners by generating layouts from text conditions, and thus collaborate with visual generative models. We propose LayoutGPT, a method to compose in-context visual demonstrations in style sheet language to enhance visual planning skills of LLMs. We show that LayoutGPT can generate plausible layouts in multiple domains, ranging from 2D images to 3D indoor scenes. LayoutGPT also shows superior performance in converting challenging language concepts like numerical and spatial relations to layout arrangements for faithful text-to-image generation. When combined with a downstream image generation model, LayoutGPT outperforms text-to-image models/systems by 20-40\% and achieves comparable performance as human users in designing visual layouts for numerical and spatial correctness. Lastly, LayoutGPT achieves comparable performance to supervised methods in 3D indoor scene synthesis, demonstrating its effectiveness and potential in multiple visual domains.

----

## [0] Data Pruning via Moving-one-Sample-out

**Authors**: *Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, Xiaojuan Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3abe23bf7e295b44369c24465d68987a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3abe23bf7e295b44369c24465d68987a-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a novel data-pruning approach called moving-one-sample-out (MoSo), which aims to identify and remove the least informative samples from the training set. The core insight behind MoSo is to determine the importance of each sample by assessing its impact on the optimal empirical risk. This is achieved by measuring the extent to which the empirical risk changes when a particular sample is excluded from the training set. Instead of using the computationally expensive leaving-one-out-retraining procedure, we propose an efficient first-order approximator that only requires gradient information from different training stages. The key idea behind our approximation is that samples with gradients that are consistently aligned with the average gradient of the training set are more informative and should receive higher scores, which could be intuitively understood as follows: if the gradient from a specific sample is consistent with the average gradient vector, it implies that optimizing the network using the sample will yield a similar effect on all remaining samples.  Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and achieves satisfactory performance across various settings. Experimental results demonstrate that MoSo effectively mitigates severe performance degradation at high pruning ratios and outperforms state-of-the-art methods by a large margin across various settings.

----

## [0] Alternation makes the adversary weaker in two-player games

**Authors**: *Volkan Cevher, Ashok Cutkosky, Ali Kavis, Georgios Piliouras, Stratis Skoulakis, Luca Viano*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3acb49252187efa352a1ae0e4b066ced-Abstract-Conference.html)

**Abstract**:

Motivated by alternating game-play in two-player games, we study an altenating variant of the \textit{Online Linear Optimization} (OLO). In alternating OLO,  a \textit{learner} at each round $t \in [n]$ selects a vector $x^t$ and then an \textit{adversary} selects a cost-vector $c^t \in [-1,1]^n$. The learner then experiences cost $(c^t + c^{t-1})^\top x^t$ instead of $(c^t)^\top x^t$ as in standard OLO. We establish that under this small twist, the $\Omega(\sqrt{T})$ lower bound on the regret is no longer valid. More precisely, we present two online learning algorithms for alternating OLO that respectively admit $\mathcal{O}((\log n)^{4/3} T^{1/3})$ regret for the $n$-dimensional simplex and $\mathcal{O}(\rho \log T)$ regret for the ball of radius $\rho>0$. Our results imply that in alternating game-play, an agent can always guarantee $\mathcal{\tilde{O}}((\log n)^{4/3} T^{1/3})$ regardless the strategies of the other agent while the regret bound improves to $\mathcal{O}(\log T)$ in case the agent admits only two actions.

----

## [0] Spuriosity Didn't Kill the Classifier: Using Invariant Predictions to Harness Spurious Features

**Authors**: *Cian Eastwood, Shashank Singh, Andrei Liviu Nicolicioiu, Marin Vlastelica Pogancic, Julius von Kügelgen, Bernhard Schölkopf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3acbe9dc3a1e8d48a57b16e9aef91879-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3acbe9dc3a1e8d48a57b16e9aef91879-Abstract-Conference.html)

**Abstract**:

To avoid failures on out-of-distribution data, recent works have sought to extract features that have an invariant or stable relationship with the label across domains, discarding "spurious" or unstable features whose relationship with the label changes across domains. However, unstable features often carry complementary information that could boost performance if used correctly in the test domain. In this work, we show how this can be done without test-domain labels. In particular, we prove that pseudo-labels based on stable features provide sufficient guidance for doing so, provided that stable and unstable features are conditionally independent given the label. Based on this theoretical insight, we propose Stable Feature Boosting (SFB), an algorithm for: (i) learning a predictor that separates stable and conditionally-independent unstable features; and (ii) using the stable-feature predictions to adapt the unstable-feature predictions in the test domain. Theoretically, we prove that SFB can learn an asymptotically-optimal predictor without test-domain labels. Empirically, we demonstrate the effectiveness of SFB on real and synthetic data.

----

## [0] A Pseudo-Semantic Loss for Autoregressive Models with Logical Constraints

**Authors**: *Kareem Ahmed, Kai-Wei Chang, Guy Van den Broeck*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3accfe8332366a6f740d8740cd4cd653-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3accfe8332366a6f740d8740cd4cd653-Abstract-Conference.html)

**Abstract**:

Neuro-symbolic AI bridges the gap between purely symbolic and neural approaches to learning. This often requires maximizing the likelihood of a symbolic constraint w.r.t the neural network's output distribution. Such output distributions are typically assumed to be fully-factorized. This limits the applicability of neuro-symbolic learning to the more expressive auto-regressive distributions, e.g., transformers. Under such distributions, computing the likelihood of even simple constraints is #P-hard. Instead of attempting to enforce the constraint on the entire likelihood distribution, we propose to do so on a random, local approximation thereof. More precisely, we approximate the likelihood of the constraint with the pseudolikelihood of the constraint centered around a model sample. Our approach is factorizable, allowing us to reuse solutions to sub-problems---a main tenet for the efficient computation of neuro-symbolic losses. It also provides a local, high fidelity approximation of the likelihood: it exhibits low entropy and KL-divergence around the model sample. We tested our approach on Sudoku and shortest-path prediction cast as auto-regressive generation, and observe that we greatly improve upon the base model's ability to predict logically-consistent outputs. We also tested our approach on the task of detoxifying large language models. We observe that using a simple constraint disallowing a list of toxic words, we are able to steer the model's outputs away from toxic generations, achieving SoTA compared to previous approaches.

----

## [0] Physics-Informed Bayesian Optimization of Variational Quantum Circuits

**Authors**: *Kim Nicoli, Christopher J. Anders, Lena Funcke, Tobias Hartung, Karl Jansen, Stefan Kühn, Klaus-Robert Müller, Paolo Stornati, Pan Kessel, Shinichi Nakajima*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3adb85a348a18cdd74ce99fbbab20301-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3adb85a348a18cdd74ce99fbbab20301-Abstract-Conference.html)

**Abstract**:

In this paper, we propose a novel and powerful method to harness Bayesian optimization for variational quantum eigensolvers (VQEs) - a hybrid quantum-classical protocol used to approximate the ground state of a quantum Hamiltonian. Specifically, we derive a VQE-kernel which incorporates important prior information about quantum circuits: the kernel feature map of the VQE-kernel exactly matches the known functional form of the VQE's objective function and thereby significantly reduces the posterior uncertainty.Moreover, we propose a novel acquisition function for Bayesian optimization called \emph{Expected Maximum Improvement over Confident Regions} (EMICoRe) which can actively exploit the inductive bias of the VQE-kernel by treating regions with low predictive uncertainty as indirectly "observed". As a result, observations at as few as three points in the search domain are sufficient to determine the complete objective function along an entire one-dimensional subspace of the optimization landscape. Our numerical experiments demonstrate that our approach improves over state-of-the-art baselines.

----

## [0] Rubik's Cube: High-Order Channel Interactions with a Hierarchical Receptive Field

**Authors**: *Naishan Zheng, Man Zhou, Chong Zhou, Chen Change Loy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ae86071c169649bff21188c536163dc-Abstract-Conference.html)

**Abstract**:

Image restoration techniques, spanning from the convolution to the transformer paradigm, have demonstrated robust spatial representation capabilities to deliver high-quality performance.Yet, many of these methods, such as convolution and the Feed Forward Network (FFN) structure of transformers, primarily leverage the basic first-order channel interactions and have not maximized the potential benefits of higher-order modeling. To address this limitation, our research dives into understanding relationships within the channel dimension and introduces a simple yet efficient, high-order channel-wise operator tailored for image restoration. Instead of merely mimicking high-order spatial interaction, our approach offers several added benefits: Efficiency: It adheres to the zero-FLOP and zero-parameter principle, using a spatial-shifting mechanism across channel-wise groups. Simplicity: It turns the favorable channel interaction and aggregation capabilities into element-wise multiplications and convolution units with $1 \times 1$ kernel. Our new formulation expands the first-order channel-wise interactions seen in previous works to arbitrary high orders, generating a hierarchical receptive field akin to a Rubik's cube through the combined action of shifting and interactions. Furthermore, our proposed Rubik's cube convolution is a flexible operator that can be incorporated into existing image restoration networks, serving as a drop-in replacement for the standard convolution unit with fewer parameters overhead. We conducted experiments across various low-level vision tasks, including image denoising, low-light image enhancement, guided image super-resolution, and image de-blurring. The results consistently demonstrate that our Rubik's cube operator enhances performance across all tasks. Code is publicly available at https://github.com/zheng980629/RubikCube.

----

## [0] Closing the Computational-Statistical Gap in Best Arm Identification for Combinatorial Semi-bandits

**Authors**: *Ruo-Chun Tzeng, Po-An Wang, Alexandre Proutière, Chi-Jen Lu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ae8a7d6fc6d0d45e7c1ad9d4b063a01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ae8a7d6fc6d0d45e7c1ad9d4b063a01-Abstract-Conference.html)

**Abstract**:

We study the best arm identification problem in combinatorial semi-bandits in the fixed confidence setting. We present Perturbed Frank-Wolfe Sampling (P-FWS), an algorithm that (i) runs in polynomial time, (ii) achieves the instance-specific minimal sample complexity in the high confidence regime, and (iii) enjoys polynomial sample complexity guarantees in the moderate confidence regime. To our best knowledge, existing algorithms cannot achieve (ii) and (iii) simultaneously in vanilla bandits. With P-FWS, we close the computational-statistical gap in best arm identification in combinatorial semi-bandits. The design of P-FWS starts from the optimization problem that defines the information-theoretical and instance-specific sample complexity lower bound. P-FWS solves this problem in an online manner using, in each round, a single iteration of the Frank-Wolfe algorithm. Structural properties of the problem are leveraged to make the P-FWS successive updates computationally efficient. In turn, P-FWS only relies on a simple linear maximization oracle.

----

## [0] Imitation Learning from Imperfection: Theoretical Justifications and Algorithms

**Authors**: *Ziniu Li, Tian Xu, Zeyu Qin, Yang Yu, Zhi-Quan Luo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3af25aa3de8b7b02ddbd1b6be5031be8-Abstract-Conference.html)

**Abstract**:

Imitation learning (IL) algorithms excel in acquiring high-quality policies from expert data for sequential decision-making tasks. But, their effectiveness is hampered when faced with limited expert data. To tackle this challenge, a novel framework called (offline) IL with supplementary data has been proposed, which enhances learning by incorporating an additional yet imperfect dataset obtained inexpensively from sub-optimal policies. Nonetheless, learning becomes challenging due to the potential inclusion of out-of-expert-distribution samples. In this work, we propose a mathematical formalization of this framework, uncovering its limitations. Our theoretical analysis reveals that a naive approach—applying the behavioral cloning (BC) algorithm concept to the combined set of expert and supplementary data—may fall short of vanilla BC, which solely relies on expert data. This deficiency arises due to the distribution shift between the two data sources. To address this issue, we propose a new importance-sampling-based technique for selecting data within the expert distribution. We prove that the proposed method eliminates the gap of the naive approach, highlighting its efficacy when handling imperfect data. Empirical studies demonstrate that our method outperforms previous state-of-the-art methods in tasks including robotic locomotion control, Atari video games, and image classification. Overall, our work underscores the potential of improving IL by leveraging diverse data sources through effective data selection.

----

## [0] Detection Based Part-level Articulated Object Reconstruction from Single RGBD Image

**Authors**: *Yuki Kawana, Tatsuya Harada*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3af8c40dcf1bc94fa570a5e42edf219d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3af8c40dcf1bc94fa570a5e42edf219d-Abstract-Conference.html)

**Abstract**:

We propose an end-to-end trainable, cross-category method for reconstructing multiple man-made articulated objects from a single RGBD image, focusing on part-level shape reconstruction and pose and kinematics estimation. We depart from previous works that rely on learning instance-level latent space, focusing on man-made articulated objects with predefined part counts. Instead, we propose a novel alternative approach that employs part-level representation, representing instances as combinations of detected parts. While our detect-then-group approach effectively handles instances with diverse part structures and various part counts, it faces issues of false positives, varying part sizes and scales, and an increasing model size due to end-to-end training. To address these challenges, we propose 1) test-time kinematics-aware part fusion to improve detection performance while suppressing false positives, 2) anisotropic scale normalization for part shape learning to accommodate various part sizes and scales, and 3) a balancing strategy for cross-refinement between feature space and output space to improve part detection while maintaining model size. Evaluation on both synthetic and real data demonstrates that our method successfully reconstructs variously structured multiple instances that previous works cannot handle, and outperforms prior works in shape reconstruction and kinematics estimation.

----

## [0] FlatMatch: Bridging Labeled Data and Unlabeled Data with Cross-Sharpness for Semi-Supervised Learning

**Authors**: *Zhuo Huang, Li Shen, Jun Yu, Bo Han, Tongliang Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b11c5cc84b6da2838db348b37dbd1a2-Abstract-Conference.html)

**Abstract**:

Semi-Supervised Learning (SSL) has been an effective way to leverage abundant unlabeled data with extremely scarce labeled data. However, most SSL methods are commonly based on instance-wise consistency between different data transformations. Therefore, the label guidance on labeled data is hard to be propagated to unlabeled data. Consequently, the learning process on labeled data is much faster than on unlabeled data which is likely to fall into a local minima that does not favor unlabeled data, leading to sub-optimal generalization performance. In this paper, we propose FlatMatch which minimizes a cross-sharpness measure to ensure consistent learning performance between the two datasets. Specifically, we increase the empirical risk on labeled data to obtain a worst-case model which is a failure case needing to be enhanced. Then, by leveraging the richness of unlabeled data, we penalize the prediction difference (i.e., cross-sharpness) between the worst-case model and the original model so that the learning direction is beneficial to generalization on unlabeled data. Therefore, we can calibrate the learning process without being limited to insufficient label information. As a result, the mismatched learning performance can be mitigated, further enabling the effective exploitation of unlabeled data and improving SSL performance. Through comprehensive validation, we show FlatMatch achieves state-of-the-art results in many SSL settings.

----

## [0] Neural Sculpting: Uncovering hierarchically modular task structure in neural networks through pruning and network analysis

**Authors**: *Shreyas Malakarjun Patil, Loizos Michael, Constantine Dovrolis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b1675de6b49cc00084374213f8c38ae-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b1675de6b49cc00084374213f8c38ae-Abstract-Conference.html)

**Abstract**:

Natural target functions and tasks typically exhibit hierarchical modularity -- they can be broken down into simpler sub-functions that are organized in a hierarchy. Such sub-functions have two important features: they have a distinct set of inputs (input-separability) and they are reused as inputs higher in the hierarchy (reusability). Previous studies have established that hierarchically modular neural networks, which are inherently sparse, offer benefits such as learning efficiency, generalization, multi-task learning, and transfer. However, identifying the underlying sub-functions and their hierarchical structure for a given task can be challenging. The high-level question in this work is: if we learn a task using a sufficiently deep neural network, how can we uncover the underlying hierarchy of sub-functions in that task? As a starting point, we examine the domain of Boolean functions, where it is easier to determine whether a task is hierarchically modular. We propose an approach based on iterative unit and edge pruning (during training), combined with network analysis for module detection and hierarchy inference. Finally, we demonstrate that this method can uncover the hierarchical modularity of a wide range of Boolean functions and two vision tasks based on the MNIST digits dataset.

----

## [0] Elastic Decision Transformer

**Authors**: *Yueh-Hua Wu, Xiaolong Wang, Masashi Hamaya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b3889d313ba9476c12c2d77ea66b24f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b3889d313ba9476c12c2d77ea66b24f-Abstract-Conference.html)

**Abstract**:

This paper introduces Elastic Decision Transformer (EDT), a significant advancement over the existing Decision Transformer (DT) and its variants. Although DT purports to generate an optimal trajectory, empirical evidence suggests it struggles with trajectory stitching, a process involving the generation of an optimal or near-optimal trajectory from the best parts of a set of sub-optimal trajectories. The proposed EDT differentiates itself by facilitating trajectory stitching during action inference at test time, achieved by adjusting the history length maintained in DT. Further, the EDT optimizes the trajectory by retaining a longer history when the previous trajectory is optimal and a shorter one when it is sub-optimal, enabling it to "stitch" with a more optimal trajectory. Extensive experimentation demonstrates EDT's ability to bridge the performance gap between DT-based and Q Learning-based approaches. In particular, the EDT outperforms Q Learning-based methods in a multi-task regime on the D4RL locomotion benchmark and Atari games.

----

## [0] Asymptotically Optimal Quantile Pure Exploration for Infinite-Armed Bandits

**Authors**: *Evelyn Xiao-Yue Gong, Mark Sellke*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b3a83a5d86e1d424daefed43d998079-Abstract-Conference.html)

**Abstract**:

We study pure exploration with infinitely many bandit arms generated \iid from an unknown distribution. Our goal is to efficiently select a single high quality arm whose average reward is, with probability $1-\delta$, within $\varepsilon$ of being with the top $\eta$-fraction of arms; this is a natural adaptation of the classical PAC guarantee for infinite action sets. We consider both the fixed confidence and fixed budget settings, aiming respectively for optimal \emph{expected} and \emph{fixed} sample complexity.For fixed confidence, we give an algorithm with expected sample complexity $O\left(\frac{\log (1/\eta)\log (1/\delta)}{\eta\varepsilon^2}\right)$. This is optimal except for the $\log (1/\eta)$ factor, and the $\delta$-dependence closes a quadratic gap in the literature. For fixed budget, we show the asymptotically optimal sample complexity as $\delta\to 0$ is $c^{-1}\log(1/\delta)\big(\log\log(1/\delta)\big)^2$ to leading order; equivalently, the optimal failure probability with exactly $N$ samples decays as $\exp\big(-(1\pm o(1))\frac{cN}{\log^2 N}\big)$.The value of $c$ depends explicitly on the problem parameters (including the unknown arm distribution) through a certain Fisher information distance. Even the strictly super-linear dependence on $\log(1/\delta)$ was not known and resolves a question of Grossman-Moshkovitz (FOCS 2015).

----

## [0] Learning Probabilistic Symmetrization for Architecture Agnostic Equivariance

**Authors**: *Jinwoo Kim, Dat Nguyen, Ayhan Suleymanzade, Hyeokjun An, Seunghoon Hong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b5c7c9c5c7bd77eb73d0baec7a07165-Abstract-Conference.html)

**Abstract**:

We present a novel framework to overcome the limitations of equivariant architectures in learning functions with group symmetries. In contrary to equivariant architectures, we use an arbitrary base model such as an MLP or a transformer and symmetrize it to be equivariant to the given group by employing a small equivariant network that parameterizes the probabilistic distribution underlying the symmetrization. The distribution is end-to-end trained with the base model which can maximize performance while reducing sample complexity of symmetrization. We show that this approach ensures not only equivariance to given group but also universal approximation capability in expectation. We implement our method on various base models, including patch-based transformers that can be initialized from pretrained vision transformers, and test them for a wide range of symmetry groups including permutation and Euclidean groups and their combinations. Empirical tests show competitive results against tailored equivariant architectures, suggesting the potential for learning equivariant functions for diverse groups using a non-equivariant universal base architecture. We further show evidence of enhanced learning in symmetric modalities, like graphs, when pretrained from non-symmetric modalities, like vision. Code is available at https://github.com/jw9730/lps.

----

## [0] Distributionally Robust Linear Quadratic Control

**Authors**: *Bahar Taskesen, Dan A. Iancu, Çagil Koçyigit, Daniel Kuhn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b7a66b2d1258e892c89f485b8f896e0-Abstract-Conference.html)

**Abstract**:

Linear-Quadratic-Gaussian (LQG) control is a fundamental control paradigm that is studied in various fields such as engineering, computer science, economics, and neuroscience. It involves controlling a system with linear dynamics and imperfect observations, subject to additive noise, with the goal of minimizing a quadratic cost function for the state and control variables. In this work, we consider a generalization of the discrete-time, finite-horizon LQG problem, where the noise distributions are unknown and belong to Wasserstein ambiguity sets centered at nominal (Gaussian) distributions. The objective is to minimize a worst-case cost across all distributions in the ambiguity set, including non-Gaussian distributions. Despite the added complexity, we prove that a control policy that is linear in the observations is optimal for this problem, as in the classic LQG problem. We propose a numerical solution method that efficiently characterizes this optimal control policy. Our method uses the Frank-Wolfe algorithm to identify the least-favorable distributions within the Wasserstein ambiguity sets and computes the controller's optimal policy using Kalman filter estimation under these distributions.

----

## [0] Fully Dynamic k-Clustering in Õ(k) Update Time

**Authors**: *Sayan Bhattacharya, Martín Costa, Silvio Lattanzi, Nikos Parotsidis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3b7ba46201bf15e5c3935272afae50db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3b7ba46201bf15e5c3935272afae50db-Abstract-Conference.html)

**Abstract**:

We present a $O(1)$-approximate fully dynamic algorithm for the $k$-median and $k$-means problems on metric spaces with amortized update time $\tilde O(k)$ and worst-case query time $\tilde O(k^2)$. We complement our theoretical analysis with the first in-depth experimental study for the dynamic $k$-median problem on general metrics, focusing on comparing our dynamic algorithm to the current state-of-the-art by Henzinger and Kale [ESA'20]. Finally, we also provide a lower bound for dynamic $k$-median which shows that any $O(1)$-approximate algorithm with $\tilde O(\text{poly}(k))$ query time must have $\tilde \Omega(k)$ amortized update time, even in the incremental setting.

----

## [0] FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models

**Authors**: *Lihe Yang, Xiaogang Xu, Bingyi Kang, Yinghuan Shi, Hengshuang Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ba7560b4c3e66d760fbdd472cf4a5a9-Abstract-Conference.html)

**Abstract**:

Semantic segmentation has witnessed tremendous progress due to the proposal of various advanced network architectures. However, they are extremely hungry for delicate annotations to train, and the acquisition is laborious and unaffordable. Therefore, we present FreeMask in this work, which resorts to synthetic images from generative models to ease the burden of both data collection and annotation procedures. Concretely, we first synthesize abundant training images conditioned on the semantic masks provided by realistic datasets. This yields extra well-aligned image-mask training pairs for semantic segmentation models. We surprisingly observe that, solely trained with synthetic images, we already achieve comparable performance with real ones (e.g., 48.3 vs. 48.5 mIoU on ADE20K, and 49.3 vs. 50.5 on COCO-Stuff). Then, we investigate the role of synthetic images by joint training with real images, or pre-training for real images. Meantime, we design a robust filtering principle to suppress incorrectly synthesized regions. In addition, we propose to inequally treat different semantic masks to prioritize those harder ones and sample more corresponding synthetic images for them. As a result, either jointly trained or pre-trained with our filtered and re-sampled synthesized images, segmentation models can be greatly enhanced, e.g., from 48.7 to 52.0 on ADE20K.

----

## [0] RS-Del: Edit Distance Robustness Certificates for Sequence Classifiers via Randomized Deletion

**Authors**: *Zhuoqun Huang, Neil G. Marchant, Keane Lucas, Lujo Bauer, Olga Ohrimenko, Benjamin I. P. Rubinstein*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ba82362eb0aa75487069f19fde794fe-Abstract-Conference.html)

**Abstract**:

Randomized smoothing is a leading approach for constructing classifiers that are certifiably robust against adversarial examples. Existing work on randomized smoothing has focused on classifiers with continuous inputs, such as images, where $\ell_p$-norm bounded adversaries are commonly studied. However, there has been limited work for classifiers with discrete or variable-size inputs, such as for source code, which require different threat models and smoothing mechanisms. In this work, we adapt randomized smoothing for discrete sequence classifiers to provide certified robustness against edit distance-bounded adversaries. Our proposed smoothing mechanism randomized deletion (RS-Del) applies random deletion edits, which are (perhaps surprisingly) sufficient to confer robustness against adversarial deletion, insertion and substitution edits. Our proof of certification deviates from the established Neyman-Pearson approach, which is intractable in our setting, and is instead organized around longest common subsequences. We present a case study on malware detectionâ€”a binary classification problem on byte sequences where classifier evasion is a well-established threat model. When applied to the popular MalConv malware detection model, our smoothing mechanism RS-Del achieves a certified accuracy of 91% at an edit distance radius of 128 bytes.

----

## [0] Flow: Per-instance Personalized Federated Learning

**Authors**: *Kunjal Panchal, Sunav Choudhary, Nisarg Parikh, Lijun Zhang, Hui Guan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3baf4eeffad860ca9c54aeab632716b4-Abstract-Conference.html)

**Abstract**:

Federated learning (FL) suffers from data heterogeneity, where the diverse data distributions across clients make it challenging to train a single global model effectively. Existing personalization approaches aim to address the data heterogeneity issue by creating a personalized model for each client from the global model that fits their local data distribution. However, these personalized models may achieve lower accuracy than the global model in some clients, resulting in limited performance improvement compared to that without personalization. To overcome this limitation, we propose a per-instance personalization FL algorithm Flow. Flow creates dynamic personalized models that are adaptive not only to each client’s data distributions but also to each client’s data instances. The personalized model allows each instance to dynamically determine whether it prefers the local parameters or its global counterpart to make correct predictions, thereby improving clients’accuracy. We provide theoretical analysis on the convergence of Flow and empirically demonstrate the superiority of Flow in improving clients’ accuracy compared to state-of-the-art personalization approaches on both vision and language-based tasks.

----

## [0] MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile Wireless Sensing

**Authors**: *Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, Lihua Xie*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3baf7a39d07e9f4f1e258a412df94521-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3baf7a39d07e9f4f1e258a412df94521-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

4D human perception plays an essential role in a myriad of applications, such as home automation and metaverse avatar simulation. However, existing solutions which mainly rely on cameras and wearable devices are either privacy intrusive or inconvenient to use. To address these issues, wireless sensing has emerged as a promising alternative, leveraging LiDAR, mmWave radar, and WiFi signals for device-free human sensing. In this paper, we propose MM-Fi, the first multi-modal non-intrusive 4D human dataset with 27 daily or rehabilitation action categories, to bridge the gap between wireless sensing and high-level human perception tasks. MM-Fi consists of over 320k synchronized frames of five modalities from 40 human subjects. Various annotations are provided to support potential sensing tasks, e.g., human pose estimation and action recognition. Extensive experiments have been conducted to compare the sensing capacity of each or several modalities in terms of multiple tasks. We envision that MM-Fi can contribute to wireless sensing research with respect to action recognition, human pose estimation, multi-modal learning, cross-modal supervision, and interdisciplinary healthcare research.

----

## [0] Live Graph Lab: Towards Open, Dynamic and Real Transaction Graphs with NFT

**Authors**: *Zhen Zhang, Bingqiao Luo, Shengliang Lu, Bingsheng He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be31c1a2fdcb7b748c53c3f4cb0e9d2-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be31c1a2fdcb7b748c53c3f4cb0e9d2-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Numerous studies have been conducted to investigate the properties of large-scale temporal graphs. Despite the ubiquity of these graphs in real-world scenarios, it's usually impractical for us to obtain the whole real-time graphs due to privacy concerns and technical limitations. In this paper, we introduce the concept of {\it Live Graph Lab} for temporal graphs, which enables open, dynamic and real transaction graphs from blockchains. Among them, Non-fungible tokens (NFTs) have become one of the most prominent parts of blockchain over the past several years. With more than \$40 billion market capitalization, this decentralized ecosystem produces massive, anonymous and real transaction activities, which naturally forms a complicated transaction network. However, there is limited understanding about the characteristics of this emerging NFT ecosystem from a temporal graph analysis perspective. To mitigate this gap, we instantiate a live graph with NFT transaction network and investigate its dynamics to provide new observations and insights. Specifically, through downloading and parsing the NFT transaction activities, we obtain a temporal graph with more than 4.5 million nodes and 124 million edges. Then, a series of measurements are presented to understand the properties of the NFT ecosystem. Through comparisons with social, citation, and web networks, our analyses give intriguing findings and point out potential directions for future exploration. Finally, we also study machine learning models in this live graph to enrich the current datasets and provide new opportunities for the graph community. The source codes and dataset are available at https://livegraphlab.github.io.

----

## [0] CMMA: Benchmarking Multi-Affection Detection in Chinese Multi-Modal Conversations

**Authors**: *Yazhou Zhang, Yang Yu, Qing Guo, Benyou Wang, Dongming Zhao, Sagar Uprety, Dawei Song, Qiuchi Li, Jing Qin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Human communication has a multi-modal and multi-affection nature. The inter-relatedness of different emotions and sentiments poses a challenge to jointly detect multiple human affections with multi-modal clues. Recent advances in this field employed multi-task learning paradigms to render the inter-relatedness across tasks, but the scarcity of publicly available resources sets a limit to the potential of works. To fill this gap, we build the first Chinese Multi-modal Multi-Affection conversation (CMMA) dataset, which contains 3,000 multi-party conversations and 21,795 multi-modal utterances collected from various styles of TV-series. CMMA contains a wide variety of affection labels, including sentiment, emotion, sarcasm and humor, as well as the novel inter-correlations values between certain pairs of tasks. Moreover, it provides the topic and speaker information in conversations, which promotes better modeling of conversational context. On the dataset, we empirically analyze the influence of different data modalities and conversational contexts on different affection analysis tasks, and exhibit the practical benefit of inter-task correlations. The full dataset will be publicly available for research\footnote{https://github.com/annoymity2022/Chinese-Dataset}

----

## [0] Inverse Preference Learning: Preference-based RL without a Reward Function

**Authors**: *Joey Hejna, Dorsa Sadigh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3be7859b36d9440372cae0a293f2e4cc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3be7859b36d9440372cae0a293f2e4cc-Abstract-Conference.html)

**Abstract**:

Reward functions are difficult to design and often hard to align with human intent. Preference-based Reinforcement Learning (RL) algorithms address these problems by learning reward functions from human feedback. However, the majority of preference-based RL methods na\"ively combine supervised reward models with off-the-shelf RL algorithms. Contemporary approaches have sought to improve performance and query complexity by using larger and more complex reward architectures such as transformers. Instead of using highly complex architectures, we develop a new and parameter-efficient algorithm, Inverse Preference Learning (IPL), specifically designed for learning from offline preference data. Our key insight is that for a fixed policy, the $Q$-function encodes all information about the reward function, effectively making them interchangeable. Using this insight, we completely eliminate the need for a learned reward function. Our resulting algorithm is simpler and more parameter-efficient. Across a suite of continuous control and robotics benchmarks, IPL attains competitive performance compared to more complex approaches that leverage transformer-based and non-Markovian reward functions while having fewer algorithmic hyperparameters and learned network parameters. Our code is publicly released.

----

## [0] Matrix Compression via Randomized Low Rank and Low Precision Factorization

**Authors**: *Rajarshi Saha, Varun Srivastava, Mert Pilanci*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3bf4b55960aaa23553cd2a6bdc6e1b57-Abstract-Conference.html)

**Abstract**:

Matrices are exceptionally useful in various fields of study as they provide a convenient framework to organize and manipulate data in a structured manner. However, modern matrices can  involve billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Although prohibitively large, such matrices are often approximately low rank. We propose an algorithm that exploits this structure to obtain a low rank decomposition of any matrix $\mathbf{A}$ as $\mathbf{A} \approx \mathbf{L}\mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are the low rank factors. The total number of elements in $\mathbf{L}$ and $\mathbf{R}$ can be significantly less than that in $\mathbf{A}$. Furthermore, the entries of $\mathbf{L}$ and $\mathbf{R}$ are quantized to low precision formats -- compressing $\mathbf{A}$ by giving us a low rank and low precision factorization. Our algorithm first computes an approximate basis of the range space of $\mathbf{A}$ by randomly sketching its columns, followed by a quantization of the vectors constituting this basis. It then computes approximate projections of the columns of $\mathbf{A}$ onto this quantized basis. We derive upper bounds on the approximation error of our algorithm, and analyze the impact of target rank and quantization bit-budget. The tradeoff between compression ratio and approximation accuracy allows for flexibility in choosing these parameters based on specific application requirements. We empirically demonstrate the efficacy of our algorithm in image compression, nearest neighbor classification of image and text embeddings, and compressing the layers of LlaMa-$7$b. Our results illustrate that we can achieve compression ratios as aggressive as one bit per matrix coordinate, all while surpassing or maintaining the performance of traditional compression techniques.

----

## [0] OpenLane-V2: A Topology Reasoning Benchmark for Unified 3D HD Mapping

**Authors**: *Huijie Wang, Tianyu Li, Yang Li, Li Chen, Chonghao Sima, Zhenbo Liu, Bangjun Wang, Peijin Jia, Yuting Wang, Shengyin Jiang, Feng Wen, Hang Xu, Ping Luo, Junchi Yan, Wei Zhang, Hongyang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c0a4c8c236144f1b99b7e1531debe9c-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c0a4c8c236144f1b99b7e1531debe9c-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Accurately depicting the complex traffic scene is a vital component for autonomous vehicles to execute correct judgments. However, existing benchmarks tend to oversimplify the scene by solely focusing on lane perception tasks. Observing that human drivers rely on both lanes and traffic signals to operate their vehicles safely, we present OpenLane-V2, the first dataset on topology reasoning for traffic scene structure. The objective of the presented dataset is to advance research in understanding the structure of road scenes by examining the relationship between perceived entities, such as traffic elements and lanes. Leveraging existing datasets, OpenLane-V2 consists of 2,000 annotated road scenes that describe traffic elements and their correlation to the lanes. It comprises three primary sub-tasks, including the 3D lane detection inherited from OpenLane, accompanied by corresponding metrics to evaluate the modelâ€™s performance. We evaluate various state-of-the-art methods, and present their quantitative and qualitative results on OpenLane-V2 to indicate future avenues for investigating topology reasoning in traffic scenes.

----

## [0] Prompt-augmented Temporal Point Process for Streaming Event Sequence

**Authors**: *Siqiao Xue, Yan Wang, Zhixuan Chu, Xiaoming Shi, Caigao Jiang, Hongyan Hao, Gangwei Jiang, Xiaoyun Feng, James Zhang, Jun Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c129892b4f9c8326aba665425a470c5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c129892b4f9c8326aba665425a470c5-Abstract-Conference.html)

**Abstract**:

Neural Temporal Point Processes (TPPs)  are the prevalent paradigm for modeling continuous-time event sequences, such as user activities on the web and financial transactions. In real world applications, the event data typically comes in a streaming manner, where the distribution of the patterns may shift over time. Under the privacy and memory constraints commonly seen in real scenarios, how to continuously monitor a TPP to learn the streaming event sequence is an important yet under-investigated problem. In this work, we approach this problem by adopting Continual Learning (CL), which aims to enable a model to continuously learn a sequence of tasks without catastrophic forgetting. While CL for event sequence is less well studied, we present a simple yet effective framework, PromptTPP, by integrating the base TPP with a continuous-time retrieval prompt pool. In our proposed framework, prompts are small learnable parameters, maintained in a memory space and jointly optimized with the base TPP so that the model is properly instructed to learn event streams arriving sequentially without buffering past examples or task-specific attributes. We formalize a novel and realistic experimental setup for modeling event streams, where PromptTPP consistently sets state-of-the-art performance across two real user behavior datasets.

----

## [0] Leveraging Locality and Robustness to Achieve Massively Scalable Gaussian Process Regression

**Authors**: *Robert Allison, Anthony Stephenson, Samuel F, Edward O. Pyzer-Knapp*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c2b60a3f269c404e9329ee119f2d34a-Abstract-Conference.html)

**Abstract**:

The accurate predictions and principled uncertainty measures provided by GP regression incur $O(n^3)$ cost which is prohibitive for modern-day large-scale applications. This has motivated extensive work on computationally efficient approximations. We introduce a new perspective by exploring robustness properties and limiting behaviour of GP nearest-neighbour (GPnn) prediction. We demonstrate through theory and simulation that as the data-size $n$ increases, accuracy of estimated parameters and GP model assumptions become increasingly irrelevant to GPnn predictive accuracy. Consequently, it is sufficient to spend small amounts of work on parameter estimation in order to achieve high MSE accuracy, even in the presence of gross misspecification. In contrast, as $n \rightarrow \infty$, uncertainty calibration and NLL are shown to remain sensitive to just one parameter, the additive noise-variance; but we show that this source of inaccuracy can be corrected for, thereby achieving both well-calibrated uncertainty measures and accurate predictions at remarkably low computational cost. We exhibit a very simple GPnn regression algorithm with stand-out performance compared to other state-of-the-art GP approximations as measured on large UCI datasets. It operates at a small fraction of those other methods' training costs, for example on a basic laptop taking about 30 seconds to train on a dataset of size $n = 1.6 \times 10^6$.

----

## [0] Building the Bridge of Schrödinger: A Continuous Entropic Optimal Transport Benchmark

**Authors**: *Nikita Gushchin, Alexander Kolesov, Petr Mokrov, Polina Karpikova, Andrei Spiridonov, Evgeny Burnaev, Alexander Korotin*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c4688b6a76f25f2311daa0d75a58f1a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Over the last several years, there has been significant progress in developing neural solvers for the Schrödinger Bridge (SB) problem and applying them to generative modelling. This new research field is justifiably fruitful as it is interconnected with the practically well-performing diffusion models and theoretically grounded entropic optimal transport (EOT). Still, the area lacks non-trivial tests allowing a researcher to understand how well the methods solve SB or its equivalent continuous EOT problem. We fill this gap and propose a novel way to create pairs of probability distributions for which the ground truth OT solution is known by the construction. Our methodology is generic and works for a wide range of OT formulations, in particular, it covers the EOT which is equivalent to SB (the main interest of our study). This development allows us to create continuous benchmark distributions with the known EOT and SB solutions on high-dimensional spaces such as spaces of images. As an illustration, we use these benchmark pairs to test how well existing neural EOT/SB solvers actually compute the EOT solution. Our code for constructing benchmark pairs under different setups is available at: https://github.com/ngushchin/EntropicOTBenchmark

----

## [0] Safety Gymnasium: A Unified Safe Reinforcement Learning Benchmark

**Authors**: *Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Josef Dai, Yaodong Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c557a3d6a48cc99444f85e924c66753-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called Safety-Gymnasium, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.

----

## [0] Direct Training of SNN using Local Zeroth Order Method

**Authors**: *Bhaskar Mukhoty, Velibor Bojkovic, William de Vazelhes, Xiaohan Zhao, Giulia De Masi, Huan Xiong, Bin Gu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c5e64f26a97db6a2b0bbb788236431e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c5e64f26a97db6a2b0bbb788236431e-Abstract-Conference.html)

**Abstract**:

Spiking neural networks are becoming increasingly popular for their low energy requirement in real-world tasks with accuracy comparable to traditional ANNs. SNN training algorithms face the loss of gradient information and non-differentiability due to the Heaviside function in minimizing the model loss over model parameters. To circumvent this problem, the surrogate method employs a differentiable approximation of the Heaviside function in the backward pass, while the forward pass continues to use the Heaviside as the spiking function. We propose to use the zeroth-order technique at the local or neuron level in training SNNs, motivated by its regularizing and potential energy-efficient effects and establish a theoretical connection between it and the existing surrogate methods. We perform experimental validation of the technique on standard static datasets (CIFAR-10, CIFAR-100, ImageNet-100) and neuromorphic datasets (DVS-CIFAR-10, DVS-Gesture, N-Caltech-101, NCARS) and obtain results that offer improvement over the state-of-the-art results. The proposed method also lends itself to efficient implementations of the back-propagation method, which could provide 3-4 times overall speedup in training time. The code is available at \url{https://github.com/BhaskarMukhoty/LocalZO}.

----

## [0] Discover and Align Taxonomic Context Priors for Open-world Semi-Supervised Learning

**Authors**: *Yu Wang, Zhun Zhong, Pengchong Qiao, Xuxin Cheng, Xiawu Zheng, Chang Liu, Nicu Sebe, Rongrong Ji, Jie Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c646b713f5de2cf1ab1939d49a4036d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c646b713f5de2cf1ab1939d49a4036d-Abstract-Conference.html)

**Abstract**:

Open-world Semi-Supervised Learning (OSSL) is a realistic and challenging task, aiming to classify unlabeled samples from both seen and novel classes using partially labeled samples from the seen classes. Previous works typically explore the relationship of samples as priors on the pre-defined single-granularity labels to help novel class recognition. In fact, classes follow a taxonomy and samples can be classified at multiple levels of granularity, which contains more underlying relationships for supervision. We thus argue that learning with single-granularity labels results in sub-optimal representation learning and inaccurate pseudo labels, especially with unknown classes. In this paper, we take the initiative to explore and propose a uniformed framework, called Taxonomic context prIors Discovering and Aligning (TIDA), which exploits the relationship of samples under various granularity. It allows us to discover multi-granularity semantic concepts as taxonomic context priors (i.e., sub-class, target-class, and super-class), and then collaboratively leverage them to enhance representation learning and improve the quality of pseudo labels.Specifically, TIDA comprises two components: i) A taxonomic context discovery module that constructs a set of hierarchical prototypes in the latent space to discover the underlying taxonomic context priors; ii) A taxonomic context-based prediction alignment module that enforces consistency across hierarchical predictions to build the reliable relationship between classes among various granularity and provide additions supervision. We demonstrate that these two components are mutually beneficial for an effective OSSL framework, which is theoretically explained from the perspective of the EM algorithm. Extensive experiments on seven commonly used datasets show that TIDA can significantly improve the performance and achieve a new state of the art. The source codes are publicly available at https://github.com/rain305f/TIDA.

----

## [0] Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models

**Authors**: *Julien Siems, Konstantin Ditschuneit, Winfried Ripken, Alma Lindborg, Maximilian Schambach, Johannes S. Otterbach, Martin Genzel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c6696d70d364337cf98dcb7c652a770-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c6696d70d364337cf98dcb7c652a770-Abstract-Conference.html)

**Abstract**:

Generalized Additive Models (GAMs) have recently experienced a resurgence in popularity due to their interpretability, which arises from expressing the target value as a sum of non-linear transformations of the features. Despite the current enthusiasm for GAMs, their susceptibility to concurvity — i.e., (possibly non-linear) dependencies between the features — has hitherto been largely overlooked. Here, we demonstrate how concurvity can severly impair the interpretability of GAMs and propose a remedy: a conceptually simple, yet effective regularizer which penalizes pairwise correlations of the non-linearly transformed feature variables. This procedure is applicable to any differentiable additive model, such as Neural Additive Models or NeuralProphet, and enhances interpretability by eliminating ambiguities due to self-canceling feature contributions. We validate the effectiveness of our regularizer in experiments on synthetic as well as real-world datasets for time-series and tabular data. Our experiments show that concurvity in GAMs can be reduced without significantly compromising prediction quality, improving interpretability and reducing variance in the feature importances.

----

## [0] Mutual Information Regularized Offline Reinforcement Learning

**Authors**: *Xiao Ma, Bingyi Kang, Zhongwen Xu, Min Lin, Shuicheng Yan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3c6bd2021c10462c5164638d22f3d5d8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3c6bd2021c10462c5164638d22f3d5d8-Abstract-Conference.html)

**Abstract**:

The major challenge of offline RL is the distribution shift that appears when out-of-distribution actions are queried, which makes the policy improvement direction biased by extrapolation errors. Most existing methods address this problem by penalizing the policy or value for deviating from the behavior policy during policy improvement or evaluation. In this work, we propose a novel MISA framework to approach offline RL from the perspective of Mutual Information between States and Actions in the dataset by directly constraining the policy improvement direction. MISA constructs lower bounds of mutual information parameterized by the policy and Q-values. We show that optimizing this lower bound is equivalent to maximizing the likelihood of a one-step improved policy on the offline dataset. Hence, we constrain the policy improvement direction to lie in the data manifold. The resulting algorithm simultaneously augments the policy evaluation and improvement by adding mutual information regularizations. MISA is a general framework that unifies conservative Q-learning (CQL) and behavior regularization methods (e.g., TD3+BC) as special cases. We introduce 3 different variants of MISA, and empirically demonstrate that tighter mutual information lower bound gives better offline RL performance. In addition, our extensive experiments show MISA significantly outperforms a wide range of baselines on various tasks of the D4RL benchmark, e.g., achieving 742.9 total points on gym-locomotion tasks. Our code is attached and will be released upon publication.

----

## [0] Have it your way: Individualized Privacy Assignment for DP-SGD

**Authors**: *Franziska Boenisch, Christopher Mühl, Adam Dziedzic, Roy Rinberg, Nicolas Papernot*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cbf627fa24fb6cb576e04e689b9428b-Abstract-Conference.html)

**Abstract**:

When training a machine learning model with differential privacy, one sets a privacy budget. This uniform budget represents an overall maximal privacy violation that any user is willing to face by contributing their data to the training set. We argue that this approach is limited because different users may have different privacy expectations. Thus, setting a uniform privacy budget across all points may be overly conservative for some users or, conversely, not sufficiently protective for others. In this paper, we capture these preferences through individualized privacy budgets. To demonstrate their practicality, we introduce a variant of Differentially Private Stochastic Gradient Descent (DP-SGD) which supports such individualized budgets. DP-SGD is the canonical approach to training models with differential privacy. We modify its data sampling and gradient noising mechanisms to arrive at our approach, which we call Individualized DP-SGD (IDP-SGD). Because IDP-SGD provides privacy guarantees tailored to the preferences of individual users and their data points, we empirically find it to improve privacy-utility trade-offs.

----

## [0] Penguin: Parallel-Packed Homomorphic Encryption for Fast Graph Convolutional Network Inference

**Authors**: *Ran Ran, Nuo Xu, Tao Liu, Wei Wang, Gang Quan, Wujie Wen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cc685788a311fa35d8d41df93e288ca-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cc685788a311fa35d8d41df93e288ca-Abstract-Conference.html)

**Abstract**:

The marriage of Graph Convolutional Network (GCN) and Homomorphic Encryption (HE) enables the inference of graph data on the cloud with significantly enhanced client data privacy. However, the tremendous computation and memory overhead associated with HE operations challenges the practicality of HE-based GCN inference. GCN inference involves a sequence of expensive matrix-matrix multiplications, and we observe that directly applying the state-of-the-art HE-based secure matrix-matrix multiplication solutions to accelerate HE-GCN inference is far less efficient as it does not exploit the unique aggregation mechanism of two-dimension graph node-features in GCN layer computation. As a result, in this paper, we propose a novel HE-based ciphertext packing technique, i.e., Penguin, that can take advantage of the unique computation pattern during the HE-GCN inference to significantly reduce the computation and memory overhead associated with HE operations.Specifically, Penguin employs (i) an effective two-dimension parallel packing technique for feature ciphertext with optimal graph node partitioning and graph feature interleaving, and (ii) an interleaved assembly technique that can effectively make use of the blank slots to merge ciphertexts after feature reduction and significantly reduce the costly rotation operation.We provide theoretical analysis and experimental validation to demonstrate the speedup achieved by Penguin in accelerating GCN inference using popular GCN models and datasets. Our results show that Penguin can achieve up to $\sim10\times$ speedup and around $\sim79$% reduction in computational memory overhead, significantly outperforming state-of-the-art solutions. To the best of our knowledge, this is the first work that can ensure the protection of both graph structure and features when accelerating HE-GCN inference on encrypted data. Our code is publicly available at https://github.com/ranran0523/Penguin.

----

## [0] Learning Dynamic Attribute-factored World Models for Efficient Multi-object Reinforcement Learning

**Authors**: *Fan Feng, Sara Magliacane*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cc87f2bd3e3b4df8f9217326761c322-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cc87f2bd3e3b4df8f9217326761c322-Abstract-Conference.html)

**Abstract**:

In many reinforcement learning tasks, the agent has to learn to interact with many objects of different types and generalize to unseen combinations and numbers of objects. Often a task is a composition of previously learned tasks (e.g. block stacking).These are examples of compositional generalization, in which we compose object-centric representations to solve complex tasks. Recent works have shown the benefits of object-factored representations and hierarchical abstractions for improving sample efficiency in these settings. On the other hand, these methods do not fully exploit the benefits of factorization in terms of object attributes. In this paper, we address this opportunity and introduce the Dynamic Attribute FacTored RL (DAFT-RL) framework. In DAFT-RL, we leverage object-centric representation learning to extract objects from visual inputs. We learn to classify them into classes and infer their latent parameters. For each class of object, we learn a class template graph that describes how the dynamics and reward of an object of this class factorize according to its attributes. We also learn an interaction pattern graph that describes how objects of different classes interact with each other at the attribute level. Through these graphs and a dynamic interaction graph that models the interactions between objects, we can learn a policy that can then be directly applied in a new environment by estimating the interactions and latent parameters.We evaluate DAFT-RL in three benchmark datasets and show our framework outperforms the state-of-the-art in generalizing across unseen objects with varying attributes and latent parameters, as well as in the composition of previously learned tasks.

----

## [0] Statistical Insights into HSIC in High Dimensions

**Authors**: *Tao Zhang, Yaowu Zhang, Tingyou Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3cfc102893d47c46295cb437949dccb5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3cfc102893d47c46295cb437949dccb5-Abstract-Conference.html)

**Abstract**:

Measuring the nonlinear dependence between random vectors and testing for their statistical independence is a fundamental problem in statistics. One of the most popular dependence measures is the Hilbert-Schmidt independence criterion (HSIC), which has attracted increasing attention in recent years. However, most existing works have focused on either fixed or very high-dimensional covariates. In this work, we bridge the gap between these two scenarios and provide statistical insights into the performance of HSIC when the dimensions grow at different rates. We first show that, under the null hypothesis, the rescaled HSIC converges in distribution to a standard normal distribution. Then we provide a general condition for the HSIC based tests to have nontrivial power in high dimensions. By decomposing this condition, we illustrate how the ability of HSIC to measure nonlinear dependence changes with increasing dimensions. Moreover, we demonstrate that, depending on the sample size, the covariate dimensions and the dependence structures within covariates, the HSIC can capture different types of associations between random vectors. We also conduct extensive numerical studies to validate our theoretical results.

----

## [0] Fair Adaptive Experiments

**Authors**: *Waverly Wei, Xinwei Ma, Jingshen Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d007df4ae13adf9001f8969555b11bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d007df4ae13adf9001f8969555b11bd-Abstract-Conference.html)

**Abstract**:

Randomized experiments have been the gold standard for assessing the effectiveness of a treatment, policy, or intervention, spanning various fields, including social sciences, biomedical studies, and e-commerce. The classical complete randomization approach assigns treatments based on a pre-specified probability and may lead to inefficient use of data. Adaptive experiments improve upon complete randomization by sequentially learning and updating treatment assignment probabilities using accrued evidence during the experiment. Hence, they can help achieve efficient data use and higher estimation efficiency. However, their application can also raise fairness and equity concerns, as assignment probabilities may vary drastically across groups of participants. Furthermore, when treatment is expected to be extremely beneficial to certain groups of participants, it is more appropriate to expose many of these participants to favorable treatment. In response to these challenges, we propose a fair adaptive experiment strategy that simultaneously enhances data use efficiency, achieves an ``envy-free'' treatment assignment guarantee, and improves the overall welfare of participants. An important feature of our proposed strategy is that we do not impose parametric modeling assumptions on the outcome variables, making it more versatile and applicable to a wider array of applications. Through our theoretical investigation, we characterize the convergence rate of the estimated treatment effects and the associated standard deviations at the group level and further prove that our adaptive treatment assignment algorithm, despite not having a closed-form expression, approaches the optimal allocation rule asymptotically. Our proof strategy takes into account the fact that the allocation decisions in our design depend on sequentially accumulated data, which poses a significant challenge in characterizing the properties and conducting statistical inference of our method. We further provide simulation evidence and two synthetic data studies to showcase the performance of our fair adaptive experiment strategy.

----

## [0] Alexa Arena: A User-Centric Interactive Platform for Embodied AI

**Authors**: *Qiaozi Gao, Govind Thattai, Suhaila Shakiah, Xiaofeng Gao, Shreyas Pansare, Vasu Sharma, Gaurav S. Sukhatme, Hangjie Shi, Bofei Yang, Desheng Zhang, Lucy Hu, Karthika Arumugam, Shui Hu, Matthew Wen, Dinakar Guthy, Shunan Chung, Rohan Khanna, Osman Ipek, Leslie Ball, Kate Bland, Heather Rocker, Michael Johnston, Reza Ghanadan, Dilek Hakkani-Tur, Prem Natarajan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d0758f0b95e19abc68c1c8070d36510-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d0758f0b95e19abc68c1c8070d36510-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We introduce Alexa Arena, a user-centric simulation platform to facilitate research in building assistive conversational embodied agents. Alexa Arena features multi-room layouts and an abundance of interactable objects. With user-friendly graphics and control mechanisms, the platform supports the development of gamified robotic tasks readily accessible to general human users, allowing high-efficiency data collection and EAI system evaluation. Along with the platform, we introduce a dialog-enabled task completion benchmark with online human evaluations.

----

## [0] Synthetic Combinations: A Causal Inference Framework for Combinatorial Interventions

**Authors**: *Abhineet Agarwal, Anish Agarwal, Suhas Vijaykumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d17b7f7d52c83ab6e97e2dc0bda2e71-Abstract-Conference.html)

**Abstract**:

We consider a setting where there are $N$ heterogeneous units and $p$ interventions. Our goal is to learn unit-specific potential outcomes for any combination of these $p$ interventions, i.e., $N \times 2^p$ causal parameters. Choosing a combination of interventions is a problem that naturally arises in a variety of applications such as factorial design experiments and recommendation engines (e.g., showing a set of movies that maximizes engagement for a given user). Running $N \times 2^p$ experiments to estimate the various parameters is likely expensive and/or infeasible as $N$ and $p$ grow. Further, with observational data there is likely confounding, i.e., whether or not a unit is seen under a combination is correlated with its potential outcome under that combination. We study this problem under a novel model that imposes latent structure across both units and combinations of interventions. Specifically, we assume latent similarity in potential outcomes across units (i.e., the matrix of potential outcomes is approximately rank $r$) and regularity in how combinations of interventions interact (i.e., the coefficients in the Fourier expansion of the potential outcomes is approximately $s$ sparse). We establish identification for all $N \times 2^p$ parameters despite unobserved confounding. We propose an estimation procedure, Synthetic Combinations, and establish finite-sample consistency under precise conditions on the observation pattern. We show that Synthetic Combinations is able to consistently estimate unit-specific potential outcomes given a total of $\text{poly}(r) \times \left( N + s^2p\right)$ observations. In comparison, previous methods that do not exploit structure across both units and combinations have poorer sample complexity scaling as $\min(N \times s^2p, \ \ r \times (N + 2^p))$.

----

## [0] PUCA: Patch-Unshuffle and Channel Attention for Enhanced Self-Supervised Image Denoising

**Authors**: *Hyemi Jang, Junsung Park, Dahuin Jung, Jaihyun Lew, Ho Bae, Sungroh Yoon*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d226fb8fbd6ee6ec70d0427f1319707-Abstract-Conference.html)

**Abstract**:

Although supervised image denoising networks have shown remarkable performance on synthesized noisy images, they often fail in practice due to the difference between real and synthesized noise. Since clean-noisy image pairs from the real world are extremely costly to gather, self-supervised learning, which utilizes noisy input itself as a target, has been studied. To prevent a self-supervised denoising model from learning identical mapping, each output pixel should not be influenced by its corresponding input pixel; This requirement is known as J-invariance. Blind-spot networks (BSNs) have been a prevalent choice to ensure J-invariance in self-supervised image denoising. However, constructing variations of BSNs by injecting additional operations such as downsampling can expose blinded information, thereby violating J-invariance. Consequently, convolutions designed specifically for BSNs have been allowed only, limiting architectural flexibility. To overcome this limitation, we propose PUCA, a novel J-invariant U-Net architecture, for self-supervised denoising. PUCA leverages patch-unshuffle/shuffle to dramatically expand receptive fields while maintaining J-invariance and dilated attention blocks (DABs) for global context incorporation. Experimental results demonstrate that PUCA achieves state-of-the-art performance, outperforming existing methods in self-supervised image denoising.

----

## [0] Projection Regret: Reducing Background Bias for Novelty Detection via Diffusion Models

**Authors**: *Sungik Choi, Hankook Lee, Honglak Lee, Moontae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d27d607586984908900eaa8ce19c96c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d27d607586984908900eaa8ce19c96c-Abstract-Conference.html)

**Abstract**:

Novelty detection is a fundamental task of machine learning which aims to detect abnormal (i.e. out-of-distribution (OOD)) samples. Since diffusion models have recently emerged as the de facto standard generative framework with surprising generation results, novelty detection via diffusion models has also gained much attention. Recent methods have mainly utilized the reconstruction property of in-distribution samples. However, they often suffer from detecting OOD samples that share similar background information to the in-distribution data. Based on our observation that diffusion models can project any sample to an in-distribution sample with similar background information, we propose Projection Regret (PR), an efficient novelty detection method that mitigates the bias of non-semantic information. To be specific, PR computes the perceptual distance between the test image and its diffusion-based projection to detect abnormality. Since the perceptual distance often fails to capture semantic changes when the background information is dominant, we cancel out the background bias by comparing it against recursive projections. Extensive experiments demonstrate that PR outperforms the prior art of generative-model-based novelty detection methods by a significant margin.

----

## [0] Versatile Energy-Based Probabilistic Models for High Energy Physics

**Authors**: *Taoli Cheng, Aaron C. Courville*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d4c0a618d0acd7921493e4f30395c22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d4c0a618d0acd7921493e4f30395c22-Abstract-Conference.html)

**Abstract**:

As a classical generative modeling approach, energy-based models have the natural advantage of flexibility in the form of the energy function. Recently, energy-based models have achieved great success in modeling high-dimensional data in computer vision and natural language processing. In line with these advancements, we build a multi-purpose energy-based probabilistic model for High Energy Physics events at the Large Hadron Collider.  This framework builds on a powerful generative model and describes higher-order inter-particle interactions. It suits different encoding architectures and builds on implicit generation. As for applicative aspects, it can serve as a powerful parameterized event generator for physics simulation, a generic anomalous signal detector free from spurious correlations, and an augmented event classifier for particle identification.

----

## [0] User-Level Differential Privacy With Few Examples Per User

**Authors**: *Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Raghu Meka, Chiyuan Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d57795f0e263aa69577f1bbceade46b-Abstract-Conference.html)

**Abstract**:

Previous work on user-level differential privacy (DP) [Ghazi et al. NeurIPS 2021, Bun et al. STOC 2023] obtained generic algorithms that work for various learning tasks. However, their focus was on the *example-rich* regime, where the users have so many examples that each user could themselves solve the problem. In this work we consider the *example-scarce* regime, where each user has only a few examples, and obtain the following results:* For approximate-DP, we give a generic transformation of any item-level DP algorithm to a user-level DP algorithm. Roughly speaking, the latter gives a (multiplicative) savings of $O_{\varepsilon,\delta}(\sqrt{m})$ in terms of the number of users required for achieving the same utility, where $m$ is the number of examples per user. This algorithm, while recovering most known bounds for specific problems, also gives new bounds, e.g., for PAC learning. * For pure-DP, we present a simple technique for adapting the exponential mechanism [McSherry & Talwar, FOCS 2007] to the user-level setting. This gives new bounds for a variety of tasks, such as private PAC learning, hypothesis selection, and distribution learning. For some of these problems, we show that our bounds are near-optimal.

----

## [0] Neural Lighting Simulation for Urban Scenes

**Authors**: *Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, Raquel Urtasun*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d7259031023c5aa463187c4a31c95c8-Abstract-Conference.html)

**Abstract**:

Different outdoor illumination conditions drastically alter the appearance of urban scenes, and they can harm the performance of image-based robot perception systems if not seen during training. Camera simulation provides a cost-effective solution to create a large dataset of images captured under different lighting conditions. Towards this goal, we propose LightSim, a neural lighting camera simulation system that enables diverse, realistic, and controllable data generation. LightSim automatically builds lighting-aware digital twins at scale from collected raw sensor data and decomposes the scene into dynamic actors and static background with accurate geometry, appearance, and estimated scene lighting. These digital twins enable actor insertion, modification, removal, and rendering from a new viewpoint, all in a lighting-aware manner. LightSim then combines physically-based and learnable deferred rendering to perform realistic relighting of modified scenes, such as altering the sun location and modifying the shadows or changing the sun brightness, producing spatially- and temporally-consistent camera videos. Our experiments show that LightSim generates more realistic relighting results  than prior work. Importantly,  training perception models on data generated by LightSim can significantly improve their performance. Our project page is available at https://waabi.ai/lightsim/.

----

## [0] Learning to Compress Prompts with Gist Tokens

**Authors**: *Jesse Mu, Xiang Li, Noah D. Goodman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d77c6dcc7f143aa2154e7f4d5e22d68-Abstract-Conference.html)

**Abstract**:

Prompting is the primary way to utilize the multitask capabilities of language models (LMs), but prompts occupy valuable space in the input context window, and repeatedly encoding the same prompt is computationally inefficient. Finetuning and distillation methods allow for specialization of LMs without prompting, but require retraining the model for each task. To avoid this trade-off entirely, we present gisting, which trains an LM to compress prompts into smaller sets of "gist" tokens which can be cached and reused for compute efficiency. Gist models can be trained with no additional cost over standard instruction finetuning by simply modifying Transformer attention masks to encourage prompt compression. On decoder (LLaMA-7B) and encoder-decoder (FLAN-T5-XXL) LMs, gisting enables up to 26x compression of prompts, resulting in up to 40% FLOPs reductions, 4.2% wall time speedups, and storage savings, all with minimal loss in output quality.

----

## [0] A Heavy-Tailed Algebra for Probabilistic Programming

**Authors**: *Feynman T. Liang, Liam Hodgkinson, Michael W. Mahoney*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3d8f7945cd7f4446cb05a390d4c00558-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3d8f7945cd7f4446cb05a390d4c00558-Abstract-Conference.html)

**Abstract**:

Despite the successes of probabilistic models based on passing noise through neural networks, recent work has identified that such methods often fail to capture tail behavior accurately---unless the tails of the base distribution are appropriately calibrated.  To overcome this deficiency, we propose a systematic approach for analyzing the tails of random variables, and we illustrate how this approach can be used during the static analysis (before drawing samples) pass of a probabilistic programming language (PPL) compiler.  To characterize how the tails change under various operations, we develop an algebra which acts on a three-parameter family of tail asymptotics and which is based on the generalized Gamma distribution.  Our algebraic operations are closed under addition and multiplication; they are capable of distinguishing sub-Gaussians with differing scales; and they handle ratios sufficiently well to reproduce the tails of most important statistical distributions directly from their definitions.  Our empirical results confirm that inference algorithms that leverage our heavy-tailed algebra attain superior performance across a number of density modeling and variational inference (VI) tasks.

----

## [0] AIMS: All-Inclusive Multi-Level Segmentation for Anything

**Authors**: *Lu Qi, Jason Kuen, Weidong Guo, Jiuxiang Gu, Zhe Lin, Bo Du, Yu Xu, Ming-Hsuan Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3da292ced54290c19fc55d9dba3da793-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3da292ced54290c19fc55d9dba3da793-Abstract-Conference.html)

**Abstract**:

Despite the progress of image segmentation for accurate visual entity segmentation, completing the diverse requirements of image editing applications for different-level region-of-interest selections remains unsolved. In this paper, we propose a new task, All-Inclusive Multi-Level Segmentation (AIMS), which segments visual regions into three levels: part, entity, and relation (two entities with some semantic relationships). We also build a unified AIMS model through multi-dataset multi-task training to address the two major challenges of annotation inconsistency and task correlation. Specifically, we propose task complementarity, association, and prompt mask encoder for three-level predictions. Extensive experiments demonstrate the effectiveness and generalization capacity of our method compared to other state-of-the-art methods on a single dataset or the concurrent work on segment anything. We will make our code and training model publicly available.

----

## [0] Performance Bounds for Policy-Based Average Reward Reinforcement Learning Algorithms

**Authors**: *Yashaswini Murthy, Mehrdad Moharrami, R. Srikant*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3da8e709fa1a7d9e23bee89d3c25b5b4-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3da8e709fa1a7d9e23bee89d3c25b5b4-Abstract-Conference.html)

**Abstract**:

Many policy-based reinforcement learning (RL) algorithms can be viewed as instantiations of approximate policy iteration (PI), i.e., where policy improvement and policy evaluation are both performed approximately. In applications where the average reward objective is the meaningful performance metric, often discounted reward formulations are used with the discount factor being close to $1,$ which is equivalent to making the expected horizon very large. However, the corresponding theoretical bounds for error performance scale with the square of the horizon. Thus, even after dividing the total reward by the length of the horizon, the corresponding performance bounds for average reward problems go to infinity. Therefore, an open problem has been to obtain meaningful performance bounds for approximate PI and RL algorithms for the average-reward setting.  In this paper, we solve this open problem by obtaining the first non-trivial finite time error bounds for average-reward MDPs which go to zero in the limit as policy evaluation and policy improvement errors go to zero.

----

## [0] Understanding Few-Shot Learning: Measuring Task Relatedness and Adaptation Difficulty via Attributes

**Authors**: *Minyang Hu, Hong Chang, Zong Guo, Bingpeng Ma, Shiguang Shan, Xilin Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3df38ca67befaed9c03b95ffee07d9f8-Abstract-Conference.html)

**Abstract**:

Few-shot learning (FSL) aims to learn novel tasks with very few labeled samples by leveraging experience from \emph{related} training tasks.    In this paper, we try to understand FSL by exploring two key questions:    (1) How to quantify the relationship between \emph{ training} and \emph{novel} tasks?    (2) How does the relationship affect the \emph{adaptation difficulty} on novel tasks for different models?    To answer the first question, we propose Task Attribute Distance (TAD) as a metric to quantify the task relatedness via attributes.    Unlike other metrics, TAD is independent of models, making it applicable to different FSL models.    To address the second question, we utilize TAD metric to establish a theoretical connection between task relatedness and task adaptation difficulty.    By deriving the generalization error bound on a novel task, we discover how TAD measures the adaptation difficulty on novel tasks for different models.    To validate our theoretical results, we conduct experiments on three benchmarks.    Our experimental results confirm that TAD metric effectively quantifies the task relatedness and reflects the adaptation difficulty on novel tasks for various FSL methods, even if some of them do not learn attributes explicitly or human-annotated attributes are not provided.    Our code is available at     \href{https://github.com/hu-my/TaskAttributeDistance}{https://github.com/hu-my/TaskAttributeDistance}.

----

## [0] Locally Invariant Explanations: Towards Stable and Unidirectional Explanations through Local Invariant Learning

**Authors**: *Amit Dhurandhar, Karthikeyan Natesan Ramamurthy, Kartik Ahuja, Vijay Arya*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3df874367ce2c43891aab1ab23ae6959-Abstract-Conference.html)

**Abstract**:

Locally interpretable model agnostic explanations (LIME) method is one of the most popular methods used to explain black-box models at a per example level. Although many variants have been proposed, few provide a simple way to produce high fidelity explanations that are also stable and intuitive. In this work, we provide a novel perspective by proposing a model agnostic local explanation method inspired by the invariant risk minimization (IRM) principle -- originally proposed for (global) out-of-distribution generalization -- to provide such high fidelity explanations that are also stable and unidirectional across nearby examples. Our method is based on a game theoretic formulation where we theoretically show that our approach has a strong tendency to eliminate features where the gradient of the black-box function abruptly changes sign in the locality of the example we want to explain, while in other cases it is more careful and will choose a more conservative (feature) attribution, a behavior which can be highly desirable for recourse. Empirically, we show on tabular, image and text data that the quality of our explanations with neighborhoods formed using random perturbations are much better than LIME and in some cases even comparable to other methods that use realistic neighbors sampled from the data manifold. This is desirable given that learning a manifold to either create realistic neighbors or to project explanations is typically expensive or may even be impossible. Moreover, our algorithm is simple and efficient to train, and can ascertain stable input features for local decisions of a black-box without access to side information such as a (partial) causal graph as has been seen in some recent works.

----

## [0] Quantification of Uncertainty with Adversarial Models

**Authors**: *Kajetan Schweighofer, Lukas Aichberger, Mykyta Ielanskyi, Günter Klambauer, Sepp Hochreiter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e0b96206965f5f05b0b4550c0e73ff0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e0b96206965f5f05b0b4550c0e73ff0-Abstract-Conference.html)

**Abstract**:

Quantifying uncertainty is important for actionable predictions in real-world applications. A crucial part of predictive uncertainty quantification is the estimation of epistemic uncertainty, which is defined as an integral of the product between a divergence function and the posterior. Current methods such as Deep Ensembles or MC dropout underperform at estimating the epistemic uncertainty, since they primarily consider the posterior when sampling models. We suggest Quantification of Uncertainty with Adversarial Models (QUAM) to better estimate the epistemic uncertainty. QUAM identifies regions where the whole product under the integral is large, not just the posterior. Consequently, QUAM has lower approximation error of the epistemic uncertainty compared to previous methods. Models for which the product is large correspond to adversarial models (not adversarial examples!). Adversarial models have both a high posterior as well as a high divergence between their predictions and that of a reference model. Our experiments show that QUAM excels in capturing epistemic uncertainty for deep learning models and outperforms previous methods on challenging tasks in the vision domain.

----

## [0] NeuroGF: A Neural Representation for Fast Geodesic Distance and Path Queries

**Authors**: *Qijian Zhang, Junhui Hou, Yohanes Yudhi Adikusuma, Wenping Wang, Ying He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e22abb329d44080460b0eb11bf21da1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e22abb329d44080460b0eb11bf21da1-Abstract-Conference.html)

**Abstract**:

Geodesics play a critical role in many geometry processing applications. Traditional algorithms for computing geodesics on 3D mesh models are often inefficient and slow, which make them impractical for scenarios requiring extensive querying of arbitrary point-to-point geodesics. Recently, deep implicit functions have gained popularity for 3D geometry representation, yet there is still no research on neural implicit representation of geodesics. To bridge this gap, we make the first attempt to represent geodesics using implicit learning frameworks. Specifically, we propose neural geodesic field (NeuroGF), which can be learned to encode all-pairs geodesics of a given 3D mesh model, enabling to efficiently and accurately answer queries of arbitrary point-to-point geodesic distances and paths. Evaluations on common 3D object models and real-captured scene-level meshes demonstrate our exceptional performances in terms of representation accuracy and querying efficiency. Besides, NeuroGF also provides a convenient way of jointly encoding both 3D geometry and geodesics in a unified representation. Moreover, the working mode of per-model overfitting is further extended to generalizable learning frameworks that can work on various input formats such as unstructured point clouds, which also show satisfactory performances for unseen shapes and categories. Our code and data are available at https://github.com/keeganhk/NeuroGF.

----

## [0] A Trichotomy for Transductive Online Learning

**Authors**: *Steve Hanneke, Shay Moran, Jonathan Shafer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e32af2df2cd13dfbcbe6e8d38111068-Abstract-Conference.html)

**Abstract**:

We present new upper and lower bounds on the number of learner mistakes in the `transductive' online learning setting of Ben-David, Kushilevitz and Mansour (1997).    This setting is similar to standard online learning, except that the adversary fixes a sequence of instances $x_1,\dots,x_n$ to be labeled at the start of the game, and this sequence is known to the learner.    Qualitatively, we prove a \emph{trichotomy}, stating that the minimal number of mistakes made by the learner as $n$ grows can take only one of precisely three possible values: $n$, $\Theta\left(\log (n)\right)$, or $\Theta(1)$.    Furthermore, this behavior is determined by a combination of the VC dimension and the Littlestone dimension.    Quantitatively, we show a variety of bounds relating the number of mistakes to well-known combinatorial dimensions.    In particular, we improve the known lower bound on the constant in the $\Theta(1)$ case from $\Omega\left(\sqrt{\log(d)}\right)$ to $\Omega(\log(d))$ where $d$ is the Littlestone dimension.    Finally, we extend our results to cover multiclass classification and the agnostic setting.

----

## [0] Evolutionary Neural Architecture Search for Transformer in Knowledge Tracing

**Authors**: *Shangshang Yang, Xiaoshan Yu, Ye Tian, Xueming Yan, Haiping Ma, Xingyi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e53d82a1113e3d240059a9195668edc-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e53d82a1113e3d240059a9195668edc-Abstract-Conference.html)

**Abstract**:

Knowledge tracing (KT) aims to trace students' knowledge states by predicting whether students answer correctly on exercises. Despite the excellent performance of existing Transformer-based KT approaches, they are criticized for the manually selected input features for fusion and the defect of single global context modelling to directly capture students' forgetting behavior in KT, when the related records are distant from the current record in terms of time. To address the issues, this paper first considers adding convolution operations to the Transformer to enhance its local context modelling ability used for students' forgetting behavior, then proposes an evolutionary neural architecture search approach to automate the input feature selection and automatically determine where to apply which operation for achieving the balancing of the local/global context modelling. In the search space, the original global path containing the attention module in Transformer is replaced with the sum of a global path and a local path that could contain different convolutions, and the selection of input features is also considered. To search the best architecture, we employ an effective evolutionary algorithm to explore the search space and also suggest a search space reduction strategy to accelerate the convergence of the algorithm. Experimental results on the two largest and most challenging education datasets demonstrate the effectiveness of the architecture found by the proposed approach.

----

## [0] Learning threshold neurons via edge of stability

**Authors**: *Kwangjun Ahn, Sébastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, Yi Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e592c571de69a43d7a870ea89c7e33a-Abstract-Conference.html)

**Abstract**:

Existing analyses of neural network training often operate under the unrealistic assumption of an extremely small learning rate. This lies in stark contrast to practical wisdom and empirical studies, such as the work of J. Cohen et al. (ICLR 2021), which exhibit startling new phenomena (the "edge of stability"' or "unstable convergence") and potential benefits for generalization in the large learning rate regime. Despite a flurry of recent works on this topic, however, the latter effect is still poorly understood. In this paper, we take a step towards understanding genuinely non-convex training dynamics with large learning rates by performing a detailed analysis of gradient descent for simplified models of two-layer neural networks. For these models, we provably establish the edge of stability phenomenon and discover a sharp phase transition for the step size below which the neural network fails to learn ``threshold-like'' neurons (i.e., neurons with a non-zero first-layer bias). This elucidates one possible mechanism by which the edge of stability can in fact lead to better generalization, as threshold neurons are basic building blocks with useful inductive bias for many tasks.

----

## [0] k-Means Clustering with Distance-Based Privacy

**Authors**: *Alessandro Epasto, Vahab Mirrokni, Shyam Narayanan, Peilin Zhong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3e8d9bf1dd1eb9d3d9d500fb3543c87b-Abstract-Conference.html)

**Abstract**:

In this paper, we initiate the study of Euclidean clustering with Distance-based privacy. Distance-based privacy is motivated by the fact that it is often only needed to protect the privacy of exact, rather than approximate, locations. We provide constant-approximate algorithms for $k$-means and $k$-median clustering, with additive error depending only on the attacker's precision bound $\rho$, rather than the radius $\Lambda$ of the space. In addition, we empirically demonstrate that our algorithm performs significantly better than previous differentially private clustering algorithms, as well as naive distance-based private clustering baselines.

----

## [0] StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models

**Authors**: *Yinghao Aaron Li, Cong Han, Vinay S. Raghavan, Gavin Mischler, Nima Mesgarani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eaad2a0b62b5ed7a2e66c2188bb1449-Abstract-Conference.html)

**Abstract**:

In this paper, we present StyleTTS 2, a text-to-speech (TTS) model that leverages style diffusion and adversarial training with large speech language models (SLMs) to achieve human-level TTS synthesis. StyleTTS 2 differs from its predecessor by modeling styles as a latent random variable through diffusion models to generate the most suitable style for the text without requiring reference speech, achieving efficient latent diffusion while benefiting from the diverse speech synthesis offered by diffusion models. Furthermore, we employ large pre-trained SLMs, such as WavLM, as discriminators with our novel differentiable duration modeling for end-to-end training, resulting in improved speech naturalness. StyleTTS 2 surpasses human recordings on the single-speaker LJSpeech dataset and matches it on the multispeaker VCTK dataset as judged by native English speakers. Moreover, when trained on the LibriTTS dataset, our model outperforms previous publicly available models for zero-shot speaker adaptation. This work achieves the first human-level TTS on both single and multispeaker datasets, showcasing the potential of style diffusion and adversarial training with large SLMs. The audio demos and source code are available at https://styletts2.github.io/.

----

## [0] Large Language Models Are Zero-Shot Time Series Forecasters

**Authors**: *Nate Gruver, Marc Finzi, Shikai Qiu, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eb7ca52e8207697361b2c0fb3926511-Abstract-Conference.html)

**Abstract**:

By encoding time series as a string of numerical digits, we can frame time series forecasting as next-token prediction in text. Developing this approach, we find that large language models (LLMs) such as GPT-3 and LLaMA-2 can surprisingly zero-shot extrapolate time series at a level comparable to or exceeding the performance of purpose-built time series models trained on the downstream tasks. To facilitate this performance, we propose procedures for effectively tokenizing time series data and converting discrete distributions over tokens into highly flexible densities over continuous values. We argue the success of LLMs for time series stems from their ability to naturally represent multimodal distributions, in conjunction with biases for simplicity, and repetition, which align with the salient features in many time series, such as repeated seasonal trends. We also show how LLMs can naturally handle missing data without imputation through non-numerical text, accommodate textual side information, and answer questions to help explain predictions.  While we find that increasing model size generally improves performance on time series, we show GPT-4 can perform worse than GPT-3 because of how it tokenizes numbers, and poor uncertainty calibration, which is likely the result of alignment interventions such as RLHF.

----

## [0] Learning Mixtures of Gaussians Using the DDPM Objective

**Authors**: *Kulin Shah, Sitan Chen, Adam R. Klivans*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec077b4af90f2556b517b556e186f64-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec077b4af90f2556b517b556e186f64-Abstract-Conference.html)

**Abstract**:

Recent works have shown that diffusion models can learn essentially any distribution provided one can perform score estimation.Yet it remains poorly understood under what settings score estimation is possible, let alone when practical gradient-based algorithms for this task can provably succeed. In this work, we give the first provably efficient results for one of the most fundamental distribution families, Gaussian mixture models.We prove that GD on the denoising diffusion probabilistic model (DDPM) objective can efficiently recover the ground truth parameters of the mixture model in the following two settings:1. We show GD with random initialization learns mixtures of two spherical Gaussians in $d$ dimensions with $1/\text{poly}(d)$-separated centers.2. We show GD with a warm start learns mixtures of $K$ spherical Gaussians with $\Omega(\sqrt{\log(\min(K,d))})$-separated centers.A key ingredient in our proofs is a new connection between score-based methods and two other approaches to distribution learning, EM and spectral methods.

----

## [0] Graph Convolutional Kernel Machine versus Graph Convolutional Networks

**Authors**: *Zhihao Wu, Zhao Zhang, Jicong Fan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec6c6fc9065aa57785eb05dffe7c3db-Abstract-Conference.html)

**Abstract**:

Graph convolutional networks (GCN) with one or two hidden layers have been widely used in handling graph data that are prevalent in various disciplines. Many studies showed that the gain of making GCNs deeper is tiny or even negative. This implies that the complexity of graph data is often limited and shallow models are often sufficient to extract expressive features for various tasks such as node classification. Therefore, in this work, we present a framework called graph convolutional kernel machine (GCKM) for graph-based machine learning. GCKMs are built upon kernel functions integrated with graph convolution. An example is the graph convolutional kernel support vector machine (GCKSVM) for node classification, for which we analyze the generalization error bound and discuss the impact of the graph structure. Compared to GCNs, GCKMs require much less effort in architecture design, hyperparameter tuning, and optimization. More importantly, GCKMs are guaranteed to obtain globally optimal solutions and have strong generalization ability and high interpretability. GCKMs are composable, can be extended to large-scale data, and are applicable to various tasks (e.g., node or graph classification, clustering, feature extraction, dimensionality reduction). The numerical results on benchmark datasets show that, besides the aforementioned advantages, GCKMs have at least competitive accuracy compared to GCNs.

----

## [0] First Order Stochastic Optimization with Oblivious Noise

**Authors**: *Ilias Diakonikolas, Sushrut Karmalkar, Jong Ho Park, Christos Tzamos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ec90b0eec9c1151c152ba865713f184-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ec90b0eec9c1151c152ba865713f184-Abstract-Conference.html)

**Abstract**:

We initiate the study of stochastic optimization with oblivious noise, broadly generalizing the standard heavy-tailed noise setup.In our setting, in addition to random observation noise, the stochastic gradient may be subject to independent \emph{oblivious noise}, which may not have bounded moments and is not necessarily centered. Specifically, we assume access to a noisy oracle for the stochastic gradient of $f$ at $x$,  which returns a vector $\nabla f(\gamma, x) + \xi$, where $\gamma$ is the  bounded variance observation noise and $\xi$ is the oblivious noise that is independent of $\gamma$ and $x$. The only assumption we make on the oblivious noise $\xi$ is that $\Pr[\xi = 0] \ge \alpha$, for some $\alpha \in (0, 1)$.In this setting, it is not information-theoretically possible to recover a single solution close to the target when the fraction of inliers $\alpha$ is less than $1/2$. Our main result is an efficient {\em list-decodable} learner that recovers a small list of candidates at least one of which is close to the true solution. On the other hand, if $\alpha = 1-\epsilon$, where $0< \epsilon < 1/2$ is sufficiently smallconstant, the algorithm recovers a single solution.Along the way, we develop a rejection-sampling-based algorithm to perform noisy location estimation, which may be of independent interest.

----

## [0] CHAMMI: A benchmark for channel-adaptive models in microscopy imaging

**Authors**: *Zitong Sam Chen, Chau Pham, Siqi Wang, Michael Doron, Nikita Moshkov, Bryan A. Plummer, Juan C. Caicedo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ecca655ac67685fdc2155da0eceda6b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ecca655ac67685fdc2155da0eceda6b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Most neural networks assume that input images have a fixed number of channels (three for RGB images). However, there are many settings where the number of channels may vary, such as microscopy images where the number of channels changes depending on instruments and experimental goals. Yet, there has not been a systemic attempt to create and evaluate neural networks that are invariant to the number and type of channels. As a result, trained models remain specific to individual studies and are hardly reusable for other microscopy settings. In this paper, we present a benchmark for investigating channel-adaptive models in microscopy imaging, which consists of 1) a dataset of varied-channel single-cell images, and 2) a biologically relevant evaluation framework. In addition, we adapted several existing techniques to create channel-adaptive models and compared their performance on this benchmark to fixed-channel, baseline models. We find that channel-adaptive models can generalize better to out-of-domain tasks and can be computationally efficient. We contribute a curated dataset and an evaluation API to facilitate objective comparisons in future research and applications.

----

## [0] A Theory of Link Prediction via Relational Weisfeiler-Leman on Knowledge Graphs

**Authors**: *Xingyue Huang, Miguel Romero, Ismail Ilkan Ceylan, Pablo Barceló*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eceb70f47690051d6769739fbf6294b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eceb70f47690051d6769739fbf6294b-Abstract-Conference.html)

**Abstract**:

Graph neural networks are prominent models for representation learning over graph-structured data. While the capabilities and limitations of these models are well-understood for simple graphs, our understanding remains incomplete in the context of knowledge graphs. Our goal is to provide a systematic understanding of the landscape of graph neural networks for knowledge graphs pertaining to the prominent task of link prediction. Our analysis entails a unifying perspective on seemingly unrelated models and unlocks a series of other models. The expressive power of various models is characterized via a corresponding relational Weisfeiler-Leman algorithm. This analysis is extended to provide a precise logical characterization of the class of functions captured by a class of graph neural networks. The theoretical findings presented in this paper explain the benefits of some widely employed practical design choices, which are validated empirically.

----

## [0] Bayes beats Cross Validation: Efficient and Accurate Ridge Regression via Expectation Maximization

**Authors**: *Shu Yu Tew, Mario Boley, Daniel F. Schmidt*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3eec5006051d9544e717067de3220198-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3eec5006051d9544e717067de3220198-Abstract-Conference.html)

**Abstract**:

We present a novel method for tuning the regularization hyper-parameter, $\lambda$, of a ridge regression that is faster to compute than leave-one-out cross-validation (LOOCV) while yielding estimates of the regression parameters of equal, or particularly in the setting of sparse covariates, superior quality to those obtained by minimising the LOOCV risk. The LOOCV risk can suffer from multiple and bad local minima for finite $n$ and thus requires the specification of a set of candidate $\lambda$, which can fail to provide good solutions. In contrast, we show that the proposed method is guaranteed to find a unique optimal solution for large enough $n$, under relatively mild conditions, without requiring the specification of any difficult to determine hyper-parameters. This is based on a Bayesian formulation of ridge regression that we prove to have a unimodal posterior for large enough $n$, allowing for  both the optimal $\lambda$ and the regression coefficients to be jointly learned within an iterative expectation maximization (EM) procedure. Importantly, we show that by utilizing an appropriate preprocessing step, a single iteration of the main EM loop can be implemented in $O(\min(n, p))$ operations, for input data with $n$ rows and $p$ columns. In contrast, evaluating a single value of $\lambda$ using fast LOOCV costs $O(n \min(n, p))$ operations when using the same preprocessing. This advantage amounts to an asymptotic improvement of a factor of $l$ for $l$ candidate values for $\lambda$ (in the regime $q, p \in O(\sqrt{n})$ where $q$ is the number of regression targets).

----

## [0] Segment Everything Everywhere All at Once

**Authors**: *Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Wang, Lijuan Wang, Jianfeng Gao, Yong Jae Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ef61f7e4afacf9a2c5b71c726172b86-Abstract-Conference.html)

**Abstract**:

In this work, we present SEEM, a promotable and interactive model for segmenting everything everywhere all at once in an image. In SEEM, we propose a novel and versatile decoding mechanism that enables diverse prompting for all types of segmentation tasks, aiming at a universal interface that behaves like large language models (LLMs). More specifically, SEEM is designed with four desiderata:i) Versatility. We introduce a new visual prompt to unify different spatial queries including points, boxes, scribbles, and masks, which can further generalize to a different referring image; ii) Compositionality. We learn a joint visual-semantic space between text and visual prompts, which facilitates the dynamic composition of two prompt types required for various segmentation tasks, as shown in Fig. 1;iii) Interactivity. We further incorporate learnable memory prompts into the decoder to retain segmentation history through mask-guided cross-attention from the decoder to image features; iv) Semantic awareness. We use a text encoder to encode text queries and mask labels into the same semantic space for open-vocabulary segmentation. We conduct a comprehensive empirical study to validate the effectiveness of SEEM across diverse segmentation tasks. The results demonstrate that SEEM exhibits robust generalizing to unseen user intents as it learns to compose prompts of different types in a unified representation space. Our approach achieves competitive performance on interactive segmentation, generic segmentation, referring segmentation, and video object segmentation on 9 datasets with minimum 1/100 supervision in a single set of weights.

----

## [0] PUe: Biased Positive-Unlabeled Learning Enhancement by Causal Inference

**Authors**: *Xutao Wang, Hanting Chen, Tianyu Guo, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html)

**Abstract**:

Positive-Unlabeled (PU) learning aims to achieve high-accuracy binary classification with limited labeled positive examples and numerous unlabeled ones. Existing cost-sensitive-based methods often rely on strong assumptions that examples with an observed positive label were selected entirely at random. In fact, the uneven distribution of labels is prevalent in real-world PU problems, indicating that most actual positive and unlabeled data are subject to selection bias. In this paper, we propose a PU learning enhancement (PUe) algorithm based on causal inference theory, which employs normalized propensity scores and normalized inverse probability weighting (NIPW) techniques to reconstruct the loss function, thus obtaining a consistent, unbiased estimate of the classifier and enhancing the model's performance. Moreover, we investigate and propose a method for estimating propensity scores in deep learning using regularization techniques when the labeling mechanism is unknown. Our experiments on three benchmark datasets demonstrate the proposed PUe algorithm significantly improves the accuracy of classifiers on non-uniform label distribution datasets compared to advanced cost-sensitive PU methods. Codes are available at https://github.com/huawei-noah/Noah-research/tree/master/PUe and https://gitee.com/mindspore/models/tree/master/research/cv/PUe.

----

## [0] Sparse Modular Activation for Efficient Sequence Modeling

**Authors**: *Liliang Ren, Yang Liu, Shuohang Wang, Yichong Xu, Chenguang Zhu, ChengXiang Zhai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f0739410e1c9c5da04fa10c1f3f86b6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f0739410e1c9c5da04fa10c1f3f86b6-Abstract-Conference.html)

**Abstract**:

Recent hybrid models combining Linear State Space Models (SSMs) with self-attention mechanisms have demonstrated impressive results across a range of sequence modeling tasks. However, current approaches apply attention modules statically and uniformly to all elements in the input sequences, leading to sub-optimal quality-efficiency trade-offs. To address this limitation, we introduce Sparse Modular Activation (SMA), a general mechanism enabling neural networks to sparsely and dynamically activate sub-modules for sequence elements in a differentiable manner. Through allowing each element to skip non-activated sub-modules, SMA reduces computation and memory consumption of neural networks at both training and inference stages. To validate the effectiveness of SMA on sequence modeling, we design a novel neural architecture, SeqBoat, which employs SMA to sparsely activate a Gated Attention Unit (GAU) based on the state representations learned from an SSM. By constraining the GAU to only conduct local attention on the activated inputs, SeqBoat can achieve linear inference complexity with theoretically infinite attention span, and provide substantially better quality-efficiency trade-off than the chunking-based models. With experiments on a wide range of tasks, including long sequence modeling, speech classification and language modeling, SeqBoat brings new state-of-the-art results among hybrid models with linear complexity, and reveals the amount of attention needed for each task through the learned sparse activation patterns. Our code is publicly available at https://github.com/renll/SeqBoat.

----

## [0] BuildingsBench: A Large-Scale Dataset of 900K Buildings and Benchmark for Short-Term Load Forecasting

**Authors**: *Patrick Emami, Abhijeet Sahu, Peter Graf*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f17bf868966df01ca125e5bbc9ee24e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f17bf868966df01ca125e5bbc9ee24e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Short-term forecasting of residential and commercial building energy consumption is widely used in power systems and continues to grow in importance. Data-driven short-term load forecasting (STLF), although promising, has suffered from a lack of open, large-scale datasets with high building diversity. This has hindered exploring the pretrain-then-fine-tune paradigm for STLF. To help address this, we present BuildingsBench, which consists of: 1) Buildings-900K, a large-scale dataset of 900K simulated buildings representing the U.S. building stock; and 2) an evaluation platform with over 1,900 real residential and commercial buildings from 7 open datasets. BuildingsBench benchmarks two under-explored tasks: zero-shot STLF, where a pretrained model is evaluated on unseen buildings without fine-tuning, and transfer learning, where a pretrained model is fine-tuned on a target building. The main finding of our benchmark analysis is that synthetically pretrained models generalize surprisingly well to real commercial buildings. An exploration of the effect of increasing dataset size and diversity on zero-shot commercial building performance reveals a power-law with diminishing returns. We also show that fine-tuning pretrained models on real commercial and residential buildings improves performance for a majority of target buildings. We hope that BuildingsBench encourages and facilitates future research on generalizable STLF. All datasets and code can be accessed from https://github.com/NREL/BuildingsBench.

----

## [0] Efficient Bayesian Learning Curve Extrapolation using Prior-Data Fitted Networks

**Authors**: *Steven Adriaensen, Herilalaina Rakotoarison, Samuel Müller, Frank Hutter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f1a5e8bfcc3005724d246abe454c1e5-Abstract-Conference.html)

**Abstract**:

Learning curve extrapolation aims to predict model performance in later epochs of training, based on the performance in earlier epochs.In this work, we argue that, while the inherent uncertainty in the extrapolation of learning curves warrants a Bayesian approach, existing methods are (i) overly restrictive, and/or (ii) computationally expensive. We describe the first application of prior-data fitted neural networks (PFNs) in this context. A PFN is a transformer, pre-trained on data generated from a prior, to perform approximate Bayesian inference in a single forward pass. We propose LC-PFN, a PFN trained to extrapolate 10 million artificial right-censored learning curves generated from a parametric prior proposed in prior art using MCMC. We demonstrate that LC-PFN can approximate the posterior predictive distribution more accurately than MCMC, while being over 10 000 times faster. We also show that the same LC-PFN achieves competitive performance extrapolating a total of 20 000 real learning curves from four learning curve benchmarks (LCBench, NAS-Bench-201, Taskset, and PD1) that stem from training a wide range of model architectures (MLPs, CNNs, RNNs, and Transformers) on 53 different datasets with varying input modalities (tabular, image, text, and protein data). Finally, we investigate its potential in the context of model selection and find that a simple LC-PFN based predictive early stopping criterion obtains 2 - 6x speed-ups on 45 of these datasets, at virtually no overhead.

----

## [0] Unified Off-Policy Learning to Rank: a Reinforcement Learning Perspective

**Authors**: *Zeyu Zhang, Yi Su, Hui Yuan, Yiran Wu, Rishab Balasubramanian, Qingyun Wu, Huazheng Wang, Mengdi Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f1b6e97a5eb3b10e6b0c99b022988eb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f1b6e97a5eb3b10e6b0c99b022988eb-Abstract-Conference.html)

**Abstract**:

Off-policy Learning to Rank (LTR) aims to optimize a ranker from data collected by a deployed logging policy. However, existing off-policy learning to rank methods often make strong assumptions about how users generate the click data, i.e., the click model, and hence need to tailor their methods specifically under different click models. In this paper, we unified the ranking process under general stochastic click models as a Markov Decision Process (MDP), and the optimal ranking could be learned with offline reinforcement learning (RL) directly. Building upon this, we leverage offline RL techniques for off-policy LTR and propose the Click Model-Agnostic Unified Off-policy Learning to Rank (CUOLR) method, which could be easily applied to a wide range of click models. Through a dedicated formulation of the MDP, we show that offline RL algorithms can adapt to various click models without complex debiasing techniques and prior knowledge of the model. Results on various large-scale datasets demonstrate that CUOLR consistently outperforms the state-of-the-art off-policy learning to rank algorithms while maintaining consistency and robustness under different click models.

----

## [0] Trust Region-Based Safe Distributional Reinforcement Learning for Multiple Constraints

**Authors**: *Dohyeong Kim, Kyungjae Lee, Songhwai Oh*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f20f2b0315c72201e23512fdbd1ee91-Abstract-Conference.html)

**Abstract**:

In safety-critical robotic tasks, potential failures must be reduced, and multiple constraints must be met, such as avoiding collisions, limiting energy consumption, and maintaining balance.Thus, applying safe reinforcement learning (RL) in such robotic tasks requires to handle multiple constraints and use risk-averse constraints rather than risk-neutral constraints.To this end, we propose a trust region-based safe RL algorithm for multiple constraints called a safe distributional actor-critic (SDAC).Our main contributions are as follows: 1) introducing a gradient integration method to manage infeasibility issues in multi-constrained problems, ensuring theoretical convergence, and 2) developing a TD($\lambda$) target distribution to estimate risk-averse constraints with low biases. We evaluate SDAC through extensive experiments involving multi- and single-constrained robotic tasks.While maintaining high scores, SDAC shows 1.93 times fewer steps to satisfy all constraints in multi-constrained tasks and 1.78 times fewer constraint violations in single-constrained tasks compared to safe RL baselines.Code is available at: https://github.com/rllab-snu/Safe-Distributional-Actor-Critic.

----

## [0] The Contextual Lasso: Sparse Linear Models via Deep Neural Networks

**Authors**: *Ryan Thompson, Amir Dezfouli, Robert Kohn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f226824426a4d6ae3d3efad8883fc53-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f226824426a4d6ae3d3efad8883fc53-Abstract-Conference.html)

**Abstract**:

Sparse linear models are one of several core tools for interpretable machine learning, a field of emerging importance as predictive models permeate decision-making in many domains. Unfortunately, sparse linear models are far less flexible as functions of their input features than black-box models like deep neural networks. With this capability gap in mind, we study a not-uncommon situation where the input features dichotomize into two groups: explanatory features, which are candidates for inclusion as variables in an interpretable model, and contextual features, which select from the candidate variables and determine their effects. This dichotomy leads us to the contextual lasso, a new statistical estimator that fits a sparse linear model to the explanatory features such that the sparsity pattern and coefficients vary as a function of the contextual features. The fitting process learns this function nonparametrically via a deep neural network. To attain sparse coefficients, we train the network with a novel lasso regularizer in the form of a projection layer that maps the network's output onto the space of $\ell_1$-constrained linear models. An extensive suite of experiments on real and synthetic data suggests that the learned models, which remain highly transparent, can be sparser than the regular lasso without sacrificing the predictive power of a standard deep neural network.

----

## [0] No Representation Rules Them All in Category Discovery

**Authors**: *Sagar Vaze, Andrea Vedaldi, Andrew Zisserman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f52ab4322e967efd312c38a68d07f01-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f52ab4322e967efd312c38a68d07f01-Abstract-Conference.html)

**Abstract**:

In this paper we tackle the problem of Generalized Category Discovery (GCD). Specifically, given a dataset with labelled and unlabelled images, the task is to cluster all images in the unlabelled subset, whether or not they belong to the labelled categories. Our first contribution is to recognise that most existing GCD benchmarks only contain labels for a single clustering of the data, making it difficult to ascertain whether models are leveraging the available labels to solve the GCD task, or simply solving an unsupervised clustering problem. As such, we present a synthetic dataset, named 'Clevr-4', for category discovery. Clevr-4 contains four equally valid partitions of the data, i.e based on object 'shape', 'texture' or 'color' or 'count'. To solve the task, models are required to extrapolate the taxonomy specified by labelled set, rather than simply latch onto a single natural grouping of the data. We use this dataset to demonstrate the limitations of unsupervised clustering in the GCD setting, showing that even very strong unsupervised models fail on Clevr-4. We further use Clevr-4 to examine the weaknesses of existing GCD algorithms, and propose a new method which addresses these shortcomings, leveraging consistent findings from the representation learning literature to do so. Our simple solution, which is based on `Mean Teachers' and termed $\mu$GCD, substantially outperforms implemented baselines on Clevr-4. Finally, when we transfer these findings to real data on the challenging Semantic Shift Benchmark suite, we find that $\mu$GCD outperforms all prior work, setting a new state-of-the-art.

----

## [0] CS4ML: A general framework for active learning with arbitrary data based on Christoffel functions

**Authors**: *Juan M. Cardenas, Ben Adcock, Nick C. Dexter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3f8c7eb848ffec848f3ed2b7ca44915d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3f8c7eb848ffec848f3ed2b7ca44915d-Abstract-Conference.html)

**Abstract**:

We introduce a general framework for active learning in regression problems. Our framework extends the standard setup by allowing for general types of data, rather than merely pointwise samples of the target function. This generalization covers many cases of practical interest, such as data acquired in transform domains (e.g., Fourier data), vector-valued data (e.g., gradient-augmented data), data acquired along continuous curves, and, multimodal data (i.e., combinations of different types of measurements).  Our framework considers random sampling according to a finite number of sampling measures and arbitrary nonlinear approximation spaces (model classes). We introduce the concept of \textit{generalized Christoffel functions} and show how these can be used to optimize the sampling measures. We prove that this leads to near-optimal sample complexity in various important cases. This paper focuses on applications in scientific computing, where active learning is often desirable, since it is usually expensive to generate data. We demonstrate the efficacy of our framework for gradient-augmented learning with polynomials, Magnetic Resonance Imaging (MRI) using generative models and adaptive sampling for solving PDEs using Physics-Informed Neural Networks (PINNs).

----

## [0] Two Heads are Better Than One: A Simple Exploration Framework for Efficient Multi-Agent Reinforcement Learning

**Authors**: *Jiahui Li, Kun Kuang, Baoxiang Wang, Xingchen Li, Fei Wu, Jun Xiao, Long Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fa2d2b637122007845a2fbb7c21453b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fa2d2b637122007845a2fbb7c21453b-Abstract-Conference.html)

**Abstract**:

Exploration strategy plays an important role in reinforcement learning, especially in sparse-reward tasks. In cooperative multi-agent reinforcement learning~(MARL), designing a suitable exploration strategy is much more challenging due to the large state space and the complex interaction among agents. Currently, mainstream exploration methods in MARL either contribute to exploring the unfamiliar states which are large and sparse, or measuring the interaction among agents with high computational costs. We found an interesting phenomenon that different kinds of exploration plays a different role in different MARL scenarios, and choosing a suitable one is often more effective than designing an exquisite algorithm. In this paper, we propose a exploration method that incorporate the \underline{C}uri\underline{O}sity-based and \underline{IN}fluence-based exploration~(COIN) which is simple but effective in various situations. First, COIN measures the influence of each agent on the other agents based on mutual information theory and designs it as intrinsic rewards which are applied to each individual value function. Moreover, COIN computes the curiosity-based intrinsic rewards via prediction errors which are added to the extrinsic reward. For integrating the two kinds of intrinsic rewards, COIN utilizes a novel framework in which they complement each other and lead to a sufficient and effective exploration on cooperative MARL tasks. We perform extensive experiments on different challenging benchmarks, and results across different scenarios show the superiority of our method.

----

## [0] Cross-Scale MAE: A Tale of Multiscale Exploitation in Remote Sensing

**Authors**: *Maofeng Tang, Andrei Cozma, Konstantinos Georgiou, Hairong Qi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fadcbd0437f4717723ff3f6f7216800-Abstract-Conference.html)

**Abstract**:

Remote sensing images present unique challenges to image analysis due to the extensive geographic coverage, hardware limitations, and misaligned multi-scale images. This paper revisits the classical multi-scale representation learning prob- lem but under the general framework of self-supervised learning for remote sensing image understanding. We present Cross-Scale MAE, a self-supervised model built upon the Masked Auto-Encoder (MAE). During pre-training, Cross-Scale MAE employs scale augmentation techniques and enforces cross-scale consistency constraints through both contrastive and generative losses to ensure consistent and meaningful representations well-suited for a wide range of downstream tasks. Further, our implementation leverages the xFormers library to accelerate network pre-training on a single GPU while maintaining the quality of learned represen- tations. Experimental evaluations demonstrate that Cross-Scale MAE exhibits superior performance compared to standard MAE and other state-of-the-art remote sensing MAE methods.

----

## [0] MotionGPT: Human Motion as a Foreign Language

**Authors**: *Biao Jiang, Xin Chen, Wen Liu, Jingyi Yu, Gang Yu, Tao Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html)

**Abstract**:

Though the advancement of pre-trained large language models unfolds, the exploration of building a unified model for language and other multimodal data, such as motion, remains challenging and untouched so far. Fortunately, human motion displays a semantic coupling akin to human language, often perceived as a form of body language. By fusing language data with large-scale motion models, motion-language pre-training that can enhance the performance of motion-related tasks becomes feasible. Driven by this insight, we propose MotionGPT, a unified, versatile, and user-friendly motion-language model to handle multiple motion-relevant tasks. Specifically, we employ the discrete vector quantization for human motion and transfer 3D motion into motion tokens, similar to the generation process of word tokens. Building upon this "motion vocabulary", we perform language modeling on both motion and text in a unified manner, treating human motion as a specific language. Moreover, inspired by prompt learning, we pre-train MotionGPT with a mixture of motion-language data and fine-tune it on prompt-based question-and-answer tasks. Extensive experiments demonstrate that MotionGPT achieves state-of-the-art performances on multiple motion tasks including text-driven motion generation, motion captioning, motion prediction, and motion in-between.

----

## [0] Model-Free Reinforcement Learning with the Decision-Estimation Coefficient

**Authors**: *Dylan J. Foster, Noah Golowich, Jian Qian, Alexander Rakhlin, Ayush Sekhari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fcd0f8747f9217c6dbc45ed138b1fde-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fcd0f8747f9217c6dbc45ed138b1fde-Abstract-Conference.html)

**Abstract**:

We consider the problem of interactive decision making, encompassing structured bandits and reinforcementlearning with general function approximation. Recently, Foster et al. (2021) introduced theDecision-Estimation Coefficient, a measure of statistical complexity that lower bounds the optimal regret for interactive decisionmaking, as well as a meta-algorithm, Estimation-to-Decisions, which achieves upperbounds in terms of the same quantity. Estimation-to-Decisions is a reduction, which liftsalgorithms for (supervised) online estimation into algorithms fordecision making. In this paper, we show that by combining Estimation-to-Decisions witha specialized form of "optimistic" estimation introduced byZhang (2022), it is possible to obtain guaranteesthat improve upon those of Foster et al. (2021) byaccommodating more lenient notions of estimation error. We use this approach to derive regret bounds formodel-free reinforcement learning with value function approximation, and give structural results showing when it can and cannot help more generally.

----

## [0] FlowPG: Action-constrained Policy Gradient with Normalizing Flows

**Authors**: *Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3fd9fe8ec6d7238bf71784797399bb61-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3fd9fe8ec6d7238bf71784797399bb61-Abstract-Conference.html)

**Abstract**:

Action-constrained reinforcement learning (ACRL) is a popular approach for solving safety-critical and resource-allocation related decision making problems. A major challenge in ACRL is to ensure agent taking a valid action satisfying constraints in each RL step. Commonly used approach of using a projection layer on top of the policy network requires solving an optimization program which can result in longer training time, slow convergence, and zero gradient problem. To address this, first we use a normalizing flow model to learn an invertible, differentiable mapping between the feasible action space and the support of a simple distribution on a latent variable, such as Gaussian. Second, learning the flow model requires sampling from the feasible action space, which is also challenging. We develop multiple methods, based on Hamiltonian Monte-Carlo and probabilistic sentential decision diagrams for such action sampling for convex and non-convex constraints. Third, we integrate the learned normalizing flow with the DDPG algorithm. By design, a well-trained normalizing flow will transform policy output into a valid action without requiring an optimization solver. Empirically, our approach results in significantly fewer constraint violations (upto an order-of-magnitude for several instances) and is multiple times faster on a variety of continuous control tasks.

----

## [0] Distributionally Robust Bayesian Optimization with φ-divergences

**Authors**: *Hisham Husain, Vu Nguyen, Anton van den Hengel*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3feb8ed3c33c3310b45f80be7dfef707-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3feb8ed3c33c3310b45f80be7dfef707-Abstract-Conference.html)

**Abstract**:

The study of robustness has received much attention due to its inevitability in data-driven settings where many systems face uncertainty. One such example of concern is Bayesian Optimization (BO), where uncertainty is multi-faceted, yet there only exists a limited number of works dedicated to this direction. In particular, there is the work of Kirschner et al., which bridges the existing literature of Distributionally Robust Optimization (DRO) by casting the BO problem from the lens of DRO. While this work is pioneering, it admittedly suffers from various practical shortcomings such as finite contexts assumptions, leaving behind the main question \textit{Can one devise a computationally tractable algorithm for solving this DRO-BO problem}? In this work, we tackle this question to a large degree of generality by considering robustness against data-shift in $\varphi$-divergences, which subsumes many popular choices, such as the $\chi^2$-divergence, Total Variation, and the extant Kullback-Leibler (KL) divergence. We show that the DRO-BO problem in this setting is equivalent to a finite-dimensional optimization problem which, even in the continuous context setting, can be easily implemented with provable sublinear regret bounds. We then show experimentally that our method surpasses existing methods, attesting to the theoretical results.

----

## [0] Connected Superlevel Set in (Deep) Reinforcement Learning and its Application to Minimax Theorems

**Authors**: *Sihan Zeng, Thinh T. Doan, Justin Romberg*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/3ff48dde82306fe8f26f3e51dd1054d7-Abstract-Conference.html)

**Abstract**:

The aim of this paper is to improve the understanding of the optimization landscape for policy optimization problems in reinforcement learning. Specifically, we show that the superlevel set of the objective function with respect to the policy parameter is always a connected set both in the tabular setting and under policies represented by a class of neural networks. In addition, we show that the optimization objective as a function of the policy parameter and reward satisfies a stronger “equiconnectedness” property. To our best knowledge, these are novel and previously unknown discoveries.We present an application of the connectedness of these superlevel sets to the derivation of minimax theorems for robust reinforcement learning. We show that any minimax optimization program which is convex on one side and is equiconnected on the other side observes the minimax equality (i.e. has a Nash equilibrium). We find that this exact structure is exhibited by an interesting class of robust reinforcement learning problems under an adversarial reward attack, and the validity of its minimax equality immediately follows. This is the first time such a result is established in the literature.

----

## [0] Towards Efficient and Accurate Winograd Convolution via Full Quantization

**Authors**: *Tianqi Chen, Weixiang Xu, Weihan Chen, Peisong Wang, Jian Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/400a2e6a82520b690810b97fd67fcc4e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/400a2e6a82520b690810b97fd67fcc4e-Abstract-Conference.html)

**Abstract**:

The Winograd algorithm is an efficient convolution implementation, which performs calculations in the transformed domain. To further improve the computation efficiency, recent works propose to combine it with model quantization. Although Post-Training Quantization has the advantage of low computational cost and has been successfully applied in many other scenarios, a severe accuracy drop exists when utilizing it in Winograd convolution. Besides, despite the Winograd algorithm consisting of four stages, most existing methods only quantize the element-wise multiplication stage, leaving a considerable portion of calculations in full precision.In this paper, observing the inconsistency among different transformation procedures, we present PTQ-Aware Winograd (PAW) to optimize them collaboratively under a unified objective function. Moreover, we explore the full quantization of faster Winograd (tile size $\geq4$) for the first time. We further propose a hardware-friendly method called Factorized Scale Quantization (FSQ), which can effectively balance the significant range differences in the Winograd domain. Experiments demonstrate the effectiveness of our method, e.g., with 8-bit quantization and a tile size of 6, our method outperforms the previous Winograd PTQ method by 8.27\% and 5.38\% in terms of the top-1 accuracy on ResNet-18 and ResNet-34, respectively.

----

## [0] Quantum Bayesian Optimization

**Authors**: *Zhongxiang Dai, Gregory Kang Ruey Lau, Arun Verma, Yao Shu, Bryan Kian Hsiang Low, Patrick Jaillet*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/401aa72e0e3be680348a5b0ffdb1a5aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/401aa72e0e3be680348a5b0ffdb1a5aa-Abstract-Conference.html)

**Abstract**:

Kernelized bandits, also known as Bayesian optimization (BO), has been a prevalent method for optimizing complicated black-box reward functions. Various BO algorithms have been theoretically shown to enjoy upper bounds on their cumulative regret which are sub-linear in the number $T$ of iterations, and a regret lower bound of $\Omega(\sqrt{T})$ has been derived which represents the unavoidable regrets for any classical BO algorithm. Recent works on quantum bandits have shown that with the aid of quantum computing, it is possible to achieve tighter regret upper bounds better than their corresponding classical lower bounds. However, these works are restricted to either multi-armed or linear bandits, and are hence not able to solve sophisticated real-world problems with non-linear reward functions. To this end, we introduce the quantum-Gaussian process-upper confidence bound (Q-GP-UCB) algorithm. To the best of our knowledge, our Q-GP-UCB is the first BO algorithm able to achieve a regret upper bound of $\mathcal{O}(\text{poly}\log T)$, which is significantly smaller than its regret lower bound of $\Omega(\sqrt{T})$ in the classical setting. Moreover, thanks to our novel analysis of the confidence ellipsoid, our Q-GP-UCB with the linear kernel achieves a smaller regret than the quantum linear UCB algorithm from the previous work. We use simulations, as well as an experiment using a real quantum computer, to verify that the theoretical quantum speedup achieved by our Q-GP-UCB is also potentially relevant in practice.

----

## [0] Interpretable Reward Redistribution in Reinforcement Learning: A Causal Approach

**Authors**: *Yudi Zhang, Yali Du, Biwei Huang, Ziyan Wang, Jun Wang, Meng Fang, Mykola Pechenizkiy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/402e12102d6ec3ea3df40ce1b23d423a-Abstract-Conference.html)

**Abstract**:

A major challenge in reinforcement learning is to determine which state-action pairs are responsible for future rewards that are delayed. Reward redistribution serves as a solution to re-assign credits for each time step from observed sequences.  While the majority of current approaches construct the reward redistribution in an uninterpretable manner, we propose to explicitly model the contributions of state and action from a causal perspective, resulting in an interpretable reward redistribution and preserving policy invariance. In this paper, we start by studying the role of causal generative models in reward redistribution by characterizing the generation of Markovian rewards and trajectory-wise long-term return and further propose a framework, called Generative Return Decomposition (GRD), for policy optimization in delayed reward scenarios. Specifically, GRD first identifies the unobservable Markovian rewards and causal relations in the generative process. Then,  GRD makes use of the identified causal generative model to form a compact representation to train policy over the most favorable subspace of the state space of the agent. Theoretically, we show that the unobservable Markovian reward function is identifiable, as well as the underlying causal structure and causal models. Experimental results show that our method outperforms state-of-the-art methods and the provided visualization further demonstrates the interpretability of our method.The project page is located at https://reedzyd.github.io/GenerativeReturnDecomposition/.

----

## [0] Guarantees for Self-Play in Multiplayer Games via Polymatrix Decomposability

**Authors**: *Revan MacQueen, James R. Wright*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40386e4770bebd63fdf47cbc67341c0b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40386e4770bebd63fdf47cbc67341c0b-Abstract-Conference.html)

**Abstract**:

Self-play is a technique for machine learning in multi-agent systems where a learning algorithm learns by interacting with copies of itself. Self-play is useful for generating large quantities of data for learning, but has the drawback that the agents the learner will face post-training may have dramatically different behavior than the learner came to expect by interacting with itself. For the special case of two-player constant-sum games, self-play that reaches Nash equilibrium is guaranteed to produce strategies that perform well against any post-training opponent; however, no such guarantee exists for multiplayer games. We show that in games that approximately decompose into a set of two-player constant-sum games (called constant-sum polymatrix games) where global $\epsilon$-Nash equilibria are boundedly far from Nash equilibria in each subgame (called subgame stability), any no-external-regret algorithm that learns by self-play will produce a strategy with bounded vulnerability. For the first time, our results identify a structural property of multiplayer games that enable performance guarantees for the strategies produced by a broad class of self-play algorithms. We demonstrate our findings through experiments on Leduc poker.

----

## [0] VCC: Scaling Transformers to 128K Tokens or More by Prioritizing Important Tokens

**Authors**: *Zhanpeng Zeng, Cole Hawkins, Mingyi Hong, Aston Zhang, Nikolaos Pappas, Vikas Singh, Shuai Zheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4054556fcaa934b0bf76da52cf4f92cb-Abstract-Conference.html)

**Abstract**:

Transformers are central in modern natural language processing and computer vision applications. Despite recent works devoted to reducing the quadratic cost of such models with respect to sequence length, dealing with ultra long sequences (e.g., $>$16K tokens) remains challenging. Applications such as answering questions based on a book or summarizing a scientific article are inefficient or infeasible. Here, we propose to significantly improve the efficiency of Transformers for ultra long sequences, by compressing the sequence into a much smaller representation at each layer. Specifically, by exploiting the fact that in many tasks, only a small subset of special tokens, which we call VIP-tokens, are most relevant to the final prediction, we propose a VIP-token centric compression (VCC) scheme which selectively compresses the sequence based on their impact on approximating the representation of the VIP-tokens. Compared with competitive baselines, our algorithm is not only efficient (achieving more than $3\times$ compute efficiency gain compared to baselines on 4K and 16K lengths), but also offers competitive/better performance on a large number of tasks. Further, we show that our algorithm scales to 128K tokens (or more) while consistently offering accuracy improvement. Code is available at https://github.com/mlpen/VCC.

----

## [0] Greatness in Simplicity: Unified Self-Cycle Consistency for Parser-Free Virtual Try-On

**Authors**: *Chenghu Du, Junyin Wang, Shuqing Liu, Shengwu Xiong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4065a881baab1744bfba208a4361bbb1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4065a881baab1744bfba208a4361bbb1-Abstract-Conference.html)

**Abstract**:

Image-based virtual try-on tasks remain challenging, primarily due to inherent complexities associated with non-rigid garment deformation modeling and strong feature entanglement of clothing within human body. Recent groundbreaking formulations, such as in-painting, cycle consistency, and knowledge distillation, have facilitated self-supervised generation of try-on images. However, these paradigms necessitate the disentanglement of garment features within human body features through auxiliary tasks, such as leveraging 'teacher knowledge' and dual generators. The potential presence of irresponsible prior knowledge in the auxiliary task can serve as a significant bottleneck for the main generator (e.g., 'student model') in the downstream task. Moreover, existing garment deformation methods lack the ability to perceive the correlation between the garment and the human body in the real world, leading to unrealistic alignment effects. To tackle these limitations, we present a new parser-free virtual try-on network based on unified self-cycle consistency (USC-PFN), which enables robust translation between different garments using just a single generator, faithfully replicating non-rigid geometric deformation of garments in real-life scenarios. Specifically, we first propose a self-cycle consistency architecture with a circular mode. It utilizes real unpaired garment-person images exclusively as input for training, effectively eliminating the impact of irresponsible prior knowledge at the model input end. Additionally, we formulate a Markov Random Field to simulate a more natural and realistic garment deformation. Furthermore, USC-PFN can leverage a general generator for self-supervised cycle training. Experiments demonstrate that our method achieves state-of-the-art performance on a popular virtual try-on benchmark.

----

## [0] VPGTrans: Transfer Visual Prompt Generator across LLMs

**Authors**: *Ao Zhang, Hao Fei, Yuan Yao, Wei Ji, Li Li, Zhiyuan Liu, Tat-Seng Chua*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/407106f4b56040b2e8dcad75a6e461e5-Abstract-Conference.html)

**Abstract**:

Since developing a new multimodal LLM (MLLM) by pre-training on tremendous image-text pairs from scratch can be exceedingly resource-consuming, connecting an existing LLM with a comparatively lightweight visual prompt generator (VPG) becomes a feasible paradigm. However, further tuning the VPG component of the MLLM still incurs significant computational costs, such as thousands of GPU hours and millions of training data points. An alternative solution is transferring an existing VPG from one MLLM to the target MLLM. In this work, we investigate VPG transferability across LLMs for the first time, aiming to reduce the cost of VPG training.  Specifically, we explore VPG transfer across different LLM sizes (e.g., small-to-large) and types.  We identify key factors to maximize transfer efficiency, based on which we develop a simple yet highly effective two-stage transfer framework, called VPGTrans. Notably, it enables VPG transfer from BLIP-2 OPT 2.7B to BLIP-2 OPT 6.7B with less than 10% of the GPU hours using only 10.7% of the training data compared to training a VPG for OPT 6.7B from scratch. Furthermore, we provide a series of intriguing findings and discuss potential explanations behind them. Finally, we showcase the practical value of our VPGTrans approach, by customizing two novel MLLMs, including VL-LLaMA and VL-Vicuna, with recently released LLaMA and Vicuna LLMs.

----

## [0] Nearest Neighbour with Bandit Feedback

**Authors**: *Stephen Pasteris, Chris Hicks, Vasilios Mavroudis*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/4078c8b648dc107aedbdf561dd4edc2a-Abstract-Conference.html)

**Abstract**:

In this paper we adapt the nearest neighbour rule to the contextual bandit problem. Our algorithm handles the fully adversarial setting in which no assumptions at all are made about the data-generation process. When combined with a sufficiently fast data-structure for (perhaps approximate) adaptive nearest neighbour search, such as a navigating net, our algorithm is extremely efficient - having a per trial running time polylogarithmic in both the number of trials and actions, and taking only quasi-linear space. We give generic regret bounds for our algorithm and further analyse them when applied to the stochastic bandit problem in euclidean space. A side result of this paper is that, when applied to the online classification problem with stochastic labels, our algorithm can, under certain conditions, have sublinear regret whilst only finding a single nearest neighbour per trial - in stark contrast to the k-nearest neighbours algorithm.

----

## [0] Generative Neural Fields by Mixtures of Neural Implicit Functions

**Authors**: *Tackgeun You, Mijeong Kim, Jungtaek Kim, Bohyung Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/407fb8c5f3fda374c57d1bb18313ea5d-Abstract-Conference.html)

**Abstract**:

We propose a novel approach to learning the generative neural fields represented by linear combinations of implicit basis networks. Our algorithm learns basis networks in the form of implicit neural representations and their coefficients in a latent space by either conducting meta-learning or adopting auto-decoding paradigms. The proposed method easily enlarges the capacity of generative neural fields by increasing the number of basis networks while maintaining the size of a network for inference to be small through their weighted model averaging. Consequently, sampling instances using the model is efficient in terms of latency and memory footprint. Moreover, we customize denoising diffusion probabilistic model for a target task to sample latent mixture coefficients, which allows our final model to generate unseen data effectively. Experiments show that our approach achieves competitive generation performance on diverse benchmarks for images, voxel data, and NeRF scenes without sophisticated designs for specific modalities and domains.

----

## [0] MAViL: Masked Audio-Video Learners

**Authors**: *Po-Yao Huang, Vasu Sharma, Hu Xu, Chaitanya Ryali, Haoqi Fan, Yanghao Li, Shang-Wen Li, Gargi Ghosh, Jitendra Malik, Christoph Feichtenhofer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40b60852a4abdaa696b5a1a78da34635-Abstract-Conference.html)

**Abstract**:

We present Masked Audio-Video Learners (MAViL) to learn audio-visual representations with three complementary forms of self-supervision: (1) reconstructing masked raw audio and video inputs, (2) intra-modal and inter-modal contrastive learning with masking, and (3) self-training to predict aligned and contextualized audio-video representations learned from the first two objectives. Empirically, MAViL achieves state-of-the-art audio-video classification performance on AudioSet (53.3 mAP) and VGGSound (67.1\% accuracy), surpassing recent self-supervised models and supervised models that utilize external labeled data. Notably, pre-training with MAViL not only enhances performance in multimodal classification and retrieval tasks, but it also improves the representations of each modality in isolation, without relying on information from the other modality during uni-modal fine-tuning or inference. The code and models are available at https://github.com/facebookresearch/MAViL.

----

## [0] Combating Representation Learning Disparity with Geometric Harmonization

**Authors**: *Zhihan Zhou, Jiangchao Yao, Feng Hong, Ya Zhang, Bo Han, Yanfeng Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/40bb79c081828bebdc39d65a82367246-Abstract-Conference.html)

**Abstract**:

Self-supervised learning (SSL) as an effective paradigm of representation learning has achieved tremendous success on various curated datasets in diverse scenarios. Nevertheless, when facing the long-tailed distribution in real-world applications, it is still hard for existing methods to capture transferable and robust representation. The attribution is that the vanilla SSL methods that pursue the sample-level uniformity easily leads to representation learning disparity, where head classes with the huge sample number dominate the feature regime but tail classes with the small sample number passively collapse. To address this problem, we propose a novel Geometric Harmonization (GH) method to encourage the category-level uniformity in representation learning, which is more benign to the minority and almost does not hurt the majority under long-tailed distribution. Specially, GH measures the population statistics of the embedding space on top of self-supervised learning, and then infer an fine-grained instance-wise calibration to constrain the space expansion of head classes and avoid the passive collapse of tail classes. Our proposal does not alter the setting of SSL and can be easily integrated into existing methods in a low-cost manner. Extensive results on a range of benchmark datasets show the effectiveness of \methodspace with high tolerance to the distribution skewness.

----

## [0] BioMassters: A Benchmark Dataset for Forest Biomass Estimation using Multi-modal Satellite Time-series

**Authors**: *Andrea Nascetti, Ritu Yadav, Kirill Brodt, Qixun Qu, Hongwei Fan, Yuri Shendryk, Isha Shah, Christine Chung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/40daf2a00278c4bea1b26cd4c8a654f8-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/40daf2a00278c4bea1b26cd4c8a654f8-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Above Ground Biomass is an important variable as forests play a crucial role in mitigating climate change as they act as an efficient, natural and cost-effective carbon sink. Traditional field and airborne LiDAR measurements have been proven to provide reliable estimations of forest biomass. Nevertheless, the use of these techniques at a large scale can be challenging and expensive. Satellite data have been widely used as a valuable tool in estimating biomass on a global scale. However, the full potential of dense multi-modal satellite time series data, in combination with modern deep learning approaches, has yet to be fully explored. The aim of the "BioMassters" data challenge and benchmark dataset is to investigate the potential of multi-modal satellite data (Sentinel-1 SAR and Sentinel-2 MSI) to estimate forest biomass at a large scale using the Finnish Forest Centre's open forest and nature airborne LiDAR data as a reference. The performance of the top three baseline models shows the potential of deep learning to produce accurate and higher-resolution biomass maps. Our benchmark dataset is publically available at https://huggingface.co/datasets/nascetti-a/BioMassters (doi:10.57967/hf/1009) and the implementation of the top three winning models are available at https://github.com/drivendataorg/the-biomassters.

----

## [0] Online Inventory Problems: Beyond the iid Setting with Online Convex Optimization

**Authors**: *Massil Hihat, Stéphane Gaïffas, Guillaume Garrigos, Simon Bussy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/41128e5b3a7622da5b17588757599077-Abstract-Conference.html)

**Abstract**:

We study multi-product inventory control problems where a manager makes sequential replenishment decisions based on partial historical information in order to minimize its cumulative losses. Our motivation is to consider general demands, losses and dynamics to go beyond standard models which usually rely on newsvendor-type losses, fixed dynamics, and unrealistic i.i.d. demand assumptions. We propose MaxCOSD, an online algorithm that has provable guarantees even for problems with non-i.i.d. demands and stateful dynamics, including for instance perishability. We consider what we call non-degeneracy assumptions on the demand process, and argue that they are necessary to allow learning.

----

## [0] On kernel-based statistical learning theory in the mean field limit

**Authors**: *Christian Fiedler, Michael Herty, Sebastian Trimpe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/411fa9d368b5485be4c6bb62615b365e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/411fa9d368b5485be4c6bb62615b365e-Abstract-Conference.html)

**Abstract**:

In many applications of machine learning, a large number of variables are considered. Motivated by machine learning of interacting particle systems, we consider the situation when the number of input variables goes to infinity. First, we continue the recent investigation of the mean field limit of kernels and their reproducing kernel Hilbert spaces, completing the existing theory. Next, we provide results relevant for approximation with such kernels in the mean field limit, including a representer theorem. Finally, we use these kernels in the context of statistical learning in the mean field limit, focusing on Support Vector Machines. In particular, we show mean field convergence of empirical and infinite-sample solutions as well as the convergence of the corresponding risks. On the one hand, our results establish rigorous mean field limits in the context of kernel methods, providing new theoretical tools and insights for large-scale problems. On the other hand, our setting corresponds to a new form of limit of learning problems, which seems to have not been investigated yet in the statistical learning theory literature.

----

## [0] Benchmarking Encoder-Decoder Architectures for Biplanar X-ray to 3D Bone Shape Reconstruction

**Authors**: *Mahesh Shakya, Bishesh Khanal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/412732f172bdd5ad0efde2fafa110700-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Various deep learning models have been proposed for 3D bone shape reconstruction from two orthogonal (biplanar) X-ray images.However, it is unclear how these models compare against each other since they are evaluated on different anatomy, cohort and (often privately held) datasets.Moreover, the impact of the commonly optimized image-based segmentation metrics such as dice score on the estimation of clinical parameters relevant in 2D-3D bone shape reconstruction is not well known.To move closer toward clinical translation, we propose a benchmarking framework that evaluates tasks relevant to real-world clinical scenarios, including reconstruction of fractured bones, bones with implants, robustness to population shift, and error in estimating clinical parameters.Our open-source platform provides reference implementations of 8 models (many of whose implementations were not publicly available), APIs to easily collect and preprocess 6 public datasets, and the implementation of automatic clinical parameter and landmark extraction methods. We present an extensive evaluation of 8 2D-3D models on equal footing using 6 public datasets comprising images for four different anatomies.Our results show that attention-based methods that capture global spatial relationships tend to perform better across all anatomies and datasets; performance on clinically relevant subgroups may be overestimated without disaggregated reporting; ribs are substantially more difficult to reconstruct compared to femur, hip and spine; and the dice score improvement does not always bring corresponding improvement in the automatic estimation of clinically relevant parameters.

----

## [0] 3D-LLM: Injecting the 3D World into Large Language Models

**Authors**: *Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/413885e70482b95dcbeeddc1daf39177-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/413885e70482b95dcbeeddc1daf39177-Abstract-Conference.html)

**Abstract**:

Large language models (LLMs) and Vision-Language Models (VLMs) have been proved to excel at multiple tasks, such as commonsense reasoning. Powerful as these models can be, they are not grounded in the 3D physical world, which involves richer concepts such as spatial relationships, affordances, physics, layout, and so on. In this work, we propose to inject the 3D world into large language models, and introduce a whole new family of 3D-LLMs. Specifically, 3D-LLMs can take 3D point clouds and their features as input and perform a diverse set of 3D-related tasks, including captioning, dense captioning, 3D question answering, task decomposition, 3Dgrounding, 3D-assisted dialog, navigation, and so on. Using three types of prompting mechanisms that we design, we are able to collect over 300k 3D-language data covering these tasks. To efficiently train 3D-LLMs, we first utilize a 3D feature extractor that obtains 3D features from rendered multi-view images. Then, we use 2D VLMs as our backbones to train our 3D-LLMs. By introducing a 3D localization mechanism, 3D-LLMs could better capture 3D spatial information.  Experiments on ScanQA  show that our model outperforms state-of-the-art baselines by a large margin (\textit{e.g.}, the BLEU-1 score surpasses state-of-the-art score by 9\%). Furthermore, experiments on our held-in datasets for 3D captioning, task composition, and 3D-assisted dialogue show that our model outperforms 2D VLMs. Qualitative examples also show that our model could perform more tasks beyond the scope of existing LLMs and VLMs. Our model and data will be publicly available.

----



[Go to the previous page](NIPS-2023-list800.md)

[Go to the next page](NIPS-2023-list802.md)

[Go to the catalog section](README.md)