## [0] Training Energy-Based Normalizing Flow with Score-Matching Objectives

**Authors**: *Chen-Hao Chao, Wei-Fang Sun, Yen-Chang Hsu, Zsolt Kira, Chun-Yi Lee*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8882d370cdafec9885b918a8cfac642e-Abstract-Conference.html)

**Abstract**:

In this paper, we establish a connection between the parameterization of flow-based and energy-based generative models, and present a new flow-based modeling approach called energy-based normalizing flow (EBFlow). We demonstrate that by optimizing EBFlow with score-matching objectives, the computation of Jacobian determinants for linear transformations can be entirely bypassed. This feature enables the use of arbitrary linear layers in the construction of flow-based models without increasing the computational time complexity of each training iteration from $\mathcal{O}(D^2L)$ to $\mathcal{O}(D^3L)$ for an $L$-layered model that accepts $D$-dimensional inputs. This makes the training of EBFlow more efficient than the commonly-adopted maximum likelihood training method. In addition to the reduction in runtime, we enhance the training stability and empirical performance of EBFlow through a number of techniques developed based on our analysis of the score-matching methods. The experimental results demonstrate that our approach achieves a significant speedup compared to maximum likelihood estimation while outperforming prior methods with a noticeable margin in terms of negative log-likelihood (NLL).

----

## [0] LayoutPrompter: Awaken the Design Ability of Large Language Models

**Authors**: *Jiawei Lin, Jiaqi Guo, Shizhao Sun, Zijiang Yang, Jian-Guang Lou, Dongmei Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88a129e44f25a571ae8b838057c46855-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88a129e44f25a571ae8b838057c46855-Abstract-Conference.html)

**Abstract**:

Conditional graphic layout generation, which automatically maps user constraints to high-quality layouts, has attracted widespread attention today. Although recent works have achieved promising performance, the lack of versatility and data efficiency hinders their practical applications. In this work, we propose LayoutPrompter, which leverages large language models (LLMs) to address the above problems through in-context learning. LayoutPrompter is made up of three key components, namely input-output serialization, dynamic exemplar selection and layout ranking. Specifically, the input-output serialization component meticulously designs the input and output formats for each layout generation task. Dynamic exemplar selection is responsible for selecting the most helpful prompting exemplars for a given input. And a layout ranker is used to pick the highest quality layout from multiple outputs of LLMs. We conduct experiments on all existing layout generation tasks using four public datasets. Despite the simplicity of our approach, experimental results show that LayoutPrompter can compete with or even outperform state-of-the-art approaches on these tasks without any model training or fine-tuning. This demonstrates the effectiveness of this versatile and training-free approach. In addition, the ablation studies show that LayoutPrompter is significantly superior to the training-based baseline in a low-data regime, further indicating the data efficiency of LayoutPrompter. Our project is available at https://github.com/microsoft/LayoutGeneration/tree/main/LayoutPrompter.

----

## [0] Classification of Heavy-tailed Features in High Dimensions: a Superstatistical Approach

**Authors**: *Urte Adomaityte, Gabriele Sicuro, Pierpaolo Vivo*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88be023075a5a3ff3dc3b5d26623fa22-Abstract-Conference.html)

**Abstract**:

We characterise the learning of a mixture of two clouds of data points with generic centroids via empirical risk minimisation in the high dimensional regime, under the assumptions of generic convex loss and convex regularisation. Each cloud of data points is obtained via a double-stochastic process, where the sample is obtained from a Gaussian distribution whose variance is itself a random parameter sampled from a scalar distribution $\varrho$. As a result, our analysis covers a large family of data distributions, including the case of power-law-tailed distributions with no covariance, and allows us to test recent ''Gaussian universality'' claims. We study the generalisation performance of the obtained estimator, we analyse the role of regularisation, and we analytically characterise the separability transition.

----

## [0] CoLA: Exploiting Compositional Structure for Automatic and Efficient Numerical Linear Algebra

**Authors**: *Andres Potapczynski, Marc Finzi, Geoff Pleiss, Andrew Gordon Wilson*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88c3c482430a62d35e03926a22e4b67e-Abstract-Conference.html)

**Abstract**:

Many areas of machine learning and science involve large linear algebra problems, such as eigendecompositions, solving linear systems, computing matrix exponentials, and trace estimation. The matrices involved often have Kronecker, convolutional, block diagonal, sum, or product structure. In this paper, we propose a simple but general framework for large-scale linear algebra problems in machine learning, named CoLA (Compositional Linear Algebra). By combining a linear operator abstraction with compositional dispatch rules, CoLA automatically constructs memory and runtime efficient numerical algorithms. Moreover, CoLA provides memory efficient automatic differentiation, low precision computation, and GPU acceleration in both JAX and PyTorch, while also accommodating new objects, operations, and rules in downstream packages via multiple dispatch. CoLA can accelerate many algebraic operations, while making it easy to prototype matrix structures and algorithms, providing an appealing drop-in tool for virtually any computational effort that requires linear algebra. We showcase its efficacy across a broad range of applications, including partial differential equations, Gaussian processes, equivariant model construction, and unsupervised learning.

----

## [0] Large language models implicitly learn to straighten neural sentence trajectories to construct a predictive representation of natural language

**Authors**: *Eghbal A. Hosseini, Evelina Fedorenko*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/88dddaf430b5bc38ab8228902bb61821-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/88dddaf430b5bc38ab8228902bb61821-Abstract-Conference.html)

**Abstract**:

Predicting upcoming events is critical to our ability to effectively interact with ourenvironment and conspecifics. In natural language processing, transformer models,which are trained on next-word prediction, appear to construct a general-purposerepresentation of language that can support diverse downstream tasks. However, westill lack an understanding of how a predictive objective shapes such representations.Inspired by recent work in vision neuroscience Hénaff et al. (2019), here we test ahypothesis about predictive representations of autoregressive transformer models.In particular, we test whether the neural trajectory of a sequence of words in asentence becomes progressively more straight as it passes through the layers of thenetwork. The key insight behind this hypothesis is that straighter trajectories shouldfacilitate prediction via linear extrapolation. We quantify straightness using a 1-dimensional curvature metric, and present four findings in support of the trajectorystraightening hypothesis: i) In trained models, the curvature progressively decreasesfrom the first to the middle layers of the network. ii) Models that perform better onthe next-word prediction objective, including larger models and models trained onlarger datasets, exhibit greater decreases in curvature, suggesting that this improvedability to straighten sentence neural trajectories may be the underlying driver ofbetter language modeling performance. iii) Given the same linguistic context, thesequences that are generated by the model have lower curvature than the groundtruth (the actual continuations observed in a language corpus), suggesting thatthe model favors straighter trajectories for making predictions. iv) A consistentrelationship holds between the average curvature and the average surprisal ofsentences in the middle layers of models, such that sentences with straighter neuraltrajectories also have lower surprisal. Importantly, untrained models don’t exhibitthese behaviors. In tandem, these results support the trajectory straighteninghypothesis and provide a possible mechanism for how the geometry of the internalrepresentations of autoregressive models supports next word prediction.

----

## [0] Weakly Coupled Deep Q-Networks

**Authors**: *Ibrahim El Shar, Daniel Jiang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8912b4892064a4f08a0c04f92913c134-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8912b4892064a4f08a0c04f92913c134-Abstract-Conference.html)

**Abstract**:

We propose weakly coupled deep Q-networks (WCDQN), a novel deep reinforcement learning algorithm that enhances performance in a class of structured problems called weakly coupled Markov decision processes (WCMDP). WCMDPs consist of multiple independent subproblems connected by an action space constraint, which is a structural property that frequently emerges in practice. Despite this appealing structure, WCMDPs quickly become intractable as the number of subproblems grows. WCDQN employs a single network to train multiple DQN ``subagents,'' one for each subproblem, and then combine their solutions to establish an upper bound on the optimal action value. This guides the main DQN agent towards optimality. We show that the tabular version, weakly coupled Q-learning (WCQL), converges almost surely to the optimal action value. Numerical experiments show faster convergence compared to DQN and related techniques in settings with as many as 10 subproblems, $3^{10}$ total actions, and a continuous state space.

----

## [0] Provably Fast Convergence of Independent Natural Policy Gradient for Markov Potential Games

**Authors**: *Youbang Sun, Tao Liu, Ruida Zhou, P. R. Kumar, Shahin Shahrampour*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8936fa1691764912d9519e1b5673ea66-Abstract-Conference.html)

**Abstract**:

This work studies an independent natural policy gradient (NPG) algorithm for the multi-agent reinforcement learning problem in Markov potential games. It is shown that, under mild technical assumptions and the introduction of the \textit{suboptimality gap}, the independent NPG method with an oracle providing exact policy evaluation asymptotically reaches an $\epsilon$-Nash Equilibrium (NE) within $\mathcal{O}(1/\epsilon)$ iterations. This improves upon the previous best result of $\mathcal{O}(1/\epsilon^2)$ iterations and is of the same order, $\mathcal{O}(1/\epsilon)$, that is achievable for the single-agent case. Empirical results for a synthetic potential game and a congestion game are presented to verify the theoretical bounds.

----

## [0] MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under nonparametrized geometrical variability

**Authors**: *Fabien Casenave, Brian Staber, Xavier Roynard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89379d5fc6eb34ff98488202fb52b9d0-Abstract-Conference.html)

**Abstract**:

When learning simulations for modeling physical phenomena in industrial designs, geometrical variabilities are of prime interest. While classical regression techniques prove effective for parameterized geometries, practical scenarios often involve the absence of shape parametrization during the inference stage, leaving us with only mesh discretizations as available data. Learning simulations from such mesh-based representations poses significant challenges, with recent advances relying heavily on deep graph neural networks to overcome the limitations of conventional machine learning approaches. Despite their promising results, graph neural networks exhibit certain drawbacks, including their dependency on extensive datasets and limitations in providing built-in predictive uncertainties or handling large meshes. In this work, we propose a machine learning method that do not rely on graph neural networks. Complex geometrical shapes and variations with fixed topology are dealt with using well-known mesh morphing onto a common support, combined with classical dimensionality reduction techniques and Gaussian processes. The proposed methodology can easily deal with large meshes without the need for explicit shape parameterization and provides crucial predictive uncertainties, which are essential for informed decision-making. In the considered numerical experiments, the proposed method is competitive with respect to existing graph neural networks, regarding training efficiency and accuracy of the predictions.

----

## [0] Adaptive Online Replanning with Diffusion Models

**Authors**: *Siyuan Zhou, Yilun Du, Shun Zhang, Mengdi Xu, Yikang Shen, Wei Xiao, Dit-Yan Yeung, Chuang Gan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/893a5db6100028ec814cfd99fe92c31b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/893a5db6100028ec814cfd99fe92c31b-Abstract-Conference.html)

**Abstract**:

Diffusion models have risen a promising approach to data-driven planning, and have demonstrated impressive robotic control, reinforcement learning, and video planning performance. Given an effective planner, an important question to consider is replanning -- when given plans should be regenerated due to both action execution error and external environment changes.  Direct plan execution, without replanning, is problematic as errors from individual actions rapidly accumulate and environments are partially observable and stochastic. Simultaneously, replanning at each timestep incurs a substantial computational cost, and may prevent successful task execution, as different generated plans prevent consistent progress to any particular goal. In this paper, we explore how we may effectively replan with diffusion models. We propose a principled approach to determine when to replan, based on the diffusion model's estimated likelihood of existing generated plans. We further present an approach to replan existing trajectories to ensure that new plans follow the same goal state as the original trajectory, which may efficiently bootstrap off previously generated plans.  We illustrate how a combination of our proposed additions significantly improves the performance of diffusion planners leading to 38\% gains over past diffusion planning approaches on Maze2D and further enables handling of stochastic and long-horizon robotic control tasks.

----

## [0] SODA: Robust Training of Test-Time Data Adaptors

**Authors**: *Zige Wang, Yonggang Zhang, Zhen Fang, Long Lan, Wenjing Yang, Bo Han*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/893ca2e5ff5bb258da30e0a82f4c8de9-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/893ca2e5ff5bb258da30e0a82f4c8de9-Abstract-Conference.html)

**Abstract**:

Adapting models deployed to test distributions can mitigate the performance degradation caused by distribution shifts. However, privacy concerns may render model parameters inaccessible. One promising approach involves utilizing zeroth-order optimization (ZOO) to train a data adaptor to adapt the test data to fit the deployed models. Nevertheless, the data adaptor trained with ZOO typically brings restricted improvements due to the potential corruption of data features caused by the data adaptor. To address this issue, we revisit ZOO in the context of test-time data adaptation. We find that the issue directly stems from the unreliable estimation of the gradients used to optimize the data adaptor, which is inherently due to the unreliable nature of the pseudo-labels assigned to the test data. Based on this observation, we propose pseudo-label-robust data adaptation (SODA) to improve the performance of data adaptation. Specifically, SODA leverages high-confidence predicted labels as reliable labels to optimize the data adaptor with ZOO for label prediction. For data with low-confidence predictions, SODA encourages the adaptor to preserve data information to mitigate data corruption. Empirical results indicate that SODA can significantly enhance the performance of deployed models in the presence of distribution shifts without requiring access to model parameters.

----

## [0] Training Neural Networks is NP-Hard in Fixed Dimension

**Authors**: *Vincent Froese, Christoph Hertrich*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8948a8d039ed52d1031db6c7c2373378-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8948a8d039ed52d1031db6c7c2373378-Abstract-Conference.html)

**Abstract**:

We study the parameterized complexity of training two-layer neural networks with respect to the dimension of the input data and the number of hidden neurons, considering ReLU and linear threshold activation functions. Albeit the computational complexity of these problems has been studied numerous times in recent years, several questions are still open. We answer questions by Arora et al. (ICLR 2018) and Khalife and Basu (IPCO 2022) showing that both problems are NP-hard for two dimensions, which excludes any polynomial-time algorithm for constant dimension. We also answer a question by Froese et al. (JAIR 2022) proving W[1]-hardness for four ReLUs (or two linear threshold neurons) with zero training error. Finally, in the ReLU case, we show fixed-parameter tractability for the combined parameter number of dimensions and number of ReLUs if the network is assumed to compute a convex map. Our results settle the complexity status regarding these parameters almost completely.

----

## [0] GlyphControl: Glyph Conditional Control for Visual Text Generation

**Authors**: *Yukang Yang, Dongnan Gui, Yuhui Yuan, Weicong Liang, Haisong Ding, Han Hu, Kai Chen*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8951bbdcf234132bcce680825e7cb354-Abstract-Conference.html)

**Abstract**:

Recently, there has been an increasing interest in developing diffusion-based text-to-image generative models capable of generating coherent and well-formed visual text. In this paper, we propose a novel and efficient approach called GlyphControl to address this task. Unlike existing methods that rely on character-aware text encoders like ByT5 and require retraining of text-to-image models, our approach leverages additional glyph conditional information to enhance the performance of the off-the-shelf Stable-Diffusion model in generating accurate visual text. By incorporating glyph instructions, users can customize the content, location, and size of the generated text according to their specific requirements. To facilitate further research in visual text generation, we construct a training benchmark dataset called LAION-Glyph. We evaluate the effectiveness of our approach by measuring OCR-based metrics, CLIP score, and FID of the generated visual text. Our empirical evaluations demonstrate that GlyphControl outperforms the recent DeepFloyd IF approach in terms of OCR accuracy, CLIP score, and FID, highlighting the efficacy of our method.

----

## [0] Domain Adaptive Imitation Learning with Visual Observation

**Authors**: *Sungho Choi, Seungyul Han, Woojun Kim, Jongseong Chae, Whiyoung Jung, Youngchul Sung*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/899511e37a8e01e1bd6f6f1d377cc250-Abstract-Conference.html)

**Abstract**:

In this paper, we consider domain-adaptive imitation learning with visual observation, where an agent in a target domain learns to perform a task by observing expert demonstrations in a source domain. Domain adaptive imitation learning arises in practical scenarios where a robot, receiving visual sensory data, needs to mimic movements by visually observing other robots from different angles or observing robots of different shapes. To overcome the domain shift in cross-domain imitation learning with visual observation, we propose a novel framework for extracting domain-independent behavioral features from input observations that can be used to train the learner, based on dual feature extraction and image reconstruction. Empirical results demonstrate that our approach outperforms previous algorithms for imitation learning from visual observation with domain shift.

----

## [0] Discriminative Feature Attributions: Bridging Post Hoc Explainability and Inherent Interpretability

**Authors**: *Usha Bhalla, Suraj Srinivas, Himabindu Lakkaraju*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89beb2a345269f3f9afe48cee35403aa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89beb2a345269f3f9afe48cee35403aa-Abstract-Conference.html)

**Abstract**:

With the increased deployment of machine learning models in various real-world applications, researchers and practitioners alike have emphasized the need for explanations of model behaviour. To this end, two broad strategies have been outlined in prior literature to explain models. Post hoc explanation methods explain the behaviour of complex black-box models by identifying features critical to model predictions; however, prior work has shown that these explanations may not be faithful, in that they incorrectly attribute high importance to features that are unimportant or non-discriminative for the underlying task. Inherently interpretable models, on the other hand, circumvent these issues by explicitly encoding explanations into model architecture, meaning their explanations are naturally faithful, but they often exhibit poor predictive performance due to their limited expressive power. In this work, we identify a key reason for the lack of faithfulness of feature attributions: the lack of robustness of the underlying black-box models, especially the erasure of unimportant distractor features in the input. To address this issue, we propose Distractor Erasure Tuning (DiET), a method that adapts black-box models to be robust to distractor erasure, thus providing discriminative and faithful attributions. This strategy naturally combines the ease-of-use of post hoc explanations with the faithfulness of inherently interpretable models. We perform extensive experiments on semi-synthetic and real-world datasets, and show that DiET produces models that (1) closely approximate the original black-box models they are intended to explain, and (2) yield explanations that match approximate ground truths available by construction.

----

## [0] LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models

**Authors**: *Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya K, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Galit Sarfaty, Gregory M. Dickinson, Haggai Porat, Jason Hegland, Jessica Wu, Joe Nudell, Joel Niklaus, John J. Nay, Jonathan H. Choi, Kevin Tobia, Margaret Hagan, Megan Ma, Michael A. Livermore, Nikon Rasumov-Rahe, Nils Holzenberger, Noam Kolt, Peter Henderson, Sean Rehaag, Sharad Goel, Shang Gao, Spencer Williams, Sunny Gandhi, Tom Zur, Varun Iyer, Zehua Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/89e44582fd28ddfea1ea4dcb0ebbf4b0-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The advent of large language models (LLMs) and their adoption by the legal community has given rise to the question: what types of legal reasoning can LLMs perform? To enable greater study of this question, we present LegalBench: a collaboratively constructed legal reasoning benchmark consisting of 162 tasks covering six different types of legal reasoning. LegalBench was built through an interdisciplinary process, in which we collected tasks designed and hand-crafted by legal professionals. Because these subject matter experts took a leading role in construction, tasks either measure legal reasoning capabilities that are practically useful, or measure reasoning skills that lawyers find interesting. To enable cross-disciplinary conversations about LLMs in the law, we additionally show how popular legal frameworks for describing legal reasoning—which distinguish between its many forms—correspond to LegalBench tasks, thus giving lawyers and LLM developers a common vocabulary. This paper describes LegalBench, presents an empirical evaluation of 20 open-source and commercial LLMs, and illustrates the types of research explorations LegalBench enables.

----

## [0] Detecting hidden confounding in observational data using multiple environments

**Authors**: *Rickard Karlsson, Jesse H. Krijthe*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/89e541b817ea043a971840a926e12b37-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/89e541b817ea043a971840a926e12b37-Abstract-Conference.html)

**Abstract**:

A common assumption in causal inference from observational data is that there is no hidden confounding. Yet it is, in general, impossible to verify the presence of hidden confounding factors from a single dataset. Under the assumption of independent causal mechanisms underlying the data-generating process, we demonstrate a way to detect unobserved confounders when having multiple observational datasets coming from different environments. We present a theory for testable conditional independencies that are only absent when there is hidden confounding and examine cases where we violate its assumptions: degenerate & dependent mechanisms, and faithfulness violations. Additionally, we propose a procedure to test these independencies and study its empirical finite-sample behavior using simulation studies and semi-synthetic data based on a real-world dataset. In most cases, the proposed procedure correctly predicts the presence of hidden confounding, particularly when the confounding bias is large.

----

## [0] Information Geometry of the Retinal Representation Manifold

**Authors**: *Xuehao Ding, Dongsoo Lee, Joshua Melander, George Sivulka, Surya Ganguli, Stephen Baccus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a267516a7a697965c6ae4f48b908605-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a267516a7a697965c6ae4f48b908605-Abstract-Conference.html)

**Abstract**:

The ability for the brain to discriminate among visual stimuli is constrained by their retinal representations. Previous studies of visual discriminability have been limited to either low-dimensional artificial stimuli or pure theoretical considerations without a realistic encoding model. Here we propose a novel framework for understanding stimulus discriminability achieved by retinal representations of naturalistic stimuli with the method of information geometry. To model the joint probability distribution of neural responses conditioned on the stimulus, we created a stochastic encoding model of a population of salamander retinal ganglion cells based on a three-layer convolutional neural network model. This model not only accurately captured the mean response to natural scenes but also a variety of second-order statistics. With the model and the proposed theory, we computed the Fisher information metric over stimuli to study the most discriminable stimulus directions. We found that the most discriminable stimulus varied substantially across stimuli, allowing an examination of the relationship between the most discriminable stimulus and the current stimulus. By examining responses generated by the most discriminable stimuli we further found that the most discriminative response mode is often aligned with the most stochastic mode. This finding carries the important implication that under natural scenes, retinal noise correlations are information-limiting rather than increasing information transmission as has been previously speculated. We additionally observed that sensitivity saturates less in the population than for single cells and that as a function of firing rate, Fisher information varies less than sensitivity. We conclude that under natural scenes, population coding benefits from complementary coding and helps to equalize the information carried by different firing rates, which may facilitate decoding of the stimulus under principles of information maximization.

----

## [0] RoboHive: A Unified Framework for Robot Learning

**Authors**: *Vikash Kumar, Rutav M. Shah, Gaoyue Zhou, Vincent Moens, Vittorio Caggiano, Abhishek Gupta, Aravind Rajeswaran*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a84a4341c375b8441b36836bb343d4e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a84a4341c375b8441b36836bb343d4e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present RoboHive, a comprehensive software platform and ecosystem for research in the field of Robot Learning and Embodied Artificial Intelligence. Our platform encompasses a diverse range of pre-existing and novel environments, including dexterous manipulation with the Shadow Hand, whole-arm manipulation tasks with Franka and Fetch robots, quadruped locomotion, among others. Included environments are organized within and cover multiple domains such as hand manipulation, locomotion, multi-task, multi-agent, muscles, etc. In comparison to prior works, RoboHive offers a streamlined and unified task interface taking dependency on only a minimal set of well-maintained packages, features tasks with high physics fidelity and rich visual diversity, and supports common hardware drivers for real-world deployment. The unified interface of RoboHive offers a convenient and accessible abstraction for algorithmic research in imitation, reinforcement, multi-task, and hierarchical learning. Furthermore, RoboHive includes expert demonstrations and baseline results for most environments, providing a standard for benchmarking and comparisons. Details: https://sites.google.com/view/robohive

----

## [0] Sequential Memory with Temporal Predictive Coding

**Authors**: *Mufeng Tang, Helen Barron, Rafal Bogacz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a8b9c7f979e8819a7986b3ef825c08a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a8b9c7f979e8819a7986b3ef825c08a-Abstract-Conference.html)

**Abstract**:

Forming accurate memory of sequential stimuli is a fundamental function of biological agents. However, the computational mechanism underlying sequential memory in the brain remains unclear. Inspired by neuroscience theories and recent successes in applying predictive coding (PC) to \emph{static} memory tasks, in this work we propose a novel PC-based model for \emph{sequential} memory, called \emph{temporal predictive coding} (tPC). We show that our tPC models can memorize and retrieve sequential inputs accurately with a biologically plausible neural implementation. Importantly, our analytical study reveals that tPC can be viewed as a classical Asymmetric Hopfield Network (AHN) with an implicit statistical whitening process, which leads to more stable performance in sequential memory tasks of structured inputs. Moreover, we find that tPC exhibits properties consistent with behavioral observations and theories in neuroscience, thereby strengthening its biological relevance. Our work establishes a possible computational mechanism underlying sequential memory in the brain that can also be theoretically interpreted using existing memory model frameworks.

----

## [0] Transportability for Bandits with Data from Different Environments

**Authors**: *Alexis Bellot, Alan Malek, Silvia Chiappa*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8a8ce53beb3775522305e0a6033d4455-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8a8ce53beb3775522305e0a6033d4455-Abstract-Conference.html)

**Abstract**:

A unifying theme in the design of intelligent agents is to efficiently optimize a policy based on what prior knowledge of the problem is available and what actions can be taken to learn more about it. Bandits are a canonical instance of this task that has been intensely studied in the literature. Most methods, however, typically rely solely on an agent's experimentation in a single environment (or multiple closely related environments). In this paper, we relax this assumption and consider the design of bandit algorithms from a combination of batch data and qualitative assumptions about the relatedness across different environments, represented in the form of causal models. In particular, we show that it is possible to exploit invariances across environments, wherever they may occur in the underlying causal model, to consistently improve learning. The resulting bandit algorithm has a sub-linear regret bound with an explicit dependency on a term that captures how informative related environments are for the task at hand; and may have substantially lower regret than experimentation-only bandit instances.

----

## [0] Students Parrot Their Teachers: Membership Inference on Model Distillation

**Authors**: *Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, Florian Tramèr*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b07d224a643b02e7571e083578a86d2-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b07d224a643b02e7571e083578a86d2-Abstract-Conference.html)

**Abstract**:

Model distillation is frequently proposed as a technique to reduce the privacy leakage of machine learning. These empirical privacy defenses rely on the intuition that distilled student'' models protect the privacy of training data, as they only interact with this data indirectly through ateacher'' model. In this work, we design membership inference attacks to systematically study the privacy provided by knowledge distillation to both the teacher and student training sets. Our new attacks show that distillation alone provides only limited privacy across a number of domains. We explain the success of our attacks on distillation by showing that membership inference attacks on a private dataset can succeed even if the target model is never queried on any actual training points, but only on inputs whose predictions are highly influenced by training data. Finally, we show that our attacks are strongest when student and teacher sets are similar, or when the attacker can poison the teacher set.

----

## [0] DiffuseBot: Breeding Soft Robots With Physics-Augmented Generative Diffusion Models

**Authors**: *Tsun-Hsuan Johnson Wang, Juntian Zheng, Pingchuan Ma, Yilun Du, Byungchul Kim, Andrew Spielberg, Josh B. Tenenbaum, Chuang Gan, Daniela Rus*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b1008098947ad59144c18a78337f937-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b1008098947ad59144c18a78337f937-Abstract-Conference.html)

**Abstract**:

Nature evolves creatures with a high complexity of morphological and behavioral intelligence, meanwhile computational methods lag in approaching that diversity and efficacy.  Co-optimization of artificial creatures' morphology and control in silico shows promise for applications in physical soft robotics and virtual character creation; such approaches, however, require developing new learning algorithms that can reason about function atop pure structure. In this paper, we present DiffuseBot, a physics-augmented diffusion model that generates soft robot morphologies capable of excelling in a wide spectrum of tasks. \name bridges the gap between virtually generated content and physical utility by (i) augmenting the diffusion process with a physical dynamical simulation which provides a certificate of performance, and (ii) introducing a co-design procedure that jointly optimizes physical design and control by leveraging information about physical sensitivities from differentiable simulation.  We showcase a range of simulated and fabricated robots along with their capabilities. Check our website: https://diffusebot.github.io/

----

## [0] Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks

**Authors**: *Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b4add8b0aa8749d80a34ca5d941c355-Abstract-Conference.html)

**Abstract**:

We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset. We demonstrate efficacy of our attack when unlearning is performed via retraining from scratch, the idealized setting of machine unlearning which other efficient methods attempt to emulate, as well as against the approximate unlearning approach of Graves et al. (2021).

----

## [0] Thought Cloning: Learning to Think while Acting by Imitating Human Thinking

**Authors**: *Shengran Hu, Jeff Clune*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b4ba64e549c410185c4d3eac3a81726-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b4ba64e549c410185c4d3eac3a81726-Abstract-Conference.html)

**Abstract**:

Language is often considered a key aspect of human thinking, providing us with exceptional abilities to generalize, explore, plan, replan, and adapt to new situations. However, Reinforcement Learning (RL) agents are far from human-level performance in any of these abilities. We hypothesize one reason for such cognitive deficiencies is that they lack the benefits of thinking in language and that we can improve AI agents by training them to $\textit{think like humans do}$. We introduce a novel Imitation Learning framework, Thought Cloning, where the idea is to not just clone the behaviors of human demonstrators, $\textit{but also the thoughts humans have as they perform these behaviors}$. While we expect Thought Cloning to truly shine at scale on internet-sized datasets (e.g. online videos with transcripts), here we conduct experiments in a domain where the thinking and action data are synthetically generated. Results reveal that Thought Cloning learns much faster than Behavioral Cloning and its performance advantage grows the further out of distribution test tasks are, highlighting its ability to better handle novel situations. Thought Cloning also provides important benefits for AI Safety and Interpretability, and makes it easier to debug and improve AI. Because we can observe the agentâ€™s thoughts, we can (1) more easily diagnose why things are going wrong, making it easier to fix the problem, (2) steer the agent by correcting its thinking, or (3) prevent it from doing unsafe things it plans to do. Overall, by training agents $\textit{how to think}$ as well as behave, Thought Cloning creates safer, more powerful agents.

----

## [0] SNEkhorn: Dimension Reduction with Symmetric Entropic Affinities

**Authors**: *Hugues Van Assel, Titouan Vayer, Rémi Flamary, Nicolas Courty*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b54ecd9823fff6d37e61ece8f87e534-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b54ecd9823fff6d37e61ece8f87e534-Abstract-Conference.html)

**Abstract**:

Many approaches in machine learning rely on a weighted graph to encode thesimilarities between samples in a dataset. Entropic affinities (EAs), which are notably used in the popular Dimensionality Reduction (DR) algorithm t-SNE, are particular instances of such graphs. To ensure robustness to heterogeneous sampling densities, EAs assign a kernel bandwidth parameter to every sample in such a way that the entropy of each row in the affinity matrix is kept constant at a specific value, whose exponential is known as perplexity. EAs are inherently asymmetric and row-wise stochastic, but they are used in DR approaches after undergoing heuristic symmetrization methods that violate both the row-wise constant entropy and stochasticity properties. In this work, we uncover a novel characterization of EA as an optimal transport problem, allowing a natural symmetrization that can be computed efficiently using dual ascent. The corresponding novel affinity matrix derives advantages from symmetric doubly stochastic normalization in terms of clustering performance, while also effectively controlling the entropy of each row thus making it particularly robust to varying noise levels. Following, we present a new DR algorithm, SNEkhorn, that leverages this new affinity matrix. We show its clear superiority to state-of-the-art approaches with several indicators on both synthetic and real-world datasets.

----

## [0] Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach

**Authors**: *Nurendra Choudhary, Nikhil Rao, Chandan K. Reddy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b6a8b010e9a266aad40a024c5976d5c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b6a8b010e9a266aad40a024c5976d5c-Abstract-Conference.html)

**Abstract**:

The progress in hyperbolic neural networks (HNNs) research is hindered by their absence of inductive bias mechanisms, which are essential for generalizing to new tasks and facilitating scalable learning over large datasets. In this paper, we aim to alleviate these issues by learning generalizable inductive biases from the nodesâ€™ local subgraph and transfer them for faster learning over new subgraphs with a disjoint set of nodes, edges, and labels in a few-shot setting. We introduce a novel method, Hyperbolic GRAph Meta Learner (H-GRAM), that, for the tasks of node classification and link prediction, learns transferable information from a set of support local subgraphs in the form of hyperbolic meta gradients and label hyperbolic protonets to enable faster learning over a query set of new tasks dealing with disjoint subgraphs. Furthermore, we show that an extension of our meta-learning framework also mitigates the scalability challenges seen in HNNs faced by existing approaches. Our comparative analysis shows that H-GRAM effectively learns and transfers information in multiple challenging few-shot settings compared to other state-of-the-art baselines. Additionally, we demonstrate that, unlike standard HNNs, our approach is able to scale over large graph datasets and improve performance over its Euclidean counterparts.

----

## [0] FELM: Benchmarking Factuality Evaluation of Large Language Models

**Authors**: *Shiqi Chen, Yiran Zhao, Jinghan Zhang, I-Chun Chern, Siyang Gao, Pengfei Liu, Junxian He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b8a7960d343e023a6a0afe37eee6022-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b8a7960d343e023a6a0afe37eee6022-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Assessing factuality of text generated by large language models (LLMs) is an emerging yet crucial research area, aimed at alerting users to potential errors and guiding the development of more reliable LLMs. Nonetheless, the evaluators assessing factuality necessitate suitable evaluation themselves to gauge progress and foster advancements. This direction remains under-explored, resulting in substantial impediments to the progress of factuality evaluators. To mitigate this issue, we introduce a benchmark for Factuality Evaluation of large Language Models, referred to as FELM. In this benchmark, we collect responses generated from LLMs and annotate factuality labels in a fine-grained manner. Contrary to previous studies that primarily concentrate on the factuality of world knowledge (e.g. information from Wikipedia), FELM focuses on factuality across diverse domains, spanning from world knowledge to math and reasoning. Our annotation is based on text segments, which can help pinpoint specific factual errors. The factuality annotations are further supplemented by predefined error types and reference links that either support or contradict the statement. In our experiments, we investigate the performance of several LLM-based factuality evaluators on FELM, including both vanilla LLMs and those augmented with retrieval mechanisms and chain-of-thought processes. Our findings reveal that while retrieval aids factuality evaluation, current LLMs are far from satisfactory to faithfully detect factual errors.

----

## [0] AircraftVerse: A Large-Scale Multimodal Dataset of Aerial Vehicle Designs

**Authors**: *Adam B. Cobb, Anirban Roy, Daniel Elenius, F. Michael Heim, Brian Swenson, Sydney Whittington, James D. Walker, Theodore Bapty, Joseph Hite, Karthik Ramani, Christopher McComb, Susmit Jha*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8b94879b177d9780c17f5a78f62a6a8a-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8b94879b177d9780c17f5a78f62a6a8a-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We present AircraftVerse, a publicly available aerial vehicle design dataset. Aircraft design encompasses different physics domains and, hence, multiple modalities of representation. The evaluation of these designs requires the use of scientific analytical and simulation models ranging from computer-aided design tools for structural and manufacturing analysis, computational fluid dynamics tools for drag and lift computation, battery models for energy estimation, and simulation models for flight control and dynamics. AircraftVerse contains $27{,}714$  diverse air vehicle designs - the largest corpus of designs with this level of complexity. Each design comprises the following artifacts: a symbolic design tree describing topology, propulsion subsystem, battery subsystem, and other design details; a STandard for the Exchange of Product (STEP) model data; a 3D CAD design using a stereolithography (STL) file format; a 3D point cloud for the shape of the design; and evaluation results from high fidelity state-of-the-art physics models that characterize performance metrics such as maximum flight distance and hover-time. We also present baseline surrogate models that use different modalities of design representation to predict design performance metrics, which we provide as part of our dataset release. Finally, we discuss the potential impact of this dataset on the use of learning in aircraft design, and more generally, in the emerging field of deep learning for scientific design. AircraftVerse is accompanied by a datasheet as suggested in the recent literature, and it is released under Creative Commons Attribution-ShareAlike (CC BY-SA) license. The dataset with baseline models are hosted at http://doi.org/10.5281/zenodo.6525446, code at https://github.com/SRI-CSL/AircraftVerse, and the dataset description at  https://uavdesignverse.onrender.com/.

----

## [0] On Separate Normalization in Self-supervised Transformers

**Authors**: *Xiaohui Chen, Yinkai Wang, Yuanqi Du, Soha Hassoun, Liping Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ba80c47b9d3dced79ee835b7d3bf72a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ba80c47b9d3dced79ee835b7d3bf72a-Abstract-Conference.html)

**Abstract**:

Self-supervised training methods for transformers have demonstrated remarkable performance across various domains. Previous transformer-based models, such as masked autoencoders (MAE), typically utilize a single normalization layer for both the [CLS] symbol and the tokens. We propose in this paper a simple modification that employs separate normalization layers for the tokens and the [CLS] symbol to better capture their distinct characteristics and enhance downstream task performance. Our method aims to alleviate the potential negative effects of using the same normalization statistics for both token types, which may not be optimally aligned with their individual roles. We empirically show that by utilizing a separate normalization layer, the [CLS] embeddings can better encode the global contextual information and are distributed more uniformly in its anisotropic space. When replacing the conventional normalization layer with the two separate layers, we observe an average  2.7% performance improvement over the image, natural language, and graph domains.

----

## [0] PAD: A Dataset and Benchmark for Pose-agnostic Anomaly Detection

**Authors**: *Qiang Zhou, Weize Li, Lihan Jiang, Guoliang Wang, Guyue Zhou, Shanghang Zhang, Hao Zhao*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bc5aef775aacc1650a9790f1428bcea-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bc5aef775aacc1650a9790f1428bcea-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Object anomaly detection is an important problem in the field of machine vision and has seen remarkable progress recently. However, two significant challenges hinder its research and application. First, existing datasets lack comprehensive visual information from various pose angles. They usually have an unrealistic assumption that the anomaly-free training dataset is pose-aligned, and the testing samples have the same pose as the training data. However, in practice, anomaly may exist in any regions on a object, the training and query samples may have different poses, calling for the study on pose-agnostic anomaly detection. Second, the absence of a consensus on experimental protocols for pose-agnostic anomaly detection leads to unfair comparisons of different methods, hindering the research on pose-agnostic anomaly detection. To address these issues, we develop Multi-pose Anomaly Detection (MAD) dataset and Pose-agnostic Anomaly Detection (PAD) benchmark, which takes the first step to address the pose-agnostic anomaly detection problem. Specifically, we build MAD using 20 complex-shaped LEGO toys including 4K views with various poses, and high-quality and diverse 3D anomalies in both simulated and real environments. Additionally, we propose a novel method OmniposeAD, trained using MAD, specifically designed for pose-agnostic anomaly detection. Through comprehensive evaluations, we demonstrate the relevance of our dataset and method. Furthermore, we provide an open-source benchmark library, including dataset and baseline methods that cover 8 anomaly detection paradigms, to facilitate future research and application in this domain. Code, data, and models are publicly available at https://github.com/EricLee0224/PAD.

----

## [0] Modulated Neural ODEs

**Authors**: *Ilze Amanda Auzina, Çagatay Yildiz, Sara Magliacane, Matthias Bethge, Efstratios Gavves*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bc74514d554a90c996576f6c373f5f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bc74514d554a90c996576f6c373f5f3-Abstract-Conference.html)

**Abstract**:

Neural ordinary differential equations (NODEs) have been proven useful for learning non-linear dynamics of arbitrary trajectories. However, current NODE methods capture variations across trajectories only via the initial state value or by auto-regressive encoder updates. In this work, we introduce Modulated Neural ODEs (MoNODEs), a novel framework that sets apart dynamics states from underlying static factors of variation and improves the existing NODE methods.  In particular, we introduce *time-invariant modulator variables* that are learned from the data. We incorporate our proposed framework into four existing NODE variants. We test MoNODE on oscillating systems, videos and human walking trajectories, where each trajectory has trajectory-specific modulation. Our framework consistently improves the existing model ability to generalize to new dynamic parameterizations and to perform far-horizon forecasting. In addition, we verify that the proposed modulator variables are informative of the true unknown factors of variation as measured by $R^2$ scores.

----

## [0] DrugCLIP: Contrasive Protein-Molecule Representation Learning for Virtual Screening

**Authors**: *Bowen Gao, Bo Qiang, Haichuan Tan, Yinjun Jia, Minsi Ren, Minsi Lu, Jingjing Liu, Wei-Ying Ma, Yanyan Lan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bd31288ad8e9a31d519fdeede7ee47d-Abstract-Conference.html)

**Abstract**:

Virtual screening, which identifies potential drugs from vast compound databases to bind with a particular protein pocket, is a critical step in AI-assisted drug discovery. Traditional docking methods are highly time-consuming, and can only work with a restricted search library in real-life applications. Recent supervised learning approaches using scoring functions for binding-affinity prediction, although promising, have not yet surpassed docking methods due to their strong dependency on limited data with reliable binding-affinity labels. In this paper, we propose a novel contrastive learning framework, DrugCLIP, by reformulating virtual screening as a dense retrieval task and employing contrastive learning to align representations of binding protein pockets and molecules from a large quantity of pairwise data without explicit binding-affinity scores. We also introduce a biological-knowledge inspired data augmentation strategy to learn better protein-molecule representations. Extensive experiments show that DrugCLIP significantly outperforms traditional docking and supervised learning methods on diverse virtual screening benchmarks with highly reduced computation time, especially in zero-shot setting.

----

## [0] On the Convergence of Black-Box Variational Inference

**Authors**: *Kyurae Kim, Jisu Oh, Kaiwen Wu, Yian Ma, Jacob R. Gardner*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8bea36ac39e11ebe49e9eddbd4b8bd3a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8bea36ac39e11ebe49e9eddbd4b8bd3a-Abstract-Conference.html)

**Abstract**:

We provide the first convergence guarantee for black-box variational inference (BBVI) with the reparameterization gradient.  While preliminary investigations worked on simplified versions of BBVI (e.g., bounded domain, bounded support, only optimizing for the scale, and such), our setup does not need any such algorithmic modifications.  Our results hold for log-smooth posterior densities with and without strong log-concavity and the location-scale variational family.  Notably, our analysis reveals that certain algorithm design choices commonly employed in practice, such as nonlinear parameterizations of the scale matrix, can result in suboptimal convergence rates.  Fortunately, running BBVI with proximal stochastic gradient descent fixes these limitations and thus achieves the strongest known convergence guarantees.  We evaluate this theoretical insight by comparing proximal SGD against other standard implementations of BBVI on large-scale Bayesian inference problems.

----

## [0] DRAUC: An Instance-wise Distributionally Robust AUC Optimization Framework

**Authors**: *Siran Dai, Qianqian Xu, Zhiyong Yang, Xiaochun Cao, Qingming Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c086821724b99f4c756648bb0f165db-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c086821724b99f4c756648bb0f165db-Abstract-Conference.html)

**Abstract**:

The Area Under the ROC Curve (AUC) is a widely employed metric in long-tailed classification scenarios. Nevertheless, most existing methods primarily assume that training and testing examples are drawn i.i.d. from the same distribution, which is often unachievable in practice. Distributionally Robust Optimization (DRO) enhances model performance by optimizing it for the local worst-case scenario, but directly integrating AUC optimization with DRO results in an intractable optimization problem. To tackle this challenge, methodically we propose an instance-wise surrogate loss of Distributionally Robust AUC (DRAUC) and build our optimization framework on top of it. Moreover, we highlight that conventional DRAUC may induce label bias, hence introducing distribution-aware DRAUC as a more suitable metric for robust AUC learning. Theoretically, we affirm that the generalization gap between the training loss and testing error diminishes if the training set is sufficiently large. Empirically, experiments on corrupted benchmark datasets demonstrate the effectiveness of our proposed method. Code is available at: https://github.com/EldercatSAM/DRAUC.

----

## [0] iSCAN: Identifying Causal Mechanism Shifts among Nonlinear Additive Noise Models

**Authors**: *Tianyu Chen, Kevin Bello, Bryon Aragam, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c1d92835eb4e601f396c97ec60439fe-Abstract-Conference.html)

**Abstract**:

Structural causal models (SCMs) are widely used in various disciplines to represent causal relationships among variables in complex systems.Unfortunately, the underlying causal structure is often unknown, and estimating it from data remains a challenging task. In many situations, however, the end goal is to localize the changes (shifts) in the causal mechanisms between related datasets instead of learning the full causal structure of the individual datasets. Some applications include root cause analysis, analyzing gene regulatory network structure changes between healthy and cancerous individuals, or explaining distribution shifts. This paper focuses on identifying the causal mechanism shifts in two or more related datasets over the same set of variables---without estimating the entire DAG structure of each SCM.Prior work under this setting assumed linear models with Gaussian noises; instead, in this work we assume that each SCM belongs to the more general class of nonlinear additive noise models (ANMs).A key technical contribution of this work is to show that the Jacobian of the score function for the mixture distribution allows for the identification of shifts under general non-parametric functional mechanisms.Once the shifted variables are identified, we leverage recent work to estimate the structural differences, if any, for the shifted variables.Experiments on synthetic and real-world data are provided to showcase the applicability of this approach.Code implementing the proposed method is open-source and publicly available at  https://github.com/kevinsbello/iSCAN.

----

## [0] Optimal Learners for Realizable Regression: PAC Learning and Online Learning

**Authors**: *Idan Attias, Steve Hanneke, Alkis Kalavasis, Amin Karbasi, Grigoris Velegkas*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c22e5e918198702765ecff4b20d0a90-Abstract-Conference.html)

**Abstract**:

In this work, we aim to characterize the statistical complexity of realizable regression both in the PAC learning setting and the online learning setting. Previous work had established the sufficiency of finiteness of the fat shattering dimension for PAC learnability and the necessity of finiteness of the scaled Natarajan dimension, but little progress had been made towards a more complete characterization since  the work of Simon 1997 (SICOMP '97). To this end,  we first introduce a minimax instance optimal learner for realizable regression and propose a novel dimension that both qualitatively and quantitatively characterizes which classes of real-valued predictors are learnable.  We then identify a combinatorial dimension related to the graph dimension that characterizes ERM learnability in the realizable setting. Finally, we establish a necessary condition for learnability based on a combinatorial dimension related to the DS dimension, and conjecture that it may also be sufficient in this context. Additionally, in the context of online learning we provide a dimension that characterizes the minimax instance optimal cumulative loss up to a constant factor and design an optimal online learner for realizable regression, thus resolving an open question raised by Daskalakis and Golowich in STOC '22.

----

## [0] Sounding Bodies: Modeling 3D Spatial Sound of Humans Using Body Pose and Audio

**Authors**: *Xudong Xu, Dejan Markovic, Jacob Sandakly, Todd Keebler, Steven Krenn, Alexander Richard*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c234d9c7e738a793947e0282c36eb95-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c234d9c7e738a793947e0282c36eb95-Abstract-Conference.html)

**Abstract**:

While 3D human body modeling has received much attention in computer vision, modeling the acoustic equivalent, i.e. modeling 3D spatial audio produced by body motion and speech, has fallen short in the community. To close this gap, we present a model that can generate accurate 3D spatial audio for full human bodies. The system consumes, as input, audio signals from headset microphones and body pose, and produces, as output, a 3D sound field surrounding the transmitter's body, from which spatial audio can be rendered at any arbitrary position in the 3D space. We collect a first-of-its-kind multimodal dataset of human bodies, recorded with multiple cameras and a spherical array of 345 microphones. In an empirical evaluation, we demonstrate that our model can produce accurate body-induced sound fields when trained with a suitable loss. Dataset and code are available online.

----

## [0] Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering

**Authors**: *Noah Hollmann, Samuel Müller, Frank Hutter*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c2df4c35cdbee764ebb9e9d0acd5197-Abstract-Conference.html)

**Abstract**:

As the field of automated machine learning (AutoML) advances, it becomes increasingly important to incorporate domain knowledge into these systems.We present an approach for doing so by harnessing the power of large language models (LLMs). Specifically, we introduce Context-Aware Automated Feature Engineering (CAAFE), a feature engineering method for tabular datasets that utilizes an LLM to iteratively generate additional semantically meaningful features for tabular datasets based on the description of the dataset. The method produces both Python code for creating new features and explanations for the utility of the generated features.Despite being methodologically simple, CAAFE improves performance on 11 out of 14 datasets -- boosting mean ROC AUC performance from 0.798 to 0.822 across all dataset - similar to the improvement achieved by using a random forest instead of logistic regression on our datasets. Furthermore, CAAFE is interpretable by providing a textual explanation for each generated feature.CAAFE paves the way for more extensive semi-automation in data science tasks and emphasizes the significance of context-aware solutions that can extend the scope of AutoML systems to semantic AutoML. We release our code, a simple demo and a python package.

----

## [0] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning

**Authors**: *Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Lifelong learning offers a promising paradigm of building a generalist agent that learns and adapts over its lifespan. Unlike traditional lifelong learning problems in image and text domains, which primarily involve the transfer of declarative knowledge of entities and concepts, lifelong learning in decision-making (LLDM) also necessitates the transfer of procedural knowledge, such as actions and behaviors. To advance research in LLDM, we introduce LIBERO, a novel benchmark of lifelong learning for robot manipulation. Specifically, LIBERO highlights five key research topics in LLDM: 1) how to efficiently transfer declarative knowledge, procedural knowledge, or the mixture of both; 2) how to design effective policy architectures and 3) effective algorithms for LLDM; 4) the robustness of a lifelong learner with respect to task ordering; and 5) the effect of model pretraining for LLDM. We develop an extendible procedural generation pipeline that can in principle generate infinitely many tasks. For benchmarking purpose, we create four task suites (130 tasks in total) that we use to investigate the above-mentioned research topics. To support sample-efficient learning, we provide high-quality human-teleoperated demonstration data for all tasks. Our extensive experiments present several insightful or even unexpected discoveries: sequential finetuning outperforms existing lifelong learning methods in forward transfer, no single visual encoder architecture excels at all types of knowledge transfer, and naive supervised pretraining can hinder agents' performance in the subsequent LLDM.

----

## [0] On quantum backpropagation, information reuse, and cheating measurement collapse

**Authors**: *Amira Abbas, Robbie King, Hsin-Yuan Huang, William J. Huggins, Ramis Movassagh, Dar Gilboa, Jarrod R. McClean*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3caae2f725c8e2a55ecd600563d172-Abstract-Conference.html)

**Abstract**:

The success of modern deep learning hinges on the ability to train neural networks at scale. Through clever reuse of intermediate information, backpropagation facilitates training through gradient computation at a total cost roughly proportional to running the function, rather than incurring an additional factor proportional to the number of parameters -- which can now be in the trillions.  Naively, one expects that quantum measurement collapse entirely rules out the reuse of quantum information as in backpropagation. But recent developments in shadow tomography, which assumes access to multiple copies of a quantum state, have challenged that notion.  Here, we investigate whether parameterized quantum models can train as efficiently as classical neural networks. We show that achieving backpropagation scaling is impossible without access to multiple copies of a state.  With this added ability, we introduce an algorithm with foundations in shadow tomography that matches backpropagation scaling in quantum resources while reducing classical auxiliary computational costs to open problems in shadow tomography. These results highlight the nuance of reusing quantum information for practical purposes and clarify the unique difficulties in training large quantum models, which could alter the course of quantum machine learning.

----

## [0] First Order Methods with Markovian Noise: from Acceleration to Variational Inequalities

**Authors**: *Aleksandr Beznosikov, Sergey Samsonov, Marina Sheshukova, Alexander V. Gasnikov, Alexey Naumov, Eric Moulines*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c3e38ce55a0fa44bc325bc6fdb7f4e5-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c3e38ce55a0fa44bc325bc6fdb7f4e5-Abstract-Conference.html)

**Abstract**:

This paper delves into stochastic optimization problems that involve Markovian noise. We present a unified approach for the theoretical analysis of first-order gradient methods for stochastic optimization and variational inequalities. Our approach covers scenarios for both non-convex and strongly convex minimization problems. To achieve an optimal (linear) dependence on the mixing time of the underlying noise sequence, we use the randomized batching scheme, which is based on the multilevel Monte Carlo method. Moreover, our technique allows us to eliminate the limiting assumptions of previous research on Markov noise, such as the need for a bounded domain and uniformly bounded stochastic gradients. Our extension to variational inequalities under Markovian noise is original. Additionally, we provide lower bounds that match the oracle complexity of our method in the case of strongly convex optimization problems.

----

## [0] Fair, Polylog-Approximate Low-Cost Hierarchical Clustering

**Authors**: *Marina Knittel, Max Springer, John P. Dickerson, MohammadTaghi Hajiaghayi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c533f53f76c4df9a7f08e7cb676d132-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c533f53f76c4df9a7f08e7cb676d132-Abstract-Conference.html)

**Abstract**:

Research in fair machine learning, and particularly clustering, has been crucial in recent years given the many ethical controversies that modern intelligent systems have posed. Ahmadian et al. [2020] established the study of fairness in hierarchical clustering, a stronger, more structured variant of its well-known flat counterpart, though their proposed algorithm that optimizes for Dasgupta's [2016] famous cost function was highly theoretical. Knittel et al. [2023] then proposed the first practical fair approximation for cost, however they were unable to break the polynomial-approximate barrier they posed as a hurdle of interest. We break this barrier, proposing the first truly polylogarithmic-approximate low-cost fair hierarchical clustering, thus greatly bridging the gap between the best fair and vanilla hierarchical clustering approximations.

----

## [0] A Novel Approach for Effective Multi-View Clustering with Information-Theoretic Perspective

**Authors**: *Chenhang Cui, Yazhou Ren, Jingyu Pu, Jiawei Li, Xiaorong Pu, Tianyi Wu, Yutao Shi, Lifang He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c64bc3f7796d31caa7c3e6b969bf7da-Abstract-Conference.html)

**Abstract**:

Multi-view clustering (MVC) is a popular technique for improving clustering performance using various data sources. However, existing methods primarily focus on acquiring consistent information while often neglecting the issue of redundancy across multiple views.This study presents a new approach called Sufficient Multi-View Clustering (SUMVC) that examines the multi-view clustering framework from an information-theoretic standpoint. Our proposed method consists of two parts. Firstly, we develop a simple and reliable multi-view clustering method SCMVC (simple consistent multi-view clustering) that employs variational analysis to generate consistent information. Secondly, we propose a sufficient representation lower bound to enhance consistent information and minimise unnecessary information among views. The proposed SUMVC method offers a promising solution to the problem of multi-view clustering and provides a new perspective for analyzing multi-view data.  To verify the effectiveness of our model, we conducted a theoretical analysis based on the Bayes Error Rate, and experiments on multiple multi-view datasets demonstrate the superior performance of SUMVC.

----

## [0] OpenShape: Scaling Up 3D Shape Representation Towards Open-World Understanding

**Authors**: *Minghua Liu, Ruoxi Shi, Kaiming Kuang, Yinhao Zhu, Xuanlin Li, Shizhong Han, Hong Cai, Fatih Porikli, Hao Su*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c7304e77c832ddc70075dfee081ca6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c7304e77c832ddc70075dfee081ca6c-Abstract-Conference.html)

**Abstract**:

We introduce OpenShape, a method for learning multi-modal joint representations of text, image, and point clouds. We adopt the commonly used multi-modal contrastive learning framework for representation alignment, but with a specific focus on scaling up 3D representations to enable open-world 3D shape understanding. To achieve this, we scale up training data by ensembling multiple 3D datasets and propose several strategies to automatically filter and enrich noisy text descriptions. We also explore and compare strategies for scaling 3D backbone networks and introduce a novel hard negative mining module for more efficient training. We evaluate OpenShape on zero-shot 3D classification benchmarks and demonstrate its superior capabilities for open-world recognition. Specifically, OpenShape achieves a zero-shot accuracy of 46.8% on the 1,156-category Objaverse-LVIS benchmark, compared to less than 10% for existing methods. OpenShape also achieves an accuracy of 85.3% on ModelNet40, outperforming previous zero-shot baseline methods by 20% and performing on par with some fully-supervised methods. Furthermore, we show that our learned embeddings encode a wide range of visual and semantic concepts (e.g., subcategories, color, shape, style) and facilitate fine-grained text-3D and image-3D interactions. Due to their alignment with CLIP embeddings, our learned shape representations can also be integrated with off-the-shelf CLIP-based models for various applications, such as point cloud captioning and point cloud-conditioned image generation.

----

## [0] KuaiSim: A Comprehensive Simulator for Recommender Systems

**Authors**: *Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c7f8f98f9a8f5650922dd4545254f28-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c7f8f98f9a8f5650922dd4545254f28-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs.To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses.The resulting simulator can support three levels of recommendation problems: the request level list-wise recommendation task, the whole-session level sequential recommendation task, and the cross-session level retention optimization task. For each task, KuaiSim also provides evaluation protocols and baseline recommendation algorithms that further serve as benchmarks for future research. We also restructure existing competitive simulators on the Kuairand Dataset and compare them against KuaiSim to future assess their performance and behavioral differences. Furthermore, to showcase KuaiSim's flexibility in accommodating different datasets, we demonstrate its versatility and robustness when deploying it on the ML-1m dataset. The implementation code is available online to ease reproducibility \footnote{https://github.com/Applied-Machine-Learning-Lab/KuaiSim}.

----

## [0] Optimizing over trained GNNs via symmetry breaking

**Authors**: *Shiqiang Zhang, Juan S. Campos, Christian Feldmann, David Walz, Frederik Sandfort, Miriam Mathea, Calvin Tsay, Ruth Misener*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c8cd1b78cdae751265c88efc136e5bd-Abstract-Conference.html)

**Abstract**:

Optimization over trained machine learning models has applications including: verification, minimizing neural acquisition functions, and integrating a trained surrogate into a larger decision-making problem. This paper formulates and solves optimization problems constrained by trained graph neural networks (GNNs). To circumvent the symmetry issue caused by graph isomorphism, we propose two types of symmetry-breaking constraints: one indexing a node 0 and one indexing the remaining nodes by lexicographically ordering their neighbor sets. To guarantee that adding these constraints will not remove all symmetric solutions, we construct a graph indexing algorithm and prove that the resulting graph indexing satisfies the proposed symmetry-breaking constraints. For the classical GNN architectures considered in this paper, optimizing over a GNN with a fixed graph is equivalent to optimizing over a dense neural network. Thus, we study the case where the input graph is not fixed, implying that each edge is a decision variable, and develop two mixed-integer optimization formulations. To test our symmetry-breaking strategies and optimization formulations, we consider an application in molecular design.

----

## [0] REx: Data-Free Residual Quantization Error Expansion

**Authors**: *Edouard Yvinec, Arnaud Dapogny, Matthieu Cord, Kevin Bailly*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8c96b559340daa7bb29f56ccfbbc9c2f-Abstract-Conference.html)

**Abstract**:

Deep neural networks (DNNs) are ubiquitous in computer vision and natural language processing, but suffer from high inference cost. This problem can be addressed by quantization, which consists in converting floating point operations into a lower bit-width format. With the growing concerns on privacy rights, we focus our efforts on data-free methods. However, such techniques suffer from their lack of adaptability to the target devices, as a hardware typically only supports specific bit widths. Thus, to adapt to a variety of devices, a quantization method shall be flexible enough to find good accuracy v.s. speed trade-offs for every bit width and target device. To achieve this, we propose REx, a quantization method that leverages residual error expansion, along with group sparsity. We show experimentally that REx enables better trade-offs (in terms of accuracy given any target bit-width) on both convnets and transformers for computer vision, as well as NLP models. In particular, when applied to large language models, we show that REx elegantly solves the outlier problem that hinders state-of-the-art quantization methods.In addition, REx is backed off by strong theoretical guarantees on the preservation of the predictive function of the original model. Lastly, we show that REx is agnostic to the quantization operator and can be used in combination with previous quantization work.

----

## [0] A Unified, Scalable Framework for Neural Population Decoding

**Authors**: *Mehdi Azabou, Vinam Arora, Venkataramana Ganesh, Ximeng Mao, Santosh Nachimuthu, Michael Mendelson, Blake A. Richards, Matthew G. Perich, Guillaume Lajoie, Eva L. Dyer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ca113d122584f12a6727341aaf58887-Abstract-Conference.html)

**Abstract**:

Our ability to use deep learning approaches to decipher neural activity would likely benefit from greater scale, in terms of both the model size and the datasets. However, the integration of many neural recordings into one unified model is challenging, as each recording contains the activity of different neurons from different individual animals. In this paper, we introduce a training framework and architecture designed to model the population dynamics of neural activity across diverse, large-scale neural recordings. Our method first tokenizes individual spikes within the dataset to build an efficient representation of neural events that captures the fine temporal structure of neural activity. We then employ cross-attention and a PerceiverIO backbone to further construct a latent tokenization of neural population activities. Utilizing this architecture and training framework, we construct a large-scale multi-session model trained on large datasets from seven nonhuman primates, spanning over 158 different sessions of recording from over 27,373 neural units and over 100 hours of recordings. In a number of different tasks, we demonstrate that our pretrained model can be rapidly adapted to new, unseen sessions with unspecified neuron correspondence, enabling few-shot performance with minimal labels. This work presents a powerful new approach for building deep learning tools to analyze neural data and stakes out a clear path to training at scale for neural decoding models.

----

## [0] Species196: A One-Million Semi-supervised Dataset for Fine-grained Species Recognition

**Authors**: *Wei He, Kai Han, Ying Nie, Chengcheng Wang, Yunhe Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8cb92f326d01fd7f4371283ee2fa6386-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8cb92f326d01fd7f4371283ee2fa6386-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The development of foundation vision models has pushed the general visual recognition to a high level, but cannot well address the fine-grained recognition in specialized domain such as invasive species classification. Identifying and managing invasive species has strong social and ecological value. Currently, most invasive species datasets are limited in scale and cover a narrow range of species, which restricts the development of deep-learning based invasion biometrics systems. To fill the gap of this area, we introduced Species196, a large-scale semi-supervised dataset of 196-category invasive species. It collects over 19K images with expert-level accurate annotations (Species196-L), and 1.2M unlabeled images of invasive species (Species196-U). The dataset provides four experimental settings for benchmarking the existing models and algorithms, namely, supervised learning, semi-supervised learning and self-supervised pretraining. To facilitate future research on these four learning paradigms, we conduct an empirical study of the representative methods on the introduced dataset. The dataset will be made publicly available at https://species-dataset.github.io/.

----

## [0] PTADisc: A Cross-Course Dataset Supporting Personalized Learning in Cold-Start Scenarios

**Authors**: *Liya Hu, Zhiang Dong, Jingyuan Chen, Guifeng Wang, Zhihua Wang, Zhou Zhao, Fei Wu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8cf04c64d1734e5f7e63418a2a4d49de-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

The focus of our work is on diagnostic tasks in personalized learning, such as cognitive diagnosis and knowledge tracing. The goal of these tasks is to assess students' latent proficiency on knowledge concepts through analyzing their historical learning records. However, existing research has been limited to single-course scenarios; cross-course studies have not been explored due to a lack of dataset. We address this issue by constructing PTADisc, a Diverse, Immense, Student-centered dataset that emphasizes its sufficient Cross-course information for personalized learning. PTADisc includes 74 courses, 1,530,100 students, 4,054 concepts, 225,615 problems, and over 680 million student response logs. Based on PTADisc, we developed a model-agnostic Cross-Course Learner Modeling Framework (CCLMF) which utilizes relationships between students' proficiency across courses to alleviate the difficulty of diagnosing student knowledge state in cold-start scenarios. CCLMF uses a meta network to generate personalized mapping functions between courses. The experimental results on PTADisc verify the effectiveness of CCLMF with an average improvement of 4.2% on AUC. We also report the performance of baseline models for cognitive diagnosis and knowledge tracing over PTADisc, demonstrating that our dataset supports a wide scope of research in personalized learning. Additionally, PTADisc contains valuable programming logs and student-group information that are worth exploring in the future.

----

## [0] Adaptive Contextual Perception: How To Generalize To New Backgrounds and Ambiguous Objects

**Authors**: *Zhuofan Ying, Peter Hase, Mohit Bansal*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d2c36836fb0e7d78fe68762ff8b5f1e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d2c36836fb0e7d78fe68762ff8b5f1e-Abstract-Conference.html)

**Abstract**:

Biological vision systems make adaptive use of context to recognize objects in new settings with novel contexts as well as occluded or blurry objects in familiar settings. In this paper, we investigate how vision models adaptively use context for out-of-distribution (OOD) generalization and leverage our analysis results to improve model OOD generalization. First, we formulate two distinct OOD settings where the contexts are either beneficial Object-Disambiguation or irrelevant Background-Invariance, reflecting the diverse contextual challenges faced in biological vision. We then analyze model performance in these two different OOD settings and demonstrate that models that excel in one setting tend to struggle in the other. Notably, prior works on learning causal features improve on one setting but hurt on the other. This underscores the importance of generalizing across both OOD settings, as this ability is crucial for both human cognition and robust AI systems. Next, to better understand the model properties contributing to OOD generalization, we use representational geometry analysis and our own probing methods to examine a population of models, and we discover that those with more factorized representations and appropriate feature weighting are more successful in handling Object-Disambiguation and Background-Invariance tests. We further validate these findings through causal intervention, manipulating representation factorization and feature weighting to demonstrate their causal effect on performance. Motivated by our analysis results, we propose new augmentation methods aimed at enhancing model generalization. The proposed methods outperform strong baselines, yielding improvements in both in-distribution and OOD tests. We conclude that, in order to replicate the generalization abilities of biological vision, computer vision models must have factorized object vs. background representations and appropriately weigh both kinds of features.

----

## [0] PUG: Photorealistic and Semantically Controllable Synthetic Data for Representation Learning

**Authors**: *Florian Bordes, Shashank Shekhar, Mark Ibrahim, Diane Bouchacourt, Pascal Vincent, Ari Morcos*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d352fd0f07fde4a74f9476603b3773b-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d352fd0f07fde4a74f9476603b3773b-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

Synthetic image datasets offer unmatched advantages for designing and evaluating deep neural networks: they make it possible to (i) render as many data samples as needed, (ii) precisely control each scene and yield granular ground truth labels (and captions), (iii) precisely control distribution shifts between training and testing to isolate variables of interest for sound experimentation.Despite such promise, the use of synthetic image data is still limited -- and often played down -- mainly due to their lack of realism. Most works therefore rely on datasets of real images, which have often been scraped from public images on the internet, and may have issues with regards to privacy, bias, and copyright, while offering little control over how objects precisely appear.In this work, we present a path to democratize the use of photorealistic synthetic data: we develop a new generation of interactive environments for representation learning research, that offer both controllability and realism. We use the Unreal Engine, a powerful game engine well known in the entertainment industry, to produce PUG (Photorealistic Unreal Graphics) environments and datasets for representation learning. Using PUG for evaluation and fine-tuning, we demonstrate the potential of PUG to both enable more rigorous evaluations and to improve model training.

----

## [0] On the Gini-impurity Preservation For Privacy Random Forests

**Authors**: *Xinran Xie, Man-Jie Yuan, Xuetong Bai, Wei Gao, Zhi-Hua Zhou*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d6b1d775014eff18256abeb207202ad-Abstract-Conference.html)

**Abstract**:

Random forests have been one successful ensemble algorithms in machine learning. Various techniques have been utilized to preserve the privacy of random forests from anonymization, differential privacy, homomorphic encryption, etc., whereas it rarely takes into account some crucial ingredients of learning algorithm. This work presents a new encryption to preserve data's Gini impurity, which plays a crucial role during the construction of random forests. Our basic idea is to modify the structure of binary search tree to store several examples in each node,  and encrypt data features by incorporating label and order information. Theoretically, we prove that our scheme preserves the minimum Gini impurity in ciphertexts without decrypting, and present the security guarantee for encryption. For random forests, we encrypt data features based on our Gini-impurity-preserving scheme, and take the homomorphic encryption scheme CKKS to encrypt data labels  due to their importance and privacy. We  conduct extensive experiments to show the effectiveness, efficiency and security of our proposed method.

----

## [0] Debiasing Pretrained Generative Models by Uniformly Sampling Semantic Attributes

**Authors**: *Walter Gerych, Kevin Hickey, Luke Buquicchio, Kavin Chandrasekaran, Abdulaziz Alajaji, Elke A. Rundensteiner, Emmanuel Agu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8d7060b2ee6ff728692398783e3d59d1-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8d7060b2ee6ff728692398783e3d59d1-Abstract-Conference.html)

**Abstract**:

Generative models are being increasingly used in science and industry applications. Unfortunately,  they often perpetuate the biases present in their training sets, such as societal biases causing certain groups to be underrepresented in the data. For instance, image generators may overwhelmingly produce images of white people due to few non-white samples in their training data. It is imperative to debias generative models so they synthesize an equal number of instances for each group, while not requiring retraining of the model to avoid  prohibitive expense. We thus propose a distribution mapping module that produces samples from a fair noise distribution, such that the  pretrained generative model produces semantically uniform outputs -  an equal number of instances for each group - when conditioned on these samples. This does not involve retraining the generator, nor does it require any real training data. Experiments on debiasing generators trained on popular real-world  datasets show that our method outperforms existing approaches.

----

## [0] Improved Algorithms for Stochastic Linear Bandits Using Tail Bounds for Martingale Mixtures

**Authors**: *Hamish Flynn, David Reeb, Melih Kandemir, Jan R. Peters*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8db0d67d22e0ec08c95b810be3a66907-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8db0d67d22e0ec08c95b810be3a66907-Abstract-Conference.html)

**Abstract**:

We present improved algorithms with worst-case regret guarantees for the stochastic linear bandit problem. The widely used "optimism in the face of uncertainty" principle reduces a stochastic bandit problem to the construction of a confidence sequence for the unknown reward function. The performance of the resulting bandit algorithm depends on the size of the confidence sequence, with smaller confidence sets yielding better empirical performance and stronger regret guarantees. In this work, we use a novel tail bound for adaptive martingale mixtures to construct confidence sequences which are suitable for stochastic bandits. These confidence sequences allow for efficient action selection via convex programming. We prove that a linear bandit algorithm based on our confidence sequences is guaranteed to achieve competitive worst-case regret. We show that our confidence sequences are tighter than competitors, both empirically and theoretically. Finally, we demonstrate that our tighter confidence sequences give improved performance in several hyperparameter tuning tasks.

----

## [0] Tame a Wild Camera: In-the-Wild Monocular Camera Calibration

**Authors**: *Shengjie Zhu, Abhinav Kumar, Masa Hu, Xiaoming Liu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8db9279f593652ee9bb2223b4a2c43fa-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8db9279f593652ee9bb2223b4a2c43fa-Abstract-Conference.html)

**Abstract**:

3D sensing for monocular in-the-wild images, e.g., depth estimation and 3D object detection, has become increasingly important.However, the unknown intrinsic parameter hinders their development and deployment.Previous methods for the monocular camera calibration rely on specific 3D objects or strong geometry prior, such as using a checkerboard or imposing a Manhattan World assumption.This work instead calibrates intrinsic via exploiting the monocular 3D prior.Given an undistorted image as input, our method calibrates the complete 4 Degree-of-Freedom (DoF) intrinsic parameters.First, we show intrinsic is determined by the two well-studied monocular priors: monocular depthmap and surface normal map.However, this solution necessitates a low-bias and low-variance depth estimation.Alternatively, we introduce the incidence field, defined as the incidence rays between points in 3D space and pixels in the 2D imaging plane.We show that: 1) The incidence field is a pixel-wise parametrization of the intrinsic invariant to image cropping and resizing.2) The incidence field is a learnable monocular 3D prior, determined pixel-wisely by up-to-sacle monocular depthmap and surface normal.With the estimated incidence field, a robust RANSAC algorithm recovers intrinsic.We show the effectiveness of our method through superior performance on synthetic and zero-shot testing datasets.Beyond calibration, we demonstrate downstream applications in image manipulation detection \& restoration, uncalibrated two-view pose estimation, and 3D sensing.

----

## [0] ATTA: Anomaly-aware Test-Time Adaptation for Out-of-Distribution Detection in Segmentation

**Authors**: *Zhitong Gao, Shipeng Yan, Xuming He*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8dcc306a2522c60a78f047ab8739e631-Abstract-Conference.html)

**Abstract**:

Recent advancements in dense out-of-distribution (OOD) detection have primarily focused on scenarios where the training and testing datasets share a similar domain, with the assumption that no domain shift exists between them. However, in real-world situations, domain shift often exits and significantly affects the accuracy of existing out-of-distribution (OOD) detection models. In this work, we propose a dual-level OOD detection framework to handle domain shift and semantic shift jointly. The first level distinguishes whether domain shift exists in the image by leveraging global low-level features, while the second level identifies pixels with semantic shift by utilizing dense high-level feature maps. In this way, we can selectively adapt the model to unseen domains as well as enhance model's capacity in detecting novel classes. We validate the efficacy of our proposed method on several OOD segmentation benchmarks, including those with significant domain shifts and those without, observing consistent performance improvements across various baseline models. Code is available at https://github.com/gaozhitong/ATTA.

----

## [0] Regression with Cost-based Rejection

**Authors**: *Xin Cheng, Yuzhou Cao, Haobo Wang, Hongxin Wei, Bo An, Lei Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ddcba644f602835a52b962d9a119eea-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ddcba644f602835a52b962d9a119eea-Abstract-Conference.html)

**Abstract**:

Learning with rejection is an important framework that can refrain from making predictions to avoid critical mispredictions by balancing between prediction and rejection. Previous studies on cost-based rejection only focused on the classification setting, which cannot handle the continuous and infinite target space in the regression setting. In this paper, we investigate a novel regression problem called regression with cost-based rejection, where the model can reject to make predictions on some examples given certain rejection costs. To solve this problem, we first formulate the expected risk for this problem and then derive the Bayes optimal solution, which shows that the optimal model should reject to make predictions on the examples whose variance is larger than the rejection cost when the mean squared error is used as the evaluation metric. Furthermore, we propose to train the model by a surrogate loss function that considers rejection as binary classification and we provide conditions for the model consistency, which implies that the Bayes optimal solution can be recovered by our proposed surrogate loss. Extensive experiments demonstrate the effectiveness of our proposed method.

----

## [0] A State Representation for Diminishing Rewards

**Authors**: *Ted Moskovitz, Samo Hromadka, Ahmed Touati, Diana Borsa, Maneesh Sahani*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df0fe2bba0f14208a10c1cb22e71552-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df0fe2bba0f14208a10c1cb22e71552-Abstract-Conference.html)

**Abstract**:

A common setting in multitask reinforcement learning (RL) demands that an agent rapidly adapt to various stationary reward functions randomly sampled from a fixed distribution. In such situations, the successor representation (SR) is a popular framework which supports rapid policy evaluation by decoupling a policy's expected discounted, cumulative state occupancies from a specific reward function. However, in the natural world, sequential tasks are rarely independent, and instead reflect shifting priorities based on the availability and subjective perception of rewarding stimuli. Reflecting this disjunction, in this paper we study the phenomenon of diminishing marginal utility and introduce a novel state representation, the $\lambda$ representation ($\lambda$R) which, surprisingly, is required for policy evaluation in this setting and which generalizes the SR as well as several other state representations from the literature. We establish the $\lambda$R's formal properties and examine its normative advantages in the context of machine learning, as well as its usefulness for studying natural behaviors, particularly foraging.

----

## [0] Unified Segment-to-Segment Framework for Simultaneous Sequence Generation

**Authors**: *Shaolei Zhang, Yang Feng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df705957a5262de3cb37ba9f1fb96f3-Abstract-Conference.html)

**Abstract**:

Simultaneous sequence generation is a pivotal task for real-time scenarios, such as streaming speech recognition, simultaneous machine translation and simultaneous speech translation, where the target sequence is generated while receiving the source sequence. The crux of achieving high-quality generation with low latency lies in identifying the optimal moments for generating, accomplished by learning a mapping between the source and target sequences. However, existing methods often rely on task-specific heuristics for different sequence types, limiting the modelâ€™s capacity to adaptively learn the source-target mapping and hindering the exploration of multi-task learning for various simultaneous tasks. In this paper, we propose a unified segment-to-segment framework (Seg2Seg) for simultaneous sequence generation, which learns the mapping in an adaptive and unified manner. During the process of simultaneous generation, the model alternates between waiting for a source segment and generating a target segment, making the segment serve as the natural bridge between the source and target. To accomplish this, Seg2Seg introduces a latent segment as the pivot between source to target and explores all potential source-target mappings via the proposed expectation training, thereby learning the optimal moments for generating. Experiments on multiple simultaneous generation tasks demonstrate that Seg2Seg achieves state-of-the-art performance and exhibits better generality across various tasks.

----

## [0] DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting

**Authors**: *Salva Rühling Cachay, Bo Zhao, Hailey Joren, Rose Yu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8df90a1440ce782d1f5607b7a38f2531-Abstract-Conference.html)

**Abstract**:

While diffusion models can successfully generate data and make predictions, they are predominantly designed for static images. We propose an approach for training diffusion models for dynamics forecasting that leverages the temporal dynamics encoded in the data, directly coupling it with the diffusion steps in the network. We train a stochastic, time-conditioned interpolator and a backbone forecaster networkthat mimic the forward and reverse processes of conventional diffusion models, respectively. This design choice naturally encodes multi-step and long-range forecasting capabilities, allowing for highly flexible, continuous-time sampling trajectories and the ability to trade-off performance with accelerated sampling at inference time. In addition, the dynamics-informed diffusion process imposes a strong inductive bias, allowing for improved computational efficiency compared to traditional Gaussian noise-based diffusion models. Our approach performs competitively on probabilistic skill score metrics in complex dynamics forecasting of sea surface temperatures, Navier-Stokes flows, and spring mesh systems.

----

## [0] Towards a Comprehensive Benchmark for High-Level Synthesis Targeted to FPGAs

**Authors**: *Yunsheng Bai, Atefeh Sohrabizadeh, Zongyue Qin, Ziniu Hu, Yizhou Sun, Jason Cong*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8dfc3a2720a4112243a285b98e0d4415-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

High-level synthesis (HLS) aims to raise the abstraction layer in hardware design, enabling the design of domain-specific accelerators (DSAs) like field-programmable gate arrays (FPGAs) using C/C++ instead of hardware description languages (HDLs). Compiler directives in the form of pragmas play a crucial role in modifying the microarchitecture within the HLS framework. However, the space of possible microarchitectures grows exponentially with the number of pragmas. Moreover, the evaluation of each candidate design using the HLS tool consumes significant time, ranging from minutes to hours, leading to a time-consuming optimization process. To accelerate this process, machine learning models have been used to predict design quality in milliseconds. However, existing open-source datasets for training such models are limited in terms of design complexity and available optimizations. In this paper, we present HLSyn, the first benchmark that addresses these limitations. It contains more complex programs with a wider range of optimization pragmas, making it a comprehensive dataset for training and evaluating design quality prediction models. The HLSyn benchmark consists of 42 unique programs/kernels, resulting in over 42,000 labeled designs. We conduct an extensive comparison of state-of-the-art baselines to assess their effectiveness in predicting design quality. As an ongoing project, we anticipate expanding the HLSyn benchmark in terms of both quantity and variety of programs to further support the development of this field.

----

## [0] Energy Discrepancies: A Score-Independent Loss for Energy-Based Models

**Authors**: *Tobias Schröder, Zijing Ou, Jen Lim, Yingzhen Li, Sebastian J. Vollmer, Andrew Duncan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e176ef071f00f1b233461c5ad5e1b24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e176ef071f00f1b233461c5ad5e1b24-Abstract-Conference.html)

**Abstract**:

Energy-based models are a simple yet powerful class of probabilistic models, but their widespread adoption has been limited by the computational burden of training them. We propose a novel loss function called Energy Discrepancy (ED) which does not rely on the computation of scores or expensive Markov chain Monte Carlo. We show that energy discrepancy approaches the explicit score matching and negative log-likelihood loss under different limits, effectively interpolating between both. Consequently, minimum energy discrepancy estimation overcomes the problem of nearsightedness encountered in score-based estimation methods, while also enjoying theoretical guarantees. Through numerical experiments, we demonstrate that ED learns low-dimensional data distributions faster and more accurately than explicit score matching or contrastive divergence. For high-dimensional image data, we describe how the manifold hypothesis puts limitations on our approach and demonstrate the effectiveness of energy discrepancy by training the energy-based model as a prior of a variational decoder model.

----

## [0] Learning to Group Auxiliary Datasets for Molecule

**Authors**: *Tinglin Huang, Ziniu Hu, Rex Ying*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e2571d13f432b301d4c5e3cc70227a6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e2571d13f432b301d4c5e3cc70227a6-Abstract-Conference.html)

**Abstract**:

The limited availability of annotations in small molecule datasets presents a challenge to machine learning models. To address this, one common strategy is to collaborate with additional auxiliary datasets. However, having more data does not always guarantee improvements. Negative transfer can occur when the knowledge in the target dataset differs or contradicts that of the auxiliary molecule datasets. In light of this, identifying the auxiliary molecule datasets that can benefit the target dataset when jointly trained remains a critical and unresolved problem. Through an empirical analysis, we observe that combining graph structure similarity and task similarity can serve as a more reliable indicator for identifying high-affinity auxiliary datasets. Motivated by this insight, we propose MolGroup, which separates the dataset affinity into task and structure affinity to predict the potential benefits of each auxiliary molecule dataset. MolGroup achieves this by utilizing a routing mechanism optimized through a bi-level optimization framework. Empowered by the meta gradient, the routing mechanism is optimized toward maximizing the target dataset's performance and quantifies the affinity as the gating score. As a result, MolGroup is capable of predicting the optimal combination of auxiliary datasets for each target dataset. Our extensive experiments demonstrate the efficiency and effectiveness of MolGroup, showing an average improvement of 4.41%/3.47% for GIN/Graphormer trained with the group of molecule datasets selected by MolGroup on 11 target molecule datasets.

----

## [0] Equivariant Spatio-Temporal Attentive Graph Networks to Simulate Physical Dynamics

**Authors**: *Liming Wu, Zhichao Hou, Jirui Yuan, Yu Rong, Wenbing Huang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e2a75e0c7b579a6cf176dc0858cde55-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e2a75e0c7b579a6cf176dc0858cde55-Abstract-Conference.html)

**Abstract**:

Learning to represent and simulate the dynamics of physical systems is a crucial yet challenging task. Existing equivariant Graph Neural Network (GNN) based methods have encapsulated the symmetry of physics, \emph{e.g.}, translations, rotations, etc, leading to better generalization ability. Nevertheless, their frame-to-frame formulation of the task overlooks the non-Markov property mainly incurred by unobserved dynamics in the environment. In this paper, we reformulate dynamics simulation as a spatio-temporal prediction task, by employing the trajectory in the past period to recover the Non-Markovian interactions. We propose Equivariant Spatio-Temporal Attentive Graph Networks (ESTAG), an equivariant version of spatio-temporal GNNs, to fulfil our purpose. At its core, we design a novel Equivariant Discrete Fourier Transform (EDFT) to extract periodic patterns from the history frames, and then construct an Equivariant Spatial Module (ESM) to accomplish spatial message passing, and an Equivariant Temporal Module (ETM) with the forward attention and equivariant pooling mechanisms to aggregate temporal message. We evaluate our model on three real datasets corresponding to the molecular-, protein- and macro-level. Experimental results verify the effectiveness of ESTAG compared to typical spatio-temporal GNNs and equivariant GNNs.

----

## [0] Differentially Private Decoupled Graph Convolutions for Multigranular Topology Protection

**Authors**: *Eli Chien, Wei-Ning Chen, Chao Pan, Pan Li, Ayfer Özgür, Olgica Milenkovic*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e3db2040672d85fd12e6313945594fe-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e3db2040672d85fd12e6313945594fe-Abstract-Conference.html)

**Abstract**:

Graph Neural Networks (GNNs) have proven to be highly effective in solving real-world learning problems that involve graph-structured data. However, GNNs can also inadvertently expose sensitive user information and interactions through their model predictions. To address these privacy concerns, Differential Privacy (DP) protocols are employed to control the trade-off between provable privacy protection and model utility. Applying standard DP approaches to GNNs directly is not advisable due to two main reasons. First, the prediction of node labels, which relies on neighboring node attributes through graph convolutions, can lead to privacy leakage. Second, in practical applications, the privacy requirements for node attributes and graph topology may differ. In the latter setting, existing DP-GNN models fail to provide multigranular trade-offs between graph topology privacy, node attribute privacy, and GNN utility. To address both limitations, we propose a new framework termed Graph Differential Privacy (GDP), specifically tailored to graph learning. GDP ensures both provably private model parameters as well as private predictions. Additionally, we describe a novel unified notion of graph dataset adjacency to analyze the properties of GDP for different levels of graph topology privacy. Our findings reveal that DP-GNNs, which rely on graph convolutions, not only fail to meet the requirements for multigranular graph topology privacy but also necessitate the injection of DP noise that scales at least linearly with the maximum node degree. In contrast, our proposed Differentially Private Decoupled Graph Convolutions (DPDGCs) represent a more flexible and efficient alternative to graph convolutions that still provides the necessary guarantees of GDP. To validate our approach, we conducted extensive experiments on seven node classification benchmarking and illustrative synthetic datasets. The results demonstrate that DPDGCs significantly outperform existing DP-GNNs in terms of privacy-utility trade-offs.

----

## [0] Team-PSRO for Learning Approximate TMECor in Large Team Games via Cooperative Reinforcement Learning

**Authors**: *Stephen McAleer, Gabriele Farina, Gaoyue Zhou, Mingzhi Wang, Yaodong Yang, Tuomas Sandholm*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e4ccc9ca6ae2225c4cbb7782ab48daf-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e4ccc9ca6ae2225c4cbb7782ab48daf-Abstract-Conference.html)

**Abstract**:

Recent algorithms have achieved superhuman performance at a number of two-player zero-sum games such as poker and go. However, many real-world situations are multi-player games. Zero-sum two-team games, such as bridge and football, involve two teams where each member of the team shares the same reward with every other member of that team, and each team has the negative of the reward of the other team. A popular solution concept in this setting, called TMECor, assumes that teams can jointly correlate their strategies before play, but are not able to communicate during play. This setting is harder than two-player zero-sum games because each player on a team has different information and must use their public actions to signal to other members of the team. Prior works either have game-theoretic guarantees but only work in very small games, or are able to scale to large games but do not have game-theoretic guarantees. In this paper we introduce two algorithms: Team-PSRO, an extension of PSRO from two-player games to team games, and Team-PSRO Mix-and-Match which improves upon Team PSRO by better using population policies. In Team-PSRO, in every iteration both teams learn a joint best response to the opponent's meta-strategy via reinforcement learning. As the reinforcement learning joint best response approaches the optimal best response, Team-PSRO is guaranteed to converge to a TMECor. In experiments on Kuhn poker and Liar's Dice, we show that a tabular version of Team-PSRO converges to TMECor, and a version of Team PSRO using deep cooperative reinforcement learning beats self-play reinforcement learning in the large game of Google Research Football.

----

## [0] Learning Linear Causal Representations from Interventions under General Nonlinear Mixing

**Authors**: *Simon Buchholz, Goutham Rajendran, Elan Rosenfeld, Bryon Aragam, Bernhard Schölkopf, Pradeep Ravikumar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e5de4cb639ef718f44060dc257cb04f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e5de4cb639ef718f44060dc257cb04f-Abstract-Conference.html)

**Abstract**:

We study the problem of learning causal representations from unknown, latent interventions in a general setting, where the latent distribution is Gaussian but the mixing function is completely general. We prove strong identifiability results given unknown single-node interventions, i.e., without having access to the intervention targets. This generalizes prior works which have focused on weaker classes, such as linear maps or paired counterfactual data. This is also the first instance of identifiability from non-paired interventions for deep neural network embeddings and general causal structures. Our proof relies on carefully uncovering the high-dimensional geometric structure present in the data distribution after a non-linear density transformation, which we capture by analyzing quadratic forms of precision matrices of the latent distributions. Finally, we propose a contrastive algorithm to identify the latent variables in practice and evaluate its performance on various tasks.

----

## [0] Disentanglement via Latent Quantization

**Authors**: *Kyle Hsu, William Dorrell, James C. R. Whittington, Jiajun Wu, Chelsea Finn*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e63972d4d9d81b31459d787466ce271-Abstract-Conference.html)

**Abstract**:

In disentangled representation learning, a model is asked to tease apart a dataset's underlying sources of variation and represent them independently of one another. Since the model is provided with no ground truth information about these sources, inductive biases take a paramount role in enabling disentanglement. In this work, we construct an inductive bias towards encoding to and decoding from an organized latent space. Concretely, we do this by (i) quantizing the latent space into discrete code vectors with a separate learnable scalar codebook per dimension and (ii) applying strong model regularization via an unusually high weight decay. Intuitively, the latent space design forces the encoder to combinatorially construct codes from a small number of distinct scalar values, which in turn enables the decoder to assign a consistent meaning to each value. Regularization then serves to drive the model towards this parsimonious strategy. We demonstrate the broad applicability of this approach by adding it to both basic data-reconstructing (vanilla autoencoder) and latent-reconstructing (InfoGAN) generative models. For reliable evaluation, we also propose InfoMEC, a new set of metrics for disentanglement that is cohesively grounded in information theory and fixes well-established shortcomings in previous metrics. Together with regularization, latent quantization dramatically improves the modularity and explicitness of learned representations on a representative suite of benchmark datasets. In particular, our quantized-latent autoencoder (QLAE) consistently outperforms strong methods from prior work in these key disentanglement properties without compromising data reconstruction.

----

## [0] Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies

**Authors**: *Oscar Li, James Harrison, Jascha Sohl-Dickstein, Virginia Smith, Luke Metz*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e69a97cbdd91ac0808603fa589d6c17-Abstract-Conference.html)

**Abstract**:

Unrolled computation graphs are prevalent throughout machine learning but present challenges to automatic differentiation (AD) gradient estimation methods when their loss functions exhibit extreme local sensitivtiy, discontinuity, or blackbox characteristics. In such scenarios, online evolution strategies methods are a more capable alternative, while being more parallelizable than vanilla evolution strategies (ES) by interleaving partial unrolls and gradient updates. In this work, we propose a general class of unbiased online evolution strategies methods. We analytically and empirically characterize the variance of this class of gradient estimators and identify the one with the least variance, which we term Noise-Reuse Evolution Strategies (NRES). Experimentally, we show NRES results in faster convergence than existing AD and ES methods in terms of wall-clock time and number of unroll steps across a variety of applications, including learning dynamical systems, meta-training learned optimizers, and reinforcement learning.

----

## [0] Beyond Black-Box Advice: Learning-Augmented Algorithms for MDPs with Q-Value Predictions

**Authors**: *Tongxin Li, Yiheng Lin, Shaolei Ren, Adam Wierman*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e806d3c56ed5f1dab85d601e13cbe38-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e806d3c56ed5f1dab85d601e13cbe38-Abstract-Conference.html)

**Abstract**:

We study the tradeoff between consistency and robustness in the context of a single-trajectory time-varying Markov Decision Process (MDP) with untrusted machine-learned advice. Our work departs from the typical approach of treating advice as coming from black-box sources by instead considering a setting where additional information about  how the advice is generated is available. We prove a first-of-its-kind consistency and robustness tradeoff given Q-value advice under a general MDP model that includes both continuous and discrete state/action spaces. Our results highlight that utilizing Q-value advice enables dynamic pursuit of the better of machine-learned advice and a robust baseline, thus result in near-optimal performance guarantees, which provably improves what can be obtained solely with black-box advice.

----

## [0] Graph Contrastive Learning with Stable and Scalable Spectral Encoding

**Authors**: *Deyu Bo, Yuan Fang, Yang Liu, Chuan Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9a6582caa59fda0302349702965171-Abstract-Conference.html)

**Abstract**:

Graph contrastive learning (GCL) aims to learn representations by capturing the agreements between different graph views. Traditional GCL methods generate views in the spatial domain, but it has been recently discovered that the spectral domain also plays a vital role in complementing spatial views. However, existing spectral-based graph views either ignore the eigenvectors that encode valuable positional information or suffer from high complexity when trying to address the instability of spectral features. To tackle these challenges, we first design an informative, stable, and scalable spectral encoder, termed EigenMLP, to learn effective representations from the spectral features. Theoretically, EigenMLP is invariant to the rotation and reflection transformations on eigenvectors and robust against perturbations. Then, we propose a spatial-spectral contrastive framework (Sp$^{2}$GCL) to capture the consistency between the spatial information encoded by graph neural networks and the spectral information learned by EigenMLP, thus effectively fusing these two graph views. Experiments on the node- and graph-level datasets show that our method not only learns effective graph representations but also achieves a 2--10x speedup over other spectral-based methods.

----

## [0] A Tale of Two Features: Stable Diffusion Complements DINO for Zero-Shot Semantic Correspondence

**Authors**: *Junyi Zhang, Charles Herrmann, Junhwa Hur, Luisa Polania Cabrera, Varun Jampani, Deqing Sun, Ming-Hsuan Yang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9bdc23f169a05ea9b72ccef4574551-Abstract-Conference.html)

**Abstract**:

Text-to-image diffusion models have made significant advances in generating and editing high-quality images.  As a result, numerous approaches have explored the ability of diffusion model features to understand and process single images for downstream tasks, e.g., classification, semantic segmentation, and stylization. However, significantly less is known about what these features reveal across multiple, different images and objects.  In this work, we exploit Stable Diffusion (SD) features for semantic and dense correspondence and discover that with simple post-processing, SD features can perform quantitatively similar to SOTA representations. Interestingly, the qualitative analysis reveals that SD features have very different properties compared to existing representation learning features, such as the recently released DINOv2: while DINOv2 provides sparse but accurate matches, SD features provide high-quality spatial information but sometimes inaccurate semantic matches. We demonstrate that a simple fusion of these two features works surprisingly well, and a zero-shot evaluation using nearest neighbors on these fused features provides a significant performance gain over state-of-the-art methods on benchmark datasets, e.g., SPair-71k, PF-Pascal, and TSS.  We also show that these correspondences can enable interesting applications such as instance swapping in two images. Project page: https://sd-complements-dino.github.io/.

----

## [0] SatLM: Satisfiability-Aided Language Models Using Declarative Prompting

**Authors**: *Xi Ye, Qiaochu Chen, Isil Dillig, Greg Durrett*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8e9c7d4a48bdac81a58f983a64aaf42b-Abstract-Conference.html)

**Abstract**:

Prior work has combined chain-of-thought prompting in large language models (LLMs) with programmatic representations to perform effective and transparent reasoning. While such an approach works well for tasks that only require forward reasoning (e.g., straightforward arithmetic), it is less effective for constraint solving problems that require more sophisticated planning and search. In this paper, we propose a new satisfiability-aided language modeling (SatLM) approach for improving the reasoning capabilities of LLMs. We use an LLM to generate a declarative task specification rather than an imperative program and leverage an off-the-shelf automated theorem prover to derive the final answer. This approach has two key advantages. The declarative specification is closer to the problem description than the reasoning steps are, so the LLM can parse it out of the description more accurately. Furthermore, by offloading the actual reasoning task to an automated theorem prover, our approach can guarantee the correctness of the answer with respect to the parsed specification and avoid planning errors in the solving process. We evaluate SATLM on 8 different datasets and show that it consistently outperforms program-aided LMs in the imperative paradigm. In particular, SATLM outperforms program-aided LMs by 23% on a challenging subset of the GSM arithmetic reasoning dataset; SATLM also achieves a new SoTA on LSAT and BoardgameQA, surpassing previous models that are trained on the respective training sets.

----

## [0] A normative theory of social conflict

**Authors**: *Sergey Shuvaev, Evgeny Amelchenko, Dmitry A. Smagin, Natalia N. Kudryavtseva, Grigori Enikolopov, Alexei A. Koulakov*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ec61d4084443d29c9e47ac60f9aea31-Abstract-Conference.html)

**Abstract**:

Social conflict is a survival mechanism yielding both normal and pathological behaviors. To understand its underlying principles, we collected behavioral and whole-brain neural data from mice advancing through stages of social conflict. We modeled the animals’ interactions as a normal-form game using Bayesian inference to account for the partial observability of animals’ strengths. We find that our behavioral and neural data are consistent with the first-level Theory of Mind (1-ToM) model where mice form “primary” beliefs about the strengths of all mice involved and “secondary” beliefs that estimate the beliefs of their opponents. Our model identifies the brain regions that carry the information about these beliefs and offers a framework for studies of social behaviors in partially observable settings.

----

## [0] Learning Invariant Representations of Graph Neural Networks via Cluster Generalization

**Authors**: *Donglin Xia, Xiao Wang, Nian Liu, Chuan Shi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ed2293e714b7692b63117e330e551e8-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ed2293e714b7692b63117e330e551e8-Abstract-Conference.html)

**Abstract**:

Graph neural networks (GNNs) have become increasingly popular in modeling graph-structured data due to their ability to learn node representations by aggregating local structure information. However, it is widely acknowledged that the test graph structure may differ from the training graph structure, resulting in a structure shift. In this paper, we experimentally  find that the performance of GNNs drops significantly when the structure shift happens, suggesting that the learned models may be biased towards specific structure patterns. To address this challenge, we propose the Cluster Information Transfer (\textbf{CIT}) mechanism, which can learn invariant representations for GNNs, thereby improving their generalization ability to various and unknown test graphs with structure shift. The CIT mechanism achieves this by combining different cluster information with the nodes while preserving their cluster-independent information. By generating nodes across different clusters, the mechanism significantly enhances the diversity of the nodes and helps GNNs learn the invariant representations. We provide a theoretical analysis of the CIT mechanism, showing that the impact of changing clusters during structure shift can be mitigated after transfer. Additionally, the proposed mechanism is a plug-in that can be easily used to improve existing GNNs. We comprehensively evaluate our proposed method on three typical structure shift scenarios, demonstrating its effectiveness in enhancing GNNs' performance.

----

## [0] Transformers learn to implement preconditioned gradient descent for in-context learning

**Authors**: *Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, Suvrit Sra*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ed3d610ea4b68e7afb30ea7d01422c6-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ed3d610ea4b68e7afb30ea7d01422c6-Abstract-Conference.html)

**Abstract**:

Several recent works demonstrate that transformers can implement algorithms like gradient descent. By a careful construction of weights, these works show that multiple layers of transformers are expressive enough to simulate iterations of gradient descent. Going beyond the question of expressivity, we ask: \emph{Can transformers learn to implement such algorithms by training over random problem instances?} To our knowledge, we make the first theoretical progress on this question via an analysis of the loss landscape for linear transformers trained over random instances of linear regression. For a single attention layer, we prove the global minimum of the training objective implements a single iteration of preconditioned gradient descent. Notably, the preconditioning matrix not only adapts to the input distribution but also to the variance induced by data inadequacy.  For a transformer with $L$ attention layers, we prove certain critical points of the training objective implement $L$ iterations of preconditioned gradient descent. Our results call for future theoretical studies on learning algorithms by training transformers.

----

## [0] Linear Time Algorithms for k-means with Multi-Swap Local Search

**Authors**: *Junyu Huang, Qilong Feng, Ziyun Huang, Jinhui Xu, Jianxin Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8eec8d7bcecf034304174e6b57dbc19a-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8eec8d7bcecf034304174e6b57dbc19a-Abstract-Conference.html)

**Abstract**:

The local search methods have been widely used to solve the clustering problems. In practice, local search algorithms for clustering problems mainly adapt the single-swap strategy, which enables them to handle large-scale datasets and achieve linear running time in the data size. However, compared with multi-swap local search algorithms, there is a considerable gap on the approximation ratios of the single-swap local search algorithms. Although the current multi-swap local search algorithms provide small constant approximation, the proposed algorithms tend to have large polynomial running time, which cannot be used to handle large-scale datasets. In this paper, we propose a multi-swap local search algorithm for the $k$-means problem with linear running time in the data size. Given a swap size $t$, our proposed algorithm can achieve a $(50(1+\frac{1}{t})+\epsilon)$-approximation, which improves the current best result 509 (ICML 2019) with linear running time in the data size. Our proposed method, compared with previous multi-swap local search algorithms, is the first one to achieve linear running time in the data size. To obtain a more practical algorithm for the problem with better clustering quality and running time, we propose a sampling-based method which accelerates the process of clustering cost update during swaps. Besides, a recombination mechanism is proposed to find potentially better solutions. Empirical experiments show that our proposed algorithms achieve better performances compared with branch and bound solver (NeurIPS 2022) and other existing state-of-the-art local search algorithms on both small and large datasets.

----

## [0] VaRT: Variational Regression Trees

**Authors**: *Sebastian Salazar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8eff4196f50c43eda7bcf0f0cf87a0d0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8eff4196f50c43eda7bcf0f0cf87a0d0-Abstract-Conference.html)

**Abstract**:

Decision trees are a well-established tool in machine learning for classification and regression tasks. In this paper, we introduce a novel non-parametric Bayesian model that uses variational inference to approximate a posterior distribution over the space of stochastic decision trees. We evaluate the model's performance on 18 datasets and demonstrate its competitiveness with other state-of-the-art methods in regression tasks. We also explore its application to causal inference problems. We provide a fully vectorized implementation of our algorithm in PyTorch.

----

## [0] STREAMER: Streaming Representation Learning and Event Segmentation in a Hierarchical Manner

**Authors**: *Ramy Mounir, Sujal Vijayaraghavan, Sudeep Sarkar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f0d446441a938d9de420a8ab8d7fd36-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f0d446441a938d9de420a8ab8d7fd36-Abstract-Conference.html)

**Abstract**:

We present a novel self-supervised approach for hierarchical representation learning and segmentation of perceptual inputs in a streaming fashion. Our research addresses how to semantically group streaming inputs into chunks at various levels of a hierarchy while simultaneously learning, for each chunk, robust global representations throughout the domain. To achieve this, we propose STREAMER, an architecture that is trained layer-by-layer, adapting to the complexity of the input domain. In our approach, each layer is trained with two primary objectives: making accurate predictions into the future and providing necessary information to other levels for achieving the same objective. The event hierarchy is constructed by detecting prediction error peaks at different levels, where a detected boundary triggers a bottom-up information flow. At an event boundary, the encoded representation of inputs at one layer becomes the input to a higher-level layer. Additionally, we design a communication module that facilitates top-down and bottom-up exchange of information during the prediction process. Notably, our model is fully self-supervised and trained in a streaming manner, enabling a single pass on the training data. This means that the model encounters each input only once and does not store the data. We evaluate the performance of our model on the egocentric EPIC-KITCHENS dataset, specifically focusing on temporal event segmentation. Furthermore, we conduct event retrieval experiments using the learned representations to demonstrate the high quality of our video event representations. Illustration videos and code are available on our project page: https://ramymounir.com/publications/streamer

----

## [0] Pgx: Hardware-Accelerated Parallel Game Simulators for Reinforcement Learning

**Authors**: *Sotetsu Koyamada, Shinri Okano, Soichiro Nishimori, Yu Murata, Keigo Habara, Haruka Kita, Shin Ishii*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f153093758af93861a74a1305dfdc18-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We propose Pgx, a suite of board game reinforcement learning (RL) environments written in JAX and optimized for GPU/TPU accelerators. By leveraging JAX's auto-vectorization and parallelization over accelerators, Pgx can efficiently scale to thousands of simultaneous simulations over accelerators. In our experiments on a DGX-A100 workstation, we discovered that Pgx can simulate RL environments 10-100x faster than existing implementations available in Python. Pgx includes RL environments commonly used as benchmarks in RL research, such as backgammon, chess, shogi, and Go. Additionally, Pgx offers miniature game sets and baseline models to facilitate rapid research cycles. We demonstrate the efficient training of the Gumbel AlphaZero algorithm with Pgx environments. Overall, Pgx provides high-performance environment simulators for researchers to accelerate their RL experiments. Pgx is available at https://github.com/sotetsuk/pgx.

----

## [0] Robust Distributed Learning: Tight Error Bounds and Breakdown Point under Data Heterogeneity

**Authors**: *Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Rafael Pinot, Geovani Rizk*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f182e220092f7f1fc44f3313023f5a0-Abstract-Conference.html)

**Abstract**:

The theory underlying robust distributed learning algorithms, designed to resist adversarial machines, matches empirical observations when data is homogeneous. Under data heterogeneity however, which is the norm in practical scenarios, established lower bounds on the learning error are essentially vacuous and greatly mismatch empirical observations. This is because the heterogeneity model considered is too restrictive and does not cover basic learning tasks such as least-squares regression. We consider in this paper a more realistic heterogeneity model, namely $(G,B)$-gradient dissimilarity, and show that it covers a larger class of learning problems than existing theory. Notably, we show that the breakdown point under heterogeneity is lower than the classical fraction $\frac{1}{2}$. We also prove a new lower bound on the learning error of any distributed learning algorithm. We derive a matching upper bound for a robust variant of distributed gradient descent, and empirically show that our analysis reduces the gap between theory and practice.

----

## [0] Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

**Authors**: *Zixuan Jiang, Jiaqi Gu, Hanqing Zhu, David Z. Pan*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f1bacee31caf990a4f08d84f0ccb322-Abstract-Conference.html)

**Abstract**:

Transformers have achieved great success in machine learning applications.Normalization techniques, such as Layer Normalization (LayerNorm, LN) and Root Mean Square Normalization (RMSNorm), play a critical role in accelerating and stabilizing the training of Transformers.While LayerNorm recenters and rescales input vectors, RMSNorm only rescales the vectors by their RMS value.Despite being more computationally efficient, RMSNorm may compromise the representation ability of Transformers.There is currently no consensus regarding the preferred normalization technique, as some models employ LayerNorm while others utilize RMSNorm, especially in recent large language models.It is challenging to convert Transformers with one normalization to the other type.While there is an ongoing disagreement between the two normalization types,we propose a solution to unify two mainstream Transformer architectures, Pre-LN and Pre-RMSNorm Transformers.By removing the inherent redundant mean information in the main branch of Pre-LN Transformers, we can reduce LayerNorm to RMSNorm, achieving higher efficiency.We further propose the Compressed RMSNorm (CRMSNorm) and Pre-CRMSNorm Transformer based on a lossless compression of the zero-mean vectors.We formally establish the equivalence of Pre-LN, Pre-RMSNorm, and Pre-CRMSNorm Transformer variants in both training and inference.It implies that Pre-LN Transformers can be substituted with Pre-(C)RMSNorm counterparts at almost no cost, offering the same arithmetic functionality along with free efficiency improvement.Experiments demonstrate that we can reduce the training and inference time of Pre-LN Transformers by 1% - 10%.

----

## [0] Multimodal Clinical Benchmark for Emergency Care (MC-BEC): A Comprehensive Benchmark for Evaluating Foundation Models in Emergency Medicine

**Authors**: *Emma Chen, Aman Kansal, Julie Chen, Boyang Tom Jin, Julia Rachel Reisler, David A. Kim, Pranav Rajpurkar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f61049e8fe5b9ed714860b951066f1e-Abstract-Datasets_and_Benchmarks.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f61049e8fe5b9ed714860b951066f1e-Abstract-Datasets_and_Benchmarks.html)

**Abstract**:

We propose the Multimodal Clinical Benchmark for Emergency Care (MC-BEC), a comprehensive benchmark for evaluating foundation models in Emergency Medicine using a dataset of 100K+ continuously monitored Emergency Department visits from 2020-2022. MC-BEC focuses on clinically relevant prediction tasks at timescales from minutes to days, including predicting patient decompensation, disposition, and emergency department (ED) revisit, and includes a standardized evaluation framework with train-test splits and evaluation metrics. The multimodal dataset includes a wide range of detailed clinical data, including triage information, prior diagnoses and medications, continuously measured vital signs, electrocardiogram and photoplethysmograph waveforms, orders placed and medications administered throughout the visit, free-text reports of imaging studies, and information on ED diagnosis, disposition, and subsequent revisits. We provide performance baselines for each prediction task to enable the evaluation of multimodal, multitask models. We believe that MC-BEC will encourage researchers to develop more effective, generalizable, and accessible foundation models for multimodal clinical data.

----

## [0] AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix

**Authors**: *Yun Yue, Zhiling Ye, Jiadi Jiang, Yongchao Liu, Ke Zhang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f9d459c19b59b5400ce396e0f8c23e0-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f9d459c19b59b5400ce396e0f8c23e0-Abstract-Conference.html)

**Abstract**:

Adaptive optimizers, such as Adam, have achieved remarkable success in deep learning. A key component of these optimizers is the so-called preconditioning matrix, providing enhanced gradient information and regulating the step size of each gradient direction. In this paper, we propose a novel approach to designing the preconditioning matrix by utilizing the gradient difference between two successive steps as the diagonal elements. These diagonal elements are closely related to the Hessian and can be perceived as an approximation of the inner product between the Hessian row vectors and difference of the adjacent parameter vectors. Additionally, we introduce an auto-switching function that enables the preconditioning matrix to switch dynamically between Stochastic Gradient Descent (SGD) and the adaptive optimizer. Based on these two techniques, we develop a new optimizer named AGD that enhances the generalization performance. We evaluate AGD on public datasets of Natural Language Processing (NLP), Computer Vision (CV), and Recommendation Systems (RecSys). Our experimental results demonstrate that AGD outperforms the state-of-the-art (SOTA) optimizers, achieving highly competitive or significantly better predictive performance. Furthermore, we analyze how AGD is able to switch automatically between SGD and the adaptive optimizer and its actual effects on various scenarios. The code is available at https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch/atorch/optimizers.

----

## [0] PDP: Parameter-free Differentiable Pruning is All You Need

**Authors**: *Minsik Cho, Saurabh Adya, Devang Naik*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8f9f4eb32b9081a90f2a0b2627eb2a24-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8f9f4eb32b9081a90f2a0b2627eb2a24-Abstract-Conference.html)

**Abstract**:

DNN pruning is a popular way to reduce the size of a model, improve the inferencelatency, and minimize the power consumption on DNN accelerators. However,existing approaches might be too complex, expensive or ineffective to apply toa variety of vision/language tasks, DNN architectures and to honor structuredpruning constraints. In this paper, we propose an efficient yet effective train-timepruning scheme, Parameter-free Differentiable Pruning (PDP), which offers state-of-the-art qualities in model size, accuracy, and training cost. PDP uses a dynamicfunction of weights during training to generate soft pruning masks for the weightsin a parameter-free manner for a given pruning target. While differentiable, thesimplicity and efficiency of PDP make it universal enough to deliver state-of-the-artrandom/structured/channel pruning results on various vision and natural languagetasks. For example, for MobileNet-v1, PDP can achieve 68.2% top-1 ImageNet1kaccuracy at 86.6% sparsity, which is 1.7% higher accuracy than those from thestate-of-the-art algorithms. Also, PDP yields over 83.1% accuracy on Multi-GenreNatural Language Inference with 90% sparsity for BERT, while the next best fromthe existing techniques shows 81.5% accuracy. In addition, PDP can be applied tostructured pruning, such as N:M pruning and channel pruning. For 1:4 structuredpruning of ResNet18, PDP improved the top-1 ImageNet1k accuracy by over 3.6%over the state-of-the-art. For channel pruning of ResNet50, PDP reduced the top-1ImageNet1k accuracy by 0.6% from the state-of-the-art.

----

## [0] ExPT: Synthetic Pretraining for Few-Shot Experimental Design

**Authors**: *Tung Nguyen, Sudhanshu Agrawal, Aditya Grover*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fab4407e1fe9006b39180525c0d323c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fab4407e1fe9006b39180525c0d323c-Abstract-Conference.html)

**Abstract**:

Experimental design is a fundamental problem in many science and engineering fields. In this problem, sample efficiency is crucial due to the time, money, and safety costs of real-world design evaluations. Existing approaches either rely on active data collection or access to large, labeled datasets of past experiments, making them impractical in many real-world scenarios. In this work, we address the more challenging yet realistic setting of few-shot experimental design, where only a few labeled data points of input designs and their corresponding values are available. We approach this problem as a conditional generation task, where a model conditions on a few labeled examples and the desired output to generate an optimal input design. To this end, we introduce Experiment Pretrained Transformers (ExPT), a foundation model for few-shot experimental design that employs a novel combination of synthetic pretraining with in-context learning. In ExPT, we only assume knowledge of a finite collection of unlabelled data points from the input domain and pretrain a transformer neural network to optimize diverse synthetic functions defined over this domain. Unsupervised pretraining allows ExPT to adapt to any design task at test time in an in-context fashion by conditioning on a few labeled data points from the target task and generating the candidate optima. We evaluate ExPT on few-shot experimental design in challenging domains and demonstrate its superior generality and performance compared to existing methods. The source code is available at https://github.com/tung-nd/ExPT.git.

----

## [0] ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings

**Authors**: *Shibo Hao, Tianyang Liu, Zhen Wang, Zhiting Hu*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fd1a81c882cd45f64958da6284f4a3f-Abstract-Conference.html)

**Abstract**:

Integrating large language models (LLMs) with various tools has led to increased attention in the field. Existing approaches either involve fine-tuning the LLM, which is both computationally costly and limited to a fixed set of tools, or prompting LLMs by in-context tool demonstrations. Although the latter method offers adaptability to new tools, it struggles with the inherent context length constraint of LLMs when many new tools are presented, and mastering a new set of tools with few-shot examples remains challenging, resulting in suboptimal performance. To address these limitations, we propose a novel solution, named ToolkenGPT, wherein LLMs effectively learn to master tools as predicting tokens through tool embeddings for solving complex tasks. In this framework, each tool is transformed into vector embeddings and plugged into the language model head. Once the function is triggered during text generation, the LLM enters a special function mode to execute the tool calls. Our experiments show that function embeddings effectively help LLMs understand tool use and improve on several tasks, including numerical reasoning, knowledge-based question answering and embodied decision-making.

----

## [0] CLIP4HOI: Towards Adapting CLIP for Practical Zero-Shot HOI Detection

**Authors**: *Yunyao Mao, Jiajun Deng, Wengang Zhou, Li Li, Yao Fang, Houqiang Li*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8fd5bc08e744fe0dfe798c61d1575a22-Abstract-Conference.html)

**Abstract**:

Zero-shot Human-Object Interaction (HOI) detection aims to identify both seen and unseen HOI categories. A strong zero-shot HOI detector is supposed to be not only capable of discriminating novel interactions but also robust to positional distribution discrepancy between seen and unseen categories when locating human-object pairs. However, top-performing zero-shot HOI detectors rely on seen and predefined unseen categories to distill knowledge from CLIP and jointly locate human-object pairs without considering the potential positional distribution discrepancy, leading to impaired transferability. In this paper, we introduce CLIP4HOI, a novel framework for zero-shot HOI detection. CLIP4HOI is developed on the vision-language model CLIP and ameliorates the above issues in the following two aspects. First, to avoid the model from overfitting to the joint positional distribution of seen human-object pairs, we seek to tackle the problem of zero-shot HOI detection in a disentangled two-stage paradigm. To be specific, humans and objects are independently identified and all feasible human-object pairs are processed by Human-Object interactor for pairwise proposal generation. Second, to facilitate better transferability, the CLIP model is elaborately adapted into a fine-grained HOI classifier for proposal discrimination, avoiding data-sensitive knowledge distillation. Finally, experiments on prevalent benchmarks show that our CLIP4HOI outperforms previous approaches on both rare and unseen categories, and sets a series of state-of-the-art records under a variety of zero-shot settings.

----

## [0] Transformer-based Planning for Symbolic Regression

**Authors**: *Parshin Shojaee, Kazem Meidani, Amir Barati Farimani, Chandan K. Reddy*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/8ffb4e3118280a66b192b6f06e0e2596-Abstract-Conference.html)

**Abstract**:

Symbolic regression (SR) is a challenging task in machine learning that involves finding a mathematical expression for a function based on its values. Recent advancements in SR have demonstrated the effectiveness of pre-trained transformer models in generating equations as sequences, leveraging large-scale pre-training on synthetic datasets and offering notable advantages in terms of inference time over classical Genetic Programming (GP) methods. However, these models primarily rely on supervised pre-training objectives borrowed from text generation and overlook equation discovery goals like accuracy and complexity. To address this, we propose TPSR, a Transformer-based Planning strategy for Symbolic Regression that incorporates Monte Carlo Tree Search planning algorithm into the transformer decoding process. Unlike conventional decoding strategies, TPSR enables the integration of non-differentiable equation verification feedback, such as fitting accuracy and complexity, as external sources of knowledge into the transformer equation generation process. Extensive experiments on various datasets show that our approach outperforms state-of-the-art methods, enhancing the model's fitting-complexity trade-off, extrapolation abilities, and robustness to noise.

----

## [0] Exploring Geometry of Blind Spots in Vision models

**Authors**: *Sriram Balasubramanian, Gaurang Sriramanan, Vinu Sankar Sadasivan, Soheil Feizi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90043ebd68500f9efe84fedf860a64f3-Abstract-Conference.html)

**Abstract**:

Despite the remarkable success of deep neural networks in a myriad of settings, several works have demonstrated their overwhelming sensitivity to near-imperceptible perturbations, known as adversarial attacks. On the other hand, prior works have also observed that deep networks can be under-sensitive, wherein large-magnitude perturbations in input space do not induce appreciable changes to network activations. In this work, we study in detail the phenomenon of under-sensitivity in vision models such as CNNs and Transformers, and present techniques to study the geometry and extent of “equi-confidence” level sets of such networks. We propose a Level Set Traversal algorithm that iteratively explores regions of high confidence with respect to the input space using orthogonal components of the local gradients. Given a source image, we use this algorithm to identify inputs that lie in the same equi-confidence level set as the source image despite being perceptually similar to arbitrary images from other classes. We further observe that the source image is linearly connected by a high-confidence path to these inputs, uncovering a star-like structure for level sets of deep networks. Furthermore, we attempt to identify and estimate the extent of these connected higher-dimensional regions over which the model maintains a high degree of confidence.

----

## [0] Provable benefits of annealing for estimating normalizing constants: Importance Sampling, Noise-Contrastive Estimation, and beyond

**Authors**: *Omar Chehab, Aapo Hyvärinen, Andrej Risteski*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/90080022263cddafddd4a0726f1fb186-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/90080022263cddafddd4a0726f1fb186-Abstract-Conference.html)

**Abstract**:

Recent research has developed several Monte Carlo methods for estimating the normalization constant (partition function) based on the idea of annealing. This means sampling successively from a path of distributions which interpolate between a tractable "proposal" distribution and the unnormalized "target" distribution. Prominent estimators in this family include annealed importance sampling and annealed noise-contrastive estimation (NCE). Such methods hinge on a number of design choices: which estimator to use, which path of distributions to use and whether to use a path at all; so far, there is no definitive theory on which choices are efficient. Here, we evaluate each design choice by the asymptotic estimation error it produces. First, we show that using NCE is more efficient than the importance sampling estimator, but in the limit of infinitesimal path steps, the difference vanishes. Second, we find that using the geometric path brings down the estimation error from an exponential to a polynomial function of the parameter distance between the target and proposal distributions. Third, we find that the arithmetic path, while rarely used, can offer optimality properties over the universally-used geometric path. In fact, in a particular limit, the optimal path is arithmetic. Based on this theory, we finally propose a two-step estimator to approximate the optimal path in an efficient way.

----

## [0] Strategic Distribution Shift of Interacting Agents via Coupled Gradient Flows

**Authors**: *Lauren E. Conger, Franca Hoffmann, Eric Mazumdar, Lillian J. Ratliff*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/902c462e821e5e639ac3422b48b65932-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/902c462e821e5e639ac3422b48b65932-Abstract-Conference.html)

**Abstract**:

We propose a novel framework for analyzing the dynamics of distribution shift in real-world systems that captures the feedback loop between learning algorithms and the distributions on which they are deployed. Prior work largely models feedback-induced distribution shift as adversarial or via an overly simplistic distribution-shift structure. In contrast, we propose a coupled partial differential equation model that captures fine-grained changes in the distribution over time by accounting for complex dynamics that arise due to strategic responses to algorithmic decision-making, non-local endogenous population interactions, and other exogenous sources of distribution shift. We consider two common settings in machine learning: cooperative settings with information asymmetries, and competitive settings where a learner faces strategic users. For both of these settings, when the algorithm retrains via gradient descent, we prove asymptotic convergence of the retraining procedure to a steady-state, both in finite and in infinite dimensions, obtaining explicit rates in terms of the model parameters. To do so we derive new results on the convergence of coupled PDEs that extends what is known on multi-species systems. Empirically, we show that our approach captures well-documented forms of distribution shifts like polarization and disparate impacts that simpler models cannot capture.

----

## [0] Learning Time-Invariant Representations for Individual Neurons from Population Dynamics

**Authors**: *Lu Mi, Trung Le, Tianxing He, Eli Shlizerman, Uygar Sümbül*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9032e5c9ec394ce768a2fa9bdc56af6c-Abstract-Conference.html)

**Abstract**:

Neurons can display highly variable dynamics. While such variability presumably supports the wide range of behaviors generated by the organism, their gene expressions are relatively stable in the adult brain. This suggests that neuronal activity is a combination of its time-invariant identity and the inputs the neuron receives from the rest of the circuit. Here, we propose a self-supervised learning based method to assign time-invariant representations to individual neurons based on permutation-, and population size-invariant summary of population recordings. We fit dynamical models to neuronal activity to learn a representation by considering the activity of both the individual and the neighboring population. Our self-supervised approach and use of implicit representations enable robust inference against imperfections such as partial overlap of neurons across sessions, trial-to-trial variability, and limited availability of molecular (transcriptomic) labels for downstream supervised tasks. We demonstrate our method on a public multimodal dataset of mouse cortical neuronal activity and transcriptomic labels. We report >35\% improvement in predicting the transcriptomic subclass identity and >20\% improvement in predicting class identity with respect to the state-of-the-art.

----

## [0] GeoTMI: Predicting Quantum Chemical Property with Easy-to-Obtain Geometry via Positional Denoising

**Authors**: *Hyeonsu Kim, Jeheon Woo, Seonghwan Kim, Seokhyun Moon, Jun Hyeong Kim, Woo Youn Kim*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/903c5eb12f2389c4847574df90503d63-Abstract-Conference.html)

**Abstract**:

As quantum chemical properties have a dependence on their geometries, graph neural networks (GNNs) using 3D geometric information have achieved high prediction accuracy in many tasks. However, they often require 3D geometries obtained from high-level quantum mechanical calculations, which are practically infeasible, limiting their applicability to real-world problems. To tackle this, we propose a new training framework, GeoTMI, that employs denoising process to predict properties accurately using easy-to-obtain geometries (corrupted versions of correct geometries, such as those obtained from low-level calculations). Our starting point was the idea that the correct geometry is the best description of the target property. Hence, to incorporate information of the correct, GeoTMI aims to maximize mutual information between three variables: the correct and the corrupted geometries and the property. GeoTMI also explicitly updates the corrupted input to approach the correct geometry as it passes through the GNN layers, contributing to more effective denoising. We investigated the performance of the proposed method using 3D GNNs for three prediction tasks: molecular properties, a chemical reaction property, and relaxed energy in a heterogeneous catalytic system. Our results showed consistent improvements in accuracy across various tasks, demonstrating the effectiveness and robustness of GeoTMI.

----

## [0] PRED: Pre-training via Semantic Rendering on LiDAR Point Clouds

**Authors**: *Hao Yang, Haiyang Wang, Di Dai, Liwei Wang*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/903f778fe1341e5351b5b63e0e6b197f-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/903f778fe1341e5351b5b63e0e6b197f-Abstract-Conference.html)

**Abstract**:

Pre-training is crucial in 3D-related fields such as autonomous driving where point cloud annotation is costly and challenging. Many recent studies on point cloud pre-training, however, have overlooked the issue of incompleteness, where only a fraction of the points are captured by LiDAR, leading to ambiguity during the training phase. On the other hand, images offer more comprehensive information and richer semantics that can bolster point cloud encoders in addressing the incompleteness issue inherent in point clouds. Yet, incorporating images into point cloud pre-training presents its own challenges due to occlusions, potentially causing misalignments between points and pixels. In this work, we propose PRED, a novel image-assisted pre-training framework for outdoor point clouds in an occlusion-aware manner. The main ingredient of our framework is a Birds-Eye-View (BEV) feature map conditioned semantic rendering, leveraging the semantics of images for supervision through neural rendering. We further enhance our model's performance by incorporating point-wise masking with a high mask ratio (95%). Extensive experiments demonstrate PRED's superiority over prior point cloud pre-training methods, providing significant improvements on various large-scale datasets for 3D perception tasks. Codes will be available at https://github.com/PRED4pc/PRED.

----

## [0] Active Observing in Continuous-time Control

**Authors**: *Samuel Holt, Alihan Hüyük, Mihaela van der Schaar*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9050e8d5b5de08d16e65dc79ad5c0146-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9050e8d5b5de08d16e65dc79ad5c0146-Abstract-Conference.html)

**Abstract**:

The control of continuous-time environments while actively deciding when to take costly observations in time is a crucial yet unexplored problem, particularly relevant to real-world scenarios such as medicine, low-power systems, and resource management. Existing approaches either rely on continuous-time control methods that take regular, expensive observations in time or discrete-time control with costly observation methods, which are inapplicable to continuous-time settings due to the compounding discretization errors introduced by time discretization. In this work, we are the first to formalize the continuous-time control problem with costly observations. Our key theoretical contribution shows that observing at regular time intervals is not optimal in certain environments, while irregular observation policies yield higher expected utility.  This perspective paves the way for the development of novel methods that can take irregular observations in continuous-time control with costly observations. We empirically validate our theoretical findings in various continuous-time environments, including a cancer simulation, by constructing a simple initial method to solve this new problem, with a heuristic threshold on the variance of reward rollouts in an offline continuous-time model-based model predictive control (MPC) planner. Although determining the optimal method remains an open problem, our work offers valuable insights and understanding of this unique problem, laying the foundation for future research in this area.

----

## [0] Principled Weight Initialisation for Input-Convex Neural Networks

**Authors**: *Pieter-Jan Hoedt, Günter Klambauer*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/9062b7d6e522dadf4f7d85d49b60d81e-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/9062b7d6e522dadf4f7d85d49b60d81e-Abstract-Conference.html)

**Abstract**:

Input-Convex Neural Networks (ICNNs) are networks that guarantee convexity in their input-output mapping. These networks have been successfully applied for energy-based modelling, optimal transport problems and learning invariances.The convexity of ICNNs is achieved by using non-decreasing convex activation functions and non-negative weights. Because of these peculiarities, previous initialisation strategies, which implicitly assume centred weights, are not effective for ICNNs. By studying signal propagation through layers with non-negative weights, we are able to derive a principled weight initialisation for ICNNs. Concretely, we generalise signal propagation theory by removing the assumption that weights are sampled from a centred distribution. In a set of experiments, we demonstrate that our principled initialisation effectively accelerates learning in ICNNs and leads to better generalisation. Moreover, we find that, in contrast to common belief, ICNNs can be trained without skip-connections when initialised correctly. Finally, we apply ICNNs to a real-world drug discovery task and show that they allow for more effective molecular latent space exploration.

----

## [0] Automatic Grouping for Efficient Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Yifan Zang, Jinmin He, Kai Li, Haobo Fu, Qiang Fu, Junliang Xing, Jian Cheng*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/906c860f1b7515a8ffec02dcdac74048-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/906c860f1b7515a8ffec02dcdac74048-Abstract-Conference.html)

**Abstract**:

Grouping is ubiquitous in natural systems and is essential for promoting efficiency in team coordination. This paper proposes a novel formulation of Group-oriented Multi-Agent Reinforcement Learning (GoMARL), which learns automatic grouping without domain knowledge for efficient cooperation. In contrast to existing approaches that attempt to directly learn the complex relationship between the joint action-values and individual utilities, we empower subgroups as a bridge to model the connection between small sets of agents and encourage cooperation among them, thereby improving the learning efficiency of the whole team. In particular, we factorize the joint action-values as a combination of group-wise values, which guide agents to improve their policies in a fine-grained fashion. We present an automatic grouping mechanism to generate dynamic groups and group action-values. We further introduce a hierarchical control for policy learning that drives the agents in the same group to specialize in similar policies and possess diverse strategies for various groups. Experiments on the StarCraft II micromanagement tasks and Google Research Football scenarios verify our method's effectiveness. Extensive component studies show how grouping works and enhances performance.

----

## [0] On the Minimax Regret for Online Learning with Feedback Graphs

**Authors**: *Khaled Eldowa, Emmanuel Esposito, Tommaso Cesari, Nicolò Cesa-Bianchi*

**Conference**: *nips 2023*

**URL**: [http://papers.nips.cc/paper_files/paper/2023/hash/908f03779b5b063413fbf0247a46a403-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2023/hash/908f03779b5b063413fbf0247a46a403-Abstract-Conference.html)

**Abstract**:

In this work, we improve on the upper and lower bounds for the regret of online learning with strongly observable undirected feedback graphs. The best known upper bound for this problem is $\mathcal{O}\bigl(\sqrt{\alpha T\ln K}\bigr)$, where $K$ is the number of actions, $\alpha$ is the independence number of the graph, and $T$ is the time horizon. The $\sqrt{\ln K}$ factor is known to be necessary when $\alpha = 1$ (the experts case). On the other hand, when $\alpha = K$ (the bandits case), the minimax rate is known to be $\Theta\bigl(\sqrt{KT}\bigr)$, and a lower bound $\Omega\bigl(\sqrt{\alpha T}\bigr)$ is known to hold for any $\alpha$. Our improved upper bound $\mathcal{O}\bigl(\sqrt{\alpha T(1+\ln(K/\alpha))}\bigr)$ holds for any $\alpha$ and matches the lower bounds for bandits and experts, while interpolating intermediate cases. To prove this result, we use FTRL with $q$-Tsallis entropy for a carefully chosen value of $q \in [1/2, 1)$ that varies with $\alpha$. The analysis of this algorithm requires a new bound on the variance term in the regret. We also show how to extend our techniques to time-varying graphs, without requiring prior knowledge of their independence numbers. Our upper bound is complemented by an improved $\Omega\bigl(\sqrt{\alpha T(\ln K)/(\ln\alpha)}\bigr)$ lower bound for all $\alpha > 1$, whose analysis relies on a novel reduction to multitask learning. This shows that a logarithmic factor is necessary as soon as $\alpha < K$.

----



[Go to the previous page](NIPS-2023-list1900.md)

[Go to the next page](NIPS-2023-list1902.md)

[Go to the catalog section](README.md)