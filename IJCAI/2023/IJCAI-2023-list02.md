## [200] RePaint-NeRF: NeRF Editting via Semantic Masks and Diffusion Models

**Authors**: *Xingchen Zhou, Ying He, F. Richard Yu, Jianqiang Li, You Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/201](https://doi.org/10.24963/ijcai.2023/201)

**Abstract**:

The emergence of Neural Radiance Fields (NeRF) has promoted the development of synthesized high-fidelity views of the intricate real world. However, it is still a very demanding task to repaint the content in NeRF. In this paper, we propose a novel framework that can take RGB images as input and alter the 3D content in neural scenes. Our work leverages existing diffusion models to guide changes in the designated 3D content. Specifically, we semantically select the target object and a pre-trained diffusion model will guide the NeRF model to generate new 3D objects, which can improve the editability, diversity, and application range of NeRF. Experiment results show that our algorithm is effective for editing 3D objects in NeRF under different text prompts, including editing appearance, shape, and more. We validate our method on both real-world datasets and synthetic-world datasets for these editing tasks. Please visit https://repaintnerf.github.io for a better view of our results.

----

## [201] Dichotomous Image Segmentation with Frequency Priors

**Authors**: *Yan Zhou, Bo Dong, Yuanfeng Wu, Wentao Zhu, Geng Chen, Yanning Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/202](https://doi.org/10.24963/ijcai.2023/202)

**Abstract**:

Dichotomous image segmentation (DIS) has a wide range of real-world applications and gained increasing research attention in recent years. In this paper, we propose to tackle DIS with informative frequency priors. Our model, called FP-DIS, stems from the fact that prior knowledge in the frequency domain can provide valuable cues to identify fine-grained object boundaries. Specifically, we propose a frequency prior generator to jointly utilize a fixed filter and learnable filters to extract informative frequency priors. Before embedding the frequency priors into the network, we first harmonize the multi-scale side-out features to reduce their heterogeneity. This is achieved by our feature harmonization module, which is based on a gating mechanism to harmonize the grouped features. Finally, we propose a frequency prior embedding module to embed the frequency priors into multi-scale features through an adaptive modulation strategy. Extensive experiments on the benchmark dataset, DIS5K, demonstrate that our FP-DIS outperforms state-of-the-art methods by a large margin in terms of key evaluation metrics.

----

## [202] A Solution to Co-occurence Bias: Attributes Disentanglement via Mutual Information Minimization for Pedestrian Attribute Recognition

**Authors**: *Yibo Zhou, Hai-Miao Hu, Jinzuo Yu, Zhenbo Xu, Weiqing Lu, Yuran Cao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/203](https://doi.org/10.24963/ijcai.2023/203)

**Abstract**:

Recent studies on pedestrian attribute recognition progress with either explicit or implicit modeling of the co-occurence among attributes. Considering that this known a prior is highly variable and unforeseeable regarding the specific scenarios, we show that current methods can actually suffer in generalizing such fitted attributes interdependencies onto scenes or identities off the dataset distribution, resulting in the underlined bias of attributes co-occurence. To render models robust in realistic scenes, we propose the attributes-disentangled feature learning to ensure the recognition of an attribute not inferring on the existence of others, and which is sequentially formulated as a problem of mutual information minimization. Rooting from it, practical strategies are devised to efficiently decouple attributes, which substantially improve the baseline and establish state-of-the-art performance on realistic datasets like PETAzs and RAPzs.

----

## [203] Vision Language Navigation with Knowledge-driven Environmental Dreamer

**Authors**: *Fengda Zhu, Vincent CS Lee, Xiaojun Chang, Xiaodan Liang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/204](https://doi.org/10.24963/ijcai.2023/204)

**Abstract**:

Vision-language navigation (VLN) requires an agent to perceive visual observation in a house scene and navigate step-by-step following natural language instruction. Due to the high cost of data annotation and data collection, current VLN datasets provide limited instruction-trajectory data samples. Learning vision-language alignment for VLN from limited data is challenging since visual observation and language instruction are both complex and diverse. Previous works only generate augmented data based on original scenes while failing to generate data samples from unseen scenes, which limits the generalization ability of the navigation agent. In this paper, we introduce the Knowledge-driven Environmental Dreamer (KED), a method that leverages the knowledge of the embodied environment and generates unseen scenes for a navigation agent to learn. Generating an unseen environment with texture consistency and structure consistency is challenging. To address this problem, we incorporate three knowledge-driven regularization objectives into the KED and adopt a reweighting mechanism for self-adaptive optimization. Our KED method is able to generate unseen embodied environments without extra annotations. We use KED to successfully generate 270 houses and 500K instruction-trajectory pairs. The navigation agent with the KED method outperforms the state-of-the-art methods on various VLN benchmarks, such as R2R, R4R, and RxR. Both qualitative and quantitative experiments prove that our proposed KED method is able to high-quality augmentation data with texture consistency and structure consistency.

----

## [204] Efficient Multi-View Inverse Rendering Using a Hybrid Differentiable Rendering Method

**Authors**: *Xiangyang Zhu, Yiling Pan, Bailin Deng, Bin Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/205](https://doi.org/10.24963/ijcai.2023/205)

**Abstract**:

Recovering the shape and appearance of real-world objects from natural 2D images is a long-standing and challenging inverse rendering problem. In this paper, we introduce a novel hybrid differentiable rendering method to efficiently reconstruct the 3D geometry and reflectance of a scene from multi-view images captured by conventional hand-held cameras. Our method follows an analysis-by-synthesis approach and consists of two phases. In the initialization phase, we use traditional SfM and MVS methods to reconstruct a virtual scene roughly matching the real scene. Then in the optimization phase, we adopt a hybrid approach to refine the geometry and reflectance, where the geometry is first optimized using an approximate differentiable rendering method, and the reflectance is optimized afterward using a physically-based differentiable rendering method. Our hybrid approach combines the efficiency of approximate methods with the high-quality results of physically-based methods. Extensive experiments on synthetic and real data demonstrate that our method can produce reconstructions with similar or higher quality than state-of-the-art methods while being more efficient.

----

## [205] Towards Accurate Video Text Spotting with Text-wise Semantic Reasoning

**Authors**: *Xinyan Zu, Haiyang Yu, Bin Li, Xiangyang Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/206](https://doi.org/10.24963/ijcai.2023/206)

**Abstract**:

Video text spotting (VTS) aims at extracting texts from videos, where text detection, tracking and recognition are conducted simultaneously. There have been some works that can tackle VTS; however, they may ignore the underlying semantic relationships among texts within a frame. We observe that the texts within a frame usually share similar semantics, which suggests that, if one text is predicted incorrectly by a text recognizer, it still has a chance to be corrected via semantic reasoning. In this paper, we propose an accurate video text spotter, VLSpotter, that reads texts visually, linguistically, and semantically. For ‘visually’, we propose a plug-and-play text-focused super-resolution module to alleviate motion blur and enhance video quality. For ‘linguistically’, a language model is employed to capture intra-text context to mitigate wrongly spelled text predictions. For ‘semantically’, we propose a text-wise semantic reasoning module to model inter-text semantic relationships and reason for better results. The experimental results on multiple VTS benchmarks demonstrate that the proposed VLSpotter outperforms the existing state-of-the-art methods in end-to-end video text spotting.

----

## [206] A Regular Matching Constraint for String Variables

**Authors**: *Roberto Amadini, Peter J. Stuckey*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/207](https://doi.org/10.24963/ijcai.2023/207)

**Abstract**:

Using a regular language as a pattern for string matching is nowadays a common -and sometimes unsafe- operation, provided as a built-in feature by most programming languages. A proper constraint solver over string variables should support most of the operations over regular expressions and related constructs. However, state-of-the-art string solvers natively support only the membership relation of a string variable to a regular language. Here we take a step forward by defining a specialised propagator for the match operation, returning the leftmost position where a pattern can match a given string. Empirical evidences show the effectiveness of our approach, implemented within the constraint programming framework, and tested against state-of-the-art string solvers.

----

## [207] Learning Constraint Networks over Unknown Constraint Languages

**Authors**: *Christian Bessiere, Clément Carbonnel, Areski Himeur*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/208](https://doi.org/10.24963/ijcai.2023/208)

**Abstract**:

Constraint acquisition is the task of learning a constraint network from examples of solutions and non-solutions. Existing constraint acquisition systems typically require advance knowledge of the target network's constraint language, which significantly narrows their scope of applicability. In this paper we propose a constraint acquisition method that computes a suitable constraint language as part of the learning process, eliminating the need for any advance knowledge. We report preliminary experiments on various acquisition benchmarks.

----

## [208] Faster Exact MPE and Constrained Optimization with Deterministic Finite State Automata

**Authors**: *Filippo Bistaffa*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/209](https://doi.org/10.24963/ijcai.2023/209)

**Abstract**:

We propose a concise function representation based on deterministic finite state automata for exact most probable explanation and constrained optimization tasks in graphical models. We then exploit our concise representation within Bucket Elimination (BE). We denote our version of BE as FABE. FABE significantly improves the performance of BE in terms of runtime and memory requirements by minimizing redundancy. Indeed, results on most probable explanation and weighted constraint satisfaction benchmarks show that FABE often outperforms the state of the art, leading to significant runtime improvements (up to 2 orders of magnitude in our tests).

----

## [209] Constraints First: A New MDD-based Model to Generate Sentences Under Constraints

**Authors**: *Alexandre Bonlarron, Aurélie Calabrèse, Pierre Kornprobst, Jean-Charles Régin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/210](https://doi.org/10.24963/ijcai.2023/210)

**Abstract**:

This paper introduces a new approach to generating strongly constrained texts. We consider standardized sentence generation for the typical application of vision screening. To solve this problem, we formalize it as a discrete combinatorial optimization problem and utilize multivalued decision diagrams (MDD), a well-known data structure to deal with constraints. In our context, one key strength of MDD is to compute an exhaustive set of solutions without performing any search. Once the sentences are obtained, we apply a language model (GPT-2) to keep the best ones. We detail this for English and also for French where the agreement and conjugation rules are known to be more complex. Finally, with the help of GPT-2, we get hundreds of bona-fide candidate sentences. When compared with the few dozen sentences usually available in the well-known vision screening test (MNREAD), this brings a major breakthrough in the field of standardized sentence generation. Also, as it can be easily adapted for other languages, it has the potential to make the MNREAD test even more valuable and usable. More generally, this paper highlights MDD as a convincing alternative for constrained text generation, especially when the constraints are hard to satisfy, but also for many other prospects.

----

## [210] Learning When to Use Automatic Tabulation in Constraint Model Reformulation

**Authors**: *Carlo Cena, Özgür Akgün, Zeynep Kiziltan, Ian Miguel, Peter Nightingale, Felix Ulrich-Oltean*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/211](https://doi.org/10.24963/ijcai.2023/211)

**Abstract**:

Combinatorial optimisation has numerous practical applications, such as planning, logistics, or circuit design. Problems such as these can be solved by approaches such as Boolean Satisfiability (SAT) or Constraint Programming (CP). Solver performance is affected significantly by the model chosen to represent a given problem, which has led to the study of model reformulation. One such method is tabulation: rewriting the expression of some of the model constraints in terms of a single “table” constraint. Successfully applying this process means identifying expressions amenable to trans- formation, which has typically been done manually. Recent work introduced an automatic tabulation using a set of hand-designed heuristics to identify constraints to tabulate. However, the performance of these heuristics varies across problem classes and solvers. Recent work has shown learning techniques to be increasingly useful in the context of automatic model reformulation. The goal of this study is to understand whether it is possible to improve the performance of such heuristics, by learning a model to predict whether or not to activate them for a given instance. Experimental results suggest that a random forest classifier is the most robust choice, improving the performance of four different SAT and CP solvers.

----

## [211] A Fast Algorithm for Consistency Checking Partially Ordered Time

**Authors**: *Leif Eriksson, Victor Lagerkvist*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/212](https://doi.org/10.24963/ijcai.2023/212)

**Abstract**:

Partially ordered models of time occur naturally in applications where agents/processes cannot perfectly communicate with each other, and can be traced back to the seminal work of Lamport. In this paper we consider the problem of deciding if a (likely incomplete) description of a system of events is consistent, the network consistency problem for the point algebra of partially ordered time (POT). While the classical complexity of this problem has been fully settled, comparably little is known of the fine-grained complexity of POT except that it can
be solved in O*((0.368n)^n) time by enumerating ordered partitions. We construct a much faster algorithm with a run-time bounded by O*((0.26n)^n), which, e.g., is roughly 1000 times faster than the naive enumeration algorithm in a problem with 20 events. This is achieved by a sophisticated enumeration of structures similar to total orders, which are then greedily expanded toward a solution. While similar ideas have been explored earlier for related problems it turns out that the analysis for POT is non-trivial and requires significant new ideas.

----

## [212] Improved Algorithms for Allen's Interval Algebra by Dynamic Programming with Sublinear Partitioning

**Authors**: *Leif Eriksson, Victor Lagerkvist*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/213](https://doi.org/10.24963/ijcai.2023/213)

**Abstract**:

Allen's interval algebra is one of the most well-known calculi in qualitative temporal reasoning with numerous applications in artificial intelligence. Very recently, there has been a surge of improvements in the fine-grained complexity of NP-hard reasoning tasks in this algebra, which has improved the running time from the naive 2^O(n^2) to O*((1.0615n)^n), and even faster algorithms are known for unit intervals and the case when we a bounded number of overlapping intervals. 
Despite these improvements the best known lower bound is still only 2^o(n) under the exponential-time hypothesis and major improvements in either direction seemingly require fundamental advances in computational complexity. 
In this paper we propose a novel framework for solving NP-hard qualitative reasoning problems which we refer to as dynamic programming with sublinear partitioning. 
 Using this technique we obtain a major improvement of O*((cn/log(n))^n) for Allen's interval algebra. To demonstrate that the technique is applicable to further problem domains we apply it to a problem in qualitative spatial reasoning, the cardinal direction calculus, and solve it in O*((cn/log(n))^(2n/3)) time. Hence, not only do we significantly advance the state-of-the-art for NP-hard qualitative reasoning problems, but obtain a novel algorithmic technique that is likely applicable to many problems where 2^O(n) time algorithms are unlikely.

----

## [213] New Bounds and Constraint Programming Models for the Weighted Vertex Coloring Problem

**Authors**: *Olivier Goudet, Cyril Grelier, David Lesaint*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/214](https://doi.org/10.24963/ijcai.2023/214)

**Abstract**:

This paper addresses the weighted vertex coloring problem (WVCP) which is an NP-hard variant of the graph coloring problem with various applications.
Given a vertex-weighted graph, the problem consists of partitioning vertices in independent sets (colors) so as to minimize the sum of the maximum weights of the colors.
We first present an iterative procedure to reduce the size of WVCP instances and prove new upper bounds on the objective value and the number of colors.
Alternative constraint programming models are then introduced which rely on primal and dual encodings of the problem and use symmetry breaking constraints.
A large number of experiments are conducted on benchmark instances.
We analyze the impact of using specific bounds to reduce the search space and speed up the exact resolution of instances.
New optimality proofs are reported for some benchmark instances.

----

## [214] Unifying Core-Guided and Implicit Hitting Set Based Optimization

**Authors**: *Hannes Ihalainen, Jeremias Berg, Matti Järvisalo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/215](https://doi.org/10.24963/ijcai.2023/215)

**Abstract**:

Two of the most central algorithmic paradigms implemented in practical solvers for maximum satisfiability (MaxSAT) and other related declarative paradigms for NP-hard combinatorial optimization are the core-guided (CG) and implicit hitting set (IHS) approaches. We develop a general unifying algorithmic framework, based on the recent notion of abstract cores, that captures both CG and IHS computations. The framework offers a unified way of establishing the correctness of variants of the approaches, and can be instantiated in novel ways giving rise to new algorithmic variants of the core-guided and IHS approaches. We illustrate the latter aspect by developing a prototype implementation of an algorithm variant for MaxSAT based on the framework.

----

## [215] Co-Certificate Learning with SAT Modulo Symmetries

**Authors**: *Markus Kirchweger, Tomás Peitl, Stefan Szeider*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/216](https://doi.org/10.24963/ijcai.2023/216)

**Abstract**:

We present a new SAT-based method for generating all graphs up to isomorphism that satisfy a given co-NP property. Our method extends the SAT Modulo Symmetry (SMS) framework with a technique that we call co-certificate learning. If SMS generates a candidate graph that violates the given  co-NP property,
we obtain a certificate for this violation, i.e., `co-certificate' for the co-NP property. The co-certificate gives rise to a clause that the SAT solver, serving as SMS's backend, learns as part of its CDCL procedure. We demonstrate that SMS plus co-certificate learning is a powerful method that allows us to improve the best-known lower bound on the size of Kochen-Specker vector systems, a problem that is central to the foundations of quantum mechanics and has been studied for over half a century. Our approach is orders of magnitude faster and scales significantly better than a recently proposed SAT-based method.

----

## [216] Differentiable Model Selection for Ensemble Learning

**Authors**: *James Kotary, Vincenzo Di Vito, Ferdinando Fioretto*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/217](https://doi.org/10.24963/ijcai.2023/217)

**Abstract**:

Model selection is a strategy aimed at creating accurate and robust models by identifying the optimal model for classifying any particular input sample. This paper proposes a novel framework for differentiable selection of groups of models by integrating machine learning and combinatorial optimization.
The framework is tailored for ensemble learning with a strategy that learns to combine the predictions of appropriately selected pre-trained ensemble models. It does so by modeling the ensemble learning task as a differentiable selection program trained end-to-end over a pretrained ensemble to optimize task performance. The proposed framework demonstrates its versatility and effectiveness, outperforming conventional and advanced consensus rules across a variety of classification tasks.

----

## [217] Backpropagation of Unrolled Solvers with Folded Optimization

**Authors**: *James Kotary, My H. Dinh, Ferdinando Fioretto*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/218](https://doi.org/10.24963/ijcai.2023/218)

**Abstract**:

The integration of constrained optimization models as components in deep networks has led to promising advances on many specialized learning tasks. 
A central challenge in this setting is backpropagation through the solution of an optimization problem, which typically lacks a closed form. One typical strategy is algorithm unrolling, which relies on automatic differentiation through the operations of an iterative solver. While flexible and general, unrolling can encounter accuracy and efficiency issues in practice. These issues can be avoided by analytical differentiation of the optimization, but current frameworks impose rigid requirements on the optimization problem's form. This paper provides theoretical insights into the backward pass of unrolled optimization, leading to a system for generating  efficiently solvable analytical models of backpropagation. Additionally, it proposes a unifying view of unrolling and analytical differentiation through optimization mappings. Experiments over various model-based learning tasks demonstrate the advantages of the approach both computationally and in terms of enhanced expressiveness.

----

## [218] Solving the Identifying Code Set Problem with Grouped Independent Support

**Authors**: *Anna L. D. Latour, Arunabha Sen, Kuldeep S. Meel*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/219](https://doi.org/10.24963/ijcai.2023/219)

**Abstract**:

An important problem in network science is finding an optimal placement of sensors in nodes in order to uniquely detect failures in the network. This problem can be modelled as an identifying code set (ICS) problem, introduced by Karpovsky et al. in 1998. The ICS problem aims to find a cover of a set S, such that the elements in the cover define a unique signature for each of the elements of S, and to minimise the cover’s cardinality. In this work, we study a generalised identifying code set (GICS) problem, where a unique signature must be found for each subset of S that has a cardinality of at most k (instead of just each element of S). The concept of an independent support of a Boolean formula was introduced by Chakraborty et al. in 2014 to speed up propositional model counting, by identifying a subset of variables whose truth assignments uniquely define those of the other variables.

In this work, we introduce an extended version of independent support, grouped independent support (GIS), and show how to reduce the GICS problem to the GIS problem. We then propose a new solving method for finding a GICS, based on finding a GIS. We show that the prior state-of-the-art approaches yield integer-linear programming (ILP) models whose sizes grow exponentially with the problem size and k, while our GIS encoding only grows polynomially with the problem size and k. While the ILP approach can solve the GICS problem on networks of at most 494 nodes, the GIS-based method can handle networks of up to 21 363 nodes; a ∼40× improvement. The GIS-based method shows up to a 520× improvement on the ILP-based method in terms of median solving time. For the majority of the instances that can be encoded and solved by both methods, the cardinality of the solution returned by the GIS-based method is less than 10% larger than the cardinality of the solution found by the ILP method.

----

## [219] A New Variable Ordering for In-processing Bounded Variable Elimination in SAT Solvers

**Authors**: *Shuolin Li, Chu-Min Li, Mao Luo, Jordi Coll, Djamal Habet, Felip Manyà*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/220](https://doi.org/10.24963/ijcai.2023/220)

**Abstract**:

Bounded Variable Elimination (BVE) is an important Boolean formula simplification technique in which the variable ordering is crucial. We define a new variable ordering based on variable activity, called ESA (variable Elimination Scheduled by Activity), for in-processing BVE in Conflict-Driven Clause Learning (CDCL) SAT solvers, and incorporate it into several state-of-the-art CDCL SAT solvers. Experimental results show that the new ESA ordering consistently makes these solvers solve more instances on the benchmark set including all the 5675 instances used in the Crafted, Application and Main tracks of all SAT Competitions up to 2022. In particular, one of these solvers with ESA, Kissat_MAB_ESA, won the Anniversary track of the SAT Competition 2022.  The behaviour of ESA and the reason of its effectiveness are also analyzed.

----

## [220] A Bitwise GAC Algorithm for Alldifferent Constraints

**Authors**: *Zhe Li, Yaohua Wang, Zhanshan Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/221](https://doi.org/10.24963/ijcai.2023/221)

**Abstract**:

The generalized arc consistency (GAC) algorithm is the prevailing solution for alldifferent constraint problems. The core part of GAC for alldifferent constraints is excavating and enumerating all the strongly connected components (SCCs) of the graph model. This causes a large amount of complex data structures to maintain the node information, leading to a large overhead both in time and memory space. More critically, the complexity of the data structures further precludes the coordination of different optimization schemes for GAC. To solve this problem, the key observation of this paper is that the GAC algorithm only cares whether a node of the graph model is in an SCC or not, rather than which SCCs it belongs to. Based on this observation, we propose AllDiffbit, which employs bitwise data structures and operations to efficiently determine if a node is in an SCC. This greatly reduces the corresponding overhead, and enhances the ability to incorporate existing optimizations to work in a synergistic way. Our experiments show that AllDiffbit outperforms the state-of-the-art GAC algorithms over 60%.

----

## [221] Flaws of Termination and Optimality in ADOPT-based Algorithms

**Authors**: *Koji Noshiro, Koji Hasebe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/222](https://doi.org/10.24963/ijcai.2023/222)

**Abstract**:

A distributed constraint optimization problem (DCOP) is a framework to model multi-agent coordination problems. Asynchronous distributed optimization (ADOPT) is a well-known complete DCOP algorithm, and owing to its superior characteristics, many variants have been proposed over the last decade. It is considered proven that ADOPT-based algorithms have the key properties of termination and optimality, which guarantee that the algorithms terminate in a finite time and obtain an optimal solution, respectively. In this paper, we present counterexamples to the termination and optimality of ADOPT-based algorithms. The flaws are classified into three types, at least one of which exists in each of ADOPT and seven of its variants that we analyzed. In other words, the algorithms may potentially not terminate or terminate with a suboptimal solution. We also propose an amended version of ADOPT that avoids the flaws in existing algorithms and prove that it has the properties of termination and optimality.

----

## [222] Fast Algorithms for SAT with Bounded Occurrences of Variables

**Authors**: *Junqiang Peng, Mingyu Xiao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/223](https://doi.org/10.24963/ijcai.2023/223)

**Abstract**:

We present fast algorithms for the general CNF satisfiability problem (SAT) with running-time bound O*({c_d}^n), where c_d is a function of the maximum occurrence d of variables (d can also be the average occurrence when each variable appears at least twice), and n is the number of variables in the input formula. Similar to SAT with bounded clause lengths, SAT with bounded occurrences of variables has also been extensively studied in the literature. Especially, the running-time bounds for small values of d, such as d=3 and d=4, have become bottlenecks for algorithms evaluated by the formula length L and other algorithms. In this paper, we show that SAT can be solved in time O*(1.1238^n) for d=3 and O*(1.2628^n) for d=4, improving the previous results O*(1.1279^n) and O*(1.2721^n) obtained by WahlstrÃ¶m (SAT 2005) nearly 20 years ago. For d>=5, we obtain a running time bound of O*(1.0641^{dn}), implying a bound of O*(1.0641^L) with respect to the formula length L, which is also a slight improvement over the previous bound.

----

## [223] Computing Twin-width with SAT and Branch & Bound

**Authors**: *André Schidler, Stefan Szeider*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/224](https://doi.org/10.24963/ijcai.2023/224)

**Abstract**:

The graph width-measure twin-width recently attracted great attention because of its solving power and generality. Many prominent NP-hard problems are tractable on graphs of bounded twin-width if a certificate for the twin-width bound is provided as an input. Bounded twin-width subsumes other prominent structural restrictions such as bounded treewidth and bounded rank-width.
Computing such a  certificate is NP-hard itself, already for twin-width 4, and the only known implemented algorithm for twin-width computation is based on a SAT encoding.

In this paper, we propose two new algorithmic approaches for computing twin-width that
significantly improve the state of the art.
Firstly, we develop a SAT encoding that is far more compact than the known encoding and consequently scales to larger graphs. Secondly, we propose a new Branch & Bound algorithm for twin-width that, on many graphs, is significantly faster than the SAT encoding. It utilizes a sophisticated caching system for partial solutions.
Both algorithmic approaches are based on new conceptual insights into twin-width computation,
including the reordering of contractions.

----

## [224] Optimal Decision Trees For Interpretable Clustering with Constraints

**Authors**: *Pouya Shati, Eldan Cohen, Sheila A. McIlraith*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/225](https://doi.org/10.24963/ijcai.2023/225)

**Abstract**:

Constrained clustering is a semi-supervised task that employs a limited amount of labelled data, formulated as constraints, to incorporate domain-specific knowledge and to significantly improve clustering accuracy. Previous work has considered exact optimization formulations that can guarantee optimal clustering while satisfying all constraints, however these approaches lack interpretability. Recently, decision trees have been used to produce inherently interpretable clustering solutions, however existing approaches do not support clustering constraints and do not provide strong theoretical guarantees on solution quality. In this work, we present a novel SAT-based framework for interpretable clustering that supports clustering constraints and that also provides strong theoretical guarantees on solution quality. We also present new insight into the trade-off between interpretability and satisfaction of such user-provided constraints. Our framework is the first approach for interpretable and constrained clustering. Experiments with a range of real-world and synthetic datasets demonstrate that our approach can produce high-quality and interpretable constrained clustering solutions.

----

## [225] Engineering an Efficient Approximate DNF-Counter

**Authors**: *Mate Soos, Divesh Aggarwal, Sourav Chakraborty, Kuldeep S. Meel, Maciej Obremski*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/226](https://doi.org/10.24963/ijcai.2023/226)

**Abstract**:

Model counting is a fundamental problem with many practical applications, including query evaluation in probabilistic databases and failure-probability estimation of networks. In this work, we focus on a variant of this problem where the underlying 
 formula is expressed in Disjunctive Normal Form (DNF), also known as #DNF. This problem has been shown to be #P-complete, making it intractable to solve exactly. Much research has therefore been focused on obtaining approximate solutions, particularly in the form of (epsilon, delta) approximations.

The primary contribution of this paper is a new approach, called pepin, to approximate #DNF counting that achieves (nearly) optimal time complexity and outperforms 
existing FPRAS. Our approach is based on the recent breakthrough in the context of union of 
sets in streaming. We demonstrate the effectiveness of our approach through extensive experiments and show that it provides an affirmative answer to the challenge of efficiently computing #DNF.

----

## [226] Solving Quantum-Inspired Perfect Matching Problems via Tutte-Theorem-Based Hybrid Boolean Constraints

**Authors**: *Moshe Y. Vardi, Zhiwei Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/227](https://doi.org/10.24963/ijcai.2023/227)

**Abstract**:

Determining the satisfiability of Boolean constraint-satisfaction problems with different types of constraints, that is hybrid constraints,  is a well-studied problem with important applications. We study a new application of hybrid Boolean constraints, which arises in quantum computing. The problem relates to constrained perfect matching in edge-colored graphs. While general-purpose hybrid constraint solvers can be powerful, we show that direct encodings of the constrained-matching problem as hybrid constraints scale poorly and special techniques are still needed. We propose a novel encoding based on Tutte's Theorem in graph theory as well as optimization techniques. Empirical results demonstrate that our encoding, in suitable languages with advanced SAT solvers, scales significantly better than a number of competing approaches on constrained-matching benchmarks. Our study identifies the necessity of designing problem-specific encodings when applying powerful general-purpose constraint solvers.

----

## [227] Eliminating the Computation of Strongly Connected Components in Generalized Arc Consistency Algorithm for AllDifferent Constraint

**Authors**: *Luhan Zhen, Zhanshan Li, Yanzhi Li, Hongbo Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/228](https://doi.org/10.24963/ijcai.2023/228)

**Abstract**:

AllDifferent constraint is widely used in Constraint Programming to model real world problems. Existing Generalized Arc Consistency (GAC) algorithms map an AllDifferent constraint onto a bipartite graph and utilize the structure of Strongly Connected Components (SCCs) in the graph to filter values. Calculating SCCs is time-consuming in the existing algorithms, so we propose a novel GAC algorithm for AllDifferent constraint in this paper, which eliminates the computation of SCCs. We prove that all redundant edges in the bipartite graph point to some alternating cycles. Our algorithm exploits this property and uses a more efficient method to filter values, which is based on breadth-first search. Experimental results on the XCSP3 benchmark suite show that our algorithm considerably outperforms the state-of-the-art GAC algorithms.

----

## [228] CSGCL: Community-Strength-Enhanced Graph Contrastive Learning

**Authors**: *Han Chen, Ziwen Zhao, Yuhua Li, Yixiong Zou, Ruixuan Li, Rui Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/229](https://doi.org/10.24963/ijcai.2023/229)

**Abstract**:

Graph Contrastive Learning (GCL) is an effective way to learn generalized graph representations in a self-supervised manner, and has grown rapidly in recent years. However, the underlying community semantics has not been well explored by most previous GCL methods. Research that attempts to leverage communities in GCL regards them as having the same influence on the graph, leading to extra representation errors. To tackle this issue, we define ''community strength'' to measure the difference of influence among communities. Under this premise, we propose a Community-Strength-enhanced Graph Contrastive Learning (CSGCL) framework to preserve community strength throughout the learning process. Firstly, we present two novel graph augmentation methods, Communal Attribute Voting (CAV) and Communal Edge Dropping (CED), where the perturbations of node attributes and edges are guided by community strength. Secondly, we propose a dynamic ''Team-up'' contrastive learning scheme, where community strength is used to progressively fine-tune the contrastive objective. We report extensive experiment results on three downstream tasks: node classification, node clustering, and link prediction. CSGCL achieves state-of-the-art performance compared with other GCL methods, validating that community strength brings effectiveness and generality to graph representations. Our code is available at https://github.com/HanChen-HUST/CSGCL.

----

## [229] Probabilistic Masked Attention Networks for Explainable Sequential Recommendation

**Authors**: *Huiyuan Chen, Kaixiong Zhou, Zhimeng Jiang, Chin-Chia Michael Yeh, Xiaoting Li, Menghai Pan, Yan Zheng, Xia Hu, Hao Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/230](https://doi.org/10.24963/ijcai.2023/230)

**Abstract**:

Transformer-based models are powerful for modeling temporal dynamics of user preference in sequential recommendation. Most of the variants adopt the Softmax transformation in the self-attention layers to generate dense attention probabilities. However, real-world item sequences are often noisy, containing a mixture of true-positive and false-positive interactions. Such dense attentions inevitably assign probability mass to noisy or irrelevant items, leading to sub-optimal performance and poor explainability. Here we propose a Probabilistic Masked Attention Network (PMAN) to identify the sparse pattern of attentions, which is more desirable for pruning noisy items in sequential recommendation. Specifically, we employ a probabilistic mask to achieve sparse attentions under a constrained optimization framework. As such, PMAN allows to select which information is critical to be retained or dropped in a data-driven fashion. Experimental studies on real-world benchmark datasets show that PMAN is able to improve the performance of Transformers significantly.

----

## [230] Learning Gaussian Mixture Representations for Tensor Time Series Forecasting

**Authors**: *Jiewen Deng, Jinliang Deng, Renhe Jiang, Xuan Song*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/231](https://doi.org/10.24963/ijcai.2023/231)

**Abstract**:

Tensor time series (TTS) data, a generalization of one-dimensional time series on a high-dimensional space, is ubiquitous in real-world scenarios, especially in monitoring systems involving multi-source spatio-temporal data (e.g., transportation demands and air pollutants). Compared to modeling time series or multivariate time series, which has received much attention and achieved tremendous progress in recent years, tensor time series has been paid less effort. Properly coping with the tensor time series is a much more challenging task, due to its high-dimensional and complex inner structure. In this paper, we develop a novel TTS forecasting framework, which seeks to individually model each heterogeneity component implied in the time, the location, and the source variables. We name this framework as GMRL, short for Gaussian Mixture Representation Learning. Experiment results on two real-world TTS datasets verify the superiority of our approach compared with the state-of-the-art baselines. Code and data are published on https://github.com/beginner-sketch/GMRL.

----

## [231] Adaptive Path-Memory Network for Temporal Knowledge Graph Reasoning

**Authors**: *Hao Dong, Zhiyuan Ning, Pengyang Wang, Ziyue Qiao, Pengfei Wang, Yuanchun Zhou, Yanjie Fu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/232](https://doi.org/10.24963/ijcai.2023/232)

**Abstract**:

Temporal knowledge graph (TKG) reasoning aims to predict the future missing facts based on historical information and has gained increasing research interest recently. Lots of works have been made to model the historical structural and temporal characteristics for the reasoning task. Most existing works model the graph structure mainly depending on entity representation. However, the magnitude of TKG entities in real-world scenarios is considerable, and an increasing number of new entities will arise as time goes on. Therefore, we propose a novel architecture modeling with relation feature of TKG, namely aDAptivE path-MemOry Network (DaeMon), which adaptively models the temporal path information between query subject and each object candidate across history time. It models the historical information without depending on entity representation. Specifically, DaeMon uses path memory to record the temporal path information derived from path aggregation unit across timeline considering the memory passing strategy between adjacent timestamps. Extensive experiments conducted on four real-world TKG datasets demonstrate that our proposed model obtains substantial performance improvement and outperforms the state-of-the-art up to 4.8% absolute in MRR.

----

## [232] Open Anomalous Trajectory Recognition via Probabilistic Metric Learning

**Authors**: *Qiang Gao, Xiaohan Wang, Chaoran Liu, Goce Trajcevski, Li Huang, Fan Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/233](https://doi.org/10.24963/ijcai.2023/233)

**Abstract**:

Typically, trajectories considered anomalous are the ones deviating from usual (e.g., traffic-dictated) driving patterns. However, this closed-set context fails to recognize the unknown anomalous trajectories, resulting in an insufficient self-motivated learning paradigm. In this study, we investigate the novel Anomalous Trajectory Recognition problem in an Open-world scenario (ATRO) and introduce a novel probabilistic Metric learning model, namely ATROM, to address it. Specifically, ATROM can detect the presence of unknown anomalous behavior in addition to identifying known behavior. It has a Mutual Interaction Distillation that uses contrastive metric learning to explore the interactive semantics regarding the diverse behavioral intents and a Probabilistic Trajectory Embedding that forces the trajectories with distinct behaviors to follow different Gaussian priors. More importantly, ATROM offers a probabilistic metric rule to discriminate between known and unknown behavioral patterns by taking advantage of the approximation of multiple priors. Experimental results on two large-scale trajectory datasets demonstrate the superiority of ATROM in addressing both known and unknown anomalous patterns.

----

## [233] Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification

**Authors**: *Zheng Gong, Guifeng Wang, Ying Sun, Qi Liu, Yuting Ning, Hui Xiong, Jingyu Peng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/234](https://doi.org/10.24963/ijcai.2023/234)

**Abstract**:

Recently, graph-based anomaly detection (GAD) has attracted rising attention due to its effectiveness in identifying anomalies in relational and structured data. Unfortunately, the performance of most existing GAD methods suffers from the inherent structural noises of graphs induced by hidden anomalies connected with considerable benign nodes. In this work, we propose SparseGAD, a novel GAD framework that sparsifies the structures of target graphs to effectively reduce noises and collaboratively learns node representations. It then robustly detects anomalies by uncovering the underlying dependency among node pairs in terms of homophily and heterophily, two essential connection properties of GAD. Extensive experiments on real-world datasets of GAD demonstrate that the proposed framework achieves significantly better detection quality compared with the state-of-the-art methods, even when the graph is heavily attacked. Code will be available at https://github.com/KellyGong/SparseGAD.git.

----

## [234] Targeting Minimal Rare Itemsets from Transaction Databases

**Authors**: *Amel Hidouri, Badran Raddaoui, Saïd Jabbour*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/235](https://doi.org/10.24963/ijcai.2023/235)

**Abstract**:

The computation of minimal rare itemsets is a well known task in data mining, with numerous applications, e.g., drugs effects analysis and network security, among others. This paper presents a novel approach to the computation of minimal rare itemsets. First, we introduce a generalization of the traditional minimal rare itemset model called k-minimal rare itemset. A k-minimal rare itemset is defined as an itemset that becomes frequent or rare based on the removal of at least k or at most (k âˆ’ 1) items from it. We claim that our work is the first to propose this generalization in the field of data mining. We then present a SAT-based framework for efficiently discovering k-minimal rare itemsets from large transaction databases. Afterwards, by partitioning the k-minimal rare itemset mining problem into smaller sub-problems, we aim to make it more manageable and easier to solve. Finally, to evaluate the effectiveness and efficiency of our approach, we conduct extensive experimental analysis using various popular datasets. We compare our method with existing specialized algorithms and CP-based algorithms commonly used for this task.

----

## [235] Enhancing Network by Reinforcement Learning and Neural Confined Local Search

**Authors**: *Qifu Hu, Ruyang Li, Qi Deng, Yaqian Zhao, Rengang Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/236](https://doi.org/10.24963/ijcai.2023/236)

**Abstract**:

It has been found that many real networks, such as power grids and the Internet, are non-robust, i.e., attacking a small set of nodes would cause the paralysis of the entire network. Thus, the Network Enhancement Problem~(NEP), i.e., improving the robustness of a given network by modifying its structure, has attracted increasing attention. Heuristics have been proposed to address NEP. However, a hand-engineered heuristic often has significant performance limitations. A recently proposed model solving NEP by reinforcement learning has shown superior performance than heuristics on in-distribution datasets. However, their model shows considerably inferior out-of-distribution generalization ability when enhancing networks against the degree-based targeted attack. In this paper, we propose a more effective model with stronger generalization ability by incorporating domain knowledge including measurements of local network structures and decision criteria of heuristics. We further design a hierarchical attention model to utilize the network structure directly, where the query range changes from local to global. Finally, we propose neural confined local search~(NCLS) to realize the effective search of a large neighborhood, which exploits a learned model to confine the neighborhood to avoid exhaustive enumeration. We conduct extensive experiments on synthetic and real networks to verify the ability of our models.

----

## [236] A Symbolic Approach to Computing Disjunctive Association Rules from Data

**Authors**: *Saïd Jabbour, Badran Raddaoui, Lakhdar Sais*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/237](https://doi.org/10.24963/ijcai.2023/237)

**Abstract**:

Association rule mining is one of the well-studied and most important knowledge discovery task in data mining. In this paper, we first introduce the k-disjunctive support based itemset, a generalization of the traditional model of itemset by allowing the absence of up to k items in each transaction matching the itemset. Then, to discover more expressive rules from data, we define the concept of (k, k′)-disjunctive support based association rules by considering the antecedent and the consequent of the rule as k-disjunctive and k′-disjunctive support based itemsets, respectively. Second, we provide a polynomial-time reduction of both the problems of mining k-disjunctive support based itemsets and (k, k′)-disjunctive support based association rules to the propositional satisfiability model enumeration task. Finally, we show through an extensive campaign of experiments on several popular real-life datasets the efficiency of our proposed approach

----

## [237] OSDP: Optimal Sharded Data Parallel for Distributed Deep Learning

**Authors**: *Youhe Jiang, Fangcheng Fu, Xupeng Miao, Xiaonan Nie, Bin Cui*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/238](https://doi.org/10.24963/ijcai.2023/238)

**Abstract**:

Large-scale deep learning models contribute to significant performance improvements on varieties of downstream tasks. Current data and model parallelism approaches utilize model replication and partition techniques to support the distributed training of ultra-large models. However, directly deploying these systems often leads to sub-optimal training efficiency due to the complex model architectures and the strict device memory constraints. In this paper, we propose Optimal Sharded Data Parallel (OSDP), an automated parallel training system that combines the advantages from both data and model parallelism. Given the model description and the device information, OSDP makes trade-offs between the memory consumption and the hardware utilization, thus automatically generates the distributed computation graph and maximizes the overall system throughput. In addition, OSDP introduces operator splitting to further alleviate peak memory footprints during training with negligible overheads, which enables the trainability of larger models as well as the higher throughput. Extensive experimental results of OSDP on multiple different kinds of large-scale models demonstrate that the proposed strategy outperforms the state-of-the-art in multiple regards.

----

## [238] Hawkes Process Based on Controlled Differential Equations

**Authors**: *Minju Jo, Seungji Kook, Noseong Park*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/239](https://doi.org/10.24963/ijcai.2023/239)

**Abstract**:

Hawkes processes are a popular framework to model the occurrence of sequential events, i.e., occurrence dynamics, in several fields such as social diffusion. In real-world scenarios, the inter-arrival time among events is irregular. However, existing neural network-based Hawkes process models not only i) fail to capture such complicated irregular dynamics, but also ii) resort to heuristics to calculate the log-likelihood of events since they are mostly based on neural networks designed for regular discrete inputs. To this end, we present the concept of Hawkes process based on controlled differential equations (HP-CDE), by adopting the neural controlled differential equation (neural CDE) technology which is an analogue to continuous RNNs. Since HP-CDE continuously reads data, i) irregular time-series datasets can be properly treated preserving their uneven temporal spaces, and ii) the log-likelihood can be exactly computed. Moreover, as both Hawkes processes and neural CDEs are first developed to model complicated human behavioral dynamics, neural CDE-based Hawkes processes are successful in modeling such occurrence dynamics. In our experiments with 4 real-world datasets, our method outperforms existing methods by non-trivial margins.

----

## [239] Computing (1+epsilon)-Approximate Degeneracy in Sublinear Time

**Authors**: *Valerie King, Alex Thomo, Quinton Yong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/240](https://doi.org/10.24963/ijcai.2023/240)

**Abstract**:

The problem of finding the degeneracy of a graph is a subproblem of the k-core decomposition problem. In this paper, we present a (1 + epsilon)-approximate solution to the degeneracy problem which runs in O(n log n) time, sublinear in the input size for dense graphs, by sampling a small number of neighbors adjacent to high degree nodes. This improves upon the previous work on sublinear approximate degeneracy, which implies a (4 + epsilon)-approximate ~O(n) solution. Our algorithm can be extended to an approximate O(n log n) time solution to the k-core decomposition problem. We also explore the use of our approximate algorithm as a technique for speeding up exact degeneracy computation. We prove theoretical guarantees of our algorithm and provide optimizations, which improve the running time of our algorithm in practice. Experiments on massive real-world web graphs show that our algorithm performs significantly faster than previous methods for computing degeneracy.

----

## [240] SMARTformer: Semi-Autoregressive Transformer with Efficient Integrated Window Attention for Long Time Series Forecasting

**Authors**: *Yiduo Li, Shiyi Qi, Zhe Li, Zhongwen Rao, Lujia Pan, Zenglin Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/241](https://doi.org/10.24963/ijcai.2023/241)

**Abstract**:

The success of Transformers in long time series forecasting (LTSF) can be attributed to their attention mechanisms and non-autoregressive (NAR) decoder structures, which capture long-range de- pendencies. However, time series data also contain abundant local temporal dependencies, which are often overlooked in the literature and significantly hinder forecasting performance. To address this issue, we introduce SMARTformer, which stands for SeMi-AutoRegressive Transformer. SMARTformer utilizes the Integrated Window Attention (IWA) and Semi-AutoRegressive (SAR) Decoder to capture global and local dependencies from both encoder and decoder perspectives. IWA conducts local self-attention in multi-scale windows and global attention across windows with linear com- plexity to achieve complementary clues in local and enlarged receptive fields. SAR generates subsequences iteratively, similar to autoregressive (AR) decoding, but refines the entire sequence in a NAR manner. This way, SAR benefits from both the global horizon of NAR and the local detail capturing of AR. We also introduce the Time-Independent Embedding (TIE), which better captures local dependencies by avoiding entanglements of various periods that can occur when directly adding po- sitional embedding to value embedding. Our ex- tensive experiments on five benchmark datasets demonstrate the effectiveness of SMARTformer against state-of-the-art models, achieving an improvement of 10.2% and 18.4% in multivariate and univariate long-term forecasting, respectively.

----

## [241] Do We Need an Encoder-Decoder to Model Dynamical Systems on Networks?

**Authors**: *Bing Liu, Wei Luo, Gang Li, Jing Huang, Bo Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/242](https://doi.org/10.24963/ijcai.2023/242)

**Abstract**:

As deep learning gains popularity in modelling dynamical systems, we expose an underappreciated misunderstanding relevant to modelling dynamics on networks. Strongly influenced by graph neural networks, latent vertex embeddings are naturally adopted in many neural dynamical network models. However, we show that embeddings tend to induce a model that fits observations well but simultaneously has incorrect dynamical behaviours. Recognising that previous studies narrowly focus on short-term predictions during the transient phase of a flow, we propose three tests for correct long-term behaviour, and illustrate how an embedding-based dynamical model fails these tests, and analyse the causes, particularly through the lens of topological conjugacy. In doing so, we show that the difficulties can be avoided by not using embedding. We propose a simple embedding-free alternative based on parametrising two additive vector-field components. Through extensive experiments, we verify that the proposed model can reliably recover a broad class of dynamics on different network topologies from time series data.

----

## [242] Model Conversion via Differentially Private Data-Free Distillation

**Authors**: *Bochao Liu, Pengju Wang, Shikun Li, Dan Zeng, Shiming Ge*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/243](https://doi.org/10.24963/ijcai.2023/243)

**Abstract**:

While massive valuable deep models trained on large-scale data have been released to facilitate the artificial intelligence community, they may encounter attacks in deployment which leads to privacy leakage of training data. In this work, we propose a learning approach termed differentially private data-free distillation (DPDFD) for model conversion that can convert a pretrained model (teacher) into its privacy-preserving counterpart (student) via an intermediate generator without access to training data. The learning collaborates three parties in a unified way. First, massive synthetic data are generated with the generator. Then, they are fed into the teacher and student to compute differentially private gradients by normalizing the gradients and adding noise before performing descent. Finally, the student is updated with these differentially private gradients and the generator is updated by taking the student as a fixed discriminator in an alternate manner. In addition to a privacy-preserving student, the generator can generate synthetic data in a differentially private way for other down-stream tasks. We theoretically prove that our approach can guarantee differential privacy and well convergence. Extensive experiments that significantly outperform other differentially private generative approaches demonstrate the effectiveness of our approach.

----

## [243] Gapformer: Graph Transformer with Graph Pooling for Node Classification

**Authors**: *Chuang Liu, Yibing Zhan, Xueqi Ma, Liang Ding, Dapeng Tao, Jia Wu, Wenbin Hu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/244](https://doi.org/10.24963/ijcai.2023/244)

**Abstract**:

Graph Transformers (GTs) have proved their advantage in graph-level tasks. However, existing GTs still perform unsatisfactorily on the node classification task due to 1) the overwhelming unrelated information obtained from a vast number of irrelevant distant nodes and 2) the quadratic complexity regarding the number of nodes via the fully connected attention mechanism. In this paper, we present Gapformer, a method for node classification that deeply incorporates Graph Transformer with Graph Pooling. More specifically, Gapformer coarsens the large-scale nodes of a graph into a smaller number of pooling nodes via local or global graph pooling methods, and then computes the attention solely with the pooling nodes rather than all other nodes. In such a manner, the negative influence of the overwhelming unrelated nodes is mitigated while maintaining the long-range information, and the quadratic complexity is reduced to linear complexity with respect to the fixed number of pooling nodes. Extensive experiments on 13 node classification datasets, including homophilic and heterophilic graph datasets, demonstrate the competitive performance of Gapformer over existing Graph Neural Networks and GTs.

----

## [244] Federated Probabilistic Preference Distribution Modelling with Compactness Co-Clustering for Privacy-Preserving Multi-Domain Recommendation

**Authors**: *Weiming Liu, Chaochao Chen, Xinting Liao, Mengling Hu, Jianwei Yin, Yanchao Tan, Longfei Zheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/245](https://doi.org/10.24963/ijcai.2023/245)

**Abstract**:

With the development of modern internet techniques, Cross-Domain Recommendation (CDR) systems have been widely exploited for tackling the data-sparsity problem. Meanwhile most current CDR models assume that user-item interactions are accessible across different domains. However, such knowledge sharing process will break the privacy protection policy. In this paper, we focus on the  Privacy-Preserving Multi-Domain Recommendation problem (PPMDR). The problem is challenging since different domains are sparse and heterogeneous with the privacy protection. To tackle the above issues, we propose Federated Probabilistic Preference Distribution Modelling (FPPDM). FPPDM includes two main components, i.e., local domain modelling component and global server aggregation component with federated learning strategy. The local domain modelling component aims to exploit user/item preference distributions using the rating information in the corresponding domain. The global server aggregation component is set to combine user characteristics across domains. To better extract semantic neighbors information among the users, we further provide compactness co-clustering strategy in FPPDM ++ to cluster the users with similar characteristics. Our empirical studies on benchmark datasets demonstrate that FPPDM/ FPPDM ++ significantly outperforms the state-of-the-art models.

----

## [245] Multi-Scale Subgraph Contrastive Learning

**Authors**: *Yanbei Liu, Yu Zhao, Xiao Wang, Lei Geng, Zhitao Xiao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/246](https://doi.org/10.24963/ijcai.2023/246)

**Abstract**:

Graph-level contrastive learning, aiming to learn the representations for each graph by contrasting two augmented graphs, has attracted considerable attention. Previous studies usually simply assume that a graph and its augmented graph as a positive pair, otherwise as a negative pair. However, it is well known that graph structure is always complex and multi-scale, which gives rise to a fundamental question: after graph augmentation, will the previous assumption still hold in reality? By an experimental analysis, we discover the semantic information of an augmented graph structure may be not consistent as original graph structure, and whether two augmented graphs are positive or negative pairs is highly related with the multi-scale structures. Based on this finding, we propose a multi-scale subgraph contrastive learning architecture which is able to characterize the fine-grained semantic information. Specifically, we generate global and local views at different scales based on subgraph sampling, and construct multiple contrastive relationships according to their semantic associations to provide richer self-supervised signals. Extensive experiments and parametric analyzes on eight graph classification real-world datasets well demonstrate the effectiveness of the proposed method.

----

## [246] Continuous-Time Graph Learning for Cascade Popularity Prediction

**Authors**: *Xiaodong Lu, Shuo Ji, Le Yu, Leilei Sun, Bowen Du, Tongyu Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/247](https://doi.org/10.24963/ijcai.2023/247)

**Abstract**:

Information propagation on social networks could be modeled as cascades, and many efforts have been made to predict the future popularity of cascades. However, most of the existing research treats a cascade as an individual sequence. Actually, the cascades might be correlated with each other due to the shared users or similar topics. Moreover, the preferences of users and semantics of a cascade are usually continuously evolving over time. In this paper, we propose a continuous-time graph learning method for cascade popularity prediction, which first connects different cascades via a universal sequence of user-cascade and user-user interactions and then chronologically learns on the sequence by maintaining the dynamic states of users and cascades. Specifically, for each interaction, we present an evolution learning module to continuously update the dynamic states of the related users and cascade based on their currently encoded messages and previous dynamic states. We also devise a cascade representation learning component to embed the temporal information and structural information carried by the cascade. Experiments on real-world datasets demonstrate the superiority and rationality of our approach.

----

## [247] Dynamic Group Link Prediction in Continuous-Time Interaction Network

**Authors**: *Shijie Luo, He Li, Jianbin Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/248](https://doi.org/10.24963/ijcai.2023/248)

**Abstract**:

Recently, group link prediction has received increasing attention due to its important role in analyzing relationships between individuals and groups. However, most existing group link prediction methods emphasize static settings or only make cursory exploitation of historical information, so they fail to obtain good performance in dynamic applications. To this end, we attempt to solve the group link prediction problem in continuous-time dynamic scenes with fine-grained temporal information. We propose a novel continuous-time group link prediction method CTGLP to capture the patterns of future link formation between individuals and groups. A new graph neural network CTGNN is presented to learn the latent representations of individuals by biasedly aggregating neighborhood information. Moreover, we design an importance-based group modeling function to model the embedding of a group based on its known members. CTGLP eventually learns a probability distribution and predicts the link target. Experimental results on various datasets with and without unseen nodes show that CTGLP outperforms the state-of-the-art methods by 13.4% and 13.2% on average.

----

## [248] Capturing the Long-Distance Dependency in the Control Flow Graph via Structural-Guided Attention for Bug Localization

**Authors**: *Yi-Fan Ma, Yali Du, Ming Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/249](https://doi.org/10.24963/ijcai.2023/249)

**Abstract**:

To alleviate the burden of software maintenance, bug localization, which aims to automatically locate the buggy source files based on the bug report, has drawn significant attention in the software mining community. Recent studies indicate that the program structure in source code carries more semantics reflecting the program behavior, which is beneficial for bug localization. Benefiting from the rich structural information in the Control Flow Graph (CFG), CFG-based bug localization methods have achieved the state-of-the-art performance. Existing CFG-based methods extract the semantic feature from the CFG via the graph neural network. However, the step-wise feature propagation in the graph neural network suffers from the problem of information loss when the propagation distance is long, while the long-distance dependency is rather common in the CFG. In this paper, we argue that the long-distance dependency is crucial for feature extraction from the CFG, and propose a novel bug localization model named sgAttention. In sgAttention, a particularly designed structural-guided attention is employed to globally capture the information in the CFG, where features of irrelevant nodes are masked for each node to facilitate better feature extraction from the CFG. Experimental results on four widely-used open-source software projects indicate that sgAttention averagely improves the state-of-the-art bug localization methods by 32.9\% and 29.2\% and the state-of-the-art pre-trained models by 5.8\%  and 4.9\% in terms of MAP and MRR, respectively.

----

## [249] Uncovering the Largest Community in Social Networks at Scale

**Authors**: *Shohei Matsugu, Yasuhiro Fujiwara, Hiroaki Shiokawa*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/250](https://doi.org/10.24963/ijcai.2023/250)

**Abstract**:

The Maximum k-Plex Search (MPS) can find the largest k-plex, which is a generalization of the largest clique.
Although MPS is commonly used in AI to effectively discover real-world communities of social networks, existing MPS algorithms suffer from high computational costs because they iteratively scan numerous nodes to find the largest k-plex.
Here, we present an efficient MPS algorithm called Branch-and-Merge (BnM), which outputs an exact maximum k-plex.
BnM merges unnecessary nodes to explore a smaller graph than the original one.
Extensive evaluations on real-world social networks demonstrate that BnM significantly outperforms other state-of-the-art MPS algorithms in terms of running time.

----

## [250] Reinforcement Learning Approaches for Traffic Signal Control under Missing Data

**Authors**: *Hao Mei, Junxian Li, Bin Shi, Hua Wei*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/251](https://doi.org/10.24963/ijcai.2023/251)

**Abstract**:

The emergence of reinforcement learning (RL) methods in traffic signal control (TSC) tasks has achieved promising results. Most RL approaches require the observation of the environment for the agent to decide which action is optimal for a long-term reward. However, in real-world urban scenarios, missing observation of traffic states may frequently occur due to the lack of sensors, which makes existing RL methods inapplicable on road networks with missing observation. In this work, we aim to control the traffic signals in a real-world setting, where some of the intersections in the road network are not installed with sensors and thus with no direct observations around them. To the best of our knowledge, we are the first to use RL methods to tackle the TSC problem in this real-world setting. Specifically, we propose two solutions: 1) imputes the traffic states to enable adaptive control. 2) imputes both states and rewards to enable adaptive control and the training of RL agents. Through extensive experiments on both synthetic and real-world road network traffic, we reveal that our method outperforms conventional approaches and performs consistently with different missing rates. We also investigate how missing data influences the performance of our model.

----

## [251] Discriminative-Invariant Representation Learning for Unbiased Recommendation

**Authors**: *Hang Pan, Jiawei Chen, Fuli Feng, Wentao Shi, Junkang Wu, Xiangnan He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/252](https://doi.org/10.24963/ijcai.2023/252)

**Abstract**:

Selection bias hinders recommendation models from learning unbiased user preference. Recent works empirically reveal that pursuing invariant user and item representation across biased and unbiased data is crucial for counteracting selection bias. However, our theoretical analysis reveals that simply optimizing representation invariance is insufficient for addressing the selection bias â€” recommendation performance is bounded by both representation invariance and discriminability. Worse still, current invariant representation learning methods in recommendation neglect even hurt the representation discriminability due to data sparsity and label shift. In this light, we propose a new Discriminative-Invariant Representation Learning framework for unbiased recommendation, which incorporates label-conditional clustering and prior-guided contrasting into conventional invariant representation learning to mitigate the impact of data sparsity and label shift, respectively. We conduct extensive experiments on three real-world datasets, validating the rationality and effectiveness of the proposed framework. Code and supplementary materials are available at: https://github.com/HungPaan/DIRL.

----

## [252] Semi-supervised Domain Adaptation in Graph Transfer Learning

**Authors**: *Ziyue Qiao, Xiao Luo, Meng Xiao, Hao Dong, Yuanchun Zhou, Hui Xiong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/253](https://doi.org/10.24963/ijcai.2023/253)

**Abstract**:

As a specific case of graph transfer learning, unsupervised domain adaptation on graphs aims for knowledge transfer from label-rich source graphs to unlabeled target graphs. However, graphs with topology and attributes usually have considerable cross-domain disparity and there are numerous real-world scenarios where merely a subset of nodes are labeled in the source graph. This imposes critical challenges on graph transfer learning due to serious domain shifts and label scarcity. To address these challenges, we propose a method named Semi-supervised Graph Domain Adaptation (SGDA). To deal with the domain shift, we add adaptive shift parameters to each of the source nodes, which are trained in an adversarial manner to align the cross-domain distributions of node embedding. Thus, the node classifier trained on labeled source nodes can be transferred to the target nodes. Moreover, to address the label scarcity, we propose pseudo-labeling on unlabeled nodes, which improves classification on the target graph via measuring the posterior influence of nodes based on their relative position to the class centroids. Finally, extensive experiments on a range of publicly accessible datasets validate the effectiveness of our proposed SGDA in different experimental settings.

----

## [253] Self-supervised Graph Disentangled Networks for Review-based Recommendation

**Authors**: *Yuyang Ren, Haonan Zhang, Qi Li, Luoyi Fu, Xinbing Wang, Chenghu Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/254](https://doi.org/10.24963/ijcai.2023/254)

**Abstract**:

User review data is considered as auxiliary information to alleviate the data sparsity problem and improve the quality of learned user/item or interaction representations in review-based recommender systems. However, existing methods usually model user-item interactions in a holistic manner and neglect the entanglement of the latent intents behind them, e.g., price, quality, or appearance, resulting in suboptimal representations and reducing interpretability. In this paper, we propose a Self-supervised Graph Disentangled Networks for review-based recommendation (SGDN), to separately model the user-item interactions based on the latent factors through the textual review data. To this end, we first model the distributions of interactions over latent factors from both semantic information in review data and structural information in user-item graph data, forming several factor graphs. Then a factorized message passing mechanism is designed to learn disentangled user/item and interaction representations on the factor graphs. Finally, we set an intent-aware contrastive learning task to alleviate the sparsity issue and encourage disentanglement through dynamically identifying positive and negative samples based on the learned intent distributions. Empirical results over five benchmark datasets validate the superiority of  SGDN over the state-of-the-art methods and the  interpretability of learned intent factors.

----

## [254] CONGREGATE: Contrastive Graph Clustering in Curvature Spaces

**Authors**: *Li Sun, Feiyang Wang, Junda Ye, Hao Peng, Philip S. Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/255](https://doi.org/10.24963/ijcai.2023/255)

**Abstract**:

Graph clustering is a longstanding research topic, and has achieved remarkable success with the deep learning methods in recent years. Nevertheless, we observe that several important issues largely remain open. On the one hand, graph clustering from the geometric perspective is appealing but has rarely been touched before, as it lacks a promising space for geometric clustering. On the other hand, contrastive learning boosts the deep graph clustering but usually struggles in either graph augmentation or hard sample mining. To bridge this gap, we rethink the problem of graph clustering from geometric perspective and, to the best of our knowledge, make the first attempt to introduce a heterogeneous curvature space to graph clustering problem. Correspondingly, we present a novel end-to-end contrastive graph clustering model named CONGREGATE, addressing geometric graph clustering with Ricci curvatures. To support geometric clustering, we construct a theoretically grounded Heterogeneous Curvature Space where deep representations are generated via the product of the proposed fully Riemannian graph convolutional nets. Thereafter, we train the graph clusters by an augmentation-free reweighted contrastive approach where we pay more attention to both hard negatives and hard positives in our curvature space. Empirical results on real-world graphs show that our model outperforms the state-of-the-art competitors.

----

## [255] SAD: Semi-Supervised Anomaly Detection on Dynamic Graphs

**Authors**: *Sheng Tian, Jihai Dong, Jintang Li, Wenlong Zhao, Xiaolong Xu, Baokun Wang, Bowen Song, Changhua Meng, Tianyi Zhang, Liang Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/256](https://doi.org/10.24963/ijcai.2023/256)

**Abstract**:

Anomaly detection aims to distinguish abnormal instances that deviate significantly from the majority of benign ones. As instances that appear in the real world are naturally connected and can be represented with graphs, graph neural networks become increasingly popular in tackling the anomaly detection problem. Despite the promising results, research on anomaly detection has almost exclusively focused on static graphs while the mining of anomalous patterns from dynamic graphs is rarely studied but has significant application value. In addition, anomaly detection is typically tackled from semi-supervised perspectives due to the lack of sufficient labeled data. However, most proposed methods are limited to merely exploiting labeled data, leaving a large number of unlabeled samples unexplored. In this work, we present semi-supervised anomaly detection (SAD), an end-to-end framework for anomaly detection on dynamic graphs. By a combination of a time-equipped memory bank and a pseudo-label contrastive learning module, SAD is able to fully exploit the potential of large unlabeled samples and uncover underlying anomalies on evolving graph streams. Extensive experiments on four real-world datasets demonstrate that SAD efficiently discovers anomalies from dynamic graphs and outperforms existing advanced methods even when provided with only little labeled data.

----

## [256] Causal-Based Supervision of Attention in Graph Neural Network: A Better and Simpler Choice towards Powerful Attention

**Authors**: *Hongjun Wang, Jiyuan Chen, Lun Du, Qiang Fu, Shi Han, Xuan Song*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/257](https://doi.org/10.24963/ijcai.2023/257)

**Abstract**:

Recent years have witnessed the great potential of attention mechanism in graph representation learning. However, while variants of attention-based GNNs are setting new benchmarks for numerous real-world datasets, recent works have pointed out that their induced attentions are less robust and generalizable against noisy graphs due to lack of direct supervision. In this paper, we present a new framework which utilizes the tool of causality to provide a powerful supervision signal for the learning process of attention functions. Specifically, we estimate the direct causal effect of attention to the final prediction, and then maximize such effect to guide attention attending to more meaningful neighbors. Our method can serve as a plug-and-play module for any canonical attention-based GNNs in an end-to-end fashion. Extensive experiments on a wide range of benchmark datasets illustrated that, by directly supervising attention functions, the model is able to converge faster with a clearer decision boundary, and thus yields better performances.

----

## [257] Denoised Self-Augmented Learning for Social Recommendation

**Authors**: *Tianle Wang, Lianghao Xia, Chao Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/258](https://doi.org/10.24963/ijcai.2023/258)

**Abstract**:

Social recommendation is gaining increasing attention in various online applications, including e-commerce and online streaming, where social information is leveraged to improve user-item interaction modeling. Recently, Self-Supervised Learning (SSL) has proven to be remarkably effective in addressing data sparsity through augmented learning tasks. Inspired by this, researchers have attempted to incorporate SSL into social recommendation by supplementing the primary supervised task with social-aware self-supervised signals. However, social information can be unavoidably noisy in characterizing user preferences due to the ubiquitous presence of interest-irrelevant social connections, such as colleagues or classmates who do not share many common interests. To address this challenge, we propose a novel social recommender called the Denoised Self-Augmented Learning paradigm (DSL). Our model not only preserves helpful social relations to enhance user-item interaction modeling but also enables personalized cross-view knowledge transfer through adaptive semantic alignment in embedding space. Our experimental results on various recommendation benchmarks confirm the superiority of our DSL over state-of-the-art methods. We release our model implementation at: https://github.com/HKUDS/DSL.

----

## [258] A Canonicalization-Enhanced Known Fact-Aware Framework For Open Knowledge Graph Link Prediction

**Authors**: *Yilin Wang, Minghao Hu, Zhen Huang, Dongsheng Li, Wei Luo, Dong Yang, Xicheng Lu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/259](https://doi.org/10.24963/ijcai.2023/259)

**Abstract**:

Open knowledge graph (OpenKG) link prediction aims to predict missing factual triples in the form of (head noun phrase, relation phrase, tail noun phrase). Since triples are not canonicalized, previous methods either focus on canonicalizing noun phrases (NPs) to reduce graph sparsity, or utilize textual forms to improve type compatibility. However, they neglect to canonicalize relation phrases (RPs) and triples, making OpenKG maintain high sparsity and impeding the performance. To address the above issues, we propose a Canonicalization-Enhanced Known Fact-Aware (CEKFA) framework that boosts link prediction performance through sparsity reduction of RPs and triples. First, we propose a similarity-driven RP canonicalization method to reduce RPs' sparsity by sharing knowledge of semantically similar ones. Second, to reduce the sparsity of triples, a known fact-aware triple canonicalization method is designed to retrieve relevant known facts from training data. Finally, these two types of canonical information are integrated into a general two-stage re-ranking framework that can be applied to most existing knowledge graph embedding methods. Experiment results on two OpenKG datasets, ReVerb20K and ReVerb45K, show that our approach achieves state-of-the-art results. Extensive experimental analyses illustrate the effectiveness and generalization ability of the proposed framework.

----

## [259] Intent-aware Recommendation via Disentangled Graph Contrastive Learning

**Authors**: *Yuling Wang, Xiao Wang, Xiangzhou Huang, Yanhua Yu, Haoyang Li, Mengdi Zhang, Zirui Guo, Wei Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/260](https://doi.org/10.24963/ijcai.2023/260)

**Abstract**:

Graph neural network (GNN) based recommender systems have become one of the mainstream trends due to the powerful learning ability from user behavior data. Understanding the user intents from behavior data is the key to recommender systems, which poses two basic requirements for GNN-based recommender systems. One is how to learn complex and diverse intents especially when the user behavior is usually inadequate in reality. The other is different behaviors have different intent distributions, so how to establish their relations for a more explainable  recommender system. In this paper, we present the Intent-aware Recommendation via Disentangled Graph Contrastive Learning (IDCL), which simultaneously learns interpretable intents and behavior distributions over those intents. Specifically, we first model the user behavior data as a user-item-concept graph, and design a GNN based behavior disentangling module to learn the different intents. Then we propose the intent-wise contrastive learning to enhance the intent disentangling and meanwhile infer the behavior distributions. Finally, the coding rate reduction regularization is introduced to make the behaviors of different intents orthogonal. Extensive experiments demonstrate the effectiveness of IDCL in terms of substantial improvement and the interpretability.

----

## [260] Feature Staleness Aware Incremental Learning for CTR Prediction

**Authors**: *Zhikai Wang, Yanyan Shen, Zibin Zhang, Kangyi Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/261](https://doi.org/10.24963/ijcai.2023/261)

**Abstract**:

Click-through Rate (CTR) prediction in real-world recommender systems often deals with billions of user interactions every day. To improve the training efficiency, it is common to update the CTR prediction model incrementally using the new incremental data and a subset of historical data. However, the feature embeddings of a CTR prediction model often get stale when the corresponding features do not appear in current incremental data. In the next period, the model would have a performance degradation on samples containing stale features, which we call the feature staleness problem. To mitigate this problem, we propose a Feature Staleness Aware Incremental Learning method for CTR prediction (FeSAIL) which adaptively replays samples containing stale features. We first introduce a staleness-aware sampling algorithm (SAS) to sample a fixed number of stale samples with high sampling efficiency. We then introduce a staleness-aware regularization mechanism (SAR) for a fine-grained control of the feature embedding updating. We instantiate FeSAIL with a general deep learning-based CTR prediction model and the experimental results demonstrate FeSAIL outperforms various state-of-the-art methods on four benchmark datasets. The code can be found in https://github.com/cloudcatcher888/FeSAIL.

----

## [261] KMF: Knowledge-Aware Multi-Faceted Representation Learning for Zero-Shot Node Classification

**Authors**: *Likang Wu, Junji Jiang, Hongke Zhao, Hao Wang, Defu Lian, Mengdi Zhang, Enhong Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/262](https://doi.org/10.24963/ijcai.2023/262)

**Abstract**:

Recently, Zero-Shot Node Classification (ZNC) has been an emerging and crucial task in graph data analysis. This task aims to predict nodes from unseen classes which are unobserved in the training process. Existing work mainly utilizes Graph Neural Networks (GNNs) to associate features' prototypes and labels' semantics thus enabling knowledge transfer from seen to unseen classes. However, the multi-faceted semantic orientation in the feature-semantic alignment has been neglected by previous work, i.e. the content of a node usually covers diverse topics that are relevant to the semantics of multiple labels. It's necessary to separate and judge the semantic factors that tremendously affect the cognitive ability to improve the generality of models. To this end, we propose a Knowledge-Aware Multi-Faceted framework (KMF) that enhances the richness of label semantics via the extracted KG (Knowledge Graph)-based topics. And then the content of each node is reconstructed to a topic-level representation that offers multi-faceted and fine-grained semantic relevancy to different labels. Due to the particularity of the graph's instance (i.e., node) representation, a novel geometric constraint is developed to alleviate the problem of prototype drift caused by node information aggregation. Finally, we conduct extensive experiments on several public graph datasets and design an application of zero-shot cross-domain recommendation. The quantitative results demonstrate both the effectiveness and generalization of KMF with the comparison of state-of-the-art baselines.

----

## [262] KDLGT: A Linear Graph Transformer Framework via Kernel Decomposition Approach

**Authors**: *Yi Wu, Yanyang Xu, Wenhao Zhu, Guojie Song, Zhouchen Lin, Liang Wang, Shaoguo Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/263](https://doi.org/10.24963/ijcai.2023/263)

**Abstract**:

In recent years, graph Transformers (GTs) have been demonstrated as a robust architecture for a wide range of graph learning tasks. However, the quadratic complexity of GTs limits their scalability on large-scale data, in comparison to Graph Neural Networks (GNNs). In this work, we propose the Kernel Decomposition Linear Graph Transformer (KDLGT), an accelerating framework for building scalable and powerful GTs. KDLGT employs the kernel decomposition approach to rearrange the order of matrix multiplication, thereby reducing complexity to linear. Additionally, it categorizes GTs into three distinct types and provides tailored accelerating methods for each category to encompass all types of GTs. Furthermore, we provide a theoretical analysis of the performance gap between KDLGT and self-attention to ensure its effectiveness. Under this framework, we select two representative GTs to design our models. Experiments on both real-world and synthetic datasets indicate that KDLGT not only achieves state-of-the-art performance on various datasets but also reaches an acceleration ratio of approximately 10 on graphs of certain sizes.

----

## [263] OptIForest: Optimal Isolation Forest for Anomaly Detection

**Authors**: *Haolong Xiang, Xuyun Zhang, Hongsheng Hu, Lianyong Qi, Wanchun Dou, Mark Dras, Amin Beheshti, Xiaolong Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/264](https://doi.org/10.24963/ijcai.2023/264)

**Abstract**:

Anomaly detection plays an increasingly important role in various fields for critical tasks such as intrusion detection in cybersecurity, financial risk detection, and human health monitoring. A variety of anomaly detection methods have been proposed, and a category based on the isolation forest mechanism stands out due to its simplicity, effectiveness, and efficiency, e.g., iForest is often employed as a state-of-the-art detector for real deployment. While the majority of isolation forests use the binary structure, a framework LSHiForest has demonstrated that the multi-fork isolation tree structure can lead to better detection performance. However, there is no theoretical work answering the fundamentally and practically important question on the optimal tree structure for an isolation forest with respect to the branching factor. In this paper, we establish a theory on isolation efficiency to answer the question and determine the optimal branching factor for an isolation tree. Based on the theoretical underpinning, we design a practical optimal isolation forest OptIForest incorporating clustering based learning to hash which enables more information to be learned from data for better isolation quality. The rationale of our approach relies on a better bias-variance trade-off achieved by bias reduction in OptIForest. Extensive experiments on a series of benchmarking datasets for comparative and ablation studies demonstrate that our approach can efficiently and robustly achieve better detection performance in general than the state-of-the-arts including the deep learning based methods.

----

## [264] Hierarchical Apprenticeship Learning for Disease Progression Modeling

**Authors**: *Xi Yang, Ge Gao, Min Chi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/265](https://doi.org/10.24963/ijcai.2023/265)

**Abstract**:

Disease progression modeling (DPM) plays an essential role in characterizing patients' historical pathways and predicting their future risks. Apprenticeship learning (AL) aims to induce decision-making policies by observing and imitating expert behaviors. In this paper, we investigate the incorporation of AL-derived patterns into DPM, utilizing a Time-aware Hierarchical EM Energy-based Subsequence (THEMES) AL approach. To the best of our knowledge, this is the first study incorporating AL-derived progressive and interventional patterns for DPM. We evaluate the efficacy of this approach in a challenging task of septic shock early prediction, and our results demonstrate that integrating the AL-derived patterns significantly enhances the performance of DPM.

----

## [265] Exploiting Non-Interactive Exercises in Cognitive Diagnosis

**Authors**: *Fangzhou Yao, Qi Liu, Min Hou, Shiwei Tong, Zhenya Huang, Enhong Chen, Jing Sha, Shijin Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/266](https://doi.org/10.24963/ijcai.2023/266)

**Abstract**:

Cognitive Diagnosis aims to quantify the proficiency level of students on specific knowledge concepts. Existing studies merely leverage observed historical students-exercise interaction logs to access proficiency levels. Despite effectiveness, observed interactions usually exhibit a power-law distribution, where the long tail consisting of students with few records lacks supervision signals. This phenomenon leads to inferior diagnosis among few records students. In this paper, we propose the Exercise-aware Informative Response Sampling (EIRS) framework to address the long-tail problem. EIRS is a general framework that explores the partial order between observed and unobserved responses as auxiliary ranking-based training signals to supplement cognitive diagnosis. Considering the abundance and complexity of unobserved responses, we first design an Exercise-aware Candidates Selection module, which helps our framework produce reliable potential responses for effective supplementary training. Then, we develop an Expected Ability Change-weighted Informative Sampling strategy to adaptively sample informative potential responses that contribute greatly to model training. Experiments on real-world datasets demonstrate the supremacy of our framework in long-tailed data.

----

## [266] Curriculum Multi-Level Learning for Imbalanced Live-Stream Recommendation

**Authors**: *Shuodian Yu, Junqi Jin, Li Ma, Xiaofeng Gao, Xiaopeng Wu, Haiyang Xu, Jian Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/267](https://doi.org/10.24963/ijcai.2023/267)

**Abstract**:

In large-scale e-commerce live-stream recommendation, streamers are classified into different levels based on their popularity and other metrics for marketing. Several top streamers at the head level occupy a considerable amount of exposure, resulting in an unbalanced data distribution. A unified model for all levels without consideration of imbalance issue can be biased towards head streamers and neglect the conflicts between levels. The lack of inter-level streamer correlations and intra-level streamer characteristics modeling imposes obstacles to estimating the user behaviors. To tackle these challenges, we propose a curriculum multi-level learning framework for imbalanced recommendation. We separate model parameters into shared and level-specific ones to explore the generality among all levels and discrepancy for each level respectively. The level-aware gradient descent and a curriculum sampling scheduler are designed to capture the de-biased commonalities from all levels as the shared parameters. During the specific parameters training, the hardness-aware learning rate and an adaptor are proposed to dynamically balance the training process. Finally, shared and specific parameters are combined to be the final model weights and learned in a cooperative training framework. Extensive experiments on a live-stream production dataset demonstrate the superiority of the proposed framework.

----

## [267] Basket Representation Learning by Hypergraph Convolution on Repeated Items for Next-basket Recommendation

**Authors**: *Yalin Yu, Enneng Yang, Guibing Guo, Linying Jiang, Xingwei Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/268](https://doi.org/10.24963/ijcai.2023/268)

**Abstract**:

Basket representation plays an important role in the task of next-basket recommendation. However, existing methods generally adopts pooling operations to learn a basket's representation, from which two critical issues can be identified. 
First, they treat a basket as a set of items independent and identically distributed. We find that items occurring in the same basket have much higher correlations than those randomly selected by conducting data analysis on a real dataset. 
Second, although some works have recognized the importance of items repeatedly purchased in multiple baskets, they ignore the correlations among the repeated items in a same basket, whose importance is  shown by our data analysis. In this paper, we propose a novel Basket Representation Learning (BRL) model by leveraging the correlations among intra-basket items. Specifically, we first connect all the items (in a basket) as a hyperedge, where the correlations among different items can be well exploited by hypergraph convolution operations. Meanwhile, we also connect all the repeated items in the same basket as a hyperedge, whereby their correlations can be further strengthened. We generate a negative (positive) view of the basket by data augmentation on repeated (non-repeated) items, and apply contrastive learning to force more agreements on repeated items. Finally, experimental results on three real datasets show that our approach performs better than eight baselines in ranking accuracy.

----

## [268] Commonsense Knowledge Enhanced Sentiment Dependency Graph for Sarcasm Detection

**Authors**: *Zhe Yu, Di Jin, Xiaobao Wang, Yawen Li, Longbiao Wang, Jianwu Dang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/269](https://doi.org/10.24963/ijcai.2023/269)

**Abstract**:

Sarcasm is widely utilized on social media platforms such as Twitter and Reddit. Sarcasm detection is required for analyzing people's true feelings since sarcasm is commonly used to portray a reversed emotion opposing the literal meaning. The syntactic structure is the key to make better use of commonsense when detecting sarcasm. However, it is extremely challenging to effectively and explicitly explore the information implied in syntactic structure and commonsense simultaneously. In this paper, we apply the pre-trained COMET model to generate relevant commonsense knowledge, and explore a novel scenario of constructing a commonsense-augmented sentiment graph and a commonsense-replaced dependency graph for each text. Based on this, a Commonsense Sentiment Dependency Graph Convolutional Network (CSDGCN) framework is proposed to explicitly depict the role of external commonsense and inconsistent expressions over the context for sarcasm detection by interactively modeling the sentiment and dependency information. Experimental results on several benchmark datasets reveal that our proposed method beats the state-of-the-art methods in sarcasm detection, and has a stronger interpretability.

----

## [269] Sequential Recommendation with Probabilistic Logical Reasoning

**Authors**: *Huanhuan Yuan, Pengpeng Zhao, Xuefeng Xian, Guanfeng Liu, Yanchi Liu, Victor S. Sheng, Lei Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/270](https://doi.org/10.24963/ijcai.2023/270)

**Abstract**:

Deep learning and symbolic learning are two frequently employed methods in Sequential Recommendation (SR). Recent neural-symbolic SR models demonstrate their potential to enable SR to be equipped with concurrent perception and cognition capacities. However, neural-symbolic SR remains a challenging problem due to open issues like representing users and items in logical reasoning. In this paper, we combine the Deep Neural Network (DNN) SR models with logical reasoning and propose a general framework named Sequential Recommendation with Probabilistic Logical Reasoning (short for SR-PLR). This framework allows SR-PLR to benefit from both similarity matching and logical reasoning by disentangling feature embedding and logic embedding in the DNN and probabilistic logic network. To better capture the uncertainty and evolution of user tastes, SR-PLR embeds users and items with a probabilistic method and conducts probabilistic logical reasoning on users' interaction patterns.  Then the feature and logic representations learned from the DNN and logic network are concatenated to make the prediction. Finally, experiments on various sequential recommendation models demonstrate the effectiveness of the SR-PLR. Our code is available at https://github.com/Huanhuaneryuan/SR-PLR.

----

## [270] Towards an Integrated View of Semantic Annotation for POIs with Spatial and Textual Information

**Authors**: *Dabin Zhang, Ronghui Xu, Weiming Huang, Kai Zhao, Meng Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/271](https://doi.org/10.24963/ijcai.2023/271)

**Abstract**:

Categories of Point of Interest (POI) facilitate location-based services from many aspects like location search and POI recommendation. However, POI categories are often incomplete and new POIs are being consistently generated, this rises the demand for semantic annotation for POIs, i.e., labeling the POI with a semantic category. Previous methods usually model sequential check-in information of users to learn POI features for annotation. However, users' check-ins are hardly obtained in reality, especially for those newly created POIs. In this context, we present a Spatial-Textual POI Annotation (STPA) model for static POIs, which derives POI categories using only the geographic locations and names of POIs. Specifically, we design a GCN-based spatial encoder to model spatial correlations among POIs to generate POI spatial embeddings, and an attention-based text encoder to model the semantic contexts of POIs to generate POI textual embeddings. We finally fuse the two embeddings and preserve multi-view correlations for semantic annotation. We conduct comprehensive experiments to validate the effectiveness of STPA with POI data from AMap. Experimental results demonstrate that STPA substantially outperforms several competitive baselines, which proves that STPA is a promising approach for annotating static POIs in map services.

----

## [271] Minimally Supervised Contextual Inference from Human Mobility: An Iterative Collaborative Distillation Framework

**Authors**: *Jiayun Zhang, Xinyang Zhang, Dezhi Hong, Rajesh K. Gupta, Jingbo Shang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/272](https://doi.org/10.24963/ijcai.2023/272)

**Abstract**:

The context about trips and users from mobility data is valuable for mobile service providers to understand their customers and improve their services. Existing inference methods require a large number of labels for training, which is hard to meet in practice. In this paper, we study a more practical yet challenging settingâ€”contextual inference using mobility data with minimal supervision (i.e., a few labels per class and massive unlabeled data). A typical solution is to apply semi-supervised methods that follow a self-training framework to bootstrap a model based on all features. However, using a limited labeled set brings high risk of overfitting to self-training, leading to unsatisfactory performance. We propose a novel collaborative distillation framework STCOLAB. It sequentially trains spatial and temporal modules at each iteration following the supervision of ground-truth labels. In addition, it distills knowledge to the module being trained using the logits produced by the latest trained module of the other modality, thereby mutually calibrating the two modules and combining the knowledge from both modalities. Extensive experiments on two real-world datasets show STCOLAB achieves significantly more accurate contextual inference than various baselines.

----

## [272] Towards Hierarchical Policy Learning for Conversational Recommendation with Hypergraph-based Reinforcement Learning

**Authors**: *Sen Zhao, Wei Wei, Yifan Liu, Ziyang Wang, Wendi Li, Xian-Ling Mao, Shuai Zhu, Minghui Yang, Zujie Wen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/273](https://doi.org/10.24963/ijcai.2023/273)

**Abstract**:

Conversational recommendation systems (CRS) aim to timely and proactively acquire user dynamic preferred attributes through conversations for item recommendation. In each turn of CRS, there naturally have two decision-making processes with different roles that influence each other: 1) director, which is to select the follow-up option (i.e., ask or recommend) that is more effective for reducing the action space and acquiring user preferences; and 2) actor, which is to accordingly choose primitive actions (i.e., asked attribute or recommended item) to estimate the effectiveness of the director’s option. However, existing methods heavily rely on a unified decision-making module or heuristic rules, while neglecting to distinguish the roles of different decision procedures, as well as the mutual influences between them. To address this, we propose a novel Director-Actor Hierarchical Conversational Recommender (DAHCR), where the director selects the most effective option, followed by the actor accordingly choosing primitive actions that satisfy user preferences. Specifically, we develop a dynamic hypergraph to model user preferences and introduce an intrinsic motivation to train from weak supervision over the director. Finally, to alleviate the bad effect of model bias on the mutual influence between the director and actor, we model the director’s option by sampling from a categorical distribution. Extensive experiments demonstrate that DAHCR outperforms state-of-the-art methods.

----

## [273] Online Harmonizing Gradient Descent for Imbalanced Data Streams One-Pass Classification

**Authors**: *Han Zhou, Hongpeng Yin, Xuanhong Deng, Yuyu Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/274](https://doi.org/10.24963/ijcai.2023/274)

**Abstract**:

Many real-world streaming data are sequentially collected over time and with skew-distributed classes. In this situation, online learning models may tend to favor samples from majority classes, making the wrong decisions for those from minority classes. Previous methods try to balance the instance number of different classes or assign asymmetric cost values. They usually require data-buffers to store streaming data or pre-defined cost parameters. This study alternatively shows that the imbalance of instances can be implied by the imbalance of gradients. Then, we propose the Online Harmonizing Gradient Descent (OHGD) for one-pass online classification. By harmonizing the gradient magnitude occurred by different classes, the method avoids the bias of the proposed method in favor of the majority class. Specifically, OHGD requires no data-buffer, extra parameters, or prior knowledge. It also handles imbalanced data streams the same way that it would handle balanced data streams, which facilitates its easy implementation. On top of a few common and mild assumptions, the theoretical analysis proves that OHGD enjoys a satisfying sub-linear regret bound. Extensive experimental results demonstrate the high efficiency and effectiveness in handling imbalanced data streams.

----

## [274] Totally Dynamic Hypergraph Neural Networks

**Authors**: *Peng Zhou, Zongqian Wu, Xiangxiang Zeng, Guoqiu Wen, Junbo Ma, Xiaofeng Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/275](https://doi.org/10.24963/ijcai.2023/275)

**Abstract**:

Recent dynamic hypergraph neural networks (DHGNNs) are designed to adaptively optimize the hypergraph structure to avoid the dependence on the initial hypergraph structure, thus capturing more hidden information for representation learning. However, most existing DHGNNs cannot adjust the hyperedge number and thus fail to fully explore the underlying hypergraph structure. This paper proposes a new method, namely, totally hypergraph neural network (TDHNN), to adjust the hyperedge number for optimizing the hypergraph structure. Specifically, the proposed method first captures hyperedge feature distribution to obtain dynamical hyperedge features rather than fixed ones, by conducting the sampling from the learned distribution.
The hypergraph is then constructed based on the attention coefficients of both sampled hyperedges and nodes. The node features are dynamically updated by designing a simple hypergraph convolution algorithm. Experimental results on real datasets demonstrate the effectiveness of the proposed method, compared to SOTA methods. The source code can be accessed via https://github.com/HHW-zhou/TDHNN.

----

## [275] Simplification and Improvement of MMS Approximation

**Authors**: *Hannaneh Akrami, Jugal Garg, Eklavya Sharma, Setareh Taki*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/276](https://doi.org/10.24963/ijcai.2023/276)

**Abstract**:

We consider the problem of fairly allocating a set of indivisible goods among n agents with additive valuations, using the popular fairness notion of maximin share (MMS). Since MMS allocations do not always exist, a series of works provided existence and algorithms for approximate MMS allocations. The Garg-Taki algorithm gives the current best approximation factor of (3/4 + 1/12n). Most of these results are based on complicated analyses, especially those providing better than 2/3 factor. Moreover, since no tight example is known of the Garg-Taki algorithm, it is unclear if this is the best factor of this approach. In this paper, we significantly simplify the analysis of this algorithm and also improve the existence guarantee to a factor of (3/4 + min(1/36, 3/(16n-4))). For small n, this provides a noticeable improvement. Furthermore, we present a tight example of this algorithm, showing that this may be the best factor one can hope for with the current techniques.

----

## [276] Fair and Efficient Allocation of Indivisible Chores with Surplus

**Authors**: *Hannaneh Akrami, Bhaskar Ray Chaudhury, Jugal Garg, Kurt Mehlhorn, Ruta Mehta*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/277](https://doi.org/10.24963/ijcai.2023/277)

**Abstract**:

We study fair division of indivisible chores among n agents with additive disutility functions. Two well-studied fairness notions for indivisible items are envy-freeness up to one/any item (EF1/EFX) and the standard notion of economic efficiency is Pareto optimality (PO). There is a noticeable gap between the results known for both EF1 and EFX in the goods and chores settings. The case of chores turns out to be much more challenging. We reduce this gap by providing slightly relaxed versions of the known results on goods for the chores setting. Interestingly, our algorithms run in polynomial time, unlike their analogous versions in the goods setting.  

We introduce the concept of k surplus in the chores setting which means that up to k more chores are allocated to the agents and each of them is a copy of an original chore. We present a polynomial-time algorithm which gives EF1 and PO allocations with n-1 surplus. 
    
We relax the notion of EFX slightly and define tEFX which requires that the envy from agent i to agent j is removed upon the transfer of any chore from the i's bundle to j's bundle. We give a polynomial-time algorithm that in the chores case for 3 agents returns an allocation which is either proportional or tEFX. Note that proportionality is a very strong criterion in the case of indivisible items, and hence both notions we guarantee are desirable.

----

## [277] Non-Obvious Manipulability in Extensive-Form Mechanisms: The Revelation Principle for Single-Parameter Agents

**Authors**: *Thomas Archbold, Bart de Keijzer, Carmine Ventre*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/278](https://doi.org/10.24963/ijcai.2023/278)

**Abstract**:

Recent work in algorithmic mechanism design focuses on designing mechanisms for agents with bounded rationality, modifying the constraints that must be satisfied in order to achieve incentive compatibility.  Starting with Li's strengthening of strategyproofness, obvious strategyproofness (OSP) requires truthtelling to be "obvious" over dishonesty, roughly meaning that the worst outcome from truthful actions must be no worse than the best outcome for dishonest ones. A celebrated result for dominant-strategy incentive-compatible mechanisms that allows us to restrict attention to direct mechanisms, known as the revelation principle, does not hold for OSP: the implementation details matter for the obvious incentive properties of the mechanism. Studying agent strategies in real-life mechanisms, Troyan and Morrill introduce a relaxation of strategyproofness known as non-obvious manipulability, which only requires comparing certain extrema of the agents' utility functions in order for a mechanism to be incentive-compatible. Specifically a mechanism is not obviously manipulable (NOM) if the best and worst outcomes when acting truthfully are no worse than the best and worst outcomes when acting dishonestly. In this work we first extend the cycle monotonicity framework for direct-revelation NOM mechanism design to indirect mechanisms. We then apply this to two settings, single-parameter agents and mechanisms for two agents in which one has a two-value domain, and show that under these models the revelation principle holds: direct mechanisms are just as powerful as indirect ones.

----

## [278] Temporal Network Creation Games

**Authors**: *Davide Bilò, Sarel Cohen, Tobias Friedrich, Hans Gawendowicz, Nicolas Klodt, Pascal Lenzner, George Skretas*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/279](https://doi.org/10.24963/ijcai.2023/279)

**Abstract**:

Most networks are not static objects, but instead they change over time. This observation has sparked rigorous research on temporal graphs within the last years. In temporal graphs, we have a fixed set of nodes and the connections between them are only available at certain time steps. This gives rise to a plethora of algorithmic problems on such graphs, most prominently the problem of finding temporal spanners, i.e., the computation of subgraphs that guarantee all pairs reachability via temporal paths. To the best of our knowledge, only centralized approaches for the solution of this problem are known. However, many real-world networks are not shaped by a central designer but instead they emerge and evolve by the interaction of many strategic agents. This observation is the driving force of the recent intensive research on game-theoretic network formation models.      

In this work we bring together these two recent research directions: temporal graphs and game-theoretic network formation. As a first step into this new realm, we focus on a simplified setting where a complete temporal host graph is given and the agents, corresponding to its nodes, selfishly create incident edges to ensure that they can reach all other nodes via temporal paths in the created network. This yields temporal spanners as equilibria of our game. We prove results on the convergence to and the existence of equilibrium networks, on the complexity of finding best agent strategies, and on the quality of the equilibria. By taking these first important steps, we uncover challenging open problems that call for an in-depth exploration of the creation of temporal graphs by strategic agents.

----

## [279] Schelling Games with Continuous Types

**Authors**: *Davide Bilò, Vittorio Bilò, Michelle Döring, Pascal Lenzner, Louise Molitor, Jonas Schmidt*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/280](https://doi.org/10.24963/ijcai.2023/280)

**Abstract**:

In most major cities and urban areas, residents form homogeneous neighborhoods along ethnic or socioeconomic lines. This phenomenon is widely known as residential segregation and has been studied extensively. Fifty years ago, Schelling proposed a landmark model that explains residential segregation in an elegant agent-based way. A recent stream of papers analyzed Schelling's model using game-theoretic approaches. However, all these works considered models with a given number of discrete types modeling different ethnic groups.

We focus on segregation caused by non-categorical attributes, such as household income or position in a political left-right spectrum. For this, we consider agent types that can be represented as real numbers. This opens up a great variety of reasonable models and, as a proof of concept, we focus on several natural candidates. In particular, we consider agents that evaluate their location by the average type-difference or the maximum type-difference to their neighbors, or by having a certain tolerance range for type-values of neighboring agents.We study the existence and computation of equilibria and provide bounds on the Price of Anarchy and Stability. Also, we present simulation results that compare our models and shed light on the obtained equilibria for our variants.

----

## [280] Delegated Online Search

**Authors**: *Pirmin Braun, Niklas Hahn, Martin Hoefer, Conrad Schecker*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/281](https://doi.org/10.24963/ijcai.2023/281)

**Abstract**:

In a delegation problem, a principal P with commitment power tries to pick one out of n options. Each option is drawn independently from a known distribution. Instead of inspecting the options herself, P delegates the information acquisition to a rational and self-interested agent A. After inspection, A proposes one of the options, and P can accept or reject. In this paper, we study a natural online variant of delegation, in which the agent searches through the options in an online fashion. How can we design algorithms for P that approximate the utility of her best option in hindsight?

We show that P can obtain a Θ(1/n)-approximation and provide more fine-grained bounds independent of n based on two parameters. If the ratio of maximum and minimum utility for A is bounded by a factor α, we obtain an Ω(log log α / log α)-approximation algorithm and show that this is best possible. If P cannot distinguish options with the same value for herself, we show that ratios polynomial in 1/α cannot be avoided. If the utilities of P and A for each option are related by a factor β, we obtain an Ω(1 / log β)-approximation, and O(log log β / log β) is best possible.

----

## [281] Proportionality Guarantees in Elections with Interdependent Issues

**Authors**: *Markus Brill, Evangelos Markakis, Georgios Papasotiropoulos, Jannik Peters*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/282](https://doi.org/10.24963/ijcai.2023/282)

**Abstract**:

We consider a multi-issue election setting over a set of possibly interdependent issues with the goal of achieving proportional representation of the views of the electorate. To this end, we employ a proportionality criterion suggested recently in the literature, that guarantees fair representation for all groups of voters of sufficient size. For this criterion, there exist rules that perform well in the case where all the issues have a binary domain and are independent of each other. In particular, this has been shown for Proportional Approval Voting (PAV) and for the Method of Equal Shares (MES). In this paper, we go two steps further: we generalize these guarantees for issues with a non-binary domain, and, most importantly, we consider extensions to elections with dependencies among issues, where we identify restrictions that lead to analogous results. To achieve this, we define appropriate generalizations of PAV and MES to handle conditional ballots. In addition to proportionality considerations, we also examine the computational properties of the conditional version of MES. Our findings indicate that the conditional case poses additional challenges and differs significantly from the unconditional one, both in terms of proportionality guarantees and computational complexity.

----

## [282] Outsourcing Adjudication to Strategic Jurors

**Authors**: *Ioannis Caragiannis, Nikolaj I. Schwartzbach*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/283](https://doi.org/10.24963/ijcai.2023/283)

**Abstract**:

We study a scenario where an adjudication task (e.g., the resolution of a binary dispute) is outsourced to a set of agents who are appointed as jurors. This scenario is particularly relevant in a Web3 environment, where no verification of the adjudication outcome is possible, and the appointed agents are, in principle, indifferent to the final verdict. We consider simple adjudication mechanisms that use (1) majority voting to decide the final verdict and (2) a payment function to reward the agents with the majority vote and possibly punish the ones in the minority. Agents interact with such a mechanism strategically: they exert some effort to understand how to properly judge the dispute and cast a yes/no vote that depends on this understanding and on information they have about the rest of the votes. Eventually, they vote so that their utility (i.e., their payment from the mechanism minus the cost due to their effort) is maximized. Under reasonable assumptions about how an agent's effort is related to her understanding of the dispute, we show that appropriate payment functions can be used to recover the correct adjudication outcome with high probability. Our findings follow from a detailed analysis of the induced strategic game and make use of both theoretical arguments and simulation experiments.

----

## [283] New Fairness Concepts for Allocating Indivisible Items

**Authors**: *Ioannis Caragiannis, Jugal Garg, Nidhi Rathi, Eklavya Sharma, Giovanna Varricchio*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/284](https://doi.org/10.24963/ijcai.2023/284)

**Abstract**:

For the fundamental problem of fairly dividing a set of indivisible items among agents, envy-freeness up to any item (EFX) and maximin fairness (MMS) are arguably the most compelling fairness concepts proposed till now.  Unfortunately, despite significant efforts over the past few years, whether EFX allocations always exist is still an enigmatic open problem, let alone their efficient computation. Furthermore, today we know that MMS allocations are not always guaranteed to exist. These facts weaken the usefulness of both EFX and MMS, albeit their appealing conceptual characteristics. 

We propose two alternative fairness conceptsâ€”called epistemic EFX (EEFX) and minimum EFX value fairness (MXS)---inspired by EFX and MMS. For both, we explore their relationships to well-studied fairness notions and, more importantly, prove that EEFX and MXS allocations always exist and can be computed efficiently for additive valuations. Our results justify that the new fairness concepts are excellent alternatives to EFX and MMS.

----

## [284] Optimal Seat Arrangement: What Are the Hard and Easy Cases?

**Authors**: *Esra Ceylan, Jiehua Chen, Sanjukta Roy*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/285](https://doi.org/10.24963/ijcai.2023/285)

**Abstract**:

We study four NP-hard optimal seat arrangement problems which each have as input a set of n agents, where each agent has cardinal preferences over other agents, and an n-vertex undirected graph (called the seat graph). The task is to assign each agent to a distinct vertex in the seat graph such that either the sum of utilities or the minimum utility is maximized, or it is envy-free or exchange-stable. Aiming at identifying hard and easy cases, we extensively study the algorithmic complexity of the four problems by looking into natural graph classes for the seat graph (e.g., paths, cycles, stars, or matchings), problem-specific parameters (e.g., the number of non-isolated vertices in the seat graph or the maximum number of agents towards whom an agent has non-zero preferences), and preference structures (e.g., non-negative or symmetric preferences). For strict preferences and seat graphs with disjoint edges and isolated vertices, we correct an error in the literature and show that finding an envy-free arrangement remains NP-hard in this case.

----

## [285] Rainbow Cycle Number and EFX Allocations: (Almost) Closing the Gap

**Authors**: *Shayan Chashm Jahan, Masoud Seddighin, Seyed Mohammad Seyed Javadi, Mohammad Sharifi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/286](https://doi.org/10.24963/ijcai.2023/286)

**Abstract**:

Recently, some studies on the fair allocation of indivisible goods notice a connection between a purely combinatorial problem called the Rainbow Cycle problem and a fairness notion known as EFX: assuming that the rainbow cycle number for parameter d (i.e. R(d)) is O(d^β .log(d)^γ), we can find a (1 − ϵ)-EFX allocation with O_ϵ(n^(β/β+1) .log(n)^(γ/β+1)) number of discarded goods. The best upper bound on R(d) is improved in a series of works to O(d^4), O(d^(2+o(1))), and finally to O(d^2). Also, via a simple observation, we have R(d) ∈ Ω(d).
In this paper, we introduce another problem in extremal combinatorics. For a parameter l, we define the rainbow path degree and denote it by H(l). We show that any lower bound on H(l) yields an upper bound on R(d). Next, we prove that H(l) ∈ Ω(l^2 / log(l)) which yields an almost tight upper bound of R(d) ∈ Ω(d.log(d)).  This, in turn, proves the existence of (1−ϵ)-EFX allocation with O_ϵ(√n .log(n)) number of discarded goods. In addition, for the special case of the Rainbow Cycle problem that the edges in each part form a permutation, we improve the upper bound to R(d) ≤ 2d−4. We leverage H(l) to achieve this bound.
Our conjecture is that the exact value of H(l) is ⌊l^2/2⌋ − 1. We provide some experiments that support this conjecture. Assuming this conjecture is correct, we have R(d) ∈ θ(d).

----

## [286] Exploring Leximin Principle for Fair Core-Selecting Combinatorial Auctions: Payment Rule Design and Implementation

**Authors**: *Hao Cheng, Shufeng Kong, Yanchen Deng, Caihua Liu, Xiaohu Wu, Bo An, Chongjun Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/287](https://doi.org/10.24963/ijcai.2023/287)

**Abstract**:

Core-selecting combinatorial auctions (CAs) restrict the auction result in the core such that no coalitions could improve their utilities by engaging in collusion. The minimum-revenue-core (MRC) rule is a widely used core-selecting payment rule to maximize the total utilities of all bidders. However, the MRC rule can suffer from severe unfairness since it ignores individuals' utilities. To address this limitation, we propose to explore the leximin principle to achieve fairness in core-selecting CAs since the leximin principle prefers to maximize the utility of the worst-off; the resulting bidder-leximin-optimal (BLO) payment rule is then theoretically analyzed and an effective algorithm is further provided to compute the BLO outcome. Moreover, we conduct extensive experiments to show that our algorithm returns fairer utility distributions and is faster than existing algorithms of core-selecting payment rules.

----

## [287] Deliberation as Evidence Disclosure: A Tale of Two Protocol Types

**Authors**: *Julian Chingoma, Adrian Haret*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/288](https://doi.org/10.24963/ijcai.2023/288)

**Abstract**:

We study a model inspired by deliberative practice, in which agents selectively disclose evidence about a set of alternatives prior to taking a final decision on them. We are interested in whether such a process, when iterated to termination, results in the objectively best alternatives being selectedâ€”thereby lending support to the idea that groups can be wise even when their members communicate with each other. We find that, under certain restrictions on the relative amounts of evidence, together with the actions available to the agents, there exist deliberation protocols in each of the two families we look at (i.e., simultaneous and sequential) that offer desirable guarantees. Simulation results further complement this picture, by showing how the distribution of evidence among the agents influences parameters of interest, such as the outcome of the protocols and the number of rounds until termination.

----

## [288] Adversarial Contention Resolution Games

**Authors**: *Giorgos Chionas, Bogdan S. Chlebus, Dariusz R. Kowalski, Piotr Krysta*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/289](https://doi.org/10.24963/ijcai.2023/289)

**Abstract**:

We study contention resolution (CR) on a shared channel modelled as a game with selfish players. There are n agents and the adversary chooses some k smaller than n of them as players. Each participating player in a CR game has a packet to transmit. A transmission is successful if it is performed as the only one at a round. Each player aims to minimize its packet latency. We introduce the notion of adversarial equilibrium (AE), which incorporates adversarial selection of players. We develop efficient deterministic communication algorithms that are also AE. We characterize the price of anarchy in the CR games with respect to AE.

----

## [289] Measuring a Priori Voting Power in Liquid Democracy

**Authors**: *Rachael Colley, Théo Delemazure, Hugo Gilbert*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/290](https://doi.org/10.24963/ijcai.2023/290)

**Abstract**:

We introduce new power indices to measure the a priori voting power of voters in liquid democracy elections where an underlying network restricts delegations. We argue that our power indices are natural extensions of the standard Penrose-Banzhaf index in simple voting games. 
We show that computing the criticality of a voter is #P-hard even in weighted games with weights polynomially-bounded in the size of the instance. 
However, for specific settings, such as when the underlying network is a bipartite or complete graph, recursive formulas can compute these indices for weighted voting games in pseudo-polynomial time. 
We highlight their theoretical properties and provide numerical results to illustrate how restricting the possible delegations can alter voters' voting power.

----

## [290] Measuring and Controlling Divisiveness in Rank Aggregation

**Authors**: *Rachael Colley, Umberto Grandi, César A. Hidalgo, Mariana Macedo, Carlos Navarrete*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/291](https://doi.org/10.24963/ijcai.2023/291)

**Abstract**:

In rank aggregation, members of a population rank issues to decide which are collectively preferred.  We focus instead on identifying divisive issues that express disagreements among the preferences of individuals. We analyse the properties of our divisiveness measures and their relation to existing notions of polarisation. We also study their robustness under incomplete preferences and algorithms for control and manipulation of divisiveness.  Our results advance our understanding of how to quantify disagreements in collective decision-making.

----

## [291] Inferring Private Valuations from Behavioral Data in Bilateral Sequential Bargaining

**Authors**: *Lvye Cui, Haoran Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/292](https://doi.org/10.24963/ijcai.2023/292)

**Abstract**:

Inferring bargainers' private valuations on items from their decisions is crucial for analyzing their strategic behaviors in bilateral sequential bargaining. Most existing approaches that infer agents' private information from observable data either rely on strong equilibrium assumptions or require a careful design of agents' behavior models. To overcome these weaknesses, we propose a Bayesian Learning-based Valuation Inference (BLUE) framework. Our key idea is to derive feasible intervals of bargainers' private valuations from their behavior data, using the fact that most bargainers do not choose strictly dominated strategies. We leverage these feasible intervals to guide our inference. Specifically, we first model each bargainer's behavior function (which maps his valuation and bargaining history to decisions) via a recurrent neural network. Second, we learn these behavior functions by utilizing a novel loss function defined based on feasible intervals. Third, we derive the posterior distributions of bargainers' valuations according to their behavior data and learned behavior functions. Moreover, we account for the heterogeneity of bargainer behaviors, and propose a clustering algorithm (K-Loss) to improve the efficiency of learning these behaviors. Experiments on both synthetic and real bargaining data show that our inference approach outperforms baselines.

----

## [292] Differentiable Economics for Randomized Affine Maximizer Auctions

**Authors**: *Michael J. Curry, Tuomas Sandholm, John P. Dickerson*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/293](https://doi.org/10.24963/ijcai.2023/293)

**Abstract**:

A recent approach to automated mechanism design, differentiable economics, represents auctions by rich function approximators and optimizes their performance by gradient descent. The ideal auction architecture for differentiable economics would be perfectly strategyproof, support multiple bidders and items, and be rich enough to represent the optimal (i.e. revenue-maximizing) mechanism. So far, such an architecture does not exist. There are single-bidder approaches (MenuNet, RochetNet) which are always strategyproof and can represent optimal mechanisms. RegretNet is multi-bidder and can approximate any mechanism, but is only approximately strategyproof. We present an architecture that supports multiple bidders and is perfectly strategyproof, but cannot necessarily represent the optimal mechanism. This architecture is the classic affine maximizer auction (AMA), modified to offer lotteries. By using the gradient-based optimization tools of differentiable economics, we can now train lottery AMAs, competing with or outperforming prior approaches in revenue.

----

## [293] Complexity of Efficient Outcomes in Binary-Action Polymatrix Games and Implications for Coordination Problems

**Authors**: *Argyrios Deligkas, Eduard Eiben, Gregory Z. Gutin, Philip R. Neary, Anders Yeo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/294](https://doi.org/10.24963/ijcai.2023/294)

**Abstract**:

We investigate the difficulty of finding economically efficient solutions to coordination problems on graphs. Our work focuses on two forms of coordination problem: pure-coordination games and anti-coordination games. We consider three objectives in the context of simple binary-action polymatrix games: (i) maximizing welfare, (ii) maximizing potential, and (iii) finding a welfare-maximizing Nash equilibrium. We introduce an intermediate, new graph-partition problem, termed MWDP, which is of independent interest, and we provide a  complexity dichotomy for it. This dichotomy, among other results, provides as a corollary a dichotomy for Objective (i) for general binary-action polymatrix games. In addition, it reveals that the complexity of achieving these objectives varies depending on the form of the coordination problem. Specifically, Objectives (i) and (ii) can be efficiently solved in pure-coordination games, but are NP-hard in anti-coordination games. Finally, we show that objective (iii) is NP-hard even for simple non-trivial pure-coordination games.

----

## [294] Algorithmics of Egalitarian versus Equitable Sequences of Committees

**Authors**: *Eva Michelle Deltl, Till Fluschnik, Robert Bredereck*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/295](https://doi.org/10.24963/ijcai.2023/295)

**Abstract**:

We study the election of sequences of committees, where in each of tau levels (e.g. modeling points in time) a committee consisting of k candidates from a common set of m candidates is selected. For each level, each of n agents (voters) may nominate one candidate whose selection would satisfy her. We are interested in committees which are good with respect to the satisfaction per day and per agent. More precisely, we look for egalitarian or equitable committee sequences. While both guarantee that at least x agents per day are satisfied, egalitarian committee sequences ensure that each agent is satisfied in at least y levels while equitable committee sequences ensure that each agent is satisfied in exactly y levels. We analyze the parameterized complexity of finding such committees for the parameters n, m, k, tau, x, and y, as well as combinations thereof.

----

## [295] Discrete Two Player All-Pay Auction with Complete Information

**Authors**: *Marcin Dziubinski, Krzysztof Jahn*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/296](https://doi.org/10.24963/ijcai.2023/296)

**Abstract**:

We study discrete two player all-pay auction with complete information. We provide full characterization of mixed strategy Nash equilibria and show that they constitute a subset of Nash equilibria of discrete General Lotto game. We show that equilibria are not unique in general but they are interchangeable and sets of equilibrium strategies are convex. We also show that equilibrium payoffs are unique, unless valuation of at least one of the players is an even integer number. If equilibrium payoffs are not unique, continuum of equilibrium payoffs are possible.

----

## [296] Participatory Budgeting: Data, Tools and Analysis

**Authors**: *Piotr Faliszewski, Jaroslaw Flis, Dominik Peters, Grzegorz Pierczynski, Piotr Skowron, Dariusz Stolicki, Stanislaw Szufa, Nimrod Talmon*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/297](https://doi.org/10.24963/ijcai.2023/297)

**Abstract**:

We provide a library of participatory budgeting data (Pabulib) and open source tools (Pabutools and Pabustats) for analysing this data.
    We analyse how the results of participatory budgeting elections would change if a different selection rule was applied. 
    We provide evidence that the outcomes of the Method of Equal Shares would be considerably fairer than those of the Utilitarian Greedy rule that is currently in use. We also show that the division of the projects into districts and/or categories can in many cases be avoided when using proportional rules. We find that this would increase the overall utility of the voters.

----

## [297] An Experimental Comparison of Multiwinner Voting Rules on Approval Elections

**Authors**: *Piotr Faliszewski, Martin Lackner, Krzysztof Sornat, Stanislaw Szufa*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/298](https://doi.org/10.24963/ijcai.2023/298)

**Abstract**:

In this paper, we experimentally compare major approval based multiwinner voting rules. To this end, we define a measure of similarity between two equal sized committees subject to a given election. Using synthetic elections coming from several distributions, we analyze how similar are the committees provided by prominent voting rules. Our results can be visualized as maps of voting rules, which provide a counterpoint to a purely axiomatic classification of voting rules. The strength of our proposed method is its independence from preimposed classifications (such as the satisfaction of concrete axioms), and that it indeed offers a much finer distinction than the current state of axiomatic analysis.

----

## [298] Diversity, Agreement, and Polarization in Elections

**Authors**: *Piotr Faliszewski, Andrzej Kaczmarczyk, Krzysztof Sornat, Stanislaw Szufa, Tomasz Was*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/299](https://doi.org/10.24963/ijcai.2023/299)

**Abstract**:

We consider the notions of agreement, diversity, and polarization in ordinal elections (that is, in elections where voters rank the candidates). While (computational) social choice offers good measures of agreement between the voters, such measures for the other two notions are lacking. We attempt to rectify this issue by designing appropriate measures, providing means of their (approximate) computation, and arguing that they, indeed, capture diversity and polarization well. In particular, we present "maps of preference orders" that highlight relations between the votes in a given election and which help in making arguments about their nature.

----

## [299] Revenue Maximization Mechanisms for an Uninformed Mediator with Communication Abilities

**Authors**: *Zhikang Fan, Weiran Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/300](https://doi.org/10.24963/ijcai.2023/300)

**Abstract**:

Consider a market where a seller owns an item for sale and a buyer wants to purchase it. Each player has private information, known as their type. It can be costly and difficult for the players to reach an agreement through direct communication. However, with a mediator as a trusted third party, both players can communicate privately with the mediator without worrying about leaking too much or too little information. The mediator can design and commit to a multi-round communication protocol for both players, in which they update their beliefs about the other player's type. The mediator cannot force the players to trade but can influence their behaviors by sending messages to them.

We study the problem of designing revenue-maximizing mechanisms for the mediator. We show that the mediator can, without loss of generality, focus on a set of direct and incentive-compatible mechanisms. We then formulate this problem as a mathematical program and provide an optimal solution in closed form under a regularity condition. Our mechanism is simple and has a threshold structure. We also discuss some interesting properties of the optimal mechanism, such as situations where the mediator may lose money.

----

## [300] Strategic Resource Selection with Homophilic Agents

**Authors**: *Jonathan Gadea Harder, Simon Krogmann, Pascal Lenzner, Alexander Skopalik*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/301](https://doi.org/10.24963/ijcai.2023/301)

**Abstract**:

The strategic selection of resources  by selfish agents is a classical research direction, with Resource Selection Games and Congestion Games as prominent examples. In these games, agents select available resources and their utility then depends on the number of agents using the same resources. This implies that there is no distinction between the agents, i.e., they are anonymous.

We depart from this very general setting by proposing Resource Selection Games with heterogeneous agents that strive for a joint resource usage with similar agents. So, instead of the number of other users of a given resource, our model considers agents with different types and the decisive feature is the fraction of same-type agents among the users. More precisely, similarly to Schelling Games, there is a tolerance threshold tau in [0,1] which specifies the agents' desired minimum fraction of same-type agents on a resource. Agents strive to select resources where at least a tau-fraction of those resources' users have the same type as themselves. For tau=1, our model generalizes hedonic diversity games with single-peaked utilities with a peak at 1. 

For our general model, we consider the existence and quality of equilibria and the complexity of maximizing the social welfare. Additionally, we consider a bounded rationality model, where agents can only estimate the utility of a resource, since they only know the fraction of same-type agents on a given resource, but not the exact numbers. Thus, they cannot know the impact a strategy change would have on a target resource. Interestingly, we show that this type of bounded rationality yields favorable game-theoretic properties and specific equilibria closely approximate equilibria of the full knowledge setting.

----

## [301] New Algorithms for the Fair and Efficient Allocation of Indivisible Chores

**Authors**: *Jugal Garg, Aniket Murhekar, John Qin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/302](https://doi.org/10.24963/ijcai.2023/302)

**Abstract**:

We study the problem of fairly and efficiently allocating indivisible chores among agents with additive disutility functions. We consider the widely used envy-based fairness properties of EF1 and EFX in conjunction with the efficiency property of fractional Pareto-optimality (fPO). Existence (and computation) of an allocation that is simultaneously EF1/EFX and fPO are challenging open problems, and we make progress on both of them. We show the existence of an allocation that is
- EF1 + fPO, when there are three agents,
- EF1 + fPO, when there are at most two disutility functions,
- EFX + fPO, for three agents with bivalued disutility functions.
These results are constructive, based on strongly polynomial-time algorithms. We also investigate non-existence and show that an allocation that is EFX+fPO need not exist, even for two agents.

----

## [302] First-Choice Maximality Meets Ex-ante and Ex-post Fairness

**Authors**: *Xiaoxi Guo, Sujoy Sikdar, Lirong Xia, Yongzhi Cao, Hanpin Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/303](https://doi.org/10.24963/ijcai.2023/303)

**Abstract**:

For the assignment problem where multiple indivisible items are allocated to a group of agents given their ordinal preferences, we design randomized mechanisms that satisfy first-choice maximality (FCM), i.e., maximizing the number of agents assigned their first choices, together with Pareto efficiency (PE). Our mechanisms also provide guarantees of ex-ante and ex-post fairness. The generalized eager Boston mechanism is ex-ante envy-free, and ex-post envy-free up to one item (EF1). The generalized probabilistic Boston mechanism is also ex-post EF1, and satisfies ex-ante efficiency instead of fairness. We also show that no strategyproof mechanism satisfies ex-post PE, EF1, and FCM simultaneously. In doing so, we expand the frontiers of simultaneously providing efficiency and both ex-ante and ex-post fairness guarantees for the assignment problem.

----

## [303] A Unifying Formal Approach to Importance Values in Boolean Functions

**Authors**: *Hans Harder, Simon Jantsch, Christel Baier, Clemens Dubslaff*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/304](https://doi.org/10.24963/ijcai.2023/304)

**Abstract**:

Boolean functions and their representation through logics, circuits, machine learning classifiers, or binary decision diagrams (BDDs) play a central role in the design and analysis of computing systems. Quantifying the relative impact of variables on the truth value by means of importance values can provide useful insights to steer system design and debugging. In this paper, we introduce a uniform framework for reasoning about such values, relying on a generic notion of importance value functions (IVFs). The class of IVFs is defined by axioms motivated from several notions of importance values introduced in the literature, including Ben-Or and Linial’s influence and Chockler, Halpern, and Kupferman’s notion of responsibility and blame. We establish a connection between IVFs and game-theoretic concepts such as Shapley and Banzhaf values, both of which measure the impact of players on outcomes in cooperative games. Exploiting BDD-based symbolic methods and projected model counting, we devise and evaluate practical computation schemes for IVFs.

----

## [304] Fairly Allocating Goods and (Terrible) Chores

**Authors**: *Hadi Hosseini, Aghaheybat Mammadov, Tomasz Was*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/305](https://doi.org/10.24963/ijcai.2023/305)

**Abstract**:

We study the fair allocation of mixture of indivisible goods and chores under lexicographic preferences---a subdomain of additive preferences. A prominent fairness notion for allocating indivisible items is envy-freeness up to any item (EFX). Yet, its existence and computation has remained a notable open problem. By identifying a class of instances with "terrible chores", we  show that determining the existence of an EFX allocation is NP-complete. This result immediately implies the intractability of EFX under additive preferences. Nonetheless, we propose a natural subclass of lexicographic preferences for which an EFX and Pareto optimal (PO) allocation is guaranteed to exist and can be computed efficiently for any mixed instance. Focusing on two weaker fairness notions, we investigate finding EF1 and Pareto optimal allocations for special instances with terrible chores, and show that MMS and PO allocations can be computed efficiently for any mixed instance with lexicographic preferences.

----

## [305] On Lower Bounds for Maximin Share Guarantees

**Authors**: *Halvard Hummel*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/306](https://doi.org/10.24963/ijcai.2023/306)

**Abstract**:

We study the problem of fairly allocating a set of indivisible items to a set of agents with additive valuations. Recently, Feige et al. (WINE'21) proved that a maximin share (MMS) allocation exists for all instances with n agents and no more than n + 5 items. Moreover, they proved that an MMS allocation is not guaranteed to exist for instances with 3 agents and at least 9 items, or n ≥ 4 agents and at least 3n + 3 items. In this work, we shrink the gap between these upper and lower bounds for guaranteed existence of MMS allocations. We prove that for any integer c > 0, there exists a number of agents n_c such that an MMS allocation exists for any instance with n ≥ n_c agents and at most n + c items, where n_c ≤ ⌊0.6597^c · c!⌋ for allocation of goods and n_c ≤ ⌊0.7838^c · c!⌋ for chores. Furthermore, we show that for n ≠ 3 agents, all instances with n + 6 goods have an MMS allocation.

----

## [306] Fair Division with Two-Sided Preferences

**Authors**: *Ayumi Igarashi, Yasushi Kawase, Warut Suksompong, Hanna Sumita*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/307](https://doi.org/10.24963/ijcai.2023/307)

**Abstract**:

We study a fair division setting in which a number of players are to be fairly distributed among a set of teams. In our model, not only do the teams have preferences over the players as in the canonical fair division setting, but the players also have preferences over the teams. We focus on guaranteeing envy-freeness up to one player (EF1) for the teams together with a stability condition for both sides. We show that an allocation satisfying EF1, swap stability, and individual stability always exists and can be computed in polynomial time, even when teams may have positive or negative values for players. Similarly, a balanced and swap stable allocation that satisfies a relaxation of EF1 can be computed efficiently. When teams have nonnegative values for players, we prove that an EF1 and Pareto optimal allocation exists and, if the valuations are binary, can be found in polynomial time. We also examine the compatibility between EF1 and justified envy-freeness.

----

## [307] Ties in Multiwinner Approval Voting

**Authors**: *Lukasz Janeczko, Piotr Faliszewski*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/308](https://doi.org/10.24963/ijcai.2023/308)

**Abstract**:

We study the complexity of deciding if there is a tie in a given approval-based multiwinner election, as well as the complexity of counting tied winning committees. We consider a family of Thiele rules, their greedy variants, Phragmen's sequential rule, and Method of Equal Shares. For most cases, our problems are computationally hard, but for sequential rules we find an FPT algorithm for discovering ties (parameterized by the committee size). We also show experimentally that in elections of moderate size ties are quite frequent.

----

## [308] Matchings under One-Sided Preferences with Soft Quotas

**Authors**: *Santhini K. A., Raghu Raman Ravi, Meghana Nasre*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/309](https://doi.org/10.24963/ijcai.2023/309)

**Abstract**:

Assigning applicants to posts in the presence of the preferences of applicants and quotas associated with posts is extensively investigated. For a post, lower quota guarantees, and upper quota limits the number of applicants assigned to it. Typically, quotas are assumed to be fixed, which need not be the case in practice. We address this by introducing a soft quota setting, in which every post is associated with two values – lower target and upper target which together denote a range for the intended number of applicants in any assignment. Unlike the fixed quota setting, we allow the number of applicants assigned to a post to fall outside the range.  This leads to assignments with deviation. Here, we study the problem of computing an assignment that has two orthogonal optimization objectives – minimizing the deviation (maximum or total) w.r.t. soft quotas and ensuring optimality w.r.t. preferences of applicants (rank-maximality or fairness). The order in which these objectives are considered, the different possibilities to optimize deviation combined with the well-studied notions of optimality w.r.t. preferences open up a range of optimization problems of practical importance. We present efficient algorithms based on flow-networks to solve these optimization problems.

----

## [309] Convergence in Multi-Issue Iterative Voting under Uncertainty

**Authors**: *Joshua Kavner, Reshef Meir, Francesca Rossi, Lirong Xia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/310](https://doi.org/10.24963/ijcai.2023/310)

**Abstract**:

We study strategic behavior in iterative plurality voting for multiple issues under uncertainty. We introduce a model synthesizing simultaneous multi-issue voting with local dominance theory, in which agents repeatedly update their votes based on sets of vote profiles they deem possible, and determine its convergence properties. After demonstrating that local dominance improvement dynamics may fail to converge, we present two sufficient model refinements that guarantee convergence from any initial vote profile for binary issues: constraining agents to have O-legal preferences, where issues are ordered by importance, and endowing agents with less uncertainty about issues they are modifying than others. Our empirical studies demonstrate that while cycles are common for agents without uncertainty, introducing uncertainty makes convergence almost guaranteed in practice.

----

## [310] Random Assignment of Indivisible Goods under Constraints

**Authors**: *Yasushi Kawase, Hanna Sumita, Yu Yokoi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/311](https://doi.org/10.24963/ijcai.2023/311)

**Abstract**:

We investigate the problem of random assignment of indivisible goods, in which each agent has an ordinal preference and a constraint. Our goal is to characterize the conditions under which there always exists a random assignment that simultaneously satisfies efficiency and envy-freeness. The probabilistic serial mechanism ensures the existence of such an assignment for the unconstrained setting. In this paper, we consider a more general setting in which each agent can consume a set of items only if the set satisfies her feasibility constraint. Such constraints must be taken into account in student course placements, employee shift assignments, and so on. We demonstrate that an efficient and envy-free assignment may not exist even for the simple case of partition matroid constraints, where the items are categorized, and each agent demands one item from each category. We then identify special cases in which an efficient and envy-free assignment always exists. For these cases, the probabilistic serial cannot be naturally extended; therefore, we provide mechanisms to find the desired assignment using various approaches.

----

## [311] Game Theory with Simulation of Other Players

**Authors**: *Vojtech Kovarík, Caspar Oesterheld, Vincent Conitzer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/312](https://doi.org/10.24963/ijcai.2023/312)

**Abstract**:

Game-theoretic interactions with AI agents could differ from traditional human-human interactions in various ways. One such difference is that it may be possible to simulate an AI agent (for example because its source code is known), which allows others to accurately predict the agent's actions. This could lower the bar for trust and cooperation. In this paper, we first formally define games in which one player can simulate another at a cost, and derive some basic properties of such games. Then, we prove a number of results for such games, including: (1) introducing simulation into generic-payoff normal-form games makes them easier to solve; (2) if the only obstacle to cooperation is a lack of trust in the possibly-simulated agent, simulation enables equilibria that improve the outcome for both agents; and (3) however, there are settings where introducing simulation results in strictly worse outcomes for both players.

----

## [312] Truthful Fair Mechanisms for Allocating Mixed Divisible and Indivisible Goods

**Authors**: *Zihao Li, Shengxin Liu, Xinhang Lu, Biaoshuai Tao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/313](https://doi.org/10.24963/ijcai.2023/313)

**Abstract**:

We study the problem of designing truthful and fair mechanisms when allocating a mixture of divisible and indivisible goods. We first show that there does not exist an EFM (envy-free for mixed goods) and truthful mechanism in general. This impossibility result holds even if there is only one indivisible good and one divisible good and there are only two agents. Thus, we focus on some more restricted settings. Under the setting where agents have binary valuations on indivisible goods and identical valuations on a single divisible good (e.g., money), we design an EFM and truthful mechanism. When agents have binary valuations over both divisible and indivisible goods, we first show there exist EFM and truthful mechanisms when there are only two agents or when there is a single divisible good. On the other hand, we show that the mechanism maximizing Nash welfare cannot ensure EFM and truthfulness simultaneously.

----

## [313] Auto-bidding with Budget and ROI Constrained Buyers

**Authors**: *Xiaodong Liu, Weiran Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/314](https://doi.org/10.24963/ijcai.2023/314)

**Abstract**:

In online advertising markets, an increasing number of advertisers are adopting auto-bidders to buy advertising slots. This tool simplifies the process of optimizing bids based on various financial constraints.

In our study, we focus on second-price auctions where bidders have both private budget and private ROI (return on investment) constraints. We formulate the auto-bidding system design problem as a mathematical program and analyze the auto-bidders' bidding strategy under such constraints. We demonstrate that our design ensures truthfulness, i.e., among all pure and mixed strategies, always reporting the truthful budget and ROI is an optimal strategy for the bidders. Although the program is non-convex, we provide a fast algorithm to compute the optimal bidding strategy for the bidders based on our analysis. We also study the welfare and provide a lower bound for the PoA (price of anarchy). Moreover, we prove that if all bidders utilize our auto-bidding system, a Bayesian Nash equilibrium exists. We provide a sufficient condition under which the iterated best response process converges to such an equilibrium. Finally, we conduct extensive experiments to empirically evaluate the effectiveness of our design.

----

## [314] Approximating Fair Division on D-Claw-Free Graphs

**Authors**: *Zbigniew Lonc*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/315](https://doi.org/10.24963/ijcai.2023/315)

**Abstract**:

We study the problem of fair allocation of indivisible goods that form a graph and the bundles that are distributed to agents are connected subgraphs of this graph. We focus on the maximin share and the proportional fairness criteria. It is well-known that allocations satisfying these criteria may not exist for many graphs including complete graphs and cycles. Therefore, it is natural to look for approximate allocations, i.e., allocations guaranteeing each agent a certain portion of the value that is satisfactory to her. In this paper we consider the class of graphs of goods which do not contain a star with d+1 edges (where d > 1) as an induced subgraph. For this class of graphs we prove that there is an allocation assigning each agent a connected bundle of value at least 1/d of her maximin share. Moreover, for the same class of graphs of goods, we show a theorem which specifies what fraction of the proportional share can be guaranteed to each agent if the values of single goods for the agents are bounded by a given fraction of this share.

----

## [315] Fair Division of a Graph into Compact Bundles

**Authors**: *Jayakrishnan Madathil*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/316](https://doi.org/10.24963/ijcai.2023/316)

**Abstract**:

We study the computational complexity of fair division of indivisible items in an enriched model: there is an underlying graph on the set of items. And we have to allocate the items (i.e., the vertices of the graph) to a set of agents in such a way that (a) the allocation is fair (for appropriate notions of fairness) and (b) each agent receives a bundle of items (i.e., a subset of vertices) that induces a subgraph with a specific ``nice structure.'' This model has previously been studied in the literature with the nice structure being a connected subgraph. In this paper, we propose an alternative for connectivity in fair division. We introduce compact graphs, and look for fair allocations in which each agent receives a compact bundle of items. Through compactness, we attempt to capture the idea that every agent must receive a bundle of ``closely related'' items. We prove a host of hardness and tractability results with respect to fairness concepts such as proportionality, envy-freeness and maximin share guarantee.

----

## [316] Finding Mixed-Strategy Equilibria of Continuous-Action Games without Gradients Using Randomized Policy Networks

**Authors**: *Carlos Martin, Tuomas Sandholm*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/317](https://doi.org/10.24963/ijcai.2023/317)

**Abstract**:

We study the problem of computing an approximate Nash equilibrium of continuous-action game without access to gradients. Such game access is common in reinforcement learning settings, where the environment is typically treated as a black box. To tackle this problem, we apply zeroth-order optimization techniques that combine smoothed gradient estimators with equilibrium-finding dynamics.
We model players' strategies using artificial neural networks. In particular, we use randomized policy networks to model mixed strategies. These take noise in addition to an observation as input and can flexibly represent arbitrary observation-dependent, continuous-action distributions. Being able to model such mixed strategies is crucial for tackling continuous-action games that lack pure-strategy equilibria.
We evaluate the performance of our method using an approximation of the Nash convergence metric from game theory, which measures how much players can benefit from unilaterally changing their strategy.
We apply our method to continuous Colonel Blotto games, single-item and multi-item auctions, and a visibility game.
The experiments show that our method can quickly find a high-quality approximate equilibrium.
Furthermore, they show that the dimensionality of the input noise is crucial for performance.
To our knowledge, this paper is the first to solve general continuous-action games with unrestricted mixed strategies and without any gradient information.

----

## [317] Deliberation and Voting in Approval-Based Multi-Winner Elections

**Authors**: *Kanav Mehra, Nanda Kishore Sreenivas, Kate Larson*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/318](https://doi.org/10.24963/ijcai.2023/318)

**Abstract**:

Citizen-focused democratic processes where participants deliberate on alternatives and then vote to make the final decision are increasingly popular today. While the computational social choice literature has extensively investigated voting rules, there is limited work that explicitly looks at the interplay of the deliberative process and voting. In this paper, we build a deliberation model using established models from the opinion-dynamics literature and study the effect of different deliberation mechanisms on voting outcomes achieved when using well-studied voting rules. Our results show that deliberation generally improves welfare and representation guarantees, but the results are sensitive to how the deliberation process is organized. We also show, experimentally, that simple voting rules, such as approval voting, perform as well as more sophisticated rules such as proportional approval voting or method of equal shares if deliberation is properly supported. This has ramifications on the practical use of such voting rules in citizen-focused democratic processes.

----

## [318] Learning Efficient Truthful Mechanisms for Trading Networks

**Authors**: *Takayuki Osogami, Segev Wasserkrug, Elisheva S. Shamash*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/319](https://doi.org/10.24963/ijcai.2023/319)

**Abstract**:

Trading networks are an indispensable part of today's economy, but to compete successfully with others, they must be efficient in maximizing the value they provide to the external market.  While the prior work relies on truthful disclosure of private information to achieve efficiency, we study the problem of designing mechanisms that result in efficient trading networks by incentivizing firms to truthfully reveal their private information to a third party. Additional desirable properties of such mechanisms are weak budget balance (WBB; the third party needs not invest) and individual rationality (IR; firms get non-negative utility).  Unlike combinatorial auctions, there may not exist mechanisms that simultaneously satisfy these properties ex post for trading networks.  We propose an approach for computing or learning truthful and efficient mechanisms for given networks in a Bayesian setting, where WBB and IR, respectively, are relaxed to ex ante and interim for a given distribution over the private information.  We incorporate techniques to reduce computational and sample complexity.  We empirically demonstrate that the proposed approach successfully finds the mechanisms with the relaxed properties for trading networks where achieving ex post properties is impossible.

----

## [319] Participatory Budgeting with Multiple Degrees of Projects and Ranged Approval Votes

**Authors**: *Gogulapati Sreedurga*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/320](https://doi.org/10.24963/ijcai.2023/320)

**Abstract**:

In an indivisible participatory budgeting (PB) framework, we have a limited budget that is to be distributed among a set of projects, by aggregating the preferences of voters for the projects. All the prior work on indivisible PB assumes that each project has only one possible cost. In this work, we let each project have a set of permissible costs, each reflecting a possible degree of sophistication of the project. Each voter approves a range of costs for each project, by giving an upper and lower bound on the cost that she thinks the project deserves. The outcome of a PB rule selects a subset of projects and also specifies their corresponding costs. We study different utility notions and prove that the existing positive results when every project has exactly one permissible cost can also be extended to our framework where a project has several permissible costs. We also analyze the fixed parameter tractability of the problem. Finally, we propose some important and intuitive axioms and analyze their satisfiability by different PB rules. We conclude by making some crucial remarks.

----

## [320] The Computational Complexity of Single-Player Imperfect-Recall Games

**Authors**: *Emanuel Tewolde, Caspar Oesterheld, Vincent Conitzer, Paul W. Goldberg*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/321](https://doi.org/10.24963/ijcai.2023/321)

**Abstract**:

We study single-player extensive-form games with imperfect recall, such as the Sleeping Beauty problem or the Absentminded Driver game. For such games, two natural equilibrium concepts have been proposed as alternative solution concepts to ex-ante optimality. One equilibrium concept uses generalized double halving (GDH) as a belief system and evidential decision theory (EDT), and another one uses generalized thirding (GT) as a belief system and causal decision theory (CDT). Our findings relate those three solution concepts of a game to solution concepts of a polynomial maximization problem: global optima, optimal points with respect to subsets of variables and Karush–Kuhn–Tucker (KKT) points. Based on these correspondences, we are able to settle various complexity-theoretic questions on the computation of such strategies. For ex-ante optimality and (EDT,GDH)-equilibria, we obtain NP-hardness and inapproximability, and for (CDT,GT)-equilibria we obtain CLS-completeness results.

----

## [321] Error in the Euclidean Preference Model

**Authors**: *Luke Thorburn, Maria Polukarov, Carmine Ventre*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/322](https://doi.org/10.24963/ijcai.2023/322)

**Abstract**:

Spatial models of preference, in the form of vector embeddings, are learned by many deep learning and multiagent systems, including recommender systems. Often these models are assumed to approximate a Euclidean structure, where an individual prefers alternatives positioned closer to their "ideal point", as measured by the Euclidean metric. However, previous work has shown there are ordinal preference profiles that cannot be represented with this structure if the Euclidean space has two fewer dimensions than there are individuals or alternatives. We extend this result, showing that there are situations in which almost all preference profiles cannot be represented with the Euclidean model, and derive a theoretical lower bound on the expected error when using the Euclidean model to approximate non-Euclidean preference profiles. Our results have implications for the interpretation and use of vector embeddings, because in some cases close approximation of arbitrary, true ordinal relationships can be expected only if the dimensionality of the embeddings is a substantial fraction of the number of entities represented.

----

## [322] Maximin-Aware Allocations of Indivisible Chores with Symmetric and Asymmetric Agents

**Authors**: *Tianze Wei, Bo Li, Minming Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/323](https://doi.org/10.24963/ijcai.2023/323)

**Abstract**:

The real-world deployment of fair allocation algorithms usually involves a heterogeneous population of users, which makes it challenging for the users to get complete knowledge of the allocation except for their own bundles. Recently, a new fairness notion, maximin-awareness (MMA) was proposed and it guarantees that every agent is not the worst-off one, no matter how the items that are not allocated to this agent are distributed. We adapt and generalize this notion to the case of indivisible chores and when the agents may have arbitrary weights. Due to the inherent difficulty of MMA, we also consider its up to one and up to any relaxations. A string of results on the existence and computation of MMA related fair allocations, and their connections to existing fairness concepts is given.

----

## [323] Ordinal Hedonic Seat Arrangement under Restricted Preference Domains: Swap Stability and Popularity

**Authors**: *Anaëlle Wilczynski*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/324](https://doi.org/10.24963/ijcai.2023/324)

**Abstract**:

We study a variant of hedonic games, called hedonic seat arrangements in the literature, where the goal is not to partition the agents into coalitions but to assign them to vertices of a given graph; their satisfaction is then based on the subset of agents in their neighborhood. We focus on ordinal hedonic seat arrangements where the preferences over neighborhoods are deduced from ordinal preferences over single agents and a given preference extension. In such games and for different types of preference restrictions and extensions, we investigate the existence of arrangements satisfying stability w.r.t. swaps of positions in the graph or the well-known optimality concept of popularity.

----

## [324] Truthful Auctions for Automated Bidding in Online Advertising

**Authors**: *Yidan Xing, Zhilin Zhang, Zhenzhe Zheng, Chuan Yu, Jian Xu, Fan Wu, Guihai Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/325](https://doi.org/10.24963/ijcai.2023/325)

**Abstract**:

Automated bidding, an emerging intelligent decision-making paradigm powered by machine learning, has become popular in online advertising. Advertisers in automated bidding evaluate the cumulative utilities and have private financial constraints over multiple ad auctions in a long-term period. Based on these distinct features, we consider a new ad auction model for automated bidding: the values of advertisers are public while the financial constraints, such as budget and return on investment (ROI) rate, are private types. We derive the truthfulness conditions with respect to private constraints for this multi-dimensional setting, and demonstrate any feasible allocation rule could be equivalently reduced to a series of non-decreasing functions on budget. However, the resulted allocation mapped from these non-decreasing functions generally follows an irregular shape, making it difficult to obtain a closed-form expression for the auction objective. To overcome this design difficulty, we propose a family of truthful automated bidding auction with personalized rank scores, similar to the Generalized Second-Price (GSP) auction. The intuition behind our design is to leverage personalized rank scores as the criteria to allocate items, and compute a critical ROI to transforms the constraints on budget to the same dimension as ROI. The experimental results demonstrate that the proposed auction mechanism outperforms the widely used ad auctions, such as first-price auction and second-price auction, in various automated bidding environments.

----

## [325] Approximate Envy-Freeness in Graphical Cake Cutting

**Authors**: *Sheung Man Yuen, Warut Suksompong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/326](https://doi.org/10.24963/ijcai.2023/326)

**Abstract**:

We study the problem of fairly allocating a divisible resource in the form of a graph, also known as graphical cake cutting. Unlike for the canonical interval cake, a connected envy-free allocation is not guaranteed to exist for a graphical cake. We focus on the existence and computation of connected allocations with low envy. For general graphs, we show that there is always a 1/2-additive-envy-free allocation and, if the agents' valuations are identical, a (2+\epsilon)-multiplicative-envy-free allocation for any \epsilon > 0. In the case of star graphs, we obtain a multiplicative factor of 3+\epsilon for arbitrary valuations and 2 for identical valuations. We also derive guarantees when each agent can receive more than one connected piece. All of our results come with efficient algorithms for computing the respective allocations.

----

## [326] Incentive-Compatible Selection for One or Two Influentials

**Authors**: *Yuxin Zhao, Yao Zhang, Dengji Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/327](https://doi.org/10.24963/ijcai.2023/327)

**Abstract**:

Selecting influentials in networks against strategic manipulations has attracted many researchers' attention and it also has many practical applications. Here, we aim to select one or two influentials in terms of progeny (the influential power) and prevent agents from manipulating their edges (incentive compatibility). The existing studies mostly focused on selecting a single influential for this setting. Zhang et al. [2021] studied the problem of selecting one agent and proved an upper bound of 1/(1+ln2) to approximate the optimal selection. In this paper, we first design a mechanism to actually reach the bound. Then, we move this forward to choosing two agents and propose a mechanism to achieve an approximation ratio of (3+ln2)/(4(1+ln2)) (approx. 0.54).

----

## [327] Can You Improve My Code? Optimizing Programs with Local Search

**Authors**: *Fatemeh Abdollahi, Saqib Ameen, Matthew E. Taylor, Levi H. S. Lelis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/328](https://doi.org/10.24963/ijcai.2023/328)

**Abstract**:

This paper introduces a local search method for improving an existing program with respect to a measurable objective. Program Optimization with Locally Improving Search (POLIS) exploits the structure of a program, defined by its lines. POLIS improves a single line of the program while keeping the remaining lines fixed, using existing brute-force synthesis algorithms, and continues iterating until it is unable to improve the program's performance. POLIS was evaluated with a 27-person user study, where participants wrote programs attempting to maximize the score of two single-agent games: Lunar Lander and Highway. POLIS was able to substantially improve the participants' programs with respect to the game scores. A proof-of-concept demonstration on existing Stack Overflow code measures applicability in real-world problems. These results suggest that POLIS could  be used as a helpful programming assistant for programming problems with measurable objectives.

----

## [328] Sequence Learning Using Equilibrium Propagation

**Authors**: *Malyaban Bal, Abhronil Sengupta*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/329](https://doi.org/10.24963/ijcai.2023/329)

**Abstract**:

Equilibrium Propagation (EP) is a powerful and more bio-plausible alternative to conventional learning frameworks such as backpropagation. The effectiveness of EP stems from the fact that it relies only on local computations and requires solely one kind of computational unit during both of its training phases, thereby enabling greater applicability in domains such as bio-inspired neuromorphic computing. The dynamics of the model in EP is governed by an energy function and the internal states of the model consequently converge to a steady state following the state transition rules defined by the same. However, by definition, EP requires the input to the model (a convergent RNN) to be static in both the phases of training. Thus it is not possible to design a model for sequence classification using EP with an LSTM or GRU like architecture. In this paper, we leverage recent developments in modern hopfield networks to further understand energy based models and develop solutions for complex sequence classification tasks using EP while satisfying its convergence criteria and maintaining its theoretical similarities with recurrent backpropagation. We explore the possibility of integrating modern hopfield networks as an attention mechanism with convergent RNN models used in EP, thereby extending its applicability for the first time on two different sequence classification tasks in natural language processing viz. sentiment analysis (IMDB dataset) and natural language inference (SNLI dataset). Our implementation source code is available at https://github.com/NeuroCompLab-psu/EqProp-SeqLearning.

----

## [329] Towards Collaborative Plan Acquisition through Theory of Mind Modeling in Situated Dialogue

**Authors**: *Cristian-Paul Bara, Ziqiao Ma, Yingzhuo Yu, Julie Shah, Joyce Chai*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/330](https://doi.org/10.24963/ijcai.2023/330)

**Abstract**:

Collaborative tasks often begin with partial task knowledge and incomplete plans from each partner. 
To complete these tasks, partners need to engage in situated communication with their partners and coordinate their partial plans towards a complete plan to achieve a joint task goal. 
While such collaboration seems effortless in a human-human team, it is highly challenging for human-AI collaboration. 
To address this limitation, this paper takes a step towards Collaborative Plan Acquisition, where humans and agents strive to learn and communicate with each other to acquire a complete plan for joint tasks. 
Specifically, we formulate a novel problem for agents to predict the missing task knowledge for themselves and for their partners based on rich perceptual and dialogue history. 
We extend a situated dialogue benchmark for symmetric collaborative tasks in a 3D blocks world and investigate computational strategies for plan acquisition. 
Our empirical results suggest that predicting the partner's missing knowledge is a more viable approach than predicting one's own. 
We show that explicit modeling of the partner's dialogue moves and mental states produces improved and more stable results than without.
These results provide insight for future AI agents that 
can predict what knowledge their partner is missing and, therefore, can proactively communicate such information to help the partner acquire such missing knowledge toward a common understanding of joint tasks.

----

## [330] Sketch Recognition via Part-based Hierarchical Analogical Learning

**Authors**: *Kezhen Chen, Kenneth D. Forbus, Balaji Vasan Srinivasan, Niyati Chhaya, Madeline Usher*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/331](https://doi.org/10.24963/ijcai.2023/331)

**Abstract**:

Sketch recognition has been studied for decades, but it is far from solved. Drawing styles are highly variable across people and adapting to idiosyncratic visual expressions requires data-efficient learning. Explainability also matters, so that users can see why a system got confused about something. This paper introduces a novel part-based approach for sketch recognition, based on hierarchical analogical learning, a new method to apply analogical learning to qualitative representations. Given a sketched object, our system automatically segments it into parts and constructs multi-level qualitative representations of them. Our approach performs analogical generalization at multiple levels of part descriptions and uses coarse-grained results to guide interpretation at finer levels. Experiments on the Berlin TU dataset and the Coloring Book Objects dataset show that the system can learn explainable models in a data-efficient manner.

----

## [331] Black-Box Data Poisoning Attacks on Crowdsourcing

**Authors**: *Pengpeng Chen, Yongqiang Yang, Dingqi Yang, Hailong Sun, Zhijun Chen, Peng Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/332](https://doi.org/10.24963/ijcai.2023/332)

**Abstract**:

Understanding the vulnerability of label aggregation against data poisoning attacks is key to ensuring data quality in crowdsourced label collection. State-of-the-art attack mechanisms generally assume full knowledge of the aggregation models while failing to consider the flexibility of malicious workers in selecting which instances to label. Such a setup limits the applicability of the attack mechanisms and impedes further improvement of their success rate. This paper introduces a black-box data poisoning attack framework that finds the optimal strategies for instance selection and labeling to attack unknown label aggregation models in crowdsourcing. We formulate the attack problem on top of a generic formalization of label aggregation models and then introduce a substitution approach that attacks a substitute aggregation model in replacement of the unknown model. Through extensive validation on multiple real-world datasets, we demonstrate the effectiveness of both instance selection and model substitution in improving the success rate of attacks.

----

## [332] TDG4Crowd: Test Data Generation for Evaluation of Aggregation Algorithms in Crowdsourcing

**Authors**: *Yili Fang, Chaojie Shen, Huamao Gu, Tao Han, Xinyi Ding*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/333](https://doi.org/10.24963/ijcai.2023/333)

**Abstract**:

In crowdsourcing, existing efforts mainly use real datasets collected from crowdsourcing as test datasets to evaluate the effectiveness of aggregation algorithms. However, these work ignore the fact that the datasets obtained by crowdsourcing are usually sparse and imbalanced due to limited budget. As a result, applying the same aggregation algorithm on different datasets often show contradicting conclusions. For example, on the RTE dataset, Dawid and Skene model performs significantly better than Majority Voting, while on the LableMe dataset, the experiments give the opposite conclusion. It is challenging to obtain comprehensive and balanced datasets at a low cost. To our best knowledge, little effort have been made to the fair evaluation of aggregation algorithms. To fill in this gap, we propose a novel method named TDG4Crowd  that can automatically generate comprehensive and balanced datasets. Using Kullback Leibler divergence and Kolmogorovâ€“Smirnov test, the experiment results show the superior of our method compared with others. Aggregation algorithms also perform more consistently on the synthetic datasets generated using our method.

----

## [333] Enhancing Efficient Continual Learning with Dynamic Structure Development of Spiking Neural Networks

**Authors**: *Bing Han, Feifei Zhao, Yi Zeng, Wenxuan Pan, Guobin Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/334](https://doi.org/10.24963/ijcai.2023/334)

**Abstract**:

Children possess the ability to learn multiple cognitive tasks sequentially, which is a major challenge toward the long-term goal of artificial general intelligence. Existing continual learning frameworks are usually applicable to Deep Neural Networks (DNNs) and lack the exploration on more brain-inspired, energy-efficient Spiking Neural Networks (SNNs). Drawing on continual learning mechanisms during child growth and development, we propose Dynamic Structure Development of Spiking Neural Networks (DSD-SNN) for efficient and adaptive continual learning. When learning a sequence of tasks, the DSD-SNN dynamically assigns and grows new neurons to new tasks and prunes redundant neurons, thereby increasing memory capacity and reducing computational overhead. In addition, the overlapping shared structure helps to quickly leverage all acquired knowledge to new tasks, empowering a single network capable of supporting multiple incremental tasks (without the separate sub-network mask for each task). We validate the effectiveness of the proposed model on multiple class incremental learning and task incremental learning benchmarks. Extensive experiments demonstrated that our model could significantly improve performance, learning speed and memory capacity, and reduce computational overhead. Besides, our DSD-SNN model achieves comparable performance with the DNNs-based methods, and significantly outperforms the state-of-the-art (SOTA) performance for existing SNNs-based continual learning methods.

----

## [334] Learnable Surrogate Gradient for Direct Training Spiking Neural Networks

**Authors**: *Shuang Lian, Jiangrong Shen, Qianhui Liu, Ziming Wang, Rui Yan, Huajin Tang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/335](https://doi.org/10.24963/ijcai.2023/335)

**Abstract**:

Spiking neural networks (SNNs) have increasingly drawn massive research attention due to biological interpretability and efficient computation. Recent achievements are devoted to utilizing the surrogate gradient (SG) method to avoid the dilemma of non-differentiability of spiking activity to directly train SNNs by backpropagation. However, the fixed width of the SG leads to gradient vanishing and mismatch problems, thus limiting the performance of directly trained SNNs. In this work, we propose a novel perspective to unlock the width limitation of SG, called the learnable surrogate gradient (LSG) method. The LSG method modulates the width of SG according to the change of the distribution of the membrane potentials, which is identified to be related to the decay factors based on our theoretical analysis. Then we introduce the trainable decay factors to implement the LSG method, which can optimize the width of SG automatically during training to avoid the gradient vanishing and mismatch problems caused by the limited width of SG. We evaluate the proposed LSG method on both image and neuromorphic datasets. Experimental results show that the LSG method can effectively alleviate the blocking of gradient propagation caused by the limited width of SG when training deep SNNs directly. Meanwhile, the LSG method can help SNNs achieve competitive performance on both latency and accuracy.

----

## [335] A Hierarchical Approach to Population Training for Human-AI Collaboration

**Authors**: *Yi Loo, Chen Gong, Malika Meghjani*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/336](https://doi.org/10.24963/ijcai.2023/336)

**Abstract**:

A major challenge for deep reinforcement learning (DRL) agents is to collaborate with novel partners that were not encountered by them during the training phase. This is specifically worsened by an increased variance in action responses when the DRL agents collaborate with human partners due to the lack of consistency in human behaviors. Recent work have shown that training a single agent as the best response to a diverse population of training partners significantly increases an agent's robustness to novel partners. We further enhance the population-based training approach by introducing a Hierarchical Reinforcement Learning (HRL) based method for Human-AI Collaboration. Our agent is able to learn multiple best-response policies as its low-level policy while at the same time, it learns a high-level policy that acts as a manager which allows the agent to dynamically switch between the low-level best-response policies based on its current partner. We demonstrate that our method is able to dynamically adapt to novel partners of different play styles and skill levels in the 2-player collaborative Overcooked game environment. We also conducted a human study in the same environment to test the effectiveness of our method when partnering with real human subjects. Code is available at https://gitlab.com/marvl-hipt/hipt.

----

## [336] Strategic Adversarial Attacks in AI-assisted Decision Making to Reduce Human Trust and Reliance

**Authors**: *Zhuoran Lu, Zhuoyan Li, Chun-Wei Chiang, Ming Yin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/337](https://doi.org/10.24963/ijcai.2023/337)

**Abstract**:

With the increased integration of AI technologies in human decision making processes, adversarial attacks on AI models become a greater concern than ever before as they may significantly hurt humans’ trust in AI models and decrease the effectiveness of human-AI collaboration. While many adversarial attack methods have been proposed to decrease the performance of an AI model, limited attention has been paid on understanding how these attacks will impact the human decision makers interacting with the model, and accordingly, how to strategically deploy adversarial attacks to maximize the reduction of human trust and reliance. In this paper, through a human-subject experiment, we first show that in AI-assisted decision making, the timing of the attacks largely influences how much humans decrease their trust in and reliance on AI—the decrease is particularly salient when attacks occur on decision making tasks that humans are highly confident themselves. Based on these insights, we next propose an algorithmic framework to infer the human decision maker’s hidden trust in the AI model and dynamically decide when the attacker should launch an attack to the model. Our evaluations show that following the proposed approach, attackers deploy more efficient attacks and achieve higher utility than adopting other baseline strategies.

----

## [337] Learning Heuristically-Selected and Neurally-Guided Feature for Age Group Recognition Using Unconstrained Smartphone Interaction

**Authors**: *Yingmao Miao, Qiwei Tian, Chenhao Lin, Tianle Song, Yajie Zhou, Junyi Zhao, Shuxin Gao, Minghui Yang, Chao Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/338](https://doi.org/10.24963/ijcai.2023/338)

**Abstract**:

Owing to the boom of smartphone industries, the expansion of phone users has also been significant. Besides adults, children and elders have also begun to join the population of daily smartphone users. Such an expansion indeed facilitates the further exploration of the versatility and flexibility of digitization. However, these new users may also be susceptible to issues such as addiction, fraud, and insufficient accessibility. To fully utilize the capability of mobile devices without breaching personal privacy, we build the first corpus for age group recognition on smartphones with more than 1,445,087 unrestricted actions from 2,100 subjects. Then a series of heuristically-selected and neurally-guided features are proposed to increase the separability of the above dataset. Finally, we develop AgeCare, the first implicit and continuous system incorporated with bottom-to-top functionality without any restriction on user-phone interaction scenarios, for accurate age group recognition and age-tailored assistance on smartphones. Our system performs impressively well on this dataset and significantly surpasses the state-of-the-art methods.

----

## [338] Learning When to Advise Human Decision Makers

**Authors**: *Gali Noti, Yiling Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/339](https://doi.org/10.24963/ijcai.2023/339)

**Abstract**:

Artificial intelligence (AI) systems are increasingly used for providing advice to facilitate human decision making in a wide range of domains, such as healthcare, criminal justice, and finance. Motivated by limitations of the current practice where algorithmic advice is provided to human users as a constant element in the decision-making pipeline, in this paper we raise the question of when should algorithms provide advice? We propose a novel design of AI systems in which the algorithm interacts with the human user in a two-sided manner and aims to provide advice only when it is likely to be beneficial for the user in making their decision. The results of a large-scale experiment show that our advising approach manages to provide advice at times of need and to significantly improve human decision making compared to fixed, non-interactive, advising approaches. This approach has additional advantages in facilitating human learning, preserving complementary strengths of human decision makers, and leading to more positive responsiveness to the advice.

----

## [339] A Low Latency Adaptive Coding Spike Framework for Deep Reinforcement Learning

**Authors**: *Lang Qin, Rui Yan, Huajin Tang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/340](https://doi.org/10.24963/ijcai.2023/340)

**Abstract**:

In recent years, spiking neural networks (SNNs) have been used in reinforcement learning (RL) due to their low power consumption and event-driven features. However, spiking reinforcement learning (SRL), which suffers from fixed coding methods, still faces the problems of high latency and poor versatility. In this paper, we use learnable matrix multiplication to encode and decode spikes, improving the flexibility of the coders and thus reducing latency. Meanwhile, we train the SNNs using the direct training method and use two different structures for online and offline RL algorithms, which gives our model a wider range of applications. Extensive experiments have revealed that our method achieves optimal performance with ultra-low latency (as low as 0.8% of other SRL methods) and excellent energy efficiency (up to 5X the DNNs) in different algorithms and different environments.

----

## [340] Cognitively Inspired Learning of Incremental Drifting Concepts

**Authors**: *Mohammad Rostami, Aram Galstyan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/341](https://doi.org/10.24963/ijcai.2023/341)

**Abstract**:

Humans continually expand their learned knowledge to new domains and learn new concepts without any interference with past learned experiences. In contrast, machine learning models perform poorly in a continual learning setting, where input data distribution changes over time. Inspired by the nervous system learning mechanisms, we develop a computational model that enables a deep neural network to learn new concepts and expand its learned knowledge to new domains incrementally in a continual learning setting. We rely on the Parallel Distributed Processing theory to encode abstract concepts in an embedding space in terms of a multimodal distribution. This embedding space is modeled by internal data representations in a hidden network layer. We also leverage the Complementary Learning Systems theory to equip the model with a memory mechanism to overcome catastrophic forgetting through implementing pseudo-rehearsal. Our model can generate pseudo-data points for experience replay and accumulate new experiences to past learned experiences without causing cross-task interference.

----

## [341] A New ANN-SNN Conversion Method with High Accuracy, Low Latency and Good Robustness

**Authors**: *Bingsen Wang, Jian Cao, Jue Chen, Shuo Feng, Yuan Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/342](https://doi.org/10.24963/ijcai.2023/342)

**Abstract**:

Due to the advantages of low energy consumption, high robustness and fast inference speed, Spiking Neural Networks (SNNs), with good biological interpretability and the potential to be applied on neuromorphic hardware, are regarded as the third generation of Artificial Neural Networks (ANNs). Despite having so many advantages, the biggest challenge encountered by spiking neural networks is training difficulty caused by the non-differentiability of spike signals. ANN-SNN conversion is an effective method that solves the training difficulty by converting parameters in ANNs to those in SNNs through a specific algorithm. However, the ANN-SNN conversion method also suffers from accuracy degradation and long inference time. In this paper, we reanalyzed the relationship between Integrate-and-Fire (IF) neuron model and ReLU activation function, proposed a StepReLU activation function more suitable for SNNs under membrane potential encoding, and used it to train ANNs. Then we converted the ANNs to SNNs with extremely small conversion error and introduced leakage mechanism to the SNNs and get the final models, which have high accuracy, low latency and good robustness, and have achieved the state-of-the-art performance on various datasets such as CIFAR and ImageNet.

----

## [342] The Effects of AI Biases and Explanations on Human Decision Fairness: A Case Study of Bidding in Rental Housing Markets

**Authors**: *Xinru Wang, Chen Liang, Ming Yin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/343](https://doi.org/10.24963/ijcai.2023/343)

**Abstract**:

The use of AI-based decision aids in diverse domains has inspired many empirical investigations into how AI models’ decision recommendations impact humans’ decision accuracy in AI-assisted decision making, while explorations on the impacts on humans’ decision fairness are largely lacking despite their clear importance. In this paper, using a real-world business decision making scenario—bidding in rental housing markets—as our testbed, we present an experimental study on understanding how the bias level of the AI-based decision aid as well as the provision of AI explanations affect the fairness level of humans’ decisions, both during and after their usage of the decision aid. Our results suggest that when people are assisted by an AI-based decision aid, both the higher level of racial biases the decision aid exhibits and surprisingly, the presence of AI explanations, result in more unfair human decisions across racial groups. Moreover, these impacts are partly made through triggering humans’ “disparate interactions” with AI. However, regardless of the AI bias level and the presence of AI explanations, when people return to make independent decisions after their usage of the AI-based decision aid, their decisions no longer exhibit significant unfairness across racial groups.

----

## [343] Spatial-Temporal Self-Attention for Asynchronous Spiking Neural Networks

**Authors**: *Yuchen Wang, Kexin Shi, Chengzhuo Lu, Yuguo Liu, Malu Zhang, Hong Qu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/344](https://doi.org/10.24963/ijcai.2023/344)

**Abstract**:

The brain-inspired spiking neural networks (SNNs) are receiving increasing attention due to their asynchronous event-driven characteristics and low power consumption. As attention mechanisms recently become an indispensable part of sequence dependence modeling, the combination of SNNs and attention mechanisms holds great potential for energy-efficient and high-performance computing paradigms. However, the existing works cannot benefit from both temporal-wise attention and the asynchronous characteristic of SNNs. To fully leverage the advantages of both SNNs and attention mechanisms, we propose an SNNs-based spatial-temporal self-attention (STSA) mechanism, which calculates the feature dependence across the time and space domains without destroying the asynchronous transmission properties of SNNs. To further improve the performance, we also propose a spatial-temporal relative position bias (STRPB) for STSA to consider the spatiotemporal position of spikes. Based on the STSA and STRPB, we construct a spatial-temporal spiking Transformer framework, named STS-Transformer, which is powerful and enables SNNs to work in an asynchronous event-driven manner. Extensive experiments are conducted on popular neuromorphic datasets and speech datasets, including DVS128 Gesture, CIFAR10-DVS, and Google Speech Commands, and our experimental results can outperform other state-of-the-art models.

----

## [344] Preferences and Constraints in Abstract Argumentation

**Authors**: *Gianvincenzo Alfano, Sergio Greco, Francesco Parisi, Irina Trubitsyna*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/345](https://doi.org/10.24963/ijcai.2023/345)

**Abstract**:

In recent years there has been an increasing interest in extending Dung's framework to facilitate the knowledge representation and reasoning process.
In this paper, we present an extension of Abstract Argumentation Framework (AF) that allows for the representation of preferences over arguments' truth values (3-valued preferences).
For instance, we can express a preference stating that extensions where argument a is false (i.e. defeated) are preferred to extensions where argument b is false. 
Interestingly, such a framework generalizes the well-known Preference-based AF  with no additional cost in terms of computational complexity for most of the classical argumentation semantics.
Then, we further extend AF by considering both (3-valued) preferences and 3-valued constraints, that is constraints of the form \varphi \Rightarrow v or v \Rightarrow \varphi, where \varphi is a logical formula and v is a 3-valued truth value. 
After investigating the complexity of the resulting framework,as both constraints and preferences may represent subjective knowledge of agents, 
we extend our framework by considering multiple agents and study the complexity of deciding acceptance of arguments in this context.

----

## [345] Leveraging Argumentation for Generating Robust Sample-based Explanations

**Authors**: *Leila Amgoud, Philippe Muller, Henri Trenquier*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/346](https://doi.org/10.24963/ijcai.2023/346)

**Abstract**:

Explaining predictions made by inductive classifiers has become crucial with the rise of complex models acting more and more as black-boxes. 
Abductive explanations are one of the most popular types of explanations that are provided for the purpose. They highlight feature-values that  
are sufficient for making predictions. In the literature, they are generated by exploring the whole feature space, which is unreasonable in practice. 
This paper solves the problem by introducing explanation functions that generate abductive explanations from a sample of instances. It shows 
that such functions should be defined with great care since they cannot satisfy two desirable properties at the same time, namely existence of 
explanations for every individual decision (success) and correctness of explanations (coherence). The paper provides a parameterized family of 
argumentation-based explanation functions, each of which satisfies one of the two properties. It studies their formal properties and their experimental 
behaviour on different datasets.

----

## [346] Abstraction of Nondeterministic Situation Calculus Action Theories

**Authors**: *Bita Banihashemi, Giuseppe De Giacomo, Yves Lespérance*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/347](https://doi.org/10.24963/ijcai.2023/347)

**Abstract**:

We develop a general framework for abstracting the behavior of an agent that operates in a nondeterministic domain, i.e., where the agent does not control
the outcome of the nondeterministic actions, based on the nondeterministic situation calculus and the ConGolog programming language. We assume that
we have both an abstract and a concrete nondeterministic basic action theory, and a refinement mapping which  specifies how abstract actions, decomposed into agent actions and environment reactions, are implemented by concrete ConGolog programs. This new setting supports strategic reasoning and strategy synthesis, by allowing us to quantify separately on agent actions and environment reactions. We show that if the agent has a (strong FOND) plan/strategy to achieve a goal/complete a task at the abstract level, and it can always execute the nondeterministic abstract actions to completion at the concrete level, then there exist a refinement of it that is a (strong FOND) plan/strategy to achieve the refinement of the goal/task at the concrete level.

----

## [347] Bipolar Abstract Dialectical Frameworks Are Covered by Kleene's Three-valued Logic

**Authors**: *Ringo Baumann, Maximilian Heinrich*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/348](https://doi.org/10.24963/ijcai.2023/348)

**Abstract**:

Abstract dialectical frameworks (ADFs) are one of the most powerful generalizations of classical Dung-style argumentation frameworks (AFs).
The additional expressive power comes with an increase in computational complexity, namely one level up in the polynomial hierarchy in comparison to
their AF counterparts. However, there is one important subclass, so-called bipolar ADFs (BADFs) which are as complex as classical AFs while offering strictly more modeling capacities. This property makes BADFs very attractive from a knowledge representation point of view and is the main reason why this class has received much attention recently. The semantics of ADFs rely on the Gamma-operator which takes as an input a three-valued interpretation and returns a new one. However, in order to obtain the output the original definition requires to consider any two-valued completion of a given three-valued interpretation. In this paper we formally prove that in case of BADFs we may bypass the computationally intensive procedure via applying Kleene's three-valued logic K. We therefore introduce the so-called bipolar disjunctive normal form which is simply a disjunctive normal form where any used atom possesses either a positive or a negative polarity. We then show that: First, this normal form is expressive enough to represent any BADF and secondly, the computation can be done via Kleene's K instead of dealing with two-valued completions. Inspired by the main correspondence result we present some first experiments showing the computational benefit of using Kleene.

----

## [348] REPLACE: A Logical Framework for Combining Collective Entity Resolution and Repairing

**Authors**: *Meghyn Bienvenu, Gianluca Cima, Víctor Gutiérrez-Basulto*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/349](https://doi.org/10.24963/ijcai.2023/349)

**Abstract**:

This paper considers the problem of querying dirty databases, which may contain both erroneous facts and multiple names for the same entity. While both of these data quality issues have been widely studied in isolation, our contribution is a holistic framework for jointly deduplicating and repairing data. Our REPLACE framework follows a declarative approach, utilizing logical rules to specify under which conditions a pair of entity references can or must be merged and logical constraints to specify consistency requirements. The semantics defines a space of solutions, each consisting of a set of merges to perform and a set of facts to delete, which can be further refined by applying optimality criteria. As there may be multiple optimal solutions, we use classical notions of possible and certain query answers to reason over the alternative solutions, and introduce a novel notion of most informative answer to obtain a more compact presentation of query results. We perform a detailed analysis of the data complexity of the central reasoning tasks of recognizing optimal solutions and (most informative) possible and certain answers, for each of the three notions of optimal solution and for both general and restricted specifications.

----

## [349] Augmenting Automated Spectrum Based Fault Localization for Multiple Faults

**Authors**: *Prantik Chatterjee, José Campos, Rui Abreu, Subhajit Roy*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/350](https://doi.org/10.24963/ijcai.2023/350)

**Abstract**:

Spectrum-based Fault Localization (SBFL) uses the coverage of test cases and their outcome (pass/fail) to predict the "suspiciousness'' of program components, e.g., lines of code. SBFL is, perhaps, the most successful fault localization technique due to its simplicity and scalability. However, SBFL heuristics do not perform well in scenarios where a program may have multiple faulty components. In this work, we propose a new algorithm that "augments'' previously proposed SBFL heuristics to produce a ranked list where faulty components ranked low by base SBFL metrics are ranked significantly higher. We implement our ideas in a tool, ARTEMIS, that attempts to "bubble up'' faulty components which are ranked lower by base SBFL metrics. We compare our technique to the most popular SBFL metrics and demonstrate statistically significant improvement in the developer effort for fault localization with respect to the basic strategies.

----

## [350] Automatic Verification for Soundness of Bounded QNP Abstractions for Generalized Planning

**Authors**: *Zhenhe Cui, Weidu Kuang, Yongmei Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/351](https://doi.org/10.24963/ijcai.2023/351)

**Abstract**:

Generalized planning (GP) studies the computation of general solutions for a set of planning problems. Computing general solutions with correctness guarantee has long been a key issue in GP. Abstractions are widely used to solve GP problems. For example, a popular abstraction model for GP is qualitative numeric planning (QNP), which extends classical planning with non-negative real variables that can be increased or decreased by some arbitrary amount. The refinement of correct solutions of sound abstractions are solutions with correctness guarantees for GP problems. More recent literature proposed a uniform abstraction framework for GP and gave model-theoretic definitions of sound and complete abstractions for GP problems. In this paper, based on the previous work, we explore automatic verification of sound abstractions for GP. Firstly, we present a proof-theoretic characterization for sound abstractions. Secondly, based on the characterization, we give a sufficient condition for sound abstractions with deterministic actions.   Then we study how to verify the sufficient condition when the abstraction models are bounded QNPs where integer variables can be incremented or decremented by one. To this end, we develop methods to handle counting and transitive closure, which are often used to define numerical variables. Finally, we implement a sound bounded QNP abstraction verification system and report experimental results on several domains.

----

## [351] On Translations between ML Models for XAI Purposes

**Authors**: *Alexis de Colnet, Pierre Marquis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/352](https://doi.org/10.24963/ijcai.2023/352)

**Abstract**:

In this paper, the succinctness of various ML models is studied. To be more precise, the existence of polynomial-time and polynomial-space translations between representation languages for classifiers is investigated. The languages that are considered include decision trees, random forests, several types of boosted trees, binary neural networks, Boolean multilayer perceptrons, and various logical representations of  binary classifiers. We provide a complete map indicating for every pair of languages C, C' whether or not a polynomial-time / polynomial-space translation exists from C to C'. We also explain how to take advantage of the resulting map for XAI purposes.

----

## [352] Description Logics with Pointwise Circumscription

**Authors**: *Federica Di Stefano, Magdalena Ortiz, Mantas Simkus*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/353](https://doi.org/10.24963/ijcai.2023/353)

**Abstract**:

Circumscription is one of the most powerful ways to extend Description Logics (DLs) with non-monotonic reasoning features, albeit with huge computational costs and undecidability in many cases. In this paper, we introduce pointwise circumscription for DLs, which is not only intuitive in terms of knowledge representation, but also provides a sound approximation of classic circumscription and has reduced computational complexity. Our main idea is to replace the second-order quantification step of classic circumscription with a series of (pointwise) local checks on all domain elements and their immediate neighbourhood. Our main positive results are for ontologies in DLs ALCIO and ALCI: we prove that for TBoxes of modal depth 1 (i.e. without nesting of existential or universal quantifiers) standard reasoning problems under pointwise circumscription are (co)NExpTime-complete and ExpTime-complete, respectively. The restriction of modal depth still yields a large class of ontologies useful in practice, and it is further justified by a strong undecidability result for pointwise circumscription with general TBoxes in ALCIO.

----

## [353] Parametrized Gradual Semantics Dealing with Varied Degrees of Compensation

**Authors**: *Dragan Doder, Leila Amgoud, Srdjan Vesic*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/354](https://doi.org/10.24963/ijcai.2023/354)

**Abstract**:

Compensation is a strategy that a semantics may follow when it faces  
dilemmas between quality and quantity of attackers. It allows several weak 
attacks to compensate one strong attack. It is  based on compensation degree, 
which is a tuple that indicates (i) to what extent an attack is weak and (ii) the 
number of weak attacks needed to compensate a strong one. 
Existing principles on compensation do not specify the parameters, thus it is unclear 
whether semantics satisfying them compensate at only one degree or several degrees, and which ones.
This paper proposes a parameterised family of gradual semantics, which 
unifies multiple semantics that share some principles but differ in their 
strategy regarding solving dilemmas. Indeed, we show that the two semantics taking 
the extreme values of the parameter favour respectively  quantity and quality,  while all 
the remaining ones compensate at some degree. We define three classes of compensation 
degrees and show that the novel family is able to compensate at all of them while 
none of the existing gradual semantics does.

----

## [354] Learning Small Decision Trees with Large Domain

**Authors**: *Eduard Eiben, Sebastian Ordyniak, Giacomo Paesani, Stefan Szeider*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/355](https://doi.org/10.24963/ijcai.2023/355)

**Abstract**:

One favors decision trees (DTs) of the smallest size or depth to facilitate explainability and interpretability. However, learning such an optimal DT from data is well-known to be NP-hard. To overcome this complexity barrier, Ordyniak and Szeider (AAAI 21) initiated the study of optimal DT learning under the parameterized complexity perspective. They showed that solution size (i.e., number of nodes or depth of the DT) is insufficient to obtain fixed-parameter tractability (FPT). Therefore, they proposed an FPT algorithm that utilizes two auxiliary parameters: the maximum difference (as a structural property of the data set) and maximum domain size. They left it as an open question of whether bounding the maximum domain size is necessary.

The main result of this paper answers this question. We present FPT algorithms for learning a smallest or  lowest-depth DT from data, with the only parameters solution size and maximum difference. Thus, our algorithm is significantly more potent than the one by Szeider and Ordyniak as it can handle problem inputs with features that range over unbounded domains. We also close several gaps concerning the quality of approximation one obtains by only considering DTs based on minimum support sets.

----

## [355] Explaining Answer-Set Programs with Abstract Constraint Atoms

**Authors**: *Thomas Eiter, Tobias Geibinger*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/356](https://doi.org/10.24963/ijcai.2023/356)

**Abstract**:

Answer-Set Programming (ASP) is a popular declarative reasoning and problem solving formalism. Due to the increasing interest in explainabilty, several explanation approaches have been developed for ASP. However, support for commonly used advanced language features of ASP, as for example aggregates or choice rules, is still mostly lacking. We deal with explaining ASP programs containing Abstract Constraint Atoms, which encompass the above features and others. We provide justifications for the presence, or absence, of an atom in a given answer-set. To this end, we introduce several formal notions of justification in this setting based on the one hand on a semantic characterisation utilising minimal partial models, and on the other hand on a more ruled-guided approach. We provide complexity results for checking and computing such justifications, and discuss how the semantic and syntactic approaches relate and can be jointly used to offer more insight.
Our results contribute to a basis for explaining commonly used language features and thus increase accessibility and usability of ASP as an AI tool.

----

## [356] Treewidth-Aware Complexity for Evaluating Epistemic Logic Programs

**Authors**: *Jorge Fandinno, Markus Hecher*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/357](https://doi.org/10.24963/ijcai.2023/357)

**Abstract**:

Logic programs are a popular formalism for encoding many problems relevant to knowledge representation and reasoning as well as artificial intelligence. However, for modeling rational behavior it is oftentimes required to represent the concepts of knowledge and possibility. Epistemic logic programs (ELPs) is such an extension that enables both concepts, which correspond to being true in all or some possible worlds or stable models. For these programs, the parameter treewidth has recently regained popularity. We present complexity results for the evaluation of key ELP fragments for treewidth, which are exponentially better than known results for full ELPs. Unfortunately, we prove that obtained runtimes can not be significantly improved, assuming the exponential time hypothesis. Our approach defines treewidth-aware reductions between quantified Boolean formulas and ELPs. We also establish
that the completion of a program, as used in modern solvers, can be turned treewidth-aware, thereby linearly preserving treewidth.

----

## [357] Quantitative Reasoning and Structural Complexity for Claim-Centric Argumentation

**Authors**: *Johannes Klaus Fichte, Markus Hecher, Yasir Mahmood, Arne Meier*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/358](https://doi.org/10.24963/ijcai.2023/358)

**Abstract**:

Argumentation is a well-established formalism for nonmonotonic reasoning and a vibrant area of research in AI. Claim-augmented argumentation frameworks (CAFs) have been introduced to deploy a conclusion-oriented perspective. CAFs expand argumentation frameworks by an additional step which involves retaining claims for an accepted set of arguments. We introduce a novel concept of a justification status for claims, a quantitative measure of extensions supporting a particular claim. The well-studied problems of credulous and skeptical reasoning can then be seen as simply the two endpoints of the spectrum when considered as a justification level of a claim. Furthermore, we explore the parameterized complexity of various reasoning problems for CAFs, including the quantitative reasoning for claim assertions. We begin by presenting a suitable graph representation that includes arguments and their associated claims. Our analysis includes the parameter treewidth, and we present decomposition-guided reductions between reasoning problems in CAF and the validity problem for QBF.

----

## [358] An Ensemble Approach for Automated Theorem Proving Based on Efficient Name Invariant Graph Neural Representations

**Authors**: *Achille Fokoue, Ibrahim Abdelaziz, Maxwell Crouse, Shajith Ikbal, Akihiro Kishimoto, Guilherme Lima, Ndivhuwo Makondo, Radu Marinescu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/359](https://doi.org/10.24963/ijcai.2023/359)

**Abstract**:

Using reinforcement learning for automated theorem proving has recently received much attention. Current approaches use representations of logical statements that often rely on the names used in these statements and, as a result, the models are generally not transferable from one domain to another. The size of these representations and whether to include the whole theory or part of it are other important decisions that affect the performance of these approaches as well as their runtime efficiency. In this paper, we present NIAGRA; an ensemble Name InvAriant Graph RepresentAtion. NIAGRA addresses this problem by using 1) improved Graph Neural Networks for learning name-invariant formula representations that is tailored for their unique characteristics and 2) an efficient ensemble approach for automated theorem proving. Our experimental evaluation shows state-of-the-art performance on multiple datasets from different domains with improvements up to 10% compared to the best learning-based approaches. Furthermore, transfer learning experiments show that our approach significantly outperforms other learning-based approaches by up to 28%.

----

## [359] Reverse Engineering of Temporal Queries Mediated by LTL Ontologies

**Authors**: *Marie Fortin, Boris Konev, Vladislav Ryzhikov, Yury Savateev, Frank Wolter, Michael Zakharyaschev*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/360](https://doi.org/10.24963/ijcai.2023/360)

**Abstract**:

In reverse engineering of database queries, we aim to construct a query from a given set of  answers and non-answers; it can then be used to explore the data further or as an explanation of the answers and non-answers. We investigate this query-by-example problem for queries formulated in positive fragments of linear temporal logic LTL over timestamped data, focusing on the design of suitable query languages and the combined and data complexity of deciding whether there exists a query in the given language that separates the given answers from non-answers. We consider both plain LTL queries and those mediated by LTL ontologies.

----

## [360] Disentanglement of Latent Representations via Causal Interventions

**Authors**: *Gaël Gendron, Michael Witbrock, Gillian Dobbie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/361](https://doi.org/10.24963/ijcai.2023/361)

**Abstract**:

The process of generating data such as images is controlled by independent and unknown factors of variation. The retrieval of these variables has been studied extensively in the disentanglement, causal representation learning, and independent component analysis fields. Recently, approaches merging these domains together have shown great success. Instead of directly representing the factors of variation, the problem of disentanglement can be seen as finding the interventions on one image that yield a change to a single factor. Following this assumption, we introduce a new method for disentanglement inspired by causal dynamics that combines causality theory with vector-quantized variational autoencoders. Our model considers the quantized vectors as causal variables and links them in a causal graph. It performs causal interventions on the graph and generates atomic transitions affecting a unique factor of variation in the image. We also introduce a new task of action retrieval that consists of finding the action responsible for the transition between two images. We test our method on standard synthetic and real-world disentanglement datasets. We show that it can effectively disentangle the factors of variation and perform precise interventions on high-level semantic attributes of an image without affecting its quality, even with imbalanced data distributions.

----

## [361] Safety Verification and Universal Invariants for Relational Action Bases

**Authors**: *Silvio Ghilardi, Alessandro Gianola, Marco Montali, Andrey Rivkin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/362](https://doi.org/10.24963/ijcai.2023/362)

**Abstract**:

Modeling and verification of dynamic systems operating over a relational representation of states are increasingly investigated problems in AI, Business Process Management and Database Theory. To make these systems amenable to verification, the amount of information stored in each state needs to be bounded, or restrictions are imposed on the preconditions and effects of actions. We lift these restrictions by introducing the framework of Relational Action Bases (RABs), which generalizes existing frameworks and in which unbounded relational states are evolved through actions that can (1) quantify both existentially and universally over the data, and (2) use arithmetic constraints. We then study parameterized safety of RABs via (approximated) SMT-based backward search, singling out essential meta-properties of the resulting procedure, and showing how it can be realized by an off-the-shelf combination of existing verification modules of the state-of-the-art MCMT model checker. We demonstrate the effectiveness of this approach on a benchmark of data-aware business processes. Finally, we show how universal invariants can be exploited to make this procedure fully correct.

----

## [362] Tractable Diversity: Scalable Multiperspective Ontology Management via Standpoint EL

**Authors**: *Lucía Gómez Álvarez, Sebastian Rudolph, Hannes Strass*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/363](https://doi.org/10.24963/ijcai.2023/363)

**Abstract**:

The tractability of the lightweight description logic EL has allowed for the construction of large and widely used ontologies that support semantic interoperability. However, comprehensive domains with a broad user base are often at odds with strong axiomatisations otherwise useful for inferencing, since these are usually context dependent and subject to diverging perspectives.

In this paper we introduce Standpoint EL, a multi-modal extension of EL that allows for the integrated representation of domain knowledge relative to diverse, possibly conflicting standpoints (or contexts), which can be hierarchically organised and put in relation to each other. We establish that Standpoint EL still exhibits EL's favourable PTime standard reasoning, whereas introducing additional features like empty standpoints, rigid roles, and nominals makes standard reasoning tasks intractable.

----

## [363] Ranking-based Argumentation Semantics Applied to Logical Argumentation

**Authors**: *Jesse Heyninck, Badran Raddaoui, Christian Straßer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/364](https://doi.org/10.24963/ijcai.2023/364)

**Abstract**:

In formal argumentation, a distinction can be made between extension-based semantics, where sets of
arguments are either (jointly) accepted or not, and ranking-based semantics, where grades of accept-
ability are assigned to arguments. Another important distinction is that between abstract approaches,
that abstract away from the content of arguments, and structured approaches, that specify a method
of constructing argument graphs on the basis of a knowledge base. While ranking-based semantics
have been extensively applied to abstract argumentation, few work has been done on ranking-based
semantics for structured argumentation. In this paper, we make a systematic investigation into the be-
haviour of ranking-based semantics applied to existing formalisms for structured argumentation. We
show that a wide class of ranking-based semantics gives rise to so-called culpability measures, and
are relatively robust to specific choices in argument construction methods.

----

## [364] Temporal Datalog with Existential Quantification

**Authors**: *Matthias Lanzinger, Markus Nissl, Emanuel Sallinger, Przemyslaw Andrzej Walega*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/365](https://doi.org/10.24963/ijcai.2023/365)

**Abstract**:

Existential rules, also known as tuple-generating dependencies (TGDs) or Datalog+/- rules, are heavily studied in the communities of  Knowledge Representation and Reasoning, Semantic Web, and Databases, due to their rich modelling capabilities. In this paper we consider TGDs in the temporal setting, by introducing and studying DatalogMTLE---an extension of metric temporal Datalog (DatalogMTL) obtained by allowing for existential rules in  programs.  We show that DatalogMTLE is undecidable even in the restricted cases of guarded and weakly-acyclic programs. To address this issue we introduce uniform semantics which, on the one hand, is well-suited for modelling temporal knowledge as it prevents from unintended value invention and, on the other hand, provides decidability of reasoning; in particular, it becomes 2-EXPSPACE-complete for weakly-acyclic programs but remains undecidable for guarded programs. We provide an implementation for the decidable case  and demonstrate its practical feasibility.
Thus we obtain an expressive, yet decidable,  rule-language and a system which is suitable for complex temporal reasoning with existential rules.

----

## [365] A Rule-Based Modal View of Causal Reasoning

**Authors**: *Emiliano Lorini*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/366](https://doi.org/10.24963/ijcai.2023/366)

**Abstract**:

We present a novel rule-based semantics for causal reasoning as well as a number of modal languages interpreted over it. They enable us to represent some fundamental concepts in the theory of causality including causal necessity and possibility, interventionist conditionals and Lewisian conditionals. We provide complexity results for the satisfiability checking and model checking problem for these modal languages. Moreover, we study the relationship between our rule-based semantics and the structural equation modeling (SEM) approach to causal reasoning, as well as between our rule-based semantics for causal conditionals and the standard semantics for belief base change.

----

## [366] Probabilistic Temporal Logic for Reasoning about Bounded Policies

**Authors**: *Nima Motamed, Natasha Alechina, Mehdi Dastani, Dragan Doder, Brian Logan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/367](https://doi.org/10.24963/ijcai.2023/367)

**Abstract**:

To build a theory of intention revision for agents operating in stochastic environments, we need a logic in which we can explicitly reason about their decision-making policies and those policies' uncertain outcomes. Towards this end, we propose PLBP, a novel probabilistic temporal logic for Markov Decision Processes that allows us to reason about policies of bounded size. The logic is designed so that its expressive power is sufficient for the intended applications, whilst at the same time possessing strong computational properties. We prove that the satisfiability problem for our logic is decidable, and that its model checking problem is PSPACE-complete. This allows us to e.g. algorithmically verify whether an agent's intentions are coherent, or whether a specific policy satisfies safety and/or liveness properties.

----

## [367] Shhh! The Logic of Clandestine Operations

**Authors**: *Pavel Naumov, Oliver Orejola*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/368](https://doi.org/10.24963/ijcai.2023/368)

**Abstract**:

An operation is called covert if it conceals the identity of the actor; it is called clandestine if the very fact that the operation is conducted is concealed. The paper proposes a formal semantics of clandestine operations and introduces a sound and complete logical system that describes the interplay between the distributed knowledge modality and a modality capturing coalition power to conduct clandestine operations.

----

## [368] The Parameterized Complexity of Finding Concise Local Explanations

**Authors**: *Sebastian Ordyniak, Giacomo Paesani, Stefan Szeider*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/369](https://doi.org/10.24963/ijcai.2023/369)

**Abstract**:

We consider the computational problem of finding a smallest local explanation (anchor) for classifying a given feature vector (example) by a black-box model.  After showing that the problem is NP-hard in general, we study various natural restrictions of the problem in terms of problem parameters to see whether these restrictions make the problem fixed-parameter tractable or not. We draw a detailed and systematic complexity landscape for combinations of parameters, including the size of the anchor, the size of the anchor's coverage, and parameters that capture structural aspects of the problem instance, including rank-width, twin-width, and maximum difference.

----

## [369] Relative Inconsistency Measures for Indefinite Databases with Denial Constraints

**Authors**: *Francesco Parisi, John Grant*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/370](https://doi.org/10.24963/ijcai.2023/370)

**Abstract**:

Handling conflicting information is an important challenge in AI. Measuring inconsistency is an approach that provides ways to quantify the severity of inconsistency and helps understanding the primary sources of conflicts. In particular, a relative inconsistency measure computes, by some criteria, the proportion of the knowledge base that is inconsistent. In this paper we investigate relative inconsistency measures for indefinite  databases, which allow for indefinite or partial information which is formally expressed by means of disjunctive tuples. We introduce a postulate-based definition of relative inconsistency measure for indefinite databases with denial constraints, and investigate the compliance of some relative inconsistency measures with rationality postulates for indefinite databases as well as for the special case of definite databases. Finally, we investigate the complexity of the problem of computing the value of the proposed relative inconsistency measures as well as of the problems of deciding whether the inconsistency value is lower than, greater than, or equal to a given threshold for indefinite and definite databases.

----

## [370] A Comparative Study of Ranking Formulas Based on Consistency

**Authors**: *Badran Raddaoui, Christian Straßer, Saïd Jabbour*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/371](https://doi.org/10.24963/ijcai.2023/371)

**Abstract**:

Ranking is ubiquitous in everyday life. This paper is concerned with the problem of ranking information of a knowledge base when this latter is possibly inconsistent. In particular, the key issue is to elicit a plausibility order on the formulas in an inconsistent knowledge base. We show how such ordering can be obtained by using only the inherent structure of the knowledge base. We start by introducing a principled way a reasonable ranking framework for formulas should satisfy. Then, a variety of ordering criteria have been explored to define plausibility order over formulas based on consistency. Finally, we study the behaviour of the different formula ranking semantics in terms of the proposed logical postulates as well as their (in)-compatibility.

----

## [371] On Discovering Interesting Combinatorial Integer Sequences

**Authors**: *Martin Svatos, Peter Jung, Jan Tóth, Yuyi Wang, Ondrej Kuzelka*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/372](https://doi.org/10.24963/ijcai.2023/372)

**Abstract**:

We study the problem of generating interesting integer sequences with a combinatorial interpretation. For this we introduce a two-step approach. In the first step, we generate first-order logic sentences which define some combinatorial objects, e.g., undirected graphs, permutations, matchings etc. In the second step, we use algorithms for lifted first-order model counting to generate integer sequences that count the objects encoded by the first-order logic formulas generated in the first step. For instance, if the first-order sentence defines permutations then the generated integer sequence is the sequence of factorial numbers n!. We demonstrate that our approach is able to generate interesting new sequences by showing that a non-negligible fraction of the automatically generated sequences can actually be found in the Online Encyclopaedia of Integer Sequences (OEIS) while generating many other similar sequences which are not present in OEIS and which are potentially interesting. A key technical contribution of our work is the method for generation of first-order logic sentences which is able to drastically prune the space of sentences by discarding large fraction of sentences which would lead to redundant integer sequences.

----

## [372] SAT-Based PAC Learning of Description Logic Concepts

**Authors**: *Balder ten Cate, Maurice Funk, Jean Christoph Jung, Carsten Lutz*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/373](https://doi.org/10.24963/ijcai.2023/373)

**Abstract**:

We propose bounded fitting as a scheme for learning
description logic concepts in the presence of ontologies. A main
advantage is that the resulting learning algorithms come with
theoretical guarantees regarding their generalization to unseen
examples in the sense of PAC learning. We prove that, in contrast,
several other natural learning algorithms fail to provide such
guarantees. As a further contribution, we present the system SPELL
which efficiently implements bounded fitting for the description
logic ELHr based on a SAT solver, and compare its performance to a
state-of-the-art learner.

----

## [373] Efficient Computation of General Modules for ALC Ontologies

**Authors**: *Hui Yang, Patrick Koopmann, Yue Ma, Nicole Bidoit*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/374](https://doi.org/10.24963/ijcai.2023/374)

**Abstract**:

We present a method for extracting general modules for ontologies formulated in the description logic ALC. A module for an ontology is an ideally substantially smaller ontology that preserves all entailments for a user-specified set of terms. As such, it has applications such as ontology reuse and ontology analysis. Different from classical modules, general modules may use axioms not explicitly present in the input ontology, which allows for additional conciseness. So far, general modules have only been investigated for lightweight description logics.
We present the first work that considers the more expressive description logic ALC. In particular, our contribution is a new method based on uniform interpolation supported by some new theoretical results. Our evaluation indicates that our general modules are often smaller than classical modules and uniform interpolants computed by the state-of-the-art, and compared with uniform interpolants, can be computed in significantly shorter time. Moreover, our method can be used for, and in fact, improves the computation of uniform interpolants and classical modules.

----

## [374] On the Paradox of Learning to Reason from Data

**Authors**: *Honghua Zhang, Liunian Harold Li, Tao Meng, Kai-Wei Chang, Guy Van den Broeck*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/375](https://doi.org/10.24963/ijcai.2023/375)

**Abstract**:

Logical reasoning is needed in a wide range of NLP tasks. Can a BERT model be trained end-to-end to solve logical reasoning problems presented in natural language? We attempt to answer this question in a confined problem space where there exists a set of parameters that perfectly simulates logical reasoning. We make observations that seem to contradict each other: BERT attains near-perfect accuracy on in-distribution test examples while failing to generalize to other data distributions over the exact same problem space. Our study provides an explanation for this paradox: instead of learning to emulate the correct reasoning function, BERT has, in fact, learned statistical features that inherently exist in logical reasoning problems. We also show that it is infeasible to jointly remove statistical features from data, illustrating the difficulty of learning to reason in general. Our result naturally extends to other neural models (e.g. T5) and unveils the fundamental difference between learning to reason and learning to achieve high performance on NLP benchmarks using statistical features.

----

## [375] A Multi-Modal Neural Geometric Solver with Textual Clauses Parsed from Diagram

**Authors**: *Mingliang Zhang, Fei Yin, Cheng-Lin Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/376](https://doi.org/10.24963/ijcai.2023/376)

**Abstract**:

Geometry problem solving (GPS) is a high-level mathematical reasoning requiring the capacities of multi-modal fusion and geometric knowledge application. Recently, neural solvers have shown great potential in GPS but still be short in diagram presentation and modal fusion. In this work, we convert diagrams into basic textual clauses to describe diagram features effectively, and propose a new neural solver called PGPSNet to fuse multi-modal information efficiently. Combining structural and semantic pre-training, data augmentation and self-limited decoding, PGPSNet is endowed with rich knowledge of geometry theorems and geometric representation, and therefore promotes geometric understanding and reasoning. In addition, to facilitate the research of GPS, we build a new large-scale and fine-annotated GPS dataset named PGPS9K, labeled with both fine-grained diagram annotation and interpretable solution program. Experiments on PGPS9K and an existing dataset Geometry3K validate the superiority of our method over the state-of-the-art neural solvers. Our code, dataset and appendix material are available at \url{https://github.com/mingliangzhang2018/PGPS}.

----

## [376] Enhancing Datalog Reasoning with Hypertree Decompositions

**Authors**: *Xinyue Zhang, Pan Hu, Yavor Nenov, Ian Horrocks*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/377](https://doi.org/10.24963/ijcai.2023/377)

**Abstract**:

Datalog reasoning based on the seminaive evaluation strategy evaluates rules using traditional join plans, which often leads to redundancy and inefficiency in practice, especially when the rules are complex. Hypertree decompositions help identify efficient query plans and reduce similar redundancy in query answering. However, it is unclear how this can be applied to materialisation and incremental reasoning with recursive Datalog programs. Moreover, hypertree decompositions require additional data structures and thus introduce nonnegligible overhead in both runtime and memory consumption. In this paper, we provide algorithms that exploit hypertree decompositions for the materialisation and incremental evaluation of Datalog programs. Furthermore, we combine this approach with standard Datalog reasoning algorithms in a modular fashion so that the overhead caused by the decompositions is reduced. Our empirical evaluation shows that, when the program contains complex rules, the combined approach is usually significantly faster than the baseline approach, sometimes by orders of magnitude.

----

## [377] Building Concise Logical Patterns by Constraining Tsetlin Machine Clause Size

**Authors**: *Kuruge Darshana Abeyrathna, Ahmed Abdulrahem Othman Abouzeid, Bimal Bhattarai, Charul Giri, Sondre Glimsdal, Ole-Christoffer Granmo, Lei Jiao, Rupsa Saha, Jivitesh Sharma, Svein Anders Tunheim, Xuan Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/378](https://doi.org/10.24963/ijcai.2023/378)

**Abstract**:

Tsetlin Machine (TM) is a logic-based machine learning approach with the crucial advantages of being transparent and hardware-friendly. While TMs match or surpass deep learning accuracy for an increasing number of applications, large clause pools tend to produce clauses with many literals (long clauses). As such, they become less interpretable. Further, longer clauses increase the switching activity of the clause logic in hardware, consuming more power. This paper introduces a novel variant of TM learning -- Clause Size Constrained TMs (CSC-TMs) --  where one can set a soft constraint on the clause size. As soon as a clause includes more literals than the constraint allows, it starts expelling literals. Accordingly, oversized clauses only appear transiently. To evaluate CSC-TM, we conduct classification, clustering, and regression experiments on tabular data, natural language text, images, and board games. Our results show that CSC-TM maintains accuracy with up to 80 times fewer literals. Indeed, the accuracy increases with shorter clauses for TREC and BBC Sports. After the accuracy peaks, it drops gracefully as the clause size approaches one literal. We finally analyze CSC-TM power consumption and derive new convergence properties.

----

## [378] GIDnets: Generative Neural Networks for Solving Inverse Design Problems via Latent Space Exploration

**Authors**: *Carlo Adornetto, Gianluigi Greco*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/379](https://doi.org/10.24963/ijcai.2023/379)

**Abstract**:

In a number of different fields, including Engeneering, Chemistry and Physics, the design of technological tools and device structures is increasingly supported by deep-learning based methods, which provide suggestions on crucial architectural choices based on the properties that these tools and structures should exhibit. The paper proposes a novel architecture, named GIDnet, to address this inverse design problem, which is based on exploring a suitably defined latent space associated with the possible designs. Among its distinguishing features, GIDnet is capable of identifying the most appropriate starting point for the exploration and of likely converging into a point corresponding to a design that is a feasible one. Results of a thorough experimental activity evidence that GIDnet outperforms earlier approaches in the literature.

----

## [379] CROP: Towards Distributional-Shift Robust Reinforcement Learning Using Compact Reshaped Observation Processing

**Authors**: *Philipp Altmann, Fabian Ritz, Leonard Feuchtinger, Jonas Nüßlein, Claudia Linnhoff-Popien, Thomy Phan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/380](https://doi.org/10.24963/ijcai.2023/380)

**Abstract**:

The safe application of reinforcement learning (RL) requires generalization from limited training data to unseen scenarios. Yet, fulfilling tasks under changing circumstances is a key challenge in RL. Current state-of-the-art approaches for generalization apply data augmentation techniques to increase the diversity of training data. Even though this prevents overfitting to the training environment(s), it hinders policy optimization. Crafting a suitable observation, only containing crucial information, has been shown to be a challenging task itself. To improve data efficiency and generalization capabilities, we propose Compact Reshaped Observation Processing (CROP) to reduce the state information used for policy optimization. By providing only relevant information, overfitting to a specific training layout is precluded and generalization to unseen environments is improved. We formulate three CROPs that can be applied to fully observable observation- and action-spaces and provide methodical foundation. We empirically show the improvements of CROP in a distributionally shifted safety gridworld. We furthermore provide benchmark comparisons to full observability and data-augmentation in two different-sized procedurally generated mazes.

----

## [380] Learning to Learn from Corrupted Data for Few-Shot Learning

**Authors**: *Yuexuan An, Xingyu Zhao, Hui Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/381](https://doi.org/10.24963/ijcai.2023/381)

**Abstract**:

Few-shot learning which aims to generalize knowledge learned from annotated base training data to recognize unseen novel classes has attracted considerable attention. Existing few-shot methods rely on completely clean training data. However, in the real world, the training data are always corrupted and accompanied by noise due to the disturbance in data transmission and low-quality annotation, which severely degrades the performance and generalization capability of few-shot models. To address the problem, we propose a unified peer-collaboration learning (PCL) framework to extract valid knowledge from corrupted data for few-shot learning. PCL leverages two modules to mimic the peer collaboration process which cooperatively evaluates the importance of each sample. Specifically, each module first estimates the importance weights of different samples by encoding the information provided by the other module from both global and local perspectives. Then, both modules leverage the obtained importance weights to guide the reevaluation of the loss value of each sample. In this way, the peers can mutually absorb knowledge to improve the robustness of few-shot models. Experiments verify that our framework combined with different few-shot methods can significantly improve the performance and robustness of original models.

----

## [381] Computing Abductive Explanations for Boosted Regression Trees

**Authors**: *Gilles Audemard, Steve Bellart, Jean-Marie Lagniez, Pierre Marquis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/382](https://doi.org/10.24963/ijcai.2023/382)

**Abstract**:

We present two algorithms for generating (resp. evaluating) abductive explanations for boosted regression trees. Given an instance x and an interval I containing its value F (x) for the boosted regression tree F at hand, the generation algorithm returns a (most general) term t over the Boolean conditions in F such that every instance x′ satisfying t is such that F (x′ ) ∈ I. The evaluation algorithm tackles the corresponding inverse problem: given F , x and a term t over the Boolean conditions in F such that t covers x, find the least interval I_t such that for every instance x′ covered by t we have F (x′ ) ∈ I_t . Experiments on various datasets show that the two algorithms are practical enough to be used for generating (resp. evaluating) abductive explanations for boosted regression trees based on a large number of Boolean conditions.

----

## [382] HOUDINI: Escaping from Moderately Constrained Saddles

**Authors**: *Dmitrii Avdiukhin, Grigory Yaroslavtsev*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/383](https://doi.org/10.24963/ijcai.2023/383)

**Abstract**:

We give polynomial time algorithms for escaping from high-dimensional saddle points under a moderate number of constraints. Given gradient access to a smooth function, we show that (noisy) gradient descent methods can escape from saddle points under a logarithmic number of inequality constraints. While analogous results exist for unconstrained and equality-constrained problems, we make progress on the major open question of convergence to second-order stationary points in the case of inequality constraints, without reliance on NP-oracles or altering the definitions to only account for certain constraints. Our results hold for both regular and stochastic gradient descent.

----

## [383] Scaling Goal-based Exploration via Pruning Proto-goals

**Authors**: *Akhil Bagaria, Tom Schaul*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/384](https://doi.org/10.24963/ijcai.2023/384)

**Abstract**:

One of the gnarliest challenges in reinforcement learning (RL) is exploration that scales to vast domains, where novelty-, or coverage-seeking behaviour falls short. Goal-directed, purposeful behaviours are able to overcome this, but rely on a good goal space. The core challenge in goal discovery is finding the right balance between generality (not hand-crafted) and tractability (useful, not too many). Our approach explicitly seeks the middle ground, enabling the human designer to specify a vast but meaningful proto-goal space, and an autonomous discovery process to refine this to a narrower space of controllable, reachable, novel, and relevant goals. The effectiveness of goal-conditioned exploration with the latter is then demonstrated in three challenging environments.

----

## [384] ReLiNet: Stable and Explainable Multistep Prediction with Recurrent Linear Parameter Varying Networks

**Authors**: *Alexandra Baier, Decky Aspandi, Steffen Staab*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/385](https://doi.org/10.24963/ijcai.2023/385)

**Abstract**:

Multistep prediction models are essential for the simulation and model-predictive control of dynamical systems. Verifying the safety of such models is a multi-faceted problem requiring both system-theoretic guarantees as well as establishing trust with human users. In this work, we propose a novel approach, ReLiNet (Recurrent Linear Parameter Varying Network), to ensure safety for multistep prediction of dynamical systems. Our approach simplifies a recurrent neural network to a switched linear system that is constrained to guarantee exponential stability, which acts as a surrogate for safety from a system-theoretic perspective. Furthermore, ReLiNet's computation can be reduced to a single linear model for each time step, resulting in predictions that are explainable by definition, thereby establishing trust from a human-centric perspective. Our quantitative experiments show that ReLiNet achieves prediction accuracy comparable to that of state-of-the-art recurrent neural networks, while achieving more faithful and robust explanations compared to the model-agnostic explanation method of LIME.

----

## [385] Poisoning the Well: Can We Simultaneously Attack a Group of Learning Agents?

**Authors**: *Ridhima Bector, Hang Xu, Abhay Aradhya, Chai Quek, Zinovi Rabinovich*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/386](https://doi.org/10.24963/ijcai.2023/386)

**Abstract**:

Reinforcement Learning's (RL) ubiquity has instigated research on potential threats to its training and deployment. Many works study single-learner training-time attacks that "pre-programme" behavioral triggers into a strategy. However, attacks on collections of learning agents remain largely overlooked. We remedy the situation by developing a constructive training-time attack on a population of learning agents and additionally make the attack agnostic to the population's size. The attack constitutes a sequence of environment (re)parameterizations (poisonings), generated to overcome individual differences between agents and lead the entire population to the same target behavior while minimizing effective environment modulation. Our method is demonstrated on populations of independent learners in "ghost" environments (learners do not interact or perceive each other) as well as environments with mutual awareness, with or without individual learning. From the attack perspective, we pursue an ultra-blackbox setting, i.e., the attacker's training utilizes only across-policy traces of the victim learners for both attack conditioning and evaluation. The resulting uncertainty in population behavior is managed via a novel Wasserstein distance-based Gaussian embedding of behaviors detected within the victim population. To align with prior works on environment poisoning, our experiments are based on a 3D Grid World domain and show:  a) feasibility, i.e., despite the uncertainty, the attack forces a population-wide adoption of target behavior; b) efficacy, i.e., the attack is size-agnostic and transferable. Code and Appendices are available at "bit.ly/github-rb-cep".

----

## [386] On Approximating Total Variation Distance

**Authors**: *Arnab Bhattacharyya, Sutanu Gayen, Kuldeep S. Meel, Dimitrios Myrisiotis, A. Pavan, N. V. Vinodchandran*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/387](https://doi.org/10.24963/ijcai.2023/387)

**Abstract**:

Total variation distance (TV distance) is a fundamental notion of distance between probability distributions. In this work, we introduce and study the problem of computing the TV distance of two product distributions over the domain {0,1}^n. In particular, we establish the following results.

1. The problem of exactly computing the TV distance of two product distributions is #P-complete. This is in stark contrast with other distance measures such as KL, Chi-square, and Hellinger which tensorize over the marginals leading to efficient algorithms.

2. There is a fully polynomial-time deterministic approximation scheme (FPTAS)  for computing the TV distance of two product distributions P and Q where Q is the uniform distribution. This result is extended to the case where Q has a constant number of distinct marginals. In contrast, we show that when P and Q are Bayes net distributions the relative approximation of their TV distance is NP-hard.

----

## [387] Lifelong Multi-view Spectral Clustering

**Authors**: *Hecheng Cai, Yuze Tan, Shudong Huang, Jiancheng Lv*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/388](https://doi.org/10.24963/ijcai.2023/388)

**Abstract**:

In recent years, spectral clustering has become a well-known and effective algorithm in machine learning. However, traditional spectral clustering algorithms are designed for single-view data and fixed task setting. This can become a limitation when dealing with new tasks in a sequence, as it requires accessing previously learned tasks. Hence it leads to high storage consumption, especially for multi-view datasets. In this paper, we address this limitation by introducing a lifelong multi-view clustering framework. Our approach uses view-specific knowledge libraries to capture intra-view knowledge across different tasks. Specifically, we propose two types of libraries: an orthogonal basis library that stores cluster centers in consecutive tasks, and a feature embedding library that embeds feature relations shared among correlated tasks. When a new clustering task is coming, the knowledge is iteratively transferred from libraries to encode the new task, and knowledge libraries are updated according to the online update formulation. Meanwhile, basis libraries of different views are further fused into a consensus library with adaptive weights. Experimental results show that our proposed method outperforms other competitive clustering methods on multi-view datasets by a large margin.

----

## [388] A Novel Demand Response Model and Method for Peak Reduction in Smart Grids - PowerTAC

**Authors**: *Sanjay Chandlekar, Shweta Jain, Sujit Gujar*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/389](https://doi.org/10.24963/ijcai.2023/389)

**Abstract**:

One of the widely used peak reduction methods in smart grids is demand response, where one analyzes the shift in customers' (agents') usage patterns in response to the signal from the distribution company. Often, these signals are in the form of incentives offered to agents. This work studies the effect of incentives on the probabilities of accepting such offers in a real-world smart grid simulator, PowerTAC. We first show that there exists a function that depicts the probability of an agent reducing its load as a function of the discounts offered to them. We call it reduction probability (RP). RP  function is further parametrized by the rate of reduction (RR), which can differ for each agent. We provide an optimal algorithm, MJS--ExpResponse, that outputs the discounts to each agent by maximizing the expected reduction under a budget constraint. When RRs are unknown, we propose a Multi-Armed Bandit (MAB) based online algorithm, namely MJSUCB--ExpResponse, to learn RRs. Experimentally we show that it exhibits sublinear regret. Finally, we showcase the efficacy of the proposed algorithm in mitigating demand peaks in a real-world smart grid system using the PowerTAC simulator as a test bed.

----

## [389] Boosting Few-Shot Open-Set Recognition with Multi-Relation Margin Loss

**Authors**: *Yongjuan Che, Yuexuan An, Hui Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/390](https://doi.org/10.24963/ijcai.2023/390)

**Abstract**:

Few-shot open-set recognition (FSOSR) has become a great challenge, which requires classifying known classes and rejecting the unknown ones with only limited samples. Existing FSOSR methods mainly construct an ambiguous distribution of known classes from scarce known samples without considering the latent distribution information of unknowns, which degrades the performance of open-set recognition. To address this issue, we propose a novel loss function called multi-relation margin (MRM) loss that can plug in few-shot methods to boost the performance of FSOSR. MRM enlarges the margin between different classes by extracting the multi-relationship of paired samples to dynamically refine the decision boundary for known classes and implicitly delineate the distribution of unknowns. Specifically, MRM separates the classes by enforcing a margin while concentrating samples of the same class on a hypersphere with a learnable radius. In order to better capture the distribution information of each class, MRM extracts the similarity and correlations among paired samples, ameliorating the optimization of the margin and radius. Experiments on public benchmarks reveal that methods with MRM loss can improve the unknown detection of AUROC by a significant margin while correctly classifying the known classes.

----

## [390] Ensemble Reinforcement Learning in Continuous Spaces - A Hierarchical Multi-Step Approach for Policy Training

**Authors**: *Gang Chen, Victoria Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/391](https://doi.org/10.24963/ijcai.2023/391)

**Abstract**:

Actor-critic deep reinforcement learning (DRL) algorithms have recently achieved prominent success in tackling various challenging reinforcement learning (RL) problems, particularly complex control tasks with high-dimensional continuous state and action spaces. Nevertheless, existing research showed that actor-critic DRL algorithms often failed to explore their learning environments effectively, resulting in limited learning stability and performance. To address this limitation, several ensemble DRL algorithms have been proposed lately to boost exploration and stabilize the learning process. However, most of existing ensemble algorithms do not explicitly train all base learners towards jointly optimizing the performance of the ensemble. In this paper, we propose a new technique to train an ensemble of base learners based on an innovative multi-step integration method. This training technique enables us to develop a new hierarchical learning algorithm for ensemble DRL that effectively promotes inter-learner collaboration through stable inter-learner parameter sharing. The design of our new algorithm is verified theoretically. The algorithm is also shown empirically to outperform several state-of-the-art DRL algorithms on multiple benchmark RL problems.

----

## [391] Incremental and Decremental Optimal Margin Distribution Learning

**Authors**: *Li-Jun Chen, Teng Zhang, Xuanhua Shi, Hai Jin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/392](https://doi.org/10.24963/ijcai.2023/392)

**Abstract**:

Incremental and decremental learning (IDL) deals with the tasks where new data arrives sequentially as a stream or old data turns unavailable continually due to the privacy protection. Existing IDL methods mainly focus on support vector machine and its variants with linear-type loss. There are few studies about the quadratic-type loss, whose Lagrange multipliers are unbounded and much more difficult to track. In this paper, we take the latest statistical learning framework optimal margin distribution machine (ODM) which involves a quadratic-type loss due to the optimization of margin variance, for example, and equip it with the ability to handle IDL tasks. Our proposed ID-ODM can avoid updating the Lagrange multipliers in an infinite range by determining their optimal values beforehand so as to enjoy much more efficiency. Moreover, ID-ODM is also applicable when multiple instances come and leave simultaneously. Extensive empirical studies show that ID-ODM can achieve 9.1x speedup on average with almost no generalization lost compared to retraining ODM on new data set from scratch.

----

## [392] Prompt Federated Learning for Weather Forecasting: Toward Foundation Models on Meteorological Data

**Authors**: *Shengchao Chen, Guodong Long, Tao Shen, Jing Jiang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/393](https://doi.org/10.24963/ijcai.2023/393)

**Abstract**:

To tackle the global climate challenge, it urgently needs to develop a collaborative platform for comprehensive weather forecasting on large-scale meteorological data. Despite urgency, heterogeneous meteorological sensors across countries and regions, inevitably causing multivariate heterogeneity and data exposure, become the main barrier. This paper develops a foundation model across regions capable of understanding complex meteorological data and providing weather forecasting. To relieve the data exposure concern across regions, a novel federated learning approach has been proposed to collaboratively learn a brand-new spatio-temporal Transformer-based foundation model across participants with heterogeneous meteorological data. Moreover, a novel prompt learning mechanism has been adopted to satisfy low-resourced sensors' communication and computational constraints. The effectiveness of the proposed method has been demonstrated on classical weather forecasting tasks using three meteorological datasets with multivariate time series.

----

## [393] FedOBD: Opportunistic Block Dropout for Efficiently Training Large-scale Neural Networks through Federated Learning

**Authors**: *Yuanyuan Chen, Zichen Chen, Pengcheng Wu, Han Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/394](https://doi.org/10.24963/ijcai.2023/394)

**Abstract**:

Large-scale neural networks possess considerable expressive power. They are well-suited for complex learning tasks in industrial applications. However, large-scale models pose significant challenges for training under the current Federated Learning (FL) paradigm. Existing approaches for efficient FL training often leverage model parameter dropout. However, manipulating individual model parameters is not only inefficient in meaningfully reducing the communication overhead when training  large-scale FL models, but may also be detrimental to the scaling efforts and model performance as shown by recent research. To address these issues, we propose the Federated Opportunistic Block Dropout (FedOBD) approach. The key novelty is that it decomposes large-scale models into semantic blocks so that FL participants can opportunistically upload quantized blocks, which are deemed to be significant towards training the model, to the FL server for aggregation. Extensive experiments evaluating FedOBD against four state-of-the-art approaches based on multiple real-world datasets show that it reduces the overall communication overhead by more than 88% compared to the best performing baseline approach, while achieving the highest test accuracy. To the best of our knowledge, FedOBD is the first approach to perform dropout on FL models at the block level rather than at the individual parameter level.

----

## [394] LSGNN: Towards General Graph Neural Network in Node Classification by Local Similarity

**Authors**: *Yuhan Chen, Yihong Luo, Jing Tang, Liang Yang, Siya Qiu, Chuan Wang, Xiaochun Cao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/395](https://doi.org/10.24963/ijcai.2023/395)

**Abstract**:

Heterophily has been considered as an issue that hurts the performance of Graph Neural Networks (GNNs). To address this issue, some existing work uses a graph-level weighted fusion of the information of multi-hop neighbors to include more nodes with homophily. However, the heterophily might differ among nodes, which requires to consider the local topology. Motivated by it, we propose to use the local similarity (LocalSim) to learn node-level weighted fusion, which can also serve as a plug-and-play module. For better fusion, we propose a novel and efficient Initial Residual Difference Connection (IRDC) to extract more informative multi-hop information. Moreover, we provide theoretical analysis on the effectiveness of LocalSim representing node homophily on synthetic graphs. Extensive evaluations over real benchmark datasets show that our proposed method, namely Local Similarity Graph Neural Network (LSGNN), can offer comparable or superior state-of-the-art performance on both homophilic and heterophilic graphs. Meanwhile, the plug-and-play model can significantly boost the performance of existing GNNs.

----

## [395] Graph Propagation Transformer for Graph Representation Learning

**Authors**: *Zhe Chen, Hao Tan, Tao Wang, Tianrun Shen, Tong Lu, Qiuying Peng, Cheng Cheng, Yue Qi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/396](https://doi.org/10.24963/ijcai.2023/396)

**Abstract**:

This paper presents a novel transformer architecture for graph representation learning. The core insight of our method is to fully consider the information propagation among nodes and edges in a graph when building the attention module in the transformer blocks. Specifically, we propose a new attention mechanism called Graph Propagation Attention (GPA). It explicitly passes the information among nodes and edges in three ways, i.e. node-to-node, node-to-edge, and edge-to-node, which is essential for learning graph-structured data. On this basis, we design an effective transformer architecture named Graph Propagation Transformer (GPTrans) to further help learn graph data. We verify the performance of GPTrans in a wide range of graph learning experiments on several benchmark datasets. These results show that our method outperforms many state-of-the-art transformer-based graph models with better performance. The code will be released at https://github.com/czczup/GPTrans.

----

## [396] Some General Identification Results for Linear Latent Hierarchical Causal Structure

**Authors**: *Zhengming Chen, Feng Xie, Jie Qiao, Zhifeng Hao, Ruichu Cai*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/397](https://doi.org/10.24963/ijcai.2023/397)

**Abstract**:

We study the problem of learning hierarchical causal structure among latent variables from measured variables. While some existing methods are able to recover the latent hierarchical causal structure, they mostly suffer from restricted assumptions, including the tree-structured graph constraint, no ``triangle" structure, and non-Gaussian assumptions.  In this paper, we relax these restrictions above and consider a more general and challenging scenario where the beyond tree-structured graph, the ``triangle" structure, and the arbitrary noise distribution are allowed. We investigate the identifiability of the latent hierarchical causal structure and show that by using second-order statistics, the latent hierarchical structure can be identified up to the Markov equivalence classes over latent variables. Moreover, some directions in the Markov equivalence classes of latent variables can be further identified using partially non-Gaussian data. Based on the theoretical results above, we design an effective algorithm for learning the latent hierarchical causal structure. The experimental results on synthetic data verify the effectiveness of the proposed method.

----

## [397] Deep Multi-view Subspace Clustering with Anchor Graph

**Authors**: *Chenhang Cui, Yazhou Ren, Jingyu Pu, Xiaorong Pu, Lifang He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/398](https://doi.org/10.24963/ijcai.2023/398)

**Abstract**:

Deep multi-view subspace clustering (DMVSC) has recently attracted increasing attention due to its promising performance. However, existing DMVSC methods still have two issues: (1) they mainly focus on using autoencoders to nonlinearly embed the data, while the embedding may be suboptimal for clustering because the clustering objective is rarely considered in autoencoders, and (2) existing methods typically have a quadratic or even cubic complexity, which makes it challenging to deal with large-scale data. To address these issues, in this paper we propose a novel deep multi-view subspace clustering method with anchor graph (DMCAG). To be specific, DMCAG firstly learns the embedded features for each view independently, which are used to obtain the subspace representations. To significantly reduce the complexity, we construct an anchor graph with small size for each view. Then, spectral clustering is performed on an integrated anchor graph to obtain pseudo-labels. To overcome the negative impact caused by suboptimal embedded features, we use pseudo-labels to refine the embedding process to make it more suitable for the clustering task. Pseudo-labels and embedded features are updated alternately. Furthermore, we design a strategy to keep the consistency of the labels based on contrastive learning to enhance the clustering performance. Empirical studies on real-world datasets show that our method achieves superior clustering performance over other state-of-the-art methods.

----

## [398] Neuro-Symbolic Learning of Answer Set Programs from Raw Data

**Authors**: *Daniel Cunnington, Mark Law, Jorge Lobo, Alessandra Russo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/399](https://doi.org/10.24963/ijcai.2023/399)

**Abstract**:

One of the ultimate goals of Artificial Intelligence is to assist humans in complex decision making. A promising direction for achieving this goal is Neuro-Symbolic AI, which aims to combine the interpretability of symbolic techniques with the ability of deep learning to learn from raw data. However, most current approaches require manually engineered symbolic knowledge, and where end-to-end training is considered, such approaches are either restricted to learning definite programs, or are restricted to training binary neural networks. In this paper, we introduce Neuro-Symbolic Inductive Learner (NSIL), an approach that trains a general neural network to extract latent concepts from raw data, whilst learning symbolic knowledge that maps latent concepts to target labels. The novelty of our approach is a method for biasing the learning of symbolic knowledge, based on the in-training performance of both neural and symbolic components. We evaluate NSIL on three problem domains of different complexity, including an NP-complete problem. Our results demonstrate that NSIL learns expressive knowledge, solves computationally complex problems, and achieves state-of-the-art performance in terms of accuracy and data efficiency. Code and technical appendix: https://github.com/DanCunnington/NSIL

----

## [399] Deep Symbolic Learning: Discovering Symbols and Rules from Perceptions

**Authors**: *Alessandro Daniele, Tommaso Campari, Sagar Malhotra, Luciano Serafini*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/400](https://doi.org/10.24963/ijcai.2023/400)

**Abstract**:

Neuro-Symbolic (NeSy) integration combines symbolic reasoning with Neural Networks (NNs) for tasks requiring perception and reasoning. Most NeSy systems rely on continuous relaxation of logical knowledge, and no discrete decisions are made within the model pipeline. Furthermore, these methods assume that the symbolic rules are given. In this paper, we propose Deep Symboilic Learning (DSL), a NeSy system that learns NeSy-functions, i.e., the composition of a (set of) perception functions which map continuous data to discrete symbols, and a symbolic function over the set of symbols. DSL simultaneously learns the perception and symbolic functions while being trained only on their composition (NeSy-function). The key novelty of DSL is that it can create internal (interpretable) symbolic representations and map them to perception inputs within a differentiable NN learning pipeline. The created symbols are automatically selected to generate symbolic functions that best explain the data. We provide experimental analysis to substantiate the efficacy of DSL  in simultaneously learning perception and symbolic functions.

----



[Go to the previous page](IJCAI-2023-list01.md)

[Go to the next page](IJCAI-2023-list03.md)

[Go to the catalog section](README.md)