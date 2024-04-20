## [600] Generalization through Diversity: Improving Unsupervised Environment Design

**Authors**: *Wenjun Li, Pradeep Varakantham, Dexun Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/601](https://doi.org/10.24963/ijcai.2023/601)

**Abstract**:

Agent decision making using Reinforcement Learning (RL) heavily relies on either a model or simulator of the environment (e.g., moving in an 8x8 maze with three rooms, playing Chess on an 8x8 board). Due to this dependence, small changes in the environment (e.g., positions of obstacles in the maze, size of the board) can severely affect the effectiveness of the policy learned by the agent. To that end, existing work has proposed training RL agents on an adaptive curriculum of environments (generated automatically) to improve performance on out-of-distribution (OOD) test scenarios. Specifically, existing research has employed the potential for the agent to learn in an environment (captured using Generalized Advantage Estimation, GAE) as the key factor to select the next environment(s) to train the agent. However, such a mechanism can select similar environments (with a high potential to learn) thereby making agent training redundant on all but one of those environments. To that end, we provide a principled approach to adaptively identify diverse environments based on a novel distance measure relevant to environment design. We empirically demonstrate the versatility and effectiveness of our method in comparison to multiple leading approaches for unsupervised environment design on three distinct benchmark problems used in literature.

----

## [601] Can I Really Do That? Verification of Meta-Operators via Stackelberg Planning

**Authors**: *Florian Pham, Álvaro Torralba*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/602](https://doi.org/10.24963/ijcai.2023/602)

**Abstract**:

Macro-operators are a common reformulation method in planning that adds high-level operators corresponding to a fixed sequence of primitive operators. We introduce meta-operators, which allow using different sequences of actions in each state. We show how to automatically verify whether a meta-operator is valid, i.e., the represented behavior is always doable. This can be checked at once for all instantiations of the meta-operator and all reachable states via a compilation into Stackelberg planning, a form of adversarial planning.  Our results show that meta-operators learned for multiple domains can often express useful high-level behaviors very compactly, improving planners' performance.

----

## [602] Topological Planning with Post-unique and Unary Actions

**Authors**: *Guillaume Prévost, Stéphane Cardon, Tristan Cazenave, Christophe Guettier, Éric Jacopin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/603](https://doi.org/10.24963/ijcai.2023/603)

**Abstract**:

We are interested in realistic planning problems to model the behavior of Non-Playable Characters (NPCs) in video games. Search-based action planning, introduced by the game F.E.A.R. in 2005, has an exponential time complexity allowing to control only a dozen NPCs between two frames. A close study of the plans generated in first-person shooters shows that: (1) actions are unary, (2) actions are contextually post-unique and (3) there is no two instances of the same action in an NPC’s plan. By considering (1), (2) and (3) as restrictions, we introduce new classes of problems with the Simplified Action Structure formalism which indeed allow to model realistic problems and whose instances are solvable by a linear-time algorithm. We also experimentally show that our algorithm is capable of managing millions of NPCs per frame.

----

## [603] Model Predictive Control with Reach-avoid Analysis

**Authors**: *Dejin Ren, Wanli Lu, Jidong Lv, Lijun Zhang, Bai Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/604](https://doi.org/10.24963/ijcai.2023/604)

**Abstract**:

In this paper we investigate the optimal controller synthesis problem, so that the system under the controller can reach a specified target set while satisfying given constraints. Existing model predictive control (MPC) methods learn from a set of discrete states visited by previous (sub-)optimized trajectories and thus result in computationally expensive mixed-integer nonlinear optimization. In this paper a novel MPC method is proposed based on reach-avoid analysis to solve the controller synthesis problem iteratively. The reach-avoid analysis is concerned with computing a reach-avoid set which is a set of initial states such that the system can reach the target set successfully. It not only provides terminal constraints, which ensure feasibility of MPC, but also expands discrete states in existing methods into a continuous set (i.e., reach-avoid sets) and thus leads to nonlinear optimization which is more computationally tractable online due to the absence of integer variables. Finally, we evaluate the proposed method and make comparisons with state-of-the-art ones based on several examples.

----

## [604] Formal Explanations of Neural Network Policies for Planning

**Authors**: *Renee Selvey, Alban Grastien, Sylvie Thiébaux*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/605](https://doi.org/10.24963/ijcai.2023/605)

**Abstract**:

Deep learning is increasingly used to learn policies for planning problems, yet policies represented by neural networks are difficult to interpret, verify and trust. Existing formal approaches to post-hoc explanations provide concise reasons for a single decision made by an ML model. However, understanding planning policies require explaining sequences of decisions. In this paper,  we formulate the problem of finding explanations for the sequence of decisions recommended by a learnt policy in a given state. We show that, under certain assumptions, a minimal explanation for a sequence can be computed by solving a  number of single decision explanation problems which is linear in the length of the sequence. We present experimental results of our implementation of this approach for ASNet policies for classical planning domains.

----

## [605] Optimal Decision Tree Policies for Markov Decision Processes

**Authors**: *Daniël Vos, Sicco Verwer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/606](https://doi.org/10.24963/ijcai.2023/606)

**Abstract**:

Interpretability of reinforcement learning policies is essential for many real-world tasks but learning such interpretable policies is a hard problem. Particularly, rule-based policies such as decision trees and rules lists are difficult to optimize due to their non-differentiability. While existing techniques can learn verifiable decision tree policies, there is no guarantee that the learners generate a policy that performs optimally. In this work, we study the optimization of size-limited decision trees for Markov Decision Processes (MPDs) and propose OMDTs: Optimal MDP Decision Trees. Given a user-defined size limit and MDP formulation, OMDT directly maximizes the expected discounted return for the decision tree using Mixed-Integer Linear Programming. By training optimal tree policies for different MDPs we empirically study the optimality gap for existing imitation learning techniques and find that they perform sub-optimally. We show that this is due to an inherent shortcoming of imitation learning, namely that complex policies cannot be represented using size-limited trees. In such cases, it is better to directly optimize the tree for expected return. While there is generally a trade-off between the performance and interpretability of machine learning models, we find that on small MDPs, depth 3 OMDTs often perform close to optimally.

----

## [606] Online Task Assignment with Controllable Processing Time

**Authors**: *Ruoyu Wu, Wei Bao, Liming Ge*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/607](https://doi.org/10.24963/ijcai.2023/607)

**Abstract**:

We study a new online assignment problem, called the Online Task Assignment with Controllable Processing Time. In a bipartite graph,  a set of online vertices (tasks) should be assigned to a set of offline vertices (machines) under the known adversarial distribution (KAD) assumption. We are the first to study controllable processing time in this scenario: There are  multiple processing levels for each task and higher level brings larger utility but also larger processing delay.
A machine can reject an assignment at the cost of a rejection penalty, taken from a pre-determined rejection budget. Different processing levels cause different penalties. We propose the Online Machine and Level Assignment  (OMLA) Algorithm to simultaneously assign an offline machine and a processing level to each online task. We prove that OMLA achieves 1/2-competitive ratio if each machine has unlimited rejection budget and Δ/(3Δ-1)- competitive ratio if each machine has an initial rejection budget up to Δ. Interestingly, the competitive ratios do not change under different settings on the controllable processing time and we can conclude that OMLA is "insensitive" to the controllable processing time.

----

## [607] A Rigorous Risk-aware Linear Approach to Extended Markov Ratio Decision Processes with Embedded Learning

**Authors**: *Alexander Zadorojniy, Takayuki Osogami, Orit Davidovich*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/608](https://doi.org/10.24963/ijcai.2023/608)

**Abstract**:

We consider the problem of risk-aware Markov Decision Processes (MDPs) for Safe AI. We introduce a theoretical framework, Extended Markov Ratio Decision Processes (EMRDP), that incorporates risk into MDPs and embeds environment learning into this framework. We propose an algorithm to find the optimal policy for EMRDP with theoretical guarantees. Under a certain monotonicity assumption, this algorithm runs in strongly-polynomial time both in the discounted and expected average reward models. We validate our algorithm empirically on a Grid World benchmark, evaluating its solution quality, required number of steps, and numerical stability. We find its solution quality to be stable under data noising, while its required number of steps grows with added noise. We observe its numerical stability compared to global methods.

----

## [608] Learning to Act for Perceiving in Partially Unknown Environments

**Authors**: *Leonardo Lamanna, Mohamadreza Faridghasemnia, Alfonso Gerevini, Alessandro Saetti, Alessandro Saffiotti, Luciano Serafini, Paolo Traverso*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/609](https://doi.org/10.24963/ijcai.2023/609)

**Abstract**:

Autonomous agents embedded in a physical environment need the ability to correctly perceive the state of the environment from sensory data. In partially observable environments, certain properties can be perceived only in specific situations and from certain viewpoints that can be reached by the agent by planning and executing actions. For instance, to understand whether a cup is full of coffee, an agent, equipped with a camera, needs to turn on the light and look at the cup from the top. When the proper situations to perceive the desired properties are unknown, an agent needs to learn them and plan to get in such situations. In this paper, we devise a general method to solve this problem by evaluating the confidence of a neural network online and by using symbolic planning. We experimentally evaluate the proposed approach on several synthetic datasets, and show the feasibility of our approach in a real-world scenario that involves noisy perceptions and noisy actions on a real robot.

----

## [609] Learning to Self-Reconfigure for Freeform Modular Robots via Altruism Proximal Policy Optimization

**Authors**: *Lei Wu, Bin Guo, Qiuyun Zhang, Zhuo Sun, Jieyi Zhang, Zhiwen Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/610](https://doi.org/10.24963/ijcai.2023/610)

**Abstract**:

The advantages of modular robot systems stem from their ability to change between different configurations, enabling them to adapt to complex and dynamic real-world environments. Then, how to perform the accurate and efficient change of the modular robot system, i.e., the self-reconfiguration problem, is essential. Existing reconfiguration algorithms are based on discrete motion primitives and are suitable for lattice-type modular robots. The modules of freeform modular robots are connected without alignment, and the motion space is continuous. It renders existing reconfiguration methods infeasible. In this paper, we design a parallel distributed self-reconfiguration algorithm for freeform modular robots based on multi-agent reinforcement learning to realize the automatic design of conflict-free reconfiguration controllers in continuous action spaces. To avoid conflicts, we incorporate a collaborative mechanism into reinforcement learning. Furthermore, we design the distributed termination criteria to achieve timely termination in the presence of limited communication and local observability. When compared to the baselines, simulations show that the proposed method improves efficiency and congruence, and module movement demonstrates altruism.

----

## [610] Multi-Robot Coordination and Layout Design for Automated Warehousing

**Authors**: *Yulun Zhang, Matthew C. Fontaine, Varun Bhatt, Stefanos Nikolaidis, Jiaoyang Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/611](https://doi.org/10.24963/ijcai.2023/611)

**Abstract**:

With the rapid progress in Multi-Agent Path Finding (MAPF), researchers have studied how MAPF algorithms can be deployed to coordinate hundreds of robots in large automated warehouses. While most works try to improve the throughput of such warehouses by developing better MAPF algorithms, we focus on improving the throughput by optimizing the warehouse layout. We show that, even with state-of-the-art MAPF algorithms, commonly used human-designed layouts can lead to congestion for warehouses with large numbers of robots and thus have limited scalability. We extend existing automatic scenario generation methods to optimize warehouse layouts. Results show that our optimized warehouse layouts (1) reduce traffic congestion and thus improve throughput, (2) improve the scalability of the automated warehouses by doubling the number of robots in some cases, and (3) are capable of generating layouts with user-specified diversity measures.

----

## [611] Stochastic Population Update Can Provably Be Helpful in Multi-Objective Evolutionary Algorithms

**Authors**: *Chao Bian, Yawen Zhou, Miqing Li, Chao Qian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/612](https://doi.org/10.24963/ijcai.2023/612)

**Abstract**:

Evolutionary algorithms (EAs) have been widely and successfully applied to solve multi-objective optimization problems, due to their nature of population-based search. Population update is a key component in multi-objective EAs (MOEAs), and it is performed in a greedy, deterministic manner. That is, the next-generation population is formed by selecting the first population-size ranked solutions (based on some selection criteria, e.g., non-dominated sorting, crowdedness and indicators) from the collections of the current population and newly-generated solutions. In this paper, we question this practice. We analytically present that introducing randomness into the population update procedure in MOEAs can be beneficial for the search. More specifically, we prove that the expected running time of a well-established MOEA (SMS-EMOA) for solving a commonly studied bi-objective problem, OneJumpZeroJump, can be exponentially decreased if replacing its deterministic population update mechanism by a stochastic one. Empirical studies also verify the effectiveness of the proposed stochastic population update method. This work is an attempt to challenge a common practice for the population update in MOEAs. Its positive results, which might hold more generally, should encourage the exploration of developing new MOEAs in the area.

----

## [612] The First Proven Performance Guarantees for the Non-Dominated Sorting Genetic Algorithm II (NSGA-II) on a Combinatorial Optimization Problem

**Authors**: *Sacha Cerf, Benjamin Doerr, Benjamin Hebras, Yakob Kahane, Simon Wietheger*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/613](https://doi.org/10.24963/ijcai.2023/613)

**Abstract**:

The Non-dominated Sorting Genetic Algorithm-II (NSGA-II) is one of the most prominent algorithms to solve multi-objective optimization problems. Recently, the first mathematical runtime guarantees have been obtained for this algorithm, however only for synthetic benchmark problems. 

In this work, we give the first proven performance guarantees for a classic optimization problem, the NP-complete bi-objective minimum spanning tree problem. More specifically, we show that the NSGA-II with population size N >= 4((n-1) wmax + 1) computes all extremal points of the Pareto front in an expected number of O(m^2 n wmax log(n wmax)) iterations, where n is the number of vertices, m the number of edges, and wmax is the maximum edge weight in the problem instance. This result confirms, via mathematical means, the good performance of the NSGA-II observed empirically. It also shows that mathematical analyses of this algorithm are not only possible for synthetic benchmark problems, but also for more complex combinatorial optimization problems. 
  
  As a side result, we also obtain a new analysis of the performance of the  global SEMO algorithm on the bi-objective minimum spanning tree problem, which improves the previous best result by a factor of |F|, the number of extremal points of the Pareto front, a set that can be as large as n wmax. The main reason for this improvement is our observation that both multi-objective evolutionary algorithms find the different extremal points in parallel rather than sequentially, as assumed in the previous proofs.

----

## [613] Complex Contagion Influence Maximization: A Reinforcement Learning Approach

**Authors**: *Haipeng Chen, Bryan Wilder, Wei Qiu, Bo An, Eric Rice, Milind Tambe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/614](https://doi.org/10.24963/ijcai.2023/614)

**Abstract**:

In influence maximization (IM), the goal is to find a set of seed nodes in a social network that maximizes the influence spread. While most IM problems focus on classical influence cascades (e.g., Independent Cascade and Linear Threshold) which assume individual influence cascade probability is independent of the number of neighbors, recent studies by sociologists show that many influence cascades follow a pattern called complex contagion (CC), where influence cascade probability is much higher when more neighbors are influenced. Nonetheless, there are very limited studies for complex contagion influence maximization (CCIM) problems. This is partly because CC is non-submodular, the solution of which has been an open challenge. 
In this study, we propose the first reinforcement learning (RL) approach to CCIM. We find that a key obstacle in applying existing RL approaches to CCIM is the reward sparseness issue, which comes from two distinct sources. We then design a new RL algorithm that uses the CCIM problem structure to address the issue. Empirical results show that our approach achieves the state-of-the-art performance on 9 real-world networks.

----

## [614] On Optimal Strategies for Wordle and General Guessing Games

**Authors**: *Michael Cunanan, Michael Thielscher*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/615](https://doi.org/10.24963/ijcai.2023/615)

**Abstract**:

The recent popularity of Wordle has revived interest in guessing games. We develop a general method for finding optimal strategies for guessing games while avoiding an exhaustive search. Our main contribution are several theorems that build towards a general theory to prove optimality of a strategy for a guessing game. This work is developed to apply to any guessing game, but we use Wordle as an example to present concrete results.

----

## [615] Runtime Analyses of Multi-Objective Evolutionary Algorithms in the Presence of Noise

**Authors**: *Matthieu Dinot, Benjamin Doerr, Ulysse Hennebelle, Sebastian Will*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/616](https://doi.org/10.24963/ijcai.2023/616)

**Abstract**:

In single-objective optimization, it is well known that evolutionary algorithms also without further adjustments can stand a certain amount of noise in the evaluation of the objective function. In contrast, this question is not at all understood for multi-objective optimization.

 In this work, we conduct the first mathematical runtime analysis of a simple multi-objective evolutionary algorithm (MOEA) on a classic benchmark in the presence of noise in the objective function. 
 We prove that when bit-wise prior noise with rate p <= alpha/n, alpha a suitable constant, is present, the simple evolutionary multi-objective optimizer (SEMO) without any adjustments to cope with noise finds the Pareto front of the OneMinMax benchmark in time O(n^2 log n), just as in the case without noise. Given that the problem here is to arrive at a population consisting of n+1 individuals witnessing the Pareto front, this is a surprisingly strong robustness to noise (comparably simple evolutionary algorithms cannot optimize the single-objective OneMax problem in polynomial time when p = omega(log(n)/n)). Our proofs suggest that the strong robustness of the MOEA stems from its implicit diversity mechanism designed to enable it to compute a population covering the whole Pareto front. 
 
 Interestingly this result only holds when the objective value of a solution is determined only once and the algorithm from that point on works with this, possibly noisy, objective value. We prove that when all solutions are reevaluated in each iteration, then any noise rate p = omega(log(n)/n^2) leads to a super-polynomial runtime. This is very different from single-objective optimization, where it is generally preferred to reevaluate solutions whenever their fitness is important and where examples are known such that not reevaluating solutions can lead to catastrophic performance losses.

----

## [616] Diverse Approximations for Monotone Submodular Maximization Problems with a Matroid Constraint

**Authors**: *Anh Viet Do, Mingyu Guo, Aneta Neumann, Frank Neumann*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/617](https://doi.org/10.24963/ijcai.2023/617)

**Abstract**:

Finding diverse solutions to optimization problems has been of practical interest for several decades, and recently enjoyed increasing attention in research. While submodular optimization has been rigorously studied in many fields, its diverse solutions extension has not. In this study, we consider the most basic variants of submodular optimization, and propose two simple greedy algorithms, which are known to be effective at maximizing monotone submodular functions. These are equipped with parameters that control the trade-off between objective and diversity. Our theoretical contribution shows their approximation guarantees in both objective value and diversity, as functions of their respective parameters. Our experimental investigation with maximum vertex coverage instances demonstrates their empirical differences in terms of objective-diversity trade-offs.

----

## [617] Efficient Object Search in Game Maps

**Authors**: *Jinchun Du, Bojie Shen, Shizhe Zhao, Muhammad Aamir Cheema, Adel Nadjaran Toosi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/618](https://doi.org/10.24963/ijcai.2023/618)

**Abstract**:

Video games feature a dynamic environment where locations of objects (e.g., characters, equipment, weapons, vehicles etc.) frequently change within the game world. Although searching for relevant nearby objects in such a dynamic setting is a fundamental operation, this problem has received little research attention. In this paper, we propose a simple lightweight index, called Grid Tree, to store objects and their associated textual data. Our index can be efficiently updated with the underlying updates such as object movements, and supports a variety of object search queries, including k nearest neighbors (returning the k closest objects), keyword k nearest neighbors (returning the k closest objects that satisfy query keywords), and several other variants. Our extensive experimental study, conducted on standard game maps benchmarks and real-world keywords, demonstrates that our approach has  up to 2 orders of magnitude faster update times for moving objects compared to state-of-the-art approaches such as navigation mesh and IR-tree. At the same time, query performance of our approach is similar to or better than that of IR-tree and up to two orders of magnitude faster than the other competitor.

----

## [618] Sorting and Hypergraph Orientation under Uncertainty with Predictions

**Authors**: *Thomas Erlebach, Murilo S. de Lima, Nicole Megow, Jens Schlöter*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/619](https://doi.org/10.24963/ijcai.2023/619)

**Abstract**:

Learning-augmented algorithms have been attracting increasing interest, but have only recently been considered in the setting of explorable uncertainty where precise values of uncertain input elements can be obtained by a query and the goal is to minimize the number of queries needed to solve a problem. We study learning-augmented algorithms for sorting and hypergraph orientation under uncertainty, assuming access to untrusted predictions for the uncertain values. Our algorithms provide improved performance guarantees for accurate predictions while maintaining worst-case guarantees that are best possible without predictions. For sorting, our algorithm uses the optimal number of queries for accurate predictions and at most twice the optimal number for arbitrarily wrong predictions. For hypergraph orientation, for any γ≥2, we give an algorithm that uses at most 1+1/γ times the optimal number of queries for accurate predictions and at most γ times the optimal number for arbitrarily wrong predictions. These tradeoffs are the best possible. We also consider different error metrics and show that the performance of our algorithms degrades smoothly with the prediction error in all the cases where this is possible.

----

## [619] Parameterized Local Search for Max c-Cut

**Authors**: *Jaroslav Garvardt, Niels Grüttemeier, Christian Komusiewicz, Nils Morawietz*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/620](https://doi.org/10.24963/ijcai.2023/620)

**Abstract**:

In the NP-hard Max c-Cut problem, one is given an undirected edge-weighted graph G and wants to color the vertices of G with c colors such that the total weight of edges with distinctly colored endpoints is maximal. The case with c=2 is the famous Max Cut problem. To deal with the NP-hardness of this problem, we study parameterized local search algorithms. More precisely, we study LS-Max c-Cut where we are additionally given a vertex coloring f and an integer k and the task is to find a better coloring f' that differs from f in at most k entries, if such a coloring exists; otherwise, f is k-optimal. We show that LS-Max c-Cut presumably cannot be solved in g(k) · nᴼ⁽¹⁾ time even on bipartite graphs, for all c ≥ 2. We then show an algorithm for LS-Max c-Cut with running time O((3eΔ)ᵏ · c · k³ · Δ · n), where Δ is the maximum degree of the input graph. Finally, we evaluate the practical performance of this algorithm in a hill-climbing approach as a post-processing for state-of-the-art heuristics for Max c-Cut. We show that using parameterized local search, the results of this heuristic can be further improved on a set of standard benchmark instances.

----

## [620] Exploring Structural Similarity in Fitness Landscapes via Graph Data Mining: A Case Study on Number Partitioning Problems

**Authors**: *Mingyu Huang, Ke Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/621](https://doi.org/10.24963/ijcai.2023/621)

**Abstract**:

One of the most common problem-solving heuristics is by analogy. For a given problem, a solver can be viewed as a strategic walk on its fitness landscape. Thus if a solver works for one problem instance, we expect it will also be effective for other instances whose fitness landscapes essentially share structural similarities with each other. However, due to the black-box nature of combinatorial optimization, it is far from trivial to infer such similarity in real-world scenarios. To bridge this gap, by using local optima network as a proxy of fitness landscapes, this paper proposed to leverage graph data mining techniques to conduct qualitative and quantitative analyses to explore the latent topological structural information embedded in those landscapes. In our experiments, we use the number partitioning problem as the case and our empirical results are inspiring to support the overall assumption of the existence of structural similarity between landscapes within neighboring dimensions. Besides, experiments on simulated annealing demonstrate that the performance of a meta-heuristic solver is similar on structurally similar landscapes.

----

## [621] An Exact Algorithm for the Minimum Dominating Set Problem

**Authors**: *Hua Jiang, Zhifei Zheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/622](https://doi.org/10.24963/ijcai.2023/622)

**Abstract**:

The Minimum Dominating Set  (MDS) problem is a classic NP-hard combinatorial optimization problem with many practical applications. Solving MDS is extremely challenging in computation. Previous work on exact algorithms mainly focuses on improving the theoretical time complexity and existing practical algorithms for MDS are almost based on heuristic search. In this paper, we propose a novel lower bound and an exact algorithm for MDS. The algorithm implements a branch-and-bound (BnB) approach and employs the new lower bound to reduce search space. Extensive empirical results show that the new lower bound is efficient in  reduction of the search space and the new algorithm is effective for the standard instances and real-world instances. To the best of our knowledge, this is the first effective BnB algorithm for MDS.

----

## [622] A Refined Upper Bound and Inprocessing for the Maximum K-plex Problem

**Authors**: *Hua Jiang, Fusheng Xu, Zhifei Zheng, Bowen Wang, Wei Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/623](https://doi.org/10.24963/ijcai.2023/623)

**Abstract**:

A k-plex of a graph G is an induced subgraph in which every vertex has at most k-1 nonadjacent vertices. The Maximum k-plex Problem (MKP) consists in finding a k-plex of the largest size, which is NP-hard and finds many applications. Existing exact algorithms mainly implement a branch-and-bound approach and improve performance  by integrating effective upper bounds and graph reduction rules. In this paper, we propose a refined upper bound, which can derive a tighter upper bound than existing methods,  and an inprocessing strategy, which performs graph reduction incrementally. We implement a new BnB algorithm for MKP that employs the two components to reduce the search space.  Extensive experiments show that both the refined upper bound and the inprocessing strategy are very efficient in the  reduction of search space. The new algorithm outperforms the state-of-the-art algorithms on the tested benchmarks significantly.

----

## [623] Levin Tree Search with Context Models

**Authors**: *Laurent Orseau, Marcus Hutter, Levi H. S. Lelis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/624](https://doi.org/10.24963/ijcai.2023/624)

**Abstract**:

Levin Tree Search (LTS) is a search algorithm that makes use of a policy (a probability distribution over actions) 
and comes with a theoretical guarantee on the number of expansions before reaching a goal node, depending on the quality of the policy. 
This guarantee can be used as a loss function, which we call the LTS loss, to optimize neural networks representing the policy (LTS+NN). 
In this work we show that the neural network can be substituted with parameterized context models originating from the online compression literature (LTS+CM). 
We show that the LTS loss is convex under this new model,
which allows for using standard convex optimization tools,
and obtain convergence guarantees to the optimal parameters in an online setting for a given set of solution trajectories --- guarantees that cannot be provided for neural networks. 
The new LTS+CM algorithm compares favorably against LTS+NN on several benchmarks: Sokoban (Boxoban), The Witness, and the 24-Sliding Tile puzzle (STP). The difference is particularly large on STP, where LTS+NN fails to solve most of the test instances while LTS+CM solves each test instance in a fraction of a second.
Furthermore, we show that LTS+CM is able to learn a policy that solves the Rubik's cube in only a few hundred expansions, which considerably improves upon previous machine learning techniques.

----

## [624] Front-to-End Bidirectional Heuristic Search with Consistent Heuristics: Enumerating and Evaluating Algorithms and Bounds

**Authors**: *Lior Siag, Shahaf S. Shperberg, Ariel Felner, Nathan R. Sturtevant*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/625](https://doi.org/10.24963/ijcai.2023/625)

**Abstract**:

Recent research on bidirectional heuristic search (BiHS) is based on the must-expand pairs theory  (MEP theory), which describes which pairs of nodes must be expanded during the search to guarantee the optimality of solutions. A separate line of research in BiHS has proposed algorithms that use lower bounds that are derived from consistent heuristics during search. This paper links these two directions, providing a comprehensive unifying view and showing that both existing and novel algorithms can be derived from the MEP theory. An extended set of bounds is formulated, encompassing both previously discovered bounds and new ones. Finally, the bounds are empirically evaluated by their contribution to the efficiency of the search

----

## [625] PathLAD+: An Improved Exact Algorithm for Subgraph Isomorphism Problem

**Authors**: *Yiyuan Wang, Chenghou Jin, Shaowei Cai, Qingwei Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/626](https://doi.org/10.24963/ijcai.2023/626)

**Abstract**:

The subgraph isomorphism problem (SIP) is a challenging problem with wide practical applications. In the last decade, despite being a theoretical hard problem, researchers design various algorithms for solving SIP. In this work, we propose three main heuristics and develop an improved exact algorithm for SIP. First, we design a probing search procedure to try whether the search procedure can successfully obtain a solution at first sight. Second, we design a novel matching ordering as a value-ordering heuristic, which uses some useful information obtained from the probing search procedure to preferentially select some promising target vertices. Third, we discuss the characteristics of different propagation methods in the context of SIP and present an adaptive propagation method to make a good balance between these methods. Experimental results on a broad range of real-world benchmarks show that our proposed algorithm performs better than state-of-the-art algorithms for the SIP.

----

## [626] A Fast Maximum k-Plex Algorithm Parameterized by the Degeneracy Gap

**Authors**: *Zhengren Wang, Yi Zhou, Chunyu Luo, Mingyu Xiao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/627](https://doi.org/10.24963/ijcai.2023/627)

**Abstract**:

Given a graph, the k-plex is a vertex set in which each vertex is not adjacent to at most k-1 other vertices in the set. The maximum k-plex problem, which asks for the largest k-plex from a given graph, is an important but computationally challenging problem in applications like graph search and community detection.  So far, there is a number of empirical algorithms  without sufficient theoretical explanations on the efficiency. We try to bridge this gap by defining a novel parameter of the input instance, g_k(G), the gap between the degeneracy bound and the size of maximum k-plex in the given graph, and presenting an exact algorithm parameterized by g_k(G). In other words, we design an algorithm with running time polynomial in the size of input graph and exponential in g_k(G) where k is a constant. Usually, g_k(G) is small and bounded by O(log(|V|)) in real-world graphs, indicating that the algorithm runs in polynomial time. We also carry out massive experiments and show that the algorithm is competitive with the state-of-the-art solvers. Additionally, for large k values such as 15 and 20, our algorithm has superior performance over existing algorithms.

----

## [627] A Mathematical Runtime Analysis of the Non-dominated Sorting Genetic Algorithm III (NSGA-III)

**Authors**: *Simon Wietheger, Benjamin Doerr*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/628](https://doi.org/10.24963/ijcai.2023/628)

**Abstract**:

The Non-dominated Sorting Genetic Algorithm II (NSGA-II) is the most prominent multi-objective evolutionary algorithm for real-world applications.
While it performs evidently well on bi-objective optimization problems, empirical studies suggest that it is less effective when applied to problems with more than two objectives. A recent mathematical runtime analysis confirmed this observation by proving the NGSA-II for an exponential number of iterations misses a constant factor of the Pareto front of the simple 3-objective OneMinMax problem.

In this work, we provide the first mathematical runtime analysis of the NSGA-III, a refinement of the NSGA-II aimed at better handling more than two objectives. 
We prove that the NSGA-III with sufficiently many reference points - a small constant factor more than the size of the Pareto front, as suggested for this algorithm - computes the complete Pareto front of the 3-objective OneMinMax benchmark in an expected number of O(n log n) iterations. This result holds for all population sizes (that are at least the size of the Pareto front). It shows a drastic advantage of the NSGA-III over the NSGA-II on this benchmark. The mathematical arguments used here and in the previous work on the NSGA-II suggest that similar findings are likely for other benchmarks with three or more objectives.

----

## [628] Probabilistic Rule Induction from Event Sequences with Logical Summary Markov Models

**Authors**: *Debarun Bhattacharjya, Oktie Hassanzadeh, Ronny Luss, Keerthiram Murugesan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/629](https://doi.org/10.24963/ijcai.2023/629)

**Abstract**:

Event sequences are widely available across application domains and there is a long history of models for representing and analyzing such datasets. Summary Markov models are a recent addition to the literature that help identify the subset of event types that influence event types of interest to a user. In this paper, we introduce logical summary Markov models, which are a family of models for event sequences that enable interpretable predictions through logical rules that relate historical predicates to the probability of observing an event type at any arbitrary position in the sequence. We illustrate their connection to prior parametric summary Markov models as well as probabilistic logic programs, and propose new models from this family along with efficient greedy search algorithms for learning them from data. The proposed models outperform relevant baselines on most datasets in an empirical investigation on a probabilistic prediction task. We also compare the number of influencers that various logical summary Markov models learn on real-world datasets, and conduct a brief exploratory qualitative study to gauge the promise of such symbolic models around guiding large language models for predicting societal events.

----

## [629] On the Complexity of Counterfactual Reasoning

**Authors**: *Yunqiu Han, Yizuo Chen, Adnan Darwiche*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/630](https://doi.org/10.24963/ijcai.2023/630)

**Abstract**:

We study the computational complexity of counterfactual reasoning in relation to the complexity of associational and interventional reasoning on structural causal models (SCMs). We show that counterfactual reasoning is no harder than associational or interventional reasoning on fully specified SCMs in the context of two computational frameworks. The first framework is based on the notion of treewidth and includes the classical variable elimination and jointree algorithms. The second framework is based on the more recent and refined notion of causal treewidth which is directed towards models with functional dependencies such as SCMs. Our results are constructive and based on bounding the (causal) treewidth of twin networks---used in standard counterfactual reasoning that contemplates two worlds, real and imaginary---to the (causal) treewidth of the underlying SCM structure. In particular, we show that the latter (causal) treewidth is no more than twice the former plus one. Hence, if associational or interventional reasoning is tractable on a fully specified SCM then counterfactual reasoning is tractable too. We extend our results to general counterfactual reasoning that requires contemplating more than two worlds and discuss applications of our results to counterfactual reasoning with partially specified SCMs that are coupled with data. We finally present empirical results that measure the gap between the complexities of counterfactual reasoning and associational/interventional reasoning on random SCMs.

----

## [630] Stability and Generalization of lp-Regularized Stochastic Learning for GCN

**Authors**: *Shiyu Liu, Linsen Wei, Shaogao Lv, Ming Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/631](https://doi.org/10.24963/ijcai.2023/631)

**Abstract**:

Graph convolutional networks (GCN) are viewed as one of the most popular representations among the variants of graph neural networks over graph data and have shown powerful performance in empirical experiments. That l2-based graph smoothing enforces the global smoothness of GCN, while (soft) l1-based sparse graph learning tends to promote signal sparsity to trade for discontinuity. This paper aims to quantify the trade-off of GCN between smoothness and sparsity, with the help of a general lp-regularized (1

Keywords:Uncertainty in AI: UAI: Graphical models

----

## [631] Approximate Inference in Logical Credal Networks

**Authors**: *Radu Marinescu, Haifeng Qian, Alexander G. Gray, Debarun Bhattacharjya, Francisco Barahona, Tian Gao, Ryan Riegel*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/632](https://doi.org/10.24963/ijcai.2023/632)

**Abstract**:

The Logical Credal Network or LCN is a recent probabilistic logic designed for effective aggregation and reasoning over multiple sources of imprecise knowledge. An LCN specifies a set of probability distributions over all interpretations of a set of logical formulas for which marginal and conditional probability bounds on their truth values are known. Inference in LCNs involves the exact solution of a non-convex non-linear program defined over an exponentially large number of non-negative real valued variables and, therefore, is limited to relatively small problems. In this paper, we present ARIEL -- a novel iterative message-passing scheme for approximate inference in LCNs. Inspired by classical belief propagation for graphical models, our method propagates messages that involve solving considerably smaller local non-linear programs. Experiments on several classes of LCNs demonstrate clearly that ARIEL yields high quality solutions compared with exact inference and scales to much larger problems than previously considered.

----

## [632] Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences

**Authors**: *Jie Qiao, Ruichu Cai, Siyu Wu, Yu Xiang, Keli Zhang, Zhifeng Hao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/633](https://doi.org/10.24963/ijcai.2023/633)

**Abstract**:

Learning causal structure among event types from discrete-time event sequences is a particularly important but challenging task. Existing methods, such as the multivariate Hawkes processes based methods, mostly boil down to learning the so-called Granger causality which assumes that the cause event happens strictly prior to its effect event. Such an assumption is often untenable beyond applications, especially when dealing with discrete-time event sequences in low-resolution; and typical discrete Hawkes processes mainly suffer from identifiability issues raised by the instantaneous effect, i.e., the causal relationship that occurred simultaneously due to the low-resolution data will not be captured by Granger causality. In this work, we propose Structure Hawkes Processes (SHPs) that leverage the instantaneous effect for learning the causal structure among events type in discrete-time event sequence. The proposed method is featured with the Expectation-Maximization of the likelihood function and a sparse optimization scheme. Theoretical results show that the instantaneous effect is a blessing rather than a curse, and the causal structure is identifiable under the existence of the instantaneous effect. Experiments on synthetic and real-world data verify the effectiveness of the proposed method.

----

## [633] Distributional Multi-Objective Decision Making

**Authors**: *Willem Röpke, Conor F. Hayes, Patrick Mannion, Enda Howley, Ann Nowé, Diederik M. Roijers*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/634](https://doi.org/10.24963/ijcai.2023/634)

**Abstract**:

For effective decision support in scenarios with conflicting objectives, sets of potentially optimal solutions can be presented to the decision maker. We explore both what policies these sets should contain and how such sets can be computed efficiently. With this in mind, we take a distributional approach and introduce a novel dominance criterion relating return distributions of policies directly. Based on this criterion, we present the distributional undominated set and show that it contains optimal policies otherwise ignored by the Pareto front. In addition, we propose the convex distributional undominated set and prove that it comprises all policies that maximise expected utility for multivariate risk-averse decision makers. We propose a novel algorithm to learn the distributional undominated set and further contribute pruning operators to reduce the set to the convex distributional undominated set. Through experiments, we demonstrate the feasibility and effectiveness of these methods, making this a valuable new approach for decision support in real-world problems.

----

## [634] Finding an ϵ-Close Minimal Variation of Parameters in Bayesian Networks

**Authors**: *Bahare Salmani, Joost-Pieter Katoen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/635](https://doi.org/10.24963/ijcai.2023/635)

**Abstract**:

This paper addresses the ε-close parameter tuning problem for Bayesian
networks (BNs): find a minimal ε-close amendment of probability entries
in a given set of (rows in) conditional probability tables that make a
given quantitative constraint on the BN valid. Based on the
state-of-the-art “region verification” techniques for parametric Markov
chains, we propose an algorithm whose capabilities go
beyond any existing techniques. Our experiments show that ε-close tuning
of large BN benchmarks with up to eight parameters is feasible. In
particular, by allowing (i) varied parameters in multiple CPTs and (ii)
inter-CPT parameter dependencies, we treat subclasses of parametric BNs
that have received scant attention so far.

----

## [635] The Hardness of Reasoning about Probabilities and Causality

**Authors**: *Benito van der Zander, Markus Bläser, Maciej Liskiewicz*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/636](https://doi.org/10.24963/ijcai.2023/636)

**Abstract**:

We study formal languages which are capable of fully expressing quantitative probabilistic reasoning and do-calculus reasoning for causal effects, from a computational complexity perspective. 
We focus on satisfiability problems whose instance formulas allow expressing many tasks in probabilistic and causal inference.  
The main contribution of this work is establishing the exact computational complexity of these satisfiability problems. 
We introduce a new natural complexity class, named succ∃R, which can be viewed as a succinct variant of the well-studied class ∃R, and show that these problems are complete for succ∃R. 
Our results imply even stronger limitations on the use of algorithmic methods for reasoning about probabilities and causality than  previous state-of-the-art results that rely only on the NP- or ∃R-completeness of the satisfiability problems for some restricted languages.

----

## [636] Safe Reinforcement Learning via Probabilistic Logic Shields

**Authors**: *Wen-Chi Yang, Giuseppe Marra, Gavin Rens, Luc De Raedt*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/637](https://doi.org/10.24963/ijcai.2023/637)

**Abstract**:

Safe Reinforcement learning (Safe RL) aims at learning optimal policies while staying safe. A popular solution to Safe RL is shielding, which uses a logical safety specification to prevent an RL agent from taking unsafe actions. However, traditional shielding techniques are difficult to integrate with continuous, end-to-end deep RL methods. To this end, we introduce Probabilistic Logic Policy Gradient (PLPG). PLPG is a model-based Safe RL technique that uses probabilistic logic programming to model logical safety constraints as differentiable functions. Therefore, PLPG can be seamlessly applied to any policy gradient algorithm while still providing the same convergence guarantees. In our experiments, we show that PLPG learns safer and more rewarding policies compared to other state-of-the-art shielding techniques.

----

## [637] Quantifying Consistency and Information Loss for Causal Abstraction Learning

**Authors**: *Fabio Massimo Zennaro, Paolo Turrini, Theodoros Damoulas*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/638](https://doi.org/10.24963/ijcai.2023/638)

**Abstract**:

Structural causal models provide a formalism to express causal relations between variables of interest. Models and variables can represent a system at different levels of abstraction, whereby relations may be coarsened and refined according to the need of a modeller.
However, switching between different levels of abstraction requires evaluating a trade-off between the consistency and the information loss among different models.
In this paper we introduce a family of interventional measures that an agent may use to evaluate such a trade-off. We consider four measures suited for different tasks, analyze their properties, and propose algorithms to evaluate and learn causal abstractions. Finally, we illustrate the flexibility of our setup by empirically showing how different measures and algorithmic choices may lead to different abstractions.

----

## [638] Max Markov Chain

**Authors**: *Yu Zhang, Mitchell Bucklew*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/639](https://doi.org/10.24963/ijcai.2023/639)

**Abstract**:

In this paper, we introduce Max Markov Chain (MMC), a novel model for sequential data with sparse correlations among the state variables.
It may also be viewed as a special class of approximate models for High-order Markov Chains (HMCs). 
MMC is desirable for domains where the sparse correlations are long-term and vary in their temporal stretches. 
Although generally intractable, parameter optimization for MMC can be solved analytically. 
However, based on this result,
we derive an approximate solution that is highly efficient empirically.
When compared with HMC and approximate HMC models, MMC 
combines  better sample efficiency, model parsimony, and an outstanding computational advantage. 
Such a quality allows MMC to scale to large domains 
where the competing models would struggle to perform. 
We compare MMC with several baselines with synthetic and real-world datasets to demonstrate MMC as a valuable alternative for  stochastic modeling.

----

## [639] Evaluating Human-AI Interaction via Usability, User Experience and Acceptance Measures for MMM-C: A Creative AI System for Music Composition

**Authors**: *Renaud Bougueng Tchemeube, Jeffrey Ens, Cale Plut, Philippe Pasquier, Maryam Safi, Yvan Grabit, Jean-Baptiste Rolland*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/640](https://doi.org/10.24963/ijcai.2023/640)

**Abstract**:

With the rise of artificial intelligence (AI), there has been increasing interest in human-AI co-creation in a variety of artistic domains including music as AI-driven systems are frequently able to generate human-competitive artifacts. Now, the implications of such systems for the musical practice are being investigated. This paper reports on a thorough evaluation of the user adoption of the Multi-Track Music Machine (MMM) as a minimal co-creative AI tool for music composers. To do this, we integrate MMM into Cubase, a popular Digital Audio Workstation (DAW), by producing a "1-parameter" plugin interface named MMM-Cubase, which enables human-AI co-composition. We conduct a 3-part mixed method study measuring usability, user experience and technology acceptance of the system across two groups of expert-level  composers: hobbyists and professionals. Results show positive usability and acceptance scores. Users report experiences of novelty, surprise and ease of use from using the system, and limitations on controllability and predictability of the interface when generating music. Findings indicate no significant difference between the two user groups.

----

## [640] The ACCompanion: Combining Reactivity, Robustness, and Musical Expressivity in an Automatic Piano Accompanist

**Authors**: *Carlos Cancino Chacón, Silvan Peter, Patricia Hu, Emmanouil Karystinaios, Florian Henkel, Francesco Foscarin, Gerhard Widmer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/641](https://doi.org/10.24963/ijcai.2023/641)

**Abstract**:

This paper introduces the ACCompanion, an expressive accompaniment system. Similarly to a musician who accompanies a soloist playing a given musical piece, our system can produce a human-like rendition of the accompaniment part that follows the soloist's choices in terms of tempo, dynamics, and articulation. The ACCompanion works in the symbolic domain, i.e., it needs a musical instrument capable of producing and playing MIDI data, with explicitly encoded onset, offset, and pitch for each played note. We describe the components that go into such a system, from real-time score following and prediction to expressive performance generation and online adaptation to the expressive choices of the human player. Based on our experience with repeated live demonstrations in front of various audiences, we offer an analysis of the challenges of combining these components into a system that is highly reactive and precise, while still a reliable musical partner, robust to possible performance errors and responsive to expressive variations.

----

## [641] TeSTNeRF: Text-Driven 3D Style Transfer via Cross-Modal Learning

**Authors**: *Jiafu Chen, Boyan Ji, Zhanjie Zhang, Tianyi Chu, Zhiwen Zuo, Lei Zhao, Wei Xing, Dongming Lu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/642](https://doi.org/10.24963/ijcai.2023/642)

**Abstract**:

Text-driven 3D style transfer aims at stylizing a scene according to the text and generating arbitrary novel views with consistency. Simply combining image/video style transfer methods and novel view synthesis methods results in flickering when changing viewpoints, while existing 3D style transfer methods learn styles from images instead of texts. To address this problem, we for the first time design an efficient text-driven model for 3D style transfer, named TeSTNeRF, to stylize the scene using texts via cross-modal learning: we leverage an advanced text encoder to embed the texts in order to control 3D style transfer and align the input text and output stylized images in latent space. Furthermore, to obtain better visual results, we introduce style supervision, learning feature statistics from style images and utilizing 2D stylization results to rectify abrupt color spill. Extensive experiments demonstrate that TeSTNeRF significantly outperforms existing methods and provides a new way to guide 3D style transfer.

----

## [642] Graph-based Polyphonic Multitrack Music Generation

**Authors**: *Emanuele Cosenza, Andrea Valenti, Davide Bacciu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/643](https://doi.org/10.24963/ijcai.2023/643)

**Abstract**:

Graphs can be leveraged to model polyphonic multitrack symbolic music, where notes, chords and entire sections may be linked at different levels of the musical hierarchy by tonal and rhythmic relationships. Nonetheless, there is a lack of works that consider graph representations in the context of deep learning systems for music generation. This paper bridges this gap by introducing a novel graph representation for music and a deep Variational Autoencoder that generates the structure and the content of musical graphs separately, one after the other, with a hierarchical architecture that matches the structural priors of music. By separating the structure and content of musical graphs, it is possible to condition generation by specifying which instruments are played at certain times. This opens the door to a new form of human-computer interaction in the context of music co-creation. After training the model on existing MIDI datasets, the experiments show that the model is able to generate appealing short and long musical sequences and to realistically interpolate between them, producing music that is tonally and rhythmically consistent. Finally, the visualization of the embeddings shows that the model is able to organize its latent space in accordance with known musical concepts.

----

## [643] Towards Symbiotic Creativity: A Methodological Approach to Compare Human and AI Robotic Dance Creations

**Authors**: *Allegra De Filippo, Luca Giuliani, Eleonora Mancini, Andrea Borghesi, Paola Mello, Michela Milano*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/644](https://doi.org/10.24963/ijcai.2023/644)

**Abstract**:

Artificial Intelligence (AI) has gradually attracted attention in the field of artistic creation, resulting in a debate on the evaluation of AI artistic outputs. However, there is a lack of common criteria for objective artistic evaluation both of human and AI creations. This is a frequent issue in the field of dance, where different performance metrics focus either on evaluating human or computational skills separately. This work proposes a methodological approach for the artistic evaluation of both AI and human artistic creations in the field of robotic dance. First, we define a series of common initial constraints to create robotic dance choreographies in a balanced initial setting, in collaboration with a group of human dancers and choreographer. Then, we compare both creation processes through a human audience evaluation. Finally, we investigate which choreography aspects (e.g., the music genre) have the largest impact on the evaluation, and we provide useful guidelines and future research directions for the analysis of interconnections between AI and human dance creation.

----

## [644] Automating Rigid Origami Design

**Authors**: *Jeremia Geiger, Karolis Martinkus, Oliver Richter, Roger Wattenhofer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/645](https://doi.org/10.24963/ijcai.2023/645)

**Abstract**:

Rigid origami has shown potential in large diversity of practical applications. However, current rigid origami crease pattern design mostly relies on known tessellations. This strongly limits the diversity and novelty of patterns that can be created. In this work, we build upon the recently developed principle of three units method to formulate rigid origami design as a discrete optimization problem, the rigid origami game. Our implementation allows for a simple definition of diverse objectives and thereby expands the potential of rigid origami further to optimized, application-specific crease patterns. We showcase the flexibility of our formulation through use of a diverse set of search methods in several illustrative case studies. We are not only able to construct various patterns that approximate given target shapes, but to also specify abstract, function-based rewards which result in novel, foldable and functional designs for everyday objects.

----

## [645] Collaborative Neural Rendering Using Anime Character Sheets

**Authors**: *Zuzeng Lin, Ailin Huang, Zhewei Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/646](https://doi.org/10.24963/ijcai.2023/646)

**Abstract**:

Drawing images of characters with desired poses is an essential but laborious task in anime production. Assisting artists to create is a research hotspot in recent years. In this paper, we present the Collaborative Neural Rendering (CoNR) method, which creates new images for specified poses from a few reference images (AKA Character Sheets). In general, the diverse hairstyles and garments of anime characters defies the employment of universal body models like SMPL, which fits in most nude human shapes. To overcome this, CoNR uses a compact and easy-to-obtain landmark encoding to avoid creating a unified UV mapping in the pipeline. In addition, the performance of CoNR can be significantly improved when referring to multiple reference images, thanks to feature space cross-view warping in a carefully designed neural network. Moreover, we have collected a character sheet dataset containing over 700,000 hand-drawn and synthesized images of diverse poses to facilitate research in this area. The code and dataset is available at https://github.com/megvii-research/IJCAI2023-CoNR.

----

## [646] IberianVoxel: Automatic Completion of Iberian Ceramics for Cultural Heritage Studies

**Authors**: *Pablo Navarro, Celia Cintas, Manuel J. Lucena, José Manuel Fuertes, Antonio Jesús Rueda, Rafael Jesús Segura, Carlos J. Ogáyar-Anguita, Rolando González-José, Claudio Delrieux*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/647](https://doi.org/10.24963/ijcai.2023/647)

**Abstract**:

Accurate completion of archaeological artifacts is a critical aspect in several archaeological studies, including documentation of variations in style, inference of chronological and ethnic groups, and trading routes trends, among many others. However, most available pottery is fragmented, leading to missing textural and morphological cues. 
Currently, the reassembly and completion of fragmented ceramics is a daunting and time-consuming task, done almost exclusively by hand, which requires the physical manipulation of the fragments. 
To overcome the challenges of manual reconstruction, reduce the materials' exposure and deterioration, and improve the quality of reconstructed samples, we present IberianVoxel, a novel 3D Autoencoder Generative Adversarial Network (3D AE-GAN) framework tested on an extensive database with complete and fragmented references. 
We generated a collection of 1001 3D voxelized samples and their fragmented references from Iberian wheel-made pottery profiles. The fragments generated are stratified into different size groups and across multiple pottery classes. 
Lastly, we provide quantitative and qualitative assessments to measure the quality of the reconstructed voxelized samples by our proposed method and archaeologists' evaluation.

----

## [647] Discrete Diffusion Probabilistic Models for Symbolic Music Generation

**Authors**: *Matthias Plasser, Silvan Peter, Gerhard Widmer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/648](https://doi.org/10.24963/ijcai.2023/648)

**Abstract**:

Denoising Diffusion Probabilistic Models (DDPMs) have made great strides in generating high-quality samples in both discrete and continuous domains.
However, Discrete DDPMs (D3PMs) have yet to be applied to the domain of Symbolic Music.
This work presents the direct generation of Polyphonic Symbolic Music using D3PMs.
Our model exhibits state-of-the-art sample quality, according to current quantitative evaluation metrics, and allows for flexible infilling at the note level.
We further show, that our models are accessible to post-hoc classifier guidance, widening the scope of possible applications.
However, we also cast a critical view on quantitative evaluation of music sample quality via statistical metrics, and present a simple algorithm that can confound our metrics with completely spurious, non-musical samples.

----

## [648] Learn and Sample Together: Collaborative Generation for Graphic Design Layout

**Authors**: *Haohan Weng, Danqing Huang, Tong Zhang, Chin-Yew Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/649](https://doi.org/10.24963/ijcai.2023/649)

**Abstract**:

In the process of graphic layout generation, user specifications including element attributes and their relationships are commonly used to constrain the layouts (e.g.,"put the image above the button''). It is natural to encode spatial constraints between elements using a graph. This paper presents a two-stage generation framework: a spatial graph generator and a subsequent layout decoder which is conditioned on the previous output graph. Training the two highly dependent networks separately as in previous work, we observe that the graph generator generates out-of-distribution graphs with a high frequency, which are unseen to the layout decoder during training and thus leads to huge performance drop in inference. To coordinate the two networks more effectively, we propose a novel collaborative generation strategy to perform round-way knowledge transfer between the networks in both training and inference. Experiment results on three public datasets show that our model greatly benefits from the collaborative generation and has achieved the state-of-the-art performance. Furthermore, we conduct an in-depth analysis to better understand the effectiveness of graph condition modeling.

----

## [649] DiffuseStyleGesture: Stylized Audio-Driven Co-Speech Gesture Generation with Diffusion Models

**Authors**: *Sicheng Yang, Zhiyong Wu, Minglei Li, Zhensong Zhang, Lei Hao, Weihong Bao, Ming Cheng, Long Xiao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/650](https://doi.org/10.24963/ijcai.2023/650)

**Abstract**:

The art of communication beyond speech there are gestures. The automatic co-speech gesture generation draws much attention in computer animation. It is a challenging task due to the diversity of gestures and the difficulty of matching the rhythm and semantics of the gesture to the corresponding speech. To address these problems, we present DiffuseStyleGesture, a diffusion model based speech-driven gesture generation approach. It generates high-quality, speech-matched, stylized, and diverse co-speech gestures based on given speeches of arbitrary length. Specifically, we introduce cross-local attention and self-attention to the gesture diffusion pipeline to generate better speech matched and realistic gestures. We then train our model with classifier-free guidance to control the gesture style by interpolation or extrapolation. Additionally, we improve the diversity of generated gestures with different initial gestures and noise. Extensive experiments show that our method outperforms recent approaches on speech-driven gesture generation. Our code, pre-trained models, and demos are available at https://github.com/YoungSeng/DiffuseStyleGesture.

----

## [650] NAS-FM: Neural Architecture Search for Tunable and Interpretable Sound Synthesis Based on Frequency Modulation

**Authors**: *Zhen Ye, Wei Xue, Xu Tan, Qifeng Liu, Yike Guo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/651](https://doi.org/10.24963/ijcai.2023/651)

**Abstract**:

Developing digital sound synthesizers is crucial to the music industry as it provides a low-cost way to produce high-quality sounds with rich timbres. Existing traditional synthesizers often require substantial expertise to determine the overall framework of a synthesizer and the parameters of submodules. Since expert knowledge is hard to acquire,  it hinders the flexibility to quickly design and tune digital synthesizers for diverse sounds. In this paper, we propose ``NAS-FM'', which adopts neural architecture search (NAS) to build a differentiable frequency modulation (FM) synthesizer. Tunable synthesizers with interpretable controls can be developed automatically from sounds without any prior expert knowledge and manual operating costs. In detail, we train a supernet with a specifically designed search space, including predicting the envelopes of carriers and modulators with different frequency ratios. An evolutionary search algorithm with adaptive oscillator size is then developed to find the optimal relationship between oscillators and the frequency ratio of FM. Extensive experiments on recordings of different instrument sounds show that our algorithm can build a synthesizer fully automatically, achieving better results than handcrafted synthesizers. Audio samples are available at https://nas-fm.github.io/

----

## [651] Q&A: Query-Based Representation Learning for Multi-Track Symbolic Music re-Arrangement

**Authors**: *Jingwei Zhao, Gus Xia, Ye Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/652](https://doi.org/10.24963/ijcai.2023/652)

**Abstract**:

Music rearrangement is a common music practice of reconstructing and reconceptualizing a piece using new composition or instrumentation styles, which is also an important task of automatic music generation. Existing studies typically model the mapping from a source piece to a target piece via supervised learning. In this paper, we tackle rearrangement problems via self-supervised learning, in which the mapping styles can be regarded as conditions and controlled in a flexible way. Specifically, we are inspired by the representation disentanglement idea and propose Q&A, a query-based algorithm for multi-track music rearrangement under an encoder-decoder framework. Q&A learns both a content representation from the mixture and function (style) representations from each individual track, while the latter queries the former in order to rearrange a new piece. Our current model focuses on popular music and provides a controllable pathway to four scenarios: 1) re-instrumentation, 2) piano cover generation, 3) orchestration, and 4) voice separation. Experiments show that our query system achieves high-quality rearrangement results with delicate multi-track structures, significantly outperforming the baselines.

----

## [652] Fairness and Representation in Satellite-Based Poverty Maps: Evidence of Urban-Rural Disparities and Their Impacts on Downstream Policy

**Authors**: *Emily L. Aiken, Esther Rolf, Joshua Blumenstock*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/653](https://doi.org/10.24963/ijcai.2023/653)

**Abstract**:

Poverty maps derived from satellite imagery are increasingly used to inform high-stakes policy decisions, such as the allocation of humanitarian aid and the distribution of government resources. Such poverty maps are typically constructed by training machine learning algorithms on a relatively modest amount of ``ground truth" data from surveys, and then predicting poverty levels in areas where imagery exists but surveys do not. Using survey and satellite data from ten countries, this paper investigates disparities in representation, systematic biases in prediction errors, and fairness concerns in satellite-based poverty mapping across urban and rural lines, and shows how these phenomena affect the validity of policies based on predicted maps. Our findings highlight the importance of careful error and bias analysis before using satellite-based poverty maps in real-world policy decisions.

----

## [653] Forecasting Soil Moisture Using Domain Inspired Temporal Graph Convolution Neural Networks To Guide Sustainable Crop Management

**Authors**: *Muneeza Azmat, Malvern Madondo, Arun Bawa, Kelsey Dipietro, Raya Horesh, Michael Jacobs, Raghavan Srinivasan, Fearghal O'Donncha*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/654](https://doi.org/10.24963/ijcai.2023/654)

**Abstract**:

Agriculture faces unprecedented challenges due to climate change, population growth, and water scarcity. These challenges highlight the need for efficient resource usage to optimize crop production. Conventional techniques for forecasting hydrological response features, such as soil moisture, rely on physics-based and empirical hydrological models, which necessitate significant time and domain expertise. Drawing inspiration from traditional hydrological modeling, a novel temporal graph convolution neural network has been constructed. This involves grouping units based on their time-varying hydrological properties, constructing graph topologies for each cluster based on similarity using dynamic time warping, and utilizing graph convolutions and a gated recurrent neural network to forecast soil moisture. The method has been trained, validated, and tested on field-scale time series data spanning 40 years in northeastern United States. Results show that using domain-inspired clustering with time series graph neural networks is more effective in forecasting soil moisture than existing models. This framework is being deployed as part of a pro bono social impact program that leverages hybrid cloud and AI technologies to enhance and scale non-profit and government organizations. The trained models are currently being deployed on a series of small-holding farms in central Texas.

----

## [654] Toward Job Recommendation for All

**Authors**: *Guillaume Bied, Solal Nathan, Elia Perennes, Morgane Hoffmann, Philippe Caillou, Bruno Crépon, Christophe Gaillac, Michèle Sebag*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/655](https://doi.org/10.24963/ijcai.2023/655)

**Abstract**:

This paper presents a job recommendation algorithm designed and validated in the context of the French Public Employment Service. The challenges, owing to the confidential data policy, are related with the extreme sparsity of the interaction matrix and the mandatory scalability of the algorithm, aimed to deliver recommendations to millions of job seekers in quasi real-time, considering hundreds of thousands of job ads. The experimental validation of the approach shows similar or better performances than the state of the art in terms of recall, with a gain in inference time of 2 orders of magnitude. The study includes some fairness analysis of the recommendation algorithm. The gender-related gap is shown to be statistically similar in the true data and in the counter-factual data built from the recommendations.

----

## [655] Fast and Differentially Private Fair Clustering

**Authors**: *Junyoung Byun, Jaewook Lee*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/656](https://doi.org/10.24963/ijcai.2023/656)

**Abstract**:

This study presents the first differentially private and fair clustering method, built on the recently proposed density-based fair clustering approach. The method addresses the limitations of fair clustering algorithms that necessitate the use of sensitive personal information during training or inference phases. Two novel solutions, the Gaussian mixture density function and Voronoi cell, are proposed to enhance the method's performance in terms of privacy, fairness, and utility compared to previous methods. The experimental results on both synthetic and real-world data confirm the compatibility of the proposed method with differential privacy, achieving a better fairness-utility trade-off than existing methods when privacy is not considered. Moreover, the proposed method requires significantly less computation time, being at least 3.7 times faster than the state-of-the-art.

----

## [656] Supporting Sustainable Agroecological Initiatives for Small Farmers through Constraint Programming

**Authors**: *Margot Challand, Philippe Vismara, Dimitri Justeau-Allaire, Stéphane de Tourdonnet*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/657](https://doi.org/10.24963/ijcai.2023/657)

**Abstract**:

Meeting the UN's objective of developing sustainable agriculture requires, in particular, accompanying small farms in their agroecological transition. This transition often requires making the agrosystem more complex and increasing the number of crops to increase biodiversity and ecosystem services. This paper introduces a flexible model based on Constraint Programming (CP) to address the crop allocation problem. This problem takes a cropping calendar as input and aims at allocating crops to respect several constraints.  We have shown that it is possible to model both agroecological and operational constraints at the level of a small farm. Experiments on an organic micro-farm have shown that it is possible to combine these constraints to design very different cropping scenarios and that our approach can apply to real situations. Our promising results in this case study also demonstrate the potential of AI-based tools to address small farmers' challenges in the context of the sustainable agriculture transition.

----

## [657] Towards Gender Fairness for Mental Health Prediction

**Authors**: *Jiaee Cheong, Selim Kuzucu, Sinan Kalkan, Hatice Gunes*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/658](https://doi.org/10.24963/ijcai.2023/658)

**Abstract**:

Mental health is becoming an increasingly prominent health challenge. Despite a plethora of studies analysing and mitigating bias for a variety of tasks such as face recognition and credit scoring, research on machine learning (ML) fairness for mental health has been sparse to date. In this work, we focus on gender bias in mental health and make the following contributions. First, we examine whether bias exists in existing mental health datasets and algorithms. Our experiments were conducted using Depresjon, Psykose and D-Vlog. We identify that both data and algorithmic bias exist. Second, we analyse strategies that can be deployed at the pre-processing, in-processing and post-processing stages to mitigate for bias and evaluate their effectiveness. Third, we investigate factors that impact the efficacy of existing bias mitigation strategies and outline recommendations to achieve greater gender fairness for mental health. Upon obtaining counter-intuitive results on D-Vlog dataset, we undertake further experiments and analyses, and provide practical suggestions to avoid hampering bias mitigation efforts in ML for mental health.

----

## [658] Addressing Weak Decision Boundaries in Image Classification by Leveraging Web Search and Generative Models

**Authors**: *Preetam Prabhu Srikar Dammu, Yunhe Feng, Chirag Shah*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/659](https://doi.org/10.24963/ijcai.2023/659)

**Abstract**:

Machine learning (ML) technologies are known to be riddled with ethical and operational problems, however, we are witnessing an increasing thrust by businesses to deploy them in sensitive applications. One major issue among many is that ML models do not perform equally well for underrepresented groups. This puts vulnerable populations in an even disadvantaged and unfavorable position. We propose an approach that leverages the power of web search and generative models to alleviate some of the shortcomings of discriminative models. We demonstrate our method on an image classification problem using ImageNet's People Subtree subset, and show that it is effective in enhancing robustness and mitigating bias in certain classes that represent vulnerable populations (e.g., female doctor of color). Our new method is able to (1) identify weak decision boundaries for such classes; (2) construct search queries for Google as well as text for generating images through DALL-E 2 and Stable Diffusion; and (3) show how these newly captured training samples could alleviate population bias issue. While still improving the model's overall performance considerably, we achieve a significant reduction (77.30%) in the model's gender accuracy disparity. In addition to these improvements, we observed a notable enhancement in the classifier's decision boundary, as it is characterized by fewer weakspots and an increased separation between classes. Although we showcase our method on vulnerable populations in this study, the proposed technique is extendable to a wide range of problems and domains.

----

## [659] Limited Resource Allocation in a Non-Markovian World: The Case of Maternal and Child Healthcare

**Authors**: *Panayiotis Danassis, Shresth Verma, Jackson A. Killian, Aparna Taneja, Milind Tambe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/660](https://doi.org/10.24963/ijcai.2023/660)

**Abstract**:

The success of many healthcare programs depends on participants' adherence. We consider the problem of scheduling interventions in low resource settings (e.g., placing timely support calls from health workers) to increase adherence and/or engagement. Past works have successfully developed several classes of Restless Multi-armed Bandit (RMAB) based solutions for this problem. Nevertheless, all past RMAB approaches assume that the participants' behaviour follows the Markov property. We demonstrate significant deviations from the Markov assumption on real-world data on a maternal health awareness program from our partner NGO, ARMMAN. Moreover, we extend RMABs to continuous state spaces, a previously understudied area. To tackle the generalised non-Markovian RMAB setting we (i) model each participant's trajectory as a time-series, (ii) leverage the power of time-series forecasting models to learn complex patterns and dynamics to predict future states, and (iii) propose the Time-series Arm Ranking Index (TARI) policy, a novel algorithm that selects the RMAB arms that will benefit the most from an intervention, given our future state predictions. We evaluate our approach on both synthetic data, and a secondary analysis on real data from ARMMAN, and demonstrate significant increase in engagement compared to the SOTA, deployed Whittle index solution. This translates to 16.3 hours of additional content listened, 90.8% more engagement drops prevented, and reaching more than twice as many high dropout-risk beneficiaries.

----

## [660] Disentangling Societal Inequality from Model Biases: Gender Inequality in Divorce Court Proceedings

**Authors**: *Sujan Dutta, Parth Srivastava, Vaishnavi Solunke, Swaprava Nath, Ashiqur R. KhudaBukhsh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/661](https://doi.org/10.24963/ijcai.2023/661)

**Abstract**:

Divorce is the legal dissolution of a marriage by a court. Since this is usually an unpleasant outcome of a marital union, each party may have reasons to call the decision to quit which is generally documented in detail in the court proceedings. Via a substantial corpus of 17,306 court proceedings, this paper investigates gender inequality through the lens of divorce court proceedings. To our knowledge, this is the first-ever large-scale computational analysis of gender inequality in Indian divorce, a taboo-topic for ages. While emerging data sources (e.g., public court records made available on the web) on sensitive societal issues hold promise in aiding social science research, biases present in cutting-edge natural language processing (NLP) methods may interfere with or affect such studies. A thorough analysis of potential gaps and limitations present in extant NLP resources is thus of paramount importance. In this paper, on the methodological side,  we demonstrate that existing NLP resources required several non-trivial modifications to quantify societal inequalities. On the substantive side, we find that while a large number of court cases perhaps suggest changing norms in India where women are increasingly challenging patriarchy, AI-powered analyses of these court proceedings indicate striking gender inequality with women often subjected to domestic violence.

----

## [661] Sign Language-to-Text Dictionary with Lightweight Transformer Models

**Authors**: *Jérôme Fink, Pierre Poitier, Maxime André, Loup Meurice, Benoît Frénay, Anthony Cleve, Bruno Dumas, Laurence Meurant*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/662](https://doi.org/10.24963/ijcai.2023/662)

**Abstract**:

The recent advances in deep learning have been beneficial to automatic sign language recognition (SLR). However, free-to-access, usable, and accessible tools are still not widely available to the deaf community. The need for a sign language-to-text dictionary was raised by a bilingual deaf school in Belgium and linguist experts in sign languages (SL) in order to improve the autonomy of students. To meet that need, an efficient SLR system was built based on a specific transformer model. The proposed system is able to recognize 700 different signs, with a top-10 accuracy of 83%. Those results are competitive with other systems in the literature while using 10 times less parameters than existing solutions. The integration of this model into a usable and accessible web application for the dictionary is also introduced. A user-centered human-computer interaction (HCI) methodology was followed to design and implement the user interface. To the best of our knowledge, this is the first publicly released sign language-to-text dictionary using video captured by a standard camera.

----

## [662] Find Rhinos without Finding Rhinos: Active Learning with Multimodal Imagery of South African Rhino Habitats

**Authors**: *Lucia Gordon, Nikhil Behari, Samuel Collier, Elizabeth Bondi-Kelly, Jackson A. Killian, Catherine Ressijac, Peter Boucher, Andrew Davies, Milind Tambe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/663](https://doi.org/10.24963/ijcai.2023/663)

**Abstract**:

Much of Earth's charismatic megafauna is endangered by human activities, particularly the rhino, which is at risk of extinction due to the poaching crisis in Africa. Monitoring rhinos' movement is crucial to their protection but has unfortunately proven difficult because rhinos are elusive. Therefore, instead of tracking rhinos, we propose the novel approach of mapping communal defecation sites, called middens, which give information about rhinos' spatial behavior valuable to anti-poaching, management, and reintroduction efforts. This paper provides the first-ever mapping of rhino midden locations by building classifiers to detect them using remotely sensed thermal, RGB, and LiDAR imagery in passive and active learning settings. As existing active learning methods perform poorly due to the extreme class imbalance in our dataset, we design MultimodAL, an active learning system employing a ranking technique and multimodality to achieve competitive performance with passive learning models with 94% fewer labels. Our methods could therefore save over 76 hours in labeling time when used on a similarly-sized dataset. Unexpectedly, our midden map reveals that rhino middens are not randomly distributed throughout the landscape; rather, they are clustered. Consequently, rangers should be targeted at areas with high midden densities to strengthen anti-poaching efforts, in line with UN Target 15.7.

----

## [663] CGS: Coupled Growth and Survival Model with Cohort Fairness

**Authors**: *Erhu He, Yue Wan, Benjamin H. Letcher, Jennifer H. Fair, Yiqun Xie, Xiaowei Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/664](https://doi.org/10.24963/ijcai.2023/664)

**Abstract**:

Fish modeling in complex environments is critical for understanding drivers of population dynamics in aquatic systems. This paper proposes a Bayesian network method for modeling fish survival and growth over multiple connected rivers. Traditional fish survival models capture the effect of multiple environmental drivers (e.g., stream temperature, stream flow) by adding different variables, which increases model complexity and results in very long and impractical run times (i.e., weeks). We propose a coupled survival-growth model that leverages the observations from both sources simultaneously. It also integrates the Bayesian process into the neural network model to efficiently capture complex variable relationships in the system while also conforming to known survival processes used in existing fish models.  To further reduce the performance disparity of fish body length across cohorts, we propose two approaches for enforcing fairness by the adjustment of training priorities and data augmentation. The results based on a real-world fish dataset collected in Massachusetts, US demonstrate that the proposed method can greatly improve prediction accuracy in modeling survival and body length compared to independent models on survival and growth, and effectively reduce the performance disparity across cohorts. The fish growth and movement patterns discovered by the proposed model are also consistent with prior studies in the same region, while vastly reducing run times and memory requirements.

----

## [664] Decoding the Underlying Meaning of Multimodal Hateful Memes

**Authors**: *Ming Shan Hee, Wen-Haw Chong, Roy Ka-Wei Lee*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/665](https://doi.org/10.24963/ijcai.2023/665)

**Abstract**:

Recent studies have proposed models that yielded promising performance for the hateful meme classification task. Nevertheless, these proposed models do not generate interpretable explanations that uncover the underlying meaning and support the classification output. A major reason for the lack of explainable hateful meme methods is the absence of a hateful meme dataset that contains ground truth explanations for benchmarking or training. Intuitively, having such explanations can educate and assist content moderators in interpreting and removing flagged hateful memes. This paper address this research gap by introducing Hateful meme with Reasons Dataset (HatReD), which is a new multimodal hateful meme dataset annotated with the underlying hateful contextual reasons. We also define a new conditional generation task that aims to automatically generate underlying reasons to explain hateful memes and establish the baseline performance of state-of-the-art pre-trained language models on this task.  We further demonstrate the usefulness of HatReD by analyzing the challenges of the new conditional generation task in explaining memes in seen and unseen domains. The dataset and benchmark models are made available here: https://github.com/Social-AI-Studio/HatRed

----

## [665] Computationally Assisted Quality Control for Public Health Data Streams

**Authors**: *Ananya Joshi, Kathryn Mazaitis, Roni Rosenfeld, Bryan Wilder*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/666](https://doi.org/10.24963/ijcai.2023/666)

**Abstract**:

Irregularities in public health data streams (like COVID-19 Cases) hamper data-driven decision-making for public health stakeholders. A real-time, computer-generated list of the most important, outlying data points from thousands of public health data streams could assist an expert reviewer in identifying these irregularities. However, existing outlier detection frameworks perform poorly on this task because they do not account for the data volume or for the statistical properties of public health streams. Accordingly, we developed FlaSH (Flagging Streams in public Health), a practical outlier detection framework for public health data users that uses simple, scalable models to capture these statistical properties explicitly. In an experiment where human experts evaluate FlaSH and existing methods (including deep learning approaches), FlaSH scales to the data volume of this task, matches or exceeds these other methods in mean accuracy, and identifies the outlier points that users empirically rate as more helpful. Based on these results, FlaSH has been deployed on data streams used by public health stakeholders.

----

## [666] For Women, Life, Freedom: A Participatory AI-Based Social Web Analysis of a Watershed Moment in Iran's Gender Struggles

**Authors**: *Adel Khorramrouz, Sujan Dutta, Ashiqur R. KhudaBukhsh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/667](https://doi.org/10.24963/ijcai.2023/667)

**Abstract**:

In this paper, we present a computational analysis of the Persian language Twitter discourse with the aim to estimate the shift in stance toward gender equality following the death of Mahsa Amini in police custody. We present an ensemble active learning pipeline to train a stance classifier. Our novelty lies in the involvement of Iranian women in an active role as annotators in building this AI system. Our annotators not only provide labels, but they also suggest valuable keywords for more meaningful corpus creation as well as provide short example documents for a guided sampling step. Our analyses indicate that Mahsa Amini's death triggered polarized Persian language discourse where both fractions of negative and positive tweets toward gender equality increased. The increase in positive tweets was slightly greater than the increase in negative tweets.  We also observe that with respect to account creation time, between the state-aligned Twitter accounts and pro-protest Twitter accounts, pro-protest accounts are more similar to baseline Persian Twitter activity.

----

## [667] Building a Personalized Messaging System for Health Intervention in Underprivileged Regions Using Reinforcement Learning

**Authors**: *Sarah Eve Kinsey, Jack Wolf, Nalini Saligram, Varun Ramesan, Meeta Walavalkar, Nidhi Jaswal, Sandhya Ramalingam, Arunesh Sinha, Thanh Hong Nguyen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/668](https://doi.org/10.24963/ijcai.2023/668)

**Abstract**:

This work builds an effective AI-based  message generation system for diabetes prevention in rural areas, where the diabetes rate has been increasing at an alarming rate. The messages contain information about diabetes causes and complications and the impact of nutrition and fitness on preventing diabetes. We propose to apply reinforcement learning (RL) to optimize our message selection policy over time, tailoring our messages to align with each individual participant's needs and preferences. We conduct an extensive field study in a large country in Asia which involves more than 1000 participants who are local villagers and they receive messages generated by our system, over a period of six months. Our analysis shows that with the use of AI, we can deliver significant improvements in the participants' diabetes-related knowledge, physical activity levels, and high-fat food avoidance, when compared to a static message set. Furthermore, we build a new neural network based behavior model to predict behavior changes of participants, trained on data collected during our study. By exploiting underlying characteristics of health-related behavior, we manage to significantly improve the prediction accuracy of our model compared to baselines.

----

## [668] Unified Model for Crystalline Material Generation

**Authors**: *Astrid Klipfel, Yaël Frégier, Adlane Sayede, Zied Bouraoui*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/669](https://doi.org/10.24963/ijcai.2023/669)

**Abstract**:

One of the greatest challenges facing our society is the discovery of new innovative crystal materials with specific properties. Recently, the problem of generating crystal materials has received increasing attention, however, it remains unclear to what extent, or in what way, we can develop generative models that consider both the periodicity and equivalence geometric of crystal structures. To alleviate this issue, we propose two unified models that act at the same time on crystal lattice and atomic positions using periodic equivariant architectures. Our models are capable to learn any arbitrary crystal lattice deformation by lowering the total energy to reach thermodynamic stability. Code and data are available at https://github.com/aklipf/GemsNet.

----

## [669] Machine Learning Driven Aid Classification for Sustainable Development

**Authors**: *Junho Lee, Hyeonho Song, Dongjoon Lee, Sundong Kim, Jisoo Sim, Meeyoung Cha, Kyung-Ryul Park*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/670](https://doi.org/10.24963/ijcai.2023/670)

**Abstract**:

This paper explores how machine learning can help classify aid activities by sector using the OECD Creditor Reporting System (CRS). The CRS is a key source of data for monitoring and evaluating aid flows in line with the United Nations Sustainable Development Goals (SDGs), especially SDG17 which calls for global partnership and data sharing. To address the challenges of current labor-intensive practices of assigning the code and the related human inefficiencies, we propose a machine learning solution that uses ELECTRA to suggest relevant five-digit purpose codes in CRS for aid activities, achieving an accuracy of 0.9575 for the top-3 recommendations. We also conduct qualitative research based on semi-structured interviews and focus group discussions with SDG experts who assess the model results and provide feedback. We discuss the policy, practical, and methodological implications of our work and highlight the potential of AI applications to improve routine tasks in the public sector and foster partnerships for achieving the SDGs.

----

## [670] Confidence-based Self-Corrective Learning: An Application in Height Estimation Using Satellite LiDAR and Imagery

**Authors**: *Zhili Li, Yiqun Xie, Xiaowei Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/671](https://doi.org/10.24963/ijcai.2023/671)

**Abstract**:

Widespread, and rapid, environmental transformation is underway on Earth driven by human activities. Climate shifts such as global warming have led to massive and alarming loss of ice and snow in the high-latitude regions including the Arctic, causing many natural disasters due to sea-level rise, etc. Mitigating the impacts of climate change has also become a United Nations' Sustainable Development Goal for 2030. The recent launch of the ICESat-2 satellites target on heights in the polar regions. However, the observations are only available along very narrow scan lines, leaving large no-data gaps in-between. We aim to fill the gaps by combining the height observations with high-resolution satellite imagery that have large footprints (spatial coverage). The data expansion is a challenging task as the height data are often constrained on one or a few lines per image in real applications, and the images are highly noisy for height estimation. Related work on image-based height prediction and interpolation relies on specific types of images or does not consider the highly-localized height distribution. We propose a spatial self-corrective learning framework, which explicitly uses confidence-based pseudo-interpolation, recurrent self-refinement, and truth-based correction with a regression layer to address the challenges. We carry out experiments on different landscapes in the high-latitude regions and the proposed method shows stable improvements compared to the baseline methods.

----

## [671] DenseLight: Efficient Control for Large-scale Traffic Signals with Dense Feedback

**Authors**: *Junfan Lin, Yuying Zhu, Lingbo Liu, Yang Liu, Guanbin Li, Liang Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/672](https://doi.org/10.24963/ijcai.2023/672)

**Abstract**:

Traffic Signal Control (TSC) aims to reduce the average travel time of vehicles in a road network, which in turn enhances fuel utilization efficiency, air quality, and road safety, benefiting society as a whole. Due to the complexity of long-horizon control and coordination, most prior TSC methods leverage deep reinforcement learning (RL) to search for a control policy and have witnessed great success. However, TSC still faces two significant challenges. 1) The travel time of a vehicle is delayed feedback on the effectiveness of TSC policy at each traffic intersection since it is obtained after the vehicle has left the road network. Although several heuristic reward functions have been proposed as substitutes for travel time, they are usually biased and not leading the policy to improve in the correct direction. 2) The traffic condition of each intersection is influenced by the non-local intersections since vehicles traverse multiple intersections over time. Therefore, the TSC agent is required to leverage both the local observation and the non-local traffic conditions to predict the long-horizontal traffic conditions of each intersection comprehensively. To address these challenges, we propose DenseLight, a novel RL-based TSC method that employs an unbiased reward function to provide dense feedback on policy effectiveness and a non-local enhanced TSC agent to better predict future traffic conditions for more precise traffic control. Extensive experiments and ablation studies demonstrate that DenseLight can consistently outperform advanced baselines on various road networks with diverse traffic flows. The code is available at https://github.com/junfanlin/DenseLight.

----

## [672] SUSTAINABLESIGNALS: An AI Approach for Inferring Consumer Product Sustainability

**Authors**: *Tong Lin, Tianliang Xu, Amit Zac, Sabina Tomkins*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/673](https://doi.org/10.24963/ijcai.2023/673)

**Abstract**:

The everyday consumption of household goods is a significant source of environmental pollution. The increase of online shopping affords an opportunity to provide consumers with actionable feedback on the social and environmental impact of potential purchases, at the exact moment when it is relevant. Unfortunately, consumers are inundated with ambiguous sustainability information. For example, greenwashing can make it difficult to identify environmentally friendly products. The highest-quality options, such as Life Cycle Assessment (LCA) scores or tailored impact certificates (e.g., environmentally friendly tags), designed for assessing the environmental impact of consumption, are ineffective in the setting of online shopping. They are simply too costly to provide a feasible solution when scaled up, and often rely on data from self-interested market players. We contribute an analysis of this online environment, exploring how the dynamic between sellers and consumers surfaces claims and concerns regarding sustainable consumption. In order to better provide information to consumers, we propose a machine learning method that can discover signals of sustainability from these interactions. Our method, SustainableSignals, is a first step in scaling up the provision of sustainability cues to online consumers.

----

## [673] Interpret ESG Rating's Impact on the Industrial Chain Using Graph Neural Networks

**Authors**: *Bin Liu, Jiujun He, Ziyuan Li, Xiaoyang Huang, Xiang Zhang, Guosheng Yin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/674](https://doi.org/10.24963/ijcai.2023/674)

**Abstract**:

We conduct a quantitative analysis of the development of the industry chain from the environmental, social, and governance (ESG) perspective, which is an overall measure of sustainability.  Factors that may impact the performance of the industrial chain have been studied in the literature, such as government regulation, monetary policy, etc. Our interest lies in how the sustainability change (i.e., ESG shock) affects the performance of the industrial chain. To achieve this goal, we model the industrial chain with a graph neural network (GNN) and conduct node regression on two financial performance metrics, namely, the aggregated profitability ratios and operating margin. To quantify the effects of ESG, we propose to compute the interaction between ESG shocks and industrial chain features with a cross-attention module, and then filter the original node features in the graph regression. Experiments on two real datasets demonstrate that (i) there are significant effects of ESG shocks on the industrial chain, and (ii) model parameters including regression coefficients and the attention map can explain how ESG shocks affect the  performance of the industrial chain.

----

## [674] Preventing Attacks in Interbank Credit Rating with Selective-aware Graph Neural Network

**Authors**: *Junyi Liu, Dawei Cheng, Changjun Jiang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/675](https://doi.org/10.24963/ijcai.2023/675)

**Abstract**:

Accurately credit rating on Interbank assets is essential for a healthy financial environment and substantial economic development. But individual participants tend to provide manipulated information in order to attack the rating model to produce a higher score, which may conduct serious adverse effects on the economic system, such as the 2008 global financial crisis. To this end, in this paper, we propose a novel selective-aware graph neural network model (SA-GNN) for defense the Interbank credit rating attacks. In particular, we first simulate the rating information manipulating process by structural and feature poisoning attacks. Then we build a selective-aware defense graph neural model to adaptively prioritize the poisoning training data with Bernoulli distribution similarities. Finally, we optimize the model with weighed penalization on the objection function so that the model could differentiate the attackers. Extensive experiments on our collected real-world Interbank dataset, with over 20 thousand banks and their relations, demonstrate the superior performance of our proposed method in preventing credit rating attacks compared with the state-of-the-art baselines.

----

## [675] Customized Positional Encoding to Combine Static and Time-varying Data in Robust Representation Learning for Crop Yield Prediction

**Authors**: *Qinqing Liu, Fei Dou, Meijian Yang, Ezana Amdework, Guiling Wang, Jinbo Bi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/676](https://doi.org/10.24963/ijcai.2023/676)

**Abstract**:

Accurate prediction of crop yield under the conditions of climate change is crucial to ensure food security. Transformers have shown remarkable success in modeling sequential data and hold the potential for improving crop yield prediction. To understand how weather and meteorological sequence variables affect crop yield, the positional encoding used in Transformers is typically shared across different sample sequences. We argue that it is necessary and beneficial to differentiate the positional encoding for distinct samples based on time-invariant properties of the sequences. Particularly, the sequence variables influencing crop yield vary according to static variables such as geographical locations. Sample data from southern areas may benefit from more tailored positional encoding different from that for northern areas. We propose a novel transformer based architecture for accurate and robust crop yield prediction, by introducing a Customized Positional Encoding (CPE) that encodes a sequence adaptively according to static information associated with the sequence. Empirical studies demonstrate the effectiveness of the proposed novel architecture and show that partially lin-
earized attention better captures the bias introduced by side information than softmax re-weighting. The resultant crop yield prediction model is robust to climate change, with mean-absolute-error reduced by up to 26% compared to the best baseline model in extreme drought years.

----

## [676] GreenFlow: A Computation Allocation Framework for Building Environmentally Sound Recommendation System

**Authors**: *Xingyu Lu, Zhining Liu, Yanchu Guan, Hongxuan Zhang, Chenyi Zhuang, Wenqi Ma, Yize Tan, Jinjie Gu, Guannan Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/677](https://doi.org/10.24963/ijcai.2023/677)

**Abstract**:

Given the enormous number of users and items, industrial cascade recommendation systems (RS) are continuously expanded in size and complexity to deliver relevant items, such as news, services, and commodities, to the appropriate users. In a real-world scenario with hundreds of thousands requests per second, significant computation is required to infer personalized results for each request, resulting in a massive energy consumption and carbon emission that raises concern. 

This paper proposes GreenFlow, a practical computation allocation framework for RS, that considers both accuracy and carbon emission during inference. For each stage (e.g., recall, pre-ranking, ranking, etc.) of a cascade RS, when a user triggers a request, we define two actions that determine the computation: (1) the trained instances of models with different computational complexity; and (2) the number of items to be inferred in the stage. We refer to the combinations of actions in all stages as action chains. A reward score is estimated for each action chain, followed by dynamic primal-dual optimization considering both the reward and computation budget. Extensive experiments verify the effectiveness of the framework, reducing computation consumption by 41% in an industrial mobile application while maintaining commercial revenue. Moreover, the proposed framework saves approximately 5000kWh of electricity and reduces 3 tons of carbon emissions per day.

----

## [677] Coupled Point Process-based Sequence Modeling for Privacy-preserving Network Alignment

**Authors**: *Dixin Luo, Haoran Cheng, Qingbin Li, Hongteng Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/678](https://doi.org/10.24963/ijcai.2023/678)

**Abstract**:

Network alignment aims at finding the correspondence of nodes across different networks, which is significant for many applications, e.g., fraud detection and crime network tracing across platforms. 
In practice, however, accessing the topological information of different networks is often restricted and even forbidden, considering privacy and security issues. 
Instead, what we observed might be the event sequences of the networks' nodes in the continuous-time domain. 
In this study, we develop a coupled neural point process-based (CPP) sequence modeling strategy, which provides a solution to privacy-preserving network alignment based on the event sequences. 
Our CPP consists of a coupled node embedding layer and a neural point process module. 
The coupled node embedding layer embeds one network's nodes and explicitly models the alignment matrix between the two networks.
Accordingly, it parameterizes the node embeddings of the other network by the push-forward operation. 
Given the node embeddings, the neural point process module jointly captures the dynamics of the two networks' event sequences.
We learn the CPP model in a maximum likelihood estimation framework with an inverse optimal transport (IOT) regularizer. 
Experiments show that our CPP is compatible with various point process backbones and is robust to the model misspecification issue, which achieves encouraging performance on network alignment. 
The code is available at https://github.com/Dixin-s-Lab/CNPP.

----

## [678] Group Sparse Optimal Transport for Sparse Process Flexibility Design

**Authors**: *Dixin Luo, Tingting Yu, Hongteng Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/679](https://doi.org/10.24963/ijcai.2023/679)

**Abstract**:

As a fundamental problem in Operations Research, sparse process flexibility design (SPFD) aims to design a manufacturing network across industries that achieves a trade-off between the efficiency and robustness of supply chains. 
In this study, we propose a novel solution to this problem with the help of computational optimal transport techniques.
Given a set of supply-demand pairs, we formulate the SPFD task approximately as a group sparse optimal transport (GSOT) problem, in which a group of couplings between the supplies and demands is optimized with a group sparse regularizer. 
We solve this optimization problem via an algorithmic framework of alternating direction method of multipliers (ADMM), in which the target network topology is updated by soft-thresholding shrinkage, and the couplings of the OT problems are updated via a smooth OT algorithm in parallel. 
This optimization algorithm has guaranteed convergence and provides a generalized framework for the SPFD task, which is applicable regardless of whether the supplies and demands are balanced. 
Experiments show that our GSOT-based method can outperform representative heuristic methods in various SPFD tasks.
Additionally, when implementing the GSOT method, the proposed ADMM-based optimization algorithm is comparable or superior to the commercial software Gurobi. 
The code is available at https://github.com/Dixin-s-Lab/GSOT.

----

## [679] A Prediction-and-Scheduling Framework for Efficient Order Transfer in Logistics

**Authors**: *Wenjun Lyu, Haotian Wang, Yiwei Song, Yunhuai Liu, Tian He, Desheng Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/680](https://doi.org/10.24963/ijcai.2023/680)

**Abstract**:

Order Transfer from the transfer center to delivery stations is an essential and expensive part of the logistics service chain. In practice, one vehicle sends transferred orders to multiple delivery stations in one transfer trip to achieve a better trade-off between the transfer cost and time. A key problem is generating the vehicle’s route for efficient order transfer, i.e., minimizing the order transfer time. In this paper, we explore fine-grained delivery station features, i.e., downstream couriers’ remaining working times in last-mile delivery trips and the transferred order distribution to design a Prediction-and-Scheduling framework for efficient Order Transfer called PSOT, including two components: i) a Courier’s Remaining Working Time Prediction component to predict each courier’s working time for conducting heterogeneous tasks, i.e., order pickups and deliveries, with a context-aware location embedding and an attention-based neural network; ii) a Vehicle Scheduling component to generate the vehicle’s route to served delivery stations with an order-transfer-time-aware heuristic algorithm. The evaluation results with real-world data from one of the largest logistics companies in China show PSOT improves the courier’s remaining working time prediction by up to 35.6% and reduces the average order transfer time by up to 51.3% compared to the state-of-the-art methods.

----

## [680] Fighting against Organized Fraudsters Using Risk Diffusion-based Parallel Graph Neural Network

**Authors**: *Jiacheng Ma, Fan Li, Rui Zhang, Zhikang Xu, Dawei Cheng, Yi Ouyang, Ruihui Zhao, Jianguang Zheng, Yefeng Zheng, Changjun Jiang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/681](https://doi.org/10.24963/ijcai.2023/681)

**Abstract**:

Medical insurance plays a vital role in modern society, yet organized healthcare fraud causes billions of dollars in annual losses, severely harming the sustainability of the social welfare system. Existing works mostly focus on detecting individual fraud entities or claims, ignoring hidden conspiracy patterns. Hence, they face severe challenges in tackling organized fraud. In this paper, we proposed RDPGL, a novel Risk Diffusion-based Parallel Graph Learning approach, to fighting against medical insurance criminal gangs. In particular, we first leverage a heterogeneous graph attention network to encode the local context from the beneficiary-provider graph. Then, we devise a community-aware risk diffusion model to infer the global context of organized fraud behaviors with the claim-claim relation graph. The local and global representations are parallel concatenated together and trained simultaneously in an end-to-end manner. Our approach is extensively evaluated on a real-world medical insurance dataset. The experimental results demonstrate the superiority of our proposed approach, which could detect more organized fraud claims with relatively high precision compared with state-of-the-art baselines.

----

## [681] Planning Multiple Epidemic Interventions with Reinforcement Learning

**Authors**: *Anh L. Mai, Nikunj Gupta, Azza Abouzied, Dennis E. Shasha*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/682](https://doi.org/10.24963/ijcai.2023/682)

**Abstract**:

Combating an epidemic entails finding a plan that describes when and how to apply different interventions, such as mask-wearing mandates, vaccinations, school or workplace closures. An optimal plan will curb an epidemic with minimal loss of life, disease burden, and economic cost. Finding an optimal plan is an intractable computational problem in realistic settings. Policy-makers, however, would greatly benefit from tools that can efficiently search for plans that minimize disease and economic costs especially when considering multiple possible interventions over a continuous and complex action space given a continuous and equally complex state space. We formulate this problem as a Markov decision process. Our formulation is unique in its ability to represent multiple continuous interventions over any disease model defined by ordinary differential equations. We illustrate how to effectively apply state-of-the-art actor-critic reinforcement learning algorithms (PPO and SAC) to search for plans that minimize overall costs. We empirically evaluate the learning performance of these algorithms and compare their performance to hand-crafted baselines that mimic plans constructed by policy-makers. Our method outperforms baselines. Our work confirms the viability of a computational approach to support policy-makers.

----

## [682] Temporally Aligning Long Audio Interviews with Questions: A Case Study in Multimodal Data Integration

**Authors**: *Piyush Singh Pasi, Karthikeya Battepati, Preethi Jyothi, Ganesh Ramakrishnan, Tanmay Mahapatra, Manoj Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/683](https://doi.org/10.24963/ijcai.2023/683)

**Abstract**:

The problem of audio-to-text alignment has seen significant amount of research using complete supervision during training. However, this is typically not in the context of long audio recordings wherein the text being queried does not appear verbatim within the audio file. This work is a collaboration with a non-governmental organization called CARE India that collects long audio health surveys from young mothers residing in rural parts of Bihar, India. Given a question drawn from a questionnaire that is used to guide these surveys, we aim to locate where the question is asked within a long audio recording. This is of great value to African and Asian organizations that would otherwise have to painstakingly go through long and noisy audio recordings to locate questions (and answers) of interest. Our proposed framework, INDENT, uses a cross-attention-based model and prior information on the temporal ordering of sentences to learn speech embeddings that capture the semantics of the underlying spoken text. These learnt embeddings are used to retrieve the corresponding audio segment based on text queries at inference time. We empirically demonstrate the significant effectiveness (improvement in R-avg of about 3%) of our model over those obtained using text-based heuristics.  We also show how noisy ASR, generated using state-of-the-art ASR models for Indian languages, yields better results when used in place of speech. INDENT, trained only on Hindi data is able to cater to all languages supported by the (semantically) shared text space. We illustrate this empirically on 11 Indic languages.

----

## [683] Time Series of Satellite Imagery Improve Deep Learning Estimates of Neighborhood-Level Poverty in Africa

**Authors**: *Markus B. Pettersson, Mohammad Kakooei, Julia Ortheden, Fredrik D. Johansson, Adel Daoud*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/684](https://doi.org/10.24963/ijcai.2023/684)

**Abstract**:

To combat poor health and living conditions, policymakers in Africa require temporally and geographically granular data measuring economic well-being. 
Machine learning (ML) offers a promising alternative to expensive and time-consuming survey measurements by training models to predict economic conditions from freely available satellite imagery. However,  previous efforts have failed to utilize the temporal information available in earth observation (EO) data, which may capture developments important to standards of living. In this work, we develop an EO-ML method for inferring neighborhood-level material-asset wealth using multi-temporal imagery and recurrent convolutional neural networks. Our model outperforms state-of-the-art models in several aspects of generalization, explaining  72% of the variance in wealth across held-out countries and 75%  held-out time spans. Using our geographically and temporally aware models, we created spatio-temporal material-asset data maps covering the entire continent of Africa from 1990 to 2019, making our data product the largest dataset of its kind. We showcase these results by analyzing which neighborhoods are likely to escape poverty by the year 2030, which is the deadline for when the Sustainable Development Goals (SDG) are evaluated.

----

## [684] Balancing Social Impact, Opportunities, and Ethical Constraints of Using AI in the Documentation and Vitalization of Indigenous Languages

**Authors**: *Claudio S. Pinhanez, Paulo R. Cavalin, Marisa Vasconcelos, Julio Nogima*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/685](https://doi.org/10.24963/ijcai.2023/685)

**Abstract**:

In this paper we discuss how AI can contribute to support the documentation and vitalization of Indigenous languages and how that involves a delicate balancing of ensuring social impact, exploring technical opportunities, and dealing with ethical constraints. We start by surveying previous work on using AI and NLP to support critical activities of strengthening Indigenous and endangered languages and discussing key limitations of current technologies. After presenting basic ethical constraints of working with Indigenous languages and communities, we propose that creating and deploying language technology ethically with and for Indigenous communities forces AI researchers and engineers to address some of the main shortcomings and criticisms of current technologies. Those ideas are also explored in the discussion of a real case of development of large language models for Brazilian Indigenous languages.

----

## [685] PARTNER: A Persuasive Mental Health and Legal Counselling Dialogue System for Women and Children Crime Victims

**Authors**: *Priyanshu Priya, Kshitij Mishra, Palak Totala, Asif Ekbal*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/686](https://doi.org/10.24963/ijcai.2023/686)

**Abstract**:

The World Health Organization has underlined the significance of expediting the preventive measures for crime against women and children to attain the United Nations Sustainable Development Goals 2030 (promoting well-being, gender equality, and equal access to justice). The crime victims typically need mental health and legal counselling support for their ultimate well-being and sometimes they need to be persuaded to seek desired support. Further, counselling interactions should adopt correct politeness and empathy strategies so that a warm, amicable, and respectful environment can be built to better understand the victimsâ€™ situations. To this end, we propose PARTNER, a Politeness and empAthy strategies-adaptive peRsuasive dialogue sysTem for meNtal health and LEgal counselling of cRime victims. For this, first, we create a novel mental HEalth and legAl counseLling conversational dataset HEAL, annotated with three distinct aspects, viz. counselling act, politeness strategy, and empathy strategy. Then, by formulating a novel reward function, we train a counselling dialogue system in a reinforcement learning setting to ensure correct counselling act, politeness strategy, and empathy strategy in the generated responses. Extensive empirical analysis and experimental results show that the proposed reward function ensures persuasive counselling responses with correct polite and empathetic tone in the generated responses. Further, PARTNER proves its efficacy to engage the victim by generating diverse and natural responses.

----

## [686] AudioQR: Deep Neural Audio Watermarks For QR Code

**Authors**: *Xinghua Qu, Xiang Yin, Pengfei Wei, Lu Lu, Zejun Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/687](https://doi.org/10.24963/ijcai.2023/687)

**Abstract**:

Image-based quick response (QR) code is frequently used, but creates barriers for the visual impaired people. With the goal of ``AI for good", this paper proposes the AudioQR, a barrier-free QR coding mechanism for the visually impaired population via deep neural audio watermarks. Previous audio watermarking approaches are mainly based on handcrafted pipelines, which is less secure and difficult to apply in large-scale scenarios. In contrast, AudioQR is the first comprehensive end-to-end pipeline that hides watermarks in audio imperceptibly and robustly. To achieve this, we jointly train an encoder and decoder, where the encoder is structured as a concatenation of transposed convolutions and multi-receptive field fusion modules. Moreover, we customize the decoder training with a stochastic data augmentation chain to make the watermarked audio robust towards different audio distortions, such as environment background, room impulse response when playing through the air, music surrounding, and Gaussian noise. Experiment results indicate that AudioQR can efficiently hide arbitrary information into audio without introducing significant perceptible difference. Our code is available at https://github.com/xinghua-qu/AudioQR.

----

## [687] Function Approximation for Reinforcement Learning Controller for Energy from Spread Waves

**Authors**: *Soumyendu Sarkar, Vineet Gundecha, Sahand Ghorbanpour, Alexander Shmakov, Ashwin Ramesh Babu, Avisek Naug, Alexandre Pichard, Mathieu Cocho*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/688](https://doi.org/10.24963/ijcai.2023/688)

**Abstract**:

The industrial multi-generator Wave Energy Converters (WEC) must handle multiple simultaneous waves coming from different directions called spread waves. These complex devices in challenging circumstances need controllers with multiple objectives of energy capture efficiency, reduction of structural stress to limit maintenance, and proactive protection against high waves. The Multi-Agent Reinforcement Learning (MARL) controller trained with Proximal Policy Optimization (PPO) algorithm can handle these complexities. In this paper, we explore different function approximations for the policy and critic networks in modeling the sequential nature of the system dynamics and find that they are key to better performance. We investigated the performance of a fully connected neural network (FCN), LSTM, and Transformer model variants with varying depths and gated residual connections. Our results show that the transformer model of moderate depth with gated residual connections around the multi-head attention, multi-layer perceptron, and the transformer block (STrXL) proposed in this paper is optimal and boosts energy efficiency by an average of 22.1% for these complex spread waves over the existing spring damper (SD) controller. Furthermore, unlike the default SD controller, the transformer controller almost eliminated the mechanical stress from the rotational yaw motion for angled waves. Demo: https://tinyurl.com/yueda3jh

----

## [688] Promoting Gender Equality through Gender-biased Language Analysis in Social Media

**Authors**: *Gopendra Vikram Singh, Soumitra Ghosh, Asif Ekbal*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/689](https://doi.org/10.24963/ijcai.2023/689)

**Abstract**:

Gender bias is a pervasive issue that impacts women's and marginalized groups' ability to fully participate in social, economic, and political spheres. This study introduces a novel problem of Gender-biased Language Identification and Extraction (GLIdE) from social media interactions and develops a multi-task deep framework that detects gender-biased content and identifies connected causal phrases from the text using emotional information that is present in the input. The method uses a zero-shot strategy with emotional information and a mechanism to represent gender-stereotyped information as a knowledge graph. In this work, we also introduce the first-of-its-kind Gender-biased Analysis Corpus (GAC) of 12,432 social media posts and improve the best-performing baseline for gender-biased language identification and extraction tasks by margins of 4.88% and 5 ROS points, demonstrating this through empirical evaluation and extensive qualitative analysis. By improving the accuracy of identifying and analyzing gender-biased language, this work can contribute to achieving gender equality and promoting inclusive societies, in line with the United Nations Sustainable Development Goals (UN SDGs) and the Leave No One Behind principle (LNOB). We adhere to the principles of transparency and collaboration in line with the UN SDGs by openly sharing our code and dataset.

----

## [689] Leveraging Domain Knowledge for Inclusive and Bias-aware Humanitarian Response Entry Classification

**Authors**: *Nicolò Tamagnone, Selim Fekih, Ximena Contla, Nayid Orozco, Navid Rekabsaz*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/690](https://doi.org/10.24963/ijcai.2023/690)

**Abstract**:

Accurate and rapid situation analysis during humanitarian crises is critical to delivering humanitarian aid efficiently and is fundamental to humanitarian imperatives and the Leave No One Behind (LNOB) principle. This data analysis can highly benefit from language processing systems, e.g., by classifying the text data according to a humanitarian ontology. However, approaching this by simply fine-tuning a generic large language model (LLM) involves considerable practical and ethical issues, particularly the lack of effectiveness on data-sparse and complex subdomains, and the encoding of societal biases and unwanted associations. In this work, we aim to provide an effective and ethically-aware system for humanitarian data analysis. We approach this by (1) introducing a novel architecture adjusted to the humanitarian analysis framework, (2) creating and releasing a novel humanitarian-specific LLM called HumBert, and (3) proposing a systematic way to measure and mitigate biases. Our experiments' results show the better performance of our approach on zero-shot and full-training settings in comparison with strong baseline models, while also revealing the existence of biases in the resulting LLMs. Utilizing a targeted counterfactual data augmentation approach, we significantly reduce these biases without compromising performance.

----

## [690] Optimizing Crop Management with Reinforcement Learning and Imitation Learning

**Authors**: *Ran Tao, Pan Zhao, Jing Wu, Nicolas F. Martin, Matthew T. Harrison, Carla Sofia Santos Ferreira, Zahra Kalantari, Naira Hovakimyan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/691](https://doi.org/10.24963/ijcai.2023/691)

**Abstract**:

Crop management has a significant impact on crop yield, economic profit, and the environment. Although management guidelines exist, finding the optimal management practices is challenging. Previous work used reinforcement learning (RL) and crop simulators to solve the problem, but the trained policies either have limited performance or are not deployable in the real world. In this paper, we present an intelligent crop management system that optimizes nitrogen fertilization and irrigation simultaneously via RL, imitation learning (IL), and crop simulations using the Decision Support System for Agrotechnology Transfer (DSSAT). We first use deep RL, in particular, deep Q-network, to train management policies that require a large number of state variables from the simulator as observations (denoted as full observation). We then invoke IL to train management policies that only need a few state variables that can be easily obtained or measured in the real world (denoted as partial observation) by mimicking the actions of the RL policies trained under full observation. Simulation experiments using the maize crop in Florida (US) and Zaragoza (Spain) demonstrate that the trained policies from both RL and IL techniques achieved more than 45\% improvement in economic profit while causing less environmental impact compared with a baseline method. Most importantly, the IL-trained management policies are directly deployable in the real world as they use readily available information.

----

## [691] Keeping People Active and Healthy at Home Using a Reinforcement Learning-based Fitness Recommendation Framework

**Authors**: *Elias Z. Tragos, Diarmuid O'Reilly-Morgan, James Geraci, Bichen Shi, Barry Smyth, Cailbhe Doherty, Aonghus Lawlor, Neil Hurley*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/692](https://doi.org/10.24963/ijcai.2023/692)

**Abstract**:

Recent years have seen a rise in smartphone applications promoting health and well being. We argue that there is a large and unexplored ground within the field of recommender systems (RS) for applications that promote good personal health. During the COVID-19 pandemic, with gyms being closed, the demand for at-home fitness apps increased as users wished to maintain their physical and mental health. However, maintaining long-term user engagement with fitness applications has proved a difficult task. Personalisation of the app recommendations that change over time can be a key factor for maintaining high user engagement. In this work we propose a reinforcement learning (RL) based framework for recommending sequences of body-weight exercises to home users over a mobile application interface. The framework employs a user simulator, tuned to feedback a weighted sum of realistic workout rewards, and trains a neural network model to maximise the expected reward over generated exercise sequences. We evaluate our framework within the context of a large 15 week live user trial, showing that an RL based approach leads to a significant increase in user engagement compared to a baseline recommendation algorithm.

----

## [692] Intensity-Valued Emotions Help Stance Detection of Climate Change Twitter Data

**Authors**: *Apoorva Upadhyaya, Marco Fisichella, Wolfgang Nejdl*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/693](https://doi.org/10.24963/ijcai.2023/693)

**Abstract**:

Our study focuses on the United Nations Sustainable Development Goal 13: Climate Action, by identifying public attitudes on Twitter about climate change. Public consent and participation is the key factor in dealing with climate crises. However, discussions about climate change on Twitter are often influenced by the polarised beliefs that shape the discourse and divide it into communities of climate change deniers and believers. In our work, we propose a framework that helps identify different attitudes in tweets about climate change (deny, believe, ambiguous). Previous literature often lacks an efficient architecture or ignores the characteristics of climate-denier tweets. Moreover, the presence of various emotions with different levels of intensity turns out to be relevant for shaping discussions on climate change. Therefore, our paper utilizes emotion recognition and emotion intensity prediction as auxiliary tasks for our main task of stance detection. Our framework injects the words affecting the emotions embedded in the tweet to capture the overall representation of the attitude in terms of the emotions associated with it. The final task-specific and shared feature representations are fused with efficient embedding and attention techniques to detect the correct attitude of the tweet. Extensive experiments on our novel curated dataset, two publicly available climate change datasets (ClimateICWSM-2023 and ClimateStance-2022), and a benchmark dataset for stance detection (SemEval-2016) validate the effectiveness of our approach.

----

## [693] Evaluating GPT-3 Generated Explanations for Hateful Content Moderation

**Authors**: *Han Wang, Ming Shan Hee, Md. Rabiul Awal, Kenny Tsu Wei Choo, Roy Ka-Wei Lee*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/694](https://doi.org/10.24963/ijcai.2023/694)

**Abstract**:

Recent research has focused on using large language models (LLMs) to generate explanations for hate speech through fine-tuning or prompting. Despite the growing interest in this area, these generated explanations' effectiveness and potential limitations remain poorly understood. A key concern is that these explanations, generated by LLMs, may lead to erroneous judgments about the nature of flagged content by both users and content moderators. For instance, an LLM-generated explanation might inaccurately convince a content moderator that a benign piece of content is  hateful. In light of this, we propose an analytical framework for examining hate speech explanations and conducted an extensive survey on evaluating such explanations. Specifically, we prompted GPT-3 to generate explanations for both hateful and non-hateful content, and a survey was conducted with 2,400 unique respondents to evaluate the generated explanations. Our findings reveal that (1) human evaluators rated the GPT-generated explanations as high quality in terms of linguistic fluency, informativeness, persuasiveness, and logical soundness, (2) the persuasive nature of these explanations, however, varied depending on the prompting strategy employed, and (3) this persuasiveness may result in incorrect judgments about the hatefulness of the content. Our study underscores the need for caution in applying LLM-generated explanations for content moderation. Code and results are available at https://github.com/Social-AI-Studio/GPT3-HateEval.

----

## [694] Full Scaling Automation for Sustainable Development of Green Data Centers

**Authors**: *Shiyu Wang, Yinbo Sun, Xiaoming Shi, Shiyi Zhu, Lintao Ma, James Zhang, Yangfei Zheng, Liu Jian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/695](https://doi.org/10.24963/ijcai.2023/695)

**Abstract**:

The rapid rise in cloud computing has resulted in an alarming increase in data centers' carbon emissions, which now accounts for >3% of global greenhouse gas emissions, necessitating immediate steps to combat their mounting strain on the global climate. An important focus of this effort is to improve resource utilization in order to save electricity usage.  Our proposed  Full Scaling Automation (FSA) mechanism is an effective method of dynamically adapting resources to accommodate changing workloads in large-scale cloud computing clusters, enabling the clusters in data centers to maintain their desired CPU utilization target and thus improve energy efficiency. FSA harnesses the power of deep representation learning to accurately predict the future workload of each service and automatically stabilize the corresponding target CPU usage level, unlike the previous autoscaling methods, such as Autopilot or FIRM, that need to adjust computing resources with statistical models and expert knowledge.  Our approach achieves significant performance improvement compared to the existing work in real-world datasets.  We also deployed FSA on large-scale cloud computing clusters in industrial data centers, and according to the certification of the China Environmental United Certification Center (CEC), a reduction of 947 tons of carbon dioxide, equivalent to a saving of 1538,000 kWh of electricity, was achieved during the Double 11 shopping festival of 2022, marking a critical step for our companyâ€™s strategic goal towards carbon neutrality by 2030.

----

## [695] A Quantitative Game-theoretical Study on Externalities of Long-lasting Humanitarian Relief Operations in Conflict Areas

**Authors**: *Kaiming Xiao, Haiwen Chen, Hongbin Huang, Lihua Liu, Jibing Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/696](https://doi.org/10.24963/ijcai.2023/696)

**Abstract**:

Humanitarian relief operations are often accompanied by regional conflicts around the globe, at risk of deliberate, persistent and unpredictable attacks. However, the long-term channeling of aid resources into conflict areas may influence subsequent patterns of violence and expose local communities to new risks. In this paper, we quantitatively analyze the potential externalities associated with long-lasting humanitarian relief operations based on game-theoretical modeling and online planning approaches. Specifically, we first model the problem of long-lasting humanitarian relief operations in conflict areas as an online multi-stage rescuer-and-attacker interdiction game in which aid demands are revealed in an online fashion. Both models of single-source and multiple-source relief supply policy are established respectively, and two corresponding near-optimal online algorithms are proposed. In conjunction with a real case of anti-Ebola practice in conflict areas of DR Congo, we find that 1) long-lasting humanitarian relief operations aiming alleviation of crises in conflict areas can lead to indirect funding of local rebel groups; 2) the operations can activate the rebel groups to some extent, as evidenced by the scope expansion of their activities. Furthermore, the impacts of humanitarian aid intensity, frequency and supply policies on the above externalities are quantitatively analyzed, which will provide enlightening decision-making support for the implementation of related operations in the future.

----

## [696] Quality-agnostic Image Captioning to Safely Assist People with Vision Impairment

**Authors**: *Lu Yu, Malvina Nikandrou, Jiali Jin, Verena Rieser*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/697](https://doi.org/10.24963/ijcai.2023/697)

**Abstract**:

Automated image captioning has the potential to be a useful tool for people with vision impairments. Images taken by this user group are often noisy,  which leads to incorrect and even unsafe model predictions. In this paper, we propose a quality-agnostic framework to improve the performance and robustness of image captioning models for visually impaired people. We address this problem from three angles: data, model, and evaluation. First, we show how data augmentation techniques for generating synthetic noise can address data sparsity in this domain. Second, we enhance the robustness of the model by expanding a state-of-the-art model to a dual network architecture, using the augmented data and leveraging different consistency losses. Our results demonstrate increased performance, e.g. an absolute improvement of 2.15 on CIDEr, compared to state-of-the-art image captioning networks, as well as increased robustness to noise with up to 3 points improvement on CIDEr in more noisy settings. Finally, we evaluate the prediction reliability using confidence calibration on images with different difficulty / noise levels, showing that our models perform more reliably
in safety-critical situations. The improved model is part of an assisted living application, which we develop in partnership with the Royal National Institute of Blind People.

----

## [697] GreenPLM: Cross-Lingual Transfer of Monolingual Pre-Trained Language Models at Almost No Cost

**Authors**: *Qingcheng Zeng, Lucas Garay, Peilin Zhou, Dading Chong, Yining Hua, Jiageng Wu, Yikang Pan, Han Zhou, Rob Voigt, Jie Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/698](https://doi.org/10.24963/ijcai.2023/698)

**Abstract**:

Large pre-trained models have revolutionized natural language processing (NLP) research and applications, but high training costs and limited data resources have prevented their benefits from being shared equally amongst speakers of all the world's languages. To address issues of cross-linguistic access to such models and reduce energy consumption for sustainability during large-scale model training, this study proposes an effective and energy-efficient framework called GreenPLM that uses bilingual lexicons to directly ``translate'' pre-trained language models of one language into another at almost no additional cost. We validate this approach in 18 languages' BERT models and show that this framework is comparable to, if not better than, other heuristics with high training costs. In addition, given lightweight continued pre-training on limited data where available, this framework outperforms the original monolingual language models in six out of seven tested languages with up to 200x less pre-training efforts. Aiming at the Leave No One Behind Principle (LNOB), our approach manages to reduce inequalities between languages and energy consumption greatly. We make our codes and models publicly available at https://github.com/qcznlp/GreenPLMs.

----

## [698] Mimicking the Thinking Process for Emotion Recognition in Conversation with Prompts and Paraphrasing

**Authors**: *Ting Zhang, Zhuang Chen, Ming Zhong, Tieyun Qian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/699](https://doi.org/10.24963/ijcai.2023/699)

**Abstract**:

Emotion recognition in conversation,  which aims to predict the emotion for all utterances,  has attracted considerable research attention in recent years. It is a challenging task since the  recognition of the emotion in one  utterance  involves many complex factors, such as the conversational context, the speaker's  background, and the subtle difference between emotion labels. In this paper, we propose a novel framework which mimics the thinking process when modeling these factors. Specifically, we first comprehend the conversational context with a history-oriented prompt to selectively gather  information from predecessors of the target utterance. We then  model the speaker's background with an experience-oriented prompt  to retrieve the similar utterances from all conversations.  We finally  differentiate the subtle label semantics with a paraphrasing mechanism  to elicit the intrinsic label related knowledge.
We conducted extensive experiments on three benchmarks. The empirical results demonstrate the superiority of our proposed framework over the state-of-the-art baselines.

----

## [699] Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer

**Authors**: *Yang Zhang, Lingbo Liu, Xinyu Xiong, Guanbin Li, Guoli Wang, Liang Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/700](https://doi.org/10.24963/ijcai.2023/700)

**Abstract**:

Wind power is attracting increasing attention around the world due to its renewable, pollution-free, and other advantages. However, safely and stably integrating the high permeability intermittent power energy into electric power systems remains challenging. Accurate wind power forecasting (WPF) can effectively reduce power fluctuations in power system operations. Existing methods are mainly designed for short-term predictions and lack effective spatial-temporal feature augmentation. In this work, we propose a novel end-to-end wind power forecasting model named Hierarchical Spatial-Temporal Transformer Network (HSTTN) to address the long-term WPF problems. Specifically, we construct an hourglass-shaped encoder-decoder framework with skip-connections to jointly model representations aggregated in hierarchical temporal scales, which benefits long-term forecasting. Based on this framework, we capture the inter-scale long-range temporal dependencies and global spatial correlations with two parallel Transformer skeletons and strengthen the intra-scale connections with downsampling and upsampling operations. Moreover, the complementary information from spatial and temporal features is fused and propagated in each other via Contextual Fusion Blocks (CFBs) to promote the prediction further. Extensive experimental results on two large-scale real-world datasets demonstrate the superior performance of our HSTTN over existing solutions.

----

## [700] On Optimizing Model Generality in AI-based Disaster Damage Assessment: A Subjective Logic-driven Crowd-AI Hybrid Learning Approach

**Authors**: *Yang Zhang, Ruohan Zong, Lanyu Shang, Huimin Zeng, Zhenrui Yue, Na Wei, Dong Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/701](https://doi.org/10.24963/ijcai.2023/701)

**Abstract**:

This paper focuses on the AI-based damage assessment (ADA) applications that leverage state-of-the-art AI techniques to automatically assess the disaster damage severity using online social media imagery data, which aligns well with the ''disaster risk reduction'' target under United Nations' Sustainable Development Goals (UN SDGs). This paper studies an ADA model generality problem where the objective is to address the limitation of current ADA solutions that are often optimized only for a single disaster event and lack the generality to provide accurate performance across different disaster events. To address this limitation, we work with domain experts and local community stakeholders in disaster response to develop CollabGeneral, a subjective logic-driven crowd-AI collaborative learning framework that integrates AI and crowdsourced human intelligence into a principled learning framework to address the ADA model generality problem. Extensive experiments on four real-world ADA datasets demonstrate that CollabGeneral consistently outperforms the state-of-the-art baselines by significantly improving the ADA model generality across different disasters.

----

## [701] User-Centric Democratization towards Social Value Aligned Medical AI Services

**Authors**: *Zhaonian Zhang, Richard Jiang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/702](https://doi.org/10.24963/ijcai.2023/702)

**Abstract**:

Democratic AI, aiming at developing AI systems aligned with human values, holds promise for making AI services accessible to people. However, concerns have been raised regarding the participation of non-technical individuals, potentially undermining the carefully designed values of AI systems by experts. In this paper, we investigate Democratic AI, define it mathematically, and propose a user-centric evolutionary democratic AI (u-DemAI) framework. This framework maximizes the social values of cloud-based AI services by incorporating user feedback and emulating human behavior in a community via a user-in-the-loop iteration. We apply our framework to a medical AI service for brain age estimation and demonstrate that non-expert users can consistently contribute to improving AI systems through a natural democratic process. The u-DemAI framework presents a mathematical interpretation of Democracy for AI, conceptualizing it as a natural computing process. Our experiments successfully show that involving non-tech individuals can help improve performance and simultaneously mitigate bias in AI models developed by AI experts, showcasing the potential for Democratic AI to benefit end users and regain control over AI services that shape various aspects of our lives, including our health.

----

## [702] Optimization-driven Demand Prediction Framework for Suburban Dynamic Demand-Responsive Transport Systems

**Authors**: *Louis Zigrand, Roberto Wolfler Calvo, Emiliano Traversi, Pegah Alizadeh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/703](https://doi.org/10.24963/ijcai.2023/703)

**Abstract**:

Demand-Responsive Transport (DRT) has grown over the last decade as an ecological solution to both metropolitan and suburban areas. It provides a more efficient public transport service in metropolitan areas and satisfies the mobility needs in sparse and heterogeneous suburban areas. Traditionally, DRT operators build the plannings of their drivers by relying on myopic insertion heuristics that do not take into account the dynamic nature of such a service. We thus investigate in this work the potential of a Demand Prediction Framework used specifically to build more flexible routes within a Dynamic Dial-a-Ride Problem (DaRP) solver. We show how to obtain a Machine Learning forecasting model that is explicitly designed for optimization purposes. The prediction task is further complicated by the fact that the historical dataset is significantly sparse. We finally show how the predicted travel requests can be integrated within an optimization scheme in order to compute better plannings at the start of the day. Numerical results support the fact that, despite the data sparsity challenge as well as the optimization-driven constraints that result from the DaRP model, such a look-ahead approach can improve up to 3.5% the average insertion rate of an actual DRT service.

----

## [703] Long-term Monitoring of Bird Flocks in the Wild

**Authors**: *Kshitiz, Sonu Shreshtha, Ramy Mounir, Mayank Vatsa, Richa Singh, Saket Anand, Sudeep Sarkar, Sevaram Mali Parihar*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/704](https://doi.org/10.24963/ijcai.2023/704)

**Abstract**:

Monitoring and analysis of wildlife are key to conservation planning and conflict management. The widespread use of camera traps coupled with AI-based analysis tools serves as an excellent example of successful and non-invasive use of technology for design, planning, and evaluation of conservation policies. As opposed to the typical use of camera traps that capture still images or short videos, in this project, we propose to analyze longer term videos monitoring a large flock of birds. This project, which is part of the NSF-TIH Indo-US joint R&D partnership, focuses on solving challenges associated with the analysis of long-term videos captured at feeding grounds and nesting sites, among other such locations that host large flocks of migratory birds. We foresee that the objectives of this project would lead to datasets and benchmarking tools as well as novel algorithms that would be instrumental in developing automated video analysis tools that could in turn help understand individual and social behavior of birds. The first of the key outcomes of this research will include the curation of challenging, real-world datasets for benchmarking various image and video analytics algorithms for tasks such as counting, detection, segmentation, and tracking. Our recent efforts towards this outcome is a curated dataset of 812 high-resolution, point-annotated, images (4K - 32MP) of a flock of Demoiselle cranes (Anthropoides virgo) taken from their feeding site at Khichan, Rajasthan, India. The average number of birds in each image is about 207, with a maximum count of 1500. The benchmark experiments show that state-of-the-art vision techniques struggle with tasks such as segmentation, detection, localization, and density estimation for the proposed dataset. Over the execution of this open science research, we will be scaling this dataset for segmentation and tracking in videos, as well as developing novel techniques for video analytics for wildlife monitoring.

----

## [704] On AI-Assisted Pneumoconiosis Detection from Chest X-rays

**Authors**: *Yasmeena Akhter, Rishabh Ranjan, Richa Singh, Mayank Vatsa, Santanu Chaudhury*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/705](https://doi.org/10.24963/ijcai.2023/705)

**Abstract**:

According to theWorld Health Organization, Pneumoconiosis
affects millions of workers globally,
with an estimated 260,000 deaths annually. The
burden of Pneumoconiosis is particularly high in
low-income countries, where occupational safety
standards are often inadequate, and the prevalence
of the disease is increasing rapidly. The reduced
availability of expert medical care in rural areas,
where these diseases are more prevalent, further
adds to the delayed screening and unfavourable outcomes
of the disease. This paper aims to highlight
the urgent need for early screening and detection
of Pneumoconiosis, given its significant impact on
affected individuals, their families, and societies as
a whole. With the help of low-cost machine learning
models, early screening, detection, and prevention
of Pneumoconiosis can help reduce healthcare
costs, particularly in low-income countries. In this
direction, this research focuses on designing AI solutions
for detecting different kinds of Pneumoconiosis
from chest X-ray data. This will contribute
to the Sustainable Development Goal 3 of ensuring
healthy lives and promoting well-being for all at all
ages, and present the framework for data collection
and algorithm for detecting Pneumoconiosis
for early screening. The baseline results show that
the existing algorithms are unable to address this
challenge. Therefore, it is our assertion that this
research will improve state-of-the-art algorithms of
segmentation, semantic segmentation, and classification
not only for this disease but in general medical
image analysis literature.

----

## [705] AI-Assisted Tool for Early Diagnosis and Prevention of Colorectal Cancer in Africa

**Authors**: *Bushra Ibnauf, Mohammed Aboul Ezz, Ayman Abdel Aziz, Khalid Elgazzar, Mennatullah Siam*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/706](https://doi.org/10.24963/ijcai.2023/706)

**Abstract**:

Colorectal cancer (CRC) is considered the third most common cancer worldwide and is recently increasing in Africa. It is mostly diagnosed at an advanced state causing high fatality rates, which highlights the importance of CRC early diagnosis. There are various methods used to enable early diagnosis of CRC, which are vital to increase survival rates such as colonoscopy. Recently, there are calls to start an early detection program in Egypt using colonoscopy. It can be used for diagnosis and prevention purposes to detect and remove polyps, which are benign growths that have the risk of turning into cancer. However, there tends to be a high miss rate of polyps from physicians, which motivates machine learning guided polyp segmentation methods in colonoscopy videos to aid physicians. To date, there are no large-scale video polyp segmentation dataset that is focused on African countries. It was shown in AI-assisted systems that under-served populations such as patients with African origin can be misdiagnosed. There is also a potential need in other African countries beyond Egypt to provide a cost efficient tool to record colonoscopy videos using smart phones without relying on video recording equipment. Since most of the equipment used in Africa are old and refurbished, and video recording equipment can get defective. Hence, why we propose to curate a colonoscopy video dataset focused on African patients, provide expert annotations for video polyp segmentation and provide an AI-assisted tool to record colonoscopy videos using smart phones. Our project is based on our core belief in developing research by Africans and increasing the computer vision research capacity in Africa.

----

## [706] AI and Decision Support for Sustainable Socio-Ecosystems

**Authors**: *Dimitri Justeau-Allaire*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/707](https://doi.org/10.24963/ijcai.2023/707)

**Abstract**:

The conservation and the restoration of biodiversity, in accordance with human well-being, is a necessary condition for the realization of several Sustainable Development Goals. However, there is still an important gap between biodiversity research and the management of natural areas. This research project aims to reduce this gap by proposing spatial planning methods that robustly and accurately integrate socio-ecological issues. Artificial intelligence, and notably Constraint Programming, will play a central role and will make it possible to remove the methodological obstacles that prevent us from properly addressing the complexity and heterogeneity of sustainability issues in the management of ecosystems. The whole will be articulated in three axes: (i) integrate socio-ecological dynamics into spatial planning, (ii) rely on adequate landscape metrics in spatial planning, (iii) scaling up spatial planning methods performances. The main study context of this project is the sustainable management of tropical forests, with a particular focus on New Caledonia and West Africa.

----

## [707] NutriAI: AI-Powered Child Malnutrition Assessment in Low-Resource Environments

**Authors**: *Misaal Khan, Shivang Agarwal, Mayank Vatsa, Richa Singh, Kuldeep Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/708](https://doi.org/10.24963/ijcai.2023/708)

**Abstract**:

Malnutrition among infants and young children is a pervasive public health concern, particularly in developing countries where resources are limited. Millions of children globally suffer from malnourishment and its complications1. Despite the best efforts of governments and organizations, malnourishment persists and remains a leading cause of morbidity and mortality among children under five. Physical measurements, such as weight, height, middle-upper-arm-circumference (muac), and head circumference are commonly used to assess the nutritional status of children. However, this approach can be resource-intensive and challenging to carry out on a large scale. In this research, we are developing NutriAI, a low-cost solution that leverages
small sample size classification approach to detect malnutrition by analyzing 2D images of the subjects in multiple poses. The proposed solution will not only reduce the workload of health workers but also provide a more efficient means of monitoring the nutritional status of children. On the dataset prepared as part of this research, the baseline results highlight that the modern deep learning approaches can facilitate malnutrition detection via anthropometric indicators in the presence of diversity with respect to age, gender, physical characteristics, and accessories including clothing.

----

## [708] Learning and Reasoning Multifaceted and Longitudinal Data for Poverty Estimates and Livelihood Capabilities of Lagged Regions in Rural India

**Authors**: *Atharva Kulkarni, Raya Das, Ravi S. Srivastava, Tanmoy Chakraborty*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/709](https://doi.org/10.24963/ijcai.2023/709)

**Abstract**:

Poverty is a multifaceted phenomenon linked to the lack of capabilities of households to earn a sustainable livelihood, increasingly being assessed using multidimensional indicators. Its spatial pattern depends on social, economic, political, and regional variables. Artificial intelligence has shown immense scope in analyzing the complexities and nuances of poverty. The proposed project aims to examine the poverty situation of rural India for the period of 1990-2022 based on the quality of life and livelihood indicators. The districts will be classified into ‘advanced’, ‘catching up’, ‘falling behind’, and ‘lagged’ regions. The project proposes to integrate multiple data sources, including conventional national-level large sample household surveys, census surveys, and proxy variables like daytime, and nighttime data from satellite images, and communication networks, to name a few, to provide a comprehensive view of poverty at the district level. The project also intends to examine causation and longitudinal analysis to examine the reasons for poverty. Poverty and inequality could be widening in developing countries due to demographic and growth-agglomerating policies. Therefore, targeting the lagging regions and the vulnerable population is essential to eradicate poverty and improve the quality of life to achieve the goal of ‘zero poverty’. Thus, the study also focuses on the districts with a higher share of the marginal section of the population compared to the national average to trace the performance of development indicators and their association with poverty in these regions.

----

## [709] AI-Driven Sign Language Interpretation for Nigerian Children at Home

**Authors**: *Ifeoma Nwogu, Roshan L. Peiris, Karthik Dantu, Ruchi Gamta, Emma Asonye*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/710](https://doi.org/10.24963/ijcai.2023/710)

**Abstract**:

As many as three million school age children between the ages of 5 and 14 years, live with severe to profound hearing loss in Nigeria. Many of these Deaf or Hard of Hearing (DHH) children developed their hearing loss later in life, non-congenitally, hence their parents are hearing. While their teachers in the Deaf schools they attend can often communicate effectively with them in "dialects" of American Sign Language (ASL), the unofficial sign lingua franca in Nigeria, communication at home with other family members is challenging and sometimes non-existent. This results in adverse social consequences including stigmatization, for the students. 

With the recent successes of AI in natural language understanding, the goal of automated sign language understanding is becoming more realistic using neural deep learning technologies.  To this effect, the proposed project aims at co-designing and developing an ongoing AI-driven two-way sign language interpretation tool that can be deployed in homes, to improve language accessibility and communication between the DHH students and other family members. This ensures inclusive and equitable social interactions and can promote lifelong learning opportunities for them outside of the school environment.

----

## [710] Interactive Machine Learning Solutions for Acoustic Monitoring of Animal Wildlife in Biosphere Reserves

**Authors**: *Thiago S. Gouvêa, Hannes Kath, Ilira Troshani, Bengt Lüers, Patricia P. Serafini, Ivan B. Campos, André S. Afonso, Sergio M. F. M. Leandro, Lourens Swanepoel, Nicholas Theron, Anthony M. Swemmer, Daniel Sonntag*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/711](https://doi.org/10.24963/ijcai.2023/711)

**Abstract**:

Biodiversity loss is taking place at accelerated rates globally, and a business-as-usual trajectory will lead to missing internationally established conservation goals. Biosphere reserves are sites designed to be of global significance in terms of both the biodiversity within them and their potential for sustainable development, and are therefore ideal places for the development of local solutions to global challenges. While the protection of biodiversity is a primary goal of biosphere reserves, adequate information on the state and trends of biodiversity remains a critical gap for adaptive management in biosphere reserves. Passive acoustic monitoring (PAM) is an increasingly popular method for continued, reproducible, scalable, and cost-effective monitoring of animal wildlife. PAM adoption is on the rise, but its data management and analysis requirements pose a barrier for adoption for most agencies tasked with monitoring biodiversity. As an interdisciplinary team of machine learning scientists and ecologists experienced with PAM and working at biosphere reserves in marine and terrestrial ecosystems on three different continents, we report on the co-development of interactive machine learning tools for semi-automated assessment of animal wildlife.

----

## [711] Data-Driven Invariant Learning for Probabilistic Programs (Extended Abstract)

**Authors**: *Jialu Bao, Nitesh Trivedi, Drashti Pathak, Justin Hsu, Subhajit Roy*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/712](https://doi.org/10.24963/ijcai.2023/712)

**Abstract**:

The weakest pre-expectation framework from Morgan and McIver for deductive verification of probabilistic programs generalizes binary state assertions to real-valued expectations to measure expected values of expressions over probabilistic program variables. While loop-free programs can be analyzed by mechanically transforming expectations, verifying programs with loops requires finding an invariant expectation.

We view invariant expectation synthesis as a regression problem: given an input state, predict the average value of the post-expectation in the output distribution. With this perspective, we develop the first data-driven invariant synthesis method for probabilistic programs. Unlike prior work on probabilistic invariant inference, our approach learns piecewise continuous invariants without relying on template expectations. We also develop a data-driven approach to learn sub-invariants from data, which can be used to upper- or lower-bound expected values. We implement our approaches and demonstrate their effectiveness on a variety of benchmarks from the probabilistic programming literature.

----

## [712] Half-Positional Objectives Recognized by Deterministic Büchi Automata (Extended Abstract)

**Authors**: *Patricia Bouyer, Antonio Casares, Mickael Randour, Pierre Vandenhove*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/713](https://doi.org/10.24963/ijcai.2023/713)

**Abstract**:

In two-player zero-sum games on graphs, the protagonist tries to achieve an objective while the antagonist aims to prevent it. Objectives for which both players do not need to use memory to play optimally are well-understood and characterized both in finite and infinite graphs. Less is known about the larger class of half-positional objectives, i.e., those for which the protagonist does not need memory (but for which the antagonist might). In particular, no characterization of half-positionality is known for the central class of ω-regular objectives.

Here, we characterize objectives recognizable by deterministic Büchi automata (a class of ω-regular objectives) that are half-positional, both over finite and infinite graphs. This characterization yields a polynomial-time algorithm to decide half-positionality of an objective recognized by a given deterministic Büchi automaton.

----

## [713] Online Certification of Preference-Based Fairness for Personalized Recommender Systems (Extended Abstract)

**Authors**: *Virginie Do, Sam Corbett-Davies, Jamal Atif, Nicolas Usunier*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/714](https://doi.org/10.24963/ijcai.2023/714)

**Abstract**:

Recommender systems are facing scrutiny because of their growing impact on the opportunities we have access to. Current audits for fairness are limited to coarse-grained parity assessments at the level of sensitive groups. We propose to audit for envy-freeness, a more granular criterion aligned with individual preferences: every user should prefer their recommendations to those of other users. Since auditing for envy requires to estimate the preferences of users beyond their existing recommendations, we cast the audit as a new pure exploration problem in multi-armed bandits. We propose a sample-efficient algorithm with theoretical guarantees that it does not deteriorate user experience. We also study the trade-offs achieved on real-world recommendation datasets.

----

## [714] Rewiring What-to-Watch-Next Recommendations to Reduce Radicalization Pathways (Extended Abstract)

**Authors**: *Francesco Fabbri, Yanhao Wang, Francesco Bonchi, Carlos Castillo, Michael Mathioudakis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/715](https://doi.org/10.24963/ijcai.2023/715)

**Abstract**:

Recommender systems typically suggest to users content similar to what they consumed in the past. A user, if happening to be exposed to strongly polarized content, might be steered towards more and more radicalized content by subsequent recommendations, eventually being trapped in what we call a "radicalization pathway". In this paper, we investigate how to mitigate radicalization pathways using a graph-based approach. We model the set of recommendations in a what-to-watch-next (W2W) recommender as a directed graph, where nodes correspond to content items, links to recommendations, and paths to possible user sessions. We measure the segregation score of a node representing radicalized content as the expected length of a random walk from that node to any node representing non-radicalized content. A high segregation score thus implies a larger chance of getting users trapped in radicalization pathways. We aim to reduce the prevalence of radicalization pathways by selecting a small number of edges to rewire, so as to minimize the maximum of segregation scores among all radicalized nodes while maintaining the relevance of recommendations. We propose an efficient yet effective greedy heuristic based on the absorbing random walk theory for the rewiring problem. Our experiments on real-world datasets confirm the effectiveness of our proposal.

----

## [715] Certified CNF Translations for Pseudo-Boolean Solving (Extended Abstract)

**Authors**: *Stephan Gocht, Ruben Martins, Jakob Nordström, Andy Oertel*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/716](https://doi.org/10.24963/ijcai.2023/716)

**Abstract**:

The dramatic improvements in Boolean satisfiability (SAT) solving since the turn of the millennium have made it possible to leverage conflict-driven clause learning (CDCL) solvers for many combinatorial problems in academia and industry, and the use of proof logging has played a crucial role in increasing the confidence that the results these solvers produce are correct. However, the fact that SAT proof logging is performed in conjunctive normal form (CNF) clausal format means that it has not been possible to extend guarantees of correctness to the use of SAT solvers for more expressive combinatorial paradigms, where the first step is an unverified translation of the input to CNF.

In this work, we show how cutting-planes-based reasoning can provide proof logging for solvers that translate pseudo-Boolean (a.k.a. 0-1 integer linear) decision problems to CNF and then run CDCL. We are hopeful that this is just a first step towards providing a unified proof logging approach that will extend to maximum satisfiability (MaxSAT) solving and pseudo-Boolean optimization in general.

----

## [716] Finite Entailment of UCRPQs over ALC Ontologies (Extended Abstract)

**Authors**: *Víctor Gutiérrez-Basulto, Albert Gutowski, Yazmín Ibáñez-García, Filip Murlak*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/717](https://doi.org/10.24963/ijcai.2023/717)

**Abstract**:

We investigate the problem of finite entailment of ontology-mediated queries. We consider the expressive query language, unions of conjunctive regular path queries (UCRPQs), extending  the well-known class of  unions of conjunctive queries, with regular expressions over roles. We look at ontologies formulated using the description logic ALC, and show a tight 2ExpTime upper bound for finite entailment of UCRPQs.

----

## [717] MV-Datalog+/-: Effective Rule-based Reasoning with Uncertain Observations (Extended Abstract)

**Authors**: *Matthias Lanzinger, Stefano Sferrazza, Georg Gottlob*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/718](https://doi.org/10.24963/ijcai.2023/718)

**Abstract**:

Modern data processing applications often combine information from a variety of complex sources. Oftentimes, some of these sources, like Machine-Learning systems or crowd-sourced data, are not strictly binary but associated with some degree of confidence in the observation. Ideally, reasoning over such data should take this additional information into account as much as possible. To this end, we propose extensions of Datalog and Datalog+/- to the semantics of Lukasiewicz logic Ł, one of the most common fuzzy logics. We show that such an extension preserves important properties from the classical case and how these properties can lead to efficient reasoning procedures for these new languages.

----

## [718] Finite-Trace Analysis of Stochastic Systems with Silent Transitions (Extended Abstract)

**Authors**: *Sander J. J. Leemans, Fabrizio Maria Maggi, Marco Montali*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/719](https://doi.org/10.24963/ijcai.2023/719)

**Abstract**:

In this paper, we summarise the main technical results obtained for specification probability. That is, we compute the probability that if a bounded stochastic Petri net produces a trace, that trace satisfies a given specification.

----

## [719] FastGR: Global Routing on CPU-GPU with Heterogeneous Task Graph Scheduler (Extended Abstract)

**Authors**: *Siting Liu, Yuan Pu, Peiyu Liao, Hongzhong Wu, Rui Zhang, Zhitang Chen, Wenlong Lv, Yibo Lin, Bei Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/720](https://doi.org/10.24963/ijcai.2023/720)

**Abstract**:

Running time is a key metric across the standard physical design flow stages. However, with the rapid growth in design sizes, routing runtime has become the runtime bottleneck in the physical design flow. To improve the effectiveness of the modern global router, we propose a global routing framework with GPU-accelerated routing algorithms and a heterogeneous task graph scheduler, called FastGR. Its runtime-oriented version FastGRL achieves 2.489× speedup compared with the state-of-the-art global router. Furthermore, the GPU-accelerated L-shape pattern routing used in FastGRL can contribute to 9.324× speedup over the sequential algorithm on CPU. Its quality-oriented version FastGRH offers further quality improvement over FastGRL with similar acceleration.

----

## [720] Learning Causal Effects on Hypergraphs (Extended Abstract)

**Authors**: *Jing Ma, Mengting Wan, Longqi Yang, Jundong Li, Brent J. Hecht, Jaime Teevan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/721](https://doi.org/10.24963/ijcai.2023/721)

**Abstract**:

Hypergraphs provide an effective abstraction for modeling multi-way group interactions among nodes, where each hyperedge can connect any number of nodes. Different from most existing studies which leverage statistical dependencies, we study hypergraphs from the perspective of causality. Specifically, we focus on the problem of individual treatment effect (ITE) estimation on hypergraphs, aiming to estimate how much an intervention (e.g., wearing face covering) would causally affect an outcome (e.g., COVID-19 infection) of each individual node. Existing works on ITE estimation either assume that the outcome of one individual should not be influenced by the treatment of other individuals (i.e., no interference), or assume the interference only exists between connected individuals in an ordinary graph. We argue that these assumptions can be unrealistic on real-world hypergraphs, where higher-order interference can affect the ITE estimations due to group interactions. We investigate high-order interference modeling, and propose a new causality learning framework powered by hypergraph neural networks. Extensive experiments on real-world hypergraphs verify the superiority of our framework over existing baselines.

----

## [721] Efficient Convex Optimization Requires Superlinear Memory (Extended Abstract)

**Authors**: *Annie Marsden, Vatsal Sharan, Aaron Sidford, Gregory Valiant*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/722](https://doi.org/10.24963/ijcai.2023/722)

**Abstract**:

Minimizing a convex function with access to a first order oracle---that returns the function evaluation and (sub)gradient at a query point---is a canonical optimization problem and a fundamental primitive in machine learning. Gradient-based methods are the most popular  approaches used for solving the problem, owing to their simplicity and computational efficiency. These methods, however, do not achieve the information-theoretically optimal query complexity for minimizing the underlying function to small error, which are achieved by more expensive techniques based on cutting-plane methods. Is it possible to achieve the information-theoretically query complexity without using these more complex and computationally expensive methods? In this work, we use memory as a lens to understand this, and show that is is not possible to achieve optimal query complexity without using significantly more memory than that used by gradient descent.

----

## [722] Algorithm-Hardware Co-Design for Efficient Brain-Inspired Hyperdimensional Learning on Edge (Extended Abstract)

**Authors**: *Yang Ni, Yeseong Kim, Tajana Rosing, Mohsen Imani*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/723](https://doi.org/10.24963/ijcai.2023/723)

**Abstract**:

In this paper, we propose an efficient framework to accelerate a lightweight brain-inspired learning solution, hyperdimensional computing (HDC), on existing edge systems. Through algorithm-hardware co-design, we optimize the HDC models to run them on the low-power host CPU and machine learning accelerators like Edge TPU. By treating the lightweight HDC learning model as a hyper-wide neural network, we exploit the capabilities of the accelerator and machine learning platform, while reducing training runtime costs by using bootstrap aggregating. Our experimental results conducted on mobile CPU and the Edge TPU demonstrate that our framework achieves 4.5 times faster training and 4.2 times faster inference than the baseline platform. Furthermore, compared to the embedded ARM CPU, Raspberry Pi, with similar power consumption, our framework achieves 19.4 times faster training and 8.9 times faster inference.

----

## [723] Sancus: Staleness-Aware Communication-Avoiding Full-Graph Decentralized Training in Large-Scale Graph Neural Networks (Extended Abstract)

**Authors**: *Jingshu Peng, Zhao Chen, Yingxia Shao, Yanyan Shen, Lei Chen, Jiannong Cao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/724](https://doi.org/10.24963/ijcai.2023/724)

**Abstract**:

Graph neural networks (GNNs) have emerged due to their success at modeling graph data. Yet, it is challenging for GNNs to efficiently scale to large graphs. Thus, distributed GNNs come into play. To avoid communication caused by expensive data movement between workers, we propose SANCUS, a staleness-aware communication-avoiding decentralized GNN system. By introducing a set of novel bounded embedding staleness metrics and adaptively skipping broadcasts, SANCUS abstracts decentralized GNN processing as sequential matrix multiplication and uses historical embeddings via cache. Theoretically, we show bounded approximation errors of embeddings and gradients with convergence guarantee. Empirically, we evaluate SANCUS with common GNN models via different system setups on large-scale benchmark datasets. Compared to SOTA works, SANCUS can avoid up to 74% communication with at least 1:86_ faster throughput on average without accuracy loss.

----

## [724] Translating Images into Maps (Extended Abstract)

**Authors**: *Avishkar Saha, Oscar Mendez, Chris Russell, Richard Bowden*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/725](https://doi.org/10.24963/ijcai.2023/725)

**Abstract**:

We approach instantaneous mapping, converting images to a top-down view of the world, as a translation problem. We show how a novel form of transformer network can be used to map from images and video directly to an overhead map or bird's-eye-view (BEV) of the world, in a single end-to-end network. We assume a 1-1 correspondence between a vertical scanline in the image, and rays passing through the camera location in an overhead map. This lets us formulate map generation from an image as a set of sequence-to-sequence translations. This constrained formulation, based upon a strong physical grounding of the problem, leads to a restricted transformer network that is convolutional in the horizontal direction only. The structure allows us to make efficient use of data when training, and obtains state-of-the-art results for instantaneous mapping of three large-scale datasets, including a 15\% and 30\% relative gain against existing best performing methods on the nuScenes and Argoverse datasets, respectively.

----

## [725] Bounding the Family-Wise Error Rate in Local Causal Discovery Using Rademacher Averages (Extended Abstract)

**Authors**: *Dario Simionato, Fabio Vandin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/726](https://doi.org/10.24963/ijcai.2023/726)

**Abstract**:

Causal discovery from observational data provides candidate causal relationships that need to be validated with ad-hoc experiments. Such experiments usually require major resources, and suitable techniques should therefore be applied to identify candidate relations while limiting false positives.
Local causal discovery provides a detailed overview of the variables influencing a target, and it focuses on two sets of variables. The first one, the Parent-Children set, comprises all the elements that are direct causes of the target or that are its direct consequences, while the second one, called the Markov boundary, is the minimal set of variables for the optimal prediction of the target.
In this paper we present RAveL, the first suite of algorithms for local causal discovery providing rigorous guarantees on false discoveries. Our algorithms exploit Rademacher averages, a key concept in statistical learning theory, to account for the multiple-hypothesis testing problem in high-dimensional scenarios. Moreover, we prove that state-of-the-art approaches cannot be adapted for the task due to their strong and untestable assumptions, and we complement our analyses with extensive experiments, on synthetic and real-world data.

----

## [726] Efficient Global Robustness Certification of Neural Networks via Interleaving Twin-Network Encoding (Extended Abstract)

**Authors**: *Zhilu Wang, Chao Huang, Qi Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/727](https://doi.org/10.24963/ijcai.2023/727)

**Abstract**:

The robustness of deep neural networks in safety-critical systems has received significant interest recently, which measures how sensitive the model output is under input perturbations. While most previous works focused on the local robustness property, the studies of the global robustness property, i.e., the robustness in the entire input space, are still lacking. In this work, we formulate the global robustness certification problem for ReLU neural networks and present an efficient approach to address it. Our approach includes a novel interleaving twin-network encoding scheme and an over-approximation algorithm leveraging relaxation and refinement techniques. Its timing efficiency and effectiveness are evaluated and compared with other state-of-the-art global robustness certification methods, and demonstrated via case studies on practical applications.

----

## [727] Learning Discrete Representations via Constrained Clustering for Effective and Efficient Dense Retrieval (Extended Abstract)

**Authors**: *Jingtao Zhan, Jiaxin Mao, Yiqun Liu, Jiafeng Guo, Min Zhang, Shaoping Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/728](https://doi.org/10.24963/ijcai.2023/728)

**Abstract**:

Dense Retrieval~(DR) has achieved state-of-the-art first-stage ranking effectiveness. However, the efficiency of most existing DR models is limited by the large memory cost of storing dense vectors and the time-consuming nearest neighbor search~(NNS) in vector space. Therefore, we present RepCONC, a novel retrieval model that learns discrete Representations via CONstrained Clustering. RepCONC jointly trains dual-encoders and the Product Quantization~(PQ) method to learn discrete document representations and enables fast approximate NNS with compact indexes. It models quantization as a constrained clustering process, which requires the document embeddings to be uniformly clustered around the quantization centroids. We theoretically demonstrate that the uniform clustering constraint facilitates representation distinguishability. Extensive experiments show that RepCONC substantially outperforms a wide range of existing retrieval models in terms of retrieval effectiveness, memory efficiency, and time efficiency.

----

## [728] Task Allocation on Networks with Execution Uncertainty (Extended Abstract)∗

**Authors**: *Yao Zhang, Xiuzhen Zhang, Dengji Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/729](https://doi.org/10.24963/ijcai.2023/729)

**Abstract**:

We study a single task allocation problem where each worker connects to some other workers to form a network and the task requester only connects to some of the workers. The goal is to design an allocation mechanism such that each worker is incentivized to invite her neighbours to join the allocation, although they are competing for the task. Moreover, the performance of each worker is uncertain, which is modelled as the quality level of her task execution. The literature has proposed solutions to tackle the uncertainty problem by paying them after verifying their execution. Here, we extend the problem to the network setting. We propose a new mechanism that guarantees that inviting more workers and reporting/performing according to her true ability is a dominant strategy for each worker. We believe that the new solution can be widely applied in the digital economy powered by social connections such as crowdsourcing.

----

## [729] Unsupervised Deep Subgraph Anomaly Detection (Extended Abstract)

**Authors**: *Zheng Zhang, Liang Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/730](https://doi.org/10.24963/ijcai.2023/730)

**Abstract**:

Effectively mining anomalous subgraphs in networks is crucial for various applications, including disease outbreak detection, financial fraud detection, and activity monitoring in social networks. However, identifying anomalous subgraphs poses significant challenges due to their complex topological structures, high-dimensional attributes, multiple notions of anomalies, and the vast subgraph space within a given graph. Classical shallow models rely on handcrafted anomaly measure functions, limiting their applicability when prior knowledge is unavailable. Deep learning-based methods have shown promise in detecting node-level, edge-level, and graph-level anomalies, but subgraph-level anomaly detection remains under-explored due to difficulties in subgraph representation learning, supervision, and end-to-end anomaly quantification. To address these challenges, this paper introduces a novel deep framework named Anomalous Subgraph Autoencoder (AS-GAE). AS-GAE leverages an unsupervised and weakly supervised approach to extract anomalous subgraphs. It incorporates a location-aware graph autoencoder to uncover anomalous areas based on reconstruction mismatches and introduces a supermodular graph scoring function module to assign meaningful anomaly scores to subgraphs within the identified anomalous areas. Extensive experiments on synthetic and real-world datasets demonstrate the effectiveness of our proposed method.

----

## [730] Harnessing Neighborhood Modeling and Asymmetry Preservation for Digraph Representation Learning

**Authors**: *Honglu Zhou, Advith Chegu, Samuel S. Sohn, Zuohui Fu, Gerard de Melo, Mubbasir Kapadia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/731](https://doi.org/10.24963/ijcai.2023/731)

**Abstract**:

Digraph Representation Learning aims to learn representations for directed homogeneous graphs (digraphs). Prior work is largely constrained or has poor generalizability across tasks. Most Graph Neural Networks exhibit poor performance on digraphs due to the neglect of modeling neighborhoods and preserving asymmetry. In this paper, we address these notable challenges by leveraging hyperbolic collaborative learning from multi-ordered partitioned neighborhoods and asymmetry-preserving regularizers. Our resulting formalism, Digraph Hyperbolic Networks (D-HYPR), is versatile for multiple tasks including node classification, link presence prediction, and link property prediction. The efficacy of D-HYPR was meticulously examined against 21 previous techniques, using 8 real-world digraph datasets. D-HYPR statistically significantly outperforms the current state of the art. We release our code at https://github. com/hongluzhou/dhypr.

----

## [731] Even If Explanations: Prior Work, Desiderata & Benchmarks for Semi-Factual XAI

**Authors**: *Saugat Aryal, Mark T. Keane*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/732](https://doi.org/10.24963/ijcai.2023/732)

**Abstract**:

Recently, eXplainable AI (XAI) research has focused on counterfactual explanations as post-hoc justifications for AI-system decisions (e.g., a customer refused a loan might be told “if you asked for a loan with a shorter term, it would have been approved”). Counterfactuals explain what changes to the input-features of an AI system change the output-decision. However, there is a sub-type of counterfactual, semi-factuals, that have received less attention in AI (though the Cognitive Sciences have studied them more). This paper surveys semi-factual explanation, summarising historical and recent work. It defines key desiderata for semi-factual XAI, reporting benchmark tests of historical algorithms (as well as a novel, naïve method) to provide a solid basis for future developments.

----

## [732] Good Explanations in Explainable Artificial Intelligence (XAI): Evidence from Human Explanatory Reasoning

**Authors**: *Ruth M. J. Byrne*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/733](https://doi.org/10.24963/ijcai.2023/733)

**Abstract**:

Insights from cognitive science about how people understand explanations can be instructive for the development of robust, user-centred explanations in eXplainable Artificial Intelligence (XAI).  I survey key tendencies that people exhibit when they construct explanations and make inferences from them, of relevance to the provision of automated explanations for decisions by AI systems. I first review experimental discoveries of some tendencies people exhibit when they construct explanations, including evidence on the illusion of explanatory depth, intuitive versus reflective explanations, and explanatory stances. I then consider discoveries of how people reason about causal explanations, including evidence on inference suppression, causal discounting, and explanation simplicity. I argue that central to the XAI endeavor is the requirement that automated explanations provided by an AI system should make sense to human users.

----

## [733] Temporal Knowledge Graph Completion: A Survey

**Authors**: *Borui Cai, Yong Xiang, Longxiang Gao, He Zhang, Yunfeng Li, Jianxin Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/734](https://doi.org/10.24963/ijcai.2023/734)

**Abstract**:

Knowledge graph completion (KGC) predicts missing links and is crucial for real-life knowledge graphs, which widely suffer from incompleteness. 
KGC methods assume a knowledge graph is static, but that may lead to inaccurate prediction results because many facts in the knowledge graphs change over time. 
Emerging methods have recently shown improved prediction results by further incorporating the temporal validity of facts; namely, temporal knowledge graph completion (TKGC). 
With this temporal information, TKGC methods explicitly learn the dynamic evolution of the knowledge graph that KGC methods fail to capture.
In this paper, for the first time, we comprehensively summarize the recent advances in TKGC research. 
First, we detail the background of TKGC, including the preliminary knowledge, benchmark datasets, and evaluation metrics. 
Then, we summarize existing TKGC methods based on how the temporal validity of facts is used to capture the temporal dynamics. 
Finally, we conclude the paper and present future research directions of TKGC.

----

## [734] Assessing and Enforcing Fairness in the AI Lifecycle

**Authors**: *Roberta Calegari, Gabriel G. Castañé, Michela Milano, Barry O'Sullivan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/735](https://doi.org/10.24963/ijcai.2023/735)

**Abstract**:

A significant challenge in detecting and mitigating bias is creating a mindset amongst AI developers to address unfairness. The current literature on fairness is broad, and the learning curve to distinguish where to use existing metrics and techniques for bias detection or mitigation is difficult. This survey systematises the state-of-the-art about distinct notions of fairness and relative techniques for bias mitigation according to the AI lifecycle. Gaps and challenges identified during the development of this work are also discussed.

----

## [735] Anti-unification and Generalization: A Survey

**Authors**: *David M. Cerna, Temur Kutsia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/736](https://doi.org/10.24963/ijcai.2023/736)

**Abstract**:

Anti-unification (AU) is a fundamental operation for generalization computation used for inductive inference. It is the dual operation to unification, an operation at the foundation of automated theorem proving. Interest in AU from the AI and related communities is growing, but without a systematic study of the concept nor surveys of existing work, investigations often resort to developing application-specific methods that existing approaches may cover. We provide the first survey of AU research and its applications and a general framework for categorizing existing and future developments.

----

## [736] Generalizing to Unseen Elements: A Survey on Knowledge Extrapolation for Knowledge Graphs

**Authors**: *Mingyang Chen, Wen Zhang, Yuxia Geng, Zezhong Xu, Jeff Z. Pan, Huajun Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/737](https://doi.org/10.24963/ijcai.2023/737)

**Abstract**:

Knowledge graphs (KGs) have become valuable knowledge resources in various applications, and knowledge graph embedding (KGE) methods have garnered increasing attention in recent years. However, conventional KGE methods still face challenges when it comes to handling unseen entities or relations during model testing. To address this issue, much effort has been devoted to various fields of KGs. In this paper, we use a set of general terminologies to unify these methods and refer to them collectively as Knowledge Extrapolation. We comprehensively summarize these methods, classified by our proposed taxonomy, and describe their interrelationships. Additionally, we introduce benchmarks and provide comparisons of these methods based on aspects that are not captured by the taxonomy. Finally, we suggest potential directions for future research.

----

## [737] A Survey on Proactive Dialogue Systems: Problems, Methods, and Prospects

**Authors**: *Yang Deng, Wenqiang Lei, Wai Lam, Tat-Seng Chua*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/738](https://doi.org/10.24963/ijcai.2023/738)

**Abstract**:

Proactive dialogue systems, related to a wide range of real-world conversational applications, equip the conversational agent with the capability of leading the conversation direction towards achieving pre-defined targets or fulfilling certain goals from the system side. It is empowered by advanced techniques to progress to more complicated tasks that require strategical and motivational interactions. In this survey, we provide a comprehensive overview of the prominent problems and advanced designs for conversational agent's proactivity in different types of dialogues. Furthermore, we discuss challenges that meet the real-world application needs but require a greater research focus in the future. We hope that this first survey of proactive dialogue systems can provide the community with a quick access and an overall picture to this practical problem, and stimulate more progresses on conversational AI to the next level.

----

## [738] Machine Learning for Cutting Planes in Integer Programming: A Survey

**Authors**: *Arnaud Deza, Elias B. Khalil*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/739](https://doi.org/10.24963/ijcai.2023/739)

**Abstract**:

We survey recent work on machine learning (ML) techniques for selecting cutting planes (or cuts) in mixed-integer linear programming (MILP). Despite the availability of various classes of cuts, the task of choosing a set of cuts to add to the linear programming (LP) relaxation at a given node of the branch-and-bound (B&B) tree has defied both formal and heuristic solutions to date. ML offers a promising approach for improving the cut selection process by using data to identify promising cuts that accelerate the solution of MILP instances. This paper presents an overview of the topic, highlighting recent advances in the literature, common approaches to data collection, evaluation, and ML model architectures. We analyze the empirical results in the literature in an attempt to quantify the progress that has been made and conclude by suggesting avenues for future research.

----

## [739] Game-theoretic Mechanisms for Eliciting Accurate Information

**Authors**: *Boi Faltings*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/740](https://doi.org/10.24963/ijcai.2023/740)

**Abstract**:

Artificial Intelligence often relies on information obtained from others through crowdsourcing, federated learning, or data markets. It is crucial to ensure that this data is accurate. Over the past 20 years, a variety of incentive mechanisms have been developed that use game theory to reward the accuracy of contributed data. These techniques are applicable to many settings where AI uses contributed data.

This survey categorizes the different techniques and their properties, and shows their limits and tradeoffs. It identifies open issues and points to possible directions to address these.

----

## [740] A Survey on Dataset Distillation: Approaches, Applications and Future Directions

**Authors**: *Jiahui Geng, Zongxiong Chen, Yuandou Wang, Herbert Woisetschlaeger, Sonja Schimmler, Ruben Mayer, Zhiming Zhao, Chunming Rong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/741](https://doi.org/10.24963/ijcai.2023/741)

**Abstract**:

Dataset distillation is attracting more attention in machine learning as training sets continue to grow and the cost of training state-of-the-art models becomes increasingly high. By synthesizing datasets with high information density, dataset distillation offers a range of potential applications, including support for continual learning, neural architecture search, and privacy protection. Despite recent advances, we lack a holistic understanding of the approaches and applications. Our survey aims to bridge this gap by first proposing a taxonomy of dataset distillation, characterizing existing approaches, and then systematically reviewing the data modalities, and related applications. In addition, we summarize the challenges and discuss future directions for this field of research.

----

## [741] A Survey on Intersectional Fairness in Machine Learning: Notions, Mitigation, and Challenges

**Authors**: *Usman Gohar, Lu Cheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/742](https://doi.org/10.24963/ijcai.2023/742)

**Abstract**:

The widespread adoption of Machine Learning systems, especially in more decision-critical applications such as criminal sentencing and bank loans, has led to increased concerns about fairness implications. Algorithms and metrics have been developed to mitigate and measure these discriminations. More recently, works have identified a more challenging form of bias called intersectional bias, which encompasses multiple sensitive attributes, such as race and gender, together. In this survey, we review the state-of-the-art in intersectional fairness. We present a taxonomy for intersectional notions of fairness and mitigation. Finally, we identify the key challenges and provide researchers with guidelines for future directions.

----

## [742] Survey on Online Streaming Continual Learning

**Authors**: *Nuwan Gunasekara, Bernhard Pfahringer, Heitor Murilo Gomes, Albert Bifet*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/743](https://doi.org/10.24963/ijcai.2023/743)

**Abstract**:

Stream Learning (SL) attempts to learn from a data stream efficiently. A data stream learning algorithm should adapt to input data distribution shifts without sacrificing accuracy. These distribution shifts are known as ”concept drifts” in the literature. SL provides many supervised, semi-supervised, and unsupervised methods for detecting and adjusting to concept drift. On the other hand, Continual Learning (CL) attempts to preserve previous knowledge while performing well on the current concept when confronted with concept drift. In Online Continual Learning (OCL), this learning happens online. This survey explores the intersection of those two online learning paradigms to find synergies. We identify this intersection as Online Streaming Continual Learning (OSCL). The study starts with a gentle introduction to SL and then explores CL. Next, it explores OSCL from SL and OCL perspectives to point out new research trends and give directions for future research.

----

## [743] Graph-based Molecular Representation Learning

**Authors**: *Zhichun Guo, Kehan Guo, Bozhao Nan, Yijun Tian, Roshni G. Iyer, Yihong Ma, Olaf Wiest, Xiangliang Zhang, Wei Wang, Chuxu Zhang, Nitesh V. Chawla*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/744](https://doi.org/10.24963/ijcai.2023/744)

**Abstract**:

Molecular representation learning (MRL) is a key step to build the connection between machine learning and chemical science. In particular, it encodes molecules as numerical vectors preserving the molecular structures and features, on top of which the downstream tasks (e.g., property prediction) can be performed. Recently, MRL has achieved considerable progress, especially in methods based on deep molecular graph learning. In this survey, we systematically review these graph-based molecular representation techniques, especially the methods incorporating chemical domain knowledge. Specifically, we first introduce the features of 2D and 3D molecular graphs. Then we summarize and categorize MRL methods into three groups based on their input. Furthermore, we discuss some typical chemical applications supported by MRL. To facilitate studies in this fast-developing area, we also list the benchmarks and commonly used datasets in the paper. Finally, we share our thoughts on future research directions.

----

## [744] Towards Utilitarian Online Learning - A Review of Online Algorithms in Open Feature Space

**Authors**: *Yi He, Christian Schreckenberger, Heiner Stuckenschmidt, Xindong Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/745](https://doi.org/10.24963/ijcai.2023/745)

**Abstract**:

Human intelligence comes from the capability to describe and make sense of the world surrounding us, often in a lifelong manner. Online Learning (OL) allows a model to simulate this capability, which involves processing data in sequence, making predictions, and learning from predictive errors. However, traditional OL assumes a fixed set of features to describe data, which can be restrictive. In reality, new features may emerge and old features may vanish or become obsolete, leading to an open
feature space. This dynamism can be caused by more advanced or outdated technology for sensing the world, or it can be a natural process of evolution. This paper reviews recent breakthroughs that strived to enable OL in open feature spaces, referred to as Utilitarian Online Learning (UOL). We taxonomize existing UOL models into three categories, analyze their pros and cons, and discuss their application scenarios. We also benchmark the performance of representative UOL models, highlighting open problems, challenges, and potential future directions of this emerging topic.

----

## [745] A Survey on User Behavior Modeling in Recommender Systems

**Authors**: *Zhicheng He, Weiwen Liu, Wei Guo, Jiarui Qin, Yingxue Zhang, Yaochen Hu, Ruiming Tang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/746](https://doi.org/10.24963/ijcai.2023/746)

**Abstract**:

User Behavior Modeling (UBM) plays a critical role in user interest learning, which has been extensively used in recommender systems. Crucial interactive patterns between users and items have been exploited, which brings compelling improvements in many recommendation tasks. In this paper, we attempt to provide a thorough survey of this research topic. We start by reviewing the research background of UBM. Then, we provide a systematic taxonomy of existing UBM research works, which can be categorized into four different directions including Conventional UBM, Long-Sequence UBM, Multi-Type UBM, and UBM with Side Information. Within each direction, representative models and their strengths and weaknesses are comprehensively discussed. Besides, we elaborate on the industrial practices of UBM methods with the hope of providing insights into the application value of existing UBM solutions. Finally, we summarize the survey and discuss the future prospects of this field.

----

## [746] Benchmarking eXplainable AI - A Survey on Available Toolkits and Open Challenges

**Authors**: *Phuong Quynh Le, Meike Nauta, Van Bach Nguyen, Shreyasi Pathak, Jörg Schlötterer, Christin Seifert*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/747](https://doi.org/10.24963/ijcai.2023/747)

**Abstract**:

The goal of Explainable AI (XAI) is to make the reasoning of a machine learning model accessible to humans, such that users of an AI system can evaluate and judge the underlying model. Due to the blackbox nature of XAI methods it is, however, hard to disentangle the contribution of a model and the explanation method to the final output. It might be unclear on whether an unexpected output is caused by the model or the explanation method. Explanation models, therefore, need to be evaluated in technical (e.g. fidelity to the model) and user-facing (correspondence to domain knowledge) terms. A recent survey has identified 29 different automated approaches to quantitatively evaluate explanations. In this work, we take an additional perspective and analyse which toolkits and data sets are available. We investigate which evaluation metrics are implemented in the toolkits and whether they produce the same results. We find that only a few aspects of explanation quality are currently covered, data sets are rare and evaluation results are not comparable across  different toolkits. Our survey can serve as a guide for the XAI community for identifying future directions of research, and most notably, standardisation of evaluation.

----

## [747] Curriculum Graph Machine Learning: A Survey

**Authors**: *Haoyang Li, Xin Wang, Wenwu Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/748](https://doi.org/10.24963/ijcai.2023/748)

**Abstract**:

Graph machine learning has been extensively studied in both academia and industry. However, in the literature, most existing graph machine learning models are designed to conduct training with data samples in a random order, which may suffer from suboptimal performance due to ignoring the importance of different graph data samples and their training orders for the model optimization status. To tackle this critical problem, curriculum graph machine learning (Graph CL), which integrates the strength of graph machine learning and curriculum learning, arises and attracts an increasing amount of attention from the research community. Therefore, in this paper, we comprehensively overview approaches on Graph CL and present a detailed survey of recent advances in this direction. Specifically, we first discuss the key challenges of Graph CL and provide its formal problem definition. Then, we categorize and summarize existing methods into three classes based on three kinds of graph machine learning tasks, i.e., node-level, link-level, and graph-level tasks. Finally, we share our thoughts on future research directions. To the best of our knowledge, this paper is the first survey for curriculum graph machine learning.

----

## [748] A Survey on Out-of-Distribution Evaluation of Neural NLP Models

**Authors**: *Xinzhe Li, Ming Liu, Shang Gao, Wray L. Buntine*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/749](https://doi.org/10.24963/ijcai.2023/749)

**Abstract**:

Adversarial robustness, domain generalization and dataset biases are three active lines of research contributing to out-of-distribution (OOD) evaluation on neural NLP models.

However, a comprehensive, integrated discussion of the three research lines is still lacking in the literature. This survey will 1) compare the three lines of research under a unifying definition; 2) summarize their data-generating processes and evaluation protocols for each line of research; and 3) emphasize the challenges and opportunities for future work.

----

## [749] Diffusion Models for Non-autoregressive Text Generation: A Survey

**Authors**: *Yifan Li, Kun Zhou, Wayne Xin Zhao, Ji-Rong Wen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/750](https://doi.org/10.24963/ijcai.2023/750)

**Abstract**:

Non-autoregressive (NAR) text generation has attracted much attention in the field of natural language processing, which greatly reduces the inference latency but has to sacrifice the generation accuracy. Recently, diffusion models, a class of latent variable generative models, have been introduced into NAR text generation, showing an improved text generation quality. In this survey, we review the recent progress in diffusion models for NAR text generation. As the background,  we first present the general definition of diffusion models and the text diffusion models, and then discuss their merits for NAR generation. As the core content, we further introduce two mainstream diffusion models in existing work of text diffusion, and review the key designs of the diffusion process. Moreover, we discuss the utilization of pre-trained language models (PLMs) for text diffusion models and introduce optimization techniques for text data. Finally, we discuss several promising directions and conclude this paper. Our survey aims to provide researchers with a systematic reference of related research on text diffusion models for NAR generation. We also demonstrate our collection of text diffusion models at https://github.com/RUCAIBox/Awesome-Text-Diffusion-Models.

----

## [750] Generative Diffusion Models on Graphs: Methods and Applications

**Authors**: *Chengyi Liu, Wenqi Fan, Yunqing Liu, Jiatong Li, Hang Li, Hui Liu, Jiliang Tang, Qing Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/751](https://doi.org/10.24963/ijcai.2023/751)

**Abstract**:

Diffusion models, as a novel generative paradigm, have achieved remarkable success in various image generation tasks such as image inpainting, image-to-text translation, and video generation. Graph generation is a crucial computational task on graphs with numerous real-world applications. It aims to learn the distribution of given graphs and then generate new graphs. Given the great success of diffusion models in image generation, increasing efforts have been made to leverage these techniques to advance graph generation in recent years.  In this paper, we first provide a comprehensive overview of generative diffusion models on graphs, In particular, we review representative algorithms for three variants of graph diffusion models, i.e., Score Matching with Langevin Dynamics (SMLD), Denoising Diffusion Probabilistic Model (DDPM), and Score-based Generative Model (SGM). Then, we summarize the major applications of generative diffusion models on graphs with a specific focus on molecule and protein modeling. Finally, we discuss promising directions in generative diffusion models on graph-structured data.

----

## [751] Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities

**Authors**: *Chuang Liu, Yibing Zhan, Jia Wu, Chang Li, Bo Du, Wenbin Hu, Tongliang Liu, Dacheng Tao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/752](https://doi.org/10.24963/ijcai.2023/752)

**Abstract**:

Graph neural networks have emerged as a leading architecture for many graph-level tasks, such as graph classification and graph generation. As an essential component of the architecture,  graph pooling is indispensable for obtaining a holistic graph-level representation of the whole graph. Although a great variety of methods have been proposed in this promising and fast-developing research field, to the best of our knowledge, little effort has been made to systematically summarize these works. To set the stage for the development of future works, in this paper, we attempt to fill this gap by providing a broad review of recent methods for graph pooling. Specifically, 1) we first propose a taxonomy of existing graph pooling methods with a mathematical summary for each category; 2) then, we provide an overview of the libraries related to graph pooling, including the commonly used datasets, model architectures for downstream tasks, and open-source implementations; 3) next, we further outline the applications that incorporate the idea of graph pooling in a variety of domains; 4) finally, we discuss certain critical challenges facing current studies and share our insights on future potential directions for research on the improvement of graph pooling.

----

## [752] A Unified View of Deep Learning for Reaction and Retrosynthesis Prediction: Current Status and Future Challenges

**Authors**: *Ziqiao Meng, Peilin Zhao, Yang Yu, Irwin King*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/753](https://doi.org/10.24963/ijcai.2023/753)

**Abstract**:

Reaction and retrosynthesis prediction are two fundamental tasks in computational chemistry. In recent years, these two tasks have attracted great attentions from both machine learning and drug discovery communities. Various deep learning approaches have been proposed to tackle these two problems and achieved initial success. In this survey, we conduct a comprehensive investigation on advanced deep learning-based reaction and retrosynthesis prediction models. We first summarize the design mechanism, strengths and weaknesses of the state-of-the-art approaches. Then we further discuss limitations of current solutions and open challenges in the problem itself. Last but not the least, we present some promising directions to facilitate future research. To our best knowledge, this paper is the first comprehensive and systematic survey on unified understanding of reaction and retrosynthesis prediction.

----

## [753] Complexity Results and Exact Algorithms for Fair Division of Indivisible Items: A Survey

**Authors**: *Trung Thanh Nguyen, Jörg Rothe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/754](https://doi.org/10.24963/ijcai.2023/754)

**Abstract**:

Fair allocation of indivisible goods is a central topic in many AI applications. Unfortunately, the corresponding problems are known to be NP-hard for many fairness concepts, so unless P = NP, exact polynomial-time algorithms cannot exist for them. In practical applications, however, it would be highly desirable to find exact solutions as quickly as possible. This motivates the study of algorithms that—even though they only run in exponential time—are as fast as possible and exactly solve such problems. We present known complexity results for them and give a survey of important techniques for designing such algorithms, mainly focusing on four common fairness notions: max-min fairness, maximin share, maximizing Nash social welfare, and envy-freeness. We also highlight the most challenging open problems for future work.

----

## [754] What Lies beyond the Pareto Front? A Survey on Decision-Support Methods for Multi-Objective Optimization

**Authors**: *Zuzanna Osika, Jazmin Zatarain Salazar, Diederik M. Roijers, Frans A. Oliehoek, Pradeep K. Murukannaiah*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/755](https://doi.org/10.24963/ijcai.2023/755)

**Abstract**:

We present a review that unifies decision-support methods for exploring the solutions produced by multi-objective optimization (MOO) algorithms. As MOO is applied to solve diverse problems, approaches for analyzing the trade-offs offered by these algorithms are scattered across fields. We provide an overview of the current advances on this topic, including methods for visualization, mining the solution set, and uncertainty exploration as well as emerging research directions, including interactivity, explainability, and support on ethical aspects. We synthesize these methods drawing from different fields of research to enable building a unified approach, independent of the application. Our goals are to reduce the entry barrier for researchers and practitioners on using MOO algorithms and to provide novel research directions.

----

## [755] Uncovering the Deceptions: An Analysis on Audio Spoofing Detection and Future Prospects

**Authors**: *Rishabh Ranjan, Mayank Vatsa, Richa Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/756](https://doi.org/10.24963/ijcai.2023/756)

**Abstract**:

Audio has become an increasingly crucial biometric modality due to its ability to provide an intuitive way for humans to interact with machines. It is currently being used for a range of applications including person authentication to banking to virtual assistants. Research has shown that these systems are also susceptible to spoofing and attacks. Therefore, protecting audio processing systems against fraudulent activities such as identity theft, financial fraud, and spreading misinformation, is of paramount importance. This paper reviews the current state-of-the-art techniques for detecting audio spoofing and discusses the current challenges along with open research problems. The paper further highlights the importance of considering the ethical and privacy implications of audio spoofing detection systems. Lastly, the work aims to accentuate the need for building more robust and generalizable methods, the integration of automatic speaker verification and countermeasure systems, and better evaluation protocols.

----

## [756] Heuristic-Search Approaches for the Multi-Objective Shortest-Path Problem: Progress and Research Opportunities

**Authors**: *Oren Salzman, Ariel Felner, Carlos Hernández, Han Zhang, Shao-Hung Chan, Sven Koenig*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/757](https://doi.org/10.24963/ijcai.2023/757)

**Abstract**:

In the multi-objective shortest-path problem we are interested in computing a path, or a set of paths that simultaneously balance multiple cost functions. This problem is important for a diverse range of applications such as transporting hazardous materials considering travel distance and risk. This family of problems is not new with results dating back to the 1970's. Nevertheless, the significant progress made in the field of heuristic search  resulted in a new and growing interest in the sub-field of multi-objective search. Consequently, in this paper we review the fundamental problems and techniques common to most algorithms and provide a general overview of the field. We then continue to describe recent work  with an emphasis on new challenges that emerged and the resulting research opportunities.

----

## [757] A Survey of Federated Evaluation in Federated Learning

**Authors**: *Behnaz Soltani, Yipeng Zhou, Venus Haghighi, John C. S. Lui*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/758](https://doi.org/10.24963/ijcai.2023/758)

**Abstract**:

In traditional machine learning, it is trivial to conduct model evaluation since all data samples are managed centrally by a server. However, model evaluation becomes a challenging problem in federated learning (FL), which is called federated evaluation in this work. This is because clients do not expose their original data to preserve data privacy. Federated evaluation plays a vital role in client selection, incentive mechanism design, malicious attack detection, etc. In this paper, we provide the first comprehensive survey of existing federated evaluation methods. Moreover, we explore various applications of federated evaluation for enhancing FL performance and finally present future research directions by envisioning some challenges.

----

## [758] Transformers in Time Series: A Survey

**Authors**: *Qingsong Wen, Tian Zhou, Chaoli Zhang, Weiqi Chen, Ziqing Ma, Junchi Yan, Liang Sun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/759](https://doi.org/10.24963/ijcai.2023/759)

**Abstract**:

Transformers have achieved superior performances in many tasks in natural language processing and computer vision, which also triggered great interest in the time series community. Among multiple advantages of Transformers, the ability to capture long-range dependencies and interactions is especially attractive for time series modeling, leading to exciting progress in various time series applications. In this paper, we systematically review Transformer schemes for time series modeling by highlighting their strengths as well as limitations. In particular, we examine the development of time series Transformers in two perspectives. From the perspective of network structure, we summarize the adaptations and modifications that have been made to Transformers in order to accommodate the challenges in time series analysis. From the perspective of applications, we categorize time series Transformers based on common tasks including forecasting, anomaly detection, and classification. Empirically, we perform robust analysis, model size analysis, and seasonal-trend decomposition analysis to study how Transformers perform in time series. Finally, we discuss and suggest future directions to provide useful research guidance.

----

## [759] A Systematic Survey of Chemical Pre-trained Models

**Authors**: *Jun Xia, Yanqiao Zhu, Yuanqi Du, Stan Z. Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/760](https://doi.org/10.24963/ijcai.2023/760)

**Abstract**:

Deep learning has achieved remarkable success in learning representations for molecules, which is crucial for various biochemical applications, ranging from property prediction to drug design. However, training Deep Neural Networks (DNNs) from scratch often requires abundant labeled molecules, which are expensive to acquire in the real world. To alleviate this issue, tremendous efforts have been devoted to Chemical Pre-trained Models (CPMs), where DNNs are pre-trained using large-scale unlabeled molecular databases and then fine-tuned over specific downstream tasks. Despite the prosperity, there lacks a systematic review of this fast-growing field. In this paper, we present the first survey that summarizes the current progress of CPMs. We first highlight the limitations of training molecular representation models from scratch to motivate CPM studies. Next, we systematically review recent advances on this topic from several key perspectives, including molecular descriptors, encoder architectures, pre-training strategies, and applications. We also highlight the challenges and promising avenues for future research, providing a useful resource for both machine learning and scientific communities.

----

## [760] Recent Advances in Direct Speech-to-text Translation

**Authors**: *Chen Xu, Rong Ye, Qianqian Dong, Chengqi Zhao, Tom Ko, Mingxuan Wang, Tong Xiao, Jingbo Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/761](https://doi.org/10.24963/ijcai.2023/761)

**Abstract**:

Recently, speech-to-text translation has attracted more and more attention and many studies have emerged rapidly. In this paper, we present a comprehensive survey on direct speech translation aiming to summarize the current state-of-the-art techniques. First, we categorize the existing research work into three directions based on the main challenges --- modeling burden, data scarcity, and application issues. To tackle the problem of modeling burden, two main structures have been proposed, encoder-decoder framework (Transformer and the variants) and multitask frameworks. For the challenge of data scarcity, recent work resorts to many sophisticated techniques, such as data augmentation, pre-training, knowledge distillation, and multilingual modeling. We analyze and summarize the application issues, which include real-time, segmentation, named entity, gender bias, and code-switching. Finally, we discuss some promising directions for future work.

----

## [761] A Survey on Masked Autoencoder for Visual Self-supervised Learning

**Authors**: *Chaoning Zhang, Chenshuang Zhang, Junha Song, John Seon Keun Yi, In So Kweon*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/762](https://doi.org/10.24963/ijcai.2023/762)

**Abstract**:

With the increasing popularity of masked autoencoders, self-supervised learning (SSL) in vision undertakes a similar trajectory as in NLP. Specifically, generative pretext tasks with the masked prediction have become a de facto standard SSL practice in NLP (e.g., BERT). By contrast, early attempts at generative methods in vision have been outperformed by their discriminative counterparts (like contrastive learning). However, the success of masked image modeling has revived the autoencoder-based visual pretraining method. As a milestone to bridge the gap with BERT in NLP, masked autoencoder in vision has attracted unprecedented attention. This work conducts a survey on masked autoencoders for visual SSL.

----

## [762] State-wise Safe Reinforcement Learning: A Survey

**Authors**: *Weiye Zhao, Tairan He, Rui Chen, Tianhao Wei, Changliu Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/763](https://doi.org/10.24963/ijcai.2023/763)

**Abstract**:

Despite the tremendous success of Reinforcement Learning (RL) algorithms in simulation environments, applying RL to real-world applications still faces many challenges. A major concern is safety, in another word, constraint satisfaction. State-wise constraints are one of the most common constraints in real-world applications and one of the most challenging constraints in Safe RL. Enforcing state-wise constraints is necessary and essential to many challenging tasks such as autonomous driving, robot manipulation. This paper provides a comprehensive review of existing approaches that address state-wise constraints in RL. Under the framework of State-wise Constrained Markov Decision Process (SCMDP), we will discuss the connections, differences, and trade-offs of existing approaches in terms of (i) safety guarantee and scalability, (ii) safety and reward performance, and (iii) safety after convergence and during training. We also summarize limitations of current methods and discuss potential future directions.

----

## [763] A Survey on Efficient Training of Transformers

**Authors**: *Bohan Zhuang, Jing Liu, Zizheng Pan, Haoyu He, Yuetian Weng, Chunhua Shen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/764](https://doi.org/10.24963/ijcai.2023/764)

**Abstract**:

Recent advances in Transformers have come with a huge requirement on computing resources, highlighting the importance of developing efficient training techniques to make Transformer training faster, at lower cost, and to higher accuracy by the efficient use of computation and memory resources. This survey provides the first systematic overview of the efficient training of Transformers, covering the recent progress in acceleration arithmetic and hardware, with a focus on the former. We analyze and compare methods that save computation and memory costs for intermediate tensors during training, together with techniques on hardware/algorithm co-design. We finally discuss challenges and promising areas for future research.

----

## [764] Conjure: Automatic Generation of Constraint Models from Problem Specifications (Extended Abstract)

**Authors**: *Özgür Akgün, Alan M. Frisch, Ian P. Gent, Christopher Jefferson, Ian Miguel, Peter Nightingale*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/765](https://doi.org/10.24963/ijcai.2023/765)

**Abstract**:

When solving a combinatorial problem, the formulation or model of the problem is critical to the efficiency of the solver. Automating the modelling process has long been of interest given the expertise and time required to develop an effective model of a particular problem. We describe a method to automatically produce constraint models from a problem specification written in the abstract constraint specification language Essence.   Our approach is to incrementally refine the specification into a concrete model by applying a chosen refinement rule at each step. Any non-trivial specification may be refined in multiple ways, creating a diverse space of models to choose from.

The handling of symmetries is a particularly important aspect of automated modelling. 
We show how modelling symmetries may be broken automatically as they enter a model during refinement, removing the need for an expensive symmetry detection step following model formulation.

Our approach is implemented in a system called Conjure. We compare the models produced by Conjure to constraint models from the literature that are known to be effective.  Our empirical results confirm that Conjure can reproduce successfully the kernels of the constraint models of 42 benchmark problems found in the literature.

----

## [765] Survey and Evaluation of Causal Discovery Methods for Time Series (Extended Abstract)

**Authors**: *Charles K. Assaad, Emilie Devijver, Éric Gaussier*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/766](https://doi.org/10.24963/ijcai.2023/766)

**Abstract**:

We introduce in this survey the major concepts, models, and algorithms proposed so far to infer causal relations from observational time series, a task usually referred to as causal discovery in time series. To do so, after a description of the underlying concepts and modelling assumptions, we present different methods according to the family of approaches they belong to: Granger causality, constraint-based approaches, noise-based approaches, score-based approaches, logic-based approaches, topology-based approaches, and difference-based approaches. We then evaluate several representative methods to illustrate the behaviour of different families of approaches. This illustration is conducted on both artificial and real datasets, with different characteristics. The main conclusions one can draw from this survey is that causal discovery in times series is an active research field in which new methods (in every family of approaches) are regularly proposed, and that no family or method stands out in all situations. Indeed, they all rely on assumptions that may or may not be appropriate for a particular dataset.

----

## [766] Adversarial Framework with Certified Robustness for Time-Series Domain via Statistical Features (Extended Abstract)

**Authors**: *Taha Belkhouja, Janardhan Rao Doppa*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/767](https://doi.org/10.24963/ijcai.2023/767)

**Abstract**:

Time-series data arises in many real-world applications (e.g., mobile health) and deep neural networks (DNNs) have shown great success in solving them. Despite their success, little is known about their robustness to adversarial attacks. In this paper, we propose a novel adversarial framework referred to as Time-Series Attacks via STATistical Features (TSA-STAT). To address the unique challenges of time-series domain, TSA-STAT employs constraints on statistical features of the time-series data to construct adversarial examples. Optimized polynomial transformations are used to create attacks that are more effective (in terms of successfully fooling DNNs) than those based on additive perturbations. We also provide certified bounds on the norm of the statistical features for constructing adversarial examples.  Our experiments on diverse real-world benchmark datasets show the effectiveness of TSA-STAT in fooling DNNs for time-series domain and in improving their robustness.

----

## [767] Constraint Solving Approaches to the Business-to-Business Meeting Scheduling Problem (Extended Abstract)

**Authors**: *Miquel Bofill, Jordi Coll, Marc Garcia, Jesús Giráldez-Cru, Gilles Pesant, Josep Suy, Mateu Villaret*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/768](https://doi.org/10.24963/ijcai.2023/768)

**Abstract**:

The B2B Meeting Scheduling Optimization Problem (B2BSP) consists of scheduling a set of meetings between given pairs of participants to an event, minimizing idle time periods in participants' schedules, while taking into account participants’ availability and accommodation capacity. Therefore, it constitutes a challenging combinatorial problem in many real-world B2B events.

This work presents a comparative study of several approaches to solve this problem. They are based on Constraint Programming (CP), Mixed Integer Programming (MIP) and Maximum Satisfiability (MaxSAT). The CP approach relies on using global constraints and has been implemented in MiniZinc to be able to compare CP, Lazy Clause Generation and MIP as solving technologies in this setting. A pure MIP encoding is also presented. Finally, an alternative viewpoint is considered under MaxSAT, showing the best performance when considering some implied constraints. Experimental results on real world B2B instances, as well as on crafted ones, show that the MaxSAT approach is the one with the best performance for this problem, exhibiting better solving times, sometimes even orders of magnitude smaller than CP and MIP.

----

## [768] SAT Encodings for Pseudo-Boolean Constraints Together With At-Most-One Constraints (Extended Abstract)

**Authors**: *Miquel Bofill, Jordi Coll, Peter Nightingale, Josep Suy, Felix Ulrich-Oltean, Mateu Villaret*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/769](https://doi.org/10.24963/ijcai.2023/769)

**Abstract**:

When solving a combinatorial problem using propositional satisfiability (SAT), the encoding of the constraints  is of vital importance. 
Pseudo-Boolean (PB) constraints  appear frequently in a wide variety of problems. When PB constraints occur together with at-most-one (AMO) constraints over the same variables, they can be combined into PB(AMO) constraints. 
In this paper we present new encodings  for PB(AMO) constraints. 
Our experiments show that these encodings  can be substantially smaller than those of PB constraints and allow many more instances to be solved within a time limit. 
We also observed that there is no single overall winner among the considered encodings, but efficiency of each encoding may depend on PB(AMO) characteristics such as the magnitude of coefficient values.

----

## [769] A False Sense of Security (Extended Abstract)

**Authors**: *Piero A. Bonatti*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/770](https://doi.org/10.24963/ijcai.2023/770)

**Abstract**:

The growing literature on confidentiality in knowledge representation and reasoning sometimes may cause a false sense of security, due to lack of details about
the attacker, and some misconceptions about security-related concepts. This paper
analyzes the vulnerabilities of some recent knowledge protection methods to increase the awareness about their actual effectiveness and their mutual differences.

----

## [770] Optimizing the Computation of Overriding in DLN (Extended Abstract)

**Authors**: *Piero A. Bonatti, Iliana M. Petrova, Luigi Sauro*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/771](https://doi.org/10.24963/ijcai.2023/771)

**Abstract**:

One of the factors that hinder the adoption of nonmonotonic description logics in applications is performance. Even when monotonic and nonmonotonic inferences have the same asymptotic complexity, the implementation of nonmonotonic reasoning may be significantly slower.  The family of nonmonotonic logics DLN is no exception to this behavior.

We address this issue by introducing two provably correct and complete optimizations for reasoning in DLN. The first optimization is a module extractor that has the purpose of focusing reasoning on a relevant subset of the knowledge base. The second, called optimistic evaluation, aims at exploiting incremental reasoning in a better way. 
    
Extensive experimental evaluation shows that the optimized DLN reasoning is often compatible with interactive query answering, thus bringing nonmonotonic description logics closer to practical applications.

----

## [771] SAMBA: A Generic Framework for Secure Federated Multi-Armed Bandits (Extended Abstract)

**Authors**: *Radu Ciucanu, Pascal Lafourcade, Gael Marcadet, Marta Soare*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/772](https://doi.org/10.24963/ijcai.2023/772)

**Abstract**:

We tackle the problem of secure cumulative reward maximization in multi-armed bandits in a cross-silo federated learning setting. Under the orchestration of a central server, each data owner participating at the cumulative reward computation has the guarantee that its raw data is not seen by some other participant. We rely on cryptographic schemes and propose SAMBA, a generic framework for Secure federAted Multi-armed BAndits. We show that SAMBA returns the same cumulative reward as the non-secure versions of bandit algorithms, while satisfying formally proven security properties. We also show that the overhead due to cryptographic primitives is linear in the size of the input, which is confirmed by our implementation.

----

## [772] Data-Driven Revision of Conditional Norms in Multi-Agent Systems (Extended Abstract)

**Authors**: *Davide Dell'Anna, Natasha Alechina, Fabiano Dalpiaz, Mehdi Dastani, Brian Logan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/773](https://doi.org/10.24963/ijcai.2023/773)

**Abstract**:

In multi-agent systems, norm enforcement is a mechanism for steering the behavior of individual agents in order to achieve desired system-level objectives. Due to the dynamics of multi-agent systems, however, it is hard to design norms that guarantee the achievement of the objectives in every operating context. Also, these objectives may change over time, thereby making previously defined norms ineffective. In this paper, we investigate the use of system execution data to automatically synthesise and revise conditional prohibitions with deadlines, a type of norms aimed at preventing agents from exhibiting certain patterns of behaviors. We propose DDNR (Data-Driven Norm Revision), a data-driven approach to norm revision that synthesises revised norms with respect to a data set of traces describing the behavior of the agents in the system. We evaluate DDNR using a state-of-the-art, off-the-shelf urban traffic simulator. The results show that DDNR synthesises revised norms that are significantly more accurate than the original norms in distinguishing adequate and inadequate behaviors for the achievement of the system-level objectives.

----

## [773] Interpretable Local Concept-based Explanation with Human Feedback to Predict All-cause Mortality (Extended Abstract)

**Authors**: *Radwa El Shawi, Mouaz H. Al-Mallah*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/774](https://doi.org/10.24963/ijcai.2023/774)

**Abstract**:

Machine learning models are incorporated in different fields and disciplines, some of which require high accountability and transparency, for example, the healthcare sector. A widely used category of explanation techniques attempts to explain models' predictions by quantifying the importance score of each input feature. However, summarizing such scores to provide human-interpretable explanations is challenging. Another category of explanation techniques focuses on learning a domain representation in terms of high-level human-understandable concepts and then utilizing them to explain predictions. These explanations are hampered by how concepts are constructed, which is not intrinsically interpretable. To this end, we propose Concept-based Local Explanations with Feedback (CLEF), a novel local model agnostic explanation framework for learning a set of high-level transparent concept definitions in high-dimensional tabular data that uses clinician-labeled concepts rather than raw features.

----

## [774] Core Challenges in Embodied Vision-Language Planning (Extended Abstract)

**Authors**: *Jonathan Francis, Nariaki Kitamura, Felix Labelle, Xiaopeng Lu, Ingrid Navarro, Jean Oh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/775](https://doi.org/10.24963/ijcai.2023/775)

**Abstract**:

Recent advances in the areas of Multimodal Machine Learning and Artificial Intelligence (AI) have led to the development of challenging tasks at the intersection of Computer Vision, Natural Language Processing, and Robotics. Whereas many approaches and previous survey pursuits have characterised one or two of these dimensions, there has not been a holistic analysis at the center of all three. Moreover, even when combinations of these topics are considered, more focus is placed on describing, e.g., current architectural methods, as opposed to also illustrating high-level challenges and opportunities for the field. In this survey paper, we discuss Embodied Vision-Language Planning (EVLP) tasks, a family of prominent embodied navigation and manipulation problems that jointly leverage computer vision and natural language for interaction in physical environments. We propose a taxonomy to unify these tasks and provide an in-depth analysis and comparison of the new and current algorithmic approaches, metrics, simulators, and datasets used for EVLP tasks. Finally, we present the core challenges that we believe new EVLP works should seek to address, and we advocate for task construction that enables model generalisability and furthers real-world deployment.

----

## [775] Multi-Agent Advisor Q-Learning (Extended Abstract)

**Authors**: *Sriram Ganapathi Subramanian, Matthew E. Taylor, Kate Larson, Mark Crowley*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/776](https://doi.org/10.24963/ijcai.2023/776)

**Abstract**:

In the last decade, there have been significant advances in multi-agent reinforcement learning (MARL) but there are still numerous challenges, such as high sample complexity and slow convergence to stable policies, that need to be overcome before wide-spread deployment is possible. However, many real-world environments already, in practice,  deploy sub-optimal or heuristic approaches for generating policies. An interesting question that arises is how to best use such approaches as advisors to help improve reinforcement learning in multi-agent domains. We provide a principled framework for incorporating action recommendations from online sub-optimal advisors in multi-agent settings. We describe the problem of ADvising Multiple Intelligent Reinforcement Agents (ADMIRAL) in nonrestrictive general-sum stochastic game environments and present two novel Q-learning-based algorithms: ADMIRAL - Decision Making (ADMIRAL-DM) and ADMIRAL - Advisor Evaluation (ADMIRAL-AE), which allow us to improve learning by appropriately incorporating advice from an advisor (ADMIRAL-DM), and evaluate the effectiveness of an advisor (ADMIRAL-AE). We analyze the algorithms theoretically and provide fixed point guarantees regarding their learning in general-sum stochastic games. Furthermore, extensive experiments illustrate that these algorithms: can be used in a variety of environments, have performances that compare favourably to other related baselines, can scale to large state-action spaces, and are robust to poor advice from advisors.

----

## [776] Creative Problem Solving in Artificially Intelligent Agents: A Survey and Framework (Extended Abstract)

**Authors**: *Evana Gizzi, Lakshmi Nair, Sonia Chernova, Jivko Sinapov*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/777](https://doi.org/10.24963/ijcai.2023/777)

**Abstract**:

Creative Problem Solving (CPS) is a sub-area within artificial intelligence that focuses on methods for solving off-nominal, or anomalous problems in autonomous systems. Despite many advancements in planning and learning in AI, resolving novel problems or adapting existing knowledge to a new context, especially in cases where the environment may change in unpredictable ways, remains a challenge. To stimulate further research in CPS, we contribute a definition and a framework of CPS, which we use to categorize existing AI methods in this field. We conclude our survey with open research questions, and suggested future directions.

----

## [777] Ordinal Maximin Share Approximation for Goods (Extended Abstract)

**Authors**: *Hadi Hosseini, Andrew Searns, Erel Segal-Halevi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/778](https://doi.org/10.24963/ijcai.2023/778)

**Abstract**:

In fair division of indivisible goods, l-out-of-d maximin share (MMS) is the value that an agent can guarantee by partitioning the goods into d bundles and choosing the l least preferred bundles. Most existing works aim to guarantee to all agents a constant fraction of their 1-out-of-n MMS. But this guarantee is sensitive to small perturbation in agents' cardinal valuations. We consider a more robust approximation notion, which depends only on the agents' ordinal rankings of bundles. We prove the existence of l-out-of-floor((l+1/2)n) MMS allocations of goods for any integer l greater than or equal to 1, and present a polynomial-time algorithm that finds a 1-out-of-ceiling(3n/2) MMS allocation when l = 1. We further develop an algorithm that provides a weaker ordinal approximation to MMS for any l > 1.

----

## [778] On Tackling Explanation Redundancy in Decision Trees (Extended Abstract)

**Authors**: *Yacine Izza, Alexey Ignatiev, João Marques-Silva*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/779](https://doi.org/10.24963/ijcai.2023/779)

**Abstract**:

Claims about the interpretability of decision trees can be traced back to the origins of machine learning (ML). Indeed, given some input consistent with a decision tree's path, the explanation for  the resulting prediction consists of the features in that path.   Moreover, a growing number of works propose the use of decision trees, and of other so-called interpretable models, as a possible solution for deploying ML models in high-risk applications. This paper overviews recent theoretical and practical results which demonstrate that for most decision trees, tree paths exhibit so-called explanation redundancy, in that logically sound explanations can often be significantly more succinct than what the features in the path dictates.  More importantly, such decision tree explanations can be computed in polynomial-time, and so can be produced with essentially no effort other than traversing the decision tree. The experimental results, obtained on a large range of publicly available decision trees, support the paper's claims.

----

## [779] Get Out of the BAG! Silos in AI Ethics Education: Unsupervised Topic Modeling Analysis of Global AI Curricula (Extended Abstract)

**Authors**: *Rana Tallal Javed, Osama Nasir, Melania Borit, Loïs Vanhée, Elias Zea, Shivam Gupta, Ricardo Vinuesa, Junaid Qadir*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/780](https://doi.org/10.24963/ijcai.2023/780)

**Abstract**:

This study explores the topics and trends of teaching AI ethics in higher education, using Latent Dirichlet Allocation as the analysis tool. The analyses included 166 courses from 105 universities around the world. Building on the uncovered patterns, we distil a model of current pedagogical practice, the BAG model (Build, Assess, and Govern), that combines cognitive levels, course content, and disciplines. The study critically assesses the implications of this teaching paradigm and challenges practitioners to reflect on their practices and move beyond stereotypes and biases.

----

## [780] Data-Informed Knowledge and Strategies (Extended Abstract)

**Authors**: *Junli Jiang, Pavel Naumov*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/781](https://doi.org/10.24963/ijcai.2023/781)

**Abstract**:

The article proposes a new approach to reasoning about knowledge and strategies in multiagent systems. It emphasizes data, not agents, as the source of strategic knowledge. The approach brings together Armstrong's functional dependency expression from database theory, a data-informed knowledge modality based on a recent work by Baltag and van Benthem, and a newly proposed data-informed strategy modality. The main technical result is a sound and complete logical system that describes the interplay between these three logical operators.

----

## [781] Gradient-Based Mixed Planning with Symbolic and Numeric Action Parameters (Extended Abstract)

**Authors**: *Kebing Jin, Hankz Hankui Zhuo, Zhanhao Xiao, Hai Wan, Subbarao Kambhampati*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/782](https://doi.org/10.24963/ijcai.2023/782)

**Abstract**:

Dealing with planning problems with both logical relations and numeric changes in real-world dynamic environments is challenging. Existing numeric planning systems for the problem often discretize numeric variables or impose convex constraints on numeric variables, which harms the performance when solving problems, especially when the problems contain obstacles and non-linear numeric effects. In this work, we propose a novel algorithm framework to solve numeric planning problems mixed with logical relations and numeric changes based on gradient descent. We cast the numeric planning with logical relations and numeric changes as an optimization problem. Specifically, we extend the syntax to allow parameters of action models to be either objects or real-valued numbers, which enhances the ability to model real-world numeric effects. Based on the extended modeling language, we propose a gradient-based framework to simultaneously optimize numeric parameters and compute appropriate actions to form candidate plans. The gradient-based framework is composed of an algorithmic heuristic module based on propositional operations to select actions and generate constraints for gradient descent, an algorithmic transition module to update states to the next ones, and a loss module to compute loss. We repeatedly minimize loss by updating numeric parameters and compute candidate plans until it converges into a valid plan for the planning problem.

----

## [782] Rethinking Formal Models of Partially Observable Multiagent Decision Making (Extended Abstract)

**Authors**: *Vojtech Kovarík, Martin Schmid, Neil Burch, Michael Bowling, Viliam Lisý*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/783](https://doi.org/10.24963/ijcai.2023/783)

**Abstract**:

Multiagent decision-making in partially observable environments is usually modelled as either an extensive-form game (EFG) in game theory or a partially observable stochastic game (POSG) in multiagent reinforcement learning (MARL). One issue with the current situation is that while most practical problems can be modelled in both formalisms, the relationship of the two models is unclear, which hinders the transfer of ideas between the two communities. A second issue is that while EFGs have recently seen significant algorithmic progress, their classical formalization is unsuitable for efficient presentation of the underlying ideas, such as those around decomposition.
To solve the first issue, we introduce factored-observation stochastic games (FOSGs), a minor modification of the POSG formalism which distinguishes between private and public observation and thereby greatly simplifies decomposition. To remedy the second issue, we show that FOSGs and POSGs are naturally connected to EFGs: by "unrolling" a FOSG into its tree form, we obtain an EFG. Conversely, any perfect-recall timeable EFG corresponds to some underlying FOSG in this manner. Moreover, this relationship justifies several minor modifications to the classical EFG formalization that recently appeared as an implicit response to the model's issues with decomposition. Finally, we illustrate the transfer of ideas between EFGs and MARL by presenting three key EFG techniques -- counterfactual regret minimization, sequence form, and decomposition -- in the FOSG framework.

----

## [783] Mean-Semivariance Policy Optimization via Risk-Averse Reinforcement Learning (Extended Abstract)

**Authors**: *Xiaoteng Ma, Shuai Ma, Li Xia, Qianchuan Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/784](https://doi.org/10.24963/ijcai.2023/784)

**Abstract**:

Keeping risk under control is often more crucial than maximizing expected rewards in real-world decision-making situations, such as finance, robotics, autonomous driving, etc. The most natural choice of risk measures is variance, while it penalizes the upside volatility as much as the downside part. Instead, the (downside) semivariance, which captures negative deviation of a random variable under its mean, is more suitable for risk-averse proposes. This paper aims at optimizing the mean-semivariance (MSV) criterion in reinforcement learning w.r.t. steady reward distribution. Since semivariance is time-inconsistent and does not satisfy the standard Bellman equation, the traditional dynamic programming methods are inapplicable to MSV problems directly. To tackle this challenge, we resort to Perturbation Analysis (PA) theory and establish the performance difference formula for MSV. We reveal that the MSV problem can be solved by iteratively solving a sequence of RL problems with a policy-dependent reward function. Further, we propose two on-policy algorithms based on the policy gradient theory and the trust region method. Finally, we conduct diverse experiments from simple bandit problems to continuous control tasks in MuJoCo, which demonstrate the effectiveness of our proposed methods.

----

## [784] Learning to Design Fair and Private Voting Rules (Extended Abstract)

**Authors**: *Farhad Mohsin, Ao Liu, Pin-Yu Chen, Francesca Rossi, Lirong Xia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/785](https://doi.org/10.24963/ijcai.2023/785)

**Abstract**:

Voting is used widely to aggregate preferences to make a collective decision. In this paper, we focus on evaluating and designing voting rules that support both the privacy of the voting agents and a notion of fairness over such agents. First, we introduce a novel notion of group fairness and adopt the existing notion of local differential privacy. We then evaluate the level of group fairness in several existing voting rules, as well as the trade-offs between fairness and privacy, showing that it is not possible to always obtain maximal economic efficiency with high fairness. 
Then, we present both a machine learning and a constrained optimization approach to design new voting rules that are fair while maintaining a high level of economic efficiency. Finally, we empirically examine the effect of adding noise to create local differentially private voting rules and discuss the three-way trade-off between economic efficiency, fairness, and privacy.

----

## [785] A Computational Model of Ostrom's Institutional Analysis and Development Framework (Extended Abstract)

**Authors**: *Nieves Montes, Nardine Osman, Carles Sierra*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/786](https://doi.org/10.24963/ijcai.2023/786)

**Abstract**:

Ostrom's Institutional Analysis and Development (IAD) framework represents a comprehensive theoretical effort to identify and outline the variables that determine the outcome in any social interaction. Taking inspiration from it, we define the Action Situation Language (ASL), a machine-readable logical language to express the components of a multiagent interaction, with a special focus on the rules adopted by the community. The ASL is complemented by a game engine that takes an interaction description as input and automatically grounds its semantics as an Extensive-Form Game (EFG), which can be readily analysed using standard game-theoretical solution concepts. Overall, our model allows a community of agents to perform what-if analysis on a set of rules being considered for adoption, by automatically connecting rule configurations to the outcomes they incentivize.

----

## [786] Proofs and Certificates for Max-SAT (Extended Abstract)

**Authors**: *Matthieu Py, Mohamed Sami Cherif, Djamal Habet*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/787](https://doi.org/10.24963/ijcai.2023/787)

**Abstract**:

In this paper, we present a tool, called MS-Builder, which generates certificates for the Max-SAT problem in the particular form of a sequence of equivalence-preserving transformations. To generate a certificate, MS-Builder iteratively calls a SAT oracle to get a SAT resolution refutation which is handled and adapted into a sound refutation for Max-SAT. In particular, the size of the computed Max-SAT refutation is linear with respect to the size of the initial refutation if it is semi-read-once, tree-like regular, tree-like or semi-tree-like. Additionally, we propose an extendable tool, called MS-Checker, able to verify the validity of any Max-SAT certificate using Max-SAT inference rules.

----

## [787] Automatic Recognition of the General-Purpose Communicative Functions Defined by the ISO 24617-2 Standard for Dialog Act Annotation (Extended Abstract)

**Authors**: *Eugénio Ribeiro, Ricardo Ribeiro, David Martins de Matos*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/788](https://doi.org/10.24963/ijcai.2023/788)

**Abstract**:

From the perspective of a dialog system, the identification of the intention behind the segments in a dialog is important, as it provides cues regarding the information present in the segments and how they should be interpreted. The ISO 24617-2 standard for dialog act annotation defines a hierarchically organized set of general-purpose communicative functions that correspond to different intentions that are relevant in the context of a dialog. In this paper, we explore the automatic recognition of these functions. To do so, we propose to adapt existing approaches to dialog act recognition, so that they can deal with the hierarchical classification problem. More specifically, we propose the use of an end-to-end hierarchical network with cascading outputs and maximum a posteriori path estimation to predict the communicative function at each level of the hierarchy, preserve the dependencies between the functions in the path, and decide at which level to stop. Additionally, we rely on transfer learning processes to address the data scarcity problem. Our experiments on the DialogBank show that this approach outperforms both flat and hierarchical approaches based on multiple classifiers and that each of its components plays an important role in the recognition of general-purpose communicative functions.

----

## [788] Memory-Limited Model-Based Diagnosis (Extended Abstract)

**Authors**: *Patrick Rodler*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/789](https://doi.org/10.24963/ijcai.2023/789)

**Abstract**:

Model-based diagnosis is a principled and broadly applicable AI-based approach to tackle debugging problems in a wide range of areas including software, knowledge bases, circuits, cars, and robots. Whenever the sound and complete computation of fault explanations in a given preference order (e.g., cardinality or probability) is required, all existing diagnosis algorithms suffer from an exponential space complexity. This can prevent their application on memory-restricted devices and for memory-intensive problem cases. As a remedy, we propose RBF-HS, a diagnostic search based on Korf’s seminal RBFS algorithm which can enumerate an arbitrary ﬁxed number of fault explanations in best-ﬁrst order within linear space bounds, without sacriﬁcing other desirable properties. Evaluations on real-world diagnosis cases show that RBF-HS, when used to compute minimum-cardinality fault explanations, in most cases saves substantial space while requiring only reasonably more or even less time than Reiter’s HS-Tree, one of the most inﬂuential diagnostic algorithms with the same properties.

----

## [789] Q-Learning-Based Model Predictive Variable Impedance Control for Physical Human-Robot Collaboration (Extended Abstract)

**Authors**: *Loris Roveda, Andrea Testa, Asad Ali Shahid, Francesco Braghin, Dario Piga*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/790](https://doi.org/10.24963/ijcai.2023/790)

**Abstract**:

Physical human-robot collaboration is increasingly required in many contexts. To implement an effective collaboration, the robot should be able to recognize the humanâ€™s intentions and guarantee safe and adaptive behavior along the intended motion directions. The robot-control strategies with such attributes are particularly demanded in the industrial field. Indeed, with this aim, this work proposes a Q-Learning-based Model Predictive Variable Impedance Control (Q-LMPVIC) to assist the operators in physical human-robot collaboration (pHRC) tasks. A Cartesian impedance control loop is designed to implement decoupled compliant robot dynamics. The impedance control parameters (i.e., setpoint and damping parameters) are then optimized online in order to maximize the performance of the pHRC. For this purpose, an ensemble of neural networks is designed to learn the modeling of the human-robot interaction dynamics while capturing the associated uncertainties. The derived modeling is then exploited by the model predictive controller (MPC), enhanced with stability guarantees by means of Lyapunov constraints. The MPC is solved by making use of a Q-Learning method that, in its online implementation, uses an actor-critic algorithm to approximate the exact solution. Indeed, the Q-learning method provides an accurate and highly efficient solution (in terms of computational time and resources). The proposed approach has been validated through experimental tests, in which a Franka EMIKA panda robot has been used as a test platform.

----

## [790] A Survey of Methods for Automated Algorithm Configuration (Extended Abstract)

**Authors**: *Elias Schede, Jasmin Brandt, Alexander Tornede, Marcel Wever, Viktor Bengs, Eyke Hüllermeier, Kevin Tierney*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/791](https://doi.org/10.24963/ijcai.2023/791)

**Abstract**:

Algorithm configuration (AC) is concerned with the automated search of the most suitable parameter configuration of a parametrized algorithm. There are currently a wide variety of AC problem variants and methods proposed in the literature. Existing reviews do not take into account all derivatives of the AC problem, nor do they offer a complete classification scheme. To this end, we introduce taxonomies to describe the AC problem and features of configuration methods, respectively. Existing AC literature is classified and characterized by the provided taxonomies.

----

## [791] Motion Planning Under Uncertainty with Complex Agents and Environments via Hybrid Search (Extended Abstract)

**Authors**: *Daniel Strawser, Brian Williams*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/792](https://doi.org/10.24963/ijcai.2023/792)

**Abstract**:

As autonomous systems tackle more real-world situations, mission success oftentimes cannot be guaranteed and the planner must reason about the probability of failure.  Unfortunately, computing a trajectory that satisfies mission goals while constraining the probability of failure is difficult because of the need to reason about complex, multidimensional probability distributions.  Recent methods have seen success using chance-constrained, model-based planning.  We argue there are two main drawbacks to these approaches.  First, current methods suffer from an inability to deal with expressive environment models such as 3D non-convex obstacles.  Second, most planners rely on considerable simplifications when computing trajectory risk including approximating the agent's dynamics, geometry, and uncertainty.  We apply hybrid search to the risk-bound, goal-directed planning problem.  The hybrid search consists of a region planner and a trajectory planner.  The region planner makes discrete choices by reasoning about geometric regions that the agent should visit in order to accomplish its mission.  In formulating the region planner, we propose landmark regions that help produce obstacle-free paths.  The region planner passes paths through the environment to a trajectory planner; the task of the trajectory planner is to optimize trajectories that respect the agent's dynamics and the user's desired risk of mission failure.  We discuss three approaches to modeling trajectory risk: a CDF-based approach, a sampling-based collocation method, and an algorithm named Shooting Method Monte Carlo.  A variety of 2D and 3D test cases are presented in the full paper including a linear case, a Dubins car model, and an underwater autonomous vehicle.  The method is shown to outperform other methods in terms of speed and utility of the solution.  Additionally, the models of trajectory risk are shown to better approximate risk in simulation.

----

## [792] Incremental Event Calculus for Run-Time Reasoning (Extended Abstract)

**Authors**: *Efthimis Tsilionis, Alexander Artikis, Georgios Paliouras*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/793](https://doi.org/10.24963/ijcai.2023/793)

**Abstract**:

We present a system for online, incremental composite event recognition. In streaming environments, the usual case is for data to arrive with a (variable) delay from, and to be revised by, the underlying sources. We propose RTEC_inc, an incremental version of RTEC, a composite event recognition engine with formal, declarative semantics, that has been shown to scale to several real-world data streams. RTEC deals with delayed arrival and revision of events by computing all queries from scratch. This is often inefficient since it results in redundant computations. Instead, RTEC_inc deals with delays and revisions in a more efficient way, by updating only the affected queries. We compare RTEC_inc and RTEC experimentally using real-world and synthetic datasets. The results are compatible with our complexity analysis and show that RTEC_inc outperforms RTEC in many practical cases.

----

## [793] Ethical By Designer - How to Grow Ethical Designers of Artificial Intelligence (Extended Abstract)

**Authors**: *Loïs Vanhée, Melania Borit*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/794](https://doi.org/10.24963/ijcai.2023/794)

**Abstract**:

Ethical concerns regarding Artificial Intelligence technology have fueled discussions around the ethics training received by its designers. Training designers for ethical behaviour, understood as habitual application of ethical principles in any situation, can make a significant difference in the practice of research, development, and application of AI systems. Building on interdisciplinary knowledge and practical experience from computer science, moral psychology, and pedagogy, we propose a functional way to provide this training.

----

## [794] A Logic-based Explanation Generation Framework for Classical and Hybrid Planning Problems (Extended Abstract)

**Authors**: *Stylianos Loukas Vasileiou, William Yeoh, Son Tran, Ashwin Kumar, Michael Cashmore, Daniele Magazzeni*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/795](https://doi.org/10.24963/ijcai.2023/795)

**Abstract**:

In human-aware planning systems, a planning agent might need to explain its plan to a human user when that plan appears to be non-feasible or sub-optimal. A popular approach, called model reconciliation, has been proposed as a way to bring the model of the human user closer to the agent's model. In this paper, we approach the model reconciliation problem from a different perspective, that of knowledge representation and reasoning, and demonstrate that our approach can be applied not only to classical planning problems but also hybrid systems planning problems with durative actions and events/processes.

----

## [795] Reinforcement Learning from Optimization Proxy for Ride-Hailing Vehicle Relocation (Extended Abstract)

**Authors**: *Enpeng Yuan, Wenbo Chen, Pascal Van Hentenryck*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/796](https://doi.org/10.24963/ijcai.2023/796)

**Abstract**:

Idle vehicle relocation is crucial for addressing demand-supply imbalance that frequently arises in the ride-hailing system. Current mainstream methodologies - optimization and reinforcement learning - suffer from obvious computational drawbacks. Optimization models need to be solved in real-time and often trade off model fidelity (hence quality of solutions) for computational efficiency. Reinforcement learning is expensive to train and often struggles to achieve coordination among a large fleet. This paper designs a hybrid approach that leverages the strengths of the two while overcoming their drawbacks. Specifically, it trains an optimization proxy, i.e., a machine-learning model that approximates an optimization model, and refines the proxy with reinforcement learning. This Reinforcement Learning from Optimization Proxy (RLOP) approach is efficient to train and deploy, and achieves better results than RL or optimization alone. Numerical experiments on the New York City dataset show that the RLOP approach reduces both the relocation costs and computation time significantly compared to the optimization model, while pure reinforcement learning fails to converge due to computational complexity.

----

## [796] Unsupervised and Few-Shot Parsing from Pretrained Language Models (Extended Abstract)

**Authors**: *Zhiyuan Zeng, Deyi Xiong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/797](https://doi.org/10.24963/ijcai.2023/797)

**Abstract**:

This paper proposes two Unsupervised constituent Parsing models (UPOA and UPIO) that calculate inside and outside association scores solely based on the self-attention weight matrix learned in a pretrained language model. The proposed unsupervised parsing models are further extended to few-shot parsing models (FPOA, FPIO) that use a few annotated trees to fine-tune the linear projection matrices in self-attention. Experiments on PTB and SPRML show that both unsupervised and few-shot parsing methods are better than or comparable to the previous methods.

----

## [797] Simplified Risk-aware Decision Making with Belief-dependent Rewards in Partially Observable Domains (Extended Abstract)

**Authors**: *Andrey Zhitnikov, Vadim Indelman*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/798](https://doi.org/10.24963/ijcai.2023/798)

**Abstract**:

It is a long-standing objective to ease the computation burden incurred by the decision-making problem under partial observability. Identifying the sensitivity to simplification of various components of the original problem has tremendous ramifications. Yet, algorithms for decision-making under uncertainty usually lean on approximations or heuristics without quantifying their effect. Therefore, challenging scenarios could severely impair the performance of such methods. In this paper, we extend the decision-making mechanism to the whole by removing standard approximations and considering all previously suppressed stochastic sources of variability. On top of this extension,  we scrutinize the distribution of the return. We begin from a return given a single candidate policy and continue to the pair of returns given a corresponding pair of candidate policies. Furthermore, we present novel stochastic bounds on the return and novel tools, Probabilistic Loss (PLoss) and its online accessible counterpart (PbLoss), to characterize the effect of a simplification.

----

## [798] Artificial Intelligence, Bias, and Ethics

**Authors**: *Aylin Caliskan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/799](https://doi.org/10.24963/ijcai.2023/799)

**Abstract**:

Although ChatGPT attempts to mitigate bias, when instructed to translate the gender-neutral Turkish sentences “O bir doktor. O bir hemşire” to English, the outcome is biased: “He is a doctor. She is a nurse.” In 2016, we have demonstrated that language representations trained via unsupervised learning automatically embed implicit biases documented in social cognition through the statistical regularities in language corpora. Evaluating embedding associations in language, vision, and multi-modal language-vision models reveals that large-scale sociocultural data is a source of implicit human biases regarding gender, race or ethnicity, skin color, ability, age, sexuality, religion, social class, and intersectional associations. The study of gender bias in language, vision, language-vision, and generative AI has highlighted the sexualization of women and girls in AI, while easily accessible generative AI models such as text-to-image generators amplify bias at scale. As AI increasingly automates tasks that determine life’s outcomes and opportunities, the ethics of AI bias has significant implications for human cognition, society, justice, and the future of AI. Thus, it is necessary to advance our understanding of the depth, prevalence, and complexities of bias in AI to mitigate it both in machines and society.

----

## [799] Towards Formal Verification of Neuro-symbolic Multi-agent Systems

**Authors**: *Panagiotis Kouvaros*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/800](https://doi.org/10.24963/ijcai.2023/800)

**Abstract**:

This paper outlines some of the key methods we developed towards the formal verification of multi- agent systems, covering both symbolic and connectionist systems. It discusses logic-based methods for the verification of unbounded multi-agent systems (i.e., systems composed of an arbitrary number of homogeneous agents, e.g., robot swarms), optimisation approaches for establishing the robustness of neural network models, and methods for analysing properties of neuro-symbolic multi-agent systems.

----



[Go to the previous page](IJCAI-2023-list03.md)

[Go to the next page](IJCAI-2023-list05.md)

[Go to the catalog section](README.md)