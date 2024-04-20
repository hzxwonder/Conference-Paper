## [0] Learning Dissemination Strategies for External Sources in Opinion Dynamic Models with Cognitive Biases

**Authors**: *Abdullah Al Maruf, Luyao Niu, Bhaskar Ramasubramanian, Andrew Clark, Radha Poovendran*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/1](https://doi.org/10.24963/ijcai.2023/1)

**Abstract**:

The opinions of members of a population are influenced by opinions of their peers, their own predispositions, and information from external sources via one or more information channels (e.g., news, social media). Due to individual cognitive biases, the perceptual impact of and importance assigned by agents to information on each channel can be different. In this paper, we propose a model of opinion evolution that uses prospect theory to represent perception of information from the external source along each channel. Our prospect-theoretic model reflects traits observed in humans such as loss aversion, assigning inflated (deflated) values to low (high) probability events, and evaluating outcomes relative to an individually known reference point. We consider the problem of determining information dissemination strategies for the external source to adopt in order to drive opinions of individuals towards a desired value. However, computing a strategy faces a challenge that agents' initial predispositions and functions characterizing their perceptions of information disseminated might be unknown. We overcome this challenge by using Gaussian process learning to estimate these unknown parameters. When the external source sends information over multiple channels, the problem of jointly selecting optimal dissemination strategies is in general, combinatorial. We prove that this problem is submodular, and design near-optimal dissemination algorithms. We evaluate our model on three different widely used large graphs that represent real-world social interactions. Our results indicate that the external source can effectively drive opinions towards a desired value when using prospect-theory based dissemination strategies.

----

## [1] Artificial Agents Inspired by Human Motivation Psychology for Teamwork in Hazardous Environments

**Authors**: *Anupama Arukgoda, Erandi Lakshika, Michael Barlow, Kasun Gunawardana*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/2](https://doi.org/10.24963/ijcai.2023/2)

**Abstract**:

Multi-agent literature explores personifying artificial agents with personality, emotions or cognitive biases to produce “typical”, believable agents. In
this study, we demonstrate the potential of endowing artificial agents with a motivation, using human implicit motivation psychology theory that introduces 3 motive profiles - power, achievement and affiliation, to create diverse, risk-aware agents. We first devise a framework to model these motivated agents (or agents with any inherent behavior), that can activate different strategies depending on the circumstances. We conduct experiments on a fire-fighting task domain, evaluate how motivated teams perform, and draw conclusions on appropriate team compositions to be deployed in environments with different risk levels. Our framework generates predictable agents as their resulting behaviors align with the inherent characteristics of their motives. We find that motivational diversity within teams is beneficial in dynamic collaborative environments, especially as the task risk level increases. Furthermore, we observed that the best composition in terms of the performance metrics used to evaluate team compositions, does not remain the same as the collaboration level required to achieve goals changes. These results have implications for future designs of risk-aware autonomous teams and Human-AI teams, as they highlight the prospects of creating better artificial teammates and performance gains that could be achieved through anthropomorphized motivated agents.

----

## [2] Proportionally Fair Online Allocation of Public Goods with Predictions

**Authors**: *Siddhartha Banerjee, Vasilis Gkatzelis, Safwan Hossain, Billy Jin, Evi Micha, Nisarg Shah*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/3](https://doi.org/10.24963/ijcai.2023/3)

**Abstract**:

We design online algorithms for fair allocation of public goods to a set of N agents over a sequence of T rounds and focus on improving their performance using predictions. In the basic model, a public good arrives in each round, and every agent reveals their value for it upon arrival. The algorithm must irrevocably decide the investment in this good without exceeding a total budget of B across all rounds. The algorithm can utilize (potentially noisy) predictions of each agent’s total value for all remaining goods. The algorithm's performance is measured using a proportional fairness objective, which informally demands that every group of agents be rewarded proportional to its size and the cohesiveness of its preferences. We show that no algorithm can achieve better than Θ(T/B) proportional fairness without predictions. With reasonably accurate predictions, the situation improves significantly, and Θ(log(T/B)) proportional fairness is achieved. We also extend our results to a general setting wherein a batch of L public goods arrive in each round and O(log(min(N,L)T/B)) proportional fairness is achieved. Our exact bounds are parameterized as a function of the prediction error, with performance degrading gracefully with increasing errors.

----

## [3] On the Role of Memory in Robust Opinion Dynamics

**Authors**: *Luca Becchetti, Andrea Clementi, Amos Korman, Francesco Pasquale, Luca Trevisan, Robin Vacus*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/4](https://doi.org/10.24963/ijcai.2023/4)

**Abstract**:

We investigate opinion dynamics in a fully-connected system, consisting of n agents, where one of the opinions, called correct, represents a piece of information to disseminate. 
One source agent initially holds the correct opinion and remains with this opinion throughout the execution. The goal of the remaining agents is to quickly agree on this correct opinion. At each round, one agent chosen uniformly at random is activated: unless it is the source, the agent pulls the opinions of l random agents and then updates its opinion according to some rule. 
We consider a restricted setting, in which agents have no memory and they only revise their opinions on the basis of those of the agents they currently sample. 
This setting encompasses very popular opinion dynamics, such as the voter model and best-of-k majority rules. 

Qualitatively speaking, we show that lack of memory prevents efficient  convergence. Specifically, we prove that any dynamics requires Omega(n^2) expected time, even under a strong version of the model in which activated agents have complete access to the current configuration of the entire system, i.e., the case l=n. Conversely, we prove that the simple voter model (in which l=1) correctly solves the problem, while almost matching the aforementioned lower bound. 

These results suggest that, in contrast to symmetric consensus problems (that do not involve a notion of correct opinion), fast convergence on the correct opinion using stochastic opinion dynamics may require the use of memory.

----

## [4] On a Voter Model with Context-Dependent Opinion Adoption

**Authors**: *Luca Becchetti, Vincenzo Bonifaci, Emilio Cruciani, Francesco Pasquale*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/5](https://doi.org/10.24963/ijcai.2023/5)

**Abstract**:

Opinion diffusion is a crucial phenomenon in social networks, often underlying the way in which a collection of agents develops a consensus on relevant decisions.  Voter models are well-known theoretical models to study opinion spreading in social networks and structured populations. Their simplest version assumes that an updating agent will adopt the opinion of a neighboring agent chosen at random. These models allow us to study, for example, the probability that a certain opinion will fixate into a consensus opinion, as well as the expected time it takes for a consensus opinion to emerge. 

Standard voter models are oblivious to the opinions held by the agents involved in the opinion adoption process. We propose and study a context-dependent opinion spreading process on an arbitrary social graph, in which the probability that an agent abandons opinion a in favor of opinion b depends on both a and b. We discuss the relations of the model with existing voter models and then derive theoretical results for both the fixation probability and the expected consensus time for two opinions, for both the synchronous and the asynchronous update models.

----

## [5] Scalable Verification of Strategy Logic through Three-Valued Abstraction

**Authors**: *Francesco Belardinelli, Angelo Ferrando, Wojciech Jamroga, Vadim Malvone, Aniello Murano*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/6](https://doi.org/10.24963/ijcai.2023/6)

**Abstract**:

The model checking problem for multi-agent systems against Strategy Logic specifications is known to be non-elementary. On this logic several fragments have been defined to tackle this issue but at the expense of expressiveness. In this paper, we propose a three-valued semantics for Strategy Logic upon which we define an abstraction method. We show that the latter semantics is an approximation of the classic two-valued one for Strategy Logic. Furthermore, we extend MCMAS, an open-source model checker for multi-agent specifications, to incorporate our abstraction method and present some promising experimental results.

----

## [6] Explainable Multi-Agent Reinforcement Learning for Temporal Queries

**Authors**: *Kayla Boggess, Sarit Kraus, Lu Feng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/7](https://doi.org/10.24963/ijcai.2023/7)

**Abstract**:

As multi-agent reinforcement learning (MARL) systems are increasingly deployed throughout society, it is imperative yet challenging for users to understand the emergent behaviors of MARL agents in complex environments. This work presents an approach for generating policy-level contrastive explanations for MARL to answer a temporal user query, which specifies a sequence of tasks completed by agents with possible cooperation. The proposed approach encodes the temporal query as a PCTL* logic formula and checks if the query is feasible under a given MARL policy via probabilistic model checking. Such explanations can help reconcile discrepancies between the actual and anticipated multi-agent behaviors. The proposed approach also generates correct and complete explanations to pinpoint reasons that make a user query infeasible. We have successfully applied the proposed approach to four benchmark MARL domains (up to 9 agents in one domain). Moreover, the results of a user study show that the generated explanations significantly improve user performance and satisfaction.

----

## [7] Efficient and Equitable Deployment of Mobile Vaccine Distribution Centers

**Authors**: *Da Qi Chen, Ann Li, George Z. Li, Madhav V. Marathe, Aravind Srinivasan, Leonidas Tsepenekas, Anil Vullikanti*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/8](https://doi.org/10.24963/ijcai.2023/8)

**Abstract**:

Vaccines have proven to be extremely effective in preventing the spread of COVID-19 and potentially ending the pandemic. Lack of access caused many people not getting vaccinated early, so states such as Virginia deployed mobile vaccination sites in order to distribute vaccines across the state. Here we study the problem of deciding where these facilities should be placed and moved over time in order to minimize the distance each person needs to travel in order to be vaccinated. Traditional facility location models for this problem fail to incorporate the fact that our facilities are mobile (i.e., they can move over time). To this end, we instead model vaccine distribution as the Dynamic k-Supplier problem and give the first approximation algorithms for this problem. We then run extensive simulations on real world datasets to show the efficacy of our methods. In particular, we find that natural baselines for Dynamic k-Supplier cannot take advantage of the mobility of the facilities, and perform worse than non-mobile k-Supplier algorithms.

----

## [8] Anticipatory Fictitious Play

**Authors**: *Alex Cloud, Albert Wang, Wesley Kerr*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/9](https://doi.org/10.24963/ijcai.2023/9)

**Abstract**:

Fictitious play is an algorithm for computing Nash equilibria of matrix games. Recently, machine learning variants of fictitious play have been successfully applied to complicated real-world games. This paper presents a simple modification of fictitious play which is a strict improvement over the original: it has the same theoretical worst-case convergence rate, is equally applicable in a machine learning context, and enjoys superior empirical performance. We conduct an extensive comparison of our algorithm with fictitious play, proving an optimal O(1/t) convergence rate for certain classes of games, demonstrating superior performance numerically across a variety of games, and concluding with experiments that extend these algorithms to the setting of deep multiagent reinforcement learning.

----

## [9] Safe Multi-agent Learning via Trapping Regions

**Authors**: *Aleksander Czechowski, Frans A. Oliehoek*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/10](https://doi.org/10.24963/ijcai.2023/10)

**Abstract**:

One of the main challenges of multi-agent learning lies in establishing convergence of the algorithms, as, in general, a collection of individual, self-serving agents is not guaranteed to converge with their joint policy, when learning concurrently. This is in stark contrast to most single-agent environments, and sets a prohibitive barrier for deployment in practical applications, as it induces uncertainty in long term behavior of the system. In this work, we apply the concept of trapping regions, known from qualitative theory of dynamical systems, to create safety sets in the joint strategy space for decentralized learning. We propose a binary partitioning algorithm for verification that candidate sets form trapping regions in systems with known learning dynamics, and a heuristic sampling algorithm for scenarios where learning dynamics are not known. We demonstrate the applications to a regularized version of Dirac Generative Adversarial Network,  a four-intersection traffic control scenario run in a state of the art open-source microscopic traffic simulator SUMO, and a mathematical model of economic competition.

----

## [10] Multi-Agent Intention Recognition and Progression

**Authors**: *Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, Felipe Meneguzzi, John Thangarajah*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/11](https://doi.org/10.24963/ijcai.2023/11)

**Abstract**:

For an agent in a multi-agent environment, it is often beneficial to be able to predict what other agents will do next when deciding how to act. Previous work in multi-agent intention scheduling assumes a priori knowledge of the current goals of other agents. In this paper, we present a new approach to multi-agent intention scheduling in which an agent uses online goal recognition to identify the goals currently being pursued by other agents while acting in pursuit of its own goals. We show how online goal recognition can be incorporated into an MCTS-based intention scheduler, and evaluate our approach in a range of scenarios. The results demonstrate that our approach can rapidly recognise the goals of other agents even when they are pursuing multiple goals concurrently, and has similar performance to agents which know the goals of other agents a priori.

----

## [11] Controlling Neural Style Transfer with Deep Reinforcement Learning

**Authors**: *Chengming Feng, Jing Hu, Xin Wang, Shu Hu, Bin Zhu, Xi Wu, Hongtu Zhu, Siwei Lyu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/12](https://doi.org/10.24963/ijcai.2023/12)

**Abstract**:

Controlling the degree of stylization in the Neural Style Transfer (NST) is a little tricky since it usually needs hand-engineering on hyper-parameters. In this paper, we propose the first deep Reinforcement Learning (RL) based architecture that splits one-step style transfer into a step-wise process for the NST task. Our RL-based method tends to preserve more details and structures of the content image in early steps, and synthesize more style patterns in later steps. It is a user-easily-controlled style-transfer method. Additionally, as our RL-based model performs the stylization progressively, it is lightweight and has lower computational complexity than existing one-step Deep Learning (DL) based models. Experimental results demonstrate the effectiveness and robustness of our method.

----

## [12] Cross-community Adapter Learning (CAL) to Understand the Evolving Meanings of Norm Violation

**Authors**: *Thiago Freitas dos Santos, Stephen Cranefield, Bastin Tony Roy Savarimuthu, Nardine Osman, Marco Schorlemmer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/13](https://doi.org/10.24963/ijcai.2023/13)

**Abstract**:

Cross-community learning incorporates data from different sources to leverage task-specific solutions in a target community. This approach is particularly interesting for low-resource or newly created online communities, where data formalizing interactions between agents (community members) are limited. In such scenarios, a normative system that intends to regulate online interactions faces the challenge of continuously learning the meaning of norm violation as communities' views evolve, either with changes in the understanding of what it means to violate a norm or with the emergence of new violation classes. To address this issue, we propose the Cross-community Adapter Learning (CAL) framework, which combines adapters and transformer-based models to learn the meaning of norm violations expressed as textual sentences. Additionally, we analyze the differences in the meaning of norm violations between communities, using Integrated Gradients (IG) to understand the inner workings of our model and calculate a global relevance score that indicates the relevance of words for violation detection. Results show that cross-community learning enhances CAL's performance while explaining the differences in the meaning of norm-violating behavior based on community members' feedback. We evaluate our proposal in a small set of interaction data from Wikipedia, in which the norm prohibits hate speech.

----

## [13] Learning in Multi-Memory Games Triggers Complex Dynamics Diverging from Nash Equilibrium

**Authors**: *Yuma Fujimoto, Kaito Ariu, Kenshi Abe*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/14](https://doi.org/10.24963/ijcai.2023/14)

**Abstract**:

Repeated games consider a situation where multiple agents are motivated by their independent rewards throughout learning. In general, the dynamics of their learning become complex. Especially when their rewards compete with each other like zero-sum games, the dynamics often do not converge to their optimum, i.e., the Nash equilibrium. To tackle such complexity, many studies have understood various learning algorithms as dynamical systems and discovered qualitative insights among the algorithms. However, such studies have yet to handle multi-memory games (where agents can memorize actions they played in the past and choose their actions based on their memories), even though memorization plays a pivotal role in artificial intelligence and interpersonal relationship. This study extends two major learning algorithms in games, i.e., replicator dynamics and gradient ascent, into multi-memory games. Then, we prove their dynamics are identical. Furthermore, theoretically and experimentally, we clarify that the learning dynamics diverge from the Nash equilibrium in multi-memory zero-sum games and reach heteroclinic cycles (sojourn longer around the boundary of the strategy space), providing a fundamental advance in learning in games.

----

## [14] Scalable Communication for Multi-Agent Reinforcement Learning via Transformer-Based Email Mechanism

**Authors**: *Xudong Guo, Daming Shi, Wenhui Fan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/15](https://doi.org/10.24963/ijcai.2023/15)

**Abstract**:

Communication can impressively improve cooperation in multi-agent reinforcement learning (MARL), especially for partially-observed tasks. However, existing works either broadcast the messages leading to information redundancy, or learn targeted communication by modeling all the other agents as targets, which is not scalable when the number of agents varies. In this work, to tackle the scalability problem of MARL communication for partially-observed tasks, we propose a novel framework Transformer-based Email Mechanism (TEM). The agents adopt local communication to send messages only to the ones that can be observed without modeling all the agents. Inspired by human cooperation with email forwarding, we design message chains to forward information to cooperate with the agents outside the observation range. We introduce Transformer to encode and decode the message chain to choose the next receiver selectively. Empirically, TEM outperforms the baselines on multiple cooperative MARL benchmarks. When the number of agents varies, TEM maintains superior performance without further training.

----

## [15] Beyond Strict Competition: Approximate Convergence of Multi-agent Q-Learning Dynamics

**Authors**: *Aamal Abbas Hussain, Francesco Belardinelli, Georgios Piliouras*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/16](https://doi.org/10.24963/ijcai.2023/16)

**Abstract**:

The behaviour of multi-agent learning in competitive settings is often considered under the restrictive assumption of a zero-sum game. Only under this strict requirement is the behaviour of learning well understood; beyond this, learning dynamics can often display non-convergent behaviours which prevent fixed-point analysis. Nonetheless, many relevant competitive games do not satisfy the zero-sum assumption.
Motivated by this, we study a smooth variant of Q-Learning, a popular reinforcement learning dynamics which balances the agents' tendency to maximise their payoffs with their propensity to explore the state space. We examine this dynamic in games which are `close' to network zero-sum games and find that Q-Learning converges to a neighbourhood around a unique equilibrium. The size of the neighbourhood is determined by the `distance' to the zero-sum game, as well as the exploration rates of the agents. We complement these results by providing a method whereby, given an arbitrary network game, the `nearest' network zero-sum game can be found efficiently. Importantly, our theoretical guarantees are widely applicable in different game settings, regardless of whether the dynamics ultimately reach an equilibrium, or remain non convergent.

----

## [16] Principal-Agent Boolean Games

**Authors**: *David Hyland, Julian Gutierrez, Michael J. Wooldridge*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/17](https://doi.org/10.24963/ijcai.2023/17)

**Abstract**:

We introduce and study a computational version of the principal-agent problem -- a classic problem in Economics that arises when a principal desires to contract an agent to carry out some task, but has incomplete information about the agent or their subsequent actions. The key challenge in this setting is for the principal to design a contract for the agent such that the agent's preferences are then aligned with those of the principal. We study this problem using a variation of Boolean games, where multiple players each choose valuations for Boolean variables under their control, seeking the satisfaction of a personal goal formula. In our setting, the principal can only observe some subset of these variables, and the principal chooses a contract which rewards players on the basis of the assignments they make for the variables that are observable to the principal. The principal's challenge is to design a contract so that, firstly, the principal's goal is achieved in some or all Nash equilibrium choices, and secondly, that the principal is able to verify that their goal is satisfied. In this paper, we formally define this problem and completely characterise the computational complexity of the most relevant decision problems associated with it.

----

## [17] Learning to Send Reinforcements: Coordinating Multi-Agent Dynamic Police Patrol Dispatching and Rescheduling via Reinforcement Learning

**Authors**: *Waldy Joe, Hoong Chuin Lau*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/18](https://doi.org/10.24963/ijcai.2023/18)

**Abstract**:

We address the problem of coordinating multiple agents in a dynamic police patrol scheduling via a Reinforcement Learning (RL) approach. Our approach utilizes Multi-Agent Value Function Approximation (MAVFA) with a rescheduling heuristic to learn dispatching and rescheduling policies jointly. Often, police operations are divided into multiple sectors for more effective and efficient operations. In a dynamic setting, incidents occur throughout the day across different sectors, disrupting initially-planned patrol schedules. To maximize policing effectiveness, police agents from different sectors cooperate by sending reinforcements to support one another in their incident response and even routine patrol. This poses an interesting research challenge on how to make such complex decision of dispatching and rescheduling involving multiple agents in a coordinated fashion within an operationally reasonable time. Unlike existing Multi-Agent RL (MARL) approaches which solve similar problems by either decomposing the problem or action into multiple components, our approach learns the dispatching and rescheduling policies jointly without any decomposition step. In addition, instead of directly searching over the joint action space, we incorporate an iterative best response procedure as a decentralized optimization heuristic and an explicit coordination mechanism for a scalable and coordinated decision-making. We evaluate our approach against the commonly adopted two-stage approach and conduct a series of ablation studies to ascertain the effectiveness of our proposed learning and coordination mechanisms.

----

## [18] Decentralized Anomaly Detection in Cooperative Multi-Agent Reinforcement Learning

**Authors**: *Kiarash Kazari, Ezzeldin Shereen, György Dán*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/19](https://doi.org/10.24963/ijcai.2023/19)

**Abstract**:

We consider the problem of detecting adversarial attacks against cooperative multi-agent reinforcement learning. We propose a decentralized scheme that allows agents to detect the abnormal behavior of one compromised agent. Our approach is based on a recurrent neural network (RNN) trained during cooperative learning to predict the action distribution of other agents based on local observations. The predicted distribution is used for computing a normality score for the agents, which allows the detection of the misbehavior of other agents. To explore the robustness of the proposed detection scheme, we formulate the worst-case attack against our scheme as a constrained reinforcement learning problem. We propose to compute an attack policy by optimizing the corresponding dual function using reinforcement learning. Extensive simulations on various multi-agent benchmarks show the effectiveness of the proposed detection scheme in detecting state-of-the-art attacks and in limiting the impact of undetectable attacks.

----

## [19] Synthesizing Resilient Strategies for Infinite-Horizon Objectives in Multi-Agent Systems

**Authors**: *David Klaska, Antonín Kucera, Martin Kurecka, Vít Musil, Petr Novotný, Vojtech Rehák*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/20](https://doi.org/10.24963/ijcai.2023/20)

**Abstract**:

We consider the problem of synthesizing resilient and stochastically stable strategies for systems of cooperating agents striving to minimize the expected time between consecutive visits to selected locations in a known environment. A strategy profile is resilient if it retains its functionality even if some of the agents fail, and stochastically stable if the visiting time variance is small. We design a novel specification language for objectives involving resilience and stochastic stability, and we show how to efficiently compute strategy profiles (for both autonomous and coordinated agents) optimizing these objectives. Our experiments show that our strategy synthesis algorithm can construct highly non-trivial and efficient strategy profiles for environments with general topology.

----

## [20] In Which Graph Structures Can We Efficiently Find Temporally Disjoint Paths and Walks?

**Authors**: *Pascal Kunz, Hendrik Molter, Meirav Zehavi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/21](https://doi.org/10.24963/ijcai.2023/21)

**Abstract**:

A temporal graph has an edge set that may change over discrete time steps, and a temporal path (or walk) must traverse edges that appear at increasing time steps. Accordingly, two temporal paths (or walks) are temporally disjoint if they do not visit any vertex at the same time. The study of the computational complexity of finding temporally disjoint paths or walks in temporal graphs has recently been initiated by Klobas et al.. This problem is motivated by applications in multi-agent path finding (MAPF), which include robotics, warehouse management, aircraft management, and traffic routing.

We extend Klobas et al.’s research by providing parameterized hardness results for very restricted cases, with a focus on structural parameters of the so-called underlying graph. On the positive side, we identify sufficiently simple cases where we can solve the problem efficiently. Our results reveal some surprising differences between the “path version” and the “walk version” (where vertices may be visited multiple times) of the problem, and answer several open questions posed by Klobas et al.

----

## [21] Probabilistic Planning with Prioritized Preferences over Temporal Logic Objectives

**Authors**: *Lening Li, Hazhar Rahmani, Jie Fu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/22](https://doi.org/10.24963/ijcai.2023/22)

**Abstract**:

This paper studies temporal planning in probabilistic environments, modeled as labeled Markov decision processes (MDPs), with user preferences over multiple temporal goals.  Existing works reflect such preferences as a prioritized list of goals. This paper introduces a new specification language, termed prioritized qualitative choice linear temporal logic on finite traces, which augments linear temporal logic on finite traces with prioritized conjunction and ordered disjunction from prioritized qualitative choice logic. This language allows for succinctly specifying temporal objectives with corresponding preferences accomplishing each temporal task. The finite traces that describe the system's behaviors are ranked based on their dissatisfaction scores with respect to the formula. We propose a systematic translation from the new language to a weighted deterministic finite automaton. Utilizing this computational model, we formulate and solve a problem of computing an optimal policy that minimizes the expected score of dissatisfaction given user preferences. We demonstrate the efficacy and applicability of the logic and the algorithm on several case studies with detailed analyses for each.

----

## [22] GPLight: Grouped Multi-agent Reinforcement Learning for Large-scale Traffic Signal Control

**Authors**: *Yilin Liu, Guiyang Luo, Quan Yuan, Jinglin Li, Lei Jin, Bo Chen, Rui Pan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/23](https://doi.org/10.24963/ijcai.2023/23)

**Abstract**:

The use of multi-agent reinforcement learning (MARL) methods in coordinating traffic lights (CTL) has become increasingly popular, treating each intersection as an agent. However, existing MARL approaches either treat each agent absolutely homogeneous, i.e., same network and parameter for each agent, or treat each agent completely heterogeneous, i.e., different networks and parameters for each agent. This creates a difficult balance between accuracy and complexity, especially in large-scale CTL. To address this challenge, we propose a grouped MARL method named GPLight. We first mine the similarity between agent environment considering both real-time traffic flow and static fine-grained road topology. Then we propose two loss functions to maintain a learnable and dynamic clustering, one that uses mutual information estimation for better stability, and the other that maximizes separability between groups. Finally, GPLight enforces the agents in a group to share the same network and parameters. This approach reduces complexity by promoting cooperation within the same group of agents while reflecting differences between groups to ensure accuracy. To verify the effectiveness of our method, we conduct experiments on both synthetic and real-world datasets, with up to 1,089 intersections. Compared with state-of-the-art methods, experiment results demonstrate the superiority of our proposed method, especially in large-scale CTL.

----

## [23] Deep Hierarchical Communication Graph in Multi-Agent Reinforcement Learning

**Authors**: *Zeyang Liu, Lipeng Wan, Xue Sui, Zhuoran Chen, Kewu Sun, Xuguang Lan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/24](https://doi.org/10.24963/ijcai.2023/24)

**Abstract**:

Sharing intentions is crucial for efficient cooperation in communication-enabled multi-agent reinforcement learning. Recent work applies static or undirected graphs to determine the order of interaction. However, the static graph is not general for complex cooperative tasks, and the parallel message-passing update in the undirected graph with cycles cannot guarantee convergence. To solve this problem, we propose Deep Hierarchical Communication Graph (DHCG) to learn the dependency relationships between agents based on their messages. The relationships are formulated as directed acyclic graphs (DAGs), where the selection of the proper topology is viewed as an action and trained in an end-to-end fashion. To eliminate the cycles in the graph, we apply an acyclicity constraint as intrinsic rewards and then project the graph in the admissible solution set of DAGs. As a result, DHCG removes redundant communication edges for cost improvement and guarantees convergence. To show the effectiveness of the learned graphs, we propose policy-based and value-based DHCG. Policy-based DHCG factorizes the joint policy in an auto-regressive manner, and value-based DHCG factorizes the joint value function to individual value functions and pairwise payoff functions. Empirical results show that our method improves performance across various cooperative multi-agent tasks, including Predator-Prey, Multi-Agent Coordination Challenge, and StarCraft Multi-Agent Challenge.

----

## [24] The #DNN-Verification Problem: Counting Unsafe Inputs for Deep Neural Networks

**Authors**: *Luca Marzari, Davide Corsi, Ferdinando Cicalese, Alessandro Farinelli*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/25](https://doi.org/10.24963/ijcai.2023/25)

**Abstract**:

Deep Neural Networks are increasingly adopted in critical tasks that require a high level of safety, e.g., autonomous driving.
While state-of-the-art verifiers can be employed to check whether a DNN is unsafe w.r.t. some given property (i.e., whether there is at least one unsafe input configuration), their yes/no output is not informative enough for other purposes, such as shielding, model selection, or training improvements.
In this paper, we introduce the #DNN-Verification problem, which involves counting the number of input configurations of a DNN that result in a violation of a particular safety property. We analyze the complexity of this problem and propose a novel approach that returns the exact count of violations. Due to the #P-completeness of the problem, we also propose a randomized, approximate method that provides a provable probabilistic bound of the correct count while significantly reducing computational requirements. 
We present experimental results on a set of safety-critical benchmarks that demonstrate the effectiveness of our approximate method and evaluate the tightness of the bound.

----

## [25] Discounting in Strategy Logic

**Authors**: *Munyque Mittelmann, Aniello Murano, Laurent Perrussel*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/26](https://doi.org/10.24963/ijcai.2023/26)

**Abstract**:

Discounting is an important dimension in multi-agent systems as long as we want to reason about strategies and time. It is a key aspect in economics as it captures the intuition that the far-away future is not as important as the near future. Traditional verification techniques allow to check whether there is a winning strategy for a group of agents but they do not take into account the fact that satisfying a goal sooner is different from satisfying it after a long wait. 
In this paper, we augment Strategy Logic with future discounting over a set of discounted functions D, denoted SL[D]. We consider “until” operators with discounting functions: the satisfaction value of a specification in SL[D] is a value in [0, 1], where the longer it takes to fulfill requirements, the smaller the satisfaction value is. We motivate our approach with classical examples from Game Theory and study the complexity of model-checking SL[D]-formulas.

----

## [26] Why Rumors Spread Fast in Social Networks, and How to Stop It

**Authors**: *Ahad N. Zehmakan, Charlotte Out, Sajjad Hesamipour Khelejan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/27](https://doi.org/10.24963/ijcai.2023/27)

**Abstract**:

We study a rumor spreading model where individuals are connected via a network structure. Initially, only a small subset of the individuals are spreading a rumor. Each individual who is connected to a spreader, starts spreading the rumor with some probability as a function of their trust in the spreader, quantified by the Jaccard similarity index. Furthermore, the probability that a spreader diffuses the rumor decreases over time until they fully lose their interest and stop spreading.

We focus on determining the graph parameters which govern the magnitude and pace that the rumor spreads in this model. We prove that for the rumor to spread to a sizable fraction of the individuals, the network needs to enjoy ``strong'' expansion properties and most nodes should be in ``well-connected'' communities. Both of these characteristics are, arguably, present in real-world social networks up to a certain degree, shedding light on the driving force behind the extremely fast spread of rumors in social networks.

Furthermore, we formulate a large range of countermeasures to cease the spread of a rumor. We introduce four fundamental criteria which a countermeasure ideally should possess. We evaluate all the proposed countermeasures by conducting experiments on real-world social networks such as Facebook and Twitter. We conclude that our novel decentralized countermeasures (which are executed by the individuals) generally outperform the previously studied centralized ones (which need to be imposed by a third entity such as the government).

----

## [27] Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding

**Authors**: *Keisuke Okumura*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/28](https://doi.org/10.24963/ijcai.2023/28)

**Abstract**:

This study extends the recently-developed LaCAM algorithm for multi-agent pathfinding (MAPF). LaCAM is a sub-optimal search-based algorithm that uses lazy successor generation to dramatically reduce the planning effort. We present two enhancements. First, we propose its anytime version, called LaCAM*, which eventually converges to optima, provided that solution costs are accumulated transition costs. Second, we improve the successor generation to quickly obtain initial solutions. Exhaustive experiments demonstrate their utility. For instance, LaCAM* sub-optimally solved 99% of the instances retrieved from the MAPF benchmark, where the number of agents varied up to a thousand, within ten seconds on a standard desktop PC, while ensuring eventual convergence to optima; developing a new horizon of MAPF algorithms.

----

## [28] Quick Multi-Robot Motion Planning by Combining Sampling and Search

**Authors**: *Keisuke Okumura, Xavier Défago*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/29](https://doi.org/10.24963/ijcai.2023/29)

**Abstract**:

We propose a novel algorithm to solve multi-robot motion planning (MRMP) rapidly, called Simultaneous Sampling-and-Search Planning (SSSP). Conventional MRMP studies mostly take the form of two-phase planning that constructs roadmaps and then finds inter-robot collision-free paths on those roadmaps. In contrast, SSSP simultaneously performs roadmap construction and collision-free pathfinding. This is realized by uniting techniques of single-robot sampling-based motion planning and search techniques of multi-agent pathfinding on discretized spaces. Doing so builds the small search space, leading to quick MRMP. SSSP ensures finding a solution eventually if exists. Our empirical evaluations in various scenarios demonstrate that SSSP significantly outperforms standard approaches to MRMP, i.e., solving more problem instances much faster. We also applied SSSP to planning for 32 ground robots in a dense situation.

----

## [29] Asynchronous Communication Aware Multi-Agent Task Allocation

**Authors**: *Ben Rachmut, Sofia Amador Nelke, Roie Zivan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/30](https://doi.org/10.24963/ijcai.2023/30)

**Abstract**:

Multi-agent task allocation in physical environments with spatial and temporal constraints, are hard problems that are relevant in many realistic applications. A task allocation algorithm based on Fisher market clearing (FMC_TA), that can be performed either centrally or distributively, has been shown to produce high quality allocations in comparison to both centralized and distributed state of the art incomplete optimization algorithms. However, the algorithm is synchronous and therefore depends on perfect communication between agents.

We propose FMC_ATA, an asynchronous version of FMC_TA, which is robust to message latency and message loss. In contrast to the former version of the algorithm, FMC_ATA allows agents to identify dynamic events and initiate the generation of an updated allocation. Thus, it is more compatible for dynamic environments. We further investigate the conditions in which the distributed version of the algorithm is preferred over the centralized version. Our results indicate that the proposed asynchronous distributed algorithm produces consistent results even when the communication level is extremely poor.

----

## [30] Towards a Better Understanding of Learning with Multiagent Teams

**Authors**: *David Radke, Kate Larson, Tim Brecht, Kyle Tilbury*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/31](https://doi.org/10.24963/ijcai.2023/31)

**Abstract**:

While it has long been recognized that a team of individual learning agents can be greater than the sum of its parts, recent work has shown that larger teams are not necessarily more effective than smaller ones. In this paper, we study why and under which conditions certain team structures promote effective learning for a population of individual learning agents. We show that, depending on the environment, some team structures help agents learn to specialize into specific roles, resulting in more favorable global results. However, large teams create credit assignment challenges that reduce coordination, leading to large teams performing poorly compared to smaller ones. We support our conclusions with both theoretical analysis and empirical results.

----

## [31] Multi-Agent Systems with Quantitative Satisficing Goals

**Authors**: *Senthil Rajasekaran, Suguman Bansal, Moshe Y. Vardi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/32](https://doi.org/10.24963/ijcai.2023/32)

**Abstract**:

In the study of reactive systems, qualitative properties are usually easier to model and analyze than quantitative properties. This is especially true in systems where mutually beneficial cooperation between agents is possible, such as multi-agent systems. The large number of possible payoffs available to agents in reactive systems with quantitative properties means that there are many scenarios in which agents deviate from mutually beneficial outcomes in order to gain negligible payoff improvements. This behavior often leads to less desirable outcomes for all agents involved. For this reason we study satisficing goals, derived from a decision-making approach aimed at meeting a good-enough outcome instead of pure optimization. By considering satisficing goals, we are able to employ efficient automata-based algorithms to find pure-strategy Nash equilibria. We then show that these algorithms extend to scenarios in which agents have multiple thresholds, providing an approximation of optimization while still retaining the possibility of mutually beneficial cooperation and efficient automata-based algorithms. Finally, we demonstrate a one-way correspondence between the existence of epsilon-equilibria and the existence of equilibria in games where agents have multiple thresholds.

----

## [32] Norm Deviation in Multiagent Systems: A Foundation for Responsible Autonomy

**Authors**: *Amika M. Singh, Munindar P. Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/33](https://doi.org/10.24963/ijcai.2023/33)

**Abstract**:

The power of norms in both human societies and sociotechnical systems arises from the facts that (1) societal norms, including laws and policies, characterize acceptable behavior in high-level terms and (2) they are not hard controls and can be deviated from. Thus, the design of responsibly autonomous agents faces an essential tension: these agents must both (1) respect applicable norms and (2) deviate from those norms when blindly following them may lead to diminished outcomes.

We propose a conceptual foundation for norm deviation. As a guiding framework, we adopt Habermas's theory of communicative action comprising objective, subjective, and practical validity claims regarding the suitability of deviation. 
Our analysis thus goes beyond previous studies of norm deviation and yields reasoning guidelines uniting norms and values by which to develop responsible agents.

----

## [33] CVTP3D: Cross-view Trajectory Prediction Using Shared 3D Queries for Autonomous Driving

**Authors**: *Zijian Song, Huikun Bi, Ruisi Zhang, Tianlu Mao, Zhaoqi Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/34](https://doi.org/10.24963/ijcai.2023/34)

**Abstract**:

Trajectory prediction with uncertainty is a critical and challenging task for autonomous driving. Nowadays, we can easily access sensor data represented in multiple views. However, cross-view consistency has not been evaluated by the existing models, which might lead to divergences between the multimodal predictions from different views. It is not practical and effective when the network does not comprehend the 3D scene, which could cause the downstream module in a dilemma. Instead, we predicts multimodal trajectories while maintaining cross-view consistency. We presented a cross-view trajectory prediction method using shared 3D Queries (XVTP3D). We employ a set of 3D queries shared across views to generate multi-goals that are cross-view consistent. We also proposed a random mask method and coarse-to-fine cross-attention to capture robust cross-view features. As far as we know, this is the first work that introduces the outstanding top-down paradigm in BEV detection field to a trajectory prediction problem. The results of experiments on two publicly available datasets show that XVTP3D achieved state-of-the-art performance with consistent cross-view predictions.

----

## [34] Optimal Anytime Coalition Structure Generation Utilizing Compact Solution Space Representation

**Authors**: *Redha Taguelmimt, Samir Aknine, Djamila Boukredera, Narayan Changder, Tuomas Sandholm*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/35](https://doi.org/10.24963/ijcai.2023/35)

**Abstract**:

Coalition formation is a central approach for multiagent coordination. A crucial part of coalition formation that is extensively studied in AI is coalition structure generation: partitioning agents into coalitions to maximize overall value. 
In this paper, we propose a novel method for coalition structure generation by introducing a compact and efficient representation of coalition structures. Our representation partitions the solution space into smaller, more manageable subspaces that gather structures containing coalitions of specific sizes. Our proposed method combines two new algorithms, one which leverages our compact representation and a branch-and-bound technique to generate optimal coalition structures, and another that utilizes a preprocessing phase to identify the most promising sets of coalitions to evaluate. Additionally, we show how parts of the solution space can be gathered into groups to avoid their redundant evaluation and we investigate the computational gain that is achieved by avoiding that redundant processing. Through this approach, our algorithm is able to prune the solution space more efficiently. Our results show that the proposed algorithm is superior to prior state-of-the-art methods in generating optimal coalition structures under several value distributions.

----

## [35] Modeling Moral Choices in Social Dilemmas with Multi-Agent Reinforcement Learning

**Authors**: *Elizaveta Tennant, Stephen Hailes, Mirco Musolesi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/36](https://doi.org/10.24963/ijcai.2023/36)

**Abstract**:

Practical uses of Artificial Intelligence (AI) in the real world have demonstrated the importance of embedding moral choices into intelligent agents. They have also highlighted that defining top-down ethical constraints on AI according to any one type of morality is extremely challenging and can pose risks. A bottom-up learning approach may be more appropriate for studying and developing ethical behavior in AI agents. In particular, we believe that an interesting and insightful starting point is the analysis of emergent behavior of Reinforcement Learning (RL) agents that act according to a predefined set of moral rewards in social dilemmas.

In this work, we present a systematic analysis of the choices made by intrinsically-motivated RL agents whose rewards are based on moral theories. We aim to design reward structures that are simplified yet representative of a set of key ethical systems. Therefore, we first define moral reward functions that distinguish between consequence- and norm-based agents, between morality based on societal norms or internal virtues, and between single- and mixed-virtue (e.g., multi-objective) methodologies. Then, we evaluate our approach by modeling repeated dyadic interactions between learning moral agents in three iterated social dilemma games (Prisoner's Dilemma, Volunteer's Dilemma and Stag Hunt). We analyze the impact of different types of morality on the emergence of cooperation, defection or exploitation, and the corresponding social outcomes. Finally, we discuss the implications of these findings for the development of moral agents in artificial and mixed human-AI societies.

----

## [36] Exploration via Joint Policy Diversity for Sparse-Reward Multi-Agent Tasks

**Authors**: *Pei Xu, Junge Zhang, Kaiqi Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/37](https://doi.org/10.24963/ijcai.2023/37)

**Abstract**:

Exploration under sparse rewards is a key challenge for multi-agent reinforcement learning problems. Previous works argue that complex dynamics between agents and the huge exploration space in MARL scenarios amplify the vulnerability of classical count-based exploration methods when combined with agents parameterized by neural networks, resulting in inefficient exploration. In this paper, we show that introducing constrained joint policy diversity into a classical count-based method can significantly improve exploration when agents are parameterized by neural networks. Specifically, we propose a joint policy diversity to measure the difference between current joint policy and previous joint policies, and then use a filtering-based exploration constraint to further refine the joint policy diversity. Under the sparse-reward setting, we show that the proposed method significantly outperforms the state-of-the-art methods in the multiple-particle environment, the Google Research Football, and StarCraft II micromanagement tasks. To the best of our knowledge, on the hard 3s_vs_5z task which needs non-trivial strategies to defeat enemies, our method is the first to learn winning strategies without domain knowledge under the sparse-reward setting.

----

## [37] Measuring Acoustics with Collaborative Multiple Agents

**Authors**: *Yinfeng Yu, Changan Chen, Lele Cao, Fangkai Yang, Fuchun Sun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/38](https://doi.org/10.24963/ijcai.2023/38)

**Abstract**:

As humans, we hear sound every second of our life. The sound we hear is often affected by the acoustics of the environment surrounding us. For example, a spacious hall leads to more reverberation. Room Impulse Responses (RIR) are commonly used to characterize environment acoustics as a function of the scene geometry, materials, and source/receiver locations. Traditionally, RIRs are measured by setting up a loudspeaker and microphone in the environment for all source/receiver locations, which is time-consuming and inefficient. We propose to let two robots measure the environment's acoustics by actively moving and emitting/receiving sweep signals. We also devise a collaborative multi-agent policy where these two robots are trained to explore the environment's acoustics while being rewarded for wide exploration and accurate prediction. We show that the robots learn to collaborate and move to explore environment acoustics while minimizing the prediction error. To the best of our knowledge, we present the very first problem formulation and solution to the task of collaborative environment acoustics measurements with multiple agents.

----

## [38] Dynamic Belief for Decentralized Multi-Agent Cooperative Learning

**Authors**: *Yunpeng Zhai, Peixi Peng, Chen Su, Yonghong Tian*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/39](https://doi.org/10.24963/ijcai.2023/39)

**Abstract**:

Decentralized multi-agent cooperative learning is a practical task due to the partially observed setting both in training and execution. Every agent learns to cooperate without access to the observations and policies of others. However, the decentralized training of multi-agent is of great difficulty due to non-stationarity, especially when other agents' policies are also in learning during training. To overcome this, we propose to learn a dynamic policy belief for each agent to predict the current policies of other agents and accordingly condition the policy of its own. To quickly adapt to the development of others' policies, we introduce a historical context to learn the belief inference according to a few recent action histories of other agents and a latent variational inference to model their policies by a learned distribution. We evaluate our method on the StarCraft II micro management task (SMAC) and demonstrate its superior performance in the decentralized training settings and comparable results with the state-of-the-art CTDE methods.

----

## [39] Inducing Stackelberg Equilibrium through Spatio-Temporal Sequential Decision-Making in Multi-Agent Reinforcement Learning

**Authors**: *Bin Zhang, Lijuan Li, Zhiwei Xu, Dapeng Li, Guoliang Fan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/40](https://doi.org/10.24963/ijcai.2023/40)

**Abstract**:

In multi-agent reinforcement learning (MARL), self-interested agents attempt to establish equilibrium and achieve coordination depending on game structure. However, existing MARL approaches are mostly bound by the simultaneous actions of all agents in the Markov game (MG) framework, and few works consider the formation of equilibrium strategies via asynchronous action coordination. In view of the advantages of Stackelberg equilibrium (SE) over Nash equilibrium, we construct a spatio-temporal sequential decision-making structure derived from the MG and propose an N-level policy model based on a conditional hypernetwork shared by all agents. This approach allows for asymmetric training with symmetric execution, with each agent responding optimally conditioned on the decisions made by superior agents. Agents can learn heterogeneous SE policies while still maintaining parameter sharing, which leads to reduced cost for learning and storage and enhanced scalability as the number of agents increases. Experiments demonstrate that our method effectively converges to the SE policies in repeated matrix game scenarios, and performs admirably in immensely complex settings including cooperative tasks and mixed tasks.

----

## [40] Quantifying Harm

**Authors**: *Sander Beckers, Hana Chockler, Joseph Y. Halpern*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/41](https://doi.org/10.24963/ijcai.2023/41)

**Abstract**:

In earlier work we defined a qualitative notion of harm: either harm is caused, or it is not. For practical applications, we often need to quantify harm; for example, we may want to choose the least harmful of a set of possible interventions. We first present a quantitative definition of harm in a deterministic context involving a single individual, then we consider the issues involved in dealing with uncertainty regarding the context and going from a notion of harm for a single individual to a notion of "societal harm", which involves aggregating the harm to individuals.  We show that the "obvious" way of doing this (just taking the expected harm for an individual and then summing the expected harm over all individuals) can lead to counterintuitive or inappropriate answers, and discuss alternatives, drawing on work from the decision-theory literature.

----

## [41] Analyzing Intentional Behavior in Autonomous Agents under Uncertainty

**Authors**: *Filip Cano Córdoba, Samuel Judson, Timos Antonopoulos, Katrine Bjørner, Nicholas Shoemaker, Scott J. Shapiro, Ruzica Piskac, Bettina Könighofer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/42](https://doi.org/10.24963/ijcai.2023/42)

**Abstract**:

Principled accountability for autonomous decision-making in uncertain environments requires distinguishing intentional outcomes from negligent designs from actual accidents. We propose analyzing the behavior of autonomous agents through a quantitative measure of the evidence of intentional behavior. We model an uncertain environment as a Markov Decision Process (MDP). For a given scenario, we rely on probabilistic model checking to compute the ability of the agent to influence reaching a certain event. We call this the scope of agency. We say that there is evidence of intentional behavior if the scope of agency is high and the decisions of the agent are close to being optimal for reaching the event.  Our method applies counterfactual reasoning to automatically generate relevant scenarios that can be analyzed to increase the confidence of our assessment. In a case study, we show how our method can distinguish between 'intentional' and 'accidental' traffic collisions.

----

## [42] Choose your Data Wisely: A Framework for Semantic Counterfactuals

**Authors**: *Edmund Dervakos, Konstantinos Thomas, Giorgos Filandrianos, Giorgos Stamou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/43](https://doi.org/10.24963/ijcai.2023/43)

**Abstract**:

Counterfactual explanations have been argued to be one of the most intuitive forms of explanation. They are typically defined as a minimal set of edits on a given data sample that, when applied, changes the output of a model on that sample. However, a minimal set of edits is not always clear and understandable to an end-user, as it could constitute an adversarial example (which is indistinguishable from the original data sample to an end-user). Instead, there are recent ideas that the notion of minimality in the context of counterfactuals should refer to the semantics of the data sample, and not to the feature space. In this work, we build on these ideas, and propose a framework that provides counterfactual explanations in terms of knowledge graphs. We provide an algorithm for computing such explanations (given some assumptions about the underlying knowledge), and quantitatively evaluate the framework with a user study.

----

## [43] Group Fairness in Set Packing Problems

**Authors**: *Sharmila Duppala, Juan Luque, John P. Dickerson, Aravind Srinivasan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/44](https://doi.org/10.24963/ijcai.2023/44)

**Abstract**:

Kidney exchange programs (KEPs) typically seek to match incompatible patient-donor pairs based on a utilitarian objective where the number or overall quality of transplants is maximized---implicitly penalizing certain classes of difficult to match (e.g., highly-sensitized) patients. Prioritizing the welfare of highly-sensitized (hard-to-match) patients has been studied as a natural \textit{fairness} criterion. 
We formulate the KEP problem as $k$-set packing with a probabilistic group fairness notion of proportionality fairness---namely, fair $k$-set packing (\f{}). In this work we propose algorithms that take arbitrary proportionality vectors (i.e., policy-informed demands of how to prioritize different groups) and return a probabilistically fair solution with provable guarantees. Our main contributions are randomized algorithms as well as hardness results for \f{} variants. Additionally, the tools we introduce serve to audit the price of fairness involved in prioritizing different groups in realistic KEPs and other $k$-set packing applications. We conclude with experiments on synthetic and realistic kidney exchange \textsc{FairSP} instances.

----

## [44] Incentivizing Recourse through Auditing in Strategic Classification

**Authors**: *Andrew Estornell, Yatong Chen, Sanmay Das, Yang Liu, Yevgeniy Vorobeychik*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/45](https://doi.org/10.24963/ijcai.2023/45)

**Abstract**:

The increasing automation of high-stakes decisions with direct impact on the lives and well-being of individuals raises a number of important considerations. Prominent among these is strategic behavior by individuals hoping to achieve a more desirable outcome. Two forms of such behavior are commonly studied: 1) misreporting of individual attributes, and 2) recourse, or actions that truly change such attributes. The former involves deception, and is inherently undesirable, whereas the latter may well be a desirable goal insofar as it changes true individual qualification. We study misreporting and recourse as strategic choices by individuals within a unified framework. In particular, we propose auditing as a means to incentivize recourse actions over attribute manipulation, and characterize optimal audit policies for two types of principals, utility-maximizing and recourse-maximizing. Additionally, we consider subsidies as an incentive for recourse over manipulation, and show that even a utility-maximizing principal would be willing to devote a considerable amount of audit budget to providing such subsidies. Finally, we consider the problem of optimizing fines for failed audits, and bound the total cost incurred by the population as a result of audits.

----

## [45] Sampling Ex-Post Group-Fair Rankings

**Authors**: *Sruthi Gorantla, Amit Deshpande, Anand Louis*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/46](https://doi.org/10.24963/ijcai.2023/46)

**Abstract**:

Randomized rankings have been of recent interest to achieve ex-ante fairer exposure and better robustness than deterministic rankings. We propose a set of natural axioms for randomized group-fair rankings and prove that there exists a unique distribution D that satisfies our axioms and is supported only over ex-post group-fair rankings, i.e., rankings that satisfy given lower and upper bounds on group-wise representation in the top-k ranks. Our problem formulation works even when there is implicit bias, incomplete relevance information, or only ordinal ranking is available instead of relevance scores or utility values. 

We propose two algorithms to sample a random group-fair ranking from the distribution D mentioned above. Our first dynamic programming-based algorithm samples ex-post group-fair rankings uniformly at random in time O(k^2 ell), where "ell" is the number of groups. Our second random walk-based algorithm samples ex-post group-fair rankings from a distribution epsilon-close to D in total variation distance and has expected running time O*(k^2 ell^2), when there is a sufficient gap between the given upper and lower bounds on the group-wise representation. The former does exact sampling, but the latter runs significantly faster on real-world data sets for larger values of k. We give empirical evidence that our algorithms compare favorably against recent baselines for fairness and ranking utility on real-world data sets.

----

## [46] Moral Planning Agents with LTL Values

**Authors**: *Umberto Grandi, Emiliano Lorini, Timothy Parker*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/47](https://doi.org/10.24963/ijcai.2023/47)

**Abstract**:

A moral planning agent (MPA) seeks to compare two plans or compute an optimal plan in an interactive setting with other agents, where relative ideality and optimality of plans are defined with respect to a prioritized value base. We model MPAs whose values are expressed by formulas of linear temporal logic (LTL) and define comparison for both joint plans and individual plans. We introduce different evaluation criteria for individual plans including an optimistic (risk-seeking) criterion, a pessimistic (risk-averse) one, and two criteria based on the use of anticipated responsibility. We provide complexity results for a variety of MPA problems.

----

## [47] Advancing Post-Hoc Case-Based Explanation with Feature Highlighting

**Authors**: *Eoin M. Kenny, Eoin Delaney, Mark T. Keane*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/48](https://doi.org/10.24963/ijcai.2023/48)

**Abstract**:

Explainable AI (XAI) has been proposed as a valuable tool to assist in downstream tasks involving human-AI collaboration. Perhaps the most psychologically valid XAI techniques are case-based approaches which display "whole" exemplars to explain the predictions of black-box AI systems. However, for such post-hoc XAI methods dealing with images, there has been no attempt to improve their scope by using multiple clear feature "parts" of the images to explain the predictions while linking back to relevant cases in the training data, thus allowing for more comprehensive explanations that are faithful to the underlying model. Here, we address this gap by proposing two general algorithms (latent and superpixel-based) which can isolate multiple clear feature parts in a test image, and then connect them to the explanatory cases found in the training data, before testing their effectiveness in a carefully designed user study. Results demonstrate that the proposed approach appropriately calibrates a user's feelings of "correctness" for ambiguous classifications in real world data on the ImageNet dataset, an effect which does not happen when just showing the explanation without feature highlighting.

----

## [48] Fairness via Group Contribution Matching

**Authors**: *Tianlin Li, Zhiming Li, Anran Li, Mengnan Du, Aishan Liu, Qing Guo, Guozhu Meng, Yang Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/49](https://doi.org/10.24963/ijcai.2023/49)

**Abstract**:

Fairness issues in Deep Learning models have recently received increasing attention due to their significant societal impact. Although methods for mitigating unfairness are constantly proposed, little research has been conducted to understand how discrimination and bias develop during the standard training process. In this study, we propose analyzing the contribution of each subgroup (i.e., a group of data with the same sensitive attribute) in the training process to understand the cause of such bias development process. We propose a gradient-based metric to assess training subgroup contribution disparity, showing that unequal contributions from different subgroups are one source of such unfairness. One way to balance the contribution of each subgroup is through oversampling, which ensures that an equal number of samples are drawn from each subgroup during each training iteration. However, we have found that even with a balanced number of samples, the contribution of each group remains unequal, resulting in unfairness under the oversampling strategy. To address the above issues, we propose an easy but effective group contribution matching (GCM) method to match the contribution of each subgroup. Our experiments show that our GCM effectively improves fairness and outperforms other methods significantly.

----

## [49] Negative Flux Aggregation to Estimate Feature Attributions

**Authors**: *Xin Li, Deng Pan, Chengyin Li, Yao Qiang, Dongxiao Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/50](https://doi.org/10.24963/ijcai.2023/50)

**Abstract**:

There are increasing demands for understanding deep neural networks' (DNNs) behavior spurred by growing security and/or transparency concerns. Due to multi-layer nonlinearity of the deep neural network architectures, explaining DNN predictions still remains as an open problem, preventing us from gaining a deeper understanding of the mechanisms. To enhance the explainability of DNNs, we estimate the input feature's attributions to the prediction task using divergence and flux. Inspired by the divergence theorem in vector analysis, we develop a novel Negative Flux Aggregation (NeFLAG) formulation and an efficient approximation algorithm to estimate attribution map. Unlike the previous techniques, ours doesn't rely on fitting a surrogate model nor need any path integration of gradients. Both qualitative and quantitative experiments demonstrate a superior performance of NeFLAG in generating more faithful attribution maps than the competing methods.  Our code is available at https://github.com/xinli0928/NeFLAG.

----

## [50] Robust Reinforcement Learning via Progressive Task Sequence

**Authors**: *Yike Li, Yunzhe Tian, Endong Tong, Wenjia Niu, Jiqiang Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/51](https://doi.org/10.24963/ijcai.2023/51)

**Abstract**:

Robust reinforcement learning (RL) has been a challenging problem due to the gap between simulation and the real world. Existing efforts typically address the robust RL problem by solving a max-min problem. The main idea is to maximize the cumulative reward under the worst-possible perturbations. However, the worst-case optimization either leads to overly conservative solutions or unstable training process, which further affects the policy robustness and generalization performance. In this paper, we tackle this problem from both formulation definition and algorithm design. First, we formulate the robust RL as a max-expectation optimization problem, where the goal is to find an optimal policy under both the worst cases and the non-worst cases. Then, we propose a novel framework DRRL to solve the max-expectation optimization. Given our definition of the feasible tasks, a task generation and sequencing mechanism is introduced to dynamically output tasks at appropriate difficulty level for the current policy. With these progressive tasks, DRRL realizes dynamic multi-task learning to improve the policy robustness and the training stability. Finally, extensive experiments demonstrate that the proposed method exhibits significant performance on the unmanned CarRacing game and multiple high-dimensional MuJoCo environments.

----

## [51] Towards Robust Gan-Generated Image Detection: A Multi-View Completion Representation

**Authors**: *Chi Liu, Tianqing Zhu, Sheng Shen, Wanlei Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/52](https://doi.org/10.24963/ijcai.2023/52)

**Abstract**:

GAN-generated image detection now becomes the first line of defense against the malicious uses of machine-synthesized image manipulations such as deepfakes. Although some existing detectors work well in detecting clean, known GAN samples, their success is largely attributable to overfitting unstable features such as frequency artifacts, which will cause failures when facing unknown GANs or perturbation attacks. To overcome the issue, we propose a robust detection framework based on a novel multi-view image completion representation. The framework first learns various view-to-image tasks to model the diverse distributions of genuine images. Frequency-irrelevant features can be represented from the distributional discrepancies characterized by the completion models, which are stable, generalized, and robust for detecting unknown fake patterns. Then, a multi-view classification is devised with elaborated intra- and inter-view learning strategies to enhance view-specific feature representation and cross-view feature aggregation, respectively. We evaluated the generalization ability of our framework across six popular GANs at different resolutions and its robustness against a broad range of perturbation attacks. The results confirm our method's improved effectiveness, generalization, and robustness over various baselines.

----

## [52] Explanation-Guided Reward Alignment

**Authors**: *Saaduddin Mahmud, Sandhya Saisubramanian, Shlomo Zilberstein*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/53](https://doi.org/10.24963/ijcai.2023/53)

**Abstract**:

Agents often need to infer a reward function from observations to learn desired behaviors. However, agents may infer a reward function that does not align with the original intent because there can be multiple reward functions consistent with its observations. Operating based on such misaligned rewards can be risky. Furthermore, black-box representations make it difficult to verify the learned rewards and prevent harmful behavior. We present a framework for verifying and improving reward alignment using explanations and show how explanations can help detect misalignment and reveal failure cases in novel scenarios. The problem is formulated as inverse reinforcement learning from ranked trajectories. Verification tests created from the trajectory dataset are used to iteratively validate and improve reward alignment. The agent explains its learned reward and a tester signals whether the explanation passes the test. In cases where the explanation fails, the agent offers alternative explanations to gather feedback, which is then used to improve the learned reward. We analyze the efficiency of our approach in improving reward alignment using different types of explanations and demonstrate its effectiveness in five domains.

----

## [53] Adversarial Behavior Exclusion for Safe Reinforcement Learning

**Authors**: *Md Asifur Rahman, Tongtong Liu, Sarra M. Alqahtani*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/54](https://doi.org/10.24963/ijcai.2023/54)

**Abstract**:

Learning by exploration makes reinforcement learning (RL) potentially attractive for many real-world applications. However, this learning process makes RL inherently too vulnerable to be used in real-world applications where safety is of utmost importance. Most prior studies consider exploration at odds with safety and thereby restrict it using either joint optimization of task and safety or imposing constraints for safe exploration. This paper migrates from the current convention to using exploration as a key to safety by learning safety as a robust behavior that completely excludes any behavioral pattern responsible for safety violations. Adversarial Behavior Exclusion for Safe RL (AdvEx-RL) learns a behavioral representation of the agent's safety violations by approximating an optimal adversary utilizing exploration and later uses this representation to learn a separate safety policy that excludes those unsafe behaviors. In addition, AdvEx-RL ensures safety in a task-agnostic manner by acting as a safety firewall and therefore can be integrated with any RL task policy. We demonstrate the robustness of AdvEx-RL via comprehensive experiments in standard constrained Markov decision processes (CMDP) environments under 2 white-box action space perturbations as well as with changes in environment dynamics against 7 baselines. Consistently, AdvEx-RL outperforms the baselines by achieving an average safety performance of over 75% in the continuous action space with 10 times more variations in the testing environment dynamics. By using a standalone safety policy independent of conflicting objectives, AdvEx-RL also paves the way for interpretable safety behavior analysis as we show in our user study.

----

## [54] FEAMOE: Fair, Explainable and Adaptive Mixture of Experts

**Authors**: *Shubham Sharma, Jette Henderson, Joydeep Ghosh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/55](https://doi.org/10.24963/ijcai.2023/55)

**Abstract**:

Three key properties that are desired of trustworthy machine learning models deployed in high-stakes environments are fairness, explainability, and an ability to account for various kinds of "drift". While drifts in model accuracy have been widely investigated, drifts in fairness metrics over time remain largely unexplored. In this paper, we propose FEAMOE, a novel "mixture-of-experts" inspired framework aimed at learning fairer, more interpretable models that can also rapidly adjust to drifts in both the accuracy and the fairness of a classifier. We illustrate our framework for three popular fairness measures and demonstrate how drift can be handled with respect to these fairness constraints. Experiments on multiple datasets show that our framework as applied to a mixture of linear experts is able to perform comparably to neural networks in terms of accuracy while producing fairer models. We then use the large-scale HMDA dataset and show that various models trained on HMDA demonstrate drift and FEAMOE can ably handle these drifts with respect to all the considered fairness measures and maintain model accuracy. We also prove that the proposed framework allows for producing fast Shapley value explanations, which makes computationally efficient feature attribution based explanations of model decisions readily available via FEAMOE.

----

## [55] SF-PATE: Scalable, Fair, and Private Aggregation of Teacher Ensembles

**Authors**: *Cuong Tran, Keyu Zhu, Ferdinando Fioretto, Pascal Van Hentenryck*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/56](https://doi.org/10.24963/ijcai.2023/56)

**Abstract**:

A critical concern in data-driven processes is to build models whose outcomes do not discriminate against some protected groups. In learning tasks, knowledge of the group attributes is essential to ensure non-discrimination, but in practice, these attributes may not be available due to legal and ethical requirements. To address this challenge, this paper studies a model that protects the privacy of individualsâ€™ sensitive information while also allowing it to learn non-discriminatory predictors.
A key feature of the proposed model is to enable the use of off-the-shelves and non-private fair models to create a privacy-preserving and fair model. The paper analyzes the relation between accuracy, privacy, and fairness, and assesses the benefits of the proposed models on several prediction tasks. In particular, this proposal allows both scalable and accurate training of private and fair models for very large neural networks.

----

## [56] On the Fairness Impacts of Private Ensembles Models

**Authors**: *Cuong Tran, Ferdinando Fioretto*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/57](https://doi.org/10.24963/ijcai.2023/57)

**Abstract**:

The Private Aggregation of Teacher Ensembles (PATE) is a machine learning framework that enables the creation of private models through the combination of multiple "teacher" models and a "student" model. The student model learns to predict an output based on the voting of the teachers, and the resulting model satisfies differential privacy. PATE has been shown to be effective in creating private models in semi-supervised settings or when protecting data labels is a priority. 
This paper explores whether the use of PATE can result in unfairness, and demonstrates that it can lead to accuracy disparities among groups of individuals. The paper also analyzes the algorithmic and data properties that contribute to these disproportionate impacts, why these aspects are affecting different groups disproportionately, and offers recommendations for mitigating these effects.

----

## [57] Statistically Significant Concept-based Explanation of Image Classifiers via Model Knockoffs

**Authors**: *Kaiwen Xu, Kazuto Fukuchi, Youhei Akimoto, Jun Sakuma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/58](https://doi.org/10.24963/ijcai.2023/58)

**Abstract**:

A concept-based classifier can explain the decision process of a deep learning model by human understandable concepts in image classification problems. However, sometimes concept-based explanations may cause false positives, which misregards unrelated concepts as important for the prediction task. Our goal is to find the statistically significant concept for classification to prevent misinterpretation. In this study, we propose a method using a deep learning model to learn the image concept and then using the knockoff sample to select the important concepts for prediction by controlling the False Discovery Rate (FDR) under a certain value. We evaluate the proposed method in our experiments on both synthetic and real data. Also, it shows that our method can control the FDR properly while selecting highly interpretable concepts to improve the trustworthiness of the model.

----

## [58] On Adversarial Robustness of Demographic Fairness in Face Attribute Recognition

**Authors**: *Huimin Zeng, Zhenrui Yue, Lanyu Shang, Yang Zhang, Dong Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/59](https://doi.org/10.24963/ijcai.2023/59)

**Abstract**:

Demographic fairness has become a critical objective when developing modern visual models for identity-sensitive applications, such as face attribute recognition (FAR). While great efforts have been made to improve the fairness of the models, the investigation on the adversarial robustness of the fairness (e.g., whether the fairness of the models could still be maintained under potential malicious fairness attacks) is largely ignored. Therefore, this paper explores the adversarial robustness of demographic fairness in FAR applications from both attacking and defending perspectives. In particular, we firstly present a novel fairness attack, who aims at corrupting the demographic fairness of face attribute classifiers. Next, to mitigate the effect of the fairness attack, we design an efficient defense algorithm called robust-fair training. With this defense, face attribute classifiers learn how to combat the bias introduced by the fairness attack. As such, the face attribute classifiers are not only trained to be fair, but the fairness is also robust. Our extensive experimental results show the effectiveness of both our proposed attack and defense methods across various model architectures and FAR applications. We believe our work could be strong baselines for future work on robust-fair AI models.

----

## [59] Towards Semantics- and Domain-Aware Adversarial Attacks

**Authors**: *Jianping Zhang, Yung-Chieh Huang, Weibin Wu, Michael R. Lyu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/60](https://doi.org/10.24963/ijcai.2023/60)

**Abstract**:

Language models are known to be vulnerable to textual adversarial attacks, which add human-imperceptible perturbations to the input to mislead DNNs. It is thus imperative to devise effective attack algorithms to identify the deficiencies of DNNs before real-world deployment. However, existing word-level attacks have two major deficiencies: (1) They may change the semantics of the original sentence. (2) The generated adversarial sample can appear unnatural to humans due to the introduction of out-of-domain substitute words. In this paper, to address such drawbacks, we propose a semantics- and domain-aware word-level attack method. Specifically, we greedily replace the important words in a sentence with the ones suggested by a language model. The language model is trained to be semantics- and domain-aware via contrastive learning and in-domain pre-training. Furthermore, to balance the quality of adversarial examples and the attack success rate, we propose an iterative updating framework to optimize the contrastive learning loss and the in-domain pre-training loss in circular order. Comprehensive experimental comparisons confirm the superiority of our approach. Notably, compared with state-of-the-art benchmarks, our strategy can achieve over 3\% improvement in attack success rates and 9.8\% improvement in the quality of adversarial examples.

----

## [60] Tracking Different Ant Species: An Unsupervised Domain Adaptation Framework and a Dataset for Multi-object Tracking

**Authors**: *Chamath Abeysinghe, Chris Reid, Hamid Rezatofighi, Bernd Meyer*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/61](https://doi.org/10.24963/ijcai.2023/61)

**Abstract**:

Tracking individuals is a vital part of many experiments conducted to understand collective behaviour. Ants are the paradigmatic model system for such experiments but their lack of individually distinguishing visual features and their high colony densities make it extremely difficult to perform reliable racking automatically. Additionally, the wide diversity of their species'  appearances makes a generalized approach even harder. In this paper, we propose a data-driven multi-object tracker that, for the first time, employs domain adaptation to achieve the required generalisation. This approach is built upon a joint-detection-and-tracking framework that is extended by a set of domain discriminator modules integrating an adversarial training strategy in addition to the tracking loss. In addition to this novel domain-adaptive tracking framework, we present a new dataset and a benchmark for the ant tracking problem. The dataset contains 57 video sequences with full trajectory annotation, including 30k frames captured from two different ant species moving on different background patterns. It comprises 33 and 24 sequences for source and target domains, respectively. We compare our proposed framework against other domain-adaptive and non-domain-adaptive multi-object tracking baselines using this dataset and show that incorporating domain adaptation at multiple levels of the tracking pipeline yields significant improvements. The code and the dataset are available at https://github.com/chamathabeysinghe/da-tracker.

----

## [61] RaSa: Relation and Sensitivity Aware Representation Learning for Text-based Person Search

**Authors**: *Yang Bai, Min Cao, Daming Gao, Ziqiang Cao, Chen Chen, Zhenfeng Fan, Liqiang Nie, Min Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/62](https://doi.org/10.24963/ijcai.2023/62)

**Abstract**:

Text-based person search aims to retrieve the specified person images given a textual description. The key to tackling such a challenging task is to learn powerful multi-modal representations. Towards this, we propose a Relation and Sensitivity aware representation learning method (RaSa), including two novel tasks: Relation-Aware learning (RA) and Sensitivity-Aware learning (SA). For one thing, existing methods cluster representations of all positive pairs without distinction and overlook the noise problem caused by the weak positive pairs where the text and the paired image have noise correspondences, thus leading to overfitting learning. RA offsets the overfitting risk by introducing a novel positive relation detection task (i.e., learning to distinguish strong and weak positive pairs). For another thing, learning invariant representation under data augmentation (i.e., being insensitive to some transformations) is a general practice for improving representation's robustness in existing methods. Beyond that, we encourage the representation to perceive the sensitive transformation by SA (i.e., learning to detect the replaced words), thus promoting the representation's robustness. Experiments demonstrate that RaSa outperforms existing state-of-the-art methods by 6.94%, 4.45% and 15.35% in terms of Rank@1 on CUHK-PEDES, ICFG-PEDES and RSTPReid datasets, respectively. Code is available at: https://github.com/Flame-Chasers/RaSa.

----

## [62] A Novel Learnable Interpolation Approach for Scale-Arbitrary Image Super-Resolution

**Authors**: *Jiahao Chao, Zhou Zhou, Hongfan Gao, Jiali Gong, Zhenbing Zeng, Zhengfeng Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/63](https://doi.org/10.24963/ijcai.2023/63)

**Abstract**:

Deep convolutional neural networks (CNNs) have achieved unprecedented success in single image super-resolution over the past few years. Meanwhile, there is an increasing demand for single image super-resolution with arbitrary scale factors in real-world scenarios. Many approaches adopt scale-specific multi-path learning to cope with multi-scale super-resolution with a single network. However, these methods require a large number of parameters. To achieve a better balance between the reconstruction quality and parameter amounts, we proposes a learnable interpolation method that leverages the advantages of neural networks and interpolation methods to tackle the scale-arbitrary super-resolution task. The scale factor is treated as a function parameter for generating the kernel weights for the learnable interpolation. We demonstrate that the learnable interpolation builds a bridge between neural networks and traditional interpolation methods. Experiments show that the proposed learnable interpolation requires much fewer parameters and outperforms state-of-the-art super-resolution methods.

----

## [63] MMPN: Multi-supervised Mask Protection Network for Pansharpening

**Authors**: *Changjie Chen, Yong Yang, Shuying Huang, Wei Tu, Weiguo Wan, Shengna Wei*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/64](https://doi.org/10.24963/ijcai.2023/64)

**Abstract**:

Pansharpening is to fuse a panchromatic (PAN) image with a multispectral (MS) image to obtain a high-spatial-resolution multispectral (HRMS) image. The deep learning-based pansharpening methods usually apply the convolution operation to extract features and only consider the similarity of gradient information between PAN and HRMS images, resulting in the problems of edge blur and spectral distortion in the fusion results. To solve this problem, a multi-supervised mask protection network (MMPN) is proposed to prevent spatial information from being damaged and overcome spectral distortion in the learning process. Firstly, by analyzing the relationships between high-resolution images and corresponding degraded images, a mask protection strategy (MPS) for edge protection is designed to guide the recovery of fused images. Then, based on the MPS, an MMPN containing four branches is constructed to generate the fusion and mask protection images. In MMPN, each branch employs a dual-stream multi-scale feature fusion module (DMFFM), which is built to extract and fuse the features of two input images. Finally, different loss terms are defined for the four branches, and combined into a joint loss function to realize network training. Experiments on simulated and real satellite datasets show that our method is superior to state-of-the-art methods both subjectively and objectively.

----

## [64] HDFormer: High-order Directed Transformer for 3D Human Pose Estimation

**Authors**: *Han-Yuan Chen, Jun-Yan He, Wangmeng Xiang, Zhi-Qi Cheng, Wei Liu, Hanbing Liu, Bin Luo, Yifeng Geng, Xuansong Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/65](https://doi.org/10.24963/ijcai.2023/65)

**Abstract**:

Human pose estimation is a challenging task due to its structured data sequence nature. Existing methods primarily focus on pair-wise interaction of body joints, which is insufficient for scenarios involving overlapping joints and rapidly changing poses. To overcome these issues, we introduce a novel approach, the High-order Directed Transformer (HDFormer), which leverages high-order bone and joint relationships for improved pose estimation. Specifically, HDFormer incorporates both self-attention and high-order attention to formulate a multi-order attention module. This module facilitates first-order "joint-joint", second-order "bone-joint", and high-order "hyperbone-joint" interactions, effectively addressing issues in complex and occlusion-heavy situations. In addition, modern CNN techniques are integrated into the transformer-based architecture, balancing the trade-off between performance and efficiency. HDFormer significantly outperforms state-of-the-art (SOTA) models on Human3.6M and MPI-INF-3DHP datasets, requiring only 1/10 of the parameters and significantly lower computational costs. Moreover, HDFormer demonstrates broad real-world applicability, enabling real-time, accurate 3D pose estimation. The source code is in https://github.com/hyer/HDFormer.

----

## [65] Fluid Dynamics-Inspired Network for Infrared Small Target Detection

**Authors**: *Tianxiang Chen, Qi Chu, Bin Liu, Nenghai Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/66](https://doi.org/10.24963/ijcai.2023/66)

**Abstract**:

Most infrared small target detection (ISTD) networks focus on building effective neural blocks or feature fusion modules but none describes the ISTD process from the image evolution perspective. The directional evolution of image pixels influenced by convolution, pooling and surrounding pixels is analogous to the movement of fluid elements constrained by surrounding variables ang particles. Inspired by this, we explore a novel research routine by abstracting the movement of pixels in the ISTD process as the flow of fluid in fluid dynamics (FD). Specifically, a new Fluid Dynamics-Inspired Network (FDI-Net) is devised for ISTD. Based on Taylor Central Difference (TCD) method, the TCD feature extraction block is designed, where convolution and Transformer structures are combined for local and global information. The pixel motion equation during the ISTD process is derived from the Navierâ€“Stokes (N-S) equation, constructing a N-S Refinement Module that refines extracted features with edge details. Thus, the TCD feature extraction block determines the primary movement direction of pixels during detection, while the N-S Refinement Module corrects some skewed directions of the pixel stream to supplement the edge details. Experiments on IRSTD-1k and SIRST demonstrate that our method achieves SOTA performance in terms of evaluation metrics.

----

## [66] CostFormer: Cost Transformer for Cost Aggregation in Multi-view Stereo

**Authors**: *Weitao Chen, Hongbin Xu, Zhipeng Zhou, Yang Liu, Baigui Sun, Wenxiong Kang, Xuansong Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/67](https://doi.org/10.24963/ijcai.2023/67)

**Abstract**:

The core of Multi-view Stereo(MVS) is the matching process among reference and source pixels. Cost aggregation plays a significant role in this process, while previous methods focus on handling it via CNNs. This may inherit the natural limitation of CNNs that fail to discriminate repetitive or incorrect matches due to limited local receptive fields. To handle the issue, we aim to involve Transformer into cost aggregation. However, another problem may occur due to the quadratically growing computational complexity caused by Transformer, resulting in memory overflow and inference latency. In this paper, we overcome these limits with an efficient Transformer-based cost aggregation network, namely CostFormer. The Residual Depth-Aware Cost Transformer(RDACT) is proposed to aggregate long-range features on cost volume via self-attention mechanisms along the depth and spatial dimensions. Furthermore, Residual Regression Transformer(RRT) is proposed to enhance spatial attention. The proposed method is a universal plug-in to improve learning-based MVS methods.

----

## [67] Self-Supervised Neuron Segmentation with Multi-Agent Reinforcement Learning

**Authors**: *Yinda Chen, Wei Huang, Shenglong Zhou, Qi Chen, Zhiwei Xiong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/68](https://doi.org/10.24963/ijcai.2023/68)

**Abstract**:

The performance of existing supervised neuron segmentation methods is highly dependent on the number of accurate annotations, especially when applied to large scale electron microscopy (EM) data. By extracting semantic information from unlabeled data, self-supervised methods can improve the performance of downstream tasks, among which the mask image model (MIM) has been widely used due to its simplicity and effectiveness in recovering original information from masked images. However, due to the high degree of structural locality in EM images, as well as the existence of considerable noise, many voxels contain little discriminative information, making MIM pretraining inefficient on the neuron segmentation task. To overcome this challenge, we propose a decision-based MIM that utilizes reinforcement learning (RL) to automatically search for optimal image masking ratio and masking strategy. Due to the vast exploration space, using single-agent RL for voxel prediction is impractical. Therefore, we treat each input patch as an agent with a shared behavior policy, allowing for multi-agent collaboration. Furthermore, this multi-agent model can capture dependencies between voxels, which is beneficial for the downstream segmentation task. Experiments conducted on representative EM datasets demonstrate that our approach has a significant advantage over alternative self-supervised methods on the task of neuron segmentation. Code is available at https://github.com/ydchen0806/dbMiM.

----

## [68] Null-Space Diffusion Sampling for Zero-Shot Point Cloud Completion

**Authors**: *Xinhua Cheng, Nan Zhang, Jiwen Yu, Yinhuai Wang, Ge Li, Jian Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/69](https://doi.org/10.24963/ijcai.2023/69)

**Abstract**:

Point cloud completion aims at estimating the complete data of objects from degraded observations. Despite existing completion methods achieving impressive performances, they rely heavily on degraded-complete data pairs for supervision. In this work, we propose a novel framework named Null-Space Diffusion Sampling (NSDS) to solve the point cloud completion task in a zero-shot manner.  By leveraging a pre-trained point cloud diffusion model as the off-the-shelf generator, our sampling approach can generate desired completion outputs with the guidance of the observed degraded data without any extra training. Furthermore, we propose a tolerant loop mechanism to improve the quality of completion results for hard cases. Experimental results demonstrate our zero-shot framework achieves superior completion performance than unsupervised methods and comparable performance to supervised methods in various degraded situations.

----

## [69] Robust Image Ordinal Regression with Controllable Image Generation

**Authors**: *Yi Cheng, Haochao Ying, Renjun Hu, Jinhong Wang, Wenhao Zheng, Xiao Zhang, Danny Z. Chen, Jian Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/70](https://doi.org/10.24963/ijcai.2023/70)

**Abstract**:

Image ordinal regression has been mainly studied along the line of exploiting the order of categories. However, the issues of class imbalance and category overlap that are very common in ordinal regression were largely overlooked. As a result, the performance on minority categories is often unsatisfactory. In this paper, we propose a novel framework called CIG based on controllable image generation to directly tackle these two issues. Our main idea is to generate extra training samples with specific labels near category boundaries, and the sample generation is biased toward the less-represented categories. To achieve controllable image generation, we seek to separate structural and categorical information of images based on structural similarity, categorical similarity, and reconstruction constraints. We evaluate the effectiveness of our new CIG approach in three different image ordinal regression scenarios. The results demonstrate that CIG can be flexibly integrated with off-the-shelf image encoders or ordinal regression models to achieve improvement, and further, the improvement is more significant for minority categories.

----

## [70] WiCo: Win-win Cooperation of Bottom-up and Top-down Referring Image Segmentation

**Authors**: *Zesen Cheng, Peng Jin, Hao Li, Kehan Li, Siheng Li, Xiangyang Ji, Chang Liu, Jie Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/71](https://doi.org/10.24963/ijcai.2023/71)

**Abstract**:

The top-down and bottom-up methods are two mainstreams of referring segmentation, while both methods have their own intrinsic weaknesses. Top-down methods are chiefly disturbed by Polar Negative (PN) errors owing to the lack of fine-grained cross-modal alignment. Bottom-up methods are mainly perturbed by Inferior Positive (IP) errors due to the lack of prior object information. Nevertheless, we discover that two types of methods are highly complementary for restraining respective weaknesses but the direct average combination leads to harmful interference. In this context, we build Win-win Cooperation (WiCo) to exploit complementary nature of two types of methods on both interaction and integration aspects for achieving a win-win improvement. For the interaction aspect, Complementary Feature Interaction (CFI) introduces prior object information to bottom-up branch and provides fine-grained information to top-down branch for complementary feature enhancement. For the integration aspect, Gaussian Scoring Integration (GSI) models the gaussian performance distributions of two branches and weighted integrates results by sampling confident scores from the distributions. With our WiCo, several prominent bottom-up and top-down combinations achieve remarkable improvements on three common datasets with reasonable extra costs, which justifies effectiveness and generality of our method.

----

## [71] Strip Attention for Image Restoration

**Authors**: *Yuning Cui, Yi Tao, Luoxi Jing, Alois Knoll*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/72](https://doi.org/10.24963/ijcai.2023/72)

**Abstract**:

As a long-standing task, image restoration aims to recover the latent sharp image from its degraded counterpart. In recent years, owing to the strong ability of self-attention in capturing long-range dependencies, Transformer based methods have achieved promising performance on multifarious image restoration tasks. However, the canonical self-attention leads to quadratic complexity with respect to input size, hindering its further applications in image restoration. In this paper, we propose a Strip Attention Network (SANet) for image restoration to integrate information in a more efficient and effective manner. Specifically, a strip attention unit is proposed to harvest the contextual information for each pixel from its adjacent pixels in the same row or column. By employing this operation in different directions, each location can perceive information from an expanded region. Furthermore, we apply various receptive fields in different feature groups to enhance representation learning. Incorporating these designs into a U-shaped backbone, our SANet performs favorably against state-of-the-art algorithms on several image restoration tasks. The code is available at https://github.com/c-yn/SANet.

----

## [72] RZCR: Zero-shot Character Recognition via Radical-based Reasoning

**Authors**: *Xiaolei Diao, Daqian Shi, Hao Tang, Qiang Shen, Yanzeng Li, Lei Wu, Hao Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/73](https://doi.org/10.24963/ijcai.2023/73)

**Abstract**:

The long-tail effect is a common issue that limits the performance of deep learning models on real-world datasets. Character image datasets are also affected by such unbalanced data distribution due to differences in character usage frequency. Thus, current character recognition methods are limited when applied in the real world, especially for the categories in the tail that lack training samples, e.g., uncommon characters. In this paper, we propose a zero-shot character recognition framework via radical-based reasoning, called RZCR, to improve the recognition performance of few-sample character categories in the tail. Specifically, we exploit radicals, the graphical units of characters, by decomposing and reconstructing characters according to orthography. RZCR consists of a visual semantic fusion-based radical information extractor (RIE) and a knowledge graph character reasoner (KGR). RIE aims to recognize candidate radicals and their possible structural relations from character images in parallel. The results are then fed into KGR to recognize the target character by reasoning with a knowledge graph. We validate our method on multiple datasets, and RZCR shows promising experimental results, especially on few-sample character datasets.

----

## [73] Decoupling with Entropy-based Equalization for Semi-Supervised Semantic Segmentation

**Authors**: *Chuanghao Ding, Jianrong Zhang, Henghui Ding, Hongwei Zhao, Zhihui Wang, Tengfei Xing, Runbo Hu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/74](https://doi.org/10.24963/ijcai.2023/74)

**Abstract**:

Semi-supervised semantic segmentation methods are the main solution to alleviate the problem of high annotation consumption in semantic segmentation. However, the class imbalance problem makes the model favor the head classes with sufficient training samples, resulting in poor performance of the tail classes. To address this issue, we propose a Decoupled Semi-Supervise Semantic Segmentation (DeS4) framework based on the teacher-student model. Specifically, we first propose a decoupling training strategy to split the training of the encoder and segmentation decoder, aiming at a balanced decoder. Then, a non-learnable prototype-based segmentation head is proposed to regularize the category representation distribution consistency and perform a better connection between the teacher model and the student model. Furthermore, a Multi-Entropy Sampling (MES) strategy is proposed to collect pixel representation for updating the shared prototype to get a class-unbiased head. We conduct extensive experiments of the proposed DeS4 on two challenging benchmarks (PASCAL VOC 2012 and Cityscapes) and achieve remarkable improvements over the previous state-of-the-art methods.

----

## [74] ICDA: Illumination-Coupled Domain Adaptation Framework for Unsupervised Nighttime Semantic Segmentation

**Authors**: *Chenghao Dong, Xuejing Kang, Anlong Ming*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/75](https://doi.org/10.24963/ijcai.2023/75)

**Abstract**:

The performance of nighttime semantic segmentation has been significantly improved thanks to recent unsupervised methods. However, these methods still suffer from complex domain gaps, i.e., the challenging illumination gap and the inherent dataset gap. In this paper, we propose the illumination-coupled domain adaptation framework(ICDA) to effectively avoid the illumination gap and mitigate the dataset gap by coupling daytime and nighttime images as a whole with semantic relevance. Specifically, we first design a new composite enhancement method(CEM) that considers not only illumination but also spatial consistency to construct the source and target domain pairs, which provides the basic adaptation unit for our ICDA. Next, to avoid the illumination gap, we devise the Deformable Attention Relevance(DAR) module to capture the semantic relevance inside each domain pair, which can couple the daytime and nighttime images at the feature level and adaptively guide the predictions of nighttime images. Besides, to mitigate the dataset gap and acquire domain-invariant semantic relevance, we propose the Prototype-based Class Alignment(PCA) module, which improves the usage of category information and performs fine-grained alignment. Extensive experiments show that our method reduces the complex domain gaps and achieves state-of-the-art performance for nighttime semantic segmentation.  Our code is available at https://github.com/chenghaoDong666/ICDA.

----

## [75] DFVSR: Directional Frequency Video Super-Resolution via Asymmetric and Enhancement Alignment Network

**Authors**: *Shuting Dong, Feng Lu, Zhe Wu, Chun Yuan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/76](https://doi.org/10.24963/ijcai.2023/76)

**Abstract**:

Recently, techniques utilizing frequency-based methods have gained significant attention, as they exhibit exceptional restoration capabilities for detail and structure in video super-resolution tasks. However, most of these frequency-based methods mainly have three major limitations: 1) insufficient exploration of object motion information, 2) inadequate enhancement for high-fidelity regions, and 3) loss of spatial information during convolution. In this paper, we propose a novel network, Directional Frequency Video Super-Resolution (DFVSR), to address these limitations. Specifically,  we reconsider object motion from a new perspective and propose Directional Frequency Representation (DFR), which not only borrows the property of frequency representation of detail and structure information but also contains the direction information of the object motion that is extremely significant in videos. Based on this representation,  we propose a Directional Frequency-Enhanced Alignment (DFEA) to use double enhancements of task-related information for ensuring the retention of high-fidelity frequency regions to generate the high-quality alignment feature. Furthermore, we design a novel Asymmetrical U-shaped network architecture to progressively fuse these alignment features and output the final output. This architecture enables the intercommunication of the same level of resolution in the encoder and decoder to achieve the supplement of spatial information. Powered by the above designs, our method achieves superior performance over state-of-the-art models on both quantitative and qualitative evaluations.

----

## [76] Timestamp-Supervised Action Segmentation from the Perspective of Clustering

**Authors**: *Dazhao Du, Enhan Li, Lingyu Si, Fanjiang Xu, Fuchun Sun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/77](https://doi.org/10.24963/ijcai.2023/77)

**Abstract**:

Video action segmentation under timestamp supervision has recently received much attention due to lower annotation costs. Most existing methods generate pseudo-labels for all frames in each video to train the segmentation model. However, these methods suffer from incorrect pseudo-labels, especially for the semantically unclear frames in the transition region between two consecutive actions, which we call ambiguous intervals. To address this issue, we propose a novel framework from the perspective of clustering, which includes the following two parts. First, pseudo-label ensembling generates incomplete but high-quality pseudo-label sequences, where the frames in ambiguous intervals have no pseudo-labels. Second, iterative clustering iteratively propagates the pseudo-labels to the ambiguous intervals by clustering, and thus updates the pseudo-label sequences to train the model. We further introduce a clustering loss, which encourages the features of frames within the same action segment more compact. Extensive experiments show the effectiveness of our method.

----

## [77] LION: Label Disambiguation for Semi-supervised Facial Expression Recognition with Progressive Negative Learning

**Authors**: *Zhongjing Du, Xu Jiang, Peng Wang, Qizheng Zhou, Xi Wu, Jiliu Zhou, Yan Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/78](https://doi.org/10.24963/ijcai.2023/78)

**Abstract**:

Semi-supervised deep facial expression recognition (SS-DFER) has recently attracted rising research interest due to its more practical setting of abundant unlabeled data. However, there are two main problems unconsidered in current SS-DFER methods: 1) label ambiguity, i.e., given labels mismatch with facial expressions; 2) inefficient utilization of unlabeled data with low-confidence. In this paper, we propose a novel SS-DFER method, including a Label DIsambiguation module and a PrOgressive Negative Learning module, namely LION, to simultaneously address both problems. Specifically, the label disambiguation module operates on labeled data, including data with accurate labels (clear data) and ambiguous labels (ambiguous data). It first uses clear data to calculate prototypes for all the expression classes, and then re-assign a candidate label set to all the ambiguous data. Based on the prototypes and the candidate label set, the ambiguous data can be relabeled more accurately. As for unlabeled data with low-confidence, the progressive negative learning module is developed to iteratively mine more complete complementary labels, which can guide the model to reduce the association between data and corresponding complementary labels. Experiments on three challenging datasets show that our method significantly outperforms the current state-of-the-art approaches in SS-DFER and surpasses fully-supervised baselines. Code will be available at https://github.com/NUM-7/LION.

----

## [78] Improve Video Representation with Temporal Adversarial Augmentation

**Authors**: *Jinhao Duan, Quanfu Fan, Hao Cheng, Xiaoshuang Shi, Kaidi Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/79](https://doi.org/10.24963/ijcai.2023/79)

**Abstract**:

Recent works reveal that adversarial augmentation benefits the generalization of neural networks (NNs) if used in an appropriate manner. In this paper, we introduce Temporal Adversarial Augmentation (TA), a novel video augmentation technique that utilizes temporal attention. Unlike conventional adversarial augmentation, TA is specifically designed to shift the attention distributions of neural networks with respect to video clips by maximizing a temporal-related loss function. We demonstrate that TA will obtain diverse temporal views, which significantly affect the focus of neural networks. Training with these examples remedies the flaw of unbalanced temporal information perception and enhances the ability to defend against temporal shifts, ultimately leading to better generalization. To leverage TA, we propose Temporal Video Adversarial Fine-tuning (TAF) framework for improving video representations. TAF is a model-agnostic, generic, and interpretability-friendly training strategy. We evaluate TAF with four powerful models (TSM, GST, TAM, and TPN) over three challenging temporal-related benchmarks (Something-something V1&V2 and diving48). Experimental results demonstrate that TAF effectively improves the test accuracy of these models with notable margins without introducing additional parameters or computational costs. As a byproduct, TAF also improves the robustness under out-of-distribution (OOD) settings. Code is available at https://github.com/jinhaoduan/TAF.

----

## [79] RFENet: Towards Reciprocal Feature Evolution for Glass Segmentation

**Authors**: *Ke Fan, Changan Wang, Yabiao Wang, Chengjie Wang, Ran Yi, Lizhuang Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/80](https://doi.org/10.24963/ijcai.2023/80)

**Abstract**:

Glass-like objects are widespread in daily life but remain intractable to be segmented for most existing methods. The transparent property makes it difficult to be distinguished from background, while the tiny separation boundary further impedes the acquisition of their exact contour. In this paper, by revealing the key co-evolution demand of semantic and boundary learning, we propose a Selective Mutual Evolution (SME) module to enable the reciprocal feature learning between them. Then to exploit the global shape context, we propose a Structurally Attentive Refinement (SAR) module to conduct a fine-grained feature refinement for those ambiguous points around the boundary. Finally, to further utilize the multi-scale representation, we integrate the above two modules into a cascaded structure and then introduce a Reciprocal Feature Evolution Network (RFENet) for effective glass-like object segmentation. Extensive experiments demonstrate that our RFENet achieves state-of-the-art performance on three popular public datasets. Code is available at https://github.com/VankouF/RFENet.

----

## [80] Reconstruction-Aware Prior Distillation for Semi-supervised Point Cloud Completion

**Authors**: *Zhaoxin Fan, Yulin He, Zhicheng Wang, Kejian Wu, Hongyan Liu, Jun He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/81](https://doi.org/10.24963/ijcai.2023/81)

**Abstract**:

Real-world sensors often produce incomplete, irregular, and noisy point clouds, making point cloud completion increasingly important. However, most existing completion methods rely on large paired datasets for training, which is labor-intensive. This paper proposes RaPD, a novel semi-supervised point cloud completion method that reduces the need for paired datasets. RaPD utilizes a two-stage training scheme, where a deep semantic prior is learned in stage 1 from unpaired complete and incomplete point clouds, and a semi-supervised prior distillation process is introduced in stage 2 to train a completion network using only a small number of paired samples. Additionally, a self-supervised completion module is introduced to improve performance using unpaired incomplete point clouds. Experiments on multiple datasets show that RaPD outperforms previous methods in both homologous and heterologous scenarios.

----

## [81] Sub-Band Based Attention for Robust Polyp Segmentation

**Authors**: *Xianyong Fang, Yuqing Shi, Qingqing Guo, Linbo Wang, Zhengyi Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/82](https://doi.org/10.24963/ijcai.2023/82)

**Abstract**:

This article proposes a novel spectral domain based solution to the challenging polyp segmentation. The main contribution is based on an interesting finding of the significant existence of the middle frequency sub-band during the CNN process. Consequently, a Sub-Band based Attention (SBA) module is proposed, which uniformly adopts either the high or middle sub-bands of the encoder features to boost the decoder features and thus concretely improve the feature discrimination. A strong encoder supplying informative sub-bands is also very important, while we highly value the local-and-global information enriched CNN features. Therefore, a Transformer Attended Convolution (TAC) module as the main encoder block is introduced. It takes the Transformer features to boost the CNN features with stronger long-range object contexts. The combination of SBA and TAC leads to a novel polyp segmentation framework, SBA-Net. It adopts TAC to effectively obtain encoded features which also input to SBA, so that efficient sub-bands based attention maps can be generated for progressively decoding the bottleneck features. Consequently, SBA-Net can achieve the robust polyp segmentation, as the experimental results demonstrate.

----

## [82] Incorporating Unlikely Negative Cues for Distinctive Image Captioning

**Authors**: *Zhengcong Fei, Junshi Huang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/83](https://doi.org/10.24963/ijcai.2023/83)

**Abstract**:

While recent neural image captioning models have shown great promise in terms of automatic metrics, they still struggle with generating generic sentences, which limits their use to only a handful of simple scenarios. On the other hand, negative training has been suggested as an effective way to prevent models from producing frequent yet meaningless sentences. However, when applied to image captioning, this approach may overlook low-frequency but generic and vague sentences, which can be problematic when dealing with diverse and changeable visual scenes. In this paper, we introduce a approach to improve image captioning by integrating negative knowledge that focuses on preventing the model from producing undesirable generic descriptions while addressing previous limitations. We accomplish this by training a negative teacher model that generates image-wise generic sentences with retrieval entropy-filtered data. Subsequently, the student model is required to maximize the distance with multi-level negative knowledge transferring for optimal guiding. Empirical results evaluated on MS COCO benchmark confirm that our plug-and-play framework incorporating unlikely negative knowledge leads to significant improvements in both accuracy and diversity, surpassing previous state-of-the-art methods for distinctive image captioning.

----

## [83] BPNet: Bézier Primitive Segmentation on 3D Point Clouds

**Authors**: *Rao Fu, Cheng Wen, Qian Li, Xiao Xiao, Pierre Alliez*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/84](https://doi.org/10.24963/ijcai.2023/84)

**Abstract**:

This paper proposes BPNet, a novel end-to-end deep learning framework to learn Bézier primitive segmentation on 3D point clouds. The existing works treat different primitive types separately, thus limiting them to finite shape categories. To address this issue, we seek a generalized primitive segmentation on point clouds. Taking inspiration from Bézier decomposition on NURBS models, we transfer it to guide point cloud segmentation casting off primitive types. A joint optimization framework is proposed to learn Bézier primitive segmentation and geometric fitting simultaneously on a cascaded architecture. Specifically, we introduce a soft voting regularizer to improve primitive segmentation and propose an auto-weight embedding module to cluster point features, making the network more robust and generic. We also introduce a reconstruction module where we successfully process multiple CAD models with different primitives simultaneously. We conducted extensive experiments on the synthetic ABC dataset and real-scan datasets to validate and compare our approach with different baseline methods. Experiments show superior performance over previous work in terms of segmentation, with a substantially faster inference speed.

----

## [84] Contrastive Learning for Sign Language Recognition and Translation

**Authors**: *Shiwei Gan, Yafeng Yin, Zhiwei Jiang, Kang Xia, Lei Xie, Sanglu Lu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/85](https://doi.org/10.24963/ijcai.2023/85)

**Abstract**:

There are two problems that widely exist in current end-to-end sign language processing architecture. One is the CTC spike phenomenon which weakens the visual representational ability in Continuous Sign Language Recognition (CSLR). The other one is the exposure bias problem which leads to the accumulation of translation errors during inference in  Sign Language Translation (SLT). In this paper, we tackle these issues by introducing contrast learning, aiming to enhance both visual-level feature representation and semantic-level error tolerance. Specifically, to alleviate CTC spike phenomenon and enhance visual-level representation, we design a visual contrastive loss by minimizing visual feature distance between different augmented samples of frames in one sign video, so that the model can further explore features by utilizing numerous unlabeled frames in an unsupervised way. To alleviate exposure bias problem and improve semantic-level error tolerance, we design a semantic contrastive loss by re-inputting the predicted sentence into semantic module and comparing features of ground-truth sequence  and predicted sequence, for exposing model to its own mistakes. Besides, we propose two new metrics, i.e., Blank Rate and Consecutive Wrong Word Rate to directly reflect our improvement on the two problems. Extensive experimental results on current sign language datasets demonstrate the effectiveness of our approach, which achieves state-of-the-art performance.

----

## [85] LISSNAS: Locality-based Iterative Search Space Shrinkage for Neural Architecture Search

**Authors**: *Bhavna Gopal, Arjun Sridhar, Tunhou Zhang, Yiran Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/86](https://doi.org/10.24963/ijcai.2023/86)

**Abstract**:

Search spaces hallmark the advancement of Neural Architecture Search (NAS). Large and complex search spaces with versatile building operators and structures provide more opportunities to brew promising architectures, yet pose severe challenges on efficient exploration and exploitation. Subsequently, several search space shrinkage methods optimize by selecting a single sub-region that contains some well-performing networks.  Small performance and efficiency gains are observed with these methods but such techniques leave room for significantly improved search performance and are ineffective at retaining architectural diversity. We propose LISSNAS, an automated algorithm that shrinks a large space into a diverse, small search space with SOTA search performance. Our approach leverages locality, the relationship between structural and performance similarity, to efficiently extract many pockets of well-performing networks. We showcase our method on an array of search spaces spanning various sizes and datasets. We accentuate the effectiveness of our shrunk spaces when used in one-shot search by achieving the best Top-1 accuracy in two different search spaces. Our method achieves a SOTA Top-1 accuracy of 77.6% in ImageNet under mobile constraints, best-in-class Kendal-Tau, architectural diversity, and search space size.

----

## [86] Towards Robust Scene Text Image Super-resolution via Explicit Location Enhancement

**Authors**: *Hang Guo, Tao Dai, Guanghao Meng, Shu-Tao Xia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/87](https://doi.org/10.24963/ijcai.2023/87)

**Abstract**:

Scene text image super-resolution (STISR), aiming to improve image quality while boosting downstream scene text recognition accuracy, has recently achieved great success. However, most existing methods treat the foreground (character regions) and background (non-character regions) equally in the forward process, and neglect the disturbance from the complex background, thus limiting the performance. To address these issues, in this paper, we propose a novel method LEMMA that explicitly models character regions to produce high-level text-specific guidance for super-resolution. To model the location of characters effectively, we propose the location enhancement module to extract character region features based on the attention map sequence. Besides, we propose the multi-modal alignment module to perform bidirectional visual-semantic alignment to generate high-quality prior guidance, which is then incorporated into the super-resolution branch in an adaptive manner using the proposed adaptive fusion module. Experiments on TextZoom and four scene text recognition benchmarks demonstrate the superiority of our method over other state-of-the-art methods. Code is available at https://github.com/csguoh/LEMMA.

----

## [87] Joint-MAE: 2D-3D Joint Masked Autoencoders for 3D Point Cloud Pre-training

**Authors**: *Ziyu Guo, Renrui Zhang, Longtian Qiu, Xianzhi Li, Pheng-Ann Heng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/88](https://doi.org/10.24963/ijcai.2023/88)

**Abstract**:

Masked Autoencoders (MAE) have shown promising performance in self-supervised learning for both 2D and 3D computer vision. However, existing MAE-style methods can only learn from the data of a single modality, i.e., either images or point clouds, which neglect the implicit semantic and geometric correlation between 2D and 3D. In this paper, we explore how the 2D modality can benefit 3D masked autoencoding, and propose Joint-MAE, a 2D-3D joint MAE framework for self-supervised 3D point cloud pre-training. Joint-MAE randomly masks an input 3D point cloud and its projected 2D images, and then reconstructs the masked information of the two modalities. For better cross-modal interaction, we construct our JointMAE by two hierarchical 2D-3D embedding modules, a joint encoder, and a joint decoder with modal-shared and model-specific decoders. On top of this, we further introduce two cross-modal strategies to boost the 3D representation learning, which are local-aligned attention mechanisms for 2D-3D semantic cues, and a cross-reconstruction loss for 2D-3D geometric constraints. By our pre-training paradigm, Joint-MAE achieves superior performance on multiple downstream tasks, e.g., 92.4% accuracy for linear SVM on ModelNet40 and 86.07% accuracy on the hardest split of ScanObjectNN.

----

## [88] SS-BSN: Attentive Blind-Spot Network for Self-Supervised Denoising with Nonlocal Self-Similarity

**Authors**: *Young-Joo Han, Ha-Jin Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/89](https://doi.org/10.24963/ijcai.2023/89)

**Abstract**:

Recently, numerous studies have been conducted on supervised learning-based image denoising methods. However, these methods rely on large-scale noisy-clean image pairs, which are difficult to obtain in practice. Denoising methods with self-supervised training that can be trained with only noisy images have been proposed to address the limitation. These methods are based on the convolutional neural network (CNN) and have shown promising performance. However, CNN-based methods do not consider using nonlocal self-similarities essential in the traditional method, which can cause performance limitations. This paper presents self-similarity attention (SS-Attention), a novel self-attention module that can capture nonlocal self-similarities to solve the problem. We focus on designing a lightweight self-attention module in a pixel-wise manner, which is nearly impossible to implement using the classic self-attention module due to the quadratically increasing complexity with spatial resolution. Furthermore, we integrate SS-Attention into the blind-spot network called self-similarity-based blind-spot network (SS-BSN). We conduct the experiments on real-world image denoising tasks. The proposed method quantitatively and qualitatively outperforms state-of-the-art methods in self-supervised denoising on the Smartphone Image Denoising Dataset (SIDD) and Darmstadt Noise Dataset (DND) benchmark datasets.

----

## [89] DAMO-StreamNet: Optimizing Streaming Perception in Autonomous Driving

**Authors**: *Jun-Yan He, Zhi-Qi Cheng, Chenyang Li, Wangmeng Xiang, Binghui Chen, Bin Luo, Yifeng Geng, Xuansong Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/90](https://doi.org/10.24963/ijcai.2023/90)

**Abstract**:

In the realm of autonomous driving, real-time perception or streaming perception remains under-explored. This research introduces DAMO-StreamNet, a novel framework that merges the cutting-edge elements of the YOLO series with a detailed examination of spatial and temporal perception techniques. DAMO-StreamNet's main inventions include: (1) a robust neck structure employing deformable convolution, bolstering receptive field and feature alignment capabilities; (2) a dual-branch structure synthesizing short-path semantic features and long-path temporal features, enhancing the accuracy of motion state prediction; (3) logits-level distillation facilitating efficient optimization, which aligns the logits of teacher and student networks in semantic space; and (4) a real-time prediction mechanism that updates the features of support frames with the current frame, providing smooth streaming perception during inference. Our testing shows that DAMO-StreamNet surpasses current state-of-the-art methodologies, achieving 37.8% (normal size (600, 960)) and 43.3% (large size (1200, 1920)) sAP without requiring additional data. This study not only establishes a new standard for real-time perception but also offers valuable insights for future research. The source code is at https://github.com/zhiqic/DAMO-StreamNet.

----

## [90] Independent Feature Decomposition and Instance Alignment for Unsupervised Domain Adaptation

**Authors**: *Qichen He, Siying Xiao, Mao Ye, Xiatian Zhu, Ferrante Neri, Dongde Hou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/91](https://doi.org/10.24963/ijcai.2023/91)

**Abstract**:

Existing Unsupervised Domain Adaptation (UDA) methods typically attempt to perform knowledge transfer in a domain-invariant space explicitly or implicitly. In practice, however, the obtained features is often mixed with domain-specific information which causes performance degradation. To overcome this fundamental limitation, this article presents a novel independent feature decomposition and instance alignment method (IndUDA in short). Specifically, based on an invertible flow, we project the base features into a decomposed latent space with domain-invariant and domain-specific dimensions. To drive semantic decomposition independently, we then swap the domain-invariant part across source and target domain samples with the same category and require their inverted features are consistent in class-level with the original features.  By treating domain-specific information as noise, we replace it by Gaussian noise and further regularize source model training by instance alignment, i.e., requiring the base features close to the corresponding reconstructed features, respectively. Extensive experiment results demonstrate that our method achieves state-of-the-art performance on popular UDA benchmarks. The appendix and code are available at https://github.com/ayombeach/IndUDA.

----

## [91] MILD: Modeling the Instance Learning Dynamics for Learning with Noisy Labels

**Authors**: *Chuanyang Hu, Shipeng Yan, Zhitong Gao, Xuming He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/92](https://doi.org/10.24963/ijcai.2023/92)

**Abstract**:

Despite deep learning has achieved great success, it often relies on a large amount of training data with accurate labels, which are expensive and time-consuming to collect. A prominent direction to reduce the cost is to learn with noisy labels, which are ubiquitous in the real-world applications. A critical challenge for such a learning task is to reduce the effect of network memorization on the falsely-labeled data.  In this work, we propose an iterative selection approach based on the Weibull mixture model, which identifies clean data by considering the overall learning dynamics of each data instance. In contrast to the previous small-loss heuristics, we leverage the observation that deep network is easy to memorize and hard to forget clean data. In particular, we measure the difficulty of memorization and forgetting for each instance via the transition times between being misclassified and being memorized in training, and integrate them into a novel metric for selection. Based on the proposed metric, we retain a subset of identified clean data and repeat the selection procedure to iteratively refine the clean subset, which is finally used for model training. To validate our method, we perform extensive experiments on synthetic noisy datasets and real-world web data, and our strategy outperforms existing noisy-label learning methods.

----

## [92] Diagram Visual Grounding: Learning to See with Gestalt-Perceptual Attention

**Authors**: *Xin Hu, Lingling Zhang, Jun Liu, Xinyu Zhang, Wenjun Wu, Qianying Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/93](https://doi.org/10.24963/ijcai.2023/93)

**Abstract**:

Diagram visual grounding aims to capture the correlation between language expression and local objects in the diagram, and plays an important role in the applications like textbook question answering and cross-modal retrieval. Most diagrams consist of several colors and simple geometries. This results in sparse low-level visual features, which further aggravates the gap between low-level visual and high-level semantic features of diagrams. The phenomenon brings challenges to the diagram visual grounding. To solve the above issues, we propose a gestalt-perceptual attention model to align the diagram objects and language expressions. For low-level visual features, inspired by the gestalt that simulates human visual system, we build a gestalt-perception graph network to make up the features learned by the traditional backbone network. For high-level semantic features, we design a multi-modal context attention mechanism to facilitate the interaction between diagrams and language expressions, so as to enhance the semantics of diagrams. Finally, guided by diagram features and linguistic embedding, the target query is gradually decoded to generate the coordinates of the referred object. By conducting comprehensive experiments on diagrams and natural images, we demonstrate that the proposed model achieves superior performance over the competitors. Our code will be released at https://github.com/AIProCode/GPA.

----

## [93] Dual Video Summarization: From Frames to Captions

**Authors**: *Zhenzhen Hu, Zhenshan Wang, Zijie Song, Richang Hong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/94](https://doi.org/10.24963/ijcai.2023/94)

**Abstract**:

Video summarization and video captioning both condense the video content from the perspective of visual and text modes, i.e. the keyframe selection and language description generation. Existing video-and-language learning models commonly sample multiple frames for training instead of observing all. These sampled deputies greatly improve computational efficiency, but do they represent the original video content enough with no more redundancy? In this work, we propose a dual video summarization framework and verify it in the context of video captioning. Given the video frames, we firstly extract the visual representation based on the ViT model fine-tuned on the video-text domain. Then we summarize the keyframes according to the frame-lever score.  To compress the number of keyframes as much as possible while ensuring the quality of captioning, we learn a cross-modal video summarizer to select the most semantically consistent frames according to the pseudo score label. Top K frames ( K is no more than 3% of the entire video.) are chosen to form the video representation. Moreover, to evaluate the static appearance and temporal information of video, we design the ranking scheme of video representation from two aspects: feature-oriented and sequence-oriented. Finally, we generate the descriptions with a lightweight LSTM decoder. The experiment results on the MSR-VTT and MSVD dataset reveal that, for the generative task as video captioning, a small number of keyframes can convey the same semantic information to perform well on captioning, or even better than the original sampling.

----

## [94] Part Aware Contrastive Learning for Self-Supervised Action Recognition

**Authors**: *Yilei Hua, Wenhan Wu, Ce Zheng, Aidong Lu, Mengyuan Liu, Chen Chen, Shiqian Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/95](https://doi.org/10.24963/ijcai.2023/95)

**Abstract**:

In recent years, remarkable results have been achieved in self-supervised action recognition using skeleton sequences with contrastive learning. It has been observed that the semantic distinction of human action features is often represented by local body parts, such as legs or hands, which are advantageous for skeleton-based action recognition. This paper proposes an attention-based contrastive learning framework for skeleton representation learning, called SkeAttnCLR, which integrates local similarity and global features for skeleton-based action representations. To achieve this, a multi-head attention mask module is employed to learn the soft attention mask features from the skeletons, suppressing non-salient local features while accentuating local salient features, thereby bringing similar local features closer in the feature space. Additionally, ample contrastive pairs are generated by expanding contrastive pairs based on salient and non-salient features with global features, which guide the network to learn the semantic representations of the entire skeleton. Therefore, with the attention mask mechanism, SkeAttnCLR learns local features under different data augmentation views. The experiment results demonstrate that the inclusion of local feature similarity significantly enhances skeleton-based action representation. Our proposed SkeAttnCLR outperforms state-of-the-art methods on NTURGB+D, NTU120-RGB+D, and PKU-MMD datasets. The code and settings are available at this repository: https://github.com/GitHubOfHyl97/SkeAttnCLR.

----

## [95] Orion: Online Backdoor Sample Detection via Evolution Deviance

**Authors**: *Huayang Huang, Qian Wang, Xueluan Gong, Tao Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/96](https://doi.org/10.24963/ijcai.2023/96)

**Abstract**:

Widely-used DNN models are vulnerable to backdoor attacks, where the backdoored model is only triggered by specific inputs but can maintain a high prediction accuracy on benign samples. Existing backdoor input detection strategies rely on the assumption that benign and poisoned samples are separable in the feature representation of the model. However, such an assumption can be broken by advanced feature-hidden backdoor attacks. In this paper, we propose a novel detection framework, dubbed Orion (online backdoor sample detection via evolution deviance). Specifically, we analyze how predictions evolve during a forward pass and find deviations between the shallow and deep outputs of the backdoor inputs. By introducing side nets to track such evolution divergence, Orion eliminates the need for the assumption of latent separability. Additionally, we put forward a scheme to restore the original label of backdoor samples, enabling more robust predictions. Extensive experiments on six attacks, three datasets, and two architectures verify the effectiveness of Orion. It is shown that Orion outperforms state-of-the-art defenses and can identify feature-hidden attacks with an F1-score of 90%, compared to 40% for other detection schemes. Orion can also achieve 80% label recovery accuracy on basic backdoor attacks.

----

## [96] Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation

**Authors**: *Shaofei Huang, Han Li, Yuqing Wang, Hongji Zhu, Jiao Dai, Jizhong Han, Wenge Rong, Si Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/97](https://doi.org/10.24963/ijcai.2023/97)

**Abstract**:

Audio visual segmentation (AVS) aims to segment the sounding objects for each frame of a given video. To distinguish the sounding objects from silent ones, both audio-visual semantic correspondence and temporal interaction are required. The previous method applies multi-frame cross-modal attention to conduct pixel-level interactions between audio features and visual features of multiple frames simultaneously, which is both redundant and implicit. In this paper, we propose an Audio-Queried Transformer architecture, AQFormer, where we define a set of object queries conditioned on audio information and associate each of them to particular sounding objects. Explicit object-level semantic correspondence between audio and visual modalities is established by gathering object information from visual features with predefined audio queries. Besides, an Audio-Bridged Temporal Interaction module is proposed to exchange sounding object-relevant information among multiple frames with the bridge of audio features. Extensive experiments are conducted on two AVS benchmarks to show that our method achieves state-of-the-art performances, especially 7.1% M_J and 7.6% M_F gains on the MS3 setting.

----

## [97] Semi-supervised Domain Adaptation via Prototype-based Multi-level Learning

**Authors**: *Xinyang Huang, Chuang Zhu, Wenkai Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/98](https://doi.org/10.24963/ijcai.2023/98)

**Abstract**:

In semi-supervised domain adaptation (SSDA), a few labeled target samples of each class help the model to transfer knowledge representation from the fully labeled source domain to the target domain. Many existing methods ignore the benefits of making full use of the labeled target samples from multi-level. To make better use of this additional data, we propose a novel Prototype-based Multi-level Learning (ProML) framework to better tap the potential of labeled target samples. To achieve intra-domain adaptation, we first introduce a pseudo-label aggregation based on the intra-domain optimal transport to help the model align the feature distribution of unlabeled target samples and the prototype. At the inter-domain level, we propose a cross-domain alignment loss to help the model use the target prototype for cross-domain knowledge transfer. We further propose a dual consistency based on prototype similarity and linear classifier to promote discriminative learning of compact target feature representation at the batch level. Extensive experiments on three datasets, including DomainNet, VisDA2017, and Office-Home, demonstrate that our proposed method achieves state-of-the-art performance in SSDA. Our code is available at https://github.com/bupt-ai-cz/ProML.

----

## [98] Enriching Phrases with Coupled Pixel and Object Contexts for Panoptic Narrative Grounding

**Authors**: *Tianrui Hui, Zihan Ding, Junshi Huang, Xiaoming Wei, Xiaolin Wei, Jiao Dai, Jizhong Han, Si Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/99](https://doi.org/10.24963/ijcai.2023/99)

**Abstract**:

Panoptic narrative grounding (PNG) aims to segment things and stuff objects in an image described by noun phrases of a narrative caption. As a multimodal task, an essential aspect of PNG is the visual-linguistic interaction between image and caption. The previous two-stage method aggregates visual contexts from offline-generated mask proposals to phrase features, which tend to be noisy and fragmentary. The recent one-stage method aggregates only pixel contexts from image features to phrase features, which may incur semantic misalignment due to lacking object priors. To realize more comprehensive visual-linguistic interaction, we propose to enrich phrases with coupled pixel and object contexts by designing a Phrase-Pixel-Object Transformer Decoder (PPO-TD), where both fine-grained part details and coarse-grained entity clues are aggregated to phrase features. In addition, we also propose a Phrase-Object Contrastive Loss (POCL) to pull closer the matched phrase-object pairs and push away unmatched ones for aggregating more precise object contexts from more phrase-relevant object tokens. Extensive experiments on the PNG benchmark show our method achieves new state-of-the-art performance with large margins.

----

## [99] StackFLOW: Monocular Human-Object Reconstruction by Stacked Normalizing Flow with Offset

**Authors**: *Chaofan Huo, Ye Shi, Yuexin Ma, Lan Xu, Jingyi Yu, Jingya Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/100](https://doi.org/10.24963/ijcai.2023/100)

**Abstract**:

Modeling and capturing the 3D spatial arrangement of the human and the object is the key to perceiving 3D human-object interaction from monocular images. In this work, we propose to use the Human-Object Offset between anchors which are densely sampled from the surface of human mesh and object mesh to represent human-object spatial relation. Compared with previous works which use contact map or implicit distance filed to encode 3D human-object spatial relations, our method is a simple and efficient way to encode the highly detailed spatial correlation between the human and object. Based on this representation, we propose Stacked Normalizing Flow (StackFLOW) to infer the posterior distribution of human-object spatial relations from the image. During the optimization stage, we finetune the human body pose and object 6D pose by maximizing the likelihood of samples based on this posterior distribution and minimizing the 2D-3D corresponding reprojection loss. Extensive experimental results show that our method achieves impressive results on two challenging benchmarks, BEHAVE and InterCap datasets. Our code has been publicly available at https://github.com/MoChen-bop/StackFLOW.

----

## [100] GeNAS: Neural Architecture Search with Better Generalization

**Authors**: *Joonhyun Jeong, Joonsang Yu, Geondo Park, Dongyoon Han, Youngjoon Yoo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/101](https://doi.org/10.24963/ijcai.2023/101)

**Abstract**:

Neural Architecture Search (NAS) aims to automatically excavate the optimal network architecture with superior test performance. Recent neural architecture search (NAS) approaches rely on validation loss or accuracy to find the superior network for the target data. In this paper, we investigate a new neural architecture search measure for excavating architectures with better generalization. We demonstrate that the flatness of the loss surface can be a promising proxy for predicting the generalization capability of neural network architectures. We evaluate our proposed method on various search spaces, showing similar or even better performance compared to the state-of-the-art NAS methods. Notably, the resultant architecture found by flatness measure generalizes robustly to various shifts in data distribution (e.g. ImageNet-V2,-A,-O), as well as various tasks such as object detection and semantic segmentation.

----

## [101] Guided Patch-Grouping Wavelet Transformer with Spatial Congruence for Ultra-High Resolution Segmentation

**Authors**: *Deyi Ji, Feng Zhao, Hongtao Lu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/102](https://doi.org/10.24963/ijcai.2023/102)

**Abstract**:

Most existing ultra-high resolution (UHR) segmentation methods always struggle in the dilemma of balancing memory cost and local characterization accuracy, which are both taken into account in our proposed Guided Patch-Grouping Wavelet Transformer (GPWFormer) that achieves impressive performances. In this work, GPWFormer is a Transformer (T)-CNN (C) mutual leaning framework, where T takes the whole UHR image as input and harvests both local details and fine-grained long-range contextual dependencies, while C takes downsampled image as input for learning the category-wise deep context. For the sake of high inference speed and low computation complexity, T partitions the original UHR image into patches and groups them dynamically, then learns the low-level local details with the lightweight multi-head Wavelet Transformer (WFormer) network. Meanwhile, the fine-grained long-range contextual dependencies are also captured during this process, since patches that are far away in the spatial domain can also be assigned to the same group. In addition, masks produced by C are utilized to guide the patch grouping process, providing a heuristics decision. Moreover, the congruence constraints between the two branches are also exploited to maintain the spatial consistency among the patches. Overall, we stack the multi-stage process in a pyramid way. Experiments show that GPWFormer outperforms the existing methods with significant improvements on five benchmark datasets.

----

## [102] ContrastMotion: Self-supervised Scene Motion Learning for Large-Scale LiDAR Point Clouds

**Authors**: *Xiangze Jia, Hui Zhou, Xinge Zhu, Yandong Guo, Ji Zhang, Yuexin Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/103](https://doi.org/10.24963/ijcai.2023/103)

**Abstract**:

In this paper, we propose a novel self-supervised motion estimator for LiDAR-based autonomous driving via BEV representation. Different from usually adopted self-supervised strategies for data-level structure consistency, we predict scene motion via feature-level consistency between pillars in consecutive frames, which can eliminate the effect caused by noise points and view-changing point clouds in dynamic scenes. Specifically, we propose Soft Discriminative Loss that provides the network with more pseudo-supervised signals to learn discriminative and robust features in a contrastive learning manner. We also propose Gated Multi-Frame Fusion block that learns valid compensation between point cloud frames automatically to enhance feature extraction. Finally, pillar association is proposed to predict pillar correspondence probabilities based on feature distance, and whereby further predicts scene motion. Extensive experiments show the effectiveness and superiority of our ContrastMotion on both scene flow and motion prediction tasks.

----

## [103] Text-Video Retrieval with Disentangled Conceptualization and Set-to-Set Alignment

**Authors**: *Peng Jin, Hao Li, Zesen Cheng, Jinfa Huang, Zhennan Wang, Li Yuan, Chang Liu, Jie Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/104](https://doi.org/10.24963/ijcai.2023/104)

**Abstract**:

Text-video retrieval is a challenging cross-modal task, which aims to align visual entities with natural language descriptions. Current methods either fail to leverage the local details or are computationally expensive. What's worse, they fail to leverage the heterogeneous concepts in data. In this paper, we propose the Disentangled Conceptualization and Set-to-set Alignment (DiCoSA) to simulate the conceptualizing and reasoning process of human beings. For disentangled conceptualization, we divide the coarse feature into multiple latent factors related to semantic concepts. For set-to-set alignment, where a set of visual concepts correspond to a set of textual concepts, we propose an adaptive pooling method to aggregate semantic concepts to address the partial matching. In particular, since we encode concepts independently in only a few dimensions, DiCoSA is superior at efficiency and granularity, ensuring fine-grained interactions using a similar computational complexity as coarse-grained alignment. Extensive experiments on five datasets, including MSR-VTT, LSMDC, MSVD, ActivityNet, and DiDeMo, demonstrate that our method outperforms the existing state-of-the-art methods.

----

## [104] Physics-Guided Human Motion Capture with Pose Probability Modeling

**Authors**: *Jingyi Ju, Buzhen Huang, Chen Zhu, Zhihao Li, Yangang Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/105](https://doi.org/10.24963/ijcai.2023/105)

**Abstract**:

Incorporating physics in human motion capture to avoid artifacts like floating, foot sliding, and ground penetration is a promising direction. Existing solutions always adopt kinematic results as reference motions, and the physics is treated as a post-processing module. However, due to the depth ambiguity, monocular motion capture inevitably suffers from noises, and the noisy reference often leads to failure for physics-based tracking. To address the obstacles, our key-idea is to employ physics as denoising guidance in the reverse diffusion process to reconstruct physically plausible human motion from a modeled pose probability distribution. Specifically, we first train a latent gaussian model that encodes the uncertainty of 2D-to-3D lifting to facilitate reverse diffusion. Then, a physics module is constructed to track the motion sampled from the distribution. The discrepancies between the tracked motion and image observation are used to provide explicit guidance for the reverse diffusion model to refine the motion. With several iterations, the physics-based tracking and kinematic denoising promote each other to generate a physically plausible human motion. Experimental results show that our method outperforms previous physics-based methods in both joint accuracy and success rate. More information can be found at https://github.com/Me-Ditto/Physics-Guided-Mocap.

----

## [105] SWAT: Spatial Structure Within and Among Tokens

**Authors**: *Kumara Kahatapitiya, Michael S. Ryoo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/106](https://doi.org/10.24963/ijcai.2023/106)

**Abstract**:

Modeling visual data as tokens (i.e., image patches) using attention mechanisms, feed-forward networks or convolutions has been highly effective in recent years. Such methods usually have a common pipeline: a tokenization method, followed by a set of layers/blocks for information mixing, both within and among tokens. When image patches are converted into tokens, they are often flattened, discarding the spatial structure within each patch. As a result, any processing that follows (eg: multi-head self-attention) may fail to recover and/or benefit from such information. In this paper, we argue that models can have significant gains when spatial structure is preserved during tokenization, and is explicitly used during the mixing stage. We propose two key contributions: (1) Structure-aware Tokenization and, (2) Structure-aware Mixing, both of which can be combined with existing models with minimal effort. We introduce a family of models (SWAT), showing improvements over the likes of DeiT, MLP-Mixer and Swin Transformer, across multiple benchmarks including ImageNet classification and ADE20K segmentation. Our code is available at github.com/kkahatapitiya/SWAT.

----

## [106] Spatially Constrained Adversarial Attack Detection and Localization in the Representation Space of Optical Flow Networks

**Authors**: *Hannah Kim, Celia Cintas, Girmaw Abebe Tadesse, Skyler Speakman*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/107](https://doi.org/10.24963/ijcai.2023/107)

**Abstract**:

Optical flow estimation have shown significant improvements with advances in deep neural networks. However, these flow networks have recently been shown to be vulnerable to patch-based adversarial attacks, which poses security risks in real-world applications, such as self-driving cars and robotics. We propose SADL, a Spatially constrained adversarial Attack Detection and Localization framework, to detect and localize these patch-based attack without requiring a dedicated training. The detection of an attacked input sequence is performed via iterative optimization on the features from the inner layers of flow networks, without any prior knowledge of the attacks. The novel spatially constrained optimization ensures that the detected anomalous subset of features comes from a local region. To this end, SADL provides a subset of nodes within a spatial neighborhood that contribute more to the detection, which will be utilized to localize the attack in the input sequence. The proposed SADL is validated across multiple datasets and flow networks. With patch attacks 4.8% of the size of the input image resolution on RAFT, our method successfully detects and localizes them with an average precision of 0.946 and 0.951 for KITTI-2015 and MPI-Sintel datasets, respectively. The results show that SADL consistently achieves higher detection rates than existing methods and provides new localization capabilities.

----

## [107] IMF: Integrating Matched Features Using Attentive Logit in Knowledge Distillation

**Authors**: *Jeongho Kim, Hanbeen Lee, Simon S. Woo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/108](https://doi.org/10.24963/ijcai.2023/108)

**Abstract**:

Knowledge distillation (KD) is an effective method for transferring the knowledge of a teacher model to a student model, that aims to improve the latter's performance efficiently. Although generic knowledge distillation methods such as softmax representation distillation and intermediate feature matching have demonstrated improvements with various tasks, only marginal improvements are shown in student networks due to their limited model capacity. In this work, to address the student model's limitation, we propose a novel flexible KD framework, Integrating Matched Features using Attentive Logit in Knowledge Distillation (IMF). Our approach introduces an intermediate feature distiller (IFD) to improve the overall performance of the student model by directly distilling the teacher's knowledge into branches of student models. The generated output of IFD, which is trained by the teacher model, is effectively combined by attentive logit. We use only a few blocks of the student and the trained IFD during inference, requiring an equal or less number of parameters. Through extensive experiments, we demonstrate that IMF consistently outperforms other state-of-the-art methods with a large margin over the various datasets in different tasks without extra computation.

----

## [108] Character As Pixels: A Controllable Prompt Adversarial Attacking Framework for Black-Box Text Guided Image Generation Models

**Authors**: *Ziyi Kou, Shichao Pei, Yijun Tian, Xiangliang Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/109](https://doi.org/10.24963/ijcai.2023/109)

**Abstract**:

In this paper, we study a controllable prompt adversarial attacking problem for text guided image generation (Text2Image) models in the black-box scenario, where the goal is to attack specific visual subjects (e.g., changing a brown dog to white) in a generated image by slightly, if not imperceptibly, perturbing the characters of the driven prompt (e.g., ``brown'' to ``br0wn''). Our study is motivated by the limitations of current Text2Image attacking approaches that still rely on manual trials to create adversarial prompts. To address such limitations, we develop CharGrad, a character-level gradient based attacking framework that replaces specific characters of a prompt with pixel-level similar ones by interactively learning the perturbation direction for the prompt and updating the attacking examiner for the generated image based on a novel proxy perturbation representation for characters.  We evaluate CharGrad using the texts from two public image captioning datasets. Results demonstrate that CharGrad outperforms existing text adversarial attacking approaches on attacking various subjects of generated images by black-box Text2Image models in a more effective and efficient way with less perturbation on the characters of the prompts.

----

## [109] Clustered-patch Element Connection for Few-shot Learning

**Authors**: *Jinxiang Lai, Siqian Yang, Junhong Zhou, Wenlong Wu, Xiaochen Chen, Jun Liu, Bin-Bin Gao, Chengjie Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/110](https://doi.org/10.24963/ijcai.2023/110)

**Abstract**:

Weak feature representation problem has influenced the performance of few-shot classification task for a long time. To alleviate this problem, recent researchers build connections between support and query instances through embedding patch features to generate discriminative representations. However, we observe that there exists semantic mismatches (foreground/ background) among these local patches, because the location and size of the target object are not fixed. What is worse, these mismatches result in unreliable similarity confidences, and complex dense connection exacerbates the problem. According to this, we propose a novel Clustered-patch Element Connection (CEC) layer to correct the mismatch problem. The CEC layer leverages Patch Cluster and Element Connection operations to collect and establish reliable connections with high similarity patch features, respectively. Moreover, we propose a CECNet, including CEC layer based attention module and distance metric. The former is utilized to generate a more discriminative representation benefiting from the global clustered-patch features, and the latter is introduced to reliably measure the similarity between pair-features. Extensive experiments demonstrate that our CECNet outperforms the state-of-the-art methods on classification benchmark. Furthermore, our CEC approach can be extended into few-shot segmentation and detection tasks, which achieves competitive performances.

----

## [110] RaMLP: Vision MLP via Region-aware Mixing

**Authors**: *Shenqi Lai, Xi Du, Jia Guo, Kaipeng Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/111](https://doi.org/10.24963/ijcai.2023/111)

**Abstract**:

Recently, MLP-based architectures achieved impressive results in image classification against CNNs and ViTs. However, there is an obvious limitation in that their parameters are related to image sizes, allowing them to process only fixed image sizes. Therefore, they cannot directly adapt dense prediction tasks (e.g., object detection and semantic segmentation) where images are of various sizes. Recent methods tried to address it but brought two new problems, long-range dependencies or important visual cues are ignored. This paper presents a new MLP-based architecture, Region-aware MLP (RaMLP), to satisfy various vision tasks and address the above three problems. In particular, we propose a well-designed module, Region-aware Mixing (RaM). RaM captures important local information and further aggregates these important visual clues. Based on RaM, RaMLP achieves a global receptive field even in one block. It is worth noting that, unlike most existing MLP-based architectures that adopt the same spatial weights to all samples, RaM is region-aware and adaptively determines weights to extract region-level features better. Impressively, our RaMLP outperforms state-of-the-art ViTs, CNNs, and MLPs on both ImageNet-1K image classification and downstream dense prediction tasks, including MS-COCO object detection, MS-COCO instance segmentation, and ADE20K semantic segmentation. In particular, RaMLP outperforms MLPs by a large margin (around 1.5% Apb or 1.0% mIoU) on dense prediction tasks. The training code could be found at https://github.com/xiaolai-sqlai/RaMLP.

----

## [111] Deep Unfolding Convolutional Dictionary Model for Multi-Contrast MRI Super-resolution and Reconstruction

**Authors**: *Pengcheng Lei, Faming Fang, Guixu Zhang, Ming Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/112](https://doi.org/10.24963/ijcai.2023/112)

**Abstract**:

Magnetic resonance imaging (MRI) tasks often involve multiple contrasts. Recently, numerous deep learning-based multi-contrast MRI super-resolution (SR) and reconstruction methods have been proposed to explore the complementary information from the multi-contrast images. However, these methods either construct parameter-sharing networks or manually design fusion rules, failing to accurately model the correlations between multi-contrast images and lacking certain interpretations. In this paper, we propose a multi-contrast convolutional dictionary (MC-CDic) model under the guidance of the optimization algorithm with a well-designed data fidelity term. Specifically, we bulid an observation model for the multi-contrast MR images to explicitly model the multi-contrast images as common features and unique features. In this way, only the useful information in the reference image can be transferred to the target image, while the inconsistent information will be ignored. We employ the proximal gradient algorithm to optimize the model and unroll the iterative steps into a deep CDic model. Especially, the proximal operators are replaced by learnable ResNet. In addition, multi-scale dictionaries are introduced to further improve the model performance. We test our MC-CDic model on multi-contrast MRI SR and reconstruction tasks. Experimental results demonstrate the superior performance of the proposed MC-CDic model against existing SOTA methods. Code is available at https://github.com/lpcccc-cv/MC-CDic.

----

## [112] CiT-Net: Convolutional Neural Networks Hand in Hand with Vision Transformers for Medical Image Segmentation

**Authors**: *Tao Lei, Rui Sun, Xuan Wang, Yingbo Wang, Xi He, Asoke K. Nandi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/113](https://doi.org/10.24963/ijcai.2023/113)

**Abstract**:

The hybrid architecture of convolutional neural networks (CNNs) and Transformer are very popular for medical image segmentation. However, it suffers from two challenges. First, although a CNNs branch can capture the local image features using vanilla convolution, it cannot achieve adaptive feature learning. Second, although a Transformer branch can capture the global features, it ignores the channel and cross-dimensional self-attention, resulting in a low segmentation accuracy on complex-content images. To address these challenges, we propose a novel hybrid architecture of convolutional neural networks hand in hand with vision Transformers (CiT-Net) for medical image segmentation. Our network has two advantages. First, we design a dynamic deformable convolution and apply it to the CNNs branch, which overcomes the weak feature extraction ability due to fixed-size convolution kernels and the stiff design of sharing kernel parameters among different inputs. Second, we design a shifted-window adaptive complementary attention module and a compact convolutional projection. We apply them to the Transformer branch to learn the cross-dimensional long-term dependency for medical images. Experimental results show that our CiT-Net provides better medical image segmentation results than popular SOTA methods. Besides, our CiT-Net requires lower parameters and less computational costs and does not rely on pre-training. The code is publicly available at https://github.com/SR0920/CiT-Net.

----

## [113] WBFlow: Few-shot White Balance for sRGB Images via Reversible Neural Flows

**Authors**: *Chunxiao Li, Xuejing Kang, Anlong Ming*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/114](https://doi.org/10.24963/ijcai.2023/114)

**Abstract**:

The sRGB white balance methods aim to correct  the nonlinear color cast of sRGB images without  accessing raw values.  Although existing methods  have achieved increasingly better results, their generalization  to sRGB images from multiple cameras  is still under explored.  In this paper, we propose  the network named WBFlow that not only performs  superior white balance for sRGB images but also  generalizes well to multiple cameras.  Specifically,  we take advantage of neural flow to ensure the reversibility  of WBFlow, which enables lossless rendering  of color cast sRGB images back to pseudo  raw features for linear white balancing and thus  achieves superior performance.  Furthermore, inspired  by camera transformation approaches, we  have designed a camera transformation (CT) in  pseudo raw feature space to generalize WBFlow  for different cameras via few shot learning.  By  utilizing a few sRGB images from an untrained  camera, our WBFlow can perform well on this  camera by learning the camera specific parameters  of CT.  Extensive experiments show that WBFlow  achieves superior camera generalization and accuracy  on three public datasets as well as our rendered  multiple camera sRGB dataset.  Our code is available  at https://github.com/ChunxiaoLe/WBFlow.

----

## [114] Learning Attention from Attention: Efficient Self-Refinement Transformer for Face Super-Resolution

**Authors**: *Guanxin Li, Jingang Shi, Yuan Zong, Fei Wang, Tian Wang, Yihong Gong*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/115](https://doi.org/10.24963/ijcai.2023/115)

**Abstract**:

Recently, Transformer-based architecture has been introduced into face super-resolution task due to its advantage in capturing long-range dependencies. However, these approaches tend to integrate global information in a large searching region, which neglect to focus on the most relevant information and induce blurry effect by the irrelevant textures. Some improved methods simply constrain self-attention in a local window to suppress the useless information. But it also limits the capability of recovering high-frequency details when flat areas dominate the local searching window. To improve the above issues, we propose a novel self-refinement mechanism which could adaptively achieve texture-aware reconstruction in a coarse-to-fine procedure. Generally, the primary self-attention is first conducted to reconstruct the coarse-grained textures and detect the fine-grained regions required further compensation. Then, region selection attention is performed to refine the textures on these key regions. Since self-attention considers the channel information on tokens equally, we employ a dual-branch feature integration module to privilege the important channels in feature extraction. Furthermore, we design the wavelet fusion module which integrate shallow-layer structure and deep-layer detailed feature to recover realistic face images in frequency domain. Extensive experiments demonstrate the effectiveness on a variety of datasets.

----

## [115] TG-VQA: Ternary Game of Video Question Answering

**Authors**: *Hao Li, Peng Jin, Zesen Cheng, Songyang Zhang, Kai Chen, Zhennan Wang, Chang Liu, Jie Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/116](https://doi.org/10.24963/ijcai.2023/116)

**Abstract**:

Video question answering aims at answering a question about the video content by reasoning the alignment semantics within them. However, since relying heavily on human instructions, i.e., annotations or priors, current contrastive learning-based VideoQA methods remains challenging to perform fine-grained visual-linguistic alignments. In this work, we innovatively resort to game theory, which can simulate complicated relationships among multiple players with specific interaction strategies, e.g., video, question, and answer as ternary players, to achieve fine-grained alignment for VideoQA task. Specifically, we carefully design a VideoQA-specific interaction strategy to tailor the characteristics of VideoQA, which can mathematically generate the fine-grained visual-linguistic alignment label without label-intensive efforts. Our TG-VQA outperforms existing state-of-the-art by a large margin (more than 5%) on long-term and short-term VideoQA datasets, verifying its effectiveness and generalization ability. Thanks to the guidance of game-theoretic interaction, our model impressively convergences well on limited data (10^4 videos), surpassing most of those pre-trained on large-scale data (10^7 videos).

----

## [116] Contact2Grasp: 3D Grasp Synthesis via Hand-Object Contact Constraint

**Authors**: *Haoming Li, Xinzhuo Lin, Yang Zhou, Xiang Li, Yuchi Huo, Jiming Chen, Qi Ye*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/117](https://doi.org/10.24963/ijcai.2023/117)

**Abstract**:

3D grasp synthesis generates grasping poses given an input object. Existing works tackle the problem by learning a direct mapping from objects to the distributions of grasping poses. However, because the physical contact is sensitive to small changes in pose, the high-nonlinear mapping between 3D object representation to valid poses is considerably non-smooth, leading to poor generation efficiency and restricted generality. To tackle the challenge, we introduce an intermediate variable for grasp contact areas to constrain the grasp generation; in other words, we factorize the mapping into two sequential stages by assuming that grasping poses are fully constrained given contact maps: 1) we first learn contact map distributions to generate the potential contact maps for grasps; 2) then learn a mapping from the contact maps to the grasping poses. Further, we propose a penetration-aware optimization with the generated contacts as a consistency constraint for grasp refinement. Extensive validations on two public datasets show that our method outperforms state-of-the-art methods regarding grasp generation on various metrics.

----

## [117] ALL-E: Aesthetics-guided Low-light Image Enhancement

**Authors**: *Ling Li, Dong Liang, Yuanhang Gao, Sheng-Jun Huang, Songcan Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/118](https://doi.org/10.24963/ijcai.2023/118)

**Abstract**:

Evaluating the performance of low-light image enhancement (LLE) is highly subjective, thus making integrating human preferences into image enhancement a necessity. Existing methods fail to consider this and present a series of potentially valid heuristic criteria for training enhancement models. In this paper, we propose a new paradigm, i.e., aesthetics-guided low-light image enhancement (ALL-E), which introduces aesthetic preferences to LLE and motivates training in a reinforcement learning framework with an aesthetic reward. Each pixel, functioning as an agent, refines itself by recursive actions, i.e., its corresponding adjustment curve is estimated sequentially. Extensive experiments show that integrating aesthetic assessment improves both subjective experience and objective evaluation. Our results on various benchmarks demonstrate the superiority of ALL-E over state-of-the-art methods. Source code: https://dongl-group.github.io/project pages/ALLE.html

----

## [118] Local-Global Transformer Enhanced Unfolding Network for Pan-sharpening

**Authors**: *Mingsong Li, Yikun Liu, Tao Xiao, Yuwen Huang, Gongping Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/119](https://doi.org/10.24963/ijcai.2023/119)

**Abstract**:

Pan-sharpening aims to increase the spatial resolution of the low-resolution multispectral (LrMS) image with the guidance of the corresponding panchromatic (PAN) image. Although deep learning (DL)-based pan-sharpening methods have achieved promising performance, most of them have a two-fold deficiency. For one thing, the universally adopted black box principle limits the model interpretability. For another thing, existing DL-based methods fail to efficiently capture local and global dependencies at the same time, inevitably limiting the overall performance. To address these mentioned issues, we first formulate the degradation process of the high-resolution multispectral (HrMS) image as a unified variational optimization problem, and alternately solve its data and prior subproblems by the designed iterative proximal gradient descent (PGD) algorithm. Moreover, we customize a Local-Global Transformer (LGT) to simultaneously model local and global dependencies, and further formulate an LGT-based prior module for image denoising. Besides the prior module, we also design a lightweight data module. Finally, by serially integrating the data and prior modules in each iterative stage, we unfold the iterative algorithm into a stage-wise unfolding network, Local-Global Transformer Enhanced Unfolding Network (LGTEUN), for the interpretable MS pan-sharpening. Comprehensive experimental results on three satellite data sets demonstrate the effectiveness and efficiency of LGTEUN compared with state-of-the-art (SOTA) methods. The source code is available at https://github.com/lms-07/LGTEUN.

----

## [119] PowerBEV: A Powerful Yet Lightweight Framework for Instance Prediction in Bird's-Eye View

**Authors**: *Peizheng Li, Shuxiao Ding, Xieyuanli Chen, Niklas Hanselmann, Marius Cordts, Juergen Gall*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/120](https://doi.org/10.24963/ijcai.2023/120)

**Abstract**:

Accurately perceiving instances and predicting their future motion are key tasks for autonomous vehicles, enabling them to navigate safely in complex urban traffic. While bird’s-eye view (BEV) representations are commonplace in perception for autonomous driving, their potential in a motion prediction setting is less explored. Existing approaches for BEV instance prediction from surround cameras rely on a multi-task auto-regressive setup coupled with complex post-processing to predict future instances in a spatio-temporally consistent manner. In this paper, we depart from this paradigm and propose an efficient novel end-to-end framework named PowerBEV, which differs in several design choices aimed at reducing the inherent redundancy in previous methods. First, rather than predicting the future in an auto-regressive fashion, PowerBEV uses a parallel, multi-scale module built from lightweight 2D convolutional networks. Second, we show that segmentation and centripetal backward flow are sufficient for prediction, simplifying previous multi-task objectives by eliminating redundant output modalities. Building on this output representation, we propose a simple, flow warping-based post-processing approach which produces more stable instance associations across time. Through this lightweight yet powerful design, PowerBEV outperforms state-of-the-art baselines on the NuScenes Dataset and poses an alternative paradigm for BEV instance prediction. We made our code publicly available at: https://github.com/EdwardLeeLPZ/PowerBEV.

----

## [120] On Efficient Transformer-Based Image Pre-training for Low-Level Vision

**Authors**: *Wenbo Li, Xin Lu, Shengju Qian, Jiangbo Lu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/121](https://doi.org/10.24963/ijcai.2023/121)

**Abstract**:

Pre-training has marked numerous state of the arts in high-level computer vision, while few attempts have ever been made to investigate how pre-training acts in image processing systems. In this paper, we tailor transformer-based pre-training regimes that boost various low-level tasks. To comprehensively diagnose the influence of pre-training, we design a whole set of principled evaluation tools that uncover its effects on internal representations. The observations demonstrate that pre-training plays strikingly different roles in low-level tasks. For example, pre-training introduces more local information to intermediate layers in super-resolution (SR), yielding significant performance gains, while pre-training hardly affects internal feature representations in denoising, resulting in limited gains. Further, we explore different methods of pre-training, revealing that multi-related-task pre-training is more effective and data-efficient than other alternatives. Finally, we extend our study to varying data scales and model sizes, as well as comparisons between transformers and CNNs. Based on the study, we successfully develop state-of-the-art models for multiple low-level tasks.

----

## [121] Compositional Zero-Shot Artistic Font Synthesis

**Authors**: *Xiang Li, Lei Wu, Changshuo Wang, Lei Meng, Xiangxu Meng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/122](https://doi.org/10.24963/ijcai.2023/122)

**Abstract**:

Recently, many researchers have made remarkable achievements in the field of artistic font synthesis, with impressive glyph style and effect style in the results. However, due to less exploration in style disentanglement, it is difficult for existing methods to envision a kind of unseen style (glyph-effect) compositions of artistic font, and thus can only learn the seen style compositions. To solve this problem, we propose a novel compositional zero-shot artistic font synthesis gan (CAFS-GAN), which allows the synthesis of unseen style compositions by exploring the visual independence and joint compatibility of encoding semantics between glyph and effect. Specifically, we propose two contrast-based style encoders to achieve style disentanglement due to glyph and effect intertwining in the image. Meanwhile, to preserve more glyph and effect detail, we propose a generator based on hierarchical dual styles AdaIN to reorganize content-styles representations from structure to texture gradually. Extensive experiments demonstrate the superiority of our model in generating high-quality artistic font images with unseen style compositions against other state-of-the-art methods. The source code and data is available at moonlight03.github.io/CAFS-GAN/.

----

## [122] VS-Boost: Boosting Visual-Semantic Association for Generalized Zero-Shot Learning

**Authors**: *Xiaofan Li, Yachao Zhang, Shiran Bian, Yanyun Qu, Yuan Xie, Zhongchao Shi, Jianping Fan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/123](https://doi.org/10.24963/ijcai.2023/123)

**Abstract**:

Unlike conventional zero-shot learning (CZSL) which only focuses on the recognition of unseen classes by using the classifier trained on seen classes and semantic embeddings, generalized zero-shot learning (GZSL) aims at recognizing both the seen and unseen classes, so it is more challenging due to the extreme training imbalance. Recently, some feature generation methods introduce metric learning to enhance the discriminability of visual features. Although these methods achieve good results, they focus only on metric learning in the visual feature space to enhance features and ignore the association between the feature space and the semantic space. Since the GZSL method uses semantics as prior knowledge to migrate visual knowledge to unseen classes, the consistency between visual space and semantic space is critical. To this end, we propose relational metric learning which can relate the metrics in the two spaces and make the distribution of the two spaces more consistent. Based on the generation method and relational metric learning, we proposed a novel GZSL method, termed VS-Boost, which can effectively boost the association between vision and semantics. The experimental results demonstrate that our method is effective and achieves significant gains on five benchmark datasets compared with the state-of-the-art methods.

----

## [123] Locate, Refine and Restore: A Progressive Enhancement Network for Camouflaged Object Detection

**Authors**: *Xiaofei Li, Jiaxin Yang, Shuohao Li, Jun Lei, Jun Zhang, Dong Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/124](https://doi.org/10.24963/ijcai.2023/124)

**Abstract**:

Camouflaged Object Detection (COD) aims to segment objects that blend in with their surroundings. Most existing methods mainly tackle this issue by a single-stage framework, which tends to degrade performance in the face of small objects, low-contrast objects and objects with diverse appearances. In this paper, we propose a novel Progressive Enhancement Network (PENet) for COD by imitating the human visual detection system, which follows a three-stage detection process: locate objects, refine textures and restore boundary. Specifically, our PENet contains three key modules, i.e., the object location module (OLM), the group attention module (GAM) and the context feature restoration module (CFRM). The OLM is designed to position the object globally, the GAM is developed to refine both high-level semantic and low-level texture feature representation, and the CFRM is leveraged to effectively aggregate multi-level features for progressively restoring the clear boundary. Extensive results demonstrate that our PENet significantly outperforms 32 state-of-the-art methods on four widely used benchmark datasets

----

## [124] SGAT4PASS: Spherical Geometry-Aware Transformer for PAnoramic Semantic Segmentation

**Authors**: *Xuewei Li, Tao Wu, Zhongang Qi, Gaoang Wang, Ying Shan, Xi Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/125](https://doi.org/10.24963/ijcai.2023/125)

**Abstract**:

As an important and challenging problem in computer vision, PAnoramic Semantic Segmentation (PASS) gives complete scene perception based on an ultra-wide angle of view. Usually, prevalent PASS methods with 2D panoramic image input focus on solving image distortions but lack consideration of the 3D properties of original 360 degree data. Therefore, their performance will drop a lot when inputting panoramic images with the 3D disturbance. To be more robust to 3D disturbance, we propose our Spherical Geometry-Aware Transformer for PAnoramic Semantic Segmentation (SGAT4PASS), considering 3D spherical geometry knowledge. Specifically, a spherical geometry-aware framework is proposed for PASS. It includes three modules, i.e., spherical geometry-aware image projection, spherical deformable patch embedding, and a panorama-aware loss, which takes input images with 3D disturbance into account, adds a spherical geometry-aware constraint on the existing deformable patch embedding, and indicates the pixel density of original 360 degree data, respectively. Experimental results on Stanford2D3D Panoramic datasets show that SGAT4PASS significantly improves performance and robustness, with approximately a 2% increase in mIoU, and when small 3D disturbances occur in the data, the stability of our performance is improved by an order of magnitude. Our code and supplementary material are available at https://github.com/TencentARC/SGAT4PASS.

----

## [125] Image Composition with Depth Registration

**Authors**: *Zan Li, Wencheng Wang, Fei Hou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/126](https://doi.org/10.24963/ijcai.2023/126)

**Abstract**:

Handling occlusions is still a challenging problem for image composition. It always requires the source contents to be completely in front of the target contents or needs manual interventions to adjust occlusions, which is very tedious. Though several methods have suggested exploiting priors or learning techniques for promoting occlusion determination, their potentials are much limited. This paper addresses the challenge by presenting a depth registration method for merging the source contents seamlessly into the 3D space that the target image represents. Thus, the occlusions between the source contents and target contents can be conveniently handled through pixel-wise depth comparisons, allowing the user to more efficiently focus on the designs for image composition. Experimental results show that we can conveniently handle occlusions in image composition and improve efficiency by about 4 times compared to Photoshop.

----

## [126] Complete Instances Mining for Weakly Supervised Instance Segmentation

**Authors**: *Zecheng Li, Zening Zeng, Yuqi Liang, Jin-Gang Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/127](https://doi.org/10.24963/ijcai.2023/127)

**Abstract**:

Weakly supervised instance segmentation (WSIS) using only image-level labels is a challenging task due to the difficulty of aligning coarse annotations with the finer task. However, with the advancement of deep neural networks (DNNs), WSIS has garnered significant attention. Following a proposal-based paradigm, we encounter a redundant segmentation problem resulting from a single instance being represented by multiple proposals. For example, we feed a picture of a dog and proposals into the network and expect to output only one proposal containing a dog, but the network outputs multiple proposals. To address this problem, we propose a novel approach for WSIS that focuses on the online refinement of complete instances through the use of MaskIoU heads to predict the integrity scores of proposals and a Complete Instances Mining (CIM) strategy to explicitly model the redundant segmentation problem and generate refined pseudo labels. Our approach allows the network to become aware of multiple instances and complete instances, and we further improve its robustness through the incorporation of an Anti-noise strategy. Empirical evaluations on the PASCAL VOC 2012 and MS COCO datasets demonstrate that our method achieves state-of-the-art performance with a notable margin. Our implementation will be made available at https://github.com/ZechengLi19/CIM.

----

## [127] Analyzing and Combating Attribute Bias for Face Restoration

**Authors**: *Zelin Li, Dan Zeng, Xiao Yan, Qiaomu Shen, Bo Tang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/128](https://doi.org/10.24963/ijcai.2023/128)

**Abstract**:

Face restoration (FR) recovers high resolution (HR) faces from low resolution (LR) faces and is challenging due to its ill-posed nature. With years of development, existing methods can produce quality HR faces with realistic details. However, we observe that key facial attributes (e.g., age and gender) of the restored faces could be dramatically different from the LR faces and call this phenomenon attribute bias, which is fatal when using FR for applications such as surveillance and security. Thus, we argue that FR should consider not only image quality as in existing works but also attribute bias. To this end, we thoroughly analyze attribute bias with extensive experiments and find that two major causes are the lack of attribute information in LR faces and bias in the training data. Moreover, we propose the DebiasFR framework to produce HR faces with high image quality and accurate facial attributes. The key design is to explicitly model the facial attributes, which also allows to adjust facial attributes for the output HR faces. Experiment results show that DebiasFR has comparable image quality but significantly smaller attribute bias when compared with state-of-the-art FR methods.

----

## [128] A Large-Scale Film Style Dataset for Learning Multi-frequency Driven Film Enhancement

**Authors**: *Zinuo Li, Xuhang Chen, Shuqiang Wang, Chi-Man Pun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/129](https://doi.org/10.24963/ijcai.2023/129)

**Abstract**:

Film, a classic image style, is culturally significant to the whole photographic industry since it marks the birth of photography. However, film photography is time-consuming and expensive, necessitating a more efficient method for collecting film-style photographs. Numerous datasets that have emerged in the field of image enhancement so far are not film-specific. In order to facilitate film-based image stylization research, we construct FilmSet, a large-scale and high-quality film style dataset. Our dataset includes three different film types and more than 5000 in-the-wild high resolution images. Inspired by the features of FilmSet images, we propose a novel framework called FilmNet based on Laplacian Pyramid for stylizing images across frequency bands and achieving film style outcomes. Experiments reveal that the performance of our model is superior than state-of-the-art techniques. The link of our dataset and code is https://github.com/CXH-Research/FilmNet.

----

## [129] U-Match: Two-view Correspondence Learning with Hierarchy-aware Local Context Aggregation

**Authors**: *Zizhuo Li, Shihua Zhang, Jiayi Ma*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/130](https://doi.org/10.24963/ijcai.2023/130)

**Abstract**:

Local context capturing has become the core factor for achieving leading performance in two-view correspondence learning. Recent advances have devised various local context extractors whereas typically adopting explicit neighborhood relation modeling that is restricted and inflexible. To address this issue, we introduce U-Match, an attentional graph neural network that has the flexibility to enable implicit local context awareness at multiple levels. Specifically, a hierarchy-aware graph representation (HAGR) module is designed and fleshed out by local context pooling and unpooling operations. The former encodes local context by adaptively sampling a set of nodes to form a coarse-grained graph, while the latter decodes local context by recovering the coarsened graph back to its original size. Moreover, an orthogonal fusion module is proposed for the collaborative use of HAGR module, which integrates complementary local and global information into compact feature representations without redundancy. Extensive experiments on different visual tasks prove that our method significantly surpasses the state-of-the-arts. In particular, U-Match attains an AUC at 5 degree threshold of 60.53% on the challenging YFCC100M dataset without RANSAC, outperforming the strongest prior model by 8.61 absolute percentage points. Our code is publicly available at https://github.com/ZizhuoLi/U-Match.

----

## [130] GTR: A Grafting-Then-Reassembling Framework for Dynamic Scene Graph Generation

**Authors**: *Jiafeng Liang, Yuxin Wang, Zekun Wang, Ming Liu, Ruiji Fu, Zhongyuan Wang, Bing Qin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/131](https://doi.org/10.24963/ijcai.2023/131)

**Abstract**:

Dynamic scene graph generation aims to identify visual relationships (subject-predicate-object) in frames based on spatio-temporal contextual information in the video. Previous work implicitly models the spatio-temporal interaction simultaneously, which leads to entanglement of spatio-temporal contextual information. To this end, we propose a Grafting-Then-Reassembling framework (GTR), which explicitly extracts intra-frame spatial information and inter-frame temporal information in two separate stages to decouple spatio-temporal contextual information. Specifically, we first graft a static scene graph generation model to generate static visual relationships within frames. Then we propose the temporal dependency model to extract the temporal dependencies across frames, and explicitly reassemble static visual relationships into dynamic scene graphs. Experimental results show that GTR achieves the state-of-the-art performance on Action Genome dataset. Further analyses reveal that the reassembling stage is crucial to the success of our framework.

----

## [131] Low-Confidence Samples Mining for Semi-supervised Object Detection

**Authors**: *Guandu Liu, Fangyuan Zhang, Tianxiang Pan, Jun-Hai Yong, Bin Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/132](https://doi.org/10.24963/ijcai.2023/132)

**Abstract**:

Reliable pseudo labels from unlabeled data play a key role in semi-supervised object detection (SSOD). However, the state-of-the-art SSOD methods all rely on pseudo labels with high confidence, which ignore valuable pseudo labels with lower confidence. Additionally, the insufficient excavation for unlabeled data results in an excessively low recall rate thus hurting the network training. In this paper, we propose a novel Low-confidence Samples Mining (LSM) method to utilize low confidence pseudo labels efficiently. Specifically, we develop an additional pseudo information mining (PIM) branch on account of low-resolution feature maps to extract reliable large area instances, the IoUs of which are higher than small area ones. Owing to the complementary predictions between PIM and the main branch, we further design self-distillation (SD) to compensate for both in a mutually learning manner. Meanwhile, the extensibility of the above approaches enables our LSM to apply to Faster-RCNN and Deformable-DETR respectively. On the MS-COCO benchmark, our method achieves 3.54% mAP improvement over state-of-the-art methods under 5% labeling ratios.

----

## [132] Boosting Decision-Based Black-Box Adversarial Attack with Gradient Priors

**Authors**: *Han Liu, Xingshuo Huang, Xiaotong Zhang, Qimai Li, Fenglong Ma, Wei Wang, Hongyang Chen, Hong Yu, Xianchao Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/133](https://doi.org/10.24963/ijcai.2023/133)

**Abstract**:

Decision-based methods have shown to be effective in black-box adversarial attacks, as they can obtain satisfactory performance and only require to access the final model prediction. Gradient estimation is a critical step in black-box adversarial attacks, as it will directly affect the query efficiency. Recent works have attempted to utilize gradient priors to facilitate score-based methods to obtain better results. However, these gradient priors still suffer from the edge gradient discrepancy issue and the successive iteration gradient direction issue, thus are difficult to simply extend to decision-based methods. In this paper, we propose a novel Decision-based Black-box Attack framework with Gradient Priors (DBA-GP), which seamlessly integrates the data-dependent gradient prior and time-dependent prior into the gradient estimation procedure. First, by leveraging the joint bilateral filter to deal with each random perturbation, DBA-GP can guarantee that the generated perturbations in edge locations are hardly smoothed, i.e., alleviating the edge gradient discrepancy, thus remaining the characteristics of the original image as much as possible. Second, by utilizing a new gradient updating strategy to automatically adjust the successive iteration gradient direction, DBA-GP can accelerate the convergence speed, thus improving the query efficiency. Extensive experiments have demonstrated that the proposed method outperforms other strong baselines significantly.

----

## [133] APR: Online Distant Point Cloud Registration through Aggregated Point Cloud Reconstruction

**Authors**: *Quan Liu, Yunsong Zhou, Hongzi Zhu, Shan Chang, Minyi Guo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/134](https://doi.org/10.24963/ijcai.2023/134)

**Abstract**:

For many driving safety applications, it is of great importance to accurately register LiDAR point clouds generated on distant moving vehicles. However, such point clouds have extremely different point density and sensor perspective on the same object, making registration on such point clouds very hard. In this paper, we propose a novel feature extraction framework, called APR, for online distant point cloud registration. Specifically, APR leverages an autoencoder design, where the autoencoder reconstructs a denser aggregated point cloud with several frames instead of the original single input point cloud. Our design forces the encoder to extract features with rich local geometry information based on one single input point cloud. Such features are then used for online distant point cloud registration. We conduct extensive experiments against state-of-the-art (SOTA) feature extractors on KITTI and nuScenes datasets. Results show that APR outperforms all other extractors by a large margin, increasing average registration recall of SOTA extractors by 7.1% on LoKITTI and 4.6% on LoNuScenes. Code is available at https://github.com/liuQuan98/APR.

----

## [134] Cross-Domain Facial Expression Recognition via Disentangling Identity Representation

**Authors**: *Tong Liu, Jing Li, Jia Wu, Lefei Zhang, Shanshan Zhao, Jun Chang, Jun Wan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/135](https://doi.org/10.24963/ijcai.2023/135)

**Abstract**:

Most existing cross-domain facial expression recognition (FER) works require target domain data to assist the model in analyzing distribution shifts to overcome negative effects. However, it is often hard to obtain expression images of the target domain in practical applications. Moreover, existing methods suffer from the interference of identity information, thus limiting the discriminative ability of the expression features. We exploit the idea of domain generalization (DG) and propose a representation disentanglement model to address the above problems. Specifically, we learn three independent potential subspaces corresponding to the domain, expression, and identity information from facial images. Meanwhile, the extracted expression and identity features are recovered as Fourier phase information reconstructed images, thereby ensuring that the high-level semantics of images remain unchanged after disentangling the domain information. Our proposed method can disentangle expression features from expression-irrelevant ones (i.e., identity and domain features). Therefore, the learned expression features exhibit sufficient domain invariance and discriminative ability. We conduct experiments with different settings on multiple benchmark datasets, and the results show that our method achieves superior performance compared with state-of-the-art methods.

----

## [135] Adaptive Sparse ViT: Towards Learnable Adaptive Token Pruning by Fully Exploiting Self-Attention

**Authors**: *Xiangcheng Liu, Tianyi Wu, Guodong Guo*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/136](https://doi.org/10.24963/ijcai.2023/136)

**Abstract**:

Vision transformer has emerged as a new paradigm in computer vision, showing excellent performance while accompanied by expensive computational cost. Image token pruning is one of the main approaches for ViT compression, due to the facts that the complexity is quadratic with respect to the token number, and many tokens containing only background regions do not truly contribute to the final prediction. Existing works either rely on additional modules to score the importance of individual tokens, or implement a fixed ratio pruning strategy for different input instances. In this work, we propose an adaptive sparse token pruning framework with a minimal cost. Specifically, we firstly propose an inexpensive attention head importance weighted class attention scoring mechanism. Then, learnable parameters are inserted as thresholds to distinguish informative tokens from unimportant ones. By comparing token attention scores and thresholds, we can discard useless tokens hierarchically and thus accelerate inference. The learnable thresholds are optimized in budget-aware training to balance accuracy and complexity, performing the corresponding pruning configurations for different input instances. Extensive experiments demonstrate the effectiveness of our approach. Our method improves the throughput of DeiT-S by 50% and brings only 0.2% drop in top-1 accuracy, which achieves a better trade-off between accuracy and latency than the previous methods.

----

## [136] Sph2Pob: Boosting Object Detection on Spherical Images with Planar Oriented Boxes Methods

**Authors**: *Xinyuan Liu, Hang Xu, Bin Chen, Qiang Zhao, Yike Ma, Chenggang Yan, Feng Dai*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/137](https://doi.org/10.24963/ijcai.2023/137)

**Abstract**:

Object detection on panoramic/spherical images has been developed rapidly in the past few years, where IoU-calculator is a fundamental part of various detector components, i.e. Label Assignment, Loss and NMS. Due to the low efficiency and non-differentiability of spherical Unbiased IoU, spherical approximate IoU methods have been proposed recently. We find that the key of these approximate methods is to map spherical boxes to planar boxes. However, there exists two problems in these methods: (1) they do not eliminate the influence of panoramic image distortion; (2) they break the original pose between bounding boxes. They lead to the low accuracy of these methods. Taking the two problems into account, we propose a new sphere-plane boxes transform, called Sph2Pob. Based on the Sph2Pob, we propose (1) an differentiable IoU, Sph2Pob-IoU, for spherical boxes with low time-cost and high accuracy and (2) an agent Loss, Sph2Pob-Loss, for spherical detection with high flexibility and expansibility. Extensive experiments verify the effectiveness and generality of our approaches, and Sph2Pob-IoU and Sph2Pob-Loss together boost the performance of spherical detectors. The source code is available at https://github.com/AntXinyuan/sph2pob.

----

## [137] Bi-level Dynamic Learning for Jointly Multi-modality Image Fusion and Beyond

**Authors**: *Zhu Liu, Jinyuan Liu, Guanyao Wu, Long Ma, Xin Fan, Risheng Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/138](https://doi.org/10.24963/ijcai.2023/138)

**Abstract**:

Recently, multi-modality scene perception tasks, e.g.,  image fusion and scene understanding, have attracted widespread attention for intelligent vision systems. However, early efforts always consider boosting a single task unilaterally and neglecting others, seldom investigating their underlying connections for joint promotion. To overcome these limitations, we establish the hierarchical dual tasks-driven deep model to bridge these tasks. Concretely, we firstly construct an image fusion module to fuse complementary characteristics and cascade dual task-related modules, including a discriminator for visual effects and a semantic network for feature measurement. 
We provide a  bi-level perspective to formulate image fusion and follow-up downstream tasks. To incorporate distinct task-related responses for image fusion, we consider image fusion as a primary goal and dual modules as learnable constraints. Furthermore, we develop an efficient first-order approximation to compute corresponding gradients and present dynamic weighted aggregation to balance the gradients for fusion learning. Extensive experiments demonstrate the superiority of our method, which not only produces visually pleasant fused results but also realizes significant promotion for detection and segmentation than the state-of-the-art approaches.

----

## [138] Non-Lambertian Multispectral Photometric Stereo via Spectral Reflectance Decomposition

**Authors**: *Jipeng Lv, Heng Guo, Guanying Chen, Jinxiu Liang, Boxin Shi*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/139](https://doi.org/10.24963/ijcai.2023/139)

**Abstract**:

Multispectral photometric stereo (MPS) aims at recovering the surface normal of a scene from a single-shot multispectral image captured under multispectral illuminations. Existing MPS methods adopt the Lambertian reflectance model to make the problem tractable, but it greatly limits their application to real-world surfaces. In this paper, we propose a deep neural network named NeuralMPS to solve the MPS problem under non-Lambertian spectral reflectances. Specifically, we present a spectral reflectance decomposition model to disentangle the spectral reflectance into a geometric component and a spectral component. With this decomposition, we show that the MPS problem for surfaces with a uniform material is equivalent to the conventional photometric stereo (CPS) with unknown light intensities. In this way, NeuralMPS reduces the difficulty of the non-Lambertian MPS problem by leveraging the well-studied non-Lambertian CPS methods. Experiments on both synthetic and real-world scenes demonstrate the effectiveness of our method.

----

## [139] Semantic-Aware Generation of Multi-View Portrait Drawings

**Authors**: *Biao Ma, Fei Gao, Chang Jiang, Nannan Wang, Gang Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/140](https://doi.org/10.24963/ijcai.2023/140)

**Abstract**:

Neural radiance fields (NeRF) based methods have shown amazing performance in synthesizing 3D-consistent photographic images, but fail to generate multi-view portrait drawings. The key is that the basic assumption of these methods -- a surface point is consistent when rendered from different views -- doesn't hold for drawings. In a portrait drawing, the appearance of a facial point may changes when viewed from different angles. Besides, portrait drawings usually present little 3D information and suffer from insufficient training data. To combat this challenge, in this paper, we propose a Semantic-Aware GEnerator (SAGE) for synthesizing multi-view portrait drawings. Our motivation is that facial semantic labels are view-consistent and correlate with drawing techniques. We therefore propose to collaboratively synthesize multi-view semantic maps and the corresponding portrait drawings. To facilitate training, we design a semantic-aware domain translator, which generates portrait drawings based on features of photographic faces. In addition, use data augmentation via synthesis to mitigate collapsed results. We apply SAGE to synthesize multi-view portrait drawings in diverse artistic styles. Experimental results show that SAGE achieves significantly superior or highly competitive performance, compared to existing 3D-aware image synthesis methods. The codes are available at https://github.com/AiArt-HDU/SAGE.

----

## [140] Invertible Residual Neural Networks with Conditional Injector and Interpolator for Point Cloud Upsampling

**Authors**: *Aihua Mao, Yaqi Duan, Yu-Hui Wen, Zihui Du, Hongmin Cai, Yong-Jin Liu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/141](https://doi.org/10.24963/ijcai.2023/141)

**Abstract**:

Point clouds obtained by LiDAR and other sensors are usually sparse and irregular. Low-quality point clouds have serious influence on the final performance of downstream tasks. Recently, a point cloud upsampling network with normalizing flows has been proposed to address this problem. However, the network heavily relies on designing specialized architectures to achieve invertibility. In this paper, we propose a novel invertible residual neural network for point cloud upsampling, called PU-INN, which allows unconstrained architectures to learn more expressive feature transformations. Then, we propose a conditional injector to improve nonlinear transformation ability of the neural network while guaranteeing invertibility. Furthermore, a lightweight interpolator is proposed based on semantic similarity distance in the latent space, which can intuitively reflect the interpolation changes in Euclidean space. Qualitative and quantitative results show that our method outperforms the state-of-the-art works in terms of distribution uniformity, proximity-to-surface accuracy, 3D reconstruction quality, and computation efficiency.

----

## [141] Dual Relation Knowledge Distillation for Object Detection

**Authors**: *Zhenliang Ni, Fukui Yang, Shengzhao Wen, Gang Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/142](https://doi.org/10.24963/ijcai.2023/142)

**Abstract**:

Knowledge distillation is an effective method for model compression. However, it is still a challenging topic to apply knowledge distillation to detection tasks. There are two key points resulting in poor distillation performance for detection tasks. One is the serious imbalance between foreground and background features, another one is that small object lacks enough feature representation. To solve the above issues, we propose a new distillation method named dual relation knowledge distillation (DRKD), including pixel-wise relation distillation and instance-wise relation distillation. The pixel-wise relation distillation embeds pixel-wise features in the graph space and applies graph convolution to capture the global pixel relation. By distilling the global pixel relation, the student detector can learn the relation between foreground and background features, and avoid the difficulty of distilling features directly for the feature imbalance issue. Besides, we find that instance-wise relation supplements valuable knowledge beyond independent features for small objects. Thus, the instance-wise relation distillation is designed, which calculates the similarity of different instances to obtain a relation matrix. More importantly, a relation filter module is designed to highlight valuable instance relations. The proposed dual relation knowledge distillation is general and can be easily applied for both one-stage and two-stage detectors. Our method achieves state-of-the-art performance, which improves Faster R-CNN based on ResNet50 from 38.4% to 41.6% mAP and improves RetinaNet based on ResNet50 from 37.4% to 40.3% mAP on COCO 2017.

----

## [142] OSP2B: One-Stage Point-to-Box Network for 3D Siamese Tracking

**Authors**: *Jiahao Nie, Zhiwei He, Yuxiang Yang, Zhengyi Bao, Mingyu Gao, Jing Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/143](https://doi.org/10.24963/ijcai.2023/143)

**Abstract**:

Two-stage point-to-box network acts as a critical role in the recent popular 3D Siamese tracking paradigm, which first generates proposals and then predicts corresponding proposal-wise scores. However, such a network suffers from tedious hyper-parameter tuning and task misalignment, limiting the tracking performance. Towards these concerns, we propose a simple yet effective one-stage point-to-box network for point cloud-based 3D single object tracking. It synchronizes 3D proposal generation and center-ness score prediction by a parallel predictor without tedious hyper-parameters. To guide a task-aligned score ranking of proposals, a center-aware focal loss is proposed to supervise the training of the center-ness branch, which enhances the network's discriminative ability to distinguish proposals of different quality. Besides, we design a binary target classifier to identify target-relevant points. By integrating the derived classification scores with the center-ness scores, the resulting network can effectively suppress interference proposals and further mitigate task misalignment. Finally, we present a novel one-stage Siamese tracker OSP2B equipped with the designed network. Extensive experiments on challenging benchmarks including KITTI and Waymo SOT Dataset show that our OSP2B achieves leading performance with a considerable real-time speed.

----

## [143] SLViT: Scale-Wise Language-Guided Vision Transformer for Referring Image Segmentation

**Authors**: *Shuyi Ouyang, Hongyi Wang, Shiao Xie, Ziwei Niu, Ruofeng Tong, Yen-Wei Chen, Lanfen Lin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/144](https://doi.org/10.24963/ijcai.2023/144)

**Abstract**:

Referring image segmentation aims to segment an object out of an image via a specific language expression. The main concept is establishing global visual-linguistic relationships to locate the object and identify boundaries using details of the image. Recently, various Transformer-based techniques have been proposed to efficiently leverage long-range cross-modal dependencies,  enhancing performance for referring segmentation. However, existing methods consider visual feature extraction and cross-modal fusion separately, resulting in insufficient visual-linguistic alignment in semantic space. In addition, they employ sequential structures and hence lack multi-scale information interaction. To address these limitations, we propose a Scale-Wise Language-Guided Vision Transformer (SLViT) with two appealing designs: (1) Language-Guided Multi-Scale Fusion Attention, a novel attention mechanism module for extracting rich local visual information and modeling global visual-linguistic relationships in an integrated manner. (2) An Uncertain Region Cross-Scale Enhancement module that can identify regions of high uncertainty using linguistic features and refine them via aggregated multi-scale features. We have evaluated our method on three benchmark datasets. The experimental results demonstrate that SLViT surpasses state-of-the-art methods with lower computational cost. The code is publicly available at: https://github.com/NaturalKnight/SLViT.

----

## [144] Active Visual Exploration Based on Attention-Map Entropy

**Authors**: *Adam Pardyl, Grzegorz Rypesc, Grzegorz Kurzejamski, Bartosz Zielinski, Tomasz Trzcinski*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/145](https://doi.org/10.24963/ijcai.2023/145)

**Abstract**:

Active visual exploration addresses the issue of limited sensor capabilities in real-world scenarios, where successive observations are actively chosen based on the environment. To tackle this problem, we introduce a new technique called Attention-Map Entropy (AME). It leverages the internal uncertainty of the transformer-based model to determine the most informative observations. In contrast to existing solutions, it does not require additional loss components, which simplifies the training. Through experiments, which also mimic retina-like sensors, we show that such simplified training significantly improves the performance of reconstruction, segmentation and classification on publicly available datasets.

----

## [145] Answer Mining from a Pool of Images: Towards Retrieval-Based Visual Question Answering

**Authors**: *Abhirama Subramanyam Penamakuri, Manish Gupta, Mithun Das Gupta, Anand Mishra*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/146](https://doi.org/10.24963/ijcai.2023/146)

**Abstract**:

We study visual question answering in a setting where the answer has to be mined from a pool of relevant and irrelevant images given as a context. For such a setting, a model must first retrieve relevant images from the pool and answer the question from these retrieved images. We refer to this problem as retrieval-based visual question answering (or RETVQA in short). The RETVQA is distinctively different and more challenging than the traditionally-studied Visual Question Answering (VQA), where a given question has to be answered with a single relevant image in context. Towards solving the RETVQA task, we propose a unified Multi Image BART (MI-BART) that takes a question and retrieved images using our relevance encoder for free-form fluent answer generation. Further, we introduce the largest dataset in this space, namely RETVQA, which has the following salient features: multi-image and retrieval requirement for VQA, metadata-independent questions over a pool of heterogeneous images, expecting a mix of classification-oriented and open-ended generative answers. Our proposed framework achieves an accuracy of 76.5% and a fluency of 79.3% on the proposed dataset, namely RETVQA and also outperforms state-of-the-art methods by 4.9% and 11.8% on the image segment of the publicly available WebQA dataset on the accuracy and fluency metrics, respectively.

----

## [146] Contour-based Interactive Segmentation

**Authors**: *Polina Popenova, Danil Galeev, Anna Vorontsova, Anton Konushin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/147](https://doi.org/10.24963/ijcai.2023/147)

**Abstract**:

Recent advances in interactive segmentation (IS)
allow speeding up and simplifying image editing
and labeling greatly. The majority of modern IS
approaches accept user input in the form of clicks.
However, using clicks may require too many user
interactions, especially when selecting small ob-
jects, minor parts of an object, or a group of ob-
jects of the same type. In this paper, we consider
such a natural form of user interaction as a loose
contour, and introduce a contour-based IS method.
We evaluate the proposed method on the standard
segmentation benchmarks, our novel UserContours
dataset, and its subset UserContours-G containing
difficult segmentation cases. Through experiments,
we demonstrate that a single contour provides the
same accuracy as multiple clicks, thus reducing the
required amount of user interactions.

----

## [147] XFormer: Fast and Accurate Monocular 3D Body Capture

**Authors**: *Lihui Qian, Xintong Han, Faqiang Wang, Hongyu Liu, Haoye Dong, Zhiwen Li, Huawei Wei, Zhe Lin, Chengbin Jin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/148](https://doi.org/10.24963/ijcai.2023/148)

**Abstract**:

We present XFormer, a novel human mesh and motion capture method that achieves real-time performance on consumer CPUs given only monocular images as input. The proposed network architecture contains two branches: a keypoint branch that estimates 3D human mesh vertices given 2D keypoints, and an image branch that makes prediction directly from the RGB image features. At the core of our method is a cross-modal transformer block that allows information flow across these two branches by modeling the attention between 2D keypoint coordinates and image spatial features. Our architecture is smartly designed, which enables us to train on various types of datasets including images with 2D/3D annotations, images with 3D pseudo labels, and motion capture datasets that do not have associated images. This effectively improves the accuracy and generalization ability of our system. Built on a lightweight backbone (MobileNetV3), our method runs blazing fast (over 30fps on a single CPU core) and still yields competitive accuracy. Furthermore, with a HRNet backbone, XFormer delivers state-of-the-art performance on Huamn3.6 and 3DPW datasets.

----

## [148] ViT-P3DE∗: Vision Transformer Based Multi-Camera Instance Association with Pseudo 3D Position Embeddings

**Authors**: *Minseok Seo, Hyuk-Jae Lee, Xuan Truong Nguyen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/149](https://doi.org/10.24963/ijcai.2023/149)

**Abstract**:

Multi-camera instance association, which identifies identical objects among multiple objects in multi-view images, is challenging due to several harsh constraints. To tackle this problem, most studies have employed CNNs as feature extractors but often fail under such harsh constraints. Inspired by Vision Transformer (ViT), we first develop a pure ViT-based framework for robust feature extraction through self-attention and residual connection. We then propose two novel methods to achieve robust feature learning. First, we introduce learnable pseudo 3D position embeddings (P3DEs) that represent the 3D location of an object in the world coordinate system, which is independent of the harsh constraints. To generate P3DEs, we encode the camera ID and the object's 2D position in the image using embedding tables. We then build a framework that trains P3DEs to represent an object's 3D position in a weakly supervised manner. Second, we also utilize joint patch generation (JPG). During patch generation, JPG considers an object and its surroundings as a single input patch to reinforce the relationship information between two features. Ultimately, experimental results demonstrate that both ViT-P3DE and ViT-P3DE with JPG achieve state-of-the-art performance and significantly outperform existing works, especially when dealing with extremely harsh constraints.

----

## [149] Teaching What You Should Teach: A Data-Based Distillation Method

**Authors**: *Shitong Shao, Huanran Chen, Zhen Huang, Linrui Gong, Shuai Wang, Xinxiao Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/150](https://doi.org/10.24963/ijcai.2023/150)

**Abstract**:

In real teaching scenarios, an excellent teacher always teaches what he (or she) is good at but the student is not. This gives the student the best assistance in making up for his (or her) weaknesses and becoming a good one overall. Enlightened by this, we introduce the "Teaching what you Should Teach" strategy into a knowledge distillation framework, and propose a data-based distillation method named "TST" that searches for desirable augmented samples to assist in distilling more efficiently and rationally. To be specific, we design a neural network-based data augmentation module with priori bias to find out what meets the teacher's strengths but the student's weaknesses, by learning magnitudes and probabilities to generate suitable data samples. By training the data augmentation module and the generalized distillation paradigm alternately, a student model is learned with excellent generalization ability. To verify the effectiveness of our method, we conducted extensive comparative experiments on object recognition, detection, and segmentation tasks. The results on the CIFAR-100, ImageNet-1k, MS-COCO, and Cityscapes datasets demonstrate that our method achieves state-of-the-art performance on almost all teacher-student pairs. Furthermore, we conduct visualization studies to explore what magnitudes and probabilities are needed for the distillation process.

----

## [150] Learning Prototype Classifiers for Long-Tailed Recognition

**Authors**: *Saurabh Sharma, Yongqin Xian, Ning Yu, Ambuj K. Singh*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/151](https://doi.org/10.24963/ijcai.2023/151)

**Abstract**:

The problem of long-tailed recognition (LTR) has received attention in recent years due to the fundamental power-law distribution of objects in the real-world. Most recent works in LTR use softmax classifiers that are biased in that they correlate classifier norm with the amount of training data for a given class. In this work, we show that learning prototype classifiers addresses the biased softmax problem in LTR. Prototype classifiers can deliver promising results simply using Nearest-Class-Mean (NCM), a special case where prototypes are empirical centroids. We go one step further and propose to jointly learn prototypes by using distances to prototypes in representation space as the logit scores for classification. Further, we theoretically analyze the properties of Euclidean distance based prototype classifiers that lead to stable gradient-based optimization which is robust to outliers. To enable independent distance scales along each channel, we enhance Prototype classifiers by learning channel-dependent temperature parameters. Our analysis shows that prototypes learned by Prototype classifiers are better separated than empirical centroids. Results on four LTR benchmarks show that Prototype classifier outperforms or is comparable to state-of-the-art methods. Our code is made available at https://github.com/saurabhsharma1993/prototype-classifier-ltr.

----

## [151] Divide Rows and Conquer Cells: Towards Structure Recognition for Large Tables

**Authors**: *Huawen Shen, Xiang Gao, Jin Wei, Liang Qiao, Yu Zhou, Qiang Li, Zhanzhan Cheng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/152](https://doi.org/10.24963/ijcai.2023/152)

**Abstract**:

Recent advanced Table Structure Recognition (TSR) models adopt image-to-text solutions to parse table structure. These methods can be formulated as image caption problem, i.e., input a single-table image and output table structure description in a specific text format, e.g., HTML. With the impressive success of Transformer in text generation tasks, these methods use Transformer architecture to predict HTML table text in an autoregressive manner. However, tables always emerge with a large variety of shapes and sizes. Autoregressive models usually suffer from the error accumulation problem as the length of predicted text increases, which results in unsatisfactory performance for large tables. In this paper, we propose a novel image-to-text based TSR method that relieves error accumulation problems and improves performance noticeably. At the core of our method is a cascaded two-step decoder architecture with the former decoder predicting HTML table row tags non-autoregressively and the latter predicting HTML table cell tags of each row in a semi-autoregressive manner. Compared with existing methods that predict HTML text autoregressively, the superiority of our row-to-cell progressive table parsing is twofold: (1) it generates an HTML tag sequence with a vertical-and-horizontal two-step `scanning', which better fits the inherent 2D structure of image data, (2) it performs substantially better for large tables (long sequence prediction) since it alleviates error accumulation problem specific to autoregressive models. Extensive experiments demonstrate that our method achieves competitive performance on three public benchmarks.

----

## [152] Data Level Lottery Ticket Hypothesis for Vision Transformers

**Authors**: *Xuan Shen, Zhenglun Kong, Minghai Qin, Peiyan Dong, Geng Yuan, Xin Meng, Hao Tang, Xiaolong Ma, Yanzhi Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/153](https://doi.org/10.24963/ijcai.2023/153)

**Abstract**:

The conventional lottery ticket hypothesis (LTH) claims that there exists a sparse subnetwork within a dense neural network and a proper random initialization method, called the winning ticket, such that it can be trained from scratch to almost as good as the dense counterpart. Meanwhile, the research of LTH in vision transformers (ViTs) is scarcely evaluated. In this paper, we first show that the conventional winning ticket is hard to find at weight level of ViTs by existing methods. Then, we generalize the LTH for ViTs to input data consisting of image patches inspired by the input dependence of ViTs. That is, there exists a subset of input image patches such that a ViT can be trained from scratch by using only this subset of patches and achieve similar accuracy to the ViTs trained by using all image patches. We call this subset of input patches the winning tickets, which represent a significant amount of information in the input data. We use a ticket selector to generate the winning tickets based on the informativeness of patches for various types of ViT, including DeiT, LV-ViT, and Swin Transformers. The experiments show that there is a clear difference between the performance of models trained with winning tickets and randomly selected subsets, which verifies our proposed theory. We elaborate the analogical similarity between our proposed Data-LTH-ViTs and the conventional LTH for further verifying the integrity of our theory. The Source codes are available at https://github.com/shawnricecake/vit-lottery-ticket-input.

----

## [153] Discrepancy-Guided Reconstruction Learning for Image Forgery Detection

**Authors**: *Zenan Shi, Haipeng Chen, Long Chen, Dong Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/154](https://doi.org/10.24963/ijcai.2023/154)

**Abstract**:

In this paper, we propose a novel image forgery detection paradigm for boosting the model learning capacity on both forgery-sensitive and genuine compact visual patterns. Compared to the existing methods that only focus on the discrepant-specific patterns (\eg, noises, textures, and frequencies), our method has a greater generalization. Specifically, we first propose a Discrepancy-Guided Encoder (DisGE) to extract forgery-sensitive visual patterns. DisGE consists of two branches, where the mainstream backbone branch is used to extract general semantic features, and the accessorial discrepant external attention branch is used to extract explicit forgery cues. Besides, a Double-Head Reconstruction (DouHR) module is proposed to enhance genuine compact visual patterns in different granular spaces. Under DouHR, we further introduce a Discrepancy-Aggregation Detector (DisAD) to aggregate these genuine compact visual patterns, such that the forgery detection capability on unknown patterns can be improved. Extensive experimental results on four challenging datasets validate the effectiveness of our proposed method against state-of-the-art competitors.

----

## [154] Depth-Relative Self Attention for Monocular Depth Estimation

**Authors**: *Kyuhong Shim, Jiyoung Kim, Gusang Lee, Byonghyo Shim*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/155](https://doi.org/10.24963/ijcai.2023/155)

**Abstract**:

Monocular depth estimation is very challenging because clues to the exact depth are incomplete in a single RGB image. To overcome the limitation, deep neural networks rely on various visual hints such as size, shade, and texture extracted from RGB information. However, we observe that if such hints are overly exploited, the network can be biased on RGB information without considering the comprehensive view. We propose a novel depth estimation model named RElative Depth Transformer (RED-T) that uses relative depth as guidance in self-attention. Specifically, the model assigns high attention weights to pixels of close depth and low attention weights to pixels of distant depth. As a result, the features of similar depth can become more likely to each other and thus less prone to misused visual hints. We show that the proposed model achieves competitive results in monocular depth estimation benchmarks and is less biased to RGB information. In addition, we propose a novel monocular depth estimation benchmark that limits the observable depth range during training in order to evaluate the robustness of the model for unseen depths.

----

## [155] Acoustic NLOS Imaging with Cross Modal Knowledge Distillation

**Authors**: *Ui-Hyeon Shin, Seungwoo Jang, Kwangsu Kim*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/156](https://doi.org/10.24963/ijcai.2023/156)

**Abstract**:

Acoustic non-line-of-sight (NLOS) imaging aims to reconstruct hidden scenes by analyzing reflections of acoustic waves. Despite recent developments in the field, existing methods still have limitations such as sensitivity to noise in a physical model and difficulty in reconstructing unseen objects in a deep learning model. To address these limitations, we propose a novel cross-modal knowledge distillation (CMKD) approach for acoustic NLOS imaging. Our method transfers knowledge from a well-trained image network to an audio network, effectively combining the strengths of both modalities. As a result, it is robust to noise and superior in reconstructing unseen objects. Additionally, we evaluate real-world datasets and demonstrate that the proposed method outperforms state-of-the-art methods in acoustic NLOS imaging. The experimental results indicate that CMKD is an effective solution for addressing the limitations of current acoustic NLOS imaging methods. Our code, model, and data are available at https://github.com/shineh96/Acoustic-NLOS-CMKD.

----

## [156] VGOS: Voxel Grid Optimization for View Synthesis from Sparse Inputs

**Authors**: *Jiakai Sun, Zhanjie Zhang, Jiafu Chen, Guangyuan Li, Boyan Ji, Lei Zhao, Wei Xing*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/157](https://doi.org/10.24963/ijcai.2023/157)

**Abstract**:

Neural Radiance Fields (NeRF) has shown great success in novel view synthesis due to its state-of-the-art quality and flexibility. However, NeRF requires dense input views (tens to hundreds) and a long training time (hours to days) for a single scene to generate high-fidelity images. Although using the voxel grids to represent the radiance field can significantly accelerate the optimization process, we observe that for sparse inputs, the voxel grids are more prone to overfitting to the training views and will have holes and floaters, which leads to artifacts. In this paper, we propose VGOS, an approach for fast (3-5 minutes) radiance field reconstruction from sparse inputs (3-10 views) to address these issues. To improve the performance of voxel-based radiance field in sparse input scenarios, we propose two methods: (a) We introduce an incremental voxel training strategy, which prevents overfitting by suppressing the optimization of peripheral voxels in the early stage of reconstruction. (b) We use several regularization techniques to smooth the voxels, which avoids degenerate solutions. Experiments demonstrate that VGOS achieves state-of-the-art performance for sparse inputs with super-fast convergence. Code will be available at https://github.com/SJoJoK/VGOS.

----

## [157] Appearance Prompt Vision Transformer for Connectome Reconstruction

**Authors**: *Rui Sun, Naisong Luo, Yuwen Pan, Huayu Mai, Tianzhu Zhang, Zhiwei Xiong, Feng Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/158](https://doi.org/10.24963/ijcai.2023/158)

**Abstract**:

Neural connectivity reconstruction aims to understand the function of biological reconstruction and promote basic scientific research. The intricate morphology and densely intertwined branches make it an extremely challenging task. Most previous best-performing methods adopt affinity learning or metric learning. Nevertheless, they either neglect to model explicit voxel semantics caused by implicit optimization or are hysteresis to spatial information. Furthermore, the inherent locality of 3D CNNs limits modeling long-range dependencies, leading to sub-optimal results. In this work, we propose a coherent and unified Appearance Prompt Vision Transformer (APViT) to integrate affinity and metric learning to exploit the complementarity by learning long-range spatial dependencies. The proposed APViT enjoys several merits. First, the extension continuity-aware attention module aims at constructing hierarchical attention customized for neuron extensibility and slice continuity to learn instance voxel semantic context from a global perspective and utilize continuity priors to enhance voxel spatial awareness. Second, the appearance prompt modulator is responsible for leveraging voxel-adaptive appearance knowledge conditioned on affinity rich in spatial information to instruct instance voxel semantics, exploiting the potential of affinity learning to complement metric learning. Extensive experimental results on multiple challenging benchmarks demonstrate that our APViT achieves consistent improvements with huge flexibility under the same post-processing strategy.

----

## [158] Domain-Adaptive Self-Supervised Face & Body Detection in Drawings

**Authors**: *Baris Batuhan Topal, Deniz Yuret, Tevfik Metin Sezgin*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/159](https://doi.org/10.24963/ijcai.2023/159)

**Abstract**:

Drawings are powerful means of pictorial abstraction and communication. Understanding diverse forms of drawings, including digital arts, cartoons, and comics, has been a major problem of interest for the computer vision and computer graphics communities. Although there are large amounts of digitized drawings from comic books and cartoons, they contain vast stylistic variations, which necessitate expensive manual labeling for training domain-specific recognizers. In this work, we show how self-supervised learning, based on a teacher-student network with a modified student network update design, can be used to build face and body detectors. Our setup allows exploiting large amounts of unlabeled data from the target domain when labels are provided for only a small subset of it. We further demonstrate that style transfer can be incorporated into our learning pipeline to bootstrap detectors using a vast amount of out-of-domain labeled images from natural images (i.e., images from the real world). Our combined architecture yields detectors with state-of-the-art (SOTA) and near-SOTA performance using minimal annotation effort. Our code can be accessed from https://github.com/barisbatuhan/DASS_Detector.

----

## [159] Shaken, and Stirred: Long-Range Dependencies Enable Robust Outlier Detection with PixelCNN++

**Authors**: *Barath Mohan Umapathi, Kushal Chauhan, Pradeep Shenoy, Devarajan Sridharan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/160](https://doi.org/10.24963/ijcai.2023/160)

**Abstract**:

Reliable outlier detection is critical for real-world deployment of deep learning models. Although extensively studied, likelihoods produced by deep generative models have been largely dismissed as being impractical for outlier detection. First, deep generative model likelihoods are readily biased by low-level input statistics. Second, many recent solutions for correcting these biases are computationally expensive, or do not generalize well to complex, natural datasets. Here, we explore outlier detection with a state-of-the-art deep autoregressive model: PixelCNN++. We show that biases in PixelCNN++ likelihoods arise primarily from predictions based on local dependencies. We propose two families of bijective transformations -- ``stirring'' and ``shaking'' -- which ameliorate low-level biases and isolate the contribution of long-range dependencies to PixelCNN++ likelihoods. These transformations are inexpensive and readily computed at evaluation time. We test our approaches extensively with five grayscale and six natural image datasets and show that they achieve or exceed state-of-the-art outlier detection, particularly on datasets with complex, natural images. We also show that our solutions work well with other types of generative models (generative flows and variational autoencoders) and that their efficacy is governed by each model's reliance on local dependencies. In sum, lightweight remedies suffice to achieve robust outlier detection on image data with deep generative models.

----

## [160] Temporal Constrained Feasible Subspace Learning for Human Pose Forecasting

**Authors**: *Gaoang Wang, Mingli Song*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/161](https://doi.org/10.24963/ijcai.2023/161)

**Abstract**:

Human pose forecasting is a sequential modeling task that aims to predict future poses from historical motions. Most existing approaches focus on the spatial-temporal neural network model design for learning movement patterns to reduce prediction errors. However, they usually do not strictly follow the temporal constraints in the inference stage. Even though a small Mean Per Joint Position Error (MPJPE) is achieved, some of the predicted poses are not temporal feasible solutions, which disobeys the continuity of the body movement. In this paper, we consider the temporal constrained feasible solutions for human pose forecasting, where the predicted poses of input historical poses are guaranteed to obey the temporal constraints strictly in the inference stage. Rather than direct supervision of the prediction in the original pose space, a temporal constrained subspace is explicitly learned and then followed by an inverse transformation to obtain the final predictions. We evaluate the proposed method on large-scale benchmarks, including Human3.6M, AMASS, and 3DPW. State-of-the-art performance has been achieved with the temporal constrained feasible solutions.

----

## [161] Learning Calibrated Uncertainties for Domain Shift: A Distributionally Robust Learning Approach

**Authors**: *Haoxuan Wang, Zhiding Yu, Yisong Yue, Animashree Anandkumar, Anqi Liu, Junchi Yan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/162](https://doi.org/10.24963/ijcai.2023/162)

**Abstract**:

We propose a framework for learning calibrated uncertainties under domain shifts, considering the case where the source (training) distribution differs from the target (test) distribution. We detect such domain shifts through the use of a differentiable density ratio estimator and train it together with the task network, composing an adjusted softmax predictive form that concerns the domain shift. In particular, the density ratio estimator yields a density ratio that reflects the closeness of a target (test) sample to the source (training) distribution. We employ it to adjust the uncertainty of prediction in the task network. This idea of using the density ratio is based on the distributionally robust learning (DRL) framework, which accounts for the domain shift through adversarial risk minimization. We demonstrate that our proposed method generates calibrated uncertainties that benefit many downstream tasks, such as unsupervised domain adaptation (UDA) and semi-supervised learning (SSL). On these tasks, methods like self-training and FixMatch use uncertainties to select confident pseudo-labels for re-training. Our experiments show that the introduction of DRL leads to significant improvements in cross-domain performance. We also demonstrate that the estimated density ratios show an agreement with the human selection frequencies, suggesting a positive correlation with a proxy of human perceived uncertainties.

----

## [162] Hierarchical Prompt Learning for Compositional Zero-Shot Recognition

**Authors**: *Henan Wang, Muli Yang, Kun Wei, Cheng Deng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/163](https://doi.org/10.24963/ijcai.2023/163)

**Abstract**:

Compositional Zero-Shot Learning (CZSL) aims to imitate the powerful generalization ability of human beings to recognize novel compositions of known primitive concepts that correspond to a state and an object, e.g., purple apple. To fully capture the intra- and inter-class correlations between compositional concepts, in this paper, we propose to learn them in a hierarchical manner. Specifically, we set up three hierarchical embedding spaces that respectively model the states, the objects, and their compositions, which serve as three “experts” that can be combined in inference for more accurate predictions. We achieve this based on the recent success of large-scale pretrained vision-language models, e.g., CLIP, which provides a strong initial knowledge of image-text relationships. To better adapt this knowledge to CZSL, we propose to learn three hierarchical prompts by explicitly fixing the unrelated word tokens in the three embedding spaces. Despite its simplicity, our proposed method consistently yields superior performance over current state-of-the-art approaches on three widely-used CZSL benchmarks.

----

## [163] A Dual Semantic-Aware Recurrent Global-Adaptive Network for Vision-and-Language Navigation

**Authors**: *Liuyi Wang, Zongtao He, Jiagui Tang, Ronghao Dang, Naijia Wang, Chengju Liu, Qijun Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/164](https://doi.org/10.24963/ijcai.2023/164)

**Abstract**:

Vision-and-Language Navigation (VLN) is a realistic but challenging task that requires an agent to locate the target region using verbal and visual cues. While significant advancements have been achieved recently, there are still two broad limitations: (1) The explicit information mining for significant guiding semantics concealed in both vision and language is still under-explored; (2) The previously structured map method provides the average historical appearance of visited nodes, while it ignores distinctive contributions of various images and potent information retention in the reasoning process. This work proposes a dual semantic-aware recurrent global-adaptive network (DSRG) to address the above problems. First, DSRG proposes an instruction-guidance linguistic module (IGL) and an appearance-semantics visual module (ASV) for boosting vision and language semantic learning respectively. For the memory mechanism, a global adaptive aggregation module (GAA) is devised for explicit panoramic observation fusion, and a recurrent memory fusion module (RMF) is introduced to supply implicit temporal hidden states. Extensive experimental results on the R2R and REVERIE datasets demonstrate that our method achieves better performance than existing methods. Code is available at https://github.com/CrystalSixone/DSRG.

----

## [164] Detecting Adversarial Faces Using Only Real Face Self-Perturbations

**Authors**: *Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/165](https://doi.org/10.24963/ijcai.2023/165)

**Abstract**:

Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.

----

## [165] Align, Perturb and Decouple: Toward Better Leverage of Difference Information for RSI Change Detection

**Authors**: *Supeng Wang, Yuxi Li, Ming Xie, Mingmin Chi, Yabiao Wang, Chengjie Wang, Wenbing Zhu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/166](https://doi.org/10.24963/ijcai.2023/166)

**Abstract**:

Change detection is a widely adopted technique in remote sense imagery (RSI) analysis in the discovery of long-term geomorphic evolution. To highlight the areas of semantic changes, previous effort mostly pays attention to learning representative feature descriptors of a single image, while the difference information is either modeled with simple difference operations or implicitly embedded via feature interactions. Nevertheless, such difference modeling can be noisy since it suffers from non-semantic changes and lacks explicit guidance from image content or context. In this paper, we revisit the importance of feature difference for change detection in RSI, and propose a series of operations to fully exploit the difference information: Alignment, Perturbation and Decoupling (APD). Firstly, alignment leverages contextual similarity to compensate for the non-semantic difference in feature space. Next, a difference module trained with semantic-wise perturbation is adopted to learn more generalized change estimators, which reversely bootstraps feature extraction and prediction. Finally, a decoupled dual-decoder structure is designed to predict semantic changes in both content-aware and content-agnostic manners. Extensive experiments are conducted on benchmarks of LEVIR-CD, WHU-CD and DSIFN-CD, demonstrating our proposed operations bring significant improvement and achieve competitive results under similar comparative conditions. Code is available at https://github.com/wangsp1999/CD-Research/tree/main/openAPD

----

## [166] Learning 3D Photography Videos via Self-supervised Diffusion on Single Images

**Authors**: *Xiaodong Wang, Chenfei Wu, Shengming Yin, Minheng Ni, Jianfeng Wang, Linjie Li, Zhengyuan Yang, Fan Yang, Lijuan Wang, Zicheng Liu, Yuejian Fang, Nan Duan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/167](https://doi.org/10.24963/ijcai.2023/167)

**Abstract**:

3D photography renders a static image into a video with appealing 3D visual effects. Existing approaches typically first conduct monocular depth estimation, then render the input frame to subsequent frames with various viewpoints, and finally use an inpainting model to fill those missing/occluded regions. The inpainting model plays a crucial role in rendering quality, but it is normally trained on out-of-domain data. To reduce the training and inference gap, we propose a novel self-supervised diffusion model as the inpainting module. Given a single input image, we automatically construct a training pair of the masked occluded image and the ground-truth image with random cycle rendering. The constructed training samples are closely aligned to the testing instances, without the need for data annotation. To make full use of the masked images, we designed a Masked Enhanced Block (MEB), which can be easily plugged into the UNet and enhance the semantic conditions. Towards real-world animation, we present a novel task: out-animation, which extends the space and time of input objects. Extensive experiments on real datasets show that our method achieves competitive results with existing SOTA methods.

----

## [167] Dual-view Correlation Hybrid Attention Network for Robust Holistic Mammogram Classification

**Authors**: *Zhiwei Wang, Junlin Xian, Kangyi Liu, Xin Li, Qiang Li, Xin Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/168](https://doi.org/10.24963/ijcai.2023/168)

**Abstract**:

Mammogram image is important for breast cancer screening, and typically obtained in a dual-view form, i.e., cranio-caudal (CC) and mediolateral oblique (MLO), to provide complementary information for clinical decisions. However, previous methods mostly learn features from the two views independently, which violates the clinical knowledge and ignores the importance of dual-view correlation in the feature learning. In this paper, we propose a dual-view correlation hybrid attention network (DCHA-Net) for robust holistic mammogram classification. Specifically, DCHA-Net is carefully designed to extract and reinvent deep feature maps for the two views, and meanwhile to maximize the underlying correlations between them. A hybrid attention module, consisting of local relation and non-local attention blocks, is proposed to alleviate the spatial misalignment of the paired views in the correlation maximization. A dual-view correlation loss is introduced to maximize the feature similarity between corresponding strip-like regions with equal distance to the chest wall, motivated by the fact that their features represent the same breast tissues, and thus should be highly-correlated with each other. Experimental results on the two public datasets, i.e., INbreast and CBIS-DDSM, demonstrate that the DCHA-Net can well preserve and maximize feature correlations across views, and thus outperforms previous state-of-the-art methods for classifying a whole mammogram as malignant or not.

----

## [168] Accurate MRI Reconstruction via Multi-Domain Recurrent Networks

**Authors**: *Jinbao Wei, Zhijie Wang, Kongqiao Wang, Li Guo, Xueyang Fu, Ji Liu, Xun Chen*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/169](https://doi.org/10.24963/ijcai.2023/169)

**Abstract**:

In recent years, deep convolutional neural networks (CNNs) have become dominant in MRI reconstruction from undersampled k-space. However, most existing CNNs methods reconstruct the undersampled images either in the spatial domain or in the frequency domain, and neglecting the correlation between these two domains. This hinders the further reconstruction performance improvement. To tackle this issue, in this work, we propose a new multi-domain recurrent network (MDR-Net) with multi-domain learning (MDL) blocks as its basic units to reconstruct the undersampled MR image progressively. Specifically, the MDL block interactively processes the local spatial features and the global frequency information to facilitate complementary learning, leading to fine-grained features generation. Furthermore, we introduce an effective frequency-based loss to narrow the frequency spectrum gap, compensating for over-smoothness caused by the widely used spatial reconstruction loss. Extensive experiments on public fastMRI datasets demonstrate that our MDR-Net consistently outperforms other competitive methods and is able to provide more details.

----

## [169] From Generation to Suppression: Towards Effective Irregular Glow Removal for Nighttime Visibility Enhancement

**Authors**: *Wanyu Wu, Wei Wang, Zheng Wang, Kui Jiang, Xin Xu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/170](https://doi.org/10.24963/ijcai.2023/170)

**Abstract**:

Most existing Low-Light Image Enhancement (LLIE) methods are primarily designed to improve brightness in dark regions, which suffer from severe degradation in nighttime images. However, these methods have limited exploration in another major visibility damage, the glow effects in real night scenes. Glow effects are inevitable in the presence of artificial light sources and cause further diffused blurring when directly enhanced. To settle this issue, we innovatively consider the glow suppression task as learning physical glow generation via multiple scattering estimation according to the Atmospheric Point Spread Function (APSF). In response to the challenges posed by uneven glow intensity and varying source shapes, an APSF-based Nighttime Imaging Model with Near-field Light Sources (NIM-NLS) is specifically derived to design a scalable Light-aware Blind Deconvolution Network (LBDN). The glow-suppressed result is then brightened via a Retinex-based Enhancement Module (REM). Remarkably, the proposed glow suppression method is based on zero-shot learning and does not rely on any paired or unpaired training data. Empirical evaluations demonstrate the effectiveness of the proposed method in both glow suppression and low-light enhancement tasks.

----

## [170] Hierarchical Semantic Contrast for Weakly Supervised Semantic Segmentation

**Authors**: *Yuanchen Wu, Xiaoqiang Li, Songmin Dai, Jide Li, Tong Liu, Shaorong Xie*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/171](https://doi.org/10.24963/ijcai.2023/171)

**Abstract**:

Weakly supervised semantic segmentation (WSSS) with image-level annotations has achieved great processes through class activation map (CAM). Since vanilla CAMs are hardly served as guidance to bridge the gap between full and weak supervision, recent studies explore semantic representations to make CAM fit for WSSS and demonstrate encouraging results. However, they generally exploit single-level semantics, which may hamper the model to learn a comprehensive semantic structure. Motivated by the prior that each image has multiple levels of semantics, we propose hierarchical semantic contrast (HSC) to ameliorate the above problem. It conducts semantic contrast from coarse-grained to fine-grained perspective, including ROI level, class level, and pixel level, making the model learn a better object pattern understanding. To further improve CAM quality, building upon HSC, we explore consistency regularization of cross supervision and develop momentum prototype learning to utilize abundant semantics across different images. Extensive studies manifest that our plug-and-play learning paradigm, HSC, can significantly boost CAM quality on both non-saliency-guided and saliency-guided baselines, and establish new state-of-the-art WSSS performance on PASCAL VOC 2012 dataset. Code is available at https://github.com/Wu0409/HSC_WSSS.

----

## [171] Learning Monocular Depth in Dynamic Environment via Context-aware Temporal Attention

**Authors**: *Zizhang Wu, Zhuozheng Li, Zhi-Gang Fan, Yunzhe Wu, Yuanzhu Gan, Jian Pu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/172](https://doi.org/10.24963/ijcai.2023/172)

**Abstract**:

The monocular depth estimation task has recently revealed encouraging prospects, especially for the autonomous driving task. To tackle the ill-posed problem of 3D geometric reasoning from 2D monocular images, multi-frame monocular methods are developed to leverage the perspective correlation information from sequential temporal frames. However, moving objects such as cars and trains usually violate the static scene assumption, leading to feature inconsistency deviation and misaligned cost values, which would mislead the optimization algorithm. In this work, we present CTA-Depth, a Context-aware Temporal Attention guided network for multi-frame monocular Depth estimation. Specifically, we first apply a multi-level attention enhancement module to integrate multi-level image features to obtain an initial depth and pose estimation. Then the proposed CTA-Refiner is adopted to alternatively optimize the depth and pose. During the CTA-Refiner process, context-aware temporal attention (CTA) is developed to capture the global temporal-context correlations to maintain the feature consistency and estimation integrity of moving objects. In particular, we propose a long-range geometry embedding (LGE) module to produce a long-range temporal geometry prior. Our approach achieves significant improvements (e.g., 13.5% for the Abs Rel metric on the KITTI dataset) over state-of-the-art approaches on three benchmark datasets.

----

## [172] Hyperspectral Image Denoising Using Uncertainty-Aware Adjustor

**Authors**: *Jiahua Xiao, Xing Wei*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/173](https://doi.org/10.24963/ijcai.2023/173)

**Abstract**:

Hyperspectral image (HSI) denoising has achieved promising results with the development of deep learning. A mainstream class of methods exploits the spatial-spectral correlations and recovers each band with the aids of neighboring bands, collectively referred to as spectral auxiliary networks. However, these methods treat entire adjacent spectral bands equally. In theory, clearer and nearer bands  tend to contain more reliable spectral information than noisier and farther ones with higher uncertainties. How to achieve spectral enhancement and adaptation of each adjacent band has become an urgent problem in HSI denoising. This work presents the UA-Adjustor, a comprehensive adjustor that enhances denoising performance by considering both the band-to-pixel and enhancement-to-adjustment aspects. Specifically, UA-Adjustor consists of three stages that evaluate the importance of neighboring bands, enhance neighboring bands based on uncertainty perception, and adjust the weight of spatial pixels in adjacent bands through estimated uncertainty. For its simplicity, UA-Adjustor can be flexibly plugged into existing spectral auxiliary networks to improve denoising behavior at low cost. Extensive experimental results validate that the proposed solution can improve over recent state-of-the-art (SOTA) methods on both simulated and real-world benchmarks by a large margin.

----

## [173] ViT-CX: Causal Explanation of Vision Transformers

**Authors**: *Weiyan Xie, Xiao-Hui Li, Caleb Chen Cao, Nevin L. Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/174](https://doi.org/10.24963/ijcai.2023/174)

**Abstract**:

Despite the popularity of Vision Transformers (ViTs) and eXplainable AI (XAI), only a few explanation methods have been designed specially for ViTs thus far. They mostly use attention weights of the [CLS] token on patch embeddings and often produce unsatisfactory saliency maps. This paper proposes a novel method for explaining ViTs called ViT-CX. It is based on patch embeddings, rather than attentions paid to them, and their causal impacts on the model output. Other characteristics of ViTs such as causal overdetermination are considered in the design of ViT-CX. The empirical results show that ViT-CX produces more meaningful saliency maps and does a better job revealing all important evidence for the predictions than previous methods. The explanation generated by ViT-CX also shows significantly better faithfulness to the model. The codes and appendix are available at https://github.com/vaynexie/CausalX-ViT.

----

## [174] 3D Surface Super-resolution from Enhanced 2D Normal Images: A Multimodal-driven Variational AutoEncoder Approach

**Authors**: *Wuyuan Xie, Tengcong Huang, Miaohui Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/175](https://doi.org/10.24963/ijcai.2023/175)

**Abstract**:

3D surface super-resolution is an important technical tool in virtual reality, and it is also a research hotspot in computer vision. Due to the unstructured and irregular nature of 3D object data, it is usually difficult to obtain high-quality surface details and geometry textures via a low-cost hardware setup. In this paper, we establish a multimodal-driven variational autoencoder (mmVAE) framework to perform 3D surface enhancement based on 2D normal images. To fully leverage the multimodal learning, we investigate a multimodal Gaussian mixture model (mmGMM) to align and fuse the latent feature representations from different modalities, and further propose a cross-scale encoder-decoder structure to reconstruct high-resolution normal images. Experimental results on several benchmark datasets demonstrate that our method delivers promising surface geometry structures and details in comparison with competitive advances.

----

## [175] Diagnose Like a Pathologist: Transformer-Enabled Hierarchical Attention-Guided Multiple Instance Learning for Whole Slide Image Classification

**Authors**: *Conghao Xiong, Hao Chen, Joseph J. Y. Sung, Irwin King*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/176](https://doi.org/10.24963/ijcai.2023/176)

**Abstract**:

Multiple Instance Learning (MIL) and transformers are increasingly popular in histopathology Whole Slide Image (WSI) classification. However, unlike human pathologists who selectively observe specific regions of histopathology tissues under different magnifications, most methods do not incorporate multiple resolutions of the WSIs, hierarchically and attentively, thereby leading to a loss of focus on the WSIs and information from other resolutions. To resolve this issue, we propose a Hierarchical Attention-Guided Multiple Instance Learning framework to fully exploit the WSIs. This framework can dynamically and attentively discover the discriminative regions across multiple resolutions of the WSIs. Within this framework, an Integrated Attention Transformer is proposed to further enhance the performance of the transformer and obtain a more holistic WSI (bag) representation. This transformer consists of multiple Integrated Attention Modules, which is the combination of a transformer layer and an aggregation module that produces a bag representation based on every instance representation in that bag. The experimental results show that our method achieved state-of-the-art performances on multiple datasets, including Camelyon16, TCGA-RCC, TCGA-NSCLC, and an in-house IMGC dataset. The code is available at https://github.com/BearCleverProud/HAG-MIL.

----

## [176] Universal Adaptive Data Augmentation

**Authors**: *Xiaogang Xu, Hengshuang Zhao*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/177](https://doi.org/10.24963/ijcai.2023/177)

**Abstract**:

Existing automatic data augmentation (DA) methods either ignore updating DA's parameters according to the target model's state during training or adopt update strategies that are not effective enough. In this work, we design a novel data augmentation strategy called ``Universal Adaptive Data Augmentation" (UADA). Different from existing methods, UADA would adaptively update DA's parameters according to the target model's gradient information during training: given a pre-defined set of DA operations, we randomly decide types and magnitudes of DA operations for every data batch during training, and adaptively update DA's parameters along the gradient direction of the loss concerning DA's parameters. In this way, UADA can increase the training loss of the target networks, and the target networks would learn features from harder samples to improve the generalization. Moreover, UADA is very general and can be utilized in numerous tasks, e.g., image classification, semantic segmentation and object detection. Extensive experiments with various models are conducted on CIFAR-10, CIFAR-100, ImageNet, tiny-ImageNet, Cityscapes, and VOC07+12 to prove the significant performance improvements brought by UADA.

----

## [177] Video Object Segmentation in Panoptic Wild Scenes

**Authors**: *Yuanyou Xu, Zongxin Yang, Yi Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/178](https://doi.org/10.24963/ijcai.2023/178)

**Abstract**:

In this paper, we introduce semi-supervised video object segmentation (VOS) to panoptic wild scenes and present a large-scale benchmark as well as a baseline method for it. Previous benchmarks for VOS with sparse annotations are not sufficient to train or evaluate a model that needs to process all possible objects in real-world scenarios. Our new benchmark (VIPOSeg) contains exhaustive object annotations and covers various real-world object categories which are carefully divided into subsets of thing/stuff and seen/unseen classes for comprehensive evaluation. Considering the challenges in panoptic VOS, we propose a strong baseline method named panoptic object association with transformers (PAOT), which associates multiple objects by panoptic identification in a pyramid architecture on multiple scales. Experimental results show that VIPOSeg can not only boost the performance of VOS models by panoptic training but also evaluate them comprehensively in panoptic scenes. Previous methods for classic VOS still need to improve in performance and efficiency when dealing with panoptic scenes, while our PAOT achieves SOTA performance with good efficiency on VIPOSeg and previous VOS benchmarks. PAOT also ranks 1st in the VOT2022 challenge. Our dataset and code are available at https://github.com/yoxu515/VIPOSeg-Benchmark.

----

## [178] RuleMatch: Matching Abstract Rules for Semi-supervised Learning of Human Standard Intelligence Tests

**Authors**: *Yunlong Xu, Lingxiao Yang, Hongzhi You, Zonglei Zhen, Da-Hui Wang, Xiaohong Wan, Xiaohua Xie, Ru-Yuan Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/179](https://doi.org/10.24963/ijcai.2023/179)

**Abstract**:

Raven's Progressive Matrices (RPM), one of the standard intelligence tests in human psychology, has recently emerged as a powerful tool for studying abstract visual reasoning (AVR) abilities in machines. Although existing computational models for RPM problems achieve good performance, they require a large number of labeled training examples for supervised learning. In contrast, humans can efficiently solve unlabeled RPM problems after learning from only a few example questions. Here, we develop a semi-supervised learning (SSL) method, called RuleMatch, to train deep models with a small number of labeled RPM questions along with other unlabeled questions. Moreover, instead of using pixel-level augmentation in object perception tasks, we exploit the nature of RPM problems and augment the data at the level of abstract rules. Specifically, we disrupt the possible rules contained among context images in an RPM question and force the two augmented variants of the same unlabeled sample to obey the same abstract rule and predict a common pseudo label for training. Extensive experiments show that the proposed RuleMatch achieves state-of-the-art performance on two popular RAVEN datasets. Our work makes an important stride in aligning abstract analogical visual reasoning abilities in machines and humans. Our Code is at https://github.com/ZjjConan/AVR-RuleMatch.

----

## [179] Prompt Learns Prompt: Exploring Knowledge-Aware Generative Prompt Collaboration For Video Captioning

**Authors**: *Liqi Yan, Cheng Han, Zenglin Xu, Dongfang Liu, Qifan Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/180](https://doi.org/10.24963/ijcai.2023/180)

**Abstract**:

Fine-tuning large vision-language models is a challenging task. Prompt tuning approaches have been introduced to learn fixed textual or visual prompts while freezing the pre-trained model in downstream tasks. Despite the effectiveness of prompt tuning, what do those learnable prompts learn remains unexplained. In this work, we explore whether prompts in the fine-tuning can learn knowledge-aware prompts from the pre-training, by designing two different sets of prompts in pre-training and fine-tuning phases respectively. Specifically, we present a Video-Language Prompt tuning (VL-Prompt) approach for video captioning, which first efficiently pre-train a video-language model to extract key information (e.g., actions and objects) with flexibly generated Knowledge-Aware Prompt (KAP). Then, we design a Video-Language Prompt (VLP) to transfer the knowledge from the knowledge-aware prompts and fine-tune the model to generate full captions. Experimental results show the superior performance of our approach over several state-of-the-art baselines. We further demonstrate that the video-language prompts are well learned from the knowledge-aware prompts.

----

## [180] Few-shot Classification via Ensemble Learning with Multi-Order Statistics

**Authors**: *Sai Yang, Fan Liu, Delong Chen, Jun Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/181](https://doi.org/10.24963/ijcai.2023/181)

**Abstract**:

Transfer learning has been widely adopted for few-shot classification. Recent studies reveal that obtaining good generalization representation of images on novel classes is the key to improving the few-shot classification accuracy. To address this need, we prove theoretically that leveraging ensemble learning on the base classes can correspondingly reduce the true error in the novel classes. Following this principle, a novel method named Ensemble Learning with Multi-Order Statistics (ELMOS) is proposed in this paper. In this method, after the backbone network, we use multiple branches to create the individual learners in the ensemble learning, with the goal to reduce the storage cost. We then introduce different order statistics pooling in each branch to increase the diversity of the individual learners. The learners are optimized with supervised losses during the pre-training phase. After pre-training, features from different branches are concatenated for classifier evaluation. Extensive experiments demonstrate that each branch can complement the others and our method can produce a state-of-the-art performance on multiple few-shot classification benchmark datasets.

----

## [181] Video Diffusion Models with Local-Global Context Guidance

**Authors**: *Siyuan Yang, Lu Zhang, Yu Liu, Zhizhuo Jiang, You He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/182](https://doi.org/10.24963/ijcai.2023/182)

**Abstract**:

Diffusion models have emerged as a powerful paradigm in video synthesis tasks including prediction, generation, and interpolation. Due to the limitation of the computational budget, existing methods usually implement conditional diffusion models with an autoregressive inference pipeline, in which the future fragment is predicted based on the distribution of adjacent past frames. However, only the conditions from a few previous frames can't capture the global temporal coherence, leading to inconsistent or even outrageous results in long-term video prediction. In this paper, we propose a Local-Global Context guided Video Diffusion model (LGC-VD) to capture multi-perception conditions for producing high-quality videos in both conditional/unconditional settings. In LGC-VD, the UNet is implemented with stacked residual blocks with self-attention units, avoiding the undesirable computational cost in 3D Conv. We construct a local-global context guidance strategy to capture the multi-perceptual embedding of the past fragment to boost the consistency of future prediction. Furthermore, we propose a two-stage training strategy to alleviate the effect of noisy frames for more stable predictions. Our experiments demonstrate that the proposed method achieves favorable performance on video prediction, interpolation, and unconditional video generation. We release code at https://github.com/exisas/LGC-VD.

----

## [182] Exploring Safety Supervision for Continual Test-time Domain Adaptation

**Authors**: *Xu Yang, Yanan Gu, Kun Wei, Cheng Deng*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/183](https://doi.org/10.24963/ijcai.2023/183)

**Abstract**:

Continual test-time domain adaptation aims to adapt a source pre-trained model to a continually changing target domain without using any source data. Unfortunately, existing methods based on pseudo-label learning suffer from the changing target domain environment, and the quality of generated pseudo-labels is attenuated due to the domain shift, leading to instantaneous negative learning and long-term knowledge forgetting. To solve these problems, in this paper, we propose a simple yet effective framework for exploring safety supervision with three elaborate strategies: Label Safety, Sample Safety, and Parameter Safety. Firstly, to select reliable pseudo-labels, we define and adjust the confidence threshold in a self-adaptive manner according to the test-time learning status. Secondly, a soft-weighted contrastive learning module is presented to explore the highly-correlated samples and discriminate uncorrelated ones, improving the instantaneous efficiency of the model. Finally, we frame a Soft Weight Alignment strategy to normalize the distance between the parameters of the adapted model and the source pre-trained model, which alleviates the long-term problem of knowledge forgetting and significantly improves the accuracy of the adapted model in the late adaptation stage. Extensive experimental results demonstrate that our method achieves state-of-the-art performance on several benchmark datasets.

----

## [183] Action Recognition with Multi-stream Motion Modeling and Mutual Information Maximization

**Authors**: *Yuheng Yang, Haipeng Chen, Zhenguang Liu, Yingda Lyu, Beibei Zhang, Shuang Wu, Zhibo Wang, Kui Ren*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/184](https://doi.org/10.24963/ijcai.2023/184)

**Abstract**:

Action recognition has long been a fundamental and intriguing problem in artificial intelligence. The task is challenging due to the high dimensionality nature of an action, as well as the subtle motion details to be considered. Current state-of-the-art approaches typically learn from articulated motion sequences in the straightforward 3D Euclidean space. However, the vanilla Euclidean space is not efficient for modeling important motion characteristics such as the joint-wise angular acceleration, which reveals the driving force behind the motion. Moreover, current methods typically attend to each channel equally and lack theoretical constrains on extracting task-relevant features from the input. 

In this paper, we seek to tackle these challenges from three aspects: (1) We propose to incorporate an acceleration representation, explicitly modeling the higher-order variations in motion. (2) We introduce a novel Stream-GCN network equipped with multi-stream components and channel attention, where different representations (i.e., streams) supplement each other towards a more precise action recognition while attention capitalizes on those important channels. (3) We explore feature-level supervision for maximizing the extraction of task-relevant information and formulate this into a mutual information loss. Empirically, our approach sets the new state-of-the-art performance on three benchmark datasets, NTU RGB+D, NTU RGB+D 120, and NW-UCLA.

----

## [184] Orientation-Independent Chinese Text Recognition in Scene Images

**Authors**: *Haiyang Yu, Xiaocong Wang, Bin Li, Xiangyang Xue*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/185](https://doi.org/10.24963/ijcai.2023/185)

**Abstract**:

Scene text recognition (STR) has attracted much attention due to its broad applications. The previous works pay more attention to dealing with the recognition of Latin text images with complex backgrounds by introducing language models or other auxiliary networks. Different from Latin texts, many vertical Chinese texts exist in natural scenes, which brings difficulties to current state-of-the-art STR methods. In this paper, we take the first attempt to extract orientation-independent visual features by disentangling content and orientation information of text images, thus recognizing both horizontal and vertical texts robustly in natural scenes. Specifically, we introduce a Character Image Reconstruction Network (CIRN) to recover corresponding printed character images with disentangled content and orientation information. We conduct experiments on a scene dataset for benchmarking Chinese text recognition, and the results demonstrate that the proposed method can indeed improve performance through disentangling content and orientation information. To further validate the effectiveness of our method, we additionally collect a Vertical Chinese Text Recognition (VCTR) dataset. The experimental results show that the proposed method achieves 45.63\% improvement on VCTR when introducing CIRN to the baseline model.

----

## [185] Actor-Multi-Scale Context Bidirectional Higher Order Interactive Relation Network for Spatial-Temporal Action Localization

**Authors**: *Jun Yu, Yingshuai Zheng, Shulan Ruan, Qi Liu, Zhiyuan Cheng, Jinze Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/186](https://doi.org/10.24963/ijcai.2023/186)

**Abstract**:

The key to video action detection lies in the understanding of interaction between persons and background objects in a video. Current methods usually employ object detectors to extract objects directly or use grid features to represent objects in the environment, which underestimate the great potential of multi-scale context information (e.g., objects and scenes of different sizes). How to exactly represent the multi-scale context and make full utilization of it still remains an unresolved challenge for spatial-temporal action localization. In this paper, we propose a novel Actor-Multi-Scale Context Bidirectional Higher Order Interactive Relation Network (AMCRNet) that extracts multi-scale context through multiple pooling layers with different sizes. Specifically, we develop an Interactive Relation Extraction module to model the higher-order relation between the target person and the context (e.g., other persons and objects). Along this line, we further propose a History Feature Bank and Interaction method to achieve better performance by modeling such relation across continuing video clips. Extensive experimental results on AVA2.2 and UCF101-24 demonstrate the superiority and rationality of our proposed AMCRNet.

----

## [186] Black-box Prompt Tuning for Vision-Language Model as a Service

**Authors**: *Lang Yu, Qin Chen, Jiaju Lin, Liang He*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/187](https://doi.org/10.24963/ijcai.2023/187)

**Abstract**:

In the scenario of Model-as-a-Service (MaaS), pre-trained models are usually released as inference APIs. Users are allowed to query those models with manually crafted prompts. Without accessing the network structure and gradient information, it's tricky to perform continuous prompt tuning on MaaS, especially for vision-language models (VLMs) considering cross-modal interaction. In this paper, we propose a black-box prompt tuning framework for VLMs to learn task-relevant prompts without back-propagation. In particular, the vision and language prompts are jointly optimized in the intrinsic parameter subspace with various evolution strategies. Different prompt variants are also explored to enhance the cross-model interaction. Experimental results show that our proposed black-box prompt tuning framework outperforms both hand-crafted prompt engineering and gradient-based prompt learning methods, which serves as evidence of its capability to train task-relevant prompts in a derivative-free manner.

----

## [187] DenseDINO: Boosting Dense Self-Supervised Learning with Token-Based Point-Level Consistency

**Authors**: *Yike Yuan, Xinghe Fu, Yunlong Yu, Xi Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/188](https://doi.org/10.24963/ijcai.2023/188)

**Abstract**:

In this paper, we propose a simple yet effective transformer framework for self-supervised learning called DenseDINO to learn dense visual representations. To exploit the spatial information that the dense prediction tasks require but neglected by the existing self-supervised transformers, we introduce point-level supervision across views in a novel token-based way. Specifically, DenseDINO introduces some extra input tokens called reference tokens to match the point-level features with the position prior. With the reference token, the model could maintain spatial consistency and deal with multi-object complex scene images, thus generalizing better on dense prediction tasks. Compared with the vanilla DINO, our approach obtains competitive performance when evaluated on classification in ImageNet and achieves a large margin (+7.2% mIoU) improvement in semantic segmentation on PascalVOC under the linear probing protocol for segmentation.

----

## [188] Linguistic More: Taking a Further Step toward Efficient and Accurate Scene Text Recognition

**Authors**: *Boqiang Zhang, Hongtao Xie, Yuxin Wang, Jianjun Xu, Yongdong Zhang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/189](https://doi.org/10.24963/ijcai.2023/189)

**Abstract**:

Vision model have gained increasing attention due to their simplicity and efficiency in Scene Text Recognition (STR) task. However, due to lacking the perception of linguistic knowledge and information, recent vision models suffer from two problems: (1) the pure vision-based query results in attention drift, which usually causes poor recognition and is summarized as linguistic insensitive drift (LID) problem in this paper. (2) the visual feature is suboptimal for the recognition in some vision-missing cases (e.g. occlusion, etc.). To address these issues, we propose a Linguistic Perception Vision model (LPV), which explores the linguistic capability of vision model for accurate text recognition. To alleviate the LID problem, we introduce a Cascade Position Attention (CPA) mechanism that obtains high-quality and accurate attention maps through step-wise optimization and linguistic information mining. Furthermore, a Global Linguistic Reconstruction Module (GLRM) is proposed to improve the representation of visual features by perceiving the linguistic information in the visual space, which gradually converts visual features into semantically rich ones during the cascade process. Different from previous methods, our method obtains SOTA results while keeping low complexity (92.4% accuracy with only 8.11M parameters). Code is available at https://github.com/CyrilSterling/LPV.

----

## [189] Spatially Covariant Lesion Segmentation

**Authors**: *Hang Zhang, Rongguang Wang, Jinwei Zhang, Dongdong Liu, Chao Li, Jiahao Li*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/190](https://doi.org/10.24963/ijcai.2023/190)

**Abstract**:

Compared to natural images, medical images usually show stronger visual patterns and therefore this adds flexibility and elasticity to resource-limited clinical applications by injecting proper priors into neural networks.
In this paper, we propose spatially covariant pixel-aligned classifier (SCP) to improve the computational efficiency and meantime maintain or increase accuracy for lesion segmentation.
SCP relaxes the spatial invariance constraint imposed by convolutional operations and optimizes an underlying implicit function that maps image coordinates to network weights, the parameters of which are obtained along with the backbone network training and later used for generating network weights to capture spatially covariant contextual information. 
We demonstrate the effectiveness and efficiency of the proposed SCP using two lesion segmentation tasks from different imaging modalities: white matter hyperintensity segmentation in magnetic resonance imaging and liver tumor segmentation in contrast-enhanced abdominal computerized tomography.
The network using SCP has achieved 23.8, 64.9 and 74.7 reduction in GPU memory usage, FLOPs, and network size with similar or better accuracy for lesion segmentation.

----

## [190] HOI-aware Adaptive Network for Weakly-supervised Action Segmentation

**Authors**: *Runzhong Zhang, Suchen Wang, Yueqi Duan, Yansong Tang, Yue Zhang, Yap-Peng Tan*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/191](https://doi.org/10.24963/ijcai.2023/191)

**Abstract**:

In this paper, we propose an HOI-aware adaptive network named AdaAct for weakly-supervised action segmentation. Most existing methods learn a fixed network to predict the action of each frame with the neighboring frames. However, this would result in ambiguity when estimating similar actions, such as pouring juice and pouring coffee. To address this, we aim to exploit temporally global but spatially local human-object interactions (HOI) as video-level prior knowledge for action segmentation. The long-term HOI sequence provides crucial contextual information to distinguish ambiguous actions, where our network dynamically adapts to the given HOI sequence at test time. More specifically, we first design a video HOI encoder that extracts, selects, and integrates the most representative HOI throughout the video. Then, we propose a two-branch HyperNetwork to learn an adaptive temporal encoder, which automatically adjusts the parameters based on the HOI information of various videos on the fly. Extensive experiments on two widely-used datasets including Breakfast and 50Salads demonstrate the effectiveness of our method under different evaluation metrics.

----

## [191] Learning Object Consistency and Interaction in Image Generation from Scene Graphs

**Authors**: *Yangkang Zhang, Chenye Meng, Zejian Li, Pei Chen, Guang Yang, Changyuan Yang, Lingyun Sun*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/192](https://doi.org/10.24963/ijcai.2023/192)

**Abstract**:

This paper is concerned with synthesizing images conditioned on a scene graph (SG), a set of object nodes and their edges of interactive relations. We divide existing works into image-oriented and code-oriented methods. In our analysis, the image-oriented methods do not consider object interaction in spatial hidden feature. On the other hand, in empirical study, the code-oriented methods lose object consistency as their generated images miss certain objects in the input scene graph. To alleviate these two issues, we propose Learning Object Consistency and Interaction (LOCI). To preserve object consistency, we design a consistency module with a weighted augmentation strategy for objects easy to be ignored and a matching loss between scene graphs and image codes. To learn object interaction, we design an interaction module consisting of three kinds of message propagation between the input scene graph and the learned image code. Experiments on COCO-stuff and Visual Genome datasets show our proposed method alleviates the ignorance of objects and outperforms the state-of-the-art on visual fidelity of generated images and objects.

----

## [192] Manifold-Aware Self-Training for Unsupervised Domain Adaptation on Regressing 6D Object Pose

**Authors**: *Yichen Zhang, Jiehong Lin, Ke Chen, Zelin Xu, Yaowei Wang, Kui Jia*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/193](https://doi.org/10.24963/ijcai.2023/193)

**Abstract**:

Domain gap between synthetic and real data in visual regression (e.g., 6D pose estimation) is bridged in this paper via global feature alignment and local refinement on the coarse classification of discretized anchor classes in target space, which imposes a piece-wise target manifold regularization into domain-invariant representation learning. Specifically, our method incorporates an explicit self-supervised manifold regularization, revealing consistent cumulative target dependency across domains, to a self-training scheme (e.g., the popular Self-Paced Self-Training) to encourage more discriminative transferable representations of regression tasks. Moreover, learning unified implicit neural functions to estimate relative direction and distance of targets to their nearest class bins aims to refine target classification predictions, which can gain robust performance against inconsistent feature scaling sensitive to UDA regressors. Experiment results on three public benchmarks of the challenging 6D pose estimation task can verify the effectiveness of our method, consistently achieving superior performance to the state-of-the-art for UDA on 6D pose estimation. Codes and pre-trained models are available https://github.com/Gorilla-Lab-SCUT/MAST.

----

## [193] FGNet: Towards Filling the Intra-class and Inter-class Gaps for Few-shot Segmentation

**Authors**: *Yuxuan Zhang, Wei Yang, Shaowei Wang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/194](https://doi.org/10.24963/ijcai.2023/194)

**Abstract**:

Current few-shot segmentation (FSS) approaches have made tremendous achievements based on prototypical learning techniques. However, due to the scarcity of the support data provided, FSS methods still suffer from the intra-class and inter-class gaps. In this paper, we propose a uniform network to fill both the gaps, termed FGNet. It consists of the novel design of a Self-Adaptive Module (SAM) to emphasize the query feature to generate an enhanced prototype for self-alignment. Such a prototype caters to each query sample itself since it contains the underlying intra-instance information, which gets around the intra-class appearance gap. Moreover, we design an Inter-class Feature Separation Module (IFSM) to separate the feature space of the target class from other classes, which contributes to bridging the inter-class gap. In addition, we present several new losses and a method termed B-SLIC, which help to further enhance the separation performance of FGNet. Experimental results show that FGNet reduces both the gaps for FSS by SAM and IFSM respectively, and achieves state-of-the-art performances on both PASCAL-5i and COCO-20i datasets compared with previous top-performing approaches.

----

## [194] MM-PCQA: Multi-Modal Learning for No-reference Point Cloud Quality Assessment

**Authors**: *Zicheng Zhang, Wei Sun, Xiongkuo Min, Qiyuan Wang, Jun He, Quan Zhou, Guangtao Zhai*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/195](https://doi.org/10.24963/ijcai.2023/195)

**Abstract**:

The visual quality of point clouds has been greatly emphasized since the ever-increasing 3D vision applications are expected to provide cost-effective and high-quality experiences for users.  Looking back on the development of point cloud quality assessment (PCQA), the visual quality is usually evaluated by utilizing single-modal information, i.e., either extracted from the 2D projections or 3D point cloud. The 2D projections contain rich texture and semantic information but are highly dependent on viewpoints, while the 3D point clouds are more sensitive to geometry distortions and invariant to viewpoints. Therefore, to leverage the advantages of both point cloud and projected image modalities, we propose a novel no-reference Multi-Modal Point Cloud Quality Assessment (MM-PCQA) metric. In specific, we split the point clouds into sub-models to represent local geometry distortions such as point shift and down-sampling. Then we render the point clouds into 2D image projections for texture feature extraction. To achieve the goals, the sub-models and projected images are encoded with point-based and image-based neural networks. Finally, symmetric cross-modal attention is employed to fuse multi-modal quality-aware information. Experimental results show that our approach outperforms all compared state-of-the-art methods and is far ahead of previous no-reference PCQA methods, which highlights the effectiveness of the proposed method. The code is available at https://github.com/zzc-1998/MM-PCQA.

----

## [195] STS-GAN: Can We Synthesize Solid Texture with High Fidelity from Arbitrary 2D Exemplar?

**Authors**: *Xin Zhao, Jifeng Guo, Lin Wang, Fanqi Li, Jiahao Li, Junteng Zheng, Bo Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/196](https://doi.org/10.24963/ijcai.2023/196)

**Abstract**:

Solid texture synthesis (STS), an effective way to extend a 2D exemplar to a 3D solid volume, exhibits advantages in computational photography. However, existing methods generally fail to accurately learn arbitrary textures, which may result in the failure to synthesize solid textures with high fidelity. In this paper, we propose a novel generative adversarial nets-based framework (STS-GAN) to extend the given 2D exemplar to arbitrary 3D solid textures. In STS-GAN, multi-scale 2D texture discriminators evaluate the similarity between the given 2D exemplar and slices from the generated 3D texture, promoting the 3D texture generator synthesizing realistic solid textures. Finally, experiments demonstrate that the proposed method can generate high-fidelity solid textures with similar visual characteristics to the 2D exemplar.

----

## [196] TPS++: Attention-Enhanced Thin-Plate Spline for Scene Text Recognition

**Authors**: *Tianlun Zheng, Zhineng Chen, Jinfeng Bai, Hongtao Xie, Yu-Gang Jiang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/197](https://doi.org/10.24963/ijcai.2023/197)

**Abstract**:

Text irregularities pose significant challenges to scene text recognizers. Thin-Plate Spline (TPS)-based rectification is widely regarded as an effective means to deal with them. Currently, the calculation of TPS transformation parameters purely depends on the quality of regressed text borders. It ignores the text content and often leads to unsatisfactory rectified results for severely distorted text. In this work, we introduce TPS++, an attention-enhanced TPS transformation that incorporates the attention mechanism to text rectification for the first time. TPS++ formulates the parameter calculation as a joint process of foreground control point regression and content-based attention score estimation, which is computed by a dedicated designed gated-attention block. TPS++ builds a more flexible content-aware rectifier, generating a natural text correction that is easier to read by the subsequent recognizer. Moreover, TPS++ shares the feature backbone with the recognizer in part and implements the rectification at feature-level rather than image-level, incurring only a small overhead in terms of parameters and inference time. Experiments on public benchmarks show that TPS++ consistently improves the recognition and achieves state-of-the-art accuracy. Meanwhile, it generalizes well on different backbones and recognizers. Code is at https://github.com/simplify23/TPS_PP.

----

## [197] Video Frame Interpolation with Densely Queried Bilateral Correlation

**Authors**: *Chang Zhou, Jie Liu, Jie Tang, Gangshan Wu*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/198](https://doi.org/10.24963/ijcai.2023/198)

**Abstract**:

Video Frame Interpolation (VFI) aims to synthesize non-existent intermediate frames between existent frames. Flow-based VFI algorithms estimate intermediate motion fields to warp the existent frames. Real-world motions' complexity and the reference frame's absence make motion estimation challenging. Many state-of-the-art approaches explicitly model the correlations between two neighboring frames for more accurate motion estimation. In common approaches, the receptive field of correlation modeling at higher resolution depends on the motion fields estimated beforehand. Such receptive field dependency makes common motion estimation approaches poor at coping with small and fast-moving objects. To better model correlations and to produce more accurate motion fields, we propose the Densely Queried Bilateral Correlation (DQBC) that gets rid of the receptive field dependency problem and thus is more friendly to small and fast-moving objects. The motion fields generated with the help of DQBC are further refined and up-sampled with context features. After the motion fields are fixed, a CNN-based SynthNet synthesizes the final interpolated frame. Experiments show that our approach enjoys higher accuracy and less inference time than the state-of-the-art. Source code is available at https://github.com/kinoud/DQBC.

----

## [198] Pyramid Diffusion Models for Low-light Image Enhancement

**Authors**: *Dewei Zhou, Zongxin Yang, Yi Yang*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/199](https://doi.org/10.24963/ijcai.2023/199)

**Abstract**:

Recovering noise-covered details from low-light images is challenging, and the results given by previous methods leave room for improvement. Recent diffusion models show realistic and detailed image generation through a sequence of denoising refinements and motivate us to introduce them to low-light image enhancement for recovering realistic details. However, we found two problems when doing this, i.e., 1) diffusion models keep constant resolution in one reverse process, which limits the speed; 2) diffusion models sometimes result in global degradation (e.g., RGB shift). To address the above problems, this paper proposes a Pyramid Diffusion model (PyDiff) for low-light image enhancement. PyDiff uses a novel pyramid diffusion method to perform sampling in a pyramid resolution style (i.e., progressively increasing resolution in one reverse process). Pyramid diffusion makes PyDiff much faster than vanilla diffusion models and introduces no performance degradation. Furthermore, PyDiff uses a global corrector to alleviate the global degradation that may occur in the reverse process, significantly improving the performance and making the training of diffusion models easier with little additional computational consumption. Extensive experiments on popular benchmarks show that PyDiff achieves superior performance and efficiency. Moreover, PyDiff can generalize well to unseen noise and illumination distributions. Code and supplementary materials are available at https://github.com/limuloo/PyDIff.git.

----

## [199] CADParser: A Learning Approach of Sequence Modeling for B-Rep CAD

**Authors**: *Shengdi Zhou, Tianyi Tang, Bin Zhou*

**Conference**: *ijcai 2023*

**URL**: [https://doi.org/10.24963/ijcai.2023/200](https://doi.org/10.24963/ijcai.2023/200)

**Abstract**:

Computer-Aided Design (CAD) plays a crucial role in industrial manufacturing by providing geometry information and the construction workflow for manufactured objects. The construction information enables effective re-editing of parametric CAD models. While boundary representation (B-Rep) is the standard format for representing geometry structures, JSON format is an alternative due to the lack of uniform criteria for storing the construction workflow. Regrettably, most CAD models available on the Internet only offer geometry information, omitting the construction procedure and hampering creation efficiency. This paper proposes a learning approach CADParser to infer the underlying modeling sequences given a B-Rep CAD model. It achieves this by treating the CAD geometry structure as a graph and the construction workflow as a sequence. Since the existing CAD dataset only contains two operations (i.e., Sketch and Extrusion), limiting the diversity of the CAD model creation, we also introduce a large-scale dataset incorporating a more comprehensive range of operations such as Revolution, Fillet, and Chamfer. Each model includes both the geometry structure and the construction sequences. Extensive experiments demonstrate that our method can compete with the existing state-of-the-art methods quantitatively and qualitatively. Data is available at https://drive.google.com/CADParserData.

----



[Go to the next page](IJCAI-2023-list02.md)

[Go to the catalog section](README.md)