## [0] Anytime Capacity Expansion in Medical Residency Match by Monte Carlo Tree Search

**Authors**: *Kenshi Abe, Junpei Komiyama, Atsushi Iwasaki*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/1](https://doi.org/10.24963/ijcai.2022/1)

**Abstract**:

This paper considers the capacity expansion problem in two-sided matchings, where the policymaker is allowed to allocate some extra seats as well as the standard seats. In medical residency match, each hospital accepts a limited number of doctors. Such capacity constraints are typically given in advance. However, such exogenous constraints can compromise the welfare of the doctors; some popular hospitals inevitably dismiss some of their favorite doctors. Meanwhile, it is often the case that the hospitals are also benefited to accept a few extra doctors. To tackle the problem, we propose an anytime method that the upper confidence tree searches the space of capacity expansions, each of which has a resident-optimal stable assignment that the deferred acceptance method finds. Constructing a good search tree representation significantly boosts the performance of the proposed method. Our simulation shows that the proposed method identifies an almost optimal capacity expansion with a significantly smaller computational budget than exact methods based on mixed-integer programming.

----

## [1] Socially Intelligent Genetic Agents for the Emergence of Explicit Norms

**Authors**: *Rishabh Agrawal, Nirav Ajmeri, Munindar P. Singh*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/2](https://doi.org/10.24963/ijcai.2022/2)

**Abstract**:

Norms help regulate a society. Norms may be explicit (represented in structured form) or implicit. 
We address the emergence of explicit norms by developing agents who provide and reason about explanations for norm violations in deciding sanctions and identifying alternative norms. These agents use a genetic algorithm to produce norms and reinforcement learning to learn the values of these norms.
We find that applying explanations leads to norms that provide better cohesion and goal satisfaction for the agents. Our results are stable for societies with differing attitudes of generosity.

----

## [2] An EF2X Allocation Protocol for Restricted Additive Valuations

**Authors**: *Hannaneh Akrami, Rojin Rezvan, Masoud Seddighin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/3](https://doi.org/10.24963/ijcai.2022/3)

**Abstract**:

We study the problem of fairly allocating a set of indivisible goods to a set of n agents.  Envy-freeness up to any good (EFX) criterion (which 
requires that no agent prefers the bundle of another agent after the removal of any single good) is known to be a remarkable analogue of envy-freeness when the resource is a set of indivisible goods.
In this paper, we investigate EFX  for restricted additive valuations, that is, every good has a non-negative value, and every agent is interested in only some of the goods.

We introduce a natural relaxation of EFX called EFkX which requires that no agent envies another agent after the removal of any k goods. Our main contribution is an algorithm that finds a complete (i.e., no good is discarded) EF2X allocation for restricted additive valuations. In our algorithm we devise new concepts, namely configuration and envy-elimination that might be of independent interest. 

We also use our new tools to find an EFX allocation for restricted additive valuations that discards at most n/2 -1 goods.

----

## [3] Better Collective Decisions via Uncertainty Reduction

**Authors**: *Shiri Alouf-Heffetz, Laurent Bulteau, Edith Elkind, Nimrod Talmon, Nicholas Teh*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/4](https://doi.org/10.24963/ijcai.2022/4)

**Abstract**:

We consider an agent community wishing to decide on several binary issues by means of issue-by-issue majority voting. For each issue and each agent, one of the two options is better than the other. However, some of the agents may be confused about some of the issues, in which case they may vote for the option that is objectively worse for them. A benevolent external party wants to help the agents to make better decisions, i.e., select the majority-preferred option for as many issues as possible. This party may have one of the following tools at its disposal: (1) educating some of the agents, so as to enable them to vote correctly on all issues, 
(2) appointing a subset of highly competent agents to make decisions on behalf of the entire group, or (3) guiding the agents on how to delegate their votes to other agents, in a way that is consistent with the agents' opinions. For each of these tools, we study the complexity of the decision problem faced by this external party, obtaining both NP-hardness results and fixed-parameter tractability results.

----

## [4] How Should We Vote? A Comparison of Voting Systems within Social Networks

**Authors**: *Shiri Alouf-Heffetz, Ben Armstrong, Kate Larson, Nimrod Talmon*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/5](https://doi.org/10.24963/ijcai.2022/5)

**Abstract**:

Voting is a crucial methodology for eliciting and combining agents' preferences and information across many applications. Just as there are numerous voting rules exhibiting different properties, we also see many different voting systems. In this paper we investigate how different voting systems perform as a function of the characteristics of the underlying voting population and social network. In particular, we compare direct democracy, liquid democracy, and sortition in a ground truth voting context.

Through simulations -- using both real and artificially generated social networks -- we illustrate how voter competency distributions and levels of direct participation affect group accuracy differently in each voting mechanism. Our results can be used to guide the selection of a suitable voting system based on the characteristics of a particular voting setting.

----

## [5] Public Signaling in Bayesian Ad Auctions

**Authors**: *Francesco Bacchiocchi, Matteo Castiglioni, Alberto Marchesi, Giulia Romano, Nicola Gatti*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/6](https://doi.org/10.24963/ijcai.2022/6)

**Abstract**:

We study signaling in Bayesian ad auctions, in which bidders' valuations depend on a random, unknown state of nature. The auction mechanism has complete knowledge of the actual state of nature, and it can send signals to bidders so as to disclose information about the state and increase revenue. For instance, a state may collectively encode some features of the user that are known to the mechanism only, since the latter has access to data sources unaccessible to the bidders. We study the problem of computing how the mechanism should send signals to bidders in order to maximize revenue. While this problem has already been addressed in the easier setting of second-price auctions, to the best of our knowledge, our work is the first to explore ad auctions with more than one slot. In this paper, we focus on public signaling and VCG mechanisms, under which bidders truthfully report their valuations. We start with a negative result, showing that, in general, the problem does not admit a PTAS unless P = NP, even when bidders' valuations are known to the mechanism. The rest of the paper is devoted to settings in which such negative result can be circumvented. First, we prove that, with known valuations, the problem can indeed be solved in polynomial time when either the number of states d or the number of slots m is fixed. Moreover, in the same setting, we provide an FPTAS for the case in which bidders are single minded, but d and m can be arbitrary. Then, we switch to the random valuations setting, in which these are randomly drawn according to some probability distribution. In this case, we show that the problem admits an FPTAS, a PTAS, and a QPTAS, when, respectively, d is fixed, m is fixed, and bidders' valuations are bounded away from zero.

----

## [6] Mixed Strategies for Security Games with General Defending Requirements

**Authors**: *Rufan Bai, Haoxing Lin, Xinyu Yang, Xiaowei Wu, Minming Li, Weijia Jia*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/7](https://doi.org/10.24963/ijcai.2022/7)

**Abstract**:

The Stackelberg security game is played between a defender and an attacker, where the defender needs to allocate a limited amount of resources to multiple targets in order to minimize the loss due to adversarial attack by the attacker. While allowing targets to have different values, classic settings often assume uniform requirements to defend the targets. This enables existing results that study mixed strategies (randomized allocation algorithms) to adopt a compact representation of the mixed strategies.
In this work, we initiate the study of mixed strategies for the security games in which the targets can have different defending requirements. In contrast to the case of uniform defending requirement, for which an optimal mixed strategy can be computed efficiently, we show that computing the optimal mixed strategy is NP-hard for the general defending requirements setting. However, we show that strong upper and lower bounds for the optimal mixed strategy defending result can be derived. We propose an efficient close-to-optimal Patching algorithm that computes mixed strategies that use only few pure strategies. We also study the setting when the game is played on a network and resource sharing is enabled between neighboring targets. Our experimental results demonstrate the effectiveness of our algorithm in several large real-world datasets.

----

## [7] Envy-Free and Pareto-Optimal Allocations for Agents with Asymmetric Random Valuations

**Authors**: *Yushi Bai, Paul Gölz*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/8](https://doi.org/10.24963/ijcai.2022/8)

**Abstract**:

We study the problem of allocating m indivisible items to n agents with additive utilities. It is desirable for the allocation to be both fair and efficient, which we formalize through the notions of envy-freeness and Pareto-optimality. While envy-free and Pareto-optimal allocations may not exist for arbitrary utility profiles, previous work has shown that such allocations exist with high probability assuming that all agents’ values for all items are independently drawn from a common distribution. In this paper, we consider a generalization of this model where each agent’s utilities are drawn independently from a distribution specific to the agent. We show that envy-free and Pareto-optimal allocations are likely to exist in this asymmetric model when m=Ω(n log n), which is tight up to a log log gap that also remains open in the symmetric subsetting. Furthermore, these guarantees can be achieved by a polynomial-time algorithm.

----

## [8] Achieving Envy-Freeness with Limited Subsidies under Dichotomous Valuations

**Authors**: *Siddharth Barman, Anand Krishna, Yadati Narahari, Soumyarup Sadhukhan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/9](https://doi.org/10.24963/ijcai.2022/9)

**Abstract**:

We study the problem of allocating indivisible goods among agents in a fair manner. While envy-free allocations of indivisible goods are not guaranteed to exist, envy-freeness can be achieved by additionally providing some subsidy to the agents. These subsidies can be alternatively viewed as a divisible good (money) that is fractionally assigned among the agents to realize an envy-free outcome. In this setup, we bound the subsidy required to attain envy-freeness among agents with dichotomous valuations, i.e., among agents whose marginal value for any good is either zero or one.  

We prove that, under dichotomous valuations, there exists an allocation that achieves envy-freeness with a per-agent subsidy of either 0 or 1. Furthermore, such an envy-free solution can be computed efficiently in the standard value-oracle model. Notably, our results hold for general dichotomous valuations and, in particular, do not require the (dichotomous) valuations to be additive, submodular, or even subadditive. Also, our subsidy bounds are tight and provide a linear (in the number of agents) factor improvement over the bounds known for general monotone valuations.

----

## [9] Transparency, Detection and Imitation in Strategic Classification

**Authors**: *Flavia Barsotti, Rüya Gökhan Koçer, Fernando P. Santos*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/10](https://doi.org/10.24963/ijcai.2022/10)

**Abstract**:

Given the ubiquity of AI-based decisions that affect individualsâ€™ lives, providing transparent explanations about algorithms is ethically sound and often legally mandatory. How do individuals strategically adapt following explanations? What are the consequences of adaptation for algorithmic accuracy? We simulate the interplay between explanations shared by an Institution (e.g. a bank) and the dynamics of strategic adaptation by Individuals reacting to such feedback. Our model identifies key aspects related to strategic adaptation and the challenges that an institution could face as it attempts to provide explanations. Resorting to an agent-based approach, our model scrutinizes: i) the impact of transparency in explanations, ii) the interaction between faking behavior and detection capacity and iii) the role of behavior imitation. We find that the risks of transparent explanations are alleviated if effective methods to detect faking behaviors are in place. Furthermore, we observe that behavioral imitation --- as often happens across societies --- can alleviate malicious adaptation and contribute to accuracy, even after transparent explanations.

----

## [10] Time-Constrained Participatory Budgeting Under Uncertain Project Costs

**Authors**: *Dorothea Baumeister, Linus Boes, Christian Laußmann*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/11](https://doi.org/10.24963/ijcai.2022/11)

**Abstract**:

In participatory budgeting the stakeholders collectively decide which projects from a set of proposed projects should be implemented.
This decision underlies both time and monetary constraints.
In reality it is often impossible to figure out the exact cost of each project in advance, it is only known after a project is finished.
To reduce risk, one can implement projects one after the other to be able to react to higher costs of a previous project.
However, this will increase execution time drastically.
We generalize existing frameworks to capture this setting, study desirable properties of algorithms for this problem, and show that some desirable properties are incompatible.
Then we present and analyze algorithms that trade-off desirable properties.

----

## [11] Tolerance is Necessary for Stability: Single-Peaked Swap Schelling Games

**Authors**: *Davide Bilò, Vittorio Bilò, Pascal Lenzner, Louise Molitor*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/12](https://doi.org/10.24963/ijcai.2022/12)

**Abstract**:

Residential segregation in metropolitan areas is a phenomenon that can be observed all over the world. Recently, this was investigated via game-theoretic models. There, selfish agents of two types are equipped with a monotone utility function that ensures higher utility if an agent has more same-type neighbors. The agents strategically choose their location on a given graph that serves as residential area to maximize their utility. However, sociological polls suggest that real-world agents are actually favoring mixed-type neighborhoods, and hence should be modeled via non-monotone utility functions. To address this, we study Swap Schelling Games with single-peaked utility functions. Our main finding is that tolerance, i.e., agents favoring fifty-fifty neighborhoods or being in the minority, is necessary for equilibrium existence on almost regular or bipartite graphs. Regarding the quality of equilibria, we derive (almost) tight bounds on the Price of Anarchy and the Price of Stability. In particular, we show that the latter is constant on bipartite and almost regular graphs.

----

## [12] General Opinion Formation Games with Social Group Membership

**Authors**: *Vittorio Bilò, Diodato Ferraioli, Cosimo Vinci*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/13](https://doi.org/10.24963/ijcai.2022/13)

**Abstract**:

Modeling how agents form their opinions is of paramount importance for designing marketing and electoral campaigns. In this work, we present a new framework for opinion formation which generalizes the well-known Friedkin-Johnsen model by incorporating three important features: (i) social group membership, that limits the amount of influence that people not belonging to the same group may lead on a given agent; (ii) both attraction among friends, and repulsion among enemies; (iii) different strengths of influence lead from different people on a given agent, even if the social relationships among them are the same.

We show that, despite its generality, our model always admits a pure Nash equilibrium which, under opportune mild conditions, is even unique. Next, we analyze the performances of these equilibria with respect to a social objective function defined as a convex combination, parametrized by a value λ∈[0,1], of the costs yielded by the untruthfulness of the declared opinions and the total cost of social pressure. We prove bounds on both the price of anarchy and the price of stability which show that, for not-too-extreme values of λ, performance at equilibrium are very close to optimal ones. For instance, in several interesting scenarios, the prices of anarchy and stability are both equal to max{2λ,1-λ}/min{2λ,1-λ} which never exceeds 2 for λ∈[1/5,1/2].

----

## [13] Fair Equilibria in Sponsored Search Auctions: The Advertisers' Perspective

**Authors**: *Georgios Birmpas, Andrea Celli, Riccardo Colini-Baldeschi, Stefano Leonardi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/14](https://doi.org/10.24963/ijcai.2022/14)

**Abstract**:

In this work we introduce a new class of mechanisms composed of a traditional Generalized Second Price (GSP) auction, and a fair division scheme in order to achieve some desired level of fairness between groups of Bayesian strategic advertisers. We propose two mechanisms, beta-Fair GSP and GSP-EFX, that compose GSP with, respectively, an envy-free up to one item, and an envy-free up to any item fair division scheme. The payments of GSP are adjusted in order to compensate advertisers that suffer a loss of efficiency due the fair division stage. We investigate the strategic learning implications of the deployment of sponsored search auction mechanisms that obey to such fairness criteria. We prove that, for both mechanisms, if bidders play so as to minimize their external regret they are guaranteed to reach an equilibrium with good social welfare. We also prove that the mechanisms are budget balanced, so that the payments charged by the traditional GSP mechanism are a good proxy of the total compensation offered to the advertisers. Finally, we evaluate the quality of the allocations through experiments on real-world data.

----

## [14] Understanding Distance Measures Among Elections

**Authors**: *Niclas Boehmer, Piotr Faliszewski, Rolf Niedermeier, Stanislaw Szufa, Tomasz Was*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/15](https://doi.org/10.24963/ijcai.2022/15)

**Abstract**:

Motivated by putting empirical work based on (synthetic) election data on a more solid mathematical basis, we analyze six distances among elections, including, e.g., the challenging-to-compute but very precise swap distance and the distance used to form the so-called map of elections. Among the six, the latter seems to strike the best balance between its computational complexity and expressiveness.

----

## [15] Toward Policy Explanations for Multi-Agent Reinforcement Learning

**Authors**: *Kayla Boggess, Sarit Kraus, Lu Feng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/16](https://doi.org/10.24963/ijcai.2022/16)

**Abstract**:

Advances in multi-agent reinforcement learning (MARL) enable sequential decision making for a range of exciting multi-agent applications such as cooperative AI and autonomous driving. Explaining agent decisions is crucial for improving system transparency, increasing user satisfaction, and facilitating human-agent collaboration. However, existing works on explainable reinforcement learning mostly focus on the single-agent setting and are not suitable for addressing challenges posed by multi-agent environments. We present novel methods to generate two types of policy explanations for MARL: (i) policy summarization about the agent cooperation and task sequence, and (ii) language explanations to answer queries about agent behavior. Experimental results on three MARL domains demonstrate the scalability of our methods. A user study shows that the generated explanations significantly improve user performance and increase subjective ratings on metrics such as user satisfaction.

----

## [16] Distortion in Voting with Top-t Preferences

**Authors**: *Allan Borodin, Daniel Halpern, Mohamad Latifian, Nisarg Shah*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/17](https://doi.org/10.24963/ijcai.2022/17)

**Abstract**:

A fundamental question in social choice and multi-agent systems is aggregating ordinal preferences expressed by agents into a measurably prudent collective choice. A promising line of recent work views ordinal preferences as a proxy for underlying cardinal preferences. It aims to optimize distortion, the worst-case approximation ratio of the (utilitarian) social welfare. When agents rank the set of alternatives, prior work identifies near-optimal voting rules for selecting one or more alternatives. However, ranking all the alternatives is prohibitive when there are many alternatives. 

In this work, we consider the setting where each agent ranks only her t favorite alternatives and identify almost tight bounds on the best possible distortion when selecting a single alternative or a committee of alternatives of a given size k. Our results also extend to approximating higher moments of social welfare. Along the way, we close a gap left open in prior work by identifying asymptotically tight distortion bounds for committee selection given full rankings.

----

## [17] Let's Agree to Agree: Targeting Consensus for Incomplete Preferences through Majority Dynamics

**Authors**: *Sirin Botan, Simon Rey, Zoi Terzopoulou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/18](https://doi.org/10.24963/ijcai.2022/18)

**Abstract**:

We study settings in which agents with incomplete
preferences need to make a collective decision. We
focus on a process of majority dynamics where issues
are addressed one at a time and undecided
agents follow the opinion of the majority. We assess
the effects of this process on various consensus
notions—such as the Condorcet winner—and show
that in the worst case, myopic adherence to the majority
damages existing consensus; yet, simulation
experiments indicate that the damage is often mild.
We also examine scenarios where the chair of the
decision process can control the existence (or the
identity) of consensus, by determining the order in
which the issues are discussed.

----

## [18] Incentives in Social Decision Schemes with Pairwise Comparison Preferences

**Authors**: *Felix Brandt, Patrick Lederer, Warut Suksompong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/19](https://doi.org/10.24963/ijcai.2022/19)

**Abstract**:

Social decision schemes (SDSs) map the preferences of individual voters over multiple alternatives to a probability distribution over the alternatives. In order to study properties such as efficiency, strategyproofness, and participation for SDSs, preferences over alternatives are typically lifted to preferences over lotteries using the notion of stochastic dominance (SD). However, requiring strategyproofness or strict participation with respect to this preference extension only leaves room for rather undesirable SDSs such as random dictatorships. Hence, we focus on the natural but little understood pairwise comparison (PC) preference extension, which postulates that one lottery is preferred to another if the former is more likely to return a preferred outcome. In particular, we settle three open questions raised by Brandt in Rolling the dice: Recent results in probabilistic social choice (2017): (i) there is no Condorcet-consistent SDS that satisfies PC-strategyproofness; (ii) there is no anonymous and neutral SDS that satisfies PC-efficiency and PC-strategyproofness; and (iii) there is no anonymous and neutral SDS that satisfies PC-efficiency and strict PC-participation. All three impossibilities require m>=4 alternatives and turn into possibilities when m<=3.

----

## [19] Single-Peaked Opinion Updates

**Authors**: *Robert Bredereck, Anne-Marie George, Jonas Israel, Leon Kellerhals*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/20](https://doi.org/10.24963/ijcai.2022/20)

**Abstract**:

We consider opinion diffusion for undirected networks with sequential updates when the opinions of the agents are single-peaked preference rankings. Our starting point is the study of preserving single-peakedness. We identify voting rules that, when given a single-peaked profile, output at least one ranking that is single peaked w.r.t. a single-peaked axis of the input. For such voting rules we show convergence to a stable state of the diffusion process that uses the voting rule as the agents' update rule. Further, we establish an efficient algorithm that maximises the spread of extreme opinions.

----

## [20] When Votes Change and Committees Should (Not)

**Authors**: *Robert Bredereck, Till Fluschnik, Andrzej Kaczmarczyk*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/21](https://doi.org/10.24963/ijcai.2022/21)

**Abstract**:

Electing a single committee of a small size is a classical and well-understood voting situation. Being interested in a sequence of committees, we introduce two time-dependent multistage models based on simple scoring-based voting. Therein, we are given a sequence of voting profiles (stages) over the same set of agents and candidates, and our task is to find a small committee for each stage of high score. In the conservative model we additionally require that any two consecutive committees have a small symmetric difference. Analogously, in the revolutionary model we require large symmetric differences. We prove both models to be NP-hard even for a constant number of agents, and, based on this, initiate a parameterized complexity analysis for the most natural parameters and combinations thereof. Among other results, we prove both models to be in XP yet W[1]-hard regarding the number of stages, and that being revolutionary seems to be "easier" than being conservative.

----

## [21] Network Creation with Homophilic Agents

**Authors**: *Martin Bullinger, Pascal Lenzner, Anna Melnichenko*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/22](https://doi.org/10.24963/ijcai.2022/22)

**Abstract**:

Network Creation Games are an important framework for understanding the formation of real-world networks. These games usually assume a set of indistinguishable agents strategically buying edges at a uniform price leading to a network among them. However, in real life, agents are heterogeneous and their relationships often display a bias towards similar agents, say of the same ethnic group. This homophilic behavior on the agent level can then lead to the emergent global phenomenon of social segregation. We study Network Creation Games with multiple types of homophilic agents and non-uniform edge cost, introducing two models focusing on the perception of same-type and different-type neighboring agents, respectively. Despite their different initial conditions, both our theoretical and experimental analysis show that both the composition and segregation strength of the resulting stable networks are almost identical, indicating a robust structure of social networks under homophily.

----

## [22] VidyutVanika21: An Autonomous Intelligent Broker for Smart-grids

**Authors**: *Sanjay Chandlekar, Bala Suraj Pedasingu, Easwar Subramanian, Sanjay Bhat, Praveen Paruchuri, Sujit Gujar*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/23](https://doi.org/10.24963/ijcai.2022/23)

**Abstract**:

An autonomous broker that liaises between retail customers and power-generating companies (GenCos) is essential for the smart grid ecosystem. The efficiency brought in by such brokers to the smart grid setup can be studied through a well-developed simulation environment. In this paper, we describe the design of one such energy broker called VidyutVanika21 (VV21) and analyze its performance using a simulation platform called PowerTAC (PowerTrading Agent Competition). Specifically, we discuss the retail (VV21–RM) and wholesale market (VV21–WM) modules of VV21 that help the broker achieve high net profits in a competitive setup. Supported by game-theoretic analysis, the VV21–RM designs tariff contracts that a) maintain a balanced portfolio of different types of customers; b) sustain an appropriate level of market share, and c) introduce surcharges on customers to reduce energy usage during peak demand times. The VV21–WM aims to reduce the cost of procurement by following the supply curve of the GenCo to identify its lowest ask for a particular auction which is then used to generate suitable bids. We further demonstrate the efficacy of the retail and wholesale strategies of VV21 in PowerTAC 2021 finals and through several controlled experiments.

----

## [23] Optimal Anonymous Independent Reward Scheme Design

**Authors**: *Mengjing Chen, Pingzhong Tang, Zihe Wang, Shenke Xiao, Xiwang Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/24](https://doi.org/10.24963/ijcai.2022/24)

**Abstract**:

We consider designing reward schemes that incentivize agents to create high-quality content (e.g., videos, images, text, ideas). The problem is at the center of a real-world application where the goal is to optimize the overall quality of generated content on user-generated content platforms. We focus on anonymous independent reward schemes (AIRS) that only take the quality of an agent's content as input. We prove the general problem is NP-hard. If the cost function is convex, we show the optimal AIRS can be formulated as a convex optimization problem and propose an efficient algorithm to solve it. Next, we explore the optimal linear reward scheme and prove it has a 1/2-approximation ratio, and the ratio is tight. Lastly, we show the proportional scheme can be arbitrarily bad compared to AIRS.

----

## [24] Goal Consistency: An Effective Multi-Agent Cooperative Method for Multistage Tasks

**Authors**: *Xinning Chen, Xuan Liu, Shigeng Zhang, Bo Ding, Kenli Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/25](https://doi.org/10.24963/ijcai.2022/25)

**Abstract**:

Although multistage tasks involving multiple sequential goals are common in real-world applications, they are not fully studied in multi-agent reinforcement learning (MARL). To accomplish a multi-stage task, agents have to achieve cooperation on different subtasks. Exploring the collaborative patterns of different subtasks and the sequence of completing the subtasks leads to an explosion in the search space, which poses great challenges to policy learning. Existing works designed for single-stage tasks where agents learn to cooperate only once usually suffer from low sample efficiency in multi-stage tasks as agents explore aimlessly. Inspired by humanâ€™s improving cooperation through goal consistency, we propose Multi-Agent Goal Consistency (MAGIC) framework to improve sample efficiency for learning in multi-stage tasks. MAGIC adopts a goal-oriented actor-critic model to learn both local and global views of goal cognition, which helps agents understand the task at the goal level so that they can conduct targeted exploration accordingly. Moreover, to improve exploration efficiency, MAGIC employs two-level goal consistency training to drive agents to formulate a consistent goal cognition. Experimental results show that MAGIC significantly improves sample efficiency and facilitates cooperation among agents compared with state-of-art MARL algorithms in several challenging multistage tasks.

----

## [25] On the Convergence of Fictitious Play: A Decomposition Approach

**Authors**: *Yurong Chen, Xiaotie Deng, Chenchen Li, David Mguni, Jun Wang, Xiang Yan, Yaodong Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/26](https://doi.org/10.24963/ijcai.2022/26)

**Abstract**:

Fictitious play (FP) is one of the most fundamental game-theoretical learning frameworks for computing Nash equilibrium in n-player games, which builds the foundation for modern multi-agent learning algorithms. Although FP has provable convergence guarantees on zero-sum games and potential games, many real-world problems are often a mixture of both and the convergence property of FP has not been fully studied yet. In this paper, we extend the convergence results of FP to the combinations of such games and beyond. Specifically, we derive new conditions for FP to converge by leveraging game decomposition techniques. We further develop a linear relationship unifying cooperation and competition in the sense that these two classes of games are mutually transferable. Finally, we analyse a non-convergent example of FP, the Shapley game, and develop sufficient conditions for FP to converge.

----

## [26] Two-Sided Matching over Social Networks

**Authors**: *Sung-Ho Cho, Taiki Todo, Makoto Yokoo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/27](https://doi.org/10.24963/ijcai.2022/27)

**Abstract**:

A new paradigm of mechanism design, called mechanism design over social networks, investigates agentsâ€™ incentives to diffuse the information of mechanisms to their followers over social networks. In this paper we consider it for two-sided matching, where the agents on one side, say students, are distributed over social networks and thus are not fully observable to the mechanism designer, while the agents on the other side, say colleges, are known a priori. The main purpose of this paper is to clarify the existence of mechanisms that satisfy several properties that are classified into four criteria: incentive constraints, efficiency constraints, stability constraints, and fairness constraints. We proposed three mechanisms and showed that no mechanism is better than these mechanisms, i.e., they are in the Pareto frontier according to the set of properties defined in this paper.

----

## [27] A Formal Model for Multiagent Q-Learning Dynamics on Regular Graphs

**Authors**: *Chen Chu, Yong Li, Jinzhuo Liu, Shuyue Hu, Xuelong Li, Zhen Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/28](https://doi.org/10.24963/ijcai.2022/28)

**Abstract**:

Modeling the dynamics of multi-agent learning has long been an important research topic. The focus of previous research has been either on 2-agent settings or well-mixed infinitely large agent populations. In this paper, we consider the scenario where n Q-learning agents locate on regular graphs, such that agents can only interact with their neighbors. We examine the local interactions between individuals and their neighbors, and derive a formal model to capture the Q-value dynamics of the entire population. Through comparisons with agent-based simulations on different types of regular graphs, we show that our model describes the agent learning dynamics in an exact manner.

----

## [28] Preserving Consistency in Multi-Issue Liquid Democracy

**Authors**: *Rachael Colley, Umberto Grandi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/29](https://doi.org/10.24963/ijcai.2022/29)

**Abstract**:

Liquid democracy bridges the gap between direct and representative democracy by allowing agents to vote directly on an issue or delegate to a trusted voter. Yet, when applied to votes on multiple interconnected issues, liquid democracy can lead agents to submit inconsistent votes. Two approaches are possible to maintain consistency: either modify the voters' ballots by ignoring problematic delegations, or resolve all delegations and make changes to the final votes of the agents. We show that rules based on minimising such changes are NP-complete. We propose instead to elicit and apply the agents' priorities over the delegated issues, designing and analysing two algorithms that find consistent votes from the agents' delegations in polynomial time.

----

## [29] Voting in Two-Crossing Elections

**Authors**: *Andrei Constantinescu, Roger Wattenhofer*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/30](https://doi.org/10.24963/ijcai.2022/30)

**Abstract**:

We introduce two-crossing elections as a generalization of single-crossing elections, showing a number of new results. First, we show that two-crossing elections can be recognized in polynomial time, by reduction to the well-studied consecutive ones problem. Single-crossing elections exhibit a transitive majority relation, from which many important results follow. On the other hand, we show that the classical Debord-McGarvey theorem can still be proven two-crossing, implying that any weighted majority tournament is inducible by a two-crossing election. This shows that many voting rules are NP-hard under two-crossing elections, including Kemeny and Slater. This is in contrast to the single-crossing case and outlines an important complexity boundary between single- and two-crossing. Subsequently, we show that for two-crossing elections the Young scores of all candidates can be computed in polynomial time, by formulating a totally unimodular linear program. Finally, we consider the Chamberlin-Courant rule with arbitrary disutilities and show that a winning committee can be computed in polynomial time, using an approach based on dynamic programming.

----

## [30] Multi-Agent Intention Progression with Reward Machines

**Authors**: *Michael Dann, Yuan Yao, Natasha Alechina, Brian Logan, John Thangarajah*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/31](https://doi.org/10.24963/ijcai.2022/31)

**Abstract**:

Recent work in multi-agent intention scheduling has shown that enabling agents to predict the actions of other agents when choosing their own actions can be beneficial. However existing approaches to 'intention-aware' scheduling assume that the programs of other agents are known, or are "similar" to that of the agent making the prediction. While this assumption is reasonable in some circumstances, it is less plausible when the agents are not co-designed. In this paper, we present a new approach to multi-agent intention scheduling in which agents predict the actions of other agents based on a high-level specification of the tasks performed by an agent in the form of a reward machine (RM) rather than on its (assumed) program. We show how a reward machine can be used to generate tree and rollout policies for an MCTS-based scheduler. We evaluate our approach in a range of multi-agent environments, and show that RM-based scheduling out-performs previous intention-aware scheduling approaches in settings where agents are not co-designed

----

## [31] An Analysis of the Linear Bilateral ANAC Domains Using the MiCRO Benchmark Strategy

**Authors**: *Dave de Jonge*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/32](https://doi.org/10.24963/ijcai.2022/32)

**Abstract**:

The Automated Negotiating Agents Competition (ANAC) is an annual competition that compares the state-of-the-art algorithms in the field of automated negotiation. Although in recent years ANAC has given more and more attention to more complex scenarios, the linear and bilateral negotiation domains that were used for its first few editions are still widely used as the default benchmark in automated negotiations research. In this paper, however, we argue that these domains should no longer be used, because they are too simplistic. We demonstrate this with an extremely simple new negotiation strategy called MiCRO, which does not employ any form of opponent modeling or machine learning, but nevertheless outperforms the strongest participants of ANAC 2012, 2013, 2018 and 2019. Furthermore, we provide a theoretical analysis which explains why MiCRO performs so well in the ANAC domains. This analysis may help researchers to design more challenging negotiation domains in the future.

----

## [32] Approval with Runoff

**Authors**: *Théo Delemazure, Jérôme Lang, Jean-François Laslier, M. Remzi Sanver*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/33](https://doi.org/10.24963/ijcai.2022/33)

**Abstract**:

We define a family of runoff rules that work as follows: voters cast approval ballots over candidates; two finalists are selected; and the winner is decided by majority. With approval-type ballots, there are various ways to select the finalists. We leverage known approval-based committee rules and study the obtained runoff rules from an axiomatic point of view. Then we analyze the outcome of these rules on single-peaked profiles, and on real data.

----

## [33] The Complexity of Envy-Free Graph Cutting

**Authors**: *Argyrios Deligkas, Eduard Eiben, Robert Ganian, Thekla Hamm, Sebastian Ordyniak*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/34](https://doi.org/10.24963/ijcai.2022/34)

**Abstract**:

We consider the problem of fairly dividing a set of heterogeneous divisible resources among agents with different preferences. We focus on the setting where the resources correspond to the edges of a connected graph, every agent must be assigned a connected piece of this graph, and the fairness notion considered is the classical envy freeness. The problem is NP-complete, and we analyze its complexity with respect to two natural complexity measures: the number of agents and the number of edges in the graph. While the problem remains NP-hard even for instances with 2 agents, we provide a dichotomy characterizing the complexity of the problem when the number of agents is constant based on structural properties of the graph. For the latter case, we design a polynomial-time algorithm when the graph has a constant number of edges.

----

## [34] Parameterized Complexity of Hotelling-Downs with Party Nominees

**Authors**: *Argyrios Deligkas, Eduard Eiben, Tiger-Lily Goldsmith*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/35](https://doi.org/10.24963/ijcai.2022/35)

**Abstract**:

We study a generalization of the Hotelling-Downs model through the lens of parameterized complexity. In this model, there is a set of voters on a line and a set of parties that compete over them. Each party has to choose a nominee from a set of candidates with predetermined positions on the line, where each candidate comes at a different cost. The goal of every party is to choose the most profitable nominee, given the nominees chosen by the rest of the parties; the profit of a party is the number of voters closer to their nominee minus its cost. We examine the complexity of deciding whether a pure Nash equilibrium exists for this model under several natural parameters: the number of different positions of the candidates, the discrepancy and the span of the nominees, and the overlap of the parties. We provide FPT and XP algorithms and we complement them with a W[1]-hardness result.

----

## [35] Online Approval Committee Elections

**Authors**: *Virginie Do, Matthieu Hervouin, Jérôme Lang, Piotr Skowron*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/36](https://doi.org/10.24963/ijcai.2022/36)

**Abstract**:

Assume k candidates need to be selected. The candidates appear over time. Each time one appears, it must be immediately selected or rejected---a decision that is made by a group of individuals through voting. Assume the voters use approval ballots, i.e., for each candidate they only specify whether they consider it acceptable or not. This setting can be seen as a voting variant of choosing k secretaries. Our contribution is twofold. (1) We assess to what extent the committees that are computed online can proportionally represent the voters. (2) If a prior probability over candidate approvals is available, we show how to compute committees with maximal expected score.

----

## [36] On the Ordinal Invariance of Power Indices on Coalitional Games

**Authors**: *Jean-Paul Doignon, Stefano Moretti, Meltem Öztürk*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/37](https://doi.org/10.24963/ijcai.2022/37)

**Abstract**:

In a coalitional game, the coalitions are weakly ordered according to their worths in the game.  When moreover a power index is given, the players are ranked according to the real numbers they are assigned by the power index.  If any game inducing the same ordering of the coalitions generates the same ranking of the players then, by definition, the game is (ordinally) stable for the power index, which in turn is ordinally invariant for the game.  If one is interested in ranking players of a game which is stable, re-computing the power indices when the coalitional worths slightly fluctuate or are uncertain becomes useless.  Bivalued games are easy examples of games stable for any power index which is linear.  Among general games, we characterize those that are stable for a given linear index.  Note that the Shapley and Banzhaf scores, frequently used in AI, are particular semivalues, and all semivalues are linear indices.  To check whether a game is stable for a specific semivalue, it suffices to inspect the ordering of the coalitions and to perform some direct computation based on the semivalue parameters.

----

## [37] Invasion Dynamics in the Biased Voter Process

**Authors**: *Loke Durocher, Panagiotis Karras, Andreas Pavlogiannis, Josef Tkadlec*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/38](https://doi.org/10.24963/ijcai.2022/38)

**Abstract**:

The voter process is a classic stochastic process that models the invasion of a mutant trait A (e.g., a new opinion, belief, legend, genetic mutation, magnetic spin) in a population of agents (e.g., people, genes, particles) who share a resident trait B, spread over the nodes of a graph. An agent may adopt the trait of one of its neighbors at any time, while the invasion bias r quantifies the stochastic preference towards (r>1) or against (r<1) adopting A over B. Success is measured in terms of the fixation probability, i.e., the probability that eventually all agents have adopted the mutant trait A. In this paper we study the problem of fixation probability maximization under this model: given a budget k, find a set of k agents to initiate the invasion that maximizes the fixation probability. We show that the problem is NP-hard for both regimes r>1 and r<1, while the latter case is also inapproximable within any multiplicative factor that is independent of r. On the positive side, we show that when r>1, the optimization function is submodular and thus can be greedily approximated within a factor 1-1/e. An experimental evaluation of some proposed heuristics corroborates our results.

----

## [38] Efficient Resource Allocation with Secretive Agents

**Authors**: *Soroush Ebadian, Rupert Freeman, Nisarg Shah*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/39](https://doi.org/10.24963/ijcai.2022/39)

**Abstract**:

We consider the allocation of homogeneous divisible goods to agents with linear additive valuations. Our focus is on the case where some agents are secretive and reveal no preference information, while the remaining agents reveal full preference information. We study distortion, which is the worst-case approximation ratio when maximizing social welfare given such partial information about agent preferences. As a function of the number of secretive agents k relative to the overall number of agents n, we identify the exact distortion for every p-mean welfare function, which includes the utilitarian welfare (p=1), the Nash welfare (p -> 0), and the egalitarian welfare (p -> -Inf).

----

## [39] Contests to Incentivize a Target Group

**Authors**: *Edith Elkind, Abheek Ghosh, Paul W. Goldberg*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/40](https://doi.org/10.24963/ijcai.2022/40)

**Abstract**:

We study how to incentivize agents in a target subpopulation to produce a higher output by means of rank-order allocation contests, in the context of incomplete information. We describe a symmetric Bayes--Nash equilibrium for contests that have two types of rank-based prizes: (1) prizes that are accessible only to the agents in the target group; (2) prizes that are accessible to everyone. We also specialize this equilibrium characterization to two important sub-cases: (i) contests that do not discriminate while awarding the prizes, i.e., only have prizes that are accessible to everyone; (ii) contests that have prize quotas for the groups, and each group can compete only for prizes in their share. For these models, we also study the properties of the contest that maximizes the expected total output by the agents in the target group.

----

## [40] Representation Matters: Characterisation and Impossibility Results for Interval Aggregation

**Authors**: *Ulle Endriss, Arianna Novaro, Zoi Terzopoulou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/41](https://doi.org/10.24963/ijcai.2022/41)

**Abstract**:

In the context of aggregating intervals reflecting the views of several agents into a single interval, we investigate the impact of the form of representation chosen for the intervals involved. Specifically, we ask whether there are natural rules we can define both as rules that aggregate separately the left and right endpoints of intervals and as rules that aggregate separately the left endpoints and the interval widths. We show that on discrete scales it is essentially impossible to do so, while on continuous scales we can characterise the rules meeting these requirements as those that compute a weighted average of the endpoints of the individual intervals.

----

## [41] Insight into Voting Problem Complexity Using Randomized Classes

**Authors**: *Zack Fitzsimmons, Edith Hemaspaandra*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/42](https://doi.org/10.24963/ijcai.2022/42)

**Abstract**:

The first step in classifying the complexity of an NP problem is typically showing the problem in P or NP-complete. This has been a successful first step for many problems, including voting problems. However, in this paper we show that this may not always be the best first step. We consider the problem of constructive control by replacing voters (CCRV) introduced by Loreggia et al. [2015, https://dl.acm.org/doi/10.5555/2772879.2773411] for the scoring rule First-Last, which is defined by (1, 0, ..., 0, -1). We show that this problem is equivalent to Exact Perfect Bipartite Matching, and so CCRV for First-Last can be determined in random polynomial time. So on the one hand, if CCRV for First-Last is NP-complete then RP = NP, which is extremely unlikely. On the other hand, showing that CCRV for First-Last is in P would also show that Exact Perfect Bipartite Matching is in P, which would solve a well-studied 40-year-old open problem.

Considering RP as an option for classifying problems can also help classify problems that until now had escaped classification. For example, the sole open problem in the comprehensive table from ErdÃ©lyi et al. [2021, https://doi.org/10.1007/s10458-021-09523-9] is CCRV for 2-Approval. We show that this problem is in RP, and thus easy since it is widely assumed that P = RP.

----

## [42] Approximate Strategyproof Mechanisms for the Additively Separable Group Activity Selection Problem

**Authors**: *Michele Flammini, Giovanna Varricchio*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/43](https://doi.org/10.24963/ijcai.2022/43)

**Abstract**:

We investigate strategyproof mechanisms in the Group Activity Selection Problem with the additively separable property. Namely, agents have distinct preferences for each activity and individual weights for the other agents. We evaluate our mechanisms in terms of their approximation ratio with respect to the maximum utilitarian social welfare.
We first show that, for arbitrary non-negative preferences, no deterministic mechanism can achieve a bounded approximation ratio. Thus, we provide a randomized k-approximate mechanism, where k is the number of activities, and a corresponding 2-2/(k+1) lower bound. Furthermore, we propose a tight (2 - 1/k)-approximate randomized mechanism when activities are copyable.
We then turn our attention to instances where preferences can only be unitary, that is 0 or 1. In this case, we provide a k-approximate deterministic mechanism, which we show to be the best possible one within the class of strategyproof and anonymous mechanisms. We also provide a general lower bound of  Î©({\sqrt{k}) when anonymity is no longer a constraint.  Finally, we focus on unitary preferences and weights, and prove that, while any mechanism returning the optimum is not strategyproof,  there exists a 2-approximate deterministic mechanism.

----

## [43] Picking the Right Winner: Why Tie-Breaking in Crowdsourcing Contests Matters

**Authors**: *Coral Haggiag, Sigal Oren, Ella Segev*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/44](https://doi.org/10.24963/ijcai.2022/44)

**Abstract**:

We present a complete information game-theoretic model for crowdsourcing contests. We observe that in design contests, coding contests and other domains, separating low quality submissions from high quality ones is often easy. However, pinning down the best submission is more challenging since there is no objective measure. We model this situation by assuming that each contestant has an ability, which we interpret as its probability of submitting a high-quality submission. After the contestants decide whether or not they want to participate, the organizer of the contest needs to break ties between the high quality submissions. A common assumption in the literature is that the exact tie-breaking rule does not matter as long as ties are broken consistently. However, we show that the choice of the tie-breaking rule may have significant implications on the efficiency of the contest.

Our results highlight both qualitative and quantitative differences between various deterministic tie-breaking rules. Perhaps counterintuitively, we show that in many scenarios, the utility of the organizer is maximized when ties are broken in favor of successful players with lower ability. Nevertheless, we show that the natural rule of choosing the submission of the successful player with the highest ability guarantees the organizer at least 1/3 of its utility under any tie-breaking rule. To complement these results, we provide an upper bound of  Hn ~ \ln(n) on the price of anarchy (the ratio between the social welfare of the optimal solution and the social welfare of the Nash equilibrium). We show that this ratio is tight when ties are broken in favor of players with higher abilities.

----

## [44] Can Buyers Reveal for a Better Deal?

**Authors**: *Daniel Halpern, Gregory Kehne, Jamie Tucker-Foltz*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/45](https://doi.org/10.24963/ijcai.2022/45)

**Abstract**:

We study market interactions in which buyers are allowed to credibly reveal partial information about their types to the seller. Previous recent work has studied the special case of one buyer and one good, showing that such communication can simultaneously improve social welfare and ex ante buyer utility. However, with multiple buyers, we find that the buyer-optimal signalling schemes from the one-buyer case are actually harmful to buyer welfare. Moreover, we prove several impossibility results showing that, with either multiple i.i.d. buyers or multiple i.i.d. goods, maximizing buyer utility can be at odds with social efficiency, which is surprising in contrast with the one-buyer, one-good case. Finally, we investigate the computational tractability of implementing desirable equilibrium outcomes. We find that, even with one buyer and one good, optimizing buyer utility is generally NP-hard but tractable in a practical restricted setting.

----

## [45] Two for One & One for All: Two-Sided Manipulation in Matching Markets

**Authors**: *Hadi Hosseini, Fatima Umar, Rohit Vaish*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/46](https://doi.org/10.24963/ijcai.2022/46)

**Abstract**:

Strategic behavior in two-sided matching markets has been traditionally studied in a "one-sided" manipulation setting where the agent who misreports is also the intended beneficiary. Our work investigates "two-sided" manipulation of the deferred acceptance algorithm where the misreporting agent and the manipulator (or beneficiary) are on different sides. Specifically, we generalize the recently proposed accomplice manipulation model (where a man misreports on behalf of a woman) along two complementary dimensions: (a) the two for one model, with a pair of misreporting agents (man and woman) and a single beneficiary (the misreporting woman), and (b) the one for all model, with one misreporting agent (man) and a coalition of beneficiaries (all women). Our main contribution is to develop polynomial-time algorithms for finding an optimal manipulation in both settings. We obtain these results despite the fact that an optimal one for all strategy fails to be inconspicuous, while it is unclear whether an optimal two for one strategy satisfies the inconspicuousness property. We also study the conditions under which stability of the resulting matching is preserved. Experimentally, we show that two-sided manipulations are more frequently available and offer better quality matches than their one-sided counterparts.

----

## [46] Phragmén Rules for Degressive and Regressive Proportionality

**Authors**: *Michal Jaworski, Piotr Skowron*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/47](https://doi.org/10.24963/ijcai.2022/47)

**Abstract**:

We study two concepts of proportionality in the model of approval-based committee elections. In degressive proportionality small minorities of voters are favored in comparison with the standard linear proportionality. Regressive proportionality, on the other hand, requires that larger subdivisions of voters are privileged. We introduce a new family of rules that broadly generalize Phragmén's Sequential Rule spanning the spectrum between degressive and regressive proportionality. We analyze and compare the two principles of proportionality assuming the voters and the candidates can be represented as points in an Euclidean issue space.

----

## [47] Forgiving Debt in Financial Network Games

**Authors**: *Panagiotis Kanellopoulos, Maria Kyropoulou, Hao Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/48](https://doi.org/10.24963/ijcai.2022/48)

**Abstract**:

We consider financial networks, where nodes correspond to banks and directed labeled edges correspond to debt contracts between banks. Maximizing systemic liquidity, i.e., the total money flow, is a natural objective of any financial authority. In particular, the financial authority may offer bailout money to some bank(s) or forgive the debts of others in order  to maximize liquidity, and we examine efficient ways to achieve this. We study the computational hardness of finding the optimal debt-removal and budget-constrained optimal bailout policy, respectively, and we investigate the approximation ratio provided by the greedy bailout policy compared to the optimal one. 
 
We also study financial systems from a game-theoretic standpoint. We observe that the removal of some incoming debt might be in the best interest of a bank. Assuming that a bank's well-being (i.e., utility) is aligned with the incoming payments they receive from the network, we define and analyze a game among banks who want to maximize their utility by strategically giving up some incoming payments. In addition, we extend the previous game by considering bailout payments. After formally defining the above games, we prove results about the existence and quality of pure Nash equilibria, as well as the computational complexity of finding such equilibria.

----

## [48] On Discrete Truthful Heterogeneous Two-Facility Location

**Authors**: *Panagiotis Kanellopoulos, Alexandros A. Voudouris, Rongsen Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/49](https://doi.org/10.24963/ijcai.2022/49)

**Abstract**:

We revisit the discrete heterogeneous two-facility location problem, in which there is a set of agents that occupy nodes of a line graph, and have private approval preferences over two facilities. When the facilities are located at some nodes of the line, each agent derives a cost that is equal to her total distance from the facilities she approves. The goal is to decide where to locate the two facilities, so as to (a) incentivize the agents to truthfully report their preferences, and (b) achieve a good approximation of the minimum total (social) cost or the maximum cost among all agents. For both objectives, we design deterministic strategyproof mechanisms with approximation ratios that significantly outperform the state-of-the-art, and complement these results with (almost) tight lower bounds.

----

## [49] Plurality Veto: A Simple Voting Rule Achieving Optimal Metric Distortion

**Authors**: *Fatih Erdem Kizilkaya, David Kempe*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/50](https://doi.org/10.24963/ijcai.2022/50)

**Abstract**:

The metric distortion framework posits that n voters and m candidates are jointly embedded in a metric space such that voters rank candidates that are closer to them higher.
A voting rule's purpose is to pick a candidate with minimum total distance to the voters, given only the rankings, but not the actual distances.
As a result, in the worst case, each deterministic rule picks a candidate whose total distance is at least three times larger than that of an optimal one, i.e., has distortion at least 3.
A recent breakthrough result showed that achieving this bound of 3 is possible;
however, the proof is non-constructive, and the voting rule itself is a complicated exhaustive search.

Our main result is an extremely simple voting rule, called Plurality Veto, which achieves the same optimal distortion of 3. 
Each candidate starts with a score equal to his number of first-place votes.
These scores are then gradually decreased via an n-round veto process in which a candidate drops out when his score reaches zero. One after the other, voters decrement the score of their bottom choice among the standing candidates, and the last standing candidate wins.
We give a one-paragraph proof that this voting rule achieves distortion 3.
This rule is also immensely practical, and it only makes two queries to each voter, so it has low communication overhead.
We also show that a straightforward extension can be used to give a constructive proof of the more general Ranking-Matching Lemma of Gkatzelis et al.

We also generalize Plurality Veto into a class of randomized voting rules in the following way: Plurality veto is run only for k < n rounds; then, a candidate is chosen with probability proportional to his residual score.
This general rule interpolates between Random Dictatorship (for k=0) and Plurality Veto (for k=n-1), and k controls the variance of the output.
We show that for all k, this rule has expected distortion at most 3.

----

## [50] The Dichotomous Affiliate Stable Matching Problem: Approval-Based Matching with Applicant-Employer Relations

**Authors**: *Marina Knittel, Samuel Dooley, John P. Dickerson*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/51](https://doi.org/10.24963/ijcai.2022/51)

**Abstract**:

While the stable marriage problem and its variants model a vast range of matching markets, they fail to capture complex agent relationships, such as the affiliation of applicants and employers in an interview marketplace. To model this problem, the existing literature on matching with externalities permits agents to provide complete and total rankings over matchings based off of both their own and their affiliates' matches. This complete ordering restriction is unrealistic, and further the model may have an empty core. To address this, we introduce the Dichotomous Affiliate Stable Matching (DASM) Problem, where agents' preferences indicate dichotomous acceptance or rejection of another agent in the marketplace, both for themselves and their affiliates. We also assume the agent's preferences over entire matchings are determined by a general weighted valuation function of their (and their affiliates') matches. Our results are threefold: (1) we use a human study to show that real-world matching rankings follow our assumed valuation function; (2) we prove that there always exists a stable solution by providing an efficient, easily-implementable algorithm that finds such a solution; and (3) we experimentally validate the efficiency of our algorithm versus a linear-programming-based approach.

----

## [51] Light Agents Searching for Hot Information

**Authors**: *Dariusz R. Kowalski, Dominik Pajak*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/52](https://doi.org/10.24963/ijcai.2022/52)

**Abstract**:

Agent-based crawlers are commonly used in network maintenance and information gathering. In order not to disturb the main functionality of the system, whether acting at nodes or being in transit, they need to operate online, perform a single operation fast and use small memory. They should also be preferably deterministic, as crawling agents have limited capabilities of generating a large number of truly random bits. We consider a system in which an agent receives an update, typically an insertion or deletion, of some information upon visiting a node. On request, the agent needs to output hot information, i.e., with the net occurrence above certain frequency threshold. A desired time and memory complexity of such agent should be poly-logarithmic in the number of visited nodes and inversely proportional to the frequency threshold. Ours is the first such agent with rigorous analysis and a complementary almost-matching lower bound.

----

## [52] Explaining Preferences by Multiple Patterns in Voters' Behavior

**Authors**: *Sonja Kraiczy, Edith Elkind*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/53](https://doi.org/10.24963/ijcai.2022/53)

**Abstract**:

In some preference aggregation scenarios, voters' preferences are highly structured: e.g., the set of candidates may have one-dimensional structure (so that voters' preferences are single-peaked) or be described by a binary decision tree (so that voters' preferences are group-separable). However, sometimes a single axis or a decision tree is insufficient to capture the voters' preferences; rather, there is a small number K of axes or decision trees such that each vote in the profile is consistent with one of these axes (resp., trees). In this work, we study the complexity of deciding whether voters' preferences can be explained in this manner. For K=2, we use the technique developed by Yang [2020, https://doi.org/10.3233/FAIA200099] in the context of single-peaked preferences to obtain a polynomial-time algorithm for several domains: value-restricted preferences, group-separable preferences, and a natural subdomain of group-separable preferences, namely, caterpillar group-separable preferences. For K > 2, the problem is known to be hard for single-peaked preferences; we establish that it is also hard for value-restricted and group-separable preferences. Our positive results for K=2 make use of forbidden minor characterizations of the respective domains; in particular, we establish that the domain of caterpillar group-separable preferences admits a forbidden minor characterization.

----

## [53] Biased Majority Opinion Dynamics: Exploiting Graph k-domination

**Authors**: *Hicham Lesfari, Frédéric Giroire, Stéphane Pérennes*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/54](https://doi.org/10.24963/ijcai.2022/54)

**Abstract**:

We study opinion dynamics in multi-agent networks where agents hold binary opinions and are influenced by their neighbors while being biased towards one of the two opinions, called the superior opinion. The dynamics is modeled by the following process: at each round, a randomly selected agent chooses the superior opinion with some probability α, and with probability 1-α it conforms to the opinion manifested by the majority of its neighbors. In this work, we exhibit classes of network topologies for which we prove that the expected time for consensus on the superior opinion can be exponential. This answers an open conjecture in the literature. In contrast, we show that in all cubic graphs, convergence occurs after a polynomial number of rounds for every α.
We rely on new structural graph properties by characterizing the opinion formation in terms of multiple domination, stable and decreasing structures in graphs, providing an interplay between bias, consensus and network structure. Finally, we provide both theoretical and experimental evidence for the existence of decreasing structures and relate it to the rich behavior observed on the expected convergence time of the opinion diffusion model.

----

## [54] Modelling the Dynamics of Multi-Agent Q-learning: The Stochastic Effects of Local Interaction and Incomplete Information

**Authors**: *Chin-wing Leung, Shuyue Hu, Ho-fung Leung*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/55](https://doi.org/10.24963/ijcai.2022/55)

**Abstract**:

The theoretical underpinnings of multiagent reinforcement learning has recently attracted much attention.
In this work, we focus on the generalized social learning (GSL) protocol --- an agent interaction protocol that is widely adopted in the literature, and aim to develop an accurate theoretical model for the Q-learning dynamics under this protocol.
Noting that previous models fail to characterize the effects of local interactions and incomplete information that arise from GSL, we model the Q-values dynamics of each individual agent as a system of stochastic differential equations (SDE). 
Based on the SDE, we express the time evolution of the probability density function of Q-values in the population with a Fokker-Planck equation.
We validate the correctness of our model through extensive comparisons with agent-based simulation results across different types of symmetric games. 
In addition, we show that as the interactions between agents are more limited and information is less complete, the population can converge to a outcome that is qualitatively different than that with global interactions and complete information.

----

## [55] Propositional Gossip Protocols under Fair Schedulers

**Authors**: *Joseph Livesey, Dominik Wojtczak*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/56](https://doi.org/10.24963/ijcai.2022/56)

**Abstract**:

Gossip protocols are programs that can be used by a group of agents to synchronize what information they have. Namely, assuming each agent holds a secret, the goal of a protocol is to reach a situation in which all agents know all secrets. Distributed epistemic gossip protocols use epistemic formulas in the component programs for the agents. In this paper, we study the simplest classes of such gossip protocols: propositional gossip protocols, in which whether an agent wants to initiate a call depends only on the set of secrets that the agent currently knows. It was recently shown that such a protocol can be correct, i.e., always terminates in a state where all agents know all secrets, only when its communication graph is complete. We show here that this characterization dramatically changes when the usual fairness constraints are imposed on the call scheduler used. Finally, we establish that checking the correctness of a given propositional protocol under a fair scheduler is a coNP-complete problem.

----

## [56] Proportional Budget Allocations: Towards a Systematization

**Authors**: *Maaike Los, Zoé Christoff, Davide Grossi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/57](https://doi.org/10.24963/ijcai.2022/57)

**Abstract**:

We contribute to the programme of lifting proportionality axioms from the multi-winner voting setting to participatory budgeting. We define novel proportionality axioms for participatory budgeting and test them on known proportionality-driven rules such as Phragmén and Rule X. We investigate logical implications among old and new axioms and provide a systematic overview of proportionality criteria in participatory budgeting.

----

## [57] Parameterized Algorithms for Kidney Exchange

**Authors**: *Arnab Maiti, Palash Dey*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/58](https://doi.org/10.24963/ijcai.2022/58)

**Abstract**:

In kidney exchange programs, multiple patient-donor pairs each of whom are otherwise incompatible, exchange their donors to receive compatible kidneys. The Kidney Exchange problem is typically modelled as a directed graph where every vertex is either an altruistic donor or a pair of patient and donor; directed edges are added from a donor to its compatible patients. The computational task is to find if there exists a collection of disjoint cycles and paths starting from altruistic donor vertices of length at most l_c and l_p respectively that covers at least some specific number t of non-altruistic vertices (patients). We study parameterized algorithms for the kidney exchange problem in this paper. Specifically, we design FPT algorithms parameterized by each of the following parameters: (1) the number of patients who receive kidney, (2) treewidth of the input graph + max{l_p, l_c}, and (3) the number of vertex types in the input graph when l_p <= l_c. We also present interesting algorithmic and hardness results on the kernelization complexity of the problem. Finally, we present an approximation algorithm for an important special case of Kidney Exchange.

----

## [58] Fixing Knockout Tournaments With Seeds

**Authors**: *Pasin Manurangsi, Warut Suksompong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/59](https://doi.org/10.24963/ijcai.2022/59)

**Abstract**:

Knockout tournaments constitute a popular format for organizing sports competitions. While prior results have shown that it is often possible to manipulate a knockout tournament by fixing the bracket, these results ignore the prevalent aspect of player seeds, which can significantly constrain the chosen bracket. We show that certain structural conditions that guarantee that a player can win a knockout tournament without seeds are no longer sufficient in light of seed constraints. On the other hand, we prove that when the pairwise match outcomes are generated randomly, all players are still likely to be knockout winners under the same probability threshold with seeds as without seeds. In addition, we investigate the complexity of deciding whether a manipulation is possible when seeds are present.

----

## [59] Group Wisdom at a Price: Jury Theorems with Costly Information

**Authors**: *Matteo Michelini, Adrian Haret, Davide Grossi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/60](https://doi.org/10.24963/ijcai.2022/60)

**Abstract**:

We study epistemic voting on binary issues where voters are characterized by their competence, i.e., the probability of voting for the correct alternative, and can choose between two actions: voting or abstaining. In our setting voting involves the expenditure of some effort, which is required to achieve the appropriate level of competence, whereas abstention carries no effort. We model this scenario as a game and characterize its equilibria under several variations. Our results show that when agents are aware of everyone's incentives, then the addition of effort may lead to Nash equilibria where wisdom of the crowds is lost. We further show that if agents' awareness of each other is constrained by a social network, the topology of the network may actually mitigate this effect.

----

## [60] Automated Synthesis of Mechanisms

**Authors**: *Munyque Mittelmann, Bastien Maubert, Aniello Murano, Laurent Perrussel*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/61](https://doi.org/10.24963/ijcai.2022/61)

**Abstract**:

Mechanism Design aims to design a game so that a desirable outcome is reached regardless of agents' self-interests. In this paper, we show how this problem can be rephrased as a synthesis problem, where mechanisms are automatically synthesized from a partial or complete specification in a high-level logical language. We show that Quantitative Strategy Logic is a perfect candidate for specifying mechanisms as it can express complex strategic and quantitative properties. We solve automated mechanism design in two cases: when the number of actions is bounded, and when agents play in turn.

----

## [61] Robust Solutions for Multi-Defender Stackelberg Security Games

**Authors**: *Dolev Mutzari, Yonatan Aumann, Sarit Kraus*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/62](https://doi.org/10.24963/ijcai.2022/62)

**Abstract**:

Multi-defender Stackelberg Security Games (MSSG) have recently gained increasing attention in the literature. 
However, the solutions offered to date are highly sensitive, wherein even small perturbations in the attacker's utility or slight uncertainties thereof can dramatically change the defenders' resulting payoffs and alter the equilibrium. 
In this paper, we introduce a robust model for MSSGs, which admits solutions that are resistant to small perturbations or uncertainties in the game's parameters. 
First, we formally define the notion of robustness, as well as the robust MSSG model. Then, for the non-cooperative setting, we prove the existence of a robust approximate equilibrium in any such game, and provide an efficient construction thereof. For the cooperative setting, we show that any such game admits a robust approximate (alpha) core, and provide an efficient construction thereof. 
Lastly, we show that stronger types of the core may be empty.
Interestingly, the robust solutions can substantially increase the defenders' utilities over those of the non-robust ones.

----

## [62] I Will Have Order! Optimizing Orders for Fair Reviewer Assignment

**Authors**: *Justin Payan, Yair Zick*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/63](https://doi.org/10.24963/ijcai.2022/63)

**Abstract**:

We study mechanisms that allocate reviewers to papers in a fair and efficient manner. We model reviewer assignment as an instance of a fair allocation problem, presenting an extension of the classic round-robin mechanism, called Reviewer Round Robin (RRR). Round-robin mechanisms are a standard tool to ensure envy-free up to one item (EF1) allocations. However, fairness often comes at the cost of decreased efficiency. To overcome this challenge, we carefully select an approximately optimal round-robin order. Applying a relaxation of submodularity, γ-weak submodularity, we show that greedily inserting papers into an order yields a (1+γ²)-approximation to the maximum welfare attainable by our round-robin mechanism under any order. Our Greedy Reviewer Round Robin (GRRR) approach outputs highly efficient EF1 allocations for three real conference datasets, offering comparable performance to state-of-the-art paper assignment methods in fairness, efficiency, and runtime, while providing the only EF1 guarantee.

----

## [63] Fair, Individually Rational and Cheap Adjustment

**Authors**: *Gleb Polevoy, Marcin Dziubinski*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/64](https://doi.org/10.24963/ijcai.2022/64)

**Abstract**:

Consider the practical goal of making a desired action profile played,
when the planner can only change the payoffs, bound by 
stringent constraints.
Applications include motivating people
to choose the closest school, the closest subway station, or to coordinate
on a communication protocol or an investment strategy.
Employing subsidies and tolls, we adjust the game so that choosing this predefined action profile
becomes strictly dominant. 
Inspired mainly by the work of Monderer and Tennenholtz,
where the promised subsidies do not materialise in the not played
profiles, we provide a fair and individually rational game
adjustment, such that the total outside investments sum up
to zero at any profile, thereby facilitating easy and frequent
usage of our adjustment without bearing costs, even if some
players behave unexpectedly. The resultant action profile itself needs no
adjustment. Importantly, we also prove that our adjustment minimises 
the general transfer among all such adjustments, counting the total subsidising and taxation.

----

## [64] Exploring the Benefits of Teams in Multiagent Learning

**Authors**: *David Radke, Kate Larson, Tim Brecht*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/65](https://doi.org/10.24963/ijcai.2022/65)

**Abstract**:

For problems requiring cooperation, many multiagent systems implement solutions among either individual agents or across an entire population towards a common goal. Multiagent teams are primarily studied when in conflict; however, organizational psychology (OP) highlights the benefits of teams among human populations for learning how to coordinate and cooperate. In this paper, we propose a new model of multiagent teams for reinforcement learning (RL) agents inspired by OP and early work on teams in artificial intelligence. We validate our model using complex social dilemmas that are popular in recent multiagent RL and find that agents divided into teams develop cooperative pro-social policies despite incentives to not cooperate. Furthermore, agents are better able to coordinate and learn emergent roles within their teams and achieve higher rewards compared to when the interests of all agents are aligned.

----

## [65] The Power of Media Agencies in Ad Auctions: Improving Utility through Coordinated Bidding

**Authors**: *Giulia Romano, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/66](https://doi.org/10.24963/ijcai.2022/66)

**Abstract**:

The increasing competition in digital advertising induced a proliferation of media agencies playing the role of intermediaries between advertisers and platforms selling ad slots. When a group of competing advertisers is managed by a common agency, many forms of collusion, such as bid rigging, can be implemented by coordinating bidding strategies, dramatically increasing advertisers' value. We study the problem of finding bids and monetary transfers maximizing the utility of a group of colluders, under GSP and VCG mechanisms. First, we introduce an abstract bid optimization problem---called weighted utility problem (WUP)---, which is useful in proving our results. We show that the utilities of bidding strategies are related to the length of paths in a directed acyclic weighted graph, whose structure and weights depend on the mechanism under study. This allows us to solve WUP in polynomial time by finding a shortest path of the graph. Next, we switch to our original problem, focusing on two settings that differ for the incentives they allow for. Incentive constraints ensure that colluders do not leave the agency, and they can be enforced by implementing monetary transfers between the agency and the advertisers. In particular, we study the arbitrary transfers setting, where any kind of monetary transfer to and from the advertisers is allowed, and the more realistic limited liability setting, in which no advertiser can be paid by the agency.
 In the former, we cast the problem as a WUP instance and solve it by our graph-based algorithm, while, in the latter, we formulate it as a linear program with exponentially-many variables efficiently solvable by applying the ellipsoid algorithm to its dual. This requires to solve a suitable separation problem in polynomial time, which can be done by reducing it to the weighted utility problem a WUP instance.

----

## [66] Transfer Learning Based Adaptive Automated Negotiating Agent Framework

**Authors**: *Ayan Sengupta, Shinji Nakadai, Yasser Mohammad*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/67](https://doi.org/10.24963/ijcai.2022/67)

**Abstract**:

With the availability of domain specific historical negotiation data,  the practical applications of machine learning techniques can prove to be increasingly effective in the field of automated negotiation. Yet a large portion of the literature focuses on domain independent negotiation and thus passes the possibility of leveraging any domain specific insights from historical data. Moreover, during sequential negotiation, utility functions may alter due to various reasons including market demand, partner agreements, weather conditions, etc. This poses a unique set of challenges and one can easily infer that one strategy that fits all is rather impossible in such scenarios. In this work, we present a simple yet effective method of learning an end-to-end negotiation strategy from historical negotiation data.  Next, we show that transfer learning based solutions are effective in designing adaptive strategies when underlying utility functions of agents change. Additionally, we also propose an online method of detecting and measuring such changes in the utility functions. Combining all three contributions we propose an adaptive automated negotiating agent framework that enables the automatic creation of transfer learning based negotiating agents capable of adapting to changes in utility functions. Finally, we present the results of an agent generated using our framework in different ANAC domains with 100 different utility functions each and show that our agent outperforms the benchmark score by domain independent agents by 6%.

----

## [67] Multiwinner Elections under Minimax Chamberlin-Courant Rule in Euclidean Space

**Authors**: *Chinmay Sonar, Subhash Suri, Jie Xue*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/68](https://doi.org/10.24963/ijcai.2022/68)

**Abstract**:

We consider multiwinner elections in Euclidean space using the minimax Chamberlin-Courant rule.
In this setting, voters and candidates are embedded in a d-dimensional Euclidean space,
and the goal is to choose a committee of k candidates so that the rank of any voter's
most preferred candidate in the committee is minimized. (The problem is also equivalent to the 
ordinal version of the classical k-center problem.) 
We show that the problem is NP-hard in any dimension d >= 2, and also provably hard to approximate.
Our main results are three polynomial-time approximation schemes, each of which finds a committee 
with provably good minimax score. In all cases, we show that our approximation bounds are tight or close to tight.
We mainly focus on the 1-Borda rule but some of our results also hold for the more general r-Borda.

----

## [68] Near-Tight Algorithms for the Chamberlin-Courant and Thiele Voting Rules

**Authors**: *Krzysztof Sornat, Virginia Vassilevska Williams, Yinzhan Xu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/69](https://doi.org/10.24963/ijcai.2022/69)

**Abstract**:

We present an almost optimal algorithm for the classic Chamberlin-Courant multiwinner voting rule (CC) on single-peaked preference profiles. Given n voters and m candidates, it runs in almost linear time in the input size improving the previous best O(nm^2) time algorithm. We also study multiwinner voting rules on nearly single-peaked preference profiles in terms of the candidate-deletion operation. We show a polynomial-time algorithm for CC where a given candidate-deletion set D has logarithmic size. Actually, our algorithm runs in 2^|D| * poly(n,m) time and the base of the power cannot be improved under the Strong Exponential Time Hypothesis. We also adapt these results to all non-constant Thiele rules which generalize CC with approval ballots.

----

## [69] Maxmin Participatory Budgeting

**Authors**: *Gogulapati Sreedurga, Mayank Ratan Bhardwaj, Yadati Narahari*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/70](https://doi.org/10.24963/ijcai.2022/70)

**Abstract**:

Participatory Budgeting (PB) is a popular voting method by which a limited budget is divided among a set of projects, based on the preferences of voters over the projects. PB is broadly categorised as divisible PB (if the projects are fractionally implementable) and indivisible PB (if the projects are atomic). Egalitarianism, an important objective in PB, has not received much attention in the context of indivisible PB. This paper addresses this gap through a detailed study of a natural egalitarian rule, Maxmin Participatory Budgeting (MPB), in the context of indivisible PB. Our study is in two parts: (1) computational (2) axiomatic.  In the first part, we prove that MPB is computationally hard and give pseudo-polynomial time and polynomial-time algorithms when parameterized by certain well-motivated parameters. We propose an algorithm that achieves for MPB, additive approximation guarantees for restricted spaces of instances and empirically show that our algorithm in fact gives exact optimal solutions on real-world PB datasets. We also establish an upper bound on the approximation ratio achievable for MPB by the family of exhaustive strategy-proof PB algorithms. In the second part, we undertake an axiomatic study of the MPB rule by generalizing known axioms in the literature. Our study leads to the proposal of a new axiom, maximal coverage, which captures fairness aspects. We prove that MPB satisfies maximal coverage.

----

## [70] How to Sample Approval Elections?

**Authors**: *Stanislaw Szufa, Piotr Faliszewski, Lukasz Janeczko, Martin Lackner, Arkadii Slinko, Krzysztof Sornat, Nimrod Talmon*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/71](https://doi.org/10.24963/ijcai.2022/71)

**Abstract**:

We extend the map-of-elections framework to the case of approval elections. While doing so, we study a number of statistical cultures, including some new ones, and we analyze their properties. We find that approval elections can be understood in terms of the average number of approvals in the votes, and the extent to which the votes are chaotic.

----

## [71] Search-Based Testing of Reinforcement Learning

**Authors**: *Martin Tappler, Filip Cano Córdoba, Bernhard K. Aichernig, Bettina Könighofer*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/72](https://doi.org/10.24963/ijcai.2022/72)

**Abstract**:

Evaluation of deep reinforcement learning (RL) is inherently challenging. Especially the opaqueness of learned policies and the stochastic nature of both agents and environments make testing the behavior of deep RL agents difficult. We present a search-based testing framework that enables a wide range of novel analysis capabilities for evaluating the safety and performance of deep RL agents. For safety testing, our framework utilizes a search algorithm that searches for a reference trace that solves the RL task. The backtracking states of the search, called boundary states, pose safety-critical situations. We create safety test-suites that evaluate how well the RL agent escapes safety-critical situations near these boundary states. For robust performance testing, we create a diverse set of traces via fuzz testing. These fuzz traces are used to bring the agent into a wide variety of potentially unknown states from which the average performance of the agent is compared to the average performance of the fuzz traces. We apply our search-based testing approach on RL for Nintendo's Super Mario Bros.

----

## [72] Real-Time BDI Agents: A Model and Its Implementation

**Authors**: *Andrea Traldi, Francesco Bruschetti, Marco Robol, Marco Roveri, Paolo Giorgini*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/73](https://doi.org/10.24963/ijcai.2022/73)

**Abstract**:

The BDI model proved to be effective for the developing of applications requiring high-levels of autonomy and to deal with the complexity and unpredictability of real-world scenarios. The model, however, has significant limitations in reacting and handling contingencies within the given real-time constraints. Without an explicit representation of time, existing real-time BDI implementations overlook the temporal implications during the agentâ€™s decision process that may result in delays or unresponsiveness of the system when it gets overloaded. In this paper, we redefine the BDI agent control loop inspired by traditional and well establish algorithms for real-time systems to ensure a proper reaction of agents and their effective application in typical real-time domains. Our model proposes an effective real-time management of goals, plans, and actions with respect to time constraints and resources availability. We propose an implementation of the model for a resource-collection video-game and we validate the approach against a set of significant scenarios.

----

## [73] Max-Sum with Quadtrees for Decentralized Coordination in Continuous Domains

**Authors**: *Dimitrios Troullinos, Georgios Chalkiadakis, Vasilis Samoladas, Markos Papageorgiou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/74](https://doi.org/10.24963/ijcai.2022/74)

**Abstract**:

In this paper we put forward a novel extension of the classic Max-Sum algorithm to the framework of Continuous Distributed Constrained Optimization Problems (Continuous DCOPs), by utilizing a popular geometric algorithm, namely Quadtrees. In its standard form, Max-Sum can only solve Continuous DCOPs with an a priori discretization procedure. Existing Max-Sum extensions to continuous multiagent coordination domains require additional assumptions regarding the form of the factors, such as access to the gradient, or the ability to model them as continuous piecewise linear functions. Our proposed approach has no such requirements: we model the exchanged messages with Quadtrees, and, as such, the discretization procedure is dynamic and embedded in the internal Max-Sum operations (addition and marginal maximization). We apply Max-Sum with Quadtrees to lane-free autonomous driving. Our experimental evaluation showcases the effectiveness of our approach in this challenging coordination domain.

----

## [74] Strategy Proof Mechanisms for Facility Location with Capacity Limits

**Authors**: *Toby Walsh*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/75](https://doi.org/10.24963/ijcai.2022/75)

**Abstract**:

An important feature of many real world facility location problems are capacity limits on the number of agents served by each facility. We provide a comprehensive picture of strategy proof mechanisms for facility location problems with capacity constraints that are anonymous and Pareto optimal. First, we prove a strong characterization theorem. For locating two identical facilities with capacity limits and no spare capacity, the INNERPOINT mechanism is the unique strategy proof mechanism that is both anonymous and Pareto optimal. Second, when there is spare capacity, we identify a more general class of strategy proof mechanisms that interpolates smoothly between INNERPOINT and ENDPOINT which are anonymous and Pareto optimal. Third, with two facilities of different capacities, we prove a strong impossibility theorem that no mechanism can be both anonymous and Pareto optimal except when the capacities differ by just a single agent. Fourth, with three or more facilities we prove a second impossibility theorem that no mechanism can be both anonymous and Pareto optimal even when facilities have equal capacity. Our characterization and impossibility results are all minimal as multiple mechanisms exist if we drop one property.

----

## [75] Modelling the Dynamics of Regret Minimization in Large Agent Populations: a Master Equation Approach

**Authors**: *Zhen Wang, Chunjiang Mu, Shuyue Hu, Chen Chu, Xuelong Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/76](https://doi.org/10.24963/ijcai.2022/76)

**Abstract**:

Understanding the learning dynamics in multiagent systems is an important and challenging task. Past research on multi-agent learning mostly focuses on two-agent settings. In this paper, we consider the scenario in which a population of infinitely many agents apply regret minimization in repeated symmetric games. We propose a new formal model based on the master equation approach in statistical physics to describe the evolutionary dynamics in the agent population. Our model takes the form of a partial differential equation, which describes how the probability distribution of regret evolves over time. Through experiments, we show that our theoretical results are consistent with the agent-based simulation results.

----

## [76] Monotone-Value Neural Networks: Exploiting Preference Monotonicity in Combinatorial Assignment

**Authors**: *Jakob Weissteiner, Jakob Heiss, Julien Siems, Sven Seuken*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/77](https://doi.org/10.24963/ijcai.2022/77)

**Abstract**:

Many important resource allocation problems involve the combinatorial assignment of items, e.g., auctions or course allocation. Because the bundle space grows exponentially in the number of items, preference elicitation is a key challenge in these domains. Recently, researchers have proposed ML-based mechanisms that outperform traditional mechanisms while reducing preference elicitation costs for agents. However, one major shortcoming of the ML algorithms that were used is their disregard of important prior knowledge about agents' preferences. To address this, we introduce monotone-value neural networks (MVNNs), which are designed to capture combinatorial valuations, while enforcing monotonicity and normality. On a technical level, we prove that our MVNNs are universal in the class of monotone and normalized value functions, and we provide a mixed-integer linear program (MILP) formulation to make solving MVNN-based winner determination problems (WDPs) practically feasible. We evaluate our MVNNs experimentally in spectrum auction domains. Our results show that MVNNs improve the prediction performance, they yield state-of-the-art allocative efficiency in the auction, and they also reduce the run-time of the WDPs. Our code is available on GitHub: https://github.com/marketdesignresearch/MVNN.

----

## [77] Fourier Analysis-based Iterative Combinatorial Auctions

**Authors**: *Jakob Weissteiner, Chris Wendler, Sven Seuken, Benjamin Lubin, Markus Püschel*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/78](https://doi.org/10.24963/ijcai.2022/78)

**Abstract**:

Recent advances in Fourier analysis have brought new tools to efficiently represent and learn set functions. In this paper, we bring the power of Fourier analysis to the design of combinatorial auctions (CAs). The key idea is to approximate bidders' value functions using Fourier-sparse set functions, which can be computed using a relatively small number of queries. Since this number is still too large for practical CAs, we propose a new hybrid design: we first use neural networks (NNs) to learn bidders’ values and then apply Fourier analysis to the learned representations. On a technical level, we formulate a Fourier transform-based winner determination problem and derive its mixed integer program formulation. Based on this, we devise an iterative CA that asks Fourier-based queries. We experimentally show that our hybrid ICA achieves higher efficiency than prior auction designs, leads to a fairer distribution of social welfare, and significantly reduces runtime. With this paper, we are the first to leverage Fourier analysis in CA design and lay the foundation for future work in this area. Our code is available on GitHub: https://github.com/marketdesignresearch/FA-based-ICAs.

----

## [78] Manipulating Elections by Changing Voter Perceptions

**Authors**: *Junlin Wu, Andrew Estornell, Lecheng Kong, Yevgeniy Vorobeychik*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/79](https://doi.org/10.24963/ijcai.2022/79)

**Abstract**:

The integrity of elections is central to democratic systems. However, a myriad of malicious actors aspire to influence election outcomes for financial or political benefit. A common means to such ends is by manipulating perceptions of the voting public about select candidates, for example, through misinformation. We present a formal model of the impact of perception manipulation on election outcomes in the framework of spatial voting theory, in which the preferences of voters over candidates are generated based on their relative distance in the space of issues. We show that controlling elections in this model is, in general, NP-hard, whether issues are binary or real-valued. However, we demonstrate that critical to intractability is the diversity of opinions on issues exhibited by the voting public. When voter views lack diversity, and we can instead group them into a small number of categories---for example, as a result of political polarization---the election control problem can be solved in polynomial time in the number of issues and candidates for arbitrary scoring rules.

----

## [79] Fast and Fine-grained Autoscaler for Streaming Jobs with Reinforcement Learning

**Authors**: *Mingzhe Xing, Hangyu Mao, Zhen Xiao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/80](https://doi.org/10.24963/ijcai.2022/80)

**Abstract**:

On computing clusters, the autoscaler is responsible for allocating resources for jobs or fine-grained tasks to ensure their Quality of Service. Due to a more precise resource management, fine-grained autoscaling can generally achieve better performance. However, the fine-grained autoscaling for streaming jobs needs intensive computation to model the complicated running states of tasks, and has not been adequately studied previously. In this paper, we propose a novel fine-grained autoscaler for streaming jobs based on reinforcement learning. We first organize the running states of streaming jobs as spatio-temporal graphs. To efficiently make autoscaling decisions, we propose a Neural Variational Subgraph Sampler to sample spatio-temporal subgraphs. Furthermore, we propose a mutual-information-based objective function to explicitly guide the sampler to extract more representative subgraphs. After that, the autoscaler makes decisions based on the learned subgraph representations. Experiments conducted on real-world datasets demonstrate the superiority of our method over six competitive baselines.

----

## [80] Mechanism Design with Predictions

**Authors**: *Chenyang Xu, Pinyan Lu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/81](https://doi.org/10.24963/ijcai.2022/81)

**Abstract**:

Improving algorithms via predictions is a very active research topic in recent years. This paper initiates the systematic study of mechanism design in this model. In a number of well-studied mechanism design settings, we make use of imperfect predictions to design mechanisms that perform much better than traditional mechanisms if the predictions are accurate (consistency), while always retaining worst-case guarantees even with very imprecise predictions (robustness). Furthermore, we refer to the largest prediction error sufficient to give a good performance as the error tolerance of a mechanism, and observe that an intrinsic tradeoff among consistency, robustness and error tolerance is common for mechanism design with predictions.

----

## [81] Efficient Multi-Agent Communication via Shapley Message Value

**Authors**: *Di Xue, Lei Yuan, Zongzhang Zhang, Yang Yu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/82](https://doi.org/10.24963/ijcai.2022/82)

**Abstract**:

Utilizing messages from teammates is crucial in cooperative multi-agent tasks due to the partially observable nature of the environment. Naively asking messages from all teammates without pruning may confuse individual agents, hindering the learning process and impairing the whole system's performance. Most previous work either utilizes a gate or employs an attention mechanism to extract relatively important messages. However, they do not explicitly evaluate each message's value, failing to learn an efficient communication protocol in more complex scenarios. To tackle this issue, we model the teammates of an agent as a message coalition and calculate the Shapley Message Value (SMV) of each agent within it. SMV reflects the contribution of each message to an agent and redundant messages can be spotted in this way effectively. On top of that, we design a novel framework named Shapley Message Selector (SMS), which learns to predict the SMVs of teammates for an agent solely based on local information so that the agent can only query those teammates with positive SMVs. Empirically, we demonstrate that our method can prune redundant messages and achieve comparable or better performance in various multi-agent cooperative scenarios than full communication settings and existing strong baselines.

----

## [82] On the Complexity of Calculating Approval-Based Winners in Candidates-Embedded Metrics

**Authors**: *Yongjie Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/83](https://doi.org/10.24963/ijcai.2022/83)

**Abstract**:

We study approval-based multiwinner voting where candidates are in a metric space and committees are valuated in terms of their distances to the given votes. In particular, we consider three different distance functions, and for each of them we study both the utilitarian rules and the egalitarian rules, resulting in six variants of winners determination problems. We focus on the (parameterized) complexity of these problems for both the general metric and several special metrics. For hardness results, we also discuss their approximability.

----

## [83] Environment Design for Biased Decision Makers

**Authors**: *Guanghui Yu, Chien-Ju Ho*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/84](https://doi.org/10.24963/ijcai.2022/84)

**Abstract**:

We study the environment design problem for biased decision makers. In an environment design problem, an informed principal aims to update the decision making environment to influence the decisions made by the agent. This problem is ubiquitous in various domains, e.g., a social networking platform might want to update its website to encourage more user engagement. In this work, we focus on the scenario in which the agent might exhibit biases in decision making. We relax the common assumption that the agent is rational and aim to incorporate models of biased agents in environment design. We formulate the environment design problem under the Markov decision process (MDP) and incorporate common models of biased agents through introducing general time-discounting functions. We then formalize the environment design problem as constrained optimization problems and propose corresponding algorithms. We conduct both simulations and real human-subject experiments with workers recruited from Amazon Mechanical Turk to evaluate our proposed algorithms.

----

## [84] Multi-Agent Concentrative Coordination with Decentralized Task Representation

**Authors**: *Lei Yuan, Chenghe Wang, Jianhao Wang, Fuxiang Zhang, Feng Chen, Cong Guan, Zongzhang Zhang, Chongjie Zhang, Yang Yu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/85](https://doi.org/10.24963/ijcai.2022/85)

**Abstract**:

Value-based multi-agent reinforcement learning (MARL) methods hold the promise of promoting coordination in cooperative settings. Popular MARL methods mainly focus on the scalability or the representational capacity of value functions. Such a learning paradigm can reduce agents' uncertainties and promote coordination. However, they fail to leverage the task structure decomposability, which generally exists in real-world multi-agent systems (MASs), leading to a significant amount of time exploring the optimal policy in complex scenarios. To address this limitation, we propose a novel framework Multi-Agent Concentrative Coordination (MACC) based on task decomposition, with which an agent can implicitly form local groups to reduce the learning space to facilitate coordination. In MACC, agents first learn representations for subtasks from their local information and then implement an attention mechanism to concentrate on the most relevant ones. Thus, agents can pay targeted attention to specific subtasks and improve coordination. Extensive experiments on various complex multi-agent benchmarks demonstrate that MACC achieves remarkable performance compared to existing methods.

----

## [85] Correlation-Based Algorithm for Team-Maxmin Equilibrium in Multiplayer Extensive-Form Games

**Authors**: *Youzhi Zhang, Bo An, V. S. Subrahmanian*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/86](https://doi.org/10.24963/ijcai.2022/86)

**Abstract**:

Efficient algorithms computing a Nash equilibrium have been successfully applied to large zero- sum two-player extensive-form games (e.g., poker). However, in multiplayer games, computing a Nash equilibrium is generally hard, and the equilibria are not exchangeable, which makes players face the problem of selecting one of many different Nash equilibria. In this paper, we focus on an alternative solution concept in zero-sum multiplayer extensive-form games called Team-Maxmin Equilibrium (TME). It is a Nash equilibrium that maximizes each team memberâ€™s utility. As TME is unique in general, it avoids the equilibrium selection problem. However, it is still difficult (FNP- hard) to find a TME. Computing it can be formulated as a non-convex program, but existing algorithms are capable of solving this program for only very small games. In this paper, we first refine the complexity result for computing a TME by using a correlation plan to show that a TME can be found in polynomial time in a specific class of games according to our boundary for complexity. Second, we propose an efficient correlation-based algorithm to solve the non-convex program for TME in games not belonging to this class. The algorithm combines two special correlation plans based on McCormick envelopes for convex relaxation and von Stengel-Forges polytope for correlated equilibria. We show that restricting the feasible solution space to von Stengel-Forges polytope will strictly reduce the feasible solution space after convex re- laxation of nonlinear terms. Finally, experiments show that our algorithm is about four orders of magnitude faster than the prior state of the art and can solve many previously unsolvable games.

----

## [86] Strategyproof Mechanisms for Group-Fair Facility Location Problems

**Authors**: *Houyu Zhou, Minming Li, Hau Chan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/87](https://doi.org/10.24963/ijcai.2022/87)

**Abstract**:

We study the facility location problems where agents are located on a real line and divided into groups based on criteria such as ethnicity or age. Our aim is to design mechanisms to locate a facility to approximately minimize the costs of groups of agents to the facility fairly while eliciting the agents' locations truthfully. We first explore various well-motivated group fairness cost objectives for the problems and show that many natural objectives have an unbounded approximation ratio. We then consider minimizing the maximum total group cost and minimizing the average group cost objectives. For these objectives, we show that existing classical mechanisms (e.g., median) and new group-based mechanisms provide bounded approximation ratios, where the group-based mechanisms can achieve better ratios. We also provide lower bounds for both objectives. To measure fairness between groups and within each group, we study a new notion of intergroup and intragroup fairness (IIF) . We consider two IIF objectives and provide mechanisms with tight approximation ratios.

----

## [87] Evolutionary Approach to Security Games with Signaling

**Authors**: *Adam Zychowski, Jacek Mandziuk, Elizabeth Bondi, Aravind Venugopal, Milind Tambe, Balaraman Ravindran*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/88](https://doi.org/10.24963/ijcai.2022/88)

**Abstract**:

Green Security Games have become a popular way to model scenarios involving the protection of natural resources, such as wildlife. Sensors (e.g. drones equipped with cameras) have also begun to play a role in these scenarios by providing real-time information.
Incorporating both human and sensor defender resources strategically is the  subject of recent work on Security Games with Signaling (SGS). However, current methods to solve SGS do not scale well in terms of time or memory.
We therefore propose a novel approach to SGS, which, for the first time in this domain, employs an Evolutionary Computation paradigm: EASGS. EASGS effectively searches the huge SGS solution space via suitable solution encoding in a chromosome and a specially-designed set of operators. The operators include three types of mutations, each focusing on a particular aspect of the SGS solution, optimized crossover and a local coverage improvement scheme (a memetic aspect of EASGS). We also introduce a new set of benchmark games, based on dense or locally-dense graphs that reflect real-world SGS settings.
In the majority of 342 test game instances, EASGS outperforms state-of-the-art methods, including a reinforcement learning method, in terms of time scalability, nearly constant memory utilization, and quality of the returned defender's strategies (expected payoffs).

----

## [88] Detecting Out-Of-Context Objects Using Graph Contextual Reasoning Network

**Authors**: *Manoj Acharya, Anirban Roy, Kaushik Koneripalli, Susmit Jha, Christopher Kanan, Ajay Divakaran*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/89](https://doi.org/10.24963/ijcai.2022/89)

**Abstract**:

This paper presents an approach for detecting out-of-context (OOC) objects in images. Given an image with a set of objects, our goal is to determine if an object is inconsistent with the contextual relations and detect the OOC object with a bounding box. In this work, we consider common contextual relations such as co-occurrence relations, the relative size of an object with respect to other objects, and the position of the object in the scene. We posit that contextual cues are useful to determine object labels for in-context objects and inconsistent context cues are detrimental to determining object labels for out-of-context objects. To realize this hypothesis, we propose a graph contextual reasoning network (GCRN) to detect OOC objects. GCRN consists of two separate graphs to predict object labels based on the contextual cues in the image: 1) a representation graph to learn object features based on the neighboring objects and 2) a context graph to explicitly capture contextual cues from the neighboring objects. GCRN explicitly captures the contextual cues to improve the detection of in-context objects and identify objects that violate contextual relations. 
In order to evaluate our approach, we create a large-scale dataset by adding OOC object instances to the COCO images. We also evaluate on recent OCD benchmark. Our results show that GCRN outperforms competitive baselines in detecting OOC objects and correctly detecting in-context objects. Code and data: https://nusci.csl.sri.com/project/trinity-ooc

----

## [89] Axiomatic Foundations of Explainability

**Authors**: *Leila Amgoud, Jonathan Ben-Naim*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/90](https://doi.org/10.24963/ijcai.2022/90)

**Abstract**:

Improving trust in decisions made by classification models is becoming crucial for the acceptance of automated systems, and an important way of doing that is by providing explanations for the behaviour of the models. Different explainers have been proposed in the recent literature for that purpose, however their formal properties are under-studied. 

This paper investigates theoretically explainers that provide reasons behind decisions independently of instances.  Its contributions are fourfold. The first is to lay the foundations of such explainers by proposing key axioms, i.e., 
desirable properties they would satisfy. Two axioms are incompatible leading to two subsets. The second contribution consists of demonstrating that the first subset of axioms characterizes a family of explainers that return sufficient reasons while the second characterizes a family that provides necessary reasons. This sheds light on the axioms which distinguish the two types of reasons. As a third contribution, the paper introduces various explainers of both families, and fully characterizes some of them. Those explainers make use of the whole feature space. The fourth contribution is a family of explainers that generate explanations from  finite datasets (subsets of the feature space). This family, seen as an abstraction of Anchors and LIME, violates some axioms including one which prevents incorrect explanations.

----

## [90] On Preferred Abductive Explanations for Decision Trees and Random Forests

**Authors**: *Gilles Audemard, Steve Bellart, Louenas Bounia, Frédéric Koriche, Jean-Marie Lagniez, Pierre Marquis*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/91](https://doi.org/10.24963/ijcai.2022/91)

**Abstract**:

Abductive explanations take a central place in eXplainable Artificial Intelligence (XAI) by clarifying with few features 
the way data instances are classified. However, instances may have exponentially many minimum-size abductive explanations, and
this source of complexity holds even for ``intelligible'' classifiers, such as decision trees. When the number of such abductive explanations is huge,
computing one of them, only, is often not informative enough. Especially, better explanations than the one
that is derived may exist. As a way to circumvent this issue, we propose to leverage 
a model of the explainee, making precise her / his preferences about explanations, and to compute only 
preferred explanations. In this paper, several models are pointed out and discussed. For each model, we present and
evaluate an algorithm for computing preferred majoritary reasons, where majoritary reasons are specific abductive
explanations suited to random forests. We show that in practice the preferred majoritary reasons for an instance
can be far less numerous than its majoritary reasons.

----

## [91] Individual Fairness Guarantees for Neural Networks

**Authors**: *Elias Benussi, Andrea Patanè, Matthew Wicker, Luca Laurenti, Marta Kwiatkowska*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/92](https://doi.org/10.24963/ijcai.2022/92)

**Abstract**:

We consider the problem of certifying the individual fairness (IF) of feed-forward neural networks (NNs). 
In particular, we work with the epsilon-delta-IF formulation, which, given a NN and a similarity metric learnt from data, requires that the output difference between any pair of epsilon-similar individuals is bounded by a maximum decision tolerance delta >= 0. 
Working with a range of metrics, including the Mahalanobis distance, we propose a method to overapproximate the resulting optimisation problem using piecewise-linear functions to lower and upper bound the NN's non-linearities globally over the input space.
We encode this computation as the solution of a Mixed-Integer Linear Programming problem and demonstrate that it can be used to compute IF guarantees on four datasets widely used for fairness benchmarking.
We show how this formulation can be used to encourage models' fairness at training time by modifying the NN loss, and empirically confirm our approach yields NNs that are orders of magnitude fairer than state-of-the-art methods.

----

## [92] How Does Frequency Bias Affect the Robustness of Neural Image Classifiers against Common Corruption and Adversarial Perturbations?

**Authors**: *Alvin Chan, Yew Soon Ong, Clement Tan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/93](https://doi.org/10.24963/ijcai.2022/93)

**Abstract**:

Model robustness is vital for the reliable deployment of machine learning models in real-world applications. Recent studies have shown that data augmentation can result in model over-relying on features in the low-frequency domain, sacrificing performance against low-frequency corruptions, highlighting a connection between frequency and robustness. Here, we take one step further to more directly study the frequency bias of a model through the lens of its Jacobians and its implication to model robustness. To achieve this, we propose Jacobian frequency regularization for models' Jacobians to have a larger ratio of low-frequency components. Through experiments on four image datasets, we show that biasing classifiers towards low (high)-frequency components can bring performance gain against high (low)-frequency corruption and adversarial perturbation, albeit with a tradeoff in performance for low (high)-frequency corruption. Our approach elucidates a more direct connection between the frequency bias and robustness of deep learning models.

----

## [93] Learn to Reverse DNNs from AI Programs Automatically

**Authors**: *Simin Chen, Hamed Khanpour, Cong Liu, Wei Yang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/94](https://doi.org/10.24963/ijcai.2022/94)

**Abstract**:

With the privatization deployment of DNNs on edge devices, the security of on-device DNNs has raised significant concern. To quantify the model leakage risk of on-device DNNs automatically, we propose NNReverse, the first learning-based method which can reverse DNNs from AI programs without domain knowledge. NNReverse trains a representation model to represent the semantics of binary code for DNN layers. By searching the most similar function in our database, NNReverse infers the layer type of a given functionâ€™s binary code. To represent assembly instructions semantics precisely, NNReverse proposes a more fine-grained embedding model to represent the textual and structural-semantic of assembly functions.

----

## [94] CAT: Customized Adversarial Training for Improved Robustness

**Authors**: *Minhao Cheng, Qi Lei, Pin-Yu Chen, Inderjit S. Dhillon, Cho-Jui Hsieh*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/95](https://doi.org/10.24963/ijcai.2022/95)

**Abstract**:

Adversarial training has become one of the most effective methods for improving robustness of neural networks. However, it often suffers from poor generalization on both clean and perturbed data. Current robust training method always use a uniformed perturbation strength for every samples to generate  adversarial examples during model training for improving adversarial robustness. However, we show it would lead worse training and generalizaiton error and forcing the prediction to match one-hot label.
In this paper, therefore, we propose a new algorithm, named Customized Adversarial Training (CAT), which adaptively customizes the perturbation level and the corresponding label for each training sample in adversarial training. We first show theoretically the CAT scheme improves the generalization. Also, through extensive experiments, we show that the proposed algorithm achieves better clean and robust accuracy than previous adversarial training methods. The full version of this paper is available at https://arxiv.org/abs/2002.06789.

----

## [95] PPT: Backdoor Attacks on Pre-trained Models via Poisoned Prompt Tuning

**Authors**: *Wei Du, Yichun Zhao, Boqun Li, Gongshen Liu, Shilin Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/96](https://doi.org/10.24963/ijcai.2022/96)

**Abstract**:

Recently, prompt tuning has shown remarkable performance as a new learning paradigm, which freezes pre-trained language models (PLMs) and only tunes some soft prompts. A fixed PLM only needs to be loaded with different prompts to adapt different downstream tasks. However, the prompts associated with PLMs may be added with some malicious behaviors, such as backdoors. The victim model will be implanted with a backdoor by using the poisoned prompt. In this paper, we propose to obtain the poisoned prompt for PLMs and corresponding downstream tasks by prompt tuning. We name this Poisoned Prompt Tuning method "PPT". The poisoned prompt can lead a shortcut between the specific trigger word and the target label word to be created for the PLM. So the attacker can simply manipulate the prediction of the entire model by just a small prompt. Our experiments on various text classification tasks show that PPT can achieve a 99% attack success rate with almost no accuracy sacrificed on original task. We hope this work can raise the awareness of the possible security threats hidden in the prompt.

----

## [96] SoFaiR: Single Shot Fair Representation Learning

**Authors**: *Xavier Gitiaux, Huzefa Rangwala*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/97](https://doi.org/10.24963/ijcai.2022/97)

**Abstract**:

To avoid discriminatory uses of their data, organizations can learn to map them into a representation that filters out information related to sensitive  attributes. However, all existing methods in fair representation learning generate a fairness-information trade-off. To achieve different points on the fairness-information plane, one must train different models. In this paper, we first demonstrate that fairness-information trade-offs are fully characterized by rate-distortion trade-offs.  Then, we use this key result and propose SoFaiR, a single shot fair representation learning method that generates with one trained model many points on the fairness-information plane. Besides its computational saving, our single-shot approach is, to the extent of our knowledge, the first fair representation learning method that explains what information is affected by changes in the fairness / distortion properties of the representation. Empirically, we find on three datasets that SoFaiR achieves similar fairness information trade-offs as its multi-shot counterparts.

----

## [97] Fairness without the Sensitive Attribute via Causal Variational Autoencoder

**Authors**: *Vincent Grari, Sylvain Lamprier, Marcin Detyniecki*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/98](https://doi.org/10.24963/ijcai.2022/98)

**Abstract**:

In recent years, most fairness strategies in machine learning have focused on mitigating unwanted biases by assuming that the sensitive information is available. However, in practice this is not always the case: due to privacy purposes and regulations such as RGPD in EU, many personal sensitive attributes are frequently not collected. Yet, only a few prior works address the issue of mitigating bias in such a difficult setting, in particular to meet classical fairness objectives such as Demographic Parity and Equalized Odds. By leveraging recent developments for approximate inference, we propose in this paper an approach to fill this gap. To infer a sensitive information proxy, we introduce a new variational auto-encoding-based framework named SRCVAE that relies on knowledge of the underlying causal graph. The bias mitigation is then done in an adversarial fairness approach. Our proposed method empirically achieves significant improvements over existing works in the field. We observe that the generated proxyâ€™s latent space correctly recovers sensitive information and that our approach achieves a higher accuracy while obtaining the same level of fairness on two real datasets.

----

## [98] Taking Situation-Based Privacy Decisions: Privacy Assistants Working with Humans

**Authors**: *Nadin Kökciyan, Pinar Yolum*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/99](https://doi.org/10.24963/ijcai.2022/99)

**Abstract**:

Privacy on the Web is typically managed by giving consent to individual Websites for various aspects of data usage. This paradigm requires too much human effort and thus is impractical for Internet of Things (IoT) applications where humans interact with many new devices on a daily basis. Ideally, software privacy assistants can help by making privacy decisions in different situations on behalf of the users. To realize this, we propose an agent-based model for a privacy assistant. The model identifies the contexts that a situation implies and computes the trustworthiness of these contexts. Contrary to traditional trust models that capture trust in an entity by observing large number of interactions, our proposed model can assess the trustworthiness even if the user has not interacted with the particular device before. Moreover, our model can decide which situations are inherently ambiguous and thus can request the human to make the decision. We evaluate various aspects of the model using a real-life data set and report adjustments that are needed to serve different types of users well.

----

## [99] Model Stealing Defense against Exploiting Information Leak through the Interpretation of Deep Neural Nets

**Authors**: *Jeonghyun Lee, Sungmin Han, Sangkyun Lee*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/100](https://doi.org/10.24963/ijcai.2022/100)

**Abstract**:

Model stealing techniques allow adversaries to create attack models that mimic the functionality of black-box machine learning models, querying only class membership or probability outcomes. Recently, interpretable AI is getting increasing attention, to enhance our understanding of AI models, provide additional information for diagnoses, or satisfy legal requirements. However, it has been recently reported that providing such additional information can make AI models more vulnerable to model stealing attacks. In this paper, we propose DeepDefense, the first defense mechanism that protects an AI model against model stealing attackers exploiting both class probabilities and interpretations. DeepDefense uses a misdirection model to hide the critical information of the original model against model stealing attacks, with minimal degradation on both the class probability and the interpretability of prediction output. DeepDefense is highly applicable for any model stealing scenario since it makes minimal assumptions about the model stealing adversary. In our experiments, DeepDefense shows significantly higher defense performance than the existing state-of-the-art defenses on various datasets and interpreters.

----

## [100] Investigating and Explaining the Frequency Bias in Image Classification

**Authors**: *Zhiyu Lin, Yifei Gao, Jitao Sang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/101](https://doi.org/10.24963/ijcai.2022/101)

**Abstract**:

CNNs exhibit many behaviors different from humans, one of which is the capability of employing high-frequency components. This paper discusses the frequency bias phenomenon in image classification tasks: the high-frequency components are actually much less exploited than the low- and mid- frequency components. We first investigate the frequency bias phenomenon by presenting two observations on feature discrimination and learning priority. Furthermore, we hypothesize that (1) the spectral density, (2) class consistency directly affect the frequency bias. Specifically, our investigations verify that the spectral density of datasets mainly affects the learning priority, while the class consistency mainly affects the feature discrimination.

----

## [101] AttExplainer: Explain Transformer via Attention by Reinforcement Learning

**Authors**: *Runliang Niu, Zhepei Wei, Yan Wang, Qi Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/102](https://doi.org/10.24963/ijcai.2022/102)

**Abstract**:

Transformer and its variants, built based on attention mechanisms, have recently achieved remarkable performance in many NLP tasks. Most existing works on Transformer explanation tend to reveal and utilize the attention matrix with human subjective intuitions in a qualitative manner. However, the huge size of dimensions directly challenges these methods to quantitatively analyze the attention matrix. Therefore, in this paper, we propose a novel reinforcement learning (RL) based framework for Transformer explanation via attention matrix, namely AttExplainer. The RL agent learns to perform step-by-step masking operations by observing the change in attention matrices. We have adapted our method to two scenarios, perturbation-based model explanation and text adversarial attack. Experiments on three widely used text classification benchmarks validate the effectiveness of the proposed method compared to state-of-the-art baselines. Additional studies show that our method is highly transferable and consistent with human intuition. The code of this paper is available at https://github.com/niuzaisheng/AttExplainer .

----

## [102] Counterfactual Interpolation Augmentation (CIA): A Unified Approach to Enhance Fairness and Explainability of DNN

**Authors**: *Yao Qiang, Chengyin Li, Marco Brocanelli, Dongxiao Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/103](https://doi.org/10.24963/ijcai.2022/103)

**Abstract**:

Bias in the training data can jeopardize fairness and explainability of deep neural network prediction on test data. We propose a novel bias-tailored data augmentation approach, Counterfactual Interpolation Augmentation (CIA), attempting to debias the training data by d-separating the spurious correlation between the target variable and the sensitive attribute. CIA generates counterfactual interpolations along a path simulating the distribution transitions between the input and its counterfactual example. CIA as a pre-processing approach enjoys two advantages: First, it couples with either plain training or debiasing training to markedly increase fairness over the sensitive attribute. Second, it enhances the explainability of deep neural networks by generating attribution maps via integrating counterfactual gradients. We demonstrate the superior performance of the CIA-trained deep neural network models using qualitative and quantitative experimental results. Our code is available at: https://github.com/qiangyao1988/CIA

----

## [103] BayCon: Model-agnostic Bayesian Counterfactual Generator

**Authors**: *Piotr Romashov, Martin Gjoreski, Kacper Sokol, Maria Vanina Martinez, Marc Langheinrich*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/104](https://doi.org/10.24963/ijcai.2022/104)

**Abstract**:

Generating counterfactuals to discover hypothetical predictive scenarios is the de facto standard for explaining machine learning models and their predictions. However, building a counterfactual explainer that is time-efficient, scalable, and model-agnostic, in addition to being compatible with continuous and categorical attributes, remains an open challenge. To complicate matters even more, ensuring that the contrastive instances are optimised for feature sparsity, remain close to the explained instance, and are not drawn from outside of the data manifold, is far from trivial. To address this gap we propose BayCon: a novel counterfactual generator based on probabilistic feature sampling and Bayesian optimisation. Such an approach can combine multiple objectives by employing a surrogate model to guide the counterfactual search. We demonstrate the advantages of our method through a collection of experiments based on six real-life datasets representing three regression tasks and three classification tasks.

----

## [104] What Does My GNN Really Capture? On Exploring Internal GNN Representations

**Authors**: *Luca Veyrin-Forrer, Ataollah Kamal, Stefan Duffner, Marc Plantevit, Céline Robardet*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/105](https://doi.org/10.24963/ijcai.2022/105)

**Abstract**:

Graph Neural Networks (GNNs) are very efficient at classifying graphs but their internal functioning is opaque which limits their field of application. Existing methods to explain GNN focus on disclosing the relationships between input graphs and model decision. In this article, we propose a method that goes further and isolates the internal features, hidden in the network layers, that are automatically identified by the GNN and used in the decision process. We show that this method makes possible to know the parts of the input graphs used by GNN with much less bias that SOTA methods and thus to bring confidence in the decision process.

----

## [105] Shielding Federated Learning: Robust Aggregation with Adaptive Client Selection

**Authors**: *Wei Wan, Shengshan Hu, Jianrong Lu, Leo Yu Zhang, Hai Jin, Yuanyuan He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/106](https://doi.org/10.24963/ijcai.2022/106)

**Abstract**:

Federated learning (FL) enables multiple clients to collaboratively train an accurate global model while protecting clients' data privacy. However, FL is susceptible to Byzantine attacks from malicious participants. Although the problem has gained significant attention, existing defenses have several flaws:  the server irrationally chooses malicious clients for aggregation even after they have been detected in previous rounds; the defenses perform ineffectively against sybil attacks or in the heterogeneous data setting.
    
    To overcome these issues, we propose MAB-RFL, a new method for robust aggregation in FL. By modelling the client selection as an extended multi-armed bandit (MAB) problem, we propose an adaptive client selection strategy to choose honest clients that are more likely to contribute high-quality updates. We then propose two approaches to identify malicious updates from sybil and non-sybil attacks, based on which rewards for each client selection decision can be accurately evaluated to discourage malicious behaviors. MAB-RFL achieves a satisfying balance between exploration and exploitation on the potential benign clients. Extensive experimental results show that MAB-RFL outperforms existing defenses in three attack scenarios under different percentages of attackers.

----

## [106] Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations

**Authors**: *Run Wang, Ziheng Huang, Zhikai Chen, Li Liu, Jing Chen, Lina Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/107](https://doi.org/10.24963/ijcai.2022/107)

**Abstract**:

DeepFake is becoming a real risk to society and brings potential threats to both individual privacy and political security due to the DeepFaked multimedia are realistic and convincing. However, the popular DeepFake passive detection is an ex-post forensics countermeasure and failed in blocking the disinformation spreading in advance. To address this limitation, researchers study the proactive defense techniques by adding adversarial noises into the source data to disrupt the DeepFake manipulation. However, the existing studies on proactive DeepFake defense via injecting adversarial noises are not robust, which could be easily bypassed by employing simple image reconstruction revealed in a recent study MagDR. 

In this paper, we investigate the vulnerability of the existing forgery techniques and propose a novel anti-forgery technique that helps users protect the shared facial images from attackers who are capable of applying the popular forgery techniques. Our proposed method generates perceptual-aware perturbations in an incessant manner which is vastly different from the prior studies by adding adversarial noises that is sparse. Experimental results reveal that our perceptual-aware perturbations are robust to diverse image transformations, especially the competitive evasion technique, MagDR via image reconstruction. Our findings potentially open up a new research direction towards thorough understanding and investigation of perceptual-aware adversarial attack for protecting facial images against DeepFakes in a proactive and robust manner. Code is available at https://github.com/AbstractTeen/AntiForgery.

----

## [107] Cluster Attack: Query-based Adversarial Attacks on Graph with Graph-Dependent Priors

**Authors**: *Zhengyi Wang, Zhongkai Hao, Ziqiao Wang, Hang Su, Jun Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/108](https://doi.org/10.24963/ijcai.2022/108)

**Abstract**:

While deep neural networks have achieved great success in graph analysis, recent work has shown that they are vulnerable to adversarial attacks. Compared with adversarial attacks on image classification, performing adversarial attacks on graphs is more challenging because of the discrete and non-differential nature of the adjacent matrix for a graph. In this work, we propose Cluster Attack --- a Graph Injection Attack (GIA) on node classification, which injects fake nodes into the original graph to degenerate the performance of graph neural networks (GNNs) on certain victim nodes while affecting the other nodes as little as possible. We demonstrate that a GIA problem can be equivalently formulated as a graph clustering problem; thus, the discrete optimization problem of the adjacency matrix can be solved in the context of graph clustering. In particular, we propose to measure the similarity between victim nodes by a metric of Adversarial Vulnerability, which is related to how the victim nodes will be affected by the injected fake node, and to cluster the victim nodes accordingly. Our attack is performed in a practical and unnoticeable query-based black-box manner with only a few nodes on the graphs that can be accessed. Theoretical analysis and extensive experiments demonstrate the effectiveness of our method by fooling the node classifiers with only a small number of queries.

----

## [108] MetaFinger: Fingerprinting the Deep Neural Networks with Meta-training

**Authors**: *Kang Yang, Run Wang, Lina Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/109](https://doi.org/10.24963/ijcai.2022/109)

**Abstract**:

As deep neural networks (DNNs) play a critical role in various fields, the models themselves hence are becoming an important asset that needs to be protected. To achieve this, various neural network fingerprint methods have been proposed. However, existing fingerprint methods fingerprint the decision boundary by adversarial examples, which is not robust to model modification and adversarial defenses. To fill this gap, we propose a robust fingerprint method MetaFinger, which fingerprints the inner decision area  of the model by meta-training, rather than the decision boundary. Specifically, we first generate many shadow models with DNN augmentation as meta-data. Then we optimize some images by meta-training  to ensure that only models derived from the protected model can recognize them. To demonstrate the robustness of our fingerprint approach, we evaluate our method against two types of attacks including input modification and model modification. Experiments show that our method achieves 99.34% and 97.69% query accuracy on average, surpassing existing methods over 30%, 25% on CIFAR-10 and Tiny-ImageNet, respectively. Our code is available at https://github.com/kangyangWHU/MetaFinger.

----

## [109] Approximately EFX Allocations for Indivisible Chores

**Authors**: *Shengwei Zhou, Xiaowei Wu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/110](https://doi.org/10.24963/ijcai.2022/110)

**Abstract**:

In this paper we study how to fairly allocate a set of m indivisible chores to a group of n agents, each of which has a general additive cost function on the items. Since envy-free (EF) allocation is not guaranteed to exist, we consider the notion of envy-freeness up to any item (EFX). In contrast to the fruitful results regarding the (approximation of) EFX allocations for goods, very little is known for the allocation of chores. Prior to our work, for the allocation of chores, it is known that EFX allocations always exist for two agents, or general number of agents with identical ordering cost functions. For general instances, no non-trivial approximation result regarding EFX allocation is known. In this paper we make some progress in this direction by showing that for three agents we can always compute a 5-approximation of EFX allocation in polynomial time. For n>=4 agents, our algorithm always computes an allocation that achieves an approximation ratio of 3n^2 regarding EFX. We also study the bi-valued instances, in which agents have at most two cost values on the chores, and provide polynomial time algorithms for the computation of EFX allocation when n=3, and (n-1)-approximation of EFX allocation when n>=4.

----

## [110] MotionMixer: MLP-based 3D Human Body Pose Forecasting

**Authors**: *Arij Bouazizi, Adrian Holzbock, Ulrich Kressel, Klaus Dietmayer, Vasileios Belagiannis*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/111](https://doi.org/10.24963/ijcai.2022/111)

**Abstract**:

In this work, we present MotionMixer, an efficient 3D human body pose forecasting model based solely on multi-layer perceptrons (MLPs). MotionMixer learns the spatial-temporal 3D body pose dependencies by sequentially mixing both modalities. Given a stacked sequence of 3D body poses, a spatial-MLP extracts fine-grained spatial dependencies of the body joints. The interaction of the body joints over time is then modelled by a temporal MLP. The spatial-temporal mixed features are finally aggregated and decoded to obtain the future motion. To calibrate the influence of each time step in the pose sequence, we make use of squeeze-and-excitation (SE) blocks. We evaluate our approach on Human3.6M, AMASS, and 3DPW datasets using the standard evaluation protocols. For all evaluations, we demonstrate state-of-the-art performance, while having a model with a smaller number of parameters. Our code is available at: https://github.com/MotionMLP/MotionMixer.

----

## [111] Event-driven Video Deblurring via Spatio-Temporal Relation-Aware Network

**Authors**: *Chengzhi Cao, Xueyang Fu, Yurui Zhu, Gege Shi, Zheng-Jun Zha*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/112](https://doi.org/10.24963/ijcai.2022/112)

**Abstract**:

Video deblurring with event information has attracted considerable attention. To help deblur each frame, existing methods usually compress a specific event sequence into a feature tensor with the same size as the corresponding video. However, this strategy neither considers the pixel-level spatial brightness changes nor the temporal correlation between events at each time step, resulting in insufficient use of spatio-temporal information. To address this issue, we propose a new Spatio-Temporal Relation-Attention network (STRA), for the specific event-based video deblurring. Concretely, to utilize spatial consistency between the frame and event, we model the brightness changes as an extra prior to aware blurring contexts in each frame; to record temporal relationship among different events, we develop a temporal memory block to restore long-range dependencies of event sequences continuously. In this way, the complementary information contained in the events and frames, as well as the correlation of neighboring events, can be fully utilized to recover spatial texture from events constantly. Experiments show that our STRA significantly outperforms several competing methods, e.g., on the HQF dataset, our network achieves up to 1.3 dB in terms of PSNR over the most advanced method. The code is available at https://github.com/Chengzhi-Cao/STRA.

----

## [112] KPN-MFI: A Kernel Prediction Network with Multi-frame Interaction for Video Inverse Tone Mapping

**Authors**: *Gaofeng Cao, Fei Zhou, Han Yan, Anjie Wang, Leidong Fan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/113](https://doi.org/10.24963/ijcai.2022/113)

**Abstract**:

Up to now, the image-based inverse tone mapping (iTM) models have been widely investigated, while there is little research on video-based iTM methods. It would be interesting to make use of these existing image-based models in the video iTM task. However, directly transferring the imagebased iTM models to video data without modeling spatial-temporal information remains nontrivial and challenging. Considering both the intra-frame quality and the inter-frame consistency of a video, this article presents a new video iTM method based on a kernel prediction network (KPN), which takes advantage of multi-frame interaction (MFI) module to capture temporal-spatial information for video data. Specifically, a basic encoder-decoder KPN, essentially designed for image iTM, is trained to guarantee the mapping quality within each frame. More importantly, the MFI module is incorporated to capture temporal-spatial context information and preserve the inter-frame consistency by exploiting the correction between adjacent frames. Notably, we can readily extend any existing image iTM models to video iTM ones by involving the proposed MFI module. Furthermore, we propose an inter-frame brightness consistency loss function based on the Gaussian pyramid to reduce the video temporal inconsistency. Extensive experiments demonstrate that our model outperforms state-ofthe-art image and video-based methods. The code
is available at https://github.com/caogaofeng/KPNMFI.

----

## [113] Zero-Shot Logit Adjustment

**Authors**: *Dubing Chen, Yuming Shen, Haofeng Zhang, Philip H. S. Torr*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/114](https://doi.org/10.24963/ijcai.2022/114)

**Abstract**:

Semantic-descriptor-based Generalized Zero-Shot Learning (GZSL) poses challenges in recognizing novel classes in the test phase. The development of generative models enables current GZSL techniques to probe further into the semantic-visual link, culminating in a two-stage form that includes a generator and a classifier. However, existing generation-based methods focus on enhancing the generator's effect while neglecting the improvement of the classifier. In this paper, we first analyze of two properties of the generated pseudo unseen samples: bias and homogeneity. Then, we perform variational Bayesian inference to back-derive the evaluation metrics, which reflects the balance of the seen and unseen classes. As a consequence of our derivation, the aforementioned two properties are incorporated into the classifier training as seen-unseen priors via logit adjustment. The Zero-Shot Logit Adjustment further puts semantic-based classifiers into effect in generation-based GZSL. Our experiments demonstrate that the proposed technique achieves state-of-the-art when combined with the basic generator, and it can improve various generative Zero-Shot Learning frameworks. Our codes are available on https://github.com/cdb342/IJCAI-2022-ZLA.

----

## [114] Uncertainty-Aware Representation Learning for Action Segmentation

**Authors**: *Lei Chen, Muheng Li, Yueqi Duan, Jie Zhou, Jiwen Lu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/115](https://doi.org/10.24963/ijcai.2022/115)

**Abstract**:

In this paper, we propose an uncertainty-aware representation Learning (UARL) method for action segmentation. Most existing action segmentation methods exploit continuity information of the action period to predict frame-level labels, which ignores the temporal ambiguity of the transition region between two actions. Moreover, similar periods of different actions, e.g., the beginning of some actions, will confuse the network if they are annotated with different labels, which causes spatial ambiguity. To address this, we design the UARL to exploit the transitional expression between two action periods by uncertainty learning. Specially, we model every frame of actions with an active distribution that represents the probabilities of different actions, which captures the uncertainty of the action and exploits the tendency during the action. We evaluate our method on three popular action prediction datasets: Breakfast, Georgia Tech Egocentric Activities (GTEA), and 50Salads. The experimental results demonstrate that our method achieves the performance with state-of-the-art.

----

## [115] AutoAlign: Pixel-Instance Feature Aggregation for Multi-Modal 3D Object Detection

**Authors**: *Zehui Chen, Zhenyu Li, Shiquan Zhang, Liangji Fang, Qinhong Jiang, Feng Zhao, Bolei Zhou, Hang Zhao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/116](https://doi.org/10.24963/ijcai.2022/116)

**Abstract**:

Object detection through either RGB images or the LiDAR point clouds has been extensively explored in autonomous driving. However, it remains challenging to make these two data sources complementary and beneficial to each other.  In this paper, we propose AutoAlign, an automatic feature fusion strategy for 3D object detection. Instead of establishing deterministic correspondence with camera projection matrix, we model the mapping relationship between the image and point clouds with a learnable alignment map. This map enables our model to automate the alignment of non-homogenous features in a dynamic and data-driven manner. Specifically, a cross-attention feature alignment module is devised to adaptively aggregate pixel-level image features for each voxel. To enhance the semantic consistency during feature alignment, we also design a self-supervised cross-modal feature interaction module, through which the model can learn feature aggregation with instance-level feature guidance. Extensive experimental results show that our approach can lead to 2.3 mAP and 7.0 mAP improvements on the KITTI and nuScenes datasets respectively. Notably, our best model reaches 70.9 NDS on the nuScenes testing leaderboard, achieving competitive performance among various state-of-the-arts.

----

## [116] Unsupervised Multi-Modal Medical Image Registration via Discriminator-Free Image-to-Image Translation

**Authors**: *Zekang Chen, Jia Wei, Rui Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/117](https://doi.org/10.24963/ijcai.2022/117)

**Abstract**:

In clinical practice, well-aligned multi-modal images, such as Magnetic Resonance (MR) and Computed Tomography (CT), together can provide complementary information for image-guided therapies. Multi-modal image registration is essential for the accurate alignment of these multi-modal images. However, it remains a very challenging task due to complicated and unknown spatial correspondence between different modalities. In this paper, we propose a novel translation-based unsupervised deformable image registration approach to convert the multi-modal registration problem to a mono-modal one. Specifically, our approach incorporates a discriminator-free translation network to facilitate the training of the registration network and a patchwise contrastive loss to encourage the translation network to preserve object shapes. Furthermore, we propose to replace an adversarial loss, that is widely used in previous multi-modal image registration methods, with a pixel loss in order to integrate the output of translation into the target modality. This leads to an unsupervised method requiring no ground-truth deformation or pairs of aligned images for training. We evaluate four variants of our approach on the public Learn2Reg 2021 datasets. The experimental results demonstrate that the proposed architecture achieves state-of-the-art performance. Our code is available at https://github.com/heyblackC/DFMIR.

----

## [117] SpanConv: A New Convolution via Spanning Kernel Space for Lightweight Pansharpening

**Authors**: *Zhi-Xuan Chen, Cheng Jin, Tian-Jing Zhang, Xiao Wu, Liang-Jian Deng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/118](https://doi.org/10.24963/ijcai.2022/118)

**Abstract**:

Standard convolution operations can effectively perform feature extraction and representation but result in high computational cost, largely due to the generation of the original convolution kernel corresponding to the channel dimension of the feature map, which will cause unnecessary redundancy. In this paper, we focus on kernel generation and present an interpretable span strategy, named SpanConv, for the effective construction of kernel space. Specifically, we first learn two navigated kernels with single channel as bases, then extend the two kernels by learnable coefficients, and finally span the two sets of kernels by their linear combination to construct the so-called SpanKernel. The proposed SpanConv is realized by replacing plain convolution kernel by SpanKernel. To verify the effectiveness of SpanConv, we design a simple network with SpanConv. Experiments demonstrate the proposed network significantly reduces parameters comparing with benchmark networks for remote sensing pansharpening, while achieving competitive performance and excellent generalization. Code is available at https://github.com/zhi-xuan-chen/IJCAI-2022 SpanConv.

----

## [118] Robust Single Image Dehazing Based on Consistent and Contrast-Assisted Reconstruction

**Authors**: *De Cheng, Yan Li, Dingwen Zhang, Nannan Wang, Xinbo Gao, Jiande Sun*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/119](https://doi.org/10.24963/ijcai.2022/119)

**Abstract**:

Single image dehazing as a fundamental low-level vision task, is essential for the development of robust intelligent surveillance system. In this paper, we make an early effort to consider dehazing robustness under variational haze density, which is a realistic while under-studied problem in the research filed of singe image dehazing. To properly address this problem, we propose a novel density-variational learning framework to improve the robustness of the image dehzing model assisted by a variety of negative hazy images, to better deal with various complex hazy scenarios. Specifically, the dehazing network is optimized under the consistency-regularized framework with the proposed Contrast-Assisted Reconstruction Loss (CARL). The CARL can fully exploit the negative information to facilitate the traditional positive-orient dehazing objective function, by squeezing the dehazed image to its clean target from different directions. Meanwhile, the consistency regularization keeps consistent outputs given multi-level hazy images, thus improving the model robustness. Extensive experimental results on two synthetic and  three real-world datasets demonstrate that our method significantly surpasses the state-of-the-art approaches.

----

## [119] I²R-Net: Intra- and Inter-Human Relation Network for Multi-Person Pose Estimation

**Authors**: *Yiwei Ding, Wenjin Deng, Yinglin Zheng, Pengfei Liu, Meihong Wang, Xuan Cheng, Jianmin Bao, Dong Chen, Ming Zeng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/120](https://doi.org/10.24963/ijcai.2022/120)

**Abstract**:

In this paper, we present the Intra- and Inter-Human Relation Networks I²R-Net for Multi-Person Pose Estimation. It involves two basic modules. First, the Intra-Human Relation Module operates on a single person and aims to capture Intra-Human dependencies. Second, the Inter-Human Relation Module considers the relation between multiple instances and focuses on capturing Inter-Human interactions. The Inter-Human Relation Module can be designed very lightweight by reducing the resolution of feature map, yet learn useful relation information to significantly boost the performance of the Intra-Human Relation Module. Even without bells and whistles, our method can compete or outperform current competition winners. We conduct extensive experiments on COCO, CrowdPose, and OCHuman datasets. The results demonstrate that the proposed model surpasses all the state-of-the-art methods. Concretely, the proposed method achieves 77.4% AP on CrowPose dataset and 67.8% AP on OCHuman dataset respectively, outperforming existing methods by a large margin. Additionally, the ablation study and visualization analysis also prove the effectiveness of our model.

----

## [120] Region-Aware Metric Learning for Open World Semantic Segmentation via Meta-Channel Aggregation

**Authors**: *Hexin Dong, Zifan Chen, Mingze Yuan, Yutong Xie, Jie Zhao, Fei Yu, Bin Dong, Li Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/121](https://doi.org/10.24963/ijcai.2022/121)

**Abstract**:

As one of the most challenging and practical segmentation tasks, open-world semantic segmentation requires the model to segment the anomaly regions in the images and incrementally learn to segment out-of-distribution (OOD) objects, especially under a few-shot condition. The current state-of-the-art (SOTA) method, Deep Metric Learning Network (DMLNet), relies on pixel-level metric learning, with which the identification of similar regions having different semantics is difficult. Therefore, we propose a method called region-aware metric learning (RAML), which first separates the regions of the images and generates region-aware features for further metric learning. RAML improves the integrity of the segmented anomaly regions. Moreover, we propose a novel meta-channel aggregation (MCA) module to further separate anomaly regions, forming high-quality sub-region candidates and thereby improving the model performance for OOD objects. To evaluate the proposed RAML, we have conducted extensive experiments and ablation studies on Lost And Found and Road Anomaly datasets for anomaly segmentation and the CityScapes dataset for incremental few-shot learning. The results show that the proposed RAML achieves SOTA performance in both stages of open world segmentation. Our code and appendix are available at https://github.com/czifan/RAML.

----

## [121] MNet: Rethinking 2D/3D Networks for Anisotropic Medical Image Segmentation

**Authors**: *Zhangfu Dong, Yuting He, Xiaoming Qi, Yang Chen, Huazhong Shu, Jean-Louis Coatrieux, Guanyu Yang, Shuo Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/122](https://doi.org/10.24963/ijcai.2022/122)

**Abstract**:

The nature of thick-slice scanning causes severe inter-slice discontinuities of 3D medical images, and the vanilla 2D/3D convolutional neural networks (CNNs) fail to represent sparse inter-slice information and dense intra-slice information in a balanced way, leading to severe underfitting to inter-slice features (for vanilla 2D CNNs) and overfitting to noise from long-range slices (for vanilla 3D CNNs). In this work, a novel mesh network (MNet) is proposed to balance the spatial representation inter axes via learning. 1) Our MNet latently fuses plenty of representation processes by embedding multi-dimensional convolutions deeply into basic modules, making the selections of representation processes flexible, thus balancing representation for sparse inter-slice information and dense intra-slice information adaptively. 2) Our MNet latently fuses multi-dimensional features inside each basic module, simultaneously taking the advantages of 2D (high segmentation accuracy of the easily recognized regions in 2D view) and 3D (high smoothness of 3D organ contour) representations, thus obtaining more accurate modeling for target regions. Comprehensive experiments are performed on four public datasets (CT\&MR), the results consistently demonstrate the proposed MNet outperforms the other methods. The code and datasets are available at: https://github.com/zfdong-code/MNet

----

## [122] ICGNet: Integration Context-based Reverse-Contour Guidance Network for Polyp Segmentation

**Authors**: *Xiuquan Du, Xuebin Xu, Kunpeng Ma*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/123](https://doi.org/10.24963/ijcai.2022/123)

**Abstract**:

Precise segmentation of polyps from colonoscopic images is extremely significant for the early diagnosis and treatment of colorectal cancer. However, it is still a challenging task due to: (1)the boundary between the polyp and the background is blurred makes delineation difficult; (2)the various size and shapes causes feature representation of polyps difficult. In this paper, we propose an integration context-based reverse-contour guidance network (ICGNet) to solve these challenges. The ICGNet firstly utilizes a reverse-contour guidance module to aggregate low-level edge detail information and meanwhile constraint reverse region. Then, the newly designed adaptive context module is used to adaptively extract local-global information of the current layer and complementary information of the previous layer to get larger and denser features. Lastly, an innovative hybrid pyramid pooling fusion module fuses the multi-level features generated from the decoder in the case of considering salient features and less background. Our proposed approach is evaluated on the EndoScene, Kvasir-SEG and CVC-ColonDB datasets with eight evaluation metrics, and gives competitive results compared with other state-of-the-art methods in both learning ability and generalization capability.

----

## [123] SVTR: Scene Text Recognition with a Single Visual Model

**Authors**: *Yongkun Du, Zhineng Chen, Caiyan Jia, Xiaoting Yin, Tianlun Zheng, Chenxia Li, Yuning Du, Yu-Gang Jiang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/124](https://doi.org/10.24963/ijcai.2022/124)

**Abstract**:

Dominant scene text recognition models commonly contain two building blocks, a visual model for feature extraction and a sequence model for text transcription. This hybrid architecture, although accurate, is complex and less efficient. In this study, we propose a Single Visual model for Scene Text recognition within the patch-wise image tokenization framework, which dispenses with the sequential modeling entirely. The method, termed SVTR, firstly decomposes an image text into small patches named character components. Afterward, hierarchical stages are recurrently carried out by component-level mixing, merging and/or combining. Global and local mixing blocks are devised to perceive the inter-character and intra-character patterns, leading to a multi-grained character component perception. Thus, characters are recognized by a simple linear prediction. Experimental results on both English and Chinese scene text recognition tasks demonstrate the effectiveness of SVTR. SVTR-L (Large) achieves highly competitive accuracy in English and outperforms existing methods by a large margin in Chinese, while running faster. In addition, SVTR-T (Tiny) is an effective and much smaller model, which shows appealing speed at inference. The code is publicly available at https://github.com/PaddlePaddle/PaddleOCR.

----

## [124] Learning Coated Adversarial Camouflages for Object Detectors

**Authors**: *Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Jin Zhang, Wu Zhang, Zhisong Pan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/125](https://doi.org/10.24963/ijcai.2022/125)

**Abstract**:

An adversary can fool deep neural network object detectors by generating adversarial noises. Most of the existing works focus on learning local visible noises in an adversarial "patch" fashion. However, the 2D patch attached to a 3D object tends to suffer from an inevitable reduction in attack performance as the viewpoint changes. To remedy this issue, this work proposes the Coated Adversarial Camouflage (CAC) to attack the detectors in arbitrary viewpoints. Unlike the patch trained in the 2D space, our camouflage generated by a conceptually different training framework consists of 3D rendering and dense proposals attack. Specifically, we make the camouflage perform 3D spatial transformations according to the pose changes of the object. Based on the multi-view rendering results, the top-n proposals of the region proposal network are fixed, and all the classifications in the fixed dense proposals are attacked simultaneously to output errors. In addition, we build a virtual 3D scene to fairly and reproducibly evaluate different attacks. Extensive experiments demonstrate the superiority of CAC over the existing attacks, and it shows impressive performance both in the virtual scene and the real world. This poses a potential threat to the security-critical computer vision systems.

----

## [125] D-DPCC: Deep Dynamic Point Cloud Compression via 3D Motion Prediction

**Authors**: *Tingyu Fan, Linyao Gao, Yiling Xu, Zhu Li, Dong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/126](https://doi.org/10.24963/ijcai.2022/126)

**Abstract**:

The non-uniformly distributed nature of the 3D Dynamic Point Cloud (DPC) brings significant challenges to its high-efficient inter-frame compression. This paper proposes a novel 3D sparse convolution-based Deep Dynamic Point Cloud Compression (D-DPCC) network to compensate and compress the DPC geometry with 3D motion estimation and motion compensation in the feature space. In the proposed D-DPCC network, we design a Multi-scale Motion Fusion (MMF) module to accurately estimate the 3D optical flow between the feature representations of adjacent point cloud frames. Specifically, we utilize a 3D sparse convolution-based encoder to obtain the latent representation for motion estimation in the feature space and introduce the proposed MMF module for fused 3D motion embedding. Besides, for motion compensation, we propose a 3D Adaptively Weighted Interpolation (3DAWI) algorithm with a penalty coefficient to adaptively decrease the impact of distant neighbours. We compress the motion embedding and the residual with a lossy autoencoder-based network. To our knowledge, this paper is the first work proposing an end-to-end deep dynamic point cloud compression framework. The experimental result shows that the proposed D-DPCC framework achieves an average 76% BD-Rate (Bjontegaard Delta Rate) gains against state-of-the-art Video-based Point Cloud Compression (V-PCC) v13 in inter mode.

----

## [126] SparseTT: Visual Tracking with Sparse Transformers

**Authors**: *Zhihong Fu, Zehua Fu, Qingjie Liu, Wenrui Cai, Yunhong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/127](https://doi.org/10.24963/ijcai.2022/127)

**Abstract**:

Transformers have been successfully applied to the visual tracking task and significantly promote tracking performance. The self-attention mechanism designed to model long-range dependencies is the key to the success of Transformers. However, self-attention lacks focusing on the most relevant information in the search regions, making it easy to be distracted by background. In this paper, we relieve this issue with a sparse attention mechanism by focusing the most relevant information in the search regions, which enables a much accurate tracking. Furthermore, we introduce a double-head predictor to boost the accuracy of foreground-background classification and regression of target bounding boxes, which further improve the tracking performance. Extensive experiments show that, without bells and whistles, our method significantly outperforms the state-of-the-art approaches on LaSOT, GOT-10k, TrackingNet, and UAV123, while running at 40 FPS. Notably, the training time of our method is reduced by 75% compared to that of TransT. The source code and models are available at https://github.com/fzh0917/SparseTT.

----

## [127] Lightweight Bimodal Network for Single-Image Super-Resolution via Symmetric CNN and Recursive Transformer

**Authors**: *Guangwei Gao, Zhengxue Wang, Juncheng Li, Wenjie Li, Yi Yu, Tieyong Zeng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/128](https://doi.org/10.24963/ijcai.2022/128)

**Abstract**:

Single-image super-resolution (SISR) has achieved significant breakthroughs with the development of deep learning. However, these methods are difficult to be applied in real-world scenarios since they are inevitably accompanied by the problems of computational and memory costs caused by the complex operations. To solve this issue, we propose a Lightweight Bimodal Network (LBNet) for SISR. Specifically, an effective Symmetric CNN is designed for local feature extraction and coarse image reconstruction. Meanwhile, we propose a Recursive Transformer to fully learn the long-term dependence of images thus the global information can be fully used to further refine texture details. Studies show that the hybrid of CNN and Transformer can build a more efficient model. Extensive experiments have proved that our LBNet achieves more prominent performance than other state-of-the-art methods with a relatively low computational cost and memory consumption. The code is available at https://github.com/IVIPLab/LBNet.

----

## [128] Region-Aware Temporal Inconsistency Learning for DeepFake Video Detection

**Authors**: *Zhihao Gu, Taiping Yao, Yang Chen, Ran Yi, Shouhong Ding, Lizhuang Ma*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/129](https://doi.org/10.24963/ijcai.2022/129)

**Abstract**:

The rapid development of face forgery techniques has drawn growing attention due to security concerns.  Existing deepfake video detection methods always attempt to capture the discriminative features by directly exploiting static temporal convolution to mine temporal inconsistency, without explicit exploration on the diverse temporal dynamics of different forged regions. To effectively and comprehensively capture the various inconsistency, in this paper, we propose a novel Region-Aware Temporal Filter (RATF) module which automatically generates corresponding temporal filters for different spatial regions. Specifically, we decouple the dynamic temporal kernel into a set of region-agnostic basic filters and region-sensitive aggregation weights. And different weights guide the corresponding regions to adaptively learn temporal inconsistency, which greatly enhances the overall representational ability. Moreover, to cover the long-term temporal dynamics, we divide the video into multiple snippets and propose a Cross-Snippet Attention (CSA) to promote the cross-snippet information interaction. Extensive experiments and visualizations on several benchmarks demonstrate the effectiveness of our method against state-of-the-art competitors.

----

## [129] Learning Target-aware Representation for Visual Tracking via Informative Interactions

**Authors**: *Mingzhe Guo, Zhipeng Zhang, Heng Fan, Liping Jing, Yilin Lyu, Bing Li, Weiming Hu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/130](https://doi.org/10.24963/ijcai.2022/130)

**Abstract**:

We introduce a novel backbone architecture to improve target-perception ability of feature representation for tracking. Having observed de facto frameworks perform feature matching simply using the backbone outputs for target localization, there is no direct feedback from the matching module to the backbone network, especially the shallow layers. Concretely, only the matching module can directly access the target information, while the representation learning of candidate frame is blind to the reference target. Therefore, the accumulated target-irrelevant interference in shallow stages may degrade the feature quality of deeper layers. In this paper, we approach the problem by conducting multiple branch-wise interactions inside the Siamese-like backbone networks (InBN). The core of InBN is a general interaction modeler (GIM) that injects the target information to different stages of the backbone network, leading to better target-perception of candidate feature representation with negligible computation cost. The proposed GIM module and InBN mechanism are general and applicable to different backbone types including CNN and Transformer for improvements, as evidenced on multiple benchmarks. In particular, the CNN version improves the baseline with 3.2/6.9 absolute gains of SUC on LaSOT/TNL2K. The Transformer version obtains SUC of 65.7/52.0 on LaSOT/TNL2K, which are on par with recent SOTAs.

----

## [130] Exploring Fourier Prior for Single Image Rain Removal

**Authors**: *Xin Guo, Xueyang Fu, Man Zhou, Zhen Huang, Jialun Peng, Zheng-Jun Zha*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/131](https://doi.org/10.24963/ijcai.2022/131)

**Abstract**:

Deep convolutional neural networks (CNNs) have become dominant in the task of single image rain removal. Most of current CNN methods, however, suffer from the problem of overfitting on one single synthetic dataset as they neglect the intrinsic prior of the physical properties of rain streaks. To address this issue, we propose a simple but effective prior - Fourier prior to improve the generalization ability of an image rain removal model. The Fourier prior is a kind of property of rainy images. It is based on a key observation of us - replacing the Fourier amplitude of rainy images with that of clean images greatly suppresses the synthetic and real-world rain streaks. This means the amplitude contains most of the rain streak information and the phase keeps the similar structures of the background. So it is natural for single image rain removal to process the amplitude and phase information of the rainy images separately. In this paper, we develop a two-stage model where the first stage restores the amplitude of rainy images to clean rain streaks, and the second stage restores the phase information to refine fine-grained background structures. Extensive experiments on synthetic rainy data demonstrate the power of Fourier prior. Moreover, when trained on synthetic data, a robust generalization ability to real-world images can also be obtained. The code will be publicly available at https://github.com/willinglucky/ExploringFourier-Prior-for-Single-Image-Rain-Removal.

----

## [131] Rethinking Image Aesthetics Assessment: Models, Datasets and Benchmarks

**Authors**: *Shuai He, Yongchang Zhang, Rui Xie, Dongxiang Jiang, Anlong Ming*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/132](https://doi.org/10.24963/ijcai.2022/132)

**Abstract**:

Challenges in image aesthetics assessment (IAA) arise from that images of different themes correspond to different evaluation criteria, and learning aesthetics directly from images while ignoring the impact of theme variations on human visual perception inhibits the further development of IAA; however, existing IAA datasets and models overlook this problem. To address this issue, we show that a theme-oriented dataset and model design are effective for IAA. Specifically, 1) we elaborately build a novel dataset, called TAD66K, that contains 66K images covering 47 popular themes, and each image is densely annotated by more than 1200 people with dedicated theme evaluation criteria. 2) We develop a baseline model, TANet, which can effectively extract theme information and adaptively establish perception rules  to evaluate images with different themes. 3) We develop a large-scale benchmark (the most comprehensive thus far) by comparing 17 methods with TANet on three representative datasets: AVA, FLICKR-AES and the proposed TAD66K, TANet achieves state-of-the-art performance on all three datasets. Our work offers the community an opportunity to explore more challenging directions; the code, dataset and supplementary material are available at https://github.com/woshidandan/TANet.

----

## [132] Self-supervised Semantic Segmentation Grounded in Visual Concepts

**Authors**: *Wenbin He, William Surmeier, Arvind Kumar Shekar, Liang Gou, Liu Ren*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/133](https://doi.org/10.24963/ijcai.2022/133)

**Abstract**:

Unsupervised semantic segmentation requires assigning a label to every pixel without any human annotations.  Despite recent advances in self-supervised representation learning for individual images, unsupervised semantic segmentation with pixel-level representations is still a challenging task and remains underexplored.  In this work, we propose a self-supervised pixel representation learning method for semantic segmentation by using visual concepts (i.e., groups of pixels with semantic meanings, such as parts, objects, and scenes) extracted from images.  To guide self-supervised learning, we leverage three types of relationships between pixels and concepts, including the relationships between pixels and local concepts, local and global concepts, as well as the co-occurrence of concepts.  We evaluate the learned pixel embeddings and visual concepts on three datasets, including PASCAL VOC 2012, COCO 2017, and DAVIS 2017.  Our results show that the proposed method gains consistent and substantial improvements over recent unsupervised semantic segmentation approaches, and also demonstrate that visual concepts can reveal insights into image datasets.

----

## [133] Semantic Compression Embedding for Generative Zero-Shot Learning

**Authors**: *Ziming Hong, Shiming Chen, Guo-Sen Xie, Wenhan Yang, Jian Zhao, Yuanjie Shao, Qinmu Peng, Xinge You*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/134](https://doi.org/10.24963/ijcai.2022/134)

**Abstract**:

Generative methods have been successfully applied in zero-shot learning (ZSL) by learning an implicit mapping to alleviate the visual-semantic domain gaps and synthesizing unseen samples to handle the data imbalance between seen and unseen classes. However, existing generative methods simply use visual features extracted by the pre-trained CNN backbone. These visual features lack attribute-level semantic information. Consequently, seen classes are indistinguishable, and the knowledge transfer from seen to unseen classes is limited. To tackle this issue, we propose a novel Semantic Compression Embedding Guided Generation (SC-EGG) model, which cascades a semantic compression embedding network (SCEN) and an embedding guided generative network (EGGN). The SCEN extracts a group of attribute-level local features for each sample and further compresses them into the new low-dimension visual feature. Thus, a dense-semantic visual space is obtained. The EGGN learns a mapping from the class-level semantic space to the dense-semantic visual space, thus improving the discriminability of the synthesized dense-semantic unseen visual features. Extensive experiments on three benchmark datasets, i.e., CUB, SUN and AWA2, demonstrate the signiﬁcant performance gains of SC-EGG over current state-of-the-art methods and its baselines.

----

## [134] ScaleFormer: Revisiting the Transformer-based Backbones from a Scale-wise  Perspective for Medical Image Segmentation

**Authors**: *Huimin Huang, Shiao Xie, Lanfen Lin, Yutaro Iwamoto, Xian-Hua Han, Yen-Wei Chen, Ruofeng Tong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/135](https://doi.org/10.24963/ijcai.2022/135)

**Abstract**:

Recently, a variety of vision transformers have been developed as their capability of modeling long-range dependency. In current transformer-based backbones for medical image segmentation, convolutional layers were replaced with pure transformers, or transformers were added to the deepest encoder to learn global context. However, there are mainly two challenges in a scale-wise perspective: (1) intra-scale problem: the existing methods lacked in extracting local-global cues in each scale, which may impact the signal propagation of small objects; (2) inter-scale problem: the existing methods failed to explore distinctive information from multiple scales, which may hinder the representation learning from objects with widely variable size, shape and location. To address these limitations, we propose a novel backbone, namely ScaleFormer, with two appealing designs: (1) A scale-wise intra-scale transformer is designed to couple the CNN-based local features with the transformer-based global cues in each scale, where the row-wise and column-wise global dependencies can be extracted by a lightweight Dual-Axis MSA. (2) A simple and effective spatial-aware inter-scale transformer is designed to interact among consensual regions in multiple scales, which can highlight the cross-scale dependency and resolve the complex scale variations. Experimental results on different benchmarks demonstrate that our Scale-Former outperforms the current state-of-the-art methods. The code is publicly available at: https://github.com/ZJUGiveLab/ScaleFormer.

----

## [135] AQT: Adversarial Query Transformers for Domain Adaptive Object Detection

**Authors**: *Wei-Jie Huang, Yu-Lin Lu, Shih-Yao Lin, Yusheng Xie, Yen-Yu Lin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/136](https://doi.org/10.24963/ijcai.2022/136)

**Abstract**:

Adversarial feature alignment is widely used in domain adaptive object detection. Despite the effectiveness on CNN-based detectors, its applicability to transformer-based detectors is less studied. In this paper, we present AQT (adversarial query transformers) to integrate adversarial feature alignment into detection transformers. The generator is a detection transformer which yields a sequence of feature tokens, and the discriminator consists of a novel adversarial token and a stack of cross-attention layers. The cross-attention layers take the adversarial token as the query and the feature tokens from the generator as the key-value pairs. Through adversarial learning, the adversarial token in the discriminator attends to the domain-specific feature tokens, while the generator produces domain-invariant features, especially on the attended tokens, hence realizing adversarial feature alignment on transformers. Thorough experiments over several domain adaptive object detection benchmarks demonstrate that our approach performs favorably against the state-of-the-art methods. Source code is available at https://github.com/weii41392/AQT.

----

## [136] DANet: Image Deraining via Dynamic Association Learning

**Authors**: *Kui Jiang, Zhongyuan Wang, Zheng Wang, Peng Yi, Junjun Jiang, Jinsheng Xiao, Chia-Wen Lin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/137](https://doi.org/10.24963/ijcai.2022/137)

**Abstract**:

Rain streaks and background components in a rainy input are highly correlated, making the deraining task a composition of the rain streak removal and background restoration. However, the correlation of these two components is barely considered, leading to unsatisfied deraining results. To this end, we propose a dynamic associated network (DANet) to achieve the association learning between rain streak removal and background recovery. There are two key aspects to fulfill the association learning: 1) DANet unveils the latent association knowledge between rain streak prediction and background texture recovery, and leverages it as an extra prior via an associated learning module (ALM) to promote the texture recovery. 2) DANet introduces the parametric association constraint for enhancing the compatibility of deraining model with background reconstruction, enabling it to be automatically learned from the training data. Moreover, we observe that the sampled rainy image enjoys the similar distribution to the original one. We thus propose to learn the rain distribution at the sampling space, and exploit super-resolution to reconstruct high-frequency background details for computation and memory reduction. Our proposed DANet achieves the approximate deraining performance to the state-of-the-art MPRNet but only requires 52.6\% and 23\% inference time and computational cost, respectively.

----

## [137] SatFormer: Saliency-Guided Abnormality-Aware Transformer for Retinal Disease Classification in Fundus Image

**Authors**: *Yankai Jiang, Ke Xu, Xinyue Wang, Yuan Li, Hongguang Cui, Yubo Tao, Hai Lin*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/138](https://doi.org/10.24963/ijcai.2022/138)

**Abstract**:

Automatic and accurate retinal disease diagnosis is critical to guide proper therapy and prevent potential vision loss. Previous works simply exploit the most discriminative features while ignoring the pathological visual clues of scattered subtle lesions. Therefore, without a comprehensive understanding of features from different lesion regions, they are vulnerable to noise from complex backgrounds and suffer from misclassification failures. In this paper, we address these limitations with a novel saliency-guided abnormality-aware transformer which explicitly captures the correlation between different lesion features from a global perspective with enhanced pathological semantics. The model has several merits. First, we propose a saliency enhancement module (SEM) which adaptively integrates disease related semantics and highlights potentially salient lesion regions. Second, to the best of our knowledge, this is the first work to explore comprehensive lesion feature dependencies via a tailored efficient self-attention. Third, with the saliency enhancement module and abnormality-aware attention, we propose a new variant of Vision Transformer models, called SatFormer, which outperforms the state-of-the-art methods on two public retinal disease classification benchmarks. Ablation study shows that the proposed components can be easily embedded into any Vision Transformers via a plug-and-play manner and effectively boost the performance.

----

## [138] Domain Generalization through the Lens of Angular Invariance

**Authors**: *Yujie Jin, Xu Chu, Yasha Wang, Wenwu Zhu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/139](https://doi.org/10.24963/ijcai.2022/139)

**Abstract**:

Domain generalization (DG) aims at generalizing a classifier trained on multiple source domains to an unseen target domain with domain shift. A common pervasive theme in existing DG literature is domain-invariant representation learning with various invariance assumptions. However, prior works restrict themselves to an impractical assumption for real-world challenges: If a mapping induced by a deep neural network (DNN) could align the source domains well, then such a mapping aligns a target domain as well. In this paper, we simply take DNNs as feature extractors to relax the requirement of distribution alignment. Specifically, we put forward a novel angular invariance and the accompanied norm shift assumption. Based on the proposed term of invariance, we propose a novel deep DG method dubbed Angular Invariance Domain Generalization Network (AIDGN). The optimization objective of AIDGN is developed with a von-Mises Fisher (vMF) mixture model. Extensive experiments on multiple DG benchmark datasets validate the effectiveness of the proposed AIDGN method.

----

## [139] Online Hybrid Lightweight Representations Learning: Its Application to Visual Tracking

**Authors**: *Ilchae Jung, Minji Kim, Eunhyeok Park, Bohyung Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/140](https://doi.org/10.24963/ijcai.2022/140)

**Abstract**:

This paper presents a novel hybrid representation learning framework for streaming data, where an image frame in a video is modeled by an ensemble
of two distinct deep neural networks; one is a low-bit quantized network and the other is a lightweight full-precision network. The former learns coarse
primary information with low cost while the latter conveys residual information for high fidelity to original representations. The proposed parallel architecture is effective to maintain complementary information since fixed-point arithmetic can be utilized in the quantized network and the lightweight
model provides precise representations given by a compact channel-pruned network. We incorporate the hybrid representation technique into an online
visual tracking task, where deep neural networks need to handle temporal variations of target appearances in real-time. Compared to the state-of-the-art
real-time trackers based on conventional deep neural networks, our tracking algorithm demonstrates competitive accuracy on the standard benchmarks
with a small fraction of computational cost and memory footprint.

----

## [140] Robustifying Vision Transformer without Retraining from Scratch by Test-Time Class-Conditional Feature Alignment

**Authors**: *Takeshi Kojima, Yutaka Matsuo, Yusuke Iwasawa*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/141](https://doi.org/10.24963/ijcai.2022/141)

**Abstract**:

Vision Transformer (ViT) is becoming more popular in image processing. Specifically, we investigate the effectiveness of test-time adaptation (TTA) on ViT, a technique that has emerged to correct its prediction during test-time by itself. First, we benchmark various test-time adaptation approaches on ViT-B16 and ViT-L16. It is shown that the TTA is effective on ViT and the prior-convention (sensibly selecting modulation parameters) is not necessary when using proper loss function. Based on the observation, we propose a new test-time adaptation method called class-conditional feature alignment (CFA), which minimizes both the class-conditional distribution differences and the whole distribution differences of the hidden representation between the source and target in an online manner. Experiments of image classification tasks on common corruption (CIFAR-10-C, CIFAR-100-C, and ImageNet-C) and domain adaptation (digits datasets and ImageNet-Sketch) show that CFA stably outperforms the existing baselines on various datasets. We also verify that CFA is model agnostic by experimenting on ResNet, MLP-Mixer, and several ViT variants (ViT-AugReg, DeiT, and BeiT). Using BeiT backbone, CFA achieves 19.8% top-1 error rate on ImageNet-C, outperforming the existing test-time adaptation baseline 44.0%. This is a state-of-the-art result among TTA methods that do not need to alter training phase.

----

## [141] Attention-guided Contrastive Hashing for Long-tailed Image Retrieval

**Authors**: *Xuan Kou, Chenghao Xu, Xu Yang, Cheng Deng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/142](https://doi.org/10.24963/ijcai.2022/142)

**Abstract**:

Image hashing is to represent an image using a binary code for efficient storage and accurate retrieval. Recently, deep hashing methods have shown great improvements on ideally balanced datasets, however, long-tailed data is more common due to rare samples or data collection costs in the real world. Toward that end, this paper introduces a simple yet effective model named Attention-guided Contrastive Hashing Network (ACHNet) for long-tailed hashing. Specifically, a cross attention feature enhancement module is proposed to predict the importance of features for hashing, alleviating the loss of information originated from data dimension reduction. Moreover, unlike recently sota contrastive methods that focus on instance-level discrimination, we optimize an innovative category-centered contrastive hashing to obtain discriminative results, which is more suitable for long-tailed scenarios. Experiments on two popular benchmarks verify the superiority of the proposed method.  Our code is available at: https://github.com/KUXN98/ACHNet.

----

## [142] Beyond the Prototype: Divide-and-conquer Proxies for Few-shot Segmentation

**Authors**: *Chunbo Lang, Binfei Tu, Gong Cheng, Junwei Han*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/143](https://doi.org/10.24963/ijcai.2022/143)

**Abstract**:

Few-shot segmentation, which aims to segment unseen-class objects given only a handful of densely labeled samples, has received widespread attention from the community. Existing approaches typically follow the prototype learning paradigm to perform meta-inference, which fails to fully exploit the underlying information from support image-mask pairs, resulting in various segmentation failures, e.g., incomplete objects, ambiguous boundaries, and distractor activation. To this end, we propose a simple yet versatile framework in the spirit of divide-and-conquer. Specifically, a novel self-reasoning scheme is first implemented on the annotated support image, and then the coarse segmentation mask is divided into multiple regions with different properties. Leveraging effective masked average pooling operations, a series of support-induced proxies are thus derived, each playing a specific role in conquering the above challenges. Moreover, we devise a unique parallel decoder structure that integrates proxies with similar attributes to boost the discrimination power. Our proposed approach, named divide-and-conquer proxies (DCP), allows for the development of appropriate and reliable information as a guide at the “episode” level, not just about the object cues themselves. Extensive experiments on PASCAL-5i and COCO-20i demonstrate the superiority of DCP over conventional prototype-based approaches (up to 5~10% on average), which also establishes a new state-of-the-art. Code is available at github.com/chunbolang/DCP.

----

## [143] PlaceNet: Neural Spatial Representation Learning with Multimodal Attention

**Authors**: *Chung-Yeon Lee, Youngjae Yoo, Byoung-Tak Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/144](https://doi.org/10.24963/ijcai.2022/144)

**Abstract**:

Spatial representation capable of learning a myriad of environmental features is a significant challenge for natural spatial understanding of mobile AI agents.
Deep generative models have the potential of discovering rich representations of observed 3D scenes.
However, previous approaches have been mainly evaluated on simple environments, or focused only on high-resolution rendering of small-scale scenes, hampering generalization of the representations to various spatial variability.
To address this, we present PlaceNet, a neural representation that learns through random observations in a self-supervised manner, and represents observed scenes with triplet attention using visual, topographic, and semantic cues.
We evaluate the proposed method on a large-scale multimodal scene dataset consisting of 120 million indoor scenes, and show that PlaceNet successfully generalizes to various environments with lower training loss, higher image quality and structural similarity of predicted scenes, compared to a competitive baseline model.
Additionally, analyses of the representations demonstrate that PlaceNet activates more specialized and larger numbers of kernels in the spatial representation, capturing multimodal spatial properties in complex environments.

----

## [144] What is Right for Me is Not Yet Right for You: A Dataset for Grounding Relative Directions via Multi-Task Learning

**Authors**: *Jae Hee Lee, Matthias Kerzel, Kyra Ahrens, Cornelius Weber, Stefan Wermter*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/145](https://doi.org/10.24963/ijcai.2022/145)

**Abstract**:

Understanding spatial relations is essential for intelligent agents to act and communicate in the physical world. Relative directions are spatial relations that describe the relative positions of target objects with regard to the intrinsic orientation of reference objects. Grounding relative directions is more difficult than grounding absolute directions because it not only requires a model to detect objects in the image and to identify spatial relation based on this information, but it also needs to recognize the orientation of objects and integrate this information into the reasoning process. We investigate the challenging problem of grounding relative directions with end-to-end neural networks. To this end, we provide GRiD-3D, a novel dataset that features relative directions and complements existing visual question answering (VQA) datasets, such as CLEVR, that involve only absolute directions. We also provide baselines for the dataset with two established end-to-end VQA models. Experimental evaluations show that answering questions on relative directions is feasible when questions in the dataset simulate the necessary subtasks for grounding relative directions. We discover that those subtasks are learned in an order that reflects the steps of an intuitive pipeline for processing relative directions.

----

## [145] Learning to Assemble Geometric Shapes

**Authors**: *Jinhwi Lee, Jungtaek Kim, Hyunsoo Chung, Jaesik Park, Minsu Cho*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/146](https://doi.org/10.24963/ijcai.2022/146)

**Abstract**:

Assembling parts into an object is a combinatorial problem that arises in a variety of contexts in the real world and involves numerous applications in science and engineering. Previous related work tackles limited cases with identical unit parts or jigsaw-style parts of textured shapes, which greatly mitigate combinatorial challenges of the problem. In this work, we introduce the more challenging problem of shape assembly, which involves textureless fragments of arbitrary shapes with indistinctive junctions, and then propose a learning-based approach to solving it. We demonstrate the effectiveness on shape assembly tasks with various scenarios, including the ones with abnormal fragments (e.g., missing and distorted), the different number of fragments, and different rotation discretization.

----

## [146] Iterative Geometry-Aware Cross Guidance Network for Stereo Image Inpainting

**Authors**: *Ang Li, Shanshan Zhao, Qingjie Zhang, Qiuhong Ke*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/147](https://doi.org/10.24963/ijcai.2022/147)

**Abstract**:

Currently,  single  image  inpainting  has  achieved promising results based on deep convolutional neural  networks.   However,  inpainting  on  stereo  images  with  missing  regions  has  not  been  explored thoroughly, which is also a significant but different problem. One crucial requirement for stereo image inpainting is stereo consistency.  To achieve it, we propose an Iterative Geometry-Aware Cross Guidance Network (IGGNet). The IGGNet contains two key ingredients, i.e., a Geometry-Aware Attention(GAA) module and an Iterative  Cross Guidance(ICG) strategy. The GAA module relies on the epipolar geometry  cues and learns the geometry-aware guidance from one view to another,  which is beneficial to make the corresponding regions in two views consistent. However, learning guidance from co-existing missing regions is challenging. To address  this  issue,  the  ICG  strategy  is  proposed, which can alternately narrow down the missing regions of the two views in an iterative manner. Experimental  results demonstrate that our proposed network outperforms the latest stereo  image inpainting model and state-of-the-art single image inpainting models.

----

## [147] Representation Learning for Compressed Video Action Recognition via Attentive Cross-modal Interaction with Motion Enhancement

**Authors**: *Bing Li, Jiaxin Chen, Dongming Zhang, Xiuguo Bao, Di Huang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/148](https://doi.org/10.24963/ijcai.2022/148)

**Abstract**:

Compressed video action recognition has recently drawn growing attention, since it remarkably reduces the storage and computational cost via replacing raw videos by sparsely sampled RGB frames and compressed motion cues (e.g., motion vectors and residuals). However, this task severely suffers from the coarse and noisy dynamics and the insufficient fusion of the heterogeneous RGB and motion modalities. To address the two issues above, this paper proposes a novel framework, namely Attentive Cross-modal Interaction Network with Motion Enhancement (MEACI-Net). It follows the two-stream architecture, i.e. one for the RGB modality and the other for the motion modality. Particularly, the motion stream employs a multi-scale block embedded with a denoising module to enhance representation learning. The interaction between the two streams is then strengthened by introducing the Selective Motion Complement (SMC) and Cross-Modality Augment (CMA) modules, where SMC complements the RGB modality with spatio-temporally attentive local motion features and CMA further combines the two modalities with selective feature augmentation. Extensive experiments on the UCF-101, HMDB-51 and Kinetics-400 benchmarks demonstrate the effectiveness and efficiency of MEACI-Net.

----

## [148] Self-Guided Hard Negative Generation for Unsupervised Person Re-Identification

**Authors**: *Dongdong Li, Zhigang Wang, Jian Wang, Xinyu Zhang, Errui Ding, Jingdong Wang, Zhaoxiang Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/149](https://doi.org/10.24963/ijcai.2022/149)

**Abstract**:

Recent unsupervised person re-identification (reID) methods mostly apply pseudo labels from clustering algorithms as supervision signals. Despite great success, this fashion is very likely to aggregate different identities with similar appearances into the same cluster. In result, the hard negative samples, playing important role in training reID models, are significantly reduced. To alleviate this problem, we propose a self-guided hard negative generation method for unsupervised person re-ID. Specifically, a joint framework is developed which incorporates a hard negative generation network (HNGN) and a re-ID network. To continuously generate harder negative samples to provide effective supervisions in the contrastive learning, the two networks are alternately trained in an adversarial manner to improve each other, where the reID network guides HNGN to generate challenging data and HNGN enforces the re-ID network to enhance discrimination ability. During inference, the performance of re-ID network is improved without introducing any extra parameters. Extensive experiments demonstrate that the proposed method significantly outperforms a strong baseline and also achieves better results than state-of-the-art methods.

----

## [149] MMNet: Muscle Motion-Guided Network for Micro-Expression Recognition

**Authors**: *Hanting Li, Mingzhe Sui, Zhaoqing Zhu, Feng Zhao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/150](https://doi.org/10.24963/ijcai.2022/150)

**Abstract**:

Facial micro-expressions (MEs) are involuntary facial motions revealing people’s real feelings and play an important role in the early intervention of mental illness, the national security, and many human-computer interaction systems. However, existing micro-expression datasets are limited and usually pose some challenges for training good classifiers. To model the subtle facial muscle motions, we propose a robust micro-expression recognition (MER) framework, namely muscle motion-guided network (MMNet). Specifically,  a continuous attention (CA) block is introduced to focus on modeling local subtle muscle motion patterns with little identity information, which is different from most previous methods that directly extract features from complete video frames with much identity information. Besides, we design a position calibration (PC) module based on the vision transformer. By adding the position embeddings of the face generated by the PC module at the end of the two branches, the PC module can help to add position information to facial muscle motion-pattern features for the MER. Extensive experiments on three public micro-expression datasets demonstrate that our approach outperforms state-of-the-art methods by a large margin. Code is available at https://github.com/muse1998/MMNet.

----

## [150] ER-SAN: Enhanced-Adaptive Relation Self-Attention Network for Image Captioning

**Authors**: *Jingyu Li, Zhendong Mao, Shancheng Fang, Hao Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/151](https://doi.org/10.24963/ijcai.2022/151)

**Abstract**:

Image captioning (IC), bringing vision to language, has drawn extensive attention. Precisely describing visual relations between image objects is a key challenge in IC. We argue that the visual relations, that is geometric positions (i.e., distance and size) and semantic interactions (i.e., actions and possessives), indicate the mutual correlations between objects. Existing Transformer-based methods typically resort to geometric positions to enhance the representation of visual relations, yet only using the shallow geometric is unable to precisely cover the complex and actional correlations. In this paper, we propose to enhance the correlations between objects from a comprehensive view that jointly considers explicit semantic and geometric relations, generating plausible captions with accurate relationship predictions. Specifically, we propose a novel Enhanced-Adaptive Relation Self-Attention Network (ER-SAN). We design the direction-sensitive semantic-enhanced attention, which considers content objects to semantic relations and semantic relations to content objects attention to learn explicit semantic-aware relations. Further, we devise an adaptive re-weight relation module that determines how much semantic and geometric attention should be activated to each relation feature. Extensive experiments on MS-COCO dataset demonstrate the effectiveness of our ER-SAN,  with improvements of CIDEr from 128.6% to 135.3%, achieving state-of-the-art performance. Codes will be released \url{https://github.com/CrossmodalGroup/ER-SAN}.

----

## [151] RePFormer: Refinement Pyramid Transformer for Robust Facial Landmark Detection

**Authors**: *Jinpeng Li, Haibo Jin, Shengcai Liao, Ling Shao, Pheng-Ann Heng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/152](https://doi.org/10.24963/ijcai.2022/152)

**Abstract**:

This paper presents a Refinement Pyramid Transformer (RePFormer) for robust facial landmark detection. Most facial landmark detectors focus on learning  representative image features. However, these CNN-based feature representations are not robust enough to handle complex real-world scenarios due to ignoring the internal structure of landmarks, as well as the relations between landmarks and context. In this work, we formulate the facial landmark detection task as refining landmark queries along pyramid memories. Specifically, a pyramid transformer head (PTH) is introduced to build both homologous relations among landmarks and heterologous relations between landmarks and cross-scale contexts. Besides, a dynamic landmark refinement (DLR) module is designed to decompose the landmark regression into an end-to-end refinement procedure, where the dynamically aggregated queries are transformed to residual coordinates predictions. Extensive experimental results on four facial landmark detection benchmarks and their various subsets demonstrate the superior performance and high robustness of our framework.

----

## [152] Dite-HRNet: Dynamic Lightweight High-Resolution Network for Human Pose Estimation

**Authors**: *Qun Li, Ziyi Zhang, Fu Xiao, Feng Zhang, Bir Bhanu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/153](https://doi.org/10.24963/ijcai.2022/153)

**Abstract**:

A high-resolution network exhibits remarkable capability in extracting multi-scale features for human pose estimation, but fails to capture long-range interactions between joints and has high computational complexity. To address these problems, we present a Dynamic lightweight High-Resolution Network (Dite-HRNet), which can efficiently extract multi-scale contextual information and model long-range spatial dependency for human pose estimation. Specifically, we propose two methods, dynamic split convolution and adaptive context modeling, and embed them into two novel lightweight blocks, which are named dynamic multi-scale context block and dynamic global context block. These two blocks, as the basic component units of our Dite-HRNet, are specially designed for the high-resolution networks to make full use of the parallel multi-resolution architecture. Experimental results show that the proposed network achieves superior performance on both COCO and MPII human pose estimation datasets, surpassing the state-of-the-art lightweight networks. Code is available at: https://github.com/ZiyiZhang27/Dite-HRNet.

----

## [153] Learning Graph-based Residual Aggregation Network for Group Activity Recognition

**Authors**: *Wei Li, Tianzhao Yang, Xiao Wu, Zhaoquan Yuan*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/154](https://doi.org/10.24963/ijcai.2022/154)

**Abstract**:

Group activity recognition aims to understand the overall behavior performed by a group of people. Recently, some graph-based methods have made progress by learning the relation graphs among multiple persons. However, the differences between an individual and others play an important role in identifying confusable group activities, which have not been elaborately explored by previous methods. In this paper, a novel Graph-based Residual AggregatIon Network (GRAIN) is proposed to model the differences among all persons of the whole group, which is end-to-end trainable. Specifically, a new local residual relation module is explicitly proposed to capture the local spatiotemporal differences of relevant persons, which is further combined with the multi-graph relation networks. Moreover, a weighted aggregation strategy is devised to adaptively select multi-level spatiotemporal features from the appearance-level information to high level relations. Finally, our model is capable of extracting a comprehensive representation and inferring the group activity in an end-to-end manner. The experimental results on two popular benchmarks for group activity recognition clearly demonstrate the superior performance of our method in comparison with the state-of-the-art methods.

----

## [154] TCCNet: Temporally Consistent Context-Free Network for Semi-supervised Video Polyp Segmentation

**Authors**: *Xiaotong Li, Jilan Xu, Yuejie Zhang, Rui Feng, Rui-Wei Zhao, Tao Zhang, Xuequan Lu, Shang Gao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/155](https://doi.org/10.24963/ijcai.2022/155)

**Abstract**:

Automatic video polyp segmentation (VPS) is highly valued for the early diagnosis of colorectal cancer. However, existing methods are limited in three respects: 1) most of them work on static images, while ignoring the temporal information in consecutive video frames; 2) all of them are fully supervised and easily overfit in presence of limited annotations; 3) the context of polyp (i.e., lumen, specularity and mucosa tissue) varies in an endoscopic clip, which may affect the predictions of adjacent frames. To resolve these challenges, we propose a novel Temporally Consistent Context-Free Network (TCCNet) for semi-supervised VPS. It contains a segmentation branch and a propagation branch with a co-training scheme to supervise the predictions of unlabeled image. To maintain the temporal consistency of predictions, we design a Sequence-Corrected Reverse Attention module and a Propagation-Corrected Reverse Attention module. A Context-Free Loss is also proposed to mitigate the impact of varying contexts. Extensive experiments show that even trained under 1/15 label ratio, TCCNet is comparable to the state-of-the-art fully supervised methods for VPS. Also, TCCNet surpasses existing semi-supervised methods for natural image and other medical image segmentation tasks.

----

## [155] PRNet: Point-Range Fusion Network for Real-Time LiDAR Semantic Segmentation

**Authors**: *Xiaoyan Li, Gang Zhang, Tao Jiang, Xufen Cai, Zhenhua Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/156](https://doi.org/10.24963/ijcai.2022/156)

**Abstract**:

Accurate and real-time LiDAR semantic segmentation is necessary for advanced autonomous driving systems. To guarantee a fast inference speed, previous methods utilize the highly optimized 2D convolutions to extract features on the range view (RV), which is the most compact representation of the LiDAR point clouds. However, these methods often suffer from lower accuracy for two reasons: 1) the information loss during the projection from 3D points to the RV, 2) the semantic ambiguity when 3D points labels are assigned according to the RV predictions. In this work, we introduce an end-to-end point-range fusion network (PRNet) that extracts semantic features mainly on the RV and iteratively fuses the RV features back to the 3D points for the final prediction. Besides, a novel range view projection (RVP) operation is designed to alleviate the information loss during the projection to the RV, and a point-range convolution (PRConv) is proposed to automatically mitigate the semantic ambiguity during transmitting features from the RV back to 3D points. Experiments on the SemanticKITTI and nuScenes benchmarks demonstrate that the PRNet pushes the range-based methods to a new state-of-the-art, and achieves a better speed-accuracy trade-off.

----

## [156] Unsupervised Embedding and Association Network for Multi-Object Tracking

**Authors**: *Yu-Lei Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/157](https://doi.org/10.24963/ijcai.2022/157)

**Abstract**:

How to generate robust trajectories of multiple objects without using any manual identity annotation? Recently, identity embedding features from Re-ID models are adopted to associate targets into trajectories.
However, most previous methods equipped with embedding features heavily rely on manual identity annotations, which bring a high cost for the multi-object tracking (MOT) task.
To address the above problem, we present an unsupervised embedding and association network (UEANet) for learning discriminative embedding features with pseudo identity labels. 
Specifically, we firstly generate the pseudo identity labels by adopting a Kalman filter tracker to associate multiple targets into trajectories and assign a unique identity label to each trajectory. 
Secondly, we train the transformer-based identity embedding branch and MLP-based data association branch of UEANet with these pseudo labels, and UEANet extracts branch-dependent features for the unsupervised MOT task.
Experimental results show that UEANet confirms the outstanding ability to suppress IDS and achieves comparable performance compared with state-of-the-art methods on three MOT datasets.

----

## [157] Multi-View Visual Semantic Embedding

**Authors**: *Zheng Li, Caili Guo, Zerun Feng, Jenq-Neng Hwang, Xijun Xue*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/158](https://doi.org/10.24963/ijcai.2022/158)

**Abstract**:

Visual Semantic Embedding (VSE) is a dominant method for cross-modal vision-language retrieval. Its purpose is to learn an embedding space so that visual data can be embedded in a position close to the corresponding text description. However, there are large intra-class variations in the vision-language data. For example, multiple texts describing the same image may be described from different views, and the descriptions of different views are often dissimilar. The mainstream VSE method embeds samples from the same class in similar positions, which will suppress intra-class variations and lead to inferior generalization performance. This paper proposes a Multi-View Visual Semantic Embedding (MV-VSE) framework, which learns multiple embeddings for one visual data and explicitly models intra-class variations. To optimize MV-VSE, a multi-view upper bound loss is proposed, and the multi-view embeddings are jointly optimized while retaining intra-class variations. MV-VSE is plug-and-play and can be applied to various VSE models and loss functions without excessively increasing model complexity. Experimental results on the Flickr30K and MS-COCO datasets demonstrate the superior performance of our framework.

----

## [158] Self-supervised Learning and Adaptation for Single Image Dehazing

**Authors**: *Yudong Liang, Bin Wang, Wangmeng Zuo, Jiaying Liu, Wenqi Ren*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/159](https://doi.org/10.24963/ijcai.2022/159)

**Abstract**:

Existing deep image dehazing methods usually depend on supervised learning with a large number of hazy-clean image pairs which are expensive or difficult to collect. Moreover, dehazing performance of the learned model may deteriorate significantly when the training hazy-clean image pairs are insufficient and are different from real hazy images in applications. In this paper, we show that exploiting large scale training set and adapting to real hazy images are two critical issues in learning effective deep dehazing models. Under the depth guidance estimated by a well-trained depth estimation network, we leverage the conventional atmospheric scattering model to generate massive hazy-clean image pairs for the self-supervised pre-training of dehazing network. Furthermore, self-supervised adaptation is presented to adapt pre-trained network to real hazy images. Learning without forgetting strategy is also deployed in self-supervised adaptation by combining self-supervision and model adaptation via contrastive learning. Experiments show that our proposed method performs favorably against the state-of-the-art methods, and is quite efficient, i.e., handling a 4K image in 23 ms. The codes are available at https://github.com/DongLiangSXU/SLAdehazing.

----

## [159] Feature Dense Relevance Network for Single Image Dehazing

**Authors**: *Yun Liang, Enze Huang, Zifeng Zhang, Zhuo Su, Dong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/160](https://doi.org/10.24963/ijcai.2022/160)

**Abstract**:

Existing learning-based dehazing methods do not fully use non-local information, which makes the restoration of seriously degraded region very tough. We propose a novel dehazing network by defining the Feature Dense Relevance module (FDR) and the Shallow Feature Mapping module (SFM). The FDR is defined based on multi-head attention to construct the dense relationship between different local features in the whole image. It enables the network to restore the degraded local regions by non-local information in complex scenes. In addition, the raw distant skip-connection easily leads to artifacts while it cannot deal with the shallow features effectively. Therefore, we define the SFM by combining the atmospheric scattering model and the distant skip-connection to effectively deal with the shallow features in different scales. It not only maps the degraded textures into clear textures by distant dependence, but also 
reduces artifacts and color distortions effectively. We introduce contrastive loss and focal frequency loss in the network to obtain a realitic and clear image. The extensive experiments on several synthetic and real-world datasets demonstrate that our network surpasses most of the state-of-the-art methods.

----

## [160] RMGN: A Regional Mask Guided Network for Parser-free Virtual Try-on

**Authors**: *Chao Lin, Zhao Li, Sheng Zhou, Shichang Hu, Jialun Zhang, Linhao Luo, Jiarun Zhang, Longtao Huang, Yuan He*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/161](https://doi.org/10.24963/ijcai.2022/161)

**Abstract**:

Virtual try-on (VTON) aims at fitting target clothes to reference person images, which is widely adopted in e-commerce. Existing VTON approaches can be narrowly categorized into Parser-Based (PB) and Parser-Free (PF) by whether relying on the parser information to mask the persons’clothes and synthesize try-on images. Although abandoning parser information has improved the applicability of PF methods, the ability of detail synthesizing has also been sacrificed. As a result, the distraction from original cloth may persist in synthesized images, especially in complicated postures and high resolution applications. To address the aforementioned issue, we propose a novel PF method named Regional Mask Guided Network (RMGN). More specifically, a regional mask is proposed to explicitly fuse the features of target clothes and reference persons so that the persisted distraction can be eliminated. A posture awareness loss and a multi-level feature extractor are further proposed to handle the complicated postures and synthesize high resolution images. Extensive experiments demonstrate that our proposed RMGN outperforms both state-of-the-art PB and PF methods. Ablation studies further verify the effectiveness of modules in RMGN. Code is available at https://github.com/jokerlc/RMGN-VITON.

----

## [161] Learning to Estimate Object Poses without Real Image Annotations

**Authors**: *Haotong Lin, Sida Peng, Zhize Zhou, Xiaowei Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/162](https://doi.org/10.24963/ijcai.2022/162)

**Abstract**:

This paper presents a simple yet effective approach for learning 6DoF object poses without real image annotations. Previous methods have attempted to train pose estimators on synthetic data, but they do not generalize well to real images due to the sim-to-real domain gap and produce inaccurate pose estimates. We find that, in most cases, the synthetically trained pose estimators are able to provide reasonable initialization for depth-based pose refinement methods which yield accurate pose estimates. Motivated by this, we propose a novel learning framework, which utilizes the accurate results of depth-based pose refinement methods to supervise the RGB-based pose estimator. Our method significantly outperforms previous self-supervised methods on several benchmarks. Even compared with fully-supervised methods that use real annotated data, we achieve competitive results without using any real annotation. The code is available at https://github.com/zju3dv/pvnet-depth-sup.

----

## [162] Intrinsic Image Decomposition by Pursuing Reflectance Image

**Authors**: *Tzu-Heng Lin, Pengxiao Wang, Yizhou Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/163](https://doi.org/10.24963/ijcai.2022/163)

**Abstract**:

Intrinsic image decomposition is a fundamental problem for many computer vision applications. While recent deep learning based methods have achieved very promising results on the synthetic densely labeled datasets, the results on the real-world dataset are still far from human level performance. This is mostly because collecting dense supervision on a real-world dataset is impossible. Only a sparse set of pairwise judgement from human is often used. It's very difficult for models to learn in such settings. 
In this paper, we investigate the possibilities of only using reflectance images for supervision during training. In this way, the demand for labeled data is greatly reduced. In order to achieve this goal, we take a deep investigation into the reflectance images. We find that reflectance images are actually comprised of two components: the flat surfaces with low frequency information, and the boundaries with high frequency details. Then, we propose to disentangle the learning process of the two components of the reflectance images. We argue that through this procedure, the reflectance images can be better modeled, and in the meantime, the shading images, though not supervised, can also achieve decent result. Extensive experiments show that our proposed network outperforms current state-of-the-art results by a large margin on the most challenging real-world IIW dataset. We also surprisingly find that on the densely labeled datasets (MIT and MPI-Sintel), our network can also achieve state-of-the-art results on both reflectance and shading images, when we only apply supervision on the reflectance images during training.

----

## [163] FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer

**Authors**: *Yang Lin, Tianyu Zhang, Peiqin Sun, Zheng Li, Shuchang Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/164](https://doi.org/10.24963/ijcai.2022/164)

**Abstract**:

Network quantization significantly reduces model inference complexity and has been widely used in real-world deployments. However, most existing quantization methods have been developed mainly on Convolutional Neural Networks (CNNs), and suffer severe degradation when applied to fully quantized vision transformers. In this work, we demonstrate that many of these difficulties arise because of serious inter-channel variation in LayerNorm inputs, and present, Power-of-Two Factor (PTF), a systematic method to reduce the performance degradation and inference complexity of fully quantized vision transformers. In addition, observing an extreme non-uniform distribution in attention maps, we propose Log-Int-Softmax (LIS) to sustain that and simplify inference by using 4-bit quantization and the BitShift operator. Comprehensive experiments on various transformer-based architectures and benchmarks show that our Fully Quantized Vision Transformer (FQ-ViT) outperforms previous works while even using lower bit-width on attention maps. For instance, we reach 84.89% top-1 accuracy with ViT-L on ImageNet and 50.8 mAP with Cascade Mask R-CNN (Swin-S) on COCO. To our knowledge, we are the first to achieve lossless accuracy degradation (~1%) on fully quantized vision transformers. The code is available at https://github.com/megvii-research/FQ-ViT.

----

## [164] MA-ViT: Modality-Agnostic Vision Transformers for Face Anti-Spoofing

**Authors**: *Ajian Liu, Yanyan Liang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/165](https://doi.org/10.24963/ijcai.2022/165)

**Abstract**:

The existing multi-modal face anti-spoofing (FAS) frameworks are designed based on two strategies: halfway and late fusion. However, the former requires test modalities consistent with the training input, which seriously limits its deployment scenarios. And the latter is built on multiple branches to process different modalities independently, which limits their use in applications with low memory or fast execution requirements. In this work, we present a single branch based Transformer framework, namely Modality-Agnostic Vision Transformer (MA-ViT), which aims to improve the performance of arbitrary modal attacks with the help of multi-modal data. Specifically, MA-ViT adopts the early fusion to aggregate all the available training modalitiesâ€™ data and enables flexible testing of any given modal samples. Further, we develop the Modality-Agnostic Transformer Block (MATB) in MA-ViT, which consists of two stacked attentions named Modal-Disentangle Attention (MDA) and Cross-Modal Attention (CMA), to eliminate modality-related information for each modal sequences and supplement modality-agnostic liveness features from another modal sequences, respectively. Experiments demonstrate that the single model trained based on MA-ViT can not only flexibly evaluate different modal samples, but also outperforms existing single-modal frameworks by a large margin, and approaches the multi-modal frameworks introduced with smaller FLOPs and model parameters.

----

## [165] Dynamic Group Transformer: A General Vision Transformer Backbone with Dynamic Group Attention

**Authors**: *Kai Liu, Tianyi Wu, Cong Liu, Guodong Guo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/166](https://doi.org/10.24963/ijcai.2022/166)

**Abstract**:

Recently, Transformers have shown promising performance in various vision tasks. To reduce the quadratic computation complexity caused by each query attending to all keys/values, various methods have constrained the range of attention  within local regions, where each query only attends to keys/values within a hand-crafted window. However, these hand-crafted window partition mechanisms are data-agnostic and ignore their input content, so it is likely that one query maybe attend to irrelevant keys/values. To address this issue, we propose a Dynamic Group Attention (DG-Attention), which dynamically divides all queries into multiple groups and selects the most relevant keys/values for each group.  Our DG-Attention can flexibly model more relevant dependencies without any spatial constraint that is used in hand-crafted window based attention. Built on the DG-Attention, we develop a general vision transformer backbone named Dynamic Group Transformer (DGT). Extensive experiments show that our models can outperform the state-of-the-art methods on multiple common vision tasks, including image classification, semantic segmentation, object detection, and instance segmentation.

----

## [166] Cost Ensemble with Gradient Selecting for GANs

**Authors**: *Minghui Liu, Jiali Deng, Meiyi Yang, Xuan Cheng, Nianbo Liu, Ming Liu, Xiaomin Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/167](https://doi.org/10.24963/ijcai.2022/167)

**Abstract**:

Generative Adversarial Networks(GANs) are powerful generative models on numerous tasks and datasets but are also known for their training instability and mode collapse. The latter is because the optimal transportation map is discontinuous, but DNNs can only approximate continuous ones. One way to solve the problem is to introduce multiple discriminators or generators. However, their impacts are limited because the cost function of each component is the same. That is, they are homogeneous. In contrast, multiple discriminators with different cost functions can yield various gradients for the generator, which indicates we can use them to search for more transportation maps in the latent space. Inspired by this, we have proposed a framework to combat the mode collapse problem, containing multiple discriminators with different cost functions, named CES-GAN. Unfortunately, it may also lead to the generator being hard to train because the performance between discriminators is unbalanced, according to the Cannikin Law. Thus, a gradient selecting mechanism is also proposed to pick up proper gradients. We provide mathematical statements to prove our assumptions and conduct extensive experiments to verify the performance. The results show that CES-GAN is lightweight and more effective for fighting against the mode collapse problem than similar works.

----

## [167] TopoSeg: Topology-aware Segmentation for Point Clouds

**Authors**: *Weiquan Liu, Hanyun Guo, Weini Zhang, Yu Zang, Cheng Wang, Jonathan Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/168](https://doi.org/10.24963/ijcai.2022/168)

**Abstract**:

Point cloud segmentation plays an important role in AI applications such as autonomous driving, AR, and VR. However, previous point cloud segmentation neural networks rarely pay attention to the topological correctness of the segmentation results. In this paper, focusing on the perspective of topology awareness. First, to optimize the distribution of segmented predictions from the perspective of topology, we introduce the persistent homology theory in topology into a 3D point cloud deep learning framework. Second, we propose a topology-aware 3D point cloud segmentation module, TopoSeg. Specifically, we design a topological loss function embedded in TopoSeg module, which imposes topological constraints on the segmentation of 3D point clouds. Experiments show that our proposed TopoSeg module can be easily embedded into the point cloud segmentation network and improve the segmentation performance. In addition, based on the constructed topology loss function, we propose a topology-aware point cloud edge extraction algorithm, which is demonstrated that has strong robustness.

----

## [168] Biological Instance Segmentation with a Superpixel-Guided Graph

**Authors**: *Xiaoyu Liu, Wei Huang, Yueyi Zhang, Zhiwei Xiong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/169](https://doi.org/10.24963/ijcai.2022/169)

**Abstract**:

Recent advanced proposal-free instance segmentation methods have made significant progress in biological images. However, existing methods are vulnerable to local imaging artifacts and similar object appearances, resulting in over-merge and over-segmentation. To reduce these two kinds of errors, we propose a new biological instance segmentation framework based on a superpixel-guided graph, which consists of two stages, i.e., superpixel-guided graph construction and superpixel agglomeration. Specifically, the first stage generates enough superpixels as graph nodes to avoid over-merge, and extracts node and edge features to construct an initialized graph. The second stage agglomerates superpixels into instances based on the relationship of graph nodes predicted by a graph neural network (GNN). To solve over-segmentation and prevent introducing additional over-merge, we specially design two loss functions to supervise the GNN, i.e., a repulsion-attraction (RA) loss to better distinguish the relationship of nodes in the feature space, and a maximin agglomeration score (MAS) loss to pay more attention to crucial edge classification. Extensive experiments on three representative biological datasets demonstrate the superiority of our method over existing state-of-the-art methods. Code is available at https://github.com/liuxy1103/BISSG.

----

## [169] Vision Shared and Representation Isolated Network for Person Search

**Authors**: *Yang Liu, Yingping Li, Chengyu Kong, Yuqiu Kong, Shenglan Liu, Feilong Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/170](https://doi.org/10.24963/ijcai.2022/170)

**Abstract**:

Person search is a widely-concerned computer vision task that aims to jointly solve the problems of pedestrian detection and person re-identification in panoramic scenes. However, the pedestrian detection focuses on the consistency of pedestrians, while the person re-identification attempts to extract the discriminative features of pedestrians. The inevitable conflict greatly restricts the researches on the one-stage person search methods. To address this issue, we propose a Vision Shared and Representation Isolated (VSRI) network to decouple the two conflicted subtasks simultaneously, through which two independent representations are constructed for the two subtasks. To enhance the discrimination of the re-ID representation, a Multi-Level Feature Fusion (MLFF) module is proposed. The MLFF adopts the Spatial Pyramid Feature Fusion (SPFF) module to obtain diverse features from the stem network. Moreover, the multi-head self-attention mechanism is employed to construct a Multi-head Attention Driven Extraction (MADE) module and the cascaded convolution unit is adopted to devise a Feature Decomposition and Cascaded Integration  (FDCI) module, which facilitates the MLFF to obtain more discriminative representations of the pedestrians. The proposed method outperforms the state-of-the-art methods on the mainstream datasets.

----

## [170] Copy Motion From One to Another: Fake Motion Video Generation

**Authors**: *Zhenguang Liu, Sifan Wu, Chejian Xu, Xiang Wang, Lei Zhu, Shuang Wu, Fuli Feng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/171](https://doi.org/10.24963/ijcai.2022/171)

**Abstract**:

One compelling application of artificial intelligence is to generate a video of a target person performing arbitrary desired motion (from a source person). While the state-of-the-art methods are able to synthesize a video demonstrating similar broad stroke motion details, they are generally lacking in texture details. A pertinent manifestation appears as distorted face, feet, and hands, and such
flaws are very sensitively perceived by human observers. Furthermore, current methods typically employ GANs with a L2 loss to assess the authenticity of the generated videos, inherently requiring a large amount of training samples to learn the texture details for adequate video generation. In this work, we tackle these challenges from three aspects: 1) We disentangle each video frame into
foreground (the person) and background, focusing on generating the foreground to reduce the underlying dimension of the network output. 2) We propose a theoretically motivated Gromov-Wasserstein loss that facilitates learning the mapping from a pose to a foreground image. 3) To enhance texture details, we encode facial features with geometric guidance and employ local GANs to refine the face, feet, and hands. Extensive experiments show that our method is able to generate realistic target person videos, faithfully copying complex motions from a source person. Our code and datasets are released at https://github.com/Sifann/FakeMotion.

----

## [171] Deep Video Harmonization With Color Mapping Consistency

**Authors**: *Xinyuan Lu, Shengyuan Huang, Li Niu, Wenyan Cong, Liqing Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/172](https://doi.org/10.24963/ijcai.2022/172)

**Abstract**:

Video harmonization aims to adjust the foreground of a composite video to make it compatible with the background. So far, video harmonization has only received limited attention and there is no public dataset for video harmonization. In this work, we construct a new video harmonization dataset HYouTube by adjusting the foreground of real videos to create synthetic composite videos. Moreover, we consider the temporal consistency in video harmonization task. Unlike previous works which establish the spatial correspondence, we design a novel framework based on the assumption of color mapping consistency, which leverages the color mapping of neighboring frames to refine the current frame. Extensive experiments on our HYouTube dataset prove the effectiveness of our proposed framework. Our dataset and code are available at https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube.

----

## [172] Learning Multi-dimensional Edge Feature-based AU Relation Graph for Facial Action Unit Recognition

**Authors**: *Cheng Luo, Siyang Song, Weicheng Xie, Linlin Shen, Hatice Gunes*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/173](https://doi.org/10.24963/ijcai.2022/173)

**Abstract**:

The activations of Facial Action Units (AUs) mutually influence one another. While the relationship between a pair of AUs can be complex and unique, existing approaches fail to specifically and explicitly represent such cues for each pair of AUs in each facial display. This paper proposes an AU relationship modelling approach that deep learns a unique graph to explicitly describe the relationship between each pair of AUs of the target facial display. Our approach first encodes each AU's activation status and its association with other AUs into a node feature. Then, it learns a pair of multi-dimensional edge features to describe multiple task-specific relationship cues between each pair of AUs. During both node and edge feature learning, our approach also considers the influence of the unique facial display on AUs' relationship by taking the full face representation as an input. Experimental results on BP4D and DISFA datasets show that both node and edge feature learning modules provide large performance improvements for CNN and transformer-based backbones, with our best systems achieving the state-of-the-art AU recognition results. Our approach not only has a strong capability in modelling relationship cues for AU recognition but also can be easily incorporated into various backbones. Our PyTorch code is made available at https://github.com/CVI-SZU/ME-GraphAU.

----

## [173] Long-Short Term Cross-Transformer in Compressed Domain for Few-Shot Video Classification

**Authors**: *Wenyang Luo, Yufan Liu, Bing Li, Weiming Hu, Yanan Miao, Yangxi Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/174](https://doi.org/10.24963/ijcai.2022/174)

**Abstract**:

Compared with image few-shot learning, most of the existing few-shot video classification methods perform worse on feature matching, because they fail to sufficiently exploit the temporal information and relation. Specifically, frames are usually evenly sampled, which may miss important frames. On the other hand, the heuristic model simply encodes the equally treated frames in sequence, which results in the lack of both long-term and short-term temporal modeling and interaction. To alleviate these limitations, we take advantage of the compressed domain knowledge and propose a long-short term Cross-Transformer (LSTC) for few-shot video classification. For short terms, the motion vector (MV) contains temporal cues and reflects the importance of each frame. For long terms, a video can be natively divided into a sequence of GOPs (Group Of Picture). Using this compressed domain knowledge helps to obtain a more accurate spatial-temporal feature space. Consequently, we design the long-short term selection module, short-term module, and long-term module to comprise the LSTC. Long-short term selection is performed to select informative compressed domain data. Long/short-term modules are utilized to sufficiently exploit the temporal information so that the query and support can be well-matched by cross-attention. Experimental results show the superiority of our method on various datasets.

----

## [174] Improved Deep Unsupervised Hashing with Fine-grained Semantic Similarity Mining for Multi-Label Image Retrieval

**Authors**: *Zeyu Ma, Xiao Luo, Yingjie Chen, Mixiao Hou, Jinxing Li, Minghua Deng, Guangming Lu*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/175](https://doi.org/10.24963/ijcai.2022/175)

**Abstract**:

In this paper, we study deep unsupervised hashing, a critical problem for approximate nearest neighbor research. Most recent methods solve this problem by semantic similarity reconstruction for guiding hashing network learning or contrastive learning of hash codes. However, in multi-label scenarios, these methods usually either generate an inaccurate similarity matrix without reflection of similarity ranking or suffer from the violation of the underlying assumption in contrastive learning, resulting in limited retrieval performance. To tackle this issue, we propose a novel method termed HAMAN, which explores semantics from a fine-grained view to enhance the ability of multi-label image retrieval. In particular, we reconstruct the pairwise similarity structure by matching fine-grained patch features generated by the pre-trained neural network, serving as reliable guidance for similarity preserving of hash codes. Moreover, a novel conditional contrastive learning on hash codes is proposed to adopt self-supervised learning in multi-label scenarios. According to extensive experiments on three multi-label datasets, the proposed method outperforms a broad range of state-of-the-art methods.

----

## [175] Learning Degradation Uncertainty for Unsupervised Real-world Image Super-resolution

**Authors**: *Qian Ning, Jingzhu Tang, Fangfang Wu, Weisheng Dong, Xin Li, Guangming Shi*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/176](https://doi.org/10.24963/ijcai.2022/176)

**Abstract**:

Acquiring degraded images with paired high-resolution (HR) images is often challenging, impeding the advance of image super-resolution in real-world applications. By generating realistic low-resolution (LR) images with degradation similar to that in real-world scenarios, simulated paired LR-HR data can be constructed for supervised training. However, most of the existing work ignores the degradation uncertainty of the generated realistic LR images, since only one LR image has been generated given an HR image. To address this weakness, we propose learning the degradation uncertainty of generated LR images and sampling multiple LR images from the learned LR image (mean) and degradation uncertainty (variance) and construct LR-HR pairs to train the super-resolution (SR) networks. Specifically, uncertainty can be learned by minimizing the proposed loss based on Kullback-Leibler (KL) divergence. Furthermore, the uncertainty in the feature domain is exploited by a novel perceptual loss; and we propose to calculate the adversarial loss from the gradient information in the SR stage for stable training performance and better visual quality. Experimental results on popular real-world datasets show that our proposed method has performed better than other unsupervised approaches.

----

## [176] Continual Semantic Segmentation Leveraging Image-level Labels and Rehearsal

**Authors**: *Mathieu Pagé Fortin, Brahim Chaib-draa*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/177](https://doi.org/10.24963/ijcai.2022/177)

**Abstract**:

Despite the remarkable progress of deep learning models for semantic segmentation, the success of these models is strongly limited by the following aspects: 1) large datasets with pixel-level annotations must be available and 2) training must be performed with all classes simultaneously. Indeed, in incremental learning scenarios, where new classes are added to an existing framework, these models are prone to catastrophic forgetting of previous classes. To address these two limitations, we propose a weakly-supervised mechanism for continual semantic segmentation that can leverage cheap image-level annotations and a novel rehearsal strategy that intertwines the learning of past and new classes. Specifically, we explore two rehearsal technique variants: 1) imprinting past objects on new images and 2) transferring past representations in intermediate features maps. We conduct extensive experiments on Pascal-VOC by varying the proportion of fully- and weakly-supervised data in various setups and show that our contributions consistently improve the mIoU on both past and novel classes. Interestingly, we also observe that models trained with less data in incremental steps sometimes outperform the same architectures trained with more data. We discuss the significance of these results and propose some hypotheses regarding the dynamics between forgetting and learning.

----

## [177] Multilevel Hierarchical Network with Multiscale Sampling for Video Question Answering

**Authors**: *Min Peng, Chongyang Wang, Yuan Gao, Yu Shi, Xiang-Dong Zhou*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/178](https://doi.org/10.24963/ijcai.2022/178)

**Abstract**:

Video question answering (VideoQA) is challenging given its multimodal combination of visual understanding and natural language processing. While most existing approaches ignore the visual appearance-motion information at different temporal scales, it is unknown how to incorporate the multilevel processing capacity of a deep learning model with such multiscale information. Targeting these issues, this paper proposes a novel Multilevel Hierarchical Network (MHN) with multiscale sampling for VideoQA. MHN comprises two modules, namely Recurrent Multimodal Interaction (RMI) and Parallel Visual Reasoning (PVR). With a multiscale sampling, RMI iterates the interaction of appearance-motion information at each scale and the question embeddings to build the multilevel question-guided visual representations. Thereon, with a shared transformer encoder, PVR infers the visual cues at each level in parallel to fit with answering different question types that may rely on the visual information at relevant levels. Through extensive experiments on three VideoQA datasets, we demonstrate improved performances than previous state-of-the-arts and justify the effectiveness of each part of our method.

----

## [178] Source-Adaptive Discriminative Kernels based Network for Remote Sensing Pansharpening

**Authors**: *Siran Peng, Liang-Jian Deng, Jin-Fan Hu, Yu-Wei Zhuo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/179](https://doi.org/10.24963/ijcai.2022/179)

**Abstract**:

For the pansharpening problem, previous convolutional neural networks (CNNs) mainly concatenate high-resolution panchromatic (PAN) images and low-resolution multispectral (LR-MS) images in their architectures, which ignores the distinctive attributes of different sources. In this paper, we propose a convolution network with source-adaptive discriminative kernels, called ADKNet, for the pansharpening task. Those kernels consist of spatial kernels generated from PAN images containing rich spatial details and spectral kernels generated from LR-MS images containing abundant spectral information. The kernel generating process is specially designed to extract information discriminately and effectively. Furthermore, the kernels are learned in a pixel-by-pixel manner to characterize different information in distinct areas. Extensive experimental results indicate that ADKNet outperforms current state-of-the-art (SOTA) pansharpening methods in both quantitative and qualitative assessments, in the meanwhile only with about 60,000 network parameters. Also, the proposed network is extended to the hyperspectral image super-resolution (HSISR) problem, still yields SOTA performance, proving the universality of our model. The code is available at http://github.com/liangjiandeng/ADKNet.

----

## [179] SimMC: Simple Masked Contrastive Learning of Skeleton Representations for Unsupervised Person Re-Identification

**Authors**: *Haocong Rao, Chunyan Miao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/180](https://doi.org/10.24963/ijcai.2022/180)

**Abstract**:

Recent advances in skeleton-based person re-identification (re-ID) obtain impressive performance via either hand-crafted skeleton descriptors or skeleton representation learning with deep learning paradigms. However, they typically require skeletal pre-modeling and label information for training, which leads to limited applicability of these methods. In this paper, we focus on unsupervised skeleton-based person re-ID, and present a generic Simple Masked Contrastive learning (SimMC) framework to learn effective representations from unlabeled 3D skeletons for person re-ID. Specifically, to fully exploit skeleton features within each skeleton sequence, we first devise a masked prototype contrastive learning (MPC) scheme to cluster the most typical skeleton features (skeleton prototypes) from different subsequences randomly masked from raw sequences, and contrast the inherent similarity between skeleton features and different prototypes to learn discriminative skeleton representations without using any label. Then, considering that different subsequences within the same sequence usually enjoy strong correlations due to the nature of motion continuity, we propose the masked intra-sequence contrastive learning (MIC) to capture intra-sequence pattern consistency between subsequences, so as to encourage learning more effective skeleton representations for person re-ID. Extensive experiments validate that the proposed SimMC outperforms most state-of-the-art skeleton-based methods. We further show its scalability and efficiency in enhancing the performance of existing models. Our codes are available at https://github.com/Kali-Hac/SimMC.

----

## [180] ChimeraMix: Image Classification on Small Datasets via Masked Feature Mixing

**Authors**: *Christoph Reinders, Frederik Schubert, Bodo Rosenhahn*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/181](https://doi.org/10.24963/ijcai.2022/181)

**Abstract**:

Deep convolutional neural networks require large amounts of labeled data samples. For many real-world applications, this is a major limitation which is commonly treated by augmentation methods. In this work, we address the problem of learning deep neural networks on small datasets. Our proposed architecture called ChimeraMix learns a data augmentation by generating compositions of instances. The generative model encodes images in pairs, combines the features guided by a mask, and creates new samples. For evaluation, all methods are trained from scratch without any additional data. Several experiments on benchmark datasets, e.g. ciFAIR-10, STL-10, and ciFAIR-100, demonstrate the superior performance of ChimeraMix compared to current state-of-the-art methods for classification on small datasets. Code is available at https://github.com/creinders/ChimeraMix.

----

## [181] IDPT: Interconnected Dual Pyramid Transformer for Face Super-Resolution

**Authors**: *Jingang Shi, Yusi Wang, Songlin Dong, Xiaopeng Hong, Zitong Yu, Fei Wang, Changxin Wang, Yihong Gong*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/182](https://doi.org/10.24963/ijcai.2022/182)

**Abstract**:

Face Super-resolution (FSR) task works for generating high-resolution (HR) face images from the corresponding low-resolution (LR) inputs, which has received a lot of attentions because of the wide application prospects. However, due to the diversity of facial texture and the difficulty of reconstructing detailed content from degraded images, FSR technology is still far away from being solved. In this paper, we propose a novel and effective face super-resolution framework based on Transformer, namely Interconnected Dual Pyramid Transformer (IDPT). Instead of straightly stacking cascaded feature reconstruction blocks, the proposed IDPT designs the pyramid encoder/decoder Transformer architecture to extract coarse and detailed facial textures respectively, while the relationship between the dual pyramid Transformers is further explored by a bottom pyramid feature extractor. The pyramid encoder/decoder structure is devised to adapt various characteristics of textures in different spatial spaces hierarchically. A novel fusing modulation module is inserted in each spatial layer to guide the refinement of detailed texture by the corresponding coarse texture, while fusing the shallow-layer coarse feature and corresponding deep-layer detailed feature simultaneously. Extensive experiments and visualizations on various datasets demonstrate the superiority of the proposed method for face super-resolution tasks.

----

## [182] A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space

**Authors**: *Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/183](https://doi.org/10.24963/ijcai.2022/183)

**Abstract**:

The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

----

## [183] Emotion-Controllable Generalized Talking Face Generation

**Authors**: *Sanjana Sinha, Sandika Biswas, Ravindra Yadav, Brojeshwar Bhowmick*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/184](https://doi.org/10.24963/ijcai.2022/184)

**Abstract**:

Despite the significant progress in recent years, very few of the AI-based talking face generation methods attempt to render natural emotions. Moreover, the scope of the methods is majorly limited to the characteristics of the training dataset, hence they fail to generalize to arbitrary unseen faces. In this paper, we propose a one-shot facial geometry-aware emotional talking face generation method that can generalize to arbitrary faces. We propose a graph convolutional neural network that uses speech content feature, along with an independent emotion input to generate emotion and speech-induced motion on facial geometry-aware landmark representation.  This representation is further used in our optical flow-guided texture generation network for producing the texture. We propose a two-branch texture generation network, with motion and texture branches designed to consider the motion and texture content independently. Compared to the previous emotion talking face methods, our method can adapt to arbitrary faces captured in-the-wild by fine-tuning with only a single image of the target identity in neutral emotion.

----

## [184] Harnessing Fourier Isovists and Geodesic Interaction for Long-Term Crowd Flow Prediction

**Authors**: *Samuel S. Sohn, Seonghyeon Moon, Honglu Zhou, Mihee Lee, Sejong Yoon, Vladimir Pavlovic, Mubbasir Kapadia*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/185](https://doi.org/10.24963/ijcai.2022/185)

**Abstract**:

With the rise in popularity of short-term Human Trajectory Prediction (HTP), Long-Term Crowd Flow Prediction (LTCFP) has been proposed to forecast crowd movement in large and complex environments. However, the input representations, models, and datasets for LTCFP are currently limited. To this end, we propose Fourier Isovists, a novel input representation based on egocentric visibility, which consistently improves all existing models. We also propose GeoInteractNet (GINet), which couples the layers between a multi-scale attention network (M-SCAN) and a convolutional encoder-decoder network (CED). M-SCAN approximates a super-resolution map of where humans are likely to interact on the way to their goals and produces multi-scale attention maps. The CED then uses these maps in either its encoder's inputs or its decoder's attention gates, which allows GINet to produce super-resolution predictions with substantially higher accuracy than existing models even with Fourier Isovists. In order to evaluate the scalability of models to large and complex environments, which the only existing LTCFP dataset is unsuitable for, a new synthetic crowd dataset with both real and synthetic environments has been generated. In its nascent state, LTCFP has much to gain from our key contributions. The Supplementary Materials, dataset, and code are available at sssohn.github.io/GeoInteractNet.

----

## [185] Boundary-Guided Camouflaged Object Detection

**Authors**: *Yujia Sun, Shuo Wang, Chenglizhao Chen, Tian-Zhu Xiang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/186](https://doi.org/10.24963/ijcai.2022/186)

**Abstract**:

Camouflaged object detection (COD), segmenting objects that are elegantly blended into their surroundings, is a valuable yet challenging task. Existing deep-learning methods often fall into the difficulty of accurately identifying the camouflaged object with complete and fine object structure. To this end, in this paper, we propose a novel boundary-guided network (BGNet) for camouflaged object detection. Our method explores valuable and extra object-related edge semantics to guide representation learning of COD, which forces the model to generate features that highlight object structure, thereby promoting camouflaged object detection of accurate boundary localization. Extensive experiments on three challenging benchmark datasets demonstrate that our BGNet significantly outperforms the existing 18 state-of-the-art methods under four widely-used evaluation metrics. Our code is publicly available at: https://github.com/thograce/BGNet.

----

## [186] Dynamic Domain Generalization

**Authors**: *Zhishu Sun, Zhifeng Shen, Luojun Lin, Yuanlong Yu, Zhifeng Yang, Shicai Yang, Weijie Chen*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/187](https://doi.org/10.24963/ijcai.2022/187)

**Abstract**:

Domain generalization (DG) is a fundamental yet very challenging research topic in machine learning. The existing arts mainly focus on learning domain-invariant features with limited source domains in a static model. Unfortunately, there is a lack of training-free mechanism to adjust the model when generalized to the agnostic target domains. To tackle this problem, we develop a brand-new DG variant, namely Dynamic Domain Generalization (DDG), in which the model learns to twist the network parameters to adapt to the data from different domains. Specifically, we leverage a meta-adjuster to twist the network parameters based on the static model with respect to different data from different domains. In this way, the static model is optimized to learn domain-shared features, while the meta-adjuster is designed to learn domain-specific features. To enable this process, DomainMix is exploited to simulate data from diverse domains during teaching the meta-adjuster to adapt to the agnostic target domains. This learning mechanism urges the model to generalize to different agnostic target domains via adjusting the model without training. Extensive experiments demonstrate the effectiveness of our proposed method. Code is available: https://github.com/MetaVisionLab/DDG

----

## [187] Video Frame Interpolation Based on Deformable Kernel Region

**Authors**: *Haoyue Tian, Pan Gao, Xiaojiang Peng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/188](https://doi.org/10.24963/ijcai.2022/188)

**Abstract**:

Video frame interpolation task has recently become more and more prevalent in the computer vision field. At present, a number of researches based on deep learning have achieved great success. Most of them are either based on optical flow information, or interpolation kernel, or a combination of these two methods. However, these methods have ignored that there are grid restrictions on the position of kernel region during synthesizing each target pixel. These limitations result in that they cannot well adapt to the irregularity of object shape and uncertainty of motion, which may lead to irrelevant reference pixels used for interpolation. In order to solve this problem, we revisit the deformable convolution for video interpolation, which can break the fixed grid restrictions on the kernel region, making the distribution of reference points more suitable for the shape of the object, and thus warp a more accurate interpolation frame. Experiments are conducted on four datasets to demonstrate the superior performance of the proposed model in comparison to the state-of-the-art alternatives.

----

## [188] Hypertron: Explicit Social-Temporal Hypergraph Framework for Multi-Agent Forecasting

**Authors**: *Yu Tian, Xingliang Huang, Ruigang Niu, Hongfeng Yu, Peijin Wang, Xian Sun*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/189](https://doi.org/10.24963/ijcai.2022/189)

**Abstract**:

Forecasting the future trajectories of multiple agents is a core technology for human-robot interaction systems. To predict multi-agent trajectories more accurately, it is inevitable that models need to improve interpretability and reduce redundancy. However, many methods adopt implicit weight calculation or black-box networks to learn the semantic interaction of agents, which obviously lack enough interpretation. In addition, most of the existing works model the relation among all agents in a one-to-one manner, which might lead to irrational trajectory predictions due to its redundancy and noise. To address the above issues, we present Hypertron, a human-understandable and lightweight hypergraph-based multi-agent forecasting framework, to explicitly estimate the motions of multiple agents and generate reasonable trajectories. The framework explicitly interacts among multiple agents and learns their latent intentions by our coarse-to-fine hypergraph convolution interaction module. Our experiments on several challenging real-world trajectory forecasting datasets show that Hypertron outperforms a wide array of state-of-the-art methods while saving over 60% parameters and reducing 30% inference time.

----

## [189] Automatic Recognition of Emotional Subgroups in Images

**Authors**: *Emmeke Veltmeijer, Charlotte Gerritsen, Koen V. Hindriks*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/190](https://doi.org/10.24963/ijcai.2022/190)

**Abstract**:

Both social group detection and group emotion recognition in images are growing fields of interest, but never before have they been combined. In this work we aim to detect emotional subgroups in images, which can be of great importance for crowd surveillance or event analysis. To this end, human annotators are instructed to label a set of 171 images, and their recognition strategies are analysed. Three main strategies for labeling images are identified, with each strategy assigning either 1) more weight to emotions (emotion-based fusion), 2) more weight to spatial structures (group-based fusion), or 3) equal weight to both (summation strategy). Based on these strategies, algorithms are developed to automatically recognize emotional subgroups. In particular, K-means and hierarchical clustering are used with location and emotion features derived from a fine-tuned VGG network. Additionally, we experiment with face size and gaze direction as extra input features. The best performance comes from hierarchical clustering with emotion, location and gaze direction as input.

----

## [190] Augmenting Anchors by the Detector Itself

**Authors**: *Xiaopei Wan, Guoqiu Li, Yujiu Yang, Zhenhua Guo*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/191](https://doi.org/10.24963/ijcai.2022/191)

**Abstract**:

Usually, it is difficult to determine the scale and aspect ratio of anchors for anchor-based object detection methods. Current state-of-the-art object detectors either determine anchor parameters according to objects' shape and scale in a dataset, or avoid this problem by utilizing anchor-free methods, however, the former scheme is dataset-specific and the latter methods could not get better performance than the former ones. In this paper, we propose a novel anchor augmentation method named AADI, which means Augmenting Anchors by the Detector Itself. AADI is not an anchor-free method, instead, it can convert the scale and aspect ratio of anchors from a continuous space to a discrete space, which greatly alleviates the problem of anchors' designation. Furthermore, AADI is a learning-based anchor augmentation method, but it does not add any parameters or hyper-parameters, which is beneficial for research and downstream tasks. Extensive experiments on COCO dataset demonstrate the effectiveness of AADI, specifically, AADI achieves significant performance boosts on many state-of-the-art object detectors (eg. at least +2.4 box AP on Faster R-CNN, +2.2 box AP on Mask R-CNN, and +0.9 box AP on Cascade Mask R-CNN). We hope that this simple and cost-efficient method can be widely used in object detection. Code and models are available at https://github.com/WanXiaopei/aadi.

----

## [191] Absolute Wrong Makes Better: Boosting Weakly Supervised Object Detection via Negative Deterministic Information

**Authors**: *Guanchun Wang, Xiangrong Zhang, Zelin Peng, Xu Tang, Huiyu Zhou, Licheng Jiao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/192](https://doi.org/10.24963/ijcai.2022/192)

**Abstract**:

Weakly supervised object detection (WSOD) is a challenging task, in which image-level labels (e.g., categories of the instances in the whole image) are used to train an object detector. Many existing methods follow the standard multiple instance learning (MIL) paradigm and have achieved promising performance. However, the lack of deterministic information leads to part domination and missing instances. To address these issues, this paper focuses on identifying and fully exploiting the deterministic information in WSOD. We discover that negative instances (i.e. absolutely wrong instances), ignored in most of the previous studies, normally contain valuable deterministic information. Based on this observation, we here propose a negative deterministic information (NDI) based method for improving WSOD, namely NDI-WSOD. Specifically, our method consists of two stages: NDI collecting and exploiting. In the collecting stage, we design several processes to identify and distill the NDI from negative instances online. In the exploiting stage, we utilize the extracted NDI to construct a novel negative contrastive learning mechanism and a negative guided instance selection strategy for dealing with the issues of part domination and missing instances, respectively. Experimental results on several public benchmarks including VOC 2007, VOC 2012 and MS COCO show that our method achieves satisfactory performance.

----

## [192] Iterative Few-shot Semantic Segmentation from Image Label Text

**Authors**: *Haohan Wang, Liang Liu, Wuhao Zhang, Jiangning Zhang, Zhenye Gan, Yabiao Wang, Chengjie Wang, Haoqian Wang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/193](https://doi.org/10.24963/ijcai.2022/193)

**Abstract**:

Few-shot semantic segmentation aims to learn to segment unseen class objects with the guidance of only a few support images. Most previous methods rely on the pixel-level label of support images. In this paper, we focus on a more challenging setting, in which only the image-level labels are available. We propose a general framework to firstly generate coarse masks with the help of the powerful vision-language model CLIP, and then iteratively and mutually refine the mask predictions of support and query images. Extensive experiments on PASCAL-5i and COCO-20i datasets demonstrate that our method not only outperforms the state-of-the-art weakly supervised approaches by a significant margin, but also achieves comparable or better results to recent supervised methods. Moreover, our method owns an excellent generalization ability for the images in the wild and uncommon classes. Code will be available at https://github.com/Whileherham/IMR-HSNet.

----

## [193] Spatiality-guided Transformer for 3D Dense Captioning on Point Clouds

**Authors**: *Heng Wang, Chaoyi Zhang, Jianhui Yu, Weidong Cai*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/194](https://doi.org/10.24963/ijcai.2022/194)

**Abstract**:

Dense captioning in 3D point clouds is an emerging vision-and-language task involving object-level 3D scene understanding. Apart from coarse semantic class prediction and bounding box regression as in traditional 3D object detection, 3D dense captioning aims at producing a further and finer instance-level label of natural language description on visual appearance and spatial relations for each scene object of interest. To detect and describe objects in a scene, following the spirit of neural machine translation, we propose a transformer-based encoder-decoder architecture, namely SpaCap3D, to transform objects into descriptions, where we especially investigate the relative spatiality of objects in 3D scenes and design a spatiality-guided encoder via a token-to-token spatial relation learning objective and an object-centric decoder for precise and spatiality-enhanced object caption generation. Evaluated on two benchmark datasets, ScanRefer and ReferIt3D, our proposed SpaCap3D outperforms the baseline method Scan2Cap by 4.94% and 9.61% in CIDEr@0.5IoU, respectively. Our project page with source code and supplementary files is available at https://SpaCap3D.github.io/.

----

## [194] Adaptive Convolutional Dictionary Network for CT Metal Artifact Reduction

**Authors**: *Hong Wang, Yuexiang Li, Deyu Meng, Yefeng Zheng*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/195](https://doi.org/10.24963/ijcai.2022/195)

**Abstract**:

Inspired by the great success of deep neural networks, learning-based methods have gained promising performances for metal artifact reduction (MAR) in computed tomography (CT) images. However, most of the existing approaches put less emphasis on modelling and embedding the intrinsic prior knowledge underlying this specific MAR task into their network designs. Against this issue, we propose an adaptive convolutional dictionary network (ACDNet), which leverages both model-based and learning-based methods. Specifically, we explore the prior structures of metal artifacts, e.g., non-local repetitive streaking patterns, and encode them as an explicit weighted convolutional dictionary model. Then, a simple-yet-effective algorithm is carefully designed to solve the model. By unfolding every iterative substep of the proposed algorithm into a network module, we explicitly embed the prior structure into a deep network , i.e., a clear interpretability for the MAR task. Furthermore, our ACDNet can automatically learn the prior for artifact-free CT images via training data and adaptively adjust the representation kernels for each input CT image based on its content. Hence, our method inherits the clear interpretability of model-based methods and maintains the powerful representation ability of learning-based methods. Comprehensive experiments executed on synthetic and clinical datasets show the superiority of our ACDNet in terms of effectiveness and model generalization. Code and supplementary material are available at https://github.com/hongwang01/ACDNet.

----

## [195] KUNet: Imaging Knowledge-Inspired Single HDR Image Reconstruction

**Authors**: *Hu Wang, Mao Ye, Xiatian Zhu, Shuai Li, Ce Zhu, Xue Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/196](https://doi.org/10.24963/ijcai.2022/196)

**Abstract**:

Recently, with the rise of high dynamic range (HDR) display devices, there is a great demand to transfer traditional low dynamic range (LDR) images into HDR versions. The key to success is how to solve the many-to-many mapping problem. However, the existing approaches either do not consider constraining solution space or just simply imitate the inverse camera imaging pipeline in stages, without directly formulating the HDR image generation process.  In this work, we address this problem by integrating LDR-to-HDR imaging knowledge into an UNet architecture, dubbed as Knowledge-inspired UNet (KUNet). The conversion from LDR-to-HDR image is mathematically formulated, and can be conceptually divided into recovering missing details, adjusting imaging parameters and reducing imaging noise. Accordingly, we develop a basic knowledge-inspired block (KIB) including three subnetworks corresponding to the three procedures in this HDR imaging process. The KIB blocks are cascaded in the similar way to the UNet to construct HDR image with rich global information. In addition, we also propose a knowledge inspired jump-connect structure to fit a dynamic range gap between HDR and LDR images. Experimental results demonstrate that the proposed KUNet achieves superior performance compared with the state-of-the-art methods. The code, dataset and appendix materials are available at https://github.com/wanghu178/KUNet.git.

----

## [196] I2CNet: An Intra- and Inter-Class Context Information Fusion Network for Blastocyst Segmentation

**Authors**: *Hua Wang, Linwei Qiu, Jingfei Hu, Jicong Zhang*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/197](https://doi.org/10.24963/ijcai.2022/197)

**Abstract**:

The quality of a blastocyst directly determines the embryo's implantation potential, thus making it essential to objectively and accurately identify the blastocyst morphology. In this work, we propose an automatic framework named I2CNet to perform the blastocyst segmentation task in human embryo images. The I2CNet contains two components: IntrA-Class Context Module (IACCM) and InteR-Class Context Module (IRCCM). The IACCM aggregates the representations of specific areas sharing the same category for each pixel, where the categorized regions are learned under the supervision of the groundtruth. This aggregation decomposes a K-category recognition task into K recognition tasks of two labels while maintaining the ability of garnering intra-class features. In addition, the IRCCM is designed based on the blastocyst morphology to compensate for inter-class information which is gradually gathered from inside out. Meanwhile, a weighted mapping function is applied to facilitate edges of the inter classes and stimulate some hard samples. Eventually, the learned intra- and inter-class cues are integrated from coarse to fine, rendering sufficient information interaction and fusion between multi-scale features. Quantitative and qualitative experiments demonstrate that the superiority of our model compared with other representative methods. The I2CNet achieves accuracy of 94.14% and Jaccard of 85.25% on blastocyst public dataset.

----

## [197] PACE: Predictive and Contrastive Embedding for Unsupervised Action Segmentation

**Authors**: *Jiahao Wang, Jie Qin, Yunhong Wang, Annan Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/198](https://doi.org/10.24963/ijcai.2022/198)

**Abstract**:

Action segmentation, inferring temporal positions of human actions in an untrimmed video, is an important prerequisite for various video understanding tasks. Recently, unsupervised action segmentation (UAS) has emerged as a more challenging task due to the unavailability of frame-level annotations. Existing clustering- or prediction-based UAS approaches suffer from either over-segmentation or overfitting, leading to unsatisfactory results. To address those problems,we propose Predictive And Contrastive Embedding (PACE), a unified UAS framework leveraging both predictability and similarity information for more accurate action segmentation. On the basis of an auto-regressive transformer encoder, predictive embeddings are learned by exploiting the predictability of video context, while contrastive embeddings are generated by leveraging the similarity of adjacent short video clips. Extensive experiments on three challenging benchmarks demonstrate the superiority of our method, with up to 26.9% improvements in F1-score over the state of the art.

----

## [198] Double-Check Soft Teacher for Semi-Supervised Object Detection

**Authors**: *Kuo Wang, Yuxiang Nie, Chaowei Fang, Chengzhi Han, Xuewen Wu, Xiaohui Wang, Liang Lin, Fan Zhou, Guanbin Li*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/199](https://doi.org/10.24963/ijcai.2022/199)

**Abstract**:

In the semi-supervised object detection task, due to the scarcity of labeled data and the diversity and complexity of objects to be detected, the quality of pseudo-labels generated by existing methods for unlabeled data is relatively low, which severely restricts the performance of semi-supervised object detection. In this paper, we revisit the pseudo-labeling based Teacher-Student mutual learning framework for semi-supervised object detection and identify that the inconsistency of the location and feature of the candidate object proposals between the Teacher and the Student branches are the fatal cause of the low quality of the pseudo labels. To address this issue, we propose a simple yet effective technique within the mainstream teacher-student framework, called Double Check Soft Teacher, to overcome the harm caused by insufficient quality of pseudo labels. Specifically, our proposed method leverages teacher model to generate pseudo labels for the student model. Especially, the candidate boxes generated by the student model based on the pseudo label will be sent to the teacher model for "double check", and then the teacher model will output probabilistic soft label with background class for those candidate boxes, which will be used to train the student model. Together with a pseudo labeling mechanism based on the sum of the TOP-K prediction score, which improves the recall rate of pseudo labels, Double Check Soft Teacher consistently surpasses state-of-the-art methods by significant margins on the MS-COCO benchmark, pushing the new state-of-the-art.  Source codes are available at https://github.com/wkfdb/DCST.

----

## [199] RePre: Improving Self-Supervised Vision Transformer with Reconstructive Pre-training

**Authors**: *Luya Wang, Feng Liang, Yangguang Li, Honggang Zhang, Wanli Ouyang, Jing Shao*

**Conference**: *ijcai 2022*

**URL**: [https://doi.org/10.24963/ijcai.2022/200](https://doi.org/10.24963/ijcai.2022/200)

**Abstract**:

Recently, self-supervised vision transformers have attracted unprecedented attention for their impressive representation learning ability. 
However, the dominant method, contrastive learning, mainly relies on an instance discrimination pretext task, which learns a global understanding of the image. 
This paper incorporates local feature learning into self-supervised vision transformers via Reconstructive Pre-training (RePre).
Our RePre extends contrastive frameworks by adding a branch for reconstructing raw image pixels in parallel with the existing contrastive objective. 
RePre equips with a lightweight convolution-based decoder that fuses the multi-hierarchy features from the transformer encoder. 
The multi-hierarchy features provide rich supervisions from low to high semantic information, crucial for our RePre.
Our RePre brings decent improvements on various contrastive frameworks with different vision transformer architectures. 
Transfer performance in downstream tasks outperforms supervised pre-training and state-of-the-art (SOTA) self-supervised counterparts.

----



[Go to the next page](IJCAI-2022-list02.md)

[Go to the catalog section](README.md)