## [1800] End-to-End Pipeline for Trigger Detection on Hit and Track Graphs

**Authors**: *Tingting Xuan, Yimin Zhu, Giorgian Borca-Tasciuc, Ming Xiong Liu, Yu Sun, Cameron Dean, Yasser Corrales Morales, Zhaozhong Shi, Dantong Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26870](https://doi.org/10.1609/aaai.v37i13.26870)

**Abstract**:

There has been a surge of interest in applying deep learning in particle and nuclear physics to replace labor-intensive offline data analysis with automated online machine learning tasks. This paper details a novel AI-enabled triggering solution for physics experiments in Relativistic Heavy Ion Collider and future Electron-Ion Collider. The triggering system consists of a comprehensive end-to-end pipeline based on Graph Neural Networks that classifies trigger events versus background events, makes online decisions to retain signal data, and enables efficient data acquisition. The triggering system first starts with the coordinates of pixel hits lit up by passing particles in the detector, applies three stages of event processing (hits clustering, track reconstruction, and trigger detection), and labels all processed events with the binary tag of trigger versus background events. By switching among different objective functions, we train the Graph Neural Networks in the pipeline to solve multiple tasks: the edge-level track reconstruction problem, the edge-level track adjacency matrix prediction, and the graph-level trigger detection problem. We propose a novel method to treat the events as track-graphs instead of hit-graphs. This method focuses on intertrack relations and is driven by underlying physics processing. As a result, it attains a solid performance (around 72% accuracy) for trigger detection and outperforms the baseline method using hit-graphs by 2% higher accuracy.

----

## [1801] Xaitk-Saliency: An Open Source Explainable AI Toolkit for Saliency

**Authors**: *Brian Hu, Paul Tunison, Brandon Richard Webster, Anthony Hoogs*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26871](https://doi.org/10.1609/aaai.v37i13.26871)

**Abstract**:

Advances in artificial intelligence (AI) using techniques such as deep learning have fueled the recent progress in fields such as computer vision. However, these algorithms are still often viewed as "black boxes", which cannot easily explain how they arrived at their final output decisions. Saliency maps are one commonly used form of explainable AI (XAI), which indicate the input features an algorithm paid attention to during its decision process. Here, we introduce the open source xaitk-saliency package, an XAI framework and toolkit for saliency. We demonstrate its modular and flexible nature by highlighting two example use cases for saliency maps: (1) object detection model comparison and (2) doppelganger saliency for person re-identification. We also show how the xaitk-saliency package can be paired with visualization tools to support the interactive exploration of saliency maps. Our results suggest that saliency maps may play a critical role in the verification and validation of AI models, ensuring their trusted use and deployment. The code is publicly available at: https://github.com/xaitk/xaitk-saliency.

----

## [1802] DetAIL: A Tool to Automatically Detect and Analyze Drift in Language

**Authors**: *Nishtha Madaan, Adithya Manjunatha, Hrithik Nambiar, Aviral Kumar Goel, Harivansh Kumar, Diptikalyan Saha, Srikanta Bedathur*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26872](https://doi.org/10.1609/aaai.v37i13.26872)

**Abstract**:

Machine learning and deep learning-based decision making has become part of today's software. The goal of this work is to ensure that machine learning and deep learning-based systems are as trusted as traditional software. Traditional software is made dependable by following rigorous practice like static analysis, testing, debugging, verifying, and repairing throughout the development and maintenance life-cycle. Similarly for machine learning systems, we need to keep these models up to date so that their performance is not compromised. For this, current systems rely on scheduled re-training of these models as new data kicks in. In this work, we propose DetAIL, a tool to measure the data drift that takes place when new data kicks in so that one can adaptively re-train the models whenever re-training is actually required irrespective of schedules. In addition to that, we generate various explanations at sentence level and dataset level to capture why a given payload text has drifted.

----

## [1803] PARCS: A Deployment-Oriented AI System for Robust Parcel-Level Cropland Segmentation of Satellite Images

**Authors**: *Chen Du, Yiwei Wang, Zhicheng Yang, Hang Zhou, Mei Han, Jui-Hsin Lai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26873](https://doi.org/10.1609/aaai.v37i13.26873)

**Abstract**:

Cropland segmentation of satellite images is an essential basis for crop area and yield estimation tasks in the remote sensing and computer vision interdisciplinary community. Instead of common pixel-level segmentation results with salt-and-pepper effects, a parcel-level output conforming to human recognition is required according to the clients' needs during the model deployment. However, leveraging CNN-based models requires fine-grained parcel-level labels, which is an unacceptable annotation burden. To cure these practical pain points, in this paper, we present PARCS, a holistic deployment-oriented AI system for PARcel-level Cropland Segmentation. By consolidating multi-disciplinary knowledge, PARCS has two algorithm branches. The first branch performs pixel-level crop segmentation by learning from limited labeled pixel samples with an active learning strategy to avoid parcel-level annotation costs. The second branch aims at generating the parcel regions without a learning procedure. The final parcel-level segmentation result is achieved by integrating the outputs of these two branches in tandem. The robust effectiveness of PARCS is demonstrated by its outstanding performance on public and in-house datasets (an overall accuracy of 85.3% and an mIoU of 61.7% on the public PASTIS dataset, and an mIoU of 65.16% on the in-house dataset). We also include subjective feedback from clients and discuss the lessons learned from deployment.

----

## [1804] Adaptive Temporal Planning for Multi-Robot Systems in Operations and Maintenance of Offshore Wind Farms

**Authors**: *Ferdian Jovan, Sara Bernardini*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26874](https://doi.org/10.1609/aaai.v37i13.26874)

**Abstract**:

With the fast development of offshore wind farms as renewable energy sources, maintaining them efficiently and safely becomes necessary. The high costs of operation and maintenance (O&M) are due to the length of turbine downtime and the logistics for human technician transfer. To reduce such costs, we propose a comprehensive multi-robot system that includes unmanned aerial vehicles (UAV), autonomous surface vessels (ASV), and inspection-and-repair robots (IRR). Our system, which is capable of co-managing the farms with human operators located onshore, brings down costs and significantly reduces the Health and Safety (H&S) risks of O&M by assisting human operators in performing dangerous tasks. In this paper, we focus on using AI temporal planning to coordinate the actions of the different autonomous robots that form the multi-robot system. We devise a new, adaptive planning approach that reduces failures and replanning by performing data-driven goal and domain refinement. Our experiments in both simulated and real-world scenarios prove the effectiveness and robustness of our technique. The success of our system marks the first-step towards a large-scale, multirobot solution for wind farm O&M.

----

## [1805] A Study of Students' Learning of Computing through an LP-Based Integrated Curriculum for Middle Schools

**Authors**: *Joshua Archer, Rory Eckel, Joshua Hawkins, Jianlan Wang, Darrel Musslewhite, Yuanlin Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26875](https://doi.org/10.1609/aaai.v37i13.26875)

**Abstract**:

There has been a consensus on integrating Computing into the teaching and learning of STEM (Science, Technology, Engineering and Math) subjects in K-12 (Kindergarten to 12th grade in the US education system). However, rigorous study on the impact of an integrated curriculum on students' learning in computing and/or the STEM subject(s) is still rare. In this paper, we report our research on how well an integrated curriculum helps middle school students learn Computing through the microgenetic analysis methods.

----

## [1806] AI and Parallelism in CS1: Experiences and Analysis

**Authors**: *Steven Bogaerts*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26876](https://doi.org/10.1609/aaai.v37i13.26876)

**Abstract**:

This work considers the use of AI and parallelism as a context for learning typical programming concepts in an introductory programming course (CS1). The course includes exercises in decision trees, a novel game called Find the Gnomes to introduce supervised learning, the construction and application of a vectorized neural network unit class, and obtaining speedup in training through parallelism. The exercises are designed to teach students typical introductory programming concepts while also providing a preview and motivating example of advanced CS topics. Students' understanding and motivation are considered through a detailed analysis of pre- and post-survey data gathered in several sections of the course each taught by one of four instructors across five semesters.

----

## [1807] Shared Tasks as Tutorials: A Methodical Approach

**Authors**: *Theresa Elstner, Frank Loebe, Yamen Ajjour, Christopher Akiki, Alexander Bondarenko, Maik Fröbe, Lukas Gienapp, Nikolay Kolyada, Janis Mohr, Stephan Sandfuchs, Matti Wiegmann, Jörg Frochte, Nicola Ferro, Sven Hofmann, Benno Stein, Matthias Hagen, Martin Potthast*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26877](https://doi.org/10.1609/aaai.v37i13.26877)

**Abstract**:

In this paper, we discuss the benefits and challenges of shared tasks as a teaching method. A shared task is a scientific event and a friendly competition to solve a research problem, the task. In terms of linking research and teaching, shared-task-based tutorials fulfill several faculty desires: they leverage students' interdisciplinary and heterogeneous skills, foster teamwork, and engage them in creative work that has the potential to produce original research contributions. Based on ten information retrieval (IR) courses at two universities since 2019 with shared tasks as tutorials, we derive a domain-neutral process model to capture the respective tutorial structure. Meanwhile, our teaching method has been adopted by other universities in IR courses, but also in other areas of AI such as natural language processing and robotics.

----

## [1808] Maestro: A Gamified Platform for Teaching AI Robustness

**Authors**: *Margarita Geleta, Jiacen Xu, Manikanta Loya, Junlin Wang, Sameer Singh, Zhou Li, Sergio Gago Masagué*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26878](https://doi.org/10.1609/aaai.v37i13.26878)

**Abstract**:

Although the prevention of AI vulnerabilities is critical to preserve the safety and privacy of users and businesses, educational tools for robust AI are still underdeveloped worldwide. We present the design, implementation, and assessment of Maestro. Maestro is an effective open-source game-based platform that contributes to the advancement of robust AI education. Maestro provides "goal-based scenarios" where college students are exposed to challenging life-inspired assignments in a "competitive programming" environment. We assessed Maestro's influence on students' engagement, motivation, and learning success in robust AI. This work also provides insights into the design features of online learning tools that promote active learning opportunities in the robust AI domain. We analyzed the reflection responses (measured with Likert scales) of 147 undergraduate students using Maestro in two quarterly college courses in AI. According to the results, students who felt the acquisition of new skills in robust AI tended to appreciate highly Maestro and scored highly on material consolidation, curiosity, and maestry in robust AI. Moreover, the leaderboard, our key gamification element in Maestro, has effectively contributed to students' engagement and learning. Results also indicate that Maestro can be effectively adapted to any course length and depth without losing its educational quality.

----

## [1809] Exploring Social Biases of Large Language Models in a College Artificial Intelligence Course

**Authors**: *Skylar Kolisko, Carolyn Jane Anderson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26879](https://doi.org/10.1609/aaai.v37i13.26879)

**Abstract**:

Large neural network-based language models play an increasingly important role in contemporary AI. Although these models demonstrate sophisticated text generation capabilities, they have also been shown to reproduce harmful social biases contained in their training data. This paper presents a project that guides students through an exploration of social biases in large language models.

As a final project for an intermediate college course in Artificial Intelligence, students developed a bias probe task for a previously-unstudied aspect of sociolinguistic or sociocultural bias they were interested in exploring. Through the process of constructing a dataset and evaluation metric to measure bias, students mastered key technical concepts, including how to run contemporary neural networks for natural language processing tasks; construct datasets and evaluation metrics; and analyze experimental results. Students reported their findings in an in-class presentation and a final report, recounting patterns of predictions that surprised, unsettled, and sparked interest in advocating for technology that reflects a more diverse set of backgrounds and experiences.

Through this project, students engage with and even contribute to a growing body of scholarly work on social biases in large language models.

----

## [1810] An Analysis of Engineering Students' Responses to an AI Ethics Scenario

**Authors**: *Alexi Orchard, David Radke*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26880](https://doi.org/10.1609/aaai.v37i13.26880)

**Abstract**:

In light of significant issues in the technology industry, such as algorithms that worsen racial biases, the spread of online misinformation, and the expansion of mass surveillance, it is increasingly important to teach the ethics and sociotechnical implications of developing and using artificial intelligence (AI). Using 53 survey responses from engineering undergraduates, this paper measures students' abilities to identify, mitigate, and reflect on a hypothetical AI ethics scenario. We engage with prior research on pedagogical approaches to and considerations for teaching AI ethics and highlight some of the obstacles that engineering undergraduate students experience in learning and applying AI ethics concepts.

----

## [1811] Autonomous Agents: An Advanced Course on AI Integration and Deployment

**Authors**: *Stephanie Rosenthal, Reid G. Simmons*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26881](https://doi.org/10.1609/aaai.v37i13.26881)

**Abstract**:

A majority of the courses on autonomous systems focus on robotics, despite the growing use of autonomous agents in a wide spectrum of applications, from smart homes to intelligent traffic control. Our goal in designing a new senior-level undergraduate course is to teach the integration of a variety of AI techniques in uncertain environments, without the dependence on topics such as robotic control and localization. We chose the application of an autonomous greenhouse to frame our discussions and our student projects because of the greenhouse's self-contained nature and objective metrics for successfully growing plants. We detail our curriculum design, including lecture topics and assignments, and our iterative process for updating the course over the last four years. Finally, we present some student feedback about the course and opportunities for future improvement.

----

## [1812] AI Made by Youth: A Conversational AI Curriculum for Middle School Summer Camps

**Authors**: *Yukyeong Song, Gloria Ashiya Katuka, Joanne Barrett, Xiaoyi Tian, Amit Kumar, Tom McKlin, Mehmet Celepkolu, Kristy Elizabeth Boyer, Maya Israel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26882](https://doi.org/10.1609/aaai.v37i13.26882)

**Abstract**:

As artificial intelligence permeates our lives through various tools and services, there is an increasing need to consider how to teach young learners about AI in a relevant and engaging way. One way to do so is to leverage familiar and pervasive technologies such as conversational AIs. By learning about conversational AIs, learners are introduced to AI concepts such as computers’ perception of natural language, the need for training datasets, and the design of AI-human interactions. In this experience report, we describe a summer camp curriculum designed for middle school learners composed of general AI lessons, unplugged activities, conversational AI lessons, and project activities in which the campers develop their own conversational agents. The results show that this summer camp experience fostered significant increases in learners’ ability beliefs, willingness to share their learning experience, and intent to persist in AI learning. We conclude with a discussion of how conversational AI can be used as an entry point to K-12 AI education.

----

## [1813] Learning Affects Trust: Design Recommendations and Concepts for Teaching Children - and Nearly Anyone - about Conversational Agents

**Authors**: *Jessica Van Brummelen, Mingyan Claire Tian, Maura Kelleher, Nghi Hoang Nguyen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26883](https://doi.org/10.1609/aaai.v37i13.26883)

**Abstract**:

Conversational agents are rapidly becoming commonplace. However, since these systems are typically blackboxed, users—including vulnerable populations, like children—often do not understand them deeply. For example, they might assume agents are overly intelligent, leading to frustration and distrust. Users may also overtrust agents, and thus overshare personal information or rely heavily on agents' advice. Despite this, little research investigates users' perceptions of conversational agents in-depth, and even less investigates how education might change these perceptions to be more healthy. We present workshops with associated educational conversational AI concepts to encourage healthier understanding of agents. Through studies with the curriculum with children and parents from various countries, we found participants' perceptions of agents—specifically their partner models and trust—changed. When participants discussed changes in trust of agents, we found they most often mentioned learning something. For example, they frequently mentioned learning where agents obtained information, what agents do with this information and how agents are programmed. Based on the results, we developed recommendations for teaching conversational agent concepts, including emphasizing the concepts students found most challenging, like training, turn-taking and terminology; supplementing agent development activities with related learning activities; fostering appropriate levels of trust towards agents; and fostering accurate partner models of agents. Through such pedagogy, students can learn to better understand conversational AI and what it means to have it in the world.

----

## [1814] FOLL-E: Teaching First Order Logic to Children

**Authors**: *Simon Vandevelde, Joost Vennekens*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26884](https://doi.org/10.1609/aaai.v37i13.26884)

**Abstract**:

First-order logic (FO) is an important foundation of many domains, including computer science and artificial intelligence. In recent efforts to teach basic CS and AI concepts to children, FO has so far remained absent. In this paper, we examine whether it is possible to design a learning environment that both motivates and enables children to learn the basics of FO. The key components of the learning environment are a syntax-free blocks-based notation for FO, graphics-based puzzles to solve, and a tactile environment which uses computer vision to allow the children to work with wooden blocks. The resulting FOLL-E system is intended to sharpen childrens' reasoning skills, encourage critical thinking and make them aware of the ambiguities of natural language. During preliminary testing with children, they reported that they found the notation intuitive and inviting, and that they enjoyed interacting with the application.

----

## [1815] Responsible Robotics: A Socio-Ethical Addition to Robotics Courses

**Authors**: *Joshua Vekhter, Joydeep Biswas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26885](https://doi.org/10.1609/aaai.v37i13.26885)

**Abstract**:

We are witnessing a rapid increase in real-world autonomous robotic deployments in environments ranging from indoor homes and commercial establishments to large-scale urban areas, with applications ranging from domestic assistance to urban last-mile delivery. The developers of these robots inevitably have to make impactful design decisions to ensure commercially viability, but such decisions have serious real-world consequences. Unfortunately it is not uncommon for such projects to face intense bouts of social backlash, which can be attributed to a wide variety of causes, ranging from inappropriate technical design choices to transgressions of social norms and lack of community engagement.

To better prepare students for the rigors of developing and deploying real-world robotics systems, we developed a Responsible Robotics teaching module, intended to be included in upper-division and graduate level robotics courses. Our module is structured as a role playing exercise which aims to equip students with a framework for navigating the conflicting goals of human actors which govern robots in the field. We report on instructor reflections and anonymous survey responses from offering our responsible robotics module in both a graduate-level, and an upper-division undergraduate robotics course at UT Austin. The responses indicate that students gained a deeper understanding of the socio-technical factors of real-world robotics deployments than they might have using self-study methods, and the students proactively suggested that such modules should be more broadly included in CS courses.

----

## [1816] Data Labeling for Machine Learning Engineers: Project-Based Curriculum and Data-Centric Competitions

**Authors**: *Anastasia Zhdanovskaya, Daria Baidakova, Dmitry Ustalov*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26886](https://doi.org/10.1609/aaai.v37i13.26886)

**Abstract**:

The process of training and evaluating machine learning (ML) models relies on high-quality and timely annotated datasets. While a significant portion of academic and industrial research is focused on creating new ML methods, these communities rely on open datasets and benchmarks. However, practitioners often face issues with unlabeled and unavailable data specific to their domain. We believe that building scalable and sustainable processes for collecting data of high quality for ML is a complex skill that needs focused development. To fill the need for this competency, we created a semester course on Data Collection and Labeling for Machine Learning, integrated into a bachelor program that trains data analysts and ML engineers. The course design and delivery illustrate how to overcome the challenge of putting university students with a theoretical background in mathematics, computer science, and physics through a program that is substantially different from their educational habits. Our goal was to motivate students to focus on practicing and mastering a skill that was considered unnecessary to their work. We created a system of inverse ML competitions that showed the students how high-quality and relevant data affect their work with ML models, and their mindset changed completely in the end. Project-based learning with increasing complexity of conditions at each stage helped to raise the satisfaction index of students accustomed to difficult challenges. During the course, our invited industry practitioners drew on their first-hand experience with data, which helped us avoid overtheorizing and made the course highly applicable to the students’ future career paths.

----

## [1817] Does Knowing When Help Is Needed Improve Subgoal Hint Performance in an Intelligent Data-Driven Logic Tutor?

**Authors**: *Nazia Alam, Mehak Maniktala, Behrooz Mostafavi, Min Chi, Tiffany Barnes*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26887](https://doi.org/10.1609/aaai.v37i13.26887)

**Abstract**:

The assistance dilemma is a well-recognized challenge to determine
when and how to provide help during problem solving
in intelligent tutoring systems. This dilemma is particularly
challenging to address in domains such as logic proofs,
where problems can be solved in a variety of ways. In this
study, we investigate two data-driven techniques to address
the when and how of the assistance dilemma, combining a
model that predicts when students need help learning efficient
strategies, and hints that suggest what subgoal to achieve.
We conduct a study assessing the impact of the new pedagogical
policy against a control policy without these adaptive
components. We found empirical evidence which suggests
that showing subgoals in training problems upon predictions
of the model helped the students who needed it most
and improved test performance when compared to their control
peers. Our key findings include significantly fewer steps
in posttest problem solutions for students with low prior proficiency
and significantly reduced help avoidance for all students
in training.

----

## [1818] Ripple: Concept-Based Interpretation for Raw Time Series Models in Education

**Authors**: *Mohammad Asadi, Vinitra Swamy, Jibril Frej, Julien Tuan Tu Vignoud, Mirko Marras, Tanja Käser*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26888](https://doi.org/10.1609/aaai.v37i13.26888)

**Abstract**:

Time series is the most prevalent form of input data for educational prediction tasks. The vast majority of research using time series data focuses on hand-crafted features, designed by experts for predictive performance and interpretability. However, extracting these features is labor-intensive for humans and computers. In this paper, we propose an approach that utilizes irregular multivariate time series modeling with graph neural networks to achieve comparable or better accuracy with raw time series clickstreams in comparison to hand-crafted features. Furthermore, we extend concept activation vectors for interpretability in raw time series models. We analyze these advances in the education domain, addressing the task of early student performance prediction for downstream targeted interventions and instructional support. Our experimental analysis on 23 MOOCs with millions of combined interactions over six behavioral dimensions show that models designed with our approach can (i) beat state-of-the-art educational time series baselines with no feature extraction and (ii) provide interpretable insights for personalized interventions.
Source code: https://github.com/epfl-ml4ed/ripple/.

----

## [1819] Exploring Tradeoffs in Automated School Redistricting: Computational and Ethical Perspectives

**Authors**: *Fanglan Chen, Subhodip Biswas, Zhiqian Chen, Shuo Lei, Naren Ramakrishnan, Chang-Tien Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26889](https://doi.org/10.1609/aaai.v37i13.26889)

**Abstract**:

The US public school system is administered by local school districts. Each district comprises a set of schools mapped to attendance zones which are annually assessed to meet enrollment objectives. To support school officials in redrawing attendance boundaries, existing approaches have proven promising but still suffer from several challenges, including: 1) inability to scale to large school districts, 2) high computational cost of obtaining compact school attendance zones, and 3) lack of discussion on quantifying ethical considerations underlying the redrawing of school boundaries. Motivated by these challenges, this paper approaches the school redistricting problem from both computational and ethical standpoints. First, we introduce a practical framework based on sampling methods to solve school redistricting as a graph partitioning problem. Next, the advantages of adopting a modified objective function for optimizing discrete geometry to obtain compact boundaries are examined. Lastly, alternative metrics to address ethical considerations in real-world scenarios are formally defined and thoroughly discussed. Our findings highlight the inclusiveness and efficiency advantages of the designed framework and depict how tradeoffs need to be made to obtain qualitatively different school redistricting plans.

----

## [1820] A Dataset for Learning University STEM Courses at Scale and Generating Questions at a Human Level

**Authors**: *Iddo Drori, Sarah J. Zhang, Zad Chin, Reece Shuttleworth, Albert Lu, Linda Chen, Bereket Birbo, Michele He, Pedro Lantigua, Sunny Tran, Gregory Hunter, Bo Feng, Newman Cheng, Roman Wang, Yann Hicke, Saisamrit Surbehera, Arvind Raghavan, Alexander E. Siemenn, Nikhil Singh, Jayson Lynch, Avi Shporer, Nakul Verma, Tonio Buonassisi, Armando Solar-Lezama*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27091](https://doi.org/10.1609/aaai.v37i13.27091)

**Abstract**:

We present a new dataset for learning to solve, explain, and generate university-level STEM questions from 27 courses across a dozen departments in seven universities. We scale up previous approaches to questions from courses in the departments of Mechanical Engineering, Materials Science and Engineering, Chemistry, Electrical Engineering, Computer Science, Physics, Earth Atmospheric and Planetary Sciences, Economics, Mathematics, Biological Engineering, Data Systems, and Society, and Statistics. We visualize similarities and differences between questions across courses. We demonstrate that a large foundation model is able to generate questions that are as appropriate and at the same difficulty level as human-written questions.

----

## [1821] Learning Logical Reasoning Using an Intelligent Tutoring System: A Hybrid Approach to Student Modeling

**Authors**: *Roger Nkambou, Janie Brisson, Ange Tato, Serge Robert*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26891](https://doi.org/10.1609/aaai.v37i13.26891)

**Abstract**:

In our previous works, we presented Logic-Muse as an Intelligent Tutoring System that helps learners improve logical reasoning skills in multiple contexts. Logic-Muse components were validated and argued by experts throughout the designing process (ITS researchers, logicians, and reasoning psychologists). A catalog of reasoning errors (syntactic and semantic) has been established, in addition to an explicit representation of semantic knowledge and the structures and meta-structures underlying conditional reasoning. A Bayesian network with expert validation has been developed and used in a Bayesian Knowledge Tracing (BKT) process that allows the inference of the learner skills. 
This paper presents an evaluation of the learner-model components in Logic-Muse (a bayesian learner model). We conducted a study and collected data from nearly 300 students who processed 48 reasoning activities. These data were used to develop a psychometric model for initializing the learner's model and validating the structure of the initial Bayesian network. We have also developed a neural architecture on which a model was trained to support a deep knowledge tracing (DKT) process. The proposed neural architecture improves the initial version of DKT by allowing the integration of expert knowledge (through the Bayesian Expert Validation Network) and allowing better generalization of knowledge with few samples. The results show a significant improvement in the predictive power of the learner model. The analysis of the results of the psychometric model also illustrates an excellent potential for improving the Bayesian network's structure and the learner model's initialization process.

----

## [1822] Context-Aware Analysis of Group Submissions for Group Anomaly Detection and Performance Prediction

**Authors**: *Narges Norouzi, Amir Mazaheri*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26892](https://doi.org/10.1609/aaai.v37i13.26892)

**Abstract**:

Learning exercises that activate students’ additional cognitive understanding of course concepts facilitate contextualizing the content knowledge and developing higher-order thinking and problem-solving skills. Student-generated instructional materials such as course summaries and problem sets are amongst the instructional strategies that reflect active learning and constructivist philosophy.

The contributions of this work are twofold: 1) We introduce a practical implementation of inside-outside learning strategy in an undergraduate deep learning course and will share our experiences in incorporating student-generated instructional materials learning strategy in course design, and 2) We develop a context-aware deep learning framework to draw insights from the student-generated materials for (i) Detecting anomalies in group activities and (ii) Predicting the median quiz performance of students in each group. This work opens up an avenue for effectively implementing a constructivism learning strategy in large-scale and online courses to build a sense of community between learners while providing an automated tool for instructors to identify at-risk groups.

----

## [1823] CLGT: A Graph Transformer for Student Performance Prediction in Collaborative Learning

**Authors**: *Tianhao Peng, Yu Liang, Wenjun Wu, Jian Ren, Zhao Pengrui, Yanjun Pu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26893](https://doi.org/10.1609/aaai.v37i13.26893)

**Abstract**:

Modeling and predicting the performance of students in collaborative learning paradigms is an important task. Most of the research presented in literature regarding collaborative learning focuses on the discussion forums and social learning networks. There are only a few works that investigate how students interact with each other in team projects and how such interactions affect their academic performance. In order to bridge this gap, we choose a software engineering course as the study subject. The students who participate in a software engineering course are required to team up and complete a software project together. In this work, we construct an interaction graph based on the activities of students grouped in various teams. Based on this student interaction graph, we present an extended graph transformer framework for collaborative learning (CLGT) for evaluating and predicting the performance of students. Moreover, the proposed CLGT contains an interpretation module that explains the prediction results and visualizes the student interaction patterns. The experimental results confirm that the proposed CLGT outperforms the baseline models in terms of performing predictions based on the real-world datasets. Moreover, the proposed CLGT differentiates the students with poor performance in the collaborative learning paradigm and gives teachers early warnings, so that appropriate assistance can be provided.

----

## [1824] H-AES: Towards Automated Essay Scoring for Hindi

**Authors**: *Shubhankar Singh, Anirudh Pupneja, Shivaansh Mital, Cheril Shah, Manish Bawkar, Lakshman Prasad Gupta, Ajit Kumar, Yaman Kumar, Rushali Gupta, Rajiv Ratn Shah*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26894](https://doi.org/10.1609/aaai.v37i13.26894)

**Abstract**:

The use of Natural Language Processing (NLP) for Automated Essay Scoring (AES) has been well explored in the English language, with benchmark models exhibiting performance comparable to human scorers. However, AES in Hindi and other low-resource languages remains unexplored. In this study, we reproduce and compare state-of-the-art methods for AES in the Hindi domain. We employ classical feature-based Machine Learning (ML) and advanced end-to-end models, including LSTM Networks and Fine-Tuned Transformer Architecture, in our approach and derive results comparable to those in the English language domain. Hindi being a low-resource language, lacks a dedicated essay-scoring corpus. We train and evaluate our models using translated English essays and empirically measure their performance on our own small-scale, real-world Hindi corpus. We follow this up with an in-depth analysis discussing prompt-specific behavior of different language models implemented.

----

## [1825] Detecting Exclusive Language during Pair Programming

**Authors**: *Solomon Ubani, Rodney Nielsen, Helen Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26895](https://doi.org/10.1609/aaai.v37i13.26895)

**Abstract**:

Inclusive team participation is one of the most important factors that aids effective collaboration and pair programming. In this paper, we investigated the ability of linguistic features and a transformer-based language model to detect exclusive and inclusive language. The task of detecting exclusive language was approached as a text classification problem. We created a research community resource consisting of a dataset of 40,490 labeled utterances obtained from three programming assignments involving 34 students pair programming in a remote environment. This research involves the first successful automated detection of exclusive language during pair programming. Additionally, this is the first work to perform a computational linguistic analysis on the verbal interaction common in the context of inclusive and exclusive language during pair programming.

----

## [1826] Solving Math Word Problems concerning Systems of Equations with GPT-3

**Authors**: *Mingyu Zong, Bhaskar Krishnamachari*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26896](https://doi.org/10.1609/aaai.v37i13.26896)

**Abstract**:

Researchers have been interested in developing AI tools to help students learn various mathematical subjects. One challenging set of tasks for school students is learning to solve math word problems. We explore how recent advances in natural language processing, specifically the rise of powerful transformer based models, can be applied to help math learners with such problems. Concretely, we evaluate the use of GPT-3, a 1.75B parameter transformer model recently released by OpenAI, for three related challenges pertaining to math word problems corresponding to systems of two linear equations. The three challenges are classifying word problems, extracting equations from word problems, and generating word problems. For the first challenge, we define a set of problem classes and find that GPT-3 has generally very high accuracy in classifying word problems (80%-100%), for all but one of these classes. For the second challenge, we find the accuracy for extracting equations improves with number of examples provided to the model, ranging from an accuracy of 31% for zero-shot learning to about 69% using 3-shot learning, which is further improved to a high value of 80% with fine-tuning. For the third challenge, we find that GPT-3 is able to generate problems with accuracy ranging from 33% to 93%, depending on the problem type.

----

## [1827] AI Audit: A Card Game to Reflect on Everyday AI Systems

**Authors**: *Safinah Ali, Vishesh Kumar, Cynthia Breazeal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26897](https://doi.org/10.1609/aaai.v37i13.26897)

**Abstract**:

An essential element of K-12 AI literacy is educating learners about the ethical and societal implications of AI systems. Previous work in AI ethics literacy have developed curriculum and classroom activities that engage learners in reflecting on the ethical implications of AI systems and developing responsible AI. There is little work in using game-based learning methods in AI literacy. Games are known to be compelling media to teach children about complex STEM concepts. In this work, we developed a competitive card game for middle and high school students called “AI Audit” where they play as AI start-up founders building novel AI-powered technology. Players can challenge other players with potential harms of their technology or defend their own businesses by features that mitigate these harms. The game mechanics reward systems that are ethically developed or that take steps to mitigate potential harms. In this paper, we present the game design, teacher resources for classroom deployment and early playtesting results. We discuss our reflections about using games as teaching tools for AI literacy in K-12 classrooms.

----

## [1828] Beyond Black-Boxes: Teaching Complex Machine Learning Ideas through Scaffolded Interactive Activities

**Authors**: *Brian Broll, Shuchi Grover*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26898](https://doi.org/10.1609/aaai.v37i13.26898)

**Abstract**:

Existing approaches to teaching artificial intelligence and machine learning (ML) often focus on the use of pre-trained models or fine-tuning an existing black-box architecture. We believe ML techniques and core ML topics, such as optimization and adversarial examples, can be designed for high school age students given appropriate support. Our curricular approach focuses on teaching ML ideas by enabling students to develop deep intuition about these complex concepts by first making them accessible to novices through interactive tools, pre-programmed games, and carefully designed programming activities. Then, students are able to engage with the concepts via meaningful, hands-on experiences that span the entire ML process from data collection to model optimization and inspection. This paper describes our 'AI & Cybersecurity for Teens' suite of curricular activities aimed at high school students and teachers.

----

## [1829] Exploring Artificial Intelligence in English Language Arts with StoryQ

**Authors**: *Jie Chao, Rebecca Ellis, Shiyan Jiang, Carolyn P. Rosé, William Finzer, Cansu Tatar, James Fiacco, Kenia Wiedemann*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26899](https://doi.org/10.1609/aaai.v37i13.26899)

**Abstract**:

Exploring Artificial Intelligence (AI) in English Language Arts (ELA) with StoryQ is a 10-hour curriculum module designed for high school ELA classes. The module introduces students to fundamental AI concepts and essential machine learning workflow using StoryQ, a web-based GUI environment for Grades 6-12 learners. In this module, students work with unstructured text data and learn to train, test, and improve text classification models such as intent recognition, clickbait filter, and sentiment analysis. As they interact with machine-learning language models deeply, students also gain a nuanced understanding of language and how to wield it, not just as a data structure, but as a tool in our human-human encounters as well. The current version contains eight lessons, all delivered through a full-featured online learning and teaching platform. Computers and Internet access are required to implement the module. The module was piloted in an ELA class in the Spring of 2022, and the student learning outcomes were positive. The module is currently undergoing revision and will be further tested and improved in Fall 2022.

----

## [1830] An Introduction to Rule-Based Feature and Object Perception for Middle School Students

**Authors**: *Daniella DiPaola, Parker Malachowsky, Nancye Blair Black, Sharifa Alghowinem, Xiaoxue Du, Cynthia Breazeal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26900](https://doi.org/10.1609/aaai.v37i13.26900)

**Abstract**:

The Feature Detection tool is a web-based activity that allows students to detect features in images and build their own rule-based classification algorithms. In this paper, we introduce the tool and share how it is incorporated into two, 45-minute lessons. The objective of the first lesson is to introduce students to the concept of feature detection, or how a computer can break down visual input into lower-level features. The second lesson aims to show students how these lower-level features can be incorporated into rule-based models to classify higher-order objects. We discuss how this tool can be used as a "first step" to the more complex concept ideas of data representation and neural networks.

----

## [1831] Scratch for Sports: Athletic Drills as a Platform for Experiencing, Understanding, and Developing AI-Driven Apps

**Authors**: *Vishesh Kumar, Marcelo Worsley*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26901](https://doi.org/10.1609/aaai.v37i13.26901)

**Abstract**:

Culturally relevant and sustaining implementations of computing education are increasingly leveraging young learners' passion for sports as a platform for building interest in different STEM (Science, Technology, Engineering, and Math) concepts. Numerous disciplines spanning physics, engineering, data science, and especially AI based computing are not only authentically used in professional sports in today's world, but can also be productively introduced to introduce young learnres to these disciplines and facilitate deep engagement with the same in the context of sports. In this work, we present a curriculum that includes a constellation of proprietary apps and tools we show student athletes learning sports like basketball and soccer that use AI methods like pose detection and IMU-based gesture detection to track activity and provide feedback. We also share Scratch extensions which enable rich access to sports related pose, object, and gesture detection algorithms that youth can then tinker around with and develop their own sports drill applications. We present early findings from pilot implementations of portions of these tools and curricula, which also fostered discussion relating to the failings, risks, and social harms associated with many of these different AI methods – noticeable in professional sports contexts, and relevant to youths' lives as active users of AI technologies as well as potential future creators of the same.

----

## [1832] "How Can I Code AI Responsibly?": The Effect of Computational Action on K-12 Students Learning and Creating Socially Responsible AI

**Authors**: *H. Nicole Pang, Robert Parks, Cynthia Breazeal, Hal Abelson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26902](https://doi.org/10.1609/aaai.v37i13.26902)

**Abstract**:

Teaching young people about artificial intelligence (A.I.) is recognized globally as an important education effort by organizations and programs such as UNICEF, OECD, Elements of A.I., and AI4K12. A common theme among K-12 A.I. education programs is teaching how A.I. can impact society in both positive and negative ways. We present an effective tool that teaches young people about the societal impact of A.I. that goes one step further: empowering K-12 students to use tools and frameworks to create socially responsible A.I. The computational action process is a curriculum and toolkit that gives students the lessons and tools to evaluate positive and negative impacts of A.I. and consider how they can create beneficial solutions that involve A.I. and computing technology. In a human-subject research study, 101 U.S. and international students between ages 9 and 18 participated in a one-day workshop to learn and practice the computational action process. Pre-post questionnaires measured on the Likert scale students’ perception of A.I. in society and students' desire to use A.I. in their projects. Analysis of the results shows that students who identified as female agreed more strongly with having a concern about the impacts of A.I. than those who identified as male. Students also wrote open-ended responses to questions about what socially responsible technology means to them pre- and post-study. Analysis shows that post-intervention, students were more aware of ethical considerations and what tools they can use to code A.I. responsibly. In addition, students engaged actively with tools in the computational action toolkit, specifically the novel impact matrix, to describe the positive and negative impacts of A.I. technologies like facial recognition. Students demonstrated breadth and depth of discussion of various A.I. technologies' far-reaching positive and negative impacts. These promising results indicate that the computational action process can be a helpful addition to A.I. education programs in furnishing tools for students to analyze the effects of A.I. on society and plan how they can create and use socially responsible A.I.

----

## [1833] Build-a-Bot: Teaching Conversational AI Using a Transformer-Based Intent Recognition and Question Answering Architecture

**Authors**: *Kate Pearce, Sharifa Alghowinem, Cynthia Breazeal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26903](https://doi.org/10.1609/aaai.v37i13.26903)

**Abstract**:

As artificial intelligence (AI) becomes a prominent part of modern life, AI literacy is becoming important for all citizens, not just those in technology careers. Previous research in AI education materials has largely focused on the introduction of terminology as well as AI use cases and ethics, but few allow students to learn by creating their own machine learning models. Therefore, there is a need for enriching AI educational tools with more  adaptable and flexible platforms for interested educators with any level of technical experience to utilize within their teaching material. As such, we propose the development of an open-source tool (Build-A-Bot) for students and teachers to not only create their own transformer-based chatbots based on their own course material but also learn the fundamentals of AI through the model creation process. The primary concern of this paper is the creation of an interface for students to learn the principles of artificial intelligence by using a natural language pipeline to train a customized model to answer questions based on their own school curriculums. The model uses contexts given by their instructor, such as chapters of a textbook, to answer questions and is deployed on an interactive chatbot/voice agent. The pipeline teaches students data collection, data augmentation, intent recognition, and question answering by having them work through each of these processes while creating their AI agent, diverging from previous chatbot work where students and teachers use the bots as black-boxes with no abilities for customization or the bots lack AI capabilities, with the majority of dialogue scripts being rule-based. In addition, our tool is designed to make each step of this pipeline intuitive for students at a middle-school level. Further work primarily lies in providing our tool to schools and seeking student and teacher evaluations.

----

## [1834] Develop AI Teaching and Learning Resources for Compulsory Education in China

**Authors**: *Jiachen Song, Jinglei Yu, Li Yan, Linan Zhang, Bei Liu, Yujin Zhang, Yu Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26904](https://doi.org/10.1609/aaai.v37i13.26904)

**Abstract**:

Artificial intelligence course has been required to take for compulsory education students in China. However, not all teachers and schools are fully prepared and ready. This is partially because of the lack of adequate teaching and learning resources, which requires a major expenditure of time and effort for schools and teachers to design and develop. To meet the challenge of lacking appropriate resources in teaching and learning AI from grade 1 to grade 9, we developed AI knowledge structure and instructional resources based on Chinese national curriculum for information science and technology. Our comprehensive AI syllabus contains 90 core concepts, 63 learning indicators, and 27 teaching and learning resources, which have been implemented. The resources have been taken as model courses in teacher training programs and an exemplary course has been implemented in primary schools that verified the effectiveness of our resources.

----

## [1835] Guiding Students to Investigate What Google Speech Recognition Knows about Language

**Authors**: *David S. Touretzky, Christina Gardner-McCune*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26905](https://doi.org/10.1609/aaai.v37i13.26905)

**Abstract**:

Today, children of all ages interact with speech recognition systems but are largely unaware of how they work. Teaching K-12 students to investigate how these systems employ phonological, syntactic, semantic, and cultural knowledge to resolve ambiguities in the audio signal can provide them a window on complex AI decision-making and also help them appreciate the richness and complexity of human language. We describe a browser-based tool for exploring the Google Web Speech API and a series of experiments students can engage in to measure what the service knows about language and the types of biases it exhibits. Middle school students taking an introductory AI elective were able to use the tool to explore Google’s knowledge of homophones and its ability to exploit context to disambiguate them. Older students could potentially conduct more comprehensive investigations, which we lay out here. This approach to investigating the power and limitations of speech technology through carefully designed experiments can also be applied to other AI application areas, such as face detection, object recognition, machine translation, or question answering.

----

## [1836] Literacy and STEM Teachers Adapt AI Ethics Curriculum

**Authors**: *Benjamin Walsh, Bridget Dalton, Stacey Forsyth, Tom Yeh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26906](https://doi.org/10.1609/aaai.v37i13.26906)

**Abstract**:

This article examines the ways secondary computer science and English Language Arts teachers in urban, suburban, and semi-rural schools adapted a project-based AI ethics curriculum to make it better fit their local contexts. AI ethics is an urgent topic with tangible consequences for youths’ current and future lives, but one that is rarely taught in schools. Few teachers have formal training in this area as it is an emerging field even at the university level. Exploring AI ethics involves examining biases related to race, gender, and social class, a challenging task for all teachers, and an unfamiliar one for most computer science teachers. It also requires teaching technical content which falls outside the comfort zone of most humanities teachers. Although none of our partner teachers had previously taught an AI ethics project, this study demonstrates that their expertise and experience in other domains played an essential role in providing high quality instruction. Teachers designed and redesigned tasks and incorporated texts and apps to ensure the AI ethics project would adhere to district and department level requirements; they led equity-focused inquiry in a way that both protected vulnerable students and accounted for local cultures and politics; and they adjusted technical content and developed hands-on computer science experiences to better challenge and engage their students. We use Mishra and Kohler’s TPACK framework to highlight the ways teachers leveraged their own expertise in some areas, while relying on materials and support from our research team in others, to create stronger learning experiences.

----

## [1837] MoMusic: A Motion-Driven Human-AI Collaborative Music Composition and Performing System

**Authors**: *Weizhen Bian, Yijin Song, Nianzhen Gu, Tin Yan Chan, Tsz To Lo, Tsun Sun Li, King Chak Wong, Wei Xue, Roberto Alonso Trillo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26907](https://doi.org/10.1609/aaai.v37i13.26907)

**Abstract**:

The significant development of artificial neural network architectures has facilitated the increasing adoption of automated music composition models over the past few years. However, most existing systems feature algorithmic generative structures based on hard code and predefined rules, generally excluding interactive or improvised behaviors. We propose a motion based music system, MoMusic, as a AI real time music generation system. MoMusic features a partially randomized harmonic sequencing model based on a probabilistic analysis of tonal chord progressions, mathematically abstracted through musical set theory. This model is presented against a dual dimension grid that produces resulting sounds through a posture recognition mechanism. A camera captures the users' fingers' movement and trajectories, creating coherent, partially improvised harmonic progressions. MoMusic integrates several timbrical registers, from traditional classical instruments such as the piano to a new ''human voice instrument'' created using a voice conversion technique. Our research demonstrates MoMusic's interactiveness, ability to inspire musicians, and ability to generate coherent musical material with various timbrical registers. MoMusic's capabilities could be easily expanded to incorporate different forms of posture controlled timbrical transformation, rhythmic transformation, dynamic transformation, or even digital sound processing techniques.

----

## [1838] A Multi-User Virtual World with Music Recommendations and Mood-Based Virtual Effects

**Authors**: *Charats Burch, Robert Sprowl, Mehmet Ergezer*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26908](https://doi.org/10.1609/aaai.v37i13.26908)

**Abstract**:

The SEND/RETURN (S/R) project is created to explore the efficacy of content-based music recommendations alongside a uniquely generated Unreal Engine 5 (UE5) virtual environment based on audio features. S/R employs both a k-means clustering algorithm using audio features and a fast pattern matching (FPM) algorithm using 30-second audio signals to find similar-sounding songs to recommend to users.  The feature values of the recommended song are then communicated via HTTP to the UE5 virtual environment, which changes a number of effects in real-time. All of this is being replicated from a listen-server to other clients to create a multiplayer audio session. S/R successfully creates a lightweight online environment that replicates song information to all clients and suggests new songs that alter the world around you. In this work, we extend S/R by training a convolutional neural network using Mel-spectrograms of 30-second audio samples to predict the mood of a song. This model can then orchestrate the post-processing effect in the UE5 virtual environment. The developed convolutional model had a validation accuracy of 67.5% in predicting 4 moods ('calm', 'energetic', 'happy', 'sad').

----

## [1839] Learning Adaptive Game Soundtrack Control

**Authors**: *Aaron Dorsey, Todd W. Neller, Hien G. Tran, Veysel Yilmaz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26909](https://doi.org/10.1609/aaai.v37i13.26909)

**Abstract**:

In this paper, we demonstrate a novel technique for dynamically generating an emotionally-directed video game soundtrack.  We begin with a human Conductor observing gameplay and directing associated emotions that would enhance the observed gameplay experience.  We apply supervised learning to data sampled from synchronized input gameplay features and Conductor output emotional direction features in order to fit a mathematical model to the Conductor's emotional direction.  Then, during gameplay, the emotional direction model maps gameplay state input to emotional direction output, which is then input to a music generation module that dynamically generates emotionally-relevant music during gameplay.  Our empirical study suggests that random forests serve well for modeling the Conductor for our two experimental game genres.

----

## [1840] Predicting Perceived Music Emotions with Respect to Instrument Combinations

**Authors**: *Viet Dung Nguyen, Quan H. Nguyen, Richard G. Freedman*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26910](https://doi.org/10.1609/aaai.v37i13.26910)

**Abstract**:

Music Emotion Recognition has attracted a lot of academic research work in recent years because it has a wide range of applications, including song recommendation and music visualization. As music is a way for humans to express emotion, there is a need for a machine to automatically infer the perceived emotion of pieces of music. In this paper, we compare the accuracy difference between music emotion recognition models given music pieces as a whole versus music pieces separated by instruments. To compare the models' emotion predictions, which are distributions over valence and arousal values, we provide a metric that compares two distribution curves. Using this metric, we provide empirical evidence that training Random Forest and Convolution Recurrent Neural Network with mixed instrumental music data conveys a better understanding of emotion than training the same models with music that are separated into each instrumental source.

----

## [1841] Emotion-Aware Music Recommendation

**Authors**: *Hieu Tran, Tuan Le, Anh Do, Tram Vu, Steven Bogaerts, Brian Howard*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26911](https://doi.org/10.1609/aaai.v37i13.26911)

**Abstract**:

It is common to listen to songs that match one's mood. Thus, an AI music recommendation system that is aware of the user's emotions is likely to provide a superior user experience to one that is unaware. In this paper, we present an emotion-aware music recommendation system. Multiple models are discussed and evaluated for affect identification from a live image of the user. We propose two models: DRViT, which applies dynamic routing to vision transformers, and InvNet50, which uses involution. All considered models are trained and evaluated on the AffectNet dataset. Each model outputs the user's estimated valence and arousal under the circumplex model of affect. These values are compared to the valence and arousal values for songs in a Spotify dataset, and the top-five closest-matching songs are presented to the user. Experimental results of the models and user testing are presented.

----

## [1842] Music-to-Facial Expressions: Emotion-Based Music Visualization for the Hearing Impaired

**Authors**: *Yubo Wang, Fengzhou Pan, Danni Liu, Jiaxiong Hu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26912](https://doi.org/10.1609/aaai.v37i13.26912)

**Abstract**:

While music is made to convey messages and emotions, auditory music is not equally accessible to everyone. Music visualization is a common approach to augment the listening experiences of the hearing users and to provide music experiences for the hearing-impaired. In this paper, we present a music visualization system that can turn the input of a piece of music into a series of facial expressions representative of the continuously changing sentiments in the music. The resulting facial expressions, recorded as action units, can later animate a static virtual avatar to be emotive synchronously with the music.

----

## [1843] Model AI Assignments 2023

**Authors**: *Todd W. Neller, Raechel Walker, Olivia Dias, Zeynep Yalçin, Cynthia Breazeal, Matthew E. Taylor, Michele Donini, Erin J. Talvitie, Charlie Pilgrim, Paolo Turrini, James Maher, Matthew Boutell, Justin Wilson, Narges Norouzi, Jonathan Scott*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26913](https://doi.org/10.1609/aaai.v37i13.26913)

**Abstract**:

The Model AI Assignments session seeks to gather and disseminate the best assignment designs of the Artificial Intelligence (AI) Education community.  Recognizing that assignments form the core of student learning experience, we here present abstracts of six AI assignments from the 2023 session that are easily adoptable, playfully engaging, and flexible for a variety of instructor needs.  Assignment specifications and supporting resources may be found at http://modelai.gettysburg.edu .

----

## [1844] Probabilistic Shape Models of Anatomy Directly from Images

**Authors**: *Jadie Adams*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26914](https://doi.org/10.1609/aaai.v37i13.26914)

**Abstract**:

Statistical shape modeling (SSM) is an enabling tool in medical image analysis as it allows for population-based quantitative analysis. The traditional pipeline for landmark-based SSM from images requires painstaking and cost-prohibitive steps. My thesis aims to leverage probabilistic deep learning frameworks to streamline the adoption of SSM in biomedical research and practice. The expected outcomes of this work will be new frameworks for SSM that (1) provide reliable and calibrated uncertainty quantification, (2) are effective given limited or sparsely annotated/incomplete data, and (3) can make predictions from incomplete 4D spatiotemporal data. These efforts will reduce required costs and manual labor for anatomical SSM, helping SSM become a more viable clinical tool and advancing medical practice.

----

## [1845] Modeling Strategies as Programs: How to Study Strategy Differences in Intelligent Systems with Program Synthesis

**Authors**: *James Ainooson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26915](https://doi.org/10.1609/aaai.v37i13.26915)

**Abstract**:

When faced with novel tasks, humans have the ability to form successful strategies, seemingly without much effort. Artificial systems, on the other, hand cannot, at least when the flexibility at which humans perform is considered. For my dissertation, I am using program synthesis as a tool to study the factors that affect strategy choices in intelligent systems. I am evaluating my work through agents that reason through problems from the Abstract Reasoning Corpus and The Block Design Task.

----

## [1846] Non-exponential Reward Discounting in Reinforcement Learning

**Authors**: *Raja Farrukh Ali*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26916](https://doi.org/10.1609/aaai.v37i13.26916)

**Abstract**:

Reinforcement learning methods typically discount future rewards using an exponential scheme to achieve theoretical convergence guarantees. Studies from neuroscience, psychology, and economics suggest that human and animal behavior is better captured by the hyperbolic discounting model. Hyperbolic discounting has recently been studied in deep reinforcement learning and has shown promising results. However, this area of research is seemingly understudied, with most extant and continuing research using the standard exponential discounting formulation. My dissertation examines the effects of non-exponential discounting functions (such as hyperbolic) on an agent's learning and aims to investigate their impact on multi-agent systems and generalization tasks. A key objective of this study is to link the discounting rate to an agent's approximation of the underlying hazard rate of its environment through survival analysis.

----

## [1847] Enhancing Smart, Sustainable Mobility with Game Theory and Multi-Agent Reinforcement Learning With Applications to Ridesharing

**Authors**: *Lucia Cipolina-Kun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26917](https://doi.org/10.1609/aaai.v37i13.26917)

**Abstract**:

We propose the use of game-theoretic solutions and multi- agent Reinforcement Learning in the mechanism design of smart, sustainable mobility services. In particular, we present applications to ridesharing as an example of a cost game.

----

## [1848] Assessing Learned Representations under Open-World Novelty

**Authors**: *Kaleigh Clary*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26918](https://doi.org/10.1609/aaai.v37i13.26918)

**Abstract**:

My dissertation research focuses on sequential decision-making (SDM) in complex environments, and how agents can perform well even when novelty is introduced to those environments. The problem of how agents can respond intelligently to novelty has been a long-standing challenge in AI, and poses unique problems across approaches to SDM. This question has been studied in various formulations, including open-world learning and reasoning, transfer learning, concept drift, and statistical relational learning. Classical and modern approaches in agent design offer tradeoffs in human effort for feature encoding, ease of deployment in new domains, and the development of both provably and empirically reliable policies. I propose a formalism for studying open-world novelty in SDM processes with feature-rich observations. I study the conditions under which causal-relational queries can be estimated from non-novel observations, and empirically examine the effects of open-world novelty on agent behavior.

----

## [1849] Efficient Non-parametric Neural Density Estimation and Its Application to Outlier and Anomaly Detection

**Authors**: *Joseph A. Gallego-Mejia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26919](https://doi.org/10.1609/aaai.v37i13.26919)

**Abstract**:

The main goal of this thesis is to develop efficient non-parametric density estimation methods that can be integrated with deep learning architectures, for instance, convolutional neural networks and transformers. Density estimation methods can be applied to different problems in statistics and machine learning. They may be used to solve tasks such as anomaly detection, generative models, semi-supervised learning, compression, text-to-speech, among others. The present work will mainly focus on the application of the method in anomaly and outlier detection tasks such as medical anomaly detection, fraud detection, video surveillance, time series anomaly detection, industrial damage detection, among others. A recent approach to non-parametric density estimation is neural density estimation. One advantage of these methods is that they can be integrated with deep learning architectures and trained using gradient descent. Most of these methods are based on neural network implementations of normalizing flows which transform an original simpler distribution to a more complex one. The approach of this thesis is based on a different idea that combines random Fourier features with density matrices to estimate the underlying distribution function. The method can be seen as an approximation of the popular kernel density estimation method but without the inherent computational cost.

----

## [1850] Explaining the Uncertainty in AI-Assisted Decision Making

**Authors**: *Thao Le*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26920](https://doi.org/10.1609/aaai.v37i13.26920)

**Abstract**:

The aim of this project is to improve human decision-making using explainability; specifically, how to explain the (un)certainty of machine learning models. Prior research has used uncertainty measures to promote trust and decision-making. However, the direction of explaining why the AI prediction is confident (or not confident) in its prediction needs to be addressed. By explaining the model uncertainty, we can promote trust, improve understanding and improve decision-making for users.

----

## [1851] Poisoning-Based Backdoor Attacks in Computer Vision

**Authors**: *Yiming Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26921](https://doi.org/10.1609/aaai.v37i13.26921)

**Abstract**:

Recent studies demonstrated that the training process of deep neural networks (DNNs) is vulnerable to backdoor attacks if third-party training resources (e.g., samples) are adopted. Specifically, the adversaries intend to embed hidden backdoors into DNNs, where the backdoor can be activated by pre-defined trigger patterns and leading malicious model predictions. My dissertation focuses on poisoning-based backdoor attacks in computer vision. Firstly, I study and propose more stealthy and effective attacks against image classification tasks in both physical and digital spaces. Secondly, I reveal the backdoor threats in visual object tracking, which is representative of critical video-related tasks. Thirdly, I explore how to exploit backdoor attacks as watermark techniques for positive purposes. I design a Python toolbox (i.e., BackdoorBox) that implements representative and advanced backdoor attacks and defenses under a unified and flexible framework, based on which to provide a comprehensive benchmark of existing methods at the end.

----

## [1852] Safe Interactive Autonomy for Multi-Agent Systems

**Authors**: *Yiwei Lyu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26922](https://doi.org/10.1609/aaai.v37i13.26922)

**Abstract**:

It is envisioned that in the near future autonomous systems such as multi-agent systems, will co-exist with humans, e.g., autonomous vehicles will share roads with human drivers. These safety-critical scenarios require formally provable safety guarantees so that the robots will never collide with humans or with each other. It is challenging to provide such guarantees in the real world due to the stochastic environments and inaccurate models of heterogeneous agents including robots and humans. My PhD research investigates decision-making algorithm design for provably-correct safety guarantees in mixed multi-agent systems.

----

## [1853] Theory of Mind: A Familiar Aspect of Humanity to Give Machines

**Authors**: *Joel Michelson*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26923](https://doi.org/10.1609/aaai.v37i13.26923)

**Abstract**:

My research focuses on machine models of theory of mind, a set of skills that helps humans cooperate with each other. Because these skills present themselves in behavior, inference-based measurements must be carefully designed to rule out alternate hypotheses. Producing models that display these skills requires an extensive understanding of experiences and mechanisms sufficient for learning, and the models must have robust generalization to be effective in varied domains. To address these problems, I intend to evaluate computational models of ToM using a variety of tests.

----

## [1854] Multimodal Deep Generative Models for Remote Medical Applications

**Authors**: *Catherine Ordun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26924](https://doi.org/10.1609/aaai.v37i13.26924)

**Abstract**:

Visible-to-Thermal (VT) face translation is an under-studied problem of image-to-image translation that offers an AI-enabled alternative to traditional thermal sensors. Over three phases, my Doctoral Proposal explores developing multimodal deep generative solutions that can be applied towards telemedicine applications. These include the contribution of a novel Thermal Face Contrastive GAN (TFC-GAN), exploration of hybridized diffusion-GAN models, application on real clinical thermal data at the National Institutes of Health, and exploration of strategies for Federated Learning (FL) in heterogenous data settings.

----

## [1855] Topics in Selective Classification

**Authors**: *Andrea Pugnana*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26925](https://doi.org/10.1609/aaai.v37i13.26925)

**Abstract**:

In recent decades, advancements in information technology allowed Artificial Intelligence (AI) systems to predict future outcomes with unprecedented success. This brought the widespread deployment of these methods in many fields, intending to support decision-making. A pressing question is how to make AI systems robust to common challenges in real-life scenarios and trustworthy. 
In my work, I plan to explore ways to enhance the trustworthiness of AI through the selective classification framework. In this setting, the AI system can refrain from predicting whenever it is not confident enough, allowing it to trade off coverage, i.e. the percentage of instances that receive a prediction, for performance.

----

## [1856] Knowledge-Embedded Narrative Construction from Open Source Intelligence

**Authors**: *Priyanka Ranade*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26926](https://doi.org/10.1609/aaai.v37i13.26926)

**Abstract**:

Storytelling is an innate part of language-based communication. Today, current events are reported via Open Source Intelligence (OSINT) sources like news websites, blogs, and discussion forums. Scattered and fragmented sources such as these can be better understood when organized as chains of event plot points, or narratives, that have the ability to communicate end-end stories. Though search engines can retrieve aggregated event information, they lack the ability to sequence relevant events together to form narratives about different topics. I propose an AI system inspired by Gustav Freytag’s narrative theory called the Plot Element Pyramid and use knowledge graphs to represent, chain, and reason over narratives from disparately sourced event details to better comprehend convoluted, noisy information about critical events during intelligence analysis.

----

## [1857] Learning Better Representations Using Auxiliary Knowledge

**Authors**: *Saed Rezayi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26927](https://doi.org/10.1609/aaai.v37i13.26927)

**Abstract**:

Representation Learning is the core of Machine Learning and Artificial Intelligence as it summarizes input data points into low dimensional vectors. This low dimensional vectors should be accurate portrayals of the input data, thus it is crucial to find the most effective and robust representation possible for given input as the performance of the ML task is dependent on the resulting representations. In this summary, we discuss an approach to augment representation learning which relies on external knowledge. We briefly describe the shortcoming of the existing techniques and describe how an auxiliary knowledge source could result in obtaining improved representations.

----

## [1858] Embodied, Intelligent Communication for Multi-Agent Cooperation

**Authors**: *Esmaeil Seraj*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26928](https://doi.org/10.1609/aaai.v37i13.26928)

**Abstract**:

High-performing human teams leverage intelligent and efficient communication and coordination strategies to collaboratively maximize their joint utility. Inspired by teaming behaviors among humans, I seek to develop computational methods for synthesizing intelligent communication and coordination strategies for collaborative multi-robot systems. I leverage both classical model-based control and planning approaches as well as data-driven methods such as Multi-Agent Reinforcement Learning (MARL) to provide several contributions towards enabling emergent cooperative teaming behavior across both homogeneous and heterogeneous (including agents with different capabilities) robot teams.

----

## [1859] Meta Learning in Decentralized Neural Networks: Towards More General AI

**Authors**: *Yuwei Sun*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26929](https://doi.org/10.1609/aaai.v37i13.26929)

**Abstract**:

Meta-learning usually refers to a learning algorithm that learns from other learning algorithms. The problem of uncertainty in the predictions of neural networks shows that the world is only partially predictable and a learned neural network cannot generalize to its ever-changing surrounding environments. Therefore, the question is how a predictive model can represent multiple predictions simultaneously. We aim to provide a fundamental understanding of learning to learn in the contents of Decentralized Neural Networks (Decentralized NNs) and we believe this is one of the most important questions and prerequisites to building an autonomous intelligence machine. To this end, we shall demonstrate several pieces of evidence for tackling the problems above with Meta Learning in Decentralized NNs. In particular, we will present three different approaches to building such a decentralized learning system: (1) learning from many replica neural networks, (2) building the hierarchy of neural networks for different functions, and (3) leveraging different modality experts to learn cross-modal representations.

----

## [1860] Learning and Planning under Uncertainty for Conservation Decisions

**Authors**: *Lily Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26930](https://doi.org/10.1609/aaai.v37i13.26930)

**Abstract**:

My research focuses on new techniques in machine learning and game theory to optimally allocate our scarce resources in multi-agent settings to maximize environmental sustainability. Drawing scientific questions from my close partnership with conservation organizations, I have advanced new lines of research in learning and planning under uncertainty, inspired by the low-data, noisy, and dynamic settings faced by rangers on the frontlines of protected areas.

----

## [1861] Failure-Resistant Intelligent Interaction for Reliable Human-AI Collaboration

**Authors**: *Hiromu Yakura*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26931](https://doi.org/10.1609/aaai.v37i13.26931)

**Abstract**:

My thesis is focusing on how we can overcome the gap people have against machine learning techniques that require a well-defined application scheme and can produce wrong results. I am planning to discuss the principle of the interaction design that fills such a gap based on my past projects that have explored better interactions for applying machine learning in various fields, such as malware analysis, executive coaching, photo editing, and so on. To this aim, my thesis also shed a light on the limitations of machine learning techniques, like adversarial examples, to highlight the importance of "failure-resistant intelligent interaction."

----

## [1862] Privacy-Preserving Representation Learning for Text-Attributed Networks with Simplicial Complexes

**Authors**: *Huixin Zhan, Victor S. Sheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26932](https://doi.org/10.1609/aaai.v37i13.26932)

**Abstract**:

Although recent network representation learning (NRL) works in text-attributed networks demonstrated superior performance for various graph inference tasks, learning network representations could always raise privacy concerns when nodes represent people or human-related variables. Moreover, standard NRLs that leverage structural information from a graph proceed by first encoding pairwise relationships into learned representations and then analysing its properties. This approach is fundamentally misaligned with problems where the relationships involve multiple points, and topological structure must be encoded beyond pairwise interactions. Fortunately, the machinery of topological data analysis (TDA) and, in particular, simplicial neural networks (SNNs) offer a mathematically rigorous framework to evaluate not only higher-order interactions, but also global invariant features of the observed graph to systematically learn topological structures. It is critical to investigate if the representation outputs from SNNs are more vulnerable compared to regular representation outputs from graph neural networks (GNNs) via pairwise interactions. In my dissertation, I will first study learning the representations with text attributes for simplicial complexes (RT4SC) via SNNs. Then, I will conduct research on two potential attacks on the representation outputs from SNNs: (1) membership inference attack, which infers whether a certain node of a graph is inside the training data of the GNN model; and (2) graph reconstruction attacks, which infer the confidential edges of a text-attributed network. Finally, I will study a privacy-preserving deterministic differentially private alternating direction method of multiplier to learn secure representation outputs from SNNs that capture multi-scale relationships and facilitate the passage from local structure to global invariant features on text-attributed networks.

----

## [1863] Deep Learning for Medical Prediction in Electronic Health Records

**Authors**: *Xinlu Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26933](https://doi.org/10.1609/aaai.v37i13.26933)

**Abstract**:

The widespread adoption of electronic health records (EHRs) has opened up new opportunities for using deep neural networks to enhance healthcare. However, modeling EHR data can be challenging due to its complex properties, such as missing values, data scarcity in multi-hospital systems, and multimodal irregularity. How to tackle various issues in EHRs for improving medical prediction is challenging and under exploration. I separately illustrate my works to address these issues in EHRs and discuss potential future directions.

----

## [1864] Efficient Algorithms for Regret Minimization in Billboard Advertisement (Student Abstract)

**Authors**: *Dildar Ali, Ankit Kumar Bhagat, Suman Banerjee, Yamuna Prasad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26934](https://doi.org/10.1609/aaai.v37i13.26934)

**Abstract**:

Now-a-days, billboard advertisement has emerged as an effective outdoor advertisement technique. In this case, a commercial house approaches an influence provider for a specific number of views of their advertisement content on a payment basis. If the influence provider can satisfy this then they will receive the full payment else a partial payment. If the influence provider provides more or less than the demand then
certainly this is a loss to them. This is formalized as ‘Regret’
and the goal of the influence provider will be to minimize
the ‘Regret’. In this paper, we propose simple and efficient
solution methodologies to solve this problem. Efficiency and
effectiveness have been demonstrated by experimentation.

----

## [1865] Multi-Horizon Learning in Procedurally-Generated Environments for Off-Policy Reinforcement Learning (Student Abstract)

**Authors**: *Raja Farrukh Ali, Kevin Duong, Nasik Muhammad Nafi, William H. Hsu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26935](https://doi.org/10.1609/aaai.v37i13.26935)

**Abstract**:

Value estimates at multiple timescales can help create advanced discounting functions and allow agents to form more effective predictive models of their environment. In this work, we investigate learning over multiple horizons concurrently for off-policy reinforcement learning by using an advantage-based action selection method and introducing architectural improvements. Our proposed agent learns over multiple horizons simultaneously, while using either exponential or hyperbolic discounting functions. We implement our approach on Rainbow, a value-based off-policy algorithm, and test on Procgen, a collection of procedurally-generated environments, to demonstrate the effectiveness of this approach, specifically to evaluate the agent's performance in previously unseen scenarios.

----

## [1866] Modeling Metacognitive and Cognitive Processes in Data Science Problem Solving (Student Abstract)

**Authors**: *Maryam Alomair, Shimei Pan, Lujie Karen Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26936](https://doi.org/10.1609/aaai.v37i13.26936)

**Abstract**:

Data Science (DS) is an interdisciplinary topic that is applicable to many domains. In this preliminary investigation, we use caselet, a  mini-version of a case study, as a learning tool to allow students to practice data science problem solving (DSPS).  Using a dataset collected from a real-world classroom, we performed correlation analysis to reveal the structure of cognition and metacognition processes. We also explored the similarity of different DS knowledge components based on students’ performance. In addition, we built a predictive model to characterize the relationship between metacognition, cognition, and learning gain.

----

## [1867] Hey, Siri! Why Are You Biased against Women? (Student Abstract)

**Authors**: *Surakshya Aryal, Mikel K. Ngueajio, Saurav Keshari Aryal, Gloria J. Washington*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26937](https://doi.org/10.1609/aaai.v37i13.26937)

**Abstract**:

The intersection of pervasive technology and verbal communication has resulted in the creation of Automatic Speech Recognition Systems (ASRs), which automate the conversion of spontaneous speech into texts. ASR enables human-computer interactions through speech and is rapidly integrated into our daily lives. However, the research studies on current ASR technologies have reported unfulfilled social inclusivity and accentuated biases and stereotypes towards minorities. In this work, we provide a review of examples and evidence to demonstrate preexisting sexist behavior in ASR systems through a systematic review of research literature over the past five years. For each article, we also provide the ASR technology used, highlight specific instances of reported bias, discuss the impact of this bias on the female community, and suggest possible methods of mitigation. We believe this paper will provide insights into the harm that unchecked AI-powered technologies can have on a community by contributing to the growing body of research on this topic and underscoring the need for technological inclusivity for all demographics, especially women.

----

## [1868] FV-Train: Quantum Convolutional Neural Network Training with a Finite Number of Qubits by Extracting Diverse Features (Student Abstract)

**Authors**: *Hankyul Baek, Won Joon Yun, Joongheon Kim*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26938](https://doi.org/10.1609/aaai.v37i13.26938)

**Abstract**:

Quantum convolutional neural network (QCNN) has just become as an emerging research topic as we experience the noisy intermediate-scale quantum (NISQ) era and beyond. As convolutional filters in QCNN extract intrinsic feature using quantum-based ansatz, it should use only finite number of qubits to prevent barren plateaus, and it introduces the lack of the feature information. In this paper, we propose a novel QCNN training algorithm to optimize feature extraction while using only a finite number of qubits, which is called fidelity-variation training (FV-Training).

----

## [1869] PanTop: Pandemic Topic Detection and Monitoring System (Student Abstract)

**Authors**: *Yangxiao Bai, Kaiqun Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26939](https://doi.org/10.1609/aaai.v37i13.26939)

**Abstract**:

Diverse efforts to combat the COVID-19 pandemic have continued throughout the past two years. Governments have announced plans for unprecedentedly rapid vaccine development, quarantine measures, and economic revitalization. They contribute to a more effective pandemic response by determining the precise opinions of individuals regarding these mitigation measures. In this paper, we propose a deep learning-based topic monitoring and storyline extraction system for COVID-19 that is capable of analyzing public sentiment and pandemic trends. The proposed method is able to retrieve Twitter data related to COVID-19 and conduct spatiotemporal analysis. Furthermore, a deep learning component of the system provides monitoring and modeling capabilities for topics based on advanced natural language processing models. A variety of visualization methods are applied to the project to show the distribution of each topic. Our proposed system accurately reflects how public reactions change over time along with pandemic topics.

----

## [1870] Social Intelligence towards Human-AI Teambuilding (Student Abstract)

**Authors**: *Morgan E. Bailey, Frank E. Pollick*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26940](https://doi.org/10.1609/aaai.v37i13.26940)

**Abstract**:

As Artificial Intelligence (AI) continues to develop, it becomes vital to understand more of the nuances of Human-AI interactions. This study aims to uncover how developers can design AI to feel more human in a work environment where only written feedback is possible. Participants will identify a location from Google Maps. To do this successfully, participants must rely on the answers provided by their teammates, one AI and one human. The experiment will run a 2x4 de-sign where AI's responses will either be designed in a human style (high humanness) or state a one-word answer (low humanness), the latter of which is more typical in machines and AI. The reliability of the AI will either be 60% or 90%, and the human will be 30%. Participants will be given a series of questionnaires to rate their opinions of the AI and rate feelings of trust, confidence and performance throughout the study. Following this study, the aim is to identify specific design elements that allow AI to feel human and successfully appear to have social intelligence in more interactive settings.

----

## [1871] Robust Training for AC-OPF (Student Abstract)

**Authors**: *Fuat Can Beylunioglu, Mehrdad Pirnia, P. Robert Duimering, Vijay Ganesh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26941](https://doi.org/10.1609/aaai.v37i13.26941)

**Abstract**:

Electricity network operators use computationally demanding mathematical models to optimize AC power flow (AC-OPF). Recent work applies neural networks (NN) rather than optimization methods to estimate locally optimal solutions. However, NN training data is costly and current models cannot guarantee optimal or feasible solutions. This study proposes a robust NN training approach, which starts with a small amount of seed training data and uses iterative feedback to generate additional data in regions where the model makes poor predictions. The method is applied to non-linear univariate and multivariate test functions, and an IEEE 6-bus AC-OPF system. Results suggest robust training can achieve NN prediction performance similar to, or better than, regular NN training, while using significantly less data.

----

## [1872] IdProv: Identity-Based Provenance for Synthetic Image Generation (Student Abstract)

**Authors**: *Harshil Bhatia, Jaisidh Singh, Gaurav Sangwan, Aparna Bharati, Richa Singh, Mayank Vatsa*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26942](https://doi.org/10.1609/aaai.v37i13.26942)

**Abstract**:

Recent advancements in Generative Adversarial Networks (GANs) have made it possible to obtain high-quality face images of synthetic identities. These networks see large amounts of real faces in order to learn to generate realistic looking synthetic images. However, the concept of a synthetic identity for these images is not very well-defined. In this work, we verify identity leakage from the training set containing real images into the latent space and propose a novel method, IdProv, that uses image composition to trace the source of identity signals in the generated image.

----

## [1873] Latent Space Evolution under Incremental Learning with Concept Drift (Student Abstract)

**Authors**: *Charles Bourbeau, Audrey Durand*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26943](https://doi.org/10.1609/aaai.v37i13.26943)

**Abstract**:

This work investigates the evolution of latent space when deep learning models are trained incrementally in non-stationary environments that stem from concept drift. We propose a methodology for visualizing the incurred change in latent representations. We further show that classes not targeted by concept drift can be negatively affected, suggesting that the observation of all classes during learning may regularize the latent space.

----

## [1874] Model Selection of Graph Signage Models Using Maximum Likelihood (Student Abstract)

**Authors**: *Angelina Brilliantova, Ivona Bezáková*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26944](https://doi.org/10.1609/aaai.v37i13.26944)

**Abstract**:

Complex systems across various domains can be naturally modeled as signed networks with positive and negative edges. In this work, we design a new class of signage models and show how to select the model parameters that best fit real-world datasets using maximum likelihood.

----

## [1875] Optimal Execution via Multi-Objective Multi-Armed Bandits (Student Abstract)

**Authors**: *Francois Buet-Golfouse, Peter Hill*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26945](https://doi.org/10.1609/aaai.v37i13.26945)

**Abstract**:

When trying to liquidate a large quantity of a particular stock, the price of that stock is likely to be affected by trades, thus leading to a reduced expected return if we were to sell the entire quantity at once. This leads to the problem of optimal execution, where the aim is to split the sell order into several smaller sell orders over the course of a period of time, to optimally balance stock price with market risk. This problem can be defined in terms of difference equations. Here, we show how we can reformulate this as a multi-objective problem, which we solve with a novel multi-armed bandit algorithm.

----

## [1876] Lightweight Transformer for Multi-Modal Object Detection (Student Abstract)

**Authors**: *Yue Cao, Yanshuo Fan, Junchi Bin, Zheng Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26946](https://doi.org/10.1609/aaai.v37i13.26946)

**Abstract**:

It has become a common practice for many perceptual systems to integrate information from multiple sensors to improve the accuracy of object detection. For example, autonomous vehicles use visible light, and infrared (IR) information to ensure that the car can cope with complex weather conditions. However, the accuracy of the algorithm is usually a trade-off between the computational complexity and memory consumption. In this study, we evaluate the performance and complexity of different fusion operators in multi-modal object detection tasks. On top of that, a Poolformer-based fusion operator (PoolFuser) is proposed to enhance the accuracy of detecting targets without compromising the efficiency of the detection framework.

----

## [1877] Reconsidering Deception in Social Robotics: The Role of Human Vulnerability (Student Abstract)

**Authors**: *Rachele Carli, Amro Najjar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26947](https://doi.org/10.1609/aaai.v37i13.26947)

**Abstract**:

The literature on deception in human-robot interaction (henceforth HRI) could be divided between: (i) those who consider it essential to maximise users' end utility and robotic performance; (ii) those who consider it unethical, because it is potentially dangerous for individuals' psychological integrity. 
However, it has now been proven that humans are naturally prone to anthropomorphism and emotional attachment to inanimate objects. 
Consequently, despite ethical concerns, the argument for the total elimination of deception could reveal to be a pointless exercise.
Rather, it is suggested here to conceive deception in HRI as a dynamic to be modulated and graded, in order to both promote innovation and protect fundamental human rights. To this end, the concept of vulnerability could serve as an objective balancing criterion.

----

## [1878] Know Your Enemy: Identifying Adversarial Behaviours in Deep Reinforcement Learning Agents (Student Abstract)

**Authors**: *Seán Caulfield Curley, Karl Mason, Patrick Mannion*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26948](https://doi.org/10.1609/aaai.v37i13.26948)

**Abstract**:

It has been shown that an agent can be trained with an adversarial policy which achieves high degrees of success against a state-of-the-art DRL victim despite taking unintuitive actions. This prompts the question: is this adversarial behaviour detectable through the observations of the victim alone? We find that widely used classification methods such as random forests are only able to achieve a maximum of ≈71% test set accuracy when classifying an agent for a single timestep. However, when the classifier inputs are treated as time-series data, test set classification accuracy is increased significantly to ≈98%. This is true for both classification of episodes as a whole, and for “live” classification at each timestep in an episode. These classifications can then be used to “react” to incoming attacks and increase the overall win rate against Adversarial opponents by approximately 17%. Classification of the victim’s own internal activations in response to the adversary is shown to achieve similarly impressive accuracy while also offering advantages like increased transferability to other domains.

----

## [1879] An Emotion-Guided Approach to Domain Adaptive Fake News Detection Using Adversarial Learning (Student Abstract)

**Authors**: *Arkajyoti Chakraborty, Inder Khatri, Arjun Choudhry, Pankaj Gupta, Dinesh Kumar Vishwakarma, Mukesh Prasad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26949](https://doi.org/10.1609/aaai.v37i13.26949)

**Abstract**:

Recent works on fake news detection have shown the efficacy of using emotions as a feature for improved performance. However, the cross-domain impact of emotion-guided features for fake news detection still remains an open problem. In this work, we propose an emotion-guided, domain-adaptive, multi-task approach for cross-domain fake news detection, proving the efficacy of emotion-guided models in cross-domain settings for various datasets.

----

## [1880] Deep Anomaly Detection and Search via Reinforcement Learning (Student Abstract)

**Authors**: *Chao Chen, Dawei Wang, Feng Mao, Zongzhang Zhang, Yang Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26950](https://doi.org/10.1609/aaai.v37i13.26950)

**Abstract**:

Semi-supervised anomaly detection is a data mining task which aims at learning features from partially-labeled datasets. We propose Deep Anomaly Detection and Search (DADS) with reinforcement learning. During the training process, the agent searches for possible anomalies in unlabeled dataset to enhance performance. Empirically, we compare DADS with several methods in the settings of leveraging known anomalies to detect both other known and unknown anomalies. Results show that DADS achieves good performance.

----

## [1881] Towards Deployment-Efficient and Collision-Free Multi-Agent Path Finding (Student Abstract)

**Authors**: *Feng Chen, Chenghe Wang, Fuxiang Zhang, Hao Ding, Qiaoyong Zhong, Shiliang Pu, Zongzhang Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26951](https://doi.org/10.1609/aaai.v37i13.26951)

**Abstract**:

Multi-agent pathfinding (MAPF) is essential to large-scale robotic coordination tasks. Planning-based algorithms show their advantages in collision avoidance while avoiding exponential growth in the number of agents. Reinforcement-learning (RL)-based algorithms can be deployed efficiently but cannot prevent collisions entirely due to the lack of hard constraints. This paper combines the merits of planning-based and RL-based MAPF methods to propose a deployment-efficient and collision-free MAPF algorithm. The experiments show the effectiveness of our approach.

----

## [1882] SkateboardAI: The Coolest Video Action Recognition for Skateboarding (Student Abstract)

**Authors**: *Hanxiao Chen*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26952](https://doi.org/10.1609/aaai.v37i13.26952)

**Abstract**:

Impressed by the coolest skateboarding sports program from 2021 Tokyo Olympic Games, we are the first to curate the original real-world video datasets "SkateboardAI" in the wild, even self-design and implement diverse uni-modal and multi-modal video action recognition approaches to recognize different tricks accurately. For uni-modal methods, we separately apply (1)CNN and LSTM; (2)CNN and BiLSTM; (3)CNN and BiLSTM with effective attention mechanisms; (4)Transformer-based action recognition pipeline. Transferred to the multi-modal conditions, we investigated the two-stream Inflated-3D architecture on "SkateboardAI" datasets to compare its performance with uni-modal cases. In sum, our objective is developing an excellent AI sport referee for the coolest skateboarding competitions.

----

## [1883] AsT: An Asymmetric-Sensitive Transformer for Osteonecrosis of the Femoral Head Detection (Student Abstract)

**Authors**: *Haoyang Chen, Shuai Liu, Feng Lu, Wei Li, Bin Sheng, Mi Li, Hai Jin, Albert Y. Zomaya*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26953](https://doi.org/10.1609/aaai.v37i13.26953)

**Abstract**:

Early diagnosis of osteonecrosis of the femoral head (ONFH) can inhibit the progression and improve femoral head preservation. The radiograph difference between early ONFH and healthy ones is not apparent to the naked eye. It is also hard to produce a large dataset to train the classification model. In this paper, we propose Asymmetric-Sensitive Transformer (AsT) to capture the uneven development of the bilateral femoral head to enable robust ONFH detection. Our ONFH detection is realized using the self-attention mechanism to femoral head regions while conferring sensitivity to the uneven development by the attention-shared transformer. The real-world experiment studies show that AsT achieves the best performance of AUC 0.9313 in the early diagnosis of ONFH and can find out misdiagnosis cases firmly.

----

## [1884] Self-Paced Learning Based Graph Convolutional Neural Network for Mixed Integer Programming (Student Abstract)

**Authors**: *Li Chen, Hua Xu, Ziteng Wang, Chengming Wang, Yu Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26954](https://doi.org/10.1609/aaai.v37i13.26954)

**Abstract**:

Graph convolutional neural network (GCN) based methods have achieved noticeable performance in solving mixed integer programming problems (MIPs). However, the generalization of existing work is limited due to the problem structure. This paper proposes a self-paced learning (SPL) based GCN network (SPGCN) with curriculum learning (CL) to make the utmost of samples. SPGCN employs a GCN model to imitate the branching variable selection during the branch and bound process, while the training process is conducted in a self-paced fashion. Specifically, SPGCN contains a loss-based automatic difficulty measurer, where the training loss of the sample represents the difficulty level. In each iteration, a dynamic training dataset is constructed according to the difficulty level for GCN model training. Experiments on four NP-hard datasets verify that CL can lead to generalization improvement and convergence speedup in solving MIPs, where SPL performs better than predefined CL methods.

----

## [1885] Multi-Modal Protein Knowledge Graph Construction and Applications (Student Abstract)

**Authors**: *Siyuan Cheng, Xiaozhuan Liang, Zhen Bi, Huajun Chen, Ningyu Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26955](https://doi.org/10.1609/aaai.v37i13.26955)

**Abstract**:

Existing data-centric methods for protein science generally cannot sufficiently capture and leverage biology knowledge, which may be crucial for many protein tasks. To facilitate research in this field, we create ProteinKG65, a knowledge graph for protein science. Using gene ontology and Uniprot knowledge base as a basis, we transform and integrate various kinds of knowledge with aligned descriptions and protein sequences, respectively, to GO terms and protein entities. ProteinKG65 is mainly dedicated to providing a specialized protein knowledge graph, bringing the knowledge of Gene Ontology to protein function and structure prediction. We also illustrate the potential applications of ProteinKG65 with a prototype. Our dataset can be downloaded at  https://w3id.org/proteinkg65.

----

## [1886] CasODE: Modeling Irregular Information Cascade via Neural Ordinary Differential Equations (Student Abstract)

**Authors**: *Zhangtao Cheng, Xovee Xu, Ting Zhong, Fan Zhou, Goce Trajcevski*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26956](https://doi.org/10.1609/aaai.v37i13.26956)

**Abstract**:

Predicting information cascade popularity is a fundamental problem for understanding the nature of information propagation on social media. However, existing works fail to capture an essential aspect of information propagation: the temporal irregularity of cascade event -- i.e., users' re-tweetings at random and non-periodic time instants. In this work, we present a novel framework CasODE for information cascade prediction with neural ordinary differential equations (ODEs). CasODE generalizes the discrete state transitions in RNNs to continuous-time dynamics for modeling the irregular-sampled events in information cascades. Experimental evaluations on real-world datasets demonstrate the advantages of the CasODE over baseline approaches.

----

## [1887] SR-AnoGAN: You Never Detect Alone Super Resolution in Anomaly Detection (Student Abstract)

**Authors**: *Minjong Cheon*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26957](https://doi.org/10.1609/aaai.v37i13.26957)

**Abstract**:

Despite the advance in deep learning algorithms, implementing supervised learning algorithms in medical datasets is difficult owing to the medical data's properties. This paper proposes SR-AnoGAN, which could generate higher resolution images and conduct anomaly detection more efficiently than AnoGAN. The most distinctive part of the proposed model is incorporating CNN and SRGAN into AnoGAN for reconstructing high-resolution images. Experimental results from X-ray datasets(pneumonia, covid-19) verify that the SR-AnoGAN outperforms the previous AnoGAN model through qualitative and quantitative approaches. Therefore, this paper shows the possibility of resolving data imbalance problems prevalent in the medical field, and proposing more precise diagnosis.

----

## [1888] Transformer-Based Named Entity Recognition for French Using Adversarial Adaptation to Similar Domain Corpora (Student Abstract)

**Authors**: *Arjun Choudhry, Pankaj Gupta, Inder Khatri, Aaryan Gupta, Maxime Nicol, Marie-Jean Meurs, Dinesh Kumar Vishwakarma*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26958](https://doi.org/10.1609/aaai.v37i13.26958)

**Abstract**:

Named Entity Recognition (NER) involves the identification and classification of named entities in unstructured text into predefined classes. NER in languages with limited resources, like French, is still an open problem due to the lack of large, robust, labelled datasets. In this paper, we propose a transformer-based NER approach for French using adversarial adaptation to similar domain or general corpora for improved feature extraction and better generalization. We evaluate our approach on three labelled datasets and show that our adaptation framework outperforms the corresponding non-adaptive models for various combinations of transformer models, source datasets and target corpora.

----

## [1889] Disentangling the Benefits of Self-Supervised Learning to Deployment-Driven Downstream Tasks of Satellite Images (Student Abstract)

**Authors**: *Zhuo Deng, Yibing Wei, Mingye Zhu, Xueliang Wang, Junchi Zhou, Zhicheng Yang, Hang Zhou, Zhenjie Cao, Lan Ma, Mei Han, Jui-Hsin Lai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26959](https://doi.org/10.1609/aaai.v37i13.26959)

**Abstract**:

In this paper, we investigate the benefits of self-supervised learning (SSL) to downstream tasks of satellite images. Unlike common student academic projects, this work focuses on the advantages of the SSL for deployment-driven tasks which have specific scenarios with low or high-spatial resolution images. Our preliminary experiments demonstrate the robust benefits of the SSL trained by medium-resolution (10m) images to both low-resolution (100m) scene classification case (4.25%↑) and very high-resolution (5cm) aerial image segmentation case (1.96%↑), respectively.

----

## [1890] Performance Disparities between Accents in Automatic Speech Recognition (Student Abstract)

**Authors**: *Alex DiChristofano, Henry Shuster, Shefali Chandra, Neal Patwari*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26960](https://doi.org/10.1609/aaai.v37i13.26960)

**Abstract**:

In this work, we expand the discussion of bias in Automatic Speech Recognition (ASR) through a large-scale audit. Using a large and global data set of speech, we perform an audit of some of the most popular English ASR services. We show that, even when controlling for multiple linguistic covariates, ASR service performance has a statistically significant relationship to the political alignment of the speaker's birth country with respect to the United States' geopolitical power.

----

## [1891] Demystify the Gravity Well in the Optimization Landscape (Student Abstract)

**Authors**: *Jason Xiaotian Dou, Runxue Bao, Susan Song, Shuran Yang, Yanfu Zhang, Paul Pu Liang, Haiyi Harry Mao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26961](https://doi.org/10.1609/aaai.v37i13.26961)

**Abstract**:

We provide both empirical and theoretical insights to demystify the gravity well phenomenon in the optimization landscape. We start from describe the problem setup and theoretical results (an escape time lower bound) of the Softmax Gravity Well (SGW) in the literature. Then we move toward the understanding of a recent observation called ASR gravity well. We provide an explanation of why normal distribution with high variance can lead to suboptimal plateaus from an energy function point of view. We also contribute to the empirical insights of curriculum learning by comparison of policy initialization by different normal distributions. Furthermore, we provide the ASR escape time lower bound to understand the ASR gravity well theoretically. Future work includes more specific modeling of the reward as a function of time and quantitative evaluation of normal distribution’s influence on policy initialization.

----

## [1892] AlphaSnake: Policy Iteration on a Nondeterministic NP-Hard Markov Decision Process (Student Abstract)

**Authors**: *Kevin Du, Ian Gemp, Yi Wu, Yingying Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26962](https://doi.org/10.1609/aaai.v37i13.26962)

**Abstract**:

Reinforcement learning has been used to approach well-known NP-hard combinatorial problems in graph theory. Among these, Hamiltonian cycle problems are exceptionally difficult to analyze, even when restricted to individual instances of structurally complex graphs. In this paper, we use Monte Carlo Tree Search (MCTS), the search algorithm behind many state-of-the-art reinforcement learning algorithms such as AlphaZero, to create autonomous agents that learn to play the game of Snake, a game centered on properties of Hamiltonian cycles on grid graphs. The game of Snake can be formulated as a single-player discounted Markov Decision Process (MDP), where the agent must behave optimally in a stochastic environment. Determining the optimal policy for Snake, defined as the policy that maximizes the probability of winning -- or win rate -- with higher priority and minimizes the expected number of time steps to win with lower priority, is conjectured to be NP-hard. Performance-wise, compared to prior work in the Snake game, our algorithm is the first to achieve a win rate over 0.5 (a uniform random policy achieves a win rate < 2.57 x 10^{-15}), demonstrating the versatility of AlphaZero in tackling NP-hard problems.

----

## [1893] Transformer-Based Multi-Hop Question Generation (Student Abstract)

**Authors**: *John Emerson, Yllias Chali*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26963](https://doi.org/10.1609/aaai.v37i13.26963)

**Abstract**:

Question generation is the parallel task of question answering, where given an input context and, optionally, an answer, the goal is to generate a relevant and fluent natural language question. Although recent works on question generation have experienced success by utilizing sequence-to-sequence models, there is a need for question generation models to handle increasingly complex input contexts to produce increasingly detailed questions. Multi-hop question generation is a more challenging task that aims to generate questions by connecting multiple facts from multiple input contexts. In this work, we apply a transformer model to the task of multi-hop question generation without utilizing any sentence-level supporting fact information. We utilize concepts that have proven effective in single-hop question generation, including a copy mechanism and placeholder tokens. We evaluate our model’s performance on the HotpotQA dataset using automated evaluation metrics, including BLEU, ROUGE and METEOR and show an improvement over the previous work.

----

## [1894] eCDANs: Efficient Temporal Causal Discovery from Autocorrelated and Non-stationary Data (Student Abstract)

**Authors**: *Muhammad Hasan Ferdous, Uzma Hasan, Md. Osman Gani*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26964](https://doi.org/10.1609/aaai.v37i13.26964)

**Abstract**:

Conventional temporal causal discovery (CD) methods suffer from high dimensionality, fail to identify lagged causal relationships, and often ignore dynamics in relations. In this study, we present a novel constraint-based CD approach for autocorrelated and non-stationary time series data (eCDANs) capable of detecting lagged and contemporaneous causal relationships along with temporal changes. eCDANs addresses high dimensionality by optimizing the conditioning sets while conducting conditional independence (CI) tests and identifies the changes in causal relations by introducing a surrogate variable to represent time dependency. Experiments on synthetic and real-world data show that eCDANs can identify time influence and outperform the baselines.

----

## [1895] LEAN-DMKDE: Quantum Latent Density Estimation for Anomaly Detection (Student Abstract)

**Authors**: *Joseph A. Gallego-Mejia, Oscar A. Bustos-Brinez, Fabio A. González*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26965](https://doi.org/10.1609/aaai.v37i13.26965)

**Abstract**:

This paper presents an anomaly detection model that combines the strong statistical foundation of density-estimation-based anomaly detection methods with the representation-learning ability of deep-learning models. The method combines an autoencoder, that learns a low-dimensional representation of the data, with a density-estimation model based on density matrices in an end-to-end architecture that can be trained using gradient-based optimization techniques. A systematic experimental evaluation was performed on different benchmark datasets. The experimental results show that the method is able to outperform other state-of-the-art methods.

----

## [1896] Safety Aware Neural Pruning for Deep Reinforcement Learning (Student Abstract)

**Authors**: *Briti Gangopadhyay, Pallab Dasgupta, Soumyajit Dey*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26966](https://doi.org/10.1609/aaai.v37i13.26966)

**Abstract**:

Neural network pruning is a technique of network compression by removing weights of lower importance from an optimized neural network. Often, pruned networks are compared
in terms of accuracy, which is realized in terms of rewards for Deep Reinforcement Learning (DRL) networks. However, networks that estimate control actions for safety-critical tasks, must also adhere to safety requirements along with obtaining rewards. We propose a methodology to iteratively refine the weights of a pruned neural network such that we get a sparse high-performance network without significant side effects on safety.

----

## [1897] Towards Fair and Selectively Privacy-Preserving Models Using Negative Multi-Task Learning (Student Abstract)

**Authors**: *Liyuan Gao, Huixin Zhan, Austin Chen, Victor S. Sheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26967](https://doi.org/10.1609/aaai.v37i13.26967)

**Abstract**:

Deep learning models have shown great performances in natural language processing tasks. While much attention has been paid to improvements in utility, privacy leakage and social bias are two major concerns arising in trained models. In order to tackle these problems, we protect individuals' sensitive information and mitigate gender bias simultaneously. First, we propose a selective privacy-preserving method that only obscures individuals' sensitive information. Then we propose a negative multi-task learning framework to mitigate the gender bias which contains a main task and a gender prediction task.  We analyze two existing word embeddings and evaluate them on sentiment analysis and a medical text classification task. Our experimental results show that our negative multi-task learning framework can mitigate the gender bias while keeping models’ utility.

----

## [1898] Towards Safe Reinforcement Learning via OOD Dynamics Detection in Autonomous Driving System (Student Abstract)

**Authors**: *Arnaud Gardille, Ola Ahmad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26968](https://doi.org/10.1609/aaai.v37i13.26968)

**Abstract**:

Deep reinforcement learning (DRL) has proven effective in training agents to achieve goals in complex environments. However, a trained RL agent may exhibit, during deployment, unexpected behavior when faced with a situation where its state transitions differ even slightly from the training environment. Such a situation can arise for a variety of reasons. Rapid and accurate detection of anomalous behavior appears to be a prerequisite for using DRL in safety-critical systems, such as autonomous driving. We propose a novel OOD detection algorithm based on modeling the transition function of the training environment. Our method captures the bias of model behavior when encountering subtle changes of dynamics while maintaining a low false positive rate. Preliminary evaluations on the realistic simulator CARLA corroborate the relevance of our proposed method.

----

## [1899] Neural Implicit Surface Reconstruction from Noisy Camera Observations (Student Abstract)

**Authors**: *Sarthak Gupta, Patrik Huber*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26969](https://doi.org/10.1609/aaai.v37i13.26969)

**Abstract**:

Representing 3D objects and scenes with neural radiance fields has become very popular over the last years. Recently, surface-based representations have been proposed, that allow to reconstruct 3D objects from simple photographs. However, most current techniques require an accurate camera calibration, i.e. camera parameters corresponding to each image, which is often a difficult task to do in real-life situations. To this end, we propose a method for learning 3D surfaces from noisy camera parameters. We show that we can learn camera parameters together with learning the surface representation, and demonstrate good quality 3D surface reconstruction even with noisy camera observations.

----

## [1900] Expert Data Augmentation in Imitation Learning (Student Abstract)

**Authors**: *Fuguang Han, Zongzhang Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26970](https://doi.org/10.1609/aaai.v37i13.26970)

**Abstract**:

Behavioral Cloning (BC) is a simple and effective imitation learning algorithm, which suffers from compounding error due to covariate shift. One solution is to use enough data for training. However, the amount of expert demonstrations available is usually limited. So we propose an effective method to augment expert demonstrations to alleviate the problem of compounding error in BC. It operates by estimating the similarity of states and filtering out transitions that can go back to the states similar to ones in expert demonstrations during the process of sampling. The data filtered out along with original expert demonstrations are used for training. We evaluate the performance of our method on several Atari tasks and continuous MuJoCo control tasks. Empirically, BC trained with the augmented data significantly outperform BC trained with the original expert demonstrations.

----

## [1901] Unsupervised Contrastive Representation Learning for 3D Mesh Segmentation (Student Abstract)

**Authors**: *Ayaan Haque, Hankyu Moon, Heng Hao, Sima Didari, Jae Oh Woo, Patrick Bangert*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26971](https://doi.org/10.1609/aaai.v37i13.26971)

**Abstract**:

3D deep learning is a growing field of interest due to the vast amount of information stored in 3D formats. Triangular meshes are an efficient representation for irregular, non-uniform 3D objects. However, meshes are often challenging to annotate due to their high computational complexity. Therefore, it is desirable to train segmentation networks with limited-labeled data. Self-supervised learning (SSL), a form of unsupervised representation learning, is a growing alternative to fully-supervised learning which can decrease the burden of supervision for training. Specifically, contrastive learning (CL), a form of SSL, has recently been explored to solve limited-labeled data tasks. We propose SSL-MeshCNN, a CL method for pre-training CNNs for mesh segmentation. We take inspiration from prior CL frameworks to design a novel CL algorithm specialized for meshes. Our preliminary experiments show promising results in reducing the heavy labeled data requirement needed for mesh segmentation by at least 33%.

----

## [1902] Invertible Conditional GAN Revisited: Photo-to-Manga Face Translation with Modern Architectures (Student Abstract)

**Authors**: *Taro Hatakeyama, Ryusuke Saito, Komei Hiruta, Atsushi Hashimoto, Satoshi Kurihara*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26972](https://doi.org/10.1609/aaai.v37i13.26972)

**Abstract**:

Recent style translation methods have extended their transferability from texture to geometry. However, performing translation while preserving image content when there is a significant style difference is still an open problem. To overcome this problem, we propose Invertible Conditional Fast GAN (IcFGAN) based on GAN inversion and cFGAN. It allows for unpaired photo-to-manga face translation. Experimental results show that our method could translate styles under significant style gaps, while the state-of-the-art methods could hardly preserve image content.

----

## [1903] Exploring Hypergraph of Earnings Call for Risk Prediction (Student Abstract)

**Authors**: *Yi He, Wenxin Tai, Fan Zhou, Yi Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26973](https://doi.org/10.1609/aaai.v37i13.26973)

**Abstract**:

In financial economics, studies have shown that the textual content in the earnings conference call transcript has predictive power for a firm's future risk. However, the conference call transcript is very long and contains diverse non-relevant content, which poses challenges for the text-based risk forecast. This study investigates the structural dependency within a conference call transcript by explicitly modeling the dialogue between managers and analysts. Specifically, we utilize TextRank to extract information and exploit the semantic correlation within a discussion using hypergraph learning. This novel design can improve the transcript representation performance and reduce the risk of forecast errors. Experimental results on a large-scale dataset show that our approach can significantly improve prediction performance compared to state-of-the-art text-based models.

----

## [1904] An Analysis of the Deliberation and Task Performance of an Active Logic Based Agent (Student Abstract)

**Authors**: *Anthony Herron, Darsana P. Josyula*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26974](https://doi.org/10.1609/aaai.v37i13.26974)

**Abstract**:

Active logic is a time-situated reasoner that can track the history of inferences, detect contradictions, and make parallel inferences in time. In this paper, we explore the behavior of an active-logic based agent on different sets of action selection axioms for a time-constrained target search task. We compare the performance of a baseline set of axioms that does not avoid redundant actions with five other axiom sets that avoid repeated actions but vary in their knowledge content. The results of these experiments show the importance of balancing boldness and caution for target search.

----

## [1905] Mobility Prediction via Sequential Trajectory Disentanglement (Student Abstract)

**Authors**: *Jinyu Hong, Fan Zhou, Qiang Gao, Ping Kuang, Kunpeng Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26975](https://doi.org/10.1609/aaai.v37i13.26975)

**Abstract**:

Accurately predicting human mobility is a critical task in location-based recommendation. Most prior approaches focus on fusing multiple semantics trajectories to forecast the future movement of people, and fail to consider the distinct relations in underlying context of human mobility, resulting in a narrow perspective to comprehend human motions. Inspired by recent advances in disentanglement learning, we propose a novel self-supervised method called SelfMove for next POI prediction. SelfMove seeks to disentangle the potential time-invariant and time-varying factors from massive trajectories, which provides an interpretable view to understand the complex semantics underlying human mobility representations. To address the data sparsity issue, we present two realistic trajectory augmentation approaches to help understand the intrinsic periodicity and constantly changing intents of humans. In addition, a POI-centric graph structure is proposed to explore both homogeneous and heterogeneous collaborative signals behind historical trajectories. Experiments on two real-world datasets demonstrate the superiority of SelfMove compared to the state-of-the-art baselines.

----

## [1906] A Reinforcement Learning Badminton Environment for Simulating Player Tactics (Student Abstract)

**Authors**: *Li-Chun Huang, Nai-Zen Hseuh, Yen-Che Chien, Wei-Yao Wang, Kuang-Da Wang, Wen-Chih Peng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26976](https://doi.org/10.1609/aaai.v37i13.26976)

**Abstract**:

Recent techniques for analyzing sports precisely has stimulated various approaches to improve player performance and fan engagement.
However, existing approaches are only able to evaluate offline performance since testing in real-time matches requires exhaustive costs and cannot be replicated.
To test in a safe and reproducible simulator, we focus on turn-based sports and introduce a badminton environment by simulating rallies with different angles of view and designing the states, actions, and training procedures.
This benefits not only coaches and players by simulating past matches for tactic investigation, but also researchers from rapidly evaluating their novel algorithms.
Our code is available at https://github.com/wywyWang/CoachAI-Projects/tree/main/Strategic%20Environment.

----

## [1907] Less Is More: Volatility Forecasting with Contrastive Representation Learning (Student Abstract)

**Authors**: *Yanlong Huang, Wenxin Tai, Ting Zhong, Kunpeng Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26977](https://doi.org/10.1609/aaai.v37i13.26977)

**Abstract**:

Earnings conference calls are indicative information events for volatility forecasting, which is essential for financial risk management and asset pricing. Although recent volatility forecasting models have explored the textual content of conference calls for prediction, they suffer from modeling the long-text and representing the risk-relevant information. This work proposes to identify key sentences for robust and interpretable transcript representation learning based on the cognitive theory. Specifically, we introduce TextRank to find key sentences and leverage attention mechanism to screen out the candidates by modeling the semantic correlations. Upon on the structural information of earning conference calls, we propose a structure-based contrastive learning method to facilitate the effective transcript representation. Empirical results on the benchmark dataset demonstrate the superiority of our model over competitive baselines in volatility forecasting.

----

## [1908] Understand Restart of SAT Solver Using Search Similarity Index (Student Abstract)

**Authors**: *Yoichiro Iida, Tomohiro Sonobe, Mary Inaba*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26978](https://doi.org/10.1609/aaai.v37i13.26978)

**Abstract**:

SAT solvers are widely used to solve many industrial problems because of their high performance, which is achieved by various heuristic methods.
Understanding why these methods are effective is essential to improving them. One approach to this is analyzing them using qualitative measurements.
In our previous study, we proposed search similarity index (SSI), a metric to quantify the similarity between searches. SSI significantly improved the performance of the parallel SAT solver.
Here, we apply SSI to analyze the effect of restart, a key SAT solver technique.
Experiments using SSI reveal the correlation between the difficulty of instances and the search change effect by restart, and the reason behind the effectiveness of the state-of-the-art restart method is also explained.

----

## [1909] In-Game Toxic Language Detection: Shared Task and Attention Residuals (Student Abstract)

**Authors**: *Yuanzhe Jia, Weixuan Wu, Feiqi Cao, Soyeon Caren Han*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26979](https://doi.org/10.1609/aaai.v37i13.26979)

**Abstract**:

In-game toxic language becomes the hot potato in the gaming industry and community. There have been several online game toxicity analysis frameworks and models proposed. However, it is still challenging to detect toxicity due to the nature of in-game chat, which has extremely short length. In this paper, we describe how the in-game toxic language shared task has been established using the real-world in-game chat data. In addition, we propose and introduce the model/framework for toxic language token tagging (slot filling) from the in-game chat. The data and code will be released.

----

## [1910] CKS: A Community-Based K-shell Decomposition Approach Using Community Bridge Nodes for Influence Maximization (Student Abstract)

**Authors**: *Inder Khatri, Aaryan Gupta, Arjun Choudhry, Aryan Tyagi, Dinesh Kumar Vishwakarma, Mukesh Prasad*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26980](https://doi.org/10.1609/aaai.v37i13.26980)

**Abstract**:

Social networks have enabled user-specific advertisements and recommendations on their platforms, which puts a significant focus on Influence Maximisation (IM) for target advertising and related tasks. The aim is to identify nodes in the network which can maximize the spread of information through a diffusion cascade. We propose a community structures-based approach that employs K-Shell algorithm with community structures to generate a score for the connections between seed nodes and communities. Further, our approach employs entropy within communities to ensure the proper spread of information within the communities. We validate our approach on four publicly available networks and show its superiority to four state-of-the-art approaches while still being relatively efficient.

----

## [1911] Incremental Density-Based Clustering with Grid Partitioning (Student Abstract)

**Authors**: *Jeong-Hun Kim, Tserenpurev Chuluunsaikhan, Jong-Hyeok Choi, Aziz Nasridinov*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26981](https://doi.org/10.1609/aaai.v37i13.26981)

**Abstract**:

DBSCAN is widely used in various fields, but it requires computational costs similar to those of re-clustering from scratch to update clusters when new data is inserted. To solve this, we propose an incremental density-based clustering method that rapidly updates clusters by identifying in advance regions where cluster updates will occur. Also, through extensive experiments, we show that our method provides clustering results similar to those of DBSCAN.

----

## [1912] Sequential Graph Attention Learning for Predicting Dynamic Stock Trends (Student Abstract)

**Authors**: *Tzu-Ya Lai, Wen Jung Cheng, Jun-En Ding*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26982](https://doi.org/10.1609/aaai.v37i13.26982)

**Abstract**:

The stock market is characterized by a complex relationship between companies and the market. This study combines a sequential graph structure with attention mechanisms to learn global and local information within temporal time. Specifically, our proposed “GAT-AGNN” module compares model performance across multiple industries as well as within single industries. The results show that the proposed framework outperforms the state-of-the-art methods in predicting stock trends across multiple industries on Taiwan Stock datasets.

----

## [1913] Mitigating Negative Transfer in Multi-Task Learning with Exponential Moving Average Loss Weighting Strategies (Student Abstract)

**Authors**: *Anish Lakkapragada, Essam Sleiman, Saimourya Surabhi, Dennis P. Wall*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26983](https://doi.org/10.1609/aaai.v37i13.26983)

**Abstract**:

Multi-Task Learning (MTL) is a growing subject of interest in deep learning, due to its ability to train models more efficiently on multiple tasks compared to using a group of conventional single-task models. However, MTL can be impractical as certain tasks can dominate training and hurt performance in others, thus making some tasks perform better in a single-task model compared to a multi-task one. Such problems are broadly classified as negative transfer, and many prior approaches in the literature have been made to mitigate these issues. One such current approach to alleviate negative transfer is to weight each of the losses so that they are on the same scale. Whereas current loss balancing approaches rely on either optimization or complex numerical analysis, none directly scale the losses based on their observed magnitudes. We propose multiple techniques for loss balancing based on scaling by the exponential moving average and benchmark them against current best-performing methods on three established datasets. On these datasets, they achieve comparable, if not higher, performance compared to current best-performing methods.

----

## [1914] A Federated Learning Monitoring Tool for Self-Driving Car Simulation (Student Abstract)

**Authors**: *Taejoon Lee, Hyunsu Mun, Youngseok Lee*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26984](https://doi.org/10.1609/aaai.v37i13.26984)

**Abstract**:

We propose CARLA-FLMon, which can monitor the progress of running federated learning (FL) training in the open-source autonomous driving simulation software, CARLA. The purpose of CARLA-FLMon is to visually present the status and results of federated learning training, and to provide an extensible FL training environment with which FL training can be performed repeatedly with updated learning strategies through analysis. With CARLA-FLMon, we can determine what factors have positive or negative influences on learning by visualizing training data. Then, we can optimize the parameters of the FL training model to improve the accuracy of FL. With preliminary experiments of CARLA-FLMon on lane recognition, we demonstrate that CARLA-FLmon can increase the overall accuracy from 80.33% to 93.82% by identifying lowly-contributing clients and excluding them.

----

## [1915] Summarization Attack via Paraphrasing (Student Abstract)

**Authors**: *Jiyao Li, Wei Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26985](https://doi.org/10.1609/aaai.v37i13.26985)

**Abstract**:

Many natural language processing models are perceived to be fragile on adversarial attacks. Recent work on adversarial attack has demonstrated a high success rate on sentiment analysis as well as classification models. However, attacks to summarization models have not been well studied. Summarization tasks are rarely influenced by word substitution, since advanced abstractive summary models utilize sentence level information. In this paper, we propose a paraphrasing-based attack method to attack summarization models. We first rank the sentences in the document according to their impacts to summarization. Then, we apply paraphrasing procedure to generate adversarial samples. Finally, we test our algorithm on benchmarks datasets against others methods. Our approach achieved the highest success rate and the lowest sentence substitution rate. In addition, the adversarial samples have high semantic similarity with the original sentences.

----

## [1916] Evaluating Robustness of Vision Transformers on Imbalanced Datasets (Student Abstract)

**Authors**: *Kevin Li, Rahul Duggal, Duen Horng Chau*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26986](https://doi.org/10.1609/aaai.v37i13.26986)

**Abstract**:

Data in the real world is commonly imbalanced across classes. Training neural networks on imbalanced datasets often leads to poor performance on rare classes. Existing work in this area has primarily focused on Convolution Neural Networks (CNN), which are increasingly being replaced by Self-Attention-based Vision Transformers (ViT). Fundamentally, ViTs differ from CNNs in that they offer the flexibility in learning the appropriate inductive bias conducive to improving performance. This work is among the first to evaluate the performance of ViTs under class imbalance. We find that accuracy degradation in the presence of class imbalance is much more prominent in ViTs compared to CNNs. This degradation can be partially mitigated through loss reweighting - a popular strategy that increases the loss contributed by rare classes. We investigate the impact of loss reweighting on different components of a ViT, namely, the patch embedding, self-attention backbone, and linear classifier. Our ongoing investigations reveal that loss reweighting impacts mostly the linear classifier and self-attention backbone while having a small and negligible effect on the embedding layer.

----

## [1917] On Analyzing the Role of Image for Visual-Enhanced Relation Extraction (Student Abstract)

**Authors**: *Lei Li, Xiang Chen, Shuofei Qiao, Feiyu Xiong, Huajun Chen, Ningyu Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26987](https://doi.org/10.1609/aaai.v37i13.26987)

**Abstract**:

Multimodal relation extraction is an essential task for knowledge graph construction. In this paper, we take an in-depth empirical analysis that indicates the inaccurate information in the visual scene graph leads to poor modal alignment weights, further degrading performance. Moreover, the visual shuffle experiments illustrate that the current approaches may not take full advantage of visual information. Based on the above observation, we further propose a strong baseline with an implicit fine-grained multimodal alignment based on Transformer for multimodal relation extraction. Experimental results demonstrate the better performance of our method. Codes are available at https://github.com/zjunlp/DeepKE/tree/main/example/re/multimodal.

----

## [1918] Double Policy Network for Aspect Sentiment Triplet Extraction (Student Abstract)

**Authors**: *Xuting Li, Daifeng Li, Ruo Du, Dingquan Chen, Andrew D. Madden*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26988](https://doi.org/10.1609/aaai.v37i13.26988)

**Abstract**:

Aspect Sentiment Triplet Extraction (ASTE) is the task to extract aspects, opinions and associated sentiments from sentences. Previous studies do not adequately consider the complicated interactions between aspect and opinion terms in both extraction logic and strategy. We present a novel Double Policy Network with Multi-Tag based Reward model (DPN-MTR), which adopts two networks ATE, TSOTE and a Trigger Mechanism to execute ASTE task following a more logical framework. A Multi-Tag based reward is also proposed to solve the limitations of existing studies for identifying aspect/opinion terms with multiple tokens (one term may consist of two or more tokens) to a certain extent. Extensive experiments are conducted on four widely-used benchmark datasets, and demonstrate the effectiveness of our model in generally improving the performance on ASTE significantly.

----

## [1919] Learning Generalizable Batch Active Learning Strategies via Deep Q-networks (Student Abstract)

**Authors**: *Yi-Chen Li, Wen-Jie Shen, Boyu Zhang, Feng Mao, Zongzhang Zhang, Yang Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26989](https://doi.org/10.1609/aaai.v37i13.26989)

**Abstract**:

To handle a large amount of unlabeled data, batch active learning (BAL) queries humans for the labels of a batch of the most valuable data points at every round. Most current BAL strategies are based on human-designed heuristics, such as uncertainty sampling or mutual information maximization. However, there exists a disagreement between these heuristics and the ultimate goal of BAL, i.e., optimizing the model's final performance within the query budgets. This disagreement leads to a limited generality of these heuristics. To this end, we formulate BAL as an MDP and propose a data-driven approach based on deep reinforcement learning. Our method learns the BAL strategy by maximizing the model's final performance. Experiments on the UCI benchmark show that our method can achieve competitive performance compared to existing heuristics-based approaches.

----

## [1920] Cross-Regional Fraud Detection via Continual Learning (Student Abstract)

**Authors**: *Yujie Li, Yuxuan Yang, Qiang Gao, Xin Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26990](https://doi.org/10.1609/aaai.v37i13.26990)

**Abstract**:

Detecting fraud is an urgent task to avoid transaction risks. Especially when expanding a business to new cities or new countries, developing a totally new model will bring the cost issue and result in forgetting previous knowledge. This study proposes a novel solution based on heterogeneous trade graphs, namely HTG-CFD, to prevent knowledge forgetting of cross-regional fraud detection. Specifically, a novel heterogeneous trade graph is meticulously constructed from original transactions to explore the complex semantics among different types of entities and relationships. Motivated by continual learning, we present a practical and task-oriented forgetting prevention method to alleviate knowledge forgetting in the context of cross-regional detection. Extensive experiments demonstrate that HTG-CFD promotes performance in both cross-regional and single-regional scenarios.

----

## [1921] Category-Guided Visual Question Generation (Student Abstract)

**Authors**: *Hongfei Liu, Jiali Chen, Wenhao Fang, Jiayuan Xie, Yi Cai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26991](https://doi.org/10.1609/aaai.v37i13.26991)

**Abstract**:

Visual question generation aims to generate high-quality questions related to images. Generating questions based only on images can better reduce labor costs and thus be easily applied. However, their methods tend to generate similar general questions that fail to ask questions about the specific content of each image scene. In this paper, we propose a category-guided visual question generation model that can generate questions with multiple categories that focus on different objects in an image. Specifically, our model first selects the appropriate question category based on the objects in the image and the relationships among objects. Then, we generate corresponding questions based on the selected question categories. Experiments conducted on the TDIUC dataset show that our proposed model outperforms existing models in terms of diversity and quality.

----

## [1922] Can Graph Neural Networks Learn to Solve the MaxSAT Problem? (Student Abstract)

**Authors**: *Minghao Liu, Pei Huang, Fuqi Jia, Fan Zhang, Yuchen Sun, Shaowei Cai, Feifei Ma, Jian Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26992](https://doi.org/10.1609/aaai.v37i13.26992)

**Abstract**:

The paper presents an attempt to bridge the gap between machine learning and symbolic reasoning. We build graph neural networks (GNNs) to predict the solution of the Maximum Satisfiability (MaxSAT) problem, an optimization variant of SAT. Two closely related graph representations are adopted, and we prove their theoretical equivalence. We also show that GNNs can achieve attractive performance to solve hard MaxSAT problems in certain distributions even compared with state-of-the-art solvers through experimental evaluation.

----

## [1923] Flaky Performances When Pretraining on Relational Databases (Student Abstract)

**Authors**: *Shengchao Liu, David Vázquez, Jian Tang, Pierre-André Noël*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26993](https://doi.org/10.1609/aaai.v37i13.26993)

**Abstract**:

We explore the downstream task performances for graph neural network (GNN) self-supervised learning (SSL) methods trained on subgraphs extracted from relational databases (RDBs). Intuitively, this joint use of SSL and GNNs should allow to leverage more of the available data, which could translate to better results. However, we found that naively porting contrastive SSL techniques can cause ``negative transfer'': linear evaluation on fixed representation from a pretrained model performs worse than on representations from the randomly-initialized model. Based on the conjecture that contrastive SSL conflicts with the message passing layers of the GNN, we propose InfoNode: a contrastive loss aiming to maximize the mutual information between a node's initial- and final-layer representation. The primary empirical results support our conjecture and the effectiveness of InfoNode.

----

## [1924] A Highly Efficient Marine Mammals Classifier Based on a Cross-Covariance Attended Compact Feed-Forward Sequential Memory Network (Student Abstract)

**Authors**: *Xiangrui Liu, Julian Cheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26994](https://doi.org/10.1609/aaai.v37i13.26994)

**Abstract**:

Military active sonar and marine transportation are detrimental to the livelihood of marine mammals and the ecosystem. Early detection and classification of marine mammals using machine learning can help humans to mitigate the harm to marine mammals. This paper proposes a cross-covariance attended compact Feed-Forward Sequential Memory Network (CC-FSMN). The proposed framework shows improved efficiency over multiple convolutional neural network (CNN) backbones. It also maintains a relatively decent performance.

----

## [1925] MGIA: Mutual Gradient Inversion Attack in Multi-Modal Federated Learning (Student Abstract)

**Authors**: *Xuan Liu, Siqi Cai, Lin Li, Rui Zhang, Song Guo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26995](https://doi.org/10.1609/aaai.v37i13.26995)

**Abstract**:

Recent studies have demonstrated that local training data in Federated Learning can be recovered from gradients, which are called gradient inversion attacks. These attacks display powerful effects on either computer vision or natural language processing tasks. As it is known that there are certain correlations between multi-modality data, we argue that the threat of such attacks combined with Multi-modal Learning may cause more severe effects. Different modalities may communicate through gradients to provide richer information for the attackers, thus improving the strength and efficiency of the gradient inversion attacks. In this paper, we propose the Mutual Gradient Inversion Attack (MGIA), by utilizing the shared labels between image and text modalities combined with the idea of knowledge distillation. Our experimental results show that MGIA achieves the best quality of both modality data and label recoveries in comparison with other methods. In the meanwhile, MGIA verifies that multi-modality gradient inversion attacks are more likely to disclose private information than the existing single-modality attacks.

----

## [1926] Semi-supervised Review-Aware Rating Regression (Student Abstract)

**Authors**: *Xiangkui Lu, Jun Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26996](https://doi.org/10.1609/aaai.v37i13.26996)

**Abstract**:

Semi-supervised learning is a promising solution to mitigate data sparsity in review-aware rating regression (RaRR), but it bears the risk of learning with noisy pseudo-labelled data. In this paper, we propose a paradigm called co-training-teaching (CoT2), which integrates the merits of both co-training and co-teaching towards the robust semi-supervised RaRR. Concretely, CoT2 employs two predictors and each of them alternately plays the roles of  "labeler" and "validator" to generate and validate pseudo-labelled instances. Extensive experiments show that CoT2 considerably outperforms state-of-the-art RaRR techniques, especially when training data is severely insufficient.

----

## [1927] Toplogical Data Analysis Detects and Classifies Sunspots (Student Abstract)

**Authors**: *Aidan Lytle, Neil Pritchard, Alicia Aarnio, Thomas Weighill*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26997](https://doi.org/10.1609/aaai.v37i13.26997)

**Abstract**:

In our technology-dependent modern world, it is imperative to monitor the Sun for space weather threats to critical infrastructure. Topological data analysis (TDA) is a new set of mathematical techniques used in data analysis and machine learning. We demonstrate that TDA can robustly detect and classify solar surface and coronal activity. This technique is a promising step toward future application in predictive space weather modeling.

----

## [1928] Risk-Aware Decentralized Safe Control via Dynamic Responsibility Allocation (Student Abstract)

**Authors**: *Yiwei Lyu, Wenhao Luo, John M. Dolan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26998](https://doi.org/10.1609/aaai.v37i13.26998)

**Abstract**:

In this work, we present a novel risk-aware decentralized Control Barrier Function (CBF)-based controller for multi-agent systems. The proposed decentralized controller is composed based on pairwise agent responsibility shares (a percentage), calculated from the risk evaluation of each individual agent faces in a multi-agent interaction environment. With our proposed CBF-inspired risk evaluation framework, the responsibility portions between pairwise agents are dynamically updated based on the relative risk they face. Our method allows agents with lower risk to enjoy a higher level of freedom in terms of a wider action space, and the agents exposed to higher risk are constrained more tightly on action spaces, and are therefore forced to proceed with caution.

----

## [1929] A Mutually Enhanced Bidirectional Approach for Jointly Mining User Demand and Sentiment (Student Abstract)

**Authors**: *Xue Mao, Haoda Qian, Minjie Yuan, Qiudan Li*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.26999](https://doi.org/10.1609/aaai.v37i13.26999)

**Abstract**:

User demand mining aims to identify the implicit demand from the e-commerce reviews, which are always irregular, vague and diverse. Existing sentiment analysis research mainly focuses on aspect-opinion-sentiment triplet extraction, while the deeper user demands remain unexplored. In this paper, we formulate a novel research question of jointly mining aspect-opinion-sentiment-demand, and propose a Mutually Enhanced Bidirectional Extraction (MEMB) framework for capturing the dynamic interaction among different types of information. Finally, experiments on Chinese e-commerce data demonstrate the efficacy of the proposed model.

----

## [1930] Debiasing Intrinsic Bias and Application Bias Jointly via Invariant Risk Minimization (Student Abstract)

**Authors**: *Yuzhou Mao, Liu Yu, Yi Yang, Fan Zhou, Ting Zhong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27000](https://doi.org/10.1609/aaai.v37i13.27000)

**Abstract**:

Demographic biases and social stereotypes are common in pretrained language models (PLMs), while the fine-tuning in downstream applications can also produce new biases or amplify the impact of the original biases. Existing works separate the debiasing from the fine-tuning procedure, which results in a gap between intrinsic bias and application bias. In this work, we propose a debiasing framework CauDebias to eliminate both biases, which directly combines debiasing with fine-tuning and can be applied for any PLMs in downstream tasks. We distinguish the bias-relevant (non-causal factors) and label-relevant (causal factors) parts in sentences from a causal invariant perspective. Specifically, we perform intervention on non-causal factors in different demographic groups, and then devise an invariant risk minimization loss to trade-off performance between bias mitigation and task accuracy. Experimental results on three downstream tasks show that our CauDebias can remarkably reduce biases in PLMs while minimizing the impact on downstream tasks.

----

## [1931] Label Smoothing for Emotion Detection (Student Abstract)

**Authors**: *George Maratos, Tiberiu Sosea, Cornelia Caragea*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27001](https://doi.org/10.1609/aaai.v37i13.27001)

**Abstract**:

Automatically detecting emotions from text has countless
applications, ranging from large scale opinion mining to
social robots in healthcare and education. However, emotions
are subjective in nature and are often expressed in ambiguous
ways. At the same time, detecting emotions can also require
implicit reasoning, which may not be available as surface-
level, lexical information. In this work, we conjecture that
the overconfidence of pre-trained language models such as
BERT is a critical problem in emotion detection and show
that alleviating this problem can considerably improve the
generalization performance. We carry out comprehensive
experiments on four emotion detection benchmark datasets
and show that calibrating our model predictions leads to an
average improvement of 1.35% in weighted F1 score.

----

## [1932] Counting Knot Mosaics with ALLSAT (Student Abstract)

**Authors**: *Hannah Miller*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27002](https://doi.org/10.1609/aaai.v37i13.27002)

**Abstract**:

Knot mosaics are a model of a quantum knot system.  A knot mosaic is a m-by-n grid where each location on the grid may contain any of 11 possible tiles such that the final layout has closed loops.  Oh et al. proved a recurrence relation of state matrices to count the number of m-by-n knot mosaics.  Our contribution is to use ALLSAT solvers to count knot mosaics and to experimentally try different ways to encode the AT MOST ONE constraint in SAT.  We plan to use our SAT method as a tool to list knot mosaics of interest for specific classes of knots.

----

## [1933] Novel Intent Detection and Active Learning Based Classification (Student Abstract)

**Authors**: *Ankan Mullick*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27003](https://doi.org/10.1609/aaai.v37i13.27003)

**Abstract**:

Novel intent class detection is an important problem in real world scenario for conversational agents for continuous interaction. Several research works have been done to detect novel intents in a mono-lingual (primarily English) texts and
images. But, current systems lack an end-to-end universal framework to detect novel intents across various different languages with less human annotation effort for mis-classified and system rejected samples. This paper proposes
NIDAL (Novel Intent Detection and Active Learning based
classification), a semi-supervised framework to detect novel
intents while reducing human annotation cost. Empirical results on various benchmark datasets demonstrate that this system outperforms the baseline methods by more than 10%
margin for accuracy and macro-F1. The system achieves this while maintaining overall annotation cost to be just ~6-10% of the unlabeled data available to the system.

----

## [1934] Pre-training with Scientific Text Improves Educational Question Generation (Student Abstract)

**Authors**: *Hamze Muse, Sahan Bulathwela, Emine Yilmaz*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27004](https://doi.org/10.1609/aaai.v37i13.27004)

**Abstract**:

With the boom of digital educational materials and scalable e-learning systems, the potential for realising AI-assisted personalised learning has skyrocketed. In this landscape, the automatic generation of educational questions will play a key role, enabling scalable self-assessment when a global population is manoeuvring their personalised learning journeys. We develop EduQG, a novel educational question generation model built by adapting a large language model. Our initial experiments demonstrate that EduQG can produce superior educational questions by pre-training on scientific text.

----

## [1935] Fraud's Bargain Attacks to Textual Classifiers via Metropolis-Hasting Sampling (Student Abstract)

**Authors**: *Mingze Ni, Zhensu Sun, Wei Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27005](https://doi.org/10.1609/aaai.v37i13.27005)

**Abstract**:

Recent studies on adversarial examples expose vulnerabilities of natural language processing (NLP) models. Existing techniques for generating adversarial examples are typically driven by deterministic heuristic rules that are agnostic to the optimal adversarial examples, a strategy that often results in attack failures. To this end, this research proposes Fraud's Bargain Attack (FBA), which utilizes a novel randomization mechanism to enlarge the searching space and enables high-quality adversarial examples to be generated with high probabilities. FBA applies the Metropolis-Hasting algorithm to enhance the selection of adversarial examples from all candidates proposed by a customized Word Manipulation Process (WMP). WMP perturbs one word at a time via insertion, removal, or substitution in a contextual-aware manner. Extensive experiments demonstrate that FBA outperforms the baselines in terms of attack success rate and imperceptibility.

----

## [1936] Improving Adversarial Robustness to Sensitivity and Invariance Attacks with Deep Metric Learning (Student Abstract)

**Authors**: *Anaelia Ovalle, Evan Czyzycki, Cho-Jui Hsieh*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27006](https://doi.org/10.1609/aaai.v37i13.27006)

**Abstract**:

Intentionally crafted adversarial samples have effectively exploited weaknesses in deep neural networks. A standard method in adversarial robustness assumes a framework to defend against samples crafted by minimally perturbing a sample such that its corresponding model output changes. These sensitivity attacks exploit the model's sensitivity toward task-irrelevant features. Another form of adversarial sample can be crafted via invariance attacks, which exploit the model underestimating the importance of relevant features. Previous literature has indicated a tradeoff in defending against both attack types within a strictly L-p bounded defense. To promote robustness toward both types of attacks beyond Euclidean distance metrics, we use metric learning to frame adversarial regularization as an optimal transport problem. Our preliminary results indicate that regularizing over invariant perturbations in our framework improves both invariant and sensitivity defense.

----

## [1937] LVRNet: Lightweight Image Restoration for Aerial Images under Low Visibility (Student Abstract)

**Authors**: *Esha Pahwa, Achleshwar Luthra, Pratik Narang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27007](https://doi.org/10.1609/aaai.v37i13.27007)

**Abstract**:

Learning to recover clear images from images having a combination of degrading factors is a challenging task. That being said, autonomous surveillance in low visibility conditions caused by high pollution/smoke, poor air quality index, low light, atmospheric scattering, and haze during a blizzard, etc, becomes even more important to prevent accidents. It is thus crucial to form a solution that can not only result in a high-quality image but also which is efficient enough to be deployed for everyday use. However, the lack of proper datasets available to tackle this task limits the performance of the previous methods proposed. To this end, we generate the LowVis-AFO dataset, containing 3647 paired dark-hazy and clear images. We also introduce a new lightweight deep learning model called Low-Visibility Restoration Network (LVRNet). It outperforms previous image restoration methods with low latency, achieving a PSNR value of 25.744 and an SSIM of 0.905, hence making our approach scalable and ready for practical use.

----

## [1938] Hardness of Learning AES Key (Student Abstract)

**Authors**: *Artur Pak, Sultan Nurmukhamedov, Rustem Takhanov, Zhenisbek Assylbekov*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27008](https://doi.org/10.1609/aaai.v37i13.27008)

**Abstract**:

We show hardness of learning AES key from pairs of ciphertexts under the assumption of computational closeness of AES to pairwise independence. The latter is motivated by a recent result on statistical closeness of AES to pairwise independence.

----

## [1939] Generative Pipeline for Data Augmentation of Unconstrained Document Images with Structural and Textural Degradation (Student Abstract)

**Authors**: *Arnab Poddar, Abhishek Kumar Sah, Soumyadeep Dey, Pratik Jawanpuria, Jayanta Mukhopadhyay, Prabir Kumar Biswas*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27009](https://doi.org/10.1609/aaai.v37i13.27009)

**Abstract**:

Computer vision applications for document image understanding (DIU) such as optical character recognition, word spotting, enhancement etc. suffer  from structural deformations like strike-outs and unconstrained strokes, to name a few. They also suffer from texture degradation due to blurring, aging, or  blotting-spots etc. 
The DIU applications with deep networks are limited to constrained environment and lack diverse data with text-level and pixel-level annotation simultaneously. In this work, we  propose a generative framework to produce realistic synthetic handwritten document images with simultaneous annotation of text and corresponding pixel-level spatial foreground information. The proposed approach generates realistic backgrounds with artificial handwritten texts which supplements data-augmentation in multiple unconstrained DIU systems.  The proposed framework is an early work to facilitate DIU system-evaluation in both image quality and recognition performance at a go.

----

## [1940] Neural Language Model Based Attentive Term Dependence Model for Verbose Query (Student Abstract)

**Authors**: *Dipannita Podder, Jiaul H. Paik, Pabitra Mitra*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27010](https://doi.org/10.1609/aaai.v37i13.27010)

**Abstract**:

The query-document term matching plays an important role in information retrieval. However, the retrieval performance degrades when the documents get matched with the extraneous terms of the query which frequently arises in verbose queries. To address this problem, we generate the dense vector of
the entire query and individual query terms using the pre-trained BERT (Bidirectional Encoder Representations from Transformers) model and subsequently analyze their relation to focus on the central terms. We then propose a context-aware attentive extension of unsupervised Markov Random Field-based sequential term dependence model that explicitly pays more attention to those contextually central terms. The proposed model utilizes the strengths of the pre-trained large language model for estimating the attention weight of terms and rank the documents in a single pass without any supervision.

----

## [1941] Evaluating Factors Influencing COVID-19 Outcomes across Countries Using Decision Trees (Student Abstract)

**Authors**: *Aniruddha Pokhrel, Nikesh Subedi, Saurav Keshari Aryal*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27011](https://doi.org/10.1609/aaai.v37i13.27011)

**Abstract**:

While humanity prepares for a post-pandemic world and a return to normality through worldwide vaccination campaigns, each country experienced different levels of impact based on natural, political, regulatory, and socio-economic factors. To prepare for a possible future with COVID-19 and similar outbreaks, it is imperative to understand how each of these factors impacted spread and mortality. We train and tune two decision tree regression models to predict COVID-related cases and deaths using a multitude of features. Our findings suggest that, at the country-level, GDP per capita and comorbidity mortality rate are best predictors for both outcomes. Furthermore, latitude and smoking prevalence are also significantly related to COVID-related spread and mortality.

----

## [1942] Ordinal Programmatic Weak Supervision and Crowdsourcing for Estimating Cognitive States (Student Abstract)

**Authors**: *Prakruthi Pradeep, Benedikt Boecking, Nicholas Gisolfi, Jacob R. Kintz, Torin K. Clark, Artur Dubrawski*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27012](https://doi.org/10.1609/aaai.v37i13.27012)

**Abstract**:

Crowdsourcing and weak supervision offer methods to efficiently label large datasets. Our work builds on existing weak supervision models to accommodate ordinal target classes, in an effort to recover ground truth from weak, external labels.
We define a parameterized factor function and show that our approach improves over other baselines.

----

## [1943] A Probabilistic Graph Diffusion Model for Source Localization (Student Abstract)

**Authors**: *Tangjiang Qian, Xovee Xu, Zhe Xiao, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27013](https://doi.org/10.1609/aaai.v37i13.27013)

**Abstract**:

Source localization, as a reverse problem of graph diffusion, is important for many applications such as rumor tracking, detecting computer viruses, and finding epidemic spreaders. However, it is still under-explored due to the inherent uncertainty of the diffusion process: after a long period of propagation, the same diffusion process may start with diverse sources. Most existing solutions utilize deterministic models and therefore cannot describe the diffusion uncertainty of sources. Moreover, current probabilistic approaches are hard to conduct smooth transformations with variational inference. To overcome the limitations, we propose a probabilistic framework using continuous normalizing flows with invertible transformations and graph neural networks to explicitly model the uncertainty of the diffusion source. Experimental results on two real-world datasets demonstrate the effectiveness of our model over strong baselines.

----

## [1944] Explaining Large Language Model-Based Neural Semantic Parsers (Student Abstract)

**Authors**: *Daking Rai, Yilun Zhou, Bailin Wang, Ziyu Yao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27014](https://doi.org/10.1609/aaai.v37i13.27014)

**Abstract**:

While large language models (LLMs) have demonstrated strong capability in structured prediction tasks such as semantic parsing, few amounts of research have explored the underlying mechanisms of their success. Our work studies different methods for explaining an LLM-based semantic parser and qualitatively discusses the explained model behaviors, hoping to inspire future research toward better understanding them.

----

## [1945] Fuzzy C-means: Differences on Clustering Behavior between High Dimensional and Functional Data (Student Abstract)

**Authors**: *Carlos Ramos-Carreño*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27015](https://doi.org/10.1609/aaai.v37i13.27015)

**Abstract**:

Fuzzy c-means (FCM) is a generalization of the classical k-means clustering algorithm to the case where an observation can belong to several clusters at the same time.
The algorithm was previously observed to have initialization problems when the number of desired clusters or the number of dimensions of the data are high.
We have tested FCM against clustering problems with functional data, generated from stationary Gaussian processes, and thus in principle infinite-dimensional.
We observed that when the data is more functional in nature, which can be obtained by tuning the length-scale parameter of the Gaussian process, the aforementioned problems do not appear.
This not only indicates that FCM is suitable as a clustering method for functional data, but also illustrates how functional data differs from traditional multivariate data.
In addition this seems to suggest a qualitative way to measure the latent dimensionality of the functional distribution itself.

----

## [1946] Photogrammetry and VR for Comparing 2D and Immersive Linguistic Data Collection (Student Abstract)

**Authors**: *Jacob Rubinstein, Cynthia Matuszek, Don Engel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27016](https://doi.org/10.1609/aaai.v37i13.27016)

**Abstract**:

The overarching goal of this work is to enable the collection of language describing a wide variety of objects viewed in virtual reality. We aim to create full 3D models from a small number of ‘keyframe’ images of objects found in the publicly available Grounded Language Dataset (GoLD) using photogrammetry. We will then collect linguistic descriptions by placing our models in virtual reality and having volunteers describe them. To evaluate the impact of virtual reality immersion on linguistic descriptions of the objects, we intend to apply contrastive learning to perform grounded language learning, then compare the descriptions collected from images (in GoLD) versus our models.

----

## [1947] RFC-Net: Learning High Resolution Global Features for Medical Image Segmentation on a Computational Budget (Student Abstract)

**Authors**: *Sourajit Saha, Shaswati Saha, Md. Osman Gani, Tim Oates, David Chapman*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27017](https://doi.org/10.1609/aaai.v37i13.27017)

**Abstract**:

Learning High-Resolution representations is essential for semantic segmentation. Convolutional neural network (CNN) architectures with downstream and upstream propagation flow are popular for segmentation in medical diagnosis. However, due to performing spatial downsampling and upsampling in multiple stages, information loss is inexorable. On the contrary, connecting layers densely on high spatial resolution is computationally expensive. In this work, we devise a Loose Dense Connection Strategy to connect neurons in subsequent layers with reduced parameters. On top of that, using a m-way Tree structure for feature propagation we propose Receptive Field Chain Network (RFC-Net) that learns high-resolution global features on a compressed computational space. Our experiments demonstrates that RFC Net achieves state-of-the-art performance on Kvasir and CVC-ClinicDB benchmarks for Polyp segmentation. Our code is publicly available at github.com/sourajitcs/RFC-NetAAAI23.

----

## [1948] Maximizing Influence Spread through a Dynamic Social Network (Student Abstract)

**Authors**: *Simon Schierreich*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27018](https://doi.org/10.1609/aaai.v37i13.27018)

**Abstract**:

Modern social networks are dynamic in their nature; a new connections are appearing and old connections are disappearing all the time. However, in our algorithmic and complexity studies, we usually model social networks as static graphs.

In this paper, we propose a new paradigm for the study of the well-known Target Set Selection problem, which is a fundamental problem in viral marketing and the spread of opinion through social networks. In particular, we use temporal graphs to capture the dynamic nature of social networks.

We show that the temporal interpretation is, unsurprisingly, NP-complete in general. Then, we study computational complexity of this problem for multiple restrictions of both the threshold function and the underlying graph structure and provide multiple hardness lower-bounds.

----

## [1949] Can You Answer This? - Exploring Zero-Shot QA Generalization Capabilities in Large Language Models (Student Abstract)

**Authors**: *Saptarshi Sengupta, Shreya Ghosh, Preslav Nakov, Prasenjit Mitra*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27019](https://doi.org/10.1609/aaai.v37i13.27019)

**Abstract**:

The buzz around Transformer-based language models (TLM) such as BERT, RoBERTa, etc. is well-founded owing to their impressive results on an array of tasks. However, when applied to areas needing specialized knowledge (closed-domain), such as medical, finance, etc. their performance takes drastic hits, sometimes more than their older recurrent/convolutional counterparts. In this paper, we explore zero-shot capabilities of large LMs for extractive QA. Our objective is to examine performance change in the face of domain drift i.e. when the target domain data is vastly different in semantic and statistical properties from the source domain and attempt to explain the subsequent behavior. To this end, we present two studies in this paper while planning further experiments later down the road. Our findings indicate flaws in the current generation of TLM limiting their performance on closed-domain tasks.

----

## [1950] FakeKG: A Knowledge Graph of Fake Claims for Improving Automated Fact-Checking (Student Abstract)

**Authors**: *Gautam Kishore Shahi*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27020](https://doi.org/10.1609/aaai.v37i13.27020)

**Abstract**:

False information could be dangerous if the claim is not debunked timely. Fact-checking organisations get a high volume of claims on different topics with immense velocity. The efficiency of the fact-checkers decreases due to 3V problems volume, velocity and variety. Especially during crises or elections, fact-checkers cannot handle user requests to verify the claim. Until now, no real-time curable centralised corpus of fact-checked articles is available. Also, the same claim is fact-checked by multiple fact-checking organisations with or without judgement. To fill this gap, we introduce FakeKG: A Knowledge Graph-Based approach for improving Automated Fact-checking. FakeKG is a centralised knowledge graph containing fact-checked articles from different sources that can be queried using the SPARQL endpoint. The proposed FakeKG can prescreen claim requests and filter them if the claim is already fact-checked and provide a judgement to the claim. It will also categorise the claim's domain so that the fact-checker can prioritise checking the incoming claims into different groups like health and election. This study proposes an approach for creating FakeKG and its future application for mitigating misinformation.

----

## [1951] Can Adversarial Networks Make Uninformative Colonoscopy Video Frames Clinically Informative? (Student Abstract)

**Authors**: *Vanshali Sharma, Manas Kamal Bhuyan, Pradip K. Das*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27021](https://doi.org/10.1609/aaai.v37i13.27021)

**Abstract**:

Various artifacts, such as ghost colors, interlacing, and motion blur, hinder diagnosing colorectal cancer (CRC) from videos acquired during colonoscopy. The frames containing these artifacts are called uninformative frames and are present in large proportions in colonoscopy videos. To alleviate the impact of artifacts,  we propose an adversarial network based framework to convert uninformative frames to clinically relevant frames. We examine the effectiveness of the proposed approach by evaluating the translated frames for polyp detection using YOLOv5. Preliminary results present improved detection performance along with elegant qualitative outcomes. We also examine the failure cases to determine the directions for future work.

----

## [1952] Bayesian Models for Targeted Cyber Deception Strategies (Student Abstract)

**Authors**: *Nazia Sharmin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27022](https://doi.org/10.1609/aaai.v37i13.27022)

**Abstract**:

We propose a model-driven decision support system (DSS) based on a Bayesian belief network (BBN) to support cyber deception based on a detailed model of attacker beliefs. We discuss this approach using a case study based on passively observed operating system (OS) fingerprinting data. In passive reconnaissance attackers can remain undetected while collecting information to identify systems and plan attacks. Our DSS is intended to support preventative measures to protect the network from successful reconnaissance, such as by modifying features using deception. We validate the prediction accuracy of the model in comparison with a sequential artificial neural network (ANN). We then introduce a deceptive algorithm to select a minimal set of features for OS obfuscation. We show the effectiveness of feature-modification strategies based on our methods using passively collected data to decide what features from a real operating system (OS) to modify to appear as a fake [different] OS.

----

## [1953] Scalable Negotiating Agent Strategy via Multi-Issue Policy Network (Student Abstract)

**Authors**: *Takumu Shimizu, Ryota Higa, Toki Takahashi, Katsuhide Fujita, Shinji Nakadai*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27023](https://doi.org/10.1609/aaai.v37i13.27023)

**Abstract**:

Previous research on the comprehensive negotiation strategy using deep reinforcement learning (RL) has scalability issues of not performing effectively in the large-sized domains.
We improve negotiation strategy via deep RL by considering an issue-based represented deep policy network to deal with multi-issue negotiation.
The architecture of the proposed learning agent considers the characteristics of multi-issue negotiation domains and policy-based learning.
We demonstrate that proposed method achieve equivalent or higher utility than existing negotiation agents in the large-sized domains.

----

## [1954] Efficient Dynamic Batch Adaptation (Student Abstract)

**Authors**: *Cristian Simionescu, George Stoica*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27024](https://doi.org/10.1609/aaai.v37i13.27024)

**Abstract**:

In this paper we introduce Efficient Dynamic Batch Adaptation (EDBA), which improves on a previous method that works by adjusting the composition and the size of the current batch. Our improvements allow for Dynamic Batch Adaptation to feasibly scale up for bigger models and datasets, drastically improving model convergence and generalization. We show how the method is still able to perform especially well in data-scarce scenarios, managing to obtain a test accuracy on 100 samples of CIFAR-10 of 90.68%, while the baseline only reaches 23.79%. On the full CIFAR-10 dataset, EDBA reaches convergence in ∼120 epochs while the baseline requires ∼300 epochs.

----

## [1955] Development of a Human-Agent Interaction System including Norm and Emotion in an Evacuation Situation (Student Abstract)

**Authors**: *Ephraim Sinyabe Pagou, Vivient Corneille Kamla, Igor Tchappi, Amro Najjar*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27025](https://doi.org/10.1609/aaai.v37i13.27025)

**Abstract**:

Agent-based modeling and simulation can provide a powerful test environment for crisis management scenarios. Human agent interaction has limitations in representing norms issued by an agent to a human agent that has emotions. In this study, we present an approach to the interaction between a virtual normative agent and a human agent in an evacuation scenario. Through simulation comparisons, it is shown that the method used in this study can more fully simulate the real-life out come of an emergency situation and also improves the au thenticity of the agent interaction.

----

## [1956] Persistent Homology through Image Segmentation (Student Abstract)

**Authors**: *Joshua Slater, Thomas Weighill*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27026](https://doi.org/10.1609/aaai.v37i13.27026)

**Abstract**:

The efficacy of topological data analysis (TDA) has been demonstrated in many different machine learning pipelines, particularly those in which structural characteristics of data are highly relevant. However, TDA's usability in large scale machine learning applications is hindered by the significant computational cost of generating persistence diagrams. In this work, a method that allows this computationally expensive process to be approximated by deep neural networks is proposed. Moreover, the method's practicality in estimating 0-dimensional persistence diagrams across a diverse range of images is shown.

----

## [1957] TA-DA: Topic-Aware Domain Adaptation for Scientific Keyphrase Identification and Classification (Student Abstract)

**Authors**: *Razvan-Alexandru Smadu, George-Eduard Zaharia, Andrei-Marius Avram, Dumitru-Clementin Cercel, Mihai Dascalu, Florin Pop*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27027](https://doi.org/10.1609/aaai.v37i13.27027)

**Abstract**:

Keyphrase identification and classification is a Natural Language Processing and Information Retrieval task that involves extracting relevant groups of words from a given text related to the main topic. In this work, we focus on extracting keyphrases from scientific documents. We introduce TA-DA, a Topic-Aware Domain Adaptation framework for keyphrase extraction that integrates Multi-Task Learning with Adversarial Training and Domain Adaptation. Our approach improves performance over baseline models by up to 5% in the exact match of the F1-score.

----

## [1958] Exploring the Relative Value of Collaborative Optimisation Pathways (Student Abstract)

**Authors**: *Sudarshan Sreeram*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27028](https://doi.org/10.1609/aaai.v37i13.27028)

**Abstract**:

Compression techniques in machine learning (ML) independently improve a model’s inference efficiency by reducing its memory footprint while aiming to maintain its quality. This paper lays groundwork in questioning the merit of a compression pipeline involving all techniques as opposed to skipping a few by considering a case study on a keyword spotting model: DS-CNN-S. In addition, it documents improvements to the model’s training and dataset infrastructure. For this model, preliminary findings suggest that a full-scale pipeline isn’t required to achieve a competent memory footprint and accuracy, but a more comprehensive study is required.

----

## [1959] Backforward Propagation (Student Abstract)

**Authors**: *George Stoica, Cristian Simionescu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27029](https://doi.org/10.1609/aaai.v37i13.27029)

**Abstract**:

In this paper we introduce Backforward Propagation, a method of completely eliminating Internal Covariate Shift (ICS). Unlike previous methods, which only indirectly reduce the impact of ICS while introducing other biases, we are able to have a surgical view at the effects ICS has on training neural networks. Our experiments show that ICS has a weight regularizing effect on models, and completely removing it enables for faster convergence of the neural network.

----

## [1960] Two-Streams: Dark and Light Networks with Graph Convolution for Action Recognition from Dark Videos (Student Abstract)

**Authors**: *Saurabh Suman, Nilay Naharas, Badri Narayan Subudhi, Vinit Jakhetiya*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27030](https://doi.org/10.1609/aaai.v37i13.27030)

**Abstract**:

In this article, we propose a two-stream action recognition technique for recognizing human actions from dark videos. The proposed action recognition network consists of an image enhancement network with Self-Calibrated Illumination (SCI) module, followed by a two-stream action recognition network. We have used R(2+1)D as a feature extractor for both streams with shared weights. Graph Convolutional Network (GCN), a temporal graph encoder is utilized to enhance the obtained features which are then further fed to a classification head to recognize the actions in a video. The experimental results are presented on the recent benchmark ``ARID" dark-video database.

----

## [1961] ES-Mask: Evolutionary Strip Mask for Explaining Time Series Prediction (Student Abstract)

**Authors**: *Yifei Sun, Cheng Song, Feng Lu, Wei Li, Hai Jin, Albert Y. Zomaya*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27031](https://doi.org/10.1609/aaai.v37i13.27031)

**Abstract**:

Machine learning models are increasingly used in time series prediction with promising results. The model explanation of time series prediction falls behind the model development and makes less sense to users in understanding model decisions. This paper proposes ES-Mask, a post-hoc and model-agnostic evolutionary strip mask-based saliency approach for time series applications. ES-Mask designs the mask consisting of strips with the same salient value in consecutive time steps to produce binary and sustained feature importance scores over time for easy understanding and interpretation of time series. ES-Mask uses an evolutionary algorithm to search for the optimal mask by manipulating strips in rounds, thus is agnostic to models by involving no internal model states in the search. The initial experiments on MIMIC-III data set show that ES-Mask outperforms state-of-the-art methods.

----

## [1962] Exploration on Physics-Informed Neural Networks on Partial Differential Equations (Student Abstract)

**Authors**: *Hoa Ta, Shi Wen Wong, Nathan McClanahan, Jung-Han Kimn, Kaiqun Fu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27032](https://doi.org/10.1609/aaai.v37i13.27032)

**Abstract**:

Data-driven related solutions are dominating various scientific fields with the assistance of machine learning and data analytics. Finding effective solutions has long been discussed in the area of machine learning. The recent decade has witnessed the promising performance of the Physics-Informed Neural Networks (PINN) in bridging the gap between real-world scientific problems and machine learning models. In this paper, we explore the behavior of PINN in a particular range of different diffusion coefficients under specific boundary conditions. In addition, different initial conditions of partial differential equations are solved by applying the proposed PINN. Our paper illustrates how the effectiveness of the PINN can change under various scenarios. As a result, we demonstrate a better insight into the behaviors of the PINN and how to make the proposed method more robust while encountering different scientific and engineering problems.

----

## [1963] Parallel Index-Based Search Algorithm for Coalition Structure Generation (Student Abstract)

**Authors**: *Redha Taguelmimt, Samir Aknine, Djamila Boukredera, Narayan Changder*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27033](https://doi.org/10.1609/aaai.v37i13.27033)

**Abstract**:

In this paper, we propose a novel algorithm to address the Coalition Structure Generation (CSG) problem. Specifically, we use a novel representation of the search space that enables it to be explored in a new way. We introduce an index-based exact algorithm. Our algorithm is anytime, produces optimal solutions, and can be run on large-scale problems with hundreds of agents. Our experimental evaluation on a benchmark with several value distributions shows that our representation of the search space that we combined with the proposed algorithm provides high-quality results for the CSG problem and outperforms existing state-of-the-art algorithms.

----

## [1964] The Naughtyformer: A Transformer Understands and Moderates Adult Humor (Student Abstract)

**Authors**: *Leonard Tang, Alexander Cai, Jason Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27034](https://doi.org/10.1609/aaai.v37i13.27034)

**Abstract**:

Jokes are intentionally written to be funny, but not all jokes are created the same. While recent work has shown impressive results on humor detection in text, we instead investigate the more nuanced task of detecting humor subtypes, especially of the more adult variety. To that end, we introduce a novel jokes dataset filtered from Reddit and solve the subtype
classification task using a finetuned Transformer dubbed the Naughtyformer. Moreover, we show that our model is significantly better at detecting offensiveness in jokes compared to state-of-the-art methods.

----

## [1965] Exploring the Effectiveness of Mask-Guided Feature Modulation as a Mechanism for Localized Style Editing of Real Images (Student Abstract)

**Authors**: *Snehal Singh Tomar, Maitreya Suin, A. N. Rajagopalan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27035](https://doi.org/10.1609/aaai.v37i13.27035)

**Abstract**:

The success of Deep Generative Models at high-resolution image generation has led to their extensive utilization for style editing of real images. Most existing methods work on the principle of inverting real images onto their latent space, followed by determining controllable directions. Both inversion of real images and determination of controllable latent directions are computationally expensive operations. Moreover, the determination of controllable latent directions requires additional human supervision. This work aims to explore the efficacy of mask-guided feature modulation in the latent space of a Deep Generative Model as a solution to these bottlenecks. To this end, we present the SemanticStyle Autoencoder (SSAE), a deep Generative Autoencoder model that leverages semantic mask-guided latent space manipulation for highly localized photorealistic style editing of real images. We present qualitative and quantitative results for the same and their analysis. This work shall serve as a guiding primer for future work.

----

## [1966] Global Explanations for Image Classifiers (Student Abstract)

**Authors**: *Bhavan K. Vasu, Prasad Tadepalli*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27036](https://doi.org/10.1609/aaai.v37i13.27036)

**Abstract**:

We hypothesize that deep network classifications of complex scenes can be explained using sets of relevant objects.
We employ beam search and singular value decomposition to generate local and global explanations that summarize the deep model's interpretation of a class.

----

## [1967] Quantify the Political Bias in News Edits: Experiments with Few-Shot Learners (Student Abstract)

**Authors**: *Preetika Verma, Hansin Ahuja, Kokil Jaidka*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27037](https://doi.org/10.1609/aaai.v37i13.27037)

**Abstract**:

The rapid growth of information and communication technologies in recent years, and the different forms of digital connectivity, have profoundly affected how news is generated and consumed. Digital traces and computational methods offer new opportunities to model and track the provenance of news. This project is the first study to characterize and predict how prominent news outlets make edits to news frames and their implications for geopolitical relationships and attitudes. We evaluate the feasibility of training few-shot learners on the editing patterns of articles discussing different countries, for understanding their wider implications in preserving or damaging geopolitical relationships.

----

## [1968] Anti-drifting Feature Selection via Deep Reinforcement Learning (Student Abstract)

**Authors**: *Aoran Wang, Hongyang Yang, Feng Mao, Zongzhang Zhang, Yang Yu, Xiaoyang Liu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27038](https://doi.org/10.1609/aaai.v37i13.27038)

**Abstract**:

Feature selection (FS) is a crucial procedure in machine learning pipelines for its significant benefits in removing data redundancy and mitigating model overfitting. Since concept drift is a widespread phenomenon in streaming data and could severely affect model performance, effective FS on concept drifting data streams is imminent. However, existing state-of-the-art FS algorithms fail to adjust their selection strategy adaptively when the effective feature subset changes, making them unsuitable for drifting streams. In this paper, we propose a dynamic FS method that selects effective features on concept drifting data streams via deep reinforcement learning. Specifically, we present two novel designs: (i) a skip-mode reinforcement learning environment that shrinks action space size for high-dimensional FS tasks; (ii) a curiosity mechanism that generates intrinsic rewards to address the long-horizon exploration problem. The experiment results show that our proposed method outperforms other FS methods and can dynamically adapt to concept drifts.

----

## [1969] Learning Dynamic Temporal Relations with Continuous Graph for Multivariate Time Series Forecasting (Student Abstract)

**Authors**: *Zhiyuan Wang, Fan Zhou, Goce Trajcevski, Kunpeng Zhang, Ting Zhong*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27039](https://doi.org/10.1609/aaai.v37i13.27039)

**Abstract**:

The recent advance in graph neural networks (GNNs) has inspired a few studies to leverage the dependencies of variables for time series prediction. Despite the promising results, existing GNN-based models cannot capture the global dynamic relations between variables owing to the inherent limitation of their graph learning module. Besides, multi-scale temporal information is usually ignored or simply concatenated in prior methods, resulting in inaccurate predictions. To overcome these limitations, we present CGMF, a Continuous Graph learning method for Multivariate time series Forecasting (CGMF). Our CGMF consists of a continuous graph module incorporating differential equations to capture the long-range intra- and inter-relations of the temporal embedding sequence. We also introduce a controlled differential equation-based fusion mechanism that efficiently exploits multi-scale representations to form continuous evolutional dynamics and learn rich relations and patterns shared across different scales. Comprehensive experiments demonstrate the effectiveness of our method for a variety of datasets.

----

## [1970] Enhancing Dynamic GCN for Node Attribute Forecasting with Meta Spatial-Temporal Learning (Student Abstract)

**Authors**: *Bo Wu, Xun Liang, Xiangping Zheng, Jun Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27040](https://doi.org/10.1609/aaai.v37i13.27040)

**Abstract**:

Node attribute forecasting has recently attracted considerable attention. Recent attempts have thus far utilize dynamic graph convolutional network (GCN) to predict future node attributes. However, few prior works have notice that the complex spatial and temporal interaction between nodes, which will hamper the performance of dynamic GCN. In this paper, we propose a new dynamic GCN model named meta-DGCN, leveraging meta spatial-temporal tasks to enhance the ability of dynamic GCN for better capturing node attributes in the future. Experiments show that meta-DGCN effectively modeling comprehensive spatio-temporal correlations between nodes and outperforms state-of-the-art baselines on various real-world datasets.

----

## [1971] Tackling Safe and Efficient Multi-Agent Reinforcement Learning via Dynamic Shielding (Student Abstract)

**Authors**: *Wenli Xiao, Yiwei Lyu, John M. Dolan*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27041](https://doi.org/10.1609/aaai.v37i13.27041)

**Abstract**:

Multi-agent Reinforcement Learning (MARL) has been increasingly used in safety-critical applications but has no safety guarantees, especially during training. In this paper, we propose dynamic shielding, a novel decentralized MARL framework to ensure safety in both training and deployment phases. Our framework leverages Shield, a reactive system running in parallel with the reinforcement learning algorithm to monitor and correct agents' behavior. In our algorithm, shields dynamically split and merge according to the environment state in order to maintain decentralization and avoid conservative behaviors while enjoying formal safety guarantees. We demonstrate the effectiveness of MARL with dynamic shielding in the mobile navigation scenario.

----

## [1972] Long Legal Article Question Answering via Cascaded Key Segment Learning (Student Abstract)

**Authors**: *Shugui Xie, Lin Li, Jingling Yuan, Qing Xie, Xiaohui Tao*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27042](https://doi.org/10.1609/aaai.v37i13.27042)

**Abstract**:

Current sentence-level evidence extraction based methods may lose the discourse coherence of legal articles since they tend to make the extracted sentences scattered over the article.  To solve the problem, this paper proposes a Cascaded Answer-guided key segment learning framework for long Legal article Question Answering, namely CALQA. The framework consists of three cascaded modules: Sifter, Reader, and Responder. The Sifter transfers a long legal article into several segments and works in an answer-guided way by automatically sifting out key fact segments in a coarse-to-fine approach through multiple iterations. The Reader utilizes a set of attention mechanisms to obtain semantic representations of the question and key fact segments. Finally, considering it a multi-label classification task the Responder predicts final answers in a cascaded manner. CALQA outperforms state-of-the-art methods in CAIL 2021 Law dataset.

----

## [1973] Improving Dialogue Intent Classification with a Knowledge-Enhanced Multifactor Graph Model (Student Abstract)

**Authors**: *Huinan Xu, Jinhui Pang, Shuangyong Song, Bo Zou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27043](https://doi.org/10.1609/aaai.v37i13.27043)

**Abstract**:

Although current Graph Neural Network (GNN) based models achieved good performances in Dialogue Intent Classification (DIC), they leaf the inherent domain-specific knowledge out of consideration, leading to the lack of ability of acquiring fine-grained semantic information. In this paper, we propose a Knowledge-Enhanced Multifactor Graph (KEMG) Model for DIC. We firstly present a knowledge-aware utterance encoder with the help of a domain-specific knowledge graph, fusing token-level and entity-level semantic information, then design a heterogeneous dialogue graph encoder by explicitly modeling several factors that matter to contextual modeling of dialogues. Experiment results show that our proposed method outperforms other GNN-based methods on a dataset collected from a real-world online customer service dialogue system on the e-commerce website, JD.

----

## [1974] Class Incremental Learning for Task-Oriented Dialogue System with Contrastive Distillation on Internal Representations (Student Abstract)

**Authors**: *Qiancheng Xu, Min Yang, Binzong Geng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27044](https://doi.org/10.1609/aaai.v37i13.27044)

**Abstract**:

The ability to continually learn over time by grasping new knowledge and remembering previously learned experiences is essential for developing an online task-oriented dialogue system (TDS). In this paper, we work on the class incremental learning scenario where the TDS is evaluated without specifying the dialogue domain. We employ contrastive distillation on the intermediate representations of dialogues to learn transferable representations that suffer less from catastrophic forgetting. Besides, we provide a dynamic update mechanism to explicitly preserve the learned experiences by only updating the parameters related to the new task while keeping other parameters fixed. Extensive experiments demonstrate that our method significantly outperforms the strong baselines.

----

## [1975] ACCD: An Adaptive Clustering-Based Collusion Detector in Crowdsourcing (Student Abstract)

**Authors**: *Ruoyu Xu, Gaoxiang Li, Wei Jin, Austin Chen, Victor S. Sheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27045](https://doi.org/10.1609/aaai.v37i13.27045)

**Abstract**:

Crowdsourcing is a popular method for crowd workers to collaborate on tasks. However, workers coordinate and share answers during the crowdsourcing process. The term for this is "collusion". Copies from others and repeated submissions are detrimental to the quality of the assignments. The majority of the existing research on collusion detection is limited to ground truth problems (e.g., labeling tasks) and requires a predetermined threshold to be established in advance. In this paper, we aim to detect collusion behavior of workers in an adaptive way, and propose an Adaptive Clustering Based Collusion Detection approach (ACCD) for a broad range of task types and data types solved via crowdsourcing (e.g., continuous rating with or without distributions). Extensive experiments on both real-world and synthetic datasets show the superiority of ACCD over state-of-the-art approaches.

----

## [1976] Logic Error Localization and Correction with Machine Learning (Student Abstract)

**Authors**: *Zhenyu Xu, Victor S. Sheng, Keyi Lu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27046](https://doi.org/10.1609/aaai.v37i13.27046)

**Abstract**:

We aim to propose a system repairing programs with logic errors to be functionally correct among different programming languages. Logic error program repair has always been a thorny problem: First, a logic error is usually harder to repair than a syntax error in a program because it has no diagnostic feedback from compilers. Second, it requires inferring in different ranges (i.e., the distance of related code lines) and tracking symbols across its pseudocode, source code, and test cases. Third, the logic error datasets are scarce, since an ideal logic error dataset should contain lots of components during the development procedure of a program, including a program specification, pseudocode, source code, test cases, and test reports (i.e., test case failure report). In our work, we propose novel solutions to these challenges. First, we introduce pseudocode information to assist logic error localization and correction. We construct a code-pseudocode graph to connect symbols across a source code and its pseudocode and then apply a graph neural network to localize and correct logic errors. Second, we collect logic errors generated in the process of syntax error repairing via DrRepair from 500 programs in the SPoC dataset and reconstruct them to our single logic error dataset, which we leverage to train and evaluate our models. Our experimental results show that we achieve 99.39% localization accuracy and 19.20% full repair accuracy on logic errors with five-fold cross-validation. Based on our current work, we will replenish and construct more complete public logic error datasets and propose a novel system to comprehend different programming languages from several perspectives and correct logic errors to be functionally correct.

----

## [1977] Mask-Net: Learning Context Aware Invariant Features Using Adversarial Forgetting (Student Abstract)

**Authors**: *Hemant Yadav, Rajiv Ratn Shah*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27047](https://doi.org/10.1609/aaai.v37i13.27047)

**Abstract**:

Training a robust system, e.g., Speech to Text (STT), requires large datasets. Variability present in the dataset, such as unwanted nuances and biases, is the reason for the need for large datasets to learn general representations. In this work, we propose a novel approach to induce invariance using adversarial forgetting (AF). Our initial experiments on learning invariant features such as accent on the STT task achieve better generalizations in terms of word error rate (WER) compared to traditional models. We observe an absolute improvement of 2.2% and 1.3%  on out-of-distribution and in-distribution test sets, respectively.

----

## [1978] Adaptive Constraint Partition Based Optimization Framework for Large-Scale Integer Linear Programming (Student Abstract)

**Authors**: *Huigen Ye, Hongyan Wang, Hua Xu, Chengming Wang, Yu Jiang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27048](https://doi.org/10.1609/aaai.v37i13.27048)

**Abstract**:

Integer programming problems (IPs) are challenging to be solved efficiently due to the NP-hardness, especially for large-scale IPs. To solve this type of IPs, Large neighborhood search (LNS) uses an initial feasible solution and iteratively improves it by searching a large neighborhood around the current solution. However, LNS easily steps into local optima and ignores the correlation between variables to be optimized, leading to compromised performance. This paper presents a general adaptive constraint partition-based optimization framework (ACP) for large-scale IPs that can efficiently use any existing optimization solver as a subroutine. Specifically, ACP first randomly partitions the constraints into blocks, where the number of blocks is adaptively adjusted to avoid local optima. Then, ACP uses a subroutine solver to optimize the decision variables in a randomly selected block of constraints to enhance the variable correlation. ACP is compared with LNS framework with different subroutine solvers on four IPs and a real-world IP. The experimental results demonstrate that in specified wall-clock time ACP shows better performance than SCIP and Gurobi.

----

## [1979] Clustered Federated Learning for Heterogeneous Data (Student Abstract)

**Authors**: *Xue Yu, Ziyi Liu, Yifan Sun, Wu Wang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27049](https://doi.org/10.1609/aaai.v37i13.27049)

**Abstract**:

Federated Learning (FL) aims to achieve a global model via aggregating models from all devices. However, it can diverge when the data on the users’ devices are heterogeneous. To address this issue, we propose a novel clustered FL method (FPFC) based on a nonconvex pairwise fusion penalty. FPFC can automatically identify clusters without prior knowledge of the number of clusters and the set of devices in each cluster. Our method is implemented in parallel, updates only a subset of devices at each communication round, and allows each participating device to perform inexact computation. We also provide convergence guarantees of FPFC for general nonconvex losses. Experiment results demonstrate the advantages of FPFC over existing methods.

----

## [1980] Measuring the Privacy Leakage via Graph Reconstruction Attacks on Simplicial Neural Networks (Student Abstract)

**Authors**: *Huixin Zhan, Kun Zhang, Keyi Lu, Victor S. Sheng*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27050](https://doi.org/10.1609/aaai.v37i13.27050)

**Abstract**:

In this paper, we measure the privacy leakage via studying whether graph representations can be inverted to recover the graph used to generate them via graph reconstruction attack (GRA). We propose a GRA that recovers a graph's adjacency matrix from the representations via a graph decoder that minimizes the reconstruction loss between the partial graph and the reconstructed graph. We study three types of representations that are trained on the graph, i.e., representations output from graph convolutional network (GCN), graph attention network (GAT), and our proposed simplicial neural network (SNN) via a higher-order combinatorial Laplacian. Unlike the first two types of representations that only encode pairwise relationships, the third type of representation, i.e., SNN outputs, encodes higher-order interactions (e.g., homological features) between nodes. We find that the SNN outputs reveal the lowest privacy-preserving ability to defend the GRA, followed by those of GATs and GCNs, which indicates the importance of building more private representations with higher-order node information that could defend the potential threats, such as GRAs.

----

## [1981] DyCVAE: Learning Dynamic Causal Factors for Non-stationary Series Domain Generalization (Student Abstract)

**Authors**: *Weifeng Zhang, Zhiyuan Wang, Kunpeng Zhang, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27051](https://doi.org/10.1609/aaai.v37i13.27051)

**Abstract**:

Learning domain-invariant representations is a major task of out-of-distribution generalization. To address this issue, recent efforts have taken into accounting causality, aiming at learning the causal factors with regard to tasks. However, extending existing generalization methods for adapting non-stationary time series may be ineffective, because they fail to model the underlying causal factors due to temporal-domain shifts except for source-domain shifts, as pointed out by recent studies. To this end, we propose a novel model DyCVAE to learn dynamic causal factors. The results on synthetic and real datasets demonstrate the effectiveness of our proposed model for the task of generalization in time series domain.

----

## [1982] HaPPy: Harnessing the Wisdom from Multi-Perspective Graphs for Protein-Ligand Binding Affinity Prediction (Student Abstract)

**Authors**: *Xianfeng Zhang, Yanhui Gu, Guandong Xu, Yafei Li, Jinlan Wang, Zhenglu Yang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27052](https://doi.org/10.1609/aaai.v37i13.27052)

**Abstract**:

Gathering information from multi-perspective graphs is an essential issue for many applications especially for proteinligand binding affinity prediction. Most of traditional approaches obtained such information individually with low interpretability. In this paper, we harness the rich information from multi-perspective graphs with a general model, which abstractly represents protein-ligand complexes with better interpretability while achieving excellent predictive performance. In addition, we specially analyze the protein-ligand binding affinity problem, taking into account the heterogeneity of proteins and ligands. Experimental evaluations demonstrate the effectiveness of our data representation strategy on public datasets by fusing information from different perspectives.

----

## [1983] Graph of Graphs: A New Knowledge Representation Mechanism for Graph Learning (Student Abstract)

**Authors**: *Zhiwei Zhen, Yuzhou Chen, Murat Kantarcioglu, Yulia R. Gel*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27053](https://doi.org/10.1609/aaai.v37i13.27053)

**Abstract**:

Supervised graph classification is one of the most actively developing areas in machine learning (ML), with a broad range of domain applications, from social media to bioinformatics. Given a collection of graphs with categorical labels, the goal is to predict correct classes for unlabelled graphs. However, currently available ML tools view each such graph as a standalone entity and, as such, do not account for complex interdependencies among graphs. We propose a novel knowledge representation for graph learning called a {\it Graph of Graphs} (GoG). The key idea is to construct a new abstraction where each graph in the collection is represented by a node, while an edge then reflects similarity among the graphs. Such similarity can be assessed via a suitable graph distance. As a result, the graph classification problem can be then reformulated as a node classification problem. We show that the proposed new knowledge representation approach not only improves classification performance but substantially enhances robustness against label perturbation attacks.

----

## [1984] Exploiting High-Order Interaction Relations to Explore User Intent (Student Abstract)

**Authors**: *Xiangping Zheng, Xun Liang, Bo Wu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27054](https://doi.org/10.1609/aaai.v37i13.27054)

**Abstract**:

This paper studies the problem of exploring the user intent for session-based recommendations. Its challenges come from the uncertainty of user behavior and limited information. However, current endeavors cannot fully explore the mutual interactions among sessions and do not explicitly model the complex high-order relations among items. To circumvent these critical issues, we innovatively propose a HyperGraph Convolutional Contrastive framework (termed HGCC) that consists of two crucial tasks: 1) The session-based recommendation (SBR task) that aims to capture the beyond pair-wise relationships between items and sessions. 2) The self-supervised learning (SSL task) acted as the auxiliary task to boost the former task. By jointly optimizing the two tasks, the performance of the recommendation task achieves decent gains. Experiments on multiple real-world datasets demonstrate the superiority of the proposed approach over the state-of-the-art methods.

----

## [1985] Feature Decomposition for Reducing Negative Transfer: A Novel Multi-Task Learning Method for Recommender System (Student Abstract)

**Authors**: *Jie Zhou, Qian Yu, Chuan Luo, Jing Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27055](https://doi.org/10.1609/aaai.v37i13.27055)

**Abstract**:

We propose a novel multi-task learning method termed Feature Decomposition Network (FDN). The key idea of the proposed FDN is to reduce the phenomenon of feature redundancy by explicitly decomposing features into task-specific features and task-shared features with carefully designed constraints. Experimental results show that our proposed FDN can outperform the state-of-the-art (SOTA) methods by a noticeable margin on Ali-CCP.

----

## [1986] Model-Based Offline Weighted Policy Optimization (Student Abstract)

**Authors**: *Renzhe Zhou, Zongzhang Zhang, Yang Yu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27056](https://doi.org/10.1609/aaai.v37i13.27056)

**Abstract**:

A promising direction for applying reinforcement learning to the real world is learning from offline datasets. Offline reinforcement learning aims to learn policies from pre-collected datasets without online interaction with the environment. Due to the lack of further interaction, offline reinforcement learning faces severe extrapolation error, leading to policy learning failure. In this paper, we investigate the weighted Bellman update in model-based offline reinforcement learning. We explore uncertainty estimation in ensemble dynamics models, then use a variational autoencoder to fit the behavioral prior, and finally propose an algorithm called Model-Based Offline Weighted Policy Optimization (MOWPO), which uses a combination of model confidence and behavioral prior as weights to reduce the impact of inaccurate samples on policy optimization. Experiment results show that MOWPO achieves better performance than state-of-the-art algorithms, and both the model confidence weight and the behavioral prior weight can play an active role in offline policy optimization.

----

## [1987] ConceptX: A Framework for Latent Concept Analysis

**Authors**: *Firoj Alam, Fahim Dalvi, Nadir Durrani, Hassan Sajjad, Abdul Rafae Khan, Jia Xu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27057](https://doi.org/10.1609/aaai.v37i13.27057)

**Abstract**:

The opacity of deep neural networks remains a challenge in deploying solutions where explanation is as important as precision. We present ConceptX, a human-in-the-loop framework for interpreting and annotating latent representational space in pre-trained Language Models (pLMs). We use an unsupervised method to discover concepts learned in these models and enable a graphical interface for humans to generate explanations for the concepts. To facilitate the process, we provide auto-annotations of the concepts (based on traditional linguistic ontologies). Such annotations enable development of a linguistic resource that directly represents latent concepts learned within deep NLP models. These include not just traditional linguistic concepts, but also task-specific or sensitive concepts (words grouped based on gender or religious connotation) that helps the annotators to mark bias in the model. The framework consists of two parts (i) concept discovery and (ii) annotation platform.

----

## [1988] SOREO: A System for Safe and Autonomous Drones Fleet Navigation with Reinforcement Learning

**Authors**: *Réda Alami, Hakim Hacid, Lorenzo Bellone, Michal Barcis, Enrico Natalizio*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27058](https://doi.org/10.1609/aaai.v37i13.27058)

**Abstract**:

This demonstration introduces SOREO, a system that explores the possibility of extending UAVs autonomy through machine learning. It brings a contribution to the following problem: Having a fleet of drones and a geographic area, how to learn the shortest paths between any point with regards to the base points for optimal and safe package delivery?
Starting from a set of possible actions, a virtual design of a geographic location of interest, e.g., a city, and a reward value, SOREO is capable of learning not only how to prevent collisions with obstacles, e.g., walls and buildings, but also to find the shortest path between any two points, i.e., the base and the target. SOREO exploits based on the Q-learning algorithm.

----

## [1989] A Tool for Generating Controllable Variations of Musical Themes Using Variational Autoencoders with Latent Space Regularisation

**Authors**: *Berker Banar, Nick Bryan-Kinns, Simon Colton*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27059](https://doi.org/10.1609/aaai.v37i13.27059)

**Abstract**:

A common musical composition practice is to develop musical pieces using variations of musical themes. In this study, we present an interactive tool which can generate variations of musical themes in real-time using a variational autoencoder model. Our tool is controllable using semantically meaningful musical attributes via latent space regularisation technique to increase the explainability of the model. The tool is integrated into an industry standard digital audio workstation - Ableton Live - using the Max4Live device framework and can run locally on an average personal CPU rather than requiring a costly GPU cluster. In this way we demonstrate how cutting-edge AI research can be integrated into the exiting workflows of professional and practising musicians for use in the real-world beyond the research lab.

----

## [1990] Dagster: Parallel Structured Search

**Authors**: *Mark Alexander Burgess, Charles Gretton, Josh Milthorpe, Luke Croak, Thomas Willingham, Alwen Tiu*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27060](https://doi.org/10.1609/aaai.v37i13.27060)

**Abstract**:

We demonstrate Dagster, a system that implements a new approach to scheduling interdependent (Boolean) SAT search activities in high-performance computing (HPC) environments.
Our system takes as input a set of disjunctive clauses (i.e., DIMACS CNF) and a labelled directed acyclic graph (DAG) structure describing how the clauses are decomposed into a set of interrelated problems.
Component problems are solved using standard systematic backtracking search, which may optionally be coupled to (stochastic dynamic) local search and/or clause-strengthening processes.
We demonstrate Dagster using a new Graph Maximal Determinant combinatorial case study. This demonstration paper presents a new case study, and is adjunct to the longer accepted manuscript at the Pacific Rim International Conference on Artificial Intelligence (2022).

----

## [1991] AI-SNIPS: A Platform for Network Intelligence-Based Pharmaceutical Security

**Authors**: *Timothy A. Burt, Nikos Passas, Ioannis A. Kakadiaris*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27061](https://doi.org/10.1609/aaai.v37i13.27061)

**Abstract**:

This paper presents AI-SNIPS (AI Support for Network Intelligence-based Pharmaceutical Security), a production-ready platform that enables stakeholder decision-making, secure data sharing, and interdisciplinary research in the fight against Illicit, Substandard, and Falsified Medical Products (ISFMP). AI-SNIPS takes as input cases: a case consists of one or more URLs suspected of ISFMP activity. Cases can be supplemented with ground-truth structured data (labeled keywords) such as seller PII or case notes. First, AI-SNIPS scrapes and stores relevant images and text from the provided URLs without any user intervention. Salient features for predicting case similarity are extracted from the aggregated data using a combination of rule-based and machine-learning techniques and used to construct a seller network, with the nodes representing cases (sellers) and the edges representing the similarity between two sellers. Network analysis and community detection techniques are applied to extract seller clusters ranked by profitability and their potential to harm society. Lastly, AI-SNIPS provides interpretability by distilling common word/image similarities for each cluster into signature vectors. We validate the importance of AI-SNIPS's features for distinguishing large pharmaceutical affiliate networks from small ISFMP operations using an actual ISFMP lead sheet.

----

## [1992] TgrApp: Anomaly Detection and Visualization of Large-Scale Call Graphs

**Authors**: *Mirela T. Cazzolato, Saranya Vijayakumar, Xinyi Zheng, Namyong Park, Meng-Chieh Lee, Duen Horng Chau, Pedro Fidalgo, Bruno Lages, Agma J. M. Traina, Christos Faloutsos*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27062](https://doi.org/10.1609/aaai.v37i13.27062)

**Abstract**:

Given a million-scale dataset of who-calls-whom data containing imperfect labels, how can we detect existing and new fraud patterns? We propose TgrApp, which extracts carefully designed features and provides visualizations to assist analysts in spotting fraudsters and suspicious behavior. Our TgrApp method has the following properties: (a) Scalable, as it is linear on the input size; and (b) Effective, as it allows natural interaction with human analysts, and is applicable in both supervised and unsupervised settings.

----

## [1993] TUTORING: Instruction-Grounded Conversational Agent for Language Learners

**Authors**: *Hyungjoo Chae, Minjin Kim, Chaehyeong Kim, Wonseok Jeong, Hyejoong Kim, Junmyung Lee, Jinyoung Yeo*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27063](https://doi.org/10.1609/aaai.v37i13.27063)

**Abstract**:

In this paper, we propose Tutoring bot, a generative chatbot trained on a large scale of tutor-student conversations for English-language learning. To mimic a human tutor's behavior in language education, the tutor bot leverages diverse educational instructions and grounds to each instruction as additional input context for the tutor response generation. As a single instruction generally involves multiple dialogue turns to give the student sufficient speaking practice, the tutor bot is required to monitor and capture when the current instruction should be kept or switched to the next instruction. For that, the tutor bot is learned to not only generate responses but also infer its teaching action and progress on the current conversation simultaneously by a multi-task learning scheme. Our Tutoring bot is deployed under a non-commercial use license at https://tutoringai.com.

----

## [1994] HAPI Explorer: Comprehension, Discovery, and Explanation on History of ML APIs

**Authors**: *Lingjiao Chen, Zhihua Jin, Sabri Eyuboglu, Huamin Qu, Christopher Ré, Matei Zaharia, James Zou*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27064](https://doi.org/10.1609/aaai.v37i13.27064)

**Abstract**:

Machine learning prediction APIs offered by Google, Microsoft, Amazon, and many other providers have been continuously adopted in a plethora of applications, such as visual object detection, natural language comprehension, and speech recognition. Despite the importance of a systematic study and comparison of different APIs over time, this topic is currently under-explored because of the lack of data and user-friendly exploration tools. To address this issue, we present HAPI Explorer (History of API Explorer), an interactive system that offers easy access to millions of instances of commercial API applications collected in three years, prioritize attention on user-defined instance regimes, and explain interesting patterns across different APIs, subpopulations, and time periods via visual and natural languages. HAPI Explorer can facilitate further comprehension and exploitation of ML prediction APIs.

----

## [1995] EasyRec: An Easy-to-Use, Extendable and Efficient Framework for Building Industrial Recommendation Systems

**Authors**: *Mengli Cheng, Yue Gao, Guoqiang Liu, Hongsheng Jin*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27065](https://doi.org/10.1609/aaai.v37i13.27065)

**Abstract**:

We present EasyRec, an easy-to-use, extendable and efficient recommendation framework for building industrial recommendation systems. Our EasyRec framework is superior in the following aspects:first, EasyRec adopts a modular and pluggable design pattern to reduce the efforts to build custom models; second, EasyRec implements hyper-parameter optimization and feature selection algorithms to improve model performance automatically; third, EasyRec applies online learning to adapt to the ever-changing data distribution. The code is released: https://github.com/alibaba/EasyRec.

----

## [1996] edBB-Demo: Biometrics and Behavior Analysis for Online Educational Platforms

**Authors**: *Roberto Daza, Aythami Morales, Ruben Tolosana, Luis F. Gomez, Julian Fiérrez, Javier Ortega-Garcia*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27066](https://doi.org/10.1609/aaai.v37i13.27066)

**Abstract**:

We present edBB-Demo, a demonstrator of an AI-powered research platform for student monitoring in remote education. The edBB platform aims to study the challenges associated to user recognition and behavior understanding in digital platforms. This platform has been developed for data collection, acquiring signals from a variety of sensors including keyboard, mouse, webcam, microphone, smartwatch, and an Electroencephalography band. The information captured from the sensors during the student sessions is modelled in a multimodal learning framework. The demonstrator includes: i) Biometric user authentication in an unsupervised environment; ii) Human action recognition based on remote video analysis; iii) Heart rate estimation from webcam video; and iv) Attention level estimation from facial expression analysis.

----

## [1997] DUCK: A Drone-Urban Cyber-Defense Framework Based on Pareto-Optimal Deontic Logic Agents

**Authors**: *Tonmoay Deb, Jürgen Dix, Mingi Jeong, Cristian Molinaro, Andrea Pugliese, Alberto Quattrini Li, Eugene Santos Jr., V. S. Subrahmanian, Shanchieh Yang, Youzhi Zhang*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27067](https://doi.org/10.1609/aaai.v37i13.27067)

**Abstract**:

Drone based terrorist attacks are increasing daily. It is not expected to be long before drones are used to carry out terror attacks in urban areas. We have developed the DUCK multi-agent testbed that security agencies can use to simulate drone-based attacks by diverse actors and develop a combination of surveillance camera, drone, and cyber defenses against them.

----

## [1998] NL2LTL - a Python Package for Converting Natural Language (NL) Instructions to Linear Temporal Logic (LTL) Formulas

**Authors**: *Francesco Fuggitti, Tathagata Chakraborti*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27068](https://doi.org/10.1609/aaai.v37i13.27068)

**Abstract**:

This is a demonstration of our newly released Python package NL2LTL which leverages the latest in natural language understanding (NLU) and large language models (LLMs) to translate natural language instructions to linear temporal logic (LTL) formulas. This allows direct translation to formal languages that a reasoning system can use, while at the same time, allowing the end-user to provide inputs in natural language without having to understand any details of an underlying 
formal language. The package comes with support for a set of default LTL patterns, corresponding to popular DECLARE templates, but is also fully extensible to new formulas and user inputs. The package is open-source and is free to use for the AI community under the MIT license. Open Source: https://github.com/IBM/nl2ltl. Video Link: https://bit.ly/3dHW5b1

----

## [1999] DISPUTool 20: A Modular Architecture for Multi-Layer Argumentative Analysis of Political Debates

**Authors**: *Pierpaolo Goffredo, Elena Cabrio, Serena Villata, Shohreh Haddadan, Jhonatan Torres Sanchez*

**Conference**: *aaai 2023*

**URL**: [https://doi.org/10.1609/aaai.v37i13.27069](https://doi.org/10.1609/aaai.v37i13.27069)

**Abstract**:

Political debates are one of the most salient moments of an election campaign, where candidates are challenged to discuss the main contemporary and historical issues in a country. These debates represent a natural ground for argumentative analysis, which has always been employed to investigate political discourse structure and strategy in philosophy and linguistics. In this paper, we present DISPUTool 2.0, an automated tool which relies on Argument Mining methods to analyse the political debates from the US presidential campaigns to extract argument components (i.e., premise and claim) and relations (i.e., support and attack), and highlight fallacious arguments. DISPUTool 2.0 allows also for the automatic analysis of a piece of a debate proposed by the user to identify and classify the arguments contained in the text. A REST API is provided to exploit the tool's functionalities.

----



[Go to the previous page](AAAI-2023-list09.md)

[Go to the next page](AAAI-2023-list11.md)

[Go to the catalog section](README.md)