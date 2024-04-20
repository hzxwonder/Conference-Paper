## [2600] ETDPC: A Multimodality Framework for Classifying Pages in Electronic Theses and Dissertations

**Authors**: *Muntabir Hasan Choudhury, Lamia Salsabil, William A. Ingram, Edward A. Fox, Jian Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30324](https://doi.org/10.1609/aaai.v38i21.30324)

**Abstract**:

Electronic theses and dissertations (ETDs) have been proposed, advocated, and generated for more than 25 years. Although ETDs are hosted by commercial or institutional digital library repositories, they are still an understudied type of scholarly big data, partially because they are usually longer than conference and journal papers. Segmenting ETDs will allow researchers to study sectional content. Readers can navigate to particular pages of interest, to discover and explore the content buried in these long documents. Most existing frameworks on document page classification are designed for classifying general documents, and perform poorly on ETDs. In this paper, we propose ETDPC. Its backbone is a two-stream multimodal model with a cross-attention network to classify ETD pages into 13 categories. To overcome the challenge of imbalanced labeled samples, we augmented data for minority categories and employed a hierarchical classifier. ETDPC outperforms the state-of-the-art models in all categories, achieving an F1 of 0.84 -- 0.96 for 9 out of 13 categories. We also demonstrated its data efficiency. The code and data can be found on GitHub (https://github.com/lamps-lab/ETDMiner/tree/master/etd_segmentation).

----

## [2601] Data-Driven Structural Fire Risk Prediction for City Properties

**Authors**: *Rupasree Dey, Alan Fern*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30325](https://doi.org/10.1609/aaai.v38i21.30325)

**Abstract**:

Fire Departments conduct inspections to prevent fires but it is unclear how to best allocate their limited inspection resources across the properties in a city. Currently, they use their intuition and experience to decide on which properties to inspect and lack a data-driven approach that could lead to a more principled use of inspection resources. The main contribution of this paper is to investigate such an approach, based on machine learning for predicting a fire risk score for properties in a city based on historical fire-incident data. These scores can then be used to help prioritize inspection resources toward higher-risk properties. We present a case study using data from a South Dakota fire department which contains information about properties in a city along with records of fire in- incidents. We use this data consisting of more than 72,000 properties to train a machine learning model to predict fire risk and evaluate its ability to rank the fire risk of properties in the city. We conduct and analyze experiments with variations of XG-Boost, which is an algorithm well-suited to the challenges in application, including missing data and a highly-skewed class distribution. Our evaluation of the model-generated rankings, based on ranking metrics, shows that the model significantly outperforms random rankings and other natural baselines. We also analyze the feature importance computed for the models, which provides further insight into the model behavior. This model has been integrated into an interface for displaying the rankings across a city and is ready for beta testing.

----

## [2602] Pharmacokinetics-Informed Neural Network for Predicting Opioid Administration Moments with Wearable Sensors

**Authors**: *Bhanu Teja Gullapalli, Stephanie Carreiro, Brittany P. Chapman, Eric L. Garland, Tauhidur Rahman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30326](https://doi.org/10.1609/aaai.v38i21.30326)

**Abstract**:

Long-term and high-dose prescription opioid use places individuals at risk for opioid misuse, opioid use disorder (OUD), and overdose. Existing methods for monitoring opioid use and detecting misuse rely on self-reports, which are prone to reporting bias, and toxicology testing, which may be infeasible in outpatient settings. Although wearable technologies for monitoring day-to-day health metrics have gained significant traction in recent years due to their ease of use, flexibility, and advancements in sensor technology, their application within the opioid use space remains underexplored. In the current work, we demonstrate that oral opioid administrations can be detected using physiological signals collected from a wrist sensor. More importantly, we show that models informed by opioid pharmacokinetics increase reliability in predicting the timing of opioid administrations. Forty-two individuals who were prescribed opioids as a part of their medical treatment in-hospital and after discharge were enrolled. Participants wore a wrist sensor throughout the study, while opioid administrations were tracked using electronic medical records and self-reports. We collected 1,983 hours of sensor data containing 187 opioid administrations from the inpatient setting and 927 hours of sensor data containing 40 opioid administrations from the outpatient setting. We demonstrate that a self-supervised pre-trained model, capable of learning the canonical time series of plasma concentration of the drug derived from opioid pharmacokinetics, can reliably detect opioid administration in both settings. Our work suggests the potential of pharmacokinetic-informed, data-driven models to objectively detect opioid use in daily life.

----

## [2603] VeriCompress: A Tool to Streamline the Synthesis of Verified Robust Compressed Neural Networks from Scratch

**Authors**: *Sawinder Kaur, Yi Xiao, Asif Salekin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30327](https://doi.org/10.1609/aaai.v38i21.30327)

**Abstract**:

AI's widespread integration has led to neural networks (NN) deployment on edge and similar limited-resource platforms for safety-critical scenarios. Yet, NN's fragility raises concerns about reliable inference. Moreover, constrained platforms demand compact networks. This study introduces VeriCompress, a tool that automates the search and training of compressed models with robustness guarantees. These models are well-suited for safety-critical applications and adhere to predefined architecture and size limitations, making them deployable on resource-restricted platforms. The method trains models 2-3 times faster than the state-of-the-art approaches, surpassing them by average accuracy and robustness gains of 15.1 and 9.8 percentage points, respectively. When deployed on a resource-restricted generic platform, these models require 5-8 times less memory and 2-4 times less inference time than models used in verified robustness literature. Our comprehensive evaluation across various model architectures and datasets, including MNIST, CIFAR, SVHN, and a relevant pedestrian detection dataset, showcases VeriCompress's capacity to identify compressed verified robust models with reduced computation overhead compared to current standards. This underscores its potential as a valuable tool for end users, such as developers of safety-critical applications on edge or Internet of Things platforms, empowering them to create suitable models for safety-critical, resource-constrained platforms in their respective domains.

----

## [2604] Using Adaptive Bandit Experiments to Increase and Investigate Engagement in Mental Health

**Authors**: *Harsh Kumar, Tong Li, Jiakai Shi, Ilya Musabirov, Rachel Kornfield, Jonah Meyerhoff, Ananya Bhattacharjee, Chris J. Karr, Theresa Nguyen, David C. Mohr, Anna N. Rafferty, Sofia S. Villar, Nina Deliu, Joseph Jay Williams*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30328](https://doi.org/10.1609/aaai.v38i21.30328)

**Abstract**:

Digital mental health (DMH) interventions, such as text-message-based lessons and activities, offer immense potential for accessible mental health support. While these interventions can be effective, real-world experimental testing can further enhance their design and impact. Adaptive experimentation, utilizing algorithms like Thompson Sampling for (contextual) multi-armed bandit (MAB) problems, can lead to continuous improvement and personalization. However, it remains unclear when these algorithms can simultaneously increase user experience rewards and facilitate appropriate data collection for social-behavioral scientists to analyze with sufficient statistical confidence. Although a growing body of research addresses the practical and statistical aspects of MAB and other adaptive algorithms, further exploration is needed to assess their impact across diverse real-world contexts. This paper presents a software system developed over two years that allows text-messaging intervention components to be adapted using bandit and other algorithms while collecting data for side-by-side comparison with traditional uniform random non-adaptive experiments. We evaluate the system by deploying a text-message-based DMH intervention to 1100 users, recruited through a large mental health non-profit organization, and share the path forward for deploying this system at scale. This system not only enables applications in mental health but could also serve as a model testbed for adaptive experimentation algorithms in other domains.

----

## [2605] Improving Health Information Access in the World's Largest Maternal Mobile Health Program via Bandit Algorithms

**Authors**: *Arshika Lalan, Shresth Verma, Paula Rodriguez Diaz, Panayiotis Danassis, Amrita Mahale, Kumar Madhu Sudan, Aparna Hegde, Milind Tambe, Aparna Taneja*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30329](https://doi.org/10.1609/aaai.v38i21.30329)

**Abstract**:

Harnessing the wide-spread availability of cell phones, many nonprofits have launched mobile health (mHealth) programs to deliver information via voice or text to beneficiaries in underserved communities, with maternal and infant health being a key area of such mHealth programs. Unfortunately, dwindling listenership is a major challenge, requiring targeted interventions using limited resources. This paper focuses on Kilkari, the world's largest mHealth program for maternal and child care -- with over 3 million active subscribers at a time --  launched by India's Ministry of Health and Family Welfare (MoHFW) and run by the non-profit ARMMAN. We present a system called CHAHAK that aims to reduce automated dropouts as well as boost engagement with the program through the strategic allocation of interventions to beneficiaries. Past work in a similar domain has focused on a much smaller scale mHealth program and used markovian restless multiarmed bandits to optimize a single limited intervention resource. However this paper demonstrates the challenges in adopting a markovian approach in Kilkari; therefore CHAHAK instead relies on non-markovian time-series restless bandits, and optimizes a layered set of multiple interventions to improve listenership. We use real Kilkari data from the Odisha state in India to show CHAHAK's effectiveness in harnessing multiple interventions to boost listenership, benefiting marginalized communities. When deployed CHAHAK will assist the largest maternal mHealth program to date.

----

## [2606] Combining Machine Learning and Queueing Theory for Data-Driven Incarceration-Diversion Program Management

**Authors**: *Bingxuan Li, Antonio Castellanos, Pengyi Shi, Amy R. Ward*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30330](https://doi.org/10.1609/aaai.v38i21.30330)

**Abstract**:

Incarceration-diversion programs have proven effective in reducing recidivism. Accurate prediction of the number of individuals with different characteristics in the program and their program outcomes based on given eligibility criteria is crucial for successful implementation, because this prediction serves as the foundation for determining the appropriate program size and the consequent staffing requirements. However, this task poses challenges due to the complexities arising from varied outcomes and lengths-of-stay for the diverse individuals in incarceration-diversion programs. In collaboration with an Illinois government agency, we develop a framework to address these issues. Our framework combines ML and queueing model simulation, providing accurate predictions for the program census and interpretable insights into program dynamics and the impact of different decisions in counterfactual scenarios. Additionally, we deploy a user-friendly web app beta-version that allows program managers to visualize census data by counties and race groups. We showcase two decision support use cases: Changing program admission criteria and launching similar programs in new counties.

----

## [2607] TelTrans: Applying Multi-Type Telecom Data to Transportation Evaluation and Prediction via Multifaceted Graph Modeling

**Authors**: *ChungYi Lin, Shen-Lung Tung, Hung-Ting Su, Winston H. Hsu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30331](https://doi.org/10.1609/aaai.v38i21.30331)

**Abstract**:

To address the limitations of traffic prediction from location-bound detectors, we present Geographical Cellular Traffic (GCT) flow, a novel data source that leverages the extensive coverage of cellular traffic to capture mobility patterns. Our extensive analysis validates its potential for transportation. Focusing on vehicle-related GCT flow prediction, we propose a graph neural network that integrates multivariate, temporal, and spatial facets for improved accuracy. Experiments reveal our model's superiority over baselines, especially in long-term predictions. We also highlight the potential for GCT flow integration into transportation systems.

----

## [2608] Symbol Description Reading

**Authors**: *Karol Lynch, Bradley Eck, Joern Ploennigs*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30332](https://doi.org/10.1609/aaai.v38i21.30332)

**Abstract**:

Mathematical formulas give concise representations of a document's key ideas in many natural sciences and engineering domains. The symbols that make up formulas carry semantic meaning that may differ by document or equation. What does ? mean in a given paper? Interpreting the symbols that comprise formulas requires identifying descriptions from the surrounding text. We approach this task of symbol description reading as an application of current AI technologies targeting the tuning of large language models for particular domains and automation of machine learning. Our pipeline integrates AI question answering and natural language processing to read symbol descriptions. We consider extractive and generative AI model variations and apply our pipeline on two example tasks of symbol description reading. Promising results provide motivation for wider deployment for which we describe a microservice architecture and related challenges.

----

## [2609] A Framework for Mining Speech-to-Text Transcripts of the Customer for Automated Problem Remediation

**Authors**: *Prateeti Mohapatra, Gargi Dasgupta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30333](https://doi.org/10.1609/aaai.v38i21.30333)

**Abstract**:

Technical support services get several thousand voice calls
every year. These calls vary across a range of technical issues
or maintenance requests for a suite of hardware and software
products. On receiving the call, a support agent creates a ser-
vice request artifact that contains her interpretation of the
customer’s problem. This service request goes through the life
cycle of the problem remediation process with the resolution
also being recorded as part of the service request. It has been
empirically observed that the actual complaint voiced by the
customer is often different from the recorded interpretation
in the service request. The service request created by sup-
port agents runs the risk of missing key information elements
present in the customer voice records. In this paper, we build
a framework that taps into voice calls and uses unsupervised
and supervised learning methods to enrich the service requests
with additional information. The enriched data is then used
for automated problem resolution.

----

## [2610] Redefining the Laparoscopic Spatial Sense: AI-Based Intra- and Postoperative Measurement from Stereoimages

**Authors**: *Leopold Müller, Patrick Hemmer, Moritz Queisner, Igor Sauer, Simeon Allmendinger, Johannes Jakubik, Michael Vössing, Niklas Kühl*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30334](https://doi.org/10.1609/aaai.v38i21.30334)

**Abstract**:

A significant challenge in image-guided surgery is the accurate measurement task of relevant structures such as vessel segments, resection margins, or bowel lengths. While this task is an essential component of many surgeries, it involves substantial human effort and is prone to inaccuracies. In this paper, we develop a novel human-AI-based method for laparoscopic measurements utilizing stereo vision that has been guided by practicing surgeons. Based on a holistic qualitative requirements analysis, this work proposes a comprehensive measurement method, which comprises state-of-the-art machine learning architectures, such as RAFT-Stereo and YOLOv8. The developed method is assessed in various realistic experimental evaluation environments. Our results outline the potential of our method achieving high accuracies in distance measurements with errors below 1 mm. Furthermore, on-surface measurements demonstrate robustness when applied in challenging environments with textureless regions. Overall, by addressing the inherent challenges of image-guided surgery, we lay the foundation for a more robust and accurate solution for intra- and postoperative measurements, enabling more precise, safe, and efficient surgical procedures.

----

## [2611] BERTground: A Transformer-Based Model of Background Spectra on the ISS-Based NICER Space Telescope

**Authors**: *Anh N. Nhu, Abderahmen Zoghbi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30335](https://doi.org/10.1609/aaai.v38i21.30335)

**Abstract**:

The Neutron star Interior Composition Explorer (NICER) is an International Space Station (ISS)-based Space Telescope developed by NASA and devoted to the study of high-energy X-Ray sources in the universe, including but not limited to neutron stars, pulsars, and black holes in stellar systems and active galactic nuclei (AGN). One prominent problem with NICER observations is the highly variable background spectra, obscuring actual signals of astrophysical sources and negatively affecting scientific analysis of the targets. Therefore, obtaining accurate estimations of the background spectra is crucial to filter the noise and facilitate better scientific discoveries of new astronomical objects. In this paper, we propose the very first Deep Neural Network architecture to model the NICER background spectra variation using information about the spacecraft and telescope associated with each observation. In particular, we develop a BERT-based architecture with tokenizers applied to different groups of features in our tabular dataset. We also introduce an adapted Tabular Deep Residual Network architecture as the predictor following the Transformer modules in our network. We show that our model outperforms the current state-of-the-art background model developed by the NICER team in most evaluation metrics. Finally, we discuss pathways and future work for the deployment of this model on NASA’s next versions of HEASARC Software packages.

----

## [2612] AutoMixer for Improved Multivariate Time-Series Forecasting on Business and IT Observability Data

**Authors**: *Santosh Palaskar, Vijay Ekambaram, Arindam Jati, Neelamadhav Gantayat, Avirup Saha, Seema Nagar, Nam H. Nguyen, Pankaj Dayama, Renuka Sindhgatta, Prateeti Mohapatra, Harshit Kumar, Jayant Kalagnanam, Nandyala Hemachandra, Narayan Rangaraj*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30336](https://doi.org/10.1609/aaai.v38i21.30336)

**Abstract**:

The efficiency of business processes relies on business key performance indicators (Biz-KPIs), that can be negatively impacted by IT failures. Business and IT Observability (BizITObs) data fuses both Biz-KPIs and IT event channels together as multivariate time series data. Forecasting Biz-KPIs in advance can enhance efficiency and revenue through proactive corrective measures. However, BizITObs data generally exhibit both useful and noisy inter-channel interactions between Biz-KPIs and IT events that need to be effectively decoupled. This leads to suboptimal forecasting performance when existing multivariate forecasting models are employed. To address this, we introduce AutoMixer, a time-series Foundation Model (FM) approach, grounded on the novel technique of channel-compressed pretrain and finetune workflows. AutoMixer leverages an AutoEncoder for channel-compressed pretraining and integrates it with the advanced TSMixer model for multivariate time series forecasting. This fusion greatly enhances the potency of TSMixer for accurate forecasts and also generalizes well across several downstream tasks. Through detailed experiments and dashboard analytics, we show AutoMixer's capability to consistently improve the Biz-KPI's forecasting accuracy (by 11-15%) which directly translates to actionable business insights.

----

## [2613] Attention-Based Models for Snow-Water Equivalent Prediction

**Authors**: *Krishu K. Thapa, Bhupinderjeet Singh, Supriya Savalkar, Alan Fern, Kirti Rajagopalan, Ananth Kalyanaraman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30337](https://doi.org/10.1609/aaai.v38i21.30337)

**Abstract**:

Snow Water-Equivalent (SWE)—the amount of water available if snowpack is melted—is a key decision variable used by water management agencies to make irrigation, flood control, power generation, and drought management decisions. SWE values vary spatiotemporally—affected by weather, topography, and other environmental factors. While daily SWE can be measured by Snow Telemetry (SNOTEL) stations with requisite instrumentation, such stations are spatially sparse requiring interpolation techniques to create spatiotemporal complete data. While recent efforts have explored machine learning (ML) for SWE prediction, a number of recent ML advances have yet to be considered. The main contribution of this paper is to explore one such ML advance, attention mechanisms, for SWE prediction. Our hypothesis is that attention has a unique ability to capture and exploit correlations that may exist across locations or the temporal spectrum (or both). We present a generic attention-based modeling framework for SWE prediction and adapt it to capture spatial attention and temporal attention. Our experimental results on 323 SNOTEL stations in the Western U.S. demonstrate that our attention-based models outperform other machine-learning approaches. We also provide key results highlighting the differences between spatial and temporal attention in this context and a roadmap toward deployment for generating spatially-complete SWE maps.

----

## [2614] Interactive Mars Image Content-Based Search with Interpretable Machine Learning

**Authors**: *Bhavan Vasu, Steven Lu, Emily Dunkel, Kiri L. Wagstaff, Kevin Grimes, Michael McAuley*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30338](https://doi.org/10.1609/aaai.v38i21.30338)

**Abstract**:

The NASA Planetary Data System (PDS) hosts millions of images of planets, moons, and other bodies collected throughout many missions. The ever-expanding nature of data and user engagement demands an interpretable content classification system to support scientific discovery and individual curiosity. In this paper, we leverage a prototype-based architecture to enable users to understand and validate the evidence used by a classifier trained on images from the  Mars Science Laboratory (MSL) Curiosity rover mission. In addition to providing explanations, we investigate the diversity and correctness of evidence used by the content-based classifier. The work presented in this paper will be deployed on the PDS Image Atlas, replacing its non-interpretable counterpart.

----

## [2615] Tell Me What Is Good about This Property: Leveraging Reviews for Segment-Personalized Image Collection Summarization

**Authors**: *Monika Wysoczanska, Moran Beladev, Karen Lastmann Assaraf, Fengjun Wang, Ofri Kleinfeld, Gil Amsalem, Hadas Harush Boker*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30339](https://doi.org/10.1609/aaai.v38i21.30339)

**Abstract**:

Image collection summarization techniques aim to present a compact representation of an image gallery through a carefully selected subset of images that captures its semantic content. When it comes to web content, however, the ideal selection can vary based on the user's specific intentions and preferences. This is particularly relevant at Booking.com, where presenting properties and their visual summaries that align with users' expectations is crucial. To address this challenge, in this work, we consider user intentions in the summarization of property visuals by analyzing property reviews and extracting the most significant aspects mentioned by users. By incorporating the insights from reviews in our visual summaries, we enhance the summaries by presenting the relevant content to a user. Moreover, we achieve it without the need for costly annotations. Our experiments, including human perceptual studies, demonstrate the superiority of our cross-modal approach, which we coin as CrossSummarizer over the no-personalization and image-based clustering baselines.

----

## [2616] Optimizing IT FinOps and Sustainability through Unsupervised Workload Characterization

**Authors**: *Xi Yang, Rohan R. Arora, Saurabh Jha, Chandra Narayanaswami, Cheuk Lam, Jerrold Leichter, Yu Deng, Daby M. Sow*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30340](https://doi.org/10.1609/aaai.v38i21.30340)

**Abstract**:

The widespread adoption of public and hybrid clouds, along with elastic resources and various automation tools for dynamic deployment, has accelerated the rapid provisioning of compute resources as needed. Despite these advancements, numerous resources persist unnecessarily due to factors such as poor digital hygiene, risk aversion, or the absence of effective tools, resulting in substantial costs and energy consumption. Existing threshold-based techniques prove inadequate in effectively addressing this challenge. To address this issue, we propose an unsupervised machine learning framework to automatically identify resources that can be de-provisioned completely or summoned on a schedule. Application of this approach to enterprise data has yielded promising initial results, facilitating the segregation of productive workloads with recurring demands from non-productive ones.

----

## [2617] Building Higher-Order Abstractions from the Components of Recommender Systems

**Authors**: *Serdar Kadioglu, Bernard Kleynhans*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30341](https://doi.org/10.1609/aaai.v38i21.30341)

**Abstract**:

We present a modular recommender system framework that tightly integrates yet maintains the independence of individual components, thus satisfying two of the most critical aspects of industrial applications, generality and specificity. On the one hand, we ensure that each component remains self-contained and is ready to serve in other applications beyond recommender systems. On the other hand, when these components are combined, a unified theme emerges for recommender systems. We present the details of each component in the context of recommender systems and other applications. We release each component as an open-source library, and most importantly, we release their integration under MAB2REC, an industry-strength open-source software for building bandit-based recommender systems. By bringing standalone components together, Mab2Rec realizes a powerful and scalable toolchain to build and deploy business-relevant personalization applications. Finally, we share our experience and best practices for user training, adoption, performance evaluation, deployment, and model governance within the enterprise and the broader community.

----

## [2618] End-to-End Phase Field Model Discovery Combining Experimentation, Crowdsourcing, Simulation and Learning

**Authors**: *Md. Nasim, Xinghang Zhang, Anter El-Azab, Yexiang Xue*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30342](https://doi.org/10.1609/aaai.v38i21.30342)

**Abstract**:

The availability of tera-byte scale experiment data calls for AI driven approaches which automatically discover scientific models from data. Nonetheless, significant challenges present in AI-driven scientific discovery: (i) The annotation of large scale datasets requires fundamental re-thinking in developing scalable crowdsourcing tools. (ii) The learning of scientific models from data calls for innovations beyond black-box neural nets. (iii) Novel visualization & diagnosis tools are needed for the collaboration of experimental and theoretical physicists, and computer scientists. We present Phase-Field-Lab platform for end-to-end phase field model discovery, which automatically discovers phase field physics models from experiment data, integrating experimentation, crowdsourcing, simulation and learning. Phase-Field-Lab combines (i) a streamlined annotation tool which reduces the annotation time (by ~50-75%), while increasing annotation accuracy compared to baseline; (ii) an end-to-end neural model which automatically learns phase field models from data by embedding phase field simulation and existing domain knowledge into learning; and (iii) novel interfaces and visualizations to integrate our platform into the scientific discovery cycle of domain scientists. Our platform is deployed in the analysis of nano-structure evolution in materials under extreme conditions (high temperature and irradiation). Our approach reveals new properties of nano-void defects, which otherwise cannot be detected via manual analysis.

----

## [2619] A Model for Estimating the Economic Costs of Computer Vision Systems That Use Deep Learning

**Authors**: *Neil Thompson, Martin Fleming, Benny J. Tang, Anna M. Pastwa, Nicholas Borge, Brian C. Goehring, Subhro Das*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30343](https://doi.org/10.1609/aaai.v38i21.30343)

**Abstract**:

Deep learning, the most important subfield of machine learning and artificial intelligence (AI) over the last decade, is considered one of the fundamental technologies underpinning the Fourth Industrial Revolution. But despite its record-breaking history, deep learning’s enormous appetite for compute and data means that sometimes it can be too costly to practically use. In this paper, we connect technical insights from deep learning scaling laws and transfer learning with the economics of IT to propose a framework for estimating the cost of deep learning computer vision systems to achieve a desired level of accuracy. Our tool can be of practical use to AI practitioners in industry or academia to guide investment decisions.

----

## [2620] Automated State Estimation for Summarizing the Dynamics of Complex Urban Systems Using Representation Learning

**Authors**: *Maira Alvi, Tim French, Philip Keymer, Rachel Cardell-Oliver*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30344](https://doi.org/10.1609/aaai.v38i21.30344)

**Abstract**:

Complex urban systems can be difficult to monitor, diagnose and manage because the complete states of such systems are only partially observable with sensors. State estimation techniques can be used to determine the
underlying dynamic behavior of such complex systems with their highly non-linear processes and external time-variant influences.
States can be estimated by clustering observed sensor readings.  However,
clustering performance degrades as the number of sensors and readings (i.e. feature dimension) increases. To address this problem, we propose a framework that learns a feature-centric lower dimensional representation of data for clustering to support analysis of system dynamics. We propose Unsupervised Feature Attention with Compact Representation (UFACR) to rank features contributing to a cluster assignment. These weighted features are then used to learn a reduced-dimension temporal representation of the data with a deep-learning model. The resulting low-dimensional representation can be effectively clustered into states. UFACR is evaluated on real-world and synthetic wastewater treatment plant data sets, and feature ranking outcomes were validated by Wastewater treatment domain experts. Our quantitative and qualitative experimental analyses demonstrate the effectiveness of UFACR for uncovering system dynamics in an automated and unsupervised manner to offer guidance to wastewater engineers to enhance industrial productivity and treatment efficiency.

----

## [2621] A Generalizable Theory-Driven Agent-Based Framework to Study Conflict-Induced Forced Migration

**Authors**: *Zakaria Mehrab, Logan Stundal, Srinivasan Venkatramanan, Samarth Swarup, Bryan Leroy Lewis, Henning S. Mortveit, Christopher L. Barrett, Abhishek Pandey, Chad R. Wells, Alison P. Galvani, Burton H. Singer, Seyed M. Moghadas, David Leblang, Rita R. Colwell, Madhav V. Marathe*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30345](https://doi.org/10.1609/aaai.v38i21.30345)

**Abstract**:

Large-scale population displacements arising from conflict-induced forced migration generate uncertainty and introduce several policy challenges. Addressing these concerns requires an interdisciplinary approach that integrates knowledge from both computational modeling and social sciences. We propose a generalized computational agent-based modeling framework grounded by Theory of Planned Behavior to model conflict-induced migration outflows within Ukraine during the start of that conflict in 2022. Existing migration modeling frameworks that attempt to address policy implications primarily focus on destination while leaving absent a generalized computational framework grounded by social theory focused on the conflict-induced region. We propose an agent-based framework utilizing a spatiotemporal gravity model and a Bi-threshold model over a Graph Dynamical System to update migration status of agents in conflict-induced regions at fine temporal and spatial granularity. This approach significantly outperforms previous work when examining the case of Russian invasion in Ukraine. Policy implications of the proposed framework are demonstrated by modeling the migration behavior of Ukrainian civilians attempting to flee from regions encircled by Russian forces. We also showcase the generalizability of the model by simulating a past conflict in Burundi, an alternative conflict setting. Results demonstrate the utility of the framework for assessing conflict-induced migration in varied settings as well as identifying vulnerable civilian populations.

----

## [2622] AI Evaluation Authorities: A Case Study Mapping Model Audits to Persistent Standards

**Authors**: *Arihant Chadda, Sean McGregor, Jesse Hostetler, Andrea Brennen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30346](https://doi.org/10.1609/aaai.v38i21.30346)

**Abstract**:

Intelligent system audits are labor-intensive assurance activities that are typically performed once and discarded along with the opportunity to programmatically test all similar products for the market. This study illustrates how several incidents (i.e., harms) involving Named Entity Recognition (NER) can be prevented by scaling up a previously-performed audit of NER systems. The audit instrument's diagnostic capacity is maintained through a security model that protects the underlying data (i.e., addresses Goodhart's Law). An open-source evaluation infrastructure is released along with an example derived from a real-world audit that reports aggregated findings without exposing the underlying data.

----

## [2623] When Your AI Becomes a Target: AI Security Incidents and Best Practices

**Authors**: *Kathrin Grosse, Lukas Bieringer, Tarek R. Besold, Battista Biggio, Alexandre Alahi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30347](https://doi.org/10.1609/aaai.v38i21.30347)

**Abstract**:

In contrast to vast academic efforts to study AI security, few real-world reports of AI security incidents exist. Released incidents prevent a thorough investigation of the attackers' motives, as crucial information about the company and AI application is missing. As a consequence, it often remains unknown how to avoid incidents.    
We tackle this gap and combine previous reports with freshly collected incidents to a small database of 32 AI security incidents. We analyze the attackers' target and goal, influencing factors, causes, and mitigations. Many incidents stem from non-compliance with best practices in security and privacy-enhancing technologies. 
In the case of direct AI attacks, access control may provide some mitigation, but there is little scientific work on best practices. Our paper is thus a call for action to address these gaps.

----

## [2624] AI Risk Profiles: A Standards Proposal for Pre-deployment AI Risk Disclosures

**Authors**: *Eli Sherman, Ian W. Eisenberg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30348](https://doi.org/10.1609/aaai.v38i21.30348)

**Abstract**:

As AI systems’ sophistication and proliferation have increased, awareness of the risks has grown proportionally. The AI industry is increasingly emphasizing the need for transparency, with proposals ranging from standardizing use of technical disclosures, like model cards, to regulatory licensing regimes. Since the AI value chain is complicated, with actors bringing varied expertise, perspectives, and values, it is crucial that consumers of transparency disclosures be able to understand the risks of the AI system in question. In this paper we propose a risk profiling standard which can guide downstream decision-making, including triaging further risk assessment, informing procurement and deployment, and directing regulatory frameworks. The standard is built on our proposed taxonomy of AI risks, which distills the wide variety of risks proposed in the literature into a high-level categorization. We outline the myriad data sources needed to construct informative Risk Profiles and propose a template and methodology for collating risk information into a standard, yet flexible, structure. We apply this methodology to a number of prominent AI systems using publicly available information. To conclude, we discuss design decisions for the profiles and future work.

----

## [2625] Merging AI Incidents Research with Political Misinformation Research: Introducing the Political Deepfakes Incidents Database

**Authors**: *Christina P. Walker, Daniel S. Schiff, Kaylyn Jackson Schiff*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30349](https://doi.org/10.1609/aaai.v38i21.30349)

**Abstract**:

This article presents the Political Deepfakes Incidents Database (PDID), a collection of politically-salient deepfakes, encompassing synthetically-created videos, images, and less-sophisticated `cheapfakes.' The project is driven by the rise of generative AI in politics, ongoing policy efforts to address harms, and the need to connect AI incidents and political communication research. The database contains political deepfake content, metadata, and researcher-coded descriptors drawn from political science, public policy, communication, and misinformation studies. It aims to help reveal the prevalence, trends, and impact of political deepfakes, such as those featuring major political figures or events. The PDID can benefit policymakers, researchers, journalists, fact-checkers, and the public by providing insights into deepfake usage, aiding in regulation, enabling in-depth analyses, supporting fact-checking and trust-building efforts, and raising awareness of political deepfakes. It is suitable for research and application on media effects, political discourse, AI ethics, technology governance, media literacy, and countermeasures.

----

## [2626] Does Any AI-Based Activity Contribute to Develop AI Conception? A Case Study with Italian Fifth and Sixth Grade Classes

**Authors**: *Matteo Baldoni, Cristina Baroglio, Monica Bucciarelli, Sara Capecchi, Elena Gandolfi, Cristina Gena, Francesco Ianì, Elisa Marengo, Roberto Micalizio, Amon Rapp, Ivan Nabil Ras*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30350](https://doi.org/10.1609/aaai.v38i21.30350)

**Abstract**:

Artificial Intelligence is undoubtedly becoming pervasive in everyday life of everyone.
In this setting, developing correct AI conception since childhood is not only a need to 
be addressed in educational curricula, but is also a children right.

Accordingly, several initiatives at national and international levels aim at promoting AI
and emerging technology literacy, supported also by a proliferation in the literature 
of learning courses covering a variety of topics, learning objectives and targeted ages.
Schools are therefore pushed to introduce innovative activities for children in their
curricula.

In this paper, we report the results of a case study where we tested the contribution 
of an AI block-based course in developing computational thinking, and human 
and AI minds understanding in fifth and sixth grade children.

----

## [2627] A Framework for Approaching AI Education in Educator Preparation Programs

**Authors**: *Nancye Blair Black, Stacy George, Amy Eguchi, J. Camille Dempsey, Elizabeth Langran, Lucretia Fraga, Stein Brunvand, Nicol Howard*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30351](https://doi.org/10.1609/aaai.v38i21.30351)

**Abstract**:

In recent years, the rapid advancement of artificial intelligence (AI) has fostered an urgent need to better prepare current and future educators to be able to integrate AI technologies in their teaching and to teach AI literacy to PreK-12 students. While many organizations have developed professional learning opportunities for inservice educators, a gap remains for resources specifically designed for those facilitating and enrolled in Educator Preparation Programs (EPPs). In response to this gap, the International Society for Technology in Education (ISTE) launched its first AI Explorations for EPPs Faculty Fellowship. As a result of the Faculty Fellows’ collaboration, this paper articulates a framework of seven critical strategies with the potential to address the urgent need EPPs have in preparing preservice teachers to effectively integrate AI-powered instructional tools and to teach this new area of content knowledge in PreK-12 classrooms. In addition, we provide a review of literature and an overview of the emerging needs for integrating AI education in EPPs. We demonstrate why support for preservice teachers’ critical examination and application of AI, including a focus on the issues of equity, ethics, and culturally responsive teaching, is essential to their later success in PreK-12 classrooms. Recommendations for further research and learning are also provided to promote community-wide initiatives for supporting the integration of AI in education through Educator Preparation Programs and beyond.

----

## [2628] Artificial Intelligence in the CS2023 Undergraduate Computer Science Curriculum: Rationale and Challenges

**Authors**: *Eric Eaton, Susan L. Epstein*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30352](https://doi.org/10.1609/aaai.v38i21.30352)

**Abstract**:

Roughly every decade, the ACM and IEEE professional organizations have produced recommendations for the education of undergraduate computer science students. These guidelines are used worldwide by research universities, liberal arts colleges, and community colleges. For the latest 2023 revision of the curriculum, AAAI has collaborated with ACM and IEEE to integrate artificial intelligence more broadly into this new curriculum and to address the issues it raises for students, instructors, practitioners, policy makers, and the general public. This paper describes the development process and rationale that underlie the artificial intelligence components of the CS2023 curriculum, discusses the challenges in curriculum design for such a rapidly advancing field, and examines lessons learned during this three-year process.

----

## [2629] How Teachers Can Use Large Language Models and Bloom's Taxonomy to Create Educational Quizzes

**Authors**: *Sabina Elkins, Ekaterina Kochmar, Jackie C. K. Cheung, Iulian Serban*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30353](https://doi.org/10.1609/aaai.v38i21.30353)

**Abstract**:

Question generation (QG) is a natural language processing task with an abundance of potential benefits and use cases in the educational domain. In order for this potential to be realized, QG systems must be designed and validated with pedagogical needs in mind. However, little research has assessed or designed QG approaches with the input of real teachers or students. This paper applies a large language model-based QG approach where questions are generated with learning goals derived from Bloom's taxonomy. The automatically generated questions are used in multiple experiments designed to assess how teachers use them in practice. The results demonstrate that teachers prefer to write quizzes with automatically generated questions, and that such quizzes have no loss in quality compared to handwritten versions. Further, several metrics indicate that automatically generated questions can even improve the quality of the quizzes created, showing the promise for large scale use of QG in the classroom setting.

----

## [2630] Supporting Upper Elementary Students in Learning AI Concepts with Story-Driven Game-Based Learning

**Authors**: *Anisha Gupta, Seung Y. Lee, Bradford W. Mott, Srijita Chakraburty, Krista D. Glazewski, Anne T. Ottenbreit-Leftwich, J. Adam Scribner, Cindy E. Hmelo-Silver, James C. Lester*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30354](https://doi.org/10.1609/aaai.v38i21.30354)

**Abstract**:

Artificial intelligence (AI) is quickly finding broad application in every sector of society. This rapid expansion of AI has increased the need to cultivate an AI-literate workforce, and it calls for introducing AI education into K-12 classrooms to foster students’ awareness and interest in AI. With rich narratives and opportunities for situated problem solving, story-driven game-based learning offers a promising approach for creating engaging and effective K-12 AI learning experiences. In this paper, we present our ongoing work to iteratively design, develop, and evaluate a story-driven game-based learning environment focused on AI education for upper elementary students (ages 8 to 11). The game features a science inquiry problem centering on an endangered species and incorporates a Use-Modify-Create scaffolding framework to promote student learning. We present findings from an analysis of data collected from 16 students playing the game's quest focused on AI planning. Results suggest that the scaffolding framework provided students with the knowledge they needed to advance through the quest and that overall, students experienced positive learning outcomes.

----

## [2631] ImageSTEAM: Teacher Professional Development for Integrating Visual Computing into Middle School Lessons

**Authors**: *Suren Jayasuriya, Kimberlee Swisher, Joshua D. Rego, Sreenithy Chandran, John Mativo, Terri Kurz, Cerenity E. Collins, Dawn T. Robinson, Ramana Pidaparti*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30355](https://doi.org/10.1609/aaai.v38i21.30355)

**Abstract**:

Artificial intelligence (AI) and its teaching in the K-12 grades has been championed as a vital need for the United States due to the technology's future prominence in the 21st century. However, there remain several barriers to effective AI lessons at these age groups including the broad range of interdisciplinary knowledge needed and the lack of formal training or preparation for teachers to implement these lessons. In this experience report, we present ImageSTEAM, a teacher professional development for creating lessons surrounding computer vision, machine learning, and computational photography/cameras targeted for middle school grades 6-8 classes. Teacher professional development workshops were conducted in the states of Arizona and Georgia from 2021-2023 where lessons were co-created with teachers to introduce various specific visual computing concepts while aligning to state and national standards. In addition, the use of a variety of computer vision and image processing software including custom designed Python notebooks were created as technology activities and demonstrations to be used in the classroom. Educational research showed that teachers improved their self-efficacy and outcomes for concepts in computer vision, machine learning, and artificial intelligence when participating in the program. Results from the professional development workshops highlight key opportunities and challenges in integrating this content into the standard curriculum, the benefits of a co-creation pedagogy, and the positive impact on teacher and student's learning experiences. The open-source program curriculum is available at www.imagesteam.org.

----

## [2632] Practical Sentiment Analysis for Education: The Power of Student Crowdsourcing

**Authors**: *Robert Kasumba, Marion Neumman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30356](https://doi.org/10.1609/aaai.v38i21.30356)

**Abstract**:

Sentiment analysis provides a promising tool to automatically assess the emotions voiced in written student feedback such as periodically collected unit-of-study reflections. The commonly used dictionary-based approaches are limited to major languages and fail to capture contextual differences.  Pretrained large language models have been shown to be biased and online versions raise privacy concerns. Hence, we resort to traditional supervised machine learning (ML) approaches which are designed to overcome these issues by learning from domain-specific labeled data. However, these labels are hard to come by -- in our case manually annotating student feedback is prone to bias and time-consuming, especially in high-enrollment courses. In this work, we investigate the use of student crowdsourced labels for supervised sentiment analysis for education. Specifically, we compare crowdsourced and student self-reported labels with human expert annotations and use them in various ML approaches to evaluate the performance on predicting emotions of written student feedback collected from large computer science classes. We find that the random forest model trained with student-crowdsourced labels tremendously improves the identification of reflections with negative sentiment.  In addition to our quantitative study, we describe our crowdsourcing experiment which was intentionally designed to be an educational activity in an introduction to data science course.

----

## [2633] Addressing Digital and AI Skills Gaps in European Living Areas: A Comparative Analysis of Small and Large Communities

**Authors**: *Long Pham, Barry O'Sullivan, Teresa Scantamburlo, Tai Tan Mai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30357](https://doi.org/10.1609/aaai.v38i21.30357)

**Abstract**:

As Artificial Intelligence (AI) continues to permeate various aspects of societies, understanding the disparities in AI knowledge and skills across different living areas becomes imperative. Small living areas have emerged as significant contributors to Europe's economy, offering an alternative to the bustling environment of larger cities for those seeking an improved quality of life. Nonetheless, they often encounter challenges related to digital infrastructure, access to financial resources, and digital skills gaps, limiting their economic and social growth prospects. This study investigates the digital and AI skills gaps in the context of small and large European living areas, shedding light on the potential hindrances to unleashing the full economic and social potentials of these regions in an AI-enabled economy. Drawing from a comprehensive dataset encompassing 4,006 respondents across eight EU countries, this research examines the current perceptions and understandings of AI and digital skills within two distinct population groups: residents of smaller living areas and their counterparts in larger communities. Through bivariate analysis, notable insights are revealed concerning trust in AI solutions and entities, self-assessed digital skills, AI Awareness, AI Attitudes and demography variables in both population groups. These insights may refer to the significance of addressing digital and AI skills gaps in fostering growth and preparedness for the AI-driven future. As AI becomes increasingly integral to various aspects of society, targeted interventions and policies are essential to bridge these gaps and enable individuals and communities to harness the transformative potential of AI-enabled economies.

----

## [2634] A Toolbox for Modelling Engagement with Educational Videos

**Authors**: *Yuxiang Qiu, Karim Djemili, Denis Elezi, Aaneel Shalman Srazali, María Pérez-Ortiz, Emine Yilmaz, John Shawe-Taylor, Sahan Bulathwela*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30358](https://doi.org/10.1609/aaai.v38i21.30358)

**Abstract**:

With the advancement and utility of Artificial Intelligence (AI), personalising education to a global population could be a cornerstone of new educational systems in the future. This work presents the PEEKC dataset and the TrueLearn Python library, which contains a dataset and a series of online learner state models that are essential to facilitate research on learner engagement modelling. TrueLearn family of models was designed following the "open learner" concept, using humanly-intuitive user representations. This family of scalable, online models also help end-users visualise the learner models, which may in the future facilitate user interaction with their models/recommenders. The extensive documentation and coding examples make the library highly accessible to both machine learning developers and educational data mining and learning analytics practitioners. The experiments show the utility of both the dataset and the library with predictive performance significantly exceeding comparative baseline models. The dataset contains a large amount of AI-related educational videos, which are of interest for building and validating AI-specific educational recommenders.

----

## [2635] Build Your Own Robot Friend: An Open-Source Learning Module for Accessible and Engaging AI Education

**Authors**: *Zhonghao Shi, Amy O'Connell, Zongjian Li, Siqi Liu, Jennifer Ayissi, Guy Hoffman, Mohammad Soleymani, Maja J. Mataric*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30359](https://doi.org/10.1609/aaai.v38i21.30359)

**Abstract**:

As artificial intelligence (AI) is playing an increasingly important role in our society and global economy, AI education and literacy have become necessary components in college and K-12 education to prepare students for an AI-powered society. However, current AI curricula have not yet been made accessible and engaging enough for students and schools from all socio-economic backgrounds with different educational goals. In this work, we developed an open-source learning module for college and high school students, which allows students to build their own robot companion from the ground up. This open platform can be used to provide hands-on experience and introductory knowledge about various aspects of AI, including robotics, machine learning (ML), software engineering, and mechanical engineering. Because of the social and personal nature of a socially assistive robot companion, this module also puts a special emphasis on human-centered AI, enabling students to develop a better understanding of human-AI interaction and AI ethics through hands-on learning activities. With open-source documentation, assembling manuals and affordable materials, students from different socio-economic backgrounds can personalize their learning experience based on their individual educational goals. To evaluate the student-perceived quality of our module, we conducted a usability testing workshop with 15 college students recruited from a minority-serving institution. Our results indicate that our AI module is effective, easy-to-follow, and engaging, and it increases student interest in studying AI/ML and robotics in the future. We hope that this work will contribute toward accessible and engaging AI education in human-AI interaction for college and high school students.

----

## [2636] Co-designing AI Education Curriculum with Cross-Disciplinary High School Teachers

**Authors**: *Benjamin Xie, Parth Sarin, Jacob Wolf, Raycelle C. C. Garcia, Victoria Delaney, Isabel Sieh, Anika Fuloria, Deepak Varuvel Dennison, Christine Bywater, Victor R. Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30360](https://doi.org/10.1609/aaai.v38i21.30360)

**Abstract**:

High school teachers from many disciplines have growing interests in teaching about artificial intelligence (AI). This cross-disciplinary interest reflects the prevalence of AI tools across society, such as Generative AI tools built upon Large Language Models (LLM). However, high school classes are unique and complex environments, led by teachers with limited time and resources with priorities that vary by class and the students they serve. Therefore, developing curricula about AI for classes that span many disciplines (e.g. history, art, math) must involve centering the expertise of cross-disciplinary teachers. In this study, we conducted five collaborative curricular co-design sessions with eight teachers who taught high school humanities and STEM classes. We sought to understand how teachers considered AI when it was taught in art, math, and social studies contexts, as well as opportunities and challenges they identified with incorporating AI tools into their instruction. We found that teachers considered technical skills and ethical debates around AI, opportunities for "dual exploration" between AI and disciplinary learning, and limitations of AI tools as supporting engagement and reflection but also potentially distracting. We interpreted our findings relative to co-designing adaptable AI curricula to support teaching about and with AI across high school disciplines.

----

## [2637] Detecting AI-Generated Code Assignments Using Perplexity of Large Language Models

**Authors**: *Zhenyu Xu, Victor S. Sheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30361](https://doi.org/10.1609/aaai.v38i21.30361)

**Abstract**:

Large language models like ChatGPT can generate human-like code, posing challenges for programming education as students may be tempted to misuse them on assignments. However, there are currently no robust detectors designed specifically to identify AI-generated code. This is an issue that needs to be addressed to maintain academic integrity while allowing proper utilization of language models. Previous work has explored different approaches to detect AI-generated text, including watermarks, feature analysis, and fine-tuning language models. In this paper, we address the challenge of determining whether a student's code assignment was generated by a language model. First, our proposed method identifies AI-generated code by leveraging targeted masking perturbation paired with comperhesive scoring. Rather than applying a random mask, areas of the code with higher perplexity are more intensely masked. Second, we utilize a fine-tuned CodeBERT to fill in the masked portions, producing subtle modified samples. Then, we integrate the overall perplexity, variation of code line perplexity, and burstiness into a unified score. In this scoring scheme, a higher rank for the original code suggests it's more likely to be AI-generated. This approach stems from the observation that AI-generated codes typically have lower perplexity. Therefore, perturbations often exert minimal influence on them. Conversely, sections of human-composed codes that the model struggles to understand can see their perplexity reduced by such perturbations. Our method outperforms current open-source and commercial text detectors. Specifically, it improves detection of code submissions generated by OpenAI's text-davinci-003, raising average AUC from 0.56 (GPTZero baseline) to 0.87 for our detector.

----

## [2638] CyberQ: Generating Questions and Answers for Cybersecurity Education Using Knowledge Graph-Augmented LLMs

**Authors**: *Garima Agrawal, Kuntal Pal, Yuli Deng, Huan Liu, Ying-Chih Chen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30362](https://doi.org/10.1609/aaai.v38i21.30362)

**Abstract**:

Building a skilled cybersecurity workforce is paramount to building a safer digital world. However, the diverse skill set, constantly emerging vulnerabilities, and deployment of new cyber threats make learning cybersecurity challenging. Traditional education methods struggle to cope with cybersecurity's rapidly evolving landscape and keep students engaged and motivated. Different studies on students' behaviors show that an interactive mode of education by engaging through a question-answering system or dialoguing is one of the most effective learning methodologies. There is a strong need to create advanced AI-enabled education tools to promote interactive learning in cybersecurity. Unfortunately, there are no publicly available standard question-answer datasets to build such systems for students and novice learners to learn cybersecurity concepts, tools, and techniques. The education course material and online question banks are unstructured and need to be validated and updated by domain experts, which is tedious when done manually. In this paper, we propose CyberGen, a novel unification of large language models (LLMs) and knowledge graphs (KG) to generate the questions and answers for cybersecurity automatically. Augmenting the structured knowledge from knowledge graphs in prompts improves factual reasoning and reduces hallucinations in LLMs. We used the knowledge triples from cybersecurity knowledge graphs (AISecKG) to design prompts for ChatGPT and generate questions and answers using different prompting techniques. Our question-answer dataset, CyberQ, contains around 4k pairs of questions and answers. The domain expert manually evaluated the random samples for consistency and correctness. We train the generative model using the CyberQ dataset for question answering task.

----

## [2639] Automatic Short Answer Grading for Finnish with ChatGPT

**Authors**: *Li-Hsin Chang, Filip Ginter*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30363](https://doi.org/10.1609/aaai.v38i21.30363)

**Abstract**:

Automatic short answer grading (ASAG) seeks to mitigate the burden on teachers by leveraging computational methods to evaluate student-constructed text responses. Large language models (LLMs) have recently gained prominence across diverse applications, with educational contexts being no exception. The sudden rise of ChatGPT has raised expectations that LLMs can handle numerous tasks, including ASAG. This paper aims to shed some light on this expectation by evaluating two LLM-based chatbots, namely ChatGPT built on GPT-3.5 and GPT-4, on scoring short-question answers under zero-shot and one-shot settings. Our data consists of 2000 student answers in Finnish from ten undergraduate courses. Multiple perspectives are taken into account during this assessment, encompassing those of grading system developers, teachers, and students. On our dataset, GPT-4 achieves a good QWK score (0.6+) in 44% of one-shot settings, clearly outperforming GPT-3.5 at 21%. We observe a negative association between student answer length and model performance, as well as a correlation between a smaller standard deviation among a set of predictions and lower performance. We conclude that while GPT-4 exhibits signs of being a capable grader, additional research is essential before considering its deployment as a reliable autograder.

----

## [2640] A Chain-of-Thought Prompting Approach with LLMs for Evaluating Students' Formative Assessment Responses in Science

**Authors**: *Clayton Cohn, Nicole Hutchins, Tuan Le, Gautam Biswas*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30364](https://doi.org/10.1609/aaai.v38i21.30364)

**Abstract**:

This paper explores the use of large language models (LLMs) to score and explain short-answer assessments in K-12 science. While existing methods can score more structured math and computer science assessments, they often do not provide explanations for the scores. Our study focuses on employing GPT-4 for automated assessment in middle school Earth Science, combining few-shot and active learning with chain-of-thought reasoning. Using a human-in-the-loop approach, we successfully score and provide meaningful explanations for formative assessment responses. A systematic analysis of our method's pros and cons sheds light on the potential for human-in-the-loop techniques to enhance automated grading for open-ended science assessments.

----

## [2641] Online Reinforcement Learning-Based Pedagogical Planning for Narrative-Centered Learning Environments

**Authors**: *Fahmid Morshed Fahid, Jonathan P. Rowe, Yeojin Kim, Shashank Srivastava, James C. Lester*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30365](https://doi.org/10.1609/aaai.v38i21.30365)

**Abstract**:

Pedagogical planners can provide adaptive support to students in narrative-centered learning environments by dynamically scaffolding student learning and tailoring problem scenarios. Reinforcement learning (RL) is frequently used for pedagogical planning in narrative-centered learning environments. However, RL-based pedagogical planning raises significant challenges due to the scarcity of data for training RL policies. Most prior work has relied on limited-size datasets and offline RL techniques for policy learning. Unfortunately, offline RL techniques do not support on-demand exploration and evaluation, which can adversely impact the quality of induced policies. To address the limitation of data scarcity and offline RL, we propose INSIGHT, an online RL framework for training data-driven pedagogical policies that optimize student learning in narrative-centered learning environments. The INSIGHT framework consists of three components: a narrative-centered learning environment simulator, a simulated student agent, and an RL-based pedagogical planner agent, which uses a reward metric that is associated with effective student learning processes. The framework enables the generation of synthetic data for on-demand exploration and evaluation of RL-based pedagogical planning. We have implemented INSIGHT with OpenAI Gym for a narrative-centered learning environment testbed with rule-based simulated student agents and a deep Q-learning-based pedagogical planner. Our results show that online deep RL algorithms can induce near-optimal pedagogical policies in the INSIGHT framework, while offline deep RL algorithms only find suboptimal policies even with large amounts of data.

----

## [2642] Towards Building a Language-Independent Speech Scoring Assessment

**Authors**: *Shreyansh Gupta, Abhishek Unnam, Kuldeep Yadav, Varun Aggarwal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30366](https://doi.org/10.1609/aaai.v38i21.30366)

**Abstract**:

Automatic speech scoring is crucial in language learning, providing targeted feedback to language learners by assessing pronunciation, fluency, and other speech qualities. However, the scarcity of human-labeled data for languages beyond English poses a significant challenge in developing such systems. In this work, we propose a Language-Independent scoring approach to evaluate speech without relying on labeled data in the target language. We introduce a multilingual speech scoring system that leverages representations from the wav2vec 2.0 XLSR model and a force-alignment technique based on CTC-Segmentation to construct speech features. These features are used to train a machine learning model to predict pronunciation and fluency scores. We demonstrate the potential of our method by predicting expert ratings on a speech dataset spanning five languages - English, French, Spanish, German and Portuguese, and comparing its performance against Language-Specific models trained individually on each language, as well as a jointly-trained model on all languages. Results indicate that our approach shows promise as an initial step towards a universal language independent speech scoring.

----

## [2643] MineObserver 20: A Deep Learning & In-Game Framework for Assessing Natural Language Descriptions of Minecraft Imagery

**Authors**: *Jay Mahajan, Samuel Hum, Jack Henhapl, Diya Yunus, Matthew Gadbury, Emi Brown, Jeffrey Ginger, H. Chad Lane*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30367](https://doi.org/10.1609/aaai.v38i21.30367)

**Abstract**:

MineObserver 2.0 is an AI framework that uses Computer Vision and Natural Language Processing for assessing the accuracy of learner-generated descriptions of Minecraft images that include some scientifically relevant content. The system automatically assesses the accuracy of participant observations, written in natural language, made during science learning activities that take place in Minecraft. We demonstrate our system working in real-time and describe a teacher dashboard to showcase observations, both of which advance our previous work. We present the results of a study showing that MineObserver 2.0 improves over its predecessor both in perceived accuracy of the system's generated descriptions as well as in usefulness of the system's feedback. In future work, we intend improve system generated descriptions to give more teacher control and shift the system to perform continuous learning to more rapidly respond to novel observations made by learners.

----

## [2644] RetLLM-E: Retrieval-Prompt Strategy for Question-Answering on Student Discussion Forums

**Authors**: *Chancharik Mitra, Mihran Miroyan, Rishi Jain, Vedant Kumud, Gireeja Ranade, Narges Norouzi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30368](https://doi.org/10.1609/aaai.v38i21.30368)

**Abstract**:

This paper focuses on using Large Language Models to support teaching assistants in answering questions on large student forums such as Piazza and EdSTEM. Since student questions on these forums are often closely tied to specific aspects of the institution, instructor, and course delivery, general-purpose LLMs do not directly do well on this task.
We introduce RetLLM-E, a method that combines text-retrieval and prompting approaches to enable LLMs to provide precise and high-quality answers to student questions. When presented with a student question, our system initiates a two-step process. First, it retrieves relevant context from (i) a dataset of student questions addressed by course instructors
(Q&A Retrieval) and (ii) relevant segments of course materials (Document Retrieval). RetLLM-E then prompts LLM using the retrieved text and an engineered prompt structure to
yield an answer optimized for the student question.
We present a set of quantitative and human evaluation experiments, comparing our method to ground truth answers to questions in a test set of actual student questions. Our results demonstrate that our approach provides higher-quality responses to course-related questions than an LLM operating without context or relying solely on retrieval-based context. RetLLM-E can easily be adopted in different courses, providing instructors and students with context-aware automatic responses.

----

## [2645] Mimicking the Maestro: Exploring the Efficacy of a Virtual AI Teacher in Fine Motor Skill Acquisition

**Authors**: *Hadar Mulian, Segev Shlomov, Lior Limonad, Alessia Noccaro, Silvia Buscaglione*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30369](https://doi.org/10.1609/aaai.v38i21.30369)

**Abstract**:

Motor skills, especially fine motor skills like handwriting, play an essential role in academic pursuits and everyday life. Traditional methods to teach these skills, although effective, can be time-consuming and inconsistent. With the rise of advanced technologies like robotics and artificial intelligence, there is increasing interest in automating such teaching processes. In this study, we examine the potential of a virtual AI teacher in emulating the techniques of human educators for motor skill acquisition. We introduce an AI teacher model that captures the distinct characteristics of human instructors. Using a reinforcement learning environment tailored to mimic teacher-learner interactions, we tested our AI model against four guiding hypotheses, emphasizing improved learner performance, enhanced rate of skill acquisition, and reduced variability in learning outcomes. Our findings, validated on synthetic learners, revealed significant improvements across all tested hypotheses. Notably, our model showcased robustness across different learners and settings and demonstrated adaptability to handwriting. This research underscores the potential of integrating Imitation and Reinforcement Learning models with robotics in revolutionizing the teaching of critical motor skills.

----

## [2646] Enhancing Student Performance Prediction on Learnersourced Questions with SGNN-LLM Synergy

**Authors**: *Lin Ni, Sijie Wang, Zeyu Zhang, Xiaoxuan Li, Xianda Zheng, Paul Denny, Jiamou Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30370](https://doi.org/10.1609/aaai.v38i21.30370)

**Abstract**:

Learnersourcing offers great potential for scalable education through student content creation. However, predicting student performance on learnersourced questions, which is essential for personalizing the learning experience, is challenging due to the inherent noise in student-generated data. Moreover, while conventional graph-based methods can capture the complex network of student and question interactions, they often fall short under cold start conditions where limited student engagement with questions yields sparse data.  To address both challenges, we introduce an innovative strategy that synergizes the potential of integrating Signed Graph Neural Networks (SGNNs) and Large Language Model (LLM) embeddings. Our methodology employs a signed bipartite graph to comprehensively model student answers, complemented by a contrastive learning framework that enhances noise resilience. Furthermore, LLM's contribution lies in generating foundational question embeddings, proving especially advantageous in addressing cold start scenarios characterized by limited graph data. 
Validation across five real-world datasets sourced from the PeerWise platform underscores our approach's effectiveness. Our method outperforms baselines, showcasing enhanced predictive accuracy and robustness.

----

## [2647] From Raw Video to Pedagogical Insights: A Unified Framework for Student Behavior Analysis

**Authors**: *Zefang Yu, Mingye Xie, Jingsheng Gao, Ting Liu, Yuzhuo Fu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30371](https://doi.org/10.1609/aaai.v38i21.30371)

**Abstract**:

Understanding student behavior in educational settings is critical in improving both the quality of pedagogy and the level of student engagement. While various AI-based models exist for classroom analysis, they tend to specialize in limited tasks and lack generalizability across diverse educational environments. Additionally, these models often fall short in ensuring student privacy and in providing actionable insights accessible to educators. To bridge this gap, we introduce a unified, end-to-end framework by leveraging temporal action detection techniques and advanced large language models for a more nuanced student behavior analysis. Our proposed framework provides an end-to-end pipeline that starts with raw classroom video footage and culminates in the autonomous generation of pedagogical reports. It offers a comprehensive and scalable solution for student behavior analysis. Experimental validation confirms the capability of our framework to accurately identify student behaviors and to produce pedagogically meaningful insights, thereby setting the stage for future AI-assisted educational assessments.

----

## [2648] Students' Perceptions and Preferences of Generative Artificial Intelligence Feedback for Programming

**Authors**: *Zhengdong Zhang, Zihan Dong, Yang Shi, Thomas W. Price, Noboru Matsuda, Dongkuan Xu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30372](https://doi.org/10.1609/aaai.v38i21.30372)

**Abstract**:

The rapid evolution of artificial intelligence (AI), specifically large language models (LLMs), has opened opportunities for various educational applications. This paper explored the feasibility of utilizing ChatGPT, one of the most popular LLMs, for automating feedback for Java programming assignments in an introductory computer science (CS1) class. Specifically, this study focused on three questions: 1) To what extent do students view LLM-generated feedback as formative? 2) How do students see the comparative affordances of feedback prompts that include their code, vs. those that exclude it? 3) What enhancements do students suggest for improving LLM-generated feedback? To address these questions, we generated automated feedback using the ChatGPT API for four lab assignments in a CS1 class. The survey results revealed that students perceived the feedback as aligning well with formative feedback guidelines established by Shute. Additionally, students showed a clear preference for feedback generated by including the students' code as part of the LLM prompt, and our thematic study indicated that the preference was mainly attributed to the specificity, clarity, and corrective nature of the feedback. Moreover, this study found that students generally expected specific and corrective feedback with sufficient code examples, but had diverged opinions on the tone of the feedback. This study demonstrated that ChatGPT could generate Java programming assignment feedback that students perceived as formative. It also offered insights into the specific improvements that would make the ChatGPT-generated feedback useful for students.

----

## [2649] A Picture Is Worth a Thousand Words: Co-designing Text-to-Image Generation Learning Materials for K-12 with Educators

**Authors**: *Safinah Ali, Prerna Ravi, Katherine S. Moore, Hal Abelson, Cynthia Breazeal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30373](https://doi.org/10.1609/aaai.v38i21.30373)

**Abstract**:

Text-to-image generation (TTIG) technologies are Artificial Intelligence (AI) algorithms that use natural language algorithms in combination with visual generative algorithms. TTIG tools have gained popularity in recent months, garnering interest from non-AI experts, including educators and K-12 students. While they have exciting creative potential when used by K-12 learners and educators for creative learning, they are also accompanied by serious ethical implications, such as data privacy, spreading misinformation, and algorithmic bias. Given the potential learning applications, social implications, and ethical concerns, we designed 6-hour learning materials to teach K-12 teachers from diverse subject expertise about the technical implementation, classroom applications, and ethical implications of TTIG algorithms. We piloted the learning materials titled “Demystify text-to-image generative tools for K-12 educators" with 30 teachers across two workshops with the goal of preparing them to teach about and use TTIG tools in their classrooms. We found that teachers demonstrated a technical, applied and ethical understanding of TTIG algorithms and successfully designed prototypes of teaching materials for their classrooms.

----

## [2650] Constructing Dreams Using Generative AI

**Authors**: *Safinah Ali, Prerna Ravi, Randi Williams, Daniella DiPaola, Cynthia Breazeal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30374](https://doi.org/10.1609/aaai.v38i21.30374)

**Abstract**:

Generative AI tools introduce new and accessible forms of media creation for youth. They also raise ethical concerns about the generation of fake media, data protection, privacy and ownership of AI-generated art. Since generative AI is already being used in products used by youth, it is critical that they understand how these tools work and how they can be used or misused. In this work, we facilitated students’ generative AI learning through expression of their imagined future identities. We designed a learning workshop - Dreaming with AI - where students learned about the inner workings of generative AI tools, used text-to-image generation algorithms to create their imaged future dreams, reflected on the potential benefits and harms of generative AI tools and voiced their opinions about policies for the use of these tools in classrooms. In this paper, we present the learning activities and experiences of 34 high school students who engaged in our workshops. Students reached creative learning objectives by using prompt engineering to create their future dreams, gained technical knowledge by learning the abilities, limitations, text-visual mappings and applications of generative AI, and identified most potential societal benefits and harms of generative AI.

----

## [2651] Foundations of Autonomous Vehicles: A Curriculum Model for Developing Competencies in Artificial Intelligence and the Internet of Things for Grades 7-10

**Authors**: *Elham Buxton, Elahe Javadi, Matthew Hagaman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30375](https://doi.org/10.1609/aaai.v38i21.30375)

**Abstract**:

A few states (e.g., Maryland, Georgia, and Florida) have initiated efforts to incorporate artificial intelligence outcomes in K-12 education but others are still relying on informal spaces for learning and literacy in this area. In this manuscript, we share the curriculum and content of an informal effort focused on students in grades 7-10. We combined artificial intelligence competencies with Internet of Things skills to enable meaningful learning covering all Five Big Ideas in AI. In our one-week summer camp, students experimented with perceptions by working with vision, infrared, and ultrasonic sensors. They learned about representation through work with neural network playgrounds. Students engaged in supervised learning of an image processing model and used the model to control the actions of a robot car. Natural interactions and societal impacts were assessed as students observed the robot car's behavior. 
Results demonstrate that our curriculum was successful in achieving its objectives. Excluding the robot car kit, the curriculum was created using free platforms and tools. This program could be replicated in informal settings by any educator or collaborator with a computer science background. This paper describes our summer camp curriculum, its components and their implementation, the lessons learned, and potential future enhancements.

----

## [2652] Unplugged K-12 AI Learning: Exploring Representation and Reasoning with a Facial Recognition Game

**Authors**: *Hansol Lim, Wookhee Min, Jessica Vandenberg, Veronica Cateté, Bradford W. Mott*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30376](https://doi.org/10.1609/aaai.v38i21.30376)

**Abstract**:

With the growing prevalence of AI, the need for K-12 AI education is becoming more crucial, which is prompting active research in developing engaging and age-appropriate AI learning activities. Efforts are underway, such as those by the AI4K12 initiative, to establish guidelines for organizing K- 12 AI education; however, effective instructional resources are needed by educators. In this paper, we describe our work to design, develop, and implement an unplugged activity centered on facial recognition technology for middle school students. Facial recognition is integrated into a wide range of applications throughout daily life, which makes it a familiar and engaging tool for students and an effective medium for conveying AI concepts. Our unplugged activity, “Guess Whose Face,” is designed as a board game that focuses on Representation and Reasoning from AI4K12’s 5 Big Ideas in AI. The game is crafted to enable students to develop AI competencies naturally through physical interaction. In the game, one student uses tracing paper to extract facial features from a familiar face shown on a card, such as a cartoon character or celebrity, and then other students try to guess the identity of the hidden face. We discuss details of the game, its iterative refinement, and initial findings from piloting the activity during a summer camp for rural middle school students.

----

## [2653] AI, Ethics, and Education: The Pioneering Path of Sidekick Academy

**Authors**: *Elizabeth Radday, Matt Mervis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30377](https://doi.org/10.1609/aaai.v38i21.30377)

**Abstract**:

Generative artificial intelligence (AI) is swiftly cementing its role as an indispensable tool for students transitioning from K-12 to higher education and professional spheres. Yet, harnessing its full potential requires more than mere familiarity. Students must be equipped with the skills to engage with AI both productively and ethically. Left unchecked, AI usage can pose risks, especially if students lack proper guidance or understanding of their actions. Moreover, effective interaction with AI necessitates skills in prompt engineering to yield desired outcomes. Sidekick Academy is a digital online platform where students can safely experiment with and learn about AI. This article delves into the genesis of Sidekick Academy, offering a glimpse into its lessons on how to use AI and complex debate on ethical use. It also sheds light on the academy's "sandbox" - a secure space for students to explore AI without jeopardizing their safety or privacy.

----

## [2654] From Consumers to Critical Users: Prompty, an AI Literacy Tool for High School Students

**Authors**: *Deepak Varuvel Dennison, Raycelle C. C. Garcia, Parth Sarin, Jacob Wolf, Christine Bywater, Benjamin Xie, Victor R. Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30378](https://doi.org/10.1609/aaai.v38i21.30378)

**Abstract**:

In an age where Large Language Models (LLMs) expedite the generation of text, the skills for critically evaluating and creating meaningful text using these models are often lacking. To help classroom teachers address this, we introduce Prompty, a specialized teaching tool co-designed to facilitate both critical and effective use of LLMs. Prompty serves multiple learning goals: it allows students to critically evaluate text generated by LLMs, aids in their writing practice, and provides a deeper understanding of how LLMs function—all within a student-friendly environment secured by essential guardrails. Prompty was co-designed in collaboration with high school teachers as part of CRAFT, an initiative by Stanford University to promote AI literacy. It was pilot-tested in a high school English class to serve as an AI writing assistant, focusing on the critical evaluation of machine-generated text. This trial yielded preliminary evidence that attests to the tool's effectiveness in fulfilling its educational goals. The findings from the pilot study indicate that easy-to-use tools like Prompty have great potential. These tools can be adapted to fit the goals of individual teachers. They can help in achieving subject-specific learning goals while serving as an effective way to teach AI concepts in high school.

----

## [2655] Dr RO Bott Will See You Now: Exploring AI for Wellbeing with Middle School Students

**Authors**: *Randi Williams, Sharifa Alghowinem, Cynthia Breazeal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30379](https://doi.org/10.1609/aaai.v38i21.30379)

**Abstract**:

Artificial Intelligence (AI) is permeating almost every area of society, reshaping how many people, including youth, navigate the world. Despite the increased presence of AI, most people lack a baseline knowledge of how AI works. Moreover, social barriers often hinder equal access to AI courses, perpetuating disparities in participation in the field. To address this, it is crucial to design AI curricula that are effective, inclusive, and relevant, especially to learners from backgrounds that are historically excluded from working in tech. In this paper, we present AI for Wellbeing, a curriculum where students explore conversational AI and the ethical considerations around using it to promote wellbeing. We specifically designed content, educator materials, and educational technologies to meet the interests and needs of students and educators from diverse backgrounds. We piloted AI for Wellbeing in a 5-day virtual workshop with middle school teachers and students. Then, using a mixed-methods approach, we analyzed students' work and teachers' feedback. Our results suggest that the curriculum content and design effectively engaged students, enabling them to implement meaningful AI projects for wellbeing. We hope that the design of this curriculum and insights from our evaluation will inspire future efforts to create culturally relevant K-12 AI curricula.

----

## [2656] An Effectiveness Study of Teacher-Led AI Literacy Curriculum in K-12 Classrooms

**Authors**: *Helen Zhang, Irene Lee, Katherine S. Moore*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30380](https://doi.org/10.1609/aaai.v38i21.30380)

**Abstract**:

Artificial intelligence (AI) has rapidly pervaded and reshaped almost all walks of life, but efforts to promote AI literacy in K-12 schools remain limited. There is a knowledge gap in how to prepare teachers to teach AI literacy in inclusive classrooms and how teacher-led classroom implementations can impact students. This paper reports a comparison study to investigate the effectiveness of an AI literacy curriculum when taught by classroom teachers. The experimental group included 89 middle school students who learned an AI literacy curriculum during regular school hours. The comparison group consisted of 69 students who did not learn the curriculum. Both groups completed the same pre and post-test. The results show that students in the experimental group developed a deeper understanding of AI concepts and more positive attitudes toward AI and its impact on future careers after the curriculum than those in the comparison group. This shows that the teacher-led classroom implementation successfully equipped students with a conceptual understanding of AI. Students achieved significant gains in recognizing how AI is relevant to their lives and felt empowered to thrive in the age of AI. Overall this study confirms the potential of preparing K-12 classroom teachers to offer AI education in classrooms in order to reach learners of diverse backgrounds and broaden participation in AI literacy education among young learners.

----

## [2657] "Allot?" is "A Lot!" Towards Developing More Generalized Speech Recognition System for Accessible Communication

**Authors**: *Grisha Bandodkar, Shyam Agarwal, Athul Krishna Sughosh, Sahilbir Singh, Taeyeong Choi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30381](https://doi.org/10.1609/aaai.v38i21.30381)

**Abstract**:

The proliferation of Automatic Speech Recognition (ASR) systems has revolutionized translation and transcription. However, challenges persist in ensuring inclusive communication for non-native English speakers. This study quantifies the gap between accented and native English speech using Wav2Vec 2.0, a state-of-the-art transformer model. Notably, we found that accented speech exhibits significantly higher word error rates of 30-50%, in contrast to native speakers’ 2-8% (Baevski et al. 2020). Our exploration extends to leveraging accessible online datasets to highlight the potential of enhancing speech recognition by fine-tuning the Wav2Vec 2.0 model. Through experimentation and analysis, we highlight the challenges with training models on accented speech. By refining models and addressing data quality issues, our work presents a pipeline for future investigations aimed at developing an integrated system capable of effectively engaging with a broader range of individuals with diverse backgrounds. Accurate recognition of accented speech is a pivotal step toward democratizing AI-driven communication products.

----

## [2658] EnColor: Improving Visual Accessibility with a Deep Encoder-Decoder Image Corrector for Color Vision Deficient Individuals

**Authors**: *Satyam Goyal, Kavya Sasikumar, Rohan Sheth, Akash Seelam, Taeyeong Choi, Xin Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30382](https://doi.org/10.1609/aaai.v38i21.30382)

**Abstract**:

Individuals with color vision deficiencies (CVDs) often face significant challenges in accessing vital information for decision-making. In response, we introduce EnColor—a deep Encoder-decoder Color corrector for images, enabling individuals with CVDs to perceive the contents in originally intended colorization. Our network architecture is designed to effectively capture essential visual features for reconstructing standard images into color-corrected versions. In particular, our training pipeline is integrated with a CVD simulator so as to ensure the fidelity of output throughout the lens of individuals with impaired color vision. For evaluation, we focus primarily on tomato images, considering the profound impact of color vision deficiencies on practical domains like agri-food systems. Our quantitative results demonstrate that the EnColor model achieves over 16.8% improvement over previously introduced algorithms in terms of color retention, supporting our design choices. Furthermore, a survey with 43 participants provides subjective assessments with the highest scores on our method. Additionally, specific visual examples are presented to highlight accurately restored colors. We also publicly share all our codes of EnColor as well as the baseline methods to ensure reproducibility and facilitate more studies in CVD correction.

----

## [2659] GPT4MTS: Prompt-based Large Language Model for Multimodal Time-series Forecasting

**Authors**: *Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, Yan Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30383](https://doi.org/10.1609/aaai.v38i21.30383)

**Abstract**:

Time series forecasting is an essential area of machine learning with a wide range of real-world applications. Most of the previous forecasting models aim to capture dynamic characteristics from uni-modal numerical historical data. Although extra knowledge can boost the time series forecasting performance, it is hard to collect such information. In addition, how to fuse the multimodal information is non-trivial. In this paper, we first propose a general principle of collecting the corresponding textual information from different data sources with the help of modern large language models (LLM). Then, we propose a prompt-based LLM framework to utilize both the numerical data and the textual information simultaneously, named GPT4MTS. In practice, we propose a GDELT-based multimodal time series dataset for news impact forecasting, which provides a concise and well-structured version of time series dataset with textual information for further research in communication. Through extensive experiments, we demonstrate the effectiveness of our proposed method on forecasting tasks with extra-textual information.

----

## [2660] LERMO: A Novel Web Game for AI-Enhanced Sign Language Recognition

**Authors**: *Adilson Medronha, Luís Lima, Janaína Claudio, Lucas S. Kupssinskü, Rodrigo C. Barros*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30384](https://doi.org/10.1609/aaai.v38i21.30384)

**Abstract**:

Sign language is a visual and gestural communication system used by deaf and hearing-impaired people. Despite numerous deep learning methods proposed for automatic interpretation, a gap persists in developing applications that effectively utilize these models for assisting sign language studies and inclusion. We introduce LERMO (https://lermo.app/), a web game merging machine learning and gamification to enhance sign language fingerspelling. Inspired by Wordle™, LERMO offers an interactive word-guessing game where users can play using a video camera. We create a new dataset of labeled landmark fingerspelling and design our model to ensure optimal speed and efficiency to run on a web browser. We survey approximately 40 users, which find LERMO user-friendly and innovative. From those, 95% believe LERMO could be used to enhance fingerspelling skills.

----

## [2661] Revitalizing Bahnaric Language through Neural Machine Translation: Challenges, Strategies, and Promising Outcomes

**Authors**: *Hoang Nhat Khang Vo, Duc Dong Le, Tran Minh Dat Phan, Tan Sang Nguyen, Quoc Nguyen Pham, Ngoc Oanh Tran, Quang Duc Nguyen, Tran Minh Hieu Vo, Tho Quan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30385](https://doi.org/10.1609/aaai.v38i21.30385)

**Abstract**:

The Bahnar, a minority ethnic group in Vietnam with ancient roots, hold a language of deep cultural and historical significance. The government is prioritizing the preservation and dissemination of Bahnar language through online availability and cross-generational communication. Recent AI advances, including Neural Machine Translation (NMT), have transformed translation with improved accuracy and fluency, fostering language revitalization through learning, communication, and documentation. In particular, NMT enhances accessibility for Bahnar language speakers, making information and content more available.

However, translating Vietnamese to Bahnar language faces practical hurdles due to resource limitations, particularly in the case of Bahnar language as an extremely low-resource language. These challenges encompass data scarcity, vocabulary constraints, and a lack of fine-tuning data. To address these, we propose transfer learning from selected pre-trained models to optimize translation quality and computational efficiency, capitalizing on linguistic similarities between Vietnamese and Bahnar language. Concurrently, we apply tailored augmentation strategies to adapt machine translation for the Vietnamese-Bahnar language context. Our approach is validated through superior results on bilingual Vietnamese-Bahnar language datasets when compared to baseline models. By tackling translation challenges, we help revitalize Bahnar language, ensuring information flows freely and the language thrives.

----

## [2662] Model AI Assignments 2024

**Authors**: *Todd W. Neller, Pia Bideau, David Bierbach, Wolfgang Hönig, Nir Lipovetzky, Christian Muise, Lino Coria, Claire Wong, Stephanie Rosenthal, Yu Lu, Ming Gao, Jingjing Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30386](https://doi.org/10.1609/aaai.v38i21.30386)

**Abstract**:

The Model AI Assignments session seeks to gather and dis-
seminate the best assignment designs of the Artificial In-
telligence (AI) Education community. Recognizing that as-
signments form the core of student learning experience, we

here present abstracts of five AI assignments from the 2024
session that are easily adoptable, playfully engaging, and

flexible for a variety of instructor needs. Assignment spec-
ifications and supporting resources may be found at http://modelai.gettysburg.edu.

----

## [2663] Discovering Heterogeneous Causal Effects in Relational Data

**Authors**: *Shishir Adhikari*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30387](https://doi.org/10.1609/aaai.v38i21.30387)

**Abstract**:

Causal inference in relational data should account for the non-IID nature of the data and the interference phenomenon, which occurs when a unit's outcome is influenced by the treatments or outcomes of others. Existing solutions to causal inference under interference consider either homogeneous influence from peers or specific heterogeneous influence contexts (e.g., local neighborhood structure). This thesis investigates causal reasoning in relational data and the automated discovery of heterogeneous causal effects under arbitrary heterogeneous peer influence contexts and effect modification.

----

## [2664] Knowledge Distillation from Single-Task Teachers to Multi-Task Student for End-to-End Autonomous Driving

**Authors**: *Pedram Agand*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30388](https://doi.org/10.1609/aaai.v38i21.30388)

**Abstract**:

In the domain of end-to-end autonomous driving, conventional sensor fusion techniques exhibit inadequacies, particularly when facing challenging scenarios with numerous dynamic agents. Imitation learning hampers the performance by the expert and encounters issues with out-of-distribution challenges. To overcome these limitations, we propose a transformer-based algorithm designed to fuse diverse representations from RGB-D cameras through knowledge distillation. This approach leverages insights from multi-task teachers to enhance the learning capabilities of single-task students, particularly in a Reinforcement Learning (RL) setting. Our model consists of two primary modules: the perception module, responsible for encoding observation data acquired from RGB-D cameras and performing tasks such as semantic segmentation, semantic depth cloud mapping (SDC), ego vehicle speed estimation, and traffic light state recognition. Subsequently, the control module decodes these features, incorporating additional data, including a rough simulator for static and dynamic environments, to anticipate waypoints within a latent feature space. Vehicular controls (e.g., steering, throttle, and brake) are obtained directly from measurement features and environmental states using the RL agent and are further refined by a PID algorithm that dynamically follows waypoints. The model undergoes rigorous evaluation and comparative analysis on the CARLA simulator across various scenarios, encompassing normal to adversarial conditions. Our code is available at https://github.com/pagand/e2etransfuser/ to facilitate future studies.

----

## [2665] Tiered Coalition Formation Game Stability and Simulation

**Authors**: *Nathan Arnold*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30389](https://doi.org/10.1609/aaai.v38i21.30389)

**Abstract**:

Expanding on a 2017 paper by Siler that introduced tiered coalition formation games, I have introduced a variant game and examined the stabilizability of both the original game and its variant. My thesis will contain further theoretical stability findings and the results and interpretation of a simulation based upon real data from video game matchups.

----

## [2666] Semi-factual Explanations in AI

**Authors**: *Saugat Aryal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30390](https://doi.org/10.1609/aaai.v38i21.30390)

**Abstract**:

Most of the recent works on post-hoc example-based eXplainable AI (XAI) methods revolves around employing counterfactual explanations to provide justification of the predictions made by AI systems. Counterfactuals show what changes to the input-features change the output decision. However, a lesser-known, special-case of the counterfacual is the semi-factual, which provide explanations about what changes to the input-features do not change the output decision.  Semi-factuals are potentially as useful as counterfactuals but have received little attention in the XAI literature.  My doctoral research aims to establish a comprehensive framework for the use of semi-factuals in XAI by developing novel methods for their computation, supported by user tests.

----

## [2667] Domain Engineering to Represent Human Behavior Using Multi-Agent Planning and Inductive Methodologies

**Authors**: *Salena Torres Ashton*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30391](https://doi.org/10.1609/aaai.v38i21.30391)

**Abstract**:

This research combines multi agent planning, the psycholinguistics of question asking, procedural grounded theory, and hierarchical task networks to represent domains for automated planning.

----

## [2668] The Promise of Serverless Computing within Peer-to-Peer Architectures for Distributed ML Training

**Authors**: *Amine Barrak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30392](https://doi.org/10.1609/aaai.v38i21.30392)

**Abstract**:

My thesis focuses on the integration of serverless computing with Peer to Peer (P2P) architectures in distributed Machine Learning (ML). This research aims to harness the decentralized, resilient nature of P2P systems, combined with the scalability and automation of serverless platforms. We explore using databases not just for communication but also for in-database model updates and gradient averaging, addressing the challenges of statelessness in serverless environments.

----

## [2669] Identifying, Mitigating, and Anticipating Bias in Algorithmic Decisions

**Authors**: *Joachim Baumann*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30393](https://doi.org/10.1609/aaai.v38i21.30393)

**Abstract**:

Today's machine learning (ML) applications predominantly adhere to a standard paradigm: the decision maker designs the algorithm by optimizing a model for some objective function. While this has proven to be a powerful approach in many domains, it comes with inherent side effects: the power over the algorithmic outcomes lies solely in the hands of the algorithm designer, and alternative objectives, such as fairness, are often disregarded. This is particularly problematic if the algorithm is used to make consequential decisions that affect peoples lives. My research focuses on developing principled methods to characterize and address the mismatch between these different objectives.

----

## [2670] Deep Reinforcement Learning for Communication Networks

**Authors**: *Raffaele Galliera*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30394](https://doi.org/10.1609/aaai.v38i21.30394)

**Abstract**:

This research explores optimizing communication tasks with (Multi-Agent) Reinforcement Learning (RL/MARL) in Point-to-Point and Group Communication (GC) networks. The study initially applied RL for Congestion Control in networks with dynamic link properties, yielding competitive results. Then, it focused on the challenge of effective message dissemination in GC networks, by framing a novel game-theoretic formulation and designing methods to solve the task based on MARL and Graph Convolution. Future research will deepen the exploration of MARL in GC. This will contribute to both academic knowledge and practical advancements in the next generation of communication protocols.

----

## [2671] Towards Trustworthy Autonomous Systems via Conversations and Explanations

**Authors**: *Balint Gyevnar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30395](https://doi.org/10.1609/aaai.v38i21.30395)

**Abstract**:

Autonomous systems fulfil an increasingly important role in our societies, however, AI-powered systems have seen less success over the years, as they are expected to tackle a range of social, legal, or technological challenges and modern neural network-based AI systems cannot yet provide guarantees to many of these challenges. Particularly important is that these systems are black box decision makers, eroding human oversight, contestation, and agency. To address this particular concern, my thesis focuses on integrating social explainable AI with cognitive methods and natural language processing to shed light on the internal processes of autonomous systems in a way accessible to lay users. I propose a causal explanation generation model for decision-making called CEMA based on counterfactual simulations in multi-agent systems. I also plan to integrate CEMA with a broader natural language processing pipeline to support targeted and personalised explanations that address people's cognitive biases. I hope that my research will have a positive impact on the public acceptance of autonomous agents by building towards more trustworthy AI.

----

## [2672] Temporal Dependencies and Spatio-Temporal Patterns of Time Series Models

**Authors**: *Md. Khairul Islam*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30396](https://doi.org/10.1609/aaai.v38i21.30396)

**Abstract**:

The widespread use of Artificial Intelligence (AI) has highlighted the importance of understanding AI model behavior. This understanding is crucial for practical decision-making, assessing model reliability, and ensuring trustworthiness. Interpreting time series forecasting models faces unique challenges compared to image and text data. These challenges arise from the temporal dependencies between time steps and the evolving importance of input features over time. My thesis focuses on addressing these challenges by aiming for more precise explanations of feature interactions, uncovering spatiotemporal patterns, and demonstrating the practical applicability of these interpretability techniques using real-world datasets and state-of-the-art deep learning models.

----

## [2673] Risk Management in Image Generative Models through Model Fingerprinting

**Authors**: *Changhoon Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30397](https://doi.org/10.1609/aaai.v38i21.30397)

**Abstract**:

My doctoral research delves into the realm of generative model fingerprinting, aiming to assign responsibility for the generated images. I introduce frameworks that modify generative models to incorporate each user's distinct digital fingerprint. This ensures that every piece of generated content carries a traceable identifier linked to its originator. The primary objective of my research is to achieve optimal attribution accuracy while ensuring minimal compromise on the model's performance. Additionally, I present strategies designed to enhance robustness against common adversarial manipulations, which malicious users might employ to obscure or remove these fingerprints.

----

## [2674] The Inter-batch Diversity of Samples in Experience Replay for Continual Learning

**Authors**: *Andrii Krutsylo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30398](https://doi.org/10.1609/aaai.v38i21.30398)

**Abstract**:

In a Continual Learning setting, models are trained on data with occasional distribution shifts, resulting in forgetting the information learned before each shift. Experience Replay (ER) addresses this challenge by retaining part of the old training samples and replaying them alongside current data, improving the model's understanding of the overall distribution in training batches. The crucial factor in ER performance is the diversity of samples within batches. The impact of sample diversity across a sequence of batches is investigated, introducing a new metric and an associated approach to assess and leverage this diversity. This exploration opens up significant potential for future work, as various strategies can be devised to ensure inter-batch diversity. Achieving optimal results may involve striking a balance between this novel metric and other inherent properties of a batch or sequence.

----

## [2675] Making AI Policies Transparent to Humans through Demonstrations

**Authors**: *Michael S. Lee*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30399](https://doi.org/10.1609/aaai.v38i21.30399)

**Abstract**:

Demonstrations are a powerful way of increasing the transparency of AI policies to humans. Though we can approximately model human learning from demonstrations as inverse reinforcement learning, we note that human learning can differ from algorithmic learning in key ways, e.g. humans are computationally limited and may sometimes struggle to understand all of the nuances of a demonstration. Unlike related work that provide demonstrations to humans that simply maximize information gain, I leverage concepts from the human education literature, such as the zone of proximal development and scaffolding, to show demonstrations that balance informativeness and difficulty of understanding to maximize human learning.

----

## [2676] A Privacy Preserving Federated Learning (PPFL) Based Cognitive Digital Twin (CDT) Framework for Smart Cities

**Authors**: *Sukanya Mandal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30400](https://doi.org/10.1609/aaai.v38i21.30400)

**Abstract**:

A Smart City is one that makes better use of city data to make our communities better places to live. Typically, this has 3 components: sensing (data collection), analysis and actuation. Privacy, particularly as it relates to citizen's data, is a cross-cutting theme. A Digital Twin (DT) is a virtual replica of a real-world physical entity. Cognitive Digital Twins (CDT) are DTs enhanced with cognitive AI capabilities. Both DTs and CDTs have seen adoption in the manufacturing and industrial sectors however cities are slow to adopt these because of privacy concerns. This work attempts to address these concerns by proposing a Privacy Preserving Federated Learning (PPFL) based Cognitive Digital Twin framework for Smart Cities.

----

## [2677] Thesis Summary: Operationalizing User-Inclusive Transparency in Artificial Intelligence Systems

**Authors**: *Deepa Muralidhar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30401](https://doi.org/10.1609/aaai.v38i21.30401)

**Abstract**:

Artificial intelligence system architects can increase user trust by designing systems that are inherently transparent. We propose the idea of representing an AI system as an amalgamation of the AI Model (algorithms), data (input and output, including outcomes), and the user interface with visual interpretations (e.g. graphs, Venn diagrams). By designing human controls and feedback mechanisms for AI systems that allow users to exert control over them we can integrate transparency into existing user interfaces. Our plan is to design prototypes of transparent user interfaces for AI systems using well-known usability principles. By conducting surveys we will study their impact to see if these principles help the user to work with the AI system with confidence and if the user perceives the system to be adequately transparent.

----

## [2678] Learning Generalizable and Composable Abstractions for Transfer in Reinforcement Learning

**Authors**: *Rashmeet Kaur Nayyar*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30402](https://doi.org/10.1609/aaai.v38i21.30402)

**Abstract**:

Reinforcement Learning (RL) in complex environments presents many challenges: agents require learning concise representations of both environments and behaviors for efficient reasoning and generalizing experiences to new, unseen situations. However, RL approaches can be sample-inefficient and difficult to scale, especially in long-horizon sparse reward settings. To address these issues, the goal of my doctoral research is to develop methods that automatically construct semantically meaningful state and temporal abstractions for efficient transfer and generalization. In my work, I develop hierarchical approaches for learning transferable, generalizable knowledge in the form of symbolically represented options, as well as for integrating search techniques with RL to solve new problems by efficiently composing the learned options. Empirical results show that the resulting approaches effectively learn and transfer knowledge, achieving superior sample efficiency compared to SOTA methods while also enhancing interpretability.

----

## [2679] A Hybrid AI Framework for Sensor-Based Personal Health Monitoring towards Precision Health

**Authors**: *Mbithe Nzomo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30403](https://doi.org/10.1609/aaai.v38i21.30403)

**Abstract**:

Non-communicable diseases are on the rise globally, resulting in accelerated efforts to develop personal health monitoring systems for early detection, prediction, and prevention of diseases. This is part of the vision of precision health, an emerging paradigm that focuses on preventing disease before it strikes by encouraging people to actively monitor and work towards improving their health. A key facilitator of this is the use of wearable sensors that can collect and measure physiological data.Although many sensor-based health monitoring systems have been proposed, interoperability of health data and processes, prediction of future health states, and uncertainty management remain open challenges. This research aims to alleviate these challenges through the development of a reusable framework integrating both data-driven and knowledge-driven AI within a hybrid AI architecture.

----

## [2680] Navigating Uncertainty in Epidemic Contexts with Reinforcement Learning

**Authors**: *Elizabeth Akinyi Ondula*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30404](https://doi.org/10.1609/aaai.v38i21.30404)

**Abstract**:

My research integrates stochastic epidemic models with reinforcement learning to develop effective strategies or policies to inform operational decisions. The objective is to refine policies that are attuned to diverse outbreak dynamics and to offer a tool for informed planning in real-world settings.

----

## [2681] Target Focused Shallow Transformer Framework for Efficient Visual Tracking

**Authors**: *Md Maklachur Rahman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30405](https://doi.org/10.1609/aaai.v38i21.30405)

**Abstract**:

Template learning transformer trackers have achieved significant performance improvement recently due to the longdependency learning using the self-attention (SA) mechanism. However, the typical SA mechanisms in transformers adopt a less discriminative design approach which is inadequate for focusing on the most important target information during tracking. Therefore, existing trackers are easily distracted by background information and have constraints in handling tracking challenges. The focus of our research is to develop a target-focused discriminative shallow transformer tracking framework that can learn to distinguish the target from the background and enable accurate tracking with fast speed. Extensive experiments will be performed on several popular benchmarks, including OTB100, UAV123, GOT10k, LaSOT, and TrackingNet, to demonstrate the effectiveness of the proposed framework.

----

## [2682] Learning Pattern-Based Extractors from Natural Language and Knowledge Graphs: Applying Large Language Models to Wikipedia and Linked Open Data

**Authors**: *Célian Ringwald*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30406](https://doi.org/10.1609/aaai.v38i21.30406)

**Abstract**:

Seq-to-seq transformer models have recently been successfully used for relation extraction, showing their flexibility, effectiveness, and scalability on that task. In this context, knowledge graphs aligned with Wikipedia such as DBpedia and Wikidata give us the opportunity to leverage existing texts and corresponding RDF graphs in order to extract, from these texts, the knowledge that is missing in the corresponding graphs and meanwhile improve their coverage. The goal of my thesis is to learn efficient extractors targeting specific RDF patterns and to do so by leveraging the latest language models and the dual base formed by Wikipedia on the one hand, and DBpedia and Wikidata on the other hand.

----

## [2683] Learning from an Infant's Visual Experience

**Authors**: *Deepayan Sanyal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30407](https://doi.org/10.1609/aaai.v38i21.30407)

**Abstract**:

Infants see a selective view of the world: they see some objects with high frequency and from a wide range of viewpoints (e.g., their toys during playing) while a much larger set of objects are seen much more rarely and from limited viewpoints (e.g., objects they see outdoors). Extensive, repeated visual experiences with a small number of objects during infancy plays a big role in the development of human visual skills. Internet-style datasets that are commonly used in computer vision research do not contain the regularities that result from such repeated, structured experiences with a few objects. This has led to a dearth of models that learn by exploiting these regularities. In my PhD dissertation, I use deep learning models to investigate how regularities in an infant's visual experience can be leveraged for visual representation learning.

----

## [2684] AI-Assisted Human Teamwork

**Authors**: *Sangwon Seo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30408](https://doi.org/10.1609/aaai.v38i21.30408)

**Abstract**:

Effective teamwork translates to fewer preventable errors and higher task performance in collaborative tasks. However, in time-critical tasks, successful teamwork becomes highly challenging to attain. In such settings, often, team members have partial observability of their surroundings, incur high cost of communication, and have trouble estimating the state and intent of their teammates. To assist a team in improving teamwork at task time, my doctoral research proposes an automated task-time team intervention system. Grounded in the notion of shared mental models, the system first detects whether the team is on the same page or not. It then generates effective interventions to improve teamwork. Additionally, by leveraging past demonstrations to learn a model of team behavior, this system minimizes the need for domain experts to specify teamwork models and rules.

----

## [2685] Learning Neuro-Symbolic Abstractions for Robot Planning and Learning

**Authors**: *Naman Shah*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30409](https://doi.org/10.1609/aaai.v38i21.30409)

**Abstract**:

Although state-of-the-art hierarchical robot planning algorithms allow robots to efficiently compute long-horizon motion plans for achieving user desired tasks, these methods typically rely upon environment-dependent state and action abstractions that need to be hand-designed by experts. On the other hand, non-hierarchical robot planning approaches fail to compute solutions for complex tasks that require reasoning over a long horizon. My research addresses these problems by proposing an approach for learning abstractions and developing hierarchical planners that efficiently use learned abstractions to boost robot planning performance and provide strong guarantees of reliability.

----

## [2686] The Generalization and Robustness of Transformer-Based Language Models on Commonsense Reasoning

**Authors**: *Ke Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30410](https://doi.org/10.1609/aaai.v38i21.30410)

**Abstract**:

The advent of powerful transformer-based discriminative language models and, more recently, generative GPT-family models, has led to notable advancements in natural language processing (NLP), particularly in commonsense reasoning tasks. One such task is commonsense reasoning, where performance is usually evaluated through multiple-choice question-answering benchmarks. Till date, many such benchmarks have been proposed and `leaderboards' tracking state-of-the-art performance on those benchmarks suggest that transformer-based models are approaching human-like performance. However, due to documented problems such as hallucination and bias, the research focus is shifting from merely quantifying accuracy on the task to an in-depth, context-sensitive probing of LLMs' generalization and robustness. To gain deeper insight into diagnosing these models' performance in commonsense reasoning scenarios, this thesis addresses three main studies: the generalization ability of transformer-based language models on commonsense reasoning, the trend in confidence distribution of these language models confronted with ambiguous inference tasks, and a proposed risk-centric evaluation framework for both discriminative and generative language models.

----

## [2687] Does Robin Hood Use a Lightsaber?: Automated Planning for Storytelling

**Authors**: *Nisha Simon*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30411](https://doi.org/10.1609/aaai.v38i21.30411)

**Abstract**:

Humans have been using stories to entertain, educate, and persuade audiences for centuries. The advent of modern AI tools in the form of Large Language Models (LLMs) such as chatGPT continues to fulfill this purpose. However while recent work has shown that LLMs can successfully be used  for narrative generation, they lack coherence and can be prone to repetition and stilted language. Automated Planning can therefore be combined with Natural Language text generation to create narratives (stories) that are logical, coherent, and believable. A planning model provides scaffolding to an LLM so that the LLM's language generation is context-dependent, in order to allow users to create more coherent, logical, and believable stories in a variety of domains.

----

## [2688] Autonomous Policy Explanations for Effective Human-Machine Teaming

**Authors**: *Aaquib Tabrez*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30412](https://doi.org/10.1609/aaai.v38i21.30412)

**Abstract**:

Policy explanation, a process for describing the behavior of an autonomous system, plays a crucial role in effectively conveying an agent's decision-making rationale to human collaborators and is essential for safe real-world deployments. It becomes even more critical in effective human-robot teaming, where good communication allows teams to adapt and improvise successfully during uncertain situations by enabling value alignment within the teams. This thesis proposal focuses on improving human-machine teaming by developing novel human-centered explainable AI (xAI) techniques that empower autonomous agents to communicate their capabilities and limitations via multiple modalities, teach and influence human teammates' behavior as decision-support systems, and effectively build and manage trust in HRI systems.

----

## [2689] To Know the Causes of Things: Text Mining for Causal Relations

**Authors**: *Fiona Anting Tan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30413](https://doi.org/10.1609/aaai.v38i21.30413)

**Abstract**:

Causality expresses the relation between two arguments, one of which represents the cause and the other the effect (or consequence). Causal text mining refers to the extraction and usage of causal information from text. Given an input sequence, we are interested to know if and where causal information occurs. My research is focused on the end-to-end challenges of causal text mining. This involves extracting, representing, and applying causal knowledge from unstructured text. The corresponding research questions are: (1) How to extract causal information from unstructured text effectively? (2) How to represent extracted causal relationships in a graph that is interpretable and useful for some application? (3) How can we capitalize on extracted causal knowledge for downstream tasks? What tasks or fields will benefit from such knowledge? In this paper, I outline past and on-going works, and highlight future research challenges.

----

## [2690] Data Efficient Paradigms for Personalized Assessment of Black-Box Taskable AI Systems

**Authors**: *Pulkit Verma*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30414](https://doi.org/10.1609/aaai.v38i21.30414)

**Abstract**:

The vast diversity of internal designs of taskable black-box AI systems and their nuanced zones of safe functionality make it difficult for a layperson to use them without unintended side effects. My dissertation focuses on developing paradigms that enable a user to assess and understand the limits of an AI system's safe operability. We develop a personalized AI assessment module that lets an AI system execute instruction sequences in simulators and answer queries about these executions. Our results show that such a primitive query-response interface is sufficient to efficiently derive a user-interpretable model of a system's capabilities.

----

## [2691] Neuro-Symbolic Integration for Reasoning and Learning on Knowledge Graphs

**Authors**: *Luisa Werner*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30415](https://doi.org/10.1609/aaai.v38i21.30415)

**Abstract**:

The goal of this thesis is to address knowledge graph completion tasks using neuro-symbolic methods. Neuro-symbolic methods allow the joint utilization of symbolic information defined as meta-rules in ontologies and knowledge graph embedding methods that represent entities and relations of the graph in a low-dimensional vector space. This approach has the potential to improve the resolution of knowledge graph completion tasks in terms of reliability, interpretability, data-efficiency and robustness.

----

## [2692] Visual Abstract Reasoning in Computational Imagery

**Authors**: *Yuan Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30416](https://doi.org/10.1609/aaai.v38i21.30416)

**Abstract**:

Despite current AI’s human-like behavior, super efficiency, and unbelievable ability to handle complex games, we still complain that it shows no sign of creativity, originality, or novelty outside its training set, and that it fails to develop new insights into old experience or establish understanding of new experience. In short, it generates content from its training set, but does not invent content. A fundamental reason for this is that current AI is incapable of abstraction and reasoning in an abstract, generalizable, and systematic way. Think, for instance, of what AI systems we can build if we have a base system that can answer this simple question—when two things are the same. Instead of studying these high-level questions, I put my thesis in the context of visual abstract reasoning (VAR), a task widely used in human intelligence tests. A classical example of this task is Raven’s Progressive Matrices (RPM, see Figure 1), a family of intelligence tests that was designed to measure eductive ability, i.e., the ability to make meaning out of confusion and generate high-level, usually nonverbal, schemata which make it easy to handle complexity. A similar concept to eductive ability is fluid intelligence, or the ability to discriminate and perceive complex relationships when no recourse to answers is stored in memory. Whether eductive ability or fluid intelligence, RPM points to the qualities that have been lacking in AI. To explore these qualities in AI, I propose the following research questions.

----

## [2693] Multipartite Entity Resolution: Motivating a K-Tuple Perspective (Student Abstract)

**Authors**: *Adin Aberbach, Mayank Kejriwal, Ke Shen*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30417](https://doi.org/10.1609/aaai.v38i21.30417)

**Abstract**:

Entity Resolution (ER) is the problem of algorithmically matching records, mentions, or entries that refer to the same underlying real-world entity. Traditionally, the problem assumes (at most) two datasets, between which records need to be matched. There is considerably less research in ER when k > 2 datasets are involved. The evaluation of such multipartite ER (M-ER) is especially complex, since the usual ER metrics assume (whether implicitly or explicitly) k < 3. This paper takes the first step towards motivating a k-tuple approach for evaluating M-ER. Using standard algorithms and k-tuple versions of metrics like precision and recall, our preliminary results suggest a significant difference compared to aggregated pairwise evaluation, which would first decompose the M-ER problem into independent bipartite problems and then aggregate their metrics. Hence, M-ER may be more challenging and warrant more novel approaches than current decomposition-based pairwise approaches would suggest.

----

## [2694] Preference-Aware Constrained Multi-Objective Bayesian Optimization (Student Abstract)

**Authors**: *Alaleh Ahmadianshalchi, Syrine Belakaria, Janardhan Rao Doppa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30418](https://doi.org/10.1609/aaai.v38i21.30418)

**Abstract**:

We consider the problem of constrained multi-objective optimization over black-box objectives, with user-defined preferences, with a largely infeasible input space. Our goal is  to approximate the optimal Pareto set from the small fraction of feasible inputs. The main challenges include huge design space, multiple objectives, numerous constraints, and rare feasible inputs identified only through expensive experiments. We present PAC-MOO, a novel preference-aware multi-objective Bayesian optimization algorithm to solve this problem. It leverages surrogate models for objectives and constraints to intelligently select the sequence of inputs for evaluation to achieve the target goal.

----

## [2695] Incorporating Serverless Computing into P2P Networks for ML Training: In-Database Tasks and Their Scalability Implications (Student Abstract)

**Authors**: *Amine Barrak*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30419](https://doi.org/10.1609/aaai.v38i21.30419)

**Abstract**:

Distributed ML addresses challenges from increasing data and model complexities. Peer to peer (P2P) networks in distributed ML offer scalability and fault tolerance. However, they also encounter challenges related to resource consumption, and communication overhead as the number of participating peers grows. This research introduces a novel architecture that combines serverless computing with P2P networks for distributed training. Serverless computing enhances this model with parallel processing and cost effective scalability, suitable for resource-intensive tasks. Preliminary results show that peers can offload expensive computational tasks to serverless platforms. However, their inherent statelessness necessitates strong communication methods, suggesting a pivotal role for databases. To this end, we have enhanced an in memory database to support ML training tasks.

----

## [2696] Sleep-Like Unsupervised Replay Improves Performance When Data Are Limited or Unbalanced (Student Abstract)

**Authors**: *Anthony Bazhenov, Pahan Dewasurendra, Giri Krishnan, Jean Erik Delanois*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30420](https://doi.org/10.1609/aaai.v38i21.30420)

**Abstract**:

The performance of artificial neural networks (ANNs) degrades when training data are limited or imbalanced. In contrast, the human brain can learn quickly from just a few examples. Here, we investigated the role of sleep in improving the performance of ANNs trained with limited data on the MNIST and Fashion MNIST datasets. Sleep was implemented as an unsupervised phase with local Hebbian type learning rules. We found a significant boost in accuracy after the sleep phase for models trained with limited data in the range of 0.5-10% of total MNIST or Fashion MNIST datasets. When more than 10% of the total data was used, sleep alone had a slight negative impact on performance, but this was remedied by fine-tuning on the original data. This study sheds light on a potential synaptic weight dynamics strategy employed by the brain during sleep to enhance memory performance when training data are limited or imbalanced.

----

## [2697] Coalition Formation for Task Allocation Using Multiple Distance Metrics (Student Abstract)

**Authors**: *Tuhin Kumar Biswas, Avisek Gupta, Narayan Changder, Redha Taguelmimt, Samir Aknine, Samiran Chattopadhyay, Animesh Dutta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30421](https://doi.org/10.1609/aaai.v38i21.30421)

**Abstract**:

Simultaneous Coalition Structure Generation and Assignment (SCSGA) is an important research problem in multi-agent systems. Given n agents and m tasks, the aim of SCSGA is to form m disjoint coalitions of n agents such that between the coalitions and tasks there is a one-to-one mapping, which ensures each coalition is capable of accomplishing the assigned task. SCSGA with Multi-dimensional Features (SCSGA-MF) extends the problem by introducing a d-dimensional vector for each agent and task. We propose a heuristic algorithm called Multiple Distance Metric (MDM) approach to solve SCSGA-MF. Experimental results confirm that MDM produces near optimal solutions, while being feasible for large-scale inputs within a reasonable time frame.

----

## [2698] The Inhibitor: ReLU and Addition-Based Attention for Efficient Transformers (Student Abstract)

**Authors**: *Rickard Brännvall*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30422](https://doi.org/10.1609/aaai.v38i21.30422)

**Abstract**:

To enhance the computational efficiency of quantized Transformers, we replace the dot-product and Softmax-based attention with an alternative mechanism involving addition and ReLU activation only. This side-steps the expansion to double precision often required by matrix multiplication and avoids costly Softmax evaluations but maintains much of the core functionality of conventional dot-product attention. It can enable more efficient execution and support larger quantized Transformer models on resource-constrained hardware or alternative arithmetic systems like homomorphic encryption. Training experiments on four common benchmark tasks show test set prediction scores comparable to those of conventional Transformers with dot-product attention. Our scaling experiments also suggest significant computational savings, both in plaintext and under encryption. 
In particular, we believe that the ReLU and addition-based attention mechanism introduced in this paper may enable privacy-preserving AI applications operating under homomorphic encryption by avoiding the costly multiplication of encrypted variables.

----

## [2699] JoLT: Jointly Learned Representations of Language and Time-Series for Clinical Time-Series Interpretation (Student Abstract)

**Authors**: *Yifu Cai, Arvind Srinivasan, Mononito Goswami, Arjun Choudhry, Artur Dubrawski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30423](https://doi.org/10.1609/aaai.v38i21.30423)

**Abstract**:

Time-series and text data are prevalent in healthcare and frequently co-exist, yet they are typically modeled in isolation. Even studies that jointly model time-series and text, do so by converting time-series to images or graphs. We hypothesize that explicitly modeling time-series jointly with text can improve tasks such as summarization and question answering for time-series data, which have received little attention so far. To address this gap, we introduce JoLT to jointly learn desired representations from pre-trained time-series and text models. JoLT utilizes a Querying Transformer (Q-Former) to align the time-series and text representations. Our experiments on a large real-world electrocardiography dataset for medical time-series summarization show that JoLT outperforms state-of-the-art image captioning approaches.

----

## [2700] Data-Driven Discovery of Design Specifications (Student Abstract)

**Authors**: *Angela Chen, Nicholas Gisolfi, Artur Dubrawski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30424](https://doi.org/10.1609/aaai.v38i21.30424)

**Abstract**:

Ensuring a machine learning model’s trustworthiness is crucial to prevent potential harm. One way to foster trust is through the formal verification of the model’s adherence to essential design requirements. However, this approach relies on well-defined, application-domain-centric criteria with which to test the model, and such specifications may be cumbersome to collect in practice. We propose a data-driven approach for creating specifications to evaluate a trained model effectively. Implementing this framework allows us to prove that the model will exhibit safe behavior while minimizing the false-positive prediction rate. This strategy enhances predictive accuracy and safety, providing deeper insight into the model’s strengths and weaknesses, and promotes trust through a systematic approach.

----

## [2701] Interpreting Temporal Knowledge Graph Reasoning (Student Abstract)

**Authors**: *Bin Chen, Kai Yang, Wenxin Tai, Zhangtao Cheng, Leyuan Liu, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30425](https://doi.org/10.1609/aaai.v38i21.30425)

**Abstract**:

Temporal knowledge graph reasoning is an essential task that holds immense value in diverse real-world applications. Existing studies mainly focus on leveraging structural and sequential dependencies, excelling in tasks like entity and link prediction. However, they confront a notable interpretability gap in their predictions, a pivotal facet for comprehending model behavior. In this study, we propose an innovative method, LSGAT, which not only exhibits remarkable precision in entity predictions but also enhances interpretability by identifying pivotal historical events influencing event predictions. LSGAT enables concise explanations for prediction outcomes, offering valuable insights into the otherwise enigmatic "black box" reasoning process. Through an exploration of the implications of the most influential events, it facilitates a deeper understanding of the underlying mechanisms governing predictions.

----

## [2702] The Language Model Can Have the Personality: Joint Learning for Personality Enhanced Language Model (Student Abstract)

**Authors**: *Tianyi Chen, Feiqi Cao, Yihao Ding, Soyeon Caren Han*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30426](https://doi.org/10.1609/aaai.v38i21.30426)

**Abstract**:

With the introduction of large language models, chatbots are becoming more conversational to communicate effectively and capable of handling increasingly complex tasks. To make a chatbot more relatable and engaging, we propose a new language model idea that maps the human-like personality.
In this paper, we propose a systematic Personality-Enhanced Language Model (PELM) approach by using a joint learning mechanism of personality classification and language generation tasks. The proposed PELM leverages a dataset of defined personality typology, Myers-Briggs Type Indicator, and produces a Personality-Enhanced Language Model by using a joint learning and cross-teaching structure consisting of a classification and language modelling to incorporate personalities via both distinctive types and textual information. The results show that PELM can generate better personality-based outputs than baseline models.

----

## [2703] MapLE: Matching Molecular Analogues Promptly with Low Computational Resources by Multi-Metrics Evaluation (Student Abstract)

**Authors**: *Xiaojian Chen, Chuyue Liao, Yanhui Gu, Yafei Li, Jinlan Wang, Yi Chen, Masaru Kitsuregawa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30427](https://doi.org/10.1609/aaai.v38i21.30427)

**Abstract**:

Matching molecular analogues is a computational chemistry and bioinformatics research issue which is used to identify molecules that are structurally or functionally similar to a target molecule. Recent studies on matching analogous molecules have predominantly concentrated on enhancing effectiveness, often sidelining computational efficiency, particularly in contexts of low computational resources. This oversight poses challenges in many real applications (e.g., drug discovery, catalyst generation and so forth). To tackle this issue, we propose a general strategy named MapLE, aiming to promptly match analogous molecules with low computational resources by multi-metrics evaluation. Experimental evaluation conducted on a public biomolecular dataset validates the excellent and efficient performance of the proposed strategy.

----

## [2704] Dual Mapping of 2D StyleGAN for 3D-Aware Image Generation and Manipulation (Student Abstract)

**Authors**: *Zhuo Chen, Haimei Zhao, Chaoyue Wang, Bo Yuan, Xiu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30428](https://doi.org/10.1609/aaai.v38i21.30428)

**Abstract**:

3D-aware GANs successfully solve the problem of 3D-consistency generation and furthermore provide a 3D shape of the generated object. However, the application of the volume renderer disturbs the disentanglement of the latent space, which makes it difficult to manipulate 3D-aware GANs and lowers the image quality of style-based generators. In this work, we devise a dual-mapping framework to make the generated images of pretrained 2D StyleGAN consistent in 3D space. We utilize a tri-plane representation to estimate the 3D shape of the generated object and two mapping networks to bridge the latent space of StyleGAN and the 3D tri-plane space. Our method does not alter the parameters of the pretrained generator, which means the interpretability of latent space is preserved for various image manipulations. Experiments show that our method lifts the 3D awareness of pretrained 2D StyleGAN to 3D-aware GANs and outperforms the 3D-aware GANs in controllability and image quality.

----

## [2705] STViT: Improving Self-Supervised Multi-Camera Depth Estimation with Spatial-Temporal Context and Adversarial Geometry Regularization (Student Abstract)

**Authors**: *Zhuo Chen, Haimei Zhao, Bo Yuan, Xiu Li*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30429](https://doi.org/10.1609/aaai.v38i21.30429)

**Abstract**:

Multi-camera depth estimation has recently garnered significant attention due to its substantial practical implications in the realm of autonomous driving. In this paper, we delve into the task of self-supervised multi-camera depth estimation and propose an innovative framework, STViT, featuring several noteworthy enhancements: 1) we propose a Spatial-Temporal Transformer to comprehensively exploit both local connectivity and the global context of image features, meanwhile learning enriched spatial-temporal cross-view correlations to recover 3D geometry. 2) to alleviate the severe effect of adverse conditions, e.g., rainy weather and nighttime driving, we introduce a GAN-based Adversarial Geometry Regularization Module (AGR) to further constrain the depth estimation with unpaired normal-condition depth maps and prevent the model from being incorrectly trained. Experiments on challenging autonomous driving datasets Nuscenes and DDAD show that our method achieves state-of-the-art performance.

----

## [2706] Simple Orthogonal Graph Representation Learning (Student Abstract)

**Authors**: *Taoyong Cui, Yuhan Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30430](https://doi.org/10.1609/aaai.v38i21.30430)

**Abstract**:

Graph neural networks (GNNs) have attracted significant interest recently since they can effectively process and analyze graph-structured data commonly found in real-world applications. However, the predicament that GNNs are difficult to train becomes worse as the layers increase. The essence of this problem is that stacking layers will reduce the stability of forward propagation and gradient back-propagation. And as the increasing scale of models (measured by the number of parameters), how to efficiently and effectively adapt it to particular downstream tasks becomes an intriguing research issue. In this work, motivated by the effect of orthogonality constraints, we propose a simple orthogonal training framework to impose the orthogonality constraints on GNNs, which can help models find a solution vector in a specific low dimensional subspace and stabilize the signaling processes at both the forward and backward directions. Specifically, we propose a novel polar decomposition-based orthogonal initialization (PDOI-R) algorithm, which can identify the low intrinsic dimension within the Stiefel Manifold and stabilize the training process. Extensive experiments demonstrate the effectiveness of the proposed method in multiple downstream tasks, showcasing its generality. The simple method can help existing state-of-the-art models achieve better performance.

----

## [2707] Contrastive Learning for Low-Light Raw Denoising (Student Abstract)

**Authors**: *Taoyong Cui, Yuhan Dong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30431](https://doi.org/10.1609/aaai.v38i21.30431)

**Abstract**:

Image/video denoising in low-light scenes is an extremely challenging problem due to limited photon count and high noise. In this paper, we propose a novel approach with contrastive learning to address this issue. Inspired by the success of contrastive learning used in some high-level computer vision tasks, we bring in this idea to the low-level denoising task. In order to achieve this goal, we introduce a new denoising contrastive regularization (DCR) to exploit the information of noisy images and clean images. In the feature space, DCR makes the denoised image closer to the clean image and far away from the noisy image. In addition,  we build a new feature embedding network called Wnet, which is more effective to extract high-frequency information. We conduct the experiments on a real low-light dataset that captures still images taken on a moonless clear night in 0.6 millilux and videos under starlight (no moon present). The results show that our method can achieve a higher PSNR and better visual quality compared with existing methods.

----

## [2708] Strategic Recommendation: Revenue Optimal Matching for Online Platforms (Student Abstract)

**Authors**: *Luca D'Amico-Wong, Gary Qiurui Ma, David C. Parkes*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30432](https://doi.org/10.1609/aaai.v38i21.30432)

**Abstract**:

We consider a platform in a two-sided market with unit-supply sellers and unit-demand buyers. Each buyer can transact with a subset of sellers it knows off platform and another seller that the platform recommends. Given the choice of sellers, transactions and prices form a competitive equilibrium. The platform selects one seller for each buyer, and charges a fixed percentage of prices to all transactions that it recommends. The platform seeks to maximize total revenue.

We show that the platform's problem is NP-hard, even when each buyer knows at most two buyers off platform. Finally, when each buyer values all sellers equally and knows only one buyer off platform, we provide a polynomial time algorithm that optimally solves the problem.

----

## [2709] Improving Faithfulness in Abstractive Text Summarization with EDUs Using BART (Student Abstract)

**Authors**: *Narjes Delpisheh, Yllias Chali*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30433](https://doi.org/10.1609/aaai.v38i21.30433)

**Abstract**:

Abstractive text summarization uses the summarizer’s own words to capture the main information of a source document in a summary. While it is more challenging to automate than extractive text summarization, recent advancements in deep learning approaches and pre-trained language models have improved its performance. However, abstractive text summarization still has issues such as unfaithfulness. To address this problem, we propose a new approach that utilizes important Elementary Discourse Units (EDUs) to guide BART-based text summarization. Our approach showed the improvement in truthfulness and source document coverage in comparison to some previous studies.

----

## [2710] Scene Flow Prior Based Point Cloud Completion with Masked Transformer (Student Abstract)

**Authors**: *Junzhe Ding, Yufei Que, Jin Zhang, Cheng Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30434](https://doi.org/10.1609/aaai.v38i21.30434)

**Abstract**:

It is necessary to explore an effective point cloud completion mechanism that is of great significance for real-world tasks such as autonomous driving, robotics applications, and multi-target tracking. In this paper, we propose a point cloud completion method using a self-supervised transformer model based on the contextual constraints of scene flow. Our method uses the multi-frame point cloud context relationship as a guide to generate a series of token proposals, this priori condition ensures the stability of the point cloud completion. The experimental results show that the method proposed in this paper achieves high accuracy and good stability.

----

## [2711] Kepler Light Curve Classification Using Deep Learning and Markov Transition Field (Student Abstract)

**Authors**: *Shane Donnelly, Ayan Dutta*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30435](https://doi.org/10.1609/aaai.v38i21.30435)

**Abstract**:

An exoplanet is a planet, which is not a part of our solar system.
Whether life exists in one or more of these exoplanets
has fascinated humans for centuries. NASA’s Kepler Space
Telescope has discovered more than 70% of known exoplanets
in our universe. However, manually determining whether a
Kepler light curve indicates an exoplanet or not becomes infeasible
with the large volume of data. Due to this, we propose
a deep learning-based strategy to automatically classify
a Kepler light curve. More specifically, we first convert the
light curve time series into its corresponding Markov Transition
Field (MTF) image and then classify it. Results show
that the accuracy of the proposed technique is 99.39%, which
is higher than all current state-of-the-art approaches.

----

## [2712] Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers (Student Abstract)

**Authors**: *Danilo Dordevic, Vukasin Bozic, Joseph Thommes, Daniele Coppola, Sidak Pal Singh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30436](https://doi.org/10.1609/aaai.v38i21.30436)

**Abstract**:

This work presents an analysis of the effectiveness of using standard shallow feed-forward networks to mimic the behavior of the attention mechanism in the original Transformer model, a state-of-the-art architecture for sequence-to-sequence tasks. We substitute key elements of the attention mechanism in the Transformer with simple feed-forward networks, trained using the original components via knowledge distillation. Our experiments, conducted on the IWSLT2017 dataset, reveal the capacity of these ”attentionless Transformers” to rival the performance of the original architecture. Through rigorous ablation studies, and experimenting with various replacement network types and sizes, we offer insights that support the viability of our approach. This not only sheds light on the adaptability of shallow feed-forward
networks in emulating attention mechanisms but also underscores their potential to streamline complex architectures for sequence-to-sequence tasks.

----

## [2713] A SAT + Computer Algebra System Verification of the Ramsey Problem R(3, 8) (Student Abstract)

**Authors**: *Conor Duggan, Zhengyu Li, Curtis Bright, Vijay Ganesh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30437](https://doi.org/10.1609/aaai.v38i21.30437)

**Abstract**:

The Ramsey problem R(3,8) asks for the smallest n such that every red/blue coloring of the complete graph on n vertices must contain either a blue triangle or a red 8-clique. We provide the first certifiable proof that R(3,8) = 28, automatically generated by a combination of Boolean satisfiability (SAT) solver and a computer algebra system (CAS). This SAT+CAS combination is significantly faster than a SAT-only approach. While the R(3,8) problem was first computationally solved by McKay and Min in 1992, it was not a verifiable proof. The SAT+CAS method that we use for our proof is very general and can be applied to a wide variety of combinatorial problems.

----

## [2714] PICSR: Prototype-Informed Cross-Silo Router for Federated Learning (Student Abstract)

**Authors**: *Eric Enouen, Sebastian Caldas, Mononito Goswami, Artur Dubrawski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30438](https://doi.org/10.1609/aaai.v38i21.30438)

**Abstract**:

Federated Learning is an effective approach for learning from data distributed across multiple institutions. While most existing studies are aimed at improving predictive accuracy of models, little work has been done to explain knowledge differences between institutions and the benefits of collaboration. Understanding these differences is critical in cross-silo federated learning domains, e.g., in healthcare or banking, where each institution or silo has a different underlying distribution and stakeholders want to understand how their institution compares to their partners. We introduce Prototype-Informed Cross-Silo Router (PICSR) which utilizes a mixture of experts approach to combine local models derived from multiple silos. Furthermore, by computing data similarity to prototypical samples from each silo, we are able to ground the router’s predictions in the underlying dataset distributions. Experiments on a real-world heart disease prediction dataset show that PICSR retains high performance while enabling further explanations on the differences among institutions compared to a single black-box model.

----

## [2715] Sequential Modeling of Complex Marine Navigation: Case Study on a Passenger Vessel (Student Abstract)

**Authors**: *Yimeng Fan, Pedram Agand, Mo Chen, Edward J. Park, Allison Kennedy, Chanwoo Bae*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30439](https://doi.org/10.1609/aaai.v38i21.30439)

**Abstract**:

The maritime industry's continuous commitment to sustainability has led to a dedicated exploration of methods to reduce vessel fuel consumption. This paper undertakes this challenge through a machine learning approach, leveraging a real-world dataset spanning two years of a passenger vessel in west coast Canada. Our  focus centers on the creation of a time series forecasting model given the dynamic and static states, actions, and disturbances. This model is designed to predict dynamic states based on the actions provided, subsequently serving as an evaluative tool to assess the proficiency of the vessel's operation under the captain's guidance. Additionally, it lays the foundation for future optimization algorithms, providing valuable feedback on decision-making processes. To facilitate future studies, our code is available at https://github.com/pagand/model_optimze_vessel/tree/AAAI.

----

## [2716] Local Consistency Guidance: Personalized Stylization Method of Face Video (Student Abstract)

**Authors**: *Wancheng Feng, Yingchao Liu, Jiaming Pei, Wenxuan Liu, Chunpeng Tian, Lukun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30440](https://doi.org/10.1609/aaai.v38i21.30440)

**Abstract**:

Face video stylization aims to convert real face videos into specified reference styles. While one-shot methods perform well in single-image stylization, ensuring continuity between frames and retaining the original facial expressions present challenges in video stylization. To address these issues, our approach employs a personalized diffusion model with pixel-level control. We propose Local Consistency Guidance(LCG) strategy, composed of local-cross attention and local style transfer, to ensure temporal consistency. This framework enables the synthesis of high-quality stylized face videos with excellent temporal continuity.

----

## [2717] Potential-Based Reward Shaping for Intrinsic Motivation (Student Abstract)

**Authors**: *Grant C. Forbes, David L. Roberts*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30441](https://doi.org/10.1609/aaai.v38i21.30441)

**Abstract**:

Recently there has been a proliferation of intrinsic motivation (IM) reward shaping methods to learn in complex and sparse-reward environments. These methods can often inadvertently change the set of optimal policies in an environment, leading to suboptimal behavior. Previous work on mitigating the risks of reward shaping, particularly through potential-based reward shaping (PBRS), has not been applicable to many IM methods, as they are often complex, trainable functions themselves, and therefore dependent on a wider set of variables than the traditional reward functions that PBRS was developed for. We present an extension to PBRS that we show preserves the set of optimal policies under a more general set of functions than has been previously demonstrated. We also present Potential-Based Intrinsic Motivation (PBIM), a method for converting IM rewards into a potential-based form that are useable without altering the set of optimal policies. Testing in the MiniGrid DoorKey environment, we demonstrate that PBIM successfully prevents the agent from converging to a suboptimal policy and can speed up training.

----

## [2718] Spatial-Temporal Augmentation for Crime Prediction (Student Abstract)

**Authors**: *Hongzhu Fu, Fan Zhou, Qing Guo, Qiang Gao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30442](https://doi.org/10.1609/aaai.v38i21.30442)

**Abstract**:

Crime prediction stands as a pivotal concern within the realm of urban management due to its potential threats to public safety. While prior research has predominantly focused on unraveling the intricate dependencies among urban regions and temporal dynamics, the challenges posed by the scarcity and uncertainty of historical crime data have not been thoroughly investigated. This study introduces an innovative spatial-temporal augmented learning framework for crime prediction, namely STAug. In STAug, we devise a CrimeMix to improve the ability of generalization. Furthermore, we harness a spatial-temporal aggregation to capture and incorporate multiple correlations covering the temporal, spatial, and crime-type aspects. Experiments on two real-world datasets underscore the superiority of STAug over several baselines.

----

## [2719] Evaluating the Efficacy of Prompting Techniques for Debiasing Language Model Outputs (Student Abstract)

**Authors**: *Shaz Furniturewala, Surgan Jandial, Abhinav Java, Simra Shahid, Pragyan Banerjee, Balaji Krishnamurthy, Sumit Bhatia, Kokil Jaidka*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30443](https://doi.org/10.1609/aaai.v38i21.30443)

**Abstract**:

Achieving fairness in Large Language Models (LLMs) continues to pose a persistent challenge, as these models are prone to inheriting biases from their training data, which can subsequently impact their performance in various applications. There is a need to systematically explore whether structured prompting techniques can offer opportunities for debiased text generation by LLMs. In this work, we designed an evaluative framework to test the efficacy of different prompting techniques for debiasing text along different dimensions. We aim to devise a general structured prompting approach to achieve fairness that generalizes well to different texts and LLMs.

----

## [2720] Memory-Augmenting Decoder-Only Language Models through Encoders (Student Abstract)

**Authors**: *Alessio Galatolo, Katie Winkle*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30444](https://doi.org/10.1609/aaai.v38i21.30444)

**Abstract**:

The Transformer architecture has seen a lot of attention in recent years also thanks to its ability to scale well and allow massive parallelism during training. This has made possible the development of Language Models (LMs) of increasing size and the discovery of latent abilities that completely outclass traditional methods e.g. rule-based systems. However, they also introduced new issues, like their inability to retain the history of previous interactions due to their stateless nature or the difficulty in controlling their generation. Different attempts have been made to address these issues, e.g. a `brute force' approach to solving the memory issue is to include the full conversation history in the context window, a solution that is limited by the quadratic scalability of Transformers. In this work, we explore computationally practical solutions to the memory problem. We propose to augment the decoder-only architecture of (most) Large LMs with a (relatively small) memory encoder. Its output is prepended to the decoder's input in a similar fashion to recent works in Adapters and the original Transformer architecture. Initial experiments show promising results, however future work is needed to compare with State-of-the-Art methods.

----

## [2721] Multilingual Medical Language Models: A Path to Improving Lay Health Worker Effectiveness (Student Abstract)

**Authors**: *Agasthya Gangavarapu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30445](https://doi.org/10.1609/aaai.v38i21.30445)

**Abstract**:

The COVID-19 pandemic has exacerbated the challenges faced by healthcare delivery in developing nations, placing additional strain on already fragile infrastructure and healthcare systems. This has prompted an increased reliance on lay healthcare workers (LHWs) to meet the surging demand for services. Due to limited formal training, many LHWs have resorted to using unreliable sources, such as internet searches, to access medical information.

Large language models (LLMs) offer a promising opportunity to support LHWs by providing accurate, context-sensitive information for improving healthcare delivery, provided they are appropriately fine-tuned on domain-specific multilingual data. This paper delves into critical issues and presents potential solutions for developing LLM-powered virtual assistants tailored to LHWs serving Telugu and Hindi-speaking populations. Key focal points include the customization of language and content to suit local contexts, the integration of feedback mechanisms to continuously enhance assistance quality, and the delicate balance between automation and human oversight.

----

## [2722] Enhancing Transcription Factor Prediction through Multi-Task Learning (Student Abstract)

**Authors**: *Liyuan Gao, Matthew Zhang, Victor S. Sheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30446](https://doi.org/10.1609/aaai.v38i21.30446)

**Abstract**:

Transcription factors (TFs) play a fundamental role in gene regulation by selectively binding to specific DNA sequences. Understanding the nature and behavior of these TFs is essential for insights into gene regulation dynamics. In this study, we introduce a robust multi-task learning framework specifically tailored to harness both TF-specific annotations and TF-related domain annotations, thereby enhancing the accuracy of TF predictions. Notably, we incorporate cutting-edge language models that have recently garnered attention for their outstanding performance across various fields, particularly in biological computations like protein sequence modeling. Comparative experimental analysis with existing models, DeepTFactor and TFpredict, reveals that our multi-task learning framework achieves an accuracy exceeding 92% across four evaluation metrics on the TF prediction task, surpassing both competitors. Our work marks a significant leap in the domain of TF prediction, enriching our comprehension of gene regulatory mechanisms and paving the way for the discovery of novel regulatory motifs.

----

## [2723] Engineering the Neural Collapse Geometry of Supervised-Contrastive Loss (Student Abstract)

**Authors**: *Jaidev Gill, Vala Vakilian, Christos Thrampoulidis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30447](https://doi.org/10.1609/aaai.v38i21.30447)

**Abstract**:

Supervised-contrastive loss (SCL) is an alternative to cross-entropy (CE) for classification tasks that makes use of similarities in the embedding space to allow for richer representations. Previous works have used trainable prototypes to help
improve test accuracy of SCL when training under imbalance. In this work, we propose the use of fixed prototypes to help engineering the feature geometry when training with SCL. We gain further insights by considering a limiting scenario
where the number of prototypes far outnumber the original batch size. Through this, we establish a connection to CE loss with a fixed classifier and normalized embeddings. We validate our findings by conducting a series of experiments with deep neural networks on benchmark vision datasets.

----

## [2724] BadSAM: Exploring Security Vulnerabilities of SAM via Backdoor Attacks (Student Abstract)

**Authors**: *Zihan Guan, Mengxuan Hu, Zhongliang Zhou, Jielu Zhang, Sheng Li, Ninghao Liu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30448](https://doi.org/10.1609/aaai.v38i21.30448)

**Abstract**:

Image segmentation is foundational to computer vision applications, and the Segment Anything Model (SAM) has become a leading base model for these tasks. However, SAM falters in specialized downstream challenges, leading to various customized SAM models. We introduce BadSAM, a backdoor attack tailored for SAM, revealing that customized models can harbor malicious behaviors. Using the CAMO dataset, we confirm BadSAM's efficacy and identify SAM vulnerabilities. This study paves the way for the development of more secure and customizable vision foundation models.

----

## [2725] Towards a Transformer-Based Reverse Dictionary Model for Quality Estimation of Definitions (Student Abstract)

**Authors**: *Julien Guité-Vinet, Alexandre Blondin Massé, Fatiha Sadat*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30449](https://doi.org/10.1609/aaai.v38i21.30449)

**Abstract**:

In the last years, several variants of transformers have emerged. In this paper, we compare different transformer-based models for solving the reverse dictionary task and explore their use in the context of a serious game called The Dictionary Game.

----

## [2726] Digital Twin-Driven Teat Localization and Shape Identification for Dairy Cow (Student Abstract)

**Authors**: *Aarushi Gupta, Yuexing Hao, Yuting Yang, Tiancheng Yuan, Matthias Wieland, Parminder S. Basran, Ken Birman*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30450](https://doi.org/10.1609/aaai.v38i21.30450)

**Abstract**:

Dairy owners invest heavily to keep their animals healthy. There is good reason to hope that technologies such as computer vision and artificial intelligence (AI) could reduce costs, yet obstacles arise when adapting these advanced tools to farming environments. In this work, we applied AI tools to dairy cow teat localization and teat shape classification, obtaining a model that achieves a mean average precision of 0.783. This digital twin-driven approach is intended as a first step towards automating and accelerating the detection and treatment of hyperkeratosis, mastitis, and other medical conditions that significantly burden the dairy industry.

----

## [2727] Structurally Guided Task Decomposition in Spatial Navigation Tasks (Student Abstract)

**Authors**: *Ruiqi He, Carlos G. Correa, Tom Griffiths, Mark K. Ho*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30451](https://doi.org/10.1609/aaai.v38i21.30451)

**Abstract**:

How are people able to plan so efficiently despite limited cognitive resources? We aimed to answer this question by extending an existing model of human task decomposition that can explain a wide range of simple planning problems by adding structure information to the task to facilitate planning in more complex tasks. The extended model was then applied to a more complex planning domain of spatial navigation. Our results suggest that our framework can correctly predict the navigation strategies of the majority of the participants in an online experiment.

----

## [2728] Disentanglement-Guided Spatial-Temporal Graph Neural Network for Metro Flow Forecasting (Student Abstract)

**Authors**: *Jinyu Hong, Ping Kuang, Qiang Gao, Fan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30452](https://doi.org/10.1609/aaai.v38i21.30452)

**Abstract**:

In recent intelligent transportation applications, metro flow forecasting has received much attention from researchers. Most prior arts endeavor to explore spatial or temporal dependencies while ignoring the key characteristic patterns underlying historical flows, e.g., trend and periodicity. Although the multiple granularity distillations or spatial dependency correlation can promote the flow estimation. However, the potential noise and spatial dynamics are under-explored. To this end, we propose a novel Disentanglement-Guided Spatial-Temporal Graph Neural Network or DGST to address the above concerns. It contains a Disentanglement Pre-training procedure for characteristic pattern disentanglement learning, a Characteristic Pattern Prediction for different future characteristic explorations, and a Spatial-Temporal Correlation for spatial-temporal dynamic learning. Experiments on a real-world dataset demonstrate the superiority of our DGST.

----

## [2729] Novax or Novak? Estimating Social Media Stance towards Celebrity Vaccine Hesitancy (Student Abstract)

**Authors**: *Madhav Hota, Adel Khorramrouz, Ashiqur R. KhudaBukhsh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30453](https://doi.org/10.1609/aaai.v38i21.30453)

**Abstract**:

On 15 January 2022, noted tennis player Novak Djokovic was deported from Australia due to his unvaccinated status for the COVID-19 vaccine. This paper presents a stance classifier and evaluates public reaction to this episode and the impact of this behavior on social media discourse on YouTube. We observed a significant spike of individuals who supported and opposed his behavior at the time of the episode. Supporters outnumbered those who opposed this behavior by over 4x. Our study reports a disturbing trend that following every major Djokovic win, even now, vaccine skeptics often conflate his tennis success as a fitting reply to vaccine mandates.

----

## [2730] Explainable Earnings Call Representation Learning (Student Abstract)

**Authors**: *Yanlong Huang, Yue Lei, Wenxin Tai, Zhangtao Cheng, Ting Zhong, Kunpeng Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30454](https://doi.org/10.1609/aaai.v38i21.30454)

**Abstract**:

Earnings call transcripts hold valuable insights that are vital for investors and analysts when making informed decisions. However, extracting these insights from lengthy and complex transcripts can be a challenging task. The traditional manual examination is not only time-consuming but also prone to errors and biases. Deep learning-based representation learning methods have emerged as promising and automated approaches to tackle this problem. Nevertheless, they may encounter significant challenges, such as the unreliability of the representation encoding process and certain domain-specific requirements in the context of finance. To address these issues, we propose a novel transcript representation learning model. Our model leverages the structural information of transcripts to effectively extract key insights, while endowing model with explainability via variational information bottleneck. Extensive experiments on two downstream financial tasks demonstrate the effectiveness of our approach.

----

## [2731] BertRLFuzzer: A BERT and Reinforcement Learning Based Fuzzer (Student Abstract)

**Authors**: *Piyush Jha, Joseph Scott, Jaya Sriram Ganeshna, Mudit Singh, Vijay Ganesh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30455](https://doi.org/10.1609/aaai.v38i21.30455)

**Abstract**:

We present a novel tool BertRLFuzzer, a BERT and Reinforcement Learning (RL) based fuzzer aimed at finding security vulnerabilities for Web applications. BertRLFuzzer works as follows: given a set of seed inputs, the fuzzer performs grammar-adhering and attack-provoking mutation operations on them to generate candidate attack vectors. The key insight of BertRLFuzzer is the use of RL with a BERT model as an agent to guide the fuzzer to efficiently learn grammar-adhering and attack-provoking mutation operators. In order to establish the efficacy of BertRLFuzzer we compare it against a total of 13 black box and white box fuzzers over a benchmark of 9 victim websites with over 16K LOC. We observed a significant improvement, relative to the nearest competing tool in terms of time to first attack (54% less), new vulnerabilities found (17 new vulnerabilities), and attack rate (4.4% more attack vectors generated).

----

## [2732] Multi-Scale Dynamic Graph Learning for Time Series Anomaly Detection (Student Abstract)

**Authors**: *Yixuan Jin, Yutao Wei, Zhangtao Cheng, Wenxin Tai, Chunjing Xiao, Ting Zhong*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30456](https://doi.org/10.1609/aaai.v38i21.30456)

**Abstract**:

The success of graph neural networks (GNNs) has spurred numerous new works leveraging GNNs for modeling multivariate time series anomaly detection. Despite their achieved performance improvements, most of them only consider static graph to describe the spatial-temporal dependencies between time series. Moreover, existing works neglect the time and scale-changing structures of time series. In this work, we propose MDGAD, a novel multi-scale dynamic graph structure learning approach for time series anomaly detection. We design a multi-scale graph structure learning module that captures the complex correlations among time series, constructing an evolving graph at each scale. Meanwhile, an anomaly detector is used to combine bilateral prediction errors to detect abnormal data. Experiments conducted on two time series datasets demonstrate the effectiveness of MDGAD.

----

## [2733] Power Grid Anomaly Detection via Hybrid LSTM-GIN Model (Student Abstract)

**Authors**: *Amelia Jobe, Richard Ky, Sandra Luo, Akshay Dhamsania, Sumit Purohit, Edoardo Serra*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30457](https://doi.org/10.1609/aaai.v38i21.30457)

**Abstract**:

Cyberattacks on power grids pose significant risks to national security. Power grid attacks typically lead to abnormal readings in power output, frequency, current, and voltage. Due to the interconnected structure of power grids, abnormalities can spread throughout the system and cause widespread power outages if not detected and dealt with promptly. Our research proposes a novel anomaly detection system for power grids that prevents overfitting. We created a network graph to represent the structure of the power grid, where nodes represent power grid components like generators and edges represent connections between nodes such as overhead power lines. We combine the capabilities of Long Short-Term Memory (LSTM) models with a Graph Isomorphism Network (GIN) in a hybrid model to pinpoint anomalies in the grid. We train our model on each category of nodes that serves a similar structural purpose to prevent overfitting of the model. We then assign each node in the graph a unique signature using a GIN. Our model achieved a 99.92% accuracy rate, which is significantly higher than a version of our model without structural encoding, which had an accuracy level of 97.30%. Our model allows us to capture structural and temporal components of power grids and develop an attack detection system with high accuracy without overfitting.

----

## [2734] Evaluating the Effectiveness of Explainable Artificial Intelligence Approaches (Student Abstract)

**Authors**: *Jinsun Jung, Hyeoneui Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30458](https://doi.org/10.1609/aaai.v38i21.30458)

**Abstract**:

Explainable Artificial Intelligence (XAI), a promising future technology in the field of healthcare, has attracted significant interest. Despite ongoing efforts in the development of XAI approaches, there has been inadequate evaluation of explanation effectiveness and no standardized framework for the evaluation has been established. This study aims to examine the relationship between subjective interpretability and perceived plausibility for various XAI explanations and to determine the factors affecting users' acceptance of the XAI explanation.

----

## [2735] Solar Power Generation Forecasting via Multimodal Feature Fusion (Student Abstract)

**Authors**: *Eul Ka, Seungeun Go, Minjin Kwak, Jeong-Hun Kim, Aziz Nasridinov*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30459](https://doi.org/10.1609/aaai.v38i21.30459)

**Abstract**:

Solar power generation has recently been in the spotlight as global warming continues to worsen. However, two significant problems may hinder solar power generation, considering that solar panels are installed outside. The first is soiling, which accumulates on solar panels, and the second is a decrease in sunlight owing to bad weather.
In this paper, we will demonstrate that the solar power generation forecasting can increase when considering soiling and sunlight information. We first introduce a dataset containing images of clean and soiled solar panels, sky images, and weather information. For accurate solar power generation forecasting, we propose a new multimodal model that aggregates various features related to weather, soiling, and sunlight. The experimental results demonstrated the high accuracy of our proposed multimodal model.

----

## [2736] Effective Data Distillation for Tabular Datasets (Student Abstract)

**Authors**: *Inwon Kang, Parikshit Ram, Yi Zhou, Horst Samulowitz, Oshani Seneviratne*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30460](https://doi.org/10.1609/aaai.v38i21.30460)

**Abstract**:

Data distillation is a technique of reducing a large dataset into a smaller dataset. The smaller dataset can then be used to train a model which can perform comparably to a model trained on the full dataset. Past works have examined this approach for image datasets, focusing on neural networks as target models. However, tabular datasets pose new challenges not seen in images. A sample in tabular dataset is a one dimensional vector unlike the two (or three) dimensional pixel grid of images, and Non-NN models such as XGBoost can often outperform neural network (NN) based models. Our contribution in this work is two-fold: 1) We show in our work that data distillation methods from images do not translate directly to tabular data; 2) We propose a new distillation method that consistently outperforms the baseline for multiple different models, including non-NN models such as XGBoost.

----

## [2737] Multivariate Time-Series Imagification with Time Embedding in Constrained Environments (Student Abstract)

**Authors**: *Seungwoo Kang, Ohyun Jo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30461](https://doi.org/10.1609/aaai.v38i21.30461)

**Abstract**:

We present an imagification approach for multivariate time-series data tailored to constrained NN-based forecasting model training environments. Our imagification process consists of two key steps: Re-stacking and time embedding. In the Re-stacking stage, time-series data are arranged based on high correlation, forming the first image channel using a sliding window technique. The time embedding stage adds two additional image channels by incorporating real-time information. We evaluate our method by comparing it with three benchmark imagification techniques using a simple CNN-based model. Additionally, we conduct a comparison with LSTM, a conventional time-series forecasting model. Experimental results demonstrate that our proposed approach achieves three times faster model training termination while maintaining forecasting accuracy.

----

## [2738] Decompositions in Compositional Translation of LTLf to DFA (Student Abstract)

**Authors**: *Yash Kankariya, Suguman Bansal*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30462](https://doi.org/10.1609/aaai.v38i21.30462)

**Abstract**:

Prior compositional methods in LTLf to DFA conversion have focussed on improving the composition phase. In this work, we examine improvements to the decomposition phase that result in overall improvements in LTLf to DFA translation. Our work is based on reducing the structure of the underlying Abstract Syntax Tree (AST) of a formula such that the new AST results in fewer composition operations.

----

## [2739] LaMAR: Laplacian Pyramid for Multimodal Adaptive Super Resolution (Student Abstract)

**Authors**: *Aditya Kasliwal, Aryan Kamani, Ishaan Gakhar, Pratinav Seth, Sriya Rallabandi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30463](https://doi.org/10.1609/aaai.v38i21.30463)

**Abstract**:

Recent advances in image-to-image translation involve the integration of non-visual imagery in deep models. Non-visual sensors, although more costly, often produce low-resolution images. To combat this, methods using RGB images to enhance the resolution of these modalities have been introduced. Fusing these modalities to achieve high-resolution results demands models with millions of parameters and extended inference times. We present LaMAR, a lightweight model. It employs Laplacian image pyramids combined with a low-resolution thermal image for Guided Thermal Super Resolution. By decomposing the RGB image into a Laplacian pyramid, LaMAR preserves image details and avoids high-resolution feature map computations, ensuring efficiency. With faster inference times and fewer parameters, our model demonstrates state-of-the-art results.

----

## [2740] IncepSeqNet: Advancing Signal Classification with Multi-Shape Augmentation (Student Abstract)

**Authors**: *Jongseok Kim, Ohyun Jo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30464](https://doi.org/10.1609/aaai.v38i21.30464)

**Abstract**:

This work proposes and analyzes IncepSeqNet which is a new model combining the Inception Module with the innovative Multi-Shape Augmentation technique. IncepSeqNet excels in feature extraction from sequence signal data consisting of a number of complex numbers to achieve superior classification accuracy across various SNR(Signal-to-Noise Ratio) environments. Experimental results demonstrate IncepSeqNet’s outperformance of existing models, particularly at low SNR levels. Furthermore, we have confirmed its applicability in practical 5G systems by using real-world signal data.

----

## [2741] Cluster-Based Sampling in Hindsight Experience Replay for Robotic Tasks (Student Abstract)

**Authors**: *Taeyoung Kim, Dongsoo Har*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30465](https://doi.org/10.1609/aaai.v38i21.30465)

**Abstract**:

In multi-goal reinforcement learning with a sparse binary reward, training agents is particularly challenging, due to a lack of successful experiences. To solve this problem, hindsight experience replay (HER) generates successful experiences even from unsuccessful ones. However, generating successful experiences from uniformly sampled ones is not an efficient process. In this paper, the impact of exploiting the property of achieved goals in generating successful experiences is investigated and a novel cluster-based sampling strategy is proposed. The proposed sampling strategy groups episodes with different achieved goals by using a cluster model and samples experiences in the manner of HER to create the training batch. The proposed method is validated by experiments with three robotic control tasks of the OpenAI Gym. The results of experiments demonstrate that the proposed method is substantially sample efficient and achieves better performance than baseline approaches.

----

## [2742] Generalizable Policy Improvement via Reinforcement Sampling (Student Abstract)

**Authors**: *Rui Kong, Chenyang Wu, Zongzhang Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30466](https://doi.org/10.1609/aaai.v38i21.30466)

**Abstract**:

Current policy gradient techniques excel in refining policies over sampled states but falter when generalizing to unseen states. To address this, we introduce Reinforcement Sampling (RS), a novel method leveraging a generalizable action value function to sample improved decisions. RS is able to improve the decision quality whenever the action value estimation is accurate. It works by improving the agent's decision on the fly on the states the agent is visiting. Compared with the historically experienced states in which conventional policy gradient methods improve the policy, the currently visited states are more relevant to the agent. Our method sufficiently exploits the generalizability of the value function on unseen states and sheds new light on the future development of generalizable reinforcement learning.

----

## [2743] Meta-Crafting: Improved Detection of Out-of-Distributed Texts via Crafting Metadata Space (Student Abstract)

**Authors**: *Ryan Koo, Yekyung Kim, Dongyeop Kang, Jaehyung Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30467](https://doi.org/10.1609/aaai.v38i21.30467)

**Abstract**:

Detecting out-of-distribution (OOD) samples is crucial for robust NLP models. Recent works observe two OOD types: background shifts (style change) and semantic shifts (content change), but existing detection methods vary in effectiveness for each type. To this end, we propose Meta-Crafting, a unified OOD detection method by constructing a new discriminative feature space utilizing 7 model-driven metadata chosen empirically that well detects both types of shifts. Our experimental results demonstrate state-of-the-art robustness to both shifts and significantly improved detection on stress datasets.

----

## [2744] Attacking CNNs in Histopathology with SNAP: Sporadic and Naturalistic Adversarial Patches (Student Abstract)

**Authors**: *Daya Kumar, Abhijith Sharma, Apurva Narayan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30468](https://doi.org/10.1609/aaai.v38i21.30468)

**Abstract**:

Convolutional neural networks (CNNs) are being increasingly
adopted in medical imaging. However, in the race for
developing accurate models, their robustness is often overlooked.
This elicits a significant concern given the safety-critical
nature of the healthcare system. Here, we highlight
the vulnerability of CNNs against a sporadic and naturalistic
adversarial patch attack (SNAP). We train SNAP to mislead
the ResNet50 model predicting metastasis in histopathological
scans of lymph node sections, lowering the accuracy by
27%. This work emphasizes the need for defense strategies
before deploying CNNs in critical healthcare settings.

----

## [2745] Novel Class Discovery for Representation of Real-World Heritage Data as Neural Radiance Fields (Student Abstract)

**Authors**: *Shivanand Kundargi, Tejas Anvekar, Ramesh Ashok Tabib, Uma Mudenagudi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30469](https://doi.org/10.1609/aaai.v38i21.30469)

**Abstract**:

Neural Radiance Fields (NeRF) have been extensively explored as a leading approach for modeling and representing 3D data across various domains. Their ability to capture arbitrary scale point clouds and generate novel views makes them particularly valuable for digitizing cultural heritage sites. However, despite their impressive rendering capabilities, prior methods have often overlooked a significant real-world challenge: handling open-world scenarios characterized by unstructured data containing multiple classes in a single set of unlabeled images. To address this challenge, we propose a novel method NCD-NeRF that leverages Novel-Class Discovery to effectively tackle the complexities inherent in real-world data with unlabeled classes while excelling in producing high-quality NeRF representation. To validate our approach, we conducted a benchmarking analysis using a custom-collected dataset featuring UNESCO World Heritage sites in India. We observe that our proposed NCD-NeRF can parallely discover novel classes and render high-quality 3D volumes.

----

## [2746] Automated Assessment of Fidelity and Interpretability: An Evaluation Framework for Large Language Models' Explanations (Student Abstract)

**Authors**: *Mu-Tien Kuo, Chih-Chung Hsueh, Richard Tzong-Han Tsai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30470](https://doi.org/10.1609/aaai.v38i21.30470)

**Abstract**:

As Large Language Models (LLMs) become more prevalent in various fields, it is crucial to rigorously assess the quality of their explanations. Our research introduces a task-agnostic framework for evaluating free-text rationales, drawing on insights from both linguistics and machine learning. We evaluate two dimensions of explainability: fidelity and interpretability. For fidelity, we propose methods suitable for proprietary LLMs where direct introspection of internal features is unattainable. For interpretability, we use language models instead of human evaluators, addressing concerns about subjectivity and scalability in evaluations. We apply our framework to evaluate GPT-3.5 and the impact of prompts on the quality of its explanations. In conclusion, our framework streamlines the evaluation of explanations from LLMs, promoting the development of safer models.

----

## [2747] Shallow Diffusion for Fast Speech Enhancement (Student Abstract)

**Authors**: *Yue Lei, Bin Chen, Wenxin Tai, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30471](https://doi.org/10.1609/aaai.v38i21.30471)

**Abstract**:

Recently, the field of Speech Enhancement has witnessed the success of diffusion-based generative models. However, these diffusion-based methods used to take multiple iterations to generate high-quality samples, leading to high computational costs and inefficiency. In this paper, we propose SDFEN (Shallow Diffusion for Fast spEech eNhancement), a novel approach for addressing the inefficiency problem while enhancing the quality of generated samples by reducing the iterative steps in the reverse process of diffusion method. Specifically, we introduce the shallow diffusion strategy initiating the reverse process with an adaptive time step to accelerate inference. In addition, a dedicated noisy predictor is further proposed to guide the adaptive selection of time step. Experiment results demonstrate the superiority of the proposed SDFEN in effectiveness and efficiency.

----

## [2748] A SAT Solver and Computer Algebra Attack on the Minimum Kochen-Specker Problem (Student Abstract)

**Authors**: *Zhengyu Li, Curtis Bright, Vijay Ganesh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30472](https://doi.org/10.1609/aaai.v38i21.30472)

**Abstract**:

The problem of finding the minimum three-dimensional Kochen–Specker (KS) vector system, an important problem in quantum foundations, has remained open for over 55 years. We present a new method to address this problem based on a combination of a Boolean satisfiability (SAT) solver and a computer algebra system (CAS). Our approach improved the lower bound on the size of a KS system from 22 to 24. More importantly, we provide the first computer-verifiable proof certificate of a lower bound to the KS problem with a proof size of 41.6 TiB for order 23. The efficiency is due to the powerful combination of SAT solvers and CAS-based orderly generation.

----

## [2749] Enhance Diversified Top-k MaxSAT Solving by Incorporating New Strategy for Generating Diversified Initial Assignments (Student Abstract)

**Authors**: *Jiaxin Liang, Junping Zhou, Minghao Yin*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30473](https://doi.org/10.1609/aaai.v38i21.30473)

**Abstract**:

The Diversified Top-k MaxSAT (DTKMS) problem is an extension of MaxSAT. The objective of DTKMS is to find k feasible assignments of a given formula, such that each assignment satisfies all hard clauses and the k assignments together satisfy the maximum number of soft clauses. This paper presents a local search algorithm, DTKMS-DIA, which incorporates a new approach to generating initial assignments. Experimental results indicate that DTKMS-DIA can achieve attractive performance on 826 instances compared with state-of-the-art solvers.

----

## [2750] DNIT: Enhancing Day-Night Image-to-Image Translation through Fine-Grained Feature Handling (Student Abstract)

**Authors**: *Hanyue Liu, Haonan Cheng, Long Ye*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30474](https://doi.org/10.1609/aaai.v38i21.30474)

**Abstract**:

Existing image-to-image translation methods perform less satisfactorily in the "day-night" domain due to insufficient scene feature study. To address this problem, we propose DNIT, which performs fine-grained handling of features by a nighttime image preprocessing (NIP) module and an edge fusion detection (EFD) module. The NIP module enhances brightness while minimizing noise, facilitating extracting content and style features. Meanwhile, the EFD module utilizes two types of edge images as additional constraints to optimize the generator. Experimental results show that we can generate more realistic and higher-quality images compared to other methods, proving the effectiveness of our DNIT.

----

## [2751] icsPLMs: Exploring Pre-trained Language Models in Intelligent Customer Service (Student Abstract)

**Authors**: *Shixuan Liu, Chao Wang, Shuangyong Song*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30475](https://doi.org/10.1609/aaai.v38i21.30475)

**Abstract**:

Pre-trained language models have shown their high performance of text processing in intelligent customer service platforms. However, these models do not leverage domain specific information. In this paper, we propose icsPLMs optimized for intelligent customer service on both word and sentence levels. Our experimental results represent that using targeted strategies can further improve the performance of pre-trained language models in this field.

----

## [2752] Fair Representation Learning with Maximum Mean Discrepancy Distance Constraint (Student Abstract)

**Authors**: *Alexandru Lopotenco, Ian Tong Pan, Jack Zhang, Guan Xiong Qiao*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30476](https://doi.org/10.1609/aaai.v38i21.30476)

**Abstract**:

Unsupervised learning methods such as principal component analysis (PCA), t-distributed stochastic neighbor embedding (t-SNE), and autoencoding are regularly used in dimensionality reduction within the statistical learning scene. However, despite a pivot toward fairness and explainability in machine learning over the past few years, there have been few rigorous attempts toward a generalized framework of fair and explainable representation learning. Our paper explores the possibility of such a framework that leverages maximum mean discrepancy to remove information derived from a protected class from generated representations. For the optimization, we introduce a binary search component to optimize the Lagrangian coefficients. We present rigorous mathematical analysis and experimental results of our framework applied to t-SNE.

----

## [2753] Optimizing Recall in Deep Graph Hashing Framework for Item Retrieval (Student Abstract)

**Authors**: *Fangyuan Luo, Jun Wu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30477](https://doi.org/10.1609/aaai.v38i21.30477)

**Abstract**:

Hashing-based recommendation (HR) methods, whose core idea is mapping users and items into hamming space, are common practice to improve item retrieval efficiency. However, existing HR fails to align optimization objective (i.e., Bayesian Personalized Ranking) and evaluation metric (i.e., Recall), leading to suboptimal performance. In this paper, we propose a smooth recall loss (termed as SRLoss), which targets Recall as the optimization objective. Due to the existence of discrete constraints, the optimization problem is NP-hard. To this end, we propose an approximation-adjustable gradient estimator to solve our problem. Experimental Results demonstrate the effectiveness of our proposed method.

----

## [2754] Research of Event Reconstruct Based on Multi-View Contrastive Learning (Student Abstract)

**Authors**: *Yuefeng Ma, Zhongchao He, Shumei Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30478](https://doi.org/10.1609/aaai.v38i21.30478)

**Abstract**:

The proliferation of social media exacerbates information fragmentation, posing challenges to understanding public events. We address the problem of event reconstruction with a novel Multi-view Contrast Event Reconstruction (MCER) model. MCER maximizes feature dissimilarity between different views of the same event using contrastive learning, while minimizing mutual information between distinct events. This aggregates fragmented views to reconstruct comprehensive event representations. MCER employs momentum and weight-sharing encoders in a three-tower architecture with supervised contrastive loss for multi-view representation learning. Due to the scarcity of multi-view public datasets, we construct a new Mul-view-data benchmark.Experiments demonstrate MCER’s superior performance on public data and our Mul-view-data, significantly outperforming selfsupervised methods by incorporating supervised contrastive techniques. MCER advances multi-view representation learning to counter information fragmentation and enable robust event understanding.

----

## [2755] Graph Clustering Methods Derived from Column Subset Selection (Student Abstract)

**Authors**: *Wei Mao, Guihong Wan, Haim Schweitzer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30479](https://doi.org/10.1609/aaai.v38i21.30479)

**Abstract**:

Spectral clustering is a powerful clustering technique. It leverages the spectral properties of graphs to partition data points into meaningful clusters. The most common criterion for evaluating multi-way spectral clustering is NCut. Column Subset Selection is an important optimization technique in the domain of feature selection and dimension reduction which aims to identify a subset of columns of a given data matrix that can be used to approximate the entire matrix. We show that column subset selection can be used to compute spectral clustering and use this to obtain new graph clustering algorithms.

----

## [2756] Fast and Knowledge-Free Deep Learning for General Game Playing (Student Abstract)

**Authors**: *Michal Maras, Michal Kepa, Jakub Kowalski, Marek Szykula*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30480](https://doi.org/10.1609/aaai.v38i21.30480)

**Abstract**:

We develop a method of adapting the AlphaZero model to General Game Playing (GGP) that focuses on faster model generation and requires less knowledge to be extracted from the game rules. The dataset generation uses MCTS playing instead of self-play; only the value network is used, and attention layers replace the convolutional ones. This allows us to abandon any assumptions about the action space and board topology. We implement the method within the Regular Boardgames GGP system and show that we can build models outperforming the UCT baseline for most games efficiently.

----

## [2757] Towards Robustness to Natural Variations and Distribution Shift (Student Abstract)

**Authors**: *Josué Martínez-Martínez, Olivia Brown, Rajmonda Caceres*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30481](https://doi.org/10.1609/aaai.v38i21.30481)

**Abstract**:

This research focuses on improving the robustness of machine learning systems to natural variations and distribution shifts. A design trade space is presented, and various methods are compared, including adversarial training, data augmentation techniques, and novel approaches inspired by model-based robust optimization formulations.

----

## [2758] Topological and Node Noise Filtering on 3D Meshes Using Graph Neural Networks (Student Abstract)

**Authors**: *Vladimir Mashurov, Natalia Semenova*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30482](https://doi.org/10.1609/aaai.v38i21.30482)

**Abstract**:

Topological and node noise filtration are typically considered separately. Graph Neural Networks (GNN) are commonly used for node noise filtration, as they offer high efficiency and low exploitation costs. This paper explores the solution of joint node and topological noise filtration through the use of graph neural networks. Since treating a 3D mesh as a graph is challenging, an indicator function grid representation is employed as input for GNNs to perform the joint filtering. The resulting machine learning model is inspired by point cloud to mesh reconstruction algorithms and demonstrates low computational requirements during inference, producing successful results for smooth, watertight 3D models.

----

## [2759] Enhanced Optical Character Recognition by Optical Sensor Combined with BERT and Cosine Similarity Scoring (Student Abstract)

**Authors**: *Woohyeon Moon, Sarvar Hussain Nengroo, Taeyoung Kim, Jihui Lee, Seungah Son, Dongsoo Har*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30483](https://doi.org/10.1609/aaai.v38i21.30483)

**Abstract**:

Optical character recognition(OCR) is the technology to identify text characters embedded within images. Conventional OCR models exhibit performance degradation when performing with noisy images. To solve this problem, we propose a novel model, which combines computer vision using optical sensor with natural language processing by bidirectional encoder representations from transformers(BERT) and cosine similarity scoring. The proposed model uses a confidence rate to determine whether to utilize optical sensor alone or BERT/cosine similarity scoring combined with the optical sensor. Experimental results show that the proposed model outperforms approximately 4.34 times better than the conventional OCR.

----

## [2760] Several Stories about High-Multiplicity EFx Allocation (Student Abstract)

**Authors**: *Nikita Morozov, Artur Ignatiev, Yuriy Dementiev*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30484](https://doi.org/10.1609/aaai.v38i21.30484)

**Abstract**:

Fair division is a topic that has significant social and industrial value. In this work, we study allocations that simultaneously satisfy definitions of fairness and efficiency: EFx and PO. First, we prove that the problem of finding such allocations is NP-hard for two agents. Then, we propose a concept for an ILP-based solving algorithm, the running time of which depends on the number of EFx allocations. We generate input data and analyze algorithm's running time based on the results obtained.

----

## [2761] An Empirical Study of Distributed Deep Learning Training on Edge (Student Abstract)

**Authors**: *Christine Mwase, Albert Njoroge Kahira, Zhuo Zou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30485](https://doi.org/10.1609/aaai.v38i21.30485)

**Abstract**:

Deep learning (DL), despite its success in various fields, remains expensive and inaccessible to many due to its need for powerful supercomputing and high-end GPUs. This study explores alternative computing infrastructure and methods for distributed DL on low-energy, low-cost devices. We experiment on Raspberry Pi 4 devices with ARM Cortex-A72 processors and train a ResNet-18 model on the CIFAR-10 dataset. Our findings reveal limitations and opportunities for future optimizations, paving the way for a DL toolset for low-energy edge devices.

----

## [2762] SkillCLIP: Skill Aware Modality Fusion Visual Question Answering (Student Abstract)

**Authors**: *Atharva Naik, Yash Parag Butala, Navaneethan Vaikunthan, Raghav Kapoor*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30486](https://doi.org/10.1609/aaai.v38i21.30486)

**Abstract**:

When humans are posed with a difficult problem, they often approach it by identifying key skills, honing them, and finally effectively combining them. We propose a novel method and apply it for the VizWiz VQA task to predict the visual skills needed to answer a question, and leverage expert modules to produce intermediary outputs and fuse them in a skill-aware manner. Unlike prior works in visual question-answering (VQA) that use intermediate outputs such as detected objects and Optical Character Recognition (OCR), our approach explicitly guides the model with a skill embedding on what to focus on. While our results show that using skill-aware fusion outperforms skill-unaware models for only a subset of questions, we believe our results provide interesting directions for future work. We also release our code, model, and illustrative demonstrations for future research purposes.

----

## [2763] MaxEnt Loss: Calibrating Graph Neural Networks under Out-of-Distribution Shift (Student Abstract)

**Authors**: *Dexter Neo*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30487](https://doi.org/10.1609/aaai.v38i21.30487)

**Abstract**:

We present a new, simple and effective loss function for calibrating graph neural networks (GNNs). Miscalibration is the problem whereby a model's probabilities does not reflect it's correctness, making it difficult and possibly dangerous for real-world deployment. We compare our method against other baselines on a novel ID and OOD graph form of the Celeb-A faces dataset. Our findings show that our method improves calibration for GNNs, which are not immune to miscalibration in-distribution (ID) and out-of-distribution (OOD). Our code is available for review at https://github.com/dexterdley/CS6208/tree/main/Project.

----

## [2764] Welfare Maximization in Perpetual Voting (Student Abstract)

**Authors**: *Tzeh Yuan Neoh, Nicholas Teh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30488](https://doi.org/10.1609/aaai.v38i21.30488)

**Abstract**:

We study the computational problems associated with maximizing various welfare objectives—namely utilitarian welfare, egalitarian welfare, and Nash welfare—in perpetual voting, a sequential collective decision-making framework. Prior work look into notions of fairness over time and study extensions of single-round voting rules to the multi-round setting.
We show that while a utilitarian-welfare maximizing outcome can be computed efficiently, an outcome that maximizes egalitarian or Nash welfare is computationally intractable, even in the case of two candidates. We complement this by showing that maximizing egalitarian welfare is fixed-parameter tractable in the number of agents, and maximizing egalitarian or Nash welfare is W[2]-hard and slicewise polynomial in the number of timesteps. We also provide an approximation algorithm for maximizing egalitarian welfare and study strategyproofness with respect to these welfare objectives. Finally, we show that a simple greedy algorithm can achieve approximate proportionality in this setting.

----

## [2765] When Sparse Graph Representation Learning Falls into Domain Shift: Data Augmentation for Cross-Domain Graph Meta-Learning (Student Abstract)

**Authors**: *Simin Niu, Xun Liang, Sensen Zhang, Shichao Song, Xuan Zhang, Xiaoping Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30489](https://doi.org/10.1609/aaai.v38i21.30489)

**Abstract**:

Cross-domain Graph Meta-learning (CGML) has shown its promise, where meta-knowledge is extracted from few-shot graph data in multiple relevant but distinct domains. However, several recent efforts assume target data available, which commonly does not established in practice. In this paper, we devise a novel Cross-domain Data Augmentation for Graph Meta-Learning (CDA-GML), which incorporates the superiorities of CGML and Data Augmentation， has addressed intractable shortcomings of label sparsity, domain shift, and the absence of target data simultaneously. Specifically, our method simulates instance-level and task-level domain shift to alleviate the cross-domain generalization issue in conventional graph meta-learning. Experiments show that our method outperforms the existing state-of-the-art methods.

----

## [2766] Target-Free Domain Adaptation through Cross-Adaptation (Student Abstract)

**Authors**: *Aleksander Obuchowski, Barbara Klaudel, Piotr Frackowski, Sebastian Krajna, Wasyl Badyra, Michal Czubenko, Zdzislaw Kowalczuk*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30490](https://doi.org/10.1609/aaai.v38i21.30490)

**Abstract**:

The population characteristics of the datasets related to the same task may vary significantly and merging them may harm performance. In this paper, we propose a novel method of domain adaptation called "cross-adaptation". It allows for implicit adaptation to the target domain without the need for any labeled examples across this domain. We test our approach on 9 datasets for SARS-CoV-2 detection from complete blood count from different hospitals around the world. Results show that our solution is universal with respect to various classification algorithms and allows for up to a 10pp increase in F1 score on average.

----

## [2767] Large Language Models as Planning Domain Generators (Student Abstract)

**Authors**: *James T. Oswald, Kavitha Srinivas, Harsha Kokel, Junkyu Lee, Michael Katz, Shirin Sohrabi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30491](https://doi.org/10.1609/aaai.v38i21.30491)

**Abstract**:

The creation of planning models, and in particular domain
models, is among the last bastions of tasks that require exten-
sive manual labor in AI planning; it is desirable to simplify
this process for the sake of making planning more accessi-
ble. To this end, we investigate whether large language mod-
els (LLMs) can be used to generate planning domain models
from textual descriptions. We propose a novel task for this
as well as a means of automated evaluation for generated do-
mains by comparing the sets of plans for domain instances.
Finally, we perform an empirical analysis of 7 large language
models, including coding and chat models across 9 different
planning domains. Our results show that LLMs, particularly
larger ones, exhibit some level of proficiency in generating
correct planning domains from natural language descriptions

----

## [2768] A Wireframe-Based Approach for Classifying and Acquiring Proficiency in the American Sign Language (Student Abstract)

**Authors**: *Dylan Pallickara, Sarath Sreedharan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30492](https://doi.org/10.1609/aaai.v38i21.30492)

**Abstract**:

We describe our methodology for classifying ASL (American Sign Language) gestures. Rather than operate directly on raw images of hand gestures, we extract coor-dinates and render wireframes from individual images to construct a curated training dataset. This dataset is then used in a classifier that is memory efficient and provides effective performance (94% accuracy). Because we con-struct wireframes that contain information about several angles in the joints that comprise hands, our methodolo-gy is amenable to training those interested in learning ASL by identifying targeted errors in their hand gestures.

----

## [2769] Neuroevolution of a Multi-Generator GAN (Student Abstract)

**Authors**: *Suraj Pandey*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30493](https://doi.org/10.1609/aaai.v38i21.30493)

**Abstract**:

Evolutionary Algorithms (EA) have been leveraged to tackle the challenges faced while using GANs such as mode collapse, vanishing gradient, latent space search, etc. However, the existing techniques of using EA with GANs operate backpropagation and EA in isolation from each other, leaving ample room for further exploration. This paper creates a collaborative bridge between EA and GANs by exploring a neuroevolution method for utilising both EA and backpropagation-based optimisation, simultaneously, for a multi-generator GAN architecture. Experiments conducted using a standard dataset with variants of the proposed method highlight the towering impact of each of the components involved in the proposed method.

----

## [2770] Graph Anomaly Detection with Diffusion Model-Based Graph Enhancement (Student Abstract)

**Authors**: *Shikang Pang, Chunjing Xiao, Wenxin Tai, Zhangtao Cheng, Fan Zhou*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30494](https://doi.org/10.1609/aaai.v38i21.30494)

**Abstract**:

Graph anomaly detection has gained significant research interest across various domains. Due to the lack of labeled data, contrastive learning has been applied in detecting anomalies and various scales of contrastive strategies have been initiated. However, these methods might force two instances (e.g., node-level and subgraph-level representations) with different category labels to be consistent during model training, which can adversely impact the model robustness. To tackle this problem, we present a novel contrastive learning framework with the Diffusion model-based graph Enhancement module for Graph Anomaly Detection, DEGAD. In this framework, we design a diffusion model-based graph enhancement module to manipulate neighbors to generate enhanced graphs, which can efficiently alleviate the inconsistent problem. Further, based on the enhanced graphs, we present a multi-scale contrastive module to discriminate anomalies. Experimental results demonstrate the superiority of our model.

----

## [2771] Virtual Action Actor-Critic Framework for Exploration (Student Abstract)

**Authors**: *Bumgeun Park, Taeyoung Kim, Quoc-Vinh Lai-Dang, Dongsoo Har*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30495](https://doi.org/10.1609/aaai.v38i21.30495)

**Abstract**:

Efficient exploration for an agent is challenging in reinforcement learning (RL). In this paper, a novel actor-critic framework namely virtual action actor-critic (VAAC), is proposed to address the challenge of efficient exploration in RL. This work is inspired by humans' ability to imagine the potential outcomes of their actions without actually taking them. In order to emulate this ability, VAAC introduces a new actor called virtual actor (VA), alongside the conventional actor-critic framework. Unlike the conventional actor, the VA takes the virtual action to anticipate the next state without interacting with the environment. With the virtual policy following a Gaussian distribution, the VA is trained to maximize the anticipated novelty of the subsequent state resulting from a virtual action. If any next state resulting from available actions does not exhibit high anticipated novelty, training the VA leads to an increase in the virtual policy entropy. Hence, high virtual policy entropy represents that there is no room for exploration. The proposed VAAC aims to maximize a modified Q function, which combines cumulative rewards and the negative sum of virtual policy entropy. Experimental results show that the VAAC improves the exploration performance compared to existing algorithms.

----

## [2772] Skip-GANomaly++: Skip Connections and Residual Blocks for Anomaly Detection (Student Abstract)

**Authors**: *Juneyoung Park, Jae-Ryung Hong, Min-Hye Kim, Tae-Joon Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30496](https://doi.org/10.1609/aaai.v38i21.30496)

**Abstract**:

Anomaly detection is a critical task across various domains. Fundamentally, anomaly detection models offer methods to identify unusual patterns that do not align with expected behaviors. Notably, in the medical field, detecting anomalies in medical imagery or biometrics can facilitate early diagnosis of diseases. Consequently, we propose the Skip-GANomaly++ model, an enhanced and more efficient version of the conventional anomaly detection models. The proposed model's performance was evaluated through comparative experiments. Experimental results demonstrated superior performance across most classes compared to the previous models.

----

## [2773] QuickRender: A Photorealistic Procedurally Generated Dataset with Applications to Super Resolution (Student Abstract)

**Authors**: *Morgan Payette, Charlotte Curtis*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30497](https://doi.org/10.1609/aaai.v38i21.30497)

**Abstract**:

Rendering of complex scenes from software such as Blender is time consuming,  but corresponding auxiliary data such as depth or object segmentation maps are relatively fast to generate. The auxiliary data also provides a wealth of information for tasks such as optical flow prediction.

In this paper we present the QuickRender dataset, a collection of procedurally generated scenes rendered into over 5,000 sequential image triplets along with accompanying auxiliary data. The goal of this dataset is to provide a diversity of scenes and motion while maintaining realistic behaviours. A sample application using this dataset to perform single image super resolution is also presented.

The dataset and related source code can be found at https://github.com/MP-mtroyal/MetaSRGAN.

----

## [2774] Knowledge Transfer via Compact Model in Federated Learning (Student Abstract)

**Authors**: *Jiaming Pei, Wei Li, Lukun Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30498](https://doi.org/10.1609/aaai.v38i21.30498)

**Abstract**:

Communication overhead remains a significant challenge in federated learning due to frequent global model updates. Essentially, the update of the global model can be viewed as knowledge transfer. We aim to transfer more knowledge through a compact model while reducing communication overhead. In our study, we introduce a federated learning framework where clients pre-train large models locally and the server initializes a compact model to communicate. This compact model should be light in size but still have enough knowledge to refine the global model effectively. We facilitate the knowledge transfer from local to global models based on pre-training outcomes. Our experiments show that our approach significantly reduce communication overhead without sacrificing accuracy.

----

## [2775] HyperCube: Implicit Field Representations of Voxelized 3D Models (Student Abstract)

**Authors**: *Magdalena Proszewska, Marcin Mazur, Tomasz Trzcinski, Przemyslaw Spurek*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30499](https://doi.org/10.1609/aaai.v38i21.30499)

**Abstract**:

Implicit field representations offer an effective way of generating 3D object shapes. They leverage an implicit decoder (IM-NET) trained to take a 3D point coordinate concatenated with a shape encoding and to output a value indicating whether the point is outside the shape. This approach enables the efficient rendering of visually plausible objects but also has some significant limitations, resulting in a cumbersome training procedure and empty spaces within the rendered mesh. In this paper, we introduce a new HyperCube architecture based on interval arithmetic that enables direct processing of 3D voxels, trained using a hypernetwork paradigm to enforce model convergence. The code is available at https://github.com/mproszewska/hypercube.

----

## [2776] Learning Random Noise Salient Feature Fusion Siamese Network for Low-Resolution Object Tracking (Student Abstract)

**Authors**: *Md Maklachur Rahman, Tracy Hammond*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30500](https://doi.org/10.1609/aaai.v38i21.30500)

**Abstract**:

Despite Siamese trackers’ substantial potential, they offer sub-optimal tracking performance in low-resolution (LR) contexts. We introduce a Random Noise Salient Feature Fusion Learning Network to address this issue. This method integrates random noise-infused feature maps into a similaritylearning matching model. This integration acts as an effective regularization technique, enhancing the network’s generalization capabilities in LR environments. Additionally, by integrating attention mechanisms, we enhance the discriminative ability of the network, assigning more weights to important features. This directs the network’s focus toward the most salient regions of the feature map, ensuring improved accuracy without a significant increase in parameter overhead, and maintaining a high operating speed. To validate the effectiveness of our method, we performed qualitative and quantitative comparisons with state-of-the-art (SOTA) trackers.

----

## [2777] DQSSA: A Quantum-Inspired Solution for Maximizing Influence in Online Social Networks (Student Abstract)

**Authors**: *Aryaman Rao, Parth Singh, Dinesh Kumar Vishwakarma, Mukesh Prasad*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30501](https://doi.org/10.1609/aaai.v38i21.30501)

**Abstract**:

Influence Maximization is the task of selecting optimal nodes maximising the influence spread in social networks. This study proposes a Discretized Quantum-based Salp Swarm Algorithm (DQSSA) for optimizing influence diffusion in social networks. By discretizing meta-heuristic algorithms and infusing them with quantum-inspired enhancements, we address issues like premature convergence and low efficacy. The proposed method, guided by quantum principles, offers a promising solution for Influence Maximisation. Experiments on four real-world datasets reveal DQSSA's superior performance as compared to established cutting-edge algorithms.

----

## [2778] Well-Written Knowledge Graphs: Most Effective RDF Syntaxes for Triple Linearization in End-to-End Extraction of Relations from Texts (Student Abstract)

**Authors**: *Célian Ringwald, Fabien Gandon, Catherine Faron, Franck Michel, Hanna Abi Akl*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30502](https://doi.org/10.1609/aaai.v38i21.30502)

**Abstract**:

Seq-to-seq generative models recently gained attention for solving the relation extraction task. By approaching this problem as an end-to-end task, they surpassed encoder-based-only models. Little research investigated the effects of the output syntaxes on the training process of these models. Moreover, a limited number of approaches were proposed for extracting ready-to-load knowledge graphs following the RDF standard. In this paper, we consider that a set of triples can be linearized in many different ways, and we evaluate the combined effect of the size of the language models and different RDF syntaxes on the task of relation extraction from Wikipedia abstracts.

----

## [2779] FAIR-FER: A Latent Alignment Approach for Mitigating Bias in Facial Expression Recognition (Student Abstract)

**Authors**: *Syed Sameen Ahmad Rizvi, Aryan Seth, Pratik Narang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30503](https://doi.org/10.1609/aaai.v38i21.30503)

**Abstract**:

Facial Expression Recognition (FER) is an extensively explored research problem in the domain of computer vision and artificial intelligence. FER, a supervised learning problem, requires significant training data representative of multiple socio-cultural demographic attributes. However, most of the FER dataset consists of images annotated by humans, which propagates individual and demographic biases. This work attempts to mitigate this bias using representation learning based on latent spaces, thereby increasing a deep learning model's fairness and overall accuracy.

----

## [2780] Partially Observable Hierarchical Reinforcement Learning with AI Planning (Student Abstract)

**Authors**: *Brandon Rozek, Junkyu Lee, Harsha Kokel, Michael Katz, Shirin Sohrabi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30504](https://doi.org/10.1609/aaai.v38i21.30504)

**Abstract**:

Partially observable Markov decision processes (POMDPs) challenge reinforcement learning agents due to incomplete knowledge of the environment. Even assuming monotonicity in uncertainty, it is difficult for an agent to know how and when to stop exploring for a given task. In this abstract, we discuss how to use hierarchical reinforcement learning (HRL) and AI Planning (AIP) to improve exploration when the agent knows possible valuations of unknown predicates and how to discover them. By encoding the uncertainty in an abstract planning model, the agent can derive a high-level plan which is then used to decompose the overall POMDP into a tree of semi-POMDPs for training. We evaluate our agent's performance on the MiniGrid domain and show how guided exploration may improve agent performance.

----

## [2781] Finetuning LLMs for Automatic Concept to TTI Prompt Generation (Student Abstract)

**Authors**: *Jeremy Rutter, Maneesh Reddy Chamakura, Justin Delgado, Gene Louis Kim*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30505](https://doi.org/10.1609/aaai.v38i21.30505)

**Abstract**:

Our work explores bridging the gap between large language models and text-to-image models to create a tool for quickly and easily generating high quality images from a given concept. In our experiments we successfully improved image quality with only a preliminary utilization of the available resources for finetuning.

----

## [2782] Instance-Wise Laplace Mechanism via Deep Reinforcement Learning (Student Abstract)

**Authors**: *Sehyun Ryu, Hosung Joo, Jonggyu Jang, Hyun Jong Yang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30506](https://doi.org/10.1609/aaai.v38i21.30506)

**Abstract**:

Recent research has shown a growing interest in per-instance differential privacy (pDP), highlighting the fact that each data instance within a dataset may incur distinct levels of privacy loss.
However, conventional additive noise mechanisms apply identical noise to all query outputs, thereby deteriorating data statistics.
In this study, we propose an instance-wise Laplace mechanism, which adds non-identical Laplace noises to the query output for each data instance.
A challenge arises from the complex interaction of additive noise, where the noise introduced to individual instances impacts the pDP of other instances, adding complexity and resilience to straightforward solutions.
To tackle this problem, we introduce an instance-wise Laplace mechanism algorithm via deep reinforcement learning and validate its ability to better preserve data statistics on a real dataset, compared to the original Laplace mechanism.

----

## [2783] Frequency Oracle for Sensitive Data Monitoring (Student Abstract)

**Authors**: *Richard Sances, Olivera Kotevska, Paul Laiu*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30507](https://doi.org/10.1609/aaai.v38i21.30507)

**Abstract**:

As data privacy issues grow, finding the best privacy preservation algorithm for each situation is increasingly essential. This research has focused on understanding the frequency oracles (FO) privacy preservation algorithms. FO conduct the frequency estimation of any value in the domain. The aim is to explore how each can be best used and recommend which one to use with which data type. We experimented with different data scenarios and federated learning settings. Results showed clear guidance on when to use a specific algorithm.

----

## [2784] Adapting Animal Models to Assess Sufficiency of Fluid Resuscitation in Humans (Student Abstract)

**Authors**: *Ryan Schuerkamp, Xinyu Li, Brian Kunzer, Leonard S. Weiss, Hernando Gómez, Francis X. Guyette, Michael R. Pinsky, Artur Dubrawski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30508](https://doi.org/10.1609/aaai.v38i21.30508)

**Abstract**:

Fluid resuscitation is an initial treatment frequently employed to treat shock, restore lost blood, protect tissues from injury, and prevent organ dysfunction in critically ill patients. However, it is not without risk (e.g., overly aggressive resuscitation may cause organ damage and even death). We leverage machine learning models trained to assess sufficiency of resuscitation in laboratory animals subjected to induced hemorrhage and transfer them to use with human trauma patients. Our key takeaway is that animal experiments and models can inform human healthcare, especially when human data is limited or when collecting relevant human data via potentially harmful protocols is unfeasible.

----

## [2785] Rider Posture-Based Continuous Authentication with Few-Shot Learning for Mobility Scooters (Student Abstract)

**Authors**: *Devan Shah, Ruoqi Huang, Tingting Chen, Murtuza Jadliwala*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30509](https://doi.org/10.1609/aaai.v38i21.30509)

**Abstract**:

Current practice of mobility scooter user authentication using physical keys and traditional password-based one-time security mechanisms cannot meet the needs of many mobility scooter riders, especially senior citizens having issues in recalling memory. Now seamless authentication approaches are needed to provide ongoing protection for mobility scooters against takeovers and unauthorized access. Existing continuous authentication techniques do not work well in a mobility scooter setting due to issues such as user comfort, deployment cost and enrollment time, among others. In that direction, our contributions in this research effort are two-fold: (i) we propose a novel system that incorporates advances in few-shot learning, hierarchical processing, and contextual embedding to establish continuous authentication for mobility scooter riders using only posture data. This security system, trained on data collected from real mobility scooter riders, demonstrates quick enrollment and easy deployability, while successfully serving as an unobtrusive first layer of security. (ii) we provide to the research community the largest publicly available repository of mobility scooter riders' body key-points data to enable further research in this direction.

----

## [2786] Coordination of Emergent Demand Changes via Value-Based Negotiation for Supply Chain Management (Student Abstract)

**Authors**: *Takumu Shimizu, Ryota Higa, Katsuhide Fujita, Shinji Nakadai*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30510](https://doi.org/10.1609/aaai.v38i21.30510)

**Abstract**:

We propose an automated negotiation for a reinforcement learning agent to adapt the agent to unexpected situations such as demand changes in supply chain management (SCM).
Existing studies that consider reinforcement learning and SCM assume a centralized environment where the coordination of chain components is hierarchical rather than through negotiations between agents.
This study focused on a negotiation agent that considered the value function of reinforcement learning for SCM as its utility function in automated negotiation.
We demonstrated that the proposed approach could avoid inventory shortages under increased demand requests from the terminal customer.

----

## [2787] Faithful Trip Recommender Using Diffusion Guidance (Student Abstract)

**Authors**: *Wenzheng Shu, Yanlong Huang, Wenxin Tai, Zhangtao Cheng, Bei Hui, Goce Trajcevski*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30511](https://doi.org/10.1609/aaai.v38i21.30511)

**Abstract**:

Trip recommendation aims to plan user’s travel based on their specified preferences. Traditional heuristic and statistical approaches often fail to capture the intricate nuances of user intentions, leading to subpar performance. Recent deep-learning methods show attractive accuracy but struggle to generate faithful trajectories that match user intentions. In this work, we propose a DDPM-based incremental knowledge injection module to ensure the faithfulness of the generated trajectories. Experiments on two datasets verify the effectiveness of our approach.

----

## [2788] Diverse Yet Biased: Towards Mitigating Biases in Generative AI (Student Abstract)

**Authors**: *Akshit Singh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30512](https://doi.org/10.1609/aaai.v38i21.30512)

**Abstract**:

Generative Artificial Intelligence (AI) has garnered significant attention for its remarkable ability to generate text, images, and other forms of content. However, an inherent and increasingly concerning issue within generative AI systems is bias. These AI models often exhibit an Anglo-centric bias and tend to overlook the importance of diversity. This can be attributed to their training on extensive datasets sourced from the internet, which inevitably inherit the biases present in those data sources. Employing these datasets leads to AI-generated content that mirrors and perpetuates existing biases, encompassing various aspects such as gender, ethnic and cultural stereotypes. Addressing bias in generative AI is a complex challenge that necessitates substantial efforts. In order to tackle this issue, we propose a methodology for constructing moderately sized datasets with a social inclination. These datasets can be employed to rectify existing imbalances in datasets or to train models to generate socially inclusive material. Additionally, we present preliminary findings derived from training our model on these socially inclined datasets.

----

## [2789] Confidence Is All You Need for MI Attacks (Student Abstract)

**Authors**: *Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30513](https://doi.org/10.1609/aaai.v38i21.30513)

**Abstract**:

In this evolving era of machine learning security, membership inference attacks have emerged as a potent threat to the confidentiality of sensitive data. In this attack, adversaries aim to determine whether a particular point was used during the training of a target model. This paper proposes a new method to gauge a data point’s membership in a model’s training set. Instead of correlating loss with membership, as is traditionally done, we have leveraged the fact that training examples generally exhibit higher confidence values when classified into their actual class. During training, the model is essentially being ’fit’ to the training data and might face particular difficulties in generalization to unseen data. This asymmetry leads to the model achieving higher confidence on the training data as it exploits the specific patterns and noise present in the training data. Our proposed approach leverages the confidence values generated by the machine-learning model. These confidence values provide a probabilistic measure of the model’s certainty in its predictions and can further be used to infer the membership of a given data point. Additionally, we also introduce another variant of our method that allows us to carry out this attack without knowing the ground truth(true class) of a given data point, thus offering an edge over existing label-dependent attack methods.

----

## [2790] Investigation into Training Dynamics of Learned Optimizers (Student Abstract)

**Authors**: *Jan Sobotka, Petr Simánek*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30514](https://doi.org/10.1609/aaai.v38i21.30514)

**Abstract**:

Modern machine learning heavily relies on optimization, and as deep learning models grow more complex and data-hungry, the search for efficient learning becomes crucial. Learned optimizers disrupt traditional handcrafted methods such as SGD and Adam by learning the optimization strategy itself, potentially speeding up training. However, the learned optimizers' dynamics are still not well understood. To remedy this, our work explores their optimization trajectories from the perspective of network architecture symmetries and proposed parameter update distributions.

----

## [2791] Learning to Build Solutions in Stochastic Matching Problems Using Flows (Student Abstract)

**Authors**: *William St-Arnaud, Margarida Carvalho, Golnoosh Farnadi*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30515](https://doi.org/10.1609/aaai.v38i21.30515)

**Abstract**:

Generative Flow Networks, known as GFlowNets, have been introduced in recent times, presenting an exciting possibility for neural networks to model distributions across various data structures. In this paper, we broaden their applicability to encompass scenarios where the data structures are optimal solutions of a combinatorial problem. Concretely, we propose the use of GFlowNets to learn the distribution of optimal solutions for kidney exchange problems (KEPs), a generalized form of matching problems involving cycles.

----

## [2792] DDViT: Double-Level Fusion Domain Adapter Vision Transformer (Student Abstract)

**Authors**: *Linpeng Sun, Victor S. Sheng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30516](https://doi.org/10.1609/aaai.v38i21.30516)

**Abstract**:

With the help of Vision transformers (ViTs), medical image segmentation was able to achieve outstanding performance. In particular, they overcome the limitation of convolutional neural networks (CNNs) which rely on local receptive fields. ViTs use self-attention mechanisms to consider relationships between all image pixels or patches simultaneously. However, they require large datasets for training and did not perform well on capturing low-level features. To that end, we propose DDViT, a novel ViT model that unites a CNN to alleviate data-hunger for medical image segmentation with two multi-scale feature representations. Significantly, our approach incorporates a ViT with a plug-in domain adapter (DA) with Double-Level Fusion (DLF) technique, complemented by a mutual knowledge distillation paradigm, facilitating the seamless exchange of knowledge between a universal network and specialized domain-specific network branches. The DLF framework plays a pivotal role in our encoder-decoder architecture, combining the innovation of the TransFuse module with a robust CNN-based encoder. Extensive experimentation across diverse medical image segmentation datasets underscores the remarkable efficacy of DDViT when compared to alternative approaches based on CNNs and Transformer-based models.

----

## [2793] Evaluation of Large Language Models on Code Obfuscation (Student Abstract)

**Authors**: *Adrian Swindle, Derrick McNealy, Giri P. Krishnan, Ramyaa Ramyaa*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30517](https://doi.org/10.1609/aaai.v38i21.30517)

**Abstract**:

Obfuscation intends to decrease interpretability of code and identification of code behavior. Large Language Models(LLMs) have been proposed for code synthesis and code analysis. This paper attempts to understand how well LLMs can analyse code and identify code behavior. Specifically, this paper systematically evaluates several LLMs’ capabilities to detect obfuscated code and identify behavior across a variety of obfuscation techniques with varying levels of complexity. LLMs proved to be better at detecting obfuscations that changed identifiers, even to misleading ones, compared to obfuscations involving code insertions (unused variables, as well as variables that replace constants with expressions that evaluate to those constants). Hardest to detect were obfuscations that layered multiple simple transformations. For these, only 20-40% of the LLMs’ responses were correct. Adding misleading documentation was also successful in misleading LLMs. We provide all our code to replicate results at https://github.com/SwindleA/LLMCodeObfuscation. Overall, our results suggest a gap in LLMs’ ability to understand code.

----

## [2794] Graph Anomaly Detection via Prototype-Aware Label Propagation (Student Abstract)

**Authors**: *Hui Tang, Xun Liang, Sensen Zhang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30518](https://doi.org/10.1609/aaai.v38i21.30518)

**Abstract**:

Detecting anomalies on attributed graphs is a challenging task since labelled anomalies are highly labour-intensive by taking specialized domain knowledge to make anomalous samples not as available as normal ones. Moreover, graphs contain complex structure information as well as attribute information, leading to anomalies that can be typically hidden in the structure space, attribute space, and the mix of both. In this paper, we propose a novel model for graph anomaly detection named ProGAD. Specifically, ProGAD takes advance of label propagation to infer high-quality pseudo labels by considering the structure and attribute inconsistencies between normal and abnormal samples. Meanwhile, ProGAD introduces the prior knowledge of class distribution to correct and refine pseudo labels with a prototype-aware strategy. Experiments demonstrate that ProGAD achieves strong performance compared with the current state-of-the-art methods.

----

## [2795] Gaze-Based Interaction Adaptation for People with Involuntary Head Movements (Student Abstract)

**Authors**: *Cindy Tong, Rosanna Yuen-Yan Chan*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30519](https://doi.org/10.1609/aaai.v38i21.30519)

**Abstract**:

Gaze estimation is an important research area in computer vision and machine learning. Eye-tracking and gaze-based interactions have made assistive technology (AT) more accessible to people with physical limitations. However, a non-negligible proportion of existing AT users, including those having dyskinetic cerebral palsy (CP) or severe intellectual disabilities (ID), have difficulties in using eye trackers due to their involuntary body movements. In this paper, we propose an adaptation method pertaining to head movement prediction and fixation smoothing to stabilize our target users' gaze points on the screen and improve their user experience (UX) in gaze-based interaction. Our empirical experimentation shows that our method significantly shortens the users' selection time and increases their selection accuracy.

----

## [2796] COPD-FlowNet: Elevating Non-invasive COPD Diagnosis with CFD Simulations (Student Abstract)

**Authors**: *Aryan Tyagi, Aryaman Rao, Shubhanshu Rao, Raj Kumar Singh*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30520](https://doi.org/10.1609/aaai.v38i21.30520)

**Abstract**:

Chronic Obstructive Pulmonary Disorder (COPD) is a prevalent respiratory disease that significantly impacts the quality of life of affected individuals. This paper presents COPD-FlowNet, a novel deep-learning framework that leverages a custom Generative Adversarial Network (GAN) to generate synthetic Computational Fluid Dynamics (CFD) velocity flow field images specific to the trachea of COPD patients. These synthetic images serve as a valuable resource for data augmentation and model training. Additionally, COPD-FlowNet incorporates a custom Convolutional Neural Network (CNN) architecture to predict the location of the obstruction site.

----

## [2797] Equivalence between Graph Spectral Clustering and Column Subset Selection (Student Abstract)

**Authors**: *Guihong Wan, Wei Mao, Yevgeniy R. Semenov, Haim Schweitzer*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30521](https://doi.org/10.1609/aaai.v38i21.30521)

**Abstract**:

The common criteria for evaluating spectral clustering are NCut and RatioCut. The seemingly unrelated column subset selection (CSS) problem aims to compute a column subset that linearly approximates the entire matrix. A common criterion is the approximation error in the Frobenius norm (ApproxErr). We show that any algorithm for CSS can be viewed as a clustering algorithm that minimizes NCut by applying it to a matrix formed from graph edges. Conversely, any clustering algorithm can be seen as identifying a column subset from that matrix. In both cases, ApproxErr and NCut have the same value. Analogous results hold for RatioCut with a slightly different matrix. Therefore, established results for CSS can be mapped to spectral clustering. We use this to obtain new clustering algorithms, including an optimal one that is similar to A*. This is the first nontrivial clustering algorithm with such an optimality guarantee. A variant of the weighted A* runs much faster and provides bounds on the accuracy. Finally, we use the results from spectral clustering to prove the NP-hardness of CSS from sparse matrices.

----

## [2798] Opening the Black Box: Unraveling the Classroom Dialogue Analysis (Student Abstract)

**Authors**: *Deliang Wang*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30522](https://doi.org/10.1609/aaai.v38i21.30522)

**Abstract**:

This paper explores proposing interpreting methods from explainable artificial intelligence to address the interpretability issues in deep learning-based models for classroom dialogue. Specifically, we developed a Bert-based model to automatically detect student talk moves within classroom dialogues, utilizing the TalkMoves dataset. Subsequently, we proposed three generic interpreting methods, namely saliency, input*gradient, and integrated gradient, to explain the predictions of classroom dialogue models by computing input relevance (i.e., contribution). The experimental results show that the three interpreting methods can effectively unravel the classroom dialogue analysis, thereby potentially fostering teachers' trust.

----

## [2799] The CoachAI Badminton Environment: A Novel Reinforcement Learning Environment with Realistic Opponents (Student Abstract)

**Authors**: *Kuang-Da Wang, Wei-Yao Wang, Yu-Tse Chen, Yu-Heng Lin, Wen-Chih Peng*

**Conference**: *aaai 2024*

**URL**: [https://doi.org/10.1609/aaai.v38i21.30523](https://doi.org/10.1609/aaai.v38i21.30523)

**Abstract**:

The growing demand for precise sports analysis has been explored to improve athlete performance in various sports (e.g., basketball, soccer). However, existing methods for different sports face challenges in validating strategies in environments due to simple rule-based opponents leading to performance gaps when deployed in real-world matches. In this paper, we propose the CoachAI Badminton Environment, a novel reinforcement learning (RL) environment with realistic opponents for badminton, which serves as a compelling example of a turn-based game. It supports researchers in exploring various RL algorithms with the badminton context by integrating state-of-the-art tactical-forecasting models and real badminton game records. The Badminton Benchmarks are proposed with multiple widely adopted RL algorithms to benchmark the performance of simulating matches against real players. To advance novel algorithms and developments in badminton analytics, we make our environment open-source, enabling researchers to simulate more complex badminton sports scenarios based on this foundation. Our code is available at https://github.com/wywyWang/CoachAI-Projects/tree/main/CoachAI%20Badminton%20Environment.

----



[Go to the previous page](AAAI-2024-list13.md)

[Go to the next page](AAAI-2024-list15.md)

[Go to the catalog section](README.md)