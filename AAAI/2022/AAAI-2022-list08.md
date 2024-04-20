## [1400] A Tale of Color Variants: Representation and Self-Supervised Learning in Fashion E-commerce

**Authors**: *Ujjal Kr Dutta, Sandeep Repakula, Maulik Parmar, Abhinav Ravi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21516](https://doi.org/10.1609/aaai.v36i11.21516)

**Abstract**:

In this paper, we address a crucial problem in fashion e-commerce (with respect to customer experience, as well as revenue): color variants identification, i.e., identifying fashion products that match exactly in their design (or style), but only to differ in their color. We propose a generic framework, that leverages deep visual Representation Learning at its heart, to address this problem for our fashion e-commerce platform. Our framework could be trained with supervisory signals in the form of triplets, that are obtained manually. However, it is infeasible to obtain manual annotations for the entire huge collection of data usually present in fashion e-commerce platforms, such as ours, while capturing all the difficult corner cases. But, to our rescue, interestingly we observed that this crucial problem in fashion e-commerce could also be solved by simple color jitter based image augmentation, that recently became widely popular in the contrastive Self-Supervised Learning (SSL) literature, that seeks to learn visual representations without using manual labels. This naturally led to a question in our mind: Could we leverage SSL in our use-case, and still obtain comparable performance to our supervised framework? The answer is, Yes! because, color variant fashion objects are nothing but manifestations of a style, in different colors, and a model trained to be invariant to the color (with, or without supervision), should be able to recognize this! This is what the paper further demonstrates, both qualitatively, and quantitatively, while evaluating a couple of state-of-the-art SSL techniques, and also proposing a novel method.

----

## [1401] Deploying an Artificial Intelligence Application to Detect Flood from Sentinel 1 Data

**Authors**: *Paolo Fraccaro, Nikola Stoyanov, Zaheed Gaffoor, Laura Elena Cue La Rosa, Jitendra Singh, Tatsuya Ishikawa, Blair Edwards, Anne Jones, Komminist Weldemariam*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21517](https://doi.org/10.1609/aaai.v36i11.21517)

**Abstract**:

As climate change is increasing the frequency and intensity of climate and weather hazards, improving detection and monitoring of flood events is a priority. Being weather independent and high resolution, Sentinel 1 (S1) radar satellite imagery data has become the go to data source to detect flood events accurately. However, current methods are either based on fixed thresholds to differentiate water from land or train Artificial Intelligence (AI) models based on only S1 data, despite the availability of many other relevant data sources publicly. These models also lack comprehensive validations on out-of-sample data and deployment at scale.  In this study, we investigated whether adding extra input layers could increase the performance of AI models in detecting floods from S1 data. We also provide performance across a range of 11 historical events, with results ranging between 0.93 and 0.97 accuracy, 0.53 and 0.81 IoU, and 0.68 and 0.89 F1 scores. Finally, we show the infrastructure we developed to deploy our AI models at scale to satisfy a range of use cases and user requests.

----

## [1402] Facilitating Human-Wildlife Cohabitation through Conflict Prediction

**Authors**: *Susobhan Ghosh, Pradeep Varakantham, Aniket Bhatkhande, Tamanna Ahmad, Anish Andheria, Wenjun Li, Aparna Taneja, Divy Thakkar, Milind Tambe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21518](https://doi.org/10.1609/aaai.v36i11.21518)

**Abstract**:

With increasing world population and expanded use of forests as cohabited regions, interactions and conflicts with wildlife are increasing, leading to large scale loss of  lives (animal and human) and livelihoods (economic). While community knowledge is valuable, forest officials and conservation organisations can greatly benefit from predictive analysis of human-wildlife conflict, leading to targeted interventions that can potentially help save lives and livelihoods. However, the problem of prediction is a complex socio-technical problem in the context of limited data in low-resource regions. 
    
Identifying the right features to make accurate predictions of conflicts at the required spatial granularity using a sparse conflict training dataset is the key challenge that we address in this paper. Specifically, we do an illustrative case study on human-wildlife conflicts in the Bramhapuri Forest Division in Chandrapur, Maharashtra, India. Most existing work has considered human wildlife conflicts in protected areas and to the best of our knowledge, this is the first effort at prediction of human-wildlife conflicts in unprotected areas and using those predictions for deploying interventions on the ground.

----

## [1403] PaintTeR: Automatic Extraction of Text Spans for Generating Art-Centered Questions

**Authors**: *Sujatha Das Gollapalli, See-Kiong Ng, Ying Kiat Tham, Shan Shan Chow, Jia Min Wong, Kevin Lim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21519](https://doi.org/10.1609/aaai.v36i11.21519)

**Abstract**:

We propose PaintTeR, our Paintings TextRank algorithm for extracting art-related text spans from passages on paintings. PaintTeR combines a lexicon of painting words curated automatically through distant supervision with random walks on a large-scale word co-occurrence graph for ranking passage spans for artistic characteristics. The spans extracted with PaintTeR are used in state-of-the-art Question Generation and Reading Comprehension models for designing an interactive aid that enables gallery and museum visitors focus on the artistic elements of paintings. We provide experiments on two datasets of expert-written passages on paintings to showcase the effectiveness of PaintTeR. Evaluations by both gallery experts as well as crowdworkers indicate that our proposed algorithm can be used to
select relevant and interesting art-centered questions. To the best of our knowledge, ours is the first work to effectively fine-tune question generation models using minimal supervision for a low-resource, specialized context such as gallery visits.

----

## [1404] Flexible-Window Predictions on Electronic Health Records

**Authors**: *Mehak Gupta, Raphael Poulain, Thao-Ly T. Phan, H. Timothy Bunnell, Rahmatollah Beheshti*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21520](https://doi.org/10.1609/aaai.v36i11.21520)

**Abstract**:

Various types of machine learning techniques are available for analyzing electronic health records (EHRs). For predictive tasks, most existing methods either explicitly or implicitly divide these time-series datasets into predetermined observation and prediction windows. Patients have different lengths of medical history and the desired predictions (for purposes such as diagnosis or treatment) are required at different times in the future. In this paper, we propose a method that uses a sequence-to-sequence generator model to transfer an input sequence of EHR data to a sequence of user-defined target labels, providing the end-users with ``flexible'' observation and prediction windows to define. We use adversarial and semi-supervised approaches in our design, where the sequence-to-sequence model acts as a generator and a discriminator distinguishes between the actual (observed) and generated labels. We evaluate our models through an extensive series of experiments using two large EHR datasets from adult and pediatric populations. In an obesity predicting case study, we show that our model can achieve superior results in flexible-window prediction tasks, after being trained once and even with large missing rates on the input EHR data. Moreover, using a number of attention analysis experiments, we show that the proposed model can effectively learn more relevant features in different prediction tasks.

----

## [1405] AI for Disaster Rapid Damage Assessment from Microblogs

**Authors**: *Muhammad Imran, Umair Qazi, Ferda Ofli, Steve Peterson, Firoj Alam*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21521](https://doi.org/10.1609/aaai.v36i11.21521)

**Abstract**:

Formal response organizations perform rapid damage assessments after natural and human-induced disasters to measure the extent of damage to infrastructures such as roads, bridges, and buildings. This time-critical task, when performed using traditional approaches such as experts surveying the disaster areas, poses serious challenges and delays response. This paper presents an AI-based system that leverages citizen science to collect damage images reported on social media and perform rapid damage assessment in real-time. Several image processing models in the system tackle non-trivial challenges posed by social media as a data source, such as high-volume of redundant and irrelevant content. The system determines the severity of damage using a state-of-the-art computer vision model. Together with a response organization in the US, we deployed the system to identify damage reports during a major real-world disaster. We observe that almost 42% of the images are unique, 28% relevant, and more importantly, only 10% of them contain either mild or severe damage. Experts from our partner organization provided feedback on the system's mistakes, which we used to perform additional experiments to retrain the models. Consequently, the retrained models based on expert feedback on the target domain data helped us achieve significant performance improvements.

----

## [1406] Designing a Human-in-the-Loop System for Object Detection in Floor Plans

**Authors**: *Johannes Jakubik, Patrick Hemmer, Michael Vössing, Benedikt Blumenstiel, Andrea Bartos, Kamilla Mohr*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21522](https://doi.org/10.1609/aaai.v36i11.21522)

**Abstract**:

Abstract
In recent years, companies in the Architecture, Engineering, and Construction (AEC) industry have started exploring how artificial intelligence (AI) can reduce time-consuming and repetitive tasks. One use case that can benefit from the adoption of AI is the determination of quantities in floor plans. This information is required for several planning and construction steps. Currently, the task requires companies to invest a significant amount of manual effort. Either digital floor plans are not available for existing buildings, or the formats cannot be processed due to lack of standardization. In this paper, we therefore propose a human-in-the-loop approach for the detection and classification of symbols in floor plans. The developed system calculates a measure of uncertainty for each detected symbol which is used to acquire the knowledge of human experts for those symbols that are difficult to classify. We evaluate our approach with a real-world dataset provided by an industry partner and find that the selective acquisition of human expert knowledge enhances the model’s performance by up to 12.9%—resulting in an overall prediction accuracy of 92.1% on average. We further design a pipeline for the generation of synthetic training data that allows the systems to be adapted to new construction projects with minimal manual effort. Overall, our work supports professionals in the AEC industry on their journey to the data-driven generation of business value.

----

## [1407] Bayesian Model-Based Offline Reinforcement Learning for Product Allocation

**Authors**: *Porter Jenkins, Hua Wei, J. Stockton Jenkins, Zhenhui Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21523](https://doi.org/10.1609/aaai.v36i11.21523)

**Abstract**:

Product allocation in retail is the process of placing products throughout a store to connect consumers with relevant products. Discovering a good allocation strategy is challenging due to the scarcity of data and the high cost of experimentation in the physical world. Some work explores Reinforcement learning (RL) as a solution, but these approaches are often limited because of the sim2real problem. Learning policies from logged trajectories of a system is a key step forward for RL in physical systems. Recent work has shown that model-based offline RL can improve the effectiveness of offline policy estimation through uncertainty-penalized exploration. However, existing work assumes a continuous state space and access to a covariance matrix of the environment dynamics, which is not possible in the discrete case. To solve this problem, we propose a Bayesian model-based technique that naturally produces probabilistic estimates of the environment dynamics via the posterior predictive distribution, which we use for uncertainty-penalized exploration. We call our approach Posterior Penalized Offline Policy Optimization (PPOPO). We show that our world model better fits historical data due to informative priors, and that PPOPO outperforms other offline techniques in simulation and against real-world data.

----

## [1408] Learning Space-Time Crop Yield Patterns with Zigzag Persistence-Based LSTM: Toward More Reliable Digital Agriculture Insurance

**Authors**: *Tian Jiang, Meichen Huang, Ignacio Segovia-Dominguez, Nathaniel K. Newlands, Yulia R. Gel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21524](https://doi.org/10.1609/aaai.v36i11.21524)

**Abstract**:

More than US$ 27 billion is estimated to have been paid-out in farm support in USA alone since 1991 in response to climate change impacts on agriculture, with costs likely continuing to rise. With the wider adoption of precision agriculture - an agriculture management strategy that involves gathering, processing and analyzing temporal, spatial and individual data - in both developed and developing countries, there is an increasing opportunity to harness accumulating, shareable, big data using artificial intelligence (AI) methods, collected from weather stations, field sensor networks, Internet-of-Things devices, unmanned aerial vehicles, and earth observational satellites. This requires smart algorithms tailored to agricultural data types, integrated into digital solutions that are viable, flexible, and scalable for wide deployment for a wide variety of agricultural users and decision-makers. We discuss a novel AI approach that addresses the real-world problem of developing a viable solution for reliably, timely, and cost-effectively forecasting crop status across large agricultural regions using Earth observational information in near-real-time. Our approach is based on extracting time-conditioned topological features which characterize complex spatio-temporal dependencies between crop production regions and integrating such topological signatures into Long Short Term Memory (LSTM). We discuss utility and limitations of the resulting zigzag persistence-based LSTM (ZZTop-LSTM) as a new tool for developing more informed crop insurance rate-making and accurate tracking of changing risk exposures and vulnerabilities within insurance risk areas.

----

## [1409] A Machine Learning Method for EV Range Prediction with Updates on Route Information and Traffic Conditions

**Authors**: *Dohee Kim, Hong Gi Shim, Jeong Soo Eo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21525](https://doi.org/10.1609/aaai.v36i11.21525)

**Abstract**:

Driver's anxiety about the remaining driving range of electric vehicles (EVs) has been quite improved by mounting a high-capacity battery pack. However, when EVs need to be charged, the drivers still feel uncomfortable if inaccurate range prediction is provided because the inaccuracy makes it difficult to decide when and where to charge EV. In this paper, to mitigate the EV range anxiety, a new machine learning (ML) method to enhance range prediction accuracy is proposed in a practical way. For continuously obtaining the recent traffic conditions ahead, input features indicating the near-future vehicle dynamics are connected to a long short-term memory (LSTM) network, which can consecutively utilize a relation of neighboring data, and then the output features of the LSTM network with another input features consisting of energy-related vehicle system states become another input layer for deep learning network (DNN). The proposed LSTM-DNN mixture model is trained by exploiting the driving data of about 160,000 km and the following test performance shows that the model retains the range prediction accuracy of 2 ~ 3 km in a time window of 40 min. The test results indicate that the LSTM-DNN range prediction model is able to make a far-sighted range prediction while considering varying map and traffic information to a destination.

----

## [1410] Domain Reconstruction for UWB Car Key Localization Using Generative Adversarial Networks

**Authors**: *Aleksei Kuvshinov, Daniel Knobloch, Daniel Külzer, Elen Vardanyan, Stephan Günnemann*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21526](https://doi.org/10.1609/aaai.v36i11.21526)

**Abstract**:

We consider the car key localization task using ultra-wideband (UWB) signal measurements. Given labeled data for a certain car, we train a deep classifier to make the prediction about the new points. However, due to the differences in car models and possible environmental effects that might alter the signal propagation, data collection requires considerable effort for each car. In particular, we consider a situation where the data for the new car is collected only in one environment, so we have to utilize the measurements in other environments from a different car. We propose a framework based on generative adversarial networks (GANs) to generate missing parts of the data and train the classifier on it, mitigating the necessity to collect the real data. We show that the model trained on the synthetic data performs better than the baseline trained on the collected measurements only. Furthermore, our model closes the gap to the level of performance achieved when we would have the information about the new car in multiple environments by 35%.

----

## [1411] ALPHAPROG: Reinforcement Generation of Valid Programs for Compiler Fuzzing

**Authors**: *Xiaoting Li, Xiao Liu, Lingwei Chen, Rupesh Prajapati, Dinghao Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21527](https://doi.org/10.1609/aaai.v36i11.21527)

**Abstract**:

Fuzzing is a widely-used testing technique to assure software robustness. However, automatic generation of high-quality test suites is challenging, especially for software that takes in highly-structured inputs, such as the compilers. Compiler fuzzing remains difficult as generating tons of syntactically and semantically valid programs is not trivial. Most previous methods either depend on human-crafted grammars or heuristics to learn partial language patterns. They both suffer from the completeness issue that is a classic puzzle in software testing. To mitigate the problem, we propose a knowledge-guided reinforcement learning-based approach to generating valid programs for compiler fuzzing. We first design a naive learning model which evolves with the sequential mutation rewards provided by a target compiler we test. By iterating the training cycle, the model learns to generate valid programs that can improve the testing efficacy as well. We implement the proposed method into a tool called ALPHAPROG. We analyze the framework with four different reward functions and our study reveal the effectiveness of  ALPHAPROG for compiler testing. We also reported two important bugs for a compiler production that were confirmed and addressed by the project owner, which further demonstrates ALPHAPROG's applied value in practice.

----

## [1412] Combating Sampling Bias: A Self-Training Method in Credit Risk Models

**Authors**: *Jingxian Liao, Wei Wang, Jason Xue, Anthony Lei, Xue Han, Kun Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21528](https://doi.org/10.1609/aaai.v36i11.21528)

**Abstract**:

A significant challenge in credit risk models for underwriting is the presence of bias in model training data. When most credit risk models are built using only applicants who had been funded for credit, such non-random sampling predominantly influenced by credit policymakers and previous loan performances may introduce sampling bias to the models, and thus alter their prediction of default on loan repayment when screening applications from prospective borrowers. In this paper, we propose a novel data augmentation method that aims to identify and pseudo-label parts of the historically declined loan applications to mitigate sampling bias in the training data. We also introduce a new measure to assess the performance from the business perspective, loan application approval rates at various loan default rate levels. Our proposed methods were compared to the original supervised learning model and the traditional sampling issue remedy techniques in the industry. The experiment and early production results from deployed model show that self-training method with calibrated probability as data augmentation selection criteria improved the ability of credit scoring to differentiate default loan applications and, more importantly, can increase loan approval rate up to 8.8\%,  while keeping similar default rate comparing to baselines. The results demonstrate practical implications on how future underwriting model development processes should follow.

----

## [1413] Data-Driven Real-Time Strategic Placement of Mobile Vaccine Distribution Sites

**Authors**: *Zakaria Mehrab, Mandy L. Wilson, Serina Chang, Galen Harrison, Bryan L. Lewis, Alex Telionis, Justin Crow, Dennis Kim, Scott Spillmann, Kate Peters, Jure Leskovec, Madhav V. Marathe*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21529](https://doi.org/10.1609/aaai.v36i11.21529)

**Abstract**:

The deployment of vaccines across the US provides significant defense against serious illness and death from COVID-19.  Over 70% of vaccine-eligible Americans are at least partially vaccinated, but there are pockets of the population that are under-vaccinated, such as in rural areas and some demographic groups (e.g. age, race, ethnicity). These pockets are extremely susceptible to the Delta variant, exacerbating the healthcare crisis and increasing the risk of new variants. In this paper, we describe a data-driven model that provides real-time support to Virginia public health officials by recommending mobile vaccination site placement in order to target under-vaccinated populations.  Our strategy uses fine-grained mobility data, along with US Census and vaccination uptake data, to identify locations that are most likely to be visited by unvaccinated individuals. We further extend our model to choose locations that maximize vaccine uptake among hesitant groups. We show that the top recommended sites vary substantially across some demographics, demonstrating the value of developing customized recommendation models that integrate fine-grained, heterogeneous data sources. We also validate our recommendations by analyzing the success rates of deployed vaccine sites, and show that sites placed closer to our recommended areas administered higher numbers of doses. Our model is the first of its kind to consider evolving mobility patterns in real-time for suggesting placement strategies customized for different targeted demographic groups.

----

## [1414] An Interactive Explanatory AI System for Industrial Quality Control

**Authors**: *Dennis Müller, Michael März, Stephan Scheele, Ute Schmid*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21530](https://doi.org/10.1609/aaai.v36i11.21530)

**Abstract**:

Machine learning based image classification algorithms, such as deep neural network approaches, will be increasingly employed  in  critical  settings  such  as  quality  control  in  industry,  where  transparency  and  comprehensibility  of  decisions are  crucial.  Therefore,  we  aim  to  extend  the  defect  detection task towards an interactive human-in-the-loop approach that  allows  us  to  integrate  rich  background  knowledge  and the  inference  of  complex  relationships  going  beyond  traditional  purely  data-driven  approaches.  We  propose  an  approach for an interactive support system for classifications in an industrial quality control setting that combines the advantages of both (explainable) knowledge-driven and data-driven machine learning methods, in particular inductive logic programming  and  convolutional  neural  networks,  with  human expertise  and  control.  The  resulting  system  can  assist  domain experts with decisions, provide transparent explanations for results, and integrate feedback from users; thus reducing workload  for  humans  while  both  respecting  their  expertise and without removing their agency or accountability.

----

## [1415] Inferring Multiple Tissue Properties from Magnetic Resonance Fingerprinting Images

**Authors**: *Naren Nallapareddy, Soumya Ray*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21531](https://doi.org/10.1609/aaai.v36i11.21531)

**Abstract**:

Magnetic Resonance Imaging (MRI) is a non-invasive imaging modality that is a cornerstone of diagnostic radiology. Clinical MRI scans capture a single image to highlight a single tissue property. The intensity difference between different regions of this image shows disease states that a radiologist can interpret. Magnetic Resonance Fingerprinting (MRF) is a recently proposed novel MRI technique. MRF allows the capture of multiple MR images in a single scan. This enables clinicians to analyze multiple tissue properties, potentially increasing the sensitivity of diagnosis and also allowing for the diagnosis of novel diseases. However, it is more challenging to analyze MRF images, because MRF produces much larger and noisier data than MRI. In this paper, we show how AI techniques can help solve this problem. Using a hybrid search strategy combining simulated annealing with pattern search, we show it is possible to tractably reconstruct multiple tissue properties from a single MRF image. This is a key step towards the deployment of MRF for radiological diagnosis.

----

## [1416] Learning to Rank Articles for Molecular Queries

**Authors**: *Galia Nordon, Aviram Magen, Ido Guy, Kira Radinsky*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21532](https://doi.org/10.1609/aaai.v36i11.21532)

**Abstract**:

The cost of developing new drugs is estimated at billions of dollars per year. Identification of new molecules for drugs involves scanning existing bio-medical literature for relevant information. As the potential drug molecule is novel, retrieval of relevant information using a simple direct search is less likely to be productive. Identifying relevant papers is therefore a more complex and challenging task, which requires searching for information on molecules with similar characteristics to the novel drug. In this paper, we present the novel task of ranking documents based on novel molecule queries. Given a chemical molecular structure, we wish to rank medical papers that will contribute to a researcher's understanding of the novel molecule drug potential.
We present a set of ranking algorithms and molecular embeddings to address the task. An extensive evaluation of the algorithms is performed over the molecular embeddings, studying their performance on a benchmark retrieval corpus, which we share with the community.
Additionally, we introduce a heterogeneous edge-labeled graph embedding approach to address the molecule ranking task. Our evaluation shows that the proposed embedding model can significantly improve molecule ranking methods. The system is currently deployed in a targeted drug delivery and personalized medicine research laboratory.

----

## [1417] Outlier Detection in Wind Turbine Frequency Converters Using Long-Term Sensor Data

**Authors**: *Nils Schwenzfeier, Markus Heikamp, Ole Meyer, Andre Hönnscheidt, Michael Steffes, Volker Gruhn*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21533](https://doi.org/10.1609/aaai.v36i11.21533)

**Abstract**:

Wind energy is an important source of renewable and sustainable energy and therefore an elementary component of any future energy supply. However, the operation of large wind farms places high demands on reliability and is often impacted by high maintenance and repair costs in the event of a failure. A frequency converter is one of the most important components of each wind turbine, which ensures that the frequency of the generated energy synchronises with the grid frequency and thus enables the flow of energy into the power grid. The detection of anomalies in these devices is complex due to the high frequency and multidimensionality of different sensor information from the energy control units and requires fault patterns to be discovered and detected in large time series. In this paper, we show how state-of-the-art self-supervised-learning techniques, namely LSTM autoencoders, can be successfully applied to real-world data. We describe the extensions we have made to deal with the often very noisy sensors and describe the construction of the training data set. The trained system was first tested and evaluated on synthetic data and subsequently on a large real-world data set. In both cases, it was shown that outliers can be reliably identified using our presented approach.

----

## [1418] Mitigating Low Agricultural Productivity of Smallholder Farms in Africa: Time-Series Forecasting for Environmental Stressors

**Authors**: *Maryam Tabar, Dongwon Lee, David P. Hughes, Amulya Yadav*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21534](https://doi.org/10.1609/aaai.v36i11.21534)

**Abstract**:

African smallholder farmers have struggled with low agricultural productivity for decades, partly due to their inability to proactively assess irrigation needs in their farms in the face of long-term climate change. In this paper, we tackle this challenge by employing data-driven techniques to develop forecasting tools for three widely used crop-productivity related variables (i.e., actual evapotranspiration, reference evapotranspiration, and net primary production), which can then be used by farmers to take corrective actions on their farms. Prior work in this domain, despite using data-driven methods, suffers from two major limitations: (i) they mainly focus on estimating variable values (as opposed to forecasting the future); and (ii) they mostly use classical Machine Learning (ML) prediction models, despite the abundance of data sufficient to train sophisticated deep learning models. To fill this research gap, we collaborate with PlantVillage, the world’s leading non-profit agricultural knowledge delivery platform for African farmers, to identify ∼2,200 smallholder farm locations, and gather remote-sensed data of these farms over a period of five years. Next, we propose CLIMATES, a meta-algorithm leveraging structural insights about temporal patterns of this time-series data to accurately forecast their future values. We conduct extensive experiments to evaluate its performance in this domain. Our experimental results show that CLIMATES outperforms several state-of-the-art time-series forecasting models. We also provide insights about the poor performance of some competing models. Our work is being evaluated by officials at PlantVillage for potential future deployment as an early warning system in East Africa. We release the code at https://github.com/maryam-tabar/CLIMATES.

----

## [1419] Reinforcement Learning for Datacenter Congestion Control

**Authors**: *Chen Tessler, Yuval Shpigelman, Gal Dalal, Amit Mandelbaum, Doron Haritan Kazakov, Benjamin Fuhrer, Gal Chechik, Shie Mannor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21535](https://doi.org/10.1609/aaai.v36i11.21535)

**Abstract**:

We approach the task of network congestion control in datacenters using Reinforcement Learning (RL). Successful congestion control algorithms can dramatically improve latency and overall network throughput. Until today, no such learning-based algorithms have shown practical potential in this domain. Evidently, the most popular recent deployments rely on rule-based heuristics that are tested on a predetermined set of benchmarks. Consequently, these heuristics do not generalize well to newly-seen scenarios. Contrarily, we devise an RL-based algorithm with the aim of generalizing to different configurations of real-world datacenter networks. We overcome challenges such as partial-observability, non-stationarity, and multi-objectiveness. We further propose a policy gradient algorithm that leverages the analytical structure of the reward function to approximate its derivative and improve stability. We show that these challenges prevent standard RL algorithms from operating within this domain. Our experiments, conducted on a realistic simulator that emulates communication networks' behavior, show that our method exhibits improved performance concurrently on the multiple considered metrics compared to the popular algorithms deployed today in real datacenters. Our algorithm is being productized to replace heuristics in some of the largest datacenters in the world.

----

## [1420] Adaptive Global-Local Context Fusion for Multi-Turn Spoken Language Understanding

**Authors**: *Thanh Tran, Kai Wei, Weitong Ruan, Ross McGowan, Nathan Susanj, Grant P. Strimel*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21536](https://doi.org/10.1609/aaai.v36i11.21536)

**Abstract**:

Recent years have seen significant advances in multi-turn Spoken Language Understanding (SLU), where dialogue contexts are used to guide intent classification and slot filling. However, how to selectively incorporate dialogue contexts, such as previous utterances and dialogue acts, into multi-turn SLU still remains a substantial challenge. In this work, we propose a novel contextual SLU model for multi-turn intent classification and slot filling tasks. We introduce an adaptive global-local context fusion mechanism to selectively integrate dialogue contexts into our model. The local context fusion aligns each dialogue context using multi-head attention, while the global context fusion measures overall context contribution to intent classification and slot filling tasks. Experiments show that on two benchmark datasets, our model achieves absolute F1 score improvements of 2.73% and 2.57% for the slot filling task on Sim-R and Sim M datasets, respectively. Additional experiments on a large-scale, de-identified, in-house dataset further verify the measurable accuracy gains of our proposed model.

----

## [1421] AI-Assisted Controls Change Management for Cybersecurity in the Cloud

**Authors**: *Harshal Tupsamudre, Arun Kumar, Vikas Agarwal, Nisha Gupta, Sneha Mondal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21537](https://doi.org/10.1609/aaai.v36i11.21537)

**Abstract**:

Webscale services dealing with sensitive content are increasingly being deployed in public and hybrid cloud environments. At the same time, the impact of security breaches have also increased manifold averaging at USD 3.86M per data breach. To tackle such increasing risks, regulations and security frameworks are defined that an organization must comply with. Most of these frameworks are published in natural language text that run into hundreds of pages resulting into thousands of requirements and controls. When these frameworks undergo revisions, understanding the changes, and interpreting their impact consumes huge amount of time, effort and resources.

In this paper, we propose a change management system that supports SMEs with AI-assisted automation of this extremely manual and time consuming activity. Specifically, we introduce the concept of live crosswalks – a framework that models complex relationships among security and compliance documents along with associated operations to manage the change. It uses natural language processing (NLP) and algorithmic techniques to transform the current document-driven, highly manual process into a data-driven interactive intelligent system. We present the overall design and demonstrate its efficacy over several hundreds of diversified controls through experimental evaluation.

----

## [1422] Predictive Maintenance for General Aviation Using Convolutional Transformers

**Authors**: *Hong Yang, Aidan P. LaBella, Travis Desell*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21538](https://doi.org/10.1609/aaai.v36i11.21538)

**Abstract**:

Predictive maintenance systems have the potential to significantly reduce costs for maintaining aircraft fleets as well as provide improved safety by detecting maintenance issues before they come severe. However, the development of such systems has been limited due to a lack of publicly labeled multivariate time series (MTS) sensor data. MTS classification has advanced greatly over the past decade, but there is a lack of sufficiently challenging benchmarks for new methods. This work introduces the NGAFID Maintenance Classification (NGAFID-MC) dataset as a novel benchmark in terms of difficulty, number of samples, and sequence length. NGAFID-MC consists of over 7,500 labeled flights, representing over 11,500 hours of per second flight data recorder readings of 23 sensor parameters. Using this benchmark, we demonstrate that Recurrent Neural Network (RNN) methods are not well suited for capturing temporally distant relationships and propose a new architecture called Convolutional Multiheaded Self Attention (Conv-MHSA) that achieves greater classification performance at greater computational efficiency. We also demonstrate that image inspired augmentations of cutout, mixup, and cutmix, can be used to reduce overfitting and improve generalization in MTS classification.  Our best trained models have been incorporated back into the NGAFID to allow users to potentially detect flights that require maintenance as well as provide feedback to further expand and refine the NGAFID-MC dataset.

----

## [1423] DocBed: A Multi-Stage OCR Solution for Documents with Complex Layouts

**Authors**: *Wenzhen Zhu, Negin Sokhandan, Guang Yang, Sujitha Martin, Suchitra Sathyanarayana*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21539](https://doi.org/10.1609/aaai.v36i11.21539)

**Abstract**:

Digitization of newspapers is of interest for many reasons including preservation of history, accessibility and search ability, etc. While digitization of documents such as scientific articles and magazines is prevalent in literature, one of the main challenges for digitization of newspaper lies in its complex layout (e.g. articles spanning multiple columns, text interrupted by images) analysis, which is necessary to preserve human read-order. This work provides a major breakthrough in the digitization of newspapers on three fronts: first, releasing a dataset of 3000 fully-annotated, real-world newspaper images from 21 different U.S. states representing an extensive variety of complex layouts for document layout analysis; second, proposing layout segmentation as a precursor to existing optical character recognition (OCR) engines, where multiple state-of-the-art image segmentation models and several post-processing methods are explored for document layout segmentation; third, providing a thorough and structured evaluation protocol for isolated layout segmentation and end-to-end OCR.

----

## [1424] AI Explainability 360: Impact and Design

**Authors**: *Vijay Arya, Rachel K. E. Bellamy, Pin-Yu Chen, Amit Dhurandhar, Michael Hind, Samuel C. Hoffman, Stephanie Houde, Q. Vera Liao, Ronny Luss, Aleksandra Mojsilovic, Sami Mourad, Pablo Pedemonte, Ramya Raghavendra, John T. Richards, Prasanna Sattigeri, Karthikeyan Shanmugam, Moninder Singh, Kush R. Varshney, Dennis Wei, Yunfeng Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21540](https://doi.org/10.1609/aaai.v36i11.21540)

**Abstract**:

As artificial intelligence and machine learning algorithms become  increasingly  prevalent  in  society,  multiple  stakeholders are calling for these algorithms to provide explanations. At  the  same  time,  these  stakeholders,  whether  they  be  affected  citizens,  government  regulators,  domain  experts,  or system developers, have different explanation needs. To address these needs, in 2019, we created AI Explainability 360, an open source software toolkit featuring ten  diverse  and  state-of-the-art  explainability  methods  and two  evaluation  metrics.  This  paper  examines  the  impact  of the toolkit with several case studies, statistics, and community feedback. The different ways in which users have experienced AI Explainability 360 have resulted in multiple types of impact and improvements in multiple metrics, highlighted by the adoption of the toolkit by the independent LF AI & Data Foundation. The paper also describes the flexible design of the toolkit, examples of its use, and the significant educational material and documentation available to its users.

----

## [1425] A Simulation-Based Evaluation Framework for Interactive AI Systems and Its Application

**Authors**: *Maeda F. Hanafi, Yannis Katsis, Martín Santillán Cooper, Yunyao Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21541](https://doi.org/10.1609/aaai.v36i11.21541)

**Abstract**:

Interactive AI (IAI) systems are increasingly popular as the human-centered AI design paradigm is gaining strong traction. However, evaluating IAI systems, a key step in building such systems, is particularly challenging, as their output highly depends on the performed user actions. Developers often have to rely on limited and mostly qualitative data from ad-hoc user testing to assess and improve their systems. In this paper, we present InteractEva; a systematic evaluation framework for IAI systems. We also describe how we have applied InteractEva to evaluate a commercial IAI system, leading to both quality improvements and better data-driven design decisions.

----

## [1426] Seq2Pat: Sequence-to-Pattern Generation for Constraint-Based Sequential Pattern Mining

**Authors**: *Xin Wang, Amin Hosseininasab, Pablo Colunga, Serdar Kadioglu, Willem-Jan van Hoeve*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21542](https://doi.org/10.1609/aaai.v36i11.21542)

**Abstract**:

Pattern mining is an essential part of knowledge discovery and data analytics. It is a powerful paradigm, especially when combined with constraint reasoning. In this paper, we present Seq2Pat, a constraint-based sequential pattern mining tool with a high-level declarative user interface. The library finds patterns that frequently occur in large sequence databases subject to constraints. We highlight key benefits that are desirable, especially in industrial settings where scalability, explainability, rapid experimentation, reusability, and reproducibility are of great interest. We then showcase an automated feature extraction process powered by Seq2Pat to discover high-level insights and boost downstream machine learning models for customer intent prediction.

----

## [1427] Accelerating COVID-19 Research with Graph Mining and Transformer-Based Learning

**Authors**: *Ilya Tyagin, Ankit Kulshrestha, Justin Sybrandt, Krish Matta, Michael Shtutman, Ilya Safro*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21543](https://doi.org/10.1609/aaai.v36i11.21543)

**Abstract**:

In 2020, the White House released the “Call to Action to the Tech Community on New Machine Readable COVID-19 Dataset,” wherein artificial intelligence experts are asked to collect data and develop text mining techniques that can help the science community answer high-priority scientific questions related to COVID-19. The Allen Institute for AI and collaborators announced the availability of a rapidly growing open dataset of publications, the COVID-19 Open Research Dataset (CORD-19). As the pace of research accelerates, biomedical scientists struggle to stay current. To expedite their investigations, scientists leverage hypothesis generation systems, which can automatically inspect published papers to discover novel implicit connections. We present automated general purpose hypothesis generation systems AGATHA-C and AGATHA-GP for COVID-19 research. The systems are based on the graph mining and transformer models. The systems are massively validated using retrospective information rediscovery and proactive analysis involving human-in-the-loop expert analysis. Both systems achieve high-quality predictions across domains in fast computational time and are released to the broad scientific community to accelerate biomedical research. In addition, by performing the domain expert curated study, we show that the systems are able to discover ongoing research findings such as the relationship between COVID-19 and oxytocin hormone.
All code, details, and pre-trained models are available at https://github.com/IlyaTyagin/AGATHA-C-GP.

----

## [1428] Towards an AI-Infused Interdisciplinary Curriculum for Middle-Grade Classrooms

**Authors**: *Bita Akram, Spencer Yoder, Cansu Tatar, Sankalp Boorugu, Ifeoluwa Aderemi, Shiyan Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21544](https://doi.org/10.1609/aaai.v36i11.21544)

**Abstract**:

As AI becomes more widely used across a variety of disciplines, it is increasingly important to teach AI concepts to K-12 students in order to prepare them for an AI-driven future workforce. Hence, educators and researchers have been working to develop curricula that make these concepts accessible to K-12 students. We are designing and developing a comprehensive AI curriculum delivered through a series of carefully crafted activities in an adapted \emph{Snap!} environment for middle-grade students. In this work, we lay out the proposed content of our curriculum and present the design, development, and implementation results of the first unit of our curriculum that focuses on teaching the breadth-first search algorithm. The activities in this unit have been revised after being piloted with a single high-school student. These activities were further refined after a group of K-12 teachers examined and critiqued them during a two-week professional development workshop. Our teachers created a lesson plan around the activities and implemented that lesson in a summer workshop with 14 middle school students. Our results demonstrated that our activities were successful in helping many of the students in understanding and implementing the algorithm through block-based programming while extra supplementary material was needed to assist some other students. In this paper, we explain our curriculum and technology, the results of implementing the first unit of our curriculum in a summer camp, and lessons learned for future developments.

----

## [1429] College Student Retention Risk Analysis from Educational Database Using Multi-Task Multi-Modal Neural Fusion

**Authors**: *Mohammad Arif Ul Alam*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21545](https://doi.org/10.1609/aaai.v36i11.21545)

**Abstract**:

We develop a Multimodal Spatiotemporal Neural Fusion network for MTL (MSNF-MTCL) to predict 5 important students' retention risks: future dropout, next semester dropout, type of dropout, duration of dropout and cause of dropout. First, we develop a general purpose multi-modal neural fusion network model MSNF for learning students' academic information representation by fusing spatial and temporal unstructured advising notes with spatiotemporal structured data. MSNF combines a Bidirectional Encoder Representations from Transformers (BERT)-based document embedding framework to represent each advising note, Long-Short Term Memory (LSTM) network to model temporal advising note embeddings, LSTM network to model students' temporal performance variables and students' static demographics altogether. The final fused representation from MSNF has been utilized on a Multi-Task Cascade Learning (MTCL) model towards building MSNF-MTCL for predicting 5 student retention risks. We evaluate MSNF-MTCL on a large educational database consists of 36,445 college students over 18 years period of time that provides promising performances comparing with the nearest state-of-art models. Additionally, we test the fairness of such model given the existence of biases.

----

## [1430] A Socially Relevant Focused AI Curriculum Designed for Female High School Students

**Authors**: *Lauren Alvarez, Isabella Gransbury, Veronica Cateté, Tiffany Barnes, Ákos Lédeczi, Shuchi Grover*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21546](https://doi.org/10.1609/aaai.v36i11.21546)

**Abstract**:

Historically, female students have shown low interest in the field of computer science. Previous computer science curricula have failed to address the lack of female-centered computer science activities, such as socially relevant and real-life applications. Our new summer camp curriculum introduces the topics of artificial intelligence (AI), machine learning (ML) and other real-world subjects to engage high school girls in computing by connecting lessons to relevant and cutting edge technologies. Topics range from social media bots, sentiment of natural language in different media, and the role of AI in criminal justice, and focus on programming activities in the NetsBlox and Python programming languages. Summer camp teachers were prepared in a week-long pedagogy and peer-teaching centered professional development program where they concurrently learned and practiced teaching the curriculum to one another. Then, pairs of teachers led students in learning through hands-on AI and ML activities in a half-day, two-week summer camp.  In this paper, we discuss the curriculum development and implementation, as well as survey feedback from both teachers and students.

----

## [1431] Game Design for Better Security of Combination Locks

**Authors**: *Jean-Pierre Astudillo Guerra, Karim Ahmed, Ryan Maher, Eddie Ubri, Jeremy Blum*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21547](https://doi.org/10.1609/aaai.v36i11.21547)

**Abstract**:

Dial locks are commonly used to secure a person’s items. Commercially available dial locks often use four or five wheels of letters, allowing a user to select a word as a combination. In order to evaluate the security of these locks, we create a game, with an instance created by the lock designer, and played by a lock owner and a thief. In the game, the lock owner chooses a word as a combination, and the thief creates a brute force strategy to try all possible combinations that yield words until the combination is found. To accomplish the task, the thief will solve a version of the Probabilistic Travelling Salesman Problem (PTSP) by creating an a priori tour through all the words a lock can create. The goal for the game designer, then, is to create a lock configuration that maximizes the expected length of the best possible PTSP tour. This paper describes a Genetic Algorithm (GA) approach to design a near-optimal game, i.e. a lock configuration that makes it as difficult for the thief to crack. An analysis of the output of the GA shows that the locks that the system creates are significantly more secure than both commercial locks, in the context of this game.

----

## [1432] Interactive Visualizations of Word Embeddings for K-12 Students

**Authors**: *Saptarashmi Bandyopadhyay, Jason Xu, Neel Pawar, David S. Touretzky*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21548](https://doi.org/10.1609/aaai.v36i11.21548)

**Abstract**:

Word embeddings, which represent words as dense feature vectors, are widely used in natural language processing. In their seminal paper on word2vec, Mikolov and colleagues showed that a feature space created by training a word prediction network on a large text corpus will encode semantic information that supports analogy by vector arithmetic, e.g., "king" minus "man" plus "woman" equals "queen". To help novices appreciate this idea, people have sought effective graphical representations of word embeddings.

We describe a new interactive tool for visually exploring word embeddings. Our tool allows users to define semantic dimensions by specifying opposed word pairs, e.g., gender is defined by pairs such as boy/girl and father/mother, and age by pairs such as father/son and mother/daughter. Words are plotted as points in a zoomable and rotatable 3D space, where the third ”residual” dimension encodes distance from the hyperplane defined by all the opposed word vectors with age and gender subtracted out. Our tool allows users to visualize vector analogies, drawing the vector from “king” to “man” and a parallel vector from “woman” to “king-man+woman”, which is closest to “queen”. Visually browsing the embedding space and experimenting with this tool can make word embeddings more intuitive. We include a series of experiments teachers can use to help K-12 students appreciate the strengths and limitations of this representation.

----

## [1433] Fast Heuristic Detection of Offensive Words in Wordwheel Puzzles

**Authors**: *Anand D. Blum, R. Mitchell Parry*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21549](https://doi.org/10.1609/aaai.v36i11.21549)

**Abstract**:

Offensive words appear in Wordwheel-type puzzles with a high frequency. Previous approaches to eliminating these words have focused largely on eliminating puzzles that might give rise to an offensive word. This work presents a fast, heuristic approach to detecting an offensive word within a puzzle. After a preprocessing stage, the detection occurs with a single bitwise operation on a 64-bit word. Tests show that as long as there are at least 3 taboo words possible in a puzzle, the heuristic approach is faster than a depth-first search of the puzzle. In addition to being fast, the approach is guaranteed to detect all offensive words, and has a low false positive rate.

----

## [1434] Ludus: An Optimization Framework to Balance Auto Battler Cards

**Authors**: *Nathaniel Budijono, Phoebe Goldman, Jack Maloney, Joseph B. Mueller, Phillip Walker, Jack Ladwig, Richard G. Freedman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21550](https://doi.org/10.1609/aaai.v36i11.21550)

**Abstract**:

Auto battlers are a recent genre of online deck-building games where players choose and arrange cards that then compete against other players' cards in fully-automated battles. As in other deck-building games, such as trading card games, designers must balance the cards to permit a wide variety of competitive strategies.  We present Ludus, a framework that combines automated playtesting with global search to optimize parameters for each card that will assist designers in balancing new content.  We develop a sampling-based approximation to reduce the playtesting needed during optimization.  To guide the global search, we define metrics characterizing the health of the metagame and explore their impacts on the results of the optimization process.  Our research focuses on an auto battler game we designed for AI research, but our approach is applicable to other auto battler games.

----

## [1435] Predictive Student Modelling in an Online Reading Platform

**Authors**: *Effat Farhana, Teomara Rutherford, Collin F. Lynch*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21551](https://doi.org/10.1609/aaai.v36i11.21551)

**Abstract**:

Use of technology-enhanced education and online learning systems has become more popular, especially after COVID-19. These systems capture a rich array of data as students interact with them. Predicting student performance is an essential part of technology-enhanced education systems to enable the generation of hints and provide recommendations to students. Typically, this is done through use of data on student interactions with questions without utilizing important data on the temporal ordering of students’ other interaction behavior, (e.g., reading, video watching). In this paper, we hypothesize that to predict students’ question performance, it is necessary to (i) consider other learning activities beyond question-answering and (ii) understand how these activities are related to question-solving behavior. We collected middle school physical science students’ data within a K12 reading platform, Actively Learn. This platform provides reading-support to students and collects trace data on their use of the system. We propose a transformer-based model to predict students' question scores utilizing question interaction and reading-related behaviors. Our findings show that integrating question attempts and reading-related behaviors results in better predictive power compared to using only question attempt features. The interpretable visualization of the transformer’s attention can be helpful for teachers to make tailored interventions in students’ learning.

----

## [1436] Game Balancing in Dominion: An Approach to Identifying Problematic Game Elements

**Authors**: *Cassandra Ford, Merrick Ohata*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21552](https://doi.org/10.1609/aaai.v36i11.21552)

**Abstract**:

In the popular card game Dominion, the configuration of game elements greatly affects the experience for players. If one were redesigning Dominion, therefore, it may be useful to identify game elements that reduce the number of viable strategies in any given game configuration - i.e. elements that are unbalanced. In this paper, we propose an approach that assigns credit to the outcome of an episode to individual elements. Our approach uses statistical analysis to learn the interactions and dependencies between game elements.  This learned knowledge is used to recommend elements to game designers for further consideration. Designers may then choose to modify the recommended elements with the goal of increasing the number of viable strategies.

----

## [1437] I AM AI Gradient Descent - an Open-Source Digital Game for Inquiry-Based CLIL Learning

**Authors**: *Carina Geldhauser, Andreas Daniel Matt, Christian Stussak*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21553](https://doi.org/10.1609/aaai.v36i11.21553)

**Abstract**:

We present an interactive online workshop for K-12 students, which aims in familiarizing students with core concepts of AI. The workshop consists of a variety of resources, inspired by inquiry-based learning techniques, of which we present in detail one module, centered around a browser-based game called Gradient Descent. This module introduces the mathematical concepts behind a gradient descent-based optimization algorithm through the computer game of a treasure hunt at an unknown sea surface landscape. Finally, we report on student feedback for the module in a series of content and language integrated learning in German (CLiLiG) workshops for students aged 14-17 in 30 countries.

----

## [1438] Smartphone-Based Game Development to Introduce K12 Students in Applied Artificial Intelligence

**Authors**: *Sara Guerreiro-Santalla, Alma Mallo, Tamara Baamonde, Francisco Bellas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21554](https://doi.org/10.1609/aaai.v36i11.21554)

**Abstract**:

This paper presents a structured activity based on a game design to introduce k-12 students in the topic of super-vised machine learning from a practical perspective. The activity has been developed in the scope of an Erasmus+ project called AI+, which aims to develop an AI curriculum for high school students. As established in the AI+ principles, all the teaching activities are based on the use of the student's smartphone as the core element to intro-duce an applied approach to AI in classes. In this case, a smartphone-based game app is developed by students that includes a neural network model obtained with the "Personal Image Classifier" tool of the MIT App Inventor software. From a didactic perspective, the students dealt with supervised learning to solve a problem of image classification. The main learning outcome is the under-standing of how relevant is to develop a reliable machine learning model when dealing with real world applications. This activity was tested during 2021 with more than 50 students belonging to six schools across Europe, all of them enrolled in the AI+ project.

----

## [1439] An Experience Report of Executive-Level Artificial Intelligence Education in the United Arab Emirates

**Authors**: *David Johnson, Mohammad Alsharid, Rasheed El-Bouri, Nigel Mehdi, Farah Shamout, Alexandre Szenicer, David Toman, Saqr Binghalib*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21555](https://doi.org/10.1609/aaai.v36i11.21555)

**Abstract**:

Teaching artificial intelligence (AI) is challenging. It is a fast moving field and therefore difficult to keep people updated with the state-of-the-art. Educational offerings for students are ever increasing, beyond university degree programs where AI education traditionally lay. In this paper, we present an experience report of teaching an AI course to business executives in the United Arab Emirates (UAE). Rather than focusing only on theoretical and technical aspects, we developed a course that teaches AI with a view to enabling students to understand how to incorporate it into existing business processes. We present an overview of our course, curriculum and teaching methods, and we discuss our reflections on teaching adult learners, and to students in the UAE.

----

## [1440] Authentic Integration of Ethics and AI through Sociotechnical, Problem-Based Learning

**Authors**: *Ari Krakowski, Eric Greenwald, Timothy Hurt, Brandie Nonnecke, Matthew A. Cannady*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21556](https://doi.org/10.1609/aaai.v36i11.21556)

**Abstract**:

Growing awareness of both the demand for artificial intelligence (AI) expertise and of the societal impacts of AI systems has led to calls to integrate learning of ethics alongside learning of technical skills in AI courses and pathways. In this paper, we discuss our experiences developing and piloting the TechHive:AI curriculum for high school youth that integrates AI ethics and technical learning. The design of the curriculum was guided by the following pedagogical goals: (1) to respond to the capacity-building need for critical sociotechnical competencies in AI workforce pathways; and (2) to broaden participation in AI pathways through intentional instructional design to center equity in learning experiences. We provide an overview of the 30-hour learning sequence’s instructional design, and our “4D Framework,” which we use as a heuristic to help students conceptualize and inspect AI systems. We then provide a focused description of one of three 8-hour modules that make up the sequence. Finally, we present evidence of promise from an exploratory study of TechHive:AI with a small sample of students, and discuss insights from implementation, including from our use of established resources for AI learning within the learning sequence as well as those created by our team.

----

## [1441] Preparing High School Teachers to Integrate AI Methods into STEM Classrooms

**Authors**: *Irene Lee, Beatriz Perret*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21557](https://doi.org/10.1609/aaai.v36i11.21557)

**Abstract**:

In this experience report, we describe an Artificial Intelligence (AI) Methods in Data Science (DS) curriculum and professional development (PD) program designed to prepare high school teachers with AI content knowledge and an understanding of the ethical issues posed by bias in AI to support their integration of AI methods into existing STEM classrooms. The curriculum consists of 5-day units on Data Analytics, Decision trees, Machine Learning, Neural Networks, and Transfer learning that follow a scaffolded learning progression consisting of introductions to concepts grounded in everyday experiences, hands-on activities, interactive web-based tools, and inspecting and modifying the code used to build, train and test AI models within Google Colab notebooks. The participants in the PD program were secondary school teachers from the Southwest and North-east regions of the United States who represented a variety of STEM disciplines: Biology, Chemistry, Physics, Engi-neering, and Mathematics. We share findings on teacher outcomes from the implementation of two one-week PD workshops during the summer of 2021 and share suggestions for improvements provided by teachers. We conclude with a discussion of affordances and challenges encountered in preparing teachers to integrate AI education into disciplinary classrooms.

----

## [1442] Reproducibility as a Mechanism for Teaching Fairness, Accountability, Confidentiality, and Transparency in Artificial Intelligence

**Authors**: *Ana Lucic, Maurits J. R. Bleeker, Sami Jullien, Samarth Bhargav, Maarten de Rijke*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21558](https://doi.org/10.1609/aaai.v36i11.21558)

**Abstract**:

In this work, we explain the setup for a technical, graduate-level course on Fairness, Accountability, Confidentiality, and Transparency in Artificial Intelligence (FACT-AI) at the University of Amsterdam, which teaches FACT-AI concepts through the lens of reproducibility. 
The focal point of the course is a group project based on reproducing existing FACT-AI algorithms from top AI conferences and writing a corresponding report. 
In the first iteration of the course, we created an open source repository with the code implementations from the group projects. 
In the second iteration, we encouraged students to submit their group projects to the Machine Learning Reproducibility Challenge, resulting in 9 reports from our course being accepted for publication in the ReScience journal. 
We reflect on our experience teaching the course over two years, where one year coincided with a global pandemic, and propose guidelines for teaching FACT-AI through reproducibility in graduate-level AI study programs. 
We hope this can be a useful resource for instructors who want to set up similar courses in the future.

----

## [1443] Introducing Variational Autoencoders to High School Students

**Authors**: *Zhuoyue Lyu, Safinah Ali, Cynthia Breazeal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21559](https://doi.org/10.1609/aaai.v36i11.21559)

**Abstract**:

Generative Artificial Intelligence (AI) models are a compelling way to introduce K-12 students to AI education using an artistic medium, and hence have drawn attention from K-12 AI educators. Previous Creative AI curricula mainly focus on Generative Adversarial Networks (GANs) while paying less attention to Autoregressive Models, Variational Autoencoders (VAEs), or other generative models, which have since become common in the field of generative AI. VAEs' latent-space structure and interpolation ability could effectively ground the interdisciplinary learning of AI, creative arts, and philosophy. Thus, we designed a lesson to teach high school students about VAEs. We developed a web-based game and used Plato's cave, a philosophical metaphor, to introduce how VAEs work. We used a Google Colab notebook for students to re-train VAEs with their hand-written digits to consolidate their understandings. Finally, we guided the exploration of creative VAE tools such as SketchRNN and MusicVAE to draw the connection between what they learned and real-world applications. This paper describes the lesson design and shares insights from the pilot studies with 22 students. We found that our approach was effective in teaching students about a novel AI concept.

----

## [1444] Interpretable Knowledge Tracing: Simple and Efficient Student Modeling with Causal Relations

**Authors**: *Sein Minn, Jill-Jênn Vie, Koh Takeuchi, Hisashi Kashima, Feida Zhu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21560](https://doi.org/10.1609/aaai.v36i11.21560)

**Abstract**:

Intelligent Tutoring Systems have become critically important in future learning environments. Knowledge Tracing (KT) is a crucial part of that system. It is about inferring the skill mastery of students and predicting their performance to adjust the curriculum accordingly. Deep Learning based models like Deep Knowledge Tracing (DKT) and Dynamic Key-Value Memory Network (DKVMN) have shown significant predictive performance compared with traditional models like Bayesian Knowledge Tracing (BKT) and Performance Factors Analysis (PFA). However, it is difficult to extract psychologically meaningful explanations from the tens of thousands of parameters in neural networks, that would relate to cognitive theory. There are several ways to achieve high accuracy in student performance prediction but diagnostic and prognostic reasonings are more critical in learning science.
In this work, we present Interpretable Knowledge Tracing (IKT), a simple model that relies on three meaningful features: individual skill mastery, ability profile (learning transfer across skills) and problem difficulty by using data mining techniques.
IKT’s prediction of future student performance is made using a Tree Augmented Naive Bayes Classifier (TAN), therefore its predictions are easier to explain than deep learning based student models. IKT also shows better student performance prediction than deep learning based student models without requiring a huge amount of parameters. We conduct ablation studies on each feature to examine their contribution to student performance prediction. Thus, IKT has great potential for providing adaptive and personalized instructions with causal reasoning in real-world educational systems.

----

## [1445] The Bullets Puzzle: A Paper-and-Pencil Minesweeper

**Authors**: *Todd W. Neller, Hien G. Tran*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21561](https://doi.org/10.1609/aaai.v36i11.21561)

**Abstract**:

In this paper, we introduce a technique for AI generation of the Bullets puzzle, a paper-and-pencil variant of Minesweeper. Whereas traditional Minesweeper can be lost due to the need to guess mine or non-mine positions, our puzzle is fully deducible from a minimal clue set.  Puzzle generation is based on analysis and optimization of solutions from a human-like reasoning engine that classifies types of deductions.  Additionally, we provide insights to subjective puzzle quality, minimal clue sampling trade-offs, and optimal bullet density.

----

## [1446] DeepQR: Neural-Based Quality Ratings for Learnersourced Multiple-Choice Questions

**Authors**: *Lin Ni, Qiming Bao, Xiaoxuan Li, Qianqian Qi, Paul Denny, Jim Warren, Michael Witbrock, Jiamou Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21562](https://doi.org/10.1609/aaai.v36i11.21562)

**Abstract**:

Automated question quality rating (AQQR) aims to evaluate question quality through computational means, thereby addressing emerging challenges in online learnersourced question repositories. Existing methods for AQQR rely solely on explicitly-defined criteria such as readability and word count, while not fully utilising the power of state-of-the-art deep-learning techniques. We propose DeepQR, a novel neural-network model for AQQR that is trained using multiple-choice-question (MCQ) datasets collected from PeerWise, a widely-used learnersourcing platform. Along with designing DeepQR, we investigate models based on explicitly-defined features, or semantic features, or both. We also introduce a self-attention mechanism to capture semantic correlations between MCQ components, and a contrastive-learning approach to acquire question representations using quality ratings. Extensive experiments on datasets collected from eight university-level courses illustrate that DeepQR has superior performance over six comparative models.

----

## [1447] Using Sampling to Estimate and Improve Performance of Automated Scoring Systems with Guarantees

**Authors**: *Yaman Kumar Singla, Sriram Krishna, Rajiv Ratn Shah, Changyou Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21563](https://doi.org/10.1609/aaai.v36i11.21563)

**Abstract**:

Automated Scoring (AS), the natural language processing task of scoring essays and speeches in an educational testing setting, is growing in popularity and being deployed across contexts from government examinations to companies providing language proficiency services. However, existing systems either forgo human raters entirely, thus harming the reliability of the test, or score every response by both human and machine thereby increasing costs. We target the spectrum of possible solutions in between, making use of both humans and machines to provide a higher quality test while keeping costs reasonable to democratize access to AS. In this work, we propose a combination of the existing paradigms, sampling responses to be scored by humans intelligently. We propose reward sampling and observe significant gains in accuracy (19.80% increase on average) and quadratic weighted kappa (QWK) (25.60% on average) with a relatively small human budget (30% samples) using our proposed sampling. The accuracy increase observed using standard random and importance sampling baselines are 8.6% and 12.2% respectively. Furthermore, we demonstrate the system's model agnostic nature by measuring its performance on a variety of models currently deployed in an AS setting as well as pseudo models. Finally, we propose an algorithm to estimate the accuracy/QWK with statistical guarantees (Our code is available at https://git.io/J1IOy).

----

## [1448] Artificial Intelligence Approaches to Build Ticket to Ride Maps

**Authors**: *Iain Smith, Calin Anton*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21564](https://doi.org/10.1609/aaai.v36i11.21564)

**Abstract**:

Fun, as a game trait, is challenging to evaluate. Previous research explores game arc and game refinement to improve the quality of games. Fun, for some players, is having an even chance to win while executing their strategy. To explore this, we build boards for the game Ticket to Ride while optimizing for a given win rate between four AI agents. These agents execute popular strategies human players use: one-step thinking, long route exploitation, route focus, and destination hungry strategies. We create the underlying graph of a map by connecting several planar bipartite graphs. To build the map, we use a multiple phase design, with each phase implementing several simplified Monte Carlo Tree Search components. Within a phase, the components communicate with each other passively. The experiments show that the proposed approach results in improvements over randomly generated graphs and maps.

----

## [1449] Paving the Way for Novices: How to Teach AI for K-12 Education in China

**Authors**: *Jiachen Song, Linan Zhang, Jinglei Yu, Yan Peng, Anyao Ma, Yu Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21565](https://doi.org/10.1609/aaai.v36i11.21565)

**Abstract**:

In response to the trend that artificial intelligence (AI) is becoming the main driver for social and economic development, enhancing the readiness of learners in AI is significant and important. The state council and the ministry of education of China put AI education for K-12 schools on a high priority in order to foster local AI talents and reduce educational disparities. However, the AI knowledge and technical skills are still limited for not only students but also the school teachers. Furthermore, many local schools in China, especially in the rural areas, are lack of the necessary software and hardware for teaching AI. Hence, we designed and implemented a structured series of AI courses, built on an online block-based visual programming platform. The AI courses are free and easily accessible for all. We have conducted the experimental classes in a local school and collected the results. The results show that the learners in general gained significant learning progress on AI knowledge comprehension, aroused strong interests in AI, and increased the degree of satisfaction towards the course. Especially, our practices significantly increased computational thinking of the students who were initially staying at a lower level.

----

## [1450] Teaching AI with the Hands-On AI Projects for the Classroom Series

**Authors**: *Nancye Blair Black*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21566](https://doi.org/10.1609/aaai.v36i11.21566)

**Abstract**:

The Hands-On AI Projects for the Classroom series, a collection of five guides, includes interactive projects that can be used by teachers across grade levels and subject areas to teach K-12 students about artificial intelligence (AI).

----

## [1451] StoryQ - an Online Environment for Machine Learning of Text Classification

**Authors**: *William Finzer, Jie Chao, Carolyn P. Rosé, Shiyan Jiang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21567](https://doi.org/10.1609/aaai.v36i11.21567)

**Abstract**:

The StoryQ environment provides an intuitive graphical user interface for middle and high school students to create features from unstructured text data and train and test classification models using logistic regression. StoryQ runs in a web browser, is free and requires no installation. AI concepts addressed include: features, weights, accuracy, training, bias, error analysis and cross validation. Using the software in conjunction with curriculum currently under development is expected to lead to student understanding of machine learning concepts and workflow; developing the ability to use domain knowledge and basic linguistics to identify, create, analyze, and evaluate features; becoming aware of and appreciating the roles and responsibilities of AI developers;. This paper will consist of an online demo with a brief video walkthrough.

----

## [1452] AI Snap! Blocks for Speech Input and Output, Computer Vision, Word Embeddings, and Neural Net Creation, Training, and Use

**Authors**: *Ken Kahn, Ramana Prasad, Gayathri Veera*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21568](https://doi.org/10.1609/aaai.v36i11.21568)

**Abstract**:

We will demonstrate blocks integrated into Snap! capable of a wide range of AI services, interactive AI programming guides, and a selection from thirty sample projects. Sessions and workshops in both school settings and informal learning contexts have been held in many countries. The full version of this paper includes descriptions of the Snap! blocks and unpublished descriptions of student experiences in India.

----

## [1453] Model AI Assignments 2022

**Authors**: *Todd W. Neller, Jazmin Collins, Daniel Schneider, Yim Register, Christopher Brooks, Chia-Wei Tang, Chao-Lin Liu, Roozbeh Aliabadi, Annabel Hasty, Sultan Albarakati, Haotian Fang, Harvey Yin, Joel Wilson*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21569](https://doi.org/10.1609/aaai.v36i11.21569)

**Abstract**:

The Model AI Assignments session seeks to gather and disseminate the best assignment designs of the Artificial Intelligence (AI) Education community.  Recognizing that assignments form the core of student learning experience, we here present abstracts of six AI assignments from the 2022 session that are easily adoptable, playfully engaging, and flexible for a variety of instructor needs.  Assignment specifications and supporting resources may be found at http://modelai.gettysburg.edu.

----

## [1454] Towards Robust Named Entity Recognition via Temporal Domain Adaptation and Entity Context Understanding

**Authors**: *Oshin Agarwal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21570](https://doi.org/10.1609/aaai.v36i11.21570)

**Abstract**:

Named Entity Recognition models perform well on benchmark datasets but fail to generalize well even in the same domain. The goal of my th    esis is to quantify the degree of in-domain generalization in NER, probe models for entity name vs. context learning and finally improve     their robustness, focusing on the recognition of ethnically diverse entities and new entities over time when the models are deployed.

----

## [1455] AI-Driven Road Condition Monitoring across Multiple Nations

**Authors**: *Deeksha Arya, Sanjay Kumar Ghosh, Durga Toshniwal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21571](https://doi.org/10.1609/aaai.v36i11.21571)

**Abstract**:

The doctoral work summarized here is an application of Artificial Intelligence (AI) for social good. The successful implementation would contribute towards low-cost, faster monitoring of road conditions across different nations, resulting in safer roads for everyone. Additionally, the study provides recommendations for re-using the road image data and the Deep Learning models released by any country for detecting road damage in other countries.

----

## [1456] Increasing the Diversity of Deep Generative Models

**Authors**: *Sebastian Berns*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21572](https://doi.org/10.1609/aaai.v36i11.21572)

**Abstract**:

Generative models are used in a variety of applications that require diverse output. Yet, models are primarily optimised for sample fidelity and mode coverage. My work aims to increase the output diversity of generative models for multi-solution tasks. Previously, we analysed the use of generative models in artistic settings and how its objective diverges from distribution fitting. For specific use cases, we quantified the limitations of generative models. Future work will focus on adapting generative modelling for downstream tasks that require a diverse set of high-quality artefacts.

----

## [1457] Interpretable Privacy Preservation of Text Representations Using Vector Steganography

**Authors**: *Geetanjali Bihani*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21573](https://doi.org/10.1609/aaai.v36i11.21573)

**Abstract**:

Contextual word representations generated by language models learn spurious associations present in the training corpora. Adversaries can exploit these associations to reverse-engineer the private attributes of entities mentioned in the training corpora. These findings have led to efforts towards minimizing the privacy risks of language models. However, existing approaches lack interpretability, compromise on data utility and fail to provide privacy guarantees. Thus, the goal of my doctoral research is to develop interpretable approaches towards privacy preservation of text representations that maximize data utility retention and guarantee privacy. To this end, I aim to study and develop methods to incorporate steganographic modifications within the vector geometry to obfuscate underlying spurious associations and retain the distributional semantic properties learnt during training.

----

## [1458] Using Multimodal Data and AI to Dynamically Map Flood Risk

**Authors**: *Lydia Bryan-Smith*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21574](https://doi.org/10.1609/aaai.v36i11.21574)

**Abstract**:

Classical measurements and modelling that underpin present flood warning and alert systems are based on fixed and spatially restricted static sensor networks. Computationally expensive physics-based simulations are often used that can't react in real-time to changes in environmental conditions. We want to explore contemporary artificial intelligence (AI) for predicting flood risk in real time by using a diverse range of data sources. By combining heterogeneous data sources, we aim to nowcast rapidly changing flood conditions and gain a grater understanding of urgent humanitarian needs.

----

## [1459] Towards Automating the Generation of Human-Robot Interaction Scenarios

**Authors**: *Matthew C. Fontaine*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21575](https://doi.org/10.1609/aaai.v36i11.21575)

**Abstract**:

My work studies the problem of generating scenarios to evaluate interaction between humans and robots. I expect these interactions to grow in complexity as robots become more intelligent and enter our daily lives. However, evaluating such interactions only through user studies, which are the de facto evaluation method in human-robot interaction, will quickly become infeasible as the number of possible scenarios grows exponentially with scenario complexity. Therefore, I propose automatically generating scenarios in simulation to explore the diverse possibility space of scenarios to better understand interaction and avoid costly failures in real world settings.

----

## [1460] An Algorithmic Theory of Markets and Their Application to Decentralized Markets

**Authors**: *Denizalp Goktas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21576](https://doi.org/10.1609/aaai.v36i11.21576)

**Abstract**:

Broadly speaking, I hope to dedicate my PhD to improving our understanding of algorithmic economics with the ultimate goal of building welfare improving decentralized technology for markets. In the following pages, I describe how my past work has built on the existing literature to get closer to the goal of creating such technologies, and describe what research paths this work opens up for the rest of my PhD. I believe that my research has the potential to provide algorithmic solutions to problems in machine learning, optimization, and game theory, and can be used to improve the efficiency of online marketplaces.

----

## [1461] Evaluating Explanations of Relational Graph Convolutional Network Link Predictions on Knowledge Graphs

**Authors**: *Nicholas Halliwell*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21577](https://doi.org/10.1609/aaai.v36i11.21577)

**Abstract**:

Recently, explanation methods have been proposed to evaluate the predictions of Graph Neural Networks on the task of link prediction. Evaluating explanation quality is difficult without ground truth explanations. This thesis is focused on providing a method, including datasets and scoring metrics, to quantitatively evaluate explanation methods on link prediction on Knowledge Graphs.

----

## [1462] Equilibrium Learning in Auction Markets

**Authors**: *Stefan Heidekrüger*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21578](https://doi.org/10.1609/aaai.v36i11.21578)

**Abstract**:

My dissertation investigates the computation of Bayes-Nash equilibria in auctions via multiagent learning. A particular focus lies on the game-theoretic analysis of learned gradient dynamics in such markets. This requires overcoming several technical challenges like non-differentiable utility functions and infinite-dimensional strategy spaces. Positive results may open the door for wide-ranging applications in Market Design and the economic sciences.

----

## [1463] On the Practical Robustness of the Nesterov's Accelerated Quasi-Newton Method

**Authors**: *S. Indrapriyadarsini, Hiroshi Ninomiya, Takeshi Kamio, Hideki Asai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21579](https://doi.org/10.1609/aaai.v36i11.21579)

**Abstract**:

This study focuses on the Nesterov's accelerated quasi-Newton (NAQ) method in the context of deep neural networks (DNN) and its applications. The thesis objective is to confirm the robustness and efficiency of Nesterov's acceleration to quasi-Netwon (QN) methods by developing practical algorithms for different fields of optimization problems.

----

## [1464] Creating Interactive Crowds with Reinforcement Learning

**Authors**: *Ariel Kwiatkowski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21580](https://doi.org/10.1609/aaai.v36i11.21580)

**Abstract**:

The entertainment industry, as well as the field of Computer Graphics, frequently faces the issue of creating large virtual crowds that would populate a scene. One of the ways to achieve that, particularly with modern rendering techniques, is by using simulation -- this, however, is nontrivial to design and control. The main goal of my PhD work is working towards the creation of a tool enabling the creation of virtual crowds that one can interact with, and we believe the best way to that is through Multiagent Reinforcement Learning techniques. These animated crowds can then be used both in movies and video games. Especially for the latter, it is highly desirable that both the crowd as a whole, as well as the individual characters, can react to the user's input in real time.

----

## [1465] Socially Intelligent Affective AI

**Authors**: *Aarti Malhotra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21581](https://doi.org/10.1609/aaai.v36i11.21581)

**Abstract**:

Artificial Intelligence has aimed to give the systems or agents, the ability to learn, perceive, recognize, plan, reason and act. Affective Computing has brought into focus the importance of giving AI systems, the capability to perceive, detect, utilize and generate emotion, affect, sentiment or feelings. To have a meaningful human-computer interaction, we need to design and develop a more socially intelligent and affective AI. My doctoral research goal is to delve deeper into some of these aspects, firstly by surveying computational models implemented in AI that uses emotion in decision-making or behaviour; secondly, by creating new model to predict social event context and affect in group videos; thirdly, to predict the social identities in visual scenes; and lastly to combine information about context, identities, behaviour and emotion in a social interaction scene to predict social incoherence and to recommend appropriate behaviour.

----

## [1466] Dynamic Algorithmic Impact Assessment to Promote an Ethical Use of AI in Businesses

**Authors**: *Shefeh Prisilia Mbuy*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21582](https://doi.org/10.1609/aaai.v36i11.21582)

**Abstract**:

My PhD research focus is to produce a critical review of literature in Algorithmic Impact Assessment (AIA) and to develop an AIA tool that can be used to evaluate potential unintended impact of AI systems.

----

## [1467] Creating Interpretable Data-Driven Approaches for Tropical Cyclones Forecasting

**Authors**: *Fan Meng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21583](https://doi.org/10.1609/aaai.v36i11.21583)

**Abstract**:

Tropical cyclones (TC) are extreme weather phenomena that bring heavy disasters to humans. Existing forecasting techniques contain computationally intensive dynamical models and statistical methods with complex inputs, both of which have bottlenecks in intensity forecasting, and we aim to create data-driven methods to break this forecasting bottleneck. The research goal of my PhD topic is to introduce novel methods to provide accurate and trustworthy forecasting of TC by developing interpretable machine learning models to analyze the characteristics of TC from multiple sources of data such as satellite remote sensing and observations.

----

## [1468] On Semantic Cognition, Inductive Generalization, and Language Models

**Authors**: *Kanishka Misra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21584](https://doi.org/10.1609/aaai.v36i11.21584)

**Abstract**:

My doctoral research focuses on understanding semantic knowledge in neural network models trained solely to predict natural language (referred to as language models, or LMs), by drawing on insights from the study of concepts and categories grounded in cognitive science. I propose a framework inspired by 'inductive reasoning,' a phenomenon that sheds light on how humans utilize background knowledge to make inductive leaps and generalize from new pieces of information about concepts and their properties. Drawing from experiments that study inductive reasoning, I propose to analyze semantic inductive generalization in LMs using phenomena observed in human-induction literature, investigate inductive behavior on tasks such as implicit reasoning and emergent feature recognition, and analyze and relate induction dynamics to the learned conceptual representation space.

----

## [1469] Mutual Understanding in Human-Machine Teaming

**Authors**: *Rohan R. Paleja*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21585](https://doi.org/10.1609/aaai.v36i11.21585)

**Abstract**:

Collaborative robots (i.e., "cobots") and machine learning-based virtual agents are increasingly entering the human workspace with the aim of increasing productivity, enhancing safety, and improving the quality of our lives. These agents will dynamically interact with a wide variety of people in dynamic and novel contexts, increasing the prevalence of human-machine teams in healthcare, manufacturing, and search-and-rescue. In this research, we enhance the mutual understanding within a human-machine team by enabling cobots to understand heterogeneous teammates via person-specific embeddings, identifying contexts in which xAI methods can help improve team mental model alignment, and enabling cobots to effectively communicate information that supports high-performance human-machine teaming.

----

## [1470] Using Graph-Aware Reinforcement Learning to Identify Winning Strategies in Diplomacy Games (Student Abstract)

**Authors**: *Hansin Ahuja, Lynnette Hui Xian Ng, Kokil Jaidka*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21586](https://doi.org/10.1609/aaai.v36i11.21586)

**Abstract**:

This abstract proposes an approach towards goal-oriented modeling of the detection and modeling complex social phenomena in multiparty discourse in an online political strategy game.
We developed a two-tier approach that first encodes sociolinguistic behavior as linguistic features then use reinforcement learning to estimate the advantage afforded to any player. 
In the first tier, sociolinguistic behavior, such as Friendship and Reasoning, that speakers use to influence others are encoded as linguistic features to identify the persuasive strategies applied by each player in simultaneous two-party dialogues. In the second tier, a reinforcement learning approach is used to estimate a graph-aware reward function to quantify the advantage afforded to each player based on their standing in this multiparty setup. We apply this technique to the game Diplomacy, using a dataset comprising of over 15,000 messages exchanged between 78 users. Our graph-aware approach shows robust performance compared to a context-agnostic setup.

----

## [1471] PESTO: Switching Point Based Dynamic and Relative Positional Encoding for Code-Mixed Languages (Student Abstract)

**Authors**: *Mohsin Ali, Kandukuri Sai Teja, Sumanth Manduru, Parth Patwa, Amitava Das*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21587](https://doi.org/10.1609/aaai.v36i11.21587)

**Abstract**:

NLP applications for code-mixed (CM) or mix-lingual text have gained a significant momentum recently, the main reason being the prevalence of language mixing in social media communications in multi-lingual societies like India, Mexico, Europe, parts of USA etc. Word embeddings are basic building blocks of any NLP system today, yet, word embedding for CM languages is an unexplored territory. The major bottleneck for CM word embeddings is switching points, where the language switches. These locations lack in contextually and statistical systems fail to model this phenomena due to high variance in the seen examples. In this paper we present our initial observations on applying switching point based positional encoding techniques for CM language, specifically Hinglish (Hindi - English). Results are only marginally better than SOTA, but it is evident that positional encoding could be an effective way to train position sensitive language models for CM text.

----

## [1472] A Deep Learning-Based Face Mask Detector for Autonomous Nano-Drones (Student Abstract)

**Authors**: *Eiman AlNuaimi, Elia Cereda, Rafail Psiakis, Suresh Sugumar, Alessandro Giusti, Daniele Palossi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21588](https://doi.org/10.1609/aaai.v36i11.21588)

**Abstract**:

We present a deep neural network (DNN) for visually classifying whether a person is wearing a protective face mask. Our DNN can be deployed on a resource-limited, sub-10-cm nano-drone: this robotic platform is an ideal candidate to fly in human proximity and perform ubiquitous visual perception safely. This paper describes our pipeline, starting from the dataset collection; the selection and training of a full-precision (i.e., float32) DNN; a quantization phase (i.e., int8), enabling the DNN's deployment on a parallel ultra-low power (PULP) system-on-chip aboard our target nano-drone. Results demonstrate the efficacy of our pipeline with a mean area under the ROC curve score of 0.81, which drops by only ~2% when quantized to 8-bit for deployment.

----

## [1473] Learning Modular Structures That Generalize Out-of-Distribution (Student Abstract)

**Authors**: *Arjun Ashok, Chaitanya Devaguptapu, Vineeth N. Balasubramanian*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21589](https://doi.org/10.1609/aaai.v36i11.21589)

**Abstract**:

Out-of-distribution (O.O.D.) generalization remains to be a key challenge for real-world machine learning systems. We describe a method for O.O.D. generalization that, through training, encourages models to only preserve features in the network that are well reused across multiple training domains. Our method combines two complementary neuron-level regularizers with a probabilistic differentiable binary mask over the network, to extract a modular sub-network that achieves better O.O.D. performance than the original network. Preliminary evaluation on two benchmark datasets corroborates the promise of our method.

----

## [1474] Manipulating SHAP via Adversarial Data Perturbations (Student Abstract)

**Authors**: *Hubert Baniecki, Przemyslaw Biecek*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21590](https://doi.org/10.1609/aaai.v36i11.21590)

**Abstract**:

We introduce a model-agnostic algorithm for manipulating SHapley Additive exPlanations (SHAP) with perturbation of tabular data. It is evaluated on predictive tasks from healthcare and financial domains to illustrate how crucial is the context of data distribution in interpreting machine learning models. Our method supports checking the stability of the explanations used by various stakeholders apparent in the domain of responsible AI; moreover, the result highlights the explanations' vulnerability that can be exploited by an adversary.

----

## [1475] Multi-Dimension Attention for Multi-Turn Dialog Generation (Student Abstract)

**Authors**: *Billal Belainine, Fatiha Sadat, Mounir Boukadoum*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21591](https://doi.org/10.1609/aaai.v36i11.21591)

**Abstract**:

We present a generative neural model for open and multi-turn dialog response generation that relies on a multi-dimension attention process to account for the semantic interdependence between the generated words and the conversational history, so as to identify all the words and utterances that influence each generated response. 
The performance of the model is evaluated on the wide scope DailyDialog corpus and a comparison is made with two other generative neural architectures, using several machine metrics. The results show that the proposed model improves the state of the art for generation accuracy, and its multi-dimension attention allows for a more detailed tracking of the influential words and utterances in the dialog history for response explainability by the dialog history.

----

## [1476] Deep Learning Based Side Channel Attacks on Lightweight Cryptography (Student Abstract)

**Authors**: *Alexander Benjamin, Jack Herzoff, Liljana Babinkostova, Edoardo Serra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21592](https://doi.org/10.1609/aaai.v36i11.21592)

**Abstract**:

Computing devices continue to be increasingly spread out within our everyday environments. Computers are embedded into everyday devices in order to serve the functionality of electronic components or to enable new services in their own right. Existing Substitution-Permutation Network (SPN) ciphers, such as the Advanced Encryption Standard (AES), are not suitable for devices where memory, power consumption or processing power is limited. Lightweight SPN ciphers, such as GIFT-128 provide a solution for running cryptography on low resource devices. The GIFT-128 cryptographic scheme is a building block for GIFT-COFB (Authenticated Encryption with Associated Data), one of the  finalists in the ongoing NIST lightweight cryptography standardization process (NISTIR 8369). Determination of an adequate level of security and providing subsequent mechanisms to achieve it, is one of the most pressing problems regarding embedded computing devices. In this paper we present experimental results and comparative study of Deep Learning (DL) based Side Channel Attacks on lightweight GIFT-128. To our knowledge, this is the first study of the security of GIFT-128 against DL-based SCA attacks.

----

## [1477] Annotation Cost-Sensitive Deep Active Learning with Limited Data (Student Abstract)

**Authors**: *Renaud Bernatchez, Audrey Durand, Flavie Lavoie-Cardinal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21593](https://doi.org/10.1609/aaai.v36i11.21593)

**Abstract**:

Deep learning is a promising avenue to automate tedious analysis tasks in biomedical imaging. However, its application in such a context is limited by the large amount of labeled data required to train deep learning models. While active learning may be used to reduce the amount of labeling data, many approaches do not consider the cost of annotating, which is often significant in a biomedical imaging setting. In this work we show how annotation cost can be considered and learned during active learning on a classification task on the MNIST dataset.

----

## [1478] INDEPROP: Information-Preserving De-propagandization of News Articles (Student Abstract)

**Authors**: *Aaryan Bhagat, Faraaz Mallick, Neel Karia, Ayush Kaushal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21594](https://doi.org/10.1609/aaai.v36i11.21594)

**Abstract**:

We propose INDEPROP, a novel Natural Language Processing (NLP) application for combating online disinformation by mitigating propaganda from news articles. INDEPROP (Information-Preserving De-propagandization) involves fine-grained propaganda detection and its removal while maintaining document level coherence, grammatical correctness and most importantly, preserving the news articles’ information content. We curate the first large-scale dataset of its kind consisting of around 1M tokens. We also propose a set of automatic evaluation metrics for the same and observe its high correlation with human judgment. Furthermore, we show that fine-tuning the existing propaganda detection systems on our dataset considerably improves their generalization to the test set.

----

## [1479] A Multimodal Fusion-Based LNG Detection for Monitoring Energy Facilities (Student Abstract)

**Authors**: *Junchi Bin, Choudhury A. Rahman, Shane Rogers, Shan Du, Zheng Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21595](https://doi.org/10.1609/aaai.v36i11.21595)

**Abstract**:

Fossil energy products such as liquefied natural gas (LNG) are among Canada's most important exports. Canadian engineers devote themselves to constructing visual surveillance systems for detecting potential LNG emissions in energy facilities. Beyond the previous infrared (IR) surveillance system, in this paper, a multimodal fusion-based LNG detection (MFLNGD) framework is proposed to enhance the detection quality by the integration of IR and visible (VI) cameras. Besides, a Fourier transformer is developed to fuse IR and VI features better. The experimental results suggest the effectiveness of the proposed framework.

----

## [1480] Controlling the Spread of Two Secrets in Diverse Social Networks (Student Abstract)

**Authors**: *Václav Blazej, Dusan Knop, Simon Schierreich*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21596](https://doi.org/10.1609/aaai.v36i11.21596)

**Abstract**:

Information diffusion in social networks is a well-studied concept in social choice theory. We propose the study of the diffusion of two secrets in a heterogeneous environment from the complexity perspective, that is, there are two different networks with the same set of agents (e.g., the structure of the set of followers might be different in two distinct social networks).

Formally, our model combines two group identification processes for which we do have independent desiderata---either constructive, where we would like a given group of agents to be exposed to a secret, or destructive, where a given group of agents should not be exposed to a secret. To be able to reach these targets, we can either delete an agent or introduce a previously latent agent.

Our results are mostly negative---all of the problems are NP-hard. Therefore, we propose a parameterized study with respect to the natural parameters, the number of influenced agents, the size of the required/protected agent sets, and the duration of the diffusion process. Most of the studied problems remain W[1]-hard even for a combination of these parameters. We complement these results with nearly optimal XP algorithms.

----

## [1481] Bridging the Gap between Expression and Scene Text for Referring Expression Comprehension (Student Abstract)

**Authors**: *Yuqi Bu, Jiayuan Xie, Liuwu Li, Qiong Liu, Yi Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21597](https://doi.org/10.1609/aaai.v36i11.21597)

**Abstract**:

Referring expression comprehension aims at grounding the object in an image referred to by the expression. Scene text that serves as an identifier has a natural advantage in referring to objects. However, existing methods only consider the text in the expression, but ignore the text in the image, leading to a mismatch. In this paper, we propose a novel model that can recognize the scene text. We assign the extracted scene text to its corresponding visual region and ground the target object guided by expression. Experimental results on two benchmarks demonstrate the effectiveness of our model.

----

## [1482] Numerical Approximations of Log Gaussian Cox Process (Student Abstract)

**Authors**: *Francois Buet-Golfouse, Hans Roggeman*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21598](https://doi.org/10.1609/aaai.v36i11.21598)

**Abstract**:

This paper considers a multi-state Log Gaussian Cox Process (`"LGCP'') on a graph, where transmissions amongst states are calibrated using a non-parametric approach. We thus consider multi-output LGCPs and introduce numerical approximations to compute posterior distributions extremely quickly and in a completely transparent and reproducible fashion. The model is tested on historical data and shows very good performance.

----

## [1483] Thrifty Neural Architecture Search for Medical Image Segmentation (Student Abstract)

**Authors**: *Ruibin Chen, Miao Zhang, Xin Zheng, Shirui Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21599](https://doi.org/10.1609/aaai.v36i11.21599)

**Abstract**:

Convolutional neural network (CNN) based image segmentation has been widely used in analyzing medical images and benefited many real-world disease diagnosis applications. However, existing advanced CNN-based medical image segmentation models usually contain numerous parameters that require massive computation and memory, limiting the applicability of these models in the data-constrained or hardware-constrained environments. By leveraging the recently proposed neural architecture search (NAS), this paper presents a novel approach, dubbed Thrifty NAS, to design computation and memory-efficient models for medical image segmentation automatically. The searched models by Thrifty NAS are with much fewer parameters while retaining competitive performance. More specifically, we design a micro level space for cell structure search and a macro level cell path for better network structure modeling. Extensive experimental results in different medical image datasets verify the effectiveness of the proposed method with competitive segmentation performance,  especially with minuscule neural architecture model size, i.e., 0.61M that is superior to U-Net (7.76 M) and UNet++ (9.04 M).

----

## [1484] Learning Contrastive Multi-View Graphs for Recommendation (Student Abstract)

**Authors**: *Zhangtao Cheng, Ting Zhong, Kunpeng Zhang, Joojo Walker, Fan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21600](https://doi.org/10.1609/aaai.v36i11.21600)

**Abstract**:

This paper exploits self-supervised learning (SSL) to learn more accurate and robust representations from the user-item interaction graph. Particularly, we propose a novel SSL model that effectively leverages contrastive multi-view learning and pseudo-siamese network to construct a pre-training and post-training framework. Moreover, we present three graph augmentation techniques during the pre-training stage and explore the effects of combining different augmentations, which allow us to learn general and robust representations for the GNN-based recommendation. Simple experimental evaluations on real-world datasets show that the proposed solution significantly improves the recommendation accuracy, especially for sparse data, and is also noise resistant.

----

## [1485] An Emotion-Based Multi-Task Approach to Fake News Detection (Student Abstract)

**Authors**: *Arjun Choudhry, Inder Khatri, Minni Jain*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21601](https://doi.org/10.1609/aaai.v36i11.21601)

**Abstract**:

Social media, blogs, and online articles are instant sources of news for internet users globally. But due to their unmoderated nature, a significant percentage of these texts are fake news or rumors. Their deceptive nature and ability to propagate instantly can have an adverse effect on society. In this work, we hypothesize that legitimacy of news has a correlation with its emotion, and propose a multi-task framework predicting both the emotion and legitimacy of news. Experimental results verify that our multi-task models outperform their single-task counterparts in terms of accuracy.

----

## [1486] Does the Geometry of the Data Control the Geometry of Neural Predictions? (Student Abstract)

**Authors**: *Anirudh Cowlagi, Pratik Chaudhari*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21602](https://doi.org/10.1609/aaai.v36i11.21602)

**Abstract**:

This paper studies the over-parameterization of deep neural networks using the Fisher Information Matrix from information geometry. We identify several surprising trends in the structure of its eigenspectrum, and how this structure relates to the eigenspectrum of the data correlation matrix. We identify how the eigenspectrum relates to the topology of the predictions of the model and develop a "model reduction'' method for deep networks. This ongoing investigation hypothesizes certain universal trends in the FIM of deep networks that may shed light on their effectiveness.

----

## [1487] Transformation of Emotions in Images Using Poisson Blended Generative Adversarial Networks (Student Abstract)

**Authors**: *Aristidis Dernelakis, Jungin Kim, Kevin Velasquez, Lee Stearns*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21603](https://doi.org/10.1609/aaai.v36i11.21603)

**Abstract**:

We propose a novel method for transforming the emotional content in an image to a specified target emotion. Existing techniques such as a single generative adversarial network (GAN) struggle to perform well on unconstrained images, especially when data is limited. Our method addresses this limitation by blending the outputs from two networks to better transform fine details (e.g., faces) while still operating on the broader styles of the full image. We demonstrate our method's potential through a proof-of-concept implementation.

----

## [1488] An Optimal Transport Approach to Deep Metric Learning (Student Abstract)

**Authors**: *Jason Xiaotian Dou, Lei Luo, Raymond Mingrui Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21604](https://doi.org/10.1609/aaai.v36i11.21604)

**Abstract**:

Capturing visual similarity among images is the core of many computer vision and pattern recognition tasks. This problem can be formulated in such a paradigm called metric learning. Most research in the area has been mainly focusing on improving the loss functions and similarity measures. However, due to the ignoring of geometric structure, existing methods often lead to sub-optimal results. Thus, several recent research methods took advantage of Wasserstein distance between batches of samples to characterize the spacial geometry. Although these approaches can achieve enhanced performance, the aggregation over batches definitely hinders Wasserstein distance's superior measure capability and leads to high computational complexity. To address this limitation, we propose a novel Deep Wasserstein Metric Learning framework, which employs Wasserstein distance to precisely capture the relationship among various images under ranking-based loss functions such as contrastive loss and triplet loss. Our method directly computes the distance between images, considering the geometry at a finer granularity than batch level. Furthermore, we introduce a new efficient algorithm using Sinkhorn approximation and Wasserstein measure coreset. The experimental results demonstrate the improvements of our framework over various baselines in different applications and benchmark datasets.

----

## [1489] Transformer-Based Unsupervised Learning for Early Detection of Sepsis (Student Abstract)

**Authors**: *Yutao Dou, Wei Li, Albert Y. Zomaya*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21605](https://doi.org/10.1609/aaai.v36i11.21605)

**Abstract**:

A 6-hour early detection of sepsis leads to a significant increase in the chance of surviving it.
Previous sepsis early detection studies have focused on improving the performance of supervised learning algorithms while ignoring the potential correlation in data mining, and there was no reliable method to deal with the problem of incomplete data.
In this paper, we proposed the Denoising Transformer AutoEncoder (DTAE) for the first time combining transformer and unsupervised learning. 
DTAE can learn the correlation of the features required for early detection of sepsis without the label. 
This method can effectively solve the problems of data sparsity and noise and discover the potential correlation of features by adding DTAE enhancement module without modifying the existing algorithms.
Finally, the experimental results show that the proposed method improves the existing algorithms and achieves the best results of early detection.

----

## [1490] Visual Explanations for Convolutional Neural Networks via Latent Traversal of Generative Adversarial Networks (Student Abstract)

**Authors**: *Amil Dravid, Aggelos K. Katsaggelos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21606](https://doi.org/10.1609/aaai.v36i11.21606)

**Abstract**:

Lack  of  explainability  in  artificial  intelligence,  specifically deep neural networks, remains a bottleneck for implementing models in practice. Popular techniques such as Gradient-weighted Class Activation Mapping (Grad-CAM) provide a coarse  map  of  salient  features  in  an  image,  which  rarely tells the whole story of what a convolutional neural network(CNN) learned. Using COVID-19 chest X-rays, we present a method for interpreting what a CNN has learned by utilizing Generative Adversarial Networks (GANs). Our GAN framework  disentangles  lung  structure  from  COVID-19  features. Using this GAN, we can visualize the transition of a pair of COVID negative lungs in a chest radiograph to a COVID positive  pair  by  interpolating  in  the  latent  space  of  the  GAN, which  provides  fine-grained  visualization  of  how  the  CNN responds to varying features within the lungs.

----

## [1491] Identifying ATT&CK Tactics in Android Malware Control Flow Graph through Graph Representation Learning and Interpretability (Student Abstract)

**Authors**: *Jeffrey Fairbanks, Andres Orbe, Christine Patterson, Edoardo Serra, Marion Scheepers*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21607](https://doi.org/10.1609/aaai.v36i11.21607)

**Abstract**:

To mitigate a malware threat it is important to understand the malware’s behavior. The MITRE ATT&ACK ontology specifies an enumeration of tactics, techniques, and procedures (TTP) that characterize malware. However, absent are automated procedures that would characterize, given the malware executable, which part of the execution flow is connected with a specific TTP. This paper provides an automation methodology to locate TTP in a sub-part of the control flow graph that describes the execution flow of a malware executable. This methodology merges graph representation learning and tools for machine learning explanation.

----

## [1492] Reinforcement Learning Explainability via Model Transforms (Student Abstract)

**Authors**: *Mira Finkelstein, Lucy Liu, Yoav Kolumbus, David C. Parkes, Jeffrey S. Rosenshein, Sarah Keren*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21608](https://doi.org/10.1609/aaai.v36i11.21608)

**Abstract**:

Understanding the emerging behaviors of reinforcement learning agents may be difficult because such agents are often trained using highly complex and expressive models. In recent years, most approaches developed for explaining agent behaviors rely on domain knowledge or on an analysis of the agent’s learned policy. For some domains, relevant knowledge may not be available or may be insufficient for producing meaningful explanations. We suggest using formal model abstractions and transforms, previously used mainly for expediting the search for optimal policies, to automatically explain discrepancies that may arise between the behavior of an agent and the behavior that is anticipated by an observer. We formally define this problem of Reinforcement Learning Policy Explanation(RLPE), suggest a class of transforms which can be used for explaining emergent behaviors, and suggest meth-ods for searching efficiently for an explanation. We demonstrate the approach on standard benchmarks.

----

## [1493] Wind Prediction under Random Data Corruption (Student Abstract)

**Authors**: *Conner Flansburg, Dimitrios I. Diochnos*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21609](https://doi.org/10.1609/aaai.v36i11.21609)

**Abstract**:

We study the robustness of ridge regression, lasso regression, and of a neural network, when the training set has been randomly corrupted and in response to this corruption the training-size is reduced in order to remove the corrupted data. While the neural network appears to be the most robust method among these three, nevertheless lasso regression appears to be the method of choice since it suffers less loss both when the full information is available to the learner, as well as when a significant amount of the original training set has been rendered useless because of random data corruption.

----

## [1494] Knowledge-Enhanced Scene Graph Generation with Multimodal Relation Alignment (Student Abstract)

**Authors**: *Ze Fu, Junhao Feng, Changmeng Zheng, Yi Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21610](https://doi.org/10.1609/aaai.v36i11.21610)

**Abstract**:

Existing scene graph generation methods suffer the limitations when the image lacks of sufficient visual contexts. To address this limitation, we propose a knowledge-enhanced scene graph generation model with multimodal relation alignment, which supplements the missing visual contexts by well-aligned textual knowledge. First, we represent the textual information into contextualized knowledge which is guided by the visual objects to enhance the contexts. Furthermore, we align the multimodal relation triplets by co-attention module for better semantics fusion. The experimental results show the effectiveness of our method.

----

## [1495] HuggingMolecules: An Open-Source Library for Transformer-Based Molecular Property Prediction (Student Abstract)

**Authors**: *Piotr Gainski, Lukasz Maziarka, Tomasz Danel, Stanislaw Jastrzebski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21611](https://doi.org/10.1609/aaai.v36i11.21611)

**Abstract**:

Large-scale transformer-based methods are gaining popularity as a tool for predicting the properties of chemical compounds, which is of central importance to the drug discovery process. To accelerate their development and dissemination among the community, we are releasing HuggingMolecules -- an open-source library, with a simple and unified API, that provides the implementation of several state-of-the-art transformers for molecular property prediction. In addition, we add a comparison of these methods on several regression and classification datasets. HuggingMolecules package is available at: github.com/gmum/huggingmolecules.

----

## [1496] From Video to Images: Contrastive Pretraining for Emotion Recognition from Single Image (Student Abstract)

**Authors**: *Bhanu Garg, Kijun Kim, Sudhanshu Ranjan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21612](https://doi.org/10.1609/aaai.v36i11.21612)

**Abstract**:

Emotion detection from face is an important problem and has received attention from industry and academia. Although emotion recognition from videos has a very high performance, emotion recognition from a single image stays a challenging task. In this paper, we try to use information from videos to do emotion recognition on a single image. More specifically, we leverage contrastive loss for pretraining the network on the videos and experiment with different sampling methods to select consistently hard triplets for continual learning of the network. Once the embeddings have been trained, we test them on a standard emotion classification task. Our method significantly improves the performance of the models and shows the efficacy of self-supervision in emotion recognition.

----

## [1497] An Ontological Approach towards Automatic Creation of Infographics from Formal Text (Student Abstract)

**Authors**: *Devin Garg, Tanuj Agarwal, Chiranjoy Chattopadhyay*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21613](https://doi.org/10.1609/aaai.v36i11.21613)

**Abstract**:

Infographics deal with representing data or information visually in a perceptually compelling manner. Recently, infographics have gained widespread popularity, giving rise to automated infographics synthesis from texts. Our research follows an ontological approach to automatically extract the necessary indicators from an input sentence and synthesize an infographic corresponding to it. This work includes (1) the creation of a dataset, (2) an end-to-end domain-agnostic framework, and (3) demonstrating the application of the proposed framework. The results demonstrate our framework's ability to extract the necessary textual cues from real-world textual descriptions (from various domains) and synthesize meaningful infographics.

----

## [1498] JoTA: Aligning Multilingual Job Taxonomies through Word Embeddings (Student Abstract)

**Authors**: *Anna Giabelli, Lorenzo Malandri, Fabio Mercorio, Mario Mezzanzanica*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21614](https://doi.org/10.1609/aaai.v36i11.21614)

**Abstract**:

We propose JoTA (Job Taxonomy Alignment), a domain-independent, knowledge-poor method for automatic taxonomy alignment of lexical taxonomies via word embeddings. JoTA associates all the leaf terms of the origin taxonomy to one or many concepts in the destination one, employing a scoring function, which merges the score of a hierarchical method and the score of a classification task.
JoTA is developed in the context of an EU Grant aiming at bridging the national taxonomies of EU countries towards the European Skills, Competences, Qualifications and Occupations taxonomy (ESCO) through AI.
The method reaches a 0.8 accuracy on recommending top-5 occupations and a wMRR of 0.72.

----

## [1499] MBGRLp: Multiscale Bootstrap Graph Representation Learning on Pointcloud (Student Abstract)

**Authors**: *Vandan Gorade, Azad Singh, Deepak Mishra*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21615](https://doi.org/10.1609/aaai.v36i11.21615)

**Abstract**:

Point cloud has gained a lot of attention with the availability of a large amount of point cloud data and increasing applications like city planning and self-driving cars. However, current methods, often rely on labeled information and costly processing, such as converting point cloud to voxel. We propose a self-supervised learning approach to tackle these problems, combating labelling and additional memory cost issues. Our proposed method achieves results comparable to supervised and unsupervised baselines on the widely used benchmark datasets for self-supervised point cloud classification
like ShapeNet, ModelNet10/40.

----

## [1500] Memotion Analysis through the Lens of Joint Embedding (Student Abstract)

**Authors**: *Nethra Gunti, Sathyanarayanan Ramamoorthy, Parth Patwa, Amitava Das*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21616](https://doi.org/10.1609/aaai.v36i11.21616)

**Abstract**:

Joint embedding (JE) is a way to encode multi-modal data into a vector space where text remains as the grounding key and other modalities like image are to be anchored with such keys. Meme is typically an image with embedded text onto it. Although, memes are commonly used for fun, they could also be used to spread hate and fake information. That along with its growing ubiquity over several social platforms has caused automatic analysis of memes to become a widespread topic of research. In this paper, we report our initial experiments on Memotion Analysis problem through joint embeddings. Results are marginally yielding SOTA.

----

## [1501] Contrastive Personalization Approach to Suspect Identification (Student Abstract)

**Authors**: *Devansh Gupta, Drishti Bhasin, Sarthak Bhagat, Shagun Uppal, Ponnurangam Kumaraguru, Rajiv Ratn Shah*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21617](https://doi.org/10.1609/aaai.v36i11.21617)

**Abstract**:

Targeted image retrieval has long been a challenging problem since each person has a different perception of different features leading to inconsistency among users in describing the details of a particular image. Due to this, each user needs a system personalized according to the way they have structured the image in their mind. One important application of this task is suspect identification in forensic investigations where a witness needs to identify the suspect from an existing criminal database. Existing methods require the attributes for each image or suffer from poor latency during training and inference. We propose a new approach to tackle this problem through explicit relevance feedback by introducing a novel loss function and a corresponding scoring function. For this, we leverage contrastive learning on the user feedback to generate the next set of suggested images while improving the level of personalization with each user feedback iteration.

----

## [1502] A Simplified Benchmark for Ambiguous Explanations of Knowledge Graph Link Prediction Using Relational Graph Convolutional Networks (Student Abstract)

**Authors**: *Nicholas Halliwell, Fabien Gandon, Freddy Lécué*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21618](https://doi.org/10.1609/aaai.v36i11.21618)

**Abstract**:

Relational Graph Convolutional Networks (RGCNs) are commonly used on Knowledge Graphs (KGs) to perform black box link prediction. Several algorithms have been proposed to explain their predictions. Evaluating performance of explanation methods for link prediction is difficult without ground truth explanations. Furthermore, there can be multiple explanations for a given prediction in a KG. No dataset exists where observations have multiple ground truth explanations to compare against. Additionally, no standard scoring metrics exist to compare predicted explanations against multiple ground truth explanations. We propose and evaluate a method, including a dataset, to benchmark explanation methods on the task of explainable link prediction using RGCNs.

----

## [1503] Deep Representation Debiasing via Mutual Information Minimization and Maximization (Student Abstract)

**Authors**: *Ruijiang Han, Wei Wang, Yuxi Long, Jiajie Peng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21619](https://doi.org/10.1609/aaai.v36i11.21619)

**Abstract**:

Deep representation learning has succeeded in several fields. However, pre-trained deep representations are usually biased and make downstream models sensitive to different attributes. In this work, we propose a post-processing unsupervised deep representation debiasing algorithm, DeepMinMax, which can obtain unbiased representations directly from pre-trained representations without re-training or fine-tuning the entire model. The experimental results on synthetic and real-world datasets indicate that DeepMinMax outperforms the existing state-of-the-art algorithms on downstream tasks.

----

## [1504] Class-Wise Adaptive Self Distillation for Federated Learning on Non-IID Data (Student Abstract)

**Authors**: *Yuting He, Yiqiang Chen, Xiaodong Yang, Yingwei Zhang, Bixiao Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21620](https://doi.org/10.1609/aaai.v36i11.21620)

**Abstract**:

Federated learning (FL) enables multiple clients to collaboratively train a globally generalized model while keeping local data decentralized. A key challenge in FL is to handle the heterogeneity of data distributions among clients. The local model will shift the global feature when fitting local data, which results in forgetting the global knowledge. Following the idea of knowledge distillation, the global model's prediction can be utilized to help local models preserve the global knowledge in FL. However, when the global model hasn't converged completely, its predictions tend to be less reliable on certain classes, which may results in distillation's misleading of local models. In this paper, we propose a class-wise adaptive self distillation (FedCAD) mechanism to ameliorate this problem. We design class-wise adaptive terms to soften the influence of distillation loss according to the global model's performance on each class and therefore avoid the misleading. Experiments show that our method outperforms other state-of-the-art FL algorithms on benchmark datasets.

----

## [1505] Detecting Neighborhood Gentrification at Scale via Street Views and POIs (Student Abstract)

**Authors**: *Tianyuan Huang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21621](https://doi.org/10.1609/aaai.v36i11.21621)

**Abstract**:

Neighborhood gentrification plays a significant role in shaping the social and economic status of both individuals and communities. While some efforts have been made to detect gentrification in cities, existing approaches mainly relies on estimated measures from survey data and requires substantial work of human labeling yet fails to characterize the physical appearance of neighborhoods. To this end, we introduce a novel approach to incorporate data like street view images and POI features to represent urban neighborhoods comprehensively at each timestamp. We show the effectiveness of the proposed methods with previous research on gentrification measures: each neighborhood representation we trained not only indicates its gentrification status, but also could become supplementary parts for the current measures and valid resource for researchers and policy makers.

----

## [1506] A Discriminative and Robust Feature Learning Approach for EEG-Based Motor Imagery Decoding (Student Abstract)

**Authors**: *Xiuyu Huang, Nan Zhou, Kup-Sze Choi*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21622](https://doi.org/10.1609/aaai.v36i11.21622)

**Abstract**:

Convolutional neural networks (CNNs) have been commonly applied in the area of the Electroencephalography (EEG)-based Motor Imagery (MI) classification, significantly pushing the boundary of the state-of-the-art. In order to simultaneously decode the discriminative features and eliminate the negative effects of non-Gaussian noise and outliers in the motor imagery data, in this abstract, we propose a novel robust supervision signal, called Correntropy based Center Loss (CCL), for CNN training, which utilizes the correntropy induced distance as the objective measure. It is encouraging to see that the CNN model trained by the combination of softmax loss and CCL loss outperforms the state-of-the-art models on two public datasets.

----

## [1507] A Stochastic Momentum Accelerated Quasi-Newton Method for Neural Networks (Student Abstract)

**Authors**: *S. Indrapriyadarsini, Shahrzad Mahboubi, Hiroshi Ninomiya, Takeshi Kamio, Hideki Asai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21623](https://doi.org/10.1609/aaai.v36i11.21623)

**Abstract**:

Incorporating curvature information in stochastic methods has been a challenging task. This paper proposes a momentum accelerated BFGS quasi-Newton method in both its full and limited memory forms, for solving stochastic large scale non-convex optimization problems in neural networks (NN).

----

## [1508] AsyncFL: Asynchronous Federated Learning Using Majority Voting with Quantized Model Updates (Student Abstract)

**Authors**: *Suji Jang, Hyuk Lim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21624](https://doi.org/10.1609/aaai.v36i11.21624)

**Abstract**:

Federated learning (FL) performs the global model updating in a synchronous manner in that the FL server waits for a specific number of local models from distributed devices before computing and sharing a new global model. We propose asynchronous federated learning (AsyncFL), which allows each client to continuously upload its model based on its capabilities and the FL server to determine when to asynchronously update and broadcast the global model. The asynchronous model aggregation at the FL server is performed by the Boyer–Moore majority voting algorithm for the k-bit quantized weight values. The proposed FL can speed up the convergence of the global model learning early in the FL process and reduce data exchange once the model is converged.

----

## [1509] Code Representation Learning Using Prüfer Sequences (Student Abstract)

**Authors**: *Tenzin Jinpa, Yong Gao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21625](https://doi.org/10.1609/aaai.v36i11.21625)

**Abstract**:

An effective and efficient encoding of the source code of a computer program is critical to the success of sequence-to-sequence deep neural network models for code representation learning. In this study, we propose to use the Prufer sequence of the Abstract Syntax Tree (AST) of a computer program to design a sequential representation scheme that preserves the structural information in an AST. Our representation makes it possible to develop deep-learning models in which signals carried by lexical tokens in the training examples can be exploited automatically and selectively based on their syntactic role and importance. Unlike other recently-proposed approaches, our representation is concise and lossless in terms of the structural information of the AST. Results from our experiment show that prufer-sequence-based representation is indeed highly effective and efficient.

----

## [1510] Gerrymandering under Uncertain Preferences (Student Abstract)

**Authors**: *Benjamin Kelly*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21626](https://doi.org/10.1609/aaai.v36i11.21626)

**Abstract**:

Gerrymandering is the manipulating of redistricting for political gain. While many attempts to formalize and model gerrymandering have been made, the assumption of known voter preference, or perfect information, limits the applicability of these works to model real world scenarios. To more accurately reason about gerrymandering we investigate how to adapt existing models of the problem to work with imperfect information. In our work, we formalize a definition of the gerrymandering problem under probabilistic voter preferences, reason about its complexity compared to the deterministic version, and propose a greedy algorithm to approximate the problem in polynomial time under certain conditions.

----

## [1511] FedCC: Federated Learning with Consensus Confirmation for Byzantine Attack Resistance (Student Abstract)

**Authors**: *Woocheol Kim, Hyuk Lim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21627](https://doi.org/10.1609/aaai.v36i11.21627)

**Abstract**:

In federated learning (FL), a server determines a global learning model by aggregating the local learning models of clients, and the determined global model is broadcast to all the clients. However, the global learning model can significantly deteriorate if a Byzantine attacker transmits malicious learning models trained with incorrectly labeled data. We propose a Byzantine-robust FL algorithm that, by employing a consensus confirmation method, can reduce the success probability of Byzantine attacks. After aggregating the local models from clients, the proposed FL server validates the global model candidate by sending the global model candidate to a set of randomly selected FL clients and asking them to perform local validation with their local data. If most of the validation is positive, the global model is confirmed and broadcast to all the clients. We compare the performance of the proposed FL against Byzantine attacks with that of existing FL algorithms analytically and empirically.

----

## [1512] Tracking Down Misguiding Terms for Locating Bugs in Deep Learning-Based Software (Student Abstract)

**Authors**: *Youngkyoung Kim, Misoo Kim, Eunseok Lee*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21628](https://doi.org/10.1609/aaai.v36i11.21628)

**Abstract**:

Bugs in source files (SFs) may cause software malfunction, inconveniencing users and even leading to catastrophic accidents. Therefore, the bugs in SFs should be found and fixed quickly. However, from hundreds of candidate SFs, finding buggy SFs is tedious and time consuming. To lessen the burden on developers, deep learning-based bug localization (DLBL) tools can be utilized. Text terms in bug reports and SFs play an important role. However, some terms provide incorrect information and degrade bug localization performance. Therefore, those terms are defined here as "misguiding terms," and an explainable-artificial-intelligence-based identification method is proposed. The effectiveness of the proposed method for DLBL was investigated. When misguiding terms were removed, the mean average precision of the bug localization model improved by 33% on average.

----

## [1513] Predicting RNA Mutation Effects through Machine Learning of High-Throughput Ribozyme Experiments (Student Abstract)

**Authors**: *Joseph Kitzhaber, Ashlyn Trapp, James Beck, Edoardo Serra, Francesca Spezzano, Eric Hayden, Jessica Roberts*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21629](https://doi.org/10.1609/aaai.v36i11.21629)

**Abstract**:

The ability to study "gain of function" mutations has important implications for identifying and mitigating risks to public health and national security associated with viral infections. Numerous respiratory viruses of concern have RNA genomes (e.g., SARS and flu). These RNA genomes fold into complex structures that perform several critical functions for viruses. However, our ability to predict the functional consequence of mutations in RNA structures continues to limit our ability to predict gain of function mutations caused by altered or novel RNA structures. Biological research in this area is also limited by the considerable risk of direct experimental work with viruses. Here we used small functional RNA molecules (ribozymes) as a model system of RNA structure and function. We used combinatorial DNA synthesis to generate all of the possible individual and pairs of mutations and used high-throughput sequencing to evaluate the functional consequence of each single- and double-mutant sequence. We used this data to train a machine learning model (Long Short-Term Memory). This model was also used to predict the function of sequences found in the genomes of mammals with three mutations, which were not in our training set. 
We found a strong prediction correlation in all of our experiments.

----

## [1514] Balancing the Spread of Two Opinions in Sparse Social Networks (Student Abstract)

**Authors**: *Dusan Knop, Simon Schierreich, Ondrej Suchý*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21630](https://doi.org/10.1609/aaai.v36i11.21630)

**Abstract**:

We perform an initial study of its computational complexity. It is not surprising that the problem is NP-hard even in quite restricted settings. Therefore, we investigate the complexity of the problem from the parameterized point-of-view with special focus on sparse networks, which appears often in practice. Among other things, we show that the proposed problem is in the FPT complexity class if we parameterize by the vertex cover number of the underlying graph.

----

## [1515] How to Reduce Action Space for Planning Domains? (Student Abstract)

**Authors**: *Harsha Kokel, Junkyu Lee, Michael Katz, Shirin Sohrabi, Kavitha Srinivas*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21631](https://doi.org/10.1609/aaai.v36i11.21631)

**Abstract**:

While AI planning and Reinforcement Learning (RL) solve sequential decision-making problems, they are based on different formalisms, which leads to a significant difference in their action spaces. When solving planning problems using RL algorithms, we have observed that a naive translation of the planning action space incurs severe degradation in sample complexity. In practice, those action spaces
are often engineered manually in a domain-specific manner. In this abstract, we present a method that reduces the parameters of operators in AI planning domains by introducing a parameter seed set problem and casting it as a classical planning task. Our experiment shows that our proposed method significantly reduces the number of actions in the RL environments originating from AI planning domains.

----

## [1516] A Scalable Parallel Algorithm for Balanced Sampling (Student Abstract)

**Authors**: *Alexander W. Lee, Stefan Walzer-Goldfeld, Shukry Zablah, Matteo Riondato*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21632](https://doi.org/10.1609/aaai.v36i11.21632)

**Abstract**:

We present a novel parallel algorithm for drawing balanced samples from large populations. When auxiliary variables about the population units are known, balanced sampling improves the quality of the estimations obtained from the sample. Available algorithms, e.g., the cube method, are inherently sequential, and do not scale to large populations. Our parallel algorithm is based on a variant of the cube method for stratified populations. It has the same sample quality as sequential algorithms, and almost ideal parallel speedup.

----

## [1517] TRACER: Extreme Attention Guided Salient Object Tracing Network (Student Abstract)

**Authors**: *Min Seok Lee, WooSeok Shin, Sung Won Han*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21633](https://doi.org/10.1609/aaai.v36i11.21633)

**Abstract**:

Existing studies on salient object detection (SOD) focus on extracting distinct objects with edge features and aggregating multi-level features to improve SOD performance. However, both performance gain and computational efficiency cannot be achieved, which has motivated us to study the inefficiencies in existing encoder-decoder structures to avoid this trade-off. We propose TRACER which excludes multi-decoder structures and minimizes the learning parameters usage by employing attention guided tracing modules (ATMs), as shown in Fig. 1.

----

## [1518] Social Aware Assignment of Passengers in Ridesharing (Student Abstract)

**Authors**: *Chaya Levinger, Noam Hazon, Amos Azaria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21634](https://doi.org/10.1609/aaai.v36i11.21634)

**Abstract**:

We analyze the assignment of passengers in a shared ride, which considers the social relationship among the passengers. Namely, there is a fixed number of passengers in each vehicle, and the goal is to recommend an assignment of the passengers such that the number of friendship relations is maximized. We show that the  problem is computationally hard, and we provide an approximation algorithm.

----

## [1519] SimCTC: A Simple Contrast Learning Method of Text Clustering (Student Abstract)

**Authors**: *Chen Li, Xiaoguang Yu, Shuangyong Song, Jia Wang, Bo Zou, Xiaodong He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21635](https://doi.org/10.1609/aaai.v36i11.21635)

**Abstract**:

This paper presents SimCTC, a simple contrastive learning (CL) framework that greatly advances the state-of-the-art text clustering models. In SimCTC, a pre-trained BERT model first maps the input sequence to the representation space, which is then followed by three different loss function heads: Clustering head, Instance-CL head and Cluster-CL head. Experimental results on multiple benchmark datasets demonstrate that SimCTC remarkably outperforms 6 competitive text clustering methods with 1%-6% improvement on Accuracy (ACC) and 1%-4% improvement on Normalized Mutual Information (NMI). Moreover, our results also show that the clustering performance can be further improved by setting an appropriate number of clusters in the cluster-level objective.

----

## [1520] Geotagging Social Media Posts to Landmarks Using Hierarchical BERT (Student Abstract)

**Authors**: *Menglin Li, Kwan Hui Lim*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21636](https://doi.org/10.1609/aaai.v36i11.21636)

**Abstract**:

Geographical information provided in social media data is useful for many valuable applications. However, only a small proportion of social media posts are explicitly geotagged with their posting locations, which makes the pursuit of these applications challenging. Motivated by this, we propose a 2-level hierarchical classification method that builds upon a BERT model, coupled with textual information and temporal context, which we denote HierBERT. As far as we are aware, this work is the first to utilize a 2-level hierarchical classification approach alongside BERT and temporal information for geolocation prediction. Experimental results based on two social media datasets show that HierBERT outperforms various state-of-art baselines in terms of accuracy and distance error metrics.

----

## [1521] A Probabilistic Framework for Land Deformation Prediction (Student Abstract)

**Authors**: *Rongfan Li, Fan Zhou, Goce Trajcevski, Kunpeng Zhang, Ting Zhong*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21637](https://doi.org/10.1609/aaai.v36i11.21637)

**Abstract**:

The development of InSAR (satellite Interferometric Synthetic Aperture Radar) enables accurate monitoring of land surface deformations, and has led to advances of deformation forecast for preventing landslide, which is one of the severe geological disasters. Despite the unparalleled success, existing spatio-temporal models typically make predictions on static adjacency relationships, simplifying the conditional dependencies and neglecting the distributions of variables. To overcome those limitations, we propose a Distribution Aware Probabilistic Framework (DAPF), which learns manifold embeddings while maintaining the distribution of deformations. We obtain a dynamic adjacency matrix upon which we approximate the true posterior while emphasizing the spatio-temporal characteristics. Experimental results on real-world dataset validate the superior performance of our method.

----

## [1522] Exploring Entity Interactions for Few-Shot Relation Learning (Student Abstract)

**Authors**: *Yi Liang, Shuai Zhao, Bo Cheng, Yuwei Yin, Hao Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21638](https://doi.org/10.1609/aaai.v36i11.21638)

**Abstract**:

Few-shot relation learning refers to infer facts for relations with a few observed triples. Existing metric-learning methods mostly neglect entity interactions within and between triples. In this paper, we explore this kind of fine-grained semantic meaning and propose our model TransAM. Specifically, we serialize reference entities and query entities into sequence and apply transformer structure with local-global attention to capture intra- and inter-triple entity interactions. Experiments on two public datasets with 1-shot setting prove the effectiveness of TransAM.

----

## [1523] MMAN: Metapath Based Multi-Level Graph Attention Networks for Heterogeneous Network Embedding (Student Abstract)

**Authors**: *Jie Liu, Lingyun Song, Li Gao, Xuequn Shang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21639](https://doi.org/10.1609/aaai.v36i11.21639)

**Abstract**:

Current Heterogeneous Network Embedding (HNE) models can be roughly divided into two types, i.e., relation-aware and metapath-aware models. However, they either fail to represent the non-pairwise relations in heterogeneous graph, or only capable of capturing local information around target node. In this paper, we propose a metapath based multilevel graph attention networks (MMAN) to jointly learn node embeddings on two substructures, i.e., metapath based graphs and hypergraphs extracted from original heterogeneous graph. Extensive experiments on three benchmark datasets for node classification and node clustering demonstrate the superiority of MMAN over the state-of-the-art works.

----

## [1524] The Psychology of Semantic Spaces: Experiments with Positive Emotion (Student Abstract)

**Authors**: *Xuan Liu, Kokil Jaidka, Niyati Chayya*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21640](https://doi.org/10.1609/aaai.v36i11.21640)

**Abstract**:

Psychological concepts can help computational linguists to better model the latent semantic spaces of emotions, and understand the underlying states motivating the sharing or suppressing of emotions. This abstract applies the understanding of agency and social interaction in the happiness semantic space to its role in positive emotion. First, BERT-based fine-tuning yields an expanded seed set to understand the vocabulary of the latent space. Next, results benchmarked against many emotion datasets suggest that the approach is valid, robust, offers an improvement over direct prediction, and is useful for downstream predictive tasks related to psychological states.

----

## [1525] Adaptive Safe Behavior Generation for Heterogeneous Autonomous Vehicles Using Parametric-Control Barrier Functions (Student Abstract)

**Authors**: *Yiwei Lyu, Wenhao Luo, John M. Dolan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21641](https://doi.org/10.1609/aaai.v36i11.21641)

**Abstract**:

Control Barrier Functions have been extensively studied to ensure guaranteed safety during inter-robot interactions. In this paper, we introduce the Parametric-Control Barrier Function (Parametric-CBF), a novel variant of the traditional Control Barrier Function to extend its expressivity in describing different safe behaviors among heterogeneous robots. A parametric-CBF based framework is presented to enable the ego robot to model the neighboring robots behavior and further improve the coordination efficiency during interaction while enjoying formally provable safety guarantees. We demonstrate the usage of Parametric-CBF in behavior prediction and adaptive safe control in the ramp merging scenario.

----

## [1526] Switch-GPT: An Effective Method for Constrained Text Generation under Few-Shot Settings (Student Abstract)

**Authors**: *Chang Ma, Song Zhang, Gehui Shen, Zhihong Deng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21642](https://doi.org/10.1609/aaai.v36i11.21642)

**Abstract**:

In real-world applications of natural language generation, target sentences are often required to satisfy some lexical constraints. However, the success of most neural-based models relies heavily on data, which is infeasible for data-scarce new domains. In this work, we present FewShotAmazon, the first benchmark for the task of Constrained Text Generation under few-shot settings on multiple domains. Further, we propose the Switch-GPT model, in which we utilize the strong language modeling capacity of GPT-2 to generate fluent and well-formulated sentences, while using a light attention module to decide which constraint to attend to at each step. Experiments show that the proposed Switch-GPT model is effective and remarkably outperforms the baselines. Codes will be available at https://github.com/chang-github-00/Switch-GPT.

----

## [1527] A Short-Term Tropical Cyclone Intensity Forecasting Method Based on High-Order Tensor (Student Abstract)

**Authors**: *Fan Meng, Handan Sun, Danya Xu, Pengfei Xie, Tao Song*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21643](https://doi.org/10.1609/aaai.v36i11.21643)

**Abstract**:

Tropical cyclones (TC) bring enormous harm to human beings, and it is crucial to accurately forecast the intensity of TCs, but the progress of intensity forecasting has been slow in recent years, and tropical cyclones are an extreme weather phenomenon with short duration, and the sample size of TC intensity series is small and short in length. In this paper, we devolop a tensor ARIMA model based on feature reconstruction to solve the problem, which represents multiple time series as low-rank Block Hankel Tensor(BHT), and combine the tensor decomposition technique with ARIMA for time series prediction. The method predicts the sustained maximum wind speed and central minimum pressure of TC 6-24 hours in advance, and the results show that the method exceeds the global numerical model GSM operated by the Japan Meteorological Agency (JMA) in the short term. We further checked the prediction results for a TC, and the results show the validity of the method.

----

## [1528] Early Forecast of Traffic Accident Impact Based on a Single-Snapshot Observation (Student Abstract)

**Authors**: *Guangyu Meng, Qisheng Jiang, Kaiqun Fu, Beiyu Lin, Chang-Tien Lu, Zhiqian Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21644](https://doi.org/10.1609/aaai.v36i11.21644)

**Abstract**:

Predicting and quantifying the impact of traffic accidents is necessary and critical to Intelligent Transport Systems (ITS). As a state-of-the-art technique in graph learning, current graph neural networks heavily rely on graph Fourier transform, assuming homophily among the neighborhood. However, the homophily assumption makes it challenging to characterize abrupt signals such as traffic accidents. Our paper proposes an abrupt graph wavelet network (AGWN) to model traffic accidents and predict their time durations using only one single snapshot.

----

## [1529] Enumerating Nontrivial Knot Mosaics with SAT (Student Abstract)

**Authors**: *Hannah Miller*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21645](https://doi.org/10.1609/aaai.v36i11.21645)

**Abstract**:

Mathematical knots are interesting topological objects.  Using simple arcs, lines, and crossings drawn on eleven possible tiles, knot mosaics are a representation of knots on a mosaic board.  Our contribution is using SAT solvers as a tool for enumerating nontrivial knot mosaics.  By encoding constraints for local knot mosaic properties, we computationally reduce the search space by factors of up to 6600.  Our future research directions include encoding constraints for global properties and using parallel SAT techniques to attack larger boards.

----

## [1530] Actionable Model-Centric Explanations (Student Abstract)

**Authors**: *Cecilia G. Morales, Nicholas Gisolfi, Robert Edman, James Kyle Miller, Artur Dubrawski*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21646](https://doi.org/10.1609/aaai.v36i11.21646)

**Abstract**:

We recommend using a model-centric, Boolean Satisfiability (SAT) formalism to obtain useful explanations of trained model behavior, different and complementary to what can be gleaned from LIME and SHAP, popular data-centric explanation tools in Artificial Intelligence (AI).We compare and contrast these methods, and show that data-centric methods may yield brittle explanations of limited practical utility.The model-centric framework, however, can offer actionable insights into risks of using AI models in practice. For critical applications of AI, split-second decision making is best informed by robust explanations that are invariant to properties of data, the capability offered by model-centric frameworks.

----

## [1531] A Model for the Prediction of Lifetime Profit Estimate of Dairy Cattle (Student Abstract)

**Authors**: *Vahid Naghashi, Abdoulaye Baniré Diallo*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21647](https://doi.org/10.1609/aaai.v36i11.21647)

**Abstract**:

In livestock management, the decision of animal replacement requires an estimation of the lifetime profit of the animal based on multiple factors and operational conditions. In Dairy farms, this can be associated with the profit corresponding to milk production, health condition and herd management costs, which in turn may be a function of other factors including genetics and weather conditions. Estimating the profit of a cow can be expressed as a spatio-temporal problem where knowing the first batch of production (early-profit) can allow to predict the future batch of productions (late-profit). 
This problem can be addressed either by a univariate or multivariate time series forecasting. Several approaches have been designed for time series forecasting including Auto-Regressive approaches, Recurrent Neural Network including Long Short Term Memory (LSTM) method and a very deep stack of fully-connected layers. In this paper, we proposed a LSTM based approach coupled with attention and linear layers to better capture the dairy features. We compare the model, with three other architectures including NBEATs, ARIMA, MUMU-RNN using dairy production of 292181 dairy cows. The results highlight the performence of the proposed model of the compared architectures. They also show that a univariate NBEATs could perform better than the multi-variate approach there are compared to.  We also highlight that such architecture could allow to predict late-profit with an error less than 3$ per month, opening the way of better resource management in the dairy industry.

----

## [1532] Explainable Shapley-Based Allocation (Student Abstract)

**Authors**: *Meir Nizri, Noam Hazon, Amos Azaria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21648](https://doi.org/10.1609/aaai.v36i11.21648)

**Abstract**:

The Shapley value is one of the most important normative division scheme in cooperative game theory, satisfying basic axioms. However, some allocation according to the Shapley value may seem unfair to humans.
In this paper, we develop an automatic method that generates intuitive explanations for a Shapley-based payoff allocation, which utilizes the basic axioms. Given a coalitional game, our method decomposes it to sub-games, for which it is easy to generate verbal explanations, and shows that the given game is composed of the sub-games. Since the payoff allocation for each sub-game is perceived as fair, the Shapley-based payoff allocation for the given game should seem fair as well.
We run an experiment with 210 human participants and show that when applying our method, humans perceive Shapley-based payoff allocation as significantly more fair than when using a general standard explanation.

----

## [1533] Using Reinforcement Learning for Operating Educational Campuses Safely during a Pandemic (Student Abstract)

**Authors**: *Elizabeth Akinyi Ondula, Bhaskar Krishnamachari*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21649](https://doi.org/10.1609/aaai.v36i11.21649)

**Abstract**:

The COVID-19 pandemic has brought a significant disruption not only on how schools operate but also affected student sentiments on learning and adoption to different learning strategies. We propose CampusPandemicPlanR, a reinforcement learning-based simulation tool that could be applied to suggest to campus operators how many students from each course to allow on a campus classroom each week. The tool aims to strike a balance between the conflicting goals of keeping students from getting infected, on one hand, and allowing more students to come into campus to allow them to benefit from in-person classes, on the other. Our preliminary results show that reinforcement learning is able to learn better policies over iterations, and that different Pareto-optimal tradeoffs between these conflicting goals could be obtained by varying the reward weight parameter.

----

## [1534] Grad-Align: Gradual Network Alignment via Graph Neural Networks (Student Abstract)

**Authors**: *Jin-Duk Park, Cong Tran, Won-Yong Shin, Xin Cao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21650](https://doi.org/10.1609/aaai.v36i11.21650)

**Abstract**:

Network alignment (NA) is the task of finding the correspondence of nodes between two networks. Since most existing NA methods have attempted to discover every node pair at once, they may fail to utilize node pairs that have strong consistency across different networks in the NA task. To tackle this challenge, we propose Grad-Align, a new NA method that gradually discovers node pairs by making full use of either node pairs exhibiting strong consistency or prior matching information. Specifically, the proposed method gradually aligns nodes based on both the similarity of embeddings generated using graph neural networks (GNNs) and the Tversky similarity, which is an asymmetric set similarity using the Tversky index applicable to networks with different scales. Experimental evaluation demonstrates that Grad-Align consistently outperforms state-of-the-art NA methods in terms of the alignment accuracy. Our source code is available at https://github.com/jindeok/Grad-Align.

----

## [1535] GRU4RecBE: A Hybrid Session-Based Movie Recommendation System (Student Abstract)

**Authors**: *Michael Potter, Hamlin Liu, Yash Lala, Christian Loanzon, Yizhou Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21651](https://doi.org/10.1609/aaai.v36i11.21651)

**Abstract**:

We present a novel movie recommendation system, GRU4RecBE, which extends the GRU4Rec architecture with rich item features extracted by the pre-trained BERT model. GRU4RecBE outperforms state-of-the-art session-based models over the benchmark MovieLens 1m and MovieLens 20m datasets.

----

## [1536] CL-NERIL: A Cross-Lingual Model for NER in Indian Languages (Student Abstract)

**Authors**: *Akshara Prabhakar, Gouri Sankar Majumder, Ashish Anand*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21652](https://doi.org/10.1609/aaai.v36i11.21652)

**Abstract**:

Developing Named Entity Recognition (NER) systems for Indian languages has been a long-standing challenge, mainly owing to the requirement of a large amount of annotated clean training instances. This paper proposes an end-to-end framework for NER for Indian languages in a low-resource setting by exploiting parallel corpora of English and Indian languages and an English NER dataset. The proposed framework includes an annotation projection method that combines word alignment score and NER tag prediction confidence score on source language (English) data to generate weakly labeled data in a target Indian language. We employ a variant of the Teacher-Student model and optimize it jointly on the pseudo labels of the Teacher model and predictions on the generated weakly labeled data. We also present manually annotated test sets for three Indian languages: Hindi, Bengali, and Gujarati. We evaluate the performance of the proposed framework on the test sets of the three Indian languages. Empirical results show a minimum 10% performance improvement compared to the zero-shot transfer learning model on all languages. This indicates that weakly labeled data generated using the proposed annotation projection method in target Indian languages can complement well-annotated source language data to enhance performance. Our code is publicly available at https://github.com/aksh555/CL-NERIL.

----

## [1537] Aspect-Opinion Sentiment Alignment for Cross-Domain Sentiment Analysis (Student Abstract)

**Authors**: *Haopeng Ren, Yi Cai, Yushi Zeng*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21653](https://doi.org/10.1609/aaai.v36i11.21653)

**Abstract**:

Cross-domain sentiment analysis (SA) has recently attracted significant attention, which can effectively alleviate the problem of lacking large-scale labeled data for deep neural network based methods. However, exiting unsupervised cross-domain SA models ignore the relation between the aspect and opinion, which suffer from the sentiment transfer error problem. To solve this problem, we propose an aspect-opinion sentiment alignment SA model and extensive experiments are conducted to evaluate the effectiveness of our model.

----

## [1538] XDC: Adversarial Adaptive Cross Domain Face Clustering (Student Abstract)

**Authors**: *Saed Rezayi, Handong Zhao, Sheng Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21654](https://doi.org/10.1609/aaai.v36i11.21654)

**Abstract**:

In this work we propose a scheme, called XDC, that uses adversarial learning to train an adaptive cross domain clustering model. XDC trains a classifier on a labeled dataset and assigns labels to an unlabeled dataset. We benefit from adversarial learning such that the target dataset takes part in the training. We also use an existing image classifiers in a plug-and-play fashion (i.e, it can be replaced with any other image classifier). Unlike existing works we update the parameters of the encoder and expose the target dataset to the model during training. We apply our model on two face dataset and one non-face dataset and obtain comparable results with state-of-the-art face clustering models.

----

## [1539] Integer and Constraint Programming Revisited for Mutually Orthogonal Latin Squares (Student Abstract)

**Authors**: *Noah Rubin, Curtis Bright, Brett Stevens, Kevin K. H. Cheung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21655](https://doi.org/10.1609/aaai.v36i11.21655)

**Abstract**:

We use integer programming (IP) and constraint programming (CP) to search for sets of mutually orthogonal latin squares (MOLS). We improve the performance of the solvers by formulating an extended symmetry breaking method and provide an alternative CP encoding which performs much better in practice. Using state-of-the-art solvers we are able to quickly find pairs of MOLS (or prove their nonexistence) in all orders up to and including eleven. We also analyze the effectiveness of using CP and IP solvers to search for triples of MOLS and estimate the running time of using this approach to resolve the longstanding open problem of determining the existence of a triple of MOLS of order ten.

----

## [1540] Do We Need a New Large-Scale Quality Assessment Database for Generative Inpainting Based 3D View Synthesis? (Student Abstract)

**Authors**: *Sadbhawna, Vinit Jakhetiya, Badri N. Subudhi, Harshit Shakya, Deebha Mumtaz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21656](https://doi.org/10.1609/aaai.v36i11.21656)

**Abstract**:

The advancement in Image-to-Image translation techniques using generative Deep Learning-based approaches has shown promising results for the challenging task of inpainting-based 3D view synthesis. At the same time, even the current 3D view synthesis methods often create distorted structures or blurry textures inconsistent with surrounding areas. We analyzed the recently proposed algorithms for inpainting-based 3D view synthesis and observed that these algorithms no longer produce stretching and black holes. However, the existing databases such as IETR, IRCCyN, and IVY have 3D-generated views with these artifacts. This observation suggests that the existing 3D view synthesis quality assessment algorithms can not judge the quality of most recent 3D synthesized views. With this view, through this abstract, we analyze the need for a new large-scale database and a new perceptual quality metric oriented for 3D views using a test dataset.

----

## [1541] NEUROCRYPT: Coercion-Resistant Implicit Memory Authentication (Student Abstract)

**Authors**: *Ritul Satish, Niranjan Rajesh, Argha Chakrabarty, Aditi Jain, Sristi Bafna, Arup Mondal, Debayan Gupta*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21657](https://doi.org/10.1609/aaai.v36i11.21657)

**Abstract**:

Overcoming the threat of coercion attacks in a cryptographic system has been a top priority for system designers since the birth of cyber-security. One way to overcome such a threat is to leverage implicit memory to construct a defense against rubber-hose attacks where the users themselves do not possess conscious knowledge of the trained password. We propose NeuroCrypt, a coercion-resistant authentication system that uses an improved version of the Serial Interception Sequence Learning task, employing additional auditory and haptic modalities backed by concepts borrowed from cognitive psychology. We carefully modify the visual stimuli as well as add auditory and haptic stimuli to improve the implicit learning process, resulting in faster training and longer retention. Moreover, our improvements guarantee that explicit recognition of the trained passwords remains suppressed.

----

## [1542] Towards One Shot Search Space Poisoning in Neural Architecture Search (Student Abstract)

**Authors**: *Nayan Saxena, Robert Wu, Rohan Jain*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21658](https://doi.org/10.1609/aaai.v36i11.21658)

**Abstract**:

We evaluate the robustness of a Neural Architecture Search (NAS) algorithm known as Efficient NAS (ENAS) against data agnostic poisoning attacks on the original search space with carefully designed ineffective operations. We empirically demonstrate how our one shot search space poisoning approach exploits design flaws in the ENAS controller to degrade predictive performance on classification tasks. With just two poisoning operations injected into the search space, we inflate prediction error rates for child networks upto 90% on the CIFAR-10 dataset.

----

## [1543] Optimizing Global Influenza Surveillance for Locations with Deficient Data (Student Abstract)

**Authors**: *Songwei Shan, Qi Tan, Yiu Chung Lau, Zhanwei Du, Eric H. Y. Lau, Peng Wu, Benjamin J. Cowling*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21659](https://doi.org/10.1609/aaai.v36i11.21659)

**Abstract**:

For better monitoring and controlling influenza, WHO has launched FluNet (recently integrated to FluMART) to provide a unified platform for participating countries to routinely collect influenza-related syndromic, epidemiological and virological data. However, the reported data were incomplete.We propose a novel surveillance system based on data from multiple sources to accurately assess the epidemic status of different countries, especially for those with missing surveillance data in some periods. The proposed method can automatically select a small set of reliable and informative indicators for assessing the underlying epidemic status and proper supporting data to train the predictive model. Our proactive selection method outperforms three other out-of-box methods (linear regression, multilayer perceptron, and long-short term memory) to make accurate predictions.

----

## [1544] Prototype-Based Explanations for Graph Neural Networks (Student Abstract)

**Authors**: *Yong-Min Shin, Sun-Woo Kim, Eun-Bi Yoon, Won-Yong Shin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21660](https://doi.org/10.1609/aaai.v36i11.21660)

**Abstract**:

Aside the high performance of graph neural networks (GNNs), considerable attention has recently been paid to explanations of black-box deep learning models. Unlike most studies focusing on model explanations based on a specific graph instance, we propose Prototype-bAsed GNN-Explainer (PAGE), a novel model-level explanation method for graph-level classification that explains what the underlying model has learned by providing human-interpretable prototypes. Specifically, our method performs clustering on the embedding space of the underlying GNN model; extracts embeddings in each cluster; and discovers prototypes, which serve as model explanations, by estimating the maximum common subgraph (MCS) from the extracted embeddings. Experimental evaluation demonstrates that PAGE not only provides high-quality explanations but also outperforms the state-of-the-art model-level method in terms of consistency and faithfulness that are performance metrics for quantitative evaluations.

----

## [1545] Modeling Constraints Can Identify Winning Arguments in Multi-Party Interactions (Student Abstract)

**Authors**: *Suzanna Sia, Kokil Jaidka, Niyati Chayya, Kevin Duh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21661](https://doi.org/10.1609/aaai.v36i11.21661)

**Abstract**:

In contexts where debate and
deliberation is the norm, participants are regularly presented with new information that conflicts with their original beliefs. When required to update their beliefs (belief alignment), they may choose arguments that align with their worldview (confirmation bias). We test this and competing hypotheses in a constraint-based modeling approach to predict the winning arguments in multi-party interactions in the Reddit ChangeMyView dataset. We impose structural constraints that reflect competing hypotheses on a hierarchical generative Variational Auto-encoder. Our findings suggest that when arguments are further from the initial belief state of the target, they are more likely to succeed.

----

## [1546] C3D and Localization Model for Locating and Recognizing the Actions from Untrimmed Videos (Student Abstract)

**Authors**: *Himanshu Singh, Tirupati Pallewad, Badri N. Subudhi, Vinit Jakhetiya*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21662](https://doi.org/10.1609/aaai.v36i11.21662)

**Abstract**:

In this article, we proposed a technique for action localization and recognition from long untrimmed videos. It consists of C3D CNN model followed by the action mining using the localization model, where the KNN classifier is used. We segment the video into expressible sub-action known as action-bytes. The pseudo labels have been used to train the localization model, which makes the trimmed videos untrimmed for action-bytes. We present experimental results on the recent benchmark trimmed video dataset “Thumos14”.

----

## [1547] On the Relation between Distributionally Robust Optimization and Data Curation (Student Abstract)

**Authors**: *Agnieszka Slowik, Léon Bottou, Sean B. Holden, Mateja Jamnik*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21663](https://doi.org/10.1609/aaai.v36i11.21663)

**Abstract**:

Machine learning systems based on minimizing average error have been shown to perform inconsistently across notable subsets of the data, which is not exposed by a low average error for the entire dataset. In consequential social and economic applications, where data represent people, this can lead to discrimination of underrepresented gender and ethnic groups. Distributionally Robust Optimization (DRO) seemingly addresses this problem by minimizing the worst expected risk across subpopulations. We establish theoretical results that clarify the relation between DRO and the optimization of the same loss averaged on an adequately weighted training dataset. A practical implication of our results is that neither DRO nor curating the training set should be construed as a complete solution for bias mitigation.

----

## [1548] Solving Visual Analogies Using Neural Algorithmic Reasoning (Student Abstract)

**Authors**: *Atharv Sonwane, Gautam Shroff, Lovekesh Vig, Ashwin Srinivasan, Tirtharaj Dash*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21664](https://doi.org/10.1609/aaai.v36i11.21664)

**Abstract**:

We consider a class of visual analogical reasoning problems that involve discovering the sequence of transformations by which pairs of input/output images are related, so as to analogously transform future inputs. This program synthesis task can be easily solved via symbolic search. Using a variation of the ‘neural analogical reasoning’ approach, we instead search for a sequence of elementary neural network transformations that manipulate distributed representations derived from a symbolic space, to which input images are directly encoded. We evaluate the extent to which our ‘neural reasoning’ approach generalises for images with unseen shapes and positions.

----

## [1549] Criticality-Based Advice in Reinforcement Learning (Student Abstract)

**Authors**: *Yitzhak Spielberg, Amos Azaria*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21665](https://doi.org/10.1609/aaai.v36i11.21665)

**Abstract**:

One of the ways to make reinforcement learning (RL) more efficient is by utilizing human advice. Because human advice is expensive, the central question in advice-based reinforcement learning is, how to decide in which states the agent should ask for advice.  To approach this challenge, various advice strategies have been proposed. Although all of these strategies distribute advice more efficiently than naive strategies, they rely solely on the agent's estimate of the action-value function, and therefore, are rather inefficient when this estimate is not accurate, in particular, in the early stages of the learning process.  To address this weakness, we present an approach to advice-based RL, in which the human’s role is not limited to giving advice in chosen states, but also includes hinting a-priori, before the learning procedure, in which sub-domains of the state space the agent might require more advice. For this purpose we use the concept of critical: states in which choosing the proper action is more important than in other states.

----

## [1550] Training Up to 50 Class ML Models on 3 $ IoT Hardware via Optimizing One-vs-One Algorithm (Student Abstract)

**Authors**: *Bharath Sudharsan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21666](https://doi.org/10.1609/aaai.v36i11.21666)

**Abstract**:

Multi-class classifier training using traditional meta-algorithms such as the popular One-vs-One (OvO) method may not always work well under cost-sensitive setups. Also, during inference, OvO becomes computationally challenging for higher class counts K as O(K^2) is its time complexity. In this paper, we present Opt-OvO, an optimized (resource-friendly) version of the One-vs-One algorithm to enable high-performance multi-class ML classifier training and inference directly on microcontroller units (MCUs). Opt-OvO enables billions of tiny IoT devices to self learn/train (offline) after their deployment, using live data from a wide range of IoT use-cases. We demonstrate Opt-OvO by performing live ML model training on 4 popular MCU boards using datasets of varying class counts, sizes, and feature dimensions. The most exciting finding was, on the  3 $ ESP32 chip, Opt-OvO trained a multi-class ML classifier using a dataset of class count 50 and performed unit inference in super real-time of 6.2 ms.

----

## [1551] Sampling and Counting Acyclic Orientations in Chordal Graphs (Student Abstract)

**Authors**: *Wenbo Sun*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21667](https://doi.org/10.1609/aaai.v36i11.21667)

**Abstract**:

Sampling of chordal graphs and various types of acyclic orientations over chordal graphs plays a central role in several AI applications such as causal structure learning. For a given undirected graph, an acyclic orientation is an assignment of directions to all of its edges which makes the resulting directed graph cycle-free. Sampling is often closely related to the corresponding counting problem. Counting of acyclic orientations of a given chordal graph can be done in polynomial time, but the previously known techniques do not seem to lead to a corresponding (efficient) sampler. In this work, we propose a dynamic programming framework which yields a counter and a uniform sampler, both of which run in (essentially) linear time. An interesting feature of our sampler is that it is a stand-alone algorithm that, unlike other DP-based samplers, does not need any preprocessing which determines the corresponding counts.

----

## [1552] A Repetitive Spectrum Learning Framework for Monaural Speech Enhancement in Extremely Low SNR Environments (Student Abstract)

**Authors**: *Wenxin Tai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21668](https://doi.org/10.1609/aaai.v36i11.21668)

**Abstract**:

Monaural speech enhancement (SE) at an extremely low signal-to-noise ratio (SNR) condition is a challenging problem and rarely investigated in previous studies. Most SE methods experience failures in this situation due to three major factors: overwhelmed vocals, expanded SNR range, and short-sighted feature processing modules. In this paper, we present a novel and general training paradigm dubbed repetitive learning (RL). Unlike curriculum learning that focuses on learning multiple different tasks sequentially, RL is more inclined to learn the same content repeatedly where the knowledge acquired in previous stages can be used to facilitate calibrating feature representations. We further propose an RL-based end-to-end SE method named SERL. Experimental results on TIMIT dataset validate the superior performance of our method.

----

## [1553] VeNAS: Versatile Negotiating Agent Strategy via Deep Reinforcement Learning (Student Abstract)

**Authors**: *Toki Takahashi, Ryota Higa, Katsuhide Fujita, Shinji Nakadai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21669](https://doi.org/10.1609/aaai.v36i11.21669)

**Abstract**:

Existing research in the field of automated negotiation considers a negotiation architecture in which some of the negotiation components are designed separately by reinforcement learning (RL), but comprehensive negotiation strategy design has not been achieved.
In this study, we formulated an RL model based on a Markov decision process (MDP) for bilateral multi-issue negotiations. We propose a versatile negotiating agent that can effectively learn various negotiation strategies and domains through comprehensive strategies using deep RL. We show that the proposed method can achieve the same or better utility than existing negotiation agents.

----

## [1554] Hybrid Deep Learning Model for Fake News Detection in Social Networks (Student Abstract)

**Authors**: *Bibek Upadhayay, Vahid Behzadan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21670](https://doi.org/10.1609/aaai.v36i11.21670)

**Abstract**:

The proliferation of fake news has grown into a global concern with adverse socio-political and economical impact. In recent years, machine learning has emerged as a promising approach to the automation of detecting and tracking fake news at scale. Current state of the art in the identification of fake news is generally focused on semantic analysis of the text, resulting in promising performance in automated detection of fake news. However, fake news campaigns are also evolving in response to such new technologies by mimicking semantic features of genuine news, which can significantly affect the performance of fake news classifiers trained on contextually limited features. In this work, we propose a novel hybrid deep learning model for fake news detection that augments the semantic characteristics of the news with features extracted from the structure of the dissemination network. To this end, we first extend the LIAR dataset by integrating sentiment and affective features to the data, and then use a BERT-based model to obtain a representation of the text. Moreover, we propose a novel approach for fake news detection based on Graph Attention Networks to leverage the user-centric features and graph features of news residing social network in addition to the features extracted in the previous steps. Experimental evaluation of our approach shows classification accuracy of 97% on the Politifact dataset. We also examined the generalizability of our proposed model on the BuzzFeed dataset, resulting in an accuracy 89.50%.

----

## [1555] Reducing Catastrophic Forgetting in Self Organizing Maps with Internally-Induced Generative Replay (Student Abstract)

**Authors**: *Hitesh Vaidya, Travis Desell, Alexander G. Ororbia II*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21671](https://doi.org/10.1609/aaai.v36i11.21671)

**Abstract**:

A lifelong learning agent is able to continually learn from potentially infinite streams of pattern sensory data. One major historic difficulty in building agents that adapt in this way is that neural systems struggle to retain previously-acquired knowledge when learning from new samples. This problem is known as catastrophic forgetting (interference) and remains an unsolved problem in the domain of machine learning to this day. While forgetting in the context of feedforward networks has been examined extensively over the decades, far less has been done in the context of alternative architectures such as the venerable self-organizing map (SOM), an unsupervised neural model that is often used in tasks such as clustering and dimensionality reduction. Although the competition among its internal neurons might carry the potential to improve memory retention, we observe that a fixed-sized SOM trained on task incremental data, i.e., it receives data points related to specific classes at certain temporal increments, it experiences severe interference. In this study, we propose the c-SOM, a model that is capable of reducing its own forgetting when processing information.

----

## [1556] Modeling Abstract Algebra as an OWL Ontology (Student Abstract)

**Authors**: *Michael Vance*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21672](https://doi.org/10.1609/aaai.v36i11.21672)

**Abstract**:

Description logic ontologies serve to model classifications and structural relationships, and to represent and reason about domain knowledge. Modeling the basic classification of abstract algebraic structures as an ontology demonstrates the difficulties presented by their logical semantics, and shed light on the limitations to accurately model further topics in algebra and related mathematical domains.

----

## [1557] Conditional Collaborative Filtering Process for Top-K Recommender System (Student Abstract)

**Authors**: *Guanyu Wang, Xovee Xu, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21673](https://doi.org/10.1609/aaai.v36i11.21673)

**Abstract**:

Conditional neural process (CNP) has been extensively applied into data analyzing tasks due to its excellent ability to make accurate predictions for incomplete data points. However, in literature there are only few works that studied the CNPin recommendation systems. In this work, we propose CCFP, which is a collaborative filtering method that differs from other CF models by incorporating CNP into encoder-decoder architecture. By analyzing the complete user-item interaction data, our model fits a global representation that can better rep-resenting the features of users and items. CCFP can significantly improve the recommendation performance compared to baselines by predicting items for the target users with their incomplete observation data.

----

## [1558] Augmentation of Chinese Character Representations with Compositional Graph Learning (Student Abstract)

**Authors**: *Jason Wang, Kaiqun Fu, Zhiqian Chen, Chang-Tien Lu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21674](https://doi.org/10.1609/aaai.v36i11.21674)

**Abstract**:

Chinese characters have semantic-rich compositional information in radical form. While almost all previous research has applied CNNs to extract this compositional information, our work utilizes deep graph learning on a compact, graph-based representation of Chinese characters. This allows us to exploit temporal information within the strict stroke order used in writing characters. Our results show that our stroke-based model has potential for helping large-scale language models on some Chinese natural language understanding tasks. In particular, we demonstrate that our graph model produces more interpretable embeddings shown through word subtraction analogies and character embedding visualizations.

----

## [1559] Large-Scale IP Usage Identification via Deep Ensemble Learning (Student Abstract)

**Authors**: *Zhiyuan Wang, Fan Zhou, Kunpeng Zhang, Yong Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21675](https://doi.org/10.1609/aaai.v36i11.21675)

**Abstract**:

Understanding users' behavior via IP addresses is essential towards numerous practical IP-based applications such as online content delivery, fraud prevention, and many others. Among which profiling IP address has been extensively studied, such as IP geolocation and anomaly detection. However, less is known about the scenario of an IP address, e.g., dedicated enterprise network or home broadband. In this work, we initiate the first attempt to address a large-scale IP scenario prediction problem. Specifically, we collect IP scenario data from four regions and propose a novel deep ensemble learning-based model to learn IP assignment rules and complex feature interactions. Extensive experiments support that our method can make accurate IP scenario identification and generalize from data in one region to another.

----

## [1560] BigCQ: Generating a Synthetic Set of Competency Questions Formalized into SPARQL-OWL (Student Abstract)

**Authors**: *Dawid Wisniewski, Jedrzej Potoniec, Agnieszka Lawrynowicz*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21676](https://doi.org/10.1609/aaai.v36i11.21676)

**Abstract**:

We present a method for constructing synthetic datasets of Competency Questions translated into SPARQL-OWL queries. This method is used to generate BigCQ, the largest set of CQ patterns and SPARQL-OWL templates that can provide translation examples to automate assessing the completeness and correctness of ontologies.

----

## [1561] Mixed Embedding of XLM for Unsupervised Cantonese-Chinese Neural Machine Translation (Student Abstract)

**Authors**: *Ka Ming Wong, Richard Tzong-Han Tsai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21677](https://doi.org/10.1609/aaai.v36i11.21677)

**Abstract**:

Unsupervised Neural Machines Translation is the most ideal method to apply to Cantonese and Chinese translation because parallel data is scarce in this language pair. In this paper, we proposed a method that combined a modified cross-lingual language model and performed layer to layer attention on unsupervised neural machine translation. In our experiments, we observed that our proposed method does improve the Cantonese to Chinese and Chinese to Cantonese translation by 1.088 and 0.394 BLEU scores. We finally developed a web service based on our ideal approach to provide Cantonese to Chinese Translation and vice versa.

----

## [1562] A Hybrid Evolutionary Algorithm for the Diversified Top-k Weight Clique Search Problem (Student Abstract)

**Authors**: *Jun Wu, Minghao Yin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21678](https://doi.org/10.1609/aaai.v36i11.21678)

**Abstract**:

The diversified top-k weight clique (DTKWC) search problem is an important generalization of the diversified top-k clique search problem, which extends the DTKC search problem by taking into account the weight of vertices. This problem involves finding at most k maximal weighted cliques that cover maximum weight of vertices with low overlapping in a given graph. In this study, a mixed integer linear program constraint formulation is proposed to model DTKWC search problem and an efficient hybrid evolutionary algorithm (HEA-D) based on some heuristic strategies is proposed to tackle it. Experiments on two sets of 110 graphs show that HEA-D outperforms the state-of-art methods.

----

## [1563] NeuralArTS: Structuring Neural Architecture Search with Type Theory (Student Abstract)

**Authors**: *Robert Wu, Nayan Saxena, Rohan Jain*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21679](https://doi.org/10.1609/aaai.v36i11.21679)

**Abstract**:

Neural Architecture Search (NAS) algorithms automate the task of finding optimal deep learning architectures given an initial search space of possible operations. Developing these search spaces is usually a manual affair with pre-optimized search spaces being more efficient, rather than searching from scratch. In this paper we present a new framework called Neural Architecture Type System (NeuralArTS) that categorizes the infinite set of network operations in a structured type system. We further demonstrate how NeuralArTS can be applied to convolutional layers and propose several future directions.

----

## [1564] Efficient Attribute (α, β)-Core Detection in Large Bipartite Graphs (Student Abstract)

**Authors**: *Yanping Wu, Renjie Sun, Chen Chen, Xiaoyang Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21680](https://doi.org/10.1609/aaai.v36i11.21680)

**Abstract**:

In this paper, we propose a novel problem, named rational (alpha, beta)-core detection in attribute bipartite graphs (RCD-ABG), which retrieves the connected (alpha, beta)-core with the largest rational score. A basic greedy framework with an optimized strategy is developed and extensive experiments are conducted to evaluate the performance of the techniques.

----

## [1565] Proof of Learning: Towards a Practical Blockchain Consensus Mechanism Using Directed Guiding Gradients (Student Abstract)

**Authors**: *Yongqi Wu, Xingjun Wang, Chen Chen, Guining Liu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21681](https://doi.org/10.1609/aaai.v36i11.21681)

**Abstract**:

Since Bitcoin, blockchain has attracted the attention of researchers. The consensus mechanism at the center of blockchain is often criticized for wasting a large amount of computing power for meaningless hashing. At the same time, state-of-the-art models in deep learning require increasing computing power to be trained. Proof of Learning (PoL) is dedicated to using the originally wasted computing power to train neural networks. Most of the previous PoL consensus mechanisms are based on two methods, recomputation or performance metrics. However, in practical scenarios, these methods both do not satisfy all properties necessary to build a large-scale blockchain, such as certainty, constant verification, therefore are still far away from being practical. In this paper, we observe that the opacity of deep learning models is similar to the pre-image resistance of hash functions and can naturally be used to build PoL. Based on our observation, we propose a method called Directed Guiding Gradient. Using this method, our proposed PoL consensus mechanism has a similar structure to the widely used Proof of Work (PoW), allowing us to build practical blockchain on it and train neutral networks simultaneously. In experiments, we build a blockchain on top of our proposed PoL consensus mechanism and results show that our PoL works well.

----

## [1566] Learning to Evolve on Dynamic Graphs (Student Abstract)

**Authors**: *Xintao Xiang, Tiancheng Huang, Donglin Wang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21682](https://doi.org/10.1609/aaai.v36i11.21682)

**Abstract**:

Representation learning in dynamic graphs is a challenging problem because the topology of graph and node features vary at different time. This requires the model to be able to effectively capture both graph topology information and temporal information. Most existing works are built on recurrent neural networks (RNNs), which are used to exact temporal information of dynamic graphs, and thus they inherit the same drawbacks of RNNs. In this paper, we propose Learning to Evolve on Dynamic Graphs (LEDG) - a novel algorithm that jointly learns graph information and time information. Specifically, our approach utilizes gradient-based meta-learning to learn updating strategies that have better generalization ability than RNN on snapshots. It is model-agnostic and thus can train any message passing based graph neural network (GNN) on dynamic graphs. To enhance the representation power, we disentangle the embeddings into time embeddings and graph intrinsic embeddings. We conduct experiments on various datasets and down-stream tasks, and the experimental results validate the effectiveness of our method.

----

## [1567] Automatic Slides Generation for Scholarly Papers: A Fine-Grained Dataset and Baselines (Student Abstract)

**Authors**: *Sheng Xu, Xiaojun Wan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21683](https://doi.org/10.1609/aaai.v36i11.21683)

**Abstract**:

Slides are broadly used to present the research works and there are several studies on the problem of automatic slides generation. However, the lack of dataset hinders further research. In this paper, we construct a benchmark dataset for the problem of slides generation from scholarly papers. The dataset is fine-grained and consists of aligned pairs of single slide and specific region of a paper. Then we deploy several baseline models and conduct preliminary experiments. The results show that this task is challenging and awaits more exploration. The dataset and code will be released.

----

## [1568] Crowdsourcing with Meta-Knowledge Transfer (Student Abstract)

**Authors**: *Sunyue Xu, Jing Zhang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21684](https://doi.org/10.1609/aaai.v36i11.21684)

**Abstract**:

When crowdsourced workers perform annotation tasks in an unfamiliar domain, their accuracy will dramatically decline due to the lack of expertise. Transferring knowledge from relevant domains can form a better representation for instances, which benefits the estimation of workers' expertise in truth inference models. However, existing knowledge transfer processes for crowdsourcing require a considerable number of well-collected instances in source domains. This paper proposes a novel truth inference model for crowdsourcing, where (meta-)knowledge is transferred by meta-learning and used in the estimation of workers' expertise. Our preliminary experiments demonstrate that the meta-knowledge transfer significantly reduces instances in source domains and increases the accuracy of truth inference.

----

## [1569] Multi-View Adjacency-Constrained Nearest Neighbor Clustering (Student Abstract)

**Authors**: *Jie Yang, Chin-Teng Lin*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21685](https://doi.org/10.1609/aaai.v36i11.21685)

**Abstract**:

Most existing multi-view clustering methods have problems with parameter selection and high computational complexity, and there have been very few works based on hierarchical clustering to learn the complementary information of multiple views. In this paper, we propose a Multi-view Adjacency-constrained Nearest Neighbor Clustering (MANNC) and its parameter-free version (MANNC-PF) to overcome these limitations. Experiments tested on eight real-world datasets validate the superiority of the proposed methods compared with the 13 current state-of-the-art methods.

----

## [1570] Learning to Ask for Data-Efficient Event Argument Extraction (Student Abstract)

**Authors**: *Hongbin Ye, Ningyu Zhang, Zhen Bi, Shumin Deng, Chuanqi Tan, Hui Chen, Fei Huang, Huajun Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21686](https://doi.org/10.1609/aaai.v36i11.21686)

**Abstract**:

Event argument extraction (EAE) is an important task for information extraction to discover specific argument roles. In this study, we cast EAE as a question-based cloze task and empirically analyze fixed discrete token template performance. As generating human-annotated question templates is often time-consuming and labor-intensive, we further propose a novel approach called “Learning to Ask,” which can learn optimized question templates for EAE without human annotations. Experiments using the ACE-2005 dataset demonstrate that our method based on optimized questions achieves state-of-the-art performance in both the few-shot and supervised settings.

----

## [1571] Fine-Grained Urban Flow Inference via Normalizing Flow (Student Abstract)

**Authors**: *Haoyang Yu, Xovee Xu, Ting Zhong, Fan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21687](https://doi.org/10.1609/aaai.v36i11.21687)

**Abstract**:

Fine-grained urban flow inference (FUFI) aims to infer the coarse-grained (CG) urban flow map to the corresponding fine-grained (FG) one, which plays an important role in efficient traffic monitoring and management in smart cities. In FUFI, the CG map can be obtained with only a small number of monitoring devices, greatly reducing the overhead of deploying devices and the costs of maintenance, labor, and electricity. Existing FUFI methods are mainly based on techniques from image super-resolution (SR) models, which cannot fully consider the influence of external factors and face the ill-posed problem in SR tasks. In this paper, we propose UFI-Flow, a novel approach for addressing the FUFI problem by learning the conditional distributions of CG and FG map pairs. Given the CG map and the latent variables, the corresponding FG map is inferred by invertible transformations. In addition, an augmented distribution fusion mechanism is further proposed to constrain the urban flow distribution within the influence of external factors. We provide a new large-scale real-world FUFI dataset and show that UFI-Flow significantly outperforms the strong baselines.

----

## [1572] Linking Transformer to Hawkes Process for Information Cascade Prediction (Student Abstract)

**Authors**: *Liu Yu, Xovee Xu, Ting Zhong, Goce Trajcevski, Fan Zhou*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21688](https://doi.org/10.1609/aaai.v36i11.21688)

**Abstract**:

Information cascade is typically formalized as a process of (simplified) discrete sequence of events, and recent approaches have tackled its prediction via variants of recurrent neural networks. However, the information diffusion process is essentially an evolving directed acyclic graph (DAG) in the continuous-time domain. In this paper, we propose a transformer enhanced Hawkes process (Hawkesformer), which links the hierarchical attention mechanism with Hawkes process to model the arrival stream of discrete events continuously. A two-level attention architecture is used to parameterize the intensity function of Hawkesformer, which captures the long-term dependencies between nodes in graph and better embeds the cascade evolution rate for modeling short-term outbreaks. Experimental results demonstrate the significant improvements of Hawkesformer over the state-of-the-art.

----

## [1573] Enhance Cross-Domain Aspect-Based Sentiment Analysis by Incorporating Commonsense Relational Structure (Student Abstract)

**Authors**: *Yushi Zeng, Guohua Wang, Haopeng Ren, Yi Cai*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21689](https://doi.org/10.1609/aaai.v36i11.21689)

**Abstract**:

Aspect Based Sentiment Analysis (ABSA) aims to extract aspect terms and identify the sentiment polarities towards each extracted aspect term. Currently, syntactic information is seen as the bridge for the domain adaptation and achieves remarkable performance. However, the transferable syntactic knowledge is complex and diverse, which causes the transfer error problem in domain adaptation. In our paper, we propose a domain-shared relational structure incorporated cross-domain ABSA model. The experimental results show the effectiveness of our model.

----

## [1574] Predicting the Influence of Fake and Real News Spreaders (Student Abstract)

**Authors**: *Amy Zhang, Aaron Brookhouse, Daniel Hammer, Francesca Spezzano, Liljana Babinkostova*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21690](https://doi.org/10.1609/aaai.v36i11.21690)

**Abstract**:

We study the problem of predicting the influence of a user in spreading fake (or real) news on social media. We propose a new model to address this problem which takes into account both user and tweet characteristics. We show that our model achieves an F1 score of 0.853, resp. 0.931, at predicting the influence of fake, resp. real, news spreaders, and outperforms existing baselines. We also investigate important features at predicting the influence of real vs. fake news spreaders.

----

## [1575] Understanding Stochastic Optimization Behavior at the Layer Update Level (Student Abstract)

**Authors**: *Jack Zhang, Guan Xiong Qiao, Alexandru Lopotenco, Ian Tong Pan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21691](https://doi.org/10.1609/aaai.v36i11.21691)

**Abstract**:

Popular first-order stochastic optimization methods for deep neural networks (DNNs) are usually either accelerated schemes (e.g. stochastic gradient descent (SGD) with momentum) or adaptive step-size methods (e.g. Adam/AdaMax, AdaBelief). In many contexts, including image classification with DNNs, adaptive methods tend to generalize poorly compared to SGD, i.e. get stuck in non-robust local minima; however, SGD typically converges slower. We analyze possible reasons for this behavior by modeling gradient updates as vectors of random variables and comparing them to probabilistic bounds to identify "meaningful" updates. Through experiments, we observe that only layers close to the output have "definitely non-random" update behavior. In the future, the tools developed here may be useful in rigorously quantifying and analyzing intuitions about why some optimizers and particular DNN architectures perform better than others.

----

## [1576] From "Dynamics on Graphs" to "Dynamics of Graphs": An Adaptive Echo-State Network Solution (Student Abstract)

**Authors**: *Lei Zhang, Zhiqian Chen, Chang-Tien Lu, Liang Zhao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21692](https://doi.org/10.1609/aaai.v36i11.21692)

**Abstract**:

Many real-world networks evolve over time, which results in dynamic graphs such as human mobility networks and brain networks. Usually, the “dynamics on graphs” (e.g., node attribute values evolving) are observable, and may be related to and indicative of the underlying “dynamics of graphs” (e.g., evolving of the graph topology). Traditional RNN-based methods are not adaptive or scalable for learn- ing the unknown mappings between two types of dynamic graph data. This study presents a AD-ESN, and adaptive echo state network that can automatically learn the best neural net- work architecture for certain data while keeping the efficiency advantage of echo state networks. We show that AD-ESN can successfully discover the underlying pre-defined map- ping function and unknown nonlinear map-ping between time series and graphs.

----

## [1577] A Multi-Factor Classification Framework for Completing Users' Fuzzy Queries (Student Abstract)

**Authors**: *Yaning Zhang, Liangqing Wu, Yangyang Wang, Jia Wang, Xiaoguang Yu, Shuangyong Song, Youzheng Wu, Xiaodong He*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21693](https://doi.org/10.1609/aaai.v36i11.21693)

**Abstract**:

Intent identification is the key technology in dialogue system. However, not all online queries are clear or complete. To identify users' intents from those fuzzy queries accurately, this paper proposes a multi-factor classification framework on the query level. Experimental results on our online serving system JIMI demonstrate the effectiveness of our proposed framework.

----

## [1578] Blocking Influence at Collective Level with Hard Constraints (Student Abstract)

**Authors**: *Zonghan Zhang, Subhodip Biswas, Fanglan Chen, Kaiqun Fu, Taoran Ji, Chang-Tien Lu, Naren Ramakrishnan, Zhiqian Chen*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21694](https://doi.org/10.1609/aaai.v36i11.21694)

**Abstract**:

Influence blocking maximization (IBM) is crucial in many critical real-world problems such as rumors prevention and epidemic containment. The existing work suffers from: (1) concentrating on uniform costs at the individual level, (2) mostly utilizing greedy approaches to approximate optimization, (3) lacking a proper graph representation for influence estimates. To address these issues, this research introduces a neural network model dubbed Neural Influence Blocking (\algo) for improved approximation and enhanced influence blocking effectiveness. The code is available at https://github.com/oates9895/NIB.

----

## [1579] Capsule Graph Neural Network for Multi-Label Image Recognition (Student Abstract)

**Authors**: *Xiangping Zheng, Xun Liang, Bo Wu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21695](https://doi.org/10.1609/aaai.v36i11.21695)

**Abstract**:

This paper studies the problem of learning complex relationships between multi-labels for image recognition. Its challenges come from the rich and diverse semantic information in images. However, current methods cannot fully explore the mutual interactions among labels and do not explicitly model the label co-occurrence. To overcome these shortcomings, we innovatively propose CGML that consists of two crucial modules: 1) an image representation learning module that aims to complete the feature extraction of an image whose features are expressed in the form of primary capsules; 2) a label adaptive graph convolutional network module that leverages the popular graph convolutional networks with an adaptive label correlation graph to model label dependencies. Experiments show that our approach obviously outperforms the existing state-of-the-art methods.

----

## [1580] Enhance Weakly-Supervised Aspect Detection with External Knowledge (Student Abstract)

**Authors**: *Zhuoming Zheng, Yi Cai, Liuwu Li*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21696](https://doi.org/10.1609/aaai.v36i11.21696)

**Abstract**:

Aspect detection aims to identify aspects of reviews and is an essential up-stream task of opinion mining and so on. However, existing weakly-supervised methods suffer from lacking the ability of identifying implicit aspects with infrequent aspect terms and "Misc" aspects. To tackle these problems, we propose to enhance the representation of segment with external knowledge by a weakly-supervised method. Experiments demonstrate the effectiveness of our model and the improvement by incorporating external knowledge.

----

## [1581] Efficient Deep Learning for Multi Agent Pathfinding

**Authors**: *Natalie Abreu*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21697](https://doi.org/10.1609/aaai.v36i11.21697)

**Abstract**:

Multi Agent Path Finding (MAPF) is widely needed to coordinate real-world robotic systems. New approaches turn to deep learning to solve MAPF instances, primarily using reinforcement learning, which has high computational costs. We propose a supervised learning approach to solve MAPF instances using a smaller, less costly model.

----

## [1582] Demystifying the Chinese Social Credit System: A Case Study on AI-Powered Control Systems in China

**Authors**: *Vishakha Agrawal*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21698](https://doi.org/10.1609/aaai.v36i11.21698)

**Abstract**:

In recent times, the social credit systems (SCS) and similar AI-driven mass surveillance systems have been deployed by the Chinese government in various regions. However, the discussions around the SCS are ambiguous: some people call them very controversial and a breach of human rights, while other people say that the SCS are very similar in structure to the company rankings or background checks on individuals in the United States. In reality, though, there is no monolith and there are different forms of SCS deployed in different regions of China. In this paper, I review the different models of the Chinese SCS. Then, I compare how the different systems are upholding or breaching China’s own AI Ethics guidelines.

----

## [1583] Gradient and Mangitude Based Pruning for Sparse Deep Neural Networks

**Authors**: *Kaleab Belay*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21699](https://doi.org/10.1609/aaai.v36i11.21699)

**Abstract**:

Deep Neural Networks have memory and computational demands that often render them difficult to use in low-resource environments. Also, highly dense networks are over-parameterized and thus prone to overfitting. To address these problems, we introduce a novel algorithm that prunes (sparsifies) weights from the network by taking into account their magnitudes and gradients taken against a validation dataset. Unlike existing pruning methods, our method does not require the network model to be retrained once initial training is completed. On the CIFAR-10 dataset, our method reduced the number of paramters of MobileNet by a factor of 9X, from 14 million to 1.5 million, with just a 3.8% drop in accuracy.

----

## [1584] Spectral DefocusCam: Compressive Hyperspectral Imaging from Defocus Measurements

**Authors**: *Georgia Channing*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21700](https://doi.org/10.1609/aaai.v36i11.21700)

**Abstract**:

Hyperspectral imaging is used for a wide range of tasks from medical diagnostics to crop monitoring, but traditional imagers are prohibitively expensive for widespread use. This research strives to democratize hyperspectral imaging by using machine learning to reconstruct hyperspectral volumes from snapshot imagers. I propose a tunable lens with varying amounts of defocus paired with 31-channel spectral filter array mounted on a CMOS camera. These images are then fed into a reconstruction network that aims to recover the full 31-channel hyperspectral volume from a few encoded images with different amounts of defocus.

----

## [1585] The Importance of Hyperparameter Optimisation for Facial Recognition Applications

**Authors**: *Hannah M. Claus*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21701](https://doi.org/10.1609/aaai.v36i11.21701)

**Abstract**:

This paper explores the importance of using optimisation techniques when tuning a machine learning model. The hyperparameters that need to be determined for the Artificial Neural Network (ANN) to work most efficiently are supposed to find a value that achieves the highest recognition accuracy in a face recognition application. First, the model was trained with manual optimisation of the parameters. The highest recognition accuracy that could be achieved was 96.6% with a specific set of parameters used in the ANN. However, the error rate was at 30%, which was not optimal. After utilising Grid Search as the first automated tuning method for hyperparameters, the recognition accuracy rose to 96.9% and the error rate could be minimised to be less than 1%. Applying Random Search, a recognition accuracy of 98.1% could be achieved with the same error rate. Adding further optimisation to the results from Random Search resulted in receiving an accuracy of 98.2%. Hence, the accuracy of the facial recognition application could be increased by 1.6% by applying automated optimisation methods.
Furthermore, this paper will also deal with common issues in face recognition and focus on potential solutions.

----

## [1586] Measuring Students' Engagement with Digital Interactive Textbooks by Analyzing Clickstream Data

**Authors**: *Breanne Crockett*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21702](https://doi.org/10.1609/aaai.v36i11.21702)

**Abstract**:

This paper provides an overview of my contributions to a project to measure and predict student’s mental workload when using digital interactive textbooks. The current work focuses on analysis of clickstream data from the textbook in search of viewing patterns among students. It was found that students typically fit one of three viewing patterns. These patterns can be used in further research to inform creation of new interactive texts for improved student success.

----

## [1587] Participatory Machine Learning Models in Feminicide News Alert Detection

**Authors**: *Amelia Lee Dogan*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21703](https://doi.org/10.1609/aaai.v36i11.21703)

**Abstract**:

After  criminal  recidivism  or  hiring  machine  learning  mod-els have inflicted harm, participatory machine learning meth-ods are often used as a corrective positioning. However, lit-tle guidance exists on how to develop participatory machinelearning  models  throughout  stages  of  the  machine  learningdevelopment  life-cycle.  Here  we  demonstrate  how  to  co-design  and  partner  with  community  groups,  in  the  specificcase of feminicide data activism. We co-designed and piloteda  machine  learning  model  for  the  detection  of  media  arti-cles  about  feminicide.  This  provides  a  feminist  perspectiveon  practicing  participatory  methods  in  a  co-creation  mind-set for the real-world scenario of monitoring violence againstwomen.

----

## [1588] Robust Rule Learning for Reliable and Interpretable Insight into Expertise Transfer Opportunities

**Authors**: *Willa Potosnak*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21704](https://doi.org/10.1609/aaai.v36i11.21704)

**Abstract**:

Intensive care in hospitals is distributed to different units that care for patient populations reflecting specific comorbidities, treatments, and outcomes. Unit expertise can be shared to potentially improve the quality of methods and outcomes for patients across units. We propose an algorithmic rule pruning approach for use in building short lists of human-interpretable rules that reliably identify patient beneficiaries of expertise transfers in the form of machine learning risk models. Our experimental results, obtained with two intensive care monitoring datasets, demonstrate the potential utility of the proposed method in practice.

----

## [1589] Towards Multimodal Vision-Language Models Generating Non-generic Text

**Authors**: *Wes Robbins*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21705](https://doi.org/10.1609/aaai.v36i11.21705)

**Abstract**:

While text generated by current vision-language models may be accurate and syntactically correct, it is often general.  Recent work has used optical character recognition to supplement visual information with text extracted from an image. In many cases, using text in the image improves the specificity and usefulness of generated text. We contend that vision-language models can benefit from additional information extracted from an image. We modify previous multimodal frameworks to accept relevant information from a number of auxiliary classifiers. In particular, we focus on person names as an additional set of tokens and create a novel image-caption dataset to facilitate captioning with person names. The dataset, Politicians and Athletes in Captions (PAC), consists of captioned images of well-known people in context. By fine-tuning pretrained models with this dataset, we demonstrate a model that can naturally integrate facial recognition tokens into generated text by training on limited data.

----

## [1590] Deep Learning for Personalized Preoperative Planning of Microsurgical Free Tissue Transfers

**Authors**: *Eshika Saxena*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21706](https://doi.org/10.1609/aaai.v36i11.21706)

**Abstract**:

Breast reconstruction surgery requires extensive planning, usually with a CT scan that helps surgeons identify which vessels are suitable for harvest. Currently, there is no quantitative method for preoperative planning. In this work, we successfully develop a Deep Learning algorithm to segment the vessels within the region of interest for breast reconstruction. Ultimately, this information will be used to determine the optimal reconstructive method (choice of vessels, extent of the free flap/harvested tissue) to reduce intra- and postoperative complication rates.

----

## [1591] Using Random Perturbations to Mitigate Adversarial Attacks on NLP Models

**Authors**: *Abigail Swenor*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21707](https://doi.org/10.1609/aaai.v36i11.21707)

**Abstract**:

Deep learning models have excelled in solving many problems in Natural Language Processing, but are susceptible to extensive vulnerabilities. We offer a solution to this vulnerability by using random perturbations such as spelling correction, synonym substitution, or dropping the word. These perturbations are applied to random words in random sentences to defend NLP models against adversarial attacks. Our defense methods are successful in returning attacked models to their original accuracy within statistical significance.

----

## [1592] Unsupervised Identification of Materials with Hyperspectral Images

**Authors**: *Mira Welner*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21708](https://doi.org/10.1609/aaai.v36i11.21708)

**Abstract**:

We introduce a novel technique to identify three spectra representing the three primary materials in a hyperspectral image of a scene. We accomplish this using a modified autoencoder. Further research will be conducted to verify the accuracy of these spectra.

----

## [1593] An Extraction and Representation Pipeline for Literary Characters

**Authors**: *Funing Yang*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21709](https://doi.org/10.1609/aaai.v36i11.21709)

**Abstract**:

Readers of novels need to identify and learn about the characters as they develop an understanding of the plot. The paper presents an end-to-end automated pipeline for literary character identification and ongoing work for extracting and comparing character representations for full-length English novels. The character identification pipeline involves a named entity recognition (NER) module with F1 score of 0.85, a coreference resolution module with F1 score of 0.76, and a disambiguation module using both heuristic and algorithmic approaches. Ongoing work compares event extraction as well as speech extraction pipelines for literary characters representations with case studies. The paper is the first to my knowledge that combines a modular pipeline for automated character identification, representation extraction and comparisons for full-length English novels.

----

## [1594] Building Goal-Oriented Dialogue Systems with Situated Visual Context

**Authors**: *Sanchit Agarwal, Jan Jezabek, Arijit Biswas, Emre Barut, Bill Gao, Tagyoung Chung*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21710](https://doi.org/10.1609/aaai.v36i11.21710)

**Abstract**:

Goal-oriented dialogue agents can comfortably utilize the conversational context and understand its users' goals. However, in visually driven user experiences, these conversational agents are also required to make sense of the screen context in order to provide a proper interactive experience. In this paper, we propose a novel multimodal conversational framework where the dialogue agent's next action and their arguments are derived jointly conditioned both on the conversational and the visual context. We demonstrate the proposed approach via a prototypical furniture shopping experience for a multimodal virtual assistant.

----

## [1595] PYLON: A PyTorch Framework for Learning with Constraints

**Authors**: *Kareem Ahmed, Tao Li, Thy Ton, Quan Guo, Kai-Wei Chang, Parisa Kordjamshidi, Vivek Srikumar, Guy Van den Broeck, Sameer Singh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21711](https://doi.org/10.1609/aaai.v36i11.21711)

**Abstract**:

Deep learning excels at learning task information from large amounts of data, but struggles with learning from declarative high-level knowledge that can be more succinctly expressed directly. In this work, we introduce PYLON, a neuro-symbolic training framework that builds on PyTorch to augment procedurally trained models with declaratively specified knowledge. PYLON lets users programmatically specify constraints as Python functions and compiles them into a differentiable loss, thus training predictive models that fit the data whilst satisfying the specified constraints. PYLON includes both exact as well as approximate compilers to efficiently compute the loss, employing fuzzy logic, sampling methods, and circuits, ensuring scalability even to complex models and constraints. Crucially, a guiding principle in designing PYLON is the ease with which any existing deep learning codebase can be extended to learn from constraints in a few lines code: a  function  that  expresses  the  constraint,  and  a single line to compile it into a loss. Our demo comprises of models in NLP, computer vision, logical games, and knowledge graphs that can be interactively trained using constraints as supervision.

----

## [1596] A Goal-Driven Natural Language Interface for Creating Application Integration Workflows

**Authors**: *Michelle Brachman, Christopher Bygrave, Tathagata Chakraborti, Arunima Chaudhary, Zhining Ding, Casey Dugan, David Gros, Thomas Gschwind, James M. Johnson, Jim Laredo, Christoph Miksovic, Qian Pan, Priyanshu Rai, Ramkumar Ramalingam, Paolo Scotton, Nagarjuna Surabathina, Kartik Talamadupula*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21712](https://doi.org/10.1609/aaai.v36i11.21712)

**Abstract**:

Web applications and services are increasingly important in a distributed internet filled with diverse cloud services and applications, each of which enable the completion of narrowly defined tasks. Given the explosion in the scale and diversity of such services, their composition and integration for achieving complex user goals remains a challenging task for end-users and requires a lot of development effort when specified by hand. We present a demonstration of the Goal Oriented Flow Assistant (GOFA) system, which provides a natural language solution to generate workflows for application integration. Our tool is built on a three-step pipeline: it first uses Abstract Meaning Representation (AMR) to parse utterances; it then uses a knowledge graph to validate candidates; and finally uses an AI planner to compose the candidate flow. We provide a video demonstration of the deployed system as part of our submission.

----

## [1597] UCSM-DNN: User and Card Style Modeling with Deep Neural Networks for Personalized Game AI

**Authors**: *Daegeun Choe, Youngbak Jo, Shindong Kang, Shounan An, Insoo Oh*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21713](https://doi.org/10.1609/aaai.v36i11.21713)

**Abstract**:

This paper tries to resolve long waiting time to find a matching person in player versus player mode of online sports games, such as baseball, soccer and basketball. In player versus player mode, game playing AI which is instead of player needs to be not just smart as human but also show variety to improve user experience against AI. Therefore a need to design game playing AI agents with diverse personalized styles rises. To this end, we propose a personalized game AI which encodes user style vectors and card style vectors with a general DNN, named UCSM-DNN. Extensive experiments show that UCSM-DNN shows improved performance in terms of personalized styles, which enrich user experiences. UCSM-DNN has already been integrated into popular mobile baseball game: MaguMagu 2021 as personalized game AI.

----

## [1598] AI Assisted Data Labeling with Interactive Auto Label

**Authors**: *Michael Desmond, Michelle Brachman, Evelyn Duesterwald, Casey Dugan, Narendra Nath Joshi, Qian Pan, Carolina Spina*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21714](https://doi.org/10.1609/aaai.v36i11.21714)

**Abstract**:

We demonstrate an AI assisted data labeling system which applies unsupervised and semi-supervised machine learning to facilitate accurate and efficient labeling of large data sets. Our system (1) applies representative data sampling and active learning in order to seed and maintain a semi-supervised learner that assists the human labeler (2) provides visual labeling assistance and optimizes labeling mechanics using predicted labels (3) seamlessly updates and learns from ongoing human labeling activity (4) captures and presents metrics that indicate the quality of labeling assistance, and (5) provides an interactive auto labeling interface to group, review and apply predicted labels in a scalable manner.

----

## [1599] CrowdFL: A Marketplace for Crowdsourced Federated Learning

**Authors**: *Daifei Feng, Cicilia Helena, Wei Yang Bryan Lim, Jer Shyuan Ng, Hongchao Jiang, Zehui Xiong, Jiawen Kang, Han Yu, Dusit Niyato, Chunyan Miao*

**Conference**: *aaai 2022*

**URL**: [https://doi.org/10.1609/aaai.v36i11.21715](https://doi.org/10.1609/aaai.v36i11.21715)

**Abstract**:

Amid data privacy concerns, Federated Learning (FL) has emerged as a promising machine learning paradigm that enables privacy-preserving collaborative model training. However, there exists a need for a platform that matches data owners (supply) with model requesters (demand). In this paper, we present CrowdFL, a platform to facilitate the crowdsourcing of FL model training. It coordinates client selection, model training, and reputation management, which are essential steps for the FL crowdsourcing operations. By implementing model training on actual mobile devices, we demonstrate that the platform improves model performance and training efficiency. To the best of our knowledge, it is the first platform to support crowdsourcing-based FL on edge devices.

----



[Go to the previous page](AAAI-2022-list07.md)

[Go to the next page](AAAI-2022-list09.md)

[Go to the catalog section](README.md)