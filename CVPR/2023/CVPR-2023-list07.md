## [1200] PET-NeuS: Positional Encoding Tri-Planes for Neural Surfaces

        **Authors**: *Yiqun Wang, Ivan Skorokhodov, Peter Wonka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01212](https://doi.org/10.1109/CVPR52729.2023.01212)

        **Abstract**:

        A signed distance function (SDF) parametrized by an MLP is a common ingredient of neural surface reconstruction. We build on the successful recent method NeuS to extend it by three new components. The first component is to borrow the tri-plane representation from EG3D and represent signed distance fields as a mixture of tri-planes and MLPs instead of representing it with MLPs only. Using tri-planes leads to a more expressive data structure but will also introduce noise in the reconstructed surface. The second component is to use a new type of positional encoding with learnable weights to combat noise in the reconstruction process. We divide the features in the tri-plane into multiple frequency scales and modulate them with sin and cos functions of different frequencies. The third component is to use learnable convolution operations on the tri-plane features using self-attention convolution to produce features with different frequency bands. The experiments show that PET-NeuS achieves high-fidelity surface reconstruction on standard datasets. Following previous work and using the Chamfer metric as the most important way to measure surface reconstruction quality, we are able to improve upon the NeuS baseline by 57% on Nerf-synthetic (0.84 compared to 1.97) and by 15.5% on DTU (0.71 compared to 0.84). The qualitative evaluation reveals how our method can better control the interference of high-frequency noise.

        ----

        ## [1201] RenderDiffusion: Image Diffusion for 3D Reconstruction, Inpainting and Generation

        **Authors**: *Titas Anciukevicius, Zexiang Xu, Matthew Fisher, Paul Henderson, Hakan Bilen, Niloy J. Mitra, Paul Guerrero*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01213](https://doi.org/10.1109/CVPR52729.2023.01213)

        **Abstract**:

        Diffusion models currently achieve state-of-the-art performance for both conditional and unconditional image generation. However, so far, image diffusion models do not support tasks required for 3D understanding, such as view-consistent 3D generation or single-view object reconstruction. In this paper, we present RenderDiffusion, the first diffusion model for 3D generation and inference, trained using only monocular 2D supervision. Central to our method is a novel image denoising architecture that generates and renders an intermediate three-dimensional representation of a scene in each denoising step. This enforces a strong inductive structure within the diffusion process, providing a 3D consistent representation while only requiring 2D supervision. The resulting 3D representation can be rendered from any view. We evaluate RenderDiffusion on FFHQ, AFHQ, ShapeNet and CLEVR datasets, showing competitive performance for generation of 3D scenes and inference of 3D scenes from 2D images. Additionally, our diffusion-based approach allows us to use 2D inpainting to edit 3D scenes.

        ----

        ## [1202] Score Jacobian Chaining: Lifting Pretrained 2D Diffusion Models for 3D Generation

        **Authors**: *Haochen Wang, Xiaodan Du, Jiahao Li, Raymond A. Yeh, Greg Shakhnarovich*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01214](https://doi.org/10.1109/CVPR52729.2023.01214)

        **Abstract**:

        A diffusion model learns to predict a vector field of gradients. We propose to apply chain rule on the learned gradients, and back-propagate the score of a diffusion model through the Jacobian of a differentiable renderer, which we instantiate to be a voxel radiance field. This setup aggregates 2D scores at multiple camera viewpoints into a 3D score, and re-purposes a pretrained 2D model for 3D data generation. We identify a technical challenge of distribution mismatch that arises in this application, and propose a novel estimation mechanism to resolve it. We run our algorithm on several off-the-shelf diffusion image generative models, including the recently released Stable Diffusion trained on the large-scale LAION 5B dataset.

        ----

        ## [1203] Infinite Photorealistic Worlds Using Procedural Generation

        **Authors**: *Alexander Raistrick, Lahav Lipson, Zeyu Ma, Lingjie Mei, Mingzhe Wang, Yiming Zuo, Karhan Kayan, Hongyu Wen, Beining Han, Yihan Wang, Alejandro Newell, Hei Law, Ankit Goyal, Kaiyu Yang, Jia Deng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01215](https://doi.org/10.1109/CVPR52729.2023.01215)

        **Abstract**:

        We introduce Infinigen, a procedural generator of photorealistic 3D scenes of the natural world. Infinigen is entirely procedural: every asset, from shape to texture, is generated from scratch via randomized mathematical rules, using no external source and allowing infinite variation and composition. Infinigen offers broad coverage of objects and scenes in the natural world including plants, animals, terrains, and natural phenomena such as fire, cloud, rain, and snow. Infinigen can be used to generate unlimited, diverse training data for a wide range of computer vision tasks including object detection, semantic segmentation, optical flow, and 3D reconstruction. We expect Infinigen to be a useful resource for computer vision research and beyond. Please visit infinigen.org for videos, code and pre-generated data.

        ----

        ## [1204] Diffusion-SDF: Text-to-Shape via Voxelized Diffusion

        **Authors**: *Muheng Li, Yueqi Duan, Jie Zhou, Jiwen Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01216](https://doi.org/10.1109/CVPR52729.2023.01216)

        **Abstract**:

        With the rising industrial attention to 3D virtual mod-eling technology, generating novel 3D content based on specified conditions (e.g. text) has become a hot issue. In this paper, we propose a new generative 3D modeling framework called Diffusion-SDF for the challenging task of text-to-shape synthesis. Previous approaches lack flexibility in both 3D data representation and shape generation, thereby failing to generate highly diversified 3D shapes conforming to the given text descriptions. To address this, we propose a SDF autoencoder together with the voxelized Diffusion model to learn and generate representations for voxelized signed distance fields (SDFs) of 3D shapes. Specifically, we design a novel Uinll-Net architecture that implants a local-focused inner network inside the standard U-Net architecture, which enables better reconstruction of patch-independent SDF representations. We extend our approach to further text-to-shape tasks including text-conditioned shape completion and manipulation. Experimental results show that Diffusion-SDF generates both higher quality and more diversified 3D shapes that conform well to given text descriptions when compared to previous approaches. Code is available at: https://github.com/ttlmh/Diffusion-SDF.

        ----

        ## [1205] 3D-Aware Multi-Class Image-to-Image Translation with NeRFs

        **Authors**: *Senmao Li, Joost van de Weijer, Yaxing Wang, Fahad Shahbaz Khan, Meiqin Liu, Jian Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01217](https://doi.org/10.1109/CVPR52729.2023.01217)

        **Abstract**:

        Recent advances in 3D-aware generative models (3D-aware GANs) combined with Neural Radiance Fields (NeRF) have achieved impressive results. However no prior works investigate 3D-aware GANs for 3D consistent multiclass image-to-image (3D-aware 121) translation. Naively using 2D-121 translation methods suffers from unrealistic shape/identity change. To perform 3D-aware multiclass 121 translation, we decouple this learning process into a multiclass 3D-aware GAN step and a 3D-aware 121 translation step. In the first step, we propose two novel techniques: a new conditional architecture and an effective training strategy. In the second step, based on the well-trained multiclass 3D-aware GAN architecture, that preserves view-consistency, we construct a 3D-aware 121 translation system. To further reduce the view-consistency problems, we propose several new techniques, including a U-net-like adaptor network design, a hierarchical representation constrain and a relative regularization loss. In exten-sive experiments on two datasets, quantitative and qualitative results demonstrate that we successfully perform 3D-aware 121 translation with multi-view consistency. Code is available in 3DI2I.

        ----

        ## [1206] Latent-NeRF for Shape-Guided Generation of 3D Shapes and Textures

        **Authors**: *Gal Metzer, Elad Richardson, Or Patashnik, Raja Giryes, Daniel Cohen-Or*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01218](https://doi.org/10.1109/CVPR52729.2023.01218)

        **Abstract**:

        Text-guided image generation has progressed rapidly in recent years, inspiring major breakthroughs in text-guided shape generation. Recently, it has been shown that using score distillation, one can successfully text-guide a NeRF model to generate a 3D object. We adapt the score distillation to the publicly available, and computationally efficient, Latent Diffusion Models, which apply the entire diffusion process in a compact latent space of a pretrained autoencoder. As NeRFs operate in image space, a naive solution for guiding them with latent score distillation would require encoding to the latent space at each guidance step. Instead, we propose to bring the NeRF to the latent space, resulting in a Latent-NeRF. Analyzing our Latent-NeRF, we show that while Text-to-3D models can generate impressive results, they are inherently unconstrained and may lack the ability to guide or enforce a specific 3D structure. To assist and direct the 3D generation, we propose to guide our Latent-NeRF using a Sketch-Shape: an abstract geometry that defines the coarse structure of the desired object. Then, we present means to integrate such a constraint directly into a Latent-NeRF. This unique combination of text and shape guidance allows for increased control over the generation process. We also show that latent score distillation can be successfully applied directly on 3D meshes. This allows for generating high-quality textures on a given geometry. Our experiments validate the power of our different forms of guidance and the efficiency of using latent rendering.

        ----

        ## [1207] Balanced Spherical Grid for Egocentric View Synthesis

        **Authors**: *Changwoon Choi, Sang Min Kim, Young Min Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01592](https://doi.org/10.1109/CVPR52729.2023.01592)

        **Abstract**:

        We present EgoNeRF, a practical solution to reconstruct large-scale real-world environments for VR assets. Given a few seconds of casually captured 360 video, EgoNeRF can efficiently build neural radiance fields. Motivated by the recent acceleration of NeRF using feature grids, we adopt spherical coordinate instead of conventional Cartesian coordinate. Cartesian feature grid is inefficient to represent large-scale unbounded scenes because it has a spatially uniform resolution, regardless of distance from viewers. The spherical parameterization better aligns with the rays of egocentric images, and yet enables factorization for performance enhancement. However, the naive spherical grid suffers from singularities at two poles, and also cannot represent unbounded scenes. To avoid singularities near poles, we combine two balanced grids, which results in a quasi-uniform angular grid. We also partition the radial grid exponentially and place an environment map at infinity to represent unbounded scenes. Furthermore, with our resampling technique for grid-based methods, we can increase the number of valid samples to train NeRF volume. We extensively evaluate our method in our newly introduced synthetic and real-world egocentric 360 video datasets, and it consistently achieves state-of-the-art performance.

        ----

        ## [1208] Local 3D Editing via 3D Distillation of CLIP Knowledge

        **Authors**: *Junha Hyung, Sungwon Hwang, Daejin Kim, Hyunji Lee, Jaegul Choo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01219](https://doi.org/10.1109/CVPR52729.2023.01219)

        **Abstract**:

        3D content manipulation is an important computer vision task with many real-world applications (e.g., product design, cartoon generation, and 3D Avatar editing). Recently proposed 3D CANs can generate diverse photorealistic 3D-aware contents using Neural Radiance fields (NeRF). However, manipulation of NeRF still remains a challenging problem since the visual quality tends to degrade after manipulation and suboptimal control handles such as 2D semantic maps are used for manipulations. While text-guided manipulations have shown potential in 3D editing, such approaches often lack locality. To overcome these problems, we propose Local Editing NeRF (LENeRF), which only requires text inputs for fine-grained and localized manipulation. Specifically, we present three add-on modules of LENeRF, the Latent Residual Mapper, the Attention Field Network, and the Deformation Network, which are jointly usedfor local manipulations of 3D features by estimating a 3D attention field. The 3D attention field is learned in an unsupervised way, by distilling the zero-shot mask generation capability of CLIP to the 3D space with multi-view guidance. We conduct diverse experiments and thorough evaluations both quantitatively and qualitatively.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
We will make our code publicly available.

        ----

        ## [1209] ShapeTalk: A Language Dataset and Framework for 3D Shape Edits and Deformations

        **Authors**: *Panos Achlioptas, Ian Huang, Minhyuk Sung, Sergey Tulyakov, Leonidas J. Guibas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01220](https://doi.org/10.1109/CVPR52729.2023.01220)

        **Abstract**:

        Editing 3D geometry is a challenging task requiring specialized skills. In this work, we aim to facilitate the task of editing the geometry of 3D models through the use of natural language. For example, we may want to modify a 3D chair model to “make its legs thinner” or to “open a hole in its back”. To tackle this problem in a manner that promotes open-ended language use and enables fine-grained shape edits, we introduce the most extensive existing corpus of natural language utterances describing shape differences: ShapeTalk. ShapeTalk contains over half a million discriminative utterances produced by contrasting the shapes of common 3D objects for a variety of object classes and degrees of similarity. We also introduce a generic framework, ChangeIt3D, which builds on ShapeTalk and can use an arbitrary 3D generative model of shapes to produce edits that align the output better with the edit or deformation description. Finally, we introduce metrics for the quantitative evaluation of language-assisted shape editing methods that reflect key desiderata within this editing setup. We note that ShapeTalk allows methods to be trained with explicit 3D-to-language data, bypassing the necessity of “lifting” 2D to 3D using methods like neural rendering, as required by extant 2D image-language foundation models. Our code and data are publicly available at https://changeit3d.github.io/.

        ----

        ## [1210] CoralStyleCLIP: Co-optimized Region and Layer Selection for Image Editing

        **Authors**: *Ambareesh Revanur, Debraj Basu, Shradha Agrawal, Dhwanit Agarwal, Deepak Pai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01221](https://doi.org/10.1109/CVPR52729.2023.01221)

        **Abstract**:

        Edit fidelity is a significant issue in open-world controllable generative image editing. Recently, CLIP-based approaches have traded off simplicity to alleviate these problems by introducing spatial attention in a handpicked layer of a StyleGAN. In this paper, we propose CoralStyleCLIP, which incorporates a multi-layer attention-guided blending strategy in the feature space of StyleGAN2 for obtaining high-fidelity edits. We propose multiple forms of our co-optimized region and layer selection strategy to demonstrate the variation of time complexity with the quality of edits over different architectural intricacies while preserving simplicity. We conduct extensive experimental analysis and benchmark our method against state-of-the-art CLIP-based methods. Our findings suggest that CoralStyleCLIP results in high-quality edits while preserving the ease of use.

        ----

        ## [1211] 3D-Aware Face Swapping

        **Authors**: *Yixuan Li, Chao Ma, Yichao Yan, Wenhan Zhu, Xiaokang Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01222](https://doi.org/10.1109/CVPR52729.2023.01222)

        **Abstract**:

        Face swapping is an important research topic in computer vision with wide applications in entertainment and privacy protection. Existing methods directly learn to swap 2D facial images, taking no account of the geometric information of human faces. In the presence of large pose variance between the source and the target faces, there always exist undesirable artifacts on the swapped face. In this paper, we present a novel 3D-aware face swapping method that generates high-fidelity and multi-view-consistent swapped faces from single-view source and target images. To achieve this, we take advantage of the strong geometry and texture prior of 3D human faces, where the 2D faces are projected into the latent space of a 3D generative model. By disentangling the identity and attribute features in the latent space, we succeed in swapping faces in a 3D-aware manner, being robust to pose variations while transferring fine-grained facial details. Extensive experiments demonstrate the superiority of our 3D-aware face swapping framework in terms of visual quality, identity similarity, and multi-view consistency. Code is available at https://lyx0208.github.io/3dSwap.

        ----

        ## [1212] DCFace: Synthetic Face Generation with Dual Condition Diffusion Model

        **Authors**: *Minchul Kim, Feng Liu, Anil K. Jain, Xiaoming Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01223](https://doi.org/10.1109/CVPR52729.2023.01223)

        **Abstract**:

        Generating synthetic datasets for training face recognition models is challenging because dataset generation entails more than creating high fidelity images. It involves generating multiple images of same subjects under different factors (e.g., variations in pose, illumination, expression, aging and occlusion) which follows the real image conditional distribution. Previous works have studied the generation of synthetic datasets using GAN or 3D models. In this work, we approach the problem from the aspect of combining subject appearance (ID) and external factor (style) conditions. These two conditions provide a direct way to control the inter-class and intra-class variations. To this end, we propose a Dual Condition Face Generator (DCFace) based on a diffusion model. Our novel Patch-wise style extractor and Time-step dependent ID loss enables DCFace to consistently produce face images of the same subject under different styles with precise control. Face recognition models trained on synthetic images from the proposed DCFace provide higher verification accuracies compared to previous works by 6.11% on average in 4 out of 5 test datasets, LFW, CFP-FP, CPLFW, AgeDB and CALFW. Code Link

        ----

        ## [1213] HairStep: Transfer Synthetic to Real Using Strand and Depth Maps for Single-View 3D Hair Modeling

        **Authors**: *Yujian Zheng, Zirong Jin, Moran Li, Haibin Huang, Chongyang Ma, Shuguang Cui, Xiaoguang Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01224](https://doi.org/10.1109/CVPR52729.2023.01224)

        **Abstract**:

        In this work, we tackle the challenging problem of learning-based single-view 3D hair modeling. Due to the great difficulty of collecting paired real image and 3D hair data, using synthetic data to provide prior knowledge for real domain becomes a leading solution. This unfortunately introduces the challenge of domain gap. Due to the inherent difficulty of realistic hair rendering, existing methods typically use orientation maps instead of hair images as input to bridge the gap. We firmly think an intermediate representation is essential, but we argue that orientation map using the dominant filtering-based methods is sensitive to uncertain noise and far from a competent representation. Thus, we first raise this issue up and propose a novel intermediate representation, termed as HairStep, which consists of a strand map and a depth map. It is found that HairStep not only provides sufficient information for accurate 3D hair modeling, but also is feasible to be inferred from real images. Specifically, we collect a dataset of 1,250 portrait images with two types of annotations. A learning framework is further designed to transfer real images to the strand map and depth map. It is noted that, an extra bonus of our new dataset is the first quantitative metric for 3D hair modeling. Our experiments show that HairStep narrows the domain gap between synthetic and real and achieves state-of-the-art performance on single-view 3D hair reconstruction.

        ----

        ## [1214] DiffusionRig: Learning Personalized Priors for Facial Appearance Editing

        **Authors**: *Zheng Ding, Xuaner Zhang, Zhihao Xia, Lars Jebe, Zhuowen Tu, Xiuming Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01225](https://doi.org/10.1109/CVPR52729.2023.01225)

        **Abstract**:

        We address the problem of learning person-specific facial priors from a small number (e.g., 20) of portrait photos of the same person. This enables us to edit this specific person's facial appearance, such as expression and lighting, while preserving their identity and high-frequency facial details. Key to our approach, which we dub DiffusionRig, is a diffusion model conditioned on, or “rigged by,“ crude 3D face models estimated from single in-the-wild images by an off-the-shelf estimator. On a high level, DiffusionRig learns to map simplistic renderings of 3D face models to realistic photos of a given person. Specifically, DiffusionRig is trained in two stages: It first learns generic facial priors from a large-scale face dataset and then person-specific priors from a small portrait photo collection of the person of interest. By learning the CGI-to-photo mapping with such personalized priors,DiffusionRig can “rig“ the lighting, facial expression, head pose, etc. of a portrait photo, conditioned only on coarse 3D models while preserving this person's identity and other high-frequency characteristics. Qualitative and quantitative experiments show that DiffusionRig outperforms existing approaches in both identity preservation and photorealism. Please see the project website: https://diffusionrig.github.io for the supplemental material, video, code, and data.

        ----

        ## [1215] 3D-aware Facial Landmark Detection via Multi-view Consistent Training on Synthetic Data

        **Authors**: *Libing Zeng, Lele Chen, Wentao Bao, Zhong Li, Yi Xu, Junsong Yuan, Nima K. Kalantari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01226](https://doi.org/10.1109/CVPR52729.2023.01226)

        **Abstract**:

        Accurate facial landmark detection on wild images plays an essential role in human-computer interaction, entertainment, and medical applications. Existing approaches have limitations in enforcing 3D consistency while detecting 3D/2D facial landmarks due to the lack of multi-view in-the-wild training data. Fortunately, with the recent advances in generative visual models and neural rendering, we have witnessed rapid progress towards high quality 3D image synthesis. In this work, we leverage such approaches to construct a synthetic dataset and propose a novel multi-view consistent learning strategy to improve 3D facial landmark detection accuracy on in-the-wild images. The proposed 3D-aware module can be plugged into any learning-based landmark detection algorithm to enhance its accuracy. We demonstrate the superiority of the proposed plug-in module with extensive comparison against state-of-the-art methods on several real and synthetic datasets.

        ----

        ## [1216] Parametric Implicit Face Representation for Audio-Driven Facial Reenactment

        **Authors**: *Ricong Huang, Peiwen Lai, Yipeng Qin, Guanbin Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01227](https://doi.org/10.1109/CVPR52729.2023.01227)

        **Abstract**:

        Audio-driven facial reenactment is a crucial technique that has a range of applications in film-making, virtual avatars and video conferences. Existing works either employ explicit intermediate face representations (e.g., 2D facial landmarks or 3D face models) or implicit ones (e.g., Neural Radiance Fields), thus suffering from the trade-offs between interpretability and expressive power, hence between controllability and quality of the results. In this work, we break these trade-offs with our novel parametric implicit face representation and propose a novel audio-driven facial reenactment framework that is both controllable and can generate high-quality talking heads. Specifically, our parametric implicit representation parameterizes the implicit representation with interpretable parameters of 3D face models, thereby taking the best of both explicit and implicit methods. In addition, we propose several new techniques to improve the three components of our framework, including i) incorporating contextual information into the audio-to-expression parameters encoding; ii) using conditional image synthesis to parameterize the implicit representation and implementing it with an innovative tri-plane structure for efficient learning; iii) formulating facial reenactment as a conditional image inpainting problem and proposing a novel data augmentation technique to improve model generalizability. Extensive experiments demonstrate that our method can generate more realistic results than previous methods with greater fidelity to the identities and talking styles of speakers.

        ----

        ## [1217] MEGANE: Morphable Eyeglass and Avatar Network

        **Authors**: *Junxuan Li, Shunsuke Saito, Tomas Simon, Stephen Lombardi, Hongdong Li, Jason M. Saragih*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01228](https://doi.org/10.1109/CVPR52729.2023.01228)

        **Abstract**:

        Eyeglasses play an important role in the perception of identity. Authentic virtual representations of faces can benefit greatly from their inclusion. However, modeling the geometric and appearance interactions of glasses and the face of virtual representations of humans is challenging. Glasses and faces affect each other's geometry at their contact points, and also induce appearance changes due to light transport. Most existing approaches do not capture these physical interactions since they model eyeglasses and faces independently. Others attempt to resolve interactions as a 2D image synthesis problem and suffer from view and temporal inconsistencies. In this work, we propose a 3D compositional morphable model of eyeglasses that accurately incorporates high-fidelity geometric and photometric interaction effects. To support the large variation in eyeglass topology efficiently, we employ a hybrid representation that combines surface geometry and a volumetric representation. Unlike volumetric approaches, our model naturally retains correspondences across glasses, and hence explicit modification of geometry, such as lens insertion and frame deformation, is greatly simplified. In addition, our model is relightable under point lights and natural illumination, supporting high-fidelity rendering of various frame materials, including translucent plastic and metal within a single morphable model. Importantly, our approach models global light transport effects, such as casting shadows between faces and glasses. Our morphable model for eyeglasses can also be fit to novel glasses via inverse rendering. We compare our approach to state-of-the-art methods and demonstrate significant quality improvements.

        ----

        ## [1218] CodeTalker: Speech-Driven 3D Facial Animation with Discrete Motion Prior

        **Authors**: *Jinbo Xing, Menghan Xia, Yuechen Zhang, Xiaodong Cun, Jue Wang, Tien-Tsin Wong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01229](https://doi.org/10.1109/CVPR52729.2023.01229)

        **Abstract**:

        Speech-driven 3D facial animation has been widely studied, yet there is still a gap to achieving realism and vividness due to the highly ill-posed nature and scarcity of audio-visual data. Existing works typically formulate the cross-modal mapping into a regression task, which suffers from the regression-to-mean problem leading to over-smoothed facial motions. In this paper, we propose to cast speech-driven facial animation as a code query task in a finite proxy space of the learned codebook, which effectively promotes the vividness of the generated motions by reducing the cross-modal mapping uncertainty. The codebook is learned by self-reconstruction over real facial motions and thus embedded with realistic facial motion priors. Over the discrete motion space, a temporal autoregressive model is employed to sequentially synthesize facial motions from the input speech signal, which guarantees lip-sync as well as plausible facial expressions. We demonstrate that our approach outperforms current state-of-the-art methods both qualitatively and quantitatively. Also, a user study further justifies our superiority in perceptual quality. Code and video demo are available at https://doubiiu.github.io/projects/codetalker.

        ----

        ## [1219] Reconstructing Signing Avatars from Video Using Linguistic Priors

        **Authors**: *Maria-Paola Forte, Peter Kulits, Chun-Hao Huang, Vasileios Choutas, Dimitrios Tzionas, Katherine J. Kuchenbecker, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01230](https://doi.org/10.1109/CVPR52729.2023.01230)

        **Abstract**:

        Sign language (SL) is the primary method of communication for the 70 million Deaf people around the world. Video dictionaries of isolated signs are a core SL learning tool. Replacing these with 3D avatars can aid learning and enable AR/VR applications, improving access to technology and online media. However, little work has attempted to estimate expressive 3D avatars from SL video; occlusion, noise, and motion blur make this task difficult. We address this by introducing novel linguistic priors that are universally applicable to SL and provide constraints on 3D hand pose that help resolve ambiguities within isolated signs. Our method, SGNify, captures fine-grained hand pose, facial expression, and body movement fully automatically from in-the-wild monocular SL videos. We evaluate SGNify quantitatively by using a commercial motion-capture system to compute 3D avatars synchronized with monocular video. SGNify outperforms state-of-the-art 3D body-pose-and shape-estimation methods on SL videos. A perceptual study shows that SGNify's 3D reconstructions are significantly more comprehensible and natural than those of previous methods and are on par with the source videos. Code and data are available at sgnify.is.tue.mpg.de.

        ----

        ## [1220] HARP: Personalized Hand Reconstruction from a Monocular RGB Video

        **Authors**: *Korrawe Karunratanakul, Sergey Prokudin, Otmar Hilliges, Siyu Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01231](https://doi.org/10.1109/CVPR52729.2023.01231)

        **Abstract**:

        We present HARP (HAnd Reconstruction and Personalization), a personalized hand avatar creation approach that takes a short monocular RGB video of a human hand as input and reconstructs a faithful hand avatar exhibiting a high-fidelity appearance and geometry. In contrast to the major trend of neural implicit representations, HARP models a hand with a mesh-based parametric hand model, a vertex displacement map, a normal map, and an albedo without any neural components. The explicit nature of our representation enables a truly scalable, robust, and efficient approach to hand avatar creation as validated by our experiments. HARP is optimized via gradient descent from a short sequence captured by a hand-held mobile phone and can be directly used in AR/VR applications with realtime rendering capability. To enable this, we carefully design and implement a shadow-aware differentiable rendering scheme that is robust to high degree articulations and self-shadowing regularly present in hand motions, as well as challenging lighting conditions. It also generalizes to unseen poses and novel viewpoints, producing photo-realistic renderings of hand animations. Furthermore, the learned HARP representation can be used for improving 3D hand pose estimation quality in challenging viewpoints. The key advantages of HARP are validated by the in-depth analyses on appearance reconstruction, novel view and novel pose synthesis, and 3D hand pose refinement. It is an AR/VR-ready personalized hand representation that shows superior fidelity and scalability.

        ----

        ## [1221] OmniAvatar: Geometry-Guided Controllable 3D Head Synthesis

        **Authors**: *Hongyi Xu, Guoxian Song, Zihang Jiang, Jianfeng Zhang, Yichun Shi, Jing Liu, Wan-Chun Ma, Jiashi Feng, Linjie Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01232](https://doi.org/10.1109/CVPR52729.2023.01232)

        **Abstract**:

        We present OmniAvatar, a novel geometry-guided 3D head synthesis model trained from in-the-wild unstructured images that is capable of synthesizing diverse identity-preserved 3D heads with compelling dynamic details under full disentangled control over camera poses, facial expressions, head shapes, articulated neck and jaw poses. To achieve such high level of disentangled control, we first explicitly define a novel semantic signed distance function (SDF) around a head geometry (FLAME) conditioned on the control parameters. This semantic SDF allows us to build a differentiable volumetric correspondence map from the observation space to a disentangled canonical space from all the control parameters. We then leverage the 3D-aware GAN framework (EG3D) to synthesize detailed shape and appearance of 3D full heads in the canonical space, followed by a volume rendering step guided by the volumetric correspondence map to output into the observation space. To ensure the control accuracy on the synthesized head shapes and expressions, we introduce a geometry prior loss to conform to head SDF and a control loss to conform to the expression code. Further, we enhance the temporal realism with dynamic details conditioned upon varying expressions and joint poses. Our model can synthesize more preferable identity-preserved 3D heads with compelling dynamic details compared to the state-of-the-art methods both qualitatively and quantitatively. We also provide an ablation study to justify many of our system design choices.

        ----

        ## [1222] RaBit: Parametric Modeling of 3D Biped Cartoon Characters with a Topological-Consistent Dataset

        **Authors**: *Zhongjin Luo, Shengcai Cai, Jinguo Dong, Ruibo Ming, Liangdong Qiu, Xiaohang Zhan, Xiaoguang Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01233](https://doi.org/10.1109/CVPR52729.2023.01233)

        **Abstract**:

        Assisting people in efficiently producing visually plausible 3D characters has always been a fundamental research topic in computer vision and computer graphics. Recent learning-based approaches have achieved unprecedented accuracy and efficiency in the area of 3D real human digitization. However, none of the prior works focus on modeling 3D biped cartoon characters, which are also in great demand in gaming and filming. In this paper, we introduce 3DBiCar, the first large-scale dataset of 3D biped cartoon characters, and RaBit, the corresponding parametric model. Our dataset contains 1,500 topologically consistent high-quality 3D textured models which are manually crafted by professional artists. Built upon the data, RaBit is thus designed with a SMPL-like linear blend shape model and a StyleGAN-based neural UV-texture generator, simultaneously expressing the shape, pose, and texture. To demonstrate the practicality of 3DBiCar and RaBit, various applications are conducted, including single-view reconstruction, sketch-based modeling, and 3D cartoon animation. For the single-view reconstruction setting, we find a straightforward global mapping from input images to the output UV-based texture maps tends to lose detailed appearances of some local parts (e.g., nose, ears). Thus, a part-sensitive texture reasoner is adopted to make all important local areas perceived. Experiments further demonstrate the effectiveness of our method both qualitatively and quantitatively. 3DBiCar and RaBit are available at gaplab.cuhk.edu.cn/projects/RaBit.

        ----

        ## [1223] Transfer4D: A Framework for Frugal Motion Capture and Deformation Transfer

        **Authors**: *Shubh Maheshwari, Rahul Narain, Ramya Hebbalaguppe*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01234](https://doi.org/10.1109/CVPR52729.2023.01234)

        **Abstract**:

        Animating a virtual character based on a real performance of an actor is a challenging task that currently requires expensive motion capture setups and additional effort by expert animators, rendering it accessible only to large production houses. The goal of our work is to democratize this task by developing a frugal alternative termed “Transfer4D” that uses only commodity depth sensors and further reduces animators' effort by automating the rigging and animation transfer process. Our approach can transfer motion from an incomplete, single-view depth video to a semantically similar target mesh, unlike prior works that make a stricter assumption on the source to be noise-free and watertight. To handle sparse, incomplete videos from depth video inputs and variations between source and target objects, we propose to use skeletons as an intermediary representation between motion capture and transfer. We propose a novel unsupervised skeleton extraction pipeline from a single-view depth sequence that incorporates additional geometric information, resulting in superior performance in motion reconstruction and transfer in comparison to the contemporary methods and making our approach generic. We use non-rigid reconstruction to track motion from the depth sequence, and then we rig the source object using skinning decomposition. Finally, the rig is embedded into the target object for motion retargeting.

        ----

        ## [1224] CLOTH4D: A Dataset for Clothed Human Reconstruction

        **Authors**: *Xingxing Zou, Xintong Han, Waikeung Wong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01235](https://doi.org/10.1109/CVPR52729.2023.01235)

        **Abstract**:

        Clothed human reconstruction is the cornerstone for creating the virtual world. To a great extent, the quality of recovered avatars decides whether the Metaverse is a passing fad. In this work, we introduce CLOTH4D, a clothed human dataset containing 1,000 subjects with varied appearances, 1,000 3D outfits, and over 100,000 clothed meshes with paired unclothed humans, to fill the gap in large-scale and high-quality 4D clothing data. It enjoys appealing characteristics: 1) Accurate and detailed clothing textured meshes-all clothing items are manually created and then simulated in professional software, strictly following the general standard in fashion design. 2) Separated textured clothing and under-clothing body meshes, closer to the physical world than single-layer raw scans. 3) Clothed human motion sequences simulated given a set of 289 actions, covering fundamental and complicated dynamics. Upon CLOTH4D, we novelly designed a series of temporally-aware metries to evaluate the temporal stability of the generated 3D human meshes, which has been over-looked previously. Moreover, by assessing and retraining current state-of-the-art clothed human reconstruction methods, we reveal insights, present improved performance, and propose potential future research directions, confirming our dataset's advancement. The dataset is available at.

        ----

        ## [1225] Vid2Avatar: 3D Avatar Reconstruction from Videos in the Wild via Self-supervised Scene Decomposition

        **Authors**: *Chen Guo, Tianjian Jiang, Xu Chen, Jie Song, Otmar Hilliges*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01236](https://doi.org/10.1109/CVPR52729.2023.01236)

        **Abstract**:

        We present Vid2Avatar, a method to learn human avatars from monocular in-the-wild videos. Reconstructing humans that move naturally from monocular in-the-wild videos is difficult. Solving it requires accurately separating humans from arbitrary backgrounds. Moreover, it requires reconstructing detailed 3D surface from short video sequences, making it even more challenging. Despite these challenges, our method does not require any groundtruth supervision or priors extracted from large datasets of clothed human scans, nor do we rely on any external segmentation modules. Instead, it solves the tasks of scene decomposition and surface reconstruction directly in 3D by modeling both the human and the background in the scene jointly, parameterized via two separate neural fields. Specifically, we define a temporally consistent human representation in canonical space and formulate a global optimization over the background model, the canonical human shape and texture, and per-frame human pose parameters. A coarse-to-fine sampling strategy for volume rendering and novel objectives are introduced for a clean separation of dynamic human and static background, yielding detailed and robust 3D human reconstructions. The evaluation of our method shows improvements over prior art on publicly available datasets.

        ----

        ## [1226] High-fidelity 3D Human Digitization from Single 2K Resolution Images

        **Authors**: *Sang-Hun Han, Min-Gyu Park, Ju Hong Yoon, Ju-Mi Kang, Young-Jae Park, Hae-Gon Jeon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01237](https://doi.org/10.1109/CVPR52729.2023.01237)

        **Abstract**:

        High-quality 3D human body reconstruction requires high-fidelity and large-scale training data and appropriate network design that effectively exploits the high-resolution input images. To tackle these problems, we propose a simple yet effective 3D human digitization method called 2K2K, which constructs a large-scale 2K human dataset and infers 3D human models from 2K resolution images. The proposed method separately recovers the global shape of a human and its details. The low-resolution depth network predicts the global structure from a low-resolution image, and the part-wise image-to-normal network predicts the details of the 3D human body structure. The high-resolution depth network merges the global 3D shape and the detailed structures to infer the high-resolution front and back side depth maps. Finally, an off-the-shelf mesh generator reconstructs the full 3D human model, which are available at https://github.com/SangHunHan92/2K2K. In addition, we also provide 2,050 3D human models, including texture maps, 3D joints, and SMPL parameters for research purposes. In experiments, we demonstrate competitive performance over the recent works on various datasets.

        ----

        ## [1227] Sampling is Matter: Point-Guided 3D Human Mesh Reconstruction

        **Authors**: *Jeonghwan Kim, Mi-Gyeong Gwon, Hyunwoo Park, Hyukmin Kwon, Gi-Mun Um, Wonjun Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01238](https://doi.org/10.1109/CVPR52729.2023.01238)

        **Abstract**:

        This paper presents a simple yet powerful method for 3D human mesh reconstruction from a single RGB image. Most recently, the non-local interactions of the whole mesh vertices have been effectively estimated in the transformer while the relationship between body parts also has begun to be handled via the graph model. Even though those approaches have shown the remarkable progress in 3D human mesh reconstruction, it is still difficult to directly infer the relationship between features, which are encoded from the 2D input image, and 3D coordinates of each vertex. To resolve this problem, we propose to design a simple feature sampling scheme. The key idea is to sample features in the embedded space by following the guide of points, which are estimated as projection results of 3D mesh vertices (i.e., ground truth). This helps the model to concentrate more on vertex-relevant features in the 2D space, thus leading to the reconstruction of the natural human pose. Furthermore, we apply progressive attention masking to precisely estimate local interactions between vertices even under severe occlusions. Experimental results on benchmark datasets show that the proposed method efficiently improves the performance of 3D human mesh reconstruction. The code and model are publicly available at: https://github.com/DCVL-3D/PointHMR_release.

        ----

        ## [1228] gSDF: Geometry-Driven Signed Distance Functions for 3D Hand-Object Reconstruction

        **Authors**: *Zerui Chen, Shizhe Chen, Cordelia Schmid, Ivan Laptev*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01239](https://doi.org/10.1109/CVPR52729.2023.01239)

        **Abstract**:

        Signed distance functions (SDFs) is an attractive frame-work that has recently shown promising results for 3D shape reconstruction from images. SDFs seamlessly generalize to different shape resolutions and topologies but lack explicit modelling of the underlying 3D geometry. In this work, we ex-ploit the hand structure and use it as guidance for SDF-based shape reconstruction. In particular, we address reconstruction of hands and manipulated objects from monocular RGB images. To this end, we estimate poses of hands and objects and use them to guide 3D reconstruction. More specifically, we predict kinematic chains of pose transformations and align SDFs with highly-articulated hand poses. We improve the visual features of 3D points with geometry alignment and further leverage temporal information to enhance the robustness to occlusion and motion blurs. We conduct extensive experiments on the challenging ObMan and DexYCB benchmarks and demonstrate significant improvements of the proposed method over the state of the art.

        ----

        ## [1229] Human Body Shape Completion with Implicit Shape and Flow Learning

        **Authors**: *Boyao Zhou, Di Meng, Jean-Sébastien Franco, Edmond Boyer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01240](https://doi.org/10.1109/CVPR52729.2023.01240)

        **Abstract**:

        In this paper, we investigate how to complete human body shape models by combining shape and flow estimation given two consecutive depth images. Shape completion is a challenging task in computer vision that is highly under-constrained when considering partial depth observations. Besides model based strategies that exploit strong priors, and consequently struggle to preserve fine geometric details, learning based approaches build on weaker assumptions and can benefit from efficient implicit representations. We adopt such a representation and explore how the motion flow between two consecutive frames can contribute to the shape completion task. In order to effectively exploit the flow information, our architecture combines both estimations and implements two features for robustness: First, an all-to-all attention module that encodes the correlation between points in the same frame and between corresponding points in different frames; Second, a coarse-dense to fine-sparse strategy that balances the representation ability and the computational cost. Our experiments demonstrate that the flow actually benefits human body model completion. They also show that our method outperforms the state-of-the-art approaches for shape completion on 2 benchmarks, considering different human shapes, poses, and clothing.

        ----

        ## [1230] ShapeClipper: Scalable 3D Shape Learning from Single-View Images via Geometric and CLIP-Based Consistency

        **Authors**: *Zixuan Huang, Varun Jampani, Anh Thai, Yuanzhen Li, Stefan Stojanov, James M. Rehg*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01241](https://doi.org/10.1109/CVPR52729.2023.01241)

        **Abstract**:

        We present ShapeClipper, a novel method that reconstructs 3D object shapes from real-world single-view RGB images. Instead of relying on laborious 3D, multi-view or camera pose annotation, ShapeClipper learns shape reconstruction from a set of single-view segmented images. The key idea is to facilitate shape learning via CLIP-based shape consistency, where we encourage objects with similar CLIP encodings to share similar shapes. We also leverage off-the-shelf normals as an additional geometric constraint so the model can learn better bottom-up reasoning of detailed surface geometry. These two novel consistency constraints, when used to regularize our model, improve its ability to learn both global shape structure and local geometric details. We evaluate our method over three challenging real-world datasets, Pix3D, Pascal3D+, and Open-Images, where we achieve superior performance over state-of-the-art methods.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
project website at: https://zixuanh.com/projects/shapeclipper.html

        ----

        ## [1231] PC2: Projection-Conditioned Point Cloud Diffusion for Single-Image 3D Reconstruction

        **Authors**: *Luke Melas-Kyriazi, Christian Rupprecht, Andrea Vedaldi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01242](https://doi.org/10.1109/CVPR52729.2023.01242)

        **Abstract**:

        Reconstructing the 3D shape of an object from a single RGB image is a long-standing problem in computer vision. In this paper, we propose a novel method for single-image 3D reconstruction which generates a sparse point cloud via a conditional denoising diffusion process. Our method takes as input a single RGB image along with its camera pose and gradually denoises a set of 3D points, whose positions are initially sampled randomly from a three-dimensional Gaussian distribution, into the shape of an object. The key to our method is a geometrically-consistent conditioning process which we call projection conditioning: at each step in the diffusion process, we project local image features onto the partially-denoised point cloud from the given camera pose. This projection conditioning process enables us to generate high-resolution sparse geometries that are well-aligned with the input image and can additionally be used to predict point colors after shape reconstruction. Moreover, due to the probabilistic nature of the diffusion process, our method is naturally capable of generating multiple different shapes consistent with a single input image. In contrast to prior work, our approach not only performs well on synthetic benchmarks but also gives large qualitative improvements on complex real-world data. Data and code are available at https://lukemelas.github.io/projectionconditioned-point-cloud-diffusion/.

        ----

        ## [1232] NIKI: Neural Inverse Kinematics with Invertible Neural Networks for 3D Human Pose and Shape Estimation

        **Authors**: *Jiefeng Li, Siyuan Bian, Qi Liu, Jiasheng Tang, Fan Wang, Cewu Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01243](https://doi.org/10.1109/CVPR52729.2023.01243)

        **Abstract**:

        With the progress of 3D human pose and shape estimation, state-of-the-art methods can either be robust to occlusions or obtain pixel-aligned accuracy in non-occlusion cases. However, they cannot obtain robustness and mesh-image alignment at the same time. In this work, we present NIKI (Neural Inverse Kinematics with Invertible Neural Network), which models bidirectional errors to improve the robustness to occlusions and obtain pixel-aligned accuracy. NIKI can learn from both the forward and inverse processes with invertible networks. In the inverse process, the model separates the error from the plausible 3D pose manifold for a robust 3D human pose estimation. In the forward process, we enforce the zero-error boundary conditions to improve the sensitivity to reliable joint positions for better mesh-image alignment. Furthermore, NIKI emulates the analytical inverse kinematics algorithms with the twist-and-swing decomposition for better interpretability. Experiments on standard and occlusion-specific benchmarks demonstrate the effectiveness of NIKI, where we exhibit robust and well-aligned results simultaneously. Code is available at https://github.com/Jeff-sjtu/NIKI.

        ----

        ## [1233] ARCTIC: A Dataset for Dexterous Bimanual Hand-Object Manipulation

        **Authors**: *Zicong Fan, Omid Taheri, Dimitrios Tzionas, Muhammed Kocabas, Manuel Kaufmann, Michael J. Black, Otmar Hilliges*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01244](https://doi.org/10.1109/CVPR52729.2023.01244)

        **Abstract**:

        Humans intuitively understand that inanimate objects do not move by themselves, but that state changes are typically caused by human manipulation (e.g., the opening of a book). This is not yet the case for machines. In part this is because there exist no datasets with ground-truth 3D annotations for the study of physically consistent and synchronised motion of hands and articulated objects. To this end, we introduce ARCTIC - a dataset of two hands that dexterously manipulate objects, containing 2 .1M video frames paired with accurate 3D hand and object meshes and detailed, dynamic contact information. It contains bi-manual articulation of objects such as scissors or laptops, where hand poses and object states evolve jointly in time. We propose two novel articulated hand-object interaction tasks: (1) Consistent motion reconstruction: Given a monocular video, the goal is to reconstruct two hands and articulated objects in 3D, so that their motions are spatio-temporally consistent. (2) Interaction field estimation: Dense relative hand-object distances must be estimated from images. We introduce two baselines ArcticNet and InterField, respectively and evaluate them qualitatively and quantitatively on ARCTIC. Our code and data are available at https://arctic.is.tue.mpg.de.

        ----

        ## [1234] ACR: Attention Collaboration-based Regressor for Arbitrary Two-Hand Reconstruction

        **Authors**: *Zhengdi Yu, Shaoli Huang, Chen Fang, Toby P. Breckon, Jue Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01245](https://doi.org/10.1109/CVPR52729.2023.01245)

        **Abstract**:

        Reconstructing two hands from monocular RGB images is challenging due to frequent occlusion and mutual confusion. Existing methods mainly learn an entangled representation to encode two interacting hands, which are in-credibly fragile to impaired interaction, such as truncated hands, separate hands, or external occlusion. This paper presents ACR (Attention Collaboration-based Regres-sor), which makes the first attempt to reconstruct hands in arbitrary scenarios. To achieve this, ACR explicitly miti-gates interdependencies between hands and between parts by leveraging center and part-based attention for feature extraction. However, reducing interdependence helps re-lease the input constraint while weakening the mutual reasoning about reconstructing the interacting hands. Thus, based on center attention, ACR also learns cross-hand prior that handle the interacting hands better. We evaluate our method on various types of hand reconstruction datasets. Our method significantly outperforms the best interacting-hand approaches on the InterHand2.6M dataset while yielding comparable performance with the state-of-the-art single-hand methods on the FreiHand dataset. More qualitative results on in-the-wild and hand-object interaction datasets and web images/videos further demonstrate the effectiveness of our approach for arbitrary hand reconstruction. Our code is available at this link
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/ZhengdiYu/Arbitrary-Hands-3D-Reconstruction.

        ----

        ## [1235] MIME: Human-Aware 3D Scene Generation

        **Authors**: *Hongwei Yi, Chun-Hao P. Huang, Shashank Tripathi, Lea Hering, Justus Thies, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01246](https://doi.org/10.1109/CVPR52729.2023.01246)

        **Abstract**:

        Generating realistic 3D worlds occupied by moving humans has many applications in games, architecture, and synthetic data creation. But generating such scenes is expensive and labor intensive. Recent work generates human poses and motions given a 3D scene. Here, we take the opposite approach and generate 3D indoor scenes given 3D human motion. Such motions can come from archival motion capture or from IMU sensors worn on the body, effectively turning human movement into a “scanner” of the 3D world. Intuitively, human movement indicates the free-space in a room and human contact indicates surfaces or objects that support activities such as sitting, lying or touching. We propose MIME (Mining Interaction and Movement to infer 3D Environments), which is a generative model of indoor scenes that produces furniture layouts that are consistent with the human movement. MIME uses an auto-regressive transformer architecture that takes the already generated objects in the scene as well as the human motion as input, and outputs the next plausible object. To train MIME, we build a dataset by populating the 3D FRONT scene dataset with 3D humans. Our experiments show that MIME produces more diverse and plausible 3D scenes than a recent generative scene method that does not know about human movement. Code and data are available for research at https://mime.is.tue.mpg.de.

        ----

        ## [1236] CIMI4D: A Large Multimodal Climbing Motion Dataset under Human-scene Interactions

        **Authors**: *Ming Yan, Xin Wang, Yudi Dai, Siqi Shen, Chenglu Wen, Lan Xu, Yuexin Ma, Cheng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01247](https://doi.org/10.1109/CVPR52729.2023.01247)

        **Abstract**:

        Motion capture is a long-standing research problem. Although it has been studied for decades, the majority of research focus on ground-based movements such as walking, sitting, dancing, etc. Off- grounded actions such as climbing are largely overlooked. As an important type of action in sports and firefighting field, the climbing movements is challenging to capture because of its complex back poses, intricate human-scene interactions, and difficult global localization. The research community does not have an indepth understanding of the climbing action due to the lack of specific datasets. To address this limitation, we collect CIMI4D, a large rock Climbing Motion dataset from 12 persons climbing 13 different climbing walls. The dataset consists of around 180,000 frames of pose inertial measurements, LiDAR point clouds, RGB videos, high-precision static point cloud scenes, and reconstructed scene meshes. Moreover, we frame-wise annotate touch rock holds to facilitate a detailed exploration of human-scene interaction. The core of this dataset is a blending optimization process, which corrects for the pose as it drifts and is affected by the magnetic conditions. To evaluate the merit of CIMI4D, we perform four tasks which include human pose estimations (with/without scene constraints), pose prediction, and pose generation. The experimental results demonstrate that CIMI4D presents great challenges to existing methods and enables extensive research opportunities. We share the dataset with the research community in http://www.lidarhumanmotion.net/cimi4d/.

        ----

        ## [1237] Harmonious Feature Learning for Interactive Hand-Object Pose Estimation

        **Authors**: *Zhifeng Lin, Changxing Ding, Huan Yao, Zengsheng Kuang, Shaoli Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01248](https://doi.org/10.1109/CVPR52729.2023.01248)

        **Abstract**:

        Joint hand and object pose estimation from a single image is extremely challenging as serious occlusion often occurs when the hand and object interact. Existing approaches typically first extract coarse hand and object features from a single backbone, then further enhance them with reference to each other via interaction modules. However, these works usually ignore that the hand and object are competitive in feature learning, since the backbone takes both of them as foreground and they are usually mutually occluded. In this paper, we propose a novel Harmonious Feature Learning Network (HFL-Net). HFL-Net introduces a new framework that combines the advantages of single-and double-stream backbones: it shares the parameters of the low-and high-level convolutional layers of a common ResNet-50 model for the hand and object, leaving the middle-level layers unshared. This strategy enables the hand and the object to be extracted as the sole targets by the middle-level layers, avoiding their competition in feature learning. The shared high-level layers also force their features to be harmonious, thereby facilitating their mutual feature enhancement. In particular, we propose to enhance the feature of the hand via concatenation with the feature in the same location from the object stream. A subsequent self-attention layer is adopted to deeply fuse the concatenated feature. Experimental results show that our proposed approach consistently outperforms state-of-the-art methods on the popular HO3D and Dex-Ycb databases. Notably, the performance of our model on hand pose estimation even surpasses that of existing works that only perform the single-hand pose estimation task. Code is available at https://github.com/lzjJf12/HFL-Net.

        ----

        ## [1238] AssemblyHands: Towards Egocentric Activity Understanding via 3D Hand Pose Estimation

        **Authors**: *Takehiko Ohkawa, Kun He, Fadime Sener, Tomas Hodan, Luan Tran, Cem Keskin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01249](https://doi.org/10.1109/CVPR52729.2023.01249)

        **Abstract**:

        We present AssemblyHands, a large-scale benchmark dataset with accurate 3D hand pose annotations, to facilitate the study of egocentric activities with challenging hand-object interactions. The dataset includes synchronized egocentric and exocentric images sampled from the recent Assembly101 dataset, in which participants assemble and disassemble take-apart toys. To obtain high-quality 3D hand pose annotations for the egocentric images, we develop an efficient pipeline, where we use an initial set of manual annotations to train a model to automatically annotate a much larger dataset. Our annotation model uses multi-view feature fusion and an iterative refinement scheme, and achieves an average keypoint error of 4.20 mm, which is 85% lower than the error of the original annotations in Assembly101. AssemblyHands provides 3.0M annotated images, including 490K egocentric images, making it the largest existing benchmark dataset for egocentric 3D hand pose estimation. Using this data, we develop a strong single-view baseline of 3D hand pose estimation from egocentric images. Furthermore, we design a novel action classification task to evaluate predicted 3D hand poses. Our study shows that having higher-quality hand poses directly improves the ability to recognize actions.

        ----

        ## [1239] A Characteristic Function-Based Method for Bottom-Up Human Pose Estimation

        **Authors**: *Haoxuan Qu, Yujun Cai, Lin Geng Foo, Ajay Kumar, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01250](https://doi.org/10.1109/CVPR52729.2023.01250)

        **Abstract**:

        Most recent methods formulate the task of human pose estimation as a heatmap estimation problem, and use the overall L2 loss computed from the entire heatmap to optimize the heatmap prediction. In this paper, we show that in bottom-up human pose estimation where each heatmap often contains multiple body joints, using the overall L2 loss to optimize the heatmap prediction may not be the optimal choice. This is because, minimizing the overall L2 loss cannot always lead the model to locate all the body joints across different sub-regions of the heatmap more accurately. To cope with this problem, from a novel perspective, we propose a new bottom-up human pose estimation method that optimizes the heatmap prediction via minimizing the distance between two characteristic functions respectively constructed from the predicted heatmap and the groundtruth heatmap. Our analysis presented in this paper indicates that the distance between these two characteristic functions is essentially the upper bound of the L2 losses w.r.t. sub-regions of the predicted heatmap. Therefore, via minimizing the distance between the two characteristic functions, we can optimize the model to provide a more accurate localization result for the body joints in different sub-regions of the predicted heatmap. We show the effectiveness of our proposed method through extensive experiments on the COCO dataset and the CrowdPose dataset.

        ----

        ## [1240] Unified Pose Sequence Modeling

        **Authors**: *Lin Geng Foo, Tianjiao Li, Hossein Rahmani, Qiuhong Ke, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01251](https://doi.org/10.1109/CVPR52729.2023.01251)

        **Abstract**:

        We propose a Unified Pose Sequence Modeling approach to unify heterogeneous human behavior understanding tasks based on pose data, e.g., action recognition, 3D pose estimation and 3D early action prediction. A major obstacle is that different pose-based tasks require different output data formats. Specifically, the action recognition and prediction tasks require class predictions as outputs, while 3D pose estimation requires a human pose output, which limits existing methods to leverage task-specific network architectures for each task. Hence, in this paper, we propose a novel Unified Pose Sequence (UPS) model to unify heterogeneous output formats for the aforementioned tasks by considering text-based action labels and coordinate-based human poses as language sequences. Then, by optimizing a single auto-regressive transformer, we can obtain a unified output sequence that can handle all the aforementioned tasks. Moreover, to avoid the interference brought by the heterogeneity between different tasks, a dynamic routing mechanism is also proposed to empower our UPS with the ability to learn which subsets of parameters should be shared among different tasks. To evaluate the efficacy of the proposed UPS, extensive experiments are conducted on four different tasks with four popular behavior understanding benchmarks.

        ----

        ## [1241] Scene-Aware Egocentric 3D Human Pose Estimation

        **Authors**: *Jian Wang, Diogo C. Luvizon, Weipeng Xu, Lingjie Liu, Kripasindhu Sarkar, Christian Theobalt*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01252](https://doi.org/10.1109/CVPR52729.2023.01252)

        **Abstract**:

        Egocentric 3D human pose estimation with a single head-mounted fisheye camera has recently attracted attention due to its numerous applications in virtual and augmented reality. Existing methods still struggle in challenging poses where the human body is highly occluded or is closely interacting with the scene. To address this issue, we propose a scene-aware egocentric pose estimation method that guides the prediction of the egocentric pose with scene constraints. To this end, we propose an egocentric depth estimation network to predict the scene depth map from a wide-view egocentric fisheye camera while mitigating the occlusion of the human body with a depth-inpainting network. Next, we propose a scene-aware pose estimation network that projects the 2D image features and estimated depth map of the scene into a voxel space and regresses the 3D pose with a V2V network. The voxel-based feature representation provides the direct geometric connection between 2D image features and scene geometry, and further facilitates the V2V network to constrain the predicted pose based on the estimated scene geometry. To enable the training of the aforementioned networks, we also generated a synthetic dataset, called EgoGTA, and an in-the-wild dataset based on EgoPW, called EgoPW-Scene. The experimental results of our new evaluation sequences show that the predicted 3D egocentric poses are accurate and physically plausible in terms of human-scene interaction, demonstrating that our method outperforms the state-of-the-art methods both quantitatively and qualitatively.

        ----

        ## [1242] DiffPose: Toward More Reliable 3D Pose Estimation

        **Authors**: *Jia Gong, Lin Geng Foo, Zhipeng Fan, Qiuhong Ke, Hossein Rahmani, Jun Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01253](https://doi.org/10.1109/CVPR52729.2023.01253)

        **Abstract**:

        Monocular 3D human pose estimation is quite challenging due to the inherent ambiguity and occlusion, which often lead to high uncertainty and indeterminacy. On the other hand, diffusion models have recently emerged as an effective tool for generating high-quality images from noise. In-spired by their capability, we explore a novel pose estimation framework (DiffPose) that formulates 3D pose estimation as a reverse diffusion process. We incorporate novel designs into our DiffPose to facilitate the diffusion process for 3D pose estimation: a pose-specific initialization of pose uncertainty distributions, a Gaussian Mixture Model-based forward diffusion process, and a context-conditioned re-verse diffusion process. Our proposed DiffPose significantly outperforms existing methods on the widely used pose estimation benchmarks Human3.6M and MPI-INF-3DHP. Project page: https://gongjia0208.github.io/Diffpose/.

        ----

        ## [1243] MammalNet: A Large-Scale Video Benchmark for Mammal Recognition and Behavior Understanding

        **Authors**: *Jun Chen, Ming Hu, Darren J. Coker, Michael L. Berumen, Blair R. Costelloe, Sara Beery, Anna Rohrbach, Mohamed Elhoseiny*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01254](https://doi.org/10.1109/CVPR52729.2023.01254)

        **Abstract**:

        Monitoring animal behavior can facilitate conservation efforts by providing key insights into wildlife health, population status, and ecosystem function. Automatic recognition of animals and their behaviors is critical for capitalizing on the large unlabeled datasets generated by modern video devices and for accelerating monitoring efforts at scale. However, the development of automated recognition systems is currently hindered by a lack of appropriately labeled datasets. Existing video datasets 1) do not classify animals according to established biological taxonomies; 2) are too small to facilitate large-scale behavioral studies and are often limited to a single species; and 3) do not feature temporally localized annotations and therefore do not facilitate localization of targeted behaviors within longer video sequences. Thus, we propose MammalNet, a new large-scale animal behavior dataset with taxonomy-guided annotations of mammals and their common behaviors. MammalNet contains over 18K videos totaling 539 hours, which is ~10 times larger than the largest existing animal behavior dataset [36]. It covers 17 orders, 69 families, and 173 mammal categories for animal categorization and captures 12 high-level animal behaviors that received focus in previous animal behavior studies. We establish three benchmarks on MammalNet: standard animal and behavior recognition, compositional low-shot animal and behavior recognition, and behavior detection. Our dataset and code have been made available at: https://mammalnet.github.io.

        ----

        ## [1244] Learning 3D-Aware Image Synthesis with Unknown Pose Distribution

        **Authors**: *Zifan Shi, Yujun Shen, Yinghao Xu, Sida Peng, Yiyi Liao, Sheng Guo, Qifeng Chen, Dit-Yan Yeung*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01255](https://doi.org/10.1109/CVPR52729.2023.01255)

        **Abstract**:

        Existing methods for 3D-aware image synthesis largely depend on the 3D pose distribution pre-estimated on the training set. An inaccurate estimation may mislead the model into learning faulty geometry. This work proposes PoF3D that frees generative radiance fields from the requirements of 3D pose priors. We first equip the generator with an efficient pose learner, which is able to infer a pose from a latent code, to approximate the underlying true pose distribution automatically. We then assign the discriminator a task to learn pose distribution under the supervision of the generator and to differentiate real and synthesized images with the predicted pose as the condition. The pose-free generator and the pose-aware discriminator are jointly trained in an adversarial manner. Extensive results on a couple of datasets confirm that the performance of our approach, regarding both image quality and geometry quality, is on par with state of the art. To our best knowledge, PoF3D demonstrates the feasibility of learning high-quality 3D-aware image synthesis without using 3D pose priors for the first time. Project page can be found here.

        ----

        ## [1245] Pose Synchronization under Multiple Pair-wise Relative Poses

        **Authors**: *Yifan Sun, Qixing Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01256](https://doi.org/10.1109/CVPR52729.2023.01256)

        **Abstract**:

        Pose synchronization, which seeks to estimate consistent absolute poses among a collection of objects from noisy relative poses estimated between pairs of objects in isolation, is a fundamental problem in many inverse applications. This paper studies an extreme setting where multiple relative pose estimates exist between each object pair, and the majority is incorrect. Popular methods that solve pose synchronization via recovering a low-rank matrix that encodes relative poses in block fail under this extreme setting. We introduce a three-step algorithm for pose synchronization under multiple relative pose inputs. The first step performs diffusion and clustering to compute the candidate poses of the input objects. We present a theoretical result to justify our diffusion formulation. The second step jointly optimizes the best pose for each object. The final step refines the output of the second step. Experimental results on benchmark datasets of structure-from-motion and scan-based geometry reconstruction show that our approach offers more accurate absolute poses than state-of-the-art pose synchronization techniques.

        ----

        ## [1246] ObjectMatch: Robust Registration using Canonical Object Correspondences

        **Authors**: *Can Gümeli, Angela Dai, Matthias Nießner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01257](https://doi.org/10.1109/CVPR52729.2023.01257)

        **Abstract**:

        We present ObjectMatch
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://cangumeli.github.io/ObjectMatch/, a semantic and object-centric camera pose estimator for RGB-D SLAM pipelines. Modern camera pose estimators rely on direct correspondences of overlapping regions between frames; however, they cannot align camera frames with little or no overlap. In this work, we propose to leverage indirect correspondences obtained via semantic object identification. For instance, when an object is seen from the front in one frame and from the back in another frame, we can provide additional pose constraints through canonical object correspondences. We first propose a neural network to predict such correspondences on a per-pixel level, which we then combine in our energy formulation with state-of-the-art keypoint matching solved with a joint Gauss-Newton optimization. In a pairwise setting, our method improves registration recall of state-of-the-art feature matching, including from 24% to 45% in pairs with 10% or less inter-frame overlap. In registering RGB-D sequences, our method outperforms cutting-edge SLAM baselines in challenging, low-frame-rate scenarios, achieving more than 35% reduction in trajectory error in multiple scenes.

        ----

        ## [1247] Learning Articulated Shape with Keypoint Pseudo-Labels from Web Images

        **Authors**: *Anastasis Stathopoulos, Georgios Pavlakos, Ligong Han, Dimitris N. Metaxas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01258](https://doi.org/10.1109/CVPR52729.2023.01258)

        **Abstract**:

        This paper shows that it is possible to learn models for monocular 3D reconstruction of articulated objects (e.g. horses, cows, sheep), using as few as 50–150 images labeled with 2D keypoints. Our proposed approach involves training category-specific keypoint estimators, generating 2D keypoint pseudo-labels on unlabeled web images, and using both the labeled and self-labeled sets to train 3D reconstruction models. It is based on two key insights: (1) 2D keypoint estimation networks trained on as few as 50– 150 images of a given object category generalize well and generate reliable pseudo-labels; (2) a data selection mechanism can automatically create a “curated” subset of the unlabeled web images that can be used for training - we evaluate four data selection methods. Coupling these two insights enables us to train models that effectively utilize web images, resulting in improved 3D reconstruction performance for several articulated object categories beyond the fully-supervised baseline. Our approach can quickly bootstrap a model and requires only a few images labeled with 2D keypoints. This requirement can be easily satisfied for any new object category. To showcase the practicality of our approach for predicting the 3D shape of arbitrary object categories, we annotate 2D keypoints on 250 giraffe and bear images from COCO in just 2.5 hours per category.

        ----

        ## [1248] Learning Correspondence Uncertainty via Differentiable Nonlinear Least Squares

        **Authors**: *Dominik Muhle, Lukas Koestler, Krishna Murthy Jatavallabhula, Daniel Cremers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01259](https://doi.org/10.1109/CVPR52729.2023.01259)

        **Abstract**:

        We propose a differentiable nonlinear least squares framework to account for uncertainty in relative pose estimation from feature correspondences. Specifically, we introduce a symmetric version of the probabilistic normal epipolar constraint, and an approach to estimate the co-variance of feature positions by differentiating through the camera pose estimation procedure. We evaluate our approach on synthetic, as well as the KITTI and EuRoC real-world datasets. On the synthetic dataset, we confirm that our learned covariances accurately approximate the true noise distribution. In real world experiments, we find that our approach consistently outperforms state-of-the-art non-probabilistic and probabilistic approaches, regardless of the feature extraction algorithm of choice.

        ----

        ## [1249] Efficient Second-Order Plane Adjustment

        **Authors**: *Lipu Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01260](https://doi.org/10.1109/CVPR52729.2023.01260)

        **Abstract**:

        Planes are generally used in 3D reconstruction for depth sensors, such as RGB-D cameras and LiDARs. This paper focuses on the problem of estimating the optimal planes and sensor poses to minimize the point-to-plane distance. The resulting least-squares problem is referred to as plane adjustment (PA) in the literature, which is the counterpart of bundle adjustment (BA) in visual reconstruction. Iterative methods are adopted to solve these least-squares problems. Typically, Newton's method is rarely used for a large-scale least-squares problem, due to the high computational complexity of the Hessian matrix. Instead, methods using an approximation of the Hessian matrix, such as the Levenberg-Marquardt (LM) method, are generally adopted. This paper adopts the Newton's method to efficiently solve the PA problem. Specifically, given poses, the optimal plane have a close-form solution. Thus we can eliminate planes from the cost function, which significantly reduces the number of variables. Furthermore, as the optimal planes are functions of poses, this method actually ensures that the optimal planes for the current estimated poses can be obtained at each iteration, which benefits the convergence. The difficulty lies in how to efficiently compute the Hessian matrix and the gradient of the resulting cost. This paper provides an efficient solution. Empirical evaluation shows that our algorithm outperforms the state-of-the-art algorithms.

        ----

        ## [1250] Learning a Depth Covariance Function

        **Authors**: *Eric Dexheimer, Andrew J. Davison*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01261](https://doi.org/10.1109/CVPR52729.2023.01261)

        **Abstract**:

        We propose learning a depth covariance function with applications to geometric vision tasks. Given RGB images as input, the covariance function can be flexibly used to define priors over depth functions, predictive distributions given observations, and methods for active point selection. We leverage these techniques for a selection of downstream tasks: depth completion, bundle adjustment, and monocular dense visual odometry.

        ----

        ## [1251] Privacy-Preserving Representations are not Enough: Recovering Scene Content from Camera Poses

        **Authors**: *Kunal Chelani, Torsten Sattler, Fredrik Kahl, Zuzana Kukelova*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01262](https://doi.org/10.1109/CVPR52729.2023.01262)

        **Abstract**:

        Visual localization is the task of estimating the camera pose from which a given image was taken and is central to several 3D computer vision applications. With the rapid growth in the popularity of AR/VR/MR devices and cloudbased applications, privacy issues are becoming a very important aspect of the localization process. Existing work on privacy-preserving localization aims to defend against an attacker who has access to a cloud-based service. In this paper, we show that an attacker can learn about details of a scene without any access by simply querying a localization service. The attack is based on the observation that modern visual localization algorithms are robust to variations in appearance and geometry. While this is in general a desired property, it also leads to algorithms localizing objects that are similar enough to those present in a scene. An attacker can thus query a server with a large enough set of images of objects, e.g., obtained from the Internet, and some of them will be localized. The attacker can thus learn about object placements from the camera poses returned by the service (which is the minimal information returned by such a service). In this paper, we develop a proof-of-concept version of this attack and demonstrate its practical feasibility. The attack does not place any requirements on the localization algorithm used, and thus also applies to privacy-preserving representations. Current work on privacy-preserving representations alone is thus insufficient.

        ----

        ## [1252] Objaverse: A Universe of Annotated 3D Objects

        **Authors**: *Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, Ali Farhadi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01263](https://doi.org/10.1109/CVPR52729.2023.01263)

        **Abstract**:

        Massive data corpora like WebText, Wikipedia, Conceptual Captions, WebImageText, and LAION have propelled recent dramatic progress in AI. Large neural models trained on such datasets produce impressive results and top many of today's benchmarks. A notable omisslion within this family of large-scale datasets is 3D data. Despite considerable interest and potential applications in 3D vision, datasets of high-fidelity 3D models continue to be mid-sized with limited diversity of object categories. Addressing this gap, we present Objaverse 1.0, a large dataset of objects with 800K + (and growing) 3D models with descriptive captions, tags, and animations. Objaverse improves upon present day 3D repositories in terms of scale, number of categories, and in the visual diversity of instances within a category. We demonstrate the large potential of Objaverse via four diverse applications: training generative 3D models, improving tail category segmentation on the LVIS benchmark, training open-vocabulary object-navigation models for Embodied AI, and creating a new benchmark for robustness analysis of vision models. Objaverse can open new directions for research and enable new applications across the field of AI.

        ----

        ## [1253] Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild

        **Authors**: *Garrick Brazil, Abhinav Kumar, Julian Straub, Nikhila Ravi, Justin Johnson, Georgia Gkioxari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01264](https://doi.org/10.1109/CVPR52729.2023.01264)

        **Abstract**:

        Recognizing scenes and objects in 3D from a single image is a longstanding goal of computer vision with applications in robotics and AR/VR. For 2D recognition, large datasets and scalable solutions have led to unprecedented advances. In 3D, existing benchmarks are small in size and approaches specialize in few object categories and specific domains, e.g. urban driving scenes. Motivated by the success of 2D recognition, we revisit the task of 3D object detection by introducing a large benchmark, called Omni3D. Omni3DRE-purposes and combines existing datasets resulting in 234k images annotated with more than 3 million instances and 98 categories. 3D detection at such scale is challenging due to variations in camera intrinsics and the rich diversity of scene and object types. We propose a model, called Cube R-CNN, designed to generalize across camera and scene types with a unified approach. We show that Cube R-CNN outperforms prior works on the larger Omni3D and existing benchmarks. Finally, we prove that Omni3D is a powerful dataset for 3D object recognition and show that it improves single-dataset performance and can accelerate learning on new smaller datasets via pre-training.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
We release the Omni3D benchmark and Cube R-CNN models at https://github.com/facebookresearch/omni3d.

        ----

        ## [1254] HelixSurf: A Robust and Efficient Neural Implicit Surface Learning of Indoor Scenes with Iterative Intertwined Regularization

        **Authors**: *Zhihao Liang, Zhangjin Huang, Changxing Ding, Kui Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01265](https://doi.org/10.1109/CVPR52729.2023.01265)

        **Abstract**:

        Recovery of an underlying scene geometry from multi-view images stands as a long-time challenge in computer vision research. The recent promise leverages neural implicit surface learning and differentiable volume rendering, and achieves both the recovery of scene geometry and synthesis of novel views, where deep priors of neural models are used as an inductive smoothness bias. While promising for object-level surfaces, these methods suffer when coping with complex scene surfaces. In the meanwhile, traditional multi-view stereo can recover the geometry of scenes with rich textures, by globally optimizing the local, pixel-wise correspondences across multiple views. We are thus motivated to make use of the complementary benefits from the two strategies, and propose a method termed Helix-shaped neural implicit Surface learning or HelixSurf; HelixSurf uses the intermediate prediction from one strategy as the guidance to regularize the learning of the other one, and conducts such intertwined regularization iteratively during the learning process. We also propose an efficient scheme for differentiable volume rendering in HelixSurf Experiments on surface reconstruction of indoor scenes show that our method compares favorably with existing methods and is orders of magnitude faster, even when some of existing methods are assisted with auxiliary training data. The source code is available at hups://guhub.com/Gorilla-Lab-SCUT/HelixSurf.

        ----

        ## [1255] Visual Localization using Imperfect 3D Models from the Internet

        **Authors**: *Vojtech Panek, Zuzana Kukelova, Torsten Sattler*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01266](https://doi.org/10.1109/CVPR52729.2023.01266)

        **Abstract**:

        Visual localization is a core component in many applications, including augmented reality (AR). Localization algorithms compute the camera pose of a query image w.r.t. a scene representation, which is typically built from images. This often requires capturing and storing large amounts of data, followed by running Structure-from-Motion (SfM) algorithms. An interesting, and underexplored, source of data for building scene representations are 3D models that are readily available on the Internet, e.g., hand-drawn CAD models, 3D models generated from building footprints, or from aerial images. These models allow to perform visual localization right away without the time-consuming scene capturing and model building steps. Yet, it also comes with challenges as the available 3D models are often imperfect reflections of reality. E.g., the models might only have generic or no textures at all, might only provide a simple approximation of the scene geometry, or might be stretched. This paper studies how the imperfections of these models affect localization accuracy. We create a new benchmark for this task and provide a detailed experimental evaluation based on multiple 3D models per scene. We show that 3D models from the Internet show promise as an easy-to-obtain scene representation. At the same time, there is significant room for improvement for visual localization pipelines. To foster research on this interesting and challenging task, we release our benchmark at v-pnk.github.io/cadloc.

        ----

        ## [1256] PRISE: Demystifying Deep Lucas-Kanade with Strongly Star-Convex Constraints for Multimodel Image Alignment

        **Authors**: *Yiqing Zhang, Xinming Huang, Ziming Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01267](https://doi.org/10.1109/CVPR52729.2023.01267)

        **Abstract**:

        The Lucas-Kanade (LK) method is a classic iterative homography estimation algorithm for image alignment, but often suffers from poor local optimality especially when image pairs have large distortions. To address this challenge, in this paper we propose a novel Deep Star-Convexified Lucas-Kanade (PRISE) method for multimodel image alignment by introducing strongly star-convex constraints into the optimization problem. Our basic idea is to enforce the neural network to approximately learn a star-convex loss landscape around the ground truth give any data to facilitate the convergence of the LK method to the ground truth through the high dimensional space defined by the network. This leads to a minimax learning problem, with contrastive (hinge) losses due to the definition of strong star-convexity that are appended to the original loss for training. We also provide an efficient sampling based algorithm to leverage the training cost, as well as some analysis on the quality of the solutions from PRISE. We further evaluate our approach on benchmark datasets such as MSCOCO, GoogleEarth, and GoogleMap, and demonstrate state-of-the-art results, especially for small pixel errors. Code can be downloaded from https://github.com/Zhang-VISLab.

        ----

        ## [1257] Scalable, Detailed and Mask-Free Universal Photometric Stereo

        **Authors**: *Satoshi Ikehata*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01268](https://doi.org/10.1109/CVPR52729.2023.01268)

        **Abstract**:

        In this paper, we introduce SDM-UniPS, a groundbreaking Scalable, Detailed, Mask-free, and Universal Photometric Stereo network. Our approach can recover astonishingly intricate surface normal maps, rivaling the quality of 3D scanners, even when images are captured under unknown, spatially-varying lighting conditions in uncontrolled environments. We have extended previous universal photometric stereo networks to extract spatial-light features, utilizing all available information in high-resolution input images and accounting for non-local interactions among surface points. Moreover, we present a new synthetic training dataset that encompasses a diverse range of shapes, materials, and illumination scenarios found in real-world scenes. Through extensive evaluation, we demonstrate that our method not only surpasses calibrated, lighting-specific techniques on public benchmarks, but also excels with a significantly smaller number of input images even without object masks.

        ----

        ## [1258] Enhanced Stable View Synthesis

        **Authors**: *Nishant Jain, Suryansh Kumar, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01269](https://doi.org/10.1109/CVPR52729.2023.01269)

        **Abstract**:

        We introduce an approach to enhance the novel view synthesis from images taken from a freely moving camera. The introduced approach focuses on outdoor scenes where recovering accurate geometric scaffold and camera pose is challenging, leading to inferior results using the state-of-the-art stable view synthesis (SVS) method. SVS and related methods fail for outdoor scenes primarily due to (i) overrelying on the multiview stereo (MVS) for geometric scaffold recovery and (ii) assuming COLMAP computed camera poses as the best possible estimates, despite it being well-studied that MVS 3D reconstruction accuracy is limited to scene disparity and camera-pose accuracy is sensitive to key-point correspondence selection. This work proposes a principled way to enhance novel view synthesis solutions drawing inspiration from the basics of multiple view geometry. By leveraging the complementary behavior of MVS and monocular depth, we arrive at a better scene depth per view for nearby and far points, respectively. Moreover, our approach jointly refines camera poses with image-based rendering via multiple rotation averaging graph optimization. The recovered scene depth and the camera-pose help better view-dependent on-surface feature aggregation of the entire scene. Extensive evaluation of our approach on the popular benchmark dataset, such as Tanks and Temples, shows substantial improvement in view synthesis results compared to the prior art. For instance, our method shows 1.5 dB of PSNR improvement on the Tank and Temples. Similar statistics are observed when tested on other benchmark datasets such as FVS, Mip-NeRF 360, and DTU.

        ----

        ## [1259] End-to-End Vectorized HD-map Construction with Piecewise Bézier Curve

        **Authors**: *Limeng Qiao, Wenjie Ding, Xi Qiu, Chi Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01270](https://doi.org/10.1109/CVPR52729.2023.01270)

        **Abstract**:

        Vectorized high-definition map (HD-map) construction, which focuses on the perception of centimeter-level environmental information, has attracted significant research inter-est in the autonomous driving community. Most existing approaches first obtain rasterized map with the segmentation-based pipeline and then conduct heavy post-processing for downstream-friendly vectorization. In this paper, by delving into parameterization-based methods, we pioneer a concise and elegant scheme that adopts unified piecewise Bézier curve. In order to vectorize changeful map elements end-to-end, we elaborate a simple yet effective architecture, named Piecewise Bézier HD-map Network (BeMapNet), which is formulated as a direct set prediction paradigm and postprocessing-free. Concretely, we first introduce a novel IPM-PE Align module to inject 3D geometry prior into BEV features through common position encoding in Transformer. Then a well-designed Piecewise Bézier Head is proposed to output the details of each map element, including the coordinate of control points and the segment number of curves. In addition, based on the progressively restoration of Bézier curve, we also present an efficient Point-Curve-Region Loss for supervising more robust and precise HD-map modeling. Extensive comparisons show that our method is remarkably superior to other existing SOTAs by 18.0 mAP at least
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/er-muyue/BeMapNet.

        ----

        ## [1260] DynamicStereo: Consistent Dynamic Depth from Stereo Videos

        **Authors**: *Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, Christian Rupprecht*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01271](https://doi.org/10.1109/CVPR52729.2023.01271)

        **Abstract**:

        We consider the problem of reconstructing a dynamic scene observed from a stereo camera. Most existing meth-ods for depth from stereo treat different stereo frames in-dependently, leading to temporally inconsistent depth pre-dictions. Temporal consistency is especially important for immersive AR or VR scenarios, where flickering greatly di-minishes the user experience. We propose DynamicStereo, a novel transformer-based architecture to estimate dispar-ity for stereo videos. The network learns to pool information from neighboring frames to improve the temporal consistency of its predictions. Our architecture is designed to process stereo videos efficiently through divided attention layers. We also introduce Dynamic Replica, a new bench-mark dataset containing synthetic videos of people and ani-mals in scanned environments, which provides complemen-tary training and evaluation data for dynamic stereo closer to real applications than existing datasets. Training with this dataset further improves the quality of predictions of our proposed DynamicStereo as well as prior methods. Finally, it acts as a benchmark for consistent stereo methods. Project page: https://dynamic-stereo.github.io/

        ----

        ## [1261] Shakes on a Plane: Unsupervised Depth Estimation from Unstabilized Photography

        **Authors**: *Ilya Chugunov, Yuxuan Zhang, Felix Heide*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01272](https://doi.org/10.1109/CVPR52729.2023.01272)

        **Abstract**:

        Modern mobile burst photography pipelines capture and merge a short sequence of frames to recover an enhanced image, but often disregard the 3D nature of the scene they capture, treating pixel motion between images as a 2D aggregation problem. We show that in a “long-burst”, forty-two 12-megapixel RAW frames captured in a two-second sequence, there is enough parallax information from natural hand tremor alone to recover high-quality scene depth. To this end, we devise a test-time optimization approach that fits a neural RGB-D representation to long-burst data and simultaneously estimates scene depth and camera motion. Our plane plus depth model is trained end-to-end, and performs coarse-to-fine refinement by controlling which multi-resolution volume features the network has access to at what time during training. We validate the method experimentally, and demonstrate geometrically accurate depth reconstructions with no additional hardware or separate data pre-processing and pose-estimation steps.

        ----

        ## [1262] Gated Stereo: Joint Depth Estimation from Gated and Wide-Baseline Active Stereo Cues

        **Authors**: *Stefanie Walz, Mario Bijelic, Andrea Ramazzina, Amanpreet Walia, Fahim Mannan, Felix Heide*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01273](https://doi.org/10.1109/CVPR52729.2023.01273)

        **Abstract**:

        We propose Gated Stereo, a high-resolution and long-range depth estimation technique that operates on active gated stereo images. Using active and high dynamic range passive captures, Gated Stereo exploits multi-view cues alongside time-of-flight intensity cues from active gating. To this end, we propose a depth estimation method with a monocular and stereo depth prediction branch which are combined in a final fusion stage. Each block is supervised through a combination of supervised and gated self-supervision losses. To facilitate training and validation, we acquire a long-range synchronized gated stereo dataset for automotive scenarios. We find that the method achieves an improvement of more than 50 % MAE compared to the next best RGB stereo method, and 74 % MAE to existing monocular gated methods for distances up to 160 m. Our code, models and datasets are available here
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://light.princeton.edu/gatedstereo/.

        ----

        ## [1263] K3DN: Disparity-Aware Kernel Estimation for Dual-Pixel Defocus Deblurring

        **Authors**: *Yan Yang, Liyuan Pan, Liu Liu, Miaomiao Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01274](https://doi.org/10.1109/CVPR52729.2023.01274)

        **Abstract**:

        The dual-pixel (DP) sensor captures a two-view image pair in a single snapshot by splitting each pixel in half. The disparity occurs in defocus blurred regions between the two views of the DP pair, while the in-focus sharp regions have zero disparity. This motivates us to propose a K3DN framework for DP pair deblurring, and it has three modules: i) a disparity-aware deblur module. It estimates a disparity feature map, which is used to query a trainable kernel set to estimate a blur kernel that best describes the spatially-varying blur. The kernel is constrained to be symmetrical per the DP formulation. A simple Fourier transform is performed for deblurring that follows the blur model; ii) a reblurring regularization module. It reuses the blur kernel, performs a simple convolution for reblurring, and regularizes the estimated kernel and disparity feature unsupervisedly, in the training stage; iii) a sharp region preservation module. It identifies in-focus regions that correspond to areas with zero disparity between DP images, aims to avoid the introduction of noises during the deblurring process, and improves image restoration performance. Experiments on four standard DP datasets show that the proposed K3DN outperforms state-of-the-art methods, with fewer parameters and flops at the same time.

        ----

        ## [1264] HRDFuse: Monocular 360° Depth Estimation by Collaboratively Learning Holistic-with-Regional Depth Distributions

        **Authors**: *Hao Ai, Zidong Cao, Yan-Pei Cao, Ying Shan, Lin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01275](https://doi.org/10.1109/CVPR52729.2023.01275)

        **Abstract**:

        Depth estimation from a monocular 360° image is a burgeoning problem owing to its holistic sensing of a scene. Recently, some methods, e.g., OmniFusion, have applied the tangent projection (TP) to represent a 360° image and predicted depth values via patch-wise regressions, which are merged to get a depth map with equirectangular projection (ERP) format. However, these methods suffer from 1) non-trivial process of merging plenty of patches; 2) capturing less holistic-with-regional contextual information by directly regressing the depth value of each pixel. In this paper, we propose a novel framework, HRDFuse, that subtly combines the potential of convolutional neural networks (CNNs) and transformers by collaboratively learning the holistic contextual information from the ERP and the regional structural information from the TP. Firstly, we propose a spatial feature alignment (SFA) module that learns feature similarities between the TP and ERP to aggregate the TP features into a complete ERP feature map in a pixelwise manner. Secondly, we propose a collaborative depth distribution classification (CDDC) module that learns the holistic-with-regional histograms capturing the ERP and TP depth distributions. As such, the final depth values can be predicted as a linear combination of histogram bin centers. Lastly, we adaptively combine the depth predictions from ERP and TP to obtain the final depth map. Extensive experiments show that our method predicts more smooth and accurate depth results while achieving favorably better results than the SOTA methods.

        ----

        ## [1265] OSRT: Omnidirectional Image Super-Resolution with Distortion-aware Transformer

        **Authors**: *Fanghua Yu, Xintao Wang, Mingdeng Cao, Gen Li, Ying Shan, Chao Dong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01276](https://doi.org/10.1109/CVPR52729.2023.01276)

        **Abstract**:

        Omnidirectional images (ODIs) have obtained lots of research interest for immersive experiences. Although ODIs require extremely high resolution to capture details of the entire scene, the resolutions of most ODIs are insufficient. Previous methods attempt to solve this issue by image super-resolution (SR) on equirectangular projection (ERP) images. However, they omit geometric properties of ERP in the degradation process, and their models can hardly generalize to real ERP images. In this paper, we propose Fisheye downsampling, which mimics the real-world imaging process and synthesizes more realistic low-resolution samples. Then we design a distortion-aware Transformer (OSRT) to modulate ERP distortions continuously and self-adaptively. Without a cumbersome process, OSRT outperforms previous methods by about 0.2dB on PSNR. Moreover, we propose a convenient data augmentation strategy, which synthesizes pseudo ERP images from plain images. This simple strategy can alleviate the over-fitting problem of large networks and significantly boost the performance of ODISR. Extensive experiments have demonstrated the state-of-the-art performance of our OSRT.

        ----

        ## [1266] Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM

        **Authors**: *Hengyi Wang, Jingwen Wang, Lourdes Agapito*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01277](https://doi.org/10.1109/CVPR52729.2023.01277)

        **Abstract**:

        We present Co-SLAM, a neural RGB-D SLAM system based on a hybrid representation, that performs robust camera tracking and high-fidelity surface reconstruction in real time. Co-SLAM represents the scene as a multi-resolution hash-grid to exploit its high convergence speed and ability to represent high-frequency local features. In addition, Co-SLAM incorporates one-blob encoding, to encourage surface coherence and completion in unobserved areas. This joint parametric-coordinate encoding enables real-time and robust performance by bringing the best of both worlds: fast convergence and surface hole filling. Moreover, our ray sampling strategy allows Co-SLAM to perform global bundle adjustment over all keyframes instead of requiring keyframe selection to maintain a small number of active keyframes as competing neural SLAM approaches do. Experimental results show that Co-SLAM runs at 10-17Hz and achieves state-of-the-art scene reconstruction results, and competitive tracking performance in various datasets and benchmarks (ScanNet, TUM, Replica, Synthetic RGBD). Project page: https://hengyiwang.github.io/projects/CoSLAM

        ----

        ## [1267] Few-Shot Non-Line-of-Sight Imaging with Signal-Surface Collaborative Regularization

        **Authors**: *Xintong Liu, Jianyu Wang, Leping Xiao, Xing Fu, Lingyun Qiu, Zuoqiang Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01278](https://doi.org/10.1109/CVPR52729.2023.01278)

        **Abstract**:

        The non-line-of-sight imaging technique aims to reconstruct targets from multiply reflected light. For most existing methods, dense points on the relay surface are raster scanned to obtain high-quality reconstructions, which requires a long acquisition time. In this work, we propose a signal-surface collaborative regularization (SSCR) framework that provides noise-robust reconstructions with a minimal number of measurements. Using Bayesian inference, we design joint regularizations of the estimated signal, the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$3D$</tex>
 voxel-based representation of the objects, and the 2D surface-based description of the targets. To our best knowledge, this is the first work that combines regularizations in mixed dimensions for hidden targets. Experiments on synthetic and experimental datasets illustrated the efficiency of the proposed method under both confocal and non-confocal settings. We report the reconstruction of the hidden targets with complex geometric structures with only 5 × 5 confocal measurements from public datasets, indicating an acceleration of the conventional measurement process by a factor of 10,000. Besides, the proposed method enjoys low time and memory complexity with sparse measurements. Our approach has great potential in real-time non-line-of-sight imaging applications such as rescue operations and autonomous driving.

        ----

        ## [1268] NLOST: Non-Line-of-Sight Imaging with Transformer

        **Authors**: *Yue Li, Jiayong Peng, Juntian Ye, Yueyi Zhang, Feihu Xu, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01279](https://doi.org/10.1109/CVPR52729.2023.01279)

        **Abstract**:

        Time-resolved non-line-of-sight (NLOS) imaging is based on the multi-bounce indirect reflections from the hidden objects for 3D sensing. Reconstruction from NLOS measurements remains challenging especially for complicated scenes. To boost the performance, we present NLOST, the first transformer-based neural network for NLOS reconstruction. Specifically, after extracting the shallow features with the assistance of physics-based priors, we design two spatial-temporal self attention encoders to explore both local and global correlations within 3D NLOS data by splitting or downsampling the features into different scales, respectively. Then, we design a spatial-temporal cross attention decoder to integrate local and global features in the token space of transformer, resulting in deep features with high representation capabilities. Finally, deep and shallow features are fused to reconstruct the 3D volume of hidden scenes. Extensive experimental results demonstrate the superior performance of the proposed method over existing solutions on both synthetic data and real-world data captured by different NLOS imaging systems.

        ----

        ## [1269] Listening Human Behavior: 3D Human Pose Estimation with Acoustic Signals

        **Authors**: *Yuto Shibata, Yutaka Kawashima, Mariko Isogawa, Go Irie, Akisato Kimura, Yoshimitsu Aoki*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01280](https://doi.org/10.1109/CVPR52729.2023.01280)

        **Abstract**:

        Given only acoustic signals without any high-level information, such as voices or sounds of scenes/actions, how much can we infer about the behavior of humans? Unlike existing methods, which suffer from privacy issues because they use signals that include human speech or the sounds of specific actions, we explore how low-level acoustic signals can provide enough clues to estimate 3D human poses by active acoustic sensing with a single pair of microphones and loudspeakers (see Fig. 1). This is a challenging task since sound is much more diffractive than other signals and therefore covers up the shape of objects in a scene. Accordingly, we introduce a framework that encodes multichannel audio features into 3D human poses. Aiming to capture subtle sound changes to reveal detailed pose information, we explicitly extract phase features from the acoustic signals together with typical spectrum features and feed them into our human pose estimation network. Also, we show that reflected or diffracted sounds are easily influenced by subjects' physique differences e.g., height and muscularity, which deteriorates prediction accuracy. We reduce these gaps by using a subject discriminator to improve accuracy. Our experiments suggest that with the use of only low-dimensional acoustic information, our method outperforms baseline methods. The datasets and codes used in this project will be publicly available.

        ----

        ## [1270] Towards Domain Generalization for Multi-view 3D Object Detection in Bird-Eye-View

        **Authors**: *Shuo Wang, Xinhai Zhao, Hai-Ming Xu, Zehui Chen, Dameng Yu, Jiahao Chang, Zhen Yang, Feng Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01281](https://doi.org/10.1109/CVPR52729.2023.01281)

        **Abstract**:

        Multi-view 3D object detection (MV3D-Det) in Bird-Eye-View (BEV) has drawn extensive attention due to its low cost and high efficiency. Although new algorithms for camera-only 3D object detection have been continuously proposed, most of them may risk drastic performance degradation when the domain of input images differs from that of training. In this paper, we first analyze the causes of the domain gap for the MV3D-Det task. Based on the covariate shift assumption, we find that the gap mainly attributes to the feature distribution of BEV, which is determined by the quality of both depth estimation and 2D image's feature representation. To acquire a robust depth prediction, we propose to decouple the depth estimation from the intrinsic parameters of the camera (i.e. the focal length) through converting the prediction of metric depth to that of scale-invariant depth and perform dynamic perspective augmentation to increase the diversity of the extrinsic parameters (i.e. the camera poses) by utilizing homography. Moreover, we modify the focal length values to create multiple pseudo-domains and construct an adversarial training loss to encourage the feature representation to be more domain-agnostic. Without bells and whistles, our approach, namely DG-BEV, successfully alleviates the performance drop on the unseen target domain without impairing the accuracy of the source domain. Extensive experiments on Waymo, nuScenes, and Lyft, demonstrate the generalization and effectiveness of our approach.

        ----

        ## [1271] X3KD: Knowledge Distillation Across Modalities, Tasks and Stages for Multi-Camera 3D Object Detection

        **Authors**: *Marvin Klingner, Shubhankar Borse, Varun Ravi Kumar, Behnaz Rezaei, Venkatraman Narayanan, Senthil Kumar Yogamani, Fatih Porikli*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01282](https://doi.org/10.1109/CVPR52729.2023.01282)

        **Abstract**:

        Recent advances in 3D object detection (3DOD) have obtained remarkably strong results for LiDAR-based models. In contrast, surround-view 3DOD models based on multiple camera images underperform due to the necessary view transformation of features from perspective view (PV) to a 3D world representation which is ambiguous due to missing depth information. This paper introduces X
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
KD, a comprehensive knowledge distillation framework across different modalities, tasks, and stages for multi-camera 3DOD. Specifically, we propose cross-task distillation from an instance segmentation teacher (X-IS) in the PV feature extraction stage providing supervision without ambiguous error backpropagation through the view transformation. After the transformation, we apply cross-modal feature distillation (X-FD) and adversarial training (X-AT) to improve the 3D world representation of multi-camera features through the information contained in a LiDAR-based 3DOD teacher. Finally, we also employ this teacher for cross-modal output distillation (X-OD), providing dense supervision at the prediction stage. We perform extensive ablations of knowledge distillation at different stages of multi-camera 3DOD. Our final X
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">3</sup>
KD model outperforms previous state-of-the-art approaches on the nuScenes and Waymo datasets and generalizes to RADAR-based 3DOD. Qualitative results video at https://youtu.be/1do9DPFmr38.

        ----

        ## [1272] Phase-Shifting Coder: Predicting Accurate Orientation in Oriented Object Detection

        **Authors**: *Yi Yu, Feipeng Da*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01283](https://doi.org/10.1109/CVPR52729.2023.01283)

        **Abstract**:

        With the vigorous development of computer vision, oriented object detection has gradually been featured. In this paper, a novel differentiable angle coder named phase-shifting coder (PSC) is proposed to accurately predict the orientation of objects, along with a dual-frequency version (PSCD). By mapping the rotational periodicity of different cycles into the phase of different frequencies, we provide a unified framework for various periodic fuzzy problems caused by rotational symmetry in oriented object detection. Upon such a framework, common problems in oriented object detection such as boundary discontinuity and square-like problems are elegantly solved in a unified form. Visual analysis and experiments on three datasets prove the effectiveness and the potentiality of our approach. When facing scenarios requiring high-quality bounding boxes, the proposed methods are expected to give a competitive performance. The codes are publicly available at https://github.com/open-mmlab/mmrotate.

        ----

        ## [1273] Learned Two-Plane Perspective Prior based Image Resampling for Efficient Object Detection

        **Authors**: *Anurag Ghosh, N. Dinesh Reddy, Christoph Mertz, Srinivasa G. Narasimhan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01284](https://doi.org/10.1109/CVPR52729.2023.01284)

        **Abstract**:

        Real-time efficient perception is critical for autonomous navigation and city scale sensing. Orthogonal to architectural improvements, streaming perception approaches have exploited adaptive sampling improving real-time detection performance. In this work, we propose a learnable geometry-guided prior that incorporates rough geometry of the 3D scene (a ground plane and a plane above) to resample images for efficient object detection. This significantly improves small and far-away object detection performance while also being more efficient both in terms of latency and memory. For autonomous navigation, using the same detector and scale, our approach improves detection rate by +4.1 AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">s</inf>
 or +39% and in real-time performance by +5.3 sAPs or +63% for small objects over state-of-the-art (SOTA). For fixed traffic cameras, our approach detects small objects at image scales other methods cannot. At the same scale, our approach improves detection of small objects by 195% (+12.5 AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">S</inf>
) over naive-downsampling and 63% (+4.2 AP
<inf xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">S</inf>
) over SOTA.

        ----

        ## [1274] Resource-Efficient RGBD Aerial Tracking

        **Authors**: *Jinyu Yang, Shang Gao, Zhe Li, Feng Zheng, Ales Leonardis*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01285](https://doi.org/10.1109/CVPR52729.2023.01285)

        **Abstract**:

        Aerial robots are now able to fly in complex environments, and drone-captured data gains lots of attention in object tracking. However, current research on aerial perception has mainly focused on limited categories, such as pedestrian or vehicle, and most scenes are captured in urban environments from a birds-eye view. Recently, UAVs equipped with depth cameras have been also deployed for more complex applications, while RGBD aerial tracking is still unexplored. Compared with traditional RGB object tracking, adding depth information can more effectively deal with more challenging scenes such as target and background interference. To this end, in this paper, we explore RGBD aerial tracking in an overhead space, which can greatly enlarge the development of drone-based visual perception. To boost the research, we first propose a large-scale benchmark for RGBD aerial tracking, containing 1,000 drone-captured RGBD videos with dense annotations. Then, as drone-based applications require for real-time processing with limited computational resources, we also propose an efficient RGBD tracker named EMT. Our tracker runs at over 100 fps on GPU, and 25 fps on the edge platform of NVidia Jetson NX Xavier, benefiting from its efficient multimodal fusion and feature matching. Extensive experiments show that our EMT achieves promising tracking performance. All resources are available at https://github.com/yjybuaa/RGBDAerialTracking.

        ----

        ## [1275] Toward RAW Object Detection: A New Benchmark and A New Model

        **Authors**: *Ruikang Xu, Chang Chen, Jingyang Peng, Cheng Li, Yibin Huang, Fenglong Song, Youliang Yan, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01286](https://doi.org/10.1109/CVPR52729.2023.01286)

        **Abstract**:

        In many computer vision applications (e.g., robotics and autonomous driving), high dynamic range (HDR) data is necessary for object detection algorithms to handle a variety of lighting conditions, such as strong glare. In this paper, we aim to achieve object detection on RAW sensor data, which naturally saves the HDR information from image sensors without extra equipment costs. We build a novel RAW sensor dataset, named ROD, for Deep Neural Networks (DNNs)-based object detection algorithms to be applied to HDR data. The ROD dataset contains a large amount of annotated instances of day and night driving scenes in 24-bit dynamic range. Based on the dataset, we first investigate the impact of dynamic range for DNNs-based detectors and demonstrate the importance of dynamic range adjustment for detection on RAW sensor data. Then, we propose a simple and effective adjustment method for object detection on HDR RAW sensor data, which is image adaptive and jointly optimized with the downstream detector in an end-to-end scheme. Extensive experiments demonstrate that the performance of detection on RAW sensor data is significantly superior to standard dynamic range (SDR) data in different situations. Moreover, we analyze the influence of texture information and pixel distribution of input data on the performance of the DNNs-based detector. Code and dataset will be available at https://gitee.com//mindspore/models/tree/master/research/cv/RAOD.

        ----

        ## [1276] Bi-LRFusion: Bi-Directional LiDAR-Radar Fusion for 3D Dynamic Object Detection

        **Authors**: *Yingjie Wang, Jiajun Deng, Yao Li, Jinshui Hu, Cong Liu, Yu Zhang, Jianmin Ji, Wanli Ouyang, Yanyong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01287](https://doi.org/10.1109/CVPR52729.2023.01287)

        **Abstract**:

        LiDAR and Radar are two complementary sensing approaches in that LiDAR specializes in capturing an object's 3D shape while Radar provides longer detection ranges as well as velocity hints. Though seemingly natural, how to efficiently combine them for improved feature representation is still unclear. The main challenge arises from that Radar data are extremely sparse and lack height information. Therefore, directly integrating Radar features into LiDAR-centric detection networks is not optimal. In this work, we introduce a bi-directional LiDAR-Radar fusion framework, termed Bi-LRFusion, to tackle the challenges and improve 3D detection for dynamic objects. Technically, Bi-LRFusion involves two steps: first, it enriches Radar's local features by learning important details from the LiDAR branch to alleviate the problems caused by the absence of height information and extreme sparsity; second, it combines LiDAR features with the enhanced Radar features in a unified bird's-eye-view representation. We conduct extensive experiments on nuScenes and ORR datasets, and show that our Bi-LRFusion achieves state-of-the-art performance for detecting dynamic objects. Notably, Radar data in these two datasets have different formats, which demonstrates the generalizability of our method. Codes will be published.

        ----

        ## [1277] LiDAR-in-the-Loop Hyperparameter Optimization

        **Authors**: *Félix Goudreault, Dominik Scheuble, Mario Bijelic, Nicolas Robidoux, Felix Heide*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01288](https://doi.org/10.1109/CVPR52729.2023.01288)

        **Abstract**:

        LiDAR has become a cornerstone sensing modality for 3D vision. LiDAR systems emit pulses of light into the scene, take measurements of the returned signal, and rely on hardware digital signal processing (DSP) pipelines to construct 3D point clouds from these measurements. The resulting point clouds output by these DSPs are input to downstream 3D vision models - both, in the form of training datasets or as input at inference time. Existing LiDAR DSPs are composed of cascades of parameterized operations; modifying configuration parameters results in significant changes in the point clouds and consequently the output of downstream methods. Existing methods treat LiDAR systems as fixed black boxes and construct downstream task networks more robust with respect to measurement fluctuations. Departing from this approach, the proposed method directly optimizes LiDAR sensing and DSP parameters for downstream tasks. To investigate the optimization of LiDAR system parameters, we devise a realistic LiDAR simulation method that generates raw waveforms as input to a LiDAR DSP pipeline. We optimize LiDAR parameters for both 3D object detection IoU losses and depth error metrics by solving a nonlinear multi-objective optimization problem with a 0th-order stochastic algorithm. For automotive 3D object detection models, the proposed method outperforms manual expert tuning by 39.5% mean Average Precision (mAP).

        ----

        ## [1278] Learning and Aggregating Lane Graphs for Urban Automated Driving

        **Authors**: *Martin Büchner, Jannik Zürn, Ion-George Todoran, Abhinav Valada, Wolfram Burgard*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01289](https://doi.org/10.1109/CVPR52729.2023.01289)

        **Abstract**:

        Lane graph estimation is an essential and highly challenging task in automated driving and HD map learning. Existing methods using either onboard or aerial imagery struggle with complex lane topologies, out-oj-distribution scenar-ios, or significant occlusions in the image space. Moreover, merging overlapping lane graphs to obtain consistent large-scale graphs remains difficult. To overcome these challenges, we propose a novel bottom-up approach to lane graph esti-mation from aerial imagery that aggregates multiple over-lapping graphs into a single consistent graph. Due to its modular design, our method allows us to address two complementary tasks: predicting ego-respective successor lane graphs from arbitrary vehicle positions using a graph neural network and aggregating these predictions into a consistent global lane graph. Extensive experiments on a large-scale lane graph dataset demonstrate that our approach yields highly accurate lane graphs, even in regions with severe occlusions. The presented approach to graph aggregation proves to eliminate inconsistent predictions while increasing the overall graph quality. We make our large-scale urban lane graph dataset and code publicly available at http://urban1anegraph.cs.uni-freiburg.de

        ----

        ## [1279] Center Focusing Network for Real-Time LiDAR Panoptic Segmentation

        **Authors**: *Xiaoyan Li, Gang Zhang, Boyue Wang, Yongli Hu, Baocai Yin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01290](https://doi.org/10.1109/CVPR52729.2023.01290)

        **Abstract**:

        LiDAR panoptic segmentation facilitates an autonomous vehicle to comprehensively understand the surrounding objects and scenes and is required to run in real time. The recent proposal-free methods accelerate the algorithm, but their effectiveness and efficiency are still limited owing to the difficulty of modeling non-existent instance centers and the costly center-based clustering modules. To achieve accurate and real-time LiDAR panoptic segmentation, a novel center focusing network (CFNet) is introduced. Specifically, the center focusing feature encoding (CFFE) is proposed to explicitly understand the relationships between the original LiDAR points and virtual instance centers by shifting the LiDAR points and filling in the center points. Moreover, to leverage the redundantly detected centers, a fast center deduplication module (CDM) is proposed to select only one center for each instance. Experiments on the SemanticKITTI and nuScenes panoptic segmentation benchmarks demonstrate that our CFNet outperforms all existing methods by a large margin and is 1.6 times faster than the most efficient method.

        ----

        ## [1280] Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images

        **Authors**: *Bowei Du, Yecheng Huang, Jiaxin Chen, Di Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01291](https://doi.org/10.1109/CVPR52729.2023.01291)

        **Abstract**:

        Object detection on drone images with low-latency is an important but challenging task on the resource-constrained unmanned aerial vehicle (UAV) platform. This paper investigates optimizing the detection head based on the sparse convolution, which proves effective in balancing the accuracy and efficiency. Nevertheless, it suffers from inadequate integration of contextual information of tiny objects as well as clumsy control of the mask ratio in the presence of foreground with varying scales. To address the issues above, we propose a novel global context-enhanced adaptive sparse convolutional network (CEASC). It first develops a context-enhanced group normalization (CE-GN) layer, by replacing the statistics based on sparsely sampled features with the global contextual ones, and then designs an adaptive multi-layer masking strategy to generate optimal mask ratios at distinct scales for compact foreground coverage, promoting both the accuracy and efficiency. Extensive experimental results on two major benchmarks, i.e. VisDrone and UAVDT, demonstrate that CEASC remarkably reduces the GFLOPs and accelerates the inference procedure when plugging into the typical state-of-the-art detection frameworks (e.g. RetinaNet and GFL V1) with competitive performance. Code is available at https://github.com/Cuogeihong/CEASC.

        ----

        ## [1281] MV-JAR: Masked Voxel Jigsaw and Reconstruction for LiDAR-Based Self-Supervised Pre-Training

        **Authors**: *Runsen Xu, Tai Wang, Wenwei Zhang, Runjian Chen, Jinkun Cao, Jiangmiao Pang, Dahua Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01292](https://doi.org/10.1109/CVPR52729.2023.01292)

        **Abstract**:

        This paper introduces the Masked Voxel Jigsaw and Reconstruction (MV-JAR) method for LiDAR-based self-supervised pre-training and a carefully designed data-efficient 3D object detection benchmark on the Waymo dataset. Inspired by the scene-voxel-point hierarchy in downstream 3D object detectors, we design masking and reconstruction strategies accounting for voxel distributions in the scene and local point distributions within the voxel. We employ a Reversed-Furthest-Voxel-Sampling strategy to address the uneven distribution of LiDAR points and propose MV-JAR, which combines two techniques for modeling the aforementioned distributions, resulting in superior performance. Our experiments reveal limitations in previous data-efficient experiments, which uniformly sample fine-tuning splits with varying data proportions from each LiDAR sequence, leading to similar data diversity across splits. To address this, we propose a new benchmark that samples scene sequences for diverse fine-tuning splits, ensuring adequate model convergence and providing a more accurate evaluation of pre-training methods. Experiments on our Waymo benchmark and the KITTI dataset demonstrate that MV-JAR consistently and significantly improves 3D detection performance across various data scales, achieving up to a 6.3% increase in mAPH compared to training from scratch. Codes and the benchmark are available at https://github.com/SmartBot-PJLab/MV-JAR.

        ----

        ## [1282] ALSO: Automotive Lidar Self-Supervision by Occupancy Estimation

        **Authors**: *Alexandre Boulch, Corentin Sautier, Björn Michele, Gilles Puy, Renaud Marlet*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01293](https://doi.org/10.1109/CVPR52729.2023.01293)

        **Abstract**:

        We propose a new self-supervised method for pre-training the backbone of deep perception models operating on point clouds. The core idea is to train the model on a pretext task which is the reconstruction of the surface on which the 3D points are sampled, and to use the underlying latent vectors as input to the perception head. The intuition is that if the network is able to reconstruct the scene surface, given only sparse input points, then it probably also captures some fragments of semantic information, that can be used to boost an actual perception task. This principle has a very simple formulation, which makes it both easy to implement and widely applicable to a large range of 3D sensors and deep networks performing semantic segmentation or object detection. In fact, it supports a single-stream pipeline, as opposed to most contrastive learning approaches, allowing training on limited resources. We conducted extensive experiments on various autonomous driving datasets, involving very different kinds of lidars, for both semantic segmentation and object detection. The results show the effectiveness of our method to learn useful representations without any annotation, compared to existing approaches. The code is available at github.com/valeoai/ALSO

        ----

        ## [1283] Unsupervised Intrinsic Image Decomposition with LiDAR Intensity

        **Authors**: *Shogo Sato, Yasuhiro Yao, Taiga Yoshida, Takuhiro Kaneko, Shingo Ando, Jun Shimamura*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01294](https://doi.org/10.1109/CVPR52729.2023.01294)

        **Abstract**:

        Intrinsic image decomposition (IID) is the task that decomposes a natural image into albedo and shade. While IID is typically solved through supervised learning methods, it is not ideal due to the difficulty in observing ground truth albedo and shade in general scenes. Conversely, unsupervised learning methods are currently underperforming supervised learning methods since there are no criteria for solving the ill-posed problems. Recently, light detection and ranging (LiDAR) is widely used due to its ability to make highly precise distance measurements. Thus, we have focused on the utilization of LiDAR, especially LiDAR intensity, to address this issue. In this paper, we propose unsupervised intrinsic image decomposition with LiDAR intensity (IID-LI). Since the conventional unsupervised learning methods consist of image-to-image transformations, simply inputting LiDAR intensity is not an effective approach. Therefore, we design an intensity consistency loss that computes the error between LiDAR intensity and gray-scaled albedo to provide a criterion for the ill-posed problem. In addition, LiDAR intensity is difficult to handle due to its sparsity and occlusion, hence, a LiDAR intensity densification module is proposed. We verified the estimating quality using our own dataset, which include RGB images, LiDAR intensity and human judged annotations. As a result, we achieved an estimation accuracy that outperforms conventional unsupervised learning methods.

        ----

        ## [1284] PVT-SSD: Single-Stage 3D Object Detector with Point-Voxel Transformer

        **Authors**: *Honghui Yang, Wenxiao Wang, Minghao Chen, Binbin Lin, Tong He, Hua Chen, Xiaofei He, Wanli Ouyang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01295](https://doi.org/10.1109/CVPR52729.2023.01295)

        **Abstract**:

        Recent Transformer-based 3D object detectors learn point cloud features either from point- or voxel-based representations. However, the former requires time-consuming sampling while the latter introduces quantization errors. In this paper, we present a novel Point-Voxel Transformer for single-stage 3D detection (PVT-SSD) that takes advantage of these two representations. Specifically, we first use voxel-based sparse convolutions for efficient feature encoding. Then, we propose a Point-Voxel Transformer (PVT) module that obtains long-range contexts in a cheap manner from voxels while attaining accurate positions from points. The key to associating the two different representations is our introduced input-dependent Query Initialization module, which could efficiently generate reference points and content queries. Then, PVT adaptively fuses long-range contextual and local geometric information around reference points into content queries. Further, to quickly find the neighboring points of reference points, we design the Virtual Range Image module, which generalizes the native range image to multi-sensor and multi-frame. The experiments on several autonomous driving benchmarks verify the effectiveness and efficiency of the proposed method. Code will be available.

        ----

        ## [1285] LargeKernel3D: Scaling up Kernels in 3D Sparse CNNs

        **Authors**: *Yukang Chen, Jianhui Liu, Xiangyu Zhang, Xiaojuan Qi, Jiaya Jia*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01296](https://doi.org/10.1109/CVPR52729.2023.01296)

        **Abstract**:

        Recent advance in 2D CNNs has revealed that large kernels are important. However, when directly applying large convolutional kernels in 3D CNNs, severe difficulties are met, where those successful module designs in 2D become surprisingly ineffective on 3D networks, including the popular depth-wise convolution. To address this vital challenge, we instead propose the spatial-wise partition convolution and its large-kernel module. As a result, it avoids the optimization and efficiency issues of naive 3D large kernels. Our large-kernel 3D CNN network, LargeKernel3D, yields notable improvement in 3D tasks of semantic segmentation and object detection. It achieves 73.9% mIoU on the ScanNetv2 semantic segmentation and 72.8% NDS nuScenes object detection benchmarks, ranking 1st on the nuScenes LIDAR leaderboard. The performance further boosts to 74.2% NDS with a simple multi-modal fusion. In addition, LargeKernel3D can be scaled to 17×17×17 kernel size on Waymo 3D object detection. For the first time, we show that large kernels are feasible and essential for 3D visual tasks. Our code and models is available at github.com/dvlab-research/LargeKernel3D.

        ----

        ## [1286] WeatherStream: Light Transport Automation of Single Image Deweathering

        **Authors**: *Howard Zhang, Yunhao Ba, Ethan Yang, Varan Mehra, Blake Gella, Akira Suzuki, Arnold Pfahnl, Chethan Chinder Chandrappa, Alex Wong, Achuta Kadambi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01297](https://doi.org/10.1109/CVPR52729.2023.01297)

        **Abstract**:

        Today single image deweathering is arguably more sensitive to the dataset type, rather than the model. We introduce WeatherStream, an automatic pipeline capturing all real-world weather effects (rain, snow, and rain fog degradations), along with their clean image pairs. Previous state-of-the-art methods that have attempted the all-weather removal task train on synthetic pairs, and are thus limited by the Sim2Real domain gap. Recent work has attempted to manually collect time multiplexed pairs, but the use of human labor limits the scale of such a dataset. We introduce a pipeline that uses the power of light-transport physics and a model trained on a small, initial seed dataset to reject approximately 99.6% of unwanted scenes. The pipeline is able to generalize to new scenes and degradations that can, in turn, be used to train existing models just like fully human-labeled data. Training on a dataset collected through this procedure leads to significant improvements on multiple existing weather removal methods on a carefully human-collected test set of real-world weather effects. The dataset and code can be found in the following website: http://visual.ee.ucla.edu/wstream.htm/.

        ----

        ## [1287] Mask3D: Pretraining 2D Vision Transformers by Learning Masked 3D Priors

        **Authors**: *Ji Hou, Xiaoliang Dai, Zijian He, Angela Dai, Matthias Nießner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01298](https://doi.org/10.1109/CVPR52729.2023.01298)

        **Abstract**:

        Current popular backbones in computer vision, such as Vision Transformers (ViT) and ResNets are trained to per-ceive the world from 2D images. However, to more effectively understand 3D structural priors in 2D backbones, we propose Mask3D to leverage existing large-scale RGB-D data in a self-supervised pretraining to embed these 3D priors into 2D learned feature representations. In contrast to traditional 3D contrastive learning paradigms requiring 3D reconstructions or multi-view correspondences, our approach is simple: we formulate a pre-text reconstruction task by masking RGB and depth patches in individual RGB-D frames. We demonstrate the Mask3D is particularly effective in embedding 3D priors into the powerful 2D ViT backbone, enabling improved representation learning for various scene understanding tasks, such as semantic segmentation, instance segmentation and object detection. Experiments show that Mask3D notably outperforms existing self-supervised 3D pretraining approaches on ScanNet, NYUv2, and Cityscapes image understanding tasks, with an improvement of +6.5% mIoU against the state-of-the-art Pri3D on ScanNet image semantic segmentation.

        ----

        ## [1288] DSVT: Dynamic Sparse Voxel Transformer with Rotated Sets

        **Authors**: *Haiyang Wang, Chen Shi, Shaoshuai Shi, Meng Lei, Sen Wang, Di He, Bernt Schiele, Liwei Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01299](https://doi.org/10.1109/CVPR52729.2023.01299)

        **Abstract**:

        Designing an efficient yet deployment-friendly 3D backbone to handle sparse point clouds is a fundamental problem in 3D perception. Compared with the customized sparse convolution, the attention mechanism in Transformers is more appropriate for flexibly modeling long-range relationships and is easier to be deployed in real-world applications. However, due to the sparse characteristics of point clouds, it is non-trivial to apply a standard transformer on sparse points. In this paper, we present Dynamic Sparse Voxel Transformer (DSVT), a single-stride window-based voxel Transformer backbone for outdoor 3D perception. In order to efficiently process sparse points in parallel, we propose Dynamic Sparse Window Attention, which partitions a series of local regions in each window according to its sparsity and then computes the features of all regions in a fully parallel manner. To allow the cross-set connection, we design a rotated set partitioning strategy that alternates between two partitioning configurations in consecutive self-attention layers. To support effective downsampling and better encode geometric information, we also propose an attention-style 3D pooling module on sparse points, which is powerful and deployment-friendly without utilizing any customized CUDA operations. Our model achieves state-of-the-art performance with a broad range of 3D perception tasks. More importantly, DSVT can be easily deployed by TensorRT with real-time inference speed (27Hz). Code will be available at https://github.com/Haiyang-W/DSVT.

        ----

        ## [1289] IterativePFN: True Iterative Point Cloud Filtering

        **Authors**: *Dasith de Silva Edirimuni, Xuequan Lu, Zhiwen Shao, Gang Li, Antonio Robles-Kelly, Ying He*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01300](https://doi.org/10.1109/CVPR52729.2023.01300)

        **Abstract**:

        The quality of point clouds is often limited by noise introduced during their capture process. Consequently, a fundamental 3D vision task is the removal of noise, known as point cloud filtering or denoising. State-of-the-art learning based methods focus on training neural networks to infer filtered displacements and directly shift noisy points onto the underlying clean surfaces. In high noise conditions, they iterate the filtering process. However, this iterative filtering is only done at test time and is less effective at ensuring points converge quickly onto the clean surfaces. We propose IterativePFN (iterative point cloud filtering network), which consists of multiple IterationModules that model the true iterative filtering process internally, within a single network. We train our IterativePFn network using a novel loss function that utilizes an adaptive ground truth target at each iteration to capture the relationship between intermediate filtering results during training. This ensures that the filtered results converge faster to the clean surfaces. Our method is able to obtain better performance compared to state-of-the-art methods. The source code can be found at: https://github.com/ddsediri/IterativePFN

        ----

        ## [1290] itKD: Interchange Transfer-based Knowledge Distillation for 3D Object Detection

        **Authors**: *Hyeon Cho, Junyong Choi, Geonwoo Baek, Wonjun Hwang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01301](https://doi.org/10.1109/CVPR52729.2023.01301)

        **Abstract**:

        Point-cloud based 3D object detectors recently have achieved remarkable progress. However, most studies are limited to the development of network architectures for improving only their accuracy without consideration of the computational efficiency. In this paper, we first propose an autoencoder-style framework comprising channel-wise compression and decompression via interchange transfer-based knowledge distillation. To learn the map-view feature of a teacher network, the features from teacher and student networks are independently passed through the shared autoencoder; here, we use a compressed representation loss that binds the channel-wised compression knowledge from both student and teacher networks as a kind of regularization. The decompressed features are transferred in opposite directions to reduce the gap in the interchange reconstructions. Lastly, we present an head attention loss to match the 3D object detection information drawn by the multi-head self-attention mechanism. Through extensive experiments, we verify that our method can train the lightweight model that is well-aligned with the 3D point cloud detection task and we demonstrate its superiority using the well-known public datasets; e.g., Waymo and nuScenes. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
 Our code is available at https://github.com/hyeon-jo/interchange-transfer-KD.

        ----

        ## [1291] ISBNet: a 3D Point Cloud Instance Segmentation Network with Instance-aware Sampling and Box-aware Dynamic Convolution

        **Authors**: *Tuan Duc Ngo, Binh-Son Hua, Khoi Nguyen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01302](https://doi.org/10.1109/CVPR52729.2023.01302)

        **Abstract**:

        Existing 3D instance segmentation methods are predominated by the bottom-up design - manually fine-tuned algorithm to group points into clusters followed by a refinement network. However, by relying on the quality of the clusters, these methods generate susceptible results when (1) nearby objects with the same semantic class are packed together, or (2) large objects with loosely connected regions. To address these limitations, we introduce ISBNet, a novel cluster-free method that represents instances as kernels and decodes instance masks via dynamic convolution. To efficiently generate high-recall and discriminative kernels, we propose a simple strategy named Instance-aware Farthest Point Sampling to sample candidates and leverage the local aggregation layer inspired by PointNet++ to encode candidate features. Moreover, we show that predicting and leveraging the 3D axis-aligned bounding boxes in the dynamic convolution further boosts performance. Our method set new state-of-the-art results on ScanNetV2 (55.9), S3DIS (60.8), and STPLS3D (49.2) in terms of AP and retains fast inference time (237ms per scene on Scan-NetV2). The source code and trained models are available at https://github.com/VinAIResearch/ISBNet.

        ----

        ## [1292] Symmetric Shape-Preserving Autoencoder for Unsupervised Real Scene Point Cloud Completion

        **Authors**: *Changfeng Ma, Yinuo Chen, Pengxiao Guo, Jie Guo, Chongjun Wang, Yanwen Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01303](https://doi.org/10.1109/CVPR52729.2023.01303)

        **Abstract**:

        Unsupervised completion of real scene objects is of vital importance but still remains extremely challenging in preserving input shapes, predicting accurate results, and adapting to multi-category data. To solve these problems, we propose in this paper an Unsupervised Symmetric Shape-Preserving Autoencoding Network, termed USSPA, to predict complete point clouds of objects from real scenes. One of our main observations is that many natural and manmade objects exhibit significant symmetries. To accommodate this, we devise a symmetry learning module to learn from those objects and to preserve structural symmetries. Starting from an initial coarse predictor, our autoencoder refines the complete shape with a carefully designed upsampling refinement module. Besides the discriminative process on the latent space, the discriminators of our USSPA also take predicted point clouds as direct guidance, enabling more detailed shape prediction. Clearly different from previous methods which train each category separately, our USSPA can be adapted to the training of multi-category data in one pass through a classifier-guided discriminator, with consistent performance on single category. For more accurate evaluation, we contribute to the community a real scene dataset with paired CAD models as ground truth. Extensive experiments and comparisons demonstrate our superiority and generalization and show that our method achieves state-of-the-art performance on unsupervised completion of real scene objects.

        ----

        ## [1293] GeoMAE: Masked Geometric Target Prediction for Self-supervised Point Cloud Pre-Training

        **Authors**: *Xiaoyu Tian, Haoxi Ran, Yue Wang, Hang Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01304](https://doi.org/10.1109/CVPR52729.2023.01304)

        **Abstract**:

        This paper tries to address a fundamental question in point cloud self-supervised learning: what is a good signal we should leverage to learn features from point clouds without annotations? To answer that, we introduce a point cloud representation learning framework, based on geometric feature reconstruction. In contrast to recent papers that directly adopt masked autoencoder (MAE) and only predict original coordinates or occupancy from masked point clouds, our method revisits differences between images and point clouds and identifies three self-supervised learning objectives peculiar to point clouds, namely centroid prediction, normal estimation, and curvature prediction. Combined, these three objectives yield an nontrivial self-supervised learning task and mutually facilitate models to better reason fine-grained geometry of point clouds. Our pipeline is conceptually simple and it consists of two major steps: first, it randomly masks out groups of points, followed by a Transformer-based point cloud encoder; second, a lightweight Transformer decoder predicts centroid, normal, and curvature for points in each voxel. We transfer the pre-trained Transformer encoder to a downstream peception model. On the nuScene Datset, our model achieves 3.38 mAP improvment for object detection, 2.1 mIoU gain for segmentation, and 1.7 AMOTA gain for multi-object tracking. We also conduct experiments on the Waymo Open Dataset and achieve significant performance improvements over baselines as well. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Our code is available at https://github.com/Tsinghua-MARS-Lab/GeoMAE.

        ----

        ## [1294] AnchorFormer: Point Cloud Completion from Discriminative Nodes

        **Authors**: *Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, Tao Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01305](https://doi.org/10.1109/CVPR52729.2023.01305)

        **Abstract**:

        Point cloud completion aims to recover the completed 3D shape of an object from its partial observation. A common strategy is to encode the observed points to a global feature vector and then predict the complete points through a generative process on this vector. Nevertheless, the results may suffer from the high-quality shape generation problem due to the fact that a global feature vector cannot sufficiently characterize diverse patterns in one object. In this paper, we present a new shape completion architecture, namely AnchorFormer, that innovatively leverages pattern-aware discriminative nodes, i.e., anchors, to dynamically capture regional information of objects. Technically, AnchorFormer models the regional discrimination by learning a set of anchors based on the point features of the input partial observation. Such anchors are scattered to both observed and unobserved locations through estimating particular offsets, and form sparse points together with the down-sampled points of the input observation. To reconstruct the finegrained object patterns, AnchorFormer further employs a modulation scheme to morph a canonical 2D grid at individual locations of the sparse points into a detailed 3D structure. Extensive experiments on the PCN, ShapeNet-55/34 and KITTI datasets quantitatively and qualitatively demonstrate the efficacy of AnchorFormer over the state-of-the-art point cloud completion approaches. Source code is available at https://github.com/chenzhik/AnchorFormer.

        ----

        ## [1295] SHS-Net: Learning Signed Hyper Surfaces for Oriented Normal Estimation of Point Clouds

        **Authors**: *Qing Li, Huifang Feng, Kanle Shi, Yue Gao, Yi Fang, Yu-Shen Liu, Zhizhong Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01306](https://doi.org/10.1109/CVPR52729.2023.01306)

        **Abstract**:

        We propose a novel method called SHS-Net for oriented normal estimation of point clouds by learning signed hyper surfaces, which can accurately predict normals with global consistent orientation from various point clouds. Almost all existing methods estimate oriented normals through a two-stage pipeline, i.e., unoriented normal estimation and normal orientation, and each step is implemented by a separate algorithm. However, previous methods are sensitive to parameter settings, resulting in poor results from point clouds with noise, density variations and complex geometries. In this work, we introduce signed hyper surfaces (SHS), which are parameterized by multi-layer perceptron (MLP) layers, to learn to estimate oriented normals from point clouds in an end-to-end manner. The signed hyper surfaces are implicitly learned in a high-dimensional feature space where the local and global information is aggregated. Specifically, we introduce a patch encoding module and a shape encoding module to encode a 3D point cloud into a local latent code and a global latent code, respectively. Then, an attention-weighted normal prediction module is proposed as a decoder, which takes the local and global latent codes as input to predict oriented normals. Experimental results show that our SHS-Net outperforms the state-of-the-art methods in both unoriented and oriented normal estimation on the widely used benchmarks. The code, data and pretrained models are available at https://github.com/LeoQLi/SHS-Net.

        ----

        ## [1296] NerVE: Neural Volumetric Edges for Parametric Curve Extraction from Point Cloud

        **Authors**: *Xiangyu Zhu, Dong Du, Weikai Chen, Zhiyou Zhao, Yinyu Nie, Xiaoguang Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01307](https://doi.org/10.1109/CVPR52729.2023.01307)

        **Abstract**:

        Extracting parametric edge curves from point clouds is a fundamental problem in 3D vision and geometry processing. Existing approaches mainly rely on keypoint detection, a challenging procedure that tends to generate noisy out-put, making the subsequent edge extraction error-prone. To address this issue, we propose to directly detect structured edges to circumvent the limitations of the previous point-wise methods. We achieve this goal by presenting NerVE, a novel neural volumetric edge representation that can be easily learned through a volumetric learning framework. NerVE can be seamlessly converted to a versatile piece-wise lin-ear (PWL) curve representation, enabling a unified strategy for learning all types offree-form curves. Furthermore, as NerVE encodes rich structural information, we show that edge extraction based on NerVE can be reduced to a simple graph search problem. After converting NerVE to the PWL representation, parametric curves can be obtained via off-the-shelf spline fitting algorithms. We evaluate our method on the challenging ABC dataset [19]. We show that a sim-ple network based on NerVE can already outperform the previous state-of-the-art methods by a great margin.

        ----

        ## [1297] Unsupervised Deep Probabilistic Approach for Partial Point Cloud Registration

        **Authors**: *Guofeng Mei, Hao Tang, Xiaoshui Huang, Weijie Wang, Juan Liu, Jian Zhang, Luc Van Gool, Qiang Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01308](https://doi.org/10.1109/CVPR52729.2023.01308)

        **Abstract**:

        Deep point cloud registration methods face challenges to partial overlaps and rely on labeled data. To address these issues, we propose UDPReg, an unsupervised deep probabilistic registration framework for point clouds with partial overlaps. Specifically, we first adopt a network to learn posterior probability distributions of Gaussian mixture models (GMMs) from point clouds. To handle partial point cloud registration, we apply the Sinkhorn algorithm to predict the distribution-level correspondences under the constraint of the mixing weights of GMMs. To enable unsupervised learning, we design three distribution consistency-based losses: self-consistency, cross-consistency, and local contrastive. The self-consistency loss is formulated by encouraging GMMs in Euclidean and feature spaces to share identical posterior distributions. The cross-consistency loss derives from the fact that the points of two partially overlapping point clouds belonging to the same clusters share the cluster centroids. The cross-consistency loss allows the network to flexibly learn a transformation-invariant posterior distribution of two aligned point clouds. The local contrastive loss facilitates the network to extract discriminative local features. Our UDPReg achieves competitive performance on the 3DMatch/3DLoMatch and ModelNet/ModelLoNet benchmarks.

        ----

        ## [1298] Local Connectivity-Based Density Estimation for Face Clustering

        **Authors**: *Junho Shin, Hyo-Jun Lee, Hyunseop Kim, Jong-Hyeon Baek, Daehyun Kim, Yeong Jun Koh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01309](https://doi.org/10.1109/CVPR52729.2023.01309)

        **Abstract**:

        Recent graph-based face clustering methods predict the connectivity of enormous edges, including false positive edges that link nodes with different classes. However, those false positive edges, which connect negative node pairs, have the risk of integration of different clusters when their connectivity is incorrectly estimated. This paper proposes a novel face clustering method to address this problem. The proposed clustering method employs density-based clustering, which maintains edges that have higher density. For this purpose, we propose a reliable density estimation algorithm based on local connectivity between K nearest neighbors (KNN). We effectively exclude negative pairs from the KNN graph based on the reliable density while maintaining sufficient positive pairs. Furthermore, we develop a pairwise connectivity estimation network to predict the connectivity of the selected edges. Experimental results demonstrate that the proposed clustering method significantly outperforms the state-of-the-art clustering methods on large-scale face clustering datasets and fashion image clustering datasets. Our code is available at https://github.com/illian01/LCE-PCENet

        ----

        ## [1299] Bridging Search Region Interaction with Template for RGB-T Tracking

        **Authors**: *Tianrui Hui, Zizheng Xun, Fengguang Peng, Junshi Huang, Xiaoming Wei, Xiaolin Wei, Jiao Dai, Jizhong Han, Si Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01310](https://doi.org/10.1109/CVPR52729.2023.01310)

        **Abstract**:

        RGB-T tracking aims to leverage the mutual enhancement and complement ability of RGB and TIR modalities for improving the tracking process in various scenarios, where cross-modal interaction is the key component. Some previous methods concatenate the RGB and TIR search region features directly to perform a coarse interaction process with redundant background noises introduced. Many other methods sample candidate boxes from search frames and conduct various fusion approaches on isolated pairs of RGB and TIR boxes, which limits the cross-modal interaction within local regions and brings about inadequate context modeling. To alleviate these limitations, we propose a novel Template-Bridged Search region Interaction (TBSI) module which exploits templates as the medium to bridge the cross-modal interaction between RGB and TIR search regions by gathering and distributing target-relevant object and environment contexts. Original templates are also updated with enriched multimodal contexts from the template medium. Our TBSI module is inserted into a ViT backbone for joint feature extraction, search-template matching, and cross-modal interaction. Extensive experiments on three popular RGB-T tracking benchmarks demonstrate our method achieves new state-of-the-art performances. Code is available at https://github.com/RyanHTR/TBSI.

        ----

        ## [1300] Quantum Multi-Model Fitting

        **Authors**: *Matteo Farina, Luca Magri, Willi Menapace, Elisa Ricci, Vladislav Golyanik, Federica Arrigoni*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01311](https://doi.org/10.1109/CVPR52729.2023.01311)

        **Abstract**:

        Geometric model fitting is a challenging but fundamental computer vision problem. Recently, quantum optimization has been shown to enhance robust fitting for the case of a single model, while leaving the question of multi-model fitting open. In response to this challenge, this paper shows that the latter case can significantly benefit from quantum hardware and proposes the first quantum approach to multimodel fitting (MMF). We formulate MMF as a problem that can be efficiently sampled by modern adiabatic quantum computers without the relaxation of the objective function. We also propose an iterative and decomposed version of our method, which supports real-world-sized problems. The experimental evaluation demonstrates promising results on a variety of datasets. The source code is available at: https://github.com/FarinaMatteo/qmmf.

        ----

        ## [1301] Generalizable Local Feature Pre-training for Deformable Shape Analysis

        **Authors**: *Souhaib Attaiki, Lei Li, Maks Ovsjanikov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01312](https://doi.org/10.1109/CVPR52729.2023.01312)

        **Abstract**:

        Transfer learning is fundamental for addressing problems in settings with little training data. While several transfer learning approaches have been proposed in 3D, unfortunately, these solutions typically operate on an entire 3D object or even scene-level and thus, as we show, fail to generalize to new classes, such as deformable organic shapes. In addition, there is currently a lack of understanding of what makes pre-trained features transferable across significantly different 3D shape categories. In this paper, we make a step toward addressing these challenges. First, we analyze the link between feature locality and transferability in tasks involving deformable 3D objects, while also comparing different backbones and losses for local feature pre-training. We observe that with proper training, learned features can be useful in such tasks, but, crucially, only with an appropriate choice of the receptive field size. We then propose a differentiable method for optimizing the receptive field within 3D transfer learning. Jointly, this leads to the first learnable features that can successfully generalize to unseen classes of 3D shapes such as humans and animals. Our extensive experiments show that this approach leads to state-of-the-art results on several downstream tasks such as segmentation, shape correspondence, and classification. Our code is available at https://github.com/pvnieo/vader.

        ----

        ## [1302] Similarity Metric Learning For RGB-Infrared Group Re-Identification

        **Authors**: *Jianghao Xiong, Jianhuang Lai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01313](https://doi.org/10.1109/CVPR52729.2023.01313)

        **Abstract**:

        Group re-identification (G-ReID) aims to re-identify a group of people that is observed from non-overlapping camera systems. The existing literature has mainly addressed RGB-based problems, but RGB-infrared (RGB-IR) cross-modality matching problem has not been studied yet. In this paper, we propose a metric learning method Closest Permutation Matching (CPM) for RGB-IR G-ReID. We model each group as a set of single-person features which are extracted by MPANet, then we propose the metric Closest Permutation Distance (CPD) to measure the similarity between two sets of features. CPD is invariant with order changes of group members so that it solves the layout change problem in G-ReID. Furthermore, we introduce the problem of G-ReID without person labels. In the weak-supervised case, we design the Relation-aware Module (RAM) that exploits visual context and relations among group members to produce a modality-invariant order of features in each group, with which group member features within a set can be sorted to form a robust group representation against modality change. To support the study on RGB-IR G-ReID, we construct a new large-scale RGB-IR G-ReID dataset CM-Group. The dataset contains 15,440 RGB images and 15,506 infrared images of 427 groups and 1,013 identi-ties. Extensive experiments on the new dataset demonstrate the effectiveness of the proposed models and the complexity of CM-Group. The code and dataset are available at: https://github.com/WhollyOat/CM-Group.

        ----

        ## [1303] Unsupervised Deep Asymmetric Stereo Matching with Spatially-Adaptive Self-Similarity

        **Authors**: *Taeyong Song, Sunok Kim, Kwanghoon Sohn*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01314](https://doi.org/10.1109/CVPR52729.2023.01314)

        **Abstract**:

        Unsupervised stereo matching has received a lot of attention since it enables the learning of disparity estimation without ground-truth data. However, most of the unsupervised stereo matching algorithms assume that the left and right images have consistent visual properties, i.e., symmetric, and easily fail when the stereo images are asymmetric. In this paper, we present a novel spatially-adaptive self-similarity (SASS) for unsupervised asymmetric stereo matching. It extends the concept of self-similarity and generates deep features that are robust to the asymmetries. The sampling patterns to calculate self-similarities are adaptively generated throughout the image regions to effectively encode diverse patterns. In order to learn the effective sampling patterns, we design a contrastive similarity loss with positive and negative weights. Consequently, SASS is further encouraged to encode asymmetry-agnostic features, while maintaining the distinctiveness for stereo correspondence. We present extensive experimental results including ablation studies and comparisons with different methods, demonstrating effectiveness of the proposed method under resolution and noise asymmetries.

        ----

        ## [1304] Sliced Optimal Partial Transport

        **Authors**: *Yikun Bai, Bernhard Schmitzer, Matthew Thorpe, Soheil Kolouri*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01315](https://doi.org/10.1109/CVPR52729.2023.01315)

        **Abstract**:

        Optimal transport (OT) has become exceedingly popular in machine learning, data science, and computer vision. The core assumption in the OT problem is the equal total amount of mass in source and target measures, which limits its application. Optimal Partial Transport (OPT) is a recently proposed solution to this limitation. Similar to the OT problem, the computation of OPT relies on solving a linear programming problem (often in high dimensions), which can become computationally prohibitive. In this paper, we propose an efficient algorithm for calculating the OPT problem between two non-negative measures in one dimension. Next, following the idea of sliced OT distances, we utilize slicing to define the Sliced OPT distance. Finally, we demonstrate the computational and accuracy benefits of the Sliced OPT-based method in various numerical experiments. In particular, we show an application of our proposed Sliced OPT problem in noisy point cloud registration and color adaptation. Our code is available at Github Link.

        ----

        ## [1305] DistractFlow: Improving Optical Flow Estimation via Realistic Distractions and Pseudo-Labeling

        **Authors**: *Jisoo Jeong, Hong Cai, Risheek Garrepalli, Fatih Porikli*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01316](https://doi.org/10.1109/CVPR52729.2023.01316)

        **Abstract**:

        We propose a novel data augmentation approach, DistractFlow, for training optical flow estimation models by introducing realistic distractions to the input frames. Based on a mixing ratio, we combine one of the frames in the pair with a distractor image depicting a similar domain, which allows for inducing visual perturbations congruent with natural objects and scenes. We refer to such pairs as distracted pairs. Our intuition is that using semantically meaningful distractors enables the model to learn related variations and attain robustness against challenging deviations, compared to conventional augmentation schemes focusing only on low-level aspects and modifications. More specifically, in addition to the supervised loss computed between the estimated flow for the original pair and its ground-truth flow, we include a second supervised loss defined between the distracted pair's flow and the original pair's ground-truth flow, weighted with the same mixing ratio. Furthermore, when unlabeled data is available, we extend our augmentation approach to self-supervised settings through pseudo-labeling and cross-consistency regularization. Given an original pair and its distracted version, we enforce the estimated flow on the distracted pair to agree with the flow of the original pair. Our approach allows increasing the number of available training pairs significantly without requiring additional annotations. It is agnostic to the model architecture and can be applied to training any optical flow estimation models. Our extensive evaluations on multiple benchmarks, including Sintel, KITTI, and SlowFlow, show that DistractFlow improves existing models consistently, outperforming the latest state of the art.

        ----

        ## [1306] Bayesian Posterior Approximation With Stochastic Ensembles

        **Authors**: *Oleksandr Balabanov, Bernhard Mehlig, Hampus Linander*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01317](https://doi.org/10.1109/CVPR52729.2023.01317)

        **Abstract**:

        We introduce ensembles of stochastic neural networks to approximate the Bayesian posterior, combining stochastic methods such as dropout with deep ensembles. The stochas-tic ensembles are formulated as families of distributions and trained to approximate the Bayesian posterior with variational inference. We implement stochastic ensembles based on Monte Carlo dropout, DropConnect and a novel non-parametric version of dropout and evaluate them on a toy problem and CIFAR image classification. For both tasks, we test the quality of the posteriors directly against Hamil-tonian Monte Carlo simulations. Our results show that stochastic ensembles provide more accurate posterior esti-mates than other popular baselines for Bayesian inference.

        ----

        ## [1307] V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception

        **Authors**: *Runsheng Xu, Xin Xia, Jinlong Li, Hanzhao Li, Shuo Zhang, Zhengzhong Tu, Zonglin Meng, Hao Xiang, Xiaoyu Dong, Rui Song, Hongkai Yu, Bolei Zhou, Jiaqi Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01318](https://doi.org/10.1109/CVPR52729.2023.01318)

        **Abstract**:

        Modern perception systems of autonomous vehicles are known to be sensitive to occlusions and lack the capability of long perceiving range. It has been one of the key bottlenecks that prevents Level 5 autonomy. Recent research has demonstrated that the Vehicle-to-Vehicle (V2V) cooperative perception system has great potential to revolutionize the autonomous driving industry. However, the lack of a real-world dataset hinders the progress of this field. To facilitate the development of cooperative perception, we present V2V4Real, the first large-scale real-world multi-modal dataset for V2V perception. The data is collected by two vehicles equipped with multi-modal sensors driving together through diverse scenarios. Our V2V4Real dataset covers a driving area of 410 km, comprising 20K LiDAR frames, 40K RGB frames, 240K annotated 3D bounding boxes for 5 classes, and HDMaps that cover all the driving routes. V2V4Real introduces three perception tasks, including cooperative 3D object detection, cooperative 3D object tracking, and Sim2Real domain adaptation for cooperative perception. We provide comprehensive benchmarks of recent cooperative perception algorithms on three tasks. The V2V4Real dataset can be found at research.seas.ucla.edu/mobility-lab/v2v4real/.

        ----

        ## [1308] ReasonNet: End-to-End Driving with Temporal and Global Reasoning

        **Authors**: *Hao Shao, Letian Wang, Ruobing Chen, Steven L. Waslander, Hongsheng Li, Yu Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01319](https://doi.org/10.1109/CVPR52729.2023.01319)

        **Abstract**:

        The large-scale deployment of autonomous vehicles is yet to come, and one of the major remaining challenges lies in urban dense traffic scenarios. In such cases, it remains challenging to predict the future evolution of the scene and future behaviors of objects, and to deal with rare adverse events such as the sudden appearance of occluded objects. In this paper, we present ReasonNet, a novel end-to-end driving framework that extensively exploits both temporal and global information of the driving scene. By reasoning on the temporal behavior of objects, our method can effectively process the interactions and relationships among features in different frames. Reasoning about the global information of the scene can also improve overall perception performance and benefit the detection of adverse events, especially the anticipation of potential danger from occluded objects. For comprehensive evaluation on occlusion events, we also release publicly a driving simulation benchmark DriveOcclusionSim consisting of diverse occlusion events. We conduct extensive experiments on multiple CARLA benchmarks, where our model outperforms all prior methods, ranking first on the sensor track of the public CARLA Leaderboard [53].

        ----

        ## [1309] Open-World Multi-Task Control Through Goal-Aware Representation Learning and Adaptive Horizon Prediction

        **Authors**: *Shaofei Cai, Zihao Wang, Xiaojian Ma, Anji Liu, Yitao Liang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01320](https://doi.org/10.1109/CVPR52729.2023.01320)

        **Abstract**:

        We study the problem of learning goal-conditioned policies in Minecraft, a popular, widely accessible yet challenging open-ended environment for developing human-level multi-task agents. We first identify two main challenges of learning such policies: 1) the indistinguishability of tasks from the state distribution, due to the vast scene diversity, and 2) the non-stationary nature of environment dynamics caused by partial observability. To tackle the first challenge, we propose Goal-Sensitive Backbone (GSB) for the policy to encourage the emergence of goal-relevant visual state representations. To tackle the second challenge, the policy is further fueled by an adaptive horizon prediction module that helps alleviate the learning uncertainty brought by the non-stationary dynamics. Experiments on 20 Minecraft tasks show that our method significantly outperforms the best baseline so far; in many of them, we double the performance. Our ablation and exploratory studies then explain how our approach beat the counterparts and also unveil the surprising bonus of zero-shot generalization to new scenes (biomes). We hope our agent could help shed some light on learning goal-conditioned, multi-task agents in challenging, open-ended environments like Minecraft. The code is released at https://github.com/CraftJarvis/MC-Controller.

        ----

        ## [1310] FJMP: Factorized Joint Multi-Agent Motion Prediction over Learned Directed Acyclic Interaction Graphs

        **Authors**: *Luke Rowe, Martin Ethier, Eli-Henry Dykhne, Krzysztof Czarnecki*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01321](https://doi.org/10.1109/CVPR52729.2023.01321)

        **Abstract**:

        Predicting the future motion of road agents is a critical task in an autonomous driving pipeline. In this work, we address the problem of generating a set of scene-level, or joint, future trajectory predictions in multi-agent driving scenarios. To this end, we propose FJMP, a Factorized Joint Motion Prediction framework for multi-agent interactive driving scenarios. FJMP models the future scene interaction dynamics as a sparse directed interaction graph, where edges denote explicit interactions between agents. We then prune the graph into a directed acyclic graph (DAG) and decompose the joint prediction task into a sequence of marginal and conditional predictions according to the partial ordering of the DAG, where joint future trajectories are decoded using a directed acyclic graph neural network (DAGNN). We conduct experiments on the INTERACTION and Argoverse 2 datasets and demonstrate that FJMP produces more accurate and scene-consistent joint trajectory predictions than non-factorized approaches, especially on the most interactive and kinematically interesting agents. FJMP ranks 1st on the multi-agent test leaderboard of the INTERACTION dataset.

        ----

        ## [1311] Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion

        **Authors**: *Davis Rempe, Zhengyi Luo, Xue Bin Peng, Ye Yuan, Kris Kitani, Karsten Kreis, Sanja Fidler, Or Litany*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01322](https://doi.org/10.1109/CVPR52729.2023.01322)

        **Abstract**:

        We introduce a method for generating realistic pedestrian trajectories and full-body animations that can be controlled to meet user-defined goals. We draw on recent advances in guided diffusion modeling to achieve test-time controllability of trajectories, which is normally only associated with rule-based systems. Our guided diffusion model allows users to constrain trajectories through target waypoints, speed, and specified social groups while accounting for the surrounding environment context. This trajectory diffusion model is integrated with a novel physics-based humanoid controller to form a closed-loop, full-body pedestrian animation system capable of placing large crowds in a simulated environment with varying terrains. We further propose utilizing the value function learned during RL training of the animation controller to guide diffusion to produce trajectories better suited for particular scenarios such as collision avoidance and traversing uneven terrain. Video results are available on the project page.

        ----

        ## [1312] Galactic: Scaling End-to-End Reinforcement Learning for Rearrangement at 100k Steps-Per-Second

        **Authors**: *Vincent-Pierre Berges, Andrew Szot, Devendra Singh Chaplot, Aaron Gokaslan, Roozbeh Mottaghi, Dhruv Batra, Eric Undersander*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01323](https://doi.org/10.1109/CVPR52729.2023.01323)

        **Abstract**:

        We present Galactic, a large-scale simulation and reinforcement-learning (RL) framework for robotic mobile manipulation in indoor environments. Specifically, a Fetch robot (equipped with a mobile base, 7DoF arm, RGBD camera, egomotion, and onboard sensing) is spawned in a home environment and asked to rearrange objects - by navigating to an object, picking it up, navigating to a target location, and then placing the object at the target location. Galactic is fast. In terms of simulation speed (rendering + physics), Galactic achieves over 421,000 steps-per-second (SPS) on an 8-GPU node, which is 54x faster than Habitat 2.0 [55] (7699 SPS). More importantly, Galactic was designed to optimize the entire rendering+physics+RL interplay since any bottleneck in the interplay slows down training. In terms of simulation+RL speed (rendering + physics + inference + learning), Galactic achieves over 108,000 SPS, which 88x faster than Habitat 2.0 (1243 SPS). These massive speedups not only drastically cut the wall-clock training time of existing experiments, but also unlock an unprecedented scale of new experiments. First, Galactic can train a mobile pick skill to > 80% accuracy in under 16 minutes, a 100x speedup compared to the over 24 hours it takes to train the same skill in Habitat 2.0. Second, we use Galactic to perform the largest-scale experiment to date for rearrangement using 5B steps of experience in 46 hours, which is equivalent to 20 years of robot experience. This scaling results in a single neural network composed of task-agnostic components achieving 85% success in GeometricGoal rearrangement, compared to 0% success reported in Habitat 2.0 for the same approach. The code is available at github.com/facebookresearch/galactic.

        ----

        ## [1313] Indiscernible Object Counting in Underwater Scenes

        **Authors**: *Guolei Sun, Zhaochong An, Yun Liu, Ce Liu, Christos Sakaridis, Deng-Ping Fan, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01325](https://doi.org/10.1109/CVPR52729.2023.01325)

        **Abstract**:

        Recently, indiscernible scene understanding has attracted a lot of attention in the vision community. We further advance the frontier of this field by systematically studying a new challenge named indiscernible object counting (IOC), the goal of which is to count objects that are blended with respect to their surroundings. Due to a lack of appropriate IOC datasets, we present a large-scale dataset IOCfish5K which contains a total of 5,637 high-resolution images and 659,024 annotated center points. Our dataset consists of a large number of indiscernible objects (mainly fish) in underwater scenes, making the annotation process all the more challenging. IOCfish5K is superior to existing datasets with indiscernible scenes because of its larger scale, higher image resolutions, more annotations, and denser scenes. All these aspects make it the most challenging dataset for IOC so far, supporting progress in this area. For benchmarking purposes, we select 14 mainstream methods for object counting and carefully evaluate them on IOCfish5K. Furthermore, we propose IOCFormer, a new strong baseline that combines density and regression branches in a unified framework and can effectively tackle object counting under concealed scenes. Experiments show that IOCFormer achieves state-of-the-art scores on IOCfish5K. The resources are available at github.com/GuoleiSun/Indiscernible-Object-Counting.

        ----

        ## [1314] Tracking Through Containers and Occluders in the Wild

        **Authors**: *Basile Van Hoorick, Pavel Tokmakov, Simon Stent, Jie Li, Carl Vondrick*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01326](https://doi.org/10.1109/CVPR52729.2023.01326)

        **Abstract**:

        Tracking objects with persistence in cluttered and dynamic environments remains a difficult challenge for computer vision systems. In this paper, we introduce TCOW, a new benchmark and model for visual tracking through heavy occlusion and containment. We set up a task where the goal is to, given a video sequence, segment both the projected extent of the target object, as well as the surrounding container or occluder whenever one exists. To study this task, we create a mixture of synthetic and annotated real datasets to support both supervised learning and structured evaluation of model performance under various forms of task variation, such as moving or nested containment. We evaluate two recent transformer-based video models and find that while they can be surprisingly capable of tracking targets under certain settings of task variation, there remains a considerable performance gap before we can claim a tracking model to have acquired a true notion of object permanence.

        ----

        ## [1315] Simple Cues Lead to a Strong Multi-Object Tracker

        **Authors**: *Jenny Seidenschwarz, Guillem Brasó, Victor Castro Serrano, Ismail Elezi, Laura Leal-Taixé*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01327](https://doi.org/10.1109/CVPR52729.2023.01327)

        **Abstract**:

        For a long time, the most common paradigm in Multi-Object Tracking was tracking-by-detection (TbD), where objects are first detected and then associated over video frames. For association, most models resourced to motion and appearance cues, e.g., re-identification networks. Recent approaches based on attention propose to learn the cues in a data-driven manner, showing impressive results. In this paper, we ask ourselves whether simple good old TbD methods are also capable of achieving the performance of end-to-end models. To this end, we propose two key ingredients that allow a standard re-identification network to excel at appearance-based tracking. We extensively analyse its failure cases, and show that a combination of our appearance features with a simple motion model leads to strong tracking results. Our tracker generalizes to four public datasets, namely MOT17, MOT20, BDD100k, and DanceTrack, achieving state-of-the-art performance. https://github.com/dvl-tum/GHOST.

        ----

        ## [1316] An In-Depth Exploration of Person Re-Identification and Gait Recognition in Cloth-Changing Conditions

        **Authors**: *Weijia Li, Saihui Hou, Chunjie Zhang, Chunshui Cao, Xu Liu, Yongzhen Huang, Yao Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01328](https://doi.org/10.1109/CVPR52729.2023.01328)

        **Abstract**:

        The target of person re-identification (ReID) and gait recognition is consistent, that is to match the target pedestrian under surveillance cameras. For the cloth-changing problem, video-based ReID is rarely studied due to the lack of a suitable cloth-changing benchmark, and gait recognition is often researched under controlled conditions. To tackle this problem, we propose a Cloth-Changing benchmark for Person re-identification and Gait recognition (CCPG). It is a cloth-changing dataset, and there are several highlights in CCPG, (1) it provides 200 identities and over 16K sequences are captured indoors and outdoors, (2) each identity has seven different cloth-changing statuses, which is hardly seen in previous datasets, (3) RGB and silhouettes version data are both available for research purposes. Moreover, aiming to investigate the cloth-changing problem systematically, comprehensive experiments are conducted on video-based ReID and gait recognition methods. The experimental results demonstrate the superiority of ReID and gait recognition separately in different cloth-changing conditions and suggest that gait recognition is a potential solution for addressing the cloth-changing problem. Our dataset will be available at https://github.com/BNU-IVC/CCPG.

        ----

        ## [1317] SelfME: Self-Supervised Motion Learning for Micro-Expression Recognition

        **Authors**: *Xinqi Fan, Xueli Chen, Mingjie Jiang, Ali Raza Shahid, Hong Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01329](https://doi.org/10.1109/CVPR52729.2023.01329)

        **Abstract**:

        Facial micro-expressions (MEs) refer to brief spontaneous facial movements that can reveal a person's genuine emotion. They are valuable in lie detection, criminal analysis, and other areas. While deep learning-based ME recognition (MER) methods achieved impressive success, these methods typically require pre-processing using conventional optical flow-based methods to extract facial motions as inputs. To overcome this limitation, we proposed a novel MER framework using self-supervised learning to extract facial motion for ME (SelfME). To the best of our knowledge, this is the first work using an automatically self-learned motion technique for MER. However, the self-supervised motion learning method might suffer from ignoring symmetrical facial actions on the left and right sides of faces when extracting fine features. To address this issue, we developed a symmetric contrastive vision transformer (SCViT) to constrain the learning of similar facial action features for the left and right parts of faces. Experiments were conducted on two benchmark datasets showing that our method achieved state-of-the-art performance, and ablation studies demonstrated the effectiveness of our method.

        ----

        ## [1318] LipFormer: High-fidelity and Generalizable Talking Face Generation with A Pre-learned Facial Codebook

        **Authors**: *Jiayu Wang, Kang Zhao, Shiwei Zhang, Yingya Zhang, Yujun Shen, Deli Zhao, Jingren Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01330](https://doi.org/10.1109/CVPR52729.2023.01330)

        **Abstract**:

        Generating a talking face video from the input audio sequence is a practical yet challenging task. Most existing methods either fail to capture fine facial details or need to train a specific model for each identity. We argue that a codebook pre-learned on high-quality face images can serve as a useful prior that facilitates high-fidelity and generalizable talking head synthesis. Thanks to the strong capability of the codebook in representing face textures, we simplify the talking face generation task as finding proper lip-codes to characterize the variation of lips during portrait talking. To this end, we propose LipFormer, a Transformer-based framework to model the audio-visual coherence and predict the lip-codes sequence based on input audio features. We further introduce an adaptive face warping module, which helps warp the reference face to the target pose in the feature space, to alleviate the difficulty of lip-code prediction under different poses. By this means, LipFormer can make better use of prelearned priors in images and is robust to posture change. Extensive experiments show that LipFormer can produce more realistic talking face videos compared to previous methods and faithfully generalize to unseen identities.

        ----

        ## [1319] Real-time Multi-person Eyeblink Detection in the Wild for Untrimmed Video

        **Authors**: *Wenzheng Zeng, Yang Xiao, Sicheng Wei, Jinfang Gan, Xintao Zhang, Zhiguo Cao, Zhiwen Fang, Joey Tianyi Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01331](https://doi.org/10.1109/CVPR52729.2023.01331)

        **Abstract**:

        Real-time eyeblink detection in the wild can widely serve for fatigue detection, face anti-spoofing, emotion analysis, etc. The existing research efforts generally focus on single-person cases towards trimmed video. However, multi-person scenario within untrimmed videos is also important for practical applications, which has not been well concerned yet. To address this, we shed light on this research field for the first time with essential contributions on dataset, theory, and practices. In particular, a large-scale dataset termed MPEblink that involves 686 untrimmed videos with 8748 eyeblink events is proposed under multi-person conditions. The samples are captured from uncon-strainedfilms to reveal “in the wild“ characteristics. Meanwhile, a real-time multi-person eyeblink detection method is also proposed. Being different from the existing counter-parts, our proposition runs in a one-stage spatio-temporal way with end-to-end learning capacity. Specifically, it simultaneously addresses the sub-tasks of face detection, face tracking, and human instance-level eyeblink detection. This paradigm holds 2 main advantages: (1) eyeblink features can be facilitated via the face's global context (e.g., head pose and illumination condition) with joint optimization and interaction, and (2) addressing these sub-tasks in parallel instead of sequential manner can save time remarkably to meet the real-time running requirement. Experiments on MPEblink verify the essential challenges of real-time multi-person eyeblink detection in the wild for untrimmed video. Our method also outperforms existing approaches by large margins and with a high inference speed.

        ----

        ## [1320] Skinned Motion Retargeting with Residual Perception of Motion Semantics & Geometry

        **Authors**: *Jiaxu Zhang, Junwu Weng, Di Kang, Fang Zhao, Shaoli Huang, Xuefei Zhe, Linchao Bao, Ying Shan, Jue Wang, Zhigang Tu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01332](https://doi.org/10.1109/CVPR52729.2023.01332)

        **Abstract**:

        A good motion retargeting cannot be reached without reasonable consideration of source-target differences on both the skeleton and shape geometry levels. In this work, we propose a novel Residual RETargeting network (R
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
ET) structure, which relies on two neural modification modules, to adjust the source motions to fit the target skeletons and shapes progressively. In particular, a skeleton-aware module is introduced to preserve the source motion semantics. A shape-aware module is designed to perceive the geometries of target characters to reduce interpenetration and contact-missing. Driven by our explored distance-based losses that explicitly model the motion semantics and geometry, these two modules can learn residual motion modifications on the source motion to generate plausible retargeted motion in a single inference without postprocessing. To balance these two modifications, we further present a balancing gate to conduct linear interpolation between them. Extensive experiments on the public dataset Mixamo demonstrate that our R
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
ET achieves the state-of-the-art performance, and provides a good balance between the preservation of motion semantics as well as the attenuation of interpenetration and contact-missing. Code is available at https://github.com/Kebii/R2ET.

        ----

        ## [1321] MoDi: Unconditional Motion Synthesis from Diverse Data

        **Authors**: *Sigal Raab, Inbal Leibovitch, Peizhuo Li, Kfir Aberman, Olga Sorkine-Hornung, Daniel Cohen-Or*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01333](https://doi.org/10.1109/CVPR52729.2023.01333)

        **Abstract**:

        The emergence of neural networks has revolutionized the field of motion synthesis. Yet, learning to unconditionally synthesize motions from a given distribution remains challenging, especially when the motions are highly diverse. In this work, we present MoDi - a generative model trained in an unsupervised setting from an extremely diverse, unstructured and unlabeled dataset. During inference, MoDi can synthesize high-quality, diverse motions. Despite the lack of any structure in the dataset, our model yields a well-behaved and highly structured latent space, which can be semantically clustered, constituting a strong motion prior that facilitates various applications including semantic editing and crowd animation. In addition, we present an encoder that inverts real motions into MoDi's natural motion manifold, issuing solutions to various ill-posed challenges such as completion from prefix and spatial editing. Our qualitative and quantitative experiments achieve state-of-the-art results that outperform recent SOTA techniques. Code and trained models are available at https://sigal-raab.github.io/MoDi.

        ----

        ## [1322] Recurrent Vision Transformers for Object Detection with Event Cameras

        **Authors**: *Mathias Gehrig, Davide Scaramuzza*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01334](https://doi.org/10.1109/CVPR52729.2023.01334)

        **Abstract**:

        We present Recurrent Vision Transformers (RVTs), a novel backbone for object detection with event cameras. Event cameras provide visual information with submillisecond latency at a high-dynamic range and with strong robustness against motion blur. These unique properties offer great potential for low-latency object detection and tracking in time-critical scenarios. Prior work in event-based vision has achieved outstanding detection performance but at the cost of substantial inference time, typically beyond 40 milliseconds. By revisiting the high-level design of recurrent vision backbones, we reduce inference time by a factor of 6 while retaining similar performance. To achieve this, we explore a multi-stage design that utilizes three key concepts in each stage: first, a convolutional prior that can be regarded as a conditional positional embedding. Second, local and dilated global self-attention for spatial feature interaction. Third, recurrent temporal feature aggregation to minimize latency while retaining temporal information. RVTs can be trained from scratch to reach state-of-the-art performance on event-based object detection - achieving an mAP of 47.2% on the Gen1 automotive dataset. At the same time, RVTs offer fast inference (< 12 ms on a T4 GPU) and favorable parameter efficiency (5 × fewer than prior art). Our study brings new insights into effective design choices that can be fruitful for research beyond event-based vision. Code: https://github.com/uzh-rpg/RVT

        ----

        ## [1323] Continuous Intermediate Token Learning with Implicit Motion Manifold for Keyframe Based Motion Interpolation

        **Authors**: *Clinton Ansun Mo, Kun Hu, Chengjiang Long, Zhiyong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01335](https://doi.org/10.1109/CVPR52729.2023.01335)

        **Abstract**:

        Deriving sophisticated 3D motions from sparse keyframes is a particularly challenging problem, due to continuity and exceptionally skeletal precision. The action features are often derivable accurately from the full series of keyframes, and thus, leveraging the global context with transformers has been a promising data-driven embedding approach. However, existing methods are often with inputs of interpolated intermediate frame for continuity using basic interpolation methods with keyframes, which result in a trivial local minimum during training. In this paper, we propose a novel framework to formulate latent motion manifolds with keyframe-based constraints, from which the continuous nature of intermediate token representations is considered. Particularly, our proposed framework consists of two stages for identifying a latent motion subspace, i.e., a keyframe encoding stage and an intermediate token generation stage, and a subsequent motion synthesis stage to extrapolate and compose motion data from manifolds. Through our extensive experiments conducted on both the LaFAN1 and CMU Mocap datasets, our proposed method demonstrates both superior interpolation accuracy and high visual similarity to ground truth motions.

        ----

        ## [1324] EvShutter: Transforming Events for Unconstrained Rolling Shutter Correction

        **Authors**: *Julius Erbach, Stepan Tulyakov, Patricia Vitoria, Alfredo Bochicchio, Yuanyou Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01336](https://doi.org/10.1109/CVPR52729.2023.01336)

        **Abstract**:

        Widely used Rolling Shutter (RS) CMOS sensors capture high resolution images at the expense of introducing distortions and artifacts in the presence of motion. In such situations, RS distortion correction algorithms are critical. Recent methods rely on a constant velocity assumption and require multiple frames to predict the dense displacement field. In this work, we introduce a new method, called Eventful Shutter (EvShutter)
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The evaluation code and the dataset can be found here https://github.com/juliuserbach/EvShutter, that corrects RS artifacts using a single RGB image and event information with high temporal resolution. The method firstly removes blur using a novel flow-based deblurring module and then compensates RS using a double encoder hourglass network. In contrast to previous methods, it does not rely on a constant velocity assumption and uses a simple architecture thanks to an event transformation dedicated to RS, called Filter and Flip (FnF), that transforms input events to encode only the changes between GS and RS images. To evaluate the proposed method and facilitate future research, we collect the first dataset with real events and high-quality RS images with optional blur, called RS-ERGB. We generate the RS images from GS images using a newly proposed simulator based on adaptive interpolation. The simulator permits the use of inexpensive cameras with long exposure to capture high-quality GS images. We show that on this realistic dataset the proposed method outperforms the state-of-the-art image-and event-based methods by 9.16 dB and 0.75 dB respectively in terms of PSNR and an improvement of 23 % and 21 % in LPIPS.

        ----

        ## [1325] Multi Domain Learning for Motion Magnification

        **Authors**: *Jasdeep Singh, Subrahmanyam Murala, G. Sankara Raju Kosuru*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01337](https://doi.org/10.1109/CVPR52729.2023.01337)

        **Abstract**:

        Video motion magnification makes subtle invisible motions visible, such as small chest movements while breathing, subtle vibrations in the moving objects etc. But small motions are prone to noise, illumination changes, large motions, etc. making the task difficult. Most state-of-the-art methods use hand-crafted concepts which result in small magnification, ringing artifacts etc. The deep learning-based approach has higher magnification but is prone to severe artifacts in some scenarios. We propose a new phase-based deep network for video motion magnification that operates in both domains (frequency and spatial) to address this issue. It generates motion magnification from frequency domain phase fluctuations and then improves its quality in the spatial domain. The proposed models are lightweight networks with fewer parameters (∼0.11M and ∼0. 05M). Further, the proposed networks performance is compared to the SOTA approaches and evaluated on real-world and synthetic videos. Finally, an ablation study is also conducted to show the impact of different parts of the network.

        ----

        ## [1326] Learning Event Guided High Dynamic Range Video Reconstruction

        **Authors**: *Yixin Yang, Jin Han, Jinxiu Liang, Imari Sato, Boxin Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01338](https://doi.org/10.1109/CVPR52729.2023.01338)

        **Abstract**:

        Limited by the trade-off between frame rate and exposure time when capturing moving scenes with conventional cameras, frame based HDR video reconstruction suffers from scene-dependent exposure ratio balancing and ghosting artifacts. Event cameras provide an alternative visual representation with a much higher dynamic range and temporal resolution free from the above issues, which could be an effective guidance for HDR imaging from LDR videos. In this paper, we propose a multimodal learning framework for event guided HDR video reconstruction. In order to better leverage the knowledge of the same scene from the two modalities of visual signals, a multimodal representation alignment strategy to learn a shared latent space and a fusion module tailored to complementing two types of signals for different dynamic ranges in different regions are proposed. Temporal correlations are utilized recurrently to suppress the flickering effects in the reconstructed HDR video. The proposed HDRev-Net demonstrates state-of-the-art performance quantitatively and qualitatively for both synthetic and real-world data.

        ----

        ## [1327] Joint Video Multi-Frame Interpolation and Deblurring under Unknown Exposure Time

        **Authors**: *Wei Shang, Dongwei Ren, Yi Yang, Hongzhi Zhang, Kede Ma, Wangmeng Zuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01339](https://doi.org/10.1109/CVPR52729.2023.01339)

        **Abstract**:

        Natural videos captured by consumer cameras often suffer from low framerate and motion blur due to the combination of dynamic scene complexity, lens and sensor imperfection, and less than ideal exposure setting. As a result, computational methods that jointly perform video frame interpolation and deblurring begin to emerge with the unrealistic assumption that the exposure time is known and fixed. In this work, we aim ambitiously for a more realistic and challenging task - joint video multi-frame interpolation and deblurring under unknown exposure time. Toward this goal, we first adopt a variant of supervised contrastive learning to construct an exposure-aware representation from input blurred frames. We then train two U-Nets for intramotion and inter-motion analysis, respectively, adapting to the learned exposure representation via gain tuning. We finally build our video reconstruction network upon the exposure and motion representation by progressive exposureadaptive convolution and motion refinement. Extensive experiments on both simulated and real-world datasets show that our optimized method achieves notable performance gains over the state-of-the-art on the joint video ×8 interpolation and deblurring task. Moreover, on the seemingly implausible ×16 interpolation task, our method outperforms existing methods by more than 1.5 dB in terms of PSNR.

        ----

        ## [1328] FeatER: An Efficient Network for Human Reconstruction via Feature Map-Based TransformER

        **Authors**: *Ce Zheng, Matías Mendieta, Taojiannan Yang, Guo-Jun Qi, Chen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01340](https://doi.org/10.1109/CVPR52729.2023.01340)

        **Abstract**:

        Recently, vision transformers have shown great success in a set of human reconstruction tasks such as 2D/3D human pose estimation (2D/3D HPE) and human mesh reconstruction (HMR) tasks. In these tasks, feature map representations of the human structural information are often extracted first from the image by a CNN (such as HRNet), and then further processed by transformer to predict the heatmaps for HPE or HMR. However, existing transformer architectures are not able to process these feature map inputs directly, forcing an unnatural flattening of the location-sensitive human structural information. Furthermore, much of the performance benefit in recent HPE and HMR methods has come at the cost of ever-increasing computation and memory needs. Therefore, to simultaneously address these problems, we propose FeatER, a novel transformer design that preserves the inherent structure of feature map representations when modeling attention while reducing memory and computational costs. Taking advantage of FeatER, we build an efficient network for a set of human reconstruction tasks including 2D HPE, 3D HPE, and HMR. A feature map reconstruction module is applied to improve the performance of the estimated human pose and mesh. Extensive experiments demonstrate the effectiveness of FeatER on various human pose and mesh datasets. For instance, FeatER outperforms the SOTA method Mesh- Graphormer by requiring 5% of Params and 16% of MACs on Human3.6M and 3DPW datasets. The project webpage is https://zczcwh.github.io/feater_page/.

        ----

        ## [1329] MetaFusion: Infrared and Visible Image Fusion via Meta-Feature Embedding from Object Detection

        **Authors**: *Wenda Zhao, Shigeng Xie, Fan Zhao, You He, Huchuan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01341](https://doi.org/10.1109/CVPR52729.2023.01341)

        **Abstract**:

        Fusing infrared and visible images can provide more texture details for subsequent object detection task. Conversely, detection task furnishes object semantic information to improve the infrared and visible image fusion. Thus, a joint fusion and detection learning to use their mutual promotion is attracting more attention. However, the feature gap between these two different-level tasks hinders the progress. Addressing this issue, this paper proposes an infrared and visible image fusion via meta-feature embedding from object detection. The core idea is that meta-feature embedding model is designed to generate object semantic features according to fusion network ability, and thus the semantic features are naturally compatible with fusion features. It is optimized by simulating a meta learning. Moreover, we further implement a mutual promotion learning between fusion and detection tasks to improve their performances. Comprehensive experiments on three public datasets demonstrate the effectiveness of our method. Code and model are available at: https://github.com/wdzhao123/MetaFusion.

        ----

        ## [1330] Joint HDR Denoising and Fusion: A Real-World Mobile HDR Image Dataset

        **Authors**: *Shuaizheng Liu, Xindong Zhang, Lingchen Sun, Zhetong Liang, Hui Zeng, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01342](https://doi.org/10.1109/CVPR52729.2023.01342)

        **Abstract**:

        Mobile phones have become a ubiquitous and indispensable photographing device in our daily life, while the small aperture and sensor size make mobile phones more susceptible to noise and over-saturation, resulting in low dynamic range (LDR) and low image quality. It is thus crucial to develop high dynamic range (HDR) imaging techniques for mobile phones. Unfortunately, the existing HDR image datasets are mostly constructed by DSLR cameras in daytime, limiting their applicability to the study of HDR imaging for mobile phones. In this work, we develop, for the first time to our best knowledge, an HDR image dataset by using mobile phone cameras, namely Mobile-HDR dataset. Specifically, we utilize three mobile phone cameras to collect paired LDR-HDR images in the raw image domain, covering both daytime and night-time scenes with different noise levels. We then propose a transformer based model with a pyramid cross-attention alignment module to aggregate highly correlated features from different exposure frames to perform joint HDR denoising and fusion. Experiments validate the advantages of our dataset and our method on mobile HDR imaging. Dataset and codes are available at https://github.com/shuaizhengliu/Joint-HDRDN.

        ----

        ## [1331] Visibility Constrained Wide-Band Illumination Spectrum Design for Seeing-in-the-Dark

        **Authors**: *Muyao Niu, Zhuoxiao Li, Zhihang Zhong, Yinqiang Zheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01343](https://doi.org/10.1109/CVPR52729.2023.01343)

        **Abstract**:

        Seeing-in-the-dark is one of the most important and challenging computer vision tasks due to its wide applications and extreme complexities of in-the- wild scenarios. Existing arts can be mainly divided into two threads: 1) RGB-dependent methods restore information using degraded RGB inputs only (e.g., low-light enhancement), 2) RGB-independent methods translate images captured under auxiliary near-infrared (NIR) illuminants into RGB domain (e.g., NIR2RGB translation). The latter is very attractive since it works in complete darkness and the illuminants are visually friendly to naked eyes, but tends to be unstable due to its intrinsic ambiguities. In this paper, we try to robustify NIR2RGB translation by designing the optimal spectrum of auxiliary illumination in the wide-band VIS-NIR range, while keeping visual friendliness. Our core idea is to quantify the visibility constraint implied by the human vision system and incorporate it into the design pipeline. By modeling the formation process of images in the VIS-NIR range, the optimal multiplexing of a wide range of LEDs is automatically designed in a fully differentiable manner, within the feasible region defined by the visibility constraint. We also collect a substantially expanded VIS-NIR hyperspectral image dataset for experiments by using a customized 50-band filter wheel. Experimental results show that the task can be significantly improved by using the optimized wide-band illumination than using NIR only. Codes Available: https://github.com/MyNiuuu/VCSD.

        ----

        ## [1332] Self-Supervised Blind Motion Deblurring with Deep Expectation Maximization

        **Authors**: *Ji Li, Weixi Wang, Yuesong Nan, Hui Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01344](https://doi.org/10.1109/CVPR52729.2023.01344)

        **Abstract**:

        When taking a picture, any camera shake during the shutter time can result in a blurred image. Recovering a sharp image from the one blurred by camera shake is a challenging yet important problem. Most existing deep learning methods use supervised learning to train a deep neural network (DNN) on a dataset of many pairs of blurred/latent images. In contrast, this paper presents a dataset-free deep learning method for removing uniform and non-uniform blur effects from images of static scenes. Our method involves a DNN-based re-parametrization of the latent image, and we propose a Monte Carlo Expectation Maximization (MCEM) approach to train the DNN without requiring any latent images. The Monte Carlo simulation is implemented via Langevin dynamics. Experiments showed that the proposed method outperforms existing methods significantly in removing motion blur from images of static scenes.

        ----

        ## [1333] Structure Aggregation for Cross-Spectral Stereo Image Guided Denoising

        **Authors**: *Zehua Sheng, Zhu Yu, Xiongwei Liu, Si-Yuan Cao, Yu-Qi Liu, Hui-Liang Shen, Huaqi Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01345](https://doi.org/10.1109/CVPR52729.2023.01345)

        **Abstract**:

        To obtain clean images with salient structures from noisy observations, a growing trend in current denoising studies is to seek the help of additional guidance images with high signal-to-noise ratios, which are often acquired in different spectral bands such as near infrared. Although previous guided denoising methods basically require the input images to be well-aligned, a more common way to capture the paired noisy target and guidance images is to exploit a stereo camera system. However, current studies on cross-spectral stereo matching cannot fully guarantee the pixel-level registration accuracy, and rarely consider the case of noise contamination. In this work, for the first time, we propose a guided denoising framework for cross-spectral stereo images. Instead of aligning the input images via conventional stereo matching, we aggregate structures from the guidance image to estimate a clean structure map for the noisy target image, which is then used to regress the final denoising result with a spatially variant linear representation model. Based on this, we design a neural network, called as SANet, to complete the entire guided denoising process. Experimental results show that, our SANet can effectively transfer structures from an unaligned guidance image to the restoration result, and outperforms state-of-the-art denoisers on various stereo image datasets. Besides, our structure aggregation strategy also shows its potential to handle other unaligned guided restoration tasks such as super-resolution and deblurring. The source code is available at https://github.com/lustrouselixir/SANet.

        ----

        ## [1334] Rawgment: Noise-Accounted RAW Augmentation Enables Recognition in a Wide Variety of Environments

        **Authors**: *Masakazu Yoshimura, Junji Otsuka, Atsushi Irie, Takeshi Ohashi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01346](https://doi.org/10.1109/CVPR52729.2023.01346)

        **Abstract**:

        Image recognition models that work in challenging environments (e.g., extremely dark, blurry, or high dynamic range conditions) must be useful. However, creating training datasets for such environments is expensive and hard due to the difficulties of data collection and annotation. It is desirable if we could get a robust model without the need for hard-to-obtain datasets. One simple approach is to apply data augmentation such as color jitter and blur to standard RGB (sRGB) images in simple scenes. Unfortunately, this approach struggles to yield realistic images in terms of pixel intensity and noise distribution due to not considering the non-linearity of Image Signal Processors (ISPs) and noise characteristics of image sensors. Instead, we propose a noise-accounted RAW image augmentation method. In essence, color jitter and blur augmentation are applied to a RAW image before applying non-linear ISP, resulting in realistic intensity. Furthermore, we introduce a noise amount alignment method that calibrates the domain gap in the noise property caused by the augmentation. We show that our proposed noise-accounted RAW augmentation method doubles the image recognition accuracy in challenging environments only with simple training data.

        ----

        ## [1335] Zero-Shot Noise2Noise: Efficient Image Denoising without any Data

        **Authors**: *Youssef Mansour, Reinhard Heckel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01347](https://doi.org/10.1109/CVPR52729.2023.01347)

        **Abstract**:

        Recently, self-supervised neural networks have shown excellent image denoising performance. How-ever, current dataset free methods are either computationally expensive, require a noise model, or have inad-equate image quality. In this work we show that a simple 2-layer network, without any training data or knowledge of the noise distribution, can enable high-quality image denoising at low computational cost. Our approach is motivated by Noise2Noise and Neighbor2Neighbor and works well for denoising pixel-wise independent noise. Our experiments on artificial, real-world cam-era, and microscope noise show that our method termed ZS-N2N (Zero Shot Noise2Noise) often outperforms ex-isting dataset-free methods at a reduced cost, making it suitable for use cases with scarce data availability and limited compute.

        ----

        ## [1336] Real-Time Controllable Denoising for Image and Video

        **Authors**: *Zhaoyang Zhang, Yitong Jiang, Wenqi Shao, Xiaogang Wang, Ping Luo, Kaimo Lin, Jinwei Gu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01348](https://doi.org/10.1109/CVPR52729.2023.01348)

        **Abstract**:

        Controllable image denoising aims to generate clean samples with human perceptual priors and balance sharpness and smoothness. In traditional filter-based denoising methods, this can be easily achieved by adjusting the filtering strength. However, for NN (Neural Network)-based models, adjusting the final denoising strength requires performing network inference each time, making it almost impossible for real-time user interaction. In this paper, we introduce Real-time Controllable Denoising (RCD), the first deep image and video denoising pipeline that provides a fully controllable user interface to edit arbitrary denoising levels in real-time with only one-time network inference. Unlike existing controllable denoising methods that require multiple denoisers and training stages, RCD replaces the last output layer (which usually outputs a single noise map) of an existing CNN-based model with a lightweight module that outputs multiple noise maps. We propose a novel Noise Decorrelation process to enforce the orthogonality of the noise feature maps, allowing arbitrary noise level control through noise map interpolation. This process is network-free and does not require network inference. Our experiments show that RCD can enable real-time editable image and video denoising for various existing heavy-weight models without sacrificing their original performance.

        ----

        ## [1337] Probability-based Global Cross-modal Upsampling for Pansharpening

        **Authors**: *Zeyu Zhu, Xiangyong Cao, Man Zhou, Junhao Huang, Deyu Meng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01349](https://doi.org/10.1109/CVPR52729.2023.01349)

        **Abstract**:

        Pansharpening is an essential preprocessing step for remote sensing image processing. Although deep learning (DL) approaches performed well on this task, current upsampling methods used in these approaches only utilize the local information of each pixel in the low-resolution multispectral (LRMS) image while neglecting to exploit its global information as well as the cross-modal information of the guiding panchromatic (PAN) image, which limits their performance improvement. To address this issue, this paper develops a novel probability-based global cross-modal upsampling (PGCU) method for pan-sharpening. Precisely, we first formulate the PGCU method from a probabilistic perspective and then design an efficient network module to implement it by fully utilizing the information mentioned above while simultaneously considering the channel specificity. The PGCU module consists of three blocks, i.e., information extraction (IE), distribution and expectation estimation (DEE), and fine adjustment (FA). Extensive experiments verify the superiority of the PGCU method compared with other popular upsampling methods. Additionally, experiments also show that the PGCU module can help improve the performance of existing SOTA deep learning pansharpening methods. The codes are available at https://github.com/Zeyu-Zhu/PGCU.

        ----

        ## [1338] ShadowDiffusion: When Degradation Prior Meets Diffusion Model for Shadow Removal

        **Authors**: *Lanqing Guo, Chong Wang, Wenhan Yang, Siyu Huang, Yufei Wang, Hanspeter Pfister, Bihan Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01350](https://doi.org/10.1109/CVPR52729.2023.01350)

        **Abstract**:

        Recent deep learning methods have achieved promising results in image shadow removal. However, their restored images still suffer from unsatisfactory boundary artifacts, due to the lack of degradation prior embedding and the deficiency in modeling capacity. Our work addresses these issues by proposing a unified diffusion framework that integrates both the image and degradation priors for highly effective shadow removal. In detail, we first propose a shadow degradation model, which inspires us to build a novel unrolling diffusion model, dubbed ShandowDiffusion. It remarkably improves the model's capacity in shadow removal via progressively refining the desired output with both degradation prior and diffusive generative prior, which by nature can serve as a new strong baseline for image restoration. Furthermore, ShadowDiffusion progressively refines the estimated shadow mask as an auxiliary task of the diffusion generator, which leads to more accurate and robust shadow-free image generation. We conduct extensive experiments on three popular public datasets, including ISTD, ISTD+, and SRD, to validate our method's effectiveness. Compared to the state-of-the-art methods, our model achieves a significant improvement in terms of PSNR, increasing from 31.69dB to 34. 73dB over SRD dataset. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/GuoLanqing/ShadowDiffusion

        ----

        ## [1339] Visual Recognition-Driven Image Restoration for Multiple Degradation with Intrinsic Semantics Recovery

        **Authors**: *Zizheng Yang, Jie Huang, Jiahao Chang, Man Zhou, Hu Yu, Jinghao Zhang, Feng Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01351](https://doi.org/10.1109/CVPR52729.2023.01351)

        **Abstract**:

        Deep image recognition models suffer a significant performance drop when applied to low-quality images since they are trained on high-quality images. Although many studies have investigated to solve the issue through image restoration or domain adaptation, the former focuses on visual quality rather than recognition quality, while the latter requires semantic annotations for task-specific training. In this paper, to address more practical scenarios, we propose a Visual Recognition-Driven Image Restoration network for multiple degradation, dubbed VRD-IR, to recover high-quality images from various unknown corruption types from the perspective of visual recognition within one model. Concretely, we harmonize the semantic representations of diverse degraded images into a unified space in a dynamic manner, and then optimize them towards intrinsic semantics recovery. Moreover, a prior-ascribing optimization strategy is introduced to encourage VRD-IR to couple with various downstream recognition tasks better. Our VRD-IR is corruption- and recognition-agnostic, and can be inserted into various recognition tasks directly as an image enhancement module. Extensive experiments on multiple image distortions demonstrate that our VRD-IR surpasses existing image restoration methods and show superior performance on diverse high-level tasks, including classification, detection, and person re-identification.

        ----

        ## [1340] Blind Image Quality Assessment via Vision-Language Correspondence: A Multitask Learning Perspective

        **Authors**: *Weixia Zhang, Guangtao Zhai, Ying Wei, Xiaokang Yang, Kede Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01352](https://doi.org/10.1109/CVPR52729.2023.01352)

        **Abstract**:

        We aim at advancing blind image quality assessment (BIQA), which predicts the human perception of image quality without any reference information. We develop a general and automated multitask learning scheme for BIQA to exploit auxiliary knowledge from other tasks, in a way that the model parameter sharing and the loss weighting are determined automatically. Specifically, we first describe all candidate label combinations (from multiple tasks) using a textual template, and compute the joint probability from the cosine similarities of the visual-textual embeddings. Predictions of each task can be inferred from the joint distribution, and optimized by carefully designed loss functions. Through comprehensive experiments on learning three tasks - BIQA, scene classification, and distortion type identification, we verify that the proposed BIQA method 1) benefits from the scene classification and distortion type identification tasks and outperforms the state-of-the-art on multiple IQA datasets, 2) is more robust in the group maximum differentiation competition, and 3) realigns the quality annotations from different IQA datasets more effectively. The source code is available at https://github.com/zwx8981/LIQE.

        ----

        ## [1341] Human Guided Ground-Truth Generation for Realistic Image Super-Resolution

        **Authors**: *Du Chen, Jie Liang, Xindong Zhang, Ming Liu, Hui Zeng, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01353](https://doi.org/10.1109/CVPR52729.2023.01353)

        **Abstract**:

        How to generate the ground-truth (GT) image is a critical issue for training realistic image super-resolution (Real-ISR) models. Existing methods mostly take a set of high-resolution (HR) images as GTs and apply various degradations to simulate their low-resolution (LR) counterparts. Though great progress has been achieved, such an LR-HR pair generation scheme has several limitations. First, the perceptual quality of HR images may not be high enough, limiting the quality of Real-ISR outputs. Second, existing schemes do not consider much human perception in GT generation, and the trained models tend to produce over-smoothed results or unpleasant artifacts. With the above considerations, we propose a human guided GT generation scheme. We first elaborately train multiple image enhancement models to improve the perceptual quality of HR images, and enable one LR image having multiple HR counterparts. Human subjects are then involved to annotate the high quality regions among the enhanced HR images as GTs, and label the regions with unpleasant artifacts as negative samples. A human guided GT image dataset with both positive and negative samples is then constructed, and a loss function is proposed to train the Real-ISR models. Experiments show that the Real-ISR models trained on our dataset can produce perceptually more realistic results with less artifacts. Dataset and codes can be found at https://github.com/ChrisDud0257/HGGT

        ----

        ## [1342] Real-time 6K Image Rescaling with Rate-distortion Optimization

        **Authors**: *Chenyang Qi, Xin Yang, Ka Leong Cheng, Ying-Cong Chen, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01354](https://doi.org/10.1109/CVPR52729.2023.01354)

        **Abstract**:

        Contemporary image rescaling aims at embedding a high-resolution (HR) image into a low-resolution (LR) thumbnail image that contains embedded information for HR image reconstruction. Unlike traditional image super-resolution, this enables high-fidelity HR image restoration faithful to the original one, given the embedded information in the LR thumbnail. However, state-of-the-art image rescaling methods do not optimize the LR image file size for efficient sharing and fall short of real-time performance for ultra-high-resolution (e.g., 6K) image reconstruction. To address these two challenges, we propose a novel frame-work (HyperThumbnail) for real-time 6K rate-distortion-aware image rescaling. Our framework first embeds an HR image into a JPEG LR thumbnail by an encoder with our proposed quantization prediction module, which minimizes the file size of the embedding LR JPEG thumbnail while maximizing HR reconstruction quality. Then, an efficient frequency-aware decoder reconstructs a high-fidelity HR image from the LR one in real time. Extensive experiments demonstrate that our framework outperforms previous image rescaling baselines in rate-distortion performance and can perform 6K image reconstruction in real time.

        ----

        ## [1343] Equivalent Transformation and Dual Stream Network Construction for Mobile Image Super-Resolution

        **Authors**: *Jiahao Chao, Zhou Zhou, Hongfan Gao, Jiali Gong, Zhengfeng Yang, Zhenbing Zeng, Lydia Dehbi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01355](https://doi.org/10.1109/CVPR52729.2023.01355)

        **Abstract**:

        In recent years, there has been an increasing demand for real-time super-resolution networks on mobile devices. To address this issue, many lightweight super-resolution models have been proposed. However, these models still contain time-consuming components that increase inference latency, limiting their real-world applications on mobile devices. In this paper, we propose a novel model for single-image super-resolution based on Equivalent Transformation and Dual Stream network construction (ETDS). ET method is proposed to transform time-consuming operators into time-friendly operations, such as convolution and ReLU, on mobile devices. Then, a dual stream network is designed to alleviate redundant parameters resulting from the use of ET and enhance the feature extraction ability. Taking full advantage of the advance of ET and the dual stream network structure, we develop the efficient SR model ETDS for mobile devices. The experimental results demonstrate that our ETDS achieves superior inference speed and reconstruction quality compared to previous lightweight SR methods on mobile devices. The code is available at https://github.com/ECNUSR/ETDS.

        ----

        ## [1344] Ultrahigh Resolution Image/Video Matting with Spatio-Temporal Sparsity

        **Authors**: *Yanan Sun, Chi-Keung Tang, Yu-Wing Tai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01356](https://doi.org/10.1109/CVPR52729.2023.01356)

        **Abstract**:

        Commodity ultrahigh definition (UHD) displays are becoming more affordable which demand imaging in ultrahigh resolution (UHR). This paper proposes SparseMat, a computationally efficient approach for UHR image/video matting. Note that it is infeasible to directly process UHR images at full resolution in one shot using existing matting algorithms without running out of memory on consumer-level computational platforms, e.g., Nvidia 1080Ti with 11G memory, while patch-based approaches can introduce unsightly artifacts due to patch partitioning. Instead, our method resorts to spatial and temporal sparsity for addressing general UHR matting. When processing videos, huge computation redundancy can be reduced by exploiting spatial and temporal sparsity. In this paper, we show how to effectively detect spatio-temporal sparsity, which serves as a gate to activate input pixels for the matting model. Under the guidance of such sparsity, our method with sparse high-resolution module (SHM) can avoid patch-based inference while memory efficient for full-resolution matte refinement. Extensive experiments demonstrate that SparseMat can effectively and efficiently generate high-quality alpha matte for UHR images and videos at the original high resolution in a single pass. Project page is in https://github.com/nowsyn/SparseMat.git.

        ----

        ## [1345] Comprehensive and Delicate: An Efficient Transformer for Image Restoration

        **Authors**: *Haiyu Zhao, Yuanbiao Gou, Boyun Li, Dezhong Peng, Jiancheng Lv, Xi Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01357](https://doi.org/10.1109/CVPR52729.2023.01357)

        **Abstract**:

        Vision Transformers have shown promising performance in image restoration, which usually conduct window- or channel-based attention to avoid intensive computations. Although the promising performance has been achieved, they go against the biggest success factor of Transformers to a certain extent by capturing the local instead of global de-pendency among pixels. In this paper, we propose a novel efficient image restoration Transformer that first captures the superpixel-wise global dependency, and then transfers it into each pixel. Such a coarse-to-fine paradigm is implemented through two neural blocks, i.e., condensed attention neural block (CA) and dual adaptive neural block (DA). In brief, CA employs feature aggregation, attention computation, and feature recovery to efficiently capture the global dependency at the superpixel level. To embrace the pixel-wise global dependency, DA takes a novel dual-way structure to adaptively encapsulate the globality from superpix-els into pixels. Thanks to the two neural blocks, our method achieves comparable performance while taking only ~6% FLOPs compared with SwinIR.

        ----

        ## [1346] PHA: Patch-Wise High-Frequency Augmentation for Transformer-Based Person Re-Identification

        **Authors**: *Guiwei Zhang, Yongfei Zhang, Tianyu Zhang, Bo Li, Shiliang Pu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01358](https://doi.org/10.1109/CVPR52729.2023.01358)

        **Abstract**:

        Although recent studies empirically show that injecting Convolutional Neural Networks (CNNs) into Vision Transformers (ViTs) can improve the performance of person reidentification, the rationale behind it remains elusive. From a frequency perspective, we reveal that ViTs perform worse than CNNs in preserving key high-frequency components (e.g, clothes texture details) since high-frequency components are inevitably diluted by low-frequency ones due to the intrinsic Self-Attention within ViTs. To remedy such inadequacy of the ViT, we propose a Patch-wise High-frequency Augmentation (PHA) method with two core designs. First, to enhance the feature representation ability of high-frequency components, we split patches with high-frequency components by the Discrete Haar Wavelet Transform, then empower the ViT to take the split patches as auxiliary input. Second, to prevent high-frequency components from being diluted by low-frequency ones when taking the entire sequence as input during network optimization, we propose a novel patch-wise contrastive loss. From the view of gradient optimization, it acts as an implicit augmentation to improve the representation ability of key high-frequency components. This benefits the ViT to capture key high-frequency components to extract discriminative person representations. PHA is necessary during training and can be removed during inference, without bringing extra complexity. Extensive experiments on widely-used ReID datasets validate the effectiveness of our method.

        ----

        ## [1347] PyramidFlow: High-Resolution Defect Contrastive Localization Using Pyramid Normalizing Flow

        **Authors**: *Jiarui Lei, Xiaobo Hu, Yue Wang, Dong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01359](https://doi.org/10.1109/CVPR52729.2023.01359)

        **Abstract**:

        During industrial processing, unforeseen defects may arise in products due to uncontrollable factors. Although unsupervised methods have been successful in defect localization, the usual use of pre-trained models results in lowresolution outputs, which damages visual performance. To address this issue, we propose PyramidFlow, the first fully normalizing flow method without pre-trained models that enables high-resolution defect localization. Specifically, we propose a latent template-based defect contrastive localization paradigm to reduce intra-class variance, as the pre-trained models do. In addition, PyramidFlow utilizes pyramid-like normalizing flows for multi-scale fusing and volume normalization to help generalization. Our comprehensive studies on MVTecAD demonstrate the proposed method outperforms the comparable algorithms that do not use external priors, even achieving state-of-the-art performance in more challenging BTAD scenarios.

        ----

        ## [1348] Neural Fourier Filter Bank

        **Authors**: *Zhijie Wu, Yuhe Jin, Kwang Moo Yi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01360](https://doi.org/10.1109/CVPR52729.2023.01360)

        **Abstract**:

        We present a novel method to provide efficient and highly detailed reconstructions. Inspired by wavelets, we learn a neural field that decompose the signal both spatially and frequency-wise. We follow the recent grid-based paradigm for spatial decomposition, but unlike existing work, encourage specific frequencies to be stored in each grid via Fourier features encodings. We then apply a multi-layer perceptron with sine activations, taking these Fourier encoded features in at appropriate layers so that higher-frequency components are accumulated on top of lower-frequency components sequentially, which we sum up to form the final output. We demonstrate that our method outperforms the state of the art regarding model compactness and convergence speed on multiple tasks: 2D image fitting, 3D shape reconstruction, and neural radiance fields. Our code is available at https://github.com/ubc-vision/NFFB.

        ----

        ## [1349] Restoration of Hand-Drawn Architectural Drawings using Latent Space Mapping with Degradation Generator

        **Authors**: *Nakkwan Choi, Seungjae Lee, Yongsik Lee, Seungjoon Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01361](https://doi.org/10.1109/CVPR52729.2023.01361)

        **Abstract**:

        This work presents the restoration of drawings of wooden built heritage. Hand-drawn drawings contain the most important original information but are often severely degraded over time. A novel restoration method based on the vector quantized variational autoencoders is presented. Latent space representations of drawings and noise are learned, which are used to map noisy drawings to clean drawings for restoration and to generate authentic noisy drawings for data augmentation. The proposed method is applied to the drawings archived in the Cultural Heritage Administration. Restored drawings show significant quality improvement and allow more accurate interpretations of information.

        ----

        ## [1350] Neural Preset for Color Style Transfer

        **Authors**: *Zhanghan Ke, Yuhao Liu, Lei Zhu, Nanxuan Zhao, Rynson W. H. Lau*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01362](https://doi.org/10.1109/CVPR52729.2023.01362)

        **Abstract**:

        In this paper, we present a Neural Preset technique to address the limitations of existing color style transfer methods, including visual artifacts, vast memory requirement, and slow style switching speed. Our method is based on two core designs. First, we propose Deterministic Neural Color Mapping (DNCM) to consistently operate on each pixel via an image-adaptive color mapping matrix, avoiding artifacts and supporting high-resolution inputs with a small memory footprint. Second, we develop a two-stage pipeline by dividing the task into color normalization and stylization, which allows efficient style switching by extracting color styles as presets and reusing them on normalized input images. Due to the unavailability of pairwise datasets, we describe how to train Neural Preset via a self-supervised strategy. Various advantages of Neural Preset over existing methods are demonstrated through comprehensive evaluations. Besides, we show that our trained model can naturally support multiple applications without fine-tuning, including low-light image enhancement, underwater image correction, image dehazing, and image harmonization. The project page is: https://ZHKKKe.github.io/NeuralPreset.

        ----

        ## [1351] NÜWA-LIP: Language-guided Image Inpainting with Defect-free VQGAN

        **Authors**: *Minheng Ni, Xiaoming Li, Wangmeng Zuo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01363](https://doi.org/10.1109/CVPR52729.2023.01363)

        **Abstract**:

        Language-guided image inpainting aims to fill the defective regions of an image under the guidance of text while keeping the non-defective regions unchanged. However, directly encoding the defective images is prone to have an adverse effect on the non-defective regions, giving rise to distorted structures on non-defective parts. To better adapt the text guidance to the inpainting task, this paper proposes NÜWA-LIP, which involves defect-free VQGAN (DF-VQGAN) and a multi-perspective sequence-to-sequence module (MP-S2S). To be specific, DF-VQGAN introduces relative estimation to carefully control the receptive spreading, as well as symmetrical connections to protect structure details unchanged. For harmoniously embedding text guidance into the locally defective regions, MP-S2S is employed by aggregating the complementary perspectives from low-level pixels, high-level tokens as well as the text description. Experiments show that our DF-VQGAN effectively aids the inpainting process while avoiding unexpected changes in non-defective regions. Results on three open-domain benchmarks demonstrate the superior performance of our method against state-of-the-arts. Our code, datasets, and model will be made publicly available
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/kodenii/NUWA-LIP.

        ----

        ## [1352] DualVector: Unsupervised Vector Font Synthesis with Dual-Part Representation

        **Authors**: *Ying-Tian Liu, Zhifei Zhang, Yuan-Chen Guo, Matthew Fisher, Zhaowen Wang, Song-Hai Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01364](https://doi.org/10.1109/CVPR52729.2023.01364)

        **Abstract**:

        Automatic generation of fonts can be an important aid to typeface design. Many current approaches regard glyphs as pixelated images, which present artifacts when scaling and inevitable quality losses after vectorization. On the other hand, existing vector font synthesis methods either fail to represent the shape concisely or require vector supervision during training. To push the quality of vector font synthesis to the next level, we propose a novel dual-part representation for vector glyphs, where each glyph is modeled as a collection of closed “positive” and “negative” path pairs. The glyph contour is then obtained by boolean operations on these paths. We first learn such a representation only from glyph images and devise a subsequent contour refinement step to align the contour with an image representation to further enhance details. Our method, named DualVector, outperforms state-of-the-art methods in vector font synthesis both quantitatively and qualitatively. Our synthesized vector fonts can be easily converted to common digital font formats like TrueType Font for practical use. The code is released at https://github.com/thuliu-yt16/dualvector.

        ----

        ## [1353] DATID-3D: Diversity-Preserved Domain Adaptation Using Text-to-Image Diffusion for 3D Generative Model

        **Authors**: *Gwanghyun Kim, Se Young Chun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01365](https://doi.org/10.1109/CVPR52729.2023.01365)

        **Abstract**:

        Recent 3D generative models have achieved remarkable performance in synthesizing high resolution photorealistic images with view consistency and detailed 3D shapes, but training them for diverse domains is challenging since it requires massive training images and their camera distribution information. Text-guided domain adaptation methods have shown impressive performance on converting the 2D generative model on one domain into the models on other domains with different styles by leveraging the CLIP (Contrastive Language-Image Pre-training), rather than collecting massive datasets for those domains. However, one drawback of them is that the sample diversity in the original generative model is not well-preserved in the domain-adapted generative models due to the deterministic nature of the CLIP text encoder. Text-guided domain adaptation will be even more challenging for 3D generative models not only because of catastrophic diversity loss, but also because of inferior text-image correspondence and poor image quality. Here we propose DATID-3D, a domain adaptation method tailored for 3D generative models using text-to-image diffusion models that can synthesize diverse images per text prompt without collecting additional images and camera information for the target domain. Unlike 3D extensions of prior text-guided domain adaptation methods, our novel pipeline was able to fine-tune the state-of-the-art 3D generator of the source domain to synthesize high resolution, multi-view consistent images in text-guided targeted domains without additional data, outperforming the existing text-guided domain adaptation methods in diversity and text-image correspondence. Furthermore, we propose and demonstrate diverse 3D image manipulations such as one-shot instance-selected adaptation and single-view manipulated 3D reconstruction to fully enjoy diversity in text.

        ----

        ## [1354] GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis

        **Authors**: *Ming Tao, Bing-Kun Bao, Hao Tang, Changsheng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01366](https://doi.org/10.1109/CVPR52729.2023.01366)

        **Abstract**:

        Synthesizing high-fidelity complex images from text is challenging. Based on large pretraining, the autoregressive and diffusion models can synthesize photo-realistic images. Although these large models have shown notable progress, there remain three flaws. 1) These models require tremendous training data and parameters to achieve good performance. 2) The multi-step generation design slows the image synthesis process heavily. 3) The synthesized visual features are challenging to control and require delicately designed prompts. To enable high-quality, efficient, fast, and controllable text-to-image synthesis, we propose Generative Adversarial CLIPs, namely GALIP. GALIP leverages the powerful pretrained CLIP model both in the discriminator and generator. Specifically, we propose a CLIP-based discriminator. The complex scene understanding ability of CLIP enables the discriminator to accurately assess the image quality. Furthermore, we propose a CLIP-empowered generator that induces the visual concepts from CLIP through bridge features and prompts. The CLIP-integrated generator and discriminator boost training efficiency, and as a result, our model only requires about 3% training data and 6% learnable parameters, achieving comparable results to large pretrained autoregressive and diffusion models. Moreover, our model achieves ~120×faster synthesis speed and inherits the smooth latent space from GAN. The extensive experimental results demonstrate the excellent performance of our GALIP. Code is available at https://github.com/tobran/GALIP.

        ----

        ## [1355] Fix the Noise: Disentangling Source Feature for Controllable Domain Translation

        **Authors**: *Dongyeun Lee, Jae Young Lee, Doyeon Kim, Jaehyun Choi, Jaejun Yoo, Junmo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01367](https://doi.org/10.1109/CVPR52729.2023.01367)

        **Abstract**:

        Recent studies show strong generative performance in domain translation especially by using transfer learning techniques on the unconditional generator. However, the control between different domain features using a single model is still challenging. Existing methods often require additional models, which is computationally demanding and leads to unsatisfactory visual quality. In addition, they have restricted control steps, which prevents a smooth transition. In this paper, we propose a new approach for high-quality domain translation with better controllability. The key idea is to preserve source features within a disentangled subspace of a target feature space. This allows our method to smoothly control the degree to which it preserves source features while generating images from an entirely new domain using only a single model. Our extensive experiments show that the proposed method can produce more consistent and realistic images than previous works and maintain precise controllability over different levels of transformation. The code is available at LeeDongYeun/FixNoise.

        ----

        ## [1356] Conditional Text Image Generation with Diffusion Models

        **Authors**: *Yuanzhi Zhu, Zhaohai Li, Tianwei Wang, Mengchao He, Cong Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01368](https://doi.org/10.1109/CVPR52729.2023.01368)

        **Abstract**:

        Current text recognition systems, including those for handwritten scripts and scene text, have relied heavily on image synthesis and augmentation, since it is difficult to realize real-world complexity and diversity through collecting and annotating enough real text images. In this paper, we explore the problem of text image generation, by taking advantage of the powerful abilities of Diffusion Models in generating photo-realistic and diverse image samples with given conditions, and propose a method called Conditional Text Image Generation with Diffusion Models (CTIG-DM for short). To conform to the characteristics of text images, we devise three conditions: image condition, text condition, and style condition, which can be used to control the attributes, contents, and styles of the samples in the image generation process. Specifically, four text image generation modes, namely: (1) synthesis mode, (2) augmentation mode, (3) recovery mode, and (4) imitation mode, can be derived by combining and configuring these three conditions. Extensive experiments on both handwritten and scene text demonstrate that the proposed CTIG-DM is able to produce image samples that simulate real-world complexity and diversity, and thus can boost the performance of existing text recognizers. Besides, CTIG-DM shows its appealing potential in domain adaptation and generating images containing Out-Of-Vocabulary (OOV) words.

        ----

        ## [1357] ReCo: Region-Controlled Text-to-Image Generation

        **Authors**: *Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, Lijuan Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01369](https://doi.org/10.1109/CVPR52729.2023.01369)

        **Abstract**:

        Recently, large-scale text-to-image (T2I) models have shown impressive performance in generating high-fidelity images, but with limited controllability, e.g., precisely specifying the content in a specific region with a free-form text description. In this paper, we propose an effective technique for such regional control in T2I generation. We augment T2I models' inputs with an extra set of position tokens, which represent the quantized spatial coordinates. Each region is specified by four position tokens to represent the top-left and bottom-right corners, followed by an open-ended natural language regional description. Then, we fine-tune a pre-trained T2I model with such new input interface. Our model, dubbed as ReCo (Region-Controlled T2I), enables the region control for arbitrary objects described by open-ended regional texts rather than by object labels from a constrained category set. Empirically, ReCo achieves better image quality than the T2I model strengthened by positional words (FID: 8.82 → 7.36, SceneFID: 15.54 → 6.51 on COCO), together with objects being more accurately placed, amounting to a 20.40% region classification accuracy improvement on COCO. Furthermore, we demonstrate that ReCo can better control the object count, spatial relationship, and region attributes such as color/size, with the free-form regional description. Human evaluation on PaintSkill shows that ReCo is +19.28% and +17.21% more accurate in generating images with correct object count and spatial relationship than the T2I model. Code is available at https://github.com/microsoft/Reeo.

        ----

        ## [1358] Freestyle Layout-to-Image Synthesis

        **Authors**: *Han Xue, Zhiwu Huang, Qianru Sun, Li Song, Wenjun Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01370](https://doi.org/10.1109/CVPR52729.2023.01370)

        **Abstract**:

        Typical layout-to-image synthesis (LIS) models generate images for a closed set of semantic classes, e.g., 182 common objects in COCO-Stuff. In this work, we explore the freestyle capability of the model, i.e., how far can it generate unseen semantics (e.g., classes, attributes, and styles) onto a given layout, and call the task Freestyle LIS (FLIS). Thanks to the development of large-scale pre-trained language-image models, a number of discriminative models (e.g., image classification and object detection) trained on limited base classes are empowered with the ability of unseen class prediction. Inspired by this, we opt to leverage large-scale pre-trained text-to-image diffusion models to achieve the generation of unseen semantics. The key challenge of FLIS is how to enable the diffusion model to synthesize images from a specific layout which very likely violates its pre-learned knowledge, e.g., the model never sees “a unicorn sitting on a bench” during its pre-training. To this end, we introduce a new module called Rectified Cross-Attention (RCA) that can be conveniently plugged in the diffusion model to integrate semantic masks. This “plug-in” is applied in each cross-attention layer of the model to rectify the attention maps between image and text tokens. The key idea of RCA is to enforce each text token to act on the pixels in a specified region, allowing us to freely put a wide variety of semantics from pre-trained knowledge (which is general) onto the given layout (which is specific). Extensive experiments show that the proposed diffusion network produces realistic and freestyle layout-to-image generation results with diverse text inputs, which has a high potential to spawn a bunch of interesting applications. Code is available at https://github.com/essunny310/FreestyleNet.

        ----

        ## [1359] Specialist Diffusion: Plug-and-Play Sample-Efficient Fine-Tuning of Text-to-Image Diffusion Models to Learn Any Unseen Style

        **Authors**: *Haoming Lu, Hazarapet Tunanyan, Kai Wang, Shant Navasardyan, Zhangyang Wang, Humphrey Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01371](https://doi.org/10.1109/CVPR52729.2023.01371)

        **Abstract**:

        Diffusion models have demonstrated impressive capability of text-conditioned image synthesis, and broader application horizons are emerging by personalizing those pretrained diffusion models toward generating some specialized target object or style. In this paper, we aim to learn an unseen style by simply fine-tuning a pre-trained diffusion model with a handful of images (e.g., less than 10), so that the fine-tuned model can generate high-quality images of arbitrary objects in this style. Such extremely lowshot fine-tuning is accomplished by a novel toolkit of finetuning techniques, including text-to-image customized data augmentations, a content loss to facilitate content-style disentanglement, and sparse updating that focuses on only a few time steps. Our framework, dubbed Specialist Diffusion, is plug-and-play to existing diffusion model backbones and other personalization techniques. We demonstrate it to outperform the latest few-shot personalization alternatives of diffusion models such as Textual Inversion [7] and DreamBooth [24], in terms of learning highly sophisticated styles with ultra-sample-efficient tuning. We further show that Specialist Diffusion can be integrated on top of textual inversion to boost performance further, even on highly unusual styles. Our codes are available at: https://github.com/Picsart-AI-Research/Specialist-Diffusion.

        ----

        ## [1360] Toward Verifiable and Reproducible Human Evaluation for Text-to-Image Generation

        **Authors**: *Mayu Otani, Riku Togashi, Yu Sawai, Ryosuke Ishigami, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, Shin'ichi Satoh*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01372](https://doi.org/10.1109/CVPR52729.2023.01372)

        **Abstract**:

        Human evaluation is critical for validating the performance of text-to-image generative models, as this highly cognitive process requires deep comprehension of text and images. However, our survey of 37 recent papers reveals that many works rely solely on automatic measures (e.g., FID) or perform poorly described human evaluations that are not reliable or repeatable. This paper proposes a standardized and well-defined human evaluation protocol to facilitate verifiable and reproducible human evaluation in future works. In our pilot data collection, we experimentally show that the current automatic measures are incompatible with human perception in evaluating the performance of the text-to-image generation results. Furthermore, we provide insights for designing human evaluation experiments reliably and conclusively. Finally, we make several resources publicly available to the community to facilitate easy and fast implementations.

        ----

        ## [1361] Towards Flexible Multi-modal Document Models

        **Authors**: *Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, Kota Yamaguchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01373](https://doi.org/10.1109/CVPR52729.2023.01373)

        **Abstract**:

        Creative workflows for generating graphical documents involve complex inter-related tasks, such as aligning elements, choosing appropriate fonts, or employing aesthetically harmonious colors. In this work, we attempt at building a holistic model that can jointly solve many different design tasks. Our model, which we denote by FlexDM, treats vector graphic documents as a set of multi-modal elements, and learns to predict masked fields such as element type, position, styling attributes, image, or text, using a unified architecture. Through the use of explicit multitask learning and in-domain pre-training, our model can better capture the multi-modal relationships among the different document fields. Experimental results corroborate that our single FlexDM is able to successfully solve a multitude of different design tasks, while achieving performance that is competitive with task-specific and costly baselines.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Please find the code and models at: https://cyberagentailab.github.io/flex-dm

        ----

        ## [1362] On Distillation of Guided Diffusion Models

        **Authors**: *Chenlin Meng, Robin Rombach, Ruiqi Gao, Diederik P. Kingma, Stefano Ermon, Jonathan Ho, Tim Salimans*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01374](https://doi.org/10.1109/CVPR52729.2023.01374)

        **Abstract**:

        Classifier-free guided diffusion models have recently been shown to be highly effective at high-resolution image generation, and they have been widely used in large-scale diffusion frameworks including DALL.E 2, Stable Diffusion and Imagen. However, a downside of classifier-free guided diffusion models is that they are computationally expensive at inference time since they require evaluating two diffusion models, a class-conditional model and an unconditional model, tens to hundreds of times. To deal with this limitation, we propose an approach to distilling classifier-free guided diffusion models into models that are fast to sample from: Given a pre-trained classifier-free guided model, we first learn a single model to match the output of the combined conditional and unconditional models, and then we progressively distill that model to a diffusion model that requires much fewer sampling steps. For standard diffusion models trained on the pixel-space, our approach is able to generate images visually comparable to that of the original model using as few as 4 sampling steps on ImageNet 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$64\times 64$</tex>
 and CIFAR-10, achieving FID/IS scores comparable to that of the original model while being up to 256 times faster to sample from. For diffusion models trained on the latent-space (e.g., Stable Diffusion), our approach is able to generate high-fidelity images using as few as 1 to 4 denoising steps, accelerating inference by at least 10-fold compared to existing methods on ImageNet 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$256\times 256$</tex>
 and LAION datasets. We further demonstrate the effectiveness of our approach on text-guided image editing and inpainting, where our distilled model is able to generate high-quality results using as few as 2–4 denoising steps.

        ----

        ## [1363] Dimensionality-Varying Diffusion Process

        **Authors**: *Han Zhang, Ruili Feng, Zhantao Yang, Lianghua Huang, Yu Liu, Yifei Zhang, Yujun Shen, Deli Zhao, Jingren Zhou, Fan Cheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01375](https://doi.org/10.1109/CVPR52729.2023.01375)

        **Abstract**:

        Diffusion models, which learn to reverse a signal destruction process to generate new data, typically require the signal at each step to have the same dimension. We argue that, considering the spatial redundancy in image signals, there is no need to maintain a high dimensionality in the evolution process, especially in the early generation phase. To this end, we make a theoretical generalization of the forward diffusion process via signal decomposition. Concretely, we manage to decompose an image into multiple orthogonal components and control the attenuation of each component when perturbing the image. That way, along with the noise strength increasing, we are able to diminish those inconsequential components and thus use a lower-dimensional signal to represent the source, barely losing information. Such a reformulation allows to vary dimensions in both training and inference of diffusion models. Extensive experiments on a range of datasets suggest that our approach substantially reduces the computational cost and achieves on-par or even better synthesis performance compared to baseline methods. We also show that our strategy facilitates high-resolution image synthesis and improves FID of diffusion model trained on FFHQ at 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$1024\times 1024$</tex>
 resolution from 52.40 to 10.46. Code is available at https://github.com/damo-vilab/dvdp.

        ----

        ## [1364] Shape-Aware Text-Driven Layered Video Editing

        **Authors**: *Yao-Chih Lee, Ji-Ze Genevieve Jang, Yi-Ting Chen, Elizabeth Qiu, Jia-Bin Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01376](https://doi.org/10.1109/CVPR52729.2023.01376)

        **Abstract**:

        Temporal consistency is essential for video editing applications. Existing work on layered representation of videos allows propagating edits consistently to each frame. These methods, however, can only edit object appearance rather than object shape changes due to the limitation of using a fixed UV mapping field for texture atlas. We present a shape-aware, text-driven video editing method to tackle this challenge. To handle shape changes in video editing, we first propagate the deformation field between the input and edited keyframe to all frames. We then leverage a pre-trained text-conditioned diffusion model as guidance for refining shape distortion and completing unseen regions. The experimental results demonstrate that our method can achieve shape-aware consistent video editing and compare favorably with the state-of-the-art.

        ----

        ## [1365] Rethinking Image Super Resolution from Long-Tailed Distribution Learning Perspective

        **Authors**: *Yuanbiao Gou, Peng Hu, Jiancheng Lv, Hongyuan Zhu, Xi Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01377](https://doi.org/10.1109/CVPR52729.2023.01377)

        **Abstract**:

        Existing studies have empirically observed that the resolution of the low-frequency region is easier to enhance than that of the high-frequency one. Although plentiful works have been devoted to alleviating this problem, little understanding is given to explain it. In this paper, we try to give a feasible answer from a machine learning perspective, i.e., the twin fitting problem caused by the long-tailed pixel distribution in natural images. With this explanation, we reformulate image super resolution (SR) as a long-tailed distribution learning problem and solve it by bridging the gaps of the problem between in low- and high-level vision tasks. As a result, we design a long-tailed distribution learning solution, that rebalances the gradients from the pixels in the low- and high-frequency region, by introducing a static and a learnable structure prior. The learned SR model achieves better balance on the fitting of the low- and high-frequency region so that the overall performance is improved. In the experiments, we evaluate the solution on four CNN- and one Transformer-based SR models w.r.t. six datasets and three tasks, and experimental results demonstrate its superiority.

        ----

        ## [1366] End-to-end Video Matting with Trimap Propagation

        **Authors**: *Wei-Lun Huang, Ming-Sui Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01378](https://doi.org/10.1109/CVPR52729.2023.01378)

        **Abstract**:

        The research of video matting mainly focuses on temporal coherence and has gained significant improvement via neural networks. However, matting usually relies on user-annotated trimaps to estimate alpha values, which is a labor-intensive issue. Although recent studies exploit video object segmentation methods to propagate the given trimaps, they suffer inconsistent results. Here we present a more robust and faster end-to-end video matting model equipped with trimap propagation called FTP-VM (Fast Trimap Propagation - Video Matting). The FTP-VM combines trimap propagation and video matting in one model, where the additional backbone in memory matching is replaced with the proposed lightweight trimap fusion module. The segmentation consistency loss is adopted from automotive segmentation to fit trimap segmentation with the collaboration of RNN (Recurrent Neural Network) to improve the temporal coherence. The experimental results demonstrate that the FTP-VM performs competitively both in composited and real videos only with few given trimaps. The efficiency is eight times higher than the state-of-the-art methods, which confirms its robustness and applicability in real-time scenarios. The code is available at https://github.com/csvt32745/FTP-VM.

        ----

        ## [1367] Context-Based Trit-Plane Coding for Progressive Image Compression

        **Authors**: *Seungmin Jeon, Kwang Pyo Choi, Youngo Park, Chang-Su Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01379](https://doi.org/10.1109/CVPR52729.2023.01379)

        **Abstract**:

        Trit-plane coding enables deep progressive image compression, but it cannot use autoregressive context models. In this paper, we propose the context-based trit-plane coding (CTC) algorithm to achieve progressive compression more compactly. First, we develop the context-based rate reduction module to estimate trit probabilities of latent elements accurately and thus encode the trit-planes compactly. Second, we develop the context-based distortion reduction module to refine partial latent tensors from the trit-planes and improve the reconstructed image quality. Third, we propose a retraining scheme for the decoder to attain better rate-distortion tradeoffs. Extensive experiments show that CTC outperforms the baseline trit-plane codec significantly, e.g. by -14.84% in BD-rate on the Kodak loss less dataset, while increasing the time complexity only marginally. The source codes are available at https://github.com/seungminjeon-github/CTC.

        ----

        ## [1368] Complexity-guided Slimmable Decoder for Efficient Deep Video Compression

        **Authors**: *Zhihao Hu, Dong Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01380](https://doi.org/10.1109/CVPR52729.2023.01380)

        **Abstract**:

        In this work, we propose the complexity-guided slimmable decoder (cgSlimDecoder) in combination with skip-adaptive entropy coding (SaEC) for efficient deep video compression. Specifically, given the target complexty constraints, in our cgSlimDecoder, we introduce a set of new channel width selection modules to automatically decide the optimal channel width of each slimmable convolution layer. By optimizing the complexity-rate-distortion related objective function to directly learn the parameters of the newly introduced channel width selection modules and other modules in the decoder, our cgSlimDecoder can automatically allocate the optimal numbers of parameters for different types of modules (e.g., motion/residual decoder and the motion compensation network) and simultaneously support multiple complexity levels by using a single learnt decoder instead of multiple decoders. In addition, our proposed SaEC can further accelerate the entropy decoding procedure in both motion and residual decoders by simply skipping the entropy coding process for the elements in the encoded feature maps that are already well-predicted by the hyperprior network. As demonstrated in our comprehensive experiments, our newly proposed methods cgSlimDecoder and SaEC are general and can be readily incorporated into three widely used deep video codecs (i.e., DVC, FVC and DCVC) to significantly improve their coding efficiency with negligible performance drop.

        ----

        ## [1369] Efficient Hierarchical Entropy Model for Learned Point Cloud Compression

        **Authors**: *Rui Song, Chunyang Fu, Shan Liu, Ge Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01381](https://doi.org/10.1109/CVPR52729.2023.01381)

        **Abstract**:

        Learning an accurate entropy model is a fundamental way to remove the redundancy in point cloud compression. Recently, the octree-based auto-regressive entropy model which adopts the self-attention mechanism to explore dependencies in a large-scale context is proved to be promising. However, heavy global attention computations and auto-regressive contexts are inefficient for practical applications. To improve the efficiency of the attention model, we propose a hierarchical attention structure that has a linear complexity to the context scale and maintains the global receptive field. Furthermore, we present a grouped context structure to address the serial decoding issue caused by the auto-regression while preserving the compression performance. Experiments demonstrate that the proposed entropy model achieves superior rate-distortion performance and significant decoding latency reduction compared with the state-of-the-art large-scale auto-regressive entropy model.

        ----

        ## [1370] NIRVANA: Neural Implicit Representations of Videos with Adaptive Networks and Autoregressive Patch-Wise Modeling

        **Authors**: *Shishira R. Maiya, Sharath Girish, Max Ehrlich, Hanyu Wang, Kwot Sin Lee, Patrick Poirson, Pengxiang Wu, Chen Wang, Abhinav Shrivastava*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01382](https://doi.org/10.1109/CVPR52729.2023.01382)

        **Abstract**:

        Implicit Neural Representations (INR) have recently shown to be powerful tool for high-quality video compression. However, existing works are are limiting as they do not exploit the temporal redundancy in videos, leading to a long encoding time. Additionally, these methods have fixed architectures which do not scale to longer videos or higher resolutions. To address these issues, we propose NIRVANA, which treats videos as groups of frames and fits separate networks to each group performing patch-wise prediction. The video representation is modeled autoregressively, with networks fit on a current group initialized using weights from the previous group's model. To enhance efficiency, we quantize the parameters during training, requiring no post-hoc pruning or quantization. When compared with previous works on the benchmark UVG dataset, NIRVANA improves encoding quality from 37.36 to 37.70 (in terms of PSNR) and the encoding speed by 12x, while maintaining the same compression rate. In contrast to prior video INR works which struggle with larger resolution and longer videos, we show that our algorithm scales naturally due to its patch-wise and autoregressive design. Moreover, our method achieves variable bitrate compression by adapting to videos with varying inter-frame motion. NIRVANA also achieves 6x decoding speed scaling well with more GPUs, making it practical for various deployment scenarios.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The project site can be found here.

        ----

        ## [1371] Learned Image Compression with Mixed Transformer-CNN Architectures

        **Authors**: *Jinming Liu, Heming Sun, Jiro Katto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01383](https://doi.org/10.1109/CVPR52729.2023.01383)

        **Abstract**:

        Learned image compression (LIC) methods have exhibited promising progress and superior rate-distortion performance compared with classical image compression standards. Most existing LIC methods are Convolutional Neural Networks-based (CNN-based) or Transformer-based, which have different advantages. Exploiting both advantages is a point worth exploring, which has two challenges: 1) how to effectively fuse the two methods? 2) how to achieve higher performance with a suitable complexity? In this paper, we propose an efficient parallel Transformer-CNN Mixture (TCM) block with a controllable complexity to incorporate the local modeling ability of CNN and the non-local modeling ability of transformers to improve the overall architecture of image compression models. Besides, inspired by the recent progress of entropy estimation models and attention modules, we propose a channel-wise entropy model with parameter-efficient swin-transformer-based attention (SWAtten) modules by using channel squeezing. Experimental results demonstrate our proposed method achieves state-of-the-art rate-distortion performances on three different resolution datasets (i.e., Kodak, Tecnick, CLIC Professional Validation) compared to existing LIC methods. The code is at https://github.com/jmliu206/LIC_TCM.

        ----

        ## [1372] Memory-Friendly Scalable Super-Resolution via Rewinding Lottery Ticket Hypothesis

        **Authors**: *Jin Lin, Xiaotong Luo, Ming Hong, Yanyun Qu, Yuan Xie, Zongze Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01384](https://doi.org/10.1109/CVPR52729.2023.01384)

        **Abstract**:

        Scalable deep Super-Resolution (SR) models are increasingly in demand, whose memory can be customized and tuned to the computational recourse of the platform. The existing dynamic scalable SR methods are not memory-friendly enough because multi-scale models have to be saved with a fixed size for each model. Inspired by the success of Lottery Tickets Hypothesis (LTH) on image classification, we explore the existence of unstructured scalable SR deep models, that is, we find gradual shrinkage subnetworks of extreme sparsity named winning tickets. In this paper, we propose a Memory-friendly Scalable SR framework (MSSR). The advantage is that only a single scalable model covers multiple SR models with different sizes, instead of reloading SR models of different sizes. Concretely, MSSR consists of the forward and backward stages, the former for model compression and the latter for model expansion. In the forward stage, we take advantage of LTH with rewinding weights to progressively shrink the SR model and the pruning-out masks that form nested sets. Moreover, stochastic self-distillation (SSD) is conducted to boost the performance of sub-networks. By stochastically selecting multiple depths, the current model inputs the selected features into the corresponding parts in the larger model and improves the performance of the current model based on the feedback results of the larger model. In the backward stage, the smaller SR model could be expanded by recovering and fine-tuning the pruned parameters according to the pruning-out masks obtained in the forward. Extensive experiments show the effectiveness of MMSR. The smallest-scale sub-network could achieve the sparsity of 94% and outperforms the compared lightweight SR methods.

        ----

        ## [1373] InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions

        **Authors**: *Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, Xiaogang Wang, Yu Qiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01385](https://doi.org/10.1109/CVPR52729.2023.01385)

        **Abstract**:

        Compared to the great progress of large-scale vision transformers (ViTs) in recent years, large-scale models based on convolutional neural networks (CNNs) are still in an early state. This work presents a new large-scale CNN-based foundation model, termed InternImage, which can obtain the gain from increasing parameters and training data like ViTs. Different from the recent CNNs that focus on large dense kernels, InternImage takes deformable convolution as the core operator, so that our model not only has the large effective receptive field required for downstream tasks such as detection and segmentation, but also has the adaptive spatial aggregation conditioned by input and task information. As a result, the proposed InternImage reduces the strict inductive bias of traditional CNNs and makes it possible to learn stronger and more robust patterns with large-scale parameters from massive data like ViTs. The effectiveness of our model is proven on challenging benchmarks including ImageNet, COCO, andADE20K. It is worth mentioning that InternImage-H achieved a new record 65.4 mAP on COCO test-dev and 62.9 mIoU on ADE20K, outperforming current leading CNNs and ViTs.

        ----

        ## [1374] EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention

        **Authors**: *Xinyu Liu, Houwen Peng, Ningxin Zheng, Yuqing Yang, Han Hu, Yixuan Yuan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01386](https://doi.org/10.1109/CVPR52729.2023.01386)

        **Abstract**:

        Vision transformers have shown great success due to their high model capabilities. However, their remarkable performance is accompanied by heavy computation costs, which makes them unsuitable for real-time applications. In this paper, we propose a family of high-speed vision transformers named Efficient ViT. We find that the speed of existing transformer models is commonly bounded by memory inefficient operations, especially the tensor reshaping and element-wise functions in MHSA. Therefore, we design a new building block with a sandwich layout, i.e., using a single memory-bound MHSA between efficient FFN layers, which improves memory efficiency while enhancing channel communication. Moreover, we discover that the attention maps share high similarities across heads, leading to computational redundancy. To address this, we present a cascaded group attention module feeding attention heads with different splits of the full feature, which not only saves computation cost but also improves attention diversity. Comprehensive experiments demonstrate EfficientViT outperforms existing efficient models, striking a good trade-off between speed and accuracy. For instance, our EfficientViT-M5 surpasses MobileNetV3-Large by 1.9% in accuracy, while getting 40.4% and 45.2% higher throughput on Nvidia V100 GPU and Intel Xeon CPU, respectively. Compared to the recent efficient model MobileViT-XXS, EfficientViT-M2 achieves 1.8% superior accuracy, while running 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$5.8\times/3.7\times$</tex>
 faster on the GPU/CPU, and 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$7.4\times faster$</tex>
 when converted to ONNX format. Code and models are available at here.

        ----

        ## [1375] Castling-ViT: Compressing Self-Attention via Switching Towards Linear-Angular Attention at Vision Transformer Inference

        **Authors**: *Haoran You, Yunyang Xiong, Xiaoliang Dai, Bichen Wu, Peizhao Zhang, Haoqi Fan, Peter Vajda, Yingyan Celine Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01387](https://doi.org/10.1109/CVPR52729.2023.01387)

        **Abstract**:

        Vision Transformers (ViTs) have shown impressive per-formance but still require a high computation cost as compared to convolutional neural networks (CNNs), one rea-son is that ViTs' attention measures global similarities and thus has a quadratic complexity with the number of in-put tokens. Existing efficient ViTs adopt local attention or linear attention, which sacrifice ViTs' capabilities of capturing either global or local context. In this work, we ask an important research question: Can ViTs learn both global and local context while being more efficient during inference? To this end, we propose a framework called Castling- ViT, which trains ViTs using both linear-angular attention and masked softmax-based quadratic attention, but then switches to having only linear-angular attention during inference. Our Castling- ViT leverages angular ker-nels to measure the similarities between queries and keys via spectral angles. And we further simplify it with two techniques: (1) a novel linear-angular attention mechanism: we decompose the angular kernels into linear terms and high-order residuals, and only keep the linear terms; and (2) we adopt two parameterized modules to approximate high-order residuals: a depthwise convolution and an aux-iliary masked softmax attention to help learn global and lo-cal information, where the masks for softmax attention are regularized to gradually become zeros and thus incur no overhead during inference. Extensive experiments validate the effectiveness of our Castling- ViT, e.g., achieving up to a 1.8% higher accuracy or 40% MACs reduction on classification and 1.2 higher mAP on detection under comparable FLOPs, as compared to ViTs with vanilla softmax-based at-tentions. Project page is available at here.

        ----

        ## [1376] RIFormer: Keep Your Vision Backbone Effective But Removing Token Mixer

        **Authors**: *Jiahao Wang, Songyang Zhang, Yong Liu, Taiqiang Wu, Yujiu Yang, Xihui Liu, Kai Chen, Ping Luo, Dahua Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01388](https://doi.org/10.1109/CVPR52729.2023.01388)

        **Abstract**:

        This paper studies how to keep a vision backbone effective while removing token mixers in its basic building blocks. Token mixers, as self-attention for vision transformers (ViTs), are intended to perform information communication between different spatial tokens but suffer from considerable computational cost and latency. However, directly removing them will lead to an incomplete model structure prior, and thus brings a significant accuracy drop. To this end, we first develop an RepIdentityFormer base on the re-parameterizing idea, to study the token mixer free model architecture. And we then explore the improved learning paradigm to break the limitation of simple token mixer free backbone, and summarize the empirical practice into 5 guidelines. Equipped with the proposed optimization strategy, we are able to build an extremely simple vision backbone with encouraging performance, while enjoying the high efficiency during inference. Extensive experiments and ablative analysis also demonstrate that the inductive bias of network architecture, can be incorporated into simple network structure with appropriate optimization strategy. We hope this work can serve as a starting point for the exploration of optimization-driven efficient network design.

        ----

        ## [1377] High-resolution image reconstruction with latent diffusion models from human brain activity

        **Authors**: *Yu Takagi, Shinji Nishimoto*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01389](https://doi.org/10.1109/CVPR52729.2023.01389)

        **Abstract**:

        Reconstructing visual experiences from human brain activity offers a unique way to understand how the brain represents the world, and to interpret the connection between computer vision models and our visual system. While deep generative models have recently been employed for this task, reconstructing realistic images with high semantic fidelity is still a challenging problem. Here, we propose a new method based on a diffusion model (DM) to reconstruct images from human brain activity obtained via functional magnetic resonance imaging (fMRI). More specifically, we rely on a latent diffusion model (LDM) termed Stable Diffusion. This model reduces the computational cost of DMs, while preserving their high generative performance. We also characterize the inner mechanisms of the LDM by studying how its different components (such as the latent vector of image Z, conditioning inputs C, and different elements of the denoising U-Net) relate to distinct brain functions. We show that our proposed method can reconstruct high-resolution images with high fidelity in straight-forward fashion, without the need for any additional training and fine-tuning of complex deep-learning models. We also provide a quantitative interpretation of different LDM components from a neuroscientific perspective. Overall, our study proposes a promising method for reconstructing images from human brain activity, and provides a new framework for understanding DMs. Please check out our webpage at https://sites.google.com/view/stablediffusion-withbrain/.

        ----

        ## [1378] Non-Contrastive Unsupervised Learning of Physiological Signals from Video

        **Authors**: *Jeremy Speth, Nathan Vance, Patrick J. Flynn, Adam Czajka*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01390](https://doi.org/10.1109/CVPR52729.2023.01390)

        **Abstract**:

        Subtle periodic signals such as blood volume pulse and respiration can be extracted from RGB video, enabling non-contact health monitoring at low cost. Advancements in remote pulse estimation - or remote photoplethysmography (rPPG) - are currently driven by deep learning solutions. However, modern approaches are trained and evaluated on benchmark datasets with ground truth from contact-P PG sensors. We present the first non-contrastive unsuper-vised learning framework for signal regression to mitigate the need for labelled video data. With minimal assumptions of periodicity and finite bandwidth, our approach discovers the blood volume pulse directly from unlabelled videos. We find that encouraging sparse power spectra within normal physiological bandlimits and variance over batches of power spectra is sufficient for learning visual features of periodic signals. We perform the first experiments utilizing unlabelled video data not specifically created for rPPG to train robust pulse rate estimators. Given the limited inductive biases and impressive empirical results, the approach is theoretically capable of discovering other periodic signals from video, enabling multiple physiological measurements without the need for ground truth signals.

        ----

        ## [1379] Revealing the Dark Secrets of Masked Image Modeling

        **Authors**: *Zhenda Xie, Zigang Geng, Jingcheng Hu, Zheng Zhang, Han Hu, Yue Cao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01391](https://doi.org/10.1109/CVPR52729.2023.01391)

        **Abstract**:

        Masked image modeling (MIM) as pre-training is shown to be effective for numerous vision downstream tasks, but how and where MIM works remain unclear. In this paper, we compare MIM with the long-dominant supervised pretrained models from two perspectives, the visualizations and the experiments, to uncover their key representational differences. From the visualizations, we find that MIM brings locality inductive bias to all layers of the trained models, but supervised models tend to focus locally at lower layers but more globally at higher layers. That may be the reason why MIM helps Vision Transformers that have a very large receptive field to optimize. Using MIM, the model can maintain a large diversity on attention heads in all layers. But for supervised models, the diversity on attention heads almost disappears from the last three layers and less diversity harms the fine-tuning performance. From the experiments, we find that MIM models can perform significantly better on geometric and motion tasks with weak semantics or fine-grained classification tasks, than their supervised counterparts. Without bells and whistles, a standard MIM pre-trained SwinV2-L could achieve state-of-the-art performance on pose estimation (78.9 AP on COCO test-dev and 78.0 AP on CrowdPose), depth estimation (0.287 RMSE on NYUv2 and 1.966 RMSE on KITTI), and video object tracking (70.7 SUC on LaSOT). For the semantic understanding datasets where the categories are sufficiently covered by the supervised pre-training, MIM models can still achieve highly competitive transfer performance. With a deeper understanding of MIM, we hope that our work can inspire new and solid research in this direction. Code will be available at https://github.com/zdaxie/MIM-DarkSecrets.

        ----

        ## [1380] Improving Visual Representation Learning Through Perceptual Understanding

        **Authors**: *Samyakh Tukra, Frederick Hoffman, Ken Chatfield*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01392](https://doi.org/10.1109/CVPR52729.2023.01392)

        **Abstract**:

        We present an extension to masked autoencoders (MAE) which improves on the representations learnt by the model by explicitly encouraging the learning of higher scene-level features. We do this by: (i) the introduction of a perceptual similarity term between generated and real images (ii) incorporating several techniques from the adversarial training literature including multi-scale training and adaptive discriminator augmentation. The combination of these results in not only better pixel reconstruction but also representations which appear to capture better higher-level details within images. More consequentially, we show how our method, Perceptual MAE, leads to better performance when used for downstream tasks outperforming previous methods. We achieve 78.1% top-1 accuracy linear probing on ImageNet-1K and up to 88.1% when fine-tuning, with similar results for other downstream tasks, all without use of additional pre-trained models or data.

        ----

        ## [1381] FlexiViT: One Model for All Patch Sizes

        **Authors**: *Lucas Beyer, Pavel Izmailov, Alexander Kolesnikov, Mathilde Caron, Simon Kornblith, Xiaohua Zhai, Matthias Minderer, Michael Tschannen, Ibrahim Alabdulmohsin, Filip Pavetic*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01393](https://doi.org/10.1109/CVPR52729.2023.01393)

        **Abstract**:

        Vision Transformers convert images to sequences by slicing them into patches. The size of these patches controls a speed/accuracy tradeoff, with smaller patches leading to higher accuracy at greater computational cost, but changing the patch size typically requires retraining the model. In this paper, we demonstrate that simply randomizing the patch size at training time leads to a single set of weights that performs well across a wide range of patch sizes, making it possible to tailor the model to different compute budgets at deployment time. We extensively evaluate the resulting model, which we call FlexiViT, on a wide range of tasks, including classification, image-text retrieval, open-world detection, panoptic segmentation, and semantic segmentation, concluding that it usually matches, and sometimes outperforms, standard ViT models trained at a single patch size in an otherwise identical setup. Hence, FlexiViT training is a simple drop-in improvement for ViT that makes it easy to add compute-adaptive capabilities to most models relying on a ViT backbone architecture. Code and pre-trained models are available at github.com/google-research/big_vision.

        ----

        ## [1382] AdaMAE: Adaptive Masking for Efficient Spatiotemporal Learning with Masked Autoencoders

        **Authors**: *Wele Gedara Chaminda Bandara, Naman Patel, Ali Gholami, Mehdi Nikkhah, Motilal Agrawal, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01394](https://doi.org/10.1109/CVPR52729.2023.01394)

        **Abstract**:

        Masked Autoencoders (MAEs) learn generalizable representations for image, text, audio, video, etc., by reconstructing masked input data from tokens of the visible data. Current MAE approaches for videos rely on random patch, tube, or frame based masking strategies to select these tokens. This paper proposes AdaMAE, an adaptive masking strategy for MAEs that is end-to-end trainable. Our adaptive masking strategy samples visible tokens based on the semantic context using an auxiliary sampling network. This network estimates a categorical distribution over spacetime-patch tokens. The tokens that increase the expected reconstruction error are rewarded and selected as visible tokens, motivated by the policy gradient algorithm in reinforcement learning. We show that AdaMAE samples more tokens from the high spatiotemporal information regions, thereby allowing us to mask 95% of tokens, resulting in lower memory requirements and faster pre-training. We conduct ablation studies on the Something-Something v2 (SSv2) dataset to demonstrate the efficacy of our adaptive sampling approach and report state-of-the-art results of 70.0% and 81.7% in top-1 accuracy on SSv2 and Kinetics-400 action classification datasets with a ViT-Base backbone and 800 pre-training epochs. Code and pre-trained models are available at: https://github.com/wgcban/adamae.git.

        ----

        ## [1383] SimpSON: Simplifying Photo Cleanup with Single-Click Distracting Object Segmentation Network

        **Authors**: *Chuong Huynh, Yuqian Zhou, Zhe Lin, Connelly Barnes, Eli Shechtman, Sohrab Amirghodsi, Abhinav Shrivastava*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01395](https://doi.org/10.1109/CVPR52729.2023.01395)

        **Abstract**:

        In photo editing, it is common practice to remove visual distractions to improve the overall image quality and highlight the primary subject. However, manually selecting and removing these small and dense distracting regions can be a laborious and time-consuming task. In this paper, we propose an interactive distractor selection method that is optimized to achieve the task with just a single click. Our method surpasses the precision and recall achieved by the traditional method of running panoptic segmentation and then selecting the segments containing the clicks. We also showcase how a transformer-based module can be used to identify more distracting regions similar to the user's click position. Our experiments demonstrate that the model can effectively and accurately segment unknown distracting objects interactively and in groups. By significantly simplifying the photo cleaning and retouching process, our proposed model provides inspiration for exploring rare object segmentation and group selection with a single click. More information can be found at https://github.com/hmchuong/SimpSON.

        ----

        ## [1384] Visual Dependency Transformers: Dependency Tree Emerges from Reversed Attention

        **Authors**: *Mingyu Ding, Yikang Shen, Lijie Fan, Zhenfang Chen, Zitian Chen, Ping Luo, Joshua B. Tenenbaum, Chuang Gan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01396](https://doi.org/10.1109/CVPR52729.2023.01396)

        **Abstract**:

        Humans possess a versatile mechanism for extracting structured representations of our visual world. When looking at an image, we can decompose the scene into entities and their parts as well as obtain the dependencies between them. To mimic such capability, we propose Visual Dependency Transformers (DependencyViT) 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/dingmyu/DependencyViT that can induce visual dependencies without any labels. We achieve that with a novel neural operator called reversed attention that can naturally capture long-range visual dependencies between image patches. Specifically, we formulate it as a dependency graph where a child token in reversed attention is trained to attend to its parent tokens and send information following a normalized probability distribution rather than gathering information in conventional self-attention. With such a design, hierarchies naturally emerge from reversed attention layers, and a dependency tree is progressively induced from leaf nodes to the root node unsupervisedly. DependencyViT offers several appealing benefits. (i) Entities and their parts in an image are represented by different subtrees, enabling part partitioning from dependencies; (ii) Dynamic visual pooling is made possible. The leaf nodes which rarely send messages can be pruned without hindering the model performance, based on which we propose the lightweight DependencyViT-Lite to reduce the computational and memory footprints; (iii) DependencyViT works well on both self- and weakly-supervised pretraining paradigms on ImageNet, and demonstrates its effectiveness on 8 datasets and 5 tasks, such as unsupervised part and saliency segmentation, recognition, and detection.

        ----

        ## [1385] Iterative Next Boundary Detection for Instance Segmentation of Tree Rings in Microscopy Images of Shrub Cross Sections

        **Authors**: *Alexander Gillert, Giulia Resente, Alba Anadon-Rosell, Martin Wilmking, Uwe Freiherr von Lukas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01397](https://doi.org/10.1109/CVPR52729.2023.01397)

        **Abstract**:

        We address the problem of detecting tree rings in microscopy images of shrub cross sections. This can be regarded as a special case of the instance segmentation task with several unique challenges such as the concentric circular ring shape of the objects and high precision requirements that result in inadequate performance of existing methods. We propose a new iterative method which we term Iterative Next Boundary Detection (INBD). It intuitively models the natural growth direction, starting from the center of the shrub cross section and detecting the next ring boundary in each iteration step. In our experiments, INBD shows superior performance to generic instance segmentation methods and is the only one with a built-in notion of chronological order. Our dataset and source code are available at http://github.com/alexander-g/INBD.

        ----

        ## [1386] VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

        **Authors**: *Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, Yu Qiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01398](https://doi.org/10.1109/CVPR52729.2023.01398)

        **Abstract**:

        Scale is the primary factor for building a powerful foundation model that could well generalize to a variety of downstream tasks. However, it is still challenging to train video foundation models with billions of parameters. This paper shows that video masked autoencoder (VideoMAE) is a scalable and general self-supervised pre-trainer for building video foundation models. We scale the VideoMAE in both model and data with a core design. Specifically, we present a dual masking strategy for efficient pre-training, with an encoder operating on a subset of video tokens and a decoder processing another subset of video tokens. Although VideoMAE is very efficient due to high masking ratio in encoder, masking decoder can still further reduce the overall computational cost. This enables the efficient pre-training of billion-level models in video. We also use a progressive training paradigm that involves an initial pre-training on a diverse multi-sourced unlabeled dataset, followed by a post-pre-training on a mixed labeled dataset. Finally, we successfully train a video ViT model with a billion parameters, which achieves a new state-of-the-art performance on the datasets of Kinetics (90.0% on K400 and 89.9% on K600) and Something-Something (68.7% on V1 and 77.0% on V2). In addition, we extensively verify the pre-trained video ViT models on a variety of downstream tasks, demonstrating its effectiveness as a general video representation learner.

        ----

        ## [1387] DropMAE: Masked Autoencoders with Spatial-Attention Dropout for Tracking Tasks

        **Authors**: *Qiangqiang Wu, Tianyu Yang, Ziquan Liu, Baoyuan Wu, Ying Shan, Antoni B. Chan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01399](https://doi.org/10.1109/CVPR52729.2023.01399)

        **Abstract**:

        In this paper, we study masked autoencoder (MAE) pretraining on videos for matching-based downstream tasks, including visual object tracking (VOT) and video object segmentation (VOS). A simple extension of MAE is to randomly mask out frame patches in videos and reconstruct the frame pixels. However, we find that this simple baseline heavily relies on spatial cues while ignoring temporal relations for frame reconstruction, thus leading to sub-optimal temporal matching representations for VOT and VOS. To alleviate this problem, we propose DropMAE, which adaptively performs spatial-attention dropout in the frame reconstruction to facilitate temporal correspondence learning in videos. We show that our DropMAE is a strong and efficient temporal matching learner, which achieves better finetuning results on matching-based tasks than the ImageNet-based MAE with 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$2\times$</tex>
 faster pre-training speed. Moreover, we also find that motion diversity in pre-training videos is more important than scene diversity for improving the performance on VOT and VOS. Our pre-trained DropMAE model can be directly loaded in existing ViT-based trackers for fine-tuning without further modifications. Notably, DropMAE sets new state-of-the-art performance on 8 out of 9 highly competitive video tracking and segmentation datasets. Our code and pre-trained models are available at https://github.com/jimmy-dq/DropMAE.git.

        ----

        ## [1388] SeqTrack: Sequence to Sequence Learning for Visual Object Tracking

        **Authors**: *Xin Chen, Houwen Peng, Dong Wang, Huchuan Lu, Han Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01400](https://doi.org/10.1109/CVPR52729.2023.01400)

        **Abstract**:

        In this paper, we present a new sequence-to-sequence learning framework for visual tracking, dubbed SeqTrack. It casts visual tracking as a sequence generation problem, which predicts object bounding boxes in an autoregressive fashion. This is different from prior Siamese trackers and transformer trackers, which rely on designing complicated head networks, such as classification and regression heads. SeqTrack only adopts a simple encoder-decoder transformer architecture. The encoder extracts visual features with a bidirectional transformer, while the decoder generates a sequence of bounding box values autoregressively with a causal transformer. The loss function is a plain cross-entropy. Such a sequence learning paradigm not only simplifies tracking framework, but also achieves competitive performance on benchmarks. For instance, SeqTrack gets 72.5% AUC on LaSOT, establishing a new state-of-the-art performance. Code and models are available at https://github.com/microsoft/VideoX.

        ----

        ## [1389] Bootstrapping Objectness from Videos by Relaxed Common Fate and Visual Grouping

        **Authors**: *Long Lian, Zhirong Wu, Stella X. Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01401](https://doi.org/10.1109/CVPR52729.2023.01401)

        **Abstract**:

        We study learning object segmentation from unlabeled videos. Humans can easily segment moving objects without knowing what they are. The Gestalt law of common fate, i.e., what move at the same speed belong together, has inspired unsupervised object discovery based on motion segmentation. However, common fate is not a reliable indicator of objectness: Parts of an articulated / deformable object may not move at the same speed, whereas shadows / reflections of an object always move with it but are not part of it. Our insight is to bootstrap objectness by first learning image features from relaxed common fate and then refining them based on visual appearance grouping within the image itself and across images statistically. Specifically, we learn an image segmenter first in the loop of approximating optical flow with constant segment flow plus small within-segment residual flow, and then by refining it for more coherent appearance and statistical figure-ground relevance. On unsupervised video object segmentation, using only ResNet and convolutional heads, our model surpasses the state-of-the-art by absolute gains of 7/9/5% on DAVIS16 / STv2 / FBMS59 respectively, demonstrating the effectiveness of our ideas. Our code is publicly available.

        ----

        ## [1390] Video Event Restoration Based on Keyframes for Video Anomaly Detection

        **Authors**: *Zhiwei Yang, Jing Liu, Zhaoyang Wu, Peng Wu, Xiaotao Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01402](https://doi.org/10.1109/CVPR52729.2023.01402)

        **Abstract**:

        Video anomaly detection (VAD) is a significant computer vision problem. Existing deep neural network (DNN) based VAD methods mostly follow the route of frame reconstruction or frame prediction. However, the lack of mining and learning of higher-level visual features and temporal context relationships in videos limits the further performance of these two approaches. Inspired by video codec theory, we introduce a brand-new VAD paradigm to break through these limitations: First, we propose a new task of video event restoration based on keyframes. Encouraging DNN to infer missing multiple frames based on video keyframes so as to restore a video event, which can more effectively motivate DNN to mine and learn potential higher-level visual features and comprehensive temporal context relationships in the video. To this end, we propose a novel U-shaped Swin Transformer Network with Dual Skip Connections (USTN-DSC) for video event restoration, where a cross-attention and a temporal upsampling residual skip connection are introduced to further assist in restoring complex static and dynamic motion object features in the video. In addition, we propose a simple and effective adjacent frame difference loss to constrain the motion consistency of the video sequence. Extensive experiments on benchmarks demonstrate that USTN-DSC outperforms most existing methods, validating the effectiveness of our method.

        ----

        ## [1391] Streaming Video Model

        **Authors**: *Yucheng Zhao, Chong Luo, Chuanxin Tang, Dongdong Chen, Noel Codella, Zheng-Jun Zha*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01403](https://doi.org/10.1109/CVPR52729.2023.01403)

        **Abstract**:

        Video understanding tasks have traditionally been modeled by two separate architectures, specially tailored for two distinct tasks. Sequence-based video tasks, such as action recognition, use a video backbone to directly extract spatiotemporal features, while frame-based video tasks, such as multiple object tracking (MOT), rely on single fixed-image backbone to extract spatial features. In contrast, we propose to unify video understanding tasks into one novel streaming video architecture, referred to as Streaming Vision Transformer (S-ViT). S-ViT first produces frame-level features with a memory-enabled temporally-aware spatial encoder to serve the frame-based video tasks. Then the frame features are input into a task-related temporal decoder to obtain spatiotemporal features for sequence-based tasks. The efficiency and efficacy of S-ViT is demonstrated by the state-of-the-art accuracy in the sequence-based action recognition task and the competitive advantage over conventional architecture in the frame-based MOT task. We believe that the concept of streaming video model and the implementation of S-ViT are solid steps towards a unified deep learning architecture for video understanding. Code will be available at https://github.com/yuzhms/Streaming-Video-Model.

        ----

        ## [1392] LSTFE-Net: Long Short-Term Feature Enhancement Network for Video Small Object Detection

        **Authors**: *Jinsheng Xiao, Yuanxu Wu, Yunhua Chen, Shurui Wang, Zhongyuan Wang, Jiayi Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01404](https://doi.org/10.1109/CVPR52729.2023.01404)

        **Abstract**:

        Video small object detection is a difficult task due to the lack of object information. Recent methods focus on adding more temporal information to obtain more potent high-level features, which often fail to specify the most vital information for small objects, resulting in insufficient or inappropriate features. Since information from frames at different positions contributes differently to small objects, it is not ideal to assume that using one universal method will extract proper features. We find that context information from the long-term frame and temporal information from the short-term frame are two useful cues for video small object detection. To fully utilize these two cues, we propose a long short-term feature enhancement network (LSTFE-Net) for video small object detection. First, we develop a plug-and-play spatiotemporal feature alignment module to create temporal correspondences between the short-term and current frames. Then, we propose a frame selection module to select the long-term frame that can provide the most additional context information. Finally, we propose a long short-term feature aggregation module to fuse long short-term features. Compared to other state-of-the-art methods, our LSTFE-Net achieves 4.4% absolute boosts in AP on the FL-Drones dataset. More details can be found at https://github.com/xiaojs18/LSTFE-Net.

        ----

        ## [1393] A Generalized Framework for Video Instance Segmentation

        **Authors**: *Miran Heo, Sukjun Hwang, Jeongseok Hyun, Hanjung Kim, Seoung Wug Oh, Joon-Young Lee, Seon Joo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01405](https://doi.org/10.1109/CVPR52729.2023.01405)

        **Abstract**:

        The handling of long videos with complex and occluded sequences has recently emerged as a new challenge in the video instance segmentation (VIS) community. However, existing methods have limitations in addressing this challenge. We argue that the biggest bottleneck in current approaches is the discrepancy between training and inference. To effectively bridge this gap, we propose a Generalized framework for VIS, namely GenVIS, that achieves state-of-the-art performance on challenging benchmarks without designing complicated architectures or requiring extra post-processing. The key contribution of GenVIS is the learning strategy, which includes a query-based training pipeline for sequential learning with a novel target label assignment. Additionally, we introduce a memory that effectively acquires information from previous states. Thanks to the new perspective, which focuses on building relationships between separate frames or clips, GenVIS can be flexibly executed in both online and semi-online manner. We evaluate our approach on popular VIS benchmarks, achieving state-of-the-art results on YouTube-VIS 2019/2021/2022 and Occluded VIS (OVIS). Notably, we greatly outperform the state-of-the-art on the long VIS benchmark (OVIS), improving 5.6 AP with ResNet-50 backbone. Code is available at https://github.com/miranheo/GenVIS.

        ----

        ## [1394] Referring Multi-Object Tracking

        **Authors**: *Dongming Wu, Wencheng Han, Tiancai Wang, Xingping Dong, Xiangyu Zhang, Jianbing Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01406](https://doi.org/10.1109/CVPR52729.2023.01406)

        **Abstract**:

        Existing referring understanding tasks tend to involve the detection of a single text-referred object. In this paper, we propose a new and general referring understanding task, termed referring multi-object tracking (RMOT). Its core idea is to employ a language expression as a semantic cue to guide the prediction of multi-object tracking. To the best of our knowledge, it is the first work to achieve an arbitrary number of referent object predictions in videos. To push forward RMOT, we construct one benchmark with scalable expressions based on KITTI, named Refer-KITTI. Specifically, it provides 18 videos with 818 expressions, and each expression in a video is annotated with an average of 10.7 objects. Further, we develop a transformer-based architecture TransRMOT to tackle the new task in an online manner, which achieves impressive detection performance and out-performs other counterparts. The Refer-KITTI dataset and the code are released at https://referringmot.github.io.

        ----

        ## [1395] Source-Free Video Domain Adaptation with Spatial-Temporal-Historical Consistency Learning

        **Authors**: *Kai Li, Deep Patel, Erik Kruus, Martin Renqiang Min*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01407](https://doi.org/10.1109/CVPR52729.2023.01407)

        **Abstract**:

        Source-free domain adaptation (SFDA) is an emerging research topic that studies how to adapt a pretrained source model using unlabeled target data. It is derived from unsupervised domain adaptation but has the advantage of not requiring labeled source data to learn adaptive models. This makes it particularly useful in real-world applications where access to source data is restricted. While there has been some SFDA work for images, little attention has been paid to videos. Naively extending image-based methods to videos without considering the unique properties of videos often leads to unsatisfactory results. In this paper, we propose a simple and highly flexible method for Source-Free Video Domain Adaptation (SFVDA), which extensively exploits consistency learning for videos from spatial, temporal, and historical perspectives. Our method is based on the assumption that videos of the same action category are drawn from the same low-dimensional space, regardless of the spatio-temporal variations in the high-dimensional space that cause domain shifts. To overcome domain shifts, we simulate spatio-temporal variations by applying spatial and temporal augmentations on target videos and encourage the model to make consistent predictions from a video and its augmented versions. Due to the simple design, our method can be applied to various SFVDA settings, and experiments show that our method achieves state-of-the-art performance for all the settings.

        ----

        ## [1396] Seeing What You Said: Talking Face Generation Guided by a Lip Reading Expert

        **Authors**: *Jiadong Wang, Xinyuan Qian, Malu Zhang, Robby T. Tan, Haizhou Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01408](https://doi.org/10.1109/CVPR52729.2023.01408)

        **Abstract**:

        Talking face generation, also known as speech-to-lip generation, reconstructs facial motions concerning lips given coherent speech input. The previous studies revealed the importance of lip-speech synchronization and visual quality. Despite much progress, they hardly focus on the content of lip movements i.e., the visual intelligibility of the spoken words, which is an important aspect of generation quality. To address the problem, we propose using a lipreading expert to improve the intelligibility of the generated lip regions by penalizing the incorrect generation results. Moreover, to compensate for data scarcity, we train the lip-reading expert in an audio-visual self-supervised manner. With a lip-reading expert, we propose a novel contrastive learning to enhance lip-speech synchronization, and a transformer to encode audio synchronically with video, while considering global temporal dependency of audio. For evaluation, we propose a new strategy with two different lip-reading experts to measure intelligibility of the generated videos. Rigorous experiments show that our proposal is superior to other State-of-the-art (SOTA) methods, such as Wav2Lip, in reading intelligibility i.e., over 38% Word Error Rate (WER) on LRS2 dataset and 27.8% accuracy on LRW dataset. We also achieve the SOTA performance in lip-speech synchronization and comparable performances in visual quality.

        ----

        ## [1397] Egocentric Auditory Attention Localization in Conversations

        **Authors**: *Fiona Ryan, Hao Jiang, Abhinav Shukla, James M. Rehg, Vamsi Krishna Ithapu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01409](https://doi.org/10.1109/CVPR52729.2023.01409)

        **Abstract**:

        In a noisy conversation environment such as a dinner party, people often exhibit selective auditory attention, or the ability to focus on a particular speaker while tuning out others. Recognizing who somebody is listening to in a conversation is essential for developing technologies that can understand social behavior and devices that can augment human hearing by amplifying particular sound sources. The computer vision and audio research communities have made great strides towards recognizing sound sources and speakers in scenes. In this work, we take a step further by focusing on the problem of localizing auditory attention targets in egocentric video, or detecting who in a camera wearer's field of view they are listening to. To tackle the new and challenging Selective Auditory Attention Localization problem, we propose an end-to-end deep learning approach that uses egocentric video and multichannel audio to predict the heatmap of the camera wearer's auditory attention. Our approach leverages spatiotemporal audiovisual features and holistic reasoning about the scene to make predictions, and outperforms a set of baselines on a challenging multi-speaker conversation dataset. Project page: https://fkryan.github.io/saal

        ----

        ## [1398] iQuery: Instruments as Queries for Audio-Visual Sound Separation

        **Authors**: *Jiaben Chen, Renrui Zhang, Dongze Lian, Jiaqi Yang, Ziyao Zeng, Jianbo Shi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01410](https://doi.org/10.1109/CVPR52729.2023.01410)

        **Abstract**:

        Current audiovisual separation methods share a standard architecture design where an audio encoder-decoder network is fused with visual encoding features at the encoder bottleneck. This design confounds the learning of multimodal feature encoding with robust sound decoding for audio separation. To generalize to a new instrument, one must finetune the entire visual and audio network for all musical instruments. We re-formulate the visual-sound separation task and propose Instruments as Queries (iQuery) with a flexible query expansion mechanism. Our approach ensures cross-modal consistency and cross-instrument disentanglement. We utilize “visually named” queries to initiate the learning of audio queries and use cross-modal attention to remove potential sound source interference at the estimated waveforms. To generalize to a new instrument or event class, drawing inspiration from the text-prompt design, we insert additional queries as audio prompts while freezing the attention mechanism. Experimental results on three benchmarks demonstrate that our iQuery improves audiovisual sound source separation performance. Code is available at https://github.com/JiabenChen/iQuery.

        ----

        ## [1399] Learning to Dub Movies via Hierarchical Prosody Models

        **Authors**: *Gaoxiang Cong, Liang Li, Yuankai Qi, Zheng-Jun Zha, Qi Wu, Wenyu Wang, Bin Jiang, Ming-Hsuan Yang, Qingming Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01411](https://doi.org/10.1109/CVPR52729.2023.01411)

        **Abstract**:

        Given a piece of text, a video clip and a reference audio, the movie dubbing (also known as visual voice clone, V2C) task aims to generate speeches that match the speaker's emotion presented in the video using the desired speaker voice as reference. V2C is more challenging than conventional text-to-speech tasks as it additionally requires the generated speech to exactly match the varying emotions and speaking speed presented in the video. Unlike previous works, we propose a novel movie dubbing architecture to tackle these problems via hierarchical prosody modeling, which bridges the visual information to corresponding speech prosody from three aspects: lip, face, and scene. Specifically, we align lip movement to the speech duration, and convey facial expression to speech energy and pitch via attention mechanism based on valence and arousal representations inspired by the psychology findings. Moreover, we design an emotion booster to capture the atmosphere from global video scenes. All these embeddings are used together to generate mel-spectrogram, which is then converted into speech waves by an existing vocoder. Extensive experimental results on the V2C and Chem benchmark datasets demonstrate the favourable performance of the proposed method. The code and trained models will be made available at https://github.com/GalaxyCong/HPMDubbing

        ----

        

[Go to the previous page](CVPR-2023-list06.md)

[Go to the next page](CVPR-2023-list08.md)

[Go to the catalog section](README.md)