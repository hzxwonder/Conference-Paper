## [0] Affordances from Human Videos as a Versatile Representation for Robotics

        **Authors**: *Shikhar Bahl, Russell Mendonca, Lili Chen, Unnat Jain, Deepak Pathak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.01324](https://doi.org/10.1109/CVPR52729.2023.01324)

        **Abstract**:

        Building a robot that can understand and learn to interact by watching humans has inspired several vision problems. However, despite some successful results on static datasets, it remains unclear how current models can be used on a robot directly. In this paper, we aim to bridge this gap by leveraging videos of human interactions in an environment centric manner. Utilizing internet videos of human behavior, we train a visual affordance model that estimates where and how in the scene a human is likely to interact. The structure of these behavioral affordances directly enables the robot to perform many complex tasks. We show how to seamlessly integrate our affordance model with four robot learning paradigms including offline imitation learning, exploration, goal-conditioned learning, and action parameterization for reinforcement learning. We show the efficacy of our approach, which we call Vision-Robotics Bridge (VRB) across 4 real world environments, over 10 different tasks, and 2 robotic platforms operating in the wild.

        ----

        ## [1] RefCLIP: A Universal Teacher for Weakly Supervised Referring Expression Comprehension

        **Authors**: *Lei Jin, Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Guannan Jiang, Annan Shu, Rongrong Ji*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00263](https://doi.org/10.1109/CVPR52729.2023.00263)

        **Abstract**:

        Referring Expression Comprehension (REC) is a task of grounding the referent based on an expression, and its development is greatly limited by expensive instance-level annotations. Most existing weakly supervised methods are built based on two-stage detection networks, which are computationally expensive. In this paper, we resort to the efficient one-stage detector and propose a novel weakly supervised model called RefCLIP.Specifically, RefCLIP redefines weakly supervised REC as an anchor-text matching problem, which can avoid the complex post-processing in existing methods. To achieve weakly supervised learning, we introduce anchor-based contrastive loss to optimize Re-fCLlP via numerous anchor-text pairs. Based on RefCLIP, we further propose the first model-agnostic weakly supervised training scheme for existing REC models, where RefCLIP acts as a mature teacher to generate pseudo-labels for teaching common REC models. With our careful designs, this scheme can even help existing REC models achieve better weakly supervised performance than RefCLIP, e.g., TransVg and SimREC. To validate our approaches, we conduct extensive experiments on four REC benchmarks, i.e., RefCOCO, RefCOCO+, RefCOCOg and ReferItGame. Experimental results not only report our significant performance gains over existing weakly supervised models, e.g., +24.87% on RefCOCO, but also show the 5x faster inference speed. Project: https://RefCLIP.github.io.

        ----

        ## [2] Megahertz Light Steering Without Moving Parts

        **Authors**: *Adithya Pediredla, Srinivasa G. Narasimhan, Maysamreza Chamanzar, Ioannis Gkioulekas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00009](https://doi.org/10.1109/CVPR52729.2023.00009)

        **Abstract**:

        We introduce a light steering technology that operates at megahertz frequencies, has no moving parts, and costs less than a hundred dollars. Our technology can benefit many projector and imaging systems that critically rely on high-speed, reliable, low-cost, and wavelength-independent light steering, including laser scanning projectors, LiDAR sensors, and fluorescence microscopes. Our technology uses ultrasound waves to generate a spatiotemporally-varying refractive index field inside a compressible medium, such as water, turning the medium into a dynamic traveling lens. By controlling the electrical input of the ultrasound transducers that generate the waves, we can change the lens, and thus steer light, at the speed of sound (1.5 km/s in water). We build a physical prototype of this technology, use it to realize different scanning techniques at megahertz rates (three orders of magnitude faster than commercial alternatives such as galvo mirror scanners), and demonstrate proof-of-concept projector and LiDAR applications. To encourage further innovation towards this new technology, we derive theory for its fundamental limits and develop a physically-accurate simulator for virtual design. Our technology offers a promising solution for achieving high-speed and low-cost light steering in a variety of applications.

        ----

        ## [3] Robust Dynamic Radiance Fields

        **Authors**: *Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, Jia-Bin Huang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00010](https://doi.org/10.1109/CVPR52729.2023.00010)

        **Abstract**:

        Dynamic radiance field reconstruction methods aim to model the time-varying structure and appearance of a dynamic scene. Existing methods, however, assume that accurate camera poses can be reliably estimated by Structure from Motion (SfM) algorithms. These methods, thus, are unreliable as SfM algorithms often fail or produce erroneous poses on challenging videos with highly dynamic objects, poorly textured surfaces, and rotating camera motion. We address this robustness issue by jointly estimating the static and dynamic radiance fields along with the camera parameters (poses and focal length). We demonstrate the robustness of our approach via extensive quantitative and qualitative experiments. Our results show favorable performance over the state-of-the-art dynamic view synthesis methods.

        ----

        ## [4] DBARF: Deep Bundle-Adjusting Generalizable Neural Radiance Fields

        **Authors**: *Yu Chen, Gim Hee Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00011](https://doi.org/10.1109/CVPR52729.2023.00011)

        **Abstract**:

        Recent works such as BARF and GARF can bundle adjust camera poses with neural radiance fields (NeRF) which is based on coordinate-MLPs. Despite the impressive results, these methods cannot be applied to Generalizable NeRFs (GeNeRFs) which require image feature extractions that are often based on more complicated 3D CNN or transformer architectures. In this work, we first analyze the difficulties of jointly optimizing camera poses with GeNeRFs, and then further propose our DBARF to tackle these issues. Our DBARF which bundle adjusts camera poses by taking a cost feature map as an implicit cost function can be jointly trained with GeNeRFs in a self-supervised manner. Unlike BARF and its follow-up works, which can only be applied to per-scene optimized NeRFs and need accurate initial camera poses with the exception of forward-facing scenes, our method can generalize across scenes and does not require any good initialization. Experiments show the effectiveness and generalization ability of our DBARF when evaluated on real-world datasets. Our code is available at https://aibluefisher.github.io/DBARF.

        ----

        ## [5] VDN-NeRF: Resolving Shape-Radiance Ambiguity via View-Dependence Normalization

        **Authors**: *Bingfan Zhu, Yanchao Yang, Xulong Wang, Youyi Zheng, Leonidas J. Guibas*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00012](https://doi.org/10.1109/CVPR52729.2023.00012)

        **Abstract**:

        We propose VDN-NeRF, a method to train neural radiance fields (NeRFs) for better geometry under non-Lambertian surface and dynamic lighting conditions that cause significant variation in the radiance of a point when viewed from different angles. Instead of explicitly modeling the underlying factors that result in the view-dependent phenomenon, which could be complex yet not inclusive, we develop a simple and effective technique that normalizes the view-dependence by distilling invariant information already encoded in the learned NeRFs. We then jointly train NeRFs for view synthesis with view-dependence normalization to attain quality geometry. Our experiments show that even though shape-radiance ambiguity is inevitable, the proposed normalization can minimize its effect on geometry, which essentially aligns the optimal capacity needed for explaining view-dependent variations. Our method applies to various baselines and significantly improves geometry without changing the volume rendering pipeline, even if the data is captured under a moving light source. Code is available at: https://github.com/BoifZ/VDN-NeRF.

        ----

        ## [6] AligNeRF: High-Fidelity Neural Radiance Fields via Alignment-Aware Training

        **Authors**: *Yifan Jiang, Peter Hedman, Ben Mildenhall, Dejia Xu, Jonathan T. Barron, Zhangyang Wang, Tianfan Xue*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00013](https://doi.org/10.1109/CVPR52729.2023.00013)

        **Abstract**:

        Neural Radiance Fields (NeRFs) are a powerful representation for modeling a 3D scene as a continuous function. Though NeRF is able to render complex 3D scenes with view-dependent effects, few efforts have been devoted to exploring its limits in a high-resolution setting. Specifically, existing NeRF-based methods face several limitations when reconstructing high-resolution real scenes, including a very large number of parameters, misaligned input data, and overly smooth details. In this work, we conduct the first pilot study on training NeRF with high-resolution data and propose the corresponding solutions: 1) marrying the multilayer perceptron (MLP) with convolutional layers which can encode more neighborhood information while reducing the total number of parameters; 2) a novel training strategy to address misalignment caused by moving objects or small camera calibration errors; and 3) a high-frequency aware loss. Our approach is nearly free without introducing obvious training/testing costs, while experiments on different datasets demonstrate that it can recover more high-frequency details compared with the current state-of-the-art NeRF models. Project page: https://yifanjiang19.github.io/alignerf.

        ----

        ## [7] SeaThru-NeRF: Neural Radiance Fields in Scattering Media

        **Authors**: *Deborah Levy, Amit Peleg, Naama Pearl, Dan Rosenbaum, Derya Akkaynak, Simon Korman, Tali Treibitz*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00014](https://doi.org/10.1109/CVPR52729.2023.00014)

        **Abstract**:

        Research on neural radiance fields (NeRFs) for novel view generation is exploding with new models and extensions. However, a question that remains unanswered is what happens in underwater or foggy scenes where the medium strongly influences the appearance of objects. Thus far, NeRF and its variants have ignored these cases. However, since the NeRF framework is based on volumetric rendering, it has inherent capability to account for the medium's effects, once modeled appropriately. We develop a new rendering model for NeRFs in scattering media, which is based on the SeaThru image formation model, and suggest a suitable architecture for learning both scene information and medium parameters. We demonstrate the strength of our method using simulated and real-world scenes, correctly rendering novel photorealistic views underwater. Even more excitingly, we can render clear views of these scenes, removing the medium between the camera and the scene and reconstructing the appearance and depth of far objects, which are severely occluded by the medium. Our code and unique datasets are available on the project's website.

        ----

        ## [8] Exact-NeRF: An Exploration of a Precise Volumetric Parameterization for Neural Radiance Fields

        **Authors**: *Brian K. S. Isaac-Medina, Chris G. Willcocks, Toby P. Breckon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00015](https://doi.org/10.1109/CVPR52729.2023.00015)

        **Abstract**:

        Neural Radiance Fields (NeRF) have attracted significant attention due to their ability to synthesize novel scene views with great accuracy. However, inherent to their underlying formulation, the sampling of points along a ray with zero width may result in ambiguous representations that lead to further rendering artifacts such as aliasing in the final scene. To address this issue, the recent variant mipNeRF proposes an Integrated Positional Encoding (IPE) based on a conical view frustum. Although this is expressed with an integral formulation, mip-NeRF instead approximates this integral as the expected value of a multivariate Gaussian distribution. This approximation is reliable for short frustums but degrades with highly elongated regions, which arises when dealing with distant scene objects under a larger depth of field. In this paper, we explore the use of an exact approach for calculating the IPE by using a pyramid-based integral formulation instead of an approximated conical-based one. We denote this formulation as Exact-NeRF and contribute the first approach to offer a precise analytical solution to the IPE within the NeRF domain. Our exploratory work illustrates that such an exact formulation (Exact-NeRF) matches the accuracy of mip-NeRF and furthermore provides a natural extension to more challenging scenarios without further modification, such as in the case of unbounded scenes. Our contribution aims to both address the hitherto unexplored issues of frustum approximation in earlier NeRF work and additionally provide insight into the potential future consideration of analytical solutions in future NeRF extensions.

        ----

        ## [9] Neural Residual Radiance Fields for Streamably Free-Viewpoint Videos

        **Authors**: *Liao Wang, Qiang Hu, Qihan He, Ziyu Wang, Jingyi Yu, Tinne Tuytelaars, Lan Xu, Minye Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00016](https://doi.org/10.1109/CVPR52729.2023.00016)

        **Abstract**:

        The success of the Neural Radiance Fields (NeRFs) for modeling and free-view rendering static objects has in-spired numerous attempts on dynamic scenes. Current techniques that utilize neural rendering for facilitating free-view videos (FVVs) are restricted to either offline rendering or are capable of processing only brief sequences with minimal motion. In this paper, we present a novel technique, Residual Radiance Field or ReRF, as a highly com-pact neural representation to achieve real-time FVV ren-dering on long-duration dynamic scenes. ReRF explicitly models the residual information between adjacent times-tamps in the spatial-temporal feature space, with a global coordinate-based tiny MLP as the feature decoder. Specif-ically, ReRF employs a compact motion grid along with a residual feature grid to exploit inter-frame feature similar-ities. We show such a strategy can handle large motions without sacrificing quality. We further present a sequential training scheme to maintain the smoothness and the spar-sity of the motion/residual grids. Based on ReRF, we design a special FVV codec that achieves three orders of magni-tudes compression rate and provides a companion ReRF player to support online streaming of long-duration FVVs of dynamic scenes. Extensive experiments demonstrate the effectiveness of ReRF for compactly representing dynamic radiance fields, enabling an unprecedented free-viewpoint viewing experience in speed and quality.

        ----

        ## [10] Plen-VDB: Memory Efficient VDB-Based Radiance Fields for Fast Training and Rendering

        **Authors**: *Han Yan, Celong Liu, Chao Ma, Xing Mei*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00017](https://doi.org/10.1109/CVPR52729.2023.00017)

        **Abstract**:

        In this paper, we present a new representation for neural radiance fields that accelerates both the training and the inference processes with VDB, a hierarchical data structure for sparse volumes. VDB takes both the advantages of sparse and dense volumes for compact data representation and efficient data access, being a promising data structure for NeRF data interpolation and ray marching. Our method, Plenoptic VDB (PlenVDB), directly learns the VDB data structure from a set of posed images by means of a novel training strategy and then uses it for real-time rendering. Experimental results demonstrate the effectiveness and the efficiency of our method over previous arts: First, it converges faster in the training process. Second, it delivers a more compact data format for NeRF data presentation. Finally, it renders more efficiently on commodity graphics hardware. Our mobile PlenVDB demo achieves 30+ FPS, 1280×720 resolution on an iPhone12 mobile phone. Check plenvdb.github.io for details.

        ----

        ## [11] Local Implicit Ray Function for Generalizable Radiance Field Representation

        **Authors**: *Xin Huang, Qi Zhang, Ying Feng, Xiaoyu Li, Xuan Wang, Qing Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00018](https://doi.org/10.1109/CVPR52729.2023.00018)

        **Abstract**:

        We propose LIRF (Local Implicit Ray Function), a generalizable neural rendering approach for novel view rendering. Current generalizable neural radiance fields (NeRF) methods sample a scene with a single ray per pixel and may therefore render blurred or aliased views when the input views and rendered views capture scene content with different resolutions. To solve this problem, we propose LIRF to aggregate the information from conical frustums to construct a ray. Given 3D positions within conical frustums, LIRF takes 3D coordinates and the features of conical frustums as inputs and predicts a local volumetric radiance field. Since the coordinates are continuous, LIRF renders high-quality novel views at a continuously-valued scale via volume rendering. Besides, we predict the visible weights for each input view via transformer-based feature matching to improve the performance in occluded areas. Experimental results on real-world scenes validate that our method outperforms state-of-the-art methods on novel view rendering of unseen scenes at arbitrary scales.

        ----

        ## [12] SurfelNeRF: Neural Surfel Radiance Fields for Online Photorealistic Reconstruction of Indoor Scenes

        **Authors**: *Yiming Gao, Yan-Pei Cao, Ying Shan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00019](https://doi.org/10.1109/CVPR52729.2023.00019)

        **Abstract**:

        Online reconstructing and rendering of large-scale indoor scenes is a long-standing challenge. SLAM-based methods can reconstruct 3D scene geometry progressively in real time but can not render photorealistic results. While NeRF-based methods produce promising novel view synthesis results, their long offline optimization time and lack of geometric constraints pose challenges to efficiently handling online input. Inspired by the complementary advantages of classical 3D reconstruction and NeRF, we thus investigate marrying explicit geometric representation with NeRF rendering to achieve efficient online reconstruction and high-quality rendering. We introduce SurfelNeRF, a variant of neural radiance field which employs a flexible and scalable neural surfel representation to store geometric attributes and extracted appearance features from input images. We further extend the conventional surfel-based fusion scheme to progressively integrate incoming input frames into the reconstructed global neural scene representation. In addition, we propose a highly-efficient differentiable rasterization scheme for rendering neural surfel radiance fields, which helps SurfelNeRF achieve 10× speedups in training and inference time, respectively. Experimental results show that our method achieves the state-of-the-art 23.82 PSNR and 29.58 PSNR on ScanNet in feedforward inference and perscene optimization settings, respectively.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project website: https://gymat.github.io/SurfelNeRF-web

        ----

        ## [13] Frequency-Modulated Point Cloud Rendering with Easy Editing

        **Authors**: *Yi Zhang, Xiaoyang Huang, Bingbing Ni, Wenjun Zhang, Teng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00020](https://doi.org/10.1109/CVPR52729.2023.00020)

        **Abstract**:

        We develop an effective point cloud rendering pipeline for novel view synthesis, which enables high fidelity local detail reconstruction, real-time rendering and user-friendly editing. In the heart of our pipeline is an adaptive frequency modulation module called Adaptive Frequency Net (AFNet), which utilizes a hypernetwork to learn the local texture frequency encoding that is consecutively injected into adaptive frequency activation layers to modulate the implicit radiance signal. This mechanism improves the frequency expressive ability of the network with richer frequency basis support, only at a small computational budget. To further boost performance, a preprocessing module is also proposed for point cloud geometry optimization via point opacity estimation. In contrast to implicit rendering, our pipeline supports high-fidelity interactive editing based on point cloud manipulation. Extensive experimental results on NeRF-Synthetic, ScanNet, DTU and Tanks and Temples datasets demonstrate the superior performances achieved by our method in terms of PSNR, SSIM and LPIPS, in comparison to the state-of-the-art. Code is released at https://github.com/yizhangphd/FreqPCR.

        ----

        ## [14] HexPlane: A Fast Representation for Dynamic Scenes

        **Authors**: *Ang Cao, Justin Johnson*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00021](https://doi.org/10.1109/CVPR52729.2023.00021)

        **Abstract**:

        Modeling and re-rendering dynamic 3D scenes is a challenging task in 3D vision. Prior approaches build on NeRF and rely on implicit representations. This is slow since it requires many MLP evaluations, constraining real-world applications. We show that dynamic 3D scenes can be explicitly represented by six planes of learned features, leading to an elegant solution we call HexPlane. A HexPlane computes features for points in spacetime by fusing vectors extracted from each plane, which is highly efficient. Pairing a HexPlane with a tiny MLP to regress output colors and training via volume rendering gives impressive results for novel view synthesis on dynamic scenes, matching the image quality of prior work but reducing training time by more than 100×. Extensive ablations confirm our HexPlane design and show that it is robust to different feature fusion mechanisms, coordinate systems, and decoding mechanisms. HexPlane is a simple and effective solution for representing 4D volumes, and we hope they can broadly contribute to modeling spacetime for dynamic 3D scenes.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Project page: https://caoang327.github.io/HexPlane.

        ----

        ## [15] Differentiable Shadow Mapping for Efficient Inverse Graphics

        **Authors**: *Markus Worchel, Marc Alexa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00022](https://doi.org/10.1109/CVPR52729.2023.00022)

        **Abstract**:

        We show how shadows can be efficiently generated in differentiable rendering of triangle meshes. Our central observation is that pre-filtered shadow mapping, a technique for approximating shadows based on rendering from the perspective of a light, can be combined with existing differentiable rasterizers to yield differentiable visibility information. We demonstrate at several inverse graphics problems that differentiable shadow maps are orders of magnitude faster than differentiable light transport simulation with similar accuracy - while differentiable rasterization without shadows often fails to converge.

        ----

        ## [16] Hybrid Neural Rendering for Large-Scale Scenes with Motion Blur

        **Authors**: *Peng Dai, Yinda Zhang, Xin Yu, Xiaoyang Lyu, Xiaojuan Qi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00023](https://doi.org/10.1109/CVPR52729.2023.00023)

        **Abstract**:

        Rendering novel view images is highly desirable for many applications. Despite recent progress, it remains challenging to render high-fidelity and view-consistent novel views of large-scale scenes from in-the-wild images with inevitable artifacts (e.g., motion blur). To this end, we develop a hybrid neural rendering model that makes image-based representation and neural 3D representation join forces to render high-quality, view-consistent images. Besides, images captured in the wild inevitably contain artifacts, such as motion blur, which deteriorates the quality of rendered images. Accordingly, we propose strategies to simulate blur effects on the rendered images to mitigate the negative influence of blurriness images and reduce their importance during training based on precomputed quality-aware weights. Extensive experiments on real and synthetic data demonstrate our model surpasses state-of-the-art point-based methods for novel view synthesis. The code is available at https://daipengwa.github.io/Hybrid-Rendering-ProjectPage/.

        ----

        ## [17] TensoIR: Tensorial Inverse Rendering

        **Authors**: *Haian Jin, Isabella Liu, Peijia Xu, Xiaoshuai Zhang, Songfang Han, Sai Bi, Xiaowei Zhou, Zexiang Xu, Hao Su*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00024](https://doi.org/10.1109/CVPR52729.2023.00024)

        **Abstract**:

        We propose TensoIR, a novel inverse rendering approach based on tensor factorization and neural fields. Unlike previous works that use purely MLP-based neural fields, thus suffering from low capacity and high computation costs, we extend TensoRF, a state-of-the-art approach for radiance field modeling, to estimate scene geometry, surface reflectance, and environment illumination from multi-view images captured under unknown lighting conditions. Our approach jointly achieves radiance field reconstruction and physically-based model estimation, leading to photo-realistic novel view synthesis and relighting results. Benefiting from the efficiency and extensibility of the TensoRF-based representation, our method can accurately model secondary shading effects (like shadows and indirect lighting) and generally support input images captured under single or multiple unknown lighting conditions. The low-rank tensor representation allows us to not only achieve fast and compact reconstruction but also better exploit shared information under an arbitrary number of capturing lighting conditions. We demonstrate the superiority of our method to baseline methods qualitatively and quantitatively on various challenging synthetic and real-world scenes.

        ----

        ## [18] ShadowNeuS: Neural SDF Reconstruction by Shadow Ray Supervision

        **Authors**: *Jingwang Ling, Zhibo Wang, Feng Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00025](https://doi.org/10.1109/CVPR52729.2023.00025)

        **Abstract**:

        By supervising camera rays between a scene and multi-view image planes, NeRF reconstructs a neural scene representation for the task of novel view synthesis. On the other hand, shadow rays between the light source and the scene have yet to be considered. Therefore, we propose a novel shadow ray supervision scheme that optimizes both the samples along the ray and the ray location. By supervising shadow rays, we successfully reconstruct a neural SDF of the scene from single-view images under multiple lighting conditions. Given single-view binary shadows, we train a neural network to reconstruct a complete scene not limited by the camera's line of sight. By further modeling the correlation between the image colors and the shadow rays, our technique can also be effectively extended to RGB inputs. We compare our method with previous works on challenging tasks of shape reconstruction from single-view binary shadow or RGB images and observe significant improvements. The code and data are available at https://github.com/gerwang/ShadowNeuS.

        ----

        ## [19] Realistic Saliency Guided Image Enhancement

        **Authors**: *S. Mahdi H. Miangoleh, Zoya Bylinskii, Eric Kee, Eli Shechtman, Yagiz Aksoy*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00026](https://doi.org/10.1109/CVPR52729.2023.00026)

        **Abstract**:

        Common editing operations performed by professional photographers include the cleanup operations: de-emphasizing distracting elements and enhancing subjects. These edits are challenging, requiring a delicate balance between manipulating the viewer's attention while maintaining photo realism. While recent approaches can boast successful examples of attention attenuation or amplification, most of them also suffer from frequent unrealistic edits. We propose a realism loss for saliency-guided image enhancement to maintain high realism across varying image types, while attenuating distractors and amplifying objects of interest. Evaluations with professional photographers confirm that we achieve the dual objective of realism and effectiveness, and outperform the recent approaches on their own datasets, while requiring a smaller memory footprint and runtime. We thus offer a viable solution for automating image enhancement and photo cleanup operations.

        ----

        ## [20] LightPainter: Interactive Portrait Relighting with Freehand Scribble

        **Authors**: *Yiqun Mei, He Zhang, Xuaner Zhang, Jianming Zhang, Zhixin Shu, Yilin Wang, Zijun Wei, Shi Yan, Hyunjoon Jung, Vishal M. Patel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00027](https://doi.org/10.1109/CVPR52729.2023.00027)

        **Abstract**:

        Recent portrait relighting methods have achieved realistic results of portrait lighting effects given a desired lighting representation such as an environment map. However, these methods are not intuitive for user interaction and lack precise lighting control. We introduce LightPainter, a scribble-based relighting system that allows users to interactively manipulate portrait lighting effect with ease. This is achieved by two conditional neural networks, a delighting module that recovers geometry and albedo optionally conditioned on skin tone, and a scribble-based module for re-lighting. To train the relighting module, we propose a novel scribble simulation procedure to mimic real user scribbles, which allows our pipeline to be trained without any human annotations. We demonstrate high-quality and flexible portrait lighting editing capability with both quantitative and qualitative experiments. User study comparisons with commercial lighting editing tools also demonstrate consistent user preference for our method.

        ----

        ## [21] A Unified Spatial-Angular Structured Light for Single-View Acquisition of Shape and Reflectance

        **Authors**: *Xianmin Xu, Yuxin Lin, Haoyang Zhou, Chong Zeng, Yaxin Yu, Kun Zhou, Hongzhi Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00028](https://doi.org/10.1109/CVPR52729.2023.00028)

        **Abstract**:

        We propose a unified structured light, consisting of an LED array and an LCD mask, for high-quality acquisition of both shape and reflectance from a single view. For geometry, one LED projects a set of learned mask patterns to accurately encode spatial information; the decoded results from multiple LEDs are then aggregated to produce a final depth map. For appearance, learned light patterns are cast through a transparent mask to efficiently probe angularly-varying reflectance. Per-point BRDF parameters are differentiably optimized with respect to corresponding measurements, and stored in texture maps as the final reflectance. We establish a differentiable pipeline for the joint capture to automatically optimize both the mask and light patterns towards optimal acquisition quality. The effectiveness of our light is demonstrated with a wide variety of physical objects. Our results compare favorably with state-of-the-art techniques.

        ----

        ## [22] Learning Visibility Field for Detailed 3D Human Reconstruction and Relighting

        **Authors**: *Ruichen Zheng, Peng Li, Haoqian Wang, Tao Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00029](https://doi.org/10.1109/CVPR52729.2023.00029)

        **Abstract**:

        Detailed 3D reconstruction and photo-realistic relighting of digital humans are essential for various applications. To this end, we propose a novel sparse-view 3d human reconstruction framework that closely incorporates the occupancy field and albedo field with an additional visibility field-it not only resolves occlusion ambiguity in multi-view feature aggregation, but can also be used to evaluate light attenuation for self-shadowed relighting. To enhance its training viability and efficiency, we discretize visibility onto a fixed set of sample directions and supply it with coupled geometric 3D depth feature and local 2D image feature. We further propose a novel rendering-inspired loss, namely TransferLoss, to implicitly enforce the alignment between visibility and occupancy field, enabling end-to-end joint training. Results and extensive experiments demonstrate the effectiveness of the proposed method, as it surpasses state-of-the-art in terms of reconstruction accuracy while achieving comparably accurate relighting to ray-traced ground truth.

        ----

        ## [23] Unsupervised Contour Tracking of Live Cells by Mechanical and Cycle Consistency Losses

        **Authors**: *Junbong Jang, Kwonmoo Lee, Tae-Kyun Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00030](https://doi.org/10.1109/CVPR52729.2023.00030)

        **Abstract**:

        Analyzing the dynamic changes of cellular morphology is important for understanding the various functions and characteristics of live cells, including stem cells and metastatic cancer cells. To this end, we need to track all points on the highly deformable cellular contour in every frame of live cell video. Local shapes and textures on the contour are not evident, and their motions are complex, often with expansion and contraction of local contour features. The prior arts for optical flow or deep point set tracking are unsuited due to the fluidity of cells, and previous deep contour tracking does not consider point correspondence. We propose the first deep learning-based tracking of cellular (or more generally viscoelastic materials) contours with point correspondence by fusing dense representation between two contours with cross attention. Since it is impractical to manually label dense tracking points on the contour, unsupervised learning comprised of the mechanical and cyclical consistency losses is proposed to train our contour tracker. The mechanical loss forcing the points to move perpendicular to the contour effectively helps out. For quantitative evaluation, we labeled sparse tracking points along the contour of live cells from two live cell datasets taken with phase contrast and confocal fluorescence microscopes. Our contour tracker quantitatively outperforms compared methods and produces qualitatively more favorable results. Our code and data are publicly available at https://github.com/JunbongJang/contour-tracking/

        ----

        ## [24] NeUDF: Leaning Neural Unsigned Distance Fields with Volume Rendering

        **Authors**: *Yu-Tao Liu, Li Wang, Jie Yang, Weikai Chen, Xiaoxu Meng, Bo Yang, Lin Gao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00031](https://doi.org/10.1109/CVPR52729.2023.00031)

        **Abstract**:

        Multi-view shape reconstruction has achieved impressive progresses thanks to the latest advances in neural implicit surface rendering. However, existing methods based on signed distance function (SDF) are limited to closed surfaces, failing to reconstruct a wide range of real-world objects that contain open-surface structures. In this work, we introduce a new neural rendering framework, coded NeUDF
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Visit our project page at http://geometrylearning.com/neudf/, that can reconstruct surfaces with arbitrary topologies solely from multi-view supervision. To gain the flexibility of representing arbitrary surfaces, NeUDF leverages the unsigned distance function (UDF) as surface representation. While a naive extension of an SDF-based neural renderer cannot scale to UDF, we propose two new formulations of weight function specially tailored for UDF-based volume rendering. Furthermore, to cope with open surface rendering, where the in/out test is no longer valid, we present a dedicated normal regularization strategy to resolve the surface orientation ambiguity. We extensively evaluate our method over a number of challenging datasets, including DTU [21], MGN [5], and Deep Fashion 3D [61]. Experimental results demonstrate that NeUDF can significantly outperform the state-of-the-art method in the task of multi-view surface reconstruction, especially for complex shapes with open boundaries.

        ----

        ## [25] NeAT: Learning Neural Implicit Surfaces with Arbitrary Topologies from Multi-View Images

        **Authors**: *Xiaoxu Meng, Weikai Chen, Bo Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00032](https://doi.org/10.1109/CVPR52729.2023.00032)

        **Abstract**:

        Recent progress in neural implicit functions has set new state-of-the-art in reconstructing high-fidelity 3D shapes from a collection of images. However, these approaches are limited to closed surfaces as they require the surface to be represented by a signed distance field. In this paper, we propose NeAT, a new neural rendering framework that can learn implicit surfaces with arbitrary topologies from multi-view images. In particular, NeAT represents the 3D surface as a level set of a signed distance function (SDF) with a validity branch for estimating the surface existence probability at the query positions. We also develop a novel neural volume rendering method, which uses SDF and validity to calculate the volume opacity and avoids rendering points with low validity. NeAT supports easy field-to-mesh conversion using the classic Marching Cubes algorithm. Extensive experiments on DTU [20], MGN [4], and Deep Fashion 3D [19] datasets indicate that our approach is able to faithfully reconstruct both watertight and non-watertight surfaces. In particular, NeAT significantly outperforms the state-of-the-art methods in the task of open surface reconstruction both quantitatively and qualitatively.

        ----

        ## [26] ALTO: Alternating Latent Topologies for Implicit 3D Reconstruction

        **Authors**: *Zhen Wang, Shijie Zhou, Jeong Joon Park, Despoina Paschalidou, Suya You, Gordon Wetzstein, Leonidas J. Guibas, Achuta Kadambi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00033](https://doi.org/10.1109/CVPR52729.2023.00033)

        **Abstract**:

        This work introduces alternating latent topologies (ALTO) for high-fidelity reconstruction of implicit 3D surfaces from noisy point clouds. Previous work identifies that the spatial arrangement of latent encodings is important to recover detail. One school of thought is to encode a latent vector for each point (point latents). Another school of thought is to project point latents into a grid (grid latents) which could be a voxel grid or triplane grid. Each school of thought has tradeoffs. Grid latents are coarse and lose high-frequency detail. In contrast, point latents preserve detail. However, point latents are more difficult to decode into a surface, and quality and runtime suffer. In this paper, we propose ALTO to sequentially alternate between geometric representations, before converging to an easy-to-decode latent. We find that this preserves spatial expressiveness and makes decoding lightweight. We validate ALTO on implicit 3D recovery and observe not only a performance improvement over the state-of-the-art, but a runtime improvement of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$3-10\times$</tex>
. Project website at https://visual.ee.ucla.edu/alto.htm/.

        ----

        ## [27] Controllable Mesh Generation Through Sparse Latent Point Diffusion Models

        **Authors**: *Zhaoyang Lyu, Jinyi Wang, Yuwei An, Ya Zhang, Dahua Lin, Bo Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00034](https://doi.org/10.1109/CVPR52729.2023.00034)

        **Abstract**:

        Mesh generation is of great value in various applications involving computer graphics and virtual content, yet designing generative models for meshes is challenging due to their irregular data structure and inconsistent topology of meshes in the same category. In this work, we design a novel sparse latent point diffusion model for mesh generation. Our key insight is to regard point clouds as an intermediate representation of meshes, and model the distribution of point clouds instead. While meshes can be generated from point clouds via techniques like Shape as Points (SAP), the challenges of directly generating meshes can be effectively avoided. To boost the efficiency and controllability of our mesh generation method, we propose to further encode point clouds to a set of sparse latent points with pointwise semantic meaningful features, where two DDPMs are trained in the space of sparse latent points to respectively model the distribution of the latent point positions and features at these latent points. We find that sampling in this latent space is faster than directly sampling dense point clouds. Moreover, the sparse latent points also enable us to explicitly control both the overall structures and local details of the generated meshes. Extensive experiments are conducted on the ShapeNet dataset, where our proposed sparse latent point diffusion model achieves superior performance in terms of generation quality and controllability when compared to existing methods. Project page, code and appendix: https://slide-3d.github.io.

        ----

        ## [28] Photo Pre-Training, But for Sketch

        **Authors**: *Ke Li, Kaiyue Pang, Yi-Zhe Song*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00270](https://doi.org/10.1109/CVPR52729.2023.00270)

        **Abstract**:

        The sketch community has faced up to its unique challenges over the years, that of data scarcity however still remains the most significant to date. This lack of sketch data has imposed on the community a few “peculiar” design choices - the most representative of them all is perhaps the coerced utilisation of photo-based pre-training (i.e., no sketch), for many core tasks that otherwise dictates specific sketch understanding. In this paper, we ask just the one question - can we make such photo-based pre-training, to actually benefit sketch? Our answer lies in cultivating the topology of photo data learned at pre-training, and use that as a “free” source of supervision for downstream sketch tasks. In particular, we use fine-grained sketch-based image retrieval (FG-SBIR), one of the most studied and data-hungry sketch tasks, to showcase our new perspective on pre-training. In this context, the topology-informed supervision learned from photos act as a constraint that take effect at every fine-tuning step - neighbouring photos in the pre-trained model remain neighbours under each FG-SBIR updates. We further portray this neighbourhood consistency constraint as a photo ranking problem and formulate it into a neat cross-modal triplet loss. We also show how this target is better leveraged as a meta objective rather than optimised in parallel with the main FG-SBIR objective. With just this change on pre-training, we beat all previously published results on all five product-level FG-SBIR benchmarks with significant margins (sometimes >10%). And the most beautiful thing, as we note, is such gigantic leap is made possible within just a few extra lines of code! Our implementation is available at https://github.com/KeLi-SketchX/Photo-Pre-Training-But-for-Sketch

        ----

        ## [29] Power Bundle Adjustment for Large-Scale 3D Reconstruction

        **Authors**: *Simon Weber, Nikolaus Demmel, Tin Chon Chan, Daniel Cremers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00035](https://doi.org/10.1109/CVPR52729.2023.00035)

        **Abstract**:

        We introduce Power Bundle Adjustment as an expansion type algorithm for solving large-scale bundle adjustment problems. It is based on the power series expansion of the inverse Schur complement and constitutes a new family of solvers that we call inverse expansion methods. We theoretically justify the use of power series and we prove the convergence of our approach. Using the real-world BAL dataset we show that the proposed solver challenges the state-of-the-art iterative methods and significantly accelerates the solution of the normal equation, even for reaching a very high accuracy. This easy-to-implement solver can also complement a recently presented distributed bundle adjustment framework. We demonstrate that employing the proposed Power Bundle Adjustment as a subproblem solver significantly improves speed and accuracy of the distributed optimization.

        ----

        ## [30] Neural Pixel Composition for 3D-4D View Synthesis from Multi-Views

        **Authors**: *Aayush Bansal, Michael Zollhöfer*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00036](https://doi.org/10.1109/CVPR52729.2023.00036)

        **Abstract**:

        We present Neural Pixel Composition (NPC), a novel approach for continuous 3D-4D view synthesis given a discrete set of multi-view observations as input. Existing state-of-the-art approaches require dense multi-view supervision and an extensive computational budget. The proposed formulation reliably operates on sparse and wide-baseline multi-view images/videos and can be trained efficiently within a few seconds to 10 minutes for hi-res (12MP) content. Crucial to our approach are two core novelties: 1) a representation of a pixel that contains color and depth information accumulated from multi-views for a particular location and time along a line of sight, and 2) a multi-layer perceptron (MLP) that enables the composition of this rich information provided for a pixel location to obtain the final color output. We experiment with a large variety of multi-view sequences, compare to existing approaches, and achieve better results in diverse and challenging settings.

        ----

        ## [31] Magic3D: High-Resolution Text-to-3D Content Creation

        **Authors**: *Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, Tsung-Yi Lin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00037](https://doi.org/10.1109/CVPR52729.2023.00037)

        **Abstract**:

        DreamFusion [31] has recently demonstrated the utility of a pretrained text-to-image diffusion model to optimize Neural Radiance Fields (NeRF) [23], achieving remarkable text-to-3D synthesis results. However, the method has two inherent limitations: (a) extremely slow optimization of NeRF and (b) low-resolution image space supervision on NeRF, leading to low-quality 3D models with a long processing time. In this paper, we address these limitations by utilizing a two-stage optimization framework. First, we obtain a coarse model using a low-resolution diffusion prior and accelerate with a sparse 3D hash grid structure. Using the coarse representation as the initialization, we further optimize a textured 3D mesh model with an efficient differentiable renderer interacting with a high-resolution latent diffusion model. Our method, dubbed Magic3D, can create high quality 3D mesh models in 40 minutes, which is 2× faster than DreamFusion (reportedly taking 1.5 hours on average), while also achieving higher resolution. User studies show 61.7% raters to prefer our approach over DreamFusion. Together with the image-conditioned generation capabilities, we provide users with new ways to control 3D synthesis, opening up new avenues to various creative applications.

        ----

        ## [32] 3D Video Loops from Asynchronous Input

        **Authors**: *Li Ma, Xiaoyu Li, Jing Liao, Pedro V. Sander*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00038](https://doi.org/10.1109/CVPR52729.2023.00038)

        **Abstract**:

        Looping videos are short video clips that can be looped endlessly without visible seams or artifacts. They provide a very attractive way to capture the dynamism of natural scenes. Existing methods have been mostly limited to 2D representations. In this paper, we take a step forward and propose a practical solution that enables an immersive experience on dynamic 3D looping scenes. The key challenge is to consider the per-view looping conditions from asynchronous input while maintaining view consistency for the 3D representation. We propose a novel sparse 3D video representation, namely Multi-Tile Video (MTV), which not only provides a view-consistent prior, but also greatly reduces memory usage, making the optimization of a 4D volume tractable. Then, we introduce a two-stage pipeline to construct the 3D looping MTV from completely asynchronous multi-view videos with no time overlap. A novel looping loss based on video temporal retargeting algorithms is adopted during the optimization to loop the 3D scene. Experiments of our framework have shown promise in successfully generating and rendering photorealistic 3D looping videos in real time even on mobile devices. The code, dataset, and live demos are available in https://limacv.github.io/VideoLoop3D_web/.

        ----

        ## [33] High-fidelity 3D GAN Inversion by Pseudo-multi-view Optimization

        **Authors**: *Jiaxin Xie, Hao Ouyang, Jingtan Piao, Chenyang Lei, Qifeng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00039](https://doi.org/10.1109/CVPR52729.2023.00039)

        **Abstract**:

        We present a high-fidelity 3D generative adversarial network (GAN) inversion framework that can synthesize photorealistic novel views while preserving specific details of the input image. High-fidelity 3D GAN inversion is inherently challenging due to the geometry-texture trade-off, where overfitting to a single view input image often damages the estimated geometry during the latent optimization. To solve this challenge, we propose a novel pipeline that builds on the pseudo-multi-view estimation with visibility analysis. We keep the original textures for the visible parts and utilize generative priors for the occluded parts. Extensive experiments show that our approach achieves advantageous reconstruction and novel view synthesis quality over prior work, even for images with out-of-distribution textures. The proposed pipeline also enables image attribute editing with the inverted latent code and 3D-aware texture modification. Our approach enables high-fidelity 3D rendering from a single image, which is promising for various applications of AI-generated 3D content. The source code is at https://github.com/jiaxinxie97/HFGI3D/.

        ----

        ## [34] Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field

        **Authors**: *Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Yingcong Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00040](https://doi.org/10.1109/CVPR52729.2023.00040)

        **Abstract**:

        This work explores the use of 3D generative models to synthesize training data for 3D vision tasks. The key requirements of the generative models are that the generated data should be photorealistic to match the real-world scenarios, and the corresponding 3D attributes should be aligned with given sampling labels. However, we find that the recent NeRF-based 3D GANs hardly meet the above requirements due to their designed generation pipeline and the lack of explicit 3D supervision. In this work, we propose Lift3D, an inverted 2D-to-3D generation framework to achieve the data generation objectives. Lift3D has several merits compared to prior methods: (1) Unlike previous 3D GANs that the output resolution is fixed after training, Lift3D can generalize to any camera intrinsic with higher resolution and photorealistic output. (2) By lifting well-disentangled 2D GAN to 3D object NeRF, Lift3D provides explicit 3D information of generated objects, thus offering accurate 3D annotations for downstream tasks. We evaluate the effectiveness of our framework by augmenting autonomous driving datasets. Experimental results demonstrate that our data generation framework can effectively improve the performance of 3D object detectors. Code: len-li.github.io/lift3d-web

        ----

        ## [35] 3D GAN Inversion with Facial Symmetry Prior

        **Authors**: *Fei Yin, Yong Zhang, Xuan Wang, Tengfei Wang, Xiaoyu Li, Yuan Gong, Yanbo Fan, Xiaodong Cun, Ying Shan, Cengiz Öztireli, Yujiu Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00041](https://doi.org/10.1109/CVPR52729.2023.00041)

        **Abstract**:

        Recently, a surge of high-quality 3D-aware GANs have been proposed, which leverage the generative power of neural rendering. It is natural to associate 3D GANs with GAN inversion methods to project a real image into the generator's latent space, allowing free-view consistent synthesis and editing, referred as 3D GAN inversion. Although with the facial prior preserved in pre-trained 3D GANs, reconstructing a 3D portrait with only one monocular image is still an ill-pose problem. The straightforward application of 2D GAN inversion methods focuses on texture similarity only while ignoring the correctness of 3D geometry shapes. It may raise geometry collapse effects, especially when reconstructing a side face under an extreme pose. Besides, the synthetic results in novel views are prone to be blurry. In this work, we propose a novel method to promote 3D GAN inversion by introducing facial symmetry prior. We design a pipeline and constraints to make full use of the pseudo auxiliary view obtained via image flipping, which helps obtain a view-consistent and well-structured geometry shape during the inversion process. To enhance texture fidelity in unobserved viewpoints, pseudo labels from depth-guided 3D warping can provide extra supervision. We design constraints to filter out conflict areas for optimization in asymmetric situations. Comprehensive quantitative and qualitative evaluations on image reconstruction and editing demonstrate the superiority of our method.

        ----

        ## [36] StyleIPSB: Identity-Preserving Semantic Basis of StyleGAN for High Fidelity Face Swapping

        **Authors**: *Diqiong Jiang, Dan Song, Ruofeng Tong, Min Tang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00042](https://doi.org/10.1109/CVPR52729.2023.00042)

        **Abstract**:

        Recent researches reveal that StyleGAN can generate highly realistic images, inspiring researchers to use pretrained StyleGAN to generate high-fidelity swapped faces. However, existing methods fail to meet the expectations in two essential aspects of high-fidelity face swapping. Their results are blurry without pore-level details and fail to preserve identity for challenging cases. To overcome the above artifacts, we innovatively construct a series of identity-preserving semantic bases of StyleGAN (called StyleIPSB) in respect of pose, expression, and illumination. Each basis of StyleIPSB controls one specific semantic attribute and disentangles with the others. The StyleIPSB constrains style code in the subspace of W+ space to preserve pore-level details and gives us a novel tool for high-fidelity face swapping, and we propose a three-stage framework for face swapping with StyleIPSB. Firstly, we transform the target facial images' attributes to the source image. We learn the mapping from 3D Morphable Model (3DMM) parameters, which capture the prominent semantic variance, to the coordinates of StyleIPSB that show higher identity-preserving and fidelity. Secondly, to transform detailed attributes which 3DMM does not capture, we learn the residual attribute between the reenacted face and the target face. Finally, the face is blended into the background of the target image. Extensive results and comparisons demonstrate that StyleIPSB can effectively preserve identity and pore-level details. The results of face swapping can achieve state-of-the-art performance. We will release our code at https://github.com/a686432/StyleIPSB

        ----

        ## [37] FFHQ-UV: Normalized Facial UV-Texture Dataset for 3D Face Reconstruction

        **Authors**: *Haoran Bai, Di Kang, Haoxian Zhang, Jinshan Pan, Linchao Bao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00043](https://doi.org/10.1109/CVPR52729.2023.00043)

        **Abstract**:

        We present a large-scale facial UV-texture dataset that contains over 50,000 high-quality texture UV-maps with even illuminations, neutral expressions, and cleaned facial regions, which are desired characteristics for rendering realistic 3D face models under different lighting conditions. The dataset is derived from a large-scale face image dataset namely FFHQ, with the help of our fully automatic and robust UV-texture production pipeline. Our pipeline utilizes the recent advances in StyleGAN-based facial image editing approaches to generate multi-view normalized face images from single-image inputs. An elaborated UV-texture extraction, correction, and completion procedure is then applied to produce high-quality UV-maps from the normalized face images. Compared with existing UV-texture datasets, our dataset has more diverse and higher-quality texture maps. We further train a GAN-based texture decoder as the nonlinear texture basis for parametric fitting based 3D face reconstruction. Experiments show that our method improves the reconstruction accuracy over state-of-the-art approaches, and more importantly, produces high-quality texture maps that are ready for realistic renderings. The dataset, code, and pre-trained texture decoder are publicly available at https://github.com/csbhr/FFHQ-UV.

        ----

        ## [38] Robust Model-based Face Reconstruction through Weakly-Supervised Outlier Segmentation

        **Authors**: *Chunlu Li, Andreas Morel-Forster, Thomas Vetter, Bernhard Egger, Adam Kortylewski*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00044](https://doi.org/10.1109/CVPR52729.2023.00044)

        **Abstract**:

        In this work, we aim to enhance model-based face reconstruction by avoiding fitting the model to outliers, i.e. regions that cannot be well-expressed by the model such as occluders or makeup. The core challenge for localizing outliers is that they are highly variable and difficult to annotate. To overcome this challenging problem, we introduce a joint Face-autoencoder and outlier segmentation approach (FOCUS). In particular, we exploit the fact that the outliers cannot be fitted well by the face model and hence can be localized well given a high-quality model fitting. The main challenge is that the model fitting and the outlier segmentation are mutually dependent on each other, and need to be inferred jointly. We resolve this chicken-and-egg problem with an EM-type training strategy, where a face autoencoder is trained jointly with an outlier segmentation network. This leads to a synergistic effect, in which the segmentation network prevents the face encoder from fitting to the outliers, enhancing the reconstruction quality. The improved 3D face reconstruction, in turn, enables the segmentation network to better predict the outliers. To resolve the ambiguity between outliers and regions that are difficult to fit, such as eyebrows, we build a statistical prior from synthetic data that measures the systematic bias in model fitting. Experiments on the NoW testset demonstrate that FOCUS achieves SOTA 3D face reconstruction performance among all baselines trained without 3D annotation. Moreover, our results on CelebA-HQ and AR database show that the segmentation network can localize occluders accurately despite being trained without any segmentation annotation.

        ----

        ## [39] Learning Neural Proto-Face Field for Disentangled 3D Face Modeling in the Wild

        **Authors**: *Zhenyu Zhang, Renwang Chen, Weijian Cao, Ying Tai, Chengjie Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00045](https://doi.org/10.1109/CVPR52729.2023.00045)

        **Abstract**:

        Generative models show good potential for recovering 3D faces beyond limited shape assumptions. While plausible details and resolutions are achieved, these models easily fail under extreme conditions of pose, shadow or appearance, due to the entangled fitting or lack of multi-view priors. To address this problem, this paper presents a novel Neural Proto-face Field (NPF) for unsupervised robust 3D face modeling. Instead of using constrained images as Neural Radiance Field (NeRF), NPF disentangles the common/specific facial cues, i.e., ID, expression and scene-specific details from in-the-wild photo collections. Specifically, NPF learns a face prototype to aggregate 3D-consistent identity via uncertainty modeling, extracting multi-image priors from a photo collection. NPF then learns to deform the prototype with the appropriate facial expressions, constrained by a loss of expression consistency and personal idiosyncrasies. Finally, NPF is optimized to fit a target image in the collection, recovering specific details of appearance and geometry. In this way, the generative model benefits from multi-image priors and meaningful facial structures. Extensive experiments on benchmarks show that NPF recovers superior or competitive facial shapes and textures, compared to state-of-the-art methods.

        ----

        ## [40] A Hierarchical Representation Network for Accurate and Detailed Face Reconstruction from In-The-Wild Images

        **Authors**: *Biwen Lei, Jianqiang Ren, Mengyang Feng, Miaomiao Cui, Xuansong Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00046](https://doi.org/10.1109/CVPR52729.2023.00046)

        **Abstract**:

        Limited by the nature of the low-dimensional representational capacity of 3DMM, most of the 3DMM-based face reconstruction (FR) methods fail to recover high-frequency facial details, such as wrinkles, dimples, etc. Some attempt to solve the problem by introducing detail maps or nonlinear operations, however, the results are still not vivid. To this end, we in this paper present a novel hierarchical representation network (HRN) to achieve accurate and detailed face reconstruction from a single image. Specifically, we implement the geometry disentanglement and introduce the hierarchical representation to fulfill detailed face modeling. Meanwhile, 3D priors of facial details are incorporated to enhance the accuracy and authenticity of the reconstruction results. We also propose a deretouching module to achieve better decoupling of the geometry and appearance. It is noteworthy that our framework can be extended to a multi-view fashion by considering detail consistency of different views. Extensive experiments on two single-view and two multi-view FR benchmarks demonstrate that our method outperforms the existing methods in both reconstruction accuracy and visual effects. Finally, we introduce a high-quality 3D face dataset FaceHD-100 to boost the research of high-fidelity face reconstruction. The project homepage is at https://younglbw.github.io/HRN-homepage/.

        ----

        ## [41] BlendFields: Few-Shot Example-Driven Facial Modeling

        **Authors**: *Kacper Kania, Stephan J. Garbin, Andrea Tagliasacchi, Virginia Estellers, Kwang Moo Yi, Julien Valentin, Tomasz Trzcinski, Marek Kowalski*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00047](https://doi.org/10.1109/CVPR52729.2023.00047)

        **Abstract**:

        Generating faithful visualizations of human faces requires capturing both coarse and fine-level details of the face geometry and appearance. Existing methods are either data-driven, requiring an extensive corpus of data not publicly accessible to the research community, or fail to capture fine details because they rely on geometric face models that cannot represent fine-grained details in texture with a mesh discretization and linear deformation designed to model only a coarse face geometry. We introduce a method that bridges this gap by drawing inspiration from traditional computer graphics techniques. Unseen expressions are modeled by blending appearance from a sparse set of extreme poses. This blending is performed by measuring local volumetric changes in those expressions and locally reproducing their appearance whenever a similar expression is performed at test time. We show that our method generalizes to unseen expressions, adding fine-grained effects on top of smooth volumetric deformations of a face, and demonstrate how it generalizes beyond faces.

        ----

        ## [42] Implicit Neural Head Synthesis via Controllable Local Deformation Fields

        **Authors**: *Chuhan Chen, Matthew O'Toole, Gaurav Bharaj, Pablo Garrido*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00048](https://doi.org/10.1109/CVPR52729.2023.00048)

        **Abstract**:

        High-quality reconstruction of controllable 3D head avatars from 2D videos is highly desirable for virtual human applications in movies, games, and telepresence. Neural implicit fields provide a powerful representation to model 3D head avatars with personalized shape, expressions, and facial parts, e.g., hair and mouth interior, that go beyond the linear 3D morphable model (3DMM). However, existing methods do not model faces with fine-scale facial features, or local control of facial parts that extrapolate asymmetric expressions from monocular videos. Further, most condition only on 3DMM parameters with poor(er) locality, and resolve local features with a global neural field. We build on part-based implicit shape models that decompose a global deformation field into local ones. Our novel formulation models multiple implicit deformation fields with local semantic rig-like control via 3DMM-based parameters, and representative facial landmarks. Further, we propose a local controlloss and attention mask mechanism that promote sparsity of each learned deformation field. Our formulation renders sharper locally controllable nonlinear deformations than previous implicit monocular approaches, especially mouth interior, asymmetric expressions, and facial details. Project page: https://imaging.cs.cmu.edullocal_deformation_fieldsl

        ----

        ## [43] DPE: Disentanglement of Pose and Expression for General Video Portrait Editing

        **Authors**: *Youxin Pang, Yong Zhang, Weize Quan, Yanbo Fan, Xiaodong Cun, Ying Shan, Dong-Ming Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00049](https://doi.org/10.1109/CVPR52729.2023.00049)

        **Abstract**:

        One-shot video-driven talking face generation aims at producing a synthetic talking video by transferring the facial motion from a video to an arbitrary portrait image. Head pose and facial expression are always entangled in facial motion and transferred simultaneously. However, the entanglement sets up a barrier for these methods to be used in video portrait editing directly, where it may require to modify the expression only while maintaining the pose unchanged. One challenge of decoupling pose and expression is the lack of paired data, such as the same pose but different expressions. Only a few methods attempt to tackle this challenge with the feat of 3D Morphable Models (3DMMs) for explicit disentanglement. But 3DMMs are not accurate enough to capture facial details due to the limited number of Blend-shapes, which has side effects on motion transfer. In this paper, we introduce a novel self-supervised disentanglement framework to decouple pose and expression without 3DMMs and paired data, which consists of a motion editing module, a pose generator, and an expression generator. The editing module projects faces into a latent space where pose motion and expression motion can be disentangled, and the pose or expression transfer can be performed in the latent space conveniently via addition. The two generators render the modified latent codes to images, respectively. Moreover, to guarantee the disentanglement, we propose a bidirectional cyclic training strategy with well-designed constraints. Evaluations demonstrate our method can control pose or expression independently and be used for general video editing. Code: https://github.com/Carlyx/DPE

        ----

        ## [44] GANHead: Towards Generative Animatable Neural Head Avatars

        **Authors**: *Sijing Wu, Yichao Yan, Yunhao Li, Yuhao Cheng, Wenhan Zhu, Ke Gao, Xiaobo Li, Guangtao Zhai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00050](https://doi.org/10.1109/CVPR52729.2023.00050)

        **Abstract**:

        To bring digital avatars into people's lives, it is highly demanded to efficiently generate complete, realistic, and animatable head avatars. This task is challenging, and it is difficult for existing methods to satisfy all the requirements at once. To achieve these goals, we propose GANHead (Generative Animatable Neural Head Avatar), a novel generative head model that takes advantages of both the fine-grained control over the explicit expression parameters and the realistic rendering results of implicit representations. Specifically, GANHead represents coarse geometry, fine-gained details and texture via three networks in canonical space to obtain the ability to generate complete and realistic head avatars. To achieve flexible animation, we define the deformation filed by standard linear blend skinning (LBS), with the learned continuous pose and expression bases and LBS weights. This allows the avatars to be directly animated by FLAME [22] parameters and generalize well to unseen poses and expressions. Compared to state-of-the-art (SOTA) methods, GANHead achieves superior performance on head avatar generation and raw scan fitting.

        ----

        ## [45] EDGE: Editable Dance Generation From Music

        **Authors**: *Jonathan Tseng, Rodrigo Castellon, C. Karen Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00051](https://doi.org/10.1109/CVPR52729.2023.00051)

        **Abstract**:

        Dance is an important human art form, but creating new dances can be difficult and time-consuming. In this work, we introduce Editable Dance GEneration (EDGE), a state-of-the-art method for editable dance generation that is capable of creating realistic, physically-plausible dances while remaining faithful to the input music. EDGE uses a transformer-based diffusion model paired with Jukebox, a strong music feature extractor, and confers powerful editing capabilities well-suited to dance, including joint-wise conditioning, and in-betweening. We introduce a new metric for physical plausibility, and evaluate dance quality generated by our method extensively through (1) multiple quantitative metrics on physical plausibility, beat alignment, and diversity benchmarks, and more importantly, (2) a large-scale user study, demonstrating a significant improvement over previous state-of-the-art methods. Qualitative samples from our model can be found at our website.

        ----

        ## [46] Unsupervised Volumetric Animation

        **Authors**: *Aliaksandr Siarohin, Willi Menapace, Ivan Skorokhodov, Kyle Olszewski, Jian Ren, Hsin-Ying Lee, Menglei Chai, Sergey Tulyakov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00452](https://doi.org/10.1109/CVPR52729.2023.00452)

        **Abstract**:

        We propose a novel approach for unsupervised 3D animation of non-rigid deformable objects. Our method learns the 3D structure and dynamics of objects solely from single-view RGB videos, and can decompose them into semantically meaningful parts that can be tracked and animated. Using a 3D autodecoder framework, paired with a keypoint estimator via a differentiable PnP algorithm, our model learns the underlying object geometry and parts decomposition in an entirely unsupervised manner. This allows it to perform 3D segmentation, 3D keypoint estimation, novel view synthesis, and animation. We primarily evaluate the framework on two video datasets: VoxCeleb 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 and TEDXPeople 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
. In addition, on the Cats 256
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 image dataset, we show it even learns compelling 3D geometry from still images. Finally, we show our model can obtain animatable 3D objects from a single or few images 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code and visual results available on our project website: https://snap-research.github.io/unsupervised-volumetric-animation..

        ----

        ## [47] Blowing in the Wind: CycleNet for Human Cinemagraphs from Still Images

        **Authors**: *Hugo Bertiche, Niloy J. Mitra, Kuldeep Kulkarni, Chun-Hao Paul Huang, Tuanfeng Y. Wang, Meysam Madadi, Sergio Escalera, Duygu Ceylan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00052](https://doi.org/10.1109/CVPR52729.2023.00052)

        **Abstract**:

        Cinemagraphs are short looping videos created by adding subtle motions to a static image. This kind of media is popular and engaging. However, automatic generation of cinemagraphs is an underexplored area and current solutions require tedious low-level manual authoring by artists. In this paper, we present an automatic method that allows generating human cinemagraphs from single RGB images. We investigate the problem in the context of dressed humans under the wind. At the core of our method is a novel cyclic neural network that produces looping cinemagraphs for the target loop duration. To circumvent the problem of collecting real data, we demonstrate that it is possible, by working in the image normal space, to learn garment motion dynamics on synthetic data and generalize to real data. We evaluate our method on both synthetic and real data and demonstrate that it is possible to create compelling and plausible cinemagraphs from single RGB images.

        ----

        ## [48] Generating Holistic 3D Human Motion from Speech

        **Authors**: *Hongwei Yi, Hualin Liang, Yifei Liu, Qiong Cao, Yandong Wen, Timo Bolkart, Dacheng Tao, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00053](https://doi.org/10.1109/CVPR52729.2023.00053)

        **Abstract**:

        This work addresses the problem of generating 3D holistic body motions from human speech. Given a speech recording, we synthesize sequences of 3D body poses, hand gestures, and facial expressions that are realistic and diverse. To achieve this, we first build a high-quality dataset of 3D holistic body meshes with synchronous speech. We then define a novel speech-to-motion generation framework in which the face, body, and hands are modeled separately. The separated modeling stems from the fact that face articulation strongly correlates with human speech, while body poses and hand gestures are less correlated. Specifically, we employ an autoencoder for face motions, and a compositional vector-quantized variational autoencoder (VQ- VAE) for the body and hand motions. The compositional VQ-VAE is key to generating diverse results. Additionally, we propose a cross-conditional autoregressive model that generates body poses and hand gestures, leading to coherent and realistic motions. Extensive experiments and user studies demonstrate that our proposed approach achieves state-of-the-art performance both qualitatively and quantitatively. Our dataset and code are released for research purposes at https://talkshow.is.tue.mpg.de/.

        ----

        ## [49] Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Diffusion Model

        **Authors**: *Yuming Du, Robin Kips, Albert Pumarola, Sebastian Starke, Ali K. Thabet, Artsiom Sanakoyeu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00054](https://doi.org/10.1109/CVPR52729.2023.00054)

        **Abstract**:

        With the recent surge in popularity of AR/VR applications, realistic and accurate control of 3D full-body avatars has become a highly demanded feature. A particular challenge is that only a sparse tracking signal is available from standalone HMDs (Head Mounted Devices), often limited to tracking the user's head and wrists. While this signal is resourceful for reconstructing the upper body motion, the lower body is not tracked and must be synthesized from the limited information provided by the upper body joints. In this paper, we present AGRoL, a novel conditional diffusion model specifically designed to track full bodies given sparse upper-body tracking signals. Our model is based on a simple multi-layer perceptron (MLP) architecture and a novel conditioning scheme for motion data. It can predict accurate and smooth full-body motion, particularly the challenging lower body movement. Unlike common diffusion architectures, our compact architecture can run in real-time, making it suitable for online body-tracking applications. We train and evaluate our model on AMASS motion capture dataset, and demonstrate that our approach outperforms state-of-the-art methods in generated motion accuracy and smoothness. We further justify our design choices through extensive experiments and ablation studies.

        ----

        ## [50] Learning Anchor Transformations for 3D Garment Animation

        **Authors**: *Fang Zhao, Zekun Li, Shaoli Huang, Junwu Weng, Tianfei Zhou, Guo-Sen Xie, Jue Wang, Ying Shan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00055](https://doi.org/10.1109/CVPR52729.2023.00055)

        **Abstract**:

        This paper proposes an anchor-based deformation model, namely AnchorDEF, to predict 3D garment animation from a body motion sequence. It deforms a garment mesh template by a mixture of rigid transformations with extra nonlinear displacements. A set of anchors around the mesh surface is introduced to guide the learning of rigid transformation matrices. Once the anchor transformations are found, per-vertex nonlinear displacements of the garment template can be regressed in a canonical space, which reduces the complexity of deformation space learning. By explicitly constraining the transformed anchors to satisfy the consistencies of position, normal and direction, the physical meaning of learned anchor transformations in space is guaranteed for better generalization. Furthermore, an adaptive anchor updating is proposed to optimize the anchor position by being aware of local mesh topology for learning representative anchor transformations. Qualitative and quantitative experiments on different types of garments demonstrate that AnchorDEF achieves the state-of-the-art performance on 3D garment deformation prediction in motion, especially for loose-fitting garments.

        ----

        ## [51] CloSET: Modeling Clothed Humans on Continuous Surface with Explicit Template Decomposition

        **Authors**: *Hongwen Zhang, Siyou Lin, Ruizhi Shao, Yuxiang Zhang, Zerong Zheng, Han Huang, Yandong Guo, Yebin Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00056](https://doi.org/10.1109/CVPR52729.2023.00056)

        **Abstract**:

        Creating animatable avatars from static scans requires the modeling of clothing deformations in different poses. Existing learning-based methods typically add pose-dependent deformations upon a minimally-clothed mesh template or a learned implicit template, which have limitations in capturing details or hinder end-to-end learning. In this paper, we revisit point-based solutions and propose to decompose explicit garment-related templates and then add pose-dependent wrinkles to them. In this way, the clothing deformations are disentangled such that the pose-dependent wrinkles can be better learned and applied to unseen poses. Additionally, to tackle the seam artifact issues in recent state-of-the-art point-based methods, we propose to learn point features on a body surface, which establishes a continuous and compact feature space to capture the fine-grained and pose-dependent clothing geometry. To facilitate the research in this field, we also introduce a high-quality scan dataset of humans in real-world clothing. Our approach is validated on two existing datasets and our newly introduced dataset, showing better clothing deformation results in unseen poses. The project page with code and dataset can be found at https://www.liuyebin.com/closet.

        ----

        ## [52] ECON: Explicit Clothed humans Optimized via Normal integration

        **Authors**: *Yuliang Xiu, Jinlong Yang, Xu Cao, Dimitrios Tzionas, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00057](https://doi.org/10.1109/CVPR52729.2023.00057)

        **Abstract**:

        The combination of deep learning, artist-curated scans, and Implicit Functions (IF), is enabling the creation of detailed, clothed, 3D humans from images. However, existing methods are far from perfect. IF-based methods recover free-form geometry, but produce disembodied limbs or degenerate shapes for novel poses or clothes. To increase robustness for these cases, existing work uses an explicit parametric body model to constrain surface reconstruction, but this limits the recovery of free-form surfaces such as loose clothing that deviates from the body. What we want is a method that combines the best properties of implicit representation and explicit body regularization. To this end, we make two key observations: (1) current networks are better at inferring detailed 2D maps than full-3D surfaces, and (2) a parametric model can be seen as a “canvas” for stitching together detailed surface patches. Based on these, our method, ECON, has three main steps: (1) It infers detailed 2D normal maps for the front and back side of a clothed person. (2) From these, it recovers 2.5D front and back surfaces, called d-BiNI, that are equally detailed, yet incomplete, and registers these w.r.t. each other with the help of a SMPL-X body mesh recovered from the image. (3) It “inpaints” the missing geometry between d-BiNI surfaces. If the face and hands are noisy, they can optionally be replaced with the ones of SMPL-X. As a result, ECON infers high-fidelity 3D humans even in loose clothes and challenging poses. This goes beyond previous methods, according to the quantitative evaluation on the CAPE and Renderpeople datasets. Perceptual studies also show that ECON's perceived realism is better by a large margin. Code and models are available for research purposes at econ.is.tue.mpg.de

        ----

        ## [53] PersonNeRF : Personalized Reconstruction from Photo Collections

        **Authors**: *Chung-Yi Weng, Pratul P. Srinivasan, Brian Curless, Ira Kemelmacher-Shlizerman*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00058](https://doi.org/10.1109/CVPR52729.2023.00058)

        **Abstract**:

        We present PersonNeRF, a method that takes a collection of photos of a subject (e.g. Roger Federer) captured across multiple years with arbitrary body poses and ap-pearances, and enables rendering the subject with arbitrary novel combinations of viewpoint, body pose, and appearance. PersonNeRF builds a customized neural volumetric 3D model of the subject that is able to render an entire space spanned by camera viewpoint, body pose, and appearance. A central challenge in this task is dealing with sparse observations; a given body pose is likely only observed by a single viewpoint with a single appearance, and a given appearance is only observed under a handful of different body poses. We address this issue by recovering a canonical T-pose neural volumetric representation of the subject that allows for changing appearance across different observations, but uses a shared pose-dependent motion field across all observations. We demonstrate that this approach, along with regularization of the recovered volumetric geometry to encourage smoothness, is able to recover a model that renders compelling images from novel combinations of viewpoint, pose, and appearance from these challenging unstructured photo collections, outperforming prior work for free-viewpoint human rendering.

        ----

        ## [54] 3D Human Mesh Estimation from Virtual Markers

        **Authors**: *Xiaoxuan Ma, Jiajun Su, Chunyu Wang, Wentao Zhu, Yizhou Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00059](https://doi.org/10.1109/CVPR52729.2023.00059)

        **Abstract**:

        Inspired by the success of volumetric 3D pose estimation, some recent human mesh estimators propose to estimate 3D skeletons as intermediate representations, from which, the dense 3D meshes are regressed by exploiting the mesh topology. However, body shape information is lost in extracting skeletons, leading to mediocre performance. The advanced motion capture systems solve the problem by placing dense physical markers on the body surface, which allows to extract realistic meshes from their non-rigid motions. However, they cannot be applied to wild images without markers. In this work, we present an intermediate representation, named virtual markers, which learns 64 landmark keypoints on the body surface based on the large-scale mocap data in a generative style, mimicking the effects of physical markers. The virtual markers can be accurately detected from wild images and can reconstruct the intact meshes with realistic shapes by simple interpolation. Our approach outperforms the state-of-the-art methods on three datasets. In particular, it surpasses the existing methods by a notable margin on the SURREAL dataset, which has diverse body shapes. Code is available at https://github.com/ShirleyMaxx/VirtualMarker

        ----

        ## [55] Overcoming the TradeOff between Accuracy and Plausibility in 3D Hand Shape Reconstruction

        **Authors**: *Ziwei Yu, Chen Li, Linlin Yang, Xiaoxu Zheng, Michael Bi Mi, Gim Hee Lee, Angela Yao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00060](https://doi.org/10.1109/CVPR52729.2023.00060)

        **Abstract**:

        Direct mesh fitting for 3D hand shape reconstruction is highly accurate. However, the reconstructed meshes are prone to artifacts and do not appear as plausible hand shapes. Conversely, parametric models like MANO ensure plausible hand shapes but are not as accurate as the non-parametric methods. In this work, we introduce a novel weakly-supervised hand shape estimation framework that integrates non-parametric mesh fitting with MANO model in an end-to-end fashion. Our joint model overcomes the tradeoff in accuracy and plausibility to yield well-aligned and high-quality 3D meshes, especially in challenging two-hand and hand-object interaction scenarios.

        ----

        ## [56] Recovering 3D Hand Mesh Sequence from a Single Blurry Image: A New Dataset and Temporal Unfolding

        **Authors**: *Yeonguk Oh, JoonKyu Park, Jaeha Kim, Gyeongsik Moon, Kyoung Mu Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00061](https://doi.org/10.1109/CVPR52729.2023.00061)

        **Abstract**:

        Hands, one of the most dynamic parts of our body, suffer from blur due to their active movements. However, previous 3D hand mesh recovery methods have mainly focused on sharp hand images rather than considering blur due to the absence of datasets providing blurry hand images. We first present a novel dataset BlurHand, which contains blurry hand images with 3D groundtruths. The BlurHand is constructed by synthesizing motion blur from sequential sharp hand images, imitating realistic and natural motion blurs. In addition to the new dataset, we propose BlurHandNet, a baseline network for accurate 3D hand mesh recovery from a blurry hand image. Our BlurHandNet unfolds a blurry input image to a 3D hand mesh sequence to utilize temporal information in the blurry input image, while previous works output a static single hand mesh. We demonstrate the usefulness of BlurHand for the 3D hand mesh recovery from blurry images in our experiments. The proposed BlurHandNet produces much more robust results on blurry images while generalizing well to in-the-wild images. The training codes and BlurHand dataset are available at https://github.com/laehaKim97IBlurHand_RELEASE.

        ----

        ## [57] MeMaHand: Exploiting Mesh-Mano Interaction for Single Image Two-Hand Reconstruction

        **Authors**: *Congyi Wang, Feida Zhu, Shilei Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00062](https://doi.org/10.1109/CVPR52729.2023.00062)

        **Abstract**:

        Existing methods proposed for hand reconstruction tasks usually parameterize a generic 3D hand model or predict hand mesh positions directly. The parametric representations consisting of hand shapes and rotational poses are more stable, while the non-parametric methods can predict more accurate mesh positions. In this paper, we propose to reconstruct meshes and estimate MANO parameters of two hands from a single RGB image simultaneously to utilize the merits of two kinds of hand representations. To fulfill this target, we propose novel Mesh-Mano interaction blocks (MMIBs), which take mesh vertices positions and MANO parameters as two kinds of query tokens. MMIB consists of one graph residual block to aggregate local information and two transformer encoders to model long-range dependencies. The transformer encoders are equipped with different asymmetric attention masks to model the intra-hand and inter-hand attention, respectively. Moreover, we introduce the mesh alignment refinement module to further enhance the mesh-image alignment. Extensive experiments on the InterHand2.6M benchmark demonstrate promising results over the state-of-the-art hand reconstruction methods.

        ----

        ## [58] PLIKS: A Pseudo-Linear Inverse Kinematic Solver for 3D Human Body Estimation

        **Authors**: *Karthik Shetty, Annette Birkhold, Srikrishna Jaganathan, Norbert Strobel, Markus Kowarschik, Andreas K. Maier, Bernhard Egger*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00063](https://doi.org/10.1109/CVPR52729.2023.00063)

        **Abstract**:

        We introduce PLIKS (Pseudo-Linear Inverse Kinematic Solver) for reconstruction of a 3D mesh of the human body from a single 2D image. Current techniques directly regress the shape, pose, and translation of a parametric model from an input image through a non-linear mapping with minimal flexibility to any external influences. We approach the task as a model-in-the-loop optimization problem. PLIKS is built on a linearized formulation of the parametric SMPL model. Using PLIKS, we can analytically reconstruct the human model via 2D pixel-aligned vertices. This enables us with the flexibility to use accurate camera calibration information when available. PLIKS offers an easy way to introduce additional constraints such as shape and translation. We present quantitative evaluations which confirm that PLIKS achieves more accurate reconstruction with greater than 10% improvement compared to other state-of-the-art methods with respect to the standard 3D human pose and shape benchmarks while also obtaining a reconstruction error improvement of 12.9 mm on the newer AGORA dataset.

        ----

        ## [59] CAMS: CAnonicalized Manipulation Spaces for Category-Level Functional Hand-Object Manipulation Synthesis

        **Authors**: *Juntian Zheng, Qingyuan Zheng, Lixing Fang, Yun Liu, Li Yi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00064](https://doi.org/10.1109/CVPR52729.2023.00064)

        **Abstract**:

        In this work, we focus on a novel task of category-level functional hand-object manipulation synthesis covering both rigid and articulated object categories. Given an object geometry, an initial human hand pose as well as a sparse control sequence of object poses, our goal is to generate a physically reasonable hand-object manipulation sequence that performs like human beings. To address such a challenge, we first design CAnonicalized Manipulation Spaces (CAMS), a two-level space hierarchy that canonicalizes the hand poses in an object-centric and contact-centric view. Benefiting from the representation capability of CAMS, we then present a two-stage framework for synthesizing human-like manipulation animations. Our framework achieves state-of-the-art performance for both rigid and articulated categories with impressive visual effects. Codes and video results can be found at our project homepage: https://cams-hoi.github.io/

        ----

        ## [60] Instant-NVR: Instant Neural Volumetric Rendering for Human-object Interactions from Monocular RGBD Stream

        **Authors**: *Yuheng Jiang, Kaixin Yao, Zhuo Su, Zhehao Shen, Haimin Luo, Lan Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00065](https://doi.org/10.1109/CVPR52729.2023.00065)

        **Abstract**:

        Convenient 4D modeling of human-object interactions is essential for numerous applications. However, monocular tracking and rendering of complex interaction scenarios remain challenging. In this paper, we propose Instant-NVR, a neural approach for instant volumetric human-object tracking and rendering using a single RGBD camera. It bridges traditional non-rigid tracking with recent instant radiance field techniques via a multi-thread tracking-rendering mechanism. In the tracking front-end, we adopt a robust human-object capture scheme to provide sufficient motion priors. We further introduce a separated instant neural representation with a novel hybrid deformation module for the interacting scene. We also provide an on-the-fly reconstruction scheme of the dynamic/static radiance fields via efficient motion-prior searching. Moreover, we introduce an online key frame selection scheme and a rendering-aware refinement strategy to significantly improve the appearance details for online novel-view synthesis. Extensive experiments demonstrate the effectiveness and efficiency of our approach for the instant generation of human-object radiance fields on the fly, notably achieving real-time photo-realistic novel view synthesis under complex human-object interactions. Project page: https://nowheretrix.github.io/Instant-NVR/.

        ----

        ## [61] BundleSDF: Neural 6-DoF Tracking and 3D Reconstruction of Unknown Objects

        **Authors**: *Bowen Wen, Jonathan Tremblay, Valts Blukis, Stephen Tyree, Thomas Müller, Alex Evans, Dieter Fox, Jan Kautz, Stan Birchfield*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00066](https://doi.org/10.1109/CVPR52729.2023.00066)

        **Abstract**:

        We present a near real-time (10Hz) method for 6-DoF tracking of an unknown object from a monocular RGBD video sequence, while simultaneously performing neural 3D reconstruction of the object. Our method works for arbi-trary rigid objects, even when visual texture is largely ab-sent. The object is assumed to be segmented in the first frame only. No additional information is required, and no assumption is made about the interaction agent. Key to our method is a Neural Object Field that is learned concurrently with a pose graph optimization process in order to robustly accumulate information into a consistent 3D representation capturing both geometry and appearance. A dynamic pool of posed memory frames is automatically main-tained to facilitate communication between these threads. Our approach handles challenging sequences with large pose changes, partial and full occlusion, untextured surfaces, and specular highlights. We show results on HO3D, YCBInEOAT, and BEHAVE datasets, demonstrating that our method significantly outperforms existing approaches. Project page: https://bundlesdf.github.io/

        ----

        ## [62] Human-Art: A Versatile Human-Centric Dataset Bridging Natural and Artificial Scenes

        **Authors**: *Xuan Ju, Ailing Zeng, Jianan Wang, Qiang Xu, Lei Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00067](https://doi.org/10.1109/CVPR52729.2023.00067)

        **Abstract**:

        Humans have long been recorded in a variety of forms since antiquity. For example, sculptures and paintings were the primary media for depicting human beings before the invention of cameras. However, most current human-centric computer vision tasks like human pose estimation and human image generation focus exclusively on natural images in the real world. Artificial humans, such as those in sculptures, paintings, and cartoons, are commonly neglected, making existing models fail in these scenarios. As an abstraction of life, art incorporates humans in both natural and artificial scenes. We take advantage of it and introduce the Human-Art dataset to bridge related tasks in natural and artificial scenarios. Specifically, Human-Art contains 50k high-quality images with over 123k person instances from 5 natural and 15 artificial scenarios, which are annotated with bounding boxes, keypoints, self-contact points, and text information for humans represented in both 2D and 3D. It is, therefore, comprehensive and versatile for various downstream tasks. We also provide a rich set of baseline results and detailed analyses for related tasks, including human detection, 2D and 3D human pose estimation, image generation, and motion transfer. As a challenging dataset, we hope Human-Art can provide insights for relevant research and open up new research questions.

        ----

        ## [63] Omnimatte3D: Associating Objects and Their Effects in Unconstrained Monocular Video

        **Authors**: *Mohammed Suhail, Erika Lu, Zhengqi Li, Noah Snavely, Leonid Sigal, Forrester Cole*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00068](https://doi.org/10.1109/CVPR52729.2023.00068)

        **Abstract**:

        We propose a method to decompose a video into a background and a set of foreground layers, where the background captures stationary elements while the foreground layers capture moving objects along with their associated effects (e.g. shadows and reflections). Our approach is designed for unconstrained monocular videos, with an arbitrary camera and object motion. Prior work that tackles this problem assumes that the video can be mapped onto a fixed 2D canvas, severely limiting the possible space of camera motion. Instead, our method applies recent progress in monocular camera pose and depth estimation to create a full, RGBD video layer for the background, along with a video layer for each foreground object. To solve the underconstrained decomposition problem, we propose a new loss formulation based on multi-view consistency. We test our method on challenging videos with complex camera motion and show significant qualitative improvement over current approaches.

        ----

        ## [64] On the Benefits of 3D Pose and Tracking for Human Action Recognition

        **Authors**: *Jathushan Rajasegaran, Georgios Pavlakos, Angjoo Kanazawa, Christoph Feichtenhofer, Jitendra Malik*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00069](https://doi.org/10.1109/CVPR52729.2023.00069)

        **Abstract**:

        In this work we study the benefits of using tracking and 3D poses for action recognition. To achieve this, we take the Lagrangian view on analysing actions over a trajectory of human motion rather than at a fixed point in space. Taking this stand allows us to use the tracklets of people to predict their actions. In this spirit, first we show the benefits of using 3D pose to infer actions, and study person-person in-teractions. Subsequently, we propose a Lagrangian Action Recognition model by fusing 3D pose and contextualized appearance over tracklets. To this end, our method achieves state-of-the-art performance on the AVA v2.2 dataset on both pose only settings and on standard benchmark settings. When reasoning about the action using only pose cues, our pose model achieves +10.0 mAP gain over the corresponding state-of-the-art while our fused model has a gain of +2.8 mAP over the best state-of-the-art model. Code and results are available at: https://brjathu.github.io/LART

        ----

        ## [65] Towards Stable Human Pose Estimation via Cross-View Fusion and Foot Stabilization

        **Authors**: *Li'an Zhuo, Jian Cao, Qi Wang, Bang Zhang, Liefeng Bo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00070](https://doi.org/10.1109/CVPR52729.2023.00070)

        **Abstract**:

        Towards stable human pose estimation from monocular images, there remain two main dilemmas. On the one hand, the different perspectives, i.e., front view, side view, and top view, appear the inconsistent performances due to the depth ambiguity. On the other hand, foot posture plays a significant role in complicated human pose estimation, i.e., dance and sports, and foot-ground interaction, but unfortunately, it is omitted in most general approaches and datasets. In this paper, we first propose the Cross-View Fusion (CVF) module to catch up with better 3D intermediate representation and alleviate the view inconsistency based on the vision transformer encoder. Then the optimization-based method is introduced to reconstruct the foot pose and foot-ground contact for the general multi-view datasets including AIST++ and Human3.6M. Besides, the reversible kinematic topology strategy is innovated to utilize the contact information into the full-body with foot pose regressor. Extensive experiments on the popular benchmarks demonstrate that our method outperforms the state-of-the-art approaches by achieving 40.1mm PA-MPJPE on the 3DPW test set and 43.8mm on the AIST++ test set.

        ----

        ## [66] Human Pose as Compositional Tokens

        **Authors**: *Zigang Geng, Chunyu Wang, Yixuan Wei, Ze Liu, Houqiang Li, Han Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00071](https://doi.org/10.1109/CVPR52729.2023.00071)

        **Abstract**:

        Human pose is typically represented by a coordinate vector of body joints or their heatmap embeddings. While easy for data processing, unrealistic pose estimates are admitted due to the lack of dependency modeling between the body joints. In this paper, we present a structured representation, named Pose as Compositional Tokens (PCT), to explore the joint dependency. It represents a pose by M discrete tokens with each characterizing a sub-structure with several interdependent joints (see Figure 1). The compositional design enables it to achieve a small reconstruction error at a low cost. Then we cast pose estimation as a classification task. In particular, we learn a classifier to predict the categories of the M tokens from an image. A pre-learned decoder network is used to recover the pose from the tokens without further post-processing. We show that it achieves better or comparable pose estimation results as the existing methods in general scenarios, yet continues to work well when occlusion occurs, which is ubiquitous in practice. The code and models are publicly available at https://github.com/Gengzigang/PCT.

        ----

        ## [67] PoseExaminer: Automated Testing of Out-of-Distribution Robustness in Human Pose and Shape Estimation

        **Authors**: *Qihao Liu, Adam Kortylewski, Alan L. Yuille*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00072](https://doi.org/10.1109/CVPR52729.2023.00072)

        **Abstract**:

        Human pose and shape (HPS) estimation methods achieve remarkable results. However, current HPS bench-marks are mostly designed to test models in scenarios that are similar to the training data. This can lead to critical situations in real-world applications when the observed data differs significantly from the training data and hence is out-of-distribution (OOD). It is therefore important to test and improve the OOD robustness of HPS methods. To address this fundamental problem, we develop a simulator that can be controlled in a fine-grained manner using interpretable parameters to explore the manifold of images of human pose, e.g. by varying poses, shapes, and clothes. We introduce a learning-based testing method, termed PoseExaminer, that automatically diagnoses HPS algorithms by searching over the parameter space of human pose images to find the failure modes. Our strategy for exploring this high-dimensional parameter space is a multiagent reinforcement learning system, in which the agents collaborate to explore different parts of the parameter space. We show that our PoseExaminer discovers a variety of limitations in current state-of-the-art models that are relevant in real-world scenarios but are missed by current benchmarks. For example, it finds large regions of realistic human poses that are not predicted correctly, as well as reduced performance for humans with skinny and corpulent body shapes. In addition, we show that fine-tuning HPS methods by exploiting the failure modes found by PoseExaminer improve their robustness and even their performance on standard benchmarks by a significant margin. The code are available for research purposes at https://github.com/qihao0671PoseExaminer.

        ----

        ## [68] SLOPER4D: A Scene-Aware Dataset for Global 4D Human Pose Estimation in Urban Environments

        **Authors**: *Yudi Dai, Yitai Lin, Xiping Lin, Chenglu Wen, Lan Xu, Hongwei Yi, Siqi Shen, Yuexin Ma, Cheng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00073](https://doi.org/10.1109/CVPR52729.2023.00073)

        **Abstract**:

        We present SLOPER4D, a novel scene-aware dataset collected in large urban environments to facilitate the research of global human pose estimation (GHPE) with human-scene interaction in the wild. Employing a head-mounted device integrated with a LiDAR and camera, we record 12 human subjects' activities over 10 diverse urban scenes from an egocentric view. Frame-wise annotations for 2D key points, 3D pose parameters, and global translations are provided, together with reconstructed scene point clouds. To obtain accurate 3D ground truth in such large dynamic scenes, we propose a joint optimization method to fit local SMPL meshes to the scene and fine-tune the camera calibration during dynamic motions frame by frame, resulting in plausible and scene-natural 3D human poses. Even-tually, SLOPER4D consists of 15 sequences of human motions, each of which has a trajectory length of more than 200 meters (up to 1,300 meters) and covers an area of more than 200 m
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
 (up to 30,000 m
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
), including more than 100k LiDAR frames, 300k video frames, and 500k IMU-based motion frames. With SLOPER4D, we provide a detailed and thorough analysis of two critical tasks, including camera-based 3D HPE and LiDAR-based 3D HPE in urban environments, and benchmark a new task, GHPE. The in-depth analysis demonstrates SLOPER4D poses significant challenges to existing methods and produces great research opportunities. The dataset and code are released at http://www.lidarhumanmotion.net/sloper4d/.

        ----

        ## [69] Semi-Supervised 2D Human Pose Estimation Driven by Position Inconsistency Pseudo Label Correction Module

        **Authors**: *Linzhi Huang, Yulong Li, Hongbo Tian, Yue Yang, Xiangang Li, Weihong Deng, Jieping Ye*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00074](https://doi.org/10.1109/CVPR52729.2023.00074)

        **Abstract**:

        In this paper, we delve into semi-supervised 2D human pose estimation. The previous method ignored two problems: (i) When conducting interactive training between large model and lightweight model, the pseudo label of lightweight model will be used to guide large models. (ii) The negative impact of noise pseudo labels on training. Moreover, the labels used for 2D human pose estimation are relatively complex: keypoint category and keypoint position. To solve the problems mentioned above, we propose a semi-supervised 2D human pose estimation framework driven by a position inconsistency pseudo label correction module (SSPCM). We introduce an additional auxiliary teacher and use the pseudo labels generated by the two teacher model in different periods to calculate the inconsistency score and remove outliers. Then, the two teacher models are updated through interactive training, and the student model is updated using the pseudo labels generated by two teachers. To further improve the performance of the student model, we use the semi-supervised Cut-Occlude based on pseudo keypoint perception to generate more hard and effective samples. In addition, we also proposed a new indoor overhead fisheye human keypoint dataset WEPDTOF-Pose. Extensive experiments demonstrate that our method outperforms the previous best semi-supervised 2D human pose estimation method. We will release the code and dataset at https://github.com/hlz0606/SSPCM

        ----

        ## [70] Human Pose Estimation in Extremely Low-Light Conditions

        **Authors**: *Sohyun Lee, Jaesung Rim, Boseung Jeong, Geonu Kim, Byungju Woo, Haechan Lee, Sunghyun Cho, Suha Kwak*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00075](https://doi.org/10.1109/CVPR52729.2023.00075)

        **Abstract**:

        We study human pose estimation in extremely low-light images. This task is challenging due to the difficulty of collecting real low-light images with accurate labels, and severely corrupted inputs that degrade prediction quality significantly. To address the first issue, we develop a ded-icated camera system and build a new dataset of real low-light images with accurate pose labels. Thanks to our camera system, each low-light image in our dataset is coupled with an aligned well-lit image, which enables accurate pose labeling and is used as privileged information during training. We also propose a new model and a new training strategy that fully exploit the privileged information to learn representation insensitive to lighting conditions. Our method demonstrates outstanding performance on real extremely low-light images, and extensive analyses validate that both of our model and dataset contribute to the success.

        ----

        ## [71] Flexible-Cm GAN: Towards Precise 3D Dose Prediction in Radiotherapy

        **Authors**: *Riqiang Gao, Bin Lou, Zhoubing Xu, Dorin Comaniciu, Ali Kamen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00076](https://doi.org/10.1109/CVPR52729.2023.00076)

        **Abstract**:

        Deep learning has been utilized in knowledge-based radiotherapy planning in which a system trained with a set of clinically approved plans is employed to infer a three-dimensional dose map for a given new patient. However, previous deep methods are primarily limited to simple scenarios, e.g., a fixed planning type or a consistent beam angle configuration. This in fact limits the usability of such approaches and makes them not generalizable over a larger set of clinical scenarios. Herein, we propose a novel conditional generative model, Flexible-Cm GAN, utilizing additional information regarding planning types and various beam geometries. A miss-consistency loss is proposed to deal with the challenge of having a limited set of conditions on the input data, e.g., incomplete training samples. To address the challenges of including clinical preferences, we derive a differentiable shift-dose-volume loss to incorporate the well-known dose-volume histogram constraints. During inference, users can flexibly choose a specific planning type and a set of beam angles to meet the clinical requirements. We conduct experiments on an illustrative face dataset to show the motivation of Flexible-Cm GAN and further validate our model's potential clinical values with two radiotherapy datasets. The results demonstrate the superior performance of the proposed method in a practical heterogeneous radiotherapy planning application compared to existing deep learning-based approaches.

        ----

        ## [72] DualRefine: Self-Supervised Depth and Pose Estimation Through Iterative Epipolar Sampling and Refinement Toward Equilibrium

        **Authors**: *Antyanta Bangunharcana, Ahmed Magd, Kyung-Soo Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00077](https://doi.org/10.1109/CVPR52729.2023.00077)

        **Abstract**:

        Self-supervised multi-frame depth estimation achieves high accuracy by computing matching costs of pixel correspondences between adjacent frames, injecting geometric information into the network. These pixel-correspondence candidates are computed based on the relative pose estimates between the frames. Accurate pose predictions are essential for precise matching cost computation as they influence the epipolar geometry. Furthermore, improved depth estimates can, in turn, be used to align pose estimates. Inspired by traditional structure-from-motion (SfM) principles, we propose the DualRefine model, which tightly couples depth and pose estimation through a feedback loop. Our novel update pipeline uses a deep equilibrium model framework to iteratively refine depth estimates and a hidden state of feature maps by computing local matching costs based on epipolar geometry. Importantly, we used the refined depth estimates and feature maps to compute pose updates at each step. This update in the pose estimates slowly alters the epipolar geometry during the refinement process. Experimental results on the KITTI dataset demonstrate competitive depth prediction and odometry prediction performance surpassing published self-supervised baselines
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/antabangun/DualRefine.

        ----

        ## [73] A Rotation-Translation-Decoupled Solution for Robust and Efficient Visual-Inertial Initialization

        **Authors**: *Yijia He, Bo Xu, Zhanpeng Ouyang, Hongdong Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00078](https://doi.org/10.1109/CVPR52729.2023.00078)

        **Abstract**:

        We propose a novel visual-inertial odometry (VIO) initialization method, which decouples rotation and translation estimation, and achieves higher efficiency and better robustness. Existing loosely-coupled VIO-initialization methods suffer from poor stability of visual structure-from-motion (SfM), whereas those tightly-coupled methods often ignore the gyroscope bias in the closed-form solution, resulting in limited accuracy. Moreover, the aforementioned two classes of methods are computationally expensive, because 3D point clouds need to be reconstructed simultaneously. In contrast, our new method fully combines inertial and visual measurements for both rotational and translational initialization. First, a rotation-only solution is designed for gyroscope bias estimation, which tightly couples the gyroscope and camera observations. Second, the initial velocity and gravity vector are solved with linear translation constraints in a globally optimal fashion and without reconstructing 3D point clouds. Extensive experiments have demonstrated that our method is 8 ~ 72 times faster (w.r.t. a 10-frame set) than the state-of-the-art methods, and also presents significantly higher robustness and accuracy. The source code is available at https://github.com/boxuLibrary/drt-vio-init.

        ----

        ## [74] Semidefinite Relaxations for Robust Multiview Triangulation

        **Authors**: *Linus Härenstam-Nielsen, Niclas Zeller, Daniel Cremers*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00079](https://doi.org/10.1109/CVPR52729.2023.00079)

        **Abstract**:

        We propose an approach based on convex relaxations for certifiably optimal robust multiview triangulation. To this end, we extend existing relaxation approaches to non-robust multiview triangulation by incorporating a least squares cost function. We propose two formulations, one based on epipolar constraints and one based on fractional reprojection constraints. The first is lower dimensional and remains tight under moderate noise and outlier levels, while the second is higher dimensional and therefore slower but remains tight even under extreme noise and outlier levels. We demonstrate through extensive experiments that the proposed approaches allow us to compute provably optimal re-constructions even under significant noise and a large percentage of outliers.

        ----

        ## [75] A Probabilistic Attention Model with Occlusion-aware Texture Regression for 3D Hand Reconstruction from a Single RGB Image

        **Authors**: *Zheheng Jiang, Hossein Rahmani, Sue Black, Bryan M. Williams*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00080](https://doi.org/10.1109/CVPR52729.2023.00080)

        **Abstract**:

        Recently, deep learning based approaches have shown promising results in 3D hand reconstruction from a single RGB image. These approaches can be roughly divided into model-based approaches, which are heavily dependent on the model's parameter space, and model-free approaches, which require large numbers of 3D ground truths to reduce depth ambiguity and struggle in weakly-supervised scenarios. To overcome these issues, we propose a novel probabilistic model to achieve the robustness of model-based approaches and reduced dependence on the model's parameter space of model-free approaches. The proposed probabilistic model incorporates a model-based network as a prior-net to estimate the prior probability distribution of joints and vertices. An Attention-based Mesh Vertices Uncertainty Regression (AMVUR) model is proposed to capture dependencies among vertices and the correlation between joints and mesh vertices to improve their feature representation. We further propose a learning based occlusion-aware Hand Texture Regression model to achieve high-fidelity texture reconstruction. We demonstrate the flexibility of the proposed probabilistic model to be trained in both supervised and weakly-supervised scenarios. The experimental results demonstrate our probabilistic model's state-of-the-art accuracy in 3D hand and texture reconstruction from a single image in both training schemes, including in the presence of severe occlusions.

        ----

        ## [76] Instant Multi-View Head Capture through Learnable Registration

        **Authors**: *Timo Bolkart, Tianye Li, Michael J. Black*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00081](https://doi.org/10.1109/CVPR52729.2023.00081)

        **Abstract**:

        Existing methods for capturing datasets of 3D heads in dense semantic correspondence are slow and commonly address the problem in two separate steps; multi-view stereo (MVS) reconstruction followed by non-rigid registration. To simplify this process, we introduce TEMPEH (Towards Estimation of 3D Meshes from Performances of Expressive Heads) to directly infer 3D heads in dense correspondence from calibrated multi-view images. Registering datasets of 3D scans typically requires manual parameter tuning to find the right balance between accurately fitting the scans' surfaces and being robust to scanning noise and outliers. Instead, we propose to jointly register a 3D head dataset while training TEMPEH. Specifically, during training, we minimize a geometric loss commonly used for surface registration, effectively leveraging TEMPEH as a regularizer. Our multi-view head inference builds on a volumetric feature representation that samples and fuses features from each view using camera calibration information. To account for partial occlusions and a large capture volume that enables head movements, we use view-and surface-aware feature fusion, and a spatial transformer-based head localization module, respectively. We use raw MVS scans as supervision during training, but, once trained, TEMPEH directly predicts 3D heads in dense correspondence without requiring scans. Predicting one head takes about 0.3 seconds with a median reconstruction error of 0.26 mm, 64% lower than the current state-of-the-art. This enables the efficient capture of large datasets containing multiple people and diverse facial motions. Code, model, and data are publicly available at https://tempeh.is.tue.mpg.de.

        ----

        ## [77] On the Importance of Accurate Geometry Data for Dense 3D Vision Tasks

        **Authors**: *HyunJun Jung, Patrick Ruhkamp, Guangyao Zhai, Nikolas Brasch, Yitong Li, Yannick Verdie, Jifei Song, Yiren Zhou, Anil Armagan, Slobodan Ilic, Ales Leonardis, Nassir Navab, Benjamin Busam*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00082](https://doi.org/10.1109/CVPR52729.2023.00082)

        **Abstract**:

        Learning-based methods to solve dense 3D vision problems typically train on 3D sensor data. The respectively used principle of measuring distances provides advantages and drawbacks. These are typically not compared nor discussed in the literature due to a lack of multi-modal datasets. Texture-less regions are problematic for structure from motion and stereo, reflective material poses issues for active sensing, and distances for translucent objects are intricate to measure with existing hardware. Training on inaccurate or corrupt data induces model bias and hampers generalisation capabilities. These effects remain unnoticed if the sensor measurement is considered as ground truth during the evaluation. This paper investigates the effect of sensor errors for the dense 3D vision tasks of depth estimation and reconstruction. We rigorously show the significant impact of sensor characteristics on the learned predictions and notice generalisation issues arising from various technologies in everyday household environments. For evaluation, we introduce a carefully designed dataset
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
dataset available at https://github.com/Junggy/HAMMER-dataset comprising measurements from commodity sensors, namely D-ToF, I-ToF, passive/active stereo, and monocular RGB+P. Our study quantifies the considerable sensor noise impact and paves the way to improved dense vision estimates and targeted data fusion.

        ----

        ## [78] Learning 3D Scene Priors with 2D Supervision

        **Authors**: *Yinyu Nie, Angela Dai, Xiaoguang Han, Matthias Nießner*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00083](https://doi.org/10.1109/CVPR52729.2023.00083)

        **Abstract**:

        Holistic 3D scene understanding entails estimation of both layout configuration and object geometry in a 3D environment. Recent works have shown advances in 3D scene estimation from various input modalities (e.g., images, 3D scans), by leveraging 3D supervision (e.g., 3D bounding boxes or CAD models), for which collection at scale is ex-pensive and often intractable. To address this shortcoming, we propose a new method to learn 3D scene priors of layout and shape without requiring any 3D ground truth. Instead, we rely on 2D supervision from multi-view RGB images. Our method represents a 3D scene as a latent vector, from which we can progressively decode to a sequence of objects characterized by their class categories, 3D bounding boxes, and meshes. With our trained autoregressive decoder representing the scene prior, our method facilitates many downstream applications, including scene synthesis, interpolation, and single-view reconstruction. Experiments on 3D-FRONT and ScanNet show that our method outperforms state of the art in single-view reconstruction, and achieves state-of-the-art results in scene synthesis against baselines which require for 3D supervision. Project page: https://yinyunie.github.io/sceneprior-page/

        ----

        ## [79] OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation

        **Authors**: *Tong Wu, Jiarui Zhang, Xiao Fu, Yuxin Wang, Jiawei Ren, Liang Pan, Wayne Wu, Lei Yang, Jiaqi Wang, Chen Qian, Dahua Lin, Ziwei Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00084](https://doi.org/10.1109/CVPR52729.2023.00084)

        **Abstract**:

        Recent advances in modeling 3D objects mostly rely on synthetic datasets due to the lack of large-scale real-scanned 3D databases. To facilitate the development of 3D perception, reconstruction, and generation in the real world, we propose OmniObject3D, a large vocabulary 3D object dataset with massive high-quality real-scanned 3D objects. OmniObject3D has several appealing properties: 1) Large Vocabulary: It comprises 6,000 scanned objects in 190 daily categories, sharing common classes with popular 2D datasets (e.g., ImageNet and LVIS), benefiting the pursuit of generalizable 3D representations. 2) Rich Annotations: Each 3D object is captured with both 2D and 3D sensors, providing textured meshes, point clouds, multi-view rendered images, and multiple real-captured videos. 3) Realistic Scans: The professional scanners support high-quality object scans with precise shapes and realistic appearances. With the vast exploration space offered by OmniObject3D, we carefully set up four evaluation tracks: a) robust 3D perception, b) novel-view synthesis, c) neural surface reconstruction, and d) 3D object generation. Extensive studies are performed on these four benchmarks, revealing new observations, challenges, and opportunities for future research in realistic 3D vision.

        ----

        ## [80] OpenScene: 3D Scene Understanding with Open Vocabularies

        **Authors**: *Songyou Peng, Kyle Genova, Chiyu Max Jiang, Andrea Tagliasacchi, Marc Pollefeys, Thomas A. Funkhouser*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00085](https://doi.org/10.1109/CVPR52729.2023.00085)

        **Abstract**:

        Traditional 3D scene understanding approaches rely on labeled 3D datasets to train a model for a single task with supervision. We propose OpenScene, an alternative approach where a model predicts dense features for 3D scene points that are co-embedded with text and image pixels in CLIP feature space. This zero-shot approach enables task-agnostic training and open-vocabulary queries. For example, to perform SOTA zero-shot 3D semantic segmentation it first infers CLIP features for every 3D point and later classifies them based on similarities to embeddings of arbitrary class labels. More interestingly, it enables a suite of open-vocabulary scene understanding applications that have never been done before. For example, it allows a user to enter an arbitrary text query and then see a heat map indicating which parts of a scene match. Our approach is effective at identifying objects, materials, affordances, activities, and room types in complex 3D scenes, all using a single model trained without any labeled 3D data.

        ----

        ## [81] Multi-View Azimuth Stereo via Tangent Space Consistency

        **Authors**: *Xu Cao, Hiroaki Santo, Fumio Okura, Yasuyuki Matsushita*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00086](https://doi.org/10.1109/CVPR52729.2023.00086)

        **Abstract**:

        We present a method for 3D reconstruction only using calibrated multi-view surface azimuth maps. Our method, multi-view azimuth stereo, is effective for textureless or specular surfaces, which are difficult for conventional multi-view stereo methods. We introduce the concept of tangent space consistency: Multi-view azimuth observations of a surface point should be lifted to the same tangent space. Leveraging this consistency, we recover the shape by optimizing a neural implicit surface representation. Our method harnesses the robust azimuth estimation capabilities of photometric stereo methods or polarization imaging while bypassing potentially complex zenith angle estimation. Experiments using azimuth maps from various sources validate the accurate shape recovery with our method, even without zenith angles.

        ----

        ## [82] Progressive Transformation Learning for Leveraging Virtual Images in Training

        **Authors**: *Yi-Ting Shen, Hyungtae Lee, Heesung Kwon, Shuvra S. Bhattacharyya*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00087](https://doi.org/10.1109/CVPR52729.2023.00087)

        **Abstract**:

        To effectively interrogate UAV-based images for detecting objects of interest, such as humans, it is essential to acquire large-scale UAV-based datasets that include human instances with various poses captured from widely varying viewing angles. As a viable alternative to laborious and costly data curation, we introduce Progressive Transformation Learning (PTL), which gradually augments a training dataset by adding transformed virtual images with enhanced realism. Generally, a virtual2real transformation generator in the conditional GAN framework suffers from quality degradation when a large domain gap exists between real and virtual images. To deal with the domain gap, PTL takes a novel approach that progressively iterates the following three steps: 1) select a subset from a pool of virtual images according to the domain gap, 2) transform the selected virtual images to enhance realism, and 3) add the transformed virtual images to the training set while removing them from the pool. In PTL, accurately quantifying the domain gap is critical. To do that, we theoretically demonstrate that the feature representation space of a given object detector can be modeled as a multivariate Gaussian distribution from which the Mahalanobis distance between a virtual object and the Gaussian distribution of each object category in the representation space can be readily computed. Experiments show that PTL results in a substantial performance increase over the baseline, especially in the small data and the cross-domain regime.

        ----

        ## [83] Connecting the Dots: Floorplan Reconstruction Using Two-Level Queries

        **Authors**: *Yuanwen Yue, Theodora Kontogianni, Konrad Schindler, Francis Engelmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00088](https://doi.org/10.1109/CVPR52729.2023.00088)

        **Abstract**:

        We address 2D floorplan reconstruction from 3D scans. Existing approaches typically employ heuristically designed multi-stage pipelines. Instead, we formulate floor-plan reconstruction as a single-stage structured prediction task: find a variablesize set of polygons, which in turn are variable-length sequences of ordered vertices. To solve it we develop a novel Transformer architecture that generates polygons of multiple rooms in parallel, in a holistic manner without hand-crafted intermediate stages. The model features two-level queries for polygons and corners, and includes polygon matching to make the network end-to-end trainable. Our method achieves a new state-of-the-art for two challenging datasets, Structured3D and SceneCAD, along with significantly faster inference than previous methods. Moreover, it can readily be extended to predict additional information, i.e., semantic room types and architectural elements like doors and windows. Our code and models are available at: https://github.com/ywyue/RoomFormer.

        ----

        ## [84] NeRF-Supervised Deep Stereo

        **Authors**: *Fabio Tosi, Alessio Tonioni, Daniele De Gregorio, Matteo Poggi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00089](https://doi.org/10.1109/CVPR52729.2023.00089)

        **Abstract**:

        We introduce a novel framework for training deep stereo networks effortlessly and without any ground-truth. By leveraging state-of-the-art neural rendering solutions, we generate stereo training data from image sequences collected with a single handheld camera. On top of them, a NeRF-supervised training procedure is carried out, from which we exploit rendered stereo triplets to compensate for occlusions and depth maps as proxy labels. This results in stereo networks capable of predicting sharp and detailed disparity maps. Experimental results show that models trained under this regime yield a 30-40% improvement over existing self-supervised methods on the challenging Middle-bury dataset, filling the gap to supervised models and, most times, outperforming them at zero-shot generalization.

        ----

        ## [85] Semantic Scene Completion with Cleaner Self

        **Authors**: *Fengyun Wang, Dong Zhang, Hanwang Zhang, Jinhui Tang, Qianru Sun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00090](https://doi.org/10.1109/CVPR52729.2023.00090)

        **Abstract**:

        Semantic Scene Completion (SSC) transforms an image of single-view depth and/or RGB 2D pixels into 3D voxels, each of whose semantic labels are predicted. SSC is a well-known ill-posed problem as the prediction model has to “imagine” what is behind the visible surface, which is usually represented by Truncated Signed Distance Function (TSDF). Due to the sensory imperfection of the depth camera, most existing methods based on the noisy TSDF estimated from depth values suffer from 1) incomplete volumetric predictions and 2) confused semantic labels. To this end, we use the ground-truth 3D voxels to generate a perfect visible surface, called TSDF-CAD, and then train a “cleaner” SSC model. As the model is noise-free, it is expected to focus more on the “imagination” of unseen voxels. Then, we propose to distill the intermediate “cleaner” knowledge into another model with noisy TSDF input. In particular, we use the 3D occupancy feature and the semantic relations of the “cleaner self” to supervise the counterparts of the “noisy self” to respectively address the above two incorrect predictions. Experimental results validate that our method improves the noisy counterparts with 3.1% IoU and 2.2% mIoU for measuring scene completion and SSC, and also achieves new state-of-the-art accuracy on the popular NYU dataset. The code is available at https://github.com/fereenwong/CleanerS.

        ----

        ## [86] PanelNet: Understanding 360 Indoor Environment via Panel Representation

        **Authors**: *Haozheng Yu, Lu He, Bing Jian, Weiwei Feng, Shan Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00091](https://doi.org/10.1109/CVPR52729.2023.00091)

        **Abstract**:

        Indoor 360 panoramas have two essential properties. (1) The panoramas are continuous and seamless in the horizontal direction. (2) Gravity plays an important role in indoor environment design. By leveraging these properties, we present PanelNet, a framework that understands indoor environments using a novel panel representation of 360 images. We represent an equirectangular projection (ERP) as consecutive vertical panels with corresponding 3D panel geometry. To reduce the negative impact of panoramic distortion, we incorporate a panel geometry embedding network that encodes both the local and global geometric features of a panel. To capture the geometric context in room design, we introduce Local2Global Transformer, which aggregates local information within a panel and panel-wise global context. It greatly improves the model performance with low training overhead. Our method outperforms existing methods on indoor 360 depth estimation and shows competitive results against state-of-the-art approaches on the task of indoor layout estimation and semantic segmentation.

        ----

        ## [87] Implicit View-Time Interpolation of Stereo Videos Using Multi-Plane Disparities and Non-Uniform Coordinates

        **Authors**: *Avinash Paliwal, Andrii Tsarov, Nima Khademi Kalantari*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00092](https://doi.org/10.1109/CVPR52729.2023.00092)

        **Abstract**:

        In this paper, we propose an approach for view-time interpolation of stereo videos. Specifically, we build upon X-Fields that approximates an interpolatable mapping between the input coordinates and 2D RGB images using a convolutional decoder. Our main contribution is to analyze and identify the sources of the problems with using X-Fields in our application and propose novel techniques to overcome these challenges. Specifically, we observe that X-Fields struggles to implicitly interpolate the disparities for large baseline cameras. Therefore, we propose multi-plane disparities to reduce the spatial distance of the objects in the stereo views. Moreover, we propose non-uniform time coordinates to handle the non-linear and sudden motion spikes in videos. We additionally introduce several simple, but important, improvements over X-Fields. We demonstrate that our approach is able to produce better results than the state of the art, while running in near real-time rates and having low memory and storage costs.

        ----

        ## [88] Depth Estimation from Indoor Panoramas with Neural Scene Representation

        **Authors**: *Wenjie Chang, Yueyi Zhang, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00093](https://doi.org/10.1109/CVPR52729.2023.00093)

        **Abstract**:

        Depth estimation from indoor panoramas is challenging due to the equirectangular distortions of panoramas and inaccurate matching. In this paper, we propose a practical framework to improve the accuracy and efficiency of depth estimation from multi-view indoor panoramic images with the Neural Radiance Field technology. Specifically, we develop two networks to implicitly learn the Signed Distance Function for depth measurements and the radiance field from panoramas. We also introduce a novel spherical position embedding scheme to achieve high accuracy. For better convergence, we propose an initialization method for the network weights based on the Manhattan World Assumption. Furthermore, we devise a geometric consistency loss, leveraging the surface normal, to further refine the depth estimation. The experimental results demonstrate that our proposed method outperforms state-of-the-art works by a large margin in both quantitative and qualitative evaluations. Our source code is available at https://github.com/WJ-Chang-42/IndoorPanoDepth.

        ----

        ## [89] NeuralPCI: Spatio-Temporal Neural Field for 3D Point Cloud Multi-Frame Non-Linear Interpolation

        **Authors**: *Zehan Zheng, Danni Wu, Ruisi Lu, Fan Lu, Guang Chen, Changjun Jiang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00094](https://doi.org/10.1109/CVPR52729.2023.00094)

        **Abstract**:

        In recent years, there has been a significant increase in focus on the interpolation task of computer vision. Despite the tremendous advancement of video interpolation, point cloud interpolation remains insufficiently explored. Meanwhile, the existence of numerous nonlinear large motions in real-world scenarios makes the point cloud interpolation task more challenging. In light of these issues, we present NeuralPCI: an end-to-end 4D spatiotemporal Neural field for 3D Point Cloud Interpolation, which implicitly integrates multi-frame information to handle nonlinear large motions for both indoor and outdoor scenarios. Furthermore, we construct a new multi-frame point cloud interpolation dataset called NL-Drive for large nonlinear motions in autonomous driving scenes to better demonstrate the superiority of our method. Ultimately, NeuralPCI achieves state-of-the-art performance on both DHB (Dynamic Human Bodies) and NL-Drive datasets. Beyond the interpolation task, our method can be naturally extended to point cloud extrapolation, morphing, and auto-labeling, which indicates its substantial potential in other domains. Codes are available at https://github.com/ispc-lab/NeuralPCI.

        ----

        ## [90] RIAV-MVS: Recurrent-Indexing an Asymmetric Volume for Multi-View Stereo

        **Authors**: *Changjiang Cai, Pan Ji, Qingan Yan, Yi Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00095](https://doi.org/10.1109/CVPR52729.2023.00095)

        **Abstract**:

        This paper presents a learning-based method for multi-view depth estimation from posed images. Our core idea is a “learning-to-optimize” paradigm that iteratively indexes a plane-sweeping cost volume and regresses the depth map via a convolutional Gated Recurrent Unit (GRU). Since the cost volume plays a paramount role in encoding the multi-view geometry, we aim to improve its construction both at pixel- and frame- levels. At the pixel level, we propose to break the symmetry of the Siamese network (which is typically used in MVS to extract image features) by introducing a transformer block to the reference image (but not to the source images). Such an asymmetric volume allows the network to extract global features from the reference image to predict its depth map. Given potential inaccuracies in the poses between reference and source images, we propose to incorporate a residual pose network to correct the relative poses. This essentially rectifies the cost volume at the frame level. We conduct extensive experiments on real-world MVS datasets and show that our method achieves state-of-the-art performance in terms of both within-dataset evaluation and cross-dataset generalization.

        ----

        ## [91] NeuMap: Neural Coordinate Mapping by Auto-Transdecoder for Camera Localization

        **Authors**: *Shitao Tang, Sicong Tang, Andrea Tagliasacchi, Ping Tan, Yasutaka Furukawa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00096](https://doi.org/10.1109/CVPR52729.2023.00096)

        **Abstract**:

        This paper presents an end-to-end neural mapping method for camera localization, encoding a whole scene into a grid of latent codes, with which a Transformer-based auto-decoder regresses 3D coordinates of query pixels. State-of-the-art camera localization methods require each scene to be stored as a 3D point cloud with per-point features, which takes several gigabytes of storage per scene. While compression is possible, the performance drops significantly at high compression rates. NeuMap achieves extremely high compression rates with minimal performance drop by using 1) learnable latent codes to store scene information and 2) a scene-agnostic Transformer-based auto-decoder to infer coordinates for a query pixel. The scene-agnostic network design also learns robust matching priors by training with large-scale data, and further allows us to just optimize the codes quickly for a new scene while fixing the network weights. Extensive evaluations with five benchmarks show that NeuMap outperforms all the other coordinate regression methods significantly and reaches similar performance as the feature matching methods while having a much smaller scene representation size. For example, NeuMap achieves 39.1% accuracy in Aachen night benchmark with only 6MB of data, while other compelling methods require 100MB or a few gigabytes and fail completely under high compression settings. The codes are available at https://github.com/Tangshitao/NeuMap.

        ----

        ## [92] MACARONS: Mapping and Coverage Anticipation with RGB Online Self-Supervision

        **Authors**: *Antoine Guédon, Tom Monnier, Pascal Monasse, Vincent Lepetit*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00097](https://doi.org/10.1109/CVPR52729.2023.00097)

        **Abstract**:

        We introduce a method that simultaneously learns to explore new large environments and to reconstruct them in 3D from color images only. This is closely related to the Next Best View problem (NBV), where one has to identify where to move the camera next to improve the coverage of an unknown scene. However, most of the current NBV methods rely on depth sensors, need 3D supervision and/or do not scale to large scenes. Our method requires only a color camera and no 3D supervision. It simultaneously learns in a self-supervised fashion to predict a “volume occupancy field” from color images and, from this field, to predict the NBV. Thanks to this approach, our method performs well on new scenes as it is not biased towards any training 3D data. We demonstrate this on a recent dataset made of various 3D scenes and show it performs even better than recent methods requiring a depth sensor, which is not a realistic assumption for outdoor scenes captured with a flying drone.

        ----

        ## [93] vMAP: Vectorised Object Mapping for Neural Field SLAM

        **Authors**: *Xin Kong, Shikun Liu, Marwan Taher, Andrew J. Davison*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00098](https://doi.org/10.1109/CVPR52729.2023.00098)

        **Abstract**:

        We present vMAP, an object-level dense SLAM system using neural field representations. Each object is repre-sented by a small MLP, enabling efficient, watertight object modelling without the needfor 3D priors. As an RGB-D camera browses a scene with no prior in-formation, vMAP detects object instances on-the-fly, and dynamically adds them to its map. Specifically, thanks to the power of vectorised training, vMAP can optimise as many as 50 individual objects in a single scene, with an extremely efficient training speed of 5Hz map update. We experimentally demonstrate significantly improved scene-level and object-level reconstruction quality compared to prior neural field SLAM systems. Project page: https://kxhit.github.io/vMAP.

        ----

        ## [94] Seeing a Rose in Five Thousand Ways

        **Authors**: *Yunzhi Zhang, Shangzhe Wu, Noah Snavely, Jiajun Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00099](https://doi.org/10.1109/CVPR52729.2023.00099)

        **Abstract**:

        What is a rose, visually? A rose comprises its intrinsics, including the distribution of geometry, texture, and material specific to its object category. With knowledge of these intrinsic properties, we may render roses of different sizes and shapes, in different poses, and under different lighting conditions. In this work, we build a generative model that learns to capture such object intrinsics from a single image, such as a photo of a bouquet. Such an image includes multiple instances of an object type. These instances all share the same intrinsics, but appear different due to a combination of variance within these intrinsics and differences in extrinsic factors, such as pose and illumination. Experiments show that our model successfully learns object intrinsics (distribution of geometry, texture, and material) for a wide range of objects, each from a single Internet image. Our method achieves superior results on multiple downstream tasks, including intrinsic image decomposition, shape and image generation, view synthesis, and relighting. “Nature! … Each of her works has an essence of its own; each of her phenomena a special characterisation; and yet their diversity is in unity.” - Georg Christoph Tobler

        ----

        ## [95] Propagate and Calibrate: Real-Time Passive Non-Line-of-Sight Tracking

        **Authors**: *Yihao Wang, Zhigang Wang, Bin Zhao, Dong Wang, Mulin Chen, Xuelong Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00100](https://doi.org/10.1109/CVPR52729.2023.00100)

        **Abstract**:

        Non-line-of-sight (NLOS) tracking has drawn increasing attention in recent years, due to its ability to detect object motion out of sight. Most previous works on NLOS tracking rely on active illumination, e.g., laser, and suffer from high cost and elaborate experimental conditions. Besides, these techniques are still far from practical application due to oversimplified settings. In contrast, we propose a purely passive method to track a person walking in an invisible room by only observing a relay wall, which is more in line with real application scenarios, e.g., security. To excavate imperceptible changes in videos of the relay wall, we introduce difference frames as an essential carrier of temporal-local motion messages. In addition, we propose PAC-Net, which consists of alternating propagation and calibration, making it capable of leveraging both dynamic and static messages on a frame-level granularity. To evaluate the proposed method, we build and publish the first dynamic passive NLOS tracking dataset, NLOS-Track, which fills the vacuum of realistic NLOS datasets. NLOS-Track contains thousands of NLOS video clips and corresponding trajectories. Both real-shot and synthetic data are included. Our codes and dataset are available at https://againstentropy.github.io/NLOS-Track/.

        ----

        ## [96] Seeing With Sound: Long-Range Acoustic Beamforming for Multimodal Scene Understanding

        **Authors**: *Praneeth Chakravarthula, Jim Aldon D'Souza, Ethan Tseng, Joe Bartusek, Felix Heide*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00101](https://doi.org/10.1109/CVPR52729.2023.00101)

        **Abstract**:

        Mobile robots, including autonomous vehicles rely heavily on sensors that use electromagnetic radiation like lidars, radars and cameras for perception. While effective in most scenarios, these sensors can be unreliable in unfavorable environmental conditions, including low-light scenarios and adverse weather, and they can only detect obstacles within their direct line-of-sight. Audible sound from other road users propagates as acoustic waves that carry information even in challenging scenarios. However, their low spatial resolution and lack of directional information have made them an overlooked sensing modality. In this work, we introduce long-range acoustic beamforming of sound produced by road users in-the-wild as a complementary sensing modality to traditional electromagnetic radiation-based sensors. To validate our approach and encourage further work in the field, we also introduce the first-ever multimodallong-range acoustic beamforming dataset. We propose a neural aperture expansion method for beamforming and demonstrate its effectiveness for multimodal automotive object detection when coupled with RGB images in challenging automotive scenarios, where camera-only approaches fail or are unable to provide ultra-fast acoustic sensing sampling rates. Data and code can be found here 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
light.princeton.edu/seeingwithsound.

        ----

        ## [97] Distilling Focal Knowledge from Imperfect Expert for 3D Object Detection

        **Authors**: *Jia Zeng, Li Chen, Hanming Deng, Lewei Lu, Junchi Yan, Yu Qiao, Hongyang Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00102](https://doi.org/10.1109/CVPR52729.2023.00102)

        **Abstract**:

        Multi-camera 3D object detection blossoms in recent years and most of state-of-the-art methods are built up on the bird’ s-eye- view (BEV) representations. Albeit remarkable performance, these works suffer from low efficiency. Typically, knowledge distillation can be used for model compression. However, due to unclear 3D geometry reasoning, expert features usually contain some noisy and confusing areas. In this work, we investigate on how to distill the knowledge from an imperfect expert. We propose FD3D, a Focal Distiller for 3D object detection. Specifically, a set of queries are leveraged to locate the instance-level areas for masked feature generation, to intensify feature representation ability in these areas. Moreover, these queries search out the representative fine-grained positions for refined distillation. We verify the effectiveness of our method by applying it to two popular detection models, BEVFormer and DETR3D. The results demonstrate that our method achieves improvements of 4.07 and 3.17 points respectively in terms of NDS metric on nuScenes benchmark. Code is hosted at https://github.com/OpenPerceptionX/BEVPerception-Survey-Recipe.

        ----

        ## [98] AShapeFormer : Semantics-Guided Object-Level Active Shape Encoding for 3D Object Detection via Transformers

        **Authors**: *Zechuan Li, Hongshan Yu, Zhengeng Yang, Tom Tongjia Chen, Naveed Akhtar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00104](https://doi.org/10.1109/CVPR52729.2023.00104)

        **Abstract**:

        3D object detection techniques commonly follow a pipeline that aggregates predicted object central point features to compute candidate points. However, these candidate points contain only positional information, largely ignoring the object-level shape information. This eventually leads to sub-optimal 3D object detection. In this work, we propose AShapeFormer, a semantics-guided object-level shape encoding module for 3D object detection. This is a plug-n-play module that leverages multi-head attention to encode object shape information. We also propose shape tokens and object-scene positional encoding to ensure that the shape information is fully exploited. Moreover, we introduce a semantic guidance sub-module to sample more foreground points and suppress the influence of background points for a better object shape perception. We demonstrate a straightforward enhancement of multiple existing methods with our AShapeFormer. Through extensive experiments on the popular SUN RGB-D and ScanNetV2 dataset, we show that our enhanced models are able to outperform the baselines by a considerable absolute margin of up to 8.1%. Code will be available at https://github.com/ZechuanLi/AShapeFormer

        ----

        ## [99] Benchmarking Robustness of 3D Object Detection to Common Corruptions in Autonomous Driving

        **Authors**: *Yinpeng Dong, Caixin Kang, Jinlai Zhang, Zijian Zhu, Yikai Wang, Xiao Yang, Hang Su, Xingxing Wei, Jun Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00105](https://doi.org/10.1109/CVPR52729.2023.00105)

        **Abstract**:

        3D object detection is an important task in autonomous driving to perceive the surroundings. Despite the excellent performance, the existing 3D detectors lack the robustness to real-world corruptions caused by adverse weathers, sensor noises, etc., provoking concerns about the safety and reliability of autonomous driving systems. To comprehensively and rigorously benchmark the corruption robustness of 3D detectors, in this paper we design 27 types of common corruptions for both LiDAR and camera inputs considering realworld driving scenarios. By synthesizing these corruptions on public datasets, we establish three corruption robustness benchmarks-KITTI-C, nuScenes-C, and Waymo-C. Then, we conduct large-scale experiments on 24 diverse 3D object detection models to evaluate their corruption robustness. Based on the evaluation results, we draw several important findings, including: 1) motion-level corruptions are the most threatening ones that lead to significant performance drop of all models; 2) LiDAR-camerafusion models demonstrate better robustness; 3) camera-only models are extremely vulnerable to image corruptions, showing the indispensability of LiDAR point clouds. We release the benchmarks and codes at https://github.com/thu-ml/3D_Corruptions_AD to be helpful for future studies.

        ----

        ## [100] Gaussian Label Distribution Learning for Spherical Image Object Detection

        **Authors**: *Hang Xu, Xinyuan Liu, Qiang Zhao, Yike Ma, Chenggang Yan, Feng Dai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00106](https://doi.org/10.1109/CVPR52729.2023.00106)

        **Abstract**:

        Spherical image object detection emerges in many applications from virtual reality to robotics and automatic driving, while many existing detectors use 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$l_{n}$</tex>
 -norms loss for regression of spherical bounding boxes. There are two intrinsic flaws for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$l_{n}$</tex>
 -norms loss, i.e., independent optimization of parameters and inconsistency between metric (dominated by IoU) and loss. These problems are common in planar image detection but more significant in spherical image detection. Solution for these problems has been extensively discussed in planar image detection by using IoU loss and related variants. However, these solutions cannot be migrated to spherical image object detection due to the undifferentiable of the Spherical IoU (SphIoU). In this paper, we design a simple but effective regression loss based on Gaussian Label Distribution Learning (GLDL) for spherical image object detection. Besides, we observe that the scale of the object in a spherical image varies greatly. The huge differences among objects from different categories make the sample selection strategy based on SphIoU challenging. Therefore, we propose GLDL-ATSS as a better training sample selection strategy for objects of the spherical image, which can alleviate the drawback of IoU threshold-based strategy of scale-sample imbalance. Extensive results on various two datasets with different baseline detectors show the effectiveness of our approach.

        ----

        ## [101] Deep Depth Estimation from Thermal Image

        **Authors**: *Ukcheol Shin, Jinsun Park, In So Kweon*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00107](https://doi.org/10.1109/CVPR52729.2023.00107)

        **Abstract**:

        Robust and accurate geometric understanding against adverse weather conditions is one top prioritized conditions to achieve a high-level autonomy of self-driving cars. However, autonomous driving algorithms relying on the visible spectrum band are easily impacted by weather and lighting conditions. A long-wave infrared camera, also known as a thermal imaging camera, is a potential rescue to achieve high-level robustness. However, the missing necessities are the well-established large-scale dataset and public benchmark results. To this end, in this paper, we first built a large-scale Multi-Spectral Stereo (MS
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">2</sup>
) dataset, including stereo RGB, stereo NIR, stereo thermal, and stereo LiDAR data along with GNSS/IMU information. The collected dataset provides about 195K synchronized data pairs taken from city, residential, road, campus, and suburban areas in the morning, daytime, and nighttime under clear-sky, cloudy, and rainy conditions. Secondly, we conduct an exhaustive validation process of monocular and stereo depth estimation algorithms designed on visible spectrum bands to benchmark their performance in the thermal image domain. Lastly, we propose a unified depth network that effectively bridges monocular depth and stereo depth tasks from a conditional random field approach perspective. Our dataset and source code are available at https://github.com/UkcheolShin/MS2-MultiSpectralStereoDataset.

        ----

        ## [102] LidarGait: Benchmarking 3D Gait Recognition with Point Clouds

        **Authors**: *Chuanfu Shen, Fan Chao, Wei Wu, Rui Wang, George Q. Huang, Shiqi Yu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00108](https://doi.org/10.1109/CVPR52729.2023.00108)

        **Abstract**:

        Video-based gait recognition has achieved impressive results in constrained scenarios. However, visual cameras neglect human 3D structure information, which limits the feasibility of gait recognition in the 3D wild world. Instead of extracting gait features from images, this work explores precise 3D gait features from point clouds and proposes a simple yet efficient 3D gait recognition framework, termed LidarGait. Our proposed approach projects sparse point clouds into depth maps to learn the representations with 3D geometry information, which outperforms existing point-wise and camera-based methods by a significant margin. Due to the lack of point cloud datasets, we build the first large-scale LiDAR-based gait recognition dataset, SUSTech1K, collected by a LiDAR sensor and an RGB camera. The dataset contains 25,239 sequences from 1,050 subjects and covers many variations, including visibility, views, occlusions, clothing, carrying, and scenes. Extensive experiments show that (1) 3D structure information serves as a significant feature for gait recognition. (2) LidarGait outperforms existing point-based and silhouette-based methods by a significant margin, while it also offers stable cross-view results. (3) The LiDAR sensor is superior to the RGB camera for gait recognition in the outdoor environment. The source code and dataset have been made available at https://lidargait.github.io.

        ----

        ## [103] Generalized UAV Object Detection via Frequency Domain Disentanglement

        **Authors**: *Kunyu Wang, Xueyang Fu, Yukun Huang, Chengzhi Cao, Gege Shi, Zheng-Jun Zha*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00109](https://doi.org/10.1109/CVPR52729.2023.00109)

        **Abstract**:

        When deploying the Unmanned Aerial Vehicles object detection (UAV-OD) network to complex and unseen real-world scenarios, the generalization ability is usually reduced due to the domain shift. To address this issue, this paper proposes a novel frequency domain disentanglement method to improve the UAV-OD generalization. Specifically, we first verified that the spectrum of different bands in the image has different effects to the UAV-OD generalization. Based on this conclusion, we design two learnable filters to extract domain-invariant spectrum and domain-specific spectrum, respectively. The former can be used to train the UAV-OD network and improve its capacity for generalization. In addition, we design a new instance-level contrastive loss to guide the network training. This loss enables the network to concentrate on extracting domaininvariant spectrum and domain-specific spectrum, so as to achieve better disentangling results. Experimental results on three unseen target domains demonstrate that our method has better generalization ability than both the baseline method and state-of-the-art methods.

        ----

        ## [104] Learning Compact Representations for LiDAR Completion and Generation

        **Authors**: *Yuwen Xiong, Wei-Chiu Ma, Jingkang Wang, Raquel Urtasun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00110](https://doi.org/10.1109/CVPR52729.2023.00110)

        **Abstract**:

        LiDAR provides accurate geometric measurements of the 3D world. Unfortunately, dense LiDARs are very expensive and the point clouds captured by low-beam LiDAR are often sparse. To address these issues, we present UltraLiDAR, a data-driven framework for scene-level LiDAR completion, LiDAR generation, and LiDAR manipulation. The crux of UltraLiDAR is a compact, discrete representation that encodes the point cloud's geometric structure, is robust to noise, and is easy to manipulate. We show that by aligning the representation of a sparse point cloud to that of a dense point cloud, we can densify the sparse point clouds as if they were captured by a real high-density LiDAR, drastically reducing the cost. Furthermore, by learning a prior over the discrete codebook, we can generate diverse, realistic LiDAR point clouds for self-driving. We evaluate the effectiveness of UltraLiDAR on sparse-to-dense LiDAR completion and LiDAR generation. Experiments show that densifying real-world point clouds with our approach can significantly improve the performance of downstream perception systems. Compared to prior art on LiDAR generation, our approach generates much more realistic point clouds. According to A/B test, over 98.5% of the time human participants prefer our results over those of previous methods. Please refer to project page https://waabi.ai/research/ultralidar/ for more information.

        ----

        ## [105] CXTrack: Improving 3D Point Cloud Tracking with Contextual Information

        **Authors**: *Tian-Xing Xu, Yuan-Chen Guo, Yu-Kun Lai, Song-Hai Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00111](https://doi.org/10.1109/CVPR52729.2023.00111)

        **Abstract**:

        3D single object tracking plays an essential role in many applications, such as autonomous driving. It remains a challenging problem due to the large appearance variation and the sparsity of points caused by occlusion and lim-ited sensor capabilities. Therefore, contextual information across two consecutive frames is crucial for effective object tracking. However, points containing such useful information are often overlooked and cropped out in existing methods, leading to insufficient use of important contextual knowledge. To address this issue, we propose CXTrack, a novel transformer-based network for 3D object tracking, which exploits ConteXtual information to improve the tracking results. Specifically, we design a target-centric transformer network that directly takes point features from two consecutive frames and the previous bounding box as input to explore contextual information and implicitly propagate target cues. To achieve accurate localization for objects of all sizes, we propose a transformer-based localization head with a novel center embedding module to distinguish the target from distractors. Extensive experiments on three large-scale datasets, KITTI, nuScenes and Waymo Open Dataset, show that CXTrack achieves state-of-the-art tracking performance while running at 34 FPS.

        ----

        ## [106] Multispectral Video Semantic Segmentation: A Benchmark Dataset and Baseline

        **Authors**: *Wei Ji, Jingjing Li, Cheng Bian, Zongwei Zhou, Jiaying Zhao, Alan L. Yuille, Li Cheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00112](https://doi.org/10.1109/CVPR52729.2023.00112)

        **Abstract**:

        Robust and reliable semantic segmentation in complex scenes is crucial for many real-life applications such as autonomous safe driving and nighttime rescue. In most approaches, it is typical to make use of RGB images as input. They however work well only in preferred weather conditions; when facing adverse conditions such as rainy, overexposure, or low-light, they often fail to deliver satisfactory results. This has led to the recent investigation into multispectral semantic segmentation, where RGB and thermal infrared (RGBT) images are both utilized as input. This gives rise to significantly more robust segmentation of image objects in complex scenes and under adverse conditions. Nevertheless, the present focus in single RGBT image input restricts existing methods from well addressing dynamic real-world scenes. Motivated by the above observations, in this paper, we set out to address a relatively new task of semantic segmentation of multispectral video input, which we refer to as Multispectral Video Semantic Segmentation, or MVSS in short. An in-house MVSeg dataset is thus curated, consisting of 738 calibrated RGB and thermal videos, accompanied by 3,545 fine-grained pixel-level semantic annotations of 26 categories. Our dataset contains a wide range of challenging urban scenes in both daytime and nighttime. Moreover, we propose an effective MVSS baseline, dubbed MVNet, which is to our knowledge the first model to jointly learn semantic representations from multispectral and temporal contexts. Comprehensive experiments are conducted using various semantic segmentation models on the MVSeg dataset. Empirically, the engagement of multispectral video input is shown to lead to significant improvement in semantic segmentation; the effectiveness of our MVNet baseline has also been verified.

        ----

        ## [107] LinK: Linear Kernel for LiDAR-based 3D Perception

        **Authors**: *Tao Lu, Xiang Ding, Haisong Liu, Gangshan Wu, Limin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00113](https://doi.org/10.1109/CVPR52729.2023.00113)

        **Abstract**:

        Extending the success of 2D Large Kernel to 3D perception is challenging due to: 1. the cubically-increasing overhead in processing 3D data; 2. the optimization difficulties from data scarcity and sparsity. Previous work has taken the first step to scale up the kernel size from 3 × 3 × 3 to 7 × 7 × 7 by introducing block-shared weights. However, to reduce the feature variations within a block, it only employs modest block size and fails to achieve larger kernels like the 21 × 21 × 21. To address this issue, we propose a new method, called LinK, to achieve a wider-range perception receptive field in a convolution-like manner with two core designs. The first is to replace the static kernel matrix with a linear kernel generator, which adaptively provides weights only for non-empty voxels. The second is to reuse the pre-computed aggregation results in the overlapped blocks to reduce computation complexity. The proposed method successfully enables each voxel to perceive context within a range of 21 × 21 × 21. Extensive experiments on two basic perception tasks, 3D object detection and 3D semantic segmentation, demonstrate the effectiveness of our method. Notably, we rank 1st on the public leaderboard of the 3D detection benchmark of nuScenes (LiDAR track), by simply incorporating a LinK-based backbone into the basic detector, CenterPoint. We also boost the strong segmentation baseline's mIoU with 2.7% in the SemanticKITTI test set. Code is available at https://github.com/MCG-NJU/LinK.

        ----

        ## [108] Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting

        **Authors**: *Tarasha Khurana, Peiyun Hu, David Held, Deva Ramanan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00114](https://doi.org/10.1109/CVPR52729.2023.00114)

        **Abstract**:

        Predicting how the world can evolve in the future is crucial for motion planning in autonomous systems. Classical methods are limited because they rely on costly human annotations in the form of semantic class labels, bounding boxes, and tracks or HD maps of cities to plan their motion - and thus are difficult to scale to large unlabeled datasets. One promising self-supervised task is 3D point cloud forecasting [11, 18–20] from unannotated LiDAR sequences. We show that this task requires algorithms to implicitly capture (1) sensor extrinsics (i.e., the egomotion of the autonomous vehicle), (2) sensor intrinsics (i.e., the sampling pattern specific to the particular LiDAR sensor), and (3) the shape and motion of other objects in the scene. But autonomous systems should make predictions about the world and not their sensors! To this end, we factor out (1) and (2) by recasting the task as one of spacetime (4D) occupancy forecasting. But because it is expensive to obtain ground-truth 4D occupancy, we “render” point cloud data from 4D occupancy predictions given sensor extrinsics and intrinsics, allowing one to train and test occupancy algorithms with unannotated LiDAR sequences. This also allows one to evaluate and compare point cloud forecasting algorithms across diverse datasets, sensors, and vehicles.

        ----

        ## [109] Curricular Object Manipulation in LiDAR-based Object Detection

        **Authors**: *Ziyue Zhu, Qiang Meng, Xiao Wang, Ke Wang, Liujiang Yan, Jian Yang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00115](https://doi.org/10.1109/CVPR52729.2023.00115)

        **Abstract**:

        This paper explores the potential of curriculum learning in LiDAR-based 3D object detection by proposing a curricular object manipulation (COM) framework. The framework embeds the curricular training strategy into both the loss design and the augmentation process. For the loss design, we propose the COMLoss to dynamically predict object-level difficulties and emphasize objects of different difficulties based on training stages. On top of the widely-used augmentation technique called GT-Aug in Li-DAR detection tasks, we propose a novel COMAug strategy which first clusters objects in ground-truth database based on well-designed heuristics. Group-level difficulties rather than individual ones are then predicted and updated during training for stable results. Model performance and generalization capabilities can be improved by sampling and augmenting progressively more difficult objects into the training samples. Extensive experiments and ablation studies reveal the superior and generality of the proposed framework. The code is available at https://github.com/ZZY816/COM.

        ----

        ## [110] Delivering Arbitrary-Modal Semantic Segmentation

        **Authors**: *Jiaming Zhang, Ruiping Liu, Hao Shi, Kailun Yang, Simon Reiß, Kunyu Peng, Haodong Fu, Kaiwei Wang, Rainer Stiefelhagen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00116](https://doi.org/10.1109/CVPR52729.2023.00116)

        **Abstract**:

        Multimodal fusion can make semantic segmentation more robust. However, fusing an arbitrary number of modalities remains underexplored. To delve into this problem, we create the Deliver arbitrary-modal segmentation benchmark, covering Depth, LiDAR, multiple Views, Events, and RGB. Aside from this, we provide this dataset in four severe weather conditions as well as five sensor failure cases to exploit modal complementarity and resolve partial outages. To make this possible, we present the arbitrary cross-modal segmentation model CMNEXT. It encompasses a Self-Query Hub (SQ-Hub) designed to extract effective information from any modality for subsequent fusion with the RGB representation and adds only negligible amounts of parameters ~0.1M) per additional modality. On top, to efficiently and flexibly harvest discriminative cues from the auxiliary modalities, we introduce the simple Parallel Pooling Mixer (PPX). With extensive experiments on a total of six benchmarks, our CMNEXT achieves state-of-the-art performance on the Deliver, Kitti-360, MFNet, NYU Depth V2, UrbanLF, and MCubeS datasets, allowing to scale from 1 to 81 modalities. On the freshly collected Deliver, the quad-modal CMNEXT reaches up to 66.30% in mIoU with a +9.10% gain as compared to the mono-modal baseline.
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The Deliver dataset and our code will be made publicly available at https://jamycheung.github.io/DELIVER.html.

        ----

        ## [111] Robust Outlier Rejection for 3D Registration with Variational Bayes

        **Authors**: *Haobo Jiang, Zheng Dang, Zhen Wei, Jin Xie, Jian Yang, Mathieu Salzmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00117](https://doi.org/10.1109/CVPR52729.2023.00117)

        **Abstract**:

        Learning-based outlier (mismatched correspondence) rejection for robust 3D registration generally formulates the outlier removal as an inlier/outlier classification problem. The core for this to be successful is to learn the discriminative inlier/outlier feature representations. In this paper, we develop a novel variational non-local network-based outlier rejection framework for robust alignment. By reformulating the non-local feature learning with variational Bayesian inference, the Bayesian-driven long-range dependencies can be modeled to aggregate discriminative geometric context information for inlier/outlier distinction. Specifically, to achieve such Bayesian-driven contextual dependencies, each query/key/value component in our nonlocal network predicts a prior feature distribution and a posterior one. Embedded with the inlier/outlier label, the posterior feature distribution is label-dependent and discriminative. Thus, pushing the prior to be close to the discriminative posterior in the training step enables the features sampled from this prior at test time to model highquality long-range dependencies. Notably, to achieve effective posterior feature guidance, a specific probabilistic graphical model is designed over our non-local model, which lets us derive a variational low bound as our optimization objective for model training. Finally, we propose a voting-based inlier searching strategy to cluster the high-quality hypothetical inliers for transformation estimation. Extensive experiments on 3DMatch, 3DLoMatch, and KITTI datasets verify the effectiveness of our method. Code is available at https://github.com/Jiang-HB/VBReg.

        ----

        ## [112] 3D Human Keypoints Estimation from Point Clouds in the Wild without Human Labels

        **Authors**: *Zhenzhen Weng, Alexander S. Gorban, Jingwei Ji, Mahyar Najibi, Yin Zhou, Dragomir Anguelov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00118](https://doi.org/10.1109/CVPR52729.2023.00118)

        **Abstract**:

        Training a 3D human keypoint detector from point clouds in a supervised manner requires large volumes of high quality labels. While it is relatively easy to capture large amounts of human point clouds, annotating 3D key-points is expensive, subjective, error prone and especially difficult for long-tail cases (pedestrians with rare poses, scooterists, etc.). In this work, we propose GC-KPL - Geometry Consistency inspired Key Point Leaning, an approach for learning 3D human joint locations from point clouds without human labels. We achieve this by our novel unsupervised loss formulations that account for the structure and movement of the human body. We show that by training on a large training set from Waymo Open Dataset [21] without any human annotated keypoints, we are able to achieve reasonable performance as compared to the fully supervised approach. Further, the backbone benefits from the unsupervised training and is useful in downstream few-shot learning of keypoints, where fine-tuning on only 10 percent of the labeled training data gives comparable performance to fine-tuning on the entire set. We demonstrated that GC-KPL outperforms by a large margin over SoTA when trained on entire dataset and efficiently leverages large volumes of unlabeled data.

        ----

        ## [113] Self-Supervised Pre-Training with Masked Shape Prediction for 3D Scene Understanding

        **Authors**: *Li Jiang, Zetong Yang, Shaoshuai Shi, Vladislav Golyanik, Dengxin Dai, Bernt Schiele*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00119](https://doi.org/10.1109/CVPR52729.2023.00119)

        **Abstract**:

        Masked signal modeling has greatly advanced self-supervised pre-training for language and 2D images. However, it is still not fully explored in 3D scene understanding. Thus, this paper introduces Masked Shape Prediction (MSP), a new framework to conduct masked signal modeling in 3D scenes. MSP uses the essential 3D semantic cue, i.e., geometric shape, as the prediction target for masked points. The context-enhanced shape target consisting of explicit shape context and implicit deep shape feature is proposed to facilitate exploiting contextual cues in shape prediction. Meanwhile, the pre-training architecture in MSP is carefully designed to alleviate the masked shape leakage from point coordinates. Experiments on multiple 3D understanding tasks on both indoor and outdoor datasets demon-strate the effectiveness of MSP in learning good feature representations to consistently boost downstream performance.

        ----

        ## [114] ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding

        **Authors**: *Le Xue, Mingfei Gao, Chen Xing, Roberto Martín-Martín, Jiajun Wu, Caiming Xiong, Ran Xu, Juan Carlos Niebles, Silvio Savarese*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00120](https://doi.org/10.1109/CVPR52729.2023.00120)

        **Abstract**:

        The recognition capabilities of current state-of-the-art 3D models are limited by datasets with a small number of annotated data and a pre-defined set of categories. In its 2D counterpart, recent advances have shown that similar problems can be significantly alleviated by employing knowledge from other modalities, such as language. Inspired by this, leveraging multimodal information for 3D modality could be promising to improve 3D understanding under the restricted data regime, but this line of research is not well studied. Therefore, we introduce ULIP to learn a unified representation of image, text, and 3D point cloud by pre-training with object triplets from the three modalities. To overcome the shortage of training triplets, ULIP leverages a pre-trained vision-language model that has already learned a common visual and textual space by training with massive image-text pairs. Then, ULIP learns a 3D representation space aligned with the common image-text space, using a small number of automatically synthesized triplets. ULIP is agnostic to 3D backbone networks and can easily be integrated into any 3D architecture. Experiments show that ULIP effectively improves the performance of multiple recent 3D backbones by simply pre-training them on ShapeNet55 using our framework, achieving state-of-the-art performance in both standard 3D classification and zero-shot 3D classification on ModelNet40 and ScanObjectNN. ULIP also improves the performance of PointMLP by around 3% in 3D classification on ScanObjectNN, and outperforms PointCLIP by 28.8% on top-1 accuracy for zero-shot 3D classification on ModelNet40. Our code and pre-trained models will be released.

        ----

        ## [115] Open-Vocabulary Point-Cloud Object Detection without 3D Annotation

        **Authors**: *Yuheng Lu, Chenfeng Xu, Xiaobao Wei, Xiaodong Xie, Masayoshi Tomizuka, Kurt Keutzer, Shanghang Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00121](https://doi.org/10.1109/CVPR52729.2023.00121)

        **Abstract**:

        The goal of open-vocabulary detection is to identify novel objects based on arbitrary textual descriptions. In this paper, we address open-vocabulary 3D point-cloud detection by a dividing-and-conquering strategy, which involves: 1) developing a point-cloud detector that can learn a general representation for localizing various objects, and 2) connecting textual and point-cloud representations to enable the detector to classify novel object categories based on text prompting. Specifically, we resort to rich image pretrained models, by which the point-cloud detector learns localizing objects under the supervision of predicted 2D bounding boxes from 2D pretrained detectors. Moreover, we propose a novel de-biased triplet cross-modal contrastive learning to connect the modalities of image, point-cloud and text, thereby enabling the point-cloud detector to benefit from vision-language pretrained models, i.e., CLIP. The novel use of image and vision-language pretrained models for point-cloud detectors allows for open-vocabulary 3D object detection without the need for 3D annotations. Experiments demonstrate that the proposed method improves at least 3.03 points and 7.47 points over a wide range of baselines on the ScanNet and SUN RGB-D datasets, respectively. Furthermore, we provide a comprehensive analysis to explain why our approach works. Code is available at https://github.com/lyhdet/OV-3DET

        ----

        ## [116] FlatFormer: Flattened Window Attention for Efficient Point Cloud Transformer

        **Authors**: *Zhijian Liu, Xinyu Yang, Haotian Tang, Shang Yang, Song Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00122](https://doi.org/10.1109/CVPR52729.2023.00122)

        **Abstract**:

        Transformer, as an alternative to CNN, has been proven effective in many modalities (e.g., texts and images). For 3D point cloud transformers, existing efforts focus primarily on pushing their accuracy to the state-of-the-art level. However, their latency lags behind sparse convolution-based models (3 × slower), hindering their usage in resource-constrained, latency-sensitive applications (such as autonomous driving). This inefficiency comes from point clouds' sparse and irregular nature, whereas transformers are designed for dense, regular workloads. This paper presents FlatFormer to close this latency gap by trading spatial proximity for better computational regularity. We first flatten the point cloud with window-based sorting and partition points into groups of equal sizes rather than windows of equal shapes. This effectively avoids expensive structuring and padding overheads. We then apply self-attention within groups to extract local features, alternate sorting axis to gather features from different directions, and shift windows to exchange features across groups. FlatFormer delivers state-of-the-art accuracy on Waymo Open Dataset with 4.6× speedup over (transformer-based) SST and 1.4× speedup over (sparse convolutional) CenterPoint. This is the first point cloud transformer that achieves real-time performance on edge GPUs and is faster than sparse convolutional methods while achieving on-par or even superior accuracy on large-scale benchmarks.

        ----

        ## [117] PointCMP: Contrastive Mask Prediction for Self-supervised Learning on Point Cloud Videos

        **Authors**: *Zhiqiang Shen, Xiaoxiao Sheng, Longguang Wang, Yulan Guo, Qiong Liu, Xi Zhou*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00123](https://doi.org/10.1109/CVPR52729.2023.00123)

        **Abstract**:

        Self-supervised learning can extract representations of good quality from solely unlabeled data, which is ap-pealing for point cloud videos due to their high labelling cost. In this paper, we propose a contrastive mask prediction (PointCMP) framework for self-supervised learning on point cloud videos. Specifically, our PointCMP employs a two-branch structure to achieve simultaneous learning of both local and global spatiotemporal information. On top of this two-branch structure, a mutual similarity based augmentation module is developed to synthesize hard samples at the feature level. By masking dominant tokens and erasing principal channels, we generate hard samples to facilitate learning representations with better discrimi-nation and generalization performance. Extensive experiments show that our PointCMP achieves the state-of-the-art performance on benchmark datasets and outperforms existing full-supervised counterparts. Transfer learning results demonstrate the superiority of the learned representations across different datasets and tasks.

        ----

        ## [118] E2PN: Efficient SE(3)-Equivariant Point Network

        **Authors**: *Minghan Zhu, Maani Ghaffari, William A. Clark, Huei Peng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00124](https://doi.org/10.1109/CVPR52729.2023.00124)

        **Abstract**:

        This paper proposes a convolution structure for learning SE(3)-equivariant features from 3D point clouds. It can be viewed as an equivariant version of kernel point convolutions (KPConv), a widely used convolution form to process point cloud data. Compared with existing equivariant networks, our design is simple, lightweight, fast, and easy to be integrated with existing task-specific point cloud learning pipelines. We achieve these desirable properties by combining group convolutions and quotient representations. Specifically, we discretize SO(3) to finite groups for their simplicity while using SO(2) as the stabilizer subgroup to form spherical quotient feature fields to save computations. We also propose a permutation layer to recover SO(3) features from spherical features to preserve the capacity to distinguish rotations. Experiments show that our method achieves comparable or superior performance in various tasks, including object classification, pose estimation, and keypoint-matching, while consuming much less memory and running faster than existing work. The proposed method can foster the development of equivariant models for real-world applications based on point clouds.

        ----

        ## [119] Poly-PC: A Polyhedral Network for Multiple Point Cloud Tasks at Once

        **Authors**: *Tao Xie, Shiguang Wang, Ke Wang, Linqi Yang, Zhiqiang Jiang, Xingcheng Zhang, Kun Dai, Ruifeng Li, Jian Cheng*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00125](https://doi.org/10.1109/CVPR52729.2023.00125)

        **Abstract**:

        In this work, we show that it is feasible to perform multiple tasks concurrently on point cloud with a straightforward yet effective multi-task network. Our framework, Poly-PC, tackles the inherent obstacles (e.g., different model architectures caused by task bias and conflicting gradients caused by multiple dataset domains, etc.) of multi-task learning on point cloud. Specifically, we propose a residual set abstraction (Res-SA) layer for efficient and effective scaling in both width and depth of the network, hence accommodating the needs of various tasks. We develop a weight-entanglement- based one-shot NAS technique to find optimal architectures for all tasks. Moreover, such technique entangles the weights of multiple tasks in each layer to offer task-shared parameters for efficient storage deployment while providing ancillary task-specific parameters for learning task-related features. Finally, to facilitate the training of Poly-PC, we introduce a task-prioritization-based gradient balance algorithm that leverages task prioritization to reconcile conflicting gradients, ensuring high performance for all tasks. Benefiting from the suggested techniques, models optimized by Poly-PC collectively for all tasks keep fewer total FLOPs and parameters and outperform previous methods. We also demonstrate that Poly-PC allows incremental learning and evades catastrophic forgetting when tuned to a new task.

        ----

        ## [120] Improving Graph Representation for Point Cloud Segmentation via Attentive Filtering

        **Authors**: *Nan Zhang, Zhiyi Pan, Thomas H. Li, Wei Gao, Ge Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00126](https://doi.org/10.1109/CVPR52729.2023.00126)

        **Abstract**:

        Recently, self-attention networks achieve impressive performance in point cloud segmentation due to their superiority in modeling long-range dependencies. However, compared to self-attention mechanism, we find graph convolutions show a stronger ability in capturing local geometry information with less computational cost. In this paper, we employ a hybrid architecture design to construct our Graph Convolution Network with Attentive Filtering (AF-GCN), which takes advantage of both graph convolution and selfattention mechanism. We adopt graph convolutions to aggregate local features in the shallow encoder stages, while in the deeper stages, we propose a self-attention-like module named Graph Attentive Filter (GAF) to better model long-range contexts from distant neighbors. Besides, to further improve graph representation for point cloud segmentation, we employ a Spatial Feature Projection (SFP) module for graph convolutions which helps to handle spatial variations of unstructured point clouds. Finally, a graphshared down-sampling and up-sampling strategy is introduced to make full use of the graph structures in point cloud processing. We conduct extensive experiments on multiple datasets including S3DIS, ScanNetV2, Toronto-3D, and ShapeNetPart. Experimental results show our AF-GCN obtains competitive performance.

        ----

        ## [121] BUFFER: Balancing Accuracy, Efficiency, and Generalizability in Point Cloud Registration

        **Authors**: *Sheng Ao, Qingyong Hu, Hanyun Wang, Kai Xu, Yulan Guo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00127](https://doi.org/10.1109/CVPR52729.2023.00127)

        **Abstract**:

        An ideal point cloud registration framework should have superior accuracy, acceptable efficiency, and strong generalizability: However, this is highly challenging since existing registration techniques are either not accurate enough, far from efficient, or generalized poorly. It remains an open question that how to achieve a satisfying balance between this three key elements. In this paper, we propose BUFFER, a point cloud registration method for balancing accuracy, efficiency, and generalizability. The key to our approach is to take advantage of both point-wise and patch-wise techniques, while overcoming the inherent drawbacks simultaneously. Different from a simple combination of existing methods, each component of our network has been carefully crafted to tackle specific issues. Specifically, a Point-wise Learner is first introduced to enhance computational efficiency by predicting keypoints and improving the representation capacity of features by estimating point orientations, a Patch-wise Embedder which leverages a lightweight local feature learner is then deployed to extract efficient and general patch features. Additionally, an Inliers Generator which combines simple neural layers and general features is presented to search inlier correspondences. Extensive experiments on real-world scenarios demonstrate that our method achieves the best of both worlds in accuracy, efficiency, and generalization. In particular, our method not only reaches the highest success rate on unseen domains, but also is almost 30 times faster than the strong baselines specializing in generalization. Code is available at https://github.com/aosheng1996/BUFFER.

        ----

        ## [122] TopDiG: Class-agnostic Topological Directional Graph Extraction from Remote Sensing Images

        **Authors**: *Bingnan Yang, Mi Zhang, Zhan Zhang, Zhili Zhang, Xiangyun Hu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00128](https://doi.org/10.1109/CVPR52729.2023.00128)

        **Abstract**:

        Rapid development in automatic vector extraction from remote sensing images has been witnessed in recent years. However, the vast majority of existing works concentrate on a specific target, fragile to category variety, and hardly achieve stable performance crossing different categories. In this work, we propose an innovative class-agnostic model, namely TopDiG, to directly extract topological directional graphs from remote sensing images and solve these issues. Firstly, TopDiG employs a topology-concentrated node detector (TCND) to detect nodes and obtain compact perception of topological components. Secondly, we propose a dynamic graph supervision (DGS) strategy to dynamically generate adjacency graph labels from unordered nodes. Finally, the directional graph (DiG) generator module is designed to construct topological directional graphs from predicted nodes. Experiments on the Inria, CrowdAI, GID, GF2 and Massachusetts datasets empirically demonstrate that TopDiG is class-agnostic and achieves competitive performance on all datasets.

        ----

        ## [123] Recognizing Rigid Patterns of Unlabeled Point Clouds by Complete and Continuous Isometry Invariants with no False Negatives and no False Positives

        **Authors**: *Daniel Widdowson, Vitaliy Kurlin*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00129](https://doi.org/10.1109/CVPR52729.2023.00129)

        **Abstract**:

        Rigid structures such as cars or any other solid objects are often represented by finite clouds of unlabeled points. The most natural equivalence on these point clouds is rigid motion or isometry maintaining all inter-point distances. Rigid patterns of point clouds can be reliably compared only by complete isometry invariants that can also be called equivariant descriptors without false negatives (isometric clouds having different descriptions) and without false positives (non-isometric clouds with the same description). Noise and motion in data motivate a search for invariants that are continuous under perturbations of points in a suitable metric. We propose the first continuous and complete invariant of unlabeled clouds in any Euclidean space. For a fixed dimension, the new metric for this invariant is computable in a polynomial time in the number of points.

        ----

        ## [124] Both Style and Distortion Matter: Dual-Path Unsupervised Domain Adaptation for Panoramic Semantic Segmentation

        **Authors**: *Xu Zheng, Jinjing Zhu, Yexin Liu, Zidong Cao, Chong Fu, Lin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00130](https://doi.org/10.1109/CVPR52729.2023.00130)

        **Abstract**:

        The ability of scene understanding has sparked active research for panoramic image semantic segmentation. However, the performance is hampered by distortion of the equirectangular projection (ERP) and a lack of pixel-wise annotations. For this reason, some works treat the ERP and pinhole images equally and transfer knowledge from the pinhole to ERP images via unsupervised domain adaptation (UDA). However, they fail to handle the domain gaps caused by: 1) the inherent differences between camera sensors and captured scenes; 2) the distinct image formats (e.g., ERP and pinhole images). In this paper, we propose a novel yet flexible dual-path UDA framework, DPPASS, taking ERP and tangent projection (TP) images as inputs. To reduce the domain gaps, we propose cross-projection and intra-projection training. The cross-projection training includes tangent-wise feature contrastive training and prediction consistency training. That is, the former formulates the features with the same projection locations as positive examples and vice versa, for the models' awareness of distortion, while the latter ensures the consistency of cross-model predictions between the ERP and TP. Moreover, adversarial intra-projection training is proposed to reduce the inherent gap, between the features of the pinhole images and those of the ERP and TP images, respectively. Importantly, the TP path can be freely removed after training, leading to no additional inference cost. Extensive experiments on two benchmarks show that our DPPASS achieves + 1.06% mIoU increment than the state-of-the-art approaches. https://vlis2022.github.io/cvpr23/DPPASS

        ----

        ## [125] CCuantuMM: Cycle-Consistent Quantum-Hybrid Matching of Multiple Shapes

        **Authors**: *Harshil Bhatia, Edith Tretschk, Zorah Lähner, Marcel Seelbach Benkner, Michael Moeller, Christian Theobalt, Vladislav Golyanik*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00131](https://doi.org/10.1109/CVPR52729.2023.00131)

        **Abstract**:

        Jointly matching multiple, non-rigidly deformed 3D shapes is a challenging, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{NP}$</tex>
-hard problem. A perfect matching is necessarily cycle-consistent: Following the pairwise point correspondences along several shapes must end up at the starting vertex of the original shape. Unfortunately, existing quantum shape-matching methods do not support multiple shapes and even less cycle consistency. This paper addresses the open challenges and introduces the first quantum-hybrid approach for 3D shape multi-matching; in addition, it is also cycle-consistent. Its iterative formulation is admissible to modern adiabatic quantum hardware and scales linearly with the total number of input shapes. Both these characteristics are achieved by reducing the N-shape case to a sequence of three-shape matchings, the derivation of which is our main technical contribution. Thanks to quantum annealing, high-quality solutions with low energy are retrieved for the intermediate 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mathcal{NP}$</tex>
 - hard objectives. On benchmark datasets, the proposed approach significantly outperforms extensions to multi-shape matching of a previous quantum-hybrid two-shape matching method and is on-par with classical multi-matching methods. Our source code is available at 4dqv.mpiinf.mpg.de/CCuantuMM/.

        ----

        ## [126] Enhancing Deformable Local Features by Jointly Learning to Detect and Describe Keypoints

        **Authors**: *Guilherme A. Potje, Felipe Cadar, André Araújo, Renato Martins, Erickson R. Nascimento*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00132](https://doi.org/10.1109/CVPR52729.2023.00132)

        **Abstract**:

        Local feature extraction is a standard approach in computer vision for tackling important tasks such as image matching and retrieval. The core assumption of most methods is that images undergo affine transformations, disregarding more complicated effects such as non-rigid deformations. Furthermore, incipient works tailored for non-rigid correspondence still rely on keypoint detectors designed for rigid transformations, hindering performance due to the limitations of the detector. We propose DALF (Deformation-Aware Local Features), a novel deformation-aware network for jointly detecting and describing keypoints, to handle the challenging problem of matching deformable surfaces. All network components work cooperatively through a feature fusion approach that enforces the descriptors' distinctiveness and invariance. Experiments using real deforming objects showcase the superiority of our method, where it delivers 8% improvement in matching scores compared to the previous best results. Our approach also enhances the performance of two real-world applications: deformable object retrieval and non-rigid 3D surface registration. Code for training, inference, and applications are publicly available at verlab.dcc.ufmg.br/descriptors/dalf_cvpr23.

        ----

        ## [127] Understanding and Improving Features Learned in Deep Functional Maps

        **Authors**: *Souhaib Attaiki, Maks Ovsjanikov*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00133](https://doi.org/10.1109/CVPR52729.2023.00133)

        **Abstract**:

        Deep functional maps have recently emerged as a successful paradigm for non-rigid 3D shape correspondence tasks. An essential step in this pipeline consists in learning feature functions that are used as constraints to solve for a functional map inside the network. However, the precise nature of the information learned and stored in these functions is not yet well understood. Specifically, a major question is whether these features can be used for any other objective, apart from their purely algebraic role in solving for functional map matrices. In this paper, we show that under some mild conditions, the features learned within deep functional map approaches can be used as point-wise descriptors and thus are directly comparable across different shapes, even without the necessity of solving for a functional map at test time. Furthermore, informed by our analysis, we propose effective modifications to the standard deep functional map pipeline, which promote structural properties of learned features, significantly improving the matching results. Finally, we demonstrate that previously unsuccessful attempts at using extrinsic architectures for deep functional map feature extraction can be remedied via simple architectural changes, which encourage the theoretical properties suggested by our analysis. We thus bridge the gap between intrinsic and extrinsic surface-based learning, suggesting the necessary and sufficient conditions for successful shape matching. Our code is available at https://github.com/pvnieo/clover.

        ----

        ## [128] High-Frequency Stereo Matching Network

        **Authors**: *Haoliang Zhao, Huizhou Zhou, Yongjun Zhang, Jie Chen, Yitong Yang, Yong Zhao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00134](https://doi.org/10.1109/CVPR52729.2023.00134)

        **Abstract**:

        In the field of binocular stereo matching, remarkable progress has been made by iterative methods like RAFT-Stereo and CREStereo. However, most of these methods lose information during the iterative process, making it difficult to generate more detailed difference maps that take full advantage of high-frequency information. We propose the Decouple module to alleviate the problem of data coupling and allow features containing subtle details to transfer across the iterations which proves to alleviate the problem significantly in the ablations. To further capture high-frequency details, we propose a Normalization Refinement module that unifies the disparities as a proportion of the disparities over the width of the image, which address the problem of module failure in cross-domain scenarios. Further, with the above improvements, the ResNet-like feature extractor that has not been changed for years becomes a bottleneck. Towards this end, we proposed a multi-scale and multi-stage feature extractor that introduces the channel-wise self-attention mechanism which greatly addresses this bottleneck. Our method (DLNR) ranks 1st on the Middlebury leaderboard, significantly outperforming the next best method by 13.04%. Our method also achieves SOTA performance on the KITTI-2015 benchmark for D1-fg. Code and demos are available at: https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network.

        ----

        ## [129] Rethinking Optical Flow from Geometric Matching Consistent Perspective

        **Authors**: *Qiaole Dong, Chenjie Cao, Yanwei Fu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00135](https://doi.org/10.1109/CVPR52729.2023.00135)

        **Abstract**:

        Optical flow estimation is a challenging problem remaining unsolved. Recent deep learning based optical flow models have achieved considerable success. However, these models often train networks from the scratch on standard optical flow data, which restricts their ability to robustly and geometrically match image features. In this paper, we propose a rethinking to previous optical flow estimation. We particularly leverage Geometric Image Matching (GIM) as a pre-training task for the optical flow estimation (MatchFlow) with better feature representations, as GIM shares some common challenges as optical flow estimation, and with massive labeled real-world data. Thus, matching static scenes helps to learn more fundamental feature correlations of objects and scenes with consistent displacements. Specifically, the proposed MatchFlow model employs a QuadTree attention-based network pre-trained on MegaDepth to extract coarse features for further flow regression. Extensive experiments show that our model has great cross-dataset generalization. Our method achieves 11.5% and 10.1% error reduction from GMA on Sintel clean pass and KITTI test set. At the time of anonymous submission, our MatchFlow(G) enjoys state-of-the-art performance on Sintel clean and final pass compared to published approaches with comparable computation and memory footprint. Codes and models will be released in https://github.com/DQiaole/MatchFlow.

        ----

        ## [130] Efficient Robust Principal Component Analysis via Block Krylov Iteration and CUR Decomposition

        **Authors**: *Shun Fang, Zhengqin Xu, Shiqian Wu, Shoulie Xie*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00136](https://doi.org/10.1109/CVPR52729.2023.00136)

        **Abstract**:

        Robust principal component analysis (RPCA) is widely studied in computer vision. Recently an adaptive rank estimate based RPCA has achieved top performance in low-level vision tasks without the prior rank, but both the rank estimate and RPCA optimization algorithm involve singular value decomposition, which requires extremely huge computational resource for large-scale matrices. To address these issues, an efficient RPCA (eRPCA) algorithm is proposed based on block Krylov iteration and CUR decomposition in this paper. Specifically, the Krylov iteration method is employed to approximate the eigenvalue decomposition in the rank estimation, which requires 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(ndrq+n(rq)^{2})$</tex>
 for an 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(n\times d)$</tex>
 input matrix, in which 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$q$</tex>
 is a parameter with a small value, 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$r$</tex>
 is the target rank. Based on the estimated rank, CUR decomposition is adopted to replace SVD in updating low-rank matrix component, whose complexity reduces from 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(rnd)$</tex>
 to 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(r^{2}n)$</tex>
 per iteration. Experimental results verify the efficiency and effectiveness of the proposed eRPCA over the state-of-the-art methods in various low-level vision applications.

        ----

        ## [131] VectorFloorSeg: Two-Stream Graph Attention Network for Vectorized Roughcast Floorplan Segmentation

        **Authors**: *Bingchen Yang, Haiyong Jiang, Hao Pan, Jun Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00137](https://doi.org/10.1109/CVPR52729.2023.00137)

        **Abstract**:

        Vector graphics (VG) are ubiquitous in industrial designs. In this paper, we address semantic segmentation of a typical VG, i.e., roughcast floorplans with bare wall structures, whose output can be directly used for further applications like interior furnishing and room space modeling. Previous semantic segmentation works mostly process well-decorated floorplans in raster images and usually yield aliased boundaries and outlier fragments in segmented rooms, due to pixel-level segmentation that ignores the regular elements (e.g. line segments) in vector floor-plans. To overcome these issues, we propose to fully utilize the regular elements in vector floorplans for more integral segmentation. Our pipeline predicts room segmentation from vector floorplans by dually classifying line segments as room boundaries, and regions partitioned by line segments as room segments. To fully exploit the structural relationships between lines and regions, we use two-stream graph neural networks to process the line segments and partitioned regions respectively, and devise a novel modulated graph attention layer to fuse the heterogeneous information from one stream to the other. Extensive experiments show that by directly operating on vector floorplans, we outper-form image-based methods in both mIoU and mAcc. In addition, we propose a new metric that captures room integrity and boundary regularity, which confirms that our method produces much more regular segmentations. Source code is available at https://github.com/DrZiji/VecFloorSeg.

        ----

        ## [132] TBP-Former: Learning Temporal Bird's-Eye-View Pyramid for Joint Perception and Prediction in Vision-Centric Autonomous Driving

        **Authors**: *Shaoheng Fang, Zi Wang, Yiqi Zhong, Junhao Ge, Siheng Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00138](https://doi.org/10.1109/CVPR52729.2023.00138)

        **Abstract**:

        Vision-centric joint perception and prediction (PnP) has become an emerging trend in autonomous driving research. It predicts the future states of the traffic participants in the surrounding environment from raw RGB images. However, it is still a critical challenge to synchronize features obtained at multiple camera views and timestamps due to inevitable geometric distortions and further exploit those spatial-temporal features. To address this issue, we propose a temporal bird's-eye-view pyramid transformer (TBP-Former) for vision-centric PnP; which includes two novel designs. First, a pose-synchronized BEV encoder is proposed to map raw image inputs with any camera pose at any time to a shared and synchronized BEV space for better spatial-temporal synchronization. Second, a spatial-temporal pyramid transformer is introduced to comprehensively extract multi-scale BEV features and predict future BEV states with the support of spatial priors. Extensive experiments on nuScenes dataset show that our proposed framework overall outperforms all state-of-the-art vision-based prediction methods. Code is available at: https://github.com/MediaBrain-SJTU/TBP-Former

        ----

        ## [133] Implicit Occupancy Flow Fields for Perception and Prediction in Self-Driving

        **Authors**: *Ben Agro, Quinlan Sykora, Sergio Casas, Raquel Urtasun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00139](https://doi.org/10.1109/CVPR52729.2023.00139)

        **Abstract**:

        A self-driving vehicle (SDV) must be able to perceive its surroundings and predict the future behavior of other traffic participants. Existing works either perform object detection followed by trajectory forecasting of the detected objects, or predict dense occupancy and flow grids for the whole scene. The former poses a safety concern as the number of detections needs to be kept low for efficiency reasons, sacrificing object recall. The latter is computationally expensive due to the high-dimensionality of the output grid, and suffers from the limited receptive field inherent to fully convolutional networks. Furthermore, both approaches employ many computational resources predicting areas or objects that might never be queried by the motion planner. This motivates our unified approach to perception and future prediction that implicitly represents occupancy and flow over time with a single neural network. Our method avoids unnecessary computation, as it can be directly queried by the motion planner at continuous spatio-temporal locations. Moreover, we design an architecture that overcomes the limited receptive field of previous explicit occupancy prediction methods by adding an efficient yet effective global attention mechanism. Through extensive experiments in both urban and highway settings, we demonstrate that our implicit model outperforms the current state-of-the-art. For more information, visit the project website: https://waabi.ai/research/implicito.

        ----

        ## [134] UniSim: A Neural Closed-Loop Sensor Simulator

        **Authors**: *Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, Raquel Urtasun*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00140](https://doi.org/10.1109/CVPR52729.2023.00140)

        **Abstract**:

        Rigorously testing autonomy systems is essential for making safe self-driving vehicles (SDV) a reality. It requires one to generate safety critical scenarios beyond what can be collected safely in the world, as many scenarios happen rarely on our roads. To accurately evaluate performance, we need to test the SDV on these scenarios in closed-loop, where the SDV and other actors interact with each other at each timestep. Previously recorded driving logs provide a rich resource to build these new scenarios from, but for closed loop evaluation, we need to modify the sensor data based on the new scene configuration and the SDV's decisions, as actors might be added or removed and the trajectories of existing actors and the SDV will differ from the original log. In this paper, we present UniSim, a neural sensor simulator that takes a single recorded log captured by a sensor-equipped vehicle and converts it into a realistic closed-loop multi-sensor simulation. UniSim builds neural feature grids to reconstruct both the static background and dynamic actors in the scene, and composites them together to simulate LiDAR and camera data at new viewpoints, with actors added or removed and at new placements. To better handle extrapolated views, we incorporate learnable priors for dynamic objects, and leverage a convolutional network to complete unseen regions. Our experiments show UniSim can simulate realistic sensor data with small domain gap on downstream tasks. With UniSim, we demonstrate, for the first time, closed-loop evaluation of an autonomy system on safety-critical scenarios as if it were in the real world.

        ----

        ## [135] FEND: A Future Enhanced Distribution-Aware Contrastive Learning Framework for Long-Tail Trajectory Prediction

        **Authors**: *Yuning Wang, Pu Zhang, Lei Bai, Jianru Xue*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00141](https://doi.org/10.1109/CVPR52729.2023.00141)

        **Abstract**:

        Predicting the future trajectories of the traffic agents is a gordian technique in autonomous driving. However, trajectory prediction suffers from data imbalance in the prevalent datasets, and the tailed data is often more complicated and safety-critical. In this paper, we focus on dealing with the long-tail phenomenon in trajectory prediction. Previous methods dealing with long-tail data did not take into account the variety of motion patterns in the tailed data. In this paper, we put forward a future enhanced contrastive learning framework to recognize tail trajectory patterns and form a feature space with separate pattern clusters. Furthermore, a distribution aware hyper predictor is brought up to better utilize the shaped feature space. Our method is a model-agnostic framework and can be plugged into many well-known baselines. Experimental results show that our framework outperforms the state-of-the-art long-tail prediction method on tailed samples by 9.5% on ADE and 8.5% on FDE, while maintaining or slightly improving the averaged performance. Our method also surpasses many long-tail techniques on trajectory prediction task.

        ----

        ## [136] EqMotion: Equivariant Multi-Agent Motion Prediction with Invariant Interaction Reasoning

        **Authors**: *Chenxin Xu, Robby T. Tan, Yuhong Tan, Siheng Chen, Yu Guang Wang, Xinchao Wang, Yanfeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00142](https://doi.org/10.1109/CVPR52729.2023.00142)

        **Abstract**:

        Learning to predict agent motions with relationship reasoning is important for many applications. In motion prediction tasks, maintaining motion equivariance under Euclidean geometric transformations and invariance of agent interaction is a critical and fundamental principle. However, such equivariance and invariance properties are overlooked by most existing methods. To fill this gap, we propose Eq-Motion, an efficient equivariant motion prediction model with invariant interaction reasoning. To achieve motion equivariance, we propose an equivariant geometric feature learning module to learn a Euclidean transformable feature through dedicated designs of equivariant operations. To reason agent's interactions, we propose an invariant interaction reasoning module to achieve a more stable interaction modeling. To further promote more comprehensive motion features, we propose an invariant pattern feature learning module to learn an invariant pattern feature, which cooperates with the equivariant geometric feature to enhance network expressiveness. We conduct experiments for the proposed model on four distinct scenarios: particle dynamics, molecule dynamics, human skeleton motion prediction and pedestrian trajectory prediction. Experimental results show that our method is not only generally applicable, but also achieves state-of-the-art prediction performances on all the four tasks, improving by 24.0/30.1/8.6/9.2%. Code is available at https://github.com/MediaBrain-SJTU/EqMotion.

        ----

        ## [137] Lookahead Diffusion Probabilistic Models for Refining Mean Estimation

        **Authors**: *Guoqiang Zhang, Kenta Niwa, W. Bastiaan Kleijn*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00143](https://doi.org/10.1109/CVPR52729.2023.00143)

        **Abstract**:

        We propose lookahead diffusion probabilistic models (LA-DPMs) to exploit the correlation in the outputs of the deep neural networks (DNNs) over subsequent timesteps in diffusion probabilistic models (DPMs) to refine the mean estimation of the conditional Gaussian distributions in the backward process. A typical DPM first obtains an estimate of the original data sample 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$x$</tex>
 by feeding the most recent state 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$z_{i}$</tex>
 and index 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$i$</tex>
 into the DNN model and then computes the mean vector of the conditional Gaussian distribution for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$z_{i-1}$</tex>
. We propose to calculate a more accurate estimate for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$x$</tex>
 by performing extrapolation on the two estimates of 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$x$</tex>
 that are obtained by feeding (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$z_{i+1},i+1$</tex>
) and (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$z_{i},i$</tex>
) into the DNN model. The extrapolation can be easily integrated into the backward process of existing DPMs by introducing an additional connection over two consecutive timesteps, and fine-tuning is not required. Extensive experiments showed that plugging in the additional connection into DDPM, DDIM, DEIS, S-PNDM, and high-order DPM-Solvers leads to a significant performance gain in terms of Fréchet inception distance (FID) score. Our implementation is available at https://github.com/guoqiang-zhang-x/LA-DPM.

        ----

        ## [138] Neural Volumetric Memory for Visual Locomotion Control

        **Authors**: *Ruihan Yang, Ge Yang, Xiaolong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00144](https://doi.org/10.1109/CVPR52729.2023.00144)

        **Abstract**:

        Legged robots have the potential to expand the reach of autonomy beyond paved roads. In this work, we consider the difficult problem of locomotion on challenging terrains using a single forward-facing depth camera. Due to the partial observability of the problem, the robot has to rely on past observations to infer the terrain currently beneath it. To solve this problem, we follow the paradigm in computer vision that explicitly models the 3D geometry of the scene and propose Neural Volumetric Memory (NVM), a geometric memory architecture that explicitly accounts for the SE(3) equivariance of the 3D world. NVM aggregates feature volumes from multiple camera views by first bringing them back to the ego-centric frame of the robot. We test the learned visual-locomotion policy on a physical robot and show that our approach, which explicitly introduces geometric priors during training, offers superior performance than more naïve methods. We also include ablation studies and show that the representations stored in the neural volumetric memory capture sufficient geometric information to reconstruct the scene. Our project page with videos is https://rchalyang.github.io/NVM

        ----

        ## [139] Gazeformer: Scalable, Effective and Fast Prediction of Goal-Directed Human Attention

        **Authors**: *Sounak Mondal, Zhibo Yang, Seoyoung Ahn, Dimitris Samaras, Gregory J. Zelinsky, Minh Hoai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00145](https://doi.org/10.1109/CVPR52729.2023.00145)

        **Abstract**:

        Predicting human gaze is important in Human-Computer Interaction (HCI). However, to practically serve HCI applications, gaze prediction models must be scalable, fast, and accurate in their spatial and temporal gaze predictions. Recent scanpath prediction models focus on goaldirected attention (search). Such models are limited in their application due to a common approach relying on trained target detectors for all possible objects, and the availability of human gaze data for their training (both not scalable). In response, we pose a new task called ZeroGaze, a new variant of zero-shot learning where gaze is predicted for never-before-searched objects, and we develop a novel model, Gazeformer; to solve the ZeroGaze problem. In contrast to existing methods using object detector modules, Gazeformer encodes the target using a natural language model, thus leveraging semantic similarities in scanpath prediction. We use a transformer-based encoder-decoder architecture because transformers are particularly useful for generating contextual representations. Gazeformer surpasses other models by a large margin (19%-70%) on the ZeroGaze setting. It also outperforms existing target-detection models on standard gaze prediction for both target-present and target-absent search tasks. In addition to its improved performance, Gazeformer is more than five times faster than the state-of-the-art target-present visual search model. Code can be found at https://github.com/cvlab-stonybrook/Gazeformer/

        ----

        ## [140] DrapeNet: Garment Generation and Self-Supervised Draping

        **Authors**: *Luca De Luigi, Ren Li, Benoît Guillard, Mathieu Salzmann, Pascal Fua*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00146](https://doi.org/10.1109/CVPR52729.2023.00146)

        **Abstract**:

        Recent approaches to drape garments quickly over arbitrary human bodies leverage self-supervision to eliminate the need for large training sets. However, they are designed to train one network per clothing item, which severely limits their generalization abilities. In our work, we rely on self-supervision to train a single network to drape multiple garments. This is achieved by predicting a 3D deformation field conditioned on the latent codes of a generative network, which models garments as unsigned distance fields. Our pipeline can generate and drape previously unseen garments of any topology, whose shape can be edited by manipulating their latent codes. Being fully differentiable, our formulation makes it possible to recover accurate 3D models of garments from partial observations - images or 3D scans - via gradient descent. Our code is publicly available at https://github.com/liren2515/DrapeNet

        ----

        ## [141] Tracking Multiple Deformable Objects in Egocentric Videos

        **Authors**: *Mingzhen Huang, Xiaoxing Li, Jun Hu, Honghong Peng, Siwei Lyu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00147](https://doi.org/10.1109/CVPR52729.2023.00147)

        **Abstract**:

        Most existing multiple object tracking (MOT) methods that solely rely on appearance features struggle in tracking highly deformable objects. Other MOT methods that use motion clues to associate identities across frames have difficulty handling egocentric videos effectively or efficiently. In this work, we present DogThruGlasses, a large-scale deformable multi-object tracking dataset, with 150 videos and 73K annotated frames, which is collected exclusively by smart glasses. We also propose DETracker, a new MOT method that jointly detects and tracks deformable objects in egocentric videos. DETracker uses three novel modules, namely the motion disentanglement network (MDN), the patch association network (PAN) and the patch memory network (PMN), to explicitly tackle severe ego motion and track fast morphing target objects. DETracker is end-to-end trainable and achieves near real-time speed, which outperforms existing state-of-the-art method on DogThruGlasses and YouTube-Hand.

        ----

        ## [142] Good is Bad: Causality Inspired Cloth-debiasing for Cloth-changing Person Re-identification

        **Authors**: *Zhengwei Yang, Meng Lin, Xian Zhong, Yu Wu, Zheng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00148](https://doi.org/10.1109/CVPR52729.2023.00148)

        **Abstract**:

        Entangled representation of clothing and identity (ID)-intrinsic clues are potentially concomitant in conventional person Re- IDentification (ReID). Nevertheless, eliminating the negative impact of clothing on ID remains challenging due to the lack of theory and the difficulty of isolating the exact implications. In this paper, a causality-based Auto-Intervention Model, referred to as AIM
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Codes will publicly available at https://github.com/BoomShakaY/AIM-CCReID., is first proposed to mitigate clothing bias for robust cloth-changing person ReID (CC-ReID). Specifically, we analyze the effect of clothing on the model inference and adopt a dual-branch model to simulate causal intervention. Progressively, clothing bias is eliminated automatically with model training. AIM is encouraged to learn more discriminative ID clues that are free from clothing bias. Extensive experiments on two standard CC-ReID datasets demonstrate the superiority of the proposed AIM over other state-of-the-art methods.

        ----

        ## [143] Micron-BERT: BERT-Based Facial Micro-Expression Recognition

        **Authors**: *Xuan-Bac Nguyen, Chi Nhan Duong, Xin Li, Susan Gauch, Han-Seok Seo, Khoa Luu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00149](https://doi.org/10.1109/CVPR52729.2023.00149)

        **Abstract**:

        Micro-expression recognition is one of the most challenging topics in affective computing. It aims to recognize tiny facial movements difficult for humans to perceive in a brief period, i.e., 0.25 to 0.5 seconds. Recent advances in pre-training deep Bidirectional Transformers (BERT) have significantly improved self-supervised learning tasks in computer vision. However, the standard BERT in vision problems is designed to learn only from full images or videos, and the architecture cannot accurately detect details of facial micro-expressions. This paper presents Micron-BERT (
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$(\mu$</tex>
-BERT), a novel approach to facial micro-expression recognition. The proposed method can automatically capture these movements in an unsupervised manner based on two key ideas. First, we employ Diagonal Micro-Attention (DMA) to detect tiny differences between two frames. Second, we introduce a new Patch of Interest (PoI) module to localize and highlight micro-expression interest regions and simultaneously reduce noisy backgrounds and distractions. By incorporating these components into an end-to-end deep network, the proposed 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mu$</tex>
-BERT significantly outperforms all previous work in various micro-expression tasks. 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mu$</tex>
-BERT can be trained on a large-scale unlabeled dataset, i.e., up to 8 million images, and achieves high accuracy on new unseen facial micro-expression datasets. Empirical experiments show 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$\mu$</tex>
-BERT consistently outperforms state-of-the-art performance on four micro-expression benchmarks, including SAMM, CASME II, SMIC, and CASME3, by significant margins. Code will be available at https://github.com/uark-cviu/Micron-BERT

        ----

        ## [144] MARLIN: Masked Autoencoder for facial video Representation LearnINg

        **Authors**: *Zhixi Cai, Shreya Ghosh, Kalin Stefanov, Abhinav Dhall, Jianfei Cai, Hamid Rezatofighi, Reza Haffari, Munawar Hayat*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00150](https://doi.org/10.1109/CVPR52729.2023.00150)

        **Abstract**:

        This paper proposes a self-supervised approach to learn universal facial representations from videos, that can transfer across a variety of facial analysis tasks such as Facial Attribute Recognition (FAR), Facial Expression Recognition (FER), DeepFake Detection (DFD), and Lip Synchronization (LS). Our proposed framework, named MARLIN, is a facial video masked autoencoder, that learns highly robust and generic facial embeddings from abundantly available non-annotated web crawled facial videos. As a challenging auxiliary task, MARLIN reconstructs the spatio-temporal details of the face from the densely masked facial regions which mainly include eyes, nose, mouth, lips, and skin to capture local and global aspects that in turn help in encoding generic and transferable features. Through a variety of experiments on diverse downstream tasks, we demonstrate MARLIN to be an excellent facial video encoder as well as feature extractor, that performs consistently well across a variety of downstream tasks including FAR (1.13% gain over supervised benchmark), FER (2.64% gain over unsupervised benchmark), DFD (1.86% gain over unsupervised benchmark), LS (29.36% gain for Frechet Inception Distance), and even in low data regime. Our code and models are available at https://github.com/ControlNet/MARLIN.

        ----

        ## [145] StyleSync: High-Fidelity Generalized and Personalized Lip Sync in Style-Based Generator

        **Authors**: *Jiazhi Guan, Zhanwang Zhang, Hang Zhou, Tianshu Hu, Kaisiyuan Wang, Dongliang He, Haocheng Feng, Jingtuo Liu, Errui Ding, Ziwei Liu, Jingdong Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00151](https://doi.org/10.1109/CVPR52729.2023.00151)

        **Abstract**:

        Despite recent advances in syncing lip movements with any audio waves, current methods still struggle to balance generation quality and the model's generalization ability. Previous studies either require long-term data for training or produce a similar movement pattern on all subjects with low quality. In this paper, we propose StyleSync, an effective framework that enables high-fidelity lip synchronization. We identify that a style-based generator would sufficiently enable such a charming property on both one-shot and few-shot scenarios. Specifically, we design a mask-guided spatial information encoding module that preserves the details of the given face. The mouth shapes are accurately modified by audio through modulated convolutions. Moreover, our design also enables personalized lip-sync by introducing style space and generator refinement on only limited frames. Thus the identity and talking style of a target person could be accurately preserved. Extensive experiments demonstrate the effectiveness of our method in producing high-fidelity results on a variety of scenes. Resources can be found at https:/hangz-nju-cuhk.github.io/projects/StyleSync.

        ----

        ## [146] REALIMPACT: A Dataset of Impact Sound Fields for Real Objects

        **Authors**: *Samuel Clarke, Ruohan Gao, Mason Wang, Mark Rau, Julia Xu, Jui-Hsien Wang, Doug L. James, Jiajun Wu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00152](https://doi.org/10.1109/CVPR52729.2023.00152)

        **Abstract**:

        Objects make unique sounds under different perturbations, environment conditions, and poses relative to the listener. While prior works have modeled impact sounds and sound propagation in simulation, we lack a standard dataset of impact sound fields of real objects for audio-visual learning and calibration of the sim-to-real gap. We present Realimpact,a large-scale dataset of real object impact sounds recorded under controlled conditions. Re-alimpactcontains 150,000 recordings of impact sounds of 50 everyday objects with detailed annotations, including their impact locations, microphone locations, contact force profiles, material labels, and RGBD images.**The project page and dataset are available at https://samuelpclarke.com/realimpact/ We make preliminary attempts to use our dataset as a reference to current simulation methods for estimating object impact sounds that match the real world. Moreover, we demon-strate the usefulness of our dataset as a testbed for acoustic and audio-visual learning via the evaluation of two bench-mark tasks, including listener location classification and vi-sual acoustic matching.

        ----

        ## [147] STMT: A Spatial-Temporal Mesh Transformer for MoCap-Based Action Recognition

        **Authors**: *Xiaoyu Zhu, Po-Yao Huang, Junwei Liang, Celso M. de Melo, Alexander G. Hauptmann*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00153](https://doi.org/10.1109/CVPR52729.2023.00153)

        **Abstract**:

        We study the problem of human action recognition using motion capture (MoCap) sequences. Unlike existing techniques that take multiple manual steps to derive standard-ized skeleton representations as model input, we propose a novel Spatial-Temporal Mesh Transformer (STMT) to directly model the mesh sequences. The model uses a hierarchical transformer with intra-frame off-set attention and inter-frame self-attention. The attention mechanism allows the model to freely attend between any two vertex patches to learn nonlocal relationships in the spatial-temporal domain. Masked vertex modeling and future frame prediction are used as two self-supervised tasks to fully activate the bi-directional and auto-regressive attention in our hierarchical transformer. The proposed method achieves state-of-the-art performance compared to skeleton-based and point-cloud-based models on common MoCap benchmarks. Code is available at https://github.com/zgzxy001/STMT.

        ----

        ## [148] Progressive Spatio-temporal Alignment for Efficient Event-based Motion Estimation

        **Authors**: *Xueyan Huang, Yueyi Zhang, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00154](https://doi.org/10.1109/CVPR52729.2023.00154)

        **Abstract**:

        In this paper, we propose an efficient event-based motion estimation framework for various motion models. Different from previous works, we design a progressive event-to-map alignment scheme and utilize the spatio-temporal correlations to align events. In detail, we progressively align sampled events in an event batch to the time-surface map and obtain the updated motion model by minimizing a novel time-surface loss. In addition, a dynamic batch size strategy is applied to adaptively adjust the batch size so that all events in the batch are consistent with the current motion model. Our framework has three advantages: a) the progressive scheme refines motion parameters iteratively, achieving accurate motion estimation; b) within one iteration, only a small portion of events are involved in optimization, which greatly reduces the total runtime; c) the dynamic batch size strategy ensures that the constant velocity assumption always holds. We conduct comprehensive experiments to evaluate our framework on challenging high-speed scenes with three motion models: rotational, homography, and 6-DOF models. Experimental results demonstrate that our framework achieves state-of-the-art estimation accuracy and efficiency. The code is available at https://github.com/huangxueyan/PEME.

        ----

        ## [149] Event-Based Shape from Polarization

        **Authors**: *Manasi Muglikar, Leonard Bauersfeld, Diederik Paul Moeys, Davide Scaramuzza*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00155](https://doi.org/10.1109/CVPR52729.2023.00155)

        **Abstract**:

        State-of-the-art solutions for Shape-from-Polarization (SfP) suffer from a speed-resolution tradeoff: they either sacrifice the number of polarization angles measured or necessitate lengthy acquisition times due to framerate constraints, thus compromising either accuracy or latency. We tackle this tradeoff using event cameras. Event cameras operate at microseconds resolution with negligible motion blur, and output a continuous stream of events that precisely measures how light changes over time asynchronously. We propose a setup that consists of a linear polarizer rotating at high speeds in front of an event camera. Our method uses the continuous event stream caused by the rotation to reconstruct relative intensities at multiple polarizer angles. Experiments demonstrate that our method outperforms physics-based baselines using frames, reducing the MAE by 25% in synthetic and real-world datasets. In the real world, we observe, however, that the challenging conditions (i.e., when few events are generated) harm the performance of physics-based solutions. To overcome this, we propose a learning-based approach that learns to estimate surface normals even at low event-rates, improving the physics-based approach by 52% on the real world dataset. The proposed system achieves an acquisition speed equivalent to 50 fps (>twice the framerate of the commercial polarization sensor) while retaining the spatial resolution of 1 MP. Our evaluation is based on the first large-scale dataset for event-based SfP. Code dataset and video are available under: https://rpg.ifi.uzh.ch/esfp.html https://youtu.be/sF3Ue2Zkpec

        ----

        ## [150] Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution

        **Authors**: *Yunfan Lu, Zipeng Wang, Minjie Liu, Hongjian Wang, Lin Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00156](https://doi.org/10.1109/CVPR52729.2023.00156)

        **Abstract**:

        Event cameras sense the intensity changes asynchronously and produce event streams with high dynamic range and low latency. This has inspired research endeavors utilizing events to guide the challenging video super-resolution (VSR) task. In this paper, we make the first attempt to address a novel problem of achieving VSR at random scales by taking advantages of the high temporal resolution property of events. This is hampered by the difficulties of representing the spatial-temporal information of events when guiding VSR. To this end, we propose a novel framework that incorporates the spatial-temporal interpolation of events to VSR in a unified framework. Our key idea is to learn implicit neural representations from queried spatial-temporal coordinates and features from both RGB frames and events. Our method contains three parts. Specifically, the Spatial-Temporal Fusion (STF) module first learns the 3D features from events and RGB frames. Then, the Temporal Filter (TF) module unlocks more explicit motion information from the events near the queried timestamp and generates the 2D features. Lastly, the Spatial-Temporal Implicit Representation (STIR) module recovers the SR frame in arbitrary resolutions from the outputs of these two modules. In addition, we collect a real-world dataset with spatially aligned events and RGB frames. Extensive experiments show that our method significantly surpasses the prior-arts and achieves VSR with random scales, e.g., 6.5. Code and dataset are available at https://vlis2022.github.io/cvpr23/egvsr.

        ----

        ## [151] BiFormer: Learning Bilateral Motion Estimation via Bilateral Transformer for 4K Video Frame Interpolation

        **Authors**: *Junheum Park, Jintae Kim, Chang-Su Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00157](https://doi.org/10.1109/CVPR52729.2023.00157)

        **Abstract**:

        A novel 4K video frame interpolator based on bilateral transformer (BiFormer) is proposed in this paper, which performs three steps: global motion estimation, local motion refinement, and frame synthesis. First, in global motion estimation, we predict symmetric bilateral motion fields at a coarse scale. To this end, we propose BiFormer, the first transformer-based bilateral motion estimator. Second, we refine the global motion fields efficiently using blockwise bilateral cost volumes (BBCVs). Third, we warp the input frames using the refined motion fields and blend them to synthesize an intermediate frame. Extensive experiments demonstrate that the proposed BiFormer algorithm achieves excellent interpolation performance on 4K datasets. The source codes are available at https://github.com/JunHeum/BiFormer.

        ----

        ## [152] A Unified Pyramid Recurrent Network for Video Frame Interpolation

        **Authors**: *Xin Jin, Longhai Wu, Jie Chen, Youxin Chen, Jayoon Koo, Cheul-Hee Hahm*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00158](https://doi.org/10.1109/CVPR52729.2023.00158)

        **Abstract**:

        Flow-guided synthesis provides a common framework for frame interpolation, where optical flow is estimated to guide the synthesis of intermediate frames between consecutive inputs. In this paper, we present UPR-Net, a novel Unified Pyramid Recurrent Network for frame interpolation. Cast in a flexible pyramid framework, UPR-Net exploits lightweight recurrent modules for both bi-directional flow estimation and intermediate frame synthesis. At each pyramid level, it leverages estimated bi-directional flow to generate forward-warped representations for frame synthesis; across pyramid levels, it enables iterative refinement for both optical flow and intermediate frame. In particular, we show that our iterative synthesis strategy can significantly improve the robustness of frame interpolation on large motion cases. Despite being extremely lightweight (1.7M parameters), our base version of UPR-Net achieves excellent performance on a large range of benchmarks. Code and trained models of our UPR-Net series are available at: https://github.com/srcn-iv1/UPR-Net.

        ----

        ## [153] Event-based Blurry Frame Interpolation under Blind Exposure

        **Authors**: *Wenming Weng, Yueyi Zhang, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00159](https://doi.org/10.1109/CVPR52729.2023.00159)

        **Abstract**:

        Restoring sharp high frame-rate videos from low frame-rate blurry videos is a challenging problem. Existing blurry frame interpolation methods assume a predefined and known exposure time, which suffer from severe performance drop when applied to videos captured in the wild. In this paper, we study the problem of blurry frame interpolation under blind exposure with the assistance of an event camera. The high temporal resolution of the event camera is beneficial to obtain the exposure prior that is lost during the imaging process. Besides, sharp frames can be restored using event streams and blurry frames relying on the mutual constraint among them. Therefore, we first propose an exposure estimation strategy guided by event streams to estimate the lost exposure prior, transforming the blind exposure problem well-posed. Second, we propose to model the mutual constraint with a temporal-exposure control strategy through iterative residual learning. Our blurry frame interpolation method achieves a distinct performance boost over existing methods on both synthetic and self-collected real- world datasets under blind exposure.

        ----

        ## [154] FlowFormer++: Masked Cost Volume Autoencoding for Pretraining Optical Flow Estimation

        **Authors**: *Xiaoyu Shi, Zhaoyang Huang, Dasong Li, Manyuan Zhang, Ka Chun Cheung, Simon See, Hongwei Qin, Jifeng Dai, Hongsheng Li*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00160](https://doi.org/10.1109/CVPR52729.2023.00160)

        **Abstract**:

        FlowFormer [24] introduces a transformer architecture into optical flow estimation and achieves state-of-the-art performance. The core component of FlowFormer is the transformer-based cost-volume encoder. Inspired by the recent success of masked autoencoding (MAE) pretraining in unleashing transformers' capacity of encoding visual representation, we propose Masked Cost Volume Autoencoding (MCVA) to enhance FlowFormer by pretraining the cost-volume encoder with a novel MAE scheme. Firstly, we introduce a block-sharing masking strategy to prevent masked information leakage, as the cost maps of neighboring source pixels are highly correlated. Secondly, we propose a novel pre-text reconstruction task, which encourages the cost-volume encoder to aggregate long-range information and ensures pretraining-finetuning consistency. We also show how to modify the FlowFormer architecture to accommodate masks during pretraining. Pretrained with MCVA, FlowFormer++ ranks 1st among published methods on both Sintel and KITTI-2015 benchmarks. Specifically, FlowFormer++ achieves 1.07 and 1.94 average end-point error (AEPE) on the clean and final pass of Sintel benchmark, leading to 7.76% and 7.18% error reductions from FlowFormer. FlowFormer++ obtains 4.52 F1-all on the KITTI-2015 test set, improving FlowFormer by 0.16.

        ----

        ## [155] POTTER: Pooling Attention Transformer for Efficient Human Mesh Recovery

        **Authors**: *Ce Zheng, Xianpeng Liu, Guo-Jun Qi, Chen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00161](https://doi.org/10.1109/CVPR52729.2023.00161)

        **Abstract**:

        Transformer architectures have achieved SOTA performance on the human mesh recovery (HMR) from monocular images. However, the performance gain has come at the cost of substantial memory and computational overhead. A lightweight and efficient model to reconstruct accurate human mesh is needed for real-world applications. In this paper, we propose a pure transformer architecture named POoling aTtention TransformER (POTTER) for the HMR task from single images. Observing that the conventional attention module is memory and computationally expensive, we propose an efficient pooling attention module, which significantly reduces the memory and computational cost without sacrificing performance. Furthermore, we design a new transformer architecture by integrating a High-Resolution (HR) stream for the HMR task. The high-resolution local and global features from the HR stream can be utilized for recovering more accurate human mesh. Our POTTER outperforms the SOTA method METRO by only requiring 7% of total parameters and 14% of the Multiply-Accumulate Operations on the Human3.6M (PA-MPJPE metric) and 3DPW (all three metrics) datasets. The project webpage is https://zczcwh.github.io/potter_page/.

        ----

        ## [156] Adaptive Patch Deformation for Textureless-Resilient Multi-View Stereo

        **Authors**: *Yuesong Wang, Zhaojie Zeng, Tao Guan, Wei Yang, Zhuo Chen, Wenkai Liu, Luoyuan Xu, Yawei Luo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00162](https://doi.org/10.1109/CVPR52729.2023.00162)

        **Abstract**:

        In recent years, deep learning-based approaches have shown great strength in multi-view stereo because of their outstanding ability to extract robust visual features. However, most learning-based methods need to build the cost volume and increase the receptive field enormously to get a satisfactory result when dealing with large-scale textureless regions, consequently leading to prohibitive memory consumption. To ensure both memory-friendly and textureless-resilient, we innovatively transplant the spirit of deformable convolution from deep learning into the traditional PatchMatch-based method. Specifically, for each pixel with matching ambiguity (termed unreliable pixel), we adaptively deform the patch centered on it to extend the receptive field until covering enough correlative reliable pixels (without matching ambiguity) that serve as anchors. When performing PatchMatch, constrained by the anchor pixels, the matching cost of an unreliable pixel is guaranteed to reach the global minimum at the correct depth and therefore increases the robustness of multi-view stereo significantly. To detect more anchor pixels to ensure better adaptive patch deformation, we propose to evaluate the matching ambiguity of a certain pixel by checking the convergence of the estimated depth as optimization proceeds. As a result, our method achieves state-of-the-art performance on ETH3D and Tanks and Temples while preserving low memory consumption.

        ----

        ## [157] On the Difficulty of Unpaired Infrared-to-Visible Video Translation: Fine-Grained Content-Rich Patches Transfer

        **Authors**: *Zhenjie Yu, Shuang Li, Yirui Shen, Chi Harold Liu, Shuigen Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00163](https://doi.org/10.1109/CVPR52729.2023.00163)

        **Abstract**:

        Explicit visible videos can provide sufficient visual information and facilitate vision applications. Unfortunately, the image sensors of visible cameras are sensitive to light conditions like darkness or overexposure. To make up for this, recently, infrared sensors capable of stable imaging have received increasing attention in autonomous driving and monitoring. However, most prosperous vision models are still trained on massive clear visible data, facing huge visual gaps when deploying to infrared imaging scenarios. In such cases, transferring the infrared video to a distinct visible one with fine-grained semantic patterns is a worthwhile endeavor. Previous works improve the outputs by equally optimizing each patch on the translated visible results, which is unfair for enhancing the details on content-rich patches due to the long-tail effect of pixel distribution. Here we propose a novel CPTrans framework to tackle the challenge via balancing gradients of different patches, achieving the fine-grained Content-rich Patches Transferring. Specifically, the content-aware optimization module encourages model optimization along gradients of target patches, ensuring the improvement of visual details. Additionally, the content-aware temporal normalization module enforces the generator to be robust to the motions of target patches. Moreover, we extend the existing dataset InfraredCity to more challenging adverse weather conditions (rain and snow), dubbed as InfraredCity-Adverse
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
The code and dataset are available at https://github.com/BIT-DA/12V-Processing, Extensive experiments show that the proposed CPTrans achieves state-of-the-art performance under diverse scenes while requiring less training time than competitive methods.

        ----

        ## [158] Thermal Spread Functions (TSF): Physics-Guided Material Classification

        **Authors**: *Aniket Dashpute, Vishwanath Saragadam, Emma Alexander, Florian Willomitzer, Aggelos K. Katsaggelos, Ashok Veeraraghavan, Oliver Cossairt*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00164](https://doi.org/10.1109/CVPR52729.2023.00164)

        **Abstract**:

        Robust and non-destructive material classification is a challenging but crucial first-step in numerous vision applications. We propose a physics-guided material classification framework that relies on thermal properties of the object. Our key observation is that the rate of heating and cooling of an object depends on the unique intrinsic properties of the material, namely the emissivity and diffusivity. We leverage this observation by gently heating the objects in the scene with a low-power laser for a fixed duration and then turning it off, while a thermal camera captures measurements during the heating and cooling process. We then take this spatial and temporal “thermal spread function” (TSF) to solve an inverse heat equation using the finite-differences approach, resulting in a spatially varying estimate of diffusivity and emissivity. These tuples are then used to train a classifier that produces a fine-grained material label at each spatial pixel. Our approach is extremely simple requiring only a small light source (low power laser) and a thermal camera, and produces robust classification results with 86% accuracy over 16 classes
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
Code: https://github.com/aniketdashpute/TSF.

        ----

        ## [159] Better "CMOS" Produces Clearer Images: Learning Space-Variant Blur Estimation for Blind Image Super-Resolution

        **Authors**: *Xuhai Chen, Jiangning Zhang, Chao Xu, Yabiao Wang, Chengjie Wang, Yong Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00165](https://doi.org/10.1109/CVPR52729.2023.00165)

        **Abstract**:

        Most of the existing blind image Super-Resolution (SR) methods assume that the blur kernels are space-invariant. However, the blur involved in real applications are usually space-variant due to object motion, out-of-focus, etc., resulting in severe performance drop of the advanced SR methods. To address this problem, we firstly introduce two new datasets with out-of-focus blur, i.e., NYUv2-BSR and Cityscapes-BSR, to support further researches of blind SR with space-variant blur. Based on the datasets, we design a novel Cross-MOdal fuSion network (CMOS) that estimate both blur and semantics simultaneously, which leads to improved SR results. It involves a feature Grouping Interactive Attention (GIA) module to make the two modalities in-teract more effectively and avoid inconsistency. GIA can also be used for the interaction of other features because of the universality of its structure. Qualitative and quantitative experiments compared with state-of-the-art methods on above datasets and real-world images demonstrate the superiority of our method, e.g., obtaining PSNR/SSIM by +1.91↑/+0.0048↑ on NYUv2-BSR than MANet
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/ByChelsea/CMOS.git.

        ----

        ## [160] Learning Semantic-Aware Knowledge Guidance for Low-Light Image Enhancement

        **Authors**: *Yuhui Wu, Chen Pan, Guoqing Wang, Yang Yang, Jiwei Wei, Chongyi Li, Heng Tao Shen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00166](https://doi.org/10.1109/CVPR52729.2023.00166)

        **Abstract**:

        Low-light image enhancement (LLIE) investigates how to improve illumination and produce normal-light images. The majority of existing methods improve low-light images via a global and uniform manner, without taking into account the semantic information of different regions. Without semantic priors, a network may easily deviate from a region's original color. To address this issue, we propose a novel semantic-aware knowledge-guided framework (SKF) that can assist a low-light enhancement model in learning rich and diverse priors encapsulated in a semantic segmentation model. We concentrate on incorporating semantic knowledge from three key aspects: a semantic-aware embedding module that wisely integrates semantic priors in feature representation space, a semantic-guided color histogram loss that preserves color consistency of various instances, and a semantic-guided adversarial loss that produces more natural textures by semantic priors. Our SKF is appealing in acting as a general framework in LLIE task. Extensive experiments show that models equipped with the SKF significantly outperform the baselines on multiple datasets and our SKF generalizes to different models and scenes well. The code is available at Semantic-Aware-Low-Light-Image-Enhancement.

        ----

        ## [161] CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending

        **Authors**: *Zeyu Xiao, Yutong Liu, Ruisheng Gao, Zhiwei Xiong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00167](https://doi.org/10.1109/CVPR52729.2023.00167)

        **Abstract**:

        Data augmentation (DA) is an efficient strategy for improving the performance of deep neural networks. Recent DA strategies have demonstrated utility in single image super-resolution (SR). Little research has, however, focused on the DA strategy for light field SR, in which multi-view information utilization is required. For the first time in light field SR, we propose a potent DA strategy called CutMIB to improve the performance of existing light field SR networks while keeping their structures unchanged. Specifically, Cut-MIB first cuts low-resolution (LR) patches from each view at the same location. Then CutMIB blends all LR patches to generate the blended patch and finally pastes the blended patch to the corresponding regions of high-resolution light field views, and vice versa. By doing so, CutMIB enables light field SR networks to learn from implicit geometric information during the training stage. Experimental results demonstrate that CutMIB can improve the reconstruction performance and the angular consistency of existing light field SR networks. We further verify the effectiveness of CutMIB on real-world light field SR and light field denoising. The implementation code is available at https://github.com/zeyuxiao1997/CutMIB.

        ----

        ## [162] sRGB Real Noise Synthesizing with Neighboring Correlation-Aware Noise Model

        **Authors**: *Zixuan Fu, Lanqing Guo, Bihan Wen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00168](https://doi.org/10.1109/CVPR52729.2023.00168)

        **Abstract**:

        Modeling and synthesizing real noise in the standard RGB (sRGB) domain is challenging due to the complicated noise distribution. While most of the deep noise generators proposed to synthesize sRGB real noise using an end-to-end trained model, the lack of explicit noise modeling degrades the quality of their synthesized noise. In this work, we propose to model the real noise as not only dependent on the underlying clean image pixel intensity, but also highly correlated to its neighboring noise realization within the local region. Correspondingly, we propose a novel noise synthesizing framework by explicitly learning its neighboring correlation on top of the signal dependency. With the proposed noise model, our framework greatly bridges the distribution gap between synthetic noise and real noise. We show that our generated “real” sRGB noisy images can be used for training supervised deep denoisers, thus to improve their real denoising results with a large margin, comparing to the popular classic denoisers or the deep denoisers that are trained on other sRGB noise generators. The code will be available at https://github.com/xuan611/sRGB-Real-Noise-Synthesizing.

        ----

        ## [163] Masked Image Training for Generalizable Deep Image Denoising

        **Authors**: *Haoyu Chen, Jinjin Gu, Yihao Liu, Salma Abdel Magid, Chao Dong, Qiong Wang, Hanspeter Pfister, Lei Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00169](https://doi.org/10.1109/CVPR52729.2023.00169)

        **Abstract**:

        When capturing and storing images, devices inevitably introduce noise. Reducing this noise is a critical task called image denoising. Deep learning has become the de facto method for image denoising, especially with the emergence of Transformer-based models that have achieved notable state-of-the-art results on various image tasks. However, deep learning-based methods often suffer from a lack of generalization ability. For example, deep models trained on Gaussian noise may perform poorly when tested on other noise distributions. To address this issue, we present a novel approach to enhance the generalization performance of denoising networks, known as masked training. Our method involves masking random pixels of the input image and reconstructing the missing information during training. We also mask out the features in the self-attention layers to avoid the impact of training-testing inconsistency. Our approach exhibits better generalization ability than other deep learning models and is directly applicable to real-world scenarios. Additionally, our interpretability analysis demonstrates the superiority of our method.

        ----

        ## [164] DR2: Diffusion-Based Robust Degradation Remover for Blind Face Restoration

        **Authors**: *Zhixin Wang, Ziying Zhang, Xiaoyun Zhang, Huangjie Zheng, Mingyuan Zhou, Ya Zhang, Yanfeng Wang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00170](https://doi.org/10.1109/CVPR52729.2023.00170)

        **Abstract**:

        Blind face restoration usually synthesizes degraded low-quality data with a pre-defined degradation model for training, while more complex cases could happen in the real world. This gap between the assumed and actual degradation hurts the restoration performance where artifacts are often observed in the output. However, it is expensive and infeasible to include every type of degradation to cover real-world cases in the training data. To tackle this robustness issue, we propose Diffusion-based Robust Degradation Remover (DR2) to first transform the degraded image to a coarse but degradation-invariant prediction, then employ an enhancement module to restore the coarse prediction to a high-quality image. By leveraging a well-performing denoising diffusion probabilistic model, our DR2 diffuses input images to a noisy status where various types of degradation give way to Gaussian noise, and then captures semantic information through iterative denoising steps. As a result, DR2 is robust against common degradation (e.g. blur, resize, noise and compression) and compatible with different designs of enhancement modules. Experiments in various settings show that our framework outperforms state-of-the-art methods on heavily degraded synthetic and real-world datasets.

        ----

        ## [165] Learning Distortion Invariant Representation for Image Restoration from a Causality Perspective

        **Authors**: *Xin Li, Bingchen Li, Xin Jin, Cuiling Lan, Zhibo Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00171](https://doi.org/10.1109/CVPR52729.2023.00171)

        **Abstract**:

        In recent years, we have witnessed the great advancement of Deep neural networks (DNNs) in image restoration. However, a critical limitation is that they cannot generalize well to real-world degradations with different degrees or types. In this paper, we are the first to propose a novel training strategy for image restoration from the causality perspective, to improve the generalization ability of DNNs for unknown degradations. Our method, termed Distortion Invariant representation Learning (DIL), treats each distortion type and degree as one specific confounder, and learns the distortion-invariant representation by eliminating the harmful confounding effect of each degradation. We derive our DIL with the back-door criterion in causality by modeling the interventions of different distortions from the optimization perspective. Particularly, we introduce counterfactual distortion augmentation to simulate the virtual distortion types and degrees as the confounders. Then, we instantiate the intervention of each distortion with a virtual model updating based on corresponding distorted images, and eliminate them from the meta-learning perspective. Extensive experiments demonstrate the generalization capability of our DIL on unseen distortion types and degrees. Our code will be available at https://github.com/lixinustc/Causal-IR-DIL.

        ----

        ## [166] Perception-Oriented Single Image Super-Resolution using Optimal Objective Estimation

        **Authors**: *Seung Ho Park, Young-Su Moon, Nam Ik Cho*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00172](https://doi.org/10.1109/CVPR52729.2023.00172)

        **Abstract**:

        Single-image super-resolution (SISR) networks trained with perceptual and adversarial losses provide high-contrast outputs compared to those of networks trained with distortion-oriented losses, such as 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$L1$</tex>
 or 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$L2$</tex>
. However, it has been shown that using a single perceptual loss is insufficient for accurately restoring locally varying diverse shapes in images, often generating undesirable artifacts or unnatural details. For this reason, combinations of various losses, such as perceptual, adversarial, and distortion losses, have been attempted, yet it remains challenging to find optimal combinations. Hence, in this paper, we propose a new SISR framework that applies optimal objectives for each region to generate plausible results in overall areas of high-resolution outputs. Specifically, the framework comprises two models: a predictive model that infers an optimal objective map for a given low-resolution (LR) input and a generative model that applies a target objective map to produce the corresponding SR output. The generative model is trained over our proposed objective trajectory representing a set of essential objectives, which enables the single network to learn various SR results corresponding to combined losses on the trajectory. The predictive model is trained using pairs of LR images and corresponding optimal objective maps searched from the objective trajectory. Experimental results on five benchmarks show that the proposed method outperforms state-of-the-art perception-driven SR methods in LPIPS, DISTS, PSNR, and SSIM metrics. The visual results also demonstrate the superiority of our method in perception-oriented reconstruction. The code is available at https://github.com/seungho-snu/SROOE.

        ----

        ## [167] Catch Missing Details: Image Reconstruction with Frequency Augmented Variational Autoencoder

        **Authors**: *Xinmiao Lin, Yikang Li, Jenhao Hsiao, Chiuman Ho, Yu Kong*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00173](https://doi.org/10.1109/CVPR52729.2023.00173)

        **Abstract**:

        The popular VQ-VAE models reconstruct images through learning a discrete codebook but suffer from a significant issue in the rapid quality degradation of image reconstruction as the compression rate rises. One major reason is that a higher compression rate induces more loss of visual signals on the higher frequency spectrum which reflect the details on pixel space. In this paper, a Frequency Complement Module (FCM) architecture is proposed to capture the missing frequency information for enhancing reconstruction quality. The FCM can be easily incorporated into the VQ-VAE structure, and we refer to the new model as Frequancy Augmented VAE (FAVAE). In addition, a Dynamic Spectrum Loss (DSL) is introduced to guide the FCMs to balance between various frequencies dynamically for optimal reconstruction. FA-VAE is further extended to the text-to-image synthesis task, and a Crossattention Autoregressive Transformer (CAT) is proposed to obtain more precise semantic attributes in texts. Extensive reconstruction experiments with different compression rates are conducted on several benchmark datasets, and the results demonstrate that the proposed FA-VAE is able to restore more faithfully the details compared to SOTA methods. CAT also shows improved generation quality with better image-text semantic alignment.

        ----

        ## [168] MD-VQA: Multi-Dimensional Quality Assessment for UGC Live Videos

        **Authors**: *Zicheng Zhang, Wei Wu, Wei Sun, Danyang Tu, Wei Lu, Xiongkuo Min, Ying Chen, Guangtao Zhai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00174](https://doi.org/10.1109/CVPR52729.2023.00174)

        **Abstract**:

        User-generated content (UGC) live videos are often bothered by various distortions during capture procedures and thus exhibit diverse visual qualities. Such source videos are further compressed and transcoded by media server providers before being distributed to end-users. Because of the flourishing of UGC live videos, effective video quality assessment (VQA) tools are needed to monitor and perceptually optimize live streaming videos in the distributing process. In this paper, we address UGC Live VQA problems by constructing a first-of-a-kind subjective UGC Live VQA database and developing an effective evaluation tool. Concretely, 418 source UGC videos are collected in real live streaming scenarios and 3,762 compressed ones at different bit rates are generated for the subsequent subjective VQA experiments. Based on the built database, we develop a Multi-12imensional VQA (MD-VQA) evaluator to measure the visual quality of UGC live videos from semantic, distortion, and motion aspects respectively. Extensive experimental results show that MD-VQA achieves state-of-the-art performance on both our UGC Live VQA database and existing compressed UGC VQA databases.

        ----

        ## [169] CABM: Content-Aware Bit Mapping for Single Image Super-Resolution Network with Large Input

        **Authors**: *Senmao Tian, Ming Lu, Jiaming Liu, Yandong Guo, Yurong Chen, Shunli Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00175](https://doi.org/10.1109/CVPR52729.2023.00175)

        **Abstract**:

        With the development of high-definition display devices, the practical scenario of Super-Resolution (SR) usually needs to super-resolve large input like 2K to higher resolution (4K/8K). To reduce the computational and memory cost, current methods first split the large input into local patches and then merge the SR patches into the output. These methods adaptively allocate a subnet for each patch. Quantization is a very important technique for network acceleration and has been used to design the subnets. Current methods train an MLP bit selector to determine the propoer bit for each layer. However, they uniformly sample subnets for training, making simple subnets overfitted and complicated subnets underfitted. Therefore, the trained bit selector fails to determine the optimal bit. Apart from this, the introduced bit selector brings additional cost to each layer of the 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$SR$</tex>
 network. In this paper, we propose a novel method named Content-Aware Bit Mapping (CABM), which can remove the bit selector without any performance loss. CABM also learns a bit selector for each layer during training. After training, we analyze the relation between the edge information of an input patch and the bit of each layer. We observe that the edge information can be an effective metric for the selected bit. Therefore, we design a strategy to build an Edge-to-Bit lookup table that maps the edge score of a patch to the bit of each layer during inference. The bit configuration of SR network can be determined by the lookup tables of all layers. Our strategy can find better bit configuration, resulting in more efficient mixed precision networks. We conduct detailed experiments to demonstrate the generalization ability of our method. The code will be released.

        ----

        ## [170] Initialization Noise in Image Gradients and Saliency Maps

        **Authors**: *Ann-Christin Woerl, Jan Disselhoff, Michael Wand*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00176](https://doi.org/10.1109/CVPR52729.2023.00176)

        **Abstract**:

        In this paper, we examine gradients of logits of image classification CNNs by input pixel values. We observe that these fluctuate considerably with training randomness, such as the random initialization of the networks. We extend our study to gradients of intermediate layers, obtained via GradCAM, as well as popular network saliency estimators such as DeepLIFT, SHAP, LIME, Integrated Gradients, and SmoothGrad. While empirical noise levels vary, qualitatively different attributions to image features are still possible with all of these, which comes with implications for interpreting such attributions, in particular when seeking data-driven explanations of the phenomenon generating the data. Finally, we demonstrate that the observed artefacts can be removed by marginalization over the initialization distribution by simple stochastic integration.

        ----

        ## [171] Local Implicit Normalizing Flow for Arbitrary-Scale Image Super-Resolution

        **Authors**: *Jie-En Yao, Li-Yuan Tsao, Yi-Chen Lo, Roy Tseng, Chia-Che Chang, Chun-Yi Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00177](https://doi.org/10.1109/CVPR52729.2023.00177)

        **Abstract**:

        Flow-based methods have demonstrated promising results in addressing the ill-posed nature of super-resolution (SR) by learning the distribution of high-resolution (HR) images with the normalizing flow. However, these methods can only perform a predefined fixed-scale SR, limiting their potential in real-world applications. Meanwhile, arbitrary-scale SR has gained more attention and achieved great progress. Nonetheless, previous arbitrary-scale SR methods ignore the ill-posed problem and train the model with per-pixel L1 loss, leading to blurry SR outputs. In this work, we propose “Local Implicit Normalizing Flow” (LINF) as a unified solution to the above problems. LINF models the distribution of texture details under different scaling factors with normalizing flow. Thus, LINF can generate photo-realistic HR images with rich texture details in arbitrary scale factors. We evaluate LINF with extensive experiments and show that LINF achieves the state-of-the-art perceptual quality compared with prior arbitrary-scale SR methods.

        ----

        ## [172] Deep Arbitrary-Scale Image Super-Resolution via Scale-Equivariance Pursuit

        **Authors**: *Xiaohang Wang, Xuanhong Chen, Bingbing Ni, Hang Wang, Zhengyan Tong, Yutian Liu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00178](https://doi.org/10.1109/CVPR52729.2023.00178)

        **Abstract**:

        The ability of scale-equivariance processing blocks plays a central role in arbitrary-scale image super-resolution tasks. Inspired by this crucial observation, this work proposes two novel scale-equivariant modules within a transformer-style framework to enhance arbitrary-scale image super-resolution (ASISR) performance, especially in high upsampling rate image extrapolation. In the feature extraction phase, we design a plug-in module called Adaptive Feature Extractor, which injects explicit scale information in frequency-expanded encoding, thus achieving scale-adaption in representation learning. In the upsampling phase, a learnable Neural Kriging upsampling operator is introduced, which simultaneously encodes both relative distance (i.e., scale-aware) information as well as feature similarity (i.e., with priori learned from training data) in a bilateral manner, providing scale-encoded spatial feature fusion. The above operators are easily plugged into multiple stages of a SR network, and a recent emerging pretraining strategy is also adopted to impulse the model's performance further. Extensive experimental results have demonstrated the outstanding scale-equivariance capability offered by the proposed operators and our learning framework, with much better results than previous SOTA methods at arbitrary scales for SR. Our code is available at https://github.com/neura1chen/EQSR

        ----

        ## [173] CiaoSR: Continuous Implicit Attention-in-Attention Network for Arbitrary-Scale Image Super-Resolution

        **Authors**: *Jiezhang Cao, Qin Wang, Yongqin Xian, Yawei Li, Bingbing Ni, Zhiming Pi, Kai Zhang, Yulun Zhang, Radu Timofte, Luc Van Gool*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00179](https://doi.org/10.1109/CVPR52729.2023.00179)

        **Abstract**:

        Learning continuous image representations is recently gaining popularity for image super-resolution (SR) because of its ability to reconstruct high-resolution images with arbitrary scales from low-resolution inputs. Existing methods mostly ensemble nearby features to predict the new pixel at any queried coordinate in the SR image. Such a local ensemble suffers from some limitations: i) it has no learnable parameters and it neglects the similarity of the visual features; ii) it has a limited receptive field and cannot ensemble relevant features in a large field which are important in an image. To address these issues, this paper proposes a continuous implicit attention-in-attention network, called CiaoSR. We explicitly design an implicit attention network to learn the ensemble weights for the nearby local features. Furthermore, we embed a scale-aware attention in this implicit attention network to exploit additional non-local information. Extensive experiments on benchmark datasets demonstrate CiaoSR significantly outperforms the existing single image SR methods with the same backbone. In addition, CiaoSR also achieves the state-of-the-art performance on the arbitrary-scale SR task. The effectiveness of the method is also demonstrated on the real-world SR setting. More importantly, CiaoSR can be flexibly integrated into any backbone to improve the SR performance.

        ----

        ## [174] Multiplicative Fourier Level of Detail

        **Authors**: *Yishun Dou, Zhong Zheng, Qiaoqiao Jin, Bingbing Ni*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00180](https://doi.org/10.1109/CVPR52729.2023.00180)

        **Abstract**:

        We develop a simple yet surprisingly effective implicit representing scheme called Multiplicative Fourier Level of Detail (MFLOD) motivated by the recent success of multiplicative filter network. Built on multi-resolution feature grid/volume (e.g., the sparse voxel octree), each level's feature is first modulated by a sinusoidal function and then element-wisely multiplied by a linear transformation of previous layer's representation in a layer-to-layer recursive manner, yielding the scale-aggregated encodings for a subsequent simple linear forward to get final output. In contrast to previous hybrid representations relying on interleaved multilevel fusion and nonlinear activation-based decoding, MFLOD could be elegantly characterized as a linear combination of sine basis functions with varying amplitude, frequency, and phase upon the learned multilevel features, thus offering great feasibility in Fourier analysis. Comprehensive experimental results on implicit neural representation learning tasks including image fitting, 3D shape representation, and neural radiance fields well demonstrate the superior quality and generalizability achieved by the proposed MFLOD scheme.

        ----

        ## [175] Document Image Shadow Removal Guided by Color-Aware Background

        **Authors**: *Ling Zhang, Yinghao He, Qing Zhang, Zheng Liu, Xiaolong Zhang, Chunxia Xiao*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00181](https://doi.org/10.1109/CVPR52729.2023.00181)

        **Abstract**:

        Existing works on document image shadow removal mostly depend on learning and leveraging a constant background (the color of the paper) from the image. However, the constant background is less representative and frequently ignores other background colors, such as the printed colors, resulting in distorted results. In this paper, we present a color-aware background extraction network (CBENet) for extracting a spatially varying background image that accurately depicts the background colors of the document. Furthermore, we propose a background-guided document images shadow removal network (BGShadowNet) using the predicted spatially varying background as auxiliary information, which consists of two stages. At Stage I, a background-constrained decoder is designed to promote a coarse result. Then, the coarse result is refined with a background-based attention module (BAModule) to maintain a consistent appearance and a detail improvement module (DEModule) to enhance the texture details at Stage II. Experiments on two benchmark datasets qualitatively and quantitatively validate the superiority of the proposed approach over state-of-the-arts.

        ----

        ## [176] StyleRes: Transforming the Residuals for Real Image Editing with StyleGAN

        **Authors**: *Hamza Pehlivan, Yusuf Dalva, Aysegul Dundar*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00182](https://doi.org/10.1109/CVPR52729.2023.00182)

        **Abstract**:

        We present a novel image inversion framework and a training pipeline to achieve high-fidelity image inversion with high-quality attribute editing. Inverting real images into StyleGAN's latent space is an extensively studied problem, yet the trade-off between the image reconstruction fidelity and image editing quality remains an open challenge. The low-rate latent spaces are limited in their expressiveness power for high-fidelity reconstruction. On the other hand, high-rate latent spaces result in degradation in editing quality. In this work, to achieve high-fidelity inversion, we learn residual features in higher latent codes that lower latent codes were not able to encode. This enables preserving image details in reconstruction. To achieve high-quality editing, we learn how to transform the residual features for adapting to manipulations in latent codes. We train the framework to extract residual features and transform them via a novel architecture pipeline and cycle consistency losses. We run extensive experiments and compare our method with state-of-the-art inversion methods. Qualitative metrics and visual comparisons show significant improvements. Code: https://github.com/hamzapehlivanIStyleRes.

        ----

        ## [177] TopNet: Transformer-Based Object Placement Network for Image Compositing

        **Authors**: *Sijie Zhu, Zhe Lin, Scott Cohen, Jason Kuen, Zhifei Zhang, Chen Chen*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00183](https://doi.org/10.1109/CVPR52729.2023.00183)

        **Abstract**:

        We investigate the problem of automatically placing an object into a background image for image compositing. Given a background image and a segmented object, the goal is to train a model to predict plausible placements (location and scale) of the object for compositing. The quality of the composite image highly depends on the predicted location/scale. Existing works either generate candidate bounding boxes or apply sliding-window search using global representations from background and object images, which fail to model local information in background images. However, local clues in background images are important to determine the compatibility of placing the objects with certain locations/scales. In this paper, we propose to learn the correlation between object features and all local background features with a transformer module so that detailed information can be provided on all possible location/scale configurations. A sparse contrastive loss is further proposed to train our model with sparse supervision. Our new formulation generates a 3D heatmap indicating the plausibility of all location/scale combinations in one network forward pass, which is > 10 x faster than the previous sliding-window method. It also supports interactive search when users provide a pre-defined location or scale. The proposed method can be trained with explicit annotation or in a self-supervised manner using an off-the-shelf inpainting model, and it outperforms state-of-the-art methods significantly. User study shows that the trained model generalizes well to real-world images with diverse challenging scenes and object categories.

        ----

        ## [178] VecFontSDF: Learning to Reconstruct and Synthesize High-Quality Vector Fonts via Signed Distance Functions

        **Authors**: *Zeqing Xia, Bojun Xiong, Zhouhui Lian*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00184](https://doi.org/10.1109/CVPR52729.2023.00184)

        **Abstract**:

        Font design is of vital importance in the digital content design and modern printing industry. Developing algorithms capable of automatically synthesizing vector fonts can significantly facilitate the font design process. However, existing methods mainly concentrate on raster image generation, and only a few approaches can directly synthesize vector fonts. This paper proposes an end-to-end trainable method, VecFontSDF, to reconstruct and synthesize high-quality vector fonts using signed distance functions (SDFs). Specifically, based on the proposed SDF-based implicit shape representation, VecFontSDF learns to model each glyph as shape primitives enclosed by several parabolic curves, which can be precisely converted to quadratic Bézier curves that are widely used in vector font products. In this manner, most image generation methods can be easily extended to synthesize vector fonts. Qualitative and quantitative experiments conducted on a publicly-available dataset demonstrate that our method obtains high-quality results on several tasks, including vector font reconstruction, interpolation, and few-shot vector font synthesis, markedly outperforming the state of the art.

        ----

        ## [179] CF-Font: Content Fusion for Few-Shot Font Generation

        **Authors**: *Chi Wang, Min Zhou, Tiezheng Ge, Yuning Jiang, Hujun Bao, Weiwei Xu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00185](https://doi.org/10.1109/CVPR52729.2023.00185)

        **Abstract**:

        Content and style disentanglement is an effective way to achieve few-shot font generation. It allows to transfer the style of the font image in a source domain to the style defined with a few reference images in a target domain. However, the content feature extracted using a representative font might not be optimal. In light of this, we propose a content fusion module (CFM) to project the content feature into a linear space defined by the content features of basis fonts, which can take the variation of content features caused by different fonts into consideration. Our method also allows to optimize the style representation vector of reference images through a lightweight iterative style-vector refinement (ISR) strategy. Moreover, we treat the 1D projection of a character image as a probability distribution and leverage the distance between two distributions as the reconstruction loss (namely projected character loss, PCL). Compared to L2 or L1 reconstruction loss, the distribution distance pays more attention to the global shape of characters. We have evaluated our method on a dataset of 300 fonts with 6.5k characters each. Experimental results verify that our method outperforms existing state-of-the-art few-shot font generation methods by a large margin. The source code can be found at https://github.com/wangchi95/CF-Font.

        ----

        ## [180] SIEDOB: Semantic Image Editing by Disentangling Object and Background

        **Authors**: *Wuyang Luo, Su Yang, Xinjian Zhang, Weishan Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00186](https://doi.org/10.1109/CVPR52729.2023.00186)

        **Abstract**:

        Semantic image editing provides users with a flexible tool to modify a given image guided by a corresponding segmentation map. In this task, the features of the foreground objects and the backgrounds are quite different. However, all previous methods handle backgrounds and objects as a whole using a monolithic model. Consequently, they remain limited in processing content-rich images and suffer from generating unrealistic objects and texture-inconsistent backgrounds. To address this issue, we propose a novel paradigm, Semantic Image Editing by Disentangling Object and Background (SIEDOB), the core idea of which is to explicitly leverages several heterogeneous subnetworks for objects and backgrounds. First, SIEDOB disassembles the edited input into background regions and instance-level objects. Then, we feed them into the dedicated generators. Finally, all synthesized parts are embedded in their original locations and utilize a fusion network to obtain a harmonized result. Moreover, to produce high-quality edited images, we propose some innovative designs, including Semantic-Aware Self-Propagation Module, Boundary-Anchored Patch Discriminator, and Style-Diversity Object Generator, and integrate them into SIEDOB. We conduct extensive experiments on Cityscapes and ADE20K-Room datasets and exhibit that our method remarkably outperforms the baselines, especially in synthesizing realistic and diverse objects and texture-consistent backgrounds. Code is available at https://github.com/WuyangLuo/SIEDOB.

        ----

        ## [181] MaskSketch: Unpaired Structure-guided Masked Image Generation

        **Authors**: *Dina Bashkirova, José Lezama, Kihyuk Sohn, Kate Saenko, Irfan Essa*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00187](https://doi.org/10.1109/CVPR52729.2023.00187)

        **Abstract**:

        Recent conditional image generation methods produce images of remarkable diversity, fidelity and realism. However, the majority of these methods allow conditioning only on labels or text prompts, which limits their level of control over the generation result. In this paper, we introduce MaskSketch, an image generation method that allows spatial conditioning of the generation result using a guiding sketch as an extra conditioning signal during sampling. MaskSketch utilizes a pretrained masked generative transformer, requiring no model training or paired supervision, and works with input sketches of different levels of abstraction. We show that intermediate self-attention maps of a masked generative transformer encode important structural information of the input image, such as scene layout and object shape, and we propose a novel sampling method based on this observation to enable structure-guided generation. Our results show that MaskSketch achieves high image realism and fidelity to the guiding structure. Evaluated on standard benchmark datasets, MaskSketch outperforms state-of-the-art methods for sketch-to-image translation, as well as unpaired image-to-image translation approaches. The code can be found on our project website: https://masksketch.github.io/

        ----

        ## [182] Text2Scene: Text-driven Indoor Scene Stylization with Part-Aware Details

        **Authors**: *Inwoo Hwang, Hyeonwoo Kim, Young Min Kim*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00188](https://doi.org/10.1109/CVPR52729.2023.00188)

        **Abstract**:

        We propose Text2Scene, a method to automatically create realistic textures for virtual scenes composed of multiple objects. Guided by a reference image and text descriptions, our pipeline adds detailed texture on labeled 3D geometries in the room such that the generated colors respect the hierarchical structure or semantic parts that are often composed of similar materials. Instead of applying flat stylization on the entire scene at a single step, we obtain weak semantic cues from geometric segmentation, which are further clarified by assigning initial colors to segmented parts. Then we add texture details for individual objects such that their projections on image space exhibit feature embedding aligned with the embedding of the input. The decomposition makes the entire pipeline tractable to a moderate amount of computation resources and memory. As our framework utilizes the existing resources of image and text embedding, it does not require dedicated datasets with high-quality textures designed by skillful artists. To the best of our knowledge, it is the first practical and scalable approach that can create detailed and realistic textures of the desired style that maintain structural context for scenes with multiple objects.

        ----

        ## [183] Uncovering the Disentanglement Capability in Text-to-Image Diffusion Models

        **Authors**: *Qiucheng Wu, Yujian Liu, Handong Zhao, Ajinkya Kale, Trung Bui, Tong Yu, Zhe Lin, Yang Zhang, Shiyu Chang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00189](https://doi.org/10.1109/CVPR52729.2023.00189)

        **Abstract**:

        Generative models have been widely studied in computer vision. Recently, diffusion models have drawn substantial attention due to the high quality of their generated images. A key desired property of image generative models is the ability to disentangle different attributes, which should enable modification towards a style without changing the semantic content, and the modification parameters should generalize to different images. Previous studies have found that generative adversarial networks (GANs) are inherently endowed with such disentanglement capability, so they can perform disentangled image editing without re-training or fine-tuning the network. In this work, we explore whether diffusion models are also inherently equipped with such a capability. Our finding is that for stable diffusion models, by partially changing the input text embedding from a neutral description (e.g., “a photo of person”) to one with style (e.g., “a photo of person with smile”) while fixing all the Gaussian random noises introduced during the denoising process, the generated images can be modified towards the target style without changing the semantic content. Based on this finding, we further propose a simple, light-weight image editing algorithm where the mixing weights of the two text embeddings are optimized for style matching and content preservation. This entire process only involves optimizing over around 50 parameters and does not fine-tune the diffusion model itself. Experiments show that the proposed method can modify a wide range of attributes, with the performance outperforming diffusion-model-based image-editing algorithms that require fine-tuning. The optimized weights generalize well to different images. Our code is publicly available at https://github.com/UCSB-NLP-Chang/DiffusionDisentanglement.

        ----

        ## [184] VectorFusion: Text-to-SVG by Abstracting Pixel-Based Diffusion Models

        **Authors**: *Ajay Jain, Amber Xie, Pieter Abbeel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00190](https://doi.org/10.1109/CVPR52729.2023.00190)

        **Abstract**:

        Diffusion models have shown impressive results in text-to-image synthesis. Using massive datasets of captioned images, diffusion models learn to generate raster images of highly diverse objects and scenes. However, designers frequently use vector representations of images like Scalable Vector Graphics (SVGs) for digital icons or art. Vector graphics can be scaled to any size, and are compact. We show that a text-conditioned diffusion model trained on pixel representations of images can be used to generate SVG-exportable vector graphics. We do so without access to large datasets of captioned SVGs. By optimizing a differentiable vector graphics rasterizer, our method, VectorFusion, distills abstract semantic knowledge out of a pretrained diffusion model. Inspired by recent text-to-3D work, we learn an SVG consistent with a caption using Score Distillation Sampling. To accelerate generation and improve fidelity, VectorFusion also initializes from an image sample. Experiments show greater quality than prior work, and demonstrate a range of styles including pixel art and sketches.

        ----

        ## [185] Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation

        **Authors**: *Narek Tumanyan, Michal Geyer, Shai Bagon, Tali Dekel*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00191](https://doi.org/10.1109/CVPR52729.2023.00191)

        **Abstract**:

        Large-scale text-to-image generative models have been a revolutionary breakthrough in the evolution of generative AI, synthesizing diverse images with highly complex visual concepts. However, a pivotal challenge in leveraging such models for real-world content creation is providing users with control over the generated content. In this paper, we present a new framework that takes text-to- image synthesis to the realm of image-to-image translation - given a guidance image and a target text prompt as input, our method harnesses the power of a pre-trained text-to-image diffusion model to generate a new image that complies with the target text, while preserving the semantic layout of the guidance image. Specifically, we observe and empirically demonstrate that fine-grained control over the generated structure can be achieved by manipulating spatial features and their self-attention inside the model. This results in a simple and effective approach, where features extracted from the guidance image are directly injected into the generation process of the translated image, requiring no training or fine-tuning. We demonstrate high-quality results on versatile text-guided image translation tasks, including translating sketches, rough drawings and animations into realistic images, changing the class and appearance of objects in a given image, and modifying global qualities such as lighting and color.

        ----

        ## [186] Multi-Concept Customization of Text-to-Image Diffusion

        **Authors**: *Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, Jun-Yan Zhu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00192](https://doi.org/10.1109/CVPR52729.2023.00192)

        **Abstract**:

        While generative models produce high-quality images of concepts learned from a large-scale database, a user often wishes to synthesize instantiations of their own concepts (for example, their family, pets, or items). Can we teach a model to quickly acquire a new concept, given a few examples? Furthermore, can we compose multiple new concepts together? We propose Custom Diffusion, an efficient method for augmenting existing text-to-image models. We find that only optimizing a few parameters in the text-to-image conditioning mechanism is sufficiently powerful to represent new concepts while enabling fast tuning (~ 6 minutes). Additionally, we can jointly train for multiple concepts or combine multiple fine-tuned models into one via closed-form constrained optimization. Our fine-tuned model generates variations of multiple new concepts and seamlessly composes them with existing concepts in novel settings. Our method outperforms or performs on par with several baselines and concurrent works in both qualitative and quantitative evaluations, while being memory and computationally efficient.

        ----

        ## [187] Unifying Layout Generation with a Decoupled Diffusion Model

        **Authors**: *Mude Hui, Zhizheng Zhang, Xiaoyi Zhang, Wenxuan Xie, Yuwang Wang, Yan Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00193](https://doi.org/10.1109/CVPR52729.2023.00193)

        **Abstract**:

        Layout generation aims to synthesize realistic graphic scenes consisting of elements with different attributes in-cluding category, size, position, and between-element relation. It is a crucial task for reducing the burden on heavyduty graphic design works for formatted scenes, e.g., publications, documents, and user interfaces (UIs). Diverse application scenarios impose a big challenge in unifying various layout generation subtasks, including conditional and unconditional generation. In this paper, we propose a Layout Diffusion Generative Model (LDGM) to achieve such unification with a single decoupled diffusion model. LDGM views a layout of arbitrary missing or coarse element attributes as an intermediate diffusion status from a completed layout. Since different attributes have their individual semantics and characteristics, we propose to decouple the diffusion processes for them to improve the diversity of training samples and learn the reverse process jointly to exploit global-scope contexts for facilitating generation. As a result, our LDGM can generate layouts either from scratch or conditional on arbitrary available attributes. Extensive qualitative and quantitative experiments demonstrate our proposed LDGM outperforms existing layout generation models in both functionality and performance.

        ----

        ## [188] BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models

        **Authors**: *Bo Li, Kaitao Xue, Bin Liu, Yu-Kun Lai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00194](https://doi.org/10.1109/CVPR52729.2023.00194)

        **Abstract**:

        Image-to-image translation is an important and challenging problem in computer vision and image processing. Diffusion models (DM) have shown great potentials for high-quality image synthesis, and have gained competitive performance on the task of image-to-image translation. However, most of the existing diffusion models treat image-to-image translation as conditional generation processes, and suffer heavily from the gap between distinct domains. In this paper, a novel image-to-image translation method based on the Brownian Bridge Diffusion Model (BBDM) is proposed, which models image-to-image translation as a stochastic Brownian Bridge process, and learns the translation between two domains directly through the bidirectional diffusion process rather than a conditional generation process. To the best of our knowledge, it is the first work that proposes Brownian Bridge diffusion process for image-to-image translation. Experimental results on various benchmarks demonstrate that the proposed BBDM model achieves competitive performance through both visual inspection and measurable metrics.

        ----

        ## [189] Towards Practical Plug-and-Play Diffusion Models

        **Authors**: *Hyojun Go, Yunsung Lee, Jin Young Kim, Seunghyun Lee, Myeongho Jeong, Hyun Seung Lee, Seungtaek Choi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00195](https://doi.org/10.1109/CVPR52729.2023.00195)

        **Abstract**:

        Diffusion-based generative models have achieved remarkable success in image generation. Their guidance formulation allows an external model to plug-and-play control the generation process for various tasks without finetuning the diffusion model. However, the direct use of publicly available off-the-shelf models for guidance fails due to their poor performance on noisy inputs. For that, the existing practice is to fine-tune the guidance models with labeled data corrupted with noises. In this paper, we argue that this practice has limitations in two aspects: (1) performing on inputs with extremely various noises is too hard for a single guidance model; (2) collecting labeled datasets hinders scaling up for various tasks. To tackle the limitations, we propose a novel strategy that leverages multiple experts where each expert is specialized in a particular noise range and guides the reverse process of the diffusion at its corresponding timesteps. However, as it is infeasible to manage multiple networks and utilize labeled data, we present a practical guidance framework termed Practical Plug-And-Play (PPAP), which leverages parameter-efficient fine-tuning and data-free knowledge transfer. We exhaustively conduct ImageNet class conditional generation experiments to show that our method can successfully guide diffusion with small trainable parameters and no labeled data. Finally, we show that image classifiers, depth estimators, and semantic segmentation models can guide publicly available GLIDE through our framework in a plug-and-play manner. Our code is available at https://github.com/riiid/PPAP.

        ----

        ## [190] Post-Training Quantization on Diffusion Models

        **Authors**: *Yuzhang Shang, Zhihang Yuan, Bin Xie, Bingzhe Wu, Yan Yan*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00196](https://doi.org/10.1109/CVPR52729.2023.00196)

        **Abstract**:

        Denoising diffusion (score-based) generative models have recently achieved significant accomplishments in generating realistic and diverse data. Unfortunately, the generation process of current denoising diffusion models is notoriously slow due to the lengthy iterative noise estimations, which rely on cumbersome neural networks. It prevents the diffusion models from being widely deployed, especially on edge devices. Previous works accelerate the generation process of diffusion model (DM) via finding shorter yet effective sampling trajectories. However, they overlook the cost of noise estimation with a heavy network in every iteration. In this work, we accelerate generation from the perspective of compressing the noise estimation network. Due to the difficulty of retraining DMs, we exclude mainstream training-aware compression paradigms and introduce post-training quantization (PTQ) into DM acceleration. However, the output distributions of noise estimation networks change with time-step, making previous PTQ methods fail in DMs since they are designed for single-time step scenarios. To devise a DM-specific PTQ method, we explore PTQ on DM in three aspects: quantized operations, calibration dataset, and calibration metric. We summarize and use several observations derived from all-inclusive investigations to formulate our method, which especially targets the unique multi-time-step structure of DMs. Experimentally, our method can directly quantize full-precision DMs into 8-bit models while maintaining or even improving their performance in a training-free manner. Importantly, our method can serve as a plug-and-play module on other fast-sampling methods, e.g., DDIM [24]. The code is available at https://https://github.com/42Shawn/PTQ4DM.

        ----

        ## [191] DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation

        **Authors**: *Shuai Shen, Wenliang Zhao, Zibin Meng, Wanhua Li, Zheng Zhu, Jie Zhou, Jiwen Lu*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00197](https://doi.org/10.1109/CVPR52729.2023.00197)

        **Abstract**:

        Talking head synthesis is a promising approach for the video production industry. Recently, a lot of effort has been devoted in this research area to improve the generation quality or enhance the model generalization. However, there are few works able to address both issues simultaneously, which is essential for practical applications. To this end, in this paper, we turn attention to the emerging powerful Latent Diffusion Models, and model the Talking head generation as an audio-driven temporally coherent denoising process (DiffTalk). More specifically, instead of employing audio signals as the single driving factor, we investigate the control mechanism of the talking face, and incorporate reference face images and landmarks as conditions for personality-aware generalized synthesis. In this way, the proposed DiffTalk is capable of producing high-quality talking head videos in synchronization with the source audio, and more importantly, it can be naturally generalized across different identities without further finetuning. Additionally, our DiffTalk can be gracefully tailored for higher-resolution synthesis with negligible extra computational cost. Extensive experiments show that the proposed DiffTalk efficiently synthesizes high-fidelity audio-driven talking head videos for generalized novel identities. For more video results, please refer to https://sstzal.github.io/DiffTalk/.

        ----

        ## [192] Mask-Guided Matting in the Wild

        **Authors**: *Kwanyong Park, Sanghyun Woo, Seoung Wug Oh, In So Kweon, Joon-Young Lee*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00198](https://doi.org/10.1109/CVPR52729.2023.00198)

        **Abstract**:

        Mask-guided matting has shown great practicality compared to traditional trimap-based methods. The mask-guided approach takes an easily-obtainable coarse mask as guidance and produces an accurate alpha matte. To extend the success toward practical usage, we tackle mask-guided matting in the wild, which covers a wide range of categories in their complex context robustly. To this end, we propose a simple yet effective learning framework based on two core insights: 1) learning a generalized matting model that can better understand the given mask guidance and 2) leveraging weak supervision datasets (e.g., instance segmentation dataset) to alleviate the limited diversity and scale of existing matting datasets. Extensive experimental results on multiple benchmarks, consisting of a newly proposed synthetic benchmark (Composition-Wild) and existing natural datasets, demonstrate the superiority of the proposed method. Moreover, we provide appealing results on new practical applications (e.g., panoptic matting and mask-guided video matting), showing the great generality and potential of our model.

        ----

        ## [193] Not All Image Regions Matter: Masked Vector Quantization for Autoregressive Image Generation

        **Authors**: *Mengqi Huang, Zhendong Mao, Quan Wang, Yongdong Zhang*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00199](https://doi.org/10.1109/CVPR52729.2023.00199)

        **Abstract**:

        Existing autoregressive models follow the two-stage generation paradigm that first learns a codebook in the latent space for image reconstruction and then completes the image generation autoregressively based on the learned codebook. However, existing codebook learning simply models all local region information of images without distinguishing their different perceptual importance, which brings redundancy in the learned codebook that not only limits the next stage's autoregressive model's ability to model important structure but also results in high training cost and slow generation speed. In this study, we borrow the idea of importance perception from classical image coding theory and propose a novel two-stage framework, which consists of Masked Quantization VAE (MQVAE) and Stackformer, to relieve the model from modeling redundancy. Specifically, MQ-VAE incorporates an adaptive mask module for masking redundant region features before quantization and an adaptive de-mask module for recovering the original grid image feature map to faithfully reconstruct the original images after quantization. Then, Stackformer learns to predict the combination of the next code and its position in the feature map. Comprehensive experiments on various image generation validate our effectiveness and efficiency. Code will be released at https://github.com/CrossmodalGroup/MaskedVectorQuantization.

        ----

        ## [194] Compression-Aware Video Super-Resolution

        **Authors**: *Yingwei Wang, Takashi Isobe, Xu Jia, Xin Tao, Huchuan Lu, Yu-Wing Tai*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00200](https://doi.org/10.1109/CVPR52729.2023.00200)

        **Abstract**:

        Videos stored on mobile devices or delivered on the Internet are usually in compressed format and are of various unknown compression parameters, but most video super-resolution (VSR) methods often assume ideal inputs resulting in large performance gap between experimental settings and real-world applications. In spite of a few pioneering works being proposed recently to super-resolve the compressed videos, they are not specially designed to deal with videos of various levels of compression. In this paper, we propose a novel and practical compression-aware video super-resolution model, which could adapt its video enhancement process to the estimated compression level. A compression encoder is designed to model compression levels of input frames, and a base VSR model is then conditioned on the implicitly computed representation by inserting compression-aware modules. In addition, we propose to further strengthen the VSR model by taking full advantage of meta data that is embedded naturally in compressed video streams in the procedure of information fusion. Extensive experiments are conducted to demonstrate the effectiveness and efficiency of the proposed method on compressed VSR benchmarks. The codes will be available at https://github.com/aprBlue/CAVSR

        ----

        ## [195] Neural Rate Estimator and Unsupervised Learning for Efficient Distributed Image Analytics in Split-DNN models

        **Authors**: *Nilesh A. Ahuja, Parual Datta, Bhavya Kanzariya, V. Srinivasa Somayazulu, Omesh Tickoo*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00201](https://doi.org/10.1109/CVPR52729.2023.00201)

        **Abstract**:

        Thanks to advances in computer vision and AI, there has been a large growth in the demand for cloud-based visual analytics in which images captured by a low-powered edge device are transmitted to the cloud for analytics. Use of conventional codecs (JPEG, MPEG, HEVC, etc.) for compressing such data introduces artifacts that can seriously degrade the performance of the downstream analytic tasks. Split-DNN computing has emerged as a paradigm to address such usages, in which a DNN is partitioned into a client-side portion and a server side portion. Low-complexity neural networks called ‘bottleneck units' are introduced at the split point to transform the intermediate layer features into a lower-dimensional representation better suited for compression and transmission. Optimizing the pipeline for both compression and task-performance requires high-quality estimates of the information-theoretic rate of the intermediate features. Most works on compression for image analytics use heuristic approaches to estimate the rate, leading to suboptimal performance. We propose a high-quality ‘neural rateestimator’ to address this gap. We interpret the lower-dimensional bottleneck output as a latent representation of the intermediate feature and cast the rate-distortion optimization problem as one of training an equivalent variational auto-encoder with an appropriate loss function. We show that this leads to improved rate-distortion outcomes. We further show that replacing supervised loss terms (such as cross-entropy loss) by distillation-based losses in a teacher-student framework allows for unsupervised training of bottleneck units without the need for explicit training labels. This makes our method very attractive for real world deployments where access to labeled training data is difficult or expensive. We demonstrate that our method outperforms several state-of-the-art methods by obtaining improved task accuracy at lower bi-trates on image classification and semantic segmentation tasks.

        ----

        ## [196] DNeRV: Modeling Inherent Dynamics via Difference Neural Representation for Videos

        **Authors**: *Qi Zhao, M. Salman Asif, Zhan Ma*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00202](https://doi.org/10.1109/CVPR52729.2023.00202)

        **Abstract**:

        Existing implicit neural representation (INR) methods do not fully exploit spatiotemporal redundancies in videos. Index-based INRs ignore the content-specific spatial features and hybrid INRs ignore the contextual dependency on adjacent frames, leading to poor modeling capability for scenes with large motion or dynamics. We analyze this limitation from the perspective of function fitting and reveal the importance of frame difference. To use explicit motion information, we propose Difference Neural Representation for Videos (DNeRV), which consists of two streams for content and frame difference. We also introduce a collaborative content unit for effective feature fusion. We test DNeRV for video compression, inpainting, and interpolation. D-NeRVachieves competitive results against the state-of-the-art neural compression approaches and outperforms existing implicit methods on downstream inpainting and interpolation for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$960\times 1920$</tex>
 videos.

        ----

        ## [197] Polynomial Implicit Neural Representations For Large Diverse Datasets

        **Authors**: *Rajhans Singh, Ankita Shukla, Pavan K. Turaga*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00203](https://doi.org/10.1109/CVPR52729.2023.00203)

        **Abstract**:

        Implicit neural representations (INR) have gained significant popularity for signal and image representation for many end-tasks, such as superresolution, 3D modeling, and more. Most INR architectures rely on sinusoidal positional encoding, which accounts for high-frequency information in data. However, the finite encoding size restricts the model's representational power. Higher representational power is needed to go from representing a single given image to representing large and diverse datasets. Our approach addresses this gap by representing an image with a polynomial function and eliminates the need for positional encodings. Therefore, to achieve a progressively higher degree of polynomial representation, we use element-wise multiplications between features and affine-transformed coordinate locations after every ReLU layer. The proposed method is evaluated qualitatively and quantitatively on large datasets like ImageNet. The proposed Poly-INR model performs comparably to state-of-the-art generative models without any convolution, normalization, or self-attention layers, and with far fewer trainable parameters. With much fewer training parameters and higher representative power, our approach paves the way for broader adoption of INR models for generative modeling tasks in complex domains. The code is available at https://github.com/Rajhans0/Poly_INR

        ----

        ## [198] Learning Decorrelated Representations Efficiently Using Fast Fourier Transform

        **Authors**: *Yutaro Shigeto, Masashi Shimbo, Yuya Yoshikawa, Akikazu Takeuchi*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00204](https://doi.org/10.1109/CVPR52729.2023.00204)

        **Abstract**:

        Barlow Twins and VICReg are self-supervised representation learning models that use regularizers to decorrelate features. Although these models are as effective as conventional representation learning models, their training can be computationally demanding if the dimension 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$d$</tex>
 of the projected embeddings is high. As the regularizers are defined in terms of individual elements of a cross-correlation or covariance matrix, computing the loss for 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$n$</tex>
 samples takes 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(nd^{2})$</tex>
 time. In this paper, we propose a relaxed decorre-lating regularizer that can be computed in 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$O(nd\log d)$</tex>
 time by Fast Fourier Transform. We also propose an inexpensive technique to mitigate undesirable local minima that develop with the relaxation. The proposed regularizer exhibits accuracy comparable to that of existing regularizers in down-stream tasks, whereas their training requires less memory and is faster for large 
<tex xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">$d$</tex>
. The source code is available. 
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
<sup xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">1</sup>
https://github.com/yutaro-s/scalable-decorrelation-ssl.git

        ----

        ## [199] SparseViT: Revisiting Activation Sparsity for Efficient High-Resolution Vision Transformer

        **Authors**: *Xuanyao Chen, Zhijian Liu, Haotian Tang, Li Yi, Hang Zhao, Song Han*

        **Conference**: *cvpr 2023*

        **URL**: [https://doi.org/10.1109/CVPR52729.2023.00205](https://doi.org/10.1109/CVPR52729.2023.00205)

        **Abstract**:

        High-resolution images enable neural networks to learn richer visual representations. However, this improved performance comes at the cost of growing computational complexity, hindering their usage in latency-sensitive applications. As not all pixels are equal, skipping computations for less-important regions offers a simple and effective measure to reduce the computation. This, however, is hard to be translated into actual speedup for CNNs since it breaks the regularity of the dense convolution workload. In this paper, we introduce SparseViT that revisits activation sparsity for recent window-based vision transformers (ViTs). As window attentions are naturally batched over blocks, actual speedup with window activation pruning becomes possible: i.e., ~50% latency reduction with 60% sparsity. Different layers should be assigned with different pruning ratios due to their diverse sensitivities and computational costs. We introduce sparsity-aware adaptation and apply the evolutionary search to efficiently find the optimal layerwise sparsity configuration within the vast search space. SparseViT achieves speedups of 1.5×, 1.4×, and 1.3× compared to its dense counterpart in monocular 3D object detection, 2D instance segmentation, and 2D semantic segmentation, respectively, with negligible to no loss of accuracy.

        ----

        

[Go to the next page](CVPR-2023-list02.md)

[Go to the catalog section](README.md)